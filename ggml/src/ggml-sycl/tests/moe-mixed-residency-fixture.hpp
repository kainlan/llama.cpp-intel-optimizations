// What the two Q1_0/NVFP4 mixed-residency MUL_MAT_ID tests share: test-moe-mixed-residency-q1-nvfp4.cpp, which runs
// the op for both types at three token counts, and test-moe-q1-nvfp4-dispatch-refusal.cpp, which runs one NVFP4 op on
// the same vehicle with its batched dispatch made to refuse once. It holds the model's shape and the experts moved to
// the host, the lines that say an op refused or ran another way, the capture of the backend's log, the [MOE-ROUTE]
// partition line parser with its own controls, and the planned model's lifecycle with the plan edit that moves the
// experts (test-moe-mixed-residency-q1-nvfp4.cpp describes the load). Each test includes it once and keeps its own
// cases, its main() and every skip path.
#pragma once

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"
#include "unified-cache.hpp"  // the load's candidate plan, edited and staged again before the weights load

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace moe_mixed_residency_fixture {

constexpr int      k_experts  = 16;
constexpr int      k_used     = 4;
constexpr int      k_in       = 2048;  // k: a whole number of Q1_0 (128) and of NVFP4 (64) blocks
constexpr int      k_out      = 512;   // m: output rows per expert
// The largest op's token count: the planner sizes the host recipe's workspace for a prompt op of up to n_ubatch rows.
constexpr uint32_t k_n_ubatch = 512;
constexpr uint32_t k_n_ctx    = 512;
constexpr double   k_nmse_tol = 5e-4;

// The three expert tensors share one shape. The gate ops run the first, the down ops the third.
constexpr const char * k_tensor_names[3] = { "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight",
                                             "blk.0.ffn_down_exps.weight" };
constexpr int          k_gate            = 0;
constexpr int          k_down            = 2;

// The experts the fixture moves to the host in every expert tensor. Seven of sixteen, so an op's device and host
// counts are not equal by construction (each test's one-token op picks one device and three host experts), and a
// route line that swapped the two counts fails.
constexpr int k_host_experts[] = { 1, 4, 6, 9, 10, 13, 15 };

// Lines that say the op refused a route, routed a side some other way, or failed after the throw that ends it.
constexpr const char * k_refusal_needles[] = {
    "[MOE-DECODE-REFUSAL]",
    "[MOE-PROMPT-REFUSAL]",
    "batched dispatch returned false",
    "skipping planner batched pointer-table dispatch",
    "falling back to host",
    "[CONTEXT-PLAN-BUG]",
    "recoverable runtime fallback failed",
};

inline void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// The environment cannot exercise what the test covers; each test's main() reports it as ctest's skip (77), never as
// a pass, on a line prefixed SKIP[<tag>] so each cause stays distinguishable in a log.
struct test_skip : std::runtime_error {
    test_skip(const char * tag, const std::string & why) : std::runtime_error(why), tag(tag) {}

    const char * tag;
};

// Every log chunk the backend emits, in order, so a test can read what appeared while an op ran. Also echoed to
// stderr, so the run's log carries the backend's own lines.
struct log_capture {
    std::mutex               mu;
    std::vector<std::string> chunks;

    size_t mark() {
        std::lock_guard<std::mutex> lock(mu);
        return chunks.size();
    }

    // The chunks from `from` on that contain every needle.
    std::vector<std::string> since(size_t from, std::initializer_list<const char *> needles) {
        std::lock_guard<std::mutex> lock(mu);
        std::vector<std::string>    found;
        for (size_t i = from; i < chunks.size(); ++i) {
            bool all = true;
            for (const char * needle : needles) {
                all = all && chunks[i].find(needle) != std::string::npos;
            }
            if (all) {
                found.push_back(chunks[i]);
            }
        }
        return found;
    }
};

inline log_capture g_log;

inline void log_callback(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    if (!text) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(g_log.mu);
        g_log.chunks.emplace_back(text);
    }
    std::fputs(text, stderr);
}

// The count that follows " <name>=" in a [MOE-ROUTE] line, or -1 when the field is missing or is not a plain count
// (the retained prompt route's own line writes its counts as a/b, and it is not the partition line an op expects).
inline long long route_field(const std::string & line, const char * name) {
    const std::string key = std::string(" ") + name + "=";
    const size_t      at  = line.find(key);
    if (at == std::string::npos) {
        return -1;
    }
    const char *    digits = line.c_str() + at + key.size();
    char *          end    = nullptr;
    const long long value  = std::strtoll(digits, &end, 10);
    if (end == digits || (*end != ' ' && *end != '\n' && *end != '\0')) {
        return -1;
    }
    return value;
}

// The counts of the retained hybrid partition's [MOE-ROUTE] line: occurrences on this card (local), on another card,
// on the host, and the ones no side could take.
struct route_counts {
    long long local           = -1;
    long long other_gpu       = -1;
    long long host            = -1;
    long long missing         = -1;
    long long layout_mismatch = -1;
};

inline route_counts parse_route_line(const std::string & line) {
    route_counts counts;
    counts.local           = route_field(line, "local");
    counts.other_gpu       = route_field(line, "other_gpu");
    counts.host            = route_field(line, "host");
    counts.missing         = route_field(line, "missing");
    counts.layout_mismatch = route_field(line, "layout_mismatch");
    return counts;
}

// The parser's own controls, run before anything else: it must read the partition line's counts, and it must not read
// the retained prompt route's a/b counts as counts. A parser that read neither would fail every route check for the
// wrong reason; one that read both would let the wrong line pass.
inline void check_route_parser() {
    const route_counts partition = parse_route_line(
        "[MOE-ROUTE] tensor=blk.0.ffn_gate_exps.weight layer=0 submit_device=0 layout=0 local=3 other_gpu=0 host=5 "
        "missing=0 layout_mismatch=0 unsupported=0\n");
    require(partition.local == 3 && partition.other_gpu == 0 && partition.host == 5 && partition.missing == 0 &&
                partition.layout_mismatch == 0,
            "the route parser misread a partition line");
    const route_counts retained = parse_route_line(
        "[MOE-ROUTE] tensor=blk.0.ffn_gate_exps.weight layer=0 submit_device=0 layout=0 local=3/4 other_gpu=0/0 "
        "host=5/6 unavailable=0/0 localized=0 host_rerouted=0\n");
    require(retained.local == -1 && retained.other_gpu == -1 && retained.host == -1,
            "the route parser read a retained prompt route line's a/b counts as counts");
}

// One planned model: three expert tensors of one shape.
struct expert_inventory {
    ggml_sycl_tensor_info        tensors[3]{};
    ggml_sycl_tensor_inventory   inventory{};
    ggml_sycl_placement_envelope envelope{ k_n_ctx, k_n_ubatch, 1, -1 };

    explicit expert_inventory(ggml_type type) {
        for (int i = 0; i < 3; ++i) {
            tensors[i].name  = k_tensor_names[i];
            tensors[i].type  = type;
            tensors[i].ne[0] = k_in;
            tensors[i].ne[1] = k_out;
            tensors[i].ne[2] = k_experts;
            tensors[i].ne[3] = 1;
            tensors[i].size  = ggml_row_size(type, k_in) * k_out * k_experts;
            inventory.total_size += tensors[i].size;
        }
        inventory.tensors       = tensors;
        inventory.count         = 3;
        inventory.n_expert      = k_experts;
        inventory.n_expert_used = k_used;
        inventory.n_layer       = 1;
        inventory.n_ctx         = k_n_ctx;
        inventory.n_ubatch      = k_n_ubatch;
    }
};

// One expert tensor's quantized bytes: k_experts experts of k_out rows of k_in values.
inline std::vector<uint8_t> make_weights(ggml_type type, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float>                    weight_f32(size_t(k_experts) * k_out * k_in);
    for (float & v : weight_f32) {
        v = 0.5f * dist(rng);
    }
    std::vector<uint8_t> bytes(ggml_row_size(type, k_in) * k_out * k_experts);
    const size_t         written =
        ggml_quantize_chunk(type, weight_f32.data(), bytes.data(), 0, int64_t(k_experts) * k_out, k_in, nullptr);
    require(written == bytes.size(), std::string("quantize wrote a short buffer for ") + ggml_type_name(type));
    return bytes;
}

// The model lifecycle a real load drives, through the public entry points, with one step a real load does not take:
// the plan edit between the inventory stage and the weight load (demote_host_experts). `split_path` is the GGUF file
// the weights register as read from; each test passes its own.
struct lifecycle_fixture {
    ggml_backend_t            backend = nullptr;
    ggml_sycl_load_txn        load{};
    bool                      load_open = false;
    ggml_sycl_model_token     model{};
    ggml_sycl_exec_context_id context{};
    ggml_context *            weight_ctx    = nullptr;
    ggml_backend_buffer_t     weight_buffer = nullptr;
    ggml_tensor *             tensors[3]    = {};
    std::vector<uint8_t>      weight_bytes;  // every expert tensor's quantized bytes, for the CPU reference
    // Per expert tensor and expert: true when the plan the commit published puts the expert on the host.
    std::array<std::array<bool, k_experts>, 3> on_host{};

    lifecycle_fixture(ggml_type type, const std::string & split_path, std::mt19937 & rng) {
        try {
            require(ggml_backend_sycl_model_load_begin(&load) == GGML_SYCL_LIFECYCLE_OK, "load begin failed");
            load_open = true;
            expert_inventory fixture(type);
            const auto       stage =
                ggml_backend_sycl_stage_load_inventory_plan(&fixture.inventory, &fixture.envelope, false);
            require(stage == GGML_SYCL_LIFECYCLE_OK,
                    "inventory staging result=" + std::to_string(static_cast<int>(stage)));
            demote_host_experts();
            // Staging uses temporary backends; the retained backend is created after it, so load_end's exact-queue
            // lookup cannot observe a transient context that was cleared.
            backend = ggml_backend_sycl_init(0);
            require(backend != nullptr, "ggml_backend_sycl_init(0) returned null");
            load_weights(type, split_path, rng);
            const auto commit = ggml_backend_sycl_model_load_end(load, true, &model);
            if (commit != GGML_SYCL_LIFECYCLE_BUSY) {
                load_open = false;
            }
            require(commit == GGML_SYCL_LIFECYCLE_OK, "load commit result=" + std::to_string(static_cast<int>(commit)));
            read_published_residency();
            require(ggml_backend_sycl_activate_model_plan(model) == GGML_SYCL_LIFECYCLE_OK, "plan activation failed");
            require(ggml_backend_sycl_execution_context_create(&context) == GGML_SYCL_EXECUTION_OK,
                    "context create failed");
            require(ggml_backend_sycl_execution_context_bind_backend(backend, context) == GGML_SYCL_EXECUTION_OK,
                    "context bind failed");
            require(
                ggml_backend_sycl_set_runtime_context_for_model(backend, model, k_n_ctx, k_n_ubatch, 1,
                                                                /*kv_unified=*/false, /*swa_full=*/false,
                                                                /*flash_attn_enabled=*/true) == GGML_SYCL_LIFECYCLE_OK,
                "runtime context bind failed");
        } catch (...) {
            cleanup();
            throw;
        }
    }

    // Moves k_host_experts of every expert tensor to the host in the load's candidate plan, and stages the edited copy
    // as the candidate the commit publishes. The other experts stay where the planner put them. The candidate must
    // carry no STORAGE table: only such a plan holds an IO-pool lease, which staging the copy without it would drop.
    void demote_host_experts() {
        const auto candidate = ggml_sycl::lifecycle_find_candidate_placement_plan(load.id);
        require(candidate && candidate->plan, "the inventory stage left no candidate plan");
        require(candidate->plan->lazy_tables.empty(), "the candidate plan carries a STORAGE table");
        ggml_sycl::placement_plan plan(*candidate->plan);
        for (const char * name : k_tensor_names) {
            for (int expert : k_host_experts) {
                require(plan.demote_expert_placement_to_host(name, expert),
                        std::string("the candidate plan has no placement entry for ") + name + " expert " +
                            std::to_string(expert));
            }
        }
        ggml_sycl::lifecycle_stage_placement_plan(load.id, std::move(plan), candidate->kv_info,
                                                  candidate->model_n_layer);
    }

    // What libllama's loader does for each weight, in its order (register_sycl_tensor_metadata in
    // src/llama-model-loader.cpp): register the split's file identity, the weight's identity, its host-weight row and
    // its usage while the load is open, allocate the tensors in the device's SYCL host buffer type, then fill them.
    void load_weights(ggml_type type, const std::string & split_path, std::mt19937 & rng) {
        ggml_init_params params = { 3 * ggml_tensor_overhead(), nullptr, true };
        weight_ctx              = ggml_init(params);
        require(weight_ctx != nullptr, "ggml_init for the weights failed");
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        for (int i = 0; i < 3; ++i) {
            tensors[i] = ggml_new_tensor_3d(weight_ctx, type, k_in, k_out, k_experts);
            ggml_set_name(tensors[i], k_tensor_names[i]);
            const size_t nbytes = ggml_nbytes(tensors[i]);
            ggml_backend_sycl_register_gguf_file_identity(0, split_path.c_str(), uint64_t(3 * nbytes));
            ggml_backend_sycl_register_weight_identity(tensors[i], 0, size_t(i) * nbytes, nbytes, 0);
            ggml_backend_sycl_register_host_weight_tensor(dev, tensors[i]);
            require(ggml_backend_sycl_try_register_weight_usage(k_tensor_names[i],
                                                                GGML_SYCL_TENSOR_USAGE_MOE_EXPERT_WEIGHT),
                    std::string("usage registration refused for ") + k_tensor_names[i]);
        }
        ggml_backend_buffer_type_t buft = ggml_backend_sycl_host_buffer_type_for_device(dev);
        require(buft != nullptr, "no SYCL host buffer type for the pinned card");
        weight_buffer = ggml_backend_alloc_ctx_tensors_from_buft(weight_ctx, buft);
        require(weight_buffer != nullptr, "host weight buffer allocation failed");
        ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        // One quantization serves all three tensors, which share one shape.
        weight_bytes = make_weights(type, rng);
        for (int i = 0; i < 3; ++i) {
            ggml_backend_tensor_set(tensors[i], weight_bytes.data(), 0, weight_bytes.size());
        }
    }

    // Reads the side each expert of each expert tensor runs on from the plan the commit published, the plan the op
    // routes by. Every moved expert must be on the host there, and every other one on this card: with one card visible
    // (each test's main() checks) an expert on another card means the plan is not the one this fixture built.
    void read_published_residency() {
        const auto published = ggml_sycl::lifecycle_find_placement_plan(model.model_id, model.load_txn_id);
        require(published && published->plan, "the commit published no plan");
        for (int t = 0; t < 3; ++t) {
            for (int e = 0; e < k_experts; ++e) {
                const auto placement = published->plan->lookup_expert_placement(k_tensor_names[t], e);
                require(placement.status == ggml_sycl::expert_placement_status::FOUND,
                        std::string("the published plan has no placement for ") + k_tensor_names[t] + " expert " +
                            std::to_string(e));
                require(!placement.on_device || placement.target_device == 0,
                        std::string("the published plan puts ") + k_tensor_names[t] + " expert " + std::to_string(e) +
                            " on device " + std::to_string(placement.target_device) + ", not this card");
                on_host[t][e] = !placement.on_device;
            }
            for (int e : k_host_experts) {
                const std::string moved = std::string(k_tensor_names[t]) + " expert " + std::to_string(e);
                require(on_host[t][e], "the published plan keeps moved " + moved +
                                           " on the card: the edited candidate did not reach the commit");
            }
        }
    }

    void cleanup() noexcept {
        ggml_sycl_exec_drain_ticket             ticket{};
        ggml_sycl_exec_control_host_alloc_batch batch{};
        if (context.value &&
            ggml_backend_sycl_execution_context_begin_drain(context, &ticket) == GGML_SYCL_EXECUTION_OK &&
            ggml_backend_sycl_execution_context_extract_control_host_allocs(&ticket, &batch) ==
                GGML_SYCL_EXECUTION_OK) {
            (void) ggml_backend_sycl_execution_context_release_control_host_allocs(ticket, &batch);
            (void) ggml_backend_sycl_execution_context_finish_drain(ticket, &batch);
        }
        context = {};
        // libllama frees the model's buffers before it reports the model unloaded (llama_model's destructor).
        if (weight_buffer) {
            ggml_backend_buffer_free(weight_buffer);
            weight_buffer = nullptr;
        }
        if (weight_ctx) {
            ggml_free(weight_ctx);
            weight_ctx = nullptr;
        }
        for (ggml_tensor *& tensor : tensors) {
            tensor = nullptr;
        }
        if (model.model_id) {
            (void) ggml_backend_sycl_model_unloaded_token(model);
            model = {};
        } else if (load_open) {
            (void) ggml_backend_sycl_model_load_end(load, false, nullptr);
            load_open = false;
        }
        if (backend) {
            ggml_backend_free(backend);
            backend = nullptr;
        }
    }

    ~lifecycle_fixture() { cleanup(); }
};

inline ggml_backend_buffer_t alloc_tensor(ggml_backend_buffer_type_t buft,
                                          ggml_tensor *              tensor,
                                          ggml_backend_buffer_usage  usage) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft, tensor));
    if (!buffer) {
        return nullptr;
    }
    ggml_backend_buffer_set_usage(buffer, usage);
    ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer));
    return buffer;
}

}  // namespace moe_mixed_residency_fixture
