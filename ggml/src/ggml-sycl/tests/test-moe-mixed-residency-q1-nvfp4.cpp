// Mixed-residency MUL_MAT_ID for Q1_0 and NVFP4 experts on one card: in ONE op, the experts the plan keeps on the card
// run through mmvq_moe_batched_dispatch and the experts it puts on the host run through the HOST_CPU recipe
// (ggml_sycl_cpu_moe_host_aos_execute), and the op's result matches the CPU backend.
//
// For each type, a planned model is loaded the way libllama loads one (test-moe-pp-batched-dequant.cpp describes the
// sequence): its three expert tensors sit in the card's SYCL host buffer type and are registered while the load is
// open. Between the inventory stage and the weight load, the fixture copies the load's candidate plan, moves the
// experts in k_host_experts of every expert tensor to the host (placement_plan::demote_expert_placement_to_host keeps
// the plan's byte terms consistent), and stages the copy as the candidate again. The commit's preload then
// materializes the other experts on the card and registers the moved ones as host experts. The cases take the side
// each expert runs on from the plan the commit published.
//
// Each case runs one MUL_MAT_ID op at 1 token (decode), 7 tokens and 512 tokens (prompt), on the gate tensor with the
// activation broadcast to every used expert (ne11 1) and on the down tensor with one activation row per used expert
// (ne11 k_used). A case passes when:
//   * the compute succeeds and every output value is finite: dst is filled with NaN before the op, so a row that
//     neither side wrote fails;
//   * the rows of the device-planned experts and the rows of the host-planned experts each match the CPU backend at
//     NMSE <= 5e-4, measured apart, so a side that computed nothing or the wrong thing fails on its own;
//   * the op logs exactly one [MOE-ROUTE] partition line for the tensor (GGML_SYCL_MOE_ROUTE_LOG=1, set below), and its
//     local= and host= counts equal the occurrences of device-planned and of host-planned experts in the case's ids,
//     with other_gpu=, missing= and layout_mismatch= at 0;
//   * nothing refused or ran another way: no line in k_refusal_needles appeared while it ran.
// A Q1_0/NVFP4 operand on the card has one executor, the batched dispatch, and the op ends with a named refusal and a
// failed compute when that dispatch returns false; a host operand has one, the admitted recipe, and the process aborts
// when that recipe refuses. So correct rows on both sides of a successful op are what each executor computed.
//
// The process exits 1 if any case failed. Otherwise it exits 77 when a case ran experts on one side only (a plan that
// left every picked expert on one side measures nothing here), and 0 when every case ran both sides. It also exits 77
// when there is no SYCL GPU, when more than one card is visible, or when the weights are not evictable (libllama would
// then load the device-planned experts into the device buffer, which this fixture does not model). Pin one card: the
// registration selects level_zero:1, the B50.
#include "../../../../tests/test-skip.h"  // LLAMA_TEST_EXIT_SKIP: the one definition of "77 means skip"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl.h"
#include "ggml.h"
#include "unified-cache.hpp"  // the load's candidate plan, edited and staged again before the weights load

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

namespace {

constexpr int      k_experts  = 16;
constexpr int      k_used     = 4;
constexpr int      k_in       = 2048;  // k: a whole number of Q1_0 (128) and of NVFP4 (64) blocks
constexpr int      k_out      = 512;   // m: output rows per expert
// The largest case's token count: the planner sizes the host recipe's workspace for a prompt op of up to n_ubatch rows.
constexpr uint32_t k_n_ubatch = 512;
constexpr uint32_t k_n_ctx    = 512;
constexpr double   k_nmse_tol = 5e-4;

// The three expert tensors share one shape. The gate cases run the first, the down cases the third.
constexpr const char * k_tensor_names[3] = { "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight",
                                             "blk.0.ffn_down_exps.weight" };
constexpr int          k_gate            = 0;
constexpr int          k_down            = 2;

// The experts the fixture moves to the host in every expert tensor. Seven of sixteen, so a case's device and host
// counts are not equal by construction (the one-token cases pick one device and three host experts), and a route line
// that swapped the two counts fails.
constexpr int k_host_experts[] = { 1, 4, 6, 9, 10, 13, 15 };

constexpr ggml_type k_types[]  = { GGML_TYPE_Q1_0, GGML_TYPE_NVFP4 };
constexpr int       k_tokens[] = { 1, 7, 512 };
constexpr int       k_roles[]  = { k_gate, k_down };
constexpr int       k_cases    = int(std::size(k_types) * std::size(k_tokens) * std::size(k_roles));

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

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// The environment cannot exercise what this test covers; main() reports it as ctest's skip (77), never as a pass, on a
// line prefixed SKIP[<tag>] so each cause stays distinguishable in a log.
struct test_skip : std::runtime_error {
    test_skip(const char * tag, const std::string & why) : std::runtime_error(why), tag(tag) {}

    const char * tag;
};

// Every log chunk the backend emits, in order, so a case can read what appeared while it ran. Also echoed to stderr, so
// the run's log carries the backend's own lines.
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

log_capture g_log;

void log_callback(ggml_log_level level, const char * text, void * user_data) {
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
// (the retained prompt route's own line writes its counts as a/b, and it is not the partition line a case expects).
long long route_field(const std::string & line, const char * name) {
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

route_counts parse_route_line(const std::string & line) {
    route_counts counts;
    counts.local           = route_field(line, "local");
    counts.other_gpu       = route_field(line, "other_gpu");
    counts.host            = route_field(line, "host");
    counts.missing         = route_field(line, "missing");
    counts.layout_mismatch = route_field(line, "layout_mismatch");
    return counts;
}

// The parser's own controls, run before any case: it must read the partition line's counts, and it must not read the
// retained prompt route's a/b counts as counts. A parser that read neither would fail every case for the wrong reason;
// one that read both would let the wrong line pass.
void check_route_parser() {
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
std::vector<uint8_t> make_weights(ggml_type type, std::mt19937 & rng) {
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
// the plan edit between the inventory stage and the weight load (demote_host_experts).
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

    lifecycle_fixture(ggml_type type, std::mt19937 & rng) {
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
            load_weights(type, rng);
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
    void load_weights(ggml_type type, std::mt19937 & rng) {
        ggml_init_params params = { 3 * ggml_tensor_overhead(), nullptr, true };
        weight_ctx              = ggml_init(params);
        require(weight_ctx != nullptr, "ggml_init for the weights failed");
        ggml_backend_dev_t dev        = ggml_backend_get_device(backend);
        const std::string  split_path = std::string("test-moe-mixed-residency-") + ggml_type_name(type) + ".gguf";
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
    // (main() checks) an expert on another card means the plan is not the one this fixture built.
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

ggml_backend_buffer_t alloc_tensor(ggml_backend_buffer_type_t buft,
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

struct case_data {
    ggml_type            type;
    int                  role;  // the expert tensor the op runs: k_gate or k_down
    int                  n_tokens;
    int                  ne11;  // 1: the activation broadcasts to every used expert (gate); k_used: one row each
    std::vector<float>   input;
    std::vector<int32_t> ids;
};

// Token t picks experts b, b+3, b+6 and b+9 (mod k_experts) from a base b that moves with t, so a token's picks are
// distinct and, against k_host_experts, the picks of every op here fall on both sides; 512 tokens pick every expert.
case_data make_case(ggml_type type, int role, int n_tokens, std::mt19937 & rng) {
    const int                             ne11 = role == k_gate ? 1 : k_used;
    case_data                             c{ type, role, n_tokens, ne11, {}, {} };
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    c.input.resize(size_t(k_in) * ne11 * n_tokens);
    for (float & v : c.input) {
        v = dist(rng);
    }
    c.ids.resize(size_t(k_used) * n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        const int base = (t * 5 + t / 3 + 1) % k_experts;
        for (int i = 0; i < k_used; ++i) {
            c.ids[size_t(t) * k_used + i] = (base + i * 3) % k_experts;
        }
    }
    return c;
}

// Builds MUL_MAT_ID over the case's input and ids on `backend` and computes it, with dst filled with NaN first. Fills
// `out` on success. The SYCL run passes the fixture's loaded tensor as `loaded_weights`; the CPU reference passes null,
// and `weight_bytes` go into a fresh tensor on the CPU backend.
ggml_status run_backend(ggml_backend_t               backend,
                        ggml_tensor *                loaded_weights,
                        const std::vector<uint8_t> & weight_bytes,
                        const case_data &            c,
                        std::vector<float> &         out) {
    ggml_init_params                                    params = { 1024 * 1024, nullptr, true };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), &ggml_free);
    require(ctx != nullptr, "ggml_init failed");
    ggml_tensor * weights = loaded_weights;
    if (!weights) {
        weights = ggml_new_tensor_3d(ctx.get(), c.type, k_in, k_out, k_experts);
        ggml_set_name(weights, k_tensor_names[c.role]);
    }
    ggml_tensor * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k_in, c.ne11, c.n_tokens);
    ggml_tensor * ids   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, k_used, c.n_tokens);
    ggml_set_name(input, "moe_input");
    ggml_set_name(ids, "moe_ids");
    ggml_tensor * dst = ggml_mul_mat_id(ctx.get(), weights, input, ids);
    ggml_set_name(dst, "moe_out");

    ggml_backend_buffer_type_t         buft    = ggml_backend_get_default_buffer_type(backend);
    std::vector<ggml_backend_buffer_t> buffers = {
        alloc_tensor(buft, input, GGML_BACKEND_BUFFER_USAGE_COMPUTE),
        alloc_tensor(buft, ids, GGML_BACKEND_BUFFER_USAGE_COMPUTE),
        alloc_tensor(buft, dst, GGML_BACKEND_BUFFER_USAGE_COMPUTE),
    };
    if (!loaded_weights) {
        buffers.push_back(alloc_tensor(buft, weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS));
    }

    struct buffers_guard {
        std::vector<ggml_backend_buffer_t> & b;

        ~buffers_guard() {
            for (ggml_backend_buffer_t buffer : b) {
                if (buffer) {
                    ggml_backend_buffer_free(buffer);
                }
            }
        }
    } guard{ buffers };

    for (ggml_backend_buffer_t buffer : buffers) {
        require(buffer != nullptr, "tensor buffer allocation failed");
    }
    if (!loaded_weights) {
        ggml_backend_tensor_set(weights, weight_bytes.data(), 0, weight_bytes.size());
    }
    ggml_backend_tensor_set(input, c.input.data(), 0, c.input.size() * sizeof(float));
    ggml_backend_tensor_set(ids, c.ids.data(), 0, c.ids.size() * sizeof(int32_t));
    const std::vector<float> poison(size_t(k_out) * k_used * c.n_tokens, std::numeric_limits<float>::quiet_NaN());
    ggml_backend_tensor_set(dst, poison.data(), 0, poison.size() * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, dst);
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    if (status == GGML_STATUS_SUCCESS) {
        out.resize(poison.size());
        ggml_backend_tensor_get(dst, out.data(), 0, out.size() * sizeof(float));
    }
    return status;
}

enum class case_result { PASS, FAIL, ONE_SIDED };

// Runs one case and prints its CASE line. Every log count is of the lines that appeared while the SYCL op ran.
case_result run_case(lifecycle_fixture & life, ggml_backend_t cpu, const case_data & c) {
    const char *       tensor_name = k_tensor_names[c.role];
    std::vector<float> ref;
    require(run_backend(cpu, nullptr, life.weight_bytes, c, ref) == GGML_STATUS_SUCCESS,
            std::string("CPU reference compute failed for ") + ggml_type_name(c.type));

    // What the op must route: each occurrence (token, slot) to the side the published plan puts its expert on.
    const auto & on_host    = life.on_host[c.role];
    size_t       want_local = 0;
    size_t       want_host  = 0;
    for (int32_t expert : c.ids) {
        ++(on_host[expert] ? want_host : want_local);
    }

    const size_t       from = g_log.mark();
    std::vector<float> got;
    const ggml_status  status = run_backend(life.backend, life.tensors[c.role], life.weight_bytes, c, got);

    // The device rows ([0]) and the host rows ([1]) against the CPU reference, apart.
    double err[2]    = {};
    double norm[2]   = {};
    size_t nonfinite = 0;
    if (status == GGML_STATUS_SUCCESS) {
        for (size_t o = 0; o < c.ids.size(); ++o) {
            const int side = on_host[c.ids[o]] ? 1 : 0;
            for (size_t n = 0; n < size_t(k_out); ++n) {
                const size_t i = o * k_out + n;
                if (!std::isfinite(got[i])) {
                    ++nonfinite;
                    continue;
                }
                const double d = double(ref[i]) - double(got[i]);
                err[side] += d * d;
                norm[side] += double(ref[i]) * double(ref[i]);
            }
        }
    }
    auto nmse = [&](int side) {
        return status != GGML_STATUS_SUCCESS ? NAN : err[side] / (norm[side] > 0.0 ? norm[side] : 1.0);
    };

    const std::string  route_key   = std::string("[MOE-ROUTE] tensor=") + tensor_name + " layer=";
    const auto         route_lines = g_log.since(from, { route_key.c_str() });
    const route_counts route       = route_lines.size() == 1 ? parse_route_line(route_lines[0]) : route_counts{};
    size_t             refusals    = 0;
    for (const char * needle : k_refusal_needles) {
        refusals += g_log.since(from, { needle }).size();
    }

    std::string why;
    if (status != GGML_STATUS_SUCCESS) {
        why += " compute-status";
    } else {
        if (nonfinite != 0) {
            why += " nonfinite";
        }
        if (!(nmse(0) <= k_nmse_tol)) {
            why += " nmse-device";
        }
        if (!(nmse(1) <= k_nmse_tol)) {
            why += " nmse-host";
        }
    }
    if (route_lines.size() != 1) {
        why += " route-lines";
    } else if (route.local != (long long) want_local || route.host != (long long) want_host || route.other_gpu != 0 ||
               route.missing != 0 || route.layout_mismatch != 0) {
        why += " route-counts";
    }
    if (refusals != 0) {
        why += " refusal";
    }
    // A passing case measured both executors only when the op had occurrences on both sides.
    const case_result result = !why.empty()                        ? case_result::FAIL :
                               (want_local == 0 || want_host == 0) ? case_result::ONE_SIDED :
                                                                     case_result::PASS;
    std::fprintf(stderr,
                 "CASE type=%s tensor=%s tokens=%d ne11=%d status=%d want_local=%zu want_host=%zu route_lines=%zu "
                 "local=%lld other_gpu=%lld host=%lld missing=%lld layout_mismatch=%lld nmse_device=%.3e "
                 "nmse_host=%.3e nonfinite=%zu refusals=%zu -> %s%s\n",
                 ggml_type_name(c.type), tensor_name, c.n_tokens, c.ne11, static_cast<int>(status), want_local,
                 want_host, route_lines.size(), route.local, route.other_gpu, route.host, route.missing,
                 route.layout_mismatch, nmse(0), nmse(1), nonfinite, refusals,
                 result == case_result::PASS      ? "PASS" :
                 result == case_result::ONE_SIDED ? "ONE-SIDED" :
                                                    "FAIL:",
                 why.c_str());
    return result;
}

}  // namespace

int main() {
    // Read once by the backend, so it is set before anything touches it.
    setenv("GGML_SYCL_MOE_ROUTE_LOG", "1", 1);
    ggml_log_set(log_callback, nullptr);
    try {
        check_route_parser();
        bool have_gpu = false;
        try {
            for (const auto & device : sycl::device::get_devices()) {
                have_gpu |= device.is_gpu();
            }
        } catch (const sycl::exception & e) {
            throw test_skip("no-gpu", std::string("SYCL device enumeration failed: ") + e.what());
        }
        if (!have_gpu || ggml_backend_sycl_get_device_count() < 1) {
            throw test_skip("no-gpu", "no usable SYCL GPU");
        }
        if (ggml_backend_sycl_get_device_count() > 1) {
            throw test_skip("multi-card",
                            "more than one SYCL card is visible; the expected counts are for one card "
                            "(pin ONEAPI_DEVICE_SELECTOR to one card)");
        }
        if (!ggml_backend_sycl_weights_evictable()) {
            throw test_skip("not-evictable",
                            "GGML_SYCL_WEIGHTS_EVICTABLE is off, so libllama would load the device-planned experts "
                            "into the device buffer, which this fixture does not model");
        }
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(
            ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr), &ggml_backend_free);
        require(cpu != nullptr, "CPU backend init failed");

        std::mt19937 rng(1607);
        int          failed    = 0;
        int          one_sided = 0;
        for (ggml_type type : k_types) {
            require(k_in % ggml_blck_size(type) == 0,
                    std::string("k_in is not whole blocks of ") + ggml_type_name(type));
            lifecycle_fixture life(type, rng);
            for (int n_tokens : k_tokens) {
                for (int role : k_roles) {
                    const case_result result = run_case(life, cpu.get(), make_case(type, role, n_tokens, rng));
                    failed += result == case_result::FAIL ? 1 : 0;
                    one_sided += result == case_result::ONE_SIDED ? 1 : 0;
                }
            }
        }
        if (failed != 0) {
            std::fprintf(stderr, "FAIL: %d of %d cases failed\n", failed, k_cases);
            return 1;
        }
        if (one_sided != 0) {
            std::fprintf(stderr,
                         "SKIP[one-sided]: %d of %d cases ran experts on one side only, so they measured no mixed op\n",
                         one_sided, k_cases);
            return LLAMA_TEST_EXIT_SKIP;
        }
        std::fprintf(stderr, "PASS: %d mixed-residency cases, each with device and host experts in one op\n", k_cases);
        return 0;
    } catch (const test_skip & e) {
        std::fprintf(stderr, "SKIP[%s]: %s\n", e.tag, e.what());
        return LLAMA_TEST_EXIT_SKIP;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
