// A Q1_0/NVFP4 AOS MUL_MAT_ID whose batched dispatch refuses ends the op with a named refusal and a failed compute,
// and the process goes on: the same op, run again in the same process, succeeds and matches the CPU backend.
//
// On the card such an operand has one executor, mmvq_moe_batched_dispatch, and the op refuses when it returns false.
// No ordinary run makes it return false, so this test arms the private-testing failpoint
// GGML_SYCL_Q1_NVFP4_TEST_FAILURE_BATCHED_DISPATCH, which makes the dispatch refuse once with the reason
// test-failpoint, just before its submit. It therefore links the private backend objects (ggml-sycl-private-fixtures),
// which compile the MMID route in; ggml_sycl::test_moe_mmid_route_reachable_override_guard(false) closes that route for
// the whole run, so the model plans, binds and executes as in an ordinary build.
//
// The vehicle is the mixed-residency fixture's (test-moe-mixed-residency-q1-nvfp4.cpp describes the load): a planned
// NVFP4 model of three expert tensors whose plan the fixture edits to put 7 of 16 experts on the host. The op is the
// gate tensor at one token with ids {1, 4, 7, 10}: expert 7 runs on the card and the other three on the host.
//
// The first run, with the failpoint armed, passes when:
//   * the compute returns GGML_STATUS_FAILED;
//   * the failpoint was consumed exactly once (failure_consumed +1);
//   * exactly one WARN says the Q1_0/NVFP4 AOS batched dispatch of this tensor returned false, with
//     reason=test-failpoint requested_bytes=0, and no other batched-dispatch refusal appeared;
//   * graph_compute logged the recoverable fallback with the exception text that names test-failpoint.
// The second run, with nothing armed, on the same graph, passes when it meets the mixed-residency case's bar: the
// compute succeeds, every value is finite (dst is filled with NaN first), the device rows and the host rows each match
// the CPU backend at NMSE <= 5e-4, one [MOE-ROUTE] partition line has local=1 host=3 and nothing else, nothing refused,
// and the failpoint was not consumed again.
//
// The process exits 1 if either run failed, and 0 when both passed. It exits 77 when there is no SYCL GPU, when more
// than one card is visible, when the weights are not evictable, or when the published plan leaves the op's experts on
// one side only. Pin one card: the registration selects level_zero:1, the B50.
#include "../../../../tests/test-skip.h"  // LLAMA_TEST_EXIT_SKIP: the one definition of "77 means skip"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl-test.hpp"  // test_moe_mmid_route_reachable_override_guard
#include "ggml-sycl.h"
#include "ggml.h"
#include "q1-nvfp4-production-route-test-seam.hpp"  // the batched-dispatch failpoint and its counter
#include "unified-cache.hpp"                        // the load's candidate plan, edited and staged again

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

namespace {

constexpr ggml_type k_type     = GGML_TYPE_NVFP4;
constexpr int       k_experts  = 16;
constexpr int       k_used     = 4;
constexpr int       k_in       = 2048;  // k: a whole number of NVFP4 (64) blocks
constexpr int       k_out      = 512;   // m: output rows per expert
constexpr uint32_t  k_n_ubatch = 512;
constexpr uint32_t  k_n_ctx    = 512;
constexpr double    k_nmse_tol = 5e-4;

constexpr const char * k_tensor_names[3] = { "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight",
                                             "blk.0.ffn_down_exps.weight" };
constexpr int          k_gate            = 0;

// The experts the fixture moves to the host in every expert tensor, and the op's picks: 7 on the card, 1, 4 and 10 on
// the host, so a route line that swapped the two counts fails.
constexpr int     k_host_experts[] = { 1, 4, 6, 9, 10, 13, 15 };
constexpr int32_t k_ids[k_used]    = { 1, 4, 7, 10 };

constexpr const char * k_refused_warn  = "[MOE-ROUTE] Q1_0/NVFP4 AOS batched dispatch returned false";
constexpr const char * k_refused_cause = "reason=test-failpoint requested_bytes=0";
constexpr const char * k_fallback_line =
    "recoverable runtime fallback failed: MUL_MAT_ID batched dispatch for a Q1_0/NVFP4 AOS operand refused: "
    "test-failpoint";

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

// Every log chunk the backend emits, in order, so a run can read what appeared while it ran. Also echoed to stderr, so
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

// The count that follows " <name>=" in a [MOE-ROUTE] line, or -1 when the field is missing or is not a plain count.
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

// The parser's own control, run before anything else: it must read a partition line's counts, or every run would fail
// its route check for the wrong reason.
void check_route_parser() {
    const route_counts partition = parse_route_line(
        "[MOE-ROUTE] tensor=blk.0.ffn_gate_exps.weight layer=0 submit_device=0 layout=0 local=1 other_gpu=0 host=3 "
        "missing=0 layout_mismatch=0 unsupported=0\n");
    require(partition.local == 1 && partition.other_gpu == 0 && partition.host == 3 && partition.missing == 0 &&
                partition.layout_mismatch == 0,
            "the route parser misread a partition line");
}

uint64_t failures_consumed() {
    ggml_sycl_q1_nvfp4_test_counters counters{};
    ggml_sycl_q1_nvfp4_test_counters_read(&counters);
    return counters.failure_consumed;
}

// One planned model: three expert tensors of one shape.
struct expert_inventory {
    ggml_sycl_tensor_info        tensors[3]{};
    ggml_sycl_tensor_inventory   inventory{};
    ggml_sycl_placement_envelope envelope{ k_n_ctx, k_n_ubatch, 1, -1 };

    expert_inventory() {
        for (int i = 0; i < 3; ++i) {
            tensors[i].name  = k_tensor_names[i];
            tensors[i].type  = k_type;
            tensors[i].ne[0] = k_in;
            tensors[i].ne[1] = k_out;
            tensors[i].ne[2] = k_experts;
            tensors[i].ne[3] = 1;
            tensors[i].size  = ggml_row_size(k_type, k_in) * k_out * k_experts;
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

std::vector<uint8_t> make_weights(std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float>                    weight_f32(size_t(k_experts) * k_out * k_in);
    for (float & v : weight_f32) {
        v = 0.5f * dist(rng);
    }
    std::vector<uint8_t> bytes(ggml_row_size(k_type, k_in) * k_out * k_experts);
    const size_t         written =
        ggml_quantize_chunk(k_type, weight_f32.data(), bytes.data(), 0, int64_t(k_experts) * k_out, k_in, nullptr);
    require(written == bytes.size(), "quantize wrote a short buffer");
    return bytes;
}

// The mixed-residency fixture's model lifecycle: the public entry points a real load drives, with the plan edit between
// the inventory stage and the weight load (demote_host_experts).
struct lifecycle_fixture {
    ggml_backend_t              backend = nullptr;
    ggml_sycl_load_txn          load{};
    bool                        load_open = false;
    ggml_sycl_model_token       model{};
    ggml_sycl_exec_context_id   context{};
    ggml_context *              weight_ctx    = nullptr;
    ggml_backend_buffer_t       weight_buffer = nullptr;
    ggml_tensor *               tensors[3]    = {};
    std::vector<uint8_t>        weight_bytes;  // every expert tensor's quantized bytes, for the CPU reference
    // Per expert of the gate tensor: true when the plan the commit published puts the expert on the host.
    std::array<bool, k_experts> gate_on_host{};

    explicit lifecycle_fixture(std::mt19937 & rng) {
        try {
            require(ggml_backend_sycl_model_load_begin(&load) == GGML_SYCL_LIFECYCLE_OK, "load begin failed");
            load_open = true;
            expert_inventory fixture;
            const auto       stage =
                ggml_backend_sycl_stage_load_inventory_plan(&fixture.inventory, &fixture.envelope, false);
            require(stage == GGML_SYCL_LIFECYCLE_OK,
                    "inventory staging result=" + std::to_string(static_cast<int>(stage)));
            demote_host_experts();
            // Staging uses temporary backends; the retained backend is created after it, so load_end's exact-queue
            // lookup cannot observe a transient context that was cleared.
            backend = ggml_backend_sycl_init(0);
            require(backend != nullptr, "ggml_backend_sycl_init(0) returned null");
            load_weights(rng);
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

    void load_weights(std::mt19937 & rng) {
        ggml_init_params params = { 3 * ggml_tensor_overhead(), nullptr, true };
        weight_ctx              = ggml_init(params);
        require(weight_ctx != nullptr, "ggml_init for the weights failed");
        ggml_backend_dev_t dev        = ggml_backend_get_device(backend);
        const std::string  split_path = "test-moe-q1-nvfp4-dispatch-refusal.gguf";
        for (int i = 0; i < 3; ++i) {
            tensors[i] = ggml_new_tensor_3d(weight_ctx, k_type, k_in, k_out, k_experts);
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
        weight_bytes = make_weights(rng);
        for (int i = 0; i < 3; ++i) {
            ggml_backend_tensor_set(tensors[i], weight_bytes.data(), 0, weight_bytes.size());
        }
    }

    // The side each gate expert runs on, from the plan the commit published: every moved expert on the host, every
    // other one on this card.
    void read_published_residency() {
        const auto published = ggml_sycl::lifecycle_find_placement_plan(model.model_id, model.load_txn_id);
        require(published && published->plan, "the commit published no plan");
        for (int e = 0; e < k_experts; ++e) {
            const auto placement = published->plan->lookup_expert_placement(k_tensor_names[k_gate], e);
            require(placement.status == ggml_sycl::expert_placement_status::FOUND,
                    "the published plan has no placement for gate expert " + std::to_string(e));
            require(!placement.on_device || placement.target_device == 0,
                    "the published plan puts gate expert " + std::to_string(e) + " on device " +
                        std::to_string(placement.target_device) + ", not this card");
            gate_on_host[e] = !placement.on_device;
        }
        for (int e : k_host_experts) {
            require(gate_on_host[e], "the published plan keeps moved gate expert " + std::to_string(e) +
                                         " on the card: the edited candidate did not reach the commit");
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

// One MUL_MAT_ID op on the gate tensor at one token, its activation broadcast to every used expert, built once on
// `backend` so it can be computed more than once on the same graph. The SYCL op passes the fixture's loaded tensor as
// `loaded_weights`; the CPU reference passes null, and `weight_bytes` go into a fresh tensor on the CPU backend.
struct gate_op {
    ggml_backend_t                                      backend;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ nullptr, &ggml_free };
    std::vector<ggml_backend_buffer_t>                  buffers;
    ggml_tensor *                                       dst   = nullptr;
    ggml_cgraph *                                       graph = nullptr;

    gate_op(ggml_backend_t               on_backend,
            ggml_tensor *                loaded_weights,
            const std::vector<uint8_t> & weight_bytes,
            const std::vector<float> &   input_values) :
        backend(on_backend) {
        ggml_init_params params = { 1024 * 1024, nullptr, true };
        ctx.reset(ggml_init(params));
        require(ctx != nullptr, "ggml_init failed");
        ggml_tensor * weights = loaded_weights;
        if (!weights) {
            weights = ggml_new_tensor_3d(ctx.get(), k_type, k_in, k_out, k_experts);
            ggml_set_name(weights, k_tensor_names[k_gate]);
        }
        ggml_tensor * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k_in, 1, 1);
        ggml_tensor * ids   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, k_used, 1);
        ggml_set_name(input, "moe_input");
        ggml_set_name(ids, "moe_ids");
        dst = ggml_mul_mat_id(ctx.get(), weights, input, ids);
        ggml_set_name(dst, "moe_out");

        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
        buffers.push_back(alloc_tensor(buft, input, GGML_BACKEND_BUFFER_USAGE_COMPUTE));
        buffers.push_back(alloc_tensor(buft, ids, GGML_BACKEND_BUFFER_USAGE_COMPUTE));
        buffers.push_back(alloc_tensor(buft, dst, GGML_BACKEND_BUFFER_USAGE_COMPUTE));
        if (!loaded_weights) {
            buffers.push_back(alloc_tensor(buft, weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS));
        }
        for (ggml_backend_buffer_t buffer : buffers) {
            require(buffer != nullptr, "tensor buffer allocation failed");
        }
        if (!loaded_weights) {
            ggml_backend_tensor_set(weights, weight_bytes.data(), 0, weight_bytes.size());
        }
        ggml_backend_tensor_set(input, input_values.data(), 0, input_values.size() * sizeof(float));
        ggml_backend_tensor_set(ids, k_ids, 0, sizeof(k_ids));
        graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, dst);
    }

    ~gate_op() {
        for (ggml_backend_buffer_t buffer : buffers) {
            if (buffer) {
                ggml_backend_buffer_free(buffer);
            }
        }
    }

    gate_op(const gate_op &)             = delete;
    gate_op & operator=(const gate_op &) = delete;

    // Fills dst with NaN, computes the graph and, on success, reads dst into `out`.
    ggml_status compute(std::vector<float> & out) {
        const std::vector<float> poison(size_t(k_out) * k_used, std::numeric_limits<float>::quiet_NaN());
        ggml_backend_tensor_set(dst, poison.data(), 0, poison.size() * sizeof(float));
        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        out.clear();
        if (status == GGML_STATUS_SUCCESS) {
            out.resize(poison.size());
            ggml_backend_tensor_get(dst, out.data(), 0, out.size() * sizeof(float));
        }
        return status;
    }
};

size_t count_needles(size_t from) {
    size_t n = 0;
    for (const char * needle : k_refusal_needles) {
        n += g_log.since(from, { needle }).size();
    }
    return n;
}

// The first run: the failpoint armed, the op must end with the named refusal and a failed compute.
bool run_refused(gate_op & op) {
    const uint64_t consumed_before = failures_consumed();
    const size_t   from            = g_log.mark();
    ggml_sycl_q1_nvfp4_test_failure_once(GGML_SYCL_Q1_NVFP4_TEST_FAILURE_BATCHED_DISPATCH);
    std::vector<float> got;
    const ggml_status  status   = op.compute(got);
    const uint64_t     consumed = failures_consumed() - consumed_before;
    if (consumed == 0) {
        // The dispatch never reached the failpoint; disarm it so the second run measures only itself.
        ggml_sycl_q1_nvfp4_test_failure_once(GGML_SYCL_Q1_NVFP4_TEST_FAILURE_NONE);
    }
    const size_t named_warns = g_log.since(from, { k_refused_warn, k_tensor_names[k_gate], k_refused_cause }).size();
    const size_t any_warns   = g_log.since(from, { "batched dispatch returned false" }).size();
    const size_t fallbacks   = g_log.since(from, { k_fallback_line }).size();

    std::string why;
    if (status != GGML_STATUS_FAILED) {
        why += " compute-status";
    }
    if (consumed != 1) {
        why += " failpoint-consumed";
    }
    if (named_warns != 1 || any_warns != 1) {
        why += " refusal-warn";
    }
    if (fallbacks != 1) {
        why += " fallback-line";
    }
    std::fprintf(stderr,
                 "RUN name=refused status=%d failpoint_consumed=%llu named_warns=%zu any_warns=%zu fallback_lines=%zu "
                 "-> %s%s\n",
                 static_cast<int>(status), (unsigned long long) consumed, named_warns, any_warns, fallbacks,
                 why.empty() ? "PASS" : "FAIL:", why.c_str());
    return why.empty();
}

// The second run: nothing armed, the same graph must compute what the CPU backend computes, on both sides.
bool run_recovered(gate_op &                  op,
                   const std::vector<float> & ref,
                   const lifecycle_fixture &  life,
                   size_t                     want_local,
                   size_t                     want_host) {
    const uint64_t     consumed_before = failures_consumed();
    const size_t       from            = g_log.mark();
    std::vector<float> got;
    const ggml_status  status   = op.compute(got);
    const uint64_t     consumed = failures_consumed() - consumed_before;

    // The device rows ([0]) and the host rows ([1]) against the CPU reference, apart.
    double err[2]    = {};
    double norm[2]   = {};
    size_t nonfinite = 0;
    if (status == GGML_STATUS_SUCCESS) {
        for (size_t o = 0; o < size_t(k_used); ++o) {
            const int side = life.gate_on_host[k_ids[o]] ? 1 : 0;
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

    const std::string  route_key   = std::string("[MOE-ROUTE] tensor=") + k_tensor_names[k_gate] + " layer=";
    const auto         route_lines = g_log.since(from, { route_key.c_str() });
    const route_counts route       = route_lines.size() == 1 ? parse_route_line(route_lines[0]) : route_counts{};
    const size_t       refusals    = count_needles(from);

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
    if (consumed != 0) {
        why += " failpoint-consumed";
    }
    std::fprintf(stderr,
                 "RUN name=recovered status=%d want_local=%zu want_host=%zu route_lines=%zu local=%lld other_gpu=%lld "
                 "host=%lld missing=%lld layout_mismatch=%lld nmse_device=%.3e nmse_host=%.3e nonfinite=%zu "
                 "refusals=%zu failpoint_consumed=%llu -> %s%s\n",
                 static_cast<int>(status), want_local, want_host, route_lines.size(), route.local, route.other_gpu,
                 route.host, route.missing, route.layout_mismatch, nmse(0), nmse(1), nonfinite, refusals,
                 (unsigned long long) consumed, why.empty() ? "PASS" : "FAIL:", why.c_str());
    return why.empty();
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
        // Plan, bind and run as an ordinary build does, for the whole run: the fixture's lifecycle is destroyed first.
        ggml_sycl::test_moe_mmid_route_reachable_override_guard ordinary_route(false);

        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(
            ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr), &ggml_backend_free);
        require(cpu != nullptr, "CPU backend init failed");

        std::mt19937      rng(1931);
        lifecycle_fixture life(rng);

        size_t want_local = 0;
        size_t want_host  = 0;
        for (int32_t expert : k_ids) {
            ++(life.gate_on_host[expert] ? want_host : want_local);
        }
        if (want_local == 0 || want_host == 0) {
            throw test_skip("one-sided", "the published plan puts the op's experts on one side only (local=" +
                                             std::to_string(want_local) + " host=" + std::to_string(want_host) + ")");
        }

        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float>                    input(k_in);
        for (float & v : input) {
            v = dist(rng);
        }
        std::vector<float> ref;
        {
            gate_op reference(cpu.get(), nullptr, life.weight_bytes, input);
            require(reference.compute(ref) == GGML_STATUS_SUCCESS, "CPU reference compute failed");
        }

        bool ok = true;
        {
            gate_op op(life.backend, life.tensors[k_gate], life.weight_bytes, input);
            ok = run_refused(op) && ok;
            ok = run_recovered(op, ref, life, want_local, want_host) && ok;
        }
        if (!ok) {
            std::fprintf(stderr, "FAIL: the refused run or the recovered run failed\n");
            return 1;
        }
        std::fprintf(stderr, "PASS: the refused op failed by name and the same op then matched the CPU backend\n");
        return 0;
    } catch (const test_skip & e) {
        std::fprintf(stderr, "SKIP[%s]: %s\n", e.tag, e.what());
        return LLAMA_TEST_EXIT_SKIP;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
