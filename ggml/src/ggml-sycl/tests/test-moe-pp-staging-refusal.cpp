// A MUL_MAT_ID prompt op whose staging allocation is not granted ends with a named refusal and a failed compute, and
// the process goes on: the same op, run again in the same process, succeeds and matches the CPU backend.
//
// The MoE prompt path of ggml_sycl_mul_mat_id allocates its row-map staging, then (on the per-expert path) its output
// staging and its F32 activation staging. When the card does not grant one, the op refuses through
// ggml_sycl_fallback_error: one WARN, "[MOE-PP-STAGING] refused tensor=... buffer=... rows=... bytes=... device=...",
// names what did not fit, and graph_compute turns the throw into GGML_STATUS_FAILED with its recoverable fallback line.
// No ordinary run makes those allocations fail, so this test arms one private-testing failpoint per site
// (GGML_SYCL_Q1_NVFP4_TEST_FAILURE_PP_ROW_MAP_STAGING, _PP_DST_STAGING, _PP_SRC1_F32_STAGING), each of which makes its
// allocation act as not granted, once. It therefore links the private backend objects (ggml-sycl-private-fixtures),
// which compile the MMID route in; ggml_sycl::test_moe_mmid_route_reachable_override_guard(false) closes that route
// for the whole run, so the op dispatches as in an ordinary build.
//
// The vehicle is a hand-built unplanned MoE model, no libllama: one Q4_0 expert tensor (8 experts of 256 x 256) in a
// WEIGHTS buffer of the card's device buffer type, registered and filled inside a load transaction that stages no
// placement plan, and a bound execution context. The process holds the AoS layout override, so the prompt admission
// publishes each expert as an AoS view of the device buffer (the canonical publication, which an unplanned operand
// needs). GGML_SYCL_DISABLE_GRAPH=1 runs the graph directly. The op runs at 128 tokens with 2 experts per token. That
// shape takes it past every prompt fast path to the per-expert prompt route: the fused ESIMD kernel takes Q8_0 and
// MXFP4 only, and at most 32 tokens; the retained MMVQ path declines above its adaptive threshold, experts x 16 / used
// = 64 tokens; the AoS override rules out the XMX sorted path, which wants SoA; the grouped MXFP4 down path is MXFP4
// only; and the batched oneDNN executor declines an unplanned op.
//
// First an unarmed baseline run must match the CPU backend. Then, for each failpoint, a refused run passes when:
//   * the compute returns GGML_STATUS_FAILED;
//   * the failpoint was consumed exactly once;
//   * exactly one [MOE-PP-STAGING] WARN appeared, and it names the tensor, the buffer, the rows and bytes asked, and
//     device 0;
//   * graph_compute logged the recoverable fallback once, with the exception text that names the staging.
// and the recovered run after it, nothing armed, on the same graph, passes when the compute succeeds, every value is
// finite (dst is filled with NaN first), the output matches the CPU backend at NMSE <= 5e-4, no [MOE-PP-STAGING] or
// fallback line appeared, and no failpoint was consumed.
//
// The process exits 1 if any run failed and 0 when all passed, after it has unloaded the model. It exits 77 when there
// is no SYCL GPU or more than one card is visible. Pin one card: the registration selects level_zero:1, the B50.
#include "../../../../tests/test-skip.h"  // LLAMA_TEST_EXIT_SKIP: the one definition of "77 means skip"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl-test.hpp"  // test_moe_mmid_route_reachable_override_guard, test_layout_override_guard
#include "ggml-sycl.h"
#include "ggml.h"
#include "q1-nvfp4-production-route-test-seam.hpp"  // the staging failpoints and their counter

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

constexpr ggml_type k_type     = GGML_TYPE_Q4_0;
constexpr int       k_experts  = 8;
constexpr int       k_used     = 2;
constexpr int       k_in       = 256;  // k: a whole number of Q4_0 blocks (32)
constexpr int       k_out      = 256;  // m: output rows per expert
constexpr int       k_tokens   = 128;  // above the fused kernel's 32 and the retained MMVQ path's 64
constexpr double    k_nmse_tol = 5e-4;

// A real expert name, so every name-classified path (the MoE registry, the expert tables) treats it as an expert.
constexpr const char * k_tensor_name = "blk.0.ffn_gate_exps.weight";
constexpr const char * k_split_path  = "test-moe-pp-staging-refusal.gguf";

constexpr const char * k_staging_warn  = "[MOE-PP-STAGING] refused";
constexpr const char * k_fallback_any  = "recoverable runtime fallback failed";
constexpr size_t       k_rows          = size_t(k_used) * k_tokens;     // the prompt's expert rows
constexpr size_t       k_row_map_bytes = k_rows * 2 * sizeof(int32_t);  // one mmid_row_mapping {i1, i2} per row
constexpr size_t       k_dst_bytes     = size_t(k_out) * k_rows * sizeof(float);
constexpr size_t       k_src1_bytes    = size_t(k_in) * k_rows * sizeof(float);

// One failpoint: the allocation it fails, and what the refusal must say about it.
struct staging_arm {
    const char *                    name;
    ggml_sycl_q1_nvfp4_test_failure failure;
    const char *                    buffer;
    size_t                          bytes;
    const char *                    fallback_line;
};

const staging_arm k_arms[] = {
    { "row-map",  GGML_SYCL_Q1_NVFP4_TEST_FAILURE_PP_ROW_MAP_STAGING,  "moe_pp_row_mappings",   k_row_map_bytes,
     "recoverable runtime fallback failed: MUL_MAT_ID prompt row-map staging did not fit"    },
    { "dst",      GGML_SYCL_Q1_NVFP4_TEST_FAILURE_PP_DST_STAGING,      "moe_pp_dst_contiguous", k_dst_bytes,
     "recoverable runtime fallback failed: MUL_MAT_ID prompt output staging did not fit"     },
    { "src1-f32", GGML_SYCL_Q1_NVFP4_TEST_FAILURE_PP_SRC1_F32_STAGING, "moe_pp_src1_f32",       k_src1_bytes,
     "recoverable runtime fallback failed: MUL_MAT_ID prompt activation staging did not fit" },
};

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// The environment cannot exercise what the test covers; main() reports it as ctest's skip (77), never as a pass.
struct test_skip : std::runtime_error {
    test_skip(const char * tag, const std::string & why) : std::runtime_error(why), tag(tag) {}

    const char * tag;
};

// Every log chunk the backend emits, in order, so a run can count what appeared while it ran. Also echoed to stderr,
// so the run's log carries the backend's own lines.
struct log_capture {
    std::mutex               mu;
    std::vector<std::string> chunks;

    size_t mark() {
        std::lock_guard<std::mutex> lock(mu);
        return chunks.size();
    }

    // The number of chunks from `from` on that contain every needle.
    size_t count(size_t from, std::initializer_list<const char *> needles) {
        std::lock_guard<std::mutex> lock(mu);
        size_t                      n = 0;
        for (size_t i = from; i < chunks.size(); ++i) {
            bool all = true;
            for (const char * needle : needles) {
                all = all && chunks[i].find(needle) != std::string::npos;
            }
            n += all ? 1 : 0;
        }
        return n;
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

uint64_t failures_consumed() {
    ggml_sycl_q1_nvfp4_test_counters counters{};
    ggml_sycl_q1_nvfp4_test_counters_read(&counters);
    return counters.failure_consumed;
}

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

// The unplanned model: the expert tensor registered inside a load transaction that stages no plan, a backend on the
// pinned card and a bound execution context. The destructor unloads it the way libllama does: drain the context, free
// the weights, report the model unloaded, free the backend.
struct model_fixture {
    ggml_backend_t            backend = nullptr;
    ggml_sycl_load_txn        load{};
    bool                      load_open = false;
    ggml_sycl_model_token     model{};
    ggml_sycl_exec_context_id context{};
    ggml_context *            weight_ctx    = nullptr;
    ggml_backend_buffer_t     weight_buffer = nullptr;
    ggml_tensor *             weights       = nullptr;
    std::vector<uint8_t>      weight_bytes;  // the quantized experts, for the CPU reference

    explicit model_fixture(std::mt19937 & rng) {
        try {
            require(ggml_backend_sycl_model_load_begin(&load) == GGML_SYCL_LIFECYCLE_OK, "load begin failed");
            load_open = true;
            backend   = ggml_backend_sycl_init(0);
            require(backend != nullptr, "ggml_backend_sycl_init(0) returned null");
            const ggml_init_params params = { ggml_tensor_overhead(), nullptr, true };
            weight_ctx                    = ggml_init(params);
            require(weight_ctx != nullptr, "ggml_init for the weights failed");
            weights = ggml_new_tensor_3d(weight_ctx, k_type, k_in, k_out, k_experts);
            ggml_set_name(weights, k_tensor_name);
            const size_t nbytes = ggml_nbytes(weights);
            ggml_backend_sycl_register_gguf_file_identity(0, k_split_path, uint64_t(nbytes));
            ggml_backend_sycl_register_weight_identity(weights, 0, 0, nbytes, 0);
            require(
                ggml_backend_sycl_try_register_weight_usage(k_tensor_name, GGML_SYCL_TENSOR_USAGE_MOE_EXPERT_WEIGHT),
                std::string("usage registration refused for ") + k_tensor_name);
            weight_buffer =
                ggml_backend_alloc_ctx_tensors_from_buft(weight_ctx, ggml_backend_get_default_buffer_type(backend));
            require(weight_buffer != nullptr, "the weight buffer allocation failed");
            ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            std::vector<float>                    weight_f32(size_t(k_experts) * k_out * k_in);
            for (float & v : weight_f32) {
                v = 0.5f * dist(rng);
            }
            weight_bytes.resize(nbytes);
            const size_t written = ggml_quantize_chunk(k_type, weight_f32.data(), weight_bytes.data(), 0,
                                                       int64_t(k_experts) * k_out, k_in, nullptr);
            require(written == nbytes, "quantize wrote a short buffer");
            ggml_backend_tensor_set(weights, weight_bytes.data(), 0, nbytes);

            const auto commit = ggml_backend_sycl_model_load_end(load, true, &model);
            if (commit != GGML_SYCL_LIFECYCLE_BUSY) {
                load_open = false;
            }
            require(commit == GGML_SYCL_LIFECYCLE_OK, "load commit result=" + std::to_string(static_cast<int>(commit)));
            require(ggml_backend_sycl_execution_context_create(&context) == GGML_SYCL_EXECUTION_OK,
                    "context create failed");
            require(ggml_backend_sycl_execution_context_bind_backend(backend, context) == GGML_SYCL_EXECUTION_OK,
                    "context bind failed");
        } catch (...) {
            cleanup();
            throw;
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
        weights = nullptr;
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

    ~model_fixture() { cleanup(); }

    model_fixture(const model_fixture &)             = delete;
    model_fixture & operator=(const model_fixture &) = delete;
};

// One MUL_MAT_ID prompt op, built once on `backend` so it can be computed more than once on the same graph. The SYCL
// op passes the fixture's loaded tensor as `loaded_weights`; the CPU reference passes null, and `weight_bytes` go into
// a fresh tensor on the CPU backend.
struct mmid_op {
    ggml_backend_t                                      backend;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ nullptr, &ggml_free };
    std::vector<ggml_backend_buffer_t>                  buffers;
    ggml_tensor *                                       dst   = nullptr;
    ggml_cgraph *                                       graph = nullptr;

    mmid_op(ggml_backend_t               on_backend,
            ggml_tensor *                loaded_weights,
            const std::vector<uint8_t> & weight_bytes,
            const std::vector<float> &   input_values,
            const std::vector<int32_t> & ids_values) :
        backend(on_backend) {
        ggml_init_params params = { 1024 * 1024, nullptr, true };
        ctx.reset(ggml_init(params));
        require(ctx != nullptr, "ggml_init failed");
        ggml_tensor * weights = loaded_weights;
        if (!weights) {
            weights = ggml_new_tensor_3d(ctx.get(), k_type, k_in, k_out, k_experts);
            ggml_set_name(weights, k_tensor_name);
        }
        ggml_tensor * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k_in, 1, k_tokens);
        ggml_tensor * ids   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, k_used, k_tokens);
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
        ggml_backend_tensor_set(ids, ids_values.data(), 0, ids_values.size() * sizeof(int32_t));
        graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, dst);
    }

    ~mmid_op() {
        for (ggml_backend_buffer_t buffer : buffers) {
            if (buffer) {
                ggml_backend_buffer_free(buffer);
            }
        }
    }

    mmid_op(const mmid_op &)             = delete;
    mmid_op & operator=(const mmid_op &) = delete;

    // Fills dst with NaN, computes the graph and, on success, reads dst into `out`.
    ggml_status compute(std::vector<float> & out) {
        const std::vector<float> poison(size_t(k_out) * k_rows, std::numeric_limits<float>::quiet_NaN());
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

// A run with nothing armed: the op must compute what the CPU backend computes. `name` is the RUN line's name.
bool run_matches(mmid_op & op, const std::vector<float> & ref, const std::string & name) {
    const uint64_t     consumed_before = failures_consumed();
    const size_t       from            = g_log.mark();
    std::vector<float> got;
    const ggml_status  status   = op.compute(got);
    const uint64_t     consumed = failures_consumed() - consumed_before;

    double err       = 0.0;
    double norm      = 0.0;
    size_t nonfinite = 0;
    if (status == GGML_STATUS_SUCCESS) {
        for (size_t i = 0; i < ref.size(); ++i) {
            if (!std::isfinite(got[i])) {
                ++nonfinite;
                continue;
            }
            const double d = double(ref[i]) - double(got[i]);
            err += d * d;
            norm += double(ref[i]) * double(ref[i]);
        }
    }
    const double nmse      = status != GGML_STATUS_SUCCESS ? NAN : err / (norm > 0.0 ? norm : 1.0);
    const size_t warns     = g_log.count(from, { k_staging_warn });
    const size_t fallbacks = g_log.count(from, { k_fallback_any });

    std::string why;
    if (status != GGML_STATUS_SUCCESS) {
        why += " compute-status";
    } else {
        if (nonfinite != 0) {
            why += " nonfinite";
        }
        if (!(nmse <= k_nmse_tol)) {
            why += " nmse";
        }
    }
    if (warns != 0 || fallbacks != 0) {
        why += " refusal";
    }
    if (consumed != 0) {
        why += " failpoint-consumed";
    }
    std::fprintf(stderr,
                 "RUN name=%s status=%d nmse=%.3e nonfinite=%zu staging_warns=%zu fallback_lines=%zu "
                 "failpoint_consumed=%llu -> %s%s\n",
                 name.c_str(), static_cast<int>(status), nmse, nonfinite, warns, fallbacks,
                 (unsigned long long) consumed, why.empty() ? "PASS" : "FAIL:", why.c_str());
    return why.empty();
}

// The refused run: `arm` armed, the op must end with the named refusal and a failed compute.
bool run_refused(mmid_op & op, const staging_arm & arm) {
    const uint64_t consumed_before = failures_consumed();
    const size_t   from            = g_log.mark();
    ggml_sycl_q1_nvfp4_test_failure_once(arm.failure);
    std::vector<float> got;
    const ggml_status  status   = op.compute(got);
    const uint64_t     consumed = failures_consumed() - consumed_before;
    if (consumed == 0) {
        // The op never reached the failpoint; disarm it so the next run measures only itself.
        ggml_sycl_q1_nvfp4_test_failure_once(GGML_SYCL_Q1_NVFP4_TEST_FAILURE_NONE);
    }
    const std::string named = std::string("tensor=") + k_tensor_name + " buffer=" + arm.buffer +
                              " rows=" + std::to_string(k_rows) + " bytes=" + std::to_string(arm.bytes) + " device=0";
    const size_t named_warns   = g_log.count(from, { k_staging_warn, named.c_str() });
    const size_t any_warns     = g_log.count(from, { k_staging_warn });
    const size_t fallbacks     = g_log.count(from, { arm.fallback_line });
    const size_t any_fallbacks = g_log.count(from, { k_fallback_any });

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
    if (fallbacks != 1 || any_fallbacks != 1) {
        why += " fallback-line";
    }
    std::fprintf(stderr,
                 "RUN name=refused-%s status=%d failpoint_consumed=%llu named_warns=%zu any_warns=%zu "
                 "fallback_lines=%zu any_fallback_lines=%zu -> %s%s\n",
                 arm.name, static_cast<int>(status), (unsigned long long) consumed, named_warns, any_warns, fallbacks,
                 any_fallbacks, why.empty() ? "PASS" : "FAIL:", why.c_str());
    return why.empty();
}

}  // namespace

int main() {
    // Read once by the backend, so it is set before anything touches it.
    setenv("GGML_SYCL_DISABLE_GRAPH", "1", 1);
    ggml_log_set(log_callback, nullptr);
    try {
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
                            "more than one SYCL card is visible; the refusal must name device 0 "
                            "(pin ONEAPI_DEVICE_SELECTOR to one card)");
        }
        // Dispatch as an ordinary build does, and publish the unplanned experts as AoS views, for the whole run: the
        // model is unloaded before either guard ends.
        ggml_sycl::test_moe_mmid_route_reachable_override_guard ordinary_route(false);
        ggml_sycl::test_layout_override_guard                   aos_layout(GGML_LAYOUT_AOS);

        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(
            ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr), &ggml_backend_free);
        require(cpu != nullptr, "CPU backend init failed");

        std::mt19937                          rng(5821);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float>                    input(size_t(k_in) * k_tokens);
        for (float & v : input) {
            v = dist(rng);
        }
        // Two distinct experts per token, every expert taking k_rows / k_experts rows.
        std::vector<int32_t> ids(k_rows);
        for (int t = 0; t < k_tokens; ++t) {
            ids[size_t(t) * k_used + 0] = t % k_experts;
            ids[size_t(t) * k_used + 1] = (t + 3) % k_experts;
        }

        bool ok = true;
        {
            model_fixture      model(rng);
            std::vector<float> ref;
            {
                mmid_op reference(cpu.get(), nullptr, model.weight_bytes, input, ids);
                require(reference.compute(ref) == GGML_STATUS_SUCCESS, "CPU reference compute failed");
            }
            mmid_op op(model.backend, model.weights, model.weight_bytes, input, ids);
            ok = run_matches(op, ref, "baseline") && ok;
            for (const staging_arm & arm : k_arms) {
                ok = run_refused(op, arm) && ok;
                ok = run_matches(op, ref, std::string("recovered-") + arm.name) && ok;
            }
        }
        if (!ok) {
            std::fprintf(stderr, "FAIL: a baseline, refused or recovered run failed\n");
            return 1;
        }
        std::fprintf(stderr,
                     "PASS: each staging refusal failed the op by name and the same op then matched the CPU backend\n");
        return 0;
    } catch (const test_skip & e) {
        std::fprintf(stderr, "SKIP[%s]: %s\n", e.tag, e.what());
        return LLAMA_TEST_EXIT_SKIP;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
