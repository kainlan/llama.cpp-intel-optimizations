// GPU numerics test for the full-N, small-M (M <= 8) MXFP4 GEMM on the
// stored SOA expert weight layout with int8 DPAS
// (ggml_sycl_mxfp4_stored_gemm::ggml_sycl_mxfp4_soa_gemm_dpas,
// ggml/src/ggml-sycl/mxfp4-stored-gemm.{hpp,cpp}), scored against the Task
// G3 host oracle (tests/mxfp4-stored-layout-oracle.hpp).
// (llama.cpp-vtfs, option C step 2 -- plan Task G4)
//
// NOT wired into any production dispatch path: this test calls the new
// kernel's exported entry point directly, the same way
// tests/test-sycl-onednn-pack-m-propagation.cpp calls internal SYCL-backend
// entry points directly rather than going through ggml_mul_mat.
// ggml-sycl.cpp is unmodified by this task.
//
// GPU and model-loading binaries in this fork are run only from the lead
// session, one at a time (CLAUDE.md, Hard-Won Rules) -- this binary is no
// exception:
//
//   source /opt/intel/oneapi/setvars.sh --force
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-sycl-mxfp4-stored-gemm-soa-small-m   # B50
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-sycl-mxfp4-stored-gemm-soa-small-m   # B70
//
// Coverage:
//   - Main sweep: 4 independently random experts (N=2880, K=2880, the
//     GPT-OSS expert shape) x M in {1, 2, 4, 8} -- 16 GEMM calls, each
//     scored against tests/mxfp4-stored-layout-oracle.hpp's reference_gemm
//     (ACT_Q8_1, the SAME activation representation the DPAS kernel
//     consumes) with the shared scorer (max_rel_violations, WOQ_MAX_REL_TOL
//     = 0.0258, DEFAULT_ABS_FLOOR = 0.01) -- 0 violations required.
//   - Boundary case: N=37 (not a multiple of the kernel's internal 16-row
//     N-tile), K=2880, M=8 -- exercises the partial-tile boundary check the
//     main sweep's N=2880 (=180*16, an exact multiple) never reaches.

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/mxfp4-stored-gemm.hpp"
#include "ggml.h"
#include "mxfp4-stored-layout-oracle.hpp"
#include "test-skip.h"

#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

ggml_backend_buffer_t alloc_tensor_buffer(ggml_backend_buffer_type_t buft,
                                          ggml_tensor *              tensor,
                                          ggml_backend_buffer_usage  usage) {
    const size_t          size   = ggml_backend_buft_get_alloc_size(buft, tensor);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, size);
    if (!buffer) {
        return nullptr;
    }
    ggml_backend_buffer_set_usage(buffer, usage);
    // Checked (llama.cpp-6f73 c-nvf1 nit 12): an unchecked failure here used
    // to surface later as a null tensor->data at the point of use, rather
    // than as the clean "failed to allocate a device buffer" FAIL the caller
    // already reports for every OTHER allocation failure in this function.
    if (ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer)) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buffer);
        return nullptr;
    }
    return buffer;
}

// Runs one (expert, M) GEMM on the GPU and returns the M*n_out output as
// doubles (empty on allocation/compute failure, with a printed reason).
// `soa_weight` is the expert's raw SOA MXFP4 weight buffer
// (build_soa_from_aos, mxfp4-stored-layout-oracle.hpp); `X` is M*K row-major
// f32 activations.
std::vector<double> run_gemm(ggml_backend_t               backend,
                             const std::vector<uint8_t> & soa_weight,
                             const std::vector<float> &   X,
                             int64_t                      M,
                             int64_t                      n_out,
                             int64_t                      n_k) {
    std::vector<double> empty;

    const ggml_sycl_mxfp4_stored_gemm::q8_1_activation_pack pack =
        ggml_sycl_mxfp4_stored_gemm::quantize_activations_q8_1(X.data(), M, n_k);

    ggml_init_params params = { 16 * 1024 * 1024, nullptr, true };
    ggml_context *   ctx    = ggml_init(params);
    if (!ctx) {
        std::printf("FAIL: ggml_init failed\n");
        ++g_failures;
        return empty;
    }

    ggml_tensor * t_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) soa_weight.size());
    ggml_tensor * t_act_qs = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) pack.qs.size());
    ggml_tensor * t_act_sc = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) pack.scales.size());
    ggml_tensor * t_dst    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, M * n_out);
    ggml_set_name(t_weight, "mxfp4_stored_gemm_soa_weight");
    ggml_set_name(t_act_qs, "mxfp4_stored_gemm_act_qs");
    ggml_set_name(t_act_sc, "mxfp4_stored_gemm_act_scales");
    ggml_set_name(t_dst, "mxfp4_stored_gemm_dst");

    ggml_backend_buffer_type_t dev_buft   = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_t      buf_weight = alloc_tensor_buffer(dev_buft, t_weight, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_t      buf_act_qs = alloc_tensor_buffer(dev_buft, t_act_qs, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_t      buf_act_sc = alloc_tensor_buffer(dev_buft, t_act_sc, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_t      buf_dst    = alloc_tensor_buffer(dev_buft, t_dst, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    if (!buf_weight || !buf_act_qs || !buf_act_sc || !buf_dst) {
        std::printf("FAIL: failed to allocate a device buffer\n");
        ++g_failures;
        if (buf_weight) {
            ggml_backend_buffer_free(buf_weight);
        }
        if (buf_act_qs) {
            ggml_backend_buffer_free(buf_act_qs);
        }
        if (buf_act_sc) {
            ggml_backend_buffer_free(buf_act_sc);
        }
        if (buf_dst) {
            ggml_backend_buffer_free(buf_dst);
        }
        ggml_free(ctx);
        return empty;
    }

    ggml_backend_tensor_set(t_weight, soa_weight.data(), 0, soa_weight.size());
    ggml_backend_tensor_set(t_act_qs, pack.qs.data(), 0, pack.qs.size());
    ggml_backend_tensor_set(t_act_sc, pack.scales.data(), 0, pack.scales.size() * sizeof(float));

    // ctx->stream() is the same queue production dispatch (and the sibling
    // GPU tests that go through ggml_backend_graph_compute) submits on --
    // the correct USM/context owner for the buffers allocated above, and
    // (llama.cpp-yke2, S7: unconditional sycl::property::queue::enable_profiling
    // in dpct's create_queue_impl) now reliably carries
    // sycl::property::queue::enable_profiling in every process, so it is
    // used directly for both the uploads above and every kernel launch
    // below -- no separate profiling-only queue construction, and no extra
    // wait to order it against the uploads, is needed any more.
    auto *        sycl_ctx = static_cast<ggml_backend_sycl_context *>(backend->context);
    sycl::queue & q        = *sycl_ctx->stream();

    const bool profiling_requested = std::getenv("GGML_SYCL_KERNEL_PROFILE") != nullptr;

    // Bytes moved per call, for deriving per-launch bandwidth: the whole
    // expert's SOA buffer (17 bytes per 32-element block: 16 nibble-packed +
    // 1 E8M0 scale byte) plus the M x n_k int8 activation codes, the
    // M x (n_k/32) f32 activation scales (llama.cpp-6f73 c-gngd -- omitted
    // before, closing the 4,521,600 vs 4,524,480 gap against the kernel's
    // own profile_label.bytes), plus the M x n_out f32 output. Computed here
    // (function scope, not inside either `if (profiling_requested)` block
    // below) so BOTH the queue-diag block's own bytes_moved print and the
    // later per-launch-bandwidth block (llama.cpp-kcya) can see it -- it is
    // pure arithmetic with no side effects, so computing it unconditionally
    // costs nothing when profiling is not requested.
    //
    // Per-launch bandwidth is THIS figure divided by the mean device time
    // for one launch (mean_ns), NOT the kernel-profiler CSV's aggregate
    // `bytes` column divided by mean_ns: sycl-kernel-profiler.cpp's
    // aggregate accumulates `aggregate.bytes += label.bytes` once per
    // recorded launch, so that column is the SUM over `count` launches.
    // Dividing the summed column by mean_ns overstates bandwidth by a
    // factor of `count` (llama.cpp-6f73 c-irug: a naive read once gave 34%
    // of peak where the true per-launch figure was 8.5%).
    const double bytes_moved = (double) n_out * (double) n_k * 17.0 / 32.0 + (double) M * (double) n_k +
                               (double) M * (double) (n_k / 32) * 4.0 + (double) M * (double) n_out * 4.0;

    if (profiling_requested) {
        // Confirms llama.cpp-yke2's S7 (unconditional enable_profiling in
        // dpct's create_queue_impl) actually reaches this queue: this used
        // to fail under a single-GPU selector before S7 landed (llama.cpp-6f73,
        // round 3), which is why this test built its own profiling-only
        // queue rather than asserting on `q` directly -- that workaround is
        // gone now that the assert below is expected to hold unconditionally.
        const bool has_profiling = q.has_property<sycl::property::queue::enable_profiling>();
        const bool is_in_order   = q.has_property<sycl::property::queue::in_order>();
        std::printf("  queue diag: has_property(enable_profiling)=%d has_property(in_order)=%d addr=%p\n",
                    (int) has_profiling, (int) is_in_order, (void *) &q);
        GGML_ASSERT(has_profiling);

        std::printf("  bytes_moved M=%lld n_out=%lld n_k=%lld: %.0f\n", (long long) M, (long long) n_out,
                    (long long) n_k, bytes_moved);
    }

    sycl::event event = ggml_sycl_mxfp4_stored_gemm::ggml_sycl_mxfp4_soa_gemm_dpas(
        q, t_weight->data, static_cast<const int8_t *>(t_act_qs->data), static_cast<const float *>(t_act_sc->data),
        static_cast<float *>(t_dst->data), (int) M, (int) n_out, (int) n_k, {});

    if (profiling_requested) {
        // Diagnostic for llama.cpp-6f73 (profiling investigation, round 3):
        // query the SAME event the profiler records, the same way the
        // profiler does (wait_and_throw, then command_start/command_end),
        // and print either the raw timestamps or the exception -- this is
        // exactly the condition sycl-kernel-profiler.cpp's
        // drain_pending_events hits, reproduced by hand outside the profiler
        // machinery.
        std::printf("  event diag: backend=%d (level_zero=%d)\n", (int) event.get_backend(),
                    (int) sycl::backend::ext_oneapi_level_zero);
        try {
            const auto status = event.get_info<sycl::info::event::command_execution_status>();
            std::printf("  event diag: execution_status=%d (complete=%d)\n", (int) status,
                        (int) sycl::info::event_command_status::complete);
        } catch (const sycl::exception & e) {
            std::printf("  event diag: get_info(command_execution_status) EXCEPTION: %s\n", e.what());
        }
        try {
            event.wait_and_throw();
            const uint64_t start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
            const uint64_t end   = event.get_profiling_info<sycl::info::event_profiling::command_end>();
            std::printf("  event diag: command_start=%llu command_end=%llu (end>=start: %d)\n",
                        (unsigned long long) start, (unsigned long long) end, (int) (end >= start));
        } catch (const sycl::exception & e) {
            std::printf("  event diag: wait_and_throw/get_profiling_info EXCEPTION: %s (code=%d)\n", e.what(),
                        (int) e.code().value());
        } catch (const std::exception & e) {
            std::printf("  event diag: wait_and_throw/get_profiling_info std::exception: %s\n", e.what());
        }

        // llama.cpp-kcya: per-launch bandwidth, timed independently of the
        // event-diag block above. That block queries the SAME event the
        // profiler records -- command_start/command_end on `event` alone --
        // which is fine for a single-kernel dispatch but NOT for this
        // kernel any more: ggml_sycl_mxfp4_soa_gemm_dpas is now a
        // two-kernel (partial+combine) dispatch whenever ksplit > 1
        // (mxfp4-stored-gemm.cpp), and a SYCL event's own profiling info
        // covers only the ONE kernel it was returned from -- here, only the
        // tiny combine kernel, not the much larger partial kernel that ran
        // before it. Using that timestamp pair for bandwidth would silently
        // UNDERSTATE elapsed time and OVERSTATE GB/s, exactly the class of
        // trap this file's own bytes_moved comment above warns about for
        // the CSV's aggregate `bytes` column. Host wall-clock around the
        // WHOLE call -- including a wait on the returned event, which
        // transitively waits for every kernel in the dependency chain, not
        // just the last one -- does not have that blind spot, so that is
        // what this print uses. It re-issues the SAME deterministic call
        // (same inputs, same device buffers) additional times purely for
        // timing; the output `run_gemm` ultimately returns and scores is
        // read once, after this loop, so these extra launches cannot change
        // the correctness verdict, only overwrite t_dst with the identical
        // result.
        constexpr int       BW_WARMUP = 3;
        constexpr int       BW_ITERS  = 20;
        std::vector<double> call_ns;
        call_ns.reserve(BW_ITERS);
        for (int it = 0; it < BW_WARMUP + BW_ITERS; ++it) {
            const auto  t0       = std::chrono::steady_clock::now();
            sycl::event bw_event = ggml_sycl_mxfp4_stored_gemm::ggml_sycl_mxfp4_soa_gemm_dpas(
                q, t_weight->data, static_cast<const int8_t *>(t_act_qs->data),
                static_cast<const float *>(t_act_sc->data), static_cast<float *>(t_dst->data), (int) M, (int) n_out,
                (int) n_k, {});
            bw_event.wait();
            const auto t1 = std::chrono::steady_clock::now();
            if (it >= BW_WARMUP) {
                call_ns.push_back((double) std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
            }
        }
        double mean_ns = 0.0;
        for (double v : call_ns) {
            mean_ns += v;
        }
        mean_ns /= (double) call_ns.size();
        const double bw_gbps = bytes_moved / mean_ns;  // (bytes/ns) == GB/s
        std::printf(
            "  per_launch_bandwidth M=%lld n_out=%lld n_k=%lld bytes_moved=%.0f mean_ns=%.1f GBps=%.3f (n=%zu, "
            "host wall-clock incl. every kernel in the dispatch -- NOT the event-diag "
            "command_start/command_end above)\n",
            (long long) M, (long long) n_out, (long long) n_k, bytes_moved, mean_ns, bw_gbps, call_ns.size());
    } else {
        event.wait();
    }

    std::vector<float> gpu_out((size_t) (M * n_out));
    ggml_backend_tensor_get(t_dst, gpu_out.data(), 0, gpu_out.size() * sizeof(float));

    std::vector<double> out((size_t) (M * n_out));
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = (double) gpu_out[i];
    }

    ggml_backend_buffer_free(buf_weight);
    ggml_backend_buffer_free(buf_act_qs);
    ggml_backend_buffer_free(buf_act_sc);
    ggml_backend_buffer_free(buf_dst);
    ggml_free(ctx);
    return out;
}

void score_and_report(const std::string & label, const std::vector<double> & gpu_out, const std::vector<double> & ref) {
    const Score s  = max_rel_violations(gpu_out, ref, WOQ_MAX_REL_TOL);
    const bool  ok = s.violations == 0;
    std::printf("%s [%s]: violations=%lld/%zu max_rel=%.6e max_abs_diff=%.6e (tol rel=%.4f abs_floor=%.2f)\n",
                ok ? "OK" : "FAIL", label.c_str(), (long long) s.violations, ref.size(), s.max_rel, s.max_abs_diff,
                WOQ_MAX_REL_TOL, DEFAULT_ABS_FLOOR);
    if (!ok) {
        ++g_failures;
    }
}

// One expert's random MXFP4 weight matrix, materialized both as SOA bytes
// (what the kernel reads) and as a decoded float matrix (what the oracle's
// reference_gemm consumes) -- the SAME two artifacts the G3 oracle's own
// cross-decoder check builds and validates, reused here rather than
// re-derived.
struct RandomExpert {
    std::vector<uint8_t> soa;
    std::vector<float>   decoded;  // n_out x n_k, row-major
};

RandomExpert make_random_expert(int64_t n_out, int64_t n_k, uint32_t seed) {
    std::mt19937                          rng(seed);
    std::uniform_real_distribution<float> wdist(-2.0f, 2.0f);
    std::vector<float>                    weight((size_t) (n_out * n_k));
    for (float & v : weight) {
        v = wdist(rng);
    }

    const int64_t            n_k_blocks = n_k / XMX_K;
    std::vector<block_mxfp4> aos((size_t) (n_out * n_k_blocks));
    ggml_quantize_chunk(GGML_TYPE_MXFP4, weight.data(), aos.data(), 0, n_out, n_k, nullptr);

    RandomExpert e;
    e.soa     = build_soa_from_aos(aos, n_out, n_k);
    e.decoded = decode_soa_mxfp4(e.soa.data(), n_out, n_k);
    return e;
}

void run_case(ggml_backend_t       backend,
              const RandomExpert & expert,
              int64_t              M,
              int64_t              n_out,
              int64_t              n_k,
              uint32_t             act_seed,
              const char *         label) {
    std::mt19937                          rng(act_seed);
    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);
    std::vector<float>                    X((size_t) (M * n_k));
    for (float & v : X) {
        v = xdist(rng);
    }

    const std::vector<double> ref     = reference_gemm(X, expert.decoded, M, n_out, n_k, Activation::ACT_Q8_1);
    const std::vector<double> gpu_out = run_gemm(backend, expert.soa, X, M, n_out, n_k);
    if (gpu_out.empty()) {
        // run_gemm already recorded a FAIL and printed the reason.
        return;
    }
    score_and_report(std::string(label) + " M=" + std::to_string(M) + " n_out=" + std::to_string(n_out) +
                         " n_k=" + std::to_string(n_k),
                     gpu_out, ref);
}

}  // namespace

int main(int, char ** argv) {
    // ctest supplies ONEAPI_DEVICE_SELECTOR via the registration's ENVIRONMENT. Bare invocation
    // falls back to the B50, but setenv() here is too late: libccl's static initializer constructs
    // a sycl::event at load, which makes libsycl memoize the selector before main() runs
    // (llama.cpp-2x3m). Re-exec so the child starts with it set (llama.cpp-403s: unpinned, the
    // iGPU's 231 GB "VRAM" is claimed).
    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        if (setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:1", 1) == 0) {
            execv("/proc/self/exe", argv);
        }
        std::fprintf(stderr, "warning: re-exec failed (%s); run with ONEAPI_DEVICE_SELECTOR=level_zero:1 set\n",
                     std::strerror(errno));
    }

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        std::printf("SKIP: no SYCL GPU device available\n");
        return LLAMA_TEST_EXIT_SKIP;
    }

    constexpr int64_t N = 2880, K = 2880;  // GPT-OSS expert shape
    constexpr int     M_VALUES[] = { 1, 2, 4, 8 };
    constexpr int     N_EXPERTS  = 4;

    for (int e = 0; e < N_EXPERTS; ++e) {
        std::printf("== expert %d (N=%lld K=%lld) ==\n", e, (long long) N, (long long) K);
        const RandomExpert expert = make_random_expert(N, K, 0x6d786670u + (uint32_t) e);  // "mxfp" + e
        for (int M : M_VALUES) {
            run_case(backend, expert, M, N, K, 0x61637431u + (uint32_t) (e * 16 + M), "full-N");
        }
    }

    // Boundary case: N=37 is not a multiple of the kernel's 16-row N-tile
    // (2880 = 180*16 is, so the main sweep above never exercises the
    // partial-tile boundary check).
    std::printf("== boundary expert (N=37 K=%lld) ==\n", (long long) K);
    const RandomExpert boundary_expert = make_random_expert(37, K, 0x62646479u);  // "bddy"
    run_case(backend, boundary_expert, 8, 37, K, 0x62646461u, "boundary-N37");

    ggml_backend_free(backend);

    if (g_failures) {
        std::printf("FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("PASS: mxfp4 stored-SOA small-M full-N GEMM numerics\n");
    return 0;
}
