// GPU test of the batched F32 MUL_MAT route: a batched F32 x F32 MUL_MAT on dense, device-resident activations is one
// oneDNN matmul, where the per-plane loop of ggml_sycl_op_mul_mat submits one GEMM per plane.
//
// Each case computes one MUL_MAT graph directly on the SYCL backend, every tensor in one SYCL compute buffer (not
// through ggml_backend_sched, which would give the graph inputs to the CPU). The PRIVATE_TESTING scratchpad seam counts
// the "dnnl_gemm" site, which every gemm() call passes once, and is the witness of how many GEMMs ran:
//
//   taken     the route run must count calls == engaged == 1, declined == 0: one GEMM for the whole batch. The case is
//             then computed again with a decline injected on that one call (ggml_sycl_test_inject_scratchpad_decline
//             "dnnl_gemm", 1): the route reads false, writes nothing, and the orchestrator's per-plane loop computes the
//             op, so that run counts calls == 1 + P, declined == 1, engaged == P for P planes. Its output is the loop's
//             result for the same operands, the second oracle;
//   declined  a node the predicate refuses (a partial broadcast, a non-contiguous src1) counts calls == engaged == P.
//
// A taken case's 2-D src0 against a src1 batched in dim 2 is one plain GEMM over every column, batch 1 on both sides:
// its profiler row is the unbatched one, so the count above, not the label, is its witness.
//
// Tolerances. Every comparison is scaled by S = sum over k of |src0[m][k] * src1[n][k]| for its element, computed in
// double with the reference: err = |got - ref| / S. The inputs are floats and the reference sums their exact products
// in double, so the reference's own error is negligible.
//   The route against the reference: err <= (K + 1) * 2^-24. The route asks gemm() for strict float math, so its
//   primitive computes in F32, and an F32 dot product of length K, in any summation order, is within gamma_K ~ K * u
//   (u = 2^-24) of S; the extra u is the result's rounding. A failure means the route did not compute in F32. At the
//   largest K (2304) the bound is 1.4e-4, still five times under the error f16 math was measured to give there.
//   The loop against the reference: err <= 4e-3. The per-plane loop still runs under the build's float math, which
//   in a GGML_SYCL_F16 build is fpmath_mode::f16 ("implicit f32->f16 conversions allowed", dnnl_common_types.h):
//   oneDNN may round each input to f16 (unit roundoff 2^-11), so a product is off by about 2^-10 of |a*b| and a dot
//   product by about 1e-3 of S; the bound is four times that. Whether oneDNN uses the licence depends on the shape
//   and the card (one Inkling shape was measured in F32 on the B70 and in f16 on the B50), so the loop's figure may
//   be F32 or f16.
//   The route against the loop's output: err <= 8e-3, loose for the same reason.
// The measured maxima of all three are printed per case.
//
// The Inkling cases are its three node kinds at the model's dimensions (n_embd 2304, 3 mix queries, at most 4 earlier
// blocks, n_relative 16, n_head 24, the sliding window's bias table 1024 + 1 rows) at T = 2, 179, 512 and 2048. At
// T = 2048 the largest is the bias table: 201,523,200 bytes of dst plus 3,211,328 of operands on the device, run like
// the others. A primitive oneDNN refuses to create for a shape throws from gemm(); each case prints a line before it
// runs, so a log that ends there names the shape.
//
// Per taken case the test also prints the scratchpad the route's primitive asks for: the oneDNN primitive cache is
// cleared before the route run, so afterwards it holds that one primitive and get_max_scratchpad_size() is its size.
//
// Usage (the registration supplies the env; a bare run needs it too):
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 GGML_SYCL_DISABLE_GRAPH=1 ./build/bin/test-sycl-mul-mat-f32-batched-device

#include "gemm.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-sycl-test.hpp"
#include "ggml-sycl.h"
#include "ggml.h"
#include "sycl-selector-fallback.hpp"
#include "test-skip.h"  // LLAMA_TEST_EXIT_SKIP: the one definition of "77 means skip"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#if !defined(GGML_SYCL_PRIVATE_TESTING)
#    error "this test requires GGML_SYCL_PRIVATE_TESTING (link against ggml-sycl-private-fixtures)"
#endif

namespace {

constexpr const char * SITE         = "dnnl_gemm";
constexpr double       TOL_REF      = 4e-3;  // the loop: the build's float math, f16 in a GGML_SYCL_F16 build
constexpr double       TOL_ROUTE_VS = 8e-3;

// The route against the reference: strict F32 math over a dot product of length k.
double tol_route_f32(int64_t k) {
    return (double) (k + 1) * std::ldexp(1.0, -24);
}

struct mm_case {
    std::string name;
    int64_t     a[4];       // src0 ne
    int64_t     b[4];       // src1 ne
    bool        permute_b;  // src1 is a [K, ne12, ne11] tensor viewed as [K, ne11, ne12]: not contiguous
    bool        taken;      // the route takes it
};

struct run_result {
    bool               ok = false;
    std::vector<float> out;
    uint64_t           calls = 0, declined = 0, engaged = 0;
    size_t             cache_entries = 0, scratchpad_bytes = 0;
};

float input_value(int64_t i, int64_t salt) {
    // a deterministic spread over [-1, 1) with no short period
    const uint64_t h = (uint64_t) (i * 2654435761LL + salt * 40503LL) % 1000003ULL;
    return -1.0f + 2.0f * (float) h / 1000003.0f;
}

// The logical value of src1 at (k, n, p), whatever its storage.
float b_value(const mm_case & c, int64_t k, int64_t n, int64_t p) {
    return input_value(k + c.b[0] * (n + c.b[1] * p), 7);
}

run_result run_case(ggml_backend_t backend, const mm_case & c, int inject) {
    run_result res;
    ggml_sycl_test_scratchpad_sites_reset();
    if (inject != 0 && !ggml_sycl_test_inject_scratchpad_decline(SITE, inject)) {
        fprintf(stderr, "FAIL: %s: the seam does not know the site %s\n", c.name.c_str(), SITE);
        return res;
    }
    get_dnnl_primitive_cache().clear();

    const size_t         max_nodes = 8;
    const size_t         mem_size  = ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false);
    std::vector<uint8_t> mem_buffer(mem_size);
    ggml_init_params iparams = { /*.mem_size   =*/mem_size, /*.mem_buffer =*/mem_buffer.data(), /*.no_alloc   =*/true };
    ggml_context *   ctx     = ggml_init(iparams);
    if (!ctx) {
        fprintf(stderr, "FAIL: %s: ggml_init failed\n", c.name.c_str());
        return res;
    }
    ggml_tensor * a       = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.a[0], c.a[1], c.a[2], c.a[3]);
    ggml_tensor * b_store = c.permute_b ? ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.b[0], c.b[2], c.b[1], c.b[3]) :
                                          ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.b[0], c.b[1], c.b[2], c.b[3]);
    ggml_tensor * b       = c.permute_b ? ggml_permute(ctx, b_store, 0, 2, 1, 3) : b_store;
    ggml_set_input(a);
    ggml_set_input(b_store);
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, max_nodes, false);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_sycl_buffer_type(0));
    if (!buf) {
        fprintf(stderr, "FAIL: %s: could not allocate the graph's tensors on the SYCL device\n", c.name.c_str());
        ggml_free(ctx);
        return res;
    }
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    std::vector<float> av((size_t) ggml_nelements(a));
    for (size_t i = 0; i < av.size(); ++i) {
        av[i] = input_value((int64_t) i, 3);
    }
    ggml_backend_tensor_set(a, av.data(), 0, av.size() * sizeof(float));
    std::vector<float> bv((size_t) ggml_nelements(b_store));
    for (int64_t p = 0; p < c.b[2] * c.b[3]; ++p) {
        for (int64_t n = 0; n < c.b[1]; ++n) {
            for (int64_t k = 0; k < c.b[0]; ++k) {
                // the store is [K, ne11, ne12] as is, or [K, ne12, ne11] for the permuted view
                const int64_t at = c.permute_b ? k + c.b[0] * ((p % c.b[2]) + c.b[2] * (n + c.b[1] * (p / c.b[2]))) :
                                                 k + c.b[0] * (n + c.b[1] * p);
                bv[(size_t) at]  = b_value(c, k, n, p);
            }
        }
    }
    ggml_backend_tensor_set(b_store, bv.data(), 0, bv.size() * sizeof(float));

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL: %s: graph compute returned %d\n", c.name.c_str(), (int) status);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return res;
    }
    // A recorded graph would make the route decline (it does not create a primitive inside a recording): void.
    if (ggml_sycl::test_backend_has_exec_graph(backend)) {
        fprintf(stderr,
                "FAIL: %s: the backend recorded an executable graph; the case is void (run with "
                "GGML_SYCL_DISABLE_GRAPH=1)\n",
                c.name.c_str());
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return res;
    }
    res.out.resize((size_t) ggml_nelements(out));
    ggml_backend_tensor_get(out, res.out.data(), 0, res.out.size() * sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    if (!ggml_sycl_test_scratchpad_site_counts(SITE, &res.calls, &res.declined, &res.engaged)) {
        fprintf(stderr, "FAIL: %s: the seam does not know the site %s\n", c.name.c_str(), SITE);
        return res;
    }
    ggml_sycl_test_inject_scratchpad_decline(SITE, 0);
    res.cache_entries    = get_dnnl_primitive_cache().size();
    res.scratchpad_bytes = get_dnnl_primitive_cache().get_max_scratchpad_size();
    res.ok               = true;
    return res;
}

struct errors {
    double ref_route = 0.0, ref_loop = 0.0, route_vs_loop = 0.0, abs_route = 0.0;
    bool   finite = true;
};

// One pass over dst in double: the reference and its scale S for each element, compared with each run's output.
// `loop` may be empty (a declined case has one run).
errors compare(const mm_case & c, const std::vector<float> & route, const std::vector<float> & loop) {
    errors        e;
    const int64_t K = c.a[0], M = c.a[1], N = c.b[1];
    const int64_t r2 = c.b[2] / c.a[2], r3 = c.b[3] / c.a[3];
    for (int64_t i3 = 0; i3 < c.b[3]; ++i3) {
        for (int64_t i2 = 0; i2 < c.b[2]; ++i2) {
            const int64_t a_plane = (i2 / r2) + c.a[2] * (i3 / r3);
            const int64_t p       = i2 + c.b[2] * i3;
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t m = 0; m < M; ++m) {
                    double ref = 0.0, scale = 0.0;
                    for (int64_t k = 0; k < K; ++k) {
                        const double x = input_value(k + K * (m + M * a_plane), 3);
                        const double y = b_value(c, k, n, p);
                        ref += x * y;
                        scale += std::fabs(x * y);
                    }
                    scale          = scale > 0.0 ? scale : 1.0;
                    const size_t i = (size_t) (m + M * (n + N * p));
                    const double r = route[i];
                    e.finite       = e.finite && std::isfinite(r);
                    e.ref_route    = std::fmax(e.ref_route, std::fabs(r - ref) / scale);
                    e.abs_route    = std::fmax(e.abs_route, std::fabs(r - ref));
                    if (!loop.empty()) {
                        const double l  = loop[i];
                        e.finite        = e.finite && std::isfinite(l);
                        e.ref_loop      = std::fmax(e.ref_loop, std::fabs(l - ref) / scale);
                        e.route_vs_loop = std::fmax(e.route_vs_loop, std::fabs(r - l) / scale);
                    }
                }
            }
        }
    }
    return e;
}

mm_case make_case(const std::string &            name,
                  const std::array<int64_t, 4> & a,
                  const std::array<int64_t, 4> & b,
                  bool                           permute_b,
                  bool                           taken) {
    mm_case c;
    c.name = name;
    for (int i = 0; i < 4; ++i) {
        c.a[i] = a[i];
        c.b[i] = b[i];
    }
    c.permute_b = permute_b;
    c.taken     = taken;
    return c;
}

std::vector<mm_case> make_cases() {
    std::vector<mm_case> cases;
    cases.push_back(make_case("equal batch in dim 2", { 64, 24, 6, 1 }, { 64, 10, 6, 1 }, false, true));
    cases.push_back(make_case("equal batch in dims 2 and 3", { 64, 24, 3, 2 }, { 64, 10, 3, 2 }, false, true));
    cases.push_back(make_case("2-D src0 broadcast over dim 2", { 64, 24, 1, 1 }, { 64, 10, 6, 1 }, false, true));
    for (const int64_t T : { 2, 179, 512, 2048 }) {
        const std::string t = " T=" + std::to_string(T);
        cases.push_back(make_case("Inkling mix scores" + t, { 2304, 5, T, 1 }, { 2304, 3, T, 1 }, false, true));
        cases.push_back(make_case("Inkling mix output" + t, { 5, 2304, T, 1 }, { 5, 3, T, 1 }, false, true));
        cases.push_back(make_case("Inkling bias table" + t, { 16, 1025, 1, 1 }, { 16, 24, T, 1 }, false, true));
    }
    cases.push_back(make_case("declined: partial broadcast", { 64, 24, 2, 1 }, { 64, 10, 6, 1 }, false, false));
    cases.push_back(make_case("declined: non-contiguous src1", { 64, 24, 6, 1 }, { 64, 10, 6, 1 }, true, false));
    return cases;
}

}  // namespace

int main(int, char ** argv) {
    // See tests/sycl-selector-fallback.hpp: a plain setenv() in main() is too late for libccl's static initializer.
    sycl_test_selector_fallback(argv, "level_zero:1");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        fprintf(stderr,
                "SKIP: no SYCL GPU device available -- NO DEVICE WORK WAS PERFORMED.\n"
                "      source /opt/intel/oneapi/setvars.sh --force and re-run.\n");
        return LLAMA_TEST_EXIT_SKIP;
    }
    char description[256] = {};
    ggml_backend_sycl_get_device_description(0, description, sizeof(description));
    const char * selector = getenv("ONEAPI_DEVICE_SELECTOR");
    printf("device 0: %s (ONEAPI_DEVICE_SELECTOR=%s)\n", description, selector ? selector : "(unset)");
    printf(
        "tolerances, scaled by sum|a*b|: route against the double reference (K + 1) * 2^-24 (F32), loop against it "
        "%.0e, route against loop %.0e\n",
        TOL_REF, TOL_ROUTE_VS);
    fflush(stdout);

    bool ok = true;
    for (const mm_case & c : make_cases()) {
        const int64_t P = c.b[2] * c.b[3];
        printf("case %-34s src0=[%lld,%lld,%lld,%lld] src1=[%lld,%lld,%lld,%lld]%s P=%lld: running\n", c.name.c_str(),
               (long long) c.a[0], (long long) c.a[1], (long long) c.a[2], (long long) c.a[3], (long long) c.b[0],
               (long long) c.b[1], (long long) c.b[2], (long long) c.b[3], c.permute_b ? " (permuted view)" : "",
               (long long) P);
        fflush(stdout);

        bool             case_ok = true;
        const run_result first   = run_case(backend, c, 0);
        if (!first.ok) {
            ok = false;
            continue;
        }
        run_result fallen;
        if (c.taken) {
            fallen = run_case(backend, c, 1);
            if (!fallen.ok) {
                ok = false;
                continue;
            }
        }
        const errors e = compare(c, first.out, fallen.out);

        if (c.taken) {
            printf(
                "  route: calls=%llu declined=%llu engaged=%llu | primitive cache entries=%zu scratchpad=%zu bytes\n",
                (unsigned long long) first.calls, (unsigned long long) first.declined,
                (unsigned long long) first.engaged, first.cache_entries, first.scratchpad_bytes);
            printf("  injected decline: calls=%llu declined=%llu engaged=%llu\n", (unsigned long long) fallen.calls,
                   (unsigned long long) fallen.declined, (unsigned long long) fallen.engaged);
            printf(
                "  max err/S: route vs ref %.3e (F32 bound %.3e), loop vs ref %.3e, route vs loop %.3e (max abs route "
                "err %.3e)\n",
                e.ref_route, tol_route_f32(c.a[0]), e.ref_loop, e.route_vs_loop, e.abs_route);
            if (first.calls != 1 || first.engaged != 1 || first.declined != 0) {
                fprintf(stderr, "FAIL: %s: the route did not make exactly one GEMM submission\n", c.name.c_str());
                case_ok = false;
            }
            if (fallen.calls != 1 + (uint64_t) P || fallen.declined != 1 || fallen.engaged != (uint64_t) P) {
                fprintf(
                    stderr,
                    "FAIL: %s: after the injected decline the loop did not make one GEMM per plane (want calls=%lld "
                    "declined=1 engaged=%lld)\n",
                    c.name.c_str(), (long long) (1 + P), (long long) P);
                case_ok = false;
            }
            if (first.cache_entries != 1) {
                printf(
                    "  note: the primitive cache holds %zu entries after the route run; the scratchpad figure is "
                    "their maximum\n",
                    first.cache_entries);
            }
            if (!(e.ref_loop <= TOL_REF)) {
                fprintf(stderr, "FAIL: %s: the loop differs from the double reference (%.3e)\n", c.name.c_str(),
                        e.ref_loop);
                case_ok = false;
            }
            if (!(e.route_vs_loop <= TOL_ROUTE_VS)) {
                fprintf(stderr, "FAIL: %s: the route differs from the loop (%.3e)\n", c.name.c_str(), e.route_vs_loop);
                case_ok = false;
            }
        } else {
            printf("  loop: calls=%llu declined=%llu engaged=%llu\n", (unsigned long long) first.calls,
                   (unsigned long long) first.declined, (unsigned long long) first.engaged);
            printf("  max err/S: loop vs ref %.3e (max abs err %.3e)\n", e.ref_route, e.abs_route);
            if (first.calls != (uint64_t) P || first.engaged != (uint64_t) P || first.declined != 0) {
                fprintf(stderr, "FAIL: %s: a declined node did not run one GEMM per plane (want %lld)\n",
                        c.name.c_str(), (long long) P);
                case_ok = false;
            }
        }
        if (!e.finite) {
            fprintf(stderr, "FAIL: %s: a non-finite output\n", c.name.c_str());
            case_ok = false;
        }
        // A taken case's first run is the route, held to F32; a declined case's is the loop, held to the build's math.
        const double tol_first = c.taken ? tol_route_f32(c.a[0]) : TOL_REF;
        if (!(e.ref_route <= tol_first)) {
            fprintf(stderr, "FAIL: %s: the %s differs from the double reference (%.3e > %.3e)\n", c.name.c_str(),
                    c.taken ? "route" : "loop", e.ref_route, tol_first);
            case_ok = false;
        }
        printf("  %s\n", case_ok ? "ok" : "FAILED");
        fflush(stdout);
        ok = ok && case_ok;
    }

    ggml_backend_free(backend);
    if (!ok) {
        fprintf(stderr, "test-sycl-mul-mat-f32-batched-device: FAIL\n");
        return 1;
    }
    printf("test-sycl-mul-mat-f32-batched-device: PASS\n");
    return 0;
}
