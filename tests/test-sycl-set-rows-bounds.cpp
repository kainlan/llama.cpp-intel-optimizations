// Contract gate for SET_ROWS destination-row bounds on the SYCL backend.
//
// SET_ROWS' index contract is that every entry of src1 names a real destination
// row. The CPU backend enforces it with GGML_ASSERT(i1 >= 0 && i1 < ne1)
// (ggml/src/ggml-cpu/ops.cpp:5073); the SYCL kernels used the value unchecked, so
// an out-of-range index scattered a write outside the destination tensor
// (llama.cpp-fcyv, llama.cpp-nhip).
//
// The destination here is a VIEW of a larger parent, with guard rows below and
// above it. Every write an unguarded kernel performs therefore still lands inside
// memory this test owns: it is a real bounds test that cannot itself go out of
// bounds, and cannot crash. The guard rows carry a per-row poison byte, so a
// misplaced row write is visible as a changed row.
//
// Exactly one row must change: the one the single in-range index names. That row
// is the positive control -- it proves the kernel ran and wrote, so an unchanged
// guard row means the bound held rather than that nothing happened.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "SKIP: built without GGML_USE_SYCL; this gate proves NOTHING about SET_ROWS bounds.\n");
    return 77;
}
#else

#    include "ggml-sycl.h"

// Layout shared by every case: guard rows below the view, the view, guard rows above.
static const int64_t PAD_ROWS   = 2;
static const int64_t VIEW_ROWS  = 4;
static const int64_t TAIL_ROWS  = 2;
static const int64_t TOTAL_ROWS = PAD_ROWS + VIEW_ROWS + TAIL_ROWS;

// One in-range index and two out-of-range ones, in both directions. Unguarded,
// index -1 lands on parent row PAD_ROWS-1 and index VIEW_ROWS+1 on parent row
// PAD_ROWS+VIEW_ROWS+1 -- both inside the parent, neither inside the view.
static const int64_t VALID_VIEW_ROW = 1;
static const int64_t IDX_VALUES[3]  = { VALID_VIEW_ROW, -1, VIEW_ROWS + 1 };
static const int64_t N_TOKENS       = 3;

static unsigned char poison_byte(int64_t row) {
    return (unsigned char) (0xA0 + row);
}

static bool run_case(ggml_backend_t backend, const char * label, ggml_type dst_type, ggml_type idx_type, int64_t nc) {
    printf("\n=== %s: dst=%s idx=%s ne0=%lld ===\n", label, ggml_type_name(dst_type), ggml_type_name(idx_type),
           (long long) nc);

    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * 32 + ggml_graph_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;

    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        printf("  FAIL: ggml_init returned null\n");
        return false;
    }

    // Creation order matters: ggml_backend_alloc_ctx_tensors walks the context in
    // order and initialises a view only after its parent has been allocated.
    ggml_tensor * parent = ggml_new_tensor_2d(ctx, dst_type, nc, TOTAL_ROWS);
    ggml_tensor * src    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nc, N_TOKENS);
    ggml_tensor * idx    = ggml_new_tensor_1d(ctx, idx_type, N_TOKENS);
    ggml_set_name(parent, "parent");
    ggml_set_name(src, "src");
    ggml_set_name(idx, "idx");

    const size_t  row_bytes = parent->nb[1];
    ggml_tensor * view      = ggml_view_2d(ctx, parent, nc, VIEW_ROWS, row_bytes, PAD_ROWS * row_bytes);
    ggml_set_name(view, "view");

    ggml_tensor * out = ggml_set_rows(ctx, view, src, idx);
    ggml_set_name(out, "out");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        printf("  FAIL: ggml_backend_alloc_ctx_tensors returned null\n");
        ggml_free(ctx);
        return false;
    }

    const size_t               parent_bytes = ggml_nbytes(parent);
    std::vector<unsigned char> poison(parent_bytes);
    for (int64_t r = 0; r < TOTAL_ROWS; ++r) {
        memset(poison.data() + r * row_bytes, poison_byte(r), row_bytes);
    }
    ggml_backend_tensor_set(parent, poison.data(), 0, parent_bytes);

    std::vector<float> src_data((size_t) nc * N_TOKENS);
    for (int64_t t = 0; t < N_TOKENS; ++t) {
        for (int64_t i = 0; i < nc; ++i) {
            src_data[(size_t) t * nc + i] = (float) (1000 * (t + 1) + i);
        }
    }
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));

    if (idx_type == GGML_TYPE_I64) {
        int64_t v[N_TOKENS];
        for (int64_t t = 0; t < N_TOKENS; ++t) {
            v[t] = IDX_VALUES[t];
        }
        ggml_backend_tensor_set(idx, v, 0, sizeof(v));
    } else {
        int32_t v[N_TOKENS];
        for (int64_t t = 0; t < N_TOKENS; ++t) {
            v[t] = (int32_t) IDX_VALUES[t];
        }
        ggml_backend_tensor_set(idx, v, 0, sizeof(v));
    }

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        printf("  FAIL: ggml_backend_graph_compute returned %d\n", (int) status);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    std::vector<unsigned char> after(parent_bytes);
    ggml_backend_tensor_get(parent, after.data(), 0, parent_bytes);

    const int64_t expect_written = PAD_ROWS + VALID_VIEW_ROW;

    bool ok = true;
    for (int64_t r = 0; r < TOTAL_ROWS; ++r) {
        const bool changed = memcmp(after.data() + r * row_bytes, poison.data() + r * row_bytes, row_bytes) != 0;
        if (r == expect_written) {
            if (!changed) {
                printf(
                    "  FAIL: parent row %lld was not written -- the in-range index did nothing, so an\n"
                    "        unchanged guard row would prove nothing about the bound\n",
                    (long long) r);
                ok = false;
            }
            continue;
        }
        if (changed) {
            const char * where = (r < PAD_ROWS) ? "below" : (r >= PAD_ROWS + VIEW_ROWS ? "above" : "inside");
            printf("  FAIL: SET_ROWS wrote outside the destination view: parent row %lld (%s the view) changed\n",
                   (long long) r, where);
            ok = false;
        }
    }

    if (ok) {
        printf("  OK: only parent row %lld changed; both out-of-range indices were dropped\n",
               (long long) expect_written);
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

int main() {
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        fprintf(stderr, "SKIP: no SYCL device; this gate proves NOTHING about SET_ROWS bounds.\n");
        return 77;
    }

    bool ok = true;
    // The two live dst kernels: k_set_rows (generic convert) and set_rows_sycl_q
    // (quantised blocks). k_set_rows_fp8 has no dispatch case and is unreachable.
    ok &= run_case(backend, "generic kernel", GGML_TYPE_F32, GGML_TYPE_I64, 256);
    ok &= run_case(backend, "quantised kernel", GGML_TYPE_Q8_0, GGML_TYPE_I32, 256);

    ggml_backend_free(backend);

    printf("\n%s\n", ok ? "All SET_ROWS bounds cases passed" : "SET_ROWS bounds cases FAILED");
    return ok ? 0 : 1;
}

#endif
