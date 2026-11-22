//
// ggml_backend_tensor_get returns the bytes ggml_backend_tensor_set was given.
//
// The SYCL backend may store a quantized weight in another order than ggml's block layout (SOA: one plane per
// block field). That is the backend's own business: a reader of the tensor gets ggml's bytes. test-backend-ops
// builds its CPU reference from such a read (ggml_backend_graph_copy), and llama reads token-embedding rows the
// same way.
//
// Per case: a [N_COLS, n_rows] weight in a buffer marked GGML_BACKEND_BUFFER_USAGE_WEIGHTS, written whole, then
// read back whole, as one row, as a window that starts and ends inside blocks, as its last three bytes, and
// through a view of three rows. F32 has one layout, so its case is the positive control of the harness: it
// passes on any build. Whether a quantized type is stored reordered is the backend's decision per device and
// the test does not ask; on a build where set_tensor reorders and get_tensor does not invert, those cases fail.
// The last case is larger than one readback chunk (32 MiB), so the chunk walk is covered.
// Every case then runs one MUL_MAT with the weight as src0 and reads the whole weight again: a layout the backend
// resolves for a kernel (a reorder in place, a cache copy) does not change what a read returns. That step prints
// its own ok line, and the final line counts them, so a run that skipped it cannot print the same output.
//
// SYCL backend only, no scheduler, no model; one MUL_MAT graph per case. Exit 77 when no SYCL device is visible.
//

#include "../../../../tests/test-skip.h"  // LLAMA_TEST_EXIT_SKIP: the one definition of "77 means skip"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

// Reads after a MUL_MAT that matched; main requires one per case.
int g_reads_after_mul_mat = 0;

constexpr int64_t N_COLS     = 4608;  // the weight shape of the test-backend-ops case that showed the defect
constexpr size_t  GUARD      = 64;
constexpr uint8_t GUARD_BYTE = 0xA5;

struct type_case {
    ggml_type    type;
    const char * name;
    int64_t      n_rows;
};

const type_case CASES[] = {
    { GGML_TYPE_F32,   "f32",             1000 },
    { GGML_TYPE_Q4_0,  "q4_0",            1000 },
    { GGML_TYPE_Q8_0,  "q8_0",            1000 },
    { GGML_TYPE_Q4_K,  "q4_K",            1000 },
    { GGML_TYPE_Q6_K,  "q6_K",            1000 },
    { GGML_TYPE_MXFP4, "mxfp4",           1000 },
    { GGML_TYPE_Q6_K,  "q6_K-two-chunks", 9000 }, // 34.0 MB: more than one 32 MiB readback chunk
};

// Reads [offset, offset + size) of t and compares it with want. Returns true on a match.
bool read_matches(const ggml_tensor * t,
                  size_t              offset,
                  size_t              size,
                  const uint8_t *     want,
                  const char *        case_name,
                  const char *        what) {
    std::vector<uint8_t> got(size + 2 * GUARD, GUARD_BYTE);
    ggml_backend_tensor_get(t, got.data() + GUARD, offset, size);
    for (size_t i = 0; i < GUARD; ++i) {
        if (got[i] != GUARD_BYTE || got[GUARD + size + i] != GUARD_BYTE) {
            std::fprintf(stderr, "FAIL: %s, %s: the read wrote outside its destination\n", case_name, what);
            return false;
        }
    }
    for (size_t i = 0; i < size; ++i) {
        if (got[GUARD + i] != want[i]) {
            std::fprintf(stderr, "FAIL: %s, %s: byte %zu of %zu is 0x%02x, want 0x%02x\n", case_name, what, i, size,
                         (unsigned) got[GUARD + i], (unsigned) want[i]);
            return false;
        }
    }
    return true;
}

// The positive control for "a kernel's layout resolution does not change what a read returns": one MUL_MAT with w as
// src0 on the same backend (the backend may reorder w in place or resolve a cache copy for it), then w is read again
// and must hold the bytes that were set. Returns true on a match.
bool read_after_mul_mat_matches(ggml_backend_t backend, ggml_tensor * w, const uint8_t * want, const char * case_name) {
    ggml_init_params params = {
        /*.mem_size   =*/ggml_tensor_overhead() * 4 + ggml_graph_overhead(),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "FAIL: %s, after a MUL_MAT: ggml_init failed\n", case_name);
        return false;
    }
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w->ne[0], 1);
    ggml_set_input(x);
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_set_output(y);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t buf = nullptr;
    bool                  ok  = false;
    if (!ggml_backend_supports_op(backend, y)) {
        std::fprintf(stderr, "FAIL: %s, after a MUL_MAT: the backend does not support the MUL_MAT\n", case_name);
    } else if (!(buf = ggml_backend_alloc_ctx_tensors(ctx, backend))) {
        std::fprintf(stderr, "FAIL: %s, after a MUL_MAT: ggml_backend_alloc_ctx_tensors failed\n", case_name);
    } else {
        const std::vector<float> h_x(w->ne[0], 0.5f);
        ggml_backend_tensor_set(x, h_x.data(), 0, ggml_nbytes(x));
        const ggml_status status = ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        if (status != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "FAIL: %s, after a MUL_MAT: graph compute returned %d\n", case_name, (int) status);
        } else {
            ok = read_matches(w, 0, ggml_nbytes(w), want, case_name, "whole tensor after a MUL_MAT");
            if (ok) {
                ++g_reads_after_mul_mat;
                std::fprintf(stderr, "  ok: %s, read after MUL_MAT (%zu bytes)\n", case_name, ggml_nbytes(w));
            }
        }
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
    return ok;
}

// Returns 0 on pass, 1 on failure.
int run_case(ggml_backend_t backend, const type_case & tc) {
    ggml_init_params params = {
        /*.mem_size   =*/ggml_tensor_overhead() * 4,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "FAIL: %s: ggml_init failed\n", tc.name);
        return 1;
    }
    ggml_tensor * w = ggml_new_tensor_2d(ctx, tc.type, N_COLS, tc.n_rows);
    ggml_set_name(w, (std::string("w_") + tc.name).c_str());
    const size_t  row_bytes = w->nb[1];
    ggml_tensor * rows      = ggml_view_2d(ctx, w, N_COLS, 3, row_bytes, 2 * row_bytes);
    ggml_set_name(rows, "rows_2_to_4");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        std::fprintf(stderr, "FAIL: %s: ggml_backend_alloc_ctx_tensors failed\n", tc.name);
        ggml_free(ctx);
        return 1;
    }
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Deterministic bytes. The round trip is about bytes, so the blocks need not decode to sane values.
    std::vector<uint8_t> want(ggml_nbytes(w));
    uint32_t             s = 0x9E3779B9u + (uint32_t) tc.type;
    for (auto & b : want) {
        s = s * 1664525u + 1013904223u;
        b = (uint8_t) (s >> 24);
    }
    ggml_backend_tensor_set(w, want.data(), 0, want.size());
    ggml_backend_synchronize(backend);

    const size_t block_bytes = ggml_type_size(tc.type);
    const size_t win_offset  = row_bytes + 5;
    const size_t win_size    = 3 * block_bytes + 7;
    bool         ok          = true;
    ok                       = read_matches(w, 0, want.size(), want.data(), tc.name, "whole tensor") && ok;
    ok = read_matches(w, 3 * row_bytes, row_bytes, want.data() + 3 * row_bytes, tc.name, "row 3") && ok;
    ok = read_matches(w, win_offset, win_size, want.data() + win_offset, tc.name, "window inside blocks") && ok;
    ok = read_matches(w, want.size() - 3, 3, want.data() + want.size() - 3, tc.name, "last three bytes") && ok;
    ok = read_matches(rows, 0, ggml_nbytes(rows), want.data() + 2 * row_bytes, tc.name, "view of rows 2 to 4") && ok;
    ok = read_after_mul_mat_matches(backend, w, want.data(), tc.name) && ok;

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    if (ok) {
        std::fprintf(stderr, "  ok: %s (%zu bytes)\n", tc.name, want.size());
    }
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    if (ggml_backend_sycl_get_device_count() <= 0) {
        std::fprintf(stderr,
                     "SKIP: no SYCL device available -- NO DEVICE WORK WAS PERFORMED.\n"
                     "      source /opt/intel/oneapi/setvars.sh --force and re-run.\n");
        return LLAMA_TEST_EXIT_SKIP;
    }
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        // A device was counted, so a failed init is a broken backend, not a missing device.
        std::fprintf(stderr, "FAIL: a SYCL device was counted but ggml_backend_sycl_init(0) failed\n");
        return 1;
    }

    const int n_cases  = (int) (sizeof(CASES) / sizeof(CASES[0]));
    int       failures = 0;
    for (const type_case & tc : CASES) {
        failures += run_case(backend, tc);
    }
    ggml_backend_free(backend);

    if (failures > 0 || g_reads_after_mul_mat != n_cases) {
        std::fprintf(stderr,
                     "test-sycl-weight-readback-layout: %d of %d cases failed, %d of %d reads after MUL_MAT matched\n",
                     failures, n_cases, g_reads_after_mul_mat, n_cases);
        return 1;
    }
    std::fprintf(stderr, "test-sycl-weight-readback-layout: PASS (%d cases, %d reads after MUL_MAT)\n", n_cases,
                 g_reads_after_mul_mat);
    return 0;
}
