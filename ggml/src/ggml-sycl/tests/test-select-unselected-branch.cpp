//
// The SYCL backend must not execute a node the graph did not ask for.
//
// ggml_build_forward_select puts every branch into the graph and sets GGML_TENSOR_FLAG_COMPUTE only on the
// selected one. The scheduler does not upload the inputs of an unselected branch, so a backend that runs it reads
// memory nobody wrote: on the B50 at the default context that was a GET_ROWS on the token embedding with garbage ids,
// a GPU page fault and a lost device. Every other backend skips such a node.
//
// This test shows execution through a sentinel and never uses an out-of-range index, so a backend with the defect
// fails it without faulting the card:
//   a = scale(x, 2), b = scale(x, 3), select(gf, {a, b}, idx).
// Before the compute both outputs are filled with a sentinel. After it the selected branch must hold k * x (the
// positive control: the backend ran what it was asked to run) and the unselected one must still hold the sentinel.
// It runs once with idx 0 and once with idx 1, so a backend that runs nothing cannot pass. The sentinel check needs
// a and b in distinct memory; the test refuses to pass if they overlap.
//
// SYCL backend only, no scheduler, no model, F32. Exit 77 when no SYCL device is visible.
//

#include "../../../../tests/test-skip.h"  // LLAMA_TEST_EXIT_SKIP: the one definition of "77 means skip"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr int64_t N_ELEMS    = 4096;
constexpr float   SENTINEL_A = -7777.0f;
constexpr float   SENTINEL_B = -9999.0f;
constexpr float   SCALE_A    = 2.0f;
constexpr float   SCALE_B    = 3.0f;
constexpr float   REL_TOL    = 1e-6f;

enum class branch_state { SENTINEL, COMPUTED, OTHER };

branch_state classify(const std::vector<float> & got, const std::vector<float> & x, float scale, float sentinel) {
    bool all_sentinel = true;
    bool all_computed = true;
    for (size_t i = 0; i < got.size(); ++i) {
        const float want = scale * x[i];
        all_sentinel     = all_sentinel && got[i] == sentinel;
        all_computed     = all_computed && std::fabs(got[i] - want) <= REL_TOL * std::fabs(want);
    }
    if (all_sentinel) {
        return branch_state::SENTINEL;
    }
    return all_computed ? branch_state::COMPUTED : branch_state::OTHER;
}

bool ranges_overlap(const ggml_tensor * p, const ggml_tensor * q) {
    const auto p0 = reinterpret_cast<uintptr_t>(p->data);
    const auto q0 = reinterpret_cast<uintptr_t>(q->data);
    return p0 < q0 + ggml_nbytes(q) && q0 < p0 + ggml_nbytes(p);
}

// Returns 0 on pass, 1 on failure.
int run_case(ggml_backend_t backend, int idx) {
    ggml_init_params params = {
        /*.mem_size   =*/ggml_tensor_overhead() * 8 + ggml_graph_overhead(),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    // One context per case: the COMPUTE flag lives on the tensor, so tensors reused across two selections would
    // carry the first selection's flag into the second.
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "FAIL: idx=%d ggml_init failed\n", idx);
        return 1;
    }

    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N_ELEMS);
    ggml_set_name(x, "x");
    ggml_set_input(x);
    ggml_tensor * a = ggml_scale(ctx, x, SCALE_A);
    ggml_set_name(a, "a");
    ggml_set_output(a);
    ggml_tensor * b = ggml_scale(ctx, x, SCALE_B);
    ggml_set_name(b, "b");
    ggml_set_output(b);

    ggml_cgraph * gf          = ggml_new_graph(ctx);
    ggml_tensor * branches[2] = { a, b };
    ggml_tensor * out         = ggml_build_forward_select(gf, branches, 2, idx);
    (void) out;

    ggml_tensor * selected   = idx == 0 ? a : b;
    ggml_tensor * unselected = idx == 0 ? b : a;
    if ((selected->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || (unselected->flags & GGML_TENSOR_FLAG_COMPUTE) != 0) {
        std::fprintf(stderr, "FAIL: idx=%d ggml_build_forward_select did not flag the branches as expected\n", idx);
        ggml_free(ctx);
        return 1;
    }

    // Every tensor of the context gets its own range: no allocator reuse between the two branches.
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        std::fprintf(stderr, "FAIL: idx=%d ggml_backend_alloc_ctx_tensors failed\n", idx);
        ggml_free(ctx);
        return 1;
    }
    if (ranges_overlap(a, b)) {
        std::fprintf(stderr,
                     "FAIL: idx=%d the allocator gave a (%p) and b (%p) overlapping memory; the sentinel check needs "
                     "distinct memory, so this run proves nothing\n",
                     idx, a->data, b->data);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return 1;
    }

    std::vector<float> h_x(N_ELEMS);
    for (int64_t i = 0; i < N_ELEMS; ++i) {
        h_x[i] = 1.0f + 0.25f * (float) (i % 64);
    }
    const std::vector<float> h_sent_a(N_ELEMS, SENTINEL_A);
    const std::vector<float> h_sent_b(N_ELEMS, SENTINEL_B);
    ggml_backend_tensor_set(x, h_x.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(a, h_sent_a.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, h_sent_b.data(), 0, ggml_nbytes(b));

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: idx=%d ggml_backend_graph_compute returned %d\n", idx, (int) status);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return 1;
    }

    std::vector<float> h_a(N_ELEMS);
    std::vector<float> h_b(N_ELEMS);
    ggml_backend_tensor_get(a, h_a.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_get(b, h_b.data(), 0, ggml_nbytes(b));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    const branch_state state_a          = classify(h_a, h_x, SCALE_A, SENTINEL_A);
    const branch_state state_b          = classify(h_b, h_x, SCALE_B, SENTINEL_B);
    const branch_state state_selected   = idx == 0 ? state_a : state_b;
    const branch_state state_unselected = idx == 0 ? state_b : state_a;
    const char *       name_selected    = idx == 0 ? "a" : "b";
    const char *       name_unselected  = idx == 0 ? "b" : "a";

    int rc = 0;
    if (state_selected != branch_state::COMPUTED) {
        std::fprintf(stderr, "FAIL: idx=%d the selected branch %s was not computed (%s)\n", idx, name_selected,
                     state_selected == branch_state::SENTINEL ? "still holds its sentinel" : "holds other values");
        rc = 1;
    }
    if (state_unselected == branch_state::COMPUTED) {
        std::fprintf(stderr, "FAIL: the unselected branch was executed (idx=%d, branch %s)\n", idx, name_unselected);
        rc = 1;
    } else if (state_unselected == branch_state::OTHER) {
        std::fprintf(stderr, "FAIL: idx=%d the unselected branch %s lost its sentinel to something other than its op\n",
                     idx, name_unselected);
        rc = 1;
    }
    if (rc == 0) {
        std::fprintf(stderr, "  ok: idx=%d selected %s computed, unselected %s kept its sentinel\n", idx, name_selected,
                     name_unselected);
    }
    return rc;
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
        std::fprintf(stderr, "SKIP: ggml_backend_sycl_init(0) failed -- NO DEVICE WORK WAS PERFORMED.\n");
        return LLAMA_TEST_EXIT_SKIP;
    }

    int failures = 0;
    failures += run_case(backend, 0);
    failures += run_case(backend, 1);
    ggml_backend_free(backend);

    if (failures > 0) {
        std::fprintf(stderr, "test-sycl-select-unselected-branch: %d of 2 cases failed\n", failures);
        return 1;
    }
    std::fprintf(stderr, "test-sycl-select-unselected-branch: PASS\n");
    return 0;
}
