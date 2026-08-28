// TKV-13 (B2) step 4: identity-case correctness harness for demoted-layer
// host attention (docs/plans/2026-08-27-tkv13-b2-addendum.md §4.3, §7 step
// 4). Host-only, no GPU, no model.
//
// What this proves, and why it is shaped this way:
//
// ggml_sycl_dispatch_host_flash_attn (ggml-sycl.cpp:76434) is `static` --
// internal linkage inside a single ~100k-line translation unit -- so it
// cannot be called from a separate test TU (a raw pointer to it does not
// exist outside that file, and static functions do not get an exported
// symbol even when the TU is linked into a shared library). The addendum's
// own §4.3 text anticipates exactly this gap: the wrapper can be exercised
// "minus the actual thread-pool indirection, which a unit fixture can stub
// to run synchronously". This harness instead isolates the precise
// property B2's correctness proof obligation (addendum §4) cares about:
// B2 changes WHO submits the CPU FLASH_ATTN_EXT compute and WHEN the host
// blocks on it, never the math. So this harness runs the SAME graph two
// ways and requires byte-identical output:
//
//   (a) "direct" -- ggml_backend_graph_compute() called synchronously on
//       the calling thread. This is today's B1' shape: ggml_backend_sched's
//       CPU split also just calls this, blocking the scheduler's host
//       thread until it returns (see the addendum §1 root-cause writeup).
//   (b) "through the pool" -- the exact two lines the landed wrapper
//       executes (ggml-sycl.cpp:76556-76557):
//           std::future<void> future = pool.submit([cpu_backend, graph] {
//               ggml_backend_graph_compute(cpu_backend, graph); });
//           future.get();
//       using the REAL ggml_sycl::AttnHostPool class from attn-host-pool.hpp
//       (not a stub -- it is a free header, needs no GPU, and the step-2
//       test (test-attn-host-pool.cpp) already establishes the
//       cpu_selector_v pattern this file reuses).
//
// A raw call to ggml_compute_forward_flash_attn_ext (ggml-cpu/ops.h:92) was
// considered and rejected as the "direct" path: it reads params->wdata (a
// scratch buffer for VKQ32/Q_q, sized by ggml_graph_plan) and
// params->threadpool (ggml_barrier / chunk_set/add, ops.cpp ~9235) -- both
// owned and sized exclusively by ggml_backend_graph_compute's own
// graph-plan machinery. A bare caller cannot safely fabricate either; doing
// so would crash or read uninitialized memory into the "reference" answer,
// which is a void positive control, not a real one. Going through the
// public ggml_backend_graph_compute() API on both sides is what actually
// reaches ggml_compute_forward_flash_attn_ext correctly on both paths.
//
// *** WRAPPER DEFECT FOUND WHILE BUILDING THIS HARNESS (reported, not fixed
// here -- out of this task's file scope; see llama.cpp-sbky) ***
//
// ggml_sycl_dispatch_host_flash_attn builds its graph purely manually
// (ggml-sycl.cpp:76551-76553: `graph->n_nodes = 1; graph->nodes[0] =
// &dst_host;`), with a comment claiming ggml_backend_graph_compute() "does
// not need leafs[] populated". That half is true -- ggml-cpu.c's execution
// loop never reads cgraph->leafs -- but the manual construction ALSO skips
// something that IS load-bearing: ggml_build_forward_expand() (the normal
// way to populate a graph) sets GGML_TENSOR_FLAG_COMPUTE on every non-leaf
// node it visits (ggml.c:7230-7231, ggml_visit_parents_graph), and
// ggml_graph_compute_thread's per-node dispatch loop
// (ggml-cpu.c:3113-3115) silently `continue`s past any node lacking that
// flag:
//     if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) { continue; }
// A tensor built via plain ggml_new_tensor() (as ggml_flash_attn_ext's
// result is) starts with flags == 0, so a manually-assigned dst node NEVER
// gets that flag and is silently skipped -- ggml_compute_forward is never
// called on it. ggml_backend_graph_compute() still returns
// GGML_STATUS_SUCCESS (nothing "failed"; the node was just never visited),
// and dst's freshly-allocated (therefore zero) memory is returned
// unchanged.
//
// This was caught empirically, not by inspection: an early draft of this
// harness reproduced the wrapper's exact manual construction for both
// paths (a) and (b) and case 1 (byte-identical) PASSED -- vacuously, since
// both sides silently computed nothing and compared two all-zero buffers.
// Only case 2 (the perturbation control) caught it, because a different Q
// input still produced the identical (zero) output. Swapping in
// ggml_build_forward_expand() for the SAME tensors/backend made the output
// real and input-sensitive; a control run that only widened the graph's
// node/leaf capacity (without setting the flag) stayed all-zero -- isolating
// GGML_TENSOR_FLAG_COMPUTE, specifically, as the missing ingredient, not
// graph capacity or leaf registration.
//
// Consequence for production: with GGML_SYCL_ATTN_HOST_DISPATCH=1 (default
// OFF), every demoted-layer FLASH_ATTN_EXT routed through this wrapper
// today silently returns zeroed attention output instead of a real
// result -- no crash, no log line, GGML_STATUS_SUCCESS. This blocks step 6
// (the B1'-vs-B2 bit-identical logit diff) until fixed. Likely minimal fix
// (for whoever owns ggml-sycl.cpp, not applied here): set
// `dst_host.flags |= GGML_TENSOR_FLAG_COMPUTE;` before building the graph,
// or call ggml_build_forward_expand(graph, &dst_host) instead of the
// manual n_nodes/nodes[0] assignment.
//
// Because that manual-construction shape is the one found broken, this
// harness's own make_single_node_graph() below explicitly sets the flag
// (see its comment) rather than silently reproducing the wrapper's exact
// lines -- reproducing the defect here would make both compared paths
// equally (and vacuously) broken, defeating the point of an identity check.
//
// Perturbation control (case 2): a comparison harness that always reports
// "identical" proves nothing (a-positive-control-can-itself-be-void). Case
// 2 recomputes the direct path with one Q value perturbed and asserts the
// output DIFFERS from the unperturbed run -- demonstrating the memcmp
// actually discriminates real output changes, not just comparing two
// degenerate (e.g. all-zero) buffers that happen to match. It is what
// caught the wrapper defect above.
//
// Fixture shape: a single decode-time query row (N=1, matches the actual
// TG-time production case -- "Q ... one row per token at TG", addendum §2),
// GQA (n_head=2, n_head_kv=1), F16 K/V (the KV-host buft's actual dtype),
// F32 Q, a small F16 mask with one -INFINITY entry so the online-softmax
// skip branch is exercised too, no ALiBi/softcap (kept off for a minimal,
// exactly-reproducible fixture; the identity property under test does not
// depend on those being nonzero).

#include "../attn-host-pool.hpp"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <future>
#include <random>

// The build is -DNDEBUG (Release), so assert() would compile away and the
// test would pass vacuously (llama.cpp-u2mz). Use an explicit check that
// always runs, per the test-kv-runtime-demotion.cpp / test-attn-host-pool.cpp
// precedent in this directory.
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

using ggml_sycl::AttnHostPool;

namespace {

constexpr int64_t DK    = 8;   // head_dim (K)
constexpr int64_t DV    = 8;   // head_dim (V)
constexpr int64_t N_Q   = 1;   // query rows -- TG decode shape
constexpr int64_t N_KV  = 16;  // kv length
constexpr int64_t N_HD  = 2;   // n_head
constexpr int64_t N_HKV = 1;   // n_head_kv (GQA ratio 2)

// Builds one Q/K/V/mask fixture set with deterministic pseudo-random data
// (fixed seed -- reproducible, not cryptographic). Q is F32, K/V are F16
// (the KV-host buft's real dtype), mask is F16 with one -INFINITY entry.
struct fixture {
    ggml_tensor * q;
    ggml_tensor * k;
    ggml_tensor * v;
    ggml_tensor * mask;
};

fixture make_fixture(ggml_context * ctx, uint32_t seed) {
    std::mt19937                          rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    fixture f;
    f.q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DK, N_Q, N_HD, 1);
    f.k    = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, DK, N_KV, N_HKV, 1);
    f.v    = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, DV, N_KV, N_HKV, 1);
    f.mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, N_KV, N_Q, 1, 1);

    float * qd = (float *) f.q->data;
    for (int64_t i = 0; i < ggml_nelements(f.q); i++) {
        qd[i] = dist(rng);
    }

    ggml_fp16_t * kd = (ggml_fp16_t *) f.k->data;
    for (int64_t i = 0; i < ggml_nelements(f.k); i++) {
        kd[i] = ggml_fp32_to_fp16(dist(rng));
    }

    ggml_fp16_t * vd = (ggml_fp16_t *) f.v->data;
    for (int64_t i = 0; i < ggml_nelements(f.v); i++) {
        vd[i] = ggml_fp32_to_fp16(dist(rng));
    }

    ggml_fp16_t * md = (ggml_fp16_t *) f.mask->data;
    for (int64_t i = 0; i < N_KV; i++) {
        // First kv position masked out -- exercises the online-softmax
        // "mv == -INFINITY -> continue" branch, not just the all-visible
        // case.
        md[i] = (i == 0) ? ggml_fp32_to_fp16(-INFINITY) : ggml_fp32_to_fp16(dist(rng) * 0.1f);
    }

    return f;
}

// Builds a 1-node graph wrapping dst, in its own no_alloc graph context --
// the same shape ggml_sycl_dispatch_host_flash_attn itself builds
// (ggml-sycl.cpp: "A minimal single-node graph ... does not need leafs[]
// populated") EXCEPT for one explicit line this file's top comment
// documents as the wrapper's own missing piece: without
// GGML_TENSOR_FLAG_COMPUTE, ggml_graph_compute_thread's dispatch loop
// (ggml-cpu.c:3113) silently skips this node instead of computing it.
// Normally ggml_build_forward_expand() sets this flag (ggml.c:7230); this
// harness sets it directly to stay faithful to the wrapper's manual,
// leafs-free construction while still being a REAL identity check rather
// than a vacuous zero-vs-zero comparison.
struct owned_graph {
    ggml_context * gctx;
    ggml_cgraph *  graph;
};

owned_graph make_single_node_graph(ggml_tensor * dst) {
    ggml_init_params gip{ /*.mem_size   =*/ggml_graph_overhead_custom(1, false),
                          /*.mem_buffer =*/nullptr,
                          /*.no_alloc   =*/true };
    ggml_context *   gctx  = ggml_init(gip);
    ggml_cgraph *    graph = ggml_new_graph_custom(gctx, 1, false);
    graph->n_nodes         = 1;
    graph->nodes[0]        = dst;
    dst->flags |= GGML_TENSOR_FLAG_COMPUTE;
    return { gctx, graph };
}

}  // namespace

int main() {
    ggml_init_params params{ /*.mem_size   =*/16 * 1024 * 1024,
                             /*.mem_buffer =*/nullptr,
                             /*.no_alloc   =*/false };
    ggml_context *   ctx = ggml_init(params);
    CHECK(ctx != nullptr, "ggml_init failed");

    const float scale         = 1.0f / sqrtf((float) DK);
    const float max_bias      = 0.0f;
    const float logit_softcap = 0.0f;

    ggml_backend_t cpu_be = ggml_backend_cpu_init();
    CHECK(cpu_be != nullptr, "ggml_backend_cpu_init failed");
    // Matches ggml_sycl_attn_host_cpu_backend()'s own choice exactly
    // (ggml-sycl.cpp): one query per decode-time host dispatch, so the
    // pool thread -- not this backend's own internal pool -- is the unit
    // of parallelism. Pinning both paths to 1 thread also removes thread
    // count as a variable, so any observed difference is attributable to
    // "which thread called it", not "how many threads computed it".
    ggml_backend_cpu_set_n_threads(cpu_be, 1);

    // ------------------------------------------------------------------
    // Case 1: identity. Same Q/K/V/mask, computed via path (a) direct and
    // path (b) through AttnHostPool -- must be byte-identical.
    // ------------------------------------------------------------------
    fixture f1 = make_fixture(ctx, /*seed=*/42);

    ggml_tensor * dst_a = ggml_flash_attn_ext(ctx, f1.q, f1.k, f1.v, f1.mask, scale, max_bias, logit_softcap);
    ggml_tensor * dst_b = ggml_flash_attn_ext(ctx, f1.q, f1.k, f1.v, f1.mask, scale, max_bias, logit_softcap);
    CHECK(ggml_nbytes(dst_a) == ggml_nbytes(dst_b), "case 1: dst shapes must match");
    CHECK(ggml_nbytes(dst_a) == DV * N_HD * N_Q * sizeof(float), "case 1: dst byte size sanity");

    owned_graph ga = make_single_node_graph(dst_a);
    CHECK(ggml_backend_graph_compute(cpu_be, ga.graph) == GGML_STATUS_SUCCESS,
          "case 1: path (a) direct compute failed");

    owned_graph  gb = make_single_node_graph(dst_b);
    AttnHostPool pool;
    sycl::queue  q_dev{ sycl::cpu_selector_v };
    pool.init(1, q_dev);
    CHECK(pool.is_active(), "case 1: AttnHostPool failed to activate");
    {
        ggml_backend_t    be    = cpu_be;
        ggml_cgraph *     graph = gb.graph;
        std::future<void> fut   = pool.submit([be, graph] {
            // Mirrors ggml_sycl_dispatch_host_flash_attn's exact call
            // (ggml-sycl.cpp:76556): submit the graph compute itself onto
            // the pool worker, not just a signal.
            ggml_backend_graph_compute(be, graph);
        });
        fut.get();  // synchronous for this landing -- matches the wrapper's own future.get().
    }
    pool.shutdown();

    CHECK(memcmp(dst_a->data, dst_b->data, ggml_nbytes(dst_a)) == 0,
          "case 1: direct vs pool-dispatched output must be byte-identical");

    // ------------------------------------------------------------------
    // Case 2: perturbation control. A run with one Q value changed must
    // produce DIFFERENT output -- proves case 1's memcmp is a real
    // discriminator, not a vacuous pass (e.g. two zero-filled buffers).
    // This is the check that caught the wrapper defect documented at the
    // top of this file.
    // ------------------------------------------------------------------
    ggml_tensor * q_perturbed = ggml_dup_tensor(ctx, f1.q);
    memcpy(q_perturbed->data, f1.q->data, ggml_nbytes(f1.q));
    ((float *) q_perturbed->data)[0] += 1.0f;

    ggml_tensor * dst_c = ggml_flash_attn_ext(ctx, q_perturbed, f1.k, f1.v, f1.mask, scale, max_bias, logit_softcap);
    owned_graph   gc    = make_single_node_graph(dst_c);
    CHECK(ggml_backend_graph_compute(cpu_be, gc.graph) == GGML_STATUS_SUCCESS,
          "case 2: perturbed-input compute failed");

    CHECK(memcmp(dst_a->data, dst_c->data, ggml_nbytes(dst_a)) != 0,
          "case 2 (perturbation control): a different Q input must change the output -- "
          "if this fires, case 1's byte-identical check is not a real comparison");

    ggml_free(ga.gctx);
    ggml_free(gb.gctx);
    ggml_free(gc.gctx);
    ggml_backend_free(cpu_be);
    ggml_free(ctx);

    std::printf("test-attn-host-flash-attn-identity: all ok\n");
    return 0;
}
