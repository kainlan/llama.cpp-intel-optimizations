// GPU test for llama.cpp-asdt (plan task L2b): the SYCL tiered KV buffer's
// per-view-tensor extras must stay bounded across repeated GRAPH REBUILDS,
// not grow one entry per view every rebuild.
//
// Background: llama.cpp-dfo0 plan task L2 bounded the COMPUTE-usage buffer's
// tensor_extras vector across graph rebuilds, but the pp1024 RssAnon still
// grew ~300 MB/decode afterward. A jemalloc heap profile attributed the
// residual to tiered_kv_buffer_init_tensor's VIEW branch: llama_kv_cache::
// get_k/get_v build a fresh ggml_view_* over the tiered KV buffer's
// persistent per-layer K/V tensors on every llm_graph_result::reset()
// rebuild (the same rebuild mechanism L2's header describes -- a fresh
// ggml_init over the SAME backing mem_buffer, minting new ggml_tensor
// structs at the addresses old ones occupied), and that branch allocated a
// brand-new ggml_tensor_extra_gpu for every one of those views and never
// released it. Unlike the COMPUTE buffer, this buffer type's `.reset` is
// NULL and is never reached by ggml_gallocr_alloc_graph's reset loop (that
// loop only walks galloc->buffers[], populated solely by COMPUTE-usage
// vbuffer allocation; the tiered KV buffer is allocated directly by llama's
// KV cache init and never registered there).
//
// This test allocates K/V tensors through the SAME tiered KV buffer type
// llama uses (ggml_backend_sycl_kv_buffer_type), then alternates two
// DIFFERENTLY-SIZED graphs through the same ggml_backend_sched, each
// building views of those K/V tensors (mirroring llama_kv_cache::get_k/
// get_v) in a FRESH ggml_context over the SAME raw mem_buffer every
// iteration -- exactly mirroring llm_graph_result::reset(), as the L2 test
// does for the compute-buffer case. It asserts the tiered KV buffer's
// view_extras count (read via the GGML_SYCL_PRIVATE_TESTING-only debug
// accessor ggml_backend_sycl_debug_last_kv_view_extra_count(), since
// ggml_backend_sched does not hand test code the KV buffer it allocates
// internally) stays flat at exactly 2*N_LAYERS (one distinct (view_src,
// view_offs) key per K view + one per V view, per layer) across 20 rebuilds
// instead of growing every iteration.
//
// SAME-GRAPH COLLISION CASE (design review finding, kqy7): a first design
// keyed release-and-replace purely on (view_src, view_offs), which is UNSAFE
// -- llama_kv_cache::get_k's attention-window view and cpy_k's
// ggml_set_rows() result (a whole-tensor view: see ggml_set_rows() in
// ggml.c, `ggml_view_tensor(ctx, a)`) both view the SAME K tensor at offset 0
// in ONE graph, and llama_kv_cache_dsv4's multi-stream cpy_k loop can call
// its inner cpy_k, and thus ggml_set_rows(), more than once per layer per
// graph -- so releasing on a same-key arrival can free an extra a DIFFERENT,
// still-live tensor's ->extra still points at. Layer 0 here builds exactly
// this shape: a "wide" view of k[0] at offset 0 (the attention-window
// analogue, sized n_kv) and a "narrow" view of k[0] ALSO at offset 0 (the
// set_rows-result analogue, a fixed small size), both consumed by their own
// ggml_cont in the SAME graph. The fix under test shares one extra between
// same-key, same-epoch views instead of releasing one for the other's sake
// (see kv_view_extra_entry and g_sycl_kv_view_epoch in ggml-sycl.cpp); this
// test verifies that both views' ->extra end up non-null and IDENTICAL
// (proving they shared rather than one dangling after the other's release),
// and reads back both ggml_cont outputs to confirm they contain the
// zero-filled pattern the K tensor was initialised with, not garbage from a
// freed extra's stale/dangling resolution.
//
// Two properties of g_sycl_kv_view_epoch (the process-global counter this
// fix reads, incremented in ggml_backend_sycl_buffer_reset's COMPUTE branch
// -- same-process-global pattern as L2's g_sycl_debug_last_compute_buffer_
// extra_count) worth stating explicitly: (1) with several COMPUTE buffers
// live (multi-chunk or multi-device), the counter can advance by MORE than
// one per rebuild -- harmless, since the release predicate is "strictly
// older than the current epoch", not "exactly one behind"; any advance at
// all proves at least one real rebuild happened. (2) if a graph were ever
// rebuilt with zero SYCL COMPUTE-buffer resets in between (not believed to
// occur for any graph that also uses a SYCL-resident tiered KV buffer, since
// attention's own Q/K/V projections and output always need COMPUTE-buffer
// allocation), the counter simply would not advance, and a same-key view
// from that "invisible" rebuild would take the SAME-epoch share path instead
// of being released. That is still safe: sharing an extra just means both
// tensors resolve to the same up-to-date pointer, never a stale or dangling
// one -- the failure mode is one cycle of delayed reclamation, not
// corruption.
//
// RED/GREEN evidence for the epoch fix (recorded here rather than reproduced
// by this file, since RED requires a scratch source edit that must never
// land -- see the tracker for what was actually observed on hardware).
// Reverting the epoch check in tiered_kv_buffer_init_tensor's view branch to
// unconditional release (dropping only the `candidate.epoch ==
// current_epoch` share branch, so every same-key match releases and
// replaces regardless of epoch) is PREDICTED, not built or observed, to
// reproduce the layer-0 same-graph collision: the wide attention view and
// the narrow set_rows-result view share one key within a single graph, and
// under this mutant the second arrival releases the first view's still-live
// extra, so the earlier ggml_cont's readback comes back corrupt (or the
// process crashes) instead of all-zero. It does NOT make the count this
// test prints grow -- release-and-replace on every match still keeps
// view_extras flat at N_VIEWS_PER_GRAPH, since this mutant removes only
// SHARING, not release-and-replace itself. (An earlier draft of this
// comment wrongly claimed this same mutant also grows the count by
// 2*N_LAYERS per iteration -- spec review round 1, c-hh2r #1, caught that
// growth is instead the signature of a more drastic mutant that removes the
// view_extras tracking and release logic entirely, reverting to the pre-
// L2b baseline this ticket's own description measured directly on
// hardware: 1152 live 277,704 B extras after 12 rebuilds, ~150 MB/rebuild,
// never released. That mutant predates this test file's tracking accessors
// and was not built either.)
//
// A SECOND, separate bug (found by a hardware jemalloc profile after the
// epoch fix above landed, still visible on kv_view_extras staying flat):
// the older-epoch release branch called release_extra_gpu() exactly ONCE
// regardless of how many times the same-epoch share branch had called
// retain_extra_gpu() on that entry (kv_view_extra_entry::share_count). A
// shared entry's refcount is 1 + share_count, so one release call leaves it
// at share_count (never zero, never freed) while the entry is popped from
// view_extras -- an orphaned, un-tracked, un-freed extra every rebuild for
// every key that was ever shared. kv_view_extras (container membership)
// cannot see this, since the orphan is no longer in the container; that is
// why this test also reads ggml_backend_sycl_debug_live_kv_view_extra_count()
// (extras actually not-yet-deleted, tracked via a debug_is_kv_view_extra flag
// release_extra_gpu() checks right before its one and only `delete extra`)
// and asserts IT stays flat at N_VIEWS_PER_GRAPH too. The RED that was
// ACTUALLY OBSERVED for this bug is a hardware jemalloc profile of the
// pre-fix build (tracker comments c-871e / c-0p3n on llama.cpp-asdt): 448
// live 320 KB extras allocated in tiered_kv_buffer_init_tensor via
// ggml_backend_view_init after four pp1024 decodes, reproduced identically
// under GGML_SYCL_DISABLE_GRAPH=1. The mutant this test's own tracking
// would need to reproduce that in-process is PREDICTED, not built or
// observed: revert the release loop (`for (uint32_t r = 0; r <=
// candidate.share_count; ++r) release_extra_gpu(...)`) back to a single
// `release_extra_gpu(candidate.extra);` call -- kv_view_extras should stay
// flat (this bug is invisible to it) while kv_view_extras_live should grow
// by one every iteration (this test's layer-0 K key is shared every
// iteration, so it would leak one 277 KB extra per rebuild).
//
// ggml_backend_sched_new() asserts its LAST backend entry is a CPU device
// (ggml-backend.cpp:2518); this test still runs everything on SYCL (the CPU
// backend is present only to satisfy that structural requirement, matching
// how llama.cpp itself always registers a CPU backend), and asserts the
// per-iteration count is > 0 once a rebuild has happened so a misplacement
// onto the CPU backend (where the SYCL debug accessor would silently read 0
// forever, making the "count stayed bounded" assertion pass for the wrong
// reason) fails loudly instead of passing vacuously.
//
// Usage:
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-sycl-kv-view-extra-reuse

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl.h"
#include "ggml-sycl/ggml-sycl-test.hpp"
#include "ggml.h"
#include "test-skip.h"  // LLAMA_TEST_EXIT_SKIP: the one definition of "77 means skip"

#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if !defined(GGML_SYCL_PRIVATE_TESTING)
#    error "this test requires GGML_SYCL_PRIVATE_TESTING (link against ggml-sycl-private-fixtures)"
#endif

namespace {

constexpr uint32_t N_LAYERS          = 3;
constexpr size_t   K_BYTES           = 4096;               // F16 bytes per layer's K allocation
constexpr size_t   V_BYTES           = 4096;               // F16 bytes per layer's V allocation
constexpr size_t   KV_PER_LAYER      = K_BYTES + V_BYTES;  // exact, no slack -- matches the boundary-exact
                                                           // layout tests/test-sycl-kv-planned-device-
                                                           // materialization.cpp already proves works
constexpr size_t   TOTAL_BYTES       = N_LAYERS * KV_PER_LAYER;
constexpr int64_t  N_KV_ELEMS_A      = 1024;               // larger shape: must run first (see the loop below)
constexpr int64_t  N_KV_ELEMS_B      = 512;
constexpr int64_t  N_NARROW_ELEMS    = 64;  // layer-0 "set_rows-result" analogue: same offset, smaller, fixed size
constexpr int      N_ITERS           = 20;
constexpr size_t   N_VIEWS_PER_GRAPH = 2 * N_LAYERS;  // one distinct key per K view + one per V view, per layer
                                                      // (layer 0's extra narrow K view shares its K key -- see
                                                      // build_graph's out-params and the same-graph check below)

// Builds one graph: for every layer, a view of that layer's K tensor and a
// view of its V tensor (mirroring llama_kv_cache::get_k/get_v), each copied
// into a fresh COMPUTE-buffer output via ggml_cont so the graph performs
// real SYCL dispatch rather than leaving the views as unconsumed leaves.
//
// Layer 0 additionally gets a SECOND, "narrow" view of the SAME k[0] tensor
// at the SAME offset (0) -- the same-graph collision shape the design
// review found (llama_kv_cache::get_k's wide attention view vs cpy_k's
// ggml_set_rows() whole-tensor result, both viewing K at offset 0). Its view
// tensor and ggml_cont output are handed back via out-params so the caller
// can check both K views' ->extra end up shared (not one dangling after the
// other's release) and both readbacks are still correct.
void build_graph(ggml_context * ctx,
                 ggml_cgraph *  gf,
                 ggml_tensor *  k[N_LAYERS],
                 ggml_tensor *  v[N_LAYERS],
                 int64_t        n_kv,
                 ggml_tensor ** wide_kview_l0,
                 ggml_tensor ** wide_kout_l0,
                 ggml_tensor ** narrow_kview_l0,
                 ggml_tensor ** narrow_kout_l0) {
    for (uint32_t l = 0; l < N_LAYERS; ++l) {
        ggml_tensor * kview = ggml_view_1d(ctx, k[l], n_kv, 0);
        ggml_tensor * vview = ggml_view_1d(ctx, v[l], n_kv, 0);
        ggml_tensor * kout  = ggml_cont(ctx, kview);
        ggml_tensor * vout  = ggml_cont(ctx, vview);
        ggml_set_output(kout);
        ggml_set_output(vout);
        ggml_build_forward_expand(gf, kout);
        ggml_build_forward_expand(gf, vout);

        if (l == 0) {
            ggml_tensor * narrow_kview = ggml_view_1d(ctx, k[l], N_NARROW_ELEMS, 0);
            ggml_tensor * narrow_kout  = ggml_cont(ctx, narrow_kview);
            ggml_set_output(narrow_kout);
            ggml_build_forward_expand(gf, narrow_kout);
            *wide_kview_l0   = kview;
            *wide_kout_l0    = kout;
            *narrow_kview_l0 = narrow_kview;
            *narrow_kout_l0  = narrow_kout;
        }
    }
}

}  // namespace

int main(int, char ** argv) {
    // libccl's static initializer makes libsycl memoize ONEAPI_DEVICE_SELECTOR
    // before main() runs, so a plain setenv() here is too late to steer device
    // selection (llama.cpp-2x3m c-oftt, gdb-traced): re-exec once with the
    // selector already in the environment.
    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:1", 1);
        execv("/proc/self/exe", argv);
        fprintf(stderr, "warning: re-exec failed (%s)\n", strerror(errno));
    }

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        fprintf(stderr,
                "SKIP: no SYCL GPU device available -- NO DEVICE WORK WAS PERFORMED.\n"
                "      source /opt/intel/oneapi/setvars.sh --force and re-run.\n");
        return LLAMA_TEST_EXIT_SKIP;
    }

    // ggml_backend_sched_new() requires its last backend to be a CPU device
    // (ggml-backend.cpp:2518, GGML_ASSERT) -- present as a structural
    // fallback only; every op in this graph (view, cont) is SYCL-supported,
    // so nothing is expected to actually dispatch here.
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        fprintf(stderr, "FAIL: ggml_backend_cpu_init failed\n");
        ggml_backend_free(backend);
        return 1;
    }

    // A synthetic single-device placement plan sized for N_LAYERS tiny KV
    // layers, mirroring tests/test-sycl-kv-planned-device-materialization.cpp
    // (the boundary-exact kv_per_layer arithmetic this test relies on is
    // already proven there for the K-only case; here K+V share the same
    // per-layer budget exactly, K at offset 0 and V at offset K_BYTES).
    ggml_sycl::placement_plan plan;
    plan.multi_device    = true;
    plan.device_id       = -1;
    plan.kv_per_layer    = KV_PER_LAYER;
    plan.kv_vram_bytes   = TOTAL_BYTES;
    plan.kv_host_bytes   = 0;
    plan.devices         = { 0 };
    plan.per_device_vram = { TOTAL_BYTES };
    for (uint32_t l = 0; l < N_LAYERS; ++l) {
        plan.kv_device[l]    = 0;
        plan.layer_device[l] = 0;
    }
    ggml_sycl::test_set_kv_placement_plan(plan, N_LAYERS, KV_PER_LAYER);

    // Persistent K/V tensors, allocated once through the tiered KV buffer --
    // mirrors how llama's KV cache tensors outlive every ephemeral per-decode
    // graph context and are referenced by pointer (via view_src) from each
    // rebuild's view tensors.
    ggml_init_params wparams = {
        /*.mem_size   =*/ggml_tensor_overhead() * 2 * N_LAYERS + 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * wctx = ggml_init(wparams);
    if (!wctx) {
        fprintf(stderr, "FAIL: ggml_init (KV tensor context) failed\n");
        ggml_sycl::test_clear_kv_placement_plan();
        ggml_backend_free(cpu);
        ggml_backend_free(backend);
        return 1;
    }

    ggml_tensor * k[N_LAYERS];
    ggml_tensor * v[N_LAYERS];
    for (uint32_t l = 0; l < N_LAYERS; ++l) {
        char name[32];
        k[l] = ggml_new_tensor_1d(wctx, GGML_TYPE_F16, static_cast<int64_t>(K_BYTES / sizeof(ggml_fp16_t)));
        snprintf(name, sizeof(name), "cache_k_l%u", l);
        ggml_set_name(k[l], name);
        v[l] = ggml_new_tensor_1d(wctx, GGML_TYPE_F16, static_cast<int64_t>(V_BYTES / sizeof(ggml_fp16_t)));
        snprintf(name, sizeof(name), "cache_v_l%u", l);
        ggml_set_name(v[l], name);
    }

    ggml_backend_buffer_type_t kv_buft = ggml_backend_sycl_kv_buffer_type(0);
    ggml_backend_buffer_t      kv_buf  = ggml_backend_buft_alloc_buffer(kv_buft, TOTAL_BYTES);
    if (!kv_buf) {
        fprintf(stderr, "FAIL: tiered KV buffer allocation failed\n");
        ggml_free(wctx);
        ggml_sycl::test_clear_kv_placement_plan();
        ggml_backend_free(cpu);
        ggml_backend_free(backend);
        return 1;
    }

    uint8_t * kv_base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(kv_buf));
    bool      ok      = true;
    for (uint32_t l = 0; l < N_LAYERS && ok; ++l) {
        const size_t layer_off = l * KV_PER_LAYER;
        if (ggml_backend_tensor_alloc(kv_buf, k[l], kv_base + layer_off) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "FAIL: cache_k_l%u allocation failed\n", l);
            ok = false;
            break;
        }
        if (ggml_backend_tensor_alloc(kv_buf, v[l], kv_base + layer_off + K_BYTES) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "FAIL: cache_v_l%u allocation failed\n", l);
            ok = false;
            break;
        }
        std::vector<uint8_t> zeros_k(K_BYTES, 0);
        std::vector<uint8_t> zeros_v(V_BYTES, 0);
        ggml_backend_tensor_set(k[l], zeros_k.data(), 0, zeros_k.size());
        ggml_backend_tensor_set(v[l], zeros_v.data(), 0, zeros_v.size());
    }

    ggml_backend_sched_t sched = nullptr;
    if (ok) {
        ggml_backend_t backends[2] = { backend, cpu };  // CPU MUST be last -- see the comment above.
        sched                      = ggml_backend_sched_new(backends, nullptr, 2, 4096, false, true);
        if (!sched) {
            fprintf(stderr, "FAIL: ggml_backend_sched_new failed\n");
            ok = false;
        }
    }

    // Fresh ggml_init over the SAME raw mem_buffer every iteration, exactly
    // mirroring llm_graph_result::reset() (src/llama-graph.cpp) the same way
    // tests/test-sycl-compute-buffer-extra-reuse.cpp does for the COMPUTE
    // buffer case.
    const size_t         max_nodes = 64;
    const size_t         mem_size  = ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false);
    std::vector<uint8_t> mem_buffer(mem_size);

    size_t max_extras  = 0;
    size_t last_extras = 0;
    size_t max_live    = 0;
    size_t last_live   = 0;

    for (int iter = 0; iter < N_ITERS && ok; ++iter) {
        // The LARGER shape must run first (same reasoning as the COMPUTE
        // buffer test: iteration 0 sizes the compute buffer that the ggml_cont
        // outputs land in, and a later iteration whose outputs exceed that
        // recorded size_max would force a compute-buffer reallocation --
        // irrelevant to the tiered KV buffer's own bookkeeping under test
        // here, but kept for parity with the sibling test).
        const int64_t n_kv = (iter % 2 == 0) ? N_KV_ELEMS_A : N_KV_ELEMS_B;

        ggml_backend_sched_reset(sched);

        ggml_init_params iparams = {
            /*.mem_size   =*/mem_size,
            /*.mem_buffer =*/mem_buffer.data(),
            /*.no_alloc   =*/true,
        };
        ggml_context * ctx = ggml_init(iparams);
        if (!ctx) {
            fprintf(stderr, "FAIL: ggml_init failed at iteration %d\n", iter);
            ok = false;
            break;
        }

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, max_nodes, false);

        ggml_tensor * wide_kview_l0   = nullptr;
        ggml_tensor * wide_kout_l0    = nullptr;
        ggml_tensor * narrow_kview_l0 = nullptr;
        ggml_tensor * narrow_kout_l0  = nullptr;
        build_graph(ctx, gf, k, v, n_kv, &wide_kview_l0, &wide_kout_l0, &narrow_kview_l0, &narrow_kout_l0);

        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            fprintf(stderr, "FAIL: ggml_backend_sched_alloc_graph failed at iteration %d\n", iter);
            ggml_free(ctx);
            ok = false;
            break;
        }

        const enum ggml_status status = ggml_backend_sched_graph_compute(sched, gf);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "FAIL: ggml_backend_sched_graph_compute failed at iteration %d (status=%d)\n", iter,
                    (int) status);
            ggml_free(ctx);
            ok = false;
            break;
        }

        // Same-graph collision check (design review, kqy7): the wide and
        // narrow layer-0 K views share (view_src=k[0], view_offs=0). Neither
        // may be dangling, and since same-key/same-epoch views resolve to
        // byte-identical extra contents, the fix shares one extra between
        // them rather than releasing one for the other's arrival.
        if (!wide_kview_l0->extra || !narrow_kview_l0->extra) {
            fprintf(stderr, "FAIL: layer-0 K view extra is null at iteration %d (wide=%p narrow=%p)\n", iter,
                    (void *) wide_kview_l0->extra, (void *) narrow_kview_l0->extra);
            ggml_free(ctx);
            ok = false;
            break;
        }
        if (wide_kview_l0->extra != narrow_kview_l0->extra) {
            fprintf(stderr,
                    "FAIL: layer-0 K wide/narrow views did not share one extra at iteration %d (wide=%p "
                    "narrow=%p) -- one may have released the other's still-live extra\n",
                    iter, wide_kview_l0->extra, narrow_kview_l0->extra);
            ggml_free(ctx);
            ok = false;
            break;
        }
        {
            std::vector<uint8_t> wide_readback(ggml_nbytes(wide_kout_l0));
            std::vector<uint8_t> narrow_readback(ggml_nbytes(narrow_kout_l0));
            ggml_backend_tensor_get(wide_kout_l0, wide_readback.data(), 0, wide_readback.size());
            ggml_backend_tensor_get(narrow_kout_l0, narrow_readback.data(), 0, narrow_readback.size());
            for (uint8_t byte : wide_readback) {
                if (byte != 0) {
                    fprintf(stderr, "FAIL: layer-0 wide K readback is non-zero at iteration %d -- corrupted read\n",
                            iter);
                    ggml_free(ctx);
                    ok = false;
                    break;
                }
            }
            if (ok) {
                for (uint8_t byte : narrow_readback) {
                    if (byte != 0) {
                        fprintf(stderr,
                                "FAIL: layer-0 narrow K readback is non-zero at iteration %d -- corrupted read\n",
                                iter);
                        ggml_free(ctx);
                        ok = false;
                        break;
                    }
                }
            }
        }
        if (!ok) {
            break;
        }

        last_extras = ggml_backend_sycl_debug_last_kv_view_extra_count();
        last_live   = ggml_backend_sycl_debug_live_kv_view_extra_count();
        if (last_extras > max_extras) {
            max_extras = last_extras;
        }
        if (last_live > max_live) {
            max_live = last_live;
        }
        printf("iter=%d n_kv=%lld kv_view_extras=%zu kv_view_extras_live=%zu\n", iter, (long long) n_kv, last_extras,
               last_live);

        ggml_free(ctx);

        if (last_extras != N_VIEWS_PER_GRAPH) {
            fprintf(stderr,
                    "FAIL: tiered KV buffer view_extras count %zu != N_VIEWS_PER_GRAPH=%zu at iteration %d -- view "
                    "extras from a rebuilt graph are not being released (or are being over-released)\n",
                    last_extras, N_VIEWS_PER_GRAPH, iter);
            ok = false;
            break;
        }
        // llama.cpp-asdt bug fix (jemalloc profile): the container-membership
        // count above cannot see an extra that was popped from view_extras
        // but never actually freed because a release call was missing (a
        // shared entry's refcount is 1 + share_count; releasing it fewer
        // times than that leaks it invisibly to last_extras). This is the
        // check that actually catches that bug -- it must equal last_extras
        // (one live object per tracked key) whenever the fix is correct, and
        // grows without bound pre-fix (this test's layer-0 K key is shared
        // every iteration, so it leaks one 277 KB extra per rebuild).
        if (last_live != N_VIEWS_PER_GRAPH) {
            fprintf(stderr,
                    "FAIL: tiered KV buffer LIVE extra count %zu != N_VIEWS_PER_GRAPH=%zu at iteration %d -- a "
                    "shared entry's extra was not released enough times to actually free it\n",
                    last_live, N_VIEWS_PER_GRAPH, iter);
            ok = false;
            break;
        }
    }

    printf("max_extras_observed=%zu max_live_observed=%zu (both expected flat at %zu)\n", max_extras, max_live,
           N_VIEWS_PER_GRAPH);

    if (sched) {
        ggml_backend_sched_free(sched);
    }
    ggml_backend_buffer_free(kv_buf);
    ggml_free(wctx);
    ggml_sycl::test_clear_kv_placement_plan();
    ggml_backend_free(cpu);
    ggml_backend_free(backend);

    if (!ok) {
        fprintf(stderr, "test-sycl-kv-view-extra-reuse: FAIL\n");
        return 1;
    }
    printf("test-sycl-kv-view-extra-reuse: PASS\n");
    return 0;
}
