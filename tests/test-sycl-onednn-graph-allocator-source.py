"""Source contract for llama.cpp-gwno: the oneDNN Graph SYCL allocator that
routes the compiled SDPA partition's per-execute scratch through the unified
cache instead of oneDNN's default SYCL allocator (S4 root cause, jmc5
c-uxch). Host-only, pure text assertions -- no SYCL device required,
matching test-sycl-fattn-packed-k-lifecycle-source.py's pytest-collectible
pattern (llama_test_pytest hands this file to pytest.main(), so checks must
live inside a test_*() function or pytest's "no tests collected" (exit 5) is
scored as a failure by design, not a skip).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMON_HPP = (ROOT / "ggml/src/ggml-sycl/common.hpp").read_text()
CACHE_HPP = (ROOT / "ggml/src/ggml-sycl/unified-cache.hpp").read_text()
CACHE_CPP = (ROOT / "ggml/src/ggml-sycl/unified-cache.cpp").read_text()


def test_onednn_graph_allocator_source_contract() -> None:
    checks = {
        # The engine must actually be built WITH an allocator, not the bare
        # dnnl::sycl_interop::make_engine() this whole change exists to stop
        # using as the default path.
        "engine built with allocator": "dnnl::graph::sycl_interop::make_engine_with_allocator(dev, ctx, alloc)" in COMMON_HPP,
        "allocator constructed from the two callbacks": "dnnl::graph::sycl_interop::make_allocator(\n                    ggml_sycl::onednn_graph_sycl_malloc, ggml_sycl::onednn_graph_sycl_free)" in COMMON_HPP,
        "opt-out env var gates it": "onednn_graph_allocator_enabled()" in COMMON_HPP,
        "raw allocator engine kept as the fallback": "dnnl::sycl_interop::make_engine(dev, ctx)" in COMMON_HPP,
        # Callback signatures declared as free functions (no user-data slot in
        # the oneDNN C API to carry a `this` -- see the header comment).
        "malloc callback declared": "void * onednn_graph_sycl_malloc(size_t size, size_t alignment, const void * dev, const void * ctx);" in CACHE_HPP,
        "free callback declared": "void   onednn_graph_sycl_free(void * buf, const void * dev, const void * ctx, void * event);" in CACHE_HPP,
        "opt-out env var declared": "bool onednn_graph_allocator_enabled();" in CACHE_HPP,
        # Backing storage: served from the same ONEDNN VRAM zone the primitive-API
        # scratchpad (reserve_onednn_scratch) already uses, not a fresh raw
        # allocation on every call.
        "malloc method declared": "void * onednn_graph_scratch_alloc(size_t size, size_t alignment, sycl::queue * q);" in CACHE_HPP,
        "free method declared": "void   onednn_graph_scratch_free(void * ptr, const sycl::event * event);" in CACHE_HPP,
        "malloc routes through the ONEDNN zone": "zone_alloc(vram_zone_id::ONEDNN, size, align)" in CACHE_CPP,
        "free routes through the ONEDNN zone": "zone_free(vram_zone_id::ONEDNN, ptr)" in CACHE_CPP,
        # Zone-backed reclaim is IMMEDIATE, deliberately with no event wait --
        # device-side reuse safety comes from every Graph-scratch consumer
        # submitting on the SAME in-order compute queue (ctx.stream()), not
        # from a host-side completion check. This must stay documented (the
        # assumption it rests on, and what to do if it ever stops holding),
        # not just implemented silently.
        "zone reclaim documents the in-order-queue assumption": "ctx.stream()" in CACHE_CPP and "in-order compute queue" in CACHE_CPP,
        # The zone-backed free path must NOT call event_complete()/get_info on
        # the (profiling-enabled) compute queue -- that blocks rather than
        # polls (see get_dma_queue()'s comment), so an earlier version of this
        # function turned every free() into a synchronous wait for that op's
        # device completion. Only the DIRECT-fallback path (below) may still
        # reach retain_handles_until_event(), which defers to the cache's own
        # background drain worker instead of polling here.
        "no event_complete call in the graph-scratch free path": not any(
            call in CACHE_CPP.split("void unified_cache::onednn_graph_scratch_free")[1].split("\nvoid unified_cache::")[0]
            for call in ("event_complete(*event)", "event_complete(it->", "event_complete(event")
        ),
        "direct fallback deferred via retain_handles_until_event": "retain_handles_until_event({ std::move(owner) }, *event);" in CACHE_CPP,
        # Env-tunable floor for the concurrent within-ubatch demand, additive on
        # top of the primitive-API pair (see unified_cache_get_planned_onednn_scratchpad_bytes).
        "graph scratch zone floor is additive": "bytes += onednn_graph_scratch_zone_floor_bytes()" in CACHE_CPP,
        "zone floor env var": "GGML_SYCL_ONEDNN_GRAPH_ZONE_MB" in CACHE_CPP,
        "allocator opt-out env var name": "GGML_SYCL_ONEDNN_CACHE_ALLOCATOR" in CACHE_CPP,
        # High-water byte counter (llama.cpp-gwno perf follow-up): peak
        # concurrently-outstanding bytes, exposed and logged once at teardown
        # so a finished run's log answers "did the zone floor actually cover
        # the concurrent demand" without a special env var.
        "high-water getter declared": "size_t onednn_graph_scratch_high_water_bytes() const { return onednn_graph_scratch_high_water_bytes_; }" in CACHE_HPP,
        "high-water logged once at teardown": "oneDNN Graph scratch high-water" in CACHE_CPP,
    }

    failed = [name for name, ok in checks.items() if not ok]
    assert not failed, "onednn graph allocator source contract failed: " + ", ".join(failed)
