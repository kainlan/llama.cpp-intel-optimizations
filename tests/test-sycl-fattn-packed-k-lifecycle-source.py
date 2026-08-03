"""Static mutation gate for packed-K partial-submit lifetime contracts.

This test deliberately makes no claim about live SYCL queues or GPU completion.
The failpoints are production-coupled seams reserved for the lead's GPU run.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FATTN = (ROOT / "ggml/src/ggml-sycl/fattn.cpp").read_text(encoding="utf-8")
XMX = (ROOT / "ggml/src/ggml-sycl/fattn-xmx-f16-v2.hpp").read_text(encoding="utf-8")
BACKEND = (ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text(encoding="utf-8")


def section(source: str, begin: str, end: str) -> str:
    start = source.index(begin)
    return source[start : source.index(end, start)]


def ordered(body: str, *needles: str) -> None:
    positions = [body.index(needle) for needle in needles]
    assert positions == sorted(positions), (needles, positions)


def test_failpoints_are_exact_match_and_reserved_for_gpu_lifecycle_run() -> None:
    helper = section(
        FATTN,
        "void ggml_sycl_fattn_xmx_test_failpoint",
        "ggml_sycl_fattn_xmx_packed_k::~ggml_sycl_fattn_xmx_packed_k",
    )
    assert 'std::getenv("GGML_SYCL_TEST_PACKED_K_FAIL_AFTER")' in helper
    assert "std::strcmp(selected, checkpoint) == 0" in helper
    assert "throw sycl::exception" in helper


def test_new_sidecar_publishes_retry_identity_before_injected_throw() -> None:
    update = section(
        FATTN,
        "bool ggml_sycl_fattn_xmx_update_packed_k_from_set_rows",
        "void ggml_sycl_fattn_xmx_unregister_packed_k_range",
    )
    new_alloc = section(update, "if (!reuse_alloc) {", "} else {")
    ordered(
        new_alloc,
        "packed.ready_event = zero_event",
        "packed.device      = target_device",
        "packed.D           = GGML_SYCL_FATTN_XMX_PACKED_K_D",
        "packed.n_kv        = n_kv",
        "packed.H_kv        = H_kv",
        "packed.batch       = batch",
        "packed.n_blocks    = n_blocks",
        "packed.total_bytes = total_bytes",
        'ggml_sycl_fattn_xmx_test_failpoint("sidecar-zero-to-update")',
    )

    # These are the exact production predicates a retry uses to rediscover and reuse the surviving owner.
    assert "candidate->k_handle_hash == root_handle_hash" in update
    assert "candidate->packed.device == target_device" in update
    assert "packed.handle.valid() && packed.ptr != nullptr && packed.device == target_device" in update
    assert "packed.D == GGML_SYCL_FATTN_XMX_PACKED_K_D && packed.H_kv == H_kv" in update
    assert "packed.batch == batch && packed.n_kv >= n_kv && packed.n_blocks >= n_blocks" in update
    assert "packed.total_bytes >= total_bytes" in update
    assert "add_prev_dep = ggml_sycl_should_add_dependency(packed.ready_event)" in update


def test_sidecar_propagates_prior_event_and_replaces_each_accepted_submit() -> None:
    submit = section(
        FATTN,
        "static sycl::event ggml_sycl_fattn_xmx_submit_set_rows_update",
        "}  // namespace",
    )
    assert "if (add_prev_dep)" in submit
    assert "cgh.depends_on(previous_sidecar_event)" in submit
    assert "if (add_zero_dep)" in submit
    assert "cgh.depends_on(zero_event)" in submit

    update = section(
        FATTN,
        "bool ggml_sycl_fattn_xmx_update_packed_k_from_set_rows",
        "void ggml_sycl_fattn_xmx_unregister_packed_k_range",
    )
    ordered(
        update,
        "add_prev_dep = ggml_sycl_should_add_dependency(packed.ready_event)",
        "ggml_sycl_fattn_xmx_submit_set_rows_update",
    )
    assert "packed.ready_event,\n                        add_zero_dep, add_prev_dep" in update
    ordered(
        update,
        "zero_event        = ggml_sycl::mem_fill_async",
        "packed.ready_event = zero_event",
        'ggml_sycl_fattn_xmx_test_failpoint("sidecar-zero-to-update")',
        "ggml_sycl_fattn_xmx_submit_set_rows_update",
        "packed.ready_event = update_event",
    )


def test_forced_materializer_propagates_prior_event_and_replaces_success() -> None:
    body = section(
        FATTN,
        "bool ggml_sycl_fattn_xmx_materialize_packed_k",
        "static void ggml_sycl_fattn_xmx_v2_free_split_workspace",
    )
    ordered(
        body,
        "const sycl::event previous_use = out->ready_event",
        "add_prev_dep = ggml_sycl_should_add_dependency(previous_use)",
        "zero_deps.push_back(previous_use)",
        "ggml_sycl::mem_fill_async(out->handle, 0, desc.total_packed_bytes, *stream, zero_deps)",
        "out->ready_event = zero_event",
        'ggml_sycl_fattn_xmx_test_failpoint("materializer-zero-to-pack")',
        "cgh.depends_on(zero_event)",
        "out->ready_event = pack_event",
    )
    assert ".wait(" not in body and ".wait_and_throw(" not in body


def test_packed_consumer_propagates_prior_event_then_replaces_first_and_merge() -> None:
    leaf = section(
        XMX,
        "static sycl::event launch_fattn_xmx_v2_decode_gqa_split_leaf",
        "template <int D, bool use_logit_softcap, typename Q_type>\nbool launch_fattn_xmx_v2_decode_m1n64",
    )
    ordered(
        leaf,
        "const sycl::event packed_ready_event",
        "ggml_sycl_should_add_dependency(packed_ready_event)",
        "cgh.depends_on(packed_ready_event)",
        "*packed_k_ready_event = first_event",
        'ggml_sycl_fattn_xmx_test_failpoint("packed-first-to-merge")',
        "cgh.depends_on(first_event)",
        "return merge_event",
    )

    caller = section(
        XMX,
        "bool launch_fattn_xmx_v2_decode_gqa_split_packed_tk",
        "template <int D, bool use_logit_softcap, typename Q_type>\nbool launch_fattn_xmx_v2_decode_gqa_split_packed(",
    )
    ordered(
        caller,
        "launch_fattn_xmx_v2_decode_gqa_split_leaf",
        "packed_k->ready_event = merge_event",
        "return true",
    )


def test_device_guards_cover_submission_queue_packed_identity_and_resolution() -> None:
    materializer = section(
        FATTN,
        "bool ggml_sycl_fattn_xmx_materialize_packed_k",
        "static void ggml_sycl_fattn_xmx_v2_free_split_workspace",
    )
    assert "stream_device >= 0 && stream_device != target_device" in materializer

    caller = section(
        XMX,
        "bool launch_fattn_xmx_v2_decode_gqa_split_packed_tk",
        "template <int D, bool use_logit_softcap, typename Q_type>\nbool launch_fattn_xmx_v2_decode_gqa_split_packed(",
    )
    ordered(
        caller,
        "packed_k->device != ctx.device",
        "packed_k->handle.resolve(ctx.device)",
        "if (!resolved || !resolved.on_device)",
        "launch_fattn_xmx_v2_decode_gqa_split_leaf",
    )


def test_sidecar_identity_is_handle_hash_plus_device_not_raw_address() -> None:
    lookup = section(
        FATTN,
        "ggml_sycl_fattn_xmx_packed_k * ggml_sycl_fattn_xmx_find_packed_k_sidecar",
        "bool ggml_sycl_fattn_xmx_update_packed_k_from_set_rows",
    )
    assert "entry->k_handle.valid() && entry->k_handle_hash == params.K_handle_hash" in lookup
    assert "packed.device == target_device" in lookup
    assert "k_base ==" not in lookup and "k_base !=" not in lookup

    update = section(
        FATTN,
        "bool ggml_sycl_fattn_xmx_update_packed_k_from_set_rows",
        "void ggml_sycl_fattn_xmx_unregister_packed_k_range",
    )
    assert "const size_t root_handle_hash = root_handle.stable_identity_hash()" in update
    assert "candidate->k_handle_hash == root_handle_hash" in update
    assert "candidate->packed.device == target_device" in update


def test_unregister_is_half_open_range_isolated_and_waits_via_destructor() -> None:
    reset = section(FATTN, "void ggml_sycl_fattn_xmx_packed_k::reset()", "namespace {")
    ordered(reset, "ready_event.wait()", "handle      = {}", "ready_event = {}")

    unregister = section(
        FATTN,
        "void ggml_sycl_fattn_xmx_unregister_packed_k_range",
        "// Kernel names for VTune profiling",
    )
    assert "end   = begin + size;" in unregister
    assert "if (k >= begin && k < end)" in unregister
    assert "g_packed_k_sidecars.erase(it)" in unregister
    assert "else {\n            ++it;" in unregister
    assert ".wait(" not in unregister  # erase invokes packed destructor/reset; no queue-global wait


def test_cache_clear_paths_cannot_erase_sidecars_and_teardown_is_range_scoped() -> None:
    # The sidecar registry has exactly one erase operation and no blanket clear.
    assert FATTN.count("g_packed_k_sidecars.erase(") == 1
    assert "g_packed_k_sidecars.clear(" not in FATTN

    # The only backend callers are the two KV-allocation teardown branches.
    assert BACKEND.count("ggml_sycl_fattn_xmx_unregister_packed_k_range(") == 2
    teardown = section(BACKEND, "static void tiered_kv_buffer_free", "static void * tiered_kv_buffer_get_base")
    assert teardown.count("ggml_sycl_fattn_xmx_unregister_packed_k_range(la.ptr, la.size)") == 2

    # Graph-boundary cache clearing/reset does not unregister packed-K sidecars.
    graph_clear = section(
        BACKEND,
        'ggml_sycl_staging_pool().release_all_idle("graph-boundary-host-zone-reset")',
        'impl_phase_log("reset_prefetch")',
    )
    assert "ggml_sycl_clear_staging_cache()" in graph_clear
    assert "unified_cache_reset_scratch_pool" in graph_clear
    assert "ggml_sycl_fattn_xmx_unregister_packed_k_range" not in graph_clear
