#!/usr/bin/env python3
"""Source gate for owner-first staging and w295 transactional growth contracts."""
import os
import re
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"
FATTN = (SYCL / "fattn.cpp").read_text()
RUNTIME = (SYCL / "ggml-sycl.cpp").read_text()
CACHE = (SYCL / "unified-cache.cpp").read_text()
CACHE_HPP = (SYCL / "unified-cache.hpp").read_text()
COMMON = (SYCL / "common.hpp").read_text()
COMMON_IMPL = (SYCL / "common.cpp").read_text()


# One independent-check helper for every pure-Python SYCL gate (tests/sycl_gate.py). This file used to
# carry its own copy of gate()/report_gates(); two copies drift (the copy here lacked min_checks and the
# operand annotation), so there is one.
from sycl_gate import finish, gate  # noqa: E402


def region(source: str, start: str, end: str) -> str:
    begin = source.find(start)
    if begin < 0:
        raise ValueError("region start anchor not found: %r" % start)
    finish = source.find(end, begin)
    if finish < 0:
        raise ValueError("region end anchor not found after %r: %r" % (start, end))
    return source[begin:finish]


def _blank_comments(source: str, keep_strings: bool = True) -> str:
    """Replace // and /* */ comment bodies with spaces, preserving offsets and string
    literals (or, with keep_strings=False, blanking their bodies too: a log message
    that names a function is not a call to it either). Without this, prose naming a function reads as a call to it -- both the
    census below (a doc-only commit can inflate a raw literal count with no code
    change at all -- see 073078ff1/d431c7e44) and the "scan matched nothing" control
    further down would treat a comment mention as the real thing it describes."""
    out, index, size = [], 0, len(source)
    while index < size:
        char = source[index]
        if char in "\"'":
            quote = char
            out.append(char)
            index += 1
            while index < size:
                if source[index] == "\\":
                    out.append(source[index:index + 2] if keep_strings else "  ")
                    index += 2
                    continue
                closes = source[index] == quote
                out.append(source[index] if keep_strings or closes else ("\n" if source[index] == "\n" else " "))
                index += 1
                if closes:
                    break
            continue
        if source.startswith("//", index):
            while index < size and source[index] != "\n":
                out.append(" ")
                index += 1
            continue
        if source.startswith("/*", index):
            end = source.find("*/", index + 2)
            end = size if end < 0 else end + 2
            out.extend("\n" if c == "\n" else " " for c in source[index:end])
            index = end
            continue
        out.append(char)
        index += 1
    return "".join(out)


# Positive control for _blank_comments itself: a comment mention of the token must
# be stripped (so a documentation-only commit cannot move the census below) while a
# real call immediately after survives untouched (so the blanking cannot eat code).
with gate('blank-comments-control'):
    _blank_comments_probe = _blank_comments("// see unified_alloc( in the old code\nunified_alloc(req, &owner);\n")
    assert _blank_comments_probe.count("unified_alloc(") == 1, "comment blinding did not strip a commented mention"
    assert "unified_alloc(req, &owner);" in _blank_comments_probe, "comment blinding ate real code"


def owner_first(block: str, mutation: str) -> None:
    allocation = block.index("unified_allocate_owner(req)")
    owner = block.index("from_owned_alloc", allocation)
    resolved = block.index("resolve(", owner)
    validation = block.index("if (!resolved.ptr", resolved)
    publication = block.index(mutation, validation)
    assert allocation < owner < resolved < validation < publication
    assert "unified_alloc(" not in block
    assert "from_legacy_owned_alloc" not in block


# All four FATTN legacy-owner sites are closed, including both KV-zone users.
with gate('fattn-owner-first'):
    assert FATTN.count("unified_alloc(") == 0
    assert FATTN.count("from_legacy_owned_alloc(") == 0
    assert FATTN.count("unified_allocate_owner(") == 4
    assert FATTN.count("prefer_vram_zone = ggml_sycl::vram_zone_id::KV") == 2

    owner_first(
        region(FATTN, "bool ggml_sycl_fattn_xmx_update_packed_k_from_set_rows", "void ggml_sycl_fattn_xmx_unregister"),
        "packed.handle = std::move(handle)",
    )
    owner_first(
        region(FATTN, "static bool ggml_sycl_fattn_alloc_device_owner", "template <typename T>"),
        "owner = new ggml_sycl::mem_handle",
    )
    owner_first(
        region(FATTN, "bool ggml_sycl_fattn_xmx_materialize_packed_k", "static void ggml_sycl_fattn_xmx_v2_free"),
        "out->handle      = std::move(handle)",
    )
    owner_first(
        region(FATTN, "static bool ggml_sycl_fattn_xmx_v2_alloc_split_workspace_buffer", "static bool ggml_sycl_fattn_xmx_v2_ensure"),
        "*out = std::move(handle)",
    )
    print("PASS fattn-owner-first-source-gate")

# Failure atomicity: no existing output is cleared before allocation and
# resolution succeed, so allocation failure cannot publish a raw pointer or
with gate('fattn-failure-atomicity'):
    # erase the previous owner.
    sidecar_prefix = region(FATTN, "if (!reuse_alloc) {", "const auto resolved = handle.resolve(target_device);")
    assert "packed.reset()" not in sidecar_prefix
    materializer = region(FATTN, "bool ggml_sycl_fattn_xmx_materialize_packed_k", "// Publish every field used by retry reuse")
    assert materializer.index("const auto resolved") < materializer.index("out->reset()")
    print("PASS fattn-allocation-failure-leaves-output-untouched")

# Ten coherent runtime staging/workspace owner sites were migrated. The exact
# compatibility inventory prevents either a silent regression or an unreviewed
# widening of this bounded batch.
#
# Counted comment-blind (RUNTIME_CODE, not RUNTIME): a raw literal count conflates
# documentation with call sites, and this file's own history proves it drifts on a
# comment-only commit. 073078ff1 (llama.cpp-13u6) legitimately retired one raw
# unified_alloc()+from_legacy_owned_alloc() site from ggml_sycl_copy_payload_to_
# handle_async in favour of the shared alloc_pinned_stage_handle_terminal() owner-
# first helper -- a real reduction, 55->54 code sites. The immediately following
# d431c7e44 (comment-only, no code change) then mentioned "unified_alloc()" twice in
# prose explaining that same migration, which pushed the raw literal count from 55
# to 57 with zero call sites added -- silently invalidating the raw-count census
# (llama.cpp-1s31). unified_allocate_owner( shows the same shape from a different
# pair (fe68ef272 added a comment mention, 756be1d6f later removed it): raw count
# round-tripped 25->27->26, but the comment-blind count never moved off 25. Bisected
# with `git log -S'<token>(' -- ggml/src/ggml-sycl/ggml-sycl.cpp`; verified per-commit
# with `git show <sha>:ggml/src/ggml-sycl/ggml-sycl.cpp | grep -o '<token>(' | wc -l`.
RUNTIME_CODE     = _blank_comments(RUNTIME)
COMMON_CODE      = _blank_comments(COMMON)
COMMON_IMPL_CODE = _blank_comments(COMMON_IMPL)

for _label, _code, _pins in (
    ("ggml-sycl.cpp", RUNTIME_CODE, (54, 42, 24)),
    ("common.hpp", COMMON_CODE, (3, 4, 3)),
    ("common.cpp", COMMON_IMPL_CODE, (8, 8, 4)),
):
    for _token, _pinned in zip(("unified_alloc", "from_legacy_owned_alloc", "unified_allocate_owner"), _pins):
        with gate("census %s %s(" % (_label, _token)):
            _found = _code.count(_token + "(")
            assert _found == _pinned, "%s has %d %s( call sites, the reviewed inventory is %d" % (
                _label, _found, _token, _pinned)


def call_sites(code: str, source: str, token: str) -> Counter:
    """Every call to `token` in `code` (comments and strings already blanked), keyed by the
    statement head it sits in and its argument text. `source` is the same text unblanked, so
    the key reads naturally. Two calls with the same key are counted, not merged."""
    sites = Counter()
    for match in re.finditer(r"\b%s\(" % token, code):
        head = " ".join(code[code.rfind("\n", 0, match.start()) + 1:match.start()].split())
        depth, index = 1, match.end()
        while depth:
            depth += (code[index] == "(") - (code[index] == ")")
            index += 1
        sites[(head[-45:], " ".join(source[match.end():index - 1].split()))] += 1
    return sites


# unified-cache.cpp is the allocator itself, and the one file whose sites are legitimately
# still on the legacy alloc_handle form: it implements unified_allocate_owner() on top of
# unified_alloc() and keeps bootstrap mints that cannot route through the coordinator. A raw
# literal count of it (28, until llama.cpp-gsb9) drifted with prose -- 12 of the 14 "new"
# sites were comments and log strings -- so each CODE call site is listed instead. A new
# site changes a count here and has to be classified against
# docs/design/sycl-canonical-memory-architecture.md section 3 before it is added.
CACHE_CODE = _blank_comments(CACHE, keep_strings=False)

# (statement head, argument text) -> number of sites. The comment names the function.
CACHE_UNIFIED_ALLOC_SITES = Counter({
    ("bool", "const alloc_request & req_in, alloc_handle * out"): 1,  # the definition of unified_alloc itself
    ("if (!", "areq, &h"): 1,  # acquire_offload_buffer
    ("if (!", "req, &legacy"): 1,  # unified_allocate_owner_impl: the legacy bridge it wraps
    ("if (", "req, &handle"): 2,  # unified_cache_allocate (two tiers)
    ("if (!", "req, &handle"): 4,  # reserve_onednn_scratch, release_onednn_scratch_reservation,
    #                                reserve_persistent_scratch, unified_cache_zone_allocate
    ("if (!", "req, &partial_owner"): 1,  # unified_cache_unpin_model_weights
    ("if (!", "req, &owner"): 3,  # reserve_compute_arena, reserve_scratch_pool, device_pool_alloc_chunk
    ("if (!", "req, out"): 1,  # unified_cache_reserve_moe_q8_1_scratch
    ("if (!", "req, &moe_owner"): 1,  # moe_preallocate_inference_buffers
    # oneDNN Graph-scratch DIRECT path (dded74997, llama.cpp-0oxf): unified_alloc()
    # + detail::from_legacy_owned_alloc() with no fallible step between them. Legacy form,
    # allowlisted; owner-first (unified_allocate_owner) is the migration target, tracked by llama.cpp-mrtw.
    ("ph_scratch_test_should_force_direct_fail() &&", "req, &handle"): 2,
})
CACHE_FROM_LEGACY_SITES = Counter({
    ("staging_owner_ = detail::", "std::move(owner), GGML_LAYOUT_AOS"): 1,  # the cache's own staging adopt
    ("new_direct_alloc_owner = detail::", "std::move(new_owner), layout"): 2,  # ensure_cached, ensure_cached_alloc
    ("direct_alloc_owner = detail::", "std::move(owner), layout"): 2,  # ensure_cached, ensure_cached_alloc
    ("dnn_graph_scratch_flag_slab_owner_ = detail::", "std::move(owner), GGML_LAYOUT_AOS"): 1,  # bootstrap mint
    ("mem_handle owner = detail::", "std::move(handle), GGML_LAYOUT_AOS"): 1,  # onednn_graph_scratch_alloc
    ("owner = detail::", "std::move(handle), GGML_LAYOUT_AOS"): 3,  # reserve_onednn_scratch,
    #                                release_onednn_scratch_reservation, reserve_persistent_scratch
    ("mem_handle partial_handle = detail::", "std::move(partial_owner), GGML_LAYOUT_SOA"): 1,  # unpin_model_weights
    ("compute_arena_owner_ = detail::", "std::move(owner), GGML_LAYOUT_AOS"): 1,
    ("scratch_pool_owner_ = detail::", "std::move(owner), GGML_LAYOUT_AOS"): 1,
    ("out = std::make_shared<mem_handle>(detail::", "std::move(moe_owner)"): 1,  # moe_preallocate_inference_buffers
})
CACHE_OWNER_FIRST_SITES = Counter({
    ("allocation_result allocation =", "req"): 7,
    ("allocation_result host_allocation =", "host_req"): 1,
    ("allocation_result result =", "req"): 1,  # unified_cache_zone_alloc
    ("allocation_result", "const alloc_request & req"): 1,  # the definition of unified_allocate_owner itself
})

for _token, _allowed in (
    ("unified_alloc", CACHE_UNIFIED_ALLOC_SITES),
    ("from_legacy_owned_alloc", CACHE_FROM_LEGACY_SITES),
    ("unified_allocate_owner", CACHE_OWNER_FIRST_SITES),
):
    with gate("census unified-cache.cpp %s( sites" % _token):
        _found = call_sites(CACHE_CODE, CACHE, _token)
        _extra, _missing = _found - _allowed, _allowed - _found
        assert not _extra and not _missing, (
            "unified-cache.cpp %s( call sites differ from the reviewed list; unlisted: %s; listed but gone: %s" %
            (_token, dict(_extra), dict(_missing)))

# The shared planned-scratch allocator (llama.cpp-479i). It replaced two earlier
# owner-first sites: scoped_mmvq_scratch_handle in ggml-sycl.cpp (retired by 1e4a9bf6a)
# and the per-slot allocation in the MMQ src1 staging ensure_buffer in common.hpp
# (retired by 4a348083c). Both now call this one helper, so the owner-first ordering is
# pinned here once instead of at each caller.
PLANNED_SCRATCH_HELPER = ("inline void * ggml_sycl_runtime_scratch_ensure(", "struct ggml_backend_sycl_context {")

with gate('planned-scratch-ensure-owner-first'):
    planned_scratch = region(COMMON, *PLANNED_SCRATCH_HELPER)
    owner_first(planned_scratch, "backing  = std::move(replacement)")
    # Both planned callers must keep going through the helper rather than allocating themselves.
    # Callers in common.hpp: the MMQ/MMVQ Q8_1 src1 buffer and the dense f16 dequant buffers.
    # Comment- and string-blind: a comment or a string literal naming the helper (a quoted example, a log
    # message) is not a caller.
    assert _blank_comments(COMMON, keep_strings=False).count("ggml_sycl_runtime_scratch_ensure<") == 2, \
        "a planned scratch caller stopped using the helper"
    print("PASS planned-scratch-owner-first-source-gate")

runtime_regions = (
    ("ggml_backend_sycl_context::get_staging_buffer", "ggml_backend_sycl_context::free_staging_buffer"),
    ("ggml_backend_sycl_context::ensure_mmvq_host_staging", "ggml_backend_sycl_context::ensure_readback_staging"),
    ("ggml_backend_sycl_context::ensure_readback_staging", "ggml_backend_sycl_context::new_pool_for_device"),
    ("auto                               allocate_owned_scratch", "auto release_owned_scratch"),
    ("bool ensure_device(T *&", "static bool ggml_sycl_expert_entry_weight_ptr"),
    ("auto ensure_secondary_device_buffer", "// Ensure ALL slots"),
    ("static bool ggml_sycl_moe_down_sum_shadow_record", "static void ggml_sycl_moe_down_sum_shadow_compare"),
    ("// Allocate and validate a replacement before disturbing the published buffer.", "if (pipe.enabled)"),
    ("static void ggml_sycl_mmvq_soa_pre_allocate_buffers", "static void ggml_sycl_xmx_moe_pre_allocate_buffers"),
)
for start, end in runtime_regions:
    with gate("runtime-owner-first region %s" % start.strip()):
        block = region(RUNTIME, start, end)
        assert "unified_allocate_owner(" in block, start
        assert "unified_alloc(" not in block, start
        assert "from_legacy_owned_alloc" not in block, start

# Representative replacement paths prove the old owner/raw view is not reset
# before the replacement has an accepted allocation and validated resolution.
for start, end, forbidden in (
    ("ggml_backend_sycl_context::get_staging_buffer", "ggml_sycl::allocation_result allocation", "free_staging_buffer()"),
    ("ggml_backend_sycl_context::ensure_mmvq_host_staging", "ggml_sycl::allocation_result allocation", "mmvq_host_staging_handle = {}"),
    ("ggml_backend_sycl_context::ensure_readback_staging", "ggml_sycl::allocation_result allocation", "readback_staging_handle = {}"),
    ("bool ensure_device(T *&", "ggml_sycl::allocation_result allocation", "reset_device(ptr, handle)"),
    ("auto ensure_secondary_device_buffer", "ggml_sycl::allocation_result allocation", "handle = {}"),
    ("// Allocate and validate a replacement before disturbing the published buffer.", "ggml_sycl::allocation_result allocation", "pipe.scratch_handle[b] = {}"),
):
    with gate("runtime-replacement-keeps-old-owner region %s" % start.strip()):
        assert forbidden not in region(RUNTIME, start, end), start
print("PASS runtime-workspace-owner-first-source-gate")

# The next coherent STAGING batch closes 17 legacy sites: 13 in the runtime,
# two in unified-cache, and one each in common.hpp/common.cpp. Exact legacy
# variable names are forbidden so migrated adapters cannot silently regress.
staging_runtime_regions = (
    ("struct ggml_sycl_scoped_staging_handle", "struct ggml_sycl_f16_attention_route"),
    ("struct scoped_staging_handle", "auto set_root_override"),
    ("struct ggml_sycl_pool_host", "void ggml_sycl::L2PrefetchManagerDeleter"),
    ("static bool convert_tensor_layout", "static bool ggml_sycl_select_mul_mat_layout"),
    ("static bool ggml_sycl_ensure_moe_ptr_table", "static void ggml_sycl_update_moe_hotset"),
    ("static const int32_t * ggml_sycl_get_moe_ids_device_ptr_exact", "static bool ggml_sycl_release_moe_tensor_layout"),
    ("static bool graph_preload_moe_experts", "static void graph_unpin_moe_experts"),
    ("static bool ensure_split_persistent_resources", "static sycl::queue *            g_split_merge_queue"),
    ("static bool split_secondary_gpu_ensure", "static const void * split_secondary_weight_load"),
    ("static ggml_sycl::mem_handle ggml_sycl_block_exec_alloc_host_stage_handle", "static bool ggml_sycl_block_exec_queue_matches_device"),
)
for start, end in staging_runtime_regions:
    with gate("staging-owner-first region %s" % start.strip()):
        assert "unified_allocate_owner(" in region(RUNTIME, start, end), start

with gate('staging-legacy-names-forbidden'):
    staging_runtime = "\n".join(region(RUNTIME, a, b) for a, b in staging_runtime_regions)
    for forbidden in (
        "alloc_handle xmx_staging_owner", "alloc_handle table_owner", "alloc_handle ids_pack_owner",
        "alloc_handle device_owner", "alloc_handle q8_owner", "alloc_handle f32_owner",
        "alloc_handle second_out_owner", "alloc_handle host_stage_owner",
        "ggml_sycl_take_owned_alloc_handle(alloc",
    ):
        assert forbidden not in staging_runtime, forbidden
    for forbidden_call in (
        "unified_alloc(staging_req", "unified_alloc(req, &table_owner", "unified_alloc(req, &ids_pack_owner",
        "unified_alloc(req, &device_owner", "unified_alloc(req, &q8_owner", "unified_alloc(req, &f32_owner",
        "unified_alloc(req, &second_out_owner", "unified_alloc(req, &host_stage_owner",
    ):
        assert forbidden_call not in RUNTIME, forbidden_call

# unified-cache and pp-stage staging owners. The common.hpp one (the MMQ src1 Q8_1 buffer)
# is the shared planned-scratch helper pinned above; it is part of this seam too.
STAGING_BLOCKS = (
    ("cache reserve_reorder_temp", CACHE, "bool unified_cache::reserve_reorder_temp",
     "bool unified_cache::reserve_persistent_scratch", "reorder_temp_owner_  = std::move(replacement)"),
    ("cache fill_with_host_copy", CACHE, "bool unified_cache_fill_with_host_copy",
     "bool unified_cache_copy_from_host_async", None),
    ("common planned scratch", COMMON, PLANNED_SCRATCH_HELPER[0], PLANNED_SCRATCH_HELPER[1],
     "backing  = std::move(replacement)"),
    ("common_impl pp stage", COMMON_IMPL, "void * ggml_sycl_pp_ensure_stage_buffer",
     "sycl::event ggml_sycl_pp_stage_transfer", "g_sycl_pp_config.stage_output_handle[stage] = std::move(replacement)"),
)
for label, source, start, end, mutation in STAGING_BLOCKS:
    with gate("staging-owner-first %s" % label):
        block = region(source, start, end)
        assert "unified_allocate_owner(" in block
        assert "unified_alloc(" not in block
        assert "from_legacy_owned_alloc" not in block
    # Representative failure seam: replacement owners are resolved and routing is
    # validated before old staging metadata is published or cleared.
    if mutation:
        with gate("staging-replacement-failure-seam %s" % label):
            block = region(source, start, end)
            allocation = block.index("unified_allocate_owner(")
            resolved = block.index("resolve(", allocation)
            validation = block.index("resolved.ptr", resolved)
            publication = block.index(mutation, validation)
            assert allocation < resolved < validation < publication

with gate('staging-replacement-failure-seam split_secondary'):
    block = region(RUNTIME, "static bool split_secondary_gpu_ensure", "static const void * split_secondary_weight_load")
    allocation = block.index("unified_allocate_owner(")
    resolved = block.index("resolve(", allocation)
    validation = block.index("resolved.ptr", resolved)
    publication = block.index("g_split_secondary_gpu.q8_handle = std::move(q8_replacement)", validation)
    assert allocation < resolved < validation < publication
print("PASS staging-owner-first-source-gate-17")

with gate('staging-resize'):
    # w288: resize helpers qualify success against the requested geometry.  A
    # surviving smaller pointer is never accepted after a failed growth attempt.
    secondary = region(RUNTIME, "static bool split_secondary_gpu_ensure", "// Secondary GPU weight loading")
    assert "q8_size >= q8_bytes" in secondary
    assert "f32_size >= f32_bytes" in secondary
    secondary_call = region(RUNTIME, "// Secondary GPU: H2D src1", "// CPU vec_dot:")
    assert "!split_secondary_gpu_ensure(q8_bytes, src1_f32_bytes, second_out_bytes" in secondary_call
    assert "s_second_out_dev_sz, s_second_out_dev_handle, stream_second" in secondary_call
    persistent = region(RUNTIME, "static bool ensure_split_persistent_resources", "// OOQ merge queue")
    assert "r.q8_staging_size >= need_q8" in persistent
    assert "return false;" in persistent
    assert "if (!ensure_split_persistent_resources(" in RUNTIME
    print("PASS staging-resize-capacity-qualified-source-gate")

with gate('staging-retirement'):
    # Successful secondary replacements escrow q8/f32/output old owners as one
    # transaction behind the exact queue terminal. The retirement ticket predates
    # owner-vector growth and failed barrier/publication paths drain before unwind.
    retirement = region(RUNTIME, "class ggml_sycl_old_owner_retirement", "// Construct this before direct_stage_expert")
    # Field order (ticket before the owner vector) is the property; match on the
    # declarations rather than on their column alignment, which clang-format moves
    # whenever a neighbouring member name changes length.
    assert retirement.index("publish_ticket_ =") < retirement.index("old_owners_;")
    assert "old_owners_.reserve(owner_capacity)" in retirement
    # The queue is held by pointer so the private host fixture can drive this
    # transaction with no device; null means there is no submission to fence or
    # drain, and every production caller passes a live queue.
    assert "queue_->ext_oneapi_submit_barrier()" in retirement
    assert "retain_handles_until_event_transactional(old_owners_, prior_queue_terminal, publish_ticket_)" in retirement
    assert "ggml_sycl_drain_direct_stage_queue(*queue_)" in retirement
    for owner in ("g_split_secondary_gpu.q8_handle", "g_split_secondary_gpu.f32_handle", "output_handle"):
        assert f"retirement.hold({owner})" in secondary
    assert secondary.count("retirement.secure()") == 1
    assert secondary.index("retirement.secure()") < secondary.index("g_split_secondary_gpu.q8_handle = std::move")
    assert "secondary_queue.ext_oneapi_submit_barrier()" in persistent
    print("PASS staging-in-flight-resize-owner-retention-source-gate")

with gate('xmx-moe-atomic'):
    # XMX staging and MoE pointer-table growth are failure atomic: no old state is
    # cleared before a replacement resolves, and table payload vectors are built
    # locally before the owner/size/validity tuple is published.
    xmx_stage = region(RUNTIME, "const bool staging_too_small", "// Always use host staging")
    assert "xmx_mxfp4_tiled_aos_staging_handle[device_id] = {}" not in xmx_stage
    assert xmx_stage.index("staging_resolved && staging_resolved.on_device") < xmx_stage.index("std::move(staging_handle)")
    moe_table = region(RUNTIME, "static bool ggml_sycl_ensure_moe_ptr_table", "static void ggml_sycl_update_moe_hotset")
    # Whitespace-insensitive: this pin used to be a literal with column-aligned spaces, which a
    # clang-format realignment would silently turn into a never-matching (vacuous) negative.
    assert not re.search(r"moe_expert_ptrs_handle\[device\]\s*=\s*\{\};", moe_table)
    # The three publication sites share one constructor helper, so the
    # same-allocation host-vector failure has a single seam. The "built off to the
    # side, published only once complete" property moves with it: assert the call
    # count here and the ordering inside the helper itself.
    assert moe_table.count("ggml_sycl_build_moe_table_views(count,") == 3
    table_views = region(RUNTIME, "static void ggml_sycl_build_moe_table_views", "#if defined(GGML_SYCL_PRIVATE_TESTING)")
    assert "std::vector<ggml_sycl::mem_handle> new_handles(count)" in table_views
    assert "std::vector<void *>                new_payload(count, nullptr)" in table_views
    assert table_views.index("new_payload(count, nullptr)") < table_views.index("handles.swap(new_handles)")
    assert table_views.index("handles.swap(new_handles)") < table_views.index("payload.swap(new_payload)")
    assert moe_table.count("catch (const std::bad_alloc &)") >= 3
    assert "ggml_sycl_checked_mul_size(count, sizeof(void *), &bytes)" in moe_table
    assert "return true;" in moe_table and "return false;" in moe_table
    assert "retirement.hold(extra->weight().moe_expert_ptrs_handle[device])" in moe_table
    print("PASS xmx-moe-replacement-failure-atomic-source-gate")

with gate('overflow-safe'):
    # w295 geometry is rejected before pointer arithmetic/allocation and a surviving
    # undersized XMX pointer cannot pass the capacity gate.
    xmx_convert = region(RUNTIME, "static bool convert_tensor_layout", "static bool ggml_sycl_select_mul_mat_layout")
    assert "ggml_sycl_checked_mul_size(info.total_bytes" in xmx_convert
    assert "xmx_mxfp4_tiled_aos_staging_size[device_id] < aos_expert_size" in xmx_convert
    assert xmx_convert.index("xmx_mxfp4_tiled_aos_staging_size[device_id] < aos_expert_size") < xmx_convert.index("// Always use host staging")
    assert "ggml_sycl_checked_round_up_size(static_cast<size_t>(K), MATRIX_ROW_PADDING" in RUNTIME
    assert "ggml_sycl_checked_mul_size(q8_blocks, sizeof(block_q8_1), &q8_bytes)" in RUNTIME
    assert "ggml_sycl_checked_mul_size(static_cast<size_t>(N_second), sizeof(float), &second_out_bytes)" in RUNTIME
    assert "ggml_sycl_checked_mul_size(total_batches_size, sizeof(int32_t), &ids_bytes)" in RUNTIME
    print("PASS overflow-safe-staging-geometry-source-gate")

with gate('graph-preload'):
    # Graph-preload failure propagates into graph suppression, rather than logging
    # and continuing through a stale graph path.
    refresh = region(RUNTIME, "if (refresh_moe_after_pp)", "const int descriptor_moe_graph_candidates")
    assert "if (!graph_preload_moe_experts(*sycl_ctx, cgraph))" in refresh
    assert "sycl_ctx->moe_graphs_disabled = true" in refresh
    assert "use_sycl_graph                = false" in refresh
    assert "graph_unpin_moe_experts(sycl_ctx)" in refresh
    print("PASS graph-preload-bool-propagation-source-gate")

with gate('moe-metadata'):
    # Metadata and its derived group registry are built locally and atomically
    # swapped under both writer locks; bad_alloc preserves the old epoch.
    metadata = region(RUNTIME, "// Build both views off to the side", "// Early multi-GPU setup")
    assert "new_expert_meta" in metadata and "new_expert_groups" in metadata
    assert "catch (const std::bad_alloc &)" in metadata
    assert "std::scoped_lock lock(g_moe_expert_meta_mutex, g_expert_groups_mutex)" in metadata
    assert metadata.index("catch (const std::bad_alloc &)") < metadata.index("g_moe_expert_meta.swap")
    assert "g_expert_groups.swap(new_expert_groups)" in metadata
    print("PASS moe-metadata-atomic-publication-source-gate")

with gate('moe-paired-snapshot'):
    # Reader side of the same contract. The writer publishing both registries under
    # one scoped_lock buys nothing if readers acquire them separately: a reader that
    # consults BOTH within one logical operation can otherwise pair one model's
    # metadata with the next model's groups, and expert_group_key carries no model
    # identity to catch it. Every such reader must go through one paired snapshot.
    snapshot = region(RUNTIME, "static moe_registry_snapshot moe_snapshot_registries", "// Residency against a caller-held")
    assert "std::scoped_lock      lock(g_moe_expert_meta_mutex, g_expert_groups_mutex)" in snapshot
    assert snapshot.index("scoped_lock") < snapshot.index("snapshot.meta   = g_moe_expert_meta")
    assert snapshot.index("snapshot.meta   = g_moe_expert_meta") < snapshot.index("snapshot.groups = g_expert_groups")

    # is_expert_resident must keep a lock-free overload, or a dual reader holding a
    # paired snapshot would have to re-enter the group lock to ask about residency --
    # reading a newer epoch than the metadata it holds, and deadlocking outright if
    # the snapshot lock were still held (shared_mutex is not recursive).
    assert "static bool is_expert_resident_in(const std::unordered_map<int64_t, expert_tensor_group> & groups" in RUNTIME
    resident = region(RUNTIME, "static bool is_expert_resident(int block_id", "// Forward declarations needed by moe_prestage")
    assert "return is_expert_resident_in(g_expert_groups, block_id, expert_id, device_id)" in resident

    # The two dual-consumer readers take the paired snapshot and never re-acquire
    # either mutex for the rest of the operation.
    for start, end, name in (
        ("static void moe_prestage_popular_experts", "// SOA-correct expert caching: single-expert wrapper", "prestage"),
        ("        // We need block_num for the residency checks", "    void worker_loop()", "rebalance"),
    ):
        block = region(RUNTIME, start, end)
        assert "moe_snapshot_registries()" in block, name
        assert "g_expert_groups_mutex" not in block, name
        assert "g_moe_expert_meta_mutex" not in block, name
        assert "is_expert_resident(" not in block, name
    print("PASS moe-registry-paired-snapshot-reader-source-gate")
# HM Task 2 (llama.cpp-81gt): CACHE_BACKING classification must be unforgeable.
# `alloc_constraints.cache_backing` was a public caller-writable bool, so any
# caller could mint the ownership class that shutdown exempts from destructive
# teardown refusal. Provenance now comes from a private token whose header only
# the two legitimate mint sites may include.
PROVENANCE_HEADER = "allocation-provenance.hpp"
PROVENANCE_MINTERS = ("unified-cache.cpp", "pinned-pool.cpp")


def struct_body(source: str, name: str) -> str:
    start = source.index("struct %s {" % name)
    return source[start:source.index("\n};", start)]


def check_cache_backing_not_public(header: str) -> list:
    problems = []
    for public_struct in ("alloc_constraints", "alloc_intent", "alloc_request"):
        if "cache_backing" in struct_body(header, public_struct):
            problems.append("%s still exposes caller-writable cache_backing" % public_struct)
    return problems


def _provenance_includers(skip_dot_directories: bool) -> list:
    """Sources including the private header. skip_dot_directories is a parameter so the
    real check and its control below exercise this one walk, never two copies of it."""
    includers = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in (".c", ".cpp", ".h", ".hpp"):
            continue
        relative = path.relative_to(ROOT)
        parts = relative.parts
        if ".git" in parts or any(part.startswith("build") for part in parts):
            continue
        # Dot-prefixed scratch build dirs (.build-*) hold stale copies of these same
        # sources; descending them reports findings against files nobody is editing.
        if skip_dot_directories and any(part.startswith(".") for part in parts):
            continue
        if '#include "%s"' % PROVENANCE_HEADER in path.read_text(errors="ignore"):
            includers.append(relative.as_posix())
    return includers


def check_provenance_header_private() -> list:
    problems = []
    includers = _provenance_includers(skip_dot_directories=True)
    # Positive control: a header nobody includes would let this check pass
    # vacuously forever, so absence is a failure rather than a clean result.
    if not includers:
        problems.append("no source includes %s -- the private provenance header is missing" % PROVENANCE_HEADER)
    for included_by in includers:
        if included_by.rsplit("/", 1)[-1] not in PROVENANCE_MINTERS:
            problems.append("%s includes the private provenance header" % included_by)
    return problems


def check_dot_directory_skip_is_live() -> list:
    """The dot-skip must be the reason a scratch copy is ignored, not luck.

    Builds a throwaway dot-directory holding the private include, then asserts both
    directions: the walk WOULD flag it without the skip (so the probe is real), and
    does not flag it with the skip (so the skip is in force). Self-contained -- the
    fixture is removed here, so the gate leaves no litter behind."""
    problems = []
    probe_dir = ROOT / (".contract-dotskip-probe-%d" % os.getpid())
    try:
        probe_dir.mkdir()
        (probe_dir / "stale-copy.cpp").write_text('#include "%s"\n' % PROVENANCE_HEADER)
        probe = "%s/stale-copy.cpp" % probe_dir.name
        if probe not in _provenance_includers(skip_dot_directories=False):
            problems.append("control is void: the dot-directory probe was undetectable even without the skip")
        if probe in _provenance_includers(skip_dot_directories=True):
            problems.append("dot-directory skip is not in force: %s was scanned" % probe)
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)
    if probe_dir.exists():
        problems.append("control left its fixture behind at %s" % probe_dir)
    return problems


# CACHE_BACKING's second mint path. The unified_cache constructor's staging adopt
# passes cache_backing=true as a plain bool to unified_cache_adopt_raw_host_allocation();
# the oneDNN Graph-scratch pool's completion-flag slab (llama.cpp-c6ah) does the same,
# lazily, under the cache's own mutex. Neither can route through
# unified_allocate_owner_backing() -- that adopt would depend circularly on the
# coordinator -- so both stay bootstrap mints. Unlike the token, nothing about a bool
# parameter is unforgeable by construction, so containment is asserted here instead:
# the helper must stay TU-static (unreachable from another translation unit) AND
# exactly the two-site ALLOWLIST below may pass true, one call per cohort tag, so a
# THIRD bootstrap mint cannot be added without review -- reviewed and accepted
# as a two-site allowlist rather than flipped to EXTERNAL_EXACT; see
# allocation-provenance.hpp and
# docs/design/sycl-canonical-memory-architecture.md section 3.1.
ADOPT_MINT_HELPER = "unified_cache_adopt_raw_host_allocation"
ADOPT_CACHE_BACKING_ARG = 6  # 0-based: ptr, size, queue, role, category, cohort_id, cache_backing
ADOPT_COHORT_ARG = 5  # 0-based: ptr, size, queue, role, category, cohort_id
ADOPT_CACHE_BACKING_ALLOWLIST = (
    '"unified_cache:staging"',
    '"unified_cache:onednn_graph_scratch_flag_slab"',
)


def _call_argument_lists(source: str, function: str) -> list:
    """Argument text of every CALL to `function`, skipping its declaration/definition."""
    calls = []
    for match in re.finditer(r"\b%s\s*\(" % function, source):
        if re.search(r"alloc_handle[ \t]+$", source[max(0, match.start() - 40):match.start()]):
            continue  # a signature, not a call
        depth, index = 1, match.end()
        while index < len(source) and depth:
            if source[index] == "(":
                depth += 1
            elif source[index] == ")":
                depth -= 1
            index += 1
        calls.append(source[match.end():index - 1])
    return calls


def _split_top_level_arguments(argument_text: str) -> list:
    arguments, depth, current, in_string = [], 0, "", False
    for char in argument_text:
        if char == '"':
            in_string = not in_string
        if not in_string:
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
            elif char == "," and depth == 0:
                arguments.append(current.strip())
                current = ""
                continue
        current += char
    if current.strip():
        arguments.append(current.strip())
    return arguments


def check_internal_backing_mint_stays_private(cache_cpp: str) -> list:
    problems = []
    # Scan code only. Comments in this file name the helper while explaining it.
    cache_cpp = _blank_comments(cache_cpp)
    signatures = re.findall(
        r"^[ \t]*(static[ \t]+)?alloc_handle[ \t]+%s[ \t]*\(" % ADOPT_MINT_HELPER,
        cache_cpp,
        re.MULTILINE,
    )
    # Positive control: the forward declaration and the definition. If the symbol
    # is renamed or removed this count drops and the gate fails loudly instead of
    # silently vouching for a helper it can no longer see.
    if len(signatures) != 2:
        problems.append(
            "expected 2 %s signatures (declaration + definition), found %d" % (ADOPT_MINT_HELPER, len(signatures)))
    if any(not qualifier for qualifier in signatures):
        problems.append(
            "%s is no longer TU-static -- an exported backing mint is a forgeable authority" % ADOPT_MINT_HELPER)

    calls = _call_argument_lists(cache_cpp, ADOPT_MINT_HELPER)
    # Positive control again: zero calls means the scanner stopped matching real
    # code, not that the codebase became safe.
    if not calls:
        problems.append("found no call to %s -- the call-site scan matched nothing" % ADOPT_MINT_HELPER)
    minting_cohorts = []
    for argument_text in calls:
        arguments = _split_top_level_arguments(argument_text)
        if len(arguments) > ADOPT_CACHE_BACKING_ARG and arguments[ADOPT_CACHE_BACKING_ARG] == "true":
            cohort = arguments[ADOPT_COHORT_ARG] if len(arguments) > ADOPT_COHORT_ARG else "<missing>"
            minting_cohorts.append(cohort)
    if sorted(minting_cohorts) != sorted(ADOPT_CACHE_BACKING_ALLOWLIST):
        problems.append(
            "expected exactly the 2 allowlisted bootstrap mints (cache_backing=true) to %s -- one call per "
            "cohort tag %s -- found cohorts %s" %
            (ADOPT_MINT_HELPER, list(ADOPT_CACHE_BACKING_ALLOWLIST), minting_cohorts))
    return problems



# Four independent checks: a failure of one must not hide the others.
with gate('cache-backing-not-public'):
    problems = check_cache_backing_not_public(CACHE_HPP)
    assert not problems, "\n".join(problems)
with gate('provenance-header-private'):
    problems = check_provenance_header_private()
    assert not problems, "\n".join(problems)
with gate('dot-directory-skip-is-live'):
    problems = check_dot_directory_skip_is_live()
    assert not problems, "\n".join(problems)
with gate('internal-backing-mint-stays-private'):
    problems = check_internal_backing_mint_stays_private(CACHE)
    assert not problems, "\n".join(problems)

finish(min_checks=61)
