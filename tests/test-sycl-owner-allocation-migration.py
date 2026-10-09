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

# unified_allocate_owner( in ggml-sycl.cpp is pinned as a baseline count plus a NAMED-site list. A site that is
# not one of the baseline runtime sites is added as one (function, reason) line in NAMED_OWNER_FIRST_SITES below,
# never by raising the baseline: a count says nothing about WHICH site was added or why it is legitimate, and the
# list is a plain per-line merge when two branches each add a site. The census counts the code with every named
# function's body cut out, so the baseline stays at the reviewed 24 runtime sites. Each name must be listed once.
# Each named function must hold exactly one owner-first allocation, refuse on failure (the `if (!allocation)` branch
# must contain a return or throw token -- textual, so a return in a nested lambda or a string literal satisfies it --
# or, braceless, be a return or throw statement itself; an empty one does not count), and hand the owner over only
# through mem_handle::from_owned_alloc, with no legacy unified_alloc( / from_legacy_owned_alloc in it. A function named
# ggml_backend_sycl_test_* is a PRIVATE_TESTING seam and must sit inside an `#if defined(GGML_SYCL_PRIVATE_TESTING)`
# block; several hooks may share one block. A function with any other name is production code, needs no guard, and
# is not required to have one. The next adder is the L4 lane (impl/moua-l4).
OWNER_FIRST_BASELINE_SITES = 24
NAMED_OWNER_FIRST_SITES = (
    ("ggml_backend_sycl_test_park_tenant_staging",
     "zhcn C7a (cec4a5f10): parks one host-pinned STAGING entry under a tenant cohort for the replay-only-call "
     "test; owner-first, handed over via from_owned_alloc"),
    ("ggml_sycl_reserve_host_tenants",
     "moua L4 step 3c: the host tier of a context's tenants -- one owner-first carve per host slot (must_host_pinned, "
     "pinned pool, category HOST_COMPUTE, the cohort's own name), held by the registry entry that carries the table, "
     "so no raw pointer or side cache holds the room; refuses on failure, handed over via from_owned_alloc"),
)


def function_body_span(code: str, name: str):
    """(start of the definition, end of its closing brace) of the one function `name` defined in `code`."""
    # A definition is not a call: `if (!name(...)) {` also ends in `) {`, so a head preceded by `!` or `(` is a call.
    heads = list(re.finditer(r"(?<![!(])\b%s\([^;{}]*\)\s*(?:noexcept\s*)?\{" % re.escape(name), code))
    assert len(heads) == 1, "%s: expected one definition, found %d" % (name, len(heads))
    start = heads[0].start()
    depth, index = 0, heads[0].end() - 1
    while True:
        depth += (code[index] == "{") - (code[index] == "}")
        index += 1
        if depth == 0:
            return start, index


def inside_private_testing(code: str, position: int) -> bool:
    """True when `position` sits in an `#if defined(GGML_SYCL_PRIVATE_TESTING)` / `#ifdef` block that no #else/#elif
    has turned over: the preprocessor conditions open at that point are walked as a stack."""
    stack = []  # [condition text, turned over by #else/#elif]
    for line in code[:position].split("\n"):
        directive = re.match(r"\s*#\s*(\w+)\s*(.*)", line)
        if not directive:
            continue
        kind, rest = directive.group(1), " ".join(directive.group(2).split())
        if kind in ("if", "ifdef", "ifndef"):
            stack.append([("defined(%s)" % rest) if kind == "ifdef" else (None if kind == "ifndef" else rest), False])
        elif kind in ("else", "elif") and stack:
            stack[-1][1] = True
        elif kind == "endif" and stack:
            stack.pop()
    return any(cond == "defined(GGML_SYCL_PRIVATE_TESTING)" and not turned for cond, turned in stack)


def refusal_branch_refuses(body: str, start: int) -> bool:
    """The statement `if (!allocation)` that begins at `start` in `body` refuses. Either a braced block (matched from
    its own `{`) that contains a return or throw TOKEN, or a braceless single statement, up to its `;`, that is itself
    a return or throw. Textual: a return inside a nested lambda or a string literal in the block also satisfies it.
    Anything that does not parse as one of those two shapes is False, never an exception."""
    head = re.compile(r"if\s*\(\s*!\s*allocation\s*\)\s*").match(body, start)
    if head is None:
        return False
    index = head.end()
    if index >= len(body):
        return False
    if body[index] == "{":
        depth, end = 0, index
        while end < len(body):
            depth += (body[end] == "{") - (body[end] == "}")
            end += 1
            if depth == 0:
                return re.search(r"\b(return|throw)\b", body[index:end]) is not None
        return False
    semicolon = body.find(";", index)
    if semicolon < 0:
        return False
    return re.match(r"(return|throw)\b", body[index:semicolon]) is not None


RUNTIME_PRODUCTION_CODE = RUNTIME_CODE
with gate("every named owner-first site in ggml-sycl.cpp is a guarded, owner-first site"):
    _names      = [name for name, _reason in NAMED_OWNER_FIRST_SITES]
    _duplicates = sorted({name for name in _names if _names.count(name) > 1})
    assert not _duplicates, "NAMED_OWNER_FIRST_SITES lists %s more than once" % ", ".join(_duplicates)
    _spans = []
    for _function, _reason in NAMED_OWNER_FIRST_SITES:
        assert _reason, "%s has no reason" % _function
        _start, _end = function_body_span(RUNTIME_CODE, _function)
        _body = RUNTIME_CODE[_start:_end]
        assert _body.count("unified_allocate_owner(") == 1, "%s: not exactly one owner-first allocation" % _function
        assert "unified_alloc(" not in _body and "from_legacy_owned_alloc" not in _body, _function
        _allocation = _body.find("unified_allocate_owner(")
        _refused    = _body.find("if (!allocation)", max(_allocation, 0))
        assert _refused >= 0, "%s: no `if (!allocation)` refusal after the allocation" % _function
        _wrapped    = _body.find("from_owned_alloc(std::move(allocation.owner)", _refused)
        assert _wrapped >= 0, "%s: the owner is not handed over via from_owned_alloc after the refusal check" % _function
        assert _allocation < _refused < _wrapped, "%s: allocate, refuse-check, from_owned_alloc out of order" % _function
        assert refusal_branch_refuses(_body, _refused), "%s: the `if (!allocation)` branch does not return or throw" % _function
        if _function.startswith("ggml_backend_sycl_test_"):
            assert inside_private_testing(RUNTIME_CODE, _start), \
                "%s is a test seam outside GGML_SYCL_PRIVATE_TESTING" % _function
        _spans.append((_start, _end))
    # Cut the named bodies out of the ORIGINAL text; each name is unique, so no span is cut twice.
    _kept, _from = [], 0
    for _start, _end in sorted(_spans):
        assert _start >= _from, "named owner-first functions overlap"
        _kept.append(RUNTIME_CODE[_from:_start])
        _from = _end
    _kept.append(RUNTIME_CODE[_from:])
    RUNTIME_PRODUCTION_CODE = "".join(_kept)
    assert RUNTIME_CODE.count("unified_allocate_owner(") == \
        RUNTIME_PRODUCTION_CODE.count("unified_allocate_owner(") + len(_spans)

for _label, _code, _pins in (
    ("ggml-sycl.cpp", RUNTIME_PRODUCTION_CODE, (54, 42, OWNER_FIRST_BASELINE_SITES)),
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
    # and continuing through a stale graph path. The suppression is per split and
    # per expert-residency state (moe-graph-preload-stamp.hpp), not a sticky flag.
    # The preload runs at one site, right before a record or replay.
    assert len(re.findall(r"\bgraph_preload_moe_experts\s*\(\s*\*sycl_ctx", RUNTIME_CODE)) == 1
    site = RUNTIME[RUNTIME.index("if (!graph_preload_moe_experts(*sycl_ctx, cgraph, moe_host_tier_boundary)) {"):]
    site = site[:site.index("return GGML_STATUS_SUCCESS;")]
    assert "sycl_ctx->moe_graph_preload_refused = true" in site
    assert "graph_unpin_moe_experts(sycl_ctx)" in site
    assert "compute_impl_unlocked();" in site and "record_completion(false);" in site
    print("PASS graph-preload-bool-propagation-source-gate")

with gate('graph-preload-refused'):
    # A split whose MoE preload is refused runs direct: the compute entry turns its graph off, the per-split flag is
    # decided before any path can run compute_impl (the graphlet gates read it there), and every MoE graphlet gate
    # honors it.
    compute = region(RUNTIME, "static ggml_status ggml_backend_sycl_graph_compute_unchecked(",
                     "static ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend")
    decided = re.search(r"sycl_ctx->moe_graph_preload_refused\s*=\s*ggml_sycl_moe_graph_preload_decide\([^;]*\)\s*"
                        r"==\s*ggml_sycl::moe_graph_preload_split_decision::REFUSED;", compute)
    first_compute = re.search(r"\bcompute_impl(?:_unlocked)?\(\);", compute)
    assert decided and first_compute and decided.start() < first_compute.start()
    entry = re.search(r"if \(sycl_ctx->moe_graph_preload_refused\) \{\s*GGML_SYCL_DEBUG\([^;]*\);\s*"
                      r"use_sycl_graph = false;\s*\}", compute)
    assert entry and entry.start() < compute.index("const int descriptor_moe_graph_candidates")
    # The entry block reads the decision itself: nothing between them overwrites the flag.
    assert decided.end() < entry.start()
    assert not re.search(r"moe_graph_preload_refused\s*=(?!=)", compute[decided.end():entry.start()])
    assert re.search(r"!sycl_ctx->moe_direct_dispatch_graphs_disabled\s*&&\s*!sycl_ctx->moe_graph_preload_refused\s*&&"
                     r"\s*node->op\s*==\s*GGML_OP_MUL_MAT_ID", RUNTIME)
    assert re.search(r"sycl_ctx->graphs_disabled\s*\|\|\s*sycl_ctx->moe_graph_preload_refused\s*\|\|\s*"
                     r"sycl_ctx->moe_graphs_disabled\s*\|\|\s*sycl_ctx->moe_sequence_graphs_disabled", RUNTIME)
    assert re.search(r"if \(sycl_ctx->moe_block_graphs_disabled\s*\|\|[^{};]*\bsycl_ctx->moe_graph_preload_refused\b"
                     r"[^{};]*\)\s*\{", RUNTIME)
    print("PASS graph-preload-refused-split-source-gate")

with gate('graph-preload-stamp-sites'):
    # A stamped all-host tensor skips the preload before any per-tensor work (host residency, layout selection, the
    # per-expert route probe); otherwise the skip saves nothing. Only the route-probe refusal is structural, so every
    # other preload failure goes through the bounded transient retry.
    impl = region(RUNTIME, "static bool graph_preload_moe_experts_impl(",
                  "static bool graph_preload_moe_experts(ggml_backend_sycl_context & ctx")
    skip = re.search(r"moe_graph_preload_stamp_skips_tensor\([^;{]*\)\s*\{\s*continue;", impl)
    assert skip
    for later in (r"\bbool\s+host_weights\s*=", r"\bggml_sycl_select_moe_planned_graph_layout\(",
                  r"\bggml_sycl_select_moe_graph_layout\(", r"\bggml_sycl_probe_moe_planned_layout\("):
        first = re.search(later, impl)
        assert first and skip.end() < first.start(), later
    assert RUNTIME_CODE.count("moe_graph_preload_failure::STRUCTURAL") == 1
    assert re.search(r"cannot be represented as one current-device[^}]*\}\s*"
                     r"\*failure = ggml_sycl::moe_graph_preload_failure::STRUCTURAL;\s*return false;", impl)
    assert RUNTIME_CODE.count("moe_graph_preload_stamp_failure(") == 1
    assert "moe_graph_preload_outcome::REFUSED" not in RUNTIME_CODE
    wrapper = region(RUNTIME, "static bool graph_preload_moe_experts(ggml_backend_sycl_context & ctx",
                     "\n}\n")
    assert re.search(r"moe_graph_preload_failure\s+failure\s*=\s*ggml_sycl::moe_graph_preload_failure::TRANSIENT;",
                     wrapper)
    assert "moe_graph_preload_stamp_failure(" in wrapper

    # Post-prompt work: each split claims its own slot before the helpers it gates, and only a prompt split
    # advances the epoch those claims compare against.
    compute = region(RUNTIME, "static ggml_status ggml_backend_sycl_graph_compute_unchecked(",
                     "static ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend")
    prepare = re.search(r"const bool post_prompt_prepare_due =\s*cached_is_decode && "
                        r"ggml_sycl_moe_post_prompt_claim\(cgraph, sycl_ctx->device, /\*refresh=\*/false\);", compute)
    assert prepare
    for helper in ("ggml_sycl_materialize_prompt_down_i8_before_decode(",
                   "ggml_sycl_release_prompt_down_soa_before_decode("):
        assert compute.count(helper) == 1, helper
        assert "if (post_prompt_prepare_due && " + helper in compute, helper
        assert prepare.end() < compute.index(helper), helper
    refresh_claim = re.search(r"const bool post_prompt_refresh_due =\s*cached_is_decode && "
                              r"ggml_sycl_moe_post_prompt_claim\(cgraph, sycl_ctx->device, /\*refresh=\*/true\);",
                              compute)
    hotset = "if (post_prompt_refresh_due && ggml_sycl_materialize_moe_down_i8_hotset("
    assert refresh_claim and compute.count("ggml_sycl_materialize_moe_down_i8_hotset(") == 1 and hotset in compute
    assert refresh_claim.end() < compute.index(hotset)
    assert RUNTIME_CODE.count("g_moe_prompt_epoch.fetch_add(") == 1
    assert re.search(r"if \(!cached_is_decode\) \{\s*g_moe_post_pp_preload_pending\.store\(true, "
                     r"std::memory_order_release\);\s*g_moe_prompt_epoch\.fetch_add\(1, std::memory_order_acq_rel\);",
                     compute)
    print("PASS graph-preload-stamp-sites-source-gate")

with gate('graph-preload-not-in-refresh'):
    # The post-prompt refresh does not run the pointer-table preload. Its tables and leases serve a recorded graph,
    # and the graph path prepares them right before every record or replay; direct dispatch and the descriptor
    # graphlets build their own full-local tables. Run eagerly on the first decode token it cost one host-blocking
    # table rebuild per MoE split per prompt, with or without graphs.
    compute = region(RUNTIME, "static ggml_status ggml_backend_sycl_graph_compute_unchecked(",
                     "static ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend")
    refresh = region(region(RUNTIME_CODE, "static ggml_status ggml_backend_sycl_graph_compute_unchecked(",
                            "static ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend"),
                     "if (refresh_moe_after_pp || post_prompt_refresh_due)", "const int descriptor_moe_graph_candidates")
    assert not re.search(r"\bgraph_preload_moe_experts\s*\(", refresh)
    assert not re.search(r"\bggml_sycl_moe_graph_preload_decide\s*\(", refresh)
    assert "ggml_sycl_materialize_moe_down_i8_hotset(" in refresh and "moe_prestage_popular_experts();" in refresh
    print("PASS graph-preload-not-in-refresh-source-gate")

with gate('decode-env-reads-once'):
    # Both predicates are asked on every decode call; their environment terms are read once.
    # The runtime term is tested first, so the env terms (some log when first evaluated) run only when it is false.
    capture = region(RUNTIME, "static bool persistent_tg_moe_descriptor_capture_enabled() {", "\n}\n")
    runtime = re.search(r"if \(g_moe_descriptor_capture_decode_phase && moe_layer_descriptor_executor_enabled\(\)\)\s*"
                        r"\{\s*return true;\s*\}", capture)
    static = re.search(r"static const bool\s+\w+\s*=\s*\[\]", capture)
    assert runtime and static and runtime.end() < static.start()
    lam = capture[static.start():].split("}();")[0]
    for term in ("ggml_sycl::env_persistent_tg_enabled()", "moe_graphlet_probe_enabled()",
                 "moe_block_graphlet_descriptor_capture_enabled()", "moe_descriptor_capture_probe_enabled()",
                 'std::getenv("GGML_SYCL_PERSISTENT_TG_LOG_POLICY")'):
        assert term in lam, term
    size = region(RUNTIME, "static int moe_block_graphlet_requested_size(int device) {", "\n}\n")
    assert re.search(r"static const int\s+\w+\s*=\s*\[\]", size)
    assert size.index("static const int") < size.index('std::getenv("GGML_SYCL_MOE_BLOCK_GRAPHLETS")')
    # The non-graph decode path hashes the graph only when block graphlets can run.
    compute = region(RUNTIME, "static ggml_status ggml_backend_sycl_graph_compute_unchecked(",
                     "static ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend")
    direct = region(compute[compute.rindex("bool block_graphlet_executed = false;"):],
                    "bool block_graphlet_executed = false;", "if (!block_graphlet_executed)")
    sized = re.search(r"moe_block_graphlet_requested_size\(sycl_ctx->device\)\s*>\s*0", direct)
    assert sized and sized.start() < direct.index("ggml_sycl_graph_signature(cgraph)")
    # Skipping the try still records the reject it would have recorded.
    try_fn = region(RUNTIME, "static bool moe_graph_try_block_graphlets(", "\n}\n")
    assert 'ggml_sycl_moe_aggregation_diag(sycl_ctx, "block-graphlet", "reject", "disabled")' in try_fn
    assert re.search(r"\}\s*else if \(cached_is_decode\)\s*\{[^{}]*"
                     r"sycl_ctx->moe_aggregation_last_decision\s*=\s*\"block-graphlet\";\s*"
                     r"sycl_ctx->moe_aggregation_last_reject\s*=\s*\"disabled\";\s*\}", direct)
    print("PASS decode-env-reads-once-source-gate")

with gate('futile-context-direct'):
    # Once replay is futile for a context (sticky: nothing resets exec_graph_replay_futile), every later call takes the
    # GGML_SYCL_DISABLE_GRAPH=1 path, decided before any of the per-call graph-policy scans run.
    compute = region(RUNTIME_CODE, "static ggml_status ggml_backend_sycl_graph_compute_unchecked(",
                     "static ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend")
    # llama.cpp-7pm2: the futility verdict belongs to the whole-graph slot. A futile context takes the
    # DISABLE_GRAPH path unless the split is a decode split that keyed segment slots serve (the context is in
    # segmented MoE mode, moe_graph_rerecord, or this split's MUL_MAT_ID puts it there); that one exception reopens
    # the policy pipeline below for keyed splits, by design, and keyed-segment-slots pins what it may then skip.
    futile_arm = re.compile(r"\}\s*else if \(sycl_ctx->exec_graph_replay_futile &&\s*"
                            r"!moe_segment_keyed_reachable\(sycl_ctx, probe_key, cached_is_decode\)\)\s*\{"
                            r"\s*use_sycl_graph\s*=\s*false;\s*\}\s*else if \(sycl_ctx->exec_graph\)\s*\{")
    futile = futile_arm.search(compute)
    assert futile
    for _label, _pattern, _repl in (
            ("the futile arm has no keyed exception (prompt and dense splits would lose it too)",
             r"exec_graph_replay_futile &&\s*!moe_segment_keyed_reachable\(sycl_ctx, probe_key, cached_is_decode\)\) \{",
             "exec_graph_replay_futile) {"),
            ("the futile arm is bypassed outright", r"exec_graph_replay_futile &&\s*!moe_segment_keyed_reachable\(",
             "exec_graph_replay_futile && !true && !moe_segment_keyed_reachable("),
            ("a futile context still enables graphs",
             r"(moe_segment_keyed_reachable\(sycl_ctx, probe_key, cached_is_decode\)\) \{\s*)use_sycl_graph = false;",
             r"\1use_sycl_graph = true;")):
        assert len(re.findall(_pattern, compute)) == 1, "control %r anchor" % _label
        assert futile_arm.search(re.sub(_pattern, _repl, compute, count=1)) is None, \
            "control %r was not caught" % _label

    def keyed_reachable_problems(code):
        body = region(code, "static bool moe_segment_keyed_reachable(", "\n}\n")
        problems = []
        if not re.search(r"if \(!is_decode \|\| !ggml_sycl_segmented_graph_env_allows\(\) \|\| "
                         r"ctx->moe_segment_slots\.churned\(\)\)\s*\{\s*return false;\s*\}", body):
            problems.append("the exception is not limited to decode, the segmented env, and an unchurned cache")
        if not re.search(r"if \(ctx->moe_graph_rerecord\)\s*\{\s*return true;\s*\}\s*"
                         r"return ctx->moe_segment_keyed_probes\.take\(probe\.split_id, probe\.residency\);", body):
            problems.append("the exception does not require segmented MoE mode after one probe per MUL_MAT_ID split "
                            "and residency (a context-wide probe can be spent by a vetoed split; a probe without the "
                            "residency stays spent after the change that lifts its veto; none at all pays the scans "
                            "every call)")
        # A spent probe costs a memo search, not a walk: the key comes from the preload decision's scan, which runs
        # on every call anyway, and reaches the probe through one local.
        if re.search(r"\bcgraph\b|\bfor \(|\bwhile \(", body):
            problems.append("the probe walks the graph on every futile decode call")
        try:
            decide = region(code, "static ggml_sycl::moe_graph_preload_split_decision ggml_sycl_moe_graph_preload_decide(",
                            "\n}\n")
            compute_fn = region(code, "static ggml_status ggml_backend_sycl_graph_compute_unchecked(",
                                "static ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend")
        except ValueError as error:
            return problems + [str(error)]
        call = re.search(r"moe_segment_probe_key probe_key;\s*sycl_ctx->moe_graph_preload_refused =\s*"
                         r"ggml_sycl_moe_graph_preload_decide\(cgraph, sycl_ctx->device, moe_host_tier_boundary, "
                         r"&probe_key\)", compute_fn)
        use = compute_fn.find("moe_segment_keyed_reachable(sycl_ctx, probe_key, cached_is_decode)")
        if not call or use < call.end() or len(re.findall(r"\bprobe_key\b", compute_fn)) != 3:
            problems.append("the probe does not read the key the preload decision filled for this call")
        loop = decide.find("for (int i = 0; i < cgraph->n_nodes; i++) {")
        if not re.search(r"\*probe = moe_segment_probe_key\{\};", decide[:loop] if loop >= 0 else ""):
            problems.append("a split without a MUL_MAT_ID keeps the previous call's split id")
        if not re.search(r"if \(probe->split_id == 0\)\s*\{\s*[^{}]*"
                         r"h\.mix\(static_cast<uint64_t>\(cgraph->n_nodes\)\);\s*"
                         r"h\.mix_name\(node->name, sizeof\(node->name\)\);\s*probe->split_id = h\.value\(\) \| 1;",
                         decide) or not re.search(r"probe->residency = residency\.value\(\);\s*"
                                                  r"return ggml_sycl::moe_graph_preload_split_decide\(scan\);\s*$", decide):
            problems.append("the split id does not name the split by its node count and first MUL_MAT_ID "
                            "(equal-size splits of different layers would share one probe)")
        # The residency is what re-opens a refused preload (moe_graph_preload_stamp_current), plus the prompt epoch.
        for what, pattern in (("the prompt epoch", r"residency\.mix\(g_moe_prompt_epoch\.load\("),
                              ("the replan epoch", r"residency\.mix\(in\.replan_epoch\);"),
                              ("each MUL_MAT_ID's expert storage generation", r"residency\.mix\(in\.storage_generation\);")):
            if not re.search(pattern, decide):
                problems.append("the probe's residency leaves out %s, so a spent probe survives its change" % what)
        # And nothing else: a per-call value mixed in (a clock, the cgraph address) re-opens every probe on every call,
        # which is the per-call policy scan the memo exists to stop. The hasher is fed exactly these three inputs,
        # in this order, and read once.
        uses = re.findall(r"\bresidency\.(\w+)\(([^;]*)\);", decide)
        if uses != [("mix", "g_moe_prompt_epoch.load(std::memory_order_acquire)"), ("mix", "in.replan_epoch"),
                    ("mix", "in.storage_generation"), ("value", "")] or \
                len(re.findall(r"\bresidency\b", decide)) != 6:
            problems.append("the probe's residency mixes something besides the prompt epoch, the replan epoch and the "
                            "storage generations, so a per-call value can re-open every probe on every call")
        return problems

    assert not keyed_reachable_problems(RUNTIME_CODE), keyed_reachable_problems(RUNTIME_CODE)
    _reach = region(RUNTIME_CODE, "static bool moe_segment_keyed_reachable(", "\n}\n")
    _take = "return ctx->moe_segment_keyed_probes.take(probe.split_id, probe.residency);"
    for _label, _old, _new in (
            ("prompt splits reach keyed slots", "if (!is_decode || ", "if ("),
            ("a churned cache still reaches them", " || ctx->moe_segment_slots.churned())", ")"),
            ("any split reaches them every call", _take, "return true;"),
            ("a vetoed split spends the context's probe", "take(probe.split_id, ", "take(1, "),
            ("the memo survives the residency change", "take(probe.split_id, probe.residency)", "take(probe.split_id, 0)"),
            ("the probe is never consulted", _take, "return probe.split_id != 0;"),
            ("the probe walks the graph on every call", _take,
             "uint64_t n = 0;\n    for (int i = 0; i < 4; ++i) {\n        n += i;\n    }\n    " + _take)):
        assert _reach.count(_old) == 1, _label
        assert keyed_reachable_problems(RUNTIME_CODE.replace(_reach, _reach.replace(_old, _new))), \
            "control %r was not caught" % _label
    _decide = region(RUNTIME_CODE, "static ggml_sycl::moe_graph_preload_split_decision ggml_sycl_moe_graph_preload_decide(",
                     "\n}\n") + "\n}\n"
    _compute_fn = region(RUNTIME_CODE, "static ggml_status ggml_backend_sycl_graph_compute_unchecked(",
                         "static ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend")
    for _label, _where, _old, _new in (
            ("equal-size splits of different layers share a probe", _decide,
             "h.mix_name(node->name, sizeof(node->name));", ""),
            ("the node count is not in the split id", _decide, "h.mix(static_cast<uint64_t>(cgraph->n_nodes));", ""),
            ("a split without a MUL_MAT_ID keeps the last split's id", _decide, "*probe = moe_segment_probe_key{};", ""),
            ("a split without a MUL_MAT_ID gets a probe", _decide, "    return ggml_sycl::moe_graph_preload_split_decide",
             "    probe->split_id |= 1;\n    return ggml_sycl::moe_graph_preload_split_decide"),
            ("the memo survives a storage change", _decide, "residency.mix(in.storage_generation);", ""),
            ("the memo survives a replan", _decide, "residency.mix(in.replan_epoch);", ""),
            ("the memo survives a new prompt", _decide, "residency.mix(g_moe_prompt_epoch.load(", "(void) ("),
            ("a per-call clock re-opens every probe", _decide, "residency.mix(in.storage_generation);",
             "residency.mix(in.storage_generation);\n        residency.mix(static_cast<uint64_t>("
             "std::chrono::steady_clock::now().time_since_epoch().count()));"),
            ("the cgraph address re-opens every probe", _decide, "    probe->residency = residency.value();",
             "    residency.mix(reinterpret_cast<uintptr_t>(cgraph));\n    probe->residency = residency.value();"),
            ("a name is mixed into the residency", _decide, "residency.mix(in.replan_epoch);",
             "residency.mix(in.replan_epoch);\n        residency.mix_name(node->name, sizeof(node->name));"),
            ("the residency is fed through another call", _decide, "residency.mix(in.storage_generation);",
             "residency.mix(in.storage_generation);\n        mix_into(residency, i);"),
            ("the probe reads a stale key", _compute_fn, ", &probe_key)", ", &stale_key)"),
            ("the probe recomputes its key with a walk", _compute_fn,
             "moe_segment_keyed_reachable(sycl_ctx, probe_key, cached_is_decode)",
             "moe_segment_keyed_reachable(sycl_ctx, moe_segment_probe_key_of(cgraph), cached_is_decode)")):
        assert _where.count(_old) == 1, _label
        _mut = RUNTIME_CODE.replace(_where, _where.replace(_old, _new), 1)
        assert _mut != RUNTIME_CODE, "control %r did not apply" % _label
        assert keyed_reachable_problems(_mut), "control %r was not caught" % _label
    graph_branch = compute.index("\n    if (use_sycl_graph) {\n") + 1
    scans = ("check_graph_compatibility(*sycl_ctx, cgraph)", "ggml_sycl_graph_has_host_inputs(cgraph)",
             "ggml_sycl_graph_has_op(cgraph, GGML_OP_FLASH_ATTN_EXT)",
             "moe_graph_descriptor_moe_dispatch_candidate_count(sycl_ctx, cgraph)",
             "moe_decode_segmented_graph_profitable(cgraph)", "moe_decode_segmented_graph_analyze(cgraph)")
    for scan in scans:
        assert futile.end() < compute.index(scan) < graph_branch, scan
    # The signature hash and the exec-graph key are first computed inside the graph branch, never on the way to it.
    for call in ("ggml_sycl_graph_signature(cgraph)", "sycl_exec_graph_make_key("):
        assert graph_branch < compute.index(call, futile.end()), call
    # Every scan between the decision and the graph branch is skipped for a futile context: either it needs
    # use_sycl_graph, or it is guarded on the flag itself.
    policy = compute[futile.end():graph_branch]
    assert "use_sycl_graph && cached_is_decode && ggml_sycl_has_global_plan() && ggml_sycl_graph_has_host_inputs(cgraph)" in policy
    assert re.search(r"decode_has_flash_attn_ext\s*=\s*cached_is_decode && !sycl_ctx->exec_graph_replay_futile &&\s*"
                     r"ggml_sycl_graph_has_op\(cgraph, GGML_OP_FLASH_ATTN_EXT\)", policy)
    assert re.search(r"moe_graphlet_replay_probe\s*=\s*cached_is_decode && !sycl_ctx->exec_graph_replay_futile &&\s*"
                     r"moe_graphlet_replay_probe_enabled\(\)", policy)
    # A keyed split records no descriptor MoE graphs, so it skips the candidate count too.
    assert re.search(r"\(use_sycl_graph && cached_is_decode && !moe_graphlet_replay_probe && !moe_segment_keyed_policy\)"
                     r"\s*\?\s*moe_graph_descriptor_moe_dispatch_candidate_count\(sycl_ctx, cgraph\)", policy)
    for m in re.finditer(r"moe_decode_segmented_graph_(?:profitable|analyze)\(cgraph\)", policy):
        guard = policy[policy.rindex("if (", 0, m.start()):m.start()]
        assert guard.startswith("if (use_sycl_graph && "), guard
    # The decode no-graph diagnostic names the futility gate once, ahead of the branch that rescans for other reasons.
    reason = policy.index("if (!use_sycl_graph && cached_is_decode && sycl_ctx->exec_graph_replay_futile) {")
    rest = policy.index("} else if (!use_sycl_graph && cached_is_decode) {", reason)
    assert "replay futility gate tripped for this context" in policy[reason:rest]
    assert "ggml_sycl_graph_has_host_inputs" not in policy[reason:rest]
    # The no-graph path's block graphlets record command graphs, so a replay-futile context never tries them.
    direct = region(compute[compute.rindex("bool block_graphlet_executed = false;"):],
                    "bool block_graphlet_executed = false;", "if (!block_graphlet_executed)")
    assert re.search(r"if \(cached_is_decode && !sycl_ctx->exec_graph_replay_futile &&\s*"
                     r"moe_block_graphlet_requested_size\(sycl_ctx->device\)\s*>\s*0\)", direct)
    print("PASS futile-context-direct-source-gate")

SEG_FLUSH_DEF = "static void moe_graph_segment_boundary_flush(const ggml_cgraph * cgraph, int start, int end, int device) {"
SEG_FLUSH_CALLS = {
    "record": "moe_graph_segment_boundary_flush(cgraph, seg.start, seg.end, sycl_ctx->device);",
    "replay": "moe_graph_segment_boundary_flush(cgraph, seg.start_node, seg.end_node, sycl_ctx->device);",
    "keyed record": "moe_graph_segment_boundary_flush(cgraph, item.start, item.end, sycl_ctx->device);",
    "keyed replay": "moe_graph_segment_boundary_flush(cgraph, seg.start_node, seg.end_node, sycl_ctx->device);",
}


def check_segment_boundary_flush(code: str) -> list:
    """llama.cpp-7pm2: a recorded segment runs no per-node host code, so the segment boundary owns the
    pending-slot flush. It runs before the queue enters recording mode (a flush inside the recording would
    capture a pinned-pool H2D and a host join into the graph) and before every recorded segment is replayed
    (the replay would otherwise read a CPU MoE output before its H2D scatter lands). It runs the direct path's
    per-node consumer checks over exactly the segment's nodes, so a slot the segment does not read stays
    deferred and keeps overlapping the GPU work after it. Direct segments keep the per-node flushes.
    `code` is ggml-sycl.cpp with comments blanked."""
    problems = []
    try:
        helper = region(code, SEG_FLUSH_DEF, "\n}\n")
        record = region(code, "static bool moe_graph_record_segments(", "struct moe_decode_segmented_graph_stats {")
        replay = region(code, "static void moe_graph_replay_segments(", "static bool graph_prestage_or_decline(")
        keyed_record = region(code, "static void moe_graph_record_segment_slot(", "static void moe_graph_replay_segment_slot(")
        keyed_replay = region(code, "static void moe_graph_replay_segment_slot(", "static bool moe_graph_record_segments(")
    except ValueError as error:
        return [str(error)]
    # The helper walks the segment's node range and asks each node what it consumes, as the direct path does.
    loop = re.search(r"for \(int i = start; i < end; \+\+i\) \{\s*const ggml_tensor \* node = cgraph->nodes\[i\];"
                     r"\s*if \(!node \|\| ggml_sycl_is_noop\(node\)\) \{\s*continue;\s*\}([\s\S]*?)\n    \}", helper)
    if not loop:
        problems.append("boundary flush helper does not walk the segment's nodes [start, end) skipping no-ops")
    checks = re.findall(r"flush_pending_(\w+?)_if_consumed\(node, device\);", loop.group(1) if loop else "")
    direct = set(re.findall(r"flush_pending_(\w+?)_if_consumed\(dst, ctx\.device\);", code))
    if not direct:
        problems.append("the direct path's per-node consumer flushes were not found")
    for slot in sorted(direct - set(checks)):
        problems.append("boundary flush helper does not check the %s slot the direct path checks per node" % slot)
    # A whole-slot flush publishes results the segment does not read and ends their overlap early.
    for flush in ("flush_pending_cpu_scatter()", "pipeline_scatter_drain()",
                  "flush_pending_secondary_scatter()", "flush_pending_attn_dispatch("):
        if flush in helper:
            problems.append("boundary flush helper publishes every pending slot (%s), not only what the segment "
                            "reads" % flush)
    total = len(re.findall(r"moe_graph_segment_boundary_flush\(", code)) - 1
    if total != 4:
        problems.append("expected the boundary flush at exactly 4 sites (record, replay, keyed record, keyed replay), "
                        "found %d" % total)
    # Record: after the direct-segment branch has continued, before the segment graph exists and records.
    call = SEG_FLUSH_CALLS["record"]
    direct = record.find("if (seg_size < MIN_SEGMENT_NODES || direct_fa_segment) {")
    skip = record.find("continue;", direct)
    flush = record.find(call)
    graph = record.find("sycl_ex::command_graph seg_graph(")
    begin = record.find("seg_graph.begin_recording(")
    end = record.find("seg_graph.end_recording();")
    if min(direct, skip, flush, graph, begin, end) < 0:
        problems.append("record: an anchor is missing, or the flush does not cover the segment's nodes "
                        "(direct=%d skip=%d flush=%d graph=%d begin=%d end=%d)" % (direct, skip, flush, graph, begin, end))
    elif not (direct < skip < flush < graph < begin < end):
        problems.append("record: the flush must follow the direct-segment branch and precede the segment graph "
                        "(skip=%d flush=%d graph=%d begin=%d)" % (skip, flush, graph, begin))
    elif "moe_graph_segment_boundary_flush(" in record[begin:end] or "flush_pending_" in record[begin:end]:
        problems.append("record: a pending-slot flush runs while the queue is recording")
    # Replay: inside the recorded-segment branch, before the submission; not in the direct branch.
    call = SEG_FLUSH_CALLS["replay"]
    graphed = replay.find("if (seg.exec_graph) {")
    # Every replay submits through ggml_sycl::graph_exec_submit (graph-recorder-scope.hpp), which counts the submission
    # for graph_compute's exit; the holder-census gate refuses a bare ext_oneapi_graph() in ggml-sycl.cpp.
    submit = replay.find("ggml_sycl::graph_exec_submit(*stream, *seg.exec_graph);")
    direct_branch = replay.find("} else {", graphed)
    flush = replay.find(call)
    if min(graphed, submit, direct_branch, flush) < 0:
        problems.append("replay: an anchor is missing, or the flush does not cover the segment's nodes "
                        "(graphed=%d submit=%d else=%d flush=%d)" % (graphed, submit, direct_branch, flush))
    elif not (graphed < flush < submit < direct_branch):
        problems.append("replay: the flush must precede the recorded segment's submission "
                        "(graphed=%d flush=%d submit=%d)" % (graphed, flush, submit))
    # Keyed record: after the direct-run branch has continued, before the segment graph exists and records.
    call = SEG_FLUSH_CALLS["keyed record"]
    direct = keyed_record.find("if (!item.graph) {")
    skip = keyed_record.find("continue;", direct)
    flush = keyed_record.find(call)
    _graph = re.search(r"sycl_ex::command_graph\s+seg_graph\(", keyed_record)
    graph = _graph.start() if _graph else -1
    begin = keyed_record.find("seg_graph.begin_recording(")
    end = keyed_record.find("seg_graph.end_recording();", begin)
    if min(direct, skip, flush, graph, begin, end) < 0:
        problems.append("keyed record: an anchor is missing, or the flush does not cover the segment's nodes "
                        "(direct=%d skip=%d flush=%d graph=%d begin=%d end=%d)" % (direct, skip, flush, graph, begin, end))
    elif not (direct < skip < flush < graph < begin < end):
        problems.append("keyed record: the flush must follow the direct-run branch and precede the segment graph "
                        "(skip=%d flush=%d graph=%d begin=%d)" % (skip, flush, graph, begin))
    elif "moe_graph_segment_boundary_flush(" in keyed_record[begin:end] or "flush_pending_" in keyed_record[begin:end]:
        problems.append("keyed record: a pending-slot flush runs while the queue is recording")
    # Keyed replay: inside the recorded-segment branch, before the submission; the direct run follows `continue;`.
    call = SEG_FLUSH_CALLS["keyed replay"]
    graphed = keyed_replay.find("if (seg.exec_graph) {")
    submit = keyed_replay.find("ggml_sycl::graph_exec_submit(*stream, *seg.exec_graph);")
    leave = keyed_replay.find("continue;", graphed)
    flush = keyed_replay.find(call)
    if min(graphed, submit, leave, flush) < 0:
        problems.append("keyed replay: an anchor is missing, or the flush does not cover the segment's nodes "
                        "(graphed=%d submit=%d continue=%d flush=%d)" % (graphed, submit, leave, flush))
    elif not (graphed < flush < submit < leave):
        problems.append("keyed replay: the flush must precede the recorded segment's submission "
                        "(graphed=%d flush=%d submit=%d)" % (graphed, flush, submit))
    return problems


with gate('segment-boundary-flush'):
    problems = check_segment_boundary_flush(RUNTIME_CODE)
    assert not problems, "\n".join(problems)
    # Controls: each broken ordering, range or slot set must be reported, or a pass above proves nothing.
    _rcall = SEG_FLUSH_CALLS["record"]
    _pcall = SEG_FLUSH_CALLS["replay"]
    _kcall = SEG_FLUSH_CALLS["keyed record"]
    _record_flush = "        " + _rcall + "\n        const size_t retained_baseline"
    _replay_flush = "                " + _pcall + "\n                ggml_sycl::graph_exec_submit(*stream, *seg.exec_graph);"
    # The legacy replay is the first of the two replay sites; the keyed replay is the second.
    assert RUNTIME_CODE.count(_record_flush) == 1 and RUNTIME_CODE.count(_replay_flush) == 2
    _begin = "            seg_graph.begin_recording(*stream);\n"
    _keyed_record = region(RUNTIME_CODE, "static void moe_graph_record_segment_slot(", "static void moe_graph_replay_segment_slot(")
    _keyed_replay = region(RUNTIME_CODE, "static void moe_graph_replay_segment_slot(", "static bool moe_graph_record_segments(")
    _keyed_flush = "        " + _kcall + "\n"
    _keyed_begin = "            seg_graph.begin_recording(*stream);\n            end_guard.open = true;\n"
    assert _keyed_record.count(_keyed_flush) == 1 and _keyed_record.count(_keyed_begin) == 1
    assert _keyed_replay.count(_replay_flush) == 1
    _helper = region(RUNTIME_CODE, SEG_FLUSH_DEF, "\n}\n")

    def _in_keyed(body, old, new):
        return RUNTIME_CODE.replace(body, body.replace(old, new, 1), 1)

    def _in_helper(old, new):
        assert _helper.count(old) == 1, old
        return RUNTIME_CODE.replace(_helper, _helper.replace(old, new, 1), 1)

    _full_flush = ("    flush_pending_cpu_scatter();\n"
                   "    if (ggml_sycl_pipeline_moe_enabled()) {\n        pipeline_scatter_drain();\n    }\n"
                   "    wait_pending_secondary_scatter_events(flush_pending_secondary_scatter());\n"
                   "    flush_pending_attn_dispatch(device);\n")
    controls = {
        "record flush dropped": RUNTIME_CODE.replace(_record_flush, "        const size_t retained_baseline"),
        "record flush inside the recording": RUNTIME_CODE.replace(_record_flush, "        const size_t retained_baseline")
            .replace(_begin, _begin + "            " + _rcall + "\n"),
        "record flush covers the wrong nodes": RUNTIME_CODE.replace(
            _rcall, _rcall.replace("seg.start, seg.end", "seg.start, seg.start + 1"), 1),
        "replay flush dropped": RUNTIME_CODE.replace(_replay_flush, "                ggml_sycl::graph_exec_submit(*stream, *seg.exec_graph);", 1),
        "replay flush after the submission": RUNTIME_CODE.replace(
            _replay_flush, "                ggml_sycl::graph_exec_submit(*stream, *seg.exec_graph);\n                " + _pcall, 1),
        "keyed record flush dropped": _in_keyed(_keyed_record, _keyed_flush, ""),
        "keyed record flush inside the recording": RUNTIME_CODE.replace(
            _keyed_record, _keyed_record.replace(_keyed_flush, "", 1).replace(
                _keyed_begin, _keyed_begin + "            " + _kcall + "\n", 1), 1),
        "keyed record flush covers another segment": _in_keyed(
            _keyed_record, _kcall, _kcall.replace("item.start, item.end", "0, item.end")),
        "keyed replay flush dropped": _in_keyed(
            _keyed_replay, _replay_flush, "                ggml_sycl::graph_exec_submit(*stream, *seg.exec_graph);"),
        "keyed replay flush after the submission": _in_keyed(
            _keyed_replay, _replay_flush, "                ggml_sycl::graph_exec_submit(*stream, *seg.exec_graph);\n                " + _pcall),
        "attention slot not checked": _in_helper("        flush_pending_attn_if_consumed(node, device);\n", ""),
        "CPU scatter slots not checked": _in_helper("        flush_pending_cpu_scatter_if_consumed(node, device);\n", ""),
        "helper walks the whole graph": _in_helper("for (int i = start; i < end; ++i) {",
                                                   "for (int i = 0; i < cgraph->n_nodes; ++i) {"),
        "every pending slot published again": _in_helper("    for (int i = start;", _full_flush + "    for (int i = start;"),
    }
    for label, mutated in controls.items():
        assert mutated != RUNTIME_CODE, "control %r did not apply" % label
        assert check_segment_boundary_flush(mutated), "control %r was not caught" % label
    print("PASS segment-boundary-flush-source-gate (%d controls caught)" % len(controls))

def check_segmented_call_ends(code: str) -> list:
    """llama.cpp-7pm2: a segmented record or replay bypasses compute_impl, so it does compute_impl's per-graph
    work at both ends: the MoE topology scan first (boundary MoE nodes dispatch against this split's pairs), the
    graph-completion drain of every pending slot last (outputs are visible when graph_compute returns)."""
    problems = []
    try:
        begin = region(code, "static void moe_graph_segmented_call_begin(", "\n}\n")
        end = region(code, "static void moe_graph_segmented_call_end() {", "\n}\n")
        record = region(code, "static bool moe_graph_record_segments(", "struct moe_decode_segmented_graph_stats {")
        replay = region(code, "static void moe_graph_replay_segments(", "static bool graph_prestage_or_decline(")
        keyed_record = region(code, "static void moe_graph_record_segment_slot(", "static void moe_graph_replay_segment_slot(")
        keyed_replay = region(code, "static void moe_graph_replay_segment_slot(", "static bool moe_graph_record_segments(")
    except ValueError as error:
        return [str(error)]
    # Comment-blind text: the /*reset_precomputed=*/ annotation is whitespace here.
    if not re.search(r"moe_layer_scan_graph_topology\(\*sycl_ctx, cgraph,\s*true,\s*"
                     r"capture_moe_descriptors\);\s*g_moe_descriptor_prescanned_for_dispatch = false;", begin):
        problems.append("begin does not rescan this split's MoE topology the way compute_impl does")
    if "!g_moe_descriptor_prescanned_for_dispatch" not in begin:
        problems.append("begin captures descriptors even when the decode prescan already did")
    if "ggml_sycl_cpu_tg_flush_pending();" not in end or "flush_pending_attn_dispatch(d);" not in end:
        problems.append("end does not drain every pending slot")
    if not re.search(r"for \(int d = 0; d < GGML_SYCL_MAX_DEVICES; d\+\+\) \{\s*if \(g_pending_attn_dispatch\[d\]\.active\) \{"
                     r"\s*flush_pending_attn_dispatch\(d\);\s*\}\s*release_stale_attn_dispatch\(d\);\s*\}", end):
        problems.append("end does not publish and release every device's host-attention slot")
    for name, body, first_use in (("record", record, "moe_graph_collect_dispatch_indices(cgraph)"),
                                  ("replay", replay, "while (seg_idx <"),
                                  ("keyed record", keyed_record, "moe_graph_keyed_plan(cgraph)"),
                                  ("keyed replay", keyed_replay, "while (seg_idx <")):
        at = body.find("moe_graph_segmented_call_begin(sycl_ctx, cgraph);")
        use = body.find(first_use)
        if at < 0 or use < 0 or at > use:
            problems.append("%s: begin is missing or runs after the first dispatch decision" % name)
    if not re.search(r"moe_graph_segmented_call_end\(\);\s*return true;\s*\}\s*$", record):
        problems.append("record: the successful record does not end with the drain")
    if not re.search(r"moe_graph_segmented_call_end\(\);\s*\}\s*$", replay):
        problems.append("replay: the replay does not end with the drain")
    for name, body in (("keyed record", keyed_record), ("keyed replay", keyed_replay)):
        if not re.search(r"moe_graph_segmented_call_end\(\);\s*\}\s*$", body):
            problems.append("%s: does not end with the drain" % name)
    return problems


with gate('segmented-call-owns-graph-ends'):
    problems = check_segmented_call_ends(RUNTIME_CODE)
    assert not problems, "\n".join(problems)
    _begin_call = "    moe_graph_segmented_call_begin(sycl_ctx, cgraph);\n"
    assert RUNTIME_CODE.count(_begin_call) == 4
    _keyed_record = region(RUNTIME_CODE, "static void moe_graph_record_segment_slot(", "static void moe_graph_replay_segment_slot(")
    _keyed_replay = region(RUNTIME_CODE, "static void moe_graph_replay_segment_slot(", "static bool moe_graph_record_segments(")
    _keyed_end = "    moe_graph_segmented_call_end();\n}\n"
    assert _keyed_record.count(_begin_call) == 1 and _keyed_replay.count(_begin_call) == 1
    assert _keyed_record.count(_keyed_end) == 1 and _keyed_replay.count(_keyed_end) == 1
    _end_body = region(RUNTIME_CODE, "static void moe_graph_segmented_call_end() {", "\n}\n")
    controls = {
        "no topology rescan": re.sub(
            r"moe_layer_scan_graph_topology\(\*sycl_ctx, cgraph,\s*true,\s*capture_moe_descriptors\);",
            "(void) capture_moe_descriptors;", RUNTIME_CODE),
        "rescan keeps the stale precomputed state": re.sub(
            r"moe_layer_scan_graph_topology\(\*sycl_ctx, cgraph,\s*true,\s*capture_moe_descriptors\);",
            "moe_layer_scan_graph_topology(*sycl_ctx, cgraph, false, capture_moe_descriptors);", RUNTIME_CODE),
        "begin dropped from both": RUNTIME_CODE.replace(_begin_call, ""),
        "end drain dropped from record": RUNTIME_CODE.replace(
            "    moe_graph_segmented_call_end();\n    return true;\n}", "    return true;\n}"),
        "end drain dropped from replay": RUNTIME_CODE.replace(
            "    moe_graph_segmented_call_end();\n}\n", "}\n", 1),
        "keyed record begin dropped": RUNTIME_CODE.replace(
            _keyed_record, _keyed_record.replace(_begin_call, "", 1), 1),
        "keyed replay begin dropped": RUNTIME_CODE.replace(
            _keyed_replay, _keyed_replay.replace(_begin_call, "", 1), 1),
        "keyed record end drain dropped": RUNTIME_CODE.replace(
            _keyed_record, _keyed_record.replace(_keyed_end, "}\n", 1), 1),
        "keyed replay end drain dropped": RUNTIME_CODE.replace(
            _keyed_replay, _keyed_replay.replace(_keyed_end, "}\n", 1), 1),
        "end drain without the CPU slots": RUNTIME_CODE.replace(
            "static void moe_graph_segmented_call_end() {\n    ggml_sycl_cpu_tg_flush_pending();",
            "static void moe_graph_segmented_call_end() {"),
        "end drain stops at the first device": RUNTIME_CODE.replace(
            _end_body, _end_body.replace("d < GGML_SYCL_MAX_DEVICES;", "d < 1;", 1), 1),
        "end drain keeps stale attention slots": RUNTIME_CODE.replace(
            _end_body, _end_body.replace("        release_stale_attn_dispatch(d);\n", "", 1), 1),
        "end drain publishes only an inactive slot": RUNTIME_CODE.replace(
            _end_body, _end_body.replace("if (g_pending_attn_dispatch[d].active) {", "if (false) {", 1), 1),
    }
    for label, mutated in controls.items():
        assert mutated != RUNTIME_CODE, "control %r did not apply" % label
        assert check_segmented_call_ends(mutated), "control %r was not caught" % label
    print("PASS segmented-call-owns-graph-ends-source-gate (%d controls caught)" % len(controls))

def check_keyed_segment_slots(code: str) -> list:
    """llama.cpp-7pm2: a decode split in segmented MoE mode runs from its own keyed slot, so the single-slot
    whole-graph machinery does not apply to it: the failed-graph memo, the '#' split-copy futility trip, the futile
    early return, the MoE expert preload (a keyed slot records no MUL_MAT_ID) and the one-per-phase warmup slot.
    The legacy one-slot segmented path keeps the prompt phase only. A MUL_MAT_ID that cannot be recorded vetoes the
    whole-graph capture only: decode admits it as segmented, prompt and the dense range recorder do not."""
    problems = []
    try:
        compute = region(code, "static ggml_status ggml_backend_sycl_graph_compute_unchecked(",
                         "static ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend")
        mode = region(code, "static bool moe_segment_keyed_mode(", "\n}\n")
        env = region(code, "static bool ggml_sycl_segmented_graph_env_allows() {", "\n}\n")
        compat = region(code, "static ggml_sycl_graph_compat check_graph_compatibility(ggml_backend_sycl_context & ctx, "
                        "ggml_cgraph * cgraph) {", "\n}\n")
    except ValueError as error:
        return [str(error)]
    if not re.search(r"return is_decode && ctx->moe_graph_rerecord && ggml_sycl_segmented_graph_env_allows\(\) &&\s*"
                     r"!ctx->moe_segment_slots\.churned\(\);", mode):
        problems.append("keyed mode is not limited to decode, segmented MoE mode, the env and an unchurned cache")
    if not re.search(r"static const bool allows =\s*std::getenv\(\"GGML_SYCL_GRAPH_RERECORD\"\) == nullptr && "
                     r"std::getenv\(\"GGML_SYCL_NO_SEG_GRAPH\"\) == nullptr;\s*return allows;", env):
        problems.append("segmented replay ignores the GGML_SYCL_GRAPH_RERECORD or GGML_SYCL_NO_SEG_GRAPH opt-out")
    decided = compute.find("const bool moe_segment_keyed = moe_segment_keyed_mode(sycl_ctx, is_decode_phase);")
    exempt = (
        ("failed-graph memo", r"if \(!moe_segment_keyed && is_decode_phase && sycl_ctx->moe_graph_rerecord &&\s*"
                              r"moe_segment_recording_failed_for_graph"),
        ("'#' split-copy trip", r"if \(!moe_segment_keyed && !sycl_ctx->exec_graph_replay_futile &&\s*"
                                r"\(!sycl_ctx->exec_graph_has_scanned"),
        ("futile early return", r"if \(!moe_segment_keyed && sycl_ctx->exec_graph_replay_futile\)\s*\{"),
        ("MoE expert preload", r"if \(!moe_segment_keyed\)\s*\{\s*if \(!graph_preload_moe_experts\("),
        ("phase warmup slot", r"if \(!moe_segment_keyed && warmup_n_nodes != cgraph->n_nodes\)"),
    )
    if decided < 0:
        problems.append("graph_compute does not decide keyed mode once per call")
    for label, pattern in exempt:
        m = re.search(pattern, compute)
        if not m:
            problems.append("keyed splits are not exempt from the %s" % label)
        elif decided >= 0 and m.start() < decided:
            problems.append("the %s exemption is tested before keyed mode is decided" % label)
    if not re.search(r"if \(moe_segment_keyed\)\s*\{[\s\S]*?\}\s*else if \(is_decode_phase\)\s*\{\s*"
                     r"compute_impl_unlocked\(\);\s*\}\s*else if \(segments_match\)\s*\{", compute):
        problems.append("a decode split can reach the one-slot segmented path (keyed first, churned decode direct)")
    if not re.search(r"return mul_mat_id_not_recordable \? GGML_SYCL_GRAPH_COMPAT_SEGMENTED_ONLY : "
                     r"GGML_SYCL_GRAPH_COMPAT_WHOLE_GRAPH;\s*$", compat):
        problems.append("check_graph_compatibility does not report an unrecordable MUL_MAT_ID as segmented-only")
    if re.search(r"mul_mat_id_not_recordable = true;\s*return", compat):
        problems.append("an unrecordable MUL_MAT_ID still ends the scan early")
    if not re.search(r"use_sycl_graph\s*=\s*compat == GGML_SYCL_GRAPH_COMPAT_WHOLE_GRAPH \|\|\s*"
                     r"\(compat == GGML_SYCL_GRAPH_COMPAT_SEGMENTED_ONLY && cached_is_decode &&\s*"
                     r"ggml_sycl_segmented_graph_env_allows\(\)\);", compute):
        problems.append("the entry admits a segmented-only graph outside decode or the segmented env")
    if "check_graph_compatibility(ctx_, cgraph_) != GGML_SYCL_GRAPH_COMPAT_WHOLE_GRAPH" not in code:
        problems.append("the dense range recorder accepts a graph that is not whole-graph recordable")
    return problems


with gate('keyed-segment-slots'):
    problems = check_keyed_segment_slots(RUNTIME_CODE)
    assert not problems, "\n".join(problems)
    controls = {
        "memo applies to keyed": ("if (!moe_segment_keyed && is_decode_phase && sycl_ctx->moe_graph_rerecord &&",
                                  "if (is_decode_phase && sycl_ctx->moe_graph_rerecord &&"),
        "'#' trip applies to keyed": ("if (!moe_segment_keyed && !sycl_ctx->exec_graph_replay_futile &&",
                                      "if (!sycl_ctx->exec_graph_replay_futile &&"),
        "futile return applies to keyed": ("if (!moe_segment_keyed && sycl_ctx->exec_graph_replay_futile) {",
                                           "if (sycl_ctx->exec_graph_replay_futile) {"),
        "preload applies to keyed": ("if (!moe_segment_keyed) {\n            if (!graph_preload_moe_experts(",
                                     "{\n            if (!graph_preload_moe_experts("),
        "warmup applies to keyed": ("if (!moe_segment_keyed && warmup_n_nodes != cgraph->n_nodes)",
                                    "if (warmup_n_nodes != cgraph->n_nodes)"),
        "keyed mode outlives churn": (" &&\n           !ctx->moe_segment_slots.churned();", ";"),
        "env opt-outs ignored": ('GGML_SYCL_NO_SEG_GRAPH") == nullptr;\n    return allows;', 'GGML_SYCL_NO_SEG_GRAPH") == nullptr;\n    return true;'),
        "NO_SEG_GRAPH ignored": (' && std::getenv("GGML_SYCL_NO_SEG_GRAPH") == nullptr;', ";"),
        "GRAPH_RERECORD ignored": ('std::getenv("GGML_SYCL_GRAPH_RERECORD") == nullptr && ', ""),
        "MUL_MAT_ID vetoes segmented": ("return mul_mat_id_not_recordable ? GGML_SYCL_GRAPH_COMPAT_SEGMENTED_ONLY :",
                                        "return mul_mat_id_not_recordable ? GGML_SYCL_GRAPH_COMPAT_NONE :"),
        "prompt admits segmented-only": ("(compat == GGML_SYCL_GRAPH_COMPAT_SEGMENTED_ONLY && cached_is_decode &&",
                                         "(compat == GGML_SYCL_GRAPH_COMPAT_SEGMENTED_ONLY &&"),
        "dense recorder admits segmented-only": ("check_graph_compatibility(ctx_, cgraph_) != GGML_SYCL_GRAPH_COMPAT_WHOLE_GRAPH",
                                                 "check_graph_compatibility(ctx_, cgraph_) == GGML_SYCL_GRAPH_COMPAT_NONE"),
    }
    for label, (old, new) in controls.items():
        assert RUNTIME_CODE.count(old) == 1, "control %r anchor appears %d times" % (label, RUNTIME_CODE.count(old))
        assert check_keyed_segment_slots(RUNTIME_CODE.replace(old, new)), "control %r was not caught" % label
    _churned = re.compile(r"\}\s*else if \(is_decode_phase\)\s*\{\s*compute_impl_unlocked\(\);\s*\}\s*"
                          r"(else if \(segments_match\)\s*\{)")
    assert len(_churned.findall(RUNTIME_CODE)) == 1
    assert check_keyed_segment_slots(_churned.sub(r"} \1", RUNTIME_CODE)), \
        "control 'churned decode takes the one-slot path' was not caught"
    print("PASS keyed-segment-slots-source-gate (%d controls caught)" % (len(controls) + 1))

def check_segment_graphs_drained(code: str, common: str) -> list:
    """llama.cpp-7pm2 lifetime rule: no segment exec graph is destroyed while a replay may still run it. Retired
    keyed slots are destroyed only after a queue wait, and invalidate_moe_segments destroys the one-slot segment and
    MoE dispatch graphs only after the same wait succeeded (the epoch retire waits only where retention terminals
    exist). A slot a failed drain could not prove idle is kept for the life of the process: the retired ones, and a
    slot whose recording threw. A failed per-call drain runs that call direct."""
    problems = []
    try:
        drain = region(code, "static bool moe_segment_slots_drain_retired(", "\n}\n")
        keep = region(code, "static void moe_segment_slot_keep_alive(", "\n}\n")
        compute = region(code, "static ggml_status ggml_backend_sycl_graph_compute_unchecked(",
                         "static ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend")
        retire = region(code, "bool ggml_sycl_retire_moe_segment_slots(ggml_backend_sycl_context * ctx) noexcept {",
                        "\n}\n")
        invalidate = region(common, "bool invalidate_moe_segments() {", "\n    }\n")
    except ValueError as error:
        return [str(error)]
    if not re.search(r"if \(!other_graphs && !ctx->moe_segment_slots\.has_retired\(\)\)\s*\{\s*return true;", drain):
        problems.append("the drain skips its wait while the caller still has graphs to destroy")
    wait = drain.find("ggml_sycl_trace_queue_wait(")
    gate = re.search(r"bool\s+any_graph = other_graphs;", drain)
    skip = re.search(r"if \(!any_graph\)\s*\{\s*return true;", drain)
    if wait < 0 or not gate or not skip or not (gate.start() < skip.start() < wait):
        problems.append("the drain does not wait whenever a retired slot or the caller has a graph")
    if not re.search(r"static auto \*\s+kept\s+= new std::vector<ggml_backend_sycl_context::moe_segment_slot>\(\);", keep) \
            or keep.count("kept->push_back(std::move(slot));") != 1:
        problems.append("the keep-alive does not hold its slots for the life of the process")
    # Its callers (a failed drain in drain_retired, the record catch) are not all under the graph-compute mutex, and
    # two contexts can fail at once: the list takes its own lock around every access.
    if not re.search(r"std::lock_guard<std::mutex>\s+lock\(\*kept_mutex\);\s*"
                     r"kept->push_back\(std::move\(slot\)\);", keep) or keep.count("kept->") != 1:
        problems.append("the keep-alive list is touched without its own lock")
    # The lock is leaked like the list, so a keep-alive during static destruction never locks a destroyed mutex.
    if not re.search(r"static auto \*\s+kept_mutex\s+= new std::mutex\(\);", keep) or \
            re.search(r"static std::mutex\b", keep):
        problems.append("the keep-alive lock is a static with a destructor")
    if not re.search(r"catch \(const std::exception & exc\)[\s\S]*for \(auto & slot : retired\)\s*\{\s*"
                     r"moe_segment_slot_keep_alive\(std::move\(slot\)\);\s*\}\s*return false;", drain[wait:] if wait >= 0 else ""):
        problems.append("a failed drain destroys the retired slots instead of keeping them")
    if not re.search(r"\}\s*catch \(\.\.\.\)\s*\{\s*bool drained = true;\s*try\s*\{\s*ggml_sycl_trace_queue_wait\("
                     r"sycl_ctx->stream\(\), \"segment-slot-record-failed\"[^;]*;\s*\}\s*catch \(\.\.\.\)\s*\{\s*"
                     r"drained = false;\s*\}\s*sycl_ctx->moe_segment_slots\.record_failed\(slot_key\);\s*"
                     r"if \(!drained\)\s*\{[^{}]*?sycl_ctx->moe_graphs_disabled = true;\s*"
                     r"moe_segment_slot_keep_alive\(std::move\(slot\)\);\s*\}\s*throw;", compute):
        problems.append("a recording that throws destroys its slot's graphs after a failed drain")
    if not re.search(r"if \(!moe_segment_slots_drain_retired\(sycl_ctx\)\)\s*\{\s*sycl_ctx->moe_graphs_disabled = true;\s*"
                     r"compute_impl_unlocked\(\);\s*\}\s*else if \(slot_action == gsc::action::WARMUP\)", compute):
        problems.append("a call whose drain failed still records or replays a slot")
    if not (re.search(r"legacy_graphs = !ctx->moe_dispatch_graphs\.empty\(\);", retire) and
            re.search(r"legacy_graphs = legacy_graphs \|\| seg\.exec_graph != nullptr;", retire) and
            "return moe_segment_slots_drain_retired(ctx, legacy_graphs);" in retire):
        problems.append("the retire does not drain for the one-slot graphs its caller destroys next")
    order = [invalidate.find(t) for t in ("ggml_sycl_retire_moe_segment_slots(this)", "moe_segments.clear();",
                                          "moe_dispatch_graphs.clear();")]
    if min(order) < 0 or not (order[0] < order[1] and order[0] < order[2]):
        problems.append("invalidate_moe_segments destroys segment graphs before the drain")
    if not re.search(r"if \(!ggml_sycl_retire_moe_segment_slots\(this\)\)\s*\{\s*moe_graphs_disabled = true;\s*"
                     r"return false;\s*\}", invalidate):
        problems.append("invalidate_moe_segments destroys its graphs after a failed drain")
    return problems


with gate('segment-graphs-destroyed-after-drain'):
    problems = check_segment_graphs_drained(RUNTIME_CODE, COMMON_CODE)
    assert not problems, "\n".join(problems)
    controls = (
        ("drain skips for the caller's graphs", "runtime", r"if \(!other_graphs && !ctx", "if (!ctx"),
        ("drain ignores the caller's graphs", "runtime", r"any_graph = other_graphs;", "any_graph = false;"),
        ("retire passes no legacy graphs", "runtime", r"moe_segment_slots_drain_retired\(ctx, legacy_graphs\)",
         "moe_segment_slots_drain_retired(ctx, false)"),
        ("retire ignores segment graphs", "runtime", r"legacy_graphs = legacy_graphs \|\| seg\.exec_graph != nullptr;",
         "(void) seg;"),
        ("keep-alive frees its slots", "runtime", r"kept->push_back\(std::move\(slot\)\);", "(void) slot;"),
        ("keep-alive appends without its lock", "runtime", r"std::lock_guard<std::mutex>\s+lock\(\*kept_mutex\);\s*", ""),
        ("keep-alive locks another mutex", "runtime", r"lock\(\*kept_mutex\);", "lock(g_sycl_graph_compute_mutex);"),
        ("keep-alive lock is a static with a destructor", "runtime",
         r"static auto \*\s+kept_mutex\s+= new std::mutex\(\);",
         "static std::mutex kept_mutex_storage;\n    static auto * kept_mutex = &kept_mutex_storage;"),
        ("failed drain destroys the slots", "runtime",
         r"(for \(auto & slot : retired\)\s*\{)\s*moe_segment_slot_keep_alive\(std::move\(slot\)\);", r"\1 (void) slot;"),
        ("record catch destroys the slot after a failed drain", "runtime",
         r"(moe_graphs_disabled = true;)\s*moe_segment_slot_keep_alive\(std::move\(slot\)\);", r"\1"),
        ("record catch never sees the failed drain", "runtime", r"drained = false;", "(void) 0;"),
        ("failed per-call drain still records", "runtime",
         r"(moe_graphs_disabled = true;)\s*compute_impl_unlocked\(\);\s*\}\s*else (if \(slot_action == gsc::action::WARMUP\))",
         r"\1 }\n \2"),
        ("clear before the drain", "common",
         r"(if \(!ggml_sycl_retire_moe_segment_slots\(this\)\))", r"moe_segments.clear();\n        \1"),
        ("clear after a failed drain", "common",
         r"if \(!ggml_sycl_retire_moe_segment_slots\(this\)\)\s*\{\s*moe_graphs_disabled = true;\s*return false;\s*\}",
         "(void) ggml_sycl_retire_moe_segment_slots(this);"),
    )
    for label, which, pattern, repl in controls:
        base = RUNTIME_CODE if which == "runtime" else COMMON_CODE
        assert len(re.findall(pattern, base)) == 1, "control %r anchor" % label
        mutated = re.sub(pattern, repl, base, count=1)
        args = (mutated, COMMON_CODE) if which == "runtime" else (RUNTIME_CODE, mutated)
        assert check_segment_graphs_drained(*args), "control %r was not caught" % label
    print("PASS segment-graphs-destroyed-after-drain-source-gate (%d controls caught)" % len(controls))

def check_keyed_q8_cache(code: str) -> list:
    """llama.cpp-7pm2: the MMVQ Q8 activation cache describes the last quantize that ran. A recording stores entries
    for quantizes that only run when the graph is submitted, and a replay rewrites the buffer without a store, so
    the keyed record and replay invalidate it around every recorded segment, and a failed recording invalidates it
    before the run executes directly (or the fallback does), else the first direct matmul reads a stale quantize."""
    problems = []
    inv = "sycl_ctx->mmvq_q8_activation_cache.invalidate();"
    try:
        record = region(code, "static void moe_graph_record_segment_slot(", "static void moe_graph_replay_segment_slot(")
        replay = region(code, "static void moe_graph_replay_segment_slot(", "static bool moe_graph_record_segments(")
    except ValueError as error:
        return [str(error)]
    graph = re.search(r"sycl_ex::command_graph\s+seg_graph\(", record)
    before = record.rfind(inv, 0, graph.start()) if graph else -1
    flush = record.find("moe_graph_segment_boundary_flush(")
    if not graph or before < 0 or before < flush:
        problems.append("record: no invalidate between the boundary flush and the recording")
    fb = re.search(r"catch \(const ggml_sycl_fallback_error &\)\s*\{([^{}]*)\}", record)
    if not fb or inv not in fb.group(1) or fb.group(1).find(inv) > fb.group(1).find("throw;"):
        problems.append("record: the fallback rethrow leaves the recording's Q8 entries behind")
    ex = re.search(r"catch \(const std::exception & exc\)\s*\{([\s\S]*?)\n        \}", record)
    body = ex.group(1) if ex else ""
    if inv not in body or "dispatch_direct(item.start, item.end);" not in body or \
            body.find(inv) > body.find("dispatch_direct(item.start, item.end);"):
        problems.append("record: a failed recording runs directly on the recording's Q8 entries")
    submit = record.find("ggml_sycl::graph_exec_submit(*stream, *slot.segments.back().exec_graph);")
    if submit < 0 or record.find(inv, submit) < 0:
        problems.append("record: the submitted segment's Q8 rewrite is not invalidated")
    rsub = replay.find("ggml_sycl::graph_exec_submit(*stream, *seg.exec_graph);")
    rcont = replay.find("continue;", rsub)
    if rsub < 0 or rcont < 0 or inv not in replay[rsub:rcont]:
        problems.append("replay: a replayed segment's Q8 rewrite is not invalidated")
    return problems


with gate('keyed-q8-cache-invalidation'):
    problems = check_keyed_q8_cache(RUNTIME_CODE)
    assert not problems, "\n".join(problems)
    _rec = region(RUNTIME_CODE, "static void moe_graph_record_segment_slot(", "static void moe_graph_replay_segment_slot(")
    _rep = region(RUNTIME_CODE, "static void moe_graph_replay_segment_slot(", "static bool moe_graph_record_segments(")
    _inv = "sycl_ctx->mmvq_q8_activation_cache.invalidate();"

    def _drop_nth(body, n):
        at = -1
        for _ in range(n + 1):
            at = body.find(_inv, at + 1)
        return RUNTIME_CODE.replace(body, body[:at] + body[at + len(_inv):], 1)

    _n = _rec.count(_inv)
    assert _n == 4, "expected 4 Q8 invalidates in the keyed record, found %d" % _n
    controls = {"record invalidate %d dropped" % i: _drop_nth(_rec, i) for i in range(_n)}
    controls["replay invalidate dropped"] = _drop_nth(_rep, 0)
    for label, mutated in controls.items():
        assert mutated != RUNTIME_CODE, "control %r did not apply" % label
        assert check_keyed_q8_cache(mutated), "control %r was not caught" % label
    print("PASS keyed-q8-cache-invalidation-source-gate (%d controls caught)" % len(controls))

MMID_SKIP_DEF = "static bool ggml_sycl_moe_skip_precomputed_mmid(ggml_backend_sycl_context & ctx, ggml_tensor * node) {"
SEG_DISPATCH_DEF = "static bool moe_graph_dispatch_direct_node(ggml_backend_sycl_context * sycl_ctx, ggml_tensor * node) {"
COMPUTE_IMPL_DEF = "static void ggml_backend_sycl_graph_compute_impl(ggml_backend_sycl_context * sycl_ctx, ggml_cgraph * cgraph) {"


def check_keyed_precomputed_mmid_skip(code: str) -> list:
    """llama.cpp-7pm2: a fused MoE executor running on a layer's gate MUL_MAT_ID also writes the up and down
    MUL_MAT_IDs and marks them in g_moe_precomputed_mmid_skip. ggml_sycl_mul_mat_id honors the mark only for
    prompt shapes, so on decode the dispatcher must: compute_impl did, and the keyed record/replay called
    ggml_sycl_compute_forward on every boundary instead, rerunning both nodes per layer through the generic
    route (a host readback of the routing ids each). One helper owns the skip and its side effects; compute_impl
    and every keyed direct dispatch go through it."""
    problems = []
    try:
        helper = region(code, MMID_SKIP_DEF, "\n}\n")
        dispatch = region(code, SEG_DISPATCH_DEF, "\n}\n")
        impl = region(code, COMPUTE_IMPL_DEF, "\n}\n")
        record = region(code, "static void moe_graph_record_segment_slot(", "static void moe_graph_replay_segment_slot(")
        replay = region(code, "static void moe_graph_replay_segment_slot(", "static bool moe_graph_record_segments(")
    except ValueError as error:
        return [str(error)]
    if not re.search(r"node->op\s*!=\s*GGML_OP_MUL_MAT_ID\s*\|\|\s*"
                     r"!ggml_sycl_moe_precomputed_skip_contains\(g_moe_precomputed_mmid_skip,\s*node,\s*ctx\.device\)"
                     r"\)\s*\{\s*return false;\s*\}", helper):
        problems.append("helper: does not test the exact MUL_MAT_ID mark before skipping")
    tail = helper[helper.find("return false;") + 1:]
    for effect in ("ggml_sycl_moe_residual_add_id_skip_clear_last();",
                   "ggml_sycl_moe_down_sum_shadow_compare(ctx, node);", "return true;"):
        if effect not in tail:
            problems.append("helper: a skipped node does not run %s" % effect)
    if not re.search(r"if\s*\(\s*!node\s*\|\|\s*ggml_sycl_is_noop\(node\)\s*\|\|\s*"
                     r"ggml_sycl_moe_skip_precomputed_mmid\(\*sycl_ctx,\s*node\)\s*\)\s*\{\s*return false;\s*\}\s*"
                     r"ggml_sycl_compute_forward\(\*sycl_ctx,\s*node\);\s*return true;", dispatch):
        problems.append("dispatch: the keyed direct dispatch does not consult the skip before compute_forward")
    if not re.search(r"if\s*\(ggml_sycl_moe_skip_precomputed_mmid\(\*sycl_ctx,\s*node\)\)\s*\{\s*continue;\s*\}", impl):
        problems.append("compute_impl: the decode early-skip does not go through the shared helper")
    if "g_moe_precomputed_mmid_skip, node, sycl_ctx->device" in impl:
        problems.append("compute_impl: keeps its own copy of the MUL_MAT_ID mark test")
    for name, body in (("record", record), ("replay", replay)):
        if "ggml_sycl_compute_forward(" in body:
            problems.append("%s: dispatches a node without the precomputed-skip helper" % name)
    if record.count("moe_graph_dispatch_direct_node(sycl_ctx, cgraph->nodes[i])") < 2:
        problems.append("record: the direct and recorded node loops do not both use the shared dispatch")
    if not re.search(r"if\s*\(moe_graph_dispatch_direct_node\(sycl_ctx,\s*node\)\s*&&\s*"
                     r"node->op\s*==\s*GGML_OP_MUL_MAT_ID\)\s*\{\s*"
                     r"g_graph_diag_counters\.seg_moe_dispatches\.fetch_add", replay):
        problems.append("replay: a boundary is counted or dispatched without the shared dispatch")
    if replay.count("moe_graph_dispatch_direct_node(sycl_ctx, ") < 2:
        problems.append("replay: the direct segment loop does not use the shared dispatch")
    return problems


with gate('keyed-precomputed-mmid-skip'):
    problems = check_keyed_precomputed_mmid_skip(RUNTIME_CODE)
    assert not problems, "\n".join(problems)
    _helper = region(RUNTIME_CODE, MMID_SKIP_DEF, "\n}\n")
    _dispatch = region(RUNTIME_CODE, SEG_DISPATCH_DEF, "\n}\n")
    _impl = region(RUNTIME_CODE, COMPUTE_IMPL_DEF, "\n}\n")
    _rec = region(RUNTIME_CODE, "static void moe_graph_record_segment_slot(", "static void moe_graph_replay_segment_slot(")
    _rep = region(RUNTIME_CODE, "static void moe_graph_replay_segment_slot(", "static bool moe_graph_record_segments(")

    def _swap(body, old, new, count=1):
        assert old in body, "control anchor missing: %r" % old
        return RUNTIME_CODE.replace(body, body.replace(old, new, count), 1)

    _nth = lambda body, old, new, n: RUNTIME_CODE.replace(  # noqa: E731
        body, body[:[i for i in range(len(body)) if body.startswith(old, i)][n]] + new +
        body[[i for i in range(len(body)) if body.startswith(old, i)][n] + len(old):], 1)
    controls = {
        "helper tests the node mark set": _swap(_helper, "g_moe_precomputed_mmid_skip", "g_moe_precomputed_node_skip"),
        "helper drops clear_last": _swap(_helper, "ggml_sycl_moe_residual_add_id_skip_clear_last();", ""),
        "helper drops shadow compare": _swap(_helper, "ggml_sycl_moe_down_sum_shadow_compare(ctx, node);", ""),
        "dispatch skips no mark": _swap(_dispatch, " || ggml_sycl_moe_skip_precomputed_mmid(*sycl_ctx, node)", ""),
        "compute_impl inline copy": _swap(
            _impl, "if (ggml_sycl_moe_skip_precomputed_mmid(*sycl_ctx, node)) {",
            "if (node->op == GGML_OP_MUL_MAT_ID && ggml_sycl_moe_precomputed_skip_contains("
            "g_moe_precomputed_mmid_skip, node, sycl_ctx->device)) {"),
        "record direct loop bypasses": _nth(_rec, "moe_graph_dispatch_direct_node(sycl_ctx, cgraph->nodes[i])",
                                            "ggml_sycl_compute_forward(*sycl_ctx, cgraph->nodes[i])", 0),
        "record recorded loop bypasses": _nth(_rec, "moe_graph_dispatch_direct_node(sycl_ctx, cgraph->nodes[i])",
                                              "ggml_sycl_compute_forward(*sycl_ctx, cgraph->nodes[i])", 1),
        "replay boundary bypasses": _swap(_rep, "if (moe_graph_dispatch_direct_node(sycl_ctx, node) &&",
                                          "if (ggml_sycl_compute_forward(*sycl_ctx, node) &&"),
        "replay direct segment bypasses": _nth(_rep, "moe_graph_dispatch_direct_node(sycl_ctx, ",
                                               "ggml_sycl_compute_forward(*sycl_ctx, ", 0),
    }
    for label, mutated in controls.items():
        assert mutated != RUNTIME_CODE, "control %r did not apply" % label
        assert check_keyed_precomputed_mmid_skip(mutated), "control %r was not caught" % label
    print("PASS keyed-precomputed-mmid-skip-source-gate (%d controls caught)" % len(controls))

def check_keyed_slot_input_staging(code: str, common: str) -> list:
    """llama.cpp-7pm2 review I3: a slot's graphs bake the staging copies of its inputs. The staging map is keyed by
    tensor struct (not by the slot key) and is cleared at a phase boundary before the slots drain, so the slot owns
    those handles (declared before its graphs, so it outlives them) and a replay first checks that every input
    still stages to the same allocation; a mismatch forgets the slot and runs direct."""
    problems = []
    try:
        slot = region(common, "struct moe_segment_slot {", "};")
        capture = region(code, "static void moe_segment_slot_capture_staging(", "\n}\n")
        matches = region(code, "static bool moe_segment_slot_staging_matches(", "\n}\n")
        lookup = region(code, "static ggml_sycl::mem_handle moe_segment_slot_input_staging(", "\n}\n")
        compute = region(code, "static ggml_status ggml_backend_sycl_graph_compute_unchecked(",
                         "static ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend")
    except ValueError as error:
        return [str(error)]
    staging = slot.find("std::vector<ggml_sycl::mem_handle> input_staging;")
    graphs = slot.find("std::vector<moe_graph_segment>     segments;")
    if staging < 0 or graphs < 0 or staging > graphs:
        problems.append("the slot does not own its inputs' staging handles, declared before (outliving) its graphs")
    if "ctx->graph_input_stage_lookup(t, ggml_nbytes(t), ctx->device, &staged, nullptr)" not in lookup:
        problems.append("the staging a slot holds is not the handle the staging map hands out")
    if "slot.input_staging.push_back(moe_segment_slot_input_staging(" not in capture:
        problems.append("the capture does not take each input's staging handle")
    if not re.search(r"rec\.valid\(\) != now\.valid\(\) \|\|\s*\(rec\.valid\(\) && !rec\.stable_identity_equal\(now\)\)",
                     matches) or "return false;" not in matches:
        problems.append("the replay check does not compare each input's staging by allocation identity")
    # The loop indexes the recorded staging by the input's position: a slot whose recorded list does not line up
    # with its inputs is a mismatch, never an out-of-range read.
    if not re.search(r"if \(slot\.input_staging\.size\(\) != slot\.input_refs\.size\(\)\)\s*\{\s*return false;\s*\}\s*"
                     r"for \(size_t i = 0; i < slot\.input_refs\.size\(\); \+\+i\)", matches):
        problems.append("the replay check reads recorded staging without first checking it lines up with the inputs")
    order = [compute.find(t) for t in ("moe_segment_slot_collect_inputs(cgraph, slot.input_refs);",
                                       "moe_segment_slot_refresh_inputs(sycl_ctx, cgraph, slot);",
                                       "moe_segment_slot_capture_staging(sycl_ctx, cgraph, slot);",
                                       "moe_graph_record_segment_slot(sycl_ctx, cgraph, slot);")]
    if min(order) < 0 or order != sorted(order):
        problems.append("the record does not capture the staging after refreshing it and before recording")
    if not re.search(r"\}\s*else if \(!moe_segment_slot_staging_matches\(sycl_ctx, cgraph, \*slot\)\)\s*\{\s*"
                     r"GGML_SYCL_DEBUG\([^\n]*\);\s*sycl_ctx->moe_segment_slots\.forget\(slot_key\);\s*"
                     r"compute_impl_unlocked\(\);\s*\}\s*else\s*\{\s*moe_segment_slot_refresh_inputs\(sycl_ctx, cgraph, \*slot\);"
                     r"\s*moe_graph_replay_segment_slot\(", compute):
        problems.append("a replay does not drop a slot whose inputs stage elsewhere, then refresh its inputs, "
                        "before replaying it")
    return problems


with gate('keyed-slot-input-staging'):
    problems = check_keyed_slot_input_staging(RUNTIME_CODE, COMMON_CODE)
    assert not problems, "\n".join(problems)
    controls = (
        ("staging declared after the graphs", "common",
         r"(        std::vector<ggml_sycl::mem_handle> input_staging;\n)([\s\S]*?)(        std::vector<moe_graph_segment>     segments;\n)",
         r"\2\3\1"),
        ("capture dropped", "runtime", r"\n\s*moe_segment_slot_capture_staging\(sycl_ctx, cgraph, slot\);", ""),
        ("capture before the refresh", "runtime",
         r"(moe_segment_slot_refresh_inputs\(sycl_ctx, cgraph, slot\);)(\s*)(moe_segment_slot_capture_staging\(sycl_ctx, cgraph, slot\);)",
         r"\3\2\1"),
        ("replay skips the check", "runtime", r"else if \(!moe_segment_slot_staging_matches\(sycl_ctx, cgraph, \*slot\)\)",
         "else if (false)"),
        ("mismatch keeps the slot", "runtime", r"\n\s*sycl_ctx->moe_segment_slots\.forget\(slot_key\);", ""),
        ("check ignores identity", "runtime", r" \|\|\s*\(rec\.valid\(\) && !rec\.stable_identity_equal\(now\)\)", ""),
        ("replay without the input refresh", "runtime",
         r"(compute_impl_unlocked\(\);\s*\}\s*else\s*\{)\s*moe_segment_slot_refresh_inputs\(sycl_ctx, cgraph, \*slot\);", r"\1"),
        ("capture holds nothing", "runtime", r"slot\.input_staging\.push_back\(moe_segment_slot_input_staging\(",
         "(void) (moe_segment_slot_input_staging("),
        ("check reads without the size guard", "runtime",
         r"if \(slot\.input_staging\.size\(\) != slot\.input_refs\.size\(\)\)\s*\{\s*return false;\s*\}\s*", ""),
        ("size guard admits a mismatch", "runtime",
         r"(if \(slot\.input_staging\.size\(\) != slot\.input_refs\.size\(\)\)\s*\{\s*)return false;", r"\1(void) 0;"),
    )
    for label, which, pattern, repl in controls:
        base = RUNTIME_CODE if which == "runtime" else COMMON_CODE
        assert len(re.findall(pattern, base)) == 1, "control %r anchor (%d)" % (label, len(re.findall(pattern, base)))
        mutated = re.sub(pattern, repl, base, count=1)
        args = (mutated, COMMON_CODE) if which == "runtime" else (RUNTIME_CODE, mutated)
        assert check_keyed_slot_input_staging(*args), "control %r was not caught" % label
    print("PASS keyed-slot-input-staging-source-gate (%d controls caught)" % len(controls))

def check_keyed_plan_and_key(code: str) -> list:
    """llama.cpp-7pm2 review I1: the invariants keyed replay rests on.
    Plan: every MUL_MAT_ID is a direct boundary (its routing and the yx28/CPU-expert executors change per token),
    so are the ffn_moe_* nodes of its expert section (the fused executor decides per call what it computes),
    FLASH_ATTN_EXT unless FA capture is allowed, and host-dispatched attention; only runs of at least min_nodes
    working nodes record. Key: names and storage of every leaf, node and src; storage is the buffer's owner
    identity plus the offset inside it, never a raw address. Slot: it retains the weight handles its graphs read."""
    problems = []
    try:
        plan = region(code, "static std::vector<moe_graph_keyed_item> moe_graph_keyed_plan(", "\n}\n")
        key = region(code, "static ggml_sycl::graph_segment_cache::key moe_segment_slot_key(", "\n}\n")
        ident = region(code, "static uint64_t moe_segment_buffer_identity(", "\n}\n")
        handle_ident = region(code, "static uint64_t moe_segment_handle_identity(", "\n}\n")
        record = region(code, "static void moe_graph_record_segment_slot(", "static void moe_graph_replay_segment_slot(")
    except ValueError as error:
        return [str(error)]
    if not re.search(r"boundary = node->op == GGML_OP_MUL_MAT_ID \|\| \(expert_section && moe_named\) \|\|\s*"
                     r"\(node->op == GGML_OP_FLASH_ATTN_EXT && !fa_graph\) \|\|\s*"
                     r"\(host_attn && \(node->op == GGML_OP_FLASH_ATTN_EXT \|\| node->op == GGML_OP_SET_ROWS\) &&", plan):
        problems.append("plan: MUL_MAT_ID, expert-section, FA or host-attention nodes are not all boundaries")
    if not re.search(r"const bool moe_named = std::strncmp\(node->name, \"ffn_moe_\", 8\) == 0;\s*"
                     r"if \(node->op == GGML_OP_MUL_MAT_ID\)\s*\{\s*expert_section = true;\s*\}\s*"
                     r"else if \(node->name\[0\] != '\\0' && !moe_named\)\s*\{\s*expert_section = false;\s*\}", plan):
        problems.append("plan: the expert section does not run from a MUL_MAT_ID to the next named non-ffn_moe node")
    if not re.search(r"if \(boundary\)\s*\{\s*close_run\(i\);\s*plan\.push_back\(\{ i, i \+ 1, true, false \}\);", plan):
        problems.append("plan: a boundary is not its own direct, never-graphed step")
    if "plan.push_back({ run_start, end, false, run_work >= min_nodes });" not in plan:
        problems.append("plan: a run records regardless of its size")
    for needle, what in (("names.mix_name(t->name, GGML_MAX_NAME);", "names"),
                         ("storage.mix(sb->identity);", "owner identity"),
                         ("storage.mix(sb->base ? static_cast<uint64_t>(static_cast<const char *>(t->data) - sb->base) : 0);",
                          "offset inside the buffer"),
                         ("k.signature = signature;", "signature"), ("k.names     = names.value();", "names value"),
                         ("k.storage   = storage.value();", "storage value"), ("k.n_nodes   = cgraph->n_nodes;", "n_nodes"),
                         ("k.device    = ctx->device;", "device"), ("k.is_decode = is_decode;", "phase"),
                         ("mix_tensor(cgraph->leafs[i]);", "leafs"), ("mix_tensor(node);", "nodes"),
                         ("mix_tensor(node->src[j]);", "srcs")):
        if needle not in key:
            problems.append("key: missing %s" % what)
    if re.search(r"(reinterpret_cast<u?int\w*>|\(u?int\w*_t\)\s*|static_cast<u?int\w*_t>\()\s*\(?t->data\b", key):
        problems.append("key: a raw tensor address is mixed into the key")
    if not re.search(r"return static_cast<uint64_t>\(handle\.stable_identity_hash\(\)\) \^ \(handle\.generation\(\) \*",
                     handle_ident):
        problems.append("key: a handle's identity is not its stable owner identity and generation")
    if not re.search(r"bctx->managed_handle\.has_stable_owner_identity\(\)\)\s*\{\s*"
                     r"return moe_segment_handle_identity\(bctx->managed_handle\);", ident):
        problems.append("key: a SYCL buffer is not identified by its mem_handle owner identity")
    if not re.search(r"buffer->iface\.free_buffer == ggml_backend_sycl_host_buffer_free_buffer\)\s*\{[^{}]*"
                     r"hctx->buffer_handle\.has_stable_owner_identity\(\)\)\s*\{\s*"
                     r"return moe_segment_handle_identity\(hctx->buffer_handle\);", ident):
        problems.append("key: a pinned-host buffer is not identified by its mem_handle owner identity")
    weights = re.search(r"if \(!root \|\| !ggml_sycl_tensor_is_weight\(root\)\)\s*\{\s*continue;\s*\}[\s\S]*?"
                        r"slot\.retained_handles\.push_back\(extra->data_handle\[device\]\);", record)
    push = record.find("slot.segments.push_back({ item.start, item.end, std::move(exec) });")
    if not weights or push < 0 or weights.end() > push:
        problems.append("slot: a recorded segment does not retain the weight handles its nodes read")
    fallback = re.search(r"const ggml_sycl::resolved_ptr view = ggml_sycl_resolve\(node->src\[s\], device\);\s*"
                         r"if \(view\.retention && view\.retention->valid\(\)\)\s*\{\s*"
                         r"slot\.retained_handles\.push_back\(\*view\.retention\);", record)
    if not weights or not fallback or fallback.start() < weights.end() or fallback.end() > push:
        problems.append("slot: a weight resolved through the cache fallback is not retained for the graph's life")
    if "slot.retained_handles.push_back(q8);" not in record:
        problems.append("slot: the Q8 activation buffer the graphs bake is not retained")
    return problems


with gate('keyed-plan-and-key'):
    problems = check_keyed_plan_and_key(RUNTIME_CODE)
    assert not problems, "\n".join(problems)
    controls = (
        ("M1 MUL_MAT_ID recorded", r"boundary = node->op == GGML_OP_MUL_MAT_ID \|\| ", "boundary = "),
        ("expert section never ends", r"else if \(node->name\[0\] != '\\0' && !moe_named\)", "else if (false)"),
        ("FA recorded", r"\(node->op == GGML_OP_FLASH_ATTN_EXT && !fa_graph\) \|\|", "false ||"),
        ("small runs recorded", r"run_work >= min_nodes \}\);", "true });"),
        ("boundary graphed", r"plan\.push_back\(\{ i, i \+ 1, true, false \}\);", "plan.push_back({ i, i + 1, false, true });"),
        ("M2 raw address", r"storage\.mix\(sb->identity\);", "storage.mix(reinterpret_cast<uint64_t>(t->data));"),
        ("offset dropped", r"storage\.mix\(sb->base \? [^\n]*\n", "\n"),
        ("M3 names dropped", r"k\.names     = names\.value\(\);", "k.names     = 0;"),
        ("srcs not keyed", r"mix_tensor\(node->src\[j\]\);", "(void) j;"),
        ("buffer identity from size", r"return moe_segment_handle_identity\(bctx->managed_handle\);", "return 0;"),
        ("pinned host buffers keyed by size", r"return moe_segment_handle_identity\(hctx->buffer_handle\);", "return 0;"),
        ("handle identity ignores the generation", r"\^ \(handle\.generation\(\) \* 0x9e3779b97f4a7c15ULL\)", ""),
        ("M4 weights not retained", r"slot\.retained_handles\.push_back\(extra->data_handle\[device\]\);", "(void) extra;"),
        ("Q8 not retained", r"slot\.retained_handles\.push_back\(q8\);", "(void) q8;"),
        ("fallback view not retained", r"slot\.retained_handles\.push_back\(\*view\.retention\);", "(void) view;"),
    )
    for label, pattern, repl in controls:
        assert len(re.findall(pattern, RUNTIME_CODE)) == 1, "control %r anchor (%d)" % (
            label, len(re.findall(pattern, RUNTIME_CODE)))
        assert check_keyed_plan_and_key(re.sub(pattern, repl, RUNTIME_CODE, count=1)), \
            "control %r was not caught" % label
    print("PASS keyed-plan-and-key-source-gate (%d controls caught)" % len(controls))

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

finish(min_checks=63)
