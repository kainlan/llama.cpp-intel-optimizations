"""Source contract for naming who holds a VRAM arena that a second model load cannot rebuild.

A load whose plan needs larger zones rebuilds the arena, and the rebuild waits until nothing points into the arena.
When it refuses, the line has to say who holds it. These checks pin the three parts of that answer:

Every lease a mem_handle takes on an arena chunk is counted by the site that took it. Each of the four acquiring paths
in mem-handle.cpp passes a site (a weight lease snapshot, a weight resolve, a handle made from a raw pointer, whose
caller names what keeps it, and a copy, which keeps its source's site and counts as a copy). The handle carries the tag
to its release, and a move hands it on with the lease, so the release lowers the count the acquire raised. The count
costs one relaxed atomic add on each side and takes no lock, and moves only when the per-chunk count moved, so the
counts sum to the chunks' leases.

The rebuild refusal prints those counts on a WARN line and lists the live allocations of every zone with live bytes,
through the cache's own dump, which takes no lock above the refusal's.

A cache scratch buffer blocks the rebuild only when the arena owns its pointer. The decision is the pure predicate
zone_arena_rebuild_allowed (test-zone-sizing Case 36 checks it), fed by a count that asks vram_owns for every scratch
pointer the cache holds.

Host-only text assertions, no SYCL device. llama_test_pytest hands this file to pytest.main(), so the checks live
inside test_*() functions. Checks read COMMENT-STRIPPED, whitespace-normalised text. Each check has a mutation witness
that applies the regression it guards and must make the check fail.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"
MH = (SYCL / "mem-handle.cpp").read_text()
MHH = (SYCL / "mem-handle.hpp").read_text()
UC = (SYCL / "unified-cache.cpp").read_text()
UCH = (SYCL / "unified-cache.hpp").read_text()
COMMON = (SYCL / "common.hpp").read_text()
GS = (SYCL / "ggml-sycl.cpp").read_text()

_LEXEME_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"'  # string literal (kept)
    r"|'(?:\\.|[^'\\\n])*'"  # char literal (kept)
    r"|//[^\n]*"  # line comment (dropped)
    r"|/\*.*?\*/",  # block comment (dropped)
    flags=re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def code(src: str) -> str:
    """Comment-stripped text with every run of whitespace made one space. Idempotent."""
    return re.sub(r"\s+", " ", strip_comments(src))


def block_at(text: str, start: int) -> str:
    """The brace block that opens at the first '{' at or after `start`, skipping string and char literals."""
    o = text.find("{", start) if start >= 0 else -1
    if o == -1:
        return ""
    depth, i = 0, o
    while i < len(text):
        c = text[i]
        if c in "\"'":
            j = i + 1
            while j < len(text) and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[o : i + 1]
        i += 1
    return ""


def fn_body(text: str, sig: str) -> str:
    s = text.find(sig)
    return block_at(text, s + len(sig)) if s != -1 else ""


def ordered(text: str, *parts: str) -> bool:
    pos = 0
    for p in parts:
        k = text.find(p, pos)
        if k == -1:
            return False
        pos = k + len(p)
    return bool(text)


def mutate_in(src: str, sig: str, old: str, new: str) -> str:
    """`src` as code() with the first `old` inside the function `sig` replaced by `new`."""
    c = code(src)
    body = fn_body(c, sig)
    assert body and old in body, f"witness anchor {old!r} is not in {sig!r}"
    s = c.find(body, c.find(sig))
    return c[:s] + body.replace(old, new, 1) + c[s + len(body) :]


def mutate(src: str, old: str, new: str) -> str:
    """`src` as code() with the first `old` replaced by `new`."""
    c = code(src)
    assert old in c, f"witness anchor {old!r} is not in the source"
    return c.replace(old, new, 1)


def call_args(text: str, start: int) -> list:
    """The top-level arguments of the call whose '(' is the first one at or after `start`."""
    o = text.find("(", start)
    depth, i, args, cur = 0, o, [], []
    while i < len(text):
        c = text[i]
        if c in "([{":
            depth += 1
            if depth == 1:
                i += 1
                continue
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                args.append("".join(cur).strip())
                return [a for a in args if a]
        elif c == "," and depth == 1:
            args.append("".join(cur).strip())
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    return []


# ---- every chunk lease a mem_handle takes is counted by its site ----

ACQUIRE = "arena_acquire_chunk_lease("
# The overload that takes the cache key; the one that takes a cache id forwards to it.
SNAPSHOT_SIG = "mem_handle mem_handle::from_weight_lease_snapshot(const unified_cache_key & key,"
CHUNK_PTR_SIG = "mem_handle mem_handle::from_chunk_ptr("
RESOLVE_SIG = "resolved_ptr mem_handle::resolve_slow() const"
BUMP_SIG = "static void bump_chunk_lease_for_copy("
COPY_CTOR_SIG = "mem_handle::mem_handle(const mem_handle & other)"
COPY_ASSIGN_SIG = "mem_handle & mem_handle::operator=(const mem_handle & other)"
TAKE_SIG = "mem_handle::lease_state mem_handle::take_lease_state_locked() const"
STORE_SIG = "void mem_handle::store_lease_state_locked(const lease_state & state) const"
RELEASE_SIG = "void mem_handle::release_lease_state(const lease_state & state) noexcept"

# Each acquiring path, the tag it acquires with, the call, and where the tag is kept for the release.
SITE_PATHS = (
    (
        SNAPSHOT_SIG,
        "const chunk_lease_tag lease_tag{ chunk_lease_site::WEIGHT_SNAPSHOT, false };",
        "cache->arena_acquire_chunk_lease(ptr, lease_tag);",
        "h.chunk_tag_ = lease_tag;",
    ),
    (
        CHUNK_PTR_SIG,
        "const chunk_lease_tag lease_tag{ site, false };",
        "cache->arena_acquire_chunk_lease(ptr, lease_tag);",
        "h.chunk_tag_ = lease_tag;",
    ),
    (
        RESOLVE_SIG,
        "const chunk_lease_tag lease_tag{ chunk_lease_site::WEIGHT_RESOLVE, false };",
        "cache->arena_acquire_chunk_lease(resolved.ptr, lease_tag);",
        "fresh.chunk_tag = lease_tag;",
    ),
)
BUMP_CALL = "out_vram_idx = cache->arena_acquire_chunk_lease(ptr, tag);"


def claim_every_lease_names_a_site(mh: str) -> bool:
    c = code(mh)
    calls = [m.start() for m in re.finditer(re.escape(ACQUIRE), c)]
    if len(calls) < 4 or any(len(call_args(c, k)) != 2 for k in calls):
        return False
    return all(ordered(fn_body(c, sig), *parts) for sig, *parts in SITE_PATHS) and BUMP_CALL in fn_body(c, BUMP_SIG)


# A copy takes a lease of its own, keeps its source's site and counts as a copy, so the counts by site and copy sum to
# the chunks' leases.
COPY_CTOR_TAG = "chunk_tag_ = chunk_lease_tag{ other.chunk_tag_.site, true };"
COPY_CTOR_BUMP = "bump_chunk_lease_for_copy(chunk_source_, chunk_device_, cached_.ptr, chunk_tag_, host_chunk_handle_,"
COPY_ASSIGN_TAG = "new_chunk_tag = chunk_lease_tag{ other.chunk_tag_.site, true };"
COPY_ASSIGN_BUMP = "bump_chunk_lease_for_copy(new_chunk_source, new_chunk_device, new_cached.ptr, new_chunk_tag,"


def claim_copies_count_as_copies(mh: str) -> bool:
    c = code(mh)
    return ordered(fn_body(c, COPY_CTOR_SIG), COPY_CTOR_TAG, COPY_CTOR_BUMP) and ordered(
        fn_body(c, COPY_ASSIGN_SIG), COPY_ASSIGN_TAG, COPY_ASSIGN_BUMP, "fresh.chunk_tag = new_chunk_tag;"
    )


# The tag travels with the lease state, so the release lowers the site its acquire raised. A move hands the lease to
# the new handle: the move constructor copies the taken state field by field, the move assignment stores it whole, and
# either way the tag goes with it, or the new handle's release would lower no site.
RELEASE_CALL = "cache->arena_release_chunk_lease(state.vram_chunk_idx, state.chunk_tag);"
MOVE_CTOR_SIG = "mem_handle::mem_handle(mem_handle && other) noexcept"
MOVE_ASSIGN_SIG = "mem_handle & mem_handle::operator=(mem_handle && other) noexcept"
MOVE_CTOR_TAKE = "const lease_state st = other.take_lease_state_locked();"
MOVE_CTOR_TAG = "chunk_tag_ = st.chunk_tag;"
MOVE_ASSIGN_TAKE = "fresh = other.take_lease_state_locked();"
MOVE_ASSIGN_STORE = "store_lease_state_locked(fresh);"


def claim_release_lowers_the_raising_site(mh: str, mhh: str) -> bool:
    c = code(mh)
    return (
        "state.chunk_tag = chunk_tag_;" in fn_body(c, TAKE_SIG)
        and "chunk_tag_ = {};" in fn_body(c, TAKE_SIG)
        and "chunk_tag_ = state.chunk_tag;" in fn_body(c, STORE_SIG)
        and RELEASE_CALL in fn_body(c, RELEASE_SIG)
        and "chunk_lease_tag chunk_tag;" in fn_body(code(mhh), "struct lease_state")
        and ordered(fn_body(c, MOVE_CTOR_SIG), MOVE_CTOR_TAKE, MOVE_CTOR_TAG)
        and ordered(fn_body(c, MOVE_ASSIGN_SIG), MOVE_ASSIGN_TAKE, MOVE_ASSIGN_STORE)
    )


# The count is one relaxed add on each side and no lock. It moves only when the per-chunk count moved: the acquire adds
# after it found the chunk, and the release subtracts only when the chunk's decrement did not underflow.
UC_ACQUIRE_SIG = "int unified_cache::arena_acquire_chunk_lease(const void * ptr, chunk_lease_tag tag)"
UC_RELEASE_SIG = "void unified_cache::arena_release_chunk_lease(int chunk_idx, chunk_lease_tag tag)"
SITE_ADD = "arena_chunk_leases_of(tag).fetch_add(1, std::memory_order_relaxed);"
SITE_SUB = "arena_chunk_leases_of(tag).fetch_sub(1, std::memory_order_relaxed);"
UNDERFLOW_RETURN = "lease_count.fetch_add(1); return; }"
LOCK_WORDS = ("lock", "mutex")


def claim_site_count_costs_one_relaxed_add(uc: str, uch: str) -> bool:
    c = code(uc)
    acq = fn_body(c, UC_ACQUIRE_SIG)
    rel = fn_body(c, UC_RELEASE_SIG)
    of = fn_body(code(uch), "std::atomic<uint32_t> & arena_chunk_leases_of(chunk_lease_tag tag) const")
    return (
        ordered(acq, "if (idx < 0) { return -1; }", "lease_count.fetch_add(1);", SITE_ADD, "return idx;")
        and acq.count("fetch_add(") == 2
        and ordered(rel, "lease_count.fetch_sub(1);", "if (prev == 0) {", UNDERFLOW_RETURN, SITE_SUB)
        and rel.count("fetch_sub(") == 2
        and bool(of)
        and not any(w in body for body in (acq, rel, of) for w in LOCK_WORDS)
    )


# ---- the refusal names the holders by site and lists each live zone ----

ENSURE_SIG = "bool unified_cache::ensure_planned_arena_zones(zone_arena_refusal * refusal)"
REFUSAL_LINE = '"[VRAM-ARENA] planned zones exceed active arena but live allocations prevent rebuild: "'
HOLDERS_LINE = '"[VRAM-ARENA] live in the arena on device %d (MB): %s; deferred_in_arena=%zu (%.1f MB requested) "'
SITE_LINE = '"[VRAM-ARENA] chunk leases on device %d by site:%s (sum %u of %u)\\n"'
SITE_LOOP = (
    "for (size_t s = 0; s < static_cast<size_t>(chunk_lease_site::COUNT); ++s) {",
    "for (int copy = 0; copy < 2; ++copy) {",
    "const chunk_lease_tag tag{ static_cast<chunk_lease_site>(s), copy != 0 };",
    "const uint32_t held = arena_chunk_lease_count_at(tag);",
    "site_leases += held;",
    'chunk_lease_site_name(tag.site), tag.copy ? "(copy)" : "", held);',
)
SITE_WARN_ARGS = "site_leases, chunk_leases);"
ZONE_DUMP = (
    "if (live_zone_used[z] != 0) {",
    'dump_live_zone_allocations(static_cast<vram_zone_id>(z), "arena-rebuild");',
)


def refusal_segment(uc: str) -> str:
    """ensure_planned_arena_zones() from the refusal line up to its return false, or "" when the shape is gone."""
    ens = fn_body(code(uc), ENSURE_SIG)
    i = ens.find(REFUSAL_LINE)
    r = ens.find("return false;", max(i, 0))
    return ens[i:r] if -1 not in (i, r) else ""


def claim_refusal_prints_leases_by_site(uc: str) -> bool:
    seg = refusal_segment(uc)
    w = seg.find("GGML_LOG_WARN(" + SITE_LINE)
    return (
        w != -1
        and ordered(seg, HOLDERS_LINE, *SITE_LOOP, "GGML_LOG_WARN(" + SITE_LINE, SITE_WARN_ARGS)
        and "GGML_LOG_INFO(" not in seg
    )


def claim_refusal_dumps_every_live_zone(uc: str) -> bool:
    seg = refusal_segment(uc)
    # The cache's own dump takes only the runtime registry's lock. The free wrapper would look the cache up under the
    # cache map's lock, which ranks above the scratch lock a reserve path holds when it asks for a rebuild.
    return ordered(seg, SITE_LINE, *ZONE_DUMP) and "unified_cache_dump_live_zone_allocations(" not in seg


# ---- a cache scratch buffer blocks the rebuild only when the arena owns its pointer ----

COUNT_SIG = "void unified_cache::arena_count_scratch_holders(zone_arena_holders & holders)"
OWNS_TEST = "if (vram_owns(ptr)) { ++holders.scratch_in_arena; } else { ++holders.scratch_outside; }"
COUNTED = (
    "count(scratch_pool_ptr_);",
    "count(onednn_weights_scratch_);",
    "count(onednn_activations_scratch_);",
    "count(reorder_temp_buffer_);",
    "std::lock_guard<std::mutex> lock(pp_moe_onednn_scratch_mutex_);",
    "for (const auto * slots : { &pp_moe_onednn_scratch_slots_, &pp_moe_onednn_retired_slots_ }) {",
    "count(slot.weight);",
    "count(slot.activation);",
    "count(slot.output);",
    "std::lock_guard<std::mutex> lock(persistent_scratch_mutex_);",
    "count(named.second.device_ptr);",
)
GATE = (
    "zone_arena_holders holders;",
    "holders.live_zone_bytes = live_zone_bytes;",
    "holders.chunk_leases = chunk_leases;",
    "arena_count_scratch_holders(holders);",
    "if (!zone_arena_rebuild_allowed(holders)) {",
    REFUSAL_LINE,
)
# The pointer tests the gate used to read, any one of which refuses a rebuild for a buffer outside the arena.
POINTER_TESTS = (
    "scratch_pool_ptr_ != nullptr",
    "onednn_weights_scratch_ != nullptr",
    "onednn_activations_scratch_ != nullptr",
    "reorder_temp_buffer_ != nullptr",
    "!persistent_scratches_.empty()",
    "!pp_moe_onednn_scratch_slots_.empty()",
)


def claim_scratch_blocks_only_inside_the_arena(uc: str) -> bool:
    c = code(uc)
    body = fn_body(c, COUNT_SIG)
    ens = fn_body(c, ENSURE_SIG)
    pre = ens[: ens.find(REFUSAL_LINE)] if REFUSAL_LINE in ens else ""
    return (
        OWNS_TEST in body
        and ordered(body, *COUNTED)
        and ordered(ens, *GATE)
        and not any(t in pre for t in POINTER_TESTS)
        and "holders.scratch_in_arena, holders.ring_in_arena, holders.scratch_outside, holders.live_beside_ring);"
        in refusal_segment(uc)
    )


# ---- the handles kept past one use name what keeps them ----

TENSOR_EXTRA_CALLS = (
    "data_handle[dev] = ggml_sycl::mem_handle::from_chunk_ptr(ptr, dev, layout, on_device, "
    "ggml_sycl::chunk_lease_site::TENSOR_EXTRA);",
    "data_handle[dev] = on_device ? ggml_sycl::mem_handle::from_chunk_ptr(ptr, dev, storage_layout, true, "
    "ggml_sycl::chunk_lease_site::TENSOR_EXTRA) :",
)
GS_CALLS = (
    "published_handle = ggml_sycl::mem_handle::from_chunk_ptr(produced_base, target_device, GGML_LAYOUT_AOS, true, "
    "ggml_sycl::chunk_lease_site::TENSOR_EXTRA);",
    "extra->data_handle[target_device] = ggml_sycl::mem_handle::from_chunk_ptr(staged_base, target_device, "
    "GGML_LAYOUT_AOS, true, ggml_sycl::chunk_lease_site::TENSOR_EXTRA);",
    "entry.lease = ggml_sycl::mem_handle::from_chunk_ptr(ptr, device_id, layout, device_id >= 0, "
    "ggml_sycl::chunk_lease_site::EXPERT_DISPATCH);",
)


def tight(src: str) -> str:
    """code() without the space a line break after '(' leaves, so a wrapped call reads as one."""
    return code(src).replace("( ", "(")


def claim_kept_handles_name_their_site(common: str, gs: str) -> bool:
    cc = tight(common)
    cg = tight(gs)
    return all(t in cc for t in TENSOR_EXTRA_CALLS) and all(t in cg for t in GS_CALLS)


# ---- checks ----


def test_every_lease_names_a_site():
    assert claim_every_lease_names_a_site(MH), (
        "an arena_acquire_chunk_lease call in mem-handle.cpp passes no site, or one of the four acquiring paths does "
        "not keep the tag it acquired with: its lease would count under no site, or its release under another"
    )


def test_copies_count_as_copies():
    assert claim_copies_count_as_copies(MH), (
        "a mem_handle copy does not acquire with its source's site marked as a copy, so the counts by site no longer "
        "say which leases copies hold"
    )


def test_release_lowers_the_raising_site():
    assert claim_release_lowers_the_raising_site(MH, MHH), (
        "the lease state does not carry the chunk lease tag to release_lease_state, a move does not hand it to the new "
        "handle, or the release does not pass it, so a release lowers a different site than its acquire raised"
    )


def test_site_count_costs_one_relaxed_add():
    assert claim_site_count_costs_one_relaxed_add(UC, UCH), (
        "the per-site chunk lease count is not one relaxed add on acquire and one relaxed subtract on release, takes a "
        "lock, or moves when the per-chunk count did not (no chunk found, or an underflow)"
    )


def test_refusal_prints_leases_by_site():
    assert claim_refusal_prints_leases_by_site(UC), (
        "the arena rebuild refusal does not print the chunk leases by site and copy on a WARN line after the holders "
        "line, with their sum beside the per-chunk total, or it logs at INFO"
    )


def test_refusal_dumps_every_live_zone():
    assert claim_refusal_dumps_every_live_zone(UC), (
        "the arena rebuild refusal does not list the live allocations of every zone with live bytes through the "
        "cache's own dump"
    )


def test_scratch_blocks_only_inside_the_arena():
    assert claim_scratch_blocks_only_inside_the_arena(UC), (
        "the arena rebuild gate does not ask vram_owns for every cache scratch pointer and decide through "
        "zone_arena_rebuild_allowed, or a scratch pointer test still refuses a rebuild for a buffer outside the arena"
    )


def test_kept_handles_name_their_site():
    assert claim_kept_handles_name_their_site(COMMON, GS), (
        "a from_chunk_ptr handle a tensor extra or an expert dispatch entry keeps does not name its site, so its "
        "leases count as one-use views"
    )


def test_witness_snapshot_lease_without_a_site():
    assert not claim_every_lease_names_a_site(
        mutate_in(MH, SNAPSHOT_SIG, "arena_acquire_chunk_lease(ptr, lease_tag);", "arena_acquire_chunk_lease(ptr);")
    )


def test_witness_resolve_lease_without_a_site():
    assert not claim_every_lease_names_a_site(
        mutate_in(
            MH,
            RESOLVE_SIG,
            "cache->arena_acquire_chunk_lease(resolved.ptr, lease_tag);",
            "cache->arena_acquire_chunk_lease(resolved.ptr);",
        )
    )


def test_witness_copy_lease_without_a_site():
    assert not claim_every_lease_names_a_site(
        mutate_in(MH, BUMP_SIG, "arena_acquire_chunk_lease(ptr, tag)", "arena_acquire_chunk_lease(ptr)")
    )


def test_witness_chunk_ptr_keeps_no_tag():
    assert not claim_every_lease_names_a_site(mutate_in(MH, CHUNK_PTR_SIG, "h.chunk_tag_ = lease_tag;", ""))


def test_witness_copy_counted_as_its_source():
    assert not claim_copies_count_as_copies(
        mutate_in(MH, COPY_CTOR_SIG, COPY_CTOR_TAG, "chunk_tag_ = other.chunk_tag_;")
    )


def test_witness_release_drops_the_tag():
    assert not claim_release_lowers_the_raising_site(
        mutate_in(MH, RELEASE_SIG, RELEASE_CALL, "cache->arena_release_chunk_lease(state.vram_chunk_idx);"), MHH
    )


def test_witness_take_loses_the_tag():
    assert not claim_release_lowers_the_raising_site(mutate_in(MH, TAKE_SIG, "state.chunk_tag = chunk_tag_;", ""), MHH)


def test_witness_move_constructor_drops_the_tag():
    assert not claim_release_lowers_the_raising_site(mutate_in(MH, MOVE_CTOR_SIG, MOVE_CTOR_TAG, ""), MHH)


def test_witness_move_assignment_drops_the_lease_state():
    assert not claim_release_lowers_the_raising_site(mutate_in(MH, MOVE_ASSIGN_SIG, MOVE_ASSIGN_STORE, ""), MHH)


def test_witness_site_count_sequentially_consistent():
    assert not claim_site_count_costs_one_relaxed_add(
        mutate_in(UC, UC_ACQUIRE_SIG, SITE_ADD, "arena_chunk_leases_of(tag).fetch_add(1);"), UCH
    )


def test_witness_site_count_under_a_lock():
    assert not claim_site_count_costs_one_relaxed_add(
        mutate_in(UC, UC_RELEASE_SIG, SITE_SUB, "std::lock_guard<std::mutex> g(arena_mutex_); " + SITE_SUB), UCH
    )


def test_witness_site_count_lowered_on_underflow():
    assert not claim_site_count_costs_one_relaxed_add(
        mutate_in(UC, UC_RELEASE_SIG, "lease_count.fetch_add(1); return; }", "lease_count.fetch_add(1); }"), UCH
    )


def test_witness_refusal_without_site_line():
    assert not claim_refusal_prints_leases_by_site(mutate_in(UC, ENSURE_SIG, SITE_LINE, '"x"'))


def test_witness_site_line_at_info():
    assert not claim_refusal_prints_leases_by_site(
        mutate_in(UC, ENSURE_SIG, "GGML_LOG_WARN(" + SITE_LINE, "GGML_LOG_INFO(" + SITE_LINE)
    )


def test_witness_refusal_without_zone_dump():
    assert not claim_refusal_dumps_every_live_zone(mutate_in(UC, ENSURE_SIG, ZONE_DUMP[1], ""))


def test_witness_refusal_dumps_through_the_wrapper():
    assert not claim_refusal_dumps_every_live_zone(
        mutate_in(
            UC,
            ENSURE_SIG,
            ZONE_DUMP[1],
            'unified_cache_dump_live_zone_allocations(dev_id, static_cast<vram_zone_id>(z), "arena-rebuild", 32);',
        )
    )


def test_witness_gate_tests_the_pointer():
    assert not claim_scratch_blocks_only_inside_the_arena(
        mutate_in(UC, COUNT_SIG, "if (vram_owns(ptr)) {", "if (ptr != nullptr) {")
    )


def test_witness_gate_skips_the_reorder_temp():
    assert not claim_scratch_blocks_only_inside_the_arena(mutate_in(UC, COUNT_SIG, "count(reorder_temp_buffer_);", ""))


def test_witness_gate_restores_a_pointer_test():
    assert not claim_scratch_blocks_only_inside_the_arena(
        mutate_in(
            UC,
            ENSURE_SIG,
            "if (!zone_arena_rebuild_allowed(holders)) {",
            "if (!zone_arena_rebuild_allowed(holders) || reorder_temp_buffer_ != nullptr) {",
        )
    )


def test_witness_tensor_extra_handle_untagged():
    assert not claim_kept_handles_name_their_site(
        mutate(COMMON, ", on_device, ggml_sycl::chunk_lease_site::TENSOR_EXTRA);", ", on_device);"), GS
    )


def test_witness_expert_dispatch_lease_untagged():
    assert not claim_kept_handles_name_their_site(
        COMMON, mutate(GS, ", device_id >= 0, ggml_sycl::chunk_lease_site::EXPERT_DISPATCH);", ", device_id >= 0);")
    )


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
