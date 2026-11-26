"""Source contract for naming the code that holds each VRAM arena chunk lease.

A held arena at the last model release, at a load start or at a rebuild prints its chunk leases by site. A site names
the kind of handle (a tensor extra's, a copy of one), not the code that keeps it, and a copy of a handle is made in many
places. GGML_SYCL_LEASE_HOLDERS=1 names that code. These checks pin its parts:

Every take of a chunk lease records the address it was taken for in its tag: the return address of from_chunk_ptr, of
a weight lease snapshot, of a weight resolve, and of a copy. A copy records its copier, never its source's address. That
is one store per take, whatever the knob says, and nothing on a release.

The knob is read once, out of line, in one function, and the cache keeps the answer in a const member. Only when it is
on does a take walk the call stack, claim a slot in the holder table for the stack from the recorded address outward,
write the slot back into the caller's tag and count the lease in that slot; the release lowers the slot its tag carries,
and only after the chunk's own count did not underflow. A tag without an address (a caller of the cache's lease API that
keeps no tag) counts in slot 0.

The holders report prints the counts by site at the caller's level, the level of the line that called it (ERROR at a
last release, WARN at a load start), so a tool that shows only errors shows them. With the knob on it names each holder
at that level; with it off and a lease held it says which knob names them.

Host-only text assertions, no SYCL device. llama_test_pytest hands this file to pytest.main(), so the checks live
inside test_*() functions. Checks read COMMENT-STRIPPED, whitespace-normalised text. Each check has a mutation witness
that applies the regression it guards and must make the check fail.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"


def read(name: str) -> str:
    p = SYCL / name
    return p.read_text() if p.exists() else ""


MH = read("mem-handle.cpp")
MHH = read("mem-handle.hpp")
UC = read("unified-cache.cpp")
UCH = read("unified-cache.hpp")
HOLD = read("chunk-lease-holders.hpp")
SOURCES = {p.name: p.read_text() for p in sorted(SYCL.glob("*.[ch]pp"))}

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


def guarded(body: str, guard: str, stmt: str) -> bool:
    """`stmt` occurs once in `body`, inside the block the one `guard` opens."""
    i = body.find(guard)
    return i != -1 and body.count(guard) == 1 and body.count(stmt) == 1 and stmt in block_at(body, i)


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


# ---- the tag carries the taker's address and the holder slot ----

TAG_SIG = "struct chunk_lease_tag"
TAG_FIELDS = ("chunk_lease_site site = chunk_lease_site::UNTAGGED;", "bool copy = false;",
              "const void * pc = nullptr;", "uint16_t holder = 0;")


def claim_tag_carries_pc_and_slot(hold: str, mhh: str) -> bool:
    # The tag is defined in the SYCL-free header the host-only table test compiles, and mem-handle.hpp takes it from
    # there rather than defining a second one.
    return (
        ordered(fn_body(code(hold), TAG_SIG), *TAG_FIELDS)
        and '#include "chunk-lease-holders.hpp"' in code(mhh)
        and TAG_SIG not in code(mhh)
    )


# ---- every take records the address it was taken for; a copy records its copier ----

PC = "__builtin_return_address(0)"
SNAPSHOT_SIG = "mem_handle mem_handle::from_weight_lease_snapshot(const unified_cache_key & key,"
CHUNK_PTR_SIG = "mem_handle mem_handle::from_chunk_ptr("
RESOLVE_SIG = "resolved_ptr mem_handle::resolve_slow() const"
COPY_CTOR_SIG = "mem_handle::mem_handle(const mem_handle & other)"
COPY_ASSIGN_SIG = "mem_handle & mem_handle::operator=(const mem_handle & other)"
BUMP_SIG = "static void bump_chunk_lease_for_copy("

SNAPSHOT_TAG = "chunk_lease_tag lease_tag{ chunk_lease_site::WEIGHT_SNAPSHOT, false, " + PC + " };"
CHUNK_PTR_TAG = "chunk_lease_tag lease_tag{ site, false, " + PC + " };"
RESOLVE_TAG = "chunk_lease_tag lease_tag{ chunk_lease_site::WEIGHT_RESOLVE, false, " + PC + " };"
COPY_CTOR_TAG = "chunk_tag_ = chunk_lease_tag{ other.chunk_tag_.site, true, " + PC + " };"
COPY_ASSIGN_TAG = "new_chunk_tag = chunk_lease_tag{ other.chunk_tag_.site, true, " + PC + " };"

# Each take, its tag, the acquire that writes the holder slot into that tag, and the store that keeps the tag after it.
TAKES = (
    (SNAPSHOT_SIG, SNAPSHOT_TAG, "cache->arena_acquire_chunk_lease(ptr, lease_tag);", "h.chunk_tag_ = lease_tag;"),
    (CHUNK_PTR_SIG, CHUNK_PTR_TAG, "cache->arena_acquire_chunk_lease(ptr, lease_tag);", "h.chunk_tag_ = lease_tag;"),
    (RESOLVE_SIG, RESOLVE_TAG, "cache->arena_acquire_chunk_lease(resolved.ptr, lease_tag);",
     "fresh.chunk_tag = lease_tag;"),
    (COPY_CTOR_SIG, COPY_CTOR_TAG,
     "bump_chunk_lease_for_copy(chunk_source_, chunk_device_, cached_.ptr, chunk_tag_, host_chunk_handle_,"),
    (COPY_ASSIGN_SIG, COPY_ASSIGN_TAG,
     "bump_chunk_lease_for_copy(new_chunk_source, new_chunk_device, new_cached.ptr, new_chunk_tag,",
     "fresh.chunk_tag = new_chunk_tag;"),
)


def claim_takes_record_the_takers_address(mh: str) -> bool:
    c = code(mh)
    return (
        all(ordered(fn_body(c, sig), *parts) for sig, *parts in TAKES)
        # A copy's address is its own; a tag never takes the address its source recorded.
        and "chunk_tag_.pc" not in c
        # The address is read in the take itself, not in a helper whose return address would be the take.
        and c.count(PC) - c.count("void * ret_addr = " + PC) == len(TAKES)
    )


# The acquire writes the holder slot into the caller's tag, so the copy hands its tag by reference.
BUMP_TAG_PARAM = "chunk_lease_tag & tag,"
ACQUIRE_DECLS = ("int arena_acquire_chunk_lease(const void * ptr);",
                 "int arena_acquire_chunk_lease(const void * ptr, chunk_lease_tag & tag);")


def claim_acquire_writes_the_slot_back(mh: str, uch: str) -> bool:
    c = code(mh)
    s = c.find(BUMP_SIG)
    head = c[s : c.find("{", s)] if s != -1 else ""
    return BUMP_TAG_PARAM in head and all(d in code(uch) for d in ACQUIRE_DECLS)


# ---- the holder counts move only behind the knob, the release only after the chunk's count fell ----

UC_ACQUIRE_SIG = "int unified_cache::arena_acquire_chunk_lease(const void * ptr, chunk_lease_tag & tag)"
UC_ACQUIRE_UNTAGGED_SIG = "int unified_cache::arena_acquire_chunk_lease(const void * ptr)"
UC_RELEASE_SIG = "void unified_cache::arena_release_chunk_lease(int chunk_idx, chunk_lease_tag tag)"
FLAG = "if (chunk_lease_holders_) {"
SITE_ADD = "arena_chunk_leases_of(tag).fetch_add(1, std::memory_order_relaxed);"
SITE_SUB = "arena_chunk_leases_of(tag).fetch_sub(1, std::memory_order_relaxed);"
HOLDER_CLAIM = "tag.holder = chunk_lease_holder_claim(tag);"
HOLDER_ADD = "arena_chunk_lease_holder_of(tag).fetch_add(1, std::memory_order_relaxed);"
HOLDER_SUB = "arena_chunk_lease_holder_of(tag).fetch_sub(1, std::memory_order_relaxed);"
UNDERFLOW_RETURN = "lease_count.fetch_add(1); return; }"
FLAG_MEMBER = "const bool chunk_lease_holders_ = chunk_lease_holders_enabled();"
CLAIM_SIG = "static uint16_t chunk_lease_holder_claim(const chunk_lease_tag & tag)"
CLAIM_NULL = "if (tag.pc == nullptr) { return 0; }"


def claim_holder_counts_only_behind_the_flag(uc: str, uch: str) -> bool:
    c = code(uc)
    acq = fn_body(c, UC_ACQUIRE_SIG)
    rel = fn_body(c, UC_RELEASE_SIG)
    claim = fn_body(c, CLAIM_SIG)
    return (
        ordered(acq, SITE_ADD, FLAG, HOLDER_CLAIM, HOLDER_ADD, "return idx;")
        and guarded(acq, FLAG, HOLDER_CLAIM)
        and guarded(acq, FLAG, HOLDER_ADD)
        and ordered(rel, UNDERFLOW_RETURN, SITE_SUB, FLAG, HOLDER_SUB)
        and guarded(rel, FLAG, HOLDER_SUB)
        # The cache keeps the resolved answer; a take reads a member, never the resolver or the environment.
        and FLAG_MEMBER in code(uch)
        and "chunk_lease_holders_enabled(" not in acq + rel
        # The untagged take forwards with a tag of its own, whose null address counts in slot 0.
        and "return arena_acquire_chunk_lease(ptr, tag);" in fn_body(c, UC_ACQUIRE_UNTAGGED_SIG)
        and claim.startswith("{ " + CLAIM_NULL)
        and "backtrace(" in claim
    )


# ---- the knob is read once, out of line, in one function ----

RESOLVER_SIG = "bool chunk_lease_holders_enabled()"
RESOLVER_PARTS = ("static const bool enabled = [] {", 'std::getenv("GGML_SYCL_LEASE_HOLDERS")', "return enabled;")
KNOB_READ = re.compile(r'getenv\(\s*"GGML_SYCL_LEASE_HOLDERS"\s*\)')


def claim_knob_read_once_out_of_line(uc: str, hold: str, sources: dict) -> bool:
    reads = sum(len(KNOB_READ.findall(code(src))) for src in sources.values())
    return (
        ordered(fn_body(code(uc), RESOLVER_SIG), *RESOLVER_PARTS)
        and "bool chunk_lease_holders_enabled();" in code(hold)
        and "inline bool chunk_lease_holders_enabled" not in code(hold)
        and reads == 1
    )


# ---- the report prints the counts at the caller's level and names the holders or the knob ----

REPORT_SIG = ("void unified_cache::arena_report_holders(int dev_id, const size_t * live_zone_used, "
              "zone_arena_holders & holders, const char * site, ggml_log_level level)")
HOLDER_REPORT_SIG = ("void unified_cache::arena_report_chunk_lease_holders(int dev_id, ggml_log_level level, "
                     "uint32_t chunk_leases) const")
CENSUS_CALL = "const uint32_t site_leases = arena_chunk_lease_census(leases_by_site, sizeof(leases_by_site));"
SITE_AT_LEVEL = 'ggml_log_internal(level, "[VRAM-ARENA] chunk leases on device %d by site:%s (sum %u of %u)\\n"'
HOLDER_CALL = "arena_report_chunk_lease_holders(dev_id, level, holders.chunk_leases);"
HINT_GUARD = "} else if (holders.chunk_leases != 0) {"
HINT = ('ggml_log_internal(level, "[VRAM-ARENA] set GGML_SYCL_LEASE_HOLDERS=1 to name the code that took each of '
        'the %u chunk leases on device %d\\n"')
HOLDER_LINE = ('ggml_log_internal(level, "[VRAM-ARENA] chunk lease holder on device %d: %u held, site %s%s, '
               'taken at %s\\n"')
HOLDER_SUM = 'ggml_log_internal(level, "[VRAM-ARENA] chunk leases on device %d by holder: sum %u of %u\\n"'


def claim_report_at_the_callers_level(uc: str) -> bool:
    # Adjacent string literals read as one, so a format string wrapped across lines matches its constant.
    c = code(uc).replace('" "', "")
    rep = fn_body(c, REPORT_SIG)
    hold = fn_body(c, HOLDER_REPORT_SIG)
    return (
        ordered(rep, CENSUS_CALL, SITE_AT_LEVEL, FLAG, HOLDER_CALL, HINT_GUARD, HINT)
        and guarded(rep, FLAG, HOLDER_CALL)
        and guarded(rep, HINT_GUARD, HINT)
        and "GGML_LOG_" not in rep
        and ordered(hold, "arena_chunk_leases_by_holder_[", HOLDER_LINE, HOLDER_SUM)
        and "GGML_LOG_" not in hold
    )


# ---- checks ----


def test_tag_carries_pc_and_slot():
    assert claim_tag_carries_pc_and_slot(HOLD, MHH), (
        "chunk_lease_tag does not carry the taker's address and the holder slot, or is not the one the SYCL-free "
        "chunk-lease-holders.hpp defines"
    )


def test_takes_record_the_takers_address():
    assert claim_takes_record_the_takers_address(MH), (
        "a chunk lease take (from_chunk_ptr, a weight snapshot or resolve, a copy) does not record its own return "
        "address in its tag before the acquire and keep that tag, or a copy records its source's address"
    )


def test_acquire_writes_the_slot_back():
    assert claim_acquire_writes_the_slot_back(MH, UCH), (
        "the tagged acquire does not take the tag by reference, so the holder slot it claims never reaches the "
        "handle's release"
    )


def test_holder_counts_only_behind_the_flag():
    assert claim_holder_counts_only_behind_the_flag(UC, UCH), (
        "a holder count moves outside the cached GGML_SYCL_LEASE_HOLDERS flag, a take reads the knob itself, the "
        "release lowers a holder before the chunk's count is known not to underflow, or a take with no address "
        "claims a slot"
    )


def test_knob_read_once_out_of_line():
    assert claim_knob_read_once_out_of_line(UC, HOLD, SOURCES), (
        "GGML_SYCL_LEASE_HOLDERS is not read exactly once, in the out-of-line chunk_lease_holders_enabled()'s "
        "function-local static"
    )


def test_report_at_the_callers_level():
    assert claim_report_at_the_callers_level(UC), (
        "the holders report prints the chunk leases by site or by holder at a fixed level instead of its caller's, "
        "or names the knob when no lease is held or when the knob is already on"
    )


def test_witness_chunk_ptr_records_no_address():
    assert not claim_takes_record_the_takers_address(
        mutate_in(MH, CHUNK_PTR_SIG, CHUNK_PTR_TAG, "chunk_lease_tag lease_tag{ site, false };")
    )


def test_witness_copy_ctor_records_no_address():
    assert not claim_takes_record_the_takers_address(
        mutate_in(MH, COPY_CTOR_SIG, COPY_CTOR_TAG, "chunk_tag_ = chunk_lease_tag{ other.chunk_tag_.site, true };")
    )


def test_witness_copy_ctor_records_the_source():
    assert not claim_takes_record_the_takers_address(
        mutate_in(MH, COPY_CTOR_SIG, COPY_CTOR_TAG,
                  "chunk_tag_ = chunk_lease_tag{ other.chunk_tag_.site, true, other.chunk_tag_.pc };")
    )


def test_witness_copy_assign_records_the_source():
    assert not claim_takes_record_the_takers_address(
        mutate_in(MH, COPY_ASSIGN_SIG, COPY_ASSIGN_TAG,
                  "new_chunk_tag = chunk_lease_tag{ other.chunk_tag_.site, true, other.chunk_tag_.pc };")
    )


def test_witness_copy_assign_records_no_address():
    assert not claim_takes_record_the_takers_address(
        mutate_in(MH, COPY_ASSIGN_SIG, COPY_ASSIGN_TAG,
                  "new_chunk_tag = chunk_lease_tag{ other.chunk_tag_.site, true };")
    )


def test_witness_tag_kept_before_the_acquire():
    # A tag stored before the acquire wrote its slot would carry slot 0 to a release whose acquire counted another.
    c = mutate_in(MH, CHUNK_PTR_SIG, "h.chunk_tag_ = lease_tag;", "")
    c = mutate_in(c, CHUNK_PTR_SIG, CHUNK_PTR_TAG, CHUNK_PTR_TAG + " h.chunk_tag_ = lease_tag;")
    assert not claim_takes_record_the_takers_address(c)


def test_witness_copy_hands_its_tag_by_value():
    assert not claim_acquire_writes_the_slot_back(mutate(MH, BUMP_TAG_PARAM, "chunk_lease_tag tag,"), UCH)


def test_witness_holder_counted_without_the_flag():
    assert not claim_holder_counts_only_behind_the_flag(
        mutate_in(UC, UC_ACQUIRE_SIG, HOLDER_ADD, "} " + HOLDER_ADD + " {"), UCH
    )


def test_witness_holder_lowered_before_the_underflow_check():
    c = mutate_in(UC, UC_RELEASE_SIG, FLAG + " " + HOLDER_SUB + " }", "")
    c = mutate_in(c, UC_RELEASE_SIG, "if (prev == 0) {", FLAG + " " + HOLDER_SUB + " } if (prev == 0) {")
    assert not claim_holder_counts_only_behind_the_flag(c, UCH)


def test_witness_take_reads_the_knob():
    assert not claim_holder_counts_only_behind_the_flag(
        mutate_in(UC, UC_ACQUIRE_SIG, FLAG, "if (chunk_lease_holders_enabled()) {"), UCH
    )


def test_witness_flag_not_cached():
    assert not claim_holder_counts_only_behind_the_flag(UC, mutate(UCH, FLAG_MEMBER, "bool chunk_lease_holders_;"))


def test_witness_null_address_claims_a_slot():
    assert not claim_holder_counts_only_behind_the_flag(mutate_in(UC, CLAIM_SIG, CLAIM_NULL, ""), UCH)


def test_witness_knob_read_per_call():
    assert not claim_knob_read_once_out_of_line(
        mutate_in(UC, RESOLVER_SIG, "static const bool enabled = [] {", "const bool enabled = [] {"), HOLD, SOURCES
    )


def test_witness_knob_read_twice():
    sources = dict(SOURCES)
    sources["unified-cache.cpp"] = mutate_in(UC, UC_ACQUIRE_SIG, FLAG,
                                             'if (std::getenv("GGML_SYCL_LEASE_HOLDERS") != nullptr) {')
    assert not claim_knob_read_once_out_of_line(UC, HOLD, sources)


def test_witness_site_census_at_warn():
    assert not claim_report_at_the_callers_level(
        mutate_in(UC, REPORT_SIG, SITE_AT_LEVEL, SITE_AT_LEVEL.replace("ggml_log_internal(level, ", "GGML_LOG_WARN("))
    )


def test_witness_holder_line_at_warn():
    assert not claim_report_at_the_callers_level(
        mutate_in(UC, HOLDER_REPORT_SIG, HOLDER_LINE,
                  HOLDER_LINE.replace("ggml_log_internal(level, ", "GGML_LOG_WARN("))
    )


def test_witness_hint_with_no_lease_held():
    assert not claim_report_at_the_callers_level(mutate_in(UC, REPORT_SIG, HINT_GUARD, "} else {"))


def test_witness_hint_with_the_knob_on():
    assert not claim_report_at_the_callers_level(mutate_in(UC, REPORT_SIG, HOLDER_CALL, HINT + ", 0u, 0);"))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
