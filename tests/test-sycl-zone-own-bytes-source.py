"""Source contract: every VRAM arena zone counts its own bytes.

KV and WEIGHT can share one physical allocator (a single-chunk arena gives WEIGHT none of its own; an N-chunk arena
serves both from the weight chunks). A zone counter that stored the allocator's total took the other zone's bytes as
its own, and a WEIGHT free that stored into KV's counter never lowered WEIGHT: a freed model read as live WEIGHT
bytes and refused the next load's zone rebuild. These checks pin the rule in unified_cache::zone_alloc() and
unified_cache::zone_free():

- no arm writes a zone counter itself (an allocator total stored, or a delta applied by hand): every write goes
  through zone_used_allocate() / zone_used_free() (zone-used-count.hpp), which move a counter by exactly the
  allocator's delta for that one operation, and test-zone-sizing Case 34 checks that arithmetic;
- a free lowers the counter of the zone the exact-allocation record names, which
  arena_unregister_exact_if_unleased_locked() reads from the record it unregisters, never the zone the caller names;
- so a WEIGHT free writes no other zone's counter.

Host-only text assertions, no SYCL device. llama_test_pytest hands this file to pytest.main(), so the checks live
inside test_*() functions. Checks read COMMENT-STRIPPED, whitespace-normalised text. Each check has a mutation witness
that applies the regression it guards inside the one function it names and must make the check fail.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UC = (ROOT / "ggml/src/ggml-sycl/unified-cache.cpp").read_text()

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


def mutate_in_func(src: str, sig: str, old: str, new: str, nth: int = 0) -> str:
    """`src` as code() with the nth `old` inside the function `sig` replaced by `new`; never text outside it."""
    c = code(src)
    s = c.find(sig)
    body = block_at(c, s + len(sig)) if s != -1 else ""
    assert body and body.count(old) > nth, f"witness anchor {old!r} (occurrence {nth}) is not in {sig!r}"
    k = -1
    for _ in range(nth + 1):
        k = body.find(old, k + 1)
    b = c.find(body, s)
    return c[:b] + body[:k] + new + body[k + len(old) :] + c[b + len(body) :]


ALLOC_SIG = "void * unified_cache::zone_alloc("
FREE_SIG = "bool unified_cache::zone_free(vram_zone_id zone, void * ptr)"
UNREG_SIG = "bool unified_cache::arena_unregister_exact_if_unleased_locked("

# A write of a zone counter by hand: an allocator total stored, a delta applied, or an assignment.
COUNTER_WRITE_RE = re.compile(
    r"\.used\s*(?:\.\s*(?:store|fetch_add|fetch_sub|exchange|compare_exchange_\w+)\s*\(|[-+]?=(?!=)|\+\+|--)"
)
# A call of a physical allocator outside the helper: allocate()/allocate_at()/... or free() reached through `.` or `->`.
RAW_ALLOCATOR_RE = re.compile(r"(?:->|\.)\s*(?:allocate\w*|free)\s*\(")
UNREG_CALL = "arena_unregister_exact_if_unleased_locked(zone, ptr, record_zone)"
RECORD_COUNTER = "arena_zones_[static_cast<int>(record_zone)].used"
HELPER_FREE = "zone_used_free("


def call_args(text: str, start: int) -> str:
    """The parenthesised argument list of the call whose name ends just before `start`'s '('."""
    o = text.find("(", start)
    depth = 0
    for i in range(o, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[o + 1 : i]
    return ""


def claim_no_arm_writes_a_counter_itself(uc: str) -> bool:
    c = code(uc)
    bodies = (fn_body(c, ALLOC_SIG), fn_body(c, FREE_SIG))
    return all(b for b in bodies) and not any(COUNTER_WRITE_RE.search(b) for b in bodies)


def claim_free_takes_the_record_zone(uc: str) -> bool:
    c = code(uc)
    unreg = fn_body(c, UNREG_SIG)
    sig_at = c.find(UNREG_SIG)
    unreg_params = c[sig_at : c.find("{", sig_at)] if sig_at != -1 else ""
    # the out-parameter is set from the record it unregisters, and from nothing else
    record_at = unreg.find("const allocator_group_record & exact = found->second;")
    if not (
        "vram_zone_id & record_zone" in unreg_params
        and record_at != -1
        and re.findall(r"\brecord_zone\s*=(?!=)\s*([^;]*);", unreg) == ["exact.zone"]
        and record_at < unreg.find("record_zone = exact.zone;")
    ):
        return False
    free = fn_body(c, FREE_SIG)
    starts = [m.start() for m in re.finditer(re.escape(HELPER_FREE), free)]
    if not starts:
        return False
    prev = 0
    for s in starts:
        # each free is preceded, since the previous one, by the unregister that reads the record's zone ...
        if UNREG_CALL not in free[prev:s]:
            return False
        # ... and lowers that zone's counter
        args = call_args(free, s)
        if not args.rstrip().endswith(", " + RECORD_COUNTER):
            return False
        prev = s + len(HELPER_FREE) + len(args)
    return True


def claim_weight_free_writes_no_other_counter(uc: str) -> bool:
    free = fn_body(code(uc), FREE_SIG)
    weight_arm = block_at(free, free.find("if (!alloc && zone == vram_zone_id::WEIGHT) {"))
    return (
        bool(weight_arm)
        and RECORD_COUNTER in weight_arm
        and "kv.used" not in weight_arm
        # the only zone counter zone_free names anywhere is the record's
        and not re.search(r"\.used\b", free.replace(RECORD_COUNTER, ""))
    )


def claim_arms_call_the_helper(uc: str) -> bool:
    c = code(uc)
    alloc, free = fn_body(c, ALLOC_SIG), fn_body(c, FREE_SIG)
    return (
        bool(alloc and free)
        and "zone_used_allocate(" in alloc
        and HELPER_FREE in free
        and not RAW_ALLOCATOR_RE.search(alloc)
        and not RAW_ALLOCATOR_RE.search(free)
        and '#include "zone-used-count.hpp"' in uc
    )


def test_no_arm_writes_a_counter_itself():
    assert claim_no_arm_writes_a_counter_itself(UC), (
        "zone_alloc() or zone_free() writes a zone counter itself (an allocator total stored, or a delta applied by "
        "hand) instead of through zone_used_allocate() / zone_used_free()"
    )


def test_free_takes_the_record_zone():
    assert claim_free_takes_the_record_zone(UC), (
        "a zone_free() arm lowers a counter other than the one of the zone its exact-allocation record names, or "
        "arena_unregister_exact_if_unleased_locked() does not hand back the record's zone"
    )


def test_weight_free_writes_no_other_counter():
    assert claim_weight_free_writes_no_other_counter(UC), (
        "zone_free() names a zone counter other than the record's: a WEIGHT free could lower KV (or leave WEIGHT "
        "stale)"
    )


def test_arms_call_the_helper():
    assert claim_arms_call_the_helper(UC), (
        "zone_alloc() or zone_free() calls a physical allocator directly instead of through zone_used_allocate() / "
        "zone_used_free(), so its counter can drift from the allocator"
    )


SINGLE_ALLOC = "const size_t offset = zone_used_allocate(*alloc, z.used, size, align, &zone_count);"
ARM_FREE = "zone_used_free(*alloc, zone_offset, " + RECORD_COUNTER + ");"
CHUNK_FREE = "zone_used_free(*target->allocator, chunk_off, " + RECORD_COUNTER + ");"
CHUNK_ROLLBACK = "zone_used_free(*wca.allocator, offset, z.used);"
# What both arms did before each zone counted its own bytes: the allocator's total stored into the counter.
TOTAL_STORE = " z.used.store(alloc->used(), std::memory_order_relaxed);"


def test_witness_alloc_stores_the_allocator_total():
    assert not claim_no_arm_writes_a_counter_itself(
        mutate_in_func(UC, ALLOC_SIG, SINGLE_ALLOC, SINGLE_ALLOC + TOTAL_STORE)
    )


def test_witness_free_stores_the_allocator_total():
    # the arm for a zone with its own allocator is the second helper free on *alloc
    assert not claim_no_arm_writes_a_counter_itself(mutate_in_func(UC, FREE_SIG, ARM_FREE, ARM_FREE + TOTAL_STORE, 1))


def test_witness_free_lowers_the_callers_zone():
    assert not claim_free_takes_the_record_zone(
        mutate_in_func(UC, FREE_SIG, CHUNK_FREE, "zone_used_free(*target->allocator, chunk_off, z.used);")
    )


def test_witness_unregister_hands_back_the_callers_zone():
    assert not claim_free_takes_the_record_zone(
        mutate_in_func(UC, UNREG_SIG, "record_zone = exact.zone;", "record_zone = zone;")
    )


def test_witness_weight_free_lowers_kv():
    # the single-chunk WEIGHT arm is the first helper free on *alloc
    assert not claim_weight_free_writes_no_other_counter(
        mutate_in_func(UC, FREE_SIG, ARM_FREE, "zone_used_free(*alloc, zone_offset, kv.used);", 0)
    )


def test_witness_alloc_bypasses_the_helper():
    assert not claim_arms_call_the_helper(
        mutate_in_func(UC, ALLOC_SIG, SINGLE_ALLOC, "const size_t offset = alloc->allocate(size, align);")
    )


def test_witness_alloc_rollback_bypasses_the_helper():
    assert not claim_arms_call_the_helper(mutate_in_func(UC, ALLOC_SIG, CHUNK_ROLLBACK, "wca.allocator->free(offset);"))


def test_witness_free_bypasses_the_helper():
    assert not claim_arms_call_the_helper(mutate_in_func(UC, FREE_SIG, ARM_FREE, "alloc->free(zone_offset);", 1))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
