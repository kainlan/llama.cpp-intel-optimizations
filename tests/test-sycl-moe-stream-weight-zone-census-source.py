"""Source contract: the short ring reserve's WEIGHT zone line is a diagnostic that cannot change what the reserve does.

Host-only: reads sources, builds nothing, loads no model and touches no device.

moe_stream_phase::ring_ready() prints the line from the reserve-failed branch, after it has set reserve_failed_ and
before the loan step. The line's census allocates host containers, so an exception from it would leave ring_ready()
past the flag and skip the loan step and the pending ring's hold: a diagnostic would change the reserve. This gate
pins that it cannot:
  - moe_stream_ring_warn_weight_zone() is declared and defined noexcept, its whole body is one try, and its catch
    prints a literal WARN that allocates nothing;
  - unified_cache_zone_tag_census() is declared, befriended and defined noexcept, its whole body is one try, and its
    catch resets `out` and returns false;
  - the census refuses a zone outside the arena's zone table before it reads one;
  - both of the ring's WARNs carry the device as `device=%d`, so one field parser reads either.

Neither function is compiled by the host-only ring sizing test (its SIZING_ONLY build leaves out everything that
reads the cache), so the contract is pinned on the text. Every claim is checked on COMMENT-STRIPPED, whitespace-
normalized text (sycl_gate.norm) and has a mutant that must make it fail.

Pytest-style (module-level test_* functions): register with llama_test_pytest.
"""

from sycl_gate import body, norm, read

RING_HPP = norm(read("ggml/src/ggml-sycl/moe-stream-ring.hpp"))
RING_CPP = norm(read("ggml/src/ggml-sycl/moe-stream-ring.cpp"))
CACHE_HPP = norm(read("ggml/src/ggml-sycl/unified-cache.hpp"))
CACHE = norm(read("ggml/src/ggml-sycl/unified-cache.cpp"))


def _once(text: str, old: str, new: str) -> str:
    """`text` (already normalized) with the one occurrence of `old` replaced by `new`: an anchor that matches twice
    would let a mutant edit the wrong copy."""
    assert text.count(old) == 1, f"mutant anchor must match exactly once: {old!r} x{text.count(old)}"
    return text.replace(old, new, 1)


WARN_PARAMS = "moe_stream_ring_warn_weight_zone(int device, uint32_t slots_held, size_t slot_bytes)"
WARN_DECL = "void " + WARN_PARAMS + " noexcept;"
WARN_SIG = "void " + WARN_PARAMS + " noexcept {"
WARN_HANDLER = ('{ GGML_LOG_WARN("[MOE-STREAM-RING] WEIGHT zone at the short reserve: device=%d census failed\\n", '
                'device); }')
NO_ARENA_WARN = ('"[MOE-STREAM-RING] WEIGHT zone at the short reserve: device=%d has no cache or no active '
                 'arena\\n"')

CENSUS_PARAMS = "unified_cache_zone_tag_census(int device, vram_zone_id zone, zone_tag_census & out)"
CENSUS_DECL = "bool " + CENSUS_PARAMS + " noexcept;"
CENSUS_FRIEND = "friend bool " + CENSUS_PARAMS + " noexcept;"
CENSUS_SIG = "bool " + CENSUS_PARAMS + " noexcept {"
CENSUS_HANDLER = "{ out = zone_tag_census(); return false; }"
ZONE_GUARD = "if (zone >= vram_zone_id::COUNT) { return false; }"
FIRST_ARENA_READ = "unified_cache * cache = get_existing_unified_cache_for_device(device);"


def _is_one_try(fn_body: str, handler: str) -> bool:
    """`fn_body` (a brace-balanced body) is exactly `{ try { ... } catch (...) <handler> }`: the try covers every
    statement, and the catch is the one handler given."""
    if not fn_body.startswith("{ try {"):
        return False
    tail = " catch (...) " + handler + " }"
    if not fn_body.endswith(tail):
        return False
    try_block = body(fn_body[2:], "try {")
    return bool(try_block) and fn_body == "{ try " + try_block + tail


# ---- (a) the ring's WARN cannot throw ----------------------------------------------------------------------------


def claim_warn_is_noexcept_and_one_try(hpp: str, cpp: str) -> bool:
    b = body(cpp, WARN_SIG)
    return hpp.count(WARN_DECL) == 1 and bool(b) and _is_one_try(b, WARN_HANDLER)


def test_warn_is_noexcept_and_one_try():
    assert claim_warn_is_noexcept_and_one_try(RING_HPP, RING_CPP)


def test_mutant_warn_declared_without_noexcept_fails():
    assert not claim_warn_is_noexcept_and_one_try(_once(RING_HPP, WARN_DECL, "void " + WARN_PARAMS + ";"), RING_CPP)


def test_mutant_warn_defined_without_noexcept_fails():
    mutant = _once(RING_CPP, WARN_SIG, "void " + WARN_PARAMS + " {")
    assert not claim_warn_is_noexcept_and_one_try(RING_HPP, mutant)


def test_mutant_warn_without_try_fails():
    """The body's statements outside any try: a throw would leave the function, which noexcept turns into
    std::terminate."""
    b = body(RING_CPP, WARN_SIG)
    inner = body(b[2:], "try {")
    mutant = _once(RING_CPP, b, inner)
    assert not claim_warn_is_noexcept_and_one_try(RING_HPP, mutant)


def test_mutant_warn_handler_allocates_fails():
    """A handler that builds a string can throw again from the handler."""
    heap = '{ GGML_LOG_WARN("%s\\n", std::string("census failed").c_str()); }'
    mutant = _once(RING_CPP, " catch (...) " + WARN_HANDLER, " catch (...) " + heap)
    assert not claim_warn_is_noexcept_and_one_try(RING_HPP, mutant)


def test_mutant_statement_after_the_try_fails():
    b = body(RING_CPP, WARN_SIG)
    mutant = _once(RING_CPP, b, b[:-1] + 'GGML_LOG_WARN("after\\n"); }')
    assert not claim_warn_is_noexcept_and_one_try(RING_HPP, mutant)


# ---- (b) the census cannot throw ---------------------------------------------------------------------------------


def _free_decl_count(hpp: str) -> int:
    """The census's free-function declarations: CENSUS_DECL is the friend declaration's tail too, so those are
    subtracted."""
    return hpp.count(CENSUS_DECL) - hpp.count(CENSUS_FRIEND)


def claim_census_is_noexcept_and_one_try(hpp: str, cpp: str) -> bool:
    b = body(cpp, CENSUS_SIG)
    return (_free_decl_count(hpp) == 1 and hpp.count(CENSUS_FRIEND) == 1 and bool(b)
            and _is_one_try(b, CENSUS_HANDLER))


def test_census_is_noexcept_and_one_try():
    assert claim_census_is_noexcept_and_one_try(CACHE_HPP, CACHE)


def test_mutant_census_declared_without_noexcept_fails():
    """The free declaration, not the friend's: the one CENSUS_DECL occurrence no `friend ` precedes."""
    at = [i for i in range(len(CACHE_HPP)) if CACHE_HPP.startswith(CENSUS_DECL, i)
          and not CACHE_HPP[:i].endswith("friend ")]
    assert len(at) == 1
    mutant = CACHE_HPP[:at[0]] + "bool " + CENSUS_PARAMS + ";" + CACHE_HPP[at[0] + len(CENSUS_DECL):]
    assert not claim_census_is_noexcept_and_one_try(mutant, CACHE)


def test_mutant_census_friend_without_noexcept_fails():
    mutant = _once(CACHE_HPP, CENSUS_FRIEND, "friend bool " + CENSUS_PARAMS + ";")
    assert not claim_census_is_noexcept_and_one_try(mutant, CACHE)


def test_mutant_census_defined_without_noexcept_fails():
    mutant = _once(CACHE, CENSUS_SIG, "bool " + CENSUS_PARAMS + " {")
    assert not claim_census_is_noexcept_and_one_try(CACHE_HPP, mutant)


def test_mutant_census_without_try_fails():
    b = body(CACHE, CENSUS_SIG)
    inner = body(b[2:], "try {")
    mutant = _once(CACHE, b, inner)
    assert not claim_census_is_noexcept_and_one_try(CACHE_HPP, mutant)


def test_mutant_census_handler_keeps_partial_out_fails():
    """A caller reading `out` after a false return would print a half-filled census."""
    mutant = _once(CACHE, " catch (...) " + CENSUS_HANDLER, " catch (...) { return false; }")
    assert not claim_census_is_noexcept_and_one_try(CACHE_HPP, mutant)


# ---- (c) the census refuses a zone outside the table before reading one -------------------------------------------


def claim_census_guards_the_zone(cpp: str) -> bool:
    b = body(cpp, CENSUS_SIG)
    at_guard = b.find(ZONE_GUARD)
    at_read = b.find(FIRST_ARENA_READ)
    return b.count(ZONE_GUARD) == 1 and 0 <= at_guard < at_read


def test_census_guards_the_zone():
    assert claim_census_guards_the_zone(CACHE)


def test_mutant_zone_guard_dropped_fails():
    assert not claim_census_guards_the_zone(_once(CACHE, ZONE_GUARD, ""))


def test_mutant_zone_guard_after_the_read_fails():
    """Built inside the census body: the lookup line is spelled the same in other functions of the file."""
    b = body(CACHE, CENSUS_SIG)
    moved = _once(_once(b, ZONE_GUARD, ""), FIRST_ARENA_READ, FIRST_ARENA_READ + " " + ZONE_GUARD)
    assert not claim_census_guards_the_zone(_once(CACHE, b, moved))


# ---- (d) both WARNs carry device= ------------------------------------------------------------------------------


def claim_warns_carry_device_field(cpp: str) -> bool:
    b = body(cpp, WARN_SIG)
    return bool(b) and NO_ARENA_WARN in b and "device %d" not in b


def test_warns_carry_device_field():
    assert claim_warns_carry_device_field(RING_CPP)


def test_mutant_no_arena_warn_without_equals_fails():
    old = "device=%d has no cache"
    mutant = _once(RING_CPP, old, "device %d has no cache")
    assert not claim_warns_carry_device_field(mutant)
