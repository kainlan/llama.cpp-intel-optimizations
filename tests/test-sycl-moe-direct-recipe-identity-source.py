"""Source contract for llama.cpp-1sxq: the DIRECT MMID recipe asks the binder whether a lease has an exact identity.

Host-only: reads sources, builds nothing, loads no model and touches no device.

The closed Q1_0/NVFP4 decode route binds and retains each expert lease through
moe::canonical_allocation_integration::retain(). Before llama.cpp-1sxq, ggml_sycl_moe_query_route_capability() offered
the DIRECT recipe only to a lease whose handle kind was mem_handle_kind::DIRECT. That was a second, narrower answer to
the binder's question "does this lease carry an exact, allocator-minted allocation identity?". Every expert lease
published from a SYCL device buffer is a slice of the buffer's arena-zone handle (kind ARENA_RUNTIME), which retain()
accepts, so the kind test closed the route by construction: sycl-q1-nvfp4-admitted-device logged
`[MOE-DECODE-CANONICAL] ... kind=2` and then refused with capability=local-mmvq on every card.

This gate pins one source for that fact:
  - canonical_allocation_integration declares exact_identity(), the one identity test;
  - the capability query's DIRECT candidate asks exact_identity() of the lease and asks no other identity question
    (no mem_handle_kind test, no has_stable_owner_identity());
  - retain() mints only from what exact_identity() returns: it names the handle only to ask exact_identity() and to
    copy it into the owner, and reads members only of the predicate's result. This is an allowlist of uses, so a
    second identity check cannot come back through an accessor spelling nobody listed;
  - bind() still mints through retain();
  - exact_identity() keeps every refusal retain() had before, each one unconditional: an unresolved handle, an id-less
    WEIGHT handle, an ownerless non-arena handle (ownerless DIRECT, compatibility CHUNK_LEASE), and a zero id,
    generation or extent or an out-of-range device;
  - exact_identity() accepts in one place only, its last statement `return exact;`, so no early accept can skip a
    refusal.

Every claim is checked on COMMENT-STRIPPED, whitespace-normalized text and has a mutant that must make it fail.

Pytest-style (module-level test_* functions): register with llama_test_pytest.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_LEXEME_RE = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'",
    re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def norm(text: str) -> str:
    """Comment-stripped, whitespace-canonical text, so a clang-format re-wrap cannot turn a claim red."""
    t = re.sub(r"\s+", " ", strip_comments(text))
    t = re.sub(r"\( ", "(", t)
    t = re.sub(r" \)", ")", t)
    t = re.sub(r" ,", ",", t)
    t = re.sub(r",(?! )", ", ", t)
    return t


def body(text: str, signature: str) -> str:
    """The brace-balanced body of the first definition whose text starts with `signature` (string literals are
    skipped while balancing). Empty when there is none."""
    at = text.find(signature)
    if at < 0:
        return ""
    open_at = text.find("{", at + len(signature))
    if open_at < 0:
        return ""
    depth = 0
    i = open_at
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "\"'":
            j = i + 1
            while j < n and text[j] != ch:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_at:i + 1]
        i += 1
    return ""


SYCL = norm((ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text())
MEM = norm((ROOT / "ggml/src/ggml-sycl/mem-handle.cpp").read_text())
RETENTION = norm((ROOT / "ggml/src/ggml-sycl/moe-graph-retention.hpp").read_text())

QUERY_SIG = "static moe_route_capability ggml_sycl_moe_query_route_capability("
EXACT_SIG = ("std::optional<moe::canonical_allocation_identity> moe::canonical_allocation_integration::exact_identity("
             "const mem_handle & handle) noexcept")
RETAIN_SIG = ("std::optional<moe::retained_allocation_owner> moe::canonical_allocation_integration::retain("
              "const mem_handle & handle) noexcept")
BIND_SIG = "std::optional<moe::mmid_batch_binding> moe::canonical_allocation_integration::bind("
CLASS_SIG = "class canonical_allocation_integration final"

DECLARATION = ("static std::optional<canonical_allocation_identity> exact_identity("
               "const ggml_sycl::mem_handle & handle) noexcept;")
QUERY_IDENTITY = ("direct_recipe_candidate && "
                  "ggml_sycl::moe::canonical_allocation_integration::exact_identity(*direct_recipe_candidate)")
RETAIN_CALL = "const auto identity = exact_identity(handle);"
RETAIN_MINT = ("retained_allocation_owner(identity->allocation_id, identity->generation, identity->device, "
               "identity->extent, std::move(owner))")
RETAIN_COPY = "auto retained_handle = std::make_shared<mem_handle>(handle);"
# retain() may name the handle in exactly these two places: once to ask the predicate, once to copy it into the owner.
# Any other use reads the handle a second way, whatever field or accessor it goes through. A list of forbidden fields
# and accessors fails open on every spelling nobody listed (valid(), owns_allocation(), identity()), so the claim is an
# allowlist of uses instead.
RETAIN_HANDLE_USES = ("exact_identity(handle)", "std::make_shared<mem_handle>(handle)")
# Every member retain() reads is on the predicate's result. This also covers the copy of the handle and any alias of
# it, which the allowlist above cannot see because they do not spell `handle`.
RETAIN_MEMBER_RECEIVER = "identity"

REFUSE_UNRESOLVED = "if (!handle.cached_.ptr) { return std::nullopt; }"
REFUSE_IDLESS_WEIGHT = ("if (handle.kind_ == mem_handle_kind::WEIGHT && exact.allocation_id == 0) "
                        "{ return std::nullopt; }")
REFUSE_OWNERLESS = ("if (!handle.owned_alloc_ && !handle.is_arena() && handle.kind_ != mem_handle_kind::WEIGHT) "
                    "{ return std::nullopt; }")
REFUSE_INCOMPLETE = ("if (exact.allocation_id == 0 || exact.generation == 0 || exact.device < 0 || "
                     "exact.device >= static_cast<int>(execution::max_devices) || exact.extent == 0) "
                     "{ return std::nullopt; }")
REFUSALS = (REFUSE_UNRESOLVED, REFUSE_IDLESS_WEIGHT, REFUSE_OWNERLESS, REFUSE_INCOMPLETE)
PREDICATE_ACCEPT = "return exact;"
PREDICATE_REFUSE = "return std::nullopt;"
PREDICATE_GUARD = "mem_handle_lock_guard guard(handle.lock_);"


def static_candidate(query_body: str) -> str:
    """The `static_candidate` initializer in the query: from its declaration to the terminating semicolon."""
    m = re.search(r"const bool static_candidate = ([^;]*);", query_body)
    return m.group(1) if m else ""


# ---- one predicate per claim ----------------------------------------------------------------------------------------


def declares_predicate(class_body: str) -> bool:
    """canonical_allocation_integration declares exact_identity(), noexcept, returning the identity it tested."""
    return DECLARATION in class_body


def query_asks_the_binder(query_body: str) -> bool:
    """The DIRECT candidate asks exact_identity() of the lease it was handed."""
    return QUERY_IDENTITY in static_candidate(query_body)


def query_asks_no_second_identity_question(query_body: str) -> bool:
    """No handle-kind test and no has_stable_owner_identity() anywhere in the query: one identity test, not two."""
    return (bool(query_body) and "mem_handle_kind::" not in query_body and "->kind()" not in query_body
            and "has_stable_owner_identity" not in query_body)


def retain_mints_only_from_the_predicate(retain_body: str) -> bool:
    """retain() calls exact_identity() once and mints its owner from that result."""
    return retain_body.count("exact_identity(") == 1 and RETAIN_CALL in retain_body and RETAIN_MINT in retain_body


def _context(text: str, start: int, end: int) -> str:
    return text[max(0, start - 24):end + 24].strip()


def retain_stray_handle_uses(retain_body: str) -> list:
    """Where retain() names `handle` outside RETAIN_HANDLE_USES (each allowed use is consumed once), with context."""
    rest = retain_body
    for use in RETAIN_HANDLE_USES:
        rest = rest.replace(use, "", 1)
    return [_context(rest, m.start(), m.end()) for m in re.finditer(r"\bhandle\b", rest)]


def retain_stray_member_reads(retain_body: str) -> list:
    """The member reads (`.m`, `->m`) retain() makes on anything but RETAIN_MEMBER_RECEIVER, with context."""
    receiver = re.compile(r"\b" + RETAIN_MEMBER_RECEIVER + r"\s*$")
    return [
        _context(retain_body, m.start(), m.end()) for m in re.finditer(r"(?:\.|->)\s*[A-Za-z_]", retain_body)
        if not receiver.search(retain_body[:m.start()])
    ]


def retain_reads_the_handle_only_through_the_predicate(retain_body: str) -> bool:
    """retain() keeps no parallel copy of the identity checks: it names the handle once in each allowed use and nowhere
    else, and reads members only of what exact_identity() returned."""
    return (bool(retain_body) and all(retain_body.count(use) == 1 for use in RETAIN_HANDLE_USES)
            and not retain_stray_handle_uses(retain_body) and not retain_stray_member_reads(retain_body))


def bind_goes_through_retain(bind_body: str) -> bool:
    """bind() mints its owner through retain(), so it inherits the one identity test."""
    return "auto owner = retain(handle);" in bind_body and "exact_identity(" not in bind_body


def enclosing_block_opener(text: str, at: int):
    """The text before the innermost `{` that encloses position `at`, right-stripped; None when nothing encloses it."""
    depth = 0
    for i in range(at - 1, -1, -1):
        if text[i] == "}":
            depth += 1
        elif text[i] == "{":
            if depth == 0:
                return text[:i].rstrip()
            depth -= 1
    return None


def _starts_a_statement(before: str) -> bool:
    return before == "" or before.endswith((";", "{", "}"))


def refusal_is_unconditional(exact_body: str, refusal: str) -> bool:
    """`refusal` is present, starts a statement of its own, and sits in the function body or a bare scope block: no
    if, else, loop or lambda (braced or not) decides whether it runs."""
    at = exact_body.find(refusal)
    if at < 0:
        return False
    opener = enclosing_block_opener(exact_body, at)
    return opener is not None and _starts_a_statement(opener) and _starts_a_statement(exact_body[:at].rstrip())


def predicate_refuses_what_retain_refused(exact_body: str) -> bool:
    """exact_identity() keeps every refusal retain() had, each unconditional: unresolved, id-less WEIGHT, ownerless
    non-arena, and an incomplete identity (zero id, generation or extent, or an out-of-range device)."""
    return all(refusal_is_unconditional(exact_body, r) for r in REFUSALS)


def predicate_returns(exact_body: str) -> list:
    return re.findall(r"\breturn\b[^;]*;", exact_body)


def predicate_accepts_only_after_every_refusal(exact_body: str) -> bool:
    """exact_identity() accepts in one place: its last statement is `return exact;` and every earlier return is
    `return std::nullopt;`. An early accept, in any spelling, would skip the refusals after it."""
    returns = predicate_returns(exact_body)
    return (bool(returns) and returns[-1] == PREDICATE_ACCEPT and all(r == PREDICATE_REFUSE for r in returns[:-1])
            and exact_body.endswith(PREDICATE_ACCEPT + " }"))


CLASS_BODY = body(RETENTION, CLASS_SIG)
QUERY_BODY = body(SYCL, QUERY_SIG)
EXACT_BODY = body(MEM, EXACT_SIG)
RETAIN_BODY = body(MEM, RETAIN_SIG)
BIND_BODY = body(MEM, BIND_SIG)


def test_bodies_are_found():
    assert CLASS_BODY, "class canonical_allocation_integration not found in moe-graph-retention.hpp"
    assert QUERY_BODY, "ggml_sycl_moe_query_route_capability() not found: the signature this gate keys on moved"
    assert EXACT_BODY, "canonical_allocation_integration::exact_identity() is not defined in mem-handle.cpp"
    assert RETAIN_BODY, "canonical_allocation_integration::retain() not found: the signature this gate keys on moved"
    assert BIND_BODY, "canonical_allocation_integration::bind() not found: the signature this gate keys on moved"
    assert static_candidate(QUERY_BODY), "the query has no `const bool static_candidate = ...;` initializer"


def test_binder_declares_the_one_identity_test():
    assert declares_predicate(CLASS_BODY), (
        "canonical_allocation_integration must declare " + DECLARATION + " -- the one identity test the query and "
        "retain() share")


def test_query_asks_the_binder():
    assert query_asks_the_binder(QUERY_BODY), (
        "the DIRECT candidate in ggml_sycl_moe_query_route_capability() must ask "
        "canonical_allocation_integration::exact_identity() of the lease (llama.cpp-1sxq)")


def test_query_asks_no_handle_kind():
    assert query_asks_no_second_identity_question(QUERY_BODY), (
        "ggml_sycl_moe_query_route_capability() must not test a handle kind or has_stable_owner_identity(): an "
        "arena slice is what every unified-cache VRAM buffer yields, so a kind test closes the DIRECT route by "
        "construction (llama.cpp-1sxq)")


def test_retain_mints_only_from_the_predicate():
    assert retain_mints_only_from_the_predicate(RETAIN_BODY), (
        "retain() must call exact_identity(handle) once and mint retained_allocation_owner from its result")
    assert retain_reads_the_handle_only_through_the_predicate(RETAIN_BODY), (
        "retain() may name the handle only in " + " and ".join(RETAIN_HANDLE_USES) + ", each once, and may read "
        "members only of the predicate's result; any other read is a parallel copy of the identity checks, a second "
        "source. Stray handle uses: " + "; ".join(retain_stray_handle_uses(RETAIN_BODY)) + ". Stray member reads: "
        + "; ".join(retain_stray_member_reads(RETAIN_BODY)))


def test_bind_goes_through_retain():
    assert bind_goes_through_retain(BIND_BODY), "bind() must mint its owner through retain(handle)"


def test_predicate_refuses_what_retain_refused():
    assert predicate_refuses_what_retain_refused(EXACT_BODY), (
        "exact_identity() must refuse unresolved, id-less WEIGHT, ownerless non-arena (ownerless DIRECT, "
        "CHUNK_LEASE) and incomplete identities, each unconditionally; widening it widens what the DIRECT route binds")


def test_predicate_accepts_only_after_every_refusal():
    assert predicate_accepts_only_after_every_refusal(EXACT_BODY), (
        "exact_identity() must accept only in its last statement `return exact;`, with every earlier return "
        "`return std::nullopt;`; an early accept skips the refusals after it. Returns: "
        + " | ".join(predicate_returns(EXACT_BODY)))


# ---- mutants: each must turn its own claim red ----------------------------------------------------------------------


def _mutate(b: str, old: str, new: str) -> str:
    assert old in b, f"mutant anchor missing, so the mutant would be vacuous: {old!r}"
    return b.replace(old, new, 1)


def test_mutant_direct_kind_literal_reappears_in_query():
    m = _mutate(QUERY_BODY, QUERY_IDENTITY,
                QUERY_IDENTITY + " && direct_recipe_candidate->kind() == ggml_sycl::mem_handle_kind::DIRECT")
    assert not query_asks_no_second_identity_question(m), "a DIRECT kind literal in the query must be caught"
    assert query_asks_the_binder(m)


def test_mutant_direct_kind_replaces_the_binder_in_query():
    m = _mutate(QUERY_BODY, QUERY_IDENTITY,
                "direct_recipe_candidate && direct_recipe_candidate->kind() == ggml_sycl::mem_handle_kind::DIRECT")
    assert not query_asks_the_binder(m)
    assert not query_asks_no_second_identity_question(m)


def test_mutant_stable_owner_identity_added_to_query():
    m = _mutate(QUERY_BODY, QUERY_IDENTITY, QUERY_IDENTITY + " && direct_recipe_candidate->has_stable_owner_identity()")
    assert not query_asks_no_second_identity_question(m), "a second identity question in the query must be caught"


def test_mutant_query_drops_the_identity_test():
    m = _mutate(QUERY_BODY, QUERY_IDENTITY, "direct_recipe_candidate")
    assert not query_asks_the_binder(m), "a DIRECT candidate with no identity test must be caught"


def test_mutant_retain_bypasses_the_predicate():
    # retain() reads the fields itself and never calls the predicate.
    m = _mutate(RETAIN_BODY, RETAIN_CALL,
                "const auto identity = std::optional<canonical_allocation_identity>(canonical_allocation_identity{ "
                "handle.canonical_allocation_id_, handle.canonical_generation_, handle.device_, "
                "handle.canonical_extent_ });")
    assert not retain_mints_only_from_the_predicate(m), "retain() bypassing exact_identity() must be caught"
    assert not retain_reads_the_handle_only_through_the_predicate(m)


def test_mutant_retain_keeps_a_parallel_check():
    # retain() calls the predicate but also keeps its own copy of one check.
    m = _mutate(RETAIN_BODY, RETAIN_CALL, RETAIN_CALL + " if (handle.kind_ == mem_handle_kind::WEIGHT) { return "
                "std::nullopt; }")
    assert retain_mints_only_from_the_predicate(m)
    assert not retain_reads_the_handle_only_through_the_predicate(m), (
        "a parallel identity check left in retain() must be caught")


def test_mutant_retain_keeps_a_parallel_kind_check_through_accessors():
    # The private-field list alone missed this spelling of the same check.
    m = _mutate(RETAIN_BODY, RETAIN_CALL, RETAIN_CALL + " if (handle.kind() == mem_handle_kind::CHUNK_LEASE) { return "
                "std::nullopt; }")
    assert retain_mints_only_from_the_predicate(m)
    assert not retain_reads_the_handle_only_through_the_predicate(m), (
        "an accessor-form kind check left in retain() must be caught")
    assert any("handle.kind()" in use for use in retain_stray_handle_uses(m))


def test_mutant_retain_keeps_a_parallel_owner_check_through_accessors():
    m = _mutate(RETAIN_BODY, RETAIN_CALL, RETAIN_CALL + " if (!handle.has_stable_owner_identity() || handle.device() < "
                "0) { return std::nullopt; }")
    assert retain_mints_only_from_the_predicate(m)
    assert not retain_reads_the_handle_only_through_the_predicate(m), (
        "an accessor-form owner/device check in retain() must be caught")
    assert len(retain_stray_handle_uses(m)) == 2


def test_mutant_retain_checks_ownership_through_accessors():
    # Neither name was on the old forbidden-field list, so that list let this through.
    m = _mutate(RETAIN_BODY, RETAIN_CALL, RETAIN_CALL + " if (!handle.owns_allocation() && !handle.valid()) { return "
                "std::nullopt; }")
    assert retain_mints_only_from_the_predicate(m)
    assert not retain_reads_the_handle_only_through_the_predicate(m), (
        "an owns_allocation()/valid() check left in retain() must be caught")


def test_mutant_retain_checks_the_identity_through_its_accessor():
    for check in ("if (handle.identity().allocation_id == 0) { return std::nullopt; }",
                  "if (!handle.identity().valid()) { return std::nullopt; }"):
        m = _mutate(RETAIN_BODY, RETAIN_CALL, RETAIN_CALL + " " + check)
        assert retain_mints_only_from_the_predicate(m)
        assert not retain_reads_the_handle_only_through_the_predicate(m), (
            "an identity() check left in retain() must be caught: " + check)


def test_mutant_retain_checks_its_copy_of_the_handle():
    # The copy does not spell `handle`, so only the member-read allowlist sees this.
    m = _mutate(RETAIN_BODY, RETAIN_COPY, RETAIN_COPY + " if (!retained_handle->owns_allocation()) { return "
                "std::nullopt; }")
    assert not retain_stray_handle_uses(m)
    assert retain_stray_member_reads(m), "an identity check on retain()'s copy of the handle must be caught"
    assert not retain_reads_the_handle_only_through_the_predicate(m)


def test_mutant_retain_hands_the_handle_to_a_second_check():
    # No member read at all, so only the handle-use allowlist sees this.
    m = _mutate(RETAIN_BODY, RETAIN_CALL, RETAIN_CALL + " if (!has_exact_range(handle)) { return std::nullopt; }")
    assert not retain_stray_member_reads(m)
    assert retain_stray_handle_uses(m), "the handle passed to a second identity check must be caught"
    assert not retain_reads_the_handle_only_through_the_predicate(m)


def test_mutant_bind_bypasses_retain():
    m = _mutate(BIND_BODY, "auto owner = retain(handle);", "auto owner = std::optional<retained_allocation_owner>();")
    assert not bind_goes_through_retain(m)


def test_mutant_predicate_accepts_an_idless_handle():
    m = _mutate(EXACT_BODY, "exact.allocation_id == 0 || exact.generation", "exact.generation")
    assert not predicate_refuses_what_retain_refused(m), "accepting a zero allocation id must be caught"


def test_mutant_predicate_accepts_an_idless_weight():
    m = _mutate(EXACT_BODY, REFUSE_IDLESS_WEIGHT, "")
    assert not predicate_refuses_what_retain_refused(m), "accepting an id-less WEIGHT handle must be caught"


def test_mutant_predicate_accepts_an_ownerless_handle():
    m = _mutate(EXACT_BODY, REFUSE_OWNERLESS, "")
    assert not predicate_refuses_what_retain_refused(m), "accepting an ownerless DIRECT/CHUNK_LEASE must be caught"


def test_mutant_predicate_accepts_an_unresolved_handle():
    m = _mutate(EXACT_BODY, REFUSE_UNRESOLVED, "")
    assert not predicate_refuses_what_retain_refused(m), "accepting an unresolved handle must be caught"


def test_mutant_predicate_accepts_early():
    # Every refusal string is still present, so presence alone missed this.
    m = _mutate(EXACT_BODY, PREDICATE_GUARD, PREDICATE_GUARD + " if (handle.kind_ == mem_handle_kind::CHUNK_LEASE) { "
                "return exact; }")
    assert predicate_refuses_what_retain_refused(m)
    assert not predicate_accepts_only_after_every_refusal(m), "an early accept must be caught"


def test_mutant_predicate_accepts_early_in_another_spelling():
    # `return exact;` still occurs exactly once and last, so a count of it alone would miss this.
    m = _mutate(EXACT_BODY, PREDICATE_GUARD, PREDICATE_GUARD + " if (handle.kind_ == mem_handle_kind::CHUNK_LEASE) { "
                "return std::make_optional(exact); }")
    assert m.count(PREDICATE_ACCEPT) == 1 and m.endswith(PREDICATE_ACCEPT + " }")
    assert not predicate_accepts_only_after_every_refusal(m), "an early accept in any spelling must be caught"


def test_mutant_predicate_refusal_under_a_condition():
    m = _mutate(EXACT_BODY, REFUSE_OWNERLESS, "if (handle.kind_ != mem_handle_kind::CHUNK_LEASE) { " + REFUSE_OWNERLESS
                + " }")
    assert predicate_accepts_only_after_every_refusal(m)
    assert not predicate_refuses_what_retain_refused(m), "a refusal skipped for one handle kind must be caught"


def test_mutant_predicate_refusal_under_a_braceless_condition():
    m = _mutate(EXACT_BODY, REFUSE_UNRESOLVED, "if (handle.kind_ != mem_handle_kind::CHUNK_LEASE) " + REFUSE_UNRESOLVED)
    assert predicate_accepts_only_after_every_refusal(m)
    assert not predicate_refuses_what_retain_refused(m), "a refusal under a braceless if must be caught"


def test_mutant_declaration_dropped():
    m = _mutate(CLASS_BODY, DECLARATION, "")
    assert not declares_predicate(m)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
