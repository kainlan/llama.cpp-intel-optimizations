"""Source contract for llama.cpp-n8da: the MMID owner bound is the registry's exposed count, never the physical count.

Host-only: reads sources, builds nothing, loads no model and touches no device.

moe_mmid_owner_admissible(owner, exposed_device_count) admits an owner only below the count it is given, and
test-moe-mmid-workspace-plan pins that predicate. What that host case cannot see is the wiring: which count the two
planner call sites pass. The right one is ggml_sycl_info().device_count, the count ggml_backend_sycl_reg() sizes its
device list from and the bound ggml_backend_sycl_init() refuses at. The wrong one is ggml_sycl_info().total_gpu_count,
the physical count the multi-GPU planner budgets: passing it would plan an MMID workspace on a hidden GPU that no
backend context can bind, and materialization is all-or-nothing, so the primary's pool would be withheld too.

This gate pins:
  - moe_mmid_exposed_owner_count(), the one place the bound and its rationale live, returns exactly
    ggml_sycl_info().device_count;
  - in plan_moe_mmid_workspaces() and validate_moe_mmid_execution_owners(), `exposed_devices` is declared exactly once
    (no shadowing redeclaration), as `const int exposed_devices = moe_mmid_exposed_owner_count();`, before the first
    admission check;
  - every moe_mmid_owner_admissible() call passes exactly `exposed_devices` as its count, so a call that bypasses the
    helper fails (at least one call in the planner, at least two in the validator: the on-device owner and the
    alternate owner);
  - neither function names total_gpu_count anywhere.

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


CACHE = norm((ROOT / "ggml/src/ggml-sycl/unified-cache.cpp").read_text())

PLAN_SIG = "static void plan_moe_mmid_workspaces(placement_plan & plan,"
VALIDATE_SIG = "static void validate_moe_mmid_execution_owners(placement_plan & plan)"
HELPER_SIG = "static int moe_mmid_exposed_owner_count()"

HELPER_BODY_EXPECTED = "{ return ggml_sycl_info().device_count; }"
BOUND_DECL = "const int exposed_devices = moe_mmid_exposed_owner_count();"
ADMISSIBLE_CALL = re.compile(r"moe_mmid_owner_admissible\(")
# Any declaration of the name, whatever its qualifiers: a second one shadows the bound.
ANY_DECL = re.compile(r"\b(?:int|auto|long|unsigned|size_t|int32_t|int64_t)\s+exposed_devices\s*[={]")


def admissible_counts(b: str) -> list:
    """The count argument (the text after the top-level comma) of every moe_mmid_owner_admissible() call in `b`.
    A call whose argument list cannot be split into exactly two top-level arguments yields None."""
    counts = []
    for m in ADMISSIBLE_CALL.finditer(b):
        i = m.end()
        depth = 1
        args = [""]
        while i < len(b) and depth > 0:
            ch = b[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            if ch == "," and depth == 1:
                args.append("")
            else:
                args[-1] += ch
            i += 1
        counts.append(args[1].strip() if len(args) == 2 else None)
    return counts


# ---- one predicate per claim ----------------------------------------------------------------------------------------


def helper_returns_registry_count(helper_body: str) -> bool:
    """The helper is the registry's exposed count and nothing else."""
    return helper_body == HELPER_BODY_EXPECTED


def declared_once(b: str) -> bool:
    """`exposed_devices` is declared exactly once in the body: no shadowing redeclaration."""
    return bool(b) and len(ANY_DECL.findall(b)) == 1


def bound_is_helper(b: str) -> bool:
    """That declaration takes the helper's value, before the first admission check reads it."""
    if not b or b.count(BOUND_DECL) != 1:
        return False
    first_call = ADMISSIBLE_CALL.search(b)
    return first_call is not None and b.find(BOUND_DECL) < first_call.start()


def no_physical_count(b: str) -> bool:
    """The physical count is never named in the body."""
    return bool(b) and "total_gpu_count" not in b


def every_call_passes_bound(b: str, min_calls: int) -> bool:
    """Every admission check passes exactly the bound (a call that bypasses the helper fails), and none was dropped."""
    counts = admissible_counts(b)
    return len(counts) >= min_calls and all(c == "exposed_devices" for c in counts)


HELPER_BODY = body(CACHE, HELPER_SIG)
PLAN_BODY = body(CACHE, PLAN_SIG)
VALIDATE_BODY = body(CACHE, VALIDATE_SIG)
BODIES = ((PLAN_BODY, 1), (VALIDATE_BODY, 2))


def test_bodies_are_found():
    assert HELPER_BODY, "moe_mmid_exposed_owner_count() not found: the signature this gate keys on moved"
    assert PLAN_BODY, "plan_moe_mmid_workspaces() not found: the signature this gate keys on moved"
    assert VALIDATE_BODY, "validate_moe_mmid_execution_owners() not found: the signature this gate keys on moved"


def test_helper_is_the_registry_count():
    assert helper_returns_registry_count(HELPER_BODY), (
        "moe_mmid_exposed_owner_count() must return ggml_sycl_info().device_count (the registry's exposed count), "
        f"got {HELPER_BODY!r}")


def test_planner_bounds_owners_by_the_registry_count():
    assert declared_once(PLAN_BODY), "plan_moe_mmid_workspaces() declares exposed_devices more than once (shadowing)"
    assert bound_is_helper(PLAN_BODY), (
        "plan_moe_mmid_workspaces() must take exposed_devices from moe_mmid_exposed_owner_count() before its first "
        "admission check")
    assert no_physical_count(PLAN_BODY), "plan_moe_mmid_workspaces() must never name total_gpu_count"
    assert every_call_passes_bound(PLAN_BODY, 1), (
        "every moe_mmid_owner_admissible() call in plan_moe_mmid_workspaces() must pass exposed_devices")


def test_validator_bounds_owners_by_the_registry_count():
    assert declared_once(VALIDATE_BODY), (
        "validate_moe_mmid_execution_owners() declares exposed_devices more than once (shadowing)")
    assert bound_is_helper(VALIDATE_BODY), (
        "validate_moe_mmid_execution_owners() must take exposed_devices from moe_mmid_exposed_owner_count() before "
        "its first admission check")
    assert no_physical_count(VALIDATE_BODY), "validate_moe_mmid_execution_owners() must never name total_gpu_count"
    assert every_call_passes_bound(VALIDATE_BODY, 2), (
        "both moe_mmid_owner_admissible() calls in validate_moe_mmid_execution_owners() must pass exposed_devices")


# ---- mutants: each must turn its own claim red, and only that claim where the edit allows -------------------------


def _mutate(b: str, old: str, new: str) -> str:
    assert old in b, f"mutant anchor missing, so the mutant would be vacuous: {old!r}"
    return b.replace(old, new, 1)


PLAN_CALL = "moe_mmid_owner_admissible(owner, exposed_devices)"
VALIDATE_CALL = "moe_mmid_owner_admissible(target, exposed_devices)"
PLAN_CHECK = "if (!" + PLAN_CALL + ") { continue; }"
VALIDATE_CHECK = "if (!" + VALIDATE_CALL + ") { continue; }"


def test_mutant_helper_returns_physical_count():
    m = _mutate(HELPER_BODY, "ggml_sycl_info().device_count", "ggml_sycl_info().total_gpu_count")
    assert not helper_returns_registry_count(m)


def test_mutant_shadowing_redeclaration():
    for b, n in BODIES:
        m = _mutate(b, BOUND_DECL, BOUND_DECL + " const int exposed_devices = 2;")
        assert not declared_once(m)
        assert bound_is_helper(m) and no_physical_count(m) and every_call_passes_bound(m, n)


def test_mutant_bound_not_from_helper():
    for b, n in BODIES:
        m = _mutate(b, BOUND_DECL, "const int exposed_devices = 2;")
        assert not bound_is_helper(m)
        assert declared_once(m) and no_physical_count(m) and every_call_passes_bound(m, n)


def test_mutant_physical_count_declared():
    for b, _ in BODIES:
        m = _mutate(b, BOUND_DECL, "const int exposed_devices = ggml_sycl_info().total_gpu_count;")
        assert not bound_is_helper(m)
        assert not no_physical_count(m)


def test_mutant_physical_count_named_anywhere():
    for b, n in BODIES:
        m = _mutate(b, BOUND_DECL, BOUND_DECL + " (void) ggml_sycl_info().total_gpu_count;")
        assert not no_physical_count(m)
        assert declared_once(m) and bound_is_helper(m) and every_call_passes_bound(m, n)


def test_mutant_call_bypasses_the_bound():
    # A local that is not the bound, and the registry count read directly past the helper: neither names
    # total_gpu_count, so only the per-call check can catch them.
    for b, n, call, owner in ((PLAN_BODY, 1, PLAN_CALL, "owner"), (VALIDATE_BODY, 2, VALIDATE_CALL, "target")):
        for bypass in ("total_gpus", "ggml_sycl_info().device_count"):
            m = _mutate(b, call, f"moe_mmid_owner_admissible({owner}, {bypass})")
            assert not every_call_passes_bound(m, n)
            assert declared_once(m) and bound_is_helper(m) and no_physical_count(m)


def test_mutant_admission_check_dropped():
    for b, n, check in ((PLAN_BODY, 1, PLAN_CHECK), (VALIDATE_BODY, 2, VALIDATE_CHECK)):
        m = _mutate(b, check, "")
        assert not every_call_passes_bound(m, n)


def test_mutant_bound_declared_after_first_use():
    # Declared once, but past the admission check that reads it.
    m = _mutate(PLAN_BODY, BOUND_DECL + " ", "")
    m = _mutate(m, PLAN_CHECK, PLAN_CHECK + " " + BOUND_DECL)
    assert not bound_is_helper(m)
    assert declared_once(m) and no_physical_count(m) and every_call_passes_bound(m, 1)

if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
