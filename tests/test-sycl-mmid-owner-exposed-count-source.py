"""Source contract for llama.cpp-n8da: the MMID owner bound is the registry's exposed count, never the physical count.

Host-only: reads sources, builds nothing, loads no model and touches no device.

moe_mmid_owner_admissible(owner, exposed_device_count) admits an owner only below the count it is given, and
test-moe-mmid-workspace-plan pins that predicate. What that host case cannot see is the wiring: which count the two
planner call sites pass. The right one is ggml_sycl_info().device_count, the count ggml_backend_sycl_reg() sizes its
device list from and the bound ggml_backend_sycl_init() refuses at. The wrong one is ggml_sycl_info().total_gpu_count,
the physical count the multi-GPU planner budgets: passing it would plan an MMID workspace on a hidden GPU that no
backend context can bind, and materialization is all-or-nothing, so the primary's pool would be withheld too.

This gate pins, in plan_moe_mmid_workspaces() and validate_moe_mmid_execution_owners():
  - the bound is declared once as `const int exposed_devices = ggml_sycl_info().device_count;`, before the first
    admission check;
  - every moe_mmid_owner_admissible() call passes exactly `exposed_devices` as its count (at least one call in the
    planner, at least two in the validator: the on-device owner and the alternate owner);
  - the body never names total_gpu_count.

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

BOUND_DECL = "const int exposed_devices = ggml_sycl_info().device_count;"
ADMISSIBLE_CALL = re.compile(r"moe_mmid_owner_admissible\(")


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


def claim_bound_is_the_registry_count(b: str, min_calls: int) -> bool:
    """The body declares the registry bound once, before its first admission check; every admission check passes
    exactly that bound; and the physical count is never named."""
    if not b or b.count(BOUND_DECL) != 1 or "total_gpu_count" in b:
        return False
    first_call = ADMISSIBLE_CALL.search(b)
    if first_call is None or b.find(BOUND_DECL) > first_call.start():
        return False
    counts = admissible_counts(b)
    return len(counts) >= min_calls and all(c == "exposed_devices" for c in counts)


PLAN_BODY = body(CACHE, PLAN_SIG)
VALIDATE_BODY = body(CACHE, VALIDATE_SIG)


def test_bodies_are_found():
    assert PLAN_BODY, "plan_moe_mmid_workspaces() not found: the signature this gate keys on moved"
    assert VALIDATE_BODY, "validate_moe_mmid_execution_owners() not found: the signature this gate keys on moved"


def test_planner_bounds_owners_by_the_registry_count():
    assert claim_bound_is_the_registry_count(PLAN_BODY, 1), (
        "plan_moe_mmid_workspaces() must bound MMID owners by ggml_sycl_info().device_count (the registry's exposed "
        "count) in every moe_mmid_owner_admissible() call, never by total_gpu_count")


def test_validator_bounds_owners_by_the_registry_count():
    assert claim_bound_is_the_registry_count(VALIDATE_BODY, 2), (
        "validate_moe_mmid_execution_owners() must bound both owner checks by ggml_sycl_info().device_count (the "
        "registry's exposed count), never by total_gpu_count")


# ---- mutants: each must turn the claim red --------------------------------------------------------------------------


def _mutate(b: str, old: str, new: str) -> str:
    assert old in b, f"mutant anchor missing, so the mutant would be vacuous: {old!r}"
    return b.replace(old, new, 1)


def test_mutant_physical_count_declared():
    for b, n in ((PLAN_BODY, 1), (VALIDATE_BODY, 2)):
        m = _mutate(b, BOUND_DECL, "const int exposed_devices = ggml_sycl_info().total_gpu_count;")
        assert not claim_bound_is_the_registry_count(m, n)


def test_mutant_physical_count_passed_at_a_call():
    m = _mutate(PLAN_BODY, "moe_mmid_owner_admissible(owner, exposed_devices)",
                "moe_mmid_owner_admissible(owner, ggml_sycl_info().total_gpu_count)")
    assert not claim_bound_is_the_registry_count(m, 1)
    m = _mutate(VALIDATE_BODY, "moe_mmid_owner_admissible(target, exposed_devices)",
                "moe_mmid_owner_admissible(target, total_gpus)")
    assert not claim_bound_is_the_registry_count(m, 2)


def test_mutant_admission_check_dropped():
    m = _mutate(PLAN_BODY, "if (!moe_mmid_owner_admissible(owner, exposed_devices)) { continue; }", "")
    assert not claim_bound_is_the_registry_count(m, 1)
    m = _mutate(VALIDATE_BODY, "if (!moe_mmid_owner_admissible(target, exposed_devices)) { continue; }", "")
    assert not claim_bound_is_the_registry_count(m, 2)


def test_mutant_bound_declared_after_first_use():
    # Declared once, but past the admission check that reads it.
    m = _mutate(PLAN_BODY, BOUND_DECL + " ", "")
    m = _mutate(m, "if (!moe_mmid_owner_admissible(owner, exposed_devices)) { continue; }",
                "if (!moe_mmid_owner_admissible(owner, exposed_devices)) { continue; } " + BOUND_DECL)
    assert not claim_bound_is_the_registry_count(m, 1)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
