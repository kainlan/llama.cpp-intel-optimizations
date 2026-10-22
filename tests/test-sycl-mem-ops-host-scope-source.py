"""Source contract for llama.cpp-36j0 Task 5: a pinned-pool WEIGHT-role host owner is a trusted USM copy endpoint.

Host-only: reads sources, builds nothing, loads no model and touches no device.

Why: the prefill expert streaming ring copies host-planned experts into VRAM. Those experts live in pinned-pool host
memory allocated with role WEIGHT, which never enters the pointer registry. Before this change
host_endpoint_owner_in_scope (ggml/src/ggml-sycl/mem-ops.cpp) trusted only runtime stages (role STAGING or
EXPERT_STAGING) and expert-path buffers (category EXPERT_CACHE), so a copy from a WEIGHT owner fell back to the
registry, read "unknown", and took mem_copy_submit's staged branch: a 64 MB pinned stage, a host memcpy and a wait on
the calling thread.

The change this gate pins:
  - the admission gains one disjunct, (owner.tier == alloc_tier::HOST_PINNED && owner.role == alloc_role::WEIGHT),
    so role WEIGHT is admitted only together with the HOST_PINNED tier, never on its own and never for HOST_MMAP or
    MMAP_TRACKED memory (an mmap source is pageable and must stay staged);
  - the early refusal (tier HOST_PINNED, a host zone and a device, or false) is unchanged, so a standalone pinned base
    and an unregistered pointer still keep the registry answer;
  - the graph-recording exclusion of llama.cpp-qwn5 in host_endpoint_is_usm_accessible is unchanged: a recorded copy
    keeps the endpoint's pointer for every replay, so it never asks the owner;
  - the comment above host_endpoint_owner_in_scope states the reason and names llama.cpp-36j0.

WHAT THIS DOES NOT PROVE. It reads source text with comments blanked. It does not show that a copy from such an owner
takes the direct path at run time (test-mem-ops' WEIGHT-owner case, lead-run on a device), nor that the expert handles
the ring copies from carry an owner at all.

NOT VACUOUS. Each claim has a mutant of the real source that must fail it.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEMOPS_REL = "ggml/src/ggml-sycl/mem-ops.cpp"
MEMOPS = (ROOT / MEMOPS_REL).read_text()

_LEXEME_RE = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'",
    re.DOTALL,
)

SCOPE_SIG = "static bool host_endpoint_owner_in_scope(const alloc_metadata & owner) {"
HELPER_SIG = "static bool host_endpoint_is_usm_accessible(const mem_handle & h, const void * ptr, sycl::queue & copy_queue) {"

WEIGHT_CLAUSE = "owner.tier == alloc_tier::HOST_PINNED && owner.role == alloc_role::WEIGHT"
SCOPE_REFUSAL = ("if (owner.tier != alloc_tier::HOST_PINNED || owner.host_zone == host_zone_id::COUNT || "
                 "owner.device < 0) { return false; }")
# Verbatim from master, including its line break and indentation.
RECORDING_EXCLUSION = (
    "    if (!ggml_sycl_graph_recording_active() && h.owner_metadata(&owner) && host_endpoint_owner_in_scope(owner) &&\n"
    "        queues_share_context(copy_queue, queue_for_device_or_fallback(owner.device, copy_queue))) {\n"
    "        return true;\n"
    "    }\n"
)
REASON_TAG = "llama.cpp-36j0"
# Every admission in host_endpoint_owner_in_scope, in order: the runtime stages, the expert path and the one WEIGHT
# conjunction. Nothing else may be admitted after the early refusal.
SCOPE_ADMITS = [
    "owner.role == alloc_role::STAGING",
    "owner.role == alloc_role::EXPERT_STAGING",
    "owner.category == runtime_category::EXPERT_CACHE",
    f"({WEIGHT_CLAUSE})",
]


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def flat(text: str) -> str:
    t = re.sub(r"\s+", " ", text)
    t = re.sub(r"\( ", "(", t)
    t = re.sub(r" \)", ")", t)
    return t.strip()


def body(src: str, signature: str) -> str | None:
    """The flattened, comment-blanked body of the function whose definition starts with `signature`."""
    code = flat(strip_comments(src))
    at = code.find(signature)
    if at < 0 or code.find(signature, at + 1) >= 0:
        return None
    start = at + len(signature)
    depth = 1
    for i in range(start, len(code)):
        if code[i] == "{":
            depth += 1
        elif code[i] == "}":
            depth -= 1
            if depth == 0:
                return code[start:i].strip()
    return None


def comment_above(src: str, signature: str) -> str:
    """The run of // comment lines immediately above `signature` in the raw source."""
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(signature.split("(")[0]):
            j = i - 1
            out = []
            while j >= 0 and lines[j].lstrip().startswith("//"):
                out.append(lines[j].lstrip()[2:].strip())
                j -= 1
            return " ".join(reversed(out))
    return ""


def admissions(src: str) -> list[str] | None:
    b = body(src, SCOPE_SIG)
    if b is None or not b.startswith(SCOPE_REFUSAL):
        return None
    rest = b[len(SCOPE_REFUSAL):].strip()
    if not rest.startswith("return ") or not rest.endswith(";") or rest.count(";") != 1:
        return None
    return [a.strip() for a in rest[len("return "):-1].split(" || ")]


def claim_weight_owner_admitted(src: str) -> bool:
    """One admission is exactly the HOST_PINNED-and-WEIGHT conjunction."""
    admits = admissions(src)
    return admits is not None and f"({WEIGHT_CLAUSE})" in admits


def claim_weight_never_admitted_alone(src: str) -> bool:
    """role WEIGHT appears once in the scope, inside the conjunction; no mmap tier is named there."""
    b = body(src, SCOPE_SIG)
    if b is None:
        return False
    if b.count("alloc_role::WEIGHT") != 1 or WEIGHT_CLAUSE not in b:
        return False
    return "HOST_MMAP" not in b and "MMAP_TRACKED" not in b


def claim_no_extra_admission(src: str) -> bool:
    """The admitted set is exactly SCOPE_ADMITS: no `|| true`, no further term."""
    return admissions(src) == SCOPE_ADMITS


def claim_refusal_unchanged(src: str) -> bool:
    b = body(src, SCOPE_SIG)
    return b is not None and b.startswith(SCOPE_REFUSAL)


def claim_recording_exclusion_unchanged(src: str) -> bool:
    """The helper's owner branch, comments blanked, is byte-identical to master's, recording guard first."""
    code = strip_comments(src)
    at = code.find(HELPER_SIG)
    return at >= 0 and code.count(RECORDING_EXCLUSION) == 1 and code.find(RECORDING_EXCLUSION) > at


def claim_comment_states_the_reason(src: str) -> bool:
    text = comment_above(src, SCOPE_SIG)
    return REASON_TAG in text and "WEIGHT" in text


def _once(old: str, new: str) -> str:
    assert MEMOPS.count(old) == 1, f"mutant anchor must match exactly once: {old!r} x{MEMOPS.count(old)}"
    return MEMOPS.replace(old, new, 1)


def test_weight_owner_admitted():
    assert claim_weight_owner_admitted(MEMOPS)


def test_weight_never_admitted_alone():
    assert claim_weight_never_admitted_alone(MEMOPS)


def test_no_extra_admission():
    assert claim_no_extra_admission(MEMOPS)


def test_refusal_unchanged():
    assert claim_refusal_unchanged(MEMOPS)


def test_recording_exclusion_unchanged():
    assert claim_recording_exclusion_unchanged(MEMOPS)


def test_comment_states_the_reason():
    assert claim_comment_states_the_reason(MEMOPS)


def test_mutant_clause_removed_fails():
    mutated = re.sub(r" \|\|\s*\(owner\.tier == alloc_tier::HOST_PINNED && owner\.role == alloc_role::WEIGHT\)", "",
                     MEMOPS, count=1)
    assert mutated != MEMOPS
    assert not claim_weight_owner_admitted(mutated)


def test_mutant_weight_admitted_alone_fails():
    mutated = _once(WEIGHT_CLAUSE, "owner.role == alloc_role::WEIGHT")
    assert not claim_weight_owner_admitted(mutated)
    assert not claim_weight_never_admitted_alone(mutated)


def test_mutant_mmap_tier_fails():
    mutated = _once(WEIGHT_CLAUSE, "owner.tier == alloc_tier::HOST_MMAP && owner.role == alloc_role::WEIGHT")
    assert not claim_weight_never_admitted_alone(mutated)


def test_mutant_admit_everything_fails():
    mutated = _once(f"({WEIGHT_CLAUSE});", f"({WEIGHT_CLAUSE}) || true;")
    assert not claim_no_extra_admission(mutated)


def test_mutant_extra_term_fails():
    mutated = _once(f"({WEIGHT_CLAUSE});", f"({WEIGHT_CLAUSE}) ||\n           owner.role == alloc_role::KV;")
    assert not claim_no_extra_admission(mutated)


def test_mutant_refusal_loosened_fails():
    mutated = _once(" || owner.host_zone == host_zone_id::COUNT", "")
    assert not claim_refusal_unchanged(mutated)


def test_mutant_recording_guard_dropped_fails():
    mutated = _once("if (!ggml_sycl_graph_recording_active() && h.owner_metadata(&owner)", "if (h.owner_metadata(&owner)")
    assert not claim_recording_exclusion_unchanged(mutated)


def test_mutant_recording_branch_commented_out_fails():
    mutated = _once(RECORDING_EXCLUSION, "/*\n" + RECORDING_EXCLUSION + "*/\n")
    assert claim_recording_exclusion_unchanged(MEMOPS)
    assert not claim_recording_exclusion_unchanged(mutated)


def test_mutant_reason_removed_fails():
    mutated = MEMOPS.replace(REASON_TAG, "llama.cpp-xxxx")
    assert not claim_comment_states_the_reason(mutated)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
