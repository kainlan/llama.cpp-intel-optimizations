#!/usr/bin/env python3
"""A CPU expert dispatch larger than the pinned pool must fall back, not abort (llama.cpp-sfal).

THE CONTRACT.  ggml_sycl::PinnedBufferPool is a fixed-size ring of pinned
staging buffers for host-resident MoE expert compute.  Its capacity is chosen
ONCE per device inside moe_hybrid_init_once() (ggml-sycl.cpp), from
max_dispatch_count = max over the scanned graph's MUL_MAT_ID nodes of
ids->ne[0] * ids->ne[1] -- that is, top-K times the token count OF WHATEVER
GRAPH HAPPENS TO BE COMPUTED FIRST.  That graph is llama's warmup run ("warming
up the model with an empty run"), which carries 2 tokens, so a top-8 model is
sized for 8 * 2 = 16 dispatch entries.

An expert_dispatch_entry is one (token, expert-slot) pair -- it carries both
iid1 (token index) and id (slot within that token) -- so a single MUL_MAT_ID
during real prompt processing produces up to top-K * n_ubatch entries, bounded
by the ubatch, NOT by the expert count.  On Qwen3.6-35B-A3B (n_expert=256,
n_expert_used=8, n_ubatch=512) the real bound is 4096, against a pool sized 16.

Before this ticket the dispatch site entered the pool path on
pool.is_initialized() alone and PinnedBufferPool::acquire() answered an
over-capacity request with GGML_ASSERT(n_experts <= max_experts_), aborting the
process (rc=134) at the first prompt decode.  The buffers really are too small
(pinned-buffer-pool.cpp sizes them max_experts * dim * sizeof(float)), so the
assert is a correct statement of the pool's own precondition; the defect is that
the CALLER never checked it, even though the very same block already has a
correctly-sized fallback (allocate_managed_host_pinned, used today for the
Q1_0/NVFP4 immutable-recipe path).

The fix makes the capacity a queryable precondition, PinnedBufferPool::can_serve(n),
and gates the pool branch on it, so an over-capacity dispatch takes the existing
managed host-pinned path instead of aborting.  acquire()'s assert stays, and
becomes genuinely unreachable from this caller.

WHAT IT CHECKS.  Within the body of the dispatch_cpu_compute lambda in
ggml-sycl.cpp (comments blanked, offsets preserved):
  * the pool branch's condition calls can_serve( and does NOT rest on
    is_initialized() alone;
  * acquire( appears only inside that guarded branch;
  * the managed host-pinned fallback (allocate_managed_host_pinned() for both
    the act and out buffers) survives, so the fallback the guard now routes to
    still exists.
And in pinned-buffer-pool.hpp:
  * can_serve( is declared, and acquire()'s capacity assert is still present in
    pinned-buffer-pool.cpp.

NOT VACUOUS.  `--self-test` runs the gate on five in-memory mutants (guard
reverted to is_initialized() alone; can_serve removed from the header; the act
fallback removed; the out fallback removed; the capacity assert removed) and
requires each to FAIL, then requires the unmodified tree to pass.

THE ORIGINAL RED, predicted before the first run and recorded in this ticket's
commit body:
  FAIL: the MUL_MAT_ID CPU dispatch enters the PinnedBufferPool path on
  pool.is_initialized() alone, so a dispatch larger than the pool's capacity
  aborts in acquire() instead of falling back to the managed host-pinned path
  (llama.cpp-sfal)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
POOL_HPP = ROOT / "ggml/src/ggml-sycl/pinned-buffer-pool.hpp"
POOL_CPP = ROOT / "ggml/src/ggml-sycl/pinned-buffer-pool.cpp"

LAMBDA_MARKER = "auto dispatch_cpu_compute = [&]("
CAN_SERVE = "can_serve("
IS_INIT = "is_initialized()"
ACQUIRE = ".acquire("
FALLBACK = "allocate_managed_host_pinned("
ACT_TAG = '"moe_cpu_dispatch_act"'
OUT_TAG = '"moe_cpu_dispatch_out"'
CAPACITY_ASSERT = "n_experts <= max_experts_"

RED_TEXT = (
    "FAIL: the MUL_MAT_ID CPU dispatch enters the PinnedBufferPool path on "
    "pool.is_initialized() alone, so a dispatch larger than the pool's capacity aborts in "
    "acquire() instead of falling back to the managed host-pinned path (llama.cpp-sfal)"
)


class ContractError(AssertionError):
    pass


def blank_comments(src: str) -> str:
    """Replace // and /* */ comment bodies with spaces, preserving offsets and newlines."""
    out = []
    i = 0
    n = len(src)
    while i < n:
        two = src[i : i + 2]
        if two == "//":
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif two == "/*":
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join("\n" if c == "\n" else " " for c in src[i:j]))
            i = j
        elif src[i] == '"':
            j = i + 1
            while j < n and src[j] != '"':
                if src[j] == "\\":
                    j += 1
                j += 1
            out.append(src[i : j + 1])
            i = j + 1
        else:
            out.append(src[i])
            i += 1
    return "".join(out)


def brace_block_from(src: str, start: int) -> str:
    """The `{ ... }` block whose opening brace is the first `{` at or after `start`."""
    open_at = src.find("{", start)
    if open_at < 0:
        raise ContractError("FAIL: expected a brace block and found none")
    depth = 0
    for k in range(open_at, len(src)):
        c = src[k]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[open_at : k + 1]
    raise ContractError("FAIL: brace block is unbalanced")


def lambda_body(src: str) -> str:
    """Body of the dispatch_cpu_compute lambda."""
    at = src.find(LAMBDA_MARKER)
    if at < 0:
        raise ContractError(f"FAIL: {LAMBDA_MARKER} not found in {BACKEND.name} (llama.cpp-sfal)")
    return brace_block_from(src, at)


def pool_branch_condition(body: str) -> tuple[str, int]:
    """The `if (...)` condition that guards the pool path, and the offset of that if."""
    m = re.search(r"if\s*\(([^{]*?)\)\s*\{", body[body.find("auto & pool") :])
    if not m:
        raise ContractError("FAIL: no if-condition follows the pool reference in dispatch_cpu_compute (llama.cpp-sfal)")
    base = body.find("auto & pool")
    return m.group(1), base + m.start()


def check(backend_src: str, hpp_src: str | None, cpp_src: str | None) -> None:
    # The dispatch site is checked FIRST so a pre-fix tree fails with the
    # original RED about the site's behaviour, not with a missing-declaration
    # message that says nothing about how the caller behaves.
    body = blank_comments(lambda_body(backend_src))

    condition, if_at = pool_branch_condition(body)
    if CAN_SERVE not in condition:
        raise ContractError(RED_TEXT)
    if IS_INIT in condition:
        raise ContractError(
            "FAIL: the pool branch condition still tests is_initialized(); can_serve() already implies it, "
            "and keeping both invites the capacity half being dropped again (llama.cpp-sfal)"
        )

    guarded = brace_block_from(body, if_at)
    if ACQUIRE not in guarded:
        raise ContractError("FAIL: the guarded pool branch no longer calls acquire() (llama.cpp-sfal)")
    outside = body.replace(guarded, "")
    if ACQUIRE in outside:
        raise ContractError(
            "FAIL: acquire() is called outside the can_serve()-guarded branch, so an over-capacity "
            "dispatch can still reach the pool's assert (llama.cpp-sfal)"
        )

    if FALLBACK not in body:
        raise ContractError(
            f"FAIL: dispatch_cpu_compute lost its {FALLBACK} fallback, which is the path the "
            "can_serve() guard routes an over-capacity dispatch to (llama.cpp-sfal)"
        )
    for tag in (ACT_TAG, OUT_TAG):
        if tag not in body:
            raise ContractError(
                f"FAIL: dispatch_cpu_compute no longer allocates the {tag} fallback buffer (llama.cpp-sfal)"
            )

    if hpp_src is None:
        raise ContractError(f"FAIL: {POOL_HPP} does not exist (llama.cpp-sfal)")
    if CAN_SERVE not in hpp_src:
        raise ContractError(f"FAIL: PinnedBufferPool does not declare {CAN_SERVE} (llama.cpp-sfal)")

    if cpp_src is None:
        raise ContractError(f"FAIL: {POOL_CPP} does not exist (llama.cpp-sfal)")
    if CAPACITY_ASSERT not in cpp_src:
        raise ContractError(
            "FAIL: PinnedBufferPool::acquire() lost its capacity assert; it is the pool's own "
            "precondition and must stay, so a future unguarded caller fails loudly (llama.cpp-sfal)"
        )


def self_test(backend_src: str, hpp_src: str, cpp_src: str) -> int:
    failures = []

    def expect_fail(label: str, backend: str, hpp: str | None = None, cpp: str | None = None) -> None:
        try:
            check(backend, hpp_src if hpp is None else hpp, cpp_src if cpp is None else cpp)
        except ContractError as e:
            print(f"  self-test: mutant '{label}' correctly FAILED: {e}")
            return
        failures.append(label)
        print(f"  self-test: mutant '{label}' was NOT caught")

    body = lambda_body(backend_src)

    # mutant 1: the guard reverted to is_initialized() alone -> the original RED
    reverted = body.replace("pool.can_serve(n_cpu)", "pool.is_initialized()", 1)
    if reverted == body:
        print("  self-test: cannot build guard-reverted mutant (guard text not found)")
        failures.append("guard-reverted")
    else:
        expect_fail("guard-reverted", backend_src.replace(body, reverted))

    # mutant 2: can_serve removed from the header
    expect_fail("can_serve-undeclared", backend_src, hpp_src.replace(CAN_SERVE, "capacity_ok("))

    # mutant 3 / 4: each fallback allocation removed
    expect_fail("act-fallback-removed", backend_src.replace(body, body.replace(ACT_TAG, '"mutant_act_removed"')))
    expect_fail("out-fallback-removed", backend_src.replace(body, body.replace(OUT_TAG, '"mutant_out_removed"')))

    # mutant 5: the pool's capacity assert removed
    expect_fail("capacity-assert-removed", backend_src, None, cpp_src.replace(CAPACITY_ASSERT, "true"))

    try:
        check(backend_src, hpp_src, cpp_src)
        print("  self-test: unmodified tree passes")
    except ContractError as e:
        print(f"  self-test: unmodified tree FAILS: {e}")
        failures.append("unmodified")

    if failures:
        print(f"SELF-TEST FAIL: {', '.join(failures)}")
        return 1
    print("SELF-TEST PASS: 5 mutants caught, unmodified tree passes")
    return 0


def main(argv: list[str]) -> int:
    backend_src = BACKEND.read_text()
    hpp_src = POOL_HPP.read_text() if POOL_HPP.exists() else None
    cpp_src = POOL_CPP.read_text() if POOL_CPP.exists() else None
    if "--self-test" in argv:
        if hpp_src is None or cpp_src is None:
            print("SELF-TEST FAIL: pinned-buffer-pool sources missing, nothing to mutate")
            return 1
        return self_test(backend_src, hpp_src, cpp_src)
    try:
        check(backend_src, hpp_src, cpp_src)
    except ContractError as e:
        print(e)
        return 1
    print(
        "PASS: an over-capacity MUL_MAT_ID CPU expert dispatch falls back to managed host-pinned "
        "buffers instead of aborting in PinnedBufferPool::acquire() (llama.cpp-sfal)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
