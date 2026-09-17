#!/usr/bin/env python3
"""A MoE host-expert dispatch must bridge a storage-handle miss, not kill the graph (llama.cpp-srti).

THE CONTRACT.  The MoE host-expert dispatch paths in ggml-sycl.cpp need an
activation's (src1) and output's (dst) bytes at a copy boundary, which means they
need a ggml_sycl::mem_handle -- ggml_sycl::mem_copy_async takes handles on both
sides and there is no raw-pointer overload.  They obtain one from
ggml_sycl_find_tensor_storage_handle(), which succeeds only when the tensor
carries a populated ggml_tensor_extra_gpu::data_handle[device] OR sits inside a
unified-cache "managed" buffer allocation.

Neither is guaranteed for a graph INTERMEDIATE.  The backend already knows this
and says so at the PP island site: "Intermediate activations can live in compute
buffers that are not unified-cache-managed on this device (e.g. GPT-OSS on a card
where experts spill to host).  Recover through the documented raw-pointer ABI
bridge instead of aborting."  That bridge is make_data_ptr_handle(), which wraps
an already-live tensor->data through mem_handle::from_chunk_ptr() -- reacquiring a
unified-cache arena lease when the pointer belongs to one and degrading to DIRECT
otherwise, so it stays inside the two sanctioned ownership surfaces and allocates
nothing.

Before this ticket exactly ONE of the seven sites in that family had the bridge.
The other six answered a miss with a fatal refusal: five threw
ggml_sycl_fallback_error, which no handler retries (it is deliberately rethrown
past the resource-exhaustion ladder and lands on the graph-compute boundary as
GGML_STATUS_FAILED), and one called GGML_ABORT outright.  A missing handle for one
activation therefore killed an entire llama_decode -- observed on
Qwen3.6-35B-A3B-UD-Q5_K_S on the B50, where 16782 of 30720 experts are placed
host-side: "[MOE-ROUTE] CPU dispatch missing smart src1 handle
tensor=ffn_moe_swiglu-19 device=0; refusing route" then "llama_decode: failed to
decode, ret = -3".  Layer 19 is simply the first layer with host-placed experts,
so it is the first to reach this code at all.

WHAT IT CHECKS.  In ggml-sycl.cpp (comments blanked, offsets preserved):
  * ggml_sycl_ensure_tensor_storage_handle is defined exactly once, and its body
    calls BOTH ggml_sycl_find_tensor_storage_handle( (the managed fast path is
    tried first and is unchanged) AND make_data_ptr_handle( (the bridge);
  * that body sets view_offset to 0 on the bridge path.  This is load-bearing and
    silent if wrong: make_data_ptr_handle wraps tensor->data, which ALREADY points
    at a view's own first byte, so carrying a non-zero view_offset forward would
    add the offset twice and copy the wrong rows -- wrong numbers, not a crash;
  * every fatal storage refusal in the MoE host-expert dispatch family is guarded
    by that helper and NOT by a direct ggml_sycl_find_tensor_storage_handle call.
    This is the actual defect, so it is checked per site by message.

NOT VACUOUS.  `--self-test` runs the gate on in-memory mutants -- the helper's
bridge removed, its find call removed, its view_offset reset removed, the helper
defined twice, and one mutant per guarded site reverted to a direct lookup -- and
requires every one to FAIL, then requires the unmodified tree to pass.

THE ORIGINAL RED, predicted before the first run:
  FAIL: the MoE host-expert dispatch site for <message> still guards its fatal
  storage refusal with a direct ggml_sycl_find_tensor_storage_handle() call, so a
  graph intermediate whose compute buffer is not unified-cache-managed kills the
  whole graph instead of recovering through the raw-pointer ABI bridge
  (llama.cpp-srti)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"

HELPER = "ggml_sycl_ensure_tensor_storage_handle"
FIND = "ggml_sycl_find_tensor_storage_handle("
BRIDGE = "make_data_ptr_handle("
VIEW_OFFSET_RESET = "view_offset = 0"

# Every fatal storage refusal in the MoE host-expert dispatch family, by the
# literal text the code emits. Each must be guarded by HELPER, never by a direct
# FIND. Keyed by message because line numbers in this 106k-line file drift
# constantly; a message that stops appearing is caught by MISSING_SITE below.
GUARDED_SITES = [
    "[CPU-HOST-MAT] missing smart handle for src1/dst",
    "MUL_MAT_ID shared activation missing smart src1 handle",
    "MUL_MAT_ID CPU dispatch missing smart src1 handle",
    "MUL_MAT_ID CPU dispatch missing smart dst handle",
    "MUL_MAT_ID planner CPU dispatch missing smart src1 handle",
    "MUL_MAT_ID planner CPU dispatch missing smart dst handle",
    "MUL_MAT_ID PP CPU island missing smart handle",
]

# How far back from a refusal message to look for the guard that produced it.
LOOKBACK_LINES = 18


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


def helper_definition(blanked: str) -> str:
    """Body of the shared ensure-storage helper, and proof there is exactly one."""
    defs = [
        m.start()
        for m in re.finditer(r"(?:static\s+)?bool\s+" + re.escape(HELPER) + r"\s*\(", blanked)
    ]
    if not defs:
        raise ContractError(
            f"FAIL: no shared {HELPER}() is defined, so each MoE dispatch site still resolves "
            "storage on its own and a miss stays fatal (llama.cpp-srti)"
        )
    if len(defs) > 1:
        raise ContractError(
            f"FAIL: {HELPER}() is defined {len(defs)} times; the point of the helper is that the "
            "bridge exists in exactly one place and cannot diverge between sites (llama.cpp-srti)"
        )
    return brace_block_from(blanked, defs[0])


def check(backend_src: str) -> None:
    blanked = blank_comments(backend_src)

    body = helper_definition(blanked)
    if FIND not in body:
        raise ContractError(
            f"FAIL: {HELPER}() never calls {FIND}, so it bypasses the managed fast path instead of "
            "falling back from it (llama.cpp-srti)"
        )
    if BRIDGE not in body:
        raise ContractError(
            f"FAIL: {HELPER}() never calls {BRIDGE}, so it has no raw-pointer ABI bridge and a "
            "storage-handle miss is still unrecoverable (llama.cpp-srti)"
        )
    if VIEW_OFFSET_RESET not in body:
        raise ContractError(
            f"FAIL: {HELPER}() does not reset {VIEW_OFFSET_RESET!r} on the bridge path. "
            "make_data_ptr_handle wraps tensor->data, which already points at a view's first byte, "
            "so carrying the offset forward would add it twice and copy the wrong rows -- wrong "
            "numbers, not a crash (llama.cpp-srti)"
        )

    lines = blanked.splitlines()
    # The helper's own body legitimately contains a direct FIND call; exclude its
    # line range so the per-site scan below cannot credit or blame it.
    helper_start = blanked[: blanked.find(body)].count("\n")
    helper_end = helper_start + body.count("\n")

    for message in GUARDED_SITES:
        hits = [i for i, line in enumerate(lines) if message in line]
        if not hits:
            raise ContractError(
                f"FAIL: the refusal message {message!r} no longer appears in {BACKEND.name}. This "
                "gate is anchored on it, so it would pass vacuously -- re-point the gate at the "
                "renamed site rather than deleting this entry (llama.cpp-srti)"
            )
        for at in hits:
            lo = max(0, at - LOOKBACK_LINES)
            window = lines[lo:at]
            window = [
                ln for k, ln in enumerate(window) if not (helper_start <= lo + k <= helper_end)
            ]
            text = "\n".join(window)
            if FIND in text:
                raise ContractError(
                    f"FAIL: the MoE host-expert dispatch site for {message!r} still guards its fatal "
                    "storage refusal with a direct ggml_sycl_find_tensor_storage_handle() call, so a "
                    "graph intermediate whose compute buffer is not unified-cache-managed kills the "
                    "whole graph instead of recovering through the raw-pointer ABI bridge "
                    "(llama.cpp-srti)"
                )
            if HELPER not in text:
                raise ContractError(
                    f"FAIL: the MoE host-expert dispatch site for {message!r} does not reach its "
                    f"refusal through {HELPER}(), so whatever it does resolve storage with is not the "
                    "shared bridge (llama.cpp-srti)"
                )


def self_test(backend_src: str) -> int:
    failures = []

    def expect_fail(label: str, mutated: str) -> None:
        try:
            check(mutated)
        except ContractError as e:
            print(f"  self-test: mutant '{label}' correctly FAILED: {str(e)[:120]}...")
            return
        failures.append(label)
        print(f"  self-test: mutant '{label}' was NOT caught")

    blanked = blank_comments(backend_src)
    body = helper_definition(blanked)
    # Mutate the real (un-blanked) text of the helper body so the mutants are
    # genuine source, not a blanked derivative.
    real_body = backend_src[backend_src.find(body[:60]) :][: len(body)] if body[:60] in backend_src else None
    if real_body is None:
        print("  self-test: cannot locate the helper body in unblanked source")
        return 1

    # mutant 1: bridge removed from the helper
    expect_fail("bridge-removed", backend_src.replace(real_body, real_body.replace(BRIDGE, "no_bridge_here(", 1)))
    # mutant 2: managed fast path removed from the helper
    expect_fail("find-removed", backend_src.replace(real_body, real_body.replace(FIND, "no_find_here(", 1)))
    # mutant 3: view_offset reset removed
    expect_fail(
        "view-offset-reset-removed",
        backend_src.replace(real_body, real_body.replace(VIEW_OFFSET_RESET, "view_offset = keep_it", 1)),
    )
    # mutant 4: a second definition of the helper
    dup = f"\nstatic bool {HELPER}(const ggml_tensor *, int, ggml_sycl_tensor_storage_handle *, const char *, const char *) {{ return false; }}\n"
    expect_fail("helper-defined-twice", backend_src + dup)

    # mutant 5..N: one per guarded site, reverted to a direct lookup
    for message in GUARDED_SITES:
        lines = backend_src.splitlines(keepends=True)
        idx = next((i for i, ln in enumerate(lines) if message in ln), None)
        if idx is None:
            failures.append(f"revert:{message}")
            print(f"  self-test: cannot build revert mutant for {message!r}")
            continue
        # Replace the helper call that guards this site with a direct FIND call.
        lo = max(0, idx - LOOKBACK_LINES)
        replaced = False
        for k in range(idx - 1, lo - 1, -1):
            if HELPER + "(" in lines[k]:
                lines[k] = lines[k].replace(HELPER + "(", FIND[:-1] + "(")
                replaced = True
                break
        if not replaced:
            failures.append(f"revert:{message}")
            print(f"  self-test: no helper call found to revert for {message!r}")
            continue
        expect_fail(f"revert:{message[:38]}", "".join(lines))

    try:
        check(backend_src)
        print("  self-test: unmodified tree passes")
    except ContractError as e:
        print(f"  self-test: unmodified tree FAILS: {e}")
        failures.append("unmodified")

    if failures:
        print(f"SELF-TEST FAIL: {', '.join(failures)}")
        return 1
    print(f"SELF-TEST PASS: {4 + len(GUARDED_SITES)} mutants caught, unmodified tree passes")
    return 0


def main(argv: list[str]) -> int:
    backend_src = BACKEND.read_text()
    if "--self-test" in argv:
        return self_test(backend_src)
    try:
        check(backend_src)
    except ContractError as e:
        print(e)
        return 1
    print(
        "PASS: every MoE host-expert dispatch site resolves src1/dst storage through the shared "
        "raw-pointer ABI bridge, so a non-unified-cache-managed graph intermediate no longer kills "
        "the graph (llama.cpp-srti)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
