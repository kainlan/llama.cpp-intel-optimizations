#!/usr/bin/env python3
"""The host-weight registry must classify a name collision before it warns (llama.cpp-ttws).

THE CONTRACT.  ggml_backend_sycl_register_host_weight_tensor keys its registry
on owner + tensor NAME (ggml_sycl_host_weight_registry_key), not on the
ggml_tensor object.  llama-model-loader's TENSOR_DUPLICATED path therefore
registers two distinct ggml_tensor objects under one key whenever a model ties
its output head to token_embd (Gemma 3n / Gemma 4: src/models/gemma4.cpp
creates the DUPLICATED output alias BEFORE the real tok_embd, so
ggml_get_tensor() can never reuse it; llama-family archs land in different
buffer-type contexts and miss the same way).  Both objects describe the same
GGUF bytes -- same name, type, shape, nbytes -- and the existing reconcile code
already re-points the loser at the winner's extra and releases the registry's
hold on the displaced one.  That case is EXPECTED, and until this ticket it
printed "[SYCL] host weight registry mismatch for token_embd.weight" as a WARN
(twice: the emitter mirrors to stderr) on every Gemma 4 load.

The fix routes the collision through classify_host_weight_alias()
(ggml/src/ggml-sycl/host-weight-alias.hpp): ALIAS_SAME_BYTES is logged at
DEBUG and NOT warned; SAME_TENSOR-with-a-new-extra and DIVERGENT (same key,
different metadata -- a genuine conflict) keep the WARN.

WHAT IT CHECKS.  Within the body of ggml_backend_sycl_register_host_weight_tensor
(comments blanked, offsets preserved):
  * the header is included and the classifier is called;
  * there is an ALIAS_SAME_BYTES branch, and NO ggml_sycl_diag_emit_warn( call
    inside that branch's brace block;
  * the WARN format "host weight registry mismatch for %s" survives elsewhere in
    the body, so the genuine-conflict case still warns.

NOT VACUOUS.  `--self-test` runs the gate on three in-memory mutants (classifier
call removed; a warn inserted into the alias branch; the divergent WARN removed)
and requires each to FAIL, then requires the unmodified tree to pass.

THE ORIGINAL RED, predicted before the first run and recorded in this ticket's
commit body:
  FAIL: ggml_backend_sycl_register_host_weight_tensor does not consult
  classify_host_weight_alias() before warning on a registry name collision
  (llama.cpp-ttws)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
HEADER = ROOT / "ggml/src/ggml-sycl/host-weight-alias.hpp"

FUNCTION = "ggml_backend_sycl_register_host_weight_tensor"
INCLUDE = '#include "ggml-sycl/host-weight-alias.hpp"'
CLASSIFIER = "classify_host_weight_alias("
ALIAS_KIND = "host_weight_alias_kind::ALIAS_SAME_BYTES"
WARN_CALL = "ggml_sycl_diag_emit_warn("
WARN_FORMAT = "host weight registry mismatch for %s"

RED_TEXT = (
    f"FAIL: {FUNCTION} does not consult classify_host_weight_alias() before "
    "warning on a registry name collision (llama.cpp-ttws)"
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
            # keep string literals intact (the WARN format check needs them)
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


def function_body(src: str, name: str) -> str:
    """Body of the free function definition `void name(...) {` ... to its closing `}` at column 0."""
    m = re.search(r"(?m)^void\s+" + re.escape(name) + r"\s*\(", src)
    if not m:
        raise ContractError(f"FAIL: definition of {name} not found in {BACKEND}")
    end = re.search(r"(?m)^\}\s*$", src[m.end() :])
    if not end:
        raise ContractError(f"FAIL: unterminated definition of {name}")
    return src[m.start() : m.end() + end.end()]


def brace_block_after(src: str, start: int) -> str:
    """The `{ ... }` block whose opening brace is the first `{` at or after `start`."""
    open_at = src.find("{", start)
    if open_at < 0:
        raise ContractError("FAIL: alias branch has no brace block")
    depth = 0
    for k in range(open_at, len(src)):
        c = src[k]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[open_at : k + 1]
    raise ContractError("FAIL: alias branch brace block is unbalanced")


def check(backend_src: str, header_src: str | None) -> None:
    # The print site is checked FIRST so that the original RED (no classifier
    # at the site) is the message a pre-fix tree fails with, not a missing-file
    # message that says nothing about the site's behaviour.
    body = blank_comments(function_body(backend_src, FUNCTION))

    if CLASSIFIER not in body:
        raise ContractError(RED_TEXT)

    if header_src is None:
        raise ContractError(f"FAIL: {HEADER} does not exist (llama.cpp-ttws)")
    for kind in ("SAME_TENSOR", "ALIAS_SAME_BYTES", "DIVERGENT"):
        if kind not in header_src:
            raise ContractError(f"FAIL: host-weight-alias.hpp does not define kind {kind}")

    if INCLUDE not in backend_src:
        raise ContractError(f"FAIL: {BACKEND.name} does not include host-weight-alias.hpp (llama.cpp-ttws)")

    alias_at = body.find(ALIAS_KIND)
    if alias_at < 0:
        raise ContractError(f"FAIL: {FUNCTION} has no {ALIAS_KIND} branch (llama.cpp-ttws)")
    alias_block = brace_block_after(body, alias_at)
    if WARN_CALL in alias_block:
        raise ContractError(
            f"FAIL: {FUNCTION} still warns inside the ALIAS_SAME_BYTES branch; a tied-weight alias "
            "is expected and must not print as a mismatch (llama.cpp-ttws)"
        )

    outside = body.replace(alias_block, "")
    if WARN_FORMAT not in outside:
        raise ContractError(
            f'FAIL: {FUNCTION} lost the "{WARN_FORMAT}" WARN for the genuine-conflict case (llama.cpp-ttws)'
        )


def self_test(backend_src: str, header_src: str) -> int:
    failures = []

    def expect_fail(label: str, mutated_backend: str, mutated_header: str | None = header_src) -> None:
        try:
            check(mutated_backend, mutated_header)
        except ContractError as e:
            print(f"  self-test: mutant '{label}' correctly FAILED: {e}")
            return
        failures.append(label)
        print(f"  self-test: mutant '{label}' was NOT caught")

    # mutant 1: classifier call removed -> the original RED
    expect_fail("classifier-removed", backend_src.replace(CLASSIFIER, "classify_removed("))
    # mutant 2: a warn inserted at the top of the alias branch
    body = function_body(backend_src, FUNCTION)
    alias_at = blank_comments(body).find(ALIAS_KIND)
    if alias_at < 0:
        print("  self-test: cannot build alias-warn mutant (no alias branch)")
        failures.append("alias-warn-insert")
    else:
        open_at = body.find("{", alias_at)
        mutated_body = body[: open_at + 1] + ' ggml_sycl_diag_emit_warn("mutant");' + body[open_at + 1 :]
        expect_fail("alias-warn-insert", backend_src.replace(body, mutated_body))
    # mutant 3: the divergent WARN format removed
    expect_fail("divergent-warn-removed", backend_src.replace(WARN_FORMAT, "host weight registry collision"))
    # mutant 4: header missing
    expect_fail("header-missing", backend_src, None)

    try:
        check(backend_src, header_src)
        print("  self-test: unmodified tree passes")
    except ContractError as e:
        print(f"  self-test: unmodified tree FAILS: {e}")
        failures.append("unmodified")

    if failures:
        print(f"SELF-TEST FAIL: {', '.join(failures)}")
        return 1
    print("SELF-TEST PASS: 4 mutants caught, unmodified tree passes")
    return 0


def main(argv: list[str]) -> int:
    backend_src = BACKEND.read_text()
    header_src = HEADER.read_text() if HEADER.exists() else None
    if "--self-test" in argv:
        if header_src is None:
            print("SELF-TEST FAIL: header missing, nothing to mutate")
            return 1
        return self_test(backend_src, header_src)
    try:
        check(backend_src, header_src)
    except ContractError as e:
        print(e)
        return 1
    print(f"PASS: {FUNCTION} classifies registry name collisions; tied-weight aliases do not warn (llama.cpp-ttws)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
