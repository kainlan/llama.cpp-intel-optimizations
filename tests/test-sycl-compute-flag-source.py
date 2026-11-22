#!/usr/bin/env python3
"""The SYCL backend does not execute a node the graph did not ask for.

A node without GGML_TENSOR_FLAG_COMPUTE is an unselected branch of ggml_build_forward_select. The scheduler does not
upload that node's inputs, so a kernel run for it reads memory nobody wrote (on the B50 at default context: a
GET_ROWS on the token embedding with garbage ids, a GPU page fault and a lost device).

With comments blanked:
  P. (ggml-sycl.cpp ggml_sycl_is_noop) the predicate every node walker consults returns true for a node whose
     flags do not carry GGML_TENSOR_FLAG_COMPUTE: `(node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0`.
  S. (ggml-sycl.cpp ggml_sycl_graph_signature, the definition) the signature mixes node->flags, so a graph recorded
     for one selection is not replayed for another.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"

NOOP_SIG = "static inline bool ggml_sycl_is_noop(const ggml_tensor * node) {"
GRAPH_SIG = "static uint64_t ggml_sycl_graph_signature(const ggml_cgraph * cgraph) {"
FLAG_TEST = re.compile(r"\(\s*node->flags\s*&\s*GGML_TENSOR_FLAG_COMPUTE\s*\)\s*==\s*0")
FLAG_MIX = re.compile(r"\bmix\(\s*static_cast<uint64_t>\(\s*node->flags\s*\)\s*\)\s*;")


def blank_comments(text):
    text = re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), text)
    return re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)


def body(text, signature):
    """The brace-balanced body that starts at the end of `signature` (which ends with '{'), or '' if absent."""
    start = text.find(signature)
    if start < 0:
        return ""
    open_at = start + len(signature) - 1
    depth = 0
    for i in range(open_at, len(text)):
        depth += text[i] == "{"
        depth -= text[i] == "}"
        if depth == 0:
            return text[open_at:i + 1]
    return ""


def claims(sycl):
    noop = body(sycl, NOOP_SIG)
    sig = body(sycl, GRAPH_SIG)
    return {
        "P ggml_sycl_is_noop tests GGML_TENSOR_FLAG_COMPUTE": FLAG_TEST.search(noop) is not None,
        "S ggml_sycl_graph_signature mixes node->flags": FLAG_MIX.search(sig) is not None,
    }


def check(sycl):
    return [name for name, ok in claims(sycl).items() if not ok]


def mutate_in_body(text, signature, pattern, repl):
    """Apply `pattern` -> `repl` once inside the body of `signature` only."""
    start = text.find(signature)
    b = body(text, signature)
    if start < 0 or not b:
        return text
    open_at = start + len(signature) - 1
    mutated = pattern.sub(repl, b, count=1)
    return text[:open_at] + mutated + text[open_at + len(b):]


def main():
    sycl = blank_comments((SYCL / "ggml-sycl.cpp").read_text())
    failed = check(sycl)
    if failed:
        print("compute flag source contract failed: " + ", ".join(failed), file=sys.stderr)
        return 1
    if "--self-test" in sys.argv:
        mutants = {
            "flag test removed from ggml_sycl_is_noop": mutate_in_body(
                sycl, NOOP_SIG, re.compile(r"\(\s*node->flags\s*&\s*GGML_TENSOR_FLAG_COMPUTE\s*\)\s*==\s*0\s*\|\|"),
                ""),
            "flag test inverted in ggml_sycl_is_noop": mutate_in_body(
                sycl, NOOP_SIG, re.compile(r"(\(\s*node->flags\s*&\s*GGML_TENSOR_FLAG_COMPUTE\s*\)\s*)==(\s*0)"),
                r"\1!=\2"),
            "mix(node->flags) removed from ggml_sycl_graph_signature": mutate_in_body(sycl, GRAPH_SIG, FLAG_MIX, ""),
        }
        for name, text in mutants.items():
            if text == sycl:
                print(f"compute flag source contract: mutant '{name}' did not apply", file=sys.stderr)
                return 1
            if not check(text):
                print(f"compute flag source contract: mutant '{name}' was not caught", file=sys.stderr)
                return 1
            print(f"compute flag source contract: mutant '{name}' caught")
    print("compute flag source contract: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
