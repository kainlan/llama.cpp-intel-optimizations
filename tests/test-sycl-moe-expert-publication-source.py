#!/usr/bin/env python3
"""Source gate for llama.cpp-s36q: the MUL_MAT_ID dispatch publishes AoS expert handles
with the consumer known, and the publication decision lives in one predicate.

A MUL_MAT_ID src0 that is ne[2] == 1 and unclassified by name (test-backend-ops' issue
27873 case, n_mats = 1) is indistinguishable from a dense weight at upload, so nothing
publishes its expert handle then. The dispatch-time publish
(ggml_sycl_publish_mmid_canonical_aos_experts) is the one place the consumer is known; if it
stops saying so, the retained resolver finds no expert (NOT_FOUND) and the op fails with
"retained prompt admission failed". The behaviour of the predicate is pinned by
tests/test-sycl-moe-mmvq-tables.cpp (section 7); this gate pins the wiring that a host test of
a pure header cannot see.

Runs under pytest and as a plain script. Alternate copy via GGML_SYCL_S36Q_BACKEND_SOURCE.
"""

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = Path(os.environ.get("GGML_SYCL_S36Q_BACKEND_SOURCE", str(ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp")))


def body_of(text, signature_re):
    m = re.search(signature_re, text)
    assert m, f"could not find {signature_re}; the check would pass vacuously"
    i = text.index("{", m.end())
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j]
    raise AssertionError("unbalanced braces")


def test_publish_function_asks_the_predicate():
    body = body_of(BACKEND.read_text(encoding="utf-8"),
                   r"static bool ggml_sycl_publish_backend_aos_expert_handles\s*\(")
    assert "moe_aos_expert_publication_wanted(" in body, (
        "ggml_sycl_publish_backend_aos_expert_handles no longer asks moe_aos_expert_publication_wanted; "
        "a private copy of the decision can drift from the host-tested predicate")
    assert not re.search(r"tensor->ne\[2\]\s*>\s*1", body), (
        "ggml_sycl_publish_backend_aos_expert_handles keeps a private `ne[2] > 1` test")


def test_mmid_dispatch_publish_names_its_consumer():
    body = body_of(BACKEND.read_text(encoding="utf-8"),
                   r"static bool ggml_sycl_publish_mmid_canonical_aos_experts\s*\(")
    m = re.search(r"ggml_sycl_publish_backend_aos_expert_handles\s*\((.*?)\)\s*;", body, re.S)
    assert m, "the MUL_MAT_ID dispatch publish no longer calls ggml_sycl_publish_backend_aos_expert_handles"
    assert re.search(r"\btrue\b", m.group(1)), (
        "the MUL_MAT_ID dispatch publishes without saying its consumer is a MUL_MAT_ID, so a "
        "single-expert (ne[2] == 1) unclassified src0 gets no expert handles")


if __name__ == "__main__":
    import sys

    failures = 0
    for fn_name, fn in sorted(list(globals().items())):
        if fn_name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {fn_name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {fn_name}: {exc}")
    if failures:
        print(f"{failures} test(s) failed")
        sys.exit(1)
    print("all tests passed")
