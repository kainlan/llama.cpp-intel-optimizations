#!/usr/bin/env python3
"""Source gate for llama.cpp-dfo0 (leak probe, plan task L1): the COMPUTE-usage
early return in ggml_backend_sycl_buffer_reset (ggml/src/ggml-sycl/ggml-sycl.cpp)
preserves ctx->tensor_extras across the reset on the premise that init_tensor is
never called again for that buffer -- false whenever the graph is REBUILT
(llm_graph_result::reset re-inits the ggml context, so every activation tensor is
a fresh struct with extra == nullptr, and init_tensor mints a brand-new extra for
it; the previously "preserved" entries become unreachable). This gate checks only
that a diagnostic probe exists -- env-gated, WARN level, and identifiable by a
stable log tag -- so the leak becomes measurable before anything is changed. It
does not check the leak itself (Task L2's job) and does not touch a SYCL device.

Runs under pytest (llama_test_pytest registration) and as a plain script. Point it
at an alternate copy (to exercise the RED path against a deliberately unmodified
tree) via GGML_SYCL_DFO0_BACKEND_SOURCE; the default is the in-tree file.
"""

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = Path(os.environ.get("GGML_SYCL_DFO0_BACKEND_SOURCE", str(ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp")))

backend = BACKEND.read_text()

RESET_SIG = "static void ggml_backend_sycl_buffer_reset(ggml_backend_buffer_t buffer) {"
PROBE_ENV_VAR = "GGML_SYCL_EXTRA_LEAK_PROBE"
PROBE_TAG = "[EXTRA-LEAK-PROBE]"


def matching_brace(text, open_idx):
    """Comment/string-aware brace match (self-contained copy, per this fork's
    one-file-per-gate convention)."""
    assert text[open_idx] == "{"
    depth = 0
    state = "code"
    i = open_idx
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                state = "line"
                i += 2
                continue
            if ch == "/" and nxt == "*":
                state = "block"
                i += 2
                continue
            if ch == '"':
                state = "str"
            elif ch == "'":
                state = "chr"
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
        elif state == "line":
            if ch == "\n":
                state = "code"
        elif state == "block":
            if ch == "*" and nxt == "/":
                state = "code"
                i += 2
                continue
        elif state == "str":
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                state = "code"
        elif state == "chr":
            if ch == "\\":
                i += 2
                continue
            if ch == "'":
                state = "code"
        i += 1
    raise AssertionError("unbalanced braces")


def ws_pattern(needle):
    """Compile a regex matching `needle` where a line wrap is tolerated at ANY
    boundary (self-contained copy of test-sycl-q8-dense-layout-rule-source.py's
    helper, per this fork's one-file-per-gate convention): a reflowed signature
    (e.g. a future clang-format pass wrapping the parameter list) must not read
    as "missing definition"."""
    marked = re.sub(r"([(),*&])", r" \1 ", needle)
    raw_tokens = marked.split()
    assert raw_tokens, "empty needle"

    def is_word(tok):
        return re.fullmatch(r"\w+", tok) is not None

    parts = [re.escape(raw_tokens[0])]
    for i in range(1, len(raw_tokens)):
        sep = r"\s+" if is_word(raw_tokens[i - 1]) and is_word(raw_tokens[i]) else r"\s*"
        parts.append(sep)
        parts.append(re.escape(raw_tokens[i]))
    return re.compile("".join(parts))


def ws_find(text, needle, start=0):
    """First match position of `needle` in `text`, boundary-flexible (see
    ws_pattern). Returns -1 if not found; the returned position is a real
    offset into the ORIGINAL text."""
    m = ws_pattern(needle).search(text, start)
    return m.start() if m else -1


def ws_count(text, needle):
    """Number of boundary-flexible matches of `needle` in `text` (see
    ws_pattern) -- for a "defined exactly once" style check."""
    return len(list(ws_pattern(needle).finditer(text)))


def function_body(text, signature):
    idx = ws_find(text, signature)
    assert idx >= 0, f"missing definition: {signature}"
    open_idx = text.find("{", idx)
    return text[open_idx : matching_brace(text, open_idx) + 1]


def test_buffer_reset_defined_exactly_once():
    assert ws_count(backend, RESET_SIG) == 1, "ggml_backend_sycl_buffer_reset must be defined exactly once"


def test_probe_exists_and_is_env_gated():
    body = function_body(backend, RESET_SIG)
    assert PROBE_ENV_VAR in body, (
        f"ggml_backend_sycl_buffer_reset must gate the leak probe on getenv({PROBE_ENV_VAR!r})"
    )
    assert "std::getenv(" in body, "the probe must read the env var via std::getenv"


def test_probe_logs_at_warn_with_the_stable_tag():
    # WARN, not INFO: GGML_LOG_INFO is dropped at default verbosity in every
    # tool (CLAUDE.md, common/log.cpp:444) -- an INFO-level probe would never
    # reach a normal run's output.
    body = function_body(backend, RESET_SIG)
    assert "GGML_LOG_WARN(" in body, "the probe must log via GGML_LOG_WARN, not GGML_LOG_INFO/DEBUG"
    warn_idx = body.find("GGML_LOG_WARN(")
    call_end = body.find(";", warn_idx)
    assert call_end > warn_idx, "malformed GGML_LOG_WARN( call in buffer_reset"
    warn_call = body[warn_idx:call_end]
    assert PROBE_TAG in warn_call, f"the probe's GGML_LOG_WARN call must include the stable tag {PROBE_TAG}"
    # Named sum_of_sizes, not cumulative: it is a running sum of the per-call
    # vector *sizes* (a triangular series), not a byte-accurate leak count --
    # see the lead's spec-review finding on llama.cpp-i0oh (c-cqpq, F1). The
    # per-call `preserving %zu` field is the real leak proxy.
    assert "sum_of_sizes=" in warn_call, "the probe must report a running sum_of_sizes, not just this call's count"


def test_probe_is_zero_cost_when_the_env_var_is_unset():
    # The acceptance criterion is "without the env var nothing is logged and no
    # code runs beyond one getenv cached in a function-local static": the
    # cached flag must gate the whole probe block, not just the log call.
    body = function_body(backend, RESET_SIG)
    # Whitespace-flexible: clang-format may pad the declaration to align with
    # a neighboring statement's variable name (e.g. `bool                  leak_probe`).
    flag_match = re.search(r"static\s+const\s+bool\s+leak_probe", body)
    assert flag_match, "the probe must cache its getenv result in a function-local static bool"
    flag_idx = flag_match.start()
    warn_idx = body.find("GGML_LOG_WARN(")
    assert flag_idx < warn_idx, "the cached flag must be declared before the WARN call it gates"
    if_idx = body.find("if (leak_probe", flag_idx)
    assert 0 <= if_idx < warn_idx, "the WARN call must be inside an `if (leak_probe ...)` guard"


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
