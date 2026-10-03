"""Source contract for llama.cpp-5efe C1: the SYCL device reports mmap support.

The n77l upstream merge added a fifth field, `mmap_support`, to
ggml_backend_dev_caps. The fork's rewritten ggml_backend_sycl_device_get_props
kept its four initializers, so the field was value-initialised to false and
every SYCL load under the default `-lm auto` dropped mmap and lazy mode
(src/llama-model.cpp, the caps loops in load_tensors).

The contract:

1. The initializer in ggml_backend_sycl_device_get_props names `.mmap_support`.
2. It is bound to `host_unified_known && !host_unified_memory`: neither a
   literal `true` (upstream's value, which would let the Arrow Lake iGPU, whose
   "VRAM" is system RAM, advertise mmap) nor absent (today's defect).
3. `host_unified_known` exists beside the cached `host_unified_memory` field,
   and the init loop assigns it from a query that reports whether it returned,
   so a throwing query leaves the device off rather than "discrete".
4. The load path still clears `use_mmap` for a SYCL model backend
   (`if (sycl_model_backend && ml.use_mmap)`): C1 deliberately does not remove
   it. C4 amends this clause.

Plain file I/O, not the codescout index: ggml-sycl.cpp is blind to it.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GGML_SYCL_CPP = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
COMMON_HPP = ROOT / "ggml/src/ggml-sycl/common.hpp"
COMMON_CPP = ROOT / "ggml/src/ggml-sycl/common.cpp"
LLAMA_MODEL_CPP = ROOT / "src/llama-model.cpp"

_LEXEME_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"'  # string literal (kept)
    r"|'(?:\\.|[^'\\\n])*'"  # char literal (kept)
    r"|//[^\n]*"  # line comment (dropped)
    r"|/\*.*?\*/",  # block comment (dropped)
    flags=re.DOTALL,
)


def strip_comments(src: str) -> str:
    """Remove comments, keep string literals, preserve line numbering."""

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def read_code(path: Path) -> str:
    return strip_comments(path.read_text())


def function_body(code: str, signature_re: str) -> str:
    """The brace-balanced body of the first function whose signature matches
    (braces inside the caps initializer's own `{ ... }` balance)."""
    m = re.search(signature_re, code)
    assert m, f"function not found: {signature_re}"
    open_brace = code.index("{", m.end())
    depth = 0
    for i in range(open_brace, len(code)):
        if code[i] == "{":
            depth += 1
        elif code[i] == "}":
            depth -= 1
            if depth == 0:
                return code[open_brace : i + 1]
    raise AssertionError(f"unbalanced braces after {signature_re}")


def caps_initializer() -> dict:
    """name -> expression for each `/* .name = */ expr,` entry of the caps
    initializer. Read from the RAW function text: the designated-initializer
    labels are comments, which strip_comments would remove."""
    raw = GGML_SYCL_CPP.read_text()
    body = function_body(raw, r"static void ggml_backend_sycl_device_get_props\s*\([^)]*\)")
    m = re.search(r"props\s*->\s*caps\s*=\s*\{(.*?)\};", body, flags=re.DOTALL)
    assert m, "ggml_backend_sycl_device_get_props has no `props->caps = { ... };` initializer"
    entries = {}
    for em in re.finditer(r"/\*\s*\.(\w+)\s*=\s*\*/\s*(.*?)\s*,", m.group(1) + ",", flags=re.DOTALL):
        entries[em.group(1)] = re.sub(r"\s+", " ", em.group(2))
    return entries


def test_get_props_names_mmap_support():
    entries = caps_initializer()
    assert "mmap_support" in entries, (
        "the caps initializer does not name .mmap_support, so it is value-initialised to false "
        "(the n77l merge-resolution slip, llama.cpp-5efe); it names: " + ", ".join(entries)
    )


def test_mmap_support_is_not_a_literal():
    expr = caps_initializer().get("mmap_support", "")
    assert expr, "could not read the .mmap_support expression"
    assert expr not in ("true", "false"), (
        f".mmap_support is the literal `{expr}`: it must follow the device, so the integrated GPU stays off"
    )


def test_mmap_support_is_known_and_discrete():
    expr = caps_initializer().get("mmap_support", "")
    assert re.search(r"\bhost_unified_known\b\s*&&\s*!\s*[\w.>\-\[\]]*\bhost_unified_memory\b", expr), (
        "`.mmap_support` must be `<info>.host_unified_known && !<info>.host_unified_memory`; got: " + expr
    )


def test_host_unified_known_sits_beside_the_cached_field():
    hpp = read_code(COMMON_HPP)
    assert re.search(r"\bbool\s+host_unified_known\b", hpp), "common.hpp has no `bool host_unified_known` field"
    assert re.search(r"\bbool\s+host_unified_memory\b", hpp), "common.hpp lost `bool host_unified_memory`"


def test_known_is_set_only_when_the_query_returned():
    cpp = read_code(COMMON_CPP)
    body = function_body(cpp, r"bool\s+ggml_sycl_device_query_host_unified\s*\([^)]*\)")
    assert re.search(r"get_info\s*<\s*sycl::info::device::host_unified_memory\s*>", body), (
        "the query does not read sycl::info::device::host_unified_memory"
    )
    try_part, _, catch_part = body.partition("catch")
    assert re.search(r"known\s*=\s*true", try_part), "`known` is never set true after the query returns"
    assert not re.search(r"known\s*=\s*true", catch_part), "`known` is set true on the throwing path"

    sycl = read_code(GGML_SYCL_CPP)
    assert re.search(r"\.host_unified_known\b", sycl), "ggml_sycl_info() never assigns .host_unified_known"


def test_load_path_still_clears_use_mmap_for_a_sycl_model_backend():
    code = read_code(LLAMA_MODEL_CPP)
    m = re.search(r"if\s*\(\s*sycl_model_backend\s*&&\s*ml\.use_mmap\s*\)\s*\{(.*?)\}", code, flags=re.DOTALL)
    assert m, "`if (sycl_model_backend && ml.use_mmap)` is gone: C1 must not remove it (C4 narrows it)"
    assert re.search(r"ml\.use_mmap\s*=\s*false\s*;", m.group(1)), "that branch no longer clears ml.use_mmap"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
