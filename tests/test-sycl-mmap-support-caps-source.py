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
   (`if (sycl_model_backend && ml.use_mmap)`), narrowed by C4 to a load whose
   staged plan maps no files (`!ml.sycl_load_maps_files()`): a load that maps
   its files keeps the mapping, because its host-resident weights bind
   SYCL_Mapped in place and its device weights stage from it.

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

    # ggml_sycl_init() must hand the field to the query by reference: get_props itself
    # names `host_unified_known`, so a search over the whole file would match without it
    sycl = read_code(GGML_SYCL_CPP)
    init = function_body(sycl, r"static\s+ggml_sycl_device_info\s+ggml_sycl_init\s*\(\s*\)")
    assert re.search(
        r"\.host_unified_memory\s*=\s*ggml_sycl_device_query_host_unified\s*\([^;]*\.host_unified_known\s*\)\s*;", init
    ), (
        "ggml_sycl_init() does not assign host_unified_memory from "
        "ggml_sycl_device_query_host_unified(device, <info>.host_unified_known), so the field stays false "
        "for every device and mmap_support is false everywhere"
    )


def _balanced_block(text: str, open_idx: int) -> str:
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx:i + 1]
    return text[open_idx:]


def test_load_path_clears_use_mmap_only_for_a_sycl_load_that_maps_no_files():
    code = read_code(LLAMA_MODEL_CPP)
    m = re.search(r"if\s*\(\s*sycl_model_backend\s*&&\s*ml\.use_mmap\s*\)\s*\{", code)
    assert m, "`if (sycl_model_backend && ml.use_mmap)` is gone: a SYCL load without the tier must still clear use_mmap"
    block = _balanced_block(code, m.end() - 1)
    clear = re.search(r"ml\.use_mmap\s*=\s*false\s*;", block)
    assert clear, "that branch no longer clears ml.use_mmap"
    guard = re.search(r"if\s*\(\s*!\s*ml\.sycl_load_maps_files\s*\(\s*\)\s*\)\s*\{", block)
    assert guard and guard.start() < clear.start(), (
        "the clear is not narrowed to a load whose staged plan maps no files (`if (!ml.sycl_load_maps_files())`): "
        "a load that maps its files would lose the mapping its SYCL_Mapped buffers and staging read from"
    )


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
