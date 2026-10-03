"""Source contract for llama.cpp-5efe C4 (part b): `stream_dma` refuses a file-mapped source, and four callers say so.

A device read of a mapped page is at best a slow driver-staged copy and at worst a device loss, and streaming host-tier
weights to the device is forbidden outright (design 5efe §5.3, P3). `unified_cache::stream_dma` used to copy a
`HOST_MMAP` source straight to the device. The runtime halves are the pure host test `test-sycl-getrows-refusal-terminal`
(the slice loop over a fake `stream_dma`, and the registry predicate) and, on the device, the lead-run refusal arm; this
gate pins what neither sees:

1. `dma_stream_result` has `refused_file_mapped`, and a refusal counts into `stream_dma_file_mapped_refusals`, read
   through one accessor;
2. `stream_dma` tests the registry's `is_file_mapped(src.ptr)` at entry, before it resolves defaults, takes staging
   buffers or looks at `copy_fn` (so custom-copy callers are covered), and returns the typed result without `ok`;
3. each of the four callers (`mmq.cpp`, `mmvq.cpp`, `dmmv.cpp`, `getrows.cpp`) routes `refused_file_mapped` explicitly,
   before its `mmap_direct_failed` branch, to the CPU executor (`ggml_sycl_cpu_fallback_graph`), and does not fall
   through to a streaming retry or the resource-exhaustion ladder;
4. get_rows' slice loop is the shared `get_rows_dma_retry`, whose loop breaks on a refusal before its retry test, and
   getrows.cpp's attempt reports `result.refused_file_mapped` to it; the header includes no SYCL header.

Plain file I/O, comment-stripped text.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"

_LEXEME_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"' r"|'(?:\\.|[^'\\\n])*'" r"|//[^\n]*" r"|/\*.*?\*/",
    flags=re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def balanced(text: str, open_idx: int) -> str:
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i]
    raise AssertionError("unbalanced braces")


def function_body(text: str, signature_re: str) -> str:
    hits = list(re.finditer(signature_re, text))
    assert len(hits) == 1, f"{signature_re!r} matched {len(hits)} times"
    return balanced(text, text.index("{", hits[0].end()))


def read(rel: str) -> str:
    return strip_comments((SYCL / rel).read_text())


STREAM_DMA = r"unified_cache::dma_stream_result\s+unified_cache::stream_dma\s*\("


# ---- 1. the typed result and the counter ----


def test_the_result_type_has_the_refusal_flag():
    hpp = read("unified-cache.hpp")
    hit = re.search(r"struct\s+dma_stream_result\s*\{", hpp)
    assert hit, "dma_stream_result is missing"
    body = balanced(hpp, hit.end() - 1)
    assert re.search(
        r"bool\s+refused_file_mapped\s*=\s*false\s*;", body
    ), "dma_stream_result has no refused_file_mapped"


def test_the_refusal_counter_has_one_accessor():
    hpp, cpp = read("unified-cache.hpp"), read("unified-cache.cpp")
    assert re.search(
        r"uint64_t\s+unified_cache_stream_dma_file_mapped_refusals\s*\(\s*\)", hpp
    ), "no accessor declared for stream_dma_file_mapped_refusals"
    defs = re.findall(r"uint64_t\s+unified_cache_stream_dma_file_mapped_refusals\s*\(\s*\)\s*(?:noexcept\s*)?\{", cpp)
    assert len(defs) == 1, f"{len(defs)} definitions of the accessor"
    adds = re.findall(r"g_stream_dma_file_mapped_refusals\s*\.\s*fetch_add\s*\(", cpp)
    assert len(adds) == 1, f"{len(adds)} increments of the counter, expected 1 (in stream_dma's refusal)"


# ---- 2. the entry check ----


def test_stream_dma_refuses_a_file_mapped_source_at_entry():
    body = function_body(read("unified-cache.cpp"), STREAM_DMA)
    refuse = re.search(
        r"alloc_registry\s*::\s*instance\s*\(\s*\)\s*\.\s*is_file_mapped\s*\(\s*src\s*\.\s*ptr\s*\)", body
    )
    assert refuse, "stream_dma does not ask the registry whether src.ptr is a file mapping"
    for later in (
        r"resolve_dma_defaults\s*\(",
        r"get_dma_staging_buffers\s*\(",
        r"\bcopy_fn\b",
        r"cache_location::HOST_MMAP",
    ):
        hit = re.search(later, body)
        assert hit and refuse.start() < hit.start(), f"the refusal is not before {later!r}"
    # the `if (...) { ... }` whose condition is the registry test: its own block, not whatever follows it
    block = balanced(body, body.index("{", refuse.end()))
    assert re.search(
        r"result\s*\.\s*refused_file_mapped\s*=\s*true\s*;", block
    ), "the refusal does not set the typed flag"
    assert re.search(r"g_stream_dma_file_mapped_refusals\s*\.\s*fetch_add\s*\(", block), "the refusal is not counted"
    assert re.search(r"return\s+result\s*;", block), "the refusal does not return the typed result"
    assert not re.search(r"result\s*\.\s*ok\s*=\s*true", block), "a refusal must not report ok"


# ---- 3. the four callers ----

CALLERS = (
    ("mmq.cpp", "MMQ"),
    ("mmvq.cpp", "MMVQ"),
    ("dmmv.cpp", "DMMV"),
)


def failure_block(text: str, anchor: str) -> str:
    """The `if (!result.ok) { ... }` block that follows the stream_dma call at `anchor`."""
    at = text.index(anchor)
    hit = re.compile(r"if\s*\(\s*!\s*result\s*\.\s*ok\s*\)\s*\{").search(text, at)
    assert hit, "no `if (!result.ok)` after the stream_dma call"
    return balanced(text, hit.end() - 1)


def routes_the_refusal_first(block: str) -> bool:
    refused = re.search(r"result\s*\.\s*refused_file_mapped", block)
    mmap = re.search(r"result\s*\.\s*mmap_direct_failed", block)
    if not (refused and mmap and refused.start() < mmap.start()):
        return False
    between = block[refused.start() : mmap.start()]
    return bool(re.search(r"ggml_sycl_cpu_fallback_graph\s*\(", between))


def test_each_mmq_mmvq_dmmv_caller_routes_the_refusal_explicitly_first():
    for rel, name in CALLERS:
        text = read(rel)
        calls = list(re.finditer(r"cache\s*->\s*stream_dma\s*\(", text))
        assert len(calls) == 1, f"{rel}: {len(calls)} stream_dma callers, expected 1"
        block = failure_block(text, "stream_dma(")
        assert routes_the_refusal_first(
            block
        ), f"{rel}: the !result.ok block does not route refused_file_mapped to the CPU fallback before mmap_direct_failed"


def test_the_planted_unrouted_caller_is_seen():
    planted = (
        'if (!result.ok) { if (result.mmap_direct_failed) { ggml_sycl_cpu_fallback_graph(ctx, dst, "x"); }'
        ' if (result.refused_file_mapped) { ggml_sycl_cpu_fallback_graph(ctx, dst, "y"); } }'
    )
    assert not routes_the_refusal_first(balanced(planted, planted.index("{")))


# ---- 4. get_rows' slice loop ----


def test_getrows_runs_the_shared_loop_and_reports_the_refusal_to_it():
    text = read("getrows.cpp")
    assert (
        '#include "getrows-dma-retry.hpp"' in (SYCL / "getrows.cpp").read_text()
    ), "getrows.cpp does not include the loop"
    calls = list(re.finditer(r"get_rows_dma_retry\s*\(", text))
    assert len(calls) == 1, f"{len(calls)} uses of get_rows_dma_retry, expected the one streaming loop"
    assert not re.search(
        r"for\s*\(\s*;\s*retry_count\s*<=\s*retry_limit", text
    ), "the old inline slice loop is still there"
    stream = list(re.finditer(r"cache\s*->\s*stream_dma\s*\(", text))
    assert len(stream) == 1, f"{len(stream)} stream_dma callers in getrows.cpp, expected 1"
    after = text[stream[0].end() :]
    assert re.search(
        r"refused_file_mapped\s*=\s*result\s*\.\s*refused_file_mapped", after[:900]
    ), "the attempt does not report result.refused_file_mapped to the loop"
    fallback = re.search(r"outcome\s*\.\s*refused_file_mapped", text)
    assert fallback, "getrows.cpp does not branch on the loop's refusal"
    tail = text[fallback.start() : fallback.start() + 900]
    assert re.search(r"ggml_sycl_cpu_fallback_graph\s*\(", tail), "a get_rows refusal is not routed to the CPU fallback"


def loop_header() -> str:
    return strip_comments((SYCL / "getrows-dma-retry.hpp").read_text())


def test_the_loop_breaks_on_a_refusal_before_its_retry_test():
    src = loop_header()
    body = function_body(src, r"get_rows_dma_retry\s*\([^)]*\)")
    refused = re.search(r"\.\s*refused_file_mapped", body)
    retry = re.search(r"min_slice_bytes|next_slice", body)
    assert refused and retry and refused.start() < retry.start(), "the refusal is not tested before the retry decision"
    cond = re.compile(r"if\s*\(\s*result\s*\.\s*refused_file_mapped\s*\)\s*\{").search(body)
    assert cond, "the loop has no `if (result.refused_file_mapped) {` of its own"
    block = balanced(body, cond.end() - 1)
    assert re.search(r"\bbreak\s*;", block), "a refusal does not break the loop"


def test_the_loop_header_includes_no_sycl_header():
    src = loop_header()
    assert not re.search(
        r"#\s*include\s*[<\"][^>\"]*(sycl|unified-cache|ggml-sycl)", src
    ), "getrows-dma-retry.hpp must stay host-linkable"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
