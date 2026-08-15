#!/usr/bin/env python3
"""SYCL buffer types must honour the base alignment they advertise to ggml-alloc.

ggml_backend_alloc_ctx_tensors_from_buft_impl sizes a context buffer as the sum
of per-tensor allocations padded to ggml_backend_buft_get_alignment(), and
ggml_tallocr_new then starts at aligned_offset(base, 0, alignment).  That head
offset is never added to the reservation, so a base misaligned by N bytes leaves
the buffer N bytes short and aborts later on an unrelated tensor
("not enough space in the buffer").  A SYCL buffer type therefore may not hand
back a base weaker than the alignment its get_alignment() promises.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
TLSF = ROOT / "ggml/src/ggml-sycl/tlsf-allocator.hpp"
MMVQ = ROOT / "ggml/src/ggml-sycl/mmvq.cpp"

CONSTANT = "GGML_SYCL_BUFFER_BASE_ALIGNMENT"
ALLOC_BUFFER = "static ggml_backend_buffer_t ggml_backend_sycl_buffer_type_alloc_buffer"
PUBLISH = "ggml_backend_sycl_buffer_publish"


def function(text: str, signature: str) -> str:
    start = text.index(signature)
    brace = text.index("{", start)
    depth = 0
    state = "code"
    i = brace
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                state = "line"
                i += 1
            elif ch == "/" and nxt == "*":
                state = "block"
                i += 1
            elif ch == '"':
                state = "string"
            elif ch == "'":
                state = "char"
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        elif state == "line":
            if ch == "\n":
                state = "code"
        elif state == "block":
            if ch == "*" and nxt == "/":
                state = "code"
                i += 1
        else:
            quote = '"' if state == "string" else "'"
            if ch == "\\":
                i += 1
            elif ch == quote:
                state = "code"
        i += 1
    raise AssertionError(f"unclosed function {signature}")


def function_or_none(text: str, signature: str) -> str | None:
    # A missing function is a violation to report, not an exception to raise:
    # the checker must stay usable against a tree that predates the guard.
    if signature not in text:
        return None
    return function(text, signature)


def violations(source: str) -> list[str]:
    found: list[str] = []

    # The advertised alignment is one named constant, not a repeated literal --
    # otherwise the promise and the request can drift apart silently.
    if not re.search(r"constexpr\s+size_t\s+" + CONSTANT + r"\s*=\s*\d+\w*\s*;", source):
        found.append(f"{CONSTANT} is not defined as a constant")

    for getter in (
        "ggml_backend_sycl_buffer_type_get_alignment",
        "tiered_kv_buft_get_alignment",
        "ggml_backend_sycl_split_buffer_type_get_alignment",
        "ggml_backend_sycl_tp_buffer_type_get_alignment",
    ):
        body = function_or_none(source, "static size_t " + getter)
        if body is None:
            found.append(f"{getter} is missing")
        elif CONSTANT not in body:
            found.append(f"{getter} advertises an alignment literal instead of {CONSTANT}")

    alloc = function_or_none(source, ALLOC_BUFFER)
    if alloc is None:
        return found + [f"{ALLOC_BUFFER} is missing"]

    # Every allocation request the buffer type makes must ask for at least the
    # alignment it advertises.  unified_alloc substitutes a weaker default (64)
    # when alignment is left at 0, which is what produced the 64-byte shortfall.
    requests = re.findall(r"(?:ggml_sycl::)?alloc_request\s+(\w+)\s*[{;]", alloc)
    if not requests:
        found.append("no alloc_request found in the buffer-type allocator")
    for name in requests:
        # (?<!\w) matters: without it "req" is satisfied by "runtime_req.alignment"
        # and the plain device-path request goes unchecked.
        if not re.search(r"(?<!\w)" + re.escape(name) + r"\.alignment\s*=\s*" + CONSTANT + r"\s*;", alloc):
            found.append(f"alloc_request '{name}' does not request {CONSTANT}")

    # Publishing must go through the guarded helper; a bare buffer_init would
    # hand out whatever base the allocator happened to return.
    if "ggml_backend_buffer_init(" in alloc:
        found.append("buffer-type allocator publishes a buffer without the alignment guard")
    if PUBLISH not in alloc:
        found.append(f"buffer-type allocator does not publish through {PUBLISH}")

    guard = function_or_none(source, "static ggml_backend_buffer_t " + PUBLISH)
    if guard is None:
        found.append(f"{PUBLISH} is missing")
    else:
        if f"% {CONSTANT}) != 0" not in guard:
            found.append(f"{PUBLISH} does not test the base against {CONSTANT}")
        if "return nullptr;" not in guard:
            found.append(f"{PUBLISH} does not refuse a misaligned base")
        if "GGML_LOG_WARN" not in guard:
            # INFO is dropped at default verbosity, so a misalignment would be silent.
            found.append(f"{PUBLISH} does not report a misaligned base at WARN")

    # The tiered-KV buffer type adopts an arena pointer as its allocator base;
    # that pointer is only usable if it is already aligned.
    tiered = function_or_none(source, "static ggml_backend_buffer_t tiered_kv_buft_alloc_buffer")
    if tiered is None:
        found.append("tiered_kv_buft_alloc_buffer is missing")
    elif not re.search(r"arena_first_ptr\)\s*%\s*alloc_align\)\s*==\s*0", tiered):
        found.append("tiered_kv_buft_alloc_buffer adopts arena_first_ptr without an alignment test")

    return found


def tlsf_violations(source: str) -> list[str]:
    """The allocator must keep block offsets on MIN_BLOCK_SIZE granularity.

    Offsets are what become device pointers.  split_block() puts the remainder at
    (offset + size), so rounding an allocation by the caller's `alignment` — which
    most callers leave at the allocator default of 64 — produces offsets that are
    64- but not 128/256-aligned, and every buffer cut from them inherits that.
    """
    found: list[str] = []

    body = function_or_none(source, "inline size_t tlsf_allocator::allocate")
    if body is None:
        return ["tlsf_allocator::allocate is missing"]

    # The rounding granularity must be at least MIN_BLOCK_SIZE, never bare alignment.
    if not re.search(r"granularity\s*=\s*alignment\s*>\s*MIN_BLOCK_SIZE\s*\?\s*alignment\s*:\s*MIN_BLOCK_SIZE", body):
        found.append("allocate() does not raise the rounding granularity to MIN_BLOCK_SIZE")
    if not re.search(r"size\s*=\s*\(size\s*\+\s*granularity\s*-\s*1\)\s*&\s*~\(granularity\s*-\s*1\)", body):
        found.append("allocate() does not round the size by the granularity")
    if re.search(r"size\s*=\s*\(size\s*\+\s*alignment\s*-\s*1\)\s*&\s*~\(alignment\s*-\s*1\)", body):
        found.append("allocate() still rounds the size by the caller's alignment")

    # The invariant is checked where it is produced, not merely documented.
    if not re.search(r"TLSF_ASSERT\(\(blocks_\[block_id\]\.offset\s*%\s*MIN_BLOCK_SIZE\)\s*==\s*0", body):
        found.append("allocate() does not assert the returned offset's MIN_BLOCK_SIZE alignment")

    return found


def zone_fallthrough_violations(source: str) -> list[str]:
    """A refused zone buffer must fall through to the next tier, not fail the allocator.

    The buffer-type allocator tries RUNTIME -> KV -> SCRATCH -> legacy.  Returning
    the guard's nullptr from any of the first three skips the tiers below it and
    turns a recoverable misalignment into a failed allocation.
    """
    alloc = function_or_none(source, ALLOC_BUFFER)
    if alloc is None:
        return [f"{ALLOC_BUFFER} is missing"]

    found: list[str] = []
    for zone in ("arena RUNTIME zone", "arena KV zone", "arena SCRATCH zone"):
        if re.search(r"return\s+" + PUBLISH + r"\(buft, ctx, size, \"" + re.escape(zone) + r"\"\)", alloc):
            found.append(f"'{zone}' returns the publish result instead of falling through")
        if not re.search(r"=\s*\n?\s*" + PUBLISH + r"\(buft, ctx, size, \"" + re.escape(zone) + r"\"\)", alloc):
            found.append(f"'{zone}' does not publish through the guarded helper")
    return found


def mmvq_extent_violations(source: str) -> list[str]:
    """mmvq's raw-pointer copies must supply their own length as the trusted extent.

    The grouped row-aggregation copies source plain host std::vectors, which are
    neither registered in alloc_registry nor USM chunks, so without a trusted
    extent every handle resolves to extent 0 and the bounded mem-op aborts on a
    perfectly valid copy (mem-ops.cpp require_resolved_range).
    """
    found: list[str] = []

    helper = function_or_none(source, "static ggml_sycl::mem_handle mmvq_memcpy_handle_for_raw_ptr")
    if helper is None:
        return ["mmvq_memcpy_handle_for_raw_ptr is missing"]
    if not re.search(r"trusted_extent\s*=\*/\s*bytes", helper):
        found.append("mmvq_memcpy_handle_for_raw_ptr does not forward bytes as the trusted extent")
    if not re.search(r"size_t\s+bytes", helper):
        found.append("mmvq_memcpy_handle_for_raw_ptr does not take the copy length")

    # Both wrappers must pass their length through; a call missing it would
    # otherwise silently reintroduce the extent-0 handle.
    for fn in ("static sycl::event mmvq_submit_memcpy_with_deps", "static void mmvq_memcpy_sync"):
        body = function_or_none(source, fn)
        if body is None:
            found.append(f"{fn} is missing")
            continue
        calls = re.findall(r"mmvq_memcpy_handle_for_raw_ptr\(([^;]*?)\)\s*;", body, re.S)
        if len(calls) != 2:
            found.append(f"{fn} does not build exactly two raw-pointer handles")
        for call in calls:
            if not re.search(r",\s*bytes\s*$", call.strip()):
                found.append(f"{fn} builds a handle without passing bytes")
    return found


def test_sycl_buffer_bases_honour_advertised_alignment() -> None:
    assert violations(SOURCE.read_text()) == []


def test_mmvq_raw_copies_carry_a_trusted_extent() -> None:
    assert mmvq_extent_violations(MMVQ.read_text()) == []


def test_mmvq_extent_mutations_are_witnessed() -> None:
    mmvq = MMVQ.read_text()
    mutations = [
        mmvq.replace("/*trusted_extent=*/bytes", "/*trusted_extent=*/0", 1),
        re.sub(r"mmvq_memcpy_handle_for_raw_ptr\(dst, queue_device, dst_fallback_on_device, bytes\)",
               "mmvq_memcpy_handle_for_raw_ptr(dst, queue_device, dst_fallback_on_device, 0)", mmvq, count=1),
    ]
    for index, mutated in enumerate(mutations):
        assert mutated != mmvq, f"mmvq mutation {index} did not change the source"
        assert mmvq_extent_violations(mutated), f"mmvq mutation {index} was not witnessed"


def test_tlsf_offsets_stay_block_aligned() -> None:
    assert tlsf_violations(TLSF.read_text()) == []


def test_refused_zone_buffers_fall_through() -> None:
    assert zone_fallthrough_violations(SOURCE.read_text()) == []


def test_tlsf_and_fallthrough_mutations_are_witnessed() -> None:
    tlsf = TLSF.read_text()
    tlsf_mutations = [
        # Reinstating the alignment-based rounding is the llama.cpp-f8ws defect.
        tlsf.replace("const size_t granularity = alignment > MIN_BLOCK_SIZE ? alignment : MIN_BLOCK_SIZE;",
                     "const size_t granularity = alignment;", 1),
        # \s* not \n: clang-format may join this assert onto one line.
        re.sub(r"TLSF_ASSERT\(\(blocks_\[block_id\]\.offset % MIN_BLOCK_SIZE\) == 0 &&\s*\"[^\"]*\"\);",
               "", tlsf, count=1),
    ]
    for index, mutated in enumerate(tlsf_mutations):
        assert mutated != tlsf, f"tlsf mutation {index} did not change the source"
        assert tlsf_violations(mutated), f"tlsf mutation {index} was not witnessed"

    source = SOURCE.read_text()
    fallthrough_mutations = [
        re.sub(r"if \(ggml_backend_buffer_t published =\s*\n?\s*" + PUBLISH +
               r"\(buft, ctx, size, \"arena RUNTIME zone\"\)\) \{\s*\n\s*return published;\s*\n\s*\}",
               f"return {PUBLISH}(buft, ctx, size, \"arena RUNTIME zone\");", source, count=1),
    ]
    for index, mutated in enumerate(fallthrough_mutations):
        assert mutated != source, f"fallthrough mutation {index} did not change the source"
        assert zone_fallthrough_violations(mutated), f"fallthrough mutation {index} was not witnessed"


def _drop_alignment(source: str, request: str) -> str:
    # Whitespace between the field and '=' is column alignment and drifts with
    # clang-format, so match it loosely rather than pinning today's spacing.
    # (?<!\w) keeps "req" from matching the tail of "runtime_req"/"kv_req",
    # which would silently make two mutations identical.
    return re.sub(r"(?<!\w)" + re.escape(request) + r"\.alignment\s*=\s*" + CONSTANT + r"\s*;", "", source, count=1)


def test_mutations_are_witnessed() -> None:
    source = SOURCE.read_text()
    mutations = [
        # Each of these reintroduces the llama.cpp-f8ws under-reservation.
        _drop_alignment(source, "req"),
        _drop_alignment(source, "runtime_req"),
        _drop_alignment(source, "kv_req"),
        _drop_alignment(source, "scratch_req"),
        source.replace(f"return {PUBLISH}(buft, ctx, size, \"device\");",
                       "return ggml_backend_buffer_init(buft, ggml_backend_sycl_buffer_interface, ctx, size);", 1),
        source.replace(f"% {CONSTANT}) != 0", "% 1) != 0", 1),
        re.sub(r"GGML_LOG_WARN\(\s*\n\s*\"\[SYCL\] refusing", "GGML_LOG_INFO(\n            \"[SYCL] refusing",
               source, count=1),
        source.replace("(reinterpret_cast<uintptr_t>(arena_first_ptr) % alloc_align) == 0", "true", 1),
    ]
    for index, mutated in enumerate(mutations):
        assert mutated != source, f"mutation {index} did not change the source"
        assert violations(mutated), f"mutation {index} was not witnessed"
