from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GGML_SYCL = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
SYCL_TIMELINE = ROOT / "ggml/src/ggml-sycl/sycl-timeline.cpp"

GRAPH_COMPUTE = "static ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {"
BACKEND_FREE = "static void ggml_backend_sycl_free(ggml_backend_t backend) {"

TIMELINE_FLUSH_CALL = "sycl_timeline_flush("


def read_source() -> str:
    return GGML_SYCL.read_text(encoding="utf-8")


def strip_comments(source: str) -> str:
    """Blank out comments, preserving byte offsets and line structure.

    These tests assert on the presence and placement of *calls*. Without this,
    a comment that merely names sycl_timeline_flush() -- such as the one that
    documents why the per-step call site was removed -- reads as a call and
    makes the guard vacuous in exactly the direction that hides a regression.
    """
    out = list(source)
    i = 0
    n = len(source)
    while i < n:
        c = source[i]
        if c == "/" and i + 1 < n and source[i + 1] == "/":
            while i < n and source[i] != "\n":
                out[i] = " "
                i += 1
        elif c == "/" and i + 1 < n and source[i + 1] == "*":
            out[i] = out[i + 1] = " "
            i += 2
            while i < n and not (source[i] == "*" and i + 1 < n and source[i + 1] == "/"):
                if source[i] != "\n":
                    out[i] = " "
                i += 1
            if i < n:
                out[i] = out[i + 1] = " "
                i += 2
        elif c in ('"', "'"):
            quote = c
            i += 1
            while i < n and source[i] != quote:
                i += 2 if source[i] == "\\" else 1
            i += 1
        else:
            i += 1
    return "".join(out)


def positions(source: str, needle: str) -> list[int]:
    result: list[int] = []
    start = 0
    while True:
        index = source.find(needle, start)
        if index < 0:
            return result
        result.append(index)
        start = index + len(needle)


def function_body_span(source: str, signature: str) -> tuple[int, int]:
    """Byte span of one function's body, located by brace matching."""
    begin = source.index(signature)
    open_brace = source.index("{", begin + len(signature) - 1)
    depth = 0
    for index in range(open_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return open_brace, index
    raise AssertionError(f"unbalanced braces after {signature!r}")


def test_decode_teardown_does_not_flush_the_timeline() -> None:
    """The per-step decode call site must stay gone; its e2e sibling stays.

    Removing the timeline flush from ggml_backend_sycl_graph_compute() is the
    fix for the first-step-wins capture; e2e_tg_profile_force_flush() is a
    separate instrument with its own repeat-safe flush and must be untouched.
    """
    src = strip_comments(read_source())

    e2e_flushes = positions(src, "ggml_sycl::e2e_tg_profile_force_flush(")
    assert e2e_flushes

    for e2e_flush in e2e_flushes:
        window = src[max(0, e2e_flush - 400) : e2e_flush + 500]
        assert TIMELINE_FLUSH_CALL not in window

    assert 'sycl_timeline_flush("decode-teardown")' not in src


def test_backend_free_flushes_timeline_after_kernel_profile_drain_without_waits() -> None:
    src = strip_comments(read_source())
    kernel_flush = 'ggml_sycl_kernel_profile_flush(true, "backend-free");'
    timeline_flush = 'ggml_sycl::sycl_timeline_flush("backend-free");'
    delete_backend = "delete sycl_ctx;"

    begin = src.index(kernel_flush)
    window = src[begin : begin + 500]

    assert timeline_flush in window
    assert window.index(kernel_flush) < window.index(timeline_flush) < window.index(delete_backend)
    assert "ggml_sycl::sycl_timeline_enabled()" in window
    assert "!ggml_sycl::sycl_timeline_has_flushed_file()" in window
    assert "std::atexit" not in window
    assert ".wait(" not in window
    assert "wait_and_throw" not in window


def test_one_shot_timeline_flush_is_reachable_only_from_backend_teardown() -> None:
    """The composition gate: a one-shot flush may only run on a once-per-process path.

    Each call site is individually correct, so neither test above can catch
    this. A flush placed on the per-step path wins the one-shot race at decode
    step 1, writing a single-step trace with no device spans (the drain has not
    run yet) and permanently locking out the backend-free site. That is the
    whole defect, and it lives only in the composition.
    """
    timeline_src = strip_comments(SYCL_TIMELINE.read_text(encoding="utf-8"))
    # The premise. If the flush ever becomes re-entrant this assertion fails
    # loudly, which is the signal to revisit the placement rule below rather
    # than to delete it silently.
    assert "state.successful_file_flushes > 0" in timeline_src

    src = strip_comments(read_source())

    compute_begin, compute_end = function_body_span(src, GRAPH_COMPUTE)
    assert TIMELINE_FLUSH_CALL not in src[compute_begin:compute_end]

    free_begin, free_end = function_body_span(src, BACKEND_FREE)
    calls = positions(src, TIMELINE_FLUSH_CALL)
    assert calls
    for call in calls:
        assert free_begin < call < free_end
