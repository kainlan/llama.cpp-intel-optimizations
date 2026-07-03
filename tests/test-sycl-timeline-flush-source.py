from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GGML_SYCL = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"


def read_source() -> str:
    return GGML_SYCL.read_text(encoding="utf-8")


def positions(source: str, needle: str) -> list[int]:
    result: list[int] = []
    start = 0
    while True:
        index = source.find(needle, start)
        if index < 0:
            return result
        result.append(index)
        start = index + len(needle)


def test_decode_teardown_flushes_timeline_near_e2e_force_flush_without_waits() -> None:
    src = read_source()
    e2e_flushes = positions(src, "ggml_sycl::e2e_tg_profile_force_flush(")
    assert e2e_flushes

    matching_windows = []
    for e2e_flush in e2e_flushes:
        window = src[max(0, e2e_flush - 400) : e2e_flush + 500]
        if 'ggml_sycl::sycl_timeline_flush("decode-teardown")' in window:
            matching_windows.append(window)

    assert matching_windows
    window = matching_windows[0]
    assert "cached_is_decode" in window
    assert "!g_ggml_sycl_graph_recording" in window
    assert "ggml_sycl::sycl_timeline_enabled()" in window
    assert "std::atexit" not in window
    assert ".wait(" not in window
    assert "wait_and_throw" not in window


def test_backend_free_flushes_timeline_after_kernel_profile_drain_without_waits() -> None:
    src = read_source()
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
