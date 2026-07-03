from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GGML_SYCL = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"


def read_source() -> str:
    return GGML_SYCL.read_text(encoding="utf-8")


def matching_brace(source: str, open_brace: int) -> int:
    depth = 0
    for index in range(open_brace, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    raise AssertionError("no matching brace")


def test_graph_compute_impl_has_timeline_scope_after_reentry_guard() -> None:
    src = read_source()
    assert '#include "sycl-timeline.hpp"' in src

    begin = src.index("static void ggml_backend_sycl_graph_compute_impl")
    open_brace = src.index("{", begin)
    close_brace = matching_brace(src, open_brace)
    body = src[open_brace:close_brace]

    guard = body.index("compute_impl_guard _reentry_guard")
    scope = body.index('GGML_SYCL_TIMELINE_SCOPE("ggml.graph", "graph_compute_impl"', guard)
    compute_start = body.index("auto t_compute_start", guard)

    assert guard < scope < compute_start
    scope_setup = body[guard:compute_start]
    assert ".wait(" not in scope_setup
    assert "wait_and_throw" not in scope_setup
