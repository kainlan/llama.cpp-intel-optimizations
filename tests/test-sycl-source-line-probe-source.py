from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS_CMAKE = ROOT / "tools" / "CMakeLists.txt"
PROBE_CMAKE = ROOT / "tools/sycl-source-line-probe/CMakeLists.txt"
PROBE_MAIN = ROOT / "tools/sycl-source-line-probe/main.cpp"


def test_source_line_probe_is_registered_only_for_sycl_tools() -> None:
    text = TOOLS_CMAKE.read_text(encoding="utf-8")
    sycl_block = text[text.index("if (GGML_SYCL)") : text.index("endif()", text.index("if (GGML_SYCL)"))]
    assert "add_subdirectory(sycl-source-line-probe)" in sycl_block


def test_source_line_probe_is_single_cu_with_named_hot_line() -> None:
    cmake = PROBE_CMAKE.read_text(encoding="utf-8")
    main = PROBE_MAIN.read_text(encoding="utf-8")
    assert "add_executable(sycl-source-line-probe main.cpp)" in cmake
    assert "-fsycl-device-code-split=per_kernel" in cmake
    assert "class sycl_source_line_probe_kernel" in main
    assert "SOURCE_LINE_PROBE_HOT_LINE" in main
    assert "llama-bench" not in main
    assert "/Storage" not in main
    assert "--iterations" in main
    assert "--size" in main
    assert "--json" in main


def test_source_line_probe_rejects_signed_or_whitespace_numeric_args_in_source() -> None:
    main = PROBE_MAIN.read_text(encoding="utf-8")
    assert "static bool is_ascii_digit_string" in main
    assert "std::isdigit(static_cast<unsigned char>(ch))" in main
    assert "if (!is_ascii_digit_string(text))" in main
    assert "std::stoull" in main
