from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GGML_SYCL = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
UNIFIED_CACHE = ROOT / "ggml/src/ggml-sycl/unified-cache.cpp"


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


def assert_no_waits(source: str) -> None:
    assert ".wait(" not in source
    assert "wait_and_throw" not in source


def positions(source: str, needle: str) -> list[int]:
    result: list[int] = []
    start = 0
    while True:
        try:
            index = source.index(needle, start)
        except ValueError:
            return result
        result.append(index)
        start = index + len(needle)


def assert_record_is_gated(source: str, record_pos: int) -> tuple[int, int]:
    gate = source.rindex("if (e2e_tg_profile_enabled())", 0, record_pos)
    gate_close = matching_brace(source, source.index("{", gate))
    assert gate < record_pos < gate_close
    assert_no_waits(source[gate:gate_close])
    return gate, gate_close


def call_argument(source: str, call_pos: int) -> str:
    open_paren = source.index("(", call_pos)
    close_paren = source.index(")", open_paren)
    return source[open_paren + 1 : close_paren].strip()


def host_fallback_counter_sites(source: str) -> list[int]:
    needle = "offload_stats_note_host_fallback_attempt("
    return [pos for pos in positions(source, needle) if "void " not in source[max(0, pos - 8) : pos]]


def test_unified_cache_records_host_fallback_and_zone_failures() -> None:
    src = UNIFIED_CACHE.read_text(encoding="utf-8")
    assert '#include "e2e-profile.hpp"' in src

    host_calls = host_fallback_counter_sites(src)
    host_records = positions(src, 'e2e_tg_profile_record_cache_event("host_fallback"')
    assert len(host_calls) == len(host_records)
    assert len(host_calls) >= 10
    for call in host_calls:
        bytes_arg = call_argument(src, call)
        record = src.index('e2e_tg_profile_record_cache_event("host_fallback"', call)
        assert record - call < 300
        _, gate_close = assert_record_is_gated(src, record)
        assert f'e2e_tg_profile_record_cache_event("host_fallback", {bytes_arg}, 0.0)' in src[record:gate_close]

    zone_begin = src.index("void * unified_cache::zone_alloc")
    zone_end = src.index("void unified_cache::zone_free", zone_begin)
    zone_body = src[zone_begin:zone_end]
    assert zone_body.count("zone_alloc_failures") == 2
    zone_records = positions(zone_body, 'e2e_tg_profile_record_cache_event("zone_alloc_failed"')
    assert len(zone_records) == 2
    for record in zone_records:
        _, gate_close = assert_record_is_gated(zone_body, record)
        assert 'e2e_tg_profile_record_cache_event("zone_alloc_failed", size, 0.0)' in zone_body[record:gate_close]


def test_peer_host_bounce_measure_records_transfer_stage() -> None:
    src = GGML_SYCL.read_text(encoding="utf-8")
    begin = src.index("static void ggml_sycl_measure_peer_host_bounce")
    end = src.index("static void ggml_sycl_log_peer_link_info", begin)
    body = src[begin:end]
    host_us = body.index("link.host_bounce_us")
    record = body.index('e2e_tg_profile_record_transfer("peer_host_bounce_measure"', host_us)
    gate = body.rindex("if (ggml_sycl::e2e_tg_profile_enabled())", 0, record)
    gate_close = matching_brace(body, body.index("{", gate))
    assert host_us < gate < record < gate_close
    record_call = body[record:gate_close]
    assert 'e2e_tg_profile_record_transfer("peer_host_bounce_measure", bytes, 0.0' in record_call
    assert "link.host_bounce_us" in record_call
    assert "link.host_bounce_d2h_us" in body[:record]
    assert "link.host_bounce_h2d_us" in body[:record]
    assert_no_waits(body[host_us:gate_close])
