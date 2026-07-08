from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIFIED_CACHE = ROOT / "ggml/src/ggml-sycl/unified-cache.cpp"


def positions(source: str, needle: str) -> list[int]:
    result: list[int] = []
    start = 0
    while True:
        index = source.find(needle, start)
        if index < 0:
            return result
        result.append(index)
        start = index + len(needle)


def test_host_fallback_e2e_sites_also_record_timeline_cache_spans() -> None:
    src = UNIFIED_CACHE.read_text(encoding="utf-8")
    assert '#include "sycl-timeline.hpp"' in src

    e2e_records = positions(src, 'e2e_tg_profile_record_cache_event("host_fallback"')
    timeline_records = positions(src, 'GGML_SYCL_TIMELINE_SCOPE("cache", "host_fallback"')

    assert len(e2e_records) >= 1
    assert len(timeline_records) == len(e2e_records)


def test_zone_alloc_failed_e2e_sites_also_record_timeline_cache_spans() -> None:
    src = UNIFIED_CACHE.read_text(encoding="utf-8")
    assert '#include "sycl-timeline.hpp"' in src

    e2e_records = positions(src, 'e2e_tg_profile_record_cache_event("zone_alloc_failed"')
    timeline_records = positions(src, 'GGML_SYCL_TIMELINE_SCOPE("cache", "zone_alloc_failed"')

    assert len(e2e_records) >= 2
    assert len(timeline_records) == len(e2e_records)
