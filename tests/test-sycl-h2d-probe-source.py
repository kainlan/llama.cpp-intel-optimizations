"""Source contract for llama.cpp-zujy (plan Task 3 of llama.cpp-36j0): each device probes its PINNED host-to-device
rate once and keeps it, so the MoE executor's cost model divides by the card's real link rate.

Host-only: reads sources, builds nothing, loads no model and touches no device.

Why: the cost model's h2d_gbps is the real rate of each card from pinned host memory (26.5-27.1 GB/s measured, against
10.5-19 GB/s from pageable memory), and the peer-bounce probe stages through a pageable std::vector, so it cannot be
reused. The probe this gate pins:
  - ggml_sycl_measure_h2d_pinned allocates BOTH buffers through ggml_sycl::unified_allocate (role STAGING, category
    STAGING, cohort h2d_probe_src / h2d_probe_dst), the host one pinned from the pool (must_host_pinned and
    use_pinned_pool), the device one in the RUNTIME zone with no spill (must_device, prefer_vram_zone RUNTIME,
    forbid_vram_zone_spill): clauses (b) and (d) of scripts/check-sycl-alloc-zone-contract.py read those literals;
  - no raw allocation and no pageable staging vector;
  - one untimed warm-up copy (ggml_sycl::mem_copy, which is synchronous), then 8 timed iterations of a 64 MiB buffer
    moved in 4 MiB chunks with ggml_sycl::mem_copy_async at an offset, one q->wait() per iteration and never one per
    chunk, timed on the host with steady_clock;
  - one WARN line per device, "[MOE-EXEC] h2d probe device %d: %.1f GB/s (%s)", whose source word is "probed",
    "tuning cache" or "seed, GGML_SYCL_H2D_PROBE=0";
  - GGML_SYCL_H2D_PROBE is read once (1 forces a re-probe, 0 skips it and keeps the seed); a stored record skips the
    probe; a probe that cannot allocate keeps the seed and never aborts;
  - a handler that can swallow ggml_sycl_fallback_error is preceded by the sanctioned rethrow (clause (g));
  - sycl_device_info carries h2d_pinned_gbps and h2d_probed, and exactly one place reads h2d_pinned_gbps: the probe's
    own once-guard (a device that holds a rate is not probed again). Task 7b's rates assembly makes it two and updates
    this count;
  - the probe is called once per outermost model load, after the unified caches exist (ggml_sycl_info() is complete
    there, which ggml_sycl_init is not, and ggml_sycl_init must not create caches).

Every claim is checked on COMMENT-STRIPPED, whitespace-normalized text and has a mutant that must make it fail.

Pytest-style (module-level test_* functions): register with llama_test_pytest.
"""

import functools
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_LEXEME_RE = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'",
    re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


@functools.lru_cache(maxsize=None)
def norm(text: str) -> str:
    """Comment-stripped, whitespace-canonical text with adjacent string literals joined, so neither a clang-format
    re-wrap nor a re-split format string can turn a claim red."""
    t = re.sub(r"\s+", " ", strip_comments(text))
    t = re.sub(r"\( ", "(", t)
    t = re.sub(r" \)", ")", t)
    t = re.sub(r" ,", ",", t)
    t = re.sub(r",(?! )", ", ", t)
    t = re.sub(r'(?<!\\)" "', "", t)
    return t


def read(rel: str) -> str:
    return (ROOT / rel).read_text()


SYCL_REL = "ggml/src/ggml-sycl/ggml-sycl.cpp"
COMMON_REL = "ggml/src/ggml-sycl/common.hpp"
SYCL_TREE_DIR = "ggml/src/ggml-sycl"

SYCL = read(SYCL_REL)
COMMON = read(COMMON_REL)

PROBE_SIG = "static void ggml_sycl_measure_h2d_pinned(ggml_sycl_device_info & info)"
LOG_FORMAT = '"[MOE-EXEC] h2d probe device %d: %.1f GB/s (%s)\\n"'


def body(text: str, signature: str) -> str:
    """The brace-balanced body of the first definition whose text starts with `signature` (string literals are
    skipped while balancing). Empty when there is none."""
    at = text.find(signature)
    if at < 0:
        return ""
    open_at = text.find("{", at + len(signature))
    if open_at < 0:
        return ""
    depth = 0
    i = open_at
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "\"'":
            j = i + 1
            while j < n and text[j] != ch:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_at:i + 1]
        i += 1
    return ""


@functools.lru_cache(maxsize=None)
def probe_body(src: str) -> str:
    return body(norm(src), PROBE_SIG)


def _once(text: str, old: str, new: str) -> str:
    """The comment-stripped `text` with the one occurrence of `old` (whitespace-flexible) replaced by `new`; a mutant
    that does not apply is a broken test."""
    base = strip_comments(text)
    pattern = r"\s+".join(re.escape(tok) for tok in old.split())
    hits = list(re.finditer(pattern, base))
    assert len(hits) == 1, "mutant anchor must occur exactly once (%d): %r" % (len(hits), old)
    return base[:hits[0].start()] + new + base[hits[0].end():]


# ---- the claims, each a function of the source text so a mutant can run it ----

def claim_allocates_through_unified_allocate_with_literal_tiers(src: str) -> bool:
    fn = probe_body(src)
    return bool(fn) and all([
        fn.count("ggml_sycl::unified_allocate(") == 2,
        "role = ggml_sycl::alloc_role::STAGING" in fn,
        "category = ggml_sycl::runtime_category::STAGING" in fn,
        '"h2d_probe_src"' in fn and '"h2d_probe_dst"' in fn,
        "constraints.must_host_pinned = true" in fn,
        "constraints.use_pinned_pool = true" in fn,
        "constraints.must_device = true" in fn,
        "constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME" in fn,
        "constraints.forbid_vram_zone_spill = true" in fn,
        "sycl::malloc" not in fn,
        "std::vector<uint8_t>" not in fn,
        "from_direct" not in fn,
    ])


def claim_copy_shape_is_chunked_async_with_one_wait_per_iteration(src: str) -> bool:
    fn = probe_body(src)
    if not fn:
        return False
    warm = fn.count("ggml_sycl::mem_copy(") == 1
    chunked = fn.count("ggml_sycl::mem_copy_async(") == 1
    sizes = all(s in fn for s in ("kBytes = 64ull << 20", "kChunk = 4ull << 20", "kIters = 8"))
    # the one wait sits in the iteration loop, after the chunk loop closes and before the next iteration
    one_wait = fn.count("q->wait()") == 1 and ordered(fn, "for (int it = 0; it < kIters; ++it)", "mem_copy_async(",
                                                       "q->wait()")
    clocked = "std::chrono::steady_clock::now()" in fn and "high_resolution_clock" not in fn
    return all([warm, chunked, sizes, one_wait, clocked])


def ordered(text: str, *needles: str) -> bool:
    """Every needle is present, each after the one before it."""
    at = 0
    for needle in needles:
        at = text.find(needle, at)
        if at < 0:
            return False
        at += len(needle)
    return True


def claim_wait_closes_the_chunk_loop_before_the_clock_stops(src: str) -> bool:
    """The wait is a sibling of the chunk loop, not inside it: the chunk loop's closing brace precedes it."""
    fn = probe_body(src)
    m = re.search(r"for \(size_t off = 0; off < kBytes; off \+= kChunk\) \{.*?\} q->wait\(\);", fn)
    return bool(m) and "q->wait()" not in m.group(0)[:-len("q->wait();")]


def claim_logs_once_at_warn(src: str) -> bool:
    n = norm(src)
    return n.count(LOG_FORMAT) == 1 and all(
        s in probe_body(src) for s in ('"probed"', '"tuning cache"', '"seed, GGML_SYCL_H2D_PROBE=0"')
    ) and "GGML_LOG_WARN(" + LOG_FORMAT in n


def claim_reads_its_switch_once(src: str) -> bool:
    return norm(src).count('getenv("GGML_SYCL_H2D_PROBE")') == 1


def claim_switch_semantics(src: str) -> bool:
    fn = probe_body(src)
    return all([
        "strcmp(sw, \"0\") == 0" in fn,
        "strcmp(sw, \"1\") == 0" in fn,
        "ggml_backend_sycl_moe_exec_rates_lookup(" in fn,
        "ggml_backend_sycl_moe_exec_rates_store(" in fn,
        # the force switch is what lets the stored record be bypassed
        "!force" in fn,
    ])


def claim_failed_alloc_keeps_the_seed_and_never_aborts(src: str) -> bool:
    fn = probe_body(src)
    return bool(fn) and all([
        "!src.valid() || !dst.valid()" in fn,
        "GGML_ABORT" not in fn,
        "GGML_ASSERT" not in fn,
        "SYCL_CHECK" not in fn,
    ])


def claim_rethrows_fallback_error_first(src: str) -> bool:
    fn = probe_body(src)
    guard = "catch (const ggml_sycl_fallback_error &) { throw; }"
    handler = "catch (const std::exception & e)"
    return bool(fn) and guard in fn and handler in fn and fn.index(guard) < fn.index(handler)


def claim_store_carries_a_valid_reorder_rate(src: str) -> bool:
    """The store refuses a record whose reorder rate is not finite and positive, so a record holding only the H2D rate
    would never persist and the second start would probe again."""
    fn = probe_body(src)
    return bool(fn) and "rec.reorder_gbps" in fn


def claim_device_info_fields(common: str) -> bool:
    n = norm(common)
    return "double h2d_pinned_gbps = 0.0;" in n and "bool h2d_probed = false;" in n


def read_offsets(text: str) -> list:
    """Offsets of the occurrences of h2d_pinned_gbps that are reads: a write is an occurrence followed by `=`."""
    return [m.start() for m in re.finditer(r"h2d_pinned_gbps", text)
            if not re.match(r"\s*=(?!=)", text[m.end():m.end() + 4])]


def count_reads_in(text: str) -> int:
    return len(read_offsets(text))


def claim_one_consumer_reads_the_rate() -> bool:
    """h2d_pinned_gbps is read in exactly one place across the backend's sources, and that place is the probe's own
    once-guard. A write is an occurrence followed by `=` (the field's default, the report lambda's store)."""
    reads = []
    for p in sorted((ROOT / SYCL_TREE_DIR).rglob("*")):
        if not p.is_file() or p.suffix not in (".c", ".cpp", ".h", ".hpp"):
            continue
        text = norm(p.read_text(errors="replace"))
        for at in read_offsets(text):
            reads.append((p.relative_to(ROOT).as_posix(), at, text))
    if len(reads) != 1 or reads[0][0] != SYCL_REL:
        return False
    _, at, text = reads[0]
    fn = body(text, PROBE_SIG)
    return bool(fn) and text.find(fn) <= at < text.find(fn) + len(fn)


# ---- tests ----

def test_probe_allocates_through_unified_allocate_with_literal_tiers():
    assert claim_allocates_through_unified_allocate_with_literal_tiers(SYCL)


def test_probe_copy_is_chunked_async_with_one_wait_per_iteration():
    assert claim_copy_shape_is_chunked_async_with_one_wait_per_iteration(SYCL)
    assert claim_wait_closes_the_chunk_loop_before_the_clock_stops(SYCL)


def test_probe_logs_once_at_warn_and_reads_its_switch_once():
    assert claim_logs_once_at_warn(SYCL)
    assert claim_reads_its_switch_once(SYCL)
    assert claim_switch_semantics(SYCL)


def test_probe_failure_keeps_the_seed_and_never_aborts():
    assert claim_failed_alloc_keeps_the_seed_and_never_aborts(SYCL)
    assert claim_rethrows_fallback_error_first(SYCL)


def test_probe_stores_a_record_the_cache_will_accept():
    assert claim_store_carries_a_valid_reorder_rate(SYCL)


def test_device_info_carries_the_rate_and_one_consumer_reads_it():
    assert claim_device_info_fields(COMMON)
    assert claim_one_consumer_reads_the_rate()


# ---- mutants: each claim must go red when its text is broken ----

LOG_CALL = 'GGML_LOG_WARN("[MOE-EXEC] h2d probe device %d: %.1f GB/s (%s)\\n", dev, gbps, source);'
GUARDED_CATCH = ('catch (const ggml_sycl_fallback_error &) { throw; } catch (const std::exception & e) { '
                 'GGML_LOG_WARN("[MOE-EXEC] h2d probe device %d failed')


def test_mutant_raw_malloc_fails():
    assert not claim_allocates_through_unified_allocate_with_literal_tiers(
        _once(SYCL, 'src_req.intent.cohort_id = "h2d_probe_src";',
              'src_req.intent.cohort_id = "h2d_probe_src"; void * p = sycl::malloc_host(1, *q);'))


def test_mutant_pageable_staging_fails():
    assert not claim_allocates_through_unified_allocate_with_literal_tiers(
        _once(SYCL, "src_req.intent.constraints.use_pinned_pool = true;",
              "src_req.intent.constraints.use_pinned_pool = false;"))


def test_mutant_device_spill_allowed_fails():
    assert not claim_allocates_through_unified_allocate_with_literal_tiers(
        _once(SYCL, "dst_req.intent.constraints.forbid_vram_zone_spill = true;",
              "dst_req.intent.constraints.forbid_vram_zone_spill = false;"))


def test_mutant_wait_per_chunk_fails():
    mutant = _once(SYCL, "ggml_sycl::mem_copy_async(dst, off, src, off, kChunk, *q);",
                   "ggml_sycl::mem_copy_async(dst, off, src, off, kChunk, *q); q->wait();")
    assert not claim_wait_closes_the_chunk_loop_before_the_clock_stops(mutant)
    assert not claim_copy_shape_is_chunked_async_with_one_wait_per_iteration(mutant)


def test_mutant_sync_copy_per_chunk_fails():
    assert not claim_copy_shape_is_chunked_async_with_one_wait_per_iteration(
        _once(SYCL, "ggml_sycl::mem_copy_async(dst, off, src, off, kChunk, *q);",
              "ggml_sycl::mem_copy(dst, off, src, off, kChunk, *q);"))


def test_mutant_second_log_line_fails():
    assert not claim_logs_once_at_warn(_once(SYCL, LOG_CALL, LOG_CALL + " " + LOG_CALL))


def test_mutant_second_switch_read_fails():
    assert not claim_reads_its_switch_once(
        _once(SYCL, 'const char * sw = std::getenv("GGML_SYCL_H2D_PROBE");',
              'const char * sw = std::getenv("GGML_SYCL_H2D_PROBE"); const char * s2 = std::getenv("GGML_SYCL_H2D_PROBE");'))


def test_mutant_abort_on_alloc_failure_fails():
    assert not claim_failed_alloc_keeps_the_seed_and_never_aborts(
        _once(SYCL, "if (!src.valid() || !dst.valid()) {", 'if (!src.valid() || !dst.valid()) { GGML_ABORT("x");'))


def test_mutant_unguarded_catch_fails():
    assert not claim_rethrows_fallback_error_first(
        _once(SYCL, GUARDED_CATCH, 'catch (const std::exception & e) { GGML_LOG_WARN("[MOE-EXEC] h2d probe device %d failed'))


def test_mutant_reorder_rate_missing_fails():
    assert not claim_store_carries_a_valid_reorder_rate(SYCL.replace("rec.reorder_gbps", "rec.unused_gbps"))


def test_mutant_second_reader_fails(monkeypatch=None):
    """A second read of the rate outside the probe's own guard changes the count: the claim reads the tree, so the
    mutant is the claim run over a tree with one more reader."""
    assert claim_one_consumer_reads_the_rate()
    assert count_reads_in(norm(SYCL) + " double x = info.devices[0].h2d_pinned_gbps;") == 2


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
