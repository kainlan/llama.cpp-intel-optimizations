"""Source contract for llama.cpp-zujy (plan Task 3 of llama.cpp-36j0): each device probes its PINNED host-to-device
rate once and keeps it, so the MoE executor's cost model divides by the card's real link rate.

Host-only: reads sources, builds nothing, loads no model and touches no device.

Why: the cost model's h2d_gbps is the real rate of each card from pinned host memory (26.5-27.1 GB/s measured, against
10.5-19 GB/s from pageable memory), and the peer-bounce probe stages through a pageable std::vector, so it cannot be
reused. The probe this gate pins:
  - ggml_sycl_measure_h2d_pinned takes its host source from ggml_sycl_host_expert_alloc (owner-first, role WEIGHT,
    pinned pool: the same provenance a host-planned expert has, which is what lets mem_copy_submit take its direct
    branch), and its device buffer from ggml_sycl::unified_allocate (role STAGING, category STAGING, cohort
    h2d_probe_dst) in the RUNTIME zone with no spill (must_device, prefer_vram_zone RUNTIME, forbid_vram_zone_spill):
    clauses (b) and (d) of scripts/check-sycl-alloc-zone-contract.py read those literals;
  - no raw allocation and no pageable staging vector;
  - one untimed warm-up copy (ggml_sycl::mem_copy, which is synchronous), then 8 timed iterations of a 64 MiB buffer
    moved in 4 MiB chunks with ggml_sycl::mem_copy_async at an offset, one q->wait() per iteration and never one per
    chunk, timed on the host with steady_clock;
  - one WARN line per device, "[MOE-EXEC] h2d probe device %d: %.1f GB/s (%s)", whose source word is "probed" (with the
    iteration spread appended: "probed; iterations 24.1-26.2"), "tuning cache" or "seed, GGML_SYCL_H2D_PROBE=0";
  - llama.cpp-xr4i (c-vbf4, the bimodal 9.x vs 24-26 GB/s readings): each of the 8 iterations is timed on its own, the
    rate persisted is the max over iterations 2..8 (iteration 1 is a second warm-up) through
    moe_exec_h2d_rate_from_iterations in moe-exec-cost.hpp, the spread rides in the probed line only, and a max under
    half the seed is RETRY_NO_RATE (kept seed, retried at the next load) instead of a stored rate;
  - GGML_SYCL_H2D_PROBE is read once (1 forces a re-probe, 0 skips it and keeps the seed); a stored record skips the
    probe; a probe that cannot allocate keeps the seed and never aborts;
  - a handler that can swallow ggml_sycl_fallback_error is preceded by the sanctioned rethrow (clause (g));
  - sycl_device_info carries h2d_pinned_gbps and h2d_probed, and exactly two places read h2d_pinned_gbps: the probe's
    own once-guard (a device that holds a rate is not probed again) and moe_stream_phase::weigh, which hands it to
    moe_stream_rates_assemble when h2d_probed (llama.cpp-kp1l). The assembly prices h2d from the tuning-cache record,
    else that probed rate, else the seed, with the environment override applied last, so a run with the tuning cache
    disabled still prices with the measured rate;
  - the probe is called at the end of ggml_sycl_model_loading_effects' outermost exit, right after
    ggml_sycl_preload_model_weights() (the host zones are configured and the experts placed), never from
    ggml_sycl_init (it must not create caches) nor from model_load_begin (the zones are not configured yet), and
    through ggml_sycl_info_mutable(), defined once, rather than a const_cast of the const accessor;
  - the copies and the device buffer use cache->get_queue(), never the dpct device's default_queue(): the direct branch
    needs the copy queue to share a context with the cache's own queue (measured: the default queue read 4-10 GB/s,
    staged, against 20-21 GB/s direct), and the staged baseline is read after the untimed warm-up;
  - the call sits inside the outermost-exit `if (outer)` block after the preload; a device that holds a rate is skipped
    by the loop's first statement; a stored record (unless forced) and then the env skip come before the cache and
    zone check; and a refused tuning-cache store is logged "probed, not stored", never "probed";
  - a device whose cache has no configured host zones is not probed (the host source would not be a trusted direct-copy
    source), and a copy the staged-copy census records as staged is never stored or kept: the seed stands, h2d_probed
    stays false, and both outcomes leave the device unresolved so the next load retries;
  - the probe never writes a reorder rate into the record (llama.cpp-kp1l): the record keeps the looked-up rate, 0 when
    there was none, and a reorder rate of 0 is a valid record that means unmeasured, so the seed is not persisted as a
    measurement.

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
COST_REL = "ggml/src/ggml-sycl/moe-exec-cost.hpp"
SYCL_TREE_DIR = "ggml/src/ggml-sycl"

SYCL = read(SYCL_REL)
COMMON = read(COMMON_REL)
COST = read(COST_REL)

PLAN_HPP_REL = "ggml/src/ggml-sycl/moe-stream-plan.hpp"
PLAN_CPP_REL = "ggml/src/ggml-sycl/moe-stream-plan.cpp"
IO_REL = "ggml/src/ggml-sycl/tuning-cache-io.hpp"
ENV_DOC_REL = "docs/backend/sycl-env-vars.md"
PLAN_HPP = read(PLAN_HPP_REL)
PLAN_CPP = read(PLAN_CPP_REL)
IO_HPP = read(IO_REL)
ENV_DOC = read(ENV_DOC_REL)
ASSEMBLE_SIG = "static inline moe_exec_rates moe_stream_rates_assemble("
WEIGH_SIG = "moe_exec_decision moe_stream_phase::weigh("

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

def claim_allocates_through_the_sanctioned_surfaces_with_literal_tiers(src: str) -> bool:
    fn = probe_body(src)
    return bool(fn) and all([
        fn.count("ggml_sycl::unified_allocate(") == 1,
        fn.count("ggml_sycl_host_expert_alloc(dev, k_bytes)") == 1,
        "role = ggml_sycl::alloc_role::STAGING" in fn,
        "category = ggml_sycl::runtime_category::STAGING" in fn,
        '"h2d_probe_dst"' in fn,
        "constraints.must_device = true" in fn,
        "constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME" in fn,
        "constraints.forbid_vram_zone_spill = true" in fn,
        # the host source's tier flags live in the one helper, not a second request here
        "must_host_pinned" not in fn,
        "use_pinned_pool" not in fn,
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
    sizes = all(s in fn for s in ("k_bytes = 64ull << 20;", "k_chunk = 4ull << 20;", "k_iters = 8;"))
    # the one wait sits in the iteration loop, after the chunk loop closes and before the next iteration
    one_wait = fn.count("q->wait()") == 1 and ordered(fn, "for (int it = 0; it < k_iters; ++it)", "mem_copy_async(",
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
    m = re.search(r"for \(size_t off = 0; off < k_bytes; off \+= k_chunk\) \{.*?\} q->wait\(\);", fn)
    return bool(m) and "q->wait()" not in m.group(0)[:-len("q->wait();")]


def claim_logs_once_at_warn(src: str) -> bool:
    n = norm(src)
    return n.count(LOG_FORMAT) == 1 and all(
        s in probe_body(src) for s in ('"probed"', '"tuning cache"', '"seed, GGML_SYCL_H2D_PROBE=0"',
                                       '"seed, probe copy was staged"', '"seed, host zones not configured"')
    ) and "GGML_LOG_WARN(" + LOG_FORMAT in n


def claim_reads_its_switch_once(src: str) -> bool:
    return norm(src).count('getenv("GGML_SYCL_H2D_PROBE")') == 1


GUARD = "if (ggml_sycl_h2d_pinned_resolved(info, dev)) { continue; }"
LOOKUP = "ggml_backend_sycl_moe_exec_rates_lookup("
CACHE_HIT = 'if (have && !force) { report(dev, rec.h2d_gbps, H2D_PROBE_OUTCOME_FROM_CACHE, "tuning cache"); continue; }'
SKIP = 'if (skip) { report(dev, seed.h2d_gbps, H2D_PROBE_OUTCOME_SEED_KEPT, "seed, GGML_SYCL_H2D_PROBE=0"); continue; }'


def claim_a_device_holding_a_rate_is_not_probed_again(src: str) -> bool:
    """The once-guard is the loop's first statement, ahead of the tuning-cache lookup: a device that holds a rate is
    skipped whole, which is what lets every outermost load call the probe."""
    fn = probe_body(src)
    helper = body(norm(src), RESOLVED_SIG)
    return bool(fn) and fn.count(GUARD) == 1 and ordered(fn, "for (int dev = 0;", GUARD, LOOKUP) and \
        "return info.devices[dev].h2d_pinned_gbps > 0.0;" in helper


def claim_switch_semantics(src: str) -> bool:
    fn = probe_body(src)
    return bool(fn) and all([
        # stored record first (unless forced), then the env skip, then the cache and zone check
        ordered(fn, LOOKUP, CACHE_HIT, SKIP, "get_unified_cache_for_device(dev)", "host_zones_configured()"),
        fn.count(CACHE_HIT) == 1,
        fn.count(SKIP) == 1,
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


EFFECTS_SIG = "static void ggml_sycl_model_loading_effects(bool loading, bool outer)"
LOAD_BEGIN_SIG = "ggml_sycl_lifecycle_result ggml_backend_sycl_model_load_begin(ggml_sycl_load_txn * txn)"
INIT_SIG = "static ggml_sycl_device_info ggml_sycl_init()"
CALL = "ggml_sycl_measure_h2d_pinned(ggml_sycl_info_mutable())"


OUTER_EXIT = "if (outer) { g_sycl_in_model_load.store(false, std::memory_order_release);"


def in_outer_exit_block(effects: str) -> bool:
    """The preload and the probe call both sit inside the outermost-exit `if (outer) { ... }` block, in that order, so a
    nested or non-loading exit never probes."""
    at = effects.find(OUTER_EXIT)
    if at < 0:
        return False
    block = body(effects[at:], "if (outer)")
    return bool(block) and block.count(CALL) == 1 and ordered(block, "ggml_sycl_preload_model_weights();", CALL)


def claim_probe_runs_after_preload_through_the_mutable_accessor(src: str) -> bool:
    n = norm(src)
    effects = body(n, EFFECTS_SIG)
    begin = body(n, LOAD_BEGIN_SIG)
    init = body(n, INIT_SIG)
    return bool(effects) and bool(begin) and bool(init) and all([
        n.count("ggml_sycl_measure_h2d_pinned(") == 3,  # forward declaration, definition, and the one call
        effects.count(CALL) == 1,
        ordered(effects, "ggml_sycl_preload_model_weights();", CALL),
        in_outer_exit_block(effects),
        "ggml_sycl_measure_h2d_pinned" not in begin,
        "ggml_sycl_measure_h2d_pinned" not in init,
        "const_cast<ggml_sycl_device_info" not in n,
        n.count("static ggml_sycl_device_info & ggml_sycl_info_mutable() {") == 1,
        n.count("ggml_sycl_info_mutable()") == 4,  # declaration, definition, ggml_sycl_info()'s call, the probe's call
    ])


RETRY_ONCE = ("if (outcome >= H2D_PROBE_OUTCOME_RETRY_ZONES) { const uint32_t bit = 1u << outcome; "
              "if ((d.h2d_retry_logged & bit) != 0) { return; } d.h2d_retry_logged |= bit; }")
FAILED_ONE_LINE = ('std::snprintf(what, sizeof(what), "seed, probe failed: %s", e.what()); '
                   "report(dev, seed.h2d_gbps, H2D_PROBE_OUTCOME_RETRY_FAILED, what);")


def claim_each_unresolved_reason_logs_once_per_device(src: str) -> bool:
    """A device left unresolved (host zones not configured yet) is probed again, silently, at every later load: each
    RETRY outcome logs once per device and process, through the per-device bit set in h2d_retry_logged, before the one
    WARN line. An exception logs through that same line, its text folded into the source word, so the probe holds
    exactly one logging call and a failed device prints one line, not two."""
    fn = probe_body(src)
    return bool(fn) and fn.count(RETRY_ONCE) == 1 and fn.count(FAILED_ONE_LINE) == 1 and fn.count("GGML_LOG_") == 1 \
        and ordered(fn, RETRY_ONCE, "GGML_LOG_WARN(" + LOG_FORMAT)


def claim_outcome_enum_keeps_the_resolved_values_first(src: str) -> bool:
    """`outcome >= H2D_PROBE_OUTCOME_RETRY_ZONES` means unresolved and the logged bits are one per RETRY outcome in a
    uint32_t, so the resolved names precede RETRY_ZONES, which opens the RETRY values, and a static_assert says so."""
    n = norm(src)
    m = re.search(r"enum h2d_probe_outcome \{(.*?)\};", n)
    if not m:
        return False
    names = [x.strip() for x in m.group(1).split(",") if x.strip()]
    first_retry = names.index("H2D_PROBE_OUTCOME_RETRY_ZONES") if "H2D_PROBE_OUTCOME_RETRY_ZONES" in names else -1
    ok_order = (first_retry == 3 and names[:3] == ["H2D_PROBE_OUTCOME_MEASURED", "H2D_PROBE_OUTCOME_FROM_CACHE",
                                                  "H2D_PROBE_OUTCOME_SEED_KEPT"]
                and all(x.startswith("H2D_PROBE_OUTCOME_RETRY_") for x in names[3:]) and len(names) <= 32)
    sa = re.search(r"static_assert\((.*?), \"h2d_probe_outcome", n)
    return ok_order and bool(sa) and all(c in sa.group(1) for c in (
        "H2D_PROBE_OUTCOME_MEASURED < H2D_PROBE_OUTCOME_RETRY_ZONES",
        "H2D_PROBE_OUTCOME_FROM_CACHE < H2D_PROBE_OUTCOME_RETRY_ZONES",
        "H2D_PROBE_OUTCOME_SEED_KEPT < H2D_PROBE_OUTCOME_RETRY_ZONES",
        "H2D_PROBE_OUTCOME_RETRY_ZONES < H2D_PROBE_OUTCOME_RETRY_ALLOC",
        "H2D_PROBE_OUTCOME_RETRY_ZONES < H2D_PROBE_OUTCOME_RETRY_STAGED",
        "H2D_PROBE_OUTCOME_RETRY_ZONES < H2D_PROBE_OUTCOME_RETRY_NO_RATE",
        "H2D_PROBE_OUTCOME_RETRY_ZONES < H2D_PROBE_OUTCOME_RETRY_FAILED",
        "H2D_PROBE_OUTCOME_RETRY_FAILED < 32"))


def claim_a_refused_store_is_not_reported_as_persisted(src: str) -> bool:
    fn = probe_body(src)
    return bool(fn) and all([
        fn.count("const bool stored = ggml_backend_sycl_moe_exec_rates_store(dev, &rec);") == 1,
        fn.count(MEASURED_REPORT) == 1,
    ])


def claim_zones_and_staged_copies_never_reach_the_store(src: str) -> bool:
    fn = probe_body(src)
    staged_report = ('report(dev, seed.h2d_gbps, H2D_PROBE_OUTCOME_RETRY_STAGED, "seed, probe copy was staged"); '
                     "continue;")
    zones_report = ('report(dev, seed.h2d_gbps, H2D_PROBE_OUTCOME_RETRY_ZONES, "seed, host zones not configured"); '
                    "continue;")
    census = "ggml_sycl::mem_copy_staged_wait_census_this_thread().staged_h2d"
    return bool(fn) and all([
        fn.count(census) == 2,
        staged_report in fn,
        zones_report in fn,
        ordered(fn, "cache->host_zones_configured()", zones_report, "ggml_sycl_host_expert_alloc(dev, k_bytes)"),
        # the baseline is read after the untimed warm-up, so the detection scores the timed copies only
        ordered(fn, "ggml_sycl::mem_copy(dst, src, k_bytes, *q);", "staged_before = " + census,
                "mem_copy_async(", census + " != staged_before", staged_report, "moe_exec_rates_store("),
        fn.count("moe_exec_rates_store(") == 1,
    ])


def claim_copies_run_on_the_cache_queue(src: str) -> bool:
    """mem_copy_submit takes its direct branch only when the copy queue shares a context with the cache's own queue
    (queue_for_device_or_fallback returns cache->get_queue()), so the probe copies, and allocates its device buffer,
    on that queue: never the dpct device's default queue, which is a different context and is always refused."""
    fn = probe_body(src)
    return bool(fn) and all([
        fn.count("sycl::queue * q = &cache->get_queue();") == 1,
        "default_queue" not in fn,
        "dpct::dev_mgr" not in fn,
        "dst_req.queue = q;" in fn,
        fn.count("*q)") + fn.count("*q,") >= 2,
    ])


HELPER_CALL = "ggml_sycl::moe_exec_h2d_rate_from_iterations(iter_gbps, k_iters, seed.h2d_gbps, &lo, &hi)"
ITER_LOOP = "for (int it = 0; it < k_iters; ++it)"
SPREAD_WORD = '"%s; iterations %.1f-%.1f"'
MEASURED_REPORT = ('report(dev, gbps, H2D_PROBE_OUTCOME_MEASURED, stored ? "probed" : "probed, not stored", '
                   "spread);")


def claim_each_iteration_is_timed_on_its_own(src: str) -> bool:
    """llama.cpp-xr4i: the clock starts and stops inside the iteration loop, once per iteration, and each iteration's
    rate lands in its own slot, so one slow window is a reading rather than a third of a mean."""
    fn = probe_body(src)
    loop = body(fn, ITER_LOOP) if fn else ""
    return bool(loop) and all([
        fn.count("std::chrono::steady_clock::now()") == 2,
        loop.count("std::chrono::steady_clock::now()") == 2,
        "iter_gbps[it] = " in loop,
        "double iter_gbps[k_iters];" in fn,
        ordered(loop, "steady_clock::now()", "mem_copy_async(", "q->wait()", "steady_clock::now()", "iter_gbps[it] = "),
    ])


def claim_the_rate_is_the_max_over_iterations_two_to_eight(src: str, cost: str) -> bool:
    """The probe persists what moe_exec_h2d_rate_from_iterations returns for all k_iters readings, and that helper
    skips iteration 1 (the second warm-up) and keeps the max of the rest."""
    fn = probe_body(src)
    helper = body(norm(cost), "static inline double moe_exec_h2d_rate_from_iterations(")
    return bool(fn) and bool(helper) and all([
        fn.count("moe_exec_h2d_rate_from_iterations(") == 1,
        fn.count("const double gbps = " + HELPER_CALL + ";") == 1,
        fn.count("rec.h2d_gbps = gbps;") == 1,
        ordered(fn, HELPER_CALL, "rec.h2d_gbps = gbps;"),
        "for (int i = 1; i < n; ++i)" in helper,
        "if (g > best) { best = g; }" in helper,
        "if (g < low) { low = g; }" in helper,
        "bool any = false;" in helper and "any = true;" in helper,
        "return best;" in helper,
    ])


def claim_the_spread_rides_in_the_probed_line_only(src: str) -> bool:
    """The one WARN call keeps its one format; the iteration spread is folded into the source word by the report
    lambda, and only the MEASURED report passes it."""
    fn = probe_body(src)
    return bool(fn) and all([
        fn.count(SPREAD_WORD) == 1,
        fn.count(MEASURED_REPORT) == 1,
        fn.count(", spread)") == 1,
        fn.count("const double spread[2] = { lo, hi };") == 1,
        ordered(fn, "const double * spread = nullptr", SPREAD_WORD, "GGML_LOG_WARN(" + LOG_FORMAT),
    ])


def claim_a_max_under_half_the_seed_is_retried_not_stored(src: str, cost: str) -> bool:
    """A max under MOE_EXEC_H2D_MIN_SEED_FRACTION (0.5) of the seed is the helper's zero, and a zero is
    RETRY_NO_RATE, reported before the store and never kept."""
    fn = probe_body(src)
    n = norm(cost)
    return bool(fn) and all([
        "static constexpr double MOE_EXEC_H2D_MIN_SEED_FRACTION = 0.5;" in n,
        "if (!any || !(best >= seed * MOE_EXEC_H2D_MIN_SEED_FRACTION)) { return 0.0; }" in n,
        "if (gbps <= 0.0) {" in fn,
        ordered(fn, HELPER_CALL, "if (gbps <= 0.0) {", "H2D_PROBE_OUTCOME_RETRY_NO_RATE",
                "continue;", "rec.h2d_gbps = gbps;", "moe_exec_rates_store("),
    ])


def claim_device_info_fields(common: str) -> bool:
    n = norm(common)
    return all(f in n for f in ("double h2d_pinned_gbps = 0.0;", "bool h2d_probed = false;",
                                "uint32_t h2d_retry_logged = 0;"))


def read_offsets(text: str) -> list:
    """Offsets of the occurrences of h2d_pinned_gbps that are reads: a write is an occurrence followed by `=`."""
    return [m.start() for m in re.finditer(r"h2d_pinned_gbps", text)
            if not re.match(r"\s*=(?!=)", text[m.end():m.end() + 4])]


def count_reads_in(text: str) -> int:
    return len(read_offsets(text))


def claim_two_consumers_read_the_rate(overrides: dict = None) -> bool:
    """h2d_pinned_gbps is read in exactly two places across the backend's sources, both in ggml-sycl.cpp and both under
    g_sycl_h2d_pinned_mutex: the probe's once-guard helper and the accessor the cost model's phase calls
    (llama.cpp-kp1l). A write is an occurrence followed by `=` (the field's default, the report lambda's store).
    `overrides` maps a relative path to replacement text, which is how a mutant tree is checked."""
    reads = []
    for p in sorted((ROOT / SYCL_TREE_DIR).rglob("*")):
        if not p.is_file() or p.suffix not in (".c", ".cpp", ".h", ".hpp"):
            continue
        rel = p.relative_to(ROOT).as_posix()
        text = norm(overrides[rel] if overrides and rel in overrides else p.read_text(errors="replace"))
        for at in read_offsets(text):
            reads.append((rel, at, text))
    if len(reads) != 2 or any(r[0] != SYCL_REL for r in reads):
        return False
    text = reads[0][2]
    spans = []
    for sig in (RESOLVED_SIG, ACCESSOR_SIG):
        fn = body(text, sig)
        if not fn:
            return False
        spans.append((text.find(fn), text.find(fn) + len(fn)))
    return all(any(lo <= at < hi for lo, hi in spans) for _, at, _ in reads) and \
        len({next(i for i, (lo, hi) in enumerate(spans) if lo <= at < hi) for _, at, _ in reads}) == 2



# ---- llama.cpp-kp1l (finding I4 and M8 of the final review of llama.cpp-36j0) ----

def claim_assembly_takes_h2d_from_record_then_probe_then_seed(hpp: str) -> bool:
    """moe_stream_rates_assemble prices the h2d term from, in order: the device's tuning-cache record, the device's
    probed rate (a parameter, so the function stays device-free), the seed it starts from; the environment override
    is applied after all of them. A cache-disabled run therefore still prices with the measured rate."""
    fn = body(norm(hpp), ASSEMBLE_SIG)
    sig_at = norm(hpp).find(ASSEMBLE_SIG)
    sig = norm(hpp)[sig_at:sig_at + 400] if sig_at >= 0 else ""
    return bool(fn) and all([
        "double probed_h2d_gbps" in sig,
        # the chain is one statement straight after the seed: the record, else the probed rate, else the seed stands
        "moe_exec_rates r = moe_exec_rates_seed(host_mix); "
        "if (record != nullptr && detail::moe_stream_rate_measured(record->h2d_gbps)) { "
        "r.h2d_gbps = record->h2d_gbps; } "
        "else if (detail::moe_stream_rate_measured(probed_h2d_gbps)) { "
        "r.h2d_gbps = probed_h2d_gbps; } if (record != nullptr) {"
        in fn,
        ordered(fn, "probed_h2d_gbps", "moe_exec_rates_apply_env(r, getenv_fn);"),
    ])


def claim_weigh_passes_the_devices_probed_rate(cpp: str) -> bool:
    """The phase reads the device's probed rate where it weighs (live: the probe may resolve after the first prompt)
    through the mutex-guarded accessor, and hands it to the assembly."""
    fn = body(norm(cpp), WEIGH_SIG)
    return bool(fn) and ordered(
        fn, "const double probed_h2d_gbps = ggml_sycl_h2d_probed_gbps(device_);",
        "moe_stream_rates_assemble(li.host_mix, record_valid_ ? &record_ : nullptr, probed_h2d_gbps, "
        "moe_stream_getenv)")


def claim_probe_does_not_store_the_reorder_seed(src: str) -> bool:
    """The probe measures the H2D rate only. It never writes a reorder rate into the record: it stores what the
    looked-up record held (0 = unmeasured), so the assembly's seed is not frozen into the file as a measurement."""
    fn = probe_body(src)
    return bool(fn) and all([
        not re.search(r"reorder_gbps\s*=(?!=)", fn),
        "seed.reorder_gbps" not in fn,
        "moe_exec_rates_store(dev, &rec)" in fn,
    ])


def claim_record_accepts_an_unmeasured_reorder_rate(io: str) -> bool:
    """A record's reorder rate of 0 means unmeasured and is valid; a negative or non-finite one is still refused."""
    n = norm(io)
    return ("return std::isfinite(h2d_gbps) && h2d_gbps > 0.0 && std::isfinite(reorder_gbps) && reorder_gbps >= 0.0;"
            in n)


def _doc_row(doc: str, name: str) -> str:
    for line in doc.splitlines():
        if line.startswith("| `" + name + "="):
            return line
    return ""


def claim_catalog_rows_state_the_source_order(doc: str) -> bool:
    row = _doc_row(doc, "GGML_SYCL_MOE_EXEC_H2D_GBPS")
    probe = _doc_row(doc, "GGML_SYCL_H2D_PROBE")
    return bool(row) and bool(probe) and ordered(
        row, "this environment override", "the device's tuning-cache record", "the device's probed rate",
        "the 20 GB/s seed") and "even when the tuning cache" \
        in row and "not stored" in probe and \
        "With no stored record, `GGML_SYCL_H2D_PROBE=0` skips the probe and the seed stands" in row and \
        "leaves no probed rate" not in row


PINNED_MUTEX = "static std::mutex g_sycl_h2d_pinned_mutex;"
RESOLVED_SIG = "static bool ggml_sycl_h2d_pinned_resolved(const ggml_sycl_device_info & info, int dev)"
ACCESSOR_SIG = "double ggml_sycl_h2d_probed_gbps(int device)"


def claim_probed_rate_is_published_under_one_mutex(sycl: str, common: str, cpp: str) -> bool:
    """A load's outermost exit runs the probe on the loading thread with no lock a computing thread takes, and
    moe_stream_phase::weigh runs on any thread that computes, so the rate and its flag are written and read under one
    leaf mutex: the probe's guard read and its store (the report lambda), and the accessor weigh calls. weigh itself
    touches neither field, and the accessor is declared where weigh sees it."""
    n = norm(sycl)
    probe = probe_body(sycl)
    resolved = body(n, RESOLVED_SIG)
    accessor = body(n, ACCESSOR_SIG)
    weigh = body(norm(cpp), WEIGH_SIG)
    return bool(probe) and bool(resolved) and bool(accessor) and bool(weigh) and all([
        n.count(PINNED_MUTEX) == 1,
        "std::lock_guard<std::mutex> lock(g_sycl_h2d_pinned_mutex);" in resolved,
        ordered(resolved, "lock_guard", "h2d_pinned_gbps"),
        ordered(accessor, "lock_guard", "h2d_probed", "h2d_pinned_gbps"),
        "if (ggml_sycl_h2d_pinned_resolved(info, dev)) { continue; }" in probe,
        # the store: both fields written inside one locked scope of the report lambda
        ordered(probe, "std::lock_guard<std::mutex> lock(g_sycl_h2d_pinned_mutex);", "d.h2d_pinned_gbps = gbps;",
                "d.h2d_probed = outcome != H2D_PROBE_OUTCOME_SEED_KEPT;"),
        "double ggml_sycl_h2d_probed_gbps(int device);" in norm(common),
        "ggml_sycl_h2d_probed_gbps(device_)" in weigh,
        "h2d_pinned_gbps" not in weigh and not re.search(r"h2d_probed(?!_gbps)", weigh),
    ])


# ---- tests ----

def test_probe_allocates_through_unified_allocate_with_literal_tiers():
    assert claim_allocates_through_the_sanctioned_surfaces_with_literal_tiers(SYCL)


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


def test_probe_runs_after_preload_through_the_mutable_accessor():
    assert claim_probe_runs_after_preload_through_the_mutable_accessor(SYCL)


def test_zones_and_staged_copies_never_reach_the_store():
    assert claim_zones_and_staged_copies_never_reach_the_store(SYCL)


def test_a_device_holding_a_rate_is_not_probed_again():
    assert claim_a_device_holding_a_rate_is_not_probed_again(SYCL)


def test_each_unresolved_reason_logs_once_per_device():
    assert claim_each_unresolved_reason_logs_once_per_device(SYCL)


def test_a_refused_store_is_not_reported_as_persisted():
    assert claim_a_refused_store_is_not_reported_as_persisted(SYCL)


def test_copies_run_on_the_cache_queue():
    assert claim_copies_run_on_the_cache_queue(SYCL)


def test_device_info_carries_the_rate_and_two_consumers_read_it():
    assert claim_device_info_fields(COMMON)
    assert claim_two_consumers_read_the_rate()


def test_each_iteration_is_timed_on_its_own():
    assert claim_each_iteration_is_timed_on_its_own(SYCL)


def test_the_rate_is_the_max_over_iterations_two_to_eight():
    assert claim_the_rate_is_the_max_over_iterations_two_to_eight(SYCL, COST)


def test_the_spread_rides_in_the_probed_line_only():
    assert claim_the_spread_rides_in_the_probed_line_only(SYCL)


def test_a_max_under_half_the_seed_is_retried_not_stored():
    assert claim_a_max_under_half_the_seed_is_retried_not_stored(SYCL, COST)


def test_assembly_takes_h2d_from_record_then_probe_then_seed():
    assert claim_assembly_takes_h2d_from_record_then_probe_then_seed(PLAN_HPP)
    assert claim_weigh_passes_the_devices_probed_rate(PLAN_CPP)


def test_probe_does_not_store_the_reorder_seed_and_the_record_may_say_unmeasured():
    assert claim_probe_does_not_store_the_reorder_seed(SYCL)
    assert claim_record_accepts_an_unmeasured_reorder_rate(IO_HPP)


def test_catalog_rows_state_the_source_order():
    assert claim_catalog_rows_state_the_source_order(ENV_DOC)


def test_probed_rate_is_published_under_one_mutex():
    assert claim_probed_rate_is_published_under_one_mutex(SYCL, COMMON, PLAN_CPP)


# ---- mutants: each claim must go red when its text is broken ----

LOG_CALL = 'GGML_LOG_WARN("[MOE-EXEC] h2d probe device %d: %.1f GB/s (%s)\\n", dev, gbps, source);'
GUARDED_CATCH = ("catch (const ggml_sycl_fallback_error &) { throw; } catch (const std::exception & e) { "
                 "char what[192];")


def test_mutant_raw_malloc_fails():
    assert not claim_allocates_through_the_sanctioned_surfaces_with_literal_tiers(
        _once(SYCL, 'dst_req.intent.cohort_id = "h2d_probe_dst";',
              'dst_req.intent.cohort_id = "h2d_probe_dst"; void * p = sycl::malloc_host(1, *q);'))


def test_mutant_unzoned_host_source_fails():
    """The host source built by a bare request in the probe instead of the helper: the provenance the direct branch
    trusts is then a second, unpinned spelling."""
    assert not claim_allocates_through_the_sanctioned_surfaces_with_literal_tiers(
        _once(SYCL, "ggml_sycl::mem_handle src = ggml_sycl_host_expert_alloc(dev, k_bytes);",
              "ggml_sycl::mem_handle src = ggml_sycl::unified_allocate(dst_req);"))


def test_mutant_device_spill_allowed_fails():
    assert not claim_allocates_through_the_sanctioned_surfaces_with_literal_tiers(
        _once(SYCL, "dst_req.intent.constraints.forbid_vram_zone_spill = true;",
              "dst_req.intent.constraints.forbid_vram_zone_spill = false;"))


def test_mutant_wait_per_chunk_fails():
    mutant = _once(SYCL, "ggml_sycl::mem_copy_async(dst, off, src, off, k_chunk, *q);",
                   "ggml_sycl::mem_copy_async(dst, off, src, off, k_chunk, *q); q->wait();")
    assert not claim_wait_closes_the_chunk_loop_before_the_clock_stops(mutant)
    assert not claim_copy_shape_is_chunked_async_with_one_wait_per_iteration(mutant)


def test_mutant_sync_copy_per_chunk_fails():
    assert not claim_copy_shape_is_chunked_async_with_one_wait_per_iteration(
        _once(SYCL, "ggml_sycl::mem_copy_async(dst, off, src, off, k_chunk, *q);",
              "ggml_sycl::mem_copy(dst, off, src, off, k_chunk, *q);"))


def test_mutant_second_log_line_fails():
    assert not claim_logs_once_at_warn(_once(SYCL, LOG_CALL, LOG_CALL + " " + LOG_CALL))


def test_mutant_second_switch_read_fails():
    assert not claim_reads_its_switch_once(
        _once(SYCL, 'const char * sw = std::getenv("GGML_SYCL_H2D_PROBE");',
              'const char * sw = std::getenv("GGML_SYCL_H2D_PROBE"); '
              'const char * s2 = std::getenv("GGML_SYCL_H2D_PROBE");'))


def test_mutant_abort_on_alloc_failure_fails():
    assert not claim_failed_alloc_keeps_the_seed_and_never_aborts(
        _once(SYCL, "if (!src.valid() || !dst.valid()) {", 'if (!src.valid() || !dst.valid()) { GGML_ABORT("x");'))


def test_mutant_unguarded_catch_fails():
    assert not claim_rethrows_fallback_error_first(
        _once(SYCL, GUARDED_CATCH,
              "catch (const std::exception & e) { char what[192];"))


def test_mutant_probe_called_from_init_fails():
    assert not claim_probe_runs_after_preload_through_the_mutable_accessor(
        _once(SYCL, "ggml_sycl_init_peer_link_defaults(info);",
              "ggml_sycl_init_peer_link_defaults(info); ggml_sycl_measure_h2d_pinned(info);"))


def test_mutant_probe_called_from_load_begin_fails():
    assert not claim_probe_runs_after_preload_through_the_mutable_accessor(
        _once(SYCL, "ggml_sycl_model_loading_effects(true, true); registry->bind_candidate(result.txn);",
              "ggml_sycl_model_loading_effects(true, true); " + CALL + "; registry->bind_candidate(result.txn);"))


def test_mutant_const_cast_fails():
    assert not claim_probe_runs_after_preload_through_the_mutable_accessor(
        _once(SYCL, CALL + ";",
              "ggml_sycl_measure_h2d_pinned(const_cast<ggml_sycl_device_info &>(ggml_sycl_info()));"))


def test_mutant_probe_before_preload_fails():
    mutant = _once(SYCL, CALL + ";", "")
    mutant = _once(mutant, "ggml_sycl_preload_model_weights(); }", CALL + "; ggml_sycl_preload_model_weights(); }")
    assert not claim_probe_runs_after_preload_through_the_mutable_accessor(mutant)


def test_mutant_second_mutable_accessor_definition_fails():
    assert not claim_probe_runs_after_preload_through_the_mutable_accessor(
        _once(SYCL, "ggml_sycl_device_info & info = ggml_sycl_info_mutable();",
              "ggml_sycl_device_info & info = ggml_sycl_info_mutable(); ggml_sycl_info_mutable();"))


def test_mutant_staged_check_removed_fails():
    assert not claim_zones_and_staged_copies_never_reach_the_store(
        _once(SYCL, "mem_copy_staged_wait_census_this_thread().staged_h2d != staged_before",
              "false"))


def test_mutant_call_outside_the_outer_exit_block_fails():
    # close the outer-exit block right after its first statement and run the rest, probe included, unconditionally
    mutant = _once(SYCL, OUTER_EXIT, OUTER_EXIT + " } {")
    assert not claim_probe_runs_after_preload_through_the_mutable_accessor(mutant)


def test_mutant_once_guard_inverted_fails():
    assert not claim_a_device_holding_a_rate_is_not_probed_again(
        _once(SYCL, "return info.devices[dev].h2d_pinned_gbps > 0.0;",
              "return info.devices[dev].h2d_pinned_gbps < 0.0;"))


def test_mutant_once_guard_without_continue_fails():
    assert not claim_a_device_holding_a_rate_is_not_probed_again(
        _once(SYCL, "if (ggml_sycl_h2d_pinned_resolved(info, dev)) { continue; }",
              "if (ggml_sycl_h2d_pinned_resolved(info, dev)) { }"))


def test_mutant_env_skip_disabled_fails():
    assert not claim_switch_semantics(_once(SYCL, "if (skip) {", "if (false) {"))


def test_mutant_stored_record_ignored_fails():
    assert not claim_switch_semantics(_once(SYCL, "if (have && !force) {", "if (false) {"))


def test_mutant_store_result_ignored_fails():
    assert not claim_a_refused_store_is_not_reported_as_persisted(
        _once(SYCL, "const bool stored = ggml_backend_sycl_moe_exec_rates_store(dev, &rec);",
              "ggml_backend_sycl_moe_exec_rates_store(dev, &rec); const bool stored = true;"))


def test_mutant_retry_reason_logs_every_load_fails():
    assert not claim_each_unresolved_reason_logs_once_per_device(
        _once(SYCL, "if ((d.h2d_retry_logged & bit) != 0) {", "if (false) {"))


def test_mutant_failed_logs_a_second_line_fails():
    assert not claim_each_unresolved_reason_logs_once_per_device(
        _once(SYCL, 'std::snprintf(what, sizeof(what), "seed, probe failed: %s", e.what());',
              'GGML_LOG_WARN("failed %s", e.what()); std::snprintf(what, sizeof(what), "seed, probe failed: %s", '
              "e.what());"))


def test_outcome_enum_keeps_the_resolved_values_first():
    assert claim_outcome_enum_keeps_the_resolved_values_first(SYCL)


def test_mutant_outcome_enum_reordered_fails():
    assert not claim_outcome_enum_keeps_the_resolved_values_first(
        _once(SYCL, "H2D_PROBE_OUTCOME_FROM_CACHE, H2D_PROBE_OUTCOME_SEED_KEPT, H2D_PROBE_OUTCOME_RETRY_ZONES,",
              "H2D_PROBE_OUTCOME_FROM_CACHE, H2D_PROBE_OUTCOME_RETRY_ZONES, H2D_PROBE_OUTCOME_SEED_KEPT,"))


def test_mutant_outcome_enum_static_assert_dropped_fails():
    mutant = re.sub(r"static_assert\(H2D_PROBE_OUTCOME_MEASURED.*?\);", "", strip_comments(SYCL), flags=re.S)
    assert mutant != strip_comments(SYCL)
    assert not claim_outcome_enum_keeps_the_resolved_values_first(mutant)


def test_mutant_retry_bit_never_set_fails():
    assert not claim_each_unresolved_reason_logs_once_per_device(
        _once(SYCL, "d.h2d_retry_logged |= bit;", ""))


def test_mutant_iteration_count_overmatch_fails():
    assert not claim_copy_shape_is_chunked_async_with_one_wait_per_iteration(
        _once(SYCL, "constexpr int k_iters = 8;", "constexpr int k_iters = 80;"))


def test_mutant_warn_downgraded_to_info_fails():
    assert not claim_logs_once_at_warn(_once(SYCL, "GGML_LOG_WARN(" + LOG_FORMAT.replace("\\n", "\\n"),
                                             "GGML_LOG_INFO(" + LOG_FORMAT))


def test_mutant_log_source_word_dropped_fails():
    assert not claim_logs_once_at_warn(_once(SYCL, '"seed, host zones not configured"', '"seed"'))


def test_mutant_device_info_field_dropped_fails():
    for field in ("double h2d_pinned_gbps = 0.0;", "bool h2d_probed = false;", "uint32_t h2d_retry_logged = 0;"):
        assert claim_device_info_fields(COMMON)
        assert not claim_device_info_fields(_once(COMMON, field, ""))


def test_mutant_default_queue_fails():
    assert not claim_copies_run_on_the_cache_queue(
        _once(SYCL, "sycl::queue * q = &cache->get_queue();",
              "sycl::queue * q = &dpct::dev_mgr::instance().get_device(info.gpu_dpct_ids[dev]).default_queue();"))


def test_mutant_dst_on_another_queue_fails():
    assert not claim_copies_run_on_the_cache_queue(
        _once(SYCL, "dst_req.queue = q;", "dst_req.queue = nullptr;"))


def test_mutant_baseline_before_warmup_fails():
    mutant = _once(SYCL, "ggml_sycl::mem_copy(dst, src, k_bytes, *q);\n"
                   "            const uint64_t staged_before = "
                   "ggml_sycl::mem_copy_staged_wait_census_this_thread().staged_h2d;",
                   "const uint64_t staged_before = ggml_sycl::mem_copy_staged_wait_census_this_thread().staged_h2d;\n"
                   "            ggml_sycl::mem_copy(dst, src, k_bytes, *q);")
    assert not claim_zones_and_staged_copies_never_reach_the_store(mutant)


def test_mutant_staged_rate_is_kept_fails():
    assert not claim_zones_and_staged_copies_never_reach_the_store(
        _once(SYCL, 'report(dev, seed.h2d_gbps, H2D_PROBE_OUTCOME_RETRY_STAGED, "seed, probe copy was staged");',
              'report(dev, seed.h2d_gbps, H2D_PROBE_OUTCOME_SEED_KEPT, "seed, probe copy was staged");'))


def test_mutant_store_before_staged_check_fails():
    mutant = _once(SYCL, "rec.h2d_gbps = gbps;",
                   "rec.h2d_gbps = gbps; ggml_backend_sycl_moe_exec_rates_store(dev, &rec);")
    assert not claim_zones_and_staged_copies_never_reach_the_store(mutant)


def test_mutant_zone_check_removed_fails():
    assert not claim_zones_and_staged_copies_never_reach_the_store(
        _once(SYCL, "if (cache == nullptr || !cache->host_zones_configured()) {", "if (false) {"))


def test_mutant_third_reader_fails():
    """A second read inside the probe, one outside both readers, and a dropped accessor read each break the claim."""
    assert claim_two_consumers_read_the_rate()
    inside = _once(SYCL, "if (ggml_sycl_h2d_pinned_resolved(info, dev)) { continue; }",
                   "double extra = info.devices[dev].h2d_pinned_gbps; "
                   "if (ggml_sycl_h2d_pinned_resolved(info, dev)) { continue; }")
    assert not claim_two_consumers_read_the_rate({SYCL_REL: inside})
    outside = SYCL + ("\nstatic double h2d_extra_reader(const ggml_sycl_device_info & i) "
                      "{ return i.devices[0].h2d_pinned_gbps; }\n")
    assert not claim_two_consumers_read_the_rate({SYCL_REL: outside})
    no_accessor_read = _once(SYCL, "return d.h2d_probed ? d.h2d_pinned_gbps : 0.0;", "return 0.0;")
    assert not claim_two_consumers_read_the_rate({SYCL_REL: no_accessor_read})


def test_mutant_probed_rate_published_without_the_mutex_fails():
    assert claim_probed_rate_is_published_under_one_mutex(SYCL, COMMON, PLAN_CPP)
    lock = "std::lock_guard<std::mutex> lock(g_sycl_h2d_pinned_mutex);"
    # the accessor, the once-guard helper and the store each lose their lock
    for sig in (ACCESSOR_SIG, RESOLVED_SIG):
        at = strip_comments(SYCL).find(sig)
        assert at >= 0
        cut = strip_comments(SYCL)[at:].find(lock)
        assert cut >= 0
        mutant = strip_comments(SYCL)[:at + cut] + strip_comments(SYCL)[at + cut + len(lock):]
        assert not claim_probed_rate_is_published_under_one_mutex(mutant, COMMON, PLAN_CPP)
    store = _once(SYCL, "d.h2d_pinned_gbps = gbps;", "d.h2d_pinned_gbps = gbps;")
    probe_at = strip_comments(store).find("auto report = [&info]")
    store_lock = strip_comments(store)[probe_at:].find(lock)
    unlocked_store = (strip_comments(store)[:probe_at + store_lock] +
                      strip_comments(store)[probe_at + store_lock + len(lock):])
    assert not claim_probed_rate_is_published_under_one_mutex(unlocked_store, COMMON, PLAN_CPP)
    # weigh reading a field directly, and the accessor going undeclared
    direct = _once(PLAN_CPP, "ggml_sycl_h2d_probed_gbps(device_)", "ggml_sycl_info().devices[device_].h2d_pinned_gbps")
    assert not claim_probed_rate_is_published_under_one_mutex(SYCL, COMMON, direct)
    assert not claim_probed_rate_is_published_under_one_mutex(
        SYCL, _once(COMMON, "double ggml_sycl_h2d_probed_gbps(int device);", ""), PLAN_CPP)


def test_mutant_assembly_without_the_probed_fallback_fails():
    assert claim_assembly_takes_h2d_from_record_then_probe_then_seed(PLAN_HPP)
    dropped = _once(PLAN_HPP, "} else if (detail::moe_stream_rate_measured(probed_h2d_gbps)) {\n"
                    "        r.h2d_gbps = probed_h2d_gbps;\n    }", "}")
    assert not claim_assembly_takes_h2d_from_record_then_probe_then_seed(dropped)
    swapped = _once(PLAN_HPP, "if (record != nullptr && detail::moe_stream_rate_measured(record->h2d_gbps)) {",
                    "if (detail::moe_stream_rate_measured(probed_h2d_gbps)) {\n        r.h2d_gbps = probed_h2d_gbps;\n"
                    "    } else if (record != nullptr && detail::moe_stream_rate_measured(record->h2d_gbps)) {")
    assert not claim_assembly_takes_h2d_from_record_then_probe_then_seed(swapped)
    env_first = _once(PLAN_HPP, "moe_exec_rates_apply_env(r, getenv_fn);", "")
    assert not claim_assembly_takes_h2d_from_record_then_probe_then_seed(env_first)


def test_mutant_weigh_not_passing_the_probed_rate_fails():
    assert claim_weigh_passes_the_devices_probed_rate(PLAN_CPP)
    assert not claim_weigh_passes_the_devices_probed_rate(
        _once(PLAN_CPP, "probed_h2d_gbps, moe_stream_getenv", "0.0, moe_stream_getenv"))
    assert not claim_weigh_passes_the_devices_probed_rate(
        _once(PLAN_CPP, "ggml_sycl_h2d_probed_gbps(device_)", "0.0"))


def test_mutant_probe_stores_the_reorder_seed_fails():
    assert claim_probe_does_not_store_the_reorder_seed(SYCL)
    assert not claim_probe_does_not_store_the_reorder_seed(
        _once(SYCL, "rec.h2d_gbps = gbps;", "rec.h2d_gbps = gbps; rec.reorder_gbps = seed.reorder_gbps;"))
    assert not claim_probe_does_not_store_the_reorder_seed(
        _once(SYCL, "rec.h2d_gbps = gbps;", "rec.h2d_gbps = gbps; rec.reorder_gbps = 200.0;"))


def test_mutant_unmeasured_reorder_refused_fails():
    assert claim_record_accepts_an_unmeasured_reorder_rate(IO_HPP)
    refused = IO_HPP.replace("reorder_gbps >= 0.0", "reorder_gbps > 0.0")
    assert not claim_record_accepts_an_unmeasured_reorder_rate(refused)


def test_mutant_catalog_row_without_the_probed_rate_fails():
    assert claim_catalog_rows_state_the_source_order(ENV_DOC)
    assert not claim_catalog_rows_state_the_source_order(ENV_DOC.replace("the device's probed rate (", "(", 1))


def test_mutant_iterations_timed_as_one_block_fails():
    """The old shape: one clock around all the iterations."""
    mutant = _once(SYCL, ITER_LOOP + " { const auto t0 = std::chrono::steady_clock::now();", ITER_LOOP + " {")
    mutant = _once(mutant, ITER_LOOP, "const auto t0 = std::chrono::steady_clock::now(); " + ITER_LOOP)
    assert not claim_each_iteration_is_timed_on_its_own(mutant)


def test_mutant_iteration_rate_not_kept_per_slot_fails():
    assert not claim_each_iteration_is_timed_on_its_own(_once(SYCL, "iter_gbps[it] = ", "iter_gbps[0] = "))


def test_mutant_probe_persists_a_mean_fails():
    assert not claim_the_rate_is_the_max_over_iterations_two_to_eight(
        _once(SYCL, "const double gbps = " + HELPER_CALL + ";",
              "const double gbps = (iter_gbps[1] + iter_gbps[2]) / 2;"),
        COST)


def test_mutant_helper_counts_the_first_iteration_fails():
    assert not claim_the_rate_is_the_max_over_iterations_two_to_eight(
        SYCL, _once(COST, "for (int i = 1; i < n; ++i) {", "for (int i = 0; i < n; ++i) {"))


def test_mutant_helper_keeps_the_min_fails():
    assert not claim_the_rate_is_the_max_over_iterations_two_to_eight(
        SYCL, _once(COST, "if (g > best) {", "if (g < best) {"))


def test_mutant_helper_low_end_never_tracks_the_min_fails():
    assert not claim_the_rate_is_the_max_over_iterations_two_to_eight(
        SYCL, _once(COST, "if (g < low) {", "if (g > low) {"))


def test_mutant_helper_never_sees_a_valid_iteration_fails():
    assert not claim_the_rate_is_the_max_over_iterations_two_to_eight(
        SYCL, _once(COST, "any = true;", "any = false;"))


def test_mutant_spread_dropped_from_the_probed_line_fails():
    no_spread = MEASURED_REPORT.replace(", spread)", ")")
    assert not claim_the_spread_rides_in_the_probed_line_only(_once(SYCL, MEASURED_REPORT, no_spread))


def test_mutant_spread_on_another_outcome_fails():
    assert not claim_the_spread_rides_in_the_probed_line_only(
        _once(SYCL, 'report(dev, seed.h2d_gbps, H2D_PROBE_OUTCOME_RETRY_STAGED, "seed, probe copy was staged");',
              'report(dev, seed.h2d_gbps, H2D_PROBE_OUTCOME_RETRY_STAGED, "seed, probe copy was staged", spread);'))


def test_mutant_spread_format_dropped_fails():
    assert not claim_the_spread_rides_in_the_probed_line_only(
        _once(SYCL, SPREAD_WORD, '"%s; %.1f-%.1f"'))


def test_mutant_half_seed_floor_removed_fails():
    assert not claim_a_max_under_half_the_seed_is_retried_not_stored(
        SYCL, _once(COST, "MOE_EXEC_H2D_MIN_SEED_FRACTION = 0.5;", "MOE_EXEC_H2D_MIN_SEED_FRACTION = 0.0;"))


def test_mutant_helper_ignores_the_floor_fails():
    assert not claim_a_max_under_half_the_seed_is_retried_not_stored(
        SYCL, _once(COST, "!(best >= seed * MOE_EXEC_H2D_MIN_SEED_FRACTION)", "false"))


def test_mutant_below_floor_rate_reaches_the_store_fails():
    assert not claim_a_max_under_half_the_seed_is_retried_not_stored(
        _once(SYCL, "if (gbps <= 0.0) {", "if (false) {"), COST)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
