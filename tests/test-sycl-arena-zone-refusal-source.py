"""Source contract for a second model load that cannot size the VRAM arena zones.

A process that frees one model and loads another sizes the arena zones again. The zones grow only by a rebuild, and a
rebuild waits until nothing is live in the arena. These checks pin what the load does around that wait:

(1) The rebuild refusal names who holds the arena: each zone's live bytes from the reading the refusal decided on,
    the frees still queued behind a GPU event whose bytes the zones still count, and the device's releases an arena
    lease refused. The line prints those variables, and the path only reads: it drains, frees, evicts, resets and
    allocates nothing. The RUNTIME requirement line names every term it sums.

Host-only text assertions, no SYCL device. llama_test_pytest hands this file to pytest.main(), so the checks live
inside test_*() functions. Checks read COMMENT-STRIPPED, whitespace-normalised text. Each check has a mutation witness
that applies the regression it guards and must make the check fail.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Plain file reads: the codescout index does not see inside ggml-sycl.cpp.
SYCL = (ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()
UC = (ROOT / "ggml/src/ggml-sycl/unified-cache.cpp").read_text()

_LEXEME_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"'  # string literal (kept)
    r"|'(?:\\.|[^'\\\n])*'"  # char literal (kept)
    r"|//[^\n]*"  # line comment (dropped)
    r"|/\*.*?\*/",  # block comment (dropped)
    flags=re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def code(src: str) -> str:
    """Comment-stripped text with every run of whitespace made one space. Idempotent."""
    return re.sub(r"\s+", " ", strip_comments(src))


def block_at(text: str, start: int) -> str:
    """The brace block that opens at the first '{' at or after `start`, skipping string and char literals."""
    o = text.find("{", start) if start >= 0 else -1
    if o == -1:
        return ""
    depth, i = 0, o
    while i < len(text):
        c = text[i]
        if c in "\"'":
            j = i + 1
            while j < len(text) and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[o : i + 1]
        i += 1
    return ""


def fn_body(text: str, sig: str) -> str:
    s = text.find(sig)
    return block_at(text, s + len(sig)) if s != -1 else ""


def ordered(text: str, *parts: str) -> bool:
    pos = 0
    for p in parts:
        k = text.find(p, pos)
        if k == -1:
            return False
        pos = k + len(p)
    return bool(text)


def mutate_in(src: str, sig: str, old: str, new: str) -> str:
    """`src` as code() with the first `old` inside the function `sig` replaced by `new`."""
    c = code(src)
    body = fn_body(c, sig)
    assert body and old in body, f"witness anchor {old!r} is not in {sig!r}"
    s = c.find(body, c.find(sig))
    return c[:s] + body.replace(old, new, 1) + c[s + len(body) :]


ENSURE_SIG = "bool unified_cache::ensure_planned_arena_zones()"
STORE_SIG = "static void compute_and_store_plan_for_inventory("
IMPL_SIG = "lifecycle_plan_snapshot> ggml_sycl_set_tensor_inventory_impl("
STAGE_SIG = "ggml_sycl_lifecycle_result ggml_backend_sycl_stage_load_inventory_plan("


# ---- (1) the refusal names who holds the arena ----

REFUSAL_LINE = '"[VRAM-ARENA] planned zones exceed active arena but live allocations prevent rebuild: "'
HOLDERS_LINE = '"[VRAM-ARENA] live in the arena on device %d (MB): %s; deferred_in_arena=%zu (%.1f MB requested) "'
RUNTIME_LINE = '"[UNIFIED-CACHE] Runtime zone raised to %.1f MB for PP scratch planning "'
# A queued free holds zone bytes only when it is a zone sub-allocation or points into the arena: a direct allocation
# outside the arena is in no zone's `used`, so counting it would overstate what the queue explains.
DEFERRED_IN_ARENA = "if (entry.zone_managed || vram_owns(entry.ptr)) {"
# The per-zone figures are recorded in the loop that sums live_zone, so the holders line prints the reading the
# refusal decided on. A second zone_used() read could disagree with it under a concurrent release.
ZONE_RECORD = "live_zone_used[z] = zone_used(static_cast<vram_zone_id>(z));"
ZONE_SUM = "live_zone_bytes += live_zone_used[z];"
ZONE_PRINT = "live_zone_used[z] / (1024.0 * 1024.0)"
# The variables the holders line must print, by name, outside its format string.
HOLDERS_ARGS = (
    "zones_live",
    "deferred_in_arena_count",
    "deferred_in_arena_bytes",
    "device_release_retries",
    "device_live_controls",
)
# What the diagnostic must never do: it reads counters, and a drain, free, eviction, reset or allocation here would
# change the state it reports on.
MUTATORS = (
    "process_deferred_frees",
    "zone_free(",
    "evict",
    "arena_reset(",
    "zone_reset",
    "arena_destroy(",
    "reset_model_weight_entries(",
    "scratch_pool_reset_regions(",
    "drain_all_queues",
    "finalize_retired",
    "unified_alloc",
    "unified_allocate",
)
_STRING_RE = re.compile(r'"(?:\\.|[^"\\\n])*"')


def refusal_segment(uc: str) -> str:
    """ensure_planned_arena_zones() from the refusal line up to its return false, or "" when the shape is gone."""
    ens = fn_body(code(uc), ENSURE_SIG)
    i = ens.find(REFUSAL_LINE)
    r = ens.find("return false;", max(i, 0))
    return ens[i:r] if -1 not in (i, r) else ""


def holders_call_args(seg: str) -> str:
    """The holders ERROR call from its format string to its closing ');', with every string literal emptied."""
    j = seg.find(HOLDERS_LINE)
    e = seg.find(");", j) if j != -1 else -1
    return _STRING_RE.sub('""', seg[j:e]) if e != -1 else ""


def claim_rebuild_refusal_names_the_holders(uc: str) -> bool:
    seg = refusal_segment(uc)
    if HOLDERS_LINE not in seg:
        return False
    return (
        seg.count("GGML_LOG_ERROR(") == 1
        and ordered(
            seg,
            "for (const deferred_free_entry & entry : deferred_frees_) {",
            DEFERRED_IN_ARENA,
            "++deferred_in_arena_count;",
            "deferred_in_arena_bytes += entry.size;",
        )
        and all(
            t in seg
            for t in (
                "std::lock_guard<std::mutex> deferred_lock(deferred_frees_mutex_);",
                "arena_zone_name(",
                "g_allocation_coordinators.find(dev_id)",
                "device_release_retries = it->second->retry_count();",
                "device_live_controls = it->second->live_controls();",
                "deferred_in_arena=%zu (%.1f MB requested) ",
                "device_release_retries=%zu device_live_controls=%zu cache_scratch=%d",
            )
        )
    )


def claim_holders_line_prints_its_figures(uc: str) -> bool:
    args = holders_call_args(refusal_segment(uc))
    return bool(args) and all(re.search(rf"\b{v}\b", args) for v in HOLDERS_ARGS)


def claim_diagnostic_mutates_nothing(uc: str) -> bool:
    seg = refusal_segment(uc)
    return bool(seg) and not any(t in seg for t in MUTATORS)


def claim_zone_figures_are_the_refusals_reading(uc: str) -> bool:
    ens = fn_body(code(uc), ENSURE_SIG)
    seg = refusal_segment(uc)
    return (
        ordered(ens, ZONE_RECORD, ZONE_SUM, REFUSAL_LINE)
        and ZONE_PRINT in seg
        and "zone_used(" not in seg
    )


def claim_runtime_line_names_every_term(uc: str) -> bool:
    ens = fn_body(code(uc), ENSURE_SIG)
    k = ens.find(RUNTIME_LINE)
    e = ens.find(");", k) if k != -1 else -1
    line = ens[k:e] if e != -1 else ""
    return all(
        t in line
        for t in (
            "moe_host_scatter=%.1f KB",
            "mmq_src1=%.1f MB",
            "dequant_f16=%.1f MB",
            "unified_cache_get_planned_moe_host_scatter_scratch_bytes(dev_id) / 1024.0",
            "unified_cache_get_planned_mmq_src1_scratch_bytes(dev_id)",
            "unified_cache_get_planned_dequant_f16_scratch_bytes(dev_id)",
        )
    )


def test_rebuild_refusal_names_the_holders():
    assert claim_rebuild_refusal_names_the_holders(UC), (
        "ensure_planned_arena_zones() refuses a rebuild without a second ERROR line naming each zone's live bytes, the "
        "queued deferred frees the zones still count and the device's refused releases, between the refusal line and "
        "its return false"
    )


def test_holders_line_prints_its_figures():
    assert claim_holders_line_prints_its_figures(UC), (
        f"the holders ERROR call does not pass every one of {HOLDERS_ARGS} to its format, so a figure it labels is "
        "printed as something else"
    )


def test_diagnostic_mutates_nothing():
    assert claim_diagnostic_mutates_nothing(UC), (
        f"the refusal path from its ERROR line to return false calls one of {MUTATORS}: the diagnostic must only read "
        "counters, never drain, free, evict, reset or allocate"
    )


def test_zone_figures_are_the_refusals_reading():
    assert claim_zone_figures_are_the_refusals_reading(UC), (
        "the holders line does not print the per-zone figures recorded in the loop that sums live_zone, or reads "
        "zone_used() again, so it can disagree with the refusal it explains"
    )


def test_runtime_line_names_every_term():
    assert claim_runtime_line_names_every_term(UC), (
        "the 'Runtime zone raised' line omits the host scatter, mmq_src1 or dequant_f16 term it sums, so two lines "
        "whose totals differ cannot say which term moved"
    )


def test_witness_holders_line_without_deferred_bytes():
    assert not claim_rebuild_refusal_names_the_holders(
        mutate_in(UC, ENSURE_SIG, "deferred_in_arena_bytes += entry.size;", "")
    )


def test_witness_deferred_frees_outside_the_arena_counted():
    assert not claim_rebuild_refusal_names_the_holders(mutate_in(UC, ENSURE_SIG, DEFERRED_IN_ARENA, "{"))


def test_witness_holders_line_dropped():
    assert not claim_rebuild_refusal_names_the_holders(mutate_in(UC, ENSURE_SIG, HOLDERS_LINE, '"x"'))


def test_witness_holders_line_prints_a_literal_for_live_controls():
    assert not claim_holders_line_prints_its_figures(
        mutate_in(
            UC,
            ENSURE_SIG,
            "device_release_retries, device_live_controls, has_live_scratch",
            "device_release_retries, 0, has_live_scratch",
        )
    )


def test_witness_diagnostic_drains_the_queue():
    lock = "std::lock_guard<std::mutex> deferred_lock(deferred_frees_mutex_);"
    assert not claim_diagnostic_mutates_nothing(
        mutate_in(UC, ENSURE_SIG, lock, "process_deferred_frees_locked(); " + lock)
    )


def test_witness_zone_figures_read_again():
    assert not claim_zone_figures_are_the_refusals_reading(
        mutate_in(UC, ENSURE_SIG, ZONE_PRINT, "zone_used(static_cast<vram_zone_id>(z)) / (1024.0 * 1024.0)")
    )


def test_witness_runtime_line_without_scatter():
    assert not claim_runtime_line_names_every_term(mutate_in(UC, ENSURE_SIG, "moe_host_scatter=%.1f KB, ", ""))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
