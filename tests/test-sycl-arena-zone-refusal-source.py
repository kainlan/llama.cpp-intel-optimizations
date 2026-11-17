"""Source contract for a second model load that cannot size the VRAM arena zones.

A process that frees one model and loads another sizes the arena zones again. The zones grow only by a rebuild, and a
rebuild waits until nothing is live in the arena. These checks pin what the load does around that wait:

(1) The rebuild refusal names who holds the arena: each zone's live bytes, the frees still queued behind a GPU event
    whose bytes the zones still count, and the releases an arena lease refused. The RUNTIME requirement line names
    every term it sums.

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
HOLDERS_LINE = '"[VRAM-ARENA] live in the arena on device %d (MB): %s; deferred_frees=%zu (%.1f MB) "'
RUNTIME_LINE = '"[UNIFIED-CACHE] Runtime zone raised to %.1f MB for PP scratch planning "'
# A queued free holds zone bytes only when it is a zone sub-allocation or points into the arena: a direct allocation
# outside the arena is in no zone's `used`, so counting it would overstate what the queue explains.
DEFERRED_IN_ARENA = "if (entry.zone_managed || vram_owns(entry.ptr)) {"


def claim_rebuild_refusal_names_the_holders(uc: str) -> bool:
    ens = fn_body(code(uc), ENSURE_SIG)
    i = ens.find(REFUSAL_LINE)
    j = ens.find(HOLDERS_LINE, max(i, 0))
    r = ens.find("return false;", max(i, 0))
    if -1 in (i, j, r) or not (i < j < r):
        return False
    seg = ens[i:r]
    return (
        seg.count("GGML_LOG_ERROR(") == 1
        and ordered(
            seg,
            "for (const deferred_free_entry & entry : deferred_frees_) {",
            DEFERRED_IN_ARENA,
            "++deferred_count;",
            "deferred_bytes += entry.size;",
        )
        and all(
            t in seg
            for t in (
                "std::lock_guard<std::mutex> deferred_lock(deferred_frees_mutex_);",
                "arena_zone_name(zone)",
                "zone_used(zone)",
                "g_allocation_coordinators.find(dev_id)",
                "retry_count()",
                "live_controls()",
            )
        )
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


def test_runtime_line_names_every_term():
    assert claim_runtime_line_names_every_term(UC), (
        "the 'Runtime zone raised' line omits the host scatter, mmq_src1 or dequant_f16 term it sums, so two lines "
        "whose totals differ cannot say which term moved"
    )


def test_witness_holders_line_without_deferred_bytes():
    assert not claim_rebuild_refusal_names_the_holders(
        mutate_in(UC, ENSURE_SIG, "deferred_bytes += entry.size;", "")
    )


def test_witness_deferred_frees_outside_the_arena_counted():
    assert not claim_rebuild_refusal_names_the_holders(mutate_in(UC, ENSURE_SIG, DEFERRED_IN_ARENA, "{"))


def test_witness_holders_line_dropped():
    assert not claim_rebuild_refusal_names_the_holders(mutate_in(UC, ENSURE_SIG, HOLDERS_LINE, '"x"'))


def test_witness_runtime_line_without_scatter():
    assert not claim_runtime_line_names_every_term(mutate_in(UC, ENSURE_SIG, "moe_host_scatter=%.1f KB, ", ""))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
