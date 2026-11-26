"""Source contract for the lifetime of the MXFP4 MoE decode reuse store's graph-local handles.

The store (g_mxfp4_moe_tg_reuse in mmvq.cpp, one per thread) keeps, besides its own Q8 staging buffers, a copy of the
mem_handle of the tensor whose Q8 copy it published (src1_handle) and that copy's ready event. That half is graph-local,
because a layer's GLU output sits at the same address every token. ggml_backend_sycl_graph_compute_impl invalidates it
at its start, but the segmented record and replay paths never pass through compute_impl, and nothing dropped it at the
end of any graph. A copy of a tensor extra's handle takes a chunk lease of its own, so the copy the last decode token
stored kept a VRAM arena chunk leased past the model's release, and the last release found the arena held by one
tensor_extra(copy) lease (GGML_SYCL_LEASE_HOLDERS=1 named mxfp4_moe_tg_reuse_store, GPT-OSS 20B on the B70,
2026-11-26).

The owner of that half is the graph compute that stored it, on the thread that stored it. The backend's graph compute
entry (ggml_backend_sycl_graph_compute_unchecked) drops it when the call ends, on every exit, through a scope object
declared right after the in-flight guard, so the store is empty before the in-flight count falls. The drop sits at the
entry and not in ggml_backend_sycl_graph_compute_impl, where the start's invalidation is, because the segmented record
and replay paths run MoE nodes straight from the entry without passing through compute_impl. A model release, which
never overlaps a graph compute on its contexts, finds the store empty. The model release cannot clear it itself: the
store is thread_local on the computing thread, and the release runs on whichever thread frees the model.

Host-only text assertions, no SYCL device. llama_test_pytest hands this file to pytest.main(), so the checks live
inside test_*() functions. Checks read COMMENT-STRIPPED, whitespace-normalised text. Each check has a mutation witness
that applies the regression it guards and must make the check fail.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"
GS = (SYCL / "ggml-sycl.cpp").read_text()
MMVQ = (SYCL / "mmvq.cpp").read_text()
MMVQH = (SYCL / "mmvq.hpp").read_text()

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


def mutate_in(src: str, sig: str, old: str, new: str) -> str:
    """`src` as code() with the first `old` inside the function `sig` replaced by `new`."""
    c = code(src)
    body = fn_body(c, sig)
    assert body and old in body, f"witness anchor {old!r} is not in {sig!r}"
    s = c.find(body, c.find(sig))
    return c[:s] + body.replace(old, new, 1) + c[s + len(body) :]


# ---- the graph compute drops the store's graph-local half on every exit ----

ENTRY_SIG = "static ggml_status ggml_backend_sycl_graph_compute_unchecked(ggml_backend_t backend, ggml_cgraph * cgraph)"
COMPUTE_SIG = ("static void ggml_backend_sycl_graph_compute_impl(ggml_backend_sycl_context * sycl_ctx, "
               "ggml_cgraph * cgraph)")
INFLIGHT = "graph_inflight_guard inflight_guard(g_sycl_graph_inflight);"
# The entry's phase-timing lambda returns its milliseconds before the in-flight guard; it is not an exit of the entry.
PHASE_RETURN = "return ms;"
NEW_GRAPH = "ggml_sycl_mxfp4_moe_tg_reuse_new_graph();"
END_GRAPH = "ggml_sycl_mxfp4_moe_tg_reuse_end_graph();"
SCOPE = ("struct mxfp4_moe_tg_reuse_graph_scope { ~mxfp4_moe_tg_reuse_graph_scope() { " + END_GRAPH +
         " } } mxfp4_moe_tg_reuse_scope;")


def claim_graph_compute_drops_the_store(gs: str) -> bool:
    c = code(gs)
    entry = fn_body(c, ENTRY_SIG)
    i = entry.find(INFLIGHT)
    j = entry.find(SCOPE)
    # The scope is armed right after the in-flight guard, so it ends before the guard lowers the count, at the entry's
    # own level (not inside a branch, where it would end with the branch), and no exit of the entry comes before it,
    # so every exit of every route drops the store. compute_impl keeps the start's invalidation and arms nothing: a
    # scope there would miss the segmented paths. The end is called from this scope and nowhere else.
    return (
        i != -1
        and entry.count(SCOPE) == 1
        and i < j
        and entry[i + len(INFLIGHT) : j].strip() == ""
        and entry[:j].count("return") == entry[:j].count(PHASE_RETURN)
        and fn_body(c, COMPUTE_SIG).count(NEW_GRAPH) == 1
        and c.count(END_GRAPH) == 1
    )


# ---- the end of a graph drops the copied handle and its event ----

END_SIG = "void ggml_sycl_mxfp4_moe_tg_reuse_end_graph()"
INVALIDATE_SIG = "static void mxfp4_moe_tg_reuse_invalidate()"
INVALIDATE_PARTS = (
    "g_mxfp4_moe_tg_reuse.valid = false;",
    "g_mxfp4_moe_tg_reuse.src1_handle = {};",
    "g_mxfp4_moe_tg_reuse.ready_event = {};",
    "g_mxfp4_moe_tg_reuse.ready_event_set = false;",
)
# The one place the store copies a handle it does not own.
STORE_SIG = "static void mxfp4_moe_tg_reuse_store("
STORE_COPY = "cache.src1_handle = src1_handle;"


def claim_end_drops_the_copied_handle(mmvq: str, mmvqh: str) -> bool:
    c = code(mmvq)
    inv = fn_body(c, INVALIDATE_SIG)
    # Every write of the store's src1_handle: the store's copy and the invalidation's clear, nowhere else.
    writes = re.findall(r"\b(?:cache|g_mxfp4_moe_tg_reuse)\.src1_handle = [^;]*;", c)
    return (
        fn_body(c, END_SIG).strip() == "{ mxfp4_moe_tg_reuse_invalidate(); }"
        and all(p in inv for p in INVALIDATE_PARTS)
        and STORE_COPY in fn_body(c, STORE_SIG)
        and sorted(writes) == sorted([STORE_COPY, "g_mxfp4_moe_tg_reuse.src1_handle = {};"])
        and "void ggml_sycl_mxfp4_moe_tg_reuse_end_graph();" in code(mmvqh)
    )


# ---- the model release says when a graph call is still in flight ----

RELEASE_SIG = "static void ggml_sycl_release_model_slot_resources(ggml_sycl::lifecycle::ModelToken owner)"
INFLIGHT_READ = "const int inflight = ggml_sycl_graph_inflight_count();"
INFLIGHT_GUARD = "if (inflight != 0) {"
INFLIGHT_WARN = ('ggml_sycl_diag_emit_warn( "[SYCL] model slot %u release: %d graph call(s) still in flight; a '
                 'following [VRAM-ARENA] \\"it stays\\" line may be a lease the MXFP4 MoE decode reuse store of such a '
                 'call still holds, not a leak\\n", slot, inflight);')
SLOT_RECLAIM = "ggml_sycl::unified_cache_release_model_slot(slot);"


def claim_release_warns_on_a_call_in_flight(gs: str) -> bool:
    # The drop rests on no graph call being in flight at the model release. The release checks that, and a call still
    # in flight is named at WARN, with its count, before the slot's reclaim and the last-release arena check, so an E
    # line that follows is read as that call's lease and not as a leak. Adjacent literals read as one.
    body = fn_body(code(gs).replace('" "', ""), RELEASE_SIG)
    w = body.find(INFLIGHT_WARN)
    g = body.find(INFLIGHT_GUARD)
    return (
        body.find(INFLIGHT_READ) != -1
        and body.find(INFLIGHT_READ) < g < w < body.find(SLOT_RECLAIM)
        and body[g + len(INFLIGHT_GUARD) : w].strip() == ""
        and body.count(INFLIGHT_WARN) == 1
    )


# ---- checks ----


def test_graph_compute_drops_the_store():
    assert claim_graph_compute_drops_the_store(GS), (
        "ggml_backend_sycl_graph_compute_unchecked does not arm, right after its in-flight guard, at its own level and "
        "before any exit, the one scope that drops the MXFP4 MoE decode reuse store's graph-local handles when the "
        "call ends, so the last graph's copy of a tensor handle outlives the model and holds an arena chunk lease"
    )


def test_end_drops_the_copied_handle():
    assert claim_end_drops_the_copied_handle(MMVQ, MMVQH), (
        "the end of a graph does not drop the reuse store's copied tensor handle and ready event, or the store copies "
        "a handle somewhere the invalidation does not clear"
    )


def test_witness_no_scope():
    assert not claim_graph_compute_drops_the_store(mutate_in(GS, ENTRY_SIG, SCOPE, ""))


def test_witness_scope_in_a_branch():
    assert not claim_graph_compute_drops_the_store(
        mutate_in(GS, ENTRY_SIG, SCOPE, "if (cgraph->n_nodes > 0) { " + SCOPE + " }")
    )


def test_witness_scope_after_an_early_return():
    assert not claim_graph_compute_drops_the_store(
        mutate_in(GS, ENTRY_SIG, INFLIGHT + " " + SCOPE,
                  INFLIGHT + " if (cgraph->n_nodes == 0) { return GGML_STATUS_SUCCESS; } " + SCOPE)
    )


def test_witness_scope_before_the_inflight_guard():
    # Declared before the guard, the scope would end after the in-flight count fell.
    assert not claim_graph_compute_drops_the_store(
        mutate_in(GS, ENTRY_SIG, INFLIGHT + " " + SCOPE, SCOPE + " " + INFLIGHT)
    )


def test_witness_scope_only_in_compute_impl():
    # The first placement: the segmented record and replay paths never reach it.
    moved = mutate_in(GS, ENTRY_SIG, SCOPE, "")
    assert not claim_graph_compute_drops_the_store(mutate_in(moved, COMPUTE_SIG, NEW_GRAPH, NEW_GRAPH + " " + SCOPE))


def test_witness_scope_does_nothing():
    assert not claim_graph_compute_drops_the_store(
        mutate_in(GS, ENTRY_SIG, SCOPE, SCOPE.replace(END_GRAPH, ""))
    )


def test_witness_end_does_not_invalidate():
    assert not claim_end_drops_the_copied_handle(mutate_in(MMVQ, END_SIG, "mxfp4_moe_tg_reuse_invalidate();", ""),
                                                 MMVQH)


def test_witness_invalidate_keeps_the_copy():
    assert not claim_end_drops_the_copied_handle(
        mutate_in(MMVQ, INVALIDATE_SIG, "g_mxfp4_moe_tg_reuse.src1_handle = {};", ""), MMVQH
    )


def test_witness_second_copy_site():
    assert not claim_end_drops_the_copied_handle(
        mutate_in(MMVQ, STORE_SIG, STORE_COPY, STORE_COPY + " cache.src1_handle = cache.q8_handle;"), MMVQH
    )


def test_release_warns_on_a_call_in_flight():
    assert claim_release_warns_on_a_call_in_flight(GS), (
        "ggml_sycl_release_model_slot_resources does not WARN, with the count and before the slot's reclaim, when a "
        "graph call is still in flight, so an E-level arena line from that call's reuse store reads as a leak"
    )


def release_mutant(old: str, new: str) -> str:
    return mutate_in(code(GS).replace('" "', ""), RELEASE_SIG, old, new)


def test_witness_release_without_the_warn():
    assert not claim_release_warns_on_a_call_in_flight(release_mutant(INFLIGHT_WARN, ""))


def test_witness_release_warns_when_idle():
    assert not claim_release_warns_on_a_call_in_flight(release_mutant(INFLIGHT_GUARD, "if (inflight == 0) {"))


def test_witness_release_warns_below_default_verbosity():
    assert not claim_release_warns_on_a_call_in_flight(
        release_mutant(INFLIGHT_WARN, INFLIGHT_WARN.replace("ggml_sycl_diag_emit_warn(", "GGML_LOG_INFO("))
    )


def test_witness_release_warns_after_the_reclaim():
    moved = release_mutant(INFLIGHT_GUARD + " " + INFLIGHT_WARN + " }", "")
    assert not claim_release_warns_on_a_call_in_flight(
        mutate_in(moved, RELEASE_SIG, SLOT_RECLAIM, SLOT_RECLAIM + " " + INFLIGHT_GUARD + " " + INFLIGHT_WARN + " }")
    )


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
