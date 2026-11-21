#!/usr/bin/env python3
"""Decode prologue memo of a graph llama reuses (GGML_SYCL_GRAPH_PROLOGUE_MEMO): source contract.

THE COST.  On a replayed keyed decode token the backend walks the graph three times before the device has anything
queued: the structural signature, the keyed segment slot key and the input prestage. llama reuses the graph on almost
every decode token, and a reused split keeps its ggml_cgraph::uid, its tensor structs, buffers and data pointers, so
graph-prologue-memo.hpp serves those walks from the first full pass under the same key. The memo's own behaviour (the
switch values, the key equality, arming, what is served and recorded, clear, drop_parts, the verify bookkeeping) is
unit-tested on the host (ggml/src/ggml-sycl/tests/test-graph-prologue-memo.cpp); this pins how the backend uses it.

THE CONTRACT.  With comments blanked:
  K. The key: graph_prologue_memo_key fills uid, n_nodes, n_leafs from the graph, the device from the context, the
     phase from its caller and the tenant publish generation from ggml_sycl_tenant_publish_gen().
  U. Every wrapper starts its call, right after its mode-0 return and before it reads a part, with
     graph_prologue_memo_begin under the key of its graph and the phase it was given (the prestage: the phase
     graph_compute stored at its entry), and that helper hands begin() the key.
  A. The memo arms only where nothing rewrites a graph tensor's data under an unchanged key: the helper hands begin()
     graph_prologue_memo_may_arm, which answers no while the cross-card expert path is requested and otherwise yes on
     one routable device or under a plan that keeps the context on its own device; begin() is called nowhere else.
  M. Mode 0 calls the three full functions unconditionally: the switch is read once, each wrapper's first decision
     returns its full function when the mode is OFF (the signature also when the call is not on the keyed decode
     path), before it touches the memo, and the call sites go through the wrappers (the keyed slot's prestage alone,
     through the gateway's last argument).
  R. The effect list is replayed in order with the failure handling: graph_prestage_leaf_tensors with a replay list
     runs stage_effect for exactly its entries, in order, and returns all_staged before the walk; stage_tensor reaches
     a stage action only through stage_effect; stage_effect records each tensor once, before its action, and keeps the
     three actions with their failure handling; a hit replays the held list and a replay that fails drops it.
  N. A pass that did not stage everything is not memoized: the wrapper records with the pass's own answer.
  C. A miss builds the record containers only when the memo is armed: the slot key's buffer list and the prestage's
     effect list (and with it the walk's set of recorded tensors) are handed to the full function only past an
     armed() check.
  I. Every invalidation point clears the memo: invalidate_moe_segments first of all, graph_staging_swap_retire
     through its unconditional invalidate_moe_segments, the optional-layout epoch path unconditionally, and the
     context's teardown (the report, then invalidate_moe_segments); the memo is a by-value member of the context.
  S. The slot key is served only while its distinct buffers keep the identity and base the full walk read, and the
     full walk hands those buffers out as it read them; a held key whose check fails drops every part (the key stays
     armed) before the full walk records afresh.
  V. The verify comparison covers all three parts, and a difference logs the mismatch line, which names the part. A
     prestage hit in verify mode runs the replay first, then the full pass, and compares the replay's answer and the
     held list with the pass's answer and list.
  T. The teardown line, its exact text, and when it prints (modes 1 and 2, under the trace switch or in mode 2).
  D. docs/backend/sycl-env-vars.md documents the switch, the teardown line and the mismatch line.

NOT VACUOUS.  --self-test runs the gate on in-memory mutants of the real sources and requires each to fail the named
pin, then requires the unmodified tree to pass.

RED on the tree before the change: every pin fails (the wrappers and the docs row do not exist).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYCL = ROOT / "ggml" / "src" / "ggml-sycl"
BACKEND = SYCL / "ggml-sycl.cpp"
COMMON = SYCL / "common.hpp"
DOC = ROOT / "docs" / "backend" / "sycl-env-vars.md"

ENV_NAME = "GGML_SYCL_GRAPH_PROLOGUE_MEMO"
TEARDOWN_LINE = "[PROLOGUE-MEMO] device %d: hits=%llu misses=%llu verified=%llu mismatches=%llu\\n"
MISMATCH_TAG = "[PROLOGUE-MEMO] mismatch"
PRESTAGE_PHASE = r"ctx->graph_phase_is_decode\.load\(std::memory_order_relaxed\)"

# Comments blank to spaces, keeping offsets; strings and char literals are matched first so a comment marker inside
# one is not taken for a comment.
_TOKEN = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n]){1,4}\'', re.S)


def blank_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        t = m.group(0)
        if t.startswith("//") or t.startswith("/*"):
            return "".join("\n" if c == "\n" else " " for c in t)
        return t

    return _TOKEN.sub(repl, src)


def norm(src: str) -> str:
    return re.sub(r"\s+", " ", src).strip()


def body_of(code: str, signature: str) -> str | None:
    """The brace-balanced body of the first definition matching `signature` (a regex ending at its opening brace),
    or None. Braces inside string and char literals do not count."""
    m = re.search(signature, code)
    if not m:
        return None
    i = code.find("{", m.end() - 1)
    if i < 0:
        return None
    depth = 0
    j = i
    while j < len(code):
        c = code[j]
        if c in "\"'":
            k = j + 1
            while k < len(code) and code[k] != c:
                k += 2 if code[k] == "\\" else 1
            j = k + 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return code[i:j + 1]
        j += 1
    return None


def top_level(body: str, at: int) -> bool:
    """True when offset `at` of a function body sits directly in it, not inside a nested block."""
    return body[:at].count("{") - body[:at].count("}") == 1


SIG = {
    "key": r"static ggml_sycl::graph_prologue_memo::key graph_prologue_memo_key\([^)]*\)\s*\{",
    "mode": r"static ggml_sycl::graph_prologue_memo::mode graph_prologue_memo_mode\(\)\s*\{",
    "mismatch": r"static void graph_prologue_memo_mismatch\([^)]*\)\s*\{",
    "report": r"static void graph_prologue_memo_report\([^)]*\)\s*\{",
    "may_arm": r"static bool graph_prologue_memo_may_arm\([^)]*\)\s*\{",
    "begin": r"static void graph_prologue_memo_begin\([^)]*\)\s*\{",
    "signature": r"static uint64_t graph_prologue_memo_signature\([^)]*\)\s*\{",
    "slot_key": r"static ggml_sycl::graph_segment_cache::key graph_prologue_memo_slot_key\([^)]*\)\s*\{",
    "prestage": r"static bool graph_prologue_memo_prestage\([^)]*\)\s*\{",
    "full_key": r"static ggml_sycl::graph_segment_cache::key moe_segment_slot_key\([^)]*\)\s*\{",
    "walk": r"static bool graph_prestage_leaf_tensors\([^)]*\)\s*\{",
    "gateway": r"static bool graph_prestage_or_decline\([^)]*\)\s*\{",
    "swap": r"static bool graph_staging_swap_retire\([^)]*\)\s*\{",
    "compute": r"static ggml_status ggml_backend_sycl_graph_compute_unchecked\([^)]*\)\s*\{",
    "dtor": r"ggml_backend_sycl_context::~ggml_backend_sycl_context\(\)\s*\{",
}
SIG_PIN = {"signature": "M", "slot_key": "M", "prestage": "M", "mode": "M", "key": "K", "walk": "R", "mismatch": "V",
           "report": "T", "may_arm": "A", "begin": "U"}

# Each wrapper's opening, normalized: the switch, then the mode-0 return of its full function.
FIRST = {
    "signature": r"\{ namespace gpm = ggml_sycl::graph_prologue_memo; "
                 r"const gpm::mode mode = graph_prologue_memo_mode\(\); "
                 r"if \(mode == gpm::mode::OFF \|\| !moe_segment_keyed_mode\(ctx, is_decode\)\) \{ "
                 r"return ggml_sycl_graph_signature\(cgraph\); \}",
    "slot_key": r"\{ namespace gpm = ggml_sycl::graph_prologue_memo; "
                r"const gpm::mode mode = graph_prologue_memo_mode\(\); "
                r"if \(mode == gpm::mode::OFF\) \{ return moe_segment_slot_key\(ctx, cgraph, signature, is_decode\); \}",
    "prestage": r"\{ namespace gpm = ggml_sycl::graph_prologue_memo; "
                r"const gpm::mode mode = graph_prologue_memo_mode\(\); "
                r"if \(mode == gpm::mode::OFF\) \{ return graph_prestage_leaf_tensors\(ctx, cgraph\); \}",
}
PHASE = {"signature": "is_decode", "slot_key": "is_decode", "prestage": PRESTAGE_PHASE}

# A prestage hit: the replay runs before the mode decides anything, so verify mode exercises it too.
PRESTAGE_HIT = (r"if \(held\) \{ memo\.note_hit\(\); "
                r"const bool replayed = graph_prestage_leaf_tensors\(ctx, cgraph, held\); "
                r"if \(mode == gpm::mode::ON\) \{ if \(!replayed\) \{ memo\.drop_prestage\(\); \} return replayed; \}")


def check(backend: str, common: str, doc: str) -> list[str]:
    """Every failed pin as 'X: reason'. The sources are comment-blanked already (the doc is not)."""
    out: list[str] = []

    def need(pin: str, ok: bool, reason: str) -> None:
        if not ok:
            out.append(f"{pin}: {reason}")

    bodies = {name: body_of(backend, sig) for name, sig in SIG.items()}
    for name, body in bodies.items():
        need(SIG_PIN.get(name, "I"), body is not None, f"no definition of {name} ({SIG[name]})")
    if any(b is None for b in bodies.values()):
        return out
    b = {name: norm(body) for name, body in bodies.items()}
    c = norm(common)

    # --- K: the key ----------------------------------------------------------------------------------------------
    for field, value in (("uid", "cgraph->uid"), ("n_nodes", "cgraph->n_nodes"), ("n_leafs", "cgraph->n_leafs"),
                         ("device", "ctx->device"), ("is_decode", "is_decode"),
                         ("tenant_gen", "ggml_sycl_tenant_publish_gen()")):
        need("K", re.search(r"\bk\.%s = %s;" % (field, re.escape(value)), b["key"]) is not None,
             f"the key does not take {field} from {value}")

    # --- U: every wrapper begins under its graph's key and phase --------------------------------------------------
    for name, phase in PHASE.items():
        need("U", re.match(FIRST[name] + r" auto & memo = ctx->prologue_memo; graph_prologue_memo_begin\(ctx, cgraph, "
                           r"%s\); const " % phase, b[name]) is not None,
             f"the {name} wrapper does not begin() under its graph's key and phase before it reads a part")
    need("U", b["begin"].startswith("{ ctx->prologue_memo.begin(graph_prologue_memo_key(ctx, cgraph, is_decode), "),
         "graph_prologue_memo_begin does not hand begin() the key of its graph and phase")

    # --- A: the arming condition ------------------------------------------------------------------------------------
    need("A", re.fullmatch(r"\{ ctx->prologue_memo\.begin\(graph_prologue_memo_key\(ctx, cgraph, is_decode\), "
                           r"\[ctx\] \{ return graph_prologue_memo_may_arm\(ctx\); \}\); \}", b["begin"]) is not None,
         "begin() is not handed graph_prologue_memo_may_arm as its arming condition")
    need("A", re.fullmatch(r"\{ if \(ggml_sycl_moe_multi_gpu_requested\(\)\) \{ return false; \} "
                           r"return ggml_sycl_routable_device_count\(\) <= 1 \|\| "
                           r"ggml_sycl_current_plan_is_single_device_for_graph\(ctx->device\); \}", b["may_arm"])
         is not None,
         "the arming condition is not: no cross-card expert path, and one routable device or a single-device plan")
    need("A", len(re.findall(r"\b(?:prologue_memo|memo) ?\. ?begin\(", norm(backend))) == 1,
         "begin() is called outside graph_prologue_memo_begin, without the arming condition")

    # --- M: mode 0 is today's path --------------------------------------------------------------------------------
    need("M", re.search(r"static const ggml_sycl::graph_prologue_memo::mode mode = "
                        r"ggml_sycl::graph_prologue_memo::mode_parse\(std::getenv\(\"%s\"\)\); return mode;" % ENV_NAME,
                        b["mode"]) is not None, "the switch is not read once through mode_parse")
    need("M", backend.count('"%s"' % ENV_NAME) == 1, "the switch is read at more than one site")
    for name, pattern in FIRST.items():
        need("M", re.match(pattern, b[name]) is not None,
             f"the {name} wrapper does not return its full function first when the mode is OFF")
    compute = b["compute"]
    need("M", "const uint64_t graph_hash = graph_prologue_memo_signature(sycl_ctx, cgraph, cached_is_decode);"
         in compute, "the compute's signature does not go through graph_prologue_memo_signature")
    need("M", "const gsc::key slot_key = graph_prologue_memo_slot_key(sycl_ctx, cgraph, graph_hash, is_decode_phase);"
         in compute, "the keyed slot key does not go through graph_prologue_memo_slot_key")
    need("M", re.search(r"\} else if \(slot_action == gsc::action::DIRECT\) \{ compute_impl_unlocked\(\); \} "
                        r"else if \(!graph_prestage_or_decline\(sycl_ctx, cgraph, graph_hash, true\)\) \{", compute)
         is not None, "the keyed slot's prestage does not go through the memo")
    four = re.findall(r"(?<!bool )\bgraph_prestage_or_decline\([^;{]*?,[^;{]*?,[^;{]*?,", norm(backend))
    need("M", len(four) == 1, f"{len(four)} prestage consumers pass the memo argument; only the keyed slot may")
    need("M", re.search(r"static bool graph_prestage_or_decline\(ggml_backend_sycl_context \* ctx, "
                        r"const ggml_cgraph \* cgraph, uint64_t graph_hash, bool through_prologue_memo = false\);",
                        norm(backend)) is not None,
         "the gateway's first declaration does not default the memo argument to false")
    need("M", re.search(r"const bool staged = through_prologue_memo \? graph_prologue_memo_prestage\(ctx, cgraph\) "
                        r": graph_prestage_leaf_tensors\(ctx, cgraph\);", b["gateway"]) is not None,
         "the gateway does not choose between the memo and the full pass on its last argument alone")

    # --- R: the replay ----------------------------------------------------------------------------------------------
    walk = b["walk"]
    replay = re.search(r"if \(replay\) \{ for \(const auto & effect : \*replay\) \{ "
                       r"stage_effect\(effect\.tensor, effect\.arm, effect\.source\); \} return all_staged; \}", walk)
    leafs = walk.find("for (int i = 0; i < cgraph->n_leafs; i++)")
    need("R", replay is not None and 0 <= replay.start() < leafs,
         "a replay does not run stage_effect for exactly its entries, in order, before the walk")
    stage_tensor = body_of(walk, r"auto stage_tensor = \[&\]\(const ggml_tensor \* tensor, const char \* source_desc\) \{")
    stage_effect = body_of(walk, r"auto stage_effect = \[&\]\(const ggml_tensor \* tensor, prestage_arm arm, "
                                 r"const char \* source_desc\) \{")
    need("R", stage_tensor is not None and stage_effect is not None, "no stage_tensor or stage_effect lambda")
    if stage_tensor is not None and stage_effect is not None:
        st, se = norm(stage_tensor), norm(stage_effect)
        need("R", all(a not in st for a in ("graph_input_stage(", "ensure_cached(", "ggml_sycl_get_staged_ptr_device(")),
             "stage_tensor reaches a stage action without stage_effect")
        need("R", re.search(r"if \(source_class == ggml_sycl_prestage_class::INPUT && tensor->name && "
                            r"tensor->name\[0\] != '\\0'\) \{ stage_effect\(tensor, prestage_arm::INPUT, source_desc\); "
                            r"return; \} stage_effect\(tensor, prestage_arm::CACHE, source_desc\); \}$", st) is not None,
             "stage_tensor does not end by sending a named INPUT to the INPUT action and anything else to the cache")
        need("R", re.match(r"\{ if \(record && recorded\.insert\(tensor\)\.second\) \{ "
                           r"record->push_back\(\{ tensor, arm, source_desc \}\); \}", se) is not None,
             "stage_effect does not record each tensor once, before its action")
        need("R", re.search(r"if \(arm == prestage_arm::INPUT\) \{ sycl::queue & q = \*ctx->stream\(\); "
                            r"void \* dev_ptr = ctx->graph_input_stage\(tensor, tensor->data, nbytes, q\); "
                            r"if \(!dev_ptr\) \{[^{}]*all_staged = false; return; \}", se) is not None and
             "cache->ensure_cached(" in se and
             re.search(r"ggml_sycl_get_staged_ptr_device\([^;]*; if \(staged\) \{[^{}]*\} else \{[^{}]*"
                       r"all_staged = false; \}", se) is not None,
             "stage_effect does not keep the three actions with their failure handling")
    need("R", re.search(PRESTAGE_HIT, b["prestage"]) is not None,
         "a hit does not replay the held list, or a failed replay keeps it")

    # --- N: a not-all-staged pass is not memoized ---------------------------------------------------------------------
    need("N", re.search(r"const bool staged = graph_prestage_leaf_tensors\(ctx, cgraph, nullptr, &effects\); "
                        r"memo\.record_prestage\(staged, std::move\(effects\)\); return staged; \}$", b["prestage"])
         is not None, "a miss does not record with the pass's own answer")

    # --- C: record containers only when armed ---------------------------------------------------------------------
    need("C", re.search(r"memo\.note_miss\(\); if \(!memo\.armed\(\)\) \{ "
                        r"return moe_segment_slot_key\(ctx, cgraph, signature, is_decode\); \} "
                        r"std::vector<graph_prologue_memo::identity> buffers;", b["slot_key"]) is not None and
         b["slot_key"].count("&buffers") == 1,
         "a slot key miss builds its buffer list without the memo armed")
    need("C", re.search(r"memo\.note_miss\(\); if \(!memo\.armed\(\)\) \{ "
                        r"return graph_prestage_leaf_tensors\(ctx, cgraph\); \} "
                        r"std::vector<graph_prologue_memo::effect> effects;", b["prestage"]) is not None,
         "a prestage miss builds its effect list without the memo armed")

    # --- I: invalidation -----------------------------------------------------------------------------------------------
    inv = body_of(common, r"\bbool invalidate_moe_segments\(\)\s*\{")
    need("I", inv is not None and norm(inv).startswith("{ prologue_memo.clear();"),
         "invalidate_moe_segments does not clear the memo before anything can return")
    swap = bodies["swap"]
    call = swap.find("ctx->invalidate_moe_segments();")
    need("I", call >= 0 and top_level(swap, call) and
         re.search(r"const bool segments_retired = ctx->invalidate_moe_segments\(\);", b["swap"]) is not None,
         "graph_staging_swap_retire does not call invalidate_moe_segments unconditionally")
    need("I", re.search(r"if \(sycl_ctx->optional_layout_epoch != epoch\) \{ sycl_ctx->prologue_memo\.clear\(\); "
                        r"if \(sycl_ctx->exec_graph", compute) is not None,
         "the optional-layout epoch path does not clear the memo unconditionally")
    dtor = b["dtor"]
    report_at = dtor.find("graph_prologue_memo_report(*this);")
    inv_at = dtor.find("invalidate_moe_segments();", max(report_at, 0))
    need("I", report_at >= 0 and inv_at > report_at, "the teardown does not report, then invalidate_moe_segments")
    need("I", re.search(r"\bgraph_prologue_memo prologue_memo;", c) is not None and
         re.search(r"(static|thread_local|\*|unique_ptr|shared_ptr)[^;]*\bprologue_memo\b\s*;", c) is None,
         "the memo is not a by-value member of the context")
    need("I", "using graph_prologue_memo = ggml_sycl::graph_prologue_memo::memo<ggml_tensor, ggml_backend_buffer_t>;"
         in c, "the context's memo type is not memo<ggml_tensor, ggml_backend_buffer_t>")

    # --- S: the slot key's buffers ------------------------------------------------------------------------------------
    full_key = b["full_key"]
    need("S", "if (buffers_out) { *buffers_out = std::move(seen); }" in full_key and
         full_key.find("if (buffers_out)") > full_key.find("mix_tensor(node->src[j]);") and
         "seen.push_back({ t->buffer, moe_segment_buffer_identity(t->buffer), moe_segment_buffer_base(t->buffer) });"
         in full_key, "the full slot key walk does not hand out the distinct buffers as it read them")
    need("S", re.search(r"if \(held\) \{ if \(held->signature == signature && held->is_decode == is_decode && "
                        r"memo\.slot_key_buffers_unchanged\(moe_segment_buffer_identity, moe_segment_buffer_base\)\) \{",
                        b["slot_key"]) is not None,
         "the slot key is served without checking every buffer's identity and base")
    need("S", re.search(r"return full; \} memo\.drop_parts\(\); \} memo\.note_miss\(\);", b["slot_key"]) is not None,
         "a held slot key whose check fails keeps the other parts")
    need("S", re.search(r"const ggml_sycl::graph_segment_cache::key full = "
                        r"moe_segment_slot_key\(ctx, cgraph, signature, is_decode, &buffers\); "
                        r"memo\.record_slot_key\(full, std::move\(buffers\)\); return full; \}$", b["slot_key"]) is not None,
         "a slot key miss does not record the full walk's key with its buffers")

    # --- V: verify -----------------------------------------------------------------------------------------------------
    for name, compare, part in (("signature", "signature == *held", "SIGNATURE"),
                                ("slot_key", "full == *held", "SLOT_KEY")):
        need("V", re.search(r"if \(!memo\.note_verified\(%s\)\) \{ graph_prologue_memo_mismatch\(ctx, gpm::part::%s\); \}"
                            % (re.escape(compare), part), b[name]) is not None,
             f"the {name} wrapper does not compare its full result with the held one in verify mode")
    need("V", re.search(PRESTAGE_HIT + r" std::vector<graph_prologue_memo::effect> effects; "
                        r"const bool staged = graph_prestage_leaf_tensors\(ctx, cgraph, nullptr, &effects\); "
                        r"if \(!memo\.note_verified\(replayed == staged && effects == \*held\)\) \{ "
                        r"graph_prologue_memo_mismatch\(ctx, gpm::part::PRESTAGE\); \}", b["prestage"]) is not None,
         "a verify-mode prestage hit does not run the replay, then the full pass, and compare both answers and lists")
    need("V", MISMATCH_TAG in b["mismatch"] and "GGML_LOG_WARN(" in b["mismatch"] and
         "ggml_sycl::graph_prologue_memo::part_name(which)" in b["mismatch"],
         "the mismatch line is not a WARN naming the part")

    # --- T: the teardown line -------------------------------------------------------------------------------------------
    need("T", re.search(r"if \(mode == gpm::mode::OFF \|\| "
                        r"\(mode != gpm::mode::VERIFY && !moe_hostpath_waits_enabled\(\)\)\) \{ return; \}",
                        b["report"]) is not None,
         "the teardown line does not print exactly in modes 1 and 2 under the trace switch or in mode 2")
    need("T", ('GGML_LOG_WARN("%s"' % TEARDOWN_LINE) in b["report"], "the teardown line's text changed")

    # --- D: the docs ------------------------------------------------------------------------------------------------------
    need("D", ENV_NAME in doc and TEARDOWN_LINE.replace("\\n", "") in doc and MISMATCH_TAG in doc,
         "docs/backend/sycl-env-vars.md does not document the switch, the teardown line and the mismatch line")
    return out


def load() -> tuple[str, str, str]:
    doc = DOC.read_text() if DOC.exists() else ""
    return blank_comments(BACKEND.read_text()), blank_comments(COMMON.read_text()), doc


def sub1(src: str, pattern: str, repl: str) -> str:
    out, n = re.subn(pattern, repl, src, count=1)
    if n != 1:
        raise SystemExit(f"self-test: mutant anchor missing: {pattern!r}")
    return out


def mutants(backend: str, common: str, doc: str):
    """(name, pin it must fail, sources) for each mutant."""
    def be(p, r):
        return (sub1(backend, p, r), common, doc)

    def co(p, r):
        return (backend, sub1(common, p, r), doc)

    yield "the key drops the publish generation", "K", be(r"k\.tenant_gen\s*= ggml_sycl_tenant_publish_gen\(\);", "")
    yield "the key takes the node count from the context", "K", be(r"k\.n_nodes\s*= cgraph->n_nodes;",
                                                                   "k.n_nodes = 0;")
    yield "the signature wrapper does not begin()", "U", be(
        r"(return ggml_sycl_graph_signature\(cgraph\);\s*\}\s*auto & memo = ctx->prologue_memo;)\s*"
        r"graph_prologue_memo_begin\(ctx, cgraph, is_decode\);", r"\1")
    yield "the slot key wrapper does not begin()", "U", be(
        r"(return moe_segment_slot_key\(ctx, cgraph, signature, is_decode\);\s*\}\s*auto & memo = ctx->prologue_memo;)\s*"
        r"graph_prologue_memo_begin\(ctx, cgraph, is_decode\);", r"\1")
    yield "the prestage wrapper does not begin()", "U", be(r"graph_prologue_memo_begin\(ctx, cgraph, %s\);" % PRESTAGE_PHASE,
                                                           "")
    yield "the prestage wrapper flips the phase", "U", be(r"graph_prologue_memo_begin\(ctx, cgraph, ctx->graph_phase",
                                                          "graph_prologue_memo_begin(ctx, cgraph, !ctx->graph_phase")
    yield "the memo arms without its condition", "A", be(r"\[ctx\]\s*\{\s*return graph_prologue_memo_may_arm\(ctx\);\s*\}",
                                                         "[] { return true; }")
    yield "the arming condition ignores the cross-card expert path", "A", be(
        r"if \(ggml_sycl_moe_multi_gpu_requested\(\)\) \{\s*return false;\s*\}", "")
    yield "the arming condition ignores the plan", "A", be(
        r"\|\|\s*ggml_sycl_current_plan_is_single_device_for_graph\(ctx->device\);", "|| true;")
    yield "a wrapper arms the memo by itself", "A", be(
        r"(auto & memo = ctx->prologue_memo;\s*)graph_prologue_memo_begin\(ctx, cgraph, is_decode\);",
        r"\1memo.begin(graph_prologue_memo_key(ctx, cgraph, is_decode), [] { return true; });")
    yield "the switch is read on every call", "M", be(r"static const ggml_sycl::graph_prologue_memo::mode mode =",
                                                      "const ggml_sycl::graph_prologue_memo::mode mode =")
    yield "mode 0 serves the signature memo", "M", be(
        r"if \(mode == gpm::mode::OFF \|\| !moe_segment_keyed_mode\(ctx, is_decode\)\)",
        "if (!moe_segment_keyed_mode(ctx, is_decode))")
    yield "mode 0 serves the slot key memo", "M", be(
        r"if \(mode == gpm::mode::OFF\) \{(\s*return moe_segment_slot_key)", r"if (false) {\1")
    yield "mode 0 serves the prestage memo", "M", be(
        r"if \(mode == gpm::mode::OFF\) \{(\s*return graph_prestage_leaf_tensors)", r"if (false) {\1")
    yield "the shared signature site calls the full hash", "M", be(
        r"graph_prologue_memo_signature\(sycl_ctx, cgraph, cached_is_decode\)", "ggml_sycl_graph_signature(cgraph)")
    yield "the keyed slot key skips the memo", "M", be(
        r"graph_prologue_memo_slot_key\(sycl_ctx, cgraph, graph_hash, is_decode_phase\)",
        "moe_segment_slot_key(sycl_ctx, cgraph, graph_hash, is_decode_phase)")
    yield "a second consumer passes the memo argument", "M", be(
        r"if \(!graph_prestage_or_decline\(sycl_ctx, cgraph, graph_hash\)\) \{(\s*sycl_ctx->invalidate_moe_segments\(\);)",
        r"if (!graph_prestage_or_decline(sycl_ctx, cgraph, graph_hash, true)) {\1")
    yield "the gateway always takes the memo", "M", be(
        r"through_prologue_memo \? graph_prologue_memo_prestage\(ctx, cgraph\) : graph_prestage_leaf_tensors\(ctx, cgraph\)",
        "graph_prologue_memo_prestage(ctx, cgraph)")
    yield "the replay skips the last entry", "R", be(r"for \(const auto & effect : \*replay\) \{",
                                                     "for (const auto & effect : *replay) { if (&effect == &replay->back()) "
                                                     "break;")
    yield "the replay falls into the walk", "R", be(
        r"(stage_effect\(effect\.tensor, effect\.arm, effect\.source\);\s*\})\s*return all_staged;", r"\1")
    yield "stage_tensor stages an INPUT itself", "R", be(
        r"stage_effect\(tensor, prestage_arm::INPUT, source_desc\);",
        "ctx->graph_input_stage(tensor, tensor->data, ggml_nbytes(tensor), *ctx->stream());")
    yield "stage_effect records every visit", "R", be(r"if \(record && recorded\.insert\(tensor\)\.second\)",
                                                      "if (record)")
    yield "the staging-cache failure no longer clears all_staged", "R", be(
        r"(Failed to stage %s tensor %s \(data=%p, %zu bytes\)\\n\", source_desc,\s*tensor->name, tensor->data, "
        r"nbytes\);)\s*all_staged = false;", r"\1")
    yield "a failed replay keeps the list", "R", be(r"if \(!replayed\) \{\s*memo\.drop_prestage\(\);\s*\}", "")
    yield "a miss records as staged", "N", be(r"memo\.record_prestage\(staged, std::move\(effects\)\);",
                                              "memo.record_prestage(true, std::move(effects));")
    yield "a slot key miss builds its buffers unarmed", "C", be(
        r"if \(!memo\.armed\(\)\) \{\s*return moe_segment_slot_key\(ctx, cgraph, signature, is_decode\);\s*\}", "")
    yield "a prestage miss builds its effect list unarmed", "C", be(
        r"if \(!memo\.armed\(\)\) \{\s*return graph_prestage_leaf_tensors\(ctx, cgraph\);\s*\}", "")
    yield "invalidate_moe_segments forgets the memo", "I", co(r"prologue_memo\.clear\(\);\s*(if \(!ggml_sycl_retire)",
                                                              r"\1")
    yield "invalidate_moe_segments clears after a failure can return", "I", co(
        r"prologue_memo\.clear\(\);\s*(if \(!ggml_sycl_retire_moe_graph_epoch\(this\)\) \{\s*moe_graphs_disabled = true;"
        r"\s*return false;\s*\})", r"\1 prologue_memo.clear();")
    yield "the swap retire invalidates conditionally", "I", be(
        r"const bool segments_retired\s*= ctx->invalidate_moe_segments\(\);",
        "const bool segments_retired = ctx->exec_graph ? ctx->invalidate_moe_segments() : true;")
    yield "the optional-layout path clears only with a graph", "I", be(
        r"sycl_ctx->prologue_memo\.clear\(\);\s*(if \(sycl_ctx->exec_graph \|\|)", r"\1")
    yield "the teardown does not report", "I", be(r"graph_prologue_memo_report\(\*this\);", "")
    yield "the memo is a pointer member", "I", co(r"\bgraph_prologue_memo\s+prologue_memo;",
                                                 "graph_prologue_memo * prologue_memo = nullptr;")
    yield "the slot key walk keeps its buffers", "S", be(r"if \(buffers_out\) \{\s*\*buffers_out = std::move\(seen\);\s*\}",
                                                         "")
    yield "the slot key is served without the buffer check", "S", be(
        r"&& held->is_decode == is_decode &&\s*memo\.slot_key_buffers_unchanged\(moe_segment_buffer_identity, "
        r"moe_segment_buffer_base\)", "&& held->is_decode == is_decode")
    yield "a failed buffer check keeps the other parts", "S", be(r"memo\.drop_parts\(\);", "")
    yield "a slot key miss records no buffers", "S", be(r"memo\.record_slot_key\(full, std::move\(buffers\)\);",
                                                        "memo.record_slot_key(full, {});")
    yield "verify skips the signature compare", "V", be(
        r"if \(!memo\.note_verified\(signature == \*held\)\)", "if (!memo.note_verified(true))")
    yield "verify skips the slot key compare", "V", be(
        r"if \(!memo\.note_verified\(full == \*held\)\)", "if (!memo.note_verified(true))")
    yield "verify does not run the replay", "V", be(
        r"const bool replayed = graph_prestage_leaf_tensors\(ctx, cgraph, held\);\s*if \(mode == gpm::mode::ON\) \{",
        "if (mode == gpm::mode::ON) { const bool replayed = graph_prestage_leaf_tensors(ctx, cgraph, held);")
    yield "verify compares the prestage list but not the replay's answer", "V", be(
        r"if \(!memo\.note_verified\(replayed == staged && effects == \*held\)\)",
        "if (!memo.note_verified(effects == *held))")
    yield "verify compares the answers but not the prestage list", "V", be(
        r"if \(!memo\.note_verified\(replayed == staged && effects == \*held\)\)",
        "if (!memo.note_verified(replayed == staged))")
    yield "the mismatch line is DEBUG", "V", be(r"GGML_LOG_WARN\(\s*\"\[PROLOGUE-MEMO\] mismatch",
                                                'GGML_LOG_DEBUG("[PROLOGUE-MEMO] mismatch')
    yield "the teardown line prints in mode 0", "T", be(r"if \(mode == gpm::mode::OFF \|\| \(", "if ((")
    yield "the teardown line prints without the trace", "T", be(r" && !moe_hostpath_waits_enabled\(\)\)\)", " && false))")
    yield "the teardown line drops a counter", "T", be(r"verified=%llu mismatches=%llu", "mismatches=%llu")
    yield "the docs lose the teardown line", "D", (backend, common, doc.replace("hits=%llu", "hits=%u"))


def main() -> int:
    backend, common, doc = load()
    if "--self-test" in sys.argv[1:]:
        bad = []
        n = 0
        for name, pin, sources in mutants(backend, common, doc):
            n += 1
            fired = [f for f in check(*sources) if f.startswith(pin + ":")]
            print(("PASS" if fired else "FAIL") + f": mutant '{name}' fails pin {pin}")
            if not fired:
                bad.append(name)
        if bad:
            print(f"self-test: {len(bad)} of {n} mutants not caught: {bad}")
            return 1
        print(f"self-test: all {n} mutants caught")
    failures = check(backend, common, doc)
    for f in failures:
        print("FAIL [pin " + f.replace(":", "]:", 1))
    if failures:
        return 1
    print("graph prologue memo source contract: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
