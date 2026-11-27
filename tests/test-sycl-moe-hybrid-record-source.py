#!/usr/bin/env python3
"""Source contract: how the event-chained hybrid decode route keeps its per-tensor record current, and what it counts.

Host-only: reads ggml-sycl.cpp, common.hpp, moe-hybrid-record.hpp and docs/backend/sycl-env-vars.md, builds nothing,
loads no model and touches no device.

The route (ggml_sycl_moe_hybrid_decode_event_chained) looks its tensor's record up at every decode op
(ggml_sycl_moe_hybrid_decode_record_current). The record is cached under a stamp of (replan epoch, the weight's expert
storage generation, selected rows); a miss rebuilds it (ggml_sycl_moe_hybrid_decode_record_build), which resolves
every expert of the tensor. GGML_SYCL_MOE_HYBRID_RECORD_TRACE counts that work per token, so a device run can say how
many records a token rebuilds, why, and how many experts each rebuild resolves. The host test
(tests/test-sycl-moe-hybrid-record.cpp) proves the bookkeeping; this gate holds where the backend calls it:

  T1  the lookup classifies its stamp with moe_hybrid_record_miss_classify, once, and only a NONE miss returns the
      held decision, and never a record a loan or an evict withdrew a slot from; the stamp's own current test is not
      asked a second time beside it;
  T2  the trace level is read once, in a function-local static of ggml_sycl_moe_hybrid_record_trace_level(), from
      GGML_SYCL_MOE_HYBRID_RECORD_TRACE; the name appears nowhere else in the backend;
  T3  every lookup is counted, with its miss, before the held decision returns; the lookup that finds no published
      plan counts itself before it returns; each build's outcome is counted after the build;
  T4  the build counts the experts it resolves at its one resolve, right before the call that resolves them;
  T5  the per-token line and the per-build line are WARN, each printed from one site, the per-build one only at
      level 2 and above; the per-token line's buffer is the 768 bytes the host test proves the widest line fits;
  T6  the header is plain host code: standard includes and moe-decode-hostpath.hpp only, no SYCL;
  T7  the env-var row names the variable, both line tags, the read-once rule and the token rule;
  T8  every move of a weight's expert storage generation goes through note_moe_storage_change(), which logs the
      expert the move changed: no other site writes the generation, each site that moves one expert's storage names
      that expert, and each site that clears a whole store names every expert (moe_storage_change_log::ALL). A site
      that named one expert for a cleared store would let a held record keep slots the clear withdrew;
  T9  a lookup that missed on the generation alone asks moe_hybrid_record_plan_patch (when the patch is on) and, for
      the experts the log names, patches the held record; a patch that stands is published by retaining what it
      replaced to a barrier on the queue, settling the stamp at the new generation and returning the patched record,
      ahead of the build; its count and its level-2 line come before that return;
  T10 the patch resolves only the changed experts, takes each slot from the build's slot rule, writes a copy of the
      held record (never the held one), checks the copy's shape with the build's rule, gives it a table of its own
      filled from every slot, a ready barrier over the held one's and the new events, records the block's residency
      only after every way out, hands the replaced leases and the held table back for retention, and waits for
      nothing;
  T11 the build takes its slots, its shape check and its table from those same helpers, the executor is chosen in the
      slot rule alone, and GGML_SYCL_MOE_HYBRID_RECORD_PATCH is read once;
  T12 an expert loan or evict (ggml_sycl_drop_lent_expert_views) withdraws only the record's slots that lease the
      entry it retired, from a copy it publishes in the held record's place (never the held one), counting them on the
      copy and moving their leases into the drop's retention; it keeps the stamp, and notes every expert it emptied,
      so the next lookup is a generation miss the patch serves. That lookup counts the withdrawn slots in the trace,
      once, between the held record's binding and the patch plan, and the replaced record's retained handles skip a
      withdrawn slot's empty lease.

Checks run on comment-blanked text (string literals kept). NOT VACUOUS: --self-test runs the gate on in-memory mutants
of the real sources. Each mutant names the pin that must catch it, and is reported if it survives, if another pin
catches it, or if its anchor no longer matches exactly once (NOT APPLIED). ctest registers it with --self-test.

WHAT THIS DOES NOT PROVE. It reads text. That the counts add up is the host test's job; what they read on Qwen3.8 is
the lead's device run.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
HEADER = ROOT / "ggml/src/ggml-sycl/moe-hybrid-record.hpp"
COMMON = ROOT / "ggml/src/ggml-sycl/common.hpp"
DOCS = ROOT / "docs/backend/sycl-env-vars.md"

CURRENT_SIG = "static bool ggml_sycl_moe_hybrid_decode_record_current("
BUILD_SIG = "static ggml_sycl::moe_decode_direct_outcome ggml_sycl_moe_hybrid_decode_record_build("
LEVEL_SIG = "static int ggml_sycl_moe_hybrid_record_trace_level() {"
LOOKUP_SIG = "static void ggml_sycl_moe_hybrid_record_trace_lookup("
ENV_NAME = "GGML_SYCL_MOE_HYBRID_RECORD_TRACE"

CLASSIFY = ("const ggml_sycl::moe_hybrid_record_miss miss = ggml_sycl::moe_hybrid_record_miss_classify(stamp, "
            "replan_epoch, storage_generation, selected_rows);")
HELD_RETURN = ("if (miss == ggml_sycl::MOE_HYBRID_RECORD_MISS_NONE) { *out = weight.moe_hybrid_decode[device]; "
               "return stamp.eligible && *out && (*out)->n_withdrawn == 0; }")
TRACE_READ = "const int trace = ggml_sycl_moe_hybrid_record_trace_level();"
TRACE_LOOKUP = "if (trace > 0) { ggml_sycl_moe_hybrid_record_trace_lookup(src0, miss,"
NO_PLAN = ("if (!cache || ggml_sycl_cache_plan_owner(cache)->entries.empty()) { if (trace > 0) { "
           "g_moe_hybrid_record_trace.no_plan(); } return false; }")
BUILD_CALL = "ggml_sycl_moe_hybrid_decode_record_build(ctx, src0, &built, &refusal);"
OUTCOME_COUNT = "g_moe_hybrid_record_trace.build_outcome(outcome);"
LEVEL_BODY = ("{ static const int level = [] { const char * env = std::getenv(\"GGML_SYCL_MOE_HYBRID_RECORD_TRACE\"); "
              "return env ? std::atoi(env) : 0; }(); return level; }")
RESOLVED_COUNT = ("if (ggml_sycl_moe_hybrid_record_trace_level() > 0) { "
                  "g_moe_hybrid_record_trace.resolved(all_experts.size()); }")
RESOLVE_CALL = "const ggml_sycl::moe_resolved_batch_result resolved = ggml_sycl::ggml_sycl_build_moe_resolved_batch("
TOKEN_WARN = 'GGML_LOG_WARN("[MOE-HYBRID-RECORD] %s\\n", line);'
TOKEN_BUFFER = "char line[768];"
BUILD_WARN = 'GGML_LOG_WARN("[MOE-HYBRID-RECORD-BUILD] '
BUILD_GUARD = "if (trace >= 2) {"
WAIT_CALLS = re.compile(r"\.wait\(|wait_and_throw\(|get_info<")

# T9: the lookup's patch path.
PATCH_HELD = "std::shared_ptr<const moe_hybrid_decode_record> & held = weight.moe_hybrid_decode[device];"
PATCH_PLAN = ("const ggml_sycl::moe_hybrid_record_patch_plan plan = ggml_sycl_moe_hybrid_record_patch_enabled() ? "
              "ggml_sycl::moe_hybrid_record_plan_patch(miss, stamp, held != nullptr, weight.moe_expert_storage_changes, "
              "storage_generation, &changed) : ggml_sycl::MOE_HYBRID_RECORD_PATCH_NONE;")
PATCH_CALL = ("if (plan == ggml_sycl::MOE_HYBRID_RECORD_PATCH_EXPERTS) { patched_ok = "
              "ggml_sycl_moe_hybrid_decode_record_patch(ctx, src0, *held, changed, &patched, &replaced); }")
PATCH_COUNT = "g_moe_hybrid_record_trace.patch(plan, patched_ok, changed.size());"
PATCH_WARN = 'GGML_LOG_WARN("[MOE-HYBRID-RECORD-PATCH] '
PATCH_PUBLISH = ("if (patched_ok) { ggml_sycl::retain_handles_until_event(std::move(replaced), "
                 "ctx.stream()->ext_oneapi_submit_barrier()); (void) ggml_sycl::moe_decode_direct_stamp_settle(stamp, "
                 "replan_epoch, storage_generation, selected_rows, static_cast<int>(patched->layout), "
                 "ggml_sycl::MOE_DECODE_DIRECT_OUTCOME_ELIGIBLE); held = std::move(patched); *out = held; "
                 "return true; }")

# T10: the patch itself.
PATCH_SIG = "static bool ggml_sycl_moe_hybrid_decode_record_patch("
PATCH_DECL = ("static bool ggml_sycl_moe_hybrid_decode_record_patch(ggml_backend_sycl_context & ctx, "
              "const ggml_tensor * src0, const moe_hybrid_decode_record & held, const std::vector<int32_t> & changed,")
PATCH_STEPS = (
    ("its resolve of the changed experts alone",
     "ggml_sycl::ggml_sycl_build_moe_resolved_batch(src0, device, changed.data(), changed.size(), changed.size(), "
     "held.layout,"),
    ("each slot from the build's slot rule",
     "ggml_sycl_moe_hybrid_decode_record_slot(operand, device, held.layout, held.all_host, &fresh[i], ready, &why)"),
    ("its copy of the held record", "auto patched = std::make_shared<moe_hybrid_decode_record>(held);"),
    ("the install of the moved slots into the copy",
     "ggml_sycl::moe_hybrid_record_install(patched->device_weight, patched->host_weight, patched->lease, changed, "
     "fresh, &leases)"),
    ("the copy's shape",
     "const ggml_sycl::moe_hybrid_record_shape shape = ggml_sycl::moe_hybrid_record_shape_of(patched->device_weight, "
     "patched->host_weight, patched->lease);"),
    ("the build's shape rule over it",
     "if (ggml_sycl::moe_hybrid_record_shape_refusal(patched->all_host, shape) != nullptr) { return false; }"),
    ("the copy's withdrawn count cleared, the rule having found no empty slot", "patched->n_withdrawn = 0;"),
    ("a table of the copy's own, filled from every slot",
     "patched->table = {}; if (!ggml_sycl_moe_hybrid_decode_table_fill(queue, device, extra->model_id, "
     "patched->device_weight, shape.substitute, &patched->table)) { return false; }"),
    ("the held record's ready event among the barrier's", "if (held.has_ready) { ready.push_back(held.ready); }"),
    ("the copy's ready barrier", "patched->ready = queue.ext_oneapi_submit_barrier(ready);"),
    ("the block's residency",
     "patched->route_counts->set_block_resident(patched->route_block, "
     "static_cast<int>(patched->n_as - shape.n_host));"),
    ("the replaced leases handed back",
     "replaced->insert(replaced->end(), std::make_move_iterator(leases.begin()), "
     "std::make_move_iterator(leases.end()));"),
    ("the held table handed back", "if (held.table.valid()) { replaced->push_back(held.table); }"),
    ("the copy's publication", "*out = std::move(patched);"),
)

# T11: the rules the build and the patch share.
SLOT_SIG = "static ggml_sycl::moe_decode_direct_outcome ggml_sycl_moe_hybrid_decode_record_slot("
BUILD_STEPS = (
    ("each slot from the slot rule",
     "if (ggml_sycl_moe_hybrid_decode_record_slot(operand, device, layout, all_host, &slot, ready, reason) != "
     "ggml_sycl::MOE_DECODE_DIRECT_OUTCOME_ELIGIBLE) { return ggml_sycl::MOE_DECODE_DIRECT_OUTCOME_REFUSED; }"),
    ("its shape",
     "const ggml_sycl::moe_hybrid_record_shape shape = ggml_sycl::moe_hybrid_record_shape_of(record->device_weight, "
     "record->host_weight, record->lease);"),
    ("the shape rule over it",
     "if (const char * why = ggml_sycl::moe_hybrid_record_shape_refusal(all_host, shape)) { return refuse(why); }"),
    ("its table from the shared fill",
     "ggml_sycl_moe_hybrid_decode_table_fill(queue, device, extra->model_id, record->device_weight, "
     "shape.substitute, &record->table)"),
)
PATCH_ENV = "GGML_SYCL_MOE_HYBRID_RECORD_PATCH"
PATCH_LEVEL_SIG = "static bool ggml_sycl_moe_hybrid_record_patch_enabled() {"
PATCH_LEVEL_BODY = ('{ static const bool enabled = [] { const char * env = std::getenv("' + PATCH_ENV + '"); '
                    "return env == nullptr || std::atoi(env) != 0; }(); return enabled; }")

DOC_NEEDLES = ("`GGML_SYCL_MOE_HYBRID_RECORD_TRACE=1\\|2`", "[MOE-HYBRID-RECORD]", "[MOE-HYBRID-RECORD-BUILD]",
               "[MOE-HYBRID-RECORD-PATCH]", "patches=", "withdrawn=", "Read once", "lower block")
PATCH_DOC_NEEDLES = ("`GGML_SYCL_MOE_HYBRID_RECORD_PATCH=0`", "Read once", "every expert")
HEADER_INCLUDE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]', re.M)
HEADER_ALLOWED = {"moe-decode-hostpath.hpp"}

# T8: the one function that moves the generation, and what each site that calls it names.
NOTE_SIG = "void note_moe_storage_change(int32_t expert_id) {"
NOTE_BODY = ("{ ++moe_expert_storage_generation; "
             "moe_expert_storage_changes.note(moe_expert_storage_generation, expert_id); }")
GENERATION_DECL = "uint64_t moe_expert_storage_generation = 1;"
# A write of the generation: an increment or decrement, or an assignment (an `==` compare is not one).
GENERATION_WRITE = re.compile(r"(?:\+\+|--)\s*(?:[\w.]|->)*\bmoe_expert_storage_generation\b"
                              r"|\bmoe_expert_storage_generation\s*(?:\+\+|--|[-+]?=(?!=))")
ALL = "ggml_sycl::moe_storage_change_log::ALL"
# (file, enclosing signature, the call as it reads, squashed): one expert's storage moved, or a whole store went.
NOTE_SITES = (
    ("common", "bool remember_moe_storage_handle(int expert_id,",
     "existing = std::move(record); note_moe_storage_change(expert_id); return true;"),
    ("common", "bool remember_moe_storage_handle(int expert_id,",
     "records.push_back(std::move(record)); note_moe_storage_change(expert_id); return true;"),
    ("common", "bool forget_moe_storage_handle_on_device(int expert_id, ggml_layout_mode layout, int owner_device) {",
     "if (erased) { note_moe_storage_change(expert_id); } return erased;"),
    ("common", "bool take_moe_storage_handle_on_device(int expert_id,",
     "mark_moe_device_table_stale(owner_device); } } } note_moe_storage_change(expert_id); return true;"),
    ("common", "void clear_moe_storage_handles() {",
     "moe_expert_storage_handles.clear(); note_moe_storage_change(" + ALL + ");"),
    ("common", "void clear_moe_storage_handles_for_owner(int owner_device) {",
     "if (erased) { note_moe_storage_change(" + ALL + "); }"),
    # Every expert whose record slot the drop emptied, and the one whose storage record or table slot it dropped.
    ("backend", "static void ggml_sycl_drop_lent_expert_views(",
     "if (stored && std::find(emptied.begin(), emptied.end(), expert_id) == emptied.end()) { "
     "emptied.push_back(expert_id); } for (const int32_t e : emptied) { w.note_moe_storage_change(e); }"),
    # A mutation names every expert, whatever its forget loop withdrew: the rewrite changed them all, and in every
    # layout, while the loop forgets only the AOS records on this device.
    ("backend", "static void ggml_sycl_invalidate_backend_weight_mutation(",
     "ctx->device); } extra->weight().note_moe_storage_change(" + ALL + ");"),
)

# T12: an expert loan's or evict's drop withdraws the record's slots that lease the retired entry, from a copy.
DROP_SIG = "static void ggml_sycl_drop_lent_expert_views("
WITHDRAW_SIG = "static bool ggml_sycl_moe_hybrid_decode_record_withdraw("
WITHDRAW_DECL = ("static bool ggml_sycl_moe_hybrid_decode_record_withdraw(std::shared_ptr<const moe_hybrid_decode_record> "
                 "& held,")
DROP_WITHDRAW = ("if (ggml_sycl_moe_hybrid_decode_record_withdraw(w.moe_hybrid_decode[device], device, withdrawn, "
                 "retained, emptied)) { table_dropped = true; }")
DROP_NOTES = NOTE_SITES[6][2]
DROP_RETIRED = ("moe_hybrid_decode_stamp", "moe_hybrid_decode[device].reset()",
                "ggml_sycl_moe_hybrid_decode_record_append_handles(")
WITHDRAW_MATCH = "return h.is_weight() && h.device() == device && h.key() == withdrawn;"
# The lookup that replaces a withdrawn record counts its withdrawn slots: on a device, the trace's one sign that a drop
# matched a slot by the retired entry. A drop that matched none still moves the generation for the expert whose storage
# record it dropped, so the patch and the tokens look the same either way.
WITHDRAWN_COUNT = ("if (trace > 0 && held && held->n_withdrawn > 0) { "
                   "g_moe_hybrid_record_trace.withdrawn(held->n_withdrawn); }")
# The replaced record's handles, retained to a barrier, skip a withdrawn slot's empty lease.
APPEND_SIG = "static void ggml_sycl_moe_hybrid_decode_record_append_handles("
APPEND_LEASES = "for (const ggml_sycl::mem_handle & h : record.lease) { if (h.valid()) { out.push_back(h); } }"
WITHDRAW_STEPS = (
    ("test that a slot leases the entry",
     "if (!held || std::none_of(held->lease.begin(), held->lease.end(), on_entry)) { return false; }"),
    ("copy of the held record", "auto copy = std::make_shared<moe_hybrid_decode_record>(*held);"),
    ("withdrawal from the copy, counted on it, its leases into the drop's retention",
     "copy->n_withdrawn += ggml_sycl::moe_hybrid_record_withdraw(copy->device_weight, copy->host_weight, copy->lease, "
     "on_entry, &retained, &emptied);"),
    ("publication of the copy", "held = std::move(copy);"),
)
# The fixture's store clear, inside a lambda of a private test function: matched in the squashed file text.
FIXTURE_CLEAR = ("extra.weight().moe_expert_storage_handles.clear(); extra.weight().note_moe_storage_change("
                 + ALL + ");")


class ContractError(AssertionError):
    pass


class MutantNotApplied(Exception):
    pass


_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'", re.DOTALL)


def blank_comments(text: str) -> str:
    """Comments become spaces (newlines kept); string and character literals are kept as written."""

    def repl(m: re.Match) -> str:
        t = m.group(0)
        if t.startswith("//") or t.startswith("/*"):
            return re.sub(r"[^\n]", " ", t)
        return t

    return _COMMENT.sub(repl, text)


def squash(text: str) -> str:
    t = re.sub(r"\s+", " ", text)
    t = re.sub(r"\( ", "(", t)
    return re.sub(r" \)", ")", t)


def body(code: str, sig: str, pin: str) -> str:
    """The squashed `{ ... }` body of the one definition that starts with `sig`."""
    n = code.count(sig)
    if n != 1:
        raise ContractError(f"FAIL [pin {pin}]: `{sig}` appears {n} times, not once")
    at = code.find(sig)
    open_at = code.find("{", at + len(sig) - 1 if sig.endswith("{") else at)
    depth = 0
    k = open_at
    while k < len(code):
        c = code[k]
        if c == '"':
            k += 1
            while k < len(code) and code[k] != '"':
                k += 2 if code[k] == "\\" else 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return squash(code[open_at : k + 1])
        k += 1
    raise ContractError(f"FAIL [pin {pin}]: the body of `{sig}` is unbalanced")


def matching_close(text: str, open_at: int, open_ch: str, close_ch: str) -> int:
    """The index of the bracket that closes `text[open_at]`, string literals skipped; -1 when it is unbalanced."""
    depth = 0
    k = open_at
    while k < len(text):
        c = text[k]
        if c == '"':
            k += 1
            while k < len(text) and text[k] != '"':
                k += 2 if text[k] == "\\" else 1
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return k
        k += 1
    return -1


def definition_spans(sq: str, sig: str) -> list[tuple[int, int]]:
    """The `{...}` spans of every definition in squashed `sq` whose text starts with `sig`; declarations are skipped."""
    spans = []
    at = sq.find(sig)
    while at >= 0:
        params = sq.find("(", at)
        close = matching_close(sq, params, "(", ")") if params >= 0 else -1
        if close >= 0 and sq[close + 1 : close + 3] == " {":
            open_at = close + 2
            end = matching_close(sq, open_at, "{", "}")
            if end >= 0:
                spans.append((open_at, end))
        at = sq.find(sig, at + 1)
    return spans


def once(text: str, needle: str, pin: str, what: str) -> int:
    n = text.count(needle)
    if n != 1:
        raise ContractError(f"FAIL [pin {pin}]: {what}: `{needle[:110]}` appears {n} times, not once")
    return text.find(needle)


def check(backend_raw: str, header_raw: str, docs_raw: str, common_raw: str) -> None:
    code = blank_comments(backend_raw)
    current = body(code, CURRENT_SIG, "T1")
    build = body(code, BUILD_SIG, "T4")

    # T1: one classification, and only NONE returns the held decision.
    at_classify = once(current, CLASSIFY, "T1", "the lookup's stamp classification")
    at_held = once(current, HELD_RETURN, "T1", "the held decision's return under a NONE miss")
    if at_held < at_classify:
        raise ContractError("FAIL [pin T1]: the held decision returns before the stamp is classified")
    if "moe_decode_direct_stamp_current(" in current:
        raise ContractError("FAIL [pin T1]: the lookup asks the stamp's current test beside the classification, so "
                            "the two can disagree on what the held record answers for")
    if current.count("return stamp.eligible") != 1:
        raise ContractError("FAIL [pin T1]: the held decision returns from more than one place")

    # T2: the level, read once, from the one variable.
    level = body(code, LEVEL_SIG, "T2")
    if level != LEVEL_BODY:
        raise ContractError(f"FAIL [pin T2]: ggml_sycl_moe_hybrid_record_trace_level() is not the read-once static "
                            f"of {ENV_NAME}: `{level[:160]}`")
    if squash(code).count(ENV_NAME) != 1:
        raise ContractError(f"FAIL [pin T2]: {ENV_NAME} is read in more than one place")

    # T3: every lookup counted before the held return, the no-plan return counted, each build's outcome counted.
    at_read = once(current, TRACE_READ, "T3", "the lookup's one read of the trace level")
    at_lookup = once(current, TRACE_LOOKUP, "T3", "the lookup count")
    if not at_classify < at_read < at_lookup < at_held:
        raise ContractError("FAIL [pin T3]: the lookup is not counted between its classification and the held return")
    at_no_plan = once(current, NO_PLAN, "T3", "the no-plan return and its count")
    at_build = once(current, BUILD_CALL, "T3", "the record build")
    at_outcome = once(current, OUTCOME_COUNT, "T3", "the build's outcome count")
    if not at_held < at_no_plan < at_build < at_outcome:
        raise ContractError("FAIL [pin T3]: the no-plan count, the build and its outcome count are out of order")
    if current.count("return false;") != 1:
        raise ContractError("FAIL [pin T3]: the lookup has a `return false;` besides the no-plan return, which the "
                            "trace does not count")
    if current[at_build:at_outcome].count("return") != 0:
        raise ContractError("FAIL [pin T3]: the lookup returns between the build and its outcome count")

    # T4: the build counts what it resolves, right before resolving it.
    at_count = once(build, RESOLVED_COUNT, "T4", "the build's count of the experts it resolves")
    at_resolve = once(build, RESOLVE_CALL, "T4", "the build's resolve")
    if not at_count < at_resolve or build[at_count + len(RESOLVED_COUNT) : at_resolve].strip():
        raise ContractError("FAIL [pin T4]: the resolved-experts count is not the statement right before the resolve")

    # T5: the two lines are WARN, printed once each, the per-build one at level 2.
    lookup = body(code, LOOKUP_SIG, "T5")
    once(lookup, TOKEN_WARN, "T5", "the per-token WARN line")
    once(lookup, TOKEN_BUFFER, "T5", "the per-token line's 768-byte buffer")
    sq = squash(code)
    if sq.count("[MOE-HYBRID-RECORD]") != 1 or sq.count("[MOE-HYBRID-RECORD-BUILD]") != 1:
        raise ContractError("FAIL [pin T5]: a hybrid record trace line is printed from more than one site")
    at_warn = once(current, BUILD_WARN, "T5", "the per-build WARN line")
    at_guard = current.find(BUILD_GUARD, at_outcome)
    if not at_outcome < at_guard < at_warn or current[at_guard:at_warn].count("}") != 0:
        raise ContractError("FAIL [pin T5]: the per-build line is not printed under its level-2 guard after the "
                            "outcome count")
    if WAIT_CALLS.search(lookup):
        raise ContractError("FAIL [pin T5]: the per-token count waits on the device")

    # T6: the header stays plain host code.
    header = blank_comments(header_raw)
    for m in HEADER_INCLUDE.finditer(header):
        kind, name = m.group(1), m.group(2)
        if kind == '"' and name not in HEADER_ALLOWED:
            raise ContractError(f"FAIL [pin T6]: moe-hybrid-record.hpp includes the project header `{name}`")
        if kind == "<" and ("sycl" in name or "/" in name):
            raise ContractError(f"FAIL [pin T6]: moe-hybrid-record.hpp includes `<{name}>`")
    if re.search(r"\bsycl::", header):
        raise ContractError("FAIL [pin T6]: moe-hybrid-record.hpp names a SYCL type")

    # T7: the env-var row.
    rows = [line for line in docs_raw.splitlines() if line.startswith(f"| `{ENV_NAME}")]
    if len(rows) != 1:
        raise ContractError(f"FAIL [pin T7]: docs/backend/sycl-env-vars.md has {len(rows)} rows for {ENV_NAME}, "
                            f"not one")
    for needle in DOC_NEEDLES:
        if needle not in rows[0]:
            raise ContractError(f"FAIL [pin T7]: the {ENV_NAME} row does not say `{needle}`")
    patch_rows = [line for line in docs_raw.splitlines() if line.startswith(f"| `{PATCH_ENV}")]
    if len(patch_rows) != 1 or not all(needle in patch_rows[0] for needle in PATCH_DOC_NEEDLES):
        raise ContractError(f"FAIL [pin T7]: docs/backend/sycl-env-vars.md has no one {PATCH_ENV} row that says "
                            f"{PATCH_DOC_NEEDLES}")

    # T9: the lookup's patch path, ahead of the build, published only by retaining what it replaced.
    at_patch_held = once(current, PATCH_HELD, "T9", "the held record's one binding in the lookup")
    at_plan = once(current, PATCH_PLAN, "T9", "the patch plan, asked only when the patch is on")
    at_call = once(current, PATCH_CALL, "T9", "the patch, only for the experts the log names")
    at_count = once(current, PATCH_COUNT, "T9", "the patch's count")
    at_patch_warn = once(current, PATCH_WARN, "T9", "the per-patch WARN line")
    at_publish = once(current, PATCH_PUBLISH, "T9", "the patched record's publication")
    if not at_no_plan < at_patch_held < at_plan < at_call < at_count < at_patch_warn < at_publish < at_build:
        raise ContractError("FAIL [pin T9]: the patch path is not plan, patch, count, line, publish, between the "
                            "no-plan return and the build")
    patch_guard = current.find(BUILD_GUARD, at_count)
    if not at_count < patch_guard < at_patch_warn or current[patch_guard:at_patch_warn].count("}") != 0:
        raise ContractError("FAIL [pin T9]: the per-patch line is not printed under its level-2 guard")
    if current.count("return true;") != 1 or "return true;" not in PATCH_PUBLISH:
        raise ContractError("FAIL [pin T9]: the lookup returns true other than by publishing a patch")

    # T10: the patch.
    patch = body(code, PATCH_SIG, "T10")
    if PATCH_DECL not in squash(code):
        raise ContractError("FAIL [pin T10]: the patch does not take the held record as a const reference, so it "
                            "could write the record a CPU job or a queued copy still reads")
    last = -1
    for what, needle in PATCH_STEPS:
        at = once(patch, needle, "T10", f"the patch's {what}")
        if at < last:
            raise ContractError(f"FAIL [pin T10]: the patch's {what} is out of order")
        last = at
    at_resident = patch.find(PATCH_STEPS[9][1])
    if patch.rfind("return false;") > at_resident:
        raise ContractError("FAIL [pin T10]: the patch can still give up after it recorded the block's residency, "
                            "leaving the count of a record it did not publish")
    if WAIT_CALLS.search(patch):
        raise ContractError("FAIL [pin T10]: the patch waits on the device")

    # T11: one slot rule, one shape rule and one table fill for the build and the patch; the switch read once.
    slot_rule = body(code, SLOT_SIG, "T11")
    for what, needle in BUILD_STEPS:
        once(build, needle, "T11", f"the build's {what}")
    if ("choose_moe_batch_executor(" not in slot_rule or "choose_moe_batch_executor(" in build
            or "choose_moe_batch_executor(" in patch):
        raise ContractError("FAIL [pin T11]: an expert's executor is chosen outside the slot rule, so the build and "
                            "the patch could place a moved expert apart")
    if body(code, PATCH_LEVEL_SIG, "T11") != PATCH_LEVEL_BODY:
        raise ContractError(f"FAIL [pin T11]: ggml_sycl_moe_hybrid_record_patch_enabled() is not the read-once "
                            f"static of {PATCH_ENV}")
    if squash(code).count(PATCH_ENV) != 1:
        raise ContractError(f"FAIL [pin T11]: {PATCH_ENV} is read in more than one place")

    # T8: one function moves the generation and logs the expert; each site names what its move changed.
    common_code = blank_comments(common_raw)
    common_sq = squash(common_code)
    backend_sq = squash(code)
    notes = definition_spans(common_sq, NOTE_SIG)
    if len(notes) != 1 or common_sq[notes[0][0] : notes[0][1] + 1] != NOTE_BODY:
        raise ContractError("FAIL [pin T8]: note_moe_storage_change() is not defined once as the generation's one "
                            "increment followed by the change log's note of it")
    common_writes = [m.start() for m in GENERATION_WRITE.finditer(common_sq)]
    in_note = [w for w in common_writes if notes[0][0] < w < notes[0][1]]
    declared = common_sq.count(GENERATION_DECL)
    if len(in_note) != 1 or declared != 1 or len(common_writes) != 2:
        raise ContractError(f"FAIL [pin T8]: common.hpp writes moe_expert_storage_generation {len(common_writes)} "
                            f"times; only its declaration and note_moe_storage_change() may, so a bump elsewhere "
                            f"would move the generation without a log entry")
    backend_writes = GENERATION_WRITE.findall(backend_sq)
    if backend_writes:
        raise ContractError(f"FAIL [pin T8]: ggml-sycl.cpp writes moe_expert_storage_generation directly "
                            f"(`{backend_writes[0]}`), not through note_moe_storage_change()")
    texts = {"common": common_sq, "backend": backend_sq}
    for file, sig, call in NOTE_SITES:
        sq = texts[file]
        if sq.count(call) != 1:
            raise ContractError(f"FAIL [pin T8]: `{call}` appears {sq.count(call)} times in {file}, not once")
        at = sq.find(call)
        if not any(open_at < at < end for open_at, end in definition_spans(sq, sig)):
            raise ContractError(f"FAIL [pin T8]: `{call}` is not in a definition of `{sig}`")
    if backend_sq.count(FIXTURE_CLEAR) != 1:
        raise ContractError("FAIL [pin T8]: the fixture's store clear does not note every expert")
    calls = common_sq.count("note_moe_storage_change(") + backend_sq.count("note_moe_storage_change(")
    # The definition, the six common.hpp sites, the two backend sites and the fixture's.
    if calls != 1 + len(NOTE_SITES) + 1:
        raise ContractError(f"FAIL [pin T8]: note_moe_storage_change( appears {calls} times, not "
                            f"{1 + len(NOTE_SITES) + 1}: a call site this gate does not read names an expert unchecked")

    # T12: a loan's or an evict's drop withdraws the slots that lease the retired entry, from a copy, and notes them.
    drop = body(code, DROP_SIG, "T12")
    at_withdraw = once(drop, DROP_WITHDRAW, "T12", "the drop's withdrawal of the slots that lease the retired entry")
    if not at_withdraw < drop.find(DROP_NOTES):
        raise ContractError("FAIL [pin T12]: the drop notes the experts it changed before it knows which record slots "
                            "it emptied, so an emptied slot can go unnoted and its stale table be served")
    for retired in DROP_RETIRED:
        if retired in drop:
            raise ContractError(f"FAIL [pin T12]: the drop still drops the whole record or its stamp (`{retired}`), "
                                f"so every expert loan or evict rebuilds the record over every expert")
    if WITHDRAW_DECL not in backend_sq:
        raise ContractError("FAIL [pin T12]: the withdrawal does not take the published record as a pointer to const, "
                            "so it could write the record a CPU job or a queued table copy still reads")
    spans = definition_spans(backend_sq, WITHDRAW_SIG)
    if len(spans) != 1:
        raise ContractError(f"FAIL [pin T12]: `{WITHDRAW_SIG}` is defined {len(spans)} times, not once")
    withdraw = backend_sq[spans[0][0] : spans[0][1] + 1]
    once(withdraw, WITHDRAW_MATCH, "T12", "the withdrawal's match of the retired entry by its key, as the drop's")
    last = -1
    for what, needle in WITHDRAW_STEPS:
        at = once(withdraw, needle, "T12", f"the withdrawal's {what}")
        if at < last:
            raise ContractError(f"FAIL [pin T12]: the withdrawal's {what} is out of order")
        last = at
    if "const_cast" in withdraw or WAIT_CALLS.search(withdraw):
        raise ContractError("FAIL [pin T12]: the withdrawal writes the published record or waits on the device")
    at_withdrawn = once(current, WITHDRAWN_COUNT, "T12", "the lookup's count of the withdrawn slots it refills")
    if not at_patch_held < at_withdrawn < at_plan:
        raise ContractError("FAIL [pin T12]: the withdrawn slots are not counted between the held record's binding and "
                            "the patch plan, the one place every lookup that replaces the record passes exactly once")
    appends = definition_spans(backend_sq, APPEND_SIG)
    if len(appends) != 1 or APPEND_LEASES not in backend_sq[appends[0][0] : appends[0][1] + 1]:
        raise ContractError("FAIL [pin T12]: the replaced record's handle list does not skip a withdrawn slot's empty "
                            "lease")


def self_test(raws: dict[str, str]) -> int:
    failures: list[str] = []
    n_mutants = 0

    def mutate(file: str, edits: list) -> dict[str, str]:
        out = dict(raws)
        text = out[file]
        for old, new in edits:
            if text.count(old) != 1:
                raise MutantNotApplied(f"anchor matches {text.count(old)} times in {file}: {old[:70]!r}")
            text = text.replace(old, new, 1)
        out[file] = text
        return out

    def expect_fail(name: str, pin: str, file: str, edits: list) -> None:
        nonlocal n_mutants
        n_mutants += 1
        try:
            mutated = mutate(file, edits)
        except MutantNotApplied as e:
            failures.append(f"{name}: NOT APPLIED ({e})")
            print(f"  mutant {name}: NOT APPLIED ({e})")
            return
        try:
            check(mutated["backend"], mutated["header"], mutated["docs"], mutated["common"])
        except ContractError as e:
            if f"[pin {pin}]" not in str(e):
                failures.append(f"{name}: caught, but not by pin {pin}: {str(e)[:140]}")
                print(f"  mutant {name}: caught by the WRONG pin ({str(e)[:120]}...)")
                return
            print(f"  mutant {name}: caught (pin {pin})")
            return
        failures.append(f"{name}: survived")
        print(f"  mutant {name}: SURVIVED")

    B, H, D = "backend", "header", "docs"
    held = "    if (miss == ggml_sycl::MOE_HYBRID_RECORD_MISS_NONE) {"
    expect_fail("held-on-generation-miss", "T1", B,
                [(held, "    if (miss == ggml_sycl::MOE_HYBRID_RECORD_MISS_NONE ||\n"
                        "        miss == ggml_sycl::MOE_HYBRID_RECORD_MISS_GENERATION) {")])
    expect_fail("held-on-old-test", "T1", B,
                [(held, "    if (ggml_sycl::moe_decode_direct_stamp_current(stamp, replan_epoch, storage_generation,\n"
                        "                                                   selected_rows)) {")])
    expect_fail("level-read-per-call", "T2", B,
                [("    static const int level = [] {\n        const char * env = std::getenv(\"" + ENV_NAME,
                  "    const int level = [] {\n        const char * env = std::getenv(\"" + ENV_NAME)])
    expect_fail("env-read-twice", "T2", B,
                [("    const int trace = ggml_sycl_moe_hybrid_record_trace_level();",
                  "    const int trace = std::getenv(\"GGML_SYCL_MOE_HYBRID_RECORD_TRACE\") ? 1 : 0;")])
    expect_fail("lookup-not-counted", "T3", B,
                [("        ggml_sycl_moe_hybrid_record_trace_lookup(src0, miss,",
                  "        (void) ggml_sycl_moe_hybrid_record_trace_lookup;\n        (void) (src0, miss,")])
    expect_fail("no-plan-not-counted", "T3", B,
                [("        if (trace > 0) {\n            g_moe_hybrid_record_trace.no_plan();\n        }\n", "")])
    expect_fail("outcome-not-counted", "T3", B,
                [("        g_moe_hybrid_record_trace.build_outcome(outcome);\n", "")])
    expect_fail("resolve-not-counted", "T4", B,
                [("        g_moe_hybrid_record_trace.resolved(all_experts.size());\n", "")])
    expect_fail("token-line-at-info", "T5", B,
                [('GGML_LOG_WARN("[MOE-HYBRID-RECORD] %s\\n", line);',
                  'GGML_LOG_INFO("[MOE-HYBRID-RECORD] %s\\n", line);')])
    expect_fail("token-buffer-shrunk", "T5", B,
                [("        char line[768];\n        ggml_sycl::moe_hybrid_record_line(",
                  "        char line[512];\n        ggml_sycl::moe_hybrid_record_line(")])
    expect_fail("build-line-at-info", "T5", B,
                [('GGML_LOG_WARN(\n                "[MOE-HYBRID-RECORD-BUILD] ',
                  'GGML_LOG_INFO(\n                "[MOE-HYBRID-RECORD-BUILD] ')])
    expect_fail("build-line-at-level-1", "T5", B,
                [("        if (trace >= 2) {\n            GGML_LOG_WARN(\n                \"[MOE-HYBRID-RECORD-BUILD] ",
                  "        if (trace >= 1) {\n            GGML_LOG_WARN(\n                \"[MOE-HYBRID-RECORD-BUILD] ")])
    expect_fail("header-includes-sycl", "T6", H, [("#include <cstdio>", "#include <sycl/sycl.hpp>\n#include <cstdio>")])
    expect_fail("header-includes-common", "T6", H, [("#include <cstdio>", '#include "common.hpp"\n#include <cstdio>')])
    expect_fail("docs-row-dropped", "T7", D, [(f"| `{ENV_NAME}=1\\|2`", f"| `{ENV_NAME}_OLD=1\\|2`")])
    expect_fail("docs-row-lacks-build-line", "T7", D, [("`[MOE-HYBRID-RECORD-BUILD]", "`[MOE-HYBRID-BUILD]")])
    expect_fail("docs-row-lacks-patch-line", "T7", D,
                [("`[MOE-HYBRID-RECORD-PATCH] token=T", "`[MOE-HYBRID-PATCH] token=T")])
    expect_fail("docs-patch-row-dropped", "T7", D,
                [("| `GGML_SYCL_MOE_HYBRID_RECORD_PATCH=0`", "| `GGML_SYCL_MOE_HYBRID_RECORD_NOPATCH=0`")])

    # T9: the lookup's patch path.
    expect_fail("patch-ignores-switch", "T9", B,
                [("        ggml_sycl_moe_hybrid_record_patch_enabled() ?\n", "        true ?\n")])
    expect_fail("patch-not-counted", "T9", B,
                [("        g_moe_hybrid_record_trace.patch(plan, patched_ok, changed.size());\n", "")])
    expect_fail("patch-line-at-level-1", "T9", B,
                [("        if (trace >= 2) {\n            GGML_LOG_WARN(\n                \"[MOE-HYBRID-RECORD-PATCH] ",
                  "        if (trace >= 1) {\n            GGML_LOG_WARN(\n                \"[MOE-HYBRID-RECORD-PATCH] ")])
    expect_fail("patch-published-unretained", "T9", B,
                [("        ggml_sycl::retain_handles_until_event(std::move(replaced), ctx.stream()->ext_oneapi_submit_"
                  "barrier());\n", "")])
    expect_fail("patch-published-unsettled", "T9", B,
                [("        (void) ggml_sycl::moe_decode_direct_stamp_settle(stamp,", "        (void) (stamp,")])
    expect_fail("lookup-returns-true-elsewhere", "T9", B,
                [("    *out = held;\n    return *out != nullptr;",
                  "    *out = held;\n    if (*out != nullptr) {\n        return true;\n    }\n    return *out != nullptr;")])

    # T10: the patch.
    expect_fail("patch-takes-held-mutable", "T10", B,
                [("    const moe_hybrid_decode_record &                  held,",
                  "    moe_hybrid_decode_record &                        held,")])
    expect_fail("patch-resolves-every-expert", "T10", B,
                [("        src0, device, changed.data(), changed.size(), changed.size(), held.layout,",
                  "        src0, device, all.data(), all.size(), all.size(), held.layout,")])
    expect_fail("patch-writes-held-in-place", "T10", B,
                [("std::make_shared<moe_hybrid_decode_record>(held);",
                  "std::shared_ptr<moe_hybrid_decode_record>(const_cast<moe_hybrid_decode_record *>(&held), "
                  "[](moe_hybrid_decode_record *) {});")])
    expect_fail("patch-skips-shape-rule", "T10", B,
                [("    if (ggml_sycl::moe_hybrid_record_shape_refusal(patched->all_host, shape) != nullptr) {",
                  "    if (false) {")])
    expect_fail("patch-keeps-held-table", "T10", B,
                [("        patched->table = {};\n        if (!ggml_sycl_moe_hybrid_decode_table_fill(",
                  "        if (false && !ggml_sycl_moe_hybrid_decode_table_fill(")])
    expect_fail("patch-drops-held-ready", "T10", B,
                [("        if (held.has_ready) {\n            ready.push_back(held.ready);\n        }\n", "")])
    expect_fail("patch-gives-up-after-residency", "T10", B,
                [("    replaced->insert(replaced->end(),",
                  "    if (leases.empty()) {\n        return false;\n    }\n    replaced->insert(replaced->end(),")])
    expect_fail("patch-drops-replaced-leases", "T10", B,
                [("    replaced->insert(replaced->end(), std::make_move_iterator(leases.begin()), "
                  "std::make_move_iterator(leases.end()));\n", "")])
    expect_fail("patch-drops-held-table", "T10", B,
                [("    if (held.table.valid()) {\n        replaced->push_back(held.table);\n    }\n", "")])
    expect_fail("patch-waits", "T10", B,
                [("    *out = std::move(patched);", "    queue.wait();\n    *out = std::move(patched);")])

    # T11: one slot rule, shape rule and table fill; the switch read once.
    expect_fail("build-chooses-executor-itself", "T11", B,
                [("        ggml_sycl::moe_hybrid_record_slot<ggml_sycl::mem_handle> slot;\n",
                  "        (void) ggml_sycl::choose_moe_batch_executor(operand, device, true, 0);\n"
                  "        ggml_sycl::moe_hybrid_record_slot<ggml_sycl::mem_handle> slot;\n")])
    expect_fail("build-own-shape-rule", "T11", B,
                [("    if (const char * why = ggml_sycl::moe_hybrid_record_shape_refusal(all_host, shape)) {",
                  "    if (const char * why = shape.n_host == 0 ? \"no expert runs on the CPU\" : nullptr) {")])
    expect_fail("build-own-table", "T11", B,
                [("    if (!all_host && !ggml_sycl_moe_hybrid_decode_table_fill(queue,",
                  "    if (!all_host && !ggml_sycl_moe_hybrid_decode_table_allocate(queue,")])
    expect_fail("patch-switch-per-call", "T11", B,
                [("    static const bool enabled = [] {\n        const char * env = std::getenv(\"" + PATCH_ENV,
                  "    const bool enabled = [] {\n        const char * env = std::getenv(\"" + PATCH_ENV)])
    expect_fail("patch-switch-default-off", "T11", B,
                [("std::getenv(\"" + PATCH_ENV + "\");\n        return env == nullptr || std::atoi(env) != 0;",
                  "std::getenv(\"" + PATCH_ENV + "\");\n        return env != nullptr && std::atoi(env) != 0;")])
    C = "common"
    expect_fail("clear-names-one-expert", "T8", C,
                [("        moe_expert_storage_handles.clear();\n        note_moe_storage_change(" + ALL + ");",
                  "        moe_expert_storage_handles.clear();\n        note_moe_storage_change(0);")])
    expect_fail("owner-clear-names-the-owner", "T8", C,
                [("        if (erased) {\n            note_moe_storage_change(" + ALL + ");",
                  "        if (erased) {\n            note_moe_storage_change(owner_device);")])
    expect_fail("note-does-not-log", "T8", C,
                [("        moe_expert_storage_changes.note(moe_expert_storage_generation, expert_id);\n", "")])
    expect_fail("raw-bump-in-common", "T8", C,
                [("        if (erased) {\n            note_moe_storage_change(expert_id);\n        }\n        return erased;",
                  "        if (erased) {\n            ++moe_expert_storage_generation;\n        }\n        return erased;")])
    expect_fail("raw-bump-in-backend", "T8", B,
                [("        w.note_moe_storage_change(e);", "        ++w.moe_expert_storage_generation;")])
    expect_fail("mutation-names-one-expert", "T8", B,
                [("extra->weight().note_moe_storage_change(" + ALL + ");",
                  "extra->weight().note_moe_storage_change(0);")])
    expect_fail("fixture-clear-names-one-expert", "T8", B,
                [("extra.weight().note_moe_storage_change(" + ALL + ");", "extra.weight().note_moe_storage_change(0);")])
    expect_fail("unread-note-site", "T8", B,
                [("    if (stored || table_dropped) {", "    w.note_moe_storage_change(0);\n    if (stored || table_dropped) {")])
    expect_fail("mutation-notes-only-when-nothing-withdrawn", "T8", B,
                [("    extra->weight().note_moe_storage_change(" + ALL + ");\n",
                  "    if (expert_count == 0) {\n        extra->weight().note_moe_storage_change(" + ALL + ");\n    }\n")])

    # T1, T10, T12: the drop's withdrawal, and the two places that refuse or clear a withdrawn record.
    expect_fail("held-return-serves-withdrawn", "T1", B,
                [("        return stamp.eligible && *out && (*out)->n_withdrawn == 0;",
                  "        return stamp.eligible && *out;")])
    expect_fail("patch-keeps-withdrawn-count", "T10", B, [("    patched->n_withdrawn = 0;\n", "")])
    expect_fail("drop-resets-stamp", "T12", B,
                [("    if (stored && std::find(emptied.begin(),",
                  "    w.moe_hybrid_decode_stamp[device] = {};\n    if (stored && std::find(emptied.begin(),")])
    expect_fail("drop-withdraws-unnoted", "T12", B,
                [("    std::vector<int32_t> emptied;\n", "    std::vector<int32_t> emptied;\n    std::vector<int32_t> unnoted;\n"),
                 ("                                                    emptied)) {",
                  "                                                    unnoted)) {")])
    expect_fail("withdraw-writes-held", "T12", B,
                [("    auto copy = std::make_shared<moe_hybrid_decode_record>(*held);",
                  "    auto * copy = const_cast<moe_hybrid_decode_record *>(held.get());")])
    expect_fail("withdraw-uncounted", "T12", B, [("    copy->n_withdrawn += ggml_sycl::", "    (void) ggml_sycl::")])
    expect_fail("withdraw-unpublished", "T12", B, [("    held = std::move(copy);\n", "")])
    expect_fail("withdraw-matches-by-resolve", "T12", B,
                [("    const auto on_entry = [&](const ggml_sycl::mem_handle & h) {\n        return h.is_weight() &&",
                  "    const auto on_entry = [&](const ggml_sycl::mem_handle & h) {\n        return h.resolve(device).ptr &&")])
    withdrawn_count = ("    if (trace > 0 && held && held->n_withdrawn > 0) {\n"
                       "        g_moe_hybrid_record_trace.withdrawn(held->n_withdrawn);\n    }\n")
    expect_fail("lookup-uncounts-withdrawal", "T12", B, [(withdrawn_count, "")])
    expect_fail("withdrawal-counted-after-plan", "T12", B,
                [(withdrawn_count, ""),
                 ("    std::shared_ptr<const moe_hybrid_decode_record> patched;\n",
                  withdrawn_count + "    std::shared_ptr<const moe_hybrid_decode_record> patched;\n")])
    expect_fail("append-keeps-empty-leases", "T12", B,
                [("    for (const ggml_sycl::mem_handle & h : record.lease) {\n        if (h.valid()) {\n"
                  "            out.push_back(h);\n        }\n    }\n",
                  "    out.insert(out.end(), record.lease.begin(), record.lease.end());\n")])
    expect_fail("docs-row-lacks-withdrawn", "T7", D,
                [("patch_unlogged= withdrawn=` line", "patch_unlogged=` line"), (" `withdrawn=` counts", " counts")])

    try:
        mutate(B, [("this anchor is in no source", "x")])
        failures.append("stale-anchor-not-reported")
        print("  self-test: a stale mutant anchor is NOT reported")
    except MutantNotApplied:
        print("  self-test: a stale mutant anchor is reported")

    try:
        check(raws["backend"], raws["header"], raws["docs"], raws["common"])
        print("  self-test: unmodified tree passes")
    except ContractError as e:
        print(f"  self-test: unmodified tree FAILS: {e}")
        failures.append("unmodified")

    if failures:
        print(f"SELF-TEST FAIL: {', '.join(failures)}")
        return 1
    print(f"SELF-TEST PASS: {n_mutants} mutants caught by their pins, unmodified tree passes")
    return 0


def main(argv: list[str]) -> int:
    raws = {}
    for key, path in (("backend", BACKEND), ("header", HEADER), ("docs", DOCS), ("common", COMMON)):
        try:
            raws[key] = path.read_text()
        except FileNotFoundError:
            print(f"FAIL: {path.relative_to(ROOT)} does not exist")
            return 1
    if "--self-test" in argv:
        return self_test(raws)
    try:
        check(raws["backend"], raws["header"], raws["docs"], raws["common"])
    except ContractError as e:
        print(e)
        return 1
    print("PASS: the hybrid decode record lookup is classified once and counted at every exit, at WARN")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
