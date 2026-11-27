#!/usr/bin/env python3
"""Source contract: how the event-chained hybrid decode route keeps its per-tensor record current, and what it counts.

Host-only: reads ggml-sycl.cpp, common.hpp, moe-hybrid-record.hpp, moe-storage-change-log.hpp and
docs/backend/sycl-env-vars.md, and lists which sources include moe-hybrid-record.hpp; builds nothing, loads no model
and touches no device.

The route (ggml_sycl_moe_hybrid_decode_event_chained) looks its tensor's record up at every decode op
(ggml_sycl_moe_hybrid_decode_record_current). The record is cached under a stamp of (replan epoch, the weight's expert
storage generation, selected rows); a miss rebuilds it (ggml_sycl_moe_hybrid_decode_record_build), which resolves
every expert of the tensor. GGML_SYCL_MOE_HYBRID_RECORD_TRACE counts that work per token, so a device run can say how
many records a token rebuilds, why, and how many experts each rebuild resolves. The host test
(tests/test-sycl-moe-hybrid-record.cpp) proves the bookkeeping; this gate holds where the backend calls it:

  T1  the lookup classifies its stamp with moe_hybrid_record_miss_classify, once, telling it whether the held record
      has a slot a loan or an evict withdrew (the classifier names a current stamp over one a generation miss, so the
      record is patched or built, never served and never stalled on), and only a NONE miss returns the held decision;
      the stamp's own current test is not asked a second time beside it;
  T2  the trace level is read once, in a function-local static of ggml_sycl_moe_hybrid_record_trace_level(), from
      GGML_SYCL_MOE_HYBRID_RECORD_TRACE; the name appears nowhere else in the backend;
  T3  every lookup is counted, with its miss, before the held decision returns; the lookup that finds no published
      plan counts itself before it returns; each build's outcome is counted after the build;
  T4  the build counts the experts it resolves at its one resolve, right before the call that resolves them;
  T5  the per-token line and the per-build line are WARN, each printed from one site, the per-build one only at
      level 2 and above; the per-token line's buffer is the 768 bytes the host test proves the widest line fits;
  T6  both headers are plain host code, no SYCL: moe-hybrid-record.hpp includes the standard library,
      moe-decode-hostpath.hpp and moe-storage-change-log.hpp, which defines the weight's change log and includes the
      standard library alone; common.hpp includes the change log's header and not the record's, which only
      ggml-sycl.cpp and the host test include;
  T7  the env-var row names the variable, both line tags, the read-once rule and the token rule; the patch switch's
      row says that `=0` builds the whole record instead of patching it and that the withdrawal still runs, and
      never that `=0` is the tree before the patch;
  T8  every move of a weight's expert storage generation goes through note_moe_storage_change(), which logs the
      expert the move changed: no other site writes the generation, each site that moves one expert's storage names
      that expert, and each site that clears a whole store names every expert (moe_storage_change_log::ALL). A site
      that named one expert for a cleared store would let a held record keep slots the clear withdrew;
  T9  a lookup that missed on the generation alone asks moe_hybrid_record_plan_patch (when the patch is on) and, for
      the experts the log names, patches the held record; a patch that stands is published by retaining what it
      replaced to a barrier on the queue, settling the stamp at the new generation and returning the patched record,
      ahead of the build; its count and its level-2 line, which names the reason a patch declined, come before that
      return;
  T10 the patch holds its own shared pointer to const of the held record; every way it declines names its reason
      through one helper, into the caller's reason parameter; it asks the build's plan rule again (an expert move
      rewrites the plan without moving the replan epoch) and declines unless the frame the build would take now is the
      held record's, the build's frame rule stands for the held mode and layout, and the held route-count table is
      still the device's; resolves only the changed experts and declines unless every one resolved, each operand
      answering for its moved expert; takes each slot from the build's slot rule, writes a copy of the held record
      (never the held one), declines when the install into it refuses, checks the copy's shape with the build's rule,
      gives it a table of its own filled from every slot (the one table it fills; the held table it only hands back,
      it resolves no handle, and it names the copy's table, the held one's handle until the reset, only to reset and
      fill it, however either record is spelled: `x->`, `(*x).` or `x.get()->`), a ready barrier over the held
      one's and the new events, records the block's residency only after every way out, hands the replaced leases
      and the held table back for retention, and waits for nothing;
  T11 the build takes its plan, its frame, its slots, its shape check and its table from those same helpers and
      records the expert its layout came from; the plan is asked in the plan rule alone, the routable cards and the
      MMID direct route in the frame rule alone, the executor in the slot rule alone; and
      GGML_SYCL_MOE_HYBRID_RECORD_PATCH is read once;
  T12 an expert loan or evict (ggml_sycl_drop_lent_expert_views) withdraws only the record's slots that lease the
      entry it retired, from a copy it publishes in the held record's place (never the held one), counting them on the
      copy and moving their leases into the drop's retention. It asks the moved expert's own slot first and empties
      that slot alone when it holds the entry (an expert move refuses a device copy that shares storage, so the entry
      backs only that slot); otherwise it asks every slot and counts the walk in the drop's scan_fallbacks, which the
      swap summary prints, so a view the slot rule misses is still withdrawn and shows. It keeps the stamp, and notes
      every expert it emptied,
      so the next lookup is a generation miss the patch serves. That lookup counts the withdrawn slots in the trace,
      once, between the held record's binding and the patch plan, and the replaced record's retained handles skip a
      withdrawn slot's empty lease;
  T13 a patch or a build that the expert storage moved under is not served, a defensive check (nothing in the
      lookup's own resolves or allocations moves the generation, but an expert move on another thread can overlap
      the lookup): the patch counts as declined, with that reason, and the build ends as a retry, each checked
      before its count, and the discarded record's handles wait for a barrier on the queue as a replaced record's
      do;
  T14 a build that replaces the held record retains the replaced record's handles to a barrier on the queue before
      it replaces or drops it, as a published patch retains what it replaced.

Checks run on comment-blanked text (string literals kept). NOT VACUOUS: --self-test runs the gate on in-memory mutants
of the real sources. Each mutant names the pin that must catch it, and a mutant aimed at one of a pin's several rules
names that rule's message too. A mutant is reported if it survives, if another pin or rule catches it, or if its
anchor no longer matches exactly once (NOT APPLIED). ctest registers it with --self-test.

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
LOG_HEADER = ROOT / "ggml/src/ggml-sycl/moe-storage-change-log.hpp"
COMMON = ROOT / "ggml/src/ggml-sycl/common.hpp"
DOCS = ROOT / "docs/backend/sycl-env-vars.md"

CURRENT_SIG = "static bool ggml_sycl_moe_hybrid_decode_record_current("
BUILD_SIG = "static ggml_sycl::moe_decode_direct_outcome ggml_sycl_moe_hybrid_decode_record_build("
LEVEL_SIG = "static int ggml_sycl_moe_hybrid_record_trace_level() {"
LOOKUP_SIG = "static void ggml_sycl_moe_hybrid_record_trace_lookup("
ENV_NAME = "GGML_SYCL_MOE_HYBRID_RECORD_TRACE"

# The held record's withdrawn slots go to the classifier, which names a current stamp over them a generation miss.
CLASSIFY = ("const ggml_sycl::moe_hybrid_record_miss miss = ggml_sycl::moe_hybrid_record_miss_classify(stamp, "
            "replan_epoch, storage_generation, selected_rows, held && held->n_withdrawn > 0);")
HELD_RETURN = ("if (miss == ggml_sycl::MOE_HYBRID_RECORD_MISS_NONE) { *out = held; "
               "return stamp.eligible && *out != nullptr; }")
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
              "ggml_sycl::moe_hybrid_record_plan_patch(miss, stamp, held != nullptr, "
              "weight.moe_expert_storage_changes, "
              "storage_generation, &changed) : ggml_sycl::MOE_HYBRID_RECORD_PATCH_PLAN_NONE;")
PATCH_CALL = ("if (plan == ggml_sycl::MOE_HYBRID_RECORD_PATCH_PLAN_EXPERTS) { patched_ok = "
              "ggml_sycl_moe_hybrid_decode_record_patch(ctx, src0, held, changed, &patched, &replaced, &declined); }")
PATCH_COUNT = "g_moe_hybrid_record_trace.patch(plan, patched_ok, changed.size());"
PATCH_WARN = 'GGML_LOG_WARN("[MOE-HYBRID-RECORD-PATCH] '
# The level-2 line names why a patch declined, after its outcome: the reason the patch handed back, or nothing.
PATCH_LINE_FORMAT = 'experts=%zu " "outcome=%s%s%s\\n",'
PATCH_LINE_REASON = 'declined ? ": " : "", declined ? declined : "");'
PATCH_PUBLISH = ("if (patched_ok) { ggml_sycl::retain_handles_until_event(std::move(replaced), "
                 "ctx.stream()->ext_oneapi_submit_barrier()); (void) ggml_sycl::moe_decode_direct_stamp_settle(stamp, "
                 "replan_epoch, storage_generation, selected_rows, static_cast<int>(patched->layout), "
                 "ggml_sycl::MOE_DECODE_DIRECT_OUTCOME_ELIGIBLE); held = std::move(patched); *out = held; "
                 "return true; }")

# T10: the patch itself.
PATCH_SIG = "static bool ggml_sycl_moe_hybrid_decode_record_patch("
PATCH_DECL = ("static bool ggml_sycl_moe_hybrid_decode_record_patch(ggml_backend_sycl_context & ctx, "
              "const ggml_tensor * src0, std::shared_ptr<const moe_hybrid_decode_record> held, "
              "const std::vector<int32_t> & changed,")
PATCH_REASON_PARAM = "std::vector<ggml_sycl::mem_handle> * replaced, const char ** reason) {"
# Every decline goes through the one helper that names its reason, so the level-2 line can say why.
PATCH_DECLINE = "const auto decline = [reason](const char * why) { *reason = why; return false; };"
# The held table's one use in the patch: handed back for retention, never resolved or filled.
PATCH_HELD_TABLE = "if (held->table.valid()) { replaced->push_back(held->table); }"
TABLE_FILL = "ggml_sycl_moe_hybrid_decode_table_fill("
# The copy starts as the held record, so its table is the held table's handle until the reset: the patch names the
# copy's table only to reset it and to fill it.
PATCH_COPY_TABLE_USES = ("patched->table = {};", "&patched->table)")
# What each table rule says when it fails, so a self-test mutant names the rule that must catch it.
RESOLVE_SAYS = "the patch resolves a handle"
HELD_TABLE_SAYS = "the patch names the held table other than"
COPY_TABLE_SAYS = "the patch names the copy's table other than"
# A dereference of the copy or the held record, spelled `(*x).` or `x.get()->`, is rewritten to `x->` before the
# table rules count `patched->table` and `held->table`, so another spelling of the same member is counted too.
DEREF_SPELLINGS = (re.compile(r"\(\s*\*\s*(patched|held)\s*\)\s*\.\s*"),
                   re.compile(r"\b(patched|held)\s*\.\s*get\s*\(\s*\)\s*->\s*"))
PATCH_STEPS = (
    ("decline helper", PATCH_DECLINE),
    ("the build's plan rule, asked again",
     "if (const char * why = ggml_sycl_moe_hybrid_decode_record_plan_refusal(src0, device, "
     "static_cast<int64_t>(held->n_as), &counts)) { return decline(why); }"),
    ("the frame the build would take now",
     "if (!ggml_sycl::moe_hybrid_record_patch_frame_holds(held->all_host, held->layout_expert, counts.on_device == 0, "
     "changed.front())) { return decline("),
    ("the build's frame rule over the held record's mode and layout",
     "if (const char * why = ggml_sycl_moe_hybrid_decode_record_frame_refusal(ctx, src0, held->all_host, "
     "held->layout)) { return decline(why); }"),
    ("the held record's route-count table, still the device's",
     "if (held->route_block >= 0 && held->route_counts != ggml_sycl::moe_route_counts_current(device)) { "
     "return decline("),
    ("its resolve of the changed experts alone",
     "ggml_sycl::ggml_sycl_build_moe_resolved_batch(src0, device, changed.data(), changed.size(), changed.size(), "
     "held->layout,"),
    ("check that every moved expert resolved, in a batch of their count",
     "if (!resolved || resolved.batch.operands.size() != changed.size()) { return decline("),
    ("check of each resolved operand against the moved expert it answers for",
     'if (operand.expert_id != changed[i]) { return decline("'),
    ("each slot from the build's slot rule, its refusal named",
     "if (ggml_sycl_moe_hybrid_decode_record_slot(operand, device, held->layout, held->all_host, &fresh[i], ready, "
     "&why) != ggml_sycl::MOE_DECODE_DIRECT_OUTCOME_ELIGIBLE) { return decline(why); }"),
    ("its copy of the held record", "auto patched = std::make_shared<moe_hybrid_decode_record>(*held);"),
    ("the install of the moved slots into the copy, a refusal declining",
     "if (!ggml_sycl::moe_hybrid_record_install(patched->device_weight, patched->host_weight, patched->lease, changed, "
     "fresh, &leases)) { return decline("),
    ("the copy's shape",
     "const ggml_sycl::moe_hybrid_record_shape shape = ggml_sycl::moe_hybrid_record_shape_of(patched->device_weight, "
     "patched->host_weight, patched->lease);"),
    ("the build's shape rule over it",
     "if (const char * why = ggml_sycl::moe_hybrid_record_shape_refusal(patched->all_host, shape)) { "
     "return decline(why); }"),
    ("the copy's withdrawn count cleared, the rule having found no empty slot", "patched->n_withdrawn = 0;"),
    ("a table of the copy's own, filled from every slot",
     "patched->table = {}; if (!ggml_sycl_moe_hybrid_decode_table_fill(queue, device, extra->model_id, "
     "patched->device_weight, shape.substitute, &patched->table)) { return decline("),
    ("the held record's ready event among the barrier's", "if (held->has_ready) { ready.push_back(held->ready); }"),
    ("the copy's ready barrier", "patched->ready = queue.ext_oneapi_submit_barrier(ready);"),
    ("the block's residency",
     "patched->route_counts->set_block_resident(patched->route_block, "
     "static_cast<int>(patched->n_as - shape.n_host));"),
    ("the replaced leases handed back",
     "replaced->insert(replaced->end(), std::make_move_iterator(leases.begin()), "
     "std::make_move_iterator(leases.end()));"),
    ("the held table handed back", PATCH_HELD_TABLE),
    ("the copy's publication", "*out = std::move(patched);"),
)

# T11: the rules the build and the patch share.
SLOT_SIG = "static ggml_sycl::moe_decode_direct_outcome ggml_sycl_moe_hybrid_decode_record_slot("
BUILD_STEPS = (
    ("plan from the plan rule, and its all-host mode from the plan's counts",
     "ggml_sycl::expert_residency_counts counts; if (const char * why = "
     "ggml_sycl_moe_hybrid_decode_record_plan_refusal(src0, device, n_as, &counts)) { return refuse(why); } "
     "const bool all_host = counts.on_device == 0;"),
    ("frame from the frame rule, once its mode and layout are known",
     "if (const char * why = ggml_sycl_moe_hybrid_decode_record_frame_refusal(ctx, src0, all_host, layout)) { "
     "return refuse(why); }"),
    ("record of the expert its layout came from", "record->layout_expert = layout_expert;"),
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
# The plan rule the build and the patch share: the plan's two queries, decided by the header's rule.
PLAN_SIG = "static const char * ggml_sycl_moe_hybrid_decode_record_plan_refusal("
PLAN_STEPS = (
    ("query of the host experts", "const bool has_host = plan->has_host_experts(src0->name, n_as, device);"),
    ("query of the planned residency",
     "*counts = has_host ? plan->count_planned_experts(src0->name, n_as, device) : "
     "ggml_sycl::expert_residency_counts{};"),
    ("decision by the header's rule",
     "return ggml_sycl::moe_hybrid_record_plan_refusal(has_host, counts->on_device, counts->on_target);"),
)
PLAN_QUERIES = ("has_host_experts(", "count_planned_experts(")
# The frame rule the build and the patch share: what refuses a record of the tensor on this context whatever its
# experts' slots say, beyond the plan.
FRAME_SIG = "static const char * ggml_sycl_moe_hybrid_decode_record_frame_refusal("
FRAME_STEPS = (
    ("one routable card", 'if (ggml_sycl_routable_device_count() != 1) { return "'),
    ("the MMID direct route's split AOS tensors",
     "if (!all_host && layout == GGML_LAYOUT_AOS && ggml_sycl_moe_mmid_route_reachable(ctx) && "
     'moe_mmid_direct_route_runs_type(src0->type)) { return "'),
    ("standing frame", "return nullptr; }"),
)
FRAME_QUERIES = ("ggml_sycl_routable_device_count(", "ggml_sycl_moe_mmid_route_reachable(",
                 "moe_mmid_direct_route_runs_type(")
PATCH_ENV = "GGML_SYCL_MOE_HYBRID_RECORD_PATCH"
PATCH_LEVEL_SIG = "static bool ggml_sycl_moe_hybrid_record_patch_enabled() {"
PATCH_LEVEL_BODY = ('{ static const bool enabled = [] { const char * env = std::getenv("' + PATCH_ENV + '"); '
                    "return env == nullptr || std::atoi(env) != 0; }(); return enabled; }")

DOC_NEEDLES = ("`GGML_SYCL_MOE_HYBRID_RECORD_TRACE=1\\|2`", "[MOE-HYBRID-RECORD]", "[MOE-HYBRID-RECORD-BUILD]",
               "[MOE-HYBRID-RECORD-PATCH]", "patches=", "withdrawn=", "Read once", "lower block",
               "outcome=patched\\|declined: <why>\\|unlogged", "a plan that no patch answers builds once",
               "the patch attempts asked for, resolved or not", "two or three per tensor, so about 21 to 32 moves",
               "an expert move on another thread that overlapped the lookup")
# What the row must not say, and why: it must not count three notes alone, nor blame the lookup itself for a move.
DOC_FORBIDDEN = {
    "about three": "an expert move writes two or three notes per tensor",
    "its own allocations set off": "nothing the lookup's own resolves or allocations do moves the generation",
}
PATCH_DOC_NEEDLES = ("`GGML_SYCL_MOE_HYBRID_RECORD_PATCH=0`", "Read once", "every expert", "instead of patching it",
                     "still withdraws", "parses as a non-zero integer (`on` and `true` parse as 0)")
# `=0` keeps the loan and evict withdrawal, so it is not the tree before the patch, and the row must not say it is.
PATCH_DOC_FORBIDDEN = ("as before the patch",)
HEADER_INCLUDE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]', re.M)
HEADER_ALLOWED = {"moe-decode-hostpath.hpp", "moe-storage-change-log.hpp"}
# The weight's change log is a member of every weight extra, so common.hpp includes its header; the record's
# bookkeeping is the backend's alone, so common.hpp, which every SYCL source includes, does not.
LOG_DEF = "struct moe_storage_change_log {"
LOG_INCLUDE = "moe-storage-change-log.hpp"
RECORD_INCLUDE = "moe-hybrid-record.hpp"
RECORD_INCLUDERS = {"ggml/src/ggml-sycl/ggml-sycl.cpp", "tests/test-sycl-moe-hybrid-record.cpp"}
# Where an include of the record's header could hide: every SYCL source and header, and the tests' C++ sources.
INCLUDER_GLOBS = ("ggml/src/ggml-sycl/**/*.cpp", "ggml/src/ggml-sycl/**/*.hpp", "tests/*.cpp")

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
WITHDRAW_DECL = ("static bool ggml_sycl_moe_hybrid_decode_record_withdraw("
                 "std::shared_ptr<const moe_hybrid_decode_record> "
                 "& held,")
DROP_WITHDRAW = ("if (ggml_sycl_moe_hybrid_decode_record_withdraw(w.moe_hybrid_decode[device], device, expert_id, "
                 "withdrawn, retained, emptied, scan_fallbacks)) { table_dropped = true; }")
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
    ("return with no held record", "if (!held) { return false; }"),
    ("test that the moved expert's own slot leases the entry",
     "const bool at_slot = ggml_sycl::moe_hybrid_record_slot_matches(held->lease, expert_id, on_entry);"),
    ("counted walk of every slot when it does not",
     "if (!at_slot) { ++scan_fallbacks; if (std::none_of(held->lease.begin(), held->lease.end(), on_entry)) { "
     "return false; } }"),
    ("copy of the held record", "auto copy = std::make_shared<moe_hybrid_decode_record>(*held);"),
    ("withdrawal of the one slot from the copy, counted on it, its lease into the drop's retention",
     "if (at_slot) { copy->n_withdrawn += ggml_sycl::moe_hybrid_record_withdraw_slot(copy->device_weight, "
     "copy->host_weight, copy->lease, expert_id, &retained, &emptied) ? 1 : 0; }"),
    ("withdrawal of every matching slot from the copy otherwise, counted on it, their leases into the retention",
     "else { copy->n_withdrawn += ggml_sycl::moe_hybrid_record_withdraw(copy->device_weight, copy->host_weight, "
     "copy->lease, on_entry, &retained, &emptied); }"),
    ("publication of the copy", "held = std::move(copy);"),
)
# The fixture's store clear, inside a lambda of a private test function: matched in the squashed file text.
FIXTURE_CLEAR = ("extra.weight().moe_expert_storage_handles.clear(); extra.weight().note_moe_storage_change("
                 + ALL + ");")

# T13: a patch or build the expert storage moved under is not served. Nothing in the lookup's own resolves or
# allocations moves the generation: the check guards against an expert move on another thread that overlaps the
# lookup (the graph-compute flag a transaction refuses under is one process-wide flag), which may also publish a
# withdrawn copy in the held record's place.
PATCH_MOVED = ("if (patched_ok && weight.moe_expert_storage_generation != storage_generation) { "
               "ggml_sycl_moe_hybrid_decode_record_append_handles(*patched, replaced); "
               "ggml_sycl::retain_handles_until_event(std::move(replaced), ctx.stream()->ext_oneapi_submit_barrier()); "
               "patched.reset(); patched_ok = false; declined = \"")
BUILD_MOVED = ("if (weight.moe_expert_storage_generation != storage_generation) { if (built) { "
               "std::vector<ggml_sycl::mem_handle> handles; "
               "ggml_sycl_moe_hybrid_decode_record_append_handles(*built, handles); "
               "ggml_sycl::retain_handles_until_event(std::move(handles), ctx.stream()->ext_oneapi_submit_barrier()); "
               "built.reset(); } outcome = ggml_sycl::MOE_DECODE_DIRECT_OUTCOME_RETRY; }")

# T14: a build that replaces the held record retains the replaced record's handles to a barrier before it drops it.
BUILD_REPLACED_RETAIN = ("if (held) { std::vector<ggml_sycl::mem_handle> handles; "
                         "ggml_sycl_moe_hybrid_decode_record_append_handles(*held, handles); "
                         "ggml_sycl::retain_handles_until_event(std::move(handles), "
                         "ctx.stream()->ext_oneapi_submit_barrier()); }")
BUILD_REPLACE = ("if (settled == ggml_sycl::MOE_DECODE_DIRECT_OUTCOME_ELIGIBLE) { held = std::move(built); } else { "
                 "held.reset(); }")


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


def included_headers(code: str) -> set[str]:
    """The file names (without directories) of the project headers `code` includes."""
    return {Path(m.group(2)).name for m in HEADER_INCLUDE.finditer(code) if m.group(1) == '"'}


def arrow_spelling(text: str) -> str:
    """`(*x).m` and `x.get()->m` written as `x->m`, for x the patch's copy or the held record."""
    for spelling in DEREF_SPELLINGS:
        text = spelling.sub(r"\1->", text)
    return text


def once(text: str, needle: str, pin: str, what: str) -> int:
    n = text.count(needle)
    if n != 1:
        raise ContractError(f"FAIL [pin {pin}]: {what}: `{needle[:110]}` appears {n} times, not once")
    return text.find(needle)


def check(backend_raw: str, header_raw: str, docs_raw: str, common_raw: str, log_raw: str, includers_raw: str) -> None:
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

    # T6: both headers stay plain host code, and only the change log reaches every SYCL source through common.hpp.
    header = blank_comments(header_raw)
    log_header = blank_comments(log_raw)
    for file, text, allowed in (("moe-hybrid-record.hpp", header, HEADER_ALLOWED),
                                ("moe-storage-change-log.hpp", log_header, set())):
        for m in HEADER_INCLUDE.finditer(text):
            kind, name = m.group(1), m.group(2)
            if kind == '"' and name not in allowed:
                raise ContractError(f"FAIL [pin T6]: {file} includes the project header `{name}`")
            if kind == "<" and ("sycl" in name or "/" in name):
                raise ContractError(f"FAIL [pin T6]: {file} includes `<{name}>`")
        if re.search(r"\bsycl::", text):
            raise ContractError(f"FAIL [pin T6]: {file} names a SYCL type")
    if squash(log_header).count(LOG_DEF) != 1 or LOG_DEF in squash(header):
        raise ContractError("FAIL [pin T6]: moe_storage_change_log is not defined once, in moe-storage-change-log.hpp")
    common_includes = included_headers(blank_comments(common_raw))
    if LOG_INCLUDE not in common_includes or RECORD_INCLUDE in common_includes:
        raise ContractError("FAIL [pin T6]: common.hpp does not include the change log's header alone: every SYCL "
                            "source would see the hybrid record's bookkeeping, which only the backend uses")
    if RECORD_INCLUDE not in included_headers(code):
        raise ContractError("FAIL [pin T6]: ggml-sycl.cpp does not include moe-hybrid-record.hpp itself")
    others = sorted(set(includers_raw.split()) - RECORD_INCLUDERS)
    if others:
        raise ContractError(f"FAIL [pin T6]: moe-hybrid-record.hpp is included from {others}, not only from "
                            f"{sorted(RECORD_INCLUDERS)}")

    # T7: the env-var row.
    rows = [line for line in docs_raw.splitlines() if line.startswith(f"| `{ENV_NAME}")]
    if len(rows) != 1:
        raise ContractError(f"FAIL [pin T7]: docs/backend/sycl-env-vars.md has {len(rows)} rows for {ENV_NAME}, "
                            f"not one")
    for needle in DOC_NEEDLES:
        if needle not in rows[0]:
            raise ContractError(f"FAIL [pin T7]: the {ENV_NAME} row does not say `{needle}`")
    for needle, why in DOC_FORBIDDEN.items():
        if needle in rows[0]:
            raise ContractError(f"FAIL [pin T7]: the {ENV_NAME} row says `{needle}`, but {why}")
    patch_rows = [line for line in docs_raw.splitlines() if line.startswith(f"| `{PATCH_ENV}")]
    if len(patch_rows) != 1 or not all(needle in patch_rows[0] for needle in PATCH_DOC_NEEDLES):
        raise ContractError(f"FAIL [pin T7]: docs/backend/sycl-env-vars.md has no one {PATCH_ENV} row that says "
                            f"{PATCH_DOC_NEEDLES}")
    for needle in PATCH_DOC_FORBIDDEN:
        if needle in patch_rows[0]:
            raise ContractError(f"FAIL [pin T7]: the {PATCH_ENV} row says `{needle}`, but `=0` still withdraws "
                                f"loaned and evicted slots")

    # T9: the lookup's patch path, ahead of the build, published only by retaining what it replaced.
    at_patch_held = once(current, PATCH_HELD, "T9", "the held record's one binding in the lookup")
    at_plan = once(current, PATCH_PLAN, "T9", "the patch plan, asked only when the patch is on")
    at_call = once(current, PATCH_CALL, "T9", "the patch, only for the experts the log names")
    at_count = once(current, PATCH_COUNT, "T9", "the patch's count")
    at_patch_warn = once(current, PATCH_WARN, "T9", "the per-patch WARN line")
    at_publish = once(current, PATCH_PUBLISH, "T9", "the patched record's publication")
    if not at_patch_held < at_classify:
        raise ContractError("FAIL [pin T1]: the held record is bound after the stamp is classified, so the classifier "
                            "cannot be told of its withdrawn slots")
    if not at_no_plan < at_plan < at_call < at_count < at_patch_warn < at_publish < at_build:
        raise ContractError("FAIL [pin T9]: the patch path is not plan, patch, count, line, publish, between the "
                            "no-plan return and the build")
    patch_guard = current.find(BUILD_GUARD, at_count)
    if not at_count < patch_guard < at_patch_warn or current[patch_guard:at_patch_warn].count("}") != 0:
        raise ContractError("FAIL [pin T9]: the per-patch line is not printed under its level-2 guard")
    at_line_format = once(current, PATCH_LINE_FORMAT, "T9", "the per-patch line's outcome and reason fields")
    at_line_reason = once(current, PATCH_LINE_REASON, "T9", "the per-patch line's reason, from the patch")
    if not at_patch_warn < at_line_format < at_line_reason < at_publish:
        raise ContractError("FAIL [pin T9]: the per-patch line does not name, after its outcome, the reason the "
                            "patch declined")
    if current.count("return true;") != 1 or "return true;" not in PATCH_PUBLISH:
        raise ContractError("FAIL [pin T9]: the lookup returns true other than by publishing a patch")

    # T10: the patch.
    patch = body(code, PATCH_SIG, "T10")
    if PATCH_DECL not in squash(code):
        raise ContractError("FAIL [pin T10]: the patch does not hold its own shared pointer to const of the held "
                            "record, so it could write the record a CPU job or a queued copy still reads, or copy one "
                            "a resolve released under it")
    last = -1
    for what, needle in PATCH_STEPS:
        at = once(patch, needle, "T10", f"the patch's {what}")
        if at < last:
            raise ContractError(f"FAIL [pin T10]: the patch's {what} is out of order")
        last = at
    if PATCH_REASON_PARAM not in squash(code):
        raise ContractError("FAIL [pin T10]: the patch takes no `const char ** reason` after `replaced`, so a decline "
                            "cannot say why")
    if patch.count("return false;") != 1:
        raise ContractError("FAIL [pin T10]: the patch declines other than through its decline helper, so that "
                            "decline names no reason")
    at_resident = patch.find(dict(PATCH_STEPS)["the block's residency"])
    if patch.rfind("return decline(") > at_resident:
        raise ContractError("FAIL [pin T10]: the patch can still give up after it recorded the block's residency, "
                            "leaving the count of a record it did not publish")
    if WAIT_CALLS.search(patch):
        raise ContractError("FAIL [pin T10]: the patch waits on the device")
    # The held table may be the source of a table copy still on the queue: the patch only hands it back. The copy
    # shares its handle until the reset, so the patch resolves no handle at all, and names the copy's table only to
    # reset and fill it: the fill writes the one table the patch writes.
    if ".resolve(" in patch:
        raise ContractError(f"FAIL [pin T10]: {RESOLVE_SAYS}, so it could write the held table, through "
                            "the held record or through the copy that shares its table until the reset")
    tables = arrow_spelling(patch)
    if tables.count("held->table") != PATCH_HELD_TABLE.count("held->table"):
        raise ContractError(f"FAIL [pin T10]: {HELD_TABLE_SAYS} to hand it back for retention, "
                            "so it could write a table a queued copy still reads")
    copy_table_uses = [tables.count(use) for use in PATCH_COPY_TABLE_USES]
    if copy_table_uses != [1] * len(PATCH_COPY_TABLE_USES) or tables.count("patched->table") != sum(copy_table_uses):
        raise ContractError(f"FAIL [pin T10]: {COPY_TABLE_SAYS} to reset it and fill it, so "
                            "it could use the held table's handle, which the copy holds until the reset")
    if patch.count(TABLE_FILL) != 1:
        raise ContractError("FAIL [pin T10]: the patch fills a table besides the copy's own")

    # T11: one slot rule, one shape rule and one table fill for the build and the patch; the switch read once.
    slot_rule = body(code, SLOT_SIG, "T11")
    for what, needle in BUILD_STEPS:
        once(build, needle, "T11", f"the build's {what}")
    if ("choose_moe_batch_executor(" not in slot_rule or "choose_moe_batch_executor(" in build
            or "choose_moe_batch_executor(" in patch):
        raise ContractError("FAIL [pin T11]: an expert's executor is chosen outside the slot rule, so the build and "
                            "the patch could place a moved expert apart")
    plan_rule = body(code, PLAN_SIG, "T11")
    for what, needle in PLAN_STEPS:
        once(plan_rule, needle, "T11", f"the plan rule's {what}")
    for query in PLAN_QUERIES:
        if query in build or query in patch:
            raise ContractError(f"FAIL [pin T11]: the build or the patch asks the plan `{query}` itself, so the two "
                                f"could decide a record from different plans")
    frame_rule = body(code, FRAME_SIG, "T11")
    last = -1
    for what, needle in FRAME_STEPS:
        at = once(frame_rule, needle, "T11", f"the frame rule's {what}")
        if at < last:
            raise ContractError(f"FAIL [pin T11]: the frame rule's {what} is out of order")
        last = at
    for query in FRAME_QUERIES:
        if query in build or query in patch:
            raise ContractError(f"FAIL [pin T11]: the build or the patch asks `{query}` itself, outside the frame "
                                f"rule, so the two could disagree on whether the tensor's frame stands")
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
    if not at_no_plan < at_withdrawn < at_plan:
        raise ContractError("FAIL [pin T12]: the withdrawn slots are not counted between the no-plan return and the "
                            "patch plan, the one place every lookup that replaces the record passes exactly once")
    appends = definition_spans(backend_sq, APPEND_SIG)
    if len(appends) != 1 or APPEND_LEASES not in backend_sq[appends[0][0] : appends[0][1] + 1]:
        raise ContractError("FAIL [pin T12]: the replaced record's handle list does not skip a withdrawn slot's empty "
                            "lease")

    # T13: what the expert storage moved under is not served: a patch declines, a build is retried, before its count.
    at_patch_moved = once(current, PATCH_MOVED, "T13", "the patch's discard when the expert storage moved under it")
    if not at_call < at_patch_moved < at_count:
        raise ContractError("FAIL [pin T13]: the patch's storage check is not between the patch and its count, so a "
                            "copy made before a move could be published, or counted as patched")
    at_build_moved = once(current, BUILD_MOVED, "T13", "the build's retry when the expert storage moved under it")
    if not at_build < at_build_moved < at_outcome:
        raise ContractError("FAIL [pin T13]: the build's storage check is not between the build and its outcome count, "
                            "so a record built before a move could be served or cached, or counted as it ended")

    # T14: the build's replacement retains the record it replaces.
    at_replaced_retain = once(current, BUILD_REPLACED_RETAIN, "T14", "the replaced record's retention to a barrier")
    at_replace = once(current, BUILD_REPLACE, "T14", "the build's replacement of the held record")
    if not at_build_moved < at_replaced_retain < at_replace:
        raise ContractError("FAIL [pin T14]: the replaced record's handles are not retained before the build replaces "
                            "it, so a table copy or kernel still on the queue could read storage the cache reused")


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

    def expect_fail(name: str, pin: str, file: str, edits: list, says: str | None = None) -> None:
        nonlocal n_mutants
        n_mutants += 1
        try:
            mutated = mutate(file, edits)
        except MutantNotApplied as e:
            failures.append(f"{name}: NOT APPLIED ({e})")
            print(f"  mutant {name}: NOT APPLIED ({e})")
            return
        try:
            check_all(mutated)
        except ContractError as e:
            if f"[pin {pin}]" not in str(e):
                failures.append(f"{name}: caught, but not by pin {pin}: {str(e)[:140]}")
                print(f"  mutant {name}: caught by the WRONG pin ({str(e)[:120]}...)")
                return
            if says is not None and says not in str(e):
                failures.append(f"{name}: caught by pin {pin}, but not by the rule that says `{says}`: "
                                f"{str(e)[:140]}")
                print(f"  mutant {name}: caught by the WRONG rule of pin {pin} ({str(e)[:120]}...)")
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
                  "        if (trace >= 1) {\n            GGML_LOG_WARN(\n"
                  "                \"[MOE-HYBRID-RECORD-BUILD] ")])
    expect_fail("header-includes-sycl", "T6", H, [("#include <cstdio>", "#include <sycl/sycl.hpp>\n#include <cstdio>")])
    expect_fail("header-includes-common", "T6", H, [("#include <cstdio>", '#include "common.hpp"\n#include <cstdio>')])
    expect_fail("log-header-includes-sycl", "T6", "log",
                [("#include <vector>", "#include <sycl/sycl.hpp>\n#include <vector>")])
    expect_fail("common-includes-record-header", "T6", "common",
                [('#include "moe-storage-change-log.hpp"', '#include "moe-hybrid-record.hpp"')])
    expect_fail("record-header-included-elsewhere", "T6", "includers",
                [("tests/test-sycl-moe-hybrid-record.cpp",
                  "tests/test-sycl-moe-hybrid-record.cpp\nggml/src/ggml-sycl/fattn.cpp")])
    expect_fail("backend-leans-on-common-for-record", "T6", B, [('#include "ggml-sycl/moe-hybrid-record.hpp"\n', "")])
    expect_fail("docs-row-dropped", "T7", D, [(f"| `{ENV_NAME}=1\\|2`", f"| `{ENV_NAME}_OLD=1\\|2`")])
    expect_fail("docs-row-lacks-build-line", "T7", D, [("`[MOE-HYBRID-RECORD-BUILD]", "`[MOE-HYBRID-BUILD]")])
    expect_fail("docs-row-lacks-patch-line", "T7", D,
                [("`[MOE-HYBRID-RECORD-PATCH] token=T", "`[MOE-HYBRID-PATCH] token=T")])
    expect_fail("docs-patch-line-lacks-reason", "T7", D,
                [("outcome=patched\\|declined: <why>\\|unlogged", "outcome=patched\\|declined\\|unlogged")])
    expect_fail("docs-patch-row-dropped", "T7", D,
                [("| `GGML_SYCL_MOE_HYBRID_RECORD_PATCH=0`", "| `GGML_SYCL_MOE_HYBRID_RECORD_NOPATCH=0`")])
    expect_fail("docs-patch-row-hides-withdrawal", "T7", D,
                [("an expert loan or evict still withdraws the held record's",
                  "an expert loan or evict withdraws the held record's")])
    expect_fail("docs-patch-row-claims-the-old-tree", "T7", D,
                [("instead of patching it, as every other stamp miss does either way.",
                  "instead of patching it, as before the patch.")])

    # T9: the lookup's patch path.
    expect_fail("patch-ignores-switch", "T9", B,
                [("        ggml_sycl_moe_hybrid_record_patch_enabled() ?\n", "        true ?\n")])
    expect_fail("patch-not-counted", "T9", B,
                [("        g_moe_hybrid_record_trace.patch(plan, patched_ok, changed.size());\n", "")])
    expect_fail("patch-line-at-level-1", "T9", B,
                [("        if (trace >= 2) {\n            GGML_LOG_WARN(\n                \"[MOE-HYBRID-RECORD-PATCH] ",
                  "        if (trace >= 1) {\n            GGML_LOG_WARN(\n"
                  "                \"[MOE-HYBRID-RECORD-PATCH] ")])
    expect_fail("patch-reason-unprinted", "T9", B,
                [('declined ? ": " : "", declined ? declined : "");', '"", "");')])
    expect_fail("patch-published-unretained", "T9", B,
                [("hold their leases through it.\n"
                  "        ggml_sycl::retain_handles_until_event(std::move(replaced), ctx.stream()->ext_oneapi_submit_"
                  "barrier());\n", "hold their leases through it.\n")])
    expect_fail("patch-published-unsettled", "T9", B,
                [("        (void) ggml_sycl::moe_decode_direct_stamp_settle(stamp,", "        (void) (stamp,")])
    expect_fail("lookup-returns-true-elsewhere", "T9", B,
                [("    *out = held;\n    return *out != nullptr;",
                  "    *out = held;\n    if (*out != nullptr) {\n        return true;\n    }\n"
                  "    return *out != nullptr;")])

    # T10: the patch.
    held_param = "std::shared_ptr<const moe_hybrid_decode_record>   held,"
    expect_fail("patch-takes-held-mutable", "T10", B,
                [(held_param, "std::shared_ptr<moe_hybrid_decode_record>         held,")])
    expect_fail("patch-borrows-held", "T10", B,
                [(held_param, "const moe_hybrid_decode_record &                  held,")])
    expect_fail("patch-skips-plan-rule", "T10", B,
                [("    if (const char * why =\n"
                  "            ggml_sycl_moe_hybrid_decode_record_plan_refusal(src0, device, "
                  "static_cast<int64_t>(held->n_as), &counts)) {\n        return decline(why);\n    }\n", "")])
    expect_fail("patch-skips-frame-rule", "T10", B,
                [("    if (const char * why = ggml_sycl_moe_hybrid_decode_record_frame_refusal(ctx, src0, "
                  "held->all_host, "
                  "held->layout)) {\n        return decline(why);\n    }\n", "")])
    route_check = ("    if (held->route_block >= 0 && held->route_counts != "
                   "ggml_sycl::moe_route_counts_current(device)) {\n")
    expect_fail("patch-keeps-replaced-route-counts", "T10", B, [(route_check, "    if (false) {\n")])
    expect_fail("patch-checks-route-counts-on-every-tensor", "T10", B,
                [(route_check, "    if (held->route_counts != ggml_sycl::moe_route_counts_current(device)) {\n")])
    expect_fail("patch-declines-unnamed", "T10", B,
                [('            return decline("the resolved batch is not in the moved experts\' order");',
                  "            return false;")])
    expect_fail("patch-takes-no-reason", "T10", B,
                [("replaced,\n" + " " * 53 + "const char **" + " " * 37 + "reason) {",
                  "replaced) {\n    const char *  local_reason = nullptr;\n"
                  "    const char ** reason       = &local_reason;")])
    expect_fail("patch-skips-batch-size", "T10", B,
                [("    if (!resolved || resolved.batch.operands.size() != changed.size()) {", "    if (!resolved) {")])
    expect_fail("patch-skips-order-check", "T10", B,
                [("        if (operand.expert_id != changed[i]) {\n"
                  "            return decline(\"the resolved batch is not in the moved experts' order\");\n"
                  "        }\n", "")])
    expect_fail("patch-ignores-install-failure", "T10", B,
                [("    if (!ggml_sycl::moe_hybrid_record_install(patched->device_weight,",
                  "    if (false && !ggml_sycl::moe_hybrid_record_install(patched->device_weight,")])
    expect_fail("patch-writes-held-table", "T10", B,
                [("    patched->n_withdrawn = 0;\n\n    sycl::queue & queue = *ctx.stream();\n",
                  "    patched->n_withdrawn = 0;\n\n    sycl::queue & queue = *ctx.stream();\n"
                  "    if (held->table.valid()) {\n"
                  "        static_cast<const void **>(held->table.resolve().ptr)[0] = nullptr;\n    }\n")],
                RESOLVE_SAYS)
    expect_fail("patch-fills-held-table", "T10", B,
                [("    if (!ready.empty()) {\n        // The unchanged slots' weights",
                  "    (void) ggml_sycl_moe_hybrid_decode_table_fill(queue, device, extra->model_id, "
                  "patched->device_weight,\n"
                  "                                                  shape.substitute,\n"
                  "                                                  "
                  "const_cast<ggml_sycl::mem_handle *>(&held->table));\n"
                  "    if (!ready.empty()) {\n        // The unchanged slots' weights")])
    # The copy shares the held table's handle until its reset, so a write through it before then writes the held table.
    expect_fail("patch-writes-held-table-through-copy", "T10", B,
                [("        patched->table = {};\n",
                  "        static_cast<const void **>(patched->table.resolve().ptr)[changed.front()] = nullptr;\n"
                  "        patched->table = {};\n")],
                RESOLVE_SAYS)
    expect_fail("patch-releases-held-table-through-copy", "T10", B,
                [("        patched->table = {};\n",
                  "        (void) patched->table.reset_owned_allocation();\n        patched->table = {};\n")],
                COPY_TABLE_SAYS)
    # Each spelling of the copy's and the held record's table: a resolve is the resolve rule's to catch, and a use
    # that resolves nothing is the count rules'.
    reset = "        patched->table = {};\n"
    for name, use, says in (
            ("patch-resolves-copy-table-by-deref", "(void) (*patched).table.resolve();", RESOLVE_SAYS),
            ("patch-resolves-held-table-by-deref", "(void) (*held).table.resolve();", RESOLVE_SAYS),
            ("patch-releases-copy-table-by-deref", "(void) (*patched).table.reset_owned_allocation();",
             COPY_TABLE_SAYS),
            ("patch-releases-copy-table-by-get", "(void) patched.get()->table.reset_owned_allocation();",
             COPY_TABLE_SAYS),
            ("patch-releases-held-table-by-deref",
             "(void) const_cast<ggml_sycl::mem_handle &>((*held).table).reset_owned_allocation();", HELD_TABLE_SAYS),
            ("patch-releases-held-table-by-get",
             "(void) const_cast<ggml_sycl::mem_handle &>(held.get()->table).reset_owned_allocation();",
             HELD_TABLE_SAYS)):
        expect_fail(name, "T10", B, [(reset, "        " + use + "\n" + reset)], says)
    expect_fail("patch-ignores-layout-expert", "T10", B,
                [("held->layout_expert, counts.on_device == 0,", "-1, counts.on_device == 0,")])
    expect_fail("patch-resolves-every-expert", "T10", B,
                [("        src0, device, changed.data(), changed.size(), changed.size(), held->layout,",
                  "        src0, device, all.data(), all.size(), all.size(), held->layout,")])
    expect_fail("patch-writes-held-in-place", "T10", B,
                [("patched = std::make_shared<moe_hybrid_decode_record>(*held);",
                  "patched = std::shared_ptr<moe_hybrid_decode_record>("
                  "const_cast<moe_hybrid_decode_record *>(held.get()), "
                  "[](moe_hybrid_decode_record *) {});")])
    expect_fail("patch-skips-shape-rule", "T10", B,
                [("    if (const char * why = ggml_sycl::moe_hybrid_record_shape_refusal(patched->all_host, shape)) {",
                  "    if (const char * why = false ? \"x\" : nullptr) {")])
    expect_fail("patch-keeps-held-table", "T10", B,
                [("        patched->table = {};\n        if (!ggml_sycl_moe_hybrid_decode_table_fill(",
                  "        if (false && !ggml_sycl_moe_hybrid_decode_table_fill(")])
    expect_fail("patch-drops-held-ready", "T10", B,
                [("        if (held->has_ready) {\n            ready.push_back(held->ready);\n        }\n", "")])
    expect_fail("patch-gives-up-after-residency", "T10", B,
                [("    replaced->insert(replaced->end(),",
                  "    if (leases.empty()) {\n        return false;\n    }\n    replaced->insert(replaced->end(),")])
    expect_fail("patch-drops-replaced-leases", "T10", B,
                [("    replaced->insert(replaced->end(), std::make_move_iterator(leases.begin()), "
                  "std::make_move_iterator(leases.end()));\n", "")])
    expect_fail("patch-drops-held-table", "T10", B,
                [("    if (held->table.valid()) {\n        replaced->push_back(held->table);\n    }\n", "")])
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
    expect_fail("patch-asks-plan-itself", "T11", B,
                [("    ggml_sycl::expert_residency_counts counts;\n    if (const char * why =\n"
                  "            ggml_sycl_moe_hybrid_decode_record_plan_refusal(src0, device, static_cast<int64_t>",
                  "    ggml_sycl::expert_residency_counts counts;\n    (void) ggml_sycl_cache_plan_owner(ggml_sycl::"
                  "get_unified_cache_for_device(device))->has_host_experts(src0->name, 1, device);\n"
                  "    if (const char * why =\n"
                  "            ggml_sycl_moe_hybrid_decode_record_plan_refusal(src0, device, static_cast<int64_t>")])
    expect_fail("patch-checks-mmid-itself", "T11", B,
                [("    if (const char * why = ggml_sycl_moe_hybrid_decode_record_frame_refusal(ctx, src0, "
                  "held->all_host, "
                  "held->layout)) {\n        return decline(why);\n    }\n",
                  "    if (const char * why = ggml_sycl_moe_hybrid_decode_record_frame_refusal(ctx, src0, "
                  "held->all_host, "
                  "held->layout)) {\n        return decline(why);\n    }\n"
                  "    if (!held->all_host && moe_mmid_direct_route_runs_type(src0->type)) {\n"
                  "        return decline(\"mmid\");\n    }\n")])
    expect_fail("build-checks-cards-itself", "T11", B,
                [("    if (n_as <= 0 || ggml_sycl_get_tensor_usage(src0) != tensor_usage::MOE_EXPERT_WEIGHT) {",
                  "    if (n_as <= 0 || ggml_sycl_get_tensor_usage(src0) != tensor_usage::MOE_EXPERT_WEIGHT ||\n"
                  "        ggml_sycl_routable_device_count() != 1) {")])
    expect_fail("build-skips-frame-rule", "T11", B,
                [("    if (const char * why = ggml_sycl_moe_hybrid_decode_record_frame_refusal(ctx, src0, all_host, "
                  "layout)) {\n        return refuse(why);\n    }\n", "")])
    expect_fail("frame-rule-allows-two-cards", "T11", B,
                [("    if (ggml_sycl_routable_device_count() != 1) {",
                  "    if (ggml_sycl_routable_device_count() > 2) {")])
    expect_fail("frame-rule-drops-mmid", "T11", B,
                [("    if (!all_host && layout == GGML_LAYOUT_AOS && ggml_sycl_moe_mmid_route_reachable(ctx) &&\n"
                  "        moe_mmid_direct_route_runs_type(src0->type)) {\n"
                  "        return \"the MMID direct route runs this tensor's card and host experts\";\n    }\n", "")])
    expect_fail("frame-rule-mmid-for-all-host", "T11", B,
                [("    if (!all_host && layout == GGML_LAYOUT_AOS && ggml_sycl_moe_mmid_route_reachable(ctx) &&",
                  "    if (layout == GGML_LAYOUT_AOS && ggml_sycl_moe_mmid_route_reachable(ctx) &&")])
    expect_fail("build-asks-plan-itself", "T11", B,
                [("    if (const char * why = ggml_sycl_moe_hybrid_decode_record_plan_refusal(src0, device, n_as, "
                  "&counts)) {",
                  "    if (const char * why = ggml_sycl_cache_plan_owner("
                  "ggml_sycl::get_unified_cache_for_device(device))"
                  "->has_host_experts(src0->name, n_as, device) ? nullptr : \"no host expert\") {")])
    expect_fail("plan-rule-skips-residency", "T11", B,
                [("    *counts = has_host ? plan->count_planned_experts(src0->name, n_as, device) : "
                  "ggml_sycl::expert_residency_counts{};", "    *counts = ggml_sycl::expert_residency_counts{};")])
    expect_fail("build-forgets-layout-expert", "T11", B, [("    record->layout_expert = layout_expert;\n", "")])
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
                [("        if (erased) {\n            note_moe_storage_change(expert_id);\n        }\n"
                  "        return erased;",
                  "        if (erased) {\n            ++moe_expert_storage_generation;\n        }\n"
                  "        return erased;")])
    expect_fail("raw-bump-in-backend", "T8", B,
                [("        w.note_moe_storage_change(e);", "        ++w.moe_expert_storage_generation;")])
    expect_fail("mutation-names-one-expert", "T8", B,
                [("extra->weight().note_moe_storage_change(" + ALL + ");",
                  "extra->weight().note_moe_storage_change(0);")])
    expect_fail("fixture-clear-names-one-expert", "T8", B,
                [("extra.weight().note_moe_storage_change(" + ALL + ");",
                  "extra.weight().note_moe_storage_change(0);")])
    expect_fail("unread-note-site", "T8", B,
                [("    if (stored || table_dropped) {",
                  "    w.note_moe_storage_change(0);\n    if (stored || table_dropped) {")])
    expect_fail("mutation-notes-only-when-nothing-withdrawn", "T8", B,
                [("    extra->weight().note_moe_storage_change(" + ALL + ");\n",
                  "    if (expert_count == 0) {\n"
                  "        extra->weight().note_moe_storage_change(" + ALL + ");\n    }\n")])

    # T1, T10, T12: the drop's withdrawal, and the two places that refuse or clear a withdrawn record.
    expect_fail("lookup-classifies-without-withdrawn", "T1", B,
                [("selected_rows, held && held->n_withdrawn > 0);", "selected_rows, false);")])
    expect_fail("patch-keeps-withdrawn-count", "T10", B, [("    patched->n_withdrawn = 0;\n", "")])
    expect_fail("drop-resets-stamp", "T12", B,
                [("    if (stored && std::find(emptied.begin(),",
                  "    w.moe_hybrid_decode_stamp[device] = {};\n    if (stored && std::find(emptied.begin(),")])
    expect_fail("drop-withdraws-unnoted", "T12", B,
                [("    std::vector<int32_t> emptied;\n",
                  "    std::vector<int32_t> emptied;\n    std::vector<int32_t> unnoted;\n"),
                 ("                                                    emptied, scan_fallbacks)) {",
                  "                                                    unnoted, scan_fallbacks)) {")])
    expect_fail("withdraw-writes-held", "T12", B,
                [("    auto copy = std::make_shared<moe_hybrid_decode_record>(*held);",
                  "    auto * copy = const_cast<moe_hybrid_decode_record *>(held.get());")])
    expect_fail("withdraw-uncounted", "T12", B,
                [("        copy->n_withdrawn += ggml_sycl::moe_hybrid_record_withdraw(",
                  "        (void) ggml_sycl::moe_hybrid_record_withdraw(")])
    expect_fail("withdraw-slot-uncounted", "T12", B,
                [("        copy->n_withdrawn += ggml_sycl::moe_hybrid_record_withdraw_slot(",
                  "        (void) ggml_sycl::moe_hybrid_record_withdraw_slot(")])
    expect_fail("withdraw-walks-uncounted", "T12", B, [("        ++scan_fallbacks;\n        if (std::none_of(",
                                                         "        if (std::none_of(")])
    expect_fail("withdraw-never-walks", "T12", B,
                [("        if (std::none_of(held->lease.begin(), held->lease.end(), on_entry)) {\n"
                  "            return false;\n        }\n", "        return false;\n")])
    expect_fail("withdraw-slot-never-asked", "T12", B,
                [("    const bool at_slot = ggml_sycl::moe_hybrid_record_slot_matches("
                  "held->lease, expert_id, on_entry);", "    const bool at_slot = false;")])
    expect_fail("withdraw-slot-empties-every-match", "T12", B,
                [("    if (at_slot) {\n        copy->n_withdrawn", "    if (false) {\n        copy->n_withdrawn")])
    expect_fail("withdraw-unpublished", "T12", B, [("    held = std::move(copy);\n", "")])
    expect_fail("withdraw-matches-by-resolve", "T12", B,
                [("    const auto on_entry = [&](const ggml_sycl::mem_handle & h) {\n        return h.is_weight() &&",
                  "    const auto on_entry = [&](const ggml_sycl::mem_handle & h) {\n"
                  "        return h.resolve(device).ptr &&")])
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
    # T13: what the expert storage moved under.
    patch_moved = ("    if (patched_ok && weight.moe_expert_storage_generation != storage_generation) {\n"
                   "        ggml_sycl_moe_hybrid_decode_record_append_handles(*patched, replaced);\n"
                   "        ggml_sycl::retain_handles_until_event(std::move(replaced), "
                   "ctx.stream()->ext_oneapi_submit_barrier());\n"
                   "        patched.reset();\n        patched_ok = false;\n"
                   "        declined   = \"the expert storage moved while the patch ran\";\n    }\n")
    patch_count = "        g_moe_hybrid_record_trace.patch(plan, patched_ok, changed.size());\n"
    expect_fail("patch-served-after-storage-moved", "T13", B, [(patch_moved, "")])
    expect_fail("patch-storage-checked-after-count", "T13", B,
                [(patch_moved, ""), (patch_count, patch_count + patch_moved)])
    expect_fail("patch-discard-unretained", "T13", B,
                [("        ggml_sycl_moe_hybrid_decode_record_append_handles(*patched, replaced);\n"
                  "        ggml_sycl::retain_handles_until_event(std::move(replaced), "
                  "ctx.stream()->ext_oneapi_submit_barrier());\n", "")])
    build_moved_if = "    if (weight.moe_expert_storage_generation != storage_generation) {\n        if (built) {\n"
    build_moved = (build_moved_if
                   + "            std::vector<ggml_sycl::mem_handle> handles;\n"
                   "            ggml_sycl_moe_hybrid_decode_record_append_handles(*built, handles);\n"
                   "            ggml_sycl::retain_handles_until_event(std::move(handles), "
                   "ctx.stream()->ext_oneapi_submit_barrier());\n"
                   "            built.reset();\n        }\n"
                   "        outcome = ggml_sycl::MOE_DECODE_DIRECT_OUTCOME_RETRY;\n    }\n")
    outcome_count = "        g_moe_hybrid_record_trace.build_outcome(outcome);\n"
    expect_fail("build-served-after-storage-moved", "T13", B, [(build_moved, "")])
    expect_fail("build-storage-checked-after-count", "T13", B,
                [(build_moved, ""), (outcome_count, outcome_count + build_moved)])
    expect_fail("build-storage-check-eligible-only", "T13", B,
                [(build_moved_if,
                  "    if (outcome == ggml_sycl::MOE_DECODE_DIRECT_OUTCOME_ELIGIBLE &&\n"
                  "        weight.moe_expert_storage_generation != storage_generation) {\n        if (built) {\n")])
    expect_fail("build-discard-unretained", "T13", B,
                [("            ggml_sycl_moe_hybrid_decode_record_append_handles(*built, handles);\n"
                  "            ggml_sycl::retain_handles_until_event(std::move(handles), "
                  "ctx.stream()->ext_oneapi_submit_barrier());\n", "")])
    # T14: the build's replacement of the held record.
    build_retain = ("        ggml_sycl_moe_hybrid_decode_record_append_handles(*held, handles);\n"
                    "        ggml_sycl::retain_handles_until_event(std::move(handles), "
                    "ctx.stream()->ext_oneapi_submit_barrier());\n")
    expect_fail("build-replaced-unretained", "T14", B,
                [(build_retain, "        ggml_sycl_moe_hybrid_decode_record_append_handles(*held, handles);\n")])
    build_replace = ("    if (settled == ggml_sycl::MOE_DECODE_DIRECT_OUTCOME_ELIGIBLE) {\n"
                     "        held = std::move(built);\n    } else {\n        held.reset();\n    }\n")
    retain_block = ("    if (held) {\n"
                    "        // The replaced record's table and device leases may back a table copy or kernel still on "
                    "the queue.\n"
                    "        std::vector<ggml_sycl::mem_handle> handles;\n" + build_retain + "    }\n")
    expect_fail("build-retains-after-replace", "T14", B,
                [(retain_block, ""), (build_replace, build_replace + retain_block)])
    expect_fail("docs-row-counts-three-notes", "T7", D,
                [("two or three per tensor, so about 21 to 32 moves", "about three per tensor, so about 21 moves")])
    expect_fail("docs-row-builds-every-miss", "T7", D,
                [("a plan that no patch answers builds once", "a plan builds once")])
    expect_fail("docs-patch-row-any-value", "T7", D,
                [("parses as a non-zero integer (`on` and `true` parse as 0)", "any non-zero value")])
    expect_fail("docs-row-lacks-withdrawn", "T7", D,
                [("patch_unlogged= withdrawn=` line", "patch_unlogged=` line"), (" `withdrawn=` counts", " counts")])
    expect_fail("docs-row-blames-own-allocations", "T7", D,
                [("by an expert move on another thread that overlapped the lookup",
                  "by a loan or evict its own allocations set off")])

    try:
        mutate(B, [("this anchor is in no source", "x")])
        failures.append("stale-anchor-not-reported")
        print("  self-test: a stale mutant anchor is NOT reported")
    except MutantNotApplied:
        print("  self-test: a stale mutant anchor is reported")

    try:
        check_all(raws)
        print("  self-test: unmodified tree passes")
    except ContractError as e:
        print(f"  self-test: unmodified tree FAILS: {e}")
        failures.append("unmodified")

    if failures:
        print(f"SELF-TEST FAIL: {', '.join(failures)}")
        return 1
    print(f"SELF-TEST PASS: {n_mutants} mutants caught by their pins, unmodified tree passes")
    return 0


def check_all(raws: dict[str, str]) -> None:
    check(raws["backend"], raws["header"], raws["docs"], raws["common"], raws["log"], raws["includers"])


def record_includers() -> str:
    """The sources that include moe-hybrid-record.hpp, one repo-relative path per line."""
    found = set()
    for pattern in INCLUDER_GLOBS:
        for path in ROOT.glob(pattern):
            if RECORD_INCLUDE in included_headers(blank_comments(path.read_text(errors="replace"))):
                found.add(path.relative_to(ROOT).as_posix())
    return "\n".join(sorted(found))


def main(argv: list[str]) -> int:
    raws = {}
    for key, path in (("backend", BACKEND), ("header", HEADER), ("docs", DOCS), ("common", COMMON),
                      ("log", LOG_HEADER)):
        try:
            raws[key] = path.read_text()
        except FileNotFoundError:
            print(f"FAIL: {path.relative_to(ROOT)} does not exist")
            return 1
    raws["includers"] = record_includers()
    if "--self-test" in argv:
        return self_test(raws)
    try:
        check_all(raws)
    except ContractError as e:
        print(e)
        return 1
    print("PASS: the hybrid decode record lookup is classified once and counted at every exit, at WARN")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
