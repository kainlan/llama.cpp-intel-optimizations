#!/usr/bin/env python3
"""Source gate for the type-keyed batched MoE prompt executor (llama.cpp-348c).

The batched oneDNN prompt executor (the try_pp_onednn_f16_batched lambda in ggml_sycl_mul_mat_id) used to
take MXFP4 experts only. Its f16 arm now admits a (type, layout) pair when a kernel decodes one expert of that
materialized layout to f16: every quantized type with an AOS dequant-to-f16 kernel except the two that only their
direct kernel may run on a device, and SOA too for the types the planner materializes SOA that have a per-expert SOA
dequant. These claims pin that shape:

  (a) one support table: the claim predicate and the lambda's candidate gate both ask
      ggml_sycl_moe_pp_batched_takes_layout(), which asks moe_pp_batched_supports_layout() in moe-mmvq-tables.hpp,
      and neither tests GGML_TYPE_MXFP4 itself;
  (b) the f16 table lists each required type on AOS, Q4_0, Q8_0, Q4_K and Q6_K on AOS and SOA, MXFP4 on SOA only,
      never the direct-kernel types, its non-MXFP4 types and REQUIRED_AOS are one set (a type added to the table
      without a pin is red), and every type it lists has a case in convert.cpp's
      ggml_get_to_fp16_sycl_for_layout; that case decodes SOA with a reorder kernel for Q4_0, Q4_K and Q6_K, and
      since it answers Q8_0 with the AOS kernel for every layout, the lambda takes Q8_0 SOA to
      dequantize_row_q8_0_soa_to_fp16_rowmajor and never asks it; the WOQ table is MXFP4 on SOA and XMX_TILED;
  (c) the Q4_K device-weight CPU fallback (a D2H copy per dispatch) does not fire for a prompt the executor claims,
      and the mechanism is precedence, not a term on the fallback: a claimed prompt drops the planner's prompt route
      (planner_needs_prompt_route), so an all-local claimed prompt never enters the hybrid world where the fallback
      lives, and the fallback itself stays exactly Q4_K for the hybrid ops the executor never runs;
  (d) decode keeps its ne12 == 1 / ids->ne[1] > 1 terms: the hybrid-with-plan disjunct, the two claim call sites
      and the lambda's single-token refusal;
  (e) one refusal enum with a name for every cause, ending in COUNT, which the per-(device, cause) latch
      static_asserts fits its 32-bit mask and refuses at or past; every refusal in the lambda names one; one WARN line
      per (device, cause), printed by one helper that the GEMM-stage failure uses too, so every refusal reads
      "[PP-MOE] refused". A cause reached after the claim steered the op past the MMVQ probes
      (moe_pp_batched_refusal_after_claim, exactly POST_CLAIM) prints at default verbosity; a precheck cause prints
      only under GGML_SYCL_MOE_PP_ONEDNN_F16_BATCHED_TRACE;
  (f) one [PP-MOE] INFO line per (device, type, layout), naming the type, layout, experts per dispatch and the
      planned slot;
  (g) the MXFP4 profiler label strings are byte-identical and keep their counts, the non-MXFP4 labels are moe.pp.*,
      and the MXFP4 dispatch-record profile stays MXFP4-only;
  (h) the capability query admits a prompt-phase op the executor takes only after the MMVQ check (so no type that
      has an MMVQ-ID kernel changes kernel), and MUL_MAT_ID admission takes a type with no MMVQ-ID kernel only for
      a prompt the executor claims, which is only a layer whose every expert is planned on this device: the claim's
      planned-layout lookup refuses a layer with a host or other-device expert, so a hybrid layer is never admitted;
  (i) the f16 arm runs in the weight slot the inventory planned, never a wider one: the backend keeps the planned
      slot size as the inventory gave it (no plan-time widening, which displaced routed experts with no cost
      decision), the planner site that publishes the ring's sizes computes no expert-count or f16 formula and hands
      the unified cache the inventory's slot, the arm asks the admission for one f16 expert, and it groups the active
      experts by what the slot it claimed holds;
  (j) one opt-out knob, GGML_SYCL_MOE_PP_F16_BATCHED_AOS, spelled once, read once, consulted by the support helper,
      dropping every non-MXFP4 entry (SOA ones included), so it leaves exactly the MXFP4 executor, registered in
      sycl_env_settings[] so the non-default report names it, and catalogued;
  (k) the claim takes a planned layout only where the cache holds one entry per expert at it: the claim predicate
      reads device-buffer storage from the weight's own buffer and asks moe_pp_batched_has_expert_entries, which
      declines exactly a non-AOS layout in a device buffer (GGML_SYCL_WEIGHTS_EVICTABLE=0 reorders such a tensor
      into one block over every expert) and never MXFP4, whose route is unchanged;
  (l) GGML_SYCL_MOE_PP_ONEDNN_F16 is read once, by a common.hpp accessor, and the executor's support helper asks it,
      so with it at 0 the claim takes nothing the executor would then decline, and the executor names the knob as
      route-off rather than as a candidate miss; the catalog has its row;
  (m) the f16 arm's groups alternate between the slot's two halves and chain through events: before a group
      dequantizes into a half, the stream waits, on the device, for the GEMMs that last read that half, every GEMM
      is recorded as a reader of its half, the dequant and the GEMM address the same half, and the lambda holds no
      host wait outside the debugging compare arm's CPU readback (GGML_SYCL_MOE_PP_ONEDNN_F16_BATCHED_COMPARE).
      When the slot holds every active expert (one group), a windowed op's phases each dequantize into their own
      region and are given no wait on an earlier phase's GEMMs: one pure predicate in moe-mmvq-tables.hpp
      (moe_pp_batched_phases_reuse_buffers, true unless there is exactly one group) decides both the phase's offset
      (moe_pp_batched_phase_weight_offset) and whether the wait is submitted, the executor asks it and never
      re-derives it, and the phase's region is asserted to lie in the claimed slot; the route log's weight figure
      asks the same predicate, so it reports the regions the phases really used;
  (n) the dequant profiler label's name comes from a table built once per type, never from a string the dispatch
      builds;
  (o) every type the f16 table takes runs in the device test (test-moe-pp-batched-dequant.cpp's k_types), or is named
      in UNTESTED_ON_DEVICE with its reason, and never both; the untested ones are exactly the table's types that
      ggml_quantize_requires_imatrix names, since the fixture quantizes with no importance matrix;
  (p) the device test counts each case's own lines (a [PP-MOE] line in the type's first case only, once per run, and
      a success line per case), and its cases fit the backend's per-process cap on the success line, whose value
      the test copies;
  (q) one debugging knob, GGML_SYCL_MOE_PP_F16_GROUP_EXPERTS_MAX, caps the experts in one f16 group: spelled once,
      read once into a static, registered in sycl_env_settings[] with the report's integer set from the accessor,
      and catalogued; the success line names the group count, experts per group and buffers. The device test runs
      its cases at the slot's own grouping and requires at least three groups in each; it also runs one Q4_0 case at
      each cap in k_group_caps (1 and 3), each in a child process because the knob is read once per process, and
      requires exactly the groups that cap gives, that many experts per group, and both halves. A failing child fails
      the run: the child exits 1 when its case fails, the parent counts any child that does not exit 0, and it
      counts that as a failed case;
  (r) a GEMM stage that fails partway drains before it lets go: it barriers on the GEMM events it already submitted
      (a plain barrier, after the dequants, when there are none), holds every expert's lease until that barrier,
      and only then finishes or releases its scratch claim, whose release marker follows the barrier in queue order.
      The oneDNN primitive is not ordered with the queue (llama.cpp-dboi), so without the barrier a GEMM still
      running could write the slot after the next dispatch claims it. One helper holds the leases, for this path
      and for both success branches, and one helper drains (drain_submitted_phases), for this path and for a
      deferred expert the phase loop declines (llama.cpp-8epx).
  (s) one type question for a host-staged weight (llama.cpp-3460): moe_pp_batched_takes_staged_type in
      moe-mmvq-tables.hpp is exactly an MMVQ-ID kernel (moe_mmvq_admission_supports_type) and an f16 row at AOS. The
      staged claim asks it, beside the runtime knobs (ggml_sycl_moe_pp_batched_takes_layout at AOS), and the cost
      model's device-rate seed (moe_exec_dev_gflops_seed) asks it before it seeds any type but MXFP4; neither asks
      either table term itself. So no type is seeded at the executor's rate while the claim refuses it (IQ1_S, IQ1_M
      and IQ4_XS have the f16 row and no MMVQ-ID kernel), and (h) holds for the staged claim too: it never takes a
      type with no MMVQ-ID kernel, so such a type is never admitted on a hybrid layer.

Comments are stripped and whitespace is squeezed before matching, so a comment cannot satisfy a probe and the
probes are written with single spaces. Every claim has at least one mutant below that must turn it red.
"""
import functools
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SYCL_DIR = ROOT / "ggml" / "src" / "ggml-sycl"
FILES = {
    "sycl": SYCL_DIR / "ggml-sycl.cpp",
    "tables": SYCL_DIR / "moe-mmvq-tables.hpp",
    "cost": SYCL_DIR / "moe-exec-cost.hpp",
    "convert": SYCL_DIR / "convert.cpp",
    "gemm": SYCL_DIR / "gemm.hpp",
    "common": SYCL_DIR / "common.hpp",
    "layer_plan": SYCL_DIR / "moe-layer-plan.hpp",
    "catalog": ROOT / "docs" / "backend" / "sycl-env-vars.md",
    "device_test": SYCL_DIR / "tests" / "test-moe-pp-batched-dequant.cpp",
    "ggml_c": ROOT / "ggml" / "src" / "ggml.c",
}

# Every quantized type ggml_get_to_fp16_sycl_for_layout decodes on AOS, except MXFP4 (its own SOA arm) and Q1_0 and
# NVFP4 (FORBIDDEN below). The first thirteen are the brief's list and the Qwen3.8 GSQ-RCO census (see the memory "a
# GGUF file name is not its expert quant type"); the last seven complete the f16 arm's rule in moe-mmvq-tables.hpp,
# that every type the converter decodes is admitted but the three above (llama.cpp-348c).
REQUIRED_AOS = [
    "GGML_TYPE_Q4_0", "GGML_TYPE_Q8_0", "GGML_TYPE_Q4_K", "GGML_TYPE_Q5_K", "GGML_TYPE_Q6_K", "GGML_TYPE_IQ3_XXS",
    "GGML_TYPE_IQ3_S", "GGML_TYPE_IQ4_XS", "GGML_TYPE_IQ4_NL", "GGML_TYPE_Q2_0", "GGML_TYPE_IQ2_S",
    "GGML_TYPE_IQ2_XS", "GGML_TYPE_IQ2_XXS", "GGML_TYPE_Q4_1", "GGML_TYPE_Q5_0", "GGML_TYPE_Q5_1", "GGML_TYPE_Q2_K",
    "GGML_TYPE_Q3_K", "GGML_TYPE_IQ1_S", "GGML_TYPE_IQ1_M",
]
# The types the planner may materialize SOA that have a per-expert SOA dequant to f16 (the B50 loads Q8_0 and Q6_K
# experts SOA; the device test's RED run refused both).
REQUIRED_SOA = ["GGML_TYPE_Q4_0", "GGML_TYPE_Q8_0", "GGML_TYPE_Q4_K", "GGML_TYPE_Q6_K"]
AOS_SOA_EXPR = "layout == GGML_LAYOUT_AOS || layout == GGML_LAYOUT_SOA"
# Q1_0 and NVFP4 run on a device only through their direct kernel (moe_type_needs_direct_kernel); float experts are
# not a dequant case.
FORBIDDEN = ["GGML_TYPE_Q1_0", "GGML_TYPE_NVFP4", "GGML_TYPE_F16", "GGML_TYPE_F32", "GGML_TYPE_BF16"]
AOS_EXPR = "layout == GGML_LAYOUT_AOS"
KNOB = "GGML_SYCL_MOE_PP_F16_BATCHED_AOS"
KNOB_CONST = "GGML_SYCL_MOE_PP_F16_BATCHED_AOS_ENV"
F16_KNOB = "GGML_SYCL_MOE_PP_ONEDNN_F16"
# The refusal causes the batched executor reaches only after the claim has steered the op past the MMVQ probes and
# the fused routes: a decline there loses the fast path, so it is reported at default verbosity. The others are
# prechecks (the op was never this executor's, or there is nothing to run), plus SCRATCH_UNPLANNED, whose admission
# refusal already warns at default verbosity with the sizes, once per process.
POST_CLAIM = [
    "EXPERT_NOT_LOCAL", "EXPERT_LAYOUT", "XMX_TILE_CAPS", "F16_TILED", "NO_DEQUANT", "SCRATCH_SIZE", "SCRATCH_CACHE",
    "SCRATCH_RESERVE", "SCRATCH_SLOT", "SCRATCH_SLOT_SIZE", "GEMM",
]
# A decline after the dispatch took a staged ring slot, or on a weight the ring staged in full for this executor, is
# printed whatever its cause (llama.cpp-3460): its staged experts then run on the CPU path.
WARN_CONDITION = ("(pp_f16_batched_trace || moe_pp_batched_refusal_after_claim(refusal) || pp_staged_experts > 0 || "
                  "pp_onednn_batched_claims_staged_layer) && "
                  "ggml_sycl_moe_pp_batched_refusal_first(ctx.device, refusal)")
# Types the f16 table takes that the device test does not run, each with its reason. The fixture quantizes its
# weights with no importance matrix, so these three are the whole list.
GROUP_KNOB = "GGML_SYCL_MOE_PP_F16_GROUP_EXPERTS_MAX"
GROUP_KNOB_CONST = "GGML_SYCL_MOE_PP_F16_GROUP_EXPERTS_MAX_ENV"
# The f16 arm's phase loop, from its head to the barrier on every GEMM event that ends it.
PHASE_LOOP_HEAD = "for (size_t p = 0; p < phases.size() && !gemm_declined; ++p) {"
PHASE_LOOP_END = "stream->ext_oneapi_submit_barrier(gemm_events);"
# A phase's region of the f16 slot and its wait on the readers of its buffer, both decided by the header's predicate.
REUSE_FN = "moe_pp_batched_phases_reuse_buffers"
REUSE_BODY = "{ return groups.groups != 1; }"
OFFSET_FN = "moe_pp_batched_phase_weight_offset"
OFFSET_BODY = ("return moe_pp_batched_phases_reuse_buffers(groups) ? phase_buffer * half_elems : "
               "phase_begin * weight_elems;")
PHASE_OFFSET = ("sycl::half * phase_weights = batched_weights + "
                "moe_pp_batched_phase_weight_offset(f16_groups, phase.begin, phase.buffer, half_elems, weight_elems);")
PHASE_BOUND = ("GGML_ASSERT(pp_woq_arm || (static_cast<size_t>(phase_weights - batched_weights) + "
               "(phase.end - phase.begin) * weight_elems) * sizeof(sycl::half) <= reserved.weight_size);")
# The route log's weight figure: the f16 arm's buffers when its phases reuse them, else every active expert's region.
WEIGHT_USED = ("const size_t f16_used_elems = moe_pp_batched_phases_reuse_buffers(f16_groups) ? "
               "phase_buffers * half_elems : active_count * weight_elems; "
               "const size_t weight_used_bytes = pp_woq_arm ? weight_bytes : f16_used_elems * sizeof(sycl::half);")
PHASE_WAIT = ("if (moe_pp_batched_phases_reuse_buffers(f16_groups) && !half_readers[phase.buffer].empty()) { "
              "stream->ext_oneapi_submit_barrier(half_readers[phase.buffer]); half_readers[phase.buffer].clear(); }")
# The debugging compare arm (GGML_SYCL_MOE_PP_ONEDNN_F16_BATCHED_COMPARE), the one block in the lambda that may wait on
# the host: it copies the result back and checks it on the CPU.
COMPARE_HEAD = "if (batched_compare && !active_experts.empty()) {"
# A host wait, spelled as the stream-enqueue gate spells it.
ANY_WAIT_RE = re.compile(r"\bwait\s*\(|\bwait_and_throw\s*\(")
# The planner site: the inventory's ring sizes become the backend's globals and the unified cache's planned scratch.
PLAN_SITE_HEAD = "g_tensor_inventory_pp_moe_onednn_weight_slot_bytes = inventory->pp_moe_onednn_weight_slot_bytes;"
PLAN_SITE_CALL = ("unified_cache_set_planned_pp_moe_onednn_scratch( ctx->device, "
                  "g_tensor_inventory_pp_moe_onednn_weight_slot_bytes,")
# The device test's capped cases: one Q4_0 case per cap, each in its own process.
GROUP_CAPS = "constexpr int k_group_caps[] = { 1, 3 };"
GROUP_EXACT = "groups != want_groups || group_experts != size_t(group_cap) || buffers != 2"
# The GEMM stage's failure path: its head, the barrier on the GEMMs already submitted, the leases held to it, and the
# scratch claim's release.
FAIL_HEAD = "auto gemm_stage_failed = [&](const char * what) -> bool {"
DRAIN_HEAD = "auto drain_submitted_phases = [&]() {"
DRAIN_CALL = "drain_submitted_phases(); if (tiled_claimed_op) {"
FAIL_BARRIER = ("const sycl::event drained = gemm_events.empty() ? stream->ext_oneapi_submit_barrier() : "
                "stream->ext_oneapi_submit_barrier(gemm_events);")
FAIL_RETAIN = "retain_expert_leases_until(drained);"
CLAIM_RELEASE = "batched_scratch_claim.finish_or_release();"
LEASE_HELPER = "auto retain_expert_leases_until = [&](const sycl::event & until) {"
STAGED_TYPE = "moe_pp_batched_takes_staged_type"
STAGED_TYPE_BODY = ("{ return moe_mmvq_admission_supports_type(type) && "
                    "moe_pp_batched_f16_supports_layout(type, GGML_LAYOUT_AOS); }")
STAGED_CLAIM_HEAD = "planned = plan != nullptr && moe_pp_batched_takes_staged_type(weight->type) && "
STAGED_CLAIM_KNOBS = "ggml_sycl_moe_pp_batched_takes_layout(weight->type, GGML_LAYOUT_AOS) && "
SEED_GATE = "if (!moe_pp_batched_takes_staged_type(t)) { return MOE_EXEC_DEV_GFLOPS_PER_ROW_SEED; }"
SEED_MXFP4 = "if (t == GGML_TYPE_MXFP4) { return 3990.0; }"
TABLE_TERMS = ("moe_mmvq_admission_supports_type(", "moe_pp_batched_f16_supports_layout(")
UNTESTED_ON_DEVICE = {
    "GGML_TYPE_IQ1_S": "needs an imatrix to quantize (ggml_quantize_requires_imatrix)",
    "GGML_TYPE_IQ2_XXS": "needs an imatrix to quantize (ggml_quantize_requires_imatrix)",
    "GGML_TYPE_IQ2_XS": "needs an imatrix to quantize (ggml_quantize_requires_imatrix)",
}


def strip_comments(src):
    """Drop // and /* */ comments, leaving string and char literals intact (a quoted "//" is not a comment)."""
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c in "\"'":
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            out.append(src[i : j + 1])
            i = j + 1
        elif src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            i = n if j < 0 else j + 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def squeeze(src):
    return re.sub(r"\s+", " ", src)


def load():
    """Squeezed text per file; C++ sources lose their comments first, the Markdown catalog is kept whole."""
    out = {}
    for key, path in FILES.items():
        text = path.read_text()
        out[key] = squeeze(text if path.suffix == ".md" else strip_comments(text))
    return out


def all_backend_sources():
    for path in sorted(SYCL_DIR.rglob("*")):
        if path.suffix in (".cpp", ".hpp", ".h") and "tests" not in path.relative_to(SYCL_DIR).parts:
            yield path


@functools.lru_cache(maxsize=None)
def backend_texts():
    """(comment-stripped, squeezed) text of every backend source, read once: the mutants never touch these files."""
    out = []
    for path in all_backend_sources():
        stripped = strip_comments(path.read_text())
        out.append((stripped, squeeze(stripped)))
    return tuple(out)


def match_close(text, open_pos):
    """Index just past the bracket matching text[open_pos], skipping string and char literals; -1 if unmatched."""
    pairs = {"(": ")", "{": "}", "[": "]"}
    opener = text[open_pos]
    closer = pairs[opener]
    depth = 0
    i, n = open_pos, len(text)
    while i < n:
        c = text[i]
        if c in "\"'":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def definitions(text, name):
    """Bodies (including braces) of every function definition named `name` (a call or declaration is skipped)."""
    bodies = []
    for m in re.finditer(r"\b" + re.escape(name) + r" ?\(", text):
        params_end = match_close(text, m.end() - 1)
        if params_end < 0:
            continue
        rest = text[params_end:]
        k = re.match(r" ?(const )?(noexcept )?\{", rest)
        if not k:
            continue
        open_pos = params_end + k.end() - 1
        close = match_close(text, open_pos)
        if close > 0:
            bodies.append(text[open_pos:close])
    return bodies


def one_definition(text, name):
    bodies = definitions(text, name)
    return bodies[0] if len(bodies) == 1 else None


def claim_predicate_body(sycl):
    """The three-argument ggml_sycl_moe_pp_onednn_batched_claims_tensor (the one with the plan) -- the real body."""
    for body in definitions(sycl, "ggml_sycl_moe_pp_onednn_batched_claims_tensor"):
        if "moe_classify_tensor(" in body:
            return body
    return None


def lambda_body(sycl):
    m = re.search(r"auto try_pp_onednn_f16_batched = \[&\]\(\) -> bool \{", sycl)
    if not m:
        return None
    close = match_close(sycl, m.end() - 1)
    return sycl[m.end() - 1 : close] if close > 0 else None


def statement(text, head, start=None):
    """The statement starting at `head` (or at offset `start`) up to its terminating ';' at bracket depth 0, or None."""
    if start is None:
        start = text.find(head)
    if start < 0:
        return None
    depth = 0
    i = start
    while i < len(text):
        c = text[i]
        if c in "\"'":
            j = i + 1
            while j < len(text) and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == ";" and depth == 0:
            return text[start : i + 1]
        i += 1
    return None


def enclosing_if_condition(text, pos):
    """Condition text of the innermost `if (...) { ... }` whose block contains pos, or None."""
    best = None
    for m in re.finditer(r"\bif \(", text[:pos]):
        cond_end = match_close(text, m.end() - 1)
        if cond_end < 0 or cond_end > pos:
            continue
        rest = text[cond_end:]
        k = re.match(r" ?\{", rest)
        if not k:
            continue
        open_pos = cond_end + k.end() - 1
        close = match_close(text, open_pos)
        if open_pos < pos < close:
            best = text[m.end() : cond_end - 1]
    return best


def switch_cases(body):
    """{case label: return expression} for a switch whose arms are case-label runs ending in one return."""
    out = {}
    pending = []
    for m in re.finditer(r"case (GGML_TYPE_\w+) ?:|return ([^;]*);|default ?:", body):
        if m.group(1):
            pending.append(m.group(1))
        elif m.group(2) is not None:
            for label in pending:
                out[label] = m.group(2).strip()
            pending = []
        else:
            pending = ["default"]
    return out


def case_body(switch_text, label):
    """Text after `case <label>:` up to the next type case or default, or None."""
    m = re.search(r"case " + re.escape(label) + r" ?:", switch_text)
    if not m:
        return None
    nxt = re.search(r"case GGML_TYPE_\w+ ?:|default ?:", switch_text[m.end():])
    return switch_text[m.end() : m.end() + (nxt.start() if nxt else len(switch_text))]


def enum_values(text, enum_head):
    start = text.find(enum_head)
    if start < 0:
        return None
    open_pos = text.find("{", start)
    close = match_close(text, open_pos)
    names = []
    for item in text[open_pos + 1 : close - 1].split(","):
        item = item.strip()
        if item:
            names.append(item.split("=")[0].strip())
    return names


# ---------------------------------------------------------------------------------------------------------------------
# Claims. Each returns a list of failures (empty = holds).
# ---------------------------------------------------------------------------------------------------------------------


def claim_a_one_table(s):
    fails = []
    for fn in ("moe_pp_batched_f16_supports_layout", "moe_pp_batched_woq_supports_layout",
               "moe_pp_batched_supports_layout"):
        n = len(definitions(s["tables"], fn))
        if n != 1:
            fails.append(f"{fn} is defined {n} times in moe-mmvq-tables.hpp, want 1")
        if definitions(s["sycl"], fn):
            fails.append(f"{fn} has a second definition in ggml-sycl.cpp")
    takes = one_definition(s["sycl"], "ggml_sycl_moe_pp_batched_takes_layout")
    if takes is None:
        fails.append("ggml_sycl_moe_pp_batched_takes_layout is not defined exactly once in ggml-sycl.cpp")
    else:
        if "moe_pp_batched_supports_layout(" not in takes:
            fails.append("ggml_sycl_moe_pp_batched_takes_layout does not ask moe_pp_batched_supports_layout")
        if "GGML_TYPE_" in takes:
            fails.append("ggml_sycl_moe_pp_batched_takes_layout names a type itself instead of asking the table")
    claim = claim_predicate_body(s["sycl"])
    if claim is None:
        fails.append("the plan-taking ggml_sycl_moe_pp_onednn_batched_claims_tensor definition was not found")
    else:
        if "ggml_sycl_moe_pp_batched_takes_layout(" not in claim:
            fails.append("the claim predicate does not ask ggml_sycl_moe_pp_batched_takes_layout")
        if "GGML_TYPE_" in claim:
            fails.append("the claim predicate still tests a GGML_TYPE_ itself")
    cand = statement(s["sycl"], "const bool pp_batched_candidate =")
    if cand is None:
        fails.append("the lambda's candidate gate `const bool pp_batched_candidate = ...;` was not found")
    else:
        if "ggml_sycl_moe_pp_batched_takes_layout(src0->type, route_layout)" not in cand:
            fails.append("the candidate gate does not ask ggml_sycl_moe_pp_batched_takes_layout(src0->type, ...)")
        if "src0->type ==" in cand or "GGML_TYPE_MXFP4" in cand:
            fails.append("the candidate gate still tests the src0 type itself")
    lam = lambda_body(s["sycl"])
    if lam is None:
        fails.append("the try_pp_onednn_f16_batched lambda was not found")
    else:
        if "!pp_batched_candidate" not in lam:
            fails.append("the lambda does not refuse on !pp_batched_candidate")
        if "pp_mxfp4_soa_f16_candidate" in lam:
            fails.append("the lambda still reads the MXFP4-only pp_mxfp4_soa_f16_candidate")
    return fails


def claim_b_table_types(s):
    fails = []
    f16 = one_definition(s["tables"], "moe_pp_batched_f16_supports_layout")
    woq = one_definition(s["tables"], "moe_pp_batched_woq_supports_layout")
    for_layout = one_definition(s["convert"], "ggml_get_to_fp16_sycl_for_layout")
    if f16 is None or woq is None or for_layout is None:
        return ["the f16 table, the WOQ table or convert.cpp's ggml_get_to_fp16_sycl_for_layout was not found"]
    cases = switch_cases(f16)
    for t in REQUIRED_AOS:
        want = AOS_SOA_EXPR if t in REQUIRED_SOA else AOS_EXPR
        if cases.get(t) != want:
            fails.append(f"the f16 table does not answer {t} with `{want}` (got {cases.get(t)!r})")
    for t, expr in cases.items():
        if expr == AOS_SOA_EXPR and t not in REQUIRED_SOA:
            fails.append(f"the f16 table takes {t} on SOA, but no per-expert SOA dequant is pinned for it")
    if cases.get("GGML_TYPE_MXFP4") != "layout == GGML_LAYOUT_SOA":
        fails.append(f"the f16 table must take MXFP4 on SOA only (got {cases.get('GGML_TYPE_MXFP4')!r})")
    for t in FORBIDDEN:
        if t in cases:
            fails.append(f"the f16 table names {t}")
    unpinned = sorted(t for t in cases if t not in ("default", "GGML_TYPE_MXFP4") and t not in REQUIRED_AOS)
    if unpinned:
        fails.append(f"the f16 table takes {unpinned}, which REQUIRED_AOS does not pin")
    if cases.get("default") != "false":
        fails.append("the f16 table's default must be `return false`")
    converter_types = set(re.findall(r"case (GGML_TYPE_\w+) ?:", for_layout))
    for t, expr in cases.items():
        if expr in (AOS_EXPR, AOS_SOA_EXPR) and t not in converter_types:
            fails.append(f"the f16 table takes {t} but ggml_get_to_fp16_sycl_for_layout has no case for it")
    for t in REQUIRED_SOA:
        if t == "GGML_TYPE_Q8_0":
            continue
        body = case_body(for_layout, t) or ""
        if "use_reorder" not in body or "_reorder" not in body.replace("use_reorder", ""):
            fails.append(f"ggml_get_to_fp16_sycl_for_layout's {t} case has no reorder kernel for SOA")
    lam = lambda_body(s["sycl"]) or ""
    # Each expert's reader follows its own layout (llama.cpp-3460): a staged expert is AOS beside device experts in the
    # route's layout, so the converter is picked per expert, never from the first one.
    q8_soa = ("return src0->type == GGML_TYPE_Q8_0 && expert.layout == GGML_LAYOUT_SOA;")
    if q8_soa not in lam or "active_experts.front().layout" in lam:
        fails.append("the lambda does not name each expert's Q8_0 SOA layout (pp_q8_0_soa(expert))")
    q8_kernel = lam.find("dequantize_row_q8_0_soa_to_fp16_rowmajor(")
    if q8_kernel < 0 or (enclosing_if_condition(lam, q8_kernel) or "").strip() != "pp_q8_0_soa(expert)":
        fails.append("the lambda does not take Q8_0 SOA experts to dequantize_row_q8_0_soa_to_fp16_rowmajor")
    getter = lam.find("expert.to_fp16 = ggml_get_to_fp16_sycl_for_layout(src0->type, expert.layout);")
    if getter < 0 or "!pp_q8_0_soa(expert)" not in (enclosing_if_condition(lam, getter) or ""):
        fails.append("the lambda does not ask ggml_get_to_fp16_sycl_for_layout per expert, and never for Q8_0 SOA, "
                     "which it answers with the AOS kernel")
    woq_types = set(re.findall(r"GGML_TYPE_\w+", woq))
    if woq_types != {"GGML_TYPE_MXFP4"} or "GGML_LAYOUT_SOA" not in woq or "GGML_LAYOUT_XMX_TILED" not in woq:
        fails.append("the WOQ table must be MXFP4 on SOA and XMX_TILED")
    return fails


def claim_c_q4k_fallback(s):
    sycl = s["sycl"]
    fails = []
    fallback = statement(sycl, "const bool device_weight_cpu_fallback =")
    if fallback is None:
        return ["the device_weight_cpu_fallback definition was not found"]
    if fallback != "const bool device_weight_cpu_fallback = (src0->type == GGML_TYPE_Q4_K);":
        fails.append("device_weight_cpu_fallback is not exactly Q4_K: a claim term there only strips the CPU "
                     "fallback from hybrid ops the executor never runs")
    claims = statement(sycl, "const bool pp_onednn_batched_route_claims_prompt =") or ""
    if ("prompt_batch && (pp_onednn_batched_claims_staged_layer || "
            "ggml_sycl_moe_pp_onednn_batched_claims_tensor(src0, ctx.device))") not in claims:
        fails.append("pp_onednn_batched_route_claims_prompt is not the prompt-only claim")
    needs = statement(sycl, "const bool planner_needs_prompt_route =") or ""
    if "!pp_onednn_batched_route_claims_prompt" not in needs:
        fails.append("a claimed prompt does not drop the planner's prompt route (planner_needs_prompt_route)")
    selected = statement(sycl, "const bool selected_hybrid_route =") or ""
    if "planner_needs_prompt_route" not in selected:
        fails.append("selected_hybrid_route no longer reads planner_needs_prompt_route")
    hybrid = sycl.find("if (moe_hybrid_with_plan) {")
    at = sycl.find("const bool device_weight_cpu_fallback =")
    lam = sycl.find("auto try_pp_onednn_f16_batched = [&]() -> bool {")
    if not (0 <= hybrid < at < lam):
        fails.append("the fallback is not inside the hybrid world ahead of the batched executor's branch")
    return fails


def claim_d_decode_terms(s):
    fails = []
    sycl = s["sycl"]
    for text in (
        "const bool moe_hybrid_with_plan = ne12 == 1 || selected_hybrid_route;",
        "const bool pp_onednn_batched_route_claims_prompt = prompt_batch && "
        "(pp_onednn_batched_claims_staged_layer || ggml_sycl_moe_pp_onednn_batched_claims_tensor(src0, ctx.device));",
        "const bool prompt_batch = ids && ids->ne[1] > 1;",
        "const bool pp_onednn_batched_claims_op = ne12 > 1 && "
        "(pp_onednn_batched_claims_staged_layer || ggml_sycl_moe_pp_onednn_batched_claims_tensor(src0, ctx.device));",
    ):
        if text not in sycl:
            fails.append(f"missing decode-preserving term: {text}")
    lam = lambda_body(sycl) or ""
    if not re.search(r"if \([^{]*ids->ne\[1\] <= 1[^{]*\) \{ return trace_batched_skip\(", lam):
        fails.append("the lambda no longer refuses a single-token (decode) op with trace_batched_skip")
    return fails


def claim_e_refusal_enum(s):
    fails = []
    head = "enum class moe_pp_batched_refusal : uint8_t"
    defs = sum(squeezed.count(head) for _, squeezed in backend_texts())
    if defs != 1:
        return [f"`{head}` is defined {defs} times in the backend sources, want 1"]
    holder = next(k for k in ("tables", "sycl") if head in s[k])
    values = enum_values(s[holder], head) or []
    if not values or values[-1] != "COUNT":
        fails.append("the refusal enum does not end in COUNT, so nothing bounds it at compile time")
    causes = [v for v in values if v != "COUNT"]
    if len(causes) > 32:
        fails.append(f"the refusal enum has {len(causes)} causes; the per-device latch holds at most 32")
    names = one_definition(s[holder], "moe_pp_batched_refusal_name")
    if names is None:
        fails.append("moe_pp_batched_refusal_name is not defined exactly once beside the enum")
    else:
        for v in causes:
            if f"case moe_pp_batched_refusal::{v}:" not in names:
                fails.append(f"moe_pp_batched_refusal_name has no case for {v}")
    after = one_definition(s[holder], "moe_pp_batched_refusal_after_claim")
    if after is None:
        fails.append("moe_pp_batched_refusal_after_claim is not defined exactly once beside the enum")
    else:
        loud = []
        pending = []
        for m in re.finditer(r"case moe_pp_batched_refusal::(\w+) ?:|return (true|false) ?;|default ?:", after):
            if m.group(1):
                pending.append(m.group(1))
            elif m.group(2):
                if m.group(2) == "true":
                    loud.extend(pending)
                pending = []
            else:
                pending = []
        if sorted(loud) != sorted(POST_CLAIM):
            extra = sorted(set(loud) - set(POST_CLAIM))
            missing = sorted(set(POST_CLAIM) - set(loud))
            fails.append("moe_pp_batched_refusal_after_claim is loud for the wrong causes "
                         f"(precheck printed by default: {extra}; post-claim left trace-only: {missing})")
        if "default: return false;" not in after:
            fails.append("moe_pp_batched_refusal_after_claim does not default to false (a new cause would be loud)")
    lam = lambda_body(s["sycl"]) or ""
    calls = list(re.finditer(r"\b(trace_batched_skip|reject_batched)\(", lam))
    if not calls:
        fails.append("the lambda makes no trace_batched_skip/reject_batched calls")
    for m in calls:
        if not lam.startswith("moe_pp_batched_refusal::", m.end()):
            fails.append(f"{m.group(1)}({lam[m.end():m.end() + 40]}...) does not name a moe_pp_batched_refusal")
    literal = '"[PP-MOE] refused'
    if s["sycl"].count(literal) != 1:
        fails.append(f"{literal} appears {s['sycl'].count(literal)} times in ggml-sycl.cpp, want 1")
    else:
        pos = s["sycl"].find(literal)
        lam_pos = lam.find(literal)
        cond = enclosing_if_condition(lam, lam_pos) if lam_pos >= 0 else ""
        cond = cond or ""
        if cond.strip() != WARN_CONDITION:
            fails.append(f"the refusal WARN is guarded by `{cond.strip()}`, not `{WARN_CONDITION}`")
        stmt_start = s["sycl"].rfind("GGML_LOG_", 0, pos)
        if not s["sycl"].startswith("GGML_LOG_WARN(", stmt_start):
            fails.append("the refusal line is not emitted with GGML_LOG_WARN")
        stmt = statement(s["sycl"], None, start=stmt_start) or ""
        if "moe_pp_batched_refusal_name(" not in stmt:
            fails.append("the refusal WARN does not print the cause name")
    latch = one_definition(s["sycl"], "ggml_sycl_moe_pp_batched_refusal_first")
    if latch is None or "fetch_or(" not in latch or "std::atomic<uint32_t>" not in latch:
        fails.append("ggml_sycl_moe_pp_batched_refusal_first is not one per-device atomic bitmask latch")
    else:
        bound = "static_assert(static_cast<uint32_t>(moe_pp_batched_refusal::COUNT) <= 32,"
        if bound not in latch:
            fails.append("the refusal latch does not static_assert that moe_pp_batched_refusal::COUNT fits 32 bits")
        if "refusal >= moe_pp_batched_refusal::COUNT" not in latch:
            fails.append("the refusal latch does not refuse a value at or past COUNT before it shifts")
    old_prefix = '"[MOE-PP-ONEDNN-F16-BATCHED] refused'
    if old_prefix in s["sycl"]:
        fails.append(f"{old_prefix} is still printed: every refusal must read \"[PP-MOE] refused\"")
    if "warn_batched_refusal(refusal, detail);" not in lam:
        fails.append("decline_batched does not print its refusal through warn_batched_refusal")
    gemm = re.search(r"auto gemm_stage_failed = \[&\]\(const char \* what\) -> bool \{", lam)
    gemm_body = lam[gemm.end() - 1 : match_close(lam, gemm.end() - 1)] if gemm else ""
    if "warn_batched_refusal(moe_pp_batched_refusal::GEMM, what);" not in gemm_body:
        fails.append("the GEMM-stage failure does not print its refusal through warn_batched_refusal")
    return fails


def claim_f_first_run_line(s):
    fails = []
    sycl = s["sycl"]
    literal = '"[PP-MOE] device=%d type=%s layout=%s'
    if sycl.count(literal) != 1:
        return [f"{literal} appears {sycl.count(literal)} times in ggml-sycl.cpp, want 1"]
    pos = sycl.find(literal)
    lam = lambda_body(sycl) or ""
    lam_pos = lam.find(literal)
    cond = enclosing_if_condition(lam, lam_pos) if lam_pos >= 0 else ""
    cond = cond or ""
    if cond.strip() != "ggml_sycl_moe_pp_batched_first_run(ctx.device, src0->type, route_layout)":
        fails.append(f"the [PP-MOE] line is guarded by `{cond.strip()}`, not the per-(device, type, layout) latch")
    stmt_start = sycl.rfind("GGML_LOG_", 0, pos)
    if not sycl.startswith("GGML_LOG_INFO(", stmt_start):
        fails.append("the [PP-MOE] line is not emitted with GGML_LOG_INFO")
    stmt = statement(sycl, None, start=stmt_start) or ""
    for field in ("experts=", "planned_weight_slot="):
        if field not in stmt:
            fails.append(f"the [PP-MOE] line does not name {field}")
    latch = one_definition(sycl, "ggml_sycl_moe_pp_batched_first_run")
    if latch is None or "fetch_or(" not in latch or "std::atomic<uint32_t>" not in latch:
        fails.append("ggml_sycl_moe_pp_batched_first_run is not one atomic layout bitmask per (device, type)")
    elif "[GGML_SYCL_MAX_DEVICES][GGML_TYPE_COUNT]" not in latch:
        fails.append("ggml_sycl_moe_pp_batched_first_run does not keep one mask per (device, type)")
    elif "static_assert(GGML_LAYOUT_MXFP4_DPAS < 32," not in latch:
        fails.append("no static_assert that every layout fits the first-run latch's 32-bit mask")
    return fails


def claim_g_labels(s):
    fails = []
    counts = {
        ("sycl", '"mxfp4.pp.stage"'): 2,
        ("sycl", '"mxfp4.pp.stage.act"'): 1,
        ("sycl", '"mxfp4.pp.stage.scatter"'): 1,
        ("gemm", '"mxfp4.pp.gemm"'): 2,
        ("gemm", '"mxfp4.pp.gemm.execute"'): 2,
        ("convert", '"mxfp4.pp.repack"'): 8,
        ("convert", '"mxfp4.pp.repack.soa"'): 4,
        ("convert", '"mxfp4.pp.repack.tiled"'): 4,
    }
    for (key, lit), want in counts.items():
        got = s[key].count(lit)
        if got != want:
            fails.append(f"{lit} appears {got} times in {FILES[key].name}, want {want} (byte-identical MXFP4 labels)")
    sycl = s["sycl"]
    for text, want in (
        ("const bool pp_mxfp4_labels = src0->type == GGML_TYPE_MXFP4;", 1),
        ('pp_mxfp4_labels ? "mxfp4.pp.stage.act" : "moe.pp.stage.act"', 1),
        ('pp_mxfp4_labels ? "mxfp4.pp.stage.scatter" : "moe.pp.stage.scatter"', 1),
        ('pp_mxfp4_labels ? "mxfp4.pp.stage" : "moe.pp.stage"', 2),
        ('pp_mxfp4_labels ? nullptr : "moe.pp.gemm.execute"', 1),
        ('"moe.pp.dequant."', 1),
        ("const bool pp_profile = pp_mxfp4_labels && mxfp4_pp_batched_profile_enabled();", 1),
    ):
        got = sycl.count(text)
        if got != want:
            fails.append(f"`{text}` appears {got} times in ggml-sycl.cpp, want {want}")
    gemm = s["gemm"]
    if gemm.count('label_name ? label_name : "mulmat.onednn_gemm.execute"') != 2:
        fails.append("gemm_batch_strided's two label sites do not default to mulmat.onednn_gemm.execute")
    return fails


def claim_h_capability(s):
    fails = []
    plan = s["layer_plan"]
    if "ONEDNN_BATCHED_PROMPT = 8," not in plan and "ONEDNN_BATCHED_PROMPT = 8 }" not in plan:
        fails.append("moe_route_kernel has no ONEDNN_BATCHED_PROMPT = 8")
    if 'case moe_route_kernel::ONEDNN_BATCHED_PROMPT: return "onednn-batched-prompt";' not in plan:
        fails.append("moe_route_kernel_name has no onednn-batched-prompt case")
    cap = one_definition(s["sycl"], "ggml_sycl_moe_query_route_capability")
    if cap is None:
        return fails + ["ggml_sycl_moe_query_route_capability was not found"]
    mmvq = cap.find("cap.kernel = moe_route_kernel::MMVQ_COMPAT;")
    onednn = cap.find("cap.kernel = moe_route_kernel::ONEDNN_BATCHED_PROMPT;")
    if onednn < 0:
        fails.append("the capability query never answers ONEDNN_BATCHED_PROMPT")
    else:
        if onednn < mmvq:
            fails.append("the ONEDNN_BATCHED_PROMPT branch precedes the MMVQ check, so MMVQ types change kernel")
        cond = enclosing_if_condition(cap, onednn) or ""
        if "phase == moe_route_phase::PROMPT" not in cond or "ggml_sycl_moe_pp_batched_takes_layout(" not in cond:
            fails.append("the ONEDNN_BATCHED_PROMPT branch is not prompt-only and table-derived")
    gate = ("if (!moe_mmvq_admission_supports_type(indexed_a_type) && "
            "!ggml_sycl_moe_pp_batched_admits_op(op, device)) {")
    if gate not in s["sycl"]:
        fails.append("MUL_MAT_ID admission does not fall back to ggml_sycl_moe_pp_batched_admits_op")
    admits = one_definition(s["sycl"], "ggml_sycl_moe_pp_batched_admits_op")
    if admits is None:
        fails.append("ggml_sycl_moe_pp_batched_admits_op is not defined exactly once")
    else:
        if "->ne[1] > 1" not in admits:
            fails.append("ggml_sycl_moe_pp_batched_admits_op does not require a prompt (ids ne[1] > 1)")
        if "ggml_sycl_moe_pp_onednn_batched_claims_tensor(" not in admits:
            fails.append("ggml_sycl_moe_pp_batched_admits_op does not require the executor's claim")
    # The claim reads its planned layout through the plan-taking lookup, which answers only for a layer whose every
    # expert is planned on this device: a hybrid layer (a host or other-device expert) is never claimed, so never
    # admitted for a type without an MMVQ-ID kernel.
    claim = claim_predicate_body(s["sycl"]) or ""
    if claim.count("ggml_sycl_moe_tensor_plan_primary_layout(") != 2:
        fails.append("the claim does not read its planned layout through ggml_sycl_moe_tensor_plan_primary_layout")
    lookup = next((b for b in definitions(s["sycl"], "ggml_sycl_moe_tensor_plan_primary_layout")
                   if "find_expert_entry(" in b), None)
    if lookup is None:
        fails.append("the plan-taking ggml_sycl_moe_tensor_plan_primary_layout was not found")
    else:
        if "if (!plan.all_experts_on_device(src0->name, n_experts, device)) { return false; }" not in lookup:
            fails.append("the planned-layout lookup answers for a layer not wholly planned on this device")
        per_expert = ("if (!placement || !placement->on_device || placement->target_device != device) "
                      "{ return false; }")
        if per_expert not in lookup:
            fails.append("the planned-layout lookup answers although an expert is on the host or another device")
    return fails


def claim_i_slot(s):
    fails = []
    sycl = s["sycl"]
    for gone in ("ggml_sycl_moe_pp_batched_f16_weight_slot_bytes", "ggml_sycl_pp_moe_widened_first",
                 '"[SYCL-PLAN] PP MoE oneDNN batched weight slot widened',
                 "g_tensor_inventory_pp_moe_onednn_scratch_bytes"):
        if gone in sycl:
            fails.append(f"{gone} is still in ggml-sycl.cpp: the planned weight slot must not be widened")
    # Every write to the planned weight slot's global is a reset or the inventory's own value.
    writes = re.finditer(r"\bg_tensor_inventory_pp_moe_onednn_weight_slot_bytes ?([-+*/|&^]?=)(?!=) ?([^;]*);", sycl)
    for m in writes:
        if m.group(1) != "=" or m.group(2).strip() not in ("0", "inventory->pp_moe_onednn_weight_slot_bytes"):
            fails.append(f"the planned weight slot is written as `{m.group(0)}`, not reset or taken from the inventory")
    # The planner site publishes the inventory's slot, with no formula of its own between the two.
    head = sycl.find(PLAN_SITE_HEAD)
    call = sycl.find("unified_cache_set_planned_pp_moe_onednn_scratch(", head) if head >= 0 else -1
    site = statement(sycl, None, start=call) if call >= 0 else None
    if head < 0 or site is None:
        fails.append("the planner site that takes the weight slot from the inventory and plans the ring was not found")
    else:
        site = sycl[head:call] + site
        if PLAN_SITE_CALL not in site:
            fails.append("the unified cache's planned scratch is not given the inventory's weight slot")
        for formula in ("n_expert", "255", "sizeof(sycl::half)"):
            if formula in site:
                fails.append(f"the planner site sizes the ring with `{formula}`: the weight slot is widened again")
    lam = lambda_body(sycl) or ""
    one = "checked_mul(weight_elems, sizeof(sycl::half), f16_expert_bytes)"
    if one not in lam or "weight_bytes = f16_expert_bytes;" not in lam:
        fails.append("the f16 arm does not ask the admission for one f16 expert")
    if re.search(r"checked_mul\(weight_elems, active_count\b", lam):
        fails.append("the f16 arm still asks the admission for every active expert at once")
    plan = "moe_pp_batched_f16_plan_groups(reserved.weight_size, f16_expert_bytes, active_count, "
    at = lam.find(plan)
    if at < 0:
        fails.append("the f16 arm does not group its experts by what the slot it claimed holds")
    elif at < lam.find("batched_scratch_claim.activate("):
        fails.append("the f16 arm plans its groups before it has claimed a slot")
    elif "ggml_sycl_moe_pp_f16_group_experts_max()" not in (statement(lam, None, start=at) or ""):
        fails.append(f"the group plan does not take the {GROUP_KNOB} cap")
    return fails


def claim_j_knob(s):
    fails = []
    literal = f'"{KNOB}"'
    total = sum(stripped.count(literal) for stripped, _ in backend_texts())
    if total != 1:
        fails.append(f"{literal} appears {total} times in the backend sources, want 1")
    spelling = f"static constexpr const char * {KNOB_CONST} = {literal};"
    if spelling not in s["common"]:
        fails.append(f"common.hpp does not spell the knob once as {KNOB_CONST}")
    body = one_definition(s["common"], "ggml_sycl_moe_pp_f16_batched_aos_enabled")
    if body is None or f"std::getenv({KNOB_CONST})" not in body or "static const bool" not in body:
        fails.append("ggml_sycl_moe_pp_f16_batched_aos_enabled does not read the knob once into a static")
    elif "env == nullptr || std::atoi(env) != 0" not in body:
        fails.append("the knob is not default ON (unset or non-zero)")
    row = f"{{ {KNOB_CONST}, &g_ggml_sycl_moe_pp_f16_batched_aos, 1 }},"
    if row not in s["sycl"]:
        fails.append("sycl_env_settings[] has no row for the knob, so the non-default report never names it")
    report = "g_ggml_sycl_moe_pp_f16_batched_aos = ggml_sycl_moe_pp_f16_batched_aos_enabled() ? 1 : 0;"
    if report not in s["sycl"]:
        fails.append("the report's integer is not set from the accessor, so it can disagree with what is in effect")
    if f"| `{KNOB}=0` |" not in s["catalog"]:
        fails.append(f"docs/backend/sycl-env-vars.md has no `{KNOB}=0` row")
    takes = one_definition(s["sycl"], "ggml_sycl_moe_pp_batched_takes_layout") or ""
    if "ggml_sycl_moe_pp_f16_batched_aos_enabled()" not in takes:
        fails.append("ggml_sycl_moe_pp_batched_takes_layout does not consult the knob")
    support = one_definition(s["tables"], "moe_pp_batched_supports_layout") or ""
    if "if (type != GGML_TYPE_MXFP4 && !aos_enabled) { return false; }" not in support:
        fails.append("the knob does not drop every non-MXFP4 entry (an SOA entry would survive it)")
    return fails


def claim_k_expert_entries(s):
    fails = []
    entries = one_definition(s["tables"], "moe_pp_batched_has_expert_entries")
    if entries is None:
        return ["moe_pp_batched_has_expert_entries is not defined exactly once in moe-mmvq-tables.hpp"]
    if "return type == GGML_TYPE_MXFP4 || !device_buffer || layout == GGML_LAYOUT_AOS;" not in entries:
        fails.append("moe_pp_batched_has_expert_entries does not decline exactly a non-AOS layout in a device buffer")
    claim = claim_predicate_body(s["sycl"])
    if claim is None:
        return fails + ["the plan-taking ggml_sycl_moe_pp_onednn_batched_claims_tensor definition was not found"]
    storage = "const bool device_buffer = weight->buffer != nullptr && !ggml_backend_buffer_is_host(weight->buffer);"
    if storage not in claim:
        fails.append("the claim predicate does not read device-buffer storage from the weight's own buffer")
    ask = "if (!moe_pp_batched_has_expert_entries(weight->type, planned_layout, device_buffer)) { return false; }"
    at = claim.find(ask)
    if at < 0:
        fails.append("the claim predicate does not decline a planned layout the cache holds no per-expert entries at")
    elif at < claim.find("if (!planned) { return false; }"):
        fails.append("the claim predicate asks for per-expert entries before it has read the planned layout")
    return fails


def claim_l_f16_knob(s):
    fails = []
    literal = f'"{F16_KNOB}"'
    total = sum(stripped.count(literal) for stripped, _ in backend_texts())
    if total != 1:
        fails.append(f"{literal} appears {total} times in the backend sources, want 1 (one reader)")
    body = one_definition(s["common"], "ggml_sycl_moe_pp_onednn_f16_enabled")
    if body is None or literal not in body or "static const bool" not in body:
        fails.append("ggml_sycl_moe_pp_onednn_f16_enabled does not read the knob once into a static")
    elif "env == nullptr || std::atoi(env) != 0" not in body:
        fails.append(f"{F16_KNOB} is not default ON (unset or non-zero)")
    takes = one_definition(s["sycl"], "ggml_sycl_moe_pp_batched_takes_layout") or ""
    if "if (!ggml_sycl_moe_pp_onednn_f16_enabled()) { return false; }" not in takes:
        fails.append("the support helper does not ask the knob, so the claim takes ops the executor declines")
    lam = lambda_body(s["sycl"]) or ""
    route = ("const bool pp_f16_batched_env_enabled = ggml_sycl_moe_pp_onednn_batched_route_selected() && "
             "ggml_sycl_moe_pp_onednn_f16_enabled();")
    if route not in lam:
        fails.append("the executor's route-off refusal does not cover the knob, so it is traced as a candidate miss")
    if f"| `{F16_KNOB}=0` |" not in s["catalog"]:
        fails.append(f"docs/backend/sycl-env-vars.md has no `{F16_KNOB}=0` row")
    return fails


def claim_m_chain(s):
    fails = []
    lam = lambda_body(s["sycl"]) or ""
    start = lam.find(PHASE_LOOP_HEAD)
    # The first barrier on the GEMM events after the loop head: the failure path, defined before the loop, ends
    # in the same barrier.
    end = lam.find(PHASE_LOOP_END, start) if start >= 0 else -1
    if start < 0 or end < start:
        return ["the lambda has no phase loop ending in the barrier on every GEMM event"]
    loop = lam[start:end]
    if PHASE_OFFSET not in loop:
        fails.append("a phase does not take its region of the f16 slot from moe_pp_batched_phase_weight_offset")
    elif PHASE_BOUND not in loop or loop.find(PHASE_BOUND) < loop.find(PHASE_OFFSET):
        fails.append("a phase's region is not asserted to lie in the claimed slot after it is chosen")
    if "phase.buffer * half_elems" in loop or "f16_groups.groups" in loop or "f16_groups.buffers" in loop:
        fails.append("the phase loop re-derives the buffer reuse the header's predicate decides")
    wait = PHASE_WAIT
    dequant = "dequant_expert_f16(active_experts[i], phase_weights + (i - phase.begin) * weight_elems);"
    at_wait, at_dequant = loop.find(wait), loop.find(dequant)
    if at_wait < 0:
        fails.append("a group does not wait, on the device, for the GEMMs that last read its half")
    if at_dequant < 0:
        fails.append("a group does not dequantize its experts into its own half")
    elif at_wait > at_dequant:
        fails.append("a group dequantizes into its half before it waits for that half's readers")
    if "phase_weights + (lo - phase.begin) * weight_elems" not in loop:
        fails.append("a phase's GEMM does not read the half its experts were dequantized into")
    if "half_readers[phase.buffer].push_back(*group_event);" not in loop:
        fails.append("a GEMM is not recorded as a reader of its half, so the next group to use it does not wait")
    compare = lam.find(COMPARE_HEAD)
    compare_end = match_close(lam, compare + len(COMPARE_HEAD) - 1) if compare >= 0 else -1
    for wait in ANY_WAIT_RE.finditer(lam):
        if not compare < wait.start() < compare_end:
            fails.append(f"the lambda waits on the host outside the compare arm: `{lam[wait.start() - 30:wait.end()]}`")
    helper = one_definition(s["tables"], "moe_pp_batched_f16_plan_groups")
    if helper is None:
        fails.append("moe_pp_batched_f16_plan_groups is not defined exactly once in moe-mmvq-tables.hpp")
    elif "plan.buffers = fit >= 2 ? 2 : 1;" not in helper:
        fails.append("the group plan does not split a slot that holds two experts into two halves")
    if WEIGHT_USED not in lam:
        fails.append("the route log's weight figure does not ask the reuse predicate for the slot the phases used")
    for name, want in ((REUSE_FN, REUSE_BODY), (OFFSET_FN, OFFSET_BODY)):
        body = one_definition(s["tables"], name)
        if body is None or definitions(s["sycl"], name):
            fails.append(f"{name} is not defined exactly once, in moe-mmvq-tables.hpp")
        elif want not in body:
            fails.append(f"{name} does not answer `{want}`")
    return fails


def claim_n_dequant_label(s):
    fails = []
    sycl = s["sycl"]
    lam = lambda_body(sycl) or ""
    helper = one_definition(sycl, "ggml_sycl_moe_pp_dequant_label")
    # The helper builds each name once; anywhere else, building one is a per-dispatch allocation.
    if 'std::string("moe.pp.dequant.")' in sycl.replace(helper or "", ""):
        fails.append("a dispatch still builds the moe.pp.dequant.<type> label as a std::string")
    if "ggml_sycl_moe_pp_dequant_label(src0->type)" not in lam:
        fails.append("the lambda does not take the dequant label from ggml_sycl_moe_pp_dequant_label")
    if helper is None:
        fails.append("ggml_sycl_moe_pp_dequant_label is not defined exactly once")
    elif not re.search(r"static const std::array<std::string, GGML_TYPE_COUNT> \w+ = \[\]\(\) \{", helper):
        fails.append("ggml_sycl_moe_pp_dequant_label does not build its names once, in a function-local static table")
    return fails


def claim_o_device_coverage(s):
    fails = []
    f16 = one_definition(s["tables"], "moe_pp_batched_f16_supports_layout")
    if f16 is None:
        return ["the f16 table was not found"]
    table = {t for t in switch_cases(f16) if t not in ("default", "GGML_TYPE_MXFP4")}
    m = re.search(r"constexpr ggml_type k_types\[\] = \{([^}]*)\}", s["device_test"])
    if not m:
        return ["test-moe-pp-batched-dequant.cpp has no `constexpr ggml_type k_types[] = { ... }`"]
    tested = set(re.findall(r"GGML_TYPE_\w+", m.group(1)))
    untested = set(UNTESTED_ON_DEVICE)
    for t in sorted(tested & untested):
        fails.append(f"{t} is in the device test and also named untested on device")
    for t in sorted(table - tested - untested):
        fails.append(f"the f16 table takes {t}, which the device test does not run and nothing names untested")
    for t in sorted((tested | untested) - table):
        fails.append(f"{t} is device-tested or named untested, but the f16 table does not take it")
    # The fixture quantizes with no importance matrix, so the untested types are exactly the table's types that
    # ggml_quantize_requires_imatrix names (comments stripped, so a commented-out type is not one).
    needs = one_definition(s["ggml_c"], "ggml_quantize_requires_imatrix")
    if needs is None:
        fails.append("ggml_quantize_requires_imatrix is not defined exactly once in ggml.c")
    else:
        imatrix = set(re.findall(r"GGML_TYPE_\w+", needs)) & table
        if untested != imatrix:
            fails.append(f"UNTESTED_ON_DEVICE is {sorted(untested)}, but the table types that need an imatrix "
                         f"are {sorted(imatrix)}")
    return fails


def claim_p_device_counts(s):
    fails = []
    test = s["device_test"]
    for needle, what in (
        ('g_log.count_since(from, { "[PP-MOE] device=", type_field.c_str() })', "the case's own [PP-MOE] lines"),
        ('g_log.count_since(0, { "[PP-MOE] device=", type_field.c_str() })', "the run's [PP-MOE] total"),
        ('g_log.count_since(from, { "[MOE-PP-ONEDNN-F16-BATCHED] tensor=" })', "the case's own success lines"),
    ):
        if needle not in test:
            fails.append(f"the device test does not count {what} (`{needle}`)")
    if "pp_moe != (first_of_type ? 1u : 0u) || pp_total != 1" not in test:
        fails.append("the device test does not require the [PP-MOE] line in the type's first case only, once per run")
    backend = re.search(r"static std::atomic<int> batched_log\{ 0 \}; "
                        r"if \(batched_log\.fetch_add\(1, std::memory_order_relaxed\) < (\d+)\)", s["sycl"])
    test_cap = re.search(r"constexpr int k_route_log_cap = (\d+);", test)
    if not backend:
        fails.append("the backend's batched success-line cap was not found")
    elif not test_cap or test_cap.group(1) != backend.group(1):
        got = test_cap.group(1) if test_cap else None
        fails.append(f"the device test's k_route_log_cap is {got}, the backend's cap is {backend.group(1)}")
    if "static_assert(k_cases <= k_route_log_cap," not in test:
        fails.append("the device test does not static_assert that its cases fit the success line's cap")
    return fails


def claim_q_group_knob(s):
    fails = []
    literal = f'"{GROUP_KNOB}"'
    total = sum(stripped.count(literal) for stripped, _ in backend_texts())
    if total != 1:
        fails.append(f"{literal} appears {total} times in the backend sources, want 1")
    if f"static constexpr const char * {GROUP_KNOB_CONST} = {literal};" not in s["common"]:
        fails.append(f"common.hpp does not spell the knob once as {GROUP_KNOB_CONST}")
    body = one_definition(s["common"], "ggml_sycl_moe_pp_f16_group_experts_max")
    if body is None or f"std::getenv({GROUP_KNOB_CONST})" not in body or "static const size_t" not in body:
        fails.append("ggml_sycl_moe_pp_f16_group_experts_max does not read the knob once into a static")
    elif "value < 1 || value > 1024" not in body:
        fails.append("the knob does not reject a value outside 1 to 1024")
    row = f"{{ {GROUP_KNOB_CONST}, &g_ggml_sycl_moe_pp_f16_group_experts_max, 0 }},"
    if row not in s["sycl"]:
        fails.append("sycl_env_settings[] has no row for the group knob, so the non-default report never names it")
    report = ("g_ggml_sycl_moe_pp_f16_group_experts_max = "
              "static_cast<int>(ggml_sycl_moe_pp_f16_group_experts_max());")
    if report not in s["sycl"]:
        fails.append("the group knob's report integer is not set from the accessor")
    if f"| `{GROUP_KNOB}=<1..1024>` |" not in s["catalog"]:
        fails.append(f"docs/backend/sycl-env-vars.md has no `{GROUP_KNOB}=<1..1024>` row")
    success = re.search(r'"\[MOE-PP-ONEDNN-F16-BATCHED\] tensor=[^;]*;', s["sycl"])
    if not success or "groups=%zu" not in success.group(0) or "phases.size()" not in success.group(0):
        fails.append("the batched success line does not name the number of groups (phases.size())")
    test = s["device_test"]
    if "constexpr size_t k_min_groups = 3;" not in test or "groups < k_min_groups" not in test:
        fails.append("the device test does not require at least three groups at the slot's own grouping")
    if GROUP_CAPS not in test:
        fails.append(f"the device test does not run the capped cases `{GROUP_CAPS}`")
    if "for (int cap : k_group_caps) { if (!run_capped_child(cap)) { ++failed; } }" not in test:
        fails.append("the device test does not run a child process for every cap and count each failed one")
    if "if (waited && WIFEXITED(status) && WEXITSTATUS(status) == 0) { return true; }" not in test:
        fails.append("a capped case passes on something other than its child exiting 0")
    if "make_case(GGML_TYPE_Q4_0, 300, k_used, rng), true, group_cap) ? 0 : 1;" not in test:
        fails.append("a capped case's child does not exit 1 when its case fails")
    if 'execl("/proc/self/exe", "test-moe-pp-batched-dequant", "--group-cap",' not in test:
        fails.append("a capped case does not run in a fresh process of the test itself")
    sets = test.count(f'setenv("{GROUP_KNOB}",')
    if sets != 1 or f'setenv("{GROUP_KNOB}", argv[2], 1);' not in test:
        fails.append(f"the device test sets {GROUP_KNOB} {sets} times, want once, from the child's --group-cap")
    want = ("const size_t want_groups = group_cap > 0 ? "
            "(size_t(k_experts) + size_t(group_cap) - 1) / size_t(group_cap) : 0;")
    if want not in test or GROUP_EXACT not in test:
        fails.append("a capped case does not require exactly the groups, experts per group and buffers its cap gives")
    if 'g_log.min_field_since(from, "[MOE-PP-ONEDNN-F16-BATCHED] tensor=", "groups=")' not in test:
        fails.append("the device test does not read the group count from the case's own success lines")
    return fails


def claim_r_failure_drain(s):
    fails = []
    lam = lambda_body(s["sycl"]) or ""
    at = lam.find(FAIL_HEAD)
    if at < 0:
        return ["the lambda has no gemm_stage_failed helper"]
    fail_body = lam[at:match_close(lam, at + len(FAIL_HEAD) - 1)]
    # The drain is one helper (llama.cpp-8epx), which the failure path calls once, before its XMX_TILED throw.
    drain_at = lam.find(DRAIN_HEAD)
    if drain_at < 0 or drain_at > at:
        return ["the lambda has no drain_submitted_phases helper defined ahead of gemm_stage_failed"]
    if fail_body.count("drain_submitted_phases();") != 1 or DRAIN_CALL not in fail_body:
        fails.append("a failed GEMM stage does not drain exactly once, before its XMX_TILED throw")
    if "ext_oneapi_submit_barrier" in fail_body:
        fails.append("a failed GEMM stage keeps a barrier of its own beside the shared drain")
    body = lam[drain_at:match_close(lam, drain_at + len(DRAIN_HEAD) - 1)]
    barrier, retain, release = body.find(FAIL_BARRIER), body.find(FAIL_RETAIN), body.find(CLAIM_RELEASE)
    if barrier < 0:
        fails.append("a failed GEMM stage does not barrier on the GEMMs it already submitted")
    if retain < 0:
        fails.append("a failed GEMM stage does not hold the experts' leases until that barrier")
    if body.count(CLAIM_RELEASE) != 1:
        fails.append("a failed GEMM stage does not finish or release its scratch claim exactly once")
    elif barrier >= 0 and release < barrier:
        fails.append("a failed GEMM stage releases its scratch claim before the barrier on its GEMMs")
    elif retain >= 0 and release < retain:
        fails.append("a failed GEMM stage releases its scratch claim before it holds the experts' leases")
    helper_at = lam.find(LEASE_HELPER)
    helper = lam[helper_at:match_close(lam, helper_at + len(LEASE_HELPER) - 1)] if helper_at >= 0 else ""
    for needle in ("for (pp_onednn_batched_expert & expert : active_experts) {",
                   "leases.push_back(std::move(expert.lease));",
                   "ggml_sycl::retain_handles_until_event(std::move(leases), until);"):
        if needle not in helper:
            fails.append(f"retain_expert_leases_until does not hold every expert's lease until its event (`{needle}`)")
    if helper_at > drain_at:
        fails.append("retain_expert_leases_until is defined after the drain that calls it")
    if lam.count("retain_expert_leases_until(done);") != 2:
        fails.append("the two success branches do not hold the leases through retain_expert_leases_until(done)")
    if lam.count("retain_handles_until_event(") != 1:
        fails.append("the lambda holds leases outside retain_expert_leases_until")
    return fails


def claim_s_staged_type(s):
    fails = []
    helper = one_definition(s["tables"], STAGED_TYPE)
    if helper is None:
        return [f"{STAGED_TYPE} is not defined exactly once in moe-mmvq-tables.hpp"]
    if helper != STAGED_TYPE_BODY:
        fails.append(f"{STAGED_TYPE} is not exactly an MMVQ-ID kernel and an f16 row at AOS")
    claim = claim_predicate_body(s["sycl"])
    if claim is None:
        return fails + ["the plan-taking ggml_sycl_moe_pp_onednn_batched_claims_tensor definition was not found"]
    staged = statement(claim, STAGED_CLAIM_HEAD)
    if staged is None:
        fails.append(f"the staged claim does not ask {STAGED_TYPE} about the weight's type")
    elif STAGED_CLAIM_KNOBS not in staged:
        fails.append("the staged claim does not ask ggml_sycl_moe_pp_batched_takes_layout at AOS (the runtime knobs)")
    for term in TABLE_TERMS:
        if term in claim:
            fails.append(f"the claim predicate asks {term}...) itself, a second source for the type question")
    seed = one_definition(s["cost"], "moe_exec_dev_gflops_seed")
    if seed is None:
        return fails + ["moe_exec_dev_gflops_seed is not defined exactly once in moe-exec-cost.hpp"]
    gate = seed.find(SEED_GATE)
    if gate < 0:
        fails.append(f"moe_exec_dev_gflops_seed does not seed the per-row rate for a type {STAGED_TYPE} refuses")
    else:
        head = seed[:gate]
        if head.count("return ") != 1 or SEED_MXFP4 not in head:
            fails.append(f"moe_exec_dev_gflops_seed seeds a type other than MXFP4 before it asks {STAGED_TYPE}")
    for term in TABLE_TERMS:
        if term in seed:
            fails.append(f"moe_exec_dev_gflops_seed asks {term}...) itself, a second source for the type question")
    return fails


CLAIMS = {
    "a": claim_a_one_table,
    "b": claim_b_table_types,
    "c": claim_c_q4k_fallback,
    "d": claim_d_decode_terms,
    "e": claim_e_refusal_enum,
    "f": claim_f_first_run_line,
    "g": claim_g_labels,
    "h": claim_h_capability,
    "i": claim_i_slot,
    "j": claim_j_knob,
    "k": claim_k_expert_entries,
    "l": claim_l_f16_knob,
    "m": claim_m_chain,
    "n": claim_n_dequant_label,
    "o": claim_o_device_coverage,
    "p": claim_p_device_counts,
    "q": claim_q_group_knob,
    "r": claim_r_failure_drain,
    "s": claim_s_staged_type,
}


@pytest.fixture(scope="module")
def sources():
    return load()


@pytest.mark.parametrize("claim", sorted(CLAIMS))
def test_claim_holds(sources, claim):
    fails = CLAIMS[claim](sources)
    assert not fails, f"claim ({claim}) failed:\n  " + "\n  ".join(fails)


# (claim, file, old, new). `old` is matched in the squeezed, comment-stripped text and must occur in it.
MUTANTS = [
    ("a", "sycl", "ggml_sycl_moe_pp_batched_takes_layout(weight->type, planned_layout)",
     "weight->type == GGML_TYPE_MXFP4 && ggml_sycl_moe_pp_batched_takes_layout(weight->type, planned_layout)"),
    ("a", "sycl", "const bool pp_batched_candidate =",
     "const bool pp_batched_candidate = src0->type == GGML_TYPE_MXFP4 &&"),
    ("a", "sycl", "!pp_batched_candidate", "!pp_mxfp4_soa_f16_candidate"),
    ("b", "tables", "case GGML_TYPE_IQ4_XS:", ""),
    ("b", "tables", "case GGML_TYPE_MXFP4: return layout == GGML_LAYOUT_SOA;",
     "case GGML_TYPE_MXFP4: return layout == GGML_LAYOUT_AOS;"),
    ("b", "tables", "case GGML_TYPE_IQ4_XS:", "case GGML_TYPE_IQ4_XS: case GGML_TYPE_Q1_0:"),
    ("b", "tables", " || layout == GGML_LAYOUT_SOA;", ";"),
    ("b", "convert", "return dequantize_row_q6_K_sycl_reorder;", "return nullptr;"),
    ("b", "sycl", "if (pp_q8_0_soa(expert)) {", "if (false) {"),
    ("b", "sycl", "} else if (!pp_q8_0_soa(expert)) {", "} else {"),
    # One converter for every expert, from the first one's layout (the shape before llama.cpp-3460).
    ("b", "sycl", "ggml_get_to_fp16_sycl_for_layout(src0->type, expert.layout);",
     "ggml_get_to_fp16_sycl_for_layout(src0->type, active_experts.front().layout);"),
    ("c", "sycl", "const bool device_weight_cpu_fallback = (src0->type == GGML_TYPE_Q4_K);",
     "const bool device_weight_cpu_fallback = (src0->type == GGML_TYPE_Q4_K) && "
     "!pp_onednn_batched_route_claims_prompt;"),
    ("c", "sycl", "!exact_layout_override && !pp_onednn_batched_route_claims_prompt;", "!exact_layout_override;"),
    ("d", "sycl", "const bool moe_hybrid_with_plan = ne12 == 1 || selected_hybrid_route;",
     "const bool moe_hybrid_with_plan = selected_hybrid_route;"),
    ("d", "sycl", "const bool pp_onednn_batched_claims_op = ne12 > 1 && ",
     "const bool pp_onednn_batched_claims_op = "),
    ("d", "sycl", "if (ids->ne[1] <= 1) { return trace_batched_skip(",
     "if (ids->ne[1] < 1) { return trace_batched_skip("),
    ("e", "sycl", "return trace_batched_skip(moe_pp_batched_refusal::NO_ACTIVE_EXPERTS",
     'return trace_batched_skip("no-active-experts"); '
     "return trace_batched_skip(moe_pp_batched_refusal::NO_ACTIVE_EXPERTS"),
    ("e", "sycl", "ggml_sycl_moe_pp_batched_refusal_first(ctx.device, refusal)", "true"),
    ("e", "tables", "case moe_pp_batched_refusal::GEMM:", ""),
    ("e", "tables", "GEMM, COUNT", "GEMM"),
    ("e", "sycl", "static_assert(static_cast<uint32_t>(moe_pp_batched_refusal::COUNT) <= 32,",
     "static_assert(true,"),
    ("e", "sycl", "refusal >= moe_pp_batched_refusal::COUNT", "false"),
    # A post-claim cause left trace-only, and a precheck cause printed without the trace knob.
    ("e", "tables", "case moe_pp_batched_refusal::EXPERT_NOT_LOCAL: case moe_pp_batched_refusal::EXPERT_LAYOUT:",
     "case moe_pp_batched_refusal::EXPERT_LAYOUT:"),
    ("e", "tables", "case moe_pp_batched_refusal::GEMM: return true;",
     "case moe_pp_batched_refusal::GEMM: case moe_pp_batched_refusal::DECODE: return true;"),
    ("e", "sycl", "(pp_f16_batched_trace || moe_pp_batched_refusal_after_claim(refusal) || pp_staged_experts > 0 || "
     "pp_onednn_batched_claims_staged_layer) &&", "pp_f16_batched_trace &&"),
    ("e", "sycl", "(pp_f16_batched_trace || moe_pp_batched_refusal_after_claim(refusal) || pp_staged_experts > 0 || "
     "pp_onednn_batched_claims_staged_layer) &&",
     "(true || moe_pp_batched_refusal_after_claim(refusal) || pp_staged_experts > 0 || "
     "pp_onednn_batched_claims_staged_layer) &&"),
    # A decline after a staged slot left quiet when its cause is a precheck (llama.cpp-3460).
    ("e", "sycl", "|| pp_staged_experts > 0 || pp_onednn_batched_claims_staged_layer) &&",
     "|| pp_onednn_batched_claims_staged_layer) &&"),
    # A precheck decline on a weight the ring staged in full left quiet (llama.cpp-3460).
    ("e", "sycl", "|| pp_staged_experts > 0 || pp_onednn_batched_claims_staged_layer) &&",
     "|| pp_staged_experts > 0) &&"),
    ("f", "sycl", "if (ggml_sycl_moe_pp_batched_first_run(ctx.device, src0->type, route_layout))", "if (true)"),
    ("f", "sycl", "ggml_sycl_moe_pp_batched_first_run(ctx.device, src0->type, route_layout)",
     "ggml_sycl_moe_pp_batched_first_run(ctx.device, src0->type, GGML_LAYOUT_AOS)"),
    ("f", "sycl", "planned_weight_slot=", "planned_weight="),
    ("g", "sycl", 'pp_mxfp4_labels ? "mxfp4.pp.stage.act" : "moe.pp.stage.act"', '"moe.pp.stage.act"'),
    ("g", "sycl", "const bool pp_profile = pp_mxfp4_labels && mxfp4_pp_batched_profile_enabled();",
     "const bool pp_profile = mxfp4_pp_batched_profile_enabled();"),
    ("g", "gemm", 'label_name ? label_name : "mulmat.onednn_gemm.execute"', '"mulmat.onednn_gemm.execute"'),
    ("h", "sycl", "phase == moe_route_phase::PROMPT && ggml_sycl_moe_pp_batched_takes_layout(",
     "ggml_sycl_moe_pp_batched_takes_layout("),
    ("h", "sycl", "->ne[1] > 1 && ggml_sycl_moe_pp_onednn_batched_claims_tensor(",
     "ggml_sycl_moe_pp_onednn_batched_claims_tensor("),
    ("h", "sycl", " && !ggml_sycl_moe_pp_batched_admits_op(op, device)) {", ") {"),
    # The widening back, the admission asked for every expert, and the groups sized from the request, not the slot.
    ("i", "sycl", "g_tensor_inventory_pp_moe_onednn_weight_slot_bytes = inventory->pp_moe_onednn_weight_slot_bytes;",
     "g_tensor_inventory_pp_moe_onednn_weight_slot_bytes = std::max(inventory->pp_moe_onednn_weight_slot_bytes, "
     "size_t(inventory->n_expert) << 20);"),
    ("i", "sycl", "weight_bytes = f16_expert_bytes;", "weight_bytes = f16_expert_bytes * active_count;"),
    ("i", "sycl", "moe_pp_batched_f16_plan_groups(reserved.weight_size, ",
     "moe_pp_batched_f16_plan_groups(weight_bytes * active_count, "),
    ("j", "sycl", "ggml_sycl_moe_pp_f16_batched_aos_enabled()", "true"),
    ("j", "tables", "if (type != GGML_TYPE_MXFP4 && !aos_enabled)", "if (layout == GGML_LAYOUT_AOS && !aos_enabled)"),
    ("j", "common", "return env == nullptr || std::atoi(env) != 0; }(); return aos_enabled;",
     "return env != nullptr && std::atoi(env) != 0; }(); return aos_enabled;"),
    ("k", "tables", "|| !device_buffer || layout == GGML_LAYOUT_AOS;", "|| layout == GGML_LAYOUT_AOS;"),
    ("k", "sycl", "weight->buffer != nullptr && !ggml_backend_buffer_is_host(weight->buffer);", "false;"),
    ("k", "sycl",
     "if (!moe_pp_batched_has_expert_entries(weight->type, planned_layout, device_buffer)) { return false; }", ""),
    ("e", "sycl", "warn_batched_refusal(moe_pp_batched_refusal::GEMM, what);", ""),
    ("e", "sycl", "warn_batched_refusal(refusal, detail);", ""),
    ("h", "sycl", "if (!plan.all_experts_on_device(src0->name, n_experts, device)) { return false; }", ""),
    ("h", "sycl",
     "= plan.find_expert_entry(layer_id, static_cast<int>(e), role); if (!placement || !placement->on_device ||",
     "= plan.find_expert_entry(layer_id, static_cast<int>(e), role); if (!placement ||"),
    ("j", "sycl", f"{{ {KNOB_CONST}, &g_ggml_sycl_moe_pp_f16_batched_aos, 1 }},", ""),
    ("j", "sycl", "g_ggml_sycl_moe_pp_f16_batched_aos = ggml_sycl_moe_pp_f16_batched_aos_enabled() ? 1 : 0;", ""),
    ("j", "catalog", f"| `{KNOB}=0` |", f"| `{KNOB}` |"),
    ("l", "sycl", "if (!ggml_sycl_moe_pp_onednn_f16_enabled()) { return false; }", ""),
    ("l", "sycl", " && ggml_sycl_moe_pp_onednn_f16_enabled();", ";"),
    ("l", "common", 'std::getenv("GGML_SYCL_MOE_PP_ONEDNN_F16"); return env == nullptr || std::atoi(env) != 0;',
     'std::getenv("GGML_SYCL_MOE_PP_ONEDNN_F16"); return env != nullptr && std::atoi(env) != 0;'),
    # The wait on the half's readers gone, made a host wait, or the readers never recorded; both halves one.
    ("m", "sycl", "stream->ext_oneapi_submit_barrier(half_readers[phase.buffer]);", ""),
    ("m", "sycl", "stream->ext_oneapi_submit_barrier(half_readers[phase.buffer]);",
     "sycl::event::wait(half_readers[phase.buffer]);"),
    ("m", "sycl", "half_readers[phase.buffer].push_back(*group_event);", ""),
    ("m", "sycl", PHASE_OFFSET, "sycl::half * phase_weights = batched_weights;"),
    ("m", "tables", "plan.buffers = fit >= 2 ? 2 : 1;", "plan.buffers = 1;"),
    # Every phase reuses a buffer again (the one-group serialization back); the wait taken whatever the predicate
    # says; the offset taken from the buffer whatever the predicate says; the offset never by buffer; the bound gone.
    ("m", "tables", "{ return groups.groups != 1; }", "{ return true; }"),
    ("m", "tables", "? phase_buffer * half_elems : phase_begin * weight_elems;", "? phase_begin * weight_elems : "
     "phase_begin * weight_elems;"),
    ("m", "sycl", "if (moe_pp_batched_phases_reuse_buffers(f16_groups) && !half_readers[phase.buffer].empty())",
     "if (!half_readers[phase.buffer].empty())"),
    ("m", "sycl", PHASE_OFFSET, "sycl::half * phase_weights = batched_weights + phase.buffer * half_elems;"),
    ("m", "sycl", PHASE_BOUND, ""),
    # The route log's weight figure: one window's buffers whatever the predicate says, the plan re-derived there.
    ("m", "sycl", WEIGHT_USED, "const size_t weight_used_bytes = pp_woq_arm ? weight_bytes : "
     "phase_buffers * half_elems * sizeof(sycl::half);"),
    ("m", "sycl", "f16_used_elems = moe_pp_batched_phases_reuse_buffers(f16_groups) ?",
     "f16_used_elems = f16_groups.groups != 1 ?"),
    ("m", "sycl", "phase_buffers * half_elems : active_count * weight_elems;",
     "phase_buffers * half_elems : phase_buffers * half_elems;"),
    # A second definition of either function, beside the first or in the executor's file.
    ("m", "tables", "inline bool moe_pp_batched_phases_reuse_buffers(",
     "inline bool moe_pp_batched_phases_reuse_buffers(const moe_pp_batched_f16_groups & groups, int) { "
     "return groups.groups != 1; } inline bool moe_pp_batched_phases_reuse_buffers("),
    ("m", "tables", "inline size_t moe_pp_batched_phase_weight_offset(",
     "inline size_t moe_pp_batched_phase_weight_offset(int) { return 0; } "
     "inline size_t moe_pp_batched_phase_weight_offset("),
    ("m", "sycl", "static size_t g_tensor_inventory_pp_moe_onednn_weight_slot_bytes = 0;",
     "static size_t moe_pp_batched_phase_weight_offset(size_t b) { return b; } "
     "static size_t g_tensor_inventory_pp_moe_onednn_weight_slot_bytes = 0;"),
    # The wait on the other half's readers; a host wait after the phase loop, outside the compare arm.
    ("m", "sycl", "stream->ext_oneapi_submit_barrier(half_readers[phase.buffer]);",
     "stream->ext_oneapi_submit_barrier(half_readers[1 - phase.buffer]);"),
    ("m", "sycl", "gemm_profile_end = ggml_sycl_submit_marker<class mxfp4_pp_profile_gemm_end_marker>(*stream);",
     "gemm_profile_end = ggml_sycl_submit_marker<class mxfp4_pp_profile_gemm_end_marker>(*stream); stream->wait();"),
    ("n", "sycl", "ggml_sycl_moe_pp_dequant_label(src0->type)",
     'std::string("moe.pp.dequant.") + ggml_type_name(src0->type)'),
    ("n", "sycl", "static const std::array<std::string, GGML_TYPE_COUNT> names = []() {",
     "const std::array<std::string, GGML_TYPE_COUNT> names = []() {"),
    ("b", "tables", "case GGML_TYPE_Q2_0: case GGML_TYPE_Q4_1:",
     "case GGML_TYPE_Q2_0: case GGML_TYPE_Q4_1: case GGML_TYPE_TQ2_0:"),
    ("o", "device_test", "GGML_TYPE_IQ4_NL,", ""),
    ("o", "device_test", "constexpr ggml_type k_types[] = {", "constexpr ggml_type k_types[] = { GGML_TYPE_IQ1_S,"),
    ("o", "ggml_c", "type == GGML_TYPE_IQ1_S;", "type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M;"),
    ("p", "device_test", 'g_log.count_since(from, { "[PP-MOE] device=", type_field.c_str() })',
     'g_log.count_since(0, { "[PP-MOE] device=", type_field.c_str() })'),
    ("p", "device_test", "pp_moe != (first_of_type ? 1u : 0u) || pp_total != 1", "pp_moe != 1"),
    ("p", "sycl", "if (batched_log.fetch_add(1, std::memory_order_relaxed) < 96)",
     "if (batched_log.fetch_add(1, std::memory_order_relaxed) < 64)"),
    ("p", "device_test", "static_assert(k_cases <= k_route_log_cap,", "static_assert(true,"),
    ("i", "sycl", "static size_t g_tensor_inventory_pp_moe_onednn_weight_slot_bytes = 0;",
     "static size_t g_tensor_inventory_pp_moe_onednn_weight_slot_bytes = 0; "
     "static size_t g_tensor_inventory_pp_moe_onednn_scratch_bytes = 0;"),
    # The n_expert x align256 widening re-added at the planner site, on the global or on the call's argument.
    ("i", "sycl", PLAN_SITE_HEAD,
     PLAN_SITE_HEAD + " g_tensor_inventory_pp_moe_onednn_weight_slot_bytes = std::max("
     "g_tensor_inventory_pp_moe_onednn_weight_slot_bytes, ((f16_bytes + 255) & ~static_cast<size_t>(255)) * "
     "static_cast<size_t>(inventory->n_expert));"),
    ("i", "sycl", PLAN_SITE_CALL,
     "unified_cache_set_planned_pp_moe_onednn_scratch( ctx->device, ((f16_bytes + 255) & ~static_cast<size_t>(255)) * "
     "static_cast<size_t>(inventory->n_expert),"),
    ("q", "common", "value < 1 || value > 1024", "value < 1"),
    ("q", "sycl", f"{{ {GROUP_KNOB_CONST}, &g_ggml_sycl_moe_pp_f16_group_experts_max, 0 }},", ""),
    ("q", "catalog", f"| `{GROUP_KNOB}=<1..1024>` |", f"| `{GROUP_KNOB}` |"),
    ("q", "sycl", 'out=%.1fMB " "groups=%zu', 'out=%.1fMB " "phases=%zu'),
    ("q", "device_test", "groups < k_min_groups", "groups < 1"),
    # No single-expert case; no child run; the cap set for the whole run; the child's cap never set; a loose count.
    ("q", "device_test", GROUP_CAPS, "constexpr int k_group_caps[] = { 3 };"),
    ("q", "device_test", "if (!run_capped_child(cap)) {", "if (false) {"),
    ("q", "device_test", 'setenv("GGML_SYCL_MOE_ROUTE_LOG", "1", 1);',
     f'setenv("GGML_SYCL_MOE_ROUTE_LOG", "1", 1); setenv("{GROUP_KNOB}", "3", 1);'),
    ("q", "device_test", f'setenv("{GROUP_KNOB}", argv[2], 1);', ""),
    ("q", "device_test", GROUP_EXACT, "groups < want_groups"),
    # A waited child passes whatever its status; the child exits 0 on a failed case; a failed child is not counted.
    ("q", "device_test", "if (waited && WIFEXITED(status) && WEXITSTATUS(status) == 0) {", "if (waited) {"),
    ("q", "device_test", "true, group_cap) ? 0 : 1;", "true, group_cap) ? 0 : 0;"),
    ("q", "device_test", "if (!run_capped_child(cap)) { ++failed; }", "if (!run_capped_child(cap)) { }"),
    # No barrier on the GEMMs; the scratch released before the barrier; the leases not held; the helper holds none.
    ("r", "sycl", FAIL_BARRIER, "const sycl::event drained = stream->ext_oneapi_submit_barrier();"),
    ("r", "sycl", FAIL_BARRIER + " " + FAIL_RETAIN + " " + CLAIM_RELEASE,
     CLAIM_RELEASE + " " + FAIL_BARRIER + " " + FAIL_RETAIN),
    ("r", "sycl", FAIL_RETAIN, ""),
    # The failure path no longer drains, or keeps its own copy of the drain beside the helper.
    ("r", "sycl", DRAIN_CALL, "if (tiled_claimed_op) {"),
    ("r", "sycl", DRAIN_CALL, FAIL_BARRIER + " " + DRAIN_CALL),
    ("r", "sycl", "ggml_sycl::retain_handles_until_event(std::move(leases), until);", ""),
    ("s", "tables", "return moe_mmvq_admission_supports_type(type) && moe_pp_batched_f16_supports_layout(",
     "return moe_pp_batched_f16_supports_layout("),
    ("s", "tables", "&& moe_pp_batched_f16_supports_layout(type, GGML_LAYOUT_AOS); }",
     "&& moe_pp_batched_f16_supports_layout(type, GGML_LAYOUT_SOA); }"),
    ("s", "sycl", STAGED_CLAIM_HEAD, "planned = plan != nullptr && moe_mmvq_admission_supports_type(weight->type) && "),
    ("s", "sycl", STAGED_CLAIM_HEAD,
     STAGED_CLAIM_HEAD + "moe_pp_batched_f16_supports_layout(weight->type, GGML_LAYOUT_AOS) && "),
    ("s", "sycl", STAGED_CLAIM_HEAD + STAGED_CLAIM_KNOBS, STAGED_CLAIM_HEAD),
    ("s", "cost", "if (!moe_pp_batched_takes_staged_type(t)) {",
     "if (!moe_pp_batched_f16_supports_layout(t, GGML_LAYOUT_AOS)) {"),
    ("s", "cost", SEED_MXFP4, SEED_MXFP4 + " if (t == GGML_TYPE_IQ4_XS) { return 5000.0; }"),
    ("s", "cost", SEED_GATE, ""),
]


@pytest.mark.parametrize("claim,key,old,new", MUTANTS, ids=[f"{m[0]}-{i}" for i, m in enumerate(MUTANTS)])
def test_mutant_turns_claim_red(sources, claim, key, old, new):
    assert old in sources[key], f"mutant anchor missing from {FILES[key].name}: {old!r}"
    mutated = dict(sources)
    mutated[key] = sources[key].replace(old, new, 1)
    assert CLAIMS[claim](mutated), f"claim ({claim}) stayed green under the mutant {old!r} -> {new!r}"


def test_claims_are_not_vacuous(sources):
    """Positive controls: the probes find the real anchors, so a red claim is a missing shape, not a parse miss."""
    assert lambda_body(sources["sycl"]) is not None
    assert claim_predicate_body(sources["sycl"]) is not None
    assert one_definition(sources["convert"], "ggml_get_to_fp16_sycl_for_layout") is not None
    assert one_definition(sources["sycl"], "ggml_sycl_moe_query_route_capability") is not None
    assert one_definition(sources["ggml_c"], "ggml_quantize_requires_imatrix") is not None
    assert one_definition(sources["tables"], STAGED_TYPE) is not None
    assert one_definition(sources["cost"], "moe_exec_dev_gflops_seed") is not None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
