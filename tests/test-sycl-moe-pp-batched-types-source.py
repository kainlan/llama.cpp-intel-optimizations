#!/usr/bin/env python3
"""Source gate for the type-keyed batched MoE prompt executor (llama.cpp-348c).

The batched oneDNN prompt executor (the try_pp_onednn_f16_batched lambda in ggml_sycl_mul_mat_id) used to
take MXFP4 experts only. Its f16 arm now takes every expert type that has an AOS dequant-to-f16 kernel, and SOA too
for the types the planner materializes SOA that have a per-expert SOA dequant. These claims pin that shape:

  (a) one support table: the claim predicate and the lambda's candidate gate both ask
      ggml_sycl_moe_pp_batched_takes_layout(), which asks moe_pp_batched_supports_layout() in moe-mmvq-tables.hpp,
      and neither tests GGML_TYPE_MXFP4 itself;
  (b) the f16 table lists each required type on AOS, Q4_0, Q8_0, Q4_K and Q6_K on AOS and SOA, MXFP4 on SOA only,
      never the direct-kernel types, and every type it lists has a case in convert.cpp's
      ggml_get_to_fp16_sycl_for_layout; that case decodes SOA with a reorder kernel for Q4_0, Q4_K and Q6_K, and
      since it answers Q8_0 with the AOS kernel for every layout, the lambda takes Q8_0 SOA to
      dequantize_row_q8_0_soa_to_fp16_rowmajor and never asks it; the WOQ table is MXFP4 on SOA and XMX_TILED;
  (c) the Q4_K device-weight CPU fallback (a D2H copy per dispatch) does not fire for a prompt the executor claims,
      and the mechanism is precedence, not a term on the fallback: a claimed prompt drops the planner's prompt route
      (planner_needs_prompt_route), so an all-local claimed prompt never enters the hybrid world where the fallback
      lives, and the fallback itself stays exactly Q4_K for the hybrid ops the executor never runs;
  (d) decode keeps its ne12 == 1 / ids->ne[1] > 1 terms: the hybrid-with-plan disjunct, the two claim call sites
      and the lambda's single-token refusal;
  (e) one refusal enum with a name for every cause, every refusal in the lambda names one, and one WARN line per
      (device, cause) under GGML_SYCL_MOE_PP_ONEDNN_F16_BATCHED_TRACE, printed by one helper that the GEMM-stage
      failure uses too, so every refusal reads "[PP-MOE] refused";
  (f) one [PP-MOE] INFO line per (device, type), naming the type, layout, experts per dispatch and the planned slot;
  (g) the MXFP4 profiler label strings are byte-identical and keep their counts, the non-MXFP4 labels are moe.pp.*,
      and the MXFP4 dispatch-record profile stays MXFP4-only;
  (h) the capability query admits a prompt-phase op the executor takes only after the MMVQ check (so no type that
      has an MMVQ-ID kernel changes kernel), and MUL_MAT_ID admission takes a type with no MMVQ-ID kernel only for
      a prompt the executor claims, which is only a layer whose every expert is planned on this device: the claim's
      planned-layout lookup refuses a layer with a host or other-device expert, so a hybrid layer is never admitted;
  (i) the planned weight slot is widened at plan time from the f16 table, so the executor never needs an
      unplanned allocation;
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
  (m) the plan-time widening of the ring's weight slot is logged once per load (the early and the late plan of one
      load both populate the inventory), and no unread ring total is kept beside the unified cache's own.

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
    "convert": SYCL_DIR / "convert.cpp",
    "gemm": SYCL_DIR / "gemm.hpp",
    "common": SYCL_DIR / "common.hpp",
    "layer_plan": SYCL_DIR / "moe-layer-plan.hpp",
    "catalog": ROOT / "docs" / "backend" / "sycl-env-vars.md",
}

# The brief's list (Q4_0 plus the Qwen3.8 GSQ-RCO mix types it named) plus the rest of the GSQ-RCO census
# (iq4_nl, q2_0, iq2_s, iq2_xs, iq2_xxs): see the memory "a GGUF file name is not its expert quant type".
REQUIRED_AOS = [
    "GGML_TYPE_Q4_0", "GGML_TYPE_Q8_0", "GGML_TYPE_Q4_K", "GGML_TYPE_Q5_K", "GGML_TYPE_Q6_K", "GGML_TYPE_IQ3_XXS",
    "GGML_TYPE_IQ3_S", "GGML_TYPE_IQ4_XS", "GGML_TYPE_IQ4_NL", "GGML_TYPE_Q2_0", "GGML_TYPE_IQ2_S",
    "GGML_TYPE_IQ2_XS", "GGML_TYPE_IQ2_XXS",
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
    q8_soa = ("const bool pp_q8_0_soa = src0->type == GGML_TYPE_Q8_0 && "
              "active_experts.front().layout == GGML_LAYOUT_SOA;")
    if q8_soa not in lam:
        fails.append("the lambda does not name its Q8_0 SOA experts (pp_q8_0_soa)")
    q8_kernel = lam.find("dequantize_row_q8_0_soa_to_fp16_rowmajor(")
    if q8_kernel < 0 or (enclosing_if_condition(lam, q8_kernel) or "").strip() != "pp_q8_0_soa":
        fails.append("the lambda does not take Q8_0 SOA experts to dequantize_row_q8_0_soa_to_fp16_rowmajor")
    getter = lam.find("pp_to_fp16 = ggml_get_to_fp16_sycl_for_layout(")
    if getter < 0 or "!pp_q8_0_soa" not in (enclosing_if_condition(lam, getter) or ""):
        fails.append("the lambda asks ggml_get_to_fp16_sycl_for_layout for Q8_0 SOA, which answers the AOS kernel")
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
    if "prompt_batch && ggml_sycl_moe_pp_onednn_batched_claims_tensor(src0, ctx.device)" not in claims:
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
        "ggml_sycl_moe_pp_onednn_batched_claims_tensor(src0, ctx.device);",
        "const bool prompt_batch = ids && ids->ne[1] > 1;",
        "const bool pp_onednn_batched_claims_op = ne12 > 1 && "
        "ggml_sycl_moe_pp_onednn_batched_claims_tensor(src0, ctx.device);",
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
    values = enum_values(s[holder], head)
    if not values or len(values) > 32:
        fails.append(f"the refusal enum has {len(values or [])} values; the per-device latch holds at most 32")
    names = one_definition(s[holder], "moe_pp_batched_refusal_name")
    if names is None:
        fails.append("moe_pp_batched_refusal_name is not defined exactly once beside the enum")
    else:
        for v in values or []:
            if f"case moe_pp_batched_refusal::{v}:" not in names:
                fails.append(f"moe_pp_batched_refusal_name has no case for {v}")
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
        if "pp_f16_batched_trace" not in cond or "ggml_sycl_moe_pp_batched_refusal_first(" not in cond:
            fails.append("the refusal WARN is not guarded by the trace knob and the per-(device, cause) latch")
        stmt_start = s["sycl"].rfind("GGML_LOG_", 0, pos)
        if not s["sycl"].startswith("GGML_LOG_WARN(", stmt_start):
            fails.append("the refusal line is not emitted with GGML_LOG_WARN")
        stmt = statement(s["sycl"], None, start=stmt_start) or ""
        if "moe_pp_batched_refusal_name(" not in stmt:
            fails.append("the refusal WARN does not print the cause name")
    latch = one_definition(s["sycl"], "ggml_sycl_moe_pp_batched_refusal_first")
    if latch is None or "fetch_or(" not in latch or "std::atomic<uint32_t>" not in latch:
        fails.append("ggml_sycl_moe_pp_batched_refusal_first is not one per-device atomic bitmask latch")
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
    if cond.strip() != "ggml_sycl_moe_pp_batched_first_run(ctx.device, src0->type)":
        fails.append(f"the [PP-MOE] line is guarded by `{cond.strip()}`, not the per-(device, type) latch")
    stmt_start = sycl.rfind("GGML_LOG_", 0, pos)
    if not sycl.startswith("GGML_LOG_INFO(", stmt_start):
        fails.append("the [PP-MOE] line is not emitted with GGML_LOG_INFO")
    stmt = statement(sycl, None, start=stmt_start) or ""
    for field in ("experts=", "planned_weight_slot="):
        if field not in stmt:
            fails.append(f"the [PP-MOE] line does not name {field}")
    latch = one_definition(sycl, "ggml_sycl_moe_pp_batched_first_run")
    if latch is None or "fetch_or(" not in latch or "std::atomic<uint64_t>" not in latch:
        fails.append("ggml_sycl_moe_pp_batched_first_run is not one per-device atomic bitmask latch")
    if "static_assert(GGML_TYPE_COUNT <= 64" not in sycl:
        fails.append("no static_assert that every ggml_type fits the 64-bit first-run mask")
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


def claim_i_sizing(s):
    fails = []
    sycl = s["sycl"]
    body = one_definition(sycl, "ggml_sycl_moe_pp_batched_f16_weight_slot_bytes")
    if body is None:
        return ["ggml_sycl_moe_pp_batched_f16_weight_slot_bytes is not defined exactly once"]
    # The sizing asks the executor's own predicate on AOS (one fact, one source): the f16 arm's entries for every
    # type but MXFP4, under the same knobs the dispatch reads.
    if "ggml_sycl_moe_pp_batched_takes_layout(tensor.type, GGML_LAYOUT_AOS)" not in body:
        fails.append("the f16 slot sizing does not ask the executor's predicate which AOS experts it takes")
    if "n_expert" not in body:
        fails.append("the f16 slot sizing does not size for every expert (n_expert)")
    if "sizeof(sycl::half)" not in body:
        fails.append("the f16 slot sizing does not count f16 bytes")
    assign = statement(sycl, "g_tensor_inventory_pp_moe_onednn_weight_slot_bytes = std::max(") or ""
    if "ggml_sycl_moe_pp_batched_f16_weight_slot_bytes(" not in assign:
        fails.append("the planned weight slot is not widened by the f16 slot sizing")
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


def claim_m_widening(s):
    fails = []
    sycl = s["sycl"]
    literal = '"[SYCL-PLAN] PP MoE oneDNN batched weight slot widened'
    if sycl.count(literal) != 1:
        return [f"{literal} appears {sycl.count(literal)} times in ggml-sycl.cpp, want 1"]
    cond = enclosing_if_condition(sycl, sycl.find(literal)) or ""
    if "&& ggml_sycl_pp_moe_widened_first(ctx->device, plan_owner, " not in cond or "||" in cond:
        fails.append("the widened line is not guarded by the per-load latch, so each of a load's two plans logs it")
    latch = one_definition(sycl, "ggml_sycl_pp_moe_widened_first")
    if latch is None:
        fails.append("ggml_sycl_pp_moe_widened_first is not defined exactly once")
    else:
        if "last.owner == owner && last.from_bytes == from_bytes && last.to_bytes == to_bytes" not in latch:
            fails.append("the widening latch does not key on the load owner and both slot sizes")
        if "owner.generation == 0" not in latch:
            fails.append("the widening latch latches a load no staged owner holds")
    if "g_tensor_inventory_pp_moe_onednn_scratch_bytes" in sycl:
        fails.append("g_tensor_inventory_pp_moe_onednn_scratch_bytes is kept although nothing reads it")
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
    "i": claim_i_sizing,
    "j": claim_j_knob,
    "k": claim_k_expert_entries,
    "l": claim_l_f16_knob,
    "m": claim_m_widening,
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
    ("b", "sycl", "if (pp_q8_0_soa) {", "if (false) {"),
    ("b", "sycl", "src0->type != GGML_TYPE_MXFP4 && !pp_q8_0_soa", "src0->type != GGML_TYPE_MXFP4"),
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
    ("f", "sycl", "if (ggml_sycl_moe_pp_batched_first_run(ctx.device, src0->type))", "if (true)"),
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
    ("i", "sycl", "ggml_sycl_moe_pp_batched_f16_weight_slot_bytes(inventory))", "size_t(0))"),
    ("i", "sycl", "ggml_sycl_moe_pp_batched_takes_layout(tensor.type, GGML_LAYOUT_AOS)", "true"),
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
    ("m", "sycl", "&& ggml_sycl_pp_moe_widened_first(ctx->device, plan_owner, ",
     "|| ggml_sycl_pp_moe_widened_first(ctx->device, plan_owner, "),
    ("m", "sycl", "last.owner == owner && ", ""),
    ("m", "sycl", "static size_t g_tensor_inventory_pp_moe_onednn_weight_slot_bytes = 0;",
     "static size_t g_tensor_inventory_pp_moe_onednn_weight_slot_bytes = 0; "
     "static size_t g_tensor_inventory_pp_moe_onednn_scratch_bytes = 0;"),
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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
