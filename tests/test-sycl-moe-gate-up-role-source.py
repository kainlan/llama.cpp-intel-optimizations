#!/usr/bin/env python3
"""Source contract: the fused gate_up expert role, and the seam where the direct MMID route steps aside for the batched
prompt executor.

Host-only: reads ggml-sycl.cpp, moe-mmvq-tables.hpp, the loader's expert-name header and llama-model.cpp, and the
device test test-moe-pp-batched-dequant.cpp. It builds nothing, loads no model and touches no device.

A model that holds its gate and up expert matrices as one (ffn_gate_up_exps, Inkling 2 Mini) gives the backend a fourth
expert role. The graph splits and activates the fused output itself (build_moe_ffn), so the role has no gate or up
partner. NVFP4 experts of that role, and of the separate roles, run their prompts on the batched oneDNN executor, which
the Q1_0/NVFP4 direct MMID route runs ahead of. This gate holds both:

  G1  the role enum is GATE 0, UP 1, DOWN 2, GATE_UP 3, UNKNOWN 4, and the profile's role count is UNKNOWN + 1, which
      sizes the role-indexed arrays and bounds their index;
  G2  the classifier tests the fused name before the gate name (the fused name contains "ffn_gate") and answers GATE_UP;
  G3  every switch on the role names all five roles and has no default arm, and there are at least the two known ones
      (the role's name and the gate/up/down group registry, where GATE_UP only breaks);
  G4  the batched executor's resident claim admits GATE_UP and its staged claim does not: the ring's windows stage the
      separate gate, up and down host matrices only;
  G5  none of the paired gate/up paths takes GATE_UP: the gate/up XMX bundle and the two single-XMX helpers name GATE
      and UP alone, and the pair scan matches a GLU only to a layer that has both a gate output and an up output;
  G6  the PP MoE ring's consumer test skips UNKNOWN only, so a fused weight's type counts;
  G7  the loader sizes the ring from one name test, the header's, which names all four expert families and is
      called in the sizing loop with no wrapper; the loader holds no strstr of an expert family's name;
  S1  the seam predicate: a prompt, the executor's claim and a planned ring weight slot, all three;
  S2  the direct route's gate ends with `!pp_direct_deferred`, defined once before it from the route's types, the seam
      predicate, the device's claim and its planned ring weight slot;
  S3  the claim the MMID entry asks answers false for the weight of a node on its second pass;
  S4  the wrapper runs the node once, and once more only when the first pass named it, with the claim suppressed for its
      weight and the suppression ended on every exit; the dispatcher calls the wrapper, and only the wrapper calls the
      MMID entry;
  S5  a decline before the executor's first submission reaches the re-entry and not a throw: the executor marks its
      first submission once, after its scratch claim is marked used and after its last scratch decline, before its
      first queue call, with no queue call before it; the deferred throw in the decline tail needs that mark; the call
      site asks for the second pass right after the executor returns false; the decline tail records the cause;
  S6  after the first submission a deferred node's decline throws: the GEMM stage's tail throws for it, and the only
      bare `return false;` statements after the mark are the GEMM tail's, the deferred-expert propagation (whose
      declines go through the decline tail) and the build without oneDNN;
  S7  the prompt count skips a node's second pass, whose snapshot its first pass counted;
  S8  the device test runs NVFP4 on the plain fixture and on a fused gate_up fixture and two special-scale fixtures,
      writes no 0xFF scale, and fails a case that printed the re-entry line;
  S9  no other prompt dispatch takes a node the direct route stepped aside for: the planned-local batched MMVQ try,
      which runs ahead of the executor under two GPUs or GGML_SYCL_MOE_SINGLE_GPU_PP_PLANNED_BATCHED=1 and is neither
      the executor nor the direct route, is called once, behind `!pp_direct_deferred`.

Checks run on comment-blanked text. NOT VACUOUS: --self-test runs the gate on in-memory mutants of the real sources.
Each mutant names the pin that must catch it, and is reported if it survives, if a different pin catches it, or if its
anchor no longer matches exactly once (NOT APPLIED). ctest registers it with --self-test.

WHAT THIS DOES NOT PROVE. It reads text. That the executor's NVFP4 and fused gate_up results match the CPU's is the
device test's job (test-moe-pp-batched-dequant), and that the seam pays is the lead's run on a card.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "backend": ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp",
    "tables": ROOT / "ggml/src/ggml-sycl/moe-mmvq-tables.hpp",
    "names": ROOT / "src/llama-sycl-expert-names.h",
    "model": ROOT / "src/llama-model.cpp",
    "device_test": ROOT / "ggml/src/ggml-sycl/tests/test-moe-pp-batched-dequant.cpp",
}

ROLES = ("MOE_TENSOR_GATE", "MOE_TENSOR_UP", "MOE_TENSOR_DOWN", "MOE_TENSOR_GATE_UP", "MOE_TENSOR_UNKNOWN")
ENUM = ("enum moe_tensor_type : int { MOE_TENSOR_GATE = 0, MOE_TENSOR_UP = 1, MOE_TENSOR_DOWN = 2, "
        "MOE_TENSOR_GATE_UP = 3, MOE_TENSOR_UNKNOWN = 4, };")
ROLE_COUNT = "static constexpr int MOE_PROFILE_ROLE_COUNT = static_cast<int>(MOE_TENSOR_UNKNOWN) + 1;"
ROLE_ARRAYS = ("std::array<route_stats, MOE_PROFILE_ROLE_COUNT> routes{};",
               "std::array<role_timing, MOE_PROFILE_ROLE_COUNT> role_timings{};")
ROLE_INDEX = "return idx >= 0 && idx < MOE_PROFILE_ROLE_COUNT ? idx : static_cast<int>(MOE_TENSOR_UNKNOWN);"
CLASSIFY_FN = "static moe_tensor_type moe_classify_tensor(const char * name) {"
FUSED_TEST = 'if (strstr(name, "ffn_gate_up_exps")) { return MOE_TENSOR_GATE_UP; }'
GATE_TEST = 'if (strstr(name, "ffn_gate_exps")) { return MOE_TENSOR_GATE; }'
NAME_FN = "static const char * moe_tensor_type_name(moe_tensor_type type) {"
REGISTRY_SWITCH = "switch (meta.tensor_role) {"
REGISTRY_FUSED = "case MOE_TENSOR_GATE_UP: break; case MOE_TENSOR_UNKNOWN: break; }"
CLAIM_FN = "bool host_staged) {"
CLAIM_ROLE = "const moe_tensor_type role = moe_classify_tensor(weight->name ? weight->name : \"\");"
CLAIM_TRIPLET = "const bool triplet_role = role == MOE_TENSOR_GATE || role == MOE_TENSOR_UP || role == MOE_TENSOR_DOWN;"
CLAIM_GATE = "if (!triplet_role && !(role == MOE_TENSOR_GATE_UP && !host_staged)) { return false; }"
CLAIM_PLANNED = "layout_mode planned_layout = GGML_LAYOUT_AOS;"
BUNDLE_FN = "static bool ggml_sycl_moe_gateup_bundle4_supported(const ggml_tensor * src0, int device) {"
BUNDLE_ROLE = "if (moe_kind != MOE_TENSOR_GATE && moe_kind != MOE_TENSOR_UP) {"
SINGLE_FNS = ("static bool ggml_sycl_moe_gateup_single_xmx_enabled_from_env(const char * env_value, "
              "const char * name) {",
              "static bool ggml_sycl_moe_gateup_single_xmx_enabled(const char * name) {")
SINGLE_ROLE = "return moe_kind == MOE_TENSOR_GATE || moe_kind == MOE_TENSOR_UP;"
SCAN_FN = "static void moe_layer_scan_graph_topology("
GLU_PAIR = "if (!gate_out || !up_out) { continue; }"
GLU_OP = "if (node->op != GGML_OP_GLU || !node->src[0] || !node->src[1]) { continue; }"
RING_FN = "static bool ggml_sycl_inventory_ring_has_consumer(const ggml_sycl_tensor_inventory * inventory) {"
RING_SKIP = "if (moe_classify_tensor(t.name) == MOE_TENSOR_UNKNOWN || t.ne[0] <= 0 || t.ne[1] <= 0) { continue; }"
NAMES_FN = "static inline bool llama_sycl_is_moe_expert_weight_name(const char * name) {"
NAMES_KINDS = 'static const char * kinds[] = { "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_up_exps" };'
LOADER_CALL = "if (!llama_sycl_is_moe_expert_weight_name(tensor.name)) { continue; }"
LOADER_INCLUDE = '#include "llama-sycl-expert-names.h"'
SEAM_FN = ("inline bool moe_mmid_direct_defers_to_executor(bool prompt, bool executor_claims, "
           "size_t ring_weight_slot_bytes) {")
SEAM_BODY = "{ return prompt && executor_claims && ring_weight_slot_bytes > 0; }"
ROUTE_CLAIM = "const bool pp_onednn_batched_route = ggml_sycl_moe_pp_onednn_batched_claims_tensor(src0, ctx.device);"
DEFERRED_DEF = ("const bool pp_direct_deferred = ggml_sycl_moe_mmid_direct_route_runs_type(src0->type) && "
                "moe_mmid_direct_defers_to_executor(ne12 != 1, pp_onednn_batched_route, "
                "ggml_sycl::unified_cache_get_planned_pp_moe_onednn_weight_slot_bytes(ctx.device));")
DIRECT_GATE = ("ggml_sycl_moe_mmid_direct_route_runs_type(src0->type) && direct_layout == GGML_LAYOUT_AOS && "
               "direct_admission && !pp_direct_deferred) {")
SUPPRESS_FN = "static bool ggml_sycl_moe_pp_onednn_batched_claims_tensor(const ggml_tensor * weight, int device) {"
SUPPRESS_BODY = ("{ if (weight != nullptr && weight == g_mmid_direct_reentry.suppressed_weight) { return false; } "
                 "return ggml_sycl_moe_pp_onednn_batched_claims_tensor(nullptr, weight, device, false); }")
WRAPPER_FN = ("static void ggml_sycl_mul_mat_id_with_direct_reentry(ggml_backend_sycl_context & ctx, "
              "ggml_tensor * dst) {")
WRAPPER_BODY = ("{ g_mmid_direct_reentry.requested = nullptr; ggml_sycl_mul_mat_id(ctx, dst); "
                "if (g_mmid_direct_reentry.requested != dst) { return; } g_mmid_direct_reentry.requested = nullptr; "
                "g_mmid_direct_reentry.suppressed_weight = dst->src[0]; struct suppression_end { ~suppression_end() "
                "{ g_mmid_direct_reentry.suppressed_weight = nullptr; } } end; ggml_sycl_mul_mat_id(ctx, dst); }")
ENTRY_CALL = "ggml_sycl_mul_mat_id(ctx, dst);"
DISPATCH_CALL = "ggml_sycl_mul_mat_id_with_direct_reentry(ctx, dst);"
LAMBDA = "auto try_pp_onednn_f16_batched = [&]() -> bool {"
DECLINE_FN = ("const auto decline_batched = [&](moe_pp_batched_refusal refusal, const char * detail, "
              "int64_t expert_id, bool route_logged) -> bool {")
DECLINE_RECORD = "pp_batched_refusal = refusal;"
DEFERRED_THROW = "if (pp_direct_deferred && pp_batched_submitted) {"
SUBMITTED = "pp_batched_submitted = true;"
MARK_USED = "batched_scratch_claim.mark_used();"
LAST_SCRATCH_DECLINE = "return reject_batched(moe_pp_batched_refusal::SCRATCH_SLOT_SIZE, nullptr);"
FIRST_SUBMISSION = "stream->ext_oneapi_submit_barrier(ready_events);"
GEMM_FN = "auto gemm_stage_failed = [&](const char * what) -> bool {"
GEMM_DEFERRED = "if (pp_direct_deferred) {"
CALL_SITE = "if (try_pp_onednn_f16_batched()) { return; }"
REENTRY_OPEN = "if (pp_direct_deferred) {"
REENTRY_TAIL = "g_mmid_direct_reentry.requested = dst; return; }"
REENTRY_LATCH = "ggml_sycl_moe_pp_direct_reentry_first(ctx.device, pp_batched_refusal)"
REENTRY_TEXT = "direct MMID route re-entered"
POST_SUBMIT_RETURNS = 3
COUNT_FN = "static void ggml_sycl_moe_route_counts_add_prompt("
COUNT_SKIP = "if (src0 == g_mmid_direct_reentry.suppressed_weight) { return; }"
QUEUE_CALL = re.compile(
    r"\b(?:submit|ext_oneapi_submit_barrier|parallel_for|single_task|memcpy|memset|fill|copy|wait|wait_and_throw|"
    r"ggml_sycl_safe_memcpy|finish_or_release|record_done)\s*\(|\bdnnl::"
)
TEST_FUSED_NAME = '"blk.0.ffn_gate_up_exps.weight"'
TEST_VARIANTS = ("constexpr fixture_variant k_nvfp4_variants[] = { fixture_variant::SMALL_SCALES, "
                 "fixture_variant::LARGE_SCALES, fixture_variant::FUSED_GATE_UP };")
TEST_SCALES = re.compile(r"constexpr uint8_t (k_small_scales|k_large_scales)\[\] = \{([^}]*)\};")
TEST_REENTRY = 'const size_t reentered = g_log.count_since(from, { "direct MMID route re-entered" });'
TEST_REFUSAL = "if (refused != 0 || prompt_rf != 0 || unplanned != 0 || reentered != 0) {"
PLANNED_LOCAL_TRY = "try_pp_planned_local_batched()"
PLANNED_LOCAL_CALL = ("if (!pp_cpu_reference_force_router && pp_planned_local_batched_enabled && !pp_direct_deferred && "
                      "try_pp_planned_local_batched()) { return; }")


class ContractError(AssertionError):
    pass


class MutantNotApplied(Exception):
    pass


_COMMENT_OR_LITERAL = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'", re.DOTALL
)


def blank_comments(text: str) -> str:
    def repl(m: re.Match) -> str:
        t = m.group(0)
        if t.startswith("//") or t.startswith("/*"):
            return "".join("\n" if c == "\n" else " " for c in t)
        return t

    return _COMMENT_OR_LITERAL.sub(repl, text)


def squash(text: str) -> str:
    t = re.sub(r"\s+", " ", text)
    t = re.sub(r"\( ", "(", t)
    return re.sub(r" \)", ")", t)


def brace_text(text: str, open_at: int, pin: str) -> str:
    """The balanced block opening at `open_at`, skipping string and character literals."""
    depth = 0
    k = open_at
    n = len(text)
    while k < n:
        c = text[k]
        if c in "\"'":
            k += 1
            while k < n and text[k] != c:
                k += 2 if text[k] == "\\" else 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[open_at : k + 1]
        k += 1
    raise ContractError(f"FAIL [pin {pin}]: unbalanced block")


def need_once(hay: str, needle: str, pin: str, what: str) -> int:
    n = squash(needle)
    if hay.count(n) != 1:
        raise ContractError(f"FAIL [pin {pin}]: {what}: `{needle}` appears {hay.count(n)} times, not once")
    return hay.find(n)


def block(hay: str, marker: str, pin: str, what: str) -> str:
    """The `{ ... }` block that opens at the end of the one `marker` in `hay`, or right after it."""
    at = need_once(hay, marker, pin, what) + len(squash(marker))
    open_at = at - 1 if marker.endswith("{") else hay.find("{", at)
    return brace_text(hay, open_at, pin)


class Sources:
    def __init__(self, raws: dict[str, str]):
        self.raw = raws
        self.sq = {name: squash(blank_comments(text)) for name, text in raws.items()}


def role_switches(code: str) -> list[str]:
    """Every switch block of `code` (squashed) whose own arms, nested switches blanked, name a role."""
    out = []
    for m in re.finditer(r"\bswitch \(", code):
        body = brace_text(code, code.find("{", m.end()), "G3")
        inner = body[1:]
        own = body
        for n in re.finditer(r"\bswitch \(", inner):
            nested = brace_text(inner, inner.find("{", n.end()), "G3")
            own = own.replace(nested, "{}", 1)
        if "case MOE_TENSOR_" in own:
            out.append(own)
    return out


def check_roles(s: Sources) -> None:
    be = s.sq["backend"]
    # G1
    need_once(be, ENUM, "G1", "the role enum")
    need_once(be, ROLE_COUNT, "G1", "the profile's role count")
    for arr in ROLE_ARRAYS:
        need_once(be, arr, "G1", "a role-indexed array")
    need_once(be, ROLE_INDEX, "G1", "the role index's bound")
    # G2
    classify = block(be, CLASSIFY_FN, "G2", "the classifier")
    fused_at = classify.find(squash(FUSED_TEST))
    gate_at = classify.find(squash(GATE_TEST))
    if fused_at < 0 or gate_at < 0 or fused_at > gate_at:
        raise ContractError("FAIL [pin G2]: the classifier does not test the fused name, answering GATE_UP, before the "
                            "gate name")
    # G3
    switches = role_switches(be)
    if len(switches) < 2:
        raise ContractError(f"FAIL [pin G3]: {len(switches)} switches on the role, fewer than the two known ones")
    for sw in switches:
        missing = [r for r in ROLES if f"case {r}:" not in sw]
        if missing or "default:" in sw:
            raise ContractError(f"FAIL [pin G3]: a switch on the role misses {missing} or has a default arm: "
                                f"`{sw[:120]}`")
    if squash(REGISTRY_FUSED) not in block(be, REGISTRY_SWITCH, "G3", "the group registry's switch"):
        raise ContractError("FAIL [pin G3]: the group registry does something for GATE_UP other than break")
    block(be, NAME_FN, "G3", "the role's name")
    # G4
    claim = block(be, CLAIM_FN, "G4", "the batched executor's claim")
    at = [claim.find(squash(x)) for x in (CLAIM_ROLE, CLAIM_TRIPLET, CLAIM_GATE, CLAIM_PLANNED)]
    if min(at) < 0 or at != sorted(at):
        raise ContractError("FAIL [pin G4]: the claim does not admit the triplet roles and GATE_UP unstaged only, "
                            "before it reads the plan")
    # G5
    if squash(BUNDLE_ROLE) not in block(be, BUNDLE_FN, "G5", "the gate/up XMX bundle"):
        raise ContractError("FAIL [pin G5]: the gate/up XMX bundle admits a role other than GATE and UP")
    for fn in SINGLE_FNS:
        if squash(SINGLE_ROLE) not in block(be, fn, "G5", "a single-XMX helper"):
            raise ContractError("FAIL [pin G5]: a single-XMX helper admits a role other than GATE and UP")
    scan = block(be, SCAN_FN, "G5", "the pair scan")
    glu_at = scan.find(squash(GLU_OP))
    pair_at = scan.find(squash(GLU_PAIR))
    if glu_at < 0 or pair_at < glu_at or scan.count(squash(GLU_PAIR)) != 1:
        raise ContractError("FAIL [pin G5]: the pair scan matches a GLU to a layer without both a gate and an up "
                            "output")
    # G6
    ring = block(be, RING_FN, "G6", "the ring's consumer test")
    if squash(RING_SKIP) not in ring or len(re.findall(r"MOE_TENSOR_\w+", ring)) != 1:
        raise ContractError("FAIL [pin G6]: the ring's consumer test skips a role other than UNKNOWN")
    # G7
    names = block(s.sq["names"], NAMES_FN, "G7", "the expert-name test")
    if squash(NAMES_KINDS) not in names:
        raise ContractError("FAIL [pin G7]: the expert-name test does not name the four expert families")
    model = s.sq["model"]
    need_once(model, LOADER_INCLUDE, "G7", "the loader's include of the name test")
    need_once(model, LOADER_CALL, "G7", "the ring sizing's name test")
    if re.search(r'strstr\([^,;]*, "[^"]*ffn_[a-z_]*exps', model):
        raise ContractError("FAIL [pin G7]: the loader spells an expert family's name test of its own")


def check_seam(s: Sources) -> None:
    be = s.sq["backend"]
    # S1
    if block(s.sq["tables"], SEAM_FN, "S1", "the seam predicate") != squash(SEAM_BODY):
        raise ContractError("FAIL [pin S1]: the seam predicate is not a prompt, the claim and a planned ring slot")
    # S2
    claim_at = need_once(be, ROUTE_CLAIM, "S2", "the MMID entry's claim")
    def_at = need_once(be, DEFERRED_DEF, "S2", "the seam's definition")
    gate_at = need_once(be, DIRECT_GATE, "S2", "the direct route's gate")
    if not claim_at < def_at < gate_at:
        raise ContractError("FAIL [pin S2]: the claim, the seam and the direct route's gate are out of order")
    if be.count("pp_direct_deferred =") != 1:
        raise ContractError("FAIL [pin S2]: pp_direct_deferred is assigned more than once")
    # S3
    if block(be, SUPPRESS_FN, "S3", "the MMID entry's claim") != squash(SUPPRESS_BODY):
        raise ContractError("FAIL [pin S3]: the claim does not answer false for the weight of a node's second pass")
    # S4
    if block(be, WRAPPER_FN, "S4", "the re-entry wrapper") != squash(WRAPPER_BODY):
        raise ContractError("FAIL [pin S4]: the wrapper does not run the node once more, with the claim suppressed, "
                            "only when the first pass named it")
    if be.count(squash(ENTRY_CALL)) != 2:
        raise ContractError("FAIL [pin S4]: the MMID entry is called outside the wrapper")
    need_once(be, DISPATCH_CALL, "S4", "the dispatcher's call of the wrapper")
    # S5
    lam = block(be, LAMBDA, "S5", "the batched executor")
    flag_at = need_once(lam, SUBMITTED, "S5", "the executor's first-submission mark")
    if be.count(squash(SUBMITTED)) != 1:
        raise ContractError("FAIL [pin S5]: the first-submission mark is set outside the executor")
    used_at = need_once(lam, MARK_USED, "S5", "the scratch claim's use")
    last_decline_at = need_once(lam, LAST_SCRATCH_DECLINE, "S5", "the last scratch decline")
    first_at = need_once(lam, FIRST_SUBMISSION, "S5", "the executor's first queue call")
    if not last_decline_at < used_at < flag_at < first_at:
        raise ContractError("FAIL [pin S5]: the first-submission mark is not between the claim's use and the first "
                            "queue call")
    early = QUEUE_CALL.search(lam[:flag_at])
    if early:
        raise ContractError(f"FAIL [pin S5]: the executor calls the queue before its first-submission mark: "
                            f"`{lam[max(0, early.start() - 60):early.end()]}`")
    decline = block(lam, DECLINE_FN, "S5", "the decline tail")
    need_once(decline, DECLINE_RECORD, "S5", "the decline tail's cause")
    throw = block(decline, DEFERRED_THROW, "S5", "the deferred throw")
    if "throw ggml_sycl_fallback_error(" not in throw or not decline.endswith("return false; }"):
        raise ContractError("FAIL [pin S5]: the decline tail does not throw for a submitted deferred node and return "
                            "false otherwise")
    call_at = need_once(be, CALL_SITE, "S5", "the executor's call site")
    after = be[call_at + len(squash(CALL_SITE)) :].lstrip()
    if not after.startswith(squash(REENTRY_OPEN)):
        raise ContractError("FAIL [pin S5]: the re-entry does not follow the executor's call site")
    reentry = brace_text(after, len(squash(REENTRY_OPEN)) - 1, "S5")
    if (not reentry.endswith(squash(REENTRY_TAIL)) or squash(REENTRY_LATCH) not in reentry or
            "GGML_LOG_WARN(" not in reentry or REENTRY_TEXT not in reentry or "throw" in reentry):
        raise ContractError("FAIL [pin S5]: the re-entry does not warn once per cause and ask for the second pass")
    # S6
    gemm = block(lam, GEMM_FN, "S6", "the GEMM stage's tail")
    deferred = block(gemm, GEMM_DEFERRED, "S6", "the GEMM stage's deferred throw")
    if "throw ggml_sycl_fallback_error(" not in deferred or not gemm.endswith("return false; }"):
        raise ContractError("FAIL [pin S6]: the GEMM stage's tail does not throw for a deferred node")
    if gemm.find("drain_submitted_phases();") > gemm.find(squash(GEMM_DEFERRED)):
        raise ContractError("FAIL [pin S6]: the GEMM stage's deferred throw does not follow the drain")
    returns = lam[flag_at:].count("return false;")
    if returns != POST_SUBMIT_RETURNS:
        raise ContractError(f"FAIL [pin S6]: {returns} bare `return false;` after the first submission, not the "
                            f"{POST_SUBMIT_RETURNS} known ones: a new one must throw for a deferred node")
    # S7
    count = block(be, COUNT_FN, "S7", "the prompt count")
    if squash(COUNT_SKIP) not in count or count.find(squash(COUNT_SKIP)) > count.find("->add_prompt("):
        raise ContractError("FAIL [pin S7]: the prompt count counts a node's second pass")
    # S8
    test = s.sq["device_test"]
    if test.count(TEST_FUSED_NAME) != 1 or "GGML_TYPE_NVFP4, };" not in test:
        raise ContractError("FAIL [pin S8]: the device test lacks the plain NVFP4 case or the fused gate_up tensor")
    need_once(test, TEST_VARIANTS, "S8", "the NVFP4 fixtures")
    scales = TEST_SCALES.findall(test)
    if len(scales) != 2 or any(re.search(r"0x[fF][fF]\b", body) for _, body in scales):
        raise ContractError("FAIL [pin S8]: the special scales are not two cycles free of 0xFF")
    need_once(test, TEST_REENTRY, "S8", "the device test's re-entry count")
    need_once(test, TEST_REFUSAL, "S8", "the device test's refusal verdict")
    # S9
    if be.count(PLANNED_LOCAL_TRY) != 1:
        raise ContractError(f"FAIL [pin S9]: the planned-local batched try is called {be.count(PLANNED_LOCAL_TRY)} "
                            "times, not once")
    need_once(be, PLANNED_LOCAL_CALL, "S9", "the planned-local batched try behind the deferral")


def check(s: Sources) -> None:
    check_roles(s)
    check_seam(s)


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
            check(Sources(mutated))
        except ContractError as e:
            if f"[pin {pin}]" not in str(e):
                failures.append(f"{name}: caught by the wrong pin ({e})")
                print(f"  mutant {name}: caught by the wrong pin, not {pin}: {e}")
            else:
                print(f"  mutant {name}: caught by {pin}")
            return
        failures.append(f"{name}: SURVIVED")
        print(f"  mutant {name}: SURVIVED (pin {pin})")

    B, T, N, M, D = "backend", "tables", "names", "model", "device_test"
    # G1-G3: the role, its classifier and its switches.
    expect_fail("enum-unknown-reuses-three", "G1", B,
                [("    MOE_TENSOR_UNKNOWN = 4,\n", "    MOE_TENSOR_UNKNOWN = 3,\n")])
    expect_fail("role-count-fixed", "G1", B, [(ROLE_COUNT, "static constexpr int MOE_PROFILE_ROLE_COUNT = 4;")])
    expect_fail("role-array-fixed", "G1", B,
                [("    std::array<role_timing, MOE_PROFILE_ROLE_COUNT> role_timings{};",
                  "    std::array<role_timing, 4> role_timings{};")])
    expect_fail("classifier-gate-first", "G2", B,
                [('    if (strstr(name, "ffn_gate_up_exps")) {\n        return MOE_TENSOR_GATE_UP;\n    }\n', ""),
                 ('    if (strstr(name, "ffn_up_exps")) {\n',
                  '    if (strstr(name, "ffn_gate_up_exps")) {\n        return MOE_TENSOR_GATE_UP;\n    }\n'
                  '    if (strstr(name, "ffn_up_exps")) {\n')])
    expect_fail("name-switch-default", "G3", B,
                [('        case MOE_TENSOR_GATE_UP:\n            return "gate_up";\n',
                  '        default:\n            return "gate_up";\n')])
    expect_fail("registry-switch-default", "G3", B,
                [("                case MOE_TENSOR_UNKNOWN:\n                    break;\n            }\n"
                  "            n_groups_registered++;",
                  "                default:\n                    break;\n            }\n"
                  "            n_groups_registered++;")])
    expect_fail("registry-groups-fused-as-gate", "G3", B,
                [("                case MOE_TENSOR_GATE:\n                    grp.gate_key",
                  "                case MOE_TENSOR_GATE:\n                case MOE_TENSOR_GATE_UP:\n"
                  "                    grp.gate_key"),
                 ("                case MOE_TENSOR_GATE_UP:\n                    // The registry",
                  "                    // The registry")])
    # G4: the claim.
    expect_fail("staged-claim-admits-fused", "G4", B,
                [("!(role == MOE_TENSOR_GATE_UP && !host_staged)", "!(role == MOE_TENSOR_GATE_UP)")])
    expect_fail("claim-refuses-fused", "G4", B,
                [("if (!triplet_role && !(role == MOE_TENSOR_GATE_UP && !host_staged)) {", "if (!triplet_role) {")])
    # G5: the paired paths.
    expect_fail("bundle-admits-fused", "G5", B,
                [("    if (moe_kind != MOE_TENSOR_GATE && moe_kind != MOE_TENSOR_UP) {\n        return false;\n    }\n"
                  "    if (!ggml_sycl_xmx_tiled_enabled_for_device(device)) {",
                  "    if (moe_kind == MOE_TENSOR_DOWN) {\n        return false;\n    }\n"
                  "    if (!ggml_sycl_xmx_tiled_enabled_for_device(device)) {")])
    expect_fail("single-xmx-admits-fused", "G5", B,
                [("    const moe_tensor_type moe_kind = moe_classify_tensor(name);\n"
                  "    return moe_kind == MOE_TENSOR_GATE || moe_kind == MOE_TENSOR_UP;\n}\n\n"
                  "static bool ggml_sycl_moe_gateup_single_xmx_enabled(",
                  "    const moe_tensor_type moe_kind = moe_classify_tensor(name);\n"
                  "    return moe_kind != MOE_TENSOR_DOWN;\n}\n\n"
                  "static bool ggml_sycl_moe_gateup_single_xmx_enabled(")])
    expect_fail("glu-pairs-gate-alone", "G5", B,
                [("            if (!gate_out || !up_out) {", "            if (!gate_out) {")])
    # G6, G7: the ring's consumer and the loader.
    expect_fail("ring-consumer-skips-fused", "G6", B,
                [("        if (moe_classify_tensor(t.name) == MOE_TENSOR_UNKNOWN || t.ne[0] <= 0",
                  "        if (moe_classify_tensor(t.name) == MOE_TENSOR_UNKNOWN ||\n"
                  "            moe_classify_tensor(t.name) == MOE_TENSOR_GATE_UP || t.ne[0] <= 0")])
    expect_fail("names-lack-fused", "G7", N, [(', "ffn_gate_up_exps" }', " }")])
    expect_fail("loader-own-test", "G7", M,
                [("            if (!llama_sycl_is_moe_expert_weight_name(tensor.name)) {",
                  '            if (!tensor.name || !strstr(tensor.name, "ffn_gate_exps")) {')])
    # S1-S4: the seam, the suppression and the wrapper.
    expect_fail("seam-ignores-ring", "S1", T,
                [("    return prompt && executor_claims && ring_weight_slot_bytes > 0;",
                  "    return prompt && executor_claims;")])
    expect_fail("seam-defers-decode", "S2", B, [("moe_mmid_direct_defers_to_executor(\n            ne12 != 1,",
                                                 "moe_mmid_direct_defers_to_executor(\n            true,")])
    expect_fail("direct-gate-ignores-seam", "S2", B,
                [("direct_admission &&\n        !pp_direct_deferred) {", "direct_admission) {")])
    expect_fail("claim-not-suppressed", "S3", B,
                [("    if (weight != nullptr && weight == g_mmid_direct_reentry.suppressed_weight) {\n"
                  "        return false;\n    }\n"
                  "    return ggml_sycl_moe_pp_onednn_batched_claims_tensor(nullptr, weight",
                  "    return ggml_sycl_moe_pp_onednn_batched_claims_tensor(nullptr, weight")])
    expect_fail("wrapper-retries-unasked", "S4", B,
                [("    if (g_mmid_direct_reentry.requested != dst) {\n        return;\n    }\n", "")])
    expect_fail("wrapper-keeps-suppression", "S4", B,
                [("        ~suppression_end() { g_mmid_direct_reentry.suppressed_weight = nullptr; }",
                  "        ~suppression_end() {}")])
    expect_fail("dispatcher-skips-wrapper", "S4", B,
                [("            ggml_sycl_mul_mat_id_with_direct_reentry(ctx, dst);",
                  "            ggml_sycl_mul_mat_id(ctx, dst);")])
    # S5, S6: before and after the first submission.
    expect_fail("mark-before-scratch-decline", "S5", B,
                [("            batched_scratch_claim.mark_used();\n"
                  "            // Every decline from here on follows device work this dispatch submitted (see "
                  "pp_batched_submitted).\n            pp_batched_submitted = true;\n",
                  "            batched_scratch_claim.mark_used();\n"),
                 ("            if (!pp_moe_onednn_claim_scratch_slot(ctx.device, ring_depth, scratch_slot)) {",
                  "            pp_batched_submitted = true;\n"
                  "            if (!pp_moe_onednn_claim_scratch_slot(ctx.device, ring_depth, scratch_slot)) {")])
    expect_fail("queue-call-before-mark", "S5", B,
                [("            batched_scratch_claim.mark_used();\n",
                  "            stream->wait();\n            batched_scratch_claim.mark_used();\n")])
    expect_fail("decline-throws-unsubmitted", "S5", B,
                [("                if (pp_direct_deferred && pp_batched_submitted) {",
                  "                if (pp_direct_deferred) {")])
    expect_fail("decline-forgets-cause", "S5", B,
                [("                pp_batched_refusal  = refusal;\n", "")])
    expect_fail("reentry-throws", "S5", B,
                [("            g_mmid_direct_reentry.requested = dst;\n            return;\n        }",
                  '            throw ggml_sycl_fallback_error("declined");\n        }')])
    expect_fail("gemm-tail-returns-for-deferred", "S6", B,
                [("                if (pp_direct_deferred) {\n                    static thread_local std::string "
                  "deferred_gemm_msg;",
                  "                if (false) {\n"
                  "                    static thread_local std::string deferred_gemm_msg;")])
    expect_fail("bare-return-after-submission", "S6", B,
                [("                    if (!window_ready.empty()) {\n"
                  "                        stream->ext_oneapi_submit_barrier"
                  "(window_ready);",
                  "                    if (window_ready.size() > 4096) {\n                        return false;\n"
                  "                    }\n                    if (!window_ready.empty()) {\n"
                  "                        stream->ext_oneapi_submit_barrier(window_ready);")])
    # S7, S8: the prompt count and the device test.
    expect_fail("count-twice", "S7", B,
                [("    if (src0 == g_mmid_direct_reentry.suppressed_weight) {\n        return;\n    }\n", "")])
    expect_fail("test-writes-ff", "S8", D, [("{ 0x7e, 0xfe, 0x78, 0x7f }", "{ 0x7e, 0xff, 0x78, 0x7f }")])
    expect_fail("test-drops-fused", "S8", D,
                [("fixture_variant::LARGE_SCALES,\n                                                    "
                  "fixture_variant::FUSED_GATE_UP };",
                  "fixture_variant::LARGE_SCALES };")])
    expect_fail("test-ignores-reentry", "S8", D,
                [("unplanned != 0 || reentered != 0) {", "unplanned != 0) {")])
    # S9: the planned-local batched try.
    expect_fail("planned-local-takes-deferred", "S9", B,
                [("pp_planned_local_batched_enabled && !pp_direct_deferred &&", "pp_planned_local_batched_enabled &&")])
    expect_fail("planned-local-second-call", "S9", B,
                [("        if (try_pp_onednn_f16_batched()) {\n",
                  "        if (pp_planned_local_batched_enabled && try_pp_planned_local_batched()) {\n"
                  "            return;\n        }\n        if (try_pp_onednn_f16_batched()) {\n")])

    # The NOT APPLIED report must fire on an anchor the sources do not hold, or a stale mutant would pass unseen.
    try:
        mutate(B, [("this anchor is in no source", "x")])
        failures.append("stale-anchor-not-reported")
        print("  self-test: a stale mutant anchor is NOT reported")
    except MutantNotApplied:
        print("  self-test: a stale mutant anchor is reported")

    try:
        check(Sources(raws))
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
    raws = {name: path.read_text() for name, path in FILES.items()}
    if "--self-test" in argv:
        return self_test(raws)
    try:
        check(Sources(raws))
    except ContractError as e:
        print(e)
        return 1
    print("PASS: the fused gate_up role is its own role, and the direct MMID route steps aside for the batched "
          "executor only until the executor's first submission")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
