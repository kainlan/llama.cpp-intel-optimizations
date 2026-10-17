#!/usr/bin/env python3
"""Source contract for the q4_0 / q8_0 KV-cache flash attention on DPAS (llama.cpp-2szs).

ggml/src/ggml-sycl/fattn-dpas.cpp is Torchit1's XMX (DPAS) flash-attention kernel for a block-quantized KV cache, ported
into this fork. The fork's other flash-attention routes read only F16 (or FP8) K/V rows, so before the port a q4_0/q8_0
KV made FLASH_ATTN_EXT fall to ggml-cpu, which aborts on the f16 Q this build feeds it. These checks pin the properties
the port's correctness and safety rest on; none of them can be observed from a passing GPU run alone.

- The DPAS route is chosen before every other route in ggml_sycl_flash_attn_ext: after the operand pointers are resolved,
  before the D == 512 branch, the paged V2 block, the oneDNN plan and dispatch_ncols.
- supports_op admits a quantized K/V only through the DPAS predicate, called right after the GGML_SYCL_FLASH_ATTN_EXT
  master switch and before the F16/FP8 type checks. The signature of ggml_sycl_flash_attn_ext_supported and the
  supports_op case line stay byte-identical: the alloc-zone contract's mutation matrix plants from both.
- GGML_SYCL_FA_DPAS defaults OFF, is read once in one function of the whole tree, admits the kernel only when every SYCL
  GPU the process sees is a Battlemage card (supports_op has no device to ask), and has its catalog row. The walk is
  over total_gpu_count: the scheduler collapses device_count to 1 when several GPUs are visible, which would hide the
  iGPU from it.
- The kernel TU holds no allocation of its own and reads no tensor->data: pointers come from the resolved fattn_params,
  the split workspace from the unified-cache helper, retained until the launch's event, with no host wait.
- Dead KV cells (mask -inf) may hold NaN: the score uses fattn_mask_apply, never "+ mask", and V goes through
  fattn_v_zero_if_dead.
- A decode launch is observed as "dpas", which the positive allowlist classifies OTHER, so SYCL-graph AUTO stays off.
- The dispatch-debug line names DPAS; the Xe-core count is derived from this fork's nsm (compute units, not CU/16).
- Torchit1's authorship stays visible, and the TU can be compiled out (GGML_SYCL_FA_DPAS_NO_KERNELS) with the knob and
  the hook left in place.

Host-only text assertions, collected by pytest (llama_test_pytest). Checks run on comment-stripped, whitespace-squashed
text, so a comment quoting a call cannot satisfy a positive check.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL_DIR = ROOT / "ggml/src/ggml-sycl"

PATHS = {
    "fattn": SYCL_DIR / "fattn.cpp",
    "dpas": SYCL_DIR / "fattn-dpas.cpp",
    "dpas_hpp": SYCL_DIR / "fattn-dpas.hpp",
    "sycl": SYCL_DIR / "ggml-sycl.cpp",
    "common": SYCL_DIR / "common.hpp",
    "env_doc": ROOT / "docs/backend/sycl-env-vars.md",
}

# Trees searched for any other read of the knob.
CENSUS_DIRS = (SYCL_DIR, ROOT / "ggml/include", ROOT / "src", ROOT / "common", ROOT / "tools")
CENSUS_SUFFIXES = (".c", ".cpp", ".h", ".hpp")
KNOB_LITERAL = '"GGML_SYCL_FA_DPAS"'

SOURCE_SHA = "24a3a71896cdf1d21b625c13c63f98b05bdde8e7"

FLASH_SIG = "void ggml_sycl_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_sycl::sycl_tensor safe_dst) {"
SUPPORTED_SIG = "bool ggml_sycl_flash_attn_ext_supported(const ggml_tensor * dst) {"
PRED_SIG = "static bool ggml_sycl_fattn_dpas_supported(const ggml_tensor * dst) {"
KNOB_SIG = "static bool ggml_sycl_fa_dpas_enabled() {"
DISPATCH_SIG = ("static void ggml_sycl_fattn_dpas_dispatch(ggml_backend_sycl_context & ctx, const fattn_params & params, "
                "const ggml_tensor * dst) {")
CLASSIFY_SIG = "static kernel_family classify(const char * kernel) {"

# Exact-text anchors of scripts/check-sycl-alloc-zone-contract.py's mutation matrix (b1_tree and the case-line mutant).
ANCHOR_SUPPORTED = "bool ggml_sycl_flash_attn_ext_supported(const ggml_tensor * dst) {\n    // Enabled by default;"
ANCHOR_CASE = "            return ggml_sycl_flash_attn_ext_supported(op);\n"

USE_DPAS = "const bool use_dpas = ggml_sycl_fattn_dpas_supported(dst);"
HOOK = "if (use_dpas) { ggml_sycl_fattn_dpas_dispatch(ctx, params, dst); return; }"
SUPPORTS_CLAUSE = "if (ggml_sycl_fattn_dpas_supported(dst)) { return true; }"

COMMENT_OR_LITERAL = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|/\*.*?\*/|//[^\n]*', re.S)


def strip_comments(text: str) -> str:
    """Comments become one space; string and character literals are kept."""
    def repl(m: re.Match) -> str:
        s = m.group(0)
        return " " if s.startswith("/") else s
    return COMMENT_OR_LITERAL.sub(repl, text)


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def code(text: str) -> str:
    return squash(strip_comments(text))


def matching_brace(text: str, open_idx: int) -> int:
    """Brace match on comment-stripped text that still holds string and character literals."""
    assert text[open_idx] == "{"
    depth = 0
    i = open_idx
    while i < len(text):
        ch = text[i]
        if ch == '"' or ch == "'":
            j = i + 1
            while j < len(text) and text[j] != ch:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise AssertionError("unclosed brace")


def function_or_none(squashed: str, signature: str) -> str | None:
    sig = squash(signature)
    start = squashed.find(sig)
    if start < 0:
        return None
    brace = squashed.index("{", start + len(sig) - 1)
    return squashed[start:matching_brace(squashed, brace) + 1]


def load() -> dict[str, str]:
    return {k: (p.read_text(encoding="utf-8") if p.exists() else "") for k, p in PATHS.items()}


def census_files() -> dict[str, str]:
    out = {}
    for d in CENSUS_DIRS:
        for p in sorted(d.rglob("*")):
            if p.suffix in CENSUS_SUFFIXES and p.is_file():
                text = p.read_text(encoding="utf-8", errors="replace")
                if KNOB_LITERAL in text:
                    out[str(p.relative_to(ROOT))] = text
    return out


def hook_violations(src: dict[str, str]) -> list[str]:
    found = []
    body = function_or_none(code(src["fattn"]), FLASH_SIG)
    if body is None:
        return ["ggml_sycl_flash_attn_ext not found in fattn.cpp"]
    if body.count(squash(USE_DPAS)) != 1:
        found.append("ggml_sycl_flash_attn_ext does not compute use_dpas from ggml_sycl_fattn_dpas_supported exactly once")
    hook = body.find(squash(HOOK))
    if hook < 0 or body.count(squash(HOOK)) != 1:
        return found + ["the DPAS hook (if (use_dpas) { ggml_sycl_fattn_dpas_dispatch(...); return; }) is missing or not unique"]
    for label, needle in (("the params.dst fill", "params.dst = safe_dst.resolve_as<float>();"),
                          ("the graph pointer snapshot", "ctx.fa_graph_ptrs.push_back(snap);")):
        at = body.find(needle)
        if at < 0 or at > hook:
            found.append("the DPAS hook does not come after %s" % label)
    for label, needle in (("the D == 512 branch", "if (D == 512) {"),
                          ("a dispatch_ncols call", "dispatch_ncols<"),
                          ("a oneDNN call", "ggml_sycl_flash_attn_ext_onednn"),
                          ("a oneDNN route call", "ggml_sycl_fattn_onednn_")):
        for m in re.finditer(re.escape(needle), body):
            if m.start() < hook:
                found.append("%s comes before the DPAS hook" % label)
                break
    v2 = body.rfind("#if GGML_SYCL_FA_V2_ENABLED")
    if v2 >= 0 and v2 < hook:
        found.append("the paged V2 block comes before the DPAS hook")
    for operand in ("K", "V"):
        needle = ("GGML_ASSERT(use_dpas || %s->type == GGML_TYPE_F16 || ggml_sycl_type_is_fp8_e4m3(%s->type));"
                  % (operand, operand))
        if needle not in body:
            found.append("the %s type assert does not admit a quantized KV only through use_dpas" % operand)
    return found


def supports_violations(src: dict[str, str]) -> list[str]:
    found = []
    if src["fattn"].count(ANCHOR_SUPPORTED) != 1:
        found.append("the alloc-zone contract anchor 'bool ggml_sycl_flash_attn_ext_supported(const ggml_tensor * dst) {"
                     " // Enabled by default;' is not in fattn.cpp exactly once")
    if src["sycl"].count(ANCHOR_CASE) != 1:
        found.append("the supports_op case line 'return ggml_sycl_flash_attn_ext_supported(op);' is not in ggml-sycl.cpp "
                     "exactly once")
    body = function_or_none(code(src["fattn"]), SUPPORTED_SIG)
    if body is None:
        return found + ["ggml_sycl_flash_attn_ext_supported not found"]
    at = body.find(squash(SUPPORTS_CLAUSE))
    if at < 0:
        return found + ["ggml_sycl_flash_attn_ext_supported does not admit through ggml_sycl_fattn_dpas_supported"]
    master = body.find("if (!fa_ext_enabled) {")
    types = body.find("const bool kv_is_fp8")
    if master < 0 or master > at:
        found.append("the DPAS clause comes before the GGML_SYCL_FLASH_ATTN_EXT master switch")
    if types < 0 or types < at:
        found.append("the DPAS clause comes after the F16/FP8 K/V type checks, which decline a quantized KV first")
    return found


def knob_violations(src: dict[str, str], census: dict[str, str]) -> list[str]:
    found = []
    fattn = code(src["fattn"])
    knob = function_or_none(fattn, KNOB_SIG)
    if knob is None:
        return ["ggml_sycl_fa_dpas_enabled not found in fattn.cpp"]
    for label, needle in (
        ("read once into a static const", "static const bool enabled = ["),
        ("read GGML_SYCL_FA_DPAS", 'std::getenv("GGML_SYCL_FA_DPAS")'),
        ("default OFF (unset or 0 is off)", "if (env == nullptr || std::atoi(env) == 0) { return false; }"),
        ("walk every SYCL GPU the process sees (total_gpu_count: the scheduler collapses device_count to 1)",
         "ggml_sycl_info().total_gpu_count"),
        ("classify each device with family_from_device", "ggml_sycl::family_from_device("),
        ("require Battlemage", "ggml_sycl::sycl_gpu_family::ARC_BATTLEMAGE"),
        ("say once at WARN what it decided", "GGML_LOG_WARN("),
        ("compile out with the TU", "#ifdef GGML_SYCL_FA_DPAS_NO_KERNELS"),
    ):
        if needle not in knob:
            found.append("ggml_sycl_fa_dpas_enabled does not %s" % label)
    reads = 0
    for name, text in census.items():
        n = code(text).count(KNOB_LITERAL)
        reads += n
        if n and name != "ggml/src/ggml-sycl/fattn.cpp":
            found.append("%s names GGML_SYCL_FA_DPAS in code: the knob must be read in one function" % name)
    if reads != 1 or knob.count(KNOB_LITERAL) != 1:
        found.append("GGML_SYCL_FA_DPAS is named in code %d time(s) in the tree, %d of them in ggml_sycl_fa_dpas_enabled;"
                     " want exactly one, there" % (reads, knob.count(KNOB_LITERAL)))
    pred = function_or_none(fattn, PRED_SIG)
    if pred is None:
        found.append("ggml_sycl_fattn_dpas_supported not found in fattn.cpp")
    elif not pred[len(squash(PRED_SIG)):].lstrip().startswith("if (!ggml_sycl_fa_dpas_enabled()) { return false; }"):
        found.append("ggml_sycl_fattn_dpas_supported does not test the knob first")
    rows = [line for line in src["env_doc"].splitlines() if line.startswith("| `GGML_SYCL_FA_DPAS=1` |")]
    if len(rows) != 1:
        found.append("docs/backend/sycl-env-vars.md has %d catalog rows for GGML_SYCL_FA_DPAS=1; want one" % len(rows))
    elif "default **OFF**" not in rows[0]:
        found.append("the GGML_SYCL_FA_DPAS catalog row does not say default **OFF**")
    return found


def predicate_violations(src: dict[str, str]) -> list[str]:
    found = []
    pred = function_or_none(code(src["fattn"]), PRED_SIG)
    if pred is None:
        return ["ggml_sycl_fattn_dpas_supported not found in fattn.cpp"]
    for label, needle in (
        ("admit an F16 Q (this build casts Q to f16 for D != 512)",
         "if (Q->type != GGML_TYPE_F32 && Q->type != GGML_TYPE_F16) { return false; }"),
        ("require K and V of one quantized type", "if ((K->type != GGML_TYPE_Q4_0 && K->type != GGML_TYPE_Q8_0) || V->type != K->type) {"),
        ("decline paged and multi-sequence sources", "ggml_sycl_fattn_d512_has_paged_or_multiseq_sources(dst)"),
        ("take its head list from fattn-dpas.hpp", "ggml_sycl_fattn_dpas_head_ok("),
        ("keep K/V rows on 2-byte bounds (even strides and view offset: the kernel reads dwords)",
         "for (const ggml_tensor * t : { K, V }) { if (t->nb[0] != ggml_type_size(t->type) || t->nb[1] % 2 != 0 || "
         "t->nb[2] % 2 != 0 || t->nb[3] % 2 != 0 || t->view_offs % 2 != 0) { return false; } }"),
        ("decline ALiBi and logit softcap (the kernel applies neither)",
         "if (max_bias != 0.0f || logit_softcap != 0.0f) { return false; }"),
        ("take only a single-plane F16 mask (the kernel reads one mask plane with a half stride)",
         "if (mask != nullptr && (mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || "
         "mask->nb[0] != sizeof(sycl::half))) { return false; }"),
    ):
        if needle not in pred:
            found.append("ggml_sycl_fattn_dpas_supported does not %s" % label)
    for name in ("fattn", "dpas", "dpas_hpp"):
        if "FA_DPAS_MAX_Q" in code(src[name]):
            found.append("%s caps the query rows (FA_DPAS_MAX_Q): every ne01 runs on DPAS, a declined op aborts on the "
                         "CPU" % name)
    return found


def memory_violations(src: dict[str, str]) -> list[str]:
    found = []
    dpas = code(src["dpas"])
    if not dpas:
        return ["ggml/src/ggml-sycl/fattn-dpas.cpp is missing"]
    for needle in ("ggml_sycl_pool_alloc", "sycl::malloc", "malloc_device", "malloc_shared", "malloc_host", "sycl::free",
                   "->data", "wait_and_throw", ".wait()"):
        if needle in dpas:
            found.append("fattn-dpas.cpp contains %r: allocation goes through the unified cache in fattn.cpp, pointers"
                         " come from the resolved fattn_params, and nothing waits on the host" % needle)
    disp = function_or_none(code(src["fattn"]), DISPATCH_SIG)
    if disp is None:
        return found + ["ggml_sycl_fattn_dpas_dispatch not found in fattn.cpp"]
    for label, needle in (("allocate its workspace through the unified-cache split-workspace helper",
                           "ggml_sycl_fattn_xmx_v2_alloc_split_workspace_buffer("),
                          ("retain the workspace until the launch's event", "ggml_sycl::retain_handles_until_event("),
                          ("retain it for every split launch (nsplit > 1 is when the kernel writes it)",
                           "if (info.nsplit > 1) { ggml_sycl::retain_handles_until_event({ cache->dpas_parts_handle, "
                           "cache->dpas_meta_handle }, ev); }")):
        if needle not in disp:
            found.append("ggml_sycl_fattn_dpas_dispatch does not %s" % label)
    for needle in ("wait_and_throw", ".wait()"):
        if needle in disp:
            found.append("ggml_sycl_fattn_dpas_dispatch contains %r: the workspace is sized once, nothing waits" % needle)
    return found


def dead_cell_violations(src: dict[str, str]) -> list[str]:
    found = []
    dpas = code(src["dpas"])
    if "fattn_mask_apply(" not in dpas:
        found.append("fattn-dpas.cpp does not mask the score with fattn_mask_apply")
    if "fattn_v_zero_if_dead(" not in dpas:
        found.append("fattn-dpas.cpp does not pass the V scale through fattn_v_zero_if_dead")
    if re.search(r"\+\s*mv\b", dpas):
        found.append("fattn-dpas.cpp adds the mask to the score (NaN + -inf is NaN for a dead cell)")
    return found


def observe_violations(src: dict[str, str]) -> list[str]:
    found = []
    disp = function_or_none(code(src["fattn"]), DISPATCH_SIG)
    if disp is None:
        return ["ggml_sycl_fattn_dpas_dispatch not found in fattn.cpp"]
    for label, needle in (
        ("observe a decode launch as dpas", 'if (params.ne01 <= 1) { ctx.fa_decode_kernel_obs.observe("dpas"); }'),
        ("print a dispatch-debug line naming DPAS", '"[SYCL] fattn selected [%d/%d] DPAS '),
        ("gate it on GGML_SYCL_FA_DISPATCH_DEBUG", "ggml_sycl_fattn_dispatch_debug_env()"),
        ("bound it by GGML_SYCL_FA_DISPATCH_DEBUG_LIMIT", "ggml_sycl_fattn_dispatch_debug_limit()"),
        ("record the e2e attention stage", 'ggml_sycl::e2e_tg_profile_record(ggml_sycl::e2e_tg_stage::ATTENTION, "dpas"'),
        ("derive Xe cores from this fork's nsm (compute units)", "ggml_sycl_info().devices[ctx.device].nsm / 8"),
        ("compile out with the TU", "#ifdef GGML_SYCL_FA_DPAS_NO_KERNELS"),
    ):
        if needle not in disp:
            found.append("ggml_sycl_fattn_dpas_dispatch does not %s" % label)
    classify = function_or_none(code(src["common"]), CLASSIFY_SIG)
    if classify is None:
        found.append("fa_decode_kernel_observation::classify not found in common.hpp")
    elif '"dpas"' in classify:
        found.append("classify() names dpas: admitting DPAS to the graph-safe allowlist needs its own verification")
    return found


def port_violations(src: dict[str, str]) -> list[str]:
    found = []
    dpas_raw = src["dpas"]
    if not dpas_raw:
        return ["ggml/src/ggml-sycl/fattn-dpas.cpp is missing"]
    head = dpas_raw[:3000]
    for needle in ("Torchit1", "sycl-fa-q4-xmx", SOURCE_SHA, "MIT"):
        if needle not in head:
            found.append("the fattn-dpas.cpp header does not carry %r (attribution)" % needle)
    dpas = code(dpas_raw)
    if "#ifndef GGML_SYCL_FA_DPAS_NO_KERNELS" not in dpas:
        found.append("fattn-dpas.cpp's kernels are not guarded by GGML_SYCL_FA_DPAS_NO_KERNELS (the compile-out seam)")
    for needle in ("hw_info", "nsm * 2", "fattn_dpas_arch_ok", "g_ggml_sycl_fa_dpas"):
        if needle in dpas:
            found.append("fattn-dpas.cpp still uses upstream's %r; the fork gates in fattn.cpp" % needle)
    if "q_f16" not in dpas:
        found.append("fattn-dpas.cpp does not read an F16 Q (q_f16)")
    return found


CHECKS = (hook_violations, supports_violations, predicate_violations, memory_violations, dead_cell_violations,
          observe_violations, port_violations)


def all_violations(src: dict[str, str], census: dict[str, str]) -> list[str]:
    out = []
    for check in CHECKS:
        out += check(src)
    out += knob_violations(src, census)
    return out


def test_dpas_route_contract() -> None:
    violations = all_violations(load(), census_files())
    assert violations == [], "\n".join(["FAIL: " + v for v in violations])


def _replace(src: dict[str, str], key: str, old: str, new: str) -> dict[str, str]:
    assert src[key].count(old) >= 1, "mutation anchor %r not in %s" % (old, key)
    out = dict(src)
    out[key] = src[key].replace(old, new, 1)
    return out


def _append(src: dict[str, str], key: str, text: str) -> dict[str, str]:
    out = dict(src)
    out[key] = src[key] + "\n" + text + "\n"
    return out


def test_mutations_are_witnessed() -> None:
    src = load()
    census = census_files()
    hook_raw = "    if (use_dpas) {\n        ggml_sycl_fattn_dpas_dispatch(ctx, params, dst);\n        return;\n    }\n"
    no_hook = _replace(src, "fattn", hook_raw, "")
    clause_raw = "    if (ggml_sycl_fattn_dpas_supported(dst)) {\n        return true;\n    }\n"
    no_clause = _replace(src, "fattn", clause_raw, "")
    mutants = [
        ("hook removed", no_hook, hook_violations, "missing"),
        ("hook moved into the D == 512 branch",
         _replace(no_hook, "fattn", "    if (D == 512) {\n", "    if (D == 512) {\n" + hook_raw), hook_violations,
         "D == 512"),
        ("type assert no longer tied to use_dpas",
         _replace(src, "fattn", "GGML_ASSERT(use_dpas || K->type", "GGML_ASSERT(true || K->type"), hook_violations,
         "K type assert"),
        ("supports clause removed", no_clause, supports_violations, "does not admit"),
        ("supports clause after the type checks",
         _replace(no_clause, "fattn", "    if (!kv_is_f16 && !kv_is_fp8) {\n        return false;\n    }\n",
                  "    if (!kv_is_f16 && !kv_is_fp8) {\n        return false;\n    }\n" + clause_raw),
         supports_violations, "after the F16/FP8"),
        ("supports signature takes a device",
         _replace(src, "fattn", "bool ggml_sycl_flash_attn_ext_supported(const ggml_tensor * dst) {",
                  "bool ggml_sycl_flash_attn_ext_supported(const ggml_tensor * dst, int device) {"),
         supports_violations, "contract anchor"),
        ("case line passes a device",
         _replace(src, "sycl", ANCHOR_CASE, "            return ggml_sycl_flash_attn_ext_supported(op, device);\n"),
         supports_violations, "case line"),
        ("knob default ON",
         _replace(src, "fattn", "if (env == nullptr || std::atoi(env) == 0) {", "if (env != nullptr && std::atoi(env) == 0) {"),
         None, "default OFF"),
        ("knob read on every call",
         _replace(src, "fattn", "static const bool enabled = [] {\n        const char * env = std::getenv(\"GGML_SYCL_FA_DPAS\");",
                  "const bool enabled = [] {\n        const char * env = std::getenv(\"GGML_SYCL_FA_DPAS\");"),
         None, "static const"),
        ("Battlemage requirement dropped",
         _replace(src, "fattn", "ggml_sycl::sycl_gpu_family::ARC_BATTLEMAGE", "ggml_sycl::sycl_gpu_family::DATA_CENTER_MAX"),
         None, "require Battlemage"),
        ("knob walks only the scheduler's devices",
         _replace(src, "fattn", "ggml_sycl_info().total_gpu_count", "ggml_sycl_info().device_count"), None,
         "walk every SYCL GPU"),
        ("predicate tests the knob second",
         _replace(src, "fattn", "    if (!ggml_sycl_fa_dpas_enabled()) {\n        return false;\n    }\n", ""), None,
         "knob first"),
        ("catalog row says default ON",
         _replace(src, "env_doc", "| `GGML_SYCL_FA_DPAS=1` | Opt-in (default **OFF**)", "| `GGML_SYCL_FA_DPAS=1` | Opt-in (default ON)"),
         None, "default **OFF**"),
        ("F16 Q not admitted",
         _replace(src, "fattn", "if (Q->type != GGML_TYPE_F32 && Q->type != GGML_TYPE_F16) {", "if (Q->type != GGML_TYPE_F32) {"),
         predicate_violations, "F16 Q"),
        ("query-row cap reintroduced", _append(src, "dpas", "static constexpr int FA_DPAS_MAX_Q = 31;"),
         predicate_violations, "FA_DPAS_MAX_Q"),
        ("pool allocation in the kernel TU", _append(src, "dpas", "static ggml_sycl_pool_alloc<float> zz_parts;"),
         memory_violations, "ggml_sycl_pool_alloc"),
        ("odd K/V view offset admitted",
         _replace(src, "fattn", "t->nb[3] % 2 != 0 ||\n            t->view_offs % 2 != 0) {", "t->nb[3] % 2 != 0) {"),
         predicate_violations, "2-byte bounds"),
        ("ALiBi admitted",
         _replace(src, "fattn", "if (max_bias != 0.0f || logit_softcap != 0.0f) {", "if (logit_softcap != 0.0f) {"),
         predicate_violations, "ALiBi"),
        ("multi-plane mask admitted", _replace(src, "fattn", "mask->ne[2] != 1 || ", ""), predicate_violations,
         "single-plane F16 mask"),
        ("a two-way split launch left unretained",
         _replace(src, "fattn", "if (info.nsplit > 1) {", "if (info.nsplit > 2) {"), memory_violations,
         "every split launch"),
        ("tensor->data read in the kernel TU", _append(src, "dpas", "static const void * zz(const ggml_tensor * t) { return t->data; }"),
         memory_violations, "->data"),
        ("host wait in the dispatch",
         _replace(src, "fattn", "    ggml_sycl_fattn_dpas_launch_info info{};",
                  "    ctx.stream()->wait_and_throw();\n    ggml_sycl_fattn_dpas_launch_info info{};"),
         memory_violations, "wait_and_throw"),
        ("score adds the mask", _replace(src, "dpas", "fattn_mask_apply(", "zz_mask_apply("), dead_cell_violations,
         "fattn_mask_apply"),
        ("Torchit's mask add restored", _append(src, "dpas", "static float zz(float s, float mv) { return s + mv; }"),
         dead_cell_violations, "adds the mask"),
        ("V dead-cell zeroing dropped", _replace(src, "dpas", "fattn_v_zero_if_dead(", "zz_v_keep("), dead_cell_violations,
         "fattn_v_zero_if_dead"),
        ("decode launch not observed",
         _replace(src, "fattn", 'ctx.fa_decode_kernel_obs.observe("dpas");', 'ctx.fa_decode_kernel_obs.observe("esimd_f16");'),
         observe_violations, "observe a decode launch"),
        ("dpas admitted to the graph-safe allowlist",
         _replace(src, "common", 'if (std::strcmp(kernel, "esimd_f16") == 0) {',
                  'if (std::strcmp(kernel, "esimd_f16") == 0 || std::strcmp(kernel, "dpas") == 0) {'),
         observe_violations, "classify() names dpas"),
        ("debug line no longer names DPAS",
         _replace(src, "fattn", '"[SYCL] fattn selected [%d/%d] DPAS ', '"[SYCL] fattn selected [%d/%d] dp '),
         observe_violations, "naming DPAS"),
        ("upstream's Xe-core formula",
         _replace(src, "fattn", "ggml_sycl_info().devices[ctx.device].nsm / 8", "ggml_sycl_info().devices[ctx.device].nsm * 2"),
         observe_violations, "Xe cores"),
        ("attribution sha dropped", _replace(src, "dpas", SOURCE_SHA, "0000000"), port_violations, SOURCE_SHA),
        ("compile-out seam dropped", _replace(src, "dpas", "#ifndef GGML_SYCL_FA_DPAS_NO_KERNELS", "#if 1"),
         port_violations, "compile-out seam"),
        ("upstream arch gate restored", _append(src, "dpas", "static bool zz(int d) { return ggml_sycl_info().devices[d].hw_info.arch; }"),
         port_violations, "hw_info"),
    ]
    for desc, mutated, check, expect in mutants:
        assert mutated != src, "mutation %r did not change the sources" % desc
        got = check(mutated) if check else knob_violations(mutated, census)
        assert any(expect in v for v in got), "mutation %r was not witnessed (want %r): %s" % (desc, expect, got)

    # A second read of the knob anywhere in the tree.
    planted = dict(census)
    planted["ggml/src/ggml-sycl/ggml-sycl.cpp"] = 'static const bool zz = std::getenv("GGML_SYCL_FA_DPAS") != nullptr;\n'
    got = knob_violations(src, planted)
    assert any("ggml-sycl.cpp names GGML_SYCL_FA_DPAS" in v for v in got), "a second knob read was not witnessed: %s" % got


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
