"""Source contract for llama.cpp-w013: the single-device KV context room is cost-based (owner ruling 2026-10-10).

Host-only: reads sources, builds nothing, loads no model and touches no device.

Why: with the room always held, GPT-OSS 20B on the B50 at its full requested context decoded at 22.3 t/s against
25.35 t/s with eleven layers' attention demoted to the host, while Qwen3.8 IQ3_XXS on the B70 at -c 4096 decoded at
10.49 t/s with the room against 9.70 without it. The sign of the trade depends on the model, so the planner weighs it.

The fix this gate pins:
  - hold_kv_context_room decides each device layer's room on its own, in the order the runtime demotes KV layers
    (plan_runtime_kv_demotion: full attention latest first, then SWA latest first), so a refused room is what that
    demotion would move;
  - a layer's room is held iff A > B, per decode token:
      A = host attention    = the layer's KV at F cells / host_attn_gbps, F = n_ctx_context x ctx_fill_pct / 100
      B = displaced experts = the layer's room x n_expert_used / n_expert / cpu_expert_gbps
    and a dense model (no expert to displace) holds its room as before;
  - with no requested context (n_ctx_context 0) it returns before weighing anything;
  - the three parameters are placement_kv_info fields, filled once where the backend copies the inventory in, from
    GGML_SYCL_PLAN_CTX_FILL_PCT / GGML_SYCL_PLAN_HOST_ATTN_GBPS / GGML_SYCL_PLAN_CPU_EXPERT_GBPS (defaults 50 / 30 /
    40, a nonsense value keeps the default, a fill above 100 is 100), and catalogued in docs/backend/sycl-env-vars.md;
  - under this cost every full-attention layer decides alike (its KV per cell cancels out of A/B), so only a SWA
    layer can differ; the room's line therefore states the decision per attention class ("held all N full-attention
    layer(s)" / "refused all N full-attention layer(s) by cost", and the same for SWA) and gives the deciding layer's A
    and B in microseconds per token with the three parameters; a fully refused room logs one WARN saying so;
  - the requested context still never becomes the planning n_ctx (llama.cpp-fkpg c-sc3u).

Every claim is checked on COMMENT-STRIPPED, whitespace-normalized text with adjacent string literals joined, and has a
mutant that must make it fail.

Pytest-style (module-level test_* functions): register with llama_test_pytest.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_LEXEME_RE = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'",
    re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def norm(text: str) -> str:
    """Comment-stripped, whitespace-canonical text with adjacent string literals joined, so neither a clang-format
    re-wrap nor a re-split format string can turn a claim red."""
    t = re.sub(r"\s+", " ", strip_comments(text))
    t = re.sub(r"\( ", "(", t)
    t = re.sub(r" \)", ")", t)
    t = re.sub(r" ,", ",", t)
    t = re.sub(r",(?! )", ", ", t)
    t = re.sub(r'(?<!\\)" "', "", t)
    return t


def read(rel: str) -> str:
    return (ROOT / rel).read_text()


CACHE = read("ggml/src/ggml-sycl/unified-cache.cpp")
CACHE_HPP = read("ggml/src/ggml-sycl/unified-cache.hpp")
SYCL = read("ggml/src/ggml-sycl/ggml-sycl.cpp")
DOC = read("docs/backend/sycl-env-vars.md")


def body(text: str, signature: str) -> str:
    """The brace-balanced body of the first definition whose text starts with `signature` (string literals are
    skipped while balancing). Empty when there is none."""
    at = text.find(signature)
    if at < 0:
        return ""
    open_at = text.find("{", at + len(signature))
    if open_at < 0:
        return ""
    depth = 0
    i = open_at
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "\"'":
            j = i + 1
            while j < n and text[j] != ch:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_at:i + 1]
        i += 1
    return ""


def ordered(text: str, *needles: str) -> bool:
    """Every needle is present, each after the one before it."""
    at = 0
    for needle in needles:
        at = text.find(needle, at)
        if at < 0:
            return False
        at += len(needle)
    return True


PLAN_SIG = "placement_plan compute_placement_plan(const std::vector<placement_tensor_info> & tensor_inventory,"
MULTI_PLAN_SIG = ("placement_plan compute_multi_device_plan(const std::vector<device_budget> & device_budgets, "
                  "const std::vector<placement_tensor_info> & tensor_inventory,")
ROOM_SIG = "static kv_context_room hold_kv_context_room(placement_plan & plan,"
WEIGH_SIG = "static kv_context_room_cost weigh_kv_context_room(const placement_kv_info & kv_info,"
ROOM_LOG_SIG = "static void log_kv_context_room(const kv_context_room & room,"
FILL_SIG = ("static void populate_inventory_globals(ggml_backend_sycl_context * ctx, "
            "const ggml_sycl_tensor_inventory * inventory)")
PARAM_SIG = "static double ggml_sycl_plan_cost_param(const char * name, double def, double max)"

ROOM_CALL = "room = hold_kv_context_room(plan, kv_info, device_id, n_experts, remaining);"
COPY_CTX = "g_placement_kv_info.n_ctx_context = inventory->n_ctx_context;"
NO_DOUBLE_MAX = "std::numeric_limits<double>::max()"

# name -> (placement_kv_info field, default constant, upper bound at the fill site, unit, source the doc row cites)
PARAMS = {
    "GGML_SYCL_PLAN_CTX_FILL_PCT": ("ctx_fill_pct", "PLACEMENT_CTX_FILL_PCT_DEFAULT", "100.0", "%",
                                    "llama.cpp-w013"),
    "GGML_SYCL_PLAN_HOST_ATTN_GBPS": ("host_attn_gbps", "PLACEMENT_HOST_ATTN_GBPS_DEFAULT", NO_DOUBLE_MAX, "GB/s",
                                      "unified-memory-manager-design.md"),
    "GGML_SYCL_PLAN_CPU_EXPERT_GBPS": ("cpu_expert_gbps", "PLACEMENT_CPU_EXPERT_GBPS_DEFAULT", NO_DOUBLE_MAX, "GB/s",
                                       "llama.cpp-8bgi"),
}


def fill(name: str) -> str:
    field, const, bound, _, _ = PARAMS[name]
    return f'g_placement_kv_info.{field} = ggml_sycl_plan_cost_param("{name}", ggml_sycl::{const}, {bound});'


def header_default(hpp: str, const: str) -> float | None:
    m = re.search(rf"constexpr double {const} = ([0-9.]+);", norm(hpp))
    return float(m.group(1)) if m else None


# ---- the parameters -------------------------------------------------------------------------------------------------


def claim_params_are_kv_info_fields(hpp: str) -> bool:
    """Each parameter is a placement_kv_info field defaulting to its named constant, so a test sets it directly and
    an unset field behaves as the default run does."""
    n = norm(hpp)
    at = n.find("struct placement_kv_info {")
    kv = body(n[at:], "struct placement_kv_info") if at >= 0 else ""
    return bool(kv) and all(f"double {field} = {const};" in kv for field, const, _, _, _ in PARAMS.values())


def claim_params_are_documented(doc: str, hpp: str) -> bool:
    """Each variable has a row in the env catalog giving its default (equal to the header's constant), its unit, and
    the measurement or ruling the default comes from."""
    for name, (_, const, _, unit, source) in PARAMS.items():
        rows = re.findall(rf"^\| `{name}=<[^>]*>` \| \*\*([0-9.]+)\*\* \|(.*)$", doc, re.MULTILINE)
        default = header_default(hpp, const)
        if len(rows) != 1 or default is None or float(rows[0][0]) != default:
            return False
        if unit not in rows[0][1] or source not in rows[0][1] or "us/token" not in rows[0][1]:
            return False
    return True


def claim_fill_site_reads_each_once(sycl: str, cache: str, hpp: str) -> bool:
    """The backend reads each variable once, where it copies the inventory into its KV inputs, right after the
    requested context; nothing else reads them."""
    s = norm(sycl)
    site = body(s, FILL_SIG)
    others = norm(cache) + norm(hpp)
    if not site:
        return False
    for name in PARAMS:
        if s.count(f'"{name}"') != 1 or name in others:
            return False
        if not ordered(site, COPY_CTX, fill(name)):
            return False
    return True


PARSE_REJECT = "if (end == env || !std::isfinite(value) || value <= 0.0) { return def; }"
PARSE_CLAMP = "return std::min(value, max);"


def claim_nonsense_keeps_the_default(sycl: str) -> bool:
    """An unset, unparsable, non-finite, zero or negative value keeps the default; a value above the bound is the
    bound (the fill is a percent, so 100)."""
    b = body(norm(sycl), PARAM_SIG)
    return (bool(b) and ordered(b, 'const char * env = std::getenv(name);', "if (env == nullptr) { return def; }",
                                "const double value = std::strtod(env, &end);", PARSE_REJECT, PARSE_CLAMP))


# ---- the decision ---------------------------------------------------------------------------------------------------


FILLED = ("const uint32_t filled = static_cast<uint32_t>(static_cast<double>(kv_info.n_ctx_context) * "
          "std::clamp(kv_info.ctx_fill_pct, 0.0, 100.0) / 100.0);")
HOST_READ = "const size_t host_read = kv_info.kv_bytes_for_layer_at(static_cast<uint32_t>(layer_id), filled);"
MOE = "const bool moe = n_experts > 0 && kv_info.n_expert_used > 0;"
P_HIT = "const double p_hit = moe ? std::min(1.0, static_cast<double>(kv_info.n_expert_used) / n_experts) : 0.0;"
A_US = "cost.host_attn_us = static_cast<double>(host_read) / (kv_info.host_attn_gbps * 1e3);"
B_US = "cost.expert_us = static_cast<double>(extra) * p_hit / (kv_info.cpu_expert_gbps * 1e3);"
HOLD_IFF = "cost.held = !moe || cost.host_attn_us > cost.expert_us;"


def claim_room_weighs_attention_against_experts(cache: str) -> bool:
    """A = the layer's KV at the expected fill, read at the host-attention rate; B = the layer's room times the chance
    a token uses a displaced expert, read at the CPU expert rate; held iff A > B, or always for a dense model."""
    b = body(norm(cache), WEIGH_SIG)
    return bool(b) and ordered(b, FILLED, HOST_READ, MOE, P_HIT, A_US, B_US, HOLD_IFF)


WEIGH_CALL = "const kv_context_room_cost cost = weigh_kv_context_room(kv_info, layer_id, extra, n_experts);"
COUNT_HELD = ("if (cost.held) { room.wanted += std::min(extra, SIZE_MAX - room.wanted); room.n_held++; } else { "
              "room.refused += std::min(extra, SIZE_MAX - room.refused); room.n_refused++; }")
DECIDER = "if (room.decision.layer_id < 0 || (room.decision.held && !cost.held)) { room.decision = cost; }"
BOUND = ("room.held = std::min(room.wanted, remaining); remaining -= room.held; "
         "plan.kv_context_reserve_bytes = room.held; plan.vram_bytes += room.held;")


def claim_room_holds_only_the_layers_that_pass(cache: str) -> bool:
    """Only a layer whose room passes the cost test adds to what the planner wants; the existing bound (no more than
    is left) applies to that sum; the deciding layer is the first refused one, else the first held."""
    b = body(norm(cache), ROOM_SIG)
    return (bool(b) and ordered(b, WEIGH_CALL, COUNT_HELD, DECIDER, BOUND) and b.count("room.wanted +=") == 1
            and ROOM_CALL in body(norm(cache), PLAN_SIG))


SWA_COUNT = "if (swa_pass) { room.n_swa_layers++; room.n_swa_held += cost.held ? 1 : 0; }"


def claim_room_counts_each_attention_class(cache: str) -> bool:
    """The SWA layers are counted apart, so the line can say what each attention class did."""
    b = body(norm(cache), ROOM_SIG)
    return bool(b) and ordered(b, WEIGH_CALL, "room.n_layers++;", SWA_COUNT, COUNT_HELD)


DEVICE_LAYERS = "if (layer_id >= 0 && owner == device_id) { layers.push_back(layer_id); }"
SORT = "std::sort(layers.begin(), layers.end());"
PASSES = "for (const bool swa_pass : { false, true }) {"
LATEST_FIRST = "for (auto it = layers.rbegin(); it != layers.rend(); ++it) {"
PASS_FILTER = "if (kv_info.is_swa_layer(layer_id) != swa_pass) { continue; }"


def claim_decision_iterates_latest_first(cache: str) -> bool:
    """The device layers are weighed in the runtime's demotion order: full attention latest first, then SWA latest
    first."""
    b = body(norm(cache), ROOM_SIG)
    return bool(b) and ordered(b, DEVICE_LAYERS, SORT, PASSES, LATEST_FIRST, PASS_FILTER, WEIGH_CALL)


NO_REQUEST = "kv_context_room room; if (kv_info.n_ctx_context == 0) { return room; }"


def claim_no_request_returns_before_any_cost(cache: str) -> bool:
    """With no requested context the function returns before it looks at a layer or weighs anything."""
    b = body(norm(cache), ROOM_SIG)
    return (bool(b) and b.startswith("{ " + NO_REQUEST)
            and b.find(NO_REQUEST) < b.find("weigh_kv_context_room(") and b.find(NO_REQUEST) < b.find("for ("))


PLANNER_N_CTX = "plan.planner_n_ctx = kv_info.n_ctx;"
COPY_N_CTX = "g_placement_kv_info.n_ctx = inventory->n_ctx;"


def claim_context_never_becomes_the_planning_n_ctx(cache: str, hpp: str, sycl: str) -> bool:
    """The requested context sizes the room and its cost only: no assignment carries it into an n_ctx (the planner's,
    kv_info's, the oneDNN Graph-scratch floor's), so the load's planning shape keeps its own (llama.cpp-fkpg). The
    needle takes any identifier that ends in n_ctx, so planner_n_ctx is covered in both plan builders."""
    texts = [norm(t) for t in (cache, hpp, sycl)]
    leak = re.compile(r"\b\w*n_ctx\s*=[^;=]*\bn_ctx_context\b")
    return (all(leak.search(t) is None for t in texts) and PLANNER_N_CTX in body(texts[0], PLAN_SIG)
            and COPY_N_CTX in body(texts[2], FILL_SIG))


# ---- the line -------------------------------------------------------------------------------------------------------


SUMMARY_SIG = "static std::string kv_context_room_class_summary(size_t n_held, size_t n_layers, const char * kind)"
SUMMARY_FORMS = ('"no %s layer with a room", kind', '"held all %zu %s layer(s)", n_layers, kind',
                 '"refused all %zu %s layer(s) by cost", n_layers, kind',
                 '"held %zu of %zu %s layer(s), refused %zu by cost", n_held, n_layers, kind, n_layers - n_held')
FULL_SUMMARY = ('const std::string full = kv_context_room_class_summary(room.n_held - room.n_swa_held, '
                'room.n_layers - room.n_swa_layers, "full-attention");')
SWA_SUMMARY = 'const std::string swa = kv_context_room_class_summary(room.n_swa_held, room.n_swa_layers, "SWA");'
DECISION = "[PLACEMENT] KV context room on device %d: %s, %s"
DECISION_ARGS = "device_id, full.c_str(), swa.c_str(), room.refused / mb,"
COSTS = ("host attention %.1f us/token vs displaced experts %.1f us/token (fill %.4g%%, host-attn %.4g GB/s, "
         "cpu-expert %.4g GB/s)")
COST_ARGS = ("room.decision.host_attn_us, room.decision.expert_us, kv_info.ctx_fill_pct, kv_info.host_attn_gbps, "
             "kv_info.cpu_expert_gbps")
TAIL = "n_ctx_context is the requested n_ctx.\\n"
ALL_REFUSED = "if (room.n_held == 0) { GGML_LOG_WARN("
ALL_REFUSED_SAYS = ("every layer's room was refused by cost", "the runtime re-places the overflow at context creation")
WARN = "const bool warn = room.displaced_bytes() > 0 || room.held < room.wanted || room.n_refused > 0;"


def claim_room_line_states_the_decision(cache: str) -> bool:
    """Both lines say, per attention class, whether its layers were held or refused by cost ("held all N" or "refused
    all N" for the full-attention layers, which decide alike), and give the deciding layer's A and B in microseconds
    per token with the three parameters; a fully refused room is one WARN saying every room was refused and that the
    runtime re-places the overflow; otherwise the line is a WARN whenever the room cost experts, could not hold all it
    wanted, or refused a layer."""
    n = norm(cache)
    b = body(n, ROOM_LOG_SIG)
    summary = body(n, SUMMARY_SIG)
    if not b or not summary or "if (room.n_layers == 0) { return; }" not in b:
        return False
    if not ordered(summary, *SUMMARY_FORMS) or not ordered(b, FULL_SUMMARY, SWA_SUMMARY, ALL_REFUSED):
        return False
    at = b.find(ALL_REFUSED)
    if at < 0:
        return False
    refused = body(b[at:], "if (room.n_held == 0)")
    rest = b[b.find(refused, at) + len(refused):] if refused else ""
    return (bool(rest)
            and all(s in refused
                    for s in (DECISION, DECISION_ARGS, COSTS, COST_ARGS, TAIL, "return;") + ALL_REFUSED_SAYS)
            and all(s in rest for s in (DECISION, DECISION_ARGS, COSTS, COST_ARGS, TAIL, WARN))
            and "ggml_log_internal(warn ? GGML_LOG_LEVEL_WARN : GGML_LOG_LEVEL_INFO," in rest)


def test_params_are_kv_info_fields():
    assert claim_params_are_kv_info_fields(CACHE_HPP)


def test_params_are_documented():
    assert claim_params_are_documented(DOC, CACHE_HPP)


def test_fill_site_reads_each_once():
    assert claim_fill_site_reads_each_once(SYCL, CACHE, CACHE_HPP)


def test_nonsense_keeps_the_default():
    assert claim_nonsense_keeps_the_default(SYCL)


def test_room_weighs_attention_against_experts():
    assert claim_room_weighs_attention_against_experts(CACHE)


def test_room_holds_only_the_layers_that_pass():
    assert claim_room_holds_only_the_layers_that_pass(CACHE)


def test_decision_iterates_latest_first():
    assert claim_decision_iterates_latest_first(CACHE)


def test_no_request_returns_before_any_cost():
    assert claim_no_request_returns_before_any_cost(CACHE)


def test_context_never_becomes_the_planning_n_ctx():
    assert claim_context_never_becomes_the_planning_n_ctx(CACHE, CACHE_HPP, SYCL)


def test_room_line_states_the_decision():
    assert claim_room_line_states_the_decision(CACHE)


def test_room_counts_each_attention_class():
    assert claim_room_counts_each_attention_class(CACHE)


# ---- mutants: each must turn its claim red --------------------------------------------------------------------------


def _once(text: str, old: str, new: str) -> str:
    """`text` with the one occurrence of `old` (matched on normalized text) replaced by `new`."""
    n = norm(text)
    assert n.count(old) == 1, f"mutant anchor must match exactly once: {old!r} x{n.count(old)}"
    return n.replace(old, new, 1)


def _doc_once(old: str, new: str) -> str:
    assert DOC.count(old) == 1, f"doc mutant anchor must match exactly once: {old!r} x{DOC.count(old)}"
    return DOC.replace(old, new, 1)


def test_mutant_param_not_a_field_fails():
    assert not claim_params_are_kv_info_fields(
        _once(CACHE_HPP, "double host_attn_gbps = PLACEMENT_HOST_ATTN_GBPS_DEFAULT;", "double host_attn_gbps = 0;"))


def test_mutant_doc_default_differs_fails():
    """The catalog says 40 while the code defaults to 50."""
    assert not claim_params_are_documented(
        _doc_once("| `GGML_SYCL_PLAN_CTX_FILL_PCT=<pct>` | **50** |", "| `GGML_SYCL_PLAN_CTX_FILL_PCT=<pct>` | **40** |"),
        CACHE_HPP)


def test_mutant_header_default_differs_fails():
    assert not claim_params_are_documented(
        DOC, _once(CACHE_HPP, "constexpr double PLACEMENT_CPU_EXPERT_GBPS_DEFAULT = 40.0;",
                   "constexpr double PLACEMENT_CPU_EXPERT_GBPS_DEFAULT = 48.0;"))


def test_mutant_doc_row_missing_fails():
    rows = [line for line in DOC.splitlines() if line.startswith("| `GGML_SYCL_PLAN_HOST_ATTN_GBPS=")]
    assert len(rows) == 1
    assert not claim_params_are_documented(_doc_once(rows[0] + "\n", ""), CACHE_HPP)


def test_mutant_doc_row_without_source_fails():
    rows = [line for line in DOC.splitlines() if line.startswith("| `GGML_SYCL_PLAN_CPU_EXPERT_GBPS=")]
    assert len(rows) == 1 and rows[0].count("llama.cpp-8bgi") >= 1
    assert not claim_params_are_documented(_doc_once(rows[0], rows[0].replace("llama.cpp-8bgi", "a measurement")),
                                           CACHE_HPP)


def test_mutant_second_reader_fails():
    """The planner reads a variable itself as well: two sources for one parameter."""
    mutant = CACHE + '\nstatic const char * g_w013_mutant = std::getenv("GGML_SYCL_PLAN_HOST_ATTN_GBPS");\n'
    assert not claim_fill_site_reads_each_once(SYCL, mutant, CACHE_HPP)


def test_mutant_fill_site_drops_a_param_fails():
    assert not claim_fill_site_reads_each_once(
        _once(SYCL, fill("GGML_SYCL_PLAN_CPU_EXPERT_GBPS"), ""), CACHE, CACHE_HPP)


def test_mutant_fill_unclamped_fails():
    """A fill above 100% would expect more cells filled than the context has."""
    assert not claim_fill_site_reads_each_once(
        _once(SYCL, fill("GGML_SYCL_PLAN_CTX_FILL_PCT"),
              fill("GGML_SYCL_PLAN_CTX_FILL_PCT").replace("100.0);", NO_DOUBLE_MAX + ");")), CACHE, CACHE_HPP)


def test_mutant_param_read_before_the_context_fails():
    """Read before the requested context is copied: still one read, but not at the site that fills the room's
    inputs together."""
    n = _once(SYCL, fill("GGML_SYCL_PLAN_HOST_ATTN_GBPS"), "")
    assert n.count(COPY_CTX) == 1
    mutant = n.replace(COPY_CTX, fill("GGML_SYCL_PLAN_HOST_ATTN_GBPS") + " " + COPY_CTX, 1)
    assert not claim_fill_site_reads_each_once(mutant, CACHE, CACHE_HPP)


def test_mutant_zero_accepted_fails():
    """A zero bandwidth would divide by zero; a zero fill would refuse every MoE room."""
    assert not claim_nonsense_keeps_the_default(_param_mutant("value <= 0.0) { return def; }",
                                                              "value < 0.0) { return def; }"))


def _param_mutant(old: str, new: str) -> str:
    """ggml-sycl.cpp with `old` replaced by `new` inside ggml_sycl_plan_cost_param only (other parsers in the file
    share its idioms)."""
    n = norm(SYCL)
    b = body(n, PARAM_SIG)
    assert b.count(old) == 1, f"mutant anchor must match exactly once in the parser: {old!r} x{b.count(old)}"
    return n.replace(b, b.replace(old, new, 1), 1)


def test_mutant_garbage_accepted_fails():
    """An unparsable value read as 0 by strtod and then kept as 0: the reject must test that strtod parsed anything."""
    assert not claim_nonsense_keeps_the_default(_param_mutant("end == env || ", ""))


def test_mutant_no_bound_fails():
    assert not claim_nonsense_keeps_the_default(_param_mutant(PARSE_CLAMP, "return value;"))


def test_mutant_host_read_undivided_fails():
    assert not claim_room_weighs_attention_against_experts(
        _once(CACHE, A_US, "cost.host_attn_us = static_cast<double>(host_read);"))


def test_mutant_expert_read_undivided_fails():
    assert not claim_room_weighs_attention_against_experts(
        _once(CACHE, B_US, "cost.expert_us = static_cast<double>(extra) * p_hit;"))


def test_mutant_comparison_flipped_fails():
    assert not claim_room_weighs_attention_against_experts(
        _once(CACHE, "cost.host_attn_us > cost.expert_us;", "cost.host_attn_us < cost.expert_us;"))


def test_mutant_dense_weighed_fails():
    """A dense model weighed like a MoE one: at a zero expected fill its room, which displaces nothing, is refused."""
    assert not claim_room_weighs_attention_against_experts(
        _once(CACHE, HOLD_IFF, "cost.held = cost.host_attn_us > cost.expert_us;"))


def test_mutant_every_expert_hit_fails():
    """p_hit 1: every displaced expert read by every token, whatever n_expert_used / n_expert says."""
    assert not claim_room_weighs_attention_against_experts(_once(CACHE, P_HIT, "const double p_hit = 1.0;"))


def test_mutant_attention_at_the_whole_context_fails():
    """The host read at the requested context instead of the expected fill: the fill parameter would do nothing."""
    assert not claim_room_weighs_attention_against_experts(
        _once(CACHE, HOST_READ, HOST_READ.replace(", filled);", ", kv_info.n_ctx_context);")))


def test_mutant_refused_rooms_still_held_fails():
    assert not claim_room_holds_only_the_layers_that_pass(
        _once(CACHE, COUNT_HELD, "room.wanted += std::min(extra, SIZE_MAX - room.wanted); room.n_held++;"))


def test_mutant_room_unbounded_fails():
    assert not claim_room_holds_only_the_layers_that_pass(
        _once(CACHE, "room.held = std::min(room.wanted, remaining);", "room.held = room.wanted;"))


def test_mutant_decider_is_the_last_layer_fails():
    assert not claim_room_holds_only_the_layers_that_pass(_once(CACHE, DECIDER, "room.decision = cost;"))


def test_mutant_planner_ignores_n_expert_fails():
    """The call passes no expert count: the weigh could not tell a MoE model from a dense one."""
    assert not claim_room_holds_only_the_layers_that_pass(
        _once(CACHE, ROOM_CALL, "room = hold_kv_context_room(plan, kv_info, device_id, 0, remaining);"))


def test_mutant_earliest_first_fails():
    assert not claim_decision_iterates_latest_first(
        _once(CACHE, LATEST_FIRST, "for (auto it = layers.begin(); it != layers.end(); ++it) {"))


def test_mutant_swa_first_fails():
    assert not claim_decision_iterates_latest_first(_once(CACHE, PASSES, "for (const bool swa_pass : { true, false }) {"))


def test_mutant_unsorted_fails():
    """The layers in the hash map's order: no order at all."""
    assert not claim_decision_iterates_latest_first(_once(CACHE, SORT, ""))


def test_mutant_no_early_return_fails():
    assert not claim_no_request_returns_before_any_cost(
        _once(CACHE, NO_REQUEST, "kv_context_room room;"))


def test_mutant_planner_n_ctx_from_the_context_fails():
    """The single-device plan's planning n_ctx taken from the request (the multi-device builder sets it too, so the
    mutant edits the single-device body only)."""
    n = norm(CACHE)
    b = body(n, PLAN_SIG)
    assert b.count(PLANNER_N_CTX) == 1
    mutant = n.replace(b, b.replace(PLANNER_N_CTX, "plan.planner_n_ctx = kv_info.n_ctx_context;"), 1)
    assert not claim_context_never_becomes_the_planning_n_ctx(mutant, CACHE_HPP, SYCL)


def test_mutant_multi_device_planner_n_ctx_from_the_context_fails():
    """The multi-device plan's planning n_ctx taken from the request: the leak needle must reach an identifier that
    only ends in n_ctx, since nothing else pins this site."""
    n = norm(CACHE)
    b = body(n, MULTI_PLAN_SIG)
    assert b.count(PLANNER_N_CTX) == 1
    mutant = n.replace(b, b.replace(PLANNER_N_CTX, "plan.planner_n_ctx = kv_info.n_ctx_context;"), 1)
    assert not claim_context_never_becomes_the_planning_n_ctx(mutant, CACHE_HPP, SYCL)


def test_mutant_kv_info_n_ctx_from_the_context_fails():
    assert not claim_context_never_becomes_the_planning_n_ctx(
        CACHE, CACHE_HPP, _once(SYCL, COPY_N_CTX, "g_placement_kv_info.n_ctx = inventory->n_ctx_context;"))


def test_mutant_no_full_refusal_line_fails():
    """A fully refused room logs nothing (the old 'nothing wanted, nothing to say' return): a default run could not
    tell the room was refused."""
    n = norm(CACHE)
    b = body(n, ROOM_LOG_SIG)
    at = b.find(ALL_REFUSED)
    assert at >= 0
    refused = body(b[at:], "if (room.n_held == 0)")
    whole = "if (room.n_held == 0) " + refused
    assert n.count(whole) == 1
    assert not claim_room_line_states_the_decision(n.replace(whole, "if (room.n_held == 0) { return; }", 1))


def test_mutant_refusal_not_a_warn_fails():
    assert not claim_room_line_states_the_decision(_once(CACHE, WARN, WARN.replace(" || room.n_refused > 0", "")))


def test_mutant_line_drops_the_parameters_fails():
    """Both lines print the costs without the parameters that produced them."""
    n = norm(CACHE)
    assert n.count(COSTS) == 2 and n.count(COST_ARGS) == 2
    mutant = n.replace(COSTS, COSTS.split(" (fill")[0]).replace(COST_ARGS, "room.decision.host_attn_us, "
                                                                            "room.decision.expert_us")
    assert not claim_room_line_states_the_decision(mutant)


def test_mutant_line_drops_the_decision_fails():
    n = norm(CACHE)
    assert n.count(DECISION) == 2 and n.count(DECISION_ARGS) == 2
    mutant = n.replace(DECISION, "[PLACEMENT] KV context room on device %d").replace(DECISION_ARGS,
                                                                                 "device_id, room.refused / mb,")
    assert not claim_room_line_states_the_decision(mutant)


def test_mutant_full_summary_counts_every_layer_fails():
    """The full-attention part counts the SWA layers too: a held SWA layer would read as a held full-attention one."""
    assert not claim_room_line_states_the_decision(
        _once(CACHE, FULL_SUMMARY, 'const std::string full = kv_context_room_class_summary(room.n_held, '
              'room.n_layers, "full-attention");'))


def test_mutant_summary_hides_a_full_refusal_fails():
    """No "refused all" form: a class whose every room was refused reads as a partial split."""
    assert not claim_room_line_states_the_decision(
        _once(CACHE, '} else if (n_held == 0) { snprintf(text, sizeof(text), "refused all %zu %s layer(s) by cost", '
              'n_layers, kind); }', "}"))


def test_mutant_swa_not_counted_fails():
    assert not claim_room_counts_each_attention_class(_once(CACHE, SWA_COUNT, ""))
