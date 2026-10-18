"""Source contract for llama.cpp-w013 and llama.cpp-7fmt: the single-device KV context room is cost-based (owner ruling
2026-10-10), with per-type CPU expert rates and a per-split host cost.

Host-only: reads sources, builds nothing, loads no model and touches no device.

Why: with the room always held, GPT-OSS 20B on the B50 at its full requested context decoded at 22.3 t/s against
25.35 t/s with eleven layers' attention demoted to the host, while Qwen3.8 IQ3_XXS on the B70 at -c 4096 decoded at
10.49 t/s with the room against 9.70 without it. The sign of the trade depends on the model, so the planner weighs it.

The fix this gate pins:
  - hold_kv_context_room decides each device layer's room on its own, in the order the runtime demotes KV layers
    (plan_runtime_kv_demotion: full attention latest first, then SWA latest first, read from kv-runtime-demotion.cpp);
    the order picks only the deciding layer the line reports, not what is held;
  - a layer's room is held iff A > B, per decode token, with the arithmetic in kv-room-cost.hpp (kv_room_weigh,
    which test-kv-room-cost runs on the measured shapes):
      A = host attention    = the layer's KV at F cells / host_attn_gbps + n_splits x host_split_us,
                              F = n_ctx_context x ctx_fill_pct / 100, n_splits = 3 with an indexer key cache, else 1
      B = displaced experts = the layer's room x n_expert_used / n_expert / the device's expert rate
    and a dense model (no expert to displace) holds its room as before; both rates are asserted finite and positive
    before they divide and the split cost finite and not negative before it adds, so a zero cannot refuse a room
    through a NaN;
  - the device's expert rate (llama.cpp-7fmt) is the byte-weighted harmonic mean over its routed-expert tensors of each
    type's measured rate (a constexpr table whose every entry names the measurement it came from, checked at compile
    time: finite, positive and named), a type with none counting at cpu_expert_gbps; the seven Qwen3.8 GSQ-RCO types
    carry the one rate their mix was measured at, and the catalog row gives the mix's shares;
    GGML_SYCL_PLAN_CPU_EXPERT_GBPS, when set, prices every type; for the model's default context a type with no measured
    rate leaves the room unpriced, so every layer's room is refused unweighed and one WARN names the types and the
    override; the rate is computed once, before the room is held;
  - with no requested context (n_ctx_context 0) it returns before weighing anything;
  - the four parameters are placement_kv_info fields, filled once where the backend copies the inventory in, from
    GGML_SYCL_PLAN_CTX_FILL_PCT / GGML_SYCL_PLAN_HOST_ATTN_GBPS / GGML_SYCL_PLAN_CPU_EXPERT_GBPS /
    GGML_SYCL_PLAN_HOST_SPLIT_US (defaults 50 / 30 / 40 / 250, a nonsense value keeps the default and does not count
    as set, a fill above 100 is 100), and catalogued in docs/backend/sycl-env-vars.md, the CPU row with the measured
    table and its method, the split row with both measurements and the admissible window; no other C/C++ source of
    the backend names them, except the unpriced WARN's one hint;
  - the split term does not scale with a layer's KV per cell, so layers can decide differently; the room's line states
    the decision per attention class ("held all N full-attention layer(s)" / "refused all N ... by cost" / "held K of N
    ..., refused M by cost", and the same for SWA) and gives the deciding layer's A and B in microseconds per token with
    each term's source (the fill and its parameters, the splits and their cost, the expert rate with each measurement
    after the types it priced and then the unmeasured types, or "override"), and the refused bytes only when a layer was
    refused; a fully refused room logs one WARN saying so;
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


CACHE_REL = "ggml/src/ggml-sycl/unified-cache.cpp"
CACHE_HPP_REL = "ggml/src/ggml-sycl/unified-cache.hpp"
ROOM_COST_REL = "ggml/src/ggml-sycl/kv-room-cost.hpp"
SYCL_REL = "ggml/src/ggml-sycl/ggml-sycl.cpp"
FATTN_REL = "ggml/src/ggml-sycl/fattn.cpp"
DEMOTION_REL = "ggml/src/ggml-sycl/kv-runtime-demotion.cpp"

CACHE = read(CACHE_REL)
CACHE_HPP = read(CACHE_HPP_REL)
ROOM_COST = read(ROOM_COST_REL)
SYCL = read(SYCL_REL)
DEMOTION = read(DEMOTION_REL)
DOC = read("docs/backend/sycl-env-vars.md")
# Every C/C++ source of the SYCL backend, by path from the repository root: where a second reader would hide.
SYCL_TREE = {
    p.relative_to(ROOT).as_posix(): p.read_text(errors="replace")
    for p in sorted((ROOT / "ggml/src/ggml-sycl").rglob("*"))
    if p.is_file() and p.suffix in (".c", ".cpp", ".h", ".hpp")
}


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
PARAM_SIG = "static double ggml_sycl_plan_cost_param(const char * name, double def, double max, bool * set = nullptr)"

ROOM_CALL = "room = hold_kv_context_room(plan, kv_info, device_id, n_experts, room_expert_rate, remaining);"
COPY_CTX = "g_placement_kv_info.n_ctx_context = inventory->n_ctx_context;"
NO_DOUBLE_MAX = "std::numeric_limits<double>::max()"

# name -> (placement_kv_info field, default constant, upper bound at the fill site, unit, source the doc row cites,
#          the placement_kv_info flag the parser sets when the variable supplied the value, or None)
PARAMS = {
    "GGML_SYCL_PLAN_CTX_FILL_PCT": ("ctx_fill_pct", "PLACEMENT_CTX_FILL_PCT_DEFAULT", "100.0", "%",
                                    "llama.cpp-w013", None),
    "GGML_SYCL_PLAN_HOST_ATTN_GBPS": ("host_attn_gbps", "PLACEMENT_HOST_ATTN_GBPS_DEFAULT", NO_DOUBLE_MAX, "GB/s",
                                      "unified-memory-manager-design.md", None),
    "GGML_SYCL_PLAN_CPU_EXPERT_GBPS": ("cpu_expert_gbps", "PLACEMENT_CPU_EXPERT_GBPS_DEFAULT", NO_DOUBLE_MAX, "GB/s",
                                       "llama.cpp-8bgi", "cpu_expert_gbps_set"),
    "GGML_SYCL_PLAN_HOST_SPLIT_US": ("host_split_us", "PLACEMENT_HOST_SPLIT_US_DEFAULT", NO_DOUBLE_MAX,
                                     "us per CPU split", "llama.cpp-7fmt", None),
}


def fill(name: str) -> str:
    field, const, bound, _, _, flag = PARAMS[name]
    set_arg = f", &g_placement_kv_info.{flag}" if flag else ""
    return f'g_placement_kv_info.{field} = ggml_sycl_plan_cost_param("{name}", ggml_sycl::{const}, {bound}{set_arg});'


def header_default(hpp: str, const: str) -> float | None:
    m = re.search(rf"constexpr double {const} = ([0-9.]+);", norm(hpp))
    return float(m.group(1)) if m else None


# ---- the parameters -------------------------------------------------------------------------------------------------


def claim_params_are_kv_info_fields(hpp: str) -> bool:
    """Each parameter is a placement_kv_info field defaulting to its named constant, so a test sets it directly and
    an unset field behaves as the default run does; the override flag defaults to unset."""
    n = norm(hpp)
    at = n.find("struct placement_kv_info {")
    kv = body(n[at:], "struct placement_kv_info") if at >= 0 else ""
    return (bool(kv) and all(f"double {field} = {const};" in kv for field, const, _, _, _, _ in PARAMS.values())
            and all(f"bool {flag} = false;" in kv for _, _, _, _, _, flag in PARAMS.values() if flag))


def claim_params_are_documented(doc: str, hpp: str) -> bool:
    """Each variable has a row in the env catalog giving its default (equal to the header's constant), its unit, and
    the measurement or ruling the default comes from."""
    for name, (_, const, _, unit, source, _) in PARAMS.items():
        rows = re.findall(rf"^\| `{name}=<[^>]*>` \| \*\*([0-9.]+)\*\* \|(.*)$", doc, re.MULTILINE)
        default = header_default(hpp, const)
        if len(rows) != 1 or default is None or float(rows[0][0]) != default:
            return False
        if unit not in rows[0][1] or source not in rows[0][1] or "us/token" not in rows[0][1]:
            return False
    return True


# The unpriced room's WARN names the override that would price it (llama.cpp-7fmt). It is a string the line prints, not
# a read, so exactly that one hint is exempt, and only inside the room's line.
OVERRIDE_HINT = "GGML_SYCL_PLAN_CPU_EXPERT_GBPS=<GB/s> prices every type"


def _scanned(rel: str, text: str) -> str:
    n = norm(text)
    if rel == CACHE_REL and n.count(OVERRIDE_HINT) == 1 and body(n, ROOM_LOG_SIG).count(OVERRIDE_HINT) == 1:
        n = n.replace(OVERRIDE_HINT, "", 1)
    return n


def claim_fill_site_reads_each_once(tree: dict[str, str]) -> bool:
    """The backend reads each variable once, where it copies the inventory into its KV inputs, right after the
    requested context; nothing else reads them. Every C/C++ source of the backend is scanned (comments stripped): the
    name occurs once in ggml-sycl.cpp, in that site, and in no other file, except the one override hint the unpriced
    room's WARN prints."""
    if not all(rel in tree for rel in (SYCL_REL, CACHE_REL, CACHE_HPP_REL, ROOM_COST_REL)):
        return False
    s = norm(tree[SYCL_REL])
    site = body(s, FILL_SIG)
    others = [_scanned(rel, text) for rel, text in tree.items() if rel != SYCL_REL]
    if not site:
        return False
    for name in PARAMS:
        if s.count(name) != 1 or any(name in text for text in others):
            return False
        if not ordered(site, COPY_CTX, fill(name)):
            return False
    return True


PARSE_UNSET = "if (set != nullptr) { *set = false; }"
PARSE_REJECT = "if (end == env || !std::isfinite(value) || value <= 0.0) { return def; }"
PARSE_SET = "if (set != nullptr) { *set = true; }"
PARSE_CLAMP = "return std::min(value, max);"


def claim_nonsense_keeps_the_default(sycl: str) -> bool:
    """An unset, unparsable, non-finite, zero or negative value keeps the default and does not count as set; a value
    above the bound is the bound (the fill is a percent, so 100). Only an accepted value sets the flag, so a nonsense
    override cannot price every expert type at the fallback."""
    b = body(norm(sycl), PARAM_SIG)
    return (bool(b) and b.count(PARSE_SET) == 1
            and ordered(b, PARSE_UNSET, 'const char * env = std::getenv(name);', "if (env == nullptr) { return def; }",
                        "const double value = std::strtod(env, &end);", PARSE_REJECT, PARSE_SET, PARSE_CLAMP))


# ---- the decision ---------------------------------------------------------------------------------------------------


# The arithmetic, in kv-room-cost.hpp.
FILLED_SIG = "static inline uint32_t kv_room_filled_cells(uint32_t n_ctx_context, double ctx_fill_pct)"
FILLED = ("return static_cast<uint32_t>(static_cast<double>(n_ctx_context) * std::clamp(ctx_fill_pct, 0.0, 100.0) / "
          "100.0);")
ARITH_SIG = "static inline kv_room_cost_terms kv_room_weigh(const kv_room_cost_inputs & in)"
MOE = "const bool moe = in.n_experts > 0 && in.n_expert_used > 0;"
P_HIT = "const double p_hit = moe ? std::min(1.0, static_cast<double>(in.n_expert_used) / in.n_experts) : 0.0;"
FILL_US = "cost.fill_us = static_cast<double>(in.host_read_bytes) / (in.host_attn_gbps * 1e3);"
A_US = "cost.host_attn_us = cost.fill_us + static_cast<double>(in.n_host_splits) * in.host_split_us;"
B_US = "cost.expert_us = static_cast<double>(in.room_bytes) * p_hit / (in.cpu_expert_gbps * 1e3);"
HOLD_IFF = "cost.held = !moe || (in.priced && cost.host_attn_us > cost.expert_us);"
# The planner's weigh, in unified-cache.cpp: the guards, then the inputs.
HOST_GUARD = "GGML_ASSERT(std::isfinite(kv_info.host_attn_gbps) && kv_info.host_attn_gbps > 0.0);"
RATE_GUARD = "GGML_ASSERT(std::isfinite(rate.gbps) && rate.gbps > 0.0);"
SPLIT_GUARD = "GGML_ASSERT(std::isfinite(kv_info.host_split_us) && kv_info.host_split_us >= 0.0);"
FILLED_CALL = "const uint32_t filled = kv_room_filled_cells(kv_info.n_ctx_context, kv_info.ctx_fill_pct);"
HOST_READ = "in.host_read_bytes = kv_info.kv_bytes_for_layer_at(il, filled);"
SPLITS = "in.n_host_splits = placement_host_attn_splits(kv_info.idx_k_width(il) > 0);"
INPUTS = ("in.room_bytes = extra; in.n_experts = n_experts; in.n_expert_used = kv_info.n_expert_used; "
          "in.host_attn_gbps = kv_info.host_attn_gbps; in.host_split_us = kv_info.host_split_us; "
          "in.cpu_expert_gbps = rate.gbps; in.priced = rate.priced;")
WEIGH_RETURN = "return { kv_room_weigh(in), layer_id, in.n_host_splits };"


def claim_room_weighs_attention_against_experts(cache: str, room_cost: str) -> bool:
    """A = the layer's KV at the expected fill, read at the host-attention rate, plus the layer's CPU splits at the
    split cost; B = the layer's room times the chance a token uses a displaced expert, read at the device's expert
    rate; held iff A > B, or always for a dense model, and never when the room is not priced. Both rates are asserted
    finite and positive and the split cost finite and not negative before anything is weighed, as the fill is
    clamped: a zero rate would make A or B inf or NaN, and NaN > B is false, so the room would be refused without a
    word. The planner's weigh feeds kv_room_weigh exactly these inputs and returns its terms."""
    r = norm(room_cost)
    arith = body(r, ARITH_SIG)
    b = body(norm(cache), WEIGH_SIG)
    return (FILLED in body(r, FILLED_SIG) and bool(arith)
            and ordered(arith, MOE, P_HIT, FILL_US, A_US, B_US, HOLD_IFF) and bool(b)
            and ordered(b, HOST_GUARD, RATE_GUARD, SPLIT_GUARD, FILLED_CALL, HOST_READ, SPLITS, INPUTS, WEIGH_RETURN))


SPLIT_MAP_SIG = "constexpr uint32_t placement_host_attn_splits(bool has_indexer)"
SPLIT_MAP = "{ return has_indexer ? 3 : 1; }"


def claim_split_mapping(room_cost: str) -> bool:
    """A layer with an indexer key cache pays 3 CPU splits, any other layer with its own KV 1: the scheduler census
    (36 CPU splits for Qwen3.8's 12 demoted layers; 12 for GPT-OSS 20B's 12)."""
    return body(norm(room_cost), SPLIT_MAP_SIG) == SPLIT_MAP


# The measured CPU expert rates (llama.cpp-7fmt comments c-ncj4, c-plz8): ggml type -> (GB/s, the measurement).
# The seven Qwen3.8 GSQ-RCO types were measured as one mix, so each carries the mix's rate and source until
# llama.cpp-073y measures them one by one.
GPTOSS_SOURCE = "gpt-oss-20b 2026-10-18 (c-ncj4)"
QWEN_MIX_SOURCE = "qwen38-gsq-rco-mix 2026-10-18 (c-plz8)"
QWEN_MIX = ("GGML_TYPE_IQ2_XXS", "GGML_TYPE_IQ2_XS", "GGML_TYPE_IQ3_XXS", "GGML_TYPE_IQ4_NL", "GGML_TYPE_IQ3_S",
            "GGML_TYPE_IQ2_S", "GGML_TYPE_Q2_0")
TABLE = {"GGML_TYPE_MXFP4": (5.6, GPTOSS_SOURCE)} | {name: (22.0, QWEN_MIX_SOURCE) for name in QWEN_MIX}
TABLE_SIG = "constexpr placement_cpu_expert_rate placement_cpu_expert_rate_measured(ggml_type type)"
TABLE_HEAD = "{ switch (type) { "
TABLE_TAIL = "default: return { 0.0, nullptr }; } }"
TABLE_GROUP_RE = re.compile(r'((?:case GGML_TYPE_\w+: )+)return \{ ([0-9.]+), "([^"\\]*)" \}; ')
TABLE_VALID = ('static_assert(placement_cpu_expert_rates_valid(), '
               '"a measured CPU expert rate must be finite and positive and name its measurement");')
VALID_SIG = "constexpr bool placement_cpu_expert_rates_valid()"
VALID_TEST = ("const placement_cpu_expert_rate r = placement_cpu_expert_rate_measured(static_cast<ggml_type>(t)); "
              "if (r.source == nullptr ? r.gbps != 0.0 : !(r.gbps > 0.0 && "
              "r.gbps <= std::numeric_limits<double>::max())) { return false; }")


def table_of(room_cost: str) -> dict[str, tuple[float, str]] | None:
    """The switch's entries, type -> (rate, source), when the switch is nothing but groups of case labels each
    returning a rate and a source, then the unmeasured default; None otherwise or when a type repeats."""
    b = body(norm(room_cost), TABLE_SIG)
    if not b.startswith(TABLE_HEAD) or not b.endswith(TABLE_TAIL):
        return None
    middle = b[len(TABLE_HEAD):-len(TABLE_TAIL)]
    if TABLE_GROUP_RE.sub("", middle) != "":
        return None
    table: dict[str, tuple[float, str]] = {}
    n_labels = 0
    for labels, rate, source in TABLE_GROUP_RE.findall(middle):
        for name in re.findall(r"case (GGML_TYPE_\w+): ", labels):
            table[name] = (float(rate), source)
            n_labels += 1
    return table if len(table) == n_labels else None


def claim_expert_rate_table(room_cost: str) -> bool:
    """The table holds exactly the measured rates, each with the measurement it came from, every other type is
    unmeasured (0, no source), and a compile-time check rejects an entry that is not finite and positive or that
    has a rate and no source."""
    n = norm(room_cost)
    return table_of(room_cost) == TABLE and TABLE_VALID in n and VALID_TEST in body(n, VALID_SIG)


ADD_SIG = "void add(ggml_type type, size_t n_bytes)"
ADD_STEPS = ("const bool typed = type >= 0 && type < GGML_TYPE_COUNT;",
             "const double gbps = typed ? placement_cpu_expert_rate_measured(type).gbps : 0.0;",
             "const int bit = typed ? static_cast<int>(type) : static_cast<int>(GGML_TYPE_COUNT);", "bytes += n_bytes;",
             "if (gbps > 0.0) { measured_us += static_cast<double>(n_bytes) / (gbps * 1e3); "
             "measured_mask |= uint64_t(1) << bit; } else { unmeasured_bytes += n_bytes; "
             "unmeasured_mask |= uint64_t(1) << bit; }")
RATE_SIG = ("static inline kv_room_expert_rate kv_room_expert_rate_for(const kv_room_expert_types & types, "
            "double cpu_expert_gbps, bool cpu_expert_gbps_set, bool n_ctx_context_is_default)")
RATE_OVERRIDE = "if (cpu_expert_gbps_set || types.bytes == 0) { return rate; }"
RATE_UNMEASURED = "if (types.unmeasured_mask != 0 && n_ctx_context_is_default) { rate.priced = false; return rate; }"
RATE_MEAN = ("const double read_us = types.measured_us + static_cast<double>(types.unmeasured_bytes) / "
             "(cpu_expert_gbps * 1e3); rate.gbps = static_cast<double>(types.bytes) / (read_us * 1e3); return rate;")


def claim_expert_rate_rule(room_cost: str) -> bool:
    """B's rate: the override, when set, prices every type (and a device with no routed experts keeps the fallback);
    otherwise, for the model's default context, a type with no measured rate leaves the room unpriced; otherwise the
    byte-weighted harmonic mean of the measured rates, an unmeasured type at the fallback. Each tensor's bytes go to
    its type's measured rate, or to the unmeasured pool with an untyped tensor."""
    r = norm(room_cost)
    rate = body(r, RATE_SIG)
    return (ordered(body(r, ADD_SIG), *ADD_STEPS) and bool(rate)
            and ordered(rate, "rate.gbps = cpu_expert_gbps;", "rate.overridden = cpu_expert_gbps_set;", RATE_OVERRIDE,
                        RATE_UNMEASURED, RATE_MEAN)
            and rate.count("rate.priced = false;") == 1 and "rate.priced = true;" not in rate)


EXPERT_BRANCH = "if (n_experts > 0 && usage == tensor_usage::MOE_EXPERT_WEIGHT) {"
TYPES_DECL = "kv_room_expert_types room_expert_types;"
TYPES_ADD = "room_expert_types.add(tensor.type, src_size);"
PLAN_CPU_GUARD = "GGML_ASSERT(std::isfinite(kv_info.cpu_expert_gbps) && kv_info.cpu_expert_gbps > 0.0);"
RATE_CALL = ("const kv_room_expert_rate room_expert_rate = kv_room_expert_rate_for(room_expert_types, "
             "kv_info.cpu_expert_gbps, kv_info.cpu_expert_gbps_set, kv_info.n_ctx_context_is_default);")


def claim_planner_reads_the_rate_once(cache: str) -> bool:
    """The planner collects every routed-expert tensor's type and bytes in the inventory loop, asserts the fallback
    rate finite and positive (it divides in the mean), computes the rate once before the room is held, and hands that
    one rate to the room."""
    b = body(norm(cache), PLAN_SIG)
    return (bool(b) and ordered(b, TYPES_DECL, EXPERT_BRANCH, TYPES_ADD, PLAN_CPU_GUARD, RATE_CALL,
                                "kv_context_room room;", ROOM_CALL)
            and b.count(TYPES_ADD) == 1 and b.count("kv_room_expert_rate_for(") == 1)


WEIGH_CALL = ("const kv_context_room_cost cost = weigh_kv_context_room(kv_info, layer_id, extra, n_experts, "
              "room.expert_rate);")
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
    first. The order picks only the deciding layer the line reports; the runtime's own order is pinned at its owner by
    claim_runtime_demotes_full_then_swa_latest_first."""
    b = body(norm(cache), ROOM_SIG)
    return bool(b) and ordered(b, DEVICE_LAYERS, SORT, PASSES, LATEST_FIRST, PASS_FILTER, WEIGH_CALL)


DEMOTE_SIG = "kv_demotion_result plan_runtime_kv_demotion(const kv_demotion_input & in)"
DEMOTE_PASS = "auto demote_pass = [&](bool swa_pass) {"
DEMOTE_LATEST_FIRST = "for (int l = n_layers - 1; l >= 0; --l) {"
DEMOTE_FILTER = "if (is_swa != swa_pass) { continue; }"
DEMOTE_FULL = "demote_pass(false);"
DEMOTE_SWA = "demote_pass(true);"


def claim_runtime_demotes_full_then_swa_latest_first(demotion: str) -> bool:
    """The order the planner's loop, its comment and the catalog row name is the runtime's own: plan_runtime_kv_demotion
    demotes full-attention layers latest first, then SWA layers latest first. Read from its owner, so a change there
    turns this red instead of leaving the planner describing an order nothing follows."""
    b = body(norm(demotion), DEMOTE_SIG)
    return (bool(b) and b.count(DEMOTE_FULL) == 1 and b.count(DEMOTE_SWA) == 1
            and ordered(b, DEMOTE_PASS, DEMOTE_LATEST_FIRST, DEMOTE_FILTER, DEMOTE_FULL, DEMOTE_SWA))


NO_REQUEST = "kv_context_room room; if (kv_info.n_ctx_context == 0) { return room; }"
KEEP_RATE = "room.expert_rate = expert_rate;"


def claim_no_request_returns_before_any_cost(cache: str) -> bool:
    """With no requested context the function returns before it looks at a layer or weighs anything."""
    b = body(norm(cache), ROOM_SIG)
    return (bool(b) and b.startswith("{ " + NO_REQUEST + " " + KEEP_RATE)
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
DECISION_ARGS = "device_id, full.c_str(), swa.c_str(),"
REFUSED_NOTE = ('char refused_note[64] = ""; if (room.refused > 0) { snprintf(refused_note, sizeof(refused_note), '
                '" (%.1f MB refused by cost)", room.refused / mb); }')
REFUSED_SLOT = "[PLACEMENT] KV context room on device %d: %s, %s%s; held %.1f MB of %.1f MB"
REFUSED_ARGS = "device_id, full.c_str(), swa.c_str(), refused_note, room.held / mb, room.wanted / mb,"
COSTS = ("host attention %.1f us/token (%.1f fill at %.4g%% and host-attn %.4g GB/s + %u split(s) x %.4g us) vs "
         "displaced experts %.1f us/token (cpu-expert %.4g GB/s, %s)")
COST_ARGS = ("room.decision.host_attn_us, room.decision.fill_us, kv_info.ctx_fill_pct, kv_info.host_attn_gbps, "
             "room.decision.n_splits, kv_info.host_split_us, room.decision.expert_us, room.expert_rate.gbps, "
             "rate_source.c_str()")
RATE_SOURCE = "const std::string rate_source = kv_room_rate_source(room.expert_rate, ggml_type_name);"
UNMEASURED_LOCAL = ("const std::string unmeasured = kv_room_type_names(room.expert_rate.types.unmeasured_mask, "
                    "ggml_type_name);")
UNPRICED = "if (!room.expert_rate.priced && room.n_held == 0) { " + UNMEASURED_LOCAL + " GGML_LOG_WARN("
UNPRICED_SAYS = ("refused for every layer without weighing", "include %s, with no measured CPU rate", OVERRIDE_HINT,
                 "device_id, unmeasured.c_str(),", "return;")
TAIL = "n_ctx_context is the requested n_ctx.\\n"
ALL_REFUSED = "if (room.n_held == 0) { GGML_LOG_WARN("
ALL_REFUSED_SAYS = ("every layer's room was refused by cost", "the runtime re-places the overflow at context creation")
WARN = "const bool warn = room.displaced_bytes() > 0 || room.held < room.wanted || room.n_refused > 0;"


def claim_room_line_states_the_decision(cache: str) -> bool:
    """Both lines say, per attention class, whether its layers were held or refused by cost ("held all N", "refused
    all N", or the mixed form when the split term makes layers differ), and give the deciding layer's A and B in
    microseconds per token with each term's source; a fully refused room is one WARN saying every room was refused
    and that the runtime re-places the overflow; otherwise the line is a WARN whenever the room cost experts, could
    not hold all it wanted, or refused a layer, and gives the refused bytes in parentheses only when a layer was
    refused."""
    n = norm(cache)
    b = body(n, ROOM_LOG_SIG)
    summary = body(n, SUMMARY_SIG)
    if not b or not summary or "if (room.n_layers == 0) { return; }" not in b:
        return False
    if not ordered(summary, *SUMMARY_FORMS) or not ordered(b, FULL_SUMMARY, SWA_SUMMARY, RATE_SOURCE, ALL_REFUSED):
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
            and ordered(rest, REFUSED_NOTE, REFUSED_SLOT, REFUSED_ARGS)
            and "ggml_log_internal(warn ? GGML_LOG_LEVEL_WARN : GGML_LOG_LEVEL_INFO," in rest)


def claim_unpriced_room_is_one_warn(cache: str) -> bool:
    """A room left unpriced (the default context with an unmeasured expert type) logs one WARN before the cost lines:
    it says every layer was refused without weighing, names the unmeasured types and the override that would price
    them, and returns, so no line prints an A and B that were never compared."""
    b = body(norm(cache), ROOM_LOG_SIG)
    at = b.find(UNPRICED)
    if at < 0 or b.count(UNPRICED) != 1 or not ordered(b, "if (room.n_layers == 0) { return; }", UNPRICED,
                                                         RATE_SOURCE, ALL_REFUSED):
        return False
    unpriced = body(b[at:], "if (!room.expert_rate.priced && room.n_held == 0)")
    return bool(unpriced) and all(s in unpriced for s in UNPRICED_SAYS) and COSTS not in unpriced


SOURCE_SIG = ("static inline std::string kv_room_rate_source(const kv_room_expert_rate & rate, "
              "kv_room_type_name_fn type_name)")
SOURCE_STEPS = ('if (rate.overridden) { return "override"; }',
                'if (rate.types.bytes == 0) { return "no routed experts"; }',
                "uint64_t left = rate.types.measured_mask;", "for (int t = 0; t < GGML_TYPE_COUNT; ++t) {",
                "if (((left >> t) & 1) == 0) { continue; }",
                "const char * from = placement_cpu_expert_rate_measured(static_cast<ggml_type>(t)).source;",
                "for (int u = t; u < GGML_TYPE_COUNT; ++u) {",
                "const char * other = placement_cpu_expert_rate_measured(static_cast<ggml_type>(u)).source;",
                "if (((left >> u) & 1) != 0 && std::strcmp(other, from) == 0) { same |= uint64_t(1) << u; }",
                "left &= ~same;", 'source += kv_room_type_names(same, type_name) + " via " + from;',
                "if (rate.types.unmeasured_mask != 0) {", 'source += source.empty() ? "unmeasured " : "; unmeasured ";',
                "source += kv_room_type_names(rate.types.unmeasured_mask, type_name);")
NAMES_SIG = "static inline std::string kv_room_type_names(uint64_t mask, kv_room_type_name_fn type_name)"
NAMES_STEPS = ("for (int t = 0; t <= GGML_TYPE_COUNT; ++t) {", "if (((mask >> t) & 1) == 0) { continue; }",
               'names += t == GGML_TYPE_COUNT ? "untyped" : type_name(static_cast<ggml_type>(t));')


def claim_line_names_the_rate_source(cache: str, room_cost: str) -> bool:
    """The rate's source reads "override" when the override priced every type, otherwise each measurement after the
    types it priced, then the unmeasured types, by ggml type name ("untyped" for a tensor with none). The helpers
    live in kv-room-cost.hpp, where test-kv-room-cost runs them; the room's line calls them with ggml_type_name and
    the planner defines no copy of its own."""
    r = norm(room_cost)
    log = body(norm(cache), ROOM_LOG_SIG)
    return (ordered(body(r, SOURCE_SIG), *SOURCE_STEPS) and ordered(body(r, NAMES_SIG), *NAMES_STEPS)
            and RATE_SOURCE in log and UNMEASURED_LOCAL in log
            and all(re.search(rf"\bstd::string {name}\(", norm(cache)) is None
                    for name in ("kv_room_type_names", "kv_room_rate_source")))


def doc_row(doc: str, name: str) -> str:
    rows = [line for line in doc.splitlines() if line.startswith(f"| `{name}=")]
    return rows[0] if len(rows) == 1 else ""


TABLE_DOC_SAYS = ("c-ncj4", "c-plz8", "2026-10-18", "26.35", "load ~30", "-t 4", "harmonic mean", "override",
                  "-c 131072", "-c 4096", "refuses the room for every layer", "it can only refuse",
                  "measured together as one mix", "`.bias` excluded", "iq3_s 21.8 %", "iq4_nl 19.8 %",
                  "q2_0 16.5 %", "iq2_s 12.5 %", "iq2_xs 11.3 %", "iq2_xxs 9.1 %", "iq3_xxs 9.0 %",
                  "`ffn_down_exps` is iq4_nl and q2_0", "unvalidated rate", "llama.cpp-073y")
SPLIT_DOC_SAYS = ("2026-10-18", "~250", "3 ms", "~700", "25 ms", "175 < s < 1495", "indexer key cache pays 3")


def claim_measurements_are_documented(doc: str, room_cost: str) -> bool:
    """The CPU row lists each measured type at the table's rate with the method, host state, date and threads, the
    unmeasured rule and the override; the split row gives both measured values and the window that keeps every
    pinned arm's verdict."""
    cpu = doc_row(doc, "GGML_SYCL_PLAN_CPU_EXPERT_GBPS")
    split = doc_row(doc, "GGML_SYCL_PLAN_HOST_SPLIT_US")
    table = table_of(room_cost) or {}
    return (bool(table) and doc_lists_the_table(cpu, table) and all(s in cpu for s in TABLE_DOC_SAYS)
            and all(s in split for s in SPLIT_DOC_SAYS))


def doc_lists_the_table(cpu: str, table: dict[str, tuple[float, str]]) -> bool:
    """Each measurement is listed once, as "source `<source>`", after its rate and every type it priced (each type's
    name in backquotes), all of them after the previous measurement."""
    start = 0
    for source in dict.fromkeys(src for _, src in table.values()):
        anchor = f"source `{source}`"
        at = cpu.find(anchor, start)
        if at < 0 or cpu.count(anchor) != 1:
            return False
        segment = cpu[start:at]
        rates = {rate for rate, src in table.values() if src == source}
        names = [name[len("GGML_TYPE_"):].lower() for name, (_, src) in table.items() if src == source]
        if len(rates) != 1 or f" {rates.pop():g} GB/s" not in segment:
            return False
        if not all(f"`{name}`" in segment for name in names):
            return False
        start = at + len(anchor)
    return True


def test_params_are_kv_info_fields():
    assert claim_params_are_kv_info_fields(CACHE_HPP)


def test_params_are_documented():
    assert claim_params_are_documented(DOC, ROOM_COST)


def test_measurements_are_documented():
    assert claim_measurements_are_documented(DOC, ROOM_COST)


def test_split_mapping():
    assert claim_split_mapping(ROOM_COST)


def test_expert_rate_table():
    assert claim_expert_rate_table(ROOM_COST)


def test_expert_rate_rule():
    assert claim_expert_rate_rule(ROOM_COST)


def test_planner_reads_the_rate_once():
    assert claim_planner_reads_the_rate_once(CACHE)


def test_unpriced_room_is_one_warn():
    assert claim_unpriced_room_is_one_warn(CACHE)


def test_line_names_the_rate_source():
    assert claim_line_names_the_rate_source(CACHE, ROOM_COST)


def test_fill_site_reads_each_once():
    assert claim_fill_site_reads_each_once(SYCL_TREE)


def test_nonsense_keeps_the_default():
    assert claim_nonsense_keeps_the_default(SYCL)


def test_room_weighs_attention_against_experts():
    assert claim_room_weighs_attention_against_experts(CACHE, ROOM_COST)


def test_room_holds_only_the_layers_that_pass():
    assert claim_room_holds_only_the_layers_that_pass(CACHE)


def test_decision_iterates_latest_first():
    assert claim_decision_iterates_latest_first(CACHE)


def test_runtime_demotes_full_then_swa_latest_first():
    assert claim_runtime_demotes_full_then_swa_latest_first(DEMOTION)


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


def test_mutant_override_flag_defaults_to_set_fails():
    """A placement_kv_info that reads as overridden by default: a test fixture's experts would all be priced at the
    fallback, and no default-context room would ever be refused for an unmeasured type."""
    assert not claim_params_are_kv_info_fields(
        _once(CACHE_HPP, "bool cpu_expert_gbps_set = false;", "bool cpu_expert_gbps_set = true;"))


def test_mutant_doc_default_differs_fails():
    """The catalog says 40 while the code defaults to 50."""
    assert not claim_params_are_documented(
        _doc_once("| `GGML_SYCL_PLAN_CTX_FILL_PCT=<pct>` | **50** |", "| `GGML_SYCL_PLAN_CTX_FILL_PCT=<pct>` | **40** |"),
        ROOM_COST)


def test_mutant_header_default_differs_fails():
    assert not claim_params_are_documented(
        DOC, _once(ROOM_COST, "constexpr double PLACEMENT_CPU_EXPERT_GBPS_DEFAULT = 40.0;",
                   "constexpr double PLACEMENT_CPU_EXPERT_GBPS_DEFAULT = 48.0;"))


def test_mutant_split_default_differs_fails():
    """The split cost defaults to the Qwen measurement while the catalog says 250."""
    assert not claim_params_are_documented(
        DOC, _once(ROOM_COST, "constexpr double PLACEMENT_HOST_SPLIT_US_DEFAULT = 250.0;",
                   "constexpr double PLACEMENT_HOST_SPLIT_US_DEFAULT = 700.0;"))


def test_mutant_doc_row_missing_fails():
    rows = [line for line in DOC.splitlines() if line.startswith("| `GGML_SYCL_PLAN_HOST_ATTN_GBPS=")]
    assert len(rows) == 1
    assert not claim_params_are_documented(_doc_once(rows[0] + "\n", ""), ROOM_COST)


def test_mutant_split_row_missing_fails():
    row = doc_row(DOC, "GGML_SYCL_PLAN_HOST_SPLIT_US")
    assert row
    assert not claim_params_are_documented(_doc_once(row + "\n", ""), ROOM_COST)


def test_mutant_doc_row_without_source_fails():
    rows = [line for line in DOC.splitlines() if line.startswith("| `GGML_SYCL_PLAN_CPU_EXPERT_GBPS=")]
    assert len(rows) == 1 and rows[0].count("llama.cpp-8bgi") >= 1
    assert not claim_params_are_documented(_doc_once(rows[0], rows[0].replace("llama.cpp-8bgi", "a measurement")),
                                           ROOM_COST)


def test_mutant_doc_rate_differs_fails():
    """The catalog lists the MXFP4 rate the table no longer holds."""
    row = doc_row(DOC, "GGML_SYCL_PLAN_CPU_EXPERT_GBPS")
    assert row.count("`mxfp4` 5.6 GB/s") == 1
    assert not claim_measurements_are_documented(_doc_once("`mxfp4` 5.6 GB/s", "`mxfp4` 6 GB/s"), ROOM_COST)


def test_mutant_doc_source_differs_fails():
    """The catalog attributes the seven types' rate to a measurement the table does not name."""
    assert not claim_measurements_are_documented(
        _doc_once(f"source `{QWEN_MIX_SOURCE}`", "source `qwen38-iq3_xxs 2026-10-18 (c-plz8)`"), ROOM_COST)


def test_mutant_doc_drops_a_mix_type_fails():
    """The catalog lists six of the seven types the mix's rate prices."""
    assert not claim_measurements_are_documented(_doc_once("`iq2_xxs` and `iq3_xxs` 22 GB/s each",
                                                           "`iq2_xxs` 22 GB/s each"), ROOM_COST)


def test_mutant_doc_share_differs_fails():
    """A share the census did not find: the row would describe another mix than the one measured."""
    assert not claim_measurements_are_documented(_doc_once("iq2_xs 11.3 %", "iq2_xs 13.3 %"), ROOM_COST)


def test_mutant_doc_says_the_mix_validates_every_mix_fails():
    """The row no longer says another mix of the seven reads an unvalidated rate."""
    row = doc_row(DOC, "GGML_SYCL_PLAN_CPU_EXPERT_GBPS")
    assert not claim_measurements_are_documented(
        _doc_once(row, row.replace("reads an unvalidated rate", "reads the measured rate")), ROOM_COST)


def test_mutant_doc_method_missing_fails():
    """The CPU row gives the rates but not how they were measured."""
    row = doc_row(DOC, "GGML_SYCL_PLAN_CPU_EXPERT_GBPS")
    assert not claim_measurements_are_documented(_doc_once(row, row.replace("c-ncj4", "a probe")), ROOM_COST)


def test_mutant_doc_window_missing_fails():
    row = doc_row(DOC, "GGML_SYCL_PLAN_HOST_SPLIT_US")
    assert not claim_measurements_are_documented(_doc_once(row, row.replace("175 < s < 1495", "a wide range")),
                                                 ROOM_COST)


def _tree_with(rel: str, text: str) -> dict[str, str]:
    """The backend's sources with the file at `rel` replaced by `text`."""
    assert rel in SYCL_TREE, f"mutant target must be a scanned source: {rel}"
    tree = dict(SYCL_TREE)
    tree[rel] = text
    return tree


def test_mutant_planner_reads_the_override_fails():
    """The planner reads the CPU expert variable itself: the override hint's exemption must not cover a read."""
    mutant = CACHE + '\nstatic const char * g_7fmt_mutant = std::getenv("GGML_SYCL_PLAN_CPU_EXPERT_GBPS");\n'
    assert not claim_fill_site_reads_each_once(_tree_with(CACHE_REL, mutant))


def test_mutant_override_hint_outside_the_line_fails():
    """The hint string a second time, outside the room's line: the exemption covers the one hint the WARN prints."""
    mutant = CACHE + '\nstatic const char * g_7fmt_mutant = "' + OVERRIDE_HINT + '";\n'
    assert not claim_fill_site_reads_each_once(_tree_with(CACHE_REL, mutant))


def test_mutant_split_read_by_the_header_fails():
    """The cost header reads the split variable: a reader in the arithmetic is a second source."""
    mutant = ROOM_COST + '\nstatic const char * g_7fmt_mutant = std::getenv("GGML_SYCL_PLAN_HOST_SPLIT_US");\n'
    assert not claim_fill_site_reads_each_once(_tree_with(ROOM_COST_REL, mutant))


def test_mutant_override_flag_not_passed_fails():
    """The fill site reads the CPU expert variable without its flag: an override would price only the unmeasured
    types and never be reported."""
    site = fill("GGML_SYCL_PLAN_CPU_EXPERT_GBPS")
    assert not claim_fill_site_reads_each_once(
        _tree_with(SYCL_REL, _once(SYCL, site, site.replace(", &g_placement_kv_info.cpu_expert_gbps_set", ""))))


def test_mutant_split_not_read_fails():
    mutant = _once(SYCL, fill("GGML_SYCL_PLAN_HOST_SPLIT_US"), "")
    assert not claim_fill_site_reads_each_once(_tree_with(SYCL_REL, mutant))


def test_mutant_second_reader_fails():
    """The planner reads a variable itself as well: two sources for one parameter."""
    mutant = CACHE + '\nstatic const char * g_w013_mutant = std::getenv("GGML_SYCL_PLAN_HOST_ATTN_GBPS");\n'
    assert not claim_fill_site_reads_each_once(_tree_with(CACHE_REL, mutant))


def test_mutant_reader_in_another_tu_fails():
    """The flash-attention TU reads a variable too: a reader outside the planner's own files is still a second
    source."""
    mutant = SYCL_TREE[FATTN_REL] + '\nstatic const char * g_w013_mutant = std::getenv("GGML_SYCL_PLAN_CTX_FILL_PCT");\n'
    assert not claim_fill_site_reads_each_once(_tree_with(FATTN_REL, mutant))


def test_mutant_fill_site_drops_a_param_fails():
    assert not claim_fill_site_reads_each_once(_tree_with(SYCL_REL, _once(SYCL, fill("GGML_SYCL_PLAN_CPU_EXPERT_GBPS"), "")))


def test_mutant_fill_unclamped_fails():
    """A fill above 100% would expect more cells filled than the context has."""
    assert not claim_fill_site_reads_each_once(
        _tree_with(SYCL_REL, _once(SYCL, fill("GGML_SYCL_PLAN_CTX_FILL_PCT"),
                                   fill("GGML_SYCL_PLAN_CTX_FILL_PCT").replace("100.0);", NO_DOUBLE_MAX + ");"))))


def test_mutant_param_read_before_the_context_fails():
    """Read before the requested context is copied: still one read, but not at the site that fills the room's
    inputs together."""
    n = _once(SYCL, fill("GGML_SYCL_PLAN_HOST_ATTN_GBPS"), "")
    assert n.count(COPY_CTX) == 1
    mutant = n.replace(COPY_CTX, fill("GGML_SYCL_PLAN_HOST_ATTN_GBPS") + " " + COPY_CTX, 1)
    assert not claim_fill_site_reads_each_once(_tree_with(SYCL_REL, mutant))


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


def test_mutant_nonsense_counts_as_set_fails():
    """The flag set before the value is checked: a nonsense override would read as one and price every type at the
    default."""
    n = _param_mutant(PARSE_SET + " ", "")
    assert not claim_nonsense_keeps_the_default(n.replace(PARSE_REJECT, PARSE_SET + " " + PARSE_REJECT, 1))


def test_mutant_flag_never_cleared_fails():
    """A second fill (the parser runs at both inventory entry points) would keep a stale override."""
    assert not claim_nonsense_keeps_the_default(_param_mutant(PARSE_UNSET + " ", ""))


def test_mutant_garbage_accepted_fails():
    """An unparsable value read as 0 by strtod and then kept as 0: the reject must test that strtod parsed anything."""
    assert not claim_nonsense_keeps_the_default(_param_mutant("end == env || ", ""))


def test_mutant_no_bound_fails():
    assert not claim_nonsense_keeps_the_default(_param_mutant(PARSE_CLAMP, "return value;"))


def test_mutant_host_rate_unguarded_fails():
    """A test that sets host_attn_gbps to 0 would refuse every MoE room through a NaN instead of stopping."""
    assert not claim_room_weighs_attention_against_experts(_once(CACHE, HOST_GUARD, ""), ROOM_COST)


def test_mutant_cpu_rate_zero_admitted_fails():
    assert not claim_room_weighs_attention_against_experts(
        _once(CACHE, RATE_GUARD, RATE_GUARD.replace("rate.gbps > 0.0", "rate.gbps >= 0.0")), ROOM_COST)


def test_mutant_split_cost_unguarded_fails():
    """A NaN split cost makes A NaN, and NaN > B is false: every MoE room refused without a word."""
    assert not claim_room_weighs_attention_against_experts(_once(CACHE, SPLIT_GUARD, ""), ROOM_COST)


def test_mutant_host_read_undivided_fails():
    assert not claim_room_weighs_attention_against_experts(
        CACHE, _once(ROOM_COST, FILL_US, "cost.fill_us = static_cast<double>(in.host_read_bytes);"))


def test_mutant_expert_read_undivided_fails():
    assert not claim_room_weighs_attention_against_experts(
        CACHE, _once(ROOM_COST, B_US, "cost.expert_us = static_cast<double>(in.room_bytes) * p_hit;"))


def test_mutant_split_term_dropped_fails():
    """A without its splits: master's w013 A, which holds GPT-OSS 20B's default-context room (test-kv-room-cost)."""
    assert not claim_room_weighs_attention_against_experts(
        CACHE, _once(ROOM_COST, A_US, "cost.host_attn_us = cost.fill_us;"))


def test_mutant_split_cost_not_fed_fails():
    """The planner leaves the split cost out of the inputs: A's split term would be zero whatever the variable says."""
    assert not claim_room_weighs_attention_against_experts(
        _once(CACHE, "in.host_split_us = kv_info.host_split_us;", ""), ROOM_COST)


def test_mutant_single_cpu_rate_fails():
    """B at the one fallback rate instead of the device's: master's w013 B."""
    assert not claim_room_weighs_attention_against_experts(
        _once(CACHE, "in.cpu_expert_gbps = rate.gbps;", "in.cpu_expert_gbps = kv_info.cpu_expert_gbps;"), ROOM_COST)


def test_mutant_unpriced_room_weighed_fails():
    """The planner drops the priced flag: an unpriced room would be weighed at the fallback rate."""
    assert not claim_room_weighs_attention_against_experts(_once(CACHE, "in.priced = rate.priced;", ""), ROOM_COST)


def test_mutant_comparison_flipped_fails():
    assert not claim_room_weighs_attention_against_experts(
        CACHE, _once(ROOM_COST, "cost.host_attn_us > cost.expert_us);", "cost.host_attn_us < cost.expert_us);"))


def test_mutant_dense_weighed_fails():
    """A dense model weighed like a MoE one: at a zero expected fill its room, which displaces nothing, is refused."""
    assert not claim_room_weighs_attention_against_experts(
        CACHE, _once(ROOM_COST, HOLD_IFF, "cost.held = in.priced && cost.host_attn_us > cost.expert_us;"))


def test_mutant_unpriced_room_held_fails():
    """The priced flag ignored in the decision: a default context with an unmeasured type would weigh it anyway."""
    assert not claim_room_weighs_attention_against_experts(
        CACHE, _once(ROOM_COST, HOLD_IFF, "cost.held = !moe || cost.host_attn_us > cost.expert_us;"))


def test_mutant_every_expert_hit_fails():
    """p_hit 1: every displaced expert read by every token, whatever n_expert_used / n_expert says."""
    assert not claim_room_weighs_attention_against_experts(CACHE, _once(ROOM_COST, P_HIT, "const double p_hit = 1.0;"))


def test_mutant_attention_at_the_whole_context_fails():
    """The host read at the requested context instead of the expected fill: the fill parameter would do nothing."""
    assert not claim_room_weighs_attention_against_experts(
        _once(CACHE, HOST_READ, HOST_READ.replace(", filled);", ", kv_info.n_ctx_context);")), ROOM_COST)


def test_mutant_split_mapping_flipped_fails():
    """An indexer layer charged one split and any other three: Qwen3.8's A would lose two thirds of its split term."""
    assert not claim_split_mapping(_once(ROOM_COST, SPLIT_MAP[2:-2], "return has_indexer ? 1 : 3;"))


def test_mutant_split_mapping_ignores_the_indexer_fails():
    assert not claim_room_weighs_attention_against_experts(
        _once(CACHE, SPLITS, "in.n_host_splits = placement_host_attn_splits(false);"), ROOM_COST)


def test_mutant_table_entry_changed_fails():
    """MXFP4 at the old single rate: GPT-OSS 20B's default-context room would be held again."""
    assert not claim_expert_rate_table(_once(ROOM_COST, "return { 5.6,", "return { 40.0,"))


def test_mutant_table_entry_added_fails():
    """An unmeasured type given a rate in the table: it would no longer refuse a default-context room."""
    assert not claim_expert_rate_table(
        _once(ROOM_COST, TABLE_TAIL, 'case GGML_TYPE_Q8_0: return { 30.0, "a probe" }; ' + TABLE_TAIL))


def test_mutant_table_mix_type_dropped_fails():
    """q2_0 left out of the mix: 16.5 % of Qwen3.8's expert bytes would read as unmeasured and refuse its default
    room, the regression the table's seven entries exist to prevent."""
    assert not claim_expert_rate_table(_once(ROOM_COST, "case GGML_TYPE_Q2_0: ", ""))


def test_mutant_table_source_changed_fails():
    """The mix's rate attributed to one type's measurement: the line would claim a per-type rate nobody measured."""
    assert not claim_expert_rate_table(
        _once(ROOM_COST, f'"{QWEN_MIX_SOURCE}"', '"iq3_xxs 2026-10-18 (c-plz8)"'))


def test_mutant_table_unchecked_fails():
    assert not claim_expert_rate_table(_once(ROOM_COST, TABLE_VALID, ""))


def test_mutant_table_check_accepts_an_unnamed_rate_fails():
    """The check passes a rate with no source: an entry the line cannot attribute would build."""
    assert not claim_expert_rate_table(
        _once(ROOM_COST, "if (r.source == nullptr ? r.gbps != 0.0 :", "if (r.gbps == 0.0 ? false :"))


def test_mutant_unmeasured_rule_inverted_fails():
    """Refusing unweighed for a context the caller named, and weighing at the fallback for the default one."""
    assert not claim_expert_rate_rule(
        _once(ROOM_COST, RATE_UNMEASURED,
              RATE_UNMEASURED.replace("&& n_ctx_context_is_default", "&& !n_ctx_context_is_default")))


def test_mutant_unmeasured_rule_dropped_fails():
    """No rule: commit 1's default-context room weighed at 40 GB/s for every unmeasured type."""
    assert not claim_expert_rate_rule(_once(ROOM_COST, RATE_UNMEASURED, ""))


def test_mutant_unmeasured_rule_holds_fails():
    """The rule turned into a hold: the default context could then make a room held, not only refuse one."""
    assert not claim_expert_rate_rule(_once(ROOM_COST, "rate.priced = false;", "rate.priced = true;"))


def test_mutant_override_ignored_fails():
    """The override no longer prices every type: a measured type keeps its table rate under it."""
    assert not claim_expert_rate_rule(
        _once(ROOM_COST, RATE_OVERRIDE, "if (types.bytes == 0) { return rate; }"))


def test_mutant_unmeasured_bytes_counted_as_measured_fails():
    """An unmeasured type's bytes added to the measured time at no cost: the mean would be the measured rate alone."""
    assert not claim_expert_rate_rule(
        _once(ROOM_COST, "unmeasured_bytes += n_bytes;", ""))


def test_mutant_expert_types_not_collected_fails():
    """The planner never adds a tensor: every device reads as having no routed experts and keeps the fallback."""
    assert not claim_planner_reads_the_rate_once(_once(CACHE, TYPES_ADD, ""))


def test_mutant_rate_after_the_room_fails():
    """The rate computed after the room is held: the room would be weighed at an empty rate."""
    n = _once(CACHE, RATE_CALL + " ", "")
    assert n.count(ROOM_CALL) == 1
    assert not claim_planner_reads_the_rate_once(n.replace(ROOM_CALL, ROOM_CALL + " " + RATE_CALL, 1))


def test_mutant_fallback_unguarded_fails():
    assert not claim_planner_reads_the_rate_once(_once(CACHE, PLAN_CPU_GUARD, ""))


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
        _once(CACHE, ROOM_CALL, ROOM_CALL.replace("device_id, n_experts,", "device_id, 0,")))


def test_mutant_room_forgets_the_rate_fails():
    """The room keeps a default rate instead of the planner's: every layer weighed at 0 GB/s."""
    assert not claim_no_request_returns_before_any_cost(_once(CACHE, KEEP_RATE, ""))


def test_mutant_earliest_first_fails():
    assert not claim_decision_iterates_latest_first(
        _once(CACHE, LATEST_FIRST, "for (auto it = layers.begin(); it != layers.end(); ++it) {"))


def test_mutant_swa_first_fails():
    assert not claim_decision_iterates_latest_first(_once(CACHE, PASSES, "for (const bool swa_pass : { true, false }) {"))


def test_mutant_unsorted_fails():
    """The layers in the hash map's order: no order at all."""
    assert not claim_decision_iterates_latest_first(_once(CACHE, SORT, ""))


def test_mutant_runtime_demotes_earliest_first_fails():
    assert not claim_runtime_demotes_full_then_swa_latest_first(
        _once(DEMOTION, DEMOTE_LATEST_FIRST, "for (int l = 0; l < n_layers; ++l) {"))


def test_mutant_runtime_demotes_swa_first_fails():
    n = norm(DEMOTION)
    assert n.count(DEMOTE_FULL) == 1 and n.count(DEMOTE_SWA) == 1
    mutant = n.replace(DEMOTE_FULL, "@@").replace(DEMOTE_SWA, DEMOTE_FULL).replace("@@", DEMOTE_SWA)
    assert not claim_runtime_demotes_full_then_swa_latest_first(mutant)


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


OLD_COSTS = ("host attention %.1f us/token vs displaced experts %.1f us/token (fill %.4g%%, host-attn %.4g GB/s, "
             "cpu-expert %.4g GB/s)")
OLD_COST_ARGS = ("room.decision.host_attn_us, room.decision.expert_us, kv_info.ctx_fill_pct, kv_info.host_attn_gbps, "
                 "kv_info.cpu_expert_gbps")


def test_mutant_line_drops_the_terms_fails():
    """Both lines print w013's form: A and B with the three parameters, but neither the split term nor the rate's
    source, so a reader cannot tell a measured rate from the fallback."""
    n = norm(CACHE)
    assert n.count(COSTS) == 2 and n.count(COST_ARGS) == 2
    assert not claim_room_line_states_the_decision(n.replace(COSTS, OLD_COSTS).replace(COST_ARGS, OLD_COST_ARGS))


def test_mutant_unpriced_line_missing_fails():
    """No unpriced WARN: an unweighed refusal would print as "refused by cost" with an A and B never compared."""
    n = norm(CACHE)
    b = body(n, ROOM_LOG_SIG)
    at = b.find(UNPRICED)
    assert at >= 0
    head = "if (!room.expert_rate.priced && room.n_held == 0)"
    block = head + " " + body(b[at:], head)
    assert n.count(block) == 1
    assert not claim_unpriced_room_is_one_warn(n.replace(block, "", 1))


def test_mutant_unpriced_line_without_the_override_fails():
    assert not claim_unpriced_room_is_one_warn(_once(CACHE, OVERRIDE_HINT, "a variable prices every type"))


def test_mutant_unpriced_line_falls_through_fails():
    """The unpriced WARN does not return: the cost WARN prints after it."""
    n = norm(CACHE)
    b = body(n, ROOM_LOG_SIG)
    at = b.find(UNPRICED)
    block = body(b[at:], "if (!room.expert_rate.priced && room.n_held == 0)")
    assert block.endswith(" return; }") and n.count(block) == 1
    assert not claim_unpriced_room_is_one_warn(n.replace(block, block[:-len(" return; }")] + " }", 1))


def test_mutant_source_hides_the_override_fails():
    """An overridden rate labelled with the table's types: the line would claim a measurement it did not use."""
    assert not claim_line_names_the_rate_source(
        CACHE, _once(ROOM_COST, 'if (rate.overridden) { return "override"; }', ""))


def test_mutant_source_hides_unmeasured_types_fails():
    assert not claim_line_names_the_rate_source(
        CACHE, _once(ROOM_COST, "source += kv_room_type_names(rate.types.unmeasured_mask, type_name);", ""))


def test_mutant_source_drops_the_measurement_fails():
    """The types without the measurement they were priced by: the line would read as the pre-attribution form."""
    assert not claim_line_names_the_rate_source(
        CACHE, _once(ROOM_COST, 'source += kv_room_type_names(same, type_name) + " via " + from;',
                     "source += kv_room_type_names(same, type_name);"))


def test_mutant_source_lumps_every_measurement_fails():
    """Every measured type grouped under the first type's measurement: MXFP4 would print as the Qwen mix's."""
    assert not claim_line_names_the_rate_source(
        CACHE, _once(ROOM_COST, " && std::strcmp(other, from) == 0", ""))


def test_mutant_warn_names_the_measured_types_fails():
    """The unpriced WARN names the measured types, not the ones it refused for."""
    assert not claim_line_names_the_rate_source(
        _once(CACHE, UNMEASURED_LOCAL, UNMEASURED_LOCAL.replace("unmeasured_mask", "measured_mask")), ROOM_COST)


def test_mutant_planner_keeps_its_own_names_fails():
    """A second copy of the name helper in the planner: the line could name types the host test never checked."""
    mutant = norm(CACHE) + ' static std::string kv_room_type_names(uint64_t mask) { return ""; }'
    assert not claim_line_names_the_rate_source(mutant, ROOM_COST)


def test_mutant_line_drops_the_decision_fails():
    n = norm(CACHE)
    assert n.count(DECISION) == 2 and n.count(DECISION_ARGS) == 2
    mutant = n.replace(DECISION, "[PLACEMENT] KV context room on device %d").replace(DECISION_ARGS, "device_id,")
    assert not claim_room_line_states_the_decision(mutant)


def test_mutant_refused_note_always_printed_fails():
    """The parenthetical printed whatever was refused: a room held whole would read "(0.0 MB refused by cost)"."""
    unconditional = ('snprintf(refused_note, sizeof(refused_note), " (%.1f MB refused by cost)", '
                     'room.refused / mb);')
    assert not claim_room_line_states_the_decision(
        _once(CACHE, REFUSED_NOTE, 'char refused_note[64] = ""; ' + unconditional))


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


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
