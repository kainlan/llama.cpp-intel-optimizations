"""Source contract for llama.cpp-7fmt: without -c the load holds the KV room for the model's default context.

Host-only: reads sources, builds nothing, loads no model and touches no device.

The defect (B70, Qwen3.8 IQ3_XXS, no -c, master 8b5946b53): common passed llama_model_params::n_ctx_hint = 0 when no -c
was given, so the SYCL load logged "no KV context room ... no requested n_ctx reached the load" and held no room. The
context then opened at the training context (llama_context resolves n_ctx 0 to n_ctx_train, 262144), and the runtime
demoted 12 attention layers' KV to the host: 9.9 t/s decode, against 12.1 t/s with byte-identical output when the same
context is asked for with -c 262144 and the room is weighed and held. The default context is the context the caller
creates, so it is the requested context in the sense of llama.cpp-ak0p.

The fix this gate pins:
  - include/llama.h defines LLAMA_N_CTX_HINT_DEFAULT (0xFFFFFFFFu, distinct from 0, which keeps its meaning: no
    request, no room), and the n_ctx_hint field documents it;
  - common asks for it when -c is absent or 0 (common_model_params_to_llama), and for no room (0) when -c is negative;
    after a fit the hint is the context the fit left in cparams, and stays the default when the fit resolved none (the
    SYCL build's fit is a no-op), and a negative -c keeps no room through a fit; the draft model of speculative
    decoding asks for the context its draft context is created with;
  - src/llama-n-ctx-hint.h resolves the hint in one place: the default to n_ctx_train padded to 256, as llama_context
    resolves n_ctx 0 to n_ctx_train and pads it; 0 to no room; any other hint padded the same way; and it says
    whether the result is the default. src/llama-model.cpp calls it once, where it fills the SYCL inventory. No C/C++
    source outside include/llama.h, the resolver and common/common.cpp names the sentinel, so no backend sees it, and
    no backend statement compares the hint or the context against its value spelled another way (an all-ones
    literal, UINT32_MAX, std::numeric_limits<uint32_t>::max(), ~0u), and the backend's 0xFFFFFFFF literals are the
    six named slot and mask constants;
  - the inventory carries the provenance (n_ctx_context_is_default, appended, with the size pins moved), the backend
    copies it into its KV inputs next to n_ctx_context, and every KV context room line says when n_ctx_context is the
    model's default context, so a log reader sees why a training-context-sized room was weighed;
  - the provenance enters the room's decision in one direction only: for the default context, a routed-expert type
    with no measured CPU rate leaves the room unpriced (kv_room_expert_rate_for, kv-room-cost.hpp), and an unpriced
    room is refused for every layer, which is the placement a load with no requested context makes. The planner reads
    the flag for that call and for the line's note, and nowhere else in the backend reads it, so it can never make a
    room held. Commit 1 alone weighed the default context like a named one, which held GPT-OSS 20B's room on the B50
    and cost decode 2.4x (llama.cpp-7fmt comment c-ncj4).

Every claim is checked on COMMENT-STRIPPED, whitespace-normalized text with adjacent string literals joined, and has a
mutant that must make it fail. `--self-test` runs every mutant and names the claim each one turned red.

Pytest-style (module-level test_* functions): register with llama_test_pytest.
"""

import re
import sys
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


LLAMA_H_REL = "include/llama.h"
RESOLVER_REL = "src/llama-n-ctx-hint.h"
COMMON_REL = "common/common.cpp"
SPEC_REL = "common/speculative.cpp"
MODEL_REL = "src/llama-model.cpp"
CONTEXT_REL = "src/llama-context.cpp"
SYCL_H_REL = "ggml/include/ggml-sycl.h"
SYCL_REL = "ggml/src/ggml-sycl/ggml-sycl.cpp"
CACHE_REL = "ggml/src/ggml-sycl/unified-cache.cpp"
CACHE_HPP_REL = "ggml/src/ggml-sycl/unified-cache.hpp"
ROOM_COST_REL = "ggml/src/ggml-sycl/kv-room-cost.hpp"
FATTN_REL = "ggml/src/ggml-sycl/fattn.cpp"
BENCH_REL = "tools/llama-bench/llama-bench.cpp"
TOOL_REL = "tools/sycl-plan-dry-run/sycl-plan-dry-run.cpp"


def read_or_empty(rel: str) -> str:
    p = ROOT / rel
    return p.read_text() if p.is_file() else ""


LLAMA_H = read(LLAMA_H_REL)
RESOLVER = read_or_empty(RESOLVER_REL)
COMMON = read(COMMON_REL)
SPEC = read(SPEC_REL)
MODEL = read(MODEL_REL)
CONTEXT = read(CONTEXT_REL)
SYCL_H = read(SYCL_H_REL)
SYCL = read(SYCL_REL)
CACHE = read(CACHE_REL)
CACHE_HPP = read(CACHE_HPP_REL)
ROOM_COST = read_or_empty(ROOM_COST_REL)
TOOL = read(TOOL_REL)
# Every C/C++ source of the SYCL backend, by path from the repository root: where a second reader of the provenance
# would hide.
BACKEND_TREE = {
    p.relative_to(ROOT).as_posix(): p.read_text(errors="replace")
    for p in sorted((ROOT / "ggml/src/ggml-sycl").rglob("*"))
    if p.is_file() and p.suffix in (".c", ".cpp", ".h", ".hpp")
}

SENTINEL = "LLAMA_N_CTX_HINT_DEFAULT"
# Every C/C++ source outside tests/, by path from the repository root: where a second reader of the sentinel would
# hide. Only files that spell it are kept; the rest name it zero times by construction.
_SOURCE_DIRS = ("include", "src", "common", "tools", "examples", "ggml")
_SOURCE_SUFFIXES = (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".cu", ".cuh", ".m", ".mm")
TREE = {}
for _d in _SOURCE_DIRS:
    for _p in sorted((ROOT / _d).rglob("*")):
        if _p.is_file() and _p.suffix in _SOURCE_SUFFIXES:
            _text = _p.read_text(errors="replace")
            if SENTINEL in _text:
                TREE[_p.relative_to(ROOT).as_posix()] = _text


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


def comment_above(raw: str, anchor: str) -> str:
    """The contiguous // comment lines directly above the one line that contains `anchor`, joined."""
    lines = raw.splitlines()
    hits = [i for i, line in enumerate(lines) if anchor in line]
    if len(hits) != 1:
        return ""
    out = []
    i = hits[0] - 1
    while i >= 0 and lines[i].strip().startswith("//"):
        out.append(lines[i].strip()[2:].strip())
        i -= 1
    return " ".join(reversed(out))


# ---- the API --------------------------------------------------------------------------------------------------------


DEFINE = "#define LLAMA_N_CTX_HINT_DEFAULT 0xFFFFFFFFu"
FIELD = "uint32_t n_ctx_hint;"


def claim_api_defines_the_default(llama_h: str) -> bool:
    """llama.h defines the sentinel once, as a value 0 is not, says what it names (the model's default context,
    n_ctx_train), and the n_ctx_hint field's comment documents it next to 0's meaning."""
    above_define = comment_above(llama_h, DEFINE)
    above_field = comment_above(llama_h, FIELD)
    return (norm(llama_h).count(DEFINE) == 1 and "default context" in above_define and "n_ctx_train" in above_define
            and SENTINEL in above_field and "0 = unknown" in above_field and "n_ctx_train" in above_field)


# ---- common ---------------------------------------------------------------------------------------------------------


COMMON_SIG = "struct llama_model_params common_model_params_to_llama(common_params & params)"
# common maps -c to the hint in one function, which common_model_params_to_llama and every tool that takes -c ask, so
# the sentinel keeps one setter (llama.cpp-h187: the plan dry run mapped -c itself).
COMMON_HINT_SIG = "uint32_t common_n_ctx_hint(int32_t n_ctx)"
COMMON_HINT = "return n_ctx > 0 ? (uint32_t) n_ctx : n_ctx == 0 ? LLAMA_N_CTX_HINT_DEFAULT : 0;"
COMMON_HINT_CALL = "mparams.n_ctx_hint = common_n_ctx_hint(params.n_ctx);"
COMMON_FIT = "common_fit_params(params.model.path.c_str(), &mparams, &cparams,"
COMMON_FIT_REFRESH = "if (params.n_ctx >= 0 && cparams.n_ctx > 0) { mparams.n_ctx_hint = cparams.n_ctx; }"
COMMON_LOAD = "llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);"


def claim_common_asks_for_the_default(common: str) -> bool:
    """Without -c (or with -c 0, which the parser documents as "loaded from model") common asks for the model's
    default context, and a negative -c, which names no context, asks for no room. After a fit the hint is the context
    the fit left in cparams (it is what the context is created with); a fit that resolved none (cparams.n_ctx still
    0, as in the SYCL build, whose fit is a no-op) keeps the default instead of overwriting it with 0, and a negative
    -c keeps no room through a fit: cparams.n_ctx holds its unsigned wrap (-1 wraps to the sentinel's value), so the
    signed -c gates the refresh. Those are the only two setters."""
    n = norm(common)
    b = body(n, COMMON_SIG)
    hint = body(n, COMMON_HINT_SIG)
    return (bool(b) and bool(hint) and hint == "{ " + COMMON_HINT + " }" and COMMON_HINT_CALL in b
            and ordered(n, COMMON_FIT, COMMON_FIT_REFRESH, COMMON_LOAD)
            and n.count("mparams.n_ctx_hint =") == 2 and n.count(SENTINEL) == 1)


SPEC_SIG = "common_speculative_init_result::common_speculative_init_result("
SPEC_STEPS = ("auto mparams = common_model_params_to_llama(params);", "cparams.n_ctx = llama_n_ctx(ctx_tgt);",
              "mparams.n_ctx_hint = cparams.n_ctx;",
              "llama_model * model_dft = llama_model_load_from_file(model_path.c_str(), mparams);")


def claim_draft_asks_for_its_context(spec: str) -> bool:
    """The draft model's load asks for the context its draft context is created with (the target's), not for the
    draft model's own default, and asks before it loads."""
    n = norm(spec)
    at = n.find(SPEC_SIG)
    return at >= 0 and ordered(n[at:], *SPEC_STEPS) and n.count("mparams.n_ctx_hint =") == 1


# ---- the resolution -------------------------------------------------------------------------------------------------


RESOLVE_SIG = ("static inline llama_n_ctx_hint_resolution llama_n_ctx_hint_resolve(uint32_t n_ctx_hint, "
               "uint32_t n_ctx_train)")
DEFAULT_BRANCH_RE = re.compile(
    r"if \(n_ctx_hint == LLAMA_N_CTX_HINT_DEFAULT\) \{ const uint32_t n_ctx_context = GGML_PAD\(n_ctx_train, (\d+)\);"
    r" return \{ n_ctx_context, n_ctx_context != 0 \}; \}")
EXPLICIT_RE = re.compile(r"return \{ n_ctx_hint != 0 \? GGML_PAD\(n_ctx_hint, (\d+)\) : 0, false \};")
CONTEXT_DEFAULT = "cparams.n_ctx = params.n_ctx == 0 ? hparams.n_ctx_train : params.n_ctx;"
CONTEXT_PAD_RE = re.compile(r"cparams\.n_ctx = GGML_PAD\(cparams\.n_ctx, (\d+)\);")


def claim_resolver_resolves_the_default(resolver: str, context: str) -> bool:
    """The default resolves first, to n_ctx_train padded with the constant llama_context pads n_ctx with after it
    resolves n_ctx 0 to n_ctx_train, and reports itself as the default whenever it holds a room; any other hint is
    the caller's (never the default), 0 is no room, and the rest are padded the same way."""
    r = body(norm(resolver), RESOLVE_SIG)
    c = norm(context)
    default = DEFAULT_BRANCH_RE.search(r)
    explicit = EXPLICIT_RE.search(r)
    pad = CONTEXT_PAD_RE.findall(c)
    return (bool(r) and default is not None and explicit is not None and default.start() < explicit.start()
            and len(DEFAULT_BRANCH_RE.findall(r)) == 1 and len(EXPLICIT_RE.findall(r)) == 1
            and CONTEXT_DEFAULT in c and len(pad) == 1 and default.group(1) == pad[0] and explicit.group(1) == pad[0])


POPULATE_SIG = "static void llama_model_sycl_populate_inventory(ggml_sycl_tensor_inventory & inventory,"
RESOLVER_INCLUDE = '#include "llama-n-ctx-hint.h"'
RESOLVE_CALL = "const llama_n_ctx_hint_resolution ctx_room = llama_n_ctx_hint_resolve(n_ctx_hint, hparams.n_ctx_train);"
PICK_CTX = "inventory.n_ctx_context = ctx_room.n_ctx_context;"
PICK_DEFAULT = "inventory.n_ctx_context_is_default = ctx_room.is_default;"


def claim_loader_resolves_once(model: str) -> bool:
    """llama-model.cpp resolves the hint in exactly one place, where it fills the inventory, and copies both the
    context and its provenance; it neither resolves the hint itself nor calls the resolver anywhere else."""
    m = norm(model)
    pop = body(m, POPULATE_SIG)
    return (RESOLVER_INCLUDE in m and bool(pop) and m.count("llama_n_ctx_hint_resolve(") == 1
            and ordered(pop, RESOLVE_CALL, PICK_CTX, PICK_DEFAULT)
            and pop.count("inventory.n_ctx_context =") == 1 and pop.count("inventory.n_ctx_context_is_default =") == 1)


SENTINEL_OWNERS = {LLAMA_H_REL: 1, RESOLVER_REL: 1, COMMON_REL: 1}
# The backend's sources outside its tests, and the public SYCL header: where the sentinel's value could be read without
# its name.
VALUE_TREE = {rel: text for rel, text in BACKEND_TREE.items() if not rel.startswith("ggml/src/ggml-sycl/tests/")}
VALUE_TREE[SYCL_H_REL] = SYCL_H
# Every spelling of 0xFFFFFFFF a backend statement could compare the hint against.
ALL_ONES_RE = re.compile(r"\b0[xX][fF]{8}(?![0-9a-fA-F])|\bUINT32_MAX\b|std::numeric_limits<uint32_t>::max\(\)"
                         r"|~0[uU]\b|~uint32_t\b")
HEX_ALL_ONES_RE = re.compile(r"\b0[xX][fF]{8}(?![0-9a-fA-F])")
CTX_NAMES_RE = re.compile(r"\bn_ctx_(?:hint|context)\b")
# The backend's 0xFFFFFFFF literals, by file and the text before the literal: three "no model slot" values, the
# pending-range mask of every term, and topk's sign flip. A sixth constant with this value would be a way to read the
# sentinel by another name.
ALL_ONES_SITES = {
    (SYCL_H_REL, "#define GGML_SYCL_MODEL_SLOT_NONE"),
    ("ggml/src/ggml-sycl/layer-stream-owner.hpp", "static constexpr uint32_t layer_stream_owner_slot_none ="),
    ("ggml/src/ggml-sycl/moe-discovery-state.hpp", "static constexpr uint32_t moe_owner_slot_none ="),
    ("ggml/src/ggml-sycl/pending-range.hpp", "constexpr pending_term_mask PENDING_TERM_ALL ="),
    ("ggml/src/ggml-sycl/topk.cpp", "return bits ^ ((bits >> 31) ?"),
    (CACHE_HPP_REL, "constexpr uint32_t MODEL_SLOT_NONE ="),
}


def statements(text: str) -> list[str]:
    """The comment-stripped, whitespace-normalized statements of a source: text between ';', '{' and '}', with each
    preprocessor line a statement of its own."""
    lines = [";" + line + ";" if line.lstrip().startswith("#") else line for line in strip_comments(text).splitlines()]
    flat = re.sub(r"\s+", " ", "\n".join(lines))
    return [s.strip() for s in re.split(r"[;{}]", flat) if s.strip()]


def backend_never_spells_the_value(value_tree: dict[str, str]) -> bool:
    """No backend statement names the hint or the context next to an all-ones value, and every 0xFFFFFFFF literal is
    one of the named sites, each found once."""
    found = []
    for rel, text in value_tree.items():
        for s in statements(text):
            if ALL_ONES_RE.search(s) and CTX_NAMES_RE.search(s):
                return False
            for m in HEX_ALL_ONES_RE.finditer(s):
                found.append((rel, s[:m.start()].strip()))
    return sorted(found) == sorted(ALL_ONES_SITES)


def claim_sentinel_has_one_reader(tree: dict[str, str], value_tree: dict[str, str]) -> bool:
    """Outside tests, the sentinel is spelled by its definition, the one comparison in the resolver and the one setter
    in common, and nowhere else: no backend, tool or second libllama site reads or sets it (one fact, one source). Nor
    does the backend read it by value: no statement there compares n_ctx_hint or n_ctx_context with 0xFFFFFFFF in any
    spelling, and the backend's 0xFFFFFFFF literals are the named slot and mask constants. A constant defined from
    UINT32_MAX and compared later is not seen; the inventory carries the provenance, so the backend has no reason to."""
    counts = {rel: norm(text).count(SENTINEL) for rel, text in tree.items()}
    counts = {rel: c for rel, c in counts.items() if c > 0}
    return counts == SENTINEL_OWNERS and backend_never_spells_the_value(value_tree)


TOOL_MAIN_SIG = "int main(int argc, char ** argv)"
TOOL_HINT = "params.n_ctx_hint = common_n_ctx_hint(args.n_ctx);"
# Every write to the hint: plain and compound assignment, and increment or decrement either side of it.
HINT_WRITE_RE = re.compile(r"\bn_ctx_hint\s*(?:<<|>>|[-+*/%|&^])?=(?!=)"
                           r"|\bn_ctx_hint\s*(?:\+\+|--)"
                           r"|(?:\+\+|--)\s*(?:\w+\s*(?:\.|->)\s*)*n_ctx_hint\b")


def claim_tool_asks_common(tool: str) -> bool:
    """The plan dry run keeps -c as given and maps it to the load's hint by asking common_n_ctx_hint, in main, and
    writes the hint nowhere else in the tool: a raw cast would plan no room for the default context (0 is not the
    sentinel) and a negative -c would plan a context of four billion cells (llama.cpp-h187). The args test checks the
    mapping only by calling common, so this is the tool's one witness."""
    n = norm(tool)
    main = body(n, TOOL_MAIN_SIG)
    return bool(main) and TOOL_HINT in main and len(HINT_WRITE_RE.findall(n)) == 1


# ---- the backend ----------------------------------------------------------------------------------------------------


INVENTORY_STRUCT = "struct ggml_sycl_tensor_inventory {"
# libllama's resolved storage inputs follow the provenance, in its padding (test-sycl-mapped-loader-source pins their
# offsets), the named micro-batch follows them (test-sycl-explicit-ubatch-placement-source), and the lazy tables are
# appended after it (th32 §4.1).
INV_TAIL = ("uint32_t kv_layer_count; uint32_t n_ctx_context; const uint32_t * kv_idx_k_width_per_layer; "
            "bool n_ctx_context_is_default; bool use_mmap_resolved; bool use_mlock_resolved; "
            "bool host_weight_layout_cpu; uint32_t n_ubatch_context; "
            "const struct ggml_sycl_lazy_table_info * lazy_tables; uint32_t lazy_table_count; }")
SIZE_PINS = (
    "static_assert(sizeof(ggml_sycl_tensor_inventory) == 208,",
    "static_assert(offsetof(ggml_sycl_tensor_inventory, n_ctx_context_is_default) == 184,",
)
FILL_SIG = ("static void populate_inventory_globals(ggml_backend_sycl_context * ctx, "
            "const ggml_sycl_tensor_inventory * inventory)")
# The stage fills g_placement_kv_info through the one KV intake the plan dry run also uses (llama.cpp-4zup), so the copy
# is written once, in the intake, over the KV inputs it is handed.
INTAKE_SIG = ("static void ggml_sycl_inventory_fill_kv_info(ggml_sycl::placement_kv_info & kv, "
              "const ggml_sycl_tensor_inventory * inventory)")
INTAKE_CALL = "ggml_sycl_inventory_fill_kv_info(g_placement_kv_info, inventory);"
COPY_CTX = "kv.n_ctx_context = inventory->n_ctx_context;"
COPY_DEFAULT = "kv.n_ctx_context_is_default = inventory->n_ctx_context_is_default;"
KV_FIELD = "bool n_ctx_context_is_default = false;"


def claim_backend_carries_the_source(sycl_h: str, sycl: str, cache_hpp: str) -> bool:
    """The inventory ends with the provenance, appended after the indexer widths with the size pins moved to match;
    the backend copies it once, next to the context, into a KV input that defaults to a named context: the copy is in
    the KV intake, and the stage hands the intake g_placement_kv_info."""
    h = norm(sycl_h)
    at = h.find(INVENTORY_STRUCT)
    inv = body(h[at:], "struct ggml_sycl_tensor_inventory") if at >= 0 else ""
    s = norm(sycl)
    fill = body(s, FILL_SIG)
    intake = body(s, INTAKE_SIG)
    hpp = norm(cache_hpp)
    kv_at = hpp.find("struct placement_kv_info {")
    kv = body(hpp[kv_at:], "struct placement_kv_info") if kv_at >= 0 else ""
    return (inv.endswith(INV_TAIL) and all(p in s for p in SIZE_PINS) and bool(fill) and INTAKE_CALL in fill
            and bool(intake) and ordered(intake, COPY_CTX, COPY_DEFAULT) and s.count(COPY_DEFAULT) == 1
            and KV_FIELD in kv)


ROOM_LOG_SIG = "static void log_kv_context_room(const kv_context_room & room,"
NOTE = " (n_ctx_context is the model's default, training context: the caller named no n_ctx)"
SOURCE_NOTE = f'const char * ctx_source = kv_info.n_ctx_context_is_default ? "{NOTE}" : "";'
ROOM_EMPTY = "if (room.n_layers == 0) { return; }"
UNPRICED = "if (!room.expert_rate.priced && room.n_held == 0) {"
UNPRICED_ARGS = "kv_info.n_ctx_context, kv_info.n_ctx, ctx_source, kv_info.cpu_expert_gbps);"
ALL_REFUSED = "if (room.n_held == 0) {"
REFUSED_SLOT = "for n_ctx_context=%u over planner n_ctx=%u%s is held"
REFUSED_ARGS = "room.refused / mb, kv_info.n_ctx_context, kv_info.n_ctx, ctx_source,"
HELD_SLOT = "for n_ctx_context=%u over planner n_ctx=%u%s, before the routed experts%s;"
HELD_ARGS = "kv_info.n_ctx_context, kv_info.n_ctx, ctx_source, short_note,"


def claim_room_lines_name_the_default(cache: str) -> bool:
    """Every KV context room line (the unpriced WARN, the all-refused WARN and the held line) prints, right after the
    context it weighed, a note that says it is the model's default (training) context, chosen by the provenance flag
    alone; the note is empty for a context the caller named, and the no-room line (n_ctx_context 0) is left as it
    was."""
    b = body(norm(cache), ROOM_LOG_SIG)
    if not b or b.count(SOURCE_NOTE) != 1 or not ordered(b, ROOM_EMPTY, SOURCE_NOTE, UNPRICED, ALL_REFUSED):
        return False
    up = b.find(UNPRICED)
    unpriced = body(b[up:], UNPRICED[:-2])
    at = b.find(ALL_REFUSED)
    refused = body(b[at:], ALL_REFUSED[:-2])
    rest = b[at + len(ALL_REFUSED) - 1 + len(refused):] if refused else ""
    return (bool(unpriced) and REFUSED_SLOT in unpriced and UNPRICED_ARGS in unpriced
            and bool(rest) and REFUSED_SLOT in refused and REFUSED_ARGS in refused
            and HELD_SLOT in rest and HELD_ARGS in rest and b.count("ctx_source") == 4)


FLAG = "n_ctx_context_is_default"
PLAN_SIG = "placement_plan compute_placement_plan(const std::vector<placement_tensor_info> & tensor_inventory,"
RATE_CALL = ("const kv_room_expert_rate room_expert_rate = kv_room_expert_rate_for(room_expert_types, "
             "kv_info.cpu_expert_gbps, kv_info.cpu_expert_gbps_set, kv_info.n_ctx_context_is_default);")
RATE_SIG = ("static inline kv_room_expert_rate kv_room_expert_rate_for(const kv_room_expert_types & types, "
            "double cpu_expert_gbps, bool cpu_expert_gbps_set, bool n_ctx_context_is_default)")
RATE_RULE = "if (types.unmeasured_mask != 0 && n_ctx_context_is_default) { rate.priced = false; return rate; }"
WEIGH_SIG = "static inline kv_room_cost_terms kv_room_weigh(const kv_room_cost_inputs & in)"
HOLD_IFF = "cost.held = !moe || (in.priced && cost.host_attn_us > cost.expert_us);"
# The backend files that may spell the flag, and how often (comments stripped; tests/ excluded): the copy (both sides)
# and the offset pin with its message, the KV input, the planner's two reads (the line's note and the rate call), and
# the rate rule's parameter and its one test.
FLAG_OWNERS = {SYCL_REL: 4, CACHE_HPP_REL: 1, CACHE_REL: 2, ROOM_COST_REL: 2}


def claim_default_only_refuses(tree: dict[str, str]) -> bool:
    """The provenance reaches the room's decision through one call and one rule, and the rule can only refuse: the
    planner passes it to kv_room_expert_rate_for once, where a default context with an unmeasured expert type
    leaves the room unpriced and returns, and kv_room_weigh lets an unpriced room only lose a hold. No other backend
    source reads it."""
    backend = {rel: text for rel, text in tree.items() if not rel.startswith("ggml/src/ggml-sycl/tests/")}
    counts = {rel: norm(text).count(FLAG) for rel, text in backend.items()}
    counts = {rel: c for rel, c in counts.items() if c > 0}
    if counts != FLAG_OWNERS:
        return False
    cache = norm(tree[CACHE_REL])
    cost = norm(tree[ROOM_COST_REL])
    rate = body(cost, RATE_SIG)
    return (RATE_CALL in body(cache, PLAN_SIG) and SOURCE_NOTE in cache and rate.count(FLAG) == 1
            and RATE_RULE in rate and HOLD_IFF in body(cost, WEIGH_SIG) and body(cost, WEIGH_SIG).count("priced") == 1)


# ---- the claims hold on the tree ------------------------------------------------------------------------------------


def test_api_defines_the_default():
    assert claim_api_defines_the_default(LLAMA_H)


def test_common_asks_for_the_default():
    assert claim_common_asks_for_the_default(COMMON)


def test_draft_asks_for_its_context():
    assert claim_draft_asks_for_its_context(SPEC)


def test_resolver_resolves_the_default():
    assert claim_resolver_resolves_the_default(RESOLVER, CONTEXT)


def test_loader_resolves_once():
    assert claim_loader_resolves_once(MODEL)


def test_sentinel_has_one_reader():
    assert claim_sentinel_has_one_reader(TREE, VALUE_TREE)


def test_tool_asks_common():
    assert claim_tool_asks_common(TOOL)


def test_backend_carries_the_source():
    assert claim_backend_carries_the_source(SYCL_H, SYCL, CACHE_HPP)


def test_room_lines_name_the_default():
    assert claim_room_lines_name_the_default(CACHE)


def test_default_only_refuses():
    assert claim_default_only_refuses(BACKEND_TREE)


# ---- mutants: each must turn its claim red --------------------------------------------------------------------------


def _once(text: str, old: str, new: str) -> str:
    """`text` with the one occurrence of `old` (matched on normalized text) replaced by `new`."""
    n = norm(text)
    assert n.count(old) == 1, f"mutant anchor must match exactly once: {old!r} x{n.count(old)}"
    return n.replace(old, new, 1)


def _raw_once(text: str, old: str, new: str) -> str:
    """`text` with the one raw occurrence of `old` replaced (for claims that read comments)."""
    assert text.count(old) == 1, f"mutant anchor must match exactly once: {old!r} x{text.count(old)}"
    return text.replace(old, new, 1)


def _tree_with(rel: str, text: str) -> dict[str, str]:
    tree = dict(TREE)
    tree[rel] = text
    return tree


def test_mutant_api_without_the_define_fails():
    assert not claim_api_defines_the_default(_raw_once(LLAMA_H, DEFINE + "\n", ""))


def test_mutant_api_default_is_zero_fails():
    """A default spelled 0 cannot be told from no request: every run without -c would hold no room again."""
    assert not claim_api_defines_the_default(_raw_once(LLAMA_H, DEFINE, "#define LLAMA_N_CTX_HINT_DEFAULT 0"))


def test_mutant_api_field_undocumented_fails():
    """The field's comment no longer names the sentinel: a caller reading the struct sees only 0's meaning."""
    above = comment_above(LLAMA_H, FIELD)
    assert SENTINEL in above
    lines = [line for line in LLAMA_H.splitlines(keepends=True)]
    mutant = "".join(line.replace(SENTINEL, "a sentinel") if line.strip().startswith("//") and SENTINEL in line
                     and "#define" not in line else line for line in lines)
    assert not claim_api_defines_the_default(mutant)


def test_mutant_common_keeps_zero_without_c_fails():
    """master's line: without -c the hint is 0 and no room is held for the context llama then creates."""
    assert not claim_common_asks_for_the_default(
        _once(COMMON, COMMON_HINT, "return n_ctx > 0 ? (uint32_t) n_ctx : 0;"))


def test_mutant_common_negative_c_asks_for_the_default_fails():
    """A negative -c, which names no context, asks for the default: the load would hold the training context's room
    for a context llama_context cannot create."""
    assert not claim_common_asks_for_the_default(
        _once(COMMON, COMMON_HINT, "return n_ctx > 0 ? (uint32_t) n_ctx : LLAMA_N_CTX_HINT_DEFAULT;"))


def test_mutant_common_load_skips_the_mapping_fails():
    """common_model_params_to_llama sets the hint some other way: the mapping is still right, and unasked."""
    assert not claim_common_asks_for_the_default(
        _once(COMMON, COMMON_HINT_CALL, "mparams.n_ctx_hint = (uint32_t) params.n_ctx;"))


def test_mutant_common_fit_refresh_unguarded_fails():
    """Unguarded, a fit that resolved nothing (the SYCL build's no-op fit leaves cparams.n_ctx 0 without -c)
    overwrites the default with 0, and the load holds no room again."""
    assert not claim_common_asks_for_the_default(
        _once(COMMON, COMMON_FIT_REFRESH, "mparams.n_ctx_hint = cparams.n_ctx;"))


def test_mutant_common_fit_refresh_on_c_only_fails():
    """Guarded on -c (master's guard), a fit that resolves the default context itself (a shrunk n_ctx_train) is not
    carried, so the load would plan for n_ctx_train while the context opens at the fit's size."""
    assert not claim_common_asks_for_the_default(
        _once(COMMON, COMMON_FIT_REFRESH, "if (params.n_ctx > 0) { mparams.n_ctx_hint = cparams.n_ctx; }"))


def test_mutant_common_fit_refresh_unsigned_only_fails():
    """Guarded on the unsigned cparams.n_ctx alone: with -fit on and -c -1, cparams.n_ctx is the wrap of -1, which is
    LLAMA_N_CTX_HINT_DEFAULT, so the hint becomes the default and the load holds the default context's room for a -c
    that asked for none (any other negative -c becomes a named context of nearly 2^32 cells)."""
    assert not claim_common_asks_for_the_default(
        _once(COMMON, COMMON_FIT_REFRESH, "if (cparams.n_ctx > 0) { mparams.n_ctx_hint = cparams.n_ctx; }"))


def test_mutant_common_second_setter_fails():
    """A second site in common sets the sentinel: two sources for the one rule."""
    assert not claim_common_asks_for_the_default(
        _once(COMMON, COMMON_LOAD, "mparams.n_ctx_hint = LLAMA_N_CTX_HINT_DEFAULT; " + COMMON_LOAD))


def test_mutant_draft_keeps_its_own_default_fails():
    """The draft load asks for the draft model's own training context while its context opens at the target's."""
    assert not claim_draft_asks_for_its_context(_once(SPEC, "mparams.n_ctx_hint = cparams.n_ctx;", ""))


def test_mutant_draft_hint_after_the_load_fails():
    n = _once(SPEC, "mparams.n_ctx_hint = cparams.n_ctx;", "")
    load = SPEC_STEPS[-1]
    assert n.count(load) == 1
    assert not claim_draft_asks_for_its_context(n.replace(load, load + " mparams.n_ctx_hint = cparams.n_ctx;", 1))


def _resolver_default_branch() -> str:
    m = DEFAULT_BRANCH_RE.search(norm(RESOLVER))
    assert m is not None, "the resolver's default branch was not found"
    return m.group(0)


def test_mutant_resolver_reads_the_default_as_a_context_fails():
    """No default branch: 0xFFFFFFFF padded as a literal context wraps to 0, so no room (the defect)."""
    assert not claim_resolver_resolves_the_default(_once(RESOLVER, _resolver_default_branch(), ""), CONTEXT)


def test_mutant_resolver_pads_the_default_differently_fails():
    branch = _resolver_default_branch()
    assert not claim_resolver_resolves_the_default(
        _once(RESOLVER, branch, branch.replace("GGML_PAD(n_ctx_train, 256)", "GGML_PAD(n_ctx_train, 512)")), CONTEXT)


def test_mutant_resolver_default_unpadded_fails():
    """The room for n_ctx_train unpadded falls short of the context llama_context creates for n_ctx 0."""
    branch = _resolver_default_branch()
    assert not claim_resolver_resolves_the_default(
        _once(RESOLVER, branch, branch.replace("GGML_PAD(n_ctx_train, 256)", "n_ctx_train")), CONTEXT)


def test_mutant_resolver_default_not_reported_fails():
    """The default is resolved but reported as a named context: the room's line cannot say why it is so large."""
    branch = _resolver_default_branch()
    assert not claim_resolver_resolves_the_default(
        _once(RESOLVER, branch, branch.replace("n_ctx_context != 0 }", "false }")), CONTEXT)


def test_mutant_resolver_explicit_reported_as_default_fails():
    assert not claim_resolver_resolves_the_default(
        _once(RESOLVER, ": 0, false };", ": 0, true };"), CONTEXT)


def test_mutant_resolver_explicit_after_default_order_fails():
    """The explicit branch first: the sentinel is padded as a literal context before the default branch is reached."""
    n = norm(RESOLVER)
    branch = _resolver_default_branch()
    explicit = EXPLICIT_RE.search(n).group(0)
    swapped = n.replace(branch, "@@B@@", 1).replace(explicit, "@@E@@", 1)
    swapped = swapped.replace("@@B@@", explicit, 1).replace("@@E@@", branch, 1)
    assert not claim_resolver_resolves_the_default(swapped, CONTEXT)


def test_mutant_context_pads_differently_fails():
    """The comparison is real: llama_context changing its padding turns the claim red until the resolver follows."""
    assert not claim_resolver_resolves_the_default(
        RESOLVER, _once(CONTEXT, "cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);",
                        "cparams.n_ctx = GGML_PAD(cparams.n_ctx, 512);"))


def test_mutant_context_default_not_the_training_context_fails():
    """llama_context resolving n_ctx 0 to something else would make n_ctx_train the wrong default."""
    assert not claim_resolver_resolves_the_default(
        RESOLVER, _once(CONTEXT, CONTEXT_DEFAULT, "cparams.n_ctx = params.n_ctx == 0 ? 4096 : params.n_ctx;"))


def test_mutant_loader_resolves_itself_fails():
    """master's own resolution in llama-model.cpp: a second source for the rule, and the default reads as 0."""
    m = _once(MODEL, RESOLVE_CALL, "")
    m = m.replace(PICK_CTX, "inventory.n_ctx_context = n_ctx_hint != 0 ? GGML_PAD(n_ctx_hint, 256) : 0;", 1)
    assert not claim_loader_resolves_once(m)


def test_mutant_loader_drops_the_provenance_fails():
    assert not claim_loader_resolves_once(_once(MODEL, PICK_DEFAULT, ""))


def test_mutant_loader_resolves_twice_fails():
    """A second call elsewhere in llama-model.cpp, here in the placement envelope."""
    env_sig = "static ggml_sycl_placement_envelope llama_model_sycl_make_placement_envelope()"
    n = norm(MODEL)
    at = n.find(env_sig)
    assert at >= 0
    k = n.find("{", at) + 1
    mutant = n[:k] + " (void) llama_n_ctx_hint_resolve(0, 0);" + n[k:]
    assert not claim_loader_resolves_once(mutant)


def test_mutant_backend_reads_the_sentinel_fails():
    """The backend compares the sentinel itself: the fact would have a second source past the dlopen boundary."""
    rel = CACHE_REL
    text = CACHE + "\nstatic bool ggml_sycl_ctx_is_default(uint32_t n) { return n == LLAMA_N_CTX_HINT_DEFAULT; }\n"
    assert not claim_sentinel_has_one_reader(_tree_with(rel, text), VALUE_TREE)


def test_mutant_model_reads_the_sentinel_fails():
    text = MODEL + "\nstatic bool llama_model_hint_is_default(uint32_t n) { return n == LLAMA_N_CTX_HINT_DEFAULT; }\n"
    assert not claim_sentinel_has_one_reader(_tree_with(MODEL_REL, text), VALUE_TREE)


def test_mutant_bench_asks_for_the_default_fails():
    """llama-bench always names its test's context; it must not start asking for the default."""
    bench = read(BENCH_REL)
    assert "mparams.n_ctx_hint = n_ctx();" in bench
    text = bench.replace("mparams.n_ctx_hint = n_ctx();", "mparams.n_ctx_hint = LLAMA_N_CTX_HINT_DEFAULT;", 1)
    assert not claim_sentinel_has_one_reader(_tree_with(BENCH_REL, text), VALUE_TREE)


def test_mutant_resolver_without_the_sentinel_fails():
    """The resolver stops spelling the sentinel (it resolves nothing): the owner set no longer matches."""
    tree = dict(TREE)
    tree.pop(RESOLVER_REL, None)
    assert not claim_sentinel_has_one_reader(tree, VALUE_TREE)


def _value_tree_with(rel: str, text: str) -> dict[str, str]:
    tree = dict(VALUE_TREE)
    tree[rel] = text
    return tree


def test_mutant_tool_casts_c_itself_fails():
    """The tool hands -c to the load as a raw cast instead of asking common."""
    assert not claim_tool_asks_common(_once(TOOL, TOOL_HINT, "params.n_ctx_hint = (uint32_t) args.n_ctx;"))


def test_mutant_tool_overwrites_the_hint_fails():
    """The tool asks common, then sets the hint a second time."""
    assert not claim_tool_asks_common(_once(TOOL, TOOL_HINT, TOOL_HINT + " params.n_ctx_hint = 0;"))


def test_mutant_tool_adjusts_the_hint_after_asking_fails():
    """The tool asks common, then adjusts the hint with a compound assignment."""
    assert not claim_tool_asks_common(_once(TOOL, TOOL_HINT, TOOL_HINT + " params.n_ctx_hint += 1;"))


def test_mutant_tool_increments_the_hint_after_asking_fails():
    """The tool asks common, then increments the hint."""
    assert not claim_tool_asks_common(_once(TOOL, TOOL_HINT, TOOL_HINT + " params.n_ctx_hint++;"))


def test_mutant_tool_decrements_the_hint_before_use_fails():
    """The tool asks common, then pre-decrements the hint."""
    assert not claim_tool_asks_common(_once(TOOL, TOOL_HINT, TOOL_HINT + " --params.n_ctx_hint;"))


def test_mutant_tool_asks_outside_main_fails():
    """The call moves out of main into a helper nothing calls, and main leaves the default hint."""
    text = _once(TOOL, TOOL_HINT, "") + "\nstatic void h187_mutant(llama_model_params & params, " \
        "const sycl_plan_dry_run_args & args) { " + TOOL_HINT + " }\n"
    assert not claim_tool_asks_common(text)


def test_mutant_backend_compares_the_context_to_all_ones_fails():
    """The planner recognises the default by its value: the inventory's flag would have a second source."""
    text = CACHE + ("\nstatic bool g_7fmt_mutant(const ggml_sycl::placement_kv_info & k) "
                    "{ return k.n_ctx_context == 0xFFFFFFFFu; }\n")
    assert not claim_sentinel_has_one_reader(TREE, _value_tree_with(CACHE_REL, text))


def test_mutant_backend_compares_the_hint_to_uint32_max_fails():
    """The same comparison spelled UINT32_MAX, in the backend TU that copies the inventory in."""
    text = SYCL + ("\nstatic bool g_7fmt_mutant(const ggml_sycl_tensor_inventory * inv) "
                   "{ return inv->n_ctx_context ==\n    UINT32_MAX; }\n")
    assert not claim_sentinel_has_one_reader(TREE, _value_tree_with(SYCL_REL, text))


def test_mutant_backend_names_the_value_fails():
    """A new all-ones constant, which a later comparison could read the sentinel by."""
    text = CACHE_HPP + "\nconstexpr uint32_t PLACEMENT_N_CTX_DEFAULT = 0xFFFFFFFFu;\n"
    assert not claim_sentinel_has_one_reader(TREE, _value_tree_with(CACHE_HPP_REL, text))


def test_mutant_backend_drops_the_provenance_fails():
    assert not claim_backend_carries_the_source(SYCL_H, _once(SYCL, COPY_DEFAULT, ""), CACHE_HPP)


def test_mutant_stage_skips_the_intake_fails():
    """The stage stops handing g_placement_kv_info to the intake: the copy still exists, and reaches no KV input."""
    assert not claim_backend_carries_the_source(SYCL_H, _once(SYCL, INTAKE_CALL, ""), CACHE_HPP)


def test_mutant_inventory_without_the_field_fails():
    assert not claim_backend_carries_the_source(
        _once(SYCL_H, "bool n_ctx_context_is_default; bool use_mmap_resolved;", "bool use_mmap_resolved;"),
        SYCL, CACHE_HPP)


def test_mutant_size_pin_not_moved_fails():
    assert not claim_backend_carries_the_source(
        SYCL_H, _once(SYCL, SIZE_PINS[0], "static_assert(sizeof(ggml_sycl_tensor_inventory) == 184,"), CACHE_HPP)


def test_mutant_provenance_inserted_mid_struct_fails():
    """Inserted before the indexer widths instead of appended: every pinned offset after it moves."""
    h = _once(SYCL_H, "bool n_ctx_context_is_default; bool use_mmap_resolved;", "bool use_mmap_resolved;")
    tail = "uint32_t n_ctx_context; const uint32_t * kv_idx_k_width_per_layer;"
    assert h.count(tail) == 1
    h = h.replace(tail, tail.replace("n_ctx_context;", "n_ctx_context; bool n_ctx_context_is_default;"), 1)
    assert not claim_backend_carries_the_source(h, SYCL, CACHE_HPP)


def test_mutant_kv_input_defaults_to_the_default_fails():
    """A KV input that defaults to the default context: every test fixture and stale producer would read as one."""
    assert not claim_backend_carries_the_source(
        SYCL_H, SYCL, _once(CACHE_HPP, KV_FIELD, "bool n_ctx_context_is_default = true;"))


def test_mutant_held_line_drops_the_note_fails():
    n = _once(CACHE, HELD_SLOT, HELD_SLOT.replace("n_ctx=%u%s,", "n_ctx=%u,"))
    assert not claim_room_lines_name_the_default(n.replace(HELD_ARGS, HELD_ARGS.replace(" ctx_source,", ""), 1))


def _log_block_mutant(head: str, old: str, new: str) -> str:
    """CACHE (normalized) with `old` replaced by `new` once inside the room line's block that opens with `head`."""
    n = norm(CACHE)
    b = body(n, ROOM_LOG_SIG)
    at = b.find(head)
    assert at >= 0, f"block not found: {head!r}"
    block = body(b[at:], head[:-2])
    assert block.count(old) == 1 and n.count(block) == 1, f"mutant anchor must match once in the block: {old!r}"
    return n.replace(block, block.replace(old, new, 1), 1)


def test_mutant_refused_line_drops_the_note_fails():
    n = _log_block_mutant(ALL_REFUSED, REFUSED_SLOT, REFUSED_SLOT.replace("n_ctx=%u%s is", "n_ctx=%u is"))
    assert not claim_room_lines_name_the_default(n.replace(REFUSED_ARGS, REFUSED_ARGS.replace(" ctx_source,", ""), 1))


def test_mutant_unpriced_line_drops_the_note_fails():
    """The unpriced WARN, which only a default context can print, says nothing about where its context came from."""
    n = _log_block_mutant(UNPRICED, REFUSED_SLOT, REFUSED_SLOT.replace("n_ctx=%u%s is", "n_ctx=%u is"))
    assert not claim_room_lines_name_the_default(n.replace(UNPRICED_ARGS, UNPRICED_ARGS.replace(" ctx_source,", ""), 1))


def _backend_with(rel: str, text: str) -> dict[str, str]:
    tree = dict(BACKEND_TREE)
    tree[rel] = text
    return tree


def test_mutant_default_holds_every_room_fails():
    """The default context holds its room whatever the cost: commit 1 alone, which held GPT-OSS 20B's room on the B50
    (9.9 against 23.5 t/s)."""
    mutant = _once(CACHE, "if (cost.held) {", "if (cost.held || kv_info.n_ctx_context_is_default) {")
    assert not claim_default_only_refuses(_backend_with(CACHE_REL, mutant))


def test_mutant_rule_never_fires_fails():
    """The planner passes false for the provenance: an unmeasured type is weighed at the fallback for the default
    context too."""
    mutant = _once(CACHE, RATE_CALL, RATE_CALL.replace("kv_info.n_ctx_context_is_default);", "false);"))
    assert not claim_default_only_refuses(_backend_with(CACHE_REL, mutant))


def test_mutant_default_prices_instead_of_refusing_fails():
    """For the default context an unmeasured type is priced at a made-up rate instead of refusing the room."""
    mutant = _once(ROOM_COST, RATE_RULE, RATE_RULE.replace("rate.priced = false;", "rate.gbps = 1e9;"))
    assert not claim_default_only_refuses(_backend_with(ROOM_COST_REL, mutant))


def test_mutant_unpriced_room_forced_held_fails():
    """An unpriced room made held instead of refused: the provenance would then hold rooms."""
    mutant = _once(ROOM_COST, HOLD_IFF, "cost.held = !moe || !in.priced || cost.host_attn_us > cost.expert_us;")
    assert not claim_default_only_refuses(_backend_with(ROOM_COST_REL, mutant))


def test_mutant_second_backend_reader_fails():
    """The flash-attention TU reads the provenance: a second way into a decision, past the one rule."""
    text = BACKEND_TREE[FATTN_REL] + "\nstatic bool g_7fmt_mutant(const ggml_sycl::placement_kv_info & k) " \
        "{ return k.n_ctx_context_is_default; }\n"
    assert not claim_default_only_refuses(_backend_with(FATTN_REL, text))


def test_mutant_note_unconditional_fails():
    """Every room reads as the default context, an explicit -c included."""
    assert not claim_room_lines_name_the_default(
        _once(CACHE, SOURCE_NOTE, f'const char * ctx_source = "{NOTE}";'))


def test_mutant_note_guessed_from_the_size_fails():
    """The note guessed from the context's size instead of the provenance: an explicit -c 262144 would read as the
    default, and a default context of 4096 would not."""
    assert not claim_room_lines_name_the_default(
        _once(CACHE, SOURCE_NOTE, SOURCE_NOTE.replace("kv_info.n_ctx_context_is_default ?",
                                                      "kv_info.n_ctx_context > 131072 ?")))


def test_mutant_note_after_the_lines_fails():
    """Chosen after the refused line returns: the all-refused WARN would print an undeclared note."""
    n = _once(CACHE, SOURCE_NOTE, "")
    assert n.count(ALL_REFUSED) == 1
    anchor = "char refused_note[64]"
    assert n.count(anchor) == 1
    assert not claim_room_lines_name_the_default(n.replace(anchor, SOURCE_NOTE + " " + anchor, 1))


# ---- --self-test ----------------------------------------------------------------------------------------------------


def self_test() -> int:
    """Runs every claim on the tree and every mutant, naming the claim each mutant turned red."""
    g = globals()
    claims = sorted(n for n in g if n.startswith("test_") and not n.startswith("test_mutant_") and callable(g[n]))
    mutants = sorted(n for n in g if n.startswith("test_mutant_") and callable(g[n]))
    bad = 0
    for name in claims:
        try:
            g[name]()
            print(f"  claim holds on the tree: {name[len('test_'):]}")
        except AssertionError as e:
            print(f"  claim FAILS on the tree: {name[len('test_'):]} {e}")
            bad += 1
    for name in mutants:
        try:
            g[name]()
            print(f"  self-test: mutant {name[len('test_mutant_'):]} caught")
        except AssertionError as e:
            print(f"  self-test: mutant {name[len('test_mutant_'):]} NOT caught {e}")
            bad += 1
    print(f"self-test: {len(claims)} claim(s), {len(mutants)} mutant(s), {bad} failure(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        sys.exit(self_test())

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
