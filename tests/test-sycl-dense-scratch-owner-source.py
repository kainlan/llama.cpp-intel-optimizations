"""Source contract for llama.cpp-45qo: the dense scratch plan's inputs are owner terms (canonical contract 12.4).

Host-only: reads sources, builds nothing, loads no model and touches no device.

The defect: the dense Q8_1 src1 and f16 dequant scratch plans (llama.cpp-479i, llama.cpp-kpjw) kept their inputs (the
bytes per token, the largest f16 weight, the n_ubatch) as device-global atomics. A setter merged its new input with the
stored one through zone_dense_scratch_merge_input(prev, next, other_model_live): the larger survived while another
backend context was live on the device, and the new one replaced it when none was. So a model unloaded beside one that
stayed live left its larger inputs in every later context's RUNTIME zone requirement on that device, and a model loaded
beside a live but contextless one replaced, and shrank, that one's plan. Neither reset had an owner key, which 12.4
forbids.

The fix this gate pins: the inputs are terms in the owner table llama.cpp-ihk6 built for the compute and state terms
(planned-load-terms.hpp), one contribution per (model slot, slot generation) per device:

  D1. the table carries the dense scratch terms, its term count follows the enum (COUNT), it stores an owner's terms
      all or nothing, and its drop and its newer-generation sweep are one walk (clear_slot);
  D2. both setters take the owner instead of a live flag and go through the one owner store every planned load term
      setter shares: store under the table's leaf mutex, republish in the same locked step, and name a swept stale
      contribution after the lock;
  D3. a store validates the figure on the merge the device will read once it lands (merged_accepted_if_set) before
      it stores; a rejected figure overwrites the owner's inputs for that buffer with 0 and raises its flag;
  D4. the dense atomics the readers load are published from the merge of the owners whose figure for each buffer was
      accepted (merged_accepted), each half's n_ubatch from its own merge, by one function, and nothing else writes
      them; the publish runs on every set and every drop, so an owner's inputs leave with it;
  D5. the runtime re-plan takes the owner and re-derives from that owner's own inputs, refuses only when that owner's
      own figure was rejected (llama.cpp-kpjw: one model's overflow does not invalidate the plan of a model already
      running), and publishes whenever it changed the table;
  D5a. an owner's n_ubatch only rises while the owner lives (raised_value): two contexts of one model share it, so a
      second context at a smaller n_ubatch cannot shrink the plan the first one's buffers were sized from;
  D6. no device-global merge remains: no zone_dense_scratch_merge_input, no other_model_live, no
      ggml_sycl_other_backend_context_live;
  D7. the load-time plan is keyed by the owner of the load effect stage_inventory_plan holds on this thread, passed
      down beside its transaction id, so the stores land under that held effect and no second lease is taken;
  D8. the runtime-context transaction re-plans the published model's own plan, and its guard restores only the hold
      (the owner's n_ubatch only rises, so there is no earlier plan to put back);
  D9. the decision is recorded in canonical contract 12.4 and in the cross-model state audit;
  D10. a model whose own figure was rejected has no dense scratch planned, so the runtime transaction refuses its
      context by name, with a WARN naming the owner and the half, before the fit check and after a re-plan that a
      rejection refused, instead of letting the other owners' plan pass it through.

The table itself is unit-tested in ggml/src/ggml-sycl/tests/test-planned-load-terms.cpp; this gate pins how the
backend reaches it. Every claim is checked on COMMENT-STRIPPED, whitespace-normalized text and has a mutant that must
make that claim fail.

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
    """Comment-stripped, whitespace-canonical text, so a clang-format re-wrap cannot turn a claim red."""
    t = re.sub(r"\s+", " ", strip_comments(text))
    t = re.sub(r"\( ", "(", t)
    t = re.sub(r" \)", ")", t)
    t = re.sub(r" ,", ",", t)
    t = re.sub(r",(?! )", ", ", t)
    return t


def read(rel: str) -> str:
    return (ROOT / rel).read_text()


SYCL = read("ggml/src/ggml-sycl/ggml-sycl.cpp")
CACHE = read("ggml/src/ggml-sycl/unified-cache.cpp")
CACHE_H = read("ggml/src/ggml-sycl/unified-cache.hpp")
TABLE = read("ggml/src/ggml-sycl/planned-load-terms.hpp")
ZONE = read("ggml/src/ggml-sycl/zone-sizing.cpp")
ZONE_H = read("ggml/src/ggml-sycl/zone-sizing.hpp")
CONTRACT = read("docs/design/sycl-canonical-memory-architecture.md")
AUDIT = read("docs/backend/sycl-cross-model-state-audit.md")


def body(text: str, signature: str) -> str:
    """The brace-balanced body of the first definition whose normalized text starts with `signature` (string
    literals are skipped while balancing). A signature that ends with its opening brace skips a declaration of the same
    name. Empty when there is none."""
    at = text.find(signature)
    if at < 0:
        return ""
    open_at = at + len(signature) - 1 if signature.endswith("{") else text.find("{", at + len(signature))
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


def _once(raw: str, old: str, new: str, within: str = "") -> str:
    """The mutant: `old` replaced by `new` in the source, all three compared as norm() text so a clang-format re-wrap
    of the source moves no anchor. The anchor must match exactly once, in the body of `within` when it is given (for
    an anchor the file repeats), else in the whole file. The claims norm() their input, so they read the result."""
    text, o, n = norm(raw), norm(old), norm(new)
    scope = body(text, within) if within else text
    assert scope.count(o) == 1, f"mutant anchor must match exactly once: {old!r} x{scope.count(o)}"
    return text.replace(scope, scope.replace(o, n, 1), 1) if within else text.replace(o, n, 1)


# ---- signatures -----------------------------------------------------------------------------------------------------

SET_MANY_SIG = ("bool set(int device, lifecycle::SlotToken owner, const planned_load_term_value * values, "
                "int n_values, planned_load_terms_replaced * replaced = nullptr)")
DROP_SIG_T = "bool drop(lifecycle::SlotToken owner)"
Q8_SET_SIG = ("bool unified_cache_set_planned_mmq_src1_scratch(int device_id, lifecycle::SlotToken owner, "
              "size_t bytes_per_token, uint32_t n_ubatch)")
F16_SET_SIG = ("bool unified_cache_set_planned_dequant_f16_scratch(int device_id, lifecycle::SlotToken owner, "
               "size_t max_weight_bytes, size_t src1_bytes_per_token, uint32_t n_ubatch)")
Q8_STORE_SIG = "static bool planned_mmq_src1_scratch_store_locked(int device_id, lifecycle::SlotToken owner,"
F16_STORE_SIG = "static bool planned_dequant_f16_scratch_store_locked(int device_id, lifecycle::SlotToken owner,"
PUBLISH_SIG = "static void planned_load_terms_publish_locked()"
DENSE_PUBLISH_SIG = "static void planned_dense_scratch_publish_locked(int device_id)"
REPLAN_SIG = ("bool unified_cache_replan_planned_dense_scratch(int device_id, lifecycle::SlotToken owner, "
              "uint32_t n_ubatch)")
POPULATE_SIG = ("static void populate_inventory_globals(ggml_backend_sycl_context * ctx, "
                "const ggml_sycl_tensor_inventory * inventory)")
STAGE_SIG = ("ggml_sycl_lifecycle_result ggml_backend_sycl_stage_inventory_plan(const ggml_sycl_tensor_inventory * "
             "inventory,")
STORE_HELPER_SIG = ("template <typename F> static bool planned_load_terms_store(int device_id, lifecycle::SlotToken "
                    "owner, F && store_locked)")
REJECTED_LOCKED_SIG = ("static bool planned_dense_scratch_owner_rejected_locked(int device_id, lifecycle::SlotToken "
                       "owner, bool * q8_rejected, bool * f16_rejected)")
REJECTED_SIG = ("bool unified_cache_planned_dense_scratch_owner_rejected(int device_id, lifecycle::SlotToken owner, "
                "bool * q8_rejected, bool * f16_rejected)")
REFUSED_SIG = ("static bool ggml_sycl_dense_scratch_owner_refused(int device, ggml_sycl::lifecycle::SlotToken owner, "
               "bool probe_mode)")
RAISED_SIG = ("size_t raised_value(int device, lifecycle::SlotToken owner, planned_load_term term, size_t value) "
              "const")
TXN_SIG = "static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction(ggml_backend_t backend,"
GUARD_SIG = "struct ggml_sycl_dense_scratch_txn_guard {"

LOCK = "std::lock_guard<std::mutex> lock(g_planned_load_terms.mutex);"
PUBLISH_CALL = "planned_load_terms_publish_locked();"

# Each half's inputs, read over the owners whose figure for that half was accepted.
DENSE_READS = (("MMQ_SRC1_BYTES_PER_TOKEN", "q8_flag"), ("DENSE_SCRATCH_N_UBATCH", "q8_flag"),
               ("DEQUANT_F16_WEIGHT_BYTES", "f16_flag"), ("DEQUANT_F16_SRC1_BYTES_PER_TOKEN", "f16_flag"),
               ("DENSE_SCRATCH_N_UBATCH", "f16_flag"))

DENSE_TERMS = ("MMQ_SRC1_BYTES_PER_TOKEN", "DEQUANT_F16_WEIGHT_BYTES", "DEQUANT_F16_SRC1_BYTES_PER_TOKEN",
               "DENSE_SCRATCH_N_UBATCH", "MMQ_SRC1_REJECTED", "DEQUANT_F16_REJECTED")

# The published dense values: each one's only writer is planned_dense_scratch_publish_locked.
DENSE_ATOMICS = ("g_planned_mmq_src1_scratch_bytes", "g_planned_dequant_f16_src0_bytes",
                 "g_planned_dequant_f16_src1_bytes", "g_planned_mmq_src1_bytes_per_token",
                 "g_planned_dequant_f16_weight_bytes_max", "g_planned_dequant_f16_src1_bytes_per_token",
                 "g_planned_dense_scratch_n_ubatch", "g_planned_dense_scratch_invalid")


def _writes(text: str, name: str) -> int:
    """Writes of `name[...]` (any index depth) by store, exchange, read-modify-write, assignment or increment."""
    idx = r"(?:\[[^\]]*\])+"
    pat = (rf"\b{name}{idx}\s*\.\s*(?:store|exchange|fetch_\w+|compare_exchange_\w+)\s*\("
           rf"|\b{name}{idx}\s*(?:[-+*/|&^]?=(?!=)|\+\+|--)"
           rf"|(?:\+\+|--)\s*{name}{idx}")
    return len(re.findall(pat, text))


# ---- D1: the table ------------------------------------------------------------------------------------------------


def claim_table_carries_the_dense_terms_all_or_nothing(table: str) -> bool:
    """The dense scratch inputs and flags are table terms, a contribution holds one value per term, a store checks
    every term before it changes anything, and drop and sweep are the same walk."""
    t = norm(table)
    enum = body(t, "enum class planned_load_term")
    s = body(t, SET_MANY_SIG)
    d = body(t, DROP_SIG_T)
    return (bool(enum) and all(re.search(rf"\b{n}\b", enum) for n in DENSE_TERMS)
            and "size_t values[planned_load_term_count] = {};" in t
            and re.search(r"\bDEQUANT_F16_REJECTED, COUNT, \}", enum) is not None
            and "constexpr int planned_load_term_count = static_cast<int>(planned_load_term::COUNT);" in t
            and ordered(s, "for (int i = 0; i < n_values; ++i) { "
                           "if (!term_in_range(values[i].term)) { return false; } }",
                        "(void) clear_slot(owner.slot, owner.generation, slot_generations::OLDER, replaced);",
                        "c.generation = owner.generation;",
                        "c.values[static_cast<int>(values[i].term)] = values[i].value;")
            and "return clear_slot(owner.slot, owner.generation, slot_generations::EXACT, nullptr);" in d
            and t.count("contributions_[d][slot]") == 1)


# ---- D2: the setters ----------------------------------------------------------------------------------------------


def claim_owner_stores_share_one_helper(cache: str) -> bool:
    """The one owner store: the caller's locked store under the table's leaf mutex, the republish in that locked step,
    and the swept devices named after the lock is released. Every planned load term setter goes through it."""
    t = norm(cache)
    h = body(t, STORE_HELPER_SIG)
    term = body(t, "static bool unified_cache_set_planned_load_term(int device_id, lifecycle::SlotToken owner,")
    return (bool(h) and ordered(h, "planned_load_terms_replaced replaced;", LOCK,
                                "if (!store_locked(&replaced)) { return false; }", PUBLISH_CALL, "}",
                                "planned_load_terms_name_replaced(owner, device_id, replaced);", "return true;")
            and h.count(PUBLISH_CALL) == 1 and t.count("planned_load_terms_name_replaced(owner, device_id,") == 1
            and ordered(term, "return planned_load_terms_store(device_id, owner, [&](planned_load_terms_replaced * "
                              "replaced) {", "return g_planned_load_terms.owners.set(device_id, owner, term, bytes, "
                                             "replaced);")
            and LOCK not in term)


def _setter_is_owner_store(cache: str, sig: str, store_call: str) -> bool:
    b = body(norm(cache), sig)
    return (bool(b) and claim_owner_stores_share_one_helper(cache)
            and ordered(b, "bool planned = false;",
                        "const bool stored = planned_load_terms_store(device_id, owner, "
                        "[&](planned_load_terms_replaced * replaced) {", "return " + store_call,
                        "return stored && planned;")
            and LOCK not in b and PUBLISH_CALL not in b and ".store(" not in b and "other_model_live" not in b)


def claim_q8_setter_is_the_owner_store(cache: str) -> bool:
    """The Q8_1 setter stores the owner's inputs under the leaf mutex, republishes in that locked step, and names a
    swept stale contribution after it."""
    return _setter_is_owner_store(
        cache, Q8_SET_SIG,
        "planned_mmq_src1_scratch_store_locked(device_id, owner, bytes_per_token, n_ubatch, &planned, replaced);")


def claim_f16_setter_is_the_owner_store(cache: str) -> bool:
    """The f16 dequant setter, the same way."""
    return _setter_is_owner_store(
        cache, F16_SET_SIG,
        "planned_dequant_f16_scratch_store_locked(device_id, owner, max_weight_bytes, src1_bytes_per_token, n_ubatch, "
        "&planned, replaced);")


# ---- D3: validate on the merge, then store ------------------------------------------------------------------------


def claim_q8_store_validates_the_merge_first(cache: str) -> bool:
    """The Q8_1 figure is sized on the merge the device reads once the owner's inputs land, over the owners whose Q8_1
    figure was accepted, before anything is stored; a rejected figure overwrites the owner's input with 0 and raises
    its flag, an accepted one clears the flag."""
    b = body(norm(cache), Q8_STORE_SIG)
    return (bool(b) and ordered(
        b, "owners.merged_accepted_if_set(device_id, owner, planned_load_term::MMQ_SRC1_BYTES_PER_TOKEN, "
        "bytes_per_token, planned_load_term::MMQ_SRC1_REJECTED);",
        "owners.merged_accepted_if_set(device_id, owner, planned_load_term::DENSE_SCRATCH_N_UBATCH, n_ubatch, "
        "planned_load_term::MMQ_SRC1_REJECTED);",
        "zone_mmq_src1_scratch_bytes(merged_per_token, static_cast<uint32_t>(merged_n_ubatch), &bytes);",
        "if (!*planned) {", "{ planned_load_term::MMQ_SRC1_BYTES_PER_TOKEN, 0 },",
        "{ planned_load_term::MMQ_SRC1_REJECTED, 1 },",
        "return owners.set(device_id, owner, rejected, 2, replaced);",
        "{ planned_load_term::MMQ_SRC1_BYTES_PER_TOKEN, bytes_per_token },",
        "{ planned_load_term::DENSE_SCRATCH_N_UBATCH, n_ubatch },", "{ planned_load_term::MMQ_SRC1_REJECTED, 0 },",
        "return owners.set(device_id, owner, accepted, 3, replaced);")
        and b.count("owners.set(") == 2 and ".store(" not in b)


def claim_f16_store_validates_the_merge_first(cache: str) -> bool:
    """The f16 dequant figure, the same way."""
    b = body(norm(cache), F16_STORE_SIG)
    return (bool(b) and ordered(
        b, "owners.merged_accepted_if_set(device_id, owner, planned_load_term::DEQUANT_F16_WEIGHT_BYTES, "
        "max_weight_bytes, planned_load_term::DEQUANT_F16_REJECTED);",
        "planned_load_term::DEQUANT_F16_SRC1_BYTES_PER_TOKEN, src1_bytes_per_token, "
        "planned_load_term::DEQUANT_F16_REJECTED);",
        "owners.merged_accepted_if_set(device_id, owner, planned_load_term::DENSE_SCRATCH_N_UBATCH, n_ubatch, "
        "planned_load_term::DEQUANT_F16_REJECTED);",
        "zone_dequant_f16_plan_bytes(merged_weights, merged_per_token, static_cast<uint32_t>(merged_n_ubatch), "
        "&src0_bytes, &src1_bytes);",
        "if (!*planned) {", "{ planned_load_term::DEQUANT_F16_WEIGHT_BYTES, 0 },",
        "{ planned_load_term::DEQUANT_F16_SRC1_BYTES_PER_TOKEN, 0 },",
        "{ planned_load_term::DEQUANT_F16_REJECTED, 1 },",
        "return owners.set(device_id, owner, rejected, 3, replaced);",
        "{ planned_load_term::DEQUANT_F16_WEIGHT_BYTES, max_weight_bytes },",
        "{ planned_load_term::DEQUANT_F16_REJECTED, 0 },",
        "return owners.set(device_id, owner, accepted, 4, replaced);")
        and b.count("owners.set(") == 2 and ".store(" not in b)


# ---- D4: the published plan is the merge, with one writer ---------------------------------------------------------


def claim_dense_plan_is_published_from_the_merge(cache: str) -> bool:
    """Every device's publish derives each half of the dense plan from the merge of each input over the owners whose
    figure for that half was accepted; that function is the only writer of every published dense value; and it runs
    inside the one publish every set and drop calls."""
    t = norm(cache)
    pub = body(t, PUBLISH_SIG)
    dense = body(t, DENSE_PUBLISH_SIG)
    reads = all(f"owners.merged_accepted(device_id, planned_load_term::{n}, {flag});" in dense
                for n, flag in DENSE_READS)
    one_writer = all(_writes(t, a) >= 1 and _writes(t, a) == _writes(dense, a) for a in DENSE_ATOMICS)
    return (bool(pub) and bool(dense) and reads and one_writer
            and ordered(pub, "for (int device_id = 0; device_id < GGML_SYCL_MAX_DEVICES; ++device_id) {",
                        "planned_dense_scratch_publish_locked(device_id);")
            and t.count("planned_dense_scratch_publish_locked(device_id);") == 1
            and "const planned_load_term q8_flag = planned_load_term::MMQ_SRC1_REJECTED;" in dense
            and "const planned_load_term f16_flag = planned_load_term::DEQUANT_F16_REJECTED;" in dense
            and "owners.merged(" not in dense
            and ordered(dense, "zone_mmq_src1_scratch_bytes(q8_per_tok, static_cast<uint32_t>(q8_n_ubatch),",
                        "zone_dequant_f16_plan_bytes(f16_weights, f16_per_tok, static_cast<uint32_t>(f16_n_ubatch),",
                        "g_planned_mmq_src1_scratch_bytes[device_id].store(q8_valid ? q8_bytes : 0,",
                        "g_planned_dense_scratch_invalid[device_id][0].store(!q8_valid,")
            and "g_planned_dense_scratch_n_ubatch[device_id][0].store(q8_n_ubatch <= UINT32_MAX ? "
                "static_cast<uint32_t>(q8_n_ubatch) : 0," in dense
            and "g_planned_dense_scratch_n_ubatch[device_id][1].store(f16_n_ubatch <= UINT32_MAX ? "
                "static_cast<uint32_t>(f16_n_ubatch) : 0," in dense)


# ---- D5: the runtime re-plan --------------------------------------------------------------------------------------


def claim_replan_rederives_the_owners_own_inputs(cache: str) -> bool:
    """The re-plan reads only the owner's own inputs, in the same locked step as both plan stores and the publish; an
    owner that planned nothing on the device re-derives nothing, and an owner whose own figure was rejected refuses
    (its inputs for that buffer read 0). The device-wide flag refuses no owner's re-plan. A store that changed the
    table is published even when the other store was refused, so the table and the published atomics never
    disagree."""
    b = body(norm(cache), REPLAN_SIG)
    gets = ("if (planned_dense_scratch_owner_rejected_locked(device_id, owner, nullptr, nullptr)) { return false; }",
            "planned_load_term::DENSE_SCRATCH_N_UBATCH, &owner_n_ubatch)", "owner_n_ubatch == 0 ||",
            "planned_load_term::MMQ_SRC1_BYTES_PER_TOKEN, &q8_per_token)",
            "planned_load_term::DEQUANT_F16_WEIGHT_BYTES, &f16_weights)",
            "planned_load_term::DEQUANT_F16_SRC1_BYTES_PER_TOKEN, &f16_per_token)")
    return (bool(b) and ordered(b, LOCK, *gets,
                                "const bool q8_stored = planned_mmq_src1_scratch_store_locked(device_id, owner, "
                                "q8_per_token, next_n_ubatch, &q8, nullptr);",
                                "const bool f16_stored = q8_stored && planned_dequant_f16_scratch_store_locked("
                                "device_id, owner, f16_weights, f16_per_token, next_n_ubatch, &f16, nullptr);",
                                "if (q8_stored) { " + PUBLISH_CALL + " }", "return f16_stored && q8 && f16;")
            and b.count(PUBLISH_CALL) == 1
            and b.count("g_planned_load_terms.owners.get(device_id, owner,") == 4
            and "g_planned_dense_scratch_invalid" not in b
            and ".merged(" not in b and "_bytes_per_token[device_id]" not in b
            and "_weight_bytes_max[device_id]" not in b)


def claim_owner_n_ubatch_only_rises(cache: str, table: str) -> bool:
    """An owner's n_ubatch only rises while it lives: the re-plan stores the larger of the owner's current value and
    the new one (raised_value) into both plan stores, never the new value alone, and nothing lowers it short of the
    owner's drop."""
    b = body(norm(cache), REPLAN_SIG)
    raised = body(norm(table), RAISED_SIG)
    return (bool(b) and bool(raised)
            and ordered(b, "const size_t raised = g_planned_load_terms.owners.raised_value(device_id, owner, "
                           "planned_load_term::DENSE_SCRATCH_N_UBATCH, n_ubatch);",
                        "if (raised > UINT32_MAX) { return false; }",
                        "const uint32_t next_n_ubatch = static_cast<uint32_t>(raised);",
                        "q8_per_token, next_n_ubatch, &q8, nullptr);", "f16_per_token, next_n_ubatch, &f16, nullptr);")
            and ", n_ubatch, &q8" not in b and ", n_ubatch, &f16" not in b
            and ordered(raised, "size_t current = 0;", "if (!get(device, owner, term, &current)) { return value; }",
                        "return current > value ? current : value;")
            and "unified_cache_get_planned_dense_scratch_owner_n_ubatch" not in norm(cache))


# ---- D6: no device-global merge remains -----------------------------------------------------------------------------


def claim_no_device_global_merge_remains(sycl: str, cache: str, cache_h: str, zone: str, zone_h: str) -> bool:
    """The live-flag merge, the flag and the context census that fed it are gone from the backend."""
    texts = [norm(x) for x in (sycl, cache, cache_h, zone, zone_h)]
    return all("zone_dense_scratch_merge_input" not in t and not re.search(r"\bother_model_live\b", t)
               and "ggml_sycl_other_backend_context_live" not in t for t in texts)


# ---- D7: the load-time owner ----------------------------------------------------------------------------------------


def claim_load_plan_is_keyed_by_the_open_load(sycl: str) -> bool:
    """populate_inventory_globals keys both plan stores by the owner of the load effect stage_inventory_plan holds on
    this thread, which the stage's scope sets beside the transaction id and restores with it; with no staged load it
    stores nothing. No second lease is taken, so a failed acquire cannot read as "no open load"."""
    t = norm(sycl)
    b = body(t, POPULATE_SIG)
    stage = body(t, STAGE_SIG)
    owner_writes = re.findall(r"\bg_sycl_plan_load_effect_owner\s*=(?!=)", t)
    return (bool(b) and bool(stage) and ordered(
        b, "const ggml_sycl::lifecycle::SlotToken plan_owner = g_sycl_plan_load_effect_owner;",
        "const bool owned = plan_owner.generation != 0;",
        "owned && ggml_sycl::unified_cache_set_planned_mmq_src1_scratch(ctx->device, plan_owner,",
        "owned && ggml_sycl::unified_cache_set_planned_dequant_f16_scratch(ctx->device, plan_owner,")
        and "acquire_load_effect(" not in b
        and "ggml_sycl_acquire_plan_load_effect" not in t
        and "static thread_local ggml_sycl::lifecycle::SlotToken g_sycl_plan_load_effect_owner{};" in t
        and ordered(stage, "auto effect = registry.acquire_load_effect(registry.bound_candidate());",
                    "previous_owner(g_sycl_plan_load_effect_owner) {", "g_sycl_plan_load_effect_txn = txn;",
                    "g_sycl_plan_load_effect_owner = owner;",
                    "~plan_effect_scope() { g_sycl_plan_load_effect_txn = previous; "
                    "g_sycl_plan_load_effect_owner = previous_owner; }",
                    "} authority{ effect.owner.load.value, effect.owner.owner };")
        and len(owner_writes) == 2
        and t.count("unified_cache_set_planned_mmq_src1_scratch(") == 1
        and t.count("unified_cache_set_planned_dequant_f16_scratch(") == 1)


# ---- D8: the runtime-context transaction ----------------------------------------------------------------------------


def claim_runtime_txn_replans_the_published_models_own_plan(sycl: str) -> bool:
    """The transaction re-plans the published plan's model's own inputs and nothing else in the backend re-plans. Its
    guard restores only the hold: the owner's n_ubatch only rises, so a rollback has no earlier plan to put back."""
    t = norm(sycl)
    txn = body(t, TXN_SIG)
    guard = body(t, GUARD_SIG)
    return (bool(txn) and bool(guard)
            and ordered(txn, "ggml_sycl_dense_scratch_txn_guard dense_guard(*ctx, !probe_mode);",
                        "const bool replanned = ggml_sycl::unified_cache_replan_planned_dense_scratch(ctx->device, "
                        "current_token.owner, next_kv_info.n_ubatch);")
            and "unified_cache_set_planned_scratch_hold(device, prev_hold, prev_owner);" in guard
            and "unified_cache_replan_planned_dense_scratch(" not in guard
            and "n_ubatch" not in guard and "plan_owner" not in guard
            and t.count("unified_cache_replan_planned_dense_scratch(") == 1)


# ---- D9: the decision is recorded -----------------------------------------------------------------------------------


def claim_decision_is_recorded(contract: str, audit: str) -> bool:
    """Canonical contract 12.4 carries a row for the dense scratch inputs naming this ticket, and the cross-model
    audit records them as owner-keyed."""
    sec = contract[contract.find("### 12.4"):]
    sec = sec[:sec.find("\n### ", 1)] if "\n### " in sec[1:] else sec
    rows = [ln for ln in sec.splitlines() if ln.startswith("|") and "llama.cpp-45qo" in ln]
    return (any("dense scratch" in ln.lower() and "(slot, slotgeneration)" in ln.lower() for ln in rows)
            and "llama.cpp-45qo" in audit and "zone_dense_scratch_merge_input" in audit)


# ---- D10: a rejected owner's context is refused by name -----------------------------------------------------------


def claim_rejected_owner_is_refused_by_name(sycl: str, cache: str) -> bool:
    """The owner-scoped query reads the owner's own two flags under the leaf mutex (the re-plan's refusal reads them
    through the same locked helper); the backend helper WARNs naming the owner's slot, generation and the rejected half;
    and the transaction refuses through it before the fit check and again when a re-plan was refused by a rejection, so
    an overflowing figure never becomes a small plan that the other owners' figures pass through."""
    c = norm(cache)
    t = norm(sycl)
    locked = body(c, REJECTED_LOCKED_SIG)
    query = body(c, REJECTED_SIG)
    helper = body(t, REFUSED_SIG)
    txn = body(t, TXN_SIG)
    refusal = ("if (ggml_sycl_dense_scratch_owner_refused(ctx->device, current_token.owner, probe_mode)) { "
               "return refuse(\"a dense scratch figure of this model overflowed its sizing\"); }")
    after_replan = ("if (!replanned && ggml_sycl_dense_scratch_owner_refused(ctx->device, current_token.owner, "
                    "probe_mode)) { return refuse(\"a dense scratch figure of this model overflowed its sizing\"); }")
    fit_at = txn.find("unified_cache_dense_scratch_runtime_fit(")
    return (bool(locked) and bool(query) and bool(helper) and fit_at >= 0
            and ordered(locked, "(void) g_planned_load_terms.owners.get(device_id, owner, "
                                "planned_load_term::MMQ_SRC1_REJECTED, &q8);",
                        "(void) g_planned_load_terms.owners.get(device_id, owner, "
                        "planned_load_term::DEQUANT_F16_REJECTED, &f16);",
                        "return q8 != 0 || f16 != 0;")
            and ordered(query, LOCK, "return planned_dense_scratch_owner_rejected_locked(device_id, owner, "
                                     "q8_rejected, f16_rejected);")
            and ordered(helper, "if (!ggml_sycl::unified_cache_planned_dense_scratch_owner_rejected(device, owner, "
                                "&q8, &f16)) { return false; }", "GGML_LOG_WARN(", "owner.slot,", "return true;")
            and "model slot %u generation %llu" in helper and "Q8_1 src1" in helper and "f16 dequant" in helper
            and 0 <= txn.find(refusal) < fit_at
            and ordered(txn, "const bool replanned = ", after_replan))


# ---- positive runs --------------------------------------------------------------------------------------------------


def test_table_carries_the_dense_terms_all_or_nothing():
    assert claim_table_carries_the_dense_terms_all_or_nothing(TABLE)


def test_q8_setter_is_the_owner_store():
    assert claim_q8_setter_is_the_owner_store(CACHE)


def test_f16_setter_is_the_owner_store():
    assert claim_f16_setter_is_the_owner_store(CACHE)


def test_q8_store_validates_the_merge_first():
    assert claim_q8_store_validates_the_merge_first(CACHE)


def test_f16_store_validates_the_merge_first():
    assert claim_f16_store_validates_the_merge_first(CACHE)


def test_dense_plan_is_published_from_the_merge():
    assert claim_dense_plan_is_published_from_the_merge(CACHE)


def test_replan_rederives_the_owners_own_inputs():
    assert claim_replan_rederives_the_owners_own_inputs(CACHE)


def test_owner_n_ubatch_only_rises():
    assert claim_owner_n_ubatch_only_rises(CACHE, TABLE)


def test_owner_stores_share_one_helper():
    assert claim_owner_stores_share_one_helper(CACHE)


def test_rejected_owner_is_refused_by_name():
    assert claim_rejected_owner_is_refused_by_name(SYCL, CACHE)


def test_no_device_global_merge_remains():
    assert claim_no_device_global_merge_remains(SYCL, CACHE, CACHE_H, ZONE, ZONE_H)


def test_load_plan_is_keyed_by_the_open_load():
    assert claim_load_plan_is_keyed_by_the_open_load(SYCL)


def test_runtime_txn_replans_the_published_models_own_plan():
    assert claim_runtime_txn_replans_the_published_models_own_plan(SYCL)


def test_decision_is_recorded():
    assert claim_decision_is_recorded(CONTRACT, AUDIT)


# ---- mutants: each must fail the claim it names ---------------------------------------------------------------------


def test_mutant_term_count_written_as_a_literal_fails():
    assert not claim_table_carries_the_dense_terms_all_or_nothing(
        _once(TABLE, "constexpr int planned_load_term_count = static_cast<int>(planned_load_term::COUNT);",
              "constexpr int planned_load_term_count = 8;"))


def test_mutant_table_without_a_dense_term_fails():
    assert not claim_table_carries_the_dense_terms_all_or_nothing(_once(TABLE, "DEQUANT_F16_REJECTED,", ""))


def test_mutant_table_storing_before_checking_every_term_fails():
    assert not claim_table_carries_the_dense_terms_all_or_nothing(
        _once(TABLE, "if (!term_in_range(values[i].term)) { return false; }", "", within=SET_MANY_SIG))


def test_mutant_drop_with_its_own_walk_fails():
    assert not claim_table_carries_the_dense_terms_all_or_nothing(
        _once(TABLE, "return clear_slot(owner.slot, owner.generation, slot_generations::EXACT, nullptr);",
              "bool dropped = false; for (int d = 0; d < GGML_SYCL_MAX_DEVICES; ++d) { "
              "planned_load_term_contribution & c = contributions_[d][slot]; (void) c; } return dropped;"))


def test_mutant_q8_setter_storing_a_published_value_directly_fails():
    assert not claim_q8_setter_is_the_owner_store(
        _once(CACHE, "return stored && planned;", "g_planned_mmq_src1_scratch_bytes[device_id].store(0, "
              "std::memory_order_release); return stored && planned;", within=Q8_SET_SIG))


def test_mutant_q8_setter_with_its_own_lock_fails():
    assert not claim_q8_setter_is_the_owner_store(
        _once(CACHE, "bool planned = false;", "bool planned = false; " + LOCK, within=Q8_SET_SIG))


def test_mutant_owner_store_publishing_outside_the_lock_fails():
    assert not claim_f16_setter_is_the_owner_store(
        _once(CACHE, "planned_load_terms_publish_locked(); } planned_load_terms_name_replaced(",
              "} planned_load_terms_publish_locked(); planned_load_terms_name_replaced(", within=STORE_HELPER_SIG))


def test_mutant_owner_store_never_naming_the_sweep_fails():
    assert not claim_f16_setter_is_the_owner_store(
        _once(CACHE, "planned_load_terms_name_replaced(owner, device_id, replaced);", "",
              within=STORE_HELPER_SIG))


def test_mutant_term_setter_with_its_own_store_sequence_fails():
    assert not claim_owner_stores_share_one_helper(
        _once(CACHE, "return planned_load_terms_store(device_id, owner, [&](planned_load_terms_replaced * replaced) {",
              LOCK + " return planned_load_terms_store(device_id, owner, [&](planned_load_terms_replaced * replaced) {",
              within="static bool unified_cache_set_planned_load_term(int device_id, lifecycle::SlotToken owner,"))


def test_mutant_q8_store_validating_the_owner_alone_fails():
    assert not claim_q8_store_validates_the_merge_first(
        _once(CACHE, "zone_mmq_src1_scratch_bytes(merged_per_token,",
              "zone_mmq_src1_scratch_bytes(bytes_per_token,", within=Q8_STORE_SIG))


def test_mutant_q8_rejection_storing_the_input_fails():
    assert not claim_q8_store_validates_the_merge_first(
        _once(CACHE, "{ planned_load_term::MMQ_SRC1_BYTES_PER_TOKEN, 0 },",
              "{ planned_load_term::MMQ_SRC1_BYTES_PER_TOKEN, bytes_per_token },", within=Q8_STORE_SIG))


def test_mutant_f16_rejection_storing_the_weight_fails():
    assert not claim_f16_store_validates_the_merge_first(
        _once(CACHE, "{ planned_load_term::DEQUANT_F16_WEIGHT_BYTES, 0 },",
              "{ planned_load_term::DEQUANT_F16_WEIGHT_BYTES, max_weight_bytes },", within=F16_STORE_SIG))


def test_mutant_f16_rejection_keeping_the_src1_input_fails():
    assert not claim_f16_store_validates_the_merge_first(
        _once(CACHE, "{ planned_load_term::DEQUANT_F16_SRC1_BYTES_PER_TOKEN, 0 },", "", within=F16_STORE_SIG))


def test_mutant_f16_acceptance_keeping_the_flag_fails():
    assert not claim_f16_store_validates_the_merge_first(
        _once(CACHE, "{ planned_load_term::DEQUANT_F16_REJECTED, 0 },", "", within=F16_STORE_SIG))


def test_mutant_f16_store_validating_the_owner_alone_fails():
    assert not claim_f16_store_validates_the_merge_first(
        _once(CACHE, "zone_dequant_f16_plan_bytes(merged_weights,",
              "zone_dequant_f16_plan_bytes(max_weight_bytes,", within=F16_STORE_SIG))


def test_mutant_publish_skipping_the_dense_plan_fails():
    assert not claim_dense_plan_is_published_from_the_merge(
        _once(CACHE, "planned_dense_scratch_publish_locked(device_id);", "", within=PUBLISH_SIG))


def test_mutant_second_writer_of_a_dense_value_fails():
    assert not claim_dense_plan_is_published_from_the_merge(
        _once(CACHE, "uint32_t unified_cache_get_planned_dense_scratch_n_ubatch(int device_id, bool f16) {",
              "uint32_t unified_cache_get_planned_dense_scratch_n_ubatch(int device_id, bool f16) { "
              "g_planned_dense_scratch_n_ubatch[0][0].store(0, std::memory_order_release);"))


def test_mutant_q8_store_validating_against_rejected_owners_fails():
    assert not claim_q8_store_validates_the_merge_first(
        _once(CACHE, "bytes_per_token, planned_load_term::MMQ_SRC1_REJECTED);",
              "bytes_per_token, planned_load_term::DEQUANT_F16_REJECTED);", within=Q8_STORE_SIG))


def test_mutant_dense_publish_merging_rejected_owners_fails():
    assert not claim_dense_plan_is_published_from_the_merge(
        _once(CACHE, "owners.merged_accepted(device_id, planned_load_term::MMQ_SRC1_BYTES_PER_TOKEN, q8_flag);",
              "owners.merged(device_id, planned_load_term::MMQ_SRC1_BYTES_PER_TOKEN);", within=DENSE_PUBLISH_SIG))


def test_mutant_dense_publish_invalidating_on_any_owners_flag_fails():
    assert not claim_dense_plan_is_published_from_the_merge(
        _once(CACHE, "const bool f16_valid = f16_n_ubatch <= UINT32_MAX &&",
              "const bool f16_valid = owners.merged(device_id, f16_flag) == 0 && f16_n_ubatch <= UINT32_MAX &&",
              within=DENSE_PUBLISH_SIG))


def test_mutant_dense_publish_one_n_ubatch_for_both_halves_fails():
    assert not claim_dense_plan_is_published_from_the_merge(
        _once(CACHE, "g_planned_dense_scratch_n_ubatch[device_id][1].store(f16_n_ubatch <= UINT32_MAX ? "
              "static_cast<uint32_t>(f16_n_ubatch) : 0,", "g_planned_dense_scratch_n_ubatch[device_id][1].store("
              "q8_n_ubatch <= UINT32_MAX ? static_cast<uint32_t>(q8_n_ubatch) : 0,", within=DENSE_PUBLISH_SIG))


def test_mutant_dense_publish_sizing_q8_at_the_f16_n_ubatch_fails():
    assert not claim_dense_plan_is_published_from_the_merge(
        _once(CACHE, "zone_mmq_src1_scratch_bytes(q8_per_tok, static_cast<uint32_t>(q8_n_ubatch),",
              "zone_mmq_src1_scratch_bytes(q8_per_tok, static_cast<uint32_t>(f16_n_ubatch),",
              within=DENSE_PUBLISH_SIG))


def test_mutant_replan_from_the_merged_inputs_fails():
    assert not claim_replan_rederives_the_owners_own_inputs(
        _once(CACHE, "planned_mmq_src1_scratch_store_locked(device_id, owner, q8_per_token,",
              "planned_mmq_src1_scratch_store_locked(device_id, owner, "
              "g_planned_mmq_src1_bytes_per_token[device_id].load(std::memory_order_acquire),", within=REPLAN_SIG))


def test_mutant_replan_for_an_owner_that_planned_nothing_fails():
    assert not claim_replan_rederives_the_owners_own_inputs(
        _once(CACHE, "owner_n_ubatch == 0 ||", "", within=REPLAN_SIG))


def test_mutant_replan_not_republishing_fails():
    assert not claim_replan_rederives_the_owners_own_inputs(
        _once(CACHE, "if (q8_stored) { planned_load_terms_publish_locked(); }", "", within=REPLAN_SIG))


def test_mutant_replan_publishing_only_when_both_stored_fails():
    assert not claim_replan_rederives_the_owners_own_inputs(
        _once(CACHE, "if (q8_stored) { planned_load_terms_publish_locked(); }",
              "if (!f16_stored) { return false; } planned_load_terms_publish_locked();", within=REPLAN_SIG))


def test_mutant_replan_ignoring_the_owners_rejection_fails():
    assert not claim_replan_rederives_the_owners_own_inputs(
        _once(CACHE, "if (planned_dense_scratch_owner_rejected_locked(device_id, owner, nullptr, nullptr)) { "
              "return false; }", "", within=REPLAN_SIG))


def test_mutant_replan_refusing_every_owner_on_an_invalid_device_fails():
    assert not claim_replan_rederives_the_owners_own_inputs(
        _once(CACHE, "if (planned_dense_scratch_owner_rejected_locked(device_id, owner, nullptr, nullptr)) {",
              "if (planned_dense_scratch_owner_rejected_locked(device_id, owner, nullptr, nullptr) || "
              "g_planned_dense_scratch_invalid[device_id][0].load(std::memory_order_acquire)) {", within=REPLAN_SIG))


def test_mutant_replan_storing_the_new_n_ubatch_unconditionally_fails():
    assert not claim_owner_n_ubatch_only_rises(
        _once(CACHE, "q8_per_token, next_n_ubatch, &q8, nullptr);", "q8_per_token, n_ubatch, &q8, nullptr);",
              within=REPLAN_SIG), TABLE)


def test_mutant_raised_value_taking_the_new_value_fails():
    assert not claim_owner_n_ubatch_only_rises(
        CACHE, _once(TABLE, "return current > value ? current : value;", "return value;", within=RAISED_SIG))


def test_mutant_merge_helper_back_fails():
    assert not claim_no_device_global_merge_remains(
        SYCL, CACHE, CACHE_H, ZONE + "\nsize_t zone_dense_scratch_merge_input(size_t prev, size_t next, "
        "bool other_model_live) { return other_model_live && prev > next ? prev : next; }\n", ZONE_H)


def test_mutant_context_census_back_fails():
    assert not claim_no_device_global_merge_remains(
        SYCL + "\nstatic bool ggml_sycl_other_backend_context_live(int device, "
        "const ggml_backend_sycl_context * self);\n",
        CACHE, CACHE_H, ZONE, ZONE_H)


def test_mutant_load_plan_without_an_owner_fails():
    assert not claim_load_plan_is_keyed_by_the_open_load(
        _once(SYCL, "owned && ggml_sycl::unified_cache_set_planned_mmq_src1_scratch(ctx->device, plan_owner,",
              "ggml_sycl::unified_cache_set_planned_mmq_src1_scratch(ctx->device, "
              "ggml_sycl::lifecycle::SlotToken{ 0, 1 },"))


def test_mutant_load_plan_taking_a_second_lease_fails():
    assert not claim_load_plan_is_keyed_by_the_open_load(
        _once(SYCL, "const ggml_sycl::lifecycle::SlotToken plan_owner = g_sycl_plan_load_effect_owner;",
              "const ggml_sycl::lifecycle::SlotToken plan_owner = ggml_sycl::lifecycle::global_registry()."
              "acquire_load_effect({ g_sycl_plan_load_effect_txn }).owner.owner;"))


def test_mutant_stage_scope_not_restoring_the_owner_fails():
    assert not claim_load_plan_is_keyed_by_the_open_load(
        _once(SYCL, "g_sycl_plan_load_effect_owner = previous_owner; }", "}", within=STAGE_SIG))


def test_mutant_stage_scope_passing_another_owner_fails():
    assert not claim_load_plan_is_keyed_by_the_open_load(
        _once(SYCL, "} authority{ effect.owner.load.value, effect.owner.owner };",
              "} authority{ effect.owner.load.value, ggml_sycl::lifecycle::SlotToken{} };", within=STAGE_SIG))


def test_mutant_txn_replanning_another_owner_fails():
    assert not claim_runtime_txn_replans_the_published_models_own_plan(
        _once(SYCL, "unified_cache_replan_planned_dense_scratch(ctx->device, current_token.owner,",
              "unified_cache_replan_planned_dense_scratch(ctx->device, ggml_sycl::lifecycle::SlotToken{ 0, 1 },"))


def test_mutant_guard_re_planning_an_earlier_n_ubatch_fails():
    assert not claim_runtime_txn_replans_the_published_models_own_plan(
        _once(SYCL, "ggml_sycl::unified_cache_set_planned_scratch_hold(device, prev_hold, prev_owner);",
              "(void) ggml_sycl::unified_cache_replan_planned_dense_scratch(device, ggml_sycl::lifecycle::SlotToken{}, "
              "512); ggml_sycl::unified_cache_set_planned_scratch_hold(device, prev_hold, prev_owner);",
              within=GUARD_SIG))


def test_mutant_a_second_replan_site_fails():
    assert not claim_runtime_txn_replans_the_published_models_own_plan(
        _once(SYCL, "static void populate_inventory_globals(ggml_backend_sycl_context * ctx, "
              "const ggml_sycl_tensor_inventory * inventory) {",
              "static void populate_inventory_globals(ggml_backend_sycl_context * ctx, "
              "const ggml_sycl_tensor_inventory * inventory) { (void) ggml_sycl::unified_cache_replan_planned_dense_"
              "scratch(ctx->device, ggml_sycl::lifecycle::SlotToken{}, 512);"))


def test_mutant_txn_without_the_rejected_owner_refusal_fails():
    assert not claim_rejected_owner_is_refused_by_name(
        _once(SYCL, "if (ggml_sycl_dense_scratch_owner_refused(ctx->device, current_token.owner, probe_mode)) { "
              "return refuse(\"a dense scratch figure of this model overflowed its sizing\"); }", "",
              within=TXN_SIG), CACHE)


def test_mutant_txn_discarding_the_replan_result_fails():
    assert not claim_rejected_owner_is_refused_by_name(
        _once(SYCL, "if (!replanned && ggml_sycl_dense_scratch_owner_refused(ctx->device, current_token.owner, "
              "probe_mode)) {", "if (false) {", within=TXN_SIG), CACHE)


def test_mutant_refusal_helper_silent_fails():
    assert not claim_rejected_owner_is_refused_by_name(
        _once(SYCL, "GGML_LOG_WARN(", "GGML_LOG_DEBUG(", within=REFUSED_SIG), CACHE)


def test_mutant_owner_query_reading_only_the_q8_flag_fails():
    assert not claim_rejected_owner_is_refused_by_name(
        SYCL, _once(CACHE, "planned_load_term::DEQUANT_F16_REJECTED, &f16);",
                    "planned_load_term::MMQ_SRC1_REJECTED, &f16);", within=REJECTED_LOCKED_SIG))


def test_mutant_contract_without_the_row_fails():
    assert not claim_decision_is_recorded(CONTRACT.replace("llama.cpp-45qo", "llama.cpp-XXXX"), AUDIT)


def test_mutant_audit_without_the_ticket_fails():
    assert not claim_decision_is_recorded(CONTRACT, AUDIT.replace("llama.cpp-45qo", "llama.cpp-XXXX"))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
