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

  D1. the table carries the dense scratch terms, stores an owner's terms all or nothing, and its drop and its newer-
      generation sweep are one walk (clear_slot);
  D2. both setters take the owner instead of a live flag, store under the table's leaf mutex, republish in the same
      locked step, and name a swept stale contribution after the lock;
  D3. a store validates the figure on the merge the device will read once it lands (merged_if_set) before it stores,
      and a rejected figure stores only the owner's flag;
  D4. the dense atomics the readers load are published from the owners' merge, by one function, and nothing else
      writes them; the publish runs on every set and every drop, so an owner's inputs leave with it;
  D5. the runtime re-plan takes the owner and re-derives from that owner's own inputs;
  D6. no device-global merge remains: no zone_dense_scratch_merge_input, no other_model_live, no
      ggml_sycl_other_backend_context_live;
  D7. the load-time plan is keyed by the open load's owner, from the load effect, held across both stores;
  D8. the runtime-context transaction re-plans, and its guard restores, the published model's own plan;
  D9. the decision is recorded in canonical contract 12.4 and in the cross-model state audit.

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
REPLAN_SIG = "bool unified_cache_replan_planned_dense_scratch(int device_id, lifecycle::SlotToken owner, uint32_t n_ubatch)"
OWNER_NUB_SIG = "uint32_t unified_cache_get_planned_dense_scratch_owner_n_ubatch(int device_id, lifecycle::SlotToken owner)"
POPULATE_SIG = ("static void populate_inventory_globals(ggml_backend_sycl_context * ctx, "
                "const ggml_sycl_tensor_inventory * inventory)")
EFFECT_HELPER_SIG = "static ggml_sycl::lifecycle::load_effect_lease ggml_sycl_acquire_plan_load_effect() {"
TXN_SIG = "static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction(ggml_backend_t backend,"
GUARD_SIG = "struct ggml_sycl_dense_scratch_txn_guard {"

LOCK = "std::lock_guard<std::mutex> lock(g_planned_load_terms.mutex);"
PUBLISH_CALL = "planned_load_terms_publish_locked();"

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


# ---- D1: the table ----------------------------------------------------------------------------------------------------


def claim_table_carries_the_dense_terms_all_or_nothing(table: str) -> bool:
    """The dense scratch inputs and flags are table terms, a contribution holds one value per term, a store checks
    every term before it changes anything, and drop and sweep are the same walk."""
    t = norm(table)
    enum = body(t, "enum class planned_load_term")
    s = body(t, SET_MANY_SIG)
    d = body(t, DROP_SIG_T)
    return (bool(enum) and all(re.search(rf"\b{n}\b", enum) for n in DENSE_TERMS)
            and "size_t values[planned_load_term_count] = {};" in t
            and "constexpr int planned_load_term_count = 8;" in t
            and ordered(s, "for (int i = 0; i < n_values; ++i) { if (!term_in_range(values[i].term)) { return false; } }",
                        "(void) clear_slot(owner.slot, owner.generation, slot_generations::OLDER, replaced);",
                        "c.generation = owner.generation;",
                        "c.values[static_cast<int>(values[i].term)] = values[i].value;")
            and "return clear_slot(owner.slot, owner.generation, slot_generations::EXACT, nullptr);" in d
            and t.count("contributions_[d][slot]") == 1)


# ---- D2: the setters ----------------------------------------------------------------------------------------------------


def _setter_is_owner_store(cache: str, sig: str, store_call: str) -> bool:
    b = body(norm(cache), sig)
    return (bool(b) and ordered(b, LOCK, "if (!" + store_call, "return false;", PUBLISH_CALL, "}",
                                "planned_load_terms_name_replaced(owner, device_id, replaced);", "return planned;")
            and b.count(PUBLISH_CALL) == 1 and ".store(" not in b and "other_model_live" not in b)


def claim_q8_setter_is_the_owner_store(cache: str) -> bool:
    """The Q8_1 setter stores the owner's inputs under the leaf mutex, republishes in that locked step, and names a swept
    stale contribution after it."""
    return _setter_is_owner_store(
        cache, Q8_SET_SIG,
        "planned_mmq_src1_scratch_store_locked(device_id, owner, bytes_per_token, n_ubatch, &planned, &replaced))")


def claim_f16_setter_is_the_owner_store(cache: str) -> bool:
    """The f16 dequant setter, the same way."""
    return _setter_is_owner_store(
        cache, F16_SET_SIG,
        "planned_dequant_f16_scratch_store_locked(device_id, owner, max_weight_bytes, src1_bytes_per_token, n_ubatch, "
        "&planned, &replaced))")


# ---- D3: validate on the merge, then store ----------------------------------------------------------------------------


def claim_q8_store_validates_the_merge_first(cache: str) -> bool:
    """The Q8_1 figure is sized on the merge the device reads once the owner's inputs land, before anything is stored;
    a rejected figure stores the owner's flag and no input, an accepted one clears the flag."""
    b = body(norm(cache), Q8_STORE_SIG)
    return (bool(b) and ordered(
        b, "owners.merged_if_set(device_id, owner, planned_load_term::MMQ_SRC1_BYTES_PER_TOKEN, bytes_per_token);",
        "owners.merged_if_set(device_id, owner, planned_load_term::DENSE_SCRATCH_N_UBATCH, n_ubatch);",
        "zone_mmq_src1_scratch_bytes(merged_per_token, static_cast<uint32_t>(merged_n_ubatch), &bytes);",
        "if (!*planned) {", "{ planned_load_term::MMQ_SRC1_BYTES_PER_TOKEN, 0 },",
        "{ planned_load_term::MMQ_SRC1_REJECTED, 1 },", "return owners.set(device_id, owner, rejected, 2, replaced);",
        "{ planned_load_term::MMQ_SRC1_BYTES_PER_TOKEN, bytes_per_token },",
        "{ planned_load_term::DENSE_SCRATCH_N_UBATCH, n_ubatch },", "{ planned_load_term::MMQ_SRC1_REJECTED, 0 },",
        "return owners.set(device_id, owner, accepted, 3, replaced);")
        and b.count("owners.set(") == 2 and ".store(" not in b)


def claim_f16_store_validates_the_merge_first(cache: str) -> bool:
    """The f16 dequant figure, the same way."""
    b = body(norm(cache), F16_STORE_SIG)
    return (bool(b) and ordered(
        b, "owners.merged_if_set(device_id, owner, planned_load_term::DEQUANT_F16_WEIGHT_BYTES, max_weight_bytes);",
        "planned_load_term::DEQUANT_F16_SRC1_BYTES_PER_TOKEN, src1_bytes_per_token);",
        "owners.merged_if_set(device_id, owner, planned_load_term::DENSE_SCRATCH_N_UBATCH, n_ubatch);",
        "zone_dequant_f16_plan_bytes(merged_weights, merged_per_token, static_cast<uint32_t>(merged_n_ubatch), "
        "&src0_bytes, &src1_bytes);",
        "if (!*planned) {", "{ planned_load_term::DEQUANT_F16_REJECTED, 1 },",
        "return owners.set(device_id, owner, rejected, 3, replaced);",
        "{ planned_load_term::DEQUANT_F16_WEIGHT_BYTES, max_weight_bytes },",
        "{ planned_load_term::DEQUANT_F16_REJECTED, 0 },", "return owners.set(device_id, owner, accepted, 4, replaced);")
        and b.count("owners.set(") == 2 and ".store(" not in b)


# ---- D4: the published plan is the merge, with one writer --------------------------------------------------------------


def claim_dense_plan_is_published_from_the_merge(cache: str) -> bool:
    """Every device's publish derives the dense plan from the owners' merge of each input and flag; that function is the
    only writer of every published dense value; and it runs inside the one publish every set and drop calls."""
    t = norm(cache)
    pub = body(t, PUBLISH_SIG)
    dense = body(t, DENSE_PUBLISH_SIG)
    reads = all(f"owners.merged(device_id, planned_load_term::{n})" in dense for n in DENSE_TERMS)
    one_writer = all(_writes(t, a) >= 1 and _writes(t, a) == _writes(dense, a) for a in DENSE_ATOMICS)
    return (bool(pub) and bool(dense) and reads and one_writer
            and ordered(pub, "for (int device_id = 0; device_id < GGML_SYCL_MAX_DEVICES; ++device_id) {",
                        "planned_dense_scratch_publish_locked(device_id);")
            and t.count("planned_dense_scratch_publish_locked(device_id);") == 1
            and ordered(dense, "zone_mmq_src1_scratch_bytes(", "zone_dequant_f16_plan_bytes(",
                        "g_planned_mmq_src1_scratch_bytes[device_id].store(q8_valid ? q8_bytes : 0,",
                        "g_planned_dense_scratch_invalid[device_id][0].store(!q8_valid,"))


# ---- D5: the runtime re-plan ----------------------------------------------------------------------------------------------


def claim_replan_rederives_the_owners_own_inputs(cache: str) -> bool:
    """The re-plan reads only the owner's own inputs, in the same locked step as both plan stores and the publish, and
    an owner that planned nothing on the device re-derives nothing."""
    b = body(norm(cache), REPLAN_SIG)
    gets = ("planned_load_term::DENSE_SCRATCH_N_UBATCH, &owner_n_ubatch)", "owner_n_ubatch == 0 ||",
            "planned_load_term::MMQ_SRC1_BYTES_PER_TOKEN, &q8_per_token)",
            "planned_load_term::DEQUANT_F16_WEIGHT_BYTES, &f16_weights)",
            "planned_load_term::DEQUANT_F16_SRC1_BYTES_PER_TOKEN, &f16_per_token)")
    return (bool(b) and ordered(b, LOCK, *gets,
                                "planned_mmq_src1_scratch_store_locked(device_id, owner, q8_per_token, n_ubatch, &q8, "
                                "nullptr)",
                                "planned_dequant_f16_scratch_store_locked(device_id, owner, f16_weights, f16_per_token, "
                                "n_ubatch, &f16, nullptr)",
                                PUBLISH_CALL, "return q8 && f16;")
            and b.count("g_planned_load_terms.owners.get(device_id, owner,") == 4
            and ".merged(" not in b and "_bytes_per_token[device_id]" not in b
            and "_weight_bytes_max[device_id]" not in b)


def claim_owner_n_ubatch_reads_the_owner(cache: str) -> bool:
    """The rollback's earlier value is the owner's own n_ubatch, read under the lock, 0 when it planned nothing."""
    b = body(norm(cache), OWNER_NUB_SIG)
    return bool(b) and ordered(
        b, LOCK, "if (!g_planned_load_terms.owners.get(device_id, owner, planned_load_term::DENSE_SCRATCH_N_UBATCH, "
        "&n_ubatch) || n_ubatch > UINT32_MAX) { return 0; }", "return static_cast<uint32_t>(n_ubatch);")


# ---- D6: no device-global merge remains ------------------------------------------------------------------------------------


def claim_no_device_global_merge_remains(sycl: str, cache: str, cache_h: str, zone: str, zone_h: str) -> bool:
    """The live-flag merge, the flag and the context census that fed it are gone from the backend."""
    texts = [norm(x) for x in (sycl, cache, cache_h, zone, zone_h)]
    return all("zone_dense_scratch_merge_input" not in t and not re.search(r"\bother_model_live\b", t)
               and "ggml_sycl_other_backend_context_live" not in t for t in texts)


# ---- D7: the load-time owner ------------------------------------------------------------------------------------------------


def claim_load_plan_is_keyed_by_the_open_load(sycl: str) -> bool:
    """populate_inventory_globals takes one load effect, the open load's on this thread, before both plan stores, and
    keys both by that effect's owner; with no effect it stores nothing."""
    t = norm(sycl)
    b = body(t, POPULATE_SIG)
    helper = body(t, EFFECT_HELPER_SIG)
    return (bool(b) and ordered(
        b, "const auto plan_effect = ggml_sycl_acquire_plan_load_effect();",
        "plan_effect && ggml_sycl::unified_cache_set_planned_mmq_src1_scratch(ctx->device, plan_effect.owner.owner,",
        "plan_effect && ggml_sycl::unified_cache_set_planned_dequant_f16_scratch(ctx->device, plan_effect.owner.owner,")
        and b.count("ggml_sycl_acquire_plan_load_effect()") == 1
        and b.count("unified_cache_set_planned_mmq_src1_scratch(") == 1
        and b.count("unified_cache_set_planned_dequant_f16_scratch(") == 1
        and helper == ("{ return ggml_sycl::lifecycle::global_registry().acquire_load_effect({ "
                       "g_sycl_plan_load_effect_txn }); }")
        and t.count("unified_cache_set_planned_mmq_src1_scratch(") == 1
        and t.count("unified_cache_set_planned_dequant_f16_scratch(") == 1)


# ---- D8: the runtime-context transaction ------------------------------------------------------------------------------------


def claim_runtime_txn_replans_the_published_models_own_plan(sycl: str) -> bool:
    """The transaction re-plans the published plan's model's own inputs, its guard records and restores that owner's own
    n_ubatch, and nothing else in the backend re-plans."""
    t = norm(sycl)
    txn = body(t, TXN_SIG)
    guard = body(t, GUARD_SIG)
    return (bool(txn) and bool(guard)
            and ordered(txn, "ggml_sycl_dense_scratch_txn_guard dense_guard(*ctx, current_token.owner, !probe_mode);",
                        "ggml_sycl::unified_cache_replan_planned_dense_scratch(ctx->device, current_token.owner, "
                        "next_kv_info.n_ubatch);")
            and ordered(guard, "plan_owner(model_owner),",
                        "prev_n_ubatch = ggml_sycl::unified_cache_get_planned_dense_scratch_owner_n_ubatch(device, "
                        "plan_owner);",
                        "(void) ggml_sycl::unified_cache_replan_planned_dense_scratch(device, plan_owner, "
                        "prev_n_ubatch);")
            and "unified_cache_get_planned_dense_scratch_n_ubatch(" not in guard
            and t.count("unified_cache_replan_planned_dense_scratch(") == 2)


# ---- D9: the decision is recorded ----------------------------------------------------------------------------------------


def claim_decision_is_recorded(contract: str, audit: str) -> bool:
    """Canonical contract 12.4 carries a row for the dense scratch inputs naming this ticket, and the cross-model
    audit records them as owner-keyed."""
    sec = contract[contract.find("### 12.4"):]
    sec = sec[:sec.find("\n### ", 1)] if "\n### " in sec[1:] else sec
    rows = [ln for ln in sec.splitlines() if ln.startswith("|") and "llama.cpp-45qo" in ln]
    return (any("dense scratch" in ln.lower() and "(slot, slotgeneration)" in ln.lower() for ln in rows)
            and "llama.cpp-45qo" in audit and "zone_dense_scratch_merge_input" in audit)


# ---- positive runs -------------------------------------------------------------------------------------------------------


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


def test_owner_n_ubatch_reads_the_owner():
    assert claim_owner_n_ubatch_reads_the_owner(CACHE)


def test_no_device_global_merge_remains():
    assert claim_no_device_global_merge_remains(SYCL, CACHE, CACHE_H, ZONE, ZONE_H)


def test_load_plan_is_keyed_by_the_open_load():
    assert claim_load_plan_is_keyed_by_the_open_load(SYCL)


def test_runtime_txn_replans_the_published_models_own_plan():
    assert claim_runtime_txn_replans_the_published_models_own_plan(SYCL)


def test_decision_is_recorded():
    assert claim_decision_is_recorded(CONTRACT, AUDIT)


# ---- mutants: each must fail the claim it names ----------------------------------------------------------------------------


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


def test_mutant_q8_setter_with_a_live_flag_fails():
    assert not claim_q8_setter_is_the_owner_store(
        _once(CACHE, "planned_load_terms_publish_locked(); }", "planned_load_terms_publish_locked(); "
              "g_planned_mmq_src1_scratch_bytes[device_id].store(0, std::memory_order_release); }", within=Q8_SET_SIG))


def test_mutant_q8_setter_publishing_outside_the_lock_fails():
    assert not claim_q8_setter_is_the_owner_store(
        _once(CACHE, "planned_load_terms_publish_locked(); } planned_load_terms_name_replaced(",
              "} planned_load_terms_publish_locked(); planned_load_terms_name_replaced(", within=Q8_SET_SIG))


def test_mutant_f16_setter_never_naming_the_sweep_fails():
    assert not claim_f16_setter_is_the_owner_store(
        _once(CACHE, "planned_load_terms_name_replaced(owner, device_id, replaced);", "", within=F16_SET_SIG))


def test_mutant_q8_store_validating_the_owner_alone_fails():
    assert not claim_q8_store_validates_the_merge_first(
        _once(CACHE, "zone_mmq_src1_scratch_bytes(merged_per_token,", "zone_mmq_src1_scratch_bytes(bytes_per_token,",
              within=Q8_STORE_SIG))


def test_mutant_q8_rejection_storing_the_input_fails():
    assert not claim_q8_store_validates_the_merge_first(
        _once(CACHE, "{ planned_load_term::MMQ_SRC1_BYTES_PER_TOKEN, 0 },",
              "{ planned_load_term::MMQ_SRC1_BYTES_PER_TOKEN, bytes_per_token },", within=Q8_STORE_SIG))


def test_mutant_f16_acceptance_keeping_the_flag_fails():
    assert not claim_f16_store_validates_the_merge_first(
        _once(CACHE, "{ planned_load_term::DEQUANT_F16_REJECTED, 0 },", "", within=F16_STORE_SIG))


def test_mutant_f16_store_validating_the_owner_alone_fails():
    assert not claim_f16_store_validates_the_merge_first(
        _once(CACHE, "zone_dequant_f16_plan_bytes(merged_weights,", "zone_dequant_f16_plan_bytes(max_weight_bytes,",
              within=F16_STORE_SIG))


def test_mutant_publish_skipping_the_dense_plan_fails():
    assert not claim_dense_plan_is_published_from_the_merge(
        _once(CACHE, "planned_dense_scratch_publish_locked(device_id);", "", within=PUBLISH_SIG))


def test_mutant_second_writer_of_a_dense_value_fails():
    assert not claim_dense_plan_is_published_from_the_merge(
        _once(CACHE, "uint32_t unified_cache_get_planned_dense_scratch_n_ubatch(int device_id) {",
              "uint32_t unified_cache_get_planned_dense_scratch_n_ubatch(int device_id) { "
              "g_planned_dense_scratch_n_ubatch[0].store(0, std::memory_order_release);"))


def test_mutant_dense_publish_ignoring_a_flag_fails():
    assert not claim_dense_plan_is_published_from_the_merge(
        _once(CACHE, "owners.merged(device_id, planned_load_term::MMQ_SRC1_REJECTED) == 0 &&", "",
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
        _once(CACHE, "planned_load_terms_publish_locked(); return q8 && f16;", "return q8 && f16;", within=REPLAN_SIG))


def test_mutant_owner_n_ubatch_reading_the_device_fails():
    assert not claim_owner_n_ubatch_reads_the_owner(
        _once(CACHE, "return static_cast<uint32_t>(n_ubatch);",
              "return g_planned_dense_scratch_n_ubatch[device_id].load(std::memory_order_acquire);",
              within=OWNER_NUB_SIG))


def test_mutant_merge_helper_back_fails():
    assert not claim_no_device_global_merge_remains(
        SYCL, CACHE, CACHE_H, ZONE + "\nsize_t zone_dense_scratch_merge_input(size_t prev, size_t next, bool other_model_live)"
        " { return other_model_live && prev > next ? prev : next; }\n", ZONE_H)


def test_mutant_context_census_back_fails():
    assert not claim_no_device_global_merge_remains(
        SYCL + "\nstatic bool ggml_sycl_other_backend_context_live(int device, const ggml_backend_sycl_context * self);\n",
        CACHE, CACHE_H, ZONE, ZONE_H)


def test_mutant_load_plan_without_an_owner_fails():
    assert not claim_load_plan_is_keyed_by_the_open_load(
        _once(SYCL, "plan_effect && ggml_sycl::unified_cache_set_planned_mmq_src1_scratch(ctx->device, "
              "plan_effect.owner.owner,", "ggml_sycl::unified_cache_set_planned_mmq_src1_scratch(ctx->device, "
              "ggml_sycl::lifecycle::SlotToken{ 0, 1 },"))


def test_mutant_load_plan_from_the_live_mask_fails():
    assert not claim_load_plan_is_keyed_by_the_open_load(
        _once(SYCL, "const auto plan_effect = ggml_sycl_acquire_plan_load_effect();",
              "const auto plan_effect = ggml_sycl::lifecycle::global_registry().acquire_load_effect("
              "ggml_sycl::lifecycle::global_registry().bound_candidate());"))


def test_mutant_effect_helper_naming_another_txn_fails():
    assert not claim_load_plan_is_keyed_by_the_open_load(
        _once(SYCL, "acquire_load_effect({ g_sycl_plan_load_effect_txn })", "acquire_load_effect({ 1 })",
              within=EFFECT_HELPER_SIG))


def test_mutant_txn_replanning_another_owner_fails():
    assert not claim_runtime_txn_replans_the_published_models_own_plan(
        _once(SYCL, "unified_cache_replan_planned_dense_scratch(ctx->device, current_token.owner,",
              "unified_cache_replan_planned_dense_scratch(ctx->device, ggml_sycl::lifecycle::SlotToken{ 0, 1 },"))


def test_mutant_guard_restoring_the_device_merge_fails():
    assert not claim_runtime_txn_replans_the_published_models_own_plan(
        _once(SYCL, "prev_n_ubatch = ggml_sycl::unified_cache_get_planned_dense_scratch_owner_n_ubatch(device, "
              "plan_owner);", "prev_n_ubatch = ggml_sycl::unified_cache_get_planned_dense_scratch_n_ubatch(device);"))


def test_mutant_a_second_replan_site_fails():
    assert not claim_runtime_txn_replans_the_published_models_own_plan(
        _once(SYCL, "static void populate_inventory_globals(ggml_backend_sycl_context * ctx, "
              "const ggml_sycl_tensor_inventory * inventory) {",
              "static void populate_inventory_globals(ggml_backend_sycl_context * ctx, "
              "const ggml_sycl_tensor_inventory * inventory) { (void) ggml_sycl::unified_cache_replan_planned_dense_"
              "scratch(ctx->device, ggml_sycl::lifecycle::SlotToken{}, 512);"))


def test_mutant_contract_without_the_row_fails():
    assert not claim_decision_is_recorded(CONTRACT.replace("llama.cpp-45qo", "llama.cpp-XXXX"), AUDIT)


def test_mutant_audit_without_the_ticket_fails():
    assert not claim_decision_is_recorded(CONTRACT, AUDIT.replace("llama.cpp-45qo", "llama.cpp-XXXX"))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
