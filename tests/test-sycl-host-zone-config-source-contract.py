#!/usr/bin/env python3
# llama.cpp-glkg: pins the ordering/agreement/wiring fixes for the host-zone
# capacity bug (host zones configured lazily at S1-PRELOAD, 80+ seconds after
# weight loading already consumed the pinned pool's budget through the
# pre-zone runtime fallback -- see the task for the full trace). Each check
# below documents its own RED (git-archive extract of master 7b03ea03f, the
# commit this task branched from) and GREEN (this branch) state; run with
# --root <dir> to point at an extracted tree instead of this checkout.
#
# llama.cpp-jed8: the A-family checks pin the 8c8a0afae shape, not the
# 528397d03 candidate this gate was first written against. 528397d03 put the
# eager configure call inside the exported ggml_backend_sycl_set_tensor_inventory
# body, under g_tensor_inventory_mutex. 8c8a0afae moved it: the exported setter
# delegates to a private ggml_sycl_set_tensor_inventory_impl that finalizes the
# plan and captures the candidate snapshot UNDER the inventory lock, and the
# guarded ggml_backend_sycl_stage_inventory_plan wrapper provisions that
# retained snapshot OUTSIDE the lock (canonical contract 12.5). A 528397d03
# extract is the positive control for A3 (its setter body carries the call).
#
# llama.cpp-16el: check E's anchor moved with the reserve parse (lenient
# parse_env_mb_value -> strict ggml_sycl::detail::parse_host_reserve_mb) and
# E1/E2 pin the strict-parse and warn-and-fall-through behaviour.
#
# --self-test re-evaluates every check against a mutated copy of the text it
# reads and requires the check to flip to RED, so a check whose anchor silently
# stopped matching real code cannot pass vacuously.
import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--root", default=None, help="repo root to check (default: this checkout)")
parser.add_argument("--self-test", action="store_true", help="run mutation witnesses proving each check is not vacuous")
args = parser.parse_args()

root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
sycl_cpp = (root / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()
cache_cpp = (root / "ggml/src/ggml-sycl/unified-cache.cpp").read_text()
pool_cpp = (root / "ggml/src/ggml-sycl/pinned-pool.cpp").read_text()
env_doc = (root / "docs/backend/sycl-env-vars.md").read_text()


def function_window(source: str, signature_anchor: str, window: int = 12000) -> str:
    """Return the exact function body starting at `signature_anchor`, via
    brace-depth matching from the first '{' after the anchor to its matching
    '}'. `window` is an upper bound only (guards against a malformed/absent
    close), unlike the fixed-size window some other source-contract tests in
    this repo use -- a fixed window risks silently spilling into the NEXT
    function's body when two checked functions sit close together (as
    get_max_size and alloc_buffer do here), which would let one function's
    text satisfy a check meant to scope to a different function.

    Returns "" when the anchor is absent. Every check that asserts the
    ABSENCE of some text inside a window must therefore also require the
    window to be non-empty, or a renamed function would pass it vacuously.
    """
    idx = source.find(signature_anchor)
    if idx < 0:
        return ""
    start = source.find("{", idx)
    if start < 0:
        return ""
    depth = 0
    end = min(len(source), start + window)
    for i in range(start, end):
        c = source[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return source[start:i + 1]
    return source[start:end]


def statement_window(source: str, statement_anchor: str, window: int = 1400) -> str:
    """Return `window` chars starting AT `statement_anchor` itself.

    Unlike function_window(), the anchor here is not a function signature
    ending in '{' -- it is a mid-function statement, so searching for the
    next '{' after it would skip forward past an unrelated later block.
    """
    idx = source.find(statement_anchor)
    if idx < 0:
        return ""
    return source[idx:idx + window]


# Real code text the checks anchor on (never comments).
LOCK_STMT = "std::lock_guard<std::mutex> lock(g_tensor_inventory_mutex);"
PLAN_STORE_CALL = "compute_and_store_plan_for_inventory(ctx, vram_budget, budget_pct);"
IMPL_DELEGATE_CALL = "ggml_sycl_set_tensor_inventory_impl(backend, inventory);"
SNAPSHOT_CAPTURE = "const auto snapshot = " + IMPL_DELEGATE_CALL
# The retained-candidate overload (snapshot->plan), not the current-plan
# overload S1-PRELOAD uses.
PROVISION_CALL = "ggml_sycl::ggml_sycl_configure_host_zones_for_plan(cache, snapshot->plan);"
CURRENT_PLAN_PROVISION_CALL = "ggml_sycl::ggml_sycl_configure_host_zones_for_plan(cache);"
CONFIGURE_TOKEN = "configure_host_zones"

# Every window the checks read, keyed by name. Witness mutants operate on a
# copy of this dict so each check can be re-evaluated against mutated text.
texts = {
    "stage_fn": function_window(
        sycl_cpp,
        "ggml_sycl_lifecycle_result ggml_backend_sycl_stage_inventory_plan(const ggml_sycl_tensor_inventory *   inventory,",
        window=30000),
    "impl_fn": function_window(
        sycl_cpp,
        "static std::shared_ptr<const ggml_sycl::lifecycle_plan_snapshot> ggml_sycl_set_tensor_inventory_impl(",
        window=30000),
    "setter_fn": function_window(
        sycl_cpp,
        "void ggml_backend_sycl_set_tensor_inventory(ggml_backend_t backend, const ggml_sycl_tensor_inventory * inventory) {"),
    "get_max_size_fn": function_window(
        sycl_cpp,
        "static size_t ggml_backend_sycl_host_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {"),
    "alloc_buffer_fn": function_window(
        sycl_cpp,
        "static ggml_backend_buffer_t ggml_backend_sycl_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,"),
    "populate_zone_sizing_fn": function_window(
        cache_cpp,
        "static void populate_host_zone_sizing(placement_plan &                           plan,",
        window=45000),
    # llama.cpp-16el: 3200 chars, not the 1400 default -- the strict-parse
    # switch (four cases, two log lines) sits after the glkg comment block and
    # the default window ended inside it.
    "host_arena_ctor_window": statement_window(
        cache_cpp, "size_t host_mem_budget = g_unified_cache_host_budget;", window=3200),
    "cache_cpp": cache_cpp,
    "pool_cpp": pool_cpp,
    "env_doc": env_doc,
}


# --- A: host zones provisioned eagerly, before create_tensor allocates any
# weight buffer -- not lazily at S1-PRELOAD, which runs after weight loading
# has already gone through the pre-zone runtime-pinned fallback. The eager
# call lives in the guarded stage_inventory_plan wrapper (8c8a0afae shape):
# the wrapper captures the candidate snapshot from the private setter impl,
# then provisions that exact retained snapshot, in that order, in the same
# function. On master the call existed only in S1-PRELOAD.
def check_a_eager_in_stage_wrapper(t):
    fn = t["stage_fn"]
    capture_idx = fn.find(SNAPSHOT_CAPTURE)
    provision_idx = fn.find(PROVISION_CALL)
    return (
        capture_idx >= 0
        and provision_idx >= 0
        # positional: provisioning must come AFTER the snapshot is captured
        and provision_idx > capture_idx
        # and it must consume the retained snapshot, never the current plan
        and CURRENT_PLAN_PROVISION_CALL not in fn
    )


def witness_a(t):
    t["stage_fn"] = t["stage_fn"].replace(PROVISION_CALL, "")
    return t


# --- A1: the inventory lock is scoped to the private impl, which finalizes
# the plan under it; the stage wrapper that does the physical provisioning
# never takes it (canonical 12.5: provision the retained plan OUTSIDE the
# lock). Both halves are required -- the impl window must be non-empty and
# hold the lock plus the plan store, so a renamed impl cannot pass the
# absence half vacuously.
def check_a1_lock_scoped_to_impl(t):
    impl = t["impl_fn"]
    stage = t["stage_fn"]
    return (
        impl != ""
        and stage != ""
        and LOCK_STMT in impl
        and PLAN_STORE_CALL in impl
        and impl.find(LOCK_STMT) < impl.find(PLAN_STORE_CALL)
        and LOCK_STMT not in stage
    )


def witness_a1(t):
    # Reintroduce the 528397d03 shape's lock into the provisioning path.
    t["stage_fn"] = t["stage_fn"].replace(PROVISION_CALL, LOCK_STMT + "\n" + PROVISION_CALL)
    return t


# --- A3 (xgr6 invariant): neither setter body -- the exported
# ggml_backend_sycl_set_tensor_inventory nor its private impl -- contains a
# configure_host_zones call. 528397d03 had the eager call inside the exported
# setter, under the lock; that shape is the RED this pins against. Absence
# checks fail closed: both windows must be non-empty and the exported setter
# must actually delegate to the impl.
def check_a3_no_configure_in_setter(t):
    impl = t["impl_fn"]
    setter = t["setter_fn"]
    return (
        impl != ""
        and setter != ""
        and IMPL_DELEGATE_CALL in setter
        and CONFIGURE_TOKEN not in setter
        and CONFIGURE_TOKEN not in impl
    )


def witness_a3(t):
    # The 528397d03 defect: eager provisioning inside the exported setter.
    t["setter_fn"] = t["setter_fn"].replace(
        "(void) " + IMPL_DELEGATE_CALL,
        "(void) " + IMPL_DELEGATE_CALL + "\n    " + CURRENT_PLAN_PROVISION_CALL)
    return t


# --- B: ggml_backend_sycl_host_buffer_type_get_max_size must mirror the zone
# select_zone() (unified_alloc, unified-cache.cpp) ACTUALLY uses for this
# buft's requests -- WEIGHT, unconditionally, because alloc_buffer always sets
# category=HOST_COMPUTE and select_zone's WEIGHT branch is checked before any
# role-only STAGING fallback. The old code re-derived alloc_buffer's ROLE
# computation (in_model_load || weights_evictable) instead, which disagreed
# with the real zone once in_model_load went false post-load.
TARGET_ZONE_WEIGHT = "constexpr ggml_sycl::host_zone_id target_zone = ggml_sycl::host_zone_id::WEIGHT;"
OLD_ROLE_TERNARY = "in_model_load || weights_evictable"


def check_b_mirror(t):
    fn = t["get_max_size_fn"]
    return TARGET_ZONE_WEIGHT in fn and OLD_ROLE_TERNARY not in fn


def witness_b_mirror(t):
    t["get_max_size_fn"] = t["get_max_size_fn"].replace(
        TARGET_ZONE_WEIGHT,
        "const ggml_sycl::host_zone_id target_zone = (in_model_load || weights_evictable) ? "
        "ggml_sycl::host_zone_id::WEIGHT : ggml_sycl::host_zone_id::STAGING;")
    return t


# select_zone's real precedence (unified-cache.cpp): WEIGHT branch fires on
# role==WEIGHT OR cat==HOST_COMPUTE OR cat==EXPERT_CACHE, checked before the
# STAGING fallback. This is the reason B's unconditional WEIGHT is correct
# for this call site (alloc_buffer always sets category=HOST_COMPUTE) --
# pinned so a future rewrite of select_zone's precedence is forced to revisit
# the mirror in get_max_size too.
SELECT_ZONE_WEIGHT_BRANCH = "role == alloc_role::WEIGHT || cat == runtime_category::HOST_COMPUTE ||"
SELECT_ZONE_EXPERT_BRANCH = "cat == runtime_category::EXPERT_CACHE"


def check_b_select_zone(t):
    src = t["cache_cpp"]
    return SELECT_ZONE_WEIGHT_BRANCH in src and SELECT_ZONE_EXPERT_BRANCH in src


def witness_b_select_zone(t):
    # Drop HOST_COMPUTE from the WEIGHT branch: the runtime-category route
    # that alloc_buffer relies on.
    t["cache_cpp"] = t["cache_cpp"].replace(SELECT_ZONE_WEIGHT_BRANCH, "role == alloc_role::WEIGHT ||")
    return t


# alloc_buffer's category must still be unconditionally HOST_COMPUTE for this
# buft -- if it ever became conditional, B's unconditional WEIGHT would need
# to change with it.
ALLOC_BUFFER_CATEGORY = "req.intent.category                     = ggml_sycl::runtime_category::HOST_COMPUTE;"


def check_b_alloc_buffer_category(t):
    return ALLOC_BUFFER_CATEGORY in t["alloc_buffer_fn"]


def witness_b_alloc_buffer_category(t):
    t["alloc_buffer_fn"] = t["alloc_buffer_fn"].replace(
        ALLOC_BUFFER_CATEGORY,
        "req.intent.category = in_model_load ? ggml_sycl::runtime_category::HOST_COMPUTE : "
        "ggml_sycl::runtime_category::STAGING;")
    return t


# --- C: the WEIGHT zone (where every runtime HOST_COMPUTE request lands,
# per B/select_zone above) reserves headroom for the context's own host
# buffers (llama_context::output_reserve() et al.), not just weight bytes.
def check_c_headroom(t):
    fn = t["populate_zone_sizing_fn"]
    return (
        "k_host_runtime_reserve_bytes" in fn
        and "plan.host_zone_weight_bytes =" in fn
        and "k_host_runtime_reserve_bytes;" in fn
    )


def witness_c_headroom(t):
    t["populate_zone_sizing_fn"] = t["populate_zone_sizing_fn"].replace(" +\n        k_host_runtime_reserve_bytes;", ";")
    return t


# --- D: the pre-zone runtime-pinned fallback (the actual path that consumed
# the pool's budget before configure_zones ever ran, on master) now WARNs
# per chunk grown, naming the running total against the budget.
FALLBACK_WARN_FMT = (
    '"[HOST-ARENA] runtime pinned chunk %zu: +%.1f MB, %.1f of %.1f GB committed before host zones exist\\n"')


def check_d_fallback_warn(t):
    return FALLBACK_WARN_FMT in t["pool_cpp"]


def witness_d_fallback_warn(t):
    t["pool_cpp"] = t["pool_cpp"].replace(FALLBACK_WARN_FMT, '"[HOST-ARENA] runtime pinned chunk grown\\n"')
    return t


# --- A2: configure_zones()'s capacity-shortfall WARN names the same
# accumulator the growth guard reads (total_allocated_, which also counts
# runtime_chunks_) instead of only the chunks_-only total_capacity the old
# message printed next to the footprint with no other context.
CAPACITY_WARN_FMT = "zone-chunk bytes=%.1f MB, runtime-chunk bytes=%.1f MB, total committed=%.1f MB"


def check_a2_capacity_warn(t):
    src = t["pool_cpp"]
    return "runtime_capacity" in src and CAPACITY_WARN_FMT in src


def witness_a2_capacity_warn(t):
    t["pool_cpp"] = t["pool_cpp"].replace(CAPACITY_WARN_FMT, "capacity=%.1f MB")
    return t


# --- E: GGML_SYCL_HOST_RESERVE_MB wired as a real override of the pinned
# pool's BUDGET at construction, not parsed only by a callerless function.
# llama.cpp-16el moved the parse from the lenient parse_env_mb_value() helper
# to the strict ggml_sycl::detail::parse_host_reserve_mb() (host-reserve-env.cpp);
# the anchor is the getenv read plus that call, both inside the ctor window.
DEAD_RESOLVER_SIG = "static size_t resolve_host_reserve_bytes("
RESERVE_ENV_READ = 'std::getenv("GGML_SYCL_HOST_RESERVE_MB")'
RESERVE_PARSE_CALL = "ggml_sycl::detail::parse_host_reserve_mb(reserve_raw)"


def check_e_reserve_wired(t):
    win = t["host_arena_ctor_window"]
    return DEAD_RESOLVER_SIG not in t["cache_cpp"] and RESERVE_ENV_READ in win and RESERVE_PARSE_CALL in win


def witness_e_reserve_wired(t):
    t["cache_cpp"] = t["cache_cpp"] + "\nstatic size_t resolve_host_reserve_bytes(size_t s) { return s; }\n"
    return t


# --- E1 (llama.cpp-16el): the reserve variable never goes back through the
# lenient helper, which accepts `6144junk` as 6144, ignores ERANGE and lets the
# MiB->byte multiply wrap. Absence check, so it fails closed: the strict call
# must be present in the same window for the absence half to count.
LENIENT_RESERVE_PARSE = 'parse_env_mb_value("GGML_SYCL_HOST_RESERVE_MB"'


def check_e1_no_lenient_reserve_parse(t):
    win = t["host_arena_ctor_window"]
    return win != "" and RESERVE_PARSE_CALL in win and LENIENT_RESERVE_PARSE not in t["cache_cpp"]


def witness_e1_no_lenient_reserve_parse(t):
    # The pre-16el shape: the budget parsed by the shared lenient helper.
    t["cache_cpp"] = t["cache_cpp"].replace(
        RESERVE_PARSE_CALL,
        'parse_env_mb_value("GGML_SYCL_HOST_RESERVE_MB", host_reserve_mb)')
    t["host_arena_ctor_window"] = t["host_arena_ctor_window"].replace(
        RESERVE_PARSE_CALL,
        'parse_env_mb_value("GGML_SYCL_HOST_RESERVE_MB", host_reserve_mb)')
    return t


# --- E2 (llama.cpp-16el): a rejected value is WARNed once, naming the
# variable, the raw string and the reason, and falls through to the auto
# calculation (the REJECTED case assigns nothing to host_mem_budget). Anchors
# on the WARN's format text and the case label, both real code.
REJECTED_CASE = "case ggml_sycl::detail::host_reserve_parse_status::REJECTED:"
REJECTED_WARN_FMT = "\"[HOST-ARENA] ignoring GGML_SYCL_HOST_RESERVE_MB='%s': %s; using the auto-computed \""


def check_e2_rejection_warns_and_falls_through(t):
    win = t["host_arena_ctor_window"]
    case_idx = win.find(REJECTED_CASE)
    warn_idx = win.find(REJECTED_WARN_FMT)
    if case_idx < 0 or warn_idx < 0 or warn_idx < case_idx:
        return False
    # Between the case label and its break, nothing may assign the budget:
    # rejection must leave host_mem_budget == 0 so the auto calculation runs.
    end_idx = win.find("break;", warn_idx)
    return end_idx > warn_idx and "host_mem_budget =" not in win[case_idx:end_idx]


def witness_e2_rejection_warns_and_falls_through(t):
    # Silent rejection: drop the WARN.
    t["host_arena_ctor_window"] = t["host_arena_ctor_window"].replace(REJECTED_WARN_FMT, '"rejected"')
    return t


# --- E (docs): the env-var catalog documents the wiring and disambiguates it
# from the cache's own small staging buffer.
def check_e_docs(t):
    doc = t["env_doc"]
    return (
        "GGML_SYCL_HOST_RESERVE_MB=<MB>" in doc
        and "GGML_SYCL_HOST_STAGING_MB=<MB>" in doc
        and "a pinned-pool zone, and not the pool's overall budget" in doc
    )


def witness_e_docs(t):
    t["env_doc"] = t["env_doc"].replace("GGML_SYCL_HOST_RESERVE_MB=<MB>", "")
    return t


# (name, check, witness): the witness reproduces the defect textually in a
# copy of `texts`; the check must then fail against that copy.
checks = [
    ("A: host zones provisioned eagerly in stage_inventory_plan after the snapshot capture, from the retained snapshot",
     check_a_eager_in_stage_wrapper, witness_a),
    ("A1: inventory lock scoped to the setter impl; provisioning in stage_inventory_plan runs outside it",
     check_a1_lock_scoped_to_impl, witness_a1),
    ("A3: neither the exported setter nor its impl contains a configure_host_zones call",
     check_a3_no_configure_in_setter, witness_a3),
    ("B: get_max_size mirrors select_zone's actual WEIGHT routing unconditionally",
     check_b_mirror, witness_b_mirror),
    ("B: select_zone's WEIGHT-before-STAGING precedence unchanged (the reason B is correct)",
     check_b_select_zone, witness_b_select_zone),
    ("B: alloc_buffer still sets category=HOST_COMPUTE unconditionally for this buft",
     check_b_alloc_buffer_category, witness_b_alloc_buffer_category),
    ("C: WEIGHT zone reserves headroom for runtime host-compute requests",
     check_c_headroom, witness_c_headroom),
    ("D: pre-zone runtime chunk growth WARNs with running total vs budget",
     check_d_fallback_warn, witness_d_fallback_warn),
    ("A2: configure_zones capacity-shortfall WARN names budget/zone/runtime bytes",
     check_a2_capacity_warn, witness_a2_capacity_warn),
    ("E: GGML_SYCL_HOST_RESERVE_MB wired at budget construction, dead resolver removed",
     check_e_reserve_wired, witness_e_reserve_wired),
    ("E1: the reserve variable is parsed by the strict parser, never the lenient helper",
     check_e1_no_lenient_reserve_parse, witness_e1_no_lenient_reserve_parse),
    ("E2: a rejected reserve value WARNs once and falls through to the auto calculation",
     check_e2_rejection_warns_and_falls_through, witness_e2_rejection_warns_and_falls_through),
    ("E: env-var catalog documents the wiring and the staging-buffer distinction",
     check_e_docs, witness_e_docs),
]

failed = [name for name, check, _witness in checks if not check(texts)]

mutant_failures = []
witnessed = 0
if args.self_test:
    for name, check, witness in checks:
        mutated = witness(dict(texts))
        if mutated == texts:
            mutant_failures.append(name + ": witness did not change the text (anchor missing)")
        elif check(mutated):
            mutant_failures.append(name + ": mutation was not detected (check is vacuous)")
        else:
            witnessed += 1
            print("host-zone-config source contract: witness flipped -- " + name)

if failed or mutant_failures:
    if failed:
        print("host-zone-config source contract failed: " + ", ".join(failed), file=sys.stderr)
    if mutant_failures:
        print("host-zone-config source self-test failed: " + ", ".join(mutant_failures), file=sys.stderr)
    raise SystemExit(1)

if args.self_test:
    print("host-zone-config source contract: self-test PASS (%d witness mutants, one per check)" % witnessed)
print("host-zone-config source contract: PASS")
