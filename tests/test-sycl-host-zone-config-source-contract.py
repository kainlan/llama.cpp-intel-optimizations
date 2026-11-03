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
# guarded ggml_backend_sycl_stage_load_inventory_plan wrapper provisions that
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
import re
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--root", default=None, help="repo root to check (default: this checkout)")
parser.add_argument("--self-test", action="store_true", help="run mutation witnesses proving each check is not vacuous")
args = parser.parse_args()

root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
sycl_cpp = (root / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()
cache_cpp = (root / "ggml/src/ggml-sycl/unified-cache.cpp").read_text()
cache_hpp = (root / "ggml/src/ggml-sycl/unified-cache.hpp").read_text()
pool_cpp = (root / "ggml/src/ggml-sycl/pinned-pool.cpp").read_text()
plan_cpp = (root / "ggml/src/ggml-sycl/host-zone-plan.cpp").read_text()
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
        "ggml_sycl_lifecycle_result ggml_backend_sycl_stage_load_inventory_plan("
        "const ggml_sycl_tensor_inventory *   inventory,",
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
    "intent_fn": function_window(
        sycl_cpp, "static ggml_sycl::alloc_intent ggml_sycl_host_buft_intent(ggml_sycl_host_buft_kind kind) {"),
    "host_max_size_fn": function_window(
        sycl_cpp,
        "static size_t ggml_sycl_host_buft_max_size(ggml_backend_buffer_type_t buft, int device, "
        "ggml_sycl::host_zone_id zone) {"),
    "host_alloc_fn": function_window(
        sycl_cpp, "static ggml_backend_buffer_t ggml_sycl_host_buft_alloc(ggml_backend_buffer_type_t buft,"),
    "kv_get_max_size_fn": function_window(
        sycl_cpp, "static size_t ggml_backend_sycl_kv_host_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {"),
    "kv_alloc_buffer_fn": function_window(
        sycl_cpp,
        "static ggml_backend_buffer_t ggml_backend_sycl_kv_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,"),
    "kv_buft_fn": function_window(sycl_cpp, "ggml_backend_buffer_type_t ggml_backend_sycl_kv_host_buffer_type() {"),
    "hc_get_max_size_fn": function_window(
        sycl_cpp,
        "static size_t ggml_backend_sycl_host_compute_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {"),
    "hc_alloc_fn": function_window(
        sycl_cpp,
        "static ggml_backend_buffer_t ggml_backend_sycl_host_compute_buffer_alloc(ggml_backend_buffer_type_t buft,",
        window=20000),
    "hc_iface": statement_window(
        sycl_cpp, "static const ggml_backend_buffer_type_i ggml_backend_sycl_host_compute_buffer_type_interface = {",
        window=700),
    "select_zone_fn": function_window(plan_cpp, "host_zone_id select_zone(alloc_role       role,"),
    "populate_zone_sizing_fn": function_window(
        cache_cpp,
        "static void populate_host_zone_sizing(placement_plan &                           plan,",
        window=45000),
    "weight_zone_terms_fn": function_window(
        cache_cpp,
        "host_weight_zone_terms placement_plan_host_weight_zone_terms(const placement_plan & plan) {"),
    "charge_host_weight_fn": function_window(
        cache_hpp, "void charge_host_weight(placement_host_backing backing, size_t bytes) noexcept {"),
    "uncharge_host_weight_fn": function_window(
        cache_hpp, "void uncharge_host_weight(placement_host_backing backing, size_t bytes) noexcept {"),
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
# call lives in the guarded stage_load_inventory_plan wrapper (8c8a0afae shape):
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


# --- B: the host bufts' size and alloc share one builder (design 5efe section 7, the C4 part f clauses "The host
# bufts' size and alloc share one builder", amending this gate). Before C4, SYCL_Host's get_max_size hard-coded
# WEIGHT because every HOST_COMPUTE request routed there; after C4 a post-load SYCL_Host buffer is a context's and
# routes to SCRATCH, so a get_max_size that names its own zone chunks against the wrong one (the glkg shape). Each
# buft's get_max_size therefore builds the same intent its alloc_buffer builds and asks select_zone, on the buft's
# own device.
HOST_INTENT = "ggml_sycl_host_buft_intent(ggml_sycl_host_buft_kind::HOST)"
KV_INTENT = "ggml_sycl_host_buft_intent(ggml_sycl_host_buft_kind::KV_HOST)"
TP_INTENT = "ggml_sycl_host_buft_intent(ggml_sycl_host_buft_kind::HOST_COMPUTE_TP)"
SELECT_FROM_INTENT = ("ggml_sycl::select_zone(intent.role, intent.category, intent.lifetime,\n"
                      "                                                                  "
                      "intent.constraints.persistent_owner, false);")
HOST_DEVICE = "const int                     device = ggml_sycl_host_buft_device(buft);"


def flat_ws(text):
    return re.sub(r"\s+", " ", text)


def check_b_host_builder(t):
    size_fn, alloc_fn = t["get_max_size_fn"], t["alloc_buffer_fn"]
    return (
        bool(size_fn) and bool(alloc_fn)
        and HOST_INTENT in size_fn and SELECT_FROM_INTENT in size_fn and HOST_DEVICE in size_fn
        and "return ggml_sycl_host_buft_alloc(buft, size, " + HOST_INTENT + ");" in alloc_fn
    )


def witness_b_host_builder(t):
    # SYCL_Host's get_max_size with WEIGHT hard-coded again.
    t["get_max_size_fn"] = t["get_max_size_fn"].replace(SELECT_FROM_INTENT, "ggml_sycl::host_zone_id::WEIGHT;")
    return t


# No host get_max_size names a zone or calls get_current_device_id(), and SYCL_Host's and SYCL_KV_Host's paths take
# the device from ggml_sycl_host_buft_device and nowhere else (no ctx->device on those paths). The shared helper the
# three answer through takes the zone as a parameter and names none.
HOST_SIZE_WINDOWS = ("get_max_size_fn", "kv_get_max_size_fn", "hc_get_max_size_fn", "host_max_size_fn")


def check_b_no_zone_no_current_device(t):
    for key in HOST_SIZE_WINDOWS:
        fn = t[key]
        if not fn or "host_zone_id::" in fn or "get_current_device_id(" in fn:
            return False
    for key in ("get_max_size_fn", "alloc_buffer_fn", "kv_get_max_size_fn", "kv_alloc_buffer_fn", "host_alloc_fn"):
        fn = t[key]
        if not fn or "get_current_device_id(" in fn or "ctx->device" in fn:
            return False
    return "const int exact_device = ggml_sycl_host_buft_device(buft);" in t["host_alloc_fn"]


def witness_b_no_zone_no_current_device(t):
    # SYCL_Host's get_max_size back on get_current_device_id().
    t["get_max_size_fn"] = t["get_max_size_fn"].replace("ggml_sycl_host_buft_device(buft)", "get_current_device_id()")
    return t


# SYCL_KV_Host has its own iface: its own get_name, and its alloc_buffer and get_max_size slots call the builder with
# role KV. A copy of SYCL_Host's iface would allocate host KV with SYCL_Host's role (design 5efe section M57 (a)).
def check_b_kv_own_iface(t):
    buft = flat_ws(t["kv_buft_fn"])
    return (
        "/* .get_name = */ ggml_backend_sycl_kv_host_buffer_type_name," in buft
        and "/* .alloc_buffer = */ ggml_backend_sycl_kv_host_buffer_type_alloc_buffer," in buft
        and "/* .get_max_size = */ ggml_backend_sycl_kv_host_buffer_type_get_max_size," in buft
        and "ggml_backend_sycl_host_buffer_type()->iface" not in buft
        and KV_INTENT in t["kv_get_max_size_fn"] and SELECT_FROM_INTENT in t["kv_get_max_size_fn"]
        and "return ggml_sycl_host_buft_alloc(buft, size, " + KV_INTENT + ");" in t["kv_alloc_buffer_fn"]
        and "intent.role      = ggml_sycl::alloc_role::KV;" in t["intent_fn"]
    )


def witness_b_kv_own_iface(t):
    # SYCL_KV_Host back on the copied SYCL_Host iface.
    t["kv_buft_fn"] = t["kv_buft_fn"].replace(
        "ggml_backend_sycl_kv_host_buffer_type_alloc_buffer,",
        "ggml_backend_sycl_host_buffer_type()->iface.alloc_buffer,")
    return t


# SYCL_HostCompute's iface names its own get_max_size, which asks ggml_sycl_host_compute_is_tp(), the predicate its
# alloc_buffer asks, and outside TP mode delegates to the device buft's get_max_size.
def check_b_host_compute(t):
    fn = t["hc_get_max_size_fn"]
    return (
        "/* .get_max_size     = */ ggml_backend_sycl_host_compute_buffer_type_get_max_size," in t["hc_iface"]
        and "if (!ggml_sycl_host_compute_is_tp()) {\n        return ggml_backend_sycl_buffer_type_get_max_size(buft);"
        in fn
        and TP_INTENT in fn and SELECT_FROM_INTENT.replace("\n" + " " * 66, "\n" + " " * 68) in fn
        and "if (ggml_sycl_host_compute_is_tp()) {" in t["hc_alloc_fn"]
    )


def witness_b_host_compute(t):
    # The glkg defect reversed: the host builder's answer outside TP mode.
    t["hc_get_max_size_fn"] = t["hc_get_max_size_fn"].replace(
        "return ggml_backend_sycl_buffer_type_get_max_size(buft);",
        "return ggml_sycl_host_buft_max_size(buft, -1, ggml_sycl::host_zone_id::STAGING);")
    return t


# select_zone has no category clause for WEIGHT and sends UNOWNED there: the WEIGHT returns are role WEIGHT and
# UNOWNED-or-persistent, and the HOST_COMPUTE/EXPERT_CACHE category clause returns SCRATCH.
def check_b_select_zone(t):
    fn = flat_ws(t["select_zone_fn"])
    weight_conds = re.findall(r"if \(([^{]*)\) \{ return host_zone_id::WEIGHT; \}", fn)
    return (
        weight_conds == ["role == alloc_role::WEIGHT", "lifetime == alloc_lifetime::UNOWNED || persistent_owner"]
        and "if (category == runtime_category::HOST_COMPUTE || category == runtime_category::EXPERT_CACHE) "
            "{ return host_zone_id::SCRATCH; }" in fn
    )


def witness_b_select_zone(t):
    # The pre-C4 category clause for WEIGHT restored.
    t["select_zone_fn"] = t["select_zone_fn"].replace(
        "lifetime == alloc_lifetime::UNOWNED || persistent_owner",
        "lifetime == alloc_lifetime::UNOWNED || persistent_owner || category == runtime_category::HOST_COMPUTE")
    return t


# SYCL_Host takes role WEIGHT only under in_model_load; weights_evictable no longer chooses its role.
def check_b_weight_only_in_load(t):
    fn = t["intent_fn"]
    load = fn.find("if (g_sycl_in_model_load.load(std::memory_order_acquire)) {")
    weight = fn.find("intent.role     = ggml_sycl::alloc_role::WEIGHT;")
    other = fn.find("} else {", load)
    compute = fn.find("intent.role     = ggml_sycl::alloc_role::COMPUTE;", other)
    return (
        0 <= load < weight < other < compute
        and fn.count("alloc_role::WEIGHT") == 1
        and "weights_evictable" not in fn
    )


def witness_b_weight_only_in_load(t):
    t["intent_fn"] = t["intent_fn"].replace(
        "if (g_sycl_in_model_load.load(std::memory_order_acquire)) {",
        "if (g_sycl_in_model_load.load(std::memory_order_acquire) || ggml_backend_sycl_weights_evictable()) {")
    return t


# --- C: the WEIGHT zone (where the UNOWNED runtime HOST_COMPUTE requests land,
# per select_zone above) reserves headroom for them, not just weight bytes. The
# zone is sized as the named terms of placement_plan_host_weight_zone_terms(),
# the HOST_COMPUTE reserve is one of them, and no ratio over the weights stands in for a term.
WEIGHT_ZONE_FROM_TERMS = "plan.host_zone_weight_bytes = weight_terms.total();"
RESERVE_TERM = "terms.host_compute_reserve = k_host_compute_reserve_bytes;"


def check_c_headroom(t):
    terms_fn = t["weight_zone_terms_fn"]
    return (
        bool(terms_fn)
        and "k_host_compute_reserve_bytes = 64ull * 1024ull * 1024ull;" in terms_fn
        and RESERVE_TERM in terms_fn
    )


def witness_c_headroom(t):
    t["weight_zone_terms_fn"] = t["weight_zone_terms_fn"].replace(RESERVE_TERM, "terms.host_compute_reserve = 0;")
    return t


# --- C2: the WEIGHT zone is the sum of those named terms, never a ratio over
# the host weights (slack nobody can name is a second source for a fact the
# plan already knows).
# Any write to the zone size counts, compound ones included (`*= k_..._ratio`
# after the terms assignment would otherwise pass); `==` comparisons do not.
WEIGHT_ZONE_WRITE_RE = re.compile(r"host_zone_weight_bytes\s*[-+*/]?=(?!=)")


def check_c2_weight_zone_is_terms(t):
    fn = t["populate_zone_sizing_fn"]
    terms_fn = t["weight_zone_terms_fn"]
    return (
        bool(fn) and bool(terms_fn)
        and "placement_plan_host_weight_zone_terms(plan)" in fn
        and WEIGHT_ZONE_FROM_TERMS in fn
        and len(WEIGHT_ZONE_WRITE_RE.findall(fn)) == 1
        and "_ratio" not in fn
        and "_ratio" not in terms_fn
    )


def witness_c2_weight_zone_is_terms(t):
    t["populate_zone_sizing_fn"] = t["populate_zone_sizing_fn"].replace(
        WEIGHT_ZONE_FROM_TERMS,
        "plan.host_zone_weight_bytes = static_cast<size_t>(plan.weight_host_bytes * k_weight_headroom_ratio);")
    return t


# --- C3: the walk that sums the host weights and the plan's charge of
# weight_host_bytes ask one predicate which host backings the WEIGHT zone
# holds, so the two figures the [HOST-PLAN-BUG] WARN compares cannot disagree
# on which entries they count.
WALK_ASKS_PREDICATE = "placement_plan::host_backing_sizes_weight_zone(entry.host_backing)"
CHARGE_ASKS_PREDICATE = "host_backing_sizes_weight_zone(backing)"


def check_c3_one_backing_predicate(t):
    return (
        WALK_ASKS_PREDICATE in t["weight_zone_terms_fn"]
        and CHARGE_ASKS_PREDICATE in t["charge_host_weight_fn"]
        and CHARGE_ASKS_PREDICATE in t["uncharge_host_weight_fn"]
    )


def witness_c3_one_backing_predicate(t):
    t["weight_zone_terms_fn"] = t["weight_zone_terms_fn"].replace(WALK_ASKS_PREDICATE, "true")
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
    ("A: host zones provisioned eagerly in stage_load_inventory_plan after the snapshot capture, "
     "from the retained snapshot",
     check_a_eager_in_stage_wrapper, witness_a),
    ("A1: inventory lock scoped to the setter impl; provisioning in stage_load_inventory_plan runs outside it",
     check_a1_lock_scoped_to_impl, witness_a1),
    ("A3: neither the exported setter nor its impl contains a configure_host_zones call",
     check_a3_no_configure_in_setter, witness_a3),
    ("B: SYCL_Host's get_max_size and alloc_buffer share the host builder, and get_max_size asks select_zone",
     check_b_host_builder, witness_b_host_builder),
    ("B: no host get_max_size names a zone or calls get_current_device_id; the device is the buft's",
     check_b_no_zone_no_current_device, witness_b_no_zone_no_current_device),
    ("B: SYCL_KV_Host has its own iface whose slots call the builder with role KV",
     check_b_kv_own_iface, witness_b_kv_own_iface),
    ("B: SYCL_HostCompute's own get_max_size asks the TP predicate and delegates to the device buft outside TP",
     check_b_host_compute, witness_b_host_compute),
    ("B: select_zone has no category clause for WEIGHT and sends UNOWNED to WEIGHT",
     check_b_select_zone, witness_b_select_zone),
    ("B: SYCL_Host takes role WEIGHT only under in_model_load",
     check_b_weight_only_in_load, witness_b_weight_only_in_load),
    ("C: WEIGHT zone reserves headroom for runtime host-compute requests",
     check_c_headroom, witness_c_headroom),
    ("C2: WEIGHT zone is the sum of its named terms, not a ratio",
     check_c2_weight_zone_is_terms, witness_c2_weight_zone_is_terms),
    ("C3: the WEIGHT-term walk and charge_host_weight ask one backing predicate",
     check_c3_one_backing_predicate, witness_c3_one_backing_predicate),
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
