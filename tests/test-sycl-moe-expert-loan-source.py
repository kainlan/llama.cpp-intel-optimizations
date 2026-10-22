"""Source contract for llama.cpp-mwmr (plan Task 11b of llama.cpp-36j0): the expert loan transaction.

Host-only: reads sources, builds nothing, loads no model and touches no device. The behaviour is the device test
sycl-moe-expert-loan (lead-run); this gate pins the properties a later edit could undo while that test still passes, or
that it cannot see at all.

A prompt phase lends the device's least popular resident expert triplets to the prefill ring. The transaction
(ggml_sycl_expert_loan_transaction_run in ggml-sycl.cpp) selects whole triplets from the published plan, mints pinned
host storage owner-first, and under the inventory mutex has the cache lend each triplet
(unified_cache::loan_expert_triplet), then flips the plan copy and publishes it once. It runs at a graph boundary. What this gate pins (the retire, the boundary and the ready event are lead ruling c-tdra):
  - the loan code never waits and never asks an event's status (rulings A/D, c-3bde and c-fjx1): no .wait(, no
    wait_and_throw(, no get_info<, no event_complete(, and not finalize_retired_entries_locked(, which asks;
  - the device free is gated on the copy event, with no host_task: each D2H copy is submitted on the DMA queue reading
    the entry's own allocation owner, then each device entry's last write becomes its copy (has_write_event stamped from
    the copy, and the copy ran behind the fill), so release_entry_allocation_locked queues the free behind the copy;
    then the transaction takes a lease holding the entry in place, retires it exactly, registers the host copy behind
    its copy event, and only then drops that lease;
  - the retire is exact: it withdraws only the device entry's key, and its mirror only while it names that entry, so
    the loan never calls the id-wide retire_expert_entry_exact;
  - a host registration the checks could not foresee failing puts every device entry back (un-retired, recounted,
    its mirror restored) after withdrawing the host copies already published, and what the registration checks
    itself (the load effect, the plan coherence, a busy host entry) is checked before anything is copied;
  - g_sycl_graph_compute_mutex is never named in the loan code;
  - lock order: the host storage is minted before g_tensor_inventory_mutex is taken, and the cache is asked to lend
    under it; the cache takes direct_stage_mutex_ and rw_mutex_ together, with std::lock;
  - the loan runs at the graph boundary, never inside a graph: an in-graph caller records a request with a lock-free
    compare-and-swap; ggml_backend_sycl_synchronize() runs the pending loan after it clears the graph's compute flag
    and before the deferred-free pass; the transaction refuses "graph active"; and mul_mat_id names no loan entry;
  - the CPU expert consumer orders on the host copy's ready event: the planned prompt route carries it into the dispatch
    entry, and the planner CPU dispatch adds it to the copy events it waits on before building the tasks;
  - the exact-owner check (c-sepd) and the layout check (c-n4gw) refuse before anything is copied;
  - the registration is the typed register_host_expert_owned, carrying the minted owner, never the untyped overload;
  - a probe returns before any lease, allocation, lock or loan, and a loan publishes once, then bumps the exec-graph
    epoch;
  - the WARN format strings;
  - the device test's registration (name, labels, selector, strict leases, serial).

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


def strip_cmake_comments(src: str) -> str:
    """CMake comments run from # to the end of the line; none of the lines this gate reads quotes a #."""
    return re.sub(r"#[^\n]*", "", src)


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
SYCL = read("ggml/src/ggml-sycl/ggml-sycl.cpp")
SYCL_CMAKE = read("ggml/src/ggml-sycl/CMakeLists.txt")


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


LOAN_SIG = "bool unified_cache::loan_expert_triplet("
EXACT_SIG = "expert_retire_status unified_cache::retire_expert_entry_at_locked("
RESTORE_SIG = "void unified_cache::restore_expert_entry_at_locked("
TXN_SIG = "static ggml_sycl_txn_result ggml_sycl_expert_loan_transaction_run("
RESOLVE_SIG = "static bool ggml_sycl_expert_loan_resolve_registered("
WRAPPER_SIG = "static ggml_sycl_txn_result ggml_sycl_run_expert_loan_transaction("
REQUEST_SIG = "static void ggml_sycl_request_expert_loan("
PENDING_DEVICE_SIG = "static bool ggml_sycl_run_pending_expert_loan_on_device("
PENDING_SIG = "static void ggml_sycl_run_pending_expert_loan("
SYNC_SIG = "static void ggml_backend_sycl_synchronize(ggml_backend_t backend)"
MMID_SIG = "static void ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx, ggml_tensor * dst)"
DISPATCH_SIG = "auto dispatch_cpu_entries_now = [&](const std::vector<expert_dispatch_entry> & entries)"


def loan_bodies(cache: str, sycl: str) -> list[str]:
    """The loan code, cache side and backend side. A missing body is an empty string, which every claim that reads it
    treats as a failure."""
    c = norm(cache)
    s = norm(sycl)
    return [body(c, LOAN_SIG), body(c, EXACT_SIG), body(c, RESTORE_SIG), body(s, TXN_SIG), body(s, RESOLVE_SIG),
            body(s, WRAPPER_SIG), body(s, REQUEST_SIG), body(s, PENDING_DEVICE_SIG), body(s, PENDING_SIG)]


# --- waits and status queries ---------------------------------------------------------------------------------------

FORBIDDEN_WAITS = (".wait(", "wait_and_throw(", "get_info<", "event_complete(", "finalize_retired_entries_locked(")


def claim_loan_never_waits(cache: str, sycl: str) -> bool:
    bodies = loan_bodies(cache, sycl)
    return all(bodies) and not any(token in b for b in bodies for token in FORBIDDEN_WAITS)


# --- the device free is gated on the copy event ---------------------------------------------------------------------

SRC_IS_OWNER = "src_owner[i] = e.direct_alloc_owner;"
COPY = "out.copy_events[i] = mem_copy_async(members[i].host_owner, src_owner[i], bytes[i], dq, deps[i]);"
DMA_QUEUE = "sycl::queue & dq = get_dma_queue();"
DEPS = ("if (e.has_write_event) { deps[i].push_back(e.last_write_event); }",
        "if (e.has_ready_event) { deps[i].push_back(e.ready_event); }")
STAMP_EVENT = "device_entry[i]->last_write_event = out.copy_events[i];"
STAMP_FLAG = "device_entry[i]->has_write_event = true;"
HOLD = 'held[i] = make_direct_entry_handle(device_key, cache_device, *device_entry[i], "expert_loan/held");'
DEVICE_RETIRE = "out.retire_status[i] = retire_expert_entry_at_locked(device_key, &released_mirrors[i]);"
REGISTER = ("register_host_expert_owned(m.host_keys[k], host_ptr, bytes[i], GGML_LAYOUT_AOS, "
            "k == 0 ? &out.host_handles[i] : nullptr, m.host_owner, &out.copy_events[i])")
DROP_HELD = "for (std::shared_ptr<mem_handle> & lease : held) { lease.reset(); }"


def claim_device_free_gated_on_the_copy(cache: str, sycl: str) -> bool:
    b = body(norm(cache), LOAN_SIG)
    bodies = loan_bodies(cache, sycl)
    once = (SRC_IS_OWNER, COPY, STAMP_EVENT, STAMP_FLAG, HOLD, DEVICE_RETIRE, REGISTER)
    return (all(bodies) and not any("host_task" in x for x in bodies) and all(b.count(x) == 1 for x in once)
            and all(d in b for d in DEPS) and b.count(DROP_HELD) == 2
            and ordered(b, SRC_IS_OWNER, DMA_QUEUE, COPY, STAMP_EVENT, STAMP_FLAG, HOLD, DEVICE_RETIRE, REGISTER)
            and b.find(DROP_HELD) > b.find(REGISTER))


# --- the retire is exact --------------------------------------------------------------------------------------------

MIRROR_MATCH = ("if (direct != direct_expert_entries_.end() && direct->second.ptr == it->second.device_ptr && "
                "direct->second.location == it->second.location && direct->second.layout == it->second.layout) {")
MIRROR_MOVE = "*released_mirror = std::move(direct->second);"
EXACT_FIND = "auto it = entries_.find(exact);"
EXACT_REMAP = "remap_or_erase_id_mapping_locked(exact.id, exact);"


def claim_retire_is_exact(cache: str) -> bool:
    c = norm(cache)
    exact = body(c, EXACT_SIG)
    loan = body(c, LOAN_SIG)
    return (bool(exact) and bool(loan) and all(x in exact for x in (EXACT_FIND, MIRROR_MATCH, MIRROR_MOVE, EXACT_REMAP))
            and "for (auto & pair : entries_)" not in exact and "retire_expert_entry_exact(" not in loan)


# --- a failed host registration puts the device entries back --------------------------------------------------------

UNRETIRE = "it->second.retired = false;"
COUNT_BACK = "retired_pending_count_.fetch_sub(1, std::memory_order_acq_rel);"
MAPPING_BACK = "id_to_key_.emplace(exact.id, exact);"
MIRROR_BACK = "direct_expert_entries_.emplace(exact.id, std::move(mirror));"
HOST_WITHDRAW = ("(void) retire_expert_entry_at_locked({ cache_entry_type::MOE_EXPERT, members[i].host_keys[k], -1, -1 }, "
                 "&released[r++]);")
RESTORE_CALL = ("restore_expert_entry_at_locked(make_direct_stage_key(cache_entry_type::MOE_EXPERT, members[i].key, "
                "GGML_LAYOUT_AOS), released_mirrors[i]);")
REGISTRATION_REFUSED = 'out.reason = "host registration failed";'


def claim_failed_registration_restores(cache: str) -> bool:
    c = norm(cache)
    restore = body(c, RESTORE_SIG)
    b = body(c, LOAN_SIG)
    return (bool(restore) and all(x in restore for x in (UNRETIRE, COUNT_BACK, MAPPING_BACK, MIRROR_BACK))
            and b.count(RESTORE_CALL) == 1 and b.count(HOST_WITHDRAW) == 1
            and ordered(b, REGISTER, HOST_WITHDRAW, RESTORE_CALL, DROP_HELD, REGISTRATION_REFUSED))


# --- what the registration checks is checked before anything is copied ----------------------------------------------

LOAD_EFFECT = 'if (acquire_bound_load_effect().failed()) { out.reason = "load effect unavailable"; return false; }'
COHERENCE = ("if (cache_placement_coherence(this).coherence == placement_cache_coherence::TRANSIENT_MISMATCH) "
             '{ out.reason = "plan not coherent"; return false; }')
HOST_BUSY = ("if (host != entries_.end() && (host->second.retired || host->second.in_use_count.load() != 0)) "
             '{ refusal = "host entry busy"; break; }')
REFUSE_EARLY = "if (refusal) { out.reason = refusal; return false; }"


def claim_registration_checked_before_the_copies(cache: str) -> bool:
    b = body(norm(cache), LOAN_SIG)
    return (all(b.count(x) == 1 for x in (LOAD_EFFECT, COHERENCE, HOST_BUSY, REFUSE_EARLY))
            and ordered(b, LOAD_EFFECT, COHERENCE, HOST_BUSY, REFUSE_EARLY, COPY))


# --- the graph compute mutex ----------------------------------------------------------------------------------------

def claim_graph_mutex_untouched(cache: str, sycl: str) -> bool:
    bodies = loan_bodies(cache, sycl)
    return all(bodies) and not any("g_sycl_graph_compute_mutex" in b for b in bodies)


# --- lock order -----------------------------------------------------------------------------------------------------

HOST_ALLOC = "ggml_sycl::mem_handle host = ggml_sycl_host_expert_alloc(device, e->dst_size);"
INVENTORY_LOCK = "std::unique_lock<std::mutex> lock(g_tensor_inventory_mutex);"
LEND = "if (!cache->loan_expert_triplet(triplet.members, record)) {"
CACHE_LOCKS = ("std::unique_lock<std::shared_mutex> direct_lock(direct_stage_mutex_, std::defer_lock); "
               "std::unique_lock<std::shared_mutex> cache_lock(rw_mutex_, std::defer_lock); "
               "std::lock(direct_lock, cache_lock);")


def claim_lock_order(cache: str, sycl: str) -> bool:
    txn = body(norm(sycl), TXN_SIG)
    loan = body(norm(cache), LOAN_SIG)
    return (txn.count(INVENTORY_LOCK) == 1 and txn.count(HOST_ALLOC) == 1 and ordered(txn, HOST_ALLOC, INVENTORY_LOCK, LEND)
            and "g_tensor_inventory_mutex" not in loan and loan.count(CACHE_LOCKS) >= 3
            and "std::unique_lock<std::shared_mutex> cache_lock(rw_mutex_);" not in loan)


# --- the loan runs at the graph boundary ----------------------------------------------------------------------------

REQUEST_SLOTS = "static std::atomic<size_t> g_sycl_expert_loan_requests[GGML_SYCL_MAX_DEVICES];"
REQUEST_CAS = ("request.compare_exchange_weak(seen, bytes_wanted, std::memory_order_release, "
               "std::memory_order_relaxed)")
SET_INACTIVE = "ggml_sycl::unified_cache_set_graph_compute_active(false);"
RUN_PENDING = "ggml_sycl_run_pending_expert_loan(backend);"
DEFERRED_FREES = "cache->process_deferred_frees_public();"
GRAPH_ACTIVE = ('if (ggml_sycl::unified_cache_is_graph_compute_active()) { '
                'return refuse(ggml_sycl_txn_result::BUSY, "graph active"); }')
PENDING_SKIP = ("if (request.load(std::memory_order_acquire) == 0 || ggml_sycl::unified_cache_is_graph_compute_active()) "
                "{ return false; }")
PENDING_TAKE = "const size_t bytes_wanted = request.exchange(0, std::memory_order_acq_rel);"
PENDING_RUN = "(void) ggml_sycl_expert_loan_transaction_run(device, bytes_wanted, false, resolver, out);"
MMID_FORBIDDEN = ("ggml_sycl_expert_loan_transaction_run(", "ggml_sycl_run_expert_loan_transaction(",
                  "ggml_sycl_run_pending_expert_loan", "loan_expert_triplet(")


def claim_request_is_lock_free(sycl: str) -> bool:
    s = norm(sycl)
    b = body(s, REQUEST_SIG)
    return (s.count(REQUEST_SLOTS) == 1 and b.count(REQUEST_CAS) == 1 and "mutex" not in b and "lock(" not in b
            and "ggml_sycl_expert_loan_transaction_run(" not in b)


def claim_loan_runs_at_the_graph_boundary(sycl: str) -> bool:
    s = norm(sycl)
    sync = body(s, SYNC_SIG)
    txn = body(s, TXN_SIG)
    pending = body(s, PENDING_DEVICE_SIG)
    mmid = body(s, MMID_SIG)
    return (bool(mmid) and s.count(RUN_PENDING) == 1 and ordered(sync, SET_INACTIVE, RUN_PENDING, DEFERRED_FREES)
            and txn.count(GRAPH_ACTIVE) == 1 and ordered(txn, GRAPH_ACTIVE, "registry.acquire_live_update(", INVENTORY_LOCK)
            and ordered(pending, PENDING_SKIP, PENDING_TAKE, PENDING_RUN)
            and not any(x in mmid for x in MMID_FORBIDDEN))


# --- the CPU expert consumer orders on the host copy's ready event --------------------------------------------------

ROUTE_READY = ("expert_dispatch_entry host_entry = ggml_sycl_make_expert_dispatch_entry(i12, id, i02, "
               "ggml_sycl::mem_handle::HOST_DEVICE, route.actual_layout, std::move(route.lease), false); "
               "host_entry.has_ready_event = route.has_ready_event; host_entry.ready_event = route.ready_event; "
               "cpu_entries.push_back(std::move(host_entry));")
DISPATCH_READY = "if (entry.has_ready_event) { copy_events.push_back(entry.ready_event); }"
DISPATCH_WAIT = "sycl::event::wait(copy_events);"


def claim_host_consumer_orders_on_the_copy(sycl: str) -> bool:
    s = norm(sycl)
    mmid = body(s, MMID_SIG)
    dispatch = body(mmid, DISPATCH_SIG)
    return (mmid.count(ROUTE_READY) == 1 and dispatch.count(DISPATCH_READY) == 1
            and ordered(dispatch, DISPATCH_READY, DISPATCH_WAIT, "cpu_expert_task task;"))


# --- the exact-owner and layout checks ------------------------------------------------------------------------------

LAYOUT_CHECK = 'if (e.layout != GGML_LAYOUT_AOS) { refusal = "layout"; break; }'
OWNER_CHECK = ("if (e.storage_owner || e.pool_allocated || !e.direct_alloc_owner.owner_metadata(&md) || "
               "md.ptr != e.device_ptr || md.size != e.size || md.cache_backing || md.tier != alloc_tier::DEVICE_VRAM) "
               '{ refusal = "shared storage"; break; }')
USE_CHECK = 'if (e.direct_alloc_owner.owner_use_count() != 1) { refusal = "storage still referenced"; break; }'


def claim_owner_checks_refuse_before_copying(cache: str) -> bool:
    b = body(norm(cache), LOAN_SIG)
    return all(b.count(x) == 1 for x in (LAYOUT_CHECK, OWNER_CHECK, USE_CHECK, REFUSE_EARLY)) and ordered(
        b, LAYOUT_CHECK, OWNER_CHECK, USE_CHECK, SRC_IS_OWNER, REFUSE_EARLY, COPY)


# --- the typed registration -----------------------------------------------------------------------------------------

def claim_registration_is_typed(cache: str) -> bool:
    b = body(norm(cache), LOAN_SIG)
    return b.count(REGISTER) == 1 and "register_host_expert(" not in b


# --- probe, publish once, epoch -------------------------------------------------------------------------------------

PROBE = ('if (probe_mode) { result.accepted = n_selected != 0; result.bytes_freed = selected; '
         'result.n_triplets = n_selected; return n_selected != 0 ? finish(ggml_sycl_txn_result::ACCEPTED, "OK") : '
         'finish(ggml_sycl_txn_result::REFUSED, "nothing lendable"); }')
LEASE = "registry.acquire_live_update(current_token)"
FORGET = "(void) extra->forget_moe_storage_handle_on_device(k.expert_id, GGML_LAYOUT_AOS, device);"
REMEMBER = ("(void) extra->remember_moe_storage_handle(k.expert_id, GGML_LAYOUT_AOS, std::move(record.host_handles[r]), "
            "&record.copy_events[r], 0, logical);")
FLIP = "(void) next_plan.update_expert_placement(k.layer_id, k.expert_id, k.role, false, -1);"
PUBLISH = "ggml_sycl_publish_prepared_plan_locked(prepared_publication);"
EPOCH = "ggml_sycl_optional_layouts_retired();"
UNLOCK = "} lock.unlock(); if (!swapped) {"


def claim_probe_changes_nothing_and_a_loan_publishes_once(sycl: str) -> bool:
    b = body(norm(sycl), TXN_SIG)
    return (b.count(PROBE) == 1 and b.count(PUBLISH) == 1 and b.count(FLIP) == 1
            and ordered(b, PROBE, LEASE, HOST_ALLOC, INVENTORY_LOCK, LEND, FORGET, REMEMBER, FLIP, PUBLISH, EPOCH, UNLOCK))


# --- WARN formats ---------------------------------------------------------------------------------------------------

LOAN_WARN = ('GGML_LOG_WARN("[MOE-EXEC] loan device %d: %u triplets, %zu MiB lent to the ring (%s)\\n", device, lent, '
             "freed / (1024 * 1024), reason);")
REFUSAL_WARN = 'GGML_LOG_WARN("[MOE-EXEC] loan device %d refused: %s\\n", device, '


def claim_warn_formats(sycl: str) -> bool:
    b = body(norm(sycl), TXN_SIG)
    return b.count(LOAN_WARN) == 1 and b.count(REFUSAL_WARN) == 2 and ordered(b, UNLOCK, LOAN_WARN)


# --- the device test's registration ---------------------------------------------------------------------------------

REG_ADD = "add_test(NAME sycl-moe-expert-loan COMMAND test-moe-expert-loan)"
REG_PROPS = "set_tests_properties(sycl-moe-expert-loan PROPERTIES"
REG_NEEDLES = ('LABELS "sycl;cache;moe"',
               'ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1;GGML_SYCL_STRICT_LEASES=1;',
               "RUN_SERIAL TRUE",
               "SKIP_RETURN_CODE 77")


def claim_device_test_registration(cmake: str) -> bool:
    t = re.sub(r"\s+", " ", strip_cmake_comments(cmake))
    at = t.find(REG_PROPS)
    if t.count(REG_ADD) != 1 or at < 0:
        return False
    props = t[at:t.find(")", at)]
    return all(n in props for n in REG_NEEDLES)


# --- positive tests -------------------------------------------------------------------------------------------------

def test_loan_never_waits():
    assert claim_loan_never_waits(CACHE, SYCL)


def test_device_free_gated_on_the_copy():
    assert claim_device_free_gated_on_the_copy(CACHE, SYCL)


def test_retire_is_exact():
    assert claim_retire_is_exact(CACHE)


def test_failed_registration_restores():
    assert claim_failed_registration_restores(CACHE)


def test_registration_checked_before_the_copies():
    assert claim_registration_checked_before_the_copies(CACHE)


def test_graph_mutex_untouched():
    assert claim_graph_mutex_untouched(CACHE, SYCL)


def test_lock_order():
    assert claim_lock_order(CACHE, SYCL)


def test_request_is_lock_free():
    assert claim_request_is_lock_free(SYCL)


def test_loan_runs_at_the_graph_boundary():
    assert claim_loan_runs_at_the_graph_boundary(SYCL)


def test_host_consumer_orders_on_the_copy():
    assert claim_host_consumer_orders_on_the_copy(SYCL)


def test_owner_checks_refuse_before_copying():
    assert claim_owner_checks_refuse_before_copying(CACHE)


def test_registration_is_typed():
    assert claim_registration_is_typed(CACHE)


def test_probe_changes_nothing_and_a_loan_publishes_once():
    assert claim_probe_changes_nothing_and_a_loan_publishes_once(SYCL)


def test_warn_formats():
    assert claim_warn_formats(SYCL)


def test_device_test_registration():
    assert claim_device_test_registration(SYCL_CMAKE)


def test_comment_cannot_satisfy_a_claim():
    """A needle that survives only in a comment does not count: the stripped text loses it."""
    commented = norm("// " + COPY + "\n")
    assert COPY not in commented


# --- mutants --------------------------------------------------------------------------------------------------------

def _once(text: str, old: str, new: str) -> str:
    """`text` with the one occurrence of `old` (matched on normalized text) replaced by `new`."""
    n = norm(text)
    assert n.count(old) == 1, f"mutant anchor must match exactly once: {old!r} x{n.count(old)}"
    return n.replace(old, new, 1)


def _once_in(text: str, signature: str, old: str, new: str) -> str:
    """`text`, normalized, with the one occurrence of `old` inside the body of `signature` replaced by `new`: for an
    anchor that also occurs elsewhere in the file."""
    n = norm(text)
    b = body(n, signature)
    assert b and n.count(b) == 1, f"body of {signature!r} must be present and unique"
    assert b.count(old) == 1, f"mutant anchor must match exactly once in the body: {old!r} x{b.count(old)}"
    return n.replace(b, b.replace(old, new, 1), 1)


def test_mutant_loan_waits_on_its_copy_fails():
    assert not claim_loan_never_waits(_once(CACHE, COPY, COPY + " out.copy_events[i].wait();"), SYCL)


def test_mutant_loan_asks_an_event_status_fails():
    assert not claim_loan_never_waits(
        _once(CACHE, COPY, COPY + " (void) out.copy_events[i].get_info<sycl::info::event::command_execution_status>();"),
        SYCL)


def test_mutant_loan_finalizes_inline_fails():
    """The id-wide retire's immediate finalize asks the fill event's status: the loan leaves it to the deferred pass."""
    assert not claim_loan_never_waits(
        _once(CACHE, "out.accepted = true; out.reason = \"OK\";",
              "(void) finalize_retired_entries_locked(); out.accepted = true; out.reason = \"OK\";"), SYCL)


def test_mutant_transaction_waits_fails():
    assert not claim_loan_never_waits(CACHE, _once(SYCL, LEND, "cache->get_queue().wait_and_throw(); " + LEND))


def test_mutant_pending_runner_waits_fails():
    assert not claim_loan_never_waits(CACHE, _once(SYCL, PENDING_TAKE, "cache_queue_wait.wait(); " + PENDING_TAKE))


def test_mutant_retire_chained_through_a_host_task_fails():
    """c-tdra dropped the host_task: a submit can block on its in-flight same-queue dependency (memory fact 8)."""
    assert not claim_device_free_gated_on_the_copy(
        _once(CACHE, COPY, COPY + " dq.submit([&](sycl::handler & h) { h.depends_on(out.copy_events[i]); "
                                  "h.host_task([] {}); });"), SYCL)


def test_mutant_device_entry_not_stamped_fails():
    """Without the stamp the deferred free waits only for the fill: the storage could go back mid-copy."""
    assert not claim_device_free_gated_on_the_copy(_once(CACHE, STAMP_EVENT, ""), SYCL)


def test_mutant_write_flag_not_set_fails():
    assert not claim_device_free_gated_on_the_copy(_once(CACHE, STAMP_FLAG, ""), SYCL)


def test_mutant_retire_before_copy_fails():
    """The device entry retired before its bytes are read out: the copy source could be reclaimed under it."""
    n = _once(CACHE, DEVICE_RETIRE, "")
    n = n.replace(DMA_QUEUE, DEVICE_RETIRE + " " + DMA_QUEUE, 1)
    assert not claim_device_free_gated_on_the_copy(n, SYCL)


def test_mutant_host_registered_before_the_retire_fails():
    """c-tdra: the host copy is registered once the device entry is withdrawn, never alongside it."""
    n = _once(CACHE, DEVICE_RETIRE, "")
    n = n.replace(REGISTRATION_REFUSED, REGISTRATION_REFUSED + " " + DEVICE_RETIRE, 1)
    assert not claim_device_free_gated_on_the_copy(n, SYCL)


def test_mutant_lease_dropped_before_the_registration_fails():
    """The held lease dropped early: a registration failure could no longer put the withdrawn entry back."""
    assert not claim_device_free_gated_on_the_copy(_once(CACHE, HOLD, HOLD + " " + DROP_HELD), SYCL)


def test_mutant_copy_source_not_the_owner_fails():
    """A raw-pointer copy handle retains nothing: the deferred free could release the storage mid-copy."""
    assert not claim_device_free_gated_on_the_copy(
        _once(CACHE, SRC_IS_OWNER,
              "src_owner[i] = make_copy_handle_for_raw_ptr(e.device_ptr, GGML_LAYOUT_AOS, cache_device, e.size);"),
        SYCL)


def test_mutant_copy_not_behind_the_fill_fails():
    assert not claim_device_free_gated_on_the_copy(_once(CACHE, DEPS[0], ""), SYCL)


def test_mutant_host_copy_published_ready_fails():
    """The host entry published READY while its copy is still in flight: a reader could see stale bytes."""
    assert not claim_device_free_gated_on_the_copy(
        _once(CACHE, REGISTER, REGISTER.replace(", m.host_owner, &out.copy_events[i])", ", m.host_owner)")), SYCL)


def test_mutant_loan_uses_the_id_wide_retire_fails():
    """retire_expert_entry_exact sweeps every AOS entry under the id, the host copy under the same id too."""
    assert not claim_retire_is_exact(
        _once(CACHE, DEVICE_RETIRE,
              "out.retire_status[i] = retire_expert_entry_exact(members[i].key, GGML_LAYOUT_AOS, \"loan\");"))


def test_mutant_exact_retire_drops_any_mirror_fails():
    """The mirror erased whatever it names: a host copy's mirror published under the same id would go."""
    assert not claim_retire_is_exact(
        _once(CACHE, MIRROR_MATCH, "if (direct != direct_expert_entries_.end()) {"))


def test_mutant_registration_failure_leaves_the_entry_retired_fails():
    """No rollback: the plan still names a device copy nobody can resolve."""
    assert not claim_failed_registration_restores(_once(CACHE, RESTORE_CALL, ""))


def test_mutant_restore_keeps_the_retirement_count_fails():
    """The count of retired entries left one too high: the deferred-free pass keeps looking for an entry that is live."""
    assert not claim_failed_registration_restores(_once_in(CACHE, RESTORE_SIG, COUNT_BACK, ""))


def test_mutant_registration_failure_keeps_the_host_copies_fails():
    assert not claim_failed_registration_restores(_once(CACHE, HOST_WITHDRAW, ""))


def test_mutant_load_effect_unchecked_fails():
    assert not claim_registration_checked_before_the_copies(_once(CACHE, LOAD_EFFECT, ""))


def test_mutant_coherence_unchecked_fails():
    assert not claim_registration_checked_before_the_copies(_once(CACHE, COHERENCE, ""))


def test_mutant_busy_host_entry_unchecked_fails():
    assert not claim_registration_checked_before_the_copies(_once(CACHE, HOST_BUSY, ""))


def test_mutant_transaction_takes_the_graph_mutex_fails():
    assert not claim_graph_mutex_untouched(
        CACHE, _once_in(SYCL, TXN_SIG, INVENTORY_LOCK,
                        "std::lock_guard<std::mutex> graph(g_sycl_graph_compute_mutex); " + INVENTORY_LOCK))


def test_mutant_pending_runner_takes_the_graph_mutex_fails():
    assert not claim_graph_mutex_untouched(
        CACHE, _once(SYCL, PENDING_TAKE, "std::lock_guard<std::mutex> graph(g_sycl_graph_compute_mutex); " + PENDING_TAKE))


def test_mutant_allocates_under_the_inventory_lock_fails():
    n = _once_in(SYCL, TXN_SIG, INVENTORY_LOCK, "")
    n = _once_in(n, TXN_SIG, HOST_ALLOC, INVENTORY_LOCK + " " + HOST_ALLOC)
    assert not claim_lock_order(CACHE, n)


def test_mutant_lends_without_the_inventory_lock_fails():
    assert not claim_lock_order(CACHE, _once_in(SYCL, TXN_SIG, INVENTORY_LOCK, ""))


def test_mutant_cache_takes_rw_mutex_alone_fails():
    """rw_mutex_ alone, not with direct_stage_mutex_ through std::lock: the order direct staging established breaks."""
    stamp_block = "{ " + CACHE_LOCKS + " unified_cache_entry * device_entry[3] = { nullptr, nullptr, nullptr };"
    n = _once(CACHE, stamp_block, "{ std::unique_lock<std::shared_mutex> cache_lock(rw_mutex_); "
                                  "unified_cache_entry * device_entry[3] = { nullptr, nullptr, nullptr };")
    assert not claim_lock_order(n, SYCL)


def test_mutant_request_takes_a_mutex_fails():
    """An in-graph caller may hold g_sycl_graph_compute_mutex (L2): a request that locks could invert L1 under it."""
    assert not claim_request_is_lock_free(
        _once(SYCL, REQUEST_CAS, "std::lock_guard<std::mutex> lock(g_tensor_inventory_mutex), " + REQUEST_CAS))


def test_mutant_loan_runs_before_the_graph_flag_clears_fails():
    n = _once_in(SYCL, SYNC_SIG, RUN_PENDING, "")
    n = _once_in(n, SYNC_SIG, SET_INACTIVE, RUN_PENDING + " " + SET_INACTIVE)
    assert not claim_loan_runs_at_the_graph_boundary(n)


def test_mutant_loan_runs_after_the_deferred_frees_fails():
    n = _once_in(SYCL, SYNC_SIG, RUN_PENDING, "")
    n = _once_in(n, SYNC_SIG, DEFERRED_FREES, DEFERRED_FREES + " " + RUN_PENDING)
    assert not claim_loan_runs_at_the_graph_boundary(n)


def test_mutant_transaction_runs_inside_a_graph_fails():
    assert not claim_loan_runs_at_the_graph_boundary(_once(SYCL, GRAPH_ACTIVE, ""))


def test_mutant_pending_runner_ignores_an_active_graph_fails():
    assert not claim_loan_runs_at_the_graph_boundary(
        _once(SYCL, PENDING_SKIP, "if (request.load(std::memory_order_acquire) == 0) { return false; }"))


def test_mutant_mul_mat_id_runs_the_loan_fails():
    """Plan Task 11b's call from mul_mat_id's first prompt op is superseded: it would take L1 under L2."""
    anchor = "auto dispatch_cpu_entries_now = [&]"
    assert not claim_loan_runs_at_the_graph_boundary(
        _once_in(SYCL, MMID_SIG, anchor,
                 "(void) ggml_sycl_run_expert_loan_transaction(nullptr, 1, false, nullptr); " + anchor))


def test_mutant_route_drops_the_ready_event_fails():
    """The planned prompt host route without its ready event: the CPU job could read the host copy before it lands."""
    assert not claim_host_consumer_orders_on_the_copy(
        _once(SYCL, ROUTE_READY, ROUTE_READY.replace(
            "host_entry.has_ready_event = route.has_ready_event; host_entry.ready_event = route.ready_event; ", "")))


def test_mutant_dispatch_ignores_the_ready_event_fails():
    assert not claim_host_consumer_orders_on_the_copy(_once(SYCL, DISPATCH_READY, ""))


def test_mutant_bulk_view_lent_fails():
    """c-sepd: the storage_owner term names a bulk view's refusal by the field a bulk entry carries.

    Not a behavioural RED: a bulk entry carries no direct_alloc_owner (unified-cache.cpp, the bulk staging loop that
    sets entry.storage_owner), so without this term owner_metadata() still fails and the device case still reads
    "shared storage". Only this mutant pins the term.
    """
    assert not claim_owner_checks_refuse_before_copying(
        _once(CACHE, OWNER_CHECK, OWNER_CHECK.replace("e.storage_owner || ", "")))


def test_mutant_wider_owner_lent_fails():
    """c-sepd: a one-sided size term passes an owner wider than the entry that starts at its pointer (expert 0 of a
    bulk allocation); the use count would then refuse it as "storage still referenced", the wrong reason."""
    assert not claim_owner_checks_refuse_before_copying(
        _once(CACHE, OWNER_CHECK, OWNER_CHECK.replace("md.size != e.size", "md.size < e.size")))


def test_mutant_shared_owner_lent_fails():
    assert not claim_owner_checks_refuse_before_copying(_once(CACHE, USE_CHECK, ""))


def test_mutant_soa_lent_as_aos_fails():
    """c-n4gw: an SOA entry copied as AOS hands the host garbage."""
    assert not claim_owner_checks_refuse_before_copying(_once(CACHE, LAYOUT_CHECK, ""))


def test_mutant_refusal_falls_through_to_the_copy_fails():
    assert not claim_owner_checks_refuse_before_copying(_once(CACHE, REFUSE_EARLY, ""))


def test_mutant_untyped_registration_fails():
    """The untyped overload drops the owner: the lent entry's leases would read as ownerless."""
    assert not claim_registration_is_typed(
        _once(CACHE, REGISTER, "register_host_expert(m.host_keys[k], host_ptr, bytes[i], GGML_LAYOUT_AOS, "
                               "k == 0 ? &out.host_handles[i] : nullptr, {}, mem_handle{})"))


def test_mutant_probe_falls_through_fails():
    assert not claim_probe_changes_nothing_and_a_loan_publishes_once(
        _once(SYCL, PROBE, PROBE.replace(' return n_selected != 0 ? finish(ggml_sycl_txn_result::ACCEPTED, "OK") : '
                                         'finish(ggml_sycl_txn_result::REFUSED, "nothing lendable");', "")))


def test_mutant_publish_per_triplet_fails():
    assert not claim_probe_changes_nothing_and_a_loan_publishes_once(_once(SYCL, FLIP, FLIP + " " + PUBLISH))


def test_mutant_device_lease_kept_fails():
    """The extra keeps its lease on the withdrawn device copy: the retired entry is never reclaimed."""
    assert not claim_probe_changes_nothing_and_a_loan_publishes_once(_once(SYCL, FORGET, ""))


def test_mutant_no_epoch_bump_fails():
    """A recorded exec graph keeps replaying a withdrawn copy's baked pointer."""
    assert not claim_probe_changes_nothing_and_a_loan_publishes_once(_once(SYCL, PUBLISH + " " + EPOCH, PUBLISH))


def test_mutant_loan_warn_reworded_fails():
    assert not claim_warn_formats(_once(SYCL, LOAN_WARN, LOAN_WARN.replace("lent to the ring", "lent")))


def test_mutant_refusal_warn_dropped_fails():
    assert not claim_warn_formats(
        _once(SYCL, "if (stop) { " + REFUSAL_WARN + "stop); }", ""))


def test_mutant_registration_without_strict_leases_fails():
    assert not claim_device_test_registration(SYCL_CMAKE.replace(";GGML_SYCL_STRICT_LEASES=1", "", 1))


def test_mutant_registration_not_serial_fails():
    t = re.sub(r"\s+", " ", strip_cmake_comments(SYCL_CMAKE))
    at = t.find(REG_PROPS)
    props_end = t.find(")", at)
    assert "RUN_SERIAL TRUE" in t[at:props_end]
    assert not claim_device_test_registration(t[:at] + t[at:props_end].replace("RUN_SERIAL TRUE", "") + t[props_end:])


def test_mutant_registration_wrong_label_fails():
    assert not claim_device_test_registration(SYCL_CMAKE.replace('LABELS "sycl;cache;moe"', 'LABELS "sycl;moe"', 1))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
