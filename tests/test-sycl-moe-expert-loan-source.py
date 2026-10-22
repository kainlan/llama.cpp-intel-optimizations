"""Source contract for llama.cpp-mwmr and llama.cpp-sif1 (plan Tasks 11b and 12 of llama.cpp-36j0): the expert loan
transaction and the refill that undoes it.

Host-only: reads sources, builds nothing, loads no model and touches no device. The behaviour is the device test
sycl-moe-expert-loan (lead-run); this gate pins the properties a later edit could undo while that test still passes, or
that it cannot see at all.

A prompt phase lends the device's least popular resident expert triplets to the prefill ring. The transaction
(ggml_sycl_expert_txn_run in ggml-sycl.cpp, entered for a loan through ggml_sycl_expert_loan_transaction_run) selects
whole triplets from the published plan, mints pinned
host storage owner-first, and under the inventory mutex has the cache lend each triplet
(unified_cache::loan_expert_triplet), then flips the plan copy and publishes it once. It runs at a graph boundary.
What this gate pins (the retire, the boundary and the ready event are lead ruling c-tdra):
  - the loan code never waits and never asks an event's status (c-3bde and c-fjx1): no .wait(, no wait_and_throw(,
    no get_info<, no event_complete(, and not finalize_retired_entries_locked(, which asks;
  - the device free is gated on the copy event, with no host_task: each D2H copy is submitted on the DMA queue reading
    the entry's own allocation owner; the transaction then re-checks each device entry and mints the lease that holds
    it in place, all three before any entry changes, since minting allocates; then each device entry's last write
    becomes its copy (has_write_event stamped from the copy, and the copy ran behind the fill), so
    release_entry_allocation_locked queues the free behind the copy; then it retires each entry exactly, registers
    the host copy behind its copy event, and only then drops the leases;
  - the retire is exact: it withdraws only the device entry's key, and its mirror only while it names that entry, so
    the loan never calls the id-wide retire_expert_entry_exact;
  - a host registration the checks could not foresee failing puts every device entry back (un-retired, recounted,
    its mirror restored) after withdrawing the host copies already published, only while each is still the entry the
    loan registered, and what the registration checks itself (the load effect, the plan coherence, an existing host
    entry) is checked before anything is copied; the vector the rollback collects into is sized before any entry
    changes, and anything the registration throws (after the G-CATCH fallback rethrow) fails it like a refusal; the
    retire takes the mirror and the id mapping out as the maps' own nodes and the restore re-inserts them, never
    emplacing, after every host copy is withdrawn, so the rollback allocates no node (llama.cpp-mwmr);
  - an exception while lending (an allocation failure) does not abandon the triplets already lent: the publication's
    allocations come before the first triplet is lent, each extra's device lease is dropped before anything that
    allocates, a lent triplet is flipped in the plan whatever followed, the loop stops, and the exception is rethrown
    after the inventory lock is released, on each of the three exits;
  - an existing host entry is never replaced (c-mwmr-3): any entry under a host id, idle or not, refuses the triplet
    as "host entry exists", the registration is asked not to replace one and refuses when it would, and the refusal
    reason reaches the transaction's refusal WARN;
  - g_sycl_graph_compute_mutex is never named in the loan code;
  - lock order: the host storage is minted before g_tensor_inventory_mutex is taken, and the cache is asked to lend
    under it; the cache takes direct_stage_mutex_ and rw_mutex_ together, with std::lock;
  - the loan runs at the graph boundary, never inside a graph: an in-graph caller records a request with a lock-free
    compare-and-swap; ggml_backend_sycl_synchronize() runs the pending loan after it clears the graph's compute flag
    and before the deferred-free pass; the transaction refuses "graph active"; and mul_mat_id names no loan entry;
  - a loan is refused "host zones not configured", REFUSED and so never re-posted, before anything is selected,
    leased, allocated or lent, when its device's cache has no host zones (lead ruling, llama.cpp-sif1): an unzoned
    pinned copy sends every later copy from it down the staged branch. The transaction asks the same query the pinned
    pool's allocation branches on, so a loan that passes the check mints only host-zone slices; a probe gets the same
    answer, and the refill direction does not ask;
  - a loan refused BUSY (a transient refusal) re-posts its request for a later boundary, and a REFUSED one does not;
    the runner, not the transaction, WARNs a re-posted BUSY refusal: the first in a row and then one in every 64,
    with the count, and any other outcome resets the count;
  - the CPU expert consumer orders on the host copy's ready event: the planned prompt route carries it into the dispatch
    entry, and the planner CPU dispatch adds it to the copy events it waits on before building the tasks, skipping one
    equal to the last it pushed; the batched CPU dispatch skips one equal to the last job input it pushed;
  - the exact-owner check (c-sepd) and the layout check (c-n4gw) refuse before anything is copied;
  - the registration is the typed register_host_expert_owned, carrying the minted owner, never the untyped overload;
  - a probe returns before any lease, allocation, lock or loan, and a loan publishes once, then bumps the exec-graph
    epoch;
  - the WARN format strings;
  - the device test's registration (name, labels, selector, strict leases, serial).

After the prompt, the decode graphs re-admit the lent triplets (llama.cpp-sif1): the same body, run with the REFILL
direction, mints device storage owner-first from the WEIGHT zone, and under the inventory mutex has the cache re-admit
each triplet (unified_cache::refill_lent_triplet), then flips the plan copy back on the device and publishes it once.
What this gate pins for it:
  - one body runs both directions, and its lock, its publication and its epoch bump are the one copy both share;
  - the refill code never waits, under the same forbidden list as the loan;
  - the copy reads the host copy's own typed owner into the caller's device owner on the BCS queue (the copy engine),
    behind the host copy's writes, and the device entry is published IN_PROGRESS with the copy as its ready event and
    last write;
  - the cache checks everything that would make the publication wrong before copying, inserts the device entries and
    mints their leases before anything published changes (a throw there withdraws only what it inserted), and from
    the mirror swap on allocates nothing; the host copy is retired entry by entry, only while it still names the lent
    storage, never by the id-wide retire; the publication runs inside a planned materialization session, as the loan's
    registration does; and resolve_expert lets no retired entry under the exact key (the withdrawn host copy, until
    the deferred-free pass finalizes it) shadow the id mapping that already names the device entry;
  - lock order: the device storage is minted before g_tensor_inventory_mutex, the cache never names that mutex, and no
    refill code names g_sycl_graph_compute_mutex;
  - each extra forgets its host-copy record before remembering the device copy, the plan flips back, and the ledger
    loses the triplet;
  - an accepted loan disarms the refill cursor, a disarmed cursor selects nothing, and a loan between the selection
    and the lock cancels the refill; each device's ledger is reached only under g_tensor_inventory_mutex;
  - ggml_backend_sycl_synchronize() reaches the refill at every call, after the pending loan and the deferred-free
    pass; the graph that just ran arms the cursor (decode) or disarms it (prompt), only a decode graph re-admits, and
    mul_mat_id names no refill entry;
  - a context synchronizes more than once per graph, so the refill runs only when a decode graph_compute call has
    been counted on the context since its last refill: one decode graph re-admits at most
    GGML_SYCL_MOE_REFILL_PER_GRAPH triplets, and only graph_compute counts one;
  - GGML_SYCL_MOE_REFILL_PER_GRAPH is read once per process, defaults to 1, and no value turns the refill off;
  - the teardown line `[MOE-EXEC] refill device %d: %u/%u triplets re-admitted` prints once from counts taken as they
    print, ggml_backend_sycl_free() emits it right after the [MOE-EXEC] summary, a refusal is WARNed by the boundary
    runner alone (first in a row, then one in every 64), the body WARNs no refill line but a dropped triplet's, and
    the refill never touches the loan's busy streak;
  - each refusal the transaction treats as final names a refusal the cache really returns, and drops its triplet;
  - a lent triplet the preparation can never prepare (no tensor resolves a member, or the device has no WEIGHT arena,
    which is told apart from a full zone) is dropped too, not returned BUSY: under the inventory lock, after the
    publication is checked unchanged, and only while it is the ledger's back;
  - each dropped triplet is WARNed once, after the lock is released, on both exits that follow a drop, naming the
    device, the layer, the expert and the reason; a drop resets the runner's refusal count;
  - one call moves every prepared triplet the cursor still holds: the move loop stops only where a triplet did not
    move, so at more than one per call each moved triplet is counted, its bytes summed, and the next is checked
    against the ledger's new back.

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
COMMON = read("ggml/src/ggml-sycl/common.hpp")


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
TXN_SIG = "static ggml_sycl_txn_result ggml_sycl_expert_txn_run("
LOAN_TXN_SIG = "static ggml_sycl_txn_result ggml_sycl_expert_loan_transaction_run("
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
    return [body(c, LOAN_SIG), body(c, EXACT_SIG), body(c, RESTORE_SIG), body(s, TXN_SIG), body(s, LOAN_TXN_SIG),
            body(s, RESOLVE_SIG), body(s, WRAPPER_SIG), body(s, REQUEST_SIG), body(s, PENDING_DEVICE_SIG),
            body(s, PENDING_SIG)]


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
ENTRY_CHANGED = '{ refusal = "entry changed"; break; }'
DEVICE_RETIRE = "out.retire_status[i] = retire_expert_entry_at_locked(device_key, &device_nodes[i]);"
REGISTER = ("register_host_expert_owned(m.host_keys[k], host_ptr, bytes[i], GGML_LAYOUT_AOS, "
            "k == 0 ? &out.host_handles[i] : nullptr, m.host_owner, &out.copy_events[i], false)")
DROP_HELD = "for (std::shared_ptr<mem_handle> & lease : held) { lease.reset(); }"


def claim_device_free_gated_on_the_copy(cache: str, sycl: str) -> bool:
    b = body(norm(cache), LOAN_SIG)
    bodies = loan_bodies(cache, sycl)
    once = (SRC_IS_OWNER, COPY, STAMP_EVENT, STAMP_FLAG, HOLD, DEVICE_RETIRE, REGISTER)
    return (all(bodies) and not any("host_task" in x for x in bodies) and all(b.count(x) == 1 for x in once)
            and all(d in b for d in DEPS) and b.count(DROP_HELD) == 2 and b.count(ENTRY_CHANGED) == 1
            and ordered(b, SRC_IS_OWNER, DMA_QUEUE, COPY, ENTRY_CHANGED, HOLD, STAMP_EVENT, STAMP_FLAG, DEVICE_RETIRE,
                        REGISTER)
            and b.find(DROP_HELD) > b.find(REGISTER))


# --- the retire is exact --------------------------------------------------------------------------------------------

MIRROR_MATCH = ("if (direct != direct_expert_entries_.end() && direct->second.ptr == it->second.device_ptr && "
                "direct->second.location == it->second.location && direct->second.layout == it->second.layout) {")
MIRROR_MOVE = "released->mirror = direct_expert_entries_.extract(direct);"
EXACT_FIND = "auto it = entries_.find(exact);"
EXACT_REMAP = "remap_or_extract_id_mapping_locked(exact.id, exact, &released->mapping);"


def claim_retire_is_exact(cache: str) -> bool:
    c = norm(cache)
    exact = body(c, EXACT_SIG)
    loan = body(c, LOAN_SIG)
    return (bool(exact) and bool(loan) and all(x in exact for x in (EXACT_FIND, MIRROR_MATCH, MIRROR_MOVE, EXACT_REMAP))
            and "for (auto & pair : entries_)" not in exact and "retire_expert_entry_exact(" not in loan)


# --- a failed host registration puts the device entries back --------------------------------------------------------

UNRETIRE = "it->second.retired = false;"
COUNT_BACK = "retired_pending_count_.fetch_sub(1, std::memory_order_acq_rel);"
MAPPING_BACK = ("if (!released.mapping.empty() && id_to_key_.find(exact.id) == id_to_key_.end()) "
                "{ id_to_key_.insert(std::move(released.mapping)); }")
MIRROR_BACK = ("if (!released.mirror.empty() && direct_expert_entries_.find(exact.id) == direct_expert_entries_.end()) "
               "{ direct_expert_entries_.insert(std::move(released.mirror)); }")
HOST_WITHDRAW = ("if (host != entries_.end() && host->second.direct_alloc_owner.owner_control_id() == host_control) "
                 "{ (void) retire_expert_entry_at_locked(host_key, &host_nodes[r++]); }")
RESTORE_CALL = ("restore_expert_entry_at_locked(make_direct_stage_key(cache_entry_type::MOE_EXPERT, members[i].key, "
                "GGML_LAYOUT_AOS), device_nodes[i]);")
REGISTRATION_REFUSED = 'out.reason = "host registration failed";'
RELEASED_SIZED = ("std::vector<expert_retire_nodes> host_nodes(members[0].host_keys.size() + "
                  "members[1].host_keys.size() + members[2].host_keys.size());")
REGISTER_CATCH = ("} catch (const ggml_sycl_fallback_error &) { throw; } catch (...) { published = false; } "
                  "if (!published) { failed = true; break; }")


def claim_failed_registration_restores(cache: str) -> bool:
    c = norm(cache)
    restore = body(c, RESTORE_SIG)
    b = body(c, LOAN_SIG)
    return (bool(restore) and all(x in restore for x in (UNRETIRE, COUNT_BACK, MAPPING_BACK, MIRROR_BACK))
            and b.count(RESTORE_CALL) == 1 and b.count(HOST_WITHDRAW) == 1
            and b.count("std::vector<expert_retire_nodes> host_nodes(") == 1 and b.count(REGISTER_CATCH) == 1
            and ordered(b, COPY, RELEASED_SIZED, HOLD, STAMP_EVENT)
            and ordered(b, REGISTER, REGISTER_CATCH, HOST_WITHDRAW, RESTORE_CALL, DROP_HELD, REGISTRATION_REFUSED))


REMAP_SIG = "void unified_cache::remap_or_extract_id_mapping_locked("
REMAP_EXTRACT = "} else if (erased) { *erased = id_to_key_.extract(mapped); } else { id_to_key_.erase(mapped); }"
TWO_PASS = HOST_WITHDRAW + " } } for (int i = 0; i < 3; ++i) { " + RESTORE_CALL
NODE_BUILDERS = ("emplace(", "id_to_key_[", "direct_expert_entries_[", "make_shared<", "push_back(")


def claim_rollback_reinserts_nodes(cache: str) -> bool:
    """The retire moves the mirror and the id mapping out as the maps' own nodes, and the restore re-inserts them
    rather than building new ones, so the rollback after step 4 allocates no node. Every host copy is withdrawn before
    any device entry goes back, so no map holds more entries at a restore than when its node came out, and a re-insert
    never grows the bucket array either."""
    c = norm(cache)
    exact = body(c, EXACT_SIG)
    restore = body(c, RESTORE_SIG)
    remap = body(c, REMAP_SIG)
    b = body(c, LOAN_SIG)
    return (bool(exact) and bool(restore) and bool(remap) and bool(b)
            and exact.count(MIRROR_MOVE) == 1 and exact.count(EXACT_REMAP) == 1 and remap.count(REMAP_EXTRACT) == 1
            and restore.count(MAPPING_BACK) == 1 and restore.count(MIRROR_BACK) == 1
            and not any(x in exact or x in restore for x in NODE_BUILDERS)
            and b.count(TWO_PASS) == 1)


def claim_registration_catches_everything(cache: str) -> bool:
    """Whatever the host registration throws after the device entries are withdrawn fails it like a refusal, so the
    rollback runs; only the op fallback is rethrown, as G-CATCH requires (nothing in a registration throws it)."""
    b = body(norm(cache), LOAN_SIG)
    return (b.count(REGISTER_CATCH) == 1 and "catch (const std::bad_alloc &)" not in b
            and ordered(b, DEVICE_RETIRE, "try { published = host_ptr && " + REGISTER, REGISTER_CATCH, TWO_PASS))


# --- what the registration checks is checked before anything is copied ----------------------------------------------

LOAD_EFFECT = 'if (acquire_bound_load_effect().failed()) { out.reason = "load effect unavailable"; return false; }'
COHERENCE = ("if (cache_placement_coherence(this).coherence == placement_cache_coherence::TRANSIENT_MISMATCH) "
             '{ out.reason = "plan not coherent"; return false; }')
HOST_EXISTS = ("if (entries_.find({ cache_entry_type::MOE_EXPERT, host_key, -1, -1 }) != entries_.end()) "
               '{ refusal = "host entry exists"; break; }')
REFUSE_EARLY = "if (refusal) { out.reason = refusal; return false; }"


def claim_registration_checked_before_the_copies(cache: str) -> bool:
    b = body(norm(cache), LOAN_SIG)
    return (all(b.count(x) == 1 for x in (LOAD_EFFECT, COHERENCE, HOST_EXISTS, REFUSE_EARLY))
            and ordered(b, LOAD_EFFECT, COHERENCE, HOST_EXISTS, REFUSE_EARLY, COPY))


# --- an existing host entry is never replaced (c-mwmr-3) ------------------------------------------------------------

OWNED_SIG = "bool unified_cache::register_host_expert_owned("
REG_BODY_SIG = "const mem_handle & typed_owner, const sycl::event * ready_event, bool replace_existing)"
NO_REPLACE = "if (old != entries_.end()) { if (!replace_existing) { return false; }"
REPLACE = "erase_entry_locked(old);"
FORWARD = "ready_event, replace_existing);"
STOP_FROM_RECORD = "stop = record.reason;"
REFUSE_STOP = "return refuse(stop_code, stop);"
# A loan never sets `transient`, so its refusals stay REFUSED; only a refill's transient refusal is BUSY.
STOP_CODE = ("const ggml_sycl_txn_result stop_code = transient ? ggml_sycl_txn_result::BUSY : "
             "ggml_sycl_txn_result::REFUSED;")


def claim_existing_host_entry_never_replaced(cache: str, sycl: str) -> bool:
    c = norm(cache)
    loan = body(c, LOAN_SIG)
    reg = body(c, REG_BODY_SIG)
    owned = body(c, OWNED_SIG)
    txn = body(norm(sycl), TXN_SIG)
    return (loan.count(HOST_EXISTS) == 1 and loan.count(REGISTER) == 1 and ordered(loan, HOST_EXISTS, COPY, REGISTER)
            and reg.count(NO_REPLACE) == 1 and ordered(reg, NO_REPLACE, REPLACE)
            and owned.count(FORWARD) == 2
            and txn.count(STOP_FROM_RECORD) == 1 and txn.count(REFUSE_STOP) == 1 and txn.count(STOP_CODE) == 1
            and ordered(txn, LEND, STOP_FROM_RECORD, STOP_CODE, REFUSE_STOP))


# --- the graph compute mutex ----------------------------------------------------------------------------------------

def claim_graph_mutex_untouched(cache: str, sycl: str) -> bool:
    bodies = loan_bodies(cache, sycl)
    return all(bodies) and not any("g_sycl_graph_compute_mutex" in b for b in bodies)


# --- lock order -----------------------------------------------------------------------------------------------------

HOST_ALLOC = "ggml_sycl::mem_handle host = ggml_sycl_host_expert_alloc(device, e->dst_size);"
INVENTORY_LOCK = "std::unique_lock<std::mutex> lock(g_tensor_inventory_mutex);"
LEND = "loaned = cache->loan_expert_triplet(triplet.members, record);"
CACHE_LOCKS = ("std::unique_lock<std::shared_mutex> direct_lock(direct_stage_mutex_, std::defer_lock); "
               "std::unique_lock<std::shared_mutex> cache_lock(rw_mutex_, std::defer_lock); "
               "std::lock(direct_lock, cache_lock);")


def claim_lock_order(cache: str, sycl: str) -> bool:
    txn = body(norm(sycl), TXN_SIG)
    loan = body(norm(cache), LOAN_SIG)
    return (txn.count(INVENTORY_LOCK) == 1 and txn.count(HOST_ALLOC) == 1
            and ordered(txn, HOST_ALLOC, INVENTORY_LOCK, LEND)
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
PENDING_SKIP = ("if (request.load(std::memory_order_acquire) == 0 || "
                "ggml_sycl::unified_cache_is_graph_compute_active()) { return false; }")
PENDING_TAKE = "const size_t bytes_wanted = request.exchange(0, std::memory_order_acq_rel);"
PENDING_RUN = ("const ggml_sycl_txn_result code = ggml_sycl_expert_loan_transaction_run(device, bytes_wanted, false, "
               "resolver, result, false);")
NOT_BUSY_RETURNS = ("if (code != ggml_sycl_txn_result::BUSY) { busy_streak.store(0, std::memory_order_relaxed); "
                    "return true; }")
REPOST = "ggml_sycl_request_expert_loan(device, bytes_wanted); return true; }"
MMID_FORBIDDEN = ("ggml_sycl_expert_loan_transaction_run(", "ggml_sycl_run_expert_loan_transaction(",
                  "ggml_sycl_run_pending_expert_loan", "loan_expert_triplet(")


def claim_request_is_lock_free(sycl: str) -> bool:
    s = norm(sycl)
    b = body(s, REQUEST_SIG)
    return (s.count(REQUEST_SLOTS) == 1 and b.count(REQUEST_CAS) == 1 and "mutex" not in b and "lock(" not in b
            and "ggml_sycl_expert_loan_transaction_run(" not in b)


def claim_busy_reposts_the_request(sycl: str) -> bool:
    """A transient refusal after the request was taken puts it back for a later boundary; a deterministic one does
    not, since it would only refuse again."""
    pending = body(norm(sycl), PENDING_DEVICE_SIG)
    return (pending.count(REPOST) == 1 and pending.count("ggml_sycl_request_expert_loan(") == 1
            and pending.count(NOT_BUSY_RETURNS) == 1 and ordered(pending, PENDING_TAKE, PENDING_RUN, NOT_BUSY_RETURNS,
                                                                  REPOST))


STREAK_SLOTS = "static std::atomic<uint32_t> g_sycl_expert_loan_busy_streak[GGML_SYCL_MAX_DEVICES];"
TXN_WARN_BUSY = "ggml_sycl_expert_loan_result * out, bool warn_busy = true) {"
REFUSE_GATE = "if (!probe_mode && !refill && (warn_busy || code != ggml_sycl_txn_result::BUSY)) { " + \
    'GGML_LOG_WARN("[MOE-EXEC] loan device %d refused: %s\\n", device, reason); }'
BUSY_EVERY = "constexpr uint32_t warn_every = 64;"
BUSY_COUNT = "const uint32_t n = busy_streak.fetch_add(1, std::memory_order_relaxed) + 1;"
BUSY_WARN = ('if ((n - 1) % warn_every == 0) { GGML_LOG_WARN("[MOE-EXEC] loan device %d refused: %s (%u in a row, '
             're-posted; logged once per %u)\\n", device, result->reason, n, warn_every); }')


def claim_busy_warn_is_rate_limited(sycl: str) -> bool:
    """A held live-update lease refuses BUSY at every boundary while it lasts. The runner asks the transaction not to
    WARN a BUSY refusal and WARNs it itself: the first in a row, then one in every 64, with the count; any other
    outcome resets the count."""
    s = norm(sycl)
    txn = body(s, TXN_SIG)
    pending = body(s, PENDING_DEVICE_SIG)
    return (s.count(STREAK_SLOTS) == 1 and s.count(TXN_WARN_BUSY) == 1 and txn.count(REFUSE_GATE) == 1
            and pending.count("GGML_LOG_WARN(") == 1
            and ordered(pending, PENDING_RUN, NOT_BUSY_RETURNS, BUSY_EVERY, BUSY_COUNT, BUSY_WARN, REPOST))


def claim_loan_runs_at_the_graph_boundary(sycl: str) -> bool:
    s = norm(sycl)
    sync = body(s, SYNC_SIG)
    txn = body(s, TXN_SIG)
    pending = body(s, PENDING_DEVICE_SIG)
    mmid = body(s, MMID_SIG)
    return (bool(mmid) and s.count(RUN_PENDING) == 1 and ordered(sync, SET_INACTIVE, RUN_PENDING, DEFERRED_FREES)
            and txn.count(GRAPH_ACTIVE) == 1
            and ordered(txn, GRAPH_ACTIVE, "registry.acquire_live_update(", INVENTORY_LOCK)
            and ordered(pending, PENDING_SKIP, PENDING_TAKE, PENDING_RUN)
            and not any(x in mmid for x in MMID_FORBIDDEN))


# --- a loan without host zones is refused (lead ruling, llama.cpp-sif1) ----------------------------------------------

INVALID_ARGS = 'return refuse(ggml_sycl_txn_result::REFUSED, "invalid arguments"); }'
ZONES_REFUSAL = ('if (!refill && !cache->host_zones_configured()) { '
                 'return refuse(ggml_sycl_txn_result::REFUSED, "host zones not configured"); }')
LOAN_SELECT = "ggml_sycl::moe_expert_loan_select("
UNIFIED_ALLOC_SIG = "bool unified_alloc(const alloc_request & req_in, alloc_handle * out)"
PINNED_ZONE_BRANCH = ("if (req.intent.constraints.use_pinned_pool) { if (!ucache->host_zones_configured()) { "
                      "ptr = ucache->host_pool_alloc(")


def claim_a_loan_without_host_zones_is_refused(cache: str, sycl: str) -> bool:
    """An unzoned pinned copy makes every later copy from it, the ring's and the refill's, a synchronous staged
    bounce, so a loan whose device has no host zones is refused by name, as REFUSED (the runner re-posts only BUSY),
    right after the argument check: before the selection, the probe's answer, the live-update lease, the host storage
    and the lend. It asks the one query the pinned pool's allocation branches on to pick a zone over the zoneless
    pool, so a loan that passes mints only zone slices."""
    txn = body(norm(sycl), TXN_SIG)
    alloc = body(norm(cache), UNIFIED_ALLOC_SIG)
    return (txn.count(ZONES_REFUSAL) == 1 and txn.count("host_zones_configured(") == 1
            and ordered(txn, INVALID_ARGS, ZONES_REFUSAL, GRAPH_ACTIVE, LOAN_SELECT, PROBE, LEASE, HOST_ALLOC,
                        INVENTORY_LOCK, LEND)
            and alloc.count(PINNED_ZONE_BRANCH) == 1)


# --- the CPU expert consumer orders on the host copy's ready event --------------------------------------------------

ROUTE_READY = ("expert_dispatch_entry host_entry = ggml_sycl_make_expert_dispatch_entry(i12, id, i02, "
               "ggml_sycl::mem_handle::HOST_DEVICE, route.actual_layout, std::move(route.lease), false); "
               "host_entry.has_ready_event = route.has_ready_event; host_entry.ready_event = route.ready_event; "
               "cpu_entries.push_back(std::move(host_entry));")
DISPATCH_READY = ("if (entry.has_ready_event && (!last_ready || *last_ready != entry.ready_event)) { "
                  "copy_events.push_back(entry.ready_event); last_ready = &entry.ready_event; }")
DISPATCH_WAIT = "sycl::event::wait(copy_events);"


def claim_host_consumer_orders_on_the_copy(sycl: str) -> bool:
    s = norm(sycl)
    mmid = body(s, MMID_SIG)
    dispatch = body(mmid, DISPATCH_SIG)
    return (mmid.count(ROUTE_READY) == 1 and dispatch.count(DISPATCH_READY) == 1
            and ordered(dispatch, DISPATCH_READY, DISPATCH_WAIT, "cpu_expert_task task;"))


BATCH_READY = ("for (const expert_dispatch_entry & entry : entries) { if (entry.has_ready_event && "
               "(job_inputs.events.empty() || job_inputs.events.back() != entry.ready_event)) { "
               "job_inputs.events.push_back(entry.ready_event); } }")


def claim_batched_dispatch_skips_a_repeated_ready_event(sycl: str) -> bool:
    """The batched CPU dispatch adds each entry's ready event to its job's inputs once per run of equal events, so a
    lent expert's many rows do not each add its copy's event again."""
    return body(norm(sycl), MMID_SIG).count(BATCH_READY) == 1


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
REMEMBER = ("(void) extra->remember_moe_storage_handle(k.expert_id, GGML_LAYOUT_AOS, "
            "std::move(record.host_handles[r]), &record.copy_events[r], 0, logical);")
FLIP = "(void) next_plan->update_expert_placement(k.layer_id, k.expert_id, k.role, refill, refill ? device : -1);"
PUBLISH = "ggml_sycl_publish_prepared_plan_locked(prepared_publication);"
EPOCH = "ggml_sycl_optional_layouts_retired();"
UNLOCK = "} lock.unlock(); warn_dropped(); if (!swapped) {"


def claim_probe_changes_nothing_and_a_loan_publishes_once(sycl: str) -> bool:
    b = body(norm(sycl), TXN_SIG)
    return (b.count(PROBE) == 1 and b.count(PUBLISH) == 1 and b.count(FLIP) == 1
            and ordered(b, PROBE, LEASE, HOST_ALLOC, INVENTORY_LOCK, LEND, FORGET, REMEMBER, FLIP, PUBLISH, EPOCH,
                        UNLOCK))


# --- an exception while lending publishes what was lent -------------------------------------------------------------

HOIST_SNAPSHOT = "auto next = std::make_shared<ggml_sycl::lifecycle_plan_snapshot>(*current);"
HOIST_PLAN = "auto next_plan = std::make_shared<ggml_sycl::placement_plan>(*current->plan);"
FORGET_LOOP_END = FORGET + " } for (int r = 0; r < 3; ++r) {"
CATCH_LENDING = ('} catch (const ggml_sycl_fallback_error &) { throw; } catch (...) { '
                 'escaped = std::current_exception(); '
                 'stop = refill ? "exception while refilling" : "exception while lending"; }')
FLIP_IF_LOANED = "if (loaned) { const ggml_sycl::expert_placement_key placed[3] = {"
BREAK_ESCAPED = "if (escaped) { break; }"
RETHROW = "if (escaped) { std::rethrow_exception(escaped); }"
RETHROW_EXITS = ("if (lent == 0) { lock.unlock(); warn_dropped(); " + RETHROW + " return refuse(stop_code, stop); }",
                 RETHROW + ' return finish(ggml_sycl_txn_result::REFUSED, "concurrent transaction won the CAS");',
                 RETHROW + " return finish(ggml_sycl_txn_result::ACCEPTED, reason);")


def claim_exception_publishes_what_was_lent(sycl: str) -> bool:
    b = body(norm(sycl), TXN_SIG)
    lend_at = b.find(LEND)
    return (lend_at >= 0 and "make_shared<" not in b[lend_at:] and b.count(CATCH_LENDING) == 1
            and b.count(RETHROW) == 3 and all(b.count(x) == 1 for x in RETHROW_EXITS)
            and ordered(b, INVENTORY_LOCK, HOIST_SNAPSHOT, HOIST_PLAN, LEND, FORGET_LOOP_END, REMEMBER, CATCH_LENDING,
                        FLIP_IF_LOANED, FLIP, BREAK_ESCAPED, PUBLISH, UNLOCK, LOAN_WARN, RETHROW_EXITS[2]))


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


# --- the refill (llama.cpp-sif1, plan Task 12) ----------------------------------------------------------------------

REFILL_SIG = "bool unified_cache::refill_lent_triplet("
SELECT_SIG = "static std::vector<ggml_sycl::expert_placement_key> ggml_sycl_expert_refill_select("
FINAL_SIG = "static bool ggml_sycl_expert_refill_refusal_is_final(const char * reason)"
DEVICE_ALLOC_SIG = "static ggml_sycl::mem_handle ggml_sycl_device_expert_alloc(int device, size_t size)"
REFILL_RUN_SIG = "static uint32_t ggml_sycl_expert_refill_transaction_run("
REFILL_API_SIG = "static uint32_t ggml_sycl_run_expert_refill_transaction("
ARM_SIG = "static void ggml_sycl_expert_refill_set_armed(int device, bool armed)"
BOUNDARY_DEVICE_SIG = "static uint32_t ggml_sycl_run_expert_refill_at_boundary_on_device("
BOUNDARY_SIG = "static void ggml_sycl_run_expert_refill_at_boundary(ggml_backend_t backend)"
LEDGER_SIG = "static ggml_sycl_expert_lent_ledger & ggml_sycl_expert_lent_ledger_locked("
REPORT_SIG = "static void ggml_sycl_expert_refill_log_report()"
PER_GRAPH_SIG = "static uint32_t ggml_sycl_moe_refill_per_graph()"
FREE_SIG = "static void ggml_backend_sycl_free(ggml_backend_t backend)"


def refill_bodies(cache: str, sycl: str) -> list[str]:
    """The refill code, cache side and backend side; the shared transaction body is in loan_bodies()."""
    c = norm(cache)
    s = norm(sycl)
    return [body(c, REFILL_SIG), body(s, SELECT_SIG), body(s, FINAL_SIG), body(s, DEVICE_ALLOC_SIG),
            body(s, REFILL_RUN_SIG), body(s, REFILL_API_SIG), body(s, ARM_SIG), body(s, BOUNDARY_DEVICE_SIG),
            body(s, BOUNDARY_SIG), body(s, LEDGER_SIG)]


DIRECTION_ENUM = "enum class ggml_sycl_expert_txn_direction { LOAN, REFILL, };"
LOAN_ENTRY = ("return ggml_sycl_expert_txn_run(ggml_sycl_expert_txn_direction::LOAN, device, bytes_wanted, probe_mode, "
              "resolver, out, warn_busy);")
REFILL_ENTRY = ("const ggml_sycl_txn_result answer = ggml_sycl_expert_txn_run(ggml_sycl_expert_txn_direction::REFILL, "
                "device, max_triplets, false, resolver, result, false);")
REFILL_FLAG = "const bool refill = direction == ggml_sycl_expert_txn_direction::REFILL;"
REFILL_CALL = "loaned = cache->refill_lent_triplet(triplet.refill_members, refilled);"


def claim_one_body_two_directions(sycl: str) -> bool:
    """One body runs both directions: the loan's entry and the refill's call it with their direction, and its lock,
    its publication and its epoch bump are the one copy both share."""
    s = norm(sycl)
    txn = body(s, TXN_SIG)
    return (s.count(DIRECTION_ENUM) == 1 and s.count("ggml_sycl_expert_txn_run(") == 3
            and body(s, LOAN_TXN_SIG).count(LOAN_ENTRY) == 1 and body(s, REFILL_RUN_SIG).count(REFILL_ENTRY) == 1
            and txn.count(REFILL_FLAG) == 1 and txn.count(LEND) == 1 and txn.count(REFILL_CALL) == 1
            and txn.count(INVENTORY_LOCK) == 1 and txn.count(PUBLISH) == 1 and txn.count(EPOCH) == 1
            and ordered(txn, REFILL_FLAG, INVENTORY_LOCK, REFILL_CALL, LEND, PUBLISH, EPOCH))


def claim_refill_never_waits(cache: str, sycl: str) -> bool:
    bodies = refill_bodies(cache, sycl)
    return all(bodies) and not any(token in b for b in bodies for token in FORBIDDEN_WAITS)


REFILL_SRC = "src_owner[i] = e.direct_alloc_owner;"
REFILL_COPY = "out.copy_events[i] = mem_copy_async(members[i].device_owner, src_owner[i], bytes[i], bq, deps[i]);"
BCS_QUEUE = "sycl::queue & bq = get_bcs_queue();"
ENTRY_OWNER = "entry.direct_alloc_owner = members[i].device_owner;"
ENTRY_EVENTS = ("entry.state = cache_entry_state::IN_PROGRESS;", "entry.has_ready_event = true;",
                "entry.ready_event = out.copy_events[i];", "entry.last_write_event = out.copy_events[i];",
                "entry.has_write_event = true;")


def claim_refill_copy_is_owner_to_owner_on_the_copy_engine(cache: str) -> bool:
    """The copy reads the host copy's own typed owner and writes the caller's device owner, so mem_copy_async takes
    the direct DMA and retains both until it completes; it runs on the BCS queue behind the host copy's writes, and
    the device entry is published IN_PROGRESS with the copy as its ready event and its last write."""
    b = body(norm(cache), REFILL_SIG)
    return (all(b.count(x) == 1 for x in (REFILL_SRC, REFILL_COPY, BCS_QUEUE, ENTRY_OWNER) + ENTRY_EVENTS)
            and all(d in b for d in DEPS) and "make_copy_handle_for_raw_ptr(" not in b and "host_task" not in b
            and ordered(b, REFILL_SRC, BCS_QUEUE, REFILL_COPY, ENTRY_OWNER))


EMPLACE = "entries_.emplace(device_key, device_entry[i]);"
MINT_MIRROR = ("device_mirror[i].handle = make_direct_entry_handle(device_key, cache_device, stored, "
               '"expert_refill/mirror");')
MINT_OUT = ("out.device_handles[i] = mem_handle::from_weight_lease_locked(device_key, cache_device, stored.device_ptr, "
            "stored.layout, true, &stored);")
SWAP_MIRROR = "std::swap(direct_expert_entries_.find(key)->second, device_mirror[i]);"
REMAP_ID = "id_to_key_.find(key)->second = make_direct_stage_key(cache_entry_type::MOE_EXPERT, key, GGML_LAYOUT_AOS);"
HOST_RETIRE = ("if (host != entries_.end() && !host->second.retired && "
               "host->second.direct_alloc_owner.owner_control_id() == src_control[i]) "
               "{ (void) retire_expert_entry_at_locked(host_key, &host_nodes[r++]); }")
PUBLICATION_FAILED = 'refusal = "publication failed";'
WITHDRAW_INSERTED = "for (int i = 0; i < inserted; ++i) { (void) transition_to_retired_locked("
REFILL_CATCH = "} catch (const ggml_sycl_fallback_error &) { throw; } catch (...) { " + WITHDRAW_INSERTED


def claim_refill_allocates_before_it_changes_anything(cache: str) -> bool:
    """Under the locks, the device entries' insertion and their lease minting are the only allocations, and they come
    before anything already published changes; a throw there withdraws the inserted entries (nothing resolves them
    yet). From the mirror swap on, nothing builds a node or allocates, so the host copies cannot be left half
    retired. The vector the host retires hand their nodes back into is sized before the locks."""
    b = body(norm(cache), REFILL_SIG)
    swap_at = b.find(SWAP_MIRROR)
    tail = b[swap_at:b.find("} } } ", swap_at) + 6] if swap_at >= 0 else ""
    needles = (EMPLACE, MINT_MIRROR, MINT_OUT, SWAP_MIRROR, REMAP_ID, HOST_RETIRE, PUBLICATION_FAILED, REFILL_CATCH,
               RELEASED_SIZED)
    return (all(b.count(x) == 1 for x in needles)
            and bool(tail) and not any(x in tail for x in NODE_BUILDERS + ("make_direct_entry_handle(",))
            and ordered(b, REFILL_COPY, RELEASED_SIZED, CACHE_LOCKS, EMPLACE, MINT_MIRROR, MINT_OUT, REFILL_CATCH,
                        PUBLICATION_FAILED, SWAP_MIRROR, REMAP_ID, HOST_RETIRE))


REFILL_SESSION = 'scoped_planned_materialization materialization(this, "expert refill");'


def claim_refill_publishes_in_a_planned_materialization(cache: str) -> bool:
    """The device entries are published inside a planned materialization session, as the loan registers its host copies
    in one: the session opens after the copy is submitted and before the locks the publication takes."""
    b = body(norm(cache), REFILL_SIG)
    return (b.count(REFILL_SESSION) == 1 and b.count("begin_planned_materialization(") == 0
            and ordered(b, REFILL_COPY, REFILL_SESSION, RELEASED_SIZED, CACHE_LOCKS, EMPLACE))


RESOLVE_EXPERT_SIG = "expert_resolve_result unified_cache::resolve_expert(const expert_resolve_request & req)"
RESOLVE_PAST_RETIRED = "if (entry_it == entries_.end() || entry_it->second.retired) {"
RESOLVE_MAPPED = ("const auto mapped = entries_.find(id_it->second); "
                  "if (mapped != entries_.end()) { entry_it = mapped; }")


def claim_a_retired_host_entry_does_not_shadow_the_refill(cache: str) -> bool:
    """resolve_expert looks the exact key up first. A refill leaves the withdrawn host copy there, retired, until the
    deferred-free pass finalizes it, while the id already maps to the device entry; so a retired entry under the
    exact key falls through to the id mapping, and the mapping replaces it only when it names an entry."""
    b = body(norm(cache), RESOLVE_EXPERT_SIG)
    return (b.count(RESOLVE_PAST_RETIRED) == 1 and b.count(RESOLVE_MAPPED) == 1
            and ordered(b, "entry_it = entries_.find(explicit_key);", RESOLVE_PAST_RETIRED, "id_to_key_.find(req.key)",
                        RESOLVE_MAPPED, "if (entry.retired) {"))


def claim_refill_retires_the_host_copy_exactly(cache: str) -> bool:
    """The host copy is withdrawn entry by entry under the loan's host ids, only while each still names the lent
    storage, after the device entry has taken the mirror and the id mapping over; never by the id-wide retire."""
    b = body(norm(cache), REFILL_SIG)
    return (b.count(HOST_RETIRE) == 1 and "retire_expert_entry_exact(" not in b
            and ordered(b, SWAP_MIRROR, REMAP_ID, HOST_RETIRE))


REFILL_CHECKS = ('refusal = "not lent";', 'refusal = "layout";', 'refusal = "not a lent copy";',
                 'refusal = "host copy not published";', 'refusal = "device entry exists";',
                 'refusal = "device storage size";')
REFILL_LOAD_EFFECT = ('const cache_load_effect load_effect = acquire_bound_load_effect(); if (load_effect.failed()) '
                      '{ out.reason = "load effect unavailable"; return false; }')


def claim_refill_checks_before_copying(cache: str) -> bool:
    """What would make the publication wrong is checked, all three members, before anything is copied: the load effect,
    the plan coherence, that each host copy is a loan's and is what resolves its id, that no device entry is in the
    way, and that the device storage is exactly the host copy's size."""
    b = body(norm(cache), REFILL_SIG)
    return (all(b.count(x) == 1 for x in (REFILL_LOAD_EFFECT, COHERENCE, REFUSE_EARLY) + REFILL_CHECKS)
            and ordered(b, REFILL_LOAD_EFFECT, COHERENCE, *REFILL_CHECKS, REFILL_SRC, REFUSE_EARLY, REFILL_COPY))


DEVICE_ALLOC = "ggml_sycl::mem_handle storage = ggml_sycl_device_expert_alloc(device, e->dst_size);"
ZONE_ALLOC = ("ggml_sycl::allocation_result allocation = ggml_sycl::unified_cache_zone_allocate_owner(device, "
              "ggml_sycl::vram_zone_id::WEIGHT, size);")


def claim_refill_lock_order(cache: str, sycl: str) -> bool:
    """The device storage is minted owner-first from the WEIGHT zone before g_tensor_inventory_mutex is taken, and the
    cache is asked to re-admit under it; the cache takes direct_stage_mutex_ and rw_mutex_ together, with std::lock,
    and never the inventory mutex; no refill code names g_sycl_graph_compute_mutex."""
    txn = body(norm(sycl), TXN_SIG)
    refill = body(norm(cache), REFILL_SIG)
    alloc = body(norm(sycl), DEVICE_ALLOC_SIG)
    return (txn.count(DEVICE_ALLOC) == 1 and alloc.count(ZONE_ALLOC) == 1
            and ordered(txn, DEVICE_ALLOC, INVENTORY_LOCK, REFILL_CALL)
            and "g_tensor_inventory_mutex" not in refill and refill.count(CACHE_LOCKS) >= 2
            and "std::unique_lock<std::shared_mutex> cache_lock(rw_mutex_);" not in refill
            and not any("g_sycl_graph_compute_mutex" in b for b in refill_bodies(cache, sycl)))


FORGET_HOST = ("(void) extra->forget_moe_storage_handle_on_device(k.expert_id, GGML_LAYOUT_AOS, "
               "ggml_sycl::mem_handle::HOST_DEVICE);")
REMEMBER_DEVICE = ("(void) extra->remember_moe_storage_handle(k.expert_id, GGML_LAYOUT_AOS, "
                   "std::move(refilled.device_handles[r]), &refilled.copy_events[r], 0, logical);")
POP_READMITTED = ("if (refill) { ledger.triplets.pop_back(); } "
                  "else { ledger.triplets.push_back({ placed[0], placed[1], placed[2] }); }")


def claim_refill_flips_back_and_forgets_the_host_copy(sycl: str) -> bool:
    """Each extra drops its lease on the withdrawn host copy before anything allocates, then remembers the device copy
    behind its copy event; the plan flips the triplet back on the device; the ledger loses it; the plan is published
    once, then the exec-graph epoch is bumped."""
    b = body(norm(sycl), TXN_SIG)
    return (b.count(FORGET_HOST) == 1 and b.count(REMEMBER_DEVICE) == 1 and b.count(POP_READMITTED) == 1
            and ordered(b, REFILL_CALL, FORGET_HOST, REMEMBER_DEVICE, LEND, CATCH_LENDING, FLIP_IF_LOANED, FLIP,
                        POP_READMITTED, PUBLISH, EPOCH))


DISARM = "if (!refill && lent != 0) { ledger.refill_armed = false; }"
CANCELLED = '"cancelled by a loan";'
SELECT_ARMED = "if (ledger.refill_armed) {"
SELECT_REMAINING = "*remaining = ledger.refill_armed ? static_cast<uint32_t>(ledger.triplets.size()) : 0;"
TXN_REMAINING = "result.n_remaining = ledger.refill_armed ? static_cast<uint32_t>(ledger.triplets.size()) : 0;"
RESERVE = "if (!refill) { ledger.triplets.reserve(ledger.triplets.size() + prepared.size()); }"


def claim_a_loan_cancels_the_refills(sycl: str) -> bool:
    """An accepted loan disarms the cursor, and a disarmed cursor selects nothing and holds nothing, so the refill
    still to run is cancelled and its triplets stay lent with the loan's; a loan accepted between the selection and
    the lock cancels too. The ledger room a loan appends into is reserved before its first triplet moves."""
    s = norm(sycl)
    txn = body(s, TXN_SIG)
    select = body(s, SELECT_SIG)
    return (txn.count(DISARM) == 1 and txn.count(CANCELLED) == 1 and txn.count(TXN_REMAINING) == 1
            and select.count(SELECT_ARMED) == 1 and select.count(SELECT_REMAINING) == 1
            and ordered(txn, HOIST_PLAN, RESERVE, CANCELLED, REFILL_CALL, LEND, POP_READMITTED, DISARM, TXN_REMAINING,
                        PUBLISH))


LEDGER_LOCK = "std::lock_guard<std::mutex> lock(g_tensor_inventory_mutex);"
LEDGER_IN_TXN = "ggml_sycl_expert_lent_ledger & ledger = ggml_sycl_expert_lent_ledger_locked(device, *current);"


def claim_ledger_is_guarded_by_the_inventory_mutex(sycl: str) -> bool:
    """Every site that reaches a device's ledger holds g_tensor_inventory_mutex: the selection and the arming take it,
    the body reaches the ledger only after its own lock, and the one direct reader besides those is the private
    fixture's teardown, under the same lock. The four subscripts are the declaration and those three."""
    s = norm(sycl)
    select = body(s, SELECT_SIG)
    arm = body(s, ARM_SIG)
    txn = body(s, TXN_SIG)
    return (s.count("g_sycl_expert_lent[") == 4
            and s.count("static ggml_sycl_expert_lent_ledger g_sycl_expert_lent[GGML_SYCL_MAX_DEVICES];") == 1
            and body(s, LEDGER_SIG).count("g_sycl_expert_lent[device]") == 1
            and ordered(select, LEDGER_LOCK, "ggml_sycl_expert_lent_ledger_locked(device, current)")
            and ordered(arm, LEDGER_LOCK, "g_sycl_expert_lent[device]")
            and txn.count(LEDGER_IN_TXN) == 1 and ordered(txn, INVENTORY_LOCK, LEDGER_IN_TXN)
            and "::ggml_sycl_publish_plan_locked(found); g_sycl_expert_lent[device] = ggml_sycl_expert_lent_ledger{};"
            in s)


SET_INACTIVE_TO_REFILL = (SET_INACTIVE, RUN_PENDING, DEFERRED_FREES,
                          "ggml_sycl_run_expert_refill_at_boundary(backend);")
RUN_REFILL = "ggml_sycl_run_expert_refill_at_boundary(backend);"
BOUNDARY_SKIP = ("if (device < 0 || device >= GGML_SYCL_MAX_DEVICES || "
                 "g_sycl_expert_lent_count[device].load(std::memory_order_acquire) == 0) { return 0; }")
BOUNDARY_ARM = "ggml_sycl_expert_refill_set_armed(device, decode); if (!decode) { return 0; }"
BOUNDARY_PHASE = ("ctx->device, ctx->graph_phase_is_decode.load(std::memory_order_relaxed), "
                  "ctx->decode_graph_count.load(std::memory_order_relaxed), &ctx->refill_decode_graph, "
                  "ggml_sycl_moe_refill_per_graph(), nullptr, nullptr);")
MMID_REFILL_FORBIDDEN = ("ggml_sycl_expert_txn_run(", "ggml_sycl_run_expert_refill",
                         "ggml_sycl_expert_refill_transaction_run(", "refill_lent_triplet(")


def claim_refill_runs_at_the_decode_boundary(sycl: str) -> bool:
    """ggml_backend_sycl_synchronize() reaches the refill at every call, once the compute flag is clear, after the
    pending loan and the deferred-free pass (which reclaims the loan's withdrawn device entries and returns the zone
    room the ring released). It asks, without a lock, whether the device has lent anything; the graph that just ran
    then arms the cursor (decode) or disarms it (prompt), and only a decode graph re-admits; how many, and how often,
    is claim_one_decode_graph_admits_one_refill's. mul_mat_id names no refill entry; the body refuses "graph
    active"."""
    s = norm(sycl)
    sync = body(s, SYNC_SIG)
    boundary = body(s, BOUNDARY_DEVICE_SIG)
    mmid = body(s, MMID_SIG)
    return (s.count(RUN_REFILL) == 1 and ordered(sync, *SET_INACTIVE_TO_REFILL)
            and boundary.count(BOUNDARY_SKIP) == 1 and boundary.count(BOUNDARY_ARM) == 1
            and ordered(boundary, BOUNDARY_SKIP, BOUNDARY_ARM, "ggml_sycl_expert_refill_transaction_run(")
            and body(s, BOUNDARY_SIG).count(BOUNDARY_PHASE) == 1
            and body(s, TXN_SIG).count(GRAPH_ACTIVE) == 1
            and bool(mmid) and not any(x in mmid for x in MMID_REFILL_FORBIDDEN))


ENV_NAME = '"GGML_SYCL_MOE_REFILL_PER_GRAPH"'
PER_GRAPH_ONCE = "static const uint32_t per_graph = [] {"
PER_GRAPH_GETENV = 'const char * text = std::getenv("GGML_SYCL_MOE_REFILL_PER_GRAPH");'
PER_GRAPH_DEFAULT = "if (!text || !*text) { return 1u; }"
PER_GRAPH_RANGE = "if (end == text || *end != '\\0' || value < 1 || value > 1024) {"


def sycl_sources_outside_tests() -> list[str]:
    root = ROOT / "ggml" / "src" / "ggml-sycl"
    return [p.read_text(errors="replace") for p in sorted(root.rglob("*"))
            if p.suffix in (".cpp", ".hpp", ".h") and "tests" not in p.relative_to(root).parts]


def claim_refill_env_read_once(sycl: str, sources: list[str]) -> bool:
    """GGML_SYCL_MOE_REFILL_PER_GRAPH is read in one place, once per process (a function-local static), defaults to 1
    and keeps 1 for anything outside 1..1024, so no value turns the refill off."""
    b = body(norm(sycl), PER_GRAPH_SIG)
    return (sum(norm(src).count(ENV_NAME) for src in sources) == 1
            and all(b.count(x) == 1 for x in (PER_GRAPH_ONCE, PER_GRAPH_GETENV, PER_GRAPH_DEFAULT, PER_GRAPH_RANGE))
            and ordered(b, PER_GRAPH_ONCE, PER_GRAPH_GETENV, PER_GRAPH_DEFAULT, PER_GRAPH_RANGE))


TEARDOWN_WARN = ('GGML_LOG_WARN("[MOE-EXEC] refill device %d: %u/%u triplets re-admitted\\n", device, readmitted, '
                 "lent);")
TAKE_LENT = "const uint32_t lent = g_sycl_expert_lent_total[device].exchange(0, std::memory_order_acq_rel);"
TAKE_READMITTED = ("const uint32_t readmitted = g_sycl_expert_readmitted_total[device].exchange(0, "
                   "std::memory_order_acq_rel);")
FREE_REPORTS = "ggml_sycl_moe_exec_log_report(); ggml_sycl_expert_refill_log_report();"
REFILL_BUSY_WARN = ('if ((n - 1) % warn_every == 0) { GGML_LOG_WARN("[MOE-EXEC] refill device %d refused: %s (%u in a '
                    'row; logged once per %u)\\n", device, result->reason, n, warn_every); }')
REFILL_RESET = ("if (code == ggml_sycl_txn_result::ACCEPTED || result->n_dropped != 0 || "
                'std::strcmp(result->reason, "nothing to refill") == 0) { streak.store(0, std::memory_order_relaxed); '
                "return remaining; }")
DROP_WARN = ('GGML_LOG_WARN("[MOE-EXEC] refill device %d: layer %d expert %d stays on the host: %s\\n", device, '
             "dropped[i].first.layer_id, dropped[i].first.expert_id, dropped[i].second);")


def claim_refill_warn_formats(sycl: str) -> bool:
    """The teardown line prints next to the [MOE-EXEC] summary, once: the counts are taken as they print. A refill's
    refusal is WARNed by its boundary runner alone, the first in a row and then one in every 64, with the count; an
    accepted refill, one that dropped a triplet (its drop is WARNed already) or one with nothing to do resets the
    count. The body WARNs no refill line but the drop's (one per decode graph would flood the log; a triplet is
    dropped once), and the loan's busy streak is not the refill's."""
    s = norm(sycl)
    report = body(s, REPORT_SIG)
    boundary = body(s, BOUNDARY_DEVICE_SIG)
    txn = body(s, TXN_SIG)
    return (s.count(TEARDOWN_WARN) == 1 and report.count(TEARDOWN_WARN) == 1
            and ordered(report, TAKE_LENT, TAKE_READMITTED, TEARDOWN_WARN)
            and body(s, FREE_SIG).count(FREE_REPORTS) == 1
            and boundary.count("GGML_LOG_WARN(") == 1 and boundary.count(BUSY_EVERY) == 1
            and ordered(boundary, REFILL_RESET, BUSY_EVERY, REFILL_BUSY_WARN)
            and txn.count('GGML_LOG_WARN("[MOE-EXEC] refill') == 1 and txn.count(DROP_WARN) == 1
            and not any("g_sycl_expert_loan_busy_streak" in b for b in refill_bodies("", sycl)[1:]))


FINAL_REASONS = ("invalid member", "not lent", "layout", "not a lent copy", "host copy not published",
                 "device storage size")
FINAL_POP = ("stop = refilled.reason; transient = !ggml_sycl_expert_refill_refusal_is_final(stop); "
             "if (!transient) { drop_back(stop); } break;")


def claim_final_refusals_name_the_cache_refusals(cache: str, sycl: str) -> bool:
    """Each reason the transaction treats as final (the triplet leaves the ledger rather than being retried at every
    decode graph) is a refusal the cache's refill really returns, so a renamed refusal cannot silently turn final into
    retried forever."""
    refill = body(norm(cache), REFILL_SIG)
    final = body(norm(sycl), FINAL_SIG)
    return (all(('"%s"' % r) in final for r in FINAL_REASONS)
            and all(('refusal = "%s";' % r) in refill for r in FINAL_REASONS)
            and body(norm(sycl), TXN_SIG).count(FINAL_POP) == 1)


# --- a lent triplet the refill can never prepare leaves the ledger, WARNed (llama.cpp-sif1) -------------------------

RESOLVE_FAILED = ("if (!e || !(*resolver)(*e, target) || !target.extra) { "
                  'stop = refill ? "no tensor for a lent expert" : "no tensor for a selected expert"; break; }')
ARENA_TOLD_APART = ("const bool arena = cache->arena_active() && "
                    "cache->zone_capacity(ggml_sycl::vram_zone_id::WEIGHT) != 0; "
                    'stop = arena ? "WEIGHT zone full" : "no WEIGHT arena"; transient = arena; break;')
UNPREPARABLE = ("} else if (refill && !transient) { unpreparable = { keys[3 * t], keys[3 * t + 1], keys[3 * t + 2] }; "
                "unpreparable_reason = stop; }")
EARLY_REFUSAL = "if (prepared.empty() && !unpreparable_reason) {"
PLAN_CHANGED = 'return refuse(ggml_sycl_txn_result::BUSY, "busy (plan changed while acquiring the transaction lock)");'
DROP_AT_BACK = ("if (unpreparable_reason && !ledger.triplets.empty() && ledger.triplets.back() == unpreparable) "
                "{ drop_back(unpreparable_reason); }")
DROP_COUNTED = "result.n_dropped = static_cast<uint32_t>(n_dropped);"
LENT_COUNT_STORE = ("g_sycl_expert_lent_count[device].store(static_cast<uint32_t>(ledger.triplets.size()), "
                    "std::memory_order_release);")


def claim_an_unpreparable_triplet_leaves_the_ledger(sycl: str) -> bool:
    """A refill's preparation that fails for good on a triplet (no tensor resolves a member, or the device has no WEIGHT
    arena, told apart from a full zone, which stays transient) does not return early: under the inventory lock, once
    the publication is checked unchanged, the triplet leaves the ledger if it is the back by then (every triplet
    selected before it moved), and the ledger count is stored after. Left there, it would stop every later refill at
    the same triplet (llama.cpp-sif1)."""
    b = body(norm(sycl), TXN_SIG)
    needles = (RESOLVE_FAILED, ARENA_TOLD_APART, UNPREPARABLE, EARLY_REFUSAL, DROP_AT_BACK, DROP_COUNTED,
               LENT_COUNT_STORE)
    return (all(b.count(x) == 1 for x in needles)
            and ordered(b, RESOLVE_FAILED, DEVICE_ALLOC, ARENA_TOLD_APART, UNPREPARABLE, EARLY_REFUSAL, INVENTORY_LOCK,
                        PLAN_CHANGED, LEDGER_IN_TXN, REFILL_CALL, POP_READMITTED, DROP_AT_BACK, DROP_COUNTED,
                        LENT_COUNT_STORE))


DROP_BACK = ("auto drop_back = [&](const char * reason) { GGML_ASSERT(n_dropped < dropped.size()); "
             "dropped[n_dropped++] = { ledger.triplets.back()[0], reason }; ledger.triplets.pop_back(); };")
WARN_DROPPED_SIG = "auto warn_dropped = [&]"
WARN_EACH = "for (size_t i = 0; i < n_dropped; ++i) {"
WARN_AFTER_UNLOCK = "lock.unlock(); warn_dropped();"


def claim_each_dropped_triplet_is_warned(sycl: str) -> bool:
    """Both drops, the cache's final refusal and the unpreparable triplet, go through one recorder that pops the back
    and keeps its key and reason; each recorded drop is WARNed once, naming the device, the layer, the expert and the
    reason, after the inventory lock is released, on both exits that follow the drops: the one where nothing moved
    and the one that publishes. Nothing filters the WARN (llama.cpp-sif1)."""
    b = body(norm(sycl), TXN_SIG)
    at = b.find(WARN_DROPPED_SIG)
    warn = body(b[at:], WARN_DROPPED_SIG) if at >= 0 else ""
    return (b.count(DROP_BACK) == 1 and b.count("drop_back(") == 2 and b.count(DROP_WARN) == 1
            and warn.count(DROP_WARN) == 1 and warn.count(WARN_EACH) == 1 and warn.count("if (") == 0
            and b.count("warn_dropped();") == 2 and b.count(WARN_AFTER_UNLOCK) == 2
            and ordered(b, DROP_BACK, WARN_DROPPED_SIG, FINAL_POP, DROP_AT_BACK,
                        "if (lent == 0) { " + WARN_AFTER_UNLOCK, PUBLISH, WARN_AFTER_UNLOCK + " if (!swapped) {"))


GRAPH_COMPUTE_SIG = ("static ggml_status ggml_backend_sycl_graph_compute_unchecked(ggml_backend_t backend, "
                     "ggml_cgraph * cgraph)")
DECODE_COUNTER = "std::atomic<uint64_t> decode_graph_count{ 0 };"
REFILL_MARK = "std::atomic<uint64_t> refill_decode_graph{ 0 };"
DECODE_BUMP = "if (cached_is_decode) { sycl_ctx->decode_graph_count.fetch_add(1, std::memory_order_relaxed);"
ONCE_PER_GRAPH = ("if (refilled_at->exchange(decode_graphs, std::memory_order_relaxed) == decode_graphs) { "
                  "*result = ggml_sycl_expert_loan_result{};")
SKIP_REASON = 'result->reason = "no decode graph since the last refill"; return result->n_remaining; }'


MOVE_LOOP_SIG = "for (prepared_triplet & triplet : prepared)"
LOOP_TAIL = "++lent; freed += refill ? refilled.bytes_readmitted : record.bytes_freed; } if (escaped) { break; } }"
LOOP_STOPS = ("transient = ledger.refill_armed; break; }", FINAL_POP, "stop = record.reason; break; }",
              "if (escaped) { break; }")


def claim_the_move_loop_moves_every_prepared_triplet(sycl: str) -> bool:
    """The move loop stops only where a triplet did not move: the cursor no longer holds it at its back (cancelled by a
    loan, or changed), the cache refuses it (refill or loan), or an exception escaped. After a triplet moves, the
    loop counts it, sums its bytes and goes on, so a refill of more than one per call checks the next triplet against
    the back the first one's pop left (llama.cpp-sif1)."""
    loop = body(body(norm(sycl), TXN_SIG), MOVE_LOOP_SIG)
    return (loop.count("break;") == len(LOOP_STOPS) and all(loop.count(x) == 1 for x in LOOP_STOPS)
            and loop.endswith(LOOP_TAIL)
            and ordered(loop, CANCELLED, REFILL_CALL, FINAL_POP, POP_READMITTED, LOOP_TAIL))


def claim_one_decode_graph_admits_one_refill(sycl: str, common: str, sources: list[str]) -> bool:
    """A context synchronizes more than once per graph, so the boundary alone does not bound the refill. Each context
    counts its decode graph_compute calls (graph_compute alone bumps the count, and only for a decode graph) beside
    the count its last refill ran at; the boundary passes both, and once it has armed the cursor it runs the refill
    only when the count has moved, so one decode graph re-admits at most GGML_SYCL_MOE_REFILL_PER_GRAPH triplets
    (llama.cpp-sif1)."""
    s = norm(sycl)
    c = norm(common)
    boundary = body(s, BOUNDARY_DEVICE_SIG)
    return (c.count(DECODE_COUNTER) == 1 and c.count(REFILL_MARK) == 1
            and body(s, GRAPH_COMPUTE_SIG).count(DECODE_BUMP) == 1
            and sum(norm(src).count("decode_graph_count.fetch_add(") for src in sources) == 1
            and boundary.count(ONCE_PER_GRAPH) == 1 and boundary.count(SKIP_REASON) == 1
            and ordered(boundary, BOUNDARY_SKIP, BOUNDARY_ARM, ONCE_PER_GRAPH, SKIP_REASON,
                        "ggml_sycl_expert_refill_transaction_run(")
            and body(s, BOUNDARY_SIG).count(BOUNDARY_PHASE) == 1)


# --- positive tests -------------------------------------------------------------------------------------------------

def test_loan_never_waits():
    assert claim_loan_never_waits(CACHE, SYCL)


def test_device_free_gated_on_the_copy():
    assert claim_device_free_gated_on_the_copy(CACHE, SYCL)


def test_retire_is_exact():
    assert claim_retire_is_exact(CACHE)


def test_failed_registration_restores():
    assert claim_failed_registration_restores(CACHE)


def test_rollback_reinserts_nodes():
    assert claim_rollback_reinserts_nodes(CACHE)


def test_registration_catches_everything():
    assert claim_registration_catches_everything(CACHE)


def test_busy_warn_is_rate_limited():
    assert claim_busy_warn_is_rate_limited(SYCL)


def test_batched_dispatch_skips_a_repeated_ready_event():
    assert claim_batched_dispatch_skips_a_repeated_ready_event(SYCL)


def test_registration_checked_before_the_copies():
    assert claim_registration_checked_before_the_copies(CACHE)


def test_existing_host_entry_never_replaced():
    assert claim_existing_host_entry_never_replaced(CACHE, SYCL)


def test_graph_mutex_untouched():
    assert claim_graph_mutex_untouched(CACHE, SYCL)


def test_lock_order():
    assert claim_lock_order(CACHE, SYCL)


def test_request_is_lock_free():
    assert claim_request_is_lock_free(SYCL)


def test_loan_runs_at_the_graph_boundary():
    assert claim_loan_runs_at_the_graph_boundary(SYCL)


def test_busy_reposts_the_request():
    assert claim_busy_reposts_the_request(SYCL)


def test_exception_publishes_what_was_lent():
    assert claim_exception_publishes_what_was_lent(SYCL)


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


def test_one_body_two_directions():
    assert claim_one_body_two_directions(SYCL)


def test_refill_never_waits():
    assert claim_refill_never_waits(CACHE, SYCL)


def test_refill_copy_is_owner_to_owner_on_the_copy_engine():
    assert claim_refill_copy_is_owner_to_owner_on_the_copy_engine(CACHE)


def test_refill_allocates_before_it_changes_anything():
    assert claim_refill_allocates_before_it_changes_anything(CACHE)


def test_refill_publishes_in_a_planned_materialization():
    assert claim_refill_publishes_in_a_planned_materialization(CACHE)


def test_a_retired_host_entry_does_not_shadow_the_refill():
    assert claim_a_retired_host_entry_does_not_shadow_the_refill(CACHE)


def test_refill_retires_the_host_copy_exactly():
    assert claim_refill_retires_the_host_copy_exactly(CACHE)


def test_refill_checks_before_copying():
    assert claim_refill_checks_before_copying(CACHE)


def test_refill_lock_order():
    assert claim_refill_lock_order(CACHE, SYCL)


def test_refill_flips_back_and_forgets_the_host_copy():
    assert claim_refill_flips_back_and_forgets_the_host_copy(SYCL)


def test_a_loan_cancels_the_refills():
    assert claim_a_loan_cancels_the_refills(SYCL)


def test_ledger_is_guarded_by_the_inventory_mutex():
    assert claim_ledger_is_guarded_by_the_inventory_mutex(SYCL)


def test_refill_runs_at_the_decode_boundary():
    assert claim_refill_runs_at_the_decode_boundary(SYCL)


def test_refill_env_read_once():
    assert claim_refill_env_read_once(SYCL, sycl_sources_outside_tests())


def test_refill_warn_formats():
    assert claim_refill_warn_formats(SYCL)


def test_final_refusals_name_the_cache_refusals():
    assert claim_final_refusals_name_the_cache_refusals(CACHE, SYCL)


def test_comment_cannot_satisfy_a_claim():
    """A needle that survives only in a comment does not count: the stripped text loses it."""
    commented = norm("// " + COPY + "\n")
    assert COPY not in commented


# --- mutants --------------------------------------------------------------------------------------------------------

def _once(text: str, old: str, new: str) -> str:
    """`text` with the one occurrence of `old` (matched on normalized text) replaced by `new`."""
    assert new != old, f"a mutant must change its anchor: {old!r}"
    n = norm(text)
    assert n.count(old) == 1, f"mutant anchor must match exactly once: {old!r} x{n.count(old)}"
    return n.replace(old, new, 1)


def _once_in(text: str, signature: str, old: str, new: str) -> str:
    """`text`, normalized, with the one occurrence of `old` inside the body of `signature` replaced by `new`: for an
    anchor that also occurs elsewhere in the file."""
    assert new != old, f"a mutant must change its anchor: {old!r}"
    n = norm(text)
    b = body(n, signature)
    assert b and n.count(b) == 1, f"body of {signature!r} must be present and unique"
    assert b.count(old) == 1, f"mutant anchor must match exactly once in the body: {old!r} x{b.count(old)}"
    return n.replace(b, b.replace(old, new, 1), 1)


def test_mutant_loan_waits_on_its_copy_fails():
    assert not claim_loan_never_waits(_once(CACHE, COPY, COPY + " out.copy_events[i].wait();"), SYCL)


def test_mutant_loan_asks_an_event_status_fails():
    assert not claim_loan_never_waits(
        _once(CACHE, COPY,
              COPY + " (void) out.copy_events[i].get_info<sycl::info::event::command_execution_status>();"),
        SYCL)


def test_mutant_loan_finalizes_inline_fails():
    """The id-wide retire's immediate finalize asks the fill event's status: the loan leaves it to the deferred pass."""
    assert not claim_loan_never_waits(
        _once_in(CACHE, LOAN_SIG, "out.accepted = true; out.reason = \"OK\";",
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


def test_mutant_lease_minted_after_the_stamp_fails():
    """The lease minted beside the stamp and the retire: a mint that throws for a later member leaves an earlier one
    stamped and retired, its lease gone, with no host copy and the plan unflipped."""
    n = _once(CACHE, HOLD, "")
    n = n.replace(STAMP_FLAG, STAMP_FLAG + " " + HOLD, 1)
    assert not claim_device_free_gated_on_the_copy(n, SYCL)


def test_mutant_copy_source_not_the_owner_fails():
    """A raw-pointer copy handle retains nothing: the deferred free could release the storage mid-copy."""
    assert not claim_device_free_gated_on_the_copy(
        _once_in(CACHE, LOAN_SIG, SRC_IS_OWNER,
                 "src_owner[i] = make_copy_handle_for_raw_ptr(e.device_ptr, GGML_LAYOUT_AOS, cache_device, e.size);"),
        SYCL)


def test_mutant_copy_not_behind_the_fill_fails():
    assert not claim_device_free_gated_on_the_copy(_once_in(CACHE, LOAN_SIG, DEPS[0], ""), SYCL)


def test_mutant_host_copy_published_ready_fails():
    """The host entry published READY while its copy is still in flight: a reader could see stale bytes."""
    assert not claim_device_free_gated_on_the_copy(
        _once(CACHE, REGISTER, REGISTER.replace(", m.host_owner, &out.copy_events[i], false)",
                                                ", m.host_owner, nullptr, false)")), SYCL)


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
    """The count of retired entries left one too high: the deferred-free pass keeps looking for an entry that is
    live."""
    assert not claim_failed_registration_restores(_once_in(CACHE, RESTORE_SIG, COUNT_BACK, ""))


def test_mutant_registration_failure_keeps_the_host_copies_fails():
    assert not claim_failed_registration_restores(_once(CACHE, HOST_WITHDRAW, ""))


def test_mutant_load_effect_unchecked_fails():
    assert not claim_registration_checked_before_the_copies(_once(CACHE, LOAD_EFFECT, ""))


def test_mutant_coherence_unchecked_fails():
    assert not claim_registration_checked_before_the_copies(_once_in(CACHE, LOAN_SIG, COHERENCE, ""))


def test_mutant_existing_host_entry_unchecked_fails():
    assert not claim_registration_checked_before_the_copies(_once(CACHE, HOST_EXISTS, ""))


def test_mutant_idle_host_entry_let_through_fails():
    """c-mwmr-3: refusing only a retired or leased host entry lets an idle one through to be replaced, storage and
    all, by a registration the loan could not undo."""
    busy_only = ("if (host != entries_.end() && (host->second.retired || host->second.in_use_count.load() != 0)) "
                 '{ refusal = "host entry exists"; break; }')
    assert not claim_existing_host_entry_never_replaced(_once(CACHE, HOST_EXISTS, busy_only), SYCL)


def test_mutant_loan_registration_may_replace_fails():
    assert not claim_existing_host_entry_never_replaced(
        _once(CACHE, REGISTER, REGISTER.replace("&out.copy_events[i], false)", "&out.copy_events[i], true)")), SYCL)


def test_mutant_registration_ignores_replace_existing_fails():
    assert not claim_existing_host_entry_never_replaced(
        _once(CACHE, NO_REPLACE, "if (old != entries_.end()) {"), SYCL)


def test_mutant_typed_registration_drops_replace_existing_fails():
    owned = body(norm(CACHE), OWNED_SIG)
    dropped = owned.replace(FORWARD, "ready_event);")
    assert not claim_existing_host_entry_never_replaced(norm(CACHE).replace(owned, dropped, 1), SYCL)


def test_mutant_refusal_reason_never_reaches_the_warn_fails():
    """A loan refusal whose reason is not the record's: the WARN would not name "host entry exists"."""
    assert not claim_existing_host_entry_never_replaced(
        CACHE, _once(SYCL, STOP_FROM_RECORD, 'stop = "loan refused";'))


def test_mutant_rollback_vector_sized_in_the_rollback_fails():
    """The rollback allocating its vector after host copies are published: a throw there strands them."""
    n = _once_in(CACHE, LOAN_SIG, RELEASED_SIZED, "")
    n = n.replace(REGISTRATION_REFUSED, RELEASED_SIZED + " " + REGISTRATION_REFUSED, 1)
    assert not claim_failed_registration_restores(n)


def test_mutant_registration_allocation_escapes_fails():
    """The typed wrapper's allocation failure escapes step 4 with the device entries withdrawn and no rollback."""
    assert not claim_failed_registration_restores(
        _once(CACHE, REGISTER_CATCH, "} catch (const std::length_error &) { published = false; } "
                                     "if (!published) { failed = true; break; }"))


def test_mutant_rollback_withdraws_another_entry_fails():
    """The rollback withdrawing whatever entry sits under the host id, not only the one this loan registered."""
    assert not claim_failed_registration_restores(
        _once(CACHE, HOST_WITHDRAW, "{ (void) retire_expert_entry_at_locked(host_key, &host_nodes[r++]); }"))


def test_mutant_restore_emplaces_the_mapping_fails():
    """A fresh mapping node: the rollback allocates, and a throw there leaves the triplet withdrawn."""
    assert not claim_rollback_reinserts_nodes(_once(CACHE, MAPPING_BACK, "id_to_key_.emplace(exact.id, exact);"))


def test_mutant_restore_emplaces_the_mirror_fails():
    assert not claim_rollback_reinserts_nodes(
        _once(CACHE, MIRROR_BACK, MIRROR_BACK.replace("direct_expert_entries_.insert(std::move(released.mirror));",
                                                      "direct_expert_entries_.emplace(exact.id, "
                                                      "std::move(released.mirror.mapped()));")))


def test_mutant_retire_erases_the_mapping_fails():
    """The mapping erased, not kept: the restore would have to build a new node."""
    assert not claim_rollback_reinserts_nodes(
        _once(CACHE, EXACT_REMAP, "remap_or_erase_id_mapping_locked(exact.id, exact);"))


def test_mutant_remap_frees_the_node_it_was_asked_to_keep_fails():
    assert not claim_rollback_reinserts_nodes(_once(CACHE, REMAP_EXTRACT, "} else { id_to_key_.erase(mapped); }"))


def test_mutant_rollback_restores_between_withdrawals_fails():
    """One pass: a member's device node goes back while a later member's host copies are still in the maps, which
    can then hold more entries than when the node came out, so the re-insert can grow the bucket array."""
    assert not claim_rollback_reinserts_nodes(
        _once(CACHE, TWO_PASS, HOST_WITHDRAW + " } " + RESTORE_CALL + " } for (int i = 0; i < 3; ++i) {"))


def test_mutant_registration_catch_only_bad_alloc_fails():
    """Any other exception from the registration escapes with the device entries withdrawn and no rollback."""
    assert not claim_registration_catches_everything(
        _once(CACHE, REGISTER_CATCH, "} catch (const std::bad_alloc &) { published = false; } "
                                     "if (!published) { failed = true; break; }"))


def test_mutant_registration_catch_swallows_the_fallback_fails():
    assert not claim_registration_catches_everything(
        _once(CACHE, REGISTER_CATCH, REGISTER_CATCH.replace("} catch (const ggml_sycl_fallback_error &) { throw; }",
                                                            "}")))


def test_mutant_transaction_takes_the_graph_mutex_fails():
    assert not claim_graph_mutex_untouched(
        CACHE, _once_in(SYCL, TXN_SIG, INVENTORY_LOCK,
                        "std::lock_guard<std::mutex> graph(g_sycl_graph_compute_mutex); " + INVENTORY_LOCK))


def test_mutant_pending_runner_takes_the_graph_mutex_fails():
    assert not claim_graph_mutex_untouched(
        CACHE,
        _once(SYCL, PENDING_TAKE, "std::lock_guard<std::mutex> graph(g_sycl_graph_compute_mutex); " + PENDING_TAKE))


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


def test_mutant_busy_drops_the_request_fails():
    """A request taken and then refused BUSY is lost: the ring never gets its bytes."""
    assert not claim_busy_reposts_the_request(_once(SYCL, REPOST, "return true; }"))


def test_mutant_refused_loan_reposted_fails():
    """Every refusal re-posted: a deterministic one would be retried at every boundary, and refused every time."""
    assert not claim_busy_reposts_the_request(
        _once(SYCL, NOT_BUSY_RETURNS,
              NOT_BUSY_RETURNS.replace("code != ggml_sycl_txn_result::BUSY", "code == ggml_sycl_txn_result::ACCEPTED")))


def test_mutant_busy_warns_at_every_boundary_fails():
    """A held lease refuses at every synchronize: one WARN per boundary floods the log."""
    assert not claim_busy_warn_is_rate_limited(
        _once(SYCL, BUSY_WARN, BUSY_WARN.replace("if ((n - 1) % warn_every == 0) {", "if (true) {")))


def test_mutant_busy_streak_never_reset_fails():
    """The count carried across an accepted loan: the next run of refusals starts mid-count, its first unseen."""
    assert not claim_busy_warn_is_rate_limited(
        _once(SYCL, NOT_BUSY_RETURNS, "if (code != ggml_sycl_txn_result::BUSY) { return true; }"))


def test_mutant_runner_lets_the_transaction_warn_busy_fails():
    assert not claim_busy_warn_is_rate_limited(
        _once(SYCL, PENDING_RUN, PENDING_RUN.replace("resolver, result, false);", "resolver, result);")))


def test_mutant_transaction_ignores_warn_busy_fails():
    assert not claim_busy_warn_is_rate_limited(
        _once_in(SYCL, TXN_SIG, REFUSE_GATE,
                 REFUSE_GATE.replace(
                     "if (!probe_mode && !refill && (warn_busy || code != ggml_sycl_txn_result::BUSY)) {",
                     "if (!probe_mode && !refill) {")))


def test_mutant_busy_warn_drops_the_count_fails():
    assert not claim_busy_warn_is_rate_limited(
        _once(SYCL, BUSY_WARN, BUSY_WARN.replace(" (%u in a row, re-posted; logged once per %u)", "")
              .replace("result->reason, n, warn_every);", "result->reason);")))


def test_mutant_lending_exception_escapes_unpublished_fails():
    """No catch: the triplets already lent stay withdrawn in the cache, forgotten by their extras, and on_device in the
    plan."""
    assert not claim_exception_publishes_what_was_lent(
        _once(SYCL, CATCH_LENDING, "} catch (const ggml_sycl_fallback_error &) { throw; }"))


def test_mutant_lent_triplet_unflipped_on_a_throw_fails():
    """The triplet whose remember threw is withdrawn in the cache, so the plan must say it is on the host."""
    assert not claim_exception_publishes_what_was_lent(
        _once(SYCL, FLIP_IF_LOANED, "if (!escaped) { const ggml_sycl::expert_placement_key placed[3] = {"))


def test_mutant_lending_exception_swallowed_fails():
    """The exception dropped after the publication: the caller never learns the loan stopped on a failure."""
    swallowed = RETHROW_EXITS[2][len(RETHROW) + 1:]
    assert not claim_exception_publishes_what_was_lent(_once(SYCL, RETHROW_EXITS[2], swallowed))


def test_mutant_publication_allocates_after_lending_fails():
    """The snapshot allocated after the loop: a throw there strands every triplet lent."""
    n = _once_in(SYCL, TXN_SIG, HOIST_SNAPSHOT, "")
    late = "next->plan = std::move(next_plan);"
    n = _once_in(n, TXN_SIG, late, HOIST_SNAPSHOT + " " + late)
    assert not claim_exception_publishes_what_was_lent(n)


def test_mutant_forget_interleaved_with_remember_fails():
    """One loop for both: a remember that throws leaves the later members' extras holding their withdrawn copies."""
    assert not claim_exception_publishes_what_was_lent(_once(SYCL, FORGET_LOOP_END, FORGET))


def test_mutant_route_drops_the_ready_event_fails():
    """The planned prompt host route without its ready event: the CPU job could read the host copy before it lands."""
    assert not claim_host_consumer_orders_on_the_copy(
        _once(SYCL, ROUTE_READY, ROUTE_READY.replace(
            "host_entry.has_ready_event = route.has_ready_event; host_entry.ready_event = route.ready_event; ", "")))


def test_mutant_dispatch_ignores_the_ready_event_fails():
    assert not claim_host_consumer_orders_on_the_copy(_once(SYCL, DISPATCH_READY, ""))


def test_mutant_dispatch_pushes_every_row_fails():
    """Each row of a lent expert adds its copy's event again to the blocking host wait."""
    assert not claim_host_consumer_orders_on_the_copy(
        _once(SYCL, DISPATCH_READY, "if (entry.has_ready_event) { copy_events.push_back(entry.ready_event); }"))


def test_mutant_batched_dispatch_pushes_every_row_fails():
    """Each row of a lent expert adds its copy's event again to the batched job's inputs."""
    assert not claim_batched_dispatch_skips_a_repeated_ready_event(
        _once(SYCL, BATCH_READY, "for (const expert_dispatch_entry & entry : entries) { if (entry.has_ready_event) { "
                                 "job_inputs.events.push_back(entry.ready_event); } }"))


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
    assert not claim_owner_checks_refuse_before_copying(_once_in(CACHE, LOAN_SIG, LAYOUT_CHECK, ""))


def test_mutant_refusal_falls_through_to_the_copy_fails():
    assert not claim_owner_checks_refuse_before_copying(_once_in(CACHE, LOAN_SIG, REFUSE_EARLY, ""))


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


# --- refill mutants (llama.cpp-sif1) ----------------------------------------------------------------------------------


def test_mutant_refill_runs_the_loan_direction_fails():
    assert not claim_one_body_two_directions(_once(SYCL, REFILL_ENTRY, REFILL_ENTRY.replace("::REFILL,", "::LOAN,")))


def test_mutant_refill_publishes_twice_fails():
    assert not claim_one_body_two_directions(_once_in(SYCL, TXN_SIG, REFILL_CALL, REFILL_CALL + " " + PUBLISH))


def test_mutant_refill_waits_on_its_copy_fails():
    assert not claim_refill_never_waits(_once(CACHE, REFILL_COPY, REFILL_COPY + " out.copy_events[i].wait();"), SYCL)


def test_mutant_refill_boundary_drains_the_queue_fails():
    assert not claim_refill_never_waits(CACHE, _once(SYCL, BOUNDARY_ARM, "queue->wait_and_throw(); " + BOUNDARY_ARM))


def test_mutant_refill_copies_from_a_raw_pointer_fails():
    mutant = _once_in(CACHE, REFILL_SIG, REFILL_SRC,
                      "src_owner[i] = make_copy_handle_for_raw_ptr(e.device_ptr, e.size);")
    assert not claim_refill_copy_is_owner_to_owner_on_the_copy_engine(mutant)


def test_mutant_refill_copies_on_the_compute_queue_fails():
    mutant = _once(CACHE, BCS_QUEUE, "sycl::queue & bq = get_queue();")
    assert not claim_refill_copy_is_owner_to_owner_on_the_copy_engine(mutant)


def test_mutant_refill_publishes_the_device_entry_ready_fails():
    mutant = _once_in(CACHE, REFILL_SIG, ENTRY_EVENTS[0], "entry.state = cache_entry_state::READY;")
    assert not claim_refill_copy_is_owner_to_owner_on_the_copy_engine(mutant)


def test_mutant_refill_copy_ignores_the_host_writes_fails():
    mutant = _once(CACHE, REFILL_COPY, REFILL_COPY.replace(", bq, deps[i]);", ", bq, {});"))
    assert not claim_refill_copy_is_owner_to_owner_on_the_copy_engine(mutant)


def test_mutant_refill_mints_after_the_swap_fails():
    mutant = _once(CACHE, SWAP_MIRROR,
                   SWAP_MIRROR + ' (void) make_direct_entry_handle(key, cache_device, stored, "x");')
    assert not claim_refill_allocates_before_it_changes_anything(mutant)


def test_mutant_refill_builds_a_mapping_node_after_the_swap_fails():
    mutant = _once(CACHE, REMAP_ID, "id_to_key_[key] = make_direct_stage_key(cache_entry_type::MOE_EXPERT, key, "
                                    "GGML_LAYOUT_AOS);")
    assert not claim_refill_allocates_before_it_changes_anything(mutant)


def test_mutant_refill_sizes_the_nodes_under_the_locks_fails():
    sized = _once_in(CACHE, REFILL_SIG, RELEASED_SIZED, "std::vector<expert_retire_nodes> host_nodes;")
    mutant = _once_in(sized, REFILL_SIG, EMPLACE, RELEASED_SIZED + " " + EMPLACE)
    assert not claim_refill_allocates_before_it_changes_anything(mutant)


def test_mutant_refill_catch_keeps_the_inserted_entries_fails():
    mutant = _once(CACHE, WITHDRAW_INSERTED, "for (int i = 0; i < 0; ++i) { (void) transition_to_retired_locked(")
    assert not claim_refill_allocates_before_it_changes_anything(mutant)


def test_mutant_refill_publishes_outside_a_planned_materialization_fails():
    assert not claim_refill_publishes_in_a_planned_materialization(_once(CACHE, REFILL_SESSION, ";"))


def test_mutant_refill_opens_its_session_after_the_locks_fails():
    moved = _once(CACHE, REFILL_SESSION, ";")
    mutant = _once_in(moved, REFILL_SIG, EMPLACE, REFILL_SESSION + " " + EMPLACE)
    assert not claim_refill_publishes_in_a_planned_materialization(mutant)


def test_mutant_retired_host_entry_shadows_the_refill_fails():
    mutant = _once(CACHE, RESOLVE_PAST_RETIRED, "if (entry_it == entries_.end()) {")
    assert not claim_a_retired_host_entry_does_not_shadow_the_refill(mutant)


def test_mutant_resolve_takes_a_missing_mapping_fails():
    mutant = _once(CACHE, RESOLVE_MAPPED, "entry_it = entries_.find(id_it->second);")
    assert not claim_a_retired_host_entry_does_not_shadow_the_refill(mutant)


def test_mutant_refill_uses_the_id_wide_retire_fails():
    mutant = _once(CACHE, HOST_RETIRE, "(void) retire_expert_entry_exact(id, GGML_LAYOUT_AOS, \"refill\");")
    assert not claim_refill_retires_the_host_copy_exactly(mutant)


def test_mutant_refill_retires_whatever_the_host_id_names_fails():
    mutant = _once(CACHE, HOST_RETIRE, HOST_RETIRE.replace(
        " && host->second.direct_alloc_owner.owner_control_id() == src_control[i]", ""))
    assert not claim_refill_retires_the_host_copy_exactly(mutant)


def test_mutant_refill_skips_the_device_entry_check_fails():
    mutant = _once(CACHE, 'refusal = "device entry exists";', ";")
    assert not claim_refill_checks_before_copying(mutant)


def test_mutant_refill_skips_the_coherence_check_fails():
    mutant = _once_in(CACHE, REFILL_SIG, COHERENCE, ";")
    assert not claim_refill_checks_before_copying(mutant)


def test_mutant_refill_refuses_after_the_copy_fails():
    early = _once_in(CACHE, REFILL_SIG, REFUSE_EARLY, ";")
    mutant = _once(early, REFILL_COPY, REFILL_COPY + " " + REFUSE_EARLY)
    assert not claim_refill_checks_before_copying(mutant)


def test_mutant_refill_allocates_under_the_inventory_lock_fails():
    moved = _once_in(SYCL, TXN_SIG, DEVICE_ALLOC, "ggml_sycl::mem_handle storage;")
    mutant = _once_in(moved, TXN_SIG, INVENTORY_LOCK, INVENTORY_LOCK + " " + DEVICE_ALLOC)
    assert not claim_refill_lock_order(CACHE, mutant)


def test_mutant_refill_cache_takes_the_inventory_mutex_fails():
    mutant = _once(CACHE, REFILL_LOAD_EFFECT, "std::lock_guard<std::mutex> inv(g_tensor_inventory_mutex); "
                   + REFILL_LOAD_EFFECT)
    assert not claim_refill_lock_order(mutant, SYCL)


def test_mutant_refill_boundary_takes_the_graph_mutex_fails():
    mutant = _once(SYCL, BOUNDARY_ARM, "std::lock_guard<std::mutex> g(g_sycl_graph_compute_mutex); " + BOUNDARY_ARM)
    assert not claim_refill_lock_order(CACHE, mutant)


def test_mutant_refill_keeps_the_host_lease_fails():
    mutant = _once(SYCL, FORGET_HOST, ";")
    assert not claim_refill_flips_back_and_forgets_the_host_copy(mutant)


def test_mutant_refill_leaves_the_plan_on_the_host_fails():
    mutant = _once(SYCL, FLIP, "(void) next_plan->update_expert_placement(k.layer_id, k.expert_id, k.role, false, -1);")
    assert not claim_refill_flips_back_and_forgets_the_host_copy(mutant)


def test_mutant_refill_keeps_the_triplet_in_the_ledger_fails():
    mutant = _once(SYCL, POP_READMITTED, POP_READMITTED.replace("if (refill) { ledger.triplets.pop_back(); } else ",
                                                                "if (!refill) "))
    assert not claim_refill_flips_back_and_forgets_the_host_copy(mutant)


def test_mutant_loan_leaves_the_cursor_armed_fails():
    assert not claim_a_loan_cancels_the_refills(_once(SYCL, DISARM, "if (!refill && lent != 0) { }"))


def test_mutant_selection_ignores_the_cursor_fails():
    mutant = _once(SYCL, SELECT_REMAINING, "*remaining = static_cast<uint32_t>(ledger.triplets.size());")
    assert not claim_a_loan_cancels_the_refills(mutant)


def test_mutant_refill_ignores_a_loan_between_select_and_lock_fails():
    assert not claim_a_loan_cancels_the_refills(_once(SYCL, CANCELLED, '"refill refused";'))


def test_mutant_arming_skips_the_inventory_lock_fails():
    assert not claim_ledger_is_guarded_by_the_inventory_mutex(_once_in(SYCL, ARM_SIG, LEDGER_LOCK, ";"))


def test_mutant_boundary_writes_the_ledger_unlocked_fails():
    mutant = _once(SYCL, BOUNDARY_ARM, "g_sycl_expert_lent[device].refill_armed = decode; " + BOUNDARY_ARM)
    assert not claim_ledger_is_guarded_by_the_inventory_mutex(mutant)


def test_mutant_refill_runs_before_the_deferred_frees_fails():
    sync_moved = _once(SYCL, RUN_REFILL, ";")
    mutant = _once_in(sync_moved, SYNC_SIG, DEFERRED_FREES, RUN_REFILL + " " + DEFERRED_FREES)
    assert not claim_refill_runs_at_the_decode_boundary(mutant)


def test_mutant_refill_boundary_ignores_the_phase_fails():
    mutant = _once_in(SYCL, BOUNDARY_SIG, "ctx->graph_phase_is_decode.load(std::memory_order_relaxed)", "true")
    assert not claim_refill_runs_at_the_decode_boundary(mutant)


def test_mutant_prompt_graph_leaves_the_cursor_armed_fails():
    mutant = _once(SYCL, BOUNDARY_ARM, "if (!decode) { return 0; } ggml_sycl_expert_refill_set_armed(device, true);")
    assert not claim_refill_runs_at_the_decode_boundary(mutant)


def test_mutant_mul_mat_id_runs_the_refill_fails():
    n = norm(SYCL)
    mmid = body(n, MMID_SIG)
    mutant = n.replace(mmid, mmid.replace("{", "{ ggml_sycl_run_expert_refill_at_boundary_on_device(0, true, 1, "
                                               "nullptr, nullptr);", 1), 1)
    assert not claim_refill_runs_at_the_decode_boundary(mutant)


def test_mutant_refill_env_read_twice_fails():
    mutant = _once(SYCL, BOUNDARY_ARM, 'if (std::getenv("GGML_SYCL_MOE_REFILL_PER_GRAPH")) { } ' + BOUNDARY_ARM)
    assert not claim_refill_env_read_once(mutant, [mutant])


def test_mutant_refill_env_read_every_call_fails():
    mutant = _once(SYCL, PER_GRAPH_ONCE, "const uint32_t per_graph = [] {")
    assert not claim_refill_env_read_once(mutant, [mutant])


def test_mutant_refill_env_unset_turns_it_off_fails():
    mutant = _once(SYCL, PER_GRAPH_DEFAULT, "if (!text || !*text) { return 0u; }")
    assert not claim_refill_env_read_once(mutant, [mutant])


def test_mutant_teardown_reprints_the_counts_fails():
    mutant = _once(SYCL, TAKE_LENT,
                   "const uint32_t lent = g_sycl_expert_lent_total[device].load(std::memory_order_acquire);")
    assert not claim_refill_warn_formats(mutant)


def test_mutant_free_skips_the_refill_line_fails():
    assert not claim_refill_warn_formats(_once(SYCL, FREE_REPORTS, "ggml_sycl_moe_exec_log_report();"))


def test_mutant_refill_body_warns_every_graph_fails():
    mutant = _once_in(SYCL, TXN_SIG, REFILL_CALL,
                      REFILL_CALL + ' GGML_LOG_WARN("[MOE-EXEC] refill device %d\\n", device);')
    assert not claim_refill_warn_formats(mutant)


def test_mutant_refill_warns_every_refusal_fails():
    mutant = _once(SYCL, REFILL_BUSY_WARN, REFILL_BUSY_WARN.replace("if ((n - 1) % warn_every == 0) ", "if (true) "))
    assert not claim_refill_warn_formats(mutant)


def test_mutant_refill_count_never_reset_fails():
    mutant = _once(SYCL, REFILL_RESET, REFILL_RESET.replace("streak.store(0, std::memory_order_relaxed); ", ""))
    assert not claim_refill_warn_formats(mutant)


def test_mutant_refill_resets_the_loan_busy_streak_fails():
    mutant = _once(SYCL, BOUNDARY_ARM, "g_sycl_expert_loan_busy_streak[device].store(0, std::memory_order_relaxed); "
                   + BOUNDARY_ARM)
    assert not claim_refill_warn_formats(mutant)


def test_mutant_renamed_refusal_turns_final_into_retried_fails():
    mutant = _once_in(CACHE, REFILL_SIG, 'refusal = "not lent";', 'refusal = "not on loan";')
    assert not claim_final_refusals_name_the_cache_refusals(mutant, SYCL)


def test_mutant_final_refusal_kept_in_the_ledger_fails():
    mutant = _once(SYCL, FINAL_POP, FINAL_POP.replace("if (!transient) { drop_back(stop); } ", ""))
    assert not claim_final_refusals_name_the_cache_refusals(CACHE, mutant)


# --- a loan without host zones is refused (lead ruling, llama.cpp-sif1) ----------------------------------------------


def test_a_loan_without_host_zones_is_refused():
    assert claim_a_loan_without_host_zones_is_refused(CACHE, SYCL)


def test_mutant_loan_without_host_zones_lends_fails():
    assert not claim_a_loan_without_host_zones_is_refused(CACHE, _once(SYCL, ZONES_REFUSAL, ""))


def test_mutant_loan_without_host_zones_is_reposted_fails():
    mutant = _once(SYCL, ZONES_REFUSAL, ZONES_REFUSAL.replace("::REFUSED,", "::BUSY,"))
    assert not claim_a_loan_without_host_zones_is_refused(CACHE, mutant)


def test_mutant_host_zones_checked_after_the_host_storage_fails():
    mutant = _once(_once(SYCL, ZONES_REFUSAL, ""), HOST_ALLOC, ZONES_REFUSAL + " " + HOST_ALLOC)
    assert not claim_a_loan_without_host_zones_is_refused(CACHE, mutant)


def test_mutant_probe_answers_before_the_host_zones_fails():
    mutant = _once(_once(SYCL, ZONES_REFUSAL, ""), PROBE, PROBE + " " + ZONES_REFUSAL)
    assert not claim_a_loan_without_host_zones_is_refused(CACHE, mutant)


def test_mutant_loan_asks_a_second_zone_query_fails():
    mutant = _once(SYCL, ZONES_REFUSAL, ZONES_REFUSAL.replace("!cache->host_zones_configured()",
                                                              "!cache->host_zone_capacity(host_zone_id::WEIGHT)"))
    assert not claim_a_loan_without_host_zones_is_refused(CACHE, mutant)


def test_mutant_pinned_pool_stops_branching_on_the_zones_fails():
    mutant = _once_in(CACHE, UNIFIED_ALLOC_SIG, PINNED_ZONE_BRANCH,
                      PINNED_ZONE_BRANCH.replace("!ucache->host_zones_configured()", "!ucache->arena_active()"))
    assert not claim_a_loan_without_host_zones_is_refused(mutant, SYCL)


# --- a lent triplet the refill can never prepare leaves the ledger, each drop WARNed, and one decode graph admits one
# --- refill (llama.cpp-sif1, quality review) --------------------------------------------------------------------------


def test_an_unpreparable_triplet_leaves_the_ledger():
    assert claim_an_unpreparable_triplet_leaves_the_ledger(SYCL)


def test_each_dropped_triplet_is_warned():
    assert claim_each_dropped_triplet_is_warned(SYCL)


def test_one_decode_graph_admits_one_refill():
    assert claim_one_decode_graph_admits_one_refill(SYCL, COMMON, sycl_sources_outside_tests())


def test_mutant_unpreparable_triplet_stops_every_later_refill_fails():
    """The early return the review found: an unpreparable back triplet returns before the lock, every boundary."""
    mutant = _once(SYCL, EARLY_REFUSAL, "if (prepared.empty()) {")
    assert not claim_an_unpreparable_triplet_leaves_the_ledger(mutant)


def test_mutant_resolver_failure_is_not_recorded_fails():
    mutant = _once(SYCL, UNPREPARABLE, UNPREPARABLE.replace("refill && !transient", "refill && transient"))
    assert not claim_an_unpreparable_triplet_leaves_the_ledger(mutant)


def test_mutant_unpreparable_triplet_dropped_from_behind_the_back_fails():
    mutant = _once(SYCL, DROP_AT_BACK, DROP_AT_BACK.replace(" && ledger.triplets.back() == unpreparable", ""))
    assert not claim_an_unpreparable_triplet_leaves_the_ledger(mutant)


def test_mutant_unpreparable_triplet_dropped_before_the_plan_check_fails():
    mutant = _once_in(_once(SYCL, DROP_AT_BACK, ""), TXN_SIG, INVENTORY_LOCK, INVENTORY_LOCK + " " + DROP_AT_BACK)
    assert not claim_an_unpreparable_triplet_leaves_the_ledger(mutant)


def test_mutant_no_weight_arena_returns_busy_fails():
    mutant = _once(SYCL, ARENA_TOLD_APART, ARENA_TOLD_APART.replace("transient = arena;", "transient = true;"))
    assert not claim_an_unpreparable_triplet_leaves_the_ledger(mutant)


def test_mutant_no_weight_arena_reads_as_a_full_zone_fails():
    mutant = _once(SYCL, ARENA_TOLD_APART, 'stop = "WEIGHT zone full"; transient = true; break;')
    assert not claim_an_unpreparable_triplet_leaves_the_ledger(mutant)


def test_mutant_ledger_count_stored_before_the_drop_fails():
    mutant = _once(_once_in(SYCL, TXN_SIG, LENT_COUNT_STORE, ""), DROP_AT_BACK, LENT_COUNT_STORE + " " + DROP_AT_BACK)
    assert not claim_an_unpreparable_triplet_leaves_the_ledger(mutant)


def test_mutant_drop_unwarned_when_nothing_moved_fails():
    mutant = _once(SYCL, "if (lent == 0) { " + WARN_AFTER_UNLOCK, "if (lent == 0) { lock.unlock();")
    assert not claim_each_dropped_triplet_is_warned(mutant)


def test_mutant_drop_unwarned_when_something_moved_fails():
    mutant = _once(SYCL, WARN_AFTER_UNLOCK + " if (!swapped) {", "lock.unlock(); if (!swapped) {")
    assert not claim_each_dropped_triplet_is_warned(mutant)


def test_mutant_drop_warned_under_the_lock_fails():
    mutant = _once(SYCL, DROP_BACK, DROP_BACK.replace("ledger.triplets.pop_back(); };",
                                                      "ledger.triplets.pop_back(); " + DROP_WARN + " };"))
    assert not claim_each_dropped_triplet_is_warned(mutant)


def test_mutant_drop_warned_once_per_call_fails():
    mutant = _once(SYCL, WARN_EACH, "for (size_t i = 0; i < n_dropped && i < 1; ++i) {")
    assert not claim_each_dropped_triplet_is_warned(mutant)


def test_mutant_drop_warn_filtered_fails():
    mutant = _once(SYCL, DROP_WARN, "if (n_dropped > 1) { " + DROP_WARN + " }")
    assert not claim_each_dropped_triplet_is_warned(mutant)


def test_mutant_final_refusal_popped_unrecorded_fails():
    mutant = _once(SYCL, FINAL_POP, FINAL_POP.replace("drop_back(stop);", "ledger.triplets.pop_back();"))
    assert not claim_each_dropped_triplet_is_warned(mutant)


def test_the_move_loop_moves_every_prepared_triplet():
    assert claim_the_move_loop_moves_every_prepared_triplet(SYCL)


def test_mutant_move_loop_stops_after_one_triplet_fails():
    """The build the two-triplet device case must catch: the loop stops after the first triplet that moved."""
    mutant = _once_in(SYCL, TXN_SIG, "++lent;", "++lent; break;")
    assert not claim_the_move_loop_moves_every_prepared_triplet(mutant)


def test_mutant_refill_stops_after_one_readmitted_triplet_fails():
    mutant = _once_in(SYCL, TXN_SIG, POP_READMITTED, POP_READMITTED + " if (refill) { ++lent; break; }")
    assert not claim_the_move_loop_moves_every_prepared_triplet(mutant)


def test_mutant_refill_at_every_synchronize_fails():
    """The review's finding: without the count, every synchronize after a decode graph re-admits again."""
    mutant = _once(SYCL, ONCE_PER_GRAPH, "if (false) { *result = ggml_sycl_expert_loan_result{};")
    assert not claim_one_decode_graph_admits_one_refill(mutant, COMMON, [mutant, COMMON])


def test_mutant_every_graph_counts_as_decode_fails():
    mutant = _once(SYCL, DECODE_BUMP,
                   "sycl_ctx->decode_graph_count.fetch_add(1, std::memory_order_relaxed); if (cached_is_decode) {")
    assert not claim_one_decode_graph_admits_one_refill(mutant, COMMON, [mutant, COMMON])


def test_mutant_boundary_counts_a_graph_itself_fails():
    mutant = _once_in(SYCL, BOUNDARY_SIG, BOUNDARY_PHASE,
                      BOUNDARY_PHASE + " ctx->decode_graph_count.fetch_add(1, std::memory_order_relaxed);")
    assert not claim_one_decode_graph_admits_one_refill(mutant, COMMON, [mutant, COMMON])


def test_mutant_refill_mark_is_not_the_contexts_fails():
    mutant = _once(COMMON, REFILL_MARK, "")
    assert not claim_one_decode_graph_admits_one_refill(SYCL, mutant, [SYCL, mutant])


def test_mutant_refill_runs_before_the_count_is_checked_fails():
    run = ("const uint32_t remaining = ggml_sycl_expert_refill_transaction_run(device, max_triplets, resolver, result, "
           "&code);")
    mutant = _once_in(SYCL, BOUNDARY_DEVICE_SIG, run, "")
    mutant = _once_in(mutant, BOUNDARY_DEVICE_SIG, ONCE_PER_GRAPH, run + " " + ONCE_PER_GRAPH)
    assert not claim_one_decode_graph_admits_one_refill(mutant, COMMON, [mutant, COMMON])


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
