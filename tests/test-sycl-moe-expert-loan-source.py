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
    no get_info<, no event_complete(, not finalize_retired_entries_locked(, which asks, and no
    drain_retained_handles(; the one sanctioned wait on the loan's path is the boundary's (below), not the loan's;
  - the device free is gated on the copy event, with no host_task: each D2H copy is submitted on the DMA queue reading
    the entry's own allocation owner; the transaction then re-checks each device entry and mints the lease that holds
    it in place, all three before any entry changes, since minting allocates; then each device entry's last write
    becomes its copy (has_write_event stamped from the copy, and the copy ran behind the fill), so
    release_entry_allocation_locked queues the free behind the copy; then it retires each entry exactly, registers
    the host copy behind its copy event, and only then hands the leases to the transaction (a rollback drops them);
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
  - an existing host entry is never replaced: any entry under a host id, idle or not, refuses the triplet, as
    "host entry retiring" while that entry is retired and waits for the deferred-free pass and as "host entry exists"
    while it is live, the registration is asked not to replace one and refuses when it would, and the refusal reason
    reaches the transaction's refusal WARN;
  - g_sycl_graph_compute_mutex is never named in the loan code;
  - lock order: the host storage is minted before g_tensor_inventory_mutex is taken, and the cache is asked to lend
    under it; the cache takes direct_stage_mutex_ and rw_mutex_ together, with std::lock;
  - the loan runs at the graph boundary, never inside a graph: an in-graph caller records a request with a lock-free
    compare-and-swap; ggml_backend_sycl_synchronize() runs the pending loan after it clears the graph's compute flag
    and before the deferred-free pass; the transaction refuses "graph active"; and mul_mat_id names no loan entry;
  - the boundary that accepted a loan drains the loan's retained records (llama.cpp-6aj4, c-769m): the runner
    reports an accepted loan that lent, and synchronize then calls ggml_sycl_expert_loan_boundary_drain, after the
    runner and before the deferred-free pass, so that pass erases the withdrawn device entries and returns their room
    before the next graph's reserve. It is the one sanctioned wait (synchronize has just waited the whole stream): it
    runs only for an accepted loan, waits at most 100 ms on drain_retained_handles (the whole store; the call takes no
    scope), and on a timeout leaves the retention standing, the erase one pass late, with one trace line;
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
  - each extra drops every view it holds of a withdrawn device entry before it forgets the entry's record and
    remembers the host copy (llama.cpp-6aj4): ggml_sycl_drop_lent_expert_views matches each view by the key the loan
    retired the entry under and never resolves one, since a retired entry refuses a resolve once the cache generation
    moves; it keeps a lease on every view it takes, and the transaction retains those leases until the device's
    execution queue passes a barrier, after the loop and before the publication, so no kernel still reading a dropped
    table sees a freed buffer; and every drop leaves the device table stale (the slot's payload nulled, the storage
    generation bumped, and the table marked stale through mark_moe_device_table_stale, whose marks are pinned, behind
    its device guard as its first statement): a dropped pointer-table lease, route table or hybrid record slot each
    sets the flag that marks it, so a drop of the table's views alone still does;
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

Both directions size their copies from the expert's tensor, never from a cache entry. Direct staging pads an AOS
quantized expert's tail to MATRIX_ROW_PADDING columns and the plan's dst_size does not count it, so an entry's size can
be larger than the expert and dst_size smaller than its staged size. What this gate pins:
  - the preparation asks the cache only whether the entry to copy from is there (the AOS device entry for a loan, the
    lent host copy for a refill), stops with the refusal the cache would give when there is none, and mints a loan's
    host copy at the tensor's bytes per expert and a refill's device storage at the staged size the layout gives
    those bytes, never from an entry's size or from dst_size;
  - the query reads its entry under direct_stage_mutex_ and rw_mutex_, shared, taken together with std::lock, and
    returns only whether the entry is there, not retired;
  - the cache's checks stay as the guards: a loan's device entry or host copy shorter than the expert, and a refill's
    host copy shorter than the expert or device storage not exactly the staged size, are still refused. So an expert
    that goes to the host and back does not grow.

Every claim is checked on COMMENT-STRIPPED, whitespace-normalized text with adjacent string literals joined, where a
guard on g_tensor_inventory_mutex reads the same with or without its template argument, and has a mutant that must make
it fail.

Pytest-style (module-level test_* functions): register with llama_test_pytest.
"""

import functools
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_LEXEME_RE = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'",
    re.DOTALL,
)


# The mutant tests below each rebuild the claims on a copy of one source with one edit. The unmutated sources stay
# cached because every mutant reads them; a mutated text is used for its one claim and evicted. The caches are small on
# purpose: a key holds a whole source, so a large cache would keep hundreds of dead mutants in memory. A hit returns
# what the same text gave before, so each mutant is still judged on its own text.
_WHOLE_SOURCE = 1 << 20  # a text this long is a whole source or a whole stripped copy of one, not a body cut from it


def content_cache(fn):
    """Cache `fn` by its text argument in two tiers: the few whole-source texts stay hot, and the many short texts cut
    from them are held in a larger cache whose keys are each under a megabyte."""
    short = functools.lru_cache(maxsize=256)(fn)
    whole = functools.lru_cache(maxsize=8)(fn)

    @functools.wraps(fn)
    def cached(text):
        return (whole if len(text) >= _WHOLE_SOURCE else short)(text)

    return cached


@functools.lru_cache(maxsize=8)
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


# A guard on g_tensor_inventory_mutex with its template argument spelled out, which norm reads in the deduced spelling
# the claims name: they pin that the lock is taken, not how its type is spelled, as the other gates read it.
_INVENTORY_GUARD_RE = re.compile(r"std::(lock_guard|unique_lock) ?<[^<>;]*> ?(\w+)\(g_tensor_inventory_mutex\)")


@content_cache
def norm(text: str) -> str:
    """Comment-stripped, whitespace-canonical text with adjacent string literals joined, so neither a clang-format
    re-wrap nor a re-split format string can turn a claim red, and every guard on g_tensor_inventory_mutex in its
    deduced spelling, so neither can a guard that names its template argument."""
    t = re.sub(r"\s+", " ", strip_comments(text))
    t = re.sub(r"\( ", "(", t)
    t = re.sub(r" \)", ")", t)
    t = re.sub(r" ,", ",", t)
    t = re.sub(r",(?! )", ", ", t)
    t = re.sub(r'(?<!\\)" "', "", t)
    t = _INVENTORY_GUARD_RE.sub(r"std::\1 \2(g_tensor_inventory_mutex)", t)
    return t


def read(rel: str) -> str:
    return (ROOT / rel).read_text()


CACHE = read("ggml/src/ggml-sycl/unified-cache.cpp")
SYCL = read("ggml/src/ggml-sycl/ggml-sycl.cpp")
SYCL_CMAKE = read("ggml/src/ggml-sycl/CMakeLists.txt")
COMMON = read("ggml/src/ggml-sycl/common.hpp")
PLAN_HPP = read("ggml/src/ggml-sycl/moe-stream-plan.hpp")
ROUTE_HPP = read("ggml/src/ggml-sycl/moe-route-table.hpp")


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
REQUEST_SIG = "static void ggml_sycl_request_expert_loan("
PENDING_DEVICE_SIG = "static bool ggml_sycl_run_pending_expert_loan_on_device("
PENDING_SIG = "static bool ggml_sycl_run_pending_expert_loan("
SYNC_SIG = "static void ggml_backend_sycl_synchronize(ggml_backend_t backend)"
MMID_SIG = "static void ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx, ggml_tensor * dst)"
DISPATCH_SIG = "auto dispatch_cpu_entries_now = [&](const std::vector<expert_dispatch_entry> & entries)"


def loan_bodies(cache: str, sycl: str) -> list[str]:
    """The loan code, cache side and backend side. A missing body is an empty string, which every claim that reads it
    treats as a failure."""
    c = norm(cache)
    s = norm(sycl)
    return [body(c, LOAN_SIG), body(c, EXACT_SIG), body(c, RESTORE_SIG), body(s, TXN_SIG), body(s, LOAN_TXN_SIG),
            body(s, RESOLVE_SIG), body(s, REQUEST_SIG), body(s, PENDING_DEVICE_SIG), body(s, PENDING_SIG)]


# --- waits and status queries ---------------------------------------------------------------------------------------

# The loan bodies never wait or ask (c-3bde): they run on inference paths. The one sanctioned wait on the loan's path is
# ggml_sycl_expert_loan_boundary_drain, which synchronize calls after the runner (claim_loan_boundary_drains_its_room);
# it is not a loan body, and draining the retained store from one is forbidden here like any other wait.
EVENT_WAITS = (".wait(", "wait_and_throw(", "get_info<", "event_complete(", "finalize_retired_entries_locked(")
FORBIDDEN_WAITS = EVENT_WAITS + ("drain_retained_handles(",)


def claim_loan_never_waits(cache: str, sycl: str) -> bool:
    bodies = loan_bodies(cache, sycl)
    return all(bodies) and not any(token in b for b in bodies for token in FORBIDDEN_WAITS)


# --- the device free is gated on the copy event ---------------------------------------------------------------------

SRC_IS_OWNER = "src_owner[i] = e.direct_alloc_owner;"
COPY = "out.copy_events[i] = mem_copy_async(members[i].owner, src_owner[i], bytes[i], dq, deps[i]);"
DMA_QUEUE = "sycl::queue & dq = get_dma_queue();"
DEPS = ("if (e.has_write_event) { deps[i].push_back(e.last_write_event); }",
        "if (e.has_ready_event) { deps[i].push_back(e.ready_event); }")
STAMP_EVENT = "device_entry[i]->last_write_event = out.copy_events[i];"
STAMP_FLAG = "device_entry[i]->has_write_event = true;"
HOLD = 'held[i] = make_direct_entry_handle(device_key, cache_device, *device_entry[i], "expert_loan/held");'
ENTRY_CHANGED = '{ refusal = "entry changed"; break; }'
DEVICE_RETIRE = "out.retire_status[i] = retire_expert_entry_at_locked(device_key, &device_nodes[i]);"
REGISTER = ("register_host_expert_owned(m.host_keys[k], host_ptr, bytes[i], GGML_LAYOUT_AOS, "
            "k == 0 ? &out.handles[i] : nullptr, m.owner, &out.copy_events[i], false, &out.device_keys[i])")
DROP_HELD = "for (std::shared_ptr<mem_handle> & lease : held) { lease.reset(); }"
HAND_HELD = "for (int i = 0; i < 3; ++i) { out.retired_leases[i] = std::move(held[i]); }"


def claim_device_free_gated_on_the_copy(cache: str, sycl: str) -> bool:
    b = body(norm(cache), LOAN_SIG)
    bodies = loan_bodies(cache, sycl)
    once = (SRC_IS_OWNER, COPY, STAMP_EVENT, STAMP_FLAG, HOLD, DEVICE_RETIRE, REGISTER)
    return (all(bodies) and not any("host_task" in x for x in bodies) and all(b.count(x) == 1 for x in once)
            and all(d in b for d in DEPS) and b.count(DROP_HELD) == 1 and b.count(HAND_HELD) == 1
            and b.count(ENTRY_CHANGED) == 1
            and ordered(b, SRC_IS_OWNER, DMA_QUEUE, COPY, ENTRY_CHANGED, HOLD, STAMP_EVENT, STAMP_FLAG, DEVICE_RETIRE,
                        REGISTER)
            and b.find(DROP_HELD) > b.find(REGISTER) and b.find(HAND_HELD) > b.find(REGISTER))


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
RESTORE_CALL = "restore_expert_entry_at_locked(expert_device_key(members[i].key), device_nodes[i]);"
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
HOST_EXISTS = ("const auto host = entries_.find(expert_host_key(host_id)); "
               "if (host != entries_.end()) { "
               'refusal = host->second.retired ? "host entry retiring" : "host entry exists"; break; }')
REFUSE_EARLY = "if (refusal) { out.reason = refusal; return false; }"


def claim_registration_checked_before_the_copies(cache: str) -> bool:
    b = body(norm(cache), LOAN_SIG)
    return (all(b.count(x) == 1 for x in (LOAD_EFFECT, COHERENCE, HOST_EXISTS, REFUSE_EARLY))
            and ordered(b, LOAD_EFFECT, COHERENCE, HOST_EXISTS, REFUSE_EARLY, COPY))


# --- an existing host entry is never replaced (c-mwmr-3) ------------------------------------------------------------

OWNED_SIG = "bool unified_cache::register_host_expert_owned("
REG_BODY_SIG = ("const mem_handle & typed_owner, const sycl::event * ready_event, bool replace_existing, "
                "const unified_cache_key * moved_from)")
NO_REPLACE = "if (old != entries_.end()) { if (!replace_existing) { return false; }"
REPLACE = "erase_entry_locked(old);"
FORWARD = "ready_event, replace_existing, moved_from);"
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

# Each triplet's storage is minted in the preparation, which the transaction runs for every triplet before it takes
# the inventory lock; the preparation never names that lock, and the transaction mints nothing itself.
PREP_SIG = "static const char * ggml_sycl_expert_prepare_triplet("
PREPARE = ("stop = ggml_sycl_expert_prepare_triplet(*cache, device, *current->plan, &keys[3 * t], *resolver, triplet, "
           "&transient);")
HOST_ALLOC = "storage = ggml_sycl_host_expert_alloc(device, target.logical_bytes);"
MINT_CALLS = ("ggml_sycl_host_expert_alloc(", "ggml_sycl_device_expert_alloc(")
INVENTORY_LOCK = "std::unique_lock lock(g_tensor_inventory_mutex);"
# One statement moves each triplet, the cache body picked by a switch on its kind with no default, so a new kind is a
# compile warning rather than a move through another kind's body.
MOVE_CALL = ("switch (triplet.kind) { case ggml_sycl_expert_move_kind::LOAN: case ggml_sycl_expert_move_kind::EVICT: "
             "loaned = cache->loan_expert_triplet(triplet.members, record); break; "
             "case ggml_sycl_expert_move_kind::ADMIT: loaned = cache->admit_host_triplet(triplet.members, record); "
             "break; case ggml_sycl_expert_move_kind::REFILL: "
             "loaned = cache->refill_lent_triplet(triplet.members, record); break; }")
LEND = "cache->loan_expert_triplet(triplet.members, record)"
CACHE_LOCKS = ("std::unique_lock<std::shared_mutex> direct_lock(direct_stage_mutex_, std::defer_lock); "
               "std::unique_lock<std::shared_mutex> cache_lock(rw_mutex_, std::defer_lock); "
               "std::lock(direct_lock, cache_lock);")


def claim_storage_minted_before_the_inventory_lock(sycl: str) -> bool:
    """Minting can allocate and wait on the pool; under the inventory lock it would hold every other expert move and
    every tensor lookup behind it. So the preparation, which mints, runs before the lock and never names it, and the
    transaction itself mints nothing."""
    s = norm(sycl)
    txn = body(s, TXN_SIG)
    prep = body(s, PREP_SIG)
    return (bool(prep) and txn.count(PREPARE) == 1 and not any(m in txn for m in MINT_CALLS)
            and "g_tensor_inventory_mutex" not in prep and txn.find("g_tensor_inventory_mutex") > txn.find(PREPARE))


def claim_lock_order(cache: str, sycl: str) -> bool:
    txn = body(norm(sycl), TXN_SIG)
    prep = body(norm(sycl), PREP_SIG)
    loan = body(norm(cache), LOAN_SIG)
    return (txn.count(INVENTORY_LOCK) == 1 and prep.count(HOST_ALLOC) == 1 and txn.count(MOVE_CALL) == 1
            and claim_storage_minted_before_the_inventory_lock(sycl) and ordered(txn, PREPARE, INVENTORY_LOCK, LEND)
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
PENDING_RUN = ("const ggml_sycl_txn_result code = ggml_sycl_expert_loan_transaction_run(device, bytes_wanted, "
               "slot_bytes, false, resolver, result, false);")
NOT_BUSY_RETURNS = ("if (code != ggml_sycl_txn_result::BUSY) { busy_streak.store(0, std::memory_order_relaxed); "
                    "return true; }")
REPOST = "ggml_sycl_request_expert_loan(device, bytes_wanted, slot_bytes); return true; }"
MMID_FORBIDDEN = ("ggml_sycl_expert_loan_transaction_run(", "ggml_sycl_run_pending_expert_loan", "loan_expert_triplet(")


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
REFUSE_GATE = "if (!probe_mode && !refill && !named && (warn_busy || code != ggml_sycl_txn_result::BUSY)) { " + \
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


# The boundary that accepted a loan drains the loan's retained records before the deferred-free pass (llama.cpp-6aj4,
# c-769m): the copies' owners and the dropped views go, so that pass erases the withdrawn device entries and
# returns their room before the next graph's reserve. The runner reports only an accepted loan that lent, so a decode
# boundary pays nothing; the wait is bounded at 100 ms, and a timeout leaves the retention to a later pass with one
# trace line. The drain asks no event itself.
LOAN_DRAIN_SIG = "static bool ggml_sycl_expert_loan_boundary_drain(int device, bool loan_accepted)"
RUN_PENDING_ACCEPTED = "const bool loan_accepted = ggml_sycl_run_pending_expert_loan(backend);"
LOAN_DRAIN_CALL = "(void) ggml_sycl_expert_loan_boundary_drain(sycl_ctx->device, loan_accepted);"
PENDING_LENT = ("return ggml_sycl_run_pending_expert_loan_on_device(ctx->device, nullptr, &result) && "
                "result.accepted && result.n_triplets != 0;")
LOAN_DRAIN_BOUND = "constexpr uint32_t drain_ms = 100;"
LOAN_DRAIN_ONLY_ACCEPTED = "if (!loan_accepted || ggml_sycl::drain_retained_handles(true, drain_ms)) { return true; }"
LOAN_DRAIN_TRACE = ('if (ggml_sycl_moe_exec_config().trace >= 1) { GGML_LOG_INFO("[MOE-EXEC-TRACE] loan device %d: '
                    'room pending, boundary drain timed out\\n", device); } return false;')


def claim_loan_boundary_drains_its_room(sycl: str) -> bool:
    s = norm(sycl)
    sync = body(s, SYNC_SIG)
    drain = body(s, LOAN_DRAIN_SIG)
    return (s.count(LOAN_DRAIN_CALL) == 1 and sync.count(LOAN_DRAIN_CALL) == 1
            and ordered(sync, SET_INACTIVE, RUN_PENDING_ACCEPTED, LOAN_DRAIN_CALL, DEFERRED_FREES, RUN_REFILL)
            and body(s, PENDING_SIG).count(PENDING_LENT) == 1
            and drain.count("drain_retained_handles(") == 1
            and ordered(drain, LOAN_DRAIN_BOUND, LOAN_DRAIN_ONLY_ACCEPTED, LOAN_DRAIN_TRACE)
            and not any(token in drain for token in EVENT_WAITS))


# --- a loan without host zones is refused (lead ruling, llama.cpp-sif1) ----------------------------------------------

INVALID_ARGS = 'return refuse(ggml_sycl_txn_result::REFUSED, "invalid arguments"); }'
ZONES_REFUSAL = ('if (!refill && !cache->host_zones_configured()) { '
                 'return refuse(ggml_sycl_txn_result::REFUSED, "host zones not configured"); }')
LOAN_SELECT = ("ggml_sycl::moe_expert_loan_select(*current->plan, device, amount, slot_bytes, &selected, "
               "&slots_funded);")
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
            and ordered(txn, INVALID_ARGS, ZONES_REFUSAL, GRAPH_ACTIVE, LOAN_SELECT, PROBE, LEASE, PREPARE,
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
               "{ refusal = MOE_EXPERT_MOVE_SHARED_STORAGE; break; }")
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
         'result.slots_funded = slots_funded; result.n_triplets = n_selected; '
         'return n_selected != 0 ? finish(ggml_sycl_txn_result::ACCEPTED, "OK") : '
         'finish(ggml_sycl_txn_result::REFUSED, "nothing lendable"); }')
LEASE = "registry.acquire_live_update(current_token)"
FORGET = "(void) extra->forget_moe_storage_handle_on_device(k.expert_id, GGML_LAYOUT_AOS, device);"
DROP = ("ggml_sycl_drop_lent_expert_views(extra, device, k.expert_id, record.device_keys[r], lent_views, "
        "view_scan_fallbacks);")
# Every direction remembers each new copy with the tensor's bytes per expert as its range, never a copy's size.
REMEMBER = ("(void) triplet.extras[r]->remember_moe_storage_handle(triplet.members[r].placement.expert_id, "
            "GGML_LAYOUT_AOS, std::move(record.handles[r]), &record.copy_events[r], 0, "
            "triplet.members[r].logical_bytes);")
# A triplet goes to the device (a refill or an admit) or to the host (a loan or an evict); the flip follows that direction.
FLIP = ("(void) next_plan->update_expert_placement(k.layer_id, k.expert_id, k.role, to_device, "
        "to_device ? device : -1);")
# A lent member's host bytes are the pinned copy the transaction minted; one moved to the device is staged from the
# load's own source again, the mapping when the load maps files. The ring's plan index reads the backing.
BACKING = ("(void) next_plan->set_expert_host_backing(k.layer_id, k.expert_id, k.role, "
           "to_device && next_plan->host_backing_maps_files ? ggml_sycl::placement_host_backing::FILE_MAPPED : "
           "ggml_sycl::placement_host_backing::PINNED);")
PUBLISH = "ggml_sycl_publish_prepared_plan_locked(prepared_publication);"
EPOCH = "ggml_sycl_optional_layouts_retired();"
UNLOCK = "} lock.unlock(); warn_dropped(); if (!swapped) {"


def claim_probe_changes_nothing_and_a_loan_publishes_once(sycl: str) -> bool:
    b = body(norm(sycl), TXN_SIG)
    return (b.count(PROBE) == 1 and b.count(PUBLISH) == 1 and b.count(FLIP) == 1 and b.count(BACKING) == 1
            and ordered(b, PROBE, LEASE, PREPARE, INVENTORY_LOCK, LEND, DROP, FORGET, REMEMBER, FLIP, BACKING,
                        PUBLISH,
                        EPOCH, UNLOCK))


# --- a loan drops the extras' views of each withdrawn device entry by its key (llama.cpp-6aj4) -----------------------

DROP_SIG = "static void ggml_sycl_drop_lent_expert_views("
DROP_MATCH = "return h.is_weight() && h.device() == device && h.key() == withdrawn;"


def claim_lent_views_dropped_by_key(sycl: str) -> bool:
    b = body(norm(sycl), DROP_SIG)
    return b.count(DROP_MATCH) == 1 and "resolve(" not in b


# Each view the helper takes off an extra keeps a lease in `retained` before it goes, and the transaction keeps every
# such lease until the device's execution queue passes a barrier, after the loop: a kernel still reading a dropped
# table never sees a freed buffer, wherever the loan runs.
RETAIN_RECORD = "retained.push_back(record.handle);"
RETAIN_SLOT = "retained.push_back(std::move(w.moe_expert_handles[device][slot]));"
# The pointer table's leases on the entry come out of its keyed bag by the entry's key, into `retained`.
LEASE_TAKE = "w.moe_expert_ptrs_leases[device].take_key(ggml_sycl::moe_lease_key{ device, withdrawn }, retained)"
RETAIN_LEASE = LEASE_TAKE
# A route table that leases the entry is invalidated and every lease it held moves into `retained`.
RETAIN_ROUTES = "if (held.lease.valid()) { retained.push_back(std::move(held.lease)); }"
# The hybrid decode record keeps its slots that do not lease the entry: the drop withdraws only those that do, from
# a copy it publishes in the record's place, and their leases go to `retained` with the other views'.
RETAIN_HYBRID = ("ggml_sycl_moe_hybrid_decode_record_withdraw(w.moe_hybrid_decode[device], device, expert_id, "
                 "withdrawn, retained, emptied, scan_fallbacks)")
RETAIN_VIEWS = ("ggml_sycl::retain_handles_until_event(std::move(lent_views), "
                "queue ? queue->ext_oneapi_submit_barrier() : sycl::event{});")


def claim_lent_views_retained_to_a_barrier(sycl: str) -> bool:
    n = norm(sycl)
    helper = body(n, DROP_SIG)
    txn = body(n, TXN_SIG)
    return (all(helper.count(x) == 1 for x in (RETAIN_RECORD, RETAIN_SLOT, RETAIN_LEASE, RETAIN_ROUTES, RETAIN_HYBRID))
            and ordered(helper, RETAIN_RECORD, "list.erase(", RETAIN_SLOT, RETAIN_LEASE, RETAIN_ROUTES,
                        "routes.invalidate();", RETAIN_HYBRID)
            and txn.count(RETAIN_VIEWS) == 1
            and ordered(txn, LEND, DROP, REMEMBER, BREAK_ESCAPED, RETAIN_VIEWS, PUBLISH))


# Dropping a view leaves the device table stale, so neither the full-local fast path nor the transient table trusts a
# raw pointer to the withdrawn entry once the zone hands its room to the ring: the slot's kernel payload is nulled with
# the slot, a dropped storage record or slot bumps the storage generation, and any drop marks the table stale through
# the one member that does it for every site (its four marks pinned in common.hpp).
STALE_SIG = "void mark_moe_device_table_stale(int dev)"
STALE_MARKS = ("moe_device_table_valid[dev] = false;", "moe_full_local_probe_generation[dev] = 0;",
               "moe_full_local_probe_layout[dev] = GGML_LAYOUT_AOS;", "moe_full_local_probe_ok[dev] = false;")
NULL_PAYLOAD = "w.moe_expert_ptr_payload[device][slot] = nullptr;"
# Each expert whose record slot the drop emptied, and the one whose storage record or table slot it dropped.
BUMP_GENERATION = ("if (stored && std::find(emptied.begin(), emptied.end(), expert_id) == emptied.end()) { "
                   "emptied.push_back(expert_id); } for (const int32_t e : emptied) { "
                   "w.note_moe_storage_change(e); }")
MARK_STALE = "if (stored || table_dropped) { w.mark_moe_device_table_stale(device); }"
# The flag behind a drop of the table's views alone: a pointer-table lease erased (the holder a prompt graph's preload
# leaves, with no storage record and no slot naming the entry), the route table invalidated, or a hybrid record slot
# withdrawn. Without it a table whose only view of the entry was its leases stays valid, its raw pointer into room the
# ring reuses.
# The lease flag is set from how many leases the keyed take moved: any, and the table is stale.
TABLE_FROM_LEASES = "table_dropped = " + LEASE_TAKE + " != 0;"
TABLE_FROM_ROUTES = "routes.invalidate(); table_dropped = true; }"
TABLE_FROM_HYBRID = "if (" + RETAIN_HYBRID + ") { table_dropped = true; }"
# The member's guard is its first statement: inverted, every stale mark is a no-op on a valid device.
STALE_GUARD = "{ if (!ggml_sycl_valid_device_index(dev)) { return; }"


def claim_lent_views_mark_the_table_stale(sycl: str, common: str) -> bool:
    helper = body(norm(sycl), DROP_SIG)
    member = body(norm(common), STALE_SIG)
    return (member.startswith(STALE_GUARD) and all(member.count(x) == 1 for x in STALE_MARKS)
            and helper.count(NULL_PAYLOAD) == 1 and ordered(helper, RETAIN_SLOT, NULL_PAYLOAD, "stored = true;")
            and helper.count(BUMP_GENERATION) == 1 and helper.count(MARK_STALE) == 1
            and all(helper.count(x) == 1 for x in (TABLE_FROM_LEASES, TABLE_FROM_ROUTES, TABLE_FROM_HYBRID))
            and helper.count("table_dropped = true;") == 2
            and ordered(helper, TABLE_FROM_LEASES, RETAIN_ROUTES, TABLE_FROM_ROUTES, TABLE_FROM_HYBRID,
                        BUMP_GENERATION, MARK_STALE))


# No view is found by a walk (llama.cpp-hgm4): the pointer table's leases come out of a bag keyed by (device, entry
# key), the same pair the drop's match reads, and the route table and the hybrid decode record are asked at the moved
# expert's own slot, where its retired entry is (an expert move refuses a device copy that shares storage). A table or
# record whose slot does not hold the entry is still walked whole, so no view is missed, and each such walk counts in
# the transaction's view_scan_fallbacks, which the swap summary prints.
LEASE_KEY_OF = "if (!h.is_weight()) { return false; } *out = moe_lease_key{ h.device(), h.key() }; return true;"
LEASE_KEY_OF_SIG = "bool operator()(const mem_handle & h, moe_lease_key * out) const"
LEASE_BAG = "using moe_lease_bag = keyed_handle_bag<mem_handle, moe_lease_key, moe_lease_key_hash, moe_lease_key_of>;"
LEASE_MEMBER = "ggml_sycl::moe_lease_bag moe_expert_ptrs_leases[GGML_SYCL_MAX_DEVICES];"
ROUTE_SLOT = "bool route_holds = slot < routes.experts.size() && on_entry(routes.experts[slot].lease);"
ROUTE_WALK = ("if (!route_holds && !routes.experts.empty()) { ++scan_fallbacks; route_holds = "
              "std::any_of(routes.experts.begin(), routes.experts.end(), "
              "[&](const ggml_sycl::moe_route_entry & route) { return on_entry(route.lease); }); }")
ROUTE_DROP = "if (route_holds) { for (ggml_sycl::moe_route_entry & held : routes.experts) {"
RECORD_WITHDRAW_DEF = ("static bool ggml_sycl_moe_hybrid_decode_record_withdraw(std::shared_ptr<const "
                       "moe_hybrid_decode_record> & held, int device, int expert_id, const "
                       "ggml_sycl::unified_cache_key & withdrawn, std::vector<ggml_sycl::mem_handle> & retained, "
                       "std::vector<int32_t> & emptied, uint32_t & scan_fallbacks)")
RECORD_SLOT = "const bool at_slot = ggml_sycl::moe_hybrid_record_slot_matches(held->lease, expert_id, on_entry);"
RECORD_WALK = ("if (!at_slot) { ++scan_fallbacks; if (std::none_of(held->lease.begin(), held->lease.end(), on_entry)) "
               "{ return false; } }")
RECORD_SLOT_WITHDRAW = ("if (at_slot) { copy->n_withdrawn += ggml_sycl::moe_hybrid_record_withdraw_slot("
                        "copy->device_weight, copy->host_weight, copy->lease, expert_id, &retained, &emptied) ? 1 : 0; "
                        "}")
TXN_WALKS = ("result.plan_chunks_cloned = next_plan->entries.chunks_cloned(); "
             "result.view_scan_fallbacks = view_scan_fallbacks;")


def definition(text: str, signature: str) -> str:
    """The body of the definition of `signature` (its text followed by ` {`), past any declaration of it."""
    at = text.find(signature + " {")
    return body(text[at:], signature) if at >= 0 else ""


def claim_lent_views_found_without_a_walk(sycl: str, route_hpp: str, common: str) -> bool:
    n = norm(sycl)
    helper = body(n, DROP_SIG)
    record = definition(n, RECORD_WITHDRAW_DEF)
    txn = body(n, TXN_SIG)
    r = norm(route_hpp)
    return (body(r, LEASE_KEY_OF_SIG).count(LEASE_KEY_OF) == 1 and r.count(LEASE_BAG) == 1
            and norm(common).count(LEASE_MEMBER) == 1
            and helper.count("moe_expert_ptrs_leases") == 1 and helper.count(LEASE_TAKE) == 1
            and all(helper.count(x) == 1 for x in (ROUTE_SLOT, ROUTE_WALK, ROUTE_DROP))
            and ordered(helper, ROUTE_SLOT, ROUTE_WALK, ROUTE_DROP, "routes.invalidate();")
            and helper.count("++scan_fallbacks;") == 1
            and all(record.count(x) == 1 for x in (RECORD_SLOT, RECORD_SLOT_WITHDRAW, RECORD_WALK))
            and ordered(record, RECORD_SLOT, RECORD_WALK, RECORD_SLOT_WITHDRAW)
            and record.count("++scan_fallbacks;") == 1
            and txn.count(DROP) == 1 and txn.count(TXN_WALKS) == 1
            and txn.count("uint32_t view_scan_fallbacks = 0;") == 1)


# --- an exception while lending publishes what was lent -------------------------------------------------------------

HOIST_SNAPSHOT = "auto next = std::make_shared<ggml_sycl::lifecycle_plan_snapshot>(*current);"
HOIST_PLAN = "auto next_plan = std::make_shared<ggml_sycl::placement_plan>(*current->plan);"
FORGET_LOOP_END = FORGET + " } } for (int r = 0; r < 3; ++r) {"
CATCH_LENDING = ('} catch (const ggml_sycl_fallback_error &) { throw; } catch (...) { '
                 'escaped = std::current_exception(); '
                 'stop = ggml_sycl_expert_move_texts_of(triplet.kind).exception; }')
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
                        FLIP_IF_LOANED, FLIP, BACKING, BREAK_ESCAPED, PUBLISH, UNLOCK, LOAN_WARN, RETHROW_EXITS[2]))


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
# The refill and the named admit share one cache body; the refill runs it with admit false, which keeps every refill
# check the refill had: admit only changes the source rule, a whole pinned host copy that is a host-zone slice. Both
# size the copy from the tensor and mint the device storage at the staged size.
REFILL_BODY_SIG = "bool unified_cache::copy_host_triplet_to_device("
REFILL_WRAP = "{ return copy_host_triplet_to_device(members, false, out); }"
SELECT_SIG = "static std::vector<ggml_sycl::expert_placement_key> ggml_sycl_expert_refill_select("
FINAL_SIG = "static bool ggml_sycl_expert_move_refusal_is_final(const char * reason)"
DEVICE_ALLOC_SIG = "static ggml_sycl::mem_handle ggml_sycl_device_expert_alloc(int device, size_t size)"
REFILL_RUN_SIG = "static uint32_t ggml_sycl_expert_refill_transaction_run("
ARM_SIG = "static void ggml_sycl_expert_refill_set_armed(int device, bool armed)"
BOUNDARY_DEVICE_SIG = "static uint32_t ggml_sycl_run_expert_refill_at_boundary_on_device("
BOUNDARY_SIG = "static void ggml_sycl_run_expert_refill_at_boundary(ggml_backend_t backend)"
LEDGER_SIG = "static ggml_sycl_expert_lent_ledger & ggml_sycl_expert_lent_ledger_locked("
RECONCILE_SIG = "static void ggml_sycl_expert_lent_ledger_reconcile_locked("
RECONCILE_DEF = RECONCILE_SIG + "int device, const ggml_sycl::lifecycle_plan_snapshot * current) noexcept {"
REPORT_SIG = "static void ggml_sycl_expert_refill_log_report()"
PER_GRAPH_SIG = "static uint32_t ggml_sycl_moe_refill_per_graph()"
FREE_SIG = "static void ggml_backend_sycl_free(ggml_backend_t backend)"


def refill_bodies(cache: str, sycl: str) -> list[str]:
    """The refill code, cache side and backend side; the shared transaction body is in loan_bodies()."""
    c = norm(cache)
    s = norm(sycl)
    return [body(c, REFILL_SIG), body(c, REFILL_BODY_SIG), body(s, SELECT_SIG), body(s, FINAL_SIG),
            body(s, DEVICE_ALLOC_SIG),
            body(s, REFILL_RUN_SIG), body(s, ARM_SIG), body(s, BOUNDARY_DEVICE_SIG), body(s, BOUNDARY_SIG),
            body(s, LEDGER_SIG)]


DIRECTION_ENUM = "enum class ggml_sycl_expert_txn_direction { LOAN, REFILL, NAMED, };"
LOAN_ENTRY = ("return ggml_sycl_expert_txn_run(ggml_sycl_expert_txn_direction::LOAN, device, bytes_wanted, slot_bytes, "
              "probe_mode, resolver, out, warn_busy);")
REFILL_ENTRY = ("const ggml_sycl_txn_result answer = ggml_sycl_expert_txn_run(ggml_sycl_expert_txn_direction::REFILL, "
                "device, max_triplets, 0, false, resolver, result, false);")
NAMED_ENTRY = ("const ggml_sycl_txn_result code = ggml_sycl_expert_txn_run(ggml_sycl_expert_txn_direction::NAMED, device, "
               "moves.size(), 0, false, resolver, &answer, false, &moves);")
MOVE_TXN_SIG = "static ggml_sycl_txn_result ggml_sycl_expert_move_transaction_run("
REFILL_FLAG = "const bool refill = direction == ggml_sycl_expert_txn_direction::REFILL;"
REFILL_CALL = "cache->refill_lent_triplet(triplet.members, record)"
# Each triplet's kind picks the cache body that moves it and the words its refusals use; the call's direction picks
# the kind of every triplet it prepares.
KIND_ENUM = "enum class ggml_sycl_expert_move_kind { LOAN, REFILL, ADMIT, EVICT, };"
KIND_OF_CALL = ("const ggml_sycl_expert_move_kind kind = refill ? ggml_sycl_expert_move_kind::REFILL : named ? "
                "ggml_sycl_expert_move_kind::ADMIT : ggml_sycl_expert_move_kind::LOAN;")
KIND_TO_DEVICE = "const bool to_device = ggml_sycl_expert_move_to_device(triplet.kind);"


def claim_one_body_every_direction(sycl: str) -> bool:
    """One body runs every direction: the loan's entry, the refill's and the named move's call it with their
    direction, and its lock, its publication and its epoch bump are the one copy all share. One statement moves every
    triplet, its cache body picked by the triplet's kind."""
    s = norm(sycl)
    txn = body(s, TXN_SIG)
    return (s.count(DIRECTION_ENUM) == 1 and s.count(KIND_ENUM) == 1 and s.count("ggml_sycl_expert_txn_run(") == 4
            and body(s, LOAN_TXN_SIG).count(LOAN_ENTRY) == 1 and body(s, REFILL_RUN_SIG).count(REFILL_ENTRY) == 1
            and body(s, MOVE_TXN_SIG).count(NAMED_ENTRY) == 1
            and txn.count(REFILL_FLAG) == 1 and txn.count(KIND_OF_CALL) == 1 and txn.count(KIND_TO_DEVICE) == 1
            and txn.count(MOVE_CALL) == 1 and txn.count("loan_expert_triplet(") == 1
            and txn.count("refill_lent_triplet(") == 1 and txn.count("admit_host_triplet(") == 1
            and txn.count(INVENTORY_LOCK) == 1 and txn.count(PUBLISH) == 1 and txn.count(EPOCH) == 1
            and ordered(txn, REFILL_FLAG, KIND_OF_CALL, PREPARE, INVENTORY_LOCK, KIND_TO_DEVICE, MOVE_CALL, PUBLISH,
                        EPOCH))


def claim_refill_never_waits(cache: str, sycl: str) -> bool:
    bodies = refill_bodies(cache, sycl)
    return all(bodies) and not any(token in b for b in bodies for token in FORBIDDEN_WAITS)


REFILL_SRC = "src_owner[i] = e.direct_alloc_owner;"
REFILL_COPY = "out.copy_events[i] = mem_copy_async(members[i].owner, src_owner[i], bytes[i], bq, deps[i]);"
BCS_QUEUE = "sycl::queue & bq = get_bcs_queue();"
ENTRY_OWNER = "entry.direct_alloc_owner = members[i].owner;"
ENTRY_EVENTS = ("entry.state = cache_entry_state::IN_PROGRESS;", "entry.has_ready_event = true;",
                "entry.ready_event = out.copy_events[i];", "entry.last_write_event = out.copy_events[i];",
                "entry.has_write_event = true;")


def claim_refill_copy_is_owner_to_owner_on_the_copy_engine(cache: str) -> bool:
    """The copy reads the host copy's own typed owner and writes the caller's device owner, so mem_copy_async takes
    the direct DMA and retains both until it completes; it runs on the BCS queue behind the host copy's writes, and
    the device entry is published IN_PROGRESS with the copy as its ready event and its last write."""
    b = body(norm(cache), REFILL_BODY_SIG)
    return (all(b.count(x) == 1 for x in (REFILL_SRC, REFILL_COPY, BCS_QUEUE, ENTRY_OWNER) + ENTRY_EVENTS)
            and all(d in b for d in DEPS) and "make_copy_handle_for_raw_ptr(" not in b and "host_task" not in b
            and ordered(b, REFILL_SRC, BCS_QUEUE, REFILL_COPY, ENTRY_OWNER))


EMPLACE = "entries_.emplace(device_key, device_entry[i]);"
MINT_MIRROR = ("device_mirror[i].handle = make_direct_entry_handle(device_key, cache_device, stored, "
               '"expert_refill/mirror");')
MINT_OUT = ("out.handles[i] = mem_handle::from_weight_lease_locked(device_key, cache_device, stored.device_ptr, "
            "stored.layout, true, &stored);")
SWAP_MIRROR = "std::swap(direct_expert_entries_.find(key)->second, device_mirror[i]);"
REMAP_ID = "id_to_key_.find(key)->second = expert_device_key(key);"
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
    b = body(norm(cache), REFILL_BODY_SIG)
    swap_at = b.find(SWAP_MIRROR)
    tail = b[swap_at:b.find("} } } ", swap_at) + 6] if swap_at >= 0 else ""
    needles = (EMPLACE, MINT_MIRROR, MINT_OUT, SWAP_MIRROR, REMAP_ID, HOST_RETIRE, PUBLICATION_FAILED, REFILL_CATCH,
               RELEASED_SIZED)
    return (all(b.count(x) == 1 for x in needles)
            and bool(tail) and not any(x in tail for x in NODE_BUILDERS + ("make_direct_entry_handle(",))
            and ordered(b, REFILL_COPY, RELEASED_SIZED, CACHE_LOCKS, EMPLACE, MINT_MIRROR, MINT_OUT, REFILL_CATCH,
                        PUBLICATION_FAILED, SWAP_MIRROR, REMAP_ID, HOST_RETIRE))


REFILL_SESSION = 'scoped_planned_materialization materialization(this, admit ? "expert admit" : "expert refill");'


def claim_refill_publishes_in_a_planned_materialization(cache: str) -> bool:
    """The device entries are published inside a planned materialization session, as the loan registers its host copies
    in one: the session opens after the copy is submitted and before the locks the publication takes."""
    b = body(norm(cache), REFILL_BODY_SIG)
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
    b = body(norm(cache), REFILL_BODY_SIG)
    return (b.count(HOST_RETIRE) == 1 and "retire_expert_entry_exact(" not in b
            and ordered(b, SWAP_MIRROR, REMAP_ID, HOST_RETIRE))


REFILL_CHECKS = ('refusal = admit ? "not on host" : "not lent";', 'refusal = "layout";',
                 'refusal = admit ? "host copy not pinned" : "not a lent copy";',
                 'refusal = "host copy not published";', 'refusal = "device entry exists";',
                 'refusal = "host copy size";', 'refusal = "device storage size";')
REFILL_LOAD_EFFECT = ('const cache_load_effect load_effect = acquire_bound_load_effect(); if (load_effect.failed()) '
                      '{ out.reason = "load effect unavailable"; return false; }')


def claim_refill_checks_before_copying(cache: str) -> bool:
    """What would make the publication wrong is checked, all three members, before anything is copied: the load effect,
    the plan coherence, that each host copy is a loan's and is what resolves its id, that no device entry is in the
    way, and that the device storage is exactly the host copy's size. The refill runs the shared body with admit
    false, so each check is the refill's."""
    b = body(norm(cache), REFILL_BODY_SIG)
    return (body(norm(cache), REFILL_SIG) == REFILL_WRAP
            and all(b.count(x) == 1 for x in (REFILL_LOAD_EFFECT, COHERENCE, REFUSE_EARLY) + REFILL_CHECKS)
            and ordered(b, REFILL_LOAD_EFFECT, COHERENCE, *REFILL_CHECKS, REFILL_SRC, REFUSE_EARLY, REFILL_COPY))


# A move to the device mints every member's storage at what the load gave it (expert_triplet_member_bytes): its
# triplet's slot when it has one, else its own staged size, from the staged sizes the layout gives the three members'
# tensors' bytes per expert, refill and admit alike, never at a host copy's size. The load stages each member of a
# slotted triplet at the slot, so the block an admitted member takes is one any evicted member gave back.
DEVICE_BYTES = ("staged[r] = ggml_sycl::staged_expert_bytes(e->layout, target.type, target.ncols, "
                "target.logical_bytes);")
DEVICE_MEMBER = "const size_t member_bytes = ggml_sycl::expert_triplet_member_bytes(staged, r);"
DEVICE_ALLOC = "storage = ggml_sycl_device_expert_alloc(device, member_bytes);"
ZONE_ALLOC = ("ggml_sycl::allocation_result allocation = ggml_sycl::unified_cache_zone_allocate_owner(device, "
              "ggml_sycl::vram_zone_id::WEIGHT, size);")


def claim_refill_lock_order(cache: str, sycl: str) -> bool:
    """The device storage is minted owner-first from the WEIGHT zone, in the preparation, before
    g_tensor_inventory_mutex is taken, and the cache is asked to re-admit under it; the cache takes direct_stage_mutex_
    and rw_mutex_ together, with std::lock, and never the inventory mutex; no refill code names
    g_sycl_graph_compute_mutex."""
    txn = body(norm(sycl), TXN_SIG)
    prep = body(norm(sycl), PREP_SIG)
    refill = body(norm(cache), REFILL_BODY_SIG)
    alloc = body(norm(sycl), DEVICE_ALLOC_SIG)
    return (prep.count(DEVICE_ALLOC) == 1 and alloc.count(ZONE_ALLOC) == 1
            and claim_storage_minted_before_the_inventory_lock(sycl)
            and ordered(txn, PREPARE, INVENTORY_LOCK, REFILL_CALL)
            and "g_tensor_inventory_mutex" not in refill and refill.count(CACHE_LOCKS) >= 2
            and "std::unique_lock<std::shared_mutex> cache_lock(rw_mutex_);" not in refill
            and not any("g_sycl_graph_compute_mutex" in b for b in refill_bodies(cache, sycl)))


FORGET_HOST = ("(void) triplet.extras[r]->forget_moe_storage_handle_on_device(triplet.members[r].placement.expert_id, "
               "GGML_LAYOUT_AOS, ggml_sycl::mem_handle::HOST_DEVICE);")
# A move to the device drops the host copy's records; a loan drops the device copy's views and records.
FORGET_BY_DIRECTION = ("if (to_device) { for (int r = 0; r < 3; ++r) { " + FORGET_HOST + " } } else {")
# Each lent triplet's device bytes sit beside it in the ledger and move with it; a named move keeps no ledger,
# its admits and evicts alike marked moved with their device bytes for the caller's sums.
POP_READMITTED = ("switch (triplet.kind) { case ggml_sycl_expert_move_kind::LOAN: "
                  "ledger.triplets.push_back({ placed[0], placed[1], placed[2] }); "
                  "ledger.triplet_bytes.push_back(ggml_sycl::moe_expert_triplet_dst_bytes(*next_plan, placed)); break; "
                  "case ggml_sycl_expert_move_kind::REFILL: ledger.triplets.pop_back(); "
                  "ledger.triplet_bytes.pop_back(); break; "
                  "case ggml_sycl_expert_move_kind::ADMIT: case ggml_sycl_expert_move_kind::EVICT: "
                  "(*named_moves)[triplet.named].moved = true; "
                  "(*named_moves)[triplet.named].device_bytes = record.device_bytes; break; }")


def claim_refill_flips_back_and_forgets_the_host_copy(sycl: str) -> bool:
    """Each extra drops its lease on the withdrawn host copy before anything allocates, then remembers the device copy
    behind its copy event; the plan flips the triplet back on the device; the ledger loses it; the plan is published
    once, then the exec-graph epoch is bumped."""
    b = body(norm(sycl), TXN_SIG)
    return (b.count(FORGET_HOST) == 1 and b.count(FORGET_BY_DIRECTION) == 1 and b.count(REMEMBER) == 1
            and b.count(POP_READMITTED) == 1
            and ordered(b, MOVE_CALL, FORGET_BY_DIRECTION, FORGET, REMEMBER, CATCH_LENDING, FLIP_IF_LOANED, FLIP,
                        BACKING, POP_READMITTED, PUBLISH, EPOCH))


DISARM = "if (!refill && lent != 0) { ledger.refill_armed = false; }"
CANCELLED = '"cancelled by a loan";'
SELECT_ARMED = "if (ledger.refill_armed) {"
SELECT_REMAINING = "*remaining = ledger.refill_armed ? static_cast<uint32_t>(ledger.triplets.size()) : 0;"
TXN_REMAINING = "result.n_remaining = ledger.refill_armed ? static_cast<uint32_t>(ledger.triplets.size()) : 0;"
RESERVE = ("if (!refill && !named) { ledger.triplets.reserve(ledger.triplets.size() + prepared.size()); "
           "ledger.triplet_bytes.reserve(ledger.triplet_bytes.size() + prepared.size()); }")


def claim_a_loan_cancels_the_refills(sycl: str) -> bool:
    """An accepted loan disarms the cursor, and a disarmed cursor selects nothing and holds nothing, so the refill
    still to run is cancelled and its triplets stay lent with the loan's; a loan accepted between the selection and
    the lock cancels too. The ledger room a loan appends into is reserved before its first triplet moves."""
    s = norm(sycl)
    txn = body(s, TXN_SIG)
    select = body(s, SELECT_SIG)
    return (txn.count(DISARM) == 1 and txn.count(CANCELLED) == 1 and txn.count(TXN_REMAINING) == 1
            and select.count(SELECT_ARMED) == 1 and select.count(SELECT_REMAINING) == 1
            and ordered(txn, HOIST_PLAN, RESERVE, CANCELLED, MOVE_CALL, POP_READMITTED, DISARM, TXN_REMAINING,
                        PUBLISH))


LEDGER_LOCK = "std::lock_guard lock(g_tensor_inventory_mutex);"
OWE_SIG = "static void ggml_sycl_expert_bulk_refill_owe(int device, uint64_t first_graph)"
TAKE_SIG = "static uint64_t ggml_sycl_expert_bulk_refill_take(int device)"
OWE_NEXT_SIG = "static uint32_t ggml_sycl_expert_bulk_refill_owe_next_context(int device)"
OWED_NEXT_SIG = "static uint32_t ggml_sycl_expert_bulk_refill_owed_next_context(int device)"
BETWEEN_SIG = "static uint32_t ggml_sycl_run_expert_bulk_refill_between_contexts("
LEDGER_IN_TXN = "ggml_sycl_expert_lent_ledger & ledger = ggml_sycl_expert_lent_ledger_locked(device, *current);"


def claim_ledger_is_guarded_by_the_inventory_mutex(sycl: str) -> bool:
    """Every site that reaches a device's ledger holds g_tensor_inventory_mutex: the selection and the arming take it,
    the body reaches the ledger only after its own lock, and the one direct reader besides those is the private
    fixture's teardown, under the same lock. The reconcile helper subscripts it too; its only callers are the
    accessor and the publication, whose caller holds the lock, so the helper's name appears exactly four times: the
    forward declaration, the definition and those two calls. The bulk refill's owe and take, and the debt an ended
    context leaves (its owe, the report's read of it and the next context's refill), take the lock before they reach
    it (llama.cpp-i4ny). The ten subscripts are the declaration, the helper, the accessor's return, the arming, those
    six and the teardown."""
    s = norm(sycl)
    select = body(s, SELECT_SIG)
    arm = body(s, ARM_SIG)
    txn = body(s, TXN_SIG)
    return (s.count("g_sycl_expert_lent[") == 10
            and s.count("static ggml_sycl_expert_lent_ledger g_sycl_expert_lent[GGML_SYCL_MAX_DEVICES];") == 1
            and body(s, LEDGER_SIG).count("g_sycl_expert_lent[device]") == 1
            and s.count(RECONCILE_DEF) == 1
            and body(s[s.find(RECONCILE_DEF):], RECONCILE_SIG).count("g_sycl_expert_lent[device]") == 1
            and s.count("ggml_sycl_expert_lent_ledger_reconcile_locked(") == 4
            and ordered(select, LEDGER_LOCK, "ggml_sycl_expert_lent_ledger_locked(device, current)")
            and ordered(arm, LEDGER_LOCK, "g_sycl_expert_lent[device]")
            and ordered(body(s, OWE_SIG), LEDGER_LOCK, "g_sycl_expert_lent[device]")
            and ordered(body(s, TAKE_SIG), LEDGER_LOCK, "g_sycl_expert_lent[device]")
            and all(ordered(body(s, sig), LEDGER_LOCK, "g_sycl_expert_lent[device]")
                    for sig in (OWE_NEXT_SIG, OWED_NEXT_SIG, BETWEEN_SIG))
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


FINAL_REASONS = ("invalid arguments", "invalid member", "device storage not owned", "not lent", "not on host", "layout",
                 "not a lent copy", "host copy not pinned", "host copy not published", "host copy size",
                 "device storage size")
# Where the shared cache body returns each: a refusal the refill and the admit name differently is one ternary.
FINAL_REFUSALS = {
    "invalid arguments": 'out.reason = "invalid arguments";',
    "invalid member": 'refusal = "invalid member";',
    "device storage not owned": 'refusal = "device storage not owned";',
    "not lent": 'refusal = admit ? "not on host" : "not lent";',
    "not on host": 'refusal = admit ? "not on host" : "not lent";',
    "layout": 'refusal = "layout";',
    "not a lent copy": 'refusal = admit ? "host copy not pinned" : "not a lent copy";',
    "host copy not pinned": 'refusal = admit ? "host copy not pinned" : "not a lent copy";',
    "host copy not published": 'refusal = "host copy not published";',
    "host copy size": 'refusal = "host copy size";',
    "device storage size": 'refusal = "device storage size";',
}
FINAL_POP = ("stop = record.reason; if (triplet.kind == ggml_sycl_expert_move_kind::REFILL) { "
             "transient = !ggml_sycl_expert_move_refusal_is_final(stop); if (!transient) { drop_back(stop); } } break;")


def claim_final_refusals_name_the_cache_refusals(cache: str, sycl: str) -> bool:
    """Each reason the transaction treats as final (a refill's triplet leaves the ledger rather than being retried at
    every decode graph; a named triplet is answered REFUSED) is a refusal the cache's shared body really returns, so a
    renamed refusal cannot silently turn final into retried forever."""
    refill = body(norm(cache), REFILL_BODY_SIG)
    final = body(norm(sycl), FINAL_SIG)
    return (all(('"%s"' % r) in final for r in FINAL_REASONS)
            and all(FINAL_REFUSALS[r] in refill for r in FINAL_REASONS)
            and body(norm(sycl), TXN_SIG).count(FINAL_POP) == 1)


# --- a lent triplet the refill can never prepare leaves the ledger, WARNed (llama.cpp-sif1) -------------------------

RESOLVE_FAILED = "if (!resolved || !target.extra) { return texts.no_tensor; }"
# The refill's row of the kind table: the words its preparation and its log use.
REFILL_TEXTS = ('case ggml_sycl_expert_move_kind::REFILL: return { "no tensor for a lent expert", "not lent", '
                '"exception while refilling", "refill" };')
ARENA_TOLD_APART = ("const bool arena = cache.arena_active() && "
                    "cache.zone_capacity(ggml_sycl::vram_zone_id::WEIGHT) != 0; *transient = arena; "
                    'return arena ? "WEIGHT zone full" : "no WEIGHT arena";')
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
    the same triplet. The preparation tells the two apart and the transaction records the triplet."""
    s = norm(sycl)
    b = body(s, TXN_SIG)
    prep = body(s, PREP_SIG)
    needles = (UNPREPARABLE, EARLY_REFUSAL, DROP_AT_BACK, DROP_COUNTED, LENT_COUNT_STORE)
    return (all(b.count(x) == 1 for x in needles) and s.count(REFILL_TEXTS) == 1
            and prep.count(RESOLVE_FAILED) == 1 and prep.count(ARENA_TOLD_APART) == 1
            and ordered(prep, RESOLVE_FAILED, DEVICE_ALLOC, ARENA_TOLD_APART)
            and ordered(b, PREPARE, UNPREPARABLE, EARLY_REFUSAL, INVENTORY_LOCK, PLAN_CHANGED, LEDGER_IN_TXN, MOVE_CALL,
                        POP_READMITTED, DROP_AT_BACK, DROP_COUNTED, LENT_COUNT_STORE))


DROP_BACK = ("auto drop_back = [&](const char * reason) { GGML_ASSERT(n_dropped < dropped.size()); "
             "dropped[n_dropped++] = { ledger.triplets.back()[0], reason }; ledger.triplets.pop_back(); "
             "ledger.triplet_bytes.pop_back(); };")
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


MOVE_LOOP_SIG = "for (ggml_sycl_expert_prepared_triplet & triplet : prepared)"
LOOP_TAIL = "++lent; (to_device ? taken : freed) += record.device_bytes; } if (escaped) { break; } }"
LOOP_STOPS = ("transient = ledger.refill_armed; break; }", FINAL_POP, "if (escaped) { break; }")


def claim_the_move_loop_moves_every_prepared_triplet(sycl: str) -> bool:
    """The move loop stops only where a triplet did not move: the cursor no longer holds it at its back (cancelled by a
    loan, or changed), the cache refuses a loan or a refill, or an exception escaped. A switch case's break leaves the
    switch, not the loop. After a triplet moves, the loop counts it, sums its device bytes in its direction and goes
    on, so a refill of more than one per call checks the next triplet against the back the first one's pop left."""
    loop = body(body(norm(sycl), TXN_SIG), MOVE_LOOP_SIG)
    return (loop.count("break;") == len(LOOP_STOPS) + POP_READMITTED.count("break;") + MOVE_CALL.count("break;")
            and all(loop.count(x) == 1 for x in LOOP_STOPS) and loop.count(POP_READMITTED) == 1
            and loop.endswith(LOOP_TAIL)
            and ordered(loop, CANCELLED, MOVE_CALL, FINAL_POP, POP_READMITTED, LOOP_TAIL))


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


# --- an unengaged prompt's loan comes back at once (llama.cpp-i4ny) -------------------------------------------------

GATE = read("scripts/sycl-moe-exec-gate.sh")
REFILLS_SIG = "static uint32_t ggml_sycl_run_expert_refills_at_boundary_on_device("
OWED_FIELD = "uint64_t bulk_refill_owed_at = 0;"
RECONCILE_CLEAR = "ledger.refill_armed = false; ledger.bulk_refill_owed_at = 0;"
OWE_SKIP = ("if (device < 0 || device >= GGML_SYCL_MAX_DEVICES || "
            "g_sycl_expert_lent_count[device].load(std::memory_order_acquire) == 0) { return; }")
OWE_SET = "ledger.bulk_refill_owed_at = ledger.triplets.empty() ? 0 : first_graph;"
TAKE_CLEAR = "const uint64_t owed_at = ledger.bulk_refill_owed_at; ledger.bulk_refill_owed_at = 0; return owed_at;"
BULK_TAKE = ("const bool lent_any = device >= 0 && device < GGML_SYCL_MAX_DEVICES && "
             "g_sycl_expert_lent_count[device].load(std::memory_order_acquire) != 0; "
             "const uint64_t owed_at = lent_any ? ggml_sycl_expert_bulk_refill_take(device) : 0;")
BULK_ELSE = ("if (!decode || owed_at == 0 || decode_graphs < owed_at) { "
             "return ggml_sycl_run_expert_refill_at_boundary_on_device(device, decode, decode_graphs, refilled_at, "
             "max_triplets, resolver, out); }")
BULK_LENT = "const uint32_t lent = g_sycl_expert_lent_count[device].load(std::memory_order_acquire);"
BULK_RUN = ("const uint32_t remaining = ggml_sycl_run_expert_refill_at_boundary_on_device(device, decode, "
            "decode_graphs, refilled_at, lent, resolver, result);")
BULK_GUARD = "if (result->accepted) {"
BULK_WARN = ('GGML_LOG_WARN("[MOE-EXEC] bulk refill device %d: %u of %u lent triplets re-admitted in one transaction '
             'at decode graph %llu after the prompt (its ring did not engage)\\n", device, result->n_triplets, lent, '
             "(unsigned long long) (decode_graphs - owed_at + 1));")
BOUNDARY_CALLS_REFILLS = ("(void) ggml_sycl_run_expert_refills_at_boundary_on_device(ctx->moe_stream.get(), "
                          + BOUNDARY_PHASE)
OWE_FROM_PHASE = ("if (phase != nullptr && phase->take_prompt_ended() && !phase->last_prompt_ring_engaged()) { "
                  "ggml_sycl_expert_bulk_refill_owe(device, decode_graphs); }")


def claim_an_unengaged_prompts_loan_comes_back_at_once(sycl: str) -> bool:
    """The synchronize boundary reaches the bulk refill through one runner, which the boundary hands the context's
    expert stream phase. First it owes the device a bulk refill when the phase says a prompt ended whose ring did not
    engage, at the decode graph graph compute has just counted, so the line reads decode graph 1; it is the owe's only
    call. The owe asks without the lock whether anything is lent, then marks the ledger under the lock, and only a
    ledger with triplets; a publication under another identity clears the mark with the ledger. The runner then asks
    without the lock and takes the mark under it at every boundary with lent triplets: a prompt graph's boundary drops
    it, and so does a mark for a later decode graph than this one. A decode graph's boundary runs the per-graph
    refill with max_triplets the ledger's count, one transaction, and WARNs one line, only when it was accepted, that
    names the decode graph since the prompt. Its WARN is the runner's only one."""
    s = norm(sycl)
    owe = body(s, OWE_SIG)
    take = body(s, TAKE_SIG)
    refills = body(s, REFILLS_SIG)
    reconcile = body(s[s.find(RECONCILE_DEF):], RECONCILE_SIG) if RECONCILE_DEF in s else ""
    return (s.count(OWED_FIELD) == 1 and s.count(RECONCILE_CLEAR) == 1 and reconcile.count(RECONCILE_CLEAR) == 1
            and owe.count(OWE_SET) == 1 and ordered(owe, OWE_SKIP, LEDGER_LOCK, OWE_SET)
            and take.count(TAKE_CLEAR) == 1 and ordered(take, LEDGER_LOCK, TAKE_CLEAR)
            and s.count("ggml_sycl_expert_bulk_refill_take(") == 2
            and s.count("ggml_sycl_expert_bulk_refill_owe(") == 2 and refills.count(OWE_FROM_PHASE) == 1
            and ordered(refills, OWE_FROM_PHASE, BULK_TAKE, BULK_ELSE, BULK_LENT, BULK_RUN, BULK_GUARD, BULK_WARN)
            and refills.count("GGML_LOG_WARN(") == 1
            and body(s, BOUNDARY_SIG).count(BOUNDARY_CALLS_REFILLS) == 1)


LEDGER_CALLS = ("ggml_sycl_expert_bulk_refill_owe(", "ggml_sycl_expert_bulk_refill_take(",
                "ggml_sycl_expert_bulk_refill_owe_next_context(", "ggml_sycl_expert_bulk_refill_owed_next_context(",
                "ggml_sycl_run_expert_bulk_refill_between_contexts(",
                "ggml_sycl_run_expert_refills_at_boundary_on_device(")


def graph_lock_segment(sycl: str) -> str:
    """Graph compute's text from taking the process-global graph lock to its end. Not to the first unlock: that one
    sits in the compute lambda, and the replay path still holds the lock past it. Empty when the lock is missing."""
    g = body(norm(sycl), GRAPH_COMPUTE_SIG)
    at = g.find(GRAPH_LOCK)
    return g[at:] if at >= 0 else ""


def claim_the_debt_is_never_marked_under_the_graph_lock(sycl: str) -> bool:
    """The ledger's lock, g_tensor_inventory_mutex, is rank L1 and the graph lock, g_sycl_graph_compute_mutex, rank L2;
    canonical contract section 12.5 forbids taking a lower rank while holding a higher one. So no call that takes the
    ledger's lock for the bulk refill debt sits in graph compute while it holds the graph lock: the owe runs at the
    synchronize boundary, and the next context's half before the lock is taken."""
    seg = graph_lock_segment(sycl)
    return seg != "" and not any(call in seg for call in LEDGER_CALLS)


def gate_regex(gate: str, name: str) -> str:
    """The value the gate script assigns to `name`, with a `$RE_...` reference to an earlier one expanded. Empty
    when the gate has no such assignment."""
    m = re.search(r"^%s=(['\"])(.*)\1$" % re.escape(name), gate, re.M)
    if not m:
        return ""
    return re.sub(r"\$(RE_[A-Z_]+)", lambda r: gate_regex(gate, r.group(1)), m.group(2))


def bulk_refill_line(sycl: str) -> str:
    """The runner's bulk refill WARN as it prints for nine of nine triplets on device 0 at decode graph 1: its
    format string with the printf conversions filled in. Empty when the runner has no such WARN."""
    m = re.search(r'GGML_LOG_WARN\("(\[MOE-EXEC\] bulk refill[^"]*)\\n"', body(norm(sycl), REFILLS_SIG))
    if not m:
        return ""
    fmt = re.sub(r"%(?:llu|u)", "%d", m.group(1))
    try:
        return fmt % (0, 9, 9, 1)
    except (TypeError, ValueError):
        return ""


def claim_bulk_refill_line_scores_as_the_gate_reads_it(sycl: str, gate: str) -> bool:
    """The gate's FAIL(refill_late) needs the bulk refill line at decode graph 1 (RE_BULK_REFILL_FIRST), and its
    refill_not_bulk control keeps that line out of RE_REFILL, which reads the exit report. The line the runner prints
    at decode graph 1 matches the first and not the second, under the gate's own regexes."""
    line = bulk_refill_line(sycl)
    first = gate_regex(gate, "RE_BULK_REFILL_FIRST")
    exit_report = gate_regex(gate, "RE_REFILL")
    return (bool(line) and bool(first) and bool(exit_report)
            and re.search(first, line) is not None and re.search(exit_report, line) is None)


# --- a loan an ended context leaves lent comes back at the next context (llama.cpp-i4ny) ---------------------------

NEXT_FIELD = "bool bulk_refill_next_context = false; };"
RECONCILE_NEXT = "ledger.bulk_refill_owed_at = 0; ledger.bulk_refill_next_context = false;"
OWE_NEXT_SET = ("ledger.bulk_refill_next_context = !ledger.triplets.empty(); "
                "return static_cast<uint32_t>(ledger.triplets.size());")
OWED_NEXT_READ = "return ledger.bulk_refill_next_context ? static_cast<uint32_t>(ledger.triplets.size()) : 0;"
BETWEEN_SKIP = "if (g_sycl_expert_lent_count[device].load(std::memory_order_acquire) == 0) { return 0; }"
BETWEEN_TAKE = ("const bool owed = ledger.bulk_refill_next_context; ledger.bulk_refill_next_context = false; "
                "if (!owed) { return static_cast<uint32_t>(ledger.triplets.size()); }")
BETWEEN_CURSOR = "std::atomic<uint64_t> refilled_at{ 0 };"
BETWEEN_RUN = ("const uint32_t remaining = ggml_sycl_run_expert_refill_at_boundary_on_device(device, true, 1, "
               "&refilled_at, lent, resolver, result);")
BETWEEN_WARN = ('GGML_LOG_WARN("[MOE-EXEC] bulk refill device %d: %u of %u lent triplets re-admitted in one '
                'transaction between contexts (the context that lent them ended)\\n", device, result->n_triplets, '
                "lent);")
FREE_OWE = ("if (sycl_ctx && sycl_ctx->graph_compute_seq != 0) { "
            "(void) ggml_sycl_expert_bulk_refill_owe_next_context(sycl_ctx->device); }")
ENTRY_CALL = ("(void) ggml_sycl_run_expert_bulk_refill_between_contexts("
              "static_cast<ggml_backend_sycl_context *>(backend->context)->device, nullptr, nullptr);")
GRAPH_LOCK = "std::unique_lock<std::mutex> global_graph_lock(g_sycl_graph_compute_mutex);"
OWED_REPORT_READ = "const uint32_t owed = ggml_sycl_expert_bulk_refill_owed_next_context(device);"
OWED_WARN = ('GGML_LOG_WARN("[MOE-EXEC] refill device %d: %u/%u triplets re-admitted (%u owed to the next '
             'context)\\n", device, readmitted, lent, owed);')


def claim_an_ended_contexts_loan_comes_back_at_the_next_context(sycl: str) -> bool:
    """A context that computed a graph and ends with triplets lent owes them to the next context. Its backend free
    marks the ledger under the lock before the refill report, copying nothing, so a process that exits after its last
    context pays nothing. The report reads the debt under the lock and prints it after the counts. Every graph entry,
    before the process-global graph lock and so before any MUL_MAT_ID, reaches the next context's half. It asks
    without the lock whether anything is lent, then takes the debt under it, clearing it whether or not it acts. With
    a debt it runs the deferred-free pass, which returns the room the ended context's ring held, and then the per-graph
    refill once, on a cursor of its own, with max_triplets the ledger's count. It WARNs one line, only when that was
    accepted, and no other. A publication under another identity clears the debt with the ledger."""
    s = norm(sycl)
    reconcile = body(s[s.find(RECONCILE_DEF):], RECONCILE_SIG) if RECONCILE_DEF in s else ""
    between = body(s, BETWEEN_SIG)
    return (s.count(NEXT_FIELD) == 1 and reconcile.count(RECONCILE_NEXT) == 1
            and ordered(body(s, OWE_NEXT_SIG), LEDGER_LOCK, OWE_NEXT_SET)
            and ordered(body(s, OWED_NEXT_SIG), LEDGER_LOCK, OWED_NEXT_READ)
            and ordered(between, BETWEEN_SKIP, LEDGER_LOCK, BETWEEN_TAKE, DEFERRED_FREES, BULK_LENT, BETWEEN_CURSOR,
                        BETWEEN_RUN, BULK_GUARD, BETWEEN_WARN)
            and between.count("GGML_LOG_WARN(") == 1
            and body(s, FREE_SIG).count(FREE_OWE) == 1 and ordered(body(s, FREE_SIG), FREE_OWE, FREE_REPORTS)
            and body(s, GRAPH_COMPUTE_SIG).count(ENTRY_CALL) == 1
            and ordered(body(s, GRAPH_COMPUTE_SIG), ENTRY_CALL, GRAPH_LOCK)
            and ordered(body(s, REPORT_SIG), TAKE_LENT, TAKE_READMITTED, OWED_REPORT_READ, OWED_WARN, TEARDOWN_WARN))


def warn_line(text: str, signature: str, prefix: str, values: tuple) -> str:
    """The first WARN in the body of `signature` whose format starts with `prefix`, its printf conversions filled
    with `values`. Empty when there is none."""
    m = re.search(r'GGML_LOG_WARN\("(%s[^"]*)\\n"' % re.escape(prefix), body(norm(text), signature))
    if not m:
        return ""
    try:
        return re.sub(r"%(?:llu|u)", "%d", m.group(1)) % values
    except (TypeError, ValueError):
        return ""


def claim_debt_lines_score_as_the_gate_reads_them(sycl: str, gate: str) -> bool:
    """The free's report with a debt keeps the prefix the gate's RE_REFILL reads, so the gate still counts it as the
    exit report (the positive control). The next context's bulk line is a bulk refill line (RE_BULK_REFILL) that is
    neither the exit report (RE_REFILL) nor the decode-graph-1 line refill_late wants (RE_BULK_REFILL_FIRST). Only
    Q_tg and Q_noc score refill balance, and both are single-context llama-completion runs whose ledger the
    decode-graph-1 bulk refill has emptied by their free, so the debt form never reaches a scored line."""
    owed = warn_line(sycl, REPORT_SIG, "[MOE-EXEC] refill device", (0, 6, 9, 3))
    between = warn_line(sycl, BETWEEN_SIG, "[MOE-EXEC]", (0, 9, 9))
    exit_report = gate_regex(gate, "RE_REFILL")
    bulk = gate_regex(gate, "RE_BULK_REFILL")
    first = gate_regex(gate, "RE_BULK_REFILL_FIRST")
    return (bool(owed) and "owed to the next context" in owed and bool(between)
            and bool(exit_report) and bool(bulk) and bool(first)
            and re.search(exit_report, owed) is not None
            and re.search(bulk, between) is not None
            and re.search(exit_report, between) is None and re.search(first, between) is None)


# --- positive tests -------------------------------------------------------------------------------------------------

# --- copies are sized from the tensor, never from a cache entry -----------------------------------------------------

# The preparation asks the cache only whether the entry to copy from is there: an entry's size, which direct staging
# may have padded, decides nothing.
SOURCE_CHECK = "if (!cache.has_expert_source(target.key, to_device)) { return texts.no_source; }"
SOURCE_SIG = "bool unified_cache::has_expert_source(ggml_sycl_cache_id key, bool to_device) const"
SHARED_CACHE_LOCKS = ("std::shared_lock<std::shared_mutex> direct_lock(direct_stage_mutex_, std::defer_lock); "
                      "std::shared_lock<std::shared_mutex> cache_lock(rw_mutex_, std::defer_lock); "
                      "std::lock(direct_lock, cache_lock);")
SOURCE_FIND = "const auto it = entries_.find(to_device ? expert_host_key(key) : expert_device_key(key));"
SOURCE_RETURN = "return it != entries_.end() && !it->second.retired;"
# A loan copies the expert's own bytes from offset 0 of the device entry, which must hold them, into a host copy that
# holds them too; the entry's padded tail stays on the device.
LOAN_ENTRY_GUARD = 'if (e.size < members[i].logical_bytes) { refusal = "device entry size"; break; }'
LOAN_SIZE_GUARD = 'if (host_md[i].size < members[i].logical_bytes) { refusal = "host storage too small"; break; }'
LOAN_COPY_BYTES = "bytes[i] = members[i].logical_bytes; device_size[i] = e.size;"
# A refill and an admit copy the expert's own bytes from a host copy that holds them, into device storage of exactly
# what the load gives each member (expert_triplet_member_bytes), from the three members' staged sizes as AOS.
REFILL_STAGED = ("staged[i] = staged_expert_bytes(GGML_LAYOUT_AOS, members[i].type, members[i].ncols, "
                 "members[i].logical_bytes);")
REFILL_SIZE_GUARD = ('const size_t logical = members[i].logical_bytes; if (e.size < logical) { '
                     'refusal = "host copy size"; break; } const size_t member_bytes = '
                     'expert_triplet_member_bytes(staged, i); if (device_size[i] != member_bytes || '
                     'member_bytes < logical) { refusal = "device storage size"; break; }')


def claim_copies_are_sized_from_the_tensor(cache: str, sycl: str) -> bool:
    """The preparation mints each member's copy from its tensor's bytes per expert, never from a cache entry's size
    or the plan's dst_size: a loan's host copy at those bytes, a refill's and an admit's device storage at what the
    load gives each member, its triplet's slot or else its own staged size. The cache is asked only
    whether the source entry is there, under its two locks, and the cache's size checks stay as the guards: a loan's
    against the device entry and the host copy, a refill's against the host copy and the device storage. So an
    expert that goes to the host and back does not grow."""
    s = norm(sycl)
    txn = body(s, TXN_SIG)
    prep = body(s, PREP_SIG)
    c = norm(cache)
    source = body(c, SOURCE_SIG)
    loan = body(c, LOAN_SIG)
    refill = body(c, REFILL_BODY_SIG)
    return (prep.count(SOURCE_CHECK) == 1 and prep.count(DEVICE_BYTES) == 1 and prep.count(DEVICE_MEMBER) == 1
            and prep.count(DEVICE_ALLOC) == 1 and prep.count(HOST_ALLOC) == 1
            and len(re.findall(r"\bggml_sycl_device_expert_alloc\(", prep)) == 1
            and ordered(prep, RESOLVE_FAILED, SOURCE_CHECK, DEVICE_BYTES, DEVICE_MEMBER, DEVICE_ALLOC, HOST_ALLOC)
            and "dst_size" not in prep and "staged_expert_bytes(" not in txn
            and source.count(SHARED_CACHE_LOCKS) == 1 and source.count(SOURCE_FIND) == 1
            and source.count(SOURCE_RETURN) == 1 and ordered(source, SHARED_CACHE_LOCKS, SOURCE_FIND, SOURCE_RETURN)
            and ".size" not in source
            and "loan_copy_bytes(" not in c and "refill_copy_bytes(" not in c
            and loan.count(LOAN_ENTRY_GUARD) == 1 and loan.count(LOAN_SIZE_GUARD) == 1
            and loan.count(LOAN_COPY_BYTES) == 1 and ordered(loan, LOAN_ENTRY_GUARD, LOAN_SIZE_GUARD, LOAN_COPY_BYTES)
            and refill.count(REFILL_STAGED) == 1 and refill.count(REFILL_SIZE_GUARD) == 1
            and ordered(refill, REFILL_STAGED, REFILL_SIZE_GUARD))


# --- a moved copy belongs to its source's models --------------------------------------------------------------------

# A model load's end tags what it staged, and a model teardown never reclaims an untagged entry, so a copy a move
# publishes after the load would outlive its model unless it is tagged by inheritance. Each copy takes the owners and
# tag of the entry it replaces, and that entry's pending load when the copy's own stamp left none, through one helper:
# the device entry an admit or a refill publishes, under the locks that re-check the host copy it replaces and before
# it is inserted, its mirror keeping the entry's pending load; and the host copy a loan or an evict
# registers, from the device entry the loan retired and holds in place, before the registration inserts it.
INHERIT_DEF = ("void inherit_entry_ownership(unified_cache_entry & dst, const unified_cache_entry & src) noexcept { "
               "dst.owner_mask |= src.owner_mask; dst.owner_tagged = dst.owner_tagged || src.owner_tagged; "
               "if (dst.pending_load_txn_id == 0) { dst.pending_load_txn_id = src.pending_load_txn_id; } }")
ADMIT_INHERIT = "inherit_entry_ownership(device_entry[i], it->second);"
ADMIT_MIRROR = "device_mirror[i].pending_load_txn_id = device_entry[i].pending_load_txn_id;"
REGISTER_BODY_SIG = ("bool unified_cache::register_host_expert(ggml_sycl_cache_id key, void * ptr, size_t size, "
                     "ggml_layout_mode layout, mem_handle * out_handle, std::shared_ptr<void> allocation_owner, "
                     "const mem_handle & typed_owner,")
REGISTER_OWNED_SIG = "bool unified_cache::register_host_expert_owned("
REGISTER_STAMP = "stamp_pending_owner(cache_entry, load_effect_guard);"
REGISTER_INHERIT = ("if (source) { inherit_entry_ownership(cache_entry, *source); "
                    "mirror_pending_txn = cache_entry.pending_load_txn_id; }")
REGISTER_INSERT = "entries_[cache_key] = cache_entry;"
REGISTER_MIRROR_TXN = "entry.pending_load_txn_id = mirror_pending_txn;"
LOAN_DEVICE_KEY = "out.device_keys[i] = device_key;"


def claim_a_moved_copy_belongs_to_its_sources_models(cache: str) -> bool:
    c = norm(cache)
    admit = body(c, REFILL_BODY_SIG)
    reg = body(c, REGISTER_BODY_SIG)
    owned = body(c, REGISTER_OWNED_SIG)
    loan = body(c, LOAN_SIG)
    return (c.count(INHERIT_DEF) == 1 and c.count("inherit_entry_ownership(") == 3
            and admit.count(ADMIT_INHERIT) == 1 and admit.count(ADMIT_MIRROR) == 1
            and ordered(admit, REFILL_SESSION, CACHE_LOCKS, ADMIT_INHERIT, ADMIT_MIRROR, EMPLACE, SWAP_MIRROR)
            and reg.count(REGISTER_INHERIT) == 1 and reg.count(REGISTER_MIRROR_TXN) == 1
            and ordered(reg, REGISTER_STAMP, REGISTER_INHERIT, REGISTER_INSERT, REGISTER_MIRROR_TXN)
            and owned.count(FORWARD) == 2
            and loan.count(REGISTER) == 1 and ordered(loan, LOAN_DEVICE_KEY, REGISTER))


# A moved copy whose source entry is gone would be published owned by nobody, and a model teardown never reclaims an
# untagged entry. The mover holds its source in place until its registrations return, so a missing source means the
# move is not what it was checked to be: the registration looks the source up first, under its locks, and refuses with
# one WARN that names the key, before it can erase an existing entry under the key, insert the copy or write its
# mirror row. The loan's rollback then withdraws the copies it registered and puts the device entries back.
REGISTER_SOURCE_FIND = ("const unified_cache_entry * source = nullptr; if (moved_from) { "
                        "const auto found = entries_.find(*moved_from); if (found == entries_.end()) {")
REGISTER_SOURCE_WARN = ('GGML_LOG_WARN("[UNIFIED-CACHE] register_host_expert refused model=%llu name_hash=0x%llx: '
                        'the entry its moved copy replaces is gone\\n", (unsigned long long) key.model_id, '
                        "(unsigned long long) key.name_hash);")
REGISTER_SOURCE_REFUSE = "return false; } source = &found->second; }"
REGISTER_OLD_ERASE = "erase_entry_locked(old);"


def claim_a_missing_source_refuses_before_the_insert(cache: str) -> bool:
    reg = body(norm(cache), REGISTER_BODY_SIG)
    return (reg.count(REGISTER_SOURCE_FIND) == 1
            and ordered(reg, CACHE_LOCKS, REGISTER_SOURCE_FIND, REGISTER_SOURCE_WARN, REGISTER_SOURCE_REFUSE,
                        REGISTER_OLD_ERASE, REGISTER_INSERT, REGISTER_MIRROR_TXN))


# --- the evict hands the id over before it retires ------------------------------------------------------------------

# Direct staging maps the id to the device entry, so a retire that found the mapping still there walked every entry of
# the cache under both locks looking for another to hand it to (13.6 ms a triplet at 75k entries). The loan names the
# host copy's key first, as an admit names the device entry before it retires the host copy, so its retire finds the id
# handed over and walks nothing, and the deferred-free pass that erases the entry later finds the same. A rollback hands
# the id back to the device entry before it withdraws any host copy, so neither walk runs there either, unless the host
# key is another registrant's live entry, which keeps it. A registration that fails after it named its copy under the
# id hands the id back to the entry it was to replace, which its mover puts back, so the rollback finds the id to hand
# back without building a mapping. The remap counts each walk, which the device test reads across an evict.
LOAN_HOST_KEY = "const unified_cache_key host_key = expert_host_key(members[i].key);"
LOAN_REPOINT = "expert_id_hand_over(id_to_key_, members[i].key, device_key, host_key);"
LOAN_HAND_BACK = ("expert_id_hand_back(id_to_key_, members[i].key, host_key, device_key, held_by_another, "
                  "spare_mapping[i]);")
# The three helpers the loan and its rollback call, which hold the mapping edits; the key helpers name the two keys.
HAND_OVER_SIG = "static void expert_id_hand_over("
HAND_BACK_SIG = "static void expert_id_hand_back("
SPARE_SIG = "static typename Map::node_type expert_spare_mapping("
HAND_OVER_BODY = ("const auto mapped = ids.find(id); if (mapped != ids.end() && mapped->second == device_key) "
                  "{ mapped->second = host_key; }")
HAND_BACK_BODY = ("const auto mapped = ids.find(id); if (mapped != ids.end() && mapped->second == host_key && "
                  "!held_by_another) { mapped->second = device_key; } else if (mapped == ids.end() && !spare.empty()) "
                  "{ ids.insert(std::move(spare)); }")
SPARE_BODY = ("const auto mapped = ids.find(id); if (mapped == ids.end() || !(mapped->second == device_key)) "
              "{ return {}; } Map spare; return spare.extract(spare.emplace(id, device_key).first);")
DEVICE_KEY_BODY = "return make_direct_stage_key(cache_entry_type::MOE_EXPERT, id, GGML_LAYOUT_AOS);"
HOST_KEY_BODY = "return unified_cache_key{ cache_entry_type::MOE_EXPERT, id, -1, -1 };"
LOAN_HELD_BY_ANOTHER = ("const bool held_by_another = host != entries_.end() && !host->second.retired && "
                        "host->second.direct_alloc_owner.owner_control_id() != members[i].owner.owner_control_id();")
REGISTER_HAND_BACK = ("if (id != id_to_key_.end() && id->second == cache_key) { if (moved_from && "
                      "detail::cache_id_equal(moved_from->id, key)) { id->second = *moved_from; } "
                      "else { id_to_key_.erase(id); } }")
REMAP_EARLY = "if (mapped == id_to_key_.end() || !(mapped->second == removed_key)) { return; }"
REMAP_WALK_COUNT = "id_remap_walks_.fetch_add(1, std::memory_order_relaxed);"
REMAP_WALK = "for (const auto & pair : entries_) {"


def claim_the_evict_hands_the_id_over_before_it_retires(cache: str) -> bool:
    c = norm(cache)
    loan = body(c, LOAN_SIG)
    reg = body(c, REGISTER_BODY_SIG)
    remap = body(c, REMAP_SIG)
    return (loan.count(LOAN_REPOINT) == 1 and loan.count(LOAN_HAND_BACK) == 1 and loan.count(LOAN_HELD_BY_ANOTHER) == 1
            and body(c, HAND_OVER_SIG).count(HAND_OVER_BODY) == 1 and body(c, HAND_BACK_SIG).count(HAND_BACK_BODY) == 1
            and body(c, "static unified_cache_key expert_device_key(").count(DEVICE_KEY_BODY) == 1
            and body(c, "static unified_cache_key expert_host_key(").count(HOST_KEY_BODY) == 1
            and ordered(loan, STAMP_FLAG, LOAN_HOST_KEY, LOAN_REPOINT, DEVICE_RETIRE, REGISTER)
            and ordered(loan, REGISTER_CATCH, LOAN_HELD_BY_ANOTHER, LOAN_HAND_BACK, HOST_WITHDRAW, RESTORE_CALL)
            and reg.count(REGISTER_HAND_BACK) == 1
            and remap.count(REMAP_WALK_COUNT) == 1 and ordered(remap, REMAP_EARLY, REMAP_WALK_COUNT, REMAP_WALK))


def test_the_evict_hands_the_id_over_before_it_retires():
    assert claim_the_evict_hands_the_id_over_before_it_retires(CACHE)


def test_mutant_evict_retires_with_the_id_still_on_the_device_entry_fails():
    """Without the hand-over the retire finds the id on its entry and walks the whole cache for a replacement."""
    assert not claim_the_evict_hands_the_id_over_before_it_retires(_once_in(CACHE, LOAN_SIG, LOAN_REPOINT, ""))


def test_mutant_evict_hands_the_id_over_after_the_retire_fails():
    moved = _once_in(CACHE, LOAN_SIG, LOAN_REPOINT, "")
    assert not claim_the_evict_hands_the_id_over_before_it_retires(
        _once_in(moved, LOAN_SIG, DEVICE_RETIRE, DEVICE_RETIRE + " " + LOAN_REPOINT))


def test_mutant_rollback_leaves_the_id_on_the_host_key_fails():
    """The rollback puts the device entry back with the id still naming the host key its withdrawn copy held."""
    assert not claim_the_evict_hands_the_id_over_before_it_retires(_once_in(CACHE, LOAN_SIG, LOAN_HAND_BACK, ""))


def test_mutant_rollback_takes_the_id_from_another_registrant_fails():
    assert not claim_the_evict_hands_the_id_over_before_it_retires(
        _once_in(CACHE, HAND_BACK_SIG, " && !held_by_another", ""))


# The functions that look an expert's entries up by key, and the least number of times each names the two keys through
# the helpers (the loan's checks, hand-over, rollback and restore; the copy to the device's checks, its device entries
# and the host copies it retires; the source probe's one lookup).
KEY_HELPER_USERS = ((LOAN_SIG, 4, 3), (REFILL_BODY_SIG, 6, 3), (SOURCE_SIG, 1, 1))
HAND_BUILT_HOST_KEY = re.compile(r"\{\s*cache_entry_type::MOE_EXPERT\s*,")


def claim_the_loan_names_its_keys_through_the_helpers(cache: str) -> bool:
    """The loan, the copy to the device and the source probe build no key by hand: a second spelling of the device
    entry's layout or of the host key's sentinels is one a helper edit would leave behind."""
    for signature, device_uses, host_uses in KEY_HELPER_USERS:
        text = body(norm(cache), signature)
        if (not text or "make_direct_stage_key(" in text or HAND_BUILT_HOST_KEY.search(text)
                or text.count("expert_device_key(") < device_uses or text.count("expert_host_key(") < host_uses):
            return False
    return True


def test_the_loan_names_its_keys_through_the_helpers():
    assert claim_the_loan_names_its_keys_through_the_helpers(CACHE)


DEVICE_KEY_BY_HAND = "make_direct_stage_key(cache_entry_type::MOE_EXPERT, {0}, GGML_LAYOUT_AOS)"
HOST_KEY_BY_HAND = "unified_cache_key{{ cache_entry_type::MOE_EXPERT, {0}, -1, -1 }}"


KEY_BY_HAND_MUTANTS = (
    ("loan-device", LOAN_SIG, "entries_.find(expert_device_key(members[i].key))",
     "entries_.find(" + DEVICE_KEY_BY_HAND.format("members[i].key") + ")"),
    ("loan-host", LOAN_SIG, "entries_.find(expert_host_key(host_id))",
     "entries_.find(" + HOST_KEY_BY_HAND.format("host_id") + ")"),
    ("copy-device", REFILL_BODY_SIG, "entries_.find(expert_device_key(key)) != entries_.end()) { refusal = "
     "\"device entry exists\";",
     "entries_.find(" + DEVICE_KEY_BY_HAND.format("key") + ") != entries_.end()) { refusal = "
     "\"device entry exists\";"),
    ("copy-host", REFILL_BODY_SIG, "const unified_cache_key host_key = expert_host_key(id);",
     "const unified_cache_key host_key = " + HOST_KEY_BY_HAND.format("id") + ";"),
    ("source-host", SOURCE_SIG, "entries_.find(to_device ? expert_host_key(key) : expert_device_key(key))",
     "entries_.find(to_device ? " + HOST_KEY_BY_HAND.format("key") + " : expert_device_key(key))"),
    ("source-device", SOURCE_SIG, "entries_.find(to_device ? expert_host_key(key) : expert_device_key(key))",
     "entries_.find(to_device ? expert_host_key(key) : " + DEVICE_KEY_BY_HAND.format("key") + ")"),
)


def test_mutant_a_key_spelled_by_hand_fails():
    for name, signature, old, new in KEY_BY_HAND_MUTANTS:
        assert not claim_the_loan_names_its_keys_through_the_helpers(_once_in(CACHE, signature, old, new)), name


def test_mutant_hand_over_moves_whatever_the_id_names_fails():
    """The id handed over from an entry it no longer names would point a reader at the host key of a stale copy."""
    assert not claim_the_evict_hands_the_id_over_before_it_retires(
        _once_in(CACHE, HAND_OVER_SIG, " && mapped->second == device_key", ""))


def test_mutant_device_key_names_another_layout_fails():
    assert not claim_the_evict_hands_the_id_over_before_it_retires(
        _once_in(CACHE, "static unified_cache_key expert_device_key(", "GGML_LAYOUT_AOS", "GGML_LAYOUT_SOA"))


def test_mutant_failed_registration_drops_the_moved_id_fails():
    """A registration whose mirror fails erases the id its copy took, and the rollback has no mapping to hand back."""
    assert not claim_the_evict_hands_the_id_over_before_it_retires(
        _once_in(CACHE, REGISTER_BODY_SIG, REGISTER_HAND_BACK,
                 "if (id != id_to_key_.end() && id->second == cache_key) { id_to_key_.erase(id); }"))


def test_mutant_remap_walks_uncounted_fails():
    assert not claim_the_evict_hands_the_id_over_before_it_retires(_once_in(CACHE, REMAP_SIG, REMAP_WALK_COUNT, ""))


# A rollback that finds the id mapping gone still puts the device entry back under its id. Something can erase the
# mapping while the locks are released between step 3 and the rollback (an id-wide retire of the copy this loan
# registered under the id finds no live entry to hand it to), and the restore re-inserts only a node its own retire
# took out, which the hand-over left behind. So the loan mints a spare mapping node for each id it hands over, in the
# check loop, where minting may still throw and nothing has changed, and the rollback inserts it when the id maps to
# nothing: no node is built in the rollback.
SPARE_DECL = "id_map::node_type spare_mapping[3];"
SPARE_MINT = "spare_mapping[i] = expert_spare_mapping(id_to_key_, members[i].key, device_key);"
SPARE_INSERT = "ids.insert(std::move(spare));"


def claim_a_rollback_restores_a_lost_id(cache: str) -> bool:
    loan = body(norm(cache), LOAN_SIG)
    at = loan.find("if (failed) {")
    rollback = loan[at:loan.find(REGISTRATION_REFUSED, at)] if at >= 0 else ""
    hand_back = body(norm(cache), HAND_BACK_SIG)
    return (loan.count(SPARE_DECL) == 1 and loan.count(SPARE_MINT) == 1
            and body(norm(cache), SPARE_SIG).count(SPARE_BODY) == 1 and hand_back.count(SPARE_INSERT) == 1
            and ordered(loan, SPARE_DECL, HOLD, SPARE_MINT, LOAN_REPOINT)
            and ordered(rollback, LOAN_HAND_BACK, HOST_WITHDRAW, RESTORE_CALL)
            and bool(rollback) and not any(x in rollback for x in NODE_BUILDERS)
            and not any(x in hand_back for x in NODE_BUILDERS))


def test_a_rollback_restores_a_lost_id():
    assert claim_a_rollback_restores_a_lost_id(CACHE)


def test_mutant_rollback_leaves_a_lost_id_lost_fails():
    """The device entry goes back with no id mapping: every reader that resolves the expert through its id misses."""
    assert not claim_a_rollback_restores_a_lost_id(_once_in(CACHE, HAND_BACK_SIG, SPARE_INSERT, ""))


def test_mutant_rollback_builds_the_mapping_fails():
    """Emplaced in the rollback, the mapping allocates where nothing may throw."""
    assert not claim_a_rollback_restores_a_lost_id(
        _once_in(CACHE, HAND_BACK_SIG, SPARE_INSERT, "ids.emplace(id, device_key);"))


def test_mutant_spare_minted_after_the_hand_over_fails():
    """Minted once the ids are handed over, a throw would leave the entries changed."""
    moved = _once_in(CACHE, LOAN_SIG, SPARE_MINT, "")
    assert not claim_a_rollback_restores_a_lost_id(
        _once_in(moved, LOAN_SIG, LOAN_REPOINT, LOAN_REPOINT + " " + SPARE_MINT))


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


def test_loan_boundary_drains_its_room():
    assert claim_loan_boundary_drains_its_room(SYCL)


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


def test_lent_views_dropped_by_key():
    assert claim_lent_views_dropped_by_key(SYCL)


def test_lent_views_retained_to_a_barrier():
    assert claim_lent_views_retained_to_a_barrier(SYCL)


def test_lent_views_mark_the_table_stale():
    assert claim_lent_views_mark_the_table_stale(SYCL, COMMON)


def test_warn_formats():
    assert claim_warn_formats(SYCL)


def test_device_test_registration():
    assert claim_device_test_registration(SYCL_CMAKE)


def test_one_body_every_direction():
    assert claim_one_body_every_direction(SYCL)


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


def test_an_unengaged_prompts_loan_comes_back_at_once():
    assert claim_an_unengaged_prompts_loan_comes_back_at_once(SYCL)


def test_the_debt_is_never_marked_under_the_graph_lock():
    assert claim_the_debt_is_never_marked_under_the_graph_lock(SYCL)


def test_bulk_refill_line_scores_as_the_gate_reads_it():
    assert claim_bulk_refill_line_scores_as_the_gate_reads_it(SYCL, GATE)


def test_an_ended_contexts_loan_comes_back_at_the_next_context():
    assert claim_an_ended_contexts_loan_comes_back_at_the_next_context(SYCL)


def test_debt_lines_score_as_the_gate_reads_them():
    assert claim_debt_lines_score_as_the_gate_reads_them(SYCL, GATE)


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


def test_mutant_pending_runner_drains_the_store_fails():
    """The drain belongs to the boundary: the runner draining it would be a wait inside the loan code."""
    assert not claim_loan_never_waits(
        CACHE, _once(SYCL, PENDING_TAKE, "(void) ggml_sycl::drain_retained_handles(true, 100); " + PENDING_TAKE))


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
        _once(CACHE, REGISTER, REGISTER.replace(", m.owner, &out.copy_events[i], false,",
                                                ", m.owner, nullptr, false,")), SYCL)


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
    """Refusing only a retired or leased host entry lets an idle one through to be replaced, storage and all, by a
    registration the loan could not undo."""
    busy_only = ("const auto host = entries_.find(expert_host_key(host_id)); "
                 "if (host != entries_.end() && (host->second.retired || host->second.in_use_count.load() != 0)) { "
                 'refusal = host->second.retired ? "host entry retiring" : "host entry exists"; break; }')
    assert not claim_existing_host_entry_never_replaced(_once(CACHE, HOST_EXISTS, busy_only), SYCL)


def test_mutant_retiring_host_entry_named_as_live_fails():
    """A retired host entry refused as "host entry exists": the transaction would answer a refusal that clears by
    itself as final, and a named evict right after an admit would never be tried again."""
    alike = HOST_EXISTS.replace('host->second.retired ? "host entry retiring" : "host entry exists"',
                                '"host entry exists"')
    assert not claim_existing_host_entry_never_replaced(_once(CACHE, HOST_EXISTS, alike), SYCL)


def test_mutant_loan_registration_may_replace_fails():
    assert not claim_existing_host_entry_never_replaced(
        _once(CACHE, REGISTER, REGISTER.replace("&out.copy_events[i], false,", "&out.copy_events[i], true,")), SYCL)


def test_mutant_registration_ignores_replace_existing_fails():
    assert not claim_existing_host_entry_never_replaced(
        _once(CACHE, NO_REPLACE, "if (old != entries_.end()) {"), SYCL)


def test_mutant_typed_registration_drops_replace_existing_fails():
    owned = body(norm(CACHE), OWNED_SIG)
    dropped = owned.replace(FORWARD, "ready_event, moved_from);")
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
    """The preparation, which mints, run with the inventory lock already held."""
    n = _once_in(SYCL, TXN_SIG, INVENTORY_LOCK, "")
    n = _once_in(n, TXN_SIG, PREPARE, INVENTORY_LOCK + " " + PREPARE)
    assert not claim_storage_minted_before_the_inventory_lock(n)
    assert not claim_lock_order(CACHE, n)


def test_mutant_transaction_mints_under_the_inventory_lock_fails():
    """A host copy minted by the transaction itself, under the lock, beside the preparation's."""
    n = _once_in(SYCL, TXN_SIG, INVENTORY_LOCK,
                 INVENTORY_LOCK + " ggml_sycl::mem_handle spare = ggml_sycl_host_expert_alloc(device, 64);")
    assert not claim_storage_minted_before_the_inventory_lock(n)
    assert not claim_lock_order(CACHE, n)


def test_mutant_preparation_takes_the_inventory_lock_fails():
    n = _once_in(SYCL, PREP_SIG, HOST_ALLOC, "std::lock_guard held(g_tensor_inventory_mutex); " + HOST_ALLOC)
    assert not claim_storage_minted_before_the_inventory_lock(n)


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
        _once(SYCL, REQUEST_CAS, "std::lock_guard lock(g_tensor_inventory_mutex), " + REQUEST_CAS))


def test_mutant_loan_runs_before_the_graph_flag_clears_fails():
    n = _once_in(SYCL, SYNC_SIG, RUN_PENDING, "")
    n = _once_in(n, SYNC_SIG, SET_INACTIVE, RUN_PENDING + " " + SET_INACTIVE)
    assert not claim_loan_runs_at_the_graph_boundary(n)


def test_mutant_loan_runs_after_the_deferred_frees_fails():
    n = _once_in(SYCL, SYNC_SIG, RUN_PENDING, "")
    n = _once_in(n, SYNC_SIG, DEFERRED_FREES, DEFERRED_FREES + " " + RUN_PENDING)
    assert not claim_loan_runs_at_the_graph_boundary(n)


def test_mutant_loan_boundary_drain_removed_fails():
    """Without the drain the retention outlives the boundary's pass, so the room comes back after the next reserve."""
    assert not claim_loan_boundary_drains_its_room(_once_in(SYCL, SYNC_SIG, LOAN_DRAIN_CALL, ""))


def test_mutant_loan_boundary_drain_after_the_pass_fails():
    n = _once_in(SYCL, SYNC_SIG, LOAN_DRAIN_CALL, "")
    n = _once_in(n, SYNC_SIG, RUN_REFILL, LOAN_DRAIN_CALL + " " + RUN_REFILL)
    assert not claim_loan_boundary_drains_its_room(n)


def test_mutant_loan_boundary_drains_at_every_boundary_fails():
    """A decode boundary would pay the wait, and another context's work on the queue could hold it 100 ms."""
    assert not claim_loan_boundary_drains_its_room(
        _once(SYCL, LOAN_DRAIN_ONLY_ACCEPTED,
              "if (ggml_sycl::drain_retained_handles(true, drain_ms)) { return true; }"))


def test_mutant_loan_boundary_drain_unbounded_fails():
    assert not claim_loan_boundary_drains_its_room(
        _once(SYCL, LOAN_DRAIN_BOUND, "constexpr uint32_t drain_ms = 10000;"))


def test_mutant_runner_reports_a_refused_loan_fails():
    """A BUSY or REFUSED loan lent nothing, so its boundary must not wait."""
    assert not claim_loan_boundary_drains_its_room(
        _once(SYCL, PENDING_LENT, "return ggml_sycl_run_pending_expert_loan_on_device(ctx->device, nullptr, &result);"))


def test_mutant_loan_boundary_timeout_is_silent_fails():
    assert not claim_loan_boundary_drains_its_room(_once(SYCL, LOAN_DRAIN_TRACE, "return false;"))


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
                 "(void) ggml_sycl_expert_loan_transaction_run(ctx.device, 1, 0, false, nullptr, nullptr, false); "
                 + anchor))


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
                     "if (!probe_mode && !refill && !named && (warn_busy || code != ggml_sycl_txn_result::BUSY)) {",
                     "if (!probe_mode && !refill && !named) {")))


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


def test_mutant_backing_not_published_fails():
    """The plan keeps a lent member's device backing: on a mapping load the index reads it as left mapped."""
    assert not claim_probe_changes_nothing_and_a_loan_publishes_once(_once(SYCL, BACKING, ""))


def test_mutant_backing_set_after_the_publication_fails():
    """Set on the plan after it is published, the backing never reaches the plan the ring reads."""
    mutant = _once(_once(SYCL, BACKING, ""), PUBLISH + " " + EPOCH, PUBLISH + " " + BACKING + " " + EPOCH)
    assert not claim_probe_changes_nothing_and_a_loan_publishes_once(mutant)


def test_mutant_backing_arms_swapped_fails():
    """A lent member published FILE_MAPPED and a re-admitted one PINNED: the ring stages nothing it was lent."""
    swapped = BACKING.replace("::FILE_MAPPED", "::@").replace("::PINNED", "::FILE_MAPPED").replace("::@", "::PINNED")
    assert not claim_probe_changes_nothing_and_a_loan_publishes_once(_once(SYCL, BACKING, swapped))


def test_mutant_device_lease_kept_fails():
    """The extra keeps its lease on the withdrawn device copy: the retired entry is never reclaimed."""
    assert not claim_probe_changes_nothing_and_a_loan_publishes_once(_once(SYCL, FORGET, ""))


def test_mutant_lent_views_kept_fails():
    """The extras keep their pointer-table, route-table and hybrid-record views on the withdrawn device entry: it stays
    leased, and the deferred-free pass never reclaims it (llama.cpp-6aj4)."""
    assert not claim_probe_changes_nothing_and_a_loan_publishes_once(_once(SYCL, DROP + " " + FORGET, FORGET))


def test_mutant_lent_views_dropped_after_remember_fails():
    """The views dropped in the remember loop: a remember that throws leaves the later members' views in place."""
    kept = _once(SYCL, DROP + " " + FORGET, FORGET)
    assert not claim_probe_changes_nothing_and_a_loan_publishes_once(_once(kept, REMEMBER, REMEMBER + " " + DROP))


def test_mutant_lent_views_matched_by_resolve_fails():
    """A view matched through a resolve: once the cache generation moves, the retired entry refuses it and the view
    survives the loan."""
    assert not claim_lent_views_dropped_by_key(
        _once_in(SYCL, DROP_SIG, DROP_MATCH,
                 DROP_MATCH.replace("h.key() == withdrawn", "h.resolve(device).on_device && h.key() == withdrawn")))


def test_mutant_lent_views_released_at_once_fails():
    """The transaction lets the dropped views go when it returns: a kernel another context still has on the queue can
    read a pointer table whose expert the deferred-free pass just freed."""
    assert not claim_lent_views_retained_to_a_barrier(_once(SYCL, RETAIN_VIEWS, "lent_views.clear();"))


def test_mutant_table_leases_released_at_once_fails():
    """The helper erases the pointer table's leases on the entry without keeping them: they go at once."""
    assert not claim_lent_views_retained_to_a_barrier(_once_in(SYCL, DROP_SIG, RETAIN_LEASE, ""))


def test_mutant_stale_mark_dropped_fails():
    """The member loses one of its marks: the table stays valid (moe_ptrs_ptr() hands out the old table), or the
    full-local probe keeps a verdict taken while the withdrawn entry was in the table."""
    for mark in STALE_MARKS:
        assert not claim_lent_views_mark_the_table_stale(SYCL, _once_in(COMMON, STALE_SIG, mark, "")), mark


def test_mutant_stale_mark_never_applied_fails():
    """The helper drops views but never marks the table stale: a MUL_MAT_ID trusts a table naming the withdrawn
    entry."""
    assert not claim_lent_views_mark_the_table_stale(
        _once_in(SYCL, DROP_SIG, "if (stored || table_dropped) {", "if (false) {"), COMMON)


def test_mutant_storage_generation_not_bumped_fails():
    """A dropped storage record or slot leaves the generation alone: whatever is keyed on it reads as current."""
    assert not claim_lent_views_mark_the_table_stale(
        _once_in(SYCL, DROP_SIG, "w.note_moe_storage_change(e);", ""), COMMON)


def test_mutant_slot_payload_kept_fails():
    """The slot's handle goes but its raw kernel payload stays: the next table upload carries the freed pointer."""
    assert not claim_lent_views_mark_the_table_stale(_once_in(SYCL, DROP_SIG, NULL_PAYLOAD, ""), COMMON)


def test_mutant_table_lease_drop_sets_no_flag_fails():
    """The preload's holder: only the table's leases name the entry, so with no flag the table stays valid."""
    assert not claim_lent_views_mark_the_table_stale(
        _once_in(SYCL, DROP_SIG, TABLE_FROM_LEASES, "(void) " + LEASE_TAKE + ";"), COMMON)


def test_mutant_lease_flag_inverted_fails():
    """The same holder in another spelling: the flag set when the take moved nothing, so a drop leaves it false."""
    assert not claim_lent_views_mark_the_table_stale(
        _once_in(SYCL, DROP_SIG, TABLE_FROM_LEASES, "table_dropped = " + LEASE_TAKE + " == 0;"), COMMON)


def test_lent_views_found_without_a_walk():
    assert claim_lent_views_found_without_a_walk(SYCL, ROUTE_HPP, COMMON)


def test_mutant_table_leases_walked_fails():
    """The pointer table's leases walked again, beside the keyed take."""
    mutant = _once_in(SYCL, DROP_SIG, TABLE_FROM_LEASES,
                      "for (const ggml_sycl::mem_handle & lease : w.moe_expert_ptrs_leases[device]) { (void) lease; } "
                      + TABLE_FROM_LEASES)
    assert not claim_lent_views_found_without_a_walk(mutant, ROUTE_HPP, COMMON)


def test_mutant_lease_key_drops_the_device_fails():
    """A key without the device takes another card's lease on an entry with the same key."""
    mutant = _once(ROUTE_HPP, LEASE_KEY_OF, LEASE_KEY_OF.replace("h.device()", "0"))
    assert not claim_lent_views_found_without_a_walk(SYCL, mutant, COMMON)


def test_mutant_lease_key_takes_any_handle_fails():
    """A key for a handle that is no weight's indexes table buffers the drop's match never names."""
    mutant = _once(ROUTE_HPP, LEASE_KEY_OF, LEASE_KEY_OF.replace("if (!h.is_weight()) { return false; } ", ""))
    assert not claim_lent_views_found_without_a_walk(SYCL, mutant, COMMON)


def test_mutant_leases_in_a_plain_vector_fails():
    mutant = _once(COMMON, LEASE_MEMBER,
                   "std::vector<ggml_sycl::mem_handle> moe_expert_ptrs_leases[GGML_SYCL_MAX_DEVICES];")
    assert not claim_lent_views_found_without_a_walk(SYCL, ROUTE_HPP, mutant)


def test_mutant_route_table_always_walked_fails():
    mutant = _once_in(SYCL, DROP_SIG, ROUTE_SLOT, "bool route_holds = false;")
    assert not claim_lent_views_found_without_a_walk(mutant, ROUTE_HPP, COMMON)


def test_mutant_route_table_walk_uncounted_fails():
    """A walk the summary never shows: the slot rule could miss on every drop and nothing would say so."""
    mutant = _once_in(SYCL, DROP_SIG, ROUTE_WALK, ROUTE_WALK.replace("++scan_fallbacks; ", ""))
    assert not claim_lent_views_found_without_a_walk(mutant, ROUTE_HPP, COMMON)


def test_mutant_route_table_never_walked_fails():
    """Without the walk a table whose slot does not hold the entry keeps its view of it."""
    mutant = _once_in(SYCL, DROP_SIG, ROUTE_WALK, "")
    assert not claim_lent_views_found_without_a_walk(mutant, ROUTE_HPP, COMMON)


def test_mutant_record_always_walked_fails():
    n = norm(SYCL)
    record = definition(n, RECORD_WITHDRAW_DEF)
    mutant = n.replace(record, record.replace(RECORD_SLOT, "const bool at_slot = false;", 1), 1)
    assert not claim_lent_views_found_without_a_walk(mutant, ROUTE_HPP, COMMON)


def test_mutant_record_walk_uncounted_fails():
    n = norm(SYCL)
    record = definition(n, RECORD_WITHDRAW_DEF)
    mutant = n.replace(record, record.replace(RECORD_WALK, RECORD_WALK.replace("++scan_fallbacks; ", ""), 1), 1)
    assert not claim_lent_views_found_without_a_walk(mutant, ROUTE_HPP, COMMON)


def test_mutant_walks_not_reported_fails():
    mutant = _once_in(SYCL, TXN_SIG, TXN_WALKS, "result.plan_chunks_cloned = next_plan->entries.chunks_cloned();")
    assert not claim_lent_views_found_without_a_walk(mutant, ROUTE_HPP, COMMON)


def test_mutant_route_table_drop_sets_no_flag_fails():
    assert not claim_lent_views_mark_the_table_stale(
        _once_in(SYCL, DROP_SIG, TABLE_FROM_ROUTES, "routes.invalidate(); }"), COMMON)


def test_mutant_hybrid_record_drop_sets_no_flag_fails():
    assert not claim_lent_views_mark_the_table_stale(
        _once_in(SYCL, DROP_SIG, TABLE_FROM_HYBRID, "if (" + RETAIN_HYBRID + ") { }"), COMMON)


def test_mutant_stale_guard_inverted_fails():
    """Inverted, the guard returns for every valid device, so no drop ever marks a table stale."""
    assert not claim_lent_views_mark_the_table_stale(
        SYCL, _once_in(COMMON, STALE_SIG, "if (!ggml_sycl_valid_device_index(dev)) {",
                       "if (ggml_sycl_valid_device_index(dev)) {"))


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
    assert not claim_one_body_every_direction(_once(SYCL, REFILL_ENTRY, REFILL_ENTRY.replace("::REFILL,", "::LOAN,")))


def test_mutant_refill_publishes_twice_fails():
    assert not claim_one_body_every_direction(_once_in(SYCL, TXN_SIG, REFILL_CALL, REFILL_CALL + " " + PUBLISH))


def test_mutant_refill_waits_on_its_copy_fails():
    assert not claim_refill_never_waits(_once(CACHE, REFILL_COPY, REFILL_COPY + " out.copy_events[i].wait();"), SYCL)


def test_mutant_refill_boundary_drains_the_queue_fails():
    assert not claim_refill_never_waits(CACHE, _once(SYCL, BOUNDARY_ARM, "queue->wait_and_throw(); " + BOUNDARY_ARM))


def test_mutant_refill_copies_from_a_raw_pointer_fails():
    mutant = _once_in(CACHE, REFILL_BODY_SIG, REFILL_SRC,
                      "src_owner[i] = make_copy_handle_for_raw_ptr(e.device_ptr, e.size);")
    assert not claim_refill_copy_is_owner_to_owner_on_the_copy_engine(mutant)


def test_mutant_refill_copies_on_the_compute_queue_fails():
    mutant = _once(CACHE, BCS_QUEUE, "sycl::queue & bq = get_queue();")
    assert not claim_refill_copy_is_owner_to_owner_on_the_copy_engine(mutant)


def test_mutant_refill_publishes_the_device_entry_ready_fails():
    mutant = _once_in(CACHE, REFILL_BODY_SIG, ENTRY_EVENTS[0], "entry.state = cache_entry_state::READY;")
    assert not claim_refill_copy_is_owner_to_owner_on_the_copy_engine(mutant)


def test_mutant_refill_copy_ignores_the_host_writes_fails():
    mutant = _once(CACHE, REFILL_COPY, REFILL_COPY.replace(", bq, deps[i]);", ", bq, {});"))
    assert not claim_refill_copy_is_owner_to_owner_on_the_copy_engine(mutant)


def test_mutant_refill_mints_after_the_swap_fails():
    mutant = _once(CACHE, SWAP_MIRROR,
                   SWAP_MIRROR + ' (void) make_direct_entry_handle(key, cache_device, stored, "x");')
    assert not claim_refill_allocates_before_it_changes_anything(mutant)


def test_mutant_refill_builds_a_mapping_node_after_the_swap_fails():
    mutant = _once(CACHE, REMAP_ID, "id_to_key_[key] = expert_device_key(key);")
    assert not claim_refill_allocates_before_it_changes_anything(mutant)


def test_mutant_refill_sizes_the_nodes_under_the_locks_fails():
    sized = _once_in(CACHE, REFILL_BODY_SIG, RELEASED_SIZED, "std::vector<expert_retire_nodes> host_nodes;")
    mutant = _once_in(sized, REFILL_BODY_SIG, EMPLACE, RELEASED_SIZED + " " + EMPLACE)
    assert not claim_refill_allocates_before_it_changes_anything(mutant)


def test_mutant_refill_catch_keeps_the_inserted_entries_fails():
    mutant = _once(CACHE, WITHDRAW_INSERTED, "for (int i = 0; i < 0; ++i) { (void) transition_to_retired_locked(")
    assert not claim_refill_allocates_before_it_changes_anything(mutant)


def test_mutant_refill_publishes_outside_a_planned_materialization_fails():
    assert not claim_refill_publishes_in_a_planned_materialization(_once(CACHE, REFILL_SESSION, ";"))


def test_mutant_refill_opens_its_session_after_the_locks_fails():
    moved = _once(CACHE, REFILL_SESSION, ";")
    mutant = _once_in(moved, REFILL_BODY_SIG, EMPLACE, REFILL_SESSION + " " + EMPLACE)
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
    mutant = _once_in(CACHE, REFILL_BODY_SIG, COHERENCE, ";")
    assert not claim_refill_checks_before_copying(mutant)


def test_mutant_refill_refuses_after_the_copy_fails():
    early = _once_in(CACHE, REFILL_BODY_SIG, REFUSE_EARLY, ";")
    mutant = _once(early, REFILL_COPY, REFILL_COPY + " " + REFUSE_EARLY)
    assert not claim_refill_checks_before_copying(mutant)


def test_mutant_refill_allocates_under_the_inventory_lock_fails():
    moved = _once_in(SYCL, PREP_SIG, DEVICE_ALLOC, "storage = ggml_sycl::mem_handle{};")
    mutant = _once_in(moved, TXN_SIG, INVENTORY_LOCK,
                      INVENTORY_LOCK + " ggml_sycl::mem_handle storage = ggml_sycl_device_expert_alloc(device, 64);")
    assert not claim_refill_lock_order(CACHE, mutant)


def test_mutant_refill_cache_takes_the_inventory_mutex_fails():
    mutant = _once(CACHE, REFILL_LOAD_EFFECT, "std::lock_guard inv(g_tensor_inventory_mutex); "
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
    readmit = ("case ggml_sycl_expert_move_kind::REFILL: ledger.triplets.pop_back(); ledger.triplet_bytes.pop_back(); "
               "break;")
    mutant = _once(SYCL, POP_READMITTED, POP_READMITTED.replace(readmit, "case ggml_sycl_expert_move_kind::REFILL: break;"))
    assert not claim_refill_flips_back_and_forgets_the_host_copy(mutant)


def test_mutant_refill_keeps_the_triplets_bytes_in_the_ledger_fails():
    mutant = _once(SYCL, POP_READMITTED, POP_READMITTED.replace("ledger.triplet_bytes.pop_back(); break;", "break;"))
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


TEMPLATED_INVENTORY_LOCK = "std::unique_lock<ggml_sycl::witnessed_mutex> lock(g_tensor_inventory_mutex);"
TEMPLATED_LEDGER_LOCK = "std::lock_guard<ggml_sycl::witnessed_mutex> lock(g_tensor_inventory_mutex);"


def _templated_inventory_guards(text: str) -> str:
    """`text` with every guard on g_tensor_inventory_mutex naming its template argument."""
    return re.sub(r"std::(lock_guard|unique_lock)(\s+\w+\s*\(\s*g_tensor_inventory_mutex\s*\))",
                  r"std::\1<ggml_sycl::witnessed_mutex>\2", text)


def test_templated_inventory_guards_pass():
    """Every claim that reads the inventory lock passes on the tree with each guard on g_tensor_inventory_mutex
    spelled with its template argument, and the lock-order and ledger mutants still fail there."""
    sycl = _templated_inventory_guards(SYCL)
    assert TEMPLATED_INVENTORY_LOCK in sycl and TEMPLATED_LEDGER_LOCK in sycl
    assert INVENTORY_LOCK not in sycl and LEDGER_LOCK not in sycl
    assert claim_lock_order(CACHE, sycl)
    assert claim_loan_runs_at_the_graph_boundary(sycl)
    assert claim_a_loan_without_host_zones_is_refused(CACHE, sycl)
    assert claim_probe_changes_nothing_and_a_loan_publishes_once(sycl)
    assert claim_exception_publishes_what_was_lent(sycl)
    assert claim_one_body_every_direction(sycl)
    assert claim_refill_lock_order(CACHE, sycl)
    assert claim_ledger_is_guarded_by_the_inventory_mutex(sycl)
    assert claim_an_unpreparable_triplet_leaves_the_ledger(sycl)
    assert claim_an_unengaged_prompts_loan_comes_back_at_once(sycl)
    assert claim_an_ended_contexts_loan_comes_back_at_the_next_context(sycl)
    assert not claim_lock_order(CACHE, _once_in(sycl, TXN_SIG, INVENTORY_LOCK, ""))
    assert not claim_ledger_is_guarded_by_the_inventory_mutex(_once_in(sycl, ARM_SIG, LEDGER_LOCK, ";"))


def test_mutant_boundary_writes_the_ledger_unlocked_fails():
    mutant = _once(SYCL, BOUNDARY_ARM, "g_sycl_expert_lent[device].refill_armed = decode; " + BOUNDARY_ARM)
    assert not claim_ledger_is_guarded_by_the_inventory_mutex(mutant)


def test_mutant_an_unlocked_caller_reconciles_the_ledger_fails():
    call = "ggml_sycl_expert_lent_ledger_reconcile_locked(device, &current); "
    assert not claim_ledger_is_guarded_by_the_inventory_mutex(_once(SYCL, BOUNDARY_ARM, call + BOUNDARY_ARM))


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
    mutant = _once_in(CACHE, REFILL_BODY_SIG, 'refusal = admit ? "not on host" : "not lent";',
                      'refusal = admit ? "not on host" : "not on loan";')
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


# --- the direct backend-taking entries had no production caller (llama.cpp-8xdz M2) ---------------------------------
# The runtime reaches the loan through ggml_sycl_request_expert_loan and the boundary runner, and the refill through the
# decode boundary's ggml_sycl_run_expert_refill_at_boundary. The two [[maybe_unused]] wrappers that took a backend were
# never called, and the device test reaches neither, so they stay deleted rather than carried as dead API.
DEAD_DIRECT_ENTRIES = ("ggml_sycl_run_expert_loan_transaction", "ggml_sycl_run_expert_refill_transaction")


def test_the_unused_direct_loan_and_refill_entries_are_gone():
    for name in DEAD_DIRECT_ENTRIES:
        assert name not in SYCL, name


# --- a lent triplet the refill can never prepare leaves the ledger, each drop WARNed, and one decode graph admits one
# --- refill (llama.cpp-sif1: left in the ledger, a triplet that can never be prepared stops every later refill at
# --- itself; the per-graph cap bounds how much one decode graph re-admits) -----------------------------------------


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
    mutant = _once(SYCL, DROP_BACK, DROP_BACK.replace("ledger.triplet_bytes.pop_back(); };",
                                                      "ledger.triplet_bytes.pop_back(); " + DROP_WARN + " };"))
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


def test_copies_are_sized_from_the_tensor():
    assert claim_copies_are_sized_from_the_tensor(CACHE, SYCL)


def test_storage_minted_before_the_inventory_lock():
    assert claim_storage_minted_before_the_inventory_lock(SYCL)


def test_mutant_loan_copy_sized_from_dst_size_fails():
    """The host copy minted from the plan's dst_size, short of a padded entry's expert or long past it."""
    mutant = _once_in(SYCL, PREP_SIG, HOST_ALLOC, HOST_ALLOC.replace("(device, target.logical_bytes)",
                                                                     "(device, e->dst_size)"))
    assert not claim_copies_are_sized_from_the_tensor(CACHE, mutant)


def test_mutant_refill_storage_sized_from_dst_size_fails():
    mutant = _once_in(SYCL, PREP_SIG, DEVICE_ALLOC,
                      DEVICE_ALLOC.replace("(device, member_bytes)", "(device, e->dst_size)"))
    assert not claim_copies_are_sized_from_the_tensor(CACHE, mutant)


def test_mutant_device_member_minted_at_its_own_staged_size_fails():
    """Each member minted at its own staged size even when its triplet has a slot: an admitted down projection, a few
    grains longer than a gate, splits the block an evicted gate gave back and leaves a hole no member fits."""
    mutant = _once_in(SYCL, PREP_SIG, DEVICE_ALLOC,
                      DEVICE_ALLOC.replace("(device, member_bytes)", "(device, staged[r])"))
    assert not claim_copies_are_sized_from_the_tensor(CACHE, mutant)


def test_mutant_mint_takes_the_slot_even_with_none_fails():
    """The mint asks the slot alone: a triplet with no slot (mixed quant types) would mint 0 bytes."""
    mutant = _once_in(SYCL, PREP_SIG, DEVICE_MEMBER,
                      "const size_t member_bytes = ggml_sycl::expert_triplet_slot_bytes(staged);")
    assert not claim_copies_are_sized_from_the_tensor(CACHE, mutant)


def test_mutant_cache_accepts_a_member_at_its_own_staged_size_fails():
    """The cache's check against each member's own staged size: a device copy minted short of the slot published."""
    mutant = _once_in(CACHE, REFILL_BODY_SIG, "if (device_size[i] != member_bytes ||",
                      "if (device_size[i] != staged[i] ||")
    assert not claim_copies_are_sized_from_the_tensor(mutant, SYCL)


def test_mutant_refill_device_bytes_from_the_host_copy_fails():
    """The device size taken from the host copy's bytes rather than staged from them: the padded tail the kernels
    read past an expert's own bytes is missing."""
    mutant = _once_in(SYCL, PREP_SIG, DEVICE_BYTES, "staged[r] = target.logical_bytes;")
    assert not claim_copies_are_sized_from_the_tensor(CACHE, mutant)


def test_mutant_refill_copies_the_host_copy_whole_fails():
    """The refill copying and sizing from the host copy whole, as it did before every direction sized from the tensor:
    a host copy minted larger than the expert would grow it on the way back."""
    mutant = _once_in(CACHE, REFILL_BODY_SIG, "const size_t logical = members[i].logical_bytes;",
                      "const size_t logical = e.size;")
    assert not claim_copies_are_sized_from_the_tensor(mutant, SYCL)


def test_mutant_refill_runs_the_admit_checks_fails():
    """The refill's wrapper passing admit true: any pinned host copy, not only a loan's, would be re-admitted."""
    mutant = _once_in(CACHE, REFILL_SIG, "(members, false, out)", "(members, true, out)")
    assert not claim_refill_checks_before_copying(mutant)


def test_mutant_missing_entry_does_not_stop_fails():
    """No entry to copy from must stop the preparation with the cache's refusal, not mint a copy of nothing."""
    mutant = _once_in(SYCL, PREP_SIG, SOURCE_CHECK, "")
    assert not claim_copies_are_sized_from_the_tensor(CACHE, mutant)


def test_mutant_source_query_finds_a_retired_entry_fails():
    mutant = _once_in(CACHE, SOURCE_SIG, SOURCE_RETURN, "return it != entries_.end();")
    assert not claim_copies_are_sized_from_the_tensor(mutant, SYCL)


def test_mutant_source_query_reads_one_side_fails():
    """Both directions looking up the same key: a refill would ask for the device entry, a loan for the host's."""
    mutant = _once_in(CACHE, SOURCE_SIG, "to_device ? expert_host_key(key)", "!to_device ? expert_host_key(key)")
    assert not claim_copies_are_sized_from_the_tensor(mutant, SYCL)


def test_mutant_source_query_sizes_the_copy_fails():
    """The query handing back the entry's size, which direct staging may have padded, for the preparation to mint."""
    mutant = _once_in(CACHE, SOURCE_SIG, SOURCE_RETURN,
                      "return it != entries_.end() && !it->second.retired && it->second.size != 0;")
    assert not claim_copies_are_sized_from_the_tensor(mutant, SYCL)


def test_mutant_copy_minted_before_the_source_query_fails():
    """The copies minted first and the source asked after: the order the claim pins, witnessed."""
    mutant = _once_in(SYCL, PREP_SIG, SOURCE_CHECK, "")
    mutant = _once_in(mutant, PREP_SIG, HOST_ALLOC, HOST_ALLOC + " " + SOURCE_CHECK)
    assert not claim_copies_are_sized_from_the_tensor(CACHE, mutant)


def test_mutant_transaction_sizes_a_copy_fails():
    mutant = _once_in(SYCL, TXN_SIG, PREPARE, "(void) ggml_sycl::staged_expert_bytes(GGML_LAYOUT_AOS, "
                      "GGML_TYPE_Q4_0, 32, 18); " + PREPARE)
    assert not claim_copies_are_sized_from_the_tensor(CACHE, mutant)


def test_mutant_query_without_the_direct_stage_lock_fails():
    mutant = _once_in(CACHE, SOURCE_SIG, SHARED_CACHE_LOCKS, "std::shared_lock<std::shared_mutex> cache_lock(rw_mutex_);")
    assert not claim_copies_are_sized_from_the_tensor(mutant, SYCL)


def test_mutant_loan_entry_guard_removed_fails():
    """A device entry shorter than the expert's bytes copied past its end."""
    mutant = _once_in(CACHE, LOAN_SIG, LOAN_ENTRY_GUARD, "")
    assert not claim_copies_are_sized_from_the_tensor(mutant, SYCL)


def test_mutant_loan_size_guard_removed_fails():
    mutant = _once_in(CACHE, LOAN_SIG, LOAN_SIZE_GUARD, "")
    assert not claim_copies_are_sized_from_the_tensor(mutant, SYCL)


def test_mutant_loan_copies_the_device_entry_whole_fails():
    """The loan copying the device entry whole, its padded tail with it, into a host copy minted at the expert's bytes:
    past the copy's end, and an expert that comes back grows."""
    mutant = _once_in(CACHE, LOAN_SIG, LOAN_COPY_BYTES, "bytes[i] = e.size; device_size[i] = e.size;")
    assert not claim_copies_are_sized_from_the_tensor(mutant, SYCL)


def test_mutant_refill_size_guard_removed_fails():
    mutant = _once_in(CACHE, REFILL_BODY_SIG, REFILL_SIZE_GUARD, "")
    assert not claim_copies_are_sized_from_the_tensor(mutant, SYCL)


# llama.cpp-7999: the lent ledger follows the publication. A plan published under another identity, or no plan, empties
# every device's ledger right after the publication is written, so the ring sizing that reads the ledger's lent bytes
# never counts a previous plan's loan back. The loan's own transaction reconciles too, through the ledger accessor.
PUBLISH_DEF = ("static void ggml_sycl_publish_prepared_plan_locked(ggml_sycl_prepared_plan_publication & publication) "
               "noexcept {")
PUBLICATION_WRITE = "std::atomic_store_explicit(&g_placement_publication, snapshot, std::memory_order_release); });"
RECONCILE_EVERY_DEVICE = ("for (int d = 0; d < GGML_SYCL_MAX_DEVICES; ++d) { "
                          "ggml_sycl_expert_lent_ledger_reconcile_locked(d, snapshot.get()); }")
LEDGER_ACCESSOR_DEF = "static ggml_sycl_expert_lent_ledger & ggml_sycl_expert_lent_ledger_locked("
LEDGER_ACCESSOR_RECONCILE = "ggml_sycl_expert_lent_ledger_reconcile_locked(device, &current);"


def _definition_body(n: str, definition: str, signature: str) -> str:
    """The body of the definition that starts with `definition`, past any forward declaration of the same name."""
    at = n.find(definition)
    return body(n[at:], signature) if at >= 0 else ""


def claim_the_ledger_follows_the_publication(sycl: str) -> bool:
    n = norm(sycl)
    publish = _definition_body(n, PUBLISH_DEF, "static void ggml_sycl_publish_prepared_plan_locked(")
    accessor = _definition_body(n, LEDGER_ACCESSOR_DEF, LEDGER_ACCESSOR_DEF)
    return (n.count(PUBLISH_DEF) == 1 and publish.count(RECONCILE_EVERY_DEVICE) == 1
            and n.count("ggml_sycl_expert_lent_ledger_reconcile_locked(d, ") == 1
            and ordered(publish, "return;", PUBLICATION_WRITE, RECONCILE_EVERY_DEVICE)
            and accessor.count(LEDGER_ACCESSOR_RECONCILE) == 1)


def test_the_ledger_follows_the_publication():
    assert claim_the_ledger_follows_the_publication(SYCL)


def test_mutant_publication_leaves_the_ledger_fails():
    assert not claim_the_ledger_follows_the_publication(_once(SYCL, RECONCILE_EVERY_DEVICE, "{ }"))


def test_mutant_ledger_reconciled_before_the_publication_write_fails():
    n = _once(SYCL, RECONCILE_EVERY_DEVICE, "")
    n = n.replace("ggml_sycl_tenant_publish_gen_bump(); const auto & snapshot = publication.snapshot;",
                  "ggml_sycl_tenant_publish_gen_bump(); const auto & snapshot = publication.snapshot; "
                  + RECONCILE_EVERY_DEVICE, 1)
    assert n.count(RECONCILE_EVERY_DEVICE) == 1, "the mutant must move the reconcile, not drop it"
    assert not claim_the_ledger_follows_the_publication(n)


def test_mutant_ledger_accessor_skips_the_reconcile_fails():
    assert not claim_the_ledger_follows_the_publication(_once(SYCL, LEDGER_ACCESSOR_RECONCILE, ""))


# --- the loan is counted in ring slots (llama.cpp-9x8t) -------------------------------------------------------------

# The ring's slot is its largest host matrix, so on a model that mixes expert quants a lent small-quant triplet frees
# three holes each below a slot: it funds no slot, and the ring comes back for another loan at another boundary. The
# request carries the ring's slot size beside its bytes, the boundary runner hands both to the transaction, and the
# selector counts what each matrix frees in whole slots.
SLOT_SIZES = "static std::atomic<size_t> g_sycl_expert_loan_slot_bytes[GGML_SYCL_MAX_DEVICES];"
REQUEST_DEF = "static void ggml_sycl_request_expert_loan(int device, size_t bytes_wanted, size_t slot_bytes) {"
SLOT_STORE = "g_sycl_expert_loan_slot_bytes[device].store(slot_bytes, std::memory_order_release);"
SLOT_LOAD = "const size_t slot_bytes = g_sycl_expert_loan_slot_bytes[device].load(std::memory_order_acquire);"
SLOT_RESET = "g_sycl_expert_loan_slot_bytes[device].store(0, std::memory_order_release);"
REQUEST_RESET = "g_sycl_expert_loan_requests[device].store(0, std::memory_order_release);"


def claim_the_request_carries_the_slot_size(sycl: str) -> bool:
    """The request stores its slot size before its bytes, whose release publishes it; the runner reads it after it
    has taken the bytes with acquire, so it reads a size at least as new as the one those bytes were asked in (a
    request posted after the take can overwrite it, harmlessly). The runner hands it to the loan transaction, which
    hands it to the selector, and a BUSY re-post carries it through. The plan reset that clears the request clears it
    too. A request that does not win the bytes still stores its size: the ring's geometry is constant between two graph
    boundaries (prepare() rebuilds it per micro-batch; a loan can grow it, llama.cpp-8p5x), and one micro-batch posts
    the requests in that interval, so they agree on it."""
    s = norm(sycl)
    request = body(s, REQUEST_SIG)
    pending = body(s, PENDING_DEVICE_SIG)
    return (s.count(SLOT_SIZES) == 1 and s.count(REQUEST_DEF) == 1
            and request.count(SLOT_STORE) == 1 and ordered(request, SLOT_STORE, REQUEST_CAS)
            and s.count("g_sycl_expert_loan_slot_bytes[device].store(") == 2
            and s.count(SLOT_RESET) == 1 and ordered(s, REQUEST_RESET, SLOT_RESET)
            and pending.count(SLOT_LOAD) == 1 and ordered(pending, PENDING_TAKE, SLOT_LOAD, PENDING_RUN, REPOST)
            and body(s, LOAN_TXN_SIG).count(LOAN_ENTRY) == 1 and body(s, TXN_SIG).count(LOAN_SELECT) == 1)


def test_the_request_carries_the_slot_size():
    assert claim_the_request_carries_the_slot_size(SYCL)


def test_mutant_slot_size_stored_after_the_bytes_fails():
    """Stored after the bytes publish, the size can reach the runner after it took them: it would count the loan in
    the previous request's slots, or in bytes."""
    n = _once_in(SYCL, REQUEST_SIG, SLOT_STORE + " ", "")
    request = body(n, REQUEST_SIG)
    n = n.replace(request, request[:-1] + SLOT_STORE + " }", 1)
    assert n.count(SLOT_STORE) == 1, "the mutant must move the store, not drop it"
    assert not claim_the_request_carries_the_slot_size(n)


def test_mutant_slot_size_stored_relaxed_fails():
    assert not claim_the_request_carries_the_slot_size(
        _once(SYCL, SLOT_STORE, SLOT_STORE.replace("memory_order_release", "memory_order_relaxed")))


def test_mutant_slot_size_read_before_the_bytes_are_taken_fails():
    n = _once(SYCL, SLOT_LOAD + " ", "")
    n = n.replace(PENDING_TAKE, SLOT_LOAD + " " + PENDING_TAKE, 1)
    assert n.count(SLOT_LOAD) == 1, "the mutant must move the load, not drop it"
    assert not claim_the_request_carries_the_slot_size(n)


def test_mutant_runner_drops_the_slot_size_fails():
    """The runner asks in bytes: a lent small-quant triplet funds no slot again."""
    assert not claim_the_request_carries_the_slot_size(
        _once(SYCL, PENDING_RUN, PENDING_RUN.replace("bytes_wanted, slot_bytes, false", "bytes_wanted, 0, false")))


def test_mutant_busy_repost_drops_the_slot_size_fails():
    """A BUSY loan re-posted without its slot size runs in bytes at the next boundary."""
    mutant = _once(SYCL, REPOST, REPOST.replace("bytes_wanted, slot_bytes)", "bytes_wanted, 0)"))
    assert not claim_the_request_carries_the_slot_size(mutant)
    assert not claim_busy_reposts_the_request(mutant)


def test_mutant_loan_entry_drops_the_slot_size_fails():
    assert not claim_the_request_carries_the_slot_size(
        _once(SYCL, LOAN_ENTRY, LOAN_ENTRY.replace("bytes_wanted, slot_bytes, probe_mode",
                                                   "bytes_wanted, 0, probe_mode")))


def test_mutant_transaction_selects_in_bytes_fails():
    assert not claim_the_request_carries_the_slot_size(
        _once(SYCL, LOAN_SELECT, LOAN_SELECT.replace("amount, slot_bytes, &selected", "amount, 0, &selected")))


def test_mutant_plan_reset_keeps_the_slot_size_fails():
    assert not claim_the_request_carries_the_slot_size(_once(SYCL, SLOT_RESET, ""))


SELECT_DEF = ("static inline std::vector<expert_placement_key> moe_expert_loan_select(const placement_plan & plan, "
              "int device, size_t bytes_wanted, size_t slot_bytes, size_t * bytes_selected, "
              "size_t * slots_selected = nullptr)")
SLOTS_WANTED = ("const size_t slots_wanted = slot_bytes != 0 ? bytes_wanted / slot_bytes + "
                "(bytes_wanted % slot_bytes != 0 ? 1 : 0) : 0;")
SLOT_WALK = ("for (size_t i = plan.entries.size(); i-- > 0 && "
             "(slot_bytes != 0 ? slots < slots_wanted : sum < bytes_wanted);) {")
SELECT_LENDABLE = "!detail::moe_loan_triplet_is_lendable(t, plan, bulk_memo)) { continue; }"
SLOT_BRANCH = "if (slot_bytes != 0) { size_t triplet_slots = 0; for (int r = 0; r < 3; ++r) {"
SLOT_PER_MATRIX = "triplet_slots += t.e[r]->vram_charge_size / slot_bytes; }"
SLOT_SKIP = "if (triplet_slots == 0) { continue; }"
SLOT_ADD = "slots += triplet_slots; }"
BYTES_ADD = "sum += t.e[r]->vram_charge_size;"
BYTES_OUT = "if (bytes_selected) { *bytes_selected = sum; }"
SLOTS_OUT = "if (slots_selected) { *slots_selected = slots; }"


def claim_the_selector_counts_whole_slots_per_matrix(hpp: str) -> bool:
    """With a slot size, the walk stops on slots, not bytes: it wants the shortfall's slots rounded up, each matrix
    frees vram_charge_size / slot_bytes of them (the floor per hole, never the floor of the triplet's sum), and a
    triplet that frees none is skipped after the lendable checks and before any key is taken. *bytes_selected stays
    the byte sum, *slots_selected is the funded slot count the probe multiplies by the slot size,
    and with slot_bytes 0 the walk counts bytes as before and the count stays 0."""
    sel = body(norm(hpp), SELECT_DEF)
    return (sel.count(SLOT_PER_MATRIX) == 1 and sel.count(SLOT_SKIP) == 1 and sel.count(BYTES_ADD) == 1
            and ordered(sel, SLOTS_WANTED, SLOT_WALK, SELECT_LENDABLE, SLOT_BRANCH, SLOT_PER_MATRIX, SLOT_SKIP,
                        SLOT_ADD, "out.push_back(", BYTES_ADD, BYTES_OUT, SLOTS_OUT))


def test_the_selector_counts_whole_slots_per_matrix():
    assert claim_the_selector_counts_whole_slots_per_matrix(PLAN_HPP)


def test_mutant_selector_walks_until_the_bytes_fails():
    """The walk stopped on the byte sum: freed bytes reach the shortfall before freed slots do."""
    assert not claim_the_selector_counts_whole_slots_per_matrix(
        _once(PLAN_HPP, SLOT_WALK, "for (size_t i = plan.entries.size(); i-- > 0 && sum < bytes_wanted;) {"))


def test_mutant_selector_lends_a_triplet_that_frees_no_slot_fails():
    assert not claim_the_selector_counts_whole_slots_per_matrix(_once(PLAN_HPP, SLOT_SKIP, ""))


def test_mutant_selector_floors_the_triplet_sum_fails():
    """The floor of the summed bytes over-counts: three holes of half a slot read as one slot and fund none."""
    n = _once(PLAN_HPP, SLOT_PER_MATRIX, "triplet_slots += t.e[r]->vram_charge_size; }")
    n = n.replace(SLOT_SKIP, "triplet_slots /= slot_bytes; " + SLOT_SKIP, 1)
    assert not claim_the_selector_counts_whole_slots_per_matrix(n)


def test_mutant_selector_reports_the_byte_sum_as_the_slot_count_fails():
    """The probe multiplies the count by the slot size, so a byte sum reported there would credit the remainders."""
    assert not claim_the_selector_counts_whole_slots_per_matrix(
        _once(PLAN_HPP, SLOTS_OUT, "if (slots_selected) { *slots_selected = sum; }"))


def test_mutant_selector_drops_the_slot_count_fails():
    assert not claim_the_selector_counts_whole_slots_per_matrix(_once(PLAN_HPP, SLOTS_OUT, ""))


def test_mutant_transaction_drops_the_slot_count_from_the_probe_fails():
    assert not claim_probe_changes_nothing_and_a_loan_publishes_once(
        _once(SYCL, "result.slots_funded = slots_funded;", ""))


def test_mutant_selector_rounds_the_shortfall_down_fails():
    """A shortfall one byte past a slot would be asked as one slot, and the ring would come back for the other."""
    assert not claim_the_selector_counts_whole_slots_per_matrix(
        _once(PLAN_HPP, SLOTS_WANTED, "const size_t slots_wanted = slot_bytes != 0 ? bytes_wanted / slot_bytes : 0;"))


# --- an unengaged prompt's loan comes back at once: mutants (llama.cpp-i4ny) -----------------------------------------

def test_mutant_owe_marks_the_ledger_unlocked_fails():
    mutant = _once_in(SYCL, OWE_SIG, LEDGER_LOCK, ";")
    assert not claim_an_unengaged_prompts_loan_comes_back_at_once(mutant)
    assert not claim_ledger_is_guarded_by_the_inventory_mutex(mutant)


def test_mutant_take_reads_the_ledger_unlocked_fails():
    mutant = _once_in(SYCL, TAKE_SIG, LEDGER_LOCK, ";")
    assert not claim_an_unengaged_prompts_loan_comes_back_at_once(mutant)
    assert not claim_ledger_is_guarded_by_the_inventory_mutex(mutant)


def test_mutant_owe_marks_an_empty_ledger_fails():
    mutant = _once(SYCL, OWE_SET, "ledger.bulk_refill_owed_at = first_graph;")
    assert not claim_an_unengaged_prompts_loan_comes_back_at_once(mutant)


def test_mutant_take_leaves_the_mark_fails():
    mutant = _once_in(SYCL, TAKE_SIG, "ledger.bulk_refill_owed_at = 0;", "")
    assert not claim_an_unengaged_prompts_loan_comes_back_at_once(mutant)


def test_mutant_reconcile_keeps_the_mark_fails():
    mutant = _once(SYCL, RECONCILE_CLEAR, "ledger.refill_armed = false;")
    assert not claim_an_unengaged_prompts_loan_comes_back_at_once(mutant)


def test_mutant_runner_never_takes_the_mark_fails():
    mutant = _once(SYCL, "const uint64_t owed_at = lent_any ? ggml_sycl_expert_bulk_refill_take(device) : 0;",
                   "const uint64_t owed_at = 0;")
    assert not claim_an_unengaged_prompts_loan_comes_back_at_once(mutant)


def test_mutant_runner_takes_the_mark_with_nothing_lent_asked_fails():
    mutant = _once(SYCL, "lent_any ? ggml_sycl_expert_bulk_refill_take(device) : 0;",
                   "ggml_sycl_expert_bulk_refill_take(device);")
    assert not claim_an_unengaged_prompts_loan_comes_back_at_once(mutant)


def test_mutant_prompt_graph_runs_the_bulk_refill_fails():
    mutant = _once(SYCL, "if (!decode || owed_at == 0 ||", "if (owed_at == 0 ||")
    assert not claim_an_unengaged_prompts_loan_comes_back_at_once(mutant)


def test_mutant_bulk_refill_at_the_per_graph_count_fails():
    mutant = _once_in(SYCL, REFILLS_SIG, "refilled_at, lent, resolver, result);",
                      "refilled_at, max_triplets, resolver, result);")
    assert not claim_an_unengaged_prompts_loan_comes_back_at_once(mutant)


def test_mutant_bulk_refill_warns_when_refused_fails():
    mutant = _once_in(SYCL, REFILLS_SIG, BULK_GUARD, "if (true) {")
    assert not claim_an_unengaged_prompts_loan_comes_back_at_once(mutant)


def test_mutant_bulk_refill_line_drops_the_position_fails():
    mutant = _once(SYCL, "(unsigned long long) (decode_graphs - owed_at + 1)", "(unsigned long long) decode_graphs")
    assert not claim_an_unengaged_prompts_loan_comes_back_at_once(mutant)


def test_mutant_boundary_skips_the_bulk_refill_fails():
    mutant = _once_in(SYCL, BOUNDARY_SIG, "ggml_sycl_run_expert_refills_at_boundary_on_device(",
                      "ggml_sycl_run_expert_refill_at_boundary_on_device(")
    assert not claim_an_unengaged_prompts_loan_comes_back_at_once(mutant)


def test_mutant_owe_back_under_the_graph_lock_fails():
    """The owe at graph compute's entry, under the graph lock, as it was: L1 under L2."""
    moved = _once_in(SYCL, REFILLS_SIG, OWE_FROM_PHASE, "")
    mutant = _once(moved, GRAPH_LOCK, GRAPH_LOCK + " ggml_sycl_expert_bulk_refill_owe(sycl_ctx->device, "
                   "sycl_ctx->decode_graph_count.load(std::memory_order_relaxed) + 1);")
    assert not claim_the_debt_is_never_marked_under_the_graph_lock(mutant)
    assert not claim_an_unengaged_prompts_loan_comes_back_at_once(mutant)


def test_mutant_next_contexts_half_under_the_graph_lock_fails():
    moved = _once(SYCL, ENTRY_CALL, "")
    mutant = _once(moved, GRAPH_LOCK, GRAPH_LOCK + " " + ENTRY_CALL)
    assert not claim_the_debt_is_never_marked_under_the_graph_lock(mutant)


REPLAY_LOCK_HELD = "GGML_ASSERT(global_graph_lock.owns_lock());"


def test_mutant_ledger_call_on_the_replay_path_under_the_graph_lock_fails():
    """Past the compute lambda's unlock, the replay path still holds the graph lock (it asserts so): a ledger call there
    is L1 under L2 too, whichever of the debt's calls it is."""
    for call in ("(void) ggml_sycl_expert_bulk_refill_owe_next_context(sycl_ctx->device);",
                 "(void) ggml_sycl_expert_bulk_refill_owed_next_context(sycl_ctx->device);",
                 "ggml_sycl_expert_bulk_refill_owe(sycl_ctx->device, 1);"):
        mutant = _once_in(SYCL, GRAPH_COMPUTE_SIG, REPLAY_LOCK_HELD, REPLAY_LOCK_HELD + " " + call)
        assert not claim_the_debt_is_never_marked_under_the_graph_lock(mutant), call


def test_mutant_owe_at_the_next_decode_graph_fails():
    """Graph compute has already counted the graph whose boundary this is: one more would owe the next one."""
    mutant = _once(SYCL, "ggml_sycl_expert_bulk_refill_owe(device, decode_graphs);",
                   "ggml_sycl_expert_bulk_refill_owe(device, decode_graphs + 1);")
    assert not claim_an_unengaged_prompts_loan_comes_back_at_once(mutant)


def test_mutant_owe_after_an_engaged_prompt_fails():
    mutant = _once(SYCL, "&& !phase->last_prompt_ring_engaged()", "&& phase->last_prompt_ring_engaged()")
    assert not claim_an_unengaged_prompts_loan_comes_back_at_once(mutant)


def test_mutant_owe_after_the_take_fails():
    moved = _once_in(SYCL, REFILLS_SIG, OWE_FROM_PHASE, "")
    mutant = _once(moved, "lent_any ? ggml_sycl_expert_bulk_refill_take(device) : 0;",
                   "lent_any ? ggml_sycl_expert_bulk_refill_take(device) : 0; " + OWE_FROM_PHASE)
    assert not claim_an_unengaged_prompts_loan_comes_back_at_once(mutant)


def test_mutant_bulk_refill_line_reads_as_the_exit_report_fails():
    """A bulk line that also reads in the exit report's shape satisfies RE_REFILL, which would count it as the report:
    the line keeps its decode graph 1 match, so only the negative control can fail it."""
    mutant = _once_in(SYCL, REFILLS_SIG, "(its ring did not engage)",
                      "(its ring did not engage) [MOE-EXEC] refill device 0: 0/0 triplets re-admitted")
    assert re.search(gate_regex(GATE, "RE_BULK_REFILL_FIRST"), bulk_refill_line(mutant))
    assert not claim_bulk_refill_line_scores_as_the_gate_reads_it(mutant, GATE)


def test_mutant_bulk_refill_line_at_another_position_fails():
    mutant = _once_in(SYCL, REFILLS_SIG, "at decode graph %llu after the prompt",
                      "at decode graph %llu since the prompt")
    assert not claim_bulk_refill_line_scores_as_the_gate_reads_it(mutant, GATE)


def test_mutant_gate_exit_report_regex_reads_the_bulk_line_fails():
    exit_report = "RE_REFILL='\\[MOE-EXEC\\] refill device [0-9]+: [0-9]+/[0-9]+ triplets re-admitted'"
    assert GATE.count(exit_report) == 1
    mutant = GATE.replace(exit_report, "RE_REFILL='\\[MOE-EXEC\\] (bulk )?refill device [0-9]+: '")
    assert not claim_bulk_refill_line_scores_as_the_gate_reads_it(SYCL, mutant)


def test_mutant_gate_wants_another_position_fails():
    first = 'RE_BULK_REFILL_FIRST="$RE_BULK_REFILL at decode graph 1 after the prompt"'
    assert GATE.count(first) == 1
    mutant = GATE.replace(first, first.replace("graph 1 ", "graph 2 "))
    assert not claim_bulk_refill_line_scores_as_the_gate_reads_it(SYCL, mutant)


# --- a loan an ended context leaves lent comes back at the next context: mutants (llama.cpp-i4ny) ------------------

def _ended(mutant: str) -> bool:
    return claim_an_ended_contexts_loan_comes_back_at_the_next_context(mutant)


def test_mutant_free_owes_unlocked_fails():
    mutant = _once_in(SYCL, OWE_NEXT_SIG, LEDGER_LOCK, ";")
    assert not _ended(mutant)
    assert not claim_ledger_is_guarded_by_the_inventory_mutex(mutant)


def test_mutant_report_reads_the_debt_unlocked_fails():
    mutant = _once_in(SYCL, OWED_NEXT_SIG, LEDGER_LOCK, ";")
    assert not _ended(mutant)
    assert not claim_ledger_is_guarded_by_the_inventory_mutex(mutant)


def test_mutant_next_context_takes_the_debt_unlocked_fails():
    mutant = _once_in(SYCL, BETWEEN_SIG, LEDGER_LOCK, ";")
    assert not _ended(mutant)
    assert not claim_ledger_is_guarded_by_the_inventory_mutex(mutant)


def test_mutant_free_owes_an_empty_ledger_fails():
    assert not _ended(_once(SYCL, "ledger.bulk_refill_next_context = !ledger.triplets.empty();",
                            "ledger.bulk_refill_next_context = true;"))


def test_mutant_next_context_keeps_the_debt_fails():
    assert not _ended(_once_in(SYCL, BETWEEN_SIG, "ledger.bulk_refill_next_context = false;", ""))


def test_mutant_reconcile_keeps_the_debt_fails():
    assert not _ended(_once(SYCL, RECONCILE_NEXT, "ledger.bulk_refill_owed_at = 0;"))


def test_mutant_every_graph_entry_takes_the_lock_fails():
    assert not _ended(_once(SYCL, BETWEEN_SKIP, ""))


def test_mutant_next_context_skips_the_deferred_frees_fails():
    assert not _ended(_once_in(SYCL, BETWEEN_SIG, DEFERRED_FREES, ";"))


def test_mutant_next_context_refills_one_triplet_fails():
    assert not _ended(_once_in(SYCL, BETWEEN_SIG, "&refilled_at, lent, resolver, result);",
                               "&refilled_at, 1, resolver, result);"))


def test_mutant_next_context_warns_when_refused_fails():
    assert not _ended(_once_in(SYCL, BETWEEN_SIG, BULK_GUARD, "if (true) {"))


def test_mutant_free_owes_after_the_report_fails():
    moved = _once(SYCL, FREE_OWE, "")
    assert not _ended(_once(moved, FREE_REPORTS, FREE_REPORTS + " " + FREE_OWE))


def test_mutant_temporary_backend_owes_a_live_contexts_loan_fails():
    assert not _ended(_once(SYCL, "if (sycl_ctx && sycl_ctx->graph_compute_seq != 0) {", "if (sycl_ctx) {"))


def test_mutant_next_context_refills_under_the_graph_lock_fails():
    moved = _once(SYCL, ENTRY_CALL, "")
    assert not _ended(_once(moved, GRAPH_LOCK, GRAPH_LOCK + " " + ENTRY_CALL))


def test_mutant_graph_entry_never_refills_fails():
    assert not _ended(_once(SYCL, ENTRY_CALL, ""))


def test_mutant_report_hides_the_debt_fails():
    assert not _ended(_once(SYCL, OWED_WARN, TEARDOWN_WARN))


def test_mutant_debt_report_loses_the_exit_report_prefix_fails():
    """The positive control: a debt line the gate's RE_REFILL no longer reads would drop out of refill_state."""
    mutant = _once(SYCL, "triplets re-admitted (%u owed", "triplets readmitted (%u owed")
    assert not claim_debt_lines_score_as_the_gate_reads_them(mutant, GATE)


def test_mutant_between_contexts_line_reads_as_decode_graph_1_fails():
    mutant = _once_in(SYCL, BETWEEN_SIG, "in one transaction between contexts (the context that lent them ended)",
                      "in one transaction at decode graph 1 after the prompt")
    assert not claim_debt_lines_score_as_the_gate_reads_them(mutant, GATE)


def test_mutant_between_contexts_line_reads_as_the_exit_report_fails():
    mutant = _once_in(SYCL, BETWEEN_SIG, "(the context that lent them ended)",
                      "(the context that lent them ended) [MOE-EXEC] refill device 0: 0/0 triplets re-admitted")
    assert not claim_debt_lines_score_as_the_gate_reads_them(mutant, GATE)


# --- a moved copy belongs to its source's models: mutants -----------------------------------------------------------

def test_a_moved_copy_belongs_to_its_sources_models():
    assert claim_a_moved_copy_belongs_to_its_sources_models(CACHE)


def test_mutant_helper_drops_the_tag_fails():
    """Owners copied without the tag: a copy whose owners all die reads as never attributed and is kept forever."""
    assert not claim_a_moved_copy_belongs_to_its_sources_models(
        _once(CACHE, "dst.owner_tagged = dst.owner_tagged || src.owner_tagged; ", ""))


def test_mutant_admit_publishes_untagged_fails():
    assert not claim_a_moved_copy_belongs_to_its_sources_models(_once(CACHE, ADMIT_INHERIT + " ", ""))


def test_mutant_admit_tags_after_the_insert_fails():
    """The inserted entry is a copy of device_entry[i]: tagging the local after the insert tags nothing. The call is
    moved, not dropped, so only the ordering clause (inside the second lock, before the insert) catches it."""
    moved = _once(CACHE, ADMIT_INHERIT + " ", "")
    assert not claim_a_moved_copy_belongs_to_its_sources_models(_once(moved, EMPLACE, EMPLACE + " " + ADMIT_INHERIT))


def test_mutant_admit_mirror_keeps_the_stamp_fails():
    """A mirror whose pending load differs from its entry's survives that load's abort, naming a retired entry."""
    assert not claim_a_moved_copy_belongs_to_its_sources_models(_once(CACHE, " " + ADMIT_MIRROR, ""))


def test_mutant_registration_ignores_the_source_fails():
    assert not claim_a_moved_copy_belongs_to_its_sources_models(_once(CACHE, REGISTER_INHERIT, ""))


def test_mutant_registration_tags_after_the_insert_fails():
    moved = _once(CACHE, REGISTER_INHERIT, "")
    assert not claim_a_moved_copy_belongs_to_its_sources_models(
        _once_in(moved, REGISTER_BODY_SIG, REGISTER_INSERT, REGISTER_INSERT + " " + REGISTER_INHERIT))


def test_mutant_registration_mirror_keeps_the_stamp_fails():
    assert not claim_a_moved_copy_belongs_to_its_sources_models(
        _once(CACHE, REGISTER_MIRROR_TXN, "entry.pending_load_txn_id = load_effect_guard.load_txn_id();"))


def test_mutant_owned_registration_drops_the_source_fails():
    """The typed overload forwards to the body without the source: every loan's host copy is published untagged."""
    assert not claim_a_moved_copy_belongs_to_its_sources_models(
        _once(CACHE, "allocation_owner, ready_event, replace_existing, moved_from);",
              "allocation_owner, ready_event, replace_existing);"))


def test_mutant_loan_names_no_source_fails():
    assert not claim_a_moved_copy_belongs_to_its_sources_models(
        _once(CACHE, REGISTER, REGISTER.replace(", &out.device_keys[i])", ", nullptr)")))


def test_mutant_loan_names_its_source_after_the_registration_fails():
    """The registration reads the key it is handed as the source: a key set only after the call names no entry."""
    moved = _once_in(CACHE, LOAN_SIG, " " + LOAN_DEVICE_KEY, "")
    assert not claim_a_moved_copy_belongs_to_its_sources_models(
        _once_in(moved, LOAN_SIG, REGISTER + ";", REGISTER + "; " + LOAN_DEVICE_KEY))


# --- a missing source refuses before the insert -----------------------------------------------------------------------

def test_a_missing_source_refuses_before_the_insert():
    assert claim_a_missing_source_refuses_before_the_insert(CACHE)


def test_mutant_missing_source_publishes_untagged_fails():
    """The WARN without the refusal: the copy is inserted and mirrored owned by nobody, and outlives its model."""
    assert not claim_a_missing_source_refuses_before_the_insert(
        _once_in(CACHE, REGISTER_BODY_SIG, REGISTER_SOURCE_REFUSE, "} else { source = &found->second; } }"))


def test_mutant_missing_source_looked_up_before_the_locks_fails():
    """The source looked up before the locks are taken: another thread can erase it between the lookup and the
    insert, and the copy inherits from an entry that is gone."""
    unlocked = _once_in(CACHE, REGISTER_BODY_SIG, " std::lock(direct_lock, cache_lock);", "")
    assert not claim_a_missing_source_refuses_before_the_insert(
        _once_in(unlocked, REGISTER_BODY_SIG, REGISTER_SOURCE_REFUSE,
                 REGISTER_SOURCE_REFUSE + " std::lock(direct_lock, cache_lock);"))



# --- a member's ids come from an index, not a scan of the expert table ----------------------------------------------

# A move resolves every member's canonical ids, and the table those come from has a row per expert per tensor (73728 on
# Qwen), so a scan per member cost as much as the transaction's copies. The index is built from the table wherever the
# table is replaced, in the same allocation epoch and swapped in under the same locks, so the two never disagree, and
# the lookups walk only the rows of one tensor name and expert, in table order.
GET_KEYS_SIG = "static std::vector<ggml_sycl_cache_id> ggml_sycl_get_canonical_moe_expert_keys("
MATERIALIZE_SIG = "static bool ggml_sycl_materialize_planned_expert_layout("
META_SCAN = "for (const auto & m : g_moe_expert_meta)"
KEYS_LOOKUP = ("for (uint32_t r = moe_expert_meta_first_row_locked(src0->name, expert_id); r != UINT32_MAX; "
               "r = g_moe_expert_meta_index.next_row[r]) {")
MATERIALIZE_LOOKUP = "const uint32_t r = moe_expert_meta_first_row_locked(src0->name, expert_id);"
INDEX_BUILT = "new_expert_meta_index = moe_expert_meta_index_build(new_expert_meta);"
PUBLISH_LOCK = "std::scoped_lock lock(g_moe_expert_meta_mutex, g_expert_groups_mutex);"
PUBLISH_TABLE = "g_moe_expert_meta.swap(new_expert_meta);"
PUBLISH_INDEX = "g_moe_expert_meta_index.swap(new_expert_meta_index);"
SEED_ROW = "g_moe_expert_meta.push_back({});"
SEED_INDEX = "g_moe_expert_meta_index = moe_expert_meta_index_build(g_moe_expert_meta);"


def claim_member_ids_come_from_the_meta_index(sycl: str) -> bool:
    s = norm(sycl)
    keys = body(s, GET_KEYS_SIG)
    mat = body(s, MATERIALIZE_SIG)
    return (bool(keys) and bool(mat) and META_SCAN not in keys and META_SCAN not in mat
            and keys.count(KEYS_LOOKUP) == 1 and mat.count(MATERIALIZE_LOOKUP) == 1
            and s.count("moe_expert_meta_index_build(") == 3 and s.count(INDEX_BUILT) == 1
            and s.count(PUBLISH_INDEX) == 1 and ordered(s, INDEX_BUILT, PUBLISH_LOCK, PUBLISH_TABLE, PUBLISH_INDEX)
            and s.count(SEED_INDEX) == 1 and ordered(s, SEED_ROW, SEED_INDEX))


def test_member_ids_come_from_the_meta_index():
    assert claim_member_ids_come_from_the_meta_index(SYCL)


def test_mutant_member_ids_scan_the_table_fails():
    """The scan per member back: every member of a move compares its name against every row of the table."""
    assert not claim_member_ids_come_from_the_meta_index(
        _once_in(SYCL, GET_KEYS_SIG, KEYS_LOOKUP, META_SCAN + " { const uint32_t r = 0; (void) r;"))


def test_mutant_index_left_stale_by_a_new_table_fails():
    """A new table published without its index: the lookups answer with the previous model's rows."""
    assert not claim_member_ids_come_from_the_meta_index(_once(SYCL, PUBLISH_INDEX, ""))


def test_mutant_seeded_row_leaves_the_index_behind_fails():
    """A test row seeded without its index: the lookups cannot see the row the test placed."""
    assert not claim_member_ids_come_from_the_meta_index(_once(SYCL, SEED_INDEX, ""))



# --- an expert move's retires leave the eviction latch alone, while its publication covers them ---------------------

# has_evictions_ turns off graph replay and persistent TG for the whole process until the next model load, so a decode
# run that evicted once ran every later graph by direct dispatch. An expert move's retires do not need it: its
# publication bumps the exec-graph epoch, which drops every exec graph that may bake a moved copy's pointer, and
# persistent TG never takes a graph with a MUL_MAT_ID. So (b) the cache marks the retires the move makes that the latch
# would otherwise count, the loan's device entries (a host copy is host-resident, which the latch never counts, so the
# copy to the device marks nothing), finalize leaves only marked entries out of the latch, and a restore clears the
# mark; any other retire of the same entry still latches. And (c) the exemption holds only once the
# epoch has moved: the transaction arms a guard before each cache move, disarms it again for a move the cache refused
# whole, and dismisses it only after the epoch bump in the won swap; on every other exit (a lost swap, a throw) its
# destructor sets the latch through the cache's one public setter.
CACHE_HPP = read("ggml/src/ggml-sycl/unified-cache.hpp")
FINALIZE_SIG = "size_t unified_cache::finalize_retired_entries_locked() {"
FINALIZE_LATCH = re.compile(r"if \(([^{]*)\) \{ has_evictions_\.store\(true")
MOVE_MARK_FIELD = "!entry.retired_by_expert_move"
LOAN_MARK = "device_entry[i]->retired_by_expert_move = true;"
RESTORE_CLEAR = "it->second.retired_by_expert_move = false;"
HOST_RETIRE_CALL = "(void) retire_expert_entry_at_locked(host_key, &host_nodes[r++]);"
LATCH_SETTER = "void latch_evictions() noexcept { has_evictions_.store(true, std::memory_order_release); }"
GUARD_DTOR = "~ggml_sycl_expert_move_latch_guard() { if (armed && cache) { cache->latch_evictions(); } }"
GUARD_DECL = "ggml_sycl_expert_move_latch_guard latch{ cache };"
GUARD_ARM = "const bool armed_before = latch.armed; latch.armed = true;"
GUARD_REFUSED = "if (!loaned) { latch.armed = armed_before; }"
GUARD_DISMISS = "ggml_sycl_optional_layouts_retired(); latch.armed = false; }"
SWAP_WON = "if (swapped) {"


def claim_a_move_marks_its_retires_and_only_they_skip_the_latch(cache: str, hpp: str) -> bool:
    c = norm(cache)
    latch = FINALIZE_LATCH.search(body(c, FINALIZE_SIG))
    loan = body(c, LOAN_SIG)
    copy = body(c, REFILL_BODY_SIG)
    return (latch is not None and MOVE_MARK_FIELD in latch.group(1) and "!entry.optional_layout" in latch.group(1)
            and c.count("retired_by_expert_move = true;") == 1
            and loan.count(LOAN_MARK) == 1 and ordered(loan, LOAN_MARK, DEVICE_RETIRE)
            and "retired_by_expert_move" not in copy
            and body(c, RESTORE_SIG).count(RESTORE_CLEAR) == 1
            and "bool retired_by_expert_move = false;" in norm(hpp))


def test_a_move_marks_its_retires_and_only_they_skip_the_latch():
    assert claim_a_move_marks_its_retires_and_only_they_skip_the_latch(CACHE, CACHE_HPP)


def test_mutant_finalize_latches_a_moves_retire_fails():
    """The exemption gone: a decode run's first evict turns off graph replay until the next load."""
    n = norm(CACHE)
    assert not claim_a_move_marks_its_retires_and_only_they_skip_the_latch(
        _once_in(n, FINALIZE_SIG, " && " + MOVE_MARK_FIELD, ""), CACHE_HPP)


def test_mutant_loan_retire_left_unmarked_fails():
    assert not claim_a_move_marks_its_retires_and_only_they_skip_the_latch(_once_in(CACHE, LOAN_SIG, LOAN_MARK, ""),
                                                                           CACHE_HPP)


def test_mutant_admit_marks_its_host_retire_fails():
    """A host copy is host-resident, so the latch never counted it: the mark there is dead code that reads as a rule."""
    assert not claim_a_move_marks_its_retires_and_only_they_skip_the_latch(
        _once_in(CACHE, REFILL_BODY_SIG, HOST_RETIRE_CALL,
                 "host->second.retired_by_expert_move = true; " + HOST_RETIRE_CALL),
        CACHE_HPP)


def test_mutant_restore_keeps_the_mark_fails():
    """A restored entry kept the mark: a later pressure eviction or drop of it would skip the latch."""
    assert not claim_a_move_marks_its_retires_and_only_they_skip_the_latch(
        _once_in(CACHE, RESTORE_SIG, RESTORE_CLEAR, ""), CACHE_HPP)


def claim_an_unpublished_move_sets_the_latch(sycl: str, hpp: str) -> bool:
    s = norm(sycl)
    txn = body(s, TXN_SIG)
    return (LATCH_SETTER in norm(hpp) and s.count(GUARD_DTOR) == 1
            and txn.count(GUARD_DECL) == 1 and txn.count(GUARD_ARM) == 1 and txn.count(GUARD_REFUSED) == 1
            and txn.count("latch.armed = false;") == 1 and txn.count(GUARD_DISMISS) == 1
            and ordered(txn, GUARD_DECL, GUARD_ARM, MOVE_CALL, GUARD_REFUSED, SWAP_WON, PUBLISH, GUARD_DISMISS))


def test_an_unpublished_move_sets_the_latch():
    assert claim_an_unpublished_move_sets_the_latch(SYCL, CACHE_HPP)


def test_mutant_guard_dismissed_before_the_swap_fails():
    """Dismissed as the cache moves: a lost swap leaves retires no epoch covers out of the latch."""
    n = norm(SYCL)
    moved = _once_in(n, TXN_SIG, GUARD_DISMISS, "ggml_sycl_optional_layouts_retired(); }")
    assert not claim_an_unpublished_move_sets_the_latch(
        _once_in(moved, TXN_SIG, GUARD_REFUSED, GUARD_REFUSED + " latch.armed = false;"), CACHE_HPP)


def test_mutant_guard_sets_nothing_fails():
    assert not claim_an_unpublished_move_sets_the_latch(
        _once(SYCL, GUARD_DTOR, "~ggml_sycl_expert_move_latch_guard() {}"), CACHE_HPP)


def test_mutant_guard_never_armed_fails():
    assert not claim_an_unpublished_move_sets_the_latch(_once_in(SYCL, TXN_SIG, GUARD_ARM, ""), CACHE_HPP)


# (d) The retires stay unreapable until the latch question is settled. A retired device entry is reaped by the
# deferred-free pass once its last lease goes, and the mark leaves that reap out of the latch, so a lease must hold
# each entry until the publication has bumped the epoch, or until the guard has latched on an exit that did not
# publish. The loan hands the leases that held its retires in place to the transaction (expert_move_record::
# retired_leases) instead of dropping them; the transaction moves them into one vector, reserved before the first move
# so taking them cannot throw, and declared before the guard, so the guard's destructor runs first on every exit. The
# vector is never released by hand: it goes as the call returns, after the bump or after the latch.
RECORD_LEASES = "std::shared_ptr<mem_handle> retired_leases[3];"
LEASES_DECL = "std::vector<std::shared_ptr<ggml_sycl::mem_handle>> move_leases;"
LEASES_RESERVE = "move_leases.reserve(3 * prepared.size());"
LEASES_TAKEN = ("for (std::shared_ptr<ggml_sycl::mem_handle> & lease : record.retired_leases) { if (lease) { "
                "move_leases.push_back(std::move(lease)); } }")


def claim_a_moves_retires_stay_leased_until_published(cache: str, hpp: str, sycl: str) -> bool:
    loan = body(norm(cache), LOAN_SIG)
    txn = body(norm(sycl), TXN_SIG)
    return (RECORD_LEASES in norm(hpp) and loan.count(HAND_HELD) == 1 and loan.count(DROP_HELD) == 1
            and ordered(loan, "if (failed) {", DROP_HELD, "return false; }", HAND_HELD, "out.accepted = true;")
            and txn.count("move_leases") == 3 and txn.count(LEASES_TAKEN) == 1
            and ordered(txn, LEASES_DECL, LEASES_RESERVE, GUARD_DECL, MOVE_CALL, LEASES_TAKEN, GUARD_REFUSED))


def test_a_moves_retires_stay_leased_until_published():
    assert claim_a_moves_retires_stay_leased_until_published(CACHE, CACHE_HPP, SYCL)


def test_mutant_loan_drops_its_retires_leases_fails():
    """The loan dropped them as before: the deferred-free pass can reap a marked entry before any epoch moves."""
    assert not claim_a_moves_retires_stay_leased_until_published(_once_in(CACHE, LOAN_SIG, HAND_HELD, DROP_HELD),
                                                                 CACHE_HPP, SYCL)


def test_mutant_txn_leaves_the_leases_on_the_record_fails():
    """Left on the per-triplet record, each lease goes at the end of its loop iteration, before the publication."""
    assert not claim_a_moves_retires_stay_leased_until_published(CACHE, CACHE_HPP,
                                                                 _once_in(SYCL, TXN_SIG, LEASES_TAKEN, ""))


def test_mutant_leases_outlived_by_the_guard_fails():
    """Declared after the guard, the leases go before its destructor latches: a reap in between skips the latch."""
    moved = _once_in(SYCL, TXN_SIG, LEASES_DECL + " " + LEASES_RESERVE, "")
    assert not claim_a_moves_retires_stay_leased_until_published(
        CACHE, CACHE_HPP, _once_in(moved, TXN_SIG, GUARD_DECL, GUARD_DECL + " " + LEASES_DECL + " " + LEASES_RESERVE))


def test_mutant_leases_released_before_the_bump_fails():
    assert not claim_a_moves_retires_stay_leased_until_published(
        CACHE, CACHE_HPP, _once_in(SYCL, TXN_SIG, PUBLISH, "move_leases.clear(); " + PUBLISH))


# The hooks the move cases read the latch and the id map through exist only in the testing build: a production caller
# of test_reset_evictions_latch could clear the latch outside a reclaim and turn graph replay back on over a retire no
# epoch covers.
TEST_ONLY_HOOKS = ("test_id_remap_walks()", "test_id_mapping(ggml_sycl_cache_id id", "test_reset_evictions_latch()")
PRIVATE_TESTING_IF = "#if defined(GGML_SYCL_PRIVATE_TESTING)"


def _in_private_testing(text: str, needle: str) -> bool:
    """Whether every occurrence of `needle` sits in the true branch of a GGML_SYCL_PRIVATE_TESTING conditional."""
    lines = strip_comments(text).split("\n")
    stack: list[bool] = []
    seen = False
    for line in lines:
        directive = line.strip()
        if directive.startswith("#if"):
            stack.append(directive == PRIVATE_TESTING_IF)
        elif directive.startswith("#el"):
            stack[-1] = False
        elif directive.startswith("#endif"):
            stack.pop()
        elif needle in line:
            seen = True
            if not any(stack):
                return False
    return seen


def _raw_once(text: str, old: str, new: str) -> str:
    """`text` with the one occurrence of `old` replaced on the raw text, for a mutant whose anchor is a line break."""
    assert text.count(old) == 1, f"mutant anchor must match exactly once: {old!r} x{text.count(old)}"
    return text.replace(old, new, 1)


def claim_the_move_test_hooks_are_test_only(hpp: str, cache: str) -> bool:
    return all(_in_private_testing(hpp, hook) and _in_private_testing(cache, "unified_cache::" + hook)
               for hook in TEST_ONLY_HOOKS)


def test_the_move_test_hooks_are_test_only():
    assert claim_the_move_test_hooks_are_test_only(CACHE_HPP, CACHE)


def test_mutant_latch_reset_in_production_fails():
    """The declaration moved out of the guard: production code could clear the latch."""
    hpp = _raw_once(CACHE_HPP, "    void     test_reset_evictions_latch() noexcept;\n", "")
    hpp = _raw_once(hpp, PRIVATE_TESTING_IF + "\n    // Test-only",
                "    void     test_reset_evictions_latch() noexcept;\n" + PRIVATE_TESTING_IF + "\n    // Test-only")
    assert not claim_the_move_test_hooks_are_test_only(hpp, CACHE)


def test_mutant_definitions_in_production_fail():
    cache = _raw_once(CACHE, PRIVATE_TESTING_IF + "\nvoid unified_cache::test_reset_evictions_latch()",
                  "void unified_cache::test_reset_evictions_latch()")
    cache = _raw_once(cache, "    return true;\n}\n#endif\n\nuint64_t unified_cache::test_entry_pending_load_txn(",
                  "    return true;\n}\n\nuint64_t unified_cache::test_entry_pending_load_txn(")
    assert not claim_the_move_test_hooks_are_test_only(CACHE_HPP, cache)

# --- one WEIGHT-zone slot per triplet: every member mint, load and swap alike, takes it -----------------------------

# A triplet's members need not stage alike (an AOS down projection whose row width is not a multiple of
# MATRIX_ROW_PADDING carries a tail its gate and up do not). Minted each at its own size, a swap's admitted member
# splits a block another member size gave back and leaves a hole no member fits. So one function gives a triplet's
# slot (expert_triplet_slot_bytes, the largest of the members' staged sizes rounded to the zone's grain, and 0 when that
# would pad the triplet by more than 1/128 of its members' own blocks), one gives each member's size from it
# (expert_triplet_member_bytes, the slot or else the member's own staged size), and every mint of a member asks them:
# the load's per-expert staging through ggml_sycl_expert_stage_bytes, the swap's and the refill's preparation, and the
# cache's check of the storage it is handed; the planner charges each member of a slotted triplet the slot, and each
# AOS member of a complete triplet with none its own staged size, at each plan build and again after each pass that
# re-lays out experts (the single-device PP gate/up preparation, the pack, the I8 upgrades and the multi-device layout
# passes), moving the totals of the device each changed entry is on, found by its target device, by the change, and
# warning when that is past the device's room. The load's bulk staging, one stride for every expert, declines a tensor
# whose triplets' slots exceed that stride, and the opt-in DPAS rollback restores an AOS base at its triplet's slot by
# the load's own plain copy. The logical bytes stay the member's own: every copy moves them and no more.
SLOT_HPP = read("ggml/src/ggml-sycl/expert-slot.hpp")
SLOT_SIG = "inline size_t expert_triplet_slot_bytes(const size_t (&staged)[3])"
SLOT_ROUND = "const size_t rounded = tlsf_allocator::round_request(bytes, tlsf_allocator::block_grain);"
SLOT_MAX = "slot = rounded > slot ? rounded : slot;"
SLOT_DIVISOR = "inline constexpr size_t expert_triplet_slot_padding_divisor = 128;"
SLOT_PAD = ("const size_t padding = 3 * slot - own; "
            "return padding <= own / expert_triplet_slot_padding_divisor ? slot : 0;")
MEMBER_SIG = "inline size_t expert_triplet_member_bytes(const size_t (&staged)[3], int role)"
MEMBER_BODY = "const size_t slot = expert_triplet_slot_bytes(staged); return slot != 0 ? slot : staged[role];"
RECHARGE_SIG = ("static inline std::vector<placement_charge_change> placement_entries_recharge_expert_slots("
                "placement_entry_store & entries)")
RECHARGE_CHARGES = "placement_entries_charge_expert_slots(entries);"
RECHARGE_ON_SIG = ("static inline size_t placement_entries_recharge_expert_slots_on("
                   "placement_entry_store & entries,")
RECHARGE_ON_READ = "placement_entries_recharge_expert_slots(entries)"
RECHARGE_ON_HOST = "if (!e.on_device) { continue; }"
RECHARGE_ON_KEY = "if (d.device_id != e.target_device || !d.remaining) { continue; }"
RECHARGE_ON_MOVE = ("over += placement_charge_move(c.old_charge, c.new_charge, *d.remaining, "
                    "d.device_vram ? *d.device_vram : untracked, weight_vram, vram);")
PLANNER_RECHARGE_SIG = "static size_t planner_recharge_expert_slots(placement_plan & plan,"
PLANNER_RECHARGE_CALL = ("const size_t over = placement_entries_recharge_expert_slots_on(plan.entries, devices, "
                         "plan.weight_vram_bytes, plan.vram_bytes);")
PLANNER_RECHARGE_WARN = 'if (over > 0) { GGML_LOG_WARN("[PLACEMENT-MOE] %s: expert slots re-charged %.1f MB past the '
ROOM_ONE_SIG = "static std::vector<placement_recharge_device> planner_recharge_device(int device_id,"
ROOM_ONE_BODY = ("std::vector<placement_recharge_device> devices; "
                 "devices.push_back({ device_id, &remaining, device_vram }); return devices;")
ROOM_ALL_SIG = "static std::vector<placement_recharge_device> planner_recharge_devices_of(placement_plan & plan,"
ROOM_ALL_BODY = ("std::vector<placement_recharge_device> devices; "
                 "for (size_t d = 0; d < budgets.size() && d < remaining.size(); ++d) { "
                 "size_t * device_vram = d < plan.per_device_vram.size() ? &plan.per_device_vram[d] : nullptr; "
                 "devices.push_back({ budgets[d].device_id, &remaining[d], device_vram }); } return devices;")
MULTI_LAYOUTS_SIG = "static void apply_multi_moe_per_target_layouts(placement_plan & plan,"
MULTI_UPGRADE_SIG = "static void maybe_upgrade_multi_moe_primary_layouts(placement_plan & plan,"
MULTI_RECHARGE = "planner_recharge_expert_slots(plan, planner_recharge_devices_of(plan, device_budgets, remaining),"
MULTI_LAYOUT_SET = "entry.dst_size = new_size;"
UPGRADE_RECHARGE = ("planner_recharge_expert_slots(plan, planner_recharge_device(device_id, remaining, "
                    "&plan.per_device_vram[device_index]),")
SINGLE_RECHARGE = "planner_recharge_expert_slots(plan, planner_recharge_device(device_id, remaining),"
PACK_PREPARE = ("prepare_single_moe_pp_gate_up_primary_layouts_for_pack(plan, device_id, kv_info, envelope); } "
                "placement_entries_charge_expert_slots(plan.entries); auto moe_groups = build_moe_triplet_groups(")
PACK_RECHARGE = SINGLE_RECHARGE + ' "triplet pack");'
PACK_HOST_SET = "entry.dst_size = entry.src_size;"
DOWN_I8_RECHARGE = SINGLE_RECHARGE + ' "I8 down layout upgrades");'
GATE_UP_I8_CALL = "maybe_upgrade_moe_gate_up_layouts_to_i8(plan, remaining, device_id, n_experts);"
GATE_UP_I8_RECHARGE = SINGLE_RECHARGE + ' "I8 gate/up layout upgrades");'
MULTI_DOWN_I8_RECHARGE = MULTI_RECHARGE + ' "I8 down layout upgrades");'
# Each I8 down re-charge comes right after the upgrades it follows: the single-device one after both arms of the
# down upgrade and the PP alternates, each multi-device one right after its primary's down upgrade is charged.
SINGLE_PP_ALTERNATES = "add_single_moe_pp_executable_alternates(plan, remaining, device_id, kv_info, envelope, nullptr"
SINGLE_DOWN_I8_CALL = "maybe_upgrade_moe_down_layouts_to_i8(plan, remaining, device_id, n_experts);"
SINGLE_DOWN_I8_ARMS = ("if (!planner_moe_prompt_down_specialized_layouts_enabled()) { " + SINGLE_PP_ALTERNATES
                       + ", false); " + SINGLE_DOWN_I8_CALL + " } else { " + SINGLE_DOWN_I8_CALL + " "
                       + SINGLE_PP_ALTERNATES + "); }")
SINGLE_DOWN_I8_SITE = SINGLE_DOWN_I8_ARMS + " " + DOWN_I8_RECHARGE
PRIMARY_DOWN_I8_UPGRADE = ("maybe_upgrade_moe_down_layouts_to_i8(plan, remaining[0], device_budgets[0].device_id, "
                           "n_experts, false); if (!plan.per_device_vram.empty()) { "
                           "plan.per_device_vram[0] += primary_down_charged; }")
PRIMARY_DOWN_I8_SITE = PRIMARY_DOWN_I8_UPGRADE + " " + MULTI_DOWN_I8_RECHARGE
# The second multi-device site is anchored on the end of both arms of its down upgrade, as the single-device one is,
# so neither the re-charge nor the re-charge with the charge bookkeeping can move ahead of the arms.
BRANCH_PP_ALTERNATES = "add_multi_moe_pp_executable_alternates(plan, remaining, device_budgets, kv_info, envelope);"
BRANCH_DOWN_I8_CALL = ("primary_down_charged = maybe_upgrade_moe_down_layouts_to_i8(plan, remaining[0], "
                       "device_budgets[0].device_id, n_experts, multi_static_i8_executor_supported);")
BRANCH_DOWN_I8_ARMS = ("if (!planner_moe_prompt_down_specialized_layouts_enabled()) { " + BRANCH_PP_ALTERNATES + " "
                       + BRANCH_DOWN_I8_CALL + " } else { " + BRANCH_DOWN_I8_CALL + " " + BRANCH_PP_ALTERNATES + " }")
BRANCH_DOWN_I8_TOTAL = "plan.per_device_vram[0] += primary_down_charged;"
BRANCH_DOWN_I8_SITE = BRANCH_DOWN_I8_ARMS + " " + BRANCH_DOWN_I8_TOTAL + " " + MULTI_DOWN_I8_RECHARGE
# A repeat of a triplet role in another layout or on another device is charged on its own while the load stages it as
# its member: the planner says so at WARN, naming the repeat, and a duplicated role raises the summary line to WARN.
DIAG_SIG = "static void log_moe_placement_diagnostics(const char * tag, const placement_plan & plan, int n_experts)"
DIAG_SUMMARY = ('ggml_log_internal(summary.duplicates != 0 ? GGML_LOG_LEVEL_WARN : GGML_LOG_LEVEL_INFO, '
                '"[%s] expected=%zu planned=%zu unclassified=%zu duplicates=%zu missing=%zu\\n",')
DIAG_OFF_LOOP = ("for (const placement_repeat_off_member & off : "
                 "placement_entries_repeats_off_their_member(plan.entries)) {")
DIAG_OFF_WARN = ('GGML_LOG_WARN("[%s] layer %d expert %d %s: a repeated entry (%s on %s) differs from its member '
                 '(%s on %s); the load stages it as its member, so the repeat\'s charge does not follow the slot the '
                 'load stages it at\\n", tag, off.repeat->layer_id, off.repeat->expert_id, '
                 'expert_tensor_role_name(off.repeat->expert_role),')
OFF_SIG = "static inline std::vector<placement_repeat_off_member> placement_entries_repeats_off_their_member("
OFF_TEST = "} else if (e.layout != member->layout || e.target_device != member->target_device) {"
PLAN_SLOT_SIG = "static inline size_t moe_expert_triplet_slot_bytes(const placement_entry * const (&t)[3])"
PLAN_SLOT_CALL = "return expert_triplet_slot_bytes(staged);"
PLAN_ENTRY_STAGED = "return staged_expert_bytes(e.layout, e.type, e.ne[0], e.src_size);"
CHARGE_SIG = "static inline size_t placement_entries_charge_expert_slots(placement_entry_store & entries)"
# The charge resolves a triplet as the plan's index does (first entry of each role, no entry with no layer), so the
# load's lookup of the slot and the charge agree, and charges every entry that stages as AOS what the load stages it at.
CHARGE_INDEXED = ("if (e.expert_id < 0 || e.layer_id < 0 || !detail::moe_expert_role_is_triplet(e.expert_role)) { "
                  "continue; }")
CHARGE_FIRST = "if (member) { t.repeats.push_back(&e); } else { member = &e; }"
CHARGE_COMPLETE = "if (!members[0] || !members[1] || !members[2]) { continue; }"
CHARGE_SET = "e->vram_charge_size = placement_vram_charge_bytes(std::max(slot_bytes, staged));"
CHARGE_STAGED = ("const size_t staged = moe_expert_entry_staged_bytes(*e); if (staged != 0) { " + CHARGE_SET + " }")
CHARGE_EACH = ("for (placement_entry * e : t.members) { charge(e); } "
               "for (placement_entry * e : t.repeats) { charge(e); }")
# The loan's bulk answer mirrors the load's decline of an AOS tensor whose triplets' slot passes its one stride, in the
# layout the load stages: the primary's, or the alternate's for a tensor staged through its alternates.
BULK_MIRROR_SIG = ("static inline bool moe_loan_tensor_bulk_staged(const placement_plan & plan, "
                   "const std::string & name, int device)")
BULK_MIRROR_ALT = "++n_alt; staged_layout = alt.layout;"
BULK_MIRROR = ("if (staged_layout == GGML_LAYOUT_AOS) { const size_t stride = staged_expert_bytes(GGML_LAYOUT_AOS, "
               "e.type, e.ne[0], e.src_size); if (moe_expert_triplet_slot_bytes(plan, e.layer_id, e.expert_id) > "
               "stride) { return false; } }")
# The materialization reorder runs after the last charge and is the plan order the index reads: it keeps entries that
# share a role in the order the charge saw, so the index's first entry of a role is the one charged as the member.
REORDER_SIG = "static void reorder_plan_entries_for_moe_materialization(placement_plan & plan,"
REORDER_SORT = ("std::stable_sort(indices.begin(), indices.end(), [&](size_t a, size_t b) { return "
                "moe_materialization_role_order(plan.entries[a].expert_role) < "
                "moe_materialization_role_order(plan.entries[b].expert_role); });")
STAGE_BYTES_SIG = "static size_t ggml_sycl_expert_stage_bytes(const ggml_sycl::placement_plan & plan,"
STAGE_BYTES_BODY = ("if (layout != GGML_LAYOUT_AOS) { return staged; } const size_t slot = "
                    "ggml_sycl::moe_expert_triplet_slot_bytes(plan, layer, expert); return slot >= staged ? slot : "
                    "staged; }")
# The preload reads the cache's plan once per tensor, for the bulk decline and each expert's slot sizing, and a DPAS
# rollback once for its restore: an owner fetch costs an atomic load and a shared_ptr copy.
PRELOAD_OWNER = ("const auto plan_owner = ggml_sycl_cache_plan_owner(cache); "
                 "auto try_bulk_stage_moe_tensor = [&]() -> bool {")
DPAS_OWNER = ("restore_keys.reserve(local_experts.size()); const auto plan_owner = ggml_sycl_cache_plan_owner(cache); "
              "for (int e : local_experts) {")
PRELOAD_SLOT = ("expert_preload_bytes = ggml_sycl_expert_stage_bytes(*plan_owner, expert_slot_layer, "
                "static_cast<int>(e), expert_preload_layout, expert_preload_bytes);")
PRELOAD_STAGE = ("ggml_sycl::unified_cache_direct_stage_expert(device, key, expert_aos, expert_size, "
                 "expert_preload_bytes, expert_preload_layout,")
PRELOAD_STAGE_END = PRELOAD_STAGE + " expert_fill_fn, expert_fill_ctx, moe_preload_q, &handle);"
# Layer 0 is a layer: the sizing runs for every planned layer, and only an expert the plan does not place on this
# device (layer -1) stages at its own size.
PRELOAD_GUARD = "if (expert_slot_layer >= 0) { " + PRELOAD_SLOT + " }"
PRELOAD_LAYER = "expert_slot_layer = placement.layer_id;"
BULK_LAYER = "bulk_layers.push_back(placement.layer_id);"
BULK_SLOT = ("if (bulk_layout == GGML_LAYOUT_AOS) { for (size_t e = 0; e < bulk_layers.size(); ++e) { "
             "if (ggml_sycl_expert_stage_bytes(*plan_owner, bulk_layers[e], static_cast<int>(e), bulk_layout, "
             "bulk_expert_bytes) != bulk_expert_bytes) { return false; } } }")
BULK_STAGE = "auto bulk_result = ggml_sycl::unified_cache_direct_stage_expert_tensor("
DPAS_AOS_FILL = ("if (base_layout == GGML_LAYOUT_AOS) { restore_fill_fn = nullptr; restore_fill = nullptr; "
                 "if (have_plan) {")
DPAS_SLOT = ("restore_bytes = ggml_sycl_expert_stage_bytes(*plan_owner, placement.layer_id, e, base_layout, "
             "expert_base_bytes);")
DPAS_STAGE = ("auto rollback = ggml_sycl::unified_cache_direct_stage_expert(device, key, expert_aos, tensor->nb[2], "
              "restore_bytes, base_layout, restore_fill_fn, restore_fill, s1_preload_q, &rollback_handle);")
PLANNER_CHARGES = "placement_entries_charge_expert_slots(plan.entries);"
TESTING_IF_RE = re.compile(r"#\s*if\s+defined\s*\(\s*GGML_SYCL_PRIVATE_TESTING\s*\)")
CONDITIONAL_RE = re.compile(r"#\s*(if|ifdef|ifndef|endif)\b")
FIXTURE_SLOT_RE = re.compile(r"\bexpert_triplet_slot_bytes\(")


def conditional_end(text: str, after: int) -> int:
    """Where the conditional opened just before `after` closes: the end of its matching #endif, found by counting
    the conditionals nested in it. -1 when it never closes."""
    depth = 1
    for m in CONDITIONAL_RE.finditer(text, after):
        depth += -1 if m.group(1) == "endif" else 1
        if depth == 0:
            return m.end()
    return -1


def production(text: str) -> str:
    """`text` without each GGML_SYCL_PRIVATE_TESTING block that computes a slot of its own (the device fixtures,
    which size their own entries). The block is found by balancing its conditionals, not by the comment on its
    #endif, so raw and normalized text (which has no comments) lose the same block. Unchanged when there is none."""
    out = []
    pos = 0
    for m in TESTING_IF_RE.finditer(text):
        if m.start() < pos:
            continue
        end = conditional_end(text, m.end())
        if end >= 0 and FIXTURE_SLOT_RE.search(text, m.end(), end):
            out.append(text[pos:m.start()])
            pos = end
    out.append(text[pos:])
    return "".join(out)


def claim_every_member_mint_takes_the_triplet_slot(sycl: str, cache: str, plan_hpp: str, slot_hpp: str) -> bool:
    """One function gives the slot, from the largest of the members' staged sizes rounded to the zone's grain, and
    none past 1/128 of padding; one gives each member's size from it, and the plan header asks it in one place only.
    The plan asks the slot of a triplet's three entries, the preparation and the cache ask each member's size of the
    three members they are handed; the load's per-expert staging mints at the plan's slot, through one helper, for
    every planned layer from 0, its bulk staging declines a tensor a slot would not fit (and the loan's bulk answer
    mirrors that decline, in the layout the load stages), and the DPAS rollback restores an AOS base at the slot by
    the plain copy. The planner resolves a triplet as the plan's index does (each role's first entry, no entry with no
    layer) and charges each of its entries that stages as AOS what the load stages it at: the slot when the triplet
    has one, its own staged size when it has none. It charges at each of its two plan builds (single-device and
    multi-device) and after the single-device PP preparation, and re-charges with accounting after the pack, right
    after each I8 upgrade and after each multi-device layout pass: the change of each entry on a device moves the
    totals of the device in the pass's list whose id is its target device (each list's device and room taken from the
    pass's own, index by index), and a charge past the room warns. Nothing else computes a slot."""
    s = norm(production(sycl))
    c = norm(production(cache))
    p = norm(plan_hpp)
    h = norm(slot_hpp)
    slot_body = body(h, SLOT_SIG)
    charge = body(p, CHARGE_SIG)
    recharge_on = body(p, RECHARGE_ON_SIG)
    planner_recharge = body(c, PLANNER_RECHARGE_SIG)
    multi_layouts = body(c, MULTI_LAYOUTS_SIG)
    multi_upgrade = body(c, MULTI_UPGRADE_SIG)
    return (slot_body.count(SLOT_ROUND) == 1 and slot_body.count(SLOT_MAX) == 1 and h.count(SLOT_DIVISOR) == 1
            and slot_body.rstrip().endswith(SLOT_PAD + " }")
            and body(h, MEMBER_SIG).strip() == "{ " + MEMBER_BODY + " }"
            and body(p, RECHARGE_SIG).count(RECHARGE_CHARGES) == 1
            and recharge_on.count(RECHARGE_ON_READ) == 1 and recharge_on.count(RECHARGE_ON_MOVE) == 1
            and ordered(recharge_on, RECHARGE_ON_READ, RECHARGE_ON_HOST, RECHARGE_ON_KEY, RECHARGE_ON_MOVE)
            and planner_recharge.count(PLANNER_RECHARGE_CALL) == 1
            and ordered(planner_recharge, PLANNER_RECHARGE_CALL, PLANNER_RECHARGE_WARN)
            and planner_recharge.count("GGML_LOG") == 1
            and body(c, ROOM_ONE_SIG).strip() == "{ " + ROOM_ONE_BODY + " }"
            and body(c, ROOM_ALL_SIG).strip() == "{ " + ROOM_ALL_BODY + " }"
            and multi_layouts.count(MULTI_RECHARGE) == 1 and ordered(multi_layouts, MULTI_LAYOUT_SET, MULTI_RECHARGE)
            and multi_upgrade.count(UPGRADE_RECHARGE) == 1
            and ordered(multi_upgrade, MULTI_LAYOUT_SET, UPGRADE_RECHARGE)
            and c.count(PACK_PREPARE) == 1 and ordered(c, PACK_PREPARE, PACK_HOST_SET, PACK_RECHARGE)
            and ordered(c, PACK_RECHARGE, DOWN_I8_RECHARGE, GATE_UP_I8_CALL, GATE_UP_I8_RECHARGE)
            and c.count(SINGLE_DOWN_I8_SITE) == 1
            and c.count(SINGLE_RECHARGE) == 3 and c.count(MULTI_DOWN_I8_RECHARGE) == 2
            and c.count(PRIMARY_DOWN_I8_SITE) == 1 and c.count(BRANCH_DOWN_I8_SITE) == 1
            and len(re.findall(r"\bplanner_recharge_expert_slots\(", c)) == 8
            and body(p, PLAN_SLOT_SIG).count(PLAN_SLOT_CALL) == 1
            and len(re.findall(r"\bexpert_triplet_slot_bytes\(", p)) == 1
            and PLAN_ENTRY_STAGED in p and p.count(CHARGE_SET) == 1 and ordered(p, CHARGE_SIG, CHARGE_SET)
            and ordered(charge, CHARGE_INDEXED, CHARGE_FIRST, CHARGE_COMPLETE, CHARGE_STAGED, CHARGE_EACH)
            and ordered(body(p, BULK_MIRROR_SIG), BULK_MIRROR_ALT, BULK_MIRROR)
            and body(c, REORDER_SIG).count(REORDER_SORT) == 1 and "std::sort(" not in body(c, REORDER_SIG)
            and body(s, STAGE_BYTES_SIG).rstrip().endswith(STAGE_BYTES_BODY)
            and s.count(PRELOAD_LAYER) == 1 and s.count(PRELOAD_SLOT) == 1 and s.count(PRELOAD_GUARD) == 1
            and s.count(PRELOAD_STAGE_END) == 1 and ordered(s, PRELOAD_LAYER, PRELOAD_GUARD, PRELOAD_STAGE_END)
            and s.count(BULK_SLOT) == 1 and ordered(s, BULK_LAYER, BULK_SLOT, BULK_STAGE)
            and s.count(DPAS_SLOT) == 1 and ordered(s, DPAS_AOS_FILL, DPAS_SLOT, DPAS_STAGE)
            and s.count(PRELOAD_OWNER) == 1 and ordered(s, PRELOAD_OWNER, BULK_SLOT, PRELOAD_GUARD)
            and s.count(DPAS_OWNER) == 1 and ordered(s, DPAS_OWNER, DPAS_SLOT)
            and len(re.findall(r"\bggml_sycl_expert_stage_bytes\(", s)) == 4
            and len(re.findall(r"\bexpert_triplet_slot_bytes\(", s)) == 0
            and len(re.findall(r"\bexpert_triplet_slot_bytes\(", c)) == 0
            and len(re.findall(r"\bexpert_triplet_member_bytes\(", s)) == 1
            and len(re.findall(r"\bexpert_triplet_member_bytes\(", c)) == 1
            and c.count(PLANNER_CHARGES) == 3)


def test_every_member_mint_takes_the_triplet_slot():
    assert claim_every_member_mint_takes_the_triplet_slot(SYCL, CACHE, PLAN_HPP, SLOT_HPP)


def test_every_member_mint_takes_the_triplet_slot_on_normalized_sources():
    """The mutants hand the claim normalized text, which has no comments: unchanged, it must pass as the raw text
    does, or every mutant is caught whatever it changes."""
    assert claim_every_member_mint_takes_the_triplet_slot(norm(SYCL), norm(CACHE), norm(PLAN_HPP), norm(SLOT_HPP))


def test_production_drops_the_fixtures_from_raw_and_normalized_text_alike():
    """Both forms lose every fixture's own slot and keep the load's sizing."""
    for text in (SYCL, norm(SYCL)):
        kept = production(text)
        assert len(FIXTURE_SLOT_RE.findall(text)) > 0 and len(FIXTURE_SLOT_RE.findall(kept)) == 0
        assert norm(kept).count(PRELOAD_SLOT) == 1


def test_mutant_slot_from_the_first_member_fails():
    mutant = _once_in(SLOT_HPP, SLOT_SIG, SLOT_MAX, "slot = slot == 0 ? rounded : slot;")
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, CACHE, PLAN_HPP, mutant)


def test_mutant_slot_unrounded_fails():
    mutant = _once_in(SLOT_HPP, SLOT_SIG, SLOT_ROUND, "const size_t rounded = bytes;")
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, CACHE, PLAN_HPP, mutant)


def test_mutant_load_stages_a_member_at_its_own_size_fails():
    mutant = _once(SYCL, PRELOAD_SLOT, "")
    assert not claim_every_member_mint_takes_the_triplet_slot(mutant, CACHE, PLAN_HPP, SLOT_HPP)


def test_mutant_load_slot_after_the_stage_fails():
    """The sizing runs once the stage has minted the member at its own size: too late to give it the slot."""
    mutant = _once(_once(SYCL, PRELOAD_GUARD, ""), PRELOAD_STAGE_END, PRELOAD_STAGE_END + " " + PRELOAD_GUARD)
    assert not claim_every_member_mint_takes_the_triplet_slot(mutant, CACHE, PLAN_HPP, SLOT_HPP)


def test_mutant_load_slot_skips_layer_zero_fails():
    """Layer 0's experts stage at their own sizes while the plan charges them the slot."""
    mutant = _once(SYCL, "if (expert_slot_layer >= 0) {", "if (expert_slot_layer > 0) {")
    assert not claim_every_member_mint_takes_the_triplet_slot(mutant, CACHE, PLAN_HPP, SLOT_HPP)


def test_mutant_load_slot_never_has_a_layer_fails():
    mutant = _once(SYCL, PRELOAD_LAYER, "")
    assert not claim_every_member_mint_takes_the_triplet_slot(mutant, CACHE, PLAN_HPP, SLOT_HPP)


def test_mutant_stage_helper_ignores_the_slot_fails():
    mutant = _once_in(SYCL, STAGE_BYTES_SIG, "return slot >= staged ? slot : staged;", "return staged;")
    assert not claim_every_member_mint_takes_the_triplet_slot(mutant, CACHE, PLAN_HPP, SLOT_HPP)


def test_mutant_a_second_slot_source_fails():
    """A second place computing a slot from the members' staged sizes, which could disagree with the first."""
    mutant = _once(SYCL, PRELOAD_SLOT, PRELOAD_SLOT + " (void) ggml_sycl::expert_triplet_slot_bytes({ 1, 2, 3 });")
    assert not claim_every_member_mint_takes_the_triplet_slot(mutant, CACHE, PLAN_HPP, SLOT_HPP)


def test_mutant_planner_charges_one_member_alone_fails():
    mutant = _once(PLAN_HPP, CHARGE_SET, "(void) e;")
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, CACHE, mutant, SLOT_HPP)


def test_mutant_planner_misses_a_build_fails():
    n = norm(CACHE)
    mutant = n.replace(PLANNER_CHARGES, "", 1)
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, mutant, PLAN_HPP, SLOT_HPP)


def test_mutant_slot_with_no_padding_bound_fails():
    """Every triplet takes the largest member's size: a Q4_K gate and up beside a Q5_K down pad by 14%."""
    mutant = _once_in(SLOT_HPP, SLOT_SIG, SLOT_PAD, "return slot;")
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, CACHE, PLAN_HPP, mutant)


def test_mutant_member_takes_the_slot_even_with_none_fails():
    mutant = _once_in(SLOT_HPP, MEMBER_SIG, "return slot != 0 ? slot : staged[role];", "return slot;")
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, CACHE, PLAN_HPP, mutant)


def test_mutant_multi_layouts_skip_the_recharge_fails():
    """The per-target layout pass charges a member it moved into AOS its own size, and nothing charges the slot back:
    the plan under-charges what the load stages, an OOM at load."""
    mutant = _once_in(CACHE, MULTI_LAYOUTS_SIG, MULTI_RECHARGE, "skip_recharge(plan, planner_recharge_devices_of(plan, "
                      "device_budgets, remaining),")
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, mutant, PLAN_HPP, SLOT_HPP)


def test_mutant_multi_upgrade_skips_the_recharge_fails():
    mutant = _once_in(CACHE, MULTI_UPGRADE_SIG, UPGRADE_RECHARGE, "skip_recharge(plan, planner_recharge_device("
                      "device_id, remaining, &plan.per_device_vram[device_index]),")
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, mutant, PLAN_HPP, SLOT_HPP)


def test_mutant_recharge_moves_no_total_fails():
    """The re-charge sets the slot on the entries and leaves every device total where the pass put it."""
    mutant = _once_in(PLAN_HPP, RECHARGE_ON_SIG, RECHARGE_ON_MOVE, "(void) c;")
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, CACHE, mutant, SLOT_HPP)


def test_mutant_recharge_charges_nothing_fails():
    mutant = _once_in(PLAN_HPP, RECHARGE_SIG, RECHARGE_CHARGES, "")
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, CACHE, mutant, SLOT_HPP)


def test_mutant_a_second_slot_call_in_the_plan_header_fails():
    """A second place in the plan header asking the slot of staged sizes, which could disagree with the plan's own."""
    mutant = _once_in(PLAN_HPP, CHARGE_SIG, CHARGE_SET, CHARGE_SET + " (void) expert_triplet_slot_bytes({ 1, 2, 3 });")
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, CACHE, mutant, SLOT_HPP)


def test_mutant_lost_slot_keeps_its_charge_fails():
    """A triplet that lost its slot keeps charging its AOS members the slot: the plan over-charges forever."""
    mutant = _once_in(PLAN_HPP, CHARGE_SIG, "if (staged != 0) {", "if (staged != 0 && slot_bytes != 0) {")
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, CACHE, mutant, SLOT_HPP)


def test_mutant_charge_takes_a_roles_last_entry_fails():
    """The load's lookup keeps a role's first entry: charged from the last, a duplicated role is charged another
    triplet's slot than the one it stages at."""
    mutant = _once_in(PLAN_HPP, CHARGE_SIG, CHARGE_FIRST, "member = &e;")
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, CACHE, mutant, SLOT_HPP)


def test_mutant_charge_leaves_a_repeated_role_uncharged_fails():
    mutant = _once_in(PLAN_HPP, CHARGE_SIG, "for (placement_entry * e : t.repeats) { charge(e); }", "")
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, CACHE, mutant, SLOT_HPP)


def test_mutant_charge_counts_an_entry_with_no_layer_fails():
    """The index leaves an entry with no layer out of every triplet; the charge would make it a member."""
    mutant = _once_in(PLAN_HPP, CHARGE_SIG, "e.expert_id < 0 || e.layer_id < 0 ||", "e.expert_id < 0 ||")
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, CACHE, mutant, SLOT_HPP)


def test_mutant_loan_reads_a_slotted_aos_tensor_as_bulk_fails():
    """The load stages such a tensor expert by expert; read as bulk, none of its experts is ever lent."""
    mutant = _once_in(PLAN_HPP, BULK_MIRROR_SIG, BULK_MIRROR, "")
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, CACHE, mutant, SLOT_HPP)


def test_mutant_loan_bulk_mirror_reads_the_primarys_layout_for_an_alternate_fails():
    """An alternate-staged tensor stages in the alternate's layout, not its primary's."""
    mutant = _once_in(PLAN_HPP, BULK_MIRROR_SIG, BULK_MIRROR_ALT, "++n_alt;")
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, CACHE, mutant, SLOT_HPP)


def test_mutant_materialization_reorder_sorts_unstably_fails():
    """An unstable sort may swap two entries of one role after the charge: the index then reads as the member an entry
    the charge took as a repeat, and the load stages the triplet at a slot the plan did not charge."""
    mutant = _once_in(CACHE, REORDER_SIG, "std::stable_sort(indices.begin()", "std::sort(indices.begin()")
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, mutant, PLAN_HPP, SLOT_HPP)


def test_mutant_materialization_reorder_breaks_a_role_tie_by_priority_fails():
    """Stable or not, a priority tie-break moves a later entry of a role ahead of the first when its priority is
    lower."""
    mutant = _once_in(CACHE, REORDER_SIG, "moe_materialization_role_order(plan.entries[b].expert_role); });",
                      "moe_materialization_role_order(plan.entries[b].expert_role) || "
                      "static_cast<uint8_t>(plan.entries[a].priority) < "
                      "static_cast<uint8_t>(plan.entries[b].priority); });")
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, mutant, PLAN_HPP, SLOT_HPP)


def test_mutant_recharge_ignores_the_target_device_fails():
    """Every change moves the first device's room, whatever device the entry is on."""
    mutant = _once_in(PLAN_HPP, RECHARGE_ON_SIG, RECHARGE_ON_KEY, "if (!d.remaining) { continue; }")
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, CACHE, mutant, SLOT_HPP)


def test_mutant_multi_room_list_is_empty_fails():
    """The multi-device list names no device, so no change moves any room."""
    mutant = _once_in(CACHE, ROOM_ALL_SIG, "devices.push_back({ budgets[d].device_id, &remaining[d], device_vram });",
                      "(void) device_vram;")
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, mutant, PLAN_HPP, SLOT_HPP)


def test_mutant_multi_room_keys_every_room_to_the_first_device_fails():
    mutant = _once_in(CACHE, ROOM_ALL_SIG, "{ budgets[d].device_id, &remaining[d],", "{ budgets[0].device_id, "
                      "&remaining[d],")
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, mutant, PLAN_HPP, SLOT_HPP)


def test_mutant_multi_room_reads_another_devices_room_fails():
    mutant = _once_in(CACHE, ROOM_ALL_SIG, "&remaining[d], device_vram", "&remaining[0], device_vram")
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, mutant, PLAN_HPP, SLOT_HPP)


def test_mutant_single_room_ignores_its_device_fails():
    mutant = _once_in(CACHE, ROOM_ONE_SIG, "{ device_id, &remaining, device_vram }", "{ -1, &remaining, device_vram }")
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, mutant, PLAN_HPP, SLOT_HPP)


def test_mutant_single_room_reads_no_room_fails():
    mutant = _once_in(CACHE, ROOM_ONE_SIG, "{ device_id, &remaining, device_vram }",
                      "{ device_id, nullptr, device_vram }")
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, mutant, PLAN_HPP, SLOT_HPP)


def test_mutant_recharge_overflow_demoted_to_debug_fails():
    mutant = _once_in(CACHE, PLANNER_RECHARGE_SIG, 'GGML_LOG_WARN("[PLACEMENT-MOE] %s: expert slots re-charged',
                      'GGML_LOG_DEBUG("[PLACEMENT-MOE] %s: expert slots re-charged')
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, mutant, PLAN_HPP, SLOT_HPP)


def test_mutant_recharge_overflow_prefix_changed_fails():
    mutant = _once_in(CACHE, PLANNER_RECHARGE_SIG, '"[PLACEMENT-MOE] %s: expert slots re-charged',
                      '"[PLACEMENT-MOE-MULTI] %s: expert slots re-charged')
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, mutant, PLAN_HPP, SLOT_HPP)


def test_mutant_prepare_leaves_the_lost_slot_charged_fails():
    """The single-device PP preparation moves a gate or up to SOA and the pack reads the stale slot charge."""
    mutant = _once(CACHE, PACK_PREPARE, PACK_PREPARE.replace("placement_entries_charge_expert_slots(plan.entries); ",
                                                             ""))
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, mutant, PLAN_HPP, SLOT_HPP)


def test_mutant_single_i8_down_skips_the_recharge_fails():
    mutant = _once(CACHE, DOWN_I8_RECHARGE, "")
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, mutant, PLAN_HPP, SLOT_HPP)


def test_mutant_single_i8_recharge_before_the_gate_up_upgrade_fails():
    mutant = _once(_once(CACHE, GATE_UP_I8_RECHARGE, ""), GATE_UP_I8_CALL, GATE_UP_I8_RECHARGE + " " + GATE_UP_I8_CALL)
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, mutant, PLAN_HPP, SLOT_HPP)


def test_mutant_multi_i8_down_skips_the_recharge_fails():
    n = norm(CACHE)
    mutant = n.replace(MULTI_DOWN_I8_RECHARGE, "", 1)
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, mutant, PLAN_HPP, SLOT_HPP)


def test_mutant_single_i8_down_recharge_before_the_upgrade_fails():
    """The re-charge runs before the down upgrade moves members out of AOS: the slots they lose stay charged."""
    mutant = _once(CACHE, SINGLE_DOWN_I8_SITE, DOWN_I8_RECHARGE + " " + SINGLE_DOWN_I8_ARMS)
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, mutant, PLAN_HPP, SLOT_HPP)


def test_mutant_multi_primary_i8_down_recharge_before_the_upgrade_fails():
    mutant = _once(CACHE, PRIMARY_DOWN_I8_SITE, MULTI_DOWN_I8_RECHARGE + " " + PRIMARY_DOWN_I8_UPGRADE)
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, mutant, PLAN_HPP, SLOT_HPP)


def test_mutant_multi_branch_i8_down_recharge_before_the_upgrade_fails():
    """The re-charge runs ahead of both arms of the down upgrade: the slots the upgrade takes stay charged."""
    mutant = _once(CACHE, BRANCH_DOWN_I8_SITE,
                   MULTI_DOWN_I8_RECHARGE + " " + BRANCH_DOWN_I8_ARMS + " " + BRANCH_DOWN_I8_TOTAL)
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, mutant, PLAN_HPP, SLOT_HPP)


def test_mutant_multi_branch_i8_down_total_and_recharge_before_the_upgrade_fails():
    """The charge bookkeeping and the re-charge move together ahead of the arms: still before the upgrade."""
    mutant = _once(CACHE, BRANCH_DOWN_I8_SITE,
                   BRANCH_DOWN_I8_TOTAL + " " + MULTI_DOWN_I8_RECHARGE + " " + BRANCH_DOWN_I8_ARMS)
    assert not claim_every_member_mint_takes_the_triplet_slot(SYCL, mutant, PLAN_HPP, SLOT_HPP)


def claim_a_repeat_off_its_member_warns(cache: str, plan_hpp: str) -> bool:
    """The planner names, at WARN, each repeat of a triplet role whose layout or device differs from its member's,
    and raises the placement summary to WARN when a role is duplicated."""
    diag = body(norm(cache), DIAG_SIG)
    off = body(norm(plan_hpp), OFF_SIG)
    return (diag.count(DIAG_SUMMARY) == 1 and diag.count(DIAG_OFF_LOOP) == 1 and diag.count(DIAG_OFF_WARN) == 1
            and ordered(diag, DIAG_SUMMARY, DIAG_OFF_LOOP, DIAG_OFF_WARN) and off.count(OFF_TEST) == 1)


def test_a_repeat_off_its_member_warns():
    assert claim_a_repeat_off_its_member_warns(CACHE, PLAN_HPP)


def test_mutant_duplicate_summary_demoted_to_info_fails():
    mutant = _once_in(CACHE, DIAG_SIG, "summary.duplicates != 0 ? GGML_LOG_LEVEL_WARN : GGML_LOG_LEVEL_INFO",
                      "GGML_LOG_LEVEL_INFO")
    assert not claim_a_repeat_off_its_member_warns(mutant, PLAN_HPP)


def test_mutant_repeat_off_its_member_demoted_to_info_fails():
    mutant = _once_in(CACHE, DIAG_SIG, 'GGML_LOG_WARN("[%s] layer %d expert', 'GGML_LOG_INFO("[%s] layer %d expert')
    assert not claim_a_repeat_off_its_member_warns(mutant, PLAN_HPP)


def test_mutant_repeat_off_its_member_ignores_the_device_fails():
    mutant = _once_in(PLAN_HPP, OFF_SIG, " || e.target_device != member->target_device", "")
    assert not claim_a_repeat_off_its_member_warns(CACHE, mutant)


def test_mutant_preload_sizes_from_the_global_plan_fails():
    """The global snapshot is not the cache's plan: the load would size a slot from another plan than it stages by."""
    mutant = _once(SYCL, PRELOAD_OWNER,
                   PRELOAD_OWNER.replace("ggml_sycl_cache_plan_owner(cache)", "preload_plan_owner"))
    assert not claim_every_member_mint_takes_the_triplet_slot(mutant, CACHE, PLAN_HPP, SLOT_HPP)


def test_mutant_dpas_restore_sizes_from_the_global_plan_fails():
    mutant = _once(SYCL, DPAS_OWNER, DPAS_OWNER.replace("ggml_sycl_cache_plan_owner(cache)", "preload_plan_owner"))
    assert not claim_every_member_mint_takes_the_triplet_slot(mutant, CACHE, PLAN_HPP, SLOT_HPP)


def test_mutant_bulk_stages_past_the_slot_fails():
    """The bulk copy stages a slotted triplet's members at their own stride while the plan charges the slot."""
    mutant = _once(SYCL, BULK_SLOT, "")
    assert not claim_every_member_mint_takes_the_triplet_slot(mutant, CACHE, PLAN_HPP, SLOT_HPP)


def test_mutant_dpas_restores_at_the_base_size_fails():
    mutant = _once(SYCL, DPAS_STAGE, DPAS_STAGE.replace("restore_bytes, base_layout", "expert_base_bytes, base_layout"))
    assert not claim_every_member_mint_takes_the_triplet_slot(mutant, CACHE, PLAN_HPP, SLOT_HPP)


def test_mutant_dpas_restores_aos_by_the_reorder_fill_fails():
    """The reorder fill writes its whole destination, so at the slot it would copy past the expert's bytes."""
    mutant = _once(SYCL, DPAS_AOS_FILL, "if (base_layout == GGML_LAYOUT_AOS) { if (have_plan) {")
    assert not claim_every_member_mint_takes_the_triplet_slot(mutant, CACHE, PLAN_HPP, SLOT_HPP)



# --- the plan copy shares its entries -------------------------------------------------------------------------------

# The transaction's plan copy (HOIST_PLAN) shares the published plan's entries: they live in a copy-on-write chunked
# store, so the copy costs one shared pointer per chunk and a flip clones only the chunk it writes. What this pins:
#   - the entries are that store, so HOIST_PLAN is a sharing copy, not an O(entries) one;
#   - the transaction takes the clones every prepared member's flip needs once the copy exists and before the first
#     move, so a flip allocates nothing and a moved triplet is flipped whatever followed (the claim above);
#   - the transaction reaches the copy only through the calls that write one indexed entry (the flips, the unshare)
#     or read it, and counts the chunks it cloned: a mutable iterator over the entries would clone every chunk, so
#     neither the transaction nor the two flips names one;
#   - an entry's position is named by its index, never by pointer arithmetic on the store, which a chunked store
#     cannot answer.
ENTRY_STORE = "using placement_entry_store = cow_chunked_vector<placement_entry>;"
ENTRY_MEMBER = "placement_entry_store entries;"
UNSHARE_LOOP = ("for (const ggml_sycl_expert_prepared_triplet & triplet : prepared) { for (int r = 0; r < 3; ++r) { "
                "const ggml_sycl::expert_placement_key & k = triplet.members[r].placement; "
                "(void) next_plan->unshare_expert_entry(k.layer_id, k.expert_id, k.role); } }")
UNSHARE_SIG = "bool unshare_expert_entry(int layer_id, int expert_id, expert_tensor_role role)"
UNSHARE_BODY = "entries.unshare_at(it->second);"
UPDATE_SIG = ("bool update_expert_placement(int layer_id, int expert_id, expert_tensor_role role, bool on_device, "
              "int target)")
BACKING_SIG = ("bool set_expert_host_backing(int layer_id, int expert_id, expert_tensor_role role, "
               "placement_host_backing backing)")
ENTRY_WRITE = "placement_entry & e = entries[it->second];"
COUNT_CLONES = "result.plan_chunks_cloned = next_plan->entries.chunks_cloned();"
# Every member the transaction names on its plan copy, and every call it hands the whole copy to (each takes it const).
TXN_PLAN_MEMBERS = {"unshare_expert_entry", "update_expert_placement", "set_expert_host_backing",
                    "host_backing_maps_files", "entries"}
TXN_PLAN_WHOLE = ("ggml_sycl::moe_expert_triplet_dst_bytes(*next_plan, placed)",
                  "ggml_sycl::moe_mmid_same_workspace_plan(*current->plan, *next_plan)")
# Each of those takes the plan const, so handing it the copy reads its entries without cloning a chunk: a writable
# parameter would make every entry read there a non-const index, which clones the chunk it lands in.
TXN_PLAN_WHOLE_CONST = (("plan_hpp", "static inline size_t moe_expert_triplet_dst_bytes(const placement_plan & plan,"),
                        ("cache_hpp", "bool moe_mmid_same_workspace_plan(const placement_plan & a, "
                                      "const placement_plan & b) noexcept;"))
ENTRIES_WORD_RE = re.compile(r"\bentries\b")
# A per-(layer, expert) map on the plan is copied whole by HOIST_PLAN, node by node, and a flip that writes it can
# allocate. The plan held one (expert_device, a collapsed summary nothing read); the entries and their shared index are
# the one per-expert record.
PER_EXPERT_MAP = "std::unordered_map<int, std::unordered_map<int, int>>"
ENTRY_POINTER_ARITH_RE = re.compile(r"entries\s*\.\s*data\s*\(|-\s*&\s*\w+(\.|->)entries\s*\[")


def claim_the_plan_copy_shares_its_entries(cache_hpp: str, cache: str, sycl: str, plan_hpp: str = PLAN_HPP) -> bool:
    h = norm(cache_hpp)
    by_file = {"plan_hpp": norm(plan_hpp), "cache_hpp": h}
    b = body(norm(sycl), TXN_SIG)
    lend_at = b.find(LEND)
    members = set(re.findall(r"next_plan->(\w+)", b))
    whole = b.count("*next_plan")
    flips = [body(h, UPDATE_SIG), body(h, BACKING_SIG)]
    unshare = body(h, UNSHARE_SIG)
    plan_struct = body(h, "struct placement_plan ")
    return (h.count(ENTRY_STORE) == 1 and h.count(ENTRY_MEMBER) == 1 and b.count(HOIST_PLAN) == 1
            and ENTRY_MEMBER in plan_struct and PER_EXPERT_MAP not in plan_struct
            and not re.search(r"\bexpert_device\b", h)
            and b.count(UNSHARE_LOOP) == 1 and lend_at >= 0 and ordered(b, HOIST_PLAN, UNSHARE_LOOP, LEND)
            and members == TXN_PLAN_MEMBERS and b.count("next_plan->entries") == 1 and b.count(COUNT_CLONES) == 1
            and whole == len(TXN_PLAN_WHOLE) and all(b.count(x) == 1 for x in TXN_PLAN_WHOLE)
            and all(by_file[f].count(sig) == 1 for f, sig in TXN_PLAN_WHOLE_CONST)
            and all(f and len(ENTRIES_WORD_RE.findall(f)) == 1 and f.count(ENTRY_WRITE) == 1 for f in flips)
            and len(ENTRIES_WORD_RE.findall(unshare)) == 1 and unshare.count(UNSHARE_BODY) == 1
            and not any(ENTRY_POINTER_ARITH_RE.search(norm(src)) for src in (cache_hpp, cache, sycl)))


def test_the_plan_copy_shares_its_entries():
    assert claim_the_plan_copy_shares_its_entries(CACHE_HPP, CACHE, SYCL)


def test_mutant_plan_entries_in_a_plain_vector_fails():
    """A plain vector makes HOIST_PLAN copy every entry per transaction."""
    mutant = _once(CACHE_HPP, ENTRY_STORE, "using placement_entry_store = std::vector<placement_entry>;")
    assert not claim_the_plan_copy_shares_its_entries(mutant, CACHE, SYCL)


def test_mutant_plan_keeps_a_per_expert_map_fails():
    """The collapsed per-expert device summary, back on the plan: HOIST_PLAN copies it node by node."""
    mutant = _once(CACHE_HPP, ENTRY_MEMBER, ENTRY_MEMBER + " " + PER_EXPERT_MAP + " expert_device;")
    assert not claim_the_plan_copy_shares_its_entries(mutant, CACHE, SYCL)


def test_mutant_flips_clone_after_the_first_move_fails():
    """Without the clones taken first, a flip after a move can throw and leave a moved triplet unflipped."""
    assert not claim_the_plan_copy_shares_its_entries(CACHE_HPP, CACHE, _once_in(SYCL, TXN_SIG, UNSHARE_LOOP, ""))


def test_mutant_clones_taken_inside_the_move_loop_fails():
    moved = _once_in(SYCL, TXN_SIG, UNSHARE_LOOP, "")
    mutant = _once_in(moved, TXN_SIG, "loaned = " + LEND + ";", "loaned = " + LEND + "; " + UNSHARE_LOOP)
    assert not claim_the_plan_copy_shares_its_entries(CACHE_HPP, CACHE, mutant)


def test_mutant_unshare_clones_nothing_fails():
    mutant = _once_in(CACHE_HPP, UNSHARE_SIG, UNSHARE_BODY, "")
    assert not claim_the_plan_copy_shares_its_entries(mutant, CACHE, SYCL)


def test_mutant_transaction_iterates_the_copy_mutably_fails():
    """A range-for over the non-const copy takes the mutable begin(), which clones every chunk."""
    mutant = _once_in(SYCL, TXN_SIG, COUNT_CLONES,
                      "for (ggml_sycl::placement_entry & e : next_plan->entries) { (void) e; } " + COUNT_CLONES)
    assert not claim_the_plan_copy_shares_its_entries(CACHE_HPP, CACHE, mutant)


def test_mutant_transaction_hands_the_copy_to_another_writer_fails():
    mutant = _once_in(SYCL, TXN_SIG, COUNT_CLONES, "ggml_sycl::placement_entries_charge_expert_slots("
                      "next_plan->entries); " + COUNT_CLONES)
    assert not claim_the_plan_copy_shares_its_entries(CACHE_HPP, CACHE, mutant)


def test_mutant_transaction_calls_another_plan_member_fails():
    mutant = _once_in(SYCL, TXN_SIG, COUNT_CLONES, "next_plan->build_index(); " + COUNT_CLONES)
    assert not claim_the_plan_copy_shares_its_entries(CACHE_HPP, CACHE, mutant)


def test_mutant_flip_walks_the_entries_fails():
    mutant = _once_in(CACHE_HPP, UPDATE_SIG, ENTRY_WRITE,
                      "for (placement_entry & x : entries) { (void) x; } " + ENTRY_WRITE)
    assert not claim_the_plan_copy_shares_its_entries(mutant, CACHE, SYCL)


def test_mutant_backing_flip_walks_the_entries_fails():
    mutant = _once_in(CACHE_HPP, BACKING_SIG, ENTRY_WRITE,
                      "for (placement_entry & x : entries) { (void) x; } " + ENTRY_WRITE)
    assert not claim_the_plan_copy_shares_its_entries(mutant, CACHE, SYCL)


def test_mutant_triplet_bytes_takes_the_plan_writable_fails():
    """The transaction's triplet sizing, handed the copy writable: every entry it reads would clone its chunk."""
    mutant = _once(PLAN_HPP, TXN_PLAN_WHOLE_CONST[0][1],
                   "static inline size_t moe_expert_triplet_dst_bytes(placement_plan & plan,")
    assert not claim_the_plan_copy_shares_its_entries(CACHE_HPP, CACHE, SYCL, mutant)


def test_mutant_same_workspace_takes_the_plan_writable_fails():
    """The workspace comparison, handed the copy writable: every entry it reads would clone its chunk."""
    mutant = _once(CACHE_HPP, TXN_PLAN_WHOLE_CONST[1][1],
                   "bool moe_mmid_same_workspace_plan(const placement_plan & a, placement_plan & b) noexcept;")
    assert not claim_the_plan_copy_shares_its_entries(mutant, CACHE, SYCL)


def test_mutant_entry_position_by_pointer_arithmetic_fails():
    mutant = norm(CACHE) + " size_t i = static_cast<size_t>(&entry - plan.entries.data());"
    assert not claim_the_plan_copy_shares_its_entries(CACHE_HPP, mutant, SYCL)
    mutant = norm(CACHE) + " size_t i = static_cast<size_t>(p - &plan.entries[0]);"
    assert not claim_the_plan_copy_shares_its_entries(CACHE_HPP, mutant, SYCL)

if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
