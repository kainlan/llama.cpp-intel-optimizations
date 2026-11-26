#!/usr/bin/env python3
"""Source contract: the decode-boundary expert swap driver runs once per boundary, never waits and never holds a lock
across its policy.

Host-only: reads ggml-sycl.cpp, moe-swap-driver.hpp, moe-stream-plan.hpp and unified-cache.cpp, builds nothing, loads
no model and touches no device.

The driver (GGML_SYCL_MOE_SWAP=1) asks the swap policy at each synchronize boundary which resident triplet leaves and
which host triplet enters, and sends the answer through the named move. tests/test-sycl-moe-swap-driver.cpp shows the
decisions and the bookkeeping are right; the loan test's device case shows an evict into a full host zone is BUSY. What
a unit test cannot show is where the driver runs and what it does not do there. This gate holds that by text:

  D1  one call site: ggml_backend_sycl_synchronize() calls the driver once, after the refill, and nothing else does;
      the driver is two functions, the gated boundary (ggml_sycl_run_expert_swap_at_boundary: the gates, the claim and
      the graph counts) and the claimed body it calls once after the claim (ggml_sycl_run_expert_swap_claimed), which
      reads the world and runs the step. D2 to D20 read the pair as one function;
  D2  the policy step runs once, inside the driver;
  D3  no host wait and no event status query in the driver, its helpers (every ggml_sycl_moe_swap_* function), the
      driver header or the plan-view builder: no wait() or wait_and_throw(), no retained-handle drain, no event
      get_info or command status, no synchronize, no sleep, no condition variable;
  D4  no lock is held across the policy step: every lock the driver takes is scoped to a block that ends before the
      step, or opens after it, and none is taken by a bare lock() call;
  D5  the driver is gated before the step: the switch, the multithreaded-graph latch, the compute flag, the graphs in
      flight, and the claim of the device's instance, in that order;
  D6  the switch is read once: GGML_SYCL_MOE_SWAP is spelled once, as the header's MOE_SWAP_ENV; its one getenv is in
      the accessor's static initializer; the settings table names it once and its row is overwritten from the
      accessor after the parse, so the non-default report says what is in effect;
  D7  each summary line reaches the log at WARN: every summary call is an argument of GGML_LOG_WARN, and both exist;
      the second line names the accepted transactions, evicts and admits apart, beside the decode graphs, and the
      step counts each direction's accepted transaction once (each one can cost a context what it recorded); the
      second line also names the moves of a thrown batch still unsettled at the report, a count each throw sets;
  D8  the driver header is plain host code: it includes the policy header and a short list of standard headers, and
      keeps no state outside the caller's struct (no static, thread_local, extern, mutable or volatile, no inline
      variable, no atomic or lock), reaches no device, cache, log or environment, and has no exception handler (no
      try or catch): a handler there could swallow what the transaction throws, and the header cannot name the
      planned refusal a handler must let through;
  D9  the plan-view builder keeps no state and takes no lock, logs nothing and reads no environment;
  D10 a named evict's host copy never grows the pinned host zone: the to-host branch mints an evict's copy through
      ggml_sycl_host_expert_alloc_no_growth, while a loan's copy keeps the growing mint. There is one host mint:
      ggml_sycl_host_expert_alloc takes grow_zone, and with it false its request forbids host-zone growth and keeps
      the miss out of the error log; the no-growth name only calls it with false;
  D11 the route counters are copied into the driver's own snapshot arrays: the step calls the counts callback once,
      with d.decode.data() and d.prompt.data(), the arrays the policy's block inputs point at;
  D12 a final refusal reaches the log at WARN once per reason and direction: the caller's transaction callback calls
      ggml_sycl_moe_swap_warn_refusal for every move; the helper passes over every outcome but REFUSED, keeps the
      reasons it logged in the instance's per-direction sets (touched nowhere else), and logs only a reason it had
      not logged;
  D13 the refusal that latches a block has one spelling: the plan header defines MOE_EXPERT_MOVE_SHARED_STORAGE as
      the only "shared storage" literal in the four sources; the cache's loan assigns it by name; the caller marks a
      move as latching only through ggml_sycl_moe_swap_refusal_latches_block, which compares the reason to that
      constant for a refused evict; and the step latches a block only when the move says so;
  D14 a device where no driver ran still reports: ggml_sycl_moe_swap_report, with no driver, takes both counters
      kept apart and prints them in one WARN line when either is nonzero;
  D15 the evict batch's room and the evict mint's cap are one figure: the driver's room callback returns
      unified_cache::host_zone_no_growth_room for the WEIGHT zone, the mint refuses a no-growth request past
      host_zone_no_growth_bound for that zone, and the accessor is that same bound (the zone's capacity when it has
      none) less what the zone holds, never below zero;
  D16 a replaced route table's driver hands over what is in flight only under the load the instance last saw: the
      caller carries only when the snapshot's model and load ids both match, otherwise drops the moves as open slots
      and forces the old driver's report when it dropped any, and the report prints when forced even if an earlier
      one covered every boundary;
  D17 a transaction that throws leaves its batch unsettled through an unwind guard, not a handler: the step calls the
      transaction twice (admits, evicts), and before each call it reserves the unsettled list, and the busy reasons
      when the guard counts one, each past its size (size() + n, never a bare count), then declares a guard whose
      body sets the unsettled count; the line right after the call dismisses that guard. The guard's body runs while
      a throw unwinds, where a throw ends the process, so the reserves are what make its pushes safe. The same
      reserves cover what the step records once the transaction has answered, since a moved triplet cannot be taken
      back and an allocation that failed in the middle of the recording would leave the driver not knowing what
      moved: before the admit call the busy, refused and dropped reason lists, and before the evict call the pending
      list and the busy and refused reason lists;
  D18 the hold of an admit refused for device room has its two inputs: the caller hands the step the device WEIGHT
      zone's room epoch (unified_cache::zone_room_epoch), read before the step; and the refusal's cause reaches the
      move typed, not as text: the preparation leaves the size of a refused device mint (what the member asked: its
      triplet's slot, else its own staged size) and what the whole triplet asks (expert_triplet_zone_ask_of: its
      slot and its members' bytes summed) on the prepared triplet, right where the mint fails, and reads no zone
      figures there; the transaction marks the named triplet's outcome when that refusal is transient (the "WEIGHT
      zone full" answer, and no other), and reads the WEIGHT zone's largest free block onto it, before the refused
      triplet's members go back, only for a refusal ggml_sycl_expert_zone_refusal_kept says the boundary's line keeps
      (the read walks the zone's top free class under its lock); and the callback gives the move
      MOE_SWAP_BUSY_CAUSE_ZONE_ROOM from the outcome. Without either the driver holds nothing, sends a refused admit
      again at every boundary, and no unit test sees it;
  D19 a boundary whose transaction refused admits for device room says so in one WARN line: the callback counts the
      answers that carry a refused size and keeps, by ggml_sycl_expert_zone_refusal_kept (the first, then each that
      asks more in all, the one rule the transaction's read follows too), the refused triplet's ask with its refused
      member's size and the largest free block at its refusal, all three of that one refusal, and after its loop over
      the batch, on that count alone (so a batch that moved nothing prints it too, and no admit prints its own), calls
      ggml_sycl_moe_swap_warn_zone_refusal once with the admits it sent; the helper reads the WEIGHT zone's free
      figures together (unified_cache::zone_free_figures) and logs moe_swap_zone_refusal_line at WARN,
      and once per swap instance, at the first such boundary whose zone free blocks it can read (a flag on the
      instance, set after that read and before it prints), logs them (unified_cache::zone_free_block_sizes) as
      moe_swap_zone_free_census_line at WARN;
  D20 the hold takes the transaction's own give-backs out of the epoch's move. A transaction that gives back storage
      it took moves the room epoch with no room gained, so an unmeasured give-back sends the refused admit again at
      every boundary. The meter (ggml_sycl_expert_room_meter) reads the WEIGHT zone's epoch right before and right
      after a give-back and sums the moves; the named move measures both of its give-backs with it: a refused
      triplet's minted members, in the named-refusal branch, and the prepared list, in a guard declared before the
      first triplet is prepared, which is the one writer of the answer's room_self_delta; the entry carries the sum;
      the callback reads the epoch right after the transaction returns and returns it with the sum; and the step
      holds a refused admit, and lets a held one follow the epoch, only when moe_swap_room_stood says the epoch's
      move since the boundary's read is that sum, under the epoch the transaction left. No host test runs the
      transaction's side of this, so the text is its only gate here;
  D21 the room epoch moves wherever the zone's allocators gain room. The hold waits for that move, so a site that
      stops making it holds admits silently to the bound, and no error says so. unified_cache::zone_free gives a block
      back in four arms (in the N-chunk mode the KV zone's own allocator and a chunk's, then the allocator WEIGHT
      shares with KV, then a zone's own), and each calls arena_note_room_grew for the zone's group right after its
      zone_used_free; arena_group_reset_allocators_locked calls it first; arena_note_allocators_made calls it for
      every group, and arena_reserve calls that at both places it makes the allocators; and arena_note_room_grew adds
      one to the group's epoch. zone_alloc's own zone_used_free calls undo an allocation that could not be recorded:
      no room is gained there, and none is pinned.

  D22 the claimed body has a test hook: it takes (device, decodes, graphs, resolver) and reads nothing of the backend
      context, so the private device tests run it without a context; the gated boundary calls it once, after the
      claim, with no resolver; the call the driver is handed is built by ggml_sycl_moe_swap_make_call, which the tests
      read as well; the named move is sent through the resolver the body was given; and the one other caller, the
      private tests' hook, claims the device's instance before the call and releases it after.

Checks run on comment-stripped text, string literals kept. NOT VACUOUS: --self-test runs the gate on in-memory mutants
of the real sources. Each mutant names the pin that must catch it, and is reported if it survives, if a different pin
catches it, or if its anchor no longer matches exactly once (NOT APPLIED). The self-test also proves that last report
fires. ctest registers it with --self-test.

WHAT THIS DOES NOT PROVE. It reads text. A wait reached through a function the driver calls by another name is not
seen (the named move is the admit and loan pins' business, and it waits on nothing); a lock taken inside a callee is
not seen. That the decisions are right is the unit test's job; that an evict into a full zone answers BUSY on a device
is the loan test's.
"""

from __future__ import annotations

import functools
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "sycl": ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp",
    "driver": ROOT / "ggml/src/ggml-sycl/moe-swap-driver.hpp",
    "plan": ROOT / "ggml/src/ggml-sycl/moe-stream-plan.hpp",
    "cache": ROOT / "ggml/src/ggml-sycl/unified-cache.cpp",
}

SYNC_SIG = "static void ggml_backend_sycl_synchronize(ggml_backend_t backend) {"
CALLER_SIG = "static void ggml_sycl_run_expert_swap_at_boundary("
CALLER_NAME = "ggml_sycl_run_expert_swap_at_boundary("
CORE_SIG = "static void ggml_sycl_run_expert_swap_claimed("
CORE_CALL = "ggml_sycl_run_expert_swap_claimed("
CALL_SIG = "static ggml_sycl::moe_swap_driver_call ggml_sycl_moe_swap_make_call("
HOOK_SIG = "bool test_swap_boundary("
CORE_DECL = ("static void ggml_sycl_run_expert_swap_claimed(int device, uint64_t decodes, uint64_t graphs, "
             "const ggml_sycl_expert_loan_resolver * resolver) {")
CORE_OUTER_CALL = "ggml_sycl_run_expert_swap_claimed(device, decodes, graphs, nullptr);"
CORE_MAKE_CALL = "ggml_sycl_moe_swap_make_call(inst, cache, device, decodes, graphs);"
CORE_TXN = "ggml_sycl_expert_move_transaction_run(device, request, resolver, &result);"
# The one way an instance is claimed: a guard that tries on construction and releases on every way out.
CLAIM_SIG = "struct ggml_sycl_moe_swap_claim {"
CLAIM_DECL = "ggml_sycl_moe_swap_claim claim{ inst.claimed };"
CLAIM_LOST = "if (!claim.held) {"
CLAIM_DECL_RE = re.compile(r"ggml_sycl_moe_swap_claim\s+claim\s*\{\s*inst\.claimed\s*\}\s*;")
CLAIM_TAKE = "return flag.compare_exchange_strong(expected, true, std::memory_order_acq_rel);"
REPORT_AT_FREE_SIG = "static void ggml_sycl_moe_swap_report_at_free("
REPORT_IF_HELD = "if (claim.held) { ggml_sycl_moe_swap_report(inst, device); }"
CLAIM_GIVE_BACK = "if (held) { claimed.store(false, std::memory_order_release); }"
CALLER_CALL = "ggml_sycl_run_expert_swap_at_boundary(backend);"
REFILL_CALL = "ggml_sycl_run_expert_refill_at_boundary(backend);"
STEP_CALL = "moe_swap_driver_step("
BUILDER_SIG = "static inline void moe_swap_plan_view_build("
ACCESSOR_SIG = "static bool ggml_sycl_moe_swap_enabled("
PREP_SIG = "static const char * ggml_sycl_expert_prepare_triplet("
NO_GROWTH_SIG = "static ggml_sycl::mem_handle ggml_sycl_host_expert_alloc_no_growth("
MINT_SIG = "static ggml_sycl::mem_handle ggml_sycl_host_expert_alloc(int device, size_t size, bool grow_zone) {"
STEP_SIG = "moe_swap_driver_status moe_swap_driver_step("
WARN_SIG = "static void ggml_sycl_moe_swap_warn_refusal("
LATCH_SIG = "static bool ggml_sycl_moe_swap_refusal_latches_block("
REPORT_SIG = "static void ggml_sycl_moe_swap_report("
TXN_SIG = "static ggml_sycl_txn_result ggml_sycl_expert_txn_run("
ZONE_WARN_SIG = "static void ggml_sycl_moe_swap_warn_zone_refusal("
ROOM_SIG = "static size_t ggml_sycl_moe_swap_host_room("
LOAN_SIG = "bool unified_cache::loan_expert_triplet("
NO_GROWTH_ROOM_SIG = "size_t unified_cache::host_zone_no_growth_room("
HELPER_DEF = re.compile(r"^static\s+[\w:<>,\s\*&]*?\b(ggml_sycl_moe_swap_\w+)\s*\(", re.MULTILINE)

GATES = ("ggml_sycl_moe_swap_enabled()", "g_sycl_graph_multithreaded", "unified_cache_is_graph_compute_active()",
         "ggml_sycl_graph_inflight_count()", CLAIM_DECL)
TABLE_ROW = re.compile(r"\{\s*ggml_sycl::MOE_SWAP_ENV\s*,\s*&g_ggml_sycl_moe_swap\s*,\s*0\s*\}")
ROW_OVERWRITE = re.compile(r"g_ggml_sycl_moe_swap\s*=\s*ggml_sycl_moe_swap_enabled\(\)\s*\?\s*1\s*:\s*0\s*;")
PARSE_LOOP = "*setting.dest = get_sycl_env(setting.name, setting.default_value);"
ENV_SPELLING = re.compile(r"MOE_SWAP_ENV\[\]\s*=\s*\"GGML_SYCL_MOE_SWAP\"\s*;")
SUMMARIES = ("moe_swap_driver_summary_moves(", "moe_swap_driver_summary_faults(")
FAULTS_SIG = "inline std::string moe_swap_driver_summary_faults("
FAULTS_FIELDS = ("accepted transactions %llu (evict %llu, admit %llu) over %llu decode graphs",
                 "(ull) (st.transactions_evict + st.transactions_admit), (ull) st.transactions_evict,",
                 "(ull) st.transactions_admit, (ull) st.decode_graphs,")
TXN_COUNTS = ("++st.transactions_evict;", "++st.transactions_admit;")
UNSETTLED_FIELDS = ("unsettled at report: %llu", "(ull) st.unsettled_moves)")
UNSETTLED_SET = "st.unsettled_moves = d.unsettled.size();"
TXN_CALL = "txn(d.batch);"
GUARD_DECL = re.compile(r"\bmoe_swap_detail::moe_swap_unwind_guard\s+(\w+)\s*\(")
# Each reserve grows its list past what it already holds: a bare count can be below the size, and reserves nothing.
UNSETTLED_RESERVE = re.compile(r"\bd\.unsettled\.reserve\(\s*d\.unsettled\.size\(\)\s*\+\s*\w")
BUSY_RESERVE = re.compile(r"\bst\.busy\.reserve\(\s*st\.busy\.size\(\)\s*\+\s*\w")


def _reserve_past_size(name: str) -> re.Pattern:
    return re.compile(r"\b" + re.escape(name) + r"\.reserve\(\s*" + re.escape(name) + r"\.size\(\)\s*\+\s*\w")


# What each transaction call's answers are recorded into, which the step reserves before the call: the admits' call
# (the first) counts reasons, the evicts' call (the second) also pushes the pending admits.
RECORDING_RESERVES = (("st.busy", "st.refused", "st.dropped"), ("d.pending", "st.busy", "st.refused"))
HANDLER = re.compile(r"\b(?:try|catch)\b")
LITERAL = re.compile(r"\"(?:\\.|[^\"\\\n])*\"")
SAME_LOAD = "const bool same_load = snap && snap->model_id == inst.model_id && snap->load_txn_id == inst.load_txn_id;"
CARRY_GATE = ("if (!same_load || !ggml_sycl::moe_swap_driver_carry(*inst.driver, *fresh)) { "
              "dropped = ggml_sycl::moe_swap_driver_drop_in_flight(*inst.driver); }")
FORCED_REPORT = "ggml_sycl_moe_swap_report(inst, device, dropped != 0);"
REPORT_FORCE = "if (!force && inst.driver->stats.boundaries == inst.reported) { return; }"
EVICT_MINT = re.compile(r"if\s*\(\s*out\.kind\s*==\s*ggml_sycl_expert_move_kind::EVICT\s*\)\s*\{\s*"
                        r"storage\s*=\s*ggml_sycl_host_expert_alloc_no_growth\(\s*device\s*,\s*target\.logical_bytes\s*\)"
                        r"\s*;\s*\}\s*else\s*\{\s*storage\s*=\s*ggml_sycl_host_expert_alloc\(\s*device\s*,\s*"
                        r"target\.logical_bytes\s*\)\s*;\s*\}")
MINT_FIELDS = ("req.intent.role = ggml_sycl::alloc_role::WEIGHT;", "req.intent.constraints.must_host_pinned = true;",
               "req.intent.constraints.use_pinned_pool = true;",
               "req.intent.constraints.forbid_host_zone_growth = !grow_zone;", "req.suppress_failure_log = !grow_zone;",
               "unified_allocate_owner(req)")
NO_GROWTH_CALL = "return ggml_sycl_host_expert_alloc(device, size, false);"
COUNTS_CALL = "counts(d.decode.data(), d.prompt.data());"
WARN_CALL = "ggml_sycl_moe_swap_warn_refusal(inst, device, m);"
WARN_FIELDS = ("if (m.outcome != ggml_sycl::MOE_SWAP_MOVE_REFUSED) { return; }",
               "evict ? inst.refusals_warned_evict : inst.refusals_warned_admit;",
               "if (!warned.insert(reason).second) { return; }",
               'GGML_LOG_WARN("[MOE-SWAP] device %d: %s refused: %s (logged once per reason)\\n", device, '
               'evict ? "evict" : "admit", reason);')
SHARED_LITERAL = '"shared storage"'
SHARED_DEF = 'constexpr const char MOE_EXPERT_MOVE_SHARED_STORAGE[] = "shared storage";'
SHARED_ASSIGN = "refusal = MOE_EXPERT_MOVE_SHARED_STORAGE;"
LATCH_FIELDS = ("m.kind == ggml_sycl::MOE_SWAP_MOVE_EVICT", "m.outcome == ggml_sycl::MOE_SWAP_MOVE_REFUSED",
                "std::strcmp(m.reason, ggml_sycl::MOE_EXPERT_MOVE_SHARED_STORAGE) == 0")
LATCH_SET = "m.latches_block = ggml_sycl_moe_swap_refusal_latches_block(m);"
STEP_LATCH = "if (m.latches_block) { d.latched[static_cast<size_t>(q.block)] = 1; }"
NO_DRIVER_FIELDS = ("const uint64_t off = inst.off_multithreaded.exchange(0, std::memory_order_relaxed);",
                    "const uint64_t skipped = inst.skipped.exchange(0, std::memory_order_relaxed);",
                    "if (off != 0 || skipped != 0) { GGML_LOG_WARN(\"[MOE-SWAP] device %d: no driver ran: ")
ROOM_RETURN = "return cache->host_zone_no_growth_room(ggml_sycl::host_zone_id::WEIGHT);"
MINT_BOUND = ("const size_t bound = host_zone_no_growth_bound(zone, ucache->host_zone_reserved_capacity(zone), "
              "ucache->host_zone_capacity(zone));")
MINT_CAP = "if (bound > 0 && ucache->host_zone_used(zone) + alloc_size > bound) {"
ROOM_BOUND = "host_zone_no_growth_bound(zone, host_zone_reserved_capacity(zone), capacity)"
NO_GROWTH_ROOM_FIELDS = (f"const size_t bound = {ROOM_BOUND};", "const size_t limit = bound > 0 ? bound : capacity;",
                         "return limit > used ? limit - used : 0;")
ROOM_EPOCH_READ = "call.room_epoch = cache ? cache->zone_room_epoch(ggml_sycl::vram_zone_id::WEIGHT) : 0;"
BUSY_CAUSE_SET = ("m.busy_cause = o.zone_room_bytes != 0 ? ggml_sycl::MOE_SWAP_BUSY_CAUSE_ZONE_ROOM : "
                  "ggml_sycl::MOE_SWAP_BUSY_CAUSE_TRANSIENT;")
ZONE_ROOM_MINT = ("storage = ggml_sycl_device_expert_alloc(device, member_bytes); if (!storage.valid()) { "
                  "out.mint_refused_bytes = member_bytes; "
                  "out.mint_refused_ask = ggml_sycl::expert_triplet_zone_ask_of(staged); const bool arena =")
ZONE_FREE_READ = ("if ((*named_moves)[triplet.named].zone_room_bytes != 0 && ggml_sycl_expert_zone_refusal_kept("
                  "zone_refusal_read, zone_refusal_read_ask, triplet.mint_refused_ask)) { "
                  "size_t capacity = 0; size_t used = 0; size_t avail = 0; "
                  "(void) cache->zone_free_figures(ggml_sycl::vram_zone_id::WEIGHT, capacity, used, avail, "
                  "(*named_moves)[triplet.named].zone_largest_free_at_refusal); "
                  "zone_refusal_read = true; zone_refusal_read_ask = triplet.mint_refused_ask; }")
# The refused triplet's members going back, whatever runs the release.
ZONE_REFUSED_GIVE_BACK = ("([&triplet] { for (ggml_sycl::expert_move_member & m : triplet.members) { "
                          "m.owner = ggml_sycl::mem_handle{}; } });")
KEPT_SIG = "static bool ggml_sycl_expert_zone_refusal_kept("
# The census and zone_largest_free read the allocators one visitor names, so the census counts the blocks whose top
# zone_largest_free reports, in the same three cases.
ZONE_VISIT_SIG = "template <typename F> bool unified_cache::arena_visit_zone_allocators("
ZONE_CENSUS_SIG = "bool unified_cache::zone_free_block_sizes("
ZONE_LARGEST_SIG = "size_t unified_cache::zone_largest_free("
ZONE_VISIT_CASES = ("if (arena_chunks_.size() == 1 && zone == vram_zone_id::WEIGHT) { "
                    "if (const tlsf_allocator * kv = arena_zones_[static_cast<int>(vram_zone_id::KV)].allocator.get()) "
                    "{ visit(*kv); } return true; }",
                    "if ((zone == vram_zone_id::WEIGHT || zone == vram_zone_id::KV) && "
                    "!weight_chunk_allocators_.empty()) { if (zone == vram_zone_id::KV && z.allocator) { "
                    "visit(*z.allocator); } for (const auto & wca : weight_chunk_allocators_) { "
                    "visit(*wca.allocator); } return true; }",
                    "if (z.allocator) { visit(*z.allocator); return true; } return false;")
ZONE_CENSUS_VISIT = "(void) arena_visit_zone_allocators(zone, [&sizes](const tlsf_allocator & allocator) {"
ZONE_LARGEST_VISIT = "const bool served = arena_visit_zone_allocators(zone, [&best](const tlsf_allocator & allocator) {"
ZONE_OWN_CHOICE = ("arena_zones_", "weight_chunk_allocators_", "arena_chunks_")
KEPT_BODY = "{ return !any_kept || ask.total > kept.total; }"
ZONE_ROOM_OUTCOME = "(*named_moves)[triplet.named].zone_room_bytes = transient ? triplet.mint_refused_bytes : 0;"
ZONE_ASK_OUTCOME = ("(*named_moves)[triplet.named].zone_room_ask = transient ? triplet.mint_refused_ask : "
                    "ggml_sycl::expert_triplet_zone_ask{};")
ZONE_REFUSED_COUNT = ("if (o.zone_room_bytes != 0) { ++zone_refused; "
                      "if (ggml_sycl_expert_zone_refusal_kept(zone_refused_member != 0, zone_refused_ask, "
                      "o.zone_room_ask)) { "
                      "zone_refused_member = o.zone_room_bytes; zone_refused_ask = o.zone_room_ask; "
                      "zone_refused_free = o.zone_largest_free_at_refusal; } }")
# The loop over the batch ends on the refusal WARN's call; the zone-refusal line comes right after it, on the count.
ZONE_WARN_AFTER_LOOP = (WARN_CALL + " } if (zone_refused != 0) { ggml_sycl_moe_swap_warn_zone_refusal(inst, cache, "
                        "device, zone_refused, request.admit.size(), zone_refused_member, zone_refused_ask, "
                        "zone_refused_free); }")
ZONE_WARN_FIELDS = ("(void) cache->zone_free_figures(ggml_sycl::vram_zone_id::WEIGHT, capacity, used, available, "
                    "largest_free);",
                    "const std::string line = ggml_sycl::moe_swap_zone_refusal_line(device, refused, admits, "
                    "ask.slot, ask.total, largest_at_refusal, available, largest_free); "
                    'GGML_LOG_WARN("%s\\n", line.c_str());')
ZONE_CENSUS_ONCE = "if (inst.zone_census_printed || !cache) { return; }"
ZONE_CENSUS_READ = ("if (!cache->zone_free_block_sizes(ggml_sycl::vram_zone_id::WEIGHT, free_sizes)) { return; } "
                    "inst.zone_census_printed = true;")
ZONE_CENSUS_BLOCK = ("const size_t member_block = ggml_sycl::tlsf_allocator::round_request(member, "
                     "ggml_sycl::tlsf_allocator::block_grain);")
ZONE_CENSUS_LOG = ("const std::string census = ggml_sycl::moe_swap_zone_free_census_line(device, free_sizes, member, "
                   "member_block, "
                   'GGML_SYCL_MOE_SWAP_ZONE_CENSUS_TOP); GGML_LOG_WARN("%s\\n", census.c_str());')
METER_SIG = "struct ggml_sycl_expert_room_meter {"
GIVE_BACK_SIG = "struct ggml_sycl_expert_prepared_give_back {"
ENTRY_SIG = "static ggml_sycl_txn_result ggml_sycl_expert_move_transaction_run("
ROOM_STOOD_SIG = "inline bool moe_swap_room_stood("
METER_MEASURE = ("const uint64_t before = cache->zone_room_epoch(ggml_sycl::vram_zone_id::WEIGHT); release(); "
                 "moved += cache->zone_room_epoch(ggml_sycl::vram_zone_id::WEIGHT) - before;")
GIVE_BACK_FIELDS = ("meter.give_back([this] { prepared.clear(); });", "answer->room_self_delta = meter.moved;")
GIVE_BACK_DECL = ("ggml_sycl_expert_room_meter room_meter{ named ? cache : nullptr }; "
                  "ggml_sycl_expert_prepared_give_back give_back{ prepared, room_meter, named ? out : nullptr, "
                  "meta_us, copy_us, remap_us };")
PREPARE_CALL = "stop = ggml_sycl_expert_prepare_triplet("
REFUSED_GIVE_BACK = ("room_meter.give_back([&triplet] { for (ggml_sycl::expert_move_member & m : triplet.members) { "
                     "m.owner = ggml_sycl::mem_handle{}; } });")
REFILL_BRANCH = "} else if (refill && !transient) {"
ENTRY_SELF_DELTA = "r.room_self_delta = answer.room_self_delta;"
# The entry carries the body's three parts and times the whole call itself, so a refusal before any preparation has a
# total too.
ENTRY_COST = ("r.meta_us = answer.meta_us;", "r.copy_us = answer.copy_us;", "r.remap_us = answer.remap_us;",
              "r.total_us = ggml_sycl::expert_move_us_since(call_start);")
ENTRY_CLOCK = ("ggml_sycl_expert_loan_result answer; const auto call_start = std::chrono::steady_clock::now(); "
               "const ggml_sycl_txn_result code = ggml_sycl_expert_txn_run(")
COUNT_COST = ("moe_swap_detail::moe_swap_count_cost(st, room.cost);",
              "moe_swap_detail::moe_swap_count_cost(st, answer.cost);")
# The read comes right after the transaction's call, with nothing between them.
EPOCH_AFTER_READ = ("ggml_sycl_expert_move_transaction_run(device, request, resolver, &result); "
                    "const uint64_t epoch_after = cache ? cache->zone_room_epoch(ggml_sycl::vram_zone_id::WEIGHT) : 0;")
# The cost is built field by field: the result orders its parts meta, copy, remap and the cost copy, remap, meta.
TXN_COST_FIELDS = ("ggml_sycl::moe_swap_txn_cost cost; cost.copy_us = result.copy_us; cost.remap_us = result.remap_us; "
                   "cost.meta_us = result.meta_us; cost.total_us = result.total_us; ")
TXN_ROOM_RETURN = (TXN_COST_FIELDS +
                   "return ggml_sycl::moe_swap_txn_room{ epoch_after, result.room_self_delta, cost };")
ROOM_STOOD_BODY = "return room.epoch_after - before == room.self_delta;"
ADMIT_TXN = "const moe_swap_txn_room room = txn(d.batch);"
ROOM_STOOD_USE = "const bool room_stood = moe_swap_detail::moe_swap_room_stood(c.room_epoch, room);"
HELD_FOLLOWS = "p.room_epoch = room_stood ? room.epoch_after : p.room_epoch;"
HOLD_SET = ("p.room_held = m.busy_cause == MOE_SWAP_BUSY_CAUSE_ZONE_ROOM && room_stood; "
            "p.room_epoch = room.epoch_after;")
ZONE_FREE_SIG = "bool unified_cache::zone_free("
RESET_SIG = "void unified_cache::arena_group_reset_allocators_locked("
ROOM_GREW_SIG = "void unified_cache::arena_note_room_grew("
ALLOCATORS_MADE_SIG = "void unified_cache::arena_note_allocators_made("
ZONE_FREE_GIVES_BACK = re.compile(r"\bzone_used_free\([^;]*;")
ZONE_FREE_BUMP = "arena_note_room_grew(arena_allocator_group_index(zone));"
RESET_BUMP = "{ arena_note_room_grew(group);"
ROOM_GREW_BODY = "arena_allocator_groups_[group].room_epoch.fetch_add(1, std::memory_order_release);"
ALLOCATORS_MADE_BODY = ("for (size_t group = 0; group < ARENA_ALLOCATOR_GROUP_COUNT; ++group) { "
                        "arena_note_room_grew(group); }")
ALLOCATORS_MADE_CALL = "arena_note_allocators_made();"

WAITS = re.compile(
    r"(?:\.|->)\s*wait\s*\(|\bwait_and_throw\b|\bdrain_retained_handles\b|\bwait_all\b|\bcommand_execution_status\b|"
    r"\bget_info\s*<|\bevent_status\b|\bsynchronize\s*\(|\bsleep_for\b|\bsleep_until\b|\bcondition_variable\b"
)
LOCK_DECL = re.compile(r"\bstd::(?:lock_guard|unique_lock|scoped_lock)\b")
BARE_LOCK = re.compile(r"\.\s*(?:lock|try_lock)\s*\(\s*\)")

ALLOWED_INCLUDES = {"algorithm", "cstddef", "cstdint", "cstdio", "string", "vector"}
INCLUDE = re.compile(r"^[ \t]*#[ \t]*include[ \t]*(<[^>\n]*>|\"[^\"\n]*\")", re.MULTILINE)
STATE_KEYWORDS = re.compile(r"\b(?:static|thread_local|extern|mutable|volatile)\b")
INLINE_VAR = re.compile(r"\binline\s+(?!constexpr\b)(?:const\s+)?[\w:<>,\s\*&]+?\b\w+\s*(?:=|\{|\[|;)")
SYNC_NAMES = re.compile(r"\bstd::(?:atomic\w*|mutex|shared_mutex|recursive_mutex|lock_guard|unique_lock|scoped_lock|"
                        r"call_once|once_flag)\b")
FORBIDDEN = re.compile(
    r"\bsycl\b|\b(?:malloc|calloc|realloc|free|aligned_alloc)\s*\(|\bunified_\w+|\bmem_handle\b|\balloc_owner\b|"
    r"\bGGML_LOG\w*|(?<![\w])f?printf\s*\(|\bstd::(?:cout|cerr|clog)\b|\bgetenv\s*\("
)
BUILDER_FORBIDDEN = re.compile(
    r"\b(?:static|thread_local)\b|\bstd::(?:lock_guard|unique_lock|scoped_lock|mutex)\b|\bGGML_LOG\w*|\bgetenv\s*\("
)

_LEXEME_RE = re.compile(r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'", re.DOTALL)


class ContractError(AssertionError):
    pass


class MutantNotApplied(Exception):
    pass


def fail(pin: str, what: str) -> None:
    raise ContractError(f"FAIL [pin {pin}]: {what}")


# The self-test runs the gate on a mutant of each source in turn. The unmutated sources, and what is cut from them,
# stay cached because every mutant reads them; a mutated text is used for its one check and evicted. The caches are
# small on purpose: a key holds a whole 7 MB source, so a large cache would keep hundreds of dead mutants in memory. A
# cache hit returns what the same text gave before, so each mutant is still judged on its own text.
_BIG = 8  # entries holding a whole source or its stripped copy
_SMALL = 256  # entries cut from one


@functools.lru_cache(maxsize=_BIG)
def strip_comments(src: str) -> str:
    """Comments blanked to spaces with their newlines kept, literals kept. The length is kept too, so an offset in the
    stripped text is the same offset in the source: a mutant finds its body in the stripped text, where an apostrophe in
    a comment cannot open a character literal, and edits the source there."""

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        # A line comment holds no newline: its blanks are as many as its characters.
        return " " * len(tok) if "\n" not in tok else re.sub(r"[^\n]", " ", tok)

    return _LEXEME_RE.sub(repl, src)


@functools.lru_cache(maxsize=_SMALL)
def norm(text: str) -> str:
    """Whitespace runs collapsed to one space, and none just inside parentheses, so a re-wrap cannot turn a pin red."""
    words = text.split()  # the same whitespace as \s, and far faster on the 7 MB backend source
    if not words:
        return " " if text else ""
    t = (" " if text[0].isspace() else "") + " ".join(words) + (" " if text[-1].isspace() else "")
    return t.replace("( ", "(").replace(" )", ")")


@functools.lru_cache(maxsize=None)
def _word_re(pattern: str) -> re.Pattern:
    return re.compile(pattern)


def word_matches(pattern: str, text: str) -> list:
    """The matches of `\\b` + `pattern` in `text`, for a pattern that starts with a word character. The regex engine
    finds a leading literal far faster than a leading word boundary, so the boundary is checked on the match."""
    return [m for m in _word_re(pattern).finditer(text)
            if m.start() == 0 or not (text[m.start() - 1].isalnum() or text[m.start() - 1] == "_")]


def block_span(text: str, open_at: int) -> tuple[int, int]:
    """[start, end) of the brace-balanced block opening at `open_at`, literals skipped; (-1, -1) when there is none."""
    if open_at < 0:
        return -1, -1
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
                return open_at, i + 1
        i += 1
    return -1, -1


@functools.lru_cache(maxsize=_SMALL)
def body_span(text: str, signature: str) -> tuple[int, int]:
    """[start, end) of the body of the first definition starting with `signature`; a declaration (its parameter list
    closes on a semicolon) is passed over. (-1, -1) when there is none."""
    at = text.find(signature)
    while at >= 0:
        if signature.endswith("("):
            depth = 0
            close = -1
            for i in range(at + len(signature) - 1, len(text)):
                depth += {"(": 1, ")": -1}.get(text[i], 0)
                if depth == 0:
                    close = i
                    break
            if close >= 0 and text[close + 1:close + 200].lstrip().startswith(";"):
                at = text.find(signature, close)
                continue
        return block_span(text, text.find("{", at + len(signature) - 1))
    return -1, -1


@functools.lru_cache(maxsize=_SMALL)
def body(text: str, signature: str) -> str:
    start, end = body_span(text, signature)
    return text[start:end] if start >= 0 else ""


def ordered(text: str, *needles: str) -> bool:
    at = 0
    for needle in needles:
        at = text.find(needle, at)
        if at < 0:
            return False
        at += len(needle)
    return True


def enclosing_block_end(text: str, at: int) -> int:
    """The offset just past the `}` that closes the innermost block open at `at`, literals skipped; len(text) if none."""
    depth = 0
    i = at
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
            if depth == 0:
                return i + 1
            depth -= 1
        i += 1
    return n


def call_arguments(text: str, open_paren: int) -> tuple[int, int]:
    """[start, end) of the parenthesised argument list opening at `open_paren`, literals skipped."""
    depth = 0
    i = open_paren
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "\"'":
            j = i + 1
            while j < n and text[j] != ch:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return open_paren, i + 1
        i += 1
    return -1, -1


def check(raw: dict[str, str]) -> None:
    sycl = strip_comments(raw["sycl"])
    driver = strip_comments(raw["driver"])
    plan = strip_comments(raw["plan"])
    cache = strip_comments(raw["cache"])
    sync = body(sycl, SYNC_SIG)
    outer = body(sycl, CALLER_SIG)
    core = body(sycl, CORE_SIG)
    make_call = body(sycl, CALL_SIG)
    builder = body(plan, BUILDER_SIG)
    if not sync:
        fail("D1", "ggml_backend_sycl_synchronize has no body the pin can read")
    if not outer:
        fail("D1", "the swap driver ggml_sycl_run_expert_swap_at_boundary has no definition")
    if not core or not make_call:
        fail("D22", "the claimed body ggml_sycl_run_expert_swap_claimed or ggml_sycl_moe_swap_make_call has no "
                    "definition")
    # The driver as one text, in the order its parts run: the gated boundary, the call it builds, the claimed body.
    caller = outer + "\n" + make_call + "\n" + core

    # D1 ---------------------------------------------------------------------------------------------------------------
    if sync.count(CALLER_CALL) != 1 or not ordered(sync, REFILL_CALL, CALLER_CALL):
        fail("D1", "synchronize does not call the swap driver exactly once, after the refill")
    if sycl.count(CALLER_NAME) != 2:
        fail("D1", f"the swap driver is named {sycl.count(CALLER_NAME)} times, not at its definition and one call")

    # D2 ---------------------------------------------------------------------------------------------------------------
    if sycl.count(STEP_CALL) != 1 or caller.count(STEP_CALL) != 1:
        fail("D2", "the policy step does not run exactly once, inside the swap driver")

    # D3 ---------------------------------------------------------------------------------------------------------------
    helpers = sorted(set(HELPER_DEF.findall(sycl)))
    scopes = [("the swap driver", caller), ("the driver header", driver), ("the plan-view builder", builder)]
    for name in helpers:
        scopes.append((name, _helper_body(sycl, name)))
    for where, text in scopes:
        m = WAITS.search(text)
        if m:
            fail("D3", f"{where} waits or asks an event's status: `{m.group(0).strip()}`")

    # D4 ---------------------------------------------------------------------------------------------------------------
    # The step runs inside the claimed body, which the gated boundary calls: a lock taken in either text is held across
    # the step when its block reaches the claimed body's call, or the step itself.
    for text, mark in ((outer, CORE_CALL), (core, STEP_CALL), (make_call, "\x00")):
        mark_at = text.find(mark)
        m = BARE_LOCK.search(text)
        if m:
            fail("D4", f"the swap driver takes a lock by a bare call: `{m.group(0)}`")
        for lk in LOCK_DECL.finditer(text):
            if mark_at >= 0 and lk.start() < mark_at < enclosing_block_end(text, lk.start()):
                fail("D4", "the swap driver holds a lock across the policy step")
    step_at = caller.find(STEP_CALL)

    # D5 ---------------------------------------------------------------------------------------------------------------
    if not ordered(caller[:step_at], *GATES):
        fail("D5", "the swap driver is not gated, in order, by the switch, the multithreaded latch, the compute flag, "
                   "the graphs in flight and the claim of its instance before the step")
    claim_n = norm(body(sycl, CLAIM_SIG))
    if (claim_n.count(CLAIM_TAKE) != 1 or claim_n.count(CLAIM_GIVE_BACK) != 1 or "claimed.compare_exchange" in sycl
            or len(CLAIM_DECL_RE.findall(sycl)) != 4
            or norm(body(sycl, REPORT_AT_FREE_SIG)).count(REPORT_IF_HELD) != 1):
        fail("D5", "an instance is not claimed through the one guard that takes it with a strong compare-exchange at "
                   "acq_rel and gives a held claim back with a release store, at the boundary, the backend free's "
                   "report (which reports only a claim it holds) and the private tests' boundary and report hooks")

    # D22 --------------------------------------------------------------------------------------------------------------
    outer_n, core_n = norm(outer), norm(core)
    hook_n = norm(body(sycl, HOOK_SIG))
    core_at = sycl.find(CORE_SIG)
    if core_at < 0 or CORE_DECL not in norm(sycl[core_at:sycl.find("{", core_at) + 1]):
        fail("D22", "the claimed body does not take (device, decodes, graphs, resolver)")
    if re.search(r"\bctx\b", core):
        fail("D22", "the claimed body reads the backend context, so a test cannot run it without one")
    if (outer_n.count(CORE_CALL) != 1 or outer_n.count(CORE_OUTER_CALL) != 1
            or not ordered(outer_n, CLAIM_DECL, CLAIM_LOST, CORE_CALL)):
        fail("D22", "the gated boundary does not call the claimed body once, after the claim, with no resolver")
    if core_n.count(CORE_MAKE_CALL) != 1 or "moe_swap_driver_call call;" in core_n:
        fail("D22", "the claimed body does not take the driver's call from ggml_sycl_moe_swap_make_call")
    if core_n.count(CORE_TXN) != 1 or norm(core).count("nullptr, &result") != 0:
        fail("D22", "the claimed body does not send the named move through the resolver it was given")
    if sycl.count(CORE_CALL) != 3 or not hook_n or not ordered(hook_n, CLAIM_DECL, CLAIM_LOST, CORE_CALL):
        fail("D22", "the private tests' hook is not the one other caller of the claimed body, claiming the device's "
                    "instance before it")

    # D6 ---------------------------------------------------------------------------------------------------------------
    accessor = body(sycl, ACCESSOR_SIG)
    getenvs = [m.start() for m in word_matches(r"getenv\s*\(\s*ggml_sycl::MOE_SWAP_ENV\s*\)", sycl)]
    if len(ENV_SPELLING.findall(driver)) != 1 or '"GGML_SYCL_MOE_SWAP"' in sycl:
        fail("D6", "GGML_SYCL_MOE_SWAP is not spelled once, as the header's MOE_SWAP_ENV")
    if (len(getenvs) != 1 or not accessor or "getenv(ggml_sycl::MOE_SWAP_ENV)" not in norm(accessor)
            or not re.search(r"\bstatic\s+const\s+bool\b", accessor)):
        fail("D6", "the switch is not read once, by getenv in the accessor's static initializer")
    rows = TABLE_ROW.findall(sycl)
    loop_at = sycl.find(PARSE_LOOP)
    over = ROW_OVERWRITE.search(sycl)
    if len(rows) != 1 or loop_at < 0 or not over or over.start() < loop_at:
        fail("D6", "the settings table does not name the switch once with its row overwritten from the accessor after "
                   "the parse")

    # D7 ---------------------------------------------------------------------------------------------------------------
    warn_spans = []
    for w in word_matches(r"GGML_LOG_WARN\s*\(", sycl):
        warn_spans.append(call_arguments(sycl, w.end() - 1))
    for summary in SUMMARIES:
        calls = [m.start() for m in re.finditer(re.escape(summary), sycl)]
        if not calls:
            fail("D7", f"no `{summary[:-1]}` line is logged")
        for at in calls:
            if not any(s <= at < e for s, e in warn_spans):
                line = sycl.count("\n", 0, at) + 1
                fail("D7", f"`{summary[:-1]}` at line {line} is not an argument of GGML_LOG_WARN")
    faults = norm(body(driver, FAULTS_SIG))
    if not faults or not all(f in faults for f in FAULTS_FIELDS):
        fail("D7", "the second summary line does not name the accepted transactions, evicts and admits apart, "
                   "beside the decode graphs")
    if any(driver.count(c) != 1 for c in TXN_COUNTS):
        fail("D7", "the step does not count each direction's accepted transaction exactly once")
    if not all(f in faults for f in UNSETTLED_FIELDS) or norm(driver).count(UNSETTLED_SET) != 2:
        fail("D7", "the second summary line does not name the moves left unsettled, set at each throw")

    # D8 ---------------------------------------------------------------------------------------------------------------
    includes = INCLUDE.findall(driver)
    if '"moe-swap-policy.hpp"' not in includes:
        fail("D8", "the driver header does not include the policy header")
    for inc in includes:
        if inc != '"moe-swap-policy.hpp"' and (inc.startswith('"') or inc[1:-1] not in ALLOWED_INCLUDES):
            fail("D8", f"the driver header includes {inc}, which is not the policy header or a listed standard header")
    stateless = re.sub(r"\bstatic_(?:cast|assert)\b", "  ", driver)
    m = STATE_KEYWORDS.search(stateless)
    if m:
        fail("D8", f"`{m.group(0)}` at line {stateless.count(chr(10), 0, m.start()) + 1}: state outside the caller")
    for m in INLINE_VAR.finditer(stateless):
        head = m.group(0)
        if "(" not in head and not re.search(r"\bconst\b", head.split("*")[-1]):
            fail("D8", f"an inline variable at line {stateless.count(chr(10), 0, m.start()) + 1}: global state")
    m = SYNC_NAMES.search(driver)
    if m:
        fail("D8", f"`{m.group(0)}`: the driver header shares nothing and needs no atomic or lock")
    m = FORBIDDEN.search(driver)
    if m:
        fail("D8", f"`{m.group(0).strip()}` at line {driver.count(chr(10), 0, m.start()) + 1}: device, cache, log or "
                   "environment use in the driver header")
    code = LITERAL.sub('""', driver)
    m = HANDLER.search(code)
    if m:
        fail("D8", f"`{m.group(0)}` at line {code.count(chr(10), 0, m.start()) + 1}: an exception handler in the driver "
                   "header, which could swallow what the transaction throws")

    # D9 ---------------------------------------------------------------------------------------------------------------
    if not builder:
        fail("D9", "moe_swap_plan_view_build has no definition in moe-stream-plan.hpp")
    m = BUILDER_FORBIDDEN.search(re.sub(r"\bstatic_(?:cast|assert)\b", "  ", builder))
    if m:
        fail("D9", f"the plan-view builder keeps state, locks, logs or reads the environment: `{m.group(0).strip()}`")

    # D10 --------------------------------------------------------------------------------------------------------------
    prep = body(sycl, PREP_SIG)
    mint = norm(body(sycl, MINT_SIG))
    no_growth = norm(body(sycl, NO_GROWTH_SIG))
    if len(EVICT_MINT.findall(prep)) != 1:
        fail("D10", "the to-host branch does not mint an evict's copy without growth and a loan's with the growing mint")
    if not mint or not all(f in mint for f in MINT_FIELDS):
        fail("D10", "the host mint does not forbid host-zone growth, with its miss kept out of the error log, exactly "
                    "when grow_zone is false, in a pinned WEIGHT request")
    if not no_growth or NO_GROWTH_CALL not in no_growth or "unified_allocate_owner" in no_growth:
        fail("D10", "the no-growth mint is not one call of the host mint with grow_zone false")

    # D11 --------------------------------------------------------------------------------------------------------------
    step = norm(body(driver, STEP_SIG))
    if not step or step.count(COUNTS_CALL) != 1 or len(re.findall(r"\bcounts\(", step)) != 1:
        fail("D11", "the step does not call the counts callback once, with the driver's own decode and prompt arrays")

    # D12 --------------------------------------------------------------------------------------------------------------
    warn = norm(body(sycl, WARN_SIG))
    callback = caller[step_at:]
    if not warn or not all(f in warn for f in WARN_FIELDS):
        fail("D12", "ggml_sycl_moe_swap_warn_refusal does not log each REFUSED reason once per direction at WARN")
    if norm(callback).count(WARN_CALL) != 1 or sycl.count("ggml_sycl_moe_swap_warn_refusal(") != 2:
        fail("D12", "the transaction callback does not hand every move to ggml_sycl_moe_swap_warn_refusal, once")
    if sycl.count("refusals_warned_") != warn.count("refusals_warned_") + 2:
        fail("D12", "the logged-reason sets are touched outside their declaration and the warn helper")

    # D13 --------------------------------------------------------------------------------------------------------------
    latch = norm(body(sycl, LATCH_SIG))
    if norm(plan).count(SHARED_DEF) != 1 or plan.count(SHARED_LITERAL) != 1:
        fail("D13", "the plan header does not define MOE_EXPERT_MOVE_SHARED_STORAGE as the one \"shared storage\"")
    for name, text in (("ggml-sycl.cpp", sycl), ("the driver header", driver), ("unified-cache.cpp", cache)):
        if SHARED_LITERAL in text:
            fail("D13", f"{name} spells \"shared storage\" instead of naming MOE_EXPERT_MOVE_SHARED_STORAGE")
    if norm(body(cache, LOAN_SIG)).count(SHARED_ASSIGN) != 1:
        fail("D13", "the cache's loan does not give the shared-storage refusal by its name")
    if not latch or not all(f in latch for f in LATCH_FIELDS):
        fail("D13", "the block latch is not read from a refused evict whose reason is MOE_EXPERT_MOVE_SHARED_STORAGE")
    if norm(callback).count(LATCH_SET) != 1 or "latches_block" in norm(caller[:step_at]):
        fail("D13", "the transaction callback does not set each move's latch from ggml_sycl_moe_swap_refusal_latches_block")
    if step.count(STEP_LATCH) != 1 or step.count("d.latched[static_cast<size_t>(q.block)]") != 1:
        fail("D13", "the step latches an evict's block other than when the move says the refusal latches it")

    # D14 --------------------------------------------------------------------------------------------------------------
    report = body(sycl, REPORT_SIG)
    lead = report.find("if (!inst.driver) {")
    s0, e0 = block_span(report, report.find("{", lead)) if lead >= 0 else (-1, -1)
    no_driver = norm(report[s0:e0]) if s0 >= 0 else ""
    if not no_driver or not all(f in no_driver for f in NO_DRIVER_FIELDS):
        fail("D14", "with no driver, the report does not print the boundaries turned off and skipped in a WARN line")

    # D15 --------------------------------------------------------------------------------------------------------------
    room = norm(body(sycl, ROOM_SIG))
    if not room or ROOM_RETURN not in room or "host_zone_capacity" in room or "host_zone_used" in room:
        fail("D15", "the evict batch's room is not the WEIGHT zone's no-growth room")
    if norm(cache).count(MINT_BOUND) != 1 or norm(cache).count(MINT_CAP) != 1:
        fail("D15", "the evict mint's no-growth cap is not host_zone_no_growth_bound less what the zone holds")
    # D16 --------------------------------------------------------------------------------------------------------------
    replaced = norm(caller)
    if replaced.count(SAME_LOAD) != 1 or replaced.count(CARRY_GATE) != 1 or replaced.count(FORCED_REPORT) != 1:
        fail("D16", "a replaced table's driver is carried other than under the same model and load, or its report is "
                    "not forced when moves were dropped")
    if norm(body(sycl, REPORT_SIG)).count(REPORT_FORCE) != 1:
        fail("D16", "a forced report still waits for a boundary the last report did not cover")

    accessor_room = norm(body(cache, NO_GROWTH_ROOM_SIG))
    if not accessor_room or not all(f in accessor_room for f in NO_GROWTH_ROOM_FIELDS):
        fail("D15", "the no-growth room is not host_zone_no_growth_bound (capacity when it has none) less what is "
                    "used, never below 0")

    # D17 --------------------------------------------------------------------------------------------------------------
    calls = [m.start() for m in re.finditer(re.escape(TXN_CALL), step)]
    if len(calls) != 2:
        fail("D17", f"the step calls the transaction {len(calls)} times, not once for admits and once for evicts")
    prev = 0
    for at in calls:
        seg = step[prev:at]
        decls = list(GUARD_DECL.finditer(seg))
        if not decls:
            fail("D17", "a transaction call has no unwind guard declared before it")
        guard = decls[-1]
        before, guarded = seg[:guard.start()], seg[guard.start():]
        if UNSETTLED_SET not in guarded:
            fail("D17", f"guard `{guard.group(1)}` does not set the unsettled count")
        if not UNSETTLED_RESERVE.search(before):
            fail("D17", f"the unsettled list is not reserved past its size before guard `{guard.group(1)}`, so its "
                        "pushes can throw while a throw unwinds")
        if "moe_swap_count_reason(st.busy," in guarded and not BUSY_RESERVE.search(before):
            fail("D17", f"the busy reasons are not reserved past their size before guard `{guard.group(1)}`, which "
                        "counts one")
        for name in RECORDING_RESERVES[calls.index(at)]:
            if not _reserve_past_size(name).search(seg):
                fail("D17", f"`{name}` is not reserved past its size before the transaction call that sends its "
                            "batch, so recording what moved can throw after it moved")
        if not step[at + len(TXN_CALL):].startswith(f" {guard.group(1)}.dismiss();"):
            fail("D17", f"guard `{guard.group(1)}` is not dismissed on the line right after the transaction returns")
        prev = at + len(TXN_CALL)

    # D18 --------------------------------------------------------------------------------------------------------------
    if norm(caller[:step_at]).count(ROOM_EPOCH_READ) != 1 or "call.room_epoch" in callback:
        fail("D18", "the caller does not hand the step the WEIGHT zone's room epoch, read once before the step and "
                    "left as read by the callback")
    if norm(callback).count(BUSY_CAUSE_SET) != 1 or sycl.count("busy_cause") != 1:
        fail("D18", "the transaction callback does not give each answered move its busy cause from the outcome's "
                    "zone_room, in one place")
    if (norm(prep).count(ZONE_ROOM_MINT) != 1 or len(word_matches(r"mint_refused_bytes\s*=(?!=)", sycl)) != 2
            or len(word_matches(r"mint_refused_ask\s*=(?!=)", sycl)) != 2 or "zone_free_figures(" in norm(prep)):
        fail("D18", "the preparation does not leave the size of a refused device mint on the prepared triplet, set "
                    "where the device mint fails and nowhere else, without reading the zone's figures there")
    txn_norm = norm(body(sycl, TXN_SIG))
    if (txn_norm.count(ZONE_FREE_READ) != 1 or txn_norm.count("zone_free_figures(") != 1
            or txn_norm.count(ZONE_REFUSED_GIVE_BACK) != 1
            or not ordered(txn_norm, ZONE_ROOM_OUTCOME, ZONE_FREE_READ, ZONE_REFUSED_GIVE_BACK)
            or len(word_matches(r"zone_largest_free_at_refusal\s*=(?!=)", sycl)) != 1):
        fail("D18", "the transaction does not read the largest free block at a refused device mint onto the named "
                    "triplet's outcome, before its members go back, for the refusal the boundary's line keeps alone")
    if (norm(body(sycl, TXN_SIG)).count(ZONE_ROOM_OUTCOME) != 1
            or len(word_matches(r"zone_room_bytes\s*=(?!=)", sycl)) != 2
            or norm(body(sycl, TXN_SIG)).count(ZONE_ASK_OUTCOME) != 1
            or len(word_matches(r"zone_room_ask\s*=(?!=)", sycl)) != 2):
        fail("D18", "the transaction does not mark the named triplet's outcome exactly when a device mint was refused "
                    "transiently")

    # D19 --------------------------------------------------------------------------------------------------------------
    zone_warn = norm(body(sycl, ZONE_WARN_SIG))
    if (norm(body(sycl, KEPT_SIG)).strip() != KEPT_BODY
            or len(word_matches(r"ggml_sycl_expert_zone_refusal_kept\(", sycl)) != 3):
        fail("D19", "the refusal a boundary's line keeps is not chosen by one rule, the first and then each that asks "
                    "the zone more in all, used by both the transaction's read and the callback")
    if norm(callback).count(ZONE_REFUSED_COUNT) != 1 or len(re.findall(r"\bzone_refused\b\s*(?:=(?!=)|\+=)", caller)) != 1:
        fail("D19", "the transaction callback does not count the answers refused for device room, keeping the "
                    "largest refused triplet's ask and refused member together, in one place")
    if norm(callback).count(ZONE_WARN_AFTER_LOOP) != 1 or sycl.count("ggml_sycl_moe_swap_warn_zone_refusal(") != 2:
        fail("D19", "the zone-refusal line is not printed once per boundary, after the loop over the batch, whenever "
                    "an admit was refused for device room")
    if not zone_warn or not all(f in zone_warn for f in ZONE_WARN_FIELDS) or "GGML_LOG" in zone_warn.replace(
            ZONE_WARN_FIELDS[1], "").replace(ZONE_CENSUS_LOG, ""):
        fail("D19", "ggml_sycl_moe_swap_warn_zone_refusal does not log moe_swap_zone_refusal_line at WARN with the "
                    "WEIGHT zone's free figures read together")
    if not (zone_warn.count(ZONE_CENSUS_ONCE) == 1 and zone_warn.count(ZONE_CENSUS_READ) == 1
            and zone_warn.count(ZONE_CENSUS_LOG) == 1 and zone_warn.count(ZONE_CENSUS_BLOCK) == 1
            and ordered(zone_warn, ZONE_CENSUS_BLOCK, ZONE_CENSUS_LOG)
            and ordered(zone_warn, ZONE_WARN_FIELDS[1], ZONE_CENSUS_ONCE, ZONE_CENSUS_BLOCK, ZONE_CENSUS_READ,
                        ZONE_CENSUS_LOG)
            and len(word_matches(r"zone_census_printed\s*=(?!=)", sycl)) == 2):
        fail("D19", "the WEIGHT zone's free-block census is not logged at WARN once per swap instance, at the first "
                    "refusal for room whose free blocks it read, after the refusal line")
    visit = norm(body(cache, ZONE_VISIT_SIG))
    census_body = norm(body(cache, ZONE_CENSUS_SIG))
    largest_body = norm(body(cache, ZONE_LARGEST_SIG))
    if (not ordered(visit, *ZONE_VISIT_CASES) or census_body.count(ZONE_CENSUS_VISIT) != 1
            or largest_body.count(ZONE_LARGEST_VISIT) != 1
            or any(f in census_body or f in largest_body for f in ZONE_OWN_CHOICE)):
        fail("D19", "the census and zone_largest_free do not both read the allocators one visitor names (WEIGHT on one "
                    "chunk the KV allocator, WEIGHT and KV on chunks the chunks', any other zone its own), so the "
                    "census can count blocks other than those whose top the zone reports")

    # D20 --------------------------------------------------------------------------------------------------------------
    txn = norm(body(sycl, TXN_SIG))
    give_back = norm(body(sycl, GIVE_BACK_SIG))
    if norm(body(sycl, METER_SIG)).count(METER_MEASURE) != 1:
        fail("D20", "the room meter does not read the WEIGHT zone's room epoch right before and right after a "
                    "give-back and sum the move")
    if (not all(give_back.count(f) == 1 for f in GIVE_BACK_FIELDS)
            or len(word_matches(r"room_self_delta\s*=(?!=)", sycl)) != 4):
        fail("D20", "the prepared list is not given back measured by a guard that then writes the answer's "
                    "room_self_delta, the one place it is written besides the entry's copy")
    if txn.count(GIVE_BACK_DECL) != 1 or not ordered(txn, "prepared.reserve(n_selected);", GIVE_BACK_DECL, PREPARE_CALL):
        fail("D20", "the named move does not declare its meter and its give-back guard before the first triplet is "
                    "prepared, so an exit could give minted storage back unmeasured")
    if (txn.count(REFUSED_GIVE_BACK) != 1 or not ordered(txn, ZONE_ROOM_OUTCOME, REFUSED_GIVE_BACK, REFILL_BRANCH)
            or sycl.count(".give_back(") != 2):
        fail("D20", "a refused named triplet's minted members are not given back measured, in the named-refusal "
                    "branch")
    entry = norm(body(sycl, ENTRY_SIG))
    if (entry.count(ENTRY_SELF_DELTA) != 1 or any(entry.count(c) != 1 for c in ENTRY_COST)
            or entry.count(ENTRY_CLOCK) != 1 or word_matches(r"txn_start\b", sycl)):
        fail("D20", "the named move's entry does not carry the measured sum to its caller, or does not time the whole "
                    "call from before the transaction runs")
    if (norm(callback).count(EPOCH_AFTER_READ) != 1 or norm(callback).count(TXN_ROOM_RETURN) != 1
            or sycl.count("moe_swap_txn_room{") != 1):
        fail("D20", "the transaction callback does not read the room epoch right after the transaction returns and "
                    "return it with the transaction's own sum")
    if (any(norm(driver).count(c) != 1 for c in COUNT_COST)
            or norm(driver).count("inline void moe_swap_count_cost(") != 1):
        fail("D20", "the step does not sum each transaction's cost into the stats once, at the admit transaction "
                    "and at the evict one")
    if norm(body(driver, ROOM_STOOD_SIG)).count(ROOM_STOOD_BODY) != 1:
        fail("D20", "moe_swap_room_stood does not compare the epoch's move since the boundary's read with the "
                    "transaction's own sum")
    if (not ordered(step, ADMIT_TXN, ROOM_STOOD_USE, HELD_FOLLOWS, HOLD_SET) or step.count(HOLD_SET) != 1
            or step.count(HELD_FOLLOWS) != 1 or re.search(r"\broom_epoch\s*=\s*c\.room_epoch\b", step)):
        fail("D20", "the step does not hold a refused admit, and let a held one follow the epoch, only when the room "
                    "stood, under the epoch the transaction left")

    # D21 --------------------------------------------------------------------------------------------------------------
    zone_free = norm(body(cache, ZONE_FREE_SIG))
    given_back = [m.end() for m in ZONE_FREE_GIVES_BACK.finditer(zone_free)]
    if (len(given_back) != 4 or zone_free.count(ZONE_FREE_BUMP) != 4
            or not all(zone_free[at:].startswith(" " + ZONE_FREE_BUMP) for at in given_back)):
        fail("D21", "unified_cache::zone_free does not move the group's room epoch right after each of its four "
                    "zone_used_free calls, so a block can go back to the zone with the epoch standing")
    if not norm(body(cache, RESET_SIG)).startswith(RESET_BUMP):
        fail("D21", "arena_group_reset_allocators_locked does not move the group's room epoch before it resets the "
                    "allocators")
    if (norm(body(cache, ALLOCATORS_MADE_SIG)).count(ALLOCATORS_MADE_BODY) != 1
            or cache.count(ALLOCATORS_MADE_CALL) != 2):
        fail("D21", "arena_note_allocators_made does not move every group's room epoch, or is not called at both "
                    "places the arena's allocators are made")
    if norm(body(cache, ROOM_GREW_SIG)).count(ROOM_GREW_BODY) != 1:
        fail("D21", "arena_note_room_grew does not add one to the group's room epoch")


@functools.lru_cache(maxsize=_SMALL)
def _helper_body(sycl: str, name: str) -> str:
    for m in HELPER_DEF.finditer(sycl):
        if m.group(1) == name:
            start, end = body_span(sycl, sycl[m.start():m.end()])
            return sycl[start:end] if start >= 0 else ""
    return ""


def mutate(raw: dict[str, str], edits: list[tuple[str, str, str, str]]) -> dict[str, str]:
    """Apply (file, signature, old, new) edits. With a signature the anchor must appear exactly once inside that body;
    with an empty one, once in the whole file. The body is found in the comment-stripped text, whose offsets are the
    source's."""
    out = dict(raw)
    for key, sig, old, new in edits:
        text = out[key]
        if sig:
            start, end = body_span(strip_comments(text), sig)
            if start < 0:
                raise MutantNotApplied(f"NOT APPLIED: no body for {sig!r}")
        else:
            start, end = 0, len(text)
        n = text[start:end].count(old)
        if n != 1:
            raise MutantNotApplied(f"NOT APPLIED: anchor appears {n} times, not once: {old[:70]!r}")
        at = text.index(old, start, end)
        out[key] = text[:at] + new + text[at + len(old):]
    return out


def self_test(raw: dict[str, str]) -> int:
    failures: list[str] = []
    n_mutants = 0

    def expect_fail(name: str, pin: str, edits: list[tuple[str, str, str, str]]) -> None:
        nonlocal n_mutants
        n_mutants += 1
        try:
            mutant = mutate(raw, edits)
        except MutantNotApplied as e:
            failures.append(name)
            print(f"  self-test: {name}: {e}")
            return
        try:
            check(mutant)
        except ContractError as e:
            if f"[pin {pin}]" in str(e):
                return
            failures.append(name)
            print(f"  self-test: {name}: caught by a different pin than {pin}: {e}")
            return
        failures.append(name)
        print(f"  self-test: {name}: SURVIVED (pin {pin} did not catch it)")

    ns = "namespace ggml_sycl {\n"
    expect_fail("driver-not-called", "D1", [("sycl", SYNC_SIG, CALLER_CALL, "")])
    expect_fail("driver-before-the-refill", "D1",
                [("sycl", SYNC_SIG, CALLER_CALL, ""), ("sycl", SYNC_SIG, REFILL_CALL, CALLER_CALL + "\n" + REFILL_CALL)])
    expect_fail("driver-called-twice", "D1", [("sycl", SYNC_SIG, CALLER_CALL, CALLER_CALL + "\n" + CALLER_CALL)])
    expect_fail("driver-called-from-elsewhere", "D1",
                [("sycl", "", SYNC_SIG, "static void ggml_sycl_swap_twice(ggml_backend_t backend) {\n    "
                  + CALLER_CALL + "\n}\n\n" + SYNC_SIG)])
    expect_fail("step-run-twice", "D2", [("sycl", CORE_SIG, STEP_CALL, STEP_CALL + "d, inst.view, call, c, r, t);\n"
                                          "    (void) ggml_sycl::" + STEP_CALL)])
    expect_fail("driver-waits-on-the-queue", "D3",
                [("sycl", CALLER_SIG, "if (!ggml_sycl_moe_swap_enabled()) {",
                  "ctx->stream()->wait();\n    if (!ggml_sycl_moe_swap_enabled()) {")])
    expect_fail("driver-drains-retained-handles", "D3",
                [("sycl", CALLER_SIG, "if (!ggml_sycl_moe_swap_enabled()) {",
                  "(void) ggml_sycl::drain_retained_handles(true, 100);\n    if (!ggml_sycl_moe_swap_enabled()) {")])
    expect_fail("driver-asks-an-event", "D3",
                [("sycl", CALLER_SIG, "if (!ggml_sycl_moe_swap_enabled()) {",
                  "(void) ev.get_info<sycl::info::event::command_execution_status>();\n"
                  "    if (!ggml_sycl_moe_swap_enabled()) {")])
    expect_fail("header-sleeps", "D3", [("driver", "", ns, ns + "inline void nap() { std::this_thread::sleep_for(x); }\n")])
    expect_fail("builder-waits", "D3", [("plan", BUILDER_SIG, "out.n_blocks", "q.wait();\n    out.n_blocks")])
    expect_fail("lock-across-the-step", "D4",
                [("sycl", CALLER_SIG, "if (!ggml_sycl_moe_swap_enabled()) {",
                  "std::lock_guard held(g_tensor_inventory_mutex);\n    if (!ggml_sycl_moe_swap_enabled()) {")])
    expect_fail("lock-across-the-step-in-the-claimed-body", "D4",
                [("sycl", CORE_SIG,
                  "ggml_sycl_moe_swap_instance &                      inst  = g_sycl_moe_swap[device];",
                  "std::lock_guard held(g_tensor_inventory_mutex);\n    "
                  "ggml_sycl_moe_swap_instance &                      inst  = g_sycl_moe_swap[device];")])
    expect_fail("bare-lock", "D4",
                [("sycl", CALLER_SIG, "if (!ggml_sycl_moe_swap_enabled()) {",
                  "g_tensor_inventory_mutex.lock();\n    if (!ggml_sycl_moe_swap_enabled()) {")])
    expect_fail("switch-not-read", "D5",
                [("sycl", CALLER_SIG, "if (!ggml_sycl_moe_swap_enabled()) {", "if (false) {")])
    expect_fail("graphs-in-flight-ignored", "D5",
                [("sycl", CALLER_SIG, "ggml_sycl_graph_inflight_count()", "0")])
    expect_fail("instance-not-claimed", "D5",
                [("sycl", CLAIM_SIG, "compare_exchange_strong(", "compare_exchange_weak(")])
    expect_fail("claim-weaker-than-acq-rel", "D5",
                [("sycl", CLAIM_SIG, "std::memory_order_acq_rel", "std::memory_order_relaxed")])
    expect_fail("claim-never-released", "D5",
                [("sycl", CLAIM_SIG, "claimed.store(false, std::memory_order_release);", "")])
    expect_fail("claim-released-though-not-held", "D5", [("sycl", CLAIM_SIG, "if (held) {", "if (true) {")])
    expect_fail("claim-taken-by-hand", "D5",
                [("sycl", CALLER_SIG, "if (!claim.held) {",
                  "bool expected = false;\n    if (!inst.claimed.compare_exchange_strong(expected, true)) {")])
    expect_fail("claim-deleted", "D5",
                [("sycl", CALLER_SIG,
                  "    const ggml_sycl_moe_swap_claim claim{ inst.claimed };\n"
                  "    if (!claim.held) {\n"
                  "        inst.skipped.fetch_add(1, std::memory_order_relaxed);\n"
                  "        return;\n"
                  "    }\n", "")])
    expect_fail("claim-lost-ignored", "D22", [("sycl", CALLER_SIG, "if (!claim.held) {", "if (false) {")])
    expect_fail("report-at-free-unclaimed", "D5",
                [("sycl", REPORT_AT_FREE_SIG, "if (claim.held) {", "if (true) {")])
    expect_fail("claimed-body-reads-the-context", "D22",
                [("sycl", CORE_SIG, "const int                             nb    = d.policy.n_blocks;",
                  "(void) ctx->device;\n    const int                             nb    = d.policy.n_blocks;")])
    expect_fail("claimed-body-ignores-the-resolver", "D22",
                [("sycl", CORE_SIG, "request, resolver, &result", "request, nullptr, &result")])
    expect_fail("claimed-body-builds-its-own-call", "D22",
                [("sycl", CORE_SIG, CORE_MAKE_CALL, CORE_MAKE_CALL + "\n    ggml_sycl::moe_swap_driver_call call;")])
    expect_fail("boundary-hands-a-resolver", "D22",
                [("sycl", CALLER_SIG, "/*resolver=*/nullptr", "/*resolver=*/&fixture")])
    expect_fail("hook-does-not-claim", "D5",
                [("sycl", HOOK_SIG, "const ::ggml_sycl_moe_swap_claim claim{ inst.claimed };", "")])
    call = "    ::ggml_sycl_run_expert_swap_claimed(device, decodes, graphs, resolver);\n"
    expect_fail("hook-claims-after-the-call", "D22",
                [("sycl", HOOK_SIG, call, ""),
                 ("sycl", HOOK_SIG, "    if (!claim.held) {", call + "    if (!claim.held) {")])
    expect_fail("env-spelled-in-the-backend", "D6",
                [("sycl", "", "ggml_sycl::MOE_SWAP_ENV)", "\"GGML_SYCL_MOE_SWAP\")")])
    expect_fail("env-read-at-every-boundary", "D6",
                [("sycl", ACCESSOR_SIG, "static const bool", "const bool")])
    expect_fail("env-row-not-overwritten", "D6",
                [("sycl", "", "g_ggml_sycl_moe_swap = ggml_sycl_moe_swap_enabled() ? 1 : 0;", "")])
    expect_fail("summary-at-info", "D7",
                [("sycl", "", "GGML_LOG_WARN(\"%s\\n\", ggml_sycl::moe_swap_driver_summary_moves(",
                  "GGML_LOG_INFO(\"%s\\n\", ggml_sycl::moe_swap_driver_summary_moves(")])
    expect_fail("summary-drops-the-admits", "D7",
                [("driver", FAULTS_SIG, "(ull) st.transactions_admit, (ull) st.decode_graphs,", "(ull) st.decode_graphs,")])
    expect_fail("step-forgets-the-evict-transaction", "D7", [("driver", "", "++st.transactions_evict;", "")])
    expect_fail("unsettled-not-reported", "D7",
                [("driver", FAULTS_SIG, "(ull) st.settled_moved, (ull) st.unsettled_moves)",
                  "(ull) st.settled_moved, 0ull)")])
    expect_fail("admit-throw-leaves-the-count", "D7",
                [("driver", STEP_SIG, "d.pending.resize(kept);\n            st.unsettled_moves = d.unsettled.size();",
                  "d.pending.resize(kept);")])
    expect_fail("evict-throw-leaves-the-count", "D7",
                [("driver", STEP_SIG, "d.unsettled.push_back(u);\n        }\n        st.unsettled_moves = "
                  "d.unsettled.size();", "d.unsettled.push_back(u);\n        }")])
    expect_fail("evict-throw-caught-in-the-header", "D8",
                [("driver", STEP_SIG, "txn(d.batch);\n    evict_throw.dismiss();",
                  "try {\n        txn(d.batch);\n    } catch (...) {\n        throw;\n    }\n"
                  "    evict_throw.dismiss();")])
    expect_fail("admit-guard-dismissed-before-the-call", "D17",
                [("driver", STEP_SIG, "txn(d.batch);\n        admit_throw.dismiss();",
                  "admit_throw.dismiss();\n        txn(d.batch);")])
    expect_fail("evict-guard-never-dismissed", "D17", [("driver", STEP_SIG, "evict_throw.dismiss();", "")])
    expect_fail("evict-guard-pushes-unreserved", "D17",
                [("driver", STEP_SIG, "d.unsettled.reserve(d.unsettled.size() + keep);", "")])
    expect_fail("admit-guard-counts-unreserved", "D17",
                [("driver", STEP_SIG, "st.busy.reserve(st.busy.size() + d.batch.size());", "")])
    expect_fail("admit-batch-records-refusals-unreserved", "D17",
                [("driver", STEP_SIG, "st.refused.reserve(st.refused.size() + d.batch.size());", "")])
    expect_fail("admit-batch-records-drops-unreserved", "D17",
                [("driver", STEP_SIG, "st.dropped.reserve(st.dropped.size() + d.batch.size());", "")])
    expect_fail("evict-batch-pushes-pending-unreserved", "D17",
                [("driver", STEP_SIG, "d.pending.reserve(d.pending.size() + keep);", "")])
    expect_fail("evict-batch-records-busy-unreserved", "D17",
                [("driver", STEP_SIG, "st.busy.reserve(st.busy.size() + keep);", "")])
    expect_fail("evict-batch-records-refusals-unreserved", "D17",
                [("driver", STEP_SIG, "st.refused.reserve(st.refused.size() + keep);", "")])
    expect_fail("evict-batch-reserves-a-bare-count", "D17",
                [("driver", STEP_SIG, "d.pending.reserve(d.pending.size() + keep);", "d.pending.reserve(keep);")])
    expect_fail("evict-guard-reserves-a-bare-count", "D17",
                [("driver", STEP_SIG, "d.unsettled.reserve(d.unsettled.size() + keep);", "d.unsettled.reserve(keep);")])
    expect_fail("evict-sent-without-a-guard", "D17",
                [("driver", STEP_SIG, "moe_swap_detail::moe_swap_unwind_guard evict_throw(",
                  "auto evict_throw = (")])
    expect_fail("carry-ignores-the-load", "D16",
                [("sycl", CORE_SIG, "snap->model_id == inst.model_id && snap->load_txn_id == inst.load_txn_id",
                  "snap->model_id == inst.model_id")])
    expect_fail("carry-without-the-gate", "D16",
                [("sycl", CORE_SIG, "if (!same_load || !ggml_sycl::moe_swap_driver_carry(",
                  "if (!ggml_sycl::moe_swap_driver_carry(")])
    expect_fail("drop-report-not-forced", "D16", [("sycl", CORE_SIG, "/*force=*/dropped != 0", "/*force=*/false")])
    expect_fail("report-ignores-force", "D16", [("sycl", REPORT_SIG, "if (!force && inst.driver", "if (inst.driver")])
    expect_fail("header-includes-the-cache", "D8",
                [("driver", "", "#include \"moe-swap-policy.hpp\"",
                  "#include \"moe-swap-policy.hpp\"\n#include \"unified-cache.hpp\"")])
    expect_fail("header-static-counter", "D8",
                [("driver", "", ns, ns + "inline int next_id() {\n    static int n = 0;\n    return ++n;\n}\n")])
    expect_fail("header-inline-global", "D8", [("driver", "", ns, ns + "inline int g_swaps = 0;\n")])
    expect_fail("header-atomic", "D8", [("driver", "", ns, ns + "inline int bump(std::atomic<int> & a) { return ++a; }\n")])
    expect_fail("header-logs", "D8", [("driver", "", ns, ns + "inline void say() { GGML_LOG_WARN(\"x\"); }\n")])
    expect_fail("header-reads-the-environment", "D8",
                [("driver", "", ns, ns + "inline bool on() { return getenv(\"X\") != nullptr; }\n")])
    expect_fail("builder-static-scratch", "D9",
                [("plan", BUILDER_SIG, "out.n_blocks", "static std::vector<int> scratch;\n    out.n_blocks")])
    expect_fail("builder-logs", "D9",
                [("plan", BUILDER_SIG, "out.n_blocks", "GGML_LOG_WARN(\"view\\n\");\n    out.n_blocks")])
    expect_fail("evict-mints-with-growth", "D10",
                [("sycl", PREP_SIG, "ggml_sycl_host_expert_alloc_no_growth(device, target.logical_bytes)",
                  "ggml_sycl_host_expert_alloc(device, target.logical_bytes)")])
    expect_fail("no-growth-mint-grows", "D10",
                [("sycl", NO_GROWTH_SIG, "/*grow_zone=*/false", "/*grow_zone=*/true")])
    expect_fail("mint-ignores-the-flag", "D10",
                [("sycl", MINT_SIG, "forbid_host_zone_growth = !grow_zone;", "forbid_host_zone_growth = false;")])
    expect_fail("no-growth-miss-logged", "D10", [("sycl", MINT_SIG, "req.suppress_failure_log", "(void) req.size")])
    expect_fail("no-growth-mints-its-own", "D10",
                [("sycl", NO_GROWTH_SIG, "return ggml_sycl_host_expert_alloc(",
                  "(void) ggml_sycl::unified_allocate_owner(ggml_sycl::alloc_request{});\n"
                  "    return ggml_sycl_host_expert_alloc(")])
    expect_fail("counts-into-the-wrong-arrays", "D11",
                [("driver", STEP_SIG, "counts(d.decode.data(), d.prompt.data());",
                  "counts(d.prompt.data(), d.decode.data());")])
    expect_fail("counts-read-twice", "D11",
                [("driver", STEP_SIG, "counts(d.decode.data(), d.prompt.data());",
                  "counts(d.decode.data(), d.prompt.data());\n        counts(d.decode.data(), d.prompt.data());")])
    expect_fail("refusal-logged-every-time", "D12",
                [("sycl", WARN_SIG, "if (!warned.insert(reason).second) {\n        return;\n    }",
                  "(void) warned.insert(reason);")])
    expect_fail("refusal-logged-for-busy", "D12",
                [("sycl", WARN_SIG, "if (m.outcome != ggml_sycl::MOE_SWAP_MOVE_REFUSED) {", "if (false) {")])
    expect_fail("refusal-at-info", "D12",
                [("sycl", WARN_SIG, "GGML_LOG_WARN(", "GGML_LOG_INFO(")])
    expect_fail("refusal-not-logged", "D12",
                [("sycl", CORE_SIG, "ggml_sycl_moe_swap_warn_refusal(inst, device, m);", "")])
    expect_fail("warned-set-cleared-elsewhere", "D12",
                [("sycl", REPORT_SIG, "ggml_sycl_moe_swap_fold(inst);",
                  "inst.refusals_warned_evict.clear();\n    ggml_sycl_moe_swap_fold(inst);")])
    expect_fail("cache-spells-the-refusal", "D13",
                [("cache", LOAN_SIG, "refusal = MOE_EXPERT_MOVE_SHARED_STORAGE;", 'refusal = "shared storage";')])
    expect_fail("latch-compares-a-literal", "D13",
                [("sycl", LATCH_SIG, "ggml_sycl::MOE_EXPERT_MOVE_SHARED_STORAGE", '"shared storage"')])
    expect_fail("latch-on-any-refused-move", "D13",
                [("sycl", LATCH_SIG, "m.kind == ggml_sycl::MOE_SWAP_MOVE_EVICT && ", "")])
    expect_fail("latch-not-set", "D13",
                [("sycl", CORE_SIG, "m.latches_block = ggml_sycl_moe_swap_refusal_latches_block(m);", "")])
    expect_fail("step-latches-every-refusal", "D13", [("driver", STEP_SIG, "if (m.latches_block) {", "if (true) {")])
    expect_fail("no-driver-line-drops-skipped", "D14",
                [("sycl", REPORT_SIG, "if (off != 0 || skipped != 0) {", "if (off != 0) {")])
    expect_fail("no-driver-line-at-debug", "D14", [("sycl", REPORT_SIG, "GGML_LOG_WARN(\n", "GGML_LOG_DEBUG(\n")])
    expect_fail("room-is-capacity-less-used", "D15",
                [("sycl", ROOM_SIG, "return cache->host_zone_no_growth_room(ggml_sycl::host_zone_id::WEIGHT);",
                  "return cache->host_zone_capacity(ggml_sycl::host_zone_id::WEIGHT);")])
    expect_fail("mint-caps-at-reserved", "D15",
                [("cache", "", "ucache->host_zone_used(zone) + alloc_size > bound",
                  "ucache->host_zone_used(zone) + alloc_size > ucache->host_zone_reserved_capacity(zone)")])
    expect_fail("no-growth-room-negative", "D15",
                [("cache", NO_GROWTH_ROOM_SIG, "return limit > used ? limit - used : 0;", "return limit - used;")])
    expect_fail("no-growth-room-ignores-bound", "D15",
                [("cache", NO_GROWTH_ROOM_SIG, ROOM_BOUND, "host_zone_reserved_capacity(zone)")])
    expect_fail("no-growth-room-unbounded-zone-is-zero", "D15",
                [("cache", NO_GROWTH_ROOM_SIG, "bound > 0 ? bound : capacity", "bound")])
    expect_fail("room-epoch-not-read", "D18",
                [("sycl", CALL_SIG,
                  "call.room_epoch        = cache ? cache->zone_room_epoch(ggml_sycl::vram_zone_id::WEIGHT) : 0;",
                  "call.room_epoch        = 0;")])
    expect_fail("room-epoch-read-in-the-callback", "D18",
                [("sycl", CORE_SIG, "ggml_sycl_expert_move_request request;",
                  "call.room_epoch = 0;\n            ggml_sycl_expert_move_request request;")])
    expect_fail("busy-cause-not-mapped", "D18",
                [("sycl", CORE_SIG, "o.zone_room_bytes != 0 ? ggml_sycl::MOE_SWAP_BUSY_CAUSE_ZONE_ROOM",
                  "false ? ggml_sycl::MOE_SWAP_BUSY_CAUSE_ZONE_ROOM")])
    expect_fail("refused-mint-not-recorded", "D18",
                [("sycl", PREP_SIG, "out.mint_refused_bytes = member_bytes;", "")])
    expect_fail("refused-mint-records-its-own-staged-size", "D18",
                [("sycl", PREP_SIG, "out.mint_refused_bytes = member_bytes;", "out.mint_refused_bytes = 1;")])
    expect_fail("largest-free-at-refusal-not-read", "D18",
                [("sycl", TXN_SIG, "(void) cache->zone_free_figures(", "(void) sizeof(")])
    expect_fail("largest-free-at-refusal-read-at-every-refusal", "D18",
                [("sycl", TXN_SIG, "zone_room_bytes != 0 &&\n                ggml_sycl_expert_zone_refusal_kept(",
                  "zone_room_bytes != 0 &&\n                true | ggml_sycl_expert_zone_refusal_kept(")])
    expect_fail("largest-free-at-refusal-read-in-the-preparation-too", "D18",
                [("sycl", PREP_SIG, "out.mint_refused_ask   = ggml_sycl::expert_triplet_zone_ask_of(staged);",
                  "out.mint_refused_ask   = ggml_sycl::expert_triplet_zone_ask_of(staged);\n                "
                  "{ size_t c = 0, u = 0, a = 0, l = 0; (void) cache.zone_free_figures("
                  "ggml_sycl::vram_zone_id::WEIGHT, c, u, a, l); }")])
    expect_fail("largest-free-at-refusal-read-after-the-give-back", "D18",
                [("sycl", TXN_SIG, "room_meter.give_back([&triplet] {\n                for ("
                  "ggml_sycl::expert_move_member & m : triplet.members) {\n                    "
                  "m.owner = ggml_sycl::mem_handle{};\n                }\n            });\n", ""),
                 ("sycl", TXN_SIG, "if ((*named_moves)[triplet.named].zone_room_bytes != 0 &&",
                  "room_meter.give_back([&triplet] { for (ggml_sycl::expert_move_member & m : triplet.members) { "
                  "m.owner = ggml_sycl::mem_handle{}; } });\n            "
                  "if ((*named_moves)[triplet.named].zone_room_bytes != 0 &&")])
    expect_fail("largest-free-at-refusal-read-for-a-refusal-of-no-room", "D18",
                [("sycl", TXN_SIG, "if ((*named_moves)[triplet.named].zone_room_bytes != 0 &&", "if (")])
    expect_fail("host-mint-recorded-as-a-device-one", "D18",
                [("sycl", PREP_SIG, "*transient = out.kind == ggml_sycl_expert_move_kind::EVICT;",
                  "out.mint_refused_bytes = target.logical_bytes;\n                "
                  "*transient = out.kind == ggml_sycl_expert_move_kind::EVICT;")])
    expect_fail("zone-room-for-any-transient-refusal", "D18",
                [("sycl", TXN_SIG, "zone_room_bytes = transient ? triplet.mint_refused_bytes : 0;",
                  "zone_room_bytes = triplet.mint_refused_bytes;")])
    expect_fail("zone-room-not-on-the-outcome", "D18",
                [("sycl", TXN_SIG, "(*named_moves)[triplet.named].zone_room_bytes = transient ? "
                  "triplet.mint_refused_bytes : 0;", "")])
    after_loop = ("ggml_sycl_moe_swap_warn_refusal(inst, device, m);\n            }\n")
    expect_fail("zone-refusal-line-per-admit", "D19",
                [("sycl", CORE_SIG, after_loop,
                  "ggml_sycl_moe_swap_warn_refusal(inst, device, m);\n                if (zone_refused != 0) {\n"
                  "                    ggml_sycl_moe_swap_warn_zone_refusal(inst, cache, device, zone_refused, "
                  "request.admit.size(), zone_refused_member, zone_refused_ask, zone_refused_free);\n"
                  "                }\n            }\n")])
    expect_fail("zone-refusal-line-only-when-accepted", "D19",
                [("sycl", CORE_SIG, "if (zone_refused != 0) {", "if (zone_refused != 0 && result.accepted) {")])
    expect_fail("zone-refusal-line-dropped", "D19",
                [("sycl", CORE_SIG, "if (zone_refused != 0) {", "if (false) {")])
    expect_fail("zone-refusals-not-counted", "D19", [("sycl", CORE_SIG, "++zone_refused;", "")])
    expect_fail("zone-refusal-line-at-info", "D19", [("sycl", ZONE_WARN_SIG, "GGML_LOG_WARN(\"%s\\n\", line.c_str());",
                                                      "GGML_LOG_INFO(\"%s\\n\", line.c_str());")])
    expect_fail("zone-refusal-figures-of-another-zone", "D19",
                [("sycl", ZONE_WARN_SIG, "zone_free_figures(ggml_sycl::vram_zone_id::WEIGHT",
                  "zone_free_figures(ggml_sycl::vram_zone_id::KV")])
    expect_fail("zone-refusal-figures-not-read", "D19", [("sycl", ZONE_WARN_SIG, "(void) cache->zone_free_figures(",
                                                         "(void) sizeof(")])
    expect_fail("zone-census-at-every-refusal", "D19",
                [("sycl", ZONE_WARN_SIG, "    inst.zone_census_printed = true;\n", "")])
    expect_fail("zone-census-latched-before-its-read", "D19",
                [("sycl", ZONE_WARN_SIG, "    inst.zone_census_printed = true;\n", ""),
                 ("sycl", ZONE_WARN_SIG, "    std::vector<size_t> free_sizes;\n",
                  "    inst.zone_census_printed = true;\n    std::vector<size_t> free_sizes;\n")])
    expect_fail("zone-visitor-drops-the-single-chunk-kv-case", "D19",
                [("cache", ZONE_VISIT_SIG, "if (arena_chunks_.size() == 1 && zone == vram_zone_id::WEIGHT) {",
                  "if (false) {")])
    expect_fail("zone-visitor-drops-kvs-tail-allocator", "D19",
                [("cache", ZONE_VISIT_SIG, "if (zone == vram_zone_id::KV && z.allocator) {", "if (false) {")])
    expect_fail("census-reads-another-zones-allocators", "D19",
                [("cache", ZONE_CENSUS_SIG, "arena_visit_zone_allocators(zone,",
                  "arena_visit_zone_allocators(vram_zone_id::KV,")])
    expect_fail("census-chooses-its-own-allocator", "D19",
                [("cache", ZONE_CENSUS_SIG, "(void) arena_visit_zone_allocators(zone, [&sizes](",
                  "(void) arena_zones_; (void) arena_visit_zone_allocators(zone, [&sizes](")])
    expect_fail("largest-free-chooses-its-own-allocator", "D19",
                [("cache", ZONE_LARGEST_SIG, "return served ? best : zone_available(zone);",
                  "return arena_zones_[static_cast<int>(zone)].allocator ? "
                  "arena_zones_[static_cast<int>(zone)].allocator->largest_free_block() : zone_available(zone);")])
    expect_fail("refusal-kept-on-an-equal-ask", "D19",
                [("sycl", KEPT_SIG, "ask.total > kept.total", "ask.total >= kept.total")])
    expect_fail("refusal-kept-by-another-rule-in-the-callback", "D19",
                [("sycl", CORE_SIG, "ggml_sycl_expert_zone_refusal_kept(zone_refused_member != 0, zone_refused_ask,\n"
                  "                                                               o.zone_room_ask)",
                  "zone_refused_member == 0 || o.zone_room_ask.total >= zone_refused_ask.total")])
    expect_fail("zone-census-at-info", "D19",
                [("sycl", ZONE_WARN_SIG, "GGML_LOG_WARN(\"%s\\n\", census.c_str());",
                  "GGML_LOG_INFO(\"%s\\n\", census.c_str());")])
    expect_fail("zone-census-of-another-zone", "D19",
                [("sycl", ZONE_WARN_SIG, "zone_free_block_sizes(ggml_sycl::vram_zone_id::WEIGHT",
                  "zone_free_block_sizes(ggml_sycl::vram_zone_id::KV")])
    expect_fail("refused-ask-not-recorded", "D18",
                [("sycl", PREP_SIG, "out.mint_refused_ask   = ggml_sycl::expert_triplet_zone_ask_of(staged);", "")])
    expect_fail("zone-ask-not-on-the-outcome", "D18",
                [("sycl", TXN_SIG, "transient ? triplet.mint_refused_ask : ggml_sycl::expert_triplet_zone_ask{};",
                  "ggml_sycl::expert_triplet_zone_ask{};")])
    expect_fail("refused-triplet-ask-not-kept", "D19",
                [("sycl", CORE_SIG, "zone_refused_ask    = o.zone_room_ask;", "")])
    expect_fail("refusal-line-prints-three-of-the-member", "D19",
                [("sycl", ZONE_WARN_SIG, "ask.slot, ask.total,", "ask.slot, 3 * member,")])
    expect_fail("census-counts-by-slot", "D19",
                [("sycl", ZONE_WARN_SIG, "free_sizes, member,", "free_sizes, ask.slot,")])
    expect_fail("census-counts-by-the-unrounded-member", "D19",
                [("sycl", ZONE_WARN_SIG, "free_sizes, member, member_block,", "free_sizes, member, member,")])
    expect_fail("census-block-not-the-allocators-rounding", "D19",
                [("sycl", ZONE_WARN_SIG, "ggml_sycl::tlsf_allocator::round_request(member, "
                  "ggml_sycl::tlsf_allocator::block_grain);", "member;")])
    expect_fail("largest-free-at-refusal-not-kept", "D19",
                [("sycl", CORE_SIG, "zone_refused_free   = o.zone_largest_free_at_refusal;", "")])
    expect_fail("largest-free-at-refusal-of-another-refusal", "D19",
                [("sycl", CORE_SIG,
                  "zone_refused_free   = o.zone_largest_free_at_refusal;\n                        }",
                  "}\n                        zone_refused_free = std::max(zone_refused_free, "
                  "o.zone_largest_free_at_refusal);")])
    expect_fail("self-delta-dropped-from-the-comparison", "D20",
                [("driver", ROOM_STOOD_SIG, "room.epoch_after - before == room.self_delta", "room.epoch_after == before")])
    expect_fail("refused-admit-held-though-room-arrived", "D20", [("driver", STEP_SIG, " && room_stood;", ";")])
    expect_fail("refused-admit-held-under-the-epoch-read-before", "D20",
                [("driver", STEP_SIG, "p.room_epoch  = room.epoch_after;", "p.room_epoch  = c.room_epoch;")])
    expect_fail("held-admit-does-not-follow-the-epoch", "D20",
                [("driver", STEP_SIG, "room_stood ? room.epoch_after : p.room_epoch;", "p.room_epoch;")])
    expect_fail("meter-does-not-sum-the-move", "D20",
                [("sycl", METER_SIG, "moved += cache->zone_room_epoch(ggml_sycl::vram_zone_id::WEIGHT) - before;", "")])
    expect_fail("meter-off-for-the-named-move", "D20",
                [("sycl", TXN_SIG, "room_meter{ named ? cache : nullptr };", "room_meter{ nullptr };")])
    expect_fail("refused-members-given-back-unmeasured", "D20",
                [("sycl", TXN_SIG, "room_meter.give_back([&triplet] {", "std::invoke([&triplet] {")])
    expect_fail("prepared-list-given-back-unmeasured", "D20",
                [("sycl", GIVE_BACK_SIG, "meter.give_back([this] { prepared.clear(); });", "prepared.clear();")])
    guard_decl = ("    ggml_sycl_expert_prepared_give_back give_back{ prepared, room_meter, named ? out : nullptr,\n"
                  "                                                   meta_us,  copy_us,    remap_us };\n")
    after_preparation = "    if (named && prepared.empty()) {\n"
    expect_fail("give-back-guard-declared-after-the-preparation", "D20",
                [("sycl", TXN_SIG, guard_decl, ""),
                 ("sycl", TXN_SIG, after_preparation, guard_decl + after_preparation)])
    expect_fail("self-delta-not-written-to-the-answer", "D20",
                [("sycl", GIVE_BACK_SIG, "answer->room_self_delta = meter.moved;", "(void) answer;")])
    expect_fail("self-delta-not-carried-by-the-entry", "D20",
                [("sycl", ENTRY_SIG, "r.room_self_delta = answer.room_self_delta;", "")])
    expect_fail("epoch-after-not-read", "D20",
                [("sycl", CORE_SIG,
                  "const uint64_t epoch_after = cache ? cache->zone_room_epoch(ggml_sycl::vram_zone_id::WEIGHT) : 0;",
                  "const uint64_t epoch_after = 0;")])
    expect_fail("self-delta-not-returned", "D20",
                [("sycl", CORE_SIG, "{ epoch_after, result.room_self_delta, cost }", "{ epoch_after, 0, cost }")])
    for field, wrong in (("copy", "meta"), ("remap", "copy"), ("meta", "remap")):
        pad = " " * (len("remap") - len(field))
        expect_fail(f"{field}-cost-from-the-wrong-part", "D20",
                    [("sycl", CORE_SIG, f"cost.{field}_us {pad}= result.{field}_us;",
                      f"cost.{field}_us {pad}= result.{wrong}_us;")])
    expect_fail("cost-built-positionally", "D20",
                [("sycl", CORE_SIG, "cost.copy_us  = result.copy_us;\n", ""),
                 ("sycl", CORE_SIG, "ggml_sycl::moe_swap_txn_cost cost;", "ggml_sycl::moe_swap_txn_cost cost{ 0, 0, "
                  "result.meta_us, 0 };")])
    expect_fail("admit-cost-not-counted", "D20", [("driver", "", "        " + COUNT_COST[0] + "\n", "")])
    expect_fail("evict-cost-not-counted", "D20",
                [("driver", "", "    " + COUNT_COST[1] + "\n    for (size_t i = 0; i < keep",
                  "    for (size_t i = 0; i < keep")])
    expect_fail("cost-not-returned", "D20",
                [("sycl", CORE_SIG, "{ epoch_after, result.room_self_delta, cost }",
                  "{ epoch_after, result.room_self_delta, {} }")])
    for field, anchor in (("meta", "r.meta_us         = answer.meta_us;"),
                          ("copy", "r.copy_us         = answer.copy_us;"),
                          ("remap", "r.remap_us        = answer.remap_us;"),
                          ("total", "r.total_us        = ggml_sycl::expert_move_us_since(call_start);")):
        expect_fail(f"{field}-cost-not-carried-by-the-entry", "D20", [("sycl", ENTRY_SIG, anchor, "")])
    clock = "    const auto                   call_start = std::chrono::steady_clock::now();\n"
    expect_fail("entry-clock-after-the-run", "D20",
                [("sycl", ENTRY_SIG, clock, ""),
                 ("sycl", ENTRY_SIG, "    // Each outcome goes back to its own list",
                  clock + "    // Each outcome goes back to its own list")])
    expect_fail("entry-total-from-the-body", "D20",
                [("sycl", ENTRY_SIG, "r.total_us        = ggml_sycl::expert_move_us_since(call_start);",
                  "r.total_us        = answer.meta_us + answer.copy_us + answer.remap_us;")])
    used = "arena_zones_[static_cast<int>(record_zone)].used);\n"
    bump = "arena_note_room_grew(arena_allocator_group_index(zone));\n"
    for name, call, indent in (("kv-zone-arm", "zone_used_free(*z.allocator, zone_offset, ", " " * 16),
                               ("chunk-arm", "zone_used_free(*target->allocator, chunk_off, ", " " * 8),
                               ("shared-allocator-arm", "zone_used_free(*alloc, zone_offset, ", " " * 8),
                               ("own-allocator-arm", "zone_used_free(*alloc, zone_offset, ", " " * 4)):
        expect_fail(f"zone-free-{name}-does-not-move-the-epoch", "D21",
                    [("cache", ZONE_FREE_SIG, call + used + indent + bump, call + used)])
    expect_fail("allocator-reset-does-not-move-the-epoch", "D21",
                [("cache", RESET_SIG, "arena_note_room_grew(group);", "")])
    expect_fail("allocators-made-moves-no-epoch", "D21",
                [("cache", ALLOCATORS_MADE_SIG, "arena_note_room_grew(group);", "")])
    expect_fail("allocators-made-not-noted", "D21",
                [("cache", "", "\n    arena_note_allocators_made();\n", "\n")])
    expect_fail("room-grew-moves-nothing", "D21",
                [("cache", ROOM_GREW_SIG, "room_epoch.fetch_add(1, std::memory_order_release);",
                  "room_epoch.load(std::memory_order_acquire);")])

    # The signature is compared up to its opening brace, wherever that falls: a comment inside the parameter list is
    # blanked, not removed, so it lengthens the text between the name and the brace and must not fail the pin.
    graphs_param = "uint64_t                               graphs,\n"
    expect_fail("claimed-body-drops-a-parameter", "D22",
                [("sycl", "", "                                              " + graphs_param, "")])
    expect_fail("claimed-body-renames-the-resolver", "D22",
                [("sycl", "", "* resolver) {\n    ggml_sycl_moe_swap_instance &",
                  "* res) {\n    ggml_sycl_moe_swap_instance &")])
    try:
        check(mutate(raw, [("sycl", "", graphs_param, graphs_param + "// a long note on the boundary's decode count, "
                            "so that the parameter list is longer than any fixed window of the signature\n"
                            + " " * 46)]))
        print("  self-test: a comment inside the parameter list passes")
    except ContractError as e:
        print(f"  self-test: a comment inside the parameter list FAILS: {e}")
        failures.append("comment-in-parameter-list")

    try:
        mutate(raw, [("sycl", "", "this anchor is in no source", "x")])
        failures.append("stale-anchor-not-reported")
        print("  self-test: a stale mutant anchor is NOT reported")
    except MutantNotApplied:
        print("  self-test: a stale mutant anchor is reported")

    try:
        check(raw)
        print("  self-test: unmodified sources pass")
    except ContractError as e:
        print(f"  self-test: unmodified sources FAIL: {e}")
        failures.append("unmodified")

    if failures:
        print(f"SELF-TEST FAIL: {', '.join(failures)}")
        return 1
    print(f"SELF-TEST PASS: {n_mutants} mutants caught by their pins, unmodified sources pass")
    return 0


def main(argv: list[str]) -> int:
    missing = [str(p) for p in FILES.values() if not p.exists()]
    if missing:
        print(f"FAIL: missing source(s): {', '.join(missing)}")
        return 1
    raw = {k: p.read_text() for k, p in FILES.items()}
    if "--self-test" in argv:
        return self_test(raw)
    try:
        check(raw)
    except ContractError as e:
        print(e)
        return 1
    print("PASS: the swap driver runs once per synchronize boundary after the refill, gated, without a wait or a lock "
          "across its policy, its evict copies never grow the host zone, its refusals, latch and room are named, and "
          "its hold reads the zone's room epoch and a typed refusal, less what the transaction's own give-backs moved "
          "the epoch by, and every site that gives the zone room moves that epoch")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
