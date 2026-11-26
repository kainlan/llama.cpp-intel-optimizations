#!/usr/bin/env python3
"""Source contract: the named expert move, its ADMIT and EVICT directions (ggml-sycl.cpp, unified-cache.cpp,
weight-row-padding.hpp, common.hpp, moe-decode-hostpath.hpp).

Host-only: reads sources, builds nothing, loads no model and touches no device. The behaviour is the private-fixture
device cases in sycl-moe-expert-loan (lead-run); this gate pins what an edit could undo while those still pass, or what
they cannot see.

A boundary driver names expert triplets to evict and to admit: each moves on its own, an evict from the device to
pinned host storage and an admit from its pinned host copy to the device, and all that move are published in one plan.
The entry ggml_sycl_expert_move_transaction_run runs the plan transaction's one body (ggml_sycl_expert_txn_run) in the
NAMED direction, the evicts first; the cache copies each admitted triplet in unified_cache::admit_host_triplet, the
refill's body (copy_host_triplet_to_device) with the admit's source rule, and each evicted one in
unified_cache::loan_expert_triplet, the loan's body. What this gate pins:

  A0  the bodies it reads exist, so no pin below passes on an empty string;
  A1  nothing on a named move's path waits or asks an event's status: no .wait(, wait_and_throw(, get_info<,
      event_complete(, command_execution_status, ext_oneapi_get_state, finalize_retired_entries_locked( or
      drain_retained_handles( in the entry, the transaction body, the triplet preparation, the named selection, the
      named stop, the cache's shared body, its admit wrapper, the staged-size helper, or what an evict runs besides:
      the loan's cache body and the view drop;
  A2  the storage is minted only in the triplet preparation, which the transaction runs before it takes the inventory
      lock and which never names it; the cache is asked to admit under that lock, and never takes it, so no
      unified-cache allocation runs under it;
  A3  a named move is refused BUSY "loans outstanding" while the device's lent count is not zero, before the
      live-update lease, and again under the inventory lock while the ledger holds a lent triplet;
  A4  an expert's size is its tensor's, never a cache entry's, in every direction: the resolver reads the tensor's
      bytes per expert (nb[2]) with its type and width, and the preparation hands all three to the cache. A move to
      the device mints every member's storage at its triplet's slot (expert_triplet_slot_bytes), the largest of the
      padded staged sizes of the three members' bytes, asked of the layout (staged_expert_bytes), and a move to the
      host mints pinned storage for exactly those bytes; the preparation sizes by the kind's direction alone, and the
      transaction sizes nothing. Never the host copy's size, never a member's own staged size and never computed for
      one type: the cache refuses a host copy shorter than the expert ("host copy size", a refusal a later boundary
      meets again) and storage that is not exactly the triplet's slot or is smaller than the expert, for a refill as
      for an admit, and the device entry takes the storage's size;
  A5  the copy moves exactly the expert's bytes from offset 0 of the host copy, the padded tail is zero-filled on the
      copy's queue behind the copy, and the device entry is published behind the fill (its ready event and last
      write), as the record the transaction remembers is, whose logical bytes are the tensor's bytes per expert in
      every direction, the range the record's readers resolve;
  A6  one publish per call: the body publishes once and bumps the exec-graph epoch with it, unconditionally, and the
      entry runs the body once. An exception propagates: the entry catches nothing, and the body's only rethrow after
      a publish comes after it, so the triplets moved before the exception are published and the plan, not an
      outcome list, records them;
  A7  the ledger is untouched by a named move: no ledger room reserved, no lent triplet appended or removed, no cursor
      disarmed, no ledger mirror or total stored;
  A8  the cache's admit is the refill body with the admit's source rule: refill_lent_triplet passes false and stays as
      strict ("not lent", "not a lent copy"), admit_host_triplet passes true, and an admit's host copy must be pinned
      storage it owns whole, its owner exactly the entry's base and size, and a host-zone slice ("host copy not
      pinned" otherwise), while a refill's owner need only hold the entry, as the loan that lent it requires;
  A9  a named move WARNs no refusal (its refusals skip the loan's WARN; its caller reads each answer from the
      outcomes), and outside the private test fixtures the entry has exactly one caller, the decode-boundary swap driver
      (ggml_sycl_run_expert_swap_at_boundary), which calls it once;
  A10 the named selection takes the published plan's verdict (moe_expert_move_validate) and moves only the triplets it
      lets through;
  A11 every refusal either cache body assigns, the copy to the device and the loan's copy to the host, is classed
      exactly once: final (in the transaction's one final list, ggml_sycl_expert_move_refusal_is_final) or transient
      (in this pin's TRANSIENT_REFUSALS for the copy to the device and LOAN_TRANSIENT_REFUSALS for the copy to the
      host, each with the path that clears it), and each classified reason is one a body assigns. A refusal is final
      when the same inputs give it again with nothing changed, so a renamed or new refusal cannot be retried at every
      decode graph, or answered BUSY to a named move at every boundary, unclassified;
  E1  an evict list joins the request: the entry puts the evicts first in the one list the body runs, each marked as
      a move to the host, and the named selection judges the evicts and the admits together, each triplet by its own
      list's verdict;
  E2  a named triplet's kind follows its list, EVICT for the evict list; an evict is a move to the host (the
      direction helper sends REFILL and ADMIT to the device and LOAN and EVICT to the host), so the preparation mints
      pinned storage at the tensor's bytes and the move statement hands it to the loan's cache body; the helper and
      the move statement each switch on the kind with every kind named and no default, so a new kind is a compile
      warning, not a silent fall into another kind's body; its texts are a named move's, and an exhausted pinned pool
      is BUSY for an evict while it stays REFUSED for a loan;
  E3  an evict's record path moves the tensor's storage generation: the to-host branch drops the extra's views of
      the withdrawn entry (a dropped storage record or table slot bumps the generation) and forgets its device record
      before it remembers the host copy (every remember bumps it), and the decode direct route keys its stamp on that
      generation, so a route that served the block from a full-coverage table decides again and never serves it from
      the old table, which points at retired storage;
  E4  a cache refusal of a named triplet, admit or evict, refuses that triplet alone, classified, and the next one
      moves; it never ends the list;
  E5  each moved outcome records the device storage it took or gives back; the body sums the bytes once per
      direction, as each triplet moves, writes them once the plan is published and names both directions in its
      result line, and the entry takes those sums, counts each outcome in its direction and returns it in its own list;
  M1  a named move's one result line is a WARN (every tool drops INFO) that carries, summed over the batch, the
      microseconds the cache spent in each move's copy step and in its id handover and retire, and the transaction in
      the members' metadata lookups: each cache body, the loan's and the copy to the device, records its two times in
      the move record, the preparation times its resolver call, and the body sums all three as each call returns.

Checks run on comment-stripped, whitespace-normalized text. NOT VACUOUS: --self-test runs the gate on in-memory mutants
of the real sources. Each mutant names the pin that must catch it and is reported if it survives, if another pin
catches it first, or if its anchor no longer matches exactly once in the body it edits (NOT APPLIED). The self-test
also proves that last report fires. ctest registers it with --self-test.

WHAT THIS DOES NOT PROVE. It reads text. That the copy and the fill run in that order on the device, that the tail
reads back zero, that a kernel reads the admitted bytes through the direct route, that an evicted block is no longer
served from its old table and that an evict's room comes back only after the reap are the device cases' job.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "sycl": ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp",
    "cache": ROOT / "ggml/src/ggml-sycl/unified-cache.cpp",
    "pad": ROOT / "ggml/src/ggml-sycl/weight-row-padding.hpp",
    "common": ROOT / "ggml/src/ggml-sycl/common.hpp",
    "hostpath": ROOT / "ggml/src/ggml-sycl/moe-decode-hostpath.hpp",
    "plan": ROOT / "ggml/src/ggml-sycl/moe-stream-plan.hpp",
}

_LEXEME_RE = re.compile(r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'", re.DOTALL)
_INVENTORY_GUARD_RE = re.compile(r"std::(lock_guard|unique_lock) ?<[^<>;]*> ?(\w+)\(g_tensor_inventory_mutex\)")


class ContractError(AssertionError):
    pass


class MutantNotApplied(Exception):
    pass


def strip_comments(src: str) -> str:
    """Comments blanked to their newlines, literals kept."""

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def norm(text: str) -> str:
    """Comment-stripped, whitespace-canonical text with adjacent string literals joined, and every guard on
    g_tensor_inventory_mutex in its deduced spelling, so a clang-format re-wrap cannot turn a pin red."""
    t = re.sub(r"\s+", " ", strip_comments(text))
    t = re.sub(r"\( ", "(", t)
    t = re.sub(r" \)", ")", t)
    t = re.sub(r" ,", ",", t)
    t = re.sub(r",(?! )", ", ", t)
    t = re.sub(r'(?<!\\)" "', "", t)
    return _INVENTORY_GUARD_RE.sub(r"std::\1 \2(g_tensor_inventory_mutex)", t)


def body_span(text: str, signature: str) -> tuple[int, int]:
    """The [start, end) span of the brace-balanced body of the first definition starting with `signature`, string
    and character literals skipped while balancing; (-1, -1) when there is none. A declaration, whose parameter list
    closes on a semicolon, is passed over: the decode direct route is declared early for the test cases that read it."""
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
        return block_span(text, text.find("{", at + len(signature)))
    return -1, -1


def block_span(text: str, open_at: int) -> tuple[int, int]:
    """The [start, end) span of the brace-balanced block opening at `open_at`; (-1, -1) when there is none. Comments
    are skipped as well as literals: the mutants edit raw text, whose comments hold apostrophes ("a loan's")."""
    if open_at < 0:
        return -1, -1
    depth = 0
    i = open_at
    n = len(text)
    while i < n:
        ch = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j + 1
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
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


def fail(pin: str, what: str) -> None:
    raise ContractError(f"FAIL [pin {pin}]: {what}")


# --- signatures -----------------------------------------------------------------------------------------------------

TXN_SIG = "static ggml_sycl_txn_result ggml_sycl_expert_txn_run("
ENTRY_SIG = "static ggml_sycl_txn_result ggml_sycl_expert_move_transaction_run("
SELECT_SIG = "static std::vector<ggml_sycl::expert_placement_key> ggml_sycl_expert_named_select("
STOP_SIG = "static const char * ggml_sycl_expert_named_stop("
RESOLVE_SIG = "static bool ggml_sycl_expert_loan_resolve_registered("
COPY_SIG = "bool unified_cache::copy_host_triplet_to_device("
ADMIT_SIG = "bool unified_cache::admit_host_triplet("
REFILL_SIG = "bool unified_cache::refill_lent_triplet("
STAGED_SIG = "inline size_t staged_expert_bytes("
PREP_SIG = "static const char * ggml_sycl_expert_prepare_triplet("
LOAN_SIG = "bool unified_cache::loan_expert_triplet("
DIRECTION_SIG = "static bool ggml_sycl_expert_move_to_device("
TEXTS_SIG = "static ggml_sycl_expert_move_texts ggml_sycl_expert_move_texts_of("
DROP_SIG = "static void ggml_sycl_drop_lent_expert_views("
RECORD_SIG = "bool remember_moe_storage_handle("
ROUTE_SIG = "static const void * const * ggml_sycl_moe_decode_direct_table("
STAMP_SIG = "inline bool moe_decode_direct_stamp_current("

# --- needles (normalized) -------------------------------------------------------------------------------------------

EVENT_WAITS = (".wait(", "wait_and_throw(", "get_info<", "event_complete(", "command_execution_status",
               "ext_oneapi_get_state", "finalize_retired_entries_locked(", "drain_retained_handles(")

MINT_PREFIX = "= ggml_sycl_device_expert_alloc(device, "
MINT_CALLS = ("ggml_sycl_device_expert_alloc(", "ggml_sycl_host_expert_alloc(")
PREP_CALL = ("stop = ggml_sycl_expert_prepare_triplet(*cache, device, *current->plan, &keys[3 * t], *resolver, "
             "triplet, &transient);")
INVENTORY_LOCK = "std::unique_lock lock(g_tensor_inventory_mutex);"
ADMIT_PREFIX = "cache->admit_host_triplet("
# The move statement: a switch on the triplet's kind with no default, so a new kind is a compile warning.
ADMIT_CALL = ("switch (triplet.kind) { case ggml_sycl_expert_move_kind::LOAN: case ggml_sycl_expert_move_kind::EVICT: "
              "loaned = cache->loan_expert_triplet(triplet.members, record); break; "
              "case ggml_sycl_expert_move_kind::ADMIT: loaned = cache->admit_host_triplet(triplet.members, record); "
              "break; case ggml_sycl_expert_move_kind::REFILL: "
              "loaned = cache->refill_lent_triplet(triplet.members, record); break; }")
# Where the admit is asked, for the pins that order what follows it.
ADMIT_AT = "loaned = cache->admit_host_triplet(triplet.members, record);"

LOANS_CHECK = ('if (named && g_sycl_expert_lent_count[device].load(std::memory_order_acquire) != 0) '
               '{ return refuse(ggml_sycl_txn_result::BUSY, "loans outstanding"); }')
LOANS_RECHECK = ('if (named && !ledger.triplets.empty()) { lock.unlock(); '
                 'return refuse(ggml_sycl_txn_result::BUSY, "loans outstanding"); }')
LEASE = "registry.acquire_live_update(current_token)"
LEDGER_LOCKED = "ggml_sycl_expert_lent_ledger & ledger = ggml_sycl_expert_lent_ledger_locked(device, *current);"

PREP_DIRECTION = "const bool to_device = ggml_sycl_expert_move_to_device(out.kind);"
STAGED_SIZE = ("staged[r] = "
               "ggml_sycl::staged_expert_bytes(e->layout, target.type, target.ncols, target.logical_bytes);")
SLOT = "const size_t slot = to_device ? ggml_sycl::expert_triplet_slot_bytes(staged) : 0;"
MINT = "storage = ggml_sycl_device_expert_alloc(device, slot);"
HOST_MINT = "storage = ggml_sycl_host_expert_alloc(device, target.logical_bytes);"
MEMBER_TYPE = "m.type = target.type;"
MEMBER_NCOLS = "m.ncols = target.ncols;"
MEMBER_LOGICAL = "m.logical_bytes = target.logical_bytes;"
RESOLVED_TYPE = "out.type = tensor->type;"
RESOLVED_NCOLS = "out.ncols = tensor->ne[0];"
RESOLVED_LOGICAL = "out.logical_bytes = tensor->nb[2];"
STAGED_GUARD = "if (layout != GGML_LAYOUT_AOS ||"
STAGED_AOS = "return aos_staged_weight_bytes(type, ncols, expert_bytes);"
CACHE_LOGICAL = "const size_t logical = members[i].logical_bytes;"
HOST_SIZE_GUARD = 'if (e.size < logical) { refusal = "host copy size"; break; }'
CACHE_STAGED = ("staged[i] = staged_expert_bytes(GGML_LAYOUT_AOS, members[i].type, members[i].ncols, "
                "members[i].logical_bytes);")
CACHE_SLOT = "const size_t slot = expert_triplet_slot_bytes(staged);"
SIZE_GUARD = 'if (device_size[i] != slot || slot < logical) { refusal = "device storage size"; break; }'
COPY_BYTES = "bytes[i] = logical;"
FINAL_SIG = "static bool ggml_sycl_expert_move_refusal_is_final("
ENTRY_SIZE = "entry.size = device_size[i];"
MIRROR_SIZE = "mirror.size = device_size[i];"

COPY = "out.copy_events[i] = mem_copy_async(members[i].owner, src_owner[i], bytes[i], bq, deps[i]);"
FILL = ("if (device_size[i] > bytes[i]) { out.copy_events[i] = mem_fill_async(members[i].owner, bytes[i], 0, "
        "device_size[i] - bytes[i], bq, { out.copy_events[i] }); }")
ENTRY_READY = "entry.ready_event = out.copy_events[i];"
ENTRY_WRITE = "entry.last_write_event = out.copy_events[i];"
EMPLACE = "entries_.emplace(device_key, device_entry[i]);"
REMEMBER = ("(void) triplet.extras[r]->remember_moe_storage_handle(triplet.members[r].placement.expert_id, "
            "GGML_LAYOUT_AOS, std::move(record.handles[r]), &record.copy_events[r], 0, "
            "triplet.members[r].logical_bytes);")

PUBLISH = "ggml_sycl_publish_prepared_plan_locked(prepared_publication);"
SWAP_PUBLISH = ("if (swapped) { ggml_sycl_publish_prepared_plan_locked(prepared_publication); "
                "ggml_sycl_optional_layouts_retired(); latch.armed = false; }")
RETHROW = "std::rethrow_exception(escaped);"
RETHROW_LAST = "if (escaped) { std::rethrow_exception(escaped); } return finish(ggml_sycl_txn_result::ACCEPTED, reason);"
NAMED_CALL = ("const ggml_sycl_txn_result code = ggml_sycl_expert_txn_run(ggml_sycl_expert_txn_direction::NAMED, "
              "device, moves.size(), 0, false, resolver, &answer, false, &moves);")

RESERVE_GUARD = "if (!refill && !named) { ledger.triplets.reserve("
# A named triplet, admit or evict, records only its outcome: that it moved and the device storage it took or gives
# back. Only a loan appends to the ledger and only a refill pops it.
MOVED_BRANCH = ("switch (triplet.kind) { case ggml_sycl_expert_move_kind::LOAN: ledger.triplets.push_back({ placed[0], "
                "placed[1], placed[2] }); ledger.triplet_bytes.push_back(ggml_sycl::moe_expert_triplet_dst_bytes("
                "*next_plan, placed)); break; case ggml_sycl_expert_move_kind::REFILL: ledger.triplets.pop_back(); "
                "ledger.triplet_bytes.pop_back(); break; case ggml_sycl_expert_move_kind::ADMIT: "
                "case ggml_sycl_expert_move_kind::EVICT: (*named_moves)[triplet.named].moved = true; "
                "(*named_moves)[triplet.named].device_bytes = record.device_bytes; break; }")
TAIL_GUARD = "if (!named) { if (!refill && lent != 0) { ledger.refill_armed = false; }"
TAIL_WRITES = ("ledger.refill_armed = false;", "g_sycl_expert_lent_count[device].store(",
               "g_sycl_expert_lent_bytes[device].store(", "g_sycl_expert_readmitted_total[device].fetch_add(",
               "g_sycl_expert_lent_total[device].fetch_add(", "result.n_remaining =")
CURSOR = "if (triplet.kind == ggml_sycl_expert_move_kind::REFILL && (!ledger.refill_armed ||"
NAMED_REFUSE = ("refuse_named(triplet.named, ggml_sycl_expert_move_refusal_is_final(record.reason) ? "
                "ggml_sycl_txn_result::REFUSED : ggml_sycl_txn_result::BUSY, record.reason); continue; }")
NAMED_CACHE_REFUSAL = "if (!loaned && (admit || evict)) { " + NAMED_REFUSE
DROP_GUARD = ("if (triplet.kind == ggml_sycl_expert_move_kind::REFILL) { transient = "
              "!ggml_sycl_expert_move_refusal_is_final(stop); if (!transient) { drop_back(stop); } }")

REFILL_WRAP = "{ return copy_host_triplet_to_device(members, false, out); }"
ADMIT_WRAP = "{ return copy_host_triplet_to_device(members, true, out); }"
NOT_THERE = 'refusal = admit ? "not on host" : "not lent";'
PINNED_CHECK = ("if (e.location != cache_location::HOST_PINNED || !e.host_resident || "
                "!e.direct_alloc_owner.owner_metadata(&md) || md.ptr != e.device_ptr || "
                "md.tier != alloc_tier::HOST_PINNED || (admit ? md.size != e.size : md.size < e.size) || "
                "(admit && md.host_zone == host_zone_id::COUNT)) "
                '{ refusal = admit ? "host copy not pinned" : "not a lent copy"; break; }')

REFUSE_GATE = "if (!probe_mode && !refill && !named && (warn_busy || code != ggml_sycl_txn_result::BUSY)) {"
NAMED_FORMAT = ('"[MOE-EXEC] named move device %d: %u evicted (%zu MiB off), %u admitted (%zu MiB on) '
                '(%s); copy submit %llu us, id remap %llu us, meta lookup %llu us, total %llu us\\n"')
NAMED_LINE = "GGML_LOG_WARN(" + NAMED_FORMAT
ENTRY_NAME = "ggml_sycl_expert_move_transaction_run("
# The entry's one production caller: the swap driver at the synchronize boundary.
DRIVER_SIG = "static void ggml_sycl_run_expert_swap_at_boundary("

VALIDATE = ("const ggml_sycl::moe_expert_move_check check = ggml_sycl::moe_expert_move_validate(plan, device, evict, "
            "admit);")

# The cache's refusals that clear by themselves, each with the path that clears it. Every other refusal the shared
# body assigns gives the same answer again with nothing changed, so it belongs in the transaction's final list.
TRANSIENT_REFUSALS = (
    "load effect unavailable",  # the load this thread is bound to commits or is abandoned
    "plan not coherent",  # a publication completes whose plan uses the card, giving its cache the plan
    "device entry exists",  # finalize_retired_entries_locked reaps the retired entry once it drains
    "copy submit failed",  # the runtime threw on a copy or fill submit
    "entry changed",  # the host copy changed between the check and the publish
    "publication failed",  # an allocation under the cache lock threw
)
# The same for the loan's cache body, which an evict runs: each clears by itself, so an evict meets it BUSY.
LOAN_TRANSIENT_REFUSALS = (
    "load effect unavailable",  # the load this thread is bound to commits or is abandoned
    "plan not coherent",  # a publication completes whose plan uses the card, giving its cache the plan
    "storage still referenced",  # the retention an in-flight copy keeps on the owner is released
    "host entry retiring",  # finalize_retired_entries_locked reaps the retired host entry an admit or a refill left
    "copy submit failed",  # the runtime threw on a copy submit
    "entry changed",  # the device entry changed between the check and the retire
    "host registration failed",  # an allocation under the cache lock threw
)
SUCCESS_REASON = "OK"
_REASON_ASSIGN_RE = re.compile(r"\b(?:refusal|out\.reason) = ([^;]*);")
_LITERAL_RE = re.compile(r'"((?:\\.|[^"\\])*)"')
_FINAL_LIST_RE = re.compile(r"final_reasons\[\] = \{([^}]*)\}")
# A refusal the cache gives and the transaction classifies by a named constant: its one spelling is in the plan header.
_NAMED_REASON_DEF_RE = re.compile(r"constexpr const char (MOE_EXPERT_MOVE_\w+)\[\] = (\"(?:\\.|[^\"\\])*\");")


def resolve_named_reasons(plan: str, *texts: str) -> list[str]:
    """Each named refusal constant in `texts` replaced by the literal the plan header defines it as, so A11 reads the
    reason a body assigns or lists whether it is spelled as a literal or by its name."""
    out = list(texts)
    for name, literal in _NAMED_REASON_DEF_RE.findall(plan):
        use = re.compile(r"(?:\bggml_sycl::)?\b" + name + r"\b(?!\[)")
        out = [use.sub(lambda _m: literal, t) for t in out]
    return out
# Each named triplet takes its own list's verdict: the plan judged the evicts and the admits in two lists.
VERDICT = ("const char * verdict = moves[i].to_device ? check.admit[n_admit++] : check.evict[n_evict++]; "
           "if (verdict) { moves[i].code = ggml_sycl_txn_result::REFUSED; moves[i].reason = verdict; continue; }")
SELECT_CALL = "named ? ggml_sycl_expert_named_select(*current->plan, device, *named_moves, named_place) :"

# E1: one list, the evicts first, each marked by its direction; the selection splits it back for the plan's verdict.
ENTRY_ORDER = ("moves[i].to_device = i >= n_evict; "
               "moves[i].triplet = i < n_evict ? request.evict[i] : request.admit[i - n_evict];")
SELECT_SPLIT = ("for (const ggml_sycl_expert_move_outcome & m : moves) { "
                "(m.to_device ? admit : evict).push_back(m.triplet); }")
# E2: the kind of a named triplet, the direction an evict takes, its texts and its full pool.
NAMED_PLACE = "triplet.named = named ? named_place[t] : 0;"
PREP_KIND = "triplet.kind = named && !(*named_moves)[triplet.named].to_device ? ggml_sycl_expert_move_kind::EVICT : kind;"
# The direction helper, a switch with no default: every kind named, REFILL and ADMIT to the device.
TO_DEVICE_KINDS = ("{ switch (kind) { case ggml_sycl_expert_move_kind::REFILL: case ggml_sycl_expert_move_kind::ADMIT: "
                   "return true; case ggml_sycl_expert_move_kind::LOAN: case ggml_sycl_expert_move_kind::EVICT: "
                   'return false; } GGML_ABORT("unknown expert move kind"); }')
EVICT_TEXTS = ('case ggml_sycl_expert_move_kind::EVICT: return { "no tensor for a named expert", "not resident", '
               '"exception while moving", "named move" };')
EVICT_POOL_BUSY = ('if (!storage.valid()) { *transient = out.kind == ggml_sycl_expert_move_kind::EVICT; '
                   'return "host pinned pool exhausted"; }')
LEND_CALL = "cache->loan_expert_triplet(triplet.members, record)"
# E3: the to-host branch, a loan's and an evict's, and the generation every step of it moves.
TO_HOST_BRANCH = ("} else { for (int r = 0; r < 3; ++r) { const ggml_sycl::expert_placement_key & k = "
                  "triplet.members[r].placement; ggml_tensor_extra_gpu * extra = triplet.extras[r]; "
                  "ggml_sycl_drop_lent_expert_views(extra, device, k.expert_id, record.device_keys[r], lent_views); "
                  "(void) extra->forget_moe_storage_handle_on_device(k.expert_id, GGML_LAYOUT_AOS, device); } }")
DROP_BUMP = "if (stored) { ++w.moe_expert_storage_generation; }"
DROP_STORED = ("stored = list.size() != before;", "stored = true;")
RECORD_BUMPS = ("existing = std::move(record); ++moe_expert_storage_generation; return true;",
                "records.push_back(std::move(record)); ++moe_expert_storage_generation; return true;")
ROUTE_GENERATION = "const uint64_t storage_generation = extra->weight().moe_expert_storage_generation;"
ROUTE_STAMP = ("if (ggml_sycl::moe_decode_direct_stamp_current(stamp, replan_epoch, storage_generation, "
               "selected_rows)) {")
STAMP_COMPARES = "s.expert_storage_generation == expert_storage_generation"
# E5: the body sums each direction's bytes once, as each triplet moves, and writes them once the plan is published;
# the entry copies them and only counts and splits the outcomes back per direction.
BODY_SUM = "++lent; (to_device ? taken : freed) += record.device_bytes;"
BODY_RESULT = "result.bytes_freed = freed; result.bytes_readmitted = taken;"
BODY_COUNT = "if (!m.to_device) { ++n_evicted; }"
NAMED_LINE_ARGS = ("device, n_evicted, freed / (1024 * 1024), lent - n_evicted, taken / (1024 * 1024), reason, "
                   "(unsigned long long) copy_us, (unsigned long long) remap_us, (unsigned long long) meta_us, "
                   "(unsigned long long) ggml_sycl::expert_move_us_since(txn_start));")
# M1: what the body sums into the line, and where each figure is measured.
SUM_META = "meta_us += triplet.resolve_us;"
# The line's total runs from the transaction's first statement, before the lock and the preparation.
TXN_START = "const auto txn_start = std::chrono::steady_clock::now();"
SUM_CACHE = "copy_us += record.copy_us; remap_us += record.remap_us;"
CACHE_TIMES = ("out.copy_us = copy_took;", "out.remap_us = remap_took;")
PREP_TIMED = "const bool resolved = resolver(*e, target); out.resolve_us += ggml_sycl::expert_move_us_since(resolve_from);"
ENTRY_BYTES = "r.bytes_evicted = answer.bytes_freed; r.bytes_admitted = answer.bytes_readmitted;"
ENTRY_SPLIT = ("for (ggml_sycl_expert_move_outcome & m : moves) { if (m.moved) { "
               "++(m.to_device ? r.n_admitted : r.n_evicted); } (m.to_device ? r.admitted : r.evicted).push_back(m); }")


def _private_testing_lines(raw: str) -> list[bool]:
    """Per line of comment-stripped `raw`, whether it sits inside a `#if defined(GGML_SYCL_PRIVATE_TESTING)` block
    (not its #else). Nested conditionals keep the outer block's answer."""
    stack: list[bool] = []
    out = []
    for line in strip_comments(raw).split("\n"):
        s = line.strip()
        if re.match(r"#\s*if", s):
            stack.append("GGML_SYCL_PRIVATE_TESTING" in s and "!" not in s and not re.match(r"#\s*ifndef", s))
        elif re.match(r"#\s*(else|elif)", s) and stack:
            stack[-1] = False
        elif re.match(r"#\s*endif", s) and stack:
            stack.pop()
        out.append(any(stack))
    return out


def check(raw: dict[str, str]) -> None:
    plan = norm(raw["plan"])
    if not _NAMED_REASON_DEF_RE.search(plan):
        fail("A11", "the plan header defines no named refusal (MOE_EXPERT_MOVE_SHARED_STORAGE) this pin can resolve")
    sycl, cache = resolve_named_reasons(plan, norm(raw["sycl"]), norm(raw["cache"]))
    pad = norm(raw["pad"])
    txn = body(sycl, TXN_SIG)
    entry = body(sycl, ENTRY_SIG)
    select = body(sycl, SELECT_SIG)
    stop = body(sycl, STOP_SIG)
    resolve = body(sycl, RESOLVE_SIG)
    copy = body(cache, COPY_SIG)
    admit = body(cache, ADMIT_SIG)
    refill = body(cache, REFILL_SIG)
    staged = body(pad, STAGED_SIG)
    final = body(sycl, FINAL_SIG)
    prep = body(sycl, PREP_SIG)
    common = norm(raw["common"])
    loan = body(cache, LOAN_SIG)
    direction = body(sycl, DIRECTION_SIG)
    texts = body(sycl, TEXTS_SIG)
    drop = body(sycl, DROP_SIG)
    record = body(common, RECORD_SIG)
    route = body(sycl, ROUTE_SIG)
    stamp = body(norm(raw["hostpath"]), STAMP_SIG)

    # A0 ---------------------------------------------------------------------------------------------------------------
    named = {"transaction body": txn, "entry": entry, "named selection": select, "named stop": stop,
             "registered resolver": resolve, "cache shared body": copy, "cache admit": admit, "cache refill": refill,
             "staged-size helper": staged, "final refusal list": final, "triplet preparation": prep,
             "cache loan": loan, "direction helper": direction, "kind texts": texts, "view drop": drop,
             "record remember": record, "decode direct route": route, "route stamp": stamp}
    for name, text in named.items():
        if not text:
            fail("A0", f"the {name} is missing, so every pin reading it would pass on nothing")

    # A1 ---------------------------------------------------------------------------------------------------------------
    for name in ("transaction body", "entry", "triplet preparation", "named selection", "named stop",
                 "cache shared body", "cache admit", "staged-size helper", "cache loan", "view drop"):
        for token in EVENT_WAITS:
            if token in named[name]:
                fail("A1", f"the {name} names `{token}`: a named move's path never waits or asks an event's status")

    # A2 ---------------------------------------------------------------------------------------------------------------
    if MINT_PREFIX not in prep or any(call in txn for call in MINT_CALLS):
        fail("A2", "the device storage is not minted in the triplet preparation alone")
    prep_at = txn.find(PREP_CALL)
    if txn.count(INVENTORY_LOCK) != 1 or prep_at < 0 or not ordered(txn, PREP_CALL, INVENTORY_LOCK, ADMIT_PREFIX):
        fail("A2", "the triplets are not prepared before the inventory lock, or the admit is not asked under it")
    if txn.find("g_tensor_inventory_mutex") < prep_at or "g_tensor_inventory_mutex" in prep:
        fail("A2", "the inventory mutex is named before or while the storage is minted")
    if "g_tensor_inventory_mutex" in copy or "g_tensor_inventory_mutex" in admit:
        fail("A2", "the cache takes the inventory mutex")

    # A3 ---------------------------------------------------------------------------------------------------------------
    if txn.count(LOANS_CHECK) != 1 or txn.count(LOANS_RECHECK) != 1:
        fail("A3", "a named move is not refused BUSY 'loans outstanding' while triplets are lent, before and under the "
                   "inventory lock")
    if not ordered(txn, LOANS_CHECK, LEASE, PREP_CALL, INVENTORY_LOCK, LEDGER_LOCKED, LOANS_RECHECK, ADMIT_PREFIX):
        fail("A3", "the lent-count refusal is not before the lease, or its re-check is not under the inventory lock "
                   "before the admit")

    # A4 ---------------------------------------------------------------------------------------------------------------
    if not (prep.count(PREP_DIRECTION) == 1 and prep.count(STAGED_SIZE) == 1 and prep.count(SLOT) == 1
            and prep.count(MINT) == 1 and prep.count("ggml_sycl_device_expert_alloc(") == 1
            and ordered(prep, PREP_DIRECTION, STAGED_SIZE, SLOT, MINT)):
        fail("A4", "a move to the device is not minted at the triplet's slot, decided by the kind's direction")
    if prep.count(HOST_MINT) != 1 or "staged_expert_bytes(" in txn:
        fail("A4", "a move to the host is not minted at the tensor's bytes, or the transaction sizes a copy itself")
    if not (prep.count(MEMBER_TYPE) == 1 and prep.count(MEMBER_NCOLS) == 1 and RESOLVED_TYPE in resolve
            and RESOLVED_NCOLS in resolve):
        fail("A4", "the type and width the staged size is asked with are not the resolved tensor's")
    if not (prep.count(MEMBER_LOGICAL) == 1 and RESOLVED_LOGICAL in resolve):
        fail("A4", "the expert's bytes every move sizes, copies and records are not its tensor's nb[2]")
    if not (copy.count(CACHE_LOGICAL) == 1 and copy.count(HOST_SIZE_GUARD) == 1
            and ordered(copy, CACHE_LOGICAL, HOST_SIZE_GUARD, SIZE_GUARD) and '"host copy size"' in final):
        fail("A4", "the cache does not refuse, as final, a host copy shorter than the expert's tensor bytes")
    if not (STAGED_GUARD in staged and staged.count(STAGED_AOS) == 1):
        fail("A4", "staged_expert_bytes does not ask the layout, or does not give AOS the padded size")
    if not (copy.count(CACHE_STAGED) == 1 and copy.count(CACHE_SLOT) == 1 and copy.count(SIZE_GUARD) == 1
            and ordered(copy, CACHE_STAGED, CACHE_SLOT, SIZE_GUARD)):
        fail("A4", "the cache does not refuse admit storage that is not exactly the triplet's slot")
    if not (copy.count(ENTRY_SIZE) == 1 and copy.count(MIRROR_SIZE) == 1):
        fail("A4", "the device entry is not sized as its storage")
    for name, text in (("transaction body", txn), ("triplet preparation", prep), ("cache shared body", copy),
                       ("staged-size helper", staged)):
        if re.search(r"GGML_TYPE_(?!COUNT\b)\w+", text):
            fail("A4", f"the {name} names a weight type: the staged size is asked of the layout, never computed for one")

    # A5 ---------------------------------------------------------------------------------------------------------------
    if not (copy.count(COPY) == 1 and copy.count(FILL) == 1 and copy.count(ENTRY_READY) == 1
            and copy.count(ENTRY_WRITE) == 1 and ordered(copy, COPY, FILL, ENTRY_READY, ENTRY_WRITE, EMPLACE)):
        fail("A5", "the tail fill is not behind the copy on its queue, or the entry is not published behind the fill")
    if not (copy.count(COPY_BYTES) == 1 and ordered(copy, SIZE_GUARD, COPY_BYTES, COPY)):
        fail("A5", "the copy does not move exactly the expert's bytes, so a longer host copy's tail would be read")
    if not (txn.count(REMEMBER) == 1 and ordered(txn, ADMIT_AT, REMEMBER)):
        fail("A5", "the moved record is not remembered behind its last write with the tensor's bytes as its range")

    # A6 ---------------------------------------------------------------------------------------------------------------
    if txn.count(PUBLISH) != 1 or txn.count(SWAP_PUBLISH) != 1:
        fail("A6", "the body does not publish exactly once with an unconditional epoch bump")
    at = entry.find(NAMED_CALL)
    if entry.count("ggml_sycl_expert_txn_run(") != 1 or at < 0:
        fail("A6", "the entry does not run the body exactly once in the NAMED direction")
    if entry[:at].count("{") - entry[:at].count("}") != 1:
        fail("A6", "the entry runs the body inside a nested block, so it may run it more than once")
    if re.search(r"\b(try|catch)\b", entry):
        fail("A6", "the entry catches what the body throws, so a caller could read outcomes of a call that did not end")
    if txn.count(RETHROW) != 3 or txn.count(RETHROW_LAST) != 1 or not ordered(txn, SWAP_PUBLISH, RETHROW_LAST):
        fail("A6", "the body can rethrow before it publishes what it moved, or rethrows at a place not accounted for")

    # A7 ---------------------------------------------------------------------------------------------------------------
    if txn.count(RESERVE_GUARD) != 1 or txn.count(MOVED_BRANCH) != 1 or txn.count(CURSOR) != 1:
        fail("A7", "a named move reserves, appends to or pops the ledger, or reads the refill cursor")
    tail_at = txn.find(TAIL_GUARD)
    start, end = block_span(txn, txn.find("{", tail_at) if tail_at >= 0 else -1)
    guarded = txn[start:end] if start >= 0 else ""
    if txn.count(TAIL_GUARD) != 1 or not all(guarded.count(w) == 1 for w in TAIL_WRITES):
        fail("A7", "the ledger's cursor, mirrors or totals are written outside the guard that keeps a named move out")
    if txn.count(DROP_GUARD) != 1 or not ordered(txn, ADMIT_AT, NAMED_REFUSE, DROP_GUARD):
        fail("A7", "a cache refusal of a named triplet can reach the ledger drop")

    # A8 ---------------------------------------------------------------------------------------------------------------
    if REFILL_WRAP not in refill or ADMIT_WRAP not in admit:
        fail("A8", "the refill and the admit do not run the shared body with false and true")
    if copy.count(NOT_THERE) != 1 or copy.count(PINNED_CHECK) != 1:
        fail("A8", "the admit's host copy is not required to be exactly its whole pinned storage in a host zone, or the "
                   "refill's owner bound is not the loan's")

    # A9 ---------------------------------------------------------------------------------------------------------------
    if txn.count(REFUSE_GATE) != 1:
        fail("A9", "a named move's refusal is logged")
    private = _private_testing_lines(raw["sycl"])
    stripped = strip_comments(raw["sycl"])
    lines = stripped.split("\n")
    d_start, d_end = body_span(stripped, DRIVER_SIG)
    d_first = stripped.count("\n", 0, d_start) if d_start >= 0 else -1
    d_last = stripped.count("\n", 0, d_end) if d_start >= 0 else -1
    production = []
    for k, line in enumerate(lines):
        if ENTRY_NAME in line and ENTRY_SIG.rstrip("(") not in re.sub(r"\s+", " ", line) and not private[k]:
            production.append(k)
    for k in production:
        if not d_first <= k <= d_last:
            fail("A9", f"line {k + 1} calls the named move outside the swap driver and the private test fixtures")
    if len(production) != 1:
        fail("A9", f"the swap driver calls the named move {len(production)} times, not once")

    # A10 --------------------------------------------------------------------------------------------------------------
    if select.count(VALIDATE) != 1 or select.count(VERDICT) != 1 or txn.count(SELECT_CALL) != 1:
        fail("A10", "the named selection does not take the published plan's verdict on each triplet")

    # A11 --------------------------------------------------------------------------------------------------------------
    listed = _FINAL_LIST_RE.search(final)
    finals = _LITERAL_RE.findall(listed.group(1)) if listed else []
    assigned_by = {}
    for name, text, transients in (("copy to the device", copy, TRANSIENT_REFUSALS),
                                   ("loan's copy to the host", loan, LOAN_TRANSIENT_REFUSALS)):
        assigned = [lit for rhs in _REASON_ASSIGN_RE.findall(text) for lit in _LITERAL_RE.findall(rhs)]
        if not finals or assigned.count(SUCCESS_REASON) != 1:
            fail("A11", f"the final refusal list or the {name}'s reason assignments are not where this pin reads them")
        refusals = [r for r in assigned if r != SUCCESS_REASON]
        assigned_by[name] = refusals
        for reason in refusals:
            if reason in finals and reason in transients:
                fail("A11", f'the {name}\'s refusal "{reason}" is both final and transient')
            if reason not in finals and reason not in transients:
                fail("A11", f'the {name}\'s refusal "{reason}" is neither final nor transient, so it is retried at '
                            "every boundary unclassified")
        for reason in transients:
            if reason not in refusals:
                fail("A11", f'"{reason}" is classified transient, but the {name} assigns no such refusal')
    for reason in finals:
        if not any(reason in refusals for refusals in assigned_by.values()):
            fail("A11", f'"{reason}" is classified final, but no cache body assigns such a refusal')

    # E1 ---------------------------------------------------------------------------------------------------------------
    if entry.count(ENTRY_ORDER) != 1 or select.count(SELECT_SPLIT) != 1 or not ordered(select, SELECT_SPLIT, VALIDATE,
                                                                                         VERDICT):
        fail("E1", "the evicts are not put first in the one list, or not judged with the admits by their own verdicts")

    # E2 ---------------------------------------------------------------------------------------------------------------
    if txn.count(PREP_KIND) != 1 or not ordered(txn, NAMED_PLACE, PREP_KIND, PREP_CALL):
        fail("E2", "a named triplet's kind does not follow its list before it is prepared")
    if direction != TO_DEVICE_KINDS or "default" in direction or ADMIT_CALL not in txn or LEND_CALL not in txn:
        fail("E2", "an evict is not a move to the host through the loan's cache body, or a kind's direction or cache "
                   "body is chosen by anything but a switch with every kind and no default")
    if texts.count(EVICT_TEXTS) != 1:
        fail("E2", "an evict does not answer with a named move's texts and the loan's no-source refusal")
    if prep.count(EVICT_POOL_BUSY) != 1 or not ordered(prep, HOST_MINT, EVICT_POOL_BUSY):
        fail("E2", "an evict's exhausted pinned pool is not BUSY, or a loan's is not REFUSED")

    # E3 ---------------------------------------------------------------------------------------------------------------
    if txn.count(TO_HOST_BRANCH) != 1 or not ordered(txn, ADMIT_AT, TO_HOST_BRANCH, REMEMBER):
        fail("E3", "an evict does not drop the extra's views and its device record before it remembers the host copy")
    if drop.count(DROP_BUMP) != 1 or not all(drop.count(s) == 1 for s in DROP_STORED) or not ordered(
            drop, *DROP_STORED, DROP_BUMP):
        fail("E3", "a dropped storage record or table slot does not move the tensor's storage generation")
    if not all(record.count(b) == 1 for b in RECORD_BUMPS):
        fail("E3", "remembering the host copy does not move the tensor's storage generation")
    if (route.count(ROUTE_GENERATION) != 1 or not ordered(route, ROUTE_GENERATION, ROUTE_STAMP)
            or STAMP_COMPARES not in stamp):
        fail("E3", "the decode direct route does not key its stamp on the tensor's storage generation, so it could "
                   "serve an evicted block from its old table")

    # E4 ---------------------------------------------------------------------------------------------------------------
    if txn.count(NAMED_CACHE_REFUSAL) != 1 or not ordered(txn, ADMIT_AT, NAMED_CACHE_REFUSAL, "if (!loaned) {"):
        fail("E4", "a cache refusal of a named triplet, admit or evict, does not refuse that triplet alone")

    # E5 ---------------------------------------------------------------------------------------------------------------
    if (txn.count(BODY_SUM) != 1 or txn.count(BODY_RESULT) != 1 or txn.count(BODY_COUNT) != 1
            or txn.count(NAMED_LINE_ARGS) != 1 or "+= m.device_bytes" in txn
            or not ordered(txn, BODY_SUM, BODY_RESULT, BODY_COUNT, NAMED_FORMAT, NAMED_LINE_ARGS)):
        fail("E5", "the body does not sum each direction's bytes once, as each triplet moves, or its result line does "
                   "not name both directions")
    if entry.count(ENTRY_BYTES) != 1 or entry.count(ENTRY_SPLIT) != 1 or "device_bytes" in entry:
        fail("E5", "the entry does not take the body's bytes, or does not count each outcome in its direction and "
                   "return it in its own list")

    # M1 ---------------------------------------------------------------------------------------------------------------
    if (txn.count(NAMED_LINE) != 1 or txn.count(SUM_META) != 1 or txn.count(SUM_CACHE) != 1
            or not ordered(txn, "stop = ggml_sycl_expert_prepare_triplet(", SUM_META, "std::unique_lock lock(")
            or not ordered(txn, "loaned = cache->refill_lent_triplet(triplet.members, record); break; }", SUM_CACHE,
                           "if (!loaned && (admit || evict)) {")
            or not ordered(txn, SUM_CACHE, NAMED_LINE)
            or txn.count(TXN_START) != 1 or not txn.startswith("{ " + TXN_START)):
        fail("M1", "the named move's result line is not one WARN carrying the batch's copy submission, id remap and "
                   "meta lookup times, each summed as its call returns, and the transaction's total from its start")
    if (not all(loan.count(t) == 1 and copy.count(t) == 1 for t in CACHE_TIMES) or prep.count(PREP_TIMED) != 1
            or "resolver(*e, target)" in prep.replace(PREP_TIMED, "")):
        fail("M1", "a cache body does not record its copy and handover times, or the preparation does not time its "
                   "resolver call")


def mutate(raw: dict[str, str], edits: list[tuple[str, str, str, str]]) -> dict[str, str]:
    """Apply (file, signature, old, new) edits. With a signature the anchor must appear exactly once inside that body
    (raw text); with an empty one, once in the whole file."""
    out = dict(raw)
    for key, sig, old, new in edits:
        text = out[key]
        if sig:
            start, end = body_span(text, sig)
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

    expect_fail("named-line-debug", "M1", [("sycl", TXN_SIG, 'GGML_LOG_WARN(\n            "[MOE-EXEC] named move',
                                            'GGML_LOG_DEBUG(\n            "[MOE-EXEC] named move')])
    expect_fail("meta-lookup-unsummed", "M1", [("sycl", TXN_SIG, "meta_us += triplet.resolve_us;", "")])
    start_line = "    " + TXN_START + "\n"
    lock_line = "    std::unique_lock lock(g_tensor_inventory_mutex);\n"
    expect_fail("total-from-the-lock", "M1", [("sycl", TXN_SIG, start_line, ""),
                                             ("sycl", TXN_SIG, lock_line, lock_line + start_line)])
    expect_fail("loan-copy-untimed", "M1", [("cache", LOAN_SIG, "out.copy_us  = copy_took;", "")])
    expect_fail("entry-renamed", "A0", [("sycl", "", "ggml_sycl_txn_result ggml_sycl_expert_move_transaction_run(int ",
                                         "ggml_sycl_txn_result ggml_sycl_expert_move_run(int ")])
    expect_fail("fill-waited", "A1", [("cache", COPY_SIG, "bq,\n                                                    "
                                       "{ out.copy_events[i] });",
                                       "bq,\n                                                    "
                                       "{ out.copy_events[i] });\n                out.copy_events[i].wait();")])
    expect_fail("entry-asks-status", "A1", [("sycl", ENTRY_SIG, "    if (out) {",
                                             "    (void) sycl::event{}.get_info<sycl::info::event::"
                                             "command_execution_status>();\n    if (out) {")])
    expect_fail("evict-view-drop-waits", "A1", [("sycl", DROP_SIG, "++w.moe_expert_storage_generation;",
                                                 "++w.moe_expert_storage_generation;\n"
                                                 "        dpct::get_in_order_queue().wait();")])
    expect_fail("evict-cache-loan-reaps", "A1", [("cache", LOAN_SIG, "refusal = MOE_EXPERT_MOVE_SHARED_STORAGE;",
                                                  "(void) finalize_retired_entries_locked();\n"
                                                  "                refusal = MOE_EXPERT_MOVE_SHARED_STORAGE;")])
    expect_fail("lock-before-mint", "A2", [("sycl", PREP_SIG,
                                            "            storage = ggml_sycl_device_expert_alloc(device, slot);",
                                            "            std::lock_guard early(g_tensor_inventory_mutex);\n"
                                            "            storage = ggml_sycl_device_expert_alloc(device, slot);")])
    expect_fail("lock-before-prepare", "A2", [("sycl", TXN_SIG, "        stop = ggml_sycl_expert_prepare_triplet(",
                                               "        std::lock_guard early(g_tensor_inventory_mutex);\n"
                                               "        stop = ggml_sycl_expert_prepare_triplet(")])
    expect_fail("transaction-mints", "A2", [("sycl", TXN_SIG, "    std::unique_lock lock(g_tensor_inventory_mutex);",
                                             "    ggml_sycl::mem_handle spare = ggml_sycl_host_expert_alloc(device, 1);\n"
                                             "    std::unique_lock lock(g_tensor_inventory_mutex);")])
    expect_fail("cache-takes-inventory", "A2", [("cache", ADMIT_SIG, "{\n    return copy_host_triplet_to_device(",
                                                 "{\n    std::lock_guard inv(g_tensor_inventory_mutex);\n"
                                                 "    return copy_host_triplet_to_device(")])
    expect_fail("no-lent-count-check", "A3", [("sycl", TXN_SIG,
                                               "if (named && g_sycl_expert_lent_count[device].load(",
                                               "if (false && g_sycl_expert_lent_count[device].load(")])
    expect_fail("recheck-refused-not-busy", "A3", [("sycl", TXN_SIG,
                                                    "        return refuse(ggml_sycl_txn_result::BUSY, \"loans "
                                                    "outstanding\");\n    }\n    if (!refill && !named) {",
                                                    "        return refuse(ggml_sycl_txn_result::REFUSED, \"loans "
                                                    "outstanding\");\n    }\n    if (!refill && !named) {")])
    expect_fail("minted-at-host-size", "A4", [("sycl", PREP_SIG, "ggml_sycl_device_expert_alloc(device, slot)",
                                               "ggml_sycl_device_expert_alloc(device, target.logical_bytes)")])
    expect_fail("minted-at-own-staged-size", "A4", [("sycl", PREP_SIG, "ggml_sycl_device_expert_alloc(device, slot)",
                                                     "ggml_sycl_device_expert_alloc(device, staged[r])")])
    expect_fail("slot-from-one-member", "A4", [("sycl", PREP_SIG, "ggml_sycl::expert_triplet_slot_bytes(staged)",
                                                "staged[0]")])
    expect_fail("host-copy-at-entry-size", "A4",
                [("sycl", PREP_SIG, "ggml_sycl_host_expert_alloc(device, target.logical_bytes)",
                  "ggml_sycl_host_expert_alloc(device, cache.loan_copy_bytes(target.key))")])
    expect_fail("transaction-sizes-a-copy", "A4", [("sycl", TXN_SIG, "        triplet.named = named ? named_place[t] : 0;",
                                                    "        triplet.named = named ? named_place[t] : 0;\n        (void) "
                                                    "staged_expert_bytes(GGML_LAYOUT_AOS, GGML_TYPE_COUNT, 0, 0);")])
    expect_fail("cache-checks-host-size", "A4", [("cache", COPY_SIG, "if (device_size[i] != slot || slot < logical)",
                                                  "if (device_size[i] != e.size || e.size < logical)")])
    expect_fail("refill-staged-at-host-copy-size", "A4", [("cache", COPY_SIG,
                                                           "if (device_size[i] != slot || slot < logical)",
                                                           "if (device_size[i] != (admit ? slot : e.size) || "
                                                           "slot < logical)")])
    expect_fail("cache-checks-own-staged-size", "A4", [("cache", COPY_SIG, "if (device_size[i] != slot ||",
                                                        "if (device_size[i] != staged[i] ||")])
    expect_fail("refill-copies-the-host-copy-whole", "A4", [("cache", COPY_SIG,
                                                             "const size_t logical = members[i].logical_bytes;",
                                                             "const size_t logical = admit ? members[i].logical_bytes "
                                                             ": e.size;")])
    expect_fail("logical-not-resolved", "A4", [("sycl", RESOLVE_SIG, "    out.logical_bytes = tensor->nb[2];\n", "")])
    expect_fail("staged-from-plan-size", "A4", [("sycl", PREP_SIG, "target.ncols, target.logical_bytes)",
                                                 "target.ncols, e->dst_size)")])
    expect_fail("logical-not-handed-to-cache", "A4",
                [("sycl", PREP_SIG, "        m.logical_bytes                   = target.logical_bytes;\n", "")])
    expect_fail("cache-stages-host-copy-size", "A4", [("cache", COPY_SIG, "members[i].ncols, members[i].logical_bytes)",
                                                       "members[i].ncols, e.size)")])
    expect_fail("shorter-host-copy-admitted", "A4", [("cache", COPY_SIG, "if (e.size < logical) {", "if (false) {")])
    expect_fail("host-copy-size-retried", "A4", [("sycl", FINAL_SIG, '"host copy size",', "")])
    expect_fail("copies-the-host-copy-whole", "A5", [("cache", COPY_SIG, "bytes[i]       = logical;",
                                                      "bytes[i]       = e.size;")])
    expect_fail("helper-ignores-layout", "A4", [("pad", STAGED_SIG, "layout != GGML_LAYOUT_AOS || ", "")])
    expect_fail("helper-computed-for-iq4-nl", "A4", [("pad", STAGED_SIG, "    return aos_staged_weight_bytes(",
                                                      "    if (type == GGML_TYPE_IQ4_NL) {\n        return expert_bytes "
                                                      "+ 216;\n    }\n    return aos_staged_weight_bytes(")])
    expect_fail("width-not-resolved", "A4", [("sycl", RESOLVE_SIG, "    out.ncols         = tensor->ne[0];\n", "")])
    expect_fail("entry-sized-at-host-bytes", "A4", [("cache", COPY_SIG, "entry.size                  = device_size[i];",
                                                     "entry.size                  = bytes[i];")])
    expect_fail("fill-not-behind-copy", "A5", [("cache", COPY_SIG, "{ out.copy_events[i] });", "{});")])
    expect_fail("entry-not-behind-fill", "A5", [("cache", COPY_SIG, "out.copy_events[i] = mem_fill_async(",
                                                 "(void) mem_fill_async(")])
    expect_fail("record-at-padded-size", "A5", [("sycl", TXN_SIG,
                                                 "/*logical_offset=*/0, triplet.members[r].logical_bytes);",
                                                 "/*logical_offset=*/0, record.handles[r].size());")])
    expect_fail("entry-runs-body-twice", "A6", [("sycl", ENTRY_SIG, "    if (out) {",
                                                 "    (void) ggml_sycl_expert_txn_run(ggml_sycl_expert_txn_direction::"
                                                 "NAMED, device, moves.size(), 0, false, resolver, &answer, false, "
                                                 "&moves);\n    if (out) {")])
    expect_fail("entry-catches-body", "A6", [("sycl", ENTRY_SIG, "    if (out) {",
                                               "    try {\n    } catch (...) {\n    }\n    if (out) {")])
    expect_fail("rethrow-before-publish", "A6", [("sycl", TXN_SIG, "    auto prepared_publication = ",
                                                   "    if (escaped) {\n        std::rethrow_exception(escaped);\n"
                                                   "    }\n    auto prepared_publication = ")])
    expect_fail("epoch-conditional", "A6", [("sycl", TXN_SIG, "        ggml_sycl_optional_layouts_retired();",
                                             "        if (!named) {\n            ggml_sycl_optional_layouts_retired();"
                                             "\n        }")])
    expect_fail("publishes-twice", "A6", [("sycl", TXN_SIG, "    lock.unlock();\n    warn_dropped();\n    if (!swapped) {",
                                           "    ggml_sycl_publish_prepared_plan_locked(prepared_publication);\n"
                                           "    lock.unlock();\n    warn_dropped();\n    if (!swapped) {")])
    expect_fail("named-reserves-ledger", "A7", [("sycl", TXN_SIG, "if (!refill && !named) {", "if (!refill) {")])
    expect_fail("named-appends-to-ledger", "A7", [("sycl", TXN_SIG,
                                                   "case ggml_sycl_expert_move_kind::EVICT:\n"
                                                   "                    (*named_moves)[",
                                                   "case ggml_sycl_expert_move_kind::EVICT:\n"
                                                   "                    ledger.triplets.push_back({ placed[0], placed[1], "
                                                   "placed[2] });\n                    (*named_moves)[")])
    expect_fail("evict-joins-the-loans-case", "A7", [("sycl", TXN_SIG,
                                                      "                case ggml_sycl_expert_move_kind::ADMIT:\n"
                                                      "                case ggml_sycl_expert_move_kind::EVICT:\n",
                                                      "                case ggml_sycl_expert_move_kind::ADMIT:\n")]
                + [("sycl", TXN_SIG, "                case ggml_sycl_expert_move_kind::LOAN:\n"
                                     "                    ledger.triplets.push_back(",
                    "                case ggml_sycl_expert_move_kind::LOAN:\n"
                    "                case ggml_sycl_expert_move_kind::EVICT:\n"
                    "                    ledger.triplets.push_back(")])
    expect_fail("named-refusal-dropped", "A7", [("sycl", TXN_SIG,
                                                 "if (triplet.kind == ggml_sycl_expert_move_kind::REFILL) {\n"
                                                 "                    transient",
                                                 "if (true) {\n                    transient")])
    expect_fail("tail-unguarded", "A7", [("sycl", TXN_SIG, "    if (!named) {\n        // An accepted loan",
                                          "    if (true) {\n        // An accepted loan")])
    expect_fail("refill-as-lax-as-admit", "A8", [("cache", REFILL_SIG, "(members, /*admit=*/false, out)",
                                                  "(members, /*admit=*/true, out)")])
    expect_fail("wider-host-owner-admitted", "A8", [("cache", COPY_SIG, "(admit ? md.size != e.size : md.size < e.size)",
                                                     "md.size < e.size")])
    expect_fail("refill-owner-made-exact", "A8", [("cache", COPY_SIG, "(admit ? md.size != e.size : md.size < e.size)",
                                                   "md.size != e.size")])
    expect_fail("unzoned-host-copy-admitted", "A8", [("cache", COPY_SIG,
                                                      " ||\n                (admit && md.host_zone == host_zone_id::COUNT)",
                                                      "")])
    expect_fail("named-refusal-warned", "A9",
                [("sycl", TXN_SIG, "!refill && !named && (warn_busy", "!refill && (warn_busy")])
    expect_fail("production-caller", "A9", [("sycl", "", ENTRY_SIG,
                                             "static void ggml_sycl_admit_everything(int d) {\n"
                                             "    (void) ggml_sycl_expert_move_transaction_run(d, {}, nullptr, nullptr);"
                                             "\n}\n\n" + ENTRY_SIG)])
    driver_call = "ggml_sycl_expert_move_transaction_run(device, request, nullptr, &result)"
    expect_fail("driver-calls-twice", "A9", [("sycl", DRIVER_SIG, driver_call + ";",
                                              driver_call + ";\n        (void) " + driver_call + ";")])
    expect_fail("driver-drops-its-call", "A9", [("sycl", DRIVER_SIG, driver_call, "ggml_sycl_txn_result::BUSY")])
    expect_fail("verdict-ignored", "A10", [("sycl", SELECT_SIG, "if (verdict) {", "if (false) {")])
    expect_fail("final-refusal-misspelled-in-cache", "A11", [("cache", COPY_SIG, 'refusal = "device storage not owned";',
                                                              'refusal = "device storage not own";')])
    expect_fail("final-refusal-misspelled-in-list", "A11", [("sycl", FINAL_SIG, '"device storage not owned",',
                                                             '"device storage not ownd",')])
    expect_fail("transient-refusal-misspelled", "A11", [("cache", COPY_SIG, 'out.reason = "copy submit failed";',
                                                         'out.reason = "copy submit faild";')])
    expect_fail("transient-refusal-made-final", "A11", [("sycl", FINAL_SIG, '"invalid arguments",',
                                                         '"invalid arguments", "entry changed",')])
    expect_fail("loan-refusal-unclassified", "A11", [("cache", LOAN_SIG, "refusal = MOE_EXPERT_MOVE_SHARED_STORAGE;",
                                                      'refusal = "shared storage held";')])
    expect_fail("loan-transient-made-final", "A11", [("sycl", FINAL_SIG, '"invalid arguments",',
                                                      '"invalid arguments", "storage still referenced",')])
    expect_fail("loan-final-dropped-from-the-list", "A11",
                [("sycl", FINAL_SIG, "ggml_sycl::MOE_EXPERT_MOVE_SHARED_STORAGE,", "")])
    expect_fail("live-host-entry-unclassified", "A11", [("sycl", FINAL_SIG, '"host entry exists"',
                                                         '"host entry exist"')])
    expect_fail("retiring-host-entry-made-final", "A11", [("sycl", FINAL_SIG, '"invalid arguments",',
                                                           '"invalid arguments", "host entry retiring",')])
    expect_fail("host-entries-answered-alike", "A11", [("cache", LOAN_SIG, 'host->second.retired ? "host entry '
                                                        'retiring" : "host entry exists"', '"host entry exists"')])
    expect_fail("admits-first", "E1", [("sycl", ENTRY_SIG, "moves[i].to_device = i >= n_evict;",
                                        "moves[i].to_device = i < request.admit.size();")])
    expect_fail("evicts-judged-as-admits", "E1", [("sycl", SELECT_SIG, "(m.to_device ? admit : evict).push_back(",
                                                   "admit.push_back(")])
    expect_fail("evict-kind-never-set", "E2", [("sycl", TXN_SIG, "? ggml_sycl_expert_move_kind::EVICT : kind;",
                                                "? ggml_sycl_expert_move_kind::ADMIT : kind;")])
    expect_fail("evict-copied-to-device", "E2", [("sycl", DIRECTION_SIG,
                                                  "        case ggml_sycl_expert_move_kind::EVICT:\n"
                                                  "            return false;",
                                                  "            return false;\n"
                                                  "        case ggml_sycl_expert_move_kind::EVICT:\n"
                                                  "            return true;")])
    expect_fail("direction-defaults", "E2", [("sycl", DIRECTION_SIG,
                                              "        case ggml_sycl_expert_move_kind::LOAN:\n"
                                              "        case ggml_sycl_expert_move_kind::EVICT:\n",
                                              "        default:\n")])
    expect_fail("move-statement-defaults", "E2", [("sycl", TXN_SIG,
                                                   "                case ggml_sycl_expert_move_kind::REFILL:\n"
                                                   "                    loaned = cache->refill_lent_triplet(",
                                                   "                default:\n"
                                                   "                    loaned = cache->refill_lent_triplet(")])
    expect_fail("evict-texts-missing", "E2", [("sycl", TEXTS_SIG, '"not resident", "exception while moving"',
                                               '"not on host", "exception while moving"')])
    expect_fail("evict-pool-full-refused", "E2", [("sycl", PREP_SIG,
                                                   "*transient = out.kind == ggml_sycl_expert_move_kind::EVICT;",
                                                   "*transient = false;")])
    expect_fail("evict-keeps-device-views", "E3", [("sycl", TXN_SIG,
                                                    "ggml_sycl_drop_lent_expert_views(extra, device, k.expert_id, "
                                                    "record.device_keys[r], lent_views);", "")])
    expect_fail("drop-keeps-generation", "E3", [("sycl", DROP_SIG, "++w.moe_expert_storage_generation;", "")])
    expect_fail("remember-keeps-generation", "E3", [("common", RECORD_SIG,
                                                     "records.push_back(std::move(record));\n"
                                                     "        ++moe_expert_storage_generation;",
                                                     "records.push_back(std::move(record));")])
    expect_fail("route-ignores-generation", "E3", [("sycl", ROUTE_SIG, "(stamp, replan_epoch, storage_generation, "
                                                    "selected_rows)) {",
                                                    "(stamp, replan_epoch, stamp.expert_storage_generation, "
                                                    "selected_rows)) {")])
    expect_fail("stamp-ignores-generation", "E3", [("hostpath", STAMP_SIG,
                                                    "s.expert_storage_generation == expert_storage_generation && ", "")])
    expect_fail("evict-refusal-ends-the-list", "E4", [("sycl", TXN_SIG, "if (!loaned && (admit || evict)) {",
                                                       "if (!loaned && admit) {")])
    expect_fail("evict-bytes-counted-as-admitted", "E5",
                [("sycl", ENTRY_SIG, "r.bytes_evicted   = answer.bytes_freed;",
                  "r.bytes_evicted   = answer.bytes_readmitted;")])
    expect_fail("body-sums-both-directions", "E5", [("sycl", TXN_SIG,
                                                     "(to_device ? taken : freed) += record.device_bytes;",
                                                     "freed += record.device_bytes;")])
    expect_fail("entry-sums-the-bytes-again", "E5", [("sycl", ENTRY_SIG,
                                                      "                ++(m.to_device ? r.n_admitted : r.n_evicted);",
                                                      "                ++(m.to_device ? r.n_admitted : r.n_evicted);\n"
                                                      "                (m.to_device ? r.bytes_admitted : "
                                                      "r.bytes_evicted) += m.device_bytes;")])
    expect_fail("named-sums-the-bytes-again", "E5", [("sycl", TXN_SIG, "                ++n_evicted;\n",
                                                      "                ++n_evicted;\n"
                                                      "                result.bytes_freed += m.device_bytes;\n")])

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
    raw = {k: p.read_text() for k, p in FILES.items()}
    if "--self-test" in argv:
        return self_test(raw)
    try:
        check(raw)
    except ContractError as e:
        print(e)
        return 1
    print("PASS: the named move waits for nothing, mints behind no lock, fills an admit's tail before it publishes "
          "once, moves an evict to the host with the generation bumped, and leaves the ledger alone")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
