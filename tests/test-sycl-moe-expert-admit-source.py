#!/usr/bin/env python3
"""Source contract: the named expert move's ADMIT direction (ggml-sycl.cpp, unified-cache.cpp, weight-row-padding.hpp).

Host-only: reads sources, builds nothing, loads no model and touches no device. The behaviour is the private-fixture
device cases in sycl-moe-expert-loan (lead-run); this gate pins what an edit could undo while those still pass, or what
they cannot see.

A boundary driver names expert triplets to admit: each moves from its pinned host copy to the device on its own, and
all that move are published in one plan. The entry ggml_sycl_expert_move_transaction_run runs the plan transaction's
one body (ggml_sycl_expert_txn_run) in the NAMED direction; the cache copies each triplet in
unified_cache::admit_host_triplet, the refill's body (copy_host_triplet_to_device) with the admit's source rule and
staged size. What this gate pins:

  A0  the bodies it reads exist, so no pin below passes on an empty string;
  A1  nothing on the admit's path waits or asks an event's status: no .wait(, wait_and_throw(, get_info<,
      event_complete(, command_execution_status, ext_oneapi_get_state, finalize_retired_entries_locked( or
      drain_retained_handles( in the entry, the transaction body, the triplet preparation, the named selection, the
      cache's shared body, its admit wrapper or the staged-size helper;
  A2  the storage is minted only in the triplet preparation, which the transaction runs before it takes the inventory
      lock and which never names it; the cache is asked to admit under that lock, and never takes it, so no
      unified-cache allocation runs under it;
  A3  a named move is refused BUSY "loans outstanding" while the device's lent count is not zero, before the
      live-update lease, and again under the inventory lock while the ledger holds a lent triplet;
  A4  an expert's size is its tensor's, never a cache entry's, in every direction: the resolver reads the tensor's
      bytes per expert (nb[2]) with its type and width, and the preparation hands all three to the cache. A move to
      the device mints its storage at the padded staged size of those bytes, asked of the layout
      (staged_expert_bytes), and a move to the host mints pinned storage for exactly those bytes; the preparation
      sizes by the kind's direction alone, and the transaction sizes nothing. Never the host copy's size and never
      computed for one type: the cache refuses a host copy shorter than the expert ("host copy size", a refusal a
      later boundary meets again) and storage that is not exactly the staged size or is smaller than the expert, for
      a refill as for an admit, and the device entry takes the storage's size;
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
  A9  a named move WARNs nothing (its refusals skip the loan's WARN, its result line is DEBUG), and nothing but the
      private test fixtures calls the entry;
  A10 the named selection takes the published plan's verdict (moe_expert_move_validate) and moves only the triplets it
      lets through;
  A11 every refusal the cache's shared body assigns is classed exactly once: final (in the transaction's final list,
      ggml_sycl_expert_move_refusal_is_final) or transient (in this pin's TRANSIENT_REFUSALS, each with the path
      that clears it), and each classified reason is one the body assigns. A refusal is final when the same inputs
      give it again with nothing changed, so a renamed or new refusal cannot be retried at every decode graph
      unclassified.

Checks run on comment-stripped, whitespace-normalized text. NOT VACUOUS: --self-test runs the gate on in-memory mutants
of the real sources. Each mutant names the pin that must catch it and is reported if it survives, if another pin
catches it first, or if its anchor no longer matches exactly once in the body it edits (NOT APPLIED). The self-test
also proves that last report fires. ctest registers it with --self-test.

WHAT THIS DOES NOT PROVE. It reads text. That the copy and the fill run in that order on the device, that the tail
reads back zero and that a kernel reads the admitted bytes through the direct route are the device cases' job.
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
    and character literals skipped while balancing; (-1, -1) when there is none."""
    at = text.find(signature)
    if at < 0:
        return -1, -1
    return block_span(text, text.find("{", at + len(signature)))


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

# --- needles (normalized) -------------------------------------------------------------------------------------------

EVENT_WAITS = (".wait(", "wait_and_throw(", "get_info<", "event_complete(", "command_execution_status",
               "ext_oneapi_get_state", "finalize_retired_entries_locked(", "drain_retained_handles(")

MINT_PREFIX = "= ggml_sycl_device_expert_alloc(device, "
MINT_CALLS = ("ggml_sycl_device_expert_alloc(", "ggml_sycl_host_expert_alloc(")
PREP_CALL = ("stop = ggml_sycl_expert_prepare_triplet(*cache, device, *current->plan, &keys[3 * t], *resolver, "
             "triplet, &transient);")
INVENTORY_LOCK = "std::unique_lock lock(g_tensor_inventory_mutex);"
ADMIT_PREFIX = "cache->admit_host_triplet("
ADMIT_CALL = ("loaned = !to_device ? cache->loan_expert_triplet(triplet.members, record) : admit ? "
              "cache->admit_host_triplet(triplet.members, record) : cache->refill_lent_triplet(triplet.members, record);")

LOANS_CHECK = ('if (named && g_sycl_expert_lent_count[device].load(std::memory_order_acquire) != 0) '
               '{ return refuse(ggml_sycl_txn_result::BUSY, "loans outstanding"); }')
LOANS_RECHECK = ('if (named && !ledger.triplets.empty()) { lock.unlock(); '
                 'return refuse(ggml_sycl_txn_result::BUSY, "loans outstanding"); }')
LEASE = "registry.acquire_live_update(current_token)"
LEDGER_LOCKED = "ggml_sycl_expert_lent_ledger & ledger = ggml_sycl_expert_lent_ledger_locked(device, *current);"

PREP_DIRECTION = "const bool to_device = ggml_sycl_expert_move_to_device(out.kind);"
STAGED_SIZE = ("const size_t staged = "
               "ggml_sycl::staged_expert_bytes(e->layout, target.type, target.ncols, target.logical_bytes);")
MINT = "storage = ggml_sycl_device_expert_alloc(device, staged);"
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
CACHE_STAGED = "const size_t staged = staged_expert_bytes(e.layout, members[i].type, members[i].ncols, logical);"
SIZE_GUARD = 'if (device_size[i] != staged || staged < logical) { refusal = "device storage size"; break; }'
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
                "ggml_sycl_optional_layouts_retired(); }")
RETHROW = "std::rethrow_exception(escaped);"
RETHROW_LAST = "if (escaped) { std::rethrow_exception(escaped); } return finish(ggml_sycl_txn_result::ACCEPTED, reason);"
NAMED_CALL = ("const ggml_sycl_txn_result code = ggml_sycl_expert_txn_run(ggml_sycl_expert_txn_direction::NAMED, "
              "device, moves.size(), 0, false, resolver, &answer, false, &moves);")

RESERVE_GUARD = "if (!refill && !named) { ledger.triplets.reserve("
MOVED_BRANCH = ("switch (triplet.kind) { case ggml_sycl_expert_move_kind::LOAN: ledger.triplets.push_back({ placed[0], "
                "placed[1], placed[2] }); ledger.triplet_bytes.push_back(ggml_sycl::moe_expert_triplet_dst_bytes("
                "*next_plan, placed)); break; case ggml_sycl_expert_move_kind::REFILL: ledger.triplets.pop_back(); "
                "ledger.triplet_bytes.pop_back(); break; case ggml_sycl_expert_move_kind::ADMIT: "
                "(*named_moves)[triplet.named].moved = true; break; }")
TAIL_GUARD = "if (!named) { if (!refill && lent != 0) { ledger.refill_armed = false; }"
TAIL_WRITES = ("ledger.refill_armed = false;", "g_sycl_expert_lent_count[device].store(",
               "g_sycl_expert_lent_bytes[device].store(", "g_sycl_expert_readmitted_total[device].fetch_add(",
               "g_sycl_expert_lent_total[device].fetch_add(", "result.n_remaining =")
CURSOR = "if (triplet.kind == ggml_sycl_expert_move_kind::REFILL && (!ledger.refill_armed ||"
NAMED_CACHE_REFUSAL = ("if (!loaned && admit) { refuse_named(triplet.named, "
                       "ggml_sycl_expert_move_refusal_is_final(record.reason) ? ggml_sycl_txn_result::REFUSED : "
                       "ggml_sycl_txn_result::BUSY, record.reason); continue; }")
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
NAMED_DEBUG = 'GGML_LOG_DEBUG("[MOE-EXEC] named move device %d: %u triplets, %zu MiB admitted (%s)\\n"'
ENTRY_NAME = "ggml_sycl_expert_move_transaction_run("

VALIDATE = "const ggml_sycl::moe_expert_move_check check = ggml_sycl::moe_expert_move_validate(plan, device, {}, admit);"

# The cache's refusals that clear by themselves, each with the path that clears it. Every other refusal the shared
# body assigns gives the same answer again with nothing changed, so it belongs in the transaction's final list.
TRANSIENT_REFUSALS = (
    "load effect unavailable",  # the load this thread is bound to commits or is abandoned
    "plan not coherent",  # the plan publication in flight completes
    "device entry exists",  # finalize_retired_entries_locked reaps the retired entry once it drains
    "copy submit failed",  # the runtime threw on a copy or fill submit
    "entry changed",  # the host copy changed between the check and the publish
    "publication failed",  # an allocation under the cache lock threw
)
SUCCESS_REASON = "OK"
_REASON_ASSIGN_RE = re.compile(r"\b(?:refusal|out\.reason) = ([^;]*);")
_LITERAL_RE = re.compile(r'"((?:\\.|[^"\\])*)"')
_FINAL_LIST_RE = re.compile(r"final_reasons\[\] = \{([^}]*)\}")
VERDICT = ("if (check.admit[i]) { moves[i].code = ggml_sycl_txn_result::REFUSED; moves[i].reason = check.admit[i]; "
           "continue; }")
SELECT_CALL = "named ? ggml_sycl_expert_named_select(*current->plan, device, *named_moves, named_place) :"


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
    sycl = norm(raw["sycl"])
    cache = norm(raw["cache"])
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

    # A0 ---------------------------------------------------------------------------------------------------------------
    named = {"transaction body": txn, "entry": entry, "named selection": select, "named stop": stop,
             "registered resolver": resolve, "cache shared body": copy, "cache admit": admit, "cache refill": refill,
             "staged-size helper": staged, "final refusal list": final, "triplet preparation": prep}
    for name, text in named.items():
        if not text:
            fail("A0", f"the {name} is missing, so every pin reading it would pass on nothing")

    # A1 ---------------------------------------------------------------------------------------------------------------
    for name in ("transaction body", "entry", "triplet preparation", "named selection", "named stop",
                 "cache shared body", "cache admit", "staged-size helper"):
        for token in EVENT_WAITS:
            if token in named[name]:
                fail("A1", f"the {name} names `{token}`: the admit's path never waits or asks an event's status")

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
    if not (prep.count(PREP_DIRECTION) == 1 and prep.count(STAGED_SIZE) == 1 and prep.count(MINT) == 1
            and ordered(prep, PREP_DIRECTION, STAGED_SIZE, MINT)):
        fail("A4", "a move to the device is not minted at the staged size, decided by the kind's direction")
    if prep.count(HOST_MINT) != 1 or "staged_expert_bytes(" in txn:
        fail("A4", "a move to the host is not minted at the tensor's bytes, or the transaction sizes a copy itself")
    if not (prep.count(MEMBER_TYPE) == 1 and prep.count(MEMBER_NCOLS) == 1 and RESOLVED_TYPE in resolve
            and RESOLVED_NCOLS in resolve):
        fail("A4", "the type and width the staged size is asked with are not the resolved tensor's")
    if not (prep.count(MEMBER_LOGICAL) == 1 and RESOLVED_LOGICAL in resolve):
        fail("A4", "the expert's bytes every move sizes, copies and records are not its tensor's nb[2]")
    if not (copy.count(CACHE_LOGICAL) == 1 and copy.count(HOST_SIZE_GUARD) == 1
            and ordered(copy, CACHE_LOGICAL, HOST_SIZE_GUARD, CACHE_STAGED) and '"host copy size"' in final):
        fail("A4", "the cache does not refuse, as final, a host copy shorter than the expert's tensor bytes")
    if not (STAGED_GUARD in staged and staged.count(STAGED_AOS) == 1):
        fail("A4", "staged_expert_bytes does not ask the layout, or does not give AOS the padded size")
    if not (copy.count(CACHE_STAGED) == 1 and copy.count(SIZE_GUARD) == 1 and ordered(copy, CACHE_STAGED, SIZE_GUARD)):
        fail("A4", "the cache does not refuse admit storage that is not exactly the staged size")
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
    if not (txn.count(REMEMBER) == 1 and ordered(txn, ADMIT_CALL, REMEMBER)):
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
    if txn.count(DROP_GUARD) != 1 or not ordered(txn, ADMIT_CALL, NAMED_CACHE_REFUSAL, DROP_GUARD):
        fail("A7", "a cache refusal of a named triplet can reach the ledger drop")

    # A8 ---------------------------------------------------------------------------------------------------------------
    if REFILL_WRAP not in refill or ADMIT_WRAP not in admit:
        fail("A8", "the refill and the admit do not run the shared body with false and true")
    if copy.count(NOT_THERE) != 1 or copy.count(PINNED_CHECK) != 1:
        fail("A8", "the admit's host copy is not required to be exactly its whole pinned storage in a host zone, or the "
                   "refill's owner bound is not the loan's")

    # A9 ---------------------------------------------------------------------------------------------------------------
    if txn.count(REFUSE_GATE) != 1 or txn.count(NAMED_DEBUG) != 1:
        fail("A9", "a named move's refusal or result is logged above DEBUG")
    private = _private_testing_lines(raw["sycl"])
    lines = strip_comments(raw["sycl"]).split("\n")
    for k, line in enumerate(lines):
        if ENTRY_NAME in line and ENTRY_SIG.rstrip("(") not in re.sub(r"\s+", " ", line):
            if not private[k]:
                fail("A9", f"line {k + 1} calls the named move outside the private test fixtures")

    # A10 --------------------------------------------------------------------------------------------------------------
    if select.count(VALIDATE) != 1 or select.count(VERDICT) != 1 or txn.count(SELECT_CALL) != 1:
        fail("A10", "the named selection does not take the published plan's verdict on each triplet")

    # A11 --------------------------------------------------------------------------------------------------------------
    assigned = [lit for rhs in _REASON_ASSIGN_RE.findall(copy) for lit in _LITERAL_RE.findall(rhs)]
    listed = _FINAL_LIST_RE.search(final)
    finals = _LITERAL_RE.findall(listed.group(1)) if listed else []
    if not finals or assigned.count(SUCCESS_REASON) != 1:
        fail("A11", "the final refusal list or the cache body's reason assignments are not where this pin reads them")
    refusals = [r for r in assigned if r != SUCCESS_REASON]
    for reason in refusals:
        if reason in finals and reason in TRANSIENT_REFUSALS:
            fail("A11", f'the cache refusal "{reason}" is both final and transient')
        if reason not in finals and reason not in TRANSIENT_REFUSALS:
            fail("A11", f'the cache refusal "{reason}" is neither final nor transient, so a refill retries it at '
                        "every decode graph unclassified")
    for reason in list(finals) + list(TRANSIENT_REFUSALS):
        if reason not in refusals:
            fail("A11", f'"{reason}" is classified, but the cache body assigns no such refusal')


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

    expect_fail("entry-renamed", "A0", [("sycl", "", "ggml_sycl_txn_result ggml_sycl_expert_move_transaction_run(\n",
                                         "ggml_sycl_txn_result ggml_sycl_expert_move_run(\n")])
    expect_fail("fill-waited", "A1", [("cache", COPY_SIG, "bq,\n                                                    "
                                       "{ out.copy_events[i] });",
                                       "bq,\n                                                    "
                                       "{ out.copy_events[i] });\n                out.copy_events[i].wait();")])
    expect_fail("entry-asks-status", "A1", [("sycl", ENTRY_SIG, "    if (out) {",
                                             "    (void) sycl::event{}.get_info<sycl::info::event::"
                                             "command_execution_status>();\n    if (out) {")])
    expect_fail("lock-before-mint", "A2", [("sycl", PREP_SIG,
                                            "            storage = ggml_sycl_device_expert_alloc(device, staged);",
                                            "            std::lock_guard early(g_tensor_inventory_mutex);\n"
                                            "            storage = ggml_sycl_device_expert_alloc(device, staged);")])
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
    expect_fail("minted-at-host-size", "A4", [("sycl", PREP_SIG, "ggml_sycl_device_expert_alloc(device, staged)",
                                               "ggml_sycl_device_expert_alloc(device, target.logical_bytes)")])
    expect_fail("host-copy-at-entry-size", "A4",
                [("sycl", PREP_SIG, "ggml_sycl_host_expert_alloc(device, target.logical_bytes)",
                  "ggml_sycl_host_expert_alloc(device, cache.loan_copy_bytes(target.key))")])
    expect_fail("transaction-sizes-a-copy", "A4", [("sycl", TXN_SIG, "        triplet.kind  = kind;",
                                                    "        triplet.kind  = kind;\n        (void) staged_expert_bytes("
                                                    "GGML_LAYOUT_AOS, GGML_TYPE_COUNT, 0, 0);")])
    expect_fail("cache-checks-host-size", "A4", [("cache", COPY_SIG,
                                                  "staged_expert_bytes(e.layout, members[i].type, members[i].ncols, "
                                                  "logical)",
                                                  "e.size")])
    expect_fail("refill-staged-at-host-copy-size", "A4", [("cache", COPY_SIG,
                                                           "staged_expert_bytes(e.layout, members[i].type, "
                                                           "members[i].ncols, logical)",
                                                           "(admit ? staged_expert_bytes(e.layout, members[i].type, "
                                                           "members[i].ncols, logical) : e.size)")])
    expect_fail("refill-copies-the-host-copy-whole", "A4", [("cache", COPY_SIG,
                                                             "const size_t logical = members[i].logical_bytes;",
                                                             "const size_t logical = admit ? members[i].logical_bytes "
                                                             ": e.size;")])
    expect_fail("logical-not-resolved", "A4", [("sycl", RESOLVE_SIG, "    out.logical_bytes = tensor->nb[2];\n", "")])
    expect_fail("staged-from-plan-size", "A4", [("sycl", PREP_SIG, "target.ncols, target.logical_bytes)",
                                                 "target.ncols, e->dst_size)")])
    expect_fail("logical-not-handed-to-cache", "A4",
                [("sycl", PREP_SIG, "        m.logical_bytes                   = target.logical_bytes;\n", "")])
    expect_fail("cache-stages-host-copy-size", "A4", [("cache", COPY_SIG, "members[i].ncols, logical)",
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
                                                   "case ggml_sycl_expert_move_kind::ADMIT:\n"
                                                   "                    (*named_moves)[",
                                                   "case ggml_sycl_expert_move_kind::ADMIT:\n"
                                                   "                    ledger.triplets.push_back({ placed[0], placed[1], "
                                                   "placed[2] });\n                    (*named_moves)[")])
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
    expect_fail("verdict-ignored", "A10", [("sycl", SELECT_SIG, "if (check.admit[i]) {", "if (false) {")])
    expect_fail("final-refusal-misspelled-in-cache", "A11", [("cache", COPY_SIG, 'refusal = "device storage not owned";',
                                                              'refusal = "device storage not own";')])
    expect_fail("final-refusal-misspelled-in-list", "A11", [("sycl", FINAL_SIG, '"device storage not owned",',
                                                             '"device storage not ownd",')])
    expect_fail("transient-refusal-misspelled", "A11", [("cache", COPY_SIG, 'out.reason = "copy submit failed";',
                                                         'out.reason = "copy submit faild";')])
    expect_fail("transient-refusal-made-final", "A11", [("sycl", FINAL_SIG, '"invalid arguments",',
                                                         '"invalid arguments", "entry changed",')])

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
    print("PASS: the named move's admit waits for nothing, mints the staged size behind no lock, fills the tail before "
          "it publishes once, and leaves the ledger alone")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
