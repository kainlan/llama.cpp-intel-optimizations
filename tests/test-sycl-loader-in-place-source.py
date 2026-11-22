#!/usr/bin/env python3
"""Source contract: on a load that does not map files, the loader's SYCL_Host buffer is a charged host term, and
S1-PRELOAD serves a host-planned expert from it in place under the buffer's owner, which the model owns.

The runtime halves are the host-only test sycl-loader-in-place-terms (the WEIGHT zone's terms over a hand-built plan)
and the lead-run device tests and gates. This gate pins what those cannot see: who owns a loader chunk, where its
leak check runs on every teardown path, the preload's in-place gate, and the refusal when a host-planned expert would
be registered without an owner. Each clause is a function over the comment-stripped tree (one clause reads the raw
text) that returns its problems; `--self-test` runs each witness mutant below in memory and requires its clause to
report it, and the unmodified tree to pass.

Ownership and teardown (gs = ggml-sycl.cpp)
  O1. A SYCL_Host buffer allocated inside a model load is owned by that model: ggml_sycl_host_buft_alloc mints its row
      with the owner {MODEL, model id} from ggml_sycl_loading_model_id(), which takes the id from the load effect the
      host-weight registration takes, and records the chunk with ggml_sycl_model_host_chunk_minted. Such a buffer
      keeps no BUFFER owner (owner_id 0, so free_buffer runs no BUFFER check), and the "Accepted deviation" note at
      the intent builder is gone (raw text).
  O2. The MODEL check of a model's slot release runs after unified_cache_release_model_slot: the entries that serve
      host experts in place hold the chunk until the reclaim, so a row is a leak only once it outlives the reclaim.
      The release forgets the model's chunk record after the check.
  O3. A load that rolls back keeps its leak check: free_buffer reports each freed chunk
      (ggml_sycl_model_host_chunk_freed), abort_owner_effects marks the rollback after its own
      unified_cache_release_model_slot (ggml_sycl_model_host_chunks_rolled_back), and the model's MODEL check runs
      once both have happened (ggml_sycl_model_host_chunks_check).

The preload
  P1. register_host_aos_expert serves a host-planned PINNED expert in place when the plan's loader serves host experts,
      through ggml_sycl_loader_buffer_owner, which accepts only a SYCL_Host chunk (its free_buffer) with no claim,
      lifetime MODEL, a chunk a model's load minted (model_id != 0), the expert inside the chunk, and an owner
      moe_stream_owner_streamable() accepts for the device.
  P2. No silent degrade: a host expert with no owner is counted, never registered, and the load refuses after the
      sweep with one ERROR line naming the count and the bytes, and a runtime_error.
  P4. No copy fallback: when the plan serves an expert from the loader's chunk and the gate refuses it, the plan sized
      no block for a copy, so the expert is counted with the gate's reason and returns before any room is asked
      (ggml_sycl_host_expert_alloc), and the load refuses after the sweep with one ERROR line naming the count, the
      bytes and the gate's reason, and a runtime_error.
  P3. The plan's loader term is checked against what the load holds: when one pool holds the loader's buffer
      (loader_device >= 0), a [HOST-PLAN-BUG] WARN when the model's live SYCL_Host bytes on loader_device exceed
      loader_residency.

The charge (ucc = unified-cache.cpp)
  C1. placement_plan_host_weight_zone_terms reads the loader facts, charges loader_residency only when one pool holds
      the buffer (loader_holds_weights && loader_device >= 0), sums it at TENSOR_ALIGNMENT and reports loader_served;
      populate_host_zone_sizing compares through matches_charge(plan.weight_host_bytes) and records the term as
      host_zone_loader_bytes. Each card's pool commits its own share: the zone configuration takes the card from its
      caller (int device), sizes the zones and checks its growth through
      placement_plan_host_zone_commit_terms_on_device(..., device), and names a card that leaves the term out on a
      WARN line.
  C2. The fact has one source: placement_plan_copy_host_backing_fact sets the loader facts from
      placement_loader_fact_for(info.load_maps_files, info.load_weights_evictable, info.load_loader_device, <builder's
      device>), and the kv_info fill sets load_weights_evictable from ggml_backend_sycl_weights_evictable() and
      load_loader_device from the scheduler's device count: device 0 when it sees one device, -1 otherwise.
  C3. The loader's buffers are charged as the pinned pool holds them (hpp = unified-cache.hpp), so once the plan
      admits, no loader pool growth fails (with C4's refusal): the commit terms size the zone with
      placement_loader_pool_footprint_for over plan.loader_tensor_bytes, at the buffer type's maximum size
      (host_zone_buffer_max_size of plan.loader_buffer_cap and the pool's chunk) and the pool's chunk: the WEIGHT
      zone is the record less host_zone_loader_bytes plus the footprint's zone bytes (loader_zone), and the chunks
      the pool grows for an oversize buffer are loader_grown, which total() counts. A card that does not hold the
      buffer sheds both. populate_host_zone_sizing records the tensors from its inventory, the one writer; the fact
      copy copies the cap from info.load_loader_buffer_cap, which the kv_info fill takes from the buffer type's chunk
      cap; and ggml_sycl_host_buft_max_size answers through host_zone_buffer_max_size, so the plan and the buffer
      type share one rule.
  C4. Once the plan admits, no loader pool growth fails where the pool's own budget is at least the plan's host
      budget. placement_plan_loader_pool_refusal (hpp) names
      "loader-pool-over-budget" for a plan charged the loader term (loader_holds_weights, loader_device >= 0,
      host_zone_loader_bytes > 0) whose commit total, loader_grown included (host_zone_terms.total()), exceeds
      host_budget; ggml_sycl_stage_refuse_loader_pool_plan logs it on one ERROR line, and the stage returns
      GGML_SYCL_LIFECYCLE_PLAN_REJECTED on it in both passes, for each late decision and for the early candidate. A
      plan without the term keeps the ledger's WARN-only check.

Run: python3 -I tests/test-sycl-loader-in-place-source.py [--self-test]
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GS = "ggml/src/ggml-sycl/ggml-sycl.cpp"
UCC = "ggml/src/ggml-sycl/unified-cache.cpp"
HPP = "ggml/src/ggml-sycl/unified-cache.hpp"
FILES = (GS, UCC, HPP)
RAW = "raw:"  # the raw text of a file, for the one clause that reads a comment

_LEXEME_RE = re.compile(
    r'\b(?:u8|u|U|L)?R"(?P<delim>[^ ()\\\t\n]{0,16})\((?P<raw>.*?)\)(?P=delim)"'
    r'|"(?:\\.|[^"\\\n])*"' r"|'(?:\\.|[^'\\\n])*'" r"|//[^\n]*" r"|/\*.*?\*/",
    flags=re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if m.group("delim") is not None or tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def load_tree() -> dict:
    t = {}
    for rel in FILES:
        text = (ROOT / rel).read_text()
        t[rel] = strip_comments(text)
        t[RAW + rel] = text
    return t


def balanced(text: str, open_idx: int) -> str:
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i]
    return ""


def body_of(text: str, signature_re: str) -> str:
    """The brace body after the one match of signature_re, or "" when it matches none or several times."""
    hits = list(re.finditer(signature_re, text))
    if len(hits) != 1:
        return ""
    brace = text.find("{", hits[0].end() - 1)
    return balanced(text, brace) if brace >= 0 else ""


def between(text: str, start: str, end: str) -> str:
    a = text.find(start)
    if a < 0:
        return ""
    b = text.find(end, a + len(start))
    return text[a:b] if b >= 0 else ""


def fn(t: dict, rel: str, name: str) -> str:
    return body_of(t[rel], rf"\bstatic\s+[\w:<>\s\*&]*?\b{name}\s*\([^;{{]*\)\s*(?:noexcept\s*)?\{{")


def ordered(text: str, first: str, second: str) -> bool:
    a = text.find(first)
    b = text.find(second)
    return a >= 0 and b >= 0 and a < b


MODEL_CHECK = "unified_cache_host_zone_owner_check(ggml_sycl::alloc_lifetime::MODEL"
RELEASE_SLOT = "ggml_sycl::unified_cache_release_model_slot("
REFUSAL_FMT = '"[S1-PRELOAD] %zu host-planned experts (%.1f MB) have no owner'
REFUSAL_THROW = 'throw std::runtime_error("S1 preload planned host experts have no owner");'
LOADER_BUG_FMT = "\"[HOST-PLAN-BUG] device=%d the loader's SYCL_Host buffer holds %zu B"
LOADER_DROP_FMT = "\"[HOST-PLAN] device=%d WEIGHT zone: loader_residency=0: the loader's SYCL_Host buffer"
GATE_REFUSAL_FMT = "\"[S1-PRELOAD] %zu host experts (%.1f MB) the plan serves from the loader's SYCL_Host buffer"
GATE_REFUSAL_THROW = 'throw std::runtime_error("S1 preload in-place host experts were refused by the loader buffer gate");'
GATE_REFUSAL_BLOCK = r"if\s*\(\s*loader_serves\s*&&\s*!\s*loader_in_place\s*\)\s*\{"


# ---- ownership and teardown ----


def clause_o1_model_owns_its_chunks(t: dict) -> list:
    problems = []
    alloc = fn(t, GS, "ggml_sycl_host_buft_alloc")
    if not alloc:
        return ["ggml_sycl_host_buft_alloc not found"]
    if not re.search(r"\bggml_sycl_loading_model_id\s*\(\s*\)", alloc):
        problems.append("the alloc does not take the loading model's id from ggml_sycl_loading_model_id()")
    if not re.search(r"alloc_owner_kind::MODEL\s*,\s*model_id\s*\}", alloc):
        problems.append("the alloc mints no {MODEL, model_id} owner")
    if "ggml_sycl_model_host_chunk_minted(" not in alloc:
        problems.append("the alloc does not record the model's chunk")
    if not re.search(r"owner_id\s*=\s*model_id\s*!=\s*0\s*\?\s*0\s*:", alloc):
        problems.append("a model-owned chunk keeps a BUFFER owner serial")
    model_id_fn = fn(t, GS, "ggml_sycl_loading_model_id")
    if "acquire_load_effect(" not in model_id_fn or "bound_candidate()" not in model_id_fn:
        problems.append("ggml_sycl_loading_model_id does not read the load effect of the bound candidate")
    if "Accepted deviation" in t[RAW + GS]:
        problems.append("the intent builder still calls the BUFFER owner of a model's chunk an accepted deviation")
    return problems


def clause_o2_model_check_after_reclaim(t: dict) -> list:
    body = fn(t, GS, "ggml_sycl_release_model_slot_resources")
    if not body:
        return ["ggml_sycl_release_model_slot_resources not found"]
    problems = []
    if body.count(MODEL_CHECK) != 0:
        problems.append("the slot release runs the MODEL check inline instead of through ggml_sycl_model_host_chunks_check")
    if not ordered(body, RELEASE_SLOT, "ggml_sycl_model_host_chunks_check("):
        problems.append("the MODEL check does not run after unified_cache_release_model_slot")
    if not ordered(body, "ggml_sycl_model_host_chunks_check(", "ggml_sycl_model_host_chunks_forget("):
        problems.append("the slot release does not forget the model's chunk record after its check")
    return problems


def clause_o3_rollback_keeps_its_check(t: dict) -> list:
    problems = []
    free_body = fn(t, GS, "ggml_backend_sycl_host_buffer_free_buffer")
    if not ordered(free_body, "ctx->buffer_handle = {};", "ggml_sycl_model_host_chunk_freed("):
        problems.append("free_buffer does not report a model's chunk after releasing its handle")
    abort_body = fn(t, GS, "ggml_sycl_abort_owner_effects_noexcept")
    if not ordered(abort_body, RELEASE_SLOT, "ggml_sycl_model_host_chunks_rolled_back("):
        problems.append("abort_owner_effects does not mark the rollback after its slot release")
    check = fn(t, GS, "ggml_sycl_model_host_chunks_check")
    if not re.search(re.escape(MODEL_CHECK) + r"\s*,\s*\{\s*ggml_sycl::alloc_owner_kind::MODEL\s*,", check):
        problems.append("ggml_sycl_model_host_chunks_check does not run the MODEL owner check")
    for name in ("ggml_sycl_model_host_chunk_freed", "ggml_sycl_model_host_chunks_rolled_back"):
        body = fn(t, GS, name)
        if not re.search(r"live\s*==\s*0", body) or "ggml_sycl_model_host_chunks_check(" not in body:
            problems.append(f"{name} does not run the check once the last chunk of a rolled-back load is gone")
    return problems


# ---- the preload ----


def register_lambda(t: dict) -> str:
    return between(t[GS], "auto register_host_aos_expert = ", "bool any_cached = false;")


def clause_p1_in_place_gate(t: dict) -> list:
    problems = []
    lam = register_lambda(t)
    if not lam:
        return ["register_host_aos_expert not found"]
    if "loader_serves_host_experts" not in lam:
        problems.append("the preload does not ask the plan whether its loader serves host experts")
    if "ggml_sycl_loader_buffer_owner(" not in lam:
        problems.append("the preload does not take the loader chunk's owner for an in-place expert")
    if "ggml_sycl_mapped_buffer_owner(" not in lam:
        problems.append("the preload lost the mapped in-place path")
    owner = fn(t, GS, "ggml_sycl_loader_buffer_owner")
    for needle, why in (
        ("ggml_backend_sycl_host_buffer_free_buffer", "the SYCL_Host identity"),
        ("ctx->claim", "the claim refusal"),
        ("alloc_lifetime::MODEL", "the lifetime check"),
        ("ctx->model_id == 0", "model_id check"),
        ("moe_stream_owner_streamable(", "the streamable owner check"),
        ("ctx->size", "the bounds check"),
    ):
        if needle not in owner:
            problems.append(f"ggml_sycl_loader_buffer_owner has no {why}")
    return problems


def clause_p2_no_silent_degrade(t: dict) -> list:
    problems = []
    lam = register_lambda(t)
    if re.search(r"host_owner\s*=\s*\{\s*\}\s*;", lam):
        problems.append("the preload still clears an expert's owner and registers it unowned")
    if "host_expert_unowned_count" not in lam:
        problems.append("the preload does not count a host expert left with no owner")
    body = fn(t, GS, "ggml_sycl_preload_model_weights")
    if body.count(REFUSAL_FMT) != 1:
        problems.append("the preload has no single refusal line for host experts with no owner")
    if not ordered(body, REFUSAL_FMT, REFUSAL_THROW):
        problems.append("the refusal line is not followed by the load's refusal")
    return problems


def clause_p4_gate_refusal_refuses_the_load(t: dict) -> list:
    problems = []
    lam = register_lambda(t)
    if not lam:
        return ["register_host_aos_expert not found"]
    hit = re.search(GATE_REFUSAL_BLOCK, lam)
    if not hit:
        problems.append("the preload has no branch for an in-place expert the loader gate refused")
    else:
        block = balanced(lam, hit.end() - 1)
        if "host_expert_loader_refused_count" not in block or "host_expert_loader_refused_reason" not in block:
            problems.append("a refused in-place expert is not counted with the gate's reason")
        if not re.search(r"\breturn\s+false\s*;", block):
            problems.append("a refused in-place expert falls through to the copy")
        alloc = lam.find("ggml_sycl_host_expert_alloc(")
        if alloc >= 0 and alloc < hit.start():
            problems.append("room is asked for an in-place expert before the gate's refusal returns")
    if "did not size" in lam:
        problems.append("a refused in-place expert is still copied into a block the plan did not size")
    body = fn(t, GS, "ggml_sycl_preload_model_weights")
    if body.count(GATE_REFUSAL_FMT) != 1:
        problems.append("the preload has no single refusal line for in-place experts the gate refused")
    if not ordered(body, GATE_REFUSAL_FMT, GATE_REFUSAL_THROW):
        problems.append("the gate's refusal line is not followed by the load's refusal")
    if "host_expert_loader_refused_reason[" not in body:
        problems.append("the gate's refusal line does not name the gate's reason")
    return problems


def clause_p3_loader_term_checked(t: dict) -> list:
    body = fn(t, GS, "ggml_sycl_preload_model_weights")
    problems = []
    if LOADER_BUG_FMT not in body:
        problems.append("the preload does not check the loader's SYCL_Host bytes against the plan")
    if "ggml_sycl_model_host_chunk_bytes(" not in body or ".loader_residency" not in body:
        problems.append("the check does not compare the model's live chunk bytes with loader_residency")
    # The term is charged whenever one pool holds the buffer, the multi-device plan's included, and that pool is
    # loader_device's, which need not be the plan's device.
    if not re.search(r"global_plan->loader_holds_weights\s*&&\s*global_plan->loader_device\s*>=\s*0", body):
        problems.append("the check does not run whenever the plan charges the loader term")
    if not re.search(r"ggml_sycl_model_host_chunk_bytes\s*\(\s*model_id\s*,\s*global_plan->loader_device\s*\)", body):
        problems.append("the check does not count the chunks in loader_device's pool")
    return problems


# ---- the charge ----


def clause_c1_terms(t: dict) -> list:
    problems = []
    terms = body_of(t[UCC], r"\bhost_weight_zone_terms\s+placement_plan_host_weight_zone_terms\s*\(")
    for needle in ("plan.loader_holds_weights", "plan.loader_serves_host_experts", "TENSOR_ALIGNMENT",
                   "terms.loader_residency", "terms.loader_served"):
        if needle not in terms:
            problems.append(f"placement_plan_host_weight_zone_terms does not use {needle}")
    # A buffer split across the scheduler's devices is in no one pool, so no zone can commit it.
    if not re.search(r"plan\.loader_holds_weights\s*&&\s*plan\.loader_device\s*>=\s*0", terms):
        problems.append("placement_plan_host_weight_zone_terms charges the loader term when no one pool holds it")
    populate = fn(t, UCC, "populate_host_zone_sizing")
    if not re.search(r"!\s*weight_terms\.matches_charge\s*\(\s*plan\.weight_host_bytes\s*\)", populate):
        problems.append("populate_host_zone_sizing does not compare the terms through matches_charge")
    if not re.search(r"\bplan\.host_zone_loader_bytes\s*=\s*weight_terms\.loader_residency\s*;", populate):
        problems.append("populate_host_zone_sizing does not record the loader term beside the WEIGHT zone")
    writes = re.findall(r"(?:\.|->)host_zone_loader_bytes\s*=(?!=)", t[UCC] + t[GS])
    if len(writes) != 1:
        problems.append(f"host_zone_loader_bytes is written {len(writes)} time(s), want 1")
    # Every card's zones are configured from the one plan, and only loader_device's pool holds the buffer.
    configure = body_of(t[GS], r"\bstatic\s+void\s+ggml_sycl_configure_host_zones_for_plan\s*\([^;{]*\bint\s+device\s*,"
                               r"[^;{]*plan_owner\s*\)\s*\{")
    if not configure:
        return problems + ["the plan-owner overload of ggml_sycl_configure_host_zones_for_plan, taking the card from "
                           "its caller, not found"]
    per_card = re.findall(r"placement_plan_host_zone_commit_terms_on_device\s*\(\s*plan\s*,[^;]*?,\s*device\s*\)",
                          configure)
    if len(per_card) != 2:
        problems.append(f"the zone configuration takes {len(per_card)} of its two figures (the zones it sizes and the "
                        "growth it checks) as this card's share, want 2")
    if not re.search(r"host_zone_commit_check\s*\(\s*device\s*,\s*ggml_sycl::"
                     r"placement_plan_host_zone_commit_terms_on_device\s*\(\s*plan\s*,\s*plan\.host_zone_terms\s*,",
                     configure):
        problems.append("the commit check does not compare this card's growth with its share of the plan's record")
    if configure.count(LOADER_DROP_FMT) != 2:
        problems.append("the zone configuration does not name a card that leaves the loader term out, whether the "
                        "buffer is in another card's pool or split across the cards")
    return problems


def clause_c2_one_fact(t: dict) -> list:
    problems = []
    copy = fn(t, UCC, "placement_plan_copy_host_backing_fact")
    if not re.search(r"placement_loader_fact_for\s*\(\s*info\.load_maps_files\s*,\s*info\.load_weights_evictable\s*,"
                     r"\s*info\.load_loader_device\s*,", copy):
        problems.append("the fact copy does not derive the loader fact from the load's inputs")
    for field in ("plan.loader_holds_weights", "plan.loader_device", "plan.loader_serves_host_experts"):
        if field not in copy:
            problems.append(f"the fact copy does not set {field}")
    if not re.search(r"\bkv\.load_weights_evictable\s*=\s*ggml_backend_sycl_weights_evictable\s*\(\s*\)\s*;", t[GS]):
        problems.append("the kv_info fill does not take load_weights_evictable from the backend")
    # The scheduler sees one device when it collapses several to device 0, and then libllama binds that device's
    # SYCL_Host buffer type for every layer.
    if not re.search(r"\bkv\.load_loader_device\s*=\s*ggml_backend_sycl_get_device_count\s*\(\s*\)\s*==\s*1\s*\?\s*0\s*:"
                     r"\s*-1\s*;", t[GS]):
        problems.append("the kv_info fill does not take load_loader_device from the scheduler's device count")
    for rel in (GS, UCC):
        # A member write: the [HOST-PLAN] line's "loader_device=%d" is a string, not a write.
        writes = re.findall(r"(?:\.|->)loader_(?:holds_weights|device|serves_host_experts)\s*=(?!=)", t[rel])
        allowed = 3 if rel == UCC else 0
        if len(writes) != allowed:
            problems.append(f"{rel} writes a loader fact {len(writes)} time(s), want {allowed}")
    return problems


def clause_c3_pool_footprint(t: dict) -> list:
    problems = []
    terms = body_of(t[HPP], r"\binline\s+host_zone_commit_terms\s+placement_plan_host_zone_commit_terms\s*\(")
    if not terms:
        return ["placement_plan_host_zone_commit_terms not found in unified-cache.hpp"]
    if not re.search(r"\bplacement_loader_pool_footprint_for\s*\(\s*plan\.loader_tensor_bytes\s*,\s*"
                     r"host_zone_buffer_max_size\s*\(\s*plan\.loader_buffer_cap\s*,\s*chunk_bytes\s*\)\s*,\s*"
                     r"chunk_bytes\s*\)", terms):
        problems.append("the commit terms do not replay the loader's buffers at the buffer type's maximum size")
    if not re.search(r"\bterms\.loader_zone\s*=\s*\w+\.zone\s*;", terms):
        problems.append("the commit terms do not take loader_zone from the footprint")
    if not re.search(r"\bterms\.loader_grown\s*=\s*\w+\.grown\s*;", terms):
        problems.append("the commit terms do not take loader_grown from the footprint")
    if not re.search(r"\bterms\.weight\s*=\s*plan\.host_zone_weight_bytes\s*-\s*plan\.host_zone_loader_bytes\s*\+\s*"
                     r"terms\.loader_zone\s*;", terms):
        problems.append("the WEIGHT zone does not hold the loader's zone bytes in place of its raw bytes")
    if not re.search(r"size_t\s+total\s*\(\s*\)\s*const\s*\{\s*return\s+zones\s*\(\s*\)\s*\+\s*runtime\s*\+\s*"
                     r"rounding\s*\+\s*loader_grown\s*;", t[HPP]):
        problems.append("host_zone_commit_terms::total() does not count the chunks the loader grows")
    share = body_of(t[HPP], r"\binline\s+host_zone_commit_terms\s+placement_plan_host_zone_commit_terms_on_device\s*\(")
    if not re.search(r"\bon_device\.weight\s*-=\s*std::min\s*\(\s*on_device\.weight\s*,\s*terms\.loader_zone\s*\)", share):
        problems.append("a card that does not hold the loader's buffer does not shed its zone bytes")
    for field in ("loader_zone", "loader_grown"):
        if not re.search(rf"\bon_device\.{field}\s*=\s*0\s*;", share):
            problems.append(f"a card that does not hold the loader's buffer keeps its {field}")
    populate = fn(t, UCC, "populate_host_zone_sizing")
    if not re.search(r"\bplan\.loader_tensor_bytes\.push_back\s*\(", populate) or "tensor_inventory" not in populate:
        problems.append("populate_host_zone_sizing does not record the loader's tensors from its inventory")
    writers = re.findall(r"(?:\.|->)loader_tensor_bytes\s*(?:=(?!=)|\.push_back|\.clear|\.assign)", t[UCC] + t[GS])
    if len(writers) != len(re.findall(r"(?:\.|->)loader_tensor_bytes\s*(?:=(?!=)|\.push_back|\.clear|\.assign)",
                                      populate)):
        problems.append("the loader's tensors are written outside populate_host_zone_sizing")
    copy = fn(t, UCC, "placement_plan_copy_host_backing_fact")
    if not re.search(r"\bplan\.loader_buffer_cap\s*=\s*info\.load_loader_buffer_cap\s*;", copy):
        problems.append("the fact copy does not copy the buffer type's chunk cap")
    if not re.search(r"\bkv\.load_loader_buffer_cap\s*=\s*ggml_backend_sycl_host_buffer_type_chunk_cap\s*\(\s*\)\s*;",
                     t[GS]):
        problems.append("the kv_info fill does not take the loader's buffer cap from the buffer type")
    max_size = fn(t, GS, "ggml_sycl_host_buft_max_size")
    if not re.search(r"\bggml_sycl::host_zone_buffer_max_size\s*\(\s*chunk_cap\s*,", max_size) or \
            "CHUNK_SIZE" in max_size:
        problems.append("ggml_sycl_host_buft_max_size states its own rule, not host_zone_buffer_max_size")
    return problems


STAGE_SIG = r"\bggml_sycl_lifecycle_result\s+ggml_backend_sycl_stage_load_inventory_plan\s*\([^)]*\)\s*\{"
LOADER_POOL_LATE = (r"if\s*\(\s*ggml_sycl_stage_refuse_loader_pool_plan\s*\(\s*decision\.snapshot->plan\.get\s*\(\s*\)\s*,"
                    r"\s*early\s*\)\s*\)\s*\{\s*return\s+GGML_SYCL_LIFECYCLE_PLAN_REJECTED\s*;")
LOADER_POOL_EARLY = (r"if\s*\(\s*early_candidate\s*&&\s*ggml_sycl_stage_refuse_loader_pool_plan\s*\(\s*early_candidate->plan"
                     r"\.get\s*\(\s*\)\s*,\s*early\s*\)\s*\)\s*\{\s*return\s+GGML_SYCL_LIFECYCLE_PLAN_REJECTED\s*;")


def clause_c4_loader_pool_refusal(t: dict) -> list:
    problems = []
    rule = body_of(t[HPP], r"\binline\s+std::string\s+placement_plan_loader_pool_refusal\s*\(")
    if not rule:
        return ["placement_plan_loader_pool_refusal not found in unified-cache.hpp"]
    for what, pattern in (
            ("the loader holding the weights", r"\bplan\.loader_holds_weights\b"),
            ("a loader_device", r"\bplan\.loader_device\s*>=\s*0"),
            ("a charged loader term", r"\bplan\.host_zone_loader_bytes\s*>\s*0"),
            ("the commit total, loader_grown included, against the budget",
             r"\bplan\.host_zone_terms\.total\s*\(\s*\)\s*>\s*plan\.host_budget\b"),
            ("the reason's name", r'"loader-pool-over-budget')):
        if not re.search(pattern, rule):
            problems.append(f"placement_plan_loader_pool_refusal does not test {what}")
    helper = fn(t, GS, "ggml_sycl_stage_refuse_loader_pool_plan")
    if not re.search(r"\bggml_sycl::placement_plan_loader_pool_refusal\s*\(\s*\*plan\s*\)", helper):
        problems.append("ggml_sycl_stage_refuse_loader_pool_plan does not ask placement_plan_loader_pool_refusal")
    if helper.count("GGML_LOG_") != 1 or "GGML_LOG_ERROR" not in helper:
        problems.append("ggml_sycl_stage_refuse_loader_pool_plan does not log the refusal on one ERROR line")
    stage = body_of(t[GS], STAGE_SIG)
    if not re.search(LOADER_POOL_LATE, stage):
        problems.append("the late pass does not refuse a decision's loader pool over the budget")
    if not re.search(LOADER_POOL_EARLY, stage):
        problems.append("the early pass does not refuse the candidate's loader pool over the budget")
    return problems


CLAUSES = [
    ("O1 a model owns the SYCL_Host chunks its load mints", clause_o1_model_owns_its_chunks),
    ("O2 the MODEL check runs after the reclaim", clause_o2_model_check_after_reclaim),
    ("O3 a rolled-back load keeps its leak check", clause_o3_rollback_keeps_its_check),
    ("P1 the preload serves a host expert from the loader's chunk", clause_p1_in_place_gate),
    ("P2 a host expert with no owner refuses the load", clause_p2_no_silent_degrade),
    ("P3 the loader term is checked against the load", clause_p3_loader_term_checked),
    ("P4 an in-place expert the gate refuses refuses the load", clause_p4_gate_refusal_refuses_the_load),
    ("C1 the WEIGHT zone charges the loader's buffer once, in its pool", clause_c1_terms),
    ("C2 the loader fact has one source", clause_c2_one_fact),
    ("C3 the loader's buffers are charged as the pool holds them", clause_c3_pool_footprint),
    ("C4 a loader pool over the budget refuses the load by name", clause_c4_loader_pool_refusal),
]


def run(t: dict) -> dict:
    return {label: clause(t) for label, clause in CLAUSES}


def mutate(rel: str, old: str, new: str, raw: bool = False):
    def build(t: dict) -> dict:
        key = RAW + rel if raw else rel
        if old not in t[key]:
            raise LookupError(f"{old!r} not in {key}")
        m = dict(t)
        m[key] = t[key].replace(old, new, 1)
        return m

    return build


def mutate_re(rel: str, pattern: str, new: str):
    def build(t: dict) -> dict:
        text, n = re.subn(pattern, new, t[rel], count=1, flags=re.DOTALL)
        if n != 1:
            raise LookupError(f"{pattern!r} not in {rel}")
        m = dict(t)
        m[rel] = text
        return m

    return build


MUTANTS = [
    ("buffer-owner-kept", "O1", mutate_re(GS, r"alloc_owner_ref\{\s*ggml_sycl::alloc_owner_kind::MODEL\s*,\s*model_id\s*\}",
                                          "alloc_owner_ref{ ggml_sycl::alloc_owner_kind::BUFFER, owner_id }")),
    ("deviation-note-back", "O1", mutate(GS, "static ggml_sycl::alloc_intent ggml_sycl_host_buft_intent(",
                                         "// Accepted deviation\nstatic ggml_sycl::alloc_intent "
                                         "ggml_sycl_host_buft_intent(", raw=True)),
    ("check-before-reclaim", "O2", mutate_re(
        GS, r"(static void ggml_sycl_release_model_slot_resources\([^)]*\)\s*\{)",
        r"\1 ggml_sycl_model_host_chunks_check(owner.model.value);")),
    ("no-rollback-mark", "O3", mutate_re(GS, r"ggml_sycl_model_host_chunks_rolled_back\(owner\.model\.value\);",
                                         "(void) 0;")),
    ("free-reports-nothing", "O3", mutate_re(GS, r"ggml_sycl_model_host_chunk_freed\(ctx->model_id[^;]*;", "(void) 0;")),
    ("gate-skips-streamable", "P1", mutate_re(GS, r"moe_stream_owner_streamable\((owner_md|md), device\)", "true")),
    ("refusal-dropped", "P2", mutate(GS, REFUSAL_THROW, "(void) 0;")),
    ("unowned-registered", "P2", mutate_re(GS, r"(auto register_host_aos_expert = .*?)(bool remembered_handle)",
                                           r"\1host_owner = {};\n\2")),
    ("gate-accepts-unminted", "P1", mutate_re(GS, r"(static bool ggml_sycl_loader_buffer_owner\([^;{]*\)\s*\{.*?)"
                                                  r"ctx->model_id\s*==\s*0", r"\1false")),
    ("gate-refusal-copies", "P4", mutate_re(GS, GATE_REFUSAL_BLOCK + r".*?return\s+false\s*;\s*\}", "")),
    ("gate-refusal-falls-through", "P4", mutate_re(GS, "(" + GATE_REFUSAL_BLOCK + r".*?)return\s+false\s*;(\s*\})",
                                                   r"\1\2")),
    ("gate-refusal-unthrown", "P4", mutate(GS, GATE_REFUSAL_THROW, "(void) 0;")),
    ("loader-term-unchecked", "P3", mutate(GS, LOADER_BUG_FMT, '"[HOST-PLAN] device=%d holds %zu B')),
    ("p3-counts-plan-device", "P3", mutate_re(
        GS, r"ggml_sycl_model_host_chunk_bytes\s*\(\s*model_id\s*,\s*global_plan->loader_device\s*\)",
        "ggml_sycl_model_host_chunk_bytes(model_id, global_plan->device_id)")),
    ("terms-ignore-alignment", "C1", mutate(UCC, "TENSOR_ALIGNMENT", "1")),
    ("split-charges-loader", "C1", mutate_re(UCC, r"plan\.loader_holds_weights\s*&&\s*plan\.loader_device\s*>=\s*0",
                                             "plan.loader_holds_weights")),
    ("loader-bytes-unrecorded", "C1", mutate_re(UCC, r"\bplan\.host_zone_loader_bytes\s*=\s*weight_terms\.loader_residency\s*;",
                                                "")),
    ("configure-sizes-plan-device", "C1", mutate_re(
        GS, r"(placement_plan_host_zone_commit_terms_on_device\s*\(\s*plan\s*,[^;]*?,\s*)device(?=\s*\))",
        r"\1plan.device_id")),
    ("check-host-wide-record", "C1", mutate_re(
        GS, r"ggml_sycl::placement_plan_host_zone_commit_terms_on_device\s*\(\s*plan\s*,\s*plan\.host_zone_terms\s*,"
            r"\s*device\s*\)\.total\(\)", "plan.host_zone_terms.total()")),
    ("loader-drop-silent", "C1", mutate(GS, LOADER_DROP_FMT, '"[HOST-PLAN] device=%d WEIGHT zone: the buffer')),
    ("plain-charge-compare", "C1", mutate_re(UCC, r"!\s*weight_terms\.matches_charge\s*\(\s*plan\.weight_host_bytes\s*\)",
                                             "weight_terms.host_weights != plan.weight_host_bytes")),
    ("fact-ignores-evictable", "C2", mutate(UCC, "info.load_weights_evictable", "true")),
    ("fact-ignores-loader-device", "C2", mutate(UCC, "info.load_loader_device", "0")),
    ("kv-fill-pins-loader-device", "C2", mutate_re(GS, r"\bkv\.load_loader_device\s*=[^;]*;", "kv.load_loader_device = 0;")),
    ("second-fact-writer", "C2", mutate_re(GS, r"(static void ggml_sycl_preload_model_weights\(\)\s*\{)",
                                           r"\1 const_cast<ggml_sycl::placement_plan *>(nullptr)->loader_holds_weights = true;")),
    ("footprint-ignored", "C3", mutate_re(HPP, r"\bterms\.weight\s*=\s*plan\.host_zone_weight_bytes\s*-[^;]*;",
                                          "terms.weight = plan.host_zone_weight_bytes;")),
    ("footprint-at-chunk", "C3", mutate_re(HPP, r"host_zone_buffer_max_size\s*\(\s*plan\.loader_buffer_cap\s*,\s*"
                                                r"chunk_bytes\s*\)", "chunk_bytes")),
    ("grown-uncounted", "C3", mutate_re(HPP, r"(return\s+zones\s*\(\s*\)\s*\+\s*runtime\s*\+\s*rounding)\s*\+\s*"
                                             r"loader_grown\s*;", r"\1;")),
    ("card-sheds-raw", "C3", mutate_re(HPP, r"(on_device\.weight\s*-=\s*std::min\s*\(\s*on_device\.weight\s*,\s*)"
                                            r"terms\.loader_zone", r"\1plan.host_zone_loader_bytes")),
    ("card-keeps-grown", "C3", mutate_re(HPP, r"\bon_device\.loader_grown\s*=\s*0\s*;", "")),
    ("tensors-unrecorded", "C3", mutate_re(UCC, r"\bplan\.loader_tensor_bytes\.push_back\s*\(", "(void) (")),
    ("second-tensor-writer", "C3", mutate_re(GS, r"(static void ggml_sycl_preload_model_weights\(\)\s*\{)",
                                             r"\1 const_cast<ggml_sycl::placement_plan *>(nullptr)->loader_tensor_bytes.clear();")),
    ("cap-uncopied", "C3", mutate_re(UCC, r"\bplan\.loader_buffer_cap\s*=\s*info\.load_loader_buffer_cap\s*;", "")),
    ("cap-unfilled", "C3", mutate_re(GS, r"\bkv\.load_loader_buffer_cap\s*=[^;]*;", "")),
    ("loader-refusal-late-removed", "C4", mutate_re(GS, LOADER_POOL_LATE, "if (false) { return GGML_SYCL_LIFECYCLE_PLAN_REJECTED;")),
    ("loader-refusal-early-removed", "C4", mutate_re(GS, LOADER_POOL_EARLY,
                                                     "if (false) { return GGML_SYCL_LIFECYCLE_PLAN_REJECTED;")),
    ("refusal-ignores-grown", "C4", mutate_re(HPP, r"(\bplan\.host_zone_terms\.)total(\s*\(\s*\)\s*>\s*plan\.host_budget)",
                                              r"\1zones\2")),
    ("refusal-for-every-plan", "C4", mutate_re(HPP, r"\bplan\.loader_device\s*>=\s*0", "true")),
    ("refusal-logs-nothing", "C4", mutate_re(GS, r"(static bool ggml_sycl_stage_refuse_loader_pool_plan\b.*?)GGML_LOG_ERROR",
                                             r"\1(void)")),
    ("max-size-restated", "C3", mutate_re(GS, r"ggml_sycl::host_zone_buffer_max_size\s*\(\s*chunk_cap\s*,",
                                          "std::min(chunk_cap, ggml_sycl::pinned_chunk_pool::CHUNK_SIZE) + 0 * (")),
]


def report(results: dict) -> int:
    failed = 0
    for label, problems in results.items():
        if problems:
            failed += 1
            print(f"FAIL {label}")
            for p in problems:
                print(f"     - {p}")
        else:
            print(f"ok   {label}")
    return failed


def self_test(t: dict) -> int:
    ok = True
    base = run(t)
    dirty = [label for label, problems in base.items() if problems]
    if dirty:
        print(f"  self-test: the unmodified tree FAILS {len(dirty)} clause(s): " + ", ".join(d.split()[0] for d in dirty))
        ok = False
    else:
        print("  self-test: the unmodified tree passes")
    for label, prefix, build in MUTANTS:
        try:
            mutant = build(t)
        except LookupError as e:
            print(f"  self-test: mutant '{label}' cannot be built: {e}")
            ok = False
            continue
        target = [c for c in CLAUSES if c[0].startswith(prefix + " ")]
        if len(target) != 1:
            print(f"  self-test: mutant '{label}' names no single clause {prefix}")
            ok = False
            continue
        problems = target[0][1](mutant)
        if problems:
            print(f"  self-test: mutant '{label}' caught by {prefix}: {problems[0]}")
        else:
            print(f"  self-test: mutant '{label}' was NOT caught by {prefix}")
            ok = False
    print("SELF-TEST " + ("PASS" if ok else "FAIL") + f": {len(MUTANTS)} mutants")
    return 0 if ok else 1


def main(argv: list) -> int:
    t = load_tree()
    failed = report(run(t))
    print(f"=== {len(CLAUSES)} clauses, {failed} failing ===")
    rc = 1 if failed else 0
    if "--self-test" in argv:
        rc = max(rc, self_test(t))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
