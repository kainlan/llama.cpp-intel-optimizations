#!/usr/bin/env python3
"""mem_copy_submit asks a host endpoint's owner, not only the pointer registry, whether it is host USM (llama.cpp-qwn5).

Companion of test-sycl-mem-ops-staged-wait-census-source.py, which counts the waits this removes.

THE DEFECT.  mem_copy_submit (mem-ops.cpp) bounces a host<->device copy through a fresh pinned stage, holding the
calling thread on the host, when its host endpoint is not host USM.  It decided that from the raw pointer alone:
host_ptr_is_usm_accessible() reads only the allocation registry, and pinned-pool and host-zone memory never enters
it.  So every copy to or from the pinned buffer pool -- the hybrid MoE decode route's activations, outputs and ids --
took the synchronous staged path: the flagged scatter's two H2D copies waited their deps on the submitting thread, and
each D2H into the pool waited for its chunk before returning.

THE FIX.  The handle's retained owner answers first, from its provenance, never from the pointer.
mem_handle::owner_metadata() copies the owner's metadata under the handle's lock.  host_endpoint_owner_in_scope()
admits one provenance: a host-pinned slice of a device cache's host zone (host_zone set, device >= 0) that is a
runtime stage (role STAGING or EXPERT_STAGING) or an expert-path buffer (category EXPERT_CACHE).  Such a slice comes
from the cache's pinned arena, allocated on the cache's queue, so host_endpoint_is_usm_accessible() takes the owner's
answer only when that queue (queue_for_device_or_fallback(owner.device, ...)) shares the copy queue's context.  A
standalone pinned base (require_host_usm_base) is allocated on the requesting queue, which the owner does not record,
so its host_zone is unset and it keeps the registry answer: the scope fails closed rather than classify by device.
While any thread records a command graph the owner is not asked, so recorded copies keep the branch they took
before.  Everything else falls back to the registry, which still calls mmap and external host memory unknown.

THE HAZARD THIS OPENS.  The staged branches were a hidden synchronisation: a staged H2D snapshotted its source at
submit, and a staged D2H had landed before it returned.  A direct copy reads its source, or writes its destination,
when the DMA runs.  Each mem_copy_async site whose host endpoint can be in scope was audited (commit message):
  - the MoE ids stage was one stage per ids cache entry, rewritten by a host memcpy on every call (gate, up and down
    share the entry) while the previous call's H2D could still be pending.  It is now fresh per call (pin I);
  - the CPU expert pool's ring entries are rewritten only behind the event chain (pin G): every scatter that read an
    entry was enqueued on the in-order compute queue before the next job's ids or activation copy, the job's prologue
    waits that copy before it writes, a kept sibling job holds disjoint entries, the retained route adds the last
    flush's events or a queue barrier when its copy does not order the scatter, and every job buffer is retained
    until its join;
  - backend-buffer storage stays out of scope (pin T): the SYCL device buffers' owners carry roles COMPUTE, KV and
    WEIGHT, and the SYCL_Host buffer's in-scope owner is held only as the buffer's identity, never as a copy endpoint.

THE CONTRACT.  With comments blanked:
  O. (mem-handle.hpp, mem-handle.cpp) bool owner_metadata(alloc_metadata * out) const is declared; its definition
     takes the handle's lock first, answers false without an owner or an out pointer, copies the owner's metadata and
     answers true; it reads no pointer.
  S. (mem-ops.cpp host_endpoint_owner_in_scope) refuses unless the tier is HOST_PINNED, the host zone is set and the
     device is >= 0, and then admits exactly role STAGING, role EXPERT_STAGING and category EXPERT_CACHE.
  H. (mem-ops.cpp host_endpoint_is_usm_accessible) takes the owner's answer exactly when no graph records, the owner
     answers, the owner is in scope and the owner device's cache queue shares the copy queue's context; the registry
     is asked once, as the last statement, and never before the owner.
  R. (mem-ops.cpp host_ptr_is_usm_accessible) the registry answer is unchanged: host or shared.
  M. (mem-ops.cpp mem_copy_submit) both staged branches test host_endpoint_is_usm_accessible on their host endpoint
     and the chosen copy queue; no bare registry test remains.
  F. The flagged scatter's sources reach the direct path: the pinned buffer pool allocates host-pinned, category
     EXPERT_CACHE, on its device, never as a standalone base, through unified_allocate, which returns
     from_owned_alloc's owner-retaining handle; the route publishes pool.out_handle() and pool.ids_handle(); the
     flagged scatter copies from them on *slot.stream behind its deps; the deps are waited on the host only inside the
     staged host->device branch.
  I. (common.hpp, ggml-sycl.cpp) the MoE ids cache entry keeps no stage; the staging branch allocates a fresh stage
     per call with alloc_pinned_stage_handle_terminal, writes the ids into it and copies from it.
  T. (ggml-sycl.cpp) every use of a SYCL_Host buffer context's buffer_handle is a validity, identity or reset use; the
     host buffer allocator hands its owner only to the wrap, which stores it; every other buffer type's allocation
     requests roles COMPUTE, KV or WEIGHT and never category EXPERT_CACHE.
  G. (ggml-sycl.cpp, moe-decode-hostpath.hpp) the event-chained route refuses an out-of-order queue, flushes or keeps
     every pending scatter between its entry flush and its pool reservation (a keep only through
     moe_sibling_pending_keep, which requires disjoint spans), copies the ids into its entries on that queue after the
     reservation and makes its job wait that copy, and publishes the same queue as the scatter's stream; the slot flush
     flushes the sibling then the primary, records its events as the last flush and retains every job buffer until
     its join; the retained route waits the last flush's events or a queue barrier as moe_cpu_job_order_decide says,
     which answers WAIT_QUEUE for a job that made no activation copy.

NOT VACUOUS.  --self-test runs the gate on in-memory mutants of the real sources and requires each to FAIL on its
own pin, then requires the unmodified tree to pass.

RED on the tree before the change: O, S, H, M, F and I fail; R, T and G pass, because they pin text the change
keeps.
"""

from __future__ import annotations

import functools
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYCL = ROOT / "ggml" / "src" / "ggml-sycl"
FILES = (
    ("ops_cpp", SYCL / "mem-ops.cpp"),
    ("handle_hpp", SYCL / "mem-handle.hpp"),
    ("handle_cpp", SYCL / "mem-handle.cpp"),
    ("pool_cpp", SYCL / "pinned-buffer-pool.cpp"),
    ("cache_cpp", SYCL / "unified-cache.cpp"),
    ("common", SYCL / "common.hpp"),
    ("backend", SYCL / "ggml-sycl.cpp"),
    ("hostpath", SYCL / "moe-decode-hostpath.hpp"),
)
TAG = "(llama.cpp-qwn5)"

OWNER_SIG = r"bool mem_handle::owner_metadata\(alloc_metadata \* out\) const\s*\{"
SCOPE_SIG = r"static bool host_endpoint_owner_in_scope\(const alloc_metadata & owner\)\s*\{"
HELPER_SIG = r"static bool host_endpoint_is_usm_accessible\(const mem_handle & h, const void \* ptr, sycl::queue & copy_queue\)\s*\{"
REGISTRY_SIG = r"static bool host_ptr_is_usm_accessible\(const void \* ptr\)\s*\{"
SUBMIT_SIG = r"static sycl::event mem_copy_submit\(const mem_handle &\s+dst,\s+size_t\s+dst_offset,[^)]*\)\s*\{"
FROM_OWNED_SIG = r"mem_handle mem_handle::from_owned_alloc\(alloc_owner && owner, ggml_layout_mode layout\)\s*\{"
ALLOCATE_SIG = r"\nmem_handle unified_allocate\(const alloc_request & req\)\s*\{"
POOL_INIT_SIG = r"void PinnedBufferPool::init\(sycl::queue & q, int device_id, size_t max_experts, size_t act_dim, size_t out_dim\)\s*\{"
SCATTER_SIG = r"static void moe_host_scatter_submit_flagged\([^)]*\)\s*\{"
ENTRY_SIG = r"struct moe_ids_cache_entry\s*\{"
HOST_ALLOC_SIG = r"static ggml_backend_buffer_t ggml_backend_sycl_host_buffer_type_alloc_buffer\([^)]*\)\s*\{"
HOST_WRAP_SIG = r"static ggml_backend_buffer_t ggml_backend_sycl_host_buffer_wrap\([^)]*\)\s*\{"
BUFFER_ALLOC_SIG = r"\n(?:static )?ggml_backend_buffer_t\s+(\w*alloc\w*)\s*\([^)]*\)\s*(?:try\s*)?\{"
ROUTE_SIG = r"static bool ggml_sycl_moe_hybrid_decode_event_chained\(ggml_backend_sycl_context & ctx,[^)]*\)\s*\{"
FLUSH_SLOT_SIG = r"static void flush_pending_cpu_scatter_slot\(pending_cpu_scatter & slot\)\s*\{"
FLUSH_SIG = r"static void flush_pending_cpu_scatter\(\)\s*\{"
KEEP_SIG = r"inline bool moe_sibling_pending_keep\(const moe_sibling_pending_request & r\)\s*\{"
DISJOINT_SIG = r"inline bool moe_pool_spans_disjoint\(size_t a_first, size_t a_count, size_t b_first, size_t b_count\)\s*\{"
ORDER_SIG = r"inline moe_cpu_job_order moe_cpu_job_order_decide\(const moe_cpu_job_order_request & r\)\s*\{"

H2D_TEST = "if (d.on_device && !s.on_device && !host_endpoint_is_usm_accessible(src, s.ptr, *copy_queue)) {"
D2H_TEST = "if (!d.on_device && s.on_device && !host_endpoint_is_usm_accessible(dst, d.ptr, *copy_queue)) {"
H2D_DEPS_WAIT = "staged_wait_deps(deps, MEM_COPY_STAGED_WAIT_KIND_H2D_DEPS, file, line);"

SCOPE_REFUSAL = ("if (owner.tier != alloc_tier::HOST_PINNED || owner.host_zone == host_zone_id::COUNT || "
                 "owner.device < 0) { return false; }")
SCOPE_REFUSAL_TERMS = (
    ("owner.tier != alloc_tier::HOST_PINNED", "an owner whose tier is not HOST_PINNED"),
    ("owner.host_zone == host_zone_id::COUNT", "an owner outside every host zone (a standalone pinned base)"),
    ("owner.device < 0", "an owner on no device"),
)
SCOPE_ADMITS = (
    "owner.role == alloc_role::STAGING",
    "owner.role == alloc_role::EXPERT_STAGING",
    "owner.category == runtime_category::EXPERT_CACHE",
)
HELPER_TERMS = (
    ("!ggml_sycl_graph_recording_active()", "asks the owner while a command graph records"),
    ("h.owner_metadata(&owner)", "never asks the handle's owner"),
    ("host_endpoint_owner_in_scope(owner)", "takes an owner's answer outside the provenance scope"),
    ("queues_share_context(copy_queue, queue_for_device_or_fallback(owner.device, copy_queue))",
     "takes the owner's answer without checking that its device's cache queue shares the copy queue's context"),
)
HOST_HANDLE_USES = (".valid()", ".identity()", ".has_stable_owner_identity()", " = {};")
BACKEND_BUFFER_ROLES = ("ggml_sycl::alloc_role::COMPUTE", "ggml_sycl::alloc_role::KV",
                        "ggml_sycl::alloc_role::WEIGHT", "alloc_role")
DEVICE_BUFFER_ROLE = ("const ggml_sycl::alloc_role alloc_role = is_compute_buft ? ggml_sycl::alloc_role::COMPUTE : "
                      "(is_kv_buft ? ggml_sycl::alloc_role::KV : ggml_sycl::alloc_role::WEIGHT);")

ROUTE_ENTRY_FLUSH = "flush_pending_cpu_scatter_if_consumed(dst, device);"
ROUTE_RESERVE = "const size_t first = pool.reserve(static_cast<size_t>(n_ids));"
ROUTE_KEEP = "kept = ggml_sycl::moe_sibling_pending_keep(keep);"
ROUTE_KEEP_OR_FLUSH = "if (kept) { move_pending_cpu_scatter_to_sibling(); } else { flush_pending_cpu_scatter(); }"
ROUTE_FLUSH = "} else { flush_pending_cpu_scatter(); reuse = reusable();"
ROUTE_IDS_D2H = ("const sycl::event ids_d2h = ggml_sycl::mem_copy_async(pool.ids_handle(), first * sizeof(int32_t), "
                 "ids_storage.handle, ids_storage.view_offset, ids_bytes, *stream);")
ROUTE_IDS_WAIT = "job_inputs.events.push_back(ids_d2h);"
ROUTE_SUBMIT = "cpu_pool.submit_batch({}, moe_cpu_job_prologue(std::move(job_inputs)),"
ROUTE_IN_ORDER = "!ctx.stream()->has_property<sycl::property::queue::in_order>()"
ROUTE_STREAM = "sycl::queue * stream = ctx.stream();"
ROUTE_SLOT_STREAM = "slot.stream = stream;"
FLUSH_BUFFERS = ("for (const ggml_sycl::mem_handle * h : { &s.out_handle, &s.act_handle, &s.weight_handle, "
                 "&s.act_keepalive, &s.flagged.flags_handle })")
FLUSH_RETAIN = "ggml_sycl::retain_handles_until_event(std::move(job_buffers), joins[k]);"
FLUSH_LAST = "g_cpu_scatter_last_flush.events = ws.events;"
FLUSH_BOTH = "flush_pending_cpu_scatter_slot(g_pending_scatter_sibling); flush_pending_cpu_scatter_slot(g_pending_scatter);"
KEEP_SPANS = ("return r.same_row_geometry && moe_pool_spans_disjoint(r.pending_first, r.pending_count, r.op_first, "
              "r.op_count);")
DISJOINT = "return a_first + a_count <= b_first || b_first + b_count <= a_first;"
ORDER_SWITCH = ("switch (ggml_sycl::moe_cpu_job_order_decide(order_req)) { case ggml_sycl::MOE_CPU_JOB_ORDER_WAIT_LAST_FLUSH: "
                "job_inputs.events.insert(job_inputs.events.end(), g_cpu_scatter_last_flush.events.begin(), "
                "g_cpu_scatter_last_flush.events.end()); break; case ggml_sycl::MOE_CPU_JOB_ORDER_WAIT_QUEUE: "
                "job_inputs.events.push_back(stream->ext_oneapi_submit_barrier()); break; default: break; }")
ORDER_HEAD = ("if (!r.op_from_pool) { return MOE_CPU_JOB_ORDER_OWN_BUFFERS; } if (!r.has_activation_copy || "
              "r.current_scatter_serial < r.copy_scatter_serial) { return MOE_CPU_JOB_ORDER_WAIT_QUEUE; }")


class ContractError(Exception):
    pass


def fail(pin: str, msg: str) -> None:
    raise ContractError(f"FAIL [pin {pin}]: {msg} {TAG}")


_TOKEN = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n]){1,4}\'', re.S)


@functools.lru_cache(maxsize=256)
def blank_comments(src: str) -> str:
    """Blank // and /* */ comments, keeping string literals and offsets."""
    def rep(m: re.Match) -> str:
        s = m.group(0)
        if s.startswith("//"):
            return " " * len(s)
        if s.startswith("/*"):
            return re.sub(r"[^\n]", " ", s)
        return s
    return _TOKEN.sub(rep, src)


def strip_strings(src: str) -> str:
    """Blank the contents of string literals so braces in them do not count."""
    return re.sub(r'"(?:\\.|[^"\\\n])*"', lambda m: '"' + " " * (len(m.group(0)) - 2) + '"', src)


def block_end(code: str, open_brace: int, pin: str) -> int:
    """Index of the brace that closes the one at `open_brace`."""
    depth = 0
    flat_code = strip_strings(code)
    for j in range(open_brace, len(flat_code)):
        if flat_code[j] == "{":
            depth += 1
        elif flat_code[j] == "}":
            depth -= 1
            if depth == 0:
                return j
    fail(pin, "unbalanced braces")
    return 0


def body_at(code: str, m: re.Match, pin: str) -> str:
    i = code.index("{", m.end() - 1)
    return code[i + 1:block_end(code, i, pin)]


def body_of(code: str, signature: str, pin: str) -> str:
    """The body of the first definition matching `signature` (ending at its opening brace)."""
    m = re.search(signature, code)
    if not m:
        fail(pin, f"no definition matching {signature!r}")
    return body_at(code, m, pin)


def flat(text: str) -> str:
    """Whitespace collapsed, and none kept just inside parentheses, so a wrapped call reads as one line."""
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\( ", "(", re.sub(r" \)", ")", text))


def flat_body(source: str, signature: str, pin: str) -> str:
    return flat(body_of(blank_comments(source), signature, pin))


def branch_of(flat_body_text: str, test: str, pin: str) -> tuple[int, str]:
    """(index of `test`, the text of the braced block it opens) in an already flattened body."""
    at = flat_body_text.find(test)
    if at < 0:
        fail(pin, f"no {test!r}")
    brace = at + len(test) - 1
    return at, flat_body_text[brace + 1:block_end(flat_body_text, brace, pin)]


def in_order(body: str, steps, pin: str, where: str) -> None:
    """Each (text, what) of `steps` occurs in `body`, after the one before it."""
    at = 0
    for text, what in steps:
        found = body.find(text, at)
        if found < 0:
            fail(pin, f"{where} {what} (wants {text!r}, in order)")
        at = found + len(text)


def check_owner(s: dict) -> None:
    if "bool owner_metadata(alloc_metadata * out) const;" not in flat(blank_comments(s["handle_hpp"])):
        fail("O", "mem-handle.hpp declares no bool owner_metadata(alloc_metadata * out) const")
    body = flat_body(s["handle_cpp"], OWNER_SIG, "O")
    if not body.startswith("mem_handle_lock_guard g(lock_);"):
        fail("O", "owner_metadata does not take the handle's lock before it reads the owner")
    in_order(body, (
        ("if (!owned_alloc_ || !out) { return false; }", "answers other than false for a handle without an owner"),
        ("*out = owned_alloc_.metadata();", "does not copy the owner's metadata"),
        ("return true;", "does not answer true for a handle with an owner"),
    ), "O", "owner_metadata")
    if not body.endswith("return true;"):
        fail("O", "owner_metadata does not end by answering true")
    for probe in ("ggml_sycl_get_alloc_type", "get_pointer_type", "resolve(", "ptr_"):
        if probe in body:
            fail("O", f"owner_metadata reads the pointer ({probe!r}); its answer must come from the owner")


def check_scope(s: dict) -> None:
    body = flat_body(s["ops_cpp"], SCOPE_SIG, "S")
    end = body.find(") { return false; }")
    if not body.startswith("if (") or end < 0:
        fail("S", "host_endpoint_owner_in_scope does not start by refusing out-of-scope owners")
    refusal = body[len("if ("):end].split(" || ")
    for term, what in SCOPE_REFUSAL_TERMS:
        if term not in refusal:
            fail("S", f"host_endpoint_owner_in_scope admits {what} ({term!r})")
    if not body.startswith(SCOPE_REFUSAL):
        fail("S", f"host_endpoint_owner_in_scope refuses something besides {SCOPE_REFUSAL!r}")
    rest = body[len(SCOPE_REFUSAL):].strip()
    if not rest.startswith("return ") or not rest.endswith(";") or rest.count(";") != 1:
        fail("S", f"host_endpoint_owner_in_scope does not end with one admission: {rest!r}")
    if rest == "return true;":
        fail("S", "host_endpoint_owner_in_scope admits every host-zone owner, backend-buffer storage included")
    admits = rest[len("return "):-1].split(" || ")
    for term in SCOPE_ADMITS:
        if term not in admits:
            fail("S", f"host_endpoint_owner_in_scope no longer admits {term!r}")
    extra = [a for a in admits if a not in SCOPE_ADMITS]
    if extra:
        fail("S", f"host_endpoint_owner_in_scope admits more than runtime stages and expert-path buffers: {extra!r}")


def check_helper(s: dict) -> None:
    body = flat_body(s["ops_cpp"], HELPER_SIG, "H")
    prov = body.find("h.owner_metadata(&owner)")
    reg = body.find("host_ptr_is_usm_accessible(")
    if prov < 0:
        fail("H", "host_endpoint_is_usm_accessible never asks the handle's owner")
    if 0 <= reg < prov:
        fail("H", "host_endpoint_is_usm_accessible asks the registry before the owner")
    start = body.rfind("if (", 0, prov)
    end = body.find(") { return true; }", prov)
    if start < 0 or end < 0:
        fail("H", "host_endpoint_is_usm_accessible does not answer true from one owner test")
    terms = body[start + len("if ("):end].split(" && ")
    for term, what in HELPER_TERMS:
        if term not in terms:
            fail("H", f"host_endpoint_is_usm_accessible {what} (wants {term!r})")
    extra = [t for t in terms if t not in [h[0] for h in HELPER_TERMS]]
    if extra:
        fail("H", f"host_endpoint_is_usm_accessible conditions the owner's answer on {extra!r}")
    if body.count("host_ptr_is_usm_accessible(") != 1 or not body.endswith("return host_ptr_is_usm_accessible(ptr);"):
        fail("H", "host_endpoint_is_usm_accessible does not fall back to the registry, once, as its last statement")


def check_registry(s: dict) -> None:
    body = flat_body(s["ops_cpp"], REGISTRY_SIG, "R")
    want = ("const sycl::usm::alloc alloc = ggml_sycl_get_alloc_type(ptr); "
            "return alloc == sycl::usm::alloc::host || alloc == sycl::usm::alloc::shared;")
    if body != want:
        fail("R", f"host_ptr_is_usm_accessible is no longer the registry's host-or-shared answer: {body!r}")


def check_submit(s: dict) -> None:
    body = flat_body(s["ops_cpp"], SUBMIT_SIG, "M")
    if "host_ptr_is_usm_accessible(" in body:
        fail("M", "mem_copy_submit still asks the registry directly, ignoring the endpoint's owner")
    for test, what in ((H2D_TEST, "host->device"), (D2H_TEST, "device->host")):
        if body.count(test) != 1:
            fail("M", f"the staged {what} branch does not test {test!r} exactly once")
    chosen = body.find("sycl::queue * copy_queue = &queue;")
    if chosen < 0 or chosen > body.find(H2D_TEST) or chosen > body.find(D2H_TEST):
        fail("M", "a staged branch is tested before the copy queue is chosen")


def check_flagged(s: dict) -> None:
    pool = flat_body(s["pool_cpp"], POOL_INIT_SIG, "F")
    for text, what in (
        ("req_act.device = device_id;", "the pool is not allocated on its device"),
        ("req_act.intent.category = runtime_category::EXPERT_CACHE;", "the pool is not an expert-path buffer"),
        ("req_act.intent.constraints.must_host_pinned = true;", "the pool is not host-pinned"),
        ("alloc_request req_out = req_act;", "the out pool does not reuse the pinned request"),
        ("out_handle_ = unified_allocate(req_out);", "the out pool is not an owned handle from unified_allocate"),
        ("alloc_request req_ids = req_act;", "the ids pool does not reuse the pinned request"),
        ("ids_handle_ = unified_allocate(req_ids);", "the ids pool is not an owned handle from unified_allocate"),
    ):
        if text not in pool:
            fail("F", f"PinnedBufferPool::init: {what} ({text!r})")
    if "require_host_usm_base" in pool:
        fail("F", "PinnedBufferPool::init asks for a standalone pinned base, which keeps the registry answer")
    alloc = flat_body(s["cache_cpp"], ALLOCATE_SIG, "F")
    if alloc != ("allocation_result allocation = unified_allocate_owner(req); if (!allocation) return {}; "
                 "return mem_handle::from_owned_alloc(std::move(allocation.owner), GGML_LAYOUT_AOS);"):
        fail("F", f"unified_allocate no longer returns from_owned_alloc's handle: {alloc!r}")
    owned = flat_body(s["handle_cpp"], FROM_OWNED_SIG, "F")
    if not owned.endswith("h.owned_alloc_ = std::move(owner).into_shared(); return h;"):
        fail("F", "from_owned_alloc's handle does not retain the owner")
    backend = flat(blank_comments(s["backend"]))
    for text in ("slot.out_handle = pool.out_handle();", "slot.flagged.flags_handle = pool.ids_handle();"):
        if backend.count(text) != 1:
            fail("F", f"the hybrid route does not publish {text!r} exactly once")
    scatter = flat_body(s["backend"], SCATTER_SIG, "F")
    for text, what in (
        ("sycl::queue & queue = *slot.stream;", "the flagged scatter does not copy on the slot's stream"),
        ("ggml_sycl::mem_copy_async(slot.scatter_scratch, 0, slot.out_handle, slot.pool_first * row_bytes, rows_bytes, "
         "queue, deps);", "the flagged rows are not copied from slot.out_handle behind the deps"),
        ("ggml_sycl::mem_copy_async(slot.scatter_scratch, rows_bytes, f.flags_handle, slot.pool_first * sizeof(int32_t), "
         "flags_bytes, queue, deps);", "the flags are not copied from f.flags_handle behind the deps"),
    ):
        if text not in scatter:
            fail("F", f"{what} ({text!r})")
    submit = flat_body(s["ops_cpp"], SUBMIT_SIG, "F")
    at, branch = branch_of(submit, H2D_TEST, "F")
    if submit.count(H2D_DEPS_WAIT) != 1 or H2D_DEPS_WAIT not in branch:
        fail("F", "mem_copy_submit waits a host->device copy's deps outside the staged branch, so a pinned-pool "
                  "source still holds the submitting thread")
    tail = submit[at + len(branch):]
    if "mem_copy_direct_submit(dst, dst_offset, src, src_offset, size, *copy_queue, deps," not in tail:
        fail("F", "the direct copy does not take the deps the staged branch would have waited")


def check_ids(s: dict) -> None:
    entry = flat_body(s["common"], ENTRY_SIG, "I")
    if "staging" in entry or entry.count("ggml_sycl::mem_handle") != 1:
        fail("I", "moe_ids_cache_entry keeps a stage, which the next call would rewrite under a pending H2D")
    backend = blank_comments(s["backend"])
    if "entry.staging" in backend or "staging_ids_ptr(" in backend:
        fail("I", "ggml-sycl.cpp still reads a stage kept on the MoE ids cache entry")
    flat_backend = flat(backend)
    if flat_backend.count("if (need_staging) {") != 1:
        fail("I", "there is not exactly one MoE ids staging branch")
    start = flat_backend.index("if (need_staging) {")
    end = flat_backend.find("ggml_sycl::mem_copy_async(ids_dev_handle, ids_staging_handle, ids_bytes, *stream);", start)
    if end < 0:
        fail("I", "the MoE ids staging branch does not copy from ids_staging_handle")
    region = flat_backend[start:end]
    for text, what in (
        ("ggml_sycl::mem_handle ids_staging_handle; if (!ggml_sycl::alloc_pinned_stage_handle_terminal(ids_bytes, "
         "*stream, ctx.device, \"moe-ids-stage\", false, &ids_staging_handle)) { return nullptr; }",
         "the stage is not a fresh alloc_pinned_stage_handle_terminal allocation"),
        ("const auto ids_stage = ids_staging_handle.resolve(ctx.device);", "the ids are not written through the stage"),
        ("std::memcpy(ids_stage.ptr, ids_base, ids_bytes);", "the packed ids are not written into the fresh stage"),
        ("std::memcpy(static_cast<char *>(ids_stage.ptr) + row * row_bytes,",
         "the strided ids are not written into the fresh stage"),
    ):
        if text not in region:
            fail("I", f"{what} ({text!r})")
    if re.search(r"\bstatic\b", region):
        fail("I", "the MoE ids stage outlives its call")


def check_storage(s: dict) -> None:
    code = blank_comments(s["backend"])
    flat_backend = flat(code)
    uses = 0
    for m in re.finditer(r"[\w\]\)]+(?:->|\.)buffer_handle\b", flat_backend):
        after = flat_backend[m.end():m.end() + 40]
        before = flat_backend[max(0, m.start() - 40):m.start()]
        if after.startswith(HOST_HANDLE_USES) or (before.endswith("moe_segment_handle_identity(") and
                                                  after.startswith(")")):
            uses += 1
            continue
        fail("T", f"a SYCL_Host buffer's owned handle is used beyond its validity and identity, so a copy could take "
                  f"its in-scope owner: {flat_backend[m.start():m.end() + 60]!r}")
    if uses < 5 or "ggml_sycl_narrow_storage_identity(ctx->buffer_handle.identity()," not in flat_backend:
        fail("T", "the SYCL_Host buffer's handle uses this pin classifies are gone, so it would pass vacuously")
    host_alloc = flat_body(s["backend"], HOST_ALLOC_SIG, "T")
    if len(re.findall(r"\bbuffer_handle\b", host_alloc)) != 2 or "std::move(buffer_handle)" not in host_alloc:
        fail("T", "the SYCL_Host buffer allocator uses its owned handle other than by moving it into the wrap")
    if len(re.findall(r"\bclaimed_handle\b", host_alloc)) != 3 or "std::move(claimed_handle)" not in host_alloc:
        fail("T", "the SYCL_Host buffer allocator uses its claimed slot's handle other than by moving it into the wrap")
    wrap = flat_body(s["backend"], HOST_WRAP_SIG, "T")
    if len(re.findall(r"\bhandle\b", wrap)) != 1 or "std::move(handle)" not in wrap:
        fail("T", "the SYCL_Host buffer wrap uses the owned handle other than by storing it")
    seen = 0
    for m in re.finditer(BUFFER_ALLOC_SIG, code):
        name = m.group(1)
        if name == "ggml_backend_sycl_host_buffer_type_alloc_buffer":
            continue
        body = flat(body_at(code, m, "T"))
        seen += 1
        for role in re.findall(r"intent\.role = ([^;]+);", body):
            if role not in BACKEND_BUFFER_ROLES:
                fail("T", f"{name} requests role {role!r}; a backend buffer's owner must stay out of scope "
                          f"(roles COMPUTE, KV, WEIGHT)")
        if "runtime_category::EXPERT_CACHE" in body:
            fail("T", f"{name} requests category EXPERT_CACHE, which would put a tensor's handle in scope")
        if name == "ggml_backend_sycl_buffer_type_alloc_buffer" and DEVICE_BUFFER_ROLE not in body:
            fail("T", f"the SYCL device buffer's role is no longer {DEVICE_BUFFER_ROLE!r}")
    if seen < 5:
        fail("T", f"only {seen} buffer allocation functions found; the role pin would pass vacuously")


def check_ring(s: dict) -> None:
    route = flat_body(s["backend"], ROUTE_SIG, "G")
    if ROUTE_IN_ORDER not in route:
        fail("G", "the event-chained route no longer refuses an out-of-order queue, whose order the ring relies on")
    entry = route.find(ROUTE_ENTRY_FLUSH)
    reserve = route.find(ROUTE_RESERVE)
    if entry < 0 or reserve < 0 or reserve < entry:
        fail("G", "the event-chained route does not flush the scatters its op consumes before it reserves entries")
    region = route[entry:reserve]
    for text, what in ((ROUTE_KEEP, "keeps a pending job beside this op without moe_sibling_pending_keep"),
                       (ROUTE_KEEP_OR_FLUSH, "neither keeps nor flushes the pending job before it reserves"),
                       (ROUTE_FLUSH, "does not flush the pending scatter before it reserves")):
        if text not in region:
            fail("G", f"the event-chained route {what} ({text!r})")
    ids = route.find(ROUTE_IDS_D2H)
    if ids < reserve:
        fail("G", "the event-chained route does not copy the ids into its reserved entries on the in-order queue")
    wait = route.find(ROUTE_IDS_WAIT, ids)
    submit = route.find(ROUTE_SUBMIT)
    if wait < 0 or submit < 0 or wait > submit:
        fail("G", "the event-chained route's job does not wait its ids copy, so it could rewrite entries an earlier "
                  "scatter still reads")
    if ROUTE_STREAM not in route or route.count(ROUTE_SLOT_STREAM) != 1:
        fail("G", "the event-chained route's scatter is not published on the queue its copies were made on")
    keep = flat_body(s["hostpath"], KEEP_SIG, "G")
    if not keep.endswith(KEEP_SPANS):
        fail("G", "moe_sibling_pending_keep keeps a pending pool job without disjoint spans")
    if flat_body(s["hostpath"], DISJOINT_SIG, "G") != DISJOINT:
        fail("G", "moe_pool_spans_disjoint no longer answers false for overlapping spans")
    slot_flush = flat_body(s["backend"], FLUSH_SLOT_SIG, "G")
    for text, what in ((FLUSH_BUFFERS, "does not retain every buffer a job touches"),
                       (FLUSH_RETAIN, "does not retain the job buffers until the job's join"),
                       (FLUSH_LAST, "does not record its events as the last flush")):
        if text not in slot_flush:
            fail("G", f"flush_pending_cpu_scatter_slot {what} ({text!r})")
    if flat_body(s["backend"], FLUSH_SIG, "G") != FLUSH_BOTH:
        fail("G", "flush_pending_cpu_scatter no longer flushes the sibling slot, then the primary")
    backend = flat(blank_comments(s["backend"]))
    if backend.count(ORDER_SWITCH) != 1:
        fail("G", "the retained route no longer waits the last flush's events or a queue barrier as "
                  "moe_cpu_job_order_decide says")
    after = backend.find("moe_cpu_job_prologue(std::move(job_inputs))", backend.find(ORDER_SWITCH))
    if after < 0:
        fail("G", "the retained route's order wait does not reach the job's prologue")
    if not flat_body(s["hostpath"], ORDER_SIG, "G").startswith(ORDER_HEAD):
        fail("G", "moe_cpu_job_order_decide no longer answers WAIT_QUEUE for a pool job without an activation copy")


CHECKS = (
    ("O", check_owner),
    ("S", check_scope),
    ("H", check_helper),
    ("R", check_registry),
    ("M", check_submit),
    ("F", check_flagged),
    ("I", check_ids),
    ("T", check_storage),
    ("G", check_ring),
)


def check(s: dict) -> None:
    for _, fn in CHECKS:
        fn(s)


def read_sources(root: Path) -> dict:
    out = {}
    for key, path in FILES:
        p = root / path.relative_to(ROOT)
        out[key] = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
    return out


def sub(pattern: str, repl: str, flags: int = 0, count: int = 1):
    def mutate(text: str) -> str:
        return re.sub(pattern, repl, text, count=count, flags=flags)
    return mutate


def chain(*fns):
    def mutate(text: str) -> str:
        for fn in fns:
            text = fn(text)
        return text
    return mutate


def self_test(s: dict) -> int:
    OWN = r"(bool mem_handle::owner_metadata\(alloc_metadata \* out\) const \{[\s\S]*?)"
    HELP = r"(static bool host_endpoint_is_usm_accessible\([^)]*\) \{[\s\S]*?)"
    mutants = [
        ("O", "handle_hpp", "accessor undeclared", sub(r"\n\s*bool owner_metadata\(alloc_metadata \* out\) const;", "")),
        ("O", "handle_cpp", "owner read without the lock", sub(OWN + r"\n\s*mem_handle_lock_guard g\(lock_\);", r"\1")),
        ("O", "handle_cpp", "ownerless handle not refused", sub(OWN + r"if \(!owned_alloc_ \|\| !out\)", r"\1if (!out)")),
        ("O", "handle_cpp", "ownerless handle answers true",
         sub(OWN + r"(if \(!owned_alloc_ \|\| !out\) \{\n\s*return )false;", r"\1\2true;")),
        ("O", "handle_cpp", "metadata not copied", sub(OWN + r"\*out = owned_alloc_\.metadata\(\);", r"\1(void) out;")),
        ("O", "handle_cpp", "pointer probed",
         sub(OWN + r"(\*out = owned_alloc_\.metadata\(\);)",
             r"\1if (ggml_sycl_get_alloc_type(ptr_) == sycl::usm::alloc::unknown) {\n        return false;\n    }\n    \2")),
        ("S", "ops_cpp", "tier test dropped", sub(r"owner\.tier != alloc_tier::HOST_PINNED \|\| ", "")),
        ("S", "ops_cpp", "mmap tier accepted",
         sub(r"owner\.tier != alloc_tier::HOST_PINNED", "owner.tier == alloc_tier::DEVICE_VRAM")),
        ("S", "ops_cpp", "standalone base accepted", sub(r" \|\| owner\.host_zone == host_zone_id::COUNT", "")),
        ("S", "ops_cpp", "owner on no device accepted", sub(r" \|\| owner\.device < 0", "")),
        ("S", "ops_cpp", "backend storage accepted",
         sub(r"return owner\.role == alloc_role::STAGING \|\| owner\.role == alloc_role::EXPERT_STAGING \|\|\n\s*"
             r"owner\.category == runtime_category::EXPERT_CACHE;", "return true;")),
        ("S", "ops_cpp", "COMPUTE role accepted",
         sub(r"return owner\.role == alloc_role::STAGING", "return owner.role == alloc_role::COMPUTE || owner.role == alloc_role::STAGING")),
        ("S", "ops_cpp", "expert-path category dropped",
         sub(r" \|\|\n\s*owner\.category == runtime_category::EXPERT_CACHE;", ";")),
        ("H", "ops_cpp", "recording guard dropped", sub(HELP + r"!ggml_sycl_graph_recording_active\(\) && ", r"\1")),
        ("H", "ops_cpp", "scope not applied", sub(HELP + r"host_endpoint_owner_in_scope\(owner\) &&\s*", r"\1")),
        ("H", "ops_cpp", "provenance ignored",
         sub(HELP + r"\n\s*if \(!ggml_sycl_graph_recording_active\(\)[\s\S]*?return true;\n\s*\}", r"\1")),
        ("H", "ops_cpp", "registry consulted first",
         chain(sub(r"(static bool host_endpoint_is_usm_accessible\([^)]*\) \{\n)",
                   r"\1    if (host_ptr_is_usm_accessible(ptr)) {\n        return true;\n    }\n"),
               sub(r"return host_ptr_is_usm_accessible\(ptr\);\n\}", "return false;\n}"))),
        ("H", "ops_cpp", "registry fallback removed", sub(r"return host_ptr_is_usm_accessible\(ptr\);\n\}", "return false;\n}")),
        ("H", "ops_cpp", "context check dropped",
         sub(r" &&\n\s*queues_share_context\(copy_queue, queue_for_device_or_fallback\(owner\.device, copy_queue\)\)\)", ")")),
        ("H", "ops_cpp", "context compared with itself",
         sub(r"queues_share_context\(copy_queue, queue_for_device_or_fallback\(owner\.device, copy_queue\)\)",
             "queues_share_context(copy_queue, copy_queue)")),
        ("H", "ops_cpp", "context of the copy queue's own device",
         sub(r"queue_for_device_or_fallback\(owner\.device, copy_queue\)",
             "queue_for_device_or_fallback(ggml_sycl_get_device_id_from_queue(copy_queue), copy_queue)")),
        ("R", "ops_cpp", "registry widened to unknown memory",
         sub(r"return alloc == sycl::usm::alloc::host \|\| alloc == sycl::usm::alloc::shared;",
             "return alloc != sycl::usm::alloc::device;")),
        ("M", "ops_cpp", "host->device test reverted",
         sub(r"!host_endpoint_is_usm_accessible\(src, s\.ptr, \*copy_queue\)", "!host_ptr_is_usm_accessible(s.ptr)")),
        ("M", "ops_cpp", "device->host test reverted",
         sub(r"!host_endpoint_is_usm_accessible\(dst, d\.ptr, \*copy_queue\)", "!host_ptr_is_usm_accessible(d.ptr)")),
        ("M", "ops_cpp", "host->device asks the device endpoint's owner",
         sub(r"host_endpoint_is_usm_accessible\(src, s\.ptr,", "host_endpoint_is_usm_accessible(dst, s.ptr,")),
        ("M", "ops_cpp", "context checked against the caller's queue",
         sub(r"host_endpoint_is_usm_accessible\(dst, d\.ptr, \*copy_queue\)", "host_endpoint_is_usm_accessible(dst, d.ptr, queue)")),
        ("F", "pool_cpp", "pool loses its pinned constraint",
         sub(r"\n\s*req_act\.intent\.constraints\.must_host_pinned = true;", "")),
        ("F", "pool_cpp", "pool leaves the expert-path category",
         sub(r"(req_act\.intent\.category\s*= )runtime_category::EXPERT_CACHE;", r"\1runtime_category::HOST_COMPUTE;")),
        ("F", "pool_cpp", "pool asks for a standalone base",
         sub(r"(\n(\s*)req_act\.intent\.constraints\.must_host_pinned = true;)",
             r"\1\n\2req_act.intent.constraints.require_host_usm_base = true;")),
        ("F", "cache_cpp", "unified_allocate hands out an ownerless view",
         sub(r"(\nmem_handle unified_allocate\(const alloc_request & req\) \{\n\s*allocation_result allocation = "
             r"unified_allocate_owner\(req\);\n\s*if \(!allocation\) return \{\};\n\s*)return mem_handle::from_owned_alloc"
             r"\(std::move\(allocation\.owner\), GGML_LAYOUT_AOS\);",
             r"\1return mem_handle::from_direct(allocation.owner.metadata().ptr, GGML_LAYOUT_AOS, false, "
             r"mem_handle::HOST_DEVICE, allocation.owner.metadata().size);")),
        ("F", "handle_cpp", "from_owned_alloc drops the owner",
         sub(r"\n\s*h\.owned_alloc_ = std::move\(owner\)\.into_shared\(\);", "")),
        ("F", "backend", "route publishes a raw view of the out pool",
         sub(r"slot\.out_handle(\s*)= pool\.out_handle\(\);",
             r"slot.out_handle\1= ggml_sycl::mem_handle::from_direct(out_rows, GGML_LAYOUT_AOS, false, "
             r"ggml_sycl::mem_handle::HOST_DEVICE, 0);")),
        ("F", "backend", "route publishes a raw view of the ids pool",
         sub(r"slot\.flagged\.flags_handle(\s*)= pool\.ids_handle\(\);",
             r"slot.flagged.flags_handle\1= ggml_sycl::mem_handle::from_direct(pool.ids_pool(), GGML_LAYOUT_AOS, "
             r"false, ggml_sycl::mem_handle::HOST_DEVICE, 0);")),
        ("F", "backend", "flagged rows copied from another handle",
         sub(r"(mem_copy_async\(slot\.scatter_scratch, 0, )slot\.out_handle,", r"\1slot.act_handle,")),
        ("F", "backend", "flagged scatter copies on another queue",
         sub(r"(sycl::queue &\s+queue\s+= )\*slot\.stream;", r"\1*ggml_sycl::get_shared_context_queue(slot.device_id);")),
        ("F", "ops_cpp", "deps waited before the staged test",
         chain(sub(r"\n\s*staged_wait_deps\(deps, MEM_COPY_STAGED_WAIT_KIND_H2D_DEPS, file, line\);", ""),
               sub(r"(\n    )(if \(d\.on_device && !s\.on_device && !host_endpoint_is_usm_accessible)",
                   r"\1staged_wait_deps(deps, MEM_COPY_STAGED_WAIT_KIND_H2D_DEPS, file, line);\1\2"))),
        ("I", "common", "ids stage kept on the cache entry",
         sub(r"(struct moe_ids_cache_entry \{[\s\S]*?\n(\s*)ggml_sycl::mem_handle device_handle;\n)",
             r"\1\2ggml_sycl::mem_handle staging_handle;\n")),
        ("I", "backend", "ids stage reused across calls",
         sub(r"ggml_sycl::mem_handle ids_staging_handle;\n(\s*)if \(!ggml_sycl::alloc_pinned_stage_handle_terminal\(",
             r"static ggml_sycl::mem_handle ids_staging_handle;\n\1if (!ids_staging_handle.valid() && "
             r"!ggml_sycl::alloc_pinned_stage_handle_terminal(")),
        ("I", "backend", "ids written somewhere other than the fresh stage",
         sub(r"std::memcpy\(ids_stage\.ptr, ids_base, ids_bytes\);", "std::memcpy(entry.device_ids, ids_base, ids_bytes);")),
        ("T", "backend", "SYCL_Host buffer handle used as a copy endpoint",
         sub(r"if \(!ctx->buffer_handle\.valid\(\) \|\| !base \|\| data < base\) \{",
             "ggml_sycl::mem_copy(ctx->buffer_handle, 0, ctx->buffer_handle, 0, 0, "
             "ggml_sycl_get_device(ctx->device).default_queue());\n        "
             "if (!ctx->buffer_handle.valid() || !base || data < base) {")),
        ("T", "backend", "SYCL_Host allocator copies through its owner",
         sub(r"(ggml_sycl::mem_handle buffer_handle =\n\s*ggml_sycl::detail::from_legacy_owned_alloc\(std::move\("
             r"host_buffer_owner\), GGML_LAYOUT_AOS\);)", r"\1\n    ggml_sycl::mem_fill(buffer_handle, 0, size, *exact_queue);")),
        ("T", "backend", "SYCL_Host wrap copies through its owner",
         sub(r"(auto \* ctx = new \(std::nothrow\) sycl_host_buf_ctx\{)",
             r"ggml_sycl::mem_fill(handle, 0, size, ggml_sycl_get_device(device).default_queue());\n    \1")),
        ("T", "backend", "device buffer storage becomes a stage",
         sub(r"(is_kv_buft \? ggml_sycl::alloc_role::KV : )ggml_sycl::alloc_role::WEIGHT\)", r"\1ggml_sycl::alloc_role::STAGING)")),
        ("T", "backend", "arena runtime buffer becomes a stage",
         sub(r"(runtime_req\.intent\.role\s*= )ggml_sycl::alloc_role::COMPUTE;", r"\1ggml_sycl::alloc_role::STAGING;")),
        ("T", "backend", "host compute buffer becomes expert-path memory",
         sub(r"(ggml_backend_sycl_host_compute_buffer_alloc\([\s\S]*?req\.intent\.category\s*= )"
             r"ggml_sycl::runtime_category::HOST_COMPUTE;", r"\1ggml_sycl::runtime_category::EXPERT_CACHE;")),
        ("G", "backend", "route accepts an out-of-order queue",
         sub(r" \|\|\n\s*!ctx\.stream\(\)->has_property<sycl::property::queue::in_order>\(\)\)", ")")),
        ("G", "backend", "route skips its pre-reserve flush",
         sub(r"(\} else \{\n\s*)flush_pending_cpu_scatter\(\);\n(\s*reuse = reusable\(\);)", r"\1\2")),
        ("G", "backend", "route keeps without the keep predicate",
         sub(r"kept(\s*)= ggml_sycl::moe_sibling_pending_keep\(keep\);", r"kept\1= true;")),
        ("G", "backend", "route's job does not wait its ids copy", sub(r"\n\s*job_inputs\.events\.push_back\(ids_d2h\);", "")),
        ("G", "backend", "route's scatter published on another queue",
         sub(r"(slot\.weight_pinned\s*= nullptr;\n\s*slot\.out_handle\s*= pool\.out_handle\(\);[\s\S]*?slot\.stream\s*= )stream;",
             r"\1&ggml_sycl_get_device(device).default_queue();")),
        ("G", "hostpath", "keep ignores the spans",
         sub(r"return r\.same_row_geometry && moe_pool_spans_disjoint\(r\.pending_first, r\.pending_count, r\.op_first, "
             r"r\.op_count\);", "return r.same_row_geometry;")),
        ("G", "hostpath", "overlapping spans called disjoint",
         sub(r"return a_first \+ a_count <= b_first \|\| b_first \+ b_count <= a_first;", "return a_first <= b_first || b_first <= a_first;")),
        ("G", "backend", "flags not retained until the join", sub(r", &s\.flagged\.flags_handle", "")),
        ("G", "backend", "job buffers not retained",
         sub(r"\n\s*ggml_sycl::retain_handles_until_event\(std::move\(job_buffers\), joins\[k\]\);", "")),
        ("G", "backend", "last flush not recorded", sub(r"\n\s*g_cpu_scatter_last_flush\.events(\s*)= ws\.events;", "")),
        ("G", "backend", "sibling slot not flushed",
         sub(r"(static void flush_pending_cpu_scatter\(\) \{)\n\s*flush_pending_cpu_scatter_slot\(g_pending_scatter_sibling\);", r"\1")),
        ("G", "backend", "WAIT_LAST_FLUSH waits nothing",
         sub(r"(case ggml_sycl::MOE_CPU_JOB_ORDER_WAIT_LAST_FLUSH:)\n\s*job_inputs\.events\.insert\([^;]*;", r"\1")),
        ("G", "backend", "WAIT_QUEUE submits no barrier",
         sub(r"(case ggml_sycl::MOE_CPU_JOB_ORDER_WAIT_QUEUE:)\n\s*job_inputs\.events\.push_back\(stream->ext_oneapi_submit_barrier\(\)\);",
             r"\1")),
        ("G", "hostpath", "job without an activation copy not ordered",
         sub(r"if \(!r\.has_activation_copy \|\| r\.current_scatter_serial", "if (r.current_scatter_serial")),
    ]
    failures = []
    caught = 0
    for pin, key, name, mutate in mutants:
        m = dict(s)
        m[key] = mutate(s[key] or "")
        if m[key] == s[key]:
            failures.append(f"mutant '{name}' did not apply")
            print(f"  self-test: mutant '{name}' did not apply")
            continue
        try:
            check(m)
        except ContractError as e:
            if f"[pin {pin}]" in str(e):
                caught += 1
                print(f"  self-test: mutant '{name}' caught: {e}")
                continue
            failures.append(f"mutant '{name}' failed on the wrong pin: {e}")
            print(f"  self-test: mutant '{name}' failed on the wrong pin: {e}")
            continue
        failures.append(f"mutant '{name}' passed")
        print(f"  self-test: mutant '{name}' PASSED (not caught)")
    try:
        check(s)
        print("  self-test: unmodified tree passes")
    except ContractError as e:
        failures.append(f"unmodified tree fails: {e}")
        print(f"  self-test: unmodified tree FAILS: {e}")
    if failures:
        print(f"SELF-TEST FAIL: {', '.join(failures)}")
        return 1
    print(f"SELF-TEST PASS: {caught} mutants caught, unmodified tree passes")
    return 0


def main(argv: list[str]) -> int:
    root = ROOT
    if "--root" in argv:
        root = Path(argv[argv.index("--root") + 1]).resolve()
    s = read_sources(root)
    if "--self-test" in argv:
        rc = self_test(s)
        if rc != 0:
            return rc
    failed = 0
    for pin, fn in CHECKS:
        try:
            fn(s)
            print(f"  pin {pin}: PASS")
        except ContractError as e:
            failed += 1
            print(f"  pin {pin}: {e}")
    if failed:
        print(f"FAIL: {failed} of {len(CHECKS)} pins {TAG}")
        return 1
    print("PASS: mem_copy_submit asks an in-scope host endpoint's owner before the registry, the MoE ids stage is "
          f"fresh per call, backend storage stays out of scope and the ring is rewritten only behind its events {TAG}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
