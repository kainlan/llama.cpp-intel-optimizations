#!/usr/bin/env python3
"""Gates 3, 23, 29, 30, 31, 36 and 37 (llama.cpp-zhcn): the backend half of the re-plan protocol.

C6 landed the backend side of the context-tenant plan: the chunk-cap wrapper and the
plan scopes that read it, the measure plan override and the non-owning measure backend,
the L0 token at every publish entry and its always-compiled witnesses, the two re-plan
hooks, and the shared host-dispatch predicate.  This gate pins what each of those must
keep being true, source-side.  What it checks, by gate:

  3   the in-scope branch of each compute buft's get_max_size answers from the context's
      copy and reads no zone; the wrapper `ggml_sycl_arena_chunk_cap` is the only
      function that names the arena backing, once, with `is_vm` pinned to the VM kind and
      every capacity read inside the `is_vm` branch; the pure core names no backing; each
      set has its one reader (COMMITTED: the freeze alone); the PRIVATE_TESTING buft
      factory takes its interface by name; nothing writes `max_size_override`.
  23  `ggml_backend_sycl_graph_invalidate` tests the recorded-state predicate before any
      clear and reaches none of the four process-global effects; the scoped clear body
      reaches none; `sycl_exec_graph_clear_active` is the scoped body plus exactly those
      four calls; its callers are the eleven phase-boundary, replay-trip and teardown
      sites.
  29  every `host_task(` site under ggml/src/ggml-sycl/ is censused, and none captures a
      `managed_handle`, `tp_handles`, `data_handle` or a `find_tensor_storage_handle`
      result, nor sits in a `[&]` scope that does.
  30  every namespace-scope, static, thread_local or ggml_backend_sycl_context queue is
      censused, with either the token `ggml_backend_sycl_synchronize_for_replan` waits it
      through or the reason it cannot reach a tenant slice; `ggml_backend_sycl_synchronize`
      flushes the MoE scatter lists before its first wait; the flush-disabling hook compiles only
      under GGML_SYCL_PRIVATE_TESTING.
  31  L0 is one token type over one mutex with a thread-local depth; no token is a class
      member; the fourteen publish and prepare entries construct it first, with the
      listed kind; the witness macro and every witness site is always compiled, with its
      message byte for byte; the preload's witness is `held(LOAD)`.
  36  the four plan accessors consult the override first and every other read of the
      publication is theirs; the five scheduler-path latch reads go through the one
      reader; the override has two writers, one nesting witness, and a snapshot builder
      that names no provisional-plan helper; the publish store refuses first; the measure
      backend is non-owning, syntactically.
  37  both pool phase gates skip only under a TRANSACTION token and name their site.

The walk is textual, comment- and string-stripped, so a call reached through a wrapper is
not seen; each clause names the function that must contain what it requires.  Every
clause proves itself on mutants of the real source and the gate refuses to pass
vacuously.

argv: [checkout-root]
"""
import importlib.util
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_hc = _load("holder_census", "test-sycl-holder-census-source.py")
code = _hc.code
keep = _hc.keep
func_bodies = _hc.func_bodies
struct_body = _hc.struct_body
nested_struct_body = _hc.nested_struct_body
text_of = _hc.text_of
enclosing_top = _hc.enclosing_top
top_spans = _hc.top_spans
close_of = _hc.close_of
line_of = _hc.line_of
edit = _hc.edit

SYCL = "ggml/src/ggml-sycl/"
MAIN = SYCL + "ggml-sycl.cpp"
COMMON_H = SYCL + "common.hpp"
COMMON_C = SYCL + "common.cpp"
CPUD = SYCL + "cpu-dispatch.cpp"
DPCT = SYCL + "dpct/helper.hpp"
UC_C = SYCL + "unified-cache.cpp"
UC_H = SYCL + "unified-cache.hpp"
POOL = SYCL + "pinned-pool.cpp"
CHUNK = SYCL + "chunk-cap.hpp"
HDR = "ggml/include/ggml-sycl.h"


def all_sycl_sources(root):
    out = []
    base = os.path.join(root, SYCL)
    for d, dirs, names in os.walk(base):
        dirs[:] = [x for x in dirs if x != "tests"]
        for n in names:
            if n.endswith((".cpp", ".hpp", ".h", ".cu")):
                out.append(os.path.relpath(os.path.join(d, n), root))
    return sorted(out)


def load(root):
    paths = set(all_sycl_sources(root)) | {HDR}
    return {p: open(os.path.join(root, p), encoding="utf-8").read() for p in paths}


def only_body(c, name, out, why_missing):
    bodies = func_bodies(c, name)
    if len(bodies) != 1:
        out.append("%s: %d definitions of %s (wanted exactly 1)" % (why_missing, len(bodies), name))
        return None
    return bodies[0]


def body_text(c, name, out, label):
    b = only_body(c, name, out, label)
    return text_of(c, b) if b else ""


def strip_pp(raw_c):
    return re.sub(r"(?m)^\s*#.*$", "", raw_c)


def pp_arms(raw, pos):
    """The enclosing preprocessor conditionals of `pos` as (condition text, in_else) pairs."""
    stack = []
    for m in re.finditer(r"(?m)^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$", raw[:pos]):
        kind, rest = m.group(1), m.group(2).strip()
        if kind in ("if", "ifdef", "ifndef"):
            stack.append([("!" if kind == "ifndef" else "") + rest, False])
        elif kind == "elif":
            if stack:
                stack[-1] = [rest, True]
        elif kind == "else":
            if stack:
                stack[-1][1] = True
        elif kind == "endif":
            if stack:
                stack.pop()
    return stack


def in_live_private_testing(raw, pos):
    for cond, in_else in pp_arms(raw, pos):
        if in_else:
            continue
        if re.fullmatch(r"(defined\s*\(\s*GGML_SYCL_PRIVATE_TESTING\s*\)|GGML_SYCL_PRIVATE_TESTING)", cond):
            return True
    return False


def calls(text, name):
    return [m.start() for m in re.finditer(r"(?<![\w:])(?:\w+::)*" + re.escape(name) + r"\s*\(", text)]


# --------------------------------------------------------------------------------------
# gate 3: the chunk cap
# --------------------------------------------------------------------------------------
def gate3(files, bad):
    c = code(files, MAIN)
    chunk = code(files, CHUNK)

    # The device buft: max_size_override, then the in-scope branch, then the device-index
    # check, then the arena read.
    dev = body_text(c, "ggml_backend_sycl_buffer_type_get_max_size", bad, "gate 3")
    if dev:
        order = [("max_size_override", dev.find("max_size_override")),
                 ("g_plan_scope", (re.search(r"if\s*\(\s*g_plan_scope\s*\)\s*\{", dev) or re.search("$", dev)).start()
                  if re.search(r"if\s*\(\s*g_plan_scope\s*\)\s*\{", dev) else -1),
                 ("device_count", dev.find("device_count")), ("arena_chunk_cap", dev.find("ggml_sycl_arena_chunk_cap("))]
        if any(p < 0 for _, p in order) or [p for _, p in order] != sorted(p for _, p in order):
            bad("gate 3: the device get_max_size does not run max_size_override, the in-scope branch, the device-index "
                "check and the arena read in that order: %s" % order)
        scope_m = re.search(r"if\s*\(\s*g_plan_scope\s*\)\s*\{", dev)
        scope_at = scope_m.start() if scope_m else -1
        if scope_at < 0:
            bad("gate 3: the device get_max_size has no plain `if (g_plan_scope)` branch")
        else:
            blk_open = dev.find("{", scope_at)
            blk = dev[blk_open:close_of(dev, blk_open) + 1] if blk_open >= 0 else ""
            if "ggml_sycl_plan_scope_chunk_cap(" not in blk:
                bad("gate 3: the device get_max_size's in-scope branch does not return ggml_sycl_plan_scope_chunk_cap")
            if re.search(r"zone_capacity|host_zone_largest_free_block|ggml_sycl_arena_chunk_cap", blk):
                bad("gate 3: the device get_max_size's in-scope branch reads a zone")
    # The SYCL_Host and CPU-offload bufts.
    host_fn = None
    for m in re.finditer(r"ggml_sycl_chunk_cap_buft_kind::HOST\s*,", c):
        host_fn = enclosing_top(c, m.start())
    if not host_fn:
        bad("gate 3: no get_max_size passes the HOST kind to the scope")
    else:
        h = body_text(c, host_fn.split("::")[-1], bad, "gate 3")
        a = h.find("g_plan_scope")
        z = [p for p in (h.find("zone_capacity"), h.find("host_zone_largest_free_block")) if p >= 0]
        if a < 0 or (z and min(z) < a):
            bad("gate 3: the SYCL_Host get_max_size reads a zone before its in-scope branch")
    off = body_text(c, "ggml_backend_sycl_cpu_offload_compute_get_max_size", bad, "gate 3")
    if off:
        a = off.find("g_plan_scope")
        z = [p for p in (off.find("zone_capacity"), off.find("host_zone_largest_free_block")) if p >= 0]
        if a < 0 or (z and min(z) < a) or "CPU_OFFLOAD" not in off:
            bad("gate 3: the CPU-offload get_max_size lacks its in-scope branch first")

    # The functions that answer an in-scope read read no zone themselves.
    for fn in ("ggml_sycl_plan_scope_chunk_cap", "ggml_sycl_plan_scope_load_measure_cap",
               "ggml_backend_sycl_plan_caps_freeze", "ggml_sycl_plan_caps_store", "ggml_sycl_plan_caps_find"):
        t = body_text(c, fn, bad, "gate 3")
        if re.search(r"zone_capacity|host_zone_largest_free_block|max_size_override", t):
            bad("gate 3: %s reads a zone or max_size_override itself" % fn)

    # The wrapper.
    wrapper = only_body(c, "ggml_sycl_arena_chunk_cap", bad, "gate 3")
    if wrapper:
        w = text_of(c, wrapper)
        if len(calls(w, "ggml_sycl_arena_backing")) != 1:
            bad("gate 3: the wrapper does not call ggml_sycl_arena_backing exactly once")
        if len(calls(w, "ggml_sycl_chunk_cap_core")) != 1:
            bad("gate 3: the wrapper does not call ggml_sycl_chunk_cap_core exactly once")
        if not re.search(r"is_vm\s*=\s*(?:ggml_sycl::)?ggml_sycl_arena_backing\s*\(\s*\w+\s*\)\s*==\s*"
                         r"(?:ggml_sycl::)?GGML_SYCL_ARENA_BACKING_TYPE_VM\s*;", w):
            bad("gate 3: the wrapper's is_vm is not exactly `ggml_sycl_arena_backing(dev) == "
                "GGML_SYCL_ARENA_BACKING_TYPE_VM`")
        m = re.search(r"if\s*\(\s*result\.is_vm\s*\)\s*\{", w)
        if not m:
            bad("gate 3: the wrapper has no `if (result.is_vm)` branch")
        else:
            blk_end = close_of(w, m.end() - 1)
            for r in re.finditer(r"zone_capacity\w*|ggml_sycl_compute_arena_bytes", w):
                if not (m.end() <= r.start() <= blk_end):
                    bad("gate 3: the wrapper reads a capacity (%s) outside the is_vm branch" % r.group())
    # No other chunk-cap-path function names the backing.
    # The device function's out-of-scope USM tail is 1oxa's and may read the arena; the others read nothing.
    for fn, pat in (("ggml_backend_sycl_buffer_type_get_max_size", r"ggml_sycl_arena_backing"),
                    ("ggml_backend_sycl_plan_caps_freeze", r"ggml_sycl_arena_backing|vram_arena_enabled|arena_active"),
                    ("ggml_sycl_plan_scope_load_measure_cap", r"ggml_sycl_arena_backing|vram_arena_enabled|arena_active"),
                    ("ggml_sycl_plan_scope_chunk_cap", r"ggml_sycl_arena_backing|vram_arena_enabled|arena_active"),
                    ("ggml_backend_sycl_cpu_offload_compute_get_max_size",
                     r"ggml_sycl_arena_backing|vram_arena_enabled|arena_active")):
        for b in func_bodies(c, fn):
            if re.search(pat, text_of(c, b)):
                bad("gate 3: %s reads the backing kind itself" % fn)
    # The core.
    core = only_body(chunk, "ggml_sycl_chunk_cap_core", bad, "gate 3")
    if core and re.search(r"ggml_sycl_arena_backing|vram_arena_enabled|arena_active", text_of(chunk, core)):
        bad("gate 3: the core names the backing kind")
    for pos in calls(c, "ggml_sycl_chunk_cap_core"):
        fn = enclosing_top(c, pos)
        if fn not in ("ggml_sycl_arena_chunk_cap", "ggml_backend_sycl_plan_caps_freeze_core"):
            bad("gate 3: ggml_sycl_chunk_cap_core is called from %s" % fn)
    # Each set has its one reader.
    readers = {"COMMITTED": {"ggml_backend_sycl_plan_caps_freeze"},
               "LEDGER_CURRENT": {"ggml_backend_sycl_buffer_type_get_max_size"},
               "PROBE_MIN": {"ggml_sycl_plan_scope_load_measure_cap"},
               "LOAD_TO_COMMIT": {"ggml_sycl_plan_scope_load_measure_cap"}}
    for m in re.finditer(r"ggml_sycl_arena_chunk_cap\s*\(", c):
        stmt_end = c.find(";", m.start())
        stmt = c[m.start():stmt_end]
        fn = enclosing_top(c, m.start())
        if fn and fn.startswith("ggml_sycl_arena_chunk_cap"):
            continue
        for sname in re.findall(r"ggml_sycl_chunk_cap_set::(\w+)", stmt):
            if fn not in readers.get(sname, set()):
                bad("gate 3: set %s is read from %s" % (sname, fn))
    # The factory.
    fac = body_text(c, "ggml_backend_sycl_buffer_type_make_for_testing", bad, "gate 3")
    if fac and ("ggml_backend_sycl_buffer_type_interface" not in fac or "ggml_backend_buffer_type_i" in fac):
        bad("gate 3: the PRIVATE_TESTING buft factory does not take its interface from "
            "ggml_backend_sycl_buffer_type_interface by name")
    # No writer of max_size_override on the plan path.
    for fname in files:
        if not fname.startswith(SYCL):
            continue
        t = code(files, fname)
        for m in re.finditer(r"max_size_override\s*(?:=(?!=)|\+=|-=)", t):
            ls = t.rfind("\n", 0, m.start()) + 1
            if re.match(r"\s*size_t\s+max_size_override\b", t[ls:m.end()]):
                continue  # the member's default initializer
            bad("gate 3: %s writes max_size_override (line %d)" % (fname, line_of(files[fname], m.start())))


# --------------------------------------------------------------------------------------
# gate 23: graph invalidation
# --------------------------------------------------------------------------------------
GLOBAL_CLEAR_EFFECTS = ("release_graph_retained_handles", "ggml_sycl_cpu_staging_cache_clear", "graph_unpin_moe_experts",
                        "graph_unpin_weights")
CLEAR_ACTIVE_CALLERS = 13  # the design lists eleven; optional-layouts-retire (dkw0) is the twelfth on this base, and the legacy re-record path's prestage-declined clear (master, with graph_prestage_or_decline) the thirteenth


def gate23(files, bad):
    c = code(files, MAIN)
    inv = body_text(c, "ggml_backend_sycl_graph_invalidate", bad, "gate 23")
    if inv:
        a = inv.find("sycl_exec_graph_has_recorded_state(")
        b = inv.find("sycl_exec_graph_clear_scoped(")
        if a < 0 or b < 0 or a > b:
            bad("gate 23: graph_invalidate does not test sycl_exec_graph_has_recorded_state before its clear")
        if "sycl_exec_graph_clear_active(" in inv:
            bad("gate 23: graph_invalidate calls the clear that carries the process-global effects")
        for fn in GLOBAL_CLEAR_EFFECTS:
            if fn in inv:
                bad("gate 23: graph_invalidate reaches %s" % fn)
    scoped = body_text(c, "sycl_exec_graph_clear_scoped", bad, "gate 23")
    for fn in GLOBAL_CLEAR_EFFECTS:
        if scoped and fn in scoped:
            bad("gate 23: the scoped clear body reaches %s" % fn)
    active = body_text(c, "sycl_exec_graph_clear_active", bad, "gate 23")
    if active:
        if len(calls(active, "sycl_exec_graph_clear_scoped")) != 1:
            bad("gate 23: sycl_exec_graph_clear_active does not call the scoped body exactly once")
        for fn in GLOBAL_CLEAR_EFFECTS:
            if len(calls(active, fn)) != 1:
                bad("gate 23: sycl_exec_graph_clear_active does not call %s exactly once" % fn)
        other = set(re.findall(r"(?<![\w:.>])([A-Za-z_]\w*)\s*\(", active)) - {
            "sycl_exec_graph_clear_active", "sycl_exec_graph_clear_scoped", "if"} - set(GLOBAL_CLEAR_EFFECTS)
        if other:
            bad("gate 23: sycl_exec_graph_clear_active calls more than the scoped body and the four effects: %s"
                % sorted(other))
    sites = [p for p in calls(c, "sycl_exec_graph_clear_active")
             if enclosing_top(c, p) not in ("sycl_exec_graph_clear_active", None)]
    # the forward declaration is a top-level statement with no body, so it is not a call site
    if len(sites) != CLEAR_ACTIVE_CALLERS:
        bad("gate 23: sycl_exec_graph_clear_active has %d callers (wanted the %d listed)" % (len(sites), CLEAR_ACTIVE_CALLERS))
    for p in sites:
        if enclosing_top(c, p) in ("ggml_backend_sycl_graph_invalidate", "ggml_backend_sycl_synchronize_for_replan"):
            bad("gate 23: the re-plan hook %s reaches sycl_exec_graph_clear_active" % enclosing_top(c, p))


# --------------------------------------------------------------------------------------
# gate 29: the host_task capture census
# --------------------------------------------------------------------------------------
HOST_TASK_CENSUS = {CPUD: 10, MAIN: 3, DPCT: 2}
FORBIDDEN_CAPTURE = ("managed_handle", "tp_handles", "data_handle", "find_tensor_storage_handle")


def host_task_sites(c):
    out = []
    for m in re.finditer(r"\bhost_task\s*\(", c):
        lam = c.find("[", m.end() - 1)
        if lam < 0 or c[m.end():lam].strip():
            continue
        cap_end = c.find("]", lam)
        open_b = c.find("{", cap_end)
        close_b = close_of(c, open_b)
        out.append((m.start(), c[lam + 1:cap_end], c[open_b:close_b + 1]))
    return out


def gate29(files, bad):
    for path in all_sycl_sources_in(files):
        c = code(files, path)
        sites = host_task_sites(c)
        want = HOST_TASK_CENSUS.get(path, 0)
        if len(sites) != want:
            bad("gate 29: %s has %d host_task sites (censused: %d); list and review the new one" % (path, len(sites), want))
        for pos, cap, body in sites:
            for f in FORBIDDEN_CAPTURE:
                if f in cap or f in body:
                    bad("gate 29: the host_task at %s:%d captures %s" % (path, line_of(files[path], pos), f))
            if cap.strip() == "&":
                scope_open = c.rfind("[&]", 0, pos)
                scope_text = c[scope_open:pos + len(body) + 200] if scope_open >= 0 else body
                for f in FORBIDDEN_CAPTURE:
                    if f in scope_text:
                        bad("gate 29: the [&] host_task scope at %s:%d contains %s" % (path, line_of(files[path], pos), f))


def all_sycl_sources_in(files):
    return sorted(p for p in files if p.startswith(SYCL) and "/tests/" not in p)


# --------------------------------------------------------------------------------------
# gate 30: the queue census
# --------------------------------------------------------------------------------------
# name -> ("wait", token that synchronize_for_replan's body must contain) or ("reason", why)
QUEUE_CENSUS = {
    "g_tp_shared_queues":            ("wait", "ggml_sycl_get_tp_queue("),
    "g_retained_gpu_q":              ("reason", "alias of the context's execution queue, set from the gpu_q argument"),
    "g_shared_ctx_queues":           ("wait", "get_shared_context_queue("),
    "g_recording_queue_ptr":         ("reason", "thread-local alias of the execution queue while a graph records"),
    "g_pipeline_copy_queue":         ("wait", "g_pipeline_copy_queue["),
    "g_tp_device1_worker_queue":     ("wait", "g_tp_device1_worker_queue"),
    "g_split_secondary_queue_owner": ("wait", "g_split_secondary_queue_owner"),
    "g_split_merge_queue_owner":     ("wait", "g_split_merge_queue_owner"),
    "g_split_coord_queue_owner":     ("wait", "g_split_coord_queue_owner"),
    "g_split_merge_queue":           ("reason", "raw alias of g_split_merge_queue_owner"),
    "g_block_exec_copy_queues":      ("wait", "g_block_exec_copy_queues"),
    "qptrs":                         ("wait", "ggml_sycl_execution_queue_for_device("),
    "pool_qptrs":                    ("reason", "aliases of qptrs entries"),
    "host_pool_qptrs":               ("reason", "aliases of qptrs entries"),
}


def queue_decls(files):
    """Names of every namespace-scope, static, thread_local or ggml_backend_sycl_context sycl::queue object or pointer."""
    found = {}
    pat = re.compile(r"(?m)^[ \t]*((?:static|thread_local|inline|extern)\b[^;(){}]*?"
                     r"(?:sycl::queue|queue_ptr|std::unique_ptr<sycl::queue>|std::array<std::unique_ptr<sycl::queue>[^;(){}]*>)"
                     r"[^;(){}]*?\b(\w+)\s*(?:\[[^\]]*\])?\s*(?:=|;|\{))")
    for path in all_sycl_sources_in(files):
        c = code(files, path)
        for m in pat.finditer(c):
            if "unordered_map" in m.group(1) or "typedef" in m.group(1):
                continue
            found.setdefault(m.group(2), []).append(path)
    ctx = None
    if COMMON_H in files:
        cc = code(files, COMMON_H)
        ctx = nested_struct_body(cc, "ggml_backend_sycl_context")
        if ctx:
            body = text_of(cc, ctx)
            for m in re.finditer(r"(?m)^[ \t]*(?:queue_ptr|sycl::queue\s*\*?)\s+(\w+)\s*(?:\[[^\]]*\])+\s*(?:=|;)", body):
                found.setdefault(m.group(1), []).append(COMMON_H)
    return found, ctx


def gate30(files, bad):
    c = code(files, MAIN)
    found, ctx = queue_decls(files)
    if ctx is None:
        bad("gate 30: ggml_backend_sycl_context not found in common.hpp")
    for name, paths in sorted(found.items()):
        if name not in QUEUE_CENSUS:
            bad("gate 30: %s (%s) is not in the queue census: list it with the line that waits it or its reason" %
                (name, ", ".join(sorted(set(paths)))))
    for name in QUEUE_CENSUS:
        if name not in found:
            bad("gate 30: censused queue %s is not declared any more" % name)
    sync = body_text(c, "ggml_backend_sycl_synchronize_for_replan", bad, "gate 30")
    for name, (kind, what) in QUEUE_CENSUS.items():
        if kind == "wait" and sync and what not in sync:
            bad("gate 30: ggml_backend_sycl_synchronize_for_replan does not wait %s (looked for %s)" % (name, what))
    if sync and not re.search(r"->\s*wait\s*\(\s*\)|\.wait\s*\(\s*\)", sync):
        bad("gate 30: ggml_backend_sycl_synchronize_for_replan waits nothing")
    # The MoE scatter flush precedes the first wait in ggml_backend_sycl_synchronize.
    s = body_text(c, "ggml_backend_sycl_synchronize", bad, "gate 30")
    if s:
        f = s.find("ggml_sycl_cpu_tg_flush_pending(")
        w = re.search(r"wait\s*\(", s)
        if f < 0 or (w and f > w.start()):
            bad("gate 30: ggml_backend_sycl_synchronize does not call ggml_sycl_cpu_tg_flush_pending before its first wait")
    raw = files[MAIN]
    for m in re.finditer(r"(?m)^(?:void|bool)\s+ggml_sycl_test_set_cpu_tg_flush_disabled\s*\(", raw):
        if not in_live_private_testing(raw, m.start()):
            bad("gate 30: ggml_sycl_test_set_cpu_tg_flush_disabled is compiled outside GGML_SYCL_PRIVATE_TESTING")
    if not re.search(r"ggml_sycl_test_set_cpu_tg_flush_disabled\s*\(", raw):
        bad("gate 30: ggml_sycl_test_set_cpu_tg_flush_disabled is absent")


# --------------------------------------------------------------------------------------
# gate 31: L0
# --------------------------------------------------------------------------------------
TOKEN_ENTRIES = {
    # Both public set-runtime-context entries are one-line wrappers; the body, and so the token, is the impl's.
    "ggml_sycl_set_runtime_context_for_model_impl":    "TRANSACTION",
    "ggml_backend_sycl_set_runtime_context":           "TRANSACTION",
    "ggml_backend_sycl_model_load_begin":              "LOAD",
    "ggml_backend_sycl_stage_inventory_plan":          "LOAD",
    "ggml_backend_sycl_model_load_end":                "LOAD",
    "ggml_backend_sycl_probe_runtime_context_for_model":        "LIFECYCLE",
    "ggml_backend_sycl_recheck_runtime_context_flash_attn":     "LIFECYCLE",
    "ggml_backend_sycl_activate_model_plan":           "LIFECYCLE",
    "ggml_backend_sycl_model_unloaded_token":          "LIFECYCLE",
    "ggml_backend_sycl_shutdown":                      "LIFECYCLE",
    "ggml_backend_sycl_commit_reactivate":             "LIFECYCLE",
    "ggml_backend_sycl_rollback_reactivate":           "LIFECYCLE",
    "ggml_backend_sycl_complete_unload":               "LIFECYCLE",
    "ggml_backend_sycl_can_unload":                    "LIFECYCLE",
}
# (file, function, message): each is a witness site with its literal.
WITNESS_SITES = (
    (MAIN, "ggml_backend_sycl_buffer_type_alloc_buffer", "[REPLAN-TOKEN] TRANSACTION token held at alloc_buffer entry"),
    (MAIN, "ggml_backend_sycl_buffer_type_alloc_buffer", "[REPLAN-TOKEN] TRANSACTION token held at alloc_buffer host fallback"),
    (MAIN, "ggml_backend_sycl_graph_compute", "[REPLAN-TOKEN] token held in graph compute"),
    (MAIN, "ggml_backend_sycl_plan_caps_freeze", "[REPLAN-TOKEN] chunk-cap freeze outside the fixpoint"),
    (MAIN, "ggml_backend_sycl_replan_scope_open", "[REPLAN-TOKEN] growth scope not outermost: under TRANSACTION"),
    (MAIN, "ggml_backend_sycl_replan_scope_open", "[REPLAN-TOKEN] growth scope not outermost: under LOAD"),
    (MAIN, "ggml_backend_sycl_replan_scope_open", "[REPLAN-TOKEN] growth scope not outermost: under LIFECYCLE"),
    (UC_C, "ggml_sycl_replan_token", "[REPLAN-TOKEN] release proc entered with L0 held"),
)


def gate31(files, bad):
    ucc = code(files, UC_C)
    uch = code(files, UC_H)
    c = code(files, MAIN)
    # One token type, one mutex, a thread-local depth.
    if len(re.findall(r"\bclass\s+ggml_sycl_replan_token\b\s*\{", uch)) != 1:
        bad("gate 31: ggml_sycl_replan_token is not defined exactly once")
    mutexes = set()
    for path in all_sycl_sources_in(files):
        for m in re.finditer(r"\bstd::(?:recursive_)?(?:shared_)?(?:timed_)?mutex\s+(\w*(?:replan|re_plan)\w*)",
                             code(files, path), re.I):
            mutexes.add(m.group(1))
    if mutexes != {"g_replan_txn_mutex"}:
        bad("gate 31: the re-plan mutexes are %s (wanted exactly g_replan_txn_mutex)" % sorted(mutexes))
    if not re.search(r"static\s+thread_local\s+int\s+g_replan_token_depth\b", ucc):
        bad("gate 31: the held depth is not a thread_local")
    # A blocked acquire is never silent and never abandoned: it waits in timed slices, logs the holder
    # at each, keeps waiting, and aborts only under STRICT.
    acq = func_bodies(ucc, "ggml_sycl_replan_token::acquire")
    ackeep = keep(files, UC_C)
    ak = ""
    for o, e in func_bodies(ackeep, "ggml_sycl_replan_token::acquire"):
        ak = text_of(ackeep, (o, e))
    if len(acq) != 1 or "g_replan_txn_mutex.lock()" in text_of(ucc, acq[0]) or \
            "g_replan_txn_mutex.try_lock_for(" not in text_of(ucc, acq[0]):
        bad("gate 31: the blocking acquire of L0 is not a timed wait (try_lock_for, never lock())")
    elif "[REPLAN-WAIT]" not in ak or "GGML_LOG_WARN" not in text_of(ucc, acq[0]) or \
            not re.search(r"ggml_sycl_strict_enabled\s*\(\s*\)\s*\)\s*\{\s*GGML_ABORT\s*\(", text_of(ucc, acq[0])):
        bad("gate 31: a blocked L0 acquire does not log the holder at each interval and abort under STRICT")
    wd = func_bodies(ucc, "ggml_sycl_wait_watch::ggml_sycl_wait_watch")
    if len(wd) != 1 or "[REPLAN-WAIT]" not in text_of(ackeep, wd[0]) or "GGML_ABORT" not in text_of(ucc, wd[0]) or \
            "ggml_sycl_strict_enabled" not in text_of(ucc, wd[0]):
        bad("gate 31: the wait watch does not log at each interval and abort under STRICT")
    sw = func_bodies(c, "ggml_backend_sycl_synchronize_for_replan")
    if len(sw) != 1:
        bad("gate 31: ggml_backend_sycl_synchronize_for_replan has %d definitions" % len(sw))
    else:
        swt = text_of(c, sw[0])
        first_wait = swt.find("->wait()")
        w_at = swt.find("ggml_sycl_wait_watch")
        if w_at < 0 or first_wait < 0 or w_at > first_wait or swt.count("watch.site(") < 6:
            bad("gate 31: synchronize_for_replan's queue waits are not under a wait watch that names each wait")
    # No token is a class member.
    for path in all_sycl_sources_in(files):
        cc = code(files, path)
        for head, o, e in top_spans(cc):
            hdr = strip_pp(cc[head:o])
            # The one sanctioned owner of a token outside an automatic variable is the public scope's
            # state: ggml_backend_sycl_replan_scope_open hands it to a caller that cannot hold a C++ object
            # across the backend boundary, and gate 31's scope clause below pins its shape.
            if "ggml_backend_sycl_replan_scope_state" in hdr:
                continue
            if re.search(r"\b(struct|class)\s+\w+[^;(]*$", hdr) and "ggml_sycl_replan_token" not in hdr.split("{")[0].split()[-1:]:
                if re.search(r"(?m)^\s*(?:mutable\s+)?(?:ggml_sycl::)?ggml_sycl_replan_token\s+\w+\s*(?:;|\{|=)", cc[o:e]):
                    bad("gate 31: a class member of type ggml_sycl_replan_token in %s" % path)
    # The public scope (ggml-sycl.h): llama's L0 for the fixpoint and the growth path. One owner struct holds
    # the token, only the TRANSACTION kind is open to a caller, and both functions are served as procs.
    so = func_bodies(c, "ggml_backend_sycl_replan_scope_open")
    sx = func_bodies(c, "ggml_backend_sycl_replan_scope_close")
    if len(so) != 1 or len(sx) != 1:
        bad("gate 31: the public replan scope has %d open and %d close definitions" % (len(so), len(sx)))
    else:
        sot = text_of(c, so[0])
        if not re.search(r"if\s*\(\s*kind\s*!=\s*GGML_SYCL_REPLAN_SCOPE_TRANSACTION\s*\)\s*\{\s*return\s+nullptr\s*;", sot):
            bad("gate 31: the public replan scope accepts a kind other than TRANSACTION")
        if not re.search(r"new\s+ggml_backend_sycl_replan_scope_state\s*\(\s*(?:ggml_sycl::)?GGML_SYCL_REPLAN_KIND_TRANSACTION\s*\)", sot):
            bad("gate 31: the public replan scope does not construct a TRANSACTION token")
        if "delete" not in text_of(c, sx[0]):
            bad("gate 31: the public replan scope's close does not release the owner")
    if len(re.findall(r"\bstruct\s+ggml_backend_sycl_replan_scope_state\s*\{", c)) != 1:
        bad("gate 31: ggml_backend_sycl_replan_scope_state is not defined exactly once")
    else:
        sb = struct_body(c, "ggml_backend_sycl_replan_scope_state")
        if sb is None or len(re.findall(r"\bggml_sycl_replan_token\s+\w+\s*(?:;|\{)", text_of(c, sb))) != 1:
            bad("gate 31: the public replan scope's state does not hold exactly one token")
    for proc in ("ggml_backend_sycl_replan_scope_open", "ggml_backend_sycl_replan_scope_close"):
        if not re.search(r"strcmp\(name,\s*\"%s\"\)\s*==\s*0\)\s*\{\s*return\s*\(void\s*\*\)\s*%s;" % (proc, proc), keep(files, MAIN)):
            bad("gate 31: %s is not served by the backend's proc table" % proc)
    # Entries.
    for fn, kind in TOKEN_ENTRIES.items():
        bodies = func_bodies(c, fn)
        if len(bodies) != 1:
            bad("gate 31: %s has %d definitions" % (fn, len(bodies)))
            continue
        b = text_of(c, bodies[0])
        m = re.search(r"\bggml_sycl_replan_token\s+\w+\s*\(\s*GGML_SYCL_REPLAN_KIND_(\w+)", b)
        if not m:
            bad("gate 31: %s does not construct the L0 token" % fn)
            continue
        if m.group(1) != kind:
            bad("gate 31: %s constructs a %s token (wanted %s)" % (fn, m.group(1), kind))
        before = b[1:m.start()]
        if re.search(r"[;)]\s*$", before.strip()) and re.search(r"\w+\s*\([^)]*\)\s*;", before):
            bad("gate 31: %s runs a statement before it constructs its token" % fn)
    # The wrappers of that impl hold no token of their own, so each must reach the impl, or the entry would be a
    # public door with no token at all.
    for fn in ("ggml_backend_sycl_set_runtime_context_for_model", "ggml_backend_sycl_set_runtime_context_desc"):
        bodies = func_bodies(c, fn)
        if len(bodies) != 1:
            bad("gate 31: %s has %d definitions" % (fn, len(bodies)))
        elif not re.search(r"[;{}]\s*return\s+ggml_sycl_set_runtime_context_for_model_impl\(", text_of(c, bodies[0])):
            bad("gate 31: %s does not return ggml_sycl_set_runtime_context_for_model_impl(...)" % fn)
    # The witness macro and its sites are always compiled.
    mm = re.search(r"#define\s+GGML_SYCL_WITNESS\s*\(.*?(?:\n(?!\s*#)[^\n]*\\)*\n[^\n]*\n", files[UC_H])
    macro = files[UC_H][files[UC_H].find("#define GGML_SYCL_WITNESS"):files[UC_H].find("#define GGML_SYCL_WITNESS") + 400]
    if "#define GGML_SYCL_WITNESS" not in files[UC_H] or re.search(r"NDEBUG|\bassert\s*\(", macro.split("bool arena_pp")[0]):
        bad("gate 31: the witness macro is missing or depends on NDEBUG / assert")
    sw = re.search(r"static bool ggml_sycl_witness_switch\(\)\s*\{(.*?)\n\}", files[UC_C], re.S)
    if not sw or "NDEBUG" in sw.group(1) or "GGML_SYCL_PRIVATE_TESTING" not in sw.group(1) or \
            "GGML_SYCL_WITNESS_CHECKS" not in sw.group(1):
        bad("gate 31: the witness switch reads something other than GGML_SYCL_PRIVATE_TESTING and GGML_SYCL_WITNESS_CHECKS")
    for path in all_sycl_sources_in(files):
        if re.search(r"#define\s+GGML_SYCL_PLAN_CHECK\b", files[path]):
            bad("gate 31: a second witness macro GGML_SYCL_PLAN_CHECK exists in %s" % path)
    for path, fn, msg in WITNESS_SITES:
        cc = code(files, path)
        kk = keep(files, path)
        bodies = func_bodies(cc, fn)
        if not bodies:
            bad("gate 31: %s is absent" % fn)
            continue
        text_k = "".join(text_of(kk, b) for b in bodies)
        text_c = "".join(text_of(cc, b) for b in bodies)
        if msg not in text_k:
            bad("gate 31: the witness message %r is missing from %s" % (msg, fn))
            continue
        at = text_k.find(msg)
        window = text_k[max(0, at - 400):at]
        if "GGML_SYCL_WITNESS(" not in window:
            bad("gate 31: the check for %r is not a GGML_SYCL_WITNESS" % msg)
        if re.search(r"\bassert\s*\(|NDEBUG", text_c):
            bad("gate 31: %s uses assert( or NDEBUG next to its witness" % fn)
    # load_end's bodies never overlap: a witness counts them, and load_end
    # constructs it right after the LOAD token (so the count is taken under L0).
    kk = keep(files, MAIN)
    if not re.search(r"struct\s+ggml_sycl_load_end_body_witness\s*\{\s*ggml_sycl_load_end_body_witness\(\)\s*\{[^}]*"
                     r"GGML_SYCL_WITNESS\(\s*running\s*==\s*0\s*,\s*\"\[REPLAN-TOKEN\] two load_end bodies overlapped\"\s*\)",
                     kk, re.S):
        bad("gate 31: the load_end overlap witness is missing, weakened or renamed")
    lb = func_bodies(code(files, MAIN), "ggml_backend_sycl_model_load_end")
    lt = text_of(code(files, MAIN), lb[0]) if len(lb) == 1 else ""
    if not re.search(r"ggml_sycl_replan_token\s+l0\(GGML_SYCL_REPLAN_KIND_LOAD\);\s*ggml_sycl_load_end_body_witness\s+\w+\s*;", lt):
        bad("gate 31: load_end does not construct its overlap witness right after the LOAD token")
    # The preload's check is held(LOAD), not weakened to any kind.
    m = re.search(r"GGML_SYCL_WITNESS\(\s*((?:ggml_sycl::)?ggml_sycl_replan_token_held\([^)]*\))\s*,\s*\"\[REPLAN-TOKEN\] preload without a LOAD token\"",
                  kk)
    if not m or "GGML_SYCL_REPLAN_KIND_LOAD" not in m.group(1):
        bad("gate 31: the preload's witness is not held(LOAD) with its literal message")
    # The graph-compute check is held(), any kind.
    gm = re.search(r"GGML_SYCL_WITNESS\(\s*!(?:ggml_sycl::)?ggml_sycl_replan_token_held\(\s*\)\s*,\s*\"\[REPLAN-TOKEN\] token held in graph compute\"", kk)
    if not gm:
        bad("gate 31: the graph-compute witness is not `!held()` (any kind)")


# --------------------------------------------------------------------------------------
# gate 31, owner threading (rulings M246, option A): the dispatch owner
# --------------------------------------------------------------------------------------
OWNER_READERS = ("ggml_sycl_resolve_moe_expert_route_core", "ggml_sycl_collect_decode_secondary_candidates")
ENSURE = "ggml_sycl_ensure_moe_secondary_queues_for_plan"
# every function allowed to call ENSURE, with the owner argument it must pass
ENSURE_CALLERS = {
    "ggml_sycl_resolve_moe_expert_route_core": "ggml_sycl_dispatch_owner()",
    "ggml_sycl_collect_decode_secondary_candidates": "ggml_sycl_dispatch_owner()",
    "ggml_sycl_mul_mat_id": "mmid_owner_ptr",
}
# what runs on another thread: the TLS must never be read inside, nor a function that reads it called
CROSS_THREAD_LAUNCHERS = (r"\bhost_task\s*\(", r"\bstd::async\s*\(", r"\bstd::thread\s*\(", r"\.submit\s*\(", r"\.enqueue\s*\(")
TLS_REACHERS = ("ggml_sycl_dispatch_owner", "ggml_sycl_resolve_moe_expert_route", "ggml_sycl_resolve_moe_expert_route_core",
                "ggml_sycl_collect_decode_secondary_candidates", "ggml_sycl_choose_decode_secondary_device")


def gate31_owner(files, bad):
    c = code(files, MAIN)
    k = keep(files, MAIN)
    # --- the plan scope accessor L4's claim hook consumes: a one-line read of the thread's scope
    if not re.search(r"static\s+ggml_sycl_plan_scope\s*\*\s*ggml_sycl_plan_scope_current\s*\(\s*\)\s*\{\s*return\s+g_plan_scope\s*;\s*\}", c):
        bad("gate 31 (owner): ggml_sycl_plan_scope_current() is not a one-line read of g_plan_scope")
    # --- the scope class: one constructor input, execution_current_owner, abort on a different nested owner
    sc = struct_body(c, "ggml_sycl_dispatch_owner_scope")
    if sc is None:
        bad("gate 31 (owner): class ggml_sycl_dispatch_owner_scope is absent")
        return
    st = text_of(c, sc)
    sk = text_of(k, struct_body(k, "ggml_sycl_dispatch_owner_scope"))
    ctor = re.search(r"explicit\s+ggml_sycl_dispatch_owner_scope\s*\(\s*const\s+ggml_backend_sycl_context\s*\*\s*ctx\s*\)\s*\{", st)
    if not ctor:
        bad("gate 31 (owner): the scope's constructor does not take exactly one `const ggml_backend_sycl_context * ctx`")
    else:
        cb = st[ctor.end() - 1:close_of(st, ctor.end() - 1) + 1]
        if not re.search(r"\bbound\s*=\s*ggml_sycl_execution_current_owner\s*\(\s*ctx\s*,\s*owner\s*\)", cb):
            bad("gate 31 (owner): the scope's owner does not come from ggml_sycl_execution_current_owner(ctx, owner)")
        if re.search(r"global_registry|current_active_token|g_placement_publication|ggml_sycl_global_plan|getenv", cb):
            bad("gate 31 (owner): the scope's constructor reads an owner from something other than the context")
        if not re.search(r"g_dispatch_owner_depth\s*>\s*0\s*&&\s*\(\s*bound\s*!=\s*g_dispatch_owner_bound\s*\|\|\s*\(\s*bound\s*&&\s*!\(\s*owner\s*==\s*g_dispatch_owner\s*\)\s*\)\s*\)", cb):
            bad("gate 31 (owner): a nested scope's owner is not compared with the bound one")
    if 'GGML_ABORT("[DISPATCH-OWNER] nested scope binds a different owner")' not in sk:
        bad("gate 31 (owner): a nested scope with a different owner does not abort with its literal")
    # the thread-locals are written inside the class and nowhere else
    outside = c[:sc[0]] + c[sc[1] + 1:]
    for m in re.finditer(r"(?<![\w])g_dispatch_owner(?:_bound|_depth)?\s*(?:=(?!=)|\+\+|--)", outside):
        line = outside[outside.rfind("\n", 0, m.start()) + 1:outside.find("\n", m.start())]
        if "thread_local" not in line:
            bad("gate 31 (owner): %r writes the dispatch owner outside its scope class" % line.strip()[:60])
    # --- construction: exactly one root, graph compute, from the backend's context
    sites = [m for m in re.finditer(r"\bggml_sycl_dispatch_owner_scope\s+\w+\s*\(", c)]
    if len(sites) != 1:
        bad("gate 31 (owner): %d dispatch owner scopes are constructed (wanted exactly 1, in graph compute)" % len(sites))
    for m in sites:
        if enclosing_top(c, m.start()) != "ggml_backend_sycl_graph_compute":
            bad("gate 31 (owner): a dispatch owner scope is constructed outside graph compute")
        arg = c[m.end() - 1:close_of(c, m.end() - 1) + 1]
        if not re.fullmatch(r"\(\s*backend\s*\?\s*static_cast<const\s+ggml_backend_sycl_context\s*\*>\s*\(\s*backend->context\s*\)\s*:\s*nullptr\s*\)", arg):
            bad("gate 31 (owner): the scope is not constructed from the backend's own context")
    # --- readers: only the two ctx-less chains, never inside a lambda
    for m in re.finditer(r"(?<![\w:])ggml_sycl_dispatch_owner\s*\(", c):
        fn = enclosing_top(c, m.start())
        if fn is None or fn == "ggml_sycl_dispatch_owner":  # a declarator or the reader's own definition
            continue
        if fn not in OWNER_READERS:
            bad("gate 31 (owner): ggml_sycl_dispatch_owner() is read in %s, outside any bound scope's chains" % fn)
    # no TLS read, and no function that reads it, inside anything that runs on another thread
    for pat in CROSS_THREAD_LAUNCHERS:
        for path in all_sycl_sources_in(files):
            cc = code(files, path)
            for m in re.finditer(pat, cc):
                op = cc.find("(", m.start())
                span = cc[op:close_of(cc, op) + 1] if op >= 0 else ""
                for name in TLS_REACHERS:
                    if re.search(r"(?<![\w])" + name + r"\s*\(", span):
                        bad("gate 31 (owner): %s:%d reaches the thread-local dispatch owner through %s inside work that runs on "
                            "another thread" % (path, line_of(cc, m.start()), name))
    # --- into_empty: one critical section, the owner's own snapshot, this cache only, no device-global
    ie = func_bodies(c, "ggml_sycl_republish_current_plan_into_empty")
    if len(ie) != 1:
        bad("gate 31 (owner): ggml_sycl_republish_current_plan_into_empty has %d definitions" % len(ie))
    else:
        t = text_of(c, ie[0])
        tk = text_of(k, ie[0])
        locks = re.findall(r"std::lock_guard<std::mutex>\s+\w+\s*\(\s*g_tensor_inventory_mutex\s*\)", t)
        if len(locks) != 1 or re.search(r"\b(unique_lock|scoped_lock)\b", t):
            bad("gate 31 (owner): into_empty is not exactly one g_tensor_inventory_mutex critical section")
        order = [t.find(x) for x in ("g_tensor_inventory_mutex", "lifecycle_select_placement_plan(", "get_placement_plan_snapshot()",
                                     "set_placement_plan_snapshot(")]
        order.append(t.find("installed->version", max(order[-1], 0)))
        if min(order) < 0 or order != sorted(order):
            bad("gate 31 (owner): into_empty does not lock, select the owner's snapshot, re-check emptiness, install, "
                "then compare identity, in that order")
        # What each cache state gets: an empty cache gets the owner's snapshot, a cache that
        # already holds a plan is left untouched (and a foreign load's plan is witnessed), and a cache that is not
        # one of the plan's devices gets nullptr.  The hint in front of the lock is keyed by owner and set only for
        # a non-participant.
        if "set_placement_plan_snapshot(participates ? selected : nullptr)" not in t.replace("\n", " "):
            bad("gate 31 (owner): into_empty does not install `participates ? selected : nullptr`")
        non_empty = re.search(r"if\s*\(\s*current\s*&&\s*current->plan\s*&&\s*!current->plan->entries\.empty\(\)\s*\)\s*\{", t)
        if not non_empty:
            bad("gate 31 (owner): into_empty does not test for an already-filled cache")
        else:
            nb = t[non_empty.end() - 1:close_of(t, non_empty.end() - 1) + 1]
            if "set_placement_plan_snapshot(" in nb or "return ggml_sycl_into_empty_result::NOOP" not in nb:
                bad("gate 31 (owner): into_empty does not leave a filled cache untouched (NOOP, no install)")
            if "set_into_empty_foreign_key(" not in nb or "into_empty_foreign_key()" not in nb or \
                    "into_empty: this device's cache holds" not in tk:
                bad("gate 31 (owner): into_empty does not witness a foreign load's plan in a filled cache")
            if non_empty.start() > t.find("set_placement_plan_snapshot("):
                bad("gate 31 (owner): the filled-cache test comes after the install")
            # the once-per-owner WARN is keyed by the owner alone: a key that mixed the publish generation would
            # warn once per owner per publication
            if not re.search(r"warn_key\s*=\s*ggml_sycl_into_empty_foreign_warn_key\s*\(\s*owner\s*\)\s*;", nb) or \
                    "set_into_empty_foreign_key(warn_key)" not in nb or "into_empty_foreign_key() != warn_key" not in nb:
                bad("gate 31 (owner): the foreign-plan WARN is not deduplicated on its own owner-only key")
            wf = func_bodies(c, "ggml_sycl_into_empty_foreign_warn_key")
            wt = text_of(c, wf[0]) if len(wf) == 1 else ""
            if not (re.search(r"owner\.model\.value", wt) and re.search(r"owner\.load\.value", wt)) or \
                    re.search(r"publish_gen|epoch|tenant_publish_gen", wt):
                bad("gate 31 (owner): the foreign-plan WARN key is not the owner's model and load alone")
        np_ = re.search(r"if\s*\(\s*!participates\s*\)\s*\{(.*?)\n    \}", t, re.S)
        if not np_ or "return ggml_sycl_into_empty_result::NOOP" not in np_.group(1) or \
                "set_into_empty_skip_key(skip_key)" not in np_.group(1):
            bad("gate 31 (owner): a non-participant does not get NOOP with its skip key set")
        if len(re.findall(r"set_into_empty_skip_key\s*\(", t)) != 1:
            bad("gate 31 (owner): the skip key is set somewhere other than the non-participant branch")
        hint = t.find("into_empty_skip_key() == skip_key")
        if hint < 0 or hint > t.find("std::lock_guard") or "return ggml_sycl_into_empty_result::NOOP" not in t[hint:hint + 120]:
            bad("gate 31 (owner): the owner-keyed hint is not read, with a NOOP, before the lock")
        # the key names the owning load AND the tenant publish generation, read before the lock: every plan
        # publication bumps the generation -- a stable-MMID re-publish too, which keeps the plan version yet can
        # add this device through kv_device -- so a stale "not a participant" never matches
        if not re.search(r"skip_key\s*=\s*ggml_sycl_into_empty_skip_key\s*\(\s*owner\s*,\s*"
                         r"ggml_sycl_tenant_publish_gen\s*\(\s*\)\s*\)\s*;", t) or \
                t.find("ggml_sycl_tenant_publish_gen(") > t.find("std::lock_guard"):
            bad("gate 31 (owner): the hint key is not built from the owner's load and the tenant publish generation "
                "before the lock")
        kf = func_bodies(c, "ggml_sycl_into_empty_skip_key")
        kt = text_of(c, kf[0]) if len(kf) == 1 else ""
        if not (re.search(r"owner\.model\.value", kt) and re.search(r"owner\.load\.value", kt) and
                re.search(r"\bpublish_gen\b", kt) and re.search(r"\|\s*1ULL", kt)):
            bad("gate 31 (owner): the hint key function does not mix the owner's model, load and the publish generation")
        if not all(x in t for x in ("installed->version != selected->version", "installed->model_id != selected->model_id",
                                    "installed->load_txn_id != selected->load_txn_id")):
            bad("gate 31 (owner): into_empty does not compare version, model_id and load_txn_id of the installed snapshot")
        for tok in ("g_placement_publication", "g_model_n_layer", "g_placement_kv_info", "ggml_sycl_publish_plan_locked",
                    "ggml_sycl_publish_prepared_plan_locked", "publish_cache_first_global_last", "ggml_sycl_global_plan_snapshot",
                    "ggml_sycl_republish_current_plan(", "ggml_sycl_has_global_plan", "ggml_sycl_cache_plan_owner"):
            if tok in t:
                bad("gate 31 (owner): into_empty reaches %s (it writes no device-global and reads no global authority)" % tok)
        if "[CONTEXT-PLAN-BUG] into_empty: no snapshot" not in tk or "return ggml_sycl_into_empty_result::REFUSED" not in t:
            bad("gate 31 (owner): the null selection of a bound owner is not a [CONTEXT-PLAN-BUG] refusal")
        m = re.search(r"if\s*\(\s*!selected\s*\)\s*\{(.*?)\n    \}", t, re.S)
        if not m or "REFUSED" not in m.group(1):
            bad("gate 31 (owner): the null selection branch does not refuse")
    # --- the one remaining plain republish is the preload's
    pubs = [m.start() for m in re.finditer(r"(?<![\w:])ggml_sycl_republish_current_plan\s*\(\s*\)\s*;", c)]
    if len(pubs) != 1 or "preload without a LOAD token" not in text_of(k, func_bodies(k, enclosing_top(c, pubs[0]))[0]):
        bad("gate 31 (owner): ggml_sycl_republish_current_plan() has %d callers (wanted exactly the preload's)" % len(pubs))
    # --- ensure: tri-state, owner parameter, into_empty, no global-plan guard, no plain republish
    eb = func_bodies(c, ENSURE)
    if len(eb) != 1:
        bad("gate 31 (owner): %s has %d definitions" % (ENSURE, len(eb)))
    else:
        t = text_of(c, eb[0])
        hdr = c[max(0, eb[0][0] - 400):eb[0][0]]
        if "ggml_sycl_secondary_queue_status" not in hdr or "const ggml_sycl::lifecycle::ModelToken * owner" not in hdr:
            bad("gate 31 (owner): ensure does not return the tri-state or does not take the owner")
        if len(re.findall(r"ggml_sycl_republish_current_plan_into_empty\s*\(", t)) != 2 or \
                len(re.findall(r"ggml_sycl_into_empty_result::REFUSED\)\s*\{\s*return GGML_SYCL_SECONDARY_QUEUE_REFUSED", t)) != 2:
            bad("gate 31 (owner): ensure's two installs do not each propagate a refusal")
        if re.search(r"ggml_sycl_has_global_plan|ggml_sycl_republish_current_plan\s*\(\s*\)|return\s+(?:true|false)\s*;", t):
            bad("gate 31 (owner): ensure still reads the global plan, republishes it, or returns a bool")
    # --- every caller of ensure: the three listed functions, the right owner, consumed (never dropped, never a bool)
    total = 0
    for m in re.finditer(r"(?<![\w])" + ENSURE + r"\s*\(", c):
        fn = enclosing_top(c, m.start())
        if fn is None or fn == ENSURE:  # a declarator or the definition itself
            continue
        total += 1
        if fn not in ENSURE_CALLERS:
            bad("gate 31 (owner): %s is called from %s, which has no owner source" % (ENSURE, fn))
            continue
        op = c.find("(", m.start())
        args = c[op + 1:close_of(c, op)]
        want = ENSURE_CALLERS[fn]
        if not re.search(r",\s*" + re.escape(want) + r"\s*$", args.strip()):
            bad("gate 31 (owner): a call of %s in %s does not pass %s" % (ENSURE, fn, want))
        before = c[max(0, m.start() - 160):m.start()]
        if re.search(r"\(void\)\s*(?:ggml_sycl::)?$", before):
            bad("gate 31 (owner): a call of %s in %s discards its status" % (ENSURE, fn))
        elif not re.search(r"(?:switch\s*\(\s*|ggml_sycl_secondary_queue_ready_or_throw\s*\(\s*(?:ggml_sycl::)?|queue_status\s*=\s*)$",
                           re.sub(r"(?:ggml_sycl::)?$", "", before) + "") and \
                not re.search(r"(?:switch\s*\(\s*|ready_or_throw\s*\(\s*|queue_status\s*=\s*)(?:ggml_sycl::)?\s*$", before):
            bad("gate 31 (owner): a call of %s in %s is not consumed by a switch, ready_or_throw or an explicit status" % (ENSURE, fn))
    if total != 6:
        bad("gate 31 (owner): %s has %d callers (the census is six)" % (ENSURE, total))
    # mul_mat_id reads the owner from its own context, once
    mm = func_bodies(c, "ggml_sycl_mul_mat_id")
    if len(mm) != 1 or not re.search(r"\bmmid_owner_bound\s*=\s*ggml_sycl_execution_current_owner\s*\(\s*&ctx\s*,\s*mmid_owner\s*\)",
                                      text_of(c, mm[0])):
        bad("gate 31 (owner): ggml_sycl_mul_mat_id does not read its owner from ggml_sycl_execution_current_owner(&ctx, ...)")
    # --- the refusal channel
    rf = func_bodies(c, "ggml_sycl_secondary_queue_refused_fail")
    if len(rf) != 1 or "throw ggml_sycl_fallback_error(" not in text_of(c, rf[0]) or \
            "[[noreturn]]" not in c[max(0, rf[0][0] - 120):rf[0][0]]:
        bad("gate 31 (owner): the refusal helper is not a [[noreturn]] throw of ggml_sycl_fallback_error")
    ro = func_bodies(c, "ggml_sycl_secondary_queue_ready_or_throw")
    if len(ro) != 1 or not re.search(r"case GGML_SYCL_SECONDARY_QUEUE_REFUSED:\s*ggml_sycl_secondary_queue_refused_fail\(\)\s*;",
                                      text_of(c, ro[0])):
        bad("gate 31 (owner): ready_or_throw does not fail on REFUSED")
    wr = func_bodies(c, "ggml_sycl_resolve_moe_expert_route")
    if len(wr) != 1 or not re.search(r"route\.kind\s*==\s*moe_expert_route_kind::REFUSED\s*\)\s*\{\s*ggml_sycl_secondary_queue_refused_fail\(\)",
                                      text_of(c, wr[0])):
        bad("gate 31 (owner): the resolver wrapper does not turn a REFUSED route into the failure")
    core = func_bodies(c, "ggml_sycl_resolve_moe_expert_route_core")
    if len(core) == 1:
        t = text_of(c, core[0])
        if not re.search(r"case GGML_SYCL_SECONDARY_QUEUE_REFUSED:\s*route\.kind\s*=\s*moe_expert_route_kind::REFUSED\s*;[^}]*return route;", t):
            bad("gate 31 (owner): the resolver core does not return a REFUSED route on a refusal")
        for m in re.finditer(r"(?<![\w])ggml_sycl_resolve_moe_expert_route_core\s*\(", c):
            if enclosing_top(c, m.start()) not in (None, "ggml_sycl_resolve_moe_expert_route",
                                                   "ggml_sycl_resolve_moe_expert_route_core"):
                bad("gate 31 (owner): the resolver core is called from %s (only the wrapper may, so no consumer sees REFUSED)" %
                    enclosing_top(c, m.start()))
    # a rejected choice BEFORE the chooser, never queue_available = false
    ap = re.search(r"auto append_retained_operand = \[&\]", c)
    if not ap:
        bad("gate 31 (owner): append_retained_operand is absent")
    else:
        lam = c[ap.start():close_of(c, c.find("{", ap.end())) + 1]
        i_ref = lam.find("moe_batch_reject_reason::PLAN_OWNER_REFUSED")
        i_cho = lam.find("choose_moe_batch_executor(")
        if i_ref < 0 or i_cho < 0 or i_ref > i_cho or not re.search(r"queue_status\s*==\s*GGML_SYCL_SECONDARY_QUEUE_REFUSED", lam):
            bad("gate 31 (owner): a refused owner is not a rejected choice made before the executor chooser")
        if re.search(r"queue_available\s*=\s*false", lam):
            bad("gate 31 (owner): append_retained_operand assigns queue_available = false")
    # the mul_mat_id install site fails on a refusal
    if not re.search(r"ggml_sycl_republish_current_plan_into_empty\(route_cache,\s*\*mmid_owner_ptr\)\s*==\s*"
                     r"ggml_sycl_into_empty_result::REFUSED\)\s*\{\s*ggml_sycl::ggml_sycl_secondary_queue_refused_fail\(\)", c):
        bad("gate 31 (owner): the mul_mat_id install does not fail on a refusal")
    # every catch of the fallback error rethrows, except graph compute's top-level handler
    for m in re.finditer(r"catch\s*\(\s*const\s+ggml_sycl_fallback_error\s*&[^)]*\)\s*\{", c):
        body = c[m.end() - 1:close_of(c, m.end() - 1) + 1]
        if enclosing_top(c, m.start()) != "ggml_backend_sycl_graph_compute" and not re.search(r"\bthrow\s*;", body):
            bad("gate 31 (owner): a catch of ggml_sycl_fallback_error in %s does not rethrow" % enclosing_top(c, m.start()))
    # the host test installs nothing: it reads the PRIVATE_TESTING install counter around its resolves
    ht = func_bodies(c, "test_moe_storage_handle_first_route_and_negatives")
    if len(ht) != 1 or text_of(c, ht[0]).count("ggml_sycl_into_empty_install_count()") != 2:
        bad("gate 31 (owner): the context-free host test does not bound the install counter")


# --------------------------------------------------------------------------------------
# gate 36: the override, the latch, the measure backend
# --------------------------------------------------------------------------------------
ACCESSORS = ("ggml_sycl_global_plan_snapshot", "ggml_sycl_identity_plan_snapshot", "global_placement_plan_owner",
             "cache_placement_coherence")
LATCH_READERS = ("ggml_backend_sycl_device_supports_op", "ggml_sycl_op_is_host_gate_activation_chain",
                 "ggml_sycl_op_is_planned_on_host", "ggml_backend_sycl_device_offload_op")


def gate36(files, bad):
    c = code(files, MAIN)
    # The four accessors consult the override first.
    for fn in ACCESSORS:
        bodies = func_bodies(c, fn)
        if len(bodies) != 1:
            bad("gate 36: accessor %s has %d definitions" % (fn, len(bodies)))
            continue
        b = text_of(c, bodies[0])
        stripped = b[1:].lstrip()
        if not re.match(r"(?:placement_cache_read\s+result;\s*result\.owner\s*=\s*[^;]*;\s*)?if\s*\(\s*g_measure_plan_override\b|"
                        r"if\s*\(\s*g_measure_plan_override\b", stripped) and "g_measure_plan_override" not in b[:260]:
            bad("gate 36: accessor %s does not consult the measure plan override first" % fn)
    # Every other read of the publication is an accessor's, or the publish store.
    for m in re.finditer(r"\bg_placement_publication\b", c):
        fn = enclosing_top(c, m.start())
        if fn is None:
            continue  # the definition
        tail = fn.split("::")[-1]
        if tail in ACCESSORS or tail == "ggml_sycl_publish_prepared_plan_locked":
            continue
        if re.match(r"\s*\)?\s*,\s*snapshot", c[m.end():m.end() + 40]) and "atomic_store" in c[max(0, m.start() - 40):m.start()]:
            continue
        bad("gate 36: g_placement_publication is read in %s, which is not one of the four accessors" % fn)
    # bound_load_candidate has the named list.
    for m in re.finditer(r"ggml_sycl_bound_load_candidate\s*\(", c):
        fn = enclosing_top(c, m.start())
        if fn is None:
            continue
        if fn.split("::")[-1] not in ACCESSORS + ("ggml_sycl_host_row_authorized",
                                                   "ggml_sycl_register_buffer_tensor_provenance",
                                                   "ggml_sycl_bound_load_candidate"):
            bad("gate 36: ggml_sycl_bound_load_candidate is called from %s, which is not on the named list" % fn)
    # The latch.
    for fn in LATCH_READERS:
        for b in func_bodies(c, fn):
            t = text_of(c, b)
            if "g_moe_multi_gpu_active" in t:
                bad("gate 36: %s reads g_moe_multi_gpu_active directly" % fn)
        if not func_bodies(c, fn):
            bad("gate 36: %s is absent" % fn)
    if "ggml_sycl_moe_multi_gpu_for_executor()" not in c:
        bad("gate 36: the latch reader ggml_sycl_moe_multi_gpu_for_executor is never called")
    fe = body_text(c, "ggml_sycl_moe_multi_gpu_for_executor", bad, "gate 36")
    if fe and ("g_measure_plan_override" not in fe or "ggml_sycl_moe_multi_gpu_wanted" not in fe or
               "g_moe_multi_gpu_active" not in fe):
        bad("gate 36: the latch reader does not choose between the override's plan and the latch")
    wn = [b for b in func_bodies(c, "compute_and_store_plan_for_inventory")]
    if not wn or "ggml_sycl_moe_multi_gpu_wanted" not in text_of(c, wn[0]):
        bad("gate 36: the latch writer does not call ggml_sycl_moe_multi_gpu_wanted for its condition")
    # The override's writers.
    if not re.search(r"static\s+thread_local\s+std::shared_ptr<const\s+ggml_sycl::lifecycle_plan_snapshot>\s+"
                     r"g_measure_plan_override\b", c):
        bad("gate 36: the measure plan override is not a thread_local shared_ptr")
    for m in re.finditer(r"\bg_measure_plan_override\s*(?:=(?!=)|\.reset\s*\(|\.swap\s*\()", c):
        fn = enclosing_top(c, m.start())
        if fn not in ("ggml_backend_sycl_measure_plan_override_install", "ggml_backend_sycl_measure_plan_override_clear"):
            bad("gate 36: g_measure_plan_override is written in %s" % fn)
    inst = body_text(c, "ggml_backend_sycl_measure_plan_override_install", bad, "gate 36")
    kinst = text_of(keep(files, MAIN), only_body(keep(files, MAIN), "ggml_backend_sycl_measure_plan_override_install", [], "")) \
        if func_bodies(keep(files, MAIN), "ggml_backend_sycl_measure_plan_override_install") else ""
    if inst:
        first = inst.find("g_measure_plan_override")
        if first < 0 or "return false" not in inst[:inst.find("g_measure_plan_override =") if "g_measure_plan_override =" in inst else 0]:
            bad("gate 36: install does not refuse a nest before it writes")
        if 'GGML_LOG_WARN("[CONTEXT-PLAN-BUG] measure plan override nested' not in kinst or \
                'GGML_ABORT("[CONTEXT-PLAN-BUG] measure plan override nested' not in kinst:
            bad("gate 36: install lacks its nesting witness")
    for fn in ("ggml_backend_sycl_measure_plan_override_install", "lifecycle_make_candidate_snapshot"):
        t = ""
        for path in (MAIN, UC_C):
            cc = code(files, path)
            for b in func_bodies(cc, fn):
                t += text_of(cc, b)
        if not t:
            bad("gate 36: %s is absent" % fn)
        for banned in ("ggml_sycl_make_provisional_plan_snapshot", "lifecycle_next_plan_publication_id"):
            if banned in t:
                bad("gate 36: %s names %s" % (fn, banned))
    pub = body_text(c, "ggml_sycl_publish_prepared_plan_locked", bad, "gate 36")
    if pub:
        first = pub[1:].lstrip()
        if not re.match(r"if\s*\(\s*(?:ggml_sycl_measure_plan_override_active\s*\(\s*\)|g_measure_plan_override\b[^)]*)\)", first):
            bad("gate 36: the publish store's first statement is not the override refusal")
    # The measure backend is non-owning.
    free = body_text(c, "ggml_backend_sycl_measure_free", bad, "gate 36")
    if "ggml_backend_sycl_free" in free or "ggml_backend_sycl_interface" in free:
        bad("gate 36: the measure backend's free reaches ggml_backend_sycl_free or the SYCL interface")
    iface = re.search(r"static\s+ggml_backend_i\s+ggml_backend_sycl_measure_interface\s*=\s*\{(.*?)\};", c, re.S)
    if not iface:
        bad("gate 36: the measure backend has no interface of its own")
    else:
        raw = files[MAIN]
        start = raw.find("ggml_backend_sycl_measure_interface = {")
        slots = re.findall(r"/\*\s*\.(\w+)\s*=\s*\*/\s*([^,]+),", raw[start:raw.find("};", start)])
        if len(slots) < 2 or slots[0][0] != "get_name" or slots[1] != ("free", "ggml_backend_sycl_measure_free"):
            bad("gate 36: the measure interface's get_name or free initializer is wrong: %s" % slots[:2])
        for name, init in slots[2:]:
            if init.strip() != "NULL":
                bad("gate 36: the measure interface slot .%s is not NULL" % name)
        if len(slots) < 14:
            bad("gate 36: the measure interface lists only %d slots" % len(slots))
    init = body_text(c, "ggml_backend_sycl_measure_backend_init", bad, "gate 36")
    for banned in ("ggml_backend_sycl_init(", "g_sycl_backend_refcount", "g_backend_context_by_device",
                   "ggml_backend_sycl_context", "lifecycle", "ggml_backend_sycl_interface"):
        if banned in init:
            bad("gate 36: ggml_backend_sycl_measure_backend_init names %s" % banned)
    if "ggml_backend_sycl_measure_guid()" not in init or "ggml_backend_sycl_guid()" in init:
        bad("gate 36: the measure backend does not carry its own guid")
    # (a)'s cap: RUNTIME 0, SCRATCH the compute arena.
    cw = body_text(c, "ggml_sycl_arena_chunk_cap", bad, "gate 36")
    m = re.search(r"case\s+ggml_sycl_chunk_cap_set::PROBE_MIN\s*:(.*?)break\s*;", cw, re.S)
    if not m or "ggml_sycl_compute_arena_bytes(device)" not in m.group(1) or re.search(r"\b(runtime|kv)\s*=", m.group(1)):
        bad("gate 36: the PROBE_MIN set is not {RUNTIME 0, SCRATCH ggml_sycl_compute_arena_bytes(dev)}")
    # One fact, one source: the compute arena's size has a single reader of its variable, the function the
    # model-load reservation and the probe set both call.
    readers = []
    for path in all_sycl_sources_in(files):
        for m in re.finditer(r'getenv\s*\(\s*"GGML_SYCL_COMPUTE_ARENA_MB"\s*\)', keep(files, path)):
            readers.append((path, m.start()))
    ucc_keep = keep(files, UC_C)
    sizing = func_bodies(code(files, UC_C), "ggml_sycl_compute_arena_bytes")
    if len(readers) != 1 or readers[0][0] != UC_C or len(sizing) != 1 or \
            not any(o <= readers[0][1] <= e for o, e in sizing):
        bad("gate 36: GGML_SYCL_COMPUTE_ARENA_MB is read %d time(s); the only reader is ggml_sycl_compute_arena_bytes" %
            len(readers))
    if c.count("ggml_sycl_compute_arena_bytes(") < 2:
        bad("gate 36: the model-load reservation does not size itself with ggml_sycl_compute_arena_bytes")
    # the cache's own zone sizing is the third reader of the function: a literal here is a second source
    zb = body_text(code(files, UC_C), "ensure_planned_arena_zones", bad, "gate 36")
    if zb and not re.search(r"\bscratch_zone\s*=\s*ggml_sycl_compute_arena_bytes\s*\(\s*dev_id\s*\)\s*;", zb):
        bad("gate 36: ensure_planned_arena_zones does not size the SCRATCH zone with ggml_sycl_compute_arena_bytes(dev_id)")
    lm = body_text(c, "ggml_sycl_plan_scope_load_measure_cap", bad, "gate 36")
    if lm and re.search(r"\b2\s*\*\s*1024|\bmin\s*\(\s*2|GiB", lm):
        bad("gate 36: the load-time cap carries a literal floor")


# --------------------------------------------------------------------------------------
# gate 37: the pool phase gates
# --------------------------------------------------------------------------------------
def gate37(files, bad):
    c = code(files, POOL)
    k = keep(files, POOL)
    for fn in ("grow_zone", "grow_into"):
        bodies = func_bodies(c, fn)
        if len(bodies) != 1:
            bad("gate 37: %s has %d definitions in pinned-pool.cpp" % (fn, len(bodies)))
            continue
        t = text_of(c, bodies[0])
        tk = text_of(k, bodies[0])
        held = re.search(r"ggml_sycl_replan_token_held\s*\(\s*(?:ggml_sycl::)?GGML_SYCL_REPLAN_KIND_TRANSACTION\s*\)", t)
        phase = t.find("offload_stats_phase(")
        if not held:
            bad("gate 37: %s does not test held(TRANSACTION)" % fn)
        elif phase >= 0 and held.start() > phase:
            bad("gate 37: %s reads the phase before it tests the token" % fn)
        if re.search(r"ggml_sycl_replan_token_held\s*\(\s*\)", t):
            bad("gate 37: %s tests the token with the default kind" % fn)
        if "site=" not in tk:
            bad("gate 37: %s's warning does not name its site" % fn)


GATES = (gate3, gate23, gate29, gate30, gate31, gate31_owner, gate36, gate37)


def violations(files):
    out = []
    for g in GATES:
        g(files, out.append)
    return out


# --------------------------------------------------------------------------------------
# mutants
# --------------------------------------------------------------------------------------
def mutants(files):
    M = MAIN
    yield ("the in-scope branch reading a zone",
           edit(files, M, "ggml_sycl_plan_scope_chunk_cap(g_plan_scope, buft, ggml_sycl_chunk_cap_buft_kind::DEVICE,\n                                              ctx ? ctx->device : -1);",
                "ggml_sycl_arena_chunk_cap(0, ggml_sycl_chunk_cap_set::LEDGER_CURRENT).value.cap;", "g3a"),
           "in-scope branch")
    yield ("a second backing read in the device function",
           edit(files, M, "static size_t ggml_backend_sycl_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {",
                "static size_t ggml_backend_sycl_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {\n"
                "    (void) ggml_sycl::ggml_sycl_arena_backing(0);", "g3b"), "reads the backing kind itself")
    yield ("is_vm derived from not-NONE",
           edit(files, M, "ggml_sycl::ggml_sycl_arena_backing(device) == ggml_sycl::GGML_SYCL_ARENA_BACKING_TYPE_VM;",
                "ggml_sycl::ggml_sycl_arena_backing(device) != ggml_sycl::GGML_SYCL_ARENA_BACKING_TYPE_NONE;", "g3c"),
           "is_vm is not exactly")
    yield ("a capacity read before the is_vm test",
           edit(files, M, "ggml_sycl_arena_chunk_cap_result result;\n    result.is_vm",
                "ggml_sycl_arena_chunk_cap_result result;\n    (void) ggml_sycl::get_unified_cache_for_device(device)->zone_capacity(ggml_sycl::vram_zone_id::KV);\n    result.is_vm", "g3d"),
           "outside the is_vm branch")
    yield ("a production call of the core outside the wrapper",
           edit(files, M, "static size_t ggml_sycl_host_chunk_cap_constant() {",
                "static size_t ggml_sycl_stray_core() { return ggml_sycl_chunk_cap_core(true, 1, 1, 1, 1, 1).cap; }\nstatic size_t ggml_sycl_host_chunk_cap_constant() {", "g3e"),
           "ggml_sycl_chunk_cap_core is called from")
    yield ("COMMITTED read from the scope's load path",
           edit(files, M, "device, probe ? ggml_sycl_chunk_cap_set::PROBE_MIN : ggml_sycl_chunk_cap_set::LOAD_TO_COMMIT);",
                "device, probe ? ggml_sycl_chunk_cap_set::PROBE_MIN : ggml_sycl_chunk_cap_set::COMMITTED);", "g3f"),
           "set COMMITTED is read from")
    yield ("the core reading the backing",
           edit(files, CHUNK, "ggml_sycl_chunk_cap_result result;\n    result.cap = GGML_SYCL_CHUNK_CAP_MAX;",
                "ggml_sycl_chunk_cap_result result;\n    (void) ggml_sycl_arena_backing(0);\n    result.cap = GGML_SYCL_CHUNK_CAP_MAX;", "g3g"),
           "the core names the backing kind")
    yield ("the factory with a hand-built interface",
           edit(files, M, "/* .iface    = */ ggml_backend_sycl_buffer_type_interface,\n        /* .device   = */ nullptr,",
                "/* .iface    = */ ggml_backend_buffer_type_i{},\n        /* .device   = */ nullptr,", "g3h"),
           "does not take its interface")
    yield ("a writer of max_size_override",
           edit(files, M, "static size_t ggml_sycl_host_chunk_cap_constant() {",
                "static void ggml_sycl_stray_override(ggml_backend_sycl_buffer_type_context * c) { c->max_size_override = 4096; }\nstatic size_t ggml_sycl_host_chunk_cap_constant() {", "g3i"),
           "writes max_size_override")
    yield ("the freeze reading a zone",
           edit(files, M, "size_t value = 0;\n    if (kind == ggml_sycl_chunk_cap_buft_kind::HOST) {\n        value = ggml_sycl_host_chunk_cap_constant();",
                "size_t value = 0;\n    (void) ggml_sycl::get_unified_cache_for_device(0)->zone_capacity(ggml_sycl::vram_zone_id::KV);\n    if (kind == ggml_sycl_chunk_cap_buft_kind::HOST) {\n        value = ggml_sycl_host_chunk_cap_constant();", "g3j"),
           "reads a zone or max_size_override itself")

    yield ("graph_invalidate clearing without the predicate",
           edit(files, M, "if (!sycl_exec_graph_has_recorded_state(ctx)) {\n        return;\n    }\n    sycl_exec_graph_clear_scoped(ctx, reason ? reason : \"context-replan\");",
                "sycl_exec_graph_clear_scoped(ctx, reason ? reason : \"context-replan\");", "g23a"),
           "does not test sycl_exec_graph_has_recorded_state")
    yield ("graph_invalidate calling the global clear",
           edit(files, M, "sycl_exec_graph_clear_scoped(ctx, reason ? reason : \"context-replan\");",
                "sycl_exec_graph_clear_active(ctx, reason ? reason : \"context-replan\");", "g23b"),
           "calls the clear that carries the process-global effects")
    yield ("the scoped body reaching a global effect",
           edit(files, M, "static void sycl_exec_graph_clear_scoped(ggml_backend_sycl_context * ctx, const char * reason) {",
                "static void sycl_exec_graph_clear_scoped(ggml_backend_sycl_context * ctx, const char * reason) {\n    graph_unpin_weights(ctx);", "g23c"),
           "the scoped clear body reaches graph_unpin_weights")
    yield ("clear_active missing one effect",
           edit(files, M, "    graph_unpin_moe_experts(ctx);\n    graph_unpin_weights(ctx);\n}\n\n// Owner-targeted",
                "    graph_unpin_weights(ctx);\n}\n\n// Owner-targeted", "g23d"),
           "does not call graph_unpin_moe_experts exactly once")
    yield ("a fourteenth clear_active caller (one past the thirteen listed)",
           edit(files, M, "static void ggml_sycl_release_graph_leases_for_owner(ggml_sycl::lifecycle::ModelToken owner) noexcept {",
                "static void ggml_sycl_stray_clear(ggml_backend_sycl_context * c) { sycl_exec_graph_clear_active(c, \"stray\"); }\n"
                "static void ggml_sycl_release_graph_leases_for_owner(ggml_sycl::lifecycle::ModelToken owner) noexcept {", "g23e"),
           "has 14 callers")

    yield ("an unlisted host_task",
           edit(files, M, "static size_t ggml_sycl_host_chunk_cap_constant() {",
                "static void ggml_sycl_stray_ht(sycl::queue & q) { q.submit([&](sycl::handler & cgh) { cgh.host_task([=]() {}); }); }\n"
                "static size_t ggml_sycl_host_chunk_cap_constant() {", "g29a"),
           "host_task sites")
    yield ("a host_task capturing a managed_handle",
           edit(files, M, "gate_cv.wait(lock, [&]() { return gate_open; });",
                "gate_cv.wait(lock, [&]() { return gate_open; });\n            (void) managed_handle;", "g29b"),
           "captures managed_handle")

    yield ("an unlisted queue",
           edit(files, M, "static std::unique_ptr<sycl::queue> g_split_coord_queue_owner;",
                "static std::unique_ptr<sycl::queue> g_split_coord_queue_owner;\nstatic sycl::queue * g_stray_queue = nullptr;", "g30a"),
           "g_stray_queue")
    yield ("a censused queue no longer waited",
           edit(files, M, "if (g_tp_device1_worker_queue) {\n            g_tp_device1_worker_queue->wait();\n        }",
                "", "g30b"),
           "does not wait g_tp_device1_worker_queue")
    yield ("a wait before the synchronize flush",
           edit(files, M, "static void ggml_backend_sycl_synchronize(ggml_backend_t backend) {",
                "static void ggml_backend_sycl_synchronize(ggml_backend_t backend) {\n    if (backend) { static_cast<ggml_backend_sycl_context *>(backend->context)->stream()->wait(); }", "g30c"),
           "before its first wait")
    yield ("the flush hook outside PRIVATE_TESTING",
           edit(files, M, "void ggml_sycl_test_set_cpu_tg_flush_disabled(bool exit_flush, bool synchronize_flush) {",
                "#endif\nvoid ggml_sycl_test_set_cpu_tg_flush_disabled(bool exit_flush, bool synchronize_flush) {\n#if defined(GGML_SYCL_PRIVATE_TESTING)", "g30d"),
           "outside GGML_SYCL_PRIVATE_TESTING")

    yield ("into_empty installing for a non-participant",
           edit(files, M, "cache->set_placement_plan_snapshot(participates ? selected : nullptr);",
                "cache->set_placement_plan_snapshot(selected);", "g31o28"), "does not install `participates")
    yield ("into_empty without its hint",
           edit(files, M, "if (cache->into_empty_skip_key() == skip_key) {\n        return ggml_sycl_into_empty_result::NOOP;\n    }",
                "", "g31o29"), "hint is not read")
    yield ("into_empty setting its hint for a participant",
           edit(files, M, "cache->set_placement_plan_snapshot(participates ? selected : nullptr);",
                "cache->set_placement_plan_snapshot(participates ? selected : nullptr);\n    cache->set_into_empty_skip_key(skip_key);",
                "g31o30"), "skip key is set somewhere other")
    yield ("into_empty overwriting a foreign load's plan",
           edit(files, M, "g_into_empty_foreign.fetch_add(1, std::memory_order_acq_rel);",
                "g_into_empty_foreign.fetch_add(1, std::memory_order_acq_rel);\n            cache->set_placement_plan_snapshot(selected);",
                "g31o31"), "does not leave a filled cache untouched")
    yield ("into_empty not witnessing a foreign plan",
           edit(files, M, "cache->set_into_empty_foreign_key(warn_key);", "", "g31o32"), "does not witness a foreign load")
    yield ("a hint key that ignores the publish generation",
           edit(files, M, "(publish_gen * 0xC2B2AE3D27D4EB4FULL)", "0", "g31o34"),
           "does not mix the owner's model, load and the publish generation")
    yield ("a hint key read with a constant generation",
           edit(files, M, "ggml_sycl_into_empty_skip_key(owner, ggml_sycl_tenant_publish_gen())",
                "ggml_sycl_into_empty_skip_key(owner, 0)", "g31o35"),
           "hint key is not built from the owner's load and the tenant publish generation")
    yield ("a hint key reverted to the plan id epoch, which a stable re-publish does not move",
           edit(files, M, "ggml_sycl_into_empty_skip_key(owner, ggml_sycl_tenant_publish_gen())",
                "ggml_sycl_into_empty_skip_key(owner, ggml_sycl::lifecycle_plan_publication_epoch())", "g31o36"),
           "hint key is not built from the owner's load and the tenant publish generation")
    yield ("a foreign WARN deduplicated on the per-publication key",
           edit(files, M, "const uint64_t warn_key = ggml_sycl_into_empty_foreign_warn_key(owner);",
                "const uint64_t warn_key = skip_key;", "g31o37"),
           "foreign-plan WARN is not deduplicated on its own owner-only key")
    yield ("a foreign WARN key that mixes the publish generation",
           edit(files, M, "return (owner.model.value * 0x9E3779B97F4A7C15ULL ^ owner.load.value) | 1ULL;",
                "return (owner.model.value * 0x9E3779B97F4A7C15ULL ^ owner.load.value ^ ggml_sycl_tenant_publish_gen()) | 1ULL;",
                "g31o38"),
           "foreign-plan WARN key is not the owner's model and load alone")
    yield ("a non-participant that reports an install",
           edit(files, M, "cache->set_into_empty_skip_key(skip_key);\n        return ggml_sycl_into_empty_result::NOOP;",
                "cache->set_into_empty_skip_key(skip_key);\n        return ggml_sycl_into_empty_result::INSTALLED;", "g31o33"),
           "non-participant does not get NOOP")

    yield ("an L0 acquire that blocks without a timeout",
           edit(files, UC_C, "while (!g_replan_txn_mutex.try_lock_for(interval)) {", "g_replan_txn_mutex.lock();\n        while (false) {", "g31w1"),
           "blocking acquire of L0 is not a timed wait")
    yield ("a blocked L0 acquire that never aborts under STRICT",
           edit(files, UC_C, "if (ggml_sycl_strict_enabled()) {\n                GGML_ABORT(\"[REPLAN-WAIT] a %s acquire",
                "if (false) {\n                GGML_ABORT(\"[REPLAN-WAIT] a %s acquire", "g31w2"),
           "does not log the holder at each interval and abort under STRICT")
    yield ("a watch that cannot abort under STRICT",
           edit(files, UC_C, "if (ggml_sycl_strict_enabled()) {\n                GGML_ABORT(\"[REPLAN-WAIT] %s exceeded",
                "if (false) {\n                GGML_ABORT(\"[REPLAN-WAIT] %s exceeded", "g31w3"),
           "wait watch does not log at each interval and abort under STRICT")
    yield ("synchronize_for_replan without its watch",
           edit(files, M, 'ggml_sycl_wait_watch watch("synchronize_for_replan");', "int watch_unused = 0;", "g31w4"),
           "not under a wait watch")

    yield ("a second re-plan mutex",
           edit(files, UC_C, "static std::timed_mutex g_replan_txn_mutex;",
                "static std::timed_mutex g_replan_txn_mutex;\nstatic std::mutex g_replan_extra_mutex;", "g31a"),
           "the re-plan mutexes are")
    yield ("a token member in a class",
           edit(files, M, "struct ggml_sycl_prepared_plan_publication {",
                "struct ggml_sycl_prepared_plan_publication {\n    ggml_sycl_replan_token held_token{ GGML_SYCL_REPLAN_KIND_LOAD };", "g31b"),
           "a class member of type ggml_sycl_replan_token")
    yield ("a public scope that accepts any kind",
           edit(files, M, "if (kind != GGML_SYCL_REPLAN_SCOPE_TRANSACTION) {\n        return nullptr;", "if (false) {\n        return nullptr;", "g31s1"),
           "accepts a kind other than TRANSACTION")
    yield ("a public scope that holds a LOAD token",
           edit(files, M, "return new ggml_backend_sycl_replan_scope_state(ggml_sycl::GGML_SYCL_REPLAN_KIND_TRANSACTION);",
                "return new ggml_backend_sycl_replan_scope_state(ggml_sycl::GGML_SYCL_REPLAN_KIND_LOAD);", "g31s2"),
           "does not construct a TRANSACTION token")
    yield ("a public scope that never releases",
           edit(files, M, "delete static_cast<ggml_backend_sycl_replan_scope_state *>(scope);", "(void) scope;", "g31s3"),
           "close does not release the owner")
    yield ("a public scope whose outermost witness message changed",
           edit(files, M, '"[REPLAN-TOKEN] growth scope not outermost: under LOAD"', '"[REPLAN-TOKEN] scope under LOAD"', "g31s4"),
           "witness message")
    yield ("a public scope open that is not served as a proc",
           edit(files, M, 'if (strcmp(name, "ggml_backend_sycl_replan_scope_open") == 0) {', 'if (strcmp(name, "ggml_backend_sycl_replan_scope_open_") == 0) {', "g31s5"),
           "is not served by the backend's proc table")
    yield ("a public scope state as a second token owner",
           edit(files, M, "struct ggml_backend_sycl_replan_scope_state {",
                "struct ggml_backend_sycl_replan_scope_state {\n    ggml_sycl_replan_token held_token{ GGML_SYCL_REPLAN_KIND_LOAD };", "g31s6"),
           "does not hold exactly one token")
    yield ("an entry without its token",
           edit(files, M, "void ggml_backend_sycl_commit_reactivate(void) {\n    ggml_sycl_replan_token l0(GGML_SYCL_REPLAN_KIND_LIFECYCLE);",
                "void ggml_backend_sycl_commit_reactivate(void) {", "g31c"),
           "does not construct the L0 token")
    yield ("an entry with the wrong kind",
           edit(files, M, "void ggml_backend_sycl_rollback_reactivate(void) {\n    ggml_sycl_replan_token l0(GGML_SYCL_REPLAN_KIND_LIFECYCLE);",
                "void ggml_backend_sycl_rollback_reactivate(void) {\n    ggml_sycl_replan_token l0(GGML_SYCL_REPLAN_KIND_LOAD);", "g31d"),
           "constructs a LOAD token")
    yield ("a witness rewritten as assert",
           edit(files, M, "GGML_SYCL_WITNESS(!ggml_sycl::ggml_sycl_replan_token_held(), \"[REPLAN-TOKEN] token held in graph compute\");",
                "assert(!ggml_sycl::ggml_sycl_replan_token_held() && \"[REPLAN-TOKEN] token held in graph compute\");", "g31e"),
           "is not a GGML_SYCL_WITNESS")
    yield ("a witness message changed",
           edit(files, M, "\"[REPLAN-TOKEN] token held in graph compute\"", "\"[REPLAN-TOKEN] held in graph compute\"", "g31f"),
           "witness message")
    yield ("the preload witness weakened to any kind",
           edit(files, M, "GGML_SYCL_WITNESS(ggml_sycl::ggml_sycl_replan_token_held(GGML_SYCL_REPLAN_KIND_LOAD),",
                "GGML_SYCL_WITNESS(ggml_sycl::ggml_sycl_replan_token_held(),", "g31g"),
           "preload's witness is not held(LOAD)")
    yield ("the load_end overlap witness removed",
           edit(files, M, "GGML_SYCL_WITNESS(running == 0, \"[REPLAN-TOKEN] two load_end bodies overlapped\");", "", "g31j"),
           "overlap witness is missing")
    yield ("the load_end overlap witness weakened",
           edit(files, M, "GGML_SYCL_WITNESS(running == 0,", "GGML_SYCL_WITNESS(running >= 0,", "g31k"),
           "overlap witness is missing")
    yield ("load_end not constructing its overlap witness",
           edit(files, M, "    ggml_sycl_load_end_body_witness load_end_body_witness;\n", "", "g31l"),
           "does not construct its overlap witness")
    # ---- gate 31, owner threading
    yield ("the plan scope accessor reading something else",
           edit(files, M, "static ggml_sycl_plan_scope * ggml_sycl_plan_scope_current() {\n    return g_plan_scope;", "static ggml_sycl_plan_scope * ggml_sycl_plan_scope_current() {\n    return nullptr;", "o27"),
           "ggml_sycl_plan_scope_current() is not a one-line read")
    yield ("a thread-local owner read outside any scope's chains",
           edit(files, M, "static const ggml_sycl::lifecycle::ModelToken * ggml_sycl_dispatch_owner() {\n    return g_dispatch_owner_bound ? &g_dispatch_owner : nullptr;\n}",
                "static const ggml_sycl::lifecycle::ModelToken * ggml_sycl_dispatch_owner() {\n    return g_dispatch_owner_bound ? &g_dispatch_owner : nullptr;\n}\n"
                "static bool ggml_sycl_stray_owner_read() { return ggml_sycl_dispatch_owner() != nullptr; }", "o1"),
           "is read in ggml_sycl_stray_owner_read")
    yield ("a scope built from the registry instead of the context",
           edit(files, M, "        const bool                       bound = ggml_sycl_execution_current_owner(ctx, owner);",
                "        owner = ggml_sycl::lifecycle::global_registry().current_active_token();\n"
                "        const bool                       bound = true;", "o2"),
           "does not come from ggml_sycl_execution_current_owner")
    yield ("a second scope construction",
           edit(files, M, "void ggml_backend_sycl_set_runtime_context(ggml_backend_t backend,",
                "static void ggml_sycl_stray_scope(const ggml_backend_sycl_context * c) { ggml_sycl_dispatch_owner_scope s(c); }\n"
                "void ggml_backend_sycl_set_runtime_context(ggml_backend_t backend,", "o3"),
           "dispatch owner scopes are constructed")
    yield ("a scope constructed from a different argument",
           edit(files, M, "ggml_sycl_dispatch_owner_scope dispatch_owner(\n        backend ? static_cast<const ggml_backend_sycl_context *>(backend->context) : nullptr);",
                "ggml_sycl_dispatch_owner_scope dispatch_owner(nullptr);", "o4"),
           "not constructed from the backend's own context")
    yield ("a nested scope that does not abort on a different owner",
           edit(files, M, "GGML_ABORT(\"[DISPATCH-OWNER] nested scope binds a different owner\");", "(void) 0;", "o5"),
           "does not abort with its literal")
    yield ("a nested scope that rebinds without comparing",
           edit(files, M, "(bound != g_dispatch_owner_bound || (bound && !(owner == g_dispatch_owner)))", "false", "o6"),
           "is not compared with the bound one")
    yield ("a thread-local owner read inside a host_task",
           edit(files, CPUD, "cgh.host_task([=]() { run_mul_mat(); });", "cgh.host_task([=]() { (void) ggml_sycl_dispatch_owner(); run_mul_mat(); });", "o7"),
           "inside work that runs on another thread")
    yield ("a pool lambda reaching the resolver",
           edit(files, CPUD, "cgh.host_task([=]() { run_fused(); }); });\n", "cgh.host_task([=]() { (void) ggml_sycl_resolve_moe_expert_route(nullptr, 0, 0, GGML_LAYOUT_AOS); run_fused(); }); });\n", "o8"),
           "inside work that runs on another thread")
    yield ("the owner written outside its scope class",
           edit(files, M, "static const ggml_sycl::lifecycle::ModelToken * ggml_sycl_dispatch_owner() {\n    return g_dispatch_owner_bound ? &g_dispatch_owner : nullptr;\n}",
                "static const ggml_sycl::lifecycle::ModelToken * ggml_sycl_dispatch_owner() {\n    g_dispatch_owner_bound = true;\n    return g_dispatch_owner_bound ? &g_dispatch_owner : nullptr;\n}", "o9"),
           "writes the dispatch owner outside its scope class")
    yield ("into_empty reading the process-global publication",
           edit(files, M, "    const auto current = cache->get_placement_plan_snapshot();\n    if (current && current->plan && !current->plan->entries.empty()) {",
                "    const auto current = std::atomic_load_explicit(&g_placement_publication, std::memory_order_acquire);\n    if (current && current->plan && !current->plan->entries.empty()) {", "o10"),
           "into_empty reaches g_placement_publication")
    yield ("into_empty writing a device-global",
           edit(files, M, "    cache->set_placement_plan_snapshot(participates ? selected : nullptr);",
                "    g_model_n_layer = selected->model_n_layer;\n    cache->set_placement_plan_snapshot(participates ? selected : nullptr);", "o11"),
           "into_empty reaches g_model_n_layer")
    yield ("into_empty without the identity compare",
           edit(files, M, " || installed->model_id != selected->model_id", "", "o12"),
           "does not compare version, model_id and load_txn_id")
    yield ("into_empty turning the null selection into a silent return",
           edit(files, M, "        return ggml_sycl_into_empty_result::REFUSED;\n    }\n    const auto current = cache->get_placement_plan_snapshot();",
                "        return ggml_sycl_into_empty_result::NOOP;\n    }\n    const auto current = cache->get_placement_plan_snapshot();", "o13"),
           "null selection branch does not refuse")
    yield ("into_empty taking a second lock",
           edit(files, M, "    const auto selected = ggml_sycl::lifecycle_select_placement_plan(owner.model.value,",
                "    std::lock_guard<std::mutex> second(g_tensor_inventory_mutex);\n    const auto selected = ggml_sycl::lifecycle_select_placement_plan(owner.model.value,", "o14"),
           "not exactly one g_tensor_inventory_mutex critical section")
    yield ("a plain republish restored in ensure",
           edit(files, M, "        if (owner) {\n            if (auto * cache = ggml_sycl::get_unified_cache_for_device(target_device);",
                "        if (ggml_sycl_has_global_plan()) {\n            ggml_sycl_republish_current_plan();\n        }\n        if (owner) {\n            if (auto * cache = ggml_sycl::get_unified_cache_for_device(target_device);", "o15"),
           "ensure still reads the global plan")
    yield ("a second plain republish caller",
           edit(files, M, "static const ggml_sycl::lifecycle::ModelToken * ggml_sycl_dispatch_owner() {",
                "static void ggml_sycl_stray_republish() { ggml_sycl_republish_current_plan(); }\nstatic const ggml_sycl::lifecycle::ModelToken * ggml_sycl_dispatch_owner() {", "o16"),
           "callers (wanted exactly the preload's)")
    yield ("a default owner passed at a site",
           edit(files, M, "ggml_sycl_ensure_moe_secondary_queues_for_plan(operand.owning_device(), mmid_owner_ptr);",
                "ggml_sycl_ensure_moe_secondary_queues_for_plan(operand.owning_device(), nullptr);", "o17"),
           "does not pass mmid_owner_ptr")
    yield ("the status discarded at the resolver",
           edit(files, M, "                switch (ggml_sycl_ensure_moe_secondary_queues_for_plan(route.planned_device,\n                                                                       ggml_sycl_dispatch_owner())) {",
                "                (void) ggml_sycl_ensure_moe_secondary_queues_for_plan(route.planned_device, ggml_sycl_dispatch_owner());\n"
                "                switch (GGML_SYCL_SECONDARY_QUEUE_READY) {", "o18"),
           "discards its status")
    yield ("REFUSED answered with continue in the collector",
           edit(files, M, "        if (!ggml_sycl::ggml_sycl_secondary_queue_ready_or_throw(\n                ggml_sycl::ggml_sycl_ensure_moe_secondary_queues_for_plan(d, ggml_sycl_dispatch_owner()))) {\n            continue;",
                "        if (ggml_sycl::ggml_sycl_ensure_moe_secondary_queues_for_plan(d, ggml_sycl_dispatch_owner()) != GGML_SYCL_SECONDARY_QUEUE_READY) {\n            continue;", "o19"),
           "is not consumed by a switch")
    yield ("REFUSED at the retained operand mapped to queue_available = false",
           edit(files, M, "                        ggml_sycl::moe_batch_executor_choice refused;\n                        refused.reject = ggml_sycl::moe_batch_reject_reason::PLAN_OWNER_REFUSED;",
                "                        queue_available = false;\n                        ggml_sycl::moe_batch_executor_choice refused;\n                        refused.reject = ggml_sycl::moe_batch_reject_reason::PLAN_OWNER_REFUSED;", "o20"),
           "assigns queue_available = false")
    yield ("the refusal helper not throwing",
           edit(files, M, "    throw ggml_sycl_fallback_error(\"MUL_MAT_ID secondary queue refused: plan owner\");", "    GGML_ABORT(\"unreachable\");", "o21"),
           "refusal helper is not a [[noreturn]] throw")
    yield ("the resolver wrapper not failing on REFUSED",
           edit(files, M, "    if (route.kind == moe_expert_route_kind::REFUSED) {\n        ggml_sycl_secondary_queue_refused_fail();\n    }", "", "o22"),
           "resolver wrapper does not turn a REFUSED route")
    yield ("ready_or_throw answering REFUSED with false",
           edit(files, M, "        case GGML_SYCL_SECONDARY_QUEUE_REFUSED:\n            ggml_sycl_secondary_queue_refused_fail();\n    }\n    return false;",
                "        case GGML_SYCL_SECONDARY_QUEUE_REFUSED:\n            return false;\n    }\n    return false;", "o23"),
           "ready_or_throw does not fail on REFUSED")
    yield ("a consumer calling the resolver core",
           edit(files, M, "static const ggml_sycl::lifecycle::ModelToken * ggml_sycl_dispatch_owner() {",
                "static moe_expert_route ggml_sycl_stray_core_user(const ggml_tensor * t) { return ggml_sycl::ggml_sycl_resolve_moe_expert_route_core(t, 0, 0, GGML_LAYOUT_AOS, false); }\n"
                "static const ggml_sycl::lifecycle::ModelToken * ggml_sycl_dispatch_owner() {", "o26"),
           "resolver core is called from ggml_sycl_stray_core_user")
    yield ("a catch of the fallback error that swallows it",
           edit(files, M, "static const ggml_sycl::lifecycle::ModelToken * ggml_sycl_dispatch_owner() {",
                "static void ggml_sycl_stray_swallow() { try { } catch (const ggml_sycl_fallback_error &) { } }\nstatic const ggml_sycl::lifecycle::ModelToken * ggml_sycl_dispatch_owner() {", "o24"),
           "does not rethrow")
    yield ("the host test not bounding the install counter",
           edit(files, M, "    const uint64_t installs_before = ggml_sycl_into_empty_install_count();\n", "    const uint64_t installs_before = 0;\n", "o25"),
           "host test does not bound the install counter")
    yield ("the witness macro under NDEBUG",
           edit(files, UC_H, "#define GGML_SYCL_WITNESS(cond, message)                      \\\n    do {                                                      \\\n        if (::ggml_sycl::g_sycl_witness_enabled && !(cond)) { \\",
                "#define GGML_SYCL_WITNESS(cond, message)                      \\\n    do {                                                      \\\n        if (!NDEBUG && ::ggml_sycl::g_sycl_witness_enabled && !(cond)) { \\", "g31h"),
           "depends on NDEBUG")
    yield ("the release proc's outermost check removed",
           edit(files, UC_C, "GGML_SYCL_WITNESS(g_replan_token_depth == 0, \"[REPLAN-TOKEN] release proc entered with L0 held\");",
                "", "g31i"),
           "release proc entered with L0 held")

    yield ("an accessor that does not consult the override first",
           edit(files, M, "static std::shared_ptr<const ggml_sycl::lifecycle_plan_snapshot> ggml_sycl_global_plan_snapshot() {\n    if (g_measure_plan_override) {\n        return g_measure_plan_override;\n    }\n    return std::atomic_load_explicit(&g_placement_publication, std::memory_order_acquire);",
                "static std::shared_ptr<const ggml_sycl::lifecycle_plan_snapshot> ggml_sycl_global_plan_snapshot() {\n    return std::atomic_load_explicit(&g_placement_publication, std::memory_order_acquire);", "g36a"),
           "does not consult the measure plan override first")
    yield ("a stray read of the publication",
           edit(files, M, "static size_t ggml_sycl_host_chunk_cap_constant() {",
                "static bool ggml_sycl_stray_pub() { return std::atomic_load_explicit(&g_placement_publication, std::memory_order_acquire) != nullptr; }\nstatic size_t ggml_sycl_host_chunk_cap_constant() {", "g36b"),
           "is not one of the four accessors")
    yield ("a latch read going direct",
           edit(files, M, "static bool ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {",
                "static bool ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {\n    (void) g_moe_multi_gpu_active.load();", "g36c"),
           "reads g_moe_multi_gpu_active directly")
    yield ("a second writer of the override",
           edit(files, M, "static size_t ggml_sycl_host_chunk_cap_constant() {",
                "static void ggml_sycl_stray_ov() { g_measure_plan_override.reset(); }\nstatic size_t ggml_sycl_host_chunk_cap_constant() {", "g36d"),
           "g_measure_plan_override is written in")
    yield ("install without its nesting witness",
           edit(files, M, "GGML_LOG_WARN(\"[CONTEXT-PLAN-BUG] measure plan override nested\\n\");",
                "GGML_LOG_WARN(\"measure plan override nested\\n\");", "g36e"),
           "install lacks its nesting witness")
    yield ("install naming the provisional builder",
           edit(files, M, "std::shared_ptr<const ggml_sycl::lifecycle_plan_snapshot> snapshot;\n    switch (stage) {",
                "std::shared_ptr<const ggml_sycl::lifecycle_plan_snapshot> snapshot;\n    (void) &ggml_sycl_make_provisional_plan_snapshot;\n    switch (stage) {", "g36f"),
           "names ggml_sycl_make_provisional_plan_snapshot")
    yield ("a measure free reaching the SYCL free",
           edit(files, M, "static void ggml_backend_sycl_measure_free(ggml_backend_t backend) {\n    delete backend;",
                "static void ggml_backend_sycl_measure_free(ggml_backend_t backend) {\n    ggml_backend_sycl_free(backend);", "g36g"),
           "reaches ggml_backend_sycl_free")
    yield ("a measure interface slot filled",
           edit(files, M, "/* .synchronize             = */ NULL,\n    /* .graph_plan_create       = */ NULL,\n    /* .graph_plan_free         = */ NULL,\n    /* .graph_plan_update       = */ NULL,\n    /* .graph_plan_compute      = */ NULL,\n    /* .graph_compute           = */ NULL,\n    /* .event_record            = */ NULL,",
                "/* .synchronize             = */ ggml_backend_sycl_synchronize,\n    /* .graph_plan_create       = */ NULL,\n    /* .graph_plan_free         = */ NULL,\n    /* .graph_plan_update       = */ NULL,\n    /* .graph_plan_compute      = */ NULL,\n    /* .graph_compute           = */ NULL,\n    /* .event_record            = */ NULL,", "g36h"),
           "slot .synchronize is not NULL")
    yield ("the measure init calling the real init",
           edit(files, M, "ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), device);\n    if (!dev) {\n        return nullptr;\n    }\n    return new ggml_backend{",
                "ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), device);\n    if (!dev) {\n        return nullptr;\n    }\n    (void) ggml_backend_sycl_init(device);\n    return new ggml_backend{", "g36i"),
           "names ggml_backend_sycl_init(")
    yield ("the measure backend carrying the SYCL guid",
           edit(files, M, "/* .guid    = */ ggml_backend_sycl_measure_guid(),", "/* .guid    = */ ggml_backend_sycl_guid(),", "g36j"),
           "does not carry its own guid")
    yield ("(a)'s cap with a literal floor",
           edit(files, M, "scratch = ggml_sycl::ggml_sycl_compute_arena_bytes(device);\n                    break;",
                "scratch = 2ULL * 1024 * 1024 * 1024;\n                    break;", "g36k"),
           "PROBE_MIN set is not")

    yield ("a second default for the compute arena in the reservation",
           edit(files, M, "const size_t arena_bytes = ggml_sycl::ggml_sycl_compute_arena_bytes(0);",
                'size_t arena_bytes = 512ULL << 20;\n        if (const char * e = std::getenv("GGML_SYCL_COMPUTE_ARENA_MB")) { arena_bytes = static_cast<size_t>(std::atoi(e)) << 20; }',
                "g36l"), "is read 2 time(s)")
    yield ("the reservation without the sizing function",
           edit(files, M, "const size_t arena_bytes = ggml_sycl::ggml_sycl_compute_arena_bytes(0);",
                "const size_t arena_bytes = 512ULL << 20;", "g36m"), "does not size itself with")
    yield ("the zone sizing with a literal scratch size",
           edit(files, UC_C, "size_t scratch_zone = ggml_sycl_compute_arena_bytes(dev_id);",
                "size_t scratch_zone = 512ULL << 20;", "g36o"),
           "ensure_planned_arena_zones does not size the SCRATCH zone")
    yield ("a second default in the sizing function's neighbour",
           edit(files, UC_C, "bool ggml_sycl_device_has_zones(int device) {",
                'size_t ggml_sycl_stray_arena_bytes() { const char * e = std::getenv("GGML_SYCL_COMPUTE_ARENA_MB"); return e ? 1 : 512; }\n'
                "bool ggml_sycl_device_has_zones(int device) {", "g36n"), "is read 2 time(s)")

    yield ("a pool gate without the token kind",
           edit(files, POOL, "!ggml_sycl::ggml_sycl_replan_token_held(ggml_sycl::GGML_SYCL_REPLAN_KIND_TRANSACTION)",
                "!ggml_sycl::ggml_sycl_replan_token_held()", "g37a", count=1),
           "tests the token with the default kind")
    yield ("a pool gate that drops its site",
           edit(files, POOL, "site=", "where=", "g37b", count=1), "does not name its site")
    yield ("a wrapper whose impl return sits under an if",
           edit(files, M, "    return ggml_sycl_set_runtime_context_for_model_impl(backend, model, n_ctx, n_ubatch, n_seq_max, kv_unified,\n"
                "                                                        swa_full, flash_attn_enabled, nullptr, false);",
                "    if (false) return ggml_sycl_set_runtime_context_for_model_impl(backend, model, n_ctx, n_ubatch, n_seq_max, kv_unified,\n"
                "                                                        swa_full, flash_attn_enabled, nullptr, false);", "g31w"),
           "does not return ggml_sycl_set_runtime_context_for_model_impl")


_SWEEP = []


def _sweep_one(i):
    """One mutant: None when its wanted finding was reported, otherwise the failure line."""
    name, m, frag = _SWEEP[i]
    res = violations(m)
    if any(frag in why for why in res):
        return None
    return "FAIL: mutant went undetected: %s (wanted %r, got %d other finding(s))" % (name, frag, len(res))


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else REPO
    files = load(root)
    found = violations(files)
    status = 0
    for why in found:
        print("FAIL: %s" % why)
        status = 1
    if status:
        return status
    built = list(mutants(files))
    if len(built) < 40:
        print("FAIL: only %d mutants could be built" % len(built))
        return 1
    # Each mutant is a whole gate evaluation (about 3 s), and they are independent, so they run in forked workers
    # that share the parent's source texts.  A host that cannot fork runs them one by one.
    global _SWEEP
    _SWEEP = built
    verdicts = None
    try:
        import multiprocessing
        workers = max(1, min(4, os.cpu_count() or 1))
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            verdicts = pool.map(_sweep_one, range(len(built)), chunksize=1)
    except (ImportError, OSError, ValueError):
        verdicts = None
    if verdicts is None:
        verdicts = [_sweep_one(i) for i in range(len(built))]
    for failure in verdicts:
        if failure:
            print(failure)
            status = 1
    if status == 0:
        print("PASS: gates 3, 23, 29, 30, 31, 36 and 37 hold on the real source; %d mutants caught" % len(built))
    return status


if __name__ == "__main__":
    sys.exit(main())
