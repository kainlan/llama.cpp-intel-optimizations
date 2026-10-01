#!/usr/bin/env python3
"""Gates 3, 23, 29, 30, 31, 36 and 37 (zhcn-design §5.2): the backend half of the re-plan protocol.

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
      flushes the C2t lists before its first wait; the flush-disabling hook compiles only
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
CLEAR_ACTIVE_CALLERS = 12  # the design lists eleven; optional-layouts-retire (dkw0) is the twelfth on this base


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
    # C2t: the flush precedes the first wait in ggml_backend_sycl_synchronize.
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
    "ggml_backend_sycl_set_runtime_context_for_model": "TRANSACTION",
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
    # No token is a class member.
    for path in all_sycl_sources_in(files):
        cc = code(files, path)
        for head, o, e in top_spans(cc):
            hdr = strip_pp(cc[head:o])
            if re.search(r"\b(struct|class)\s+\w+[^;(]*$", hdr) and "ggml_sycl_replan_token" not in hdr.split("{")[0].split()[-1:]:
                if re.search(r"(?m)^\s*(?:mutable\s+)?(?:ggml_sycl::)?ggml_sycl_replan_token\s+\w+\s*(?:;|\{|=)", cc[o:e]):
                    bad("gate 31: a class member of type ggml_sycl_replan_token in %s" % path)
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
    # The preload's check is held(LOAD), not weakened to any kind.
    kk = keep(files, MAIN)
    m = re.search(r"GGML_SYCL_WITNESS\(\s*(ggml_sycl_replan_token_held\([^)]*\))\s*,\s*\"\[REPLAN-TOKEN\] preload without a LOAD token\"",
                  kk)
    if not m or "GGML_SYCL_REPLAN_KIND_LOAD" not in m.group(1):
        bad("gate 31: the preload's witness is not held(LOAD) with its literal message")
    # The graph-compute check is held(), any kind.
    gm = re.search(r"GGML_SYCL_WITNESS\(\s*!ggml_sycl_replan_token_held\(\s*\)\s*,\s*\"\[REPLAN-TOKEN\] token held in graph compute\"", kk)
    if not gm:
        bad("gate 31: the graph-compute witness is not `!held()` (any kind)")


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


GATES = (gate3, gate23, gate29, gate30, gate31, gate36, gate37)


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
    yield ("a twelfth clear_active caller",
           edit(files, M, "static void ggml_sycl_release_graph_leases_for_owner(ggml_sycl::lifecycle::ModelToken owner) noexcept {",
                "static void ggml_sycl_stray_clear(ggml_backend_sycl_context * c) { sycl_exec_graph_clear_active(c, \"stray\"); }\n"
                "static void ggml_sycl_release_graph_leases_for_owner(ggml_sycl::lifecycle::ModelToken owner) noexcept {", "g23e"),
           "has 13 callers")

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

    yield ("a second re-plan mutex",
           edit(files, UC_C, "static std::mutex g_replan_txn_mutex;",
                "static std::mutex g_replan_txn_mutex;\nstatic std::mutex g_replan_extra_mutex;", "g31a"),
           "the re-plan mutexes are")
    yield ("a token member in a class",
           edit(files, M, "struct ggml_sycl_prepared_plan_publication {",
                "struct ggml_sycl_prepared_plan_publication {\n    ggml_sycl_replan_token held_token{ GGML_SYCL_REPLAN_KIND_LOAD };", "g31b"),
           "a class member of type ggml_sycl_replan_token")
    yield ("an entry without its token",
           edit(files, M, "void ggml_backend_sycl_commit_reactivate(void) {\n    ggml_sycl_replan_token l0(GGML_SYCL_REPLAN_KIND_LIFECYCLE);",
                "void ggml_backend_sycl_commit_reactivate(void) {", "g31c"),
           "does not construct the L0 token")
    yield ("an entry with the wrong kind",
           edit(files, M, "void ggml_backend_sycl_rollback_reactivate(void) {\n    ggml_sycl_replan_token l0(GGML_SYCL_REPLAN_KIND_LIFECYCLE);",
                "void ggml_backend_sycl_rollback_reactivate(void) {\n    ggml_sycl_replan_token l0(GGML_SYCL_REPLAN_KIND_LOAD);", "g31d"),
           "constructs a LOAD token")
    yield ("a witness rewritten as assert",
           edit(files, M, "GGML_SYCL_WITNESS(!ggml_sycl_replan_token_held(), \"[REPLAN-TOKEN] token held in graph compute\");",
                "assert(!ggml_sycl_replan_token_held() && \"[REPLAN-TOKEN] token held in graph compute\");", "g31e"),
           "is not a GGML_SYCL_WITNESS")
    yield ("a witness message changed",
           edit(files, M, "\"[REPLAN-TOKEN] token held in graph compute\"", "\"[REPLAN-TOKEN] held in graph compute\"", "g31f"),
           "witness message")
    yield ("the preload witness weakened to any kind",
           edit(files, M, "GGML_SYCL_WITNESS(ggml_sycl_replan_token_held(GGML_SYCL_REPLAN_KIND_LOAD),",
                "GGML_SYCL_WITNESS(ggml_sycl_replan_token_held(),", "g31g"),
           "preload's witness is not held(LOAD)")
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

    yield ("a pool gate without the token kind",
           edit(files, POOL, "!ggml_sycl::ggml_sycl_replan_token_held(ggml_sycl::GGML_SYCL_REPLAN_KIND_TRANSACTION)",
                "!ggml_sycl::ggml_sycl_replan_token_held()", "g37a", count=1),
           "tests the token with the default kind")
    yield ("a pool gate that drops its site",
           edit(files, POOL, "site=", "where=", "g37b", count=1), "does not name its site")


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
    for name, m, frag in built:
        res = violations(m)
        if not any(frag in why for why in res):
            print("FAIL: mutant went undetected: %s (wanted %r, got %d other finding(s))" % (name, frag, len(res)))
            status = 1
    if status == 0:
        print("PASS: gates 3, 23, 29, 30, 31, 36 and 37 hold on the real source; %d mutants caught" % len(built))
    return status


if __name__ == "__main__":
    sys.exit(main())
