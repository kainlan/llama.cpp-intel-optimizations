#!/usr/bin/env python3
"""Gate 27 (zhcn-design §5.2): the holder census of the context-tenant plan, producer half.

Lead ruling §B.2: there is no fourth holder class.  A cache that only COMPARES a
source holds a `mem_handle_identity` (a value built from the allocator's monotonic
retention id), never a `mem_handle`.  This gate pins the producer fixes that keep a compare-only cache from holding a handle:

  * the activation-keyed MoE caches (`moe_ids_cache_key`, `moe_down_shadow_key`, the mmvq activation
    cache and `moe_quant_cache`) hold no `mem_handle`;
  * `g_data_ptr_cache`'s value type carries no `mem_handle` and carries `tenant_gen`;
    every access to it goes through store / lookup / new_graph; the lookup compares
    BOTH `tenant_gen` against `g_tenant_publish_gen` AND `src` against
    `ggml_sycl_tensor_slice_identity(` computed at the lookup, the same function the
    fill calls; that function takes its offset from the resolved address, never from
    `view_offset`;
  * `g_tenant_publish_gen` is written only by its bump function, and the bump is
    called only from the commit and teardown sites;
  * W4 (`ggml_sycl_publish_moe_artifact_handle`) and any `release_buffer_refs` do not exist;
  * `mem_handle_identity` is built only from `canonical_allocation_id_` and
    `canonical_generation_`;
  * the runtime_minted belt is called from the W2/W3 and W6/W7 persistent writers;
  * every return path of `ggml_backend_sycl_graph_compute` runs the exit hooks (the scatter
    flush, the activation-map drop, the data-pointer-cache drop, the staged-owner publish and the
    staging-tenant release);
  * the oneDNN scratch ring's retained-owner clear carries its empty assert.

The walk is textual, comment- and string-stripped, so a call reached through a
wrapper is not seen; the clauses name the functions that must contain each call.
Every clause proves itself on mutants of the real source and the gate refuses to
pass vacuously.

argv: [checkout-root]
"""
import importlib.util
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_spec = importlib.util.spec_from_file_location(
    "tenant_plan_source", os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-sycl-context-tenant-plan-source.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
scrubbed = _base.scrubbed
close_of = _base.close_of
line_of = _base.line_of

SYCL = "ggml/src/ggml-sycl/"
SCOPE = (SYCL + "ggml-sycl.cpp", SYCL + "common.hpp", SYCL + "mem-handle.cpp", SYCL + "mem-handle.hpp",
         SYCL + "unified-cache.cpp", SYCL + "unified-cache.hpp", SYCL + "graph-recorder-scope.hpp")
MAIN = SYCL + "ggml-sycl.cpp"
COMMON = SYCL + "common.hpp"
MEMH = SYCL + "mem-handle.cpp"
RECS = SYCL + "graph-recorder-scope.hpp"

RAW_HANDLE = r"\bmem_handle\b(?!_identity)"


# A mutant changes one file, so the other files' scrubbed text is the same string object on every sweep step.
# Memoising on the text (a str caches its own hash) turns the per-mutant cost from "scrub every source" into
# "scrub the one that changed".
_SCRUB_CACHE = {}


def _scrub_once(text, keep_strings):
    key = (text, keep_strings)
    hit = _SCRUB_CACHE.get(key)
    if hit is None:
        if len(_SCRUB_CACHE) > 96:
            _SCRUB_CACHE.clear()
        hit = scrubbed(text, keep_strings)
        _SCRUB_CACHE[key] = hit
    return hit


def code(files, name):
    return _scrub_once(files[name], False)


def keep(files, name):
    return _scrub_once(files[name], True)


_BRACE = re.compile(r"[{}]")
_SPAN_CACHE = {}


def top_spans(c):
    """Spans (header_start, open, close) of every top-level `{...}`: a function, a struct, an enum.  A `namespace`
    brace is transparent.  Braces are matched on the blanked text."""
    key = (hash(c), len(c))
    hit = _SPAN_CACHE.get(key)
    if hit is not None and hit[0] == c:
        return hit[1]
    out = []
    depth = 0
    seg = 0
    open_at = -1
    ns_stack = []
    for m in _BRACE.finditer(c):
        p = m.start()
        if m.group() == "{":
            if depth == 0:
                header = c[seg:p]
                if re.search(r"\bnamespace\b[^;]*$", header) or re.search(r'\bextern\s*""\s*$', header):
                    ns_stack.append(True)
                    seg = p + 1
                    continue
                open_at = p
                head_at = seg
            depth += 1
        else:
            if depth == 0:
                if ns_stack:
                    ns_stack.pop()
                    seg = p + 1
                continue
            depth -= 1
            if depth == 0:
                out.append((head_at, open_at, p))
                seg = p + 1
        if depth == 0 and m.group() == "}":
            pass
    # a `;` between spans moves seg on; recompute headers from the last `;`/`}`/`#` line before each open brace
    fixed = []
    for head_at, o, e in out:
        semi = max(c.rfind(";", 0, o), c.rfind("}", 0, o))
        fixed.append((max(head_at, semi + 1), o, e))
    _SPAN_CACHE.clear()
    _SPAN_CACHE[key] = (c, fixed)
    return fixed


def func_bodies(c, name):
    """Bodies (open, close) of every top-level definition whose declarator names `name` (qualified or not)."""
    out = []
    for head, o, e in top_spans(c):
        header = c[head:o]
        # strip preprocessor lines
        header = re.sub(r"(?m)^\s*#.*$", "", header)
        m = re.search(r"(?<![\w:])(?:\w+::)*" + re.escape(name) + r"\s*\(", header)
        if m and not re.search(r"\b(struct|class|enum|union)\s+" + re.escape(name) + r"\b", header):
            out.append((o, e))
    return out


def struct_body(c, name):
    for head, o, e in top_spans(c):
        if re.search(r"\b(struct|class)\s+" + re.escape(name) + r"\b[^;(]*$", re.sub(r"(?m)^\s*#.*$", "", c[head:o])):
            return (o, e)
    return None


def nested_struct_body(c, name):
    """A struct declared inside another type (the common.hpp context members)."""
    m = re.search(r"\bstruct\s+" + re.escape(name) + r"\b[^;{(]*\{", c)
    if not m:
        return None
    return (m.end() - 1, close_of(c, m.end() - 1))


def text_of(c, span):
    return c[span[0]:span[1] + 1]


def enclosing_top(c, pos):
    for head, o, e in top_spans(c):
        if o <= pos <= e:
            header = re.sub(r"(?m)^\s*#.*$", "", c[head:o])
            m = re.search(r"((?:\w+::)*\w+)\s*\(", header)
            return m.group(1) if m else header.strip()[:40]
    return None


def violations(files):
    out = []

    def bad(path, c, pos, why):
        out.append("%s:%d: %s" % (path, line_of(c, pos), why))

    for p in SCOPE:
        if p not in files:
            out.append("%s: not scanned; the gate would pass vacuously" % p)
            return out
    main = code(files, MAIN)
    mainkeep = keep(files, MAIN)
    common = code(files, COMMON)
    memh = code(files, MEMH)

    # (A) the activation-keyed MoE caches hold no mem_handle as their source
    for path, c, nested, names in (
            (COMMON, common, True, ("moe_ids_cache_key",)),
            (MAIN, main, False, ("moe_down_shadow_key",))):
        for name in names:
            span = nested_struct_body(c, name) if nested else struct_body(c, name)
            if span is None:
                out.append("%s: struct %s not found" % (path, name))
                continue
            body = text_of(c, span)
            m = re.search(RAW_HANDLE, body)
            if m:
                bad(path, c, span[0] + m.start(), "%s holds a mem_handle (a compare-only cache holds a mem_handle_identity)" % name)
            if "mem_handle_identity" not in body:
                bad(path, c, span[0], "%s holds no mem_handle_identity" % name)
    # the activation caches keep their own backing handle (they own that buffer); what they must not keep is the
    # SOURCE as a handle
    for name in ("mmvq_q8_activation_cache_t", "moe_quant_cache"):
        span = nested_struct_body(common, name)
        if span is None:
            out.append("%s: struct %s not found" % (COMMON, name))
            continue
        body = text_of(common, span)
        m = re.search(r"\bmem_handle\b(?!_identity)\s+cached_src\w*", body)
        if m:
            bad(COMMON, common, span[0] + m.start(), "%s holds its source as a mem_handle (cached_src must be a mem_handle_identity)" % name)
        if not re.search(r"\bmem_handle_identity\s+cached_src\b", body):
            bad(COMMON, common, span[0], "%s has no mem_handle_identity cached_src" % name)

    # (B) g_data_ptr_cache
    entry = struct_body(main, "ggml_sycl_data_ptr_cache_entry")
    if entry is None:
        out.append("%s: struct ggml_sycl_data_ptr_cache_entry not found" % MAIN)
    else:
        body = text_of(main, entry)
        m = re.search(RAW_HANDLE, body)
        if m:
            bad(MAIN, main, entry[0] + m.start(), "g_data_ptr_cache's value type contains a mem_handle")
        if not re.search(r"\btenant_gen\b", body):
            bad(MAIN, main, entry[0], "g_data_ptr_cache's value type lacks tenant_gen")
        if "mem_handle_identity" not in body:
            bad(MAIN, main, entry[0], "g_data_ptr_cache's value type lacks the source identity")
    decl = re.search(r"unordered_map<([^;]*?)>\s*g_data_ptr_cache\s*;", main, re.S)
    if not decl:
        out.append("%s: the g_data_ptr_cache declaration was not found" % MAIN)
    else:
        if "ggml_sycl_data_ptr_cache_entry" not in decl.group(1) or re.search(RAW_HANDLE, decl.group(1)):
            bad(MAIN, main, decl.start(), "g_data_ptr_cache is not keyed to ggml_sycl_data_ptr_cache_entry")
    allowed = {"ggml_sycl_data_ptr_cache_store", "ggml_sycl_data_ptr_cache_lookup", "ggml_sycl_data_ptr_cache_new_graph"}
    uses = 0
    for m in re.finditer(r"\bg_data_ptr_cache\b", main):
        if decl and decl.start() <= m.start() <= decl.end():
            continue
        uses += 1
        fn = enclosing_top(main, m.start())
        if fn not in allowed:
            bad(MAIN, main, m.start(), "g_data_ptr_cache is touched in %s, outside store/lookup/new_graph" % fn)
    if uses < 4:
        out.append("%s: only %d g_data_ptr_cache accesses found; the gate would pass vacuously" % (MAIN, uses))
    lookup = func_bodies(main, "ggml_sycl_data_ptr_cache_lookup")
    store = func_bodies(main, "ggml_sycl_data_ptr_cache_store")
    if len(lookup) != 1 or len(store) != 1:
        out.append("%s: expected one lookup and one store, found %d and %d" % (MAIN, len(lookup), len(store)))
    else:
        lb = text_of(main, lookup[0])
        sb = text_of(main, store[0])
        if not re.search(r"\btenant_gen\s*==\s*g_tenant_publish_gen\b", lb):
            bad(MAIN, main, lookup[0][0], "the lookup does not compare tenant_gen against g_tenant_publish_gen")
        if "ggml_sycl_tensor_slice_identity(" not in lb or not re.search(r"==\s*\w+(?:->|\.)(?:second\.)?src\b|\bsrc\s*==", lb):
            bad(MAIN, main, lookup[0][0], "the lookup does not compare src against ggml_sycl_tensor_slice_identity( computed now")
        if "ggml_sycl_tensor_slice_identity(" not in sb or not re.search(r"\btenant_gen\s*=", sb):
            bad(MAIN, main, store[0][0], "the fill does not stamp tenant_gen from ggml_sycl_tensor_slice_identity(")

    # (C) the one identity function takes its offset from the resolved address
    ident = [b for b in func_bodies(main, "ggml_sycl_tensor_slice_identity")]
    if len(ident) != 1:
        out.append("%s: expected one ggml_sycl_tensor_slice_identity definition, found %d" % (MAIN, len(ident)))
    else:
        ib = text_of(main, ident[0])
        if "ggml_sycl_tensor_device_address(" not in ib or "resolve()" not in ib:
            bad(MAIN, main, ident[0][0], "the slice identity does not take its offset from the resolved address")
        for m in re.finditer(r"ggml_sycl_narrow_storage_identity\s*\(", ib):
            close = close_of(ib, m.end() - 1)
            args = ib[m.end():close]
            if re.search(r"view_offs|view_offset", args):
                bad(MAIN, main, ident[0][0] + m.start(), "the slice identity takes its offset from view_offset")
        if len(re.findall(r"ggml_sycl_narrow_storage_identity\s*\(", ib)) < 2:
            bad(MAIN, main, ident[0][0], "the slice identity must narrow both the SYCL and the SYCL_Host root")

    # (D) g_tenant_publish_gen has one writer; the bump has the listed callers
    bump = func_bodies(main, "ggml_sycl_tenant_publish_gen_bump")
    getter = func_bodies(main, "ggml_sycl_tenant_publish_gen")
    ok_spans = bump + getter
    gen_uses = 0
    for m in re.finditer(r"\bg_tenant_publish_gen\b", main):
        gen_uses += 1
        pos = m.start()
        if any(o <= pos <= e for o, e in ok_spans):
            if any(o <= pos <= e for o, e in getter) and re.match(r"\s*\.\s*(?!load\b)", main[m.end():m.end() + 12]):
                bad(MAIN, main, pos, "g_tenant_publish_gen is written outside its bump function")
            continue
        tail = main[m.end():m.end() + 14]
        if re.match(r"\s*\.\s*load\s*\(", tail) or re.match(r"\s*\{\s*1\s*\}\s*;", tail):
            continue
        # the declaration (`static std::atomic<uint64_t> g_tenant_publish_gen{ 1 };`) or a use that is not a plain load
        if re.search(r"std::atomic<uint64_t>\s*$", main[max(0, pos - 40):pos]):
            continue
        bad(MAIN, main, pos, "g_tenant_publish_gen is written outside its bump function")
    if len(bump) != 1 or gen_uses < 4:
        out.append("%s: found %d bump definition(s) and %d uses of g_tenant_publish_gen; the gate would pass vacuously" %
                   (MAIN, len(bump), gen_uses))
    callers = set()
    for m in re.finditer(r"\bggml_sycl_tenant_publish_gen_bump\s*\(", main):
        if any(o <= m.start() <= e for o, e in bump):
            continue
        if re.search(r"\bvoid\s+$", main[max(0, m.start() - 12):m.start()]):
            continue
        fn = enclosing_top(main, m.start())
        callers.add(fn)
    commit_sites = {"ggml_sycl_publish_prepared_plan_locked", "ggml_sycl_teardown_owner_effects"}
    later = {"ggml_backend_sycl_graph_invalidate"}
    for fn in callers - commit_sites - later:
        out.append("%s: ggml_sycl_tenant_publish_gen_bump is called from %s, which is not a commit or teardown site" % (MAIN, fn))
    for fn in commit_sites - callers:
        out.append("%s: %s does not call ggml_sycl_tenant_publish_gen_bump" % (MAIN, fn))

    # (E) symbols that must not exist
    for path in SCOPE:
        c = code(files, path)
        for sym, why in (("ggml_sycl_publish_moe_artifact_handle", "W4 must not exist"),
                         ("release_buffer_refs", "no step 8' (§B.2)")):
            for m in re.finditer(r"\b" + sym + r"\b", c):
                bad(path, c, m.start(), why)

    # (F) mem_handle_identity is built only from the canonical allocator ids
    ident_fn = func_bodies(memh, "mem_handle::identity")
    if len(ident_fn) != 1:
        out.append("%s: expected one mem_handle::identity definition, found %d" % (MEMH, len(ident_fn)))
    else:
        fb = text_of(memh, ident_fn[0])
        for need in ("canonical_allocation_id_", "canonical_generation_"):
            if need not in fb:
                bad(MEMH, memh, ident_fn[0][0], "mem_handle::identity does not read %s" % need)
        for forbid in ("has_stable_owner_identity", "data_", "ptr_", "resolve("):
            if forbid in fb:
                bad(MEMH, memh, ident_fn[0][0], "mem_handle::identity reads %s, which is not the allocator's id" % forbid)
    names = set()
    for path in SCOPE:
        names.update(re.findall(r"\bmem_handle_identity\b\s*[&*]?\s*(\w+)\s*(?:[;=,)({]|\Z)", code(files, path)))
    names.discard("")
    for path in SCOPE:
        c = code(files, path)
        spans = func_bodies(c, "mem_handle::identity")
        for m in re.finditer(r"\b(\w+)\s*(?:\.|->)\s*allocation_id\s*=(?!=)", c):
            if m.group(1) not in names:
                continue
            if any(o <= m.start() <= e for o, e in spans):
                continue
            bad(path, c, m.start(), "an allocation_id of a mem_handle_identity is assigned outside mem_handle::identity")
        for m in re.finditer(r"\bmem_handle_identity\s*(?:\{|\()", c):
            if any(o <= m.start() <= e for o, e in spans):
                continue
            if re.search(r"\bstruct\s+$", c[max(0, m.start() - 12):m.start()]):
                continue
            if re.match(r"\s*[{(]\s*[})]", c[m.end() - 1:m.end() + 8]):
                continue  # an empty value is the invalid identity, not a built one
            bad(path, c, m.start(), "a mem_handle_identity is constructed outside mem_handle::identity")

    # (G) the runtime_minted belt at the persistent writers W2/W3 and W6/W7
    for name in ("ggml_sycl_publish_existing_storage_handle_for_device", "ggml_sycl_publish_f16_attention_dst_handle"):
        spans = func_bodies(main, name)
        if not spans:
            out.append("%s: %s not found" % (MAIN, name))
            continue
        need = 2 if name.endswith("for_device") else 1  # the root's extra and the view's extra
        if max(text_of(main, s).count("ggml_sycl_persistent_publish_allowed(") for s in spans) < need:
            bad(MAIN, main, spans[0][0], "%s lacks the runtime_minted belt (needs %d call(s))" % (name, need))
    belt = func_bodies(main, "ggml_sycl_persistent_publish_allowed")
    if len(belt) != 1 or "runtime_minted" not in text_of(main, belt[0]) or "tenant_cohort" not in text_of(main, belt[0]):
        out.append("%s: the belt must read runtime_minted and tenant_cohort" % MAIN)
    if len(re.findall(r"\bruntime_minted\s*=\s*true\b", main)) < 3:
        out.append("%s: runtime_minted is set at fewer than the three ensure_root_extra sites" % MAIN)

    # (H) every return path of graph_compute runs the exit hooks
    gc = func_bodies(main, "ggml_backend_sycl_graph_compute")
    gc = [s for s in gc if "ggml_sycl_graph_compute_exit_status" in text_of(main, s) or True]
    exit_fn = func_bodies(main, "ggml_sycl_graph_compute_exit")
    exc_fn = func_bodies(main, "ggml_sycl_graph_compute_exception_exit")
    boundary = func_bodies(main, "ggml_backend_sycl_graph_boundary_exception_cleanup")
    if len(gc) != 1 or len(exit_fn) != 1 or len(exc_fn) != 1 or not boundary:
        out.append("%s: expected one graph_compute, exit and exception-exit; found %d, %d, %d" %
                   (MAIN, len(gc), len(exit_fn), len(exc_fn)))
    else:
        body = text_of(main, gc[0])
        base = gc[0][0]
        returns = list(re.finditer(r"\breturn\b[^;]*;", body))
        if len(returns) < 4:
            out.append("%s: only %d returns in graph_compute; the gate would pass vacuously" % (MAIN, len(returns)))
        for m in returns:
            stmt = m.group()
            if "ggml_sycl_graph_compute_exit_status(" in stmt:
                continue
            catch_at = body.rfind("catch", 0, m.start())
            seg = body[catch_at:m.start()] if catch_at >= 0 else ""
            if catch_at < 0 or not ("ggml_sycl_graph_compute_exception_exit(" in seg or
                                    "ggml_backend_sycl_graph_boundary_exception_cleanup(" in seg):
                bad(MAIN, main, base + m.start(), "a graph_compute return path skips the exit hooks: %s" % stmt.strip())
        eb = text_of(main, exit_fn[0])
        for call in ("ggml_sycl_cpu_tg_flush_pending(", "ggml_sycl_moe_ids_cache_new_graph(",
                     "ggml_sycl_data_ptr_cache_new_graph(", "ggml_sycl_graph_staged_owners_publish(",
                     "graph_input_staging_release_tenants("):
            if call not in eb:
                bad(MAIN, main, exit_fn[0][0], "the graph_compute exit does not call %s" % call)
        xb = text_of(main, exc_fn[0])
        for call in ("ggml_sycl_moe_ids_cache_new_graph(", "ggml_sycl_data_ptr_cache_new_graph(", "g_graph_staged_owners"):
            if call not in xb:
                bad(MAIN, main, exc_fn[0][0], "the exceptional exit does not handle %s" % call)
        if not any("ggml_sycl_graph_compute_exception_exit(" in text_of(main, s) for s in boundary):
            bad(MAIN, main, boundary[0][0], "the boundary exception cleanup does not run the exceptional exit")
        # the exit's branch on a recording call: the flush waits only on an eager call or on pending state at a
        # recording exit, and the staging-tenant release is an eager-exit step
        flush_at = eb.find("ggml_sycl_cpu_tg_flush_pending(")
        flush_if = eb.rfind("if (", 0, flush_at) if flush_at >= 0 else -1
        flush_cond = eb[flush_if:flush_at] if flush_if >= 0 else ""
        if "recorded_call" not in flush_cond:
            bad(MAIN, main, exit_fn[0][0], "the exit's flush does not branch on recorded_call (a recording call must not wait)")
        # the whole condition, not the word: an eager call flushes, and a recording call flushes only with pending state
        if not re.search(r"if\s*\(\s*\(\s*!\s*recorded_call\s*\|\|\s*scatter_pending\s*\)\s*&&\s*"
                         r"!\s*ggml_sycl_cpu_tg_exit_flush_skipped_for_test\s*\(\s*\)\s*\)\s*\{[^}]*"
                         r"ggml_sycl_cpu_tg_flush_pending\s*\(", eb):
            bad(MAIN, main, exit_fn[0][0], "the exit's flush condition is not (!recorded_call || scatter_pending) "
                "and the test hook: an eager call must flush, and a recording call only with pending state")
        if not re.search(r"const\s+bool\s+scatter_pending\s*=\s*recorded_call\s*&&\s*"
                         r"ggml_sycl_cpu_tg_pending_any\s*\(\s*\)\s*;", eb):
            bad(MAIN, main, exit_fn[0][0], "scatter_pending is not recorded_call && ggml_sycl_cpu_tg_pending_any()")
        # (the report's text is a string, which the scrub blanks; the branch must warn and abort under STRICT)
        if not re.search(r"if\s*\(\s*scatter_pending\s*\)\s*\{\s*GGML_LOG_WARN\s*\([^;]*;\s*"
                         r"if\s*\(\s*ggml_sycl::ggml_sycl_strict_enabled\s*\(\s*\)\s*\)\s*\{\s*GGML_ABORT\s*\(", eb):
            bad(MAIN, main, exit_fn[0][0], "pending scatter state at a recording exit is not reported "
                "(a warning, and an abort under STRICT)")
        if not re.search(r"if\s*\(\s*!\s*recorded_call\s*&&\s*ctx\s*\)\s*\{[^}]*graph_input_staging_release_tenants\(", eb):
            bad(MAIN, main, exit_fn[0][0], "the staging-tenant release is not an eager-exit step (!recorded_call)")
        if "ggml_sycl_cpu_tg_pending_any(" not in eb:
            bad(MAIN, main, exit_fn[0][0], "the exit does not read the pending MoE scatter state of a recording call")
        # recorded_call has one source: the recorder's begin counter, read before and after the call
        if "ggml_sycl::graph_record_begins()" not in text_of(main, gc[0]):
            bad(MAIN, main, gc[0][0], "graph_compute does not read ggml_sycl::graph_record_begins() before the call")
        st = func_bodies(main, "ggml_sycl_graph_compute_exit_status")
        if len(st) != 1 or not re.search(r"graph_record_begins\(\)\s*!=\s*record_begins_before", text_of(main, st[0])):
            out.append("%s: graph_compute_exit_status does not derive recorded_call from graph_record_begins()" % MAIN)
    # (H2) every recorder counts itself: the scope's constructor, and each hand-ordered recorder, note a begin
    rec = code(files, RECS)
    ctor = re.search(r"graph_recorder_scope\s*\(\s*const\s+slots\s*&\s*s\b[^{]*\{", rec)
    if not ctor or "graph_record_begin_note(" not in rec[ctor.end():close_of(rec, ctor.end() - 1)]:
        out.append("%s: the recorder scope's constructor does not call graph_record_begin_note()" % RECS)
    if len(re.findall(r"\bgraph_record_begin_slot\s*\(\)", rec)) < 3 or "static thread_local" not in rec:
        out.append("%s: the begin counter is not a thread_local read by graph_record_begins/begin_note" % RECS)
    sites = 0
    for m in re.finditer(r"\bbegin_recording\s*\(", main):
        top = [t for t in top_spans(main) if t[1] <= m.start() <= t[2]]
        if not top:
            continue
        func = main[top[0][0]:top[0][2] + 1]
        before = main[top[0][0]:m.start()]
        sites += 1
        if "graph_record_begin_note(" in func or re.search(r"\bggml_sycl_graph_recorder\s+\w+\s*\(", func) or \
                "recorder_.emplace(" in func:
            continue
        # the pause/resume pair re-opens a recording its scope already counted
        if re.search(r"recorder\s*->\s*resume\s*\(\s*\)\s*;\s*g_recording_graph_ptr\s*->\s*$", before.rstrip()[-120:]):
            continue
        bad(MAIN, main, m.start(), "a command graph begins recording with no recorder scope and no graph_record_begin_note()")
    if sites < 6:
        out.append("%s: only %d begin_recording sites found; the gate would pass vacuously" % (MAIN, sites))
    pushes = 0
    for m in re.finditer(r"\bg_graph_staged_owners\s*\.\s*push_back\b", main):
        pushes += 1
        if enclosing_top(main, m.start()) != "ggml_sycl_graph_staged_owner_add":
            bad(MAIN, main, m.start(), "g_graph_staged_owners is filled outside ggml_sycl_graph_staged_owner_add")
    if pushes != 1:
        out.append("%s: expected one g_graph_staged_owners fill, found %d" % (MAIN, pushes))

    # (I) the oneDNN scratch ring's retained-owner clear carries its empty assert
    ring = func_bodies(mainkeep, "pp_moe_onednn_bind_scratch_slot_generation")
    if len(ring) != 1:
        out.append("%s: expected one pp_moe_onednn_bind_scratch_slot_generation, found %d" % (MAIN, len(ring)))
    else:
        rb = text_of(mainkeep, ring[0])
        if (rb.count("[CONTEXT-PLAN-BUG]") < 2 or "retained_owners" not in rb or "ggml_sycl_strict_enabled()" not in rb or
                not re.search(r'GGML_ABORT\s*\(\s*"\[CONTEXT-PLAN-BUG\]', rb)):
            bad(MAIN, mainkeep, ring[0][0], "the retained_owners[slot] clear lacks its empty assert")
    return out


def edit(files, path, old, new, label, count=1):
    """Replace `old` by `new` in one file.  The anchor is matched token by token with any whitespace (or none)
    between tokens, so a clang-format realignment or rewrap does not strand it."""
    pattern = r"\s*".join(re.escape(tok) for tok in re.findall(r"\w+|[^\w\s]", old))
    text = files[path]
    if not re.search(pattern, text):
        raise SystemExit("gate 27: mutant %r: anchor not found in %s: %r" % (label, path, old[:80]))
    out = dict(files)
    out[path] = re.sub(pattern, lambda m: new, text, count=count)
    return out


def mutants(files):
    M = MAIN
    yield ("a mem_handle in moe_ids_cache_key",
           edit(files, COMMON, "ggml_sycl::mem_handle_identity     handle{};", "ggml_sycl::mem_handle                 handle{};", "k1"),
           "moe_ids_cache_key holds a mem_handle")
    yield ("a mem_handle in the mmvq activation cache",
           edit(files, COMMON, "ggml_sycl::mem_handle_identity  cached_src        = {};", "ggml_sycl::mem_handle  cached_src;", "k2"), "holds its source as a mem_handle")
    yield ("a mem_handle in moe_down_shadow_key",
           edit(files, M, "struct moe_down_shadow_key {", "struct moe_down_shadow_key {\n    ggml_sycl::mem_handle stray;", "k3"),
           "moe_down_shadow_key holds a mem_handle")
    yield ("a mem_handle in g_data_ptr_cache's value",
           edit(files, M, "struct ggml_sycl_data_ptr_cache_entry {",
                "struct ggml_sycl_data_ptr_cache_entry {\n    ggml_sycl::mem_handle keep;", "k4"),
           "value type contains a mem_handle")
    yield ("g_data_ptr_cache's value without tenant_gen",
           edit(files, M, "uint64_t                       tenant_gen = 0;", "uint64_t                       stamp = 0;", "k5"),
           "lacks tenant_gen")
    yield ("a lookup that skips the tenant_gen comparison",
           edit(files, M, "it->second.tenant_gen == g_tenant_publish_gen.load(std::memory_order_acquire) &&",
                "true &&", "k6"), "does not compare tenant_gen")
    yield ("a lookup that skips the identity comparison",
           edit(files, M, "ggml_sycl_tensor_slice_identity(tensor, device, &now) && now == it->second.src &&",
                "", "k7"), "does not compare src")
    yield ("a direct g_data_ptr_cache find outside store/lookup",
           edit(files, M, "static void ggml_sycl_moe_ids_cache_new_graph() {",
                "static void ggml_sycl_stray_ptr() { (void) g_data_ptr_cache.find({ nullptr, 0 }); }\n"
                "static void ggml_sycl_moe_ids_cache_new_graph() {", "k8"), "outside store/lookup/new_graph")
    yield ("a slice identity that takes its offset from view_offs",
           edit(files, M, "ggml_sycl_narrow_storage_identity(storage.handle.identity(), storage.handle.size(), off, span, out)",
                "ggml_sycl_narrow_storage_identity(storage.handle.identity(), storage.handle.size(), view_offs, span, out)",
                "k9"), "takes its offset from view_offset")
    yield ("a second writer of g_tenant_publish_gen",
           edit(files, M, "static void ggml_sycl_moe_ids_cache_new_graph() {",
                "static void ggml_sycl_stray_gen() { g_tenant_publish_gen.fetch_add(1); }\n"
                "static void ggml_sycl_moe_ids_cache_new_graph() {", "k10"), "written outside its bump function")
    yield ("the bump called from a graph_compute path",
           edit(files, M, "static void ggml_sycl_graph_staged_owners_publish(ggml_backend_sycl_context * ctx) {\n",
                "static void ggml_sycl_graph_staged_owners_publish(ggml_backend_sycl_context * ctx) {\n"
                "    ggml_sycl_tenant_publish_gen_bump();\n", "k11"), "not a commit or teardown site")
    yield ("W4 present",
           edit(files, M, "static void ggml_sycl_moe_ids_cache_new_graph() {",
                "static void ggml_sycl_publish_moe_artifact_handle() {}\nstatic void ggml_sycl_moe_ids_cache_new_graph() {", "k12"),
           "W4 must not exist")
    yield ("a release_buffer_refs",
           edit(files, M, "static void ggml_sycl_moe_ids_cache_new_graph() {",
                "static void release_buffer_refs() {}\nstatic void ggml_sycl_moe_ids_cache_new_graph() {", "k13"),
           "no step 8'")
    yield ("an identity built from has_stable_owner_identity",
           edit(files, MEMH, "mem_handle_identity mem_handle::identity() const {",
                "mem_handle_identity mem_handle::identity() const {\n    (void) has_stable_owner_identity;", "k14"),
           "which is not the allocator's id")
    yield ("an allocation_id assigned outside identity()",
           edit(files, MEMH, "mem_handle_identity mem_handle::identity() const {",
                "static void stray_identity(mem_handle_identity & i) { i.allocation_id = 7; }\n"
                "mem_handle_identity mem_handle::identity() const {", "k15"), "is assigned outside mem_handle::identity")
    yield ("the belt missing from the W6/W7 writer",
           edit(files, M, 'if (!ggml_sycl_persistent_publish_allowed(root_extra, handle, "publish_existing_storage_handle_for_device") ||',
                "if (false ||", "k16"), "lacks the runtime_minted belt")
    yield ("the belt missing from the W2/W3 writer",
           edit(files, M, 'if (!ggml_sycl_persistent_publish_allowed(extra, published_handle, "publish_f16_attention_dst_handle")) {',
                "if (false) {", "k17"), "lacks the runtime_minted belt")
    yield ("a graph_compute return that skips the exit hooks",
           edit(files, M, "return ggml_sycl_graph_compute_exit_status(\n            backend, ggml_backend_sycl_graph_compute_unchecked(backend, cgraph), record_begins_before);",
                "return ggml_backend_sycl_graph_compute_unchecked(backend, cgraph);", "k18"), "skips the exit hooks")
    yield ("an exit without the eager flush",
           edit(files, M, "            ggml_sycl_cpu_tg_flush_pending();\n        }\n        ggml_sycl_moe_ids_cache_new_graph();",
                "        }\n        ggml_sycl_moe_ids_cache_new_graph();", "k19"), "does not call ggml_sycl_cpu_tg_flush_pending(")
    yield ("an exit without the staged-owner publish",
           edit(files, M, "        ggml_sycl_graph_staged_owners_publish(ctx);\n        if (!recorded_call && ctx) {",
                "        if (!recorded_call && ctx) {", "k20"), "does not call ggml_sycl_graph_staged_owners_publish(")
    yield ("an exit without the staging-tenant release",
           edit(files, M, "(void) ctx->graph_input_staging_release_tenants();", "(void) 0;", "k21"),
           "does not call graph_input_staging_release_tenants(")
    yield ("an exit without the activation-map drop",
           edit(files, M, "        ggml_sycl_moe_ids_cache_new_graph();\n        ggml_sycl_data_ptr_cache_new_graph();\n        ggml_sycl_graph_staged_owners_publish(ctx);",
                "        ggml_sycl_data_ptr_cache_new_graph();\n        ggml_sycl_graph_staged_owners_publish(ctx);", "k22"),
           "does not call ggml_sycl_moe_ids_cache_new_graph(")
    yield ("a staged-owner fill outside the owner-add function",
           edit(files, M, "static void ggml_sycl_moe_ids_cache_new_graph() {",
                "static void ggml_sycl_stray_fill(ggml_sycl::mem_handle h) { g_graph_staged_owners.push_back(std::move(h)); }\n"
                "static void ggml_sycl_moe_ids_cache_new_graph() {", "k23"), "filled outside ggml_sycl_graph_staged_owner_add")
    yield ("the descriptor MoE recorder not counted",
           edit(files, M, "ggml_sycl::graph_record_begin_note();\n        g_recording_graph_ptr       = &moe_graph;",
                "g_recording_graph_ptr       = &moe_graph;", "k25"), "begins recording with no recorder scope")
    yield ("the MoE segment recorder not counted",
           edit(files, M, "ggml_sycl::graph_record_begin_note();\n            g_recording_graph_ptr       = &seg_graph;",
                "g_recording_graph_ptr       = &seg_graph;", "k26"), "begins recording with no recorder scope")
    yield ("the MoE block recorder not counted",
           edit(files, M, "ggml_sycl::graph_record_begin_note();\n            g_recording_graph_ptr       = &block_graph;",
                "g_recording_graph_ptr       = &block_graph;", "k27"), "begins recording with no recorder scope")
    yield ("the recorder scope's constructor not counting",
           edit(files, RECS, "active_      = this;\n        graph_record_begin_note();", "active_      = this;", "k28"),
           "constructor does not call graph_record_begin_note")
    yield ("a new recorder that begins uncounted",
           edit(files, M, "static void ggml_sycl_moe_ids_cache_new_graph() {",
                "static void ggml_sycl_stray_recorder(sycl_ex::command_graph<sycl_ex::graph_state::modifiable> & g, sycl::queue & q) {"
                " g.begin_recording(q); }\nstatic void ggml_sycl_moe_ids_cache_new_graph() {", "k29"),
           "begins recording with no recorder scope")
    yield ("recorded_call not derived from the counter",
           edit(files, M, "ggml_sycl::graph_record_begins() != record_begins_before", "false", "k30"),
           "does not derive recorded_call from graph_record_begins()")
    yield ("the before-read dropped",
           edit(files, M, "const uint64_t record_begins_before = ggml_sycl::graph_record_begins();",
                "const uint64_t record_begins_before = 0;", "k31"), "does not read ggml_sycl::graph_record_begins() before")
    yield ("a recording exit that always flushes",
           edit(files, M, "(!recorded_call || scatter_pending) && !ggml_sycl_cpu_tg_exit_flush_skipped_for_test()",
                "!ggml_sycl_cpu_tg_exit_flush_skipped_for_test()", "k32"), "flush does not branch on recorded_call")
    yield ("a staging-tenant release on a recording exit",
           edit(files, M, "if (!recorded_call && ctx) {", "if (ctx) {", "k33"), "not an eager-exit step")
    yield ("a recording exit that never flushes pending state",
           edit(files, M, "(!recorded_call || scatter_pending) && !ggml_sycl_cpu_tg_exit_flush_skipped_for_test()",
                "(!recorded_call) && !ggml_sycl_cpu_tg_exit_flush_skipped_for_test()", "k35"),
           "flush condition is not (!recorded_call || scatter_pending)")
    yield ("an eager exit that never flushes",
           edit(files, M, "(!recorded_call || scatter_pending) && !ggml_sycl_cpu_tg_exit_flush_skipped_for_test()",
                "(recorded_call || scatter_pending) && !ggml_sycl_cpu_tg_exit_flush_skipped_for_test()", "k36"),
           "flush condition is not (!recorded_call || scatter_pending)")
    yield ("a flush condition that cannot hold",
           edit(files, M, "(!recorded_call || scatter_pending) && !ggml_sycl_cpu_tg_exit_flush_skipped_for_test()",
                "(false && recorded_call) && !ggml_sycl_cpu_tg_exit_flush_skipped_for_test()", "k37"),
           "flush condition is not (!recorded_call || scatter_pending)")
    yield ("a pending read that is always false",
           edit(files, M, "const bool scatter_pending = recorded_call && ggml_sycl_cpu_tg_pending_any();",
                "const bool scatter_pending = false && recorded_call && ggml_sycl_cpu_tg_pending_any();", "k38"),
           "scatter_pending is not recorded_call && ggml_sycl_cpu_tg_pending_any()")
    yield ("the pending-state report dropped",
           edit(files, M, "if (scatter_pending) {\n            GGML_LOG_WARN(", "if (false) {\n            GGML_LOG_WARN(", "k39"),
           "at a recording exit is not reported")
    yield ("the pending read dropped from the exit",
           edit(files, M, "const bool scatter_pending = recorded_call && ggml_sycl_cpu_tg_pending_any();",
                "const bool scatter_pending = recorded_call;", "k34"), "does not read the pending MoE scatter state")
    yield ("the ring clear without its empty assert",
           edit(files, M, "GGML_ABORT(\"[CONTEXT-PLAN-BUG] oneDNN scratch ring slot", "GGML_ABORT(\"oneDNN scratch ring slot", "k24"),
           "lacks its empty assert")


def load(root):
    return {p: open(os.path.join(root, p), encoding="utf-8").read() for p in SCOPE}


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else REPO
    files = load(root)
    found = violations(files)
    status = 0
    for why in found:
        print("FAIL: gate 27: %s" % why)
        status = 1
    if status:
        return status
    built = list(mutants(files))
    if len(built) < 20:
        print("FAIL: gate 27: only %d mutants could be built" % len(built))
        return 1
    for name, m, frag in built:
        res = violations(m)
        if not any(frag in why for why in res):
            print("FAIL: gate 27: mutant went undetected: %s (wanted %r, got %d other finding(s))" % (name, frag, len(res)))
            status = 1
    if status == 0:
        print("PASS: gate 27 producer clauses hold on the real source; %d mutants caught" % len(built))
    return status


if __name__ == "__main__":
    sys.exit(main())
