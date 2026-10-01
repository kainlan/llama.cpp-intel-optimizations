#!/usr/bin/env python3
"""Source gates for the context-tenant planning path (ggml core clauses).

Gate 13: every scheduler call of ggml_gallocr_reserve_n (not the _size form)
in ggml-backend.cpp tests its result and returns false on failure. A reserve
that fails and is ignored leaves the scheduler believing it holds buffers.

Gate 15 (reserve clause): every `return false;` in ggml_gallocr_reserve_n_impl
is preceded, within the failure branch of the vbuffer allocation, by the
invalidation of the node and leaf layouts, by nulling the alias slots that
shared the freed vbuffer, and by resetting the tallocs. The failed reserve
has already rewritten node_allocs while its vbuffer is NULL; without
`galloc->n_nodes = 0; galloc->n_leafs = 0;` the next same-shape alloc_graph
finds no reason to reserve again and places tensors in a NULL vbuffer. Without
the alias nulling, ggml_gallocr_free frees the shared vbuffer twice. Without
the reset, the peak query reports a layout no buffer backs.

Gate 28 (order half): in the llama_context constructor, the backend_buft.push_back
loop and the final `cparams.pipeline_parallel = pipeline_parallel;` decision come
before model.create_memory(. Anything that measures the compute buffers during
construction needs the bufts and the final flag, and create_memory is the first
thing that can allocate. The later clauses of gate 28 (the fixpoint call and the
trial-decision arguments) arrive with the code they name. A dropped decision is
caught by the count check (exactly one of each statement), the two moved
create_memory mutants by the order checks.

Gate 15 (update clauses): llama_kv_cache::update returns FAILED from both K-shift
failure exits (the pending shift stays recorded, so the next decode retries it);
no `->update(` result is discarded in the memory files; memory_update returns
llama_memory_update_result, and both of decode's call sites compare it with FAILED
and return -2 (the retry site's DONE continues); each of memory_update's four failure
branches (apply, null init_full, refused post-update reserve, catch) sets
sched_need_reserve BEFORE it returns FAILED; the post-update graph_reserve sits inside
a try; update() ends `updated ? DONE : NONE`.

Gate 24: every class that overrides init_full overrides init_reserve, in the header
and in the definitions; every init_reserve body is `make_unique<X_context>(this,
n_streams)` (recurrent: init_full()); each composite's s-stream context constructor
forwards n_streams to every part it builds (dsv4 forwards to csa, hca and lid;
hybrid_idx also takes ns_ubatch{ n_streams }); llama_kv_cache_context's s-stream
constructor asserts 1 <= n_streams <= n_stream and spans exactly n_streams streams
(s1 = n_streams - 1, strm.push_back(s), idxs.resize(n_streams)); dsv4's
dsv4_build_full_sinfo builds s1 = n_stream - 1 and strm[s] = s; the full kv and dsv4
raw contexts delegate to the s-stream constructors.

Gate 22 (one strict switch; five clauses over every tracked file, read with git ls-files so it
covers ggml-sycl.cpp): (1) names: every GGML_SYCL_*STRICT* name in ggml/, src/, common/, tools/,
tests/, scripts/ and the three user-facing SYCL docs is the one switch, GGML_SYCL_STRICT_LEASES, or
the single exception GGML_SYCL_UNIFIED_ALLOC_STRICT (until llama.cpp-1obo), or the named non-member
GGML_SYCL_HANDLE_STRICT; (2) exactly one getenv of the switch, inside the function
ggml_sycl_strict_enabled() resolves to (followed one call deep); (3) the non-member stays
report-only: every read of its variable is part of an if condition whose branches hold only
GGML_LOG_* calls, and its occurrence set (definition, extern, env-table entry, three reads) is
closed, which refuses an alias, a reference and a by-reference argument; (4) every
[<FAMILY>-PLAN-BUG] GGML_ABORT is a direct statement of an if whose whole condition is one call of
ggml_sycl_strict_enabled(), no function is a second name for that call, and the census of such
aborts is not empty and holds uwlx's three; (5) the census of where each allowed name occurs on
the post-uwlx base, so a scan that reads nothing cannot pass. The names in this file are
assembled where a retired or second switch is planted, so the names clause passes on this file.
argv[4], when given, is another checkout to scan for gate 22.

All gates prove themselves on mutants of the real source (the gate must fail
on each) and refuse to pass vacuously. Limits, deliberately: the walk is
textual, so a reserve reached through a wrapper is not seen, and the
invalidation is checked as the statements between the vbuffer allocation
call and the return.
argv: [ggml-backend.cpp [ggml-alloc.c [llama-context.cpp [checkout-root]]]]
"""
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BACKEND = os.path.join(REPO, "ggml", "src", "ggml-backend.cpp")
DEFAULT_ALLOC = os.path.join(REPO, "ggml", "src", "ggml-alloc.c")
DEFAULT_CONTEXT = os.path.join(REPO, "src", "llama-context.cpp")

RESERVE_CALL = re.compile(r"\bggml_gallocr_reserve_n\s*\(")
CHECKED_CALL = re.compile(r"\bif\s*\(\s*!\s*ggml_gallocr_reserve_n\s*\(")
IMPL_SIGNATURE = "static bool ggml_gallocr_reserve_n_impl("
INVALIDATE = (
    "galloc->n_nodes = 0;",
    "galloc->n_leafs = 0;",
    "galloc->buffers[k] = NULL;",
    "ggml_dyn_tallocr_reset(galloc->buf_tallocs[k]);",
)
VBUFFER_ALLOC = "ggml_vbuffer_alloc("


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def reserve_call_violations(text):
    """(violations, call_count): each call must be `if (!call)` with a `return false;` before the block closes."""
    lines = strip_comments(text).split("\n")
    out = []
    calls = 0
    for i, line in enumerate(lines):
        if not RESERVE_CALL.search(line):
            continue
        calls += 1
        if not CHECKED_CALL.search(line):
            out.append((i + 1, "result of ggml_gallocr_reserve_n is not tested"))
            continue
        depth = 0
        opened = False
        returned = False
        for j in range(i, len(lines)):
            for ch in lines[j]:
                if ch == "{":
                    depth += 1
                    opened = True
                elif ch == "}":
                    depth -= 1
            if opened and re.search(r"\breturn\s+false\s*;", lines[j]):
                returned = True
            if opened and depth == 0:
                break
        if not returned:
            out.append((i + 1, "failed reserve does not return false"))
    return out, calls


def impl_bounds(lines):
    start = next((i for i, l in enumerate(lines) if l.startswith(IMPL_SIGNATURE)), None)
    if start is None:
        return None
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("}")), len(lines) - 1)
    return start, end


def invalidation_violations(text):
    """(violations, return_count) for every `return false;` in ggml_gallocr_reserve_n_impl."""
    lines = strip_comments(text).split("\n")
    bounds = impl_bounds(lines)
    if bounds is None:
        return [(0, "ggml_gallocr_reserve_n_impl not found")], 0
    out = []
    returns = 0
    for i in range(bounds[0], bounds[1] + 1):
        if not re.search(r"\breturn\s+false\s*;", lines[i]):
            continue
        returns += 1
        start = max((j for j in range(bounds[0], i) if VBUFFER_ALLOC in lines[j]), default=bounds[0])
        window = "\n".join(l.strip() for l in lines[start:i])
        missing = [stmt for stmt in INVALIDATE if stmt not in window]
        if missing:
            out.append((i + 1, "return false without %s before it" % ", ".join(missing)))
    return out, returns


def mutants_backend(text):
    yield "untested reserve", text.replace("if (!ggml_gallocr_reserve_n(", "(void) (ggml_gallocr_reserve_n(", 1)
    lines = text.split("\n")
    for i, l in enumerate(lines):
        if RESERVE_CALL.search(l) and CHECKED_CALL.search(l):
            for j in range(i + 1, min(i + 4, len(lines))):
                if re.search(r"\breturn\s+false\s*;", lines[j]):
                    yield "dropped return false at line %d" % (j + 1), "\n".join(lines[:j] + ["        (void) 0;"] + lines[j + 1:])
                    break


def mutants_alloc(text):
    lines = text.split("\n")
    bounds = impl_bounds(lines)
    if bounds is None:
        return
    for i in range(bounds[0], bounds[1] + 1):
        if re.search(r"\breturn\s+false\s*;", lines[i]):
            for stmt in INVALIDATE:
                k = max((j for j in range(i) if stmt in lines[j]), default=None)
                if k is None:
                    continue
                yield "dropped `%s`" % stmt, "\n".join(lines[:k] + lines[k + 1:])
            k0 = max((j for j in range(i) if INVALIDATE[0] in lines[j]), default=i)
            yield "extra return false ahead of the invalidation", "\n".join(
                lines[:k0] + ["    if (galloc == NULL) { return false; }"] + lines[k0:])
            break


CTOR_SIGNATURE = "llama_context::llama_context("
BUFT_PUSH = re.compile(r"\bbackend_buft\.push_back\(")
PP_DECISION = re.compile(r"\bcparams\.pipeline_parallel\s*=\s*pipeline_parallel\s*;")
CREATE_MEMORY = re.compile(r"\bmodel\.create_memory\(")


def ctor_lines(text):
    """The comment-stripped lines of the llama_context constructor, or None."""
    lines = strip_comments(text).split("\n")
    start = next((i for i, l in enumerate(lines) if l.startswith(CTOR_SIGNATURE)), None)
    if start is None:
        return None
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("}")), len(lines) - 1)
    return start, lines[start:end + 1]


def ctor_order_violations(text):
    """(violations, counts): enumeration and the pipeline-parallel decision precede create_memory."""
    ctor = ctor_lines(text)
    if ctor is None:
        return [(0, "llama_context constructor not found")], None
    first, lines = ctor
    # file line numbers: the slice's start plus its own offset, 1-based
    at = lambda i: first + 1 + i
    pushes = [i for i, l in enumerate(lines) if BUFT_PUSH.search(l)]
    decisions = [i for i, l in enumerate(lines) if PP_DECISION.search(l)]
    creates = [i for i, l in enumerate(lines) if CREATE_MEMORY.search(l)]
    counts = (len(pushes), len(decisions), len(creates))
    out = []
    if len(pushes) != 1 or len(decisions) != 1 or len(creates) != 1:
        out.append((0, "expected exactly one backend_buft.push_back, one pipeline_parallel decision and one "
                       "model.create_memory in the constructor, found %d, %d and %d" % counts))
        return out, counts
    if pushes[0] > creates[0]:
        out.append((at(creates[0]), "model.create_memory runs before the backend_buft.push_back loop"))
    if decisions[0] > creates[0]:
        out.append((at(creates[0]), "model.create_memory runs before cparams.pipeline_parallel is decided"))
    if decisions[0] < pushes[0]:
        out.append((at(decisions[0]), "pipeline_parallel is decided before the backends are enumerated"))
    return out, counts


def mutants_context(text):
    """Move model.create_memory( ahead of each of the two things it must follow; drop the decision."""
    lines = text.split("\n")
    stripped = strip_comments(text).split("\n")
    start = next((i for i, l in enumerate(stripped) if l.startswith(CTOR_SIGNATURE)), None)
    if start is None:
        return
    end = next((i for i in range(start + 1, len(stripped)) if stripped[i].startswith("}")), len(stripped) - 1)
    find = lambda rx: next((i for i in range(start, end + 1) if rx.search(stripped[i])), None)
    push, decision, create = find(BUFT_PUSH), find(PP_DECISION), find(CREATE_MEMORY)
    if None in (push, decision, create):
        return
    # create_memory's statement is one line, ahead of whatever line `target` names
    for name, target in (("create_memory ahead of the enumeration", push), ("create_memory ahead of the decision", decision)):
        mutated = lines[:target] + [lines[create]] + lines[target:create] + lines[create + 1:]
        yield name, "\n".join(mutated)
    yield "pipeline_parallel decision dropped", "\n".join(lines[:decision] + lines[decision + 1:])


SRC_DIR = os.path.join(REPO, "src")
SRC_FILE_RX = re.compile(r"^llama-(kv-cache|memory|context).*\.(h|cpp)$")


def load_src_files(src_dir):
    out = {}
    for name in sorted(os.listdir(src_dir)):
        if SRC_FILE_RX.match(name):
            with open(os.path.join(src_dir, name), encoding="utf-8") as f:
                out[name] = f.read()
    return out


def body_of(text, signature_rx):
    """(start_line_index, lines) of the top-level function whose first line matches, comment-stripped."""
    lines = strip_comments(text).split("\n")
    start = next((i for i, l in enumerate(lines) if re.match(signature_rx, l)), None)
    if start is None:
        return None
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("}")), len(lines) - 1)
    return start, lines[start:end + 1]


def block_of(lines, anchor_rx):
    """The statements inside the first block opened by a line matching anchor_rx (up to the closing brace at or
    left of the anchor's indentation), as a joined string; None when the anchor is missing."""
    start = next((i for i, l in enumerate(lines) if re.search(anchor_rx, l)), None)
    if start is None:
        return None
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = start + 1
    while end < len(lines):
        l = lines[end]
        if l.strip() and len(l) - len(l.lstrip()) <= indent and l.lstrip().startswith("}"):
            break
        end += 1
    return "\n".join(lines[start + 1:end])


# the failure exits of memory_update: (anchor, whether the branch must set sched_need_reserve)
MEMORY_UPDATE_EXITS = (
    (r"if \(!mctx->apply\(\)\) \{", True),
    (r"if \(!mctx\) \{", True),
    (r"if \(!gf\) \{", True),
    (r"\} catch \(", True),
)


def update_clause_violations(files):
    """Gate 15's update clauses over the llama-kv-cache / llama-memory / llama-context files."""
    out = []
    kv = files.get("llama-kv-cache.cpp", "")
    fn = body_of(kv, r"^llama_memory_update_result llama_kv_cache::update\(")
    if fn is None:
        out.append(("llama-kv-cache.cpp", 0, "llama_kv_cache::update does not return llama_memory_update_result"))
    else:
        first, lines = fn
        exits = [i for i, l in enumerate(lines) if "failed to allocate compute graph for K-shift" in l
                 or "failed to compute K-shift" in l]
        if len(exits) != 2:
            out.append(("llama-kv-cache.cpp", first + 1, "expected the two K-shift failure logs, found %d" % len(exits)))
        for i in exits:
            nxt = next((l.strip() for l in lines[i + 1:] if l.strip()), "")
            if nxt != "return LLAMA_MEMORY_UPDATE_FAILED;":
                out.append(("llama-kv-cache.cpp", first + i + 2, "a K-shift failure exit does not return FAILED"))
        if any(re.search(r"\breturn\s+updated\s*;", l) for l in lines):
            out.append(("llama-kv-cache.cpp", first + 1, "update still returns a bool-style `updated`"))
        stmts = [l.strip() for l in lines if l.strip()]
        if len(stmts) < 2 or stmts[-2] != "return updated ? LLAMA_MEMORY_UPDATE_DONE : LLAMA_MEMORY_UPDATE_NONE;":
            out.append(("llama-kv-cache.cpp", first + 1, "update's tail does not return `updated ? DONE : NONE`"))

    discarded = 0
    seen = 0
    for name, text in files.items():
        if not name.endswith(".cpp") or not (name.startswith("llama-kv-cache") or name.startswith("llama-memory")):
            continue
        for i, line in enumerate(strip_comments(text).split("\n")):
            if re.search(r"->update\(", line) and "init_update" not in line:
                seen += 1
                if re.match(r"^\s*[\w:\.\->\[\]]+->update\(", line):
                    out.append((name, i + 1, "the result of ->update( is discarded"))
    if seen < 1:
        out.append(("src", 0, "no ->update( call found in the memory files; the gate would pass vacuously"))

    ctx_cpp = files.get("llama-context.cpp", "")
    ctx_h = files.get("llama-context.h", "")
    if not re.search(r"^\s*llama_memory_update_result\s+memory_update\(bool optimize\);", strip_comments(ctx_h), re.M):
        out.append(("llama-context.h", 0, "memory_update is not declared to return llama_memory_update_result"))
    mu = body_of(ctx_cpp, r"^llama_memory_update_result llama_context::memory_update\(")
    if mu is None:
        out.append(("llama-context.cpp", 0, "llama_context::memory_update does not return llama_memory_update_result"))
    else:
        first, lines = mu
        gr = next((i for i, l in enumerate(lines) if "graph_reserve(" in l), None)
        tr = next((i for i, l in enumerate(lines) if re.search(r"\btry\s*\{", l)), None)
        ca = next((i for i, l in enumerate(lines) if re.search(r"\bcatch\s*\(", l)), None)
        if gr is None or tr is None or ca is None or not (tr < gr < ca):
            out.append(("llama-context.cpp", first + 1, "the post-update graph_reserve is not inside a try"))
        for anchor, needs_flag in MEMORY_UPDATE_EXITS:
            blk = block_of(lines, anchor)
            if blk is None:
                out.append(("llama-context.cpp", first + 1, "memory_update has no failure branch matching `%s`" % anchor))
                continue
            # a violation names the failing branch, not memory_update's signature
            at = first + next(i for i, l in enumerate(lines) if re.search(anchor, l)) + 1
            if "return LLAMA_MEMORY_UPDATE_FAILED;" not in blk:
                out.append(("llama-context.cpp", at, "the memory_update branch `%s` does not return FAILED" % anchor))
            if needs_flag and "sched_need_reserve = true;" not in blk:
                out.append(("llama-context.cpp", at, "the memory_update branch `%s` does not set sched_need_reserve" % anchor))
            elif needs_flag and "return LLAMA_MEMORY_UPDATE_FAILED;" in blk and \
                    blk.index("sched_need_reserve = true;") > blk.index("return LLAMA_MEMORY_UPDATE_FAILED;"):
                out.append(("llama-context.cpp", at, "the memory_update branch `%s` sets sched_need_reserve after it returns" % anchor))
        body_stmts = [l.strip() for l in lines if l.strip()]
        if len(body_stmts) < 2 or body_stmts[-2] != "return LLAMA_MEMORY_UPDATE_DONE;":
            out.append(("llama-context.cpp", first + 1, "memory_update does not end by returning DONE"))
    calls = 0
    cl = strip_comments(ctx_cpp).split("\n")
    for i, line in enumerate(cl):
        if re.search(r"\bmemory_update\(", line) and not line.startswith("llama_memory_update_result llama_context::"):
            calls += 1
            window = " ".join(cl[i:i + 4])
            if "LLAMA_MEMORY_UPDATE_FAILED" not in window:
                out.append(("llama-context.cpp", i + 1, "a memory_update call does not check LLAMA_MEMORY_UPDATE_FAILED"))
    if calls < 2:
        out.append(("llama-context.cpp", 0, "found %d memory_update call sites, expected decode's two" % calls))
    # each decode FAILED compare returns -2 and the retry site's DONE compare continues
    n_failed = 0
    for i, line in enumerate(cl):
        if "LLAMA_MEMORY_UPDATE_FAILED" in line and re.search(r"\bif\b", line):
            blk = block_of(cl, re.escape(line.strip()))
            if blk is None or not re.search(r"\breturn -2;", blk):
                out.append(("llama-context.cpp", i + 1, "a decode FAILED branch does not return -2"))
            n_failed += 1
    if n_failed < 2:
        out.append(("llama-context.cpp", 0, "found %d decode FAILED branches, expected two" % n_failed))
    done = [i for i, l in enumerate(cl) if re.search(r"\bif\b.*LLAMA_MEMORY_UPDATE_DONE", l)]
    if len(done) != 1 or not re.search(r"\bcontinue;", block_of(cl, re.escape(cl[done[0]].strip())) or ""):
        out.append(("llama-context.cpp", 0, "the retry site's DONE branch must exist once and continue"))
    return out


def update_clause_mutants(files):
    def with_file(name, text):
        m = dict(files)
        m[name] = text
        return m

    kv = files["llama-kv-cache.cpp"]
    for what in ("failed to allocate compute graph for K-shift", "failed to compute K-shift"):
        i = kv.index(what)
        j = kv.index("return LLAMA_MEMORY_UPDATE_FAILED;", i)
        yield "K-shift exit (%s) returns NONE" % what.split()[-1], with_file(
            "llama-kv-cache.cpp", kv[:j] + "return LLAMA_MEMORY_UPDATE_NONE;" + kv[j + len("return LLAMA_MEMORY_UPDATE_FAILED;"):])
    yield "apply discards update's result", with_file(
        "llama-kv-cache.cpp", kv.replace("return kv->update(lctx, do_shift, sc_info) != LLAMA_MEMORY_UPDATE_FAILED;",
                                         "kv->update(lctx, do_shift, sc_info);\n        return true;", 1))
    ch = files["llama-context.h"]
    yield "memory_update returns bool", with_file(
        "llama-context.h", ch.replace("llama_memory_update_result memory_update(bool optimize);", "bool memory_update(bool optimize);", 1))
    cc = files["llama-context.cpp"]
    yield "first decode call ignores FAILED", with_file(
        "llama-context.cpp", cc.replace("if (memory_update(false) == LLAMA_MEMORY_UPDATE_FAILED) {", "if (memory_update(false) == LLAMA_MEMORY_UPDATE_DONE) {", 1))
    a = cc.index("llama_memory_update_result llama_context::memory_update(")
    t = cc.index("    try {", a)
    yield "post-update reserve outside a try", with_file("llama-context.cpp", cc[:t] + "    {" + cc[t + len("    try {"):])

    def drop_after(text, anchor, token, replacement=""):
        i = text.index(anchor)
        j = text.index(token, i)
        return text[:j] + replacement + text[j + len(token):]

    d1 = "if (memory_update(false) == LLAMA_MEMORY_UPDATE_FAILED) {"
    yield "first decode call: `return -2;` deleted", with_file("llama-context.cpp", drop_after(cc, d1, "return -2;"))
    d2 = "if (update_res == LLAMA_MEMORY_UPDATE_FAILED) {"
    yield "retry site: `return -2;` deleted", with_file("llama-context.cpp", drop_after(cc, d2, "return -2;"))
    d3 = "if (update_res == LLAMA_MEMORY_UPDATE_DONE) {"
    yield "retry site: DONE no longer continues", with_file("llama-context.cpp", drop_after(cc, d3, "continue;"))
    for anchor, name in (("if (!mctx->apply()) {", "apply failure"), ("if (!mctx) {", "null init_full"),
                         ("if (!gf) {", "refused post-update reserve"), ("} catch (const std::exception & err) {", "catch handler")):
        base = cc.index("llama_memory_update_result llama_context::memory_update(")
        yield "memory_update %s: no longer returns FAILED" % name, with_file(
            "llama-context.cpp", cc[:base] + drop_after(cc[base:], anchor, "return LLAMA_MEMORY_UPDATE_FAILED;", "return LLAMA_MEMORY_UPDATE_NONE;"))
        yield "memory_update %s: no longer sets sched_need_reserve" % name, with_file(
            "llama-context.cpp", cc[:base] + drop_after(cc[base:], anchor, "sched_need_reserve = true;"))
    base = cc.index("llama_memory_update_result llama_context::memory_update(")
    ap = cc.index("if (!mctx->apply()) {", base)
    fl = cc.index("return LLAMA_MEMORY_UPDATE_FAILED;", ap)
    st = cc.index("sched_need_reserve = true;", ap)
    swapped = (cc[:st] + cc[fl:fl + len("return LLAMA_MEMORY_UPDATE_FAILED;")] + cc[st + len("sched_need_reserve = true;"):fl]
               + "sched_need_reserve = true;" + cc[fl + len("return LLAMA_MEMORY_UPDATE_FAILED;"):])
    yield "apply failure: sched_need_reserve stored after the return", with_file("llama-context.cpp", swapped)
    yield "update tail returns DONE unconditionally", with_file(
        "llama-kv-cache.cpp", kv.replace("return updated ? LLAMA_MEMORY_UPDATE_DONE : LLAMA_MEMORY_UPDATE_NONE;", "return LLAMA_MEMORY_UPDATE_DONE;", 1))


COMPOSITES = (
    # (class, source file, number of init_reserve( calls its context constructor must make with n_streams, extra tokens)
    ("llama_kv_cache_iswa_context", "llama-kv-cache-iswa.cpp", 2, ()),
    ("llama_kv_cache_dsa_context", "llama-kv-cache-dsa.cpp", 2, ()),
    ("llama_kv_cache_dsa_iswa_context", "llama-kv-cache-dsa-iswa.cpp", 2, ()),
    ("llama_kv_cache_msa_context", "llama-kv-cache-msa.cpp", 2, ()),
    ("llama_memory_hybrid_context", "llama-memory-hybrid.cpp", 2, ()),
    ("llama_memory_hybrid_iswa_context", "llama-memory-hybrid-iswa.cpp", 2, ()),
    # the indexer's per-part stream count (ns_ubatch) is s too, not the idx cache's own n_stream
    ("llama_memory_hybrid_idx_context", "llama-memory-hybrid-idx.cpp", 0,
     ("llama_memory_hybrid_context(mem, n_streams)", "std::vector<uint32_t>{ n_streams }",
      "llama_kv_cache_context(mem->get_mem_idx(), n_streams)")),
    ("llama_kv_cache_dsv4_context", "llama-kv-cache-dsv4.cpp", 3,
     ("llama_kv_cache_dsv4_raw_context>(kv->get_raw(), n_streams)",
      "llama_kv_cache_dsv4_comp_context>(kv->get_csa(), n_streams)",
      "llama_kv_cache_dsv4_comp_context>(kv->get_hca(), n_streams)",
      "llama_kv_cache_dsv4_comp_context>(kv->get_lid(), n_streams)")),
)


def stream_ctor(text, cls):
    """The comment-stripped s-stream constructor of `cls`: from its definition to the first col-0 closing brace."""
    lines = strip_comments(text).split("\n")
    for i, l in enumerate(lines):
        if l.startswith("%s::%s(" % (cls, cls)):
            end = next((k for k in range(i, len(lines)) if lines[k].startswith("}")), len(lines) - 1)
            chunk = "\n".join(lines[i:end + 1])
            if re.search(r"\buint32_t\s+n_streams\b", chunk.split("{")[0]) and "lctx" not in chunk.split("{")[0]:
                return i, chunk
    return None


def reserve_clause_violations(files):
    """Gate 24 over the llama-kv-cache / llama-memory files."""
    out = []
    heads = {n: strip_comments(t) for n, t in files.items() if n.endswith(".h")}
    n_full = sum(len(re.findall(r"init_full\(\)\s+override\s*;", t)) for t in heads.values())
    n_res = sum(len(re.findall(r"init_reserve\(uint32_t n_streams\)\s+override\s*;", t)) for t in heads.values())
    if n_full < 10:
        out.append(("src", 0, "found %d init_full overrides, expected at least ten; the gate would pass vacuously" % n_full))
    if n_res != n_full:
        out.append(("src", 0, "%d init_reserve overrides against %d init_full overrides" % (n_res, n_full)))
    full_defs, res_defs = set(), set()
    for n, t in files.items():
        if n.endswith(".cpp"):
            st = strip_comments(t)
            full_defs |= set(re.findall(r"(\w+)::init_full\(\)\s*\{", st))
            res_defs |= set(re.findall(r"(\w+)::init_reserve\(uint32_t n_streams\)\s*\{", st))
    for c in sorted(full_defs - res_defs):
        out.append(("src", 0, "%s defines init_full but not init_reserve" % c))

    # what init_reserve builds: the class's own context over (this, n_streams); the recurrent memory ignores s
    bodies = 0
    for n, t in files.items():
        if not n.endswith(".cpp"):
            continue
        for cls, body in re.findall(r"(\w+)::init_reserve\(uint32_t n_streams\)\s*\{(.*?)\n\}", strip_comments(t), re.S):
            bodies += 1
            got = " ".join(body.split())
            if cls == "llama_memory_recurrent":
                want = "GGML_UNUSED(n_streams); return init_full();"
            else:
                want = "return std::make_unique<%s_context>(this, n_streams);" % cls
            if got != want:
                out.append((n, 0, "%s::init_reserve must be `%s`, found `%s`" % (cls, want, got)))
    if bodies < n_res:
        out.append(("src", 0, "read %d init_reserve bodies against %d declarations" % (bodies, n_res)))

    for cls, fname, n_calls, tokens in COMPOSITES:
        text = files.get(fname, "")
        ctor = stream_ctor(text, cls)
        if ctor is None:
            out.append((fname, 0, "%s has no s-stream constructor" % cls))
            continue
        first, chunk = ctor
        calls = re.findall(r"init_reserve\(([^)]*)\)", chunk)
        if len(calls) != n_calls or any(a.strip() != "n_streams" for a in calls):
            out.append((fname, first + 1, "%s must forward n_streams to its %d init_reserve calls, found %r" %
                        (cls, n_calls, calls)))
        for tok in tokens:
            if tok not in chunk:
                out.append((fname, first + 1, "%s does not contain `%s`" % (cls, tok)))

    kv = strip_comments(files.get("llama-kv-cache.cpp", ""))
    m = re.search(r"llama_kv_cache_context::llama_kv_cache_context\(\n\s*llama_kv_cache \* kv,\n\s*uint32_t n_streams\)(.*?)\n\}\n", kv, re.S)
    if m is None:
        out.append(("llama-kv-cache.cpp", 0, "llama_kv_cache_context has no s-stream constructor"))
    else:
        body = m.group(1)
        for tok in ("sinfos[0].s1 = n_streams - 1;", "sinfos[0].idxs.resize(n_streams);", "s < n_streams",
                    "GGML_ASSERT(n_streams >= 1 && n_streams <= kv->get_n_stream());", "sinfos[0].strm.push_back(s);"):
            if tok not in body:
                out.append(("llama-kv-cache.cpp", 0, "the s-stream constructor lacks `%s`" % tok))
    # the full contexts are the s-stream ones over every stream, by delegation, so their equality is not a convention
    full_kv = " ".join(kv.split())
    if "llama_kv_cache_context::llama_kv_cache_context( llama_kv_cache * kv) : llama_kv_cache_context(kv, kv->get_n_stream()) {" not in full_kv:
        out.append(("llama-kv-cache.cpp", 0, "the full llama_kv_cache_context does not delegate to the s-stream constructor"))
    dsv4_text = strip_comments(files.get("llama-kv-cache-dsv4.cpp", ""))
    helper = body_of(dsv4_text, r"^static llama_kv_cache::slot_info dsv4_build_full_sinfo\(")
    if helper is None:
        out.append(("llama-kv-cache-dsv4.cpp", 0, "dsv4_build_full_sinfo is missing"))
    else:
        htext = "\n".join(helper[1])
        for tok in ("GGML_ASSERT(n_stream >= 1 && n_stream <= kv->get_n_stream());", "sinfo.s1 = n_stream - 1;",
                    "sinfo.resize(n_stream);", "s < n_stream", "sinfo.strm[s] = s;"):
            if tok not in htext:
                out.append(("llama-kv-cache-dsv4.cpp", helper[0] + 1, "dsv4_build_full_sinfo lacks `%s`" % tok))
    dsv4 = " ".join(dsv4_text.split())
    if ("llama_kv_cache_dsv4_raw_context::llama_kv_cache_dsv4_raw_context(llama_kv_cache_iswa * kv) : "
            "llama_kv_cache_dsv4_raw_context(kv, kv->get_swa()->get_n_stream()) {") not in dsv4:
        out.append(("llama-kv-cache-dsv4.cpp", 0, "the full dsv4 raw context does not delegate to the s-stream constructor"))
    return out


def reserve_clause_mutants(files):
    def with_file(name, text):
        m = dict(files)
        m[name] = text
        return m

    h = files["llama-kv-cache-iswa.h"]
    yield "an init_reserve override dropped", with_file(
        "llama-kv-cache-iswa.h", h.replace("llama_memory_context_ptr init_reserve(uint32_t n_streams) override;", "", 1))
    for cls, fname, n_calls, tokens in COMPOSITES:
        text = files[fname]
        ctor = stream_ctor(text, cls)
        if ctor is None:
            continue
        stripped_first = ctor[1].split("\n")[0]
        i = text.index(stripped_first)
        if n_calls:
            j = text.index("init_reserve(n_streams)", i)
            yield "%s forwards 1 instead of n_streams" % cls, with_file(
                fname, text[:j] + "init_reserve(1)" + text[j + len("init_reserve(n_streams)"):])
        for tok in tokens:
            k = text.index(tok, i)
            yield "%s drops `%s`" % (cls, tok[:40]), with_file(
                fname, text[:k] + tok.replace("n_streams", "1") + text[k + len(tok):])
    kv = files["llama-kv-cache.cpp"]
    yield "kv s-stream constructor spans every stream", with_file(
        "llama-kv-cache.cpp", kv.replace("sinfos[0].s1 = n_streams - 1;", "sinfos[0].s1 = kv->get_n_stream() - 1;", 1))
    idx = files["llama-memory-hybrid-idx.cpp"]
    yield "hybrid_idx ns_ubatch takes the idx cache's own stream count", with_file(
        "llama-memory-hybrid-idx.cpp",
        idx.replace("std::vector<uint32_t>{ n_streams }", "std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }", 1))
    yield "kv s-stream constructor assert dropped", with_file(
        "llama-kv-cache.cpp", kv.replace("GGML_ASSERT(n_streams >= 1 && n_streams <= kv->get_n_stream());", "", 1))
    yield "kv s-stream constructor maps every stream to stream 0", with_file(
        "llama-kv-cache.cpp", kv.replace("sinfos[0].strm.push_back(s);", "sinfos[0].strm.push_back(0);", 1))
    dvh = files["llama-kv-cache-dsv4.cpp"]
    yield "dsv4_build_full_sinfo s1 off by one", with_file(
        "llama-kv-cache-dsv4.cpp", dvh.replace("sinfo.s1 = n_stream - 1;", "sinfo.s1 = n_stream;", 1))
    yield "dsv4_build_full_sinfo maps every stream to stream 0", with_file(
        "llama-kv-cache-dsv4.cpp", dvh.replace("sinfo.strm[s] = s;", "sinfo.strm[s] = 0;", 1))
    yield "full kv context no longer delegates", with_file(
        "llama-kv-cache.cpp", kv.replace(": llama_kv_cache_context(kv, kv->get_n_stream()) {", ": status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv) {", 1))
    dv = files["llama-kv-cache-dsv4.cpp"]
    yield "full dsv4 raw context no longer delegates", with_file(
        "llama-kv-cache-dsv4.cpp", dv.replace("llama_kv_cache_dsv4_raw_context(kv, kv->get_swa()->get_n_stream()) {", "llama_kv_cache_dsv4_raw_context(kv, 1) {", 1))
    # an init_reserve that ignores n_streams, one mutant per class that builds a context
    for fname, t in files.items():
        if not fname.endswith(".cpp"):
            continue
        for cls in re.findall(r"(\w+)::init_reserve\(uint32_t n_streams\)\s*\{", t):
            a = t.index("%s::init_reserve(uint32_t n_streams)" % cls)
            if cls == "llama_memory_recurrent":
                j = t.index("return init_full();", a)
                yield "%s::init_reserve returns nullptr" % cls, with_file(fname, t[:j] + "return nullptr;" + t[j + len("return init_full();"):])
            else:
                j = t.index("(this, n_streams)", a)
                yield "%s::init_reserve ignores n_streams" % cls, with_file(fname, t[:j] + "(this)" + t[j + len("(this, n_streams)"):])
    rec = files["llama-memory-recurrent.cpp"]
    yield "recurrent init_reserve dropped", with_file(
        "llama-memory-recurrent.cpp", rec.replace("llama_memory_context_ptr llama_memory_recurrent::init_reserve(uint32_t n_streams) {", "static void unused_reserve(uint32_t n_streams) {", 1))


# ---------------------------------------------------------------------------
# Gate 22: one strict switch. GGML_SYCL_STRICT_LEASES is the only abort switch
# for ownership, lifetime and plan defects; it is read once, by the function
# ggml_sycl_strict_enabled() resolves to, and every [<FAMILY>-PLAN-BUG] abort
# asks that accessor directly. The scan is a plain file read of every tracked
# file (git ls-files), so it covers ggml-sycl.cpp, which codescout skips as
# oversize.
# ---------------------------------------------------------------------------
# The names below are assembled so this file, which the names clause scans like
# any other tests/ file, does not itself carry a retired or second switch name.
STRICT_NAME_RX = re.compile(r"GGML_SYCL_[A-Z0-9_]*STRICT[A-Z0-9_]*")
STRICT_NAME_DIRS = ("ggml/", "src/", "common/", "tools/", "tests/", "scripts/")
STRICT_NAME_DOCS = (
    "docs/backend/sycl-env-vars.md",
    "docs/backend/sycl-memory-design.md",
    "docs/design/sycl-canonical-memory-architecture.md",
)
STRICT_CODE_DIRS = ("ggml/", "src/", "common/", "tools/", "tests/")
STRICT_CODE_SUFFIXES = (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".cu", ".cuh", ".m", ".mm")
STRICT_LEASES = "GGML_SYCL_" + "STRICT_LEASES"
STRICT_ALLOC = "GGML_SYCL_" + "UNIFIED_ALLOC_STRICT"
STRICT_HANDLE = "GGML_SYCL_" + "HANDLE_STRICT"
STRICT_HANDLE_VAR = "g_ggml_sycl_" + "handle_strict"
# name -> why it is allowed. The UNIFIED_ALLOC entry carries its retirement ticket.
STRICT_NAMES_ALLOWED = {
    STRICT_LEASES: "the one abort switch",
    STRICT_ALLOC: "the single exception, until llama.cpp-1obo retires it",
}
# Excluded by name and not by the allowlist: a diagnostic report switch, not an
# abort switch (rulings section M41 m-4). Clause 3 holds it to that.
STRICT_NAMES_NONMEMBER = {
    STRICT_HANDLE: "diagnostic report switch; every read only widens GGML_LOG_* output, it never aborts",
}
# Census on the post-uwlx base, by file (line numbers drift): each name must still
# occur where the base has it, so a scan that reads nothing cannot pass.
STRICT_CENSUS = {
    STRICT_LEASES: ("ggml/src/ggml-sycl/unified-cache.cpp", "ggml/src/ggml-sycl/mem-handle.cpp",
                    "docs/backend/sycl-env-vars.md", "docs/backend/sycl-memory-design.md",
                    "docs/design/sycl-canonical-memory-architecture.md"),
    STRICT_ALLOC: ("ggml/src/ggml-sycl/unified-cache.cpp", "ggml/src/ggml-sycl/tests/test-unified-runtime-alloc.cpp"),
    STRICT_HANDLE: ("ggml/src/ggml-sycl/ggml-sycl.cpp", "tests/test-sycl-env-report.cpp",
                    "docs/backend/sycl-env-vars.md"),
}
STRICT_ACCESSOR_NAME = "ggml_sycl_strict_enabled"
STRICT_ACCESSOR_COND = re.compile(r"^(?:::)?(?:ggml_sycl::)?ggml_sycl_strict_enabled\(\s*\)$")
STRICT_PLAN_BUG = re.compile(r"GGML_ABORT\s*\(\s*\"(\[[A-Z0-9]+(?:-[A-Z0-9]+)*-PLAN-BUG\])")
# the aborts of the post-uwlx base; zhcn's own add theirs as they land
STRICT_ABORT_CENSUS = (
    ("ggml/src/ggml-sycl/mem-handle.cpp", "[CONTEXT-PLAN-BUG] retained-reap backstop", 1),
    ("ggml/src/ggml-sycl/mem-handle.cpp", "[CONTEXT-PLAN-BUG] retained-reap %s: owner", 1),
    ("ggml/src/ggml-sycl/unified-cache.cpp", "[CONTEXT-PLAN-BUG] optional-layout yield", 1),
)
# where the one non-member variable is allowed to occur, by (file, kind)
STRICT_HANDLE_SITES = {
    ("ggml/src/ggml-sycl/ggml-sycl.cpp", "definition"): 1,
    ("ggml/src/ggml-sycl/ggml-sycl.cpp", "env table"): 1,
    ("ggml/src/ggml-sycl/ggml-sycl.cpp", "read"): 1,
    ("ggml/src/ggml-sycl/common.hpp", "extern"): 1,
    ("ggml/src/ggml-sycl/common.hpp", "read"): 2,
}

_C_LITERAL = re.compile(
    r"//[^\n]*"
    r"|/\*.*?\*/"
    r"|R\"([^(\s]{0,16})\(.*?\)\1\""
    r"|\"(?:\\.|[^\"\\\n])*\""
    r"|'(?:\\.|[^'\\\n])*'",
    re.S)


def scrub_c(text, keep_strings):
    """Blank comments with spaces (newlines kept), and string and char contents too unless keep_strings.
    Offsets and line numbers match the input."""
    def blank(s):
        return re.sub(r"[^\n]", " ", s)

    def one(m):
        s = m.group(0)
        if s.startswith("/"):
            return blank(s)
        return s if keep_strings else s[0] + blank(s[1:-1]) + s[-1]
    return _C_LITERAL.sub(one, text)


_SCRUB_CACHE = {}


def scrubbed(text, keep_strings):
    key = (hash(text), len(text), keep_strings)
    hit = _SCRUB_CACHE.get(key)
    if hit is None or hit[0] != text:
        if len(_SCRUB_CACHE) > 24:
            _SCRUB_CACHE.clear()
        hit = (text, scrub_c(text, keep_strings))
        _SCRUB_CACHE[key] = hit
    return hit[1]


def line_of(text, pos):
    return text.count("\n", 0, pos) + 1


def close_of(s, i):
    """Index of the bracket closing the one opened at s[i] (strings blanked), or -1."""
    pair = {"(": ")", "{": "}", "[": "]"}
    open_ch, close_ch = s[i], pair[s[i]]
    depth = 0
    for j in range(i, len(s)):
        if s[j] == open_ch:
            depth += 1
        elif s[j] == close_ch:
            depth -= 1
            if depth == 0:
                return j
    return -1


def enclosing_brace(s, pos):
    depth = 0
    for j in range(pos - 1, -1, -1):
        if s[j] == "}":
            depth += 1
        elif s[j] == "{":
            if depth == 0:
                return j
            depth -= 1
    return -1


def open_of(s, i):
    """Index of the '(' matching the ')' at s[i], or -1."""
    depth = 0
    for j in range(i, -1, -1):
        if s[j] == ")":
            depth += 1
        elif s[j] == "(":
            depth -= 1
            if depth == 0:
                return j
    return -1


def if_condition_around(s, pos):
    """(open, close) of the `if (...)` condition that holds s[pos], or None when pos is not inside one: a call
    argument, an initializer, a while, a return and every other use gives None."""
    j = pos - 1
    depth = 0
    while j >= 0:
        c = s[j]
        if c == ")":
            depth += 1
        elif c == "(":
            if depth:
                depth -= 1
            else:
                head = s[:j].rstrip()
                word = re.search(r"([A-Za-z_]\w*)$", head)
                if word and word.group(1) == "if":
                    end = close_of(s, j)
                    return (j, end) if end > pos else None
                if word or head.endswith(("]", ")")):
                    return None
        elif c in ";{}" and not depth:
            return None
        j -= 1
    return None


def controlled_region(s, cond_close):
    """The statements an if controls, its else branches included, as text; the first is the then-branch."""
    parts = []
    i = cond_close + 1
    while True:
        while i < len(s) and s[i].isspace():
            i += 1
        if i >= len(s):
            break
        if s[i] == "{":
            end = close_of(s, i)
            if end < 0:
                break
            parts.append(s[i + 1:end])
            i = end + 1
        else:
            end = s.find(";", i)
            if end < 0:
                break
            parts.append(s[i:end + 1])
            i = end + 1
        m = re.match(r"\s*else\b\s*", s[i:])
        if not m:
            break
        i += m.end()
        if s.startswith("if", i) and re.match(r"if\b\s*\(", s[i:]):
            i = s.index("(", i)
            i = close_of(s, i) + 1
            continue
        while i < len(s) and s[i].isspace():
            i += 1
        if i < len(s) and s[i] == "{":
            end = close_of(s, i)
            parts.append(s[i + 1:end])
            i = end + 1
        else:
            end = s.find(";", i)
            parts.append(s[i:end + 1])
        break
    return parts


def only_log_calls(region):
    """True when region, with its GGML_LOG_* calls removed, holds no call, jump or throw."""
    rest = region
    while True:
        m = re.search(r"\bGGML_LOG_[A-Z]+\s*\(", rest)
        if not m:
            break
        end = close_of(rest, m.end() - 1)
        if end < 0:
            return False
        rest = rest[:m.start()] + rest[end + 1:]
    return not re.search(r"[A-Za-z_]\w*\s*\(|\b(?:return|throw|goto|break|continue)\b", rest)


def strict_scope(files):
    """The tracked files each clause reads: (name_files, code_files)."""
    name_files = {n: t for n, t in files.items() if n.startswith(STRICT_NAME_DIRS) or n in STRICT_NAME_DOCS}
    code_files = {n: t for n, t in files.items()
                  if n.startswith(STRICT_CODE_DIRS) and n.endswith(STRICT_CODE_SUFFIXES)}
    return name_files, code_files


def function_bodies(code, name):
    """Spans (start, end) of the bodies of every definition of `bool name()` in comment- and string-blanked code."""
    out = []
    for m in re.finditer(r"\bbool\s+(?:\w+::)?" + re.escape(name) + r"\s*\(\s*\)\s*\{", code):
        end = close_of(code, m.end() - 1)
        if end > 0:
            out.append((m.end() - 1, end))
    return out


def strict_names_violations(name_files):
    out = []
    for name, text in sorted(name_files.items()):
        if "STRICT" not in text:
            continue
        for m in STRICT_NAME_RX.finditer(text):
            found = m.group(0)
            if found in STRICT_NAMES_ALLOWED or found in STRICT_NAMES_NONMEMBER:
                continue
            out.append((name, line_of(text, m.start()),
                        "%s is not the strict switch or one of its two named exceptions" % found))
    for required in STRICT_NAME_DOCS + ("ggml/src/ggml-sycl/ggml-sycl.cpp", "ggml/src/ggml-sycl/unified-cache.cpp"):
        if required not in name_files:
            out.append((required, 0, "the scan did not read this tracked file; the names clause would pass vacuously"))
    return out


def strict_census_violations(name_files):
    out = []
    for name, where in STRICT_CENSUS.items():
        for path in where:
            if path in name_files and name not in name_files[path]:
                out.append((path, 0, "%s no longer occurs here; the census was taken on the post-uwlx base" % name))
    for name, why in list(STRICT_NAMES_ALLOWED.items()) + list(STRICT_NAMES_NONMEMBER.items()):
        if not why:
            out.append(("gate 22", 0, "%s carries no reason" % name))
    if "llama.cpp-1obo" not in STRICT_NAMES_ALLOWED[STRICT_ALLOC]:
        out.append(("gate 22", 0, "the UNIFIED_ALLOC exception does not name its retirement ticket"))
    return out


def strict_getenv_violations(code_files):
    """Exactly one getenv of the switch, inside the function ggml_sycl_strict_enabled() resolves to."""
    out = []
    calls = []
    literal_starts = set()
    for name, text in sorted(code_files.items()):
        if STRICT_LEASES not in text:
            continue
        code = scrubbed(text, True)
        for m in re.finditer(r"getenv\s*\(\s*(\"" + STRICT_LEASES + r"\")", code):
            calls.append((name, m.start()))
            literal_starts.add((name, m.start(1)))
    if len(calls) != 1:
        out.append(("tree", 0, "found %d getenv of the strict switch, expected exactly 1: %s" %
                    (len(calls), ["%s:%d" % (n, line_of(code_files[n], p)) for n, p in calls])))
    # a second reader that does not spell getenv: the quoted name held in a variable. Tests may set and unset it.
    for name, text in sorted(code_files.items()):
        if "/tests/" in name or name.startswith("tests/") or STRICT_LEASES not in text:
            continue
        for m in re.finditer(r"\"" + STRICT_LEASES + r"\"", scrubbed(text, True)):
            if (name, m.start()) not in literal_starts:
                out.append((name, line_of(text, m.start()), "the quoted strict switch name appears outside its one reader"))
    exports = []
    for name, text in sorted(code_files.items()):
        if STRICT_ACCESSOR_NAME not in text:
            continue
        for span in function_bodies(scrubbed(text, False), STRICT_ACCESSOR_NAME):
            exports.append((name, span))
    if len(exports) != 1:
        out.append(("tree", 0, "found %d definitions of the accessor, expected 1" % len(exports)))
        return out
    name, (start, end) = exports[0]
    code = scrubbed(code_files[name], False)
    reader = (start, end)
    fwd = re.fullmatch(r"\{\s*return\s+(\w+)\s*\(\s*\)\s*;\s*\}", code[start:end + 1])
    if fwd and fwd.group(1) != STRICT_ACCESSOR_NAME:
        spans = function_bodies(code, fwd.group(1))
        if len(spans) != 1:
            out.append((name, line_of(code, start), "the accessor forwards to %s, which has %d definitions here, "
                        "expected 1" % (fwd.group(1), len(spans))))
            return out
        reader = spans[0]
    if len(calls) == 1 and not (calls[0][0] == name and reader[0] < calls[0][1] < reader[1]):
        out.append((calls[0][0], line_of(code_files[calls[0][0]], calls[0][1]),
                    "the one getenv of the switch is not inside the function the accessor resolves to"))
    return out


def strict_handle_violations(code_files):
    """The non-member stays report-only and its occurrence set is closed."""
    out = []
    seen = {}
    reads = 0
    for name, text in sorted(code_files.items()):
        if STRICT_HANDLE_VAR not in text:
            continue
        keep = scrubbed(text, True)
        code = scrubbed(text, False)
        for m in re.finditer(r"\b" + STRICT_HANDLE_VAR + r"\b", code):
            pos = m.start()
            line = line_of(text, pos)
            sol = code.rfind("\n", 0, pos) + 1
            eol = code.find("\n", pos)
            row = keep[sol:eol if eol >= 0 else len(keep)]
            if re.match(r"\s*int\s+" + STRICT_HANDLE_VAR + r"\s*=\s*0\s*;", row):
                kind = "definition"
            elif re.match(r"\s*extern\s+int\s+" + STRICT_HANDLE_VAR + r"\s*;", row):
                kind = "extern"
            elif re.match(r"\s*\{\s*\"" + STRICT_HANDLE + r"\"\s*,\s*&" + STRICT_HANDLE_VAR + r"\s*,\s*0\s*\}", row):
                kind = "env table"
            else:
                cond = if_condition_around(code, pos)
                if cond is None:
                    out.append((name, line, "%s is used other than as part of an if condition (an alias, a reference, "
                                "an argument or an initializer)" % STRICT_HANDLE_VAR))
                    continue
                region = controlled_region(code, cond[1])
                if not region or not only_log_calls("\n".join(region)) or "GGML_LOG_" not in region[0]:
                    out.append((name, line, "a read of %s gates something other than GGML_LOG_* output" % STRICT_HANDLE_VAR))
                    continue
                kind = "read"
                reads += 1
            seen[(name, kind)] = seen.get((name, kind), 0) + 1
    for key in sorted(set(seen) | set(STRICT_HANDLE_SITES)):
        if seen.get(key, 0) != STRICT_HANDLE_SITES.get(key, 0):
            out.append((key[0], 0, "%s: %d %s site(s), the closed set has %d" %
                        (STRICT_HANDLE_VAR, seen.get(key, 0), key[1], STRICT_HANDLE_SITES.get(key, 0))))
    if reads < 1:
        out.append(("tree", 0, "no read of %s found; the clause would pass vacuously" % STRICT_HANDLE_VAR))
    return out


def strict_accessor_violations(code_files):
    """Every [<FAMILY>-PLAN-BUG] GGML_ABORT is a direct statement of an if whose whole condition is one call of
    ggml_sycl_strict_enabled(); no function is a second name for that call."""
    out = []
    found = {}
    total = 0
    for name, text in sorted(code_files.items()):
        if "PLAN-BUG" not in text and STRICT_ACCESSOR_NAME not in text:
            continue
        keep = scrubbed(text, True)
        code = scrubbed(text, False)
        for m in re.finditer(r"(\w+)\s*\(\s*\)\s*\{\s*return\s+(?:::)?(?:ggml_sycl::)?"
                             + STRICT_ACCESSOR_NAME + r"\s*\(\s*\)\s*;\s*\}", code):
            out.append((name, line_of(text, m.start()), "%s only returns %s(): a second name for one fact"
                        % (m.group(1), STRICT_ACCESSOR_NAME)))
        for m in STRICT_PLAN_BUG.finditer(keep):
            total += 1
            pos = m.start()
            line = line_of(text, pos)
            literal = re.match(r"GGML_ABORT\s*\(\s*((?:\"(?:\\.|[^\"\\])*\"\s*)+)", keep[pos:pos + 600])
            message = "".join(re.findall(r"\"((?:\\.|[^\"\\])*)\"", literal.group(1))) if literal else m.group(1)
            for want_file, want_text, _ in STRICT_ABORT_CENSUS:
                if want_file == name and message.startswith(want_text):
                    found[(want_file, want_text)] = found.get((want_file, want_text), 0) + 1
            cond = None
            # unbraced: `if (cond) GGML_ABORT(...)`; braced: the abort sits directly in the block of `if (cond) {`
            prev = code[:pos].rstrip()
            if prev.endswith(")"):
                op = open_of(code, len(prev) - 1)
                if re.search(r"(?<![\w])if$", code[:op].rstrip()):
                    cond = re.sub(r"\s+", "", code[op + 1:len(prev) - 1])
            else:
                brace = enclosing_brace(code, pos)
                head = code[:brace].rstrip() if brace >= 0 else ""
                if head.endswith(")"):
                    op = open_of(code, len(head) - 1)
                    before = code[:op].rstrip()
                    if re.search(r"(?<![\w])if$", before) and not re.search(r"\belse\s+if$", before):
                        cond = re.sub(r"\s+", "", code[op + 1:len(head) - 1])
            if cond is None or not STRICT_ACCESSOR_COND.match(cond):
                out.append((name, line, "%s abort is not a direct statement of `if (%s())` (condition: %s)"
                            % (m.group(1), STRICT_ACCESSOR_NAME, cond if cond is not None else "none")))
    if total < 1:
        out.append(("tree", 0, "no [<FAMILY>-PLAN-BUG] abort found; the clause would pass vacuously"))
    for want_file, want_text, count in STRICT_ABORT_CENSUS:
        if want_file in code_files and found.get((want_file, want_text), 0) != count:
            out.append((want_file, 0, "expected %d abort(s) tagged `%s`, found %d"
                        % (count, want_text, found.get((want_file, want_text), 0))))
    return out


def strict_violations(files):
    """Gate 22's five clauses (names, one getenv, non-member report-only, accessor, census) over tracked files."""
    name_files, code_files = strict_scope(files)
    return (strict_names_violations(name_files) + strict_getenv_violations(code_files)
            + strict_handle_violations(code_files) + strict_accessor_violations(code_files)
            + strict_census_violations(name_files))


def tracked_files(root):
    """{path: text} of every tracked file in gate 22's scope, via git ls-files; None when git cannot list them."""
    try:
        listing = subprocess.run(["git", "-C", root, "ls-files", "-z"], check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    out = {}
    for rel in listing.decode("utf-8", "replace").split("\0"):
        if not rel or not (rel.startswith(STRICT_NAME_DIRS) or rel in STRICT_NAME_DOCS):
            continue
        path = os.path.join(root, rel)
        if os.path.isfile(path) and not os.path.islink(path):
            with open(path, encoding="utf-8", errors="replace") as f:
                out[rel] = f.read()
    return out


STRICT_RETIRED = "GGML_SYCL_" + "STRICT_PLAN"
STRICT_SECOND = "GGML_SYCL_" + "PLAN_STRICT"
_GS = "ggml/src/ggml-sycl/ggml-sycl.cpp"
_UC = "ggml/src/ggml-sycl/unified-cache.cpp"
_MH = "ggml/src/ggml-sycl/mem-handle.cpp"
_CH = "ggml/src/ggml-sycl/common.hpp"
_BACKSTOP = ('    if (ggml_sycl_strict_enabled()) {\n        GGML_ABORT(\n'
             '            "[CONTEXT-PLAN-BUG] retained-reap backstop')


def strict_mutants(files):
    """(name, mutated files, fragment the violations must carry), plus (name, files, None) positive controls that
    must stay clean. Every mutant is a small edit of the real tracked source."""
    def edit(path, old, new, label):
        if old not in files[path]:
            raise AssertionError("gate 22 mutant `%s`: anchor missing in %s" % (label, path))
        return {**files, path: files[path].replace(old, new, 1)}

    def append(path, text):
        return {**files, path: files[path] + "\n" + text + "\n"}

    wrapper = "static bool strict_plan_checks_enabled() { return ggml_sycl_strict_enabled(); }\n\n"
    handle_if = "        if (!installed_source_handle) {"
    yield ("planted retired switch in ggml-sycl.cpp", append(_GS, 'static const char * g_retired = "%s";' % STRICT_RETIRED),
           "is not the strict switch")
    yield ("planted second switch in a test", append("tests/test-sycl-env-report.cpp", "// reads %s" % STRICT_SECOND),
           "is not the strict switch")
    yield ("planted second switch in a script", append("scripts/sycl-build.sh", "# export %s=1" % STRICT_SECOND),
           "is not the strict switch")
    yield ("planted second switch in a user-facing doc", append("docs/backend/sycl-env-vars.md", "| `%s=1` | x |" % STRICT_SECOND),
           "is not the strict switch")
    yield ("suffixed spelling of the one switch", append(_MH, "// %sS" % STRICT_LEASES), "is not the strict switch")
    yield ("scan that never read ggml-sycl.cpp", {k: v for k, v in files.items() if k != _GS}, "did not read this tracked file")
    yield ("second getenv of the switch", append(_MH, 'static bool second() { return std::getenv("%s") != nullptr; }' % STRICT_LEASES),
           "found 2 getenv")
    yield ("second getenv, bare and spaced, in src/", append("src/llama.cpp", 'static const char * e = getenv( "%s" );' % STRICT_LEASES),
           "found 2 getenv")
    yield ("the getenv removed", edit(_UC, 'std::getenv("%s")' % STRICT_LEASES, "nullptr", "getenv removed"), "found 0 getenv")
    yield ("the switch name held in a variable", append(_MH, 'static const char * k = "%s";' % STRICT_LEASES),
           "outside its one reader")
    reader = edit(_UC, "bool ggml_sycl_strict_enabled() {\n    return strict_lease_checks_enabled();\n}",
                  "static bool strict_other_reader() {\n    return false;\n}\n\nbool ggml_sycl_strict_enabled() {\n"
                  "    return strict_other_reader();\n}", "accessor repointed")
    yield ("the accessor resolves to a function without the getenv", reader, "not inside the function the accessor resolves to")
    yield ("a block conditioned on a read-only abort beside a report",
           edit(_CH, handle_if, '        if (%s) { GGML_ABORT("x"); }\n' % STRICT_HANDLE_VAR + handle_if, "abort beside report"),
           "gates something other than GGML_LOG_*")
    yield ("an address alias of the non-member",
           append(_GS, "static void alias() { const int * p = &%s; if (*p) { GGML_ABORT(\"x\"); } }" % STRICT_HANDLE_VAR),
           "other than as part of an if condition")
    yield ("a reference alias of the non-member",
           append(_GS, "static void alias() { const int & r = %s; if (r) { GGML_ABORT(\"x\"); } }" % STRICT_HANDLE_VAR),
           "other than as part of an if condition")
    yield ("a by-reference argument of the non-member",
           append(_GS, "static void sink(const int & v); static void pass() { sink(%s); }" % STRICT_HANDLE_VAR),
           "other than as part of an if condition")
    yield ("an unbraced abort under the non-member",
           append(_GS, "static void f() { if (%s) GGML_ABORT(\"x\"); }" % STRICT_HANDLE_VAR),
           "gates something other than GGML_LOG_*")
    yield ("an abort in the else of a non-member read",
           append(_GS, "static void f() { if (%s) { GGML_LOG_INFO(\"a\"); } else { GGML_ABORT(\"x\"); } }" % STRICT_HANDLE_VAR),
           "gates something other than GGML_LOG_*")
    yield ("a call beside the log under the non-member",
           append(_GS, "static void f() { if (%s) { GGML_LOG_INFO(\"a\"); die(); } }" % STRICT_HANDLE_VAR),
           "gates something other than GGML_LOG_*")
    yield ("an assert under the non-member",
           append(_GS, "static void f() { if (%s) { GGML_ASSERT(false); } }" % STRICT_HANDLE_VAR),
           "gates something other than GGML_LOG_*")
    yield ("a fourth, log-only read of the non-member (the set is closed)",
           append(_GS, "static void f() { if (%s) { GGML_LOG_INFO(\"a\"); } }" % STRICT_HANDLE_VAR),
           "the closed set has")
    yield ("a zhcn plan-bug abort gated on the other strict mode",
           append("src/llama-context.cpp",
                  "static void bug() { if (ggml_sycl::unified_alloc_strict_mode()) { GGML_ABORT(\"[CONTEXT-PLAN-BUG] z\"); } }"),
           "not a direct statement")
    yield ("a wrapper reintroduced and gating a uwlx abort",
           edit(_MH, _BACKSTOP, _BACKSTOP.replace("ggml_sycl_strict_enabled()", "strict_plan_checks_enabled()"), "wrapper use"),
           "not a direct statement")
    yield ("a wrapper reintroduced",
           edit(_MH, "namespace {\n", "namespace {\n\n" + wrapper, "wrapper definition"), "second name for one fact")
    yield ("a plan-bug abort with no condition",
           edit(_MH, _BACKSTOP, '    {\n        GGML_ABORT(\n            "[CONTEXT-PLAN-BUG] retained-reap backstop', "unconditioned"),
           "not a direct statement")
    yield ("a plan-bug abort under an extra condition",
           edit(_MH, _BACKSTOP, _BACKSTOP.replace("if (ggml_sycl_strict_enabled())", "if (ggml_sycl_strict_enabled() && count > 1)"),
                "extra condition"), "not a direct statement")
    tail = ': %s %s event incomplete after the caller\'s "\n            "synchronize (GGML_SYCL_STRICT_LEASES=1)",\n            reason, entry);\n    }'
    yield ("a plan-bug abort nested one block deeper",
           edit(_MH, _BACKSTOP + tail,
                '    if (ggml_sycl_strict_enabled()) {\n        if (count > 1) {\n            GGML_ABORT(\n'
                '                "[CONTEXT-PLAN-BUG] retained-reap backstop' + tail.replace("    }", "    }\n    }", 1), "nested"),
           "not a direct statement")
    yield ("an unbraced plan-bug abort under the wrong condition",
           append("src/llama-context.cpp", "static void bug() { if (ggml_sycl::unified_alloc_strict_mode()) GGML_ABORT(\"[CONTEXT-PLAN-BUG] z\"); }"),
           "not a direct statement")
    yield ("a VM-family plan-bug abort left unconditioned",
           edit(_UC, '        if (ggml_sycl_strict_enabled()) {\n            GGML_ABORT("[VM-PLAN-BUG] ggml_sycl_arena_backing read',
                '        if (true) {\n            GGML_ABORT("[VM-PLAN-BUG] ggml_sycl_arena_backing read', "vm abort"),
           "not a direct statement")
    yield ("a uwlx abort retagged out of the census",
           edit(_MH, "[CONTEXT-PLAN-BUG] retained-reap backstop: %s %s event incomplete after the caller's \"\n            \"synchronize (",
                "[CONTEXT-PLAN-BUG] reap backstop: %s %s event incomplete after the caller's \"\n            \"synchronize (", "retag"),
           "expected 1 abort(s) tagged")
    yield ("the switch dropped from a user-facing doc",
           {**files, "docs/backend/sycl-memory-design.md": files["docs/backend/sycl-memory-design.md"].replace(STRICT_LEASES, "X")},
           "no longer occurs here")
    # positive controls: legitimate shapes the clauses must not refuse
    folded = edit(_UC, "static bool strict_lease_checks_enabled() {\n    static const bool enabled = [] {\n"
                       "        const char * env = std::getenv(\"%s\");\n        return env != nullptr && std::atoi(env) != 0;\n"
                       "    }();\n    return enabled;\n}\n\nbool ggml_sycl_strict_enabled() {\n    return strict_lease_checks_enabled();\n}" % STRICT_LEASES,
                  "bool ggml_sycl_strict_enabled() {\n    static const bool enabled = [] {\n"
                  "        const char * env = std::getenv(\"%s\");\n        return env != nullptr && std::atoi(env) != 0;\n"
                  "    }();\n    return enabled;\n}" % STRICT_LEASES, "folded accessor")
    yield ("positive control: the two functions folded into the accessor", folded, None)
    yield ("positive control: an unbraced plan-bug abort on the accessor",
           append("src/llama-context.cpp", "static void ok() { if (ggml_sycl_strict_enabled()) GGML_ABORT(\"[CONTEXT-PLAN-BUG] z\"); }"),
           None)
    yield ("positive control: a qualified accessor call",
           append("src/llama-context.cpp", "static void ok() { if (::ggml_sycl_strict_enabled()) { GGML_ABORT(\"[CONTEXT-PLAN-BUG] z\"); } }"),
           None)


def main():
    backend_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BACKEND
    alloc_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_ALLOC
    context_path = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_CONTEXT
    with open(backend_path, encoding="utf-8") as f:
        backend = f.read()
    with open(alloc_path, encoding="utf-8") as f:
        alloc = f.read()
    with open(context_path, encoding="utf-8") as f:
        context = f.read()
    status = 0
    src_files = load_src_files(os.path.dirname(context_path))
    mutant_counts = {}

    bad, calls = reserve_call_violations(backend)
    if calls < 2:
        print("FAIL: gate 13: found %d ggml_gallocr_reserve_n calls in %s (expected the alloc and the reserve path); "
              "the gate would pass vacuously" % (calls, backend_path))
        status = 1
    for line_no, why in bad:
        print("FAIL: gate 13: %s:%d: %s" % (backend_path, line_no, why))
        status = 1
    missed13 = [name for name, m in mutants_backend(backend) if not reserve_call_violations(m)[0]]
    n13 = sum(1 for _ in mutants_backend(backend))
    if n13 < 3:
        print("FAIL: gate 13: only %d mutants could be built" % n13)
        status = 1
    for name in missed13:
        print("FAIL: gate 13: mutant went undetected: %s" % name)
        status = 1

    bad, returns = invalidation_violations(alloc)
    if returns < 1:
        print("FAIL: gate 15: no `return false;` found in ggml_gallocr_reserve_n_impl; the gate would pass vacuously")
        status = 1
    for line_no, why in bad:
        print("FAIL: gate 15: %s:%d: %s" % (alloc_path, line_no, why))
        status = 1
    missed15 = [name for name, m in mutants_alloc(alloc) if not invalidation_violations(m)[0]]
    n15 = sum(1 for _ in mutants_alloc(alloc))
    if n15 < 3:
        print("FAIL: gate 15: only %d mutants could be built" % n15)
        status = 1
    for name in missed15:
        print("FAIL: gate 15: mutant went undetected: %s" % name)
        status = 1

    bad, counts = ctor_order_violations(context)
    for line_no, why in bad:
        print("FAIL: gate 28: %s:%d: %s" % (context_path, line_no, why))
        status = 1
    muts28 = list(mutants_context(context))
    if len(muts28) < 3:
        print("FAIL: gate 28: only %d mutants could be built" % len(muts28))
        status = 1
    for name, m in muts28:
        if not ctor_order_violations(m)[0]:
            print("FAIL: gate 28: mutant went undetected: %s" % name)
            status = 1

    for tag, check, mutants in (("15", update_clause_violations, update_clause_mutants),
                                ("24", reserve_clause_violations, reserve_clause_mutants)):
        bad = check(src_files)
        for name, line_no, why in bad:
            print("FAIL: gate %s: %s:%d: %s" % (tag, name, line_no, why))
            status = 1
        if bad:
            continue
        built = list(mutants(src_files))
        if len(built) < 5:
            print("FAIL: gate %s: only %d mutants could be built" % (tag, len(built)))
            status = 1
        for name, m in built:
            if not check(m):
                print("FAIL: gate %s: mutant went undetected: %s" % (tag, name))
                status = 1
        mutant_counts[tag] = len(built)

    strict_root = sys.argv[4] if len(sys.argv) > 4 else REPO
    tracked = tracked_files(strict_root)
    strict_counts = None
    if tracked is None:
        print("SKIP: gate 22: git cannot list the tracked files under %s; this run proves nothing about the strict "
              "switch" % strict_root)
    else:
        bad = strict_violations(tracked)
        for name, line_no, why in bad:
            print("FAIL: gate 22: %s:%d: %s" % (name, line_no, why))
            status = 1
        if not bad:
            built = list(strict_mutants(tracked))
            if sum(1 for _, _, frag in built if frag is not None) < 25:
                print("FAIL: gate 22: only %d mutants could be built" % len(built))
                status = 1
            for name, m, frag in built:
                found = strict_violations(m)
                if frag is None and found:
                    print("FAIL: gate 22: control refused: %s: %s" % (name, found[0]))
                    status = 1
                elif frag is not None and not any(frag in why for _, _, why in found):
                    print("FAIL: gate 22: mutant went undetected: %s" % name)
                    status = 1
            strict_counts = (sum(1 for _, _, f in built if f is not None), sum(1 for _, _, f in built if f is None))

    if status == 0 and tracked is None:
        return 77
    if status == 0:
        print("PASS: %d scheduler reserve calls are tested and return false (%d mutants caught); "
              "%d reserve_n_impl failure return(s) invalidate the layout first (%d mutants caught); "
              "the constructor enumerates backends and decides pipeline_parallel before create_memory "
              "(%d mutants caught); update clauses of gate 15 (%d mutants caught) and gate 24 (%d mutants caught); "
              "gate 22's five strict clauses (%d mutants caught, %d controls kept clean)" %
              (calls, n13, returns, n15, len(muts28), mutant_counts["15"], mutant_counts["24"],
               strict_counts[0], strict_counts[1]))
    return status


if __name__ == "__main__":
    sys.exit(main())
