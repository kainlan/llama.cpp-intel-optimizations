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
llama_memory_update_result, and both of decode's call sites compare it with FAILED;
the post-update graph_reserve sits inside a try.

Gate 24: every class that overrides init_full overrides init_reserve, in the header
and in the definitions; each composite's s-stream context constructor forwards
n_streams to every part it builds (dsv4 forwards to csa, hca and lid, or returns the
named refusal); llama_kv_cache_context's s-stream constructor spans exactly n_streams
streams (s1 = n_streams - 1).

All gates prove themselves on mutants of the real source (the gate must fail
on each) and refuse to pass vacuously. Limits, deliberately: the walk is
textual, so a reserve reached through a wrapper is not seen, and the
invalidation is checked as the statements between the vbuffer allocation
call and the return.
argv: [ggml-backend.cpp [ggml-alloc.c [llama-context.cpp]]]
"""
import os
import re
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


COMPOSITES = (
    # (class, source file, number of init_reserve( calls its context constructor must make with n_streams, extra tokens)
    ("llama_kv_cache_iswa_context", "llama-kv-cache-iswa.cpp", 2, ()),
    ("llama_kv_cache_dsa_context", "llama-kv-cache-dsa.cpp", 2, ()),
    ("llama_kv_cache_dsa_iswa_context", "llama-kv-cache-dsa-iswa.cpp", 2, ()),
    ("llama_kv_cache_msa_context", "llama-kv-cache-msa.cpp", 2, ()),
    ("llama_memory_hybrid_context", "llama-memory-hybrid.cpp", 2, ()),
    ("llama_memory_hybrid_iswa_context", "llama-memory-hybrid-iswa.cpp", 2, ()),
    ("llama_memory_hybrid_idx_context", "llama-memory-hybrid-idx.cpp", 0,
     ("llama_memory_hybrid_context(mem, n_streams)", "llama_kv_cache_context(mem->get_mem_idx(), n_streams)")),
    ("llama_kv_cache_dsv4_context", "llama-kv-cache-dsv4.cpp", 3,
     ("llama_kv_cache_dsv4_raw_context>(kv->get_raw(), n_streams)",
      "llama_kv_cache_dsv4_comp_context>(kv->get_csa(), n_streams)",
      "llama_kv_cache_dsv4_comp_context>(kv->get_hca(), n_streams)",
      "llama_kv_cache_dsv4_comp_context>(kv->get_lid(), n_streams)")),
)
DSV4_REFUSAL = "dsv4 s-stream reserve not ported"


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
    if n_full < 9:
        out.append(("src", 0, "found %d init_full overrides, expected at least nine; the gate would pass vacuously" % n_full))
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

    for cls, fname, n_calls, tokens in COMPOSITES:
        text = files.get(fname, "")
        ctor = stream_ctor(text, cls)
        if ctor is None:
            out.append((fname, 0, "%s has no s-stream constructor" % cls))
            continue
        first, chunk = ctor
        calls = re.findall(r"init_reserve\(([^)]*)\)", chunk)
        if cls == "llama_kv_cache_dsv4_context":
            dsv4_fn = body_of(text, r"^llama_memory_context_ptr llama_kv_cache_dsv4::init_reserve\(")
            if dsv4_fn is not None and DSV4_REFUSAL in "\n".join(dsv4_fn[1]) or DSV4_REFUSAL in text and not calls:
                continue
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
        for tok in ("sinfos[0].s1 = n_streams - 1;", "sinfos[0].idxs.resize(n_streams);", "s < n_streams"):
            if tok not in body:
                out.append(("llama-kv-cache.cpp", 0, "the s-stream constructor lacks `%s`" % tok))
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
        if tokens:
            k = text.index(tokens[-1].replace("n_streams)", "n_streams)"), i)
            yield "%s drops `%s`" % (cls, tokens[-1][:40]), with_file(
                fname, text[:k] + tokens[-1].replace("n_streams)", "1)") + text[k + len(tokens[-1]):])
    kv = files["llama-kv-cache.cpp"]
    yield "kv s-stream constructor spans every stream", with_file(
        "llama-kv-cache.cpp", kv.replace("sinfos[0].s1 = n_streams - 1;", "sinfos[0].s1 = kv->get_n_stream() - 1;", 1))
    rec = files["llama-memory-recurrent.cpp"]
    yield "recurrent init_reserve dropped", with_file(
        "llama-memory-recurrent.cpp", rec.replace("llama_memory_context_ptr llama_memory_recurrent::init_reserve(uint32_t n_streams) {", "static void unused_reserve(uint32_t n_streams) {", 1))


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

    if status == 0:
        print("PASS: %d scheduler reserve calls are tested and return false (%d mutants caught); "
              "%d reserve_n_impl failure return(s) invalidate the layout first (%d mutants caught); "
              "the constructor enumerates backends and decides pipeline_parallel before create_memory "
              "(%d mutants caught); update clauses of gate 15 (%d mutants caught) and gate 24 (%d mutants caught)" %
              (calls, n13, returns, n15, len(muts28), mutant_counts["15"], mutant_counts["24"]))
    return status


if __name__ == "__main__":
    sys.exit(main())
