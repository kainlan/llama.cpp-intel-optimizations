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

All three gates prove themselves on mutants of the real source (the gate must fail
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

    if status == 0:
        print("PASS: %d scheduler reserve calls are tested and return false (%d mutants caught); "
              "%d reserve_n_impl failure return(s) invalidate the layout first (%d mutants caught); "
              "the constructor enumerates backends and decides pipeline_parallel before create_memory "
              "(%d mutants caught)" % (calls, n13, returns, n15, len(muts28)))
    return status


if __name__ == "__main__":
    sys.exit(main())
