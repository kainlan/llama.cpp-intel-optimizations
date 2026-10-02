"""Source gate for the measure-only context's silence (zhcn design gate 37, the
log-silence clauses, and the structural clauses of the measure-only constructor).

A load-time measure builds a transient `llama_context` through the measure-only
constructor. It prints nothing a vehicle arm scores, and the rule that makes that
true is mechanical, not a list of lines:

  Every log statement, at any level, that is reachable from the shared constructor
  body, `~llama_context`, `sched_reserve`, `sched_reserve_nothrow` or
  `sched_reserve_impl` through functions defined in llama-context.cpp is either
  guarded (in a block whose `if` tests `!measure_only`, ALLOC mode or `plan_caps`,
  or after a top-level early return that only a measure-only or MEASURE caller
  takes), or in a function every reached reference to which is guarded, or on this
  gate's allow-list with its reason.

This gate enumerates the statements from source. It reads comment-stripped text with
string contents masked, finds every top-level function of llama-context.cpp,
computes the reach set from the roots by name, and checks each log statement
(`LLAMA_LOG_*`, `fprintf`, `printf`, `llama_log_internal`) against the rule. It also
pins the structural half: the measure-only constructor creates its memory with
`measure_only` as the no-alloc argument, and every allocating, binding or ladder call
in the constructor body sits in a guarded block; the destructor's buffer-size
comparison and SYCL drain block carry `!measure_only` in their own `if`; the one
caller of `llama_graph_n_input_tensors` passes `!measure_only`.

The walk is textual, so a call reached through a function pointer is seen only as a
reference to the name, and a log that a callee prints on behalf of a guarded call site
is checked at the callee. Every clause has a mutant of the real source that must fail
it, and the gate refuses to pass on an empty census. Host-only; collected by pytest.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_CPP = (ROOT / "src/llama-context.cpp").read_text()

_LEXEME_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"'
    r"|'(?:\\.|[^'\\\n])*'"
    r"|//[^\n]*"
    r"|/\*.*?\*/",
    flags=re.DOTALL,
)


_PP_RE = re.compile(r"^[ \t]*#(?:[^\n]*\\\n)*[^\n]*", flags=re.MULTILINE)


def mask(src: str) -> str:
    """Comments become blank lines and string and character contents become spaces,
    so braces, parentheses and log names inside them are invisible. Offsets and line
    numbers are preserved."""

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok[0] + re.sub(r"[^\n]", " ", tok[1:-1]) + tok[-1]
        return re.sub(r"[^\n]", " ", tok)

    out = _LEXEME_RE.sub(repl, src)
    # preprocessor lines (with their continuations) are not part of any statement's header
    return _PP_RE.sub(lambda mt: re.sub(r"[^\n]", " ", mt.group(0)), out)


# the log sinks of llama-context.cpp: the macros, and the raw writers that bypass them
_LOG_RE = re.compile(r"\b(?:LLAMA_LOG(?:_[A-Z]+)?|fprintf|printf|puts|llama_log_internal)\s*\(")

# a function whose body is a pure struct or enum is not a function
_NOT_FUNCTION = re.compile(r"^\s*(?:typedef|using|struct|class|enum|union|namespace|extern)\b|\btemplate\b")

ROOTS = (
    "llama_context::llama_context#3",  # the shared body: the three-argument constructor
    "llama_context::~llama_context",
    "llama_context::sched_reserve",
    "llama_context::sched_reserve_nothrow",
    "llama_context::sched_reserve_impl",
)

# A line a root reaches that no `!measure_only` test can guard, with the reason. The key is
# (function, a substring of the statement). Each reason is checked below, not taken on faith.
ALLOW = {
    ("llama_context::sched_reserve_nothrow", "%s: %s\\n"): (
        "the nothrow reserve is decode's and encode's; a measure-only context never decodes and its "
        "constructor does not call it (the constructor-body census below refuses the call)"
    ),
}


class Func:
    def __init__(self, name, start, end, body_start, header):
        self.name, self.start, self.end, self.body_start, self.header = name, start, end, body_start, header
        self.blocks = []  # (open, close, header_text)


def _matching_paren(text: str, i: int) -> int:
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                return j
    raise AssertionError("unbalanced parentheses")


def parse_functions(m: str):
    """Top-level definitions of the masked text, in order. A ctor with an init list and a
    function with a trailing `const`/`noexcept` are both found by their first `(`."""
    funcs = []
    depth = 0
    boundary = 0
    ctor_count = 0
    i = 0
    n = len(m)
    while i < n:
        c = m[i]
        if c == "{":
            if depth == 0:
                header = m[boundary:i]
                first = header.find("(")
                if first != -1 and not _NOT_FUNCTION.search(header) and "=" not in header[:first]:
                    close = _matching_paren(header, first)
                    ident = re.findall(r"[~\w:]+", header[:first])
                    name = ident[-1] if ident else ""
                    # the end of the body
                    d = 0
                    j = i
                    while j < n:
                        if m[j] == "{":
                            d += 1
                        elif m[j] == "}":
                            d -= 1
                            if d == 0:
                                break
                        j += 1
                    if name == "llama_context::llama_context":
                        ctor_count += 1
                        name += "#%d" % (1 if "measure" not in header[first:close] else 3)
                    funcs.append(Func(name, boundary, j, i, header))
                    i = j
                    depth = 0
                    boundary = j + 1
                    i += 1
                    continue
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                boundary = i + 1
        elif c == ";" and depth == 0:
            boundary = i + 1
        i += 1
    return funcs


def parse_blocks(m: str, f: Func):
    stack = []
    last = f.body_start + 1
    for i in range(f.body_start, f.end + 1):
        c = m[i]
        if c == "{":
            hdr = m[last:i] if i != f.body_start else ""
            stack.append((i, hdr))
            last = i + 1
        elif c == "}":
            o, hdr = stack.pop()
            f.blocks.append((o, i, hdr))
            last = i + 1
        elif c == ";":
            last = i + 1


_IF_RE = re.compile(r"^\s*(?P<else>else\s+)?if\s*\((?P<cond>.*)\)\s*$", re.DOTALL)


def is_guard_cond(cond: str) -> bool:
    cond = " ".join(cond.split())
    if "||" in cond:
        return False
    conj = [t.strip() for t in cond.split("&&")]
    if "!measure_only" in conj:
        return True
    if "mode == sched_reserve_mode::ALLOC" in conj:
        return True
    # a planned context owns the chunk-cap copy; a measure-only context never does
    return "plan_caps" in conj


def guard_blocks(m: str, f: Func):
    """The guarded ranges of a function: blocks whose `if` is a guard, and the rest of the
    function after a top-level early return that only the unguarded path takes."""
    out = []
    for o, c, hdr in f.blocks:
        h = _IF_RE.match(hdr)
        if h and not h.group("else") and is_guard_cond(h.group("cond")):
            out.append((o, c))
    # the top-level early returns: `if (<the MEASURE or silent path>) { ... return ...; }`
    for o, c, hdr in f.blocks:
        h = _IF_RE.match(hdr)
        if not h or h.group("else"):
            continue
        depth_ok = not any(o2 < o and c2 > c and (o2, c2) != (f.body_start, f.end) for o2, c2, _ in f.blocks)
        if not depth_ok or "return" not in m[o:c]:
            continue
        cond = " ".join(h.group("cond").split())
        if cond in ("mode == sched_reserve_mode::MEASURE", "!log"):
            out.append((c, f.end))
    return out


def in_any(ranges, pos):
    return any(a < pos < b for a, b in ranges)


def statement_at(m: str, pos: int) -> str:
    j = m.index(";", pos)
    return " ".join(m[pos:j].split())


def analyze(raw: str):
    """-> (findings, census) where census names what was examined."""
    m = mask(raw)
    funcs = parse_functions(m)
    byname = {}
    for f in funcs:
        parse_blocks(m, f)
        byname.setdefault(f.name, []).append(f)
    guards = {id(f): guard_blocks(m, f) for f in funcs}

    short = {}
    for f in funcs:
        s = f.name.split("::")[-1]
        if s in ("llama_context", "~llama_context") or s.startswith("llama_context#"):
            continue
        short.setdefault(s, []).append(f)

    # references: name -> [(caller, pos)]
    refs = {s: [] for s in short}
    for f in funcs:
        body = m[f.body_start:f.end + 1]
        for s in short:
            for mt in re.finditer(r"(?<![\w:])(?:this\s*->\s*)?" + re.escape(s) + r"\b|::" + re.escape(s) + r"\b", body):
                pos = f.body_start + mt.start()
                if any(g is f for g in short[s]):
                    # a recursive or self-defining mention is not a call site
                    continue
                refs[s].append((f, pos))

    roots = [f for r in ROOTS for f in byname.get(r, [])]
    assert len(roots) == len(ROOTS), "the roots must all be defined: %s" % [f.name for f in roots]

    reach = {id(f) for f in roots}
    work = list(roots)
    while work:
        f = work.pop()
        for s, lst in refs.items():
            for caller, _ in lst:
                if caller is f:
                    for g in short[s]:
                        if id(g) not in reach:
                            reach.add(id(g))
                            work.append(g)

    # a function is fully guarded when it has a reached call site and every one is guarded
    fully = {}

    def is_fully_guarded(f, seen=()):
        if id(f) in fully:
            return fully[id(f)]
        if id(f) in {id(r) for r in roots} or id(f) in seen:
            return False
        s = f.name.split("::")[-1]
        sites = [(c, p) for c, p in refs.get(s, []) if id(c) in reach]
        ok = bool(sites) and all(
            in_any(guards[id(c)], p) or is_fully_guarded(c, seen + (id(f),)) for c, p in sites
        )
        fully[id(f)] = ok
        return ok

    findings = []
    census = []
    for f in funcs:
        if id(f) not in reach:
            continue
        body = m[f.body_start:f.end + 1]
        for mt in _LOG_RE.finditer(body):
            pos = f.body_start + mt.start()
            stmt = statement_at(raw, pos) if False else statement_at(m, pos)
            raw_stmt = " ".join(raw[pos:raw.index(";", pos)].split())
            census.append((f.name, raw_stmt))
            if in_any(guards[id(f)], pos):
                continue
            if is_fully_guarded(f):
                continue
            if any(fn == f.name and frag in raw_stmt for (fn, frag) in ALLOW):
                continue
            findings.append("%s: unguarded log reachable from a measure-only context: %s" % (f.name, raw_stmt[:100]))
    # A log inside a struct or class body is not seen by the walk. None of the helper structs
    # that precede the context's definitions may hold one.
    ctor_start = min(f.start for f in byname["llama_context::llama_context#3"])
    for mt in _LOG_RE.finditer(m[:ctor_start]):
        if not any(f.start <= mt.start() <= f.end for f in funcs):
            findings.append("a log statement in a helper struct at offset %d" % mt.start())
    return findings, census, funcs, byname, guards, m


def structural(raw: str):
    """The structural clauses of the measure-only constructor, destructor and caller."""
    bad = []
    findings, census, funcs, byname, guards, m = analyze(raw)
    ctor = byname["llama_context::llama_context#3"][0]
    body = m[ctor.body_start:ctor.end + 1]
    g = guards[id(ctor)]

    if "create_memory(params_mem, cparams, measure_only)" not in " ".join(body.split()):
        bad.append("the constructor must create its memory with measure_only as the no-alloc argument")

    # every call that allocates, binds, loads a backend or runs the ladder is in a guarded block
    forbidden = [
        "output_reserve(",
        "ggml_backend_dev_init(",
        "ggml_backend_init_by_type(",
        "create_exec(",
        "bind_exec(",
        "sched_reserve(",
        "sched_reserve_nothrow(",
        "sycl_select_auto_ubatch(",
        "sycl_resync_runtime_context_flash_attn(",
        "llama_set_abort_callback(",
        "set_n_threads_fns.emplace_back(",
        "token_ids_full_vocab",
        "ggml_threadpool",
    ]
    for tok in forbidden:
        for mt in re.finditer(re.escape(tok), body):
            if not in_any(g, ctor.body_start + mt.start()):
                bad.append("the constructor reaches %s outside a !measure_only block" % tok)

    # the destructor
    dtor = byname["llama_context::~llama_context"][0]
    dbody = " ".join(m[dtor.body_start:dtor.end + 1].split())
    if "if (!measure_only && !model.hparams.no_alloc && !opt_ctx) {" not in dbody:
        bad.append("the destructor's buffer-size comparison must carry !measure_only in its own if")
    if "if (!measure_only && sycl_exec_context_bound && sycl_exec_context.value != 0) {" not in dbody:
        bad.append("the destructor's SYCL drain block must carry !measure_only in its own if")

    # the one caller of llama_graph_n_input_tensors passes !measure_only
    calls = [" ".join(m[mt.start():m.index(")", mt.start())].split())
             for mt in re.finditer(r"\bllama_graph_n_input_tensors\s*\(", m)]
    calls = [c for c in calls if not c.startswith("llama_graph_n_input_tensors ( ggml_cgraph")
             and "ggml_cgraph * gf" not in c]
    if calls != ["llama_graph_n_input_tensors(gf, !measure_only"]:
        bad.append("llama_graph_n_input_tensors must have one caller, passing !measure_only: %s" % calls)

    # the measure-only context never gets a chunk-cap copy: nothing in the constructor sets it
    if re.search(r"\bplan_caps\s*(?:=|\.reset|\.swap)", body):
        bad.append("the measure-only constructor must not acquire plan_caps")
    return bad


# --- the tree as it is --------------------------------------------------------------------


def test_census_is_not_empty():
    findings, census, funcs, byname, guards, m = analyze(CONTEXT_CPP)
    names = {c[0] for c in census}
    # the roots with logs, and a callee only reached through a call
    for must in ("llama_context::llama_context#3", "llama_context::~llama_context",
                 "llama_context::sched_reserve_impl", "llama_context::graph_reserve",
                 "llama_context::sched_measure_impl"):
        assert must in names, "the census must see the log statements of %s" % must
    assert len(census) > 40, "the census is implausibly small: %d" % len(census)


def test_every_reached_log_is_guarded_or_allowed():
    findings, *_ = analyze(CONTEXT_CPP)
    assert not findings, "\n".join(findings)


def test_allow_list_is_live():
    # an allow entry that matches nothing is dead text
    _, census, *_ = analyze(CONTEXT_CPP)
    for fn, frag in ALLOW:
        assert any(f == fn and frag in s for f, s in census), "allow entry matches no statement: %s %r" % (fn, frag)


def test_structural_clauses():
    bad = structural(CONTEXT_CPP)
    assert not bad, "\n".join(bad)


# --- mutants ------------------------------------------------------------------------------


def _mut(old: str, new: str, count: int = 1, src: str = None) -> str:
    src = CONTEXT_CPP if src is None else src
    assert src.count(old) >= 1, "mutant anchor not found: %r" % old
    return src.replace(old, new, count)


def _fails(src: str) -> bool:
    findings, *_ = analyze(src)
    return bool(findings) or bool(structural(src))


def test_mutants_each_clause_fails():
    muts = {
        "a guard removed from a constructor log": _mut(
            'if (!measure_only) {\n        LLAMA_LOG_INFO("%s: constructing llama_context\\n", __func__);',
            'if (true) {\n        LLAMA_LOG_INFO("%s: constructing llama_context\\n", __func__);'),
        "an unguarded log added to the constructor": _mut(
            "    measure_only = measure != nullptr;\n",
            '    measure_only = measure != nullptr;\n    LLAMA_LOG_INFO("%s: hello\\n", __func__);\n'),
        "an unguarded log added to graph_reserve": _mut(
            "    GGML_ASSERT(n_outputs >= 1);\n",
            '    GGML_ASSERT(n_outputs >= 1);\n    LLAMA_LOG_DEBUG("%s: hi\\n", __func__);\n'),
        "the destructor comparison loses its guard": _mut(
            "if (!measure_only && !model.hparams.no_alloc && !opt_ctx) {", "if (!model.hparams.no_alloc && !opt_ctx) {"),
        "the destructor drain loses its guard": _mut(
            "if (!measure_only && sycl_exec_context_bound && sycl_exec_context.value != 0) {",
            "if (sycl_exec_context_bound && sycl_exec_context.value != 0) {"),
        "the n_input_tensors caller prints": _mut(
            "llama_graph_n_input_tensors(gf, !measure_only)", "llama_graph_n_input_tensors(gf, true)"),
        "the n_input_tensors early return removed": _mut(
            "    if (!log) {\n        return (int) users.size();\n    }\n", ""),
        "the measure context creates real memory": _mut(
            "create_memory(params_mem, cparams, measure_only)", "create_memory(params_mem, cparams, false)"),
        "the ladder runs in a measure-only context": _mut(
            "    if (!hparams.vocab_only && !measure_only) {\n        // llama.cpp-xojq",
            "    if (!hparams.vocab_only) {\n        // llama.cpp-xojq"),
        "the output buffer is reserved in a measure-only context": _mut(
            "    if (!hparams.vocab_only && !measure_only) {\n        // GPU backends",
            "    if (!hparams.vocab_only) {\n        // GPU backends"),
        "the vocabulary ids are built in a measure-only context": _mut(
            "    // Initialize the full vocabulary token ids for backend samplers.\n    if (!measure_only) {",
            "    // Initialize the full vocabulary token ids for backend samplers.\n    {"),
        "a log added to the sched_measure_impl": _mut(
            "    const int64_t t_start_us = ggml_time_us();\n\n    // The scheduler below is created inside the scope",
            '    LLAMA_LOG_INFO("%s: measuring\\n", __func__);\n    const int64_t t_start_us = ggml_time_us();\n\n    // The scheduler below is created inside the scope'),
        "the measure debug line loses its guard": _mut(
            'if (!measure_only) {\n        LLAMA_LOG_DEBUG("%s: measured %u graphs',
            'if (true) {\n        LLAMA_LOG_DEBUG("%s: measured %u graphs'),
        "a log in a helper reached through an unguarded call": _mut(
            "static int llama_graph_n_input_tensors(ggml_cgraph * gf, bool log) {",
            'static void llama_gate_helper() { LLAMA_LOG_INFO("x\\n"); }\nstatic int llama_graph_n_input_tensors(ggml_cgraph * gf, bool log) {\n    llama_gate_helper();'),
        "the constructor reserves through the nothrow form": _mut(
            "    measure_only = measure != nullptr;\n",
            "    measure_only = measure != nullptr;\n    sched_reserve_nothrow();\n"),
        "a backend sampler is set in a measure-only context": _mut(
            "if (!measure_only && params.samplers != nullptr && params.n_samplers > 0) {",
            "if (params.samplers != nullptr && params.n_samplers > 0) {"),
        "the constructor acquires a chunk-cap copy": _mut(
            "    measure_only = measure != nullptr;\n",
            "    measure_only = measure != nullptr;\n    plan_caps.reset();\n"),
    }
    for name, src in muts.items():
        assert _fails(src), "mutant %r slipped through" % name


def test_clean_helper_is_not_flagged():
    # a helper whose every reached call site is guarded may log
    src = _mut(
        "static int llama_graph_n_input_tensors(ggml_cgraph * gf, bool log) {",
        'static void llama_gate_helper() { LLAMA_LOG_INFO("x\\n"); }\nstatic int llama_graph_n_input_tensors(ggml_cgraph * gf, bool log) {\n    if (log) { llama_gate_helper(); }')
    # `if (log)` is not a guard cond: the call site is unguarded and the helper's log is a finding
    assert _fails(src)
    src2 = _mut(
        "    measure_only = measure != nullptr;\n",
        '    measure_only = measure != nullptr;\n    if (!measure_only) { llama_gate_helper2(); }\n')
    src2 = _mut("llama_context::llama_context(\n        const llama_model & model,\n              llama_context_params params) :",
                'static void llama_gate_helper2() { LLAMA_LOG_INFO("x\\n"); }\n\nllama_context::llama_context(\n        const llama_model & model,\n              llama_context_params params) :',
                src=src2)
    assert not _fails(src2), "a helper called only from a guarded block must pass"
