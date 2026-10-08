"""Source contract for llama.cpp-rriv: a CLEAN zone settle does not walk the runtime allocation registry.

host_zone_settle() ran once per host zone per graph and zone_settle() once per device per graph. Each took
g_runtime_alloc_mutex and walked EVERY row of g_runtime_alloc_registry (~75k on a 512-expert MoE) just to count rows
that, on a clean settle, are not there: 37.3% + 20.0% of Qwen3.8 decode CPU. The settles now ask two questions that cost
O(1) / O(log n) and enumerate rows only to explain a refusal:

  - "is any row in host zone Z"  -> runtime_registry_host_zone_live_locked(): a per-zone live-row counter;
  - "does any row's key fall in the VRAM zone's address span" -> runtime_registry_span_live_locked(): the ii25 range
    index's find_first_base_in(), with a registry scan only while an "irregular" row exists (a row whose key is not the
    base the index holds it at, which no caller produces; the scan keeps the old key-in-span answer for one).

What this file enforces, as text assertions on comment-stripped unified-cache.cpp:

(A) host_zone_settle() and zone_settle() never name g_runtime_alloc_registry. Each calls its scan helper only inside
    the block of an `if (<O(1) question>(<its zone arguments>)) {` whose condition is that question alone.
(B) The enumerating helpers (runtime_registry_scan_host_zone_locked / runtime_registry_scan_span_locked) are called only
    from those two functions, and only inside the branch the O(1) question opened.
(C) runtime_registry_host_zone_live_locked() is a counter read with no loop; runtime_registry_span_live_locked() queries
    the index first and reaches its one registry loop only after the irregular-row guard has declined the index answer.
(D) Counter and host_zone rules:
    - writer: the counters are written only inside runtime_registry_count_row_locked(), which only the three registry
      mutation helpers (emplace, erase, assign) call, so a counter cannot drift from the registry the way a hand-kept
      one would;
    - host_zone: a registered row's handle.host_zone, the field the host counters key on, is never rewritten through
      the registry (assignment, ++/--, address-of, a reference binding or a call argument);
    - accepted reads: outside the helper a counter name may only be read (a comparison operand, a returned or copied
      value); a conditional operand counts as a read only in a `return` or a plain `=` initialiser, and only while
      every parenthesis before the operand is closed (a parenthesised or nested conditional is refused even in a
      return);
    - refused: assignment, ++/--, a reference declarator or init-capture whose type spelling contains `&`/`&&`
      textually, address-of, an unsubscripted array, a call argument, and a return from a namespace-scope function
      whose header follows a column-0 `}` or the file start, or from a lambda, whose declared (leading or trailing)
      return type contains `&`.

NOT COVERED, stated so nobody mistakes this for a proof. This is a tripwire on text, not on cost or on drift:
    - it cannot see what a helper's callee does (a counter read that someone makes expensive passes);
    - a counter that is updated by the right helper with the wrong row still passes:
      ggml/src/ggml-sycl/tests/test-runtime-registry-containment.cpp compares the counters with a full scan, and
      test-unified-runtime-alloc (device) exercises the real settles;
    - a clean-settle slow path that is not a loop over the registry (another global scan, a sleep) passes;
    - functions are bounded at a closing brace in column 0, which is how this file is formatted; `#if 0` blocks and raw
      string literals are not understood by the comment stripper (ordinary string and char literals are blanked before
      braces are matched);
    - an alias or type trait that hides the `&` (`using R = size_t &; R c = COUNTER; c = 0;`,
      `std::add_lvalue_reference_t<size_t>`) passes;
    - a counter returned by reference from a member function (`struct S { size_t & get() { return COUNTER; } };`, its
      static variant) or from the FIRST function after a fresh scope opener
      (`namespace detail { size_t & get() {...} }`) passes: only functions whose header follows a column-0 `}` or the file start have their leading return type read;
    - fail-closed: a `->` member access followed by `&` in an enclosing `if`/`while` header (`if ((p)->a & 1) { return
      COUNTER; }`) is read as a trailing return type, so such a return from a by-value function is reported;
    - a host_zone passed through a parenthesised callee (`(consume)(it->second.handle.host_zone)`) passes;
    - a counter or host_zone reached through a macro, a template parameter or a pointer arithmetic expression the text
      rules do not parse passes; the rules are a name-based tripwire.

Host-only, pure text assertions. llama_test_pytest hands this file to pytest.main(), so the checks live inside test_*()
functions. Each check has a mutation witness so it is known to fail on the regression it guards.
"""

import functools
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SYCL_DIR = ROOT / "ggml/src/ggml-sycl"

_LEXEME_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"'  # string literal (kept)
    r"|'(?:\\.|[^'\\\n])*'"  # char literal (kept)
    r"|//[^\n]*"  # line comment (dropped)
    r"|/\*.*?\*/",  # block comment (dropped)
    flags=re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


CODE = strip_comments((SYCL_DIR / "unified-cache.cpp").read_text())

HOST_SETTLE = r"void\s+unified_cache::host_zone_settle\(\s*host_zone_id\s+zone\s*\)\s*\{"
VRAM_SETTLE = r"void\s+unified_cache::zone_settle\(\s*vram_zone_id\s+zone\s*\)\s*\{"
HOST_LIVE = r"static\s+bool\s+runtime_registry_host_zone_live_locked\(\s*host_zone_id\s+zone\s*\)\s*noexcept\s*\{"
SPAN_LIVE = r"static\s+bool\s+runtime_registry_span_live_locked\(\s*uintptr_t\s+lo\s*,\s*uintptr_t\s+hi\s*\)\s*noexcept\s*\{"
COUNT_ROW = r"static\s+void\s+runtime_registry_count_row_locked\("
EMPLACE = r"runtime_registry_emplace_locked\(\s*void\s*\*\s*ptr\s*,\s*runtime_alloc_record\s+rec\s*\)\s*\{"
ERASE_IT = r"runtime_registry_erase_locked\(\s*runtime_registry_iterator\s+it\s*\)\s*noexcept\s*\{"
ASSIGN = r"runtime_registry_assign_locked\(\s*void\s*\*\s*ptr\s*,\s*const\s+runtime_alloc_record\s*&\s*rec\s*\)\s*\{"

HOST_ARGS = r"zone"
SPAN_ARGS = r"zone_lo\s*,\s*zone_hi"
SCAN_HOST = "runtime_registry_scan_host_zone_locked"
SCAN_SPAN = "runtime_registry_scan_span_locked"
COUNTERS = ("g_runtime_host_zone_rows", "g_runtime_span_irregular_rows")
LOOP_RE = re.compile(r"\b(?:for|while)\s*\(")


def function_span(code: str, signature: str):
    m = re.search(signature, code)
    assert m is not None, f"{signature!r} is not defined"
    start = m.start()
    end = code.find("\n}\n", start)
    assert end != -1, f"could not bound {signature!r}"
    return start, end


def function_body(code: str, signature: str) -> str:
    start, end = function_span(code, signature)
    return code[start:end]


_LITERAL_RE = re.compile(r'"(?:\\.|[^"\\\n])*"' + r"|'(?:\\.|[^'\\\n])*'")


def blank_literals(text: str) -> str:
    """Same-length text with string and char literal contents replaced by spaces, so braces inside them do not count."""
    return _LITERAL_RE.sub(lambda m: " " * len(m.group(0)), text)


def matching_brace(text: str, open_idx: int):
    """Index of the `}` that closes the `{` at open_idx, or None (braces inside string and char literals are ignored)."""
    text = blank_literals(text)
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def settle_violations(body: str, o1_question: str, scan: str, args: str):
    """What is wrong with a settle body: the O(1) question must be the WHOLE condition of an `if`, and the scan may be
    called only inside that `if`'s block."""
    out = []
    if "g_runtime_alloc_registry" in body:
        out.append("names g_runtime_alloc_registry")
    if re.search(r"\b(?:for|while)\s*\([^)]*(?:registry|alloc_index)", body):
        out.append("loops over the registry")
    gate = re.search(r"\bif\s*\(\s*" + re.escape(o1_question) + r"\s*\(\s*" + args + r"\s*\)\s*\)\s*\{", body)
    block = (gate.end() - 1, matching_brace(body, gate.end() - 1)) if gate else None
    if gate is None:
        out.append(f"has no `if ({o1_question}(<its zone arguments>)) {{` (the question must be the whole condition)")
    elif block[1] is None:
        out.append("the question's block is unbalanced")
    calls = [m.start() for m in re.finditer(re.escape(scan) + r"\s*\(", body)]
    if not calls:
        out.append(f"never enumerates with {scan}() (the refusal needs its diagnostics)")
    for c in calls:
        if block is None or block[1] is None or not (block[0] < c < block[1]):
            out.append(f"{scan}() is called outside the block the O(1) question opened")
    return out


def test_host_zone_settle_does_not_scan_the_registry_when_clean():
    body = function_body(CODE, HOST_SETTLE)
    assert settle_violations(body, "runtime_registry_host_zone_live_locked", SCAN_HOST, HOST_ARGS) == []


def test_zone_settle_does_not_scan_the_registry_when_clean():
    body = function_body(CODE, VRAM_SETTLE)
    assert settle_violations(body, "runtime_registry_span_live_locked", SCAN_SPAN, SPAN_ARGS) == []


HOST_OK = (
    "{ std::lock_guard<std::mutex> lock(g_runtime_alloc_mutex);\n"
    "  if (runtime_registry_host_zone_live_locked(zone)) {\n"
    "      size_t n = runtime_registry_scan_host_zone_locked(zone, a, b, audit, &c);\n"
    "      if (n > 0) { return; }\n"
    "  } }\n"
)


@pytest.mark.parametrize(
    "mutant",
    [
        # the old shape: the scan is the decision
        "{ for (auto it = g_runtime_alloc_registry.begin(); it != g_runtime_alloc_registry.end(); ++it) { n++; } }",
        # the registry named in the settle again
        HOST_OK.replace("  }", "  } auto x = g_runtime_alloc_registry.size();", 1),
        # the O(1) question dropped
        HOST_OK.replace("runtime_registry_host_zone_live_locked(zone)", "true"),
        # the scan moved out of the branch the question opened
        HOST_OK.replace("  }", "  } n = runtime_registry_scan_host_zone_locked(zone, a, b, audit, &c);", 1),
        # the scan hoisted ahead of the question
        "{ n = runtime_registry_scan_host_zone_locked(zone, a, b, audit, &c);\n"
        "  if (runtime_registry_host_zone_live_locked(zone)) { return; } }",
        # no enumeration at all: the refusal would lose its diagnostics
        "{ if (runtime_registry_host_zone_live_locked(zone)) { return; } }",
        # a loop over the index in the settle
        HOST_OK.replace("  }", "  } for (auto it = g_runtime_alloc_index; ;) {}", 1),
    ],
)
def test_settle_gate_has_a_witness(mutant):
    assert settle_violations(mutant, "runtime_registry_host_zone_live_locked", SCAN_HOST, HOST_ARGS), mutant


def test_real_settle_bodies_defeat_the_question_bypass_mutants():
    host = function_body(CODE, HOST_SETTLE)
    q = "if (runtime_registry_host_zone_live_locked(zone)) {"
    assert q in host
    # asks the question, then scans on an always-true condition
    m1 = host.replace(q, "(void) runtime_registry_host_zone_live_locked(zone);\n        if (epoch_tracked || !epoch_tracked) {")
    assert settle_violations(m1, "runtime_registry_host_zone_live_locked", SCAN_HOST, HOST_ARGS)
    m2 = host.replace(q, "if (runtime_registry_host_zone_live_locked(zone) || true) {")
    assert settle_violations(m2, "runtime_registry_host_zone_live_locked", SCAN_HOST, HOST_ARGS)
    vram = function_body(CODE, VRAM_SETTLE)
    q = "if (runtime_registry_span_live_locked(zone_lo, zone_hi)) {"
    assert q in vram
    m3 = vram.replace(q, "if (runtime_registry_span_live_locked(zone_lo, zone_hi) || true) {")
    assert settle_violations(m3, "runtime_registry_span_live_locked", SCAN_SPAN, SPAN_ARGS)
    m4 = vram.replace(q, "(void) runtime_registry_span_live_locked(zone_lo, zone_hi);\n        if (true) {")
    assert settle_violations(m4, "runtime_registry_span_live_locked", SCAN_SPAN, SPAN_ARGS)


def test_real_settle_bodies_defeat_wrong_argument_and_literal_brace_mutants():
    host = function_body(CODE, HOST_SETTLE)
    q = "if (runtime_registry_host_zone_live_locked(zone)) {"
    m1 = host.replace(q, "if (runtime_registry_host_zone_live_locked(host_zone_id::KV)) {")
    assert settle_violations(m1, "runtime_registry_host_zone_live_locked", SCAN_HOST, HOST_ARGS)
    vram = function_body(CODE, VRAM_SETTLE)
    m2 = vram.replace(
        "if (runtime_registry_span_live_locked(zone_lo, zone_hi)) {",
        "if (runtime_registry_span_live_locked(0, UINTPTR_MAX)) {",
    )
    assert settle_violations(m2, "runtime_registry_span_live_locked", SCAN_SPAN, SPAN_ARGS)
    # a `{` inside a string literal must not keep the question's block open past its real end
    m3 = host.replace(
        "if (runtime_registry_host_zone_live_locked(zone)) {",
        'if (runtime_registry_host_zone_live_locked(zone)) { GGML_LOG_WARN("{"); }\n    if (true) {',
    )
    assert m3 != host and settle_violations(m3, "runtime_registry_host_zone_live_locked", SCAN_HOST, HOST_ARGS)


def test_settle_gate_accepts_the_intended_shape():
    assert settle_violations(HOST_OK, "runtime_registry_host_zone_live_locked", SCAN_HOST, HOST_ARGS) == []


SCAN_HOST_SIG = r"static\s+size_t\s+" + SCAN_HOST + r"\("
SCAN_SPAN_SIG = r"static\s+size_t\s+" + SCAN_SPAN + r"\("


def uses_outside(code: str, name: str, allowed_signatures):
    """Lines where `name(` appears outside the bodies of the allowed functions (its own definition included)."""
    spans = [function_span(code, sig) for sig in allowed_signatures]
    bad = []
    for m in re.finditer(r"\b" + re.escape(name) + r"\s*\(", code):
        if not any(s <= m.start() < e for s, e in spans):
            bad.append(code.count("\n", 0, m.start()) + 1)
    return bad


def test_the_scan_helpers_are_called_only_by_the_two_settles():
    assert uses_outside(CODE, SCAN_HOST, [SCAN_HOST_SIG, HOST_SETTLE]) == []
    assert uses_outside(CODE, SCAN_SPAN, [SCAN_SPAN_SIG, VRAM_SETTLE]) == []
    assert SCAN_HOST in function_body(CODE, HOST_SETTLE) and SCAN_SPAN in function_body(CODE, VRAM_SETTLE)


def test_uses_outside_has_a_witness():
    planted = CODE + "\nvoid f() {\n    " + SCAN_HOST + "(zone, a, b, c, d);\n}\n"
    assert uses_outside(planted, SCAN_HOST, [SCAN_HOST_SIG, HOST_SETTLE]) != []


def test_the_scan_helpers_exist_and_enumerate():
    for sig in (SCAN_HOST_SIG, SCAN_SPAN_SIG):
        body = function_body(CODE, sig)
        assert "g_runtime_alloc_registry" in body and LOOP_RE.search(body), sig


def host_live_violations(body: str):
    """What is wrong with the host-zone question: it must be a counter read, no loop and no registry access."""
    out = []
    if LOOP_RE.search(body):
        out.append("loops")
    if "g_runtime_alloc_registry" in body:
        out.append("names the registry")
    if "g_runtime_host_zone_rows" not in body:
        out.append("does not read the per-zone counter")
    return out


def test_host_zone_live_is_a_counter_read():
    assert host_live_violations(function_body(CODE, HOST_LIVE)) == []


@pytest.mark.parametrize(
    "body",
    [
        "{ for (const auto & kv : g_runtime_alloc_registry) { if (kv.second.handle.host_zone == zone) return true; } return false; }",
        "{ return g_runtime_alloc_registry.size() > 0; }",
        "{ return true; }",
        "{ size_t n = 0; while (n < 3) { n++; } return g_runtime_host_zone_rows[0] > n; }",
    ],
)
def test_host_zone_live_gate_has_a_witness(body):
    assert host_live_violations(body), body


def span_live_violations(body: str):
    """What is wrong with the span question: the index answer must be returned inside the irregular-row guard."""
    out = []
    if "g_runtime_alloc_index.find_first_base_in(" not in body:
        out.append("does not query the index")
    guard = re.search(r"if\s*\(\s*g_runtime_span_irregular_rows\s*==\s*0\s*\)\s*\{", body)
    if guard is None:
        out.append("has no irregular-row guard")
        return out
    # The index answer sits inside the guard and returns; any loop must come after the guard's block has closed.
    close = matching_brace(body, guard.end() - 1)
    if close is None:
        out.append("guard block is unbalanced")
        return out
    block = body[guard.start() : close]
    if "find_first_base_in(" not in block or "return" not in block:
        out.append("the index answer is not returned inside the guard")
    if LOOP_RE.search(body[: close + 1]):
        out.append("loops before the guard has declined the index answer")
    return out


def test_span_live_queries_the_index_before_its_one_fallback_scan():
    body = function_body(CODE, SPAN_LIVE)
    assert span_live_violations(body) == []
    assert len(LOOP_RE.findall(body)) == 1  # the documented fallback, and only that


SPAN_OK = (
    "{ if (g_runtime_span_irregular_rows == 0) {\n"
    "      return g_runtime_alloc_index.find_first_base_in(lo, hi, nullptr);\n"
    "  }\n"
    "  for (const auto & kv : g_runtime_alloc_registry) { if (in(kv.first)) return true; }\n"
    "  return false; }\n"
)


def test_span_live_gate_accepts_the_intended_shape():
    assert span_live_violations(SPAN_OK) == []


@pytest.mark.parametrize(
    "mutant",
    [
        # the old scan, unguarded
        "{ for (const auto & kv : g_runtime_alloc_registry) { if (in(kv.first)) return true; } return false; }",
        # the index never consulted
        SPAN_OK.replace("g_runtime_alloc_index.find_first_base_in(lo, hi, nullptr)", "false"),
        # the guard dropped: the scan runs before any index answer
        SPAN_OK.replace("if (g_runtime_span_irregular_rows == 0)", "if (true)"),
        # a loop ahead of the guard
        "{ for (int i = 0; i < 3; i++) {}\n" + SPAN_OK[2:],
        # a `}` inside a string literal must not close the guard early and hide the loop that follows
        "{ if (g_runtime_span_irregular_rows == 0) {\n"
        "      if (g_runtime_alloc_index.find_first_base_in(lo, hi, nullptr)) return true;\n"
        '      GGML_LOG_WARN("}");\n'
        "      for (const auto & kv : g_runtime_alloc_registry) { if (in(kv.first)) return true; }\n"
        "      return false;\n"
        "  }\n"
        "  return false; }\n",
        # the guarded block no longer returns the index answer
        SPAN_OK.replace("return g_runtime_alloc_index.find_first_base_in(lo, hi, nullptr);", "(void) g_runtime_alloc_index.find_first_base_in(lo, hi, nullptr);"),
    ],
)
def test_span_live_gate_has_a_witness(mutant):
    assert span_live_violations(mutant), mutant


COUNTER_ARRAY = "g_runtime_host_zone_rows"
_COMPARE_AFTER = re.compile(r"\s*(?:==|!=|<=|>=|<(?!<)|>(?!>))")
_READ_BEFORE = re.compile(r"(?:\breturn|[=!<>]=|=|<|>|&&|\|\|)\s*$")
_TERNARY_BEFORE = re.compile(r"[?:]\s*$")
_PLAIN_ASSIGN = re.compile(r"(?<![=!<>+\-*/%|&^])=(?!=)")


def ternary_operand_is_read(prefix: str) -> bool:
    """A conditional operand is a read only when its statement is `return <expr>` or `<decl or lvalue> = <expr>` with every
    parenthesis in <expr> closed before the operand: `(c ? X : y) = 0` and `f(c ? X : y)` are not."""
    stripped = prefix.lstrip()
    if stripped.startswith("return"):
        rest = stripped[len("return") :]
    else:
        # the initialising `=` is the first one outside every parenthesis: `f(a = 1, c ? X : y)` has none
        m = next(
            (m for m in _PLAIN_ASSIGN.finditer(stripped) if stripped.count("(", 0, m.start()) == stripped.count(")", 0, m.start())),
            None,
        )
        if m is None:
            return False
        rest = stripped[m.end() :]
    return rest.count("(") == rest.count(")")


# A reference declarator or init-capture: `T & x =`, `T&& x =`, `[&x =`, whatever the type spelling is.
_REF_DECLARATOR = re.compile(r"(?<![&\w])&{1,2}\s*\w+\s*=(?!=)|\w\s*&{1,2}\s*\w+\s*=(?!=)")


def statement_prefix(code: str, pos: int) -> str:
    """Text from the previous `;`, `{` or `}` up to pos."""
    return code[max(code.rfind(c, 0, pos) for c in ";{}") + 1 : pos]


_TRAILING_REF_RETURN = re.compile(r"\)\s*(?:(?:const|noexcept|mutable)\s*)*->\s*[^;{}()]*&")


def returns_reference(code: str, pos: int) -> bool:
    """Is pos inside a function or lambda whose declared return type contains `&`? Functions start after a column-0 `}`
    line; the leading return type is the text before the first `(`, a trailing one follows `) ->`."""
    prev = None
    for prev in re.finditer(r"^\}[^\n]*\n", code[:pos], flags=re.M):
        pass
    region = prev.end() if prev else 0
    text = blank_literals(code)
    stack = []
    for i in range(region, pos):
        if text[i] == "{":
            stack.append(i)
        elif text[i] == "}" and stack:
            stack.pop()
    if not stack:
        return False
    for depth, brace in enumerate(stack):
        header = text[region:brace] if depth == 0 else text[:brace]
        header = header[max(header.rfind(";"), header.rfind("{"), header.rfind("}"), -1) + 1 :]
        if _TRAILING_REF_RETURN.search(header):
            return True
        if depth == 0 and "&" in header.split("(")[0]:
            return True
    return False


def is_declaration(code: str, pos: int) -> bool:
    """Is pos the name in `static size_t <name>` at the start of its line (the counter's own declaration)?"""
    line_start = code.rfind("\n", 0, pos) + 1
    return re.fullmatch(r"static\s+size_t\s+", code[line_start:pos]) is not None


def counter_writes_outside_the_count_helper(code: str):
    """Every use of a counter name outside the count helper that is not a plain read: an assignment, ++/--, a reference
    binding, address-of, an array used without a subscript (decay: memset/std::fill/pointer), or a call argument
    (std::swap and friends). A plain read is a comparison operand, a returned or copied value."""
    start, end = function_span(code, COUNT_ROW)
    bad = []
    write = re.compile(
        r"\b(?:" + "|".join(COUNTERS) + r")\b(?:\s*\[[^\]]*\])?\s*(?:(?:[-+*/%|&^]|<<|>>)?=(?!=)|\+\+|--)"
        r"|(?:\+\+|--)\s*\b(?:" + "|".join(COUNTERS) + r")\b"
        # a counter bound to a non-const reference could be written through it
        r"|(?<!const )(?:auto|size_t)\s*&&?\s*\w+\s*=\s*(?:" + "|".join(COUNTERS) + r")\b"
    )

    def note(pos, text):
        bad.append((code.count("\n", 0, pos) + 1, text[:60].replace("\n", " ")))

    for m in write.finditer(code):
        if start <= m.start() < end or is_declaration(code, m.start()):
            continue
        note(m.start(), m.group(0))
    for m in re.finditer(r"\b(?:" + "|".join(COUNTERS) + r")\b", code):
        if start <= m.start() < end or is_declaration(code, m.start()):
            continue
        after = code[m.end() :]
        end_use = m.end()
        if m.group(0) == COUNTER_ARRAY:
            sub = re.match(r"\s*\[[^\]]*\]", after)
            if sub is None:
                note(m.start(), "array used without a subscript: " + code[m.start() : m.start() + 40])
                continue
            end_use += sub.end()
            after = code[end_use:]
        before = code[: m.start()]
        prefix = statement_prefix(code, m.start())
        if _REF_DECLARATOR.search(prefix):
            note(m.start(), "bound through a reference: " + code[max(0, m.start() - 30) : m.start() + 30])
            continue
        if re.match(r"\s*return\b", prefix) and returns_reference(code, m.start()):
            note(m.start(), "returned from a function returning a reference")
            continue
        if re.search(r"(?<!&)&\s*$", before):
            note(m.start(), "address-of: " + code[m.start() - 4 : m.start() + 40])
            continue
        if _TERNARY_BEFORE.search(before):
            if ternary_operand_is_read(prefix):
                continue
        elif _COMPARE_AFTER.match(after) or _READ_BEFORE.search(before):
            continue
        note(m.start(), "not a plain read: " + code[max(0, m.start() - 20) : m.start() + 40])
    return sorted(set(bad))


def test_counters_are_written_only_by_the_count_helper():
    assert counter_writes_outside_the_count_helper(CODE) == []


@pytest.mark.parametrize(
    "line",
    [
        "g_runtime_host_zone_rows[0]++;",
        "++g_runtime_host_zone_rows[1];",
        "g_runtime_host_zone_rows[2] -= 1;",
        "g_runtime_host_zone_rows[3] = 0;",
        "g_runtime_span_irregular_rows++;",
        "g_runtime_span_irregular_rows = 0;",
        "auto & c = g_runtime_host_zone_rows[0];",
        "size_t & c = g_runtime_span_irregular_rows;",
        "memset(g_runtime_host_zone_rows, 0, sizeof(g_runtime_host_zone_rows));",
        "std::fill(std::begin(g_runtime_host_zone_rows), std::end(g_runtime_host_zone_rows), 0);",
        "size_t * p = &g_runtime_span_irregular_rows; *p = 0;",
        "size_t * rows = g_runtime_host_zone_rows;",
        "std::swap(g_runtime_span_irregular_rows, x);",
        "std::swap(x, g_runtime_host_zone_rows[1]);",
        "size_t * q = &g_runtime_host_zone_rows[2];",
        "unsigned long & c = g_runtime_span_irregular_rows; c = 0;",
        "uint64_t & c = g_runtime_host_zone_rows[0]; c = 0;",
        "size_t & c = flag ? g_runtime_span_irregular_rows : other; c = 0;",
        "auto l = [&c = g_runtime_span_irregular_rows] { c = 0; };",
        "auto && c = g_runtime_host_zone_rows[1]; c = 0;",
        "decltype(auto) c = (g_runtime_span_irregular_rows);",
        "(flag ? g_runtime_span_irregular_rows : other) = 0;",
        "std::swap(flag ? g_runtime_span_irregular_rows : other, y);",
        "bump(flag ? g_runtime_host_zone_rows[0] : other);",
        "(flag ? other : g_runtime_span_irregular_rows) = 0;",
        "size_t n = f(flag ? g_runtime_span_irregular_rows : other);",
        "bump(a = 1, flag ? g_runtime_span_irregular_rows : other);",
        "std::swap(t = u, flag ? g_runtime_host_zone_rows[0] : other);",
        "x = bump(a = 1, flag ? g_runtime_span_irregular_rows : other);",
    ],
)
def test_counter_write_gate_has_a_witness(line):
    planted = CODE + "\nvoid f() {\n    " + line + "\n}\n"
    assert counter_writes_outside_the_count_helper(planted), line


@pytest.mark.parametrize(
    "func",
    [
        "static size_t & irregular_ref() noexcept {\n    return g_runtime_span_irregular_rows;\n}\n",
        "static auto & rows_ref(size_t z) {\n    return g_runtime_host_zone_rows[z];\n}\n",
        "static const size_t & rows_cref() {\n    return g_runtime_span_irregular_rows;\n}\n",
        "static auto irregular_ref() noexcept -> size_t & {\n    return g_runtime_span_irregular_rows;\n}\n",
        "static void g() {\n    auto l = []() -> size_t & { return g_runtime_span_irregular_rows; };\n}\n",
        "static void g() {\n    auto l = [](size_t z) mutable -> auto & { return g_runtime_host_zone_rows[z]; };\n}\n",
    ],
)
def test_counter_returned_by_reference_has_a_witness(func):
    assert counter_writes_outside_the_count_helper(CODE + "\n" + func)


def test_counter_returned_by_value_is_a_read():
    func = "static size_t irregular_value() noexcept {\n    return g_runtime_span_irregular_rows;\n}\n"
    func += "static void g() {\n    auto l = []() -> size_t { return g_runtime_span_irregular_rows; };\n}\n"
    assert counter_writes_outside_the_count_helper(CODE + "\n" + func) == []


@pytest.mark.parametrize(
    "line",
    [
        "if (g_runtime_span_irregular_rows == 0) { return true; }",
        "return g_runtime_host_zone_rows[0] != 0;",
        "const size_t n = flag ? g_runtime_host_zone_rows[0] : 0;",
        "size_t n; n = flag ? other : g_runtime_span_irregular_rows;",
        "return (a < b) ? g_runtime_host_zone_rows[1] : 0;",
        "const size_t n = g_runtime_host_zone_rows[1];",
        "if (g_runtime_span_irregular_rows >= 1) {}",
    ],
)
def test_counter_write_gate_allows_reads(line):
    planted = CODE + "\nvoid f() {\n    " + line + "\n}\n"
    assert counter_writes_outside_the_count_helper(planted) == [], line


def count_helper_violations(code: str):
    """Calls of the count helper outside the three registry helpers, and registry helpers that do not call it."""
    out = []
    if uses_outside(code, "runtime_registry_count_row_locked", [COUNT_ROW, EMPLACE, ERASE_IT, ASSIGN]):
        out.append("called outside the three registry helpers")
    for name, sig in (("emplace", EMPLACE), ("erase", ERASE_IT), ("assign", ASSIGN)):
        if "runtime_registry_count_row_locked(" not in function_body(code, sig):
            out.append(f"{name} does not update the counters")
    return out


def test_the_count_helper_is_called_only_by_the_registry_helpers_and_each_calls_it():
    assert count_helper_violations(CODE) == []


def test_the_count_helper_gate_has_witnesses():
    planted = CODE + "\nvoid f() {\n    runtime_registry_count_row_locked(p, h, true);\n}\n"
    assert "called outside the three registry helpers" in count_helper_violations(planted)
    for name, sig in (("emplace", EMPLACE), ("erase", ERASE_IT), ("assign", ASSIGN)):
        start, end = function_span(CODE, sig)
        body = CODE[start:end].replace("runtime_registry_count_row_locked(", "forgot(")
        forgot = CODE[:start] + body + CODE[end:]
        assert count_helper_violations(forgot) == [f"{name} does not update the counters"], name


# A registered row's host zone is counted when the row is registered and uncounted when it leaves, so it must not be
# rewritten in between. test-sycl-runtime-registry-index-source.py forbids writing a row's handle, ptr and size through the
# registry; this adds the one field the counters key on.
_ASSIGN = r"(?:(?:[-+*/%|&^]|<<|>>)?=(?!=)|\+\+|--)"
HOST_ZONE_WRITE_RE = re.compile(r"(?:->|\.)\s*second\s*\)*\s*\.\s*handle\s*\)*\s*\.\s*host_zone\s*" + _ASSIGN)
HOST_ZONE_USE_RE = re.compile(r"(?:->|\.)\s*second\s*\)*\s*\.\s*handle\s*\)*\s*\.\s*host_zone\b")


def host_zone_row_writes(code: str):
    """Rewrites of a registered row's host_zone: assignment, ++/--, address-of, a reference binding, or a call argument."""
    bad = [m.start() for m in HOST_ZONE_WRITE_RE.finditer(code)]
    for m in HOST_ZONE_USE_RE.finditer(code):
        prefix = statement_prefix(code, m.start())
        after = code[m.end() :]
        # the object the row is reached through sits left of `->second`; address-of is the token before it
        if re.search(r"(?<!&)&\s*[\w\(\)\.\->\s]*$", prefix) or _REF_DECLARATOR.search(prefix):
            bad.append(m.start())
        elif re.match(r"\s*(?:==|!=|<=|>=|<(?!<)|>(?!>)|;|\?|&&|\|\|)", after):
            continue
        elif (m_ctx := re.search(r"(?:\breturn|=|\bif\s*\(|\bwhile\s*\()\s*([\w\(\)\.\->\s]*)$", prefix)) and not re.search(
            r"\w\s*\(", m_ctx.group(1)
        ):
            continue
        else:
            bad.append(m.start())
    return sorted(set(bad))


def test_a_registered_rows_host_zone_is_never_rewritten():
    assert host_zone_row_writes(CODE) == []


@pytest.mark.parametrize(
    "line",
    [
        "it->second.handle.host_zone = host_zone_id::KV;",
        "registry.find(p)->second.handle.host_zone=host_zone_id::KV;",
        "it->second.handle.host_zone |= x;",
        "(it->second).handle.host_zone = host_zone_id::KV;",
        "it-> second.handle.host_zone = host_zone_id::KV;",
        "(it->second.handle).host_zone = host_zone_id::KV;",
        "((it)->second).handle.host_zone++;",
        "std::swap(it->second.handle.host_zone, z);",
        "std::swap(z, it->second.handle.host_zone);",
        "host_zone_id * zp = &it->second.handle.host_zone;",
        "host_zone_id & zr = it->second.handle.host_zone;",
        "consume(it->second.handle.host_zone);",
        "bool b = consume(it->second.handle.host_zone);",
        "if (consume(it->second.handle.host_zone)) {}",
        "while (bump(it->second.handle.host_zone)) {}",
    ],
)
def test_host_zone_write_gate_has_a_witness(line):
    assert host_zone_row_writes(CODE + "\nvoid f() {\n    " + line + "\n}\n"), line


@pytest.mark.parametrize(
    "line",
    [
        "if (it->second.handle.host_zone == zone) {}",
        "const host_zone_id z = it->second.handle.host_zone;",
        "return it->second.handle.host_zone;",
        "while (it->second.handle.host_zone != zone) {}",
    ],
)
def test_host_zone_write_gate_allows_reads(line):
    assert host_zone_row_writes(CODE + "\nvoid f() {\n    " + line + "\n}\n") == [], line


STATIC_NAMES = COUNTERS + (SCAN_HOST, SCAN_SPAN, "runtime_registry_host_zone_live_locked", "runtime_registry_span_live_locked")


def static_offenders(stripped_texts):
    """Names of the (comment-stripped) files that mention a file-static name."""
    return [name for name, text in sorted(stripped_texts.items()) if any(n in text for n in STATIC_NAMES)]


@functools.lru_cache(maxsize=1)
def sycl_sources_other_than_unified_cache():
    """Comment-stripped text of every SYCL source but unified-cache.cpp, read once per run."""
    texts = {}
    for path in sorted(SYCL_DIR.rglob("*")):
        if path.suffix in (".cpp", ".hpp", ".h", ".cu", ".cuh") and path.name != "unified-cache.cpp":
            texts[path.name] = strip_comments(path.read_text(errors="replace"))
    return texts


def test_counters_and_scan_helpers_stay_file_static():
    texts = sycl_sources_other_than_unified_cache()
    assert texts, "found no SYCL sources to check"
    assert static_offenders(texts) == []


@pytest.mark.parametrize("name", STATIC_NAMES)
def test_file_static_gate_has_a_witness(name):
    texts = dict(sycl_sources_other_than_unified_cache())
    texts["planted.cpp"] = strip_comments("void f() { use(" + name + "); }\n")
    assert static_offenders(texts) == ["planted.cpp"]
    texts["planted.cpp"] = strip_comments("// " + name + " is only named in a comment\n")
    assert static_offenders(texts) == []


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
