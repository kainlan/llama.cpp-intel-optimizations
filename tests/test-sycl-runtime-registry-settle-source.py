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

(A) host_zone_settle() and zone_settle() never name g_runtime_alloc_registry, contain no scan-helper call that is not
    guarded by the O(1) question, and consult runtime_registry_host_zone_live_locked() / runtime_registry_span_live_locked().
(B) The enumerating helpers (runtime_registry_scan_host_zone_locked / runtime_registry_scan_span_locked) are called only
    from those two functions, and only inside the branch the O(1) question opened.
(C) runtime_registry_host_zone_live_locked() is a counter read with no loop; runtime_registry_span_live_locked() queries
    the index first and reaches its one registry loop only after the irregular-row guard has declined the index answer.
(D) The counters are written only inside runtime_registry_count_row_locked(), which only the three registry mutation
    helpers (emplace, erase, assign) call; so a counter cannot drift from the registry the way a hand-kept one would.
    A registered row's handle.host_zone, the field the host counters key on, is never rewritten through the registry.

NOT COVERED, stated so nobody mistakes this for a proof. This is a tripwire on text, not on cost or on drift:
    - it cannot see what a helper's callee does (a counter read that someone makes expensive passes);
    - a counter that is updated by the right helper with the wrong row still passes: tests/test-runtime-registry-containment.cpp
      compares the counters with a full scan, and test-unified-runtime-alloc (device) exercises the real settles;
    - a clean-settle slow path that is not a loop over the registry (another global scan, a sleep) passes;
    - functions are bounded at a closing brace in column 0, which is how this file is formatted; `#if 0` blocks and raw
      string literals are not understood by the comment stripper.

Host-only, pure text assertions. llama_test_pytest hands this file to pytest.main(), so the checks live inside test_*()
functions. Each check has a mutation witness so it is known to fail on the regression it guards.
"""

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


def settle_violations(body: str, o1_question: str, scan: str):
    """What is wrong with a settle body: it must ask the O(1) question and may call the scan only inside its branch."""
    out = []
    if "g_runtime_alloc_registry" in body:
        out.append("names g_runtime_alloc_registry")
    if LOOP_RE.search(body) and re.search(r"\b(?:for|while)\s*\([^)]*(?:registry|alloc_index)", body):
        out.append("loops over the registry")
    q = body.find(o1_question + "(")
    if q == -1:
        out.append(f"never asks {o1_question}()")
    calls = [m.start() for m in re.finditer(re.escape(scan) + r"\s*\(", body)]
    if not calls:
        out.append(f"never enumerates with {scan}() (the refusal needs its diagnostics)")
    for c in calls:
        # The call must come after the question, and the question's `if (` must still be open: the text between the
        # question and the call has to contain an unclosed `{`.
        if q == -1 or c < q:
            out.append(f"{scan}() is called before the O(1) question")
            continue
        between = body[q:c]
        if between.count("{") <= between.count("}"):
            out.append(f"{scan}() is called outside the branch the O(1) question opened")
    return out


def test_host_zone_settle_does_not_scan_the_registry_when_clean():
    body = function_body(CODE, HOST_SETTLE)
    assert settle_violations(body, "runtime_registry_host_zone_live_locked", SCAN_HOST) == []


def test_zone_settle_does_not_scan_the_registry_when_clean():
    body = function_body(CODE, VRAM_SETTLE)
    assert settle_violations(body, "runtime_registry_span_live_locked", SCAN_SPAN) == []


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
    assert settle_violations(mutant, "runtime_registry_host_zone_live_locked", SCAN_HOST), mutant


def test_settle_gate_accepts_the_intended_shape():
    assert settle_violations(HOST_OK, "runtime_registry_host_zone_live_locked", SCAN_HOST) == []


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
    out = []
    if "g_runtime_alloc_index.find_first_base_in(" not in body:
        out.append("does not query the index")
    guard = re.search(r"if\s*\(\s*g_runtime_span_irregular_rows\s*==\s*0\s*\)\s*\{", body)
    if guard is None:
        out.append("has no irregular-row guard")
        return out
    # The index answer sits inside the guard and returns; any loop must come after the guard's block has closed.
    depth = 0
    close = None
    for i in range(guard.end() - 1, len(body)):
        if body[i] == "{":
            depth += 1
        elif body[i] == "}":
            depth -= 1
            if depth == 0:
                close = i
                break
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
        # the guarded block no longer returns the index answer
        SPAN_OK.replace("return g_runtime_alloc_index.find_first_base_in(lo, hi, nullptr);", "(void) g_runtime_alloc_index.find_first_base_in(lo, hi, nullptr);"),
    ],
)
def test_span_live_gate_has_a_witness(mutant):
    assert span_live_violations(mutant), mutant


def counter_writes_outside_the_count_helper(code: str):
    start, end = function_span(code, COUNT_ROW)
    bad = []
    write = re.compile(
        r"\b(?:" + "|".join(COUNTERS) + r")\b(?:\s*\[[^\]]*\])?\s*(?:(?:[-+*/%|&^]|<<|>>)?=(?!=)|\+\+|--)"
        r"|(?:\+\+|--)\s*\b(?:" + "|".join(COUNTERS) + r")\b"
        # a counter bound to a non-const reference could be written through it
        r"|(?<!const )(?:auto|size_t)\s*&&?\s*\w+\s*=\s*(?:" + "|".join(COUNTERS) + r")\b"
    )
    for m in write.finditer(code):
        if start <= m.start() < end:
            continue
        line_start = code.rfind("\n", 0, m.start()) + 1
        if re.fullmatch(r"static\s+size_t\s+", code[line_start : m.start()]):
            continue  # the declaration's own initialiser
        bad.append((code.count("\n", 0, m.start()) + 1, m.group(0)[:60].replace("\n", " ")))
    return bad


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
    ],
)
def test_counter_write_gate_has_a_witness(line):
    planted = CODE + "\nvoid f() {\n    " + line + "\n}\n"
    assert counter_writes_outside_the_count_helper(planted), line


@pytest.mark.parametrize(
    "line",
    [
        "if (g_runtime_span_irregular_rows == 0) { return true; }",
        "return g_runtime_host_zone_rows[0] != 0;",
        "const size_t n = g_runtime_host_zone_rows[1];",
        "if (g_runtime_span_irregular_rows >= 1) {}",
    ],
)
def test_counter_write_gate_allows_reads(line):
    planted = CODE + "\nvoid f() {\n    " + line + "\n}\n"
    assert counter_writes_outside_the_count_helper(planted) == [], line


def test_the_count_helper_is_called_only_by_the_three_registry_helpers():
    assert uses_outside(CODE, "runtime_registry_count_row_locked", [COUNT_ROW, EMPLACE, ERASE_IT, ASSIGN]) == []


@pytest.mark.parametrize("signature", [EMPLACE, ERASE_IT, ASSIGN])
def test_each_registry_mutation_helper_updates_the_counters(signature):
    assert "runtime_registry_count_row_locked(" in function_body(CODE, signature)


def test_the_count_helper_calls_have_a_witness():
    planted = CODE + "\nvoid f() {\n    runtime_registry_count_row_locked(p, h, true);\n}\n"
    assert uses_outside(planted, "runtime_registry_count_row_locked", [COUNT_ROW, EMPLACE, ERASE_IT, ASSIGN]) != []
    forgot = function_body(CODE, ERASE_IT).replace("runtime_registry_count_row_locked(", "forgot(")
    assert "runtime_registry_count_row_locked(" not in forgot


# A registered row's host zone is counted when the row is registered and uncounted when it leaves, so it must not be
# rewritten in between. test-sycl-runtime-registry-index-source.py forbids writing a row's handle, ptr and size through the
# registry; this adds the one field the counters key on.
_ASSIGN = r"(?:(?:[-+*/%|&^]|<<|>>)?=(?!=)|\+\+|--)"
HOST_ZONE_WRITE_RE = re.compile(r"(?:->|\.)second\s*\.\s*handle\s*\.\s*host_zone\s*" + _ASSIGN)


def test_a_registered_rows_host_zone_is_never_rewritten():
    assert HOST_ZONE_WRITE_RE.findall(CODE) == []


@pytest.mark.parametrize(
    "line",
    [
        "it->second.handle.host_zone = host_zone_id::KV;",
        "registry.find(p)->second.handle.host_zone=host_zone_id::KV;",
        "it->second.handle.host_zone |= x;",
    ],
)
def test_host_zone_write_gate_has_a_witness(line):
    assert HOST_ZONE_WRITE_RE.findall(CODE + "\nvoid f() {\n    " + line + "\n}\n"), line


def test_host_zone_write_gate_allows_reads():
    planted = CODE + "\nvoid f() {\n    if (it->second.handle.host_zone == zone) {}\n}\n"
    assert HOST_ZONE_WRITE_RE.findall(planted) == []


def test_counters_and_scan_helpers_stay_file_static():
    offenders = []
    names = COUNTERS + (SCAN_HOST, SCAN_SPAN, "runtime_registry_host_zone_live_locked", "runtime_registry_span_live_locked")
    for path in sorted(SYCL_DIR.rglob("*")):
        if path.suffix not in (".cpp", ".hpp", ".h", ".cu", ".cuh") or path.name == "unified-cache.cpp":
            continue
        text = strip_comments(path.read_text(errors="replace"))
        if any(n in text for n in names):
            offenders.append(path.name)
    assert offenders == []


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
