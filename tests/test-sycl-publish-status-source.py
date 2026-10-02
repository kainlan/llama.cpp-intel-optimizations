"""Source gate for the publish's status type and for the deleted BUSY retries
(zhcn design gate 22b, and the publish-function clauses of gate 37).

The runtime-context publish, `llama_context::sycl_publish_runtime_context()`,
returns `sched_reserve_result` and never waits:

- no `throw` in its body, no loop over a lifecycle result, no `sleep_for`;
- its returns are REFUSED "failed to activate exact SYCL model plan:
  result=%d" for any non-OK result, and OK at the end;
- the BUSY arm tracks moua's L0 (E). E is the mechanical fact that the bodies
  of ggml_backend_sycl_can_unload, _shutdown, _commit_reactivate,
  _rollback_reactivate and _complete_unload each construct a
  `ggml_sycl_replan_token`. While E is false the publish must return FAILED
  "SYCL backend module is not ACTIVE" for a BUSY; once E is true (it is, in
  the combined merge's base) a surviving FAILED return fails, and the arm is
  the [CONTEXT-PLAN-BUG] conversion of a BUSY, a drift detector only, since
  the backend's module guard is then the single emission site.

llama-context.cpp has no sleep_for and no BUSY retry loop, and the identifier
`llama_context_sycl_max_busy_waits` occurs nowhere in src/ and nowhere in
tests/ except the file that asserts its absence.

Comment- and string-stripped where a statement is matched. Every clause has a
mutant that must fail it. Host-only; collected by pytest.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_CPP = (ROOT / "src/llama-context.cpp").read_text()
SYCL_CPP = (ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()

_LEXEME_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"'
    r"|'(?:\\.|[^'\\\n])*'"
    r"|//[^\n]*"
    r"|/\*.*?\*/",
    flags=re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def strip_strings_and_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return '""'
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def z(text: str) -> str:
    return re.sub(r"(?<!\w) | (?!\w)", "", re.sub(r"\s+", " ", text))


def brace_body(code: str, start: int) -> str:
    i = code.index("{", start)
    depth = 0
    for j in range(i, len(code)):
        if code[j] == "{":
            depth += 1
        elif code[j] == "}":
            depth -= 1
            if depth == 0:
                return code[start : j + 1]
    raise AssertionError("unbalanced braces")


_PUBLISH_SIGNATURE = "sched_reserve_result llama_context::sycl_publish_runtime_context(bool flash_attn)"


def publish_body(raw: str = CONTEXT_CPP) -> str:
    code = z(strip_comments(raw))
    at = code.find(z(_PUBLISH_SIGNATURE) + "{")
    assert at != -1, f"{_PUBLISH_SIGNATURE} not defined"
    return brace_body(code, at)


# --- gate 22b: nothing waits on a lifecycle result -----------------------------


def no_waits_ok(raw: str) -> bool:
    code = strip_comments(raw)
    if "sleep_for" in code or "llama_context_sycl_max_busy_waits" in code:
        return False
    # a loop whose condition tests a lifecycle result
    return not re.search(r"\b(?:for|while)\s*\([^)]*GGML_SYCL_LIFECYCLE_\w+", code)


def test_llama_context_has_no_busy_retry_loop():
    assert no_waits_ok(CONTEXT_CPP), (
        "llama-context.cpp must have no sleep_for, no loop over a GGML_SYCL_LIFECYCLE_ result and no "
        "llama_context_sycl_max_busy_waits"
    )


def test_no_wait_mutants():
    anchor = "static const llm_fused_op_probe llm_fused_op_lid_probe = {"
    assert CONTEXT_CPP.count(anchor) == 1
    mutants = {
        "the shared constant left behind": "static constexpr int llama_context_sycl_max_busy_waits = 7;\n",
        "a sleep on a result": "static void f() { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }\n",
        "a retry loop over a result": "static void g(int rc) { for (int w = 0; rc == GGML_SYCL_LIFECYCLE_BUSY && w < 7; ++w) {} }\n",
        "a while loop over a result": "static void h(int rc) { while (rc == GGML_SYCL_LIFECYCLE_BUSY) {} }\n",
    }
    for name, text in mutants.items():
        assert not no_waits_ok(CONTEXT_CPP.replace(anchor, text + anchor, 1)), f"mutant {name!r} slipped through"


def busy_identifier_files() -> list:
    """Every file under src/ and tests/ that names the deleted constant,
    except the two gates that name it only to assert its absence."""
    allowed = {"test-sycl-auto-ubatch-source.py", "test-sycl-publish-status-source.py"}
    hits = []
    for base in (ROOT / "src", ROOT / "tests"):
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix not in {".cpp", ".h", ".hpp", ".py", ".txt", ".cmake", ".sh"}:
                continue
            if path.name in allowed:
                continue
            if "llama_context_sycl_max_busy_waits" in path.read_text(errors="ignore"):
                hits.append(str(path.relative_to(ROOT)))
    return hits


def test_busy_waits_identifier_is_gone():
    assert busy_identifier_files() == []


def test_busy_waits_identifier_mutant():
    planted = ROOT / "src" / "zz-planted-busy-waits.h"
    assert not planted.exists()
    try:
        planted.write_text("static constexpr int llama_context_sycl_max_busy_waits = 7;\n")
        assert busy_identifier_files() == ["src/zz-planted-busy-waits.h"]
    finally:
        planted.unlink()
    assert busy_identifier_files() == []


# --- gate 37: the publish function's type ---------------------------------------

_BUSY_ARM = z('if (rc == GGML_SYCL_LIFECYCLE_BUSY) { return { sched_reserve_status::FAILED, "SYCL backend module is not ACTIVE" }; }')
_BUSY_RETURN = z('return { sched_reserve_status::FAILED, "SYCL backend module is not ACTIVE" };')
_DRIFT_ARM = z(
    'if (rc == GGML_SYCL_LIFECYCLE_BUSY) { LLAMA_LOG_ERROR("[CONTEXT-PLAN-BUG] publish answered BUSY under the '
    'replan scope: module guard drifted\\n"); }'
)
_REFUSED_RETURN = z('return { sched_reserve_status::REFUSED, format("failed to activate exact SYCL model plan: result=%d", (int) rc) };')
_REFUSED_ARM = z('if (rc != GGML_SYCL_LIFECYCLE_OK) { ') + _REFUSED_RETURN + "}"
_OK_RETURN = z('return { sched_reserve_status::OK, "" };')


def publish_ok(body: str, e: bool) -> bool:
    """e is moua's L0 having landed: the FAILED arm is then forbidden and the
    BUSY conversion required; while it is false the FAILED arm is required."""
    if "throw" in body or "sleep_for" in body:
        return False
    if re.search(r"\b(?:for|while)\(", body.replace("for(auto&backend:backends)", "")):
        return False
    if _REFUSED_ARM not in body or not body.endswith(_OK_RETURN + "}"):
        return False
    if e:
        if _DRIFT_ARM not in body or "sched_reserve_status::FAILED" in body or _BUSY_ARM in body:
            return False
        # the conversion sits before the generic refusal, so a BUSY is still a REFUSED
        if body.index(_DRIFT_ARM) > body.index(_REFUSED_ARM):
            return False
        expected = {_REFUSED_RETURN, _OK_RETURN}
    else:
        if _BUSY_ARM not in body or _DRIFT_ARM in body:
            return False
        expected = {_BUSY_RETURN, _REFUSED_RETURN, _OK_RETURN}
    returns = re.findall(r"return\{[^;]*\};", body)
    return len(re.findall(r"\breturn\b", body)) == len(expected) and all(r in expected for r in returns)


def test_publish_returns_a_status_and_never_waits():
    e = e_is_true(SYCL_CPP)
    assert publish_ok(publish_body(), e), (
        f"E is {e}: the publish must return sched_reserve_result with the pinned returns, no throw, no wait, "
        f"and the BUSY arm of its {'second' if e else 'first'} form"
    )


def test_publish_mutants_common_to_both_forms():
    body = publish_body()
    for e in (True, False):
        # build a body in the form E selects, from the tree's own
        base = body if publish_ok(body, e) else convert(body, not e)
        assert publish_ok(base, e)
        mutants = {
            "a throw back in": (_REFUSED_RETURN, z('throw std::runtime_error("x");')),
            "REFUSED turned into FAILED": (_REFUSED_RETURN, _REFUSED_RETURN.replace("REFUSED", "FAILED")),
            "the result code dropped from the reason": ('format("failed to activate exact SYCL model plan: result=%d", (int) rc)', '"failed to activate exact SYCL model plan"'),
            "the trailing OK return replaced": (_OK_RETURN, z('return { sched_reserve_status::REFUSED, "" };')),
            "an extra return": ("continue;", 'return { sched_reserve_status::OK, "" };'),
            "a sleep before returning": (_OK_RETURN + "}", "std::this_thread::sleep_for(std::chrono::milliseconds(1));" + _OK_RETURN + "}"),
            "a retry loop around the call": ("const auto rc =", "for(int w=0;w<7;++w) const auto rc ="),
        }
        for name, (old, new) in mutants.items():
            old, new = z(old), z(new)
            assert base.count(old) >= 1, f"{name}: anchor missing: {old!r}"
            assert not publish_ok(base.replace(old, new, 1), e), f"mutant {name!r} (E={e}) slipped through"


def convert(body: str, from_e: bool) -> str:
    """The publish written in the form E=from_e selects, rewritten in the
    other form, for the controls."""
    if from_e:
        assert _DRIFT_ARM in body
        return body.replace(_DRIFT_ARM, _BUSY_ARM, 1)
    assert _BUSY_ARM in body
    return body.replace(_BUSY_ARM, _DRIFT_ARM, 1)


def test_planted_controls_c1_c2():
    # C1: the FAILED arm present with E true fails. C2: the FAILED arm absent with E false fails.
    assert not publish_ok(_with_failed_arm(), True), "C1: the arm present with E true must fail"
    assert not publish_ok(_without_failed_arm(), False), "C2: the arm absent with E false must fail"


def _with_failed_arm() -> str:
    """The publish with the FAILED BUSY arm, whatever form the tree is in."""
    body = publish_body()
    return body.replace(_DRIFT_ARM, _BUSY_ARM, 1) if _DRIFT_ARM in body else body


def _without_failed_arm() -> str:
    """The publish with the drift-detector arm instead."""
    body = publish_body()
    return body.replace(_BUSY_ARM, _DRIFT_ARM, 1) if _BUSY_ARM in body else body


def test_throwing_form_delegates_to_the_publish():
    code = z(strip_comments(CONTEXT_CPP))
    at = code.find(z("void llama_context::sycl_resync_runtime_context_flash_attn()") + "{")
    assert at != -1
    body = brace_body(code, at)
    assert body == z(
        "void llama_context::sycl_resync_runtime_context_flash_attn() { "
        "const sched_reserve_result result = sycl_publish_runtime_context(cparams.flash_attn); "
        "if (result.status != sched_reserve_status::OK) { throw std::runtime_error(result.reason); } }"
    )


# --- E: has moua's L0 landed? ---------------------------------------------------

_E_FUNCTIONS = (
    "ggml_backend_sycl_can_unload",
    "ggml_backend_sycl_shutdown",
    "ggml_backend_sycl_commit_reactivate",
    "ggml_backend_sycl_rollback_reactivate",
    "ggml_backend_sycl_complete_unload",
)


def e_is_true(sycl_src: str) -> bool:
    code = strip_strings_and_comments(sycl_src)
    for name in _E_FUNCTIONS:
        m = re.search(rf"^[\w:<>\s*&]*\b{name}\s*\([^;{{]*\)\s*(?:noexcept\s*)?\{{", code, re.M)
        if not m:
            return False
        body = brace_body(code, m.start())
        # a declaration of a ggml_sycl_replan_token object, whole identifier
        if not re.search(r"\bggml_sycl_replan_token\s+\w+\s*[({;=]", body):
            return False
    return True


_EXCLUDED = ("ggml_backend_sycl_prepare_reactivate", "ggml_backend_sycl_finalize_reactivate", "ggml_backend_sycl_cancel_unload", "ggml_backend_sycl_reg")


def _fake_sycl(with_tokens: set, extra_with_token: str = "", body_line: str = "int x = 0;") -> str:
    out = []
    for name in _E_FUNCTIONS:
        tok = "    ggml_sycl_replan_token tok(g_replan_txn_mutex);\n" if name in with_tokens else f"    {body_line}\n"
        out.append(f"int {name}(void) {{\n{tok}    return 0;\n}}\n")
    for name in _EXCLUDED:
        tok = "    ggml_sycl_replan_token tok(g_replan_txn_mutex);\n" if name == extra_with_token else "    int y = 0;\n"
        out.append(f"int {name}(void) {{\n{tok}    return 0;\n}}\n")
    return "\n".join(out)


def test_e_predicate_controls():
    full = set(_E_FUNCTIONS)
    assert e_is_true(_fake_sycl(full))
    # C3..C7: one of the five lacks the construction, the others hold it
    for name in _E_FUNCTIONS:
        assert not e_is_true(_fake_sycl(full - {name})), name
    # C8a..C8d: the token is constructed in one excluded body while one of the five lacks it
    for excluded in _EXCLUDED:
        for lacking in _E_FUNCTIONS:
            assert not e_is_true(_fake_sycl(full - {lacking}, extra_with_token=excluded)), (excluded, lacking)
    # C9: a comment is not a construction; C10: the accessor is a different identifier;
    # C11: a string literal is not a construction
    for line in ("// ggml_sycl_replan_token tok(m);", "bool h = ggml_sycl_replan_token_held();", 'const char * s = "ggml_sycl_replan_token tok(m);";'):
        assert not e_is_true(_fake_sycl(full - {"ggml_backend_sycl_shutdown"}, body_line=line)), line
    # a nested construction counts like any other
    nested = _fake_sycl(full).replace(
        "    ggml_sycl_replan_token tok(g_replan_txn_mutex);\n    return 0;",
        "    if (true) {\n        ggml_sycl_replan_token tok(g_replan_txn_mutex);\n    }\n    return 0;",
        1,
    )
    assert e_is_true(nested)
    # the try-lock form counts
    assert e_is_true(_fake_sycl(full).replace("tok(g_replan_txn_mutex)", "tok(g_replan_txn_mutex, std::try_to_lock)"))


def test_busy_arm_tracks_e():
    e = e_is_true(SYCL_CPP)
    assert publish_ok(publish_body(), e), f"E is {e}: the publish's BUSY arm must be in its {'second' if e else 'first'} form"


# --- I1: a BUSY (any non-OK) publish is re-attempted at the next boundary, never waited on ---

SYCL_H = (ROOT / "ggml/include/ggml-sycl.h").read_text()


def reattempt_ok(context_src: str) -> bool:
    """sched_reserve_nothrow leaves sched_need_reserve set on every way out but OK, and decode and
    encode answer -2 for a false return, so the next call runs the transaction (publish included) again."""
    code = z(strip_comments(context_src))
    at = code.find(z("bool llama_context::sched_reserve_nothrow()") + "{")
    if at == -1:
        return False
    body = brace_body(code, at)
    # the flag is cleared once, before the attempt, and set again on the single fall-through return false
    if body.count(z("sched_need_reserve = false;")) != 1 or body.count(z("sched_need_reserve = true;")) != 1:
        return False
    if not body.endswith(z("sched_need_reserve = true; return false; }")):
        return False
    # the only early return true is the OK arm of the attempt and the clean-flag fast path
    if len(re.findall(r"\breturn true;", body)) != 2:
        return False
    # every catch arm falls through to the flag: a catch that returned would skip it
    if re.search(r"catch\([^)]*\)\{[^}]*\breturn\b", body):
        return False
    if "catch(...)" not in body:
        return False
    for caller in ("llama_context::encode(", "llama_context::decode("):
        c = z(strip_comments(context_src))
        i = c.find(z("int llama_context::" + caller[len("llama_context::") :]))
        if i == -1:
            return False
        cb = brace_body(c, i)
        if z("if (!sched_reserve_nothrow()) {") not in cb:
            return False
        arm = cb[cb.index(z("if (!sched_reserve_nothrow()) {")) :]
        arm = arm[: arm.index("}") + 1]
        if "return-2;" not in arm:
            return False
        # work is counted once the reserve succeeded: a -2 processed nothing
        counted = cb.find("n_queued_tokens+=")
        if counted == -1 or counted < cb.index(z("if (!sched_reserve_nothrow()) {")):
            return False
    return True


def test_a_failed_publish_is_reattempted_at_the_next_boundary():
    assert reattempt_ok(CONTEXT_CPP)


def test_reattempt_mutants():
    anchor = "    sched_need_reserve = true;\n    return false;\n}"
    assert CONTEXT_CPP.count(anchor) == 1
    mutants = {
        "the flag not set on failure": anchor.replace("    sched_need_reserve = true;\n", ""),
        "a catch that returns": None,
    }
    assert not reattempt_ok(CONTEXT_CPP.replace(anchor, mutants["the flag not set on failure"], 1))
    ret = CONTEXT_CPP.replace(
        'LLAMA_LOG_ERROR("%s: unknown exception\\n", __func__);', 'LLAMA_LOG_ERROR("%s: unknown exception\\n", __func__); return false;', 1
    )
    assert ret != CONTEXT_CPP and not reattempt_ok(ret)
    dec = CONTEXT_CPP.replace(
        'LLAMA_LOG_ERROR("%s: failed to reserve the compute buffers\\n", __func__);\n        return -2;',
        'LLAMA_LOG_ERROR("%s: failed to reserve the compute buffers\\n", __func__);\n        return 0;',
        1,
    )
    assert dec != CONTEXT_CPP and not reattempt_ok(dec)
    # the decode counts its tokens before the reserve again
    early = CONTEXT_CPP.replace(
        "    // counted only once the reserve succeeded: a -2 here processed nothing\n    n_queued_tokens += n_tokens_all;\n", "", 1
    ).replace("    output_swaps.clear();\n\n    if (!sched_reserve_nothrow()) {", "    n_queued_tokens += n_tokens_all;\n    output_swaps.clear();\n\n    if (!sched_reserve_nothrow()) {", 1)
    assert early != CONTEXT_CPP and not reattempt_ok(early)


def header_busy_text_ok(h: str) -> bool:
    """The BUSY contract in ggml-sycl.h names the next-boundary re-attempt and promises no wait."""
    flat = re.sub(r"\s*//\s*", " ", h)
    i = flat.find("GGML_SYCL_LIFECYCLE_BUSY (round 1 F6; round 4 Q3)")
    if i == -1:
        return False
    para = flat[i : i + 600]
    if "MAY retry" in para or "BUSY backoff" in flat or "BUSY retry" in flat:
        return False
    return "NEXT boundary" in para and "never by waiting" in para


def test_header_says_next_boundary_not_retry():
    assert header_busy_text_ok(SYCL_H)


def test_header_mutants():
    assert not header_busy_text_ok(SYCL_H.replace("never by\n// waiting", "by\n// waiting", 1))
    assert not header_busy_text_ok(SYCL_H.replace("NEXT boundary", "next call", 1))
    assert not header_busy_text_ok(SYCL_H + "\n// BUSY backoff\n")


def comments_say_no_backoff(context_src: str) -> bool:
    """No comment in llama-context.cpp still describes a BUSY backoff: the code does not retry, and a comment
    that says it does sends the next reader looking for a loop that is gone."""
    comments = re.findall(r"//[^\n]*", context_src) + re.findall(r"/\*.*?\*/", context_src, flags=re.DOTALL)
    return not any(re.search(r"backoff|exponential", c, flags=re.IGNORECASE) for c in comments)


def test_comments_in_llama_context_say_no_backoff():
    assert comments_say_no_backoff(CONTEXT_CPP)


def test_comment_backoff_mutants():
    assert not comments_say_no_backoff(CONTEXT_CPP + "\n// the probe retries with a bounded exponential BUSY backoff\n")
    assert not comments_say_no_backoff(CONTEXT_CPP + "\n/* the full transaction's own Backoff */\n")
