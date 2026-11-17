"""Source contract: llama-completion picks the jinja engine by itself when the built-in matcher cannot render the model's
chat template.

Upstream turns the jinja engine off by default for this tool (common/arg.cpp), and the built-in matcher
(llama_chat_apply_template) knows a fixed list of templates. A model outside that list (Gemma 4) failed after the model
load with "this custom template is not supported, try using --jinja". The wiring that must hold:

(A) common/chat.h declares and common/chat.cpp defines common_chat_templates_need_jinja(), which asks the built-in
    matcher (common_chat_verify_template with use_jinja == false) about the template set's source.
(B) llama_completion() decides once, after conversation mode is resolved and before anything reads params.use_jinja
    (the example print, add_bos, common_chat_format_single, inputs.use_jinja): when the tool renders the template at all
    (params.conversation_mode && params.enable_chat_template), the user did not choose jinja (!params.use_jinja) and the
    helper says the matcher cannot render it, it sets params.use_jinja = true and says so at LOG_WRN. Raw mode (-no-cnv)
    does not apply the template, and there use_jinja also decides whether the prompt gets a BOS, so it is left alone.

Host-only, pure text assertions. llama_test_pytest hands this file to pytest.main(), so the checks live inside test_*()
functions. Checks run against COMMENT-STRIPPED text, and each has a mutation witness so it is known to fail on the
regression it guards.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMPLETION_CPP = (ROOT / "tools/completion/completion.cpp").read_text()
CHAT_H = (ROOT / "common/chat.h").read_text()
CHAT_CPP = (ROOT / "common/chat.cpp").read_text()

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


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


FN_START = "int llama_completion(int argc, char ** argv) {"
HELPER = "common_chat_templates_need_jinja"
RESOLVED_ANCHOR = "params.conversation_mode = COMMON_CONVERSATION_MODE_DISABLED;"
CONDITION_TERMS = (
    "params.conversation_mode",
    "params.enable_chat_template",
    "!params.use_jinja",
    HELPER + "(",
)


def function_body(code: str) -> str:
    """The body of llama_completion(): from its opening line to the first closing brace in column 0."""
    s = code.find(FN_START)
    assert s != -1, f"{FN_START!r} not found in tools/completion/completion.cpp"
    e = code.find("\n}\n", s)
    assert e != -1, "could not find the end of llama_completion()"
    return code[s:e]


def match_brace(text: str, open_idx: int) -> int:
    assert text[open_idx] == "{"
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    raise AssertionError("unbalanced braces")


def fallback_block(body: str):
    """(start, brace_open, end) of the `if (... HELPER(...)) { ... }` statement inside the function body, or None."""
    h = body.find(HELPER + "(")
    if h == -1:
        return None
    s = body.rfind("if (", 0, h)
    if s == -1:
        return None
    b = body.find("{", h)
    if b == -1:
        return None
    return s, b, match_brace(body, b)


def violations_completion(cpp: str) -> list[str]:
    found: list[str] = []
    code = strip_comments(cpp)
    body = function_body(code)
    blk = fallback_block(body)
    if blk is None:
        return ["llama_completion() has no `if (... " + HELPER + "(...)) { ... }` fallback block"]
    s, b, e = blk
    cond = norm(body[s:b])
    for term in CONDITION_TERMS:
        if term not in cond:
            found.append(f"the fallback condition does not name {term!r}: {cond}")
    block = norm(body[b : e + 1])
    if not re.search(r"params\.use_jinja\s*=\s*true\s*;", block):
        found.append("the fallback block does not set params.use_jinja = true")
    if "LOG_WRN(" not in block:
        found.append("the fallback block must say it switched engines at LOG_WRN (LOG_INF is dropped at default verbosity)")
    anchor = body.find(RESOLVED_ANCHOR)
    if anchor == -1:
        found.append(f"anchor {RESOLVED_ANCHOR!r} (conversation mode resolution) not found in llama_completion()")
    elif s < anchor:
        found.append("the fallback block lies before the conversation-mode resolution; the condition would read AUTO")
    other_reads = [m.start() for m in re.finditer(r"use_jinja", body) if not (s <= m.start() <= e)]
    early = [p for p in other_reads if p < s]
    if early:
        found.append("something reads use_jinja before the fallback block: " + norm(body[early[0] - 40 : early[0] + 60]))
    add_bos = body.find("const bool add_bos")
    if add_bos == -1:
        found.append("the add_bos line was not found in llama_completion()")
    elif s > add_bos:
        found.append("the fallback block lies after the add_bos line, which reads params.use_jinja")
    return found


def violations_helper(chat_h: str, chat_cpp: str) -> list[str]:
    found: list[str] = []
    h = strip_comments(chat_h)
    if not re.search(r"bool\s+" + HELPER + r"\(\s*const\s+struct\s+common_chat_templates\s*\*\s*\w*\s*\)\s*;", h):
        found.append("common/chat.h does not declare bool " + HELPER + "(const struct common_chat_templates *)")
    c = strip_comments(chat_cpp)
    m = re.search(r"bool\s+" + HELPER + r"\(\s*const\s+struct\s+common_chat_templates\s*\*\s*\w*\s*\)\s*\{", c)
    if not m:
        found.append("common/chat.cpp does not define " + HELPER + "()")
        return found
    body = norm(c[m.end() : match_brace(c, m.end() - 1)])
    if not re.search(r"common_chat_verify_template\(\s*\w+\s*,\s*(/\*[^*]*\*/\s*)?false\s*\)", body):
        found.append(HELPER + "() must ask common_chat_verify_template(<source>, false), the built-in matcher")
    if "common_chat_templates_source(" not in body:
        found.append(HELPER + "() must take the template text from common_chat_templates_source()")
    return found


# ---------------------------------------------------------------------------
# mutations (applied to raw source, so a mutated file is what a regression would look like)
# ---------------------------------------------------------------------------


def raw_block_span(raw: str):
    m = re.search(r"if \(\s*params\.conversation_mode\b[^{]*" + HELPER + r"\([^{]*\{", raw)
    assert m, "fallback block not found in raw source"
    return m.start(), match_brace(raw, m.end() - 1) + 1


def drop_term(term_regex: str):
    def mutate(raw: str) -> str:
        s, e = raw_block_span(raw)
        block = raw[s:e]
        head, sep, tail = block.partition("{")
        new_head, n = re.subn(term_regex, "", head, count=1)
        assert n == 1, term_regex
        return raw[:s] + new_head + sep + tail + raw[e:]

    return mutate


def move_after_add_bos(raw: str) -> str:
    s, e = raw_block_span(raw)
    block = raw[s:e]
    rest = raw[:s] + raw[e:]
    m = re.search(r"const bool add_bos = [^\n]*\n", rest)
    assert m, "add_bos line not found"
    return rest[: m.end()] + block + "\n" + rest[m.end() :]


def warn_to_info(raw: str) -> str:
    s, e = raw_block_span(raw)
    return raw[:s] + raw[s:e].replace("LOG_WRN", "LOG_INF") + raw[e:]


def drop_assignment(raw: str) -> str:
    s, e = raw_block_span(raw)
    block, n = re.subn(r"params\.use_jinja\s*=\s*true\s*;", "", raw[s:e])
    assert n == 1
    return raw[:s] + block + raw[e:]


def move_before_resolution(raw: str) -> str:
    s, e = raw_block_span(raw)
    block = raw[s:e]
    rest = raw[:s] + raw[e:]
    m = re.search(r"// auto enable conversation mode if chat template is available\n", rest)
    assert m, "resolution comment not found"
    return rest[: m.start()] + block + "\n\n" + rest[m.start() :]


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_the_helper_is_declared_and_asks_the_builtin_matcher():
    assert violations_helper(CHAT_H, CHAT_CPP) == []


def test_helper_gate_has_a_witness():
    assert violations_helper(CHAT_H, CHAT_CPP) == [], "the unmutated source must pass first"
    flipped = re.sub(
        r"(common_chat_verify_template\(\s*\w+\s*,\s*(?:/\*[^*]*\*/\s*)?)false", r"\1true", CHAT_CPP
    )
    assert flipped != CHAT_CPP
    assert violations_helper(CHAT_H, flipped), "asking the jinja verifier cannot detect an unsupported built-in template"
    assert violations_helper(CHAT_H.replace(HELPER, "common_chat_templates_gone"), CHAT_CPP)


def test_completion_switches_to_jinja_when_the_matcher_cannot_render():
    assert violations_completion(COMPLETION_CPP) == []


@pytest.mark.parametrize(
    "name, mutate",
    [
        ("drop conversation_mode term", drop_term(r"params\.conversation_mode\s*&&\s*")),
        ("drop enable_chat_template term", drop_term(r"params\.enable_chat_template\s*&&\s*")),
        ("drop !use_jinja term", drop_term(r"!params\.use_jinja\s*&&\s*")),
        ("drop helper term", drop_term(HELPER + r"\(chat_templates\.get\(\)\)")),
        ("move block after add_bos", move_after_add_bos),
        ("move block before the resolution", move_before_resolution),
        ("LOG_WRN to LOG_INF", warn_to_info),
        ("drop the use_jinja assignment", drop_assignment),
    ],
)
def test_completion_gate_has_a_witness(name, mutate):
    assert violations_completion(COMPLETION_CPP) == [], "the unmutated source must pass first"
    assert violations_completion(mutate(COMPLETION_CPP)), name


def test_nothing_reads_use_jinja_between_template_creation_and_the_block():
    code = strip_comments(COMPLETION_CPP)
    body = function_body(code)
    blk = fallback_block(body)
    assert blk is not None, "fallback block missing"
    created = body.find("common_chat_templates_init(")
    assert created != -1
    window = body[created : blk[0]]
    assert "use_jinja" not in window, "use_jinja is read before the fallback decides it"
    assert "g_params" not in window, "g_params->use_jinja may be read before the fallback decides it"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
