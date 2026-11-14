"""A graph input that fails its set_input() stops the ubatch before graph_compute."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_CPP = ROOT / "src/llama-context.cpp"

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


def read_code(path: Path) -> str:
    return strip_comments(path.read_text())


def balanced(code: str, open_idx: int, open_ch: str = "{", close_ch: str = "}") -> str:
    depth = 0
    for i in range(open_idx, len(code)):
        if code[i] == open_ch:
            depth += 1
        elif code[i] == close_ch:
            depth -= 1
            if depth == 0:
                return code[open_idx + 1 : i]
    raise AssertionError("unbalanced brackets")


def function_body(code: str, signature_re: str) -> str:
    m = re.search(signature_re, code)
    assert m, f"function not found: {signature_re}"
    return balanced(code, code.index("{", m.end()))



def test_process_ubatch_stops_on_a_failed_input():
    body = function_body(
        read_code(CONTEXT_CPP),
        r"llm_graph_result\s*\*\s*llama_context::process_ubatch\s*\((?:[^()]|\([^()]*\))*\)",
    )
    i_set = body.find("set_inputs(")
    i_fail = body.find("inputs_failed()")
    i_compute = body.find("graph_compute(")
    assert 0 <= i_set < i_fail < i_compute, (
        "a failed input must stop the ubatch between set_inputs and graph_compute"
    )
