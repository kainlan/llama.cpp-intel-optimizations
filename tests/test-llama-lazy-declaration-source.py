"""Source contract for llama.cpp-5efe C1: the lazy-tensor declaration matches the
create_tensor() call sites, both ways.

`llama_model_lazy_tensor_names(arch)` (src/llama-model.cpp) declares, per
architecture, the GGUF names that arch creates with TENSOR_READ_LAZY. The
lazy-range record is pre-filled from it, before the early plan and before any
create_tensor() call, because TENSOR_READ_LAZY is first knowable only at the call.
One source, checked twice: statically here, and at runtime by
llama_lazy_range_check_create(), mode-independently.

This gate:

1. extracts every create_tensor() call in src/models/*.cpp that passes
   TENSOR_READ_LAZY;
2. maps its model class to its LLM_ARCH through the registration in
   llama_model_mapping() (src/llama-model.cpp), and its tn(LLM_TENSOR_*, suffix
   [, bid]) to the GGUF name through LLM_TENSOR_NAMES (src/llama-arch.cpp), a bid
   argument becoming "blk.%d.";
3. requires a declaration entry under THAT arch for each call, and a flagged call
   site in that arch's file for each entry (an upstream merge that adds a flagged
   call, or an arch that arrives with one, fails here at build time, on every
   backend's tree, before any load). The arch key is load-bearing: gemma3n creates
   the same "per_layer_token_embd.weight" without the flag;
4. requires llama_model_lazy_tensor_attrs() to have a case for every arch that has
   names, and no case for any other arch;
5. requires the declaration to return names only: no static beside it stores an
   hparams-valued field, and llama_model_lazy_tensor_attrs() writes nothing but its
   return value (a function-local static cache would be shared by every loaded
   model).
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LLAMA_MODEL_CPP = ROOT / "src/llama-model.cpp"
LLAMA_ARCH_CPP = ROOT / "src/llama-arch.cpp"
MODELS_DIR = ROOT / "src/models"

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


def balanced(code: str, open_idx: int, open_ch: str = "(", close_ch: str = ")") -> str:
    """The text between the bracket at open_idx and its match, brackets excluded."""
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
    return balanced(code, code.index("{", m.end()), "{", "}")


def tensor_base_names() -> dict:
    """LLM_TENSOR_* -> the base name of LLM_TENSOR_NAMES."""
    code = read_code(LLAMA_ARCH_CPP)
    m = re.search(r"LLM_TENSOR_NAMES\s*=\s*\{", code)
    assert m, "LLM_TENSOR_NAMES not found in src/llama-arch.cpp"
    body = balanced(code, code.index("{", m.start()), "{", "}")
    return dict(re.findall(r"\{\s*(LLM_TENSOR_\w+)\s*,\s*\"([^\"]+)\"\s*\}", body))


def gguf_name(base_names: dict, tensor: str, suffix: str, with_bid: bool) -> str:
    assert tensor in base_names, f"{tensor} is not in LLM_TENSOR_NAMES"
    return ("blk.%d." if with_bid else "") + base_names[tensor] + "." + suffix


def split_args(args: str) -> list:
    out, depth, cur, in_str = [], 0, [], False
    i = 0
    while i < len(args):
        c = args[i]
        if in_str:
            cur.append(c)
            if c == "\\":
                cur.append(args[i + 1])
                i += 1
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
            cur.append(c)
        elif c in "({[":
            depth += 1
            cur.append(c)
        elif c in ")}]":
            depth -= 1
            cur.append(c)
        elif c == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
        i += 1
    if cur:
        out.append("".join(cur).strip())
    return out


def tn_name(base_names: dict, expr: str) -> str:
    """tn(LLM_TENSOR_X, "suffix"[, bid]) or LLM_TN(ARCH)(LLM_TENSOR_X, "suffix"[, bid]).str()."""
    m = re.search(r"\(\s*(LLM_TENSOR_\w+)\s*,\s*\"([^\"]*)\"\s*(,[^)]*)?\)", expr)
    assert m, f"cannot read a tensor name from `{expr}`"
    return gguf_name(base_names, m.group(1), m.group(2), bool(m.group(3)))


def arch_of_class() -> dict:
    """model class -> set of archs, from llama_model_mapping()'s registration."""
    code = read_code(LLAMA_MODEL_CPP)
    body = function_body(code, r"static llama_model \* llama_model_mapping\s*\([^)]*\)")
    classes = {}
    pending = []
    for m in re.finditer(r"case\s+(LLM_ARCH_\w+)\s*:|return\s+new\s+(llama_model_\w+)\s*\(", body):
        if m.group(1):
            pending.append(m.group(1))
        else:
            classes.setdefault(m.group(2), set()).update(pending)
            pending = []
    return classes


def flagged_calls() -> list:
    """(arch, gguf name, file) for every create_tensor() call passing TENSOR_READ_LAZY."""
    base_names = tensor_base_names()
    classes = arch_of_class()
    found = []
    for path in sorted(MODELS_DIR.glob("*.cpp")):
        code = read_code(path)
        for m in re.finditer(r"\bcreate_tensor\s*\(", code):
            args = balanced(code, m.end() - 1)
            if "TENSOR_READ_LAZY" not in args:
                continue
            first = split_args(args)[0]
            name = tn_name(base_names, first)
            # the enclosing model class: the nearest `llama_model_<x>::` definition above the call
            owners = re.findall(r"\b(llama_model_\w+)::\w+\s*\(", code[: m.start()])
            assert owners, f"{path.name}: a flagged create_tensor() outside any llama_model_* method"
            cls = owners[-1]
            assert cls in classes, f"{path.name}: {cls} is not registered in llama_model_mapping()"
            for arch in classes[cls]:
                found.append((arch, name, path.name))
    return found


def declaration() -> dict:
    """arch -> set of declared names, from llama_model_lazy_tensor_names()."""
    base_names = tensor_base_names()
    code = read_code(LLAMA_MODEL_CPP)
    body = function_body(code, r"const std::vector<std::string>\s*&\s*llama_model_lazy_tensor_names\s*\([^)]*\)")
    variables = {}
    for m in re.finditer(r"static\s+const\s+std::vector<std::string>\s+(\w+)\s*(?:=\s*\{([^;]*)\})?\s*;", body):
        names = set()
        for entry in split_args(m.group(2) or ""):
            names.add(tn_name(base_names, entry) if "LLM_TENSOR_" in entry else entry.strip('"'))
        variables[m.group(1)] = names
    result = {}
    for m in re.finditer(r"case\s+(LLM_ARCH_\w+)\s*:\s*return\s+(\w+)\s*;", body):
        assert m.group(2) in variables, f"{m.group(2)} is returned but not defined in the declaration"
        result[m.group(1)] = variables[m.group(2)]
    # the `default:` arm answers for every other arch: it must name nothing, or it declares a
    # name for an arch whose create_tensor() calls do not flag it, and aborts that arch's load
    default = re.search(r"\bdefault\s*:\s*return\s+(\w+)\s*;", body)
    assert default, "llama_model_lazy_tensor_names() has no `default: return <vector>;` arm"
    assert default.group(1) in variables, f"the default arm returns {default.group(1)}, which is not defined in the declaration"
    assert not variables[default.group(1)], (
        f"the default arm returns {default.group(1)} = {sorted(variables[default.group(1)])}: it declares names for every other arch"
    )
    return result


def test_declaration_exists():
    decl = declaration()
    assert decl, "llama_model_lazy_tensor_names() declares no architecture"


def test_every_flagged_call_is_declared_under_its_arch():
    decl = declaration()
    missing = [(a, n, f) for a, n, f in flagged_calls() if n not in decl.get(a, set())]
    assert not missing, "flagged create_tensor() calls with no declaration entry under their arch: " + repr(missing)


def test_every_entry_has_a_flagged_call_in_its_arch():
    calls = {(a, n) for a, n, _ in flagged_calls()}
    stale = [(a, n) for a, names in declaration().items() for n in sorted(names) if (a, n) not in calls]
    assert not stale, "declaration entries with no flagged call in that arch's model: " + repr(stale)


def test_attrs_has_a_case_for_every_arch_with_names_and_no_other():
    code = read_code(LLAMA_MODEL_CPP)
    body = function_body(code, r"llama_lazy_tensor_attrs\s+llama_model_lazy_tensor_attrs\s*\([^)]*\)")
    cases = set(re.findall(r"case\s+(LLM_ARCH_\w+)\s*:", body))
    with_names = {a for a, names in declaration().items() if names}
    assert cases == with_names, f"attrs cases {sorted(cases)} != archs with names {sorted(with_names)}"


def test_declaration_returns_names_only_and_attrs_writes_only_its_return():
    code = read_code(LLAMA_MODEL_CPP)
    names_body = function_body(code, r"const std::vector<std::string>\s*&\s*llama_model_lazy_tensor_names\s*\([^)]*\)")
    assert not re.search(r"llama_lazy_tensor_attrs", names_body), "the declaration stores an hparams-valued field"
    attrs_body = function_body(code, r"llama_lazy_tensor_attrs\s+llama_model_lazy_tensor_attrs\s*\([^)]*\)")
    assert not re.search(r"\bstatic\b", attrs_body), (
        "llama_model_lazy_tensor_attrs() holds a static: the attributes are per model, read from the loaded hparams"
    )
