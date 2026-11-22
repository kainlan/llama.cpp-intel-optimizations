#!/usr/bin/env python3
"""get_tensor decides the order of the bytes it reads from extra->own_data_layout, which only the writers of those
bytes set.

ggml_tensor_extra_gpu::own_data_layout is the order of the bytes in the tensor's own range (tensor->data). The reorder
record (optimized_feature) and extra->layout say what a kernel last resolved, which may be a cache copy elsewhere, so
a reader that inferred the own range's order from them could scramble correct bytes. With comments blanked:
  A. Under ggml/src/ggml-sycl (tests/ excluded), own_data_layout is assigned exactly at these sites and nowhere else:
     the declaration in common.hpp (= GGML_LAYOUT_AOS); in ggml_backend_sycl_buffer_set_tensor, AOS once, SOA once and
     COALESCED once; in reorder_tensor_to_soa, SOA once; in convert_tensor_to_coalesced (mmvq.cpp), COALESCED once.
  M. In set_tensor's "mark as reordered" block, the branch that calls mark_coalesced_pretransformed assigns
     COALESCED and the branch that calls mark_soa_pretransformed assigns SOA.
  K. convert_tensor_to_coalesced assigns it only under if (tensor_ptr == tensor->data): the conversion writes in place
     at the resolved pointer, which is the tensor's own range only when it equals tensor->data.
  B. ggml_sycl_update_layout_from_cache does not name own_data_layout: a cache-side copy never changes it.
  C. ggml_backend_sycl_buffer_get_tensor reads own_data_layout and names none of get_reorder(, layout.mode,
     layout.data_ptr.
  D. set_tensor's reset block (the if (extra) block that calls optimized_feature.reset_reorder) assigns
     own_data_layout = GGML_LAYOUT_AOS.

--self-test applies at least one mutant per claim and requires each to fail the claim it targets; a mutant whose
target text is not found fails the self-test.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL_DIR = ROOT / "ggml/src/ggml-sycl"
SYCL = "ggml-sycl.cpp"
MMVQ = "mmvq.cpp"
COMMON = "common.hpp"

SET_SIG = "static void ggml_backend_sycl_buffer_set_tensor("
SOA_SIG = "bool reorder_tensor_to_soa(const ggml_tensor * tensor, dpct::queue_ptr stream, const char * caller) {"
COAL_SIG = "bool convert_tensor_to_coalesced(const ggml_tensor * tensor, dpct::queue_ptr stream, const char * caller) {"
CACHE_SIG = "static void ggml_sycl_update_layout_from_cache("
GET_SIG = "static void ggml_backend_sycl_buffer_get_tensor("
EXTRA_SIG = "struct ggml_tensor_extra_gpu {"

ASSIGN = re.compile(r"\bown_data_layout\s*=(?!=)\s*([A-Za-z_][A-Za-z_0-9:]*)")
DECL = re.compile(r"\bggml_layout_mode\s+own_data_layout\s*=\s*GGML_LAYOUT_AOS\s*;")
RESET = "extra->optimized_feature.reset_reorder(tensor->name);"
SET_AOS = "extra->own_data_layout = GGML_LAYOUT_AOS;"
MARK_COAL = "extra->optimized_feature.mark_coalesced_pretransformed(tensor->name);"
MARK_SOA = "extra->optimized_feature.mark_soa_pretransformed(tensor->name);"
COAL_GUARD = "if (tensor_ptr == tensor->data)"

# (file, signature, sorted right-hand sides) for every site allowed to assign own_data_layout.
SITES = (
    (COMMON, EXTRA_SIG, ["GGML_LAYOUT_AOS"]),
    (SYCL, SET_SIG, ["GGML_LAYOUT_AOS", "GGML_LAYOUT_COALESCED", "GGML_LAYOUT_SOA"]),
    (SYCL, SOA_SIG, ["GGML_LAYOUT_SOA"]),
    (MMVQ, COAL_SIG, ["GGML_LAYOUT_COALESCED"]),
)

CA = "A own_data_layout is assigned only at the named sites, with the named values"
CM = "M set_tensor's mark block assigns the layout it marks, per branch"
CK = "K the coalesced conversion assigns only when it wrote tensor->data"
CB = "B update_layout_from_cache does not name own_data_layout"
CC = "C get_tensor decides from own_data_layout and from no other record"
CD = "D set_tensor's reset block assigns AOS"


def blank_comments(text):
    text = re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), text)
    return re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)


def block_end(text, open_at):
    """Index one past the brace that closes the block opened at `open_at`, or -1."""
    depth = 0
    for i in range(open_at, len(text)):
        depth += text[i] == "{"
        depth -= text[i] == "}"
        if depth == 0:
            return i + 1
    return -1


def body_span(text, signature):
    """(start, end) of the brace-balanced body after the first occurrence of `signature`, or None."""
    start = text.find(signature)
    if start < 0:
        return None
    open_at = text.find("{", start + len(signature) - 1)
    end = block_end(text, open_at) if open_at >= 0 else -1
    return (open_at, end) if end >= 0 else None


def body(text, signature):
    span = body_span(text, signature)
    return text[span[0]:span[1]] if span else ""


def enclosing_block(text, at):
    """The innermost brace block that contains index `at`, or ''."""
    depth = 0
    for i in range(at, -1, -1):
        depth += text[i] == "}"
        depth -= text[i] == "{"
        if depth < 0:
            end = block_end(text, i)
            return text[i:end] if end >= 0 else ""
    return ""


def guarded_by(text, at, guard):
    """True when the innermost if before `at` is `guard` and the block it opens contains `at`."""
    if_at = text.rfind("if (", 0, at)
    if if_at < 0 or not text.startswith(guard, if_at):
        return False
    open_at = text.find("{", if_at)
    return 0 <= open_at < at < block_end(text, open_at)


def load():
    files = {}
    for path in sorted(SYCL_DIR.rglob("*")):
        if path.suffix in (".cpp", ".hpp", ".h") and "tests" not in path.relative_to(SYCL_DIR).parts:
            files[str(path.relative_to(SYCL_DIR))] = blank_comments(path.read_text(errors="replace"))
    return files


def census_ok(files):
    expected = 0
    for name, signature, values in SITES:
        text = files.get(name, "")
        inner = body(text, signature)
        if sorted(m.group(1) for m in ASSIGN.finditer(inner)) != values:
            return False
        expected += len(values)
    total = sum(len(ASSIGN.findall(text)) for text in files.values())
    return total == expected and len(DECL.findall(body(files.get(COMMON, ""), EXTRA_SIG))) == 1


def claims(files):
    sycl = files.get(SYCL, "")
    set_body = body(sycl, SET_SIG)
    coal_branch = enclosing_block(set_body, set_body.find(MARK_COAL)) if MARK_COAL in set_body else ""
    soa_branch = enclosing_block(set_body, set_body.find(MARK_SOA)) if MARK_SOA in set_body else ""
    reset_block = enclosing_block(set_body, set_body.find(RESET)) if RESET in set_body else ""
    coal = body(files.get(MMVQ, ""), COAL_SIG)
    coal_assign = list(ASSIGN.finditer(coal))
    get_body = body(sycl, GET_SIG)
    return {
        CA: census_ok(files),
        CM: re.search(r"\bown_data_layout\s*=\s*GGML_LAYOUT_COALESCED\s*;", coal_branch) is not None and
            re.search(r"\bown_data_layout\s*=\s*GGML_LAYOUT_SOA\s*;", soa_branch) is not None and
            "GGML_LAYOUT_SOA" not in "".join(m.group(1) for m in ASSIGN.finditer(coal_branch)) and
            "GGML_LAYOUT_COALESCED" not in "".join(m.group(1) for m in ASSIGN.finditer(soa_branch)),
        CK: len(coal_assign) >= 1 and all(guarded_by(coal, m.start(), COAL_GUARD) for m in coal_assign),
        CB: (lambda b: b != "" and "own_data_layout" not in b)(body(sycl, CACHE_SIG)),
        CC: "own_data_layout" in get_body and
            re.search(r"get_reorder\(|layout\.mode|layout\.data_ptr", get_body) is None,
        CD: SET_AOS in reset_block and RESET in reset_block,
    }


def check(files):
    return [name for name, ok in claims(files).items() if not ok]


def mutate(files, name, signature, pattern, repl, nth=0):
    """`files` with the `nth` match of `pattern` inside the body after `signature` in `name` replaced by `repl`."""
    text = files.get(name, "")
    span = body_span(text, signature)
    if span is None:
        return files
    inner = text[span[0]:span[1]]
    matches = list(re.finditer(pattern, inner))
    if len(matches) <= nth:
        return files
    hit = matches[nth]
    out = dict(files)
    out[name] = text[:span[0]] + inner[:hit.start()] + repl + inner[hit.end():] + text[span[1]:]
    return out


def mutants(files):
    """(name, the claim it must fail, mutated files)."""
    s = lambda sig, pattern, repl, nth=0: mutate(files, SYCL, sig, pattern, repl, nth)
    m = lambda pattern, repl, nth=0: mutate(files, MMVQ, COAL_SIG, pattern, repl, nth)
    soa_assign = r"\bown_data_layout\s*=\s*GGML_LAYOUT_SOA\s*;"
    coal_assign = r"\bown_data_layout\s*=\s*GGML_LAYOUT_COALESCED\s*;"
    return [
        ("reorder_tensor_to_soa no longer records SOA", CA, s(SOA_SIG, soa_assign, "(void) 0;")),
        ("a stray writer in get_tensor", CA,
         s(GET_SIG, r"\bif \(size == 0", "const_cast<ggml_tensor_extra_gpu *>(extra)->own_data_layout = "
           "GGML_LAYOUT_AOS; if (size == 0")),
        ("the coalesced conversion records SOA", CA, m(coal_assign, "own_data_layout = GGML_LAYOUT_SOA;")),
        ("set_tensor's mark branches swapped", CM, s(SET_SIG, soa_assign, "own_data_layout = GGML_LAYOUT_COALESCED;")),
        ("the coalesced record no longer guarded by the own range", CK,
         m(re.escape(COAL_GUARD), "if (tensor_ptr != nullptr)")),
        ("update_layout_from_cache writes the record", CB,
         s(CACHE_SIG, r"extra->layout\.mode\s*=\s*layout_mode;",
           "extra->layout.mode = layout_mode; extra->own_data_layout = layout_mode;")),
        ("the reader infers from the reorder record", CC,
         s(GET_SIG, r"extra->own_data_layout", "(extra->optimized_feature.get_reorder() == reorder_mode::SOA ? "
           "GGML_LAYOUT_SOA : extra->own_data_layout)")),
        ("the reader infers from layout.mode", CC, s(GET_SIG, r"extra->own_data_layout", "extra->layout.mode")),
        ("the reader ignores the record", CC, s(GET_SIG, r"extra->own_data_layout", "GGML_LAYOUT_AOS")),
        ("the reset block records SOA", CD, s(SET_SIG, re.escape(SET_AOS), "extra->own_data_layout = GGML_LAYOUT_SOA;")),
        ("the reset assignment moved out of the reset block", CD,
         mutate(s(SET_SIG, re.escape(SET_AOS), ""), SYCL, SET_SIG, r"\bif \(is_weight_buffer && offset == 0 && size > 0\)",
                SET_AOS + " if (is_weight_buffer && offset == 0 && size > 0)")),
    ]


def main():
    files = load()
    failed = check(files)
    if failed:
        print("own data layout source contract failed: " + ", ".join(failed), file=sys.stderr)
        return 1
    if "--self-test" in sys.argv:
        for name, target, mutated in mutants(files):
            if mutated == files:
                print(f"own data layout source contract: mutant '{name}' did not apply", file=sys.stderr)
                return 1
            caught = check(mutated)
            if target not in caught:
                print(f"own data layout source contract: mutant '{name}' was not caught by '{target}' "
                      f"(failed: {caught})", file=sys.stderr)
                return 1
            print(f"own data layout source contract: mutant '{name}' caught by '{target}'")
    print("own data layout source contract: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
