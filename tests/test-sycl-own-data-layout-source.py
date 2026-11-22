#!/usr/bin/env python3
"""get_tensor decides the order of the bytes it reads from extra->own_data_layout, which only the writers of those
bytes set.

ggml_tensor_extra_gpu::own_data_layout is the order of the bytes in the tensor's own range (tensor->data). The reorder
record (optimized_feature) and extra->layout say what a kernel last resolved, which may be a cache copy elsewhere, so
a reader that inferred the own range's order from them
could scramble correct bytes. With comments and string literals blanked:
  A. Under ggml/src/ggml-sycl (tests/ excluded), every own_data_layout = ... is at one of these sites, with exactly
     this right-hand side (the whole text up to ;), and nowhere else: the declaration in common.hpp (GGML_LAYOUT_AOS);
     in ggml_backend_sycl_buffer_set_tensor, GGML_LAYOUT_AOS, GGML_LAYOUT_SOA and GGML_LAYOUT_COALESCED once each; in
     ggml_backend_sycl_set_tensor_async, GGML_LAYOUT_AOS once; in reorder_tensor_to_soa, GGML_LAYOUT_SOA once; in
     convert_tensor_to_coalesced (mmvq.cpp), GGML_LAYOUT_COALESCED once.
  R. Every other occurrence of the token own_data_layout is one of an explicit list of forms: the declaration, a
     write spelled extra->own_data_layout = <value>; the reader's one read
     extra ? extra->own_data_layout : GGML_LAYOUT_AOS; and the read-only test seam's one copy
     records.own_range_layout = extra->own_data_layout; in ggml_sycl::test_get_tensor_layout_records. A reference, a
     pointer or any other read of the field fails.
  M. In set_tensor's "mark as reordered" block, the branch that calls mark_coalesced_pretransformed assigns
     COALESCED and the branch that calls mark_soa_pretransformed assigns SOA.
  K. convert_tensor_to_coalesced assigns it only under if (tensor_ptr == tensor->data): the conversion writes in place
     at the resolved pointer, which is the tensor's own range only when it equals tensor->data.
  B. ggml_sycl_update_layout_from_cache does not name own_data_layout: a cache-side copy never changes it.
  C. ggml_backend_sycl_buffer_get_tensor takes the record from the root tensor's extra
     (static_cast<const ggml_tensor_extra_gpu *>(root->extra)), never names tensor->extra, reads own_data_layout,
     names none of get_reorder(, layout.mode, layout.data_ptr, and calls nothing but an explicit list (the header's
     mapping, the stored-bytes read, the warn helper, ggml_nbytes, GGML_ASSERT, casts, std::min/max and the scratch
     vector's resize/data), so no helper can carry the decision out of sight.
  D. set_tensor's reset block (the if (extra) block that calls optimized_feature.reset_reorder) assigns
     own_data_layout = GGML_LAYOUT_AOS.
  G. ggml_backend_sycl_set_tensor_async assigns it only under if (extra != nullptr && dst_is_own_range), and
     dst_is_own_range is declared false and set only to true (where the tensor's own storage handle took the write)
     and to dst_ptr == tensor->data (where it did not): the resolved destination need not be the own range.

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
ASYNC_SIG = "static void ggml_backend_sycl_set_tensor_async("
SOA_SIG = "bool reorder_tensor_to_soa(const ggml_tensor * tensor, dpct::queue_ptr stream, const char * caller) {"
COAL_SIG = "bool convert_tensor_to_coalesced(const ggml_tensor * tensor, dpct::queue_ptr stream, const char * caller) {"
CACHE_SIG = "static void ggml_sycl_update_layout_from_cache("
GET_SIG = "static void ggml_backend_sycl_buffer_get_tensor("
SEAM_SIG = "test_tensor_layout_records test_get_tensor_layout_records(const ggml_tensor * tensor) {"
EXTRA_SIG = "struct ggml_tensor_extra_gpu {"

# A write: the token, =, and the whole right-hand side up to the ;.
ASSIGN = re.compile(r"\bown_data_layout\s*=(?!=)\s*([^;]*?)\s*;")
TOKEN = re.compile(r"\bown_data_layout\b")
DECL = re.compile(r"\bggml_layout_mode\s+own_data_layout\s*=\s*GGML_LAYOUT_AOS\s*;")
RESET = "extra->optimized_feature.reset_reorder(tensor->name);"
SET_AOS = "extra->own_data_layout = GGML_LAYOUT_AOS;"
MARK_COAL = "extra->optimized_feature.mark_coalesced_pretransformed(tensor->name);"
MARK_SOA = "extra->optimized_feature.mark_soa_pretransformed(tensor->name);"
COAL_GUARD = "if (tensor_ptr == tensor->data)"
ASYNC_GUARD = "if (extra != nullptr && dst_is_own_range)"
ASYNC_FLAG = re.compile(r"\bdst_is_own_range\s*=(?!=)\s*([^;]*?)\s*;")
ASYNC_FLAG_VALUES = ["dst_ptr == tensor->data", "false", "true"]
MEMSET_SIG = "static void ggml_backend_sycl_buffer_memset_tensor("
ROOT_EXTRA = "static_cast<const ggml_tensor_extra_gpu *>(root->extra)"
# The forms the token may appear in (R): the declaration, a write through extra->, and the reader's one read.
WRITE_FORM = re.compile(r"\bextra->own_data_layout\s*=(?!=)\s*[^;]*;")
READ_FORM = re.compile(r"\bextra\s*\?\s*extra->own_data_layout\s*:\s*GGML_LAYOUT_AOS\s*;")
SEAM_FORM = re.compile(r"\brecords\.own_range_layout\s*=\s*extra->own_data_layout\s*;")
# The calls the reader may make (C). Member calls are on the scratch vector only.
READER_CALLS = {"ggml_sycl_buffer_read_stored_bytes", "ggml_sycl_warn_stored_order_read",
                "ggml_sycl_soa_plane_table_for_type", "ggml_sycl_soa_to_aos_window", "ggml_nbytes", "GGML_ASSERT",
                "static_cast", "std::max", "std::min"}
READER_MEMBER_CALLS = {"resize", "data"}
KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof"}
CALL = re.compile(r"(?<![\w.>:])([A-Za-z_]\w*(?:::\w+)*)\s*(?:<[^;(){}]*?>)?\s*\(")
MEMBER_CALL = re.compile(r"(?:\.|->)\s*(\w+)\s*(?:<[^;(){}]*?>)?\s*\(")

# (file, signature, sorted right-hand sides) for every site allowed to assign own_data_layout.
SITES = (
    (COMMON, EXTRA_SIG, ["GGML_LAYOUT_AOS"]),
    (SYCL, SET_SIG, ["GGML_LAYOUT_AOS", "GGML_LAYOUT_COALESCED", "GGML_LAYOUT_SOA"]),
    (SYCL, ASYNC_SIG, ["GGML_LAYOUT_AOS"]),
    (SYCL, SOA_SIG, ["GGML_LAYOUT_SOA"]),
    (MMVQ, COAL_SIG, ["GGML_LAYOUT_COALESCED"]),
)

CA = "A own_data_layout is assigned only at the named sites, with the named values"
CR = "R the token appears only as the declaration, an extra-> write, the reader's one read or the seam's one copy"
CM = "M set_tensor's mark block assigns the layout it marks, per branch"
CK = "K the coalesced conversion assigns only when it wrote tensor->data"
CB = "B update_layout_from_cache does not name own_data_layout"
CC = "C get_tensor decides from own_data_layout and from no other record"
CD = "D set_tensor's reset block assigns AOS"
CG = "G the async write records only when it wrote the tensor's own range"


def blank_strings(text):
    return re.sub(r'"(?:\\.|[^"\\\n])*"', lambda m: '"' + " " * (len(m.group(0)) - 2) + '"', text)


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
            files[str(path.relative_to(SYCL_DIR))] = blank_strings(blank_comments(path.read_text(errors="replace")))
    return files


def census_ok(files):
    expected = 0
    for name, signature, values in SITES:
        inner = body(files.get(name, ""), signature)
        if sorted(" ".join(m.group(1).split()) for m in ASSIGN.finditer(inner)) != values:
            return False
        expected += len(values)
    total = sum(len(re.findall(r"\bown_data_layout\s*=(?!=)", text)) for text in files.values())
    return total == expected and len(DECL.findall(body(files.get(COMMON, ""), EXTRA_SIG))) == 1


def token_forms_ok(files):
    """Every occurrence of the token sits inside the declaration, an extra-> write, the reader's one read, or the
    test seam's one copy."""
    reads = 0
    seam_reads = 0
    for name, text in files.items():
        spans = [m.span() for m in WRITE_FORM.finditer(text)]
        if name == COMMON:
            spans += [m.span() for m in DECL.finditer(text)]
        reader = body_span(text, GET_SIG) if name == SYCL else None
        if reader:
            for m in READ_FORM.finditer(text, reader[0], reader[1]):
                spans.append(m.span())
                reads += 1
        seam = body_span(text, SEAM_SIG) if name == SYCL else None
        if seam:
            for m in SEAM_FORM.finditer(text, seam[0], seam[1]):
                spans.append(m.span())
                seam_reads += 1
        for tok in TOKEN.finditer(text):
            if sum(a <= tok.start() < b for a, b in spans) != 1:
                return False
    return reads == 1 and seam_reads == 1


def reader_calls_ok(get_body):
    free = [m.group(1) for m in CALL.finditer(get_body)]
    member = [m.group(1) for m in MEMBER_CALL.finditer(get_body)]
    return get_body != "" and all(name in READER_CALLS or name in KEYWORDS for name in free) and \
        all(name in READER_MEMBER_CALLS for name in member)


def async_guard_ok(async_body):
    writes = list(re.finditer(r"\bown_data_layout\s*=(?!=)", async_body))
    flags = sorted(" ".join(m.group(1).split()) for m in ASYNC_FLAG.finditer(async_body))
    return len(writes) >= 1 and all(guarded_by(async_body, m.start(), ASYNC_GUARD) for m in writes) and \
        flags == ASYNC_FLAG_VALUES and re.search(r"\bbool\s+dst_is_own_range\s*=\s*false\s*;", async_body) is not None


def claims(files):
    sycl = files.get(SYCL, "")
    set_body = body(sycl, SET_SIG)
    coal_branch = enclosing_block(set_body, set_body.find(MARK_COAL)) if MARK_COAL in set_body else ""
    soa_branch = enclosing_block(set_body, set_body.find(MARK_SOA)) if MARK_SOA in set_body else ""
    reset_block = enclosing_block(set_body, set_body.find(RESET)) if RESET in set_body else ""
    coal = body(files.get(MMVQ, ""), COAL_SIG)
    coal_assign = list(re.finditer(r"\bown_data_layout\s*=(?!=)", coal))
    get_body = body(sycl, GET_SIG)
    return {
        CA: census_ok(files),
        CR: token_forms_ok(files),
        CM: re.search(r"\bown_data_layout\s*=\s*GGML_LAYOUT_COALESCED\s*;", coal_branch) is not None and
            re.search(r"\bown_data_layout\s*=\s*GGML_LAYOUT_SOA\s*;", soa_branch) is not None and
            [m.group(1) for m in ASSIGN.finditer(coal_branch)] == ["GGML_LAYOUT_COALESCED"] and
            [m.group(1) for m in ASSIGN.finditer(soa_branch)] == ["GGML_LAYOUT_SOA"],
        CK: len(coal_assign) >= 1 and all(guarded_by(coal, m.start(), COAL_GUARD) for m in coal_assign),
        CB: (lambda b: b != "" and "own_data_layout" not in b)(body(sycl, CACHE_SIG)),
        CC: get_body.count(ROOT_EXTRA) == 1 and "tensor->extra" not in get_body and
            READ_FORM.search(get_body) is not None and
            re.search(r"get_reorder\(|layout\.mode|layout\.data_ptr", get_body) is None and reader_calls_ok(get_body),
        CD: SET_AOS in reset_block and RESET in reset_block,
        CG: async_guard_ok(body(sycl, ASYNC_SIG)),
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
        ("a write whose right-hand side starts with a cast, in memset_tensor", CA,
         s(MEMSET_SIG, r"^\{", "{ static_cast<ggml_tensor_extra_gpu *>(tensor->extra)->own_data_layout = "
           "(ggml_layout_mode) 1;")),
        ("a write through a reference", CR,
         s(SET_SIG, re.escape(SET_AOS), "auto & rec = extra->own_data_layout; rec = GGML_LAYOUT_AOS;")),
        ("a pointer taken to the record", CR,
         s(ASYNC_SIG, r"\bextra->own_data_layout = GGML_LAYOUT_AOS;",
           "extra->own_data_layout = GGML_LAYOUT_AOS; (void) &extra->own_data_layout;")),
        ("the test seam takes a reference to the record", CR,
         s(SEAM_SIG, SEAM_FORM.pattern, "const auto & own = extra->own_data_layout; records.own_range_layout = own;")),
        ("the reader takes the view's extra", CC, s(GET_SIG, r"\(root->extra\)", "(tensor->extra)")),
        ("the reader's decision moved into a helper", CC,
         s(GET_SIG, READ_FORM.pattern, "ggml_sycl_stored_layout_of(root);")),
        ("the async record no longer guarded by the own range", CG,
         s(ASYNC_SIG, re.escape(ASYNC_GUARD), "if (extra != nullptr)")),
        ("the async own-range flag set without the pointer check", CG,
         s(ASYNC_SIG, r"\bdst_is_own_range\s*=\s*dst_ptr == tensor->data\s*;", "dst_is_own_range = true;")),
        ("the reset block records SOA", CD,
         s(SET_SIG, re.escape(SET_AOS), "extra->own_data_layout = GGML_LAYOUT_SOA;")),
        ("the reset assignment moved out of the reset block", CD,
         mutate(s(SET_SIG, re.escape(SET_AOS), ""), SYCL, SET_SIG,
                r"\bif \(is_weight_buffer && offset == 0 && size > 0\)",
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
