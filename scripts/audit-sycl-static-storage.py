#!/usr/bin/env python3
"""Generate the parser-grade SYCL static-storage census.

Pinned parser dependencies:
  tree-sitter==0.25.2
  tree-sitter-language-pack==1.8.1 (C++ grammar ABI 15)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from collections import Counter
from importlib.metadata import version
from io import StringIO
from pathlib import Path

PARSER_PACK_VERSION = "1.8.1"
TREE_SITTER_VERSION = "0.25.2"
FILES = (
    "ggml/src/ggml-sycl/ggml-sycl.cpp",
    "ggml/src/ggml-sycl/unified-cache.cpp",
    "ggml/src/ggml-sycl/unified-cache.hpp",
    "ggml/src/ggml-sycl/fattn.cpp",
    "ggml/src/ggml-sycl/layer-streaming.cpp",
)
COLUMNS = (
    "file", "line", "symbol", "type", "scope", "mutability", "synchronization",
    "writer_evidence", "reader_evidence", "owner_identity", "reset_teardown_disposition",
)
DECL_KINDS = {"declaration", "field_declaration"}
DECLARATOR_KINDS = {
    "identifier", "field_identifier", "init_declarator", "pointer_declarator",
    "reference_declarator", "array_declarator", "parenthesized_declarator",
    "qualified_identifier", "attributed_declarator",
}
FUNCTIONISH = {"function_declarator", "function_definition"}
SCOPE_KINDS = {
    "function_definition", "lambda_expression", "class_specifier", "struct_specifier",
    "union_specifier", "namespace_definition", "ERROR",
}
WRITE_RE = re.compile(
    r"(?:\b{n}\s*(?:\[[^\n;]*\])?\s*(?:=(?!=)|\+=|-=|\+\+|--)|"
    r"\b{n}\s*(?:\[[^\n;]*\])?\s*\.\s*(?:(?:try_)?emplace|insert(?:_or_assign)?|"
    r"clear|erase|push_\w*|pop_\w*|resize|assign|reset|store|exchange|fetch_\w*|swap|release|acquire)\s*\()"
)
RESET_RE = re.compile(
    r"\b{n}\s*(?:\[[^\n;]*\])?\s*\.\s*(?:clear|erase|reset|release|swap)\s*\(|"
    r"\b{n}\s*(?:\[[^\n;]*\])?\s*\.\s*store\s*\(\s*(?:false|0|nullptr)|"
    r"\b(?:reset|free|destroy|unregister)\w*\s*\([^;\n]*\b{n}\b", re.I,
)


def call(obj, name, *args):
    value = getattr(obj, name)
    return value(*args) if callable(value) else value


def kind(node): return call(node, "kind") if hasattr(node, "kind") else call(node, "type")
def start_byte(node): return call(node, "start_byte")
def end_byte(node): return call(node, "end_byte")
def child_count(node): return call(node, "child_count")
def child(node, i): return call(node, "child", i)
def parent(node): return call(node, "parent")
def is_missing(node): return call(node, "is_missing")
def field(node, name): return call(node, "child_by_field_name", name)


def children(node):
    return [child(node, i) for i in range(child_count(node))]


def walk(node):
    yield node
    for item in children(node):
        yield from walk(item)


def ancestors(node):
    item = parent(node)
    while item is not None:
        yield item
        item = parent(item)


def text(source_b: bytes, node) -> str:
    return source_b[start_byte(node):end_byte(node)].decode("utf-8", "replace")


def line_of(source_b: bytes, node_or_offset) -> int:
    offset = node_or_offset if isinstance(node_or_offset, int) else start_byte(node_or_offset)
    return source_b.count(b"\n", 0, offset) + 1


def contains_kind(node, kinds: set[str]) -> bool:
    return any(kind(item) in kinds for item in walk(node))


def declarator_name(source_b: bytes, node):
    if node is None:
        return None
    if kind(node) in {"identifier", "field_identifier"}:
        return text(source_b, node), node
    nested = field(node, "declarator")
    if nested is not None:
        return declarator_name(source_b, nested)
    if kind(node) == "qualified_identifier":
        names = [item for item in children(node) if kind(item) in {"identifier", "field_identifier"}]
        return (text(source_b, names[-1]), names[-1]) if names else None
    for item in children(node):
        if kind(item) in DECLARATOR_KINDS:
            result = declarator_name(source_b, item)
            if result:
                return result
    return None


def recovered_function_name(source_b: bytes, error_node):
    head = text(source_b, error_node)[:700]
    match = re.match(
        r"(?s)\s*(?:[A-Z_]\w*\s+|static\s+|inline\s+|const\s+)*"
        r"[\w:<>*& ]+?\s+(\w+)\s*\([^;{}]*\)\s*(?:try\s*)?\{", head,
    )
    return match.group(1) if match else None


def named_scope(source_b: bytes, decl, recovered_regions):
    """Return the nearest lexical storage scope; methods beat their class."""
    for item in ancestors(decl):
        item_kind = kind(item)
        if item_kind == "lambda_expression":
            return "function-local:<lambda>", True
        if item_kind == "function_definition":
            result = declarator_name(source_b, field(item, "declarator"))
            return "function-local:" + (result[0] if result else "<function>"), True
        if item_kind == "ERROR":
            region = recovered_regions.get((start_byte(item), end_byte(item)))
            if region and start_byte(decl) < region[2]:
                return "function-local:" + region[0], True
        if item_kind in {"class_specifier", "struct_specifier", "union_specifier"}:
            name = field(item, "name")
            return "class:" + (text(source_b, name) if name else "<anonymous>"), False
        if item_kind == "namespace_definition":
            name = field(item, "name")
            return ("namespace:" + text(source_b, name), False) if name else ("anonymous-namespace", False)
    return "file", False


def storage_specifiers(source_b: bytes, decl):
    return {
        text(source_b, item).strip()
        for item in children(decl)
        if kind(item) == "storage_class_specifier"
        and text(source_b, item).strip() in {"static", "thread_local", "extern"}
    }


def direct_qualifiers(source_b: bytes, node):
    return {
        text(source_b, item).strip()
        for item in children(node)
        if kind(item) == "type_qualifier"
    }


def declarator_core(node):
    return field(node, "declarator") if kind(node) == "init_declarator" else node


def is_top_level_immutable(source_b: bytes, decl, declarator):
    """Distinguish const object/pointer from const pointee/template argument."""
    declaration_qualifiers = direct_qualifiers(source_b, decl)
    core = declarator_core(declarator)
    if "constexpr" in declaration_qualifiers:
        return True
    if kind(core) == "pointer_declarator":
        return "const" in direct_qualifiers(source_b, core)
    if kind(core) == "reference_declarator":
        return True  # the binding cannot be reseated; pointee mutability is not binding mutability
    return "const" in declaration_qualifiers


def base_type(source_b: bytes, decl, first_declarator):
    pieces = []
    for item in children(decl):
        if start_byte(item) >= start_byte(first_declarator):
            break
        if kind(item) not in {"storage_class_specifier", "attribute_specifier", "attribute_declaration"}:
            value = " ".join(text(source_b, item).split())
            if value and value not in {",", ";"}:
                pieces.append(value)
    return " ".join(pieces) or "<parser-unresolved>"


def initializer_free_type(source_b: bytes, decl, first_declarator, declarator, name_node):
    base = base_type(source_b, decl, first_declarator)
    core = declarator_core(declarator)
    shape = text(source_b, core)
    before = " ".join(shape[:start_byte(name_node) - start_byte(core)].split())
    after = " ".join(shape[end_byte(name_node) - start_byte(core):].split())
    return " ".join(part for part in (base, before + after) if part).strip()


def declaration_rows(source_b: bytes, decl, recovered_regions):
    scope, local = named_scope(source_b, decl, recovered_regions)
    storage = storage_specifiers(source_b, decl)
    if scope.startswith("class:") and "static" not in storage:
        return []
    if local and not ({"static", "thread_local"} & storage):
        return []

    type_node = field(decl, "type")
    type_end = end_byte(type_node) if type_node else start_byte(decl)
    declarators = [
        item for item in children(decl)
        if kind(item) in DECLARATOR_KINDS and start_byte(item) >= type_end
    ]
    rows = []
    for declarator in declarators:
        core = declarator_core(declarator)
        if contains_kind(core, FUNCTIONISH):
            continue
        result = declarator_name(source_b, core)
        if not result:
            continue
        name, name_node = result
        initialized = kind(declarator) == "init_declarator"
        if "extern" in storage and not initialized and not scope.startswith("class:"):
            continue  # declaration only; no storage definition in this translation unit
        rows.append({
            "name": name,
            "type": initializer_free_type(source_b, decl, declarators[0], declarator, name_node),
            "scope": scope,
            "storage": storage,
            "name_node": name_node,
            "decl_line": line_of(source_b, decl),
            "immutable": is_top_level_immutable(source_b, decl, declarator),
        })
    return rows


def code_mask(source_b: bytes):
    """Mask comments and quoted literals while preserving byte offsets/newlines."""
    out = bytearray(source_b)
    i, state, quote = 0, "code", 0
    while i < len(source_b):
        if state == "code" and source_b[i:i + 2] == b"//":
            out[i:i + 2] = b"  "; i += 2; state = "line"; continue
        if state == "code" and source_b[i:i + 2] == b"/*":
            out[i:i + 2] = b"  "; i += 2; state = "block"; continue
        if state == "code" and source_b[i] in (34, 39):
            quote = source_b[i]; out[i] = 32; i += 1; state = "quote"; continue
        if state == "line":
            if source_b[i] == 10: state = "code"
            else: out[i] = 32
            i += 1; continue
        if state == "block":
            if source_b[i:i + 2] == b"*/": out[i:i + 2] = b"  "; i += 2; state = "code"
            else:
                if source_b[i] != 10: out[i] = 32
                i += 1
            continue
        if state == "quote":
            if source_b[i] == 92 and i + 1 < len(source_b):
                if source_b[i] != 10: out[i] = 32
                if source_b[i + 1] != 10: out[i + 1] = 32
                i += 2; continue
            if source_b[i] == quote: out[i] = 32; i += 1; state = "code"; continue
            if source_b[i] != 10: out[i] = 32
            i += 1; continue
        i += 1
    return bytes(out)


def balanced_body_end(masked: bytes, body_start: int, limit: int):
    """Return the first lexically balanced closing brace, if one exists."""
    depth = 0
    for pos in range(body_start, limit):
        if masked[pos] == ord("{"):
            depth += 1
        elif masked[pos] == ord("}"):
            depth -= 1
            if depth == 0:
                return pos + 1
    return None


def recovered_function_regions(source_b: bytes, gaps):
    """Map recovered ERROR functions to (name, body start, balanced body end)."""
    masked = code_mask(source_b)
    regions = {}
    for item in gaps:
        if kind(item) != "ERROR":
            continue
        name = recovered_function_name(source_b, item)
        if not name:
            continue
        body_start = source_b.find(b"{", start_byte(item), end_byte(item))
        if body_start < 0:
            continue
        body_end = balanced_body_end(masked, body_start, end_byte(item))
        if body_end is not None:
            regions[(start_byte(item), end_byte(item))] = (name, body_start, body_end)
    return regions


def recovered_function_ancestor(node, recovered_regions):
    candidates = [node, *ancestors(node)]
    for item in candidates:
        region = recovered_regions.get((start_byte(item), end_byte(item)))
        if region and start_byte(node) >= start_byte(item) and end_byte(node) <= region[2]:
            return item
    return None


def parsed_function_context(node):
    candidates = [node, *ancestors(node)]
    return next((item for item in candidates if kind(item) in {"function_definition", "lambda_expression"}), None)


def is_preprocessor_condition_recovery(source_b: bytes, gap):
    """Accept only recovery confined to a #if/#elif condition line, never its body."""
    if not any(kind(item) in {"preproc_if", "preproc_elif"} for item in ancestors(gap)):
        return False
    line_start = source_b.rfind(b"\n", 0, start_byte(gap)) + 1
    line_end = source_b.find(b"\n", start_byte(gap))
    if line_end < 0:
        line_end = len(source_b)
    line = source_b[line_start:line_end]
    return end_byte(gap) <= line_end and re.match(rb"\s*#\s*(?:if|elif)\b", line) is not None


def recovery_tail_is_structurally_parsed(masked: bytes, container, body_end: int):
    """Prove an oversized recovery function's tail is parsed top-level syntax."""
    safe = {
        "comment", ";", "declaration", "function_definition", "template_declaration",
        "namespace_definition", "linkage_specification", "class_specifier", "struct_specifier",
        "union_specifier", "enum_specifier", "type_definition", "alias_declaration",
        "static_assert_declaration", "call_expression",
    }
    cursor = body_end
    for item in children(container):
        if end_byte(item) <= body_end:
            continue
        if start_byte(item) < body_end or masked[cursor:start_byte(item)].strip():
            return False
        item_kind = kind(item)
        if item_kind not in safe and not item_kind.startswith("preproc_"):
            return False
        cursor = end_byte(item)
    return not masked[cursor:end_byte(container)].strip()


def recovery_coverage(source_b: bytes, root, declarations, recovered_regions):
    """Prove every recovery site unable to hide a census declaration, or fail."""
    gaps = [item for item in walk(root) if kind(item) == "ERROR" or is_missing(item)]
    declaration_spans = [(start_byte(item), end_byte(item)) for item in declarations]

    # Exempt storage markers only in actual function signatures. A whole parsed
    # or recovered function body is deliberately not an exemption: every marker
    # there must still belong to a parsed declaration. Nested function_definition
    # nodes are invalid C++ and can be recovery's misparse of a malformed local.
    signature_spans = []
    for item in walk(root):
        if kind(item) not in {"function_definition", "lambda_expression"}:
            continue
        if any(kind(ancestor) in {"function_definition", "lambda_expression"} for ancestor in ancestors(item)):
            continue
        body = field(item, "body")
        if body is not None:
            signature_spans.append((start_byte(item), start_byte(body)))
    recovered_functions = [
        item for item in gaps
        if (start_byte(item), end_byte(item)) in recovered_regions
        and not any(
            kind(ancestor) == "ERROR"
            and (start_byte(ancestor), end_byte(ancestor)) in recovered_regions
            for ancestor in ancestors(item)
        )
    ]
    for item in recovered_functions:
        _, body_start, _ = recovered_regions[(start_byte(item), end_byte(item))]
        signature_spans.append((start_byte(item), body_start))

    failures = []
    masked = code_mask(source_b)

    # A parsed function or lambda can still have a recovery-expanded compound_statement:
    # lexically balance its real body and reject any parser-owned tail that is
    # not independently structured. Otherwise a following file-scope object
    # can be swallowed as a non-static local and silently disappear.
    checked_parsed_bodies = set()
    for gap in gaps:
        function = parsed_function_context(gap)
        if function is None:
            continue
        for body in (item for item in ancestors(gap) if kind(item) == "compound_statement"):
            body_span = (start_byte(body), end_byte(body))
            if body_span in checked_parsed_bodies:
                continue
            checked_parsed_bodies.add(body_span)
            body_end = balanced_body_end(masked, start_byte(body), end_byte(body))
            if body_end is None:
                failures.append((line_of(source_b, body), "unproved recovery function body"))
            elif body_end < end_byte(body) and not recovery_tail_is_structurally_parsed(masked, body, body_end):
                failures.append((line_of(source_b, body_end), "unproved recovery tail after function body"))

    for match in re.finditer(rb"\b(?:static|thread_local)\b", masked):
        pos = match.start()
        if any(begin <= pos < end for begin, end in declaration_spans):
            continue
        # A function's signature-level storage marker is not an object declaration.
        if any(begin <= pos < end for begin, end in signature_spans):
            continue
        failures.append((line_of(source_b, pos), "unparsed storage-duration marker"))

    categories = Counter()
    for gap in gaps:
        own_region = recovered_regions.get((start_byte(gap), end_byte(gap)))
        if own_region is not None:
            categories["function-body-storage-proven"] += 1
            if own_region[2] < end_byte(gap) and not recovery_tail_is_structurally_parsed(
                masked, gap, own_region[2]
            ):
                failures.append((line_of(source_b, own_region[2]), "unproved recovery tail after function body"))
            continue
        function = parsed_function_context(gap)
        recovered = recovered_function_ancestor(gap, recovered_regions)
        if function is not None or recovered is not None:
            body = field(function, "body") if function is not None else None
            if body is not None and end_byte(gap) <= start_byte(body):
                categories["function-signature-non-storage"] += 1
            else:
                # Body recovery is accepted only alongside the marker scan above:
                # all static/thread_local spellings must be parsed declarations.
                categories["function-body-storage-proven"] += 1
        elif is_preprocessor_condition_recovery(source_b, gap):
            categories["preprocessor-condition-nondeclaration"] += 1
        else:
            # Namespace/file or structural-preprocessor recovery can hide an
            # implicit-static object of any user type or initializer spelling.
            failures.append((line_of(source_b, gap), f"unproved {kind(gap)} recovery at namespace/file scope"))
    return gaps, categories, failures


def lexical_evidence(lines, name, decl_line):
    """Return explicitly unscoped candidates; never claim C++ binding resolution."""
    token = re.compile(r"\b" + re.escape(name) + r"\b")
    writer = re.compile(WRITE_RE.pattern.format(n=re.escape(name)))
    reset = re.compile(RESET_RE.pattern.format(n=re.escape(name)), re.I)
    writes, reads, resets = [], [], []
    for number, line in enumerate(lines, 1):
        if number == decl_line or not token.search(line):
            continue
        excerpt = f"L{number}:{' '.join(line.strip().split())[:120]}"
        target = writes if writer.search(line) else reads
        if len(target) < 3: target.append(excerpt)
        if reset.search(line) and len(resets) < 3: resets.append(excerpt)
    prefix = "unscoped lexical candidate: "
    return (
        prefix + " | ".join(writes) if writes else "none found by unscoped lexical scan",
        prefix + " | ".join(reads) if reads else "none found by unscoped lexical scan",
        resets,
    )


def classify(row, reset_candidates):
    immutable = row["immutable"]
    typ, scope, storage, name = row["type"], row["scope"], row["storage"], row["name"]
    if "thread_local" in storage:
        synchronization = "thread-local"
    elif re.search(r"\b(atomic|mutex|once_flag|semaphore)\b", typ):
        synchronization = "intrinsic:" + ("atomic" if "atomic" in typ else "lock primitive")
    else:
        synchronization = "none evident"
    lower = (name + " " + typ).lower()
    if scope.startswith("class:"): owner = scope
    elif "thread_local" in storage: owner = "thread"
    elif re.search(r"tensor|host_ptr|k_base|alloc", lower): owner = "tensor/allocation identity or registry"
    elif re.search(r"device|gpu|queue|stream", lower): owner = "device/process slot"
    elif re.search(r"model|layer|expert|moe|weight|kv", lower): owner = "model/layer semantic state (review required)"
    else: owner = "process or function lifetime (review required)"
    if immutable:
        disposition = "not applicable: immutable binding"
    elif reset_candidates:
        disposition = "lifecycle not inferred; unscoped lexical reset candidate: " + " | ".join(reset_candidates)
    else:
        disposition = "lifecycle not inferred; no binding-resolved reset analysis"
    return "immutable binding" if immutable else "mutable", synchronization, owner, disposition


def get_parser_checked():
    found_pack, found_ts = version("tree-sitter-language-pack"), version("tree-sitter")
    if (found_pack, found_ts) != (PARSER_PACK_VERSION, TREE_SITTER_VERSION):
        raise RuntimeError(
            f"parser dependency mismatch: require tree-sitter-language-pack=={PARSER_PACK_VERSION} "
            f"and tree-sitter=={TREE_SITTER_VERSION}; found {found_pack} and {found_ts}"
        )
    from tree_sitter_language_pack import get_parser
    return get_parser("cpp"), f"tree-sitter-cpp ABI 15 (language-pack={found_pack}, tree-sitter={found_ts})"


def parse_source(parser, source):
    source_b = source.encode()
    root = call(parser.parse(source), "root_node")
    declarations = [item for item in walk(root) if kind(item) in DECL_KINDS]
    gaps = [item for item in walk(root) if kind(item) == "ERROR" or is_missing(item)]
    recovered_regions = recovered_function_regions(source_b, gaps)
    rows = [row for decl in declarations for row in declaration_rows(source_b, decl, recovered_regions)]
    gaps, categories, failures = recovery_coverage(source_b, root, declarations, recovered_regions)
    return source_b, rows, gaps, categories, failures


def self_test(parser, repo):
    source = """
struct C {
    static int member;
    void method() { static int local{1}; auto fn = [] { static int lambda_local = 2; }; }
};
static const int * const_pointee;
static int * const const_pointer = nullptr;
static std::vector<const int *> mutable_container{};
static const std::atomic<int> immutable_atomic{0};
namespace one { static std::vector<int> same; void f() { static std::vector<int> same; same.clear(); } }
namespace two { static std::vector<int> same; }
void direct_fn() { static Widget direct{3}; }
static int first, * second, third{3};
namespace ext { extern int declaration_only; extern int definition = 1; }
"""
    _, rows, gaps, _, failures = parse_source(parser, source)
    assert not gaps and not failures
    by_name = {}
    for row in rows: by_name.setdefault(row["name"], []).append(row)
    assert by_name["member"][0]["scope"] == "class:C"
    assert by_name["local"][0]["scope"] == "function-local:method"
    assert by_name["lambda_local"][0]["scope"] == "function-local:<lambda>"
    assert not by_name["const_pointee"][0]["immutable"]
    assert by_name["const_pointer"][0]["immutable"]
    assert not by_name["mutable_container"][0]["immutable"]
    assert by_name["immutable_atomic"][0]["immutable"]
    assert by_name["direct"][0]["type"] == "Widget"
    assert all(name in by_name for name in ("first", "second", "third"))
    assert "declaration_only" not in by_name and "definition" in by_name
    same_rows = by_name["same"]
    lines = source.splitlines()
    for row in same_rows:
        writes, _, resets = lexical_evidence(lines, "same", row["decl_line"])
        _, _, _, disposition = classify(row, resets)
        assert writes.startswith("unscoped lexical candidate:") or writes.startswith("none found by")
        assert not disposition.startswith("reset/teardown evidence:")

    negative_fixtures = {
        "function-body-static-recovery": "void f(){ static Widget x{; }",
        "structural-preprocessor-recovery": "#if X\nWidget implicit_global{;\n#endif",
        "namespace-direct-init-recovery": "namespace broken { Widget implicit_global{; }",
        "recovered-function-tail": "static void recovered() { #wat x }\nWidget implicit_global{;",
        "parsed-function-recovery-tail": "static void recovered() { #wat x }\nWidget implicit_global{};",
        "parsed-function-wrong-close": "static void recovered() { x; x template #if X }\nWidget implicit_global{};",
        "lambda-wrong-close": "static auto recovered = [] { x; x template #if X }\nWidget implicit_global{};",
        "template-lambda-wrong-close": "static auto recovered = []<typename T> { x; x template #if X }\nWidget implicit_global{};",
        "nested-function-lambda-wrong-close": "static void outer() { auto recovered = [] { x; x template #if X }\nstatic Widget hidden{}; }",
    }
    for fixture, broken in negative_fixtures.items():
        _, _, broken_gaps, _, broken_failures = parse_source(parser, broken)
        assert broken_gaps and broken_failures, f"{fixture} must fail closed"
        if fixture in {
            "recovered-function-tail", "parsed-function-recovery-tail", "parsed-function-wrong-close",
            "lambda-wrong-close", "template-lambda-wrong-close", "nested-function-lambda-wrong-close",
        }:
            assert any(reason == "unproved recovery tail after function body" for _, reason in broken_failures)
    audited = (repo / FILES[0]).read_text(encoding="utf-8")
    audited_b, audited_rows, _, _, audited_failures = parse_source(parser, audited)
    assert not audited_failures
    expected_file_scopes = {
        93499: "ggml_backend_sycl_interface",
        94640: "ggml_backend_sycl_device_interface",
        94700: "ggml_backend_sycl_reg_interface",
        94801: "g_sycl_seq_ids_cache",
        94857: "g_sycl_device_token_cache",
    }
    actual_file_scopes = {
        line_of(audited_b, row["name_node"]): (row["name"], row["scope"])
        for row in audited_rows
        if line_of(audited_b, row["name_node"]) in expected_file_scopes
    }
    assert actual_file_scopes == {
        line: (symbol, "file") for line, symbol in expected_file_scopes.items()
    }, f"recovered function tail scopes: {actual_file_scopes}"
    print("fixtures=PASS method-local,const-pointee,same-name,direct-init-recovery,multi-object,namespaced-extern,"
          "function-body-static-recovery,structural-preprocessor-recovery,recovered-function-tail,"
          "parsed-function-recovery-tail,parsed-function-wrong-close,lambda-wrong-close,"
          "template-lambda-wrong-close,nested-function-lambda-wrong-close,file-tail-scopes")


def main():
    parser, parser_name = get_parser_checked()
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="docs/backend/sycl-static-storage-inventory.csv")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    if args.self_test:
        self_test(parser, args.repo)
        return 0

    all_rows, reports, failures = [], [], []
    declaration_ids = set()
    reconciliation = Counter()
    for rel in FILES:
        source = (args.repo / rel).read_text(encoding="utf-8")
        source_b, parsed, gaps, gap_categories, file_failures = parse_source(parser, source)
        failures.extend(f"{rel}:{line}:{reason}" for line, reason in file_failures)
        lines = source.splitlines()
        for row in parsed:
            writes, reads, resets = lexical_evidence(lines, row["name"], row["decl_line"])
            mutability, synchronization, owner, disposition = classify(row, resets)
            scope, storage = row["scope"], row["storage"]
            if scope.startswith("function-local:"): reconciliation["function_local_objects"] += 1
            elif scope.startswith("class:"): reconciliation["class_static_objects"] += 1
            elif "static" in storage:
                reconciliation["explicit_static_nonlocal_objects"] += 1
                declaration_ids.add((rel, row["decl_line"]))
            else: reconciliation["implicit_nonlocal_objects"] += 1
            all_rows.append({
                "file": rel, "line": line_of(source_b, row["name_node"]), "symbol": row["name"],
                "type": row["type"], "scope": scope, "mutability": mutability,
                "synchronization": synchronization, "writer_evidence": writes, "reader_evidence": reads,
                "owner_identity": owner, "reset_teardown_disposition": disposition,
            })
        reports.append((rel, len(parsed), len(gaps), dict(gap_categories), hashlib.sha256(source_b).hexdigest()))

    if failures:
        print("ERROR: fail-closed recovery coverage rejected the census:", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 2

    all_rows.sort(key=lambda row: (FILES.index(row["file"]), int(row["line"]), row["symbol"]))
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader(); writer.writerows(all_rows)
    rendered = buffer.getvalue()
    output = args.repo / args.output
    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != rendered:
            print(f"ERROR: {args.output} is stale; regenerate without --check", file=sys.stderr)
            return 1
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")

    print(f"parser={parser_name}")
    for rel, rows, gaps, categories, sha in reports:
        print(f"{rel}: rows={rows} recovery_nodes={gaps} recovery_proof={categories} sha256={sha}")
    print(f"inventory_rows={len(all_rows)} lexical_candidate_leads=371 (input lacks method/SHA; not a census)")
    print(f"explicit_static_nonlocal_declarations={len(declaration_ids)}")
    print("reconciliation=" + ",".join(f"{key}:{reconciliation[key]}" for key in (
        "explicit_static_nonlocal_objects", "implicit_nonlocal_objects", "function_local_objects", "class_static_objects")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
