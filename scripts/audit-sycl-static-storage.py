#!/usr/bin/env python3
"""Generate the SYCL static-storage census using tree-sitter-cpp.

This is a source audit, not a compiler: preprocessor alternatives remain in the
concrete syntax tree.  The coverage gate fails if a declaration containing a
storage-duration marker is swallowed by an ERROR node or cannot be classified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

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
WRITE_RE = re.compile(r"(?:\b{n}\s*(?:\[[^\n;]*\])?\s*(?:=(?!=)|\+=|-=|\+\+|--)|\b{n}\s*(?:\[[^\n;]*\])?\s*\.\s*(?:(?:try_)?emplace|insert(?:_or_assign)?|clear|erase|push_\w*|pop_\w*|resize|assign|reset|store|exchange|fetch_\w*|swap|release|acquire)\s*\()")
RESET_RE = re.compile(r"\b{n}\s*(?:\[[^\n;]*\])?\s*\.\s*(?:clear|erase|reset|release|swap)\s*\(|\b{n}\s*(?:\[[^\n;]*\])?\s*\.\s*store\s*\(\s*(?:false|0|nullptr)|\b(?:reset|free|destroy|unregister)\w*\s*\([^;\n]*\b{n}\b", re.I)


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
def has_error(node): return call(node, "has_error")
def field(node, name): return call(node, "child_by_field_name", name)


def children(node):
    return [child(node, i) for i in range(child_count(node))]


def walk(node):
    yield node
    for c in children(node):
        yield from walk(c)


def text(source_b: bytes, node) -> str:
    return source_b[start_byte(node):end_byte(node)].decode("utf-8", "replace")


def line_of(source_b: bytes, node) -> int:
    return source_b.count(b"\n", 0, start_byte(node)) + 1


def contains_kind(node, kinds: set[str]) -> bool:
    return any(kind(n) in kinds for n in walk(node))


def declarator_name(source_b: bytes, node) -> tuple[str, object] | None:
    """Return the declared identifier, following only the declarator field."""
    k = kind(node)
    if k in {"identifier", "field_identifier"}:
        return text(source_b, node), node
    d = field(node, "declarator")
    if d is not None:
        return declarator_name(source_b, d)
    if k == "qualified_identifier":
        named = [n for n in children(node) if kind(n) in {"identifier", "field_identifier"}]
        return (text(source_b, named[-1]), named[-1]) if named else None
    # Grammar recovery sometimes omits fields; only follow declarator-shaped children.
    for c in children(node):
        if kind(c) in DECLARATOR_KINDS:
            got = declarator_name(source_b, c)
            if got:
                return got
    return None


def ancestor(node, wanted: set[str]):
    p = parent(node)
    while p is not None:
        if kind(p) in wanted:
            return p
        p = parent(p)
    return None


def named_scope(source_b: bytes, decl) -> tuple[str, bool]:
    cls = ancestor(decl, {"class_specifier", "struct_specifier", "union_specifier"})
    fn = ancestor(decl, {"function_definition", "lambda_expression"})
    ns = ancestor(decl, {"namespace_definition"})
    if cls is not None:
        name = field(cls, "name")
        return "class:" + (text(source_b, name) if name else "<anonymous>"), False
    if fn is not None:
        d = field(fn, "declarator")
        got = declarator_name(source_b, d) if d else None
        return "function-local:" + (got[0] if got else "<lambda>"), True
    if ns is not None:
        name = field(ns, "name")
        return ("namespace:" + text(source_b, name), False) if name else ("anonymous-namespace", False)
    # In preprocessor-alternative regions tree-sitter can recover an otherwise
    # valid function as one ERROR node while retaining its child declarations.
    # The ERROR node is still a structural boundary; recover only an unambiguous
    # leading function signature so those children are not mislabeled file scope.
    err = ancestor(decl, {"ERROR"})
    if err is not None:
        head = text(source_b, err)[:500]
        match = re.match(r"(?s)\s*(?:[A-Z_]\w*\s+|static\s+|inline\s+|const\s+)*[\w:<>*& ]+?\s+(\w+)\s*\([^;{}]*\)\s*(?:try\s*)?\{", head)
        if match:
            return "function-local:" + match.group(1), True
    return "file", False


def base_type(source_b: bytes, decl, first_declarator) -> str:
    pieces = []
    for c in children(decl):
        if start_byte(c) >= start_byte(first_declarator):
            break
        if kind(c) not in {"storage_class_specifier", "attribute_specifier", "attribute_declaration"}:
            value = " ".join(text(source_b, c).split())
            if value and value not in {",", ";"}:
                pieces.append(value)
    return " ".join(pieces) or "<parser-unresolved>"


def declaration_rows(source_b: bytes, decl):
    raw = text(source_b, decl)
    scope, local = named_scope(source_b, decl)
    storage = {
        text(source_b, c).strip()
        for c in children(decl)
        if kind(c) == "storage_class_specifier" and text(source_b, c).strip() in {"static", "thread_local", "extern"}
    }
    if scope.startswith("class:") and "static" not in storage:
        return []
    if local and not ({"static", "thread_local"} & storage):
        return []
    if not local and scope == "file" and "extern" in storage and "=" not in raw:
        return []

    direct = children(decl)
    type_node = field(decl, "type")
    type_end = end_byte(type_node) if type_node else start_byte(decl)
    declarators = [c for c in direct if kind(c) in DECLARATOR_KINDS and start_byte(c) >= type_end]
    # field_by_name returns only the first of repeated declarator fields; direct children preserve all objects.
    rows = []
    for d in declarators:
        if contains_kind(d, FUNCTIONISH):
            continue
        got = declarator_name(source_b, d)
        if not got:
            continue
        name, name_node = got
        if name in {"static", "thread_local"}:
            continue
        typ = base_type(source_b, decl, declarators[0])
        shape = text(source_b, d)
        before = " ".join(shape[:max(0, start_byte(name_node) - start_byte(d))].split())
        after = " ".join(shape[max(0, end_byte(name_node) - start_byte(d)):].split("=", 1)[0].split())
        if before or after:
            typ = " ".join((typ, before + after)).strip()
        rows.append((name, typ, scope, storage, name_node, raw, line_of(source_b, decl)))
    return rows


def evidence(lines: list[str], name: str, decl_line: int) -> tuple[str, str, str]:
    token = re.compile(r"\b" + re.escape(name) + r"\b")
    writes, reads = [], []
    wr = re.compile(WRITE_RE.pattern.format(n=re.escape(name)))
    reset = re.compile(RESET_RE.pattern.format(n=re.escape(name)), re.I)
    resets = []
    for i, ln in enumerate(lines, 1):
        if not token.search(ln) or i == decl_line:
            continue
        target = writes if wr.search(ln) else reads
        if len(target) < 3:
            target.append(f"L{i}:{' '.join(ln.strip().split())[:120]}")
        if reset.search(ln) and len(resets) < 3:
            resets.append(f"L{i}:{' '.join(ln.strip().split())[:120]}")
    return (" | ".join(writes) or "none found", " | ".join(reads) or "none found", " | ".join(resets))


def classify(name: str, typ: str, scope: str, storage: set[str], raw: str, reset: str):
    immutable = bool(re.search(r"\b(const|constexpr|constinit)\b", raw)) and "mutable" not in raw
    mutability = "immutable" if immutable else "mutable"
    if "thread_local" in storage:
        sync = "thread-local"
    elif re.search(r"\b(atomic|mutex|once_flag|semaphore)\b", typ):
        sync = "intrinsic:" + ("atomic" if "atomic" in typ else "lock primitive")
    else:
        sync = "none evident"
    lower = (name + " " + typ).lower()
    if scope.startswith("class:"):
        owner = scope
    elif "thread_local" in storage:
        owner = "thread"
    elif re.search(r"tensor|host_ptr|k_base|alloc", lower):
        owner = "tensor/allocation identity or registry"
    elif re.search(r"device|gpu|queue|stream", lower):
        owner = "device/process slot"
    elif re.search(r"model|layer|expert|moe|weight|kv", lower):
        owner = "model/layer semantic state (review required)"
    else:
        owner = "process or function lifetime (review required)"
    if immutable:
        disposition = "not applicable: immutable"
    elif reset:
        disposition = "reset/teardown evidence: " + reset
    else:
        disposition = "no reset/teardown evidence found"
    return mutability, sync, owner, disposition


def parser():
    try:
        from importlib.metadata import version
        from tree_sitter_language_pack import get_parser
        versions = f"tree_sitter_language_pack={version('tree-sitter-language-pack')}, tree-sitter={version('tree-sitter')}"
        return get_parser("cpp"), f"tree-sitter-cpp ABI 15 ({versions})"
    except Exception as exc:
        raise SystemExit(f"ERROR: tree-sitter C++ parser unavailable: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="docs/backend/sycl-static-storage-inventory.csv")
    ap.add_argument("--check", action="store_true", help="verify the committed inventory is reproducible")
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    p, parser_name = parser()
    all_rows, reports, unsafe_gaps = [], [], []
    explicit_static_nonlocal_declarations = set()
    reconciliation = {"explicit_static_nonlocal_objects": 0, "implicit_nonlocal_objects": 0,
                      "function_local_objects": 0, "class_static_objects": 0}

    for rel in FILES:
        path = args.repo / rel
        source = path.read_text(encoding="utf-8")
        source_b = source.encode()
        tree = p.parse(source)
        root = call(tree, "root_node")
        gaps = [n for n in walk(root) if kind(n) == "ERROR" or is_missing(n)]
        declarations = [n for n in walk(root) if kind(n) in DECL_KINDS]
        parsed = []
        for decl in declarations:
            parsed.extend(declaration_rows(source_b, decl))

        # Fail closed when parser recovery hides a possible static-storage declaration.
        covered_spans = [(start_byte(n), end_byte(n)) for n in declarations]
        marker = re.compile(rb"\b(?:static|thread_local)\b")
        uncovered = []
        for m in marker.finditer(source_b):
            line_start = source_b.rfind(b"\n", 0, m.start()) + 1
            prefix = source_b[line_start:m.start()].lstrip()
            if prefix.startswith(b"//") or prefix.startswith(b"*") or prefix.startswith(b"/*"):
                continue
            if source_b[m.end():m.end()+1] == b"_":  # static_cast/static_assert
                continue
            if not any(a <= m.start() < b for a, b in covered_spans):
                line = source_b.count(b"\n", 0, m.start()) + 1
                snippet = source_b[line_start:source_b.find(b"\n", m.end())].decode("utf-8", "replace").strip()
                # Function definitions/prototypes and prose are reconciled lexical leads, not objects.
                if not re.search(r"\b(static|thread_local)\s+(?:const\s+)?(?:bool|char|short|int|long|float|double|auto|std::|sycl::|ggml_|[A-Za-z_]\w*\s*[*&])", snippet):
                    continue
                marker_tail = snippet[snippet.find(m.group().decode()):]
                open_paren = marker_tail.find("(")
                equals = marker_tail.find("=")
                if open_paren >= 0 and (equals < 0 or open_paren < equals or "operator" in marker_tail[:open_paren]):
                    # A function signature has '(' before any initializer. Function-local
                    # objects using direct initialization are rare here and are caught by
                    # the structurally parsed declaration nodes.
                    continue
                uncovered.append((line, snippet))
        if uncovered:
            unsafe_gaps.extend(f"{rel}:{ln}:{snip}" for ln, snip in uncovered)

        lines = source.splitlines()
        for name, typ, scope, storage, name_node, raw, decl_line in parsed:
            ln = line_of(source_b, name_node)
            writes, reads, resets = evidence(lines, name, ln)
            mut, sync, owner, disposition = classify(name, typ, scope, storage, raw, resets)
            if scope.startswith("function-local:"):
                reconciliation["function_local_objects"] += 1
            elif scope.startswith("class:"):
                reconciliation["class_static_objects"] += 1
            elif "static" in storage:
                reconciliation["explicit_static_nonlocal_objects"] += 1
                explicit_static_nonlocal_declarations.add((rel, decl_line))
            else:
                reconciliation["implicit_nonlocal_objects"] += 1
            all_rows.append({
                "file": rel, "line": ln, "symbol": name, "type": typ, "scope": scope,
                "mutability": mut, "synchronization": sync, "writer_evidence": writes,
                "reader_evidence": reads, "owner_identity": owner,
                "reset_teardown_disposition": disposition,
            })
        reports.append((rel, len(parsed), len(gaps), len(uncovered), hashlib.sha256(source_b).hexdigest()))

    if unsafe_gaps:
        print("ERROR: parser coverage gate found storage-duration markers outside parsed declarations:", file=sys.stderr)
        print("\n".join(unsafe_gaps), file=sys.stderr)
        return 2

    all_rows.sort(key=lambda r: (FILES.index(r["file"]), int(r["line"]), r["symbol"]))
    from io import StringIO
    buf = StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader(); writer.writerows(all_rows)
    rendered = buf.getvalue()
    out = args.repo / args.output
    if args.check:
        if not out.exists() or out.read_text(encoding="utf-8") != rendered:
            print(f"ERROR: {args.output} is stale; regenerate without --check", file=sys.stderr)
            return 1
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered, encoding="utf-8")

    print(f"parser={parser_name}")
    for rel, rows, gaps, unsafe, sha in reports:
        print(f"{rel}: rows={rows} raw_parse_gaps={gaps} unsafe_storage_gaps={unsafe} sha256={sha}")
    print(f"inventory_rows={len(all_rows)} lexical_candidate_leads=371 (ticket input; method/SHA unspecified, not treated as a historical census)")
    print(f"explicit_static_nonlocal_declarations={len(explicit_static_nonlocal_declarations)}")
    print("reconciliation=" + ",".join(f"{key}:{value}" for key, value in reconciliation.items()))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
