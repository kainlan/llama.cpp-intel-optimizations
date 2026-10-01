#!/usr/bin/env python3
"""Source gate for the SYCL allocation-zone contract (llama.cpp-23mk, design section 7).

Every construction of a request type must say, by literal, which tier it asks for, and a device
request must name its arena zone and forbid the spill. Raw allocator names stay outside the
unified-cache's own allowlisted functions. The gate parses the tree with tree-sitter, so a
comment or a string literal never counts as code, and it keys every finding by AST node
(`file::function::node-kind:variable:text-hash#ordinal`), never by line or byte offset. The text
hash is of the construction's normalized text (comments dropped, whitespace collapsed) and the
ordinal counts only identical constructions in the same function, so a key survives an unrelated
edit above it and moves only when the construction itself changes.

What this unit (S2a) enforces
  (a) a request-type token inside an ERROR/MISSING region fails; a file whose root is ERROR
      (today cpu-dispatch.cpp) is also scanned lexically, and each lexical request must be
      host-only by a literal on its statement run
  (b) a literal tier flag: must_device = true or must_host_pinned = true, not both
  (d) a device request carries a prefer_vram_zone that is a literal non-COUNT enumerator, or a
      ternary whose arms all are, and forbid_vram_zone_spill = true with no later write of false
  (e) raw allocator names (calls, string literals, #define bodies) outside the allowlist
  (f) a missing tree_sitter_language_pack is a FAIL, never a skip
and the brace rule the construction-site labels need: a braceless `T x;` reports the class
site (unified-cache.hpp, the comment above alloc_intent), so every value declaration of a
request type is written `T x{}`.

Not here yet (later units, each lands with its witnesses)
  (c) interprocedural flow, so a copy / helper return / by-reference write is a construction of
      its own. Until it lands such constructions are listed as DEFER-C debt and are shrink-only,
      so a new copy cannot slip in unseen. (g)-(p) are S2c/S2d.

Debt model
  The tree has constructions that violate the rules today. They are an explicit list
  (scripts/sycl-alloc-zone-contract/debt.json) that may only shrink: a violation not on the
  list fails, and a listed entry that no longer violates fails too (delete it in the commit
  that fixes it). The allowlist (allowlist.json) is for permanent exemptions with a reason; each
  entry prints how many findings it covered and fails when that is not its exact `count`.

Pointer-only holders (`ext_alloc_request_scope` holds a `const alloc_request *`) are tokens for
clause (a) but are not constructions of a request: they carry no flag a request could be
missing. Holders of a request by value (`offload_buffer_request`) are constructions.

Mutation matrix (--mutation-matrix)
  Each case mutates an in-memory copy of the tree (nothing is written, so the checkout cannot be
  touched) and states the verdict it must flip to. The unmutated baseline must PASS first, or the
  matrix would be scored against a red baseline. A sites-that-do-not-conform-yet case cannot be
  mutated in place (the violation is already in the debt list), so a witness plants a
  twin: a request of the same shape in a fresh function, whose unmutated form must PASS and whose
  mutated form must FAIL naming the planted node.

Dependency
  Needs the tree-sitter C++ grammar: `pip install tree-sitter-language-pack` in the python3 that
  CMake finds. A missing module is a FAIL (exit 1) and the message names it; there is no skip.

Usage:
  check-sycl-alloc-zone-contract.py [--root REPO] [--mutation-matrix] [--list] [--write-debt]
"""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

try:
    from tree_sitter_language_pack import get_parser
    _PARSER = get_parser("cpp")
except Exception as exc:  # ImportError, or a pack without the grammar
    # Clause (f): a gate that cannot parse is a failing gate. SKIP/77 would read as "verified".
    print("FAIL: tree_sitter_language_pack with the C++ grammar is required (%s: %s); "
          "this gate does not skip" % (type(exc).__name__, exc))
    sys.exit(1)

SCOPE_SUBDIR = "ggml/src/ggml-sycl"
SKIP_DIRS = {"tests", "dpct", "template-instances", "docs"}
BASE_TYPES = ("alloc_request", "alloc_intent", "alloc_constraints")
ZONES = ("KV", "WEIGHT", "ONEDNN", "RUNTIME", "SCRATCH")
# Fields whose literal value the rules read. Later clauses add to this set.
TRACKED = ("must_device", "must_host_pinned", "prefer_vram_zone", "forbid_vram_zone_spill",
           "cascade_step", "unconverted_ticket", "cohort_id")
# Fields of a request that are themselves structs: a write to one is a copy, not a literal.
STRUCT_FIELDS = ("intent", "constraints")
# Clause (e). Calls match on the last name component, strings on substring.
RAW_CALLS = (
    "unified_cache_malloc_device_tracked", "unified_cache_raw_malloc_device", "sycl_aligned_malloc_device",
    "ggml_sycl_malloc_device_raw", "ggml_sycl_free_device_raw", "unified_cache_raw_free_device",
    "malloc_device", "malloc_shared", "aligned_alloc_device", "aligned_alloc_shared",
    "zeMemAllocDevice", "zeMemAllocShared", "zePhysicalMemCreate", "zeVirtualMemReserve",
)
RAW_STRINGS = ("zeMemAllocDevice", "zeMemAllocShared", "zePhysicalMemCreate", "zeVirtualMemReserve")

DEBT_DOC = ("Read by scripts/check-sycl-alloc-zone-contract.py (clauses a, b, d, e). Shrink-only: a violation not listed "
            "fails, and a listed entry that no longer violates fails. Every E-RAW entry carries a fate (deleted-by-*, "
            "converted-by-*, or sanctioned-internal) and a cite, so an entry no step will ever shrink is visible as a "
            "mislabelled allowlist entry. See README.md for regeneration.")
FATE_RE = re.compile(r"^(deleted-by|converted-by)-[A-Za-z0-9._§()-]+$|^sanctioned-internal$")

CODES = ("A-ERROR", "A-LEXICAL", "A-TOKEN", "B-BRACE", "B-TIER", "D-ZONE", "D-ZONE-COUNT",
         "D-FORBID", "D-FORBID-FALSE", "E-RAW", "DEFER-C")


# ---------------------------------------------------------------- tree-sitter accessors
# The binding exposes some members as methods and some as properties depending on version.

def _a(n, name, *args):
    v = getattr(n, name)
    return v(*args) if callable(v) else v


def kind(n): return _a(n, "kind")
def sb(n): return _a(n, "start_byte")
def eb(n): return _a(n, "end_byte")
def parent(n): return _a(n, "parent")
def fld(n, f): return _a(n, "child_by_field_name", f)
def nkids(n): return _a(n, "child_count")
def kid(n, i): return _a(n, "child", i)
def kids(n): return [kid(n, i) for i in range(nkids(n))]


def line_of(n):
    sp = _a(n, "start_position")
    return (sp[0] if isinstance(sp, tuple) else _a(sp, "row")) + 1


def walk(n):
    stack = [n]
    while stack:
        x = stack.pop()
        yield x
        stack.extend(reversed(kids(x)))


def txt(src, n):
    return src[sb(n):eb(n)].decode("utf-8", "replace")


def text_hash(s):
    """Short hash of a construction's text with comments dropped and whitespace collapsed."""
    s = re.sub(r"/\*.*?\*/|//[^\n]*", " ", s, flags=re.S)
    return hashlib.sha1(re.sub(r"\s+", " ", s).strip().encode()).hexdigest()[:8]


def base_type(s):
    s = re.sub(r"\b(const|volatile|static|thread_local|inline|constexpr|struct)\b", "", s)
    return s.strip().split("::")[-1].strip()


# ---------------------------------------------------------------- the tree

def load_tree(root):
    """rel path -> bytes for every in-scope source, sorted."""
    base = Path(root) / SCOPE_SUBDIR
    files = {}
    for p in sorted(base.rglob("*")):
        if p.suffix not in (".cpp", ".hpp") or not p.is_file():
            continue
        rel = p.relative_to(base)
        if set(rel.parts[:-1]) & SKIP_DIRS:
            continue
        files[str(rel)] = p.read_bytes()
    if not files:
        print("FAIL: no sources found under %s" % base)
        sys.exit(1)
    return files


_PARSE = {}      # sha1 -> (src, root node)
_STRUCTS = {}    # sha1 -> [(struct name, [(member base type, is_value)])]
_FACTS = {}      # (sha1, types key) -> facts dict


def _sha(src):
    return hashlib.sha1(src).hexdigest()


def parse(src):
    h = _sha(src)
    if h not in _PARSE:
        tree = _PARSER.parse_bytes(src) if hasattr(_PARSER, "parse_bytes") else _PARSER.parse(src.decode("utf-8", "replace"))
        _PARSE[h] = _a(tree, "root_node")
    return _PARSE[h]


def file_structs(src):
    h = _sha(src)
    if h not in _STRUCTS:
        root = parse(src)
        out = []
        for n in walk(root):
            if kind(n) not in ("struct_specifier", "class_specifier"):
                continue
            nm, body = fld(n, "name"), fld(n, "body")
            if nm is None or body is None:
                continue
            members = []
            for c in kids(body):
                if kind(c) != "field_declaration":
                    continue
                t = fld(c, "type")
                if t is None:
                    continue
                value = any(kind(d) in ("field_identifier", "init_declarator", "identifier")
                            for d in kids(c) if d is not t)
                members.append((base_type(txt(src, t)), value))
            out.append((txt(src, nm), members))
        _STRUCTS[h] = out
    return _STRUCTS[h]


def type_closure(files):
    """(value types, all request-bearing types). A struct that holds a request type by value is a
    request type; one that holds it only by pointer is a token for clause (a) but not a request."""
    value = set(BASE_TYPES)
    allt = set(BASE_TYPES)
    changed = True
    while changed:
        changed = False
        for src in files.values():
            for name, members in file_structs(src):
                if name in BASE_TYPES:
                    continue
                for bt, is_value in members:
                    if bt in value and name not in value and is_value:
                        value.add(name)
                        allt.add(name)
                        changed = True
                    elif bt in allt and name not in allt:
                        allt.add(name)
                        changed = True
    return value, allt


# ---------------------------------------------------------------- naming

def enclosing(src, n):
    """(function name, node of the function or None). Class members get the class prefix, and a
    lambda is named after the function that holds it, so a key survives a line shift."""
    lam = False
    classes = []
    p = parent(n)
    func = None
    fnode = None
    while p is not None:
        k = kind(p)
        if k == "lambda_expression":
            lam = True
        elif k in ("struct_specifier", "class_specifier"):
            nm = fld(p, "name")
            if nm is not None:
                classes.append(txt(src, nm))
            if kind(n) == "field_declaration" and func is None and p is parent(parent(n)):
                pass
        elif k == "function_definition" and func is None:
            d = fld(p, "declarator")
            while d is not None and kind(d) != "function_declarator":
                d = fld(d, "declarator")
            func = "?"
            if d is not None:
                nm = fld(d, "declarator")
                if nm is not None:
                    func = re.sub(r"\s+", "", txt(src, nm))
            fnode = p
        p = parent(p)
    if func is None:
        func = "<member of %s>" % classes[0] if (kind(n) == "field_declaration" and classes) else "<file scope>"
        return func, None
    if classes and "::" not in func:
        func = "::".join(reversed(classes)) + "::" + func
    return ("lambda in " if lam else "") + func, fnode


def has_error_ancestor(n):
    p = parent(n)
    while p is not None:
        if kind(p) == "ERROR" and parent(p) is not None:
            return True
        p = parent(p)
    return False


# ---------------------------------------------------------------- literal classification

def bool_class(src, n):
    t = txt(src, n).strip()
    return "true" if t == "true" else "false" if t == "false" else "expr"


def zone_class(src, n):
    """'ok' when every arm is a literal non-COUNT enumerator, 'count' when any arm is COUNT,
    otherwise 'nonliteral'."""
    k = kind(n)
    if k == "parenthesized_expression":
        inner = [c for c in kids(n) if _a(c, "is_named")]
        return zone_class(src, inner[0]) if inner else "nonliteral"
    if k == "conditional_expression":
        arms = [zone_class(src, fld(n, "consequence")), zone_class(src, fld(n, "alternative"))]
        return "count" if "count" in arms else "nonliteral" if "nonliteral" in arms else "ok"
    if k == "qualified_identifier":
        # `a::b::C` nests to the right: qualified(a, qualified(b, C)); the enumerator is the innermost name
        while fld(n, "name") is not None and kind(fld(n, "name")) == "qualified_identifier":
            n = fld(n, "name")
        scope, name = fld(n, "scope"), fld(n, "name")
        if scope is not None and name is not None and txt(src, scope).split("::")[-1].strip() == "vram_zone_id":
            nm = txt(src, name).strip()
            if nm == "COUNT":
                return "count"
            if nm in ZONES:
                return "ok"
    return "nonliteral"


def classify(src, field, n):
    if field == "prefer_vram_zone":
        return zone_class(src, n)
    if field in ("must_device", "must_host_pinned", "forbid_vram_zone_spill", "cascade_step"):
        return bool_class(src, n)
    return "expr"


def init_fields(src, lst, out, copied):
    """Fold a braced initialiser into `out` (field -> [(class, text)]). A designated pair that
    names a struct member with a non-list value is a copy; a positional element is unresolvable."""
    for c in kids(lst):
        k = kind(c)
        if k == "initializer_pair":
            desig = [x for x in kids(c) if kind(x) == "field_designator"]
            name = None
            if desig:
                fi = [x for x in kids(desig[-1]) if kind(x) == "field_identifier"]
                name = txt(src, fi[0]) if fi else None
            val = fld(c, "value")
            if val is None:
                continue
            if kind(val) == "initializer_list":
                init_fields(src, val, out, copied)
            elif name in STRUCT_FIELDS:
                copied.append(name)
            elif name in TRACKED:
                out.setdefault(name, []).append((classify(src, name, val), re.sub(r"\s+", " ", txt(src, val))[:80]))
        elif k in ("{", "}", ",", "comment"):
            continue
        else:
            copied.append("positional")


# ---------------------------------------------------------------- extraction

def declarator_info(src, d):
    """(name, form) for one declarator. form: 'plain' | 'braced' | 'copy' | 'ctor' | 'skip'."""
    k = kind(d)
    if k in ("identifier", "field_identifier"):
        return txt(src, d), "plain", None
    if k == "init_declarator":
        decl = fld(d, "declarator")
        if decl is None or kind(decl) not in ("identifier", "field_identifier"):
            return None, "skip", None
        val = fld(d, "value")
        if val is None:
            return txt(src, decl), "plain", None
        if kind(val) == "initializer_list":
            return txt(src, decl), "braced", val
        if kind(val) == "argument_list":
            return txt(src, decl), "ctor", val
        return txt(src, decl), "copy", val
    return None, "skip", None


def scope_block(n):
    p = parent(n)
    while p is not None and kind(p) not in ("compound_statement", "field_declaration_list", "declaration_list",
                                            "translation_unit"):
        p = parent(p)
    return p


_ASSIGN_CACHE = {}


def assignments_in(src_key, src, block):
    """Every `=` assignment in a block whose left side is a field chain rooted at an identifier:
    [(root name, last field, chain, rhs node, assignment node)]."""
    key = (src_key, sb(block), eb(block))
    if key in _ASSIGN_CACHE:
        return _ASSIGN_CACHE[key]
    out = []
    for n in walk(block):
        if kind(n) != "assignment_expression":
            continue
        op = [txt(src, c) for c in kids(n) if not _a(c, "is_named")]
        if op[:1] != ["="]:
            continue
        lhs, rhs = fld(n, "left"), fld(n, "right")
        chain = []
        x = lhs
        while x is not None and kind(x) == "field_expression":
            f = fld(x, "field")
            chain.append(txt(src, f) if f is not None else "?")
            x = fld(x, "argument")
        if x is None or kind(x) != "identifier" or not chain:
            continue
        chain.reverse()
        out.append((txt(src, x), chain[-1], chain, rhs, n))
    _ASSIGN_CACHE[key] = out
    return out


def scan_file(rel, src, value_types, all_types):
    """Facts about one file, as plain data: constructions, raw hits, error tokens, defined functions."""
    h = _sha(src)
    root = parse(src)
    root_error = kind(root) == "ERROR"
    constructions, raws, errtoks, funcs = [], [], [], []
    ordinal = {}

    def key_for(func, nodekind, var, node):
        base = "%s::%s::%s:%s:%s" % (rel, func, nodekind, var, text_hash(txt(src, node)))
        i = ordinal.get(base, 0)
        ordinal[base] = i + 1
        return "%s#%d" % (base, i)

    for n in walk(root):
        k = kind(n)
        if k == "function_definition":
            d = fld(n, "declarator")
            while d is not None and kind(d) != "function_declarator":
                d = fld(d, "declarator")
            if d is not None and fld(d, "declarator") is not None:
                funcs.append(re.sub(r"\s+", "", txt(src, fld(d, "declarator"))))
        elif k in ("declaration", "field_declaration"):
            t = fld(n, "type")
            if t is None or base_type(txt(src, t)) not in value_types:
                continue
            func, fnode = enclosing(src, n)
            if k == "field_declaration":
                holder = parent(parent(n))
                hn = fld(holder, "name") if holder is not None and kind(holder) in ("struct_specifier", "class_specifier") else None
                if hn is not None and txt(src, hn) in BASE_TYPES:
                    continue  # the request types' own members define their defaults; they construct nothing
            err = has_error_ancestor(n)
            ks = kids(n)
            for ki, d in enumerate(ks):
                if d is t:
                    continue
                name, form, val = declarator_info(src, d)
                if form == "skip" or name is None:
                    continue
                if k == "field_declaration" and form == "plain" and ki + 1 < len(ks):
                    # a default member initialiser is a sibling of the field name, not an init_declarator
                    nxt = ks[ki + 1]
                    if kind(nxt) == "initializer_list":
                        form, val = "braced", nxt
                    elif txt(src, nxt) == "=" and ki + 2 < len(ks):
                        form, val = ("braced", ks[ki + 2]) if kind(ks[ki + 2]) == "initializer_list" else ("copy", ks[ki + 2])
                fields, copied = {}, []
                reason = []
                if form == "braced":
                    init_fields(src, val, fields, copied)
                elif form == "copy":
                    reason.append("copy-init")
                elif form == "ctor":
                    reason.append("ctor-args")
                # later writes in the same scope, up to the next declaration of the same name
                if k == "declaration" and fnode is not None and not err:
                    blk = scope_block(n)
                    if blk is not None:
                        stop = eb(blk)
                        for sib in kids(blk):
                            if sb(sib) > eb(n) and kind(sib) == "declaration":
                                st = fld(sib, "type")
                                if st is not None and base_type(txt(src, st)) in value_types:
                                    for sd in kids(sib):
                                        nm2, _, _ = declarator_info(src, sd)
                                        if nm2 == name:
                                            stop = min(stop, sb(sib))
                        for root_name, last, chain, rhs, an in assignments_in(h, src, blk):
                            if root_name != name or not (eb(n) <= sb(an) < stop):
                                continue
                            if last in STRUCT_FIELDS:
                                if rhs is not None and kind(rhs) == "initializer_list":
                                    init_fields(src, rhs, fields, copied)
                                else:
                                    copied.append(last)
                            elif last in TRACKED:
                                fields.setdefault(last, []).append((classify(src, last, rhs), re.sub(r"\s+", " ", txt(src, rhs))[:80]))
                for c in copied:
                    reason.append(c + "-copied" if c != "positional" else "positional-init")
                if k == "field_declaration" and form == "braced" and not fields and not reason:
                    # `T x{}` in a holder struct is storage whose value is set elsewhere (clause (c));
                    # only a member initialiser that carries a flag is a construction of its own
                    continue
                constructions.append({
                    "key": key_for(func, kind(n), name, n), "func": func, "var": name, "line": line_of(n),
                    "kind": "member" if k == "field_declaration" else "decl", "type": base_type(txt(src, t)),
                    "form": form, "err": err, "fields": fields, "deferred": sorted(set(reason)),
                })
        elif k == "compound_literal_expression":
            t = fld(n, "type")
            if t is None or base_type(txt(src, t)) not in value_types:
                continue
            func, _ = enclosing(src, n)
            fields, copied = {}, []
            val = fld(n, "value")
            if val is not None and kind(val) == "initializer_list":
                init_fields(src, val, fields, copied)
            constructions.append({
                "key": key_for(func, kind(n), "<temp %s>" % base_type(txt(src, t)), n), "func": func, "var": "<temp>",
                "line": line_of(n), "kind": "temp", "type": base_type(txt(src, t)), "form": "braced",
                "err": has_error_ancestor(n), "fields": fields,
                "deferred": sorted(set("positional-init" if c == "positional" else c + "-copied" for c in copied)),
            })
        elif k == "call_expression":
            f = fld(n, "function")
            ft = txt(src, f) if f is not None else ""
            last = re.sub(r"<.*", "", re.split(r"::|\.|->", ft)[-1]).strip()
            if last in RAW_CALLS:
                func, _ = enclosing(src, n)
                raws.append({"func": func, "name": last, "line": line_of(n), "form": "call",
                             "nodekind": "call_expression", "text": ft, "err": has_error_ancestor(n)})
        elif k == "string_literal":
            s = txt(src, n)
            for r in RAW_STRINGS:
                if r in s:
                    func, _ = enclosing(src, n)
                    raws.append({"func": func, "name": r, "line": line_of(n), "form": "string",
                                 "nodekind": "string_literal", "text": s, "err": has_error_ancestor(n)})
        elif k in ("preproc_def", "preproc_function_def"):
            body = fld(n, "value")
            nm = fld(n, "name")
            if body is not None and nm is not None:
                b = txt(src, body)
                for r in RAW_CALLS:
                    if re.search(r"(?<![A-Za-z0-9_])" + re.escape(r) + r"\s*(?:<[^>]*>)?\s*\(", b):
                        raws.append({"func": "#define " + txt(src, nm), "name": r, "line": line_of(n),
                                     "form": "macro", "nodekind": "preproc", "text": r, "err": False})
        if k in ("type_identifier", "identifier") and txt(src, n) in all_types:
            if has_error_ancestor(n) or _a(n, "is_missing"):
                func, _ = enclosing(src, n)
                errtoks.append({"func": func, "tok": txt(src, n), "line": line_of(n)})

    clean = blank_comments_strings(src, root)
    lexical = []
    tok_re = re.compile(rb"\b(?:" + "|".join(sorted(all_types)).encode() + rb")\b")
    lex_count = len(tok_re.findall(clean))
    ast_count = sum(1 for n in walk(root) if kind(n) in ("type_identifier", "identifier") and txt(src, n) in all_types)
    if root_error:
        lexical = lexical_scan(src, clean, value_types)
        for r in RAW_CALLS:
            for m in re.finditer(rb"(?<![A-Za-z0-9_])" + re.escape(r.encode()) + rb"\s*(?:<[^>]*>)?\s*\(", clean):
                raws.append({"func": "<lexical>", "name": r, "line": clean.count(b"\n", 0, m.start()) + 1,
                             "form": "lexical", "nodekind": "lexical", "text": r, "err": True})
    return {"constructions": constructions, "raws": raws, "errtoks": errtoks, "funcs": funcs,
            "root_error": root_error, "lexical": lexical, "lex_count": lex_count, "ast_count": ast_count}


def blank_comments_strings(src, root):
    b = bytearray(src)
    for n in walk(root):
        if kind(n) in ("comment", "string_literal", "raw_string_literal", "char_literal"):
            for i in range(sb(n), eb(n)):
                if b[i] != 10:
                    b[i] = 32
    return bytes(b)


def lexical_scan(src, clean, value_types):
    """Request declarations found by text alone, for a file whose tree cannot be trusted. The
    statement run of a hit is the text to the next declaration of the same name."""
    text = clean.decode("utf-8", "replace")
    alt = "|".join(sorted(value_types))
    pat = re.compile(r"\b(?:" + alt + r")\b(?:\s*[&*])?\s+(\w+)\s*(?=[;{=(,)\[])")
    rows = []
    ordinal = {}
    for m in pat.finditer(text):
        name = m.group(1)
        tail = text[m.start():]
        nxt = re.search(r"\b(?:" + alt + r")\s+" + re.escape(name) + r"\b", tail[len(m.group(0)):])
        body = tail[: len(m.group(0)) + nxt.start()] if nxt else tail[:20000]
        host = bool(re.search(r"\bmust_host_pinned\s*=\s*true\b", body))
        dev = bool(re.search(r"\bmust_device\s*=\s*true\b", body))
        i = ordinal.get(name, 0)
        ordinal[name] = i + 1
        rows.append({"var": name, "key": "<lexical>::lexical:%s#%d" % (name, i), "line": text.count("\n", 0, m.start()) + 1,
                     "host_only": host and not dev})
    return rows


def facts_for(rel, src, value_types, all_types):
    key = (_sha(src), tuple(sorted(value_types)), tuple(sorted(all_types)))
    if key not in _FACTS:
        _FACTS[key] = scan_file(rel, src, value_types, all_types)
    return _FACTS[key]


# ---------------------------------------------------------------- rules

class V:
    __slots__ = ("code", "key", "file", "line", "func", "name", "msg")

    def __init__(self, code, key, file, line, func, name, msg):
        self.code, self.key, self.file, self.line, self.func, self.name, self.msg = code, key, file, line, func, name, msg

    def ident(self):
        return (self.code, self.key)

    def __str__(self):
        return "%s %s:%d %s: %s" % (self.code, self.file, self.line, self.key, self.msg)


def judge(rel, c):
    """Violations of one AST construction, as (code, message)."""
    out = []
    if c["err"]:
        return [("A-ERROR", "%s sits inside an ERROR region of the parse; nothing about it can be trusted" % c["type"])]
    if c["form"] == "plain":
        out.append(("B-BRACE", "braceless `%s %s;` reports the class definition as its construction site; write `%s %s{}`"
                    % (c["type"], c["var"], c["type"], c["var"])))
    if c["deferred"]:
        out.append(("DEFER-C", "built from a copy or call (%s); clause (c) will follow it" % ",".join(c["deferred"])))
        return out
    f = c["fields"]
    md = [x[0] for x in f.get("must_device", [])]
    mh = [x[0] for x in f.get("must_host_pinned", [])]
    dev, host = "true" in md, "true" in mh
    if dev and host:
        out.append(("B-TIER", "both must_device and must_host_pinned are set literally true; the tier is not decidable"))
        return out
    if not dev and not host:
        out.append(("B-TIER", "neither must_device nor must_host_pinned is a literal true; unified_select_tier may turn it into HOST"))
        return out
    if host:
        return out
    zones = [x[0] for x in f.get("prefer_vram_zone", [])]
    if "count" in zones:
        out.append(("D-ZONE-COUNT", "a prefer_vram_zone write names COUNT (arena bypass)"))
    elif not zones:
        out.append(("D-ZONE", "device request with no prefer_vram_zone"))
    elif any(z != "ok" for z in zones):
        out.append(("D-ZONE", "prefer_vram_zone is not a literal non-COUNT enumerator or an all-literal ternary"))
    fb = [x[0] for x in f.get("forbid_vram_zone_spill", [])]
    if "false" in fb:
        out.append(("D-FORBID-FALSE", "forbid_vram_zone_spill is written false"))
    elif "true" not in fb:
        out.append(("D-FORBID", "device request does not set forbid_vram_zone_spill = true literally"))
    return out


def analyse(files):
    """Every finding in the tree, before any allowlist or debt is applied."""
    value_types, all_types = type_closure(files)
    viols = []
    stats = {"constructions": 0, "files": len(files), "value_types": sorted(value_types), "all_types": sorted(all_types)}
    funcs = {}
    ordinal = {}
    for rel in sorted(files):
        src = files[rel]
        fa = facts_for(rel, src, value_types, all_types)
        funcs[rel] = set(fa["funcs"])
        for c in fa["constructions"]:
            stats["constructions"] += 1
            for code, msg in judge(rel, c):
                viols.append(V(code, c["key"], rel, c["line"], c["func"], c["var"], msg))
        seen = {}
        for t in fa["errtoks"]:
            i = seen.get((t["func"], t["tok"]), 0)
            seen[(t["func"], t["tok"])] = i + 1
            viols.append(V("A-ERROR", "%s::%s::token:%s#%d" % (rel, t["func"], t["tok"], i), rel, t["line"], t["func"], t["tok"],
                           "request-type token %s inside an ERROR/MISSING region" % t["tok"]))
        for r in fa["lexical"]:
            if not r["host_only"]:
                viols.append(V("A-LEXICAL", "%s::%s" % (rel, r["key"]), rel, r["line"], "<lexical>", r["var"],
                               "lexical hit in a file whose root is ERROR is not host-only by a literal"))
        if fa["lex_count"] != fa["ast_count"]:
            viols.append(V("A-TOKEN", "%s::<tokens>::lexical-vs-ast" % rel, rel, 0, "<file>", "",
                           "request-type tokens by text (%d) differ from the parse's (%d)" % (fa["lex_count"], fa["ast_count"])))
        seen = {}
        for r in fa["raws"]:
            base = "%s::%s::%s:%s:%s" % (rel, r["func"], r["nodekind"], r["name"], text_hash(r["text"]))
            i = seen.get(base, 0)
            seen[base] = i + 1
            viols.append(V("E-RAW", "%s#%d" % (base, i), rel, r["line"], r["func"], r["name"],
                           "raw allocator name %s (%s) outside the allowlist" % (r["name"], r["form"])))
    return viols, stats, funcs


# ---------------------------------------------------------------- allowlist and debt

def load_json(path, default):
    if not Path(path).exists():
        return default
    with open(path) as f:
        return json.load(f)


def apply_contract(viols, allowlist, debt):
    """(failures, report lines). Allowlist first, then the shrink-only debt."""
    fails, report = [], []
    remaining = list(viols)
    for ent in allowlist.get("entries", []):
        eid = ent["id"]
        hit = [v for v in remaining
               if v.code == ent["code"] and v.file == ent["file"] and v.func == ent["function"]
               and ("name" not in ent or v.name == ent["name"])]
        report.append("allowlist %-28s covers %d (count %s)" % (eid, len(hit), ent.get("count")))
        if not hit:
            fails.append("FAIL allowlist entry %s matches nothing (%s %s::%s); a stale or renamed exemption "
                         "protects nothing" % (eid, ent["code"], ent["file"], ent["function"]))
            continue
        if "count" in ent and ent["count"] != len(hit):
            fails.append("FAIL allowlist entry %s covers %d finding(s) but pins %d; a new one in an exempt "
                         "function is not exempt" % (eid, len(hit), ent["count"]))
        for v in hit:
            remaining.remove(v)
    debt_ids = {(d["code"], d["key"]) for d in debt.get("violations", [])}
    for d in debt.get("violations", []):
        if d["code"] == "E-RAW" and not FATE_RE.match(str(d.get("fate", ""))):
            fails.append("FAIL debt entry E-RAW %s has no valid fate (deleted-by-*, converted-by-*, sanctioned-internal)" % d["key"])
    cur = {v.ident(): v for v in remaining}
    for ident, v in sorted(cur.items()):
        if ident not in debt_ids:
            fails.append("FAIL new %s" % v)
    for ident in sorted(debt_ids - set(cur)):
        fails.append("FAIL stale debt entry %s %s no longer violates; delete it from debt.json" % ident)
    report.append("debt %d entries; %d remaining findings" % (len(debt_ids), len(cur)))
    return fails, report


def run_gate(files, allowlist, debt, quiet=False):
    viols, stats, funcs = analyse(files)
    fails, report = apply_contract(viols, allowlist, debt)
    return fails, report, viols, stats


# ---------------------------------------------------------------- mutation matrix

PLANT = "zz-plant.cpp"
REQ = "ggml_sycl::alloc_request"


def plant(body):
    return lambda files: dict(files, **{PLANT: body.encode()})


def append_to(rel, body):
    return lambda files: dict(files, **{rel: files[rel] + ("\n" + body + "\n").encode()})


def replace_token(rel, old, new):
    def f(files):
        src = files[rel].decode()
        out, n = re.subn(r"(?<![A-Za-z0-9_])" + re.escape(old) + r"(?![A-Za-z0-9_])", new, src)
        if n == 0:
            raise SystemExit("matrix setup error: %s not found in %s" % (old, rel))
        return dict(files, **{rel: out.encode()})
    return f


def good_device(name, extra=""):
    return ("void %s() {\n    %s req{};\n    req.intent.constraints.must_device = true;\n"
            "    req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME;\n"
            "    req.intent.constraints.forbid_vram_zone_spill = true;\n%s}\n" % (name, REQ, extra))


class Case:
    def __init__(self, wid, label, mutate, expect, code=None, naming=None, allowlist=None, edit_allowlist=None,
                 edit_debt=None):
        self.wid, self.label, self.mutate, self.expect = wid, label, mutate, expect
        self.code, self.naming, self.allowlist = code, naming, allowlist
        self.edit_allowlist, self.edit_debt = edit_allowlist, edit_debt


def matrix_cases():
    c = []
    A = c.append
    # 1: a device request with no zone
    A(Case("1", "device request with a zone, forbid and tier (control)", plant(good_device("zzplant_w1")), "PASS"))
    A(Case("1", "must_device request with no zone", plant(
        "void zzplant_w1() {\n    %s req{};\n    req.intent.constraints.must_device = true;\n"
        "    req.intent.constraints.forbid_vram_zone_spill = true;\n}\n" % REQ), "FAIL", "D-ZONE", "zzplant_w1"))
    # 2: forbid removed
    A(Case("2", "forbid_vram_zone_spill removed", plant(
        "void zzplant_w2() {\n    %s req{};\n    req.intent.constraints.must_device = true;\n"
        "    req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME;\n}\n" % REQ),
        "FAIL", "D-FORBID", "zzplant_w2"))
    A(Case("2", "forbid_vram_zone_spill written false", plant(good_device(
        "zzplant_w2", "    req.intent.constraints.forbid_vram_zone_spill = false;\n")), "FAIL", "D-FORBID-FALSE", "zzplant_w2"))
    # 3: ? COUNT : ONEDNN
    A(Case("3", "oneDNN scratch zone ternary with a COUNT arm", plant(
        "void zzplant_w3(bool flag) {\n    %s req{};\n    req.intent.constraints.must_device = true;\n"
        "    req.intent.constraints.prefer_vram_zone = flag ? ggml_sycl::vram_zone_id::COUNT : ggml_sycl::vram_zone_id::ONEDNN;\n"
        "    req.intent.constraints.forbid_vram_zone_spill = true;\n}\n" % REQ), "FAIL", "D-ZONE-COUNT", "zzplant_w3"))
    # 4: COUNT retry after a zoned request
    A(Case("4", "prefer_vram_zone = COUNT retry after a zoned request", plant(good_device(
        "zzplant_w4", "    req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::COUNT;\n")),
        "FAIL", "D-ZONE-COUNT", "zzplant_w4"))
    # 5: raw allocator in a new function
    A(Case("5", "unified_cache_malloc_device_tracked called from a new function", plant(
        "void zzplant_w5() {\n    void * p = unified_cache_malloc_device_tracked(1, 2, 3);\n    (void) p;\n}\n"),
        "FAIL", "E-RAW", "zzplant_w5"))
    A(Case("5", "zeMemAllocDevice named in a string literal", plant(
        "void zzplant_w5s() {\n    const char * s = \"zeMemAllocDevice\";\n    (void) s;\n}\n"),
        "FAIL", "E-RAW", "zzplant_w5s"))
    A(Case("5", "raw name in a #define body", plant(
        "#define ZZ_RAW(p) unified_cache_raw_malloc_device(p)\n"), "FAIL", "E-RAW", "ZZ_RAW"))
    # 6: comments and strings are not code
    A(Case("6", "comment and string naming must_device = true and a raw call", plant(
        "// %s r{}; r.intent.constraints.must_device = true; unified_cache_malloc_device_tracked(1);\n"
        "/* %s q{}; q.intent.constraints.must_device = true; */\n"
        "void zzplant_w6() {\n    const char * s = \"x.intent.constraints.must_device = true;\";\n    (void) s;\n"
        "    %s req{};\n    req.intent.constraints.must_host_pinned = true;\n}\n" % (REQ, REQ, REQ)), "PASS"))
    A(Case("6", "the same text outside the comment is code and FAILs (control)", plant(
        "void zzplant_w6x() {\n    %s r{};\n    r.intent.constraints.must_device = true;\n"
        "    unified_cache_malloc_device_tracked(1);\n}\n" % REQ), "FAIL", "D-ZONE", "zzplant_w6x"))
    # 7: rename an exempt function
    A(Case("7", "E-ARENA-BACKING's function renamed", replace_token("unified-cache.cpp", "arena_reserve", "arena_reserve_zz"),
        "FAIL", "E-RAW", "arena_reserve_zz"))
    A(Case("7", "E-ARENA-BACKING's entry reports the rename", replace_token("unified-cache.cpp", "arena_reserve", "arena_reserve_zz"),
        "FAIL", "allowlist", "E-ARENA-BACKING"))
    A(Case("7", "E-TEST's function renamed", replace_token("unified-cache.cpp", "ensure_cached_alloc", "ensure_cached_alloc_zz"),
        "FAIL", "E-RAW", "ensure_cached_alloc_zz"))
    # 8: an allowlist entry that matches nothing
    A(Case("8", "allowlist entry matching nothing", lambda f: f, "FAIL", "allowlist", "E-ZZ-BOGUS",
           allowlist={"id": "E-ZZ-BOGUS", "code": "E-RAW", "file": "unified-cache.cpp", "function": "no_such_function", "count": 1}))
    A(Case("8", "allowlist entry pinning a count its function does not have", lambda f: f, "FAIL", "allowlist", "E-TEST",
           edit_allowlist=lambda al: dict(al, entries=[dict(e, count=3) if e["id"] == "E-TEST" else e for e in al["entries"]])))
    # keys: an unrelated edit above a listed construction must not move its key
    A(Case("key", "lines and a function added above listed constructions leave every key in place",
           lambda f: dict(f, **{"common.cpp": b"// unrelated\n\nstatic int zz_unrelated_above() { return 1; }\n\n" + f["common.cpp"]}), "PASS"))
    A(Case("key", "editing a listed construction itself moves its key (stale entry plus new violation)",
           replace_token("common.cpp", "ggml_sycl_tp_ensure_ffn_buffers", "ggml_sycl_tp_ensure_ffn_buffers_zz"),
           "FAIL", "debt", "ggml_sycl_tp_ensure_ffn_buffers"))
    # (a) a pointer holder cannot launder the request it points at
    A(Case("1s", "a scope over a violating request still FAILs at the request's construction", plant(
        "void zzplant_w1s() {\n    %s req{};\n    req.intent.constraints.must_device = true;\n"
        "    req.intent.constraints.forbid_vram_zone_spill = true;\n    ext_alloc_request_scope scope(&req);\n}\n" % REQ),
        "FAIL", "D-ZONE", "zzplant_w1s::declaration:req:"))
    A(Case("1s", "a scope over a clean request adds no finding of its own (control)", plant(
        good_device("zzplant_w1s", "    ext_alloc_request_scope scope(&req);\n")), "PASS"))
    A(Case("fate", "an E-RAW debt entry without a fate fails", lambda f: f, "FAIL", "fate", "unified_alloc",
           edit_debt=lambda d: dict(d, violations=[{k: v for k, v in e.items() if k != "fate"} if (e["code"] == "E-RAW" and "::unified_alloc::" in e["key"]) else e
                                                   for e in d["violations"]])))
    # debt: the list is shrink-only in both directions
    A(Case("debt", "a debt entry that no longer violates is stale", lambda f: f, "FAIL", "debt", "zzgone",
           edit_debt=lambda d: dict(d, violations=list(d["violations"]) + [{"code": "D-ZONE", "key": "x.cpp::f::zzgone#0"}])))
    A(Case("debt", "a debt entry deleted makes its violation new", lambda f: f, "FAIL", "D-ZONE", "graph_input_stage",
           edit_debt=lambda d: dict(d, violations=[v for v in d["violations"]
                                                   if not (v["code"] == "D-ZONE" and "graph_input_stage" in v["key"])])))
    # 12: no tier flag
    A(Case("12", "request with no tier flag", plant(
        "void zzplant_w12() {\n    %s req{};\n    req.size = 1;\n}\n" % REQ), "FAIL", "B-TIER", "zzplant_w12"))
    A(Case("12", "request with a literal host tier (control)", plant(
        "void zzplant_w12() {\n    %s req{};\n    req.intent.constraints.must_host_pinned = true;\n}\n" % REQ), "PASS"))
    A(Case("12", "tier written as a non-literal expression", plant(
        "void zzplant_w12(bool b) {\n    %s req{};\n    req.intent.constraints.must_device = b;\n}\n" % REQ),
        "FAIL", "B-TIER", "zzplant_w12"))
    # 13: a request inside a broken #if region
    A(Case("13", "request inside a broken #if region", plant(
        "void zzplant_w13(int a) {\n#if defined(ZZ_A)\n    if (a) {\n#else\n    if (a > 1) {\n#endif\n"
        "        %s req{ .intent = { .constraints = { .must_device = true } };\n    }\n}\n" % REQ),
        "FAIL", "A-ERROR", "zz-plant.cpp"))
    # 14: ternary arms
    A(Case("14", "all-literal non-COUNT ternary (u1bb shape, control)", plant(
        "void zzplant_w14(bool kv) {\n    %s req{};\n    req.intent.constraints.must_device = true;\n"
        "    req.intent.constraints.prefer_vram_zone = kv ? ggml_sycl::vram_zone_id::KV : ggml_sycl::vram_zone_id::RUNTIME;\n"
        "    req.intent.constraints.forbid_vram_zone_spill = true;\n}\n" % REQ), "PASS"))
    A(Case("14", "ternary with one COUNT arm", plant(
        "void zzplant_w14(bool kv) {\n    %s req{};\n    req.intent.constraints.must_device = true;\n"
        "    req.intent.constraints.prefer_vram_zone = kv ? ggml_sycl::vram_zone_id::KV : ggml_sycl::vram_zone_id::COUNT;\n"
        "    req.intent.constraints.forbid_vram_zone_spill = true;\n}\n" % REQ), "FAIL", "D-ZONE-COUNT", "zzplant_w14"))
    A(Case("14", "ternary with a non-literal arm", plant(
        "void zzplant_w14(bool kv, ggml_sycl::vram_zone_id z) {\n    %s req{};\n    req.intent.constraints.must_device = true;\n"
        "    req.intent.constraints.prefer_vram_zone = kv ? ggml_sycl::vram_zone_id::KV : z;\n"
        "    req.intent.constraints.forbid_vram_zone_spill = true;\n}\n" % REQ), "FAIL", "D-ZONE", "zzplant_w14"))
    # 18: offload_buffer_request needs a literal tier too
    OBR = "ggml_sycl::offload_buffer_request"
    A(Case("18", "offload_buffer_request with no literal tier", plant(
        "void zzplant_w18() {\n    %s req{};\n    req.size = 1;\n}\n" % OBR), "FAIL", "B-TIER", "zzplant_w18"))
    A(Case("18", "offload_buffer_request with a literal host tier (control)", plant(
        "void zzplant_w18() {\n    %s req{};\n    req.intent.constraints.must_host_pinned = true;\n}\n" % OBR), "PASS"))
    # 19: the ERROR-root file is scanned lexically
    A(Case("19", "zone-less device request planted in cpu-dispatch.cpp's ERROR root", append_to("cpu-dispatch.cpp",
        "void zzplant_w19() {\n    %s zzreq{};\n    zzreq.intent.constraints.must_device = true;\n}\n" % REQ),
        "FAIL", "A-LEXICAL", "zzreq"))
    A(Case("19", "host-only request planted in the same file (control)", append_to("cpu-dispatch.cpp",
        "void zzplant_w19() {\n    %s zzreq{};\n    zzreq.intent.constraints.must_host_pinned = true;\n}\n" % REQ), "PASS"))
    # 20: a member initialiser constructs a request
    A(Case("20", "member initialiser constructing a zone-less device request", plant(
        "struct zzplant_w20 {\n    %s req{ .intent = { .constraints = { .must_device = true, .forbid_vram_zone_spill = true } } };\n};\n" % REQ),
        "FAIL", "D-ZONE", "zzplant_w20"))
    A(Case("20", "member initialiser with zone, forbid and tier (control)", plant(
        "struct zzplant_w20 {\n    %s req{ .intent = { .constraints = { .must_device = true, "
        ".prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME, .forbid_vram_zone_spill = true } } };\n};\n" % REQ), "PASS"))
    # 23: braceless declarations
    A(Case("23", "braceless request at class-member scope", plant(
        "struct zzplant_w23 {\n    %s req;\n};\n" % REQ), "FAIL", "B-BRACE", "zzplant_w23"))
    A(Case("23", "braceless request at namespace scope", plant(
        "namespace zzplant_w23ns {\n%s req;\n}\n" % REQ), "FAIL", "B-BRACE", "zz-plant.cpp::<file scope>::declaration:req:"))
    A(Case("23", "braceless request in a function", plant(
        "void zzplant_w23f() {\n    %s req;\n    req.intent.constraints.must_host_pinned = true;\n}\n" % REQ),
        "FAIL", "B-BRACE", "zzplant_w23f"))
    A(Case("23", "braced host-tier request at class-member scope (control)", plant(
        "struct zzplant_w23 {\n    %s req{ .intent = { .constraints = { .must_host_pinned = true } } };\n};\n" % REQ), "PASS"))
    return c


REQUIRED_WITNESSES = ("1", "1s", "fate", "2", "3", "4", "5", "6", "7", "8", "debt", "key", "12", "13", "14", "18", "19", "20", "23")


def planted_sightings(files):
    """Constructions and lexical rows the gate sees in the planted function or appended text."""
    value_types, all_types = type_closure(files)
    n = 0
    for rel in (PLANT, "cpu-dispatch.cpp"):
        if rel in files:
            fa = facts_for(rel, files[rel], value_types, all_types)
            n += sum(1 for c in fa["constructions"] if "zz" in c["key"] or "zz" in c["func"])
            n += sum(1 for r in fa["lexical"] if r["var"].startswith("zz"))
    return n


def evaluate_case(base_files, allowlist, debt, case):
    files = case.mutate(base_files)
    if case.expect == "PASS" and case.wid in ("1", "1s", "6", "12", "14", "18", "19", "20", "23") and planted_sightings(files) == 0:
        return False, ["control saw no planted construction, so its PASS proves nothing"]
    al = allowlist
    if case.allowlist is not None:
        al = dict(allowlist, entries=list(allowlist.get("entries", [])) + [case.allowlist])
    if case.edit_allowlist is not None:
        al = case.edit_allowlist(al)
    if case.edit_debt is not None:
        debt = case.edit_debt(debt)
    fails, _, _, _ = run_gate(files, al, debt)
    if case.expect == "PASS":
        return (not fails), fails
    if not fails:
        return False, []

    def names(f):
        if case.code == "allowlist":
            return "allowlist entry" in f and case.naming in f
        if case.code == "debt":
            return f.startswith("FAIL stale debt entry") and case.naming in f
        if case.code == "fate":
            return f.startswith("FAIL debt entry E-RAW") and "no valid fate" in f and case.naming in f
        return f.startswith("FAIL new " + (case.code or "")) and case.naming in f

    hit = [f for f in fails if names(f)]
    return bool(hit), fails


def f_witness_missing_pack():
    """Clause (f): with the pack unimportable the gate exits 1 and says why."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        Path(d, "tree_sitter_language_pack.py").write_text("raise ImportError('simulated: pack absent')\n")
        env = dict(os.environ, PYTHONPATH=d)
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--root", "/nonexistent"],
                           env=env, capture_output=True, text=True)
    return p.returncode == 1 and "tree_sitter_language_pack" in p.stdout, "rc=%d out=%r" % (p.returncode, p.stdout[:120])


def run_matrix(files, allowlist, debt):
    fails, report, _, _ = run_gate(files, allowlist, debt)
    if fails:
        print("FAIL: the unmutated baseline is red, so a matrix scored against it proves nothing:")
        for f in fails[:20]:
            print("  " + f)
        return 1
    print("baseline: PASS (the matrix is scored against a green tree)")
    bad = 0
    cases = matrix_cases()
    for w in REQUIRED_WITNESSES:
        if not any(c.wid == w and c.expect == "FAIL" for c in cases):
            print("FAIL: witness %s has no FAIL case" % w)
            bad += 1
    for case in cases:
        ok, got = evaluate_case(files, allowlist, debt, case)
        print("%s witness %-3s %-72s expect %s" % ("ok  " if ok else "FAIL", case.wid, case.label, case.expect))
        if not ok:
            bad += 1
            for g in got[:4]:
                print("       got: " + g)
            if not got:
                print("       got: PASS")
    ok, detail = f_witness_missing_pack()
    print("%s witness f   missing tree_sitter_language_pack exits 1 (%s)" % ("ok  " if ok else "FAIL", detail))
    bad += 0 if ok else 1
    n = len(cases) + 1
    print("matrix: %d case(s), %d wrong" % (n, bad))
    return 1 if bad else 0


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--data", default=None, help="directory holding allowlist.json and debt.json")
    ap.add_argument("--mutation-matrix", action="store_true")
    ap.add_argument("--list", action="store_true", help="print every finding before the allowlist and debt")
    ap.add_argument("--write-debt", action="store_true",
                    help="rewrite debt.json from the current tree; refuses to add entries. Never used by the ctest")
    ap.add_argument("--allow-growth", action="store_true", help="with --write-debt, permit new entries (key migration only)")
    a = ap.parse_args()
    data = Path(a.data) if a.data else Path(a.root) / "scripts" / "sycl-alloc-zone-contract"
    allowlist = load_json(data / "allowlist.json", {"entries": []})
    debt = load_json(data / "debt.json", {"violations": []})
    files = load_tree(a.root)
    if a.list or a.write_debt:
        viols, stats, _ = analyse(files)
        if a.list:
            for v in viols:
                print(v)
        if a.write_debt:
            rest = list(viols)
            for ent in allowlist.get("entries", []):
                rest = [v for v in rest if not (v.code == ent["code"] and v.file == ent["file"] and v.func == ent["function"]
                                                and ("name" not in ent or v.name == ent["name"]))]
            ids = sorted({v.ident() for v in rest})
            old_meta = {(d["code"], d["key"]): {k: d[k] for k in ("fate", "cite") if k in d} for d in debt.get("violations", [])}
            old_ids = {(d["code"], d["key"]) for d in debt.get("violations", [])}
            grown = sorted(set(ids) - old_ids)
            if grown and old_ids and not a.allow_growth:
                print("FAIL: --write-debt would ADD %d entr%s; debt may only shrink. Fix the violation, or pass "
                      "--allow-growth for a deliberate key migration:" % (len(grown), "y" if len(grown) == 1 else "ies"))
                for g in grown[:20]:
                    print("  %s %s" % g)
                return 1
            data.mkdir(parents=True, exist_ok=True)
            with open(data / "debt.json", "w") as f:
                f.write('{\n  "schema": 1,\n  "_doc": %s,\n  "violations": [\n' % json.dumps(DEBT_DOC))
                f.write(",\n".join("    " + json.dumps(dict({"code": c, "key": k}, **old_meta.get((c, k), {}))) for c, k in ids))
                f.write("\n  ]\n}\n")
            print("wrote %d debt entries to %s" % (len(ids), data / "debt.json"))
        return 0
    fails, report, viols, stats = run_gate(files, allowlist, debt)
    print("scope: %d files, %d constructions of %s" % (stats["files"], stats["constructions"], " ".join(stats["value_types"])))
    for r in report:
        print(r)
    by_code = {}
    for d in debt.get("violations", []):
        by_code[d["code"]] = by_code.get(d["code"], 0) + 1
    print("debt by code: " + " ".join("%s=%d" % (k, by_code[k]) for k in sorted(by_code)))
    if fails:
        for f in fails:
            print(f)
        print("FAIL: %d finding(s)" % len(fails))
        return 1
    print("PASS: no new violation, no stale debt, no stale allowlist entry")
    if a.mutation_matrix:
        return run_matrix(files, allowlist, debt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
