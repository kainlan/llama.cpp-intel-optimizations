#!/usr/bin/env python3
"""Source gate for the SYCL allocation-zone contract (llama.cpp-23mk, design section 7).

Every construction of a request type must say, by literal, which tier it asks for, and a device
request must name its arena zone and forbid the spill. Raw allocator names stay outside the
unified-cache's own allowlisted functions. The gate parses the tree with tree-sitter, so a
comment or a string literal never counts as code, and it keys every finding by AST node
(`file::function::node-kind:variable:text-hash#ordinal`), never by line or byte offset. The text
hash covers the construction's declaration and the field writes bound to it, and the ordinal
counts only identical constructions in the same function, so a key survives an unrelated edit
above it and moves only when the construction itself changes.

What this unit (S2a) enforces
  (a) a request-type token inside an ERROR/MISSING region fails; a file whose root is ERROR
      (today cpu-dispatch.cpp) is also scanned lexically, and each lexical request must be
      host-only by a literal on its statement run
  (b) a literal tier flag: must_device = true or must_host_pinned = true, not both, and the last
      write of the flag must be that literal
  (d) a device request carries a prefer_vram_zone that is a literal non-COUNT enumerator, or a
      ternary whose arms all are, and forbid_vram_zone_spill = true, and the last write of each
      is that literal. A write nested under a conditional below the construction's own block
      does not establish a flag (fail closed), though it can break one.
  (e) raw allocator names (as a call, an identifier, a string or raw-string literal, or in a
      #define body) outside the allowlist
  (f) a missing tree_sitter_language_pack is a FAIL, never a skip
and the brace rule the construction-site labels need: a braceless `T x;` reports the class
site (unified-cache.hpp, the comment above alloc_intent), so every value declaration of a
request type is written `T x{}`.

Fail closed on form. Every occurrence of a request-type token is given a syntactic role:
parameter, return, pointer or reference, a request type's own member, definition, alias,
friend, scope qualifier, sizeof/cast over a pointer, or construction. An occurrence in any other
role (an array, a `new`, a template argument such as std::vector<alloc_request>, a base class, a
default argument `T r = {}`, ...) is a B-FORM finding, so a construction form the gate cannot follow is
listed, not missed. An `using R = alloc_request;` or `typedef alloc_request RT;` extends the closure, so
`R r{}` is judged. Constructions that carry no request-type token are found by shape: a braced-init-list
returned from a request-returning function or passed to a function that takes a request (B-FORM), and an
`auto x = <request>` copy of a request variable, parameter, member, std::move or call result (DEFER-C).
A #define body that assigns a tracked field is a B-FORM finding (macro-write).

A write only counts when it is bound to the declaration (the nearest enclosing one of that name, looking
through #if branches), comes before the request is first handed to other code (passed, returned, copied,
or named in a lambda), and is not under a conditional. A whole-object assignment (`req = {}`,
`req = other;`) is a copy and defers to clause (c). The other tier flag may only ever be written a
literal false.

Not here yet (later units, each lands with its witnesses)
  (c) interprocedural flow, so a copy / helper return / by-reference write is a construction of
      its own. Until it lands such constructions are listed as DEFER-C debt and are shrink-only,
      so a new copy cannot slip in unseen. Known clause-(c) gaps, listed so S2b closes them: a
      local reference or pointer alias written through (`alloc_constraints & c = req.intent.
      constraints; c.prefer_vram_zone = COUNT;`, `p->...`), a callee that writes a request passed by
      reference, and writes to a request held as a member, in a method or through `this->`
      (`struct H { alloc_request r{}; void f() { r.intent.constraints.prefer_vram_zone = COUNT; } };`).
      Also documented gaps: token pasting (`malloc_##x`) and `#pragma message` can name a raw
      allocator without the gate seeing it.
  (g)-(p) are S2c/S2d. Witness 9 (a site that stops calling its shared `*_bytes()` function) is
      deferred to S2d as a dormant clause: its subjects, the model-shaped exact `*_bytes()`
      functions (load_reorder_temp_bytes, woq_packed_bytes, ...), do not exist in the tree yet.

Scope
  Every .cpp/.hpp under ggml/src/ggml-sycl except the skipped directories below. dpct/ is in
  scope (helper.hpp holds raw calls and ggml-sycl.cpp includes it). A new file with another
  C/C++/OpenCL extension (.h, .hh, .inl, .cc, .c, .cl, ... matched case-insensitively) is a FAIL until
  it is scanned or explicitly skipped, a required-file list and a file-count floor pin that the walk
  did not silently shrink, and the skipped directories are the scope root's own top-level ones only
  (a nested tests/ is scanned).
    tests/               test code, outside the design's scope ("non-test ggml-sycl sources")
    docs/                markdown only; nothing to scan

Debt model
  The tree has constructions that violate the rules today. They are an explicit list
  (scripts/sycl-alloc-zone-contract/debt.json) that may only shrink: a violation not on the
  list fails, and a listed entry that no longer violates fails too (delete it in the commit
  that fixes it). The allowlist (allowlist.json) is for permanent exemptions with a reason; each
  entry names its subject, pins an exact `count`, and fails when it covers anything else.

Pointer-only holders (`ext_alloc_request_scope` holds a `const alloc_request *`) are tokens for
clause (a) but are not constructions of a request: they carry no flag a request could be
missing, and the pointee is judged at its own construction. Holders of a request by value
(`offload_buffer_request`) are constructions.

Mutation matrix (--mutation-matrix)
  Each case mutates an in-memory copy of the tree (nothing is written, so the checkout cannot be
  touched, and no case sees another's mutant) and states the verdict it must flip to. The
  unmutated baseline must PASS first, or the matrix would be scored against a red baseline. A
  site that does not conform yet cannot be mutated in place (its violation is already in the
  debt list), so a witness plants a twin: a request of the same shape in a fresh function, whose
  unmutated form must PASS and whose mutated form must FAIL naming the planted node. A PASS control
  must have seen a planted construction; the few `planted=False` controls are pure negatives, a form
  with no construction to see (a pointer parameter, a raw-named function's declaration) whose roles the
  paired FAIL cases exercise, so a no-op plant cannot hide a missing detection there.

Dependency
  Needs the tree-sitter C++ grammar: `pip install tree-sitter-language-pack` in the python3 that
  CMake finds. A missing module is a FAIL (exit 1) and the message names it; there is no skip.

Usage:
  check-sycl-alloc-zone-contract.py [--root REPO] [--mutation-matrix]
  check-sycl-alloc-zone-contract.py [--root REPO] --list
  check-sycl-alloc-zone-contract.py [--root REPO] --write-debt [--allow-growth]
"""
import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
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
SKIP_DIRS = {"tests", "docs"}   # top-level directories of the scope root only; a nested tests/ is scanned
SCAN_SUFFIXES = (".cpp", ".hpp")
# Matched case-insensitively. Any other C/C++/OpenCL/CUDA suffix is a FAIL until scanned or skipped on purpose.
CXXISH_SUFFIXES = (".h", ".hh", ".hxx", ".h++", ".cc", ".cxx", ".c++", ".c", ".inl", ".ipp", ".tpp", ".cl", ".cu",
                   ".cuh", ".cppm", ".ixx")
MIN_FILES = 330                  # 335 today; a walk that quietly shrinks below this is a FAIL
REQUIRED_FILES = ("ggml-sycl.cpp", "unified-cache.cpp", "unified-cache.hpp", "common.cpp", "common.hpp", "mmvq.cpp",
                  "dpct/helper.hpp")
BASE_TYPES = ("alloc_request", "alloc_intent", "alloc_constraints")
ZONES = ("KV", "WEIGHT", "ONEDNN", "RUNTIME", "SCRATCH")
# Fields whose literal value the rules read. Later clauses add to this set.
TRACKED = ("must_device", "must_host_pinned", "prefer_vram_zone", "forbid_vram_zone_spill",
           "cascade_step", "unconverted_ticket", "cohort_id")
# Fields of a request that are themselves structs: a write to one is a copy, not a literal.
STRUCT_FIELDS = ("intent", "constraints")
# Clause (e). Names match as identifiers (a call, an address-of, a use as a value), strings on substring.
RAW_NAMES = (
    "unified_cache_malloc_device_tracked", "unified_cache_raw_malloc_device", "sycl_aligned_malloc_device",
    "ggml_sycl_malloc_device_raw", "ggml_sycl_free_device_raw", "unified_cache_raw_free_device",
    "malloc_device", "malloc_shared", "aligned_alloc_device", "aligned_alloc_shared",
    "zeMemAllocDevice", "zeMemAllocShared", "zePhysicalMemCreate", "zeVirtualMemReserve",
)
RAW_STRINGS = ("zeMemAllocDevice", "zeMemAllocShared", "zePhysicalMemCreate", "zeVirtualMemReserve")

DEBT_DOC = ("Read by scripts/check-sycl-alloc-zone-contract.py (clauses a, b, d, e). Shrink-only: a violation not listed "
            "fails, and a listed entry that no longer violates fails. Every E-RAW entry carries a fate (deleted-by-*, "
            "converted-by-*, sanctioned-internal or pending-disposition) and a cite, so an entry no step will ever "
            "shrink is visible as a mislabelled allowlist entry. Regenerate with `python3 "
            "scripts/check-sycl-alloc-zone-contract.py --write-debt` (it refuses to add entries); see README.md.")
FATE_RE = re.compile(r"^(deleted-by|converted-by)-[A-Za-z0-9._§()-]+$|^sanctioned-internal$|^sanctioned-vendored$|^pending-disposition$")
CITE_MIN = 12   # a cite names a ticket or a design/census row; "tbd" is not one

CODES = ("A-ERROR", "A-LEXICAL", "A-TOKEN", "B-BRACE", "B-FORM", "B-TIER", "D-ZONE", "D-ZONE-COUNT",
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
def same(a, b): return a is not None and b is not None and sb(a) == sb(b) and eb(a) == eb(b) and kind(a) == kind(b)


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


def norm(s):
    s = re.sub(r"/\*.*?\*/|//[^\n]*", " ", s, flags=re.S)
    return re.sub(r"\s+", " ", s).strip()


def text_hash(s):
    """Short hash of text with comments dropped and whitespace collapsed."""
    return hashlib.sha1(norm(s).encode()).hexdigest()[:8]


def base_type(s):
    s = re.sub(r"\b(const|volatile|static|thread_local|inline|constexpr|struct)\b", "", s)
    return s.strip().split("::")[-1].strip()


# ---------------------------------------------------------------- the tree

def list_sources(root):
    """(scanned: rel -> path, unscanned C++-ish files). Sorted."""
    base = Path(root) / SCOPE_SUBDIR
    scanned, unscanned = {}, []
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(base)
        if len(rel.parts) > 1 and rel.parts[0] in SKIP_DIRS:
            continue
        suffix = p.suffix.lower()
        if suffix in SCAN_SUFFIXES:
            scanned[str(rel)] = p
        elif suffix in CXXISH_SUFFIXES:
            unscanned.append(str(rel))
    return scanned, unscanned


def load_tree(root):
    """rel path -> bytes for every in-scope source, sorted."""
    scanned, unscanned = list_sources(root)
    if unscanned:
        print("FAIL: C++-ish source(s) neither scanned nor skipped by this gate: %s; add the suffix to "
              "SCAN_SUFFIXES or skip the directory with a reason" % ", ".join(unscanned))
        sys.exit(1)
    missing = [f for f in REQUIRED_FILES if f not in scanned]
    if missing:
        print("FAIL: required source file(s) not found under %s: %s; the walk or the root is wrong"
              % (Path(root) / SCOPE_SUBDIR, ", ".join(missing)))
        sys.exit(1)
    if len(scanned) < MIN_FILES:
        print("FAIL: only %d source file(s) under %s, expected at least %d; the walk shrank or the root is wrong"
              % (len(scanned), Path(root) / SCOPE_SUBDIR, MIN_FILES))
        sys.exit(1)
    return {rel: p.read_bytes() for rel, p in scanned.items()}


_PARSE = {}      # sha1 -> root node
_STRUCTS = {}    # sha1 -> ([(struct name, [(member base type, is_value)])], [(alias, target base type, is_value)])
_FUNCS = {}      # sha1 -> file_funcs rows
_FACTS = {}      # (rel, sha1, ctx digest) -> facts dict (every fact key embeds rel, so rel is in the cache key)


def _sha(src):
    return hashlib.sha1(src).hexdigest()


def parse(src):
    h = _sha(src)
    if h not in _PARSE:
        tree = _PARSER.parse_bytes(src) if hasattr(_PARSER, "parse_bytes") else _PARSER.parse(src.decode("utf-8", "replace"))
        _PARSE[h] = _a(tree, "root_node")
    return _PARSE[h]


def declared_name(src, d):
    """Name declared by a declarator node, unwrapping init/pointer/reference/array wrappers."""
    while d is not None and kind(d) in ("init_declarator", "pointer_declarator", "reference_declarator",
                                        "array_declarator", "parenthesized_declarator"):
        nd = fld(d, "declarator")
        if nd is None:
            ids = [x for x in kids(d) if kind(x) in ("identifier", "field_identifier")]
            nd = ids[0] if ids else None
        d = nd
    return txt(src, d) if d is not None and kind(d) in ("identifier", "field_identifier") else None


def file_structs(src):
    h = _sha(src)
    if h not in _STRUCTS:
        root = parse(src)
        structs, aliases = [], []
        for n in walk(root):
            k = kind(n)
            if k in ("struct_specifier", "class_specifier"):
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
                    value = any(kind(d) in ("field_identifier", "init_declarator")
                                for d in kids(c) if not same(d, t))
                    members.append((base_type(txt(src, t)), value))
                structs.append((txt(src, nm), members))
            elif k == "alias_declaration":
                nm, t = fld(n, "name"), fld(n, "type")
                if nm is not None and t is not None:
                    aliases.append((txt(src, nm), base_type(txt(src, fld(t, "type") or t)),
                                    not any(kind(x) in ("abstract_pointer_declarator", "abstract_reference_declarator")
                                            for x in kids(t))))
            elif k == "type_definition":
                t = fld(n, "type")
                for d in kids(n):
                    if t is not None and not same(d, t) and kind(d) in ("type_identifier", "pointer_declarator",
                                                                       "reference_declarator", "array_declarator"):
                        nm = declared_name(src, d) or (txt(src, d) if kind(d) == "type_identifier" else None)
                        if nm:
                            aliases.append((nm, base_type(txt(src, t)), kind(d) == "type_identifier"))
        _STRUCTS[h] = (structs, aliases)
    return _STRUCTS[h]


def type_closure(files):
    """(value types, all request-bearing types). A struct that holds a request type by value is a
    request type; one that holds it only by pointer is a token for clause (a) but not a request. An
    alias of a request type is that type."""
    value = set(BASE_TYPES)
    allt = set(BASE_TYPES)
    changed = True
    while changed:
        changed = False
        for src in files.values():
            structs, aliases = file_structs(src)
            for name, members in structs:
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
            for name, target, is_value in aliases:
                if target in value and is_value and name not in value:
                    value.add(name)
                    allt.add(name)
                    changed = True
                elif target in allt and name not in allt:
                    allt.add(name)
                    changed = True
    return value, allt


def file_funcs(src):
    """Function declarations and definitions of one file: (name, return base type, returns by value, [param (base
    type, by value or reference)]). A constructor has no return type and is skipped."""
    h = _sha(src)
    if h not in _FUNCS:
        out = []
        for n in walk(parse(src)):
            if kind(n) != "function_declarator":
                continue
            nm = fld(n, "declarator")
            if nm is None:
                continue
            name = callee_last(txt(src, nm))
            top, byval = n, True
            q = parent(n)
            while q is not None and kind(q) in ("pointer_declarator", "reference_declarator"):
                top, byval = q, False
                q = parent(q)
            ret = None
            if q is not None and kind(q) in ("declaration", "field_declaration", "function_definition"):
                t = fld(q, "type")
                ret = base_type(txt(src, t)) if t is not None else None
            params = []
            pl = fld(n, "parameters")
            for c in (kids(pl) if pl is not None else []):
                if kind(c) not in ("parameter_declaration", "optional_parameter_declaration"):
                    continue
                t = fld(c, "type")
                if t is None:
                    continue
                d = fld(c, "declarator")
                ptr = d is not None and kind(d) == "pointer_declarator"
                params.append((base_type(txt(src, t)), not ptr))
            out.append((name, ret, byval, params))
        _FUNCS[h] = out
    return _FUNCS[h]


def callee_last(text):
    """Last component of a callee spelling: a::b.c<int>  ->  c."""
    text = re.sub(r"<[^<>]*(?:<[^<>]*>[^<>]*)*>", "", text)
    return re.split(r"::|\.|->", text)[-1].strip()


class Ctx:
    """What a scan needs to know about the whole tree, computed once: the request types, the functions that take
    or return a request by value, and the members that hold one by value."""

    def __init__(self, files):
        self.value_types, self.all_types = type_closure(files)
        self.req_funcs, self.req_returning, self.req_members = set(), set(), set()
        for src in files.values():
            for name, ret, byval, params in file_funcs(src):
                if ret in self.value_types and byval:
                    self.req_returning.add(name)
                if any(pb in self.value_types and pv for pb, pv in params):
                    self.req_funcs.add(name)
            for n in walk(parse(src)):
                if kind(n) == "field_declaration":
                    t = fld(n, "type")
                    if t is not None and base_type(txt(src, t)) in self.value_types:
                        for d in kids(n):
                            if not same(d, t) and kind(d) == "field_identifier":
                                self.req_members.add(txt(src, d))
        self.digest = hashlib.sha1(repr((sorted(self.value_types), sorted(self.all_types), sorted(self.req_funcs),
                                         sorted(self.req_returning), sorted(self.req_members))).encode()).hexdigest()


# ---------------------------------------------------------------- naming

def enclosing(src, n):
    """Name of the function holding a node. Class members get the class prefix, and a lambda is
    named after the function that holds it, so a key survives a line shift."""
    lam = False
    classes = []
    p = parent(n)
    func = None
    while p is not None:
        k = kind(p)
        if k == "lambda_expression":
            lam = True
        elif k in ("struct_specifier", "class_specifier"):
            nm = fld(p, "name")
            if nm is not None:
                classes.append(txt(src, nm))
        elif k == "function_definition" and func is None:
            d = fld(p, "declarator")
            while d is not None and kind(d) != "function_declarator":
                d = fld(d, "declarator")
            func = "?"
            if d is not None:
                nm = fld(d, "declarator")
                if nm is not None:
                    func = re.sub(r"\s+", "", txt(src, nm))
        p = parent(p)
    if func is None:
        return "<member of %s>" % classes[0] if (kind(n) == "field_declaration" and classes) else "<file scope>"
    if classes and "::" not in func:
        func = "::".join(reversed(classes)) + "::" + func
    return ("lambda in " if lam else "") + func


def in_function(n):
    p = parent(n)
    while p is not None:
        if kind(p) == "function_definition":
            return True
        p = parent(p)
    return False


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


def add_write(out, field, pos, cls, cond, text):
    out.setdefault(field, []).append((pos, cls, cond, re.sub(r"\s+", " ", text)[:80]))


def init_fields(src, lst, out, copied):
    """Fold a braced initialiser into `out` (field -> [(pos, class, conditional, text)]). A designated
    pair that names a struct member with a non-list value is a copy; a positional element is unresolvable."""
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
                add_write(out, name, sb(val), classify(src, name, val), False, txt(src, val))
        elif k in ("{", "}", ",", "comment"):
            continue
        else:
            copied.append("positional")


# ---------------------------------------------------------------- extraction

def declarator_info(src, d):
    """(name, form, value) for one declarator. form: 'plain' | 'braced' | 'copy' | 'ctor' | 'array' | 'skip'."""
    k = kind(d)
    if k in ("identifier", "field_identifier"):
        return txt(src, d), "plain", None
    if k == "array_declarator" or (k == "init_declarator" and fld(d, "declarator") is not None
                                   and kind(fld(d, "declarator")) == "array_declarator"):
        return declared_name(src, d), "array", None
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


COND_KINDS = ("if_statement", "for_statement", "for_range_loop", "while_statement", "do_statement", "switch_statement",
              "case_statement", "try_statement", "catch_clause", "conditional_expression", "lambda_expression")
PREPROC_KINDS = ("preproc_if", "preproc_ifdef", "preproc_else", "preproc_elif", "preproc_elifdef")


def is_conditional(an, blk, decl):
    """True when a conditional construct sits between the assignment and the declaration's own block.
    A preprocessor branch counts too, unless the declaration sits in that same branch."""
    p = parent(an)
    while p is not None and not same(p, blk):
        if kind(p) in COND_KINDS:
            return True
        if kind(p) in PREPROC_KINDS and not (sb(p) <= sb(decl) and eb(decl) <= eb(p)):
            return True
        p = parent(p)
    return False


def block_children(p):
    """Statements of a block in source order, looking through preprocessor branches (textual, not scopes)."""
    out = []
    for c in kids(p):
        if kind(c) in PREPROC_KINDS:
            out.extend(block_children(c))
        else:
            out.append(c)
    return out


def binding_decl(src, an, name):
    """The nearest enclosing declaration of `name` visible at an assignment: the last earlier
    declaration of that name among the direct children of each enclosing block, innermost first."""
    p = parent(an)
    while p is not None:
        if kind(p) in ("compound_statement", "translation_unit", "declaration_list", "field_declaration_list"):
            best = None
            for c in block_children(p):
                if sb(c) >= sb(an):
                    break
                if kind(c) == "declaration":
                    for d in kids(c):
                        if declared_name(src, d) == name:
                            best = c
            if best is not None:
                return best
        p = parent(p)
    return None


_ASSIGN_CACHE = {}


def assignments_in(src_key, src, block):
    """Every assignment in a block whose left side is an identifier or a field chain rooted at one:
    [(root name, last field or None for a whole-object assignment, rhs node, assignment node, is compound)]."""
    key = (src_key, sb(block), eb(block))
    if key in _ASSIGN_CACHE:
        return _ASSIGN_CACHE[key]
    out = []
    for n in walk(block):
        if kind(n) != "assignment_expression":
            continue
        op = [txt(src, c) for c in kids(n) if not _a(c, "is_named")]
        if not op or not op[0].endswith("="):
            continue
        lhs, rhs = fld(n, "left"), fld(n, "right")
        chain = []
        x = lhs
        while x is not None and kind(x) == "field_expression":
            f = fld(x, "field")
            chain.append(txt(src, f) if f is not None else "?")
            x = fld(x, "argument")
        if x is None or kind(x) != "identifier":
            continue
        out.append((txt(src, x), chain[0] if chain else None, rhs, n, op[0] != "="))
    _ASSIGN_CACHE[key] = out
    return out


_IDENT_CACHE = {}


def idents_in(src_key, src, block):
    """name -> identifier nodes of a block, in source order."""
    key = (src_key, sb(block), eb(block))
    if key not in _IDENT_CACHE:
        d = {}
        for n in walk(block):
            if kind(n) == "identifier":
                d.setdefault(txt(src, n), []).append(n)
        _IDENT_CACHE[key] = d
    return _IDENT_CACHE[key]


def is_use(src, ident, decl):
    """True when an occurrence of a request variable can hand the object, or a sub-object, to code that reads
    it later: a bare use (argument, address, return, copy source) or a member chain ending in intent/constraints.
    A scalar field read or write is not a use, and neither is the left side of an assignment. Anything inside a
    lambda that does not hold the declaration counts, since the lambda may run later."""
    q = parent(ident)
    while q is not None and kind(q) not in ("compound_statement", "translation_unit"):
        if kind(q) == "lambda_expression" and not (sb(q) <= sb(decl) and eb(decl) <= eb(q)):
            return True
        q = parent(q)
    top = ident
    p = parent(top)
    last = None
    while p is not None and kind(p) == "field_expression" and same(fld(p, "argument"), top):
        f = fld(p, "field")
        last = txt(src, f) if f is not None else "?"
        top = p
        p = parent(top)
    if last is not None and last not in STRUCT_FIELDS:
        return False
    if p is not None and kind(p) == "assignment_expression" and same(fld(p, "left"), top):
        return False
    return True


def assign_last_field(src, an):
    """The last field written by an assignment, or None for a whole-object assignment."""
    chain = []
    x = fld(an, "left")
    while x is not None and kind(x) == "field_expression":
        f = fld(x, "field")
        chain.append(txt(src, f) if f is not None else "?")
        x = fld(x, "argument")
    return chain[0] if chain else None


def request_valued(src, ctx, h, node, auto_names):
    """True when an initialiser expression evidently has a request type: a request variable or parameter, a member
    chain ending in a request member or intent/constraints, std::move of one, or a call returning a request."""
    k = kind(node)
    if k == "parenthesized_expression":
        inner = [c for c in kids(node) if _a(c, "is_named")]
        return bool(inner) and request_valued(src, ctx, h, inner[0], auto_names)
    if k == "conditional_expression":
        return any(request_valued(src, ctx, h, fld(node, f), auto_names) for f in ("consequence", "alternative")
                   if fld(node, f) is not None)
    if k == "identifier":
        name = txt(src, node)
        d = binding_decl(src, node, name)
        if d is not None:
            t = fld(d, "type")
            if t is not None and (base_type(txt(src, t)) in ctx.value_types or
                                  (kind(t) == "placeholder_type_specifier" and (name, sb(d)) in auto_names)):
                return True
        q = parent(node)
        while q is not None and kind(q) != "function_definition":
            q = parent(q)
        if q is not None:
            fd = fld(q, "declarator")
            while fd is not None and kind(fd) != "function_declarator":
                fd = fld(fd, "declarator")
            pl = fld(fd, "parameters") if fd is not None else None
            for c in (kids(pl) if pl is not None else []):
                if kind(c) in ("parameter_declaration", "optional_parameter_declaration"):
                    t, dd = fld(c, "type"), fld(c, "declarator")
                    if t is not None and base_type(txt(src, t)) in ctx.value_types and declared_name(src, dd) == name \
                            and not (dd is not None and kind(dd) == "pointer_declarator"):
                        return True
        return False
    if k == "field_expression":
        f = fld(node, "field")
        return f is not None and (txt(src, f) in ctx.req_members or txt(src, f) in STRUCT_FIELDS)
    if k == "call_expression":
        fn = fld(node, "function")
        if fn is None:
            return False
        ft = re.sub(r"\s+", "", txt(src, fn))
        if ft in ("std::move", "std::forward", "move", "forward"):
            args = [c for c in kids(fld(node, "arguments")) if _a(c, "is_named")]
            return len(args) == 1 and request_valued(src, ctx, h, args[0], auto_names)
        return callee_last(ft) in ctx.req_returning
    return False


def callee_ident(src, ident):
    """For an identifier that names a callee, the outermost node of that callee expression
    (`sycl::malloc_device<float>` for the identifier `malloc_device`), else the identifier."""
    top = ident
    p = parent(top)
    while p is not None and kind(p) in ("qualified_identifier", "template_function", "field_expression"):
        top = p
        p = parent(p)
    if p is not None and kind(p) == "call_expression" and same(fld(p, "function"), top):
        return top, True
    return ident, False


# Roles that are known and need no finding; the others are B-FORM.
FINDING_ROLES = ("array", "new", "template-arg", "base-class", "cast", "range-for-copy", "unclassified", "default-arg",
                 "braced-return", "braced-arg", "macro-write")
MACRO_WRITE_RE = re.compile(r"(?:\.|->)\s*(?:" + "|".join(TRACKED) + r")\s*[|&^+-]?=(?!=)")
PTR_KINDS = ("pointer_declarator", "reference_declarator")
ABSTRACT_PTR = ("abstract_pointer_declarator", "abstract_reference_declarator")


def token_role(src, tok, value_types):
    """Syntactic role of one request-type token, or 'construction' when the declaration path owns it."""
    T = tok
    p = parent(T)
    if p is not None and kind(p) == "qualified_identifier" and same(fld(p, "scope"), T):
        return "scope-qualifier"
    while p is not None and kind(p) == "qualified_identifier":
        T = p
        p = parent(T)
    if p is None:
        return "unclassified"
    k = kind(p)
    if k == "destructor_name" or (k == "function_declarator" and same(fld(p, "declarator"), T)):
        return "ctor-name"  # a holder's own constructor or destructor declares nothing about a request
    if k == "optional_parameter_declaration" and any(
            kind(c) == "initializer_list" or (kind(c) == "compound_literal_expression" and (fld(c, "type") is None or not txt(src, fld(c, "type")).strip()))
            for c in kids(p)):
        return "default-arg"  # `T r = {}`: the grammar reads the bare braces as a compound literal of no type
    if k in ("parameter_declaration", "optional_parameter_declaration", "variadic_parameter_declaration"):
        return "param"
    if k in ("struct_specifier", "class_specifier"):
        return "def"
    if k == "base_class_clause":
        return "base-class"
    if k in ("alias_declaration", "type_definition"):
        return "alias"
    if k == "friend_declaration":
        return "friend"
    if k in ("using_declaration", "namespace_alias_definition"):
        return "using"
    if k == "new_expression":
        return "new"
    if k == "compound_literal_expression":
        return "temp"
    if k == "call_expression" and same(fld(p, "function"), T):
        return "functional-cast"
    if k == "template_argument_list":
        return "template-arg"
    if k == "type_descriptor":
        has_ptr = any(kind(x) in ABSTRACT_PTR for x in kids(p))
        q = parent(p)
        while q is not None and kind(q) not in ("template_argument_list", "sizeof_expression", "cast_expression",
                                                "alignof_expression", "parameter_declaration"):
            q = parent(q)
        if q is not None and kind(q) in ("sizeof_expression", "alignof_expression"):
            return "sizeof"
        if q is not None and kind(q) == "parameter_declaration":
            return "param"
        if q is not None and kind(q) == "cast_expression":
            return "ptrref" if has_ptr else "cast"
        if q is not None and kind(q) == "template_argument_list":
            return "ptrref" if has_ptr else "template-arg"
        return "unclassified"
    if k in ("declaration", "field_declaration", "function_definition", "for_range_loop") and same(fld(p, "type"), T):
        if k == "function_definition":
            return "return"
        roles = []
        decls = [d for d in kids(p) if not same(d, fld(p, "type"))]
        if k == "for_range_loop":
            decls = [fld(p, "declarator")] if fld(p, "declarator") is not None else []
        for d in decls:
            top = fld(d, "declarator") if kind(d) == "init_declarator" else d
            if top is None:
                continue
            kt = kind(top)
            if kt in PTR_KINDS:
                roles.append("ptrref")
            elif kt == "function_declarator":
                roles.append("return")
            elif kt == "array_declarator":
                roles.append("array")
            elif kt in ("identifier", "field_identifier"):
                if k == "for_range_loop":
                    roles.append("range-for-copy")
                else:
                    roles.append("construction" if txt(src, T).split("::")[-1].strip() in value_types else "holder")
        for bad in FINDING_ROLES:
            if bad in roles:
                return bad
        for r in ("construction", "holder", "return", "ptrref"):
            if r in roles:
                return r
        return "def"
    return "unclassified"


def statement_of(n):
    p = n
    while p is not None and kind(p) not in ("declaration", "field_declaration", "expression_statement", "alias_declaration",
                                            "type_definition", "return_statement", "function_definition",
                                            "for_range_loop", "template_declaration"):
        p = parent(p)
    return p if p is not None else n


def scan_file(rel, src, ctx):
    """Facts about one file, as plain data: constructions, form findings, raw hits, error tokens."""
    value_types, all_types = ctx.value_types, ctx.all_types
    auto_names = set()
    h = _sha(src)
    root = parse(src)
    root_error = kind(root) == "ERROR"
    constructions, raws, errtoks, forms = [], [], [], []

    def key_for(func, nodekind, var, text):
        """Key without its ordinal; analyse() numbers identical keys among the constructions that violate."""
        return "%s::%s::%s:%s:%s" % (rel, func, nodekind, var, text_hash(text))

    def bind_writes(n, name, fields, copied, err):
        """Fold the later writes bound to declaration `n` into `fields`; return their statement texts."""
        texts = []
        if not in_function(n) or err:
            return texts
        blk = scope_block(n)
        if blk is None:
            return texts
        first_use = None
        for ident in idents_in(h, src, blk).get(name, []):
            if sb(ident) >= eb(n) and same(binding_decl(src, ident, name), n) and is_use(src, ident, n):
                first_use = sb(ident)
                break
        for root_name, _, rhs, an, compound in assignments_in(h, src, blk):
            if root_name != name or not (eb(n) <= sb(an)):
                continue
            if not same(binding_decl(src, an, name), n):
                continue
            last = assign_last_field(src, an)
            texts.append(txt(src, an))
            # a write after the request has first been handed to other code does not set up what that code read
            late = first_use is not None and sb(an) > first_use
            if last is None:
                copied.append("whole-assign")
            elif last in STRUCT_FIELDS:
                if rhs is not None and kind(rhs) == "initializer_list" and not compound:
                    init_fields(src, rhs, fields, copied)
                else:
                    copied.append(last)
            elif last in TRACKED:
                add_write(fields, last, sb(an), "expr" if compound else classify(src, last, rhs),
                          is_conditional(an, blk, n) or late, txt(src, rhs) if rhs is not None else "")
        return texts

    for n in walk(root):
        k = kind(n)
        if k in ("declaration", "field_declaration"):
            t = fld(n, "type")
            auto_decl = t is not None and k == "declaration" and kind(t) == "placeholder_type_specifier"
            if t is None or (not auto_decl and base_type(txt(src, t)) not in value_types):
                continue
            func = enclosing(src, n)
            if k == "field_declaration":
                holder = parent(parent(n))
                hn = fld(holder, "name") if holder is not None and kind(holder) in ("struct_specifier", "class_specifier") else None
                if hn is not None and txt(src, hn) in BASE_TYPES:
                    continue  # the request types' own members define their defaults; they construct nothing
            err = has_error_ancestor(n)
            ks = kids(n)
            for ki, d in enumerate(ks):
                if same(d, t):
                    continue
                if auto_decl:
                    # `auto x = <request>` copies a request without naming its type; the copy is clause (c)'s
                    name = declared_name(src, d)
                    init = fld(d, "value") if kind(d) == "init_declarator" else None
                    if name is None or init is None or not request_valued(src, ctx, h, init, auto_names):
                        continue
                    auto_names.add((name, sb(n)))
                    form, val = "copy", init
                else:
                    name, form, val = declarator_info(src, d)
                if form in ("skip", "array") or name is None:
                    continue
                if k == "field_declaration" and form == "plain" and ki + 1 < len(ks):
                    # a default member initialiser is a sibling of the field name, not an init_declarator
                    nxt = ks[ki + 1]
                    if kind(nxt) == "initializer_list":
                        form, val = "braced", nxt
                    elif txt(src, nxt) == "=" and ki + 2 < len(ks):
                        form, val = ("braced", ks[ki + 2]) if kind(ks[ki + 2]) == "initializer_list" else ("copy", ks[ki + 2])
                fields, copied, reason = {}, [], []
                if form == "braced":
                    init_fields(src, val, fields, copied)
                elif form == "copy":
                    reason.append("copy-init")
                elif form == "ctor":
                    reason.append("ctor-args")
                texts = bind_writes(n, name, fields, copied, err) if k == "declaration" else []
                for c in copied:
                    reason.append(c + "-copied" if c != "positional" else "positional-init")
                if k == "field_declaration" and form == "braced" and not fields and not reason:
                    # `T x{}` in a holder struct is storage whose value is set elsewhere (clause (c));
                    # only a member initialiser that carries a flag is a construction of its own
                    continue
                for f in fields:
                    fields[f].sort()
                constructions.append({
                    "key": key_for(func, kind(n), name, txt(src, n) + " ;; " + " ;; ".join(texts)), "func": func,
                    "var": name, "line": line_of(n), "kind": "member" if k == "field_declaration" else "decl",
                    "type": "auto" if auto_decl else base_type(txt(src, t)), "form": form, "err": err, "fields": fields,
                    "deferred": sorted(set(reason + (["auto-init"] if auto_decl else []))),
                })
        elif k == "compound_literal_expression" or (k == "call_expression" and fld(n, "function") is not None
                                                    and kind(fld(n, "function")) in ("identifier", "type_identifier",
                                                                                     "qualified_identifier")
                                                    and base_type(txt(src, fld(n, "function"))) in value_types):
            t = fld(n, "type") if k == "compound_literal_expression" else fld(n, "function")
            if t is None or base_type(txt(src, t)) not in value_types:
                continue
            func = enclosing(src, n)
            fields, copied = {}, []
            val = fld(n, "value") if k == "compound_literal_expression" else None
            reason = []
            if val is not None and kind(val) == "initializer_list":
                init_fields(src, val, fields, copied)
            elif k == "call_expression":
                args = fld(n, "arguments")
                if args is not None and any(_a(c, "is_named") for c in kids(args)):
                    reason.append("ctor-args")
            for f in fields:
                fields[f].sort()
            tn = base_type(txt(src, t))
            constructions.append({
                "key": key_for(func, kind(n), "<temp %s>" % tn, txt(src, n)), "func": func, "var": "<temp>",
                "line": line_of(n), "kind": "temp", "type": tn, "form": "braced", "err": has_error_ancestor(n),
                "fields": fields,
                "deferred": sorted(set(reason + ["positional-init" if c == "positional" else c + "-copied" for c in copied])),
            })
        elif k == "return_statement":
            lst = [c for c in kids(n) if kind(c) == "initializer_list"]
            q = parent(n)
            while q is not None and kind(q) not in ("function_definition", "lambda_expression"):
                q = parent(q)
            if lst and q is not None and kind(q) == "function_definition":
                rt, rd = fld(q, "type"), fld(q, "declarator")
                if rt is not None and base_type(txt(src, rt)) in value_types and rd is not None \
                        and kind(rd) == "function_declarator":
                    forms.append({"func": enclosing(src, n), "role": "braced-return", "tok": "{...}",
                                  "line": line_of(n), "text": txt(src, n)})
        elif k == "call_expression" and fld(n, "arguments") is not None \
                and any(kind(c) == "initializer_list" for c in kids(fld(n, "arguments"))) \
                and fld(n, "function") is not None and callee_last(txt(src, fld(n, "function"))) in ctx.req_funcs:
            forms.append({"func": enclosing(src, n), "role": "braced-arg", "tok": "{...}", "line": line_of(n),
                          "text": txt(src, n)})
        elif k in ("string_literal", "raw_string_literal", "concatenated_string"):
            if k != "concatenated_string" and parent(n) is not None and kind(parent(n)) == "concatenated_string":
                continue  # judged as part of the whole concatenation
            s = txt(src, n)
            joined = "".join(txt(src, c) for c in kids(n) if kind(c) in ("string_literal", "raw_string_literal")) \
                if k == "concatenated_string" else s
            joined = joined.replace('" "', "").replace('""', "")
            for r in RAW_STRINGS:
                if r in joined or r in s:
                    raws.append({"func": enclosing(src, n), "name": r, "line": line_of(n), "form": "string",
                                 "nodekind": k, "text": s, "err": has_error_ancestor(n)})
        elif k in ("preproc_def", "preproc_function_def"):
            body = fld(n, "value")
            nm = fld(n, "name")
            if body is not None and nm is not None:
                b = txt(src, body)
                for r in RAW_NAMES:
                    if re.search(r"(?<![A-Za-z0-9_])" + re.escape(r) + r"(?![A-Za-z0-9_])", b):
                        raws.append({"func": "#define " + txt(src, nm), "name": r, "line": line_of(n),
                                     "form": "macro", "nodekind": "preproc", "text": r, "err": False})
                if MACRO_WRITE_RE.search(b):
                    forms.append({"func": "#define " + txt(src, nm), "role": "macro-write", "tok": txt(src, nm),
                                  "line": line_of(n), "text": txt(src, n)})
        elif k in ("identifier", "field_identifier") and txt(src, n) in RAW_NAMES:
            pk = kind(parent(n)) if parent(n) is not None else ""
            if pk != "function_declarator":  # the name of a declaration or definition is not a use
                top, is_call = callee_ident(src, n)
                raws.append({"func": enclosing(src, n), "name": txt(src, n), "line": line_of(n),
                             "form": "call" if is_call else "name",
                             "nodekind": "call_expression" if is_call else "identifier",
                             "text": txt(src, top) if is_call else txt(src, n), "err": has_error_ancestor(n)})
        if k in ("type_identifier", "identifier") and txt(src, n) in all_types:
            if has_error_ancestor(n) or _a(n, "is_missing"):
                errtoks.append({"func": enclosing(src, n), "tok": txt(src, n), "line": line_of(n)})
            else:
                role = token_role(src, n, value_types)
                if role in FINDING_ROLES:
                    forms.append({"func": enclosing(src, n), "role": role, "tok": txt(src, n), "line": line_of(n),
                                  "text": txt(src, statement_of(n))})

    clean = blank_comments_strings(src, root)
    lexical = []
    tok_re = re.compile(rb"\b(?:" + "|".join(sorted(all_types)).encode() + rb")\b")
    lex_count = len(tok_re.findall(clean))
    ast_count = sum(1 for n in walk(root) if kind(n) in ("type_identifier", "identifier") and txt(src, n) in all_types)
    if root_error:
        lexical = lexical_scan(src, clean, value_types)
        for r in RAW_NAMES:
            for m in re.finditer(rb"(?<![A-Za-z0-9_])" + re.escape(r.encode()) + rb"\s*(?:<[^>]*>)?\s*\(", clean):
                raws.append({"func": "<lexical>", "name": r, "line": clean.count(b"\n", 0, m.start()) + 1,
                             "form": "lexical", "nodekind": "lexical", "text": r, "err": True})
    return {"constructions": constructions, "raws": raws, "errtoks": errtoks, "forms": forms,
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


def facts_for(rel, src, ctx):
    key = (rel, _sha(src), ctx.digest)
    if key not in _FACTS:
        _FACTS[key] = scan_file(rel, src, ctx)
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


def established(ws, good):
    """A flag is established when some unconditional write is the good literal and the last write is too."""
    return bool(ws) and any(w[1] == good and not w[2] for w in ws) and ws[-1][1] == good


def judge(c):
    """Violations of one AST construction, as (code, message)."""
    if c["err"]:
        return [("A-ERROR", "%s sits inside an ERROR region of the parse; nothing about it can be trusted" % c["type"])]
    out = []
    if c["form"] == "plain":
        out.append(("B-BRACE", "braceless `%s %s;` reports the class definition as its construction site; write `%s %s{}`"
                    % (c["type"], c["var"], c["type"], c["var"])))
    w = c["fields"]
    dev = established(w.get("must_device", []), "true")
    host = established(w.get("must_host_pinned", []), "true")
    zone_ws, forbid_ws = w.get("prefer_vram_zone", []), w.get("forbid_vram_zone_spill", [])
    # A COUNT zone or a false forbid is a violation whatever a copy resolves to, so it is read before
    # the DEFER-C return; an established host tier is the only thing that excuses it.
    if not (host and not dev):
        if zone_ws and zone_ws[-1][1] == "count":
            out.append(("D-ZONE-COUNT", "the last prefer_vram_zone write names COUNT (arena bypass)"))
        if forbid_ws and forbid_ws[-1][1] == "false":
            out.append(("D-FORBID-FALSE", "the last forbid_vram_zone_spill write is false"))
    if c["deferred"]:
        out.append(("DEFER-C", "built from a copy or call (%s); clause (c) will follow it" % ",".join(c["deferred"])))
        return out
    if dev and host:
        out.append(("B-TIER", "both must_device and must_host_pinned are established true; the tier is not decidable"))
        return out
    if not dev and not host:
        out.append(("B-TIER", "neither must_device nor must_host_pinned is established by an unconditional literal true as its "
                              "last write; unified_select_tier may turn it into HOST"))
        return out
    other_name = "must_device" if host else "must_host_pinned"
    if any(x[1] != "false" for x in w.get(other_name, [])):
        # a conditional true, or a non-literal, can make the request the other tier; only a literal false is harmless
        out.append(("B-TIER", "%s has a write that is not a literal false (a conditional true or a non-literal), so the "
                              "tier is not decidable" % other_name))
        return out
    if host:
        return out
    codes = {o[0] for o in out}
    if "D-ZONE-COUNT" not in codes:
        if not zone_ws:
            out.append(("D-ZONE", "device request with no prefer_vram_zone"))
        elif not established(zone_ws, "ok"):
            out.append(("D-ZONE", "prefer_vram_zone is not established by an unconditional literal non-COUNT enumerator "
                                  "or all-literal ternary as its last write"))
    if "D-FORBID-FALSE" not in codes and not established(forbid_ws, "true"):
        out.append(("D-FORBID", "device request does not establish forbid_vram_zone_spill = true by an unconditional literal"))
    return out


def analyse(files):
    """Every finding in the tree, before any allowlist or debt is applied."""
    ctx = Ctx(files)
    viols = []
    stats = {"constructions": 0, "files": len(files), "value_types": sorted(ctx.value_types),
             "all_types": sorted(ctx.all_types)}
    for rel in sorted(files):
        fa = facts_for(rel, files[rel], ctx)
        ordinal = {}
        for c in fa["constructions"]:
            stats["constructions"] += 1
            found = judge(c)
            if not found:
                continue  # a compliant twin takes no ordinal, so adding one cannot move a listed key
            i = ordinal.get(c["key"], 0)
            ordinal[c["key"]] = i + 1
            for code, msg in found:
                viols.append(V(code, "%s#%d" % (c["key"], i), rel, c["line"], c["func"], c["var"], msg))
        seen = {}
        for t in fa["errtoks"]:
            base = "%s::%s::token:%s" % (rel, t["func"], t["tok"])
            i = seen.get(base, 0)
            seen[base] = i + 1
            viols.append(V("A-ERROR", "%s#%d" % (base, i), rel, t["line"], t["func"], t["tok"],
                           "request-type token %s inside an ERROR/MISSING region" % t["tok"]))
        for r in fa["lexical"]:
            if not r["host_only"]:
                viols.append(V("A-LEXICAL", "%s::%s" % (rel, r["key"]), rel, r["line"], "<lexical>", r["var"],
                               "lexical hit in a file whose root is ERROR is not host-only by a literal"))
        if fa["lex_count"] != fa["ast_count"]:
            viols.append(V("A-TOKEN", "%s::<tokens>::lexical-vs-ast" % rel, rel, 0, "<file>", "",
                           "request-type tokens by text (%d) differ from the parse's (%d); a request-type name sits where the "
                           "parse cannot see it (a macro body, a broken #if): needs a clause-(c) follow or a rewrite"
                           % (fa["lex_count"], fa["ast_count"])))
        seen = {}
        for f in fa["forms"]:
            base = "%s::%s::form:%s:%s" % (rel, f["func"], f["role"], text_hash(f["text"]))
            i = seen.get(base, 0)
            seen[base] = i + 1
            viols.append(V("B-FORM", "%s#%d" % (base, i), rel, f["line"], f["func"], f["tok"],
                           "%s occurs as a %s, a form the gate cannot follow to a construction; write the request as a named "
                           "`T x{}` with its literal flags, or wait for clause (c) to follow it" % (f["tok"], f["role"])))
        seen = {}
        for r in fa["raws"]:
            base = "%s::%s::%s:%s:%s" % (rel, r["func"], r["nodekind"], r["name"], text_hash(r["text"]))
            i = seen.get(base, 0)
            seen[base] = i + 1
            viols.append(V("E-RAW", "%s#%d" % (base, i), rel, r["line"], r["func"], r["name"],
                           "raw allocator name %s (%s) outside the allowlist" % (r["name"], r["form"])))
    return viols, stats


# ---------------------------------------------------------------- allowlist and debt

class DataError(Exception):
    pass


def load_json(path, default, listkey):
    """The parsed file, or `default` when it does not exist. Malformed JSON, a non-object file or a non-list
    `listkey` is a DataError, so the caller prints a FAIL line instead of a traceback."""
    if not Path(path).exists():
        return default
    try:
        with open(path) as f:
            doc = json.load(f)
    except (OSError, ValueError) as exc:
        raise DataError("%s is not readable JSON (%s)" % (path, exc))
    if not isinstance(doc, dict):
        raise DataError("%s must be a JSON object, got %s" % (path, type(doc).__name__))
    if not isinstance(doc.get(listkey, []), list):
        raise DataError("%s: %r must be a list" % (path, listkey))
    return doc


def validate_data(allowlist, debt):
    """Schema problems in the two data files, as FAIL lines (never a traceback)."""
    errs = []
    ids = Counter()
    for i, e in enumerate(allowlist.get("entries", [])):
        if not isinstance(e, dict):
            errs.append("FAIL allowlist entry #%d is not an object" % i)
            continue
        need = ("id", "code", "reason", "count") + (("key",) if "key" in e else ("file", "function")) \
            + (("name",) if e.get("code") == "E-RAW" else ())
        miss = [k for k in need if k not in e]
        if miss:
            errs.append("FAIL allowlist entry %s lacks %s" % (e.get("id", "#%d" % i), ", ".join(miss)))
            continue
        if e["code"] not in CODES:
            errs.append("FAIL allowlist entry %s has unknown code %r" % (e["id"], e["code"]))
        if not isinstance(e["count"], int) or isinstance(e["count"], bool) or e["count"] < 1:
            errs.append("FAIL allowlist entry %s needs an integer count >= 1" % e["id"])
        if not isinstance(e["reason"], str) or len(e["reason"].strip()) < CITE_MIN:
            errs.append("FAIL allowlist entry %s needs a reason of at least %d characters" % (e["id"], CITE_MIN))
        ids[e["id"]] += 1
    errs += ["FAIL allowlist id %s is used %d times" % (k, v) for k, v in ids.items() if v > 1]
    seen = Counter()
    for i, d in enumerate(debt.get("violations", [])):
        if not isinstance(d, dict) or not isinstance(d.get("code"), str) or not isinstance(d.get("key"), str):
            errs.append("FAIL debt entry #%d is malformed (needs string code and key): %r" % (i, d))
            continue
        if d["code"] not in CODES:
            errs.append("FAIL debt entry %s %s has unknown code" % (d["code"], d["key"]))
        seen[(d["code"], d["key"])] += 1
        if d["code"] == "E-RAW" and not FATE_RE.match(str(d.get("fate", ""))):
            errs.append("FAIL debt entry E-RAW %s has no valid fate (deleted-by-*, converted-by-*, sanctioned-internal, "
                        "sanctioned-vendored, pending-disposition)" % d["key"])
        if d["code"] == "E-RAW" and (not isinstance(d.get("cite"), str) or len(d["cite"].strip()) < CITE_MIN):
            errs.append("FAIL debt entry E-RAW %s has no cite (a ticket id or a design/census row, at least %d characters)"
                        % (d["key"], CITE_MIN))
    errs += ["FAIL debt entry %s %s is listed %d times" % (c, k, v) for (c, k), v in seen.items() if v > 1]
    return errs


def entry_matches(ent, v):
    if v.code != ent["code"]:
        return False
    if "key" in ent:
        return v.key == ent["key"]
    return v.file == ent["file"] and v.func == ent["function"] and ("name" not in ent or v.name == ent["name"])


def apply_contract(viols, allowlist, debt):
    """(failures, report lines). Allowlist first, then the shrink-only debt."""
    fails, report = validate_data(allowlist, debt), []
    if fails:
        return fails, report
    remaining = list(viols)
    for ent in allowlist.get("entries", []):
        eid = ent["id"]
        hit = [v for v in remaining if entry_matches(ent, v)]
        report.append("allowlist %-28s covers %d (count %s)" % (eid, len(hit), ent["count"]))
        if not hit:
            fails.append("FAIL allowlist entry %s matches nothing (%s %s); a stale or renamed exemption "
                         "protects nothing" % (eid, ent["code"], ent.get("key") or "%s::%s" % (ent["file"], ent["function"])))
            continue
        if ent["count"] != len(hit):
            fails.append("FAIL allowlist entry %s covers %d finding(s) but pins %d; a new one in an exempt "
                         "function is not exempt" % (eid, len(hit), ent["count"]))
        for v in hit:
            remaining.remove(v)
    debt_ids = {(d["code"], d["key"]) for d in debt.get("violations", [])}
    count = Counter(v.ident() for v in remaining)
    for ident, n in sorted(count.items()):
        if n > 1:
            fails.append("FAIL duplicate node key %s %s shared by %d findings; two nodes must not share a key" % (ident + (n,)))
    cur = {v.ident(): v for v in remaining}
    for ident, v in sorted(cur.items()):
        if ident not in debt_ids:
            fails.append("FAIL new %s" % v)
    for ident in sorted(debt_ids - set(cur)):
        fails.append("FAIL stale debt entry %s %s no longer violates; delete it from debt.json" % ident)
    report.append("debt %d entries; %d remaining findings" % (len(debt_ids), len(cur)))
    return fails, report


def run_gate(files, allowlist, debt):
    viols, stats = analyse(files)
    fails, report = apply_contract(viols, allowlist, debt)
    return fails, report, viols, stats


def plan_debt_write(viols, allowlist, old_debt, allow_growth):
    """(ok, message, entries). Rewriting the debt list may drop entries; adding one needs --allow-growth."""
    rest = [v for v in viols if not any(entry_matches(e, v) for e in allowlist.get("entries", []))]
    ids = sorted({v.ident() for v in rest})
    old = {(d["code"], d["key"]): {k: d[k] for k in ("fate", "cite") if k in d} for d in old_debt.get("violations", [])}
    grown = sorted(set(ids) - set(old))
    if grown and not allow_growth:
        return False, "--write-debt would ADD %d entr%s; debt may only shrink. Fix the violation, or pass --allow-growth for " \
                      "a deliberate key migration or the first seeding:\n%s" % (
                          len(grown), "y" if len(grown) == 1 else "ies", "\n".join("  %s %s" % g for g in grown[:20])), []
    return True, "", [dict({"code": c, "key": k}, **old.get((c, k), {})) for c, k in ids]


# ---------------------------------------------------------------- mutation matrix

PLANT = "zz-plant.cpp"
REQ = "ggml_sycl::alloc_request"


def plant(body, name=PLANT):
    return lambda files: dict(files, **{name: body.encode()})


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


def insert_after(rel, anchor, text):
    def f(files):
        src = files[rel].decode()
        i = src.find(anchor)
        if i < 0:
            raise SystemExit("matrix setup error: anchor %r not found in %s" % (anchor, rel))
        i += len(anchor)
        return dict(files, **{rel: (src[:i] + text + src[i:]).encode()})
    return f


def replace_in_function(rel, func_re, old, new):
    """Replace the first `old` after the first match of func_re (a definition's name)."""
    def f(files):
        src = files[rel].decode()
        m = re.search(func_re, src)
        if m is None:
            raise SystemExit("matrix setup error: %s not found in %s" % (func_re, rel))
        j = src.find(old, m.end())
        if j < 0:
            raise SystemExit("matrix setup error: %r not found after %s in %s" % (old, func_re, rel))
        return dict(files, **{rel: (src[:j] + new + src[j + len(old):]).encode()})
    return f


def good_device(name, extra=""):
    return ("void %s() {\n    %s req{};\n    req.intent.constraints.must_device = true;\n"
            "    req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME;\n"
            "    req.intent.constraints.forbid_vram_zone_spill = true;\n%s}\n" % (name, REQ, extra))


def copy_file(src_rel, dst_rel):
    return lambda files: dict(files, **{dst_rel: files[src_rel]})


class Case:
    """One matrix case. A PASS case is a control by default and must have seen something planted
    (`planted=False` opts out for a mutation that adds no construction)."""

    def __init__(self, wid, label, mutate, expect, code=None, naming=None, allowlist=None, edit_allowlist=None,
                 edit_debt=None, planted=True):
        self.wid, self.label, self.mutate, self.expect = wid, label, mutate, expect
        self.code, self.naming, self.allowlist = code, naming, allowlist
        self.edit_allowlist, self.edit_debt, self.planted = edit_allowlist, edit_debt, planted


# witness id -> what it pins. Every id must have a FAIL case; 9 is deferred and says so.
WITNESSES = {
    "1": "device request with no zone", "1s": "a pointer-holder scope cannot launder a request",
    "2": "forbid removed or false", "3": "? COUNT : ONEDNN", "4": "COUNT retry after a zoned request",
    "5": "raw allocator in a new function", "6": "comments and strings are not code",
    "7": "rename an exempt function", "8": "allowlist entry that matches nothing or pins a wrong count",
    "12": "request with no tier flag", "13": "request inside a broken #if", "14": "ternary arms",
    "18": "offload_buffer_request needs a literal tier", "19": "ERROR-root lexical pass", "20": "member initialiser",
    "23": "braceless declaration", "c1": "template argument hides a raw name", "i1": "dpct is in scope",
    "i2": "construction forms the enumeration could not see", "i3": "COUNT/false after a copy at a listed site",
    "i4": "shadowing does not credit the outer request", "i5": "identical content under another name",
    "m1": "last-write semantics", "m2": "compound assignments", "m3": "conditional writes fail closed",
    "m5": "allowlist pins name and count", "m6": "argument handling", "m7": "data-file validation",
    "m8": "raw names beyond calls", "m9": "unscanned C++ extensions", "m14": "key-matched allowlist entries",
    "key": "keys survive unrelated edits", "debt": "debt is shrink-only both ways", "fate": "E-RAW fate required",
    "r2i1": "any non-false write of the other tier flag is B-TIER", "r2i2": "whole-object assignment is a copy",
    "r2i3": "constructions with no request-type token", "r2m2": "E-RAW allowlist entries need a name",
    "r2m3": "cite and fate validation", "r2m4": "--write-debt validates its input, no tracebacks",
    "r2m6": "writes after the first use do not satisfy", "r2m7": "a macro body assigning a tracked field",
    "r2m8": "scope: anchored skips, case-insensitive suffixes, required files, floor",
    "r2n": "adjacent string literals are judged joined",
    "f": "missing tree_sitter_language_pack exits 1", "cmake": "the ctest never passes a regeneration flag",
}
WITNESSES_DEFERRED = {"9": "S2d: dormant clause; its *_bytes() subjects are absent from the tree"}


def matrix_cases():
    c = []
    A = c.append
    # 1: a device request with no zone
    A(Case("1", "device request with a zone, forbid and tier (control)", plant(good_device("zzplant_w1")), "PASS"))
    A(Case("1", "must_device request with no zone", plant(
        "void zzplant_w1() {\n    %s req{};\n    req.intent.constraints.must_device = true;\n"
        "    req.intent.constraints.forbid_vram_zone_spill = true;\n}\n" % REQ), "FAIL", "D-ZONE", "zzplant_w1"))
    # 1s: a pointer-holder scope over a request
    A(Case("1s", "a scope over a violating request still FAILs at the request's construction", plant(
        "void zzplant_w1s() {\n    %s req{};\n    req.intent.constraints.must_device = true;\n"
        "    req.intent.constraints.forbid_vram_zone_spill = true;\n    ext_alloc_request_scope scope(&req);\n}\n" % REQ),
        "FAIL", "D-ZONE", "zzplant_w1s::declaration:req:"))
    A(Case("1s", "a scope over a clean request adds no finding of its own", plant(
        good_device("zzplant_w1s", "    ext_alloc_request_scope scope(&req);\n")), "PASS"))
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
    A(Case("6", "the same text outside the comment is code and FAILs", plant(
        "void zzplant_w6x() {\n    %s r{};\n    r.intent.constraints.must_device = true;\n"
        "    unified_cache_malloc_device_tracked(1);\n}\n" % REQ), "FAIL", "D-ZONE", "zzplant_w6x"))
    # 7: rename an exempt function
    A(Case("7", "E-ARENA-BACKING's function renamed", replace_token("unified-cache.cpp", "arena_reserve", "arena_reserve_zz"),
        "FAIL", "E-RAW", "arena_reserve_zz"))
    A(Case("7", "E-ARENA-BACKING's entry reports the rename", replace_token("unified-cache.cpp", "arena_reserve", "arena_reserve_zz"),
        "FAIL", "allowlist", "E-ARENA-BACKING matches nothing"))
    A(Case("7", "E-TEST's function renamed", replace_token("unified-cache.cpp", "ensure_cached_alloc", "ensure_cached_alloc_zz"),
        "FAIL", "E-RAW", "ensure_cached_alloc_zz"))
    # 8: an allowlist entry that matches nothing, or pins a count the function does not have
    A(Case("8", "allowlist entry matching nothing", lambda f: f, "FAIL", "allowlist", "E-ZZ-BOGUS matches nothing",
           allowlist={"id": "E-ZZ-BOGUS", "code": "E-RAW", "file": "unified-cache.cpp", "function": "no_such_function",
                      "name": "malloc_device", "count": 1, "reason": "mutation-matrix test entry"}))
    A(Case("8", "allowlist entry pinning a count its function does not have", lambda f: f, "FAIL", "allowlist", "E-TEST covers 2 finding(s) but pins 3",
           edit_allowlist=lambda al: dict(al, entries=[dict(e, count=3) if e["id"] == "E-TEST" else e for e in al["entries"]])))
    # m5: the allowlist pins a name, not just a count
    A(Case("m5", "a different raw name swapped into an exempt function keeps the count and FAILs",
           replace_in_function("unified-cache.cpp", r"unified_cache::arena_reserve\s*\(", "sycl_aligned_malloc_device(",
                               "zeMemAllocDevice("), "FAIL", "allowlist", "E-ARENA-BACKING covers 1 finding(s) but pins 2"))
    A(Case("m5", "an allowlist entry without a count is rejected", lambda f: f, "FAIL", "allowlist", "E-TEST lacks count",
           edit_allowlist=lambda al: dict(al, entries=[{k: v for k, v in e.items() if k != "count"} if e["id"] == "E-TEST" else e
                                                      for e in al["entries"]])))
    # 12: no tier flag
    A(Case("12", "request with no tier flag", plant(
        "void zzplant_w12() {\n    %s req{};\n    req.size = 1;\n}\n" % REQ), "FAIL", "B-TIER", "zzplant_w12"))
    A(Case("12", "request with a literal host tier (control)", plant(
        "void zzplant_w12() {\n    %s req{};\n    req.intent.constraints.must_host_pinned = true;\n}\n" % REQ), "PASS"))
    A(Case("12", "tier written as a non-literal expression", plant(
        "void zzplant_w12(bool b) {\n    %s req{};\n    req.intent.constraints.must_device = b;\n}\n" % REQ),
        "FAIL", "B-TIER", "zzplant_w12"))
    # m1: last-write semantics on the tier
    A(Case("m1", "must_device = true then = false", plant(
        "void zzplant_m1() {\n    %s req{};\n    req.intent.constraints.must_device = true;\n"
        "    req.intent.constraints.must_device = false;\n}\n" % REQ), "FAIL", "B-TIER", "zzplant_m1"))
    A(Case("m1", "host tier plus a non-literal must_device", plant(
        "void zzplant_m1(bool b) {\n    %s req{};\n    req.intent.constraints.must_host_pinned = true;\n"
        "    req.intent.constraints.must_device = b;\n}\n" % REQ), "FAIL", "B-TIER", "zzplant_m1"))
    A(Case("m1", "host tier with must_device = false (control)", plant(
        "void zzplant_m1() {\n    %s req{};\n    req.intent.constraints.must_host_pinned = true;\n"
        "    req.intent.constraints.must_device = false;\n}\n" % REQ), "PASS"))
    # m2: compound assignments
    A(Case("m2", "forbid_vram_zone_spill &= false after a literal true", plant(good_device(
        "zzplant_m2", "    req.intent.constraints.forbid_vram_zone_spill &= false;\n")), "FAIL", "D-FORBID", "zzplant_m2"))
    A(Case("m2", "forbid_vram_zone_spill |= false after a literal true", plant(good_device(
        "zzplant_m2", "    req.intent.constraints.forbid_vram_zone_spill |= false;\n")), "FAIL", "D-FORBID", "zzplant_m2"))
    # m3: conditional writes fail closed
    A(Case("m3", "the forbid literal only inside an if", plant(
        "void zzplant_m3(bool b) {\n    %s req{};\n    req.intent.constraints.must_device = true;\n"
        "    req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME;\n"
        "    if (b) {\n        req.intent.constraints.forbid_vram_zone_spill = true;\n    }\n}\n" % REQ),
        "FAIL", "D-FORBID", "zzplant_m3"))
    A(Case("m3", "the zone literal only inside a ternary-guarded block", plant(
        "void zzplant_m3(bool b) {\n    %s req{};\n    req.intent.constraints.must_device = true;\n"
        "    req.intent.constraints.forbid_vram_zone_spill = true;\n"
        "    if (b) {\n        req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME;\n    }\n}\n" % REQ),
        "FAIL", "D-ZONE", "zzplant_m3"))
    A(Case("m3", "a plain nested block is not a condition (control)", plant(
        "void zzplant_m3() {\n    %s req{};\n    req.intent.constraints.must_device = true;\n"
        "    {\n        req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME;\n"
        "        req.intent.constraints.forbid_vram_zone_spill = true;\n    }\n}\n" % REQ), "PASS"))
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
    # c1: a scoped template argument must not hide the name
    A(Case("c1", "sycl::malloc_device<sycl::half>(...) in a new function", plant(
        "void zzplant_c1(sycl::queue & q) {\n    auto * p = sycl::malloc_device<sycl::half>(16, q);\n    (void) p;\n}\n"),
        "FAIL", "E-RAW", "zzplant_c1"))
    A(Case("c1", "sycl::malloc_device<float>(...) in a new function", plant(
        "void zzplant_c1f(sycl::queue & q) {\n    auto * p = sycl::malloc_device<float>(16, q);\n    (void) p;\n}\n"),
        "FAIL", "E-RAW", "zzplant_c1f"))
    A(Case("c1", "a nested scoped template argument", plant(
        "void zzplant_c1n(sycl::queue & q) {\n    auto * p = sycl::malloc_shared<std::pair<sycl::half, ns::T>>(16, q);\n    (void) p;\n}\n"),
        "FAIL", "E-RAW", "zzplant_c1n"))
    # i1: dpct is in scope
    A(Case("i1", "a raw call planted under dpct/", plant(
        "inline void zzplant_i1(sycl::queue & q) { auto * p = sycl::malloc_device<int>(4, q); (void) p; }\n",
        name="dpct/zz-plant.hpp"), "FAIL", "E-RAW", "zzplant_i1"))
    # i2: construction forms the enumeration could not see
    A(Case("i2", "using alias of alloc_request", plant(
        "using ZzR = %s;\nvoid zzplant_i2() {\n    ZzR r{};\n    r.intent.constraints.must_device = true;\n}\n" % REQ),
        "FAIL", "D-ZONE", "zzplant_i2"))
    A(Case("i2", "typedef of alloc_request", plant(
        "typedef %s ZzRT;\nvoid zzplant_i2() {\n    ZzRT r;\n    r.intent.constraints.must_device = true;\n}\n" % REQ),
        "FAIL", "B-BRACE", "zzplant_i2"))
    A(Case("i2", "functional-cast temporary", plant(
        "void zzplant_i2() {\n    auto r = %s();\n    (void) r;\n}\n" % REQ), "FAIL", "B-TIER", "zzplant_i2"))
    A(Case("i2", "new alloc_request{}", plant(
        "void zzplant_i2() {\n    auto * r = new %s{};\n    r->intent.constraints.must_device = true;\n}\n" % REQ),
        "FAIL", "B-FORM", "zzplant_i2"))
    A(Case("i2", "array of requests", plant(
        "void zzplant_i2() {\n    %s reqs[2]{};\n    reqs[0].intent.constraints.must_device = true;\n}\n" % REQ),
        "FAIL", "B-FORM", "zzplant_i2"))
    A(Case("i2", "std::array of requests", plant(
        "void zzplant_i2() {\n    std::array<%s, 2> reqs{};\n    (void) reqs;\n}\n" % REQ), "FAIL", "B-FORM", "zzplant_i2"))
    A(Case("i2", "std::vector of requests", plant(
        "void zzplant_i2() {\n    std::vector<%s> reqs;\n    (void) reqs;\n}\n" % REQ), "FAIL", "B-FORM", "zzplant_i2"))
    A(Case("i2", "std::optional of a request", plant(
        "void zzplant_i2() {\n    std::optional<%s> r;\n    (void) r;\n}\n" % REQ), "FAIL", "B-FORM", "zzplant_i2"))
    A(Case("i2", "a request pointer, reference and parameter are known roles (control)", plant(
        "void zzplant_i2(const %s & in, %s * out) {\n    const %s & ref = in;\n    %s * p = out;\n    (void) ref;\n    (void) p;\n}\n"
        % (REQ, REQ, REQ, REQ)), "PASS", planted=False))
    # i3: COUNT or false after a copy, at an existing debt site
    A(Case("i3", "a COUNT zone written after the intent copy at scoped_unified_device_temp::alloc", insert_after(
        "ggml-sycl.cpp", "req.intent               = ggml_sycl_transient_device_intent(cohort_id);\n",
        "        req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::COUNT;\n"),
        "FAIL", "D-ZONE-COUNT", "scoped_unified_device_temp::alloc"))
    A(Case("i3", "forbid written false after the intent copy at the same site", insert_after(
        "ggml-sycl.cpp", "req.intent               = ggml_sycl_transient_device_intent(cohort_id);\n",
        "        req.intent.constraints.forbid_vram_zone_spill = false;\n"),
        "FAIL", "D-FORBID-FALSE", "scoped_unified_device_temp::alloc"))
    # i4: shadowing does not credit the outer request
    A(Case("i4", "an inner-block request with every write does not credit the outer one", plant(
        "void zzplant_i4(bool b) {\n    %s req{};\n    req.intent.constraints.must_device = true;\n"
        "    if (b) {\n        %s req{};\n        req.intent.constraints.must_device = true;\n"
        "        req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME;\n"
        "        req.intent.constraints.forbid_vram_zone_spill = true;\n    }\n}\n" % (REQ, REQ)),
        "FAIL", "D-ZONE", "zzplant_i4"))
    A(Case("i4", "the outer request's own writes still count past a nested scope (control)", plant(
        "void zzplant_i4(bool b) {\n    %s req{};\n    req.intent.constraints.must_device = true;\n"
        "    if (b) {\n        %s q{};\n        q.intent.constraints.must_host_pinned = true;\n    }\n"
        "    req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME;\n"
        "    req.intent.constraints.forbid_vram_zone_spill = true;\n}\n" % (REQ, REQ)), "PASS"))
    # i5: identical content under another name is judged as its own file
    A(Case("i5", "common.cpp copied to zz-copy.cpp: the copy's violations are not silent",
           copy_file("common.cpp", "zz-copy.cpp"), "FAIL", "D-FORBID", "zz-copy.cpp::"))
    # m8: raw names beyond a call, and raw strings
    A(Case("m8", "a raw string literal naming zeMemAllocDevice", plant(
        "void zzplant_m8() {\n    const char * s = R\"(zeMemAllocDevice)\";\n    (void) s;\n}\n"), "FAIL", "E-RAW", "zzplant_m8"))
    A(Case("m8", "address-of a raw name", plant(
        "void zzplant_m8() {\n    auto f = &zeMemAllocDevice;\n    (void) f;\n}\n"), "FAIL", "E-RAW", "zzplant_m8"))
    A(Case("m8", "a raw template taken as a value", plant(
        "void zzplant_m8() {\n    auto f = &sycl::malloc_device<int>;\n    (void) f;\n}\n"), "FAIL", "E-RAW", "zzplant_m8"))
    A(Case("m8", "a function named like a raw allocator is a declaration, not a use (control)", plant(
        "void * ggml_sycl_malloc_device_raw(int n);\nvoid * zz_other(int n);\n"), "PASS", planted=False))
    # key: an unrelated edit above must not move a key
    A(Case("key", "lines and a function added above listed constructions leave every key in place",
           lambda f: dict(f, **{"common.cpp": b"// unrelated\n\nstatic int zz_unrelated_above() { return 1; }\n\n" + f["common.cpp"]}),
           "PASS", planted=False))
    A(Case("key", "a compliant request added in the same function as a listed construction leaves its key",
           insert_after("common.cpp", "// INT16 = 2 bytes per element (mubmt.12)\n",
                        "        {\n            ggml_sycl::alloc_request req{};\n"
                        "            req.intent.constraints.must_host_pinned = true;\n        }\n"),
           "PASS", planted=False))
    A(Case("key", "editing a listed construction itself moves its key (stale entry plus new violation)",
           replace_token("common.cpp", "ggml_sycl_tp_ensure_ffn_buffers", "ggml_sycl_tp_ensure_ffn_buffers_zz"),
           "FAIL", "debt", "ggml_sycl_tp_ensure_ffn_buffers"))
    A(Case("key", "a write added to a listed construction moves its key", insert_after(
        "common.cpp", "req.size        = alloc_size * sizeof(int16_t);\n", "        req.suppress_failure_log = true;\n"),
        "FAIL", "debt", "ggml_sycl_tp_init_quant_comm_buffers_locked"))
    # debt: the list is shrink-only in both directions
    A(Case("debt", "a debt entry that no longer violates is stale", lambda f: f, "FAIL", "debt", "zzgone",
           edit_debt=lambda d: dict(d, violations=list(d["violations"]) + [{"code": "D-ZONE", "key": "x.cpp::f::zzgone#0"}])))
    A(Case("debt", "a debt entry deleted makes its violation new", lambda f: f, "FAIL", "D-ZONE", "graph_input_stage",
           edit_debt=lambda d: dict(d, violations=[v for v in d["violations"]
                                                   if not (v["code"] == "D-ZONE" and "graph_input_stage" in v["key"])])))
    # m7: data-file validation
    A(Case("m7", "a duplicate debt entry is rejected", lambda f: f, "FAIL", "data", "is listed 2 times",
           edit_debt=lambda d: dict(d, violations=list(d["violations"]) + [d["violations"][0]])))
    A(Case("m7", "a malformed debt entry fails cleanly", lambda f: f, "FAIL", "data", "malformed",
           edit_debt=lambda d: dict(d, violations=list(d["violations"]) + [{"code": "D-ZONE"}])))
    A(Case("m7", "an unknown debt code is rejected", lambda f: f, "FAIL", "data", "unknown code",
           edit_debt=lambda d: dict(d, violations=list(d["violations"]) + [{"code": "X-NOPE", "key": "a::b#0"}])))
    # fate
    A(Case("fate", "an E-RAW debt entry without a fate fails", lambda f: f, "FAIL", "data", "no valid fate",
           edit_debt=lambda d: dict(d, violations=[{k: v for k, v in e.items() if k != "fate"} if (e["code"] == "E-RAW" and "::unified_alloc::" in e["key"]) else e
                                                   for e in d["violations"]])))

    # r2 I-1: the other tier flag may only be written a literal false
    A(Case("r2i1", "device request with a conditional must_host_pinned = true", plant(
        "void zzplant_r2i1(bool b) {\n    %s req{};\n    req.intent.constraints.must_device = true;\n"
        "    req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME;\n"
        "    req.intent.constraints.forbid_vram_zone_spill = true;\n"
        "    if (b) req.intent.constraints.must_host_pinned = true;\n}\n" % REQ), "FAIL", "B-TIER", "zzplant_r2i1"))
    A(Case("r2i1", "host request with a conditional must_device = true (zone-less device passing as host)", plant(
        "void zzplant_r2i1(bool b) {\n    %s req{};\n    req.intent.constraints.must_host_pinned = true;\n"
        "    if (b) req.intent.constraints.must_device = true;\n}\n" % REQ), "FAIL", "B-TIER", "zzplant_r2i1"))
    A(Case("r2i1", "device request with the other flag written literal false (control)", plant(good_device(
        "zzplant_r2i1", "    req.intent.constraints.must_host_pinned = false;\n")), "PASS"))
    # r2 I-2: whole-object assignment is a copy
    A(Case("r2i2", "req = {} after good flags", plant(good_device("zzplant_r2i2", "    req = {};\n")),
           "FAIL", "DEFER-C", "zzplant_r2i2"))
    A(Case("r2i2", "req = other_request after good flags", plant(
        "void zzplant_r2i2(const %s & other) {\n    %s req{};\n    req.intent.constraints.must_device = true;\n"
        "    req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME;\n"
        "    req.intent.constraints.forbid_vram_zone_spill = true;\n    req = other;\n}\n" % (REQ, REQ)),
        "FAIL", "DEFER-C", "zzplant_r2i2"))
    A(Case("r2i2", "req = a request-returning call after good flags", plant(good_device(
        "zzplant_r2i2", "    req = zz_make();\n")), "FAIL", "DEFER-C", "zzplant_r2i2"))
    # r2 I-3: constructions that carry no request-type token
    A(Case("r2i3", "braced {} returned from a request-returning function", plant(
        "%s zzplant_r2i3a() {\n    return {};\n}\n" % REQ), "FAIL", "B-FORM", "zzplant_r2i3a"))
    A(Case("r2i3", "braced designated list returned from a request-returning function", plant(
        "%s zzplant_r2i3b() {\n    return {.size = 1};\n}\n" % REQ), "FAIL", "B-FORM", "zzplant_r2i3b"))
    A(Case("r2i3", "unified_allocate({}) with a braced argument", plant(
        "void zzplant_r2i3c() {\n    ggml_sycl::unified_allocate({});\n}\n"), "FAIL", "B-FORM", "zzplant_r2i3c"))
    A(Case("r2i3", "unified_alloc({}, h) with a braced argument", plant(
        "void zzplant_r2i3d(ggml_sycl::alloc_handle * h) {\n    ggml_sycl::unified_alloc({}, h);\n}\n"),
        "FAIL", "B-FORM", "zzplant_r2i3d"))
    A(Case("r2i3", "a default argument T r = {}", plant(
        "void zzplant_r2i3e(%s r = {}) {\n    (void) r;\n}\n" % REQ), "FAIL", "B-FORM", "zzplant_r2i3e"))
    A(Case("r2i3", "auto r2 = req", plant(good_device("zzplant_r2i3f", "    auto r2 = req;\n    (void) r2;\n")),
           "FAIL", "DEFER-C", "zzplant_r2i3f::declaration:r2:"))
    A(Case("r2i3", "auto r2 = std::move(req)", plant(good_device(
        "zzplant_r2i3g", "    auto r2 = std::move(req);\n    (void) r2;\n")), "FAIL", "DEFER-C", "zzplant_r2i3g::declaration:r2:"))
    A(Case("r2i3", "auto i = a request-returning call", plant(
        "void zzplant_r2i3h() {\n    auto i = ggml_sycl_transient_device_intent(\"zz\");\n    (void) i;\n}\n"),
        "FAIL", "DEFER-C", "zzplant_r2i3h::declaration:i:"))
    A(Case("r2i3", "auto & alias of a request's intent", plant(good_device(
        "zzplant_r2i3i", "    auto & in = req.intent;\n    (void) in;\n")), "FAIL", "DEFER-C", "zzplant_r2i3i::declaration:in:"))
    A(Case("r2i3", "auto of a scalar field, a pointer and a literal (control)", plant(good_device(
        "zzplant_r2i3j", "    auto n = req.size;\n    auto * p = &req;\n    auto k = 5;\n    (void) n;\n    (void) p;\n    (void) k;\n")),
        "PASS"))
    # r2 m-6: a write after the request has been handed on does not set up what that code read
    A(Case("r2m6", "zone and forbid written after unified_allocate(req)", plant(
        "void zzplant_r2m6a() {\n    %s req{};\n    req.intent.constraints.must_device = true;\n"
        "    ggml_sycl::unified_allocate(req);\n"
        "    req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME;\n"
        "    req.intent.constraints.forbid_vram_zone_spill = true;\n}\n" % REQ), "FAIL", "D-ZONE", "zzplant_r2m6a"))
    A(Case("r2m6", "forbid written after an `if (b) { unified_allocate(req); return; }`", plant(
        "void zzplant_r2m6b(bool b) {\n    %s req{};\n    req.intent.constraints.must_device = true;\n"
        "    req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME;\n"
        "    if (b) {\n        ggml_sycl::unified_allocate(req);\n        return;\n    }\n"
        "    req.intent.constraints.forbid_vram_zone_spill = true;\n}\n" % REQ), "FAIL", "D-FORBID", "zzplant_r2m6b"))
    A(Case("r2m6", "a use inside a lambda counts as a first use", plant(
        "void zzplant_r2m6c() {\n    %s req{};\n    req.intent.constraints.must_device = true;\n"
        "    req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME;\n"
        "    auto f = [&]() { ggml_sycl::unified_allocate(req); };\n"
        "    req.intent.constraints.forbid_vram_zone_spill = true;\n    f();\n}\n" % REQ), "FAIL", "D-FORBID", "zzplant_r2m6c"))
    A(Case("r2m6", "every write before the handoff (control)", plant(good_device(
        "zzplant_r2m6d", "    ggml_sycl::unified_allocate(req);\n")), "PASS"))
    A(Case("r2m6", "scalar reads between the writes are not a handoff (control)", plant(
        "void zzplant_r2m6e() {\n    %s req{};\n    req.intent.constraints.must_device = true;\n"
        "    int sz = req.size;\n    (void) sz;\n"
        "    req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME;\n"
        "    req.intent.constraints.forbid_vram_zone_spill = true;\n    ggml_sycl::unified_allocate(req);\n}\n" % REQ),
        "PASS"))
    # r2 m-7: a macro body that assigns a tracked field
    A(Case("r2m7", "#define assigning prefer_vram_zone = COUNT", plant(
        "#define ZZ_BAD(r) r.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::COUNT\n"),
        "FAIL", "B-FORM", "ZZ_BAD"))
    A(Case("r2m7", "#define comparing a tracked field is not a write (control)", plant(
        "#define ZZ_OK(r) ((r).intent.constraints.must_device == true)\n"), "PASS", planted=False))
    # r2 nit: adjacent string literals form one literal
    A(Case("r2n", "\"zeMem\" \"AllocDevice\" adjacent literals", plant(
        "void zzplant_r2n() {\n    const char * s = \"zeMem\" \"AllocDevice\";\n    (void) s;\n}\n"),
        "FAIL", "E-RAW", "zzplant_r2n"))
    A(Case("r2n", "unrelated adjacent literals (control)", plant(
        "void zzplant_r2n() {\n    const char * s = \"abc\" \"def\";\n    (void) s;\n}\n"), "PASS", planted=False))
    # r2 m-2 / m-3: allowlist and debt entry validation
    A(Case("r2m2", "an E-RAW allowlist entry without a name is rejected", lambda f: f, "FAIL", "data", "lacks name",
           edit_allowlist=lambda al: dict(al, entries=[{k: v for k, v in e.items() if k != "name"} if e["id"] == "E-TEST" else e
                                                      for e in al["entries"]])))
    A(Case("r2m3", "an E-RAW debt entry without a cite is rejected", lambda f: f, "FAIL", "data", "has no cite",
           edit_debt=lambda d: dict(d, violations=[{k: v for k, v in e.items() if k != "cite"} if (e["code"] == "E-RAW" and "::unified_alloc::" in e["key"]) else e
                                                   for e in d["violations"]])))
    A(Case("r2m3", "a one-word cite is rejected", lambda f: f, "FAIL", "data", "has no cite",
           edit_debt=lambda d: dict(d, violations=[dict(e, cite="tbd") if (e["code"] == "E-RAW" and "::unified_alloc::" in e["key"]) else e
                                                   for e in d["violations"]])))
    A(Case("r2m3", "an unknown fate is rejected", lambda f: f, "FAIL", "data", "no valid fate",
           edit_debt=lambda d: dict(d, violations=[dict(e, fate="whatever") if (e["code"] == "E-RAW" and "::unified_alloc::" in e["key"]) else e
                                                   for e in d["violations"]])))
    A(Case("r2m3", "sanctioned-vendored and pending-disposition are valid fates (control)", lambda f: f, "PASS",
           edit_debt=lambda d: dict(d, violations=[dict(e, fate="pending-disposition") if (e["code"] == "E-RAW" and "::unified_alloc::" in e["key"]) else e
                                                   for e in d["violations"]]), planted=False))
    return c


def names_new(f, code):
    return f.startswith("FAIL new %s " % code)


def evaluate_case(base_files, allowlist, debt, case):
    files = case.mutate(base_files)
    if case.expect == "PASS" and case.planted and planted_sightings(files) == 0:
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
            return f.startswith("FAIL allowlist") and case.naming in f
        if case.code == "debt":
            return f.startswith("FAIL stale debt entry") and case.naming in f
        if case.code == "data":
            return f.startswith("FAIL debt entry") and case.naming in f or f.startswith("FAIL allowlist") and case.naming in f
        return names_new(f, case.code) and case.naming in f

    return any(names(f) for f in fails), fails


def planted_sightings(files):
    """Constructions, raw hits, forms and lexical rows the gate sees in a planted function or appended text."""
    ctx = Ctx(files)
    n = 0
    for rel in files:
        if not (rel.startswith("zz") or "/zz" in rel or rel == "cpu-dispatch.cpp"):
            continue
        fa = facts_for(rel, files[rel], ctx)
        n += sum(1 for c in fa["constructions"] if "zz" in c["key"] or "zz" in c["func"])
        n += sum(1 for r in fa["lexical"] if r["var"].startswith("zz"))
        n += sum(1 for r in fa["raws"] if "zz" in r["func"])
        n += sum(1 for r in fa["forms"] if "zz" in r["func"])
    return n


def subprocess_gate(args, env_extra=None, cwd=None):
    import subprocess
    env = dict(os.environ, **(env_extra or {}))
    p = subprocess.run([sys.executable, os.path.abspath(__file__)] + args, env=env, capture_output=True, text=True, cwd=cwd)
    return p.returncode, p.stdout + p.stderr


def f_witness_missing_pack():
    """Clause (f): with the pack unimportable the gate exits 1 and names the module."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        Path(d, "tree_sitter_language_pack.py").write_text("raise ImportError('simulated: pack absent')\n")
        rc, out = subprocess_gate(["--root", "/nonexistent"], {"PYTHONPATH": d})
    return rc == 1 and "tree_sitter_language_pack" in out, "rc=%d out=%r" % (rc, out[:100])


def m6_witnesses():
    """Argument handling: contradictory flags are errors, and a regeneration cannot add entries unasked."""
    out = []
    rc, text = subprocess_gate(["--allow-growth"])
    out.append(("--allow-growth without --write-debt is an error", rc == 2, "rc=%d" % rc))
    rc, text = subprocess_gate(["--mutation-matrix", "--list"])
    out.append(("--mutation-matrix with --list is an error", rc == 2, "rc=%d" % rc))
    rc, text = subprocess_gate(["--mutation-matrix", "--write-debt"])
    out.append(("--mutation-matrix with --write-debt is an error", rc == 2, "rc=%d" % rc))
    v = [V("D-ZONE", "a.cpp::f::declaration:req:00000000#0", "a.cpp", 1, "f", "req", "m")]
    ok, msg, _ = plan_debt_write(v, {"entries": []}, {"violations": []}, False)
    out.append(("--write-debt over an empty debt list refuses to seed without --allow-growth", not ok and "ADD" in msg, msg[:40]))
    ok, msg, ents = plan_debt_write(v, {"entries": []}, {"violations": []}, True)
    out.append(("--write-debt --allow-growth seeds", ok and len(ents) == 1, msg[:40]))
    ok, msg, ents = plan_debt_write([], {"entries": []}, {"violations": [{"code": "D-ZONE", "key": "k"}]}, False)
    out.append(("--write-debt may shrink", ok and ents == [], msg[:40]))
    return out


def r2m4_witnesses(allowlist):
    """--write-debt and the gate fail cleanly (a FAIL line, rc 1, no traceback) on data files that are malformed."""
    import tempfile
    out = []
    bad_debts = [("an entry without a key", '{"violations": [{"code": "D-ZONE"}]}', "malformed"),
                 ("malformed JSON", "{not json", "not readable JSON"),
                 ("a non-object file", "[1, 2]", "must be a JSON object"),
                 ("a non-list violations field", '{"violations": 3}', "must be a list")]
    for label, text, want in bad_debts:
        with tempfile.TemporaryDirectory() as d:
            Path(d, "allowlist.json").write_text(json.dumps(allowlist))
            Path(d, "debt.json").write_text(text)
            for flag in ("--write-debt", None):
                rc, o = subprocess_gate(["--data", d] + ([flag] if flag else []))
                ok = rc == 1 and "Traceback" not in o and want in o
                out.append(("%s in debt.json: %s fails cleanly" % (label, flag or "the gate"), ok, "rc=%d %s" % (rc, o[:60].replace("\n", " "))))
    return out


def r2m8_witnesses():
    """Scope: skips anchored at the scope root, case-insensitive suffixes, a required-file list and a floor."""
    import tempfile
    out = []
    with tempfile.TemporaryDirectory() as d:
        base = Path(d, SCOPE_SUBDIR)
        for rel in ("tests/skipped.cpp", "docs/skipped.cpp", "dpct/tests/nested.cpp", "sub/docs/nested2.cpp", "UPPER.CPP",
                    "lower.HPP"):
            (base / rel).parent.mkdir(parents=True, exist_ok=True)
            (base / rel).write_text("int x;\n")
        scanned, unscanned = list_sources(d)
        out.append(("a top-level tests/ and docs/ are skipped", "tests/skipped.cpp" not in scanned and "docs/skipped.cpp" not in scanned, sorted(scanned)))
        out.append(("a nested tests/ or docs/ is scanned", "dpct/tests/nested.cpp" in scanned and "sub/docs/nested2.cpp" in scanned, sorted(scanned)))
        out.append((".CPP and .HPP are scanned", "UPPER.CPP" in scanned and "lower.HPP" in scanned, sorted(scanned)))
    for suffix in (".C", ".c", ".cl", ".CC"):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d, SCOPE_SUBDIR)
            base.mkdir(parents=True)
            (base / ("x" + suffix)).write_text("int x;\n")
            rc, o = subprocess_gate(["--root", d])
            out.append(("an unscanned %s file fails the gate, naming it" % suffix, rc == 1 and ("x" + suffix) in o, "rc=%d %s" % (rc, o[:60])))
    with tempfile.TemporaryDirectory() as d:
        base = Path(d, SCOPE_SUBDIR)
        base.mkdir(parents=True)
        for i in range(MIN_FILES + 5):
            (base / ("f%d.cpp" % i)).write_text("int x;\n")
        rc, o = subprocess_gate(["--root", d])
        out.append(("a full-size tree missing a required file fails, naming it", rc == 1 and "ggml-sycl.cpp" in o, "rc=%d %s" % (rc, o[:80])))
    with tempfile.TemporaryDirectory() as d:
        base = Path(d, SCOPE_SUBDIR)
        (base / "dpct").mkdir(parents=True)
        for f in REQUIRED_FILES:
            (base / f).write_text("int x;\n")
        rc, o = subprocess_gate(["--root", d])
        out.append(("a tree with the required files but below the floor fails", rc == 1 and "expected at least" in o, "rc=%d %s" % (rc, o[:80])))
    return out


def m9_witness():
    """A C++-ish file with an unscanned suffix is a FAIL naming it, before the file-count pin."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        base = Path(d, SCOPE_SUBDIR)
        base.mkdir(parents=True)
        (base / "a.cpp").write_text("int a;\n")
        (base / "x.inl").write_text("int b;\n")
        rc, out = subprocess_gate(["--root", d])
    return rc == 1 and ".inl" in out, "rc=%d out=%r" % (rc, out[:100])


def cmake_witness(root):
    """The ctest registration must never pass a regeneration flag."""
    text = (Path(root) / "ggml/src/ggml-sycl/CMakeLists.txt").read_text()
    blocks = [m.group(0) for m in re.finditer(r"add_test\(NAME test-sycl-alloc-zone-contract.*?\)", text, flags=re.S)]
    ok = len(blocks) == 1 and "--write-debt" not in blocks[0] and "--allow-growth" not in blocks[0] \
        and "--mutation-matrix" in blocks[0]
    props = re.findall(r"set_tests_properties\(test-sycl-alloc-zone-contract PROPERTIES[^)]*\)", text)
    ok = ok and len(props) == 1 and re.search(r"\bTIMEOUT\s+\d+", props[0]) is not None
    return ok, "%d registration(s), timeout %s" % (len(blocks), "set" if props and "TIMEOUT" in props[0] else "MISSING")


def m14_witness(files, allowlist, debt):
    """A key-matched allowlist entry covers exactly its node; it needs its debt entry gone, and fails if stale."""
    viols, _ = analyse(files)
    v = next(x for x in viols if x.code == "D-ZONE" and "graph_input_stage" in x.key)
    ent = {"id": "E-ZZ-KEY", "code": v.code, "key": v.key, "count": 1, "reason": "mutation-matrix test entry"}
    al = dict(allowlist, entries=list(allowlist["entries"]) + [ent])
    d2 = dict(debt, violations=[d for d in debt["violations"] if (d["code"], d["key"]) != v.ident()])
    fails, _ = apply_contract(viols, al, d2)
    res = [("a key-matched entry covers its node when the debt entry is removed", not fails, fails[:2])]
    fails, _ = apply_contract(viols, al, debt)
    res.append(("the same entry leaves its debt entry stale", any(f.startswith("FAIL stale debt entry") for f in fails), fails[:1]))
    al_bad = dict(allowlist, entries=list(allowlist["entries"]) + [dict(ent, key=v.key + "x")])
    fails, _ = apply_contract(viols, al_bad, d2)
    res.append(("a key-matched entry for a node that does not exist matches nothing",
                any("E-ZZ-KEY matches nothing" in f for f in fails), fails[:1]))
    return res


def run_matrix(files, allowlist, debt, root):
    fails, report, _, _ = run_gate(files, allowlist, debt)
    if fails:
        print("FAIL: the unmutated baseline is red, so a matrix scored against it proves nothing:")
        for f in fails[:20]:
            print("  " + f)
        return 1
    print("baseline: PASS (the matrix is scored against a green tree)")
    bad = 0
    cases = matrix_cases()
    for w in sorted(set(c.wid for c in cases) - set(WITNESSES)):
        print("FAIL: case witness %s is not in WITNESSES" % w)
        bad += 1
    for w in WITNESSES:
        if w in ("f", "cmake", "m6", "m9", "m14", "r2m4", "r2m8"):
            continue
        if not any(c.wid == w and c.expect == "FAIL" for c in cases):
            print("FAIL: witness %s has no FAIL case" % w)
            bad += 1
    print("deferred: " + "; ".join("witness %s (%s)" % kv for kv in WITNESSES_DEFERRED.items()))
    if not (names_new("FAIL new D-ZONE-COUNT x", "D-ZONE-COUNT") and not names_new("FAIL new D-ZONE-COUNT x", "D-ZONE")):
        print("FAIL: the matcher's own self-test (prefix collision) is wrong")
        bad += 1
    for case in cases:
        ok, got = evaluate_case(files, allowlist, debt, case)
        print("%s witness %-4s %-80s expect %s" % ("ok  " if ok else "FAIL", case.wid, case.label, case.expect))
        if not ok:
            bad += 1
            for g in got[:4]:
                print("       got: " + g)
            if not got:
                print("       got: PASS")
    extra = [("f", "missing tree_sitter_language_pack exits 1 and names it") + f_witness_missing_pack(),
             ("m9", "an unscanned .inl file fails the gate") + m9_witness(),
             ("cmake", "the ctest registration carries no regeneration flag") + cmake_witness(root)]
    extra += [("m6", label, ok, d) for label, ok, d in m6_witnesses()]
    extra += [("r2m4", label, ok, d) for label, ok, d in r2m4_witnesses(allowlist)]
    extra += [("r2m8", label, ok, d) for label, ok, d in r2m8_witnesses()]
    extra += [("m14", label, ok, d) for label, ok, d in m14_witness(files, allowlist, debt)]
    for wid, label, ok, detail in extra:
        print("%s witness %-4s %-80s (%s)" % ("ok  " if ok else "FAIL", wid, label, str(detail)[:80]))
        bad += 0 if ok else 1
    print("matrix: %d case(s), %d wrong" % (len(cases) + len(extra), bad))
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
    ap.add_argument("--allow-growth", action="store_true", help="with --write-debt, permit new entries (key migration or seeding only)")
    a = ap.parse_args()
    if a.allow_growth and not a.write_debt:
        ap.error("--allow-growth only means something with --write-debt")
    if a.mutation_matrix and (a.list or a.write_debt):
        ap.error("--mutation-matrix cannot be combined with --list or --write-debt")
    data = Path(a.data) if a.data else Path(a.root) / "scripts" / "sycl-alloc-zone-contract"
    try:
        allowlist = load_json(data / "allowlist.json", {"entries": []}, "entries")
        debt = load_json(data / "debt.json", {"violations": []}, "violations")
    except DataError as exc:
        print("FAIL: %s" % exc)
        return 1
    if a.write_debt:
        errs = validate_data(allowlist, debt)
        if errs:
            print("\n".join(errs))
            print("FAIL: --write-debt refuses to rewrite from data files that do not validate")
            return 1
    files = load_tree(a.root)
    if a.list or a.write_debt:
        viols, _ = analyse(files)
        if a.list:
            for v in viols:
                print(v)
        if a.write_debt:
            ok, msg, ents = plan_debt_write(viols, allowlist, debt, a.allow_growth)
            if not ok:
                print("FAIL: " + msg)
                return 1
            data.mkdir(parents=True, exist_ok=True)
            with open(data / "debt.json", "w") as f:
                f.write('{\n  "schema": 1,\n  "_doc": %s,\n  "violations": [\n' % json.dumps(DEBT_DOC))
                f.write(",\n".join("    " + json.dumps(e) for e in ents))
                f.write("\n  ]\n}\n")
            print("wrote %d debt entries to %s" % (len(ents), data / "debt.json"))
        return 0
    fails, report, viols, stats = run_gate(files, allowlist, debt)
    print("scope: %d files, %d constructions of %s" % (stats["files"], stats["constructions"], " ".join(stats["value_types"])))
    for r in report:
        print(r)
    by_code = {}
    for d in debt.get("violations", []):
        if isinstance(d, dict) and "code" in d:
            by_code[d["code"]] = by_code.get(d["code"], 0) + 1
    print("debt by code: " + " ".join("%s=%d" % (k, by_code[k]) for k in sorted(by_code)))
    if fails:
        for f in fails:
            print(f)
        print("FAIL: %d finding(s)" % len(fails))
        return 1
    print("PASS: no new violation, no stale debt, no stale allowlist entry")
    if a.mutation_matrix:
        return run_matrix(files, allowlist, debt, a.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
