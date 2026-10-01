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

What this unit (S2a, S2b, S2c) enforces
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
      #define body) outside the allowlist. dpct's allocating entry points, `dpct_malloc` (identifier) and
      the classes `device_memory`, `global_memory`, `constant_memory`, `shared_memory` (type names), are forbidden
      outside dpct/helper.hpp; helper.hpp's own three raw calls are allowlisted by function name and count
      (canonical contract section 9.1, the dpct row; vendored upstream, not edited, dead in-tree; rulings M247 second)
      The host side is covered too: malloc_host, aligned_alloc_host, zeMemAllocHost, the generic `sycl::malloc` and
      `sycl::aligned_alloc` (qualified by sycl:: only, because the bare names are the C library's), and the host raw
      chain's wrappers unified_cache_raw_malloc_host and unified_cache_malloc_host_tracked.
  (f) a missing tree_sitter_language_pack is a FAIL, never a skip
  (g) rethrow first: every `catch (...)`, `catch (std::exception &)` and `catch (const std::exception &)`, in any
      spelling (east const, by value, with or without std::), is preceded in its try by a ggml_sycl_fallback_error handler
      whose body is exactly `throw;`. The scope is every scanned file, which includes every file that holds a handler.
      #define bodies are scanned lexically with the same pattern, keyed (file, macro). The unguarded handlers are shrink-only
      debt (G-CATCH, keyed file::function::catch_clause:<kind>#ordinal); exemptions are allowlist entries by file and function.
  (h) cascade_step and unconverted_ticket, keyed by the construction node, never by function. Nothing writes either today,
      so every write is a finding that only an allowlist entry for that node exempts: cascade_step = true (H-CASCADE), a copy
      of the enclosing function's own cascade_step parameter (H-CASCADE-PARAM, exempt by the allowlisted callee), a call that
      passes true (H-PASS-TRUE) or forwards its own parameter (H-PASS-FORWARD) to a function with such a parameter, and an
      unconverted_ticket string literal (H-UNCONV, the entry names the ticket). Any other expression (H-CASCADE-EXPR,
      H-PASS-EXPR, H-UNCONV-EXPR) cannot be exempted. No H-* finding can be debt, and an entry's outcome may not be TERMINAL.
      An allowlisted node that stops writing its field fails as an entry matching nothing (a DECLARED row losing its flag).
      Positional initialisation of a request type and a call through a lambda or function pointer are not followed.
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
Both rules resolve a callee by name against the tree, so these stay unseen: a call through a lambda, a function
pointer or a std::function, and a call written inside a macro body (`#define ZZ() unified_allocate({})`).
A #define body that assigns a tracked field is a B-FORM finding (macro-write).

A write only counts when it is bound to the declaration (the nearest enclosing one of that name, looking
through #if branches), comes before the request is first handed to other code (passed, returned, copied,
or named in a lambda), and is not under a conditional. A whole-object assignment (`req = {}`,
`req = other;`) is a copy and defers to clause (c). A braced assignment to `intent` or `constraints`
discards the writes bound to that subtree and re-seeds it from the list. The other tier flag may only ever
be written a literal false. Write credit is positional and does not follow control flow, so a write after
`if (b) goto done;` or after an unconditional `return;` is still credited (documented gap).

  (c) interprocedural flow. A copy of a request (`T r = req;`, `r = req;`, `auto r = std::move(req)`) inherits the
      source's fields and is a construction of its own: handed on, it must write its own cohort literal
      (C-COHORT), and its own writes are judged like any request's. A copy whose source is a parameter is a
      pass-through: the caller's request was already judged, so only the negative rules (COUNT, forbid false,
      B-TIER) and C-SITE apply to it. A callee that writes through a by-reference or pointer request parameter
      is summarised by name (to a fixpoint) and its writes are applied at the caller's call. A helper whose single
      return is a locally built request or intent is summarised the same way; a helper with several returns
      stays opaque (DEFER-C). A write through a reference or pointer alias of a request or sub-object
      (`alloc_constraints & c = req.intent.constraints; c.prefer_vram_zone = COUNT;`, `p->...`, `(*p)`, a chain
      of aliases) is a write to the root. COUNT or forbid-false written anywhere else (a member, `this->`,
      a holder, a file-scope request, the body of a by-reference callee) is a stray finding, and a by-value
      request parameter's own writes are judged as a pass-through record. A wrapper from one request type to
      another (alloc_intent -> alloc_request) must copy the source's site (C-SITE). Handing the address of a
      scalar field (`zz_f(&req.intent.constraints.must_device)`) or calling a method on the request is a first
      handoff, so a later write is not credited.
      Known clause-(c) gaps: a reference or pointer bound to a SCALAR field (`bool & r = req.intent.
      constraints.must_device; r = false;`), a call through a lambda, a function pointer or a std::function, a
      call written inside a macro body, a helper with several returns, positional goto/return control flow,
      token pasting (`malloc_##x`), `#pragma message`, and the constructor-form alias `T & c(x)`.
  (i)-(p) are S2d. Witness 9 (a site that stops calling its shared `*_bytes()` function) is
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
  check-sycl-alloc-zone-contract.py [--root REPO] [--mutation-matrix [--shard K/N]]
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
# Members of alloc_constraints the rules read (the tracked fields that are not the cohort).
CONSTRAINT_FIELDS = tuple(f for f in TRACKED if f != "cohort_id")
# A wrapper that builds one site-carrying request type from another must copy these (witness 24). A write to the
# nested intent's own site is keyed `intent.site_file`, so it is never mistaken for the request's.
SITE_FIELDS = ("site_file", "site_line")
# Writes the gate refuses wherever they land, even through a member or an alias it cannot attribute to a request.
NEGATIVE_FIELDS = ("prefer_vram_zone", "forbid_vram_zone_spill")
# Clause (e). Names match as identifiers (a call, an address-of, a use as a value), strings on substring.
RAW_NAMES = (
    "unified_cache_malloc_device_tracked", "unified_cache_raw_malloc_device", "sycl_aligned_malloc_device",
    "unified_cache_malloc_host_tracked", "unified_cache_raw_malloc_host",
    "ggml_sycl_malloc_device_raw", "ggml_sycl_free_device_raw", "unified_cache_raw_free_device",
    "malloc_device", "malloc_shared", "malloc_host", "aligned_alloc_device", "aligned_alloc_shared", "aligned_alloc_host",
    "zeMemAllocDevice", "zeMemAllocShared", "zeMemAllocHost", "zePhysicalMemCreate", "zeVirtualMemReserve",
)
RAW_STRINGS = ("zeMemAllocDevice", "zeMemAllocShared", "zeMemAllocHost", "zePhysicalMemCreate", "zeVirtualMemReserve")
# The generic USM entry points take the kind as an argument (`sycl::malloc(n, q, usm::alloc::host)`). The bare names are the
# C library's too, so they count only when qualified by sycl::.
RAW_QUALIFIED = ("malloc", "aligned_alloc")
# dpct's allocating entry points are raw allocators too (helper.hpp is vendored and holds the three raw calls they reach).
# Outside that file the function is matched as an identifier and the classes as type names, never as substrings: a
# parameter that happens to be spelled `device_memory` (memory-budget.hpp) is an identifier, not the class.
DPCT_HOME = "dpct/helper.hpp"
DPCT_FUNCS = ("dpct_malloc",)
DPCT_CLASSES = ("device_memory", "global_memory", "constant_memory", "shared_memory")

SHARDS = 4   # the ctest registers this many shards; cmake_witness pins the registration to it

DEBT_DOC = ("Read by scripts/check-sycl-alloc-zone-contract.py (clauses a-h). Shrink-only: a violation not listed "
            "fails, and a listed entry that no longer violates fails. Every E-RAW entry carries a fate (deleted-by-*, "
            "converted-by-*, sanctioned-internal, sanctioned-vendored or pending-disposition) and a cite, so an entry no step will ever "
            "shrink is visible as a mislabelled allowlist entry. Regenerate with `python3 "
            "scripts/check-sycl-alloc-zone-contract.py --write-debt` (it refuses to add entries); see README.md.")
FATE_RE = re.compile(r"^(deleted-by|converted-by)-[A-Za-z0-9._§()-]+$|^sanctioned-internal$|^sanctioned-vendored$|^pending-disposition$")
CITE_MIN = 12   # a cite names a ticket or a design/census row; "tbd" is not one

# Clause (g): a handler that can swallow ggml_sycl_fallback_error must be preceded, in the same try, by a handler for it
# whose body is exactly `throw;`. One pattern for the AST and for macro bodies, so no spelling escapes (a const in either
# position, with or without std::, with or without the &).
HANDLER_RE = re.compile(r"catch\s*\(\s*(\.\.\.|(const\s+)?(?:::)?(?:std::)?exception(?![A-Za-z0-9_])(\s+const)?\s*&?)")
GUARD_RE = re.compile(r"catch\s*\(\s*(const\s+)?(?:::)?(?:ggml_sycl::)?ggml_sycl_fallback_error(?![A-Za-z0-9_])(\s+const)?\s*&?\s*"
                      r"(?:[A-Za-z_][A-Za-z0-9_]*\s*)?\)\s*\{\s*throw\s*;\s*\}")
# Clause (h): the two class fields every writer must name by node.
H_FIELDS = ("cascade_step", "unconverted_ticket")
# What each code of clause (h) may be exempted by: a node key (or the callee's file and function for H-CASCADE-PARAM).
# The expression forms are never exempt, and no clause-(h) finding may sit in the debt list.
H_CODES = ("H-CASCADE", "H-CASCADE-PARAM", "H-CASCADE-EXPR", "H-PASS-TRUE", "H-PASS-FORWARD", "H-PASS-EXPR", "H-UNCONV",
           "H-UNCONV-EXPR")
H_NEVER_EXEMPT = ("H-CASCADE-EXPR", "H-PASS-EXPR", "H-UNCONV-EXPR")
H_OUTCOME = {"H-CASCADE": "CASCADE", "H-CASCADE-PARAM": "CASCADE", "H-PASS-TRUE": "CASCADE", "H-PASS-FORWARD": "CASCADE",
             "H-UNCONV": "UNCONVERTED"}

CODES = ("A-ERROR", "A-LEXICAL", "A-TOKEN", "B-BRACE", "B-FORM", "B-TIER", "C-COHORT", "C-SITE", "D-ZONE",
         "D-ZONE-COUNT", "D-FORBID", "D-FORBID-FALSE", "E-RAW", "G-CATCH", "DEFER-C") + H_CODES


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
_MEMBERS = {}    # sha1 -> file_members rows
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
            byval = True
            q = parent(n)
            while q is not None and kind(q) in ("pointer_declarator", "reference_declarator"):
                byval = False
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


def file_members(src):
    """(declared base type, member name) of every field declaration in a file; cached, since Ctx is rebuilt per case."""
    h = _sha(src)
    if h not in _MEMBERS:
        out = []
        for n in walk(parse(src)):
            if kind(n) == "field_declaration":
                t = fld(n, "type")
                if t is not None:
                    for d in kids(n):
                        if not same(d, t) and kind(d) == "field_identifier":
                            out.append((base_type(txt(src, t)), txt(src, d)))
        _MEMBERS[h] = out
    return _MEMBERS[h]


_IDENTS = {}           # sha1 -> every identifier spelled in a file
_STRUCT_MEMBERS = {}   # sha1 -> {struct name: [(member name, member base type)]}
_DEFS = {}             # sha1 -> [(name, return base type, returns by value, function_definition node)]


def file_idents(src):
    """Every identifier a file spells, for deciding which call summaries can reach it; cached."""
    h = _sha(src)
    if h not in _IDENTS:
        _IDENTS[h] = frozenset(txt(src, n) for n in walk(parse(src)) if kind(n) in ("identifier", "field_identifier"))
    return _IDENTS[h]


def file_struct_members(src):
    """Member names, in declaration order, of every struct in a file: a positional initialiser maps onto them."""
    h = _sha(src)
    if h not in _STRUCT_MEMBERS:
        out = {}
        for n in walk(parse(src)):
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
                for d in kids(c):
                    if same(d, t) or kind(d) not in ("field_identifier", "init_declarator", "pointer_declarator",
                                                     "reference_declarator", "array_declarator"):
                        continue
                    mn = declared_name(src, d)
                    if mn:
                        members.append((mn, base_type(txt(src, t))))
            out[txt(src, nm)] = members
        _STRUCT_MEMBERS[h] = out
    return _STRUCT_MEMBERS[h]


def file_defs(src):
    """Function definitions of one file: (name, return base type, returns by value, node)."""
    h = _sha(src)
    if h not in _DEFS:
        out = []
        for n in walk(parse(src)):
            if kind(n) != "function_definition":
                continue
            d = fld(n, "declarator")
            byval = True
            while d is not None and kind(d) != "function_declarator":
                if kind(d) in ("pointer_declarator", "reference_declarator"):
                    byval = False
                d = fld(d, "declarator")
            if d is None or fld(d, "declarator") is None or fld(n, "body") is None:
                continue
            t = fld(n, "type")
            out.append((callee_last(txt(src, fld(d, "declarator"))), base_type(txt(src, t)) if t is not None else None,
                        byval, n))
        _DEFS[h] = out
    return _DEFS[h]


_CASCADE_FUNCS = {}    # sha1 -> {function name: [index of a parameter named cascade_step]}


def file_cascade_funcs(src):
    """Functions of one file that take a parameter named `cascade_step`, by parameter position. A file that never spells
    the name skips the walk."""
    h = _sha(src)
    if h not in _CASCADE_FUNCS:
        out = {}
        if b"cascade_step" in src:
            for n in walk(parse(src)):
                if kind(n) != "function_declarator" or fld(n, "declarator") is None or fld(n, "parameters") is None:
                    continue
                ps = [c for c in kids(fld(n, "parameters")) if kind(c) in ("parameter_declaration", "optional_parameter_declaration")]
                for i, c in enumerate(ps):
                    if param_name(src, c) == "cascade_step":
                        out.setdefault(callee_last(txt(src, fld(n, "declarator"))), []).append(i)
        _CASCADE_FUNCS[h] = out
    return _CASCADE_FUNCS[h]


def has_braced_arg(src, lst):
    """True when an argument list or initialiser list has a braced-init-list element. The grammar reads a bare
    `{}` argument as a compound literal of no type, so that shape counts too."""
    for c in kids(lst):
        if kind(c) == "initializer_list":
            return True
        if kind(c) == "compound_literal_expression" and (fld(c, "type") is None or not txt(src, fld(c, "type")).strip()):
            return True
    return False


def literal_content(src, c):
    """The characters between a string literal's delimiters (a raw string's delimiter and parentheses excluded)."""
    return "".join(txt(src, x) for x in kids(c) if kind(x) in ("string_content", "raw_string_content", "escape_sequence"))


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
            for tbase, name in file_members(src):
                if tbase in self.value_types:
                    self.req_members.add(name)
        self.cascade_funcs = {}
        for src in files.values():
            for name, idx in file_cascade_funcs(src).items():
                self.cascade_funcs.setdefault(name, set()).update(idx)
        self.member_order = {}
        for src in files.values():
            for sname, members in file_struct_members(src).items():
                if sname in self.all_types:
                    self.member_order[sname] = members
        self.site_types = {t for t, m in self.member_order.items() if {"site_file", "site_line"} <= {x[0] for x in m}}
        # What a call tells the caller about a request, found by name and refined to a fixpoint: the constructions a
        # helper returns (`ret_sum`) and the writes a callee makes through a request it takes by reference (`ev_sum`).
        self.digest = hashlib.sha1(repr((sorted(self.value_types), sorted(self.all_types),
                                         sorted(self.member_order.items()), sorted(self.site_types))).encode()).hexdigest()
        self.ret_sum, self.ev_sum = {}, {}
        build_summaries(self, files)

    def file_key(self, names):
        """What a file that spells `names` can learn from the tree: the request types, and the function, member and
        call-summary facts under the names it mentions. A function added elsewhere does not move its facts."""
        return hashlib.sha1(repr((self.digest, sorted(self.req_funcs & names), sorted(self.req_returning & names),
                                  sorted(self.req_members & names),
                                  sorted((n, sorted(v)) for n, v in self.cascade_funcs.items() if n in names),
                                  [(n, self.ret_sum[n]) for n in sorted(self.ret_sum) if n in names],
                                  [(n, self.ev_sum[n]) for n in sorted(self.ev_sum) if n in names])).encode()).hexdigest()


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


def str_class(src, n):
    """'lit' for a string literal (or a ternary whose arms all are), 'null' for nullptr, else 'expr'."""
    k = kind(n)
    if k == "parenthesized_expression":
        inner = [c for c in kids(n) if _a(c, "is_named")]
        return str_class(src, inner[0]) if inner else "expr"
    if k == "conditional_expression":
        arms = [str_class(src, fld(n, "consequence")), str_class(src, fld(n, "alternative"))]
        return "lit" if arms == ["lit", "lit"] else "expr"
    if k in ("string_literal", "concatenated_string", "raw_string_literal"):
        return "lit"
    return "null" if txt(src, n).strip() in ("nullptr", "NULL", "0") else "expr"


def classify(src, field, n):
    if field in ("cohort_id", "unconverted_ticket"):
        return str_class(src, n)
    if field == "prefer_vram_zone":
        return zone_class(src, n)
    if field in ("must_device", "must_host_pinned", "forbid_vram_zone_spill", "cascade_step"):
        return bool_class(src, n)
    return "expr"


def add_write(out, field, pos, cls, cond, text):
    out.setdefault(field, []).append((pos, cls, cond, re.sub(r"\s+", " ", text)[:80]))


def merge_known(out, known, pos, which):
    """Fold a resolved copy source's writes into `out`, in order, at `pos`. A whole-request or constraints copy
    leaves the destination's own cohort alone: the cohort is exactly what a copy must name for itself."""
    for f, ws in known.items():
        if f == "cohort_id" and which != "intent":
            continue
        for i, w in enumerate(ws):
            out.setdefault(f, []).append((pos + i / 1000.0, w[1], w[2], w[3]))


def init_fields(env, lst, stype, out, copied, pfx=""):
    """Fold a braced initialiser into `out` (field -> [(pos, class, conditional, text)]). `stype` is the struct the
    list initialises, so a positional element maps onto the member in that position. A struct-valued member that is
    not itself a list is resolved against the copy sources the scan can follow, or recorded as a copy."""
    src = env.src
    order = env.ctx.member_order.get(stype) or []
    ordinal = 0

    def element(name, val):
        if val is None:
            return
        if kind(val) == "initializer_list":
            member_type = {"intent": "alloc_intent", "constraints": "alloc_constraints"}.get(name)
            init_fields(env, val, member_type, out, copied, "intent." if name == "intent" else pfx)
        elif name in STRUCT_FIELDS:
            r = resolve_source(env, val, name)
            if r is None:
                copied.append(name)
            else:
                merge_known(out, r[1], sb(val), name)
        elif name in TRACKED:
            add_write(out, name, sb(val), classify(src, name, val), False, txt(src, val))
        elif name in SITE_FIELDS:
            add_write(out, pfx + name, sb(val), "expr", False, txt(src, val))

    for c in kids(lst):
        k = kind(c)
        if k == "initializer_pair":
            desig = [x for x in kids(c) if kind(x) == "field_designator"]
            name = None
            if desig:
                fi = [x for x in kids(desig[-1]) if kind(x) == "field_identifier"]
                name = txt(src, fi[0]) if fi else None
            element(name, fld(c, "value"))
        elif k in ("{", "}", ",", "comment"):
            continue
        else:
            name = order[ordinal][0] if ordinal < len(order) else None
            ordinal += 1
            if name is None:
                copied.append("positional")
                continue
            element(name, c)


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


def unparen(n):
    """Strip any number of enclosing parentheses: `(req)` is `req`."""
    while n is not None and kind(n) == "parenthesized_expression":
        inner = [c for c in kids(n) if _a(c, "is_named")]
        if len(inner) != 1:
            break
        n = inner[0]
    return n


def strip_ref(n):
    """The object an expression names: look through parentheses, `*p` and `&x` at any depth."""
    while True:
        n = unparen(n)
        if n is not None and kind(n) == "pointer_expression" and fld(n, "argument") is not None:
            n = fld(n, "argument")
            continue
        return n


def lhs_chain(src, lhs):
    """(root identifier node or None, fields written outermost first) of an assignment's left side, looking through
    parentheses and `*p` at every level."""
    chain = []
    x = strip_ref(lhs)
    while x is not None and kind(x) == "field_expression":
        f = fld(x, "field")
        chain.append(txt(src, f) if f is not None else "?")
        x = strip_ref(fld(x, "argument"))
    return (x if x is not None and kind(x) == "identifier" else None), chain


def assignments_in(src_key, src, block):
    """Every assignment in a block whose left side is an identifier or a field chain rooted at one:
    [(root name, rhs node, assignment node, is compound)]."""
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
        x, _ = lhs_chain(src, fld(n, "left"))
        if x is None:
            continue
        out.append((txt(src, x), fld(n, "right"), n, op[0] != "="))
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
    while p is not None and (kind(p) == "parenthesized_expression"
                             or (kind(p) == "field_expression" and same(fld(p, "argument"), top))):
        if kind(p) == "field_expression":
            f = fld(p, "field")
            last = txt(src, f) if f is not None else "?"
        top = p
        p = parent(top)
    if last is not None and p is not None:
        # `zz(&req.intent.constraints.must_device)` can write the flag later, and `req.m()` can write anything
        if kind(p) == "pointer_expression" and fld(p, "argument") is not None and same(fld(p, "argument"), top) \
                and txt(src, kids(p)[0]) == "&":
            return True
        if kind(p) == "call_expression" and same(fld(p, "function"), top):
            return True
    if last is not None and last not in STRUCT_FIELDS:
        return False
    if p is not None and kind(p) == "assignment_expression" and same(fld(p, "left"), top):
        return False
    return True


def request_valued(src, ctx, node, auto_names):
    """True when an initialiser expression evidently has a request type: a request variable or parameter, a member
    chain ending in a request member or intent/constraints, std::move of one, or a call returning a request."""
    k = kind(node)
    if k == "parenthesized_expression":
        inner = [c for c in kids(node) if _a(c, "is_named")]
        return bool(inner) and request_valued(src, ctx, inner[0], auto_names)
    if k == "conditional_expression":
        return any(request_valued(src, ctx, fld(node, f), auto_names) for f in ("consequence", "alternative")
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
            return len(args) == 1 and request_valued(src, ctx, args[0], auto_names)
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


# ---------------------------------------------------------------- interprocedural follow (clause c)

def fn_params(n):
    """The parameter_declaration nodes of a function definition or a lambda."""
    d = fld(n, "declarator")
    while d is not None:
        pl = fld(d, "parameters")
        if pl is not None:
            return [c for c in kids(pl) if kind(c) in ("parameter_declaration", "optional_parameter_declaration")]
        d = fld(d, "declarator")
    return []


def param_name(src, c):
    dd = fld(c, "declarator")
    return declared_name(src, dd) if dd is not None else None


def param_decl_of(src, node, name):
    """The parameter declaration a name refers to at `node`, looking outward through lambdas to the function."""
    q = parent(node)
    while q is not None:
        if kind(q) in ("function_definition", "lambda_expression"):
            for c in fn_params(q):
                if param_name(src, c) == name:
                    return c
            if kind(q) == "function_definition":
                return None
        q = parent(q)
    return None


def bound_decl(src, node, name):
    """The declaration or parameter a name binds to at `node`."""
    d = binding_decl(src, node, name)
    return d if d is not None else param_decl_of(src, node, name)


def param_style(src, c, ctx):
    """'value', 'ref' or 'ptr' for a writable parameter of a request type, else None (a const one cannot be written)."""
    t = fld(c, "type")
    if t is None or base_type(txt(src, t)) not in ctx.value_types:
        return None
    if any(kind(x) == "type_qualifier" and txt(src, x) == "const" for x in kids(c)):
        return None
    dd = fld(c, "declarator")
    if dd is None:
        return None
    if kind(dd) == "reference_declarator":
        return "ref"
    if kind(dd) == "pointer_declarator":
        return "ptr"
    return "value" if declared_name(src, dd) else None


_REFDECL_CACHE = {}
_CALL_CACHE = {}


def ref_decls_in(src_key, src, block):
    """Reference and pointer declarations with an initialiser in a block: [(declaration, declarator, name, value)]."""
    key = (src_key, sb(block), eb(block))
    if key not in _REFDECL_CACHE:
        out = []
        for n in walk(block):
            if kind(n) != "declaration":
                continue
            for d in kids(n):
                if kind(d) != "init_declarator":
                    continue
                top, val = fld(d, "declarator"), fld(d, "value")
                if top is not None and val is not None and kind(top) in PTR_KINDS:
                    nm = declared_name(src, d)
                    if nm:
                        out.append((n, d, nm, val))
        _REFDECL_CACHE[key] = out
    return _REFDECL_CACHE[key]


def calls_in(src_key, src, block):
    """Every call in a block: [(call node, callee's last name, [argument nodes])]."""
    key = (src_key, sb(block), eb(block))
    if key not in _CALL_CACHE:
        out = []
        for n in walk(block):
            if kind(n) != "call_expression":
                continue
            fn, args = fld(n, "function"), fld(n, "arguments")
            if fn is None or args is None:
                continue
            out.append((n, callee_last(txt(src, fn)), [c for c in kids(args) if _a(c, "is_named") and kind(c) != "comment"]))
        _CALL_CACHE[key] = out
    return _CALL_CACHE[key]


def ref_root(src, val):
    """(root identifier, fields) of an initialiser naming a request or a sub-object of one (`req`, `&req`,
    `req.intent.constraints`), or None."""
    x = strip_ref(val)
    chain = []
    while x is not None and kind(x) == "field_expression":
        f = fld(x, "field")
        chain.append(txt(src, f) if f is not None else "?")
        x = strip_ref(fld(x, "argument"))
    if x is None or kind(x) != "identifier" or any(c not in STRUCT_FIELDS for c in chain):
        return None
    return x, chain


def arg_root(src, a):
    """The identifier a call argument is rooted at: `req`, `&req`, `req.intent`, `*p`, or None."""
    x = strip_ref(a)
    while x is not None and kind(x) == "field_expression":
        x = strip_ref(fld(x, "argument"))
    return x if x is not None and kind(x) == "identifier" else None


def braced_of(src, ctx, n):
    """The initializer_list of a braced value: `{...}`, the grammar's empty-type reading of a bare `{...}`, or `T{...}`."""
    if n is None:
        return None
    if kind(n) == "initializer_list":
        return n
    if kind(n) == "compound_literal_expression":
        t, v = fld(n, "type"), fld(n, "value")
        if v is not None and kind(v) == "initializer_list" and (
                t is None or not txt(src, t).strip() or base_type(txt(src, t)) in ctx.value_types):
            return v
    return None


class Env:
    """One file's scan state: the request locals built so far (a copy resolves its source against them) and the
    assignments that some request has claimed."""

    def __init__(self, rel, src, ctx):
        self.rel, self.src, self.ctx, self.h = rel, src, ctx, _sha(src)
        self.auto_names = set()
        self.built = {}      # (declaration start byte, name) -> construction record
        self.handled = set()  # start bytes of assignments attributed to a request or an alias of one
        self.done = {}       # declaration start byte -> its records (a source declared after its copy is built on demand)
        self.building = set()

    def key_for(self, func, nodekind, var, text):
        """Key without its ordinal; analyse() numbers identical keys among the constructions that violate."""
        return "%s::%s::%s:%s:%s" % (self.rel, func, nodekind, var, text_hash(text))


def resolve_source(env, val, which):
    """What the source of a copy evaluates to: ('known', fields, passthrough) for a local request this scan built or a
    helper's single returned construction, ('pass', {}, True) for a parameter (the caller's own request, judged at the
    caller's construction), or None when the gate cannot follow it. `which` names the destination: request, intent or
    constraints."""
    src, ctx = env.src, env.ctx
    n = unparen(val)
    if n is None:
        return None
    if kind(n) == "call_expression":
        fn = fld(n, "function")
        if fn is None:
            return None
        ft = re.sub(r"\s+", "", txt(src, fn))
        if ft in ("std::move", "std::forward", "move", "forward"):
            args = [c for c in kids(fld(n, "arguments")) if _a(c, "is_named")]
            return resolve_source(env, args[0], which) if len(args) == 1 else None
        rs = ctx.ret_sum.get(callee_last(ft))
        return ("known", rs, False) if rs is not None else None
    chain = []
    x = n
    while x is not None and kind(x) == "field_expression":
        f = fld(x, "field")
        chain.append(txt(src, f) if f is not None else "?")
        x = strip_ref(fld(x, "argument"))
    if x is None or kind(x) != "identifier" or any(c not in STRUCT_FIELDS for c in chain):
        return None
    name = txt(src, x)
    d = binding_decl(src, x, name)
    if d is not None:
        rec = env.built.get((sb(d), name))
        if rec is None and kind(d) == "declaration" and sb(d) not in env.building and sb(d) not in env.done:
            env.building.add(sb(d))
            process_declaration(env, d)
            env.building.discard(sb(d))
            rec = env.built.get((sb(d), name))
        if rec is None or rec["deferred"] or rec["err"]:
            return None
        return ("known", rec["fields"], rec["pass"])
    p = param_decl_of(src, x, name)
    if p is not None and base_type(txt(src, fld(p, "type"))) in ctx.value_types:
        return ("pass", {}, True)
    return None


def reset_subtree(fields, which):
    """A braced or copied assignment to `intent` or `constraints` discards what was written under it."""
    for f in list(fields):
        if which == "intent":
            if f not in SITE_FIELDS:
                del fields[f]
        elif f in CONSTRAINT_FIELDS:
            del fields[f]


def writes_for(env, decl, name, blk, fields, copied, reason, st, callee_mode, stype):
    """Fold into `fields` every write bound to the object declared at `decl`: assignments through the name or through a
    reference or pointer alias of it, and the writes a callee makes through a by-reference parameter. Returns the
    statement texts of the assignments and whether the object is ever handed to other code. `callee_mode` is a
    by-reference parameter's own summary: whole-object assignments there are opaque to the caller."""
    src, h, ctx = env.src, env.h, env.ctx
    names = [(name, decl)]
    spans = []
    for dn, d, an_name, val in ref_decls_in(h, src, blk):
        if sb(dn) < eb(decl):
            continue
        r = ref_root(src, val)
        if r is None:
            continue
        ident = r[0]
        nm = txt(src, ident)
        if any(n0 == nm and same(bound_decl(src, ident, nm), d0) for n0, d0 in names):
            names.append((an_name, dn))
            spans.append((sb(val), eb(val)))

    def binds(node, nm):
        for n0, d0 in names:
            if n0 == nm and sb(node) >= eb(d0) and same(bound_decl(src, node, nm), d0):
                return d0
        return None

    first_use = None
    for nm, dnode in names:
        for ident in idents_in(h, src, blk).get(nm, []):
            if sb(ident) < eb(dnode) or not same(bound_decl(src, ident, nm), dnode):
                continue
            if any(a <= sb(ident) < b for a, b in spans):
                continue  # the alias's own initialiser names the object without handing it anywhere
            if is_use(src, ident, dnode):
                first_use = sb(ident) if first_use is None else min(first_use, sb(ident))
                break
    items = []
    for root_name, rhs, an, compound in assignments_in(h, src, blk):
        d0 = binds(an, root_name)
        if d0 is not None:
            items.append((sb(an), 0, (rhs, an, compound, d0)))
    for call, cname, al in calls_in(h, src, blk):
        if cname not in ctx.ev_sum:
            continue
        for idx, a in enumerate(al):
            ident = arg_root(src, a)
            d0 = binds(ident, txt(src, ident)) if ident is not None else None
            if d0 is not None:
                items.append((sb(ident), 1, (call, idx, ident, d0)))
    items.sort(key=lambda x: (x[0], x[1]))
    texts = []
    for pos, which_item, payload in items:
        if which_item == 1:
            call, idx, ident, d0 = payload
            hits = [dfn for dfn in ctx.ev_sum[callee_last(txt(src, fld(call, "function")))] if idx in dfn["events"]]
            late = first_use is not None and pos > first_use
            cond_call = is_conditional(call, blk, d0) or late or len(hits) > 1
            for dfn in hits:
                evs, cps = dfn["events"][idx]
                for f, ws in evs.items():
                    for i, (cls, cnd, text) in enumerate(ws):
                        add_write(fields, f, pos + i / 1000.0, cls, cnd or cond_call, text)
                copied.extend(cps)
            continue
        rhs, an, compound, d0 = payload
        if not callee_mode:
            env.handled.add(sb(an))  # a by-reference parameter's own writes stay unclaimed: a COUNT there is refused in place
        _, chain = lhs_chain(src, fld(an, "left"))
        last = chain[0] if chain else None
        texts.append(txt(src, an))
        late = first_use is not None and pos > first_use
        cond = is_conditional(an, blk, d0)
        clean = not compound and not cond and not late and not callee_mode and rhs is not None
        if last is None:
            bl = braced_of(src, ctx, rhs) if clean else None
            if bl is not None:
                fields.clear()
                del copied[:], reason[:]
                init_fields(env, bl, stype, fields, copied)
                st["pass"], st["copy"] = False, False
                continue
            r = resolve_source(env, rhs, "request") if clean else None
            if r is None:
                copied.append("whole-assign")
            else:
                fields.clear()
                del copied[:], reason[:]
                merge_known(fields, r[1], pos, "request")
                st["pass"], st["copy"] = r[2], stype != "alloc_constraints"
        elif last in STRUCT_FIELDS:
            sub = "alloc_intent" if last == "intent" else "alloc_constraints"
            bl = braced_of(src, ctx, rhs) if clean else None
            if bl is not None:
                reset_subtree(fields, last)
                init_fields(env, bl, sub, fields, copied)
                st["pass"] = False  # the list re-seeds what the caller's request carried; it is judged here now
                continue
            r = resolve_source(env, rhs, last) if clean else None
            if r is None:
                copied.append(last)
            else:
                reset_subtree(fields, last)
                merge_known(fields, r[1], pos, last)
                st["pass"] = r[2]
        elif last in TRACKED:
            add_write(fields, last, pos, "expr" if compound else classify(src, last, rhs), cond or late,
                      txt(src, rhs) if rhs is not None else "")
        elif last in SITE_FIELDS:
            key = "intent." + last if "intent" in chain[1:] else last
            add_write(fields, key, pos, "expr", cond or late, txt(src, rhs) if rhs is not None else "")
    return texts, first_use is not None


def process_declaration(env, n):
    """The construction records of one declaration or member declaration (one per declarator)."""
    if sb(n) in env.done:
        return env.done[sb(n)]
    env.done[sb(n)] = recs_out = []
    src, ctx = env.src, env.ctx
    k = kind(n)
    t = fld(n, "type")
    auto_decl = t is not None and k == "declaration" and kind(t) == "placeholder_type_specifier"
    if t is None or (not auto_decl and base_type(txt(src, t)) not in ctx.value_types):
        return []
    func = enclosing(src, n)
    if k == "field_declaration":
        holder = parent(parent(n))
        hn = fld(holder, "name") if holder is not None and kind(holder) in ("struct_specifier", "class_specifier") else None
        if hn is not None and txt(src, hn) in BASE_TYPES:
            return []  # the request types' own members define their defaults; they construct nothing
    err = has_error_ancestor(n)
    ks = kids(n)
    stype = None if auto_decl else base_type(txt(src, t))
    which = {"alloc_intent": "intent", "alloc_constraints": "constraints"}.get(stype, "request")
    fnode = parent(n)
    while fnode is not None and kind(fnode) != "function_definition":
        fnode = parent(fnode)
    recs = recs_out
    for ki, d in enumerate(ks):
        if same(d, t):
            continue
        if auto_decl:
            # `auto x = <request>` copies a request without naming its type
            name = declared_name(src, d)
            init = fld(d, "value") if kind(d) == "init_declarator" else None
            if name is None or init is None or not request_valued(src, ctx, init, env.auto_names):
                continue
            env.auto_names.add((name, sb(n)))
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
        st = {"pass": False, "copy": False}
        if form == "braced":
            init_fields(env, val, stype, fields, copied)
        elif form == "copy":
            r = resolve_source(env, val, which)
            if r is None:
                reason.append("copy-init")
            else:
                merge_known(fields, r[1], sb(val), which)
                st["pass"], st["copy"] = r[2], stype != "alloc_constraints"
        elif form == "ctor":
            reason.append("ctor-args")
        texts, handed = [], False
        if k == "declaration" and in_function(n) and not err:
            blk = scope_block(n)
            if blk is not None:
                texts, handed = writes_for(env, n, name, blk, fields, copied, reason, st, False, stype)
        for c in copied:
            reason.append(c + "-copied" if c != "positional" else "positional-init")
        if k == "field_declaration" and form == "braced" and not fields and not reason:
            # `T x{}` in a holder struct is storage whose value is set elsewhere (clause (c));
            # only a member initialiser that carries a flag is a construction of its own
            continue
        for f in fields:
            fields[f].sort()
        inputs = []
        if k == "declaration" and fnode is not None and stype in ctx.site_types:
            for c in fn_params(fnode):
                pt, pn = fld(c, "type"), param_name(src, c)
                if pt is not None and pn and base_type(txt(src, pt)) in ctx.site_types and base_type(txt(src, pt)) != stype:
                    inputs.append(pn)
        rec = {
            "key": env.key_for(func, kind(n), name, txt(src, n) + " ;; " + " ;; ".join(texts)), "func": func,
            "var": name, "line": line_of(n), "kind": "member" if k == "field_declaration" else "decl",
            "type": "auto" if auto_decl else stype, "form": form, "err": err, "fields": fields,
            "deferred": sorted(set(reason + (["auto-init"] if auto_decl else []))),
            "pass": st["pass"], "needs_cohort": bool(st["copy"] and not st["pass"] and handed),
            "wrapper_inputs": inputs,
        }
        if k == "declaration":
            env.built[(sb(n), name)] = rec
        recs.append(rec)
    return recs


def owner_fn(n):
    p = parent(n)
    while p is not None and kind(p) not in ("function_definition", "lambda_expression"):
        p = parent(p)
    return p


_SUMCACHE = {}


def memo_summary(tag, fn, ctx, rel, src, fnode):
    """A summary depends on the function's file and on the summaries of the names that file spells."""
    key = (tag, _sha(src), sb(fnode), ctx.file_key(file_idents(src)))
    if key not in _SUMCACHE:
        _SUMCACHE[key] = fn(ctx, rel, src, fnode)
    return _SUMCACHE[key]


def return_fields(ctx, rel, src, fnode):
    """The fields of the one construction a helper returns by name, or None when the helper is not that simple."""
    env = Env(rel, src, ctx)
    body = fld(fnode, "body")
    rets = []
    for n in walk(body):
        if kind(n) == "declaration" and owner_fn(n) is not None and same(owner_fn(n), fnode):
            process_declaration(env, n)
        elif kind(n) == "return_statement" and owner_fn(n) is not None and same(owner_fn(n), fnode):
            rets.append(n)
    if len(rets) != 1:
        return None
    exprs = [c for c in kids(rets[0]) if _a(c, "is_named")]
    x = unparen(exprs[0]) if len(exprs) == 1 else None
    if x is None or kind(x) != "identifier":
        return None
    d = binding_decl(src, x, txt(src, x))
    rec = env.built.get((sb(d), txt(src, x))) if d is not None else None
    if rec is None or rec["deferred"] or rec["err"]:
        return None
    return {f: list(ws) for f, ws in rec["fields"].items()}


def param_events(ctx, rel, src, fnode):
    """What a function does to each request it takes by reference or pointer: {param index: ({field: [(class,
    conditional, text)]}, copy reasons)}."""
    env = Env(rel, src, ctx)
    body = fld(fnode, "body")
    for n in walk(body):
        if kind(n) == "declaration":
            process_declaration(env, n)
    out = {}
    for idx, c in enumerate(fn_params(fnode)):
        if param_style(src, c, ctx) not in ("ref", "ptr"):
            continue
        fields, copied, reason, st = {}, [], [], {"pass": False, "copy": False}
        writes_for(env, c, param_name(src, c), body, fields, copied, reason, st, True, base_type(txt(src, fld(c, "type"))))
        evs = {f: [(w[1], w[2], w[3]) for w in sorted(ws)] for f, ws in fields.items()}
        out[idx] = (evs, sorted(set(copied)))
    return {"events": out}


def build_summaries(ctx, files):
    """Fill ctx.ret_sum and ctx.ev_sum by iterating to a fixpoint (a helper may use another helper)."""
    ret_defs, ev_defs = {}, {}
    for rel, src in files.items():
        for name, ret, byval, fnode in file_defs(src):
            if ret in ctx.value_types and byval:
                ret_defs.setdefault(name, []).append((rel, src, fnode))
            if any(param_style(src, c, ctx) in ("ref", "ptr") for c in fn_params(fnode)):
                ev_defs.setdefault(name, []).append((rel, src, fnode))
    for _ in range(5):
        before = repr((sorted(ctx.ret_sum.items()), sorted(ctx.ev_sum.items())))
        ret = {}
        for name, defs in ret_defs.items():
            rs = memo_summary("ret", return_fields, ctx, *defs[0]) if len(defs) == 1 else None
            if rs is not None:
                ret[name] = rs
        ctx.ret_sum = ret
        ctx.ev_sum = {name: [memo_summary("ev", param_events, ctx, *d) for d in defs] for name, defs in ev_defs.items()}
        if repr((sorted(ctx.ret_sum.items()), sorted(ctx.ev_sum.items()))) == before:
            break


# Roles that are known and need no finding; the others are B-FORM.
FINDING_ROLES = ("array", "new", "template-arg", "base-class", "cast", "range-for-copy", "unclassified", "default-arg",
                 "braced-return", "braced-arg", "macro-write")
MACRO_WRITE_RE = re.compile(r"(?:\.|->)\s*(?:" + "|".join(TRACKED) + r")\s*(?:<<|>>|[|&^+*/%-])?=(?!=)")
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


def param_and_stray_records(env, root):
    """Writes the declaration pass cannot attribute to a local construction. A by-value request parameter is a
    pass-through construction of its own; a by-reference one is the caller's object (its writes are summarised for
    the callers). What no request claims -- a member written in a method or through `this->`, an alias of a member --
    is still refused when it writes COUNT to the zone or false to the forbid, and so is a helper that does so through a
    request it takes by reference (its callers fail too, since they inherit the write)."""
    src, ctx = env.src, env.ctx
    out = []
    for fnode in (d[3] for d in file_defs(src)):
        body = fld(fnode, "body")
        for c in fn_params(fnode):
            style = param_style(src, c, ctx)
            if style is None:
                continue
            name = param_name(src, c)
            stype = base_type(txt(src, fld(c, "type")))
            fields, copied, reason, st = {}, [], [], {"pass": True, "copy": False}
            texts, _ = writes_for(env, c, name, body, fields, copied, reason, st, style != "value", stype)
            if style != "value" or not texts:
                continue
            for f in fields:
                fields[f].sort()
            reason = ["positional-init" if x == "positional" else x + "-copied" for x in copied]
            func = enclosing(src, body)
            out.append({
                "key": env.key_for(func, kind(c), name, txt(src, c) + " ;; " + " ;; ".join(texts)), "func": func,
                "var": name, "line": line_of(c), "kind": "param", "type": stype, "form": "param",
                "err": has_error_ancestor(c), "fields": fields, "deferred": sorted(set(reason)), "pass": True,
                "needs_cohort": False, "wrapper_inputs": [],
            })
    for an in walk(root):
        if kind(an) != "assignment_expression" or sb(an) in env.handled:
            continue
        _, chain = lhs_chain(src, fld(an, "left"))
        if not chain or chain[0] not in NEGATIVE_FIELDS:
            continue
        rhs = fld(an, "right")
        cls = classify(src, chain[0], rhs) if rhs is not None else "expr"
        op = [txt(src, x) for x in kids(an) if not _a(x, "is_named")]
        if (chain[0] == "prefer_vram_zone" and cls != "count") or (chain[0] == "forbid_vram_zone_spill" and cls != "false"):
            continue
        if not op or op[0] != "=":
            continue
        func = enclosing(src, an)
        out.append({
            "key": env.key_for(func, "assignment_expression", txt(src, fld(an, "left")), txt(src, an)), "func": func,
            "var": txt(src, fld(an, "left")), "line": line_of(an), "kind": "stray", "type": "stray", "form": "stray",
            "err": has_error_ancestor(an), "fields": {chain[0]: [(sb(an), cls, False, txt(src, rhs))]}, "deferred": [],
            "pass": True, "needs_cohort": False, "wrapper_inputs": [],
        })
    return out


COMMENT_RE = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)


def handler_kind(text):
    """'all' for `catch (...)`, 'exception' for a std::exception handler of any spelling, else None."""
    m = HANDLER_RE.match(text)
    return None if m is None else ("all" if m.group(1) == "..." else "exception")


def catch_records(env, n):
    """The record of one catch clause that can swallow ggml_sycl_fallback_error: guarded when an earlier clause of the same
    try handles that type and rethrows with nothing else in its body."""
    src = env.src
    params = fld(n, "parameters")
    head = "catch" + (txt(src, params) if params is not None else "")
    hk = handler_kind(head)
    if hk is None:
        return []
    guarded = False
    p = parent(n)
    for c in (kids(p) if p is not None else []):
        if kind(c) != "catch_clause" or sb(c) >= sb(n):
            continue
        if GUARD_RE.fullmatch(COMMENT_RE.sub("", txt(src, c)).strip()):
            guarded = True
    return [{"func": enclosing(src, n), "kind": hk, "line": line_of(n), "guarded": guarded, "form": "clause"}]


def macro_catch_records(src, n, name):
    """Handlers spelled inside a #define body (tree-sitter keeps the body as one text node). A guard counts when it comes
    earlier in the body with no `try` between it and the handler."""
    body = COMMENT_RE.sub(" ", txt(src, fld(n, "value")).replace("\\\n", " "))
    out = []
    guards = [g.end() for g in GUARD_RE.finditer(body)]
    for m in HANDLER_RE.finditer(body):
        guarded = any(e <= m.start() and not re.search(r"\btry\b", body[e:m.start()]) for e in guards)
        out.append({"func": "#define " + name, "kind": "all" if m.group(1) == "..." else "exception", "line": line_of(n),
                    "guarded": guarded, "form": "macro"})
    return out


def own_param(src, ident):
    """True when `ident` names the enclosing function's own parameter called cascade_step (not a shadowing local)."""
    d = bound_decl(src, ident, "cascade_step")
    return d is not None and kind(d) in ("parameter_declaration", "optional_parameter_declaration")


def h_key_base(env, n, root, func):
    """Key of the construction node a write belongs to: the declaration of the object whose field is written, so a new
    construction planted beside a listed one is a different node. A write through a member, `this` or a call result is
    keyed by its own left side."""
    src = env.src
    if root is not None:
        d = bound_decl(src, root, txt(src, root))
        if d is not None:
            return env.key_for(func, "declaration" if kind(d) == "declaration" else "parameter", txt(src, root), txt(src, d))
    return env.key_for(func, "assignment", "<lhs>", txt(src, fld(n, "left")))


def h_records(env, n, k):
    """Writes of cascade_step / unconverted_ticket and the calls that pass a cascade_step parameter, as plain records."""
    src, ctx = env.src, env.ctx
    func = enclosing(src, n)
    if k == "assignment_expression":
        root, chain = lhs_chain(src, fld(n, "left"))
        if not chain or chain[0] not in H_FIELDS:
            return []
        field, rhs = chain[0], unparen(fld(n, "right"))
        compound = [txt(src, c) for c in kids(n) if not _a(c, "is_named")][:1] != ["="]
        base, text = h_key_base(env, n, root, func), txt(src, n)
    elif k == "initializer_pair":
        desig = [x for x in kids(n) if kind(x) == "field_designator"]
        fi = [x for x in kids(desig[-1]) if kind(x) == "field_identifier"] if desig else []
        field = txt(src, fi[0]) if fi else None
        if field not in H_FIELDS or fld(n, "value") is None:
            return []
        rhs, compound, text = unparen(fld(n, "value")), False, txt(src, n)
        q = parent(n)
        while q is not None and kind(q) not in ("declaration", "field_declaration", "expression_statement", "return_statement",
                                                "function_definition", "translation_unit"):
            q = parent(q)
        if q is not None and kind(q) in ("declaration", "field_declaration"):
            var = next((declared_name(src, d) for d in kids(q) if sb(d) <= sb(n) < eb(d) and declared_name(src, d)), "<temp>")
            base = env.key_for(func, "declaration", var, txt(src, q))
        else:
            base = env.key_for(func, "init", "<temp>", txt(src, q) if q is not None and kind(q) != "translation_unit" else text)
    elif k == "call_expression" and ctx.cascade_funcs:
        fn, al = fld(n, "function"), fld(n, "arguments")
        idxs = ctx.cascade_funcs.get(callee_last(txt(src, fn))) if fn is not None else None
        if not idxs or al is None:
            return []
        args = [c for c in kids(al) if _a(c, "is_named") and kind(c) != "comment"]
        out = []
        for i in sorted(idxs):
            if i >= len(args):
                continue  # the default is passed
            a = unparen(args[i])
            t = txt(src, a).strip()
            cls = "true" if t == "true" else "default" if t == "false" else \
                "forward" if kind(a) == "identifier" and t == "cascade_step" and own_param(src, a) else "expr"
            out.append({"form": "pass", "field": "cascade_step", "cls": cls, "func": func, "line": line_of(n),
                        "base": env.key_for(func, "call_expression", callee_last(txt(src, fn)), txt(src, n)),
                        "name": callee_last(txt(src, fn))})
        return out
    else:
        return []
    if field == "cascade_step":
        c = "expr" if compound else bool_class(src, rhs)
        if c == "expr" and not compound and kind(rhs) == "identifier" and txt(src, rhs) == "cascade_step" and own_param(src, rhs):
            c = "param"
        name = ""
    else:
        c = "expr" if compound else str_class(src, rhs)
        direct = kind(rhs) in ("string_literal", "concatenated_string", "raw_string_literal")
        c = "expr" if c == "lit" and not direct else c
        name = ""
        if c == "lit":
            name = "".join(literal_content(src, x) for x in (kids(rhs) if kind(rhs) == "concatenated_string" else [rhs]))
    return [{"form": "write", "field": field, "cls": c, "func": func, "line": line_of(n), "base": base, "name": name}]


def h_findings(r):
    """(code, message) of one clause-(h) record, or None when it is the default or a reset."""
    f, c = r["field"], r["cls"]
    if r["form"] == "write" and f == "cascade_step":
        if c == "true":
            return "H-CASCADE", "cascade_step = true is written at a construction that is not an allowlisted node"
        if c == "param":
            return "H-CASCADE-PARAM", "cascade_step is copied from the enclosing function's own parameter, which is only " \
                                      "allowed in an allowlisted callee"
        if c == "expr":
            return "H-CASCADE-EXPR", "cascade_step is written from an expression; only the literal true at an allowlisted node " \
                                     "or the enclosing allowlisted callee's own cascade_step parameter may be"
    elif r["form"] == "write":
        if c == "lit":
            return "H-UNCONV", "unconverted_ticket is set to %r at a construction that is not an allowlisted node for that ticket" \
                % r["name"]
        if c == "expr":
            return "H-UNCONV-EXPR", "unconverted_ticket is set from something that is not a string literal"
    else:
        if c == "true":
            return "H-PASS-TRUE", "a call passes cascade_step = true at a node that is not allowlisted"
        if c == "forward":
            return "H-PASS-FORWARD", "a call forwards the enclosing function's cascade_step parameter at a node that is not " \
                                     "an allowlisted forward"
        if c == "expr":
            return "H-PASS-EXPR", "a call passes a cascade_step that is neither the default nor the literal true nor the " \
                                  "enclosing function's own parameter"
    return None


def sycl_qualified_raw(src, n):
    """True for the name of a `sycl::malloc` / `sycl::aligned_alloc` call (with or without template arguments)."""
    if kind(n) != "identifier" or txt(src, n) not in RAW_QUALIFIED:
        return False
    p = parent(n)
    if p is not None and kind(p) == "template_function":
        p = parent(p)
    if p is None or kind(p) != "qualified_identifier":
        return False
    sc = fld(p, "scope")
    return sc is not None and txt(src, sc).split("::")[-1].strip() == "sycl"


def scan_file(rel, src, ctx):
    """Facts about one file, as plain data: constructions, form findings, raw hits, error tokens."""
    value_types, all_types = ctx.value_types, ctx.all_types
    env = Env(rel, src, ctx)
    root = parse(src)
    root_error = kind(root) == "ERROR"
    constructions, raws, errtoks, forms = [], [], [], []
    catches, hrecs = [], []
    h_on = b"cascade_step" in src or b"unconverted_ticket" in src or any(nm.encode() in src for nm in ctx.cascade_funcs)

    for n in walk(root):
        k = kind(n)
        if k == "catch_clause":
            catches.extend(catch_records(env, n))
        elif h_on and k in ("assignment_expression", "initializer_pair", "call_expression"):
            hrecs.extend(h_records(env, n, k))
        if k == "declaration":
            # `scoped_unified_alloc s({.size = 1});` and `s{ {.size = 1} };` build a request inside a constructor call
            tt = fld(n, "type")
            if tt is not None and callee_last(txt(src, tt)) in ctx.req_funcs:
                for d in kids(n):
                    v = fld(d, "value") if kind(d) == "init_declarator" else None
                    if v is not None and kind(v) in ("argument_list", "initializer_list") and has_braced_arg(src, v):
                        forms.append({"func": enclosing(src, n), "role": "braced-arg", "tok": "{...}",
                                      "line": line_of(n), "text": txt(src, n)})
        elif k == "new_expression":
            tt = fld(n, "type")
            lists = [c for c in kids(n) if kind(c) in ("argument_list", "initializer_list")]
            if tt is not None and callee_last(txt(src, tt)) in ctx.req_funcs and any(has_braced_arg(src, c) for c in lists):
                forms.append({"func": enclosing(src, n), "role": "braced-arg", "tok": "{...}", "line": line_of(n),
                              "text": txt(src, n)})
        if k in ("declaration", "field_declaration"):
            constructions.extend(process_declaration(env, n))
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
                init_fields(env, val, base_type(txt(src, t)), fields, copied)
            elif k == "call_expression":
                args = fld(n, "arguments")
                if args is not None and any(_a(c, "is_named") for c in kids(args)):
                    reason.append("ctor-args")
            for f in fields:
                fields[f].sort()
            tn = base_type(txt(src, t))
            constructions.append({
                "key": env.key_for(func, kind(n), "<temp %s>" % tn, txt(src, n)), "func": func, "var": "<temp>",
                "line": line_of(n), "kind": "temp", "type": tn, "form": "braced", "err": has_error_ancestor(n),
                "fields": fields,
                "deferred": sorted(set(reason + ["positional-init" if c == "positional" else c + "-copied" for c in copied])),
                "pass": False, "needs_cohort": False, "wrapper_inputs": [],
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
                and has_braced_arg(src, fld(n, "arguments")) \
                and fld(n, "function") is not None and callee_last(txt(src, fld(n, "function"))) in ctx.req_funcs:
            forms.append({"func": enclosing(src, n), "role": "braced-arg", "tok": "{...}", "line": line_of(n),
                          "text": txt(src, n)})
        elif k in ("string_literal", "raw_string_literal", "concatenated_string"):
            if k != "concatenated_string" and parent(n) is not None and kind(parent(n)) == "concatenated_string":
                continue  # judged as part of the whole concatenation
            s = txt(src, n)
            joined = "".join(literal_content(src, c) for c in kids(n) if kind(c) in ("string_literal", "raw_string_literal")) \
                if k == "concatenated_string" else literal_content(src, n)
            for r in RAW_STRINGS:
                if r in joined or r in s:
                    raws.append({"func": enclosing(src, n), "name": r, "line": line_of(n), "form": "string",
                                 "nodekind": k, "text": s, "err": has_error_ancestor(n)})
        elif k in ("preproc_def", "preproc_function_def"):
            body = fld(n, "value")
            nm = fld(n, "name")
            if body is not None and nm is not None:
                catches.extend(macro_catch_records(src, n, txt(src, nm)))
                b = txt(src, body)
                for r in RAW_NAMES + tuple("sycl::" + q for q in RAW_QUALIFIED):
                    if re.search(r"(?<![A-Za-z0-9_])" + re.escape(r).replace("sycl::", r"sycl\s*::\s*") + r"(?![A-Za-z0-9_])", b):
                        raws.append({"func": "#define " + txt(src, nm), "name": r, "line": line_of(n),
                                     "form": "macro", "nodekind": "preproc", "text": r, "err": False})
                if MACRO_WRITE_RE.search(b):
                    forms.append({"func": "#define " + txt(src, nm), "role": "macro-write", "tok": txt(src, nm),
                                  "line": line_of(n), "text": txt(src, n)})
        elif k in ("identifier", "field_identifier") and (txt(src, n) in RAW_NAMES or sycl_qualified_raw(src, n)):
            pk = kind(parent(n)) if parent(n) is not None else ""
            if pk != "function_declarator":  # the name of a declaration or definition is not a use
                top, is_call = callee_ident(src, n)
                raws.append({"func": enclosing(src, n), "name": txt(src, n), "line": line_of(n),
                             "form": "call" if is_call else "name",
                             "nodekind": "call_expression" if is_call else "identifier",
                             "text": txt(src, top) if is_call else txt(src, n), "err": has_error_ancestor(n)})
        if rel != DPCT_HOME and k in ("identifier", "field_identifier", "type_identifier") \
                and (txt(src, n) in DPCT_FUNCS or (k == "type_identifier" and txt(src, n) in DPCT_CLASSES)) \
                and kind(parent(n)) != "function_declarator":
            top, is_call = callee_ident(src, n)
            raws.append({"func": enclosing(src, n), "name": txt(src, n), "line": line_of(n),
                         "form": "call" if is_call else "name",
                         "nodekind": "call_expression" if is_call else k,
                         "text": txt(src, top) if is_call else txt(src, n), "err": has_error_ancestor(n)})
        if k in ("type_identifier", "identifier") and txt(src, n) in all_types:
            if has_error_ancestor(n) or _a(n, "is_missing"):
                errtoks.append({"func": enclosing(src, n), "tok": txt(src, n), "line": line_of(n)})
            else:
                role = token_role(src, n, value_types)
                if role in FINDING_ROLES:
                    forms.append({"func": enclosing(src, n), "role": role, "tok": txt(src, n), "line": line_of(n),
                                  "text": txt(src, statement_of(n))})

    constructions.extend(param_and_stray_records(env, root))

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
    return {"constructions": constructions, "raws": raws, "errtoks": errtoks, "forms": forms, "catches": catches,
            "hrecs": hrecs, "root_error": root_error, "lexical": lexical, "lex_count": lex_count, "ast_count": ast_count}


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
    key = (rel, _sha(src), ctx.file_key(file_idents(src)))
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
    for inp in [c["wrapper_inputs"]] if c["wrapper_inputs"] else []:
        # a wrapper that builds one site-carrying request type from another copies the input's site (witness 24)
        miss = [f for f in SITE_FIELDS
                if not any(not x[2] and any(re.search(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(i), x[3]) for i in inp)
                           for x in w.get(f, []))]
        if miss:
            out.append(("C-SITE", "built from %s but %s not copied from it by an unconditional assignment; the allocator "
                                  "would report this wrapper's line, not the caller's construction"
                        % (", ".join(inp), " and ".join(miss) + (" is" if len(miss) == 1 else " are"))))
    if c["needs_cohort"] and not established(w.get("cohort_id", []), "lit"):
        out.append(("C-COHORT", "a copy that is handed on must assign its own cohort_id literal; it inherits its source's "
                                "site, so without one its trace and refusal lines read as the original's"))
    if c["deferred"]:
        out.append(("DEFER-C", "built from a copy or call (%s); clause (c) will follow it" % ",".join(c["deferred"])))
        return out
    if c["form"] == "stray" or c["pass"]:
        return out  # a pass-through's tier and zone are the caller's, judged at the caller's construction
    if dev and host:
        out.append(("B-TIER", "both must_device and must_host_pinned are established true; the tier is not decidable"))
        return out
    if not dev and not host:
        out.append(("B-TIER", "neither must_device nor must_host_pinned is established by an unconditional literal true as its "
                              "last write; unified_select_tier may turn it into HOST"))
        return out
    other_name = "must_device" if host else "must_host_pinned"
    ows = w.get(other_name, [])
    if ows and not (ows[-1][1] == "false" and not ows[-1][2]):
        # a write that is a conditional, a non-literal or a true can make the request the other tier unless an
        # unconditional literal false comes after it (a copy may inherit such a write and overwrite it)
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
        for r in fa["catches"]:
            if r["guarded"]:
                continue
            base = "%s::%s::catch_%s:%s" % (rel, r["func"], "macro" if r["form"] == "macro" else "clause", r["kind"])
            i = seen.get(base, 0)
            seen[base] = i + 1
            viols.append(V("G-CATCH", "%s#%d" % (base, i), rel, r["line"], r["func"], r["kind"],
                           "a %s handler is not preceded in its try by `catch (const ggml_sycl_fallback_error &) { throw; }`, "
                           "so it can swallow a planned refusal" % ("catch (...)" if r["kind"] == "all" else "std::exception")))
        seen = {}
        for r in fa["hrecs"]:
            found = h_findings(r)
            if found is None:
                continue
            base = "%s::%s:%s" % (r["base"], r["form"], r["field"])
            i = seen.get(base, 0)
            seen[base] = i + 1
            viols.append(V(found[0], "%s#%d" % (base, i), rel, r["line"], r["func"], r.get("name", ""), found[1]))
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
        if e["code"] in H_NEVER_EXEMPT:
            errs.append("FAIL allowlist entry %s: %s cannot be exempted; only the literal true at an allowlisted node or the "
                        "enclosing callee's own parameter is a cascade_step write" % (e["id"], e["code"]))
        if e["code"] in H_OUTCOME:
            if e.get("outcome") != H_OUTCOME[e["code"]]:
                errs.append("FAIL allowlist entry %s needs outcome %s (a TERMINAL row may set neither cascade_step nor "
                            "unconverted_ticket)" % (e["id"], H_OUTCOME[e["code"]]))
            if e["code"] != "H-CASCADE-PARAM" and "key" not in e:
                errs.append("FAIL allowlist entry %s must be keyed by the construction or call node, never by function" % e["id"])
        if e["code"] == "H-UNCONV" and not (isinstance(e.get("ticket"), str) and e["ticket"].strip()):
            errs.append("FAIL allowlist entry %s needs the ticket its construction names" % e["id"])
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
        if d["code"] in H_CODES:
            errs.append("FAIL debt entry %s %s: a clause-(h) finding is allowlisted per node or fixed, never debt" % (d["code"], d["key"]))
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
        return v.key == ent["key"] and ("ticket" not in ent or v.name == ent["ticket"])
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
    bad = sorted({v.ident() for v in rest if v.code in H_CODES})
    if bad:
        return False, "clause-(h) findings cannot be debt; fix them or allowlist each node:\n%s" % "\n".join(
            "  %s %s" % b for b in bad[:20]), []
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
                 edit_debt=None, planted=True, allow_nodes=None, allow_from=None):
        self.wid, self.label, self.mutate, self.expect = wid, label, mutate, expect
        self.code, self.naming, self.allowlist = code, naming, allowlist
        self.edit_allowlist, self.edit_debt, self.planted = edit_allowlist, edit_debt, planted
        # allow_nodes: [{"code", "func", "nth", "extra"}]. Each becomes a key-matched allowlist entry for the node the gate
        # finds in the tree `allow_from` builds (default: the case's own), so a node can be listed and then mutated away.
        self.allow_nodes, self.allow_from = allow_nodes, allow_from


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
    "r3i1": "a braced assignment to intent/constraints discards earlier writes",
    "r3i2": "braced arguments in constructor and new forms", "r3m1": "parenthesised left sides",
    "r3n": "raw strings split across adjacent literals; compound macro writes",
    "f": "missing tree_sitter_language_pack exits 1", "cmake": "the ctest registrations: the plain gate and all four shards, their TIMEOUTs and labels, no regeneration flag",
    "10": "a helper writing COUNT through a by-reference request fails at the caller",
    "11": "a helper writing forbid false (or a conditional other tier) through a by-reference request fails at the caller",
    "17": "a copy is a construction of its own and must assign its own cohort literal",
    "24": "a wrapper building one request type from another copies the input's site_file and site_line",
    "s2b-alias": "a write through a reference or pointer alias of a request is a write to it",
    "s2b-member": "COUNT or forbid false written to a member, through this->, a holder or a by-value parameter",
    "s2b-firstuse": "the address of a scalar field or a method call on the request is a first handoff",
    "s2b-dpct": "dpct entry points are forbidden names outside dpct/helper.hpp; its three sites are pinned by function and count",
    "s2b-chain": "the raw chain's three links are allowlisted by name and exact count",
    "15": "a catch (...) or std::exception handler without the rethrow clause before it",
    "16": "cascade_step = true outside an allowlisted construction node", "16b": "a call passing cascade_step = true outside an allowlisted node",
    "21": "a DECLARED construction that loses its cascade_step leaves its entry matching nothing",
    "22": "unconverted_ticket is shrink-only and keyed by the construction node", "25": "a forward of cascade_step outside an allowlisted forward",
    "26": "cascade_step = <expr> other than the allowlisted callee's own parameter", "34": "a rethrow-less handler inside a #define body",
    "s2c-host": "host raw allocator names (malloc_host, aligned_alloc_host, zeMemAllocHost, sycl::malloc, the host chain's wrappers)",
    "s2c-catch": "spellings and placements of the rethrow clause", "s2c-data": "clause-(h) entries that must be refused by validation",
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
    A(Case("r2i2", "req = {} after good flags (the reset discards them)", plant(good_device("zzplant_r2i2", "    req = {};\n")),
           "FAIL", "B-TIER", "zzplant_r2i2"))
    A(Case("r2i2", "req = other_request after good flags (a local source with no flags discards them)", plant(
        "void zzplant_r2i2() {\n    %s other{};\n    %s req{};\n    req.intent.constraints.must_device = true;\n"
        "    req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME;\n"
        "    req.intent.constraints.forbid_vram_zone_spill = true;\n    req = other;\n}\n" % (REQ, REQ)),
        "FAIL", "B-TIER", "zzplant_r2i2"))
    A(Case("r2i2", "req = a parameter after good flags is a pass-through of the caller's request (control)", plant(
        "void zzplant_r2i2(const %s & other) {\n    %s req{};\n    req.intent.constraints.must_device = true;\n"
        "    req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME;\n"
        "    req.intent.constraints.forbid_vram_zone_spill = true;\n    req = other;\n}\n" % (REQ, REQ)), "PASS"))
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
    # r3 I-1: a braced assignment to a sub-struct discards what was written there
    for lab, stmt in (("req.intent = {}", "req.intent = {};"),
                      ("req.intent.constraints = {}", "req.intent.constraints = {};"),
                      ("req.intent = { .role = X }", "req.intent = { .role = ggml_sycl::alloc_role::STAGING };"),
                      ("req.intent = { .constraints = {} }", "req.intent = { .constraints = {} };"),
                      ("(req.intent.constraints) = {}", "(req.intent.constraints) = {};")):
        A(Case("r3i1", "%s after good writes" % lab, plant(good_device("zzplant_r3i1", "    %s\n" % stmt)),
               "FAIL", "B-TIER", "zzplant_r3i1"))
    A(Case("r3i1", "a conditional braced reset after good writes", plant(
        "void zzplant_r3i1(bool b) {\n    %s req{};\n    req.intent.constraints.must_device = true;\n"
        "    req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME;\n"
        "    req.intent.constraints.forbid_vram_zone_spill = true;\n    if (b) req.intent = {};\n}\n" % REQ),
        "FAIL", "DEFER-C", "zzplant_r3i1"))
    A(Case("r3i1", "a braced constraints assignment that re-seeds every flag (control)", plant(
        "void zzplant_r3i1() {\n    %s req{};\n    req.intent.constraints = { .must_device = true, "
        ".prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME, .forbid_vram_zone_spill = true };\n}\n" % REQ), "PASS"))
    A(Case("r3i1", "a braced intent assignment that re-seeds every flag (control)", plant(
        "void zzplant_r3i1() {\n    %s req{};\n    req.intent = { .constraints = { .must_device = true, "
        ".prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME, .forbid_vram_zone_spill = true } };\n}\n" % REQ), "PASS"))
    # r3 I-2: braced request arguments to a constructor, a brace-init and new
    A(Case("r3i2", "scoped_unified_alloc s({.size = 1})", plant(
        "void zzplant_r3i2a() {\n    ggml_sycl::scoped_unified_alloc s({.size = 1});\n}\n"), "FAIL", "B-FORM", "zzplant_r3i2a"))
    A(Case("r3i2", "scoped_unified_alloc s{ {.size = 1} }", plant(
        "void zzplant_r3i2b() {\n    ggml_sycl::scoped_unified_alloc s{ {.size = 1} };\n}\n"), "FAIL", "B-FORM", "zzplant_r3i2b"))
    A(Case("r3i2", "new scoped_unified_alloc({.size = 1})", plant(
        "void zzplant_r3i2c() {\n    auto * p = new ggml_sycl::scoped_unified_alloc({.size = 1});\n    (void) p;\n}\n"),
        "FAIL", "B-FORM", "zzplant_r3i2c"))
    A(Case("r3i2", "scoped_unified_alloc over a named request (control)", plant(good_device(
        "zzplant_r3i2d", "    ggml_sycl::scoped_unified_alloc s(req);\n")), "PASS"))
    # r3 M-1: parentheses around the left side
    A(Case("r3m1", "(req) = other after good writes", plant(
        "void zzplant_r3m1a() {\n    %s o{};\n    %s req{};\n    req.intent.constraints.must_device = true;\n"
        "    req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME;\n"
        "    req.intent.constraints.forbid_vram_zone_spill = true;\n    (req) = o;\n}\n" % (REQ, REQ)),
        "FAIL", "B-TIER", "zzplant_r3m1a"))
    A(Case("r3m1", "(req).intent...forbid = false after good writes", plant(good_device(
        "zzplant_r3m1b", "    (req).intent.constraints.forbid_vram_zone_spill = false;\n")), "FAIL", "D-FORBID-FALSE", "zzplant_r3m1b"))
    A(Case("r3m1", "every write through parentheses (control)", plant(
        "void zzplant_r3m1c() {\n    %s req{};\n    (req).intent.constraints.must_device = true;\n"
        "    ((req).intent).constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME;\n"
        "    (req.intent.constraints).forbid_vram_zone_spill = true;\n    ggml_sycl::unified_allocate(req);\n}\n" % REQ),
        "PASS"))
    # r3 nits: a raw string split across adjacent literals; compound macro writes
    A(Case("r3n", "R\"(zeMem)\" \"AllocDevice\"", plant(
        "void zzplant_r3n() {\n    const char * s = R\"(zeMem)\" \"AllocDevice\";\n    (void) s;\n}\n"),
        "FAIL", "E-RAW", "zzplant_r3n"))
    A(Case("r3n", "a #define with a compound assignment to a tracked field", plant(
        "#define ZZ_MUL(r) r.intent.constraints.cascade_step *= 2\n"), "FAIL", "B-FORM", "ZZ_MUL"))
    A(Case("r3n", "a #define with <<= on a tracked field", plant(
        "#define ZZ_SHL(r) r.intent.constraints.cascade_step <<= 1\n"), "FAIL", "B-FORM", "ZZ_SHL"))
    A(Case("r3n", "a #define with a relational compare is not a write (control)", plant(
        "#define ZZ_LE(r) (r.intent.constraints.cascade_step <= 1 && r.intent.constraints.cascade_step != 2)\n"),
        "PASS", planted=False))
    # ---- S2b: clause (c). The value is followed through by-reference callees, copies, helper returns and aliases.
    ZN = "ggml_sycl::vram_zone_id::"
    DEVF = ("    req.intent.constraints.must_device = true;\n"
            "    req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME;\n"
            "    req.intent.constraints.forbid_vram_zone_spill = true;\n")

    def caller(name, stmts, flags=DEVF, handoff="    ggml_sycl::unified_allocate(req);\n", params=""):
        return "void %s(%s) {\n    %s req{};\n%s%s%s}\n" % (name, params, REQ, flags, stmts, handoff)

    # 10: a helper that writes COUNT into a request it takes by reference fails at the caller
    h10 = "static void zzhelper_w10(%s & r) {\n    r.intent.constraints.prefer_vram_zone = " + ZN + "COUNT;\n}\n"
    A(Case("10", "a helper writing COUNT through a reference parameter, called after good flags",
           plant(h10 % REQ + caller("zzplant_w10", "    zzhelper_w10(req);\n")), "FAIL", "D-ZONE-COUNT", "zzplant_w10"))
    A(Case("10", "the same helper reached through a pointer parameter and `&req`", plant(
        "static void zzhelper_w10(%s * r) {\n    r->intent.constraints.prefer_vram_zone = " % REQ + ZN + "COUNT;\n}\n"
        + caller("zzplant_w10", "    zzhelper_w10(&req);\n")), "FAIL", "D-ZONE-COUNT", "zzplant_w10"))
    A(Case("10", "the helper takes the constraints sub-object and the caller passes req.intent.constraints", plant(
        "static void zzhelper_w10(ggml_sycl::alloc_constraints & c) {\n    c.prefer_vram_zone = " + ZN + "COUNT;\n}\n"
        + caller("zzplant_w10", "    zzhelper_w10(req.intent.constraints);\n")), "FAIL", "D-ZONE-COUNT", "zzplant_w10"))
    A(Case("10", "a helper that forwards the request to a second helper that writes COUNT", plant(
        h10 % REQ + "static void zzhelper_w10b(%s & r) {\n    zzhelper_w10(r);\n}\n" % REQ
        + caller("zzplant_w10", "    zzhelper_w10b(req);\n")), "FAIL", "D-ZONE-COUNT", "zzplant_w10"))
    A(Case("10", "a helper whose COUNT write sits under a condition", plant(
        "static void zzhelper_w10(%s & r, bool b) {\n    if (b) r.intent.constraints.prefer_vram_zone = " % REQ + ZN + "COUNT;\n}\n"
        + caller("zzplant_w10", "    zzhelper_w10(req, true);\n")), "FAIL", "D-ZONE-COUNT", "zzplant_w10"))
    A(Case("10", "a helper writing a non-COUNT zone through a reference (control)", plant(
        "static void zzhelper_w10(%s & r) {\n    r.intent.constraints.prefer_vram_zone = " % REQ + ZN + "KV;\n}\n"
        + caller("zzplant_w10", "    zzhelper_w10(req);\n")), "PASS"))
    A(Case("10", "a helper that establishes the tier, zone and forbid for a bare request (control)", plant(
        "static void zzhelper_w10(%s & r) {\n    r.intent.constraints.must_device = true;\n    r.intent.constraints.prefer_vram_zone = " % REQ
        + ZN + "RUNTIME;\n    r.intent.constraints.forbid_vram_zone_spill = true;\n}\n"
        + caller("zzplant_w10", "    zzhelper_w10(req);\n", flags="")), "PASS"))
    A(Case("10", "a helper taking the request by const reference writes nothing (control)", plant(
        "static void zzhelper_w10(const %s & r) {\n    (void) r;\n}\n" % REQ + caller("zzplant_w10", "    zzhelper_w10(req);\n")),
        "PASS"))
    A(Case("10", "a helper that writes through a reference after the request was handed to an allocator", plant(
        h10 % REQ + caller("zzplant_w10", "", handoff="    ggml_sycl::unified_allocate(req);\n    zzhelper_w10(req);\n")),
        "FAIL", "D-ZONE-COUNT", "zzplant_w10"))
    # 11: a helper that writes forbid_vram_zone_spill = false fails at the caller
    h11 = "static void zzhelper_w11(%s & r) {\n    r.intent.constraints.forbid_vram_zone_spill = false;\n}\n"
    A(Case("11", "a helper writing forbid = false through a reference parameter", plant(
        h11 % REQ + caller("zzplant_w11", "    zzhelper_w11(req);\n")), "FAIL", "D-FORBID-FALSE", "zzplant_w11"))
    A(Case("11", "a helper writing a conditional must_host_pinned = true makes the tier undecidable", plant(
        "static void zzhelper_w11(%s & r, bool b) {\n    if (b) r.intent.constraints.must_host_pinned = true;\n}\n" % REQ
        + caller("zzplant_w11", "    zzhelper_w11(req, true);\n")), "FAIL", "B-TIER", "zzplant_w11"))
    A(Case("11", "a helper writing forbid = true through a reference (control)", plant(
        "static void zzhelper_w11(%s & r) {\n    r.intent.constraints.forbid_vram_zone_spill = true;\n}\n" % REQ
        + caller("zzplant_w11", "    zzhelper_w11(req);\n", flags=DEVF.replace("    req.intent.constraints.forbid_vram_zone_spill = true;\n", ""))),
        "PASS"))
    A(Case("11", "a helper that reassigns the whole request it was given is opaque to the caller", plant(
        "static void zzhelper_w11(%s & r) {\n    r = {};\n}\n" % REQ + caller("zzplant_w11", "    zzhelper_w11(req);\n")),
        "FAIL", "DEFER-C", "zzplant_w11"))

    # 17: a copy is a construction of its own and must assign its own cohort literal
    cp = "    ggml_sycl::alloc_request tp_req = req;\n    tp_req.queue = nullptr;\n"
    cp_hand = "    ggml_sycl::alloc_handle h{};\n    ggml_sycl::unified_alloc(tp_req, &h);\n"
    A(Case("17", "a copy of a good request with no cohort of its own reaches unified_alloc (the tp_req shape)", plant(
        "void zzplant_w17() {\n    %s req{};\n%s%s%s}\n" % (REQ, DEVF, cp, cp_hand)), "FAIL", "C-COHORT", "zzplant_w17"))
    A(Case("17", "a copy of a request with no zone or forbid, even with a cohort of its own", plant(
        "void zzplant_w17() {\n    %s req{};\n    req.intent.constraints.must_device = true;\n%s"
        "    tp_req.intent.cohort_id = \"zz-tp\";\n%s}\n" % (REQ, cp, cp_hand)), "FAIL", "D-ZONE", "zzplant_w17"))
    A(Case("17", "the same copy with its own cohort literal (control: today's tp_req once converted)", plant(
        "void zzplant_w17() {\n    %s req{};\n%s%s    tp_req.intent.cohort_id = \"zz-tp\";\n%s}\n" % (REQ, DEVF, cp, cp_hand)),
        "PASS"))
    A(Case("17", "a copy by assignment (`T tp{}; tp = req;`) with no cohort", plant(
        "void zzplant_w17() {\n    %s req{};\n%s    %s tp_req{};\n    tp_req = req;\n%s}\n" % (REQ, DEVF, REQ, cp_hand)),
        "FAIL", "C-COHORT", "zzplant_w17"))
    A(Case("17", "a copy by assignment with its own cohort literal (control)", plant(
        "void zzplant_w17() {\n    %s req{};\n%s    %s tp_req{};\n    tp_req = req;\n    tp_req.intent.cohort_id = \"zz-tp\";\n%s}\n"
        % (REQ, DEVF, REQ, cp_hand)), "PASS"))
    A(Case("17", "`auto tp_req = req` with no cohort", plant(
        "void zzplant_w17() {\n    %s req{};\n%s    auto tp_req = req;\n%s}\n" % (REQ, DEVF, cp_hand)),
        "FAIL", "C-COHORT", "zzplant_w17"))
    A(Case("17", "a copy through std::move with no cohort", plant(
        "void zzplant_w17() {\n    %s req{};\n%s    %s tp_req = std::move(req);\n%s}\n" % (REQ, DEVF, REQ, cp_hand)),
        "FAIL", "C-COHORT", "zzplant_w17"))
    A(Case("17", "a copy whose cohort is a variable, not a literal", plant(
        "void zzplant_w17(const char * name) {\n    %s req{};\n%s%s    tp_req.intent.cohort_id = name;\n%s}\n"
        % (REQ, DEVF, cp, cp_hand)), "FAIL", "C-COHORT", "zzplant_w17"))
    A(Case("17", "a copy whose cohort literal is written under a condition", plant(
        "void zzplant_w17(bool b) {\n    %s req{};\n%s%s    if (b) tp_req.intent.cohort_id = \"zz-tp\";\n%s}\n"
        % (REQ, DEVF, cp, cp_hand)), "FAIL", "C-COHORT", "zzplant_w17"))
    A(Case("17", "a copy of a copy needs its own cohort too", plant(
        "void zzplant_w17() {\n    %s req{};\n%s%s    tp_req.intent.cohort_id = \"zz-tp\";\n"
        "    %s tp2_req = tp_req;\n    ggml_sycl::alloc_handle h{};\n    ggml_sycl::unified_alloc(tp2_req, &h);\n}\n"
        % (REQ, DEVF, cp, REQ)), "FAIL", "C-COHORT", "zzplant_w17"))
    A(Case("17", "a copy that is never handed on reaches no allocator (control)", plant(
        "void zzplant_w17() {\n    %s req{};\n%s    %s tp_req = req;\n    (void) tp_req.size;\n}\n" % (REQ, DEVF, REQ)),
        "PASS"))
    A(Case("17", "a copy that writes COUNT to the zone", plant(
        "void zzplant_w17() {\n    %s req{};\n%s%s    tp_req.intent.cohort_id = \"zz-tp\";\n"
        "    tp_req.intent.constraints.prefer_vram_zone = " % (REQ, DEVF, cp) + ZN + "COUNT;\n" + cp_hand + "}\n"),
        "FAIL", "D-ZONE-COUNT", "zzplant_w17"))
    A(Case("17", "a copy that turns a device request into a host one and says so (control)", plant(
        "void zzplant_w17() {\n    %s req{};\n%s%s    tp_req.intent.cohort_id = \"zz-tp\";\n"
        "    tp_req.intent.constraints.must_device = false;\n    tp_req.intent.constraints.must_host_pinned = true;\n%s}\n"
        % (REQ, DEVF, cp, cp_hand)), "PASS"))
    A(Case("17", "a copy that adds the host tier without clearing the device tier", plant(
        "void zzplant_w17() {\n    %s req{};\n%s%s    tp_req.intent.cohort_id = \"zz-tp\";\n"
        "    tp_req.intent.constraints.must_host_pinned = true;\n%s}\n" % (REQ, DEVF, cp, cp_hand)),
        "FAIL", "B-TIER", "zzplant_w17"))
    # copies of a parameter, and an intent taken from a helper's return
    A(Case("17", "a copy of the caller's request (a wrapper) takes no cohort and no flags of its own (control)", plant(
        "void zzplant_w17(const %s & in) {\n    %s req = in;\n    req.size = 1;\n    ggml_sycl::unified_allocate(req);\n}\n"
        % (REQ, REQ)), "PASS"))
    A(Case("17", "a copy of the caller's request that writes COUNT", plant(
        "void zzplant_w17(const %s & in) {\n    %s req = in;\n    req.intent.constraints.prefer_vram_zone = " % (REQ, REQ)
        + ZN + "COUNT;\n    ggml_sycl::unified_allocate(req);\n}\n"), "FAIL", "D-ZONE-COUNT", "zzplant_w17"))
    A(Case("17", "a copy of the caller's request whose constraints are then wiped by a braced reset", plant(
        "void zzplant_w17(const %s & in) {\n    %s req = in;\n    req.intent.constraints = {};\n"
        "    ggml_sycl::unified_allocate(req);\n}\n" % (REQ, REQ)), "FAIL", "B-TIER", "zzplant_w17"))
    mk_good = ("static ggml_sycl::alloc_intent zzmk_w17(const char * c) {\n    ggml_sycl::alloc_intent intent{};\n"
               "    intent.cohort_id = c;\n    intent.constraints.must_device = true;\n"
               "    intent.constraints.prefer_vram_zone = " + ZN + "RUNTIME;\n"
               "    intent.constraints.forbid_vram_zone_spill = true;\n    return intent;\n}\n")
    A(Case("17", "an intent taken from a helper that returns a fully established one (control)", plant(
        mk_good + "void zzplant_w17() {\n    %s req{};\n    req.intent = zzmk_w17(\"zz\");\n    ggml_sycl::unified_allocate(req);\n}\n" % REQ),
        "PASS"))
    A(Case("17", "an intent taken from a helper that returns a COUNT zone", plant(
        mk_good.replace(ZN + "RUNTIME", ZN + "COUNT") + "void zzplant_w17() {\n    %s req{};\n    req.intent = zzmk_w17(\"zz\");\n"
        "    ggml_sycl::unified_allocate(req);\n}\n" % REQ), "FAIL", "D-ZONE-COUNT", "zzplant_w17"))
    A(Case("17", "an intent taken from a helper with two returns stays opaque", plant(
        mk_good.replace("    return intent;\n", "    if (c) return intent;\n    return intent;\n")
        + "void zzplant_w17() {\n    %s req{};\n    req.intent = zzmk_w17(\"zz\");\n    ggml_sycl::unified_allocate(req);\n}\n" % REQ),
        "FAIL", "DEFER-C", "zzplant_w17"))
    # 24: a wrapper that builds one request type from another copies the input's site
    wrap = ("ggml_sycl::mem_handle zzwrap_w24(const ggml_sycl::alloc_intent & intent) {\n    %s req{};\n    req.intent = intent;\n%s"
            "    return ggml_sycl::unified_allocate(req);\n}\n")
    site_both = "    req.site_file = intent.site_file;\n    req.site_line = intent.site_line;\n"
    A(Case("24", "a wrapper from alloc_intent to alloc_request with both site fields copied (control)", plant(
        wrap % (REQ, site_both)), "PASS"))
    A(Case("24", "the same wrapper copying neither site field", plant(wrap % (REQ, "")), "FAIL", "C-SITE", "zzwrap_w24"))
    A(Case("24", "the same wrapper copying only site_file", plant(wrap % (REQ, "    req.site_file = intent.site_file;\n")),
           "FAIL", "C-SITE", "zzwrap_w24"))
    A(Case("24", "the same wrapper copying only site_line", plant(wrap % (REQ, "    req.site_line = intent.site_line;\n")),
           "FAIL", "C-SITE", "zzwrap_w24"))
    A(Case("24", "a site copy under a condition does not count", plant(
        wrap % (REQ, "    if (intent.site_line) { req.site_file = intent.site_file; req.site_line = intent.site_line; }\n")),
        "FAIL", "C-SITE", "zzwrap_w24"))
    A(Case("24", "a site taken from the wrapper's own line, not its input's", plant(
        wrap % (REQ, "    req.site_file = __FILE__;\n    req.site_line = __LINE__;\n")), "FAIL", "C-SITE", "zzwrap_w24"))
    A(Case("24", "the nested intent's site copied instead of the request's own", plant(
        wrap % (REQ, "    req.intent.site_file = intent.site_file;\n    req.intent.site_line = intent.site_line;\n")),
        "FAIL", "C-SITE", "zzwrap_w24"))
    A(Case("24", "an offload_buffer_request wrapper that copies neither", plant(
        "void zzwrap_w24(const ggml_sycl::offload_buffer_request & in) {\n    %s areq{};\n    areq.size = in.size;\n"
        "    areq.intent = in.intent;\n    ggml_sycl::unified_allocate(areq);\n}\n" % REQ), "FAIL", "C-SITE", "zzwrap_w24"))
    A(Case("24", "an offload_buffer_request wrapper that copies both (control)", plant(
        "void zzwrap_w24(const ggml_sycl::offload_buffer_request & in) {\n    %s areq{};\n    areq.size = in.size;\n"
        "    areq.intent = in.intent;\n    areq.site_file = in.site_file;\n    areq.site_line = in.site_line;\n    ggml_sycl::unified_allocate(areq);\n}\n" % REQ),
        "PASS"))
    A(Case("24", "a function taking a request and building one of the same type owes no site copy (control)", plant(
        "void zzwrap_w24(const %s & in) {\n    %s req = in;\n    req.size = in.size;\n    ggml_sycl::unified_allocate(req);\n}\n"
        % (REQ, REQ)), "PASS"))
    # ---- S2b: the gaps S2a listed. Aliases, members, parameters and the first handoff.
    ROUTE = "ggml_sycl::vram_zone_id::RUNTIME"
    A(Case("s2b-alias", "a constraints reference written COUNT through the alias", plant(good_device(
        "zzplant_alias", "    ggml_sycl::alloc_constraints & c = req.intent.constraints;\n    c.prefer_vram_zone = " + ZN + "COUNT;\n")),
        "FAIL", "D-ZONE-COUNT", "zzplant_alias"))
    A(Case("s2b-alias", "a pointer to the constraints written forbid = false through `p->`", plant(good_device(
        "zzplant_alias", "    auto * p = &req.intent.constraints;\n    p->forbid_vram_zone_spill = false;\n")),
        "FAIL", "D-FORBID-FALSE", "zzplant_alias"))
    A(Case("s2b-alias", "a pointer to the whole request written COUNT through `p->intent...`", plant(good_device(
        "zzplant_alias", "    %s * p = &req;\n    p->intent.constraints.prefer_vram_zone = %sCOUNT;\n" % (REQ, ZN))),
        "FAIL", "D-ZONE-COUNT", "zzplant_alias"))
    A(Case("s2b-alias", "an alias of an alias written COUNT", plant(good_device(
        "zzplant_alias", "    auto & in = req.intent;\n    auto & c = in.constraints;\n    c.prefer_vram_zone = " + ZN + "COUNT;\n")),
        "FAIL", "D-ZONE-COUNT", "zzplant_alias"))
    A(Case("s2b-alias", "a dereferenced alias `(*p).intent...` written COUNT", plant(good_device(
        "zzplant_alias", "    %s * p = &req;\n    (*p).intent.constraints.prefer_vram_zone = %sCOUNT;\n" % (REQ, ZN))),
        "FAIL", "D-ZONE-COUNT", "zzplant_alias"))
    A(Case("s2b-alias", "every flag established through a constraints alias (control)", plant(
        "void zzplant_alias() {\n    %s req{};\n    ggml_sycl::alloc_constraints & c = req.intent.constraints;\n"
        "    c.must_device = true;\n    c.prefer_vram_zone = %s;\n    c.forbid_vram_zone_spill = true;\n"
        "    ggml_sycl::unified_allocate(req);\n}\n" % (REQ, ROUTE)), "PASS"))
    A(Case("s2b-alias", "the alias's own initialiser does not hand the request over (control)", plant(
        "void zzplant_alias() {\n    %s req{};\n    ggml_sycl::alloc_constraints * p = &req.intent.constraints;\n"
        "    p->must_device = true;\n    p->prefer_vram_zone = %s;\n    p->forbid_vram_zone_spill = true;\n"
        "    ggml_sycl::unified_allocate(req);\n}\n" % (REQ, ROUTE)), "PASS"))
    A(Case("s2b-alias", "an alias of a holder's member written COUNT (no request of its own to blame)", plant(
        "struct zzholder { %s r{}; };\nvoid zzplant_alias(zzholder & h) {\n    ggml_sycl::alloc_constraints & c = h.r.intent.constraints;\n"
        "    c.prefer_vram_zone = %sCOUNT;\n}\n" % (REQ, ZN)), "FAIL", "D-ZONE-COUNT", "zzplant_alias"))
    A(Case("s2b-alias", "an alias of a holder's member written a real zone (control)", plant(
        "struct zzholder { %s r{}; };\nvoid zzplant_alias(zzholder & h) {\n    ggml_sycl::alloc_constraints & c = h.r.intent.constraints;\n"
        "    c.prefer_vram_zone = %s;\n}\n" % (REQ, ROUTE)), "PASS", planted=False))
    A(Case("s2b-member", "a method writing COUNT to a request member", plant(
        "struct zzmember { %s r{};\n    void zzf() { r.intent.constraints.prefer_vram_zone = %sCOUNT; }\n};\n" % (REQ, ZN)),
        "FAIL", "D-ZONE-COUNT", "zzmember::zzf"))
    A(Case("s2b-member", "a method writing forbid = false through `this->`", plant(
        "struct zzmember { %s r{};\n    void zzf() { this->r.intent.constraints.forbid_vram_zone_spill = false; }\n};\n" % REQ),
        "FAIL", "D-FORBID-FALSE", "zzmember::zzf"))
    A(Case("s2b-member", "a free function writing COUNT through a pointer to a holder", plant(
        "struct zzholder { %s r{}; };\nvoid zzplant_member(zzholder * h) {\n    h->r.intent.constraints.prefer_vram_zone = %sCOUNT;\n}\n"
        % (REQ, ZN)), "FAIL", "D-ZONE-COUNT", "zzplant_member"))
    A(Case("s2b-member", "a file-scope request written forbid = false", plant(
        "static %s zzglobal{};\nvoid zzplant_member() {\n    zzglobal.intent.constraints.forbid_vram_zone_spill = false;\n}\n" % REQ),
        "FAIL", "D-FORBID-FALSE", "zzplant_member"))
    A(Case("s2b-member", "a method writing a real zone and forbid = true to a member (control)", plant(
        "struct zzmember { %s r{};\n    void zzf() { r.intent.constraints.prefer_vram_zone = %s;\n"
        "        this->r.intent.constraints.forbid_vram_zone_spill = true; }\n};\n" % (REQ, ROUTE)), "PASS", planted=False))
    A(Case("s2b-member", "a by-value request parameter written COUNT", plant(
        "void zzplant_member(%s req) {\n    req.intent.constraints.prefer_vram_zone = %sCOUNT;\n    ggml_sycl::unified_allocate(req);\n}\n"
        % (REQ, ZN)), "FAIL", "D-ZONE-COUNT", "zzplant_member"))
    A(Case("s2b-member", "a by-value request parameter whose only writes are not flags (control)", plant(
        "void zzplant_member(%s req) {\n    req.size = 4;\n    ggml_sycl::unified_allocate(req);\n}\n" % REQ), "PASS", planted=False))
    A(Case("s2b-member", "a by-value request parameter with a cohort and a real zone written (control)", plant(
        "void zzplant_member(%s req) {\n    req.intent.cohort_id = \"zz\";\n    req.intent.constraints.prefer_vram_zone = %s;\n"
        "    ggml_sycl::unified_allocate(req);\n}\n" % (REQ, ROUTE)), "PASS"))
    # the first handoff: the address of a scalar field, and a method call on the request
    pre = ("    req.intent.constraints.prefer_vram_zone = " + ROUTE + ";\n"
           "    req.intent.constraints.forbid_vram_zone_spill = true;\n")
    A(Case("s2b-firstuse", "the tier flag written after its address was handed to a function", plant(
        "void zzplant_firstuse() {\n    %s req{};\n%s    zz_f(&req.intent.constraints.must_device);\n"
        "    req.intent.constraints.must_device = true;\n    ggml_sycl::unified_allocate(req);\n}\n" % (REQ, pre)),
        "FAIL", "B-TIER", "zzplant_firstuse"))
    A(Case("s2b-firstuse", "the address handed over after every write (control)", plant(
        "void zzplant_firstuse() {\n    %s req{};\n%s    req.intent.constraints.must_device = true;\n"
        "    zz_f(&req.intent.constraints.must_device);\n    ggml_sycl::unified_allocate(req);\n}\n" % (REQ, pre)), "PASS"))
    A(Case("s2b-firstuse", "the tier flag written after a method was called on the request", plant(
        "void zzplant_firstuse() {\n    %s req{};\n%s    req.zz_m();\n"
        "    req.intent.constraints.must_device = true;\n    ggml_sycl::unified_allocate(req);\n}\n" % (REQ, pre)),
        "FAIL", "B-TIER", "zzplant_firstuse"))
    A(Case("s2b-firstuse", "a method called on the sub-object after every write (control)", plant(
        "void zzplant_firstuse() {\n    %s req{};\n%s    req.intent.constraints.must_device = true;\n"
        "    req.intent.constraints.zz_m();\n    ggml_sycl::unified_allocate(req);\n}\n" % (REQ, pre)), "PASS"))
    A(Case("s2b-firstuse", "a scalar field read before the tier write is not a handoff (control)", plant(
        "void zzplant_firstuse() {\n    %s req{};\n%s    (void) req.intent.constraints.must_device;\n"
        "    req.intent.constraints.must_device = true;\n    ggml_sycl::unified_allocate(req);\n}\n" % (REQ, pre)), "PASS"))
    A(Case("s2b-firstuse", "documented gap: a reference bound to a scalar field is not followed (the gate passes it)", plant(
        "void zzplant_firstuse() {\n    %s req{};\n%s    req.intent.constraints.must_device = true;\n"
        "    bool & zz_ref = req.intent.constraints.must_device;\n    zz_ref = false;\n    ggml_sycl::unified_allocate(req);\n}\n"
        % (REQ, pre)), "PASS"))
    # S2b: dpct is sanctioned-vendored (canonical contract section 9.1, then rulings M247). Its three sites are allowlisted by function name and count;
    # the entry points are forbidden names outside dpct/helper.hpp, matched as identifiers, never as substrings.
    for nm in ("dpct_malloc", "dpct::dpct_malloc", "dpct::detail::dpct_malloc"):
        A(Case("s2b-dpct", "a new caller of %s outside helper.hpp" % nm, plant(
            "void zzplant_dpct(sycl::queue & q) {\n    void * p = %s(16, q);\n    (void) p;\n}\n" % nm),
            "FAIL", "E-RAW", "dpct_malloc"))
    for cls in ("device_memory", "global_memory", "constant_memory", "shared_memory"):
        A(Case("s2b-dpct", "a new %s object outside helper.hpp" % cls, plant(
            "void zzplant_dpct(sycl::queue & q) {\n    dpct::%s<int, 1> m(16);\n    (void) m;\n}\n" % cls),
            "FAIL", "E-RAW", cls))
    A(Case("s2b-dpct", "a new caller under dpct/ but not in helper.hpp", plant(
        "inline void zzplant_dpct(sycl::queue & q) { void * p = dpct::dpct_malloc(16, q); (void) p; }\n",
        name="dpct/zz-plant.hpp"), "FAIL", "E-RAW", "dpct_malloc"))
    A(Case("s2b-dpct", "names that merely contain a dpct name are not hits (control)", plant(
        "void zzplant_dpct_ok(sycl::queue & q) {\n    int my_dpct_malloc_count = 0;\n    int device_memory = 1;\n"
        "    int shared_memory = 2;\n    int global_memory_total = 3;\n    struct zz_constant_memory_pool {} z;\n"
        "    (void) my_dpct_malloc_count; (void) device_memory; (void) shared_memory; (void) global_memory_total; (void) z;\n}\n"),
        "PASS", planted=False))
    A(Case("s2b-dpct", "a comment and a string naming dpct_malloc are not hits (control)", plant(
        "// dpct_malloc(16, q) and dpct::device_memory<int, 1>\n"
        "void zzplant_dpct_ok() {\n    const char * s = \"dpct_malloc dpct::shared_memory\";\n    (void) s;\n}\n"),
        "PASS", planted=False))
    A(Case("s2b-dpct", "a second raw call inside dpct_malloc breaks its count pin", replace_in_function(
        "dpct/helper.hpp", r"static inline void \*dpct_malloc\(size_t size", "return sycl::malloc_device(size, q.get_device(), q.get_context());",
        "sycl::malloc_device(size, q.get_device(), q.get_context());\n            return sycl::malloc_device(size, q.get_device(), q.get_context());"),
        "FAIL", "allowlist", "E-DPCT-MALLOC covers 2 finding(s) but pins 1"))
    A(Case("s2b-dpct", "dpct_malloc's function renamed in helper.hpp leaves its entry matching nothing", replace_in_function(
        "dpct/helper.hpp", r"static inline void \*(?=dpct_malloc\(size_t size)", "dpct_malloc(size_t size", "dpct_malloc_zz(size_t size"),
        "FAIL", "allowlist", "E-DPCT-MALLOC"))
    A(Case("s2b-dpct", "a third raw call added to device_memory::allocate_device breaks its pin", replace_in_function(
        "dpct/helper.hpp", r"void allocate_device\(sycl::queue &q\)", "_device_ptr = (value_t *)detail::dpct_malloc(_size, q);",
        "_device_ptr = (value_t *)sycl::malloc_device(_size, q.get_device(), q.get_context());\n"
        "                _device_ptr = (value_t *)detail::dpct_malloc(_size, q);"),
        "FAIL", "allowlist", "E-DPCT-DEVMEM-DEVICE covers 2 finding(s) but pins 1"))
    A(Case("s2b-dpct", "a different raw name swapped into dpct_malloc keeps the count and FAILs", replace_in_function(
        "dpct/helper.hpp", r"static inline void \*dpct_malloc\(size_t size", "sycl::malloc_device(size", "sycl::malloc_shared(size"),
        "FAIL", "E-RAW", "malloc_shared"))
    # S2b: the raw chain's three links are allowlisted by name and count (canonical contract sections 3 and 9.1)
    A(Case("s2b-chain", "a second aligned_alloc_device call in sycl_aligned_malloc_device breaks its pin", replace_in_function(
        "unified-cache.cpp", r"static void \* sycl_aligned_malloc_device\(size_t size, const sycl::queue & queue\) \{",
        "aligned_alloc_device(", "aligned_alloc_device(0, 0, queue); aligned_alloc_device("),
        "FAIL", "allowlist", "E-CHAIN-ALIGNED covers 2 finding(s) but pins 1"))
    A(Case("s2b-chain", "a second link call in unified_cache_raw_malloc_device breaks its pin", replace_in_function(
        "unified-cache.cpp", r"void \* unified_cache_raw_malloc_device\(size_t size, const sycl::queue & queue\) \{",
        "ptr = sycl_aligned_malloc_device(size, queue);",
        "ptr = sycl_aligned_malloc_device(size, queue);\n        ptr = sycl_aligned_malloc_device(size, queue);"),
        "FAIL", "allowlist", "E-CHAIN-RAW covers 2 finding(s) but pins 1"))
    A(Case("s2b-chain", "a new caller of the chain's raw wrapper in another function is a finding", plant(
        "void zzplant_chain(const sycl::queue & q) {\n    void * p = unified_cache_raw_malloc_device(16, q);\n    (void) p;\n}\n"),
        "FAIL", "E-RAW", "unified_cache_raw_malloc_device"))
    # S2c, clause (g): every handler that can swallow ggml_sycl_fallback_error is preceded, in its try, by a handler that
    # rethrows it. Witnesses 15 and 34 (the macro form), with each spelling the rule is spelling-independent over.
    G = "catch (const ggml_sycl_fallback_error &) { throw; } "

    def tryfn(name, handlers, body="zz_op();"):
        return "void %s() {\n    try { %s } %s\n}\n" % (name, body, handlers)

    A(Case("15", "a catch (...) added in a dispatch function without the rethrow", plant(
        tryfn("zzplant_w15", "catch (...) { }")), "FAIL", "G-CATCH", "zzplant_w15"))
    A(Case("15", "the same handler after the rethrow clause (control)", plant(
        tryfn("zzplant_w15", G + "catch (...) { }")), "PASS"))
    A(Case("15", "a catch (const std::exception &) without the rethrow", plant(
        tryfn("zzplant_w15", "catch (const std::exception & e) { (void) e; }")), "FAIL", "G-CATCH", "zzplant_w15"))
    A(Case("15", "the rethrow clause with a named parameter keeps a std::exception handler legal (control)", plant(
        tryfn("zzplant_w15", "catch (const ggml_sycl_fallback_error & fe) { throw; } catch (const std::exception & e) { (void) e; }")),
        "PASS"))
    A(Case("s2c-catch", "east const: catch (std::exception const &) without the rethrow", plant(
        tryfn("zzplant_catch", "catch (std::exception const & e) { (void) e; }")), "FAIL", "G-CATCH", "zzplant_catch"))
    A(Case("s2c-catch", "a global-scope qualified handler: catch (const ::std::exception &) without the rethrow", plant(
        tryfn("zzplant_catch", "catch (const ::std::exception & e) { (void) e; }")), "FAIL", "G-CATCH", "zzplant_catch"))
    A(Case("s2c-catch", "a global-scope qualified by-value handler without the rethrow", plant(
        tryfn("zzplant_catch", "catch (::std::exception e) { }")), "FAIL", "G-CATCH", "zzplant_catch"))
    A(Case("s2c-catch", "a global-scope qualified rethrow clause keeps the handler legal (control)", plant(
        tryfn("zzplant_catch", "catch (const ::ggml_sycl_fallback_error &) { throw; } catch (const ::std::exception & e) { (void) e; }")),
        "PASS"))
    A(Case("s2c-catch", "east const with the rethrow kept (control)", plant(
        tryfn("zzplant_catch", G + "catch (std::exception const & e) { (void) e; }")), "PASS"))
    A(Case("s2c-catch", "a by-value handler slices the fallback error too", plant(
        tryfn("zzplant_catch", "catch (std::exception e) { }")), "FAIL", "G-CATCH", "zzplant_catch"))
    A(Case("s2c-catch", "a rethrow clause whose body does more than throw does not count", plant(
        tryfn("zzplant_catch", "catch (const ggml_sycl_fallback_error &) { zz_log(); throw; } catch (...) { }")),
        "FAIL", "G-CATCH", "zzplant_catch"))
    A(Case("s2c-catch", "a rethrow clause that throws a copy does not count", plant(
        tryfn("zzplant_catch", "catch (const ggml_sycl_fallback_error & e) { throw e; } catch (...) { }")),
        "FAIL", "G-CATCH", "zzplant_catch"))
    A(Case("s2c-catch", "a rethrow clause placed after the handler does not count", plant(
        tryfn("zzplant_catch", "catch (...) { } catch (const ggml_sycl_fallback_error &) { throw; }")),
        "FAIL", "G-CATCH", "zzplant_catch"))
    A(Case("s2c-catch", "a rethrow clause in another try does not count", plant(
        "void zzplant_catch() {\n    try { zz_a(); } " + G + "\n    try { zz_b(); } catch (...) { }\n}\n"),
        "FAIL", "G-CATCH", "zzplant_catch"))
    A(Case("s2c-catch", "a comment inside the rethrow clause is still exactly `throw;` (control)", plant(
        tryfn("zzplant_catch", "catch (const ggml_sycl_fallback_error &) { /* keep */ throw; } catch (...) { }")), "PASS"))
    A(Case("s2c-catch", "handlers for other types are outside the rule (control)", plant(
        tryfn("zzplant_catch", "catch (const std::runtime_error & e) { (void) e; } catch (const sycl::exception & e) { (void) e; } "
              "catch (std::exception_ptr p) { (void) p; }")), "PASS", planted=False))
    A(Case("s2c-catch", "a catch spelled in a comment or a string is not a handler (control)", plant(
        "// try { x(); } catch (...) { }\nvoid zzplant_catch() {\n    const char * s = \"catch (...) { }\";\n    (void) s;\n}\n"),
        "PASS", planted=False))
    A(Case("s2c-catch", "a catch in a lambda is a handler of the function that holds it", plant(
        "void zzplant_catch() {\n    auto f = [&]() { try { zz_op(); } catch (...) { } };\n    f();\n}\n"), "FAIL", "G-CATCH", "zzplant_catch"))
    A(Case("s2c-catch", "two unguarded handlers in an exempted function break the exemption's count", plant(
        "void zzplant_catch() {\n    try { zz_a(); } catch (...) { }\n    try { zz_b(); } catch (...) { }\n}\n"), "FAIL", "allowlist",
        "E-ZZ-CATCH covers 2 finding(s) but pins 1",
        allowlist={"id": "E-ZZ-CATCH", "code": "G-CATCH", "file": PLANT, "function": "zzplant_catch", "count": 1,
                   "reason": "mutation-matrix test entry"}))
    A(Case("s2c-catch", "the same two handlers under an exemption that pins both (control)", plant(
        "void zzplant_catch() {\n    try { zz_a(); } catch (...) { }\n    try { zz_b(); } catch (...) { }\n}\n"), "PASS",
        allowlist={"id": "E-ZZ-CATCH", "code": "G-CATCH", "file": PLANT, "function": "zzplant_catch", "count": 2,
                   "reason": "mutation-matrix test entry"}))
    A(Case("s2c-catch", "a G-CATCH debt entry whose handler is gone is stale", lambda f: f, "FAIL", "debt",
           "no longer violates", edit_debt=lambda d: dict(d, violations=d["violations"] + [{
               "code": "G-CATCH", "key": "zz-plant.cpp::zzplant_gone::catch_clause:all#0"}]), planted=False))
    mac = "#define zz_try(x) do { try { x; } %s %s } while (0)\n"
    A(Case("34", "a macro handler with no rethrow clause", plant(mac % ("", "catch (const std::exception & e) { (void) e; }")),
        "FAIL", "G-CATCH", "#define zz_try", planted=False))
    A(Case("34", "the same macro with its rethrow clause (control)", plant(mac % (G, "catch (const std::exception & e) { (void) e; }")),
        "PASS"))
    A(Case("34", "the macro respelled std::exception const & keeps the rethrow (control)", plant(
        mac % (G, "catch (std::exception const & e) { (void) e; }")), "PASS"))
    A(Case("34", "the respelled macro with the rethrow removed", plant(mac % ("", "catch (std::exception const & e) { (void) e; }")),
        "FAIL", "G-CATCH", "#define zz_try", planted=False))
    A(Case("34", "a by-value handler in a macro with the rethrow removed", plant(mac % ("", "catch (std::exception e) { }")),
        "FAIL", "G-CATCH", "#define zz_try", planted=False))
    A(Case("34", "a global-scope qualified macro handler with the rethrow removed", plant(
        mac % ("", "catch (const ::std::exception & e) { (void) e; }")), "FAIL", "G-CATCH", "#define zz_try", planted=False))
    A(Case("34", "a catch (...) in a macro with no rethrow", plant(mac % ("", "catch (...) { }")), "FAIL", "G-CATCH",
        "#define zz_try", planted=False))
    A(Case("34", "a macro whose rethrow clause guards only an earlier try", plant(
        "#define zz_try(x) do { try { x; } " + G + "try { x; } catch (...) { } } while (0)\n"), "FAIL", "G-CATCH", "#define zz_try",
        planted=False))
    A(Case("34", "the real CHECK_TRY_ERROR handler is a finding, keyed (file, macro)", lambda f: f, "FAIL", "G-CATCH",
        "common.hpp::#define CHECK_TRY_ERROR::catch_macro:exception#0", planted=False,
        edit_debt=lambda d: dict(d, violations=[e for e in d["violations"] if "CHECK_TRY_ERROR" not in e["key"]])))
    A(Case("34", "guarding the real CHECK_TRY_ERROR leaves its debt entry stale", lambda f: dict(f, **{"common.hpp": f["common.hpp"].replace(
        b"        } catch (std::exception const & e) {", b"        } catch (const ggml_sycl_fallback_error &) {  \\\n            throw;  \\\n        } catch (std::exception const & e) {", 1)}),
        "FAIL", "debt", "CHECK_TRY_ERROR", planted=False))

    # S2c, clause (h): cascade_step and unconverted_ticket, keyed by node. Nothing writes either today, so every write is new.
    CS = "    req.intent.constraints.cascade_step = true;\n"
    A(Case("16", "cascade_step = true written at a construction outside the allowlist", plant(
        good_device("zzplant_w16", CS)), "FAIL", "H-CASCADE", "zzplant_w16"))
    A(Case("16", "the same write at an allowlisted node (control)", plant(good_device("zzplant_w16", CS)), "PASS",
        allow_nodes=[{"code": "H-CASCADE", "func": "zzplant_w16"}]))
    A(Case("16", "cascade_step in a designated initialiser", plant(
        "void zzplant_w16() {\n    %s req{ .intent = { .constraints = { .must_device = true, .cascade_step = true } } };\n"
        "    (void) req;\n}\n" % REQ), "FAIL", "H-CASCADE", "zzplant_w16", planted=False))
    A(Case("16", "cascade_step written through a reference alias of the constraints", plant(good_device(
        "zzplant_w16", "    ggml_sycl::alloc_constraints & c = req.intent.constraints;\n    c.cascade_step = true;\n")),
        "FAIL", "H-CASCADE", "zzplant_w16"))
    A(Case("16", "cascade_step written through a pointer", plant(good_device(
        "zzplant_w16", "    ggml_sycl::alloc_constraints * p = &req.intent.constraints;\n    p->cascade_step = true;\n")),
        "FAIL", "H-CASCADE", "zzplant_w16"))
    A(Case("16", "cascade_step written inside a helper that takes the request by reference", plant(
        "void zzplant_w16(%s & r) {\n    r.intent.constraints.cascade_step = true;\n}\n" % REQ), "FAIL", "H-CASCADE", "zzplant_w16",
        planted=False))
    A(Case("16", "a reset to false is not a writer (control)", plant(good_device(
        "zzplant_w16", "    req.intent.constraints.cascade_step = false;\n")), "PASS"))
    A(Case("16", "a compound assignment is an expression write", plant(good_device(
        "zzplant_w16", "    req.intent.constraints.cascade_step |= true;\n")), "FAIL", "H-CASCADE-EXPR", "zzplant_w16"))
    A(Case("16", "a second construction planted beside a listed one is a new node", plant(
        "void zzplant_w16() {\n    %s req{};\n    req.intent.constraints.must_device = true;\n"
        "    req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME;\n"
        "    req.intent.constraints.forbid_vram_zone_spill = true;\n%s"
        "    %s req2{};\n    req2.intent.constraints.must_device = true;\n"
        "    req2.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME;\n"
        "    req2.intent.constraints.forbid_vram_zone_spill = true;\n    req2.intent.constraints.cascade_step = true;\n}\n"
        % (REQ, CS, REQ)), "FAIL", "H-CASCADE", "zzplant_w16", allow_nodes=[{"code": "H-CASCADE", "func": "zzplant_w16", "nth": 0}]))
    # 21: a DECLARED row that loses its flag leaves its allowlist entry matching nothing
    A(Case("21", "a DECLARED construction with its cascade_step removed", plant(good_device("zzplant_w21")), "FAIL", "allowlist",
        "matches nothing", allow_nodes=[{"code": "H-CASCADE", "func": "zzplant_w21"}],
        allow_from=plant(good_device("zzplant_w21", CS))))
    A(Case("21", "the same construction still writing it (control)", plant(good_device("zzplant_w21", CS)), "PASS",
        allow_nodes=[{"code": "H-CASCADE", "func": "zzplant_w21"}]))
    # 16b, 25: a call that passes a cascade_step parameter
    CF = "void zz_reserve(int a, bool cascade_step = false);\n"
    A(Case("16b", "a dispatch caller passing cascade_step = true", plant(CF + "void zzplant_w16b() {\n    zz_reserve(1, true);\n}\n"),
        "FAIL", "H-PASS-TRUE", "zzplant_w16b", planted=False))
    A(Case("16b", "the transaction's call passing true at an allowlisted node (control)", plant(
        CF + "void zzplant_w16b() {\n    zz_reserve(1, true);\n}\n"), "PASS",
        allow_nodes=[{"code": "H-PASS-TRUE", "func": "zzplant_w16b"}]))
    A(Case("16b", "a caller that passes the default, spelled or omitted (control)", plant(
        CF + "void zzplant_w16b() {\n    zz_reserve(1);\n    zz_reserve(2, false);\n}\n"), "PASS"))
    A(Case("16b", "a method call passing true is the same call", plant(
        "struct zz_ring { void reserve(int a, bool cascade_step = false); };\n"
        "void zzplant_w16b(zz_ring & r) {\n    r.reserve(1, true);\n}\n"), "FAIL", "H-PASS-TRUE", "zzplant_w16b", planted=False))
    A(Case("25", "the replan's restore forwarding its cascade_step parameter", plant(
        CF + "void zzplant_w25(bool cascade_step) {\n    zz_reserve(1, cascade_step);\n    zz_reserve(2, cascade_step);\n}\n"),
        "FAIL", "H-PASS-FORWARD", "zzplant_w25", planted=False,
        allow_nodes=[{"code": "H-PASS-FORWARD", "func": "zzplant_w25", "nth": 0}]))
    A(Case("25", "the new-size reserve's forward alone at an allowlisted node (control)", plant(
        CF + "void zzplant_w25(bool cascade_step) {\n    zz_reserve(1, cascade_step);\n    zz_reserve(2);\n}\n"), "PASS",
        allow_nodes=[{"code": "H-PASS-FORWARD", "func": "zzplant_w25"}]))
    A(Case("25", "a forward of a local that shadows the parameter name is an expression", plant(
        CF + "void zzplant_w25(bool b) {\n    bool cascade_step = b;\n    zz_reserve(1, cascade_step);\n}\n"),
        "FAIL", "H-PASS-EXPR", "zzplant_w25", planted=False))
    A(Case("25", "a call passing some other expression", plant(
        CF + "void zzplant_w25(bool b) {\n    zz_reserve(1, b);\n}\n"), "FAIL", "H-PASS-EXPR", "zzplant_w25", planted=False))
    # 26: cascade_step = <expr> only as the enclosing allowlisted callee's own parameter
    A(Case("26", "req.cascade_step = flag where flag is a local", plant(good_device(
        "zzplant_w26", "    bool flag = zz_flag();\n    req.intent.constraints.cascade_step = flag;\n")), "FAIL", "H-CASCADE-EXPR",
        "zzplant_w26"))
    A(Case("26", "req.cascade_step = flag where flag is a member", plant(
        "struct zz_holder { bool flag = false;\n    void zzplant_w26() {\n        %s req{};\n"
        "        req.intent.constraints.cascade_step = flag;\n    }\n};\n" % REQ), "FAIL", "H-CASCADE-EXPR", "zzplant_w26",
        planted=False))
    A(Case("26", "the same statement copying the function's own parameter, with no callee entry", plant(
        "void zzplant_w26(bool cascade_step) {\n    %s req{};\n    req.intent.constraints.cascade_step = cascade_step;\n}\n" % REQ),
        "FAIL", "H-CASCADE-PARAM", "zzplant_w26", planted=False))
    A(Case("26", "the same statement inside an allowlisted callee (control, try_alloc's shape)", plant(
        "void zzplant_w26(bool cascade_step) {\n    %s req{};\n    req.intent.constraints.must_device = true;\n"
        "    req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME;\n"
        "    req.intent.constraints.forbid_vram_zone_spill = true;\n"
        "    req.intent.constraints.cascade_step = cascade_step;\n}\n" % REQ),
        "PASS",
        allowlist={"id": "E-ZZ-CALLEE", "code": "H-CASCADE-PARAM", "file": PLANT, "function": "zzplant_w26", "count": 1,
                   "outcome": "CASCADE", "reason": "mutation-matrix test entry"}))
    A(Case("26", "a callee entry pins its count: a second copy fails", plant(
        "void zzplant_w26(bool cascade_step) {\n    %s a{};\n    a.intent.constraints.cascade_step = cascade_step;\n"
        "    %s b{};\n    b.intent.constraints.cascade_step = cascade_step;\n}\n" % (REQ, REQ)),
        "FAIL", "allowlist", "E-ZZ-CALLEE covers 2 finding(s) but pins 1", planted=False,
        allowlist={"id": "E-ZZ-CALLEE", "code": "H-CASCADE-PARAM", "file": PLANT, "function": "zzplant_w26", "count": 1,
                   "outcome": "CASCADE", "reason": "mutation-matrix test entry"}))
    A(Case("26", "a shadowing local named cascade_step is not the parameter", plant(
        "void zzplant_w26(bool b) {\n    bool cascade_step = b;\n    %s req{};\n    req.intent.constraints.cascade_step = cascade_step;\n}\n"
        % REQ), "FAIL", "H-CASCADE-EXPR", "zzplant_w26", planted=False))
    # 22: unconverted_ticket is shrink-only and keyed by the construction node
    UT = "    req.intent.constraints.unconverted_ticket = \"llama.cpp-zz1\";\n"
    one = good_device("zzplant_w22", UT)
    two = ("void zzplant_w22() {\n    %s req{};\n    req.intent.constraints.must_device = true;\n"
           "    req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME;\n"
           "    req.intent.constraints.forbid_vram_zone_spill = true;\n%s"
           "    %s req2{};\n    req2.intent.constraints.must_device = true;\n"
           "    req2.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME;\n"
           "    req2.intent.constraints.forbid_vram_zone_spill = true;\n"
           "    req2.intent.constraints.unconverted_ticket = \"llama.cpp-zz1\";\n}\n" % (REQ, UT, REQ))
    A(Case("22", "a new construction setting the ticket inside a function that already holds a listed one", plant(two), "FAIL",
        "H-UNCONV", "zzplant_w22", allow_nodes=[{"code": "H-UNCONV", "func": "zzplant_w22", "nth": 0}], allow_from=plant(one)))
    A(Case("22", "the listed construction itself (control)", plant(one), "PASS",
        allow_nodes=[{"code": "H-UNCONV", "func": "zzplant_w22"}]))
    A(Case("22", "the listed construction naming a different ticket", plant(one.replace("zz1", "zz2")), "FAIL", "H-UNCONV",
        "zzplant_w22", allow_nodes=[{"code": "H-UNCONV", "func": "zzplant_w22"}], allow_from=plant(one)))
    A(Case("22", "unconverted_ticket on a construction nobody listed", plant(one), "FAIL", "H-UNCONV", "zzplant_w22"))
    A(Case("22", "unconverted_ticket set from a variable", plant(good_device(
        "zzplant_w22", "    const char * t = zz_ticket();\n    req.intent.constraints.unconverted_ticket = t;\n")),
        "FAIL", "H-UNCONV-EXPR", "zzplant_w22"))
    A(Case("22", "unconverted_ticket reset to nullptr is not a setter (control)", plant(good_device(
        "zzplant_w22", "    req.intent.constraints.unconverted_ticket = nullptr;\n")), "PASS"))
    A(Case("22", "unconverted_ticket in a designated initialiser", plant(
        "void zzplant_w22() {\n    %s req{ .intent = { .constraints = { .must_device = true, .unconverted_ticket = \"llama.cpp-zz1\" } } };\n"
        "    (void) req;\n}\n" % REQ), "FAIL", "H-UNCONV", "zzplant_w22", planted=False))
    A(Case("22", "an entry declaring a TERMINAL outcome for a writer is rejected", plant(one), "FAIL", "data",
        "needs outcome UNCONVERTED", allow_nodes=[{"code": "H-UNCONV", "func": "zzplant_w22",
                                                    "extra": {"outcome": "TERMINAL"}}], planted=False))
    A(Case("s2c-data", "a function-keyed entry for a cascade_step write is rejected", plant(good_device("zzplant_data", CS)), "FAIL",
        "data", "must be keyed by the construction or call node", planted=False,
        allowlist={"id": "E-ZZ-FN", "code": "H-CASCADE", "file": PLANT, "function": "zzplant_data", "count": 1, "outcome": "CASCADE",
                   "reason": "mutation-matrix test entry"}))
    A(Case("s2c-data", "an entry exempting an expression write is rejected", plant(good_device(
        "zzplant_data", "    req.intent.constraints.cascade_step = zz_flag();\n")), "FAIL", "data", "cannot be exempted", planted=False,
        allowlist={"id": "E-ZZ-EXPR", "code": "H-CASCADE-EXPR", "file": PLANT, "function": "zzplant_data", "count": 1,
                   "reason": "mutation-matrix test entry"}))
    A(Case("s2c-data", "a clause-(h) finding listed as debt is rejected", plant(good_device("zzplant_data", CS)), "FAIL", "data",
        "never debt", planted=False, edit_debt=lambda d: dict(d, violations=d["violations"] + [{
            "code": "H-CASCADE", "key": "zz-plant.cpp::zzplant_data::declaration:req:00000000::write:cascade_step#0"}])))
    A(Case("s2c-data", "an H-UNCONV entry without its ticket is rejected", plant(one), "FAIL", "data", "needs the ticket",
        planted=False, allow_nodes=[{"code": "H-UNCONV", "func": "zzplant_w22", "extra": {"ticket": ""}}]))
    # S2c: the host side of clause (e). malloc_host, aligned_alloc_host, zeMemAllocHost, the generic sycl::malloc and
    # sycl::aligned_alloc (qualified only: the bare names are the C library's) and the host raw chain's own wrappers.
    for nm, call in (("malloc_host", "sycl::malloc_host(16, q)"), ("malloc_host", "sycl::malloc_host<int>(4, q)"),
                     ("aligned_alloc_host", "sycl::aligned_alloc_host(64, 16, q)"),
                     ("malloc", "sycl::malloc(16, q, sycl::usm::alloc::host)"),
                     ("aligned_alloc", "sycl::aligned_alloc(64, 16, q, sycl::usm::alloc::host)"),
                     ("zeMemAllocHost", "zeMemAllocHost(ctx, &desc, 16, 64, &p)"),
                     ("unified_cache_raw_malloc_host", "unified_cache_raw_malloc_host(16, q)"),
                     ("unified_cache_malloc_host_tracked", "unified_cache_malloc_host_tracked(16, q, \"zz\")")):
        A(Case("s2c-host", "a new caller of %s" % call.split("(")[0], plant(
            "void zzplant_host(sycl::queue & q, void * ctx) {\n    void * desc = nullptr;\n    void * p = (void *) %s;\n    (void) p; (void) desc;\n}\n"
            % call), "FAIL", "E-RAW", nm))
    A(Case("s2c-host", "a new caller through a using-declaration is still the name", plant(
        "using sycl::malloc_host;\nvoid zzplant_host(sycl::queue & q) {\n    void * p = malloc_host(16, q);\n    (void) p;\n}\n"),
        "FAIL", "E-RAW", "malloc_host"))
    A(Case("s2c-host", "a macro that expands to sycl::malloc_host", plant(
        "#define zz_alloc(n, q) sycl::malloc_host(n, q)\n"), "FAIL", "E-RAW", "#define zz_alloc", planted=False))
    A(Case("s2c-host", "a macro that expands to the generic sycl::malloc", plant(
        "#define zz_alloc(n, q) sycl :: malloc(n, q, sycl::usm::alloc::host)\n"), "FAIL", "E-RAW", "#define zz_alloc", planted=False))
    A(Case("s2c-host", "a string naming zeMemAllocHost", plant(
        "void zzplant_host() {\n    const char * s = \"zeMemAllocHost\";\n    (void) s;\n}\n"), "FAIL", "E-RAW", "zzplant_host"))
    A(Case("s2c-host", "the C library's malloc and aligned_alloc are not USM entry points (control)", plant(
        "void zzplant_host() {\n    void * a = std::malloc(16);\n    void * b = ::aligned_alloc(64, 64);\n    void * c = malloc(8);\n"
        "    (void) a; (void) b; (void) c;\n}\n"), "PASS", planted=False))
    A(Case("s2c-host", "a comment, a string and a longer identifier containing the name are not hits (control)", plant(
        "// sycl::malloc_host(16, q) and zeMemAllocHost\nvoid zzplant_host() {\n    int malloc_hostile = 1;\n    int my_malloc_host_count = 2;\n"
        "    const char * s = \"sycl::malloc_host aligned_alloc_host\";\n    (void) malloc_hostile; (void) my_malloc_host_count; (void) s;\n}\n"),
        "PASS", planted=False))
    A(Case("s2c-host", "a declaration of a function named malloc_host is not a use (control)", plant(
        "void * malloc_host(unsigned long n);\nvoid zzplant_host() {\n}\n"), "PASS", planted=False))
    A(Case("s2c-host", "a second sycl::malloc_host call in the raw wrapper breaks its pin", replace_in_function(
        "unified-cache.cpp", r"void \* unified_cache_raw_malloc_host\(size_t size, const sycl::queue & queue\) \{",
        "ptr = sycl::malloc_host(size, queue);", "ptr = sycl::malloc_host(size, queue);\n        ptr = sycl::malloc_host(size, queue);"),
        "FAIL", "allowlist", "E-CHAIN-HOST-RAW covers 3 finding(s) but pins 2"))
    A(Case("s2c-host", "the raw host wrapper renamed leaves its exemption matching nothing", replace_token(
        "unified-cache.cpp", "unified_cache_raw_malloc_host", "unified_cache_raw_malloc_host_zz"), "FAIL", "allowlist",
        "E-CHAIN-HOST-RAW matches nothing"))
    A(Case("s2c-host", "a second call in the raw wrapper's context overload breaks the same pin", replace_in_function(
        "unified-cache.cpp", r"void \* unified_cache_raw_malloc_host\(size_t size, const sycl::context & ctx\) \{",
        "ptr = sycl::malloc_host(size, ctx);", "ptr = sycl::malloc_host(size, ctx);\n        ptr = sycl::malloc_host(size, ctx);"),
        "FAIL", "allowlist", "E-CHAIN-HOST-RAW covers 3 finding(s) but pins 2"))
    A(Case("s2c-host", "a third caller of the tracked host allocation beside the two CACHE_BACKING sites", plant(
        "void zzplant_host(sycl::queue & q) {\n    void * p = unified_cache_malloc_host_tracked(16, q, \"zz\");\n    (void) p;\n}\n"),
        "FAIL", "E-RAW", "unified_cache_malloc_host_tracked"))
    A(Case("s2c-host", "the staging buffer's site renamed leaves E-BACKING-STAGING matching nothing", replace_token(
        "unified-cache.cpp", "onednn_graph_scratch_ensure_flag_slab_locked", "onednn_graph_scratch_ensure_flag_slab_locked_zz"),
        "FAIL", "allowlist", "E-BACKING-FLAG-SLAB matches nothing"))
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
    if case.allow_nodes:
        found, _ = analyse(case.allow_from(base_files) if case.allow_from else files)
        ents = []
        for j, spec in enumerate(case.allow_nodes):
            hit = sorted([v for v in found if v.code == spec["code"] and spec["func"] in v.func], key=lambda v: v.line)
            if len(hit) <= spec.get("nth", 0):
                raise SystemExit("matrix setup error: no %s finding in %s for allow_nodes of %r" % (spec["code"], spec["func"], case.label))
            v = hit[spec.get("nth", 0)]
            e = {"id": "E-ZZ-N%d" % j, "code": spec["code"], "key": v.key, "count": 1, "reason": "mutation-matrix planted node"}
            if spec["code"] in H_OUTCOME:
                e["outcome"] = H_OUTCOME[spec["code"]]
            if spec["code"] == "H-UNCONV":
                e["ticket"] = v.name
            e.update(spec.get("extra", {}))
            ents.append(e)
        al = dict(al, entries=list(al.get("entries", [])) + ents)
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
        n += sum(1 for r in fa["catches"] if "zz" in r["func"])
        n += sum(1 for r in fa["hrecs"] if "zz" in r["func"])
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
    n_cases = len(matrix_cases())
    for arg, name, why in (("4/4", "4/4", "K not below N"), ("x", "x", "not K/N"), ("0/0", "0/0", "N of zero"),
                           ("%d/%d" % (n_cases, n_cases + 1), "<cases>/<cases+1>", "a slice with no case")):
        rc, text = subprocess_gate(["--mutation-matrix", "--shard", arg])
        out.append(("--shard %s (%s) is refused before the gate runs" % (name, why), rc == 2, "rc=%d" % rc))
    rc, text = subprocess_gate(["--shard", "0/4"])
    out.append(("--shard without --mutation-matrix is an error", rc == 2, "rc=%d" % rc))
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
    """The ctest registrations never pass a regeneration flag, the plain gate is its own test, and the matrix shards are
    registered in full: a foreach over 0..N-1 whose command passes `--shard ${var}/N`, each with a TIMEOUT."""
    text = (Path(root) / "ggml/src/ggml-sycl/CMakeLists.txt").read_text()
    plain = [m.group(0) for m in re.finditer(r"add_test\(NAME test-sycl-alloc-zone-contract\s[^)]*\)", text, flags=re.S)]
    loops = [m for m in re.finditer(r"foreach\((\w+)((?:\s+\d+)+)\s*\)(.*?)endforeach\(\)", text, flags=re.S)
             if "test-sycl-alloc-zone-contract-m" in m.group(3)]
    loop = loops[0] if len(loops) == 1 else None
    plain_props = re.findall(r"set_tests_properties\(test-sycl-alloc-zone-contract PROPERTIES[^)]*\)", text)
    ok = len(plain) == 1 and "--mutation-matrix" not in plain[0] and loop is not None \
        and len(plain_props) == 1 and 'LABELS "sycl;host-only;ast"' in plain_props[0] \
        and re.search(r"\bTIMEOUT\s+120\b", plain_props[0]) is not None
    detail = "%d plain registration(s)" % len(plain)
    if loop is not None:
        var, idx, body = loop.group(1), [int(x) for x in loop.group(2).split()], loop.group(3)
        m = re.search(r"--shard\s+\$\{%s\}/(\d+)" % re.escape(var), body)
        n = int(m.group(1)) if m else -1
        props = re.findall(r"set_tests_properties\([^)]*\)", body)
        ok = ok and "--mutation-matrix" in body and n == len(idx) == SHARDS and idx == list(range(n)) \
            and "NAME test-sycl-alloc-zone-contract-m${%s}" % var in body \
            and len(props) == 1 and re.search(r"\bTIMEOUT\s+600\b", props[0]) is not None \
            and 'LABELS "sycl;host-only;ast;mutation"' in props[0]
        detail += ", %d shard(s) of %d, timeout %s" % (len(idx), n, "set" if props and "TIMEOUT" in props[0] else "MISSING")
    region = "".join(plain) + (loop.group(0) if loop is not None else "")
    ok = ok and "--write-debt" not in region and "--allow-growth" not in region
    return ok, detail


def cmake_mutants(root):
    """cmake_witness must fail on a registration that drops a shard, miscounts them, passes a regeneration flag, loses
    the plain test or a TIMEOUT, and pass on the real one: a witness that cannot fail proves nothing."""
    import tempfile
    rel = "ggml/src/ggml-sycl/CMakeLists.txt"
    text = (Path(root) / rel).read_text()
    mutants = [
        ("drops shard 3 from the foreach", lambda t: t.replace("foreach(zc_shard 0 1 2 3)", "foreach(zc_shard 0 1 2)", 1)),
        ("passes the wrong shard count", lambda t: t.replace("${zc_shard}/4", "${zc_shard}/5", 1)),
        ("passes --write-debt to a shard", lambda t: t.replace("--mutation-matrix --shard", "--write-debt --mutation-matrix --shard", 1)),
        ("loses the plain gate's test", lambda t: t.replace("add_test(NAME test-sycl-alloc-zone-contract\n", "add_test(NAME test-sycl-alloc-zone-contract-x\n", 1)),
        ("drops the plain gate's TIMEOUT", lambda t: t.replace('PROPERTIES LABELS "sycl;host-only;ast" TIMEOUT 120)', 'PROPERTIES LABELS "sycl;host-only;ast")', 1)),
        ("loses a shard's TIMEOUT", lambda t: t.replace('"sycl;host-only;ast;mutation" TIMEOUT 600', '"sycl;host-only;ast;mutation"', 1)),
        ("runs the matrix in the plain test", lambda t: t.replace("../../..)\n    set_tests_properties(test-sycl-alloc-zone-contract PROPERTIES",
                                                                    "../../.. --mutation-matrix)\n    set_tests_properties(test-sycl-alloc-zone-contract PROPERTIES", 1)),
    ]
    out = []
    for label, fn in mutants:
        m = fn(text)
        if m == text:
            out.append(("a registration that %s is refused" % label, False, "mutation did not apply"))
            continue
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "ggml/src/ggml-sycl").mkdir(parents=True)
            (Path(d) / rel).write_text(m)
            ok, detail = cmake_witness(d)
        out.append(("a registration that %s is refused" % label, not ok, detail))
    ok, detail = cmake_witness(root)
    out.append(("the real registration is accepted", ok, detail))
    return out


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


def shard_slice(cases, k, n):
    """Shard k of n: every n-th case from k. Deterministic, so the union over k is the case list and no case repeats."""
    return cases[k::n]


def parse_shard(text):
    """(k, n) of a `K/N` argument. A malformed one, or a slice with no case in it, is a ValueError naming it: an empty
    shard would report "0 wrong" having checked nothing."""
    m = re.fullmatch(r"(\d+)/(\d+)", text or "")
    if not m or not (0 <= int(m.group(1)) < int(m.group(2))):
        raise ValueError("--shard takes K/N with 0 <= K < N, got %r" % text)
    k, n = int(m.group(1)), int(m.group(2))
    cases = matrix_cases()
    if not shard_slice(cases, k, n):
        raise ValueError("--shard %s selects no case (the matrix has %d); an empty shard checks nothing" % (text, len(cases)))
    return k, n


def run_matrix(files, allowlist, debt, root, shard=None):
    """`shard` is (k, n) or None. Every shard re-checks the unmutated baseline and runs its slice of the cases. Shard 0 also
    runs the coverage checks (every witness has a FAIL case, no case names an unknown witness, the matcher's self-test)
    and the process-level witnesses, so the whole matrix is still checked exactly once; a plain run is shard 0 of 1."""
    k, n = shard if shard else (0, 1)
    fails, report, _, _ = run_gate(files, allowlist, debt)
    if fails:
        print("FAIL: the unmutated baseline is red, so a matrix scored against it proves nothing:")
        for f in fails[:20]:
            print("  " + f)
        return 1
    print("baseline: PASS (the matrix is scored against a green tree)")
    bad = 0
    all_cases = matrix_cases()
    cases = shard_slice(all_cases, k, n)
    if not cases:
        print("FAIL: shard %d/%d selects no case; an empty shard checks nothing" % (k, n))
        return 1
    print("shard %d/%d: %d of %d case(s)" % (k, n, len(cases), len(all_cases)))
    if k == 0:
        for w in sorted(set(c.wid for c in all_cases) - set(WITNESSES)):
            print("FAIL: case witness %s is not in WITNESSES" % w)
            bad += 1
        for w in WITNESSES:
            if w in ("f", "cmake", "m6", "m9", "m14", "r2m4", "r2m8"):
                continue
            if not any(c.wid == w and c.expect == "FAIL" for c in all_cases):
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
    extra = []
    if k == 0:
        extra = [("f", "missing tree_sitter_language_pack exits 1 and names it") + f_witness_missing_pack(),
                 ("m9", "an unscanned .inl file fails the gate") + m9_witness()]
        extra += [("cmake", label, ok, d) for label, ok, d in cmake_mutants(root)]
        extra += [("m6", label, ok, d) for label, ok, d in m6_witnesses()]
        extra += [("r2m4", label, ok, d) for label, ok, d in r2m4_witnesses(allowlist)]
        extra += [("r2m8", label, ok, d) for label, ok, d in r2m8_witnesses()]
        extra += [("m14", label, ok, d) for label, ok, d in m14_witness(files, allowlist, debt)]
    for wid, label, ok, detail in extra:
        print("%s witness %-4s %-80s (%s)" % ("ok  " if ok else "FAIL", wid, label, str(detail)[:80]))
        bad += 0 if ok else 1
    print("matrix shard %d/%d: %d case(s), %d wrong" % (k, n, len(cases) + len(extra), bad))
    return 1 if bad else 0


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--data", default=None, help="directory holding allowlist.json and debt.json")
    ap.add_argument("--mutation-matrix", action="store_true")
    ap.add_argument("--shard", default=None, metavar="K/N",
                    help="with --mutation-matrix, run shard K of N (shard 0 also runs the coverage and process-level witnesses)")
    ap.add_argument("--list", action="store_true", help="print every finding before the allowlist and debt")
    ap.add_argument("--write-debt", action="store_true",
                    help="rewrite debt.json from the current tree; refuses to add entries. Never used by the ctest")
    ap.add_argument("--allow-growth", action="store_true", help="with --write-debt, permit new entries (key migration or seeding only)")
    a = ap.parse_args()
    if a.allow_growth and not a.write_debt:
        ap.error("--allow-growth only means something with --write-debt")
    if a.mutation_matrix and (a.list or a.write_debt):
        ap.error("--mutation-matrix cannot be combined with --list or --write-debt")
    if a.shard is not None and not a.mutation_matrix:
        ap.error("--shard only means something with --mutation-matrix")
    shard = None
    if a.shard is not None:
        try:
            shard = parse_shard(a.shard)
        except ValueError as exc:
            ap.error(str(exc))
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
        return run_matrix(files, allowlist, debt, a.root, shard)
    return 0


if __name__ == "__main__":
    sys.exit(main())
