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

What this unit (S2a-S2d, S3-0 and its review fold) enforces
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
      (canonical contract section 9.1, the dpct row; vendored upstream, not edited, dead in-tree; rulings M247 second).
      `dpct_memcpy` and `async_dpct_memcpy` are forbidden there too: their 3-D host-staged paths build the host_buffer whose
      std::malloc is allowlisted under clause (q), so a caller outside helper.hpp is what would make that allowlist reachable
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
  (q) libc allocation primitives (S3-0): mmap, mmap64, mremap, posix_memalign, memalign, aligned_alloc, valloc, pvalloc, malloc, calloc,
      realloc, reallocarray, strdup, strndup and VirtualAlloc, as a call, an address-of or a value, bare or qualified by `std::` or a
      leading `::`, and in a #define body, outside the allowlist (code E-LIBC, keyed like E-RAW). A member (`pool.realloc`), a name
      qualified by anything else (`pool_alloc::realloc`; `sycl::malloc` is clause (e)'s) and the declaration of a function with the
      name are not hits. In a #define body and in the lexical pass the qualifier is read backwards across spaces, newlines and
      backslash continuations, so `pool :: realloc` and `p . malloc` are not hits and `std :: malloc` is. `free`, `new`, `operator new` and
      the STL containers' allocators are out of scope (see the README). A reason or a cite names no source line (file:NNN) and no
      ruling-ledger id (ruling Mnnn Rn); the CHECK_TRY_ERROR debt entry carries a cite, and a re-key of it fails. The unified cache owns every byte the
      backend allocates, so a libc allocation is a violation unless the allowlist says why it holds no tensor, KV, scratch, pinned or
      USM bytes. A debt entry for it carries a fate and a cite, as E-RAW's does.
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
  (i)-(o) and witness 9 (S2d-1). Each is keyed on a subject the tree may not have yet, and says so:
      `DORMANT <clause>: subject <symbol> absent` is printed, never a silent pass. A subject name that appears in a
      shape the matcher misses (a macro, a lambda or variable, an alias, a pointer to member) is an X-LATCH
      failure, so a clause cannot stay dormant for ever behind a respelling.
      (i) defining ggml_sycl_replan_token_held retires onednn_w_retry_lost_cas and onednn_pp_a_relock_busy_pre_l0
      (I-RETRY). (k) release_retained_referencing(... RETAINED_REAP_EVENTS_COMPLETE_BY_CALLER ...) inside
      ggml_sycl_run_runtime_context_transaction retires onednn_pp_a_reclaim_query_interim (K-INTERIM).
      (l) owner_use_count: its callers outside mem-handle.* are allowlisted (L-CALLER), no free is controlled by it,
      directly or through a local (L-FREE), and replace_within begins with replace_within_count_guard (L-GUARD).
      (j) the fit functions onednn_pp_a_bytes / onednn_pp_w_bytes read per-model sources only (J-SOURCE), and
      zone_is_onednn_reorder_eligible is called only by the late stage's classifier (J-DISPATCH).
      (m) every appendix row whose zone is SCRATCH is on the floor list, covered by a named peak, or excluded by class
      (M-SCRATCH); the lists carry no stale row (M-STALE); the floor is tied to GGML_SYCL_COMPUTE_ARENA_MB (M-FLOOR);
      the census table is scripts/sycl-alloc-zone-contract/appendix-rows.json (M-DATA when missing).
      (n) the result of each declined-result consumer (DnnlGemmWrapper::gemm, row_gemm, ..., get_scratchpad_mem) is
      consumed, in the library and in the tests that call it: an expression statement, a comma's left operand or a
      void cast fails (N-VOID); each listed declaration carries [[nodiscard]] (N-NODISCARD). Both may be debt.
      (o) each C-term consumer submits on its census row's queue, pinned at the call's argument; an acquire call
      with no row fails (O-NOROW, O-ROW, O-QUEUE).
      Witness 9: each model-shaped *_bytes() function is called by its allocation sites and by the zone sizing
      (Z9-SITE, Z9-SIZING), dormant until it is defined.
      Known S2d gaps: (n) counts every consuming position (assignment, return, condition, argument) as used;
      (j) covers only the two named fit functions; (l)'s count callers are allowlist entries added as each lands.
  (p) one routed predicate and one home for the support decision (witnesses 37-38), in five blocks, each dormant until
      its subject is defined (the subjects arrive with beni b1/b2): p-route (the oneDNN SDPA calls sit in branches that call
      ggml_sycl_fattn_onednn_dispatch_routed and test none of its flags; ggml_sycl_flash_attn_ext_onednn_plan is called in
      fattn.cpp only in ggml_sycl_fattn_onednn_route_admits, and the value function calls the routed predicate with nullptr,
      nullptr for its context and plan), p-home (getenv of GGML_SYCL_FLASH_ATTN_EXT is read once in the whole repository,
      inside ggml_sycl_flash_attn_ext_enabled, which is called once, first, in ggml_sycl_fattn_shape_supported; the routed
      predicate's body is pinned; ggml_sycl_flash_attn_ext_supported and route_admits hold no support clause; the call-site
      censuses of fattn_vec_supports_head_dim, the tile screen and kv_pair_of; the supports_op case line), p-fill (the pinned
      kv_is_fp8 fill line per fill function, and every write of kv_is_fp8 is one of them), p-layer (the KV layer name is
      parsed once, in ggml_sycl_kv_cache_layer_of) and p-charge (the charge side names no support helper, compares no head
      dim with a literal, and holds `if (params.ne00 == 512) {` once in the value function in b2 and nowhere in b1). The
      phase is b2 when onednn_graph_scratch_bytes is defined, else b1; a b2 tree that defines
      placement_plan_set_routed_head_maxima fails. Codes P-ROUTE, P-HOME, P-FILL, P-LAYER, P-CHARGE are never debt.
      The matrix builds the b1 and b2 trees by transforming today's (b1_tree, b2_tree), since a twin cannot be planted
      beside functions the clause constrains in place.
      Known (p) gaps: the route's D=512 hatch line is not pinned by text, any `if` whose condition spells 512 and calls
      ggml_sycl_fa_onednn_d512_enabled passes; the routing function's "reads the decline first" is not checked (its decline
      reader has no name yet); the charge side's walk helpers are not named, so the head-dim and helper rules cover the two
      named charge functions' bodies and the helper-name rule covers all of unified-cache.cpp/.hpp.

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
  check-sycl-alloc-zone-contract.py [--root REPO] --witnesses
  check-sycl-alloc-zone-contract.py [--root REPO] --list
  check-sycl-alloc-zone-contract.py [--root REPO] --write-debt [--allow-growth]
"""
import argparse
import copy
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
# dpct_memcpy and async_dpct_memcpy are in the list because their 3-D host-staged paths build a host_buffer, whose constructor
# std::mallocs (the E-LIBC-DPCT-HOSTBUF-MALLOC entries): a caller outside helper.hpp is what would make those allowlisted sites reachable.
DPCT_FUNCS = ("dpct_malloc", "dpct_memcpy", "async_dpct_memcpy")
DPCT_CLASSES = ("device_memory", "global_memory", "constant_memory", "shared_memory")
# Clause (q): the C library's allocation primitives (and Windows' VirtualAlloc). They are matched as a bare name or one qualified
# by `std::` or a leading `::`, so a member (`pool.realloc`, `pool_alloc::realloc`) and sycl::malloc (clause (e)'s) are not hits,
# and the declaration of a function with the name is not a use. Code E-LIBC; its debt entries carry a fate and a cite like E-RAW's.
LIBC_NAMES = ("mmap", "mmap64", "mremap", "posix_memalign", "memalign", "aligned_alloc", "valloc", "pvalloc", "malloc", "calloc",
              "realloc", "reallocarray", "strdup", "strndup", "VirtualAlloc")
RAW_CODES = ("E-RAW", "E-LIBC")
# Text (a #define body, the lexical pass of an ERROR-root file) has no tree, so the qualifier is read backwards from the name,
# across spaces, newlines and `\`-continuations (libc_text_hit). The name itself is a whole word.
LIBC_WORD_RE = re.compile(r"(?<![A-Za-z0-9_])(%s)(?![A-Za-z0-9_])" % "|".join(LIBC_NAMES))
LIBC_CALL_TAIL_RE = re.compile(r"\s*(?:<[^>]*>)?\s*\(")
LIBC_CONT_RE = re.compile(r"\\(?=\r?\n)")
# G-CATCH is the one other code whose debt entry carries a fate, and only this key: the CHECK_TRY_ERROR macro's handler, which
# step 5.4a rewrites. It can carry no other fate, and no other handler carries one.
CHECK_TRY_ERROR_KEY = "common.hpp::#define CHECK_TRY_ERROR::catch_macro:exception#0"
CHECK_TRY_ERROR_FATE = "converted-by-5.4a"
# A reason or a cite names something a reader can open in the repo. A source line rots with the next edit, and a ruling-ledger id
# with a round ("ruling M265 R2") names a ledger that is not in the tree: cite the design step or the contract section instead.
SRC_LINE_RE = re.compile(r"\.(?:hpp|cpp|h|c):\d+")
LEDGER_RE = re.compile(r"\b[Rr]ulings? M\d+ R\d+\b")

SHARDS = 8   # the ctest registers this many shards; cmake_witness pins the registration to it

DEBT_DOC = ("Read by scripts/check-sycl-alloc-zone-contract.py (clauses a-h, n and q; only N-VOID and N-NODISCARD may be debt). Shrink-only: a violation not listed "
            "fails, and a listed entry that no longer violates fails. Every E-RAW and E-LIBC entry carries a fate (deleted-by-*, "
            "converted-by-*, sanctioned-internal, sanctioned-vendored or pending-disposition) and a cite, so an entry no step will ever "
            "shrink is visible as a mislabelled allowlist entry; the one G-CATCH entry for the CHECK_TRY_ERROR macro carries "
            "converted-by-5.4a and no other fate. Regenerate with `python3 "
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

# Clauses (i)-(o), witness 9 and the dormancy latch (S2d). Only a declined result that is dropped today (N-*) is debt; the rest
# are clean on today's tree, so a finding of any other code is an allowlisted node or a fix, never a list entry.
S2D_CODES = ("I-RETRY", "K-INTERIM", "L-CALLER", "L-FREE", "L-GUARD", "J-SOURCE", "J-DISPATCH", "M-SCRATCH", "M-STALE", "M-FLOOR",
             "M-DATA", "N-VOID", "N-NODISCARD", "O-NOROW", "O-ROW", "O-QUEUE", "Z9-SITE", "Z9-SIZING", "P-ROUTE", "P-HOME", "P-FILL",
             "P-LAYER", "P-CHARGE", "X-LATCH")
S2D_DEBT = ("N-VOID", "N-NODISCARD")
S2D_NEVER_ALLOW = ("X-LATCH", "P-ROUTE", "P-HOME", "P-FILL", "P-LAYER", "P-CHARGE")
CODES = ("A-ERROR", "A-LEXICAL", "A-TOKEN", "B-BRACE", "B-FORM", "B-TIER", "C-COHORT", "C-SITE", "D-ZONE",
         "D-ZONE-COUNT", "D-FORBID", "D-FORBID-FALSE", "E-RAW", "E-LIBC", "G-CATCH", "DEFER-C") + H_CODES + S2D_CODES


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


def core_files(files):
    """The scanned tree without the test sources that clause (n) reads."""
    return {r: s for r, s in files.items() if not r.startswith((TEST_PREFIX, REPO_PREFIX))}


def load_test_sources(root):
    """Test sources that spell a name clause (n) lists, keyed `tests/<path>`: the repository's tests/ and the scope's own
    tests/ directory. Read for clause (n) only; the other clauses never see them."""
    names = [m[1].encode() for m in N_MEMBERS] + [u.encode() for u in N_UNIQUE] + [P_ENV.encode()]
    out = {}
    for base, prefix in ((Path(root) / "tests", TEST_PREFIX), (Path(root) / SCOPE_SUBDIR / "tests", TEST_PREFIX + "ggml-sycl/")):
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix.lower() in SCAN_SUFFIXES:
                b = p.read_bytes()
                if any(n in b for n in names):
                    out[prefix + str(p.relative_to(base))] = b
    return out


REPO_PREFIX = "repo/"   # sources outside the scope that read the switch, for clause (p)'s read count only
REPO_SKIP_DIRS = frozenset([".git", "docs", ".llm-wiki", "node_modules", "__pycache__"])
REPO_SUFFIXES = (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".cu", ".cuh", ".m", ".mm", ".inl", ".cl")


def load_repo_reads(root):
    """Every C or C++ source under the repository, outside docs/, .llm-wiki/, build directories, the scope itself and the two
    tests/ directories already read, that spells the switch clause (p) counts, keyed `repo/<path>`."""
    base, scope = Path(root), (Path(root) / SCOPE_SUBDIR).resolve()
    out = {}
    for d, dirs, names in os.walk(base):
        dirs[:] = sorted(x for x in dirs if x not in REPO_SKIP_DIRS and not x.startswith("build") and
                         (Path(d) / x).resolve() != scope and (Path(d) / x) != base / "tests")
        for n in sorted(names):
            if n.lower().endswith(REPO_SUFFIXES):
                b = (Path(d) / n).read_bytes()
                if P_ENV.encode() in b:
                    out[REPO_PREFIX + str((Path(d) / n).relative_to(base))] = b
    return out


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


def libc_text_hits(text, need_call):
    """(name, offset) of each libc allocation primitive in text with no tree: a #define body, or the blanked text of an
    ERROR-root file (need_call: the name must be called). A member (`p . malloc`, `p->malloc`) and a name qualified by anything
    but `std` or a bare `::` (`pool :: realloc`, `sycl :: malloc`, `T<x>::malloc`) are not hits; the qualifier is read across
    spaces, newlines and line continuations."""
    text = LIBC_CONT_RE.sub(" ", text)
    out = []
    for m in LIBC_WORD_RE.finditer(text):
        if need_call and not LIBC_CALL_TAIL_RE.match(text, m.end()):
            continue
        i = m.start()
        while i > 0 and text[i - 1].isspace():
            i -= 1
        if i >= 1 and text[i - 1] == ".":
            continue
        if i >= 2 and text[i - 2:i] == "->":
            continue
        if i >= 2 and text[i - 2:i] == "::":
            j = i - 2
            while j > 0 and text[j - 1].isspace():
                j -= 1
            k = j
            while k > 0 and (text[k - 1].isalnum() or text[k - 1] == "_"):
                k -= 1
            if k < j:
                if text[k:j] != "std":
                    continue
            elif j > 0 and text[j - 1] == ">":
                continue
        out.append((m.group(1), m.start()))
    return out


def libc_use(src, n):
    """(top, is_call) for an identifier naming a libc allocation primitive, else None. Not a use: the name of a function
    declaration or definition, or a name qualified by anything but `std` (sycl::malloc is clause (e)'s)."""
    p = parent(n)
    if p is None or kind(p) == "function_declarator":
        return None
    if kind(p) == "qualified_identifier":
        if not same(fld(p, "name"), n):
            return None
        sc = fld(p, "scope")
        if sc is not None and txt(src, sc).strip().lstrip(":").strip() != "std":
            return None
    return callee_ident(src, n)


def scan_file(rel, src, ctx):
    """Facts about one file, as plain data: constructions, form findings, raw hits, error tokens."""
    value_types, all_types = ctx.value_types, ctx.all_types
    env = Env(rel, src, ctx)
    root = parse(src)
    root_error = kind(root) == "ERROR"
    constructions, raws, errtoks, forms = [], [], [], []
    catches, hrecs, libcs = [], [], []
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
                b = LIBC_CONT_RE.sub(" ", txt(src, body))
                for r in RAW_NAMES + tuple("sycl::" + q for q in RAW_QUALIFIED):
                    if re.search(r"(?<![A-Za-z0-9_])" + re.escape(r).replace("sycl::", r"sycl\s*::\s*") + r"(?![A-Za-z0-9_])", b):
                        raws.append({"func": "#define " + txt(src, nm), "name": r, "line": line_of(n),
                                     "form": "macro", "nodekind": "preproc", "text": r, "err": False})
                for lname, _ in libc_text_hits(b, False):
                    libcs.append({"func": "#define " + txt(src, nm), "name": lname, "line": line_of(n),
                                  "form": "macro", "nodekind": "preproc", "text": lname, "err": False})
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
        if k == "identifier" and txt(src, n) in LIBC_NAMES:
            use = libc_use(src, n)
            if use is not None:
                top, is_call = use
                libcs.append({"func": enclosing(src, n), "name": txt(src, n), "line": line_of(n),
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
        ctext = clean.decode("utf-8", "replace")
        for lname, off in libc_text_hits(ctext, True):
            libcs.append({"func": "<lexical>", "name": lname, "line": ctext.count("\n", 0, off) + 1,
                          "form": "lexical", "nodekind": "lexical", "text": lname, "err": True})
    return {"constructions": constructions, "raws": raws, "errtoks": errtoks, "forms": forms, "catches": catches,
            "libcs": libcs, "hrecs": hrecs, "root_error": root_error, "lexical": lexical, "lex_count": lex_count, "ast_count": ast_count}


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
    core = core_files(files)
    ctx = Ctx(core)
    viols = []
    stats = {"constructions": 0, "files": len(core), "value_types": sorted(ctx.value_types),
             "all_types": sorted(ctx.all_types)}
    for rel in sorted(core):
        fa = facts_for(rel, core[rel], ctx)
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
        seen = {}
        for r in fa["libcs"]:
            base = "%s::%s::%s:%s:%s" % (rel, r["func"], r["nodekind"], r["name"], text_hash(r["text"]))
            i = seen.get(base, 0)
            seen[base] = i + 1
            viols.append(V("E-LIBC", "%s#%d" % (base, i), rel, r["line"], r["func"], r["name"],
                           "libc allocation primitive %s (%s) outside the allowlist: the unified cache owns every byte "
                           "the backend allocates" % (r["name"], r["form"])))
    s2d = s2d_findings(files)
    viols.extend(s2d.viols)
    stats["dormant"], stats["active"] = s2d.dormant, sorted(s2d.active)
    return viols, stats


# ---------------------------------------------------------------- S2d: clauses (i)-(o) and the dormancy latch
# These clauses key on named functions, so their facts are per-file occurrences of the names they care about, cached by
# content like the (a)-(h) facts. They hold tree-sitter nodes and live as long as the process, as _PARSE does.

TEST_PREFIX = "tests/"   # test sources, read for clause (n) only: clauses (a)-(h) never see them (the scope excludes tests)

I_ACCESSOR = "ggml_sycl_replan_token_held"
I_RETIRED = ("onednn_w_retry_lost_cas", "onednn_pp_a_relock_busy_pre_l0")
K_TXN, K_REAP, K_MODE = "ggml_sycl_run_runtime_context_transaction", "release_retained_referencing", \
    "RETAINED_REAP_EVENTS_COMPLETE_BY_CALLER"
K_INTERIM = "onednn_pp_a_reclaim_query_interim"
L_COUNT, L_REPLACE, L_GUARD = "owner_use_count", "replace_within", "replace_within_count_guard"
L_FREES = ("unified_free", "zone_free", "reset", "enqueue_deferred_zone_free")
L_TLSF_FREE = re.compile(r"tlsf\w*free|free\w*tlsf", re.I)
L_MEM_HANDLE = ("mem-handle.cpp", "mem-handle.hpp")   # the count's own implementation
J_FIT = ("onednn_pp_a_bytes", "onednn_pp_w_bytes")      # A's fit and W's term: per-model sources only
J_BANNED = re.compile(r"g_tensor_inventory_\w*|unified_cache_get_planned_(?:pp_moe_)?onednn_\w*")
J_ELIGIBLE = "zone_is_onednn_reorder_eligible"
M_FLOOR_FN, M_FLOOR_ENV = "ensure_planned_arena_zones", "GGML_SYCL_COMPUTE_ARENA_MB"

# Clause (n): the declined-result consumers. A class member is matched as `Class::name(` anywhere, or as a bare `name(`
# only inside that class; the two unique names match anywhere.
N_MEMBERS = (("DnnlGemmWrapper", "gemm"), ("DnnlGemmWrapper", "row_gemm"), ("DnnlGemmWrapper", "woq_gemm_q4_0"),
             ("DnnlGemmWrapper", "woq_gemm_q8_0"), ("DnnlGemmWrapper", "woq_gemm_q4_0_impl"),
             ("DnnlGemmWrapper", "gemm_batch_strided"), ("DnnlGemmWrapper", "woq_gemm_batch_mxfp4"),
             ("DnnlSoftmaxWrapper", "softmax"), ("DnnlEltwiseWrapper", "eltwise"), ("DnnlBinaryWrapper", "binary_broadcast_row"))
N_UNIQUE = ("get_scratchpad_mem", "ggml_sycl_mul_mat_batched_sycl")
N_LAST = frozenset([m[1] for m in N_MEMBERS] + list(N_UNIQUE))

# Clause (o): the acquire tokens whose consumers must submit on their census row's queue.
O_ACQUIRE = ("acquire_onednn_pp_scratch", "ggml_sycl_set_rows_stage_ptr")
O_EXECUTE = "dnnl::graph::sycl_interop::execute"

# Witness 9: each model-shaped exact `*_bytes()` function is called by every allocation site that draws it and by the
# zone sizing. (function, [(file, site function)]). The subjects are absent from the tree today.
Z9_SIZING_FILES = ("zone-sizing.cpp", "unified-cache.cpp")
Z9_SITES = (
    ("load_reorder_temp_bytes", (("convert.cpp", "convert_alloc_device_scratch"), ("ggml-sycl.cpp", "arena_device_alloc"),
                                 ("ggml-sycl.cpp", "ggml_sycl_fill_xmx_tiled"), ("ggml-sycl.cpp", "ggml_sycl_fill_xmx_tiled_host"),
                                 ("ggml-sycl.cpp", "sycl_unified_device_temp_alloc"),
                                 ("unified-cache.cpp", "unified_cache::reserve_reorder_temp"))),
    ("woq_packed_bytes", (("gemm.hpp", "woq_gemm_q4_0_impl"),)),
    ("mmq_work_counter_bytes", (("mmq.cpp", "get_mmq_work_counter"),)),
    ("set_rows_stage_bytes", (("set_rows.cpp", "ggml_sycl_set_rows_stage_ptr"),)),
)

# Clause (p): one routed predicate, one home for the support decision. Each block is keyed on a subject the tree gains with
# beni (b1) and says so while it is absent. Blocks: route (the route's gates and the plan are combined in one function), home
# (the support decision), fill (kv_is_fp8), layer (the KV layer name), charge (the charge side calls no support helper).
P_ENV = "GGML_SYCL_FLASH_ATTN_EXT"
P_ADMITS, P_ROUTED, P_ROUTE_ENABLED, P_DISPATCH = ("ggml_sycl_fattn_onednn_route_admits", "ggml_sycl_fattn_onednn_routed",
                                                   "ggml_sycl_fattn_onednn_route_enabled", "ggml_sycl_fattn_onednn_dispatch_routed")
P_DECLINED = "ggml_sycl_onednn_graph_dispatch_declined"     # the one counting and logging read of the per-context decline
P_ENABLED, P_SHAPE_SUPPORTED, P_SUPPORTED = ("ggml_sycl_flash_attn_ext_enabled", "ggml_sycl_fattn_shape_supported",
                                             "ggml_sycl_flash_attn_ext_supported")
P_SHAPE_OF, P_KV_PAIR_OF, P_LAYER_OF = "ggml_sycl_fattn_shape_of", "ggml_sycl_fattn_kv_pair_of", "ggml_sycl_kv_cache_layer_of"
P_LOADFILL, P_VALUEFN = "placement_plan_set_routed_head_maxima", "onednn_graph_scratch_bytes"
P_ONEDNN, P_PLAN, P_VEC, P_TILE = ("ggml_sycl_flash_attn_ext_onednn", "ggml_sycl_flash_attn_ext_onednn_plan",
                                   "fattn_vec_supports_head_dim", "ggml_sycl_fattn_d512_tile_admissible")
P_FLASH, P_FAST_POLICY, P_SUPPORTS_OP = ("ggml_sycl_flash_attn_ext", "ggml_sycl_fattn_fast_decode_policy",
                                         "ggml_backend_sycl_device_supports_op")
P_FLAGS = ("g_sycl_fa_onednn_enabled", "g_sycl_paged_v2_enabled", "ggml_sycl_fa_onednn_d512_enabled")
P_BODY_BANNED = (P_VEC, P_ENABLED, P_KV_PAIR_OF, P_TILE)               # support clauses the supported/route bodies may not hold
P_FP8_CALL_RE = re.compile(r"(?<![\w.>])(\w*fp8\w*)\s*\(")               # ... nor any fp8 type helper (ggml_sycl_type_is_fp8_e4m3)
P_SUPPORT_HELPERS = (P_VEC, P_ENABLED, P_SHAPE_SUPPORTED, P_TILE)      # support helpers the charge side may not name
# The parameter and local names are pinned too, as 4.8 spells them: a rename in b1 fails loudly, so b1 updates this pin with the code.
P_DISPATCH_BODY = (
    "out = {}; if (!ggml_sycl_fattn_onednn_route_enabled(p)) { return false; } "
    "if (ggml_sycl_onednn_graph_dispatch_declined(ctx, p)) { out.stage = GGML_SYCL_FATTN_ONEDNN_ROUTE_STAGE_DECLINED; "
    "unified_cache_count_onednn_graph_mask_declined(ctx.device, static_cast<int>(site)); return false; } "
    "out.stage = GGML_SYCL_FATTN_ONEDNN_ROUTE_STAGE_PLANNED; "
    "return ggml_sycl_fattn_onednn_routed(p, d_v, multi_seq, &ctx, &out.plan);")
P_ROUTED_BODY = "return ggml_sycl_fattn_onednn_route_admits(p, multi_seq, ctx, plan_out) && ggml_sycl_fattn_shape_supported(p, d_v);"
P_CASE = "case GGML_OP_FLASH_ATTN_EXT: return ggml_sycl_flash_attn_ext_supported(op);"
P_FILL_RE = re.compile(r"^(\w+)\.kv_is_fp8=ggml_sycl_fattn_kv_pair_of\(\1\.K_type,\1\.V_type\)==GGML_SYCL_FATTN_KV_PAIR_FP8;$")
P_WRITE_RE = re.compile(rb"(?:\.|->)kv_is_fp8[ \t\r\n]*(?:[|&^]?=)[^=]")
P_GETENV_RE = re.compile(rb'(?:std\s*::\s*)?getenv\s*\(\s*"%s"\s*\)' % P_ENV.encode())
P_LAYER_RE = re.compile(rb'"cache_[kv]_l')
P_D512_LINE = "if(params.ne00==512){"
P_HEADDIM = r"(?:\w+(?:\[[^\]]*\])*(?:\.|->))*(?:ne00|ne10|ne\[\s*0\s*\]|head_dim_k|head_dim_v|d_v|D)"
P_CMP = r"(?:==|!=|<=|>=|<|>)"
P_LITERAL_CMP_RE = re.compile(r"(?<![\w.>])%s\s*%s\s*(\d+)|(\d+)\s*%s\s*%s(?!\w)" % (P_HEADDIM, P_CMP, P_CMP, P_HEADDIM))
P_NAMED = r"(?:[A-Z][A-Z0-9_]{2,}|k_\w+|k[A-Z]\w*)(?![\w(])"
P_NAMED_CMP_RE = re.compile(r"(?<![\w.>])%s\s*%s\s*%s|%s\s*%s\s*%s(?!\w)" % (P_HEADDIM, P_CMP, P_NAMED, P_NAMED, P_CMP, P_HEADDIM))
P_ARITH_RE = re.compile(r"(?<![\w.>])%s\s*[%%&]\s*[\w(]" % P_HEADDIM)
P_SWITCH_RE = re.compile(r"switch\s*\(\s*%s\s*\)" % P_HEADDIM)
P_CASE_RE = re.compile(r"case\s+(\d+|(?!GGML_TYPE_)[A-Z][A-Z0-9_]+)\s*:")
P_TYPE_CMP_RE = re.compile(r"(?:==|!=)\s*GGML_TYPE_\w+|GGML_TYPE_\w+\s*(?:==|!=)|case\s+GGML_TYPE_\w+\s*:")
P_LATCH_RE = re.compile(r"static\s+const\s+bool\s+(\w+)\s*=\s*ggml_sycl_fa_onednn_d512_enabled\s*\(\s*\)\s*;")
P_HD512 = P_HEADDIM + r"==512"
P_COMMENT_RE = re.compile(rb'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|/\*.*?\*/|//[^\n]*', re.S)
P_NAMES = frozenset([P_ADMITS, P_ROUTED, P_ROUTE_ENABLED, P_DISPATCH, P_ENABLED, P_SHAPE_SUPPORTED, P_SUPPORTED, P_SHAPE_OF,
                     P_KV_PAIR_OF, P_LAYER_OF, P_LOADFILL, P_VALUEFN, P_ONEDNN, P_PLAN, P_VEC, P_TILE, P_DECLINED])


S2D_NAMES = frozenset({I_ACCESSOR, K_TXN, K_REAP, K_MODE, K_INTERIM, L_COUNT, L_REPLACE, L_GUARD, J_ELIGIBLE, M_FLOOR_FN}
                      | set(I_RETIRED) | set(J_FIT) | set(N_LAST) | set(O_ACQUIRE) | {b for b, _ in Z9_SITES} | P_NAMES)
N_CLASSES = tuple(sorted({c for c, _ in N_MEMBERS}))
S2D_BYTES = tuple(sorted(n.encode() for n in S2D_NAMES)) + (O_EXECUTE.encode(),) + tuple(c.encode() for c in N_CLASSES)

_S2D = {}   # sha1 -> facts
MACRO_NOISE_RE = re.compile(r'"(?:\\.|[^"\\])*"|/\*.*?\*/|//[^\n]*', re.S)   # strings and comments of a #define body


class Occ:
    """One spelling of a name the S2d clauses key on: a definition, a declaration, a call, or any other shape."""
    __slots__ = ("name", "role", "node", "top", "call", "func", "line", "scope", "form")

    def __init__(self, name, role, node, top, call, func, line, scope, form):
        self.name, self.role, self.node, self.top, self.call = name, role, node, top, call
        self.func, self.line, self.scope, self.form = func, line, scope, form


def name_role(src, n):
    """(role, top, call) for an identifier that spells a function name: 'call' (callee of a call_expression), 'def' (the name
    of a function_definition), 'decl' (a `;`-terminated declaration), else 'other'."""
    top, is_call = callee_ident(src, n)
    if is_call:
        return "call", top, parent(top)
    top = n
    p = parent(top)
    while p is not None and kind(p) in ("qualified_identifier", "template_function") and same(fld(p, "name"), top):
        top, p = p, parent(p)
    if p is not None and kind(p) == "function_declarator" and same(fld(p, "declarator"), top):
        q = parent(p)
        while q is not None and kind(q) in ("pointer_declarator", "reference_declarator", "parenthesized_declarator"):
            q = parent(q)
        if q is not None and kind(q) == "function_definition":
            return "def", top, None
        if q is not None and kind(q) in ("declaration", "field_declaration"):
            return "decl", top, None
    return "other", top, None


def scope_of(src, top):
    """The last scope component of a qualified spelling (`DnnlGemmWrapper` in `DnnlGemmWrapper::row_gemm`), else None."""
    q = top
    if kind(q) == "template_function":
        q = fld(q, "name") or q
    if kind(q) == "qualified_identifier":
        sc = fld(q, "scope")
        return None if sc is None else txt(src, sc).split("::")[-1].strip()
    return None


def s2d_facts(src):
    """{'occ': [Occ], 'fdefs': [(last name, enclosing name, function_definition node)]} for one file. A file that spells none
    of the names is not walked."""
    h = _sha(src)
    if h in _S2D:
        return _S2D[h]
    occ, fdefs, aliases = [], [], []
    if any(b in src for b in S2D_BYTES):
        for n in walk(parse(src)):
            k = kind(n)
            if k in ("identifier", "field_identifier", "type_identifier", "namespace_identifier"):
                nm = txt(src, n)
                if nm in S2D_NAMES:
                    role, top, call = name_role(src, n)
                    occ.append(Occ(nm, role, n, top, call, enclosing(src, n), line_of(n), scope_of(src, top), "ast"))
            elif k == "call_expression":
                f = fld(n, "function")
                if f is not None and kind(f) == "qualified_identifier" and norm(txt(src, f)).replace(" ", "") == O_EXECUTE:
                    occ.append(Occ(O_EXECUTE, "call", f, f, n, enclosing(src, n), line_of(n), None, "ast"))
            elif k == "function_definition":
                d = fld(n, "declarator")
                while d is not None and kind(d) != "function_declarator":
                    d = fld(d, "declarator")
                body = fld(n, "body")
                if d is not None and fld(d, "declarator") is not None and body is not None:
                    fdefs.append((callee_last(txt(src, fld(d, "declarator"))), enclosing(src, body), n))
            elif k == "alias_declaration" or k == "type_definition":
                nm, tgt = (fld(n, "name"), fld(n, "type")) if k == "alias_declaration" else (fld(n, "declarator"), fld(n, "type"))
                if nm is not None and tgt is not None and kind(nm) == "type_identifier":
                    aliases.append((txt(src, nm), re.sub(r"\b(?:const|volatile|struct|class)\b|[*&\s]", "", txt(src, tgt)).split("::")[-1]))
            elif k in ("preproc_def", "preproc_function_def"):
                nm, body = fld(n, "name"), fld(n, "value")
                words = set(re.findall(r"[A-Za-z_]\w*", MACRO_NOISE_RE.sub(" ", txt(src, body)))) if body is not None else set()
                if nm is not None:
                    words.add(txt(src, nm))
                for w in sorted(words & S2D_NAMES):
                    occ.append(Occ(w, "other", n, n, None, "#define " + (txt(src, nm) if nm is not None else "?"),
                                   line_of(n), None, "macro"))
    _S2D[h] = {"occ": occ, "fdefs": fdefs, "aliases": aliases}
    return _S2D[h]


class Index:
    """The S2d facts of a whole tree, by name."""

    def __init__(self, files):
        self.files = files
        self.src = files
        self.by_name, self.defs, self.def_by_func, self.alias_of = {}, {}, {}, {}
        for rel in sorted(files):
            fa = s2d_facts(files[rel])
            for alias, target in fa["aliases"]:
                self.alias_of.setdefault(alias, set()).add(target)
            for o in fa["occ"]:
                self.by_name.setdefault(o.name, []).append((rel, o))
            for name, func, node in fa["fdefs"]:
                self.defs.setdefault(name, []).append((rel, func, node))
                self.def_by_func.setdefault((rel, func), []).append(node)

    def resolve(self, name):
        """Every class name `name` can stand for through `using X = Y;` / `typedef Y X;` chains (the name itself included)."""
        seen, todo = {name}, [name]
        while todo:
            for t in self.alias_of.get(todo.pop(), ()):
                if t not in seen:
                    seen.add(t)
                    todo.append(t)
        return seen

    def occ(self, name, roles=None):
        return [(r, o) for r, o in self.by_name.get(name, []) if roles is None or o.role in roles]

    def fdefs(self, name):
        return self.defs.get(name, [])


def has_name(ix, name, role):
    return any(o.role == role for _, o in ix.occ(name))


class S2dOut:
    """Findings of one S2d pass, with the ordinal rule of the other clauses (a key is `base#ordinal`)."""

    def __init__(self):
        self.viols, self.dormant, self.seen, self.active = [], [], {}, set()

    def add(self, code, rel, line, func, name, base, msg):
        i = self.seen.get(base, 0)
        self.seen[base] = i + 1
        self.viols.append(V(code, "%s#%d" % (base, i), rel, line, func, name, msg))

    def latch(self, clause, ix, name, shapes, why):
        """Clause `clause` is keyed on `name`, which appears in a shape its matcher misses: fail loudly, never pass forever."""
        for rel, o in ix.occ(name):
            if o.role == "other":
                self.add("X-LATCH", rel, o.line, o.func, name, "%s::%s::latch:%s:%s" % (rel, o.func, clause, name),
                         "clause (%s) is keyed on %s, which appears here as something other than %s (%s); the clause "
                         "matches nothing, so it would stay dormant for ever: respell it, or extend the matcher" % (
                             clause, name, shapes, why))


def body_of(node):
    return fld(node, "body")


def calls_in_node(src, n, last=None):
    """call_expression nodes under `n` whose callee ends in `last` (any when None)."""
    out = []
    for c in walk(n):
        if kind(c) == "call_expression":
            f = fld(c, "function")
            if f is not None and (last is None or callee_last(txt(src, f)) == last):
                out.append(c)
    return out


def in_span(inner, outer):
    return sb(outer) <= sb(inner) and eb(inner) <= eb(outer)


def clause_i(ix, out):
    """(i): once L0's accessor is defined, the pre-L0 retry and busy branch must be gone. A definition is a
    function_definition node whose declarator names the accessor; a call or a `;` declaration is not one."""
    defined = has_name(ix, I_ACCESSOR, "def")
    out.latch("i", ix, I_ACCESSOR, "a call, a declaration or a function definition", "an alias, a macro or a lambda hides it")
    for nm in I_RETIRED:
        out.latch("i", ix, nm, "a call, a declaration or a function definition", "a static lambda, a macro or an alias hides it")
    if not defined:
        out.dormant.append("DORMANT i: subject %s (a function definition) absent" % I_ACCESSOR)
        return
    out.active.add("i")
    for nm in I_RETIRED:
        for rel, o in ix.occ(nm, ("def",)):
            out.add("I-RETRY", rel, o.line, o.func, nm, "%s::%s::retired-function:%s" % (rel, o.func, nm),
                    "%s is defined while %s is defined: L0 has landed, so the pre-L0 path it served must be deleted" % (nm, I_ACCESSOR))


def clause_k(ix, out):
    """(k): once §B's COMPLETE-mode reap sits in the transaction, the pre-§B reclaim-query interim must be gone."""
    reaped = False
    for nm, why in ((K_TXN, "a lambda, a macro or an alias hides the transaction"), (K_REAP, "a macro, a lambda or an alias hides the reap")):
        out.latch("k", ix, nm, "a call, a declaration or a function definition", why)
    for rel, o in ix.occ(K_MODE):
        p, top = (None, None) if o.form == "macro" else (parent(o.node), o.node)
        while p is not None and (kind(p) == "parenthesized_expression" or (kind(p) == "conditional_expression" and not same(
                fld(p, "condition"), top))):
            top, p = p, parent(p)
        if o.form == "macro" or (p is not None and kind(p) in ("init_declarator", "assignment_expression") and same(
                fld(p, "value" if kind(p) == "init_declarator" else "right"), top)):
            out.add("X-LATCH", rel, o.line, o.func, K_MODE, "%s::%s::latch:k:%s:alias" % (rel, o.func, K_MODE),
                    "clause (k) is keyed on a call of %s whose arguments name %s; %s is aliased here (%s), so the reap's mode "
                    "would be read through a name the clause does not follow: respell it, or extend the matcher" % (
                        K_REAP, K_MODE, K_MODE, "a macro" if o.form == "macro" else "a constant or variable initialiser"))
    for rel, func, node in ix.fdefs(K_TXN):
        src = ix.files[rel]
        calls = calls_in_node(src, body_of(node), K_REAP)
        for n in walk(body_of(node)):
            if kind(n) in ("identifier", "field_identifier") and txt(src, n) == K_MODE:
                if any(in_span(n, fld(c, "arguments")) for c in calls if fld(c, "arguments") is not None):
                    reaped = True
                else:
                    out.add("X-LATCH", rel, line_of(n), func, K_MODE, "%s::%s::latch:k:%s" % (rel, func, K_MODE),
                            "clause (k) is keyed on a call of %s whose arguments name %s, inside %s; %s appears here outside "
                            "any such call, so the clause would stay dormant: respell it, or extend the matcher" % (
                                K_REAP, K_MODE, K_TXN, K_MODE))
    if not reaped:
        out.dormant.append("DORMANT k: subject %s(... %s ...) inside %s absent" % (K_REAP, K_MODE, K_TXN))
        return
    out.active.add("k")
    for rel, o in ix.occ(K_INTERIM, ("def",)):
        out.add("K-INTERIM", rel, o.line, o.func, K_INTERIM, "%s::%s::interim-after-reap:%s" % (rel, o.func, K_INTERIM),
                "%s is defined while %s reaps in COMPLETE mode: the arithmetic refusal cannot outlive the path that removes "
                "its cause" % (K_INTERIM, K_TXN))


JUMP_KINDS = ("return_statement", "break_statement", "continue_statement", "goto_statement", "throw_statement")


def reads_count(src, n, tainted):
    for x in walk(n):
        if kind(x) in ("identifier", "field_identifier") and (txt(src, x) == L_COUNT or txt(src, x) in tainted):
            return True
    return False


def count_controlled_frees(src, fnode):
    """call_expression nodes that free or release storage and whose execution depends on a condition that reads the count,
    directly or through a local assigned from an expression that does (the laundering form), to a fixpoint. A branch that
    leaves the function or loop also controls the statements after it in its block."""
    body = body_of(fnode)
    pairs = []
    for x in walk(body):
        if kind(x) == "init_declarator" and fld(x, "declarator") is not None and fld(x, "value") is not None:
            pairs.append((declared_name(src, fld(x, "declarator")), fld(x, "value")))
        elif kind(x) == "assignment_expression" and fld(x, "left") is not None and kind(fld(x, "left")) == "identifier":
            pairs.append((txt(src, fld(x, "left")), fld(x, "right")))
    tainted, changed = set(), True
    while changed:
        changed = False
        for name, val in pairs:
            if name and name not in tainted and val is not None and reads_count(src, val, tainted):
                tainted.add(name)
                changed = True
    regions = []
    for x in walk(body):
        k = kind(x)
        cond = None
        if k in ("if_statement", "while_statement", "do_statement", "switch_statement", "conditional_expression"):
            cond = fld(x, "condition")
        elif k == "for_statement":
            cond = fld(x, "condition")
        elif k == "for_range_loop":
            cond = fld(x, "right")
        elif k == "binary_expression" and txt(src, fld(x, "operator")) in ("&&", "||"):
            if reads_count(src, fld(x, "left"), tainted):
                regions.append(fld(x, "right"))
            continue
        if cond is None or not reads_count(src, cond, tainted):
            continue
        if k == "if_statement":
            branches = [b for b in (fld(x, "consequence"), fld(x, "alternative")) if b is not None]
            regions += branches
            p = parent(x)
            if p is not None and kind(p) == "compound_statement" and any(kind(y) in JUMP_KINDS for b in branches for y in walk(b)):
                regions += [c for c in kids(p) if sb(c) > sb(x) and _a(c, "is_named")]
        elif k == "conditional_expression":
            regions += [b for b in (fld(x, "consequence"), fld(x, "alternative")) if b is not None]
        else:
            b = fld(x, "body")
            if b is not None:
                regions.append(b)
    found, seen = [], set()
    for r in regions:
        for c in calls_in_node(src, r):
            last = callee_last(txt(src, fld(c, "function")))
            if (last in L_FREES or L_TLSF_FREE.search(last)) and sb(c) not in seen:
                seen.add(sb(c))
                found.append((c, last))
    return found


def first_statement(node):
    for c in kids(body_of(node)):
        if _a(c, "is_named") and kind(c) != "comment":
            return c
    return None


def clause_l(ix, out):
    """(l): the owner count never consumes a shared handle. Callers outside mem-handle.* are an allowlist (L-CALLER); no
    freeing call is controlled by a condition that depends on the count (L-FREE); replace_within's first statement is
    replace_within_count_guard(old) (L-GUARD)."""
    out.latch("l", ix, L_COUNT, "a call", "a pointer-to-member, a macro or an alias hides it")
    out.latch("l", ix, L_REPLACE, "a call, a declaration or a function definition", "an alias, a macro or a lambda hides it")
    out.dormant.append("TODO l: the count's callers are allowlist entries (code L-CALLER), added as each lands; the design names "
                       "none of its five count-caller functions yet")
    if not ix.occ(L_COUNT, ("call", "decl", "def")):
        out.dormant.append("DORMANT l: subject %s absent" % L_COUNT)
    else:
        out.active.add("l")
        funcs = {}
        for rel, o in ix.occ(L_COUNT, ("call",)):
            if rel in L_MEM_HANDLE:
                continue
            out.add("L-CALLER", rel, o.line, o.func, L_COUNT, "%s::%s::count-caller:%s" % (rel, o.func, L_COUNT),
                    "%s is called outside mem-handle.* from a function that is not on the allowlist of the five count "
                    "consumers" % L_COUNT)
            funcs[(rel, o.func)] = True
        for (rel, func) in sorted(funcs):
            for node in ix.def_by_func.get((rel, re.sub(r"^lambda in ", "", func)), []):
                for c, last in count_controlled_frees(ix.files[rel], node):
                    out.add("L-FREE", rel, line_of(c), func, last, "%s::%s::count-controlled-free:%s:%s" % (
                        rel, func, last, text_hash(txt(ix.files[rel], c))),
                            "%s is called under a condition that depends on %s: the count may select a path that keeps the "
                            "handle's ownership intact, never one that frees or releases storage outside it" % (last, L_COUNT))
    defs = ix.fdefs(L_REPLACE)
    if not defs:
        out.dormant.append("DORMANT l: subject %s (a function definition) absent" % L_REPLACE)
        return
    out.active.add("l")
    for rel, func, node in defs:
        st = first_statement(node)
        src = ix.files[rel]
        ok = False
        if st is not None and kind(st) == "expression_statement":
            for c in kids(st):
                if kind(c) == "call_expression" and callee_last(txt(src, fld(c, "function"))) == L_GUARD \
                        and re.sub(r"\s+", "", txt(src, fld(c, "arguments"))) == "(old)":
                    ok = True
        if not ok:
            out.add("L-GUARD", rel, line_of(node), func, L_REPLACE, "%s::%s::first-statement:%s" % (rel, func, L_GUARD),
                    "%s must call %s(old) as its first statement, so the one count-selected consumer re-checks before "
                    "it consumes anything" % (L_REPLACE, L_GUARD))


def clause_j(ix, out):
    """(j): A's fit and W's term read per-model sources only (J-SOURCE, dormant until they exist); nothing outside the
    late stage's classifier calls zone_is_onednn_reorder_eligible (J-DISPATCH)."""
    for nm in J_FIT:
        out.latch("j", ix, nm, "a call, a declaration or a function definition", "an alias, a macro or a lambda hides it")
    out.latch("j", ix, J_ELIGIBLE, "a call, a declaration or a function definition",
              "a macro, an alias or a taken address hides a call from J-DISPATCH")
    out.dormant.append("TODO j: only %s are covered; the design names no fit function for the reserve's target nor the "
                       "candidate/selector bit reads, so those are not checked yet" % " / ".join(J_FIT))
    fits = [(rel, func, node) for nm in J_FIT for rel, func, node in ix.fdefs(nm)]
    if not fits:
        out.dormant.append("DORMANT j: subject %s (a function definition) absent" % " / ".join(J_FIT))
    else:
        out.active.add("j")
        for rel, func, node in fits:
            src = ix.files[rel]
            for x in walk(body_of(node)):
                if kind(x) in ("identifier", "field_identifier") and J_BANNED.fullmatch(txt(src, x)):
                    out.add("J-SOURCE", rel, line_of(x), func, txt(src, x), "%s::%s::per-model-source:%s" % (rel, func, txt(src, x)),
                            "%s reads %s, which is process- or device-global: A's fit and W's term read per-model sources only"
                            % (func, txt(src, x)))
    for rel, o in ix.occ(J_ELIGIBLE, ("call",)):
        out.add("J-DISPATCH", rel, o.line, o.func, J_ELIGIBLE, "%s::%s::eligibility-call:%s" % (rel, o.func, J_ELIGIBLE),
                "%s is called outside the late stage's classifier; the classification is made once, at the plan, and a "
                "dispatch path reads its stored bits" % J_ELIGIBLE)


# Clause (m): the SCRATCH floor list. The appendix's census table is committed as appendix-rows.json (generated by
# gen-appendix-rows.py from the design's table); SCRATCH_FLOOR_CONSUMERS is the gate's own table of (row, owner ticket).
M_TABLES = {
    "floor": {17: "beni", 21: "beni", 31: "beni", 33: "beni", 34: "beni", 61: "beni", 68: "beni", 69: "beni", 74: "beni",
              81: "beni", 90: "beni", 106: "beni", 59: "pqmm", 60: "pqmm", 66: "pqmm", 67: "pqmm", 91: "pqmm", 51: "zhcn",
              9: "6lfq", 103: "6lfq", 101: "pending (lead)", 138: "pending (lead)"},
    "covered": {62: 61},                      # row -> the listed row whose pool peak covers it
    "unreachable": {129: "reserve_compute_arena draws no SCRATCH under an arena"},
}
S2D_DATA = {"dir": None}


def appendix_rows():
    """(rows, problem): row number -> {function, zone, core, code, term}, from appendix-rows.json beside the allowlist. `rows`
    is None, with the problem named, when the file is missing, is not JSON, or has a row the clause cannot read."""
    d = S2D_DATA["dir"] or Path(__file__).resolve().parent / "sycl-alloc-zone-contract"
    p = Path(d) / "appendix-rows.json"
    if not p.exists():
        return None, "appendix-rows.json is missing"
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        return None, "appendix-rows.json is not readable JSON (%s)" % exc
    rows = doc.get("rows") if isinstance(doc, dict) else None
    if not isinstance(rows, list):
        return None, "appendix-rows.json has no `rows` list"
    out = {}
    for i, r in enumerate(rows):
        try:
            if not isinstance(r, dict) or any(not isinstance(r[k], str) for k in ("function", "zone", "core", "code", "term")):
                raise KeyError
            out[int(r["row"])] = r
        except (KeyError, TypeError, ValueError):
            return None, "appendix-rows.json row #%d is malformed (needs an integer row and string function, zone, core, code, term)" % i
    return out, None


def clause_m(ix, out):
    """(m): every appendix row whose zone today or core zone is SCRATCH is listed, covered by a named peak, or excluded by
    class (core-planned, D, REF, unreachable); a listed row whose zone is not SCRATCH fails; and the floor tie."""
    rows, problem = appendix_rows()
    if rows is None:
        out.add("M-DATA", "appendix-rows.json", 0, "<table>", "", "appendix::%s" % ("missing" if "missing" in problem else "malformed"),
                "%s: clause (m) reads the census table's zone columns from it" % problem)
        return
    floor, covered, unreach = M_TABLES["floor"], M_TABLES["covered"], M_TABLES["unreachable"]
    for r in sorted(rows):
        row = rows[r]
        scratch = "SCRATCH" in row["zone"] or "SCRATCH" in row["core"]
        if not scratch:
            continue
        if r in floor or (r in covered and covered[r] in floor):
            continue
        if row["term"].startswith("core") or row["code"] in ("D", "REF") or r in unreach:
            continue
        out.add("M-SCRATCH", "appendix-rows.json", 0, "row %d" % r, str(r), "appendix::row:%d:scratch" % r,
                "appendix row %d (%s) draws SCRATCH and is neither on the floor list, covered by a named peak, nor excluded by "
                "class" % (r, row["function"]))
    for r in sorted(floor):
        if r not in rows or not ("SCRATCH" in rows[r]["zone"] or "SCRATCH" in rows[r]["core"]):
            out.add("M-STALE", "appendix-rows.json", 0, "row %d" % r, str(r), "appendix::row:%d:stale-floor" % r,
                    "floor row %d is not a SCRATCH row of the appendix: the list cannot carry a stale number" % r)
    for r, peak in sorted(covered.items()):
        if peak not in floor:
            out.add("M-STALE", "appendix-rows.json", 0, "row %d" % r, str(r), "appendix::row:%d:stale-peak" % r,
                    "row %d is covered by row %d's peak, which is not on the floor list" % (r, peak))
    fn = [(rel, func, node) for rel, func, node in ix.fdefs(M_FLOOR_FN) if rel == "unified-cache.cpp"]
    has_floor = any(M_FLOOR_ENV in txt(ix.files[rel], body_of(node)) and re.search(
        r'getenv\s*\(\s*"%s"\s*\)' % M_FLOOR_ENV, txt(ix.files[rel], body_of(node))) for rel, _, node in fn)
    if floor and not has_floor:
        out.add("M-FLOOR", "unified-cache.cpp", 0, M_FLOOR_FN, M_FLOOR_FN, "appendix::floor-tie",
                "%d SCRATCH consumers still draw without a term, but %s does not apply the %s floor" % (len(floor), M_FLOOR_FN, M_FLOOR_ENV))
    if not floor and has_floor:
        out.add("M-FLOOR", "unified-cache.cpp", 0, M_FLOOR_FN, M_FLOOR_FN, "appendix::floor-tie",
                "the SCRATCH floor list is empty, so %s must no longer apply the %s floor" % (M_FLOOR_FN, M_FLOOR_ENV))


def class_of(func):
    """The class a function name carries (`DnnlGemmWrapper` for `DnnlGemmWrapper::row_gemm` or `<member of DnnlGemmWrapper>`)."""
    f = re.sub(r"^lambda in ", "", func)
    m = re.match(r"<member of (\w+)>$", f)
    if m:
        return m.group(1)
    return f.split("::")[-2] if "::" in f else None


def encl_class(src, node):
    """The name of the class or struct whose body holds `node`, else None (a prototype in a class body has no enclosing
    function, so `enclosing()` cannot name its class)."""
    p = parent(node)
    while p is not None:
        if kind(p) in ("class_specifier", "struct_specifier"):
            nm = fld(p, "name")
            return txt(src, nm).split("::")[-1].strip() if nm is not None else None
        p = parent(p)
    return None


def n_listed(ix, src, o):
    """The listed name an occurrence spells, as (class or None, name), or None: `Class::name` anywhere (the scope may be an
    alias of the class: `using W = DnnlGemmWrapper; W::row_gemm(...)`), a bare member name only inside that class, the two
    unique names anywhere."""
    if o.name in N_UNIQUE:
        return (None, o.name)
    if o.scope is not None:
        for c in sorted(ix.resolve(o.scope)):
            if (c, o.name) in N_MEMBERS:
                return (c, o.name)
        return None
    if kind(o.top) == "identifier" or o.role in ("def", "decl"):
        c = encl_class(src, o.node) or class_of(o.func)
        if c is not None and (c, o.name) in N_MEMBERS:
            return (c, o.name)
    return None


# a functional cast `bool(f())` is a call whose callee is a type name: the builtin types and the fixed-width integer aliases
V_FUNCTIONAL_CAST_RE = re.compile(r"(?:unsigned|signed)(?:char|int|long|short)?|(?:long|short)+(?:int|double)?|char|int|bool|float|double|"
                                  r"u?int(?:8|16|32|64)_t|u?intptr_t|size_t|ssize_t|ptrdiff_t")


def value_use(src, call):
    """'void-cast' for `(void) f()` and `static_cast<void>(f())`, 'discard' for a call whose value nothing reads, else
    'used'. The value of `ok && f()`, `ok || f()`, `c ? f() : x` and the right operand of a comma is the value of the whole
    expression, and so is the value under a `!`, a binary operator, a cast to another type or `static_cast<T>(...)`; those
    operands are followed up to the expression that consumes (or drops) it. A call that is a for-loop's initializer or
    update expression is dropped too, and so is one assigned to `std::ignore`."""
    n, p = call, parent(call)
    while p is not None:
        k = kind(p)
        if k == "cast_expression":
            t = fld(p, "type")
            if t is not None and norm(txt(src, t)) == "void":
                return "void-cast"
            n, p = p, parent(p)
            continue
        if k == "argument_list" and parent(p) is not None and kind(parent(p)) == "call_expression":
            f = fld(parent(p), "function")
            ft = re.sub(r"\s+", "", txt(src, f)) if f is not None else ""
            if ft == "void" or re.match(r"(?:static|reinterpret|const|dynamic)_cast<void>", ft):
                return "void-cast"
            if re.match(r"(?:static|reinterpret|const|dynamic)_cast<", ft) or V_FUNCTIONAL_CAST_RE.fullmatch(ft):
                n, p = parent(p), parent(parent(p))
                continue
        if k in ("parenthesized_expression", "unary_expression") or \
                (k == "binary_expression" and (txt(src, fld(p, "operator")) not in ("&&", "||") or same(fld(p, "right"), n))) or \
                (k == "conditional_expression" and not same(fld(p, "condition"), n)) or \
                (k == "comma_expression" and not same(fld(p, "left"), n)):
            n, p = p, parent(p)
            continue
        break
    if p is None:
        return "used"
    if kind(p) == "assignment_expression" and same(fld(p, "right"), n) and fld(p, "left") is not None and \
            re.sub(r"\s+", "", txt(src, fld(p, "left"))).endswith("std::ignore"):
        return "discard"
    if kind(p) in ("comma_expression", "expression_statement"):
        return "discard"
    if kind(p) == "for_statement" and (same(fld(p, "update"), n) or same(fld(p, "initializer"), n)):
        return "discard"
    return "used"


def has_nodiscard(src, top):
    p = parent(top)
    while p is not None and kind(p) != "function_declarator":
        p = parent(p)
    q = parent(p) if p is not None else None
    while q is not None and kind(q) in ("pointer_declarator", "reference_declarator", "parenthesized_declarator"):
        q = parent(q)
    return q is not None and any(kind(c) == "attribute_declaration" and "nodiscard" in txt(src, c) for c in kids(q))


def clause_n(ixt, out):
    """(n): a declined result is consumed. Each call of a listed name is the initializer of a declaration, the right side of
    an assignment, a return operand or a condition (here: anything but a discarded value); a call that is its own
    expression statement, or is cast to void, fails (N-VOID); each listed declaration carries [[nodiscard]] (N-NODISCARD)."""
    out.dormant.append("TODO n: an alias chain whose intermediate alias is declared in a file that spells no listed name or class is "
                       "not followed")
    for nm in sorted(N_LAST):
        for rel, o in ixt.occ(nm):
            lst = n_listed(ixt, ixt.files[rel], o)
            if o.role == "other":
                if o.form == "macro" or lst is not None:
                    out.add("X-LATCH", rel, o.line, o.func, nm, "%s::%s::latch:n:%s" % (rel, o.func, nm),
                            "clause (n) is keyed on calls of %s, which appears here as something other than a call, a declaration "
                            "or a function definition (%s): a call through it would not be checked, so respell it, or extend the "
                            "matcher" % (nm, "a macro" if o.form == "macro" else "a pointer to member or another reference"))
                continue
            if lst is None:
                continue
            out.active.add("n")
            cname = ("%s::%s" % lst) if lst[0] else lst[1]
            src = ixt.files[rel]
            if o.role == "call":
                use = value_use(src, o.call)
                if use != "used":
                    out.add("N-VOID", rel, o.line, o.func, cname, "%s::%s::%s:%s:%s" % (rel, o.func, use, cname, text_hash(txt(src, o.call))),
                            "the result of %s is %s: a declined result is consumed (assigned, returned or tested)" % (
                                cname, "cast to void" if use == "void-cast" else "discarded"))
            elif o.scope is None and not has_nodiscard(src, o.top):
                out.add("N-NODISCARD", rel, o.line, o.func, cname, "%s::%s::nodiscard:%s" % (rel, o.func, cname),
                        "%s is declared without [[nodiscard]], so a caller can drop the declined result silently" % cname)


# Clause (o): each C-term consumer submits on its census row's queue. A row pins, at the submission that consumes the
# acquired bytes, the call's queue argument (by position and exact text, whitespace normalised), or, for a callee that
# takes no queue, the declaration of its queue inside the callee. `consumers` entries are (callee, an identifier that must
# appear among the arguments or None, queue argument index, queue text).
O_ROWS = (
    {"term": "onednn_graph_scratch", "token": O_EXECUTE, "file": "fattn-onednn.cpp", "func": "ggml_sycl_flash_attn_ext_onednn",
     "consumers": ((O_EXECUTE, None, 1, "dnnl_stream"),),
     "decls": (("dnnl_stream", "ctx.stream_dnnl(stream)"), ("stream", "ctx.stream()")), "callee_decls": ()},
    {"term": "onednn_pp_a", "token": "acquire_onednn_pp_scratch", "file": "ggml-sycl.cpp", "func": "ggml_sycl_op_mul_mat_sycl",
     "consumers": (("to_fp16_sycl", "dst_f16", 3, "stream"),
                   ("dequantize_row_q8_0_soa_to_fp16_rowmajor", "dst_f16", 4, "stream"),
                   ("dequantize_row_q8_0_coalesced_to_fp16_rowmajor", "dst_f16", 4, "stream"),
                   ("row_gemm", None, 10, "stream")), "decls": (), "callee_decls": ()},
    {"term": "onednn_pp_a", "token": "acquire_onednn_pp_scratch", "file": "ggml-sycl.cpp", "func": "ggml_sycl_mul_mat",
     "consumers": (("dequant_weights_to_fp16", "weights_scratch", 3, "ctx.stream()"),
                   ("f32_to_fp16", "activations_scratch", 3, "ctx.stream()"),
                   ("store", "weights_scratch", 3, "*ctx.stream()"),
                   ("row_gemm", "activations_scratch", 10, "ctx.stream()")), "decls": (), "callee_decls": ()},
    {"term": "set_rows_stage", "token": "ggml_sycl_set_rows_stage_ptr", "file": "set_rows.cpp", "func": "ggml_sycl_op_set_rows",
     "consumers": (("set_rows_validate_indices", "plan", 0, "*ctx.stream(plan.owner_device, 0)"),
                   ("ggml_sycl_fattn_xmx_update_packed_k_from_set_rows", "plan", 6, "ctx.stream(plan.owner_device, 0)")),
     "decls": (), "callee_decls": (("set_rows_sycl", (("stream", "ctx.stream(device, 0)"), ("device", "plan.owner_device"))),)},
)


def arg_nodes(call):
    a = fld(call, "arguments")
    return [c for c in kids(a) if _a(c, "is_named") and kind(c) != "comment"] if a is not None else []


def squash(s):
    return re.sub(r"\s+", "", norm(s))


def var_inits(src, body, var):
    """The initializer texts (whitespace removed) of every declaration `T var = ...;` under `body`, read from the text: a
    macro with no `;` before the declaration (`GGML_TENSOR_BINARY_OP_LOCALS`) makes the parse swallow the declared name."""
    text = norm(txt(src, body))
    return [re.sub(r"\s+", "", m.group(1)) for m in re.finditer(r"[\w>*&]\s+%s\s*=\s*([^;{}]*);" % re.escape(var), text)]


def clause_o(ix, out):
    """(o): every call of an acquire token sits in a function that has a census row, and each consuming call of that row
    keeps the row's queue. A presence test could not fail (`ggml_sycl_mul_mat` holds dozens of `ctx.stream()`), so the
    pin is on the consuming call's own argument."""
    out.dormant.append("TODO o: a macro or alias that hides %s is not followed (the name is a qualified call, not an identifier)" % O_EXECUTE)
    for tok in O_ACQUIRE:
        out.latch("o", ix, tok, "a call, a declaration or a function definition", "a macro, a lambda, a pointer or an alias hides it")
    tokens = set(O_ACQUIRE) | {O_EXECUTE}
    rowkey = {(r["token"], r["file"], r["func"]) for r in O_ROWS}
    for tok in sorted(tokens):
        for rel, o in ix.occ(tok, ("call",)):
            fn = re.sub(r"^lambda in ", "", o.func)
            if (tok, rel, fn) not in rowkey:
                out.add("O-NOROW", rel, o.line, fn, tok, "%s::%s::o-acquire:%s" % (rel, fn, tok),
                        "%s is called in %s, which has no census row (clause (o)): add its row, with the queue its "
                        "consumers submit on" % (tok, fn))
    for row in O_ROWS:
        defs = [(f, n) for r, f, n in ix.fdefs(row["func"]) if r == row["file"]]
        if not defs:
            out.add("O-ROW", row["file"], 0, row["func"], row["term"], "%s::%s::o-row:%s:function-missing" % (row["file"], row["func"], row["term"]),
                    "census row %s names %s in %s, which has no such function" % (row["term"], row["func"], row["file"]))
            continue
        src = ix.files[row["file"]]
        for func, node in defs:
            body = body_of(node)
            calls = calls_in_node(src, body)
            if not any(callee_last(txt(src, fld(c, "function"))) == row["token"] or
                       re.sub(r"\s+", "", txt(src, fld(c, "function"))) == row["token"] for c in calls):
                out.add("O-ROW", row["file"], line_of(node), func, row["term"], "%s::%s::o-row:%s:acquire-missing" % (row["file"], func, row["term"]),
                        "census row %s: %s no longer calls %s" % (row["term"], func, row["token"]))
            for callee, has, qi, qtext in row["consumers"]:
                found = []
                for c in calls:
                    f = re.sub(r"\s+", "", txt(src, fld(c, "function")))
                    if (f == callee if "::" in callee else callee_last(f) == callee) and \
                            (has is None or re.search(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(has), txt(src, fld(c, "arguments")))):
                        found.append(c)
                if not found:
                    out.add("O-ROW", row["file"], line_of(node), func, callee, "%s::%s::o-row:%s:consumer-missing:%s" % (row["file"], func, row["term"], callee),
                            "census row %s: no call of %s%s remains in %s" % (row["term"], callee, " taking " + has if has else "", func))
                for c in found:
                    args = arg_nodes(c)
                    got = squash(txt(src, args[qi])) if qi < len(args) else "<no argument %d>" % qi
                    if got != squash(qtext):
                        out.add("O-QUEUE", row["file"], line_of(c), func, callee, "%s::%s::o-queue:%s:%s:%s" % (
                            row["file"], func, row["term"], callee, text_hash(txt(src, c))),
                                "census row %s: %s consumes the acquired bytes on queue `%s`, the row pins `%s`" % (
                                    row["term"], callee, got, qtext))
            for var, init in row["decls"]:
                inits = var_inits(src, body, var)
                if init.replace(" ", "") not in inits:
                    out.add("O-QUEUE", row["file"], line_of(node), func, var, "%s::%s::o-decl:%s:%s" % (row["file"], func, row["term"], var),
                            "census row %s: `%s` is no longer declared as `%s` in %s (declared as %s)" % (
                                row["term"], var, init, func, inits or "nothing"))
            for callee, decls in row["callee_decls"]:
                cdefs = [n for r, f, n in ix.fdefs(callee) if r == row["file"]]
                ok = any(all(init.replace(" ", "") in var_inits(src, body_of(n), var) for var, init in decls) for n in cdefs)
                if not ok:
                    out.add("O-QUEUE", row["file"], line_of(node), func, callee, "%s::%s::o-callee-decl:%s:%s" % (row["file"], func, row["term"], callee),
                            "census row %s: %s takes no queue, so the row pins its own declarations %s inside it, and none of "
                            "its definitions holds them" % (row["term"], callee, ", ".join("%s = %s" % d for d in decls)))


def clause_z9(ix, out):
    """Witness 9: every model-shaped exact `*_bytes()` function is called by its allocation sites and by the zone sizing;
    dormant while the function is not defined."""
    out.dormant.append("TODO 9: the design names no zone-sizing function, so any call of the bytes function in %s from a function "
                       "that is not one of its allocation sites counts as the sizing" % " or ".join(Z9_SIZING_FILES))
    for fn, sites in Z9_SITES:
        out.latch("9", ix, fn, "a call or a function definition", "an alias, a macro or a lambda hides it")
        if not has_name(ix, fn, "def"):
            out.dormant.append("DORMANT 9: subject %s (a function definition) absent" % fn)
            continue
        out.active.add("9")
        for rel, site in sites:
            hits = [(f, n) for r, f, n in ix.fdefs(site.split("::")[-1]) if r == rel]
            if not hits:
                out.add("Z9-SITE", rel, 0, site, fn, "%s::%s::bytes-site:%s:missing" % (rel, site, fn),
                        "%s is an allocation site of %s but %s defines no function of that name" % (site, fn, rel))
            elif not any(calls_in_node(ix.files[rel], body_of(n), fn) for _, n in hits):
                out.add("Z9-SITE", rel, line_of(hits[0][1]), site, fn, "%s::%s::bytes-site:%s" % (rel, site, fn),
                        "%s no longer calls %s: the site and the zone sizing must share the one function, or the plan "
                        "reserves what the site does not draw" % (site, fn))
        site_fns = {fn} | {site.split("::")[-1] for _, site in sites}
        sized = [(rel, o) for rel, o in ix.occ(fn, ("call",)) if rel in Z9_SIZING_FILES and fn_name(o.func) not in site_fns]
        if not sized:
            out.add("Z9-SIZING", Z9_SIZING_FILES[0], 0, "<zone sizing>", fn, "sizing::bytes-fn:%s" % fn,
                    "%s is called by no sizing function of %s: a call from one of its own allocation sites is not the zone "
                    "sizing, which must charge what the sites draw through the same function" % (fn, " or ".join(Z9_SIZING_FILES)))


def strip_comments_b(b):
    """`b` with every comment blanked (newlines and offsets kept); string and character literals stay."""
    return P_COMMENT_RE.sub(lambda m: m.group(0) if m.group(0)[:1] in (b'"', b"'") else re.sub(rb"[^\n]", b" ", m.group(0)), b)


def clean_text(src, node):
    return strip_comments_b(src[sb(node):eb(node)]).decode("utf-8", "replace")


def fn_name(func):
    """The unqualified function a finding sits in (`lambda in X` is X, `C::m` is m)."""
    return re.sub(r"^lambda in ", "", func).split("::")[-1]


def p_dormant(out, block, subject, shape="a function definition"):
    out.dormant.append("DORMANT p-%s: subject %s (%s) absent" % (block, subject, shape))


def p_routed_vars(src, call):
    """Names of the locals, in the function holding `call`, that are initialised from a call of the routing function (the
    FORCE arm of the design keeps its routing call in `const bool routed = ...;` and conditions the entry on `routed`)."""
    f = call
    while f is not None and kind(f) != "function_definition":
        f = parent(f)
    names = set()
    for x in (walk(f) if f is not None else ()):
        if kind(x) == "init_declarator" and fld(x, "value") is not None and \
                any(callee_last(txt(src, fld(c, "function"))) == P_DISPATCH for c in calls_in_node(src, fld(x, "value"))):
            names.add(declared_name(src, fld(x, "declarator")))
    return names


def p_dispatch_branch(src, call):
    """(ok, why): the call sits in the consequence of an `if` whose condition calls the routing function, or reads a local that
    was initialised from it, and tests none of the flags the routing function owns."""
    p, why = parent(call), "no enclosing branch whose condition calls %s" % P_DISPATCH
    viaVars = None
    while p is not None:
        if kind(p) == "if_statement":
            cond, cons = fld(p, "condition"), fld(p, "consequence")
            if cond is not None and cons is not None and in_span(call, cons):
                if viaVars is None:
                    viaVars = p_routed_vars(src, call)
                words = set(re.findall(r"[A-Za-z_]\w*", clean_text(src, cond)))
                if any(callee_last(txt(src, fld(c, "function"))) == P_DISPATCH for c in calls_in_node(src, cond)) or (viaVars & words):
                    bad = [f for f in P_FLAGS if re.search(r"(?<![\w])%s(?![\w])" % f, clean_text(src, cond))]
                    if not bad:
                        return True, ""
                    why = "its branch condition also tests %s" % ", ".join(bad)
        p = parent(p)
    return False, why


def p_callers(ix, out, code, tag, table):
    """Each (callee, allowed caller names) of `table`: a call of the callee from any other function is a finding."""
    for callee, allowed in table:
        for rel, o in ix.occ(callee, ("call",)):
            if fn_name(o.func) not in allowed:
                out.add(code, rel, o.line, o.func, callee, "%s::%s::%s:%s" % (rel, o.func, tag, callee),
                        "%s is called in %s; its callers are %s only" % (callee, o.func, ", ".join(allowed)))


def p_route(ix, out):
    out.dormant.append("TODO p-route: the charge side's walk helpers are unnamed in the design, so they are not pinned")
    out.latch("p-route", ix, P_ADMITS, "a call or a function definition", "an alias, a macro or a lambda hides it")
    out.latch("p-route", ix, P_DECLINED, "a call or a function definition", "an alias, a macro or a lambda hides it")
    for nm in (P_ROUTE_ENABLED, P_DISPATCH, P_ROUTED, P_ONEDNN, P_PLAN):
        out.latch("p-route", ix, nm, "a call or a function definition", "an alias, a macro or a lambda hides it")
    if not has_name(ix, P_ADMITS, "def"):
        p_dormant(out, "route", P_ADMITS)
        return
    out.active.add("p-route")
    fa = "fattn.cpp"
    adm = [(r, n) for r, _, n in ix.fdefs(P_ADMITS)]
    for rel, o in ix.occ(P_ONEDNN, ("call",)):
        if rel != fa:
            continue
        ok, why = p_dispatch_branch(ix.files[rel], o.call)
        if not ok:
            out.add("P-ROUTE", rel, o.line, o.func, P_ONEDNN, "%s::%s::p-route:onednn-call" % (rel, o.func),
                    "the call of %s in %s is not inside a branch the routing function governs (%s): the route's gates and "
                    "the plan are combined once, in %s" % (P_ONEDNN, o.func, why, P_ADMITS))
    for rel, o in ix.occ(P_PLAN, ("call",)):
        if rel == fa and not any(r == rel and in_span(o.call, body_of(n)) for r, n in adm):
            out.add("P-ROUTE", rel, o.line, o.func, P_PLAN, "%s::%s::p-route:plan-call" % (rel, o.func),
                    "%s is called in %s outside %s: the plan is read only through the one predicate" % (P_PLAN, o.func, P_ADMITS))
    for rel, func, node in ix.fdefs(P_VALUEFN):
        src = ix.files[rel]
        if calls_in_node(src, body_of(node), P_PLAN):
            out.add("P-ROUTE", rel, line_of(node), func, P_PLAN, "%s::%s::p-route:value-plan" % (rel, func),
                    "the value function %s calls %s directly; it calls %s with nullptr, nullptr" % (func, P_PLAN, P_ROUTED))
        calls = calls_in_node(src, body_of(node), P_ROUTED)
        bad = []
        for c in calls:
            args = [squash(txt(src, a)) for a in arg_nodes(c)]
            if not (len(args) >= 4 and args[-2:] == ["nullptr", "nullptr"]):
                bad.append(c)
        for c in bad or ([] if calls else [node]):
            out.add("P-ROUTE", rel, line_of(c), func, P_ROUTED, "%s::%s::p-route:value-predicate" % (rel, func),
                    "the value function %s must call %s, every time, with nullptr, nullptr as its last two (context and plan) "
                    "arguments%s" % (func, P_ROUTED, "" if calls else " (it makes no call)"))
    for rel, func, node in ix.fdefs(P_DISPATCH):
        src = ix.files[rel]
        dec = [sb(c) for c in calls_in_node(src, body_of(node), P_DECLINED)]
        routed = [sb(c) for c in calls_in_node(src, body_of(node), P_ROUTED)]
        if squash(clean_text(src, body_of(node))) != squash("{ %s }" % P_DISPATCH_BODY):
            out.add("P-ROUTE", rel, line_of(node), func, P_DISPATCH, "%s::%s::p-route:dispatch-body" % (rel, func),
                    "the body of %s must be exactly the pinned text (whitespace and comments aside): `%s`" % (P_DISPATCH, P_DISPATCH_BODY))
        if not dec or (routed and min(routed) < min(dec)):
            out.add("P-ROUTE", rel, line_of(node), func, P_DECLINED, "%s::%s::p-route:%s" % (
                rel, func, "decline-order" if dec else "decline-read"),
                    "%s must call %s before it calls %s: every decline is read in the routing condition, before the plan, "
                    "and %s" % (func, P_DECLINED, P_ROUTED, "the first call of the predicate comes first here" if dec else
                                "no read of it is made here"))
    p_callers(ix, out, "P-ROUTE", "p-route:caller", ((P_ADMITS, (P_ROUTED, P_SHAPE_SUPPORTED)), (P_ROUTE_ENABLED, (P_ADMITS, P_DISPATCH))))


P_HATCH_BODIES = ("{if(!%s){returnfalse;}}", "if(!%s){returnfalse;}", "{if(!%s)returnfalse;}", "if(!%s)returnfalse;")


def p_blank_hatch(src, body):
    """`body`'s text with the route's D=512 hatch blanked, and nothing else. The route latches the hatch once, in a function-local
    `static const bool V = ggml_sycl_fa_onednn_d512_enabled();`. The exempt text is then an `if` whose whole condition is
    `<head dim> == 512 && !V` (either order), or `<head dim> == 512` alone over a body that is exactly `if (!V) { return false; }`.
    A condition with any other term, or a V that is not that latch, keeps its 512 literal for the literal scan to see."""
    raw = strip_comments_b(src[sb(body):eb(body)])
    text = bytearray(raw)
    latches = [(m.group(1), m.start()) for m in P_LATCH_RE.finditer(raw.decode("utf-8", "replace"))]
    for n in walk(body):
        if kind(n) != "if_statement" or fld(n, "condition") is None:
            continue
        cond = fld(n, "condition")
        c = squash(clean_text(src, cond))
        m = re.fullmatch(r"\((.*)\)", c)
        inner = m.group(1) if m else c
        cons = squash(clean_text(src, fld(n, "consequence"))) if fld(n, "consequence") is not None else ""
        for v, at in latches:
            ev = re.escape(v)
            if at < sb(n) - sb(body) and (re.fullmatch(P_HD512 + "&&!" + ev, inner) or re.fullmatch("!" + ev + "&&" + P_HD512, inner)
                                          or (re.fullmatch(P_HD512, inner) and cons in [b % v for b in P_HATCH_BODIES])):
                a, b = sb(cond) - sb(body), eb(cond) - sb(body)
                text[a:b] = re.sub(rb"[^\n]", b" ", bytes(text[a:b]))
                break
    return bytes(text).decode("utf-8", "replace")


def p_support_clauses(text):
    """(kind, snippet) for each support clause in comment-stripped function text: a call of a support helper or of an fp8
    type helper, a GGML_TYPE_ comparison or case, and a head dim compared with an integer or a named constant, taken modulo
    or masked, or switched on."""
    names = {nm for nm in P_BODY_BANNED if re.search(r"(?<![\w])%s\s*\(" % nm, text)}
    names |= {m.group(1) for m in P_FP8_CALL_RE.finditer(text)}
    hits = [("call", nm) for nm in sorted(names)]
    hits += [("type-compare", squash(m.group(0))) for m in P_TYPE_CMP_RE.finditer(text)]
    hits += [("head-dim-literal", squash(m.group(0))) for m in P_LITERAL_CMP_RE.finditer(text)]
    hits += [("head-dim-named", squash(m.group(0))) for m in P_NAMED_CMP_RE.finditer(text)]
    hits += [("head-dim-arith", squash(m.group(0))) for m in P_ARITH_RE.finditer(text)]
    if P_SWITCH_RE.search(text):
        hits += [("head-dim-case", "case" + m.group(1)) for m in P_CASE_RE.finditer(text)]
    return hits


def p_fill_sites(ix):
    """((file, function) of each pinned fill, whether the b2 value function exists): the shape function, the dispatch function
    and the charge side's one fill (the load-maxima fill in b1, the value function in b2)."""
    b2 = bool(ix.fdefs(P_VALUEFN))
    return [("fattn.cpp", P_SHAPE_OF), ("fattn.cpp", P_FLASH), ("unified-cache.cpp", P_VALUEFN if b2 else P_LOADFILL)], b2


def p_home(ix, files, out):
    out.dormant.append("TODO p-home: a head dim copied into a local, a switch or comparison on such a local, and a support clause "
                       "spelled through a predicate other than the listed helpers are not followed (no dataflow)")
    for nm in (P_ENABLED, P_SUPPORTED, P_SHAPE_SUPPORTED):
        out.latch("p-home", ix, nm, "a call or a function definition", "an alias, a macro or a lambda hides it")
    enabled = ix.fdefs(P_ENABLED)
    if not enabled:
        p_dormant(out, "home", P_ENABLED)
        return
    out.active.add("p-home")
    reads = []
    for rel, src in sorted(files.items()):
        if P_ENV.encode() in src:
            clean = strip_comments_b(src)
            for m in P_GETENV_RE.finditer(clean):
                reads.append((rel, m.start(), clean.count(b"\n", 0, m.start()) + 1))
    spans = [(rel, sb(body_of(n)), eb(body_of(n))) for rel, _, n in enabled]
    inside = [r for r in reads if any(r[0] == a and b <= r[1] < c for a, b, c in spans)]
    for rel, off, line in reads:
        if (rel, off, line) not in inside or len(inside) > 1:
            out.add("P-HOME", rel, line, "<read>", P_ENV, "%s::p-home:getenv:%d" % (rel, line),
                    "getenv(\"%s\") is read at %s:%d; the one read of the switch is inside %s" % (P_ENV, rel, line, P_ENABLED))
    if not inside:
        rel, func, node = enabled[0]
        out.add("P-HOME", rel, line_of(node), func, P_ENV, "%s::%s::p-home:no-read" % (rel, func),
                "%s does not read getenv(\"%s\")" % (P_ENABLED, P_ENV))
    shape = ix.fdefs(P_SHAPE_SUPPORTED)
    calls = ix.occ(P_ENABLED, ("call",))
    if not shape:
        out.add("P-HOME", "fattn.cpp", 0, P_SHAPE_SUPPORTED, P_SHAPE_SUPPORTED, "p-home:shape-missing",
                "%s is defined but %s is not: the shape function is the one caller of the switch" % (P_ENABLED, P_SHAPE_SUPPORTED))
    else:
        good = False
        for rel, func, node in shape:
            st = first_statement(node)
            good = good or (st is not None and len(calls) == 1 and calls[0][0] == rel and in_span(calls[0][1].call, st))
        if not good:
            rel, func, node = shape[0]
            out.add("P-HOME", rel, line_of(node), func, P_ENABLED, "%s::%s::p-home:first-statement" % (rel, func),
                    "%s must be called exactly once under ggml/src/ggml-sycl, as the first statement of %s (%d call(s) found)" % (
                        P_ENABLED, P_SHAPE_SUPPORTED, len(calls)))
    if not ix.fdefs(P_SUPPORTED):
        out.add("P-HOME", "fattn.cpp", 0, P_SUPPORTED, P_SUPPORTED, "p-home:supported-missing",
                "%s is defined but %s is not: its body is where the no-support-clause pin is checked" % (P_ENABLED, P_SUPPORTED))
    routed = ix.fdefs(P_ROUTED)
    if not routed:
        out.add("P-HOME", "fattn.cpp", 0, P_ROUTED, P_ROUTED, "p-home:routed-missing",
                "%s is defined but %s is not" % (P_ENABLED, P_ROUTED))
    for rel, func, node in routed:
        body = squash(clean_text(ix.files[rel], body_of(node)))
        if body != squash("{ %s }" % P_ROUTED_BODY):
            out.add("P-HOME", rel, line_of(node), func, P_ROUTED, "%s::%s::p-home:routed-body" % (rel, func),
                    "the body of %s must be exactly `%s`, the support decision is not restated here" % (P_ROUTED, P_ROUTED_BODY))
    for name in (P_SUPPORTED, P_ADMITS, P_ROUTE_ENABLED):
        for rel, func, node in ix.fdefs(name):
            src = ix.files[rel]
            text = clean_text(src, body_of(node)) if name == P_SUPPORTED else p_blank_hatch(src, body_of(node))
            for knd, what in p_support_clauses(text):
                out.add("P-HOME", rel, line_of(node), func, what, "%s::%s::p-home:clause:%s:%s" % (rel, func, knd, what),
                        "%s holds a support clause (%s %s): the support decision lives in %s alone" % (
                            name, knd, what, P_SHAPE_SUPPORTED))
            if name == P_SUPPORTED:
                n = len(calls_in_node(src, body_of(node), P_SHAPE_SUPPORTED))
                if n != 1:
                    out.add("P-HOME", rel, line_of(node), func, P_SHAPE_SUPPORTED, "%s::%s::p-home:shape-call" % (rel, func),
                            "%s must call %s exactly once (%d found)" % (P_SUPPORTED, P_SHAPE_SUPPORTED, n))
    sites, _ = p_fill_sites(ix)
    fill_names = tuple(n for _, n in sites)
    p_callers(ix, out, "P-HOME", "p-home:census", ((P_VEC, (P_SHAPE_SUPPORTED, P_FAST_POLICY)), (P_TILE, (P_SHAPE_SUPPORTED, P_FLASH)),
                                                  (P_KV_PAIR_OF, (P_SHAPE_SUPPORTED,) + fill_names)))
    for rel, o in ix.occ(P_KV_PAIR_OF, ("call",)):
        if fn_name(o.func) in fill_names:
            st = o.call
            while st is not None and not (parent(st) is not None and kind(parent(st)) == "compound_statement"):
                st = parent(st)
            if st is None or not P_FILL_RE.match(squash(clean_text(ix.files[rel], st))):
                out.add("P-HOME", rel, o.line, o.func, P_KV_PAIR_OF, "%s::%s::p-home:census-fill:%s" % (rel, o.func, P_KV_PAIR_OF),
                        "%s is called in %s outside the pinned fill line: its callers are the shape function and the pinned "
                        "`<v>.kv_is_fp8 = ...` lines only" % (P_KV_PAIR_OF, o.func))
    n_tile = sum(1 for _, o in ix.occ(P_TILE, ("call",)) if fn_name(o.func) == P_FLASH)
    if n_tile != 2:
        out.add("P-HOME", "fattn.cpp", 0, P_FLASH, P_TILE, "p-home:tile-dispatch-sites",
                "%s is called at exactly the two dispatch sites of %s (%d found)" % (P_TILE, P_FLASH, n_tile))
    sup = [(rel, func, n) for rel, func, n in ix.fdefs(P_SUPPORTS_OP) if rel == "ggml-sycl.cpp"]
    found = False
    for rel, func, node in sup:
        for n in walk(body_of(node)):
            if kind(n) == "case_statement" and "GGML_OP_FLASH_ATTN_EXT" in clean_text(ix.files[rel], fld(n, "value") or n):
                found = True
                if squash(clean_text(ix.files[rel], n)) != squash(P_CASE):
                    out.add("P-HOME", rel, line_of(n), func, "GGML_OP_FLASH_ATTN_EXT", "%s::%s::p-home:case-line" % (rel, func),
                            "the FLASH_ATTN_EXT case of %s must be exactly `%s`" % (P_SUPPORTS_OP, P_CASE))
    if not found:
        out.add("P-HOME", "ggml-sycl.cpp", 0, P_SUPPORTS_OP, "GGML_OP_FLASH_ATTN_EXT", "p-home:case-missing",
                "%s has no GGML_OP_FLASH_ATTN_EXT case" % P_SUPPORTS_OP)


def p_fill(ix, out):
    for nm in (P_KV_PAIR_OF, P_SHAPE_OF):
        out.latch("p-fill", ix, nm, "a call or a function definition", "an alias, a macro or a lambda hides it")
    if not has_name(ix, P_KV_PAIR_OF, "def"):
        p_dormant(out, "fill", P_KV_PAIR_OF)
        return
    out.active.add("p-fill")
    fills, b2 = p_fill_sites(ix)
    if b2 and ix.fdefs(P_LOADFILL):
        rel, func, node = ix.fdefs(P_LOADFILL)[0]
        out.add("P-FILL", rel, line_of(node), func, P_LOADFILL, "%s::%s::p-fill:load-fill-in-b2" % (rel, func),
                "%s is deleted in b2 (the load-maxima fill is D once the value function exists)" % P_LOADFILL)
    fill_funcs = set()
    for rel, name in fills:
        defs = [(f, n) for r, f, n in ix.fdefs(name) if r == rel]
        if not defs:
            out.add("P-FILL", rel, 0, name, name, "%s::%s::p-fill:function-missing" % (rel, name),
                    "%s must define the fill function %s" % (rel, name))
            continue
        for func, node in defs:
            fill_funcs.add((rel, sb(node), eb(node)))
            src = ix.files[rel]
            own = set(re.findall(r"fattn_params\s*[&*]?\s*(\w+)", clean_text(src, node)))
            n = 0
            body_text = clean_text(src, body_of(node))
            for m in re.finditer(r"[^;{}]*;", body_text):
                st = squash(m.group(0))
                mm = P_FILL_RE.match(st)
                n += bool(mm and mm.group(1) in own)
            if n != 1:
                out.add("P-FILL", rel, line_of(node), func, name, "%s::%s::p-fill:pin" % (rel, func),
                        "%s must hold exactly one line `<v>.kv_is_fp8 = ggml_sycl_fattn_kv_pair_of(<v>.K_type, <v>.V_type) == "
                        "GGML_SYCL_FATTN_KV_PAIR_FP8;` on its own fattn_params variable (%d found)" % (func, n))
    for rel, src in sorted(ix.files.items()):
        clean = strip_comments_b(src)
        for m in P_WRITE_RE.finditer(clean):
            a = clean.rfind(b";", 0, m.start()) + 1
            a = max(a, clean.rfind(b"{", 0, m.start()) + 1, clean.rfind(b"}", 0, m.start()) + 1)
            e = clean.find(b";", m.end() - 1) + 1
            st = squash(clean[a:e].decode("utf-8", "replace"))
            ok = bool(P_FILL_RE.match(st)) and any(r == rel and s <= m.start() < t for r, s, t in fill_funcs)
            if not ok:
                line = clean.count(b"\n", 0, m.start()) + 1
                out.add("P-FILL", rel, line, "<write>", "kv_is_fp8", "%s::p-fill:write:%s" % (rel, text_hash(st)),
                        "%s:%d writes kv_is_fp8 (`%s`) outside the pinned fill lines: one fact, one source" % (rel, line, st[:80]))


def p_layer(ix, out):
    out.latch("p-layer", ix, P_LAYER_OF, "a call or a function definition", "an alias, a macro or a lambda hides it")
    if not has_name(ix, P_LAYER_OF, "def"):
        p_dormant(out, "layer", P_LAYER_OF)
        return
    out.active.add("p-layer")
    helper = [(rel, sb(n), eb(n)) for rel, _, n in ix.fdefs(P_LAYER_OF) if rel == "common.cpp"]
    n_helper = 0
    for rel, src in sorted(ix.files.items()):
        clean = strip_comments_b(src)
        for m in P_LAYER_RE.finditer(clean):
            ls = clean.rfind(b"\n", 0, m.start()) + 1
            le = clean.find(b"\n", m.end())
            line = clean[ls:le if le >= 0 else len(clean)].decode("utf-8", "replace")
            ok = False
            if any(rel == r and a <= m.start() < b for r, a, b in helper):
                n_helper += 1
                ok = True
            elif rel == "fattn.cpp" and squash(line) == squash('return tensor && strncmp(tensor->name, "cache_k_l", 9) == 0;'):
                ok = True
            elif rel == "ggml-sycl.cpp" and re.search(rb'"cache_[kv]_l0"', clean[m.start():m.start() + 12]):
                ok = True
            if not ok:
                ln = clean.count(b"\n", 0, m.start()) + 1
                out.add("P-LAYER", rel, ln, "<line>", "cache_[kv]_l", "%s::p-layer:%s" % (rel, text_hash(squash(line))),
                        "%s:%d spells a KV layer-name prefix (`%s`): the layer id is parsed once, in %s" % (
                            rel, ln, line.strip()[:80], P_LAYER_OF))
    if n_helper != 2:
        out.add("P-LAYER", "common.cpp", 0, P_LAYER_OF, P_LAYER_OF, "p-layer:helper-formats",
                "%s holds exactly two format strings, cache_k_l and cache_v_l (%d found)" % (P_LAYER_OF, n_helper))


def p_charge(ix, out):
    for nm in (P_LOADFILL, P_VALUEFN):
        out.latch("p-charge", ix, nm, "a call or a function definition", "an alias, a macro or a lambda hides it")
    subjects = [(n, rel, func, node) for n in (P_LOADFILL, P_VALUEFN) for rel, func, node in ix.fdefs(n)]
    if not subjects:
        p_dormant(out, "charge", "%s / %s" % (P_LOADFILL, P_VALUEFN))
        return
    out.active.add("p-charge")
    for rel in ("unified-cache.cpp", "unified-cache.hpp"):
        clean = strip_comments_b(ix.files.get(rel, b"")).decode("utf-8", "replace")
        for nm in P_SUPPORT_HELPERS:
            for m in re.finditer(r"(?<![\w])%s(?![\w])" % nm, clean):
                out.add("P-CHARGE", rel, clean.count("\n", 0, m.start()) + 1, "<charge side>", nm,
                        "%s::p-charge:helper:%s" % (rel, nm),
                        "the charge side (%s) names %s: a support clause there makes the charge stricter than supports_op" % (rel, nm))
    for name, rel, func, node in subjects:
        text = clean_text(ix.files[rel], body_of(node))
        d512 = 0
        for m in P_LITERAL_CMP_RE.finditer(text):
            st = squash(m.group(0))
            ln = text[:m.start()].count("\n") + line_of(body_of(node))
            if re.search(r"(?<![\w.>])params\.ne00\s*==\s*512", m.group(0)) and name == P_VALUEFN:
                d512 += 1
                continue
            out.add("P-CHARGE", rel, ln, func, st, "%s::%s::p-charge:literal:%s" % (rel, func, st),
                    "%s compares a head dim with an integer literal (`%s`): the charge side takes its dims from the facts, "
                    "the only literal is the walk's D512 count" % (func, st))
        for knd, what in p_support_clauses(text):
            if knd in ("head-dim-named", "head-dim-arith", "head-dim-case"):
                out.add("P-CHARGE", rel, line_of(node), func, what, "%s::%s::p-charge:%s:%s" % (rel, func, knd, what),
                        "%s tests a head dim (%s `%s`): the charge side takes its dims from the facts, never from a support rule" % (
                            func, knd, what))
        if name == P_VALUEFN:
            lines = [squash(m.group(0)) for m in re.finditer(r"if\s*\(\s*params\.ne00\s*==\s*512\s*\)\s*\{", text)]
            if lines.count(P_D512_LINE) != 1 or d512 != 1:
                out.add("P-CHARGE", rel, line_of(node), func, "512", "%s::%s::p-charge:d512-count" % (rel, func),
                        "%s must hold the line `if (params.ne00 == 512) {` exactly once, and no other comparison of the head dim "
                        "with 512 (%d such line(s), %d exempt comparison(s) found)" % (func, lines.count(P_D512_LINE), d512))


def s2d_findings(files):
    """(violations, dormant lines, active clauses) of clauses (i)-(o) and witness 9's clause. Tests are read for clause (n)
    only."""
    core = core_files(files)
    ix, ixt = Index(core), Index({r: s for r, s in files.items() if not r.startswith(REPO_PREFIX)})
    out = S2dOut()
    clause_i(ix, out)
    clause_k(ix, out)
    clause_l(ix, out)
    clause_j(ix, out)
    clause_m(ix, out)
    clause_n(ixt, out)
    clause_o(ix, out)
    clause_z9(ix, out)
    p_route(ix, out)
    p_home(ix, files, out)
    p_fill(ix, out)
    p_layer(ix, out)
    p_charge(ix, out)
    return out


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
            + (("name",) if e.get("code") in RAW_CODES else ())
        miss = [k for k in need if k not in e]
        if miss:
            errs.append("FAIL allowlist entry %s lacks %s" % (e.get("id", "#%d" % i), ", ".join(miss)))
            continue
        if e["code"] not in CODES:
            errs.append("FAIL allowlist entry %s has unknown code %r" % (e["id"], e["code"]))
        if e["code"] in S2D_NEVER_ALLOW:
            errs.append("FAIL allowlist entry %s: a finding of code %s is fixed, never exempted (an X-LATCH means the clause cannot "
                        "see the code; a P-* finding is a design pin)" % (e["id"], e["code"]))
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
        elif SRC_LINE_RE.search(e["reason"]):
            errs.append("FAIL allowlist entry %s reason names a source line (file:NNN), which rots with the next edit; name the function"
                        % e["id"])
        elif LEDGER_RE.search(e["reason"]):
            errs.append("FAIL allowlist entry %s reason names a ruling-ledger id (ruling Mnnn Rn), a ledger outside the repo; cite the "
                        "design step or the contract section" % e["id"])
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
        if d["code"] in S2D_CODES and d["code"] not in S2D_DEBT:
            errs.append("FAIL debt entry %s %s: a finding of this code is allowlisted per node or fixed, never debt (only N-VOID and "
                        "N-NODISCARD, the declined results dropped today, are)" % (d["code"], d["key"]))
        seen[(d["code"], d["key"])] += 1
        if d["code"] in RAW_CODES and not FATE_RE.match(str(d.get("fate", ""))):
            errs.append("FAIL debt entry %s %s has no valid fate (deleted-by-*, converted-by-*, sanctioned-internal, "
                        "sanctioned-vendored, pending-disposition)" % (d["code"], d["key"]))
        pinned = d["code"] == "G-CATCH" and d["key"] == CHECK_TRY_ERROR_KEY
        if (d["code"] in RAW_CODES or pinned) and (not isinstance(d.get("cite"), str) or len(d["cite"].strip()) < CITE_MIN):
            errs.append("FAIL debt entry %s %s has no cite (a ticket id or a design/census row, at least %d characters)"
                        % (d["code"], d["key"], CITE_MIN))
        elif isinstance(d.get("cite"), str) and LEDGER_RE.search(d["cite"]):
            errs.append("FAIL debt entry %s %s cite names a ruling-ledger id (ruling Mnnn Rn), a ledger outside the repo; cite the "
                        "design step" % (d["code"], d["key"]))
        if pinned and d.get("fate") != CHECK_TRY_ERROR_FATE:
            errs.append("FAIL debt entry G-CATCH %s must carry fate %s (step 5.4a rewrites this handler) and no other"
                        % (d["key"], CHECK_TRY_ERROR_FATE))
        if d["code"] == "G-CATCH" and d["key"] != CHECK_TRY_ERROR_KEY and d["key"].startswith(CHECK_TRY_ERROR_KEY.split("::catch_macro")[0] + "::"):
            errs.append("FAIL debt entry G-CATCH %s names the CHECK_TRY_ERROR macro's handler but is not the pinned key %s; a re-key must "
                        "move CHECK_TRY_ERROR_KEY with it, or the fate pin lapses without a failure" % (d["key"], CHECK_TRY_ERROR_KEY))
        if "fate" in d and d["code"] not in RAW_CODES and not (d["code"] == "G-CATCH" and d["key"] == CHECK_TRY_ERROR_KEY):
            errs.append("FAIL debt entry %s %s carries a fate; only E-RAW, E-LIBC and the one CHECK_TRY_ERROR G-CATCH entry may"
                        % (d["code"], d["key"]))
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
                 edit_debt=None, planted=True, allow_nodes=None, allow_from=None, active=None, dormant=None, m_edit=None):
        self.wid, self.label, self.mutate, self.expect = wid, label, mutate, expect
        # active / dormant: a clause letter the S2d pass must report as keyed on a subject / as still dormant. m_edit edits
        # a copy of M_TABLES for the run.
        self.active, self.dormant, self.m_edit = active, dormant, m_edit
        self.code, self.naming, self.allowlist = code, naming, allowlist
        self.edit_allowlist, self.edit_debt, self.planted = edit_allowlist, edit_debt, planted
        # allow_nodes: [{"code", "func", "nth", "extra"}]. Each becomes a key-matched allowlist entry for the node the gate
        # finds in the tree `allow_from` builds (default: the case's own), so a node can be listed and then mutated away.
        self.allow_nodes, self.allow_from = allow_nodes, allow_from


# witness id -> what it pins. Every id (the process-level ones aside) must have a FAIL case.
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
    "f": "missing tree_sitter_language_pack exits 1", "cmake": "the ctest registrations: the plain gate, the witnesses test and every shard, their TIMEOUTs and labels, no regeneration flag",
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
    "s3q": "clause (q): the C library's allocation primitives (mmap, posix_memalign, memalign, aligned_alloc, malloc, calloc, realloc, "
           "VirtualAlloc) outside the allowlist; the committed entries are pinned by function, name and count",
    "s31-l": "clause (l) is enforced once mem_handle::owner_use_count exists, and dormant again when it is gone",
    "s3q-data": "an E-LIBC debt entry needs a fate and a cite, and an E-LIBC allowlist entry a name",
    "s3q-pin": "the CHECK_TRY_ERROR handler's debt entry carries converted-by-5.4a and no other fate, and no other entry may carry a fate",
    "s3f-i1": "clause (q) in a #define body and in the lexical pass reads a qualifier across spaces and line continuations",
    "s3f-m4": "dpct_memcpy and async_dpct_memcpy, which reach dpct's host_buffer malloc, are forbidden outside dpct/helper.hpp",
    "s3f-m5": "clause (q) also names mmap64, mremap, valloc, pvalloc, reallocarray, strdup and strndup",
    "s3f-data": "the CHECK_TRY_ERROR pin cannot lapse by a re-key and carries a cite; no allowlist reason names a source line; no cite names a ruling-ledger id",
    "s3g-kw": "clause (q): a C++ keyword before `::` (return, else, throw, sizeof, ...) leaves the name bare, and a comparison `>` is not a template close",
    "s3g-q": "clause (q) in text reads the whole scope like the tree does (xstd, pool2, ns::std are scopes; ::std is not), across continuations in both paths",
    "s3g-data": "a ruling-ledger id is refused in any spelling, and a source line in any file form, in an allowlist reason",
    "s2c-catch": "spellings and placements of the rethrow clause", "s2c-data": "clause-(h) entries that must be refused by validation",
    "9": "a model-shaped *_bytes() function is called by its allocation sites and by the zone sizing; dormant until defined",
    "27": "clause (j): A's fit and W's term read per-model sources only; the eligibility classifier is not called outside the late stage",
    "28": "clause (i): defining L0's replan-token accessor retires the W lost-CAS retry", "31": "clause (i): ... and A's pre-L0 busy return",
    "30": "clause (k): §B's COMPLETE reap in the transaction retires the pre-§B interim",
    "29": "clause (l): the owner count never selects a free, its callers are listed, replace_within guards first",
    "32": "clause (m): every SCRATCH row of the appendix is on the floor list, covered, or excluded by class; the floor is tied to its env",
    "33": "clause (n) in tests: a declined result is consumed where a test calls the listed names",
    "35": "clause (n): a declined result is consumed, and each listed declaration is [[nodiscard]]",
    "36": "clause (o): each C-term consumer submits on its census row's queue; every acquire has a row",
    "s2d": "the appendix census table is data: a missing appendix-rows.json fails",
    "37": "clause (p): the oneDNN SDPA arms sit in routed branches; the plan is read only by route_admits",
    "38": "clause (p): the support decision has one home (switch read, shape function, fills, layer name, case line, charge side)",
}

# the witnesses that run as process-level checks in `--witnesses` rather than as matrix cases
PROCESS_WITNESSES = ("f", "cmake", "m6", "m9", "m14", "r2m4", "r2m8", "s2d")


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
    # S3-0 flips this on purpose: until then the C library's malloc and aligned_alloc were a PASS control here, because no clause saw
    # them. Clause (q) now does, so they are findings of that clause (E-LIBC), not of clause (e)'s USM names.
    A(Case("s2c-host", "the C library's malloc and aligned_alloc are clause (q)'s, not USM entry points: E-LIBC (flipped by S3-0)", plant(
        "void zzplant_host() {\n    void * a = std::malloc(16);\n    void * b = ::aligned_alloc(64, 64);\n    void * c = malloc(8);\n"
        "    (void) a; (void) b; (void) c;\n}\n"), "FAIL", "E-LIBC", "malloc"))
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
    c.extend(matrix_cases_s2d1())
    c.extend(matrix_cases_s2d1b())
    c.extend(matrix_cases_s2d2())
    c.extend(matrix_cases_s2d3())
    c.extend(matrix_cases_s2d4())
    c.extend(matrix_cases_s2d5())
    c.extend(matrix_cases_s3q())
    c.extend(matrix_cases_s31())
    c.extend(matrix_cases_s3f())
    c.extend(matrix_cases_s3g())
    return c


def replace_once(rel, old, new):
    """Replace the one occurrence of `old` in a file; a missing or ambiguous anchor is a setup error."""
    def f(files):
        src = files[rel].decode()
        if src.count(old) != 1:
            raise SystemExit("matrix setup error: %r occurs %d times in %s, expected once" % (old, src.count(old), rel))
        return dict(files, **{rel: src.replace(old, new).encode()})
    return f


def chain(*fns):
    def f(files):
        for fn in fns:
            files = fn(files)
        return files
    return f


def drop_debt(code, *needles):
    """edit_debt: drop the debt entries of `code` whose key holds every needle (a planted fix that retires them)."""
    return lambda debt: dict(debt, violations=[d for d in debt["violations"]
                                               if not (d["code"] == code and all(n in d["key"] for n in needles))])


def with_allow(*specs):
    """edit_allowlist: add one count-1 entry per (code, file, function)."""
    def f(al):
        ents = [{"id": "E-ZZ-S%d" % i, "code": c, "file": fl, "function": fn, "count": 1, "reason": "mutation-matrix planted function"}
                for i, (c, fl, fn) in enumerate(specs)]
        return dict(al, entries=list(al.get("entries", [])) + ents)
    return f


def m_without(table, key):
    def f(t):
        t[table].pop(key, None)
    return f


def m_set(table, key, val):
    def f(t):
        t[table][key] = val
    return f


# ---------------------------------------------------------------- clause (p) fixtures
# Clause (p) is dormant on today's tree: its subjects arrive with beni (b1, then b2). A matrix witness cannot plant a twin of
# a function the rules constrain in place, so it builds the tree the rules describe by transforming today's: b1_tree() is
# today's tree with the routed predicate, the support helpers, the pinned fills, the layer helper and the load-maxima fill
# in the shapes the clause pins; b2_tree() is b1 with that fill replaced by the value function. Each transform is an exact
# anchored edit (a missing or ambiguous anchor is a setup error), so a tree edit that moves an anchor fails loudly here.
B1_LOADFILL = """
void placement_plan_set_routed_head_maxima(placement_plan & plan, const std::vector<kv_layer_facts> & layers) {
    fattn_params params{};
    for (const auto & f : layers) {
        params.ne00      = f.head_dim_k;
        params.ne10      = f.head_dim_v;
        params.K_type    = f.K_type;
        params.V_type    = f.V_type;
        params.kv_is_fp8 = ggml_sycl_fattn_kv_pair_of(params.K_type, params.V_type) == GGML_SYCL_FATTN_KV_PAIR_FP8;
        if (!ggml_sycl_fattn_onednn_routed(params, f.head_dim_v, false, nullptr, nullptr)) {
            continue;
        }
        plan.planner_n_head_ctx_max = std::max(plan.planner_n_head_ctx_max, f.n_head);
    }
}
"""
B2_VALUEFN = """
size_t onednn_graph_scratch_bytes(const std::vector<kv_layer_facts> & layers) {
    fattn_params params{};
    size_t       d512 = 0;
    for (const auto & f : layers) {
        params.ne00      = f.head_dim_k;
        params.ne10      = f.head_dim_v;
        params.K_type    = f.K_type;
        params.V_type    = f.V_type;
        params.kv_is_fp8 = ggml_sycl_fattn_kv_pair_of(params.K_type, params.V_type) == GGML_SYCL_FATTN_KV_PAIR_FP8;
        if (!ggml_sycl_fattn_onednn_routed(params, f.head_dim_v, false, nullptr, nullptr)) {
            continue;
        }
        if (params.ne00 == 512) {
            ++d512;
        }
    }
    return d512;
}
"""
B1_FATTN_TAIL = """
ggml_sycl_fattn_kv_pair ggml_sycl_fattn_kv_pair_of(ggml_type k, ggml_type v) {
    if (k == GGML_TYPE_F16 && v == GGML_TYPE_F16) {
        return GGML_SYCL_FATTN_KV_PAIR_F16;
    }
    return ggml_sycl_type_is_fp8_e4m3(k) && ggml_sycl_type_is_fp8_e4m3(v) ? GGML_SYCL_FATTN_KV_PAIR_FP8 : GGML_SYCL_FATTN_KV_PAIR_NONE;
}

bool ggml_sycl_flash_attn_ext_enabled() {
    static const bool enabled = []() {
        const char * env = std::getenv("GGML_SYCL_FLASH_ATTN_EXT");
        return !env || !(strcmp(env, "0") == 0 || strcmp(env, "false") == 0);
    }();
    return enabled;
}

fattn_params ggml_sycl_fattn_shape_of(const ggml_tensor * dst) {
    fattn_params params{};
    params.ne00      = dst->src[0]->ne[0];
    params.K_type    = dst->src[1]->type;
    params.V_type    = dst->src[2]->type;
    params.kv_is_fp8 = ggml_sycl_fattn_kv_pair_of(params.K_type, params.V_type) == GGML_SYCL_FATTN_KV_PAIR_FP8;
    return params;
}

bool ggml_sycl_fattn_shape_supported(const fattn_params & p, int d_v) {
    if (!ggml_sycl_flash_attn_ext_enabled()) {
        return false;
    }
    if (!fattn_vec_supports_head_dim(p.ne00) && !(p.ne00 == 512 && ggml_sycl_fattn_onednn_route_admits(p, false, nullptr, nullptr))) {
        return false;
    }
    return ggml_sycl_fattn_kv_pair_of(p.K_type, p.V_type) != GGML_SYCL_FATTN_KV_PAIR_NONE && d_v > 0 &&
           (p.ne00 != 512 || ggml_sycl_fattn_d512_tile_admissible(nullptr));
}

static bool ggml_sycl_fattn_onednn_route_enabled(const fattn_params & p) {
    if (!g_sycl_fa_onednn_enabled || g_sycl_paged_v2_enabled) {
        return false;
    }
    static const bool d512_onednn_enabled = ggml_sycl_fa_onednn_d512_enabled();
    if (p.ne00 == 512 && !d512_onednn_enabled) {
        return false;
    }
    return true;
}

bool ggml_sycl_fattn_onednn_route_admits(const fattn_params & p, bool multi_seq, const ggml_backend_sycl_context * ctx, ggml_sycl_onednn_fa_layout_plan * plan_out) {
    if (!ggml_sycl_fattn_onednn_route_enabled(p)) {
        return false;
    }
    const ggml_sycl_onednn_fa_layout_plan plan =
        ggml_sycl_flash_attn_ext_onednn_plan(p, p.ne02, p.ne12, p.kv_is_fp8, multi_seq);
    if (plan_out != nullptr) {
        *plan_out = plan;
    }
    return plan.kind == ggml_sycl_onednn_fa_layout_kind::DIRECT ||
           plan.kind == ggml_sycl_onednn_fa_layout_kind::MATERIALIZE_REQUIRED;
}

bool ggml_sycl_fattn_onednn_routed(const fattn_params & p, int32_t d_v, bool multi_seq, const ggml_backend_sycl_context * ctx, ggml_sycl_onednn_fa_layout_plan * plan_out) {
    return ggml_sycl_fattn_onednn_route_admits(p, multi_seq, ctx, plan_out) && ggml_sycl_fattn_shape_supported(p, d_v);
}

bool ggml_sycl_fattn_onednn_dispatch_routed(ggml_backend_sycl_context & ctx, const fattn_params & p, int32_t d_v, bool multi_seq,
                                            ggml_sycl_fattn_onednn_route_site site, ggml_sycl_fattn_onednn_route_result & out) {
    out = {};
    if (!ggml_sycl_fattn_onednn_route_enabled(p)) {
        return false;
    }
    if (ggml_sycl_onednn_graph_dispatch_declined(ctx, p)) {
        out.stage = GGML_SYCL_FATTN_ONEDNN_ROUTE_STAGE_DECLINED;
        unified_cache_count_onednn_graph_mask_declined(ctx.device, static_cast<int>(site));
        return false;
    }
    out.stage = GGML_SYCL_FATTN_ONEDNN_ROUTE_STAGE_PLANNED;
    return ggml_sycl_fattn_onednn_routed(p, d_v, multi_seq, &ctx, &out.plan);
}
"""
B1_ONEDNN_TAIL = """
bool ggml_sycl_onednn_graph_dispatch_declined(ggml_backend_sycl_context & ctx, const fattn_params & p) {
    return onednn_graph_allocator_enabled() &&
           ggml_sycl_onednn_graph_interim_gate(ctx, p.kv_layer, onednn_graph_scratch_term_bytes(p.ne02, p.ne01, p.ne11));
}
"""
B1_COMMON_TAIL = """
int ggml_sycl_kv_cache_layer_of(const char * name) {
    int id = -1;
    if (sscanf(name, "cache_k_l%d", &id) == 1) {
        return id;
    }
    if (sscanf(name, "cache_v_l%d", &id) == 1) {
        return id;
    }
    return -1;
}
"""


def replace_between(rel, start, end, new):
    """Replace the text from `start` up to (not including) `end` in one file; both anchors must occur exactly once."""
    def f(files):
        src = files[rel].decode()
        if src.count(start) != 1 or src.count(end) != 1 or src.index(start) > src.index(end):
            raise SystemExit("matrix setup error: anchors of replace_between not unique or out of order in %s" % rel)
        i, j = src.index(start), src.index(end)
        return dict(files, **{rel: (src[:i] + new + src[j:]).encode()})
    return f


def replace_regex(rel, pattern, new, count):
    def f(files):
        out, n = re.subn(pattern, new, files[rel].decode())
        if n != count:
            raise SystemExit("matrix setup error: %r matched %d times in %s, expected %d" % (pattern, n, rel, count))
        return dict(files, **{rel: out.encode()})
    return f


def route_call(site, d_v="d_v"):
    return "ggml_sycl_fattn_onednn_dispatch_routed(ctx, params, %s, multi_seq, GGML_SYCL_FATTN_ONEDNN_ROUTE_SITE_%s, route)" % (d_v, site)


GATE_DEFAULT, GATE_D512 = route_call("DEFAULT"), route_call("D512", "V->ne[0]")
FORCE_OPEN = ('if (strcmp(force, "onednn") == 0) {\n                force_known = true;\n'
              "                const bool multi_seq = (params.n_seqs > 1);\n                ggml_sycl_fattn_onednn_route_result route;\n"
              "                const bool routed = " + route_call("FORCE") + ";\n"
              "                const ggml_sycl_onednn_fa_layout_plan & plan = route.plan;\n                if (routed) {\n")
FILL_LINE = "params.kv_is_fp8 = ggml_sycl_fattn_kv_pair_of(params.K_type, params.V_type) == GGML_SYCL_FATTN_KV_PAIR_FP8;"


def b1_tree(files, keep_test_reads=False):
    steps = (
        replace_between("fattn.cpp", "bool ggml_sycl_flash_attn_ext_supported(const ggml_tensor * dst) {\n    // Enabled by default;",
                        "\n// =============================================================================\n// Shape-aware FA dispatch table",
                        "bool ggml_sycl_flash_attn_ext_supported(const ggml_tensor * dst) {\n"
                        "    const fattn_params p = ggml_sycl_fattn_shape_of(dst);\n"
                        "    return ggml_sycl_fattn_shape_supported(p, (int) dst->src[2]->ne[0]);\n}\n"),
        replace_once("fattn.cpp", "    params.kv_is_fp8 = (ggml_sycl_type_is_fp8_e4m3(K->type) && ggml_sycl_type_is_fp8_e4m3(V->type));\n"
                     "    params.n_seqs    = 0;", "    ggml_sycl_fattn_apply_shape(params, dst);\n    params.n_seqs    = 0;"),
        replace_once("fattn.cpp", "    params.kv_is_fp8 = (ggml_sycl_type_is_fp8_e4m3(K->type) && ggml_sycl_type_is_fp8_e4m3(V->type));\n\n"
                     "    // Multi-token decode support", "    " + FILL_LINE + "\n\n    // Multi-token decode support"),
        replace_between("fattn.cpp", 'if (strcmp(force, "onednn") == 0 && g_sycl_fa_onednn_enabled && !g_sycl_paged_v2_enabled) {',
                        '                    GGML_SYCL_KTRACE("fattn FORCE=onednn"', FORCE_OPEN),
        replace_regex("fattn.cpp", r"ggml_sycl_flash_attn_ext_onednn_plan\((?=\s*params)", "ggml_sycl_fattn_routed_plan(", 3),
        replace_once("fattn.cpp", "if (!safe_decode && g_sycl_fa_onednn_enabled && !g_sycl_paged_v2_enabled) {",
                     "if (!safe_decode && " + GATE_DEFAULT + ") {"),
        replace_once("fattn.cpp", "if (d512_onednn_enabled && g_sycl_fa_onednn_enabled && !g_sycl_paged_v2_enabled) {",
                     "if (d512_onednn_enabled && " + GATE_D512 + ") {"),
        append_to("fattn.cpp", B1_FATTN_TAIL),
        append_to("fattn-onednn.cpp", B1_ONEDNN_TAIL),
        append_to("common.cpp", B1_COMMON_TAIL),
        append_to("unified-cache.cpp", B1_LOADFILL),
        replace_once("ggml-sycl.cpp", '    const char * prefix_k = "cache_k_l";\n    const char * prefix_v = "cache_v_l";\n'
                     "    if (strncmp(name, prefix_k, 9) == 0) {\n        layer_id = atoi(name + 9);\n"
                     "    } else if (strncmp(name, prefix_v, 9) == 0) {\n        layer_id = atoi(name + 9);\n    }\n",
                     "    layer_id = ggml_sycl_kv_cache_layer_of(name);\n"),
    ) + (() if keep_test_reads else (
        replace_regex("tests/test-sycl-fattn-onednn-gates.cpp", r'std::getenv\("GGML_SYCL_FLASH_ATTN_EXT"\)', "nullptr", 2),))
    for st in steps:
        files = st(files)
    return files


def b2_tree(files):
    return replace_once("unified-cache.cpp", B1_LOADFILL, B2_VALUEFN)(b1_tree(files))


def matrix_cases_s2d1():
    """Clauses (i)-(o) and witness 9 (S2d-1). Every FAIL case plants a violation; every PASS control either plants the
    compliant shape or edits the real tree so that the subject is present and active (`active`), or asserts the clause is
    still dormant (`dormant`)."""
    c = []
    A = c.append
    TP = "zz-plant.cpp"
    RET = "void onednn_w_retry_lost_cas() {\n}\n"
    ACC = "bool ggml_sycl_replan_token_held(ggml_sycl_replan_kind kind) {\n    return true;\n}\n"
    # 28 / 31: the pre-L0 retry and the busy branch leave once L0's accessor is defined
    A(Case("28", "the accessor defined while the W retry still exists", plant(ACC + RET), "FAIL", "I-RETRY", "onednn_w_retry_lost_cas"))
    A(Case("28", "the accessor with its default argument, split across lines, while the retry exists", plant(
        "bool\nggml_sycl_replan_token_held(\n    ggml_sycl_replan_kind kind = ggml_sycl_replan_kind::ANY) {\n    return true;\n}\n" + RET),
        "FAIL", "I-RETRY", "onednn_w_retry_lost_cas"))
    A(Case("28", "the accessor qualified by a namespace, defined out of line", plant(
        "bool ggml_sycl::ggml_sycl_replan_token_held(ggml_sycl_replan_kind kind) {\n    return true;\n}\n" + RET),
        "FAIL", "I-RETRY", "onednn_w_retry_lost_cas"))
    A(Case("28", "the same tree with the retry deleted (control)", plant(ACC), "PASS", active="i"))
    A(Case("28", "the retry with no accessor: dormant (control)", plant(RET), "PASS", dormant="i"))
    A(Case("28", "the accessor planted only as a call site, with the retry (control)", plant(
        "void zzplant_i() {\n    (void) ggml_sycl_replan_token_held(ggml_sycl_replan_kind::ANY);\n}\n" + RET), "PASS", dormant="i"))
    A(Case("28", "the accessor planted only as a ;-terminated declaration, with the retry (control)", plant(
        "bool ggml_sycl_replan_token_held(ggml_sycl_replan_kind kind = ggml_sycl_replan_kind::ANY);\n" + RET), "PASS", dormant="i"))
    A(Case("28", "the accessor spelled as a lambda variable, a shape the matcher misses: the latch fails", plant(
        "static auto ggml_sycl_replan_token_held = [](int kind) { return kind != 0; };\n" + RET), "FAIL", "X-LATCH", "latch:i"))
    A(Case("28", "the accessor spelled as a macro: the latch fails", plant(
        "#define ggml_sycl_replan_token_held(kind) true\n" + RET), "FAIL", "X-LATCH", "latch:i", planted=False))
    A(Case("31", "the accessor defined while A's pre-L0 busy return still exists", plant(
        ACC + "void onednn_pp_a_relock_busy_pre_l0() {\n}\n"), "FAIL", "I-RETRY", "onednn_pp_a_relock_busy_pre_l0"))
    A(Case("31", "the busy branch with no accessor: dormant (control)", plant(
        "void onednn_pp_a_relock_busy_pre_l0() {\n}\n"), "PASS", dormant="i"))
    A(Case("31", "the accessor with both retired functions deleted (control)", plant(ACC), "PASS", active="i"))
    # 30: §B's COMPLETE reap in the transaction retires the pre-§B interim
    REAP = ("void ggml_sycl_run_runtime_context_transaction() {\n    release_retained_referencing(h, "
            "{ RETAINED_REAP_EVENTS_COMPLETE_BY_CALLER });\n}\n")
    INTERIM = "bool onednn_pp_a_reclaim_query_interim() {\n    return false;\n}\n"
    A(Case("30", "the COMPLETE reap planted in the transaction while the interim exists", plant(REAP + INTERIM),
           "FAIL", "K-INTERIM", "onednn_pp_a_reclaim_query_interim"))
    A(Case("30", "the same tree with the interim deleted (control)", plant(REAP), "PASS", active="k"))
    A(Case("30", "today's tree: the interim and no reap (control)", plant(INTERIM), "PASS", dormant="k"))
    A(Case("30", "a reap of another mode in the transaction, with the interim (control)", plant(
        "void ggml_sycl_run_runtime_context_transaction() {\n    release_retained_referencing(h, { RETAINED_REAP_NONE });\n}\n" + INTERIM),
        "PASS", dormant="k"))
    A(Case("30", "the COMPLETE reap in another function, with the interim (control)", plant(
        "void zzplant_k() {\n    release_retained_referencing(h, { RETAINED_REAP_EVENTS_COMPLETE_BY_CALLER });\n}\n" + INTERIM),
        "PASS", dormant="k"))
    A(Case("30", "the mode held in a local, a shape the matcher misses: the latch fails", plant(
        "void ggml_sycl_run_runtime_context_transaction() {\n    auto mode = RETAINED_REAP_EVENTS_COMPLETE_BY_CALLER;\n"
        "    release_retained_referencing(h, mode);\n}\n" + INTERIM), "FAIL", "X-LATCH", "latch:k"))
    # 29: the owner count never consumes a shared handle
    ALLOW = ("L-CALLER", TP, "zzplant_l")
    A(Case("29", "(a) a free controlled by the count in an allowlisted function", plant(
        "void zzplant_l(mem_handle & h) {\n    if (h.owner_use_count() == 1) unified_free(h.get());\n}\n"),
        "FAIL", "L-FREE", "zzplant_l", edit_allowlist=with_allow(ALLOW)))
    A(Case("29", "(b) the same laundered through a local bool", plant(
        "void zzplant_l(mem_handle & h) {\n    const bool sole = h.owner_use_count() == 1;\n    (void) h;\n    if (sole) zone_free(z, p);\n}\n"),
        "FAIL", "L-FREE", "zone_free", edit_allowlist=with_allow(ALLOW)))
    A(Case("29", "(b) laundered through an integer and a second local", plant(
        "void zzplant_l(mem_handle & h) {\n    size_t n = h.owner_use_count();\n    bool shared = n > 1;\n"
        "    if (!shared) {\n        z.reset();\n    }\n}\n"), "FAIL", "L-FREE", "reset", edit_allowlist=with_allow(ALLOW)))
    A(Case("29", "(b) a free after a count-controlled early return", plant(
        "void zzplant_l(mem_handle & h) {\n    if (h.owner_use_count() > 1) {\n        return;\n    }\n    zone_free(z, p);\n}\n"),
        "FAIL", "L-FREE", "zone_free", edit_allowlist=with_allow(ALLOW)))
    A(Case("29", "(b) a free on the right of a && whose left reads the count", plant(
        "bool zzplant_l(mem_handle & h) {\n    return h.owner_use_count() == 1 && unified_free(h.get());\n}\n"),
        "FAIL", "L-FREE", "unified_free", edit_allowlist=with_allow(ALLOW)))
    A(Case("29", "(b) an enqueue_deferred_zone_free under the count", plant(
        "void zzplant_l(mem_handle & h) {\n    if (h.owner_use_count() == 1) {\n        enqueue_deferred_zone_free(z, p);\n    }\n}\n"),
        "FAIL", "L-FREE", "enqueue_deferred_zone_free", edit_allowlist=with_allow(ALLOW)))
    A(Case("29", "(c) replace_within without its guard call", plant(
        "mem_handle replace_within(mem_handle & old) {\n    return mem_handle();\n}\n"), "FAIL", "L-GUARD", "replace_within"))
    A(Case("29", "(c) replace_within with the guard moved below its first statement", plant(
        "mem_handle replace_within(mem_handle & old) {\n    int n = 0;\n    replace_within_count_guard(old);\n    return mem_handle();\n}\n"),
        "FAIL", "L-GUARD", "replace_within"))
    A(Case("29", "(c) replace_within whose first statement guards another handle", plant(
        "mem_handle replace_within(mem_handle & old) {\n    replace_within_count_guard(other);\n    return mem_handle();\n}\n"),
        "FAIL", "L-GUARD", "replace_within"))
    A(Case("29", "(c) replace_within with the guard first (control)", plant(
        "mem_handle replace_within(mem_handle & old) {\n    // the one count-selected consumer re-checks first\n"
        "    replace_within_count_guard(old);\n    return mem_handle();\n}\n"), "PASS", active="l"))
    A(Case("29", "(d) a new caller of owner_use_count outside the allowlist", plant(
        "bool zzplant_l(mem_handle & h) {\n    return h.owner_use_count() > 1;\n}\n"), "FAIL", "L-CALLER", "zzplant_l"))
    A(Case("29", "an allowlisted caller selecting between replace_within and the retired list on the count (control)", plant(
        "void zzplant_l(mem_handle & h, std::vector<mem_handle> & retired) {\n    if (h.owner_use_count() > 1) {\n"
        "        h = replace_within(h);\n    } else {\n        retired.push_back(std::move(h));\n    }\n}\n"),
        "PASS", active="l", edit_allowlist=with_allow(ALLOW)))
    A(Case("29", "an allowlisted teardown check that drops the handle unconditionally after it (control)", plant(
        "void zzplant_l(mem_handle & h) {\n    const size_t n = h.owner_use_count();\n    if (n != 1) {\n"
        "        GGML_LOG_WARN(\"shared at teardown\\n\");\n    }\n    h = mem_handle();\n}\n"),
        "PASS", active="l", edit_allowlist=with_allow(ALLOW)))
    A(Case("29", "owner_use_count named as a pointer to member, a shape the matcher misses: the latch fails", plant(
        "void zzplant_l() {\n    auto pm = &mem_handle::owner_use_count;\n    (void) pm;\n}\n"), "FAIL", "X-LATCH", "latch:l"))
    # 27: per-model sources only
    A(Case("27", "g_tensor_inventory_detail read in A's fit", plant(
        "size_t onednn_pp_a_bytes(const model_inventory & inv) {\n    return g_tensor_inventory_detail.size();\n}\n"),
        "FAIL", "J-SOURCE", "g_tensor_inventory_detail"))
    A(Case("27", "the planned weight slot getter read in A's fit", plant(
        "size_t onednn_pp_a_bytes(int dev) {\n    return unified_cache_get_planned_pp_moe_onednn_weight_slot_bytes(dev);\n}\n"),
        "FAIL", "J-SOURCE", "unified_cache_get_planned_pp_moe_onednn_weight_slot_bytes"))
    A(Case("27", "the planned oneDNN scratchpad getter read in W's term", plant(
        "size_t onednn_pp_w_bytes(int dev) {\n    return unified_cache_get_planned_onednn_scratchpad_bytes(dev);\n}\n"),
        "FAIL", "J-SOURCE", "unified_cache_get_planned_onednn_scratchpad_bytes"))
    A(Case("27", "zone_is_onednn_reorder_eligible called in the MUL_MAT selector", plant(
        "bool zzplant_select(const zone_tensor_desc & t, size_t n) {\n    return zone_is_onednn_reorder_eligible(t, n);\n}\n"),
        "FAIL", "J-DISPATCH", "zzplant_select"))
    A(Case("27", "A's fit reading the model's own inventory only (control)", plant(
        "size_t onednn_pp_a_bytes(const model_inventory & inv) {\n    return inv.max_weight_bytes();\n}\n"), "PASS", active="j"))
    A(Case("27", "A's fit spelled as a lambda variable: the latch fails", plant(
        "static auto onednn_pp_a_bytes = [](int dev) { return g_tensor_inventory_detail.size(); };\n"), "FAIL", "X-LATCH", "latch:j"))
    A(Case("27", "the classifier's own call is the allowlisted node: it moved to another function", chain(
        replace_once("zone-sizing.cpp", "path_scoped_maxima zone_scoped_maxima(", "path_scoped_maxima zone_scoped_maxima_zz("),
        ), "FAIL", "allowlist", "E-J-CLASSIFIER matches nothing", planted=False))
    # 32: the SCRATCH floor list
    A(Case("32", "appendix row 106 dropped from the floor list", plant("void zzplant_m() {\n}\n"), "FAIL", "M-SCRATCH", "appendix row 106",
           m_edit=m_without("floor", 106)))
    A(Case("32", "row 62 dropped from the covered-by-peak table: neither listed, covered nor excluded", plant("void zzplant_m() {\n}\n"),
           "FAIL", "M-SCRATCH", "appendix row 62", m_edit=m_without("covered", 62)))
    A(Case("32", "row 61, the peak that covers row 62, dropped from the floor list", plant("void zzplant_m() {\n}\n"), "FAIL", "M-SCRATCH",
           "appendix row 61", m_edit=m_without("floor", 61)))
    A(Case("32", "a floor row that is not a SCRATCH row of the appendix", plant("void zzplant_m() {\n}\n"), "FAIL", "M-STALE", "floor row 5",
           m_edit=m_set("floor", 5, "beni")))
    A(Case("32", "a floor row the appendix does not have", plant("void zzplant_m() {\n}\n"), "FAIL", "M-STALE", "floor row 999",
           m_edit=m_set("floor", 999, "beni")))
    A(Case("32", "a covered row whose peak left the list", plant("void zzplant_m() {\n}\n"), "FAIL", "M-STALE", "row 62 is covered",
           m_edit=chain_m(m_set("covered", 62, 17), m_without("floor", 17))))
    A(Case("32", "the SCRATCH floor deleted from ensure_planned_arena_zones while rows still draw", replace_once(
        "unified-cache.cpp", 'const char * arena_mb_env = std::getenv("GGML_SYCL_COMPUTE_ARENA_MB");',
        "const char * arena_mb_env = nullptr;"), "FAIL", "M-FLOOR", "does not apply"))
    A(Case("32", "the unmutated tables and floor (control)", plant("void zzplant_m() {\n}\n"), "PASS", planted=False))
    return c


def chain_m(*edits):
    def f(t):
        for e in edits:
            e(t)
    return f


def matrix_cases_s2d1b():
    """Clauses (n) and (o) and witness 9."""
    c = []
    A = c.append
    CALL = "DnnlGemmWrapper::row_gemm(ctx, 1, 2, 3, a, t, b, t, d, t, q, 4)"
    TN = "tests/zz-n.cpp"

    def body(stmt, name="zzplant_n"):
        return "void %s(ggml_backend_sycl_context & ctx) {\n    %s\n}\n" % (name, stmt)

    # 35: a declined result in the library sources
    A(Case("35", "row_gemm's result discarded as an expression statement", plant(body(CALL + ";")), "FAIL", "N-VOID", "DnnlGemmWrapper::row_gemm"))
    A(Case("35", "row_gemm's result cast to void", plant(body("(void) " + CALL + ";")), "FAIL", "N-VOID", "cast to void"))
    A(Case("35", "row_gemm's result cast with static_cast<void>", plant(body("static_cast<void>(" + CALL + ");")), "FAIL", "N-VOID", "DnnlGemmWrapper::row_gemm"))
    A(Case("35", "row_gemm's result as the left operand of a comma", plant(body("(" + CALL + ", 0);")), "FAIL", "N-VOID", "DnnlGemmWrapper::row_gemm"))
    A(Case("35", "woq_gemm_q4_0 discarded", plant(body("DnnlGemmWrapper::woq_gemm_q4_0(ctx, 1, 2, 3, a, t, b, s, z, 4, c, t, q);")),
           "FAIL", "N-VOID", "DnnlGemmWrapper::woq_gemm_q4_0"))
    A(Case("35", "get_scratchpad_mem discarded, a unique name matched anywhere", plant(body("get_scratchpad_mem(scratchpad_md, eng, ptr);")),
           "FAIL", "N-VOID", "get_scratchpad_mem"))
    A(Case("35", "ggml_sycl_mul_mat_batched_sycl discarded", plant(body("ggml_sycl_mul_mat_batched_sycl(ctx, s0, s1, dst);")),
           "FAIL", "N-VOID", "ggml_sycl_mul_mat_batched_sycl"))
    A(Case("35", "a bare softmax discarded inside DnnlSoftmaxWrapper", plant(
        "struct DnnlSoftmaxWrapper {\n    [[nodiscard]] static int softmax(int a);\n    static void zzplant_in(int a) {\n        softmax(a);\n    }\n};\n"),
        "FAIL", "N-VOID", "DnnlSoftmaxWrapper::softmax"))
    A(Case("35", "a bare softmax consumed inside DnnlSoftmaxWrapper (control)", plant(
        "struct DnnlSoftmaxWrapper {\n    [[nodiscard]] static int softmax(int a);\n    static int zzplant_in(int a) {\n        return softmax(a);\n    }\n};\n"),
        "PASS"))
    A(Case("35", "the result assigned, returned, tested and asserted (control)", plant(
        "sycl::event zzplant_n(ggml_backend_sycl_context & ctx) {\n    sycl::event e = " + CALL + ";\n    e = " + CALL + ";\n"
        "    if (DnnlGemmWrapper::woq_gemm_q4_0(ctx, 1, 2, 3, a, t, b, s, z, 4, c, t, q)) {\n        e.wait();\n    }\n"
        "    GGML_ASSERT(DnnlGemmWrapper::woq_gemm_q4_0(ctx, 1, 2, 3, a, t, b, s, z, 4, c, t, q));\n    return " + CALL + ";\n}\n"), "PASS"))
    A(Case("35", "dpct::gemm, UnifiedKernel::softmax and an unqualified oneMath gemm, all discarded (control)", plant(
        body("dpct::gemm(q, a, b);\n    UnifiedKernel::softmax(x);\n    gemm(q, a, b);\n    Other::row_gemm(q);")), "PASS"))
    A(Case("35", "a bare gemm discarded inside another class (control)", plant(
        "struct Other {\n    void f(int a) {\n        gemm(a);\n    }\n};\n"), "PASS"))
    # 33: the same, in the tests that call a listed name
    A(Case("33", "a test source discarding row_gemm's result", plant(body(CALL + ";"), TN), "FAIL", "N-VOID", "tests/zz-n.cpp"))
    A(Case("33", "a test source casting row_gemm's result to void", plant(body("(void) " + CALL + ";"), TN), "FAIL", "N-VOID", "tests/zz-n.cpp"))
    A(Case("33", "a test source consuming the result (control)", plant(
        "bool zzplant_n(ggml_backend_sycl_context & ctx) {\n    sycl::event e = " + CALL + ";\n    e.wait();\n    return true;\n}\n", TN), "PASS"))
    A(Case("33", "a test source's discard is not hidden by the scope exclusion of tests: clause (a)-(h) findings stay out of it", plant(
        body(CALL + ";\n    sycl::malloc_device<char>(1, q);"), TN), "FAIL", "N-VOID", "tests/zz-n.cpp"))
    A(Case("33", "the one real discard converted to a consumed result, its debt entry dropped (control)", replace_once(
        "tests/test-onednn-woq.cpp", "        DnnlGemmWrapper::row_gemm(*ctx, batch, out_rows, k, act_dev,",
        "        sycl::event zz_ev = DnnlGemmWrapper::row_gemm(*ctx, batch, out_rows, k, act_dev,"),
        "PASS", planted=False, edit_debt=drop_debt("N-VOID", "test-onednn-woq", "row_gemm")))
    A(Case("33", "the same conversion leaves its debt entry stale", replace_once(
        "tests/test-onednn-woq.cpp", "        DnnlGemmWrapper::row_gemm(*ctx, batch, out_rows, k, act_dev,",
        "        sycl::event zz_ev = DnnlGemmWrapper::row_gemm(*ctx, batch, out_rows, k, act_dev,"),
        "FAIL", "debt", "test-onednn-woq"))
    # N-NODISCARD
    A(Case("35", "a listed declaration without [[nodiscard]]", plant("struct DnnlBinaryWrapper {\n    static void binary_broadcast_row(int a);\n};\n"),
           "FAIL", "N-NODISCARD", "DnnlBinaryWrapper::binary_broadcast_row"))
    A(Case("35", "a listed declaration with [[nodiscard]] (control)", plant(
        "struct DnnlBinaryWrapper {\n    [[nodiscard]] static void binary_broadcast_row(int a);\n};\n"), "PASS"))
    A(Case("35", "a listed declaration with [[nodiscard]] split over lines, a pointer return (control)", plant(
        "struct DnnlGemmWrapper {\n    [[nodiscard]]\n    static const void *\n    gemm(int a);\n};\n"), "PASS"))
    A(Case("35", "an unlisted member of a listed class carries no duty (control)", plant(
        "struct DnnlGemmWrapper {\n    static void zz_unlisted(int a) {\n    }\n};\n"), "PASS", planted=False))
    # 36: each C-term consumer submits on its census row's queue
    SC = "ggml-sycl.cpp"
    AC = "bool zzplant_o() {\n    return acquire_onednn_pp_scratch(0, t, 1, 2, &s, &a);\n}\n"
    A(Case("36", "a third acquire_onednn_pp_scratch caller with no census row", plant(AC), "FAIL", "O-NOROW", "acquire_onednn_pp_scratch"))
    A(Case("36", "a second set_rows stage acquire outside the census row", plant(
        "void zzplant_o() {\n    const void * p = ggml_sycl_set_rows_stage_ptr(ctx, plan);\n}\n"), "FAIL", "O-NOROW", "ggml_sycl_set_rows_stage_ptr"))
    A(Case("36", "a graph execute planted in the SDPA file outside the census row", append_to(
        "fattn-onednn.cpp", "void zzplant_o() {\n    dnnl::graph::sycl_interop::execute(cp, s, a, b, d);\n}\n"), "FAIL", "O-NOROW",
        "dnnl::graph::sycl_interop::execute"))
    A(Case("36", "to_fp16_sycl on a queue other than stream, in the A row of ggml_sycl_op_mul_mat_sycl", replace_once(
        SC, "to_fp16_sycl(src0_dd_i, dst_f16, ne, stream);", "to_fp16_sycl(src0_dd_i, dst_f16, ne, ctx.stream());"),
        "FAIL", "O-QUEUE", "to_fp16_sycl", planted=False))
    A(Case("36", "set_rows_validate_indices on another queue", replace_once(
        "set_rows.cpp", "*ctx.stream(plan.owner_device, 0), dst, src1, plan.index_ptr", "*ctx.stream(plan.owner_device, 1), dst, src1, plan.index_ptr"),
        "FAIL", "O-QUEUE", "set_rows_validate_indices", planted=False))
    A(Case("36", "set_rows_sycl's own stream moved off the owner device", replace_once(
        "set_rows.cpp", "dpct::queue_ptr stream = ctx.stream(device, 0);", "dpct::queue_ptr stream = ctx.stream(0, 0);"),
        "FAIL", "O-QUEUE", "set_rows_sycl", planted=False))
    A(Case("36", "f32_to_fp16 in the unified PP arm on another queue", replace_once(
        SC, "f32_to_fp16(src1_data, activations_scratch, src1_elems, ctx.stream());",
        "f32_to_fp16(src1_data, activations_scratch, src1_elems, ctx.stream(ctx.device, 1));"), "FAIL", "O-QUEUE", "f32_to_fp16", planted=False))
    A(Case("36", "the SDPA's dnnl stream built from another stream", replace_once(
        "fattn-onednn.cpp", "dnnl::stream dnnl_stream = ctx.stream_dnnl(stream);", "dnnl::stream dnnl_stream = ctx.stream_dnnl(ctx.stream());"),
        "FAIL", "O-QUEUE", "dnnl_stream", planted=False))
    A(Case("36", "a comment and a string that spell the acquire call and the graph execute (control)", plant(
        "void zzplant_o() {\n    // acquire_onednn_pp_scratch(0, t, 1, 2, &s, &a);\n    const char * s = \"dnnl::graph::sycl_interop::execute(\";\n}\n"),
        "PASS"))
    A(Case("36", "a gemm.hpp-style dnnl::sycl_interop::execute is not the graph execute (control)", plant(
        "void zzplant_o() {\n    // acquire_onednn_pp_scratch is the token this file mentions\n"
        "    dnnl::sycl_interop::execute(prim, stream, args, deps);\n}\n"), "PASS"))
    A(Case("36", "an unmodified tree: every census row holds (control)", lambda f: f, "PASS", planted=False))
    # 9: the model-shaped *_bytes() functions
    DEF = "size_t woq_packed_bytes(int n, int k) {\n    return (size_t) n * k / 2;\n}\n"
    A(Case("9", "woq_packed_bytes defined, but its allocation site does not call it", plant(DEF), "FAIL", "Z9-SITE", "woq_packed_bytes"))
    A(Case("9", "woq_packed_bytes defined, but no zone sizing calls it", plant(DEF), "FAIL", "Z9-SIZING", "woq_packed_bytes"))
    A(Case("9", "woq_packed_bytes called by its site and by the zone sizing (control)", chain(
        plant(DEF),
        replace_once("gemm.hpp", "                                   int64_t                     c_stride1) {\n"
                     "        if (!a || !b_data || !scales || !zero_points || !c) {\n",
                     "                                   int64_t                     c_stride1) {\n        (void) woq_packed_bytes(m, k);\n"
                     "        if (!a || !b_data || !scales || !zero_points || !c) {\n"),
        replace_once("zone-sizing.cpp", "    const std::map<zone_group_key, size_t> freq = zone_group_frequencies(inventory);\n\n    path_scoped_maxima maxima;",
                     "    const std::map<zone_group_key, size_t> freq = zone_group_frequencies(inventory);\n    (void) woq_packed_bytes(1, 1);\n\n"
                     "    path_scoped_maxima maxima;")), "PASS", active="9"))
    A(Case("9", "woq_packed_bytes only called, never defined: dormant (control)", plant(
        "void zzplant_9() {\n    (void) woq_packed_bytes(1, 1);\n}\n"), "PASS", dormant="9"))
    A(Case("9", "woq_packed_bytes spelled as a lambda variable: the latch fails", plant(
        "static auto woq_packed_bytes = [](int n, int k) { return (size_t) n * k; };\n"), "FAIL", "X-LATCH", "latch:9"))
    return c


def matrix_cases_s2d3():
    """The S2d review fold: each case plants a respelling or a conformant shape the first S2d matrix did not."""
    c = []
    A = c.append
    CALL = "DnnlGemmWrapper::row_gemm(ctx, 1, 2, 3, a, t, b, t, d, t, q, 4)"

    def body(stmt, name="zzplant_n"):
        return "void %s(ggml_backend_sycl_context & ctx) {\n    %s\n}\n" % (name, stmt)

    def b1(*muts):
        return chain(b1_tree, *muts)

    def b2(*muts):
        return chain(b2_tree, *muts)

    ROUTED_CALL = "if (!ggml_sycl_fattn_onednn_routed(params, f.head_dim_v, false, nullptr, nullptr)) {"
    HATCH = "    if (p.ne00 == 512 && !d512_onednn_enabled) {\n        return false;\n    }\n"
    SHAPE_CALL = "    const fattn_params p = ggml_sycl_fattn_shape_of(dst);\n"
    DECLINE = ("    if (ggml_sycl_onednn_graph_dispatch_declined(ctx, p)) {\n        out.stage = GGML_SYCL_FATTN_ONEDNN_ROUTE_STAGE_DECLINED;\n"
               "        unified_cache_count_onednn_graph_mask_declined(ctx.device, static_cast<int>(site));\n        return false;\n    }\n")
    # I-1: the value function's predicate calls, in the design's argument order
    A(Case("37", "the value function calling the predicate a second time with a context (b2)", b2(replace_once(
        "unified-cache.cpp", "    return d512;\n",
        "    (void) ggml_sycl_fattn_onednn_routed(params, 0, false, &ctx, nullptr);\n    return d512;\n")),
        "FAIL", "P-ROUTE", "value-predicate", planted=False))
    A(Case("37", "the value function making no call of the predicate (b2)", b2(replace_once(
        "unified-cache.cpp", ROUTED_CALL, "if (false) {")), "FAIL", "P-ROUTE", "makes no call", planted=False))
    A(Case("37", "the value function's predicate call with the design's order and nullptr, nullptr last (control)", b2(),
           "PASS", planted=False, active="p-charge"))
    # I-4: the decline read comes before the routed call
    A(Case("37", "the routing function with no read of the decline", b1(replace_once("fattn.cpp", DECLINE, "")),
           "FAIL", "P-ROUTE", "decline-read", planted=False))
    A(Case("37", "the routing function calling the predicate before it reads the decline", b1(replace_once(
        "fattn.cpp", DECLINE, "    (void) ggml_sycl_fattn_onednn_routed(p, d_v, multi_seq, &ctx, &out.plan);\n" + DECLINE)),
        "FAIL", "P-ROUTE", "decline-order", planted=False))
    A(Case("37", "the routing function with the design's decline-first body (control)", b1(), "PASS", planted=False, active="p-route"))
    # I-2 / I-3: the hatch is latched in a function-local static; only that exact text is exempt
    A(Case("38", "the hatch in its nested-if form (control)", b1(replace_once(
        "fattn.cpp", HATCH, "    if (p.ne00 == 512) {\n        if (!d512_onednn_enabled) {\n            return false;\n        }\n    }\n")),
        "PASS", planted=False, active="p-home"))
    A(Case("38", "the hatch with its operands swapped (control)", b1(replace_once(
        "fattn.cpp", HATCH, "    if (!d512_onednn_enabled && p.ne00 == 512) {\n        return false;\n    }\n")),
        "PASS", planted=False, active="p-home"))
    A(Case("38", "a head dim literal OR-ed into the hatch condition", b1(replace_once(
        "fattn.cpp", HATCH, "    if (p.ne00 == 512 && !d512_onednn_enabled || p.ne00 == 80) {\n        return false;\n    }\n")),
        "FAIL", "P-HOME", "head-dim-literal", planted=False))
    A(Case("38", "a head dim literal in the nested hatch's outer condition", b1(replace_once(
        "fattn.cpp", HATCH, "    if (p.ne00 == 512 || p.ne00 == 80) {\n        if (!d512_onednn_enabled) {\n            return false;\n        }\n    }\n")),
        "FAIL", "P-HOME", "head-dim-literal", planted=False))
    # m-3: the other spellings of a support clause in the supported / route bodies
    for label, text in (
            ("a switch over a GGML_TYPE case", "    switch (dst->src[0]->type) {\n        case GGML_TYPE_F16:\n            return false;\n        default:\n            break;\n    }\n"),
            ("a case of a head dim literal", "    switch (p.ne00) {\n        case 64:\n            return false;\n        default:\n            break;\n    }\n"),
            ("a head dim compared with a named constant", "    constexpr int kD = 80;\n    if (p.ne00 == kD) {\n        return false;\n    }\n"),
            ("a head dim reduced modulo a literal", "    if (p.ne00 % 64 != 0) {\n        return false;\n    }\n"),
            ("an fp8 type test", "    if (ggml_sycl_type_is_fp8_e4m3(dst->src[0]->type)) {\n        return false;\n    }\n")):
        A(Case("38", "%s in ggml_sycl_flash_attn_ext_supported" % label, b1(replace_once("fattn.cpp", SHAPE_CALL, SHAPE_CALL + text)),
               "FAIL", "P-HOME", "ggml_sycl_flash_attn_ext_supported", planted=False))
    # I-6: the charge side's head-dim literals, in the old code's spelling and as extra D512 comparisons
    for label, text in (("dst->src[0]->ne[0] == 80", "        if (dst->src[0]->ne[0] == 80) {\n            continue;\n        }\n"),
                        ("f.head_dim_k != 80", "        if (f.head_dim_k != 80) {\n            continue;\n        }\n"),
                        ("src0->ne[0] != 128", "        if (src0->ne[0] != 128) {\n            continue;\n        }\n")):
        A(Case("38", "the charge side comparing %s" % label, b1(replace_once(
            "unified-cache.cpp", "        " + ROUTED_CALL, text + "        " + ROUTED_CALL)), "FAIL", "P-CHARGE", "literal", planted=False))
    A(Case("38", "a second D512 comparison that is not an if-line in the value function (b2)", b2(replace_once(
        "unified-cache.cpp", "        if (params.ne00 == 512) {\n            ++d512;\n        }\n",
        "        const bool zz512 = params.ne00 == 512;\n        if (params.ne00 == 512) {\n            ++d512;\n        }\n")),
        "FAIL", "P-CHARGE", "d512-count", planted=False))
    # I-5: a result consumed by nothing, in the expression forms the walk must follow
    A(Case("35", "row_gemm's result as the right operand of a discarded &&", plant(body("ok && " + CALL + ";")), "FAIL", "N-VOID", "DnnlGemmWrapper::row_gemm"))
    A(Case("35", "row_gemm's result as the right operand of a discarded ||", plant(body("ok || " + CALL + ";")), "FAIL", "N-VOID", "DnnlGemmWrapper::row_gemm"))
    A(Case("35", "row_gemm's result as an arm of a discarded ?:", plant(body("ok ? " + CALL + " : sycl::event();")), "FAIL", "N-VOID", "DnnlGemmWrapper::row_gemm"))
    A(Case("35", "row_gemm's result as a for-loop increment", plant(body("for (int i = 0; i < 2; " + CALL + ") {\n    }")), "FAIL", "N-VOID", "DnnlGemmWrapper::row_gemm"))
    A(Case("35", "row_gemm reached through a using alias of its class, discarded", plant(
        "using W = DnnlGemmWrapper;\nvoid zzplant_n(ggml_backend_sycl_context & ctx) {\n    W::row_gemm(ctx, 1, 2, 3, a, t, b, t, d, t, q, 4);\n}\n"),
        "FAIL", "N-VOID", "row_gemm"))
    A(Case("35", "row_gemm reached through a typedef of its class, discarded", plant(
        "typedef DnnlGemmWrapper W2;\nvoid zzplant_n(ggml_backend_sycl_context & ctx) {\n    W2::row_gemm(ctx, 1, 2, 3, a, t, b, t, d, t, q, 4);\n}\n"),
        "FAIL", "N-VOID", "row_gemm"))
    A(Case("35", "row_gemm's result consumed through &&, ?: and a call argument (control)", plant(
        "bool zzplant_n(ggml_backend_sycl_context & ctx) {\n    const bool a1 = ok && " + CALL + ";\n    const auto a2 = ok ? " + CALL + " : sycl::event();\n"
        "    consume(" + CALL + ");\n    return ok && " + CALL + ";\n}\n"), "PASS"))
    # m-5: a listed twin beside the unlisted member
    A(Case("35", "an unlisted member beside a listed, consumed twin: the listed one is checked, the other is not (control)", plant(
        "struct DnnlGemmWrapper {\n    static void zz_unlisted(int a) {\n        zz_other(a);\n    }\n    [[nodiscard]] static int gemm(int a);\n"
        "    static int zz_use(int a) {\n        return gemm(a);\n    }\n};\n"), "PASS", planted=False, active="n"))
    A(Case("35", "the listed twin without [[nodiscard]] beside an unlisted member", plant(
        "struct DnnlGemmWrapper {\n    static void zz_unlisted(int a) {\n    }\n    static int gemm(int a);\n};\n"),
        "FAIL", "N-NODISCARD", "DnnlGemmWrapper::gemm"))
    # I-8 (a): a subject named as a type is not invisible
    RET = "void onednn_w_retry_lost_cas() {\n}\n"
    for label, text in (("a using alias of a function pointer", "using ggml_sycl_replan_token_held = bool (*)(int);\n"),
                        ("a typedef of a function pointer", "typedef bool (*ggml_sycl_replan_token_held)(int);\n"),
                        ("a functor struct", "struct ggml_sycl_replan_token_held {\n    bool operator()(int) const;\n};\n")):
        A(Case("28", "the accessor spelled as %s: the latch fails" % label, plant(text + RET), "FAIL", "X-LATCH", "latch:i", planted=False))
    # I-8 (c): a retired function spelled where the definition matcher cannot see it
    ACC = "bool ggml_sycl_replan_token_held(ggml_sycl_replan_kind kind) {\n    return true;\n}\n"
    A(Case("28", "the retired retry spelled as a lambda variable beside the accessor: the latch fails", plant(
        ACC + "static auto onednn_w_retry_lost_cas = []() {};\n"), "FAIL", "X-LATCH", "latch:i", planted=False))
    A(Case("28", "the retired retry spelled as a macro beside the accessor: the latch fails", plant(
        ACC + "#define onednn_w_retry_lost_cas() 0\n"), "FAIL", "X-LATCH", "latch:i", planted=False))
    A(Case("31", "the retired busy return spelled as a lambda variable beside the accessor: the latch fails", plant(
        ACC + "static auto onednn_pp_a_relock_busy_pre_l0 = []() {};\n"), "FAIL", "X-LATCH", "latch:i", planted=False))
    # I-8 (d): the selector's eligibility call spelled as a macro or taken by address
    A(Case("27", "zone_is_onednn_reorder_eligible spelled as a macro: the latch fails", plant(
        "#define ELIG(t) zone_is_onednn_reorder_eligible(t)\n"), "FAIL", "X-LATCH", "latch:j", planted=False))
    A(Case("27", "zone_is_onednn_reorder_eligible taken by address: the latch fails", plant(
        "bool zzplant_select() {\n    auto f = &zone_is_onednn_reorder_eligible;\n    return f != nullptr;\n}\n"), "FAIL", "X-LATCH", "latch:j", planted=False))
    # I-8 (b): clause (k)'s subjects spelled where its matcher cannot see them
    INTERIM = "bool onednn_pp_a_reclaim_query_interim() {\n    return false;\n}\n"
    A(Case("30", "the reap spelled as a macro: the latch fails", plant("#define release_retained_referencing(h, m) 0\n" + INTERIM),
           "FAIL", "X-LATCH", "latch:k", planted=False))
    A(Case("30", "the transaction spelled as a lambda variable: the latch fails", plant(
        "static auto ggml_sycl_run_runtime_context_transaction = []() {};\n" + INTERIM), "FAIL", "X-LATCH", "latch:k", planted=False))
    A(Case("30", "the COMPLETE mode held in a constexpr alias: the latch fails", plant(
        "constexpr auto REAP_NOW = RETAINED_REAP_EVENTS_COMPLETE_BY_CALLER;\n" + INTERIM), "FAIL", "X-LATCH", "latch:k", planted=False))
    A(Case("30", "the COMPLETE mode behind a macro: the latch fails", plant(
        "#define REAP_NOW RETAINED_REAP_EVENTS_COMPLETE_BY_CALLER\n" + INTERIM), "FAIL", "X-LATCH", "latch:k", planted=False))
    # m-1: the sixth allocation site of witness 9
    A(Case("9", "load_reorder_temp_bytes defined, but unified_cache::reserve_reorder_temp does not call it", plant(
        "size_t load_reorder_temp_bytes(int n) {\n    return (size_t) n;\n}\n"), "FAIL", "Z9-SITE", "unified_cache::reserve_reorder_temp"))
    # m-4: an allowlist entry can never exempt an S2d finding that must be fixed
    for code in ("X-LATCH", "P-ROUTE", "P-CHARGE"):
        A(Case("m5", "an allowlist entry for %s is refused" % code, lambda f: f, "FAIL", "allowlist", "never exempted",
               edit_allowlist=with_allow((code, "fattn.cpp", "zz"))))
    return c


def matrix_cases_s2d4():
    """The S2d re-review fold: clause (p)'s other subjects latch, the dispatch body is pinned, clauses (n) and (o) latch, and
    the value walk follows the statement-level wrappers."""
    c = []
    A = c.append
    CALL = "DnnlGemmWrapper::row_gemm(ctx, 1, 2, 3, a, t, b, t, d, t, q, 4)"

    def body(stmt, name="zzplant_n"):
        return "void %s(ggml_backend_sycl_context & ctx) {\n    %s\n}\n" % (name, stmt)

    def b1(*muts):
        return chain(b1_tree, *muts)

    ROUTED_CALL = "if (!ggml_sycl_fattn_onednn_routed(params, f.head_dim_v, false, nullptr, nullptr)) {"
    FLASH_OPEN = "void ggml_sycl_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_sycl::sycl_tensor safe_dst) {\n"
    DECLINE = ("    if (ggml_sycl_onednn_graph_dispatch_declined(ctx, p)) {\n        out.stage = GGML_SYCL_FATTN_ONEDNN_ROUTE_STAGE_DECLINED;\n"
               "        unified_cache_count_onednn_graph_mask_declined(ctx.device, static_cast<int>(site));\n        return false;\n    }\n")
    COUNT = "        unified_cache_count_onednn_graph_mask_declined(ctx.device, static_cast<int>(site));\n"
    # I-A: every clause (p) subject latches, in the shapes the definition matcher cannot see
    for blk, nm, text in (
            ("p-home", "ggml_sycl_flash_attn_ext_supported", "#define ggml_sycl_flash_attn_ext_supported(dst) (dst->src[0]->ne[0] != 80)\n"),
            ("p-home", "ggml_sycl_fattn_shape_supported", "static auto ggml_sycl_fattn_shape_supported = [](const fattn_params & p, int d) { return p.ne00 != 80; };\n"),
            ("p-route", "ggml_sycl_fattn_onednn_route_enabled", "#define ggml_sycl_fattn_onednn_route_enabled(p) ((p).ne00 != 80)\n"),
            ("p-route", "ggml_sycl_fattn_onednn_route_enabled", "static auto ggml_sycl_fattn_onednn_route_enabled = [](const fattn_params & p) { return p.ne00 != 80; };\n"),
            ("p-route", "ggml_sycl_fattn_onednn_dispatch_routed", "static auto ggml_sycl_fattn_onednn_dispatch_routed = [](int x) { return x > 0; };\n"),
            ("p-route", "ggml_sycl_fattn_onednn_routed", "#define ggml_sycl_fattn_onednn_routed(p, d, m, c, o) true\n")):
        A(Case("38", "%s spelled as %s: the latch fails" % (nm, "a macro" if text.startswith("#") else "a lambda variable"),
               b1(append_to("fattn.cpp", text)), "FAIL", "X-LATCH", "latch:%s:%s" % (blk, nm), planted=False))
    A(Case("38", "ggml_sycl_flash_attn_ext_supported renamed away while still called: it is missing", b1(replace_once(
        "fattn.cpp", "bool ggml_sycl_flash_attn_ext_supported(const ggml_tensor * dst) {", "bool ggml_sycl_flash_attn_ext_supported_zz(const ggml_tensor * dst) {")),
        "FAIL", "P-HOME", "supported-missing", planted=False))
    # N1: the hatch spellings the exemption does not recognise fail loudly
    LATCH = "    static const bool d512_onednn_enabled = ggml_sycl_fa_onednn_d512_enabled();\n"
    HATCH = "    if (p.ne00 == 512 && !d512_onednn_enabled) {\n"
    A(Case("38", "the hatch with the literal on the left: 512 == p.ne00 && !V", b1(replace_once(
        "fattn.cpp", HATCH, "    if (512 == p.ne00 && !d512_onednn_enabled) {\n")), "FAIL", "P-HOME", "head-dim-literal", planted=False))
    A(Case("38", "the hatch's latch written with braces: static const bool V{...}", b1(replace_once(
        "fattn.cpp", LATCH + HATCH, "    static const bool d512_onednn_enabled{ggml_sycl_fa_onednn_d512_enabled()};\n" + HATCH)),
        "FAIL", "P-HOME", "head-dim-literal", planted=False))
    A(Case("38", "the hatch's latch written const static", b1(replace_once(
        "fattn.cpp", LATCH + HATCH, "    const static bool d512_onednn_enabled = ggml_sycl_fa_onednn_d512_enabled();\n" + HATCH)),
        "FAIL", "P-HOME", "head-dim-literal", planted=False))
    # M-2: the census-fill pin
    A(Case("38", "a second kv_pair_of call in the dispatch function, off the pinned fill line", b1(insert_after(
        "fattn.cpp", FLASH_OPEN, "    (void) ggml_sycl_fattn_kv_pair_of(params.K_type, params.V_type);\n")),
        "FAIL", "P-HOME", "census-fill", planted=False))
    # M-3: the routing function's body is exactly the design's
    DECL_CALL = "ggml_sycl_onednn_graph_dispatch_declined(ctx, p)"
    for label, mut in (
            ("the decline call discarded in place of its branch", replace_once("fattn.cpp", DECLINE, "    " + DECL_CALL + ";\n")),
            ("the decline call cast to void ahead of the branch", replace_once("fattn.cpp", DECLINE, "    (void) " + DECL_CALL + ";\n" + DECLINE)),
            ("the decline read inside a dead `if (false)`", replace_once(
                "fattn.cpp", DECLINE, "    if (false) {\n        (void) " + DECL_CALL + ";\n    }\n")),
            ("the declined-mask counter call deleted", replace_once("fattn.cpp", COUNT, ""))):
        A(Case("37", "the routing function with %s" % label, b1(mut), "FAIL", "P-ROUTE", "dispatch-body", planted=False))
    # M-4: the charge side's head-dim tests of every kind
    for label, text, kind_ in (
            ("params.ne00 % 64", "        if (params.ne00 % 64 != 0) {\n            continue;\n        }\n", "head-dim-arith"),
            ("a named constant", "        if (f.head_dim_k == kD) {\n            continue;\n        }\n", "head-dim-named"),
            ("a switch over params.ne00", "        switch (params.ne00) {\n            case 64:\n                continue;\n            default:\n                break;\n        }\n",
             "head-dim-case")):
        A(Case("38", "the charge side testing %s" % label, b1(replace_once(
            "unified-cache.cpp", "        " + ROUTED_CALL, text + "        " + ROUTED_CALL)), "FAIL", "P-CHARGE", kind_, planted=False))
    # M-5: the COMPLETE mode behind a ternary, and handed to a macro inside the transaction
    INTERIM = "bool onednn_pp_a_reclaim_query_interim() {\n    return false;\n}\n"
    A(Case("30", "the COMPLETE mode as an arm of a ternary initialiser: the latch fails", plant(
        "constexpr auto REAP_NOW = flag ? RETAINED_REAP_EVENTS_COMPLETE_BY_CALLER : RETAINED_REAP_NONE;\n" + INTERIM),
        "FAIL", "X-LATCH", "latch:k", planted=False))
    A(Case("30", "the COMPLETE mode handed to a macro inside the transaction: the latch fails", plant(
        "void ggml_sycl_run_runtime_context_transaction() {\n    RELEASE_MACRO(h, RETAINED_REAP_EVENTS_COMPLETE_BY_CALLER);\n}\n" + INTERIM),
        "FAIL", "X-LATCH", "latch:k", planted=False))
    # M-1: clauses (n) and (o) latch, and the statement-level wrappers are discards
    A(Case("35", "row_gemm spelled inside a macro: the latch fails", plant(
        "#define RG() " + CALL + "\n"), "FAIL", "X-LATCH", "latch:n", planted=False))
    A(Case("35", "row_gemm taken as a pointer to member: the latch fails", plant(
        "void zzplant_n() {\n    auto fp = &DnnlGemmWrapper::row_gemm;\n    (void) fp;\n}\n"), "FAIL", "X-LATCH", "latch:n", planted=False))
    A(Case("36", "acquire_onednn_pp_scratch spelled inside a macro: the latch fails", plant(
        "#define ACQ() acquire_onednn_pp_scratch(0, t, 1, 2, &s, &a)\n"), "FAIL", "X-LATCH", "latch:o", planted=False))
    A(Case("36", "acquire_onednn_pp_scratch taken by address: the latch fails", plant(
        "void zzplant_o() {\n    auto f = &acquire_onednn_pp_scratch;\n    (void) f;\n}\n"), "FAIL", "X-LATCH", "latch:o", planted=False))
    for label, stmt in (("assigned to std::ignore", "std::ignore = " + CALL + ";"),
                        ("under static_cast<bool> as a statement", "static_cast<bool>(" + CALL + ");"),
                        ("under a ! as a statement", "!" + CALL + ";"),
                        ("compared as a statement", CALL + " == x;"),
                        ("cast to another type as a statement", "(bool) " + CALL + ";")):
        A(Case("35", "row_gemm's result %s" % label, plant(body(stmt)), "FAIL", "N-VOID", "DnnlGemmWrapper::row_gemm"))
    A(Case("35", "row_gemm's result under !, a cast and static_cast<bool>, consumed by a condition and a return (control)", plant(
        "bool zzplant_n(ggml_backend_sycl_context & ctx) {\n    if (!" + CALL + ") {\n        return false;\n    }\n"
        "    const bool b = !" + CALL + ";\n    return static_cast<bool>(" + CALL + ") && b && (bool) " + CALL + ";\n}\n"), "PASS"))
    return c


def matrix_cases_s2d5():
    """The S2d r3 fold: the remaining clause (p) subjects latch, and the value walk follows functional and named casts."""
    c = []
    A = c.append
    CALL = "DnnlGemmWrapper::row_gemm(ctx, 1, 2, 3, a, t, b, t, d, t, q, 4)"

    def body(stmt, name="zzplant_n"):
        return "void %s(ggml_backend_sycl_context & ctx) {\n    %s\n}\n" % (name, stmt)

    for blk, nm, mac, lam in (
            ("p-route", "ggml_sycl_flash_attn_ext_onednn", "#define ggml_sycl_flash_attn_ext_onednn(c, p) 0\n",
             "static auto ggml_sycl_flash_attn_ext_onednn = [](int c, int p) { return 0; };\n"),
            ("p-route", "ggml_sycl_flash_attn_ext_onednn_plan", "#define ggml_sycl_flash_attn_ext_onednn_plan(p, a, b, f, m) 0\n",
             "static auto ggml_sycl_flash_attn_ext_onednn_plan = [](int p) { return 0; };\n"),
            ("p-fill", "ggml_sycl_fattn_shape_of", "#define ggml_sycl_fattn_shape_of(d) fattn_params{}\n",
             "static auto ggml_sycl_fattn_shape_of = [](int d) { return 0; };\n")):
        for what, text in (("a macro", mac), ("a lambda variable", lam)):
            A(Case("38", "%s spelled as %s: the latch fails" % (nm, what), chain(b1_tree, append_to("fattn.cpp", text)),
                   "FAIL", "X-LATCH", "latch:%s:%s" % (blk, nm), planted=False))
    for label, stmt in (("under a bool() functional cast", "bool(" + CALL + ");"), ("under an int() functional cast", "int(" + CALL + ");"),
                        ("under a size_t() functional cast", "size_t(" + CALL + ");"),
                        ("under a reinterpret_cast", "reinterpret_cast<long>(" + CALL + ");"),
                        ("under a const_cast", "const_cast<bool &>(" + CALL + ");"),
                        ("under a dynamic_cast", "dynamic_cast<bool>(" + CALL + ");")):
        A(Case("35", "row_gemm's result %s as a statement" % label, plant(body(stmt)), "FAIL", "N-VOID", "DnnlGemmWrapper::row_gemm"))
    A(Case("35", "row_gemm's result under a void() functional cast", plant(body("void(" + CALL + ");")), "FAIL", "N-VOID", "cast to void"))
    A(Case("35", "row_gemm's result under functional casts, returned and compared (control)", plant(
        "bool zzplant_n(ggml_backend_sycl_context & ctx) {\n    const bool b = int(" + CALL + ") != 0;\n    return b && bool(" + CALL + ");\n}\n"), "PASS"))
    return c


def s2d_witnesses(files, allowlist, debt):
    """The census table is data: with appendix-rows.json missing the gate fails naming M-DATA, never passes."""
    import tempfile
    out = []
    saved = S2D_DATA["dir"]
    real = (Path(saved) if saved else Path(__file__).resolve().parent / "sycl-alloc-zone-contract") / "appendix-rows.json"
    doc = json.loads(real.read_text())
    short = dict(doc, rows=[{k: v for k, v in doc["rows"][0].items() if k != "zone"}] + doc["rows"][1:])
    try:
        for label, text in (("a missing appendix-rows.json", None), ("an appendix-rows.json that is not JSON", "{not json"),
                            ("an appendix-rows.json with a row that lacks its zone", json.dumps(short))):
            with tempfile.TemporaryDirectory() as d:
                if text is not None:
                    (Path(d) / "appendix-rows.json").write_text(text)
                S2D_DATA["dir"] = d
                fails, _, _, _ = run_gate(files, allowlist, debt)
            out.append(("%s fails (M-DATA), it does not pass or raise" % label, any(names_new(f, "M-DATA") for f in fails),
                        "%d finding(s)" % len(fails)))
    finally:
        S2D_DATA["dir"] = saved
    fails, _, _, stats = run_gate(files, allowlist, debt)
    out.append(("the committed appendix-rows.json is read (control)", not fails, "%d finding(s)" % len(fails)))
    for clause in ("j", "l", "p-route"):
        out.append(("the gate's output names what clause (%s) leaves uncovered (TODO line)" % clause,
                    any(l.startswith("TODO %s:" % clause) for l in stats["dormant"]), "%d line(s)" % len(stats["dormant"])))
    return out


def matrix_cases_s3q():
    """S3-0: clause (q), the C library's allocation primitives, and the pin on the one G-CATCH debt entry that carries a fate."""
    c = []
    A = c.append

    def plant_q(expr, name="zzplant_q"):
        return plant("void %s() {\n    void * p = nullptr;\n    (void) %s;\n    (void) p;\n}\n" % (name, expr))

    for nm, call in (("malloc", "std::malloc(16)"), ("malloc", "malloc(16)"), ("malloc", "::malloc(16)"),
                     ("calloc", "std::calloc(4, 4)"), ("calloc", "calloc(4, 4)"), ("realloc", "realloc(nullptr, 16)"),
                     ("realloc", "std::realloc(p, 16)"), ("aligned_alloc", "::aligned_alloc(64, 64)"),
                     ("aligned_alloc", "std::aligned_alloc(64, 64)"), ("posix_memalign", "posix_memalign(&p, 64, 16)"),
                     ("memalign", "memalign(64, 16)"), ("mmap", "mmap(nullptr, 16, 3, 34, -1, 0)"),
                     ("mmap", "::mmap(nullptr, 16, 3, 34, -1, 0)"), ("VirtualAlloc", "VirtualAlloc(nullptr, 16, 0, 0)")):
        A(Case("s3q", "a new call of %s" % call.split("(")[0], plant_q(call), "FAIL", "E-LIBC", nm))
    A(Case("s3q", "the address of std::malloc taken", plant(
        "void zzplant_q() {\n    auto f = &std::malloc;\n    (void) f;\n}\n"), "FAIL", "E-LIBC", "malloc"))
    A(Case("s3q", "malloc used as a value", plant(
        "void zzplant_q() {\n    auto f = malloc;\n    (void) f;\n}\n"), "FAIL", "E-LIBC", "malloc"))
    A(Case("s3q", "a macro whose body calls std::malloc", plant("#define zz_alloc(n) std::malloc(n)\n"),
           "FAIL", "E-LIBC", "#define zz_alloc", planted=False))
    A(Case("s3q", "a macro whose body calls a bare mmap", plant("#define zz_map(n) mmap(nullptr, n, 3, 34, -1, 0)\n"),
           "FAIL", "E-LIBC", "#define zz_map", planted=False))
    A(Case("s3q", "a macro whose body spells a `  ::  calloc` call across spaces", plant("#define zz_calloc(n) ::  calloc(n, 1)\n"),
           "FAIL", "E-LIBC", "#define zz_calloc", planted=False))
    A(Case("s3q", "a call planted in the ERROR-root file cpu-dispatch.cpp is found by the lexical pass", append_to("cpu-dispatch.cpp",
        "void zzplant_q19() {\n    void * p = malloc(16);\n    (void) p;\n}\n"), "FAIL", "E-LIBC", "malloc"))
    A(Case("s3q", "a member call, a pool-qualified call and a method declaration are not hits (control)", plant(
        "struct zz_pool {\n    void * malloc(unsigned long n);\n    void * realloc(void * p, unsigned long n);\n};\n"
        "void * zz_pool::malloc(unsigned long n) { return nullptr; }\n"
        "void zzplant_q(zz_pool & pool, zz_pool * pp) {\n    void * a = pool.realloc(nullptr, 16);\n    void * b = pp->malloc(16);\n"
        "    void * c = zz_pool::malloc(8);\n    (void) a; (void) b; (void) c;\n}\n"), "PASS", planted=False))
    A(Case("s3q", "a declaration of a function named malloc or mmap is not a use (control)", plant(
        "void * malloc(unsigned long n);\nvoid * mmap(void * a, unsigned long n, int p, int f, int fd, long o);\n"
        "void zzplant_q() {\n}\n"), "PASS", planted=False))
    A(Case("s3q", "a comment, a string and longer identifiers containing the names are not hits (control)", plant(
        "// malloc(16) and std::calloc(4, 4) and mmap(0, 1, 2, 3, 4, 5)\nvoid zzplant_q() {\n    int my_malloc = 1;\n    int mallocs = 2;\n"
        "    int calloc_count = 3;\n    const char * s = \"malloc posix_memalign mmap\";\n"
        "    (void) my_malloc; (void) mallocs; (void) calloc_count; (void) s;\n}\n"), "PASS", planted=False))
    A(Case("s3q", "sycl::malloc stays clause (e)'s: one E-RAW entry covers it and no E-LIBC finding appears (control)", plant(
        "void zzplant_q(sycl::queue & q) {\n    void * p = sycl::malloc(16, q, sycl::usm::alloc::host);\n    (void) p;\n}\n"),
        "PASS", allowlist={"id": "E-ZZ-SYCL-MALLOC", "code": "E-RAW", "file": PLANT, "function": "zzplant_q", "name": "malloc",
                           "count": 1, "reason": "mutation-matrix test entry"}, planted=False))
    A(Case("s3q", "an E-LIBC allowlist entry covers its function, name and count (control)", plant_q("std::malloc(16)"), "PASS",
           allowlist={"id": "E-ZZ-LIBC", "code": "E-LIBC", "file": PLANT, "function": "zzplant_q", "name": "malloc", "count": 1,
                      "reason": "mutation-matrix test entry"}, planted=False))
    A(Case("s3q", "an E-LIBC entry pins the primitive: a calloc beside the allowlisted malloc is a finding", plant(
        "void zzplant_q() {\n    void * p = nullptr;\n    (void) std::malloc(16);\n    (void) std::calloc(1, 16);\n    (void) p;\n}\n"),
        "FAIL", "E-LIBC", "calloc", allowlist={"id": "E-ZZ-LIBC", "code": "E-LIBC", "file": PLANT, "function": "zzplant_q",
                                              "name": "malloc", "count": 1, "reason": "mutation-matrix test entry"}, planted=False))
    A(Case("s3q", "an E-LIBC entry pins the count: a second malloc breaks it", plant(
        "void zzplant_q() {\n    (void) std::malloc(16);\n    (void) std::malloc(32);\n}\n"),
        "FAIL", "allowlist", "E-ZZ-LIBC covers 2 finding(s) but pins 1", allowlist={
            "id": "E-ZZ-LIBC", "code": "E-LIBC", "file": PLANT, "function": "zzplant_q", "name": "malloc", "count": 1,
            "reason": "mutation-matrix test entry"}, planted=False))
    A(Case("s3q", "an E-LIBC entry for a function that has no such call matches nothing", plant(
        "void zzplant_q() {\n}\n"), "FAIL", "allowlist", "E-ZZ-LIBC matches nothing", allowlist={
            "id": "E-ZZ-LIBC", "code": "E-LIBC", "file": PLANT, "function": "zzplant_q", "name": "malloc", "count": 1,
            "reason": "mutation-matrix test entry"}, planted=False))
    # the committed entries
    A(Case("s3q", "a second std::malloc in dpct's host_buffer breaks its pin", replace_in_function(
        "dpct/helper.hpp", r"host_buffer\(size_t size", "_buf(std::malloc(size))", "_buf(std::malloc(size)), _zz(std::malloc(size))"),
        "FAIL", "allowlist", "E-LIBC-DPCT-HOSTBUF-MALLOC covers 3 finding(s) but pins 2", planted=False))
    A(Case("s3q", "calloc swapped in for dpct's malloc is a finding of its own", replace_in_function(
        "dpct/helper.hpp", r"host_buffer\(size_t size", "std::malloc(size)", "std::calloc(size, 1)"),
        "FAIL", "E-LIBC", "calloc", planted=False))
    A(Case("s3q", "dpct's mem_mgr constructor renamed leaves its mmap entry matching nothing", replace_in_function(
        "dpct/helper.hpp", r"class mem_mgr\s*\{", "mem_mgr()", "mem_mgr_zz()"),
        "FAIL", "allowlist", "E-LIBC-DPCT-MMGR-MMAP matches nothing", planted=False))
    A(Case("s3q", "a second mmap in cache_guard_allocator breaks its permanent entry", insert_after(
        "unified-cache.hpp", "const size_t mapping_size = usable + page_size;",
        "\n        void * zz_second = mmap(nullptr, mapping_size, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);\n        (void) zz_second;"),
        "FAIL", "allowlist", "E-LIBC-CACHE-GUARD covers 2 finding(s) but pins 1", planted=False))
    A(Case("s3q", "cache_guard_allocator renamed leaves its entry matching nothing", replace_token(
        "unified-cache.hpp", "cache_guard_allocator", "cache_guard_allocator_zz"),
        "FAIL", "allowlist", "E-LIBC-CACHE-GUARD matches nothing", planted=False))
    A(Case("s3q", "weight_cache_allocator's mmap deleted leaves its debt entry stale", replace_in_function(
        "ggml-sycl.cpp", r"template <typename T> struct weight_cache_allocator \{", "mmap(nullptr, mapping_size", "zz_map(nullptr, mapping_size"),
        "FAIL", "debt", "weight_cache_allocator::allocate::call_expression:mmap", planted=False))
    A(Case("s3q", "a second posix_memalign in weight_cache_allocator is a new finding", replace_in_function(
        "ggml-sycl.cpp", r"template <typename T> struct weight_cache_allocator \{", "if (posix_memalign(&base, alignment, aligned_total) != 0 || !base) {",
        "void * zz_b = nullptr;\n        (void) posix_memalign(&zz_b, alignment, aligned_total);\n"
        "        if (posix_memalign(&base, alignment, aligned_total) != 0 || !base) {"),
        "FAIL", "E-LIBC", "posix_memalign", planted=False))

    def libc_debt(edit):
        return lambda d: dict(d, violations=[edit(e) if e["code"] == "E-LIBC" else e for e in d["violations"]])

    A(Case("s3q-data", "an E-LIBC debt entry without a fate fails", lambda f: f, "FAIL", "data", "no valid fate",
           edit_debt=libc_debt(lambda e: {k: v for k, v in e.items() if k != "fate"}), planted=False))
    A(Case("s3q-data", "an E-LIBC debt entry with a made-up fate fails", lambda f: f, "FAIL", "data", "no valid fate",
           edit_debt=libc_debt(lambda e: dict(e, fate="whatever")), planted=False))
    A(Case("s3q-data", "an E-LIBC debt entry without a cite fails", lambda f: f, "FAIL", "data", "has no cite",
           edit_debt=libc_debt(lambda e: {k: v for k, v in e.items() if k != "cite"}), planted=False))
    A(Case("s3q-data", "an E-LIBC debt entry with a one-word cite fails", lambda f: f, "FAIL", "data", "has no cite",
           edit_debt=libc_debt(lambda e: dict(e, cite="tbd")), planted=False))
    A(Case("s3q-data", "an E-LIBC allowlist entry without a name is rejected", lambda f: f, "FAIL", "data", "lacks name",
           edit_allowlist=lambda al: dict(al, entries=[{k: v for k, v in e.items() if k != "name"} if e["id"] == "E-LIBC-CACHE-GUARD" else e
                                                       for e in al["entries"]]), planted=False))

    def catch_debt(edit, which):
        def f(d):
            out, done = [], False
            for e in d["violations"]:
                if not done and which(e):
                    e, done = edit(e), True
                out.append(e)
            return dict(d, violations=out)
        return f

    def is_check_try(e):
        return e["code"] == "G-CATCH" and e["key"] == CHECK_TRY_ERROR_KEY

    def is_other_catch(e):
        return e["code"] == "G-CATCH" and e["key"] != CHECK_TRY_ERROR_KEY

    A(Case("s3q-pin", "the CHECK_TRY_ERROR entry without its fate fails", lambda f: f, "FAIL", "data",
           "must carry fate converted-by-5.4a", edit_debt=catch_debt(lambda e: {k: v for k, v in e.items() if k != "fate"}, is_check_try),
           planted=False))
    for fate in ("deleted-by-step-7", "sanctioned-internal", "pending-disposition", "converted-by-5.4b"):
        A(Case("s3q-pin", "the CHECK_TRY_ERROR entry with fate %s fails" % fate, lambda f: f, "FAIL", "data",
               "must carry fate converted-by-5.4a", edit_debt=catch_debt(lambda e, fate=fate: dict(e, fate=fate), is_check_try),
               planted=False))
    A(Case("s3q-pin", "another G-CATCH entry carrying converted-by-5.4a fails", lambda f: f, "FAIL", "data", "carries a fate",
           edit_debt=catch_debt(lambda e: dict(e, fate="converted-by-5.4a"), is_other_catch), planted=False))
    A(Case("s3q-pin", "an N-VOID entry carrying a fate fails", lambda f: f, "FAIL", "data", "carries a fate",
           edit_debt=catch_debt(lambda e: dict(e, fate="deleted-by-step-7"), lambda e: e["code"] == "N-VOID"), planted=False))
    return c


def matrix_cases_s3f():
    """The S3-0 review fold: a qualifier across spaces, the dpct memcpy family, more libc names, and the data rules."""
    c = []
    A = c.append

    def plant_q(expr, name="zzplant_q"):
        return plant("void %s() {\n    void * p = nullptr;\n    (void) %s;\n    (void) p;\n}\n" % (name, expr))

    # I1: a #define body is text, so a qualifier is read across spaces and line continuations, backwards from the name
    A(Case("s3f-i1", "a macro body `sycl :: malloc` is clause (e)'s one E-RAW hit and no E-LIBC double count (control)", plant(
        "#define zz_sm(q, n) sycl :: malloc(n, q, sycl::usm::alloc::host)\n"), "PASS",
        allowlist={"id": "E-ZZ-SM", "code": "E-RAW", "file": PLANT, "function": "#define zz_sm", "name": "sycl::malloc",
                   "count": 1, "reason": "mutation-matrix test entry"}, planted=False))
    for lab, body in (("pool :: realloc", "#define zz_p(p, n) pool :: realloc(p, n)\n"),
                      ("p . malloc", "#define zz_p(p, n) p . malloc(n)\n"),
                      ("p -> malloc", "#define zz_p(p, n) p -> malloc(n)\n"),
                      ("p->malloc", "#define zz_p(p, n) p->malloc(n)\n"),
                      ("a scope split by a line continuation", "#define zz_p(n) my_ns::\\\n    malloc(n)\n"),
                      ("a scope and a name split by a continuation and spaces", "#define zz_p(n) my_ns  ::  \\\n  calloc(n, 1)\n"),
                      ("a template-qualified name", "#define zz_p(n) zz_pool<int>::malloc(n)\n")):
        A(Case("s3f-i1", "a macro body `%s` is not a libc hit (control)" % lab, plant(body), "PASS", planted=False))
    for lab, body in (("std :: malloc", "#define zz_s(n) std :: malloc(n)\n"), (":: malloc", "#define zz_s(n) :: malloc(n)\n"),
                      ("std ::<continuation> malloc", "#define zz_s(n) std ::\\\n    malloc(n)\n"),
                      ("a bare ::<continuation> calloc", "#define zz_s(n) ::\\\n    calloc(n, 1)\n")):
        A(Case("s3f-i1", "a macro body `%s` is still E-LIBC" % lab, plant(body), "FAIL", "E-LIBC", "#define zz_s", planted=False))
    # the lexical pass of an ERROR-root file reads the qualifier the same way
    for lab, body in (("pool :: realloc", "pool :: realloc(p, 1)"), ("p . malloc", "p . malloc(1)"), ("p -> malloc", "p -> malloc(1)"),
                      ("a scope split by a newline", "my_ns::\n    malloc(1)")):
        A(Case("s3f-i1", "the lexical pass: `%s` is not a libc hit (control)" % lab, append_to(
            "cpu-dispatch.cpp", "void zzplant_q() {\n    (void) %s;\n}\n" % body), "PASS", planted=False))
    A(Case("s3f-i1", "the lexical pass: `std :: malloc` is still E-LIBC", append_to(
        "cpu-dispatch.cpp", "void zzplant_q() {\n    void * p = std :: malloc(16);\n    (void) p;\n}\n"),
        "FAIL", "E-LIBC", "malloc"))

    # M4: the 3-D dpct_memcpy paths stage through host_buffer's std::malloc, so a caller outside helper.hpp is a raw allocation
    for fn, call in (("dpct_memcpy", "dpct::dpct_memcpy(q, a, b, 16, dpct::host_to_device)"),
                     ("async_dpct_memcpy", "dpct::async_dpct_memcpy(a, b, 16, dpct::host_to_device)"),
                     ("dpct_memcpy", "dpct_memcpy(q, a, b, 16, dpct::device_to_host)")):
        A(Case("s3f-m4", "a call of %s outside dpct/helper.hpp is an E-RAW finding (%s)" % (fn, call.split("(")[0]), plant(
            "void zzplant_d(sycl::queue & q, void * a, const void * b) {\n    %s;\n}\n" % call), "FAIL", "E-RAW", fn))
    A(Case("s3f-m4", "longer names, a comment and a string holding dpct_memcpy are not hits (control)", plant(
        "// dpct_memcpy(q, a, b, 16, 0) and async_dpct_memcpy\nvoid zzplant_d() {\n    int my_dpct_memcpy_count = 1;\n"
        "    const char * s = \"dpct_memcpy\";\n    (void) my_dpct_memcpy_count; (void) s;\n}\n"), "PASS", planted=False))

    # M5: the other mapping and duplicating primitives
    for nm, call in (("mmap64", "mmap64(nullptr, 16, 3, 34, -1, 0)"), ("mremap", "mremap(p, 16, 32, 1)"), ("valloc", "valloc(16)"),
                     ("pvalloc", "pvalloc(16)"), ("reallocarray", "reallocarray(p, 4, 4)"), ("strdup", "strdup(\"x\")"),
                     ("strndup", "strndup(\"x\", 1)"), ("strdup", "std::strdup(\"x\")")):
        A(Case("s3f-m5", "a new call of %s" % call.split("(")[0], plant_q(call), "FAIL", "E-LIBC", nm))
    A(Case("s3f-m5", "a macro whose body calls strndup", plant("#define zz_sd(s) strndup(s, 8)\n"), "FAIL", "E-LIBC",
           "#define zz_sd", planted=False))
    A(Case("s3f-m5", "a member strdup and a longer identifier are not hits (control)", plant(
        "struct zz_pool {\n    char * strdup(const char * s);\n};\nvoid zzplant_q(zz_pool & pool) {\n"
        "    char * a = pool.strdup(\"x\");\n    int my_strdup = 1;\n    (void) a; (void) my_strdup;\n}\n"), "PASS", planted=False))

    # data rules
    def rekey(d):
        return dict(d, violations=[dict(e, key=e["key"][:-1] + "1") if e["code"] == "G-CATCH" and e["key"] == CHECK_TRY_ERROR_KEY else e
                                   for e in d["violations"]])

    def pin_cite(cite):
        return lambda d: dict(d, violations=[dict(e, cite=cite) if e["code"] == "G-CATCH" and e["key"] == CHECK_TRY_ERROR_KEY else e
                                             for e in d["violations"]])

    def reason(eid, text):
        return lambda al: dict(al, entries=[dict(e, reason=text) if e["id"] == eid else e for e in al["entries"]])

    def e_libc_cite(text):
        return lambda d: dict(d, violations=[dict(e, cite=text) if e["code"] == "E-LIBC" else e for e in d["violations"]])

    A(Case("s3f-data", "re-keying the CHECK_TRY_ERROR debt entry cannot silently un-pin its fate", lambda f: f, "FAIL", "data",
           "is not the pinned key", edit_debt=rekey, planted=False))
    A(Case("s3f-data", "the CHECK_TRY_ERROR entry with a one-word cite fails", lambda f: f, "FAIL", "data", "has no cite",
           edit_debt=pin_cite("tbd"), planted=False))
    A(Case("s3f-data", "an allowlist reason naming a header line fails", lambda f: f, "FAIL", "data", "names a source line",
           edit_allowlist=reason("E-LIBC-CACHE-GUARD", "cache bookkeeping with no tensor bytes: the guard allocator (unified-cache.hpp:4368)"),
           planted=False))
    A(Case("s3f-data", "an allowlist reason naming a .cpp line fails", lambda f: f, "FAIL", "data", "names a source line",
           edit_allowlist=reason("E-CHAIN-RAW", "the raw wrapper link; the census row at ggml-sycl.cpp:24338 names it"), planted=False))
    A(Case("s3f-data", "an allowlist reason citing a ruling-ledger id with a round fails", lambda f: f, "FAIL", "data",
           "names a ruling-ledger id", edit_allowlist=reason("E-LIBC-CACHE-GUARD", "cache bookkeeping with no tensor bytes. Ruling M265 R2"),
           planted=False))
    A(Case("s3f-data", "an E-LIBC debt cite citing a ruling-ledger id with a round fails", lambda f: f, "FAIL", "data",
           "names a ruling-ledger id", edit_debt=e_libc_cite("ruling M265 R2, step 7 census item (f): deleted with S7(d)"), planted=False))
    A(Case("s3f-data", "the CHECK_TRY_ERROR cite citing a ruling-ledger id fails", lambda f: f, "FAIL", "data", "names a ruling-ledger id",
           edit_debt=pin_cite("ruling M265 R3, design 5.4a: the handler is rewritten there"), planted=False))
    return c


def matrix_cases_s3g():
    """The review fold of the S3-0 fold: keywords before `::`, the whole-scope rule in text, and the spellings of the data rules."""
    c = []
    A = c.append

    def lex(body):
        return append_to("cpu-dispatch.cpp", body)

    # I-1: a keyword before `::` is not a scope, so the name behind it is bare (a libc hit)
    for lab, body in (("return ::malloc", "#define zz_k(n) return ::malloc(n)\n"),
                      ("else ::malloc", "#define zz_k(c, n) if (c) {} else ::malloc(n)\n"),
                      ("throw ::malloc", "#define zz_k(n) throw ::malloc(n)\n"),
                      ("sizeof ::strdup", "#define zz_k(s) (void) sizeof ::strdup(s)\n"),
                      ("co_return ::malloc", "#define zz_k(n) co_return ::malloc(n)\n"),
                      ("do { return ::calloc }", "#define zz_k(n) do { return ::calloc(n, 1); } while (0)\n"),
                      ("a comparison `a > ::malloc`", "#define zz_k(a, n) (a > ::malloc(n))\n")):
        A(Case("s3g-kw", "a macro body `%s` is E-LIBC" % lab, plant(body), "FAIL", "E-LIBC", "#define zz_k", planted=False))
    for lab, body in (("return ::malloc", "void * zzplant_q() {\n    return ::malloc(16);\n}\n"),
                      ("else ::malloc", "void zzplant_q(int c) {\n    if (c) {} else ::malloc(16);\n}\n"),
                      ("sizeof ::strdup", "unsigned long zzplant_q() {\n    return sizeof ::strdup(\"x\");\n}\n"),
                      ("a comparison `a > ::malloc`", "bool zzplant_q(void * a) {\n    return a > ::malloc(16);\n}\n")):
        A(Case("s3g-kw", "the lexical pass: `%s` is E-LIBC" % lab, lex(body), "FAIL", "E-LIBC", "malloc" if "strdup" not in lab else "strdup"))
    A(Case("s3g-kw", "a macro body `zz_pool<std::vector<int>>::malloc` is a qualified name, not a hit (control)", plant(
        "#define zz_k(n) zz_pool<std::vector<int>>::malloc(n)\n"), "PASS", planted=False))
    A(Case("s3g-kw", "the lexical pass: `zz_pool<int>::malloc` is a qualified name, not a hit (control)", lex(
        "void zzplant_q() {\n    (void) zz_pool<int>::malloc(16);\n}\n"), "PASS", planted=False))

    # m4/m5: the whole scope is read, as the tree path does, and a continuation is stripped on both text paths
    A(Case("s3g-q", "a macro body `ns::std::malloc` is qualified by ns, not a hit (control)", plant(
        "#define zz_q(n) ns::std::malloc(n)\n"), "PASS", planted=False))
    A(Case("s3g-q", "the lexical pass: `ns::std::malloc` is not a hit (control)", lex(
        "void zzplant_q() {\n    (void) ns::std::malloc(16);\n}\n"), "PASS", planted=False))
    A(Case("s3g-q", "the tree path: `ns::std::malloc` is not a hit (control)", plant(
        "void zzplant_q() {\n    void * p = ns::std::malloc(16);\n    (void) p;\n}\n"), "PASS", planted=False))
    A(Case("s3g-q", "a macro body `::std::malloc` is E-LIBC", plant("#define zz_q(n) ::std::malloc(n)\n"), "FAIL", "E-LIBC",
           "#define zz_q", planted=False))
    A(Case("s3g-q", "a macro body `xstd::malloc` is a scope that merely ends in std, not a hit (control)", plant(
        "#define zz_q(n) xstd::malloc(n)\n"), "PASS", planted=False))
    A(Case("s3g-q", "the lexical pass: `xstd::malloc` is not a hit (control)", lex(
        "void zzplant_q() {\n    (void) xstd::malloc(16);\n}\n"), "PASS", planted=False))
    A(Case("s3g-q", "the lexical pass: `pool2 :: realloc` (a digit in the scope) is not a hit (control)", lex(
        "void zzplant_q(void * p) {\n    (void) pool2 :: realloc(p, 1);\n}\n"), "PASS", planted=False))
    A(Case("s3g-q", "the lexical pass: a continued #define `pool ::<cont> realloc` is not a hit (control)", lex(
        "#define zz_lx(p, n) pool ::\\\n    realloc(p, n)\n"), "PASS", planted=False))
    A(Case("s3g-q", "the lexical pass: a continued #define `std ::<cont> malloc` is E-LIBC", lex(
        "#define zz_lx2(n) std ::\\\n    malloc(n)\n"), "FAIL", "E-LIBC", "malloc"))
    A(Case("s3g-q", "a macro body `sycl ::<cont> malloc` is clause (e)'s one E-RAW hit and no E-LIBC (control)", plant(
        "#define zz_sm2(q, n) sycl ::\\\n   malloc(n, q, sycl::usm::alloc::host)\n"), "PASS",
        allowlist={"id": "E-ZZ-SM2", "code": "E-RAW", "file": PLANT, "function": "#define zz_sm2", "name": "sycl::malloc",
                   "count": 1, "reason": "mutation-matrix test entry"}, planted=False))
    A(Case("s3g-q", "a name taken as a value in an ERROR-root file is still E-LIBC, through the tree pass", lex(
        "void zzplant_q() {\n    auto f = ::malloc;\n    (void) f;\n}\n"), "FAIL", "E-LIBC", "malloc"))
    A(Case("s3g-q", "the lexical pass is call-shaped, so a declaration `malloc(` in an ERROR-root file fails closed (pinned)", lex(
        "extern void * malloc(unsigned long n);\nvoid zzplant_q() {\n}\n"), "FAIL", "E-LIBC", "malloc"))

    # m2, m3: the spellings of the data rules
    def reason(eid, text):
        return lambda al: dict(al, entries=[dict(e, reason=text) if e["id"] == eid else e for e in al["entries"]])

    def cite(code, text):
        return lambda d: dict(d, violations=[dict(e, cite=text) if e["code"] == code else e for e in d["violations"]])

    base = "cache bookkeeping with no tensor bytes; canonical contract section 9.1. "
    for lab, text in (("ruling M265, R2", "ruling M265, R2"), ("a section sign, §M265", "see §M265"), ("a bare M265 R2", "M265 R2"),
                      ("a lowercase round, ruling M265 r2", "ruling M265 r2"), ("a round-less ruling M247", "rulings M247 second"),
                      ("a ruling split by a newline", "ruling\nM265\nR2"), ("a bare (R2)", "per ruling (R2)"),
                      ("a round alone, ruling R2", "ruling R2")):
        A(Case("s3g-data", "an allowlist reason with %s fails" % lab, lambda f: f, "FAIL", "data", "names a ruling-ledger id",
               edit_allowlist=reason("E-LIBC-CACHE-GUARD", base + text), planted=False))
    A(Case("s3g-data", "a debt cite with a round-less section-sign ruling (rulings §M243) fails", lambda f: f, "FAIL", "data",
           "names a ruling-ledger id", edit_debt=cite("E-LIBC", "design 11 step 7, the dead allocator bullet; rulings §M243"), planted=False))
    for lab, text in (("a .inl file", "helper.inl:12"), ("a .cu file", "kernels.cu :12"), ("a .py file", "gate.py:5"),
                      ("a .md file", "design.md:77"), ("a #L anchor", "unified-cache.hpp#L1362"), ("a colon and a space", "common.cpp: 12"),
                      ("a .cc file", "x.cc:9")):
        A(Case("s3g-data", "an allowlist reason naming a line in %s fails" % lab, lambda f: f, "FAIL", "data", "names a source line",
               edit_allowlist=reason("E-LIBC-CACHE-GUARD", base + "see " + text), planted=False))
    return c


def matrix_cases_s31():
    """S3-1: mem_handle::owner_use_count exists, so clause (l) is enforced on the real tree."""
    c = []
    A = c.append
    A(Case("s31-l", "the real tree declares and defines owner_use_count, so clause (l) is enforced on the caller half only (replace_within is absent, so L-FREE is vacuous), not dormant", lambda f: f, "PASS",
           active="l", planted=False))
    A(Case("s31-l", "a new caller of owner_use_count outside the allowlist is a finding on the real tree", plant(
        "bool zzplant_l(mem_handle & h) {\n    return h.owner_use_count() > 1;\n}\n"), "FAIL", "L-CALLER", "zzplant_l"))
    A(Case("s31-l", "the accessor renamed in mem-handle.hpp and mem-handle.cpp makes the clause dormant again", lambda f: dict(
        f, **{rel: f[rel].replace(b"owner_use_count", b"owner_use_count_zz") for rel in ("mem-handle.hpp", "mem-handle.cpp")}),
        "PASS", dormant="l", planted=False))
    return c


def matrix_cases_s2d2():
    """Clause (p), witnesses 37 and 38. A FAIL case mutates the b1 (or b2) fixture; a PASS control asserts the block is active,
    or, on today's tree, dormant."""
    c = []
    A = c.append

    def on(tree, *muts):
        return chain(tree, *muts)

    def b1(*muts):
        return on(b1_tree, *muts)

    def b2(*muts):
        return on(b2_tree, *muts)

    SHAPE_CALL = "    const fattn_params p = ggml_sycl_fattn_shape_of(dst);\n"
    ROUTED_CALL = "if (!ggml_sycl_fattn_onednn_routed(params, f.head_dim_v, false, nullptr, nullptr)) {"
    FLASH_OPEN = "void ggml_sycl_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_sycl::sycl_tensor safe_dst) {\n"
    DEFAULT_ARM = "if (!safe_decode && " + GATE_DEFAULT + ") {"
    D512_ARM = "if (d512_onednn_enabled && " + GATE_D512 + ") {"
    # 37: the default oneDNN SDPA arm and the route
    A(Case("37", "the default arm restored to its inline flag test and direct plan call", b1(
        replace_once("fattn.cpp", DEFAULT_ARM, "if (!safe_decode && g_sycl_fa_onednn_enabled && !g_sycl_paged_v2_enabled) {"),
        replace_once("fattn.cpp", "plan      = ggml_sycl_fattn_routed_plan(params,", "plan      = ggml_sycl_flash_attn_ext_onednn_plan(params,")),
        "FAIL", "P-ROUTE", "onednn-call", planted=False))
    A(Case("37", "the default arm restored: its direct plan call is named too", b1(
        replace_once("fattn.cpp", DEFAULT_ARM, "if (!safe_decode && g_sycl_fa_onednn_enabled && !g_sycl_paged_v2_enabled) {"),
        replace_once("fattn.cpp", "plan      = ggml_sycl_fattn_routed_plan(params,", "plan      = ggml_sycl_flash_attn_ext_onednn_plan(params,")),
        "FAIL", "P-ROUTE", "plan-call", planted=False))
    A(Case("37", "the default arm's branch also tests g_sycl_fa_onednn_enabled", b1(
        replace_once("fattn.cpp", DEFAULT_ARM, "if (!safe_decode && g_sycl_fa_onednn_enabled && " + GATE_DEFAULT + ") {")),
        "FAIL", "P-ROUTE", "onednn-call", planted=False))
    A(Case("37", "the default arm's branch also tests g_sycl_paged_v2_enabled", b1(
        replace_once("fattn.cpp", DEFAULT_ARM, "if (!safe_decode && !g_sycl_paged_v2_enabled && " + GATE_DEFAULT + ") {")),
        "FAIL", "P-ROUTE", "onednn-call", planted=False))
    A(Case("37", "the d512 arm's branch also tests g_sycl_fa_onednn_enabled", b1(
        replace_once("fattn.cpp", D512_ARM, "if (d512_onednn_enabled && g_sycl_fa_onednn_enabled && " + GATE_D512 + ") {")),
        "FAIL", "P-ROUTE", "onednn-call", planted=False))
    A(Case("37", "the d512 arm without its own d512_onednn_enabled local: the routed branch alone is conformant (control)", b1(
        replace_once("fattn.cpp", D512_ARM, "if (" + GATE_D512 + ") {")), "PASS", planted=False, active="p-route"))
    A(Case("37", "the plan called directly in a new function of fattn.cpp", b1(append_to(
        "fattn.cpp", "void zzplant_p(const fattn_params & p) {\n    (void) ggml_sycl_flash_attn_ext_onednn_plan(p, 1, 1, false, false);\n}\n")),
        "FAIL", "P-ROUTE", "zzplant_p"))
    A(Case("37", "a new call of the oneDNN SDPA outside any routed branch", b1(append_to(
        "fattn.cpp", "void zzplant_p(ggml_backend_sycl_context & ctx, const fattn_params & p) {\n    (void) ggml_sycl_flash_attn_ext_onednn(ctx, p);\n}\n")),
        "FAIL", "P-ROUTE", "zzplant_p"))
    A(Case("37", "route_admits called from a function that is neither the predicate nor the shape function", b1(append_to(
        "fattn.cpp", "bool zzplant_p(const fattn_params & p) {\n    return ggml_sycl_fattn_onednn_route_admits(p, false, nullptr, nullptr);\n}\n")),
        "FAIL", "P-ROUTE", "zzplant_p"))
    A(Case("37", "the route's static gate called outside route_admits and the routing function", b1(append_to(
        "fattn.cpp", "bool zzplant_p(const fattn_params & p) {\n    return ggml_sycl_fattn_onednn_route_enabled(p);\n}\n")), "FAIL", "P-ROUTE", "zzplant_p"))
    A(Case("37", "the value function calling the plan directly (b2)", b2(replace_once(
        "unified-cache.cpp", "    return d512;\n", "    (void) ggml_sycl_flash_attn_ext_onednn_plan(params, 1, 1, false, false);\n    return d512;\n")),
        "FAIL", "P-ROUTE", "value-plan", planted=False))
    A(Case("37", "the value function giving the predicate a context (b2)", b2(replace_once(
        "unified-cache.cpp", ROUTED_CALL, "if (!ggml_sycl_fattn_onednn_routed(params, f.head_dim_v, false, &ctx, nullptr)) {")),
        "FAIL", "P-ROUTE", "value-predicate", planted=False))
    A(Case("37", "the b1 tree: every arm sits in a routed branch (control)", b1(), "PASS", planted=False, active="p-route"))
    A(Case("37", "today's tree: the inline arms are not yet a finding, the clause is dormant (control)", lambda f: f, "PASS", planted=False,
           dormant="p-route"))
    A(Case("37", "today's tree plus a direct plan call: dormant, nothing named (control)", append_to(
        "fattn.cpp", "void zzplant_p(const fattn_params & p) {\n    (void) ggml_sycl_flash_attn_ext_onednn_plan(p, 1, 1, false, false);\n}\n"),
        "PASS", dormant="p-route"))
    A(Case("37", "route_admits spelled as a lambda variable: the latch fails", plant(
        "static auto ggml_sycl_fattn_onednn_route_admits = [](int p) { return p > 0; };\n"), "FAIL", "X-LATCH", "latch:p-route"))
    # 38: the support decision has one home
    A(Case("38", "fattn_vec_supports_head_dim re-added in ggml_sycl_flash_attn_ext_supported", b1(replace_once(
        "fattn.cpp", SHAPE_CALL, SHAPE_CALL + "    if (!fattn_vec_supports_head_dim(dst->src[0]->ne[0])) {\n        return false;\n    }\n")),
        "FAIL", "P-HOME", "fattn_vec_supports_head_dim", planted=False))
    A(Case("38", "a GGML_TYPE_ comparison in ggml_sycl_flash_attn_ext_supported", b1(replace_once(
        "fattn.cpp", SHAPE_CALL, SHAPE_CALL + "    if (dst->src[0]->type != GGML_TYPE_F32) {\n        return false;\n    }\n")),
        "FAIL", "P-HOME", "type-compare", planted=False))
    A(Case("38", "a head dim compared with a literal in ggml_sycl_flash_attn_ext_supported", b1(replace_once(
        "fattn.cpp", SHAPE_CALL, SHAPE_CALL + "    if (p.ne00 == 80) {\n        return false;\n    }\n")),
        "FAIL", "P-HOME", "head-dim-literal", planted=False))
    A(Case("38", "ggml_sycl_flash_attn_ext_supported not calling the shape function", b1(replace_once(
        "fattn.cpp", "    return ggml_sycl_fattn_shape_supported(p, (int) dst->src[2]->ne[0]);\n", "    return true;\n")),
        "FAIL", "P-HOME", "shape-call", planted=False))
    A(Case("38", "&& p.ne00 != 80 appended to the routed predicate's return", b1(replace_once(
        "fattn.cpp", "&& ggml_sycl_fattn_shape_supported(p, d_v);", "&& ggml_sycl_fattn_shape_supported(p, d_v) && p.ne00 != 80;")),
        "FAIL", "P-HOME", "routed-body", planted=False))
    A(Case("38", "std::getenv of the switch planted in the route's static gate (the read count is 2)", b1(replace_once(
        "fattn.cpp", "    static const bool d512_onednn_enabled = ggml_sycl_fa_onednn_d512_enabled();\n    if (p.ne00 == 512",
        '    const char * zz_env = std::getenv("GGML_SYCL_FLASH_ATTN_EXT");\n    if (zz_env == nullptr) {\n        return false;\n    }\n'
        "    static const bool d512_onednn_enabled = ggml_sycl_fa_onednn_d512_enabled();\n    if (p.ne00 == 512")),
        "FAIL", "P-HOME", "is read at", planted=False))
    A(Case("38", "the switch read through a bare getenv in a source outside the scope", b1(plant(
        'void zz() {\n    const char * e = getenv( "GGML_SYCL_FLASH_ATTN_EXT" );\n}\n', "repo/zz-read.cpp")),
        "FAIL", "P-HOME", "repo/zz-read.cpp"))
    A(Case("38", "the tests' two reads of the switch kept (the count is 3, as before b1)", lambda f: b1_tree(f, keep_test_reads=True),
           "FAIL", "P-HOME", "tests/test-sycl-fattn-onednn-gates.cpp", planted=False))
    A(Case("38", "the shape function's call of ggml_sycl_flash_attn_ext_enabled() removed", b1(replace_once(
        "fattn.cpp", "    if (!ggml_sycl_flash_attn_ext_enabled()) {\n        return false;\n    }\n    if (!fattn_vec_supports_head_dim(p.ne00)",
        "    if (!fattn_vec_supports_head_dim(p.ne00)")), "FAIL", "P-HOME", "first-statement", planted=False))
    A(Case("38", "fattn_vec_supports_head_dim planted in the D512 dispatch arm's function", b1(insert_after(
        "fattn.cpp", FLASH_OPEN, "    if (!fattn_vec_supports_head_dim(32) && 32 != 512) {\n        return;\n    }\n")),
        "FAIL", "P-HOME", "census:fattn_vec_supports_head_dim", planted=False))
    A(Case("38", "ggml_sycl_fattn_d512_tile_admissible called in route_admits", b1(replace_once(
        "fattn.cpp", "    const ggml_sycl_onednn_fa_layout_plan plan =\n        ggml_sycl_flash_attn_ext_onednn_plan(p,",
        "    if (!ggml_sycl_fattn_d512_tile_admissible(nullptr)) {\n        return false;\n    }\n"
        "    const ggml_sycl_onednn_fa_layout_plan plan =\n        ggml_sycl_flash_attn_ext_onednn_plan(p,")),
        "FAIL", "P-HOME", "ggml_sycl_fattn_d512_tile_admissible", planted=False))
    A(Case("38", "a third tile-screen call in the dispatch function", b1(insert_after(
        "fattn.cpp", FLASH_OPEN, "    (void) ggml_sycl_fattn_d512_tile_admissible(nullptr);\n")),
        "FAIL", "P-HOME", "tile-dispatch-sites", planted=False))
    A(Case("38", "the route's hatch without the d512 flag: a head dim literal in route_admits", b1(replace_once(
        "fattn.cpp", "    if (p.ne00 == 512 && !d512_onednn_enabled) {", "    if (p.ne00 == 512) {")),
        "FAIL", "P-HOME", "head-dim-literal", planted=False))
    A(Case("38", "ggml_sycl_fattn_kv_pair_of called outside the shape function and the fills", b1(append_to(
        "fattn.cpp", "int zzplant_p(const fattn_params & p) {\n    return (int) ggml_sycl_fattn_kv_pair_of(p.K_type, p.V_type);\n}\n")),
        "FAIL", "P-HOME", "census:ggml_sycl_fattn_kv_pair_of"))
    A(Case("38", "the FLASH_ATTN_EXT case guarded by a head-dim test in ggml_backend_sycl_device_supports_op", b1(replace_once(
        "ggml-sycl.cpp", "            return ggml_sycl_flash_attn_ext_supported(op);\n",
        "            return fattn_vec_supports_head_dim(op->src[0]->ne[0]) && ggml_sycl_flash_attn_ext_supported(op);\n")),
        "FAIL", "P-HOME", "case-line", planted=False))
    for ph, tree in (("b1", b1), ("b2", b2)):
        fills = [("fattn.cpp", "    " + FILL_LINE + "\n    return params;", "shape_of"),
                 ("fattn.cpp", "    " + FILL_LINE + "\n\n    // Multi-token decode support", "flash_attn_ext"),
                 ("unified-cache.cpp", "        " + FILL_LINE + "\n", "value function" if ph == "b2" else "load-maxima fill")]
        for rel, anchor, nm in fills:
            tail = anchor[len("    " + FILL_LINE):] if rel == "fattn.cpp" else "\n"
            ind = anchor[:len(anchor) - len(anchor.lstrip())]
            A(Case("38", "kv_is_fp8 = false replacing the pinned fill in %s (%s)" % (nm, ph), tree(replace_once(
                rel, anchor, ind + "params.kv_is_fp8 = false;" + tail)), "FAIL", "P-FILL", "p-fill:pin", planted=False))
            A(Case("38", "kv_is_fp8 = false appended after the pinned fill in %s (%s)" % (nm, ph), tree(replace_once(
                rel, anchor, ind + FILL_LINE + "\n" + ind + "params.kv_is_fp8 = false;" + tail)),
                "FAIL", "P-FILL", "p-fill:write", planted=False))
    A(Case("38", "a kv_is_fp8 write outside every fill", b1(append_to(
        "fattn.cpp", "void zzplant_p(fattn_params & params) {\n    params.kv_is_fp8 |= true;\n}\n")), "FAIL", "P-FILL", "p-fill:write"))
    A(Case("38", "a kv_is_fp8 member write through a pointer, outside every fill", b1(append_to(
        "fattn.cpp", "void zzplant_p(fattn_params * params) {\n    params->kv_is_fp8 = true;\n}\n")), "FAIL", "P-FILL", "p-fill:write"))
    A(Case("38", "the local bool kv_is_fp8 is not a member write (control)", b1(append_to(
        "fattn.cpp", "bool zzplant_p() {\n    bool kv_is_fp8 = true;\n    kv_is_fp8 = false;\n    return kv_is_fp8;\n}\n")), "PASS", active="p-fill"))
    A(Case("38", "the load-maxima fill still defined in b2", b2(append_to("unified-cache.cpp", B1_LOADFILL)),
           "FAIL", "P-FILL", "load-fill-in-b2", planted=False))
    A(Case("38", "the KV buffer's strncmp parse kept in ggml-sycl.cpp beside the helper", b1(replace_once(
        "ggml-sycl.cpp", "    layer_id = ggml_sycl_kv_cache_layer_of(name);\n",
        '    layer_id = ggml_sycl_kv_cache_layer_of(name);\n    const char * prefix_v = "cache_v_l";\n    if (strncmp(name, prefix_v, 9) == 0) {\n        layer_id = atoi(name + 9);\n    }\n')),
        "FAIL", "P-LAYER", "ggml-sycl.cpp", planted=False))
    A(Case("38", "sscanf(name, \"cache_k_l%d\", &il) planted in ggml-sycl.cpp", b1(append_to(
        "ggml-sycl.cpp", 'void zzplant_p(const char * name) {\n    int il = -1;\n    sscanf(name, "cache_k_l%d", &il);\n}\n')),
        "FAIL", "P-LAYER", "ggml-sycl.cpp"))
    A(Case("38", "the helper holding a third format string", b1(replace_once(
        "common.cpp", '    return -1;\n}\n', '    if (sscanf(name, "cache_v_l%d", &id) == 1) {\n        return id;\n    }\n    return -1;\n}\n')),
        "FAIL", "P-LAYER", "helper-formats", planted=False))
    A(Case("38", "a comment that spells the layer prefix is not a parse (control)", b1(append_to(
        "fattn.cpp", '// the name is "cache_k_l0" for layer zero\nvoid zzplant_p() {\n}\n')), "PASS", active="p-layer"))
    A(Case("38", "ggml_sycl_kv_cache_layer_of spelled as a macro: the latch fails", plant(
        "#define ggml_sycl_kv_cache_layer_of(n) -1\n"), "FAIL", "X-LATCH", "latch:p-layer", planted=False))
    A(Case("38", "ggml_sycl_fattn_kv_pair_of spelled as a lambda variable: the latch fails", plant(
        "static auto ggml_sycl_fattn_kv_pair_of = [](int k, int v) { return k == v; };\n"), "FAIL", "X-LATCH", "latch:p-fill"))
    A(Case("38", "ggml_sycl_flash_attn_ext_enabled spelled as a macro: the latch fails", plant(
        "#define ggml_sycl_flash_attn_ext_enabled() true\n"), "FAIL", "X-LATCH", "latch:p-home", planted=False))
    A(Case("38", "onednn_graph_scratch_bytes spelled as a lambda variable: the latch fails", plant(
        "static auto onednn_graph_scratch_bytes = [](int n) { return (size_t) n; };\n"), "FAIL", "X-LATCH", "latch:p-charge"))
    # the charge side
    CHARGE_HELPER = "        if (!fattn_vec_supports_head_dim(f.head_dim_k) && f.head_dim_k != 512) {\n            continue;\n        }\n"
    CHARGE_LITERAL = ("        if (f.head_dim_k != 64 && f.head_dim_k != 128 && f.head_dim_k != 256 && f.head_dim_k != 512) {\n"
                      "            continue;\n        }\n")
    for ph, tree in (("b1", b1), ("b2", b2)):
        where = "placement_plan_set_routed_head_maxima's layer loop" if ph == "b1" else "onednn_graph_scratch_bytes's walk"
        A(Case("38", "the support helper and a 512 literal planted in %s" % where, tree(replace_once(
            "unified-cache.cpp", "        " + ROUTED_CALL, CHARGE_HELPER + "        " + ROUTED_CALL)),
            "FAIL", "P-CHARGE", "fattn_vec_supports_head_dim", planted=False))
        A(Case("38", "the same without the helper, four literals, planted in %s" % where, tree(replace_once(
            "unified-cache.cpp", "        " + ROUTED_CALL, CHARGE_LITERAL + "        " + ROUTED_CALL)),
            "FAIL", "P-CHARGE", "literal", planted=False))
        A(Case("38", "the charge side calling ggml_sycl_fattn_shape_supported (%s)" % ph, tree(replace_once(
            "unified-cache.cpp", "        " + ROUTED_CALL,
            "        if (!ggml_sycl_fattn_shape_supported(params, 0)) {\n            continue;\n        }\n        " + ROUTED_CALL)),
            "FAIL", "P-CHARGE", "ggml_sycl_fattn_shape_supported", planted=False))
    A(Case("38", "the D512 count line missing from the value function (b2)", b2(replace_once(
        "unified-cache.cpp", "        if (params.ne00 == 512) {\n            ++d512;\n        }\n", "        ++d512;\n")),
        "FAIL", "P-CHARGE", "d512-count", planted=False))
    A(Case("38", "the D512 count line twice in the value function (b2)", b2(replace_once(
        "unified-cache.cpp", "        if (params.ne00 == 512) {\n            ++d512;\n        }\n",
        "        if (params.ne00 == 512) {\n            ++d512;\n        }\n        if (params.ne00 == 512) {\n            ++d512;\n        }\n")),
        "FAIL", "P-CHARGE", "d512-count", planted=False))
    A(Case("38", "the D512 count line in the b1 load-maxima fill (it lands in b2 only)", b1(replace_once(
        "unified-cache.cpp", "        " + ROUTED_CALL, "        if (params.ne00 == 512) {\n            continue;\n        }\n        " + ROUTED_CALL)),
        "FAIL", "P-CHARGE", "literal", planted=False))
    A(Case("38", "the b1 tree: one read of the switch, one call of it, every block keyed on its subject (control)", b1(), "PASS",
           planted=False, active="p-home"))
    A(Case("38", "the b2 tree: the value function replaces the load-maxima fill (control)", b2(), "PASS", planted=False, active="p-charge"))
    A(Case("38", "a setenv of the switch and a commented getenv are not reads (control)", b1(plant(
        'void zz() {\n    // getenv("GGML_SYCL_FLASH_ATTN_EXT")\n    setenv("GGML_SYCL_FLASH_ATTN_EXT", "0", 1);\n}\n', "repo/zz-set.cpp")),
        "PASS", active="p-home"))
    for blk, subj in (("p-home", "ggml_sycl_flash_attn_ext_enabled"), ("p-fill", "ggml_sycl_fattn_kv_pair_of"),
                      ("p-layer", "ggml_sycl_kv_cache_layer_of"), ("p-charge", "placement_plan_set_routed_head_maxima")):
        A(Case("38", "today's tree: %s is absent, so %s is dormant (control)" % (subj, blk), lambda f: f, "PASS", planted=False, dormant=blk))
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
    saved_m = copy.deepcopy(M_TABLES)
    try:
        if case.m_edit is not None:
            edited = copy.deepcopy(M_TABLES)
            case.m_edit(edited)
            M_TABLES.clear()
            M_TABLES.update(edited)
        fails, _, _, stats = run_gate(files, al, debt)
    finally:
        M_TABLES.clear()
        M_TABLES.update(saved_m)
    if case.active is not None and case.active not in stats["active"]:
        return False, ["clause (%s) is not keyed on a subject in this tree, so the case did not exercise it" % case.active]
    if case.dormant is not None and (case.dormant in stats["active"]
                                     or not any(l.startswith("DORMANT %s:" % case.dormant) for l in stats["dormant"])):
        return False, ["clause (%s) is not dormant in this tree, so the control did not test dormancy" % case.dormant]
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
    core = core_files(files)
    ctx = Ctx(core)
    n = 0
    for rel in files:
        if not (rel.startswith("zz") or "/zz" in rel or rel == "cpu-dispatch.cpp"):
            # a planted function appended to a real file (clause (p)'s fixtures are real files)
            n += sum(1 for name, _, _ in s2d_facts(files[rel])["fdefs"] if name.startswith("zzplant"))
            continue
        if rel.startswith("zz") or "/zz" in rel:
            fa2 = s2d_facts(files[rel])
            n += len(fa2["occ"]) + len(fa2["fdefs"]) + (P_ENV.encode() in files[rel])
        if rel in core:
            fa = facts_for(rel, core[rel], ctx)
        else:
            continue
        n += sum(1 for c in fa["constructions"] if "zz" in c["key"] or "zz" in c["func"])
        n += sum(1 for r in fa["lexical"] if r["var"].startswith("zz"))
        n += sum(1 for r in fa["raws"] if "zz" in r["func"])
        n += sum(1 for r in fa["libcs"] if "zz" in r["func"])
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
    wit = [m.group(0) for m in re.finditer(r"add_test\(NAME test-sycl-alloc-zone-contract-w\s[^)]*\)", text, flags=re.S)]
    wit_props = re.findall(r"set_tests_properties\(test-sycl-alloc-zone-contract-w PROPERTIES[^)]*\)", text)
    ok = len(wit) == 1 and "--witnesses" in wit[0] and "--mutation-matrix" not in wit[0] and len(wit_props) == 1 \
        and 'LABELS "sycl;host-only;ast;mutation"' in wit_props[0] and re.search(r"\bTIMEOUT\s+600\b", wit_props[0]) is not None
    detail = "%d witnesses registration(s)" % len(wit)
    ok = ok and len(plain) == 1 and "--mutation-matrix" not in plain[0] and loop is not None \
        and len(plain_props) == 1 and 'LABELS "sycl;host-only;ast"' in plain_props[0] \
        and re.search(r"\bTIMEOUT\s+120\b", plain_props[0]) is not None
    detail += ", %d plain registration(s)" % len(plain)
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
    region = "".join(plain) + "".join(wit) + (loop.group(0) if loop is not None else "")
    ok = ok and "--write-debt" not in region and "--allow-growth" not in region
    return ok, detail


def cmake_mutants(root):
    """cmake_witness must fail on a registration that drops a shard, miscounts them, passes a regeneration flag, loses
    the plain test or a TIMEOUT, and pass on the real one: a witness that cannot fail proves nothing."""
    import tempfile
    rel = "ggml/src/ggml-sycl/CMakeLists.txt"
    text = (Path(root) / rel).read_text()
    mutants = [
        ("drops the last shard from the foreach", lambda t: t.replace(
            "foreach(zc_shard %s)" % " ".join(str(i) for i in range(SHARDS)), "foreach(zc_shard %s)" % " ".join(str(i) for i in range(SHARDS - 1)), 1)),
        ("passes the wrong shard count", lambda t: t.replace("${zc_shard}/%d" % SHARDS, "${zc_shard}/%d" % (SHARDS + 1), 1)),
        ("loses the witnesses test", lambda t: t.replace("add_test(NAME test-sycl-alloc-zone-contract-w\n", "add_test(NAME test-sycl-alloc-zone-contract-x\n", 1)),
        ("runs the matrix in the witnesses test", lambda t: t.replace("--witnesses)", "--witnesses --mutation-matrix)", 1)),
        ("loses the witnesses test's TIMEOUT", lambda t: t.replace('test-sycl-alloc-zone-contract-w PROPERTIES LABELS "sycl;host-only;ast;mutation" TIMEOUT 600',
                                                                    'test-sycl-alloc-zone-contract-w PROPERTIES LABELS "sycl;host-only;ast;mutation"', 1)),
        ("passes --write-debt to a shard", lambda t: t.replace("--mutation-matrix --shard", "--write-debt --mutation-matrix --shard", 1)),
        ("loses the plain gate's test", lambda t: t.replace("add_test(NAME test-sycl-alloc-zone-contract\n", "add_test(NAME test-sycl-alloc-zone-contract-x\n", 1)),
        ("drops the plain gate's TIMEOUT", lambda t: t.replace('PROPERTIES LABELS "sycl;host-only;ast" TIMEOUT 120)', 'PROPERTIES LABELS "sycl;host-only;ast")', 1)),
        ("loses a shard's TIMEOUT", lambda t: t.replace('-m${zc_shard} PROPERTIES LABELS "sycl;host-only;ast;mutation" TIMEOUT 600',
                                                       '-m${zc_shard} PROPERTIES LABELS "sycl;host-only;ast;mutation"', 1)),
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


def run_matrix(files, allowlist, debt, shard=None):
    """`shard` is (k, n) or None. Every shard re-checks the unmutated baseline and runs its slice of the cases; a plain run
    is shard 0 of 1. The coverage checks and the process-level witnesses run once, in `--witnesses`."""
    k, n = shard if shard else (0, 1)
    if not baseline_green(files, allowlist, debt):
        return 1
    bad = 0
    all_cases = matrix_cases()
    cases = shard_slice(all_cases, k, n)
    if not cases:
        print("FAIL: shard %d/%d selects no case; an empty shard checks nothing" % (k, n))
        return 1
    print("shard %d/%d: %d of %d case(s)" % (k, n, len(cases), len(all_cases)))
    for case in cases:
        ok, got = evaluate_case(files, allowlist, debt, case)
        print("%s witness %-4s %-80s expect %s" % ("ok  " if ok else "FAIL", case.wid, case.label, case.expect))
        if not ok:
            bad += 1
            for g in got[:4]:
                print("       got: " + g)
            if not got:
                print("       got: PASS")
    print("matrix shard %d/%d: %d case(s), %d wrong" % (k, n, len(cases), bad))
    return 1 if bad else 0


def baseline_green(files, allowlist, debt):
    fails, _, _, _ = run_gate(files, allowlist, debt)
    if fails:
        print("FAIL: the unmutated baseline is red, so a matrix scored against it proves nothing:")
        for f in fails[:20]:
            print("  " + f)
        return False
    print("baseline: PASS (the matrix is scored against a green tree)")
    return True


def run_witnesses(files, allowlist, debt, root):
    """The coverage checks (every witness has a FAIL case, no case names an unknown witness, the matcher's self-test) and the
    process-level witnesses. Run once, by its own ctest, so no matrix shard carries them."""
    if not baseline_green(files, allowlist, debt):
        return 1
    bad = 0
    all_cases = matrix_cases()
    for w in sorted(set(c.wid for c in all_cases) - set(WITNESSES)):
        print("FAIL: case witness %s is not in WITNESSES" % w)
        bad += 1
    for w in WITNESSES:
        if w in PROCESS_WITNESSES:
            continue
        if not any(c.wid == w and c.expect == "FAIL" for c in all_cases):
            print("FAIL: witness %s has no FAIL case" % w)
            bad += 1
    if not (names_new("FAIL new D-ZONE-COUNT x", "D-ZONE-COUNT") and not names_new("FAIL new D-ZONE-COUNT x", "D-ZONE")):
        print("FAIL: the matcher's own self-test (prefix collision) is wrong")
        bad += 1
    extra = [("f", "missing tree_sitter_language_pack exits 1 and names it") + f_witness_missing_pack(),
             ("m9", "an unscanned .inl file fails the gate") + m9_witness()]
    extra += [("cmake", label, ok, d) for label, ok, d in cmake_mutants(root)]
    extra += [("m6", label, ok, d) for label, ok, d in m6_witnesses()]
    extra += [("r2m4", label, ok, d) for label, ok, d in r2m4_witnesses(allowlist)]
    extra += [("r2m8", label, ok, d) for label, ok, d in r2m8_witnesses()]
    extra += [("m14", label, ok, d) for label, ok, d in m14_witness(files, allowlist, debt)]
    extra += [("s2d", label, ok, d) for label, ok, d in s2d_witnesses(files, allowlist, debt)]
    for wid, label, ok, detail in extra:
        print("%s witness %-4s %-80s (%s)" % ("ok  " if ok else "FAIL", wid, label, str(detail)[:80]))
        bad += 0 if ok else 1
    print("witnesses: %d check(s), %d wrong" % (len(extra), bad))
    return 1 if bad else 0


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--data", default=None, help="directory holding allowlist.json and debt.json")
    ap.add_argument("--mutation-matrix", action="store_true")
    ap.add_argument("--shard", default=None, metavar="K/N", help="with --mutation-matrix, run shard K of N")
    ap.add_argument("--witnesses", action="store_true",
                    help="run the coverage checks and the process-level witnesses (the cases are run by --mutation-matrix)")
    ap.add_argument("--list", action="store_true", help="print every finding before the allowlist and debt")
    ap.add_argument("--write-debt", action="store_true",
                    help="rewrite debt.json from the current tree; refuses to add entries. Never used by the ctest")
    ap.add_argument("--allow-growth", action="store_true", help="with --write-debt, permit new entries (key migration or seeding only)")
    a = ap.parse_args()
    if a.allow_growth and not a.write_debt:
        ap.error("--allow-growth only means something with --write-debt")
    if (a.mutation_matrix or a.witnesses) and (a.list or a.write_debt):
        ap.error("--mutation-matrix and --witnesses cannot be combined with --list or --write-debt")
    if a.witnesses and (a.mutation_matrix or a.shard is not None):
        ap.error("--witnesses is its own run, not a shard of the matrix")
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
    files.update(load_test_sources(a.root))
    files.update(load_repo_reads(a.root))
    S2D_DATA["dir"] = str(data)
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
    for r in stats["dormant"]:
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
        return run_matrix(files, allowlist, debt, shard)
    if a.witnesses:
        return run_witnesses(files, allowlist, debt, a.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
