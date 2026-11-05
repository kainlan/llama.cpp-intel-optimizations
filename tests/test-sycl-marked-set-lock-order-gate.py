#!/usr/bin/env python3
"""Lock-order gate for the SCRATCH marked set and the pinned pool's chunk release (docs/design/sycl-mmap-host-tier.md
§7 C4, "The lock-order gate"; canon §12.5).

A returned SCRATCH chunk is filed in the device cache's marked set under `marked_set_mutex_` (L4, last in L4) and
released through the pinned pool under the pool's `mutex_` (L5, first in L5). Its backing is freed by a move-only
`released_chunk` carrier after both locks have dropped, and its bytes leave the pool's footprint only when the
registry no longer holds the backing's row. This gate pins that discipline in the source of the listed bodies only;
it does not claim that their callers hold no lock.

Twelve clauses, each with at least one planted control below that must fail with its quoted reason:

   1. every marked-set access holds `marked_set_mutex_` (the `_locked` cores are wholly under it);
   2. that lock is taken before any pool call;
   3. no pool body holding `mutex_` calls `release_marked_chunk` or names `marked_set_mutex_`;
   4. no canon §12.5 lock, nor `deferred_frees_mutex_`, is named inside either lock's scope;
   5. a backing is released only outside both scopes, after each walk's scope closes, and every collecting vector
      and the public form's inline slot are declared before the lock;
   6. `configure_host_zones` reads `ggml_sycl_strict_enabled()` between the pool's refusal and its return;
   7. the call forms: the core only under the lock, the public form never under it, the single-slot core only from
      the public form, the lease drop through the public form, and no allocation on the lease-drop path;
   8. the carrier's destructor subtracts the footprint only on the registry's GONE answer and strands it otherwise;
   9. the registry accessor compares the whole key under `g_runtime_alloc_mutex` and writes nothing;
  10. every decision read of the footprint goes through `footprint_locked()`, with a per-mention allow-list, the
      nine log reads exempt, and the pinned committed accessor's callers counted;
  11. the stranded list is reserved before a chunk exists, bounded in the carrier, and pruned in place;
  12. the pool asserts no carrier is outstanding first, and the carrier's count drop is its last statement.

The source is read comment-stripped, so a comment that quotes a member is not a mention (the negative control).
Plain file I/O, not the codescout index, which is blind in ggml-sycl.cpp.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

POOL = "ggml/src/ggml-sycl/pinned-pool.cpp"
CACHE = "ggml/src/ggml-sycl/unified-cache.cpp"
CANON = "docs/design/sycl-canonical-memory-architecture.md"
SYCL_DIR = "ggml/src/ggml-sycl"

# ---------------------------------------------------------------------------------------------------------------
# Text views. `code` blanks comments and keeps string literals; `bare` also blanks the inside of string and character
# literals. Both keep every offset and line break of the raw text, so a position means the same thing in all three.
# ---------------------------------------------------------------------------------------------------------------

_LEXEME_RE = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|//[^\n]*|/\*.*?\*/', re.DOTALL)


def _spaces(s: str) -> str:
    return re.sub(r"[^\n]", " ", s)


def views(raw: str):
    code = []
    bare = []
    at = 0
    for m in _LEXEME_RE.finditer(raw):
        code.append(raw[at:m.start()])
        bare.append(raw[at:m.start()])
        tok = m.group(0)
        if tok[0] in "\"'":
            code.append(tok)
            bare.append(tok[0] + _spaces(tok[1:-1]) + tok[-1])
        else:
            code.append(_spaces(tok))
            bare.append(_spaces(tok))
        at = m.end()
    code.append(raw[at:])
    bare.append(raw[at:])
    return "".join(code), "".join(bare)


def match_close(text: str, open_at: int) -> int:
    """Index of the bracket closing the one at `open_at` (in a bare view, so literals hold no brackets); -1 if none."""
    pairs = {"(": ")", "{": "}", "[": "]"}
    opener = text[open_at]
    closer = pairs[opener]
    depth = 0
    for i in range(open_at, len(text)):
        c = text[i]
        if c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return i
    return -1


def block_end(text: str, start: int) -> int:
    """The index of the `}` closing the innermost block that contains `start`."""
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            if depth == 0:
                return i
            depth -= 1
    return len(text)


HEAD_RE = re.compile(r"^(?![ \t#/])[^\n;{}]*?\b((?:[A-Za-z_]\w*::)*~?[A-Za-z_]\w*)\s*\(", re.M)


class Source:
    """One file's views and its out-of-line function definitions (heads at column 0)."""

    def __init__(self, raw: str):
        self.raw = raw
        self.code, self.bare = views(raw)
        self._functions = None

    @property
    def functions(self):
        if self._functions is None:
            out = []
            bare = self.bare
            for m in HEAD_RE.finditer(bare):
                lp = bare.find("(", m.end(1))
                rp = match_close(bare, lp)
                if rp < 0:
                    continue
                j = rp + 1
                depth = 0
                brace = -1
                while j < len(bare):
                    c = bare[j]
                    if c == "(":
                        depth += 1
                    elif c == ")":
                        depth -= 1
                    elif depth == 0 and c == ";":
                        break
                    elif depth == 0 and c == "{":
                        brace = j
                        break
                    j += 1
                if brace < 0:
                    continue
                close = match_close(bare, brace)
                if close < 0:
                    continue
                out.append({"name": m.group(1), "head": m.start(), "name_at": m.start(1),
                            "params": bare[lp + 1:rp], "open": brace, "close": close})
            self._functions = out
        return self._functions

    def enclosing(self, pos: int):
        best = None
        for f in self.functions:
            if f["open"] < pos < f["close"] and (best is None or f["open"] > best["open"]):
                best = f
        return best

    def line(self, pos: int) -> int:
        return self.raw.count("\n", 0, pos) + 1


# ---------------------------------------------------------------------------------------------------------------
# The listed bodies (§7 C4: the begin entry, RETURN's filing and release, the owner-end walk, release_marked_chunk and
# its cores, release_zone_chunk, release_chunk_backing, host_release_chunk_lease, release_chunk_lease,
# free_chunk_owner, the carrier's destructor, footprint_locked, reconcile_stranded_locked, the registry accessor,
# grow_into, ~pinned_chunk_pool, configure_host_zones and marked_entries), plus the seven decision bodies of clause 10.
# `whole` names the lock a body runs wholly under: the `_locked` helpers and grow_into under the pool's mutex_ (their
# callers hold it), the cache's cores under marked_set_mutex_.
# ---------------------------------------------------------------------------------------------------------------

BODY_SPECS = {
    "begin": (CACHE, "unified_cache::host_scratch_begin_release", None, None),
    "return": (CACHE, "unified_cache::host_scratch_return_chunks", None, None),
    "owner_end": (CACHE, "unified_cache::host_scratch_owner_end", None, None),
    "public": (CACHE, "unified_cache::release_marked_chunk", None, None),
    "core_vec": (CACHE, "unified_cache::release_marked_chunk_locked", r"std::vector\s*<", "marked_set_mutex_"),
    # The single-slot overload, selected by signature: its out parameter is a released_chunk with exactly one `&`.
    "core_slot": (CACHE, "unified_cache::release_marked_chunk_locked", r"released_chunk\s*&(?!&)\s*\w+\s*$",
                  "marked_set_mutex_"),
    "lease": (CACHE, "unified_cache::host_release_chunk_lease", None, None),
    "configure": (CACHE, "unified_cache::configure_host_zones", None, None),
    "marked_entries": (CACHE, "unified_cache::marked_entries", r"^\s*$", None),
    "marked_entries_chunk": (CACHE, "unified_cache::marked_entries", r"size_t", None),
    "accessor": (CACHE, "unified_allocation_registered_exact", None, None),
    "release_zone_chunk": (POOL, "pinned_chunk_pool::release_zone_chunk", None, None),
    "release_chunk_backing": (POOL, "pinned_chunk_pool::release_chunk_backing", None, None),
    "release_chunk_lease": (POOL, "pinned_chunk_pool::release_chunk_lease", None, None),
    "free_chunk_owner": (POOL, "pinned_chunk_pool::free_chunk_owner", None, None),
    "carrier_dtor": (POOL, "pinned_chunk_pool::released_chunk::~released_chunk", None, None),
    "footprint_locked": (POOL, "pinned_chunk_pool::footprint_locked", None, "mutex_"),
    "reconcile": (POOL, "pinned_chunk_pool::reconcile_stranded_locked", None, "mutex_"),
    "grow_into": (POOL, "pinned_chunk_pool::grow_into", None, "mutex_"),
    "pool_dtor": (POOL, "pinned_chunk_pool::~pinned_chunk_pool", None, None),
    "allocate_segmented": (POOL, "pinned_chunk_pool::allocate_segmented", None, None),
    "pre_allocate": (POOL, "pinned_chunk_pool::pre_allocate", None, None),
    "configure_zones": (POOL, "pinned_chunk_pool::configure_zones", None, None),
    "grow_zone": (POOL, "pinned_chunk_pool::grow_zone", None, None),
    "allocate_from_chunks": (POOL, "pinned_chunk_pool::allocate_from_chunks", None, None),
    "allocated": (POOL, "pinned_chunk_pool::allocated", None, None),
}

CACHE_BODIES = ["begin", "return", "owner_end", "public", "core_vec", "core_slot", "lease", "configure",
                "marked_entries", "marked_entries_chunk", "accessor"]
POOL_BODIES = ["release_zone_chunk", "release_chunk_backing", "release_chunk_lease", "free_chunk_owner",
               "carrier_dtor", "footprint_locked", "reconcile", "grow_into", "pool_dtor"]
WALKS = ["begin", "owner_end", "return", "configure"]
DECISION_BODIES = ["allocate_segmented", "pre_allocate", "configure_zones", "grow_zone", "allocate_from_chunks",
                   "grow_into", "allocated"]
# Every frame the lease drop runs through (§7 C4 clause 7; §M110): none of them may allocate.
LEASE_DROP_PATH = ["lease", "release_chunk_lease", "public", "core_slot", "release_zone_chunk", "carrier_dtor",
                   "free_chunk_owner", "accessor"]

LOCK_RE = re.compile(r"std::(?:lock_guard|unique_lock|scoped_lock)\s*(?:<[^;>]*>)?\s+\w+\s*[({]([^;]*?)[)}]\s*;")
POOL_LOCK = "mutex_"
MARKED_LOCK = "marked_set_mutex_"

# Clause 10's allow-list, per mention: (body, ordinal counting every mention of total_allocated_ in that body in
# source order, exempt or not, statement kind). Exempt: only the numerator of a division inside a GGML_LOG_* call.
FOOTPRINT_ALLOWED = {
    ("footprint_locked", 1): "return",
    ("reconcile", 1): "subtract",
    ("carrier_dtor", 1): "subtract",
    ("grow_into", 2): "add",
    ("pool_dtor", 1): "reset",
}
FOOTPRINT_BODY_MENTIONS = {"footprint_locked": 1, "reconcile": 1, "carrier_dtor": 1, "grow_into": 4, "pool_dtor": 1}
FOOTPRINT_EXEMPT_LOG_READS = 9

# Clause 10's caller census of the pinned committed accessor (design §4.4 "footprint_locked()": none of its callers
# holds g_runtime_alloc_mutex, so the reconcile's registry query under the pool's mutex_ is acyclic). Written as a
# count so the next commit that adds a caller raises it by one and names its reason here.
#   ggml_sycl_configure_host_zones_for_plan: the plan's pre-size, its commit check and its log (3 reads).
#   ggml_sycl_log_host_mem: the host memory phase line (2 reads).
#   stage_trace_sample: the staging trace (1 read).
#   host_ledger_pinned_share: the host ledger's pinned share (1 read; the th32 ledger's lock-free reader).
COMMITTED_CALLERS = {
    "ggml_sycl_configure_host_zones_for_plan": 3,
    "ggml_sycl_log_host_mem": 2,
    "stage_trace_sample": 1,
    "host_ledger_pinned_share": 1,
}
COMMITTED_CALLER_COUNT = 4
COMMITTED_ACCESSOR_DEF = re.compile(r"size_t\s+pinned_pool_committed\s*\(\s*\)\s*const\s*\{")

GENERIC_LOCK_NAMES = {"mutex_", "mutex", "mtx", "g_mutex"}

# ---------------------------------------------------------------------------------------------------------------
# Tree access.
# ---------------------------------------------------------------------------------------------------------------

_SOURCES = {}


def source_for(rel: str, raw: str) -> Source:
    """A parsed Source, reused while the text is unchanged (each control edits one file; the rest parse once)."""
    hit = _SOURCES.get(rel)
    if hit is not None and hit.raw == raw:
        return hit
    s = Source(raw)
    _SOURCES[rel] = s
    return s


def tree_texts() -> dict:
    texts = {}
    for rel in (POOL, CACHE, CANON):
        p = ROOT / rel
        texts[rel] = p.read_text(errors="replace") if p.exists() else ""
    return texts


def committed_census_files(texts: dict) -> dict:
    """The production SYCL sources that name the committed accessor (the tests directory excluded)."""
    out = {}
    base = ROOT / SYCL_DIR
    for p in sorted(base.glob("*")):
        if p.suffix not in (".cpp", ".hpp") or not p.is_file():
            continue
        rel = f"{SYCL_DIR}/{p.name}"
        raw = texts.get(rel)
        if raw is None:
            raw = p.read_text(errors="replace")
        if "pinned_pool_committed" in raw:
            out[rel] = raw
    for rel, raw in texts.items():
        if rel.startswith(SYCL_DIR + "/") and "pinned_pool_committed" in raw:
            out[rel] = raw
    return out


class Tree:
    def __init__(self, texts: dict):
        self.texts = texts
        self.src = {rel: source_for(rel, raw) for rel, raw in texts.items() if rel != CANON}
        self.canon = texts.get(CANON, "")
        self.bodies = {}
        self.missing = []
        for key, (rel, name, pred, _whole) in BODY_SPECS.items():
            s = self.src[rel]
            hits = [f for f in s.functions if f["name"] == name and (pred is None or re.search(pred, f["params"]))]
            if len(hits) == 1:
                self.bodies[key] = hits[0]
            else:
                self.missing.append((key, len(hits)))
        self.regions = {rel: [] for rel in self.src}
        for key, f in self.bodies.items():
            rel = BODY_SPECS[key][0]
            bare = self.src[rel].bare
            for m in LOCK_RE.finditer(bare, f["open"], f["close"]):
                names = re.findall(r"[A-Za-z_]\w*", m.group(1))
                if not names:
                    continue
                self.regions[rel].append({"lock": names[-1], "start": m.start(), "stmt_end": m.end(),
                                          "end": block_end(bare, m.end()), "body": key})

    def body_text(self, key: str, view: str = "bare") -> str:
        f = self.bodies.get(key)
        if f is None:
            return ""
        s = self.src[BODY_SPECS[key][0]]
        return getattr(s, view)[f["open"]:f["close"] + 1]

    def body_regions(self, key: str, lock: str):
        return [r for r in self.regions[BODY_SPECS[key][0]] if r["body"] == key and r["lock"] == lock]

    def held(self, rel: str, pos: int, lock: str) -> bool:
        for r in self.regions[rel]:
            if r["lock"] == lock and r["stmt_end"] <= pos < r["end"]:
                return True
        for key, f in self.bodies.items():
            spec = BODY_SPECS[key]
            if spec[0] == rel and spec[3] == lock and f["open"] < pos < f["close"]:
                return True
        return False

    def body_key_at(self, rel: str, pos: int):
        for key, f in self.bodies.items():
            if BODY_SPECS[key][0] == rel and f["open"] < pos < f["close"]:
                return key
        return None


def call_args(bare: str, name_end: int):
    """The top-level argument texts of the call whose `(` follows `name_end`."""
    lp = bare.find("(", name_end)
    rp = match_close(bare, lp)
    inner = bare[lp + 1:rp]
    args = []
    depth = 0
    cur = ""
    for c in inner:
        if c in "([{<" and c != "<":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if c == "," and depth == 0:
            args.append(cur)
            cur = ""
        else:
            cur += c
    if cur.strip():
        args.append(cur)
    return args


def if_blocks(bare: str, lo: int, hi: int):
    """Every `if (cond) { then } [else { otherwise }]` in [lo, hi): (cond, then_span, else_span or None)."""
    out = []
    for m in re.finditer(r"\bif\s*\(", bare[lo:hi]):
        lp = lo + m.end() - 1
        rp = match_close(bare, lp)
        k = rp + 1
        while k < hi and bare[k].isspace():
            k += 1
        if k >= hi or bare[k] != "{":
            continue
        then_close = match_close(bare, k)
        else_span = None
        e = then_close + 1
        while e < hi and bare[e].isspace():
            e += 1
        if bare.startswith("else", e):
            b = e + 4
            while b < hi and bare[b].isspace():
                b += 1
            if b < hi and bare[b] == "{":
                else_span = (b, match_close(bare, b))
        out.append((bare[lp + 1:rp], (k, then_close), else_span))
    return out


def statements(body_bare: str):
    """The top-level statements of a body text that starts with `{`, as (start, end) offsets into it."""
    out = []
    depth = 0
    start = 1
    i = 1
    while i < len(body_bare) - 1:
        c = body_bare[i]
        if c in "({[":
            depth += 1
        elif c in ")}]":
            depth -= 1
            if depth == 0 and c == "}":
                # A block statement (if/for/while/scope) ends at its closing brace, unless an else follows.
                rest = body_bare[i + 1:].lstrip()
                if not rest.startswith("else") and not rest.startswith(";"):
                    out.append((start, i + 1))
                    start = i + 1
        elif c == ";" and depth == 0:
            out.append((start, i + 1))
            start = i + 1
        i += 1
    return [(a, b) for a, b in out if body_bare[a:b].strip()]


# ---------------------------------------------------------------------------------------------------------------
# The canon §12.5 census.
# ---------------------------------------------------------------------------------------------------------------


def canon_rank_rows(canon: str) -> dict:
    at = canon.find("### 12.5")
    if at < 0:
        return {}
    end = canon.find("\n### ", at + 1)
    section = canon[at:end if end > 0 else len(canon)]
    rows = {}
    for line in section.split("\n"):
        m = re.match(r"\|\s*(L[1-5]|isolated [A-Z])\s*\|", line)
        if m:
            rows[m.group(1)] = line
    return rows


def canon_census_names(canon: str) -> set:
    names = set()
    for row in canon_rank_rows(canon).values():
        for tok in re.findall(r"`([^`]+)`", row):
            t = re.sub(r"\(\)$", "", tok.strip())
            t = re.sub(r"\[[^\]]*\]", "", t)
            t = re.split(r"::|\.|->", t)[-1]
            if re.fullmatch(r"[A-Za-z_]\w*", t) and re.search(r"mutex|mtx|_lock$", t) and t not in GENERIC_LOCK_NAMES:
                names.add(t)
    # Held around owner releases and taken by the enqueue paths: named here whether or not the table carries it yet.
    names.add("deferred_frees_mutex_")
    return names


# ---------------------------------------------------------------------------------------------------------------
# The clauses. Each appends (reason, detail); a planted control passes when its reason is among the failures.
# ---------------------------------------------------------------------------------------------------------------


def clause_bodies_present(t: Tree, fails: list):
    for key, n in t.missing:
        fails.append(("listed body not found", f"{key}: {BODY_SPECS[key][1]} matched {n} definition(s)"))


def clause_canon_ranks(t: Tree, fails: list):
    rows = canon_rank_rows(t.canon)
    want = [("L4", "`unified_cache::marked_set_mutex_`"), ("L4", "`unified_cache::deferred_frees_mutex_`"),
            ("L5", "`pinned_chunk_pool::mutex_`")]
    for rank, name in want:
        if name not in rows.get(rank, ""):
            fails.append(("canon §12.5 does not rank the lock", f"{name} not in the {rank} row"))
    l4 = rows.get("L4", "")
    if "marked_set_mutex_" in l4 and l4.find("marked_set_mutex_") < l4.find("managed_host_pinned_buffers_mutex"):
        fails.append(("canon §12.5 does not rank the lock", "marked_set_mutex_ is not last in L4"))
    l5 = rows.get("L5", "")
    if "pinned_chunk_pool::mutex_" in l5 and l5.find("pinned_chunk_pool::mutex_") > l5.find("vram_zone::alloc_mutex"):
        fails.append(("canon §12.5 does not rank the lock", "pinned_chunk_pool::mutex_ is not first in L5"))


def clause_1(t: Tree, fails: list):
    s = t.src[CACHE]
    for m in re.finditer(r"\bmarked_set_\b", s.bare):
        pos = m.start()
        key = t.body_key_at(CACHE, pos)
        if key is None or key not in CACHE_BODIES:
            fails.append(("marked set access outside the listed bodies", f"{CACHE}:{s.line(pos)}"))
            continue
        if not t.held(CACHE, pos, MARKED_LOCK):
            fails.append(("marked set access outside marked_set_mutex_", f"{key} {CACHE}:{s.line(pos)}"))


def clause_2(t: Tree, fails: list):
    s = t.src[CACHE]
    for key in CACHE_BODIES:
        regions = t.body_regions(key, MARKED_LOCK)
        if not regions:
            continue
        f = t.bodies[key]
        first = min(r["start"] for r in regions)
        for m in re.finditer(r"\bhost_arena_\s*->", s.bare[f["open"]:first]):
            fails.append(("pool call before marked_set_mutex_", f"{key} {CACHE}:{s.line(f['open'] + m.start())}"))


def clause_3(t: Tree, fails: list):
    s = t.src[POOL]
    for key in POOL_BODIES:
        f = t.bodies.get(key)
        if f is None:
            continue
        for m in re.finditer(r"\brelease_marked_chunk\w*\s*\(|\bmarked_set_mutex_\b", s.bare[f["open"]:f["close"]]):
            pos = f["open"] + m.start()
            if t.held(POOL, pos, POOL_LOCK):
                fails.append(("pool mutex_ held across release_marked_chunk", f"{key} {POOL}:{s.line(pos)}"))


def clause_4(t: Tree, fails: list):
    census = canon_census_names(t.canon)
    for rel in (CACHE, POOL):
        s = t.src[rel]
        for r in t.regions[rel]:
            if r["lock"] not in (MARKED_LOCK, POOL_LOCK):
                continue
            seg = s.bare[r["stmt_end"]:r["end"]]
            for m in re.finditer(r"[A-Za-z_]\w*", seg):
                if m.group(0) in census:
                    fails.append((f"census lock {m.group(0)} under {r['lock']}",
                                  f"{r['body']} {rel}:{s.line(r['stmt_end'] + m.start())}"))
        for key, f in t.bodies.items():
            spec = BODY_SPECS[key]
            if spec[0] != rel or spec[3] is None:
                continue
            for m in re.finditer(r"[A-Za-z_]\w*", s.bare[f["open"]:f["close"]]):
                if m.group(0) in census:
                    fails.append((f"census lock {m.group(0)} under {spec[3]}",
                                  f"{key} {rel}:{s.line(f['open'] + m.start())}"))


def clause_5(t: Tree, fails: list):
    for rel in (CACHE, POOL):
        s = t.src[rel]
        for m in re.finditer(r"\brelease_chunk_backing\s*\(", s.bare):
            pos = m.start()
            f = s.enclosing(pos)
            if f is None:
                continue
            if t.held(rel, pos, MARKED_LOCK):
                fails.append(("backing release under marked_set_mutex_", f"{rel}:{s.line(pos)}"))
            if t.held(rel, pos, POOL_LOCK):
                fails.append(("backing release under pool mutex_", f"{rel}:{s.line(pos)}"))
    s = t.src[CACHE]
    for key in WALKS:
        f = t.bodies.get(key)
        regions = t.body_regions(key, MARKED_LOCK)
        if f is None or not regions:
            fails.append(("walk holds no marked_set_mutex_ scope", key))
            continue
        lock_at = min(r["start"] for r in regions)
        scope_end = max(r["end"] for r in regions)
        seg = s.bare[f["open"]:f["close"]]
        for m in re.finditer(r"\brelease_chunk_backing\s*\(", seg):
            if f["open"] + m.start() < scope_end:
                fails.append(("collected backings released under marked_set_mutex_",
                              f"{key} {CACHE}:{s.line(f['open'] + m.start())}"))
        vectors = list(re.finditer(r"std::vector\s*<\s*pinned_chunk_pool::released_chunk\s*>\s+\w+", seg))
        if not vectors:
            fails.append(("walk has no collecting vector", key))
        for m in vectors:
            if f["open"] + m.start() > lock_at:
                fails.append(("collected vector declared inside the marked_set_mutex_ scope",
                              f"{key} {CACHE}:{s.line(f['open'] + m.start())}"))
    f = t.bodies.get("public")
    regions = t.body_regions("public", MARKED_LOCK)
    if f is not None and regions:
        lock_at = min(r["start"] for r in regions)
        slots = list(re.finditer(r"pinned_chunk_pool::released_chunk\s+\w+\s*[;{]", s.bare[f["open"]:f["close"]]))
        if not slots:
            fails.append(("released_chunk slot declared inside the marked_set_mutex_ scope", "public: no slot"))
        for m in slots:
            if f["open"] + m.start() > lock_at:
                fails.append(("released_chunk slot declared inside the marked_set_mutex_ scope",
                              f"public {CACHE}:{s.line(f['open'] + m.start())}"))


def clause_6(t: Tree, fails: list):
    f = t.bodies.get("configure")
    if f is None:
        return
    bare = t.src[CACHE].bare
    found = False
    for cond, then_span, _else in if_blocks(bare, f["open"], f["close"]):
        if not re.search(r"!\s*host_arena_\s*->\s*configure_zones\s*\(", cond):
            continue
        found = True
        then = bare[then_span[0]:then_span[1]]
        ret = re.search(r"\breturn\s+false\s*;", then)
        strict = re.search(r"\bggml_sycl_strict_enabled\s*\(\s*\)", then)
        if ret is None or strict is None or strict.start() > ret.start():
            fails.append(("configure_host_zones returns the refusal without the strict read",
                          f"{CACHE}:{t.src[CACHE].line(then_span[0])}"))
    if not found:
        fails.append(("configure_host_zones returns the refusal without the strict read", "no refusal branch"))


ALLOC_TOKENS = re.compile(r"\breserve\s*\(|\bresize\s*\(|\bpush_back\s*\(|\bvector\s*<")


def clause_7(t: Tree, fails: list):
    s = t.src[CACHE]
    # A definition's head names the function qualified; skip a match whose identifier ends where a head's name ends.
    heads = {f["name_at"] + len(f["name"]) for f in s.functions}
    for m in re.finditer(r"\brelease_marked_chunk_locked\s*\(", s.bare):
        if m.start() + len("release_marked_chunk_locked") in heads:
            continue
        pos = m.start()
        line = s.line(pos)
        if not t.held(CACHE, pos, MARKED_LOCK):
            fails.append(("release_marked_chunk_locked called without marked_set_mutex_", f"{CACHE}:{line}"))
        if len(call_args(s.bare, m.end() - 1)) == 2 and t.body_key_at(CACHE, pos) != "public":
            fails.append(("single-slot core called outside the public form", f"{CACHE}:{line}"))
    for m in re.finditer(r"\brelease_marked_chunk\s*\(", s.bare):
        if m.start() + len("release_marked_chunk") in heads:
            continue
        if t.held(CACHE, m.start(), MARKED_LOCK):
            fails.append(("release_marked_chunk called under marked_set_mutex_", f"{CACHE}:{s.line(m.start())}"))
    pub = t.body_text("public")
    if pub and not any(len(call_args(pub, m.end() - 1)) == 2
                       for m in re.finditer(r"\brelease_marked_chunk_locked\s*\(", pub)):
        fails.append(("public form does not call the single-slot core", "public"))
    lease = t.body_text("lease")
    if lease:
        if re.search(r"\brelease_marked_chunk_locked\s*\(", lease):
            fails.append(("lease drop calls release_marked_chunk_locked", "lease"))
        if not re.search(r"\brelease_marked_chunk\s*\(", lease):
            fails.append(("lease drop does not call release_marked_chunk", "lease"))
    for key in LEASE_DROP_PATH:
        text = t.body_text(key)
        for m in ALLOC_TOKENS.finditer(text):
            fails.append(("lease-drop path allocates", f"{key}: {m.group(0).strip()}"))


def clause_8(t: Tree, fails: list):
    s = t.src[POOL]
    f = t.bodies.get("carrier_dtor")
    if f is not None:
        bare = s.bare
        lo, hi = f["open"], f["close"]
        seg = bare[lo:hi]
        sub = re.search(r"\btotal_allocated_\s*-=", seg)
        if sub is None:
            fails.append(("released_chunk destructor does not subtract the footprint", "carrier_dtor"))
        else:
            sub_at = lo + sub.start()
            free_m = re.search(r"\bfree_chunk_owner\s*\(", seg)
            query = re.search(r"\bunified_allocation_registered_exact\s*\(", seg)
            meta = re.search(r"\bowner_metadata\s*\(", seg)
            if free_m is None:
                fails.append(("released_chunk destructor does not free the owner", "carrier_dtor"))
            if meta is None or (free_m is not None and meta.start() > free_m.start()):
                fails.append(("released_chunk destructor reads the metadata after the owner's release", "carrier_dtor"))
            if query is None or lo + query.start() > sub_at or (free_m is not None and query.start() < free_m.start()):
                fails.append(("released_chunk destructor subtracts without the registry query", "carrier_dtor"))
            if not t.held(POOL, sub_at, POOL_LOCK):
                fails.append(("released_chunk destructor subtracts outside the pool mutex_", "carrier_dtor"))
            enclosing = None
            for cond, then_span, else_span in if_blocks(bare, lo, hi):
                for side, span in (("then", then_span), ("else", else_span)):
                    if span and span[0] < sub_at < span[1]:
                        if enclosing is None or span[0] > enclosing[2][0]:
                            enclosing = (cond, side, span, then_span, else_span)
            gone_eq = None
            if enclosing is not None:
                cond, side = enclosing[0], enclosing[1]
                eq = re.search(r"==\s*ALLOCATION_REGISTERED_GONE|ALLOCATION_REGISTERED_GONE\s*==", cond)
                ne = re.search(r"!=\s*ALLOCATION_REGISTERED_GONE|ALLOCATION_REGISTERED_GONE\s*!=", cond)
                gone_eq = (side == "then" and eq) or (side == "else" and ne)
            if not gone_eq:
                fails.append(("released_chunk destructor subtracts on the registered branch", "carrier_dtor"))
                fails.append(("released_chunk kept branch does not strand the footprint", "carrier_dtor"))
            else:
                other = enclosing[4] if enclosing[1] == "then" else enclosing[3]
                kept = bare[other[0]:other[1]] if other else ""
                if not re.search(r"\bstranded_\s*\.\s*emplace_back\s*\(", kept) or "total_allocated_" in kept:
                    fails.append(("released_chunk kept branch does not strand the footprint", "carrier_dtor"))
        if re.search(r"\bggml_sycl_strict_enabled\b", seg):
            fails.append(("released_chunk destructor names ggml_sycl_strict_enabled", "carrier_dtor"))
        code_seg = s.code[lo:hi]
        for m in re.finditer(r"\bGGML_LOG_WARN\s*\(", code_seg):
            lp = lo + m.end() - 1
            args = s.code[lp:match_close(bare, lp) + 1]
            for lit in re.findall(r'"((?:\\.|[^"\\])*)"', args):
                cause = re.search(r"(?i)\b(retr(y|ies)|scheduled|leak\w*|ownerless)\b", lit)
                if cause:
                    fails.append(("released_chunk WARN names a cause", cause.group(0)))
    for m in re.finditer(r"\btotal_allocated_\s*-=", s.bare):
        key = t.body_key_at(POOL, m.start())
        if key not in ("carrier_dtor", "reconcile"):
            fails.append(("footprint subtracted outside the released_chunk destructor", f"{POOL}:{s.line(m.start())}"))


def clause_9(t: Tree, fails: list):
    f = t.bodies.get("accessor")
    if f is None:
        return
    seg = t.body_text("accessor")
    locks = [re.findall(r"[A-Za-z_]\w*", m.group(1))[-1] for m in LOCK_RE.finditer(seg)]
    if locks != ["g_runtime_alloc_mutex"]:
        fails.append(("accessor lock is not exactly g_runtime_alloc_mutex", str(locks)))
    if not re.search(r"\bkey\s*\(\s*\)", seg):
        fails.append(("accessor compares ptr only", "accessor"))
    if re.search(r"\.\s*erase\s*\(|\.\s*emplace\w*\s*\(|\.\s*insert\s*\(|\[|->\s*second(?:\s*\.\s*\w+)*\s*=(?!=)|"
                 r"\.\s*second(?:\s*\.\s*\w+)*\s*=(?!=)", seg):
        fails.append(("accessor writes the registry", "accessor"))


def _log_spans(bare: str):
    spans = []
    for m in re.finditer(r"\bGGML_LOG_\w+\s*\(", bare):
        lp = m.end() - 1
        spans.append((lp, match_close(bare, lp)))
    return spans


def clause_10(t: Tree, fails: list):
    s = t.src[POOL]
    bare = s.bare
    logs = _log_spans(bare)
    exempt = 0
    per_body = {}
    for m in re.finditer(r"\btotal_allocated_\b", bare):
        pos = m.start()
        key = t.body_key_at(POOL, pos)
        f = s.enclosing(pos)
        body_name = key if key is not None else (f["name"] if f else "<file scope>")
        ordinal = per_body.get(body_name, 0) + 1
        per_body[body_name] = ordinal
        after = bare[m.end():m.end() + 8]
        before = bare[max(0, pos - 12):pos]
        in_log = any(a < pos < b for a, b in logs)
        if in_log and re.match(r"\s*/", after):
            exempt += 1
            continue
        if re.match(r"\s*-=", after):
            kind = "subtract"
        elif re.match(r"\s*\+=", after):
            kind = "add"
        elif re.match(r"\s*=\s*0\s*;", after):
            kind = "reset"
        elif re.match(r"\s*=(?!=)", after):
            kind = "assign"
        elif re.search(r"\breturn\s+$", before):
            kind = "return"
        else:
            kind = "read"
        if FOOTPRINT_ALLOWED.get((body_name, ordinal)) == kind:
            continue
        if kind in ("subtract", "add", "reset", "assign"):
            fails.append(("total_allocated_ written outside the allow-list", f"{body_name} #{ordinal} {kind}"))
        else:
            fails.append((f"{body_name} reads total_allocated_ outside a log call", f"{POOL}:{s.line(pos)}"))
    if exempt != FOOTPRINT_EXEMPT_LOG_READS:
        fails.append(("exempt log mentions are not exactly nine", f"{exempt} exempt"))
    for body_name, n in FOOTPRINT_BODY_MENTIONS.items():
        if per_body.get(body_name, 0) != n:
            fails.append(("allow-listed mention count differs", f"{body_name}: {per_body.get(body_name, 0)} != {n}"))
    for key in DECISION_BODIES:
        if key in t.bodies and not re.search(r"\bfootprint_locked\s*\(\s*\)", t.body_text(key)):
            fails.append((f"{key} does not call footprint_locked", key))
    fp = t.body_text("footprint_locked")
    if fp:
        rec = re.search(r"\breconcile_stranded_locked\s*\(", fp)
        ret = re.search(r"\breturn\b", fp)
        if rec is None or ret is None or rec.start() > ret.start():
            fails.append(("footprint_locked returns total_allocated_ without reconcile_stranded_locked",
                          "footprint_locked"))
    rc = t.body_text("reconcile")
    if rc and not re.search(r"\bunified_allocation_registered_exact\s*\(", rc):
        fails.append(("stranded prune without registry query", "reconcile"))
    # The committed accessor's callers: each a function that holds no g_runtime_alloc_mutex.
    callers = {}
    for rel, raw in committed_census_files(t.texts).items():
        cs = t.src[rel] if rel in t.src else source_for(rel, raw)
        for m in re.finditer(r"\bpinned_pool_committed\s*\(", cs.bare):
            line_start = cs.bare.rfind("\n", 0, m.start()) + 1
            if COMMITTED_ACCESSOR_DEF.match(cs.bare[line_start:].lstrip()):
                continue
            f = cs.enclosing(m.start())
            name = f["name"].split("::")[-1] if f else f"<file scope {rel}>"
            callers[name] = callers.get(name, 0) + 1
            if f is not None and "g_runtime_alloc_mutex" in cs.bare[f["open"]:f["close"]]:
                fails.append(("pinned_pool_committed caller holds g_runtime_alloc_mutex", name))
    if len(callers) != COMMITTED_CALLER_COUNT or callers != COMMITTED_CALLERS:
        fails.append(("pinned_pool_committed caller count differs", f"{callers}"))


def clause_11(t: Tree, fails: list):
    s = t.src[POOL]
    f = t.bodies.get("grow_into")
    if f is not None:
        lo, hi = f["open"], f["close"]
        seg = s.bare[lo:hi]
        res = re.search(r"\bstranded_\s*\.\s*reserve\s*\(", seg)
        if res is None:
            fails.append(("chunk added without stranded reserve", "grow_into"))
        else:
            args = call_args(s.bare, lo + res.end() - 1)
            arg = args[0] if args else ""
            by_count = re.search(r"\bchunks_\s*\.\s*size\s*\(", arg)
            if not (by_count and re.search(r"\bruntime_chunks_\s*\.\s*size\s*\(", arg)):
                fails.append(("stranded reserve not sized by the chunk count", arg.strip()))
            owner = re.search(r"\ballocate_pinned_chunk_owner\s*\(", seg)
            budget = re.search(r"\bfootprint_locked\s*\(\s*\)\s*\+[^;]*>\s*budget_", seg)
            if owner is None or res.start() > owner.start():
                fails.append(("stranded reserve after the owner allocation", "grow_into"))
            if budget is None or res.start() < budget.start():
                fails.append(("stranded reserve before the budget check", "grow_into"))
            caught = False
            for tm in re.finditer(r"\btry\s*\{", seg):
                t_open = lo + tm.end() - 1
                t_close = match_close(s.bare, t_open)
                if not (t_open < lo + res.start() < t_close):
                    continue
                rest = s.bare[t_close + 1:hi]
                cm = re.match(r"\s*catch\s*\(\s*const\s+std::bad_alloc\s*&\s*\w*\s*\)\s*\{", rest)
                if cm:
                    c_open = t_close + 1 + cm.end() - 1
                    if re.search(r"\breturn\s+false\s*;", s.bare[c_open:match_close(s.bare, c_open)]):
                        caught = True
            if not caught:
                fails.append(("stranded reserve without bad_alloc catch", "grow_into"))
    f = t.bodies.get("carrier_dtor")
    if f is not None:
        seg = t.body_text("carrier_dtor")
        code_seg = t.body_text("carrier_dtor", "code")
        app = re.search(r"\bstranded_\s*\.\s*emplace_back\s*\(", seg)
        guard = re.search(r"stranded_\s*\.\s*size\s*\(\s*\)\s*(>=|<)\s*[\w>\-\.]*stranded_\s*\.\s*capacity\s*\(\s*\)",
                          seg)
        msg = "stranded_ capacity %zu exhausted at %zu entries" in code_seg
        if app is None or guard is None or not msg or guard.start() > app.start():
            fails.append(("capacity assertion missing or after the append", "carrier_dtor"))
        if re.search(r"\breserve\s*\(|\bresize\s*\(|\bpush_back\s*\(|\bvector\s*<|\binsert\s*\(", seg):
            fails.append(("released_chunk destructor allocates", "carrier_dtor"))
    rc = t.body_text("reconcile")
    if rc:
        in_place = re.search(r"\bremove_if\s*\(", rc) and re.search(r"\bstranded_\s*\.\s*erase\s*\(", rc)
        rebuild = re.search(r"\bstranded_\s*=(?!=)|\bswap\s*\(|\bstranded_\s*\.\s*clear\s*\(|\bvector\s*<", rc)
        if not in_place or rebuild:
            fails.append(("reconcile rebuilds stranded_", "reconcile"))


def clause_12(t: Tree, fails: list):
    f = t.bodies.get("pool_dtor")
    if f is not None:
        body = t.body_text("pool_dtor")
        if not re.search(r"\bowning_carriers_\b", body):
            fails.append(("carrier assertion absent", "pool_dtor"))
        else:
            sts = statements(body)
            first = body[sts[0][0]:sts[0][1]] if sts else ""
            if not (re.search(r"\bGGML_ASSERT\s*\(", first) and "owning_carriers_" in first):
                fails.append(("carrier assertion after the free loop", "pool_dtor"))
    f = t.bodies.get("carrier_dtor")
    if f is not None:
        body = t.body_text("carrier_dtor")
        sts = statements(body)
        last = body[sts[-1][0]:sts[-1][1]] if sts else ""
        if not re.search(r"\bowning_carriers_\s*\.\s*fetch_sub\s*\(", last):
            fails.append(("carrier count lowered before the footprint section", "carrier_dtor"))


CLAUSES = [clause_bodies_present, clause_canon_ranks, clause_1, clause_2, clause_3, clause_4, clause_5, clause_6,
           clause_7, clause_8, clause_9, clause_10, clause_11, clause_12]


def failures(texts: dict) -> list:
    t = Tree(texts)
    fails = []
    for clause in CLAUSES:
        clause(t, fails)
    return fails


def reasons(texts: dict) -> set:
    return {r for r, _ in failures(texts)}


# ---------------------------------------------------------------------------------------------------------------
# Planted controls: (id, clause, body key or None for the whole file, old text, new text, expected reason). Each
# replaces `old`, which must occur exactly once in the body's raw text, and the gate must fail with the reason.
# ---------------------------------------------------------------------------------------------------------------

V_RELEASED = "std::vector<pinned_chunk_pool::released_chunk> released;"
LOCK_MARKED = "std::lock_guard<std::mutex> lock(marked_set_mutex_);"
SINGLE_SLOT_VECTOR = "std::vector<pinned_chunk_pool::released_chunk> v;\n    v.reserve(1);\n"

CONTROLS = [
    ("c1-marked-entries-unlocked", 1, "marked_entries", "    " + LOCK_MARKED + "\n", "",
     "marked set access outside marked_set_mutex_"),
    ("c2-pool-call-before-lock", 2, "begin", "    std::vector<pinned_chunk_pool::released_chunk> released;",
     "    (void) host_arena_->zone_capacity(host_zone_id::SCRATCH);\n"
     "    std::vector<pinned_chunk_pool::released_chunk> released;",
     "pool call before marked_set_mutex_"),
    ("c3-pool-calls-release-marked", 3, "release_zone_chunk", "    states.erase(state);",
     "    states.erase(state);\n    (void) release_marked_chunk(chunk_idx);", "pool mutex_ held across release_marked_chunk"),
    ("c4-rw-under-marked", 4, "begin", "        for (const marked_chunk_entry & entry : marked_set_) {",
     "        std::shared_lock<std::shared_mutex> rw(rw_mutex_);\n"
     "        for (const marked_chunk_entry & entry : marked_set_) {",
     "census lock rw_mutex_ under marked_set_mutex_"),
    ("c4-deferred-under-marked", 4, "begin", "        for (const marked_chunk_entry & entry : marked_set_) {",
     "        std::lock_guard<std::mutex> deferred(deferred_frees_mutex_);\n"
     "        for (const marked_chunk_entry & entry : marked_set_) {",
     "census lock deferred_frees_mutex_ under marked_set_mutex_"),
    ("c5-backing-release-under-marked", 5, "public", "        status = release_marked_chunk_locked(chunk_idx, released);",
     "        status = release_marked_chunk_locked(chunk_idx, released);\n"
     "        pinned_chunk_pool::release_chunk_backing(std::move(released));",
     "backing release under marked_set_mutex_"),
    ("c5-walk-loop-inside-scope", 5, "begin",
     "                refusals.push_back({ candidate.chunk_idx, status });\n            }\n        }\n",
     "                refusals.push_back({ candidate.chunk_idx, status });\n            }\n        }\n"
     "        for (auto & carrier : released) {\n"
     "            pinned_chunk_pool::release_chunk_backing(std::move(carrier));\n        }\n",
     "collected backings released under marked_set_mutex_"),
    ("c5-backing-release-under-pool-mutex", 5, "release_zone_chunk", "    states.erase(state);",
     "    states.erase(state);\n    release_chunk_backing(std::move(out));", "backing release under pool mutex_"),
    ("c5-vector-inside-scope", 5, "owner_end",
     "    " + V_RELEASED + "\n    std::vector<marked_chunk_entry>                candidates;\n    {\n"
     "        " + LOCK_MARKED + "\n",
     "    std::vector<marked_chunk_entry>                candidates;\n    {\n        " + LOCK_MARKED + "\n"
     "        " + V_RELEASED + "\n",
     "collected vector declared inside the marked_set_mutex_ scope"),
    ("c5-slot-inside-scope", 5, "public",
     "    pinned_chunk_pool::released_chunk released;\n    pinned_chunk_release              status = "
     "PINNED_CHUNK_RELEASE_ALREADY;\n    {\n        " + LOCK_MARKED + "\n",
     "    pinned_chunk_release              status = PINNED_CHUNK_RELEASE_ALREADY;\n    {\n        "
     + LOCK_MARKED + "\n        pinned_chunk_pool::released_chunk released;\n",
     "released_chunk slot declared inside the marked_set_mutex_ scope"),
    ("c6-no-strict-read", 6, "configure",
     "        if (ggml_sycl_strict_enabled()) {\n            GGML_ABORT(", "        if (false) {\n            GGML_ABORT(",
     "configure_host_zones returns the refusal without the strict read"),
    ("c7-public-under-marked", 7, "begin", "        for (const marked_chunk_entry & entry : marked_set_) {",
     "        (void) release_marked_chunk(0);\n        for (const marked_chunk_entry & entry : marked_set_) {",
     "release_marked_chunk called under marked_set_mutex_"),
    ("c7-core-outside-scope", 7, "begin", "    size_t released_bytes = 0;",
     "    (void) release_marked_chunk_locked(0, released, false);\n    size_t released_bytes = 0;",
     "release_marked_chunk_locked called without marked_set_mutex_"),
    ("c7-lease-calls-core", 7, "lease", "        (void) release_marked_chunk(orphaned_chunk);",
     "        pinned_chunk_pool::released_chunk slot;\n"
     "        (void) release_marked_chunk_locked(orphaned_chunk, slot);",
     "lease drop calls release_marked_chunk_locked"),
    ("c7-public-reserves", 7, "public", "    pinned_chunk_release              status = PINNED_CHUNK_RELEASE_ALREADY;",
     "    pinned_chunk_release              status = PINNED_CHUNK_RELEASE_ALREADY;\n"
     "    marked_set_.reserve(1);", "lease-drop path allocates"),
    ("c7-single-slot-builds-vector", 7, "core_slot", "    const pinned_chunk_release status =",
     "    " + SINGLE_SLOT_VECTOR + "    const pinned_chunk_release status =", "lease-drop path allocates"),
    ("c7-release-zone-chunk-builds-vector", 7, "release_zone_chunk", "    states.erase(state);",
     "    states.erase(state);\n    " + SINGLE_SLOT_VECTOR, "lease-drop path allocates"),
    ("c7-release-chunk-lease-builds-vector", 7, "release_chunk_lease",
     "    const uint32_t prev = vec[idx].lease_count.fetch_sub(1);",
     "    " + SINGLE_SLOT_VECTOR + "    const uint32_t prev = vec[idx].lease_count.fetch_sub(1);",
     "lease-drop path allocates"),
    ("c7-free-chunk-owner-builds-vector", 7, "free_chunk_owner", "    if (c.owner.valid()) {",
     "    " + SINGLE_SLOT_VECTOR + "    if (c.owner.valid()) {", "lease-drop path allocates"),
    ("c7-accessor-builds-vector", 7, "accessor", "    std::lock_guard<std::mutex> lock(g_runtime_alloc_mutex);",
     "    std::lock_guard<std::mutex> lock(g_runtime_alloc_mutex);\n    " + SINGLE_SLOT_VECTOR,
     "lease-drop path allocates"),
    ("c8-no-subtraction", 8, "carrier_dtor", "            pool_->total_allocated_ -= backing_;", "",
     "released_chunk destructor does not subtract the footprint"),
    ("c8-release-zone-chunk-subtracts", 8, "release_zone_chunk", "    states.erase(state);",
     "    states.erase(state);\n    total_allocated_ -= c.size;",
     "footprint subtracted outside the released_chunk destructor"),
    ("c8-no-registry-query", 8, "carrier_dtor",
     "    const allocation_registered_state state = unified_allocation_registered_exact(metadata);",
     "    const allocation_registered_state state = ALLOCATION_REGISTERED_GONE;",
     "released_chunk destructor subtracts without the registry query"),
    ("c8-subtracts-on-registered-branch", 8, "carrier_dtor",
     "        if (state == ALLOCATION_REGISTERED_GONE) {", "        if (state != ALLOCATION_REGISTERED_GONE) {",
     "released_chunk destructor subtracts on the registered branch"),
    ("c8-kept-branch-no-append", 8, "carrier_dtor",
     "            pool_->stranded_.emplace_back(stranded_entry{ metadata, backing_ });", "",
     "released_chunk kept branch does not strand the footprint"),
    ("c8-strict-abort-restored", 8, "carrier_dtor",
     "            pool_->stranded_.emplace_back(stranded_entry{ metadata, backing_ });",
     "            pool_->stranded_.emplace_back(stranded_entry{ metadata, backing_ });\n"
     "            if (ggml_sycl_strict_enabled()) {\n                GGML_ABORT(\"stranded\");\n            }",
     "released_chunk destructor names ggml_sycl_strict_enabled"),
    ("c8-warn-names-a-cause", 8, "carrier_dtor", "footprint stranded until the registry drops it",
     "footprint stranded until the scheduled retry", "released_chunk WARN names a cause"),
    ("c9-ptr-only-compare", 9, "accessor",
     " || !(row->second.handle.key() == metadata.key())", "", "accessor compares ptr only"),
    ("c9-accessor-erases", 9, "accessor", "        return ALLOCATION_REGISTERED_GONE;\n    }",
     "        return ALLOCATION_REGISTERED_GONE;\n    }\n    g_runtime_alloc_registry.erase(row);",
     "accessor writes the registry"),
    ("c10-allocate-segmented-skips-reconcile", 10, "allocate_segmented", "if (footprint_locked() + chunk_size_",
     "if (total_allocated_ + chunk_size_", "allocate_segmented does not call footprint_locked"),
    ("c10-pre-allocate-skips-reconcile", 10, "pre_allocate", "if (footprint_locked() + chunk_size_",
     "if (total_allocated_ + chunk_size_", "pre_allocate does not call footprint_locked"),
    ("c10-configure-zones-skips-reconcile", 10, "configure_zones", "footprint_locked() + chunk_size_ <= budget_",
     "total_allocated_ + chunk_size_ <= budget_", "configure_zones does not call footprint_locked"),
    ("c10-grow-zone-skips-reconcile", 10, "grow_zone", "if (footprint_locked() + footprint.backing > budget_)",
     "if (total_allocated_ + footprint.backing > budget_)", "grow_zone does not call footprint_locked"),
    ("c10-allocate-from-chunks-skips-reconcile", 10, "allocate_from_chunks",
     "if (footprint_locked() + footprint.backing > budget_)", "if (total_allocated_ + footprint.backing > budget_)",
     "allocate_from_chunks does not call footprint_locked"),
    ("c10-grow-into-rereads-member", 10, "grow_into", "if (footprint_locked() + backing_size > budget_)",
     "if (total_allocated_ + backing_size > budget_)", "grow_into does not call footprint_locked"),
    ("c10-grow-into-rereads-member-count", 10, "grow_into", "if (footprint_locked() + backing_size > budget_)",
     "if (total_allocated_ + backing_size > budget_)", "allow-listed mention count differs"),
    ("c10-allocated-skips-reconcile", 10, "allocated", "return footprint_locked();", "return total_allocated_;",
     "allocated does not call footprint_locked"),
    ("c10-helper-skips-reconcile", 10, "footprint_locked", "    reconcile_stranded_locked();\n", "",
     "footprint_locked returns total_allocated_ without reconcile_stranded_locked"),
    ("c10-grow-zone-local-copy", 10, "grow_zone", "    const chunk_footprint footprint = chunk_footprint_for(additional_bytes);",
     "    const size_t snapshot = total_allocated_;\n    (void) snapshot;\n"
     "    const chunk_footprint footprint = chunk_footprint_for(additional_bytes);",
     "grow_zone reads total_allocated_ outside a log call"),
    ("c10-decision-body-writes", 10, "pre_allocate", "    size_t       chunks_grown  = 0;",
     "    size_t       chunks_grown  = 0;\n    total_allocated_ = 0;", "total_allocated_ written outside the allow-list"),
    ("c10-tenth-log-read", 10, "pre_allocate", "    size_t       chunks_grown  = 0;",
     "    size_t       chunks_grown  = 0;\n"
     "    GGML_LOG_INFO(\"[SYCL] pool footprint %.1f GB\\n\", total_allocated_ / (1024.0 * 1024.0 * 1024.0));",
     "exempt log mentions are not exactly nine"),
    ("c10-second-mention-in-allowed-body", 10, "footprint_locked", "    return total_allocated_;",
     "    return total_allocated_ + (total_allocated_ & 0);", "allow-listed mention count differs"),
    ("c10-prune-without-registry-query", 10, "reconcile",
     "if (unified_allocation_registered_exact(entry.metadata) != ALLOCATION_REGISTERED_GONE) {",
     "if (entry.backing == 0) {", "stranded prune without registry query"),
    ("c11-no-reserve", 11, "grow_into",
     "        stranded_.reserve(chunks_.size() + runtime_chunks_.size() + 1);", "        (void) 0;",
     "chunk added without stranded reserve"),
    ("c11-constant-reserve", 11, "grow_into",
     "        stranded_.reserve(chunks_.size() + runtime_chunks_.size() + 1);", "        stranded_.reserve(64);",
     "stranded reserve not sized by the chunk count"),
    ("c11-reserve-after-owner", 11, "grow_into",
     ("        stranded_.reserve(chunks_.size() + runtime_chunks_.size() + 1);\n    } catch",
      "return allocate_pinned_chunk_owner(queue_, backing_size, runtime_pool, cache_backing_token{});\n        });\n"),
     ("        (void) 0;\n    } catch",
      "return allocate_pinned_chunk_owner(queue_, backing_size, runtime_pool, cache_backing_token{});\n        });\n"
      "        stranded_.reserve(chunks_.size() + runtime_chunks_.size() + 1);\n"),
     "stranded reserve after the owner allocation"),
    ("c11-capacity-assertion-missing", 11, "carrier_dtor",
     "            if (pool_->stranded_.size() >= pool_->stranded_.capacity()) {",
     "            if (false) {", "capacity assertion missing or after the append"),
    ("c11-destructor-reserves", 11, "carrier_dtor",
     "            pool_->stranded_.emplace_back(stranded_entry{ metadata, backing_ });",
     "            pool_->stranded_.reserve(pool_->stranded_.size() + 1);\n"
     "            pool_->stranded_.emplace_back(stranded_entry{ metadata, backing_ });",
     "released_chunk destructor allocates"),
    ("c11-reserve-uncaught", 11, "grow_into", "    } catch (const std::bad_alloc &) {",
     "    } catch (const sycl::exception &) {", "stranded reserve without bad_alloc catch"),
    ("c11-reconcile-rebuilds", 11, "reconcile", "    stranded_.erase(gone, stranded_.end());",
     "    std::vector<stranded_entry> kept(stranded_.begin(), gone);\n    stranded_.swap(kept);",
     "reconcile rebuilds stranded_"),
    ("c12-free-before-assertion", 12, "pool_dtor", "    GGML_ASSERT(owning_carriers_.load(",
     "    free_chunk_owner(chunks_.front(), \"pinned-base\");\n    GGML_ASSERT(owning_carriers_.load(",
     "carrier assertion after the free loop"),
    ("c12-assertion-moved-after-loop", 12, "pool_dtor",
     ("    GGML_ASSERT(owning_carriers_.load(std::memory_order_acquire) == 0 &&\n"
      "                \"released_chunk outstanding at pool destruction\");\n", "    chunks_.clear();"),
     ("", "    GGML_ASSERT(owning_carriers_.load(std::memory_order_acquire) == 0 &&\n"
          "                \"released_chunk outstanding at pool destruction\");\n    chunks_.clear();"),
     "carrier assertion after the free loop"),
    ("c12-assertion-absent", 12, "pool_dtor",
     "    GGML_ASSERT(owning_carriers_.load(std::memory_order_acquire) == 0 &&\n"
     "                \"released_chunk outstanding at pool destruction\");\n", "", "carrier assertion absent"),
    ("c12-count-lowered-early", 12, "carrier_dtor",
     "    pool_->owning_carriers_.fetch_sub(1, std::memory_order_acq_rel);\n}", "}",
     "carrier count lowered before the footprint section"),
    ("canon-row-missing", 4, None, "`unified_cache::marked_set_mutex_`", "`unified_cache::marked_set_lock_`",
     "canon §12.5 does not rank the lock"),
    ("census-fifth-caller", 10, "configure", "    host_zone_reserved_bytes_[static_cast<size_t>(host_zone_id::WEIGHT)]",
     "    (void) pinned_pool_committed();\n    host_zone_reserved_bytes_[static_cast<size_t>(host_zone_id::WEIGHT)]",
     "pinned_pool_committed caller count differs"),
]

# The one negative control (§7 C4 clause 10): a comment quoting the member must not count as a mention.
NEGATIVE_CONTROL = ("comment-quotes-member", "configure_zones", "    // Grow chunks until we have enough total capacity.",
                    "    // Grow chunks until we have enough total capacity: total_allocated_ + chunk_size_ <= budget_\n"
                    "    /* total_allocated_ -= 0; total_allocated_ = 0; */")


def apply_edit(texts: dict, key, old, new) -> dict:
    """Apply one replacement, or a tuple of them in order (a move is a removal plus a re-plant)."""
    if isinstance(old, tuple):
        assert isinstance(new, tuple) and len(new) == len(old), "a multi-edit control pairs each old with a new"
        for o, n in zip(old, new):
            texts = apply_edit(texts, key, o, n)
        return texts
    out = dict(texts)
    if key is None:
        rel = CANON
        raw = texts[rel]
        assert raw.count(old) == 1, f"control anchor occurs {raw.count(old)} times in {rel}: {old!r}"
        out[rel] = raw.replace(old, new, 1)
        return out
    rel = BODY_SPECS[key][0]
    raw = texts[rel]
    t = Tree(texts)
    f = t.bodies.get(key)
    assert f is not None, f"control body {key} not found"
    lo, hi = f["open"], f["close"] + 1
    region = raw[lo:hi]
    assert region.count(old) == 1, f"control anchor occurs {region.count(old)} times in {key}: {old!r}"
    out[rel] = raw[:lo] + region.replace(old, new, 1) + raw[hi:]
    return out


TREE = tree_texts()


def test_tree_passes():
    fails = failures(TREE)
    assert fails == [], "\n".join(f"{r}: {d}" for r, d in fails)


@pytest.mark.parametrize("control", CONTROLS, ids=[c[0] for c in CONTROLS])
def test_planted_control_fails_with_its_reason(control):
    _cid, _clause, key, old, new, reason = control
    got = reasons(apply_edit(TREE, key, old, new))
    assert reason in got, f"expected {reason!r}, got {sorted(got)}"


def test_negative_control_comment_mention_passes():
    _cid, key, old, new = NEGATIVE_CONTROL
    fails = failures(apply_edit(TREE, key, old, new))
    assert fails == [], "comment mention flagged: " + "; ".join(f"{r}: {d}" for r, d in fails)


def test_every_clause_has_a_planted_control():
    assert sorted({c[1] for c in CONTROLS}) == list(range(1, 13))


def test_canon_census_reads_the_table():
    names = canon_census_names(TREE[CANON])
    for want in ("rw_mutex_", "g_runtime_alloc_mutex", "marked_set_mutex_", "deferred_frees_mutex_",
                 "managed_allocs_mutex_"):
        assert want in names, f"{want} missing from the canon census {sorted(names)}"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
