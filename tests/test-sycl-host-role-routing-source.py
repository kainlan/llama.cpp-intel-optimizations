#!/usr/bin/env python3
"""Host role routing gate (docs/design/sycl-mmap-host-tier.md §7 C4, `test-sycl-host-role-routing-source`; §4.4).

Every host request of category HOST_COMPUTE or EXPERT_CACHE reaches a host zone by its role and lifetime through
select_zone, and §4.4 publishes each site's holder, role, zone, lifetime and owner. This gate regenerates that table
from the tree and fails on any difference from SITE_TABLE, then cross-checks §4.4's published rows against it, every
difference listed row by row in DOC_DRIFT. The other clauses pin the rest of the bullet: select_zone's rules, the
owners and the UNOWNED and UNSIZED ratchets, the DEVICE holders' lifetime, release and check, the attention staging's
refusal and plan-only growth, the plan walk's predicates and devices, the pools' one dimension rule and plan-only
sizing, the chunk-release reachability and withdrawal of §M75, the notify's device and position, that nothing in
src/ calls the compute-buffer notify, the notify's and plan entry's dlopen shape, llama_context's wiring (begin, plan
and commit positions) and trial operations, the owner's end, and the host SCRATCH fold and runtime term. The
segmented forward, the host bufts' one intent builder and the KV clauses have their own gates
(test-sycl-seg-forward-return-source, test-sycl-host-zone-config-source-contract,
test-sycl-host-zone-boundary-owner-source and the test-sycl-kv-* gates).
Three clauses of the bullet are pinned by behaviour, not by this source read:
  - the notify's zone is the host buft's device, and it tests ggml_backend_is_sycl before it reads
    backend->context: test-sycl-host-scratch-demand (test_device_is_the_host_bufts with its device-0 and
    ctx->device mutants, and test_non_sycl_refused);
  - the SCRATCH graph-boundary check counts only GRAPH records: test-sycl-host-zone-select;
  - ggml_backend_sycl_free runs the BACKEND check after `delete sycl_ctx;` (§M57 (b)): the O4 row of
    test-sycl-host-zone-boundary-owner-source, with its check-before-delete mutant.

Each clause is REQUIRED (green on this tree) or EXPECTED-RED with the task that turns it green. An expected-red
clause is scored: it must fail on this tree with its recorded reason (an ExpectedRed warning, never a skip), it
must pass on a planted fixture of the landed shape, it must fail with a distinct reason, never its recorded one,
once what it inspects is removed, and its test fails the moment the tree satisfies it, so the landing commit moves
it to REQUIRED. Every clause has at least one planted mutant that must fail with its quoted reason, and a negative
control (comments appended to the edited files) must leave every REQUIRED clause green.

The tree is read with its compile-time positive controls resolved (each `#if defined(..._MUTANT_...)` block as the
production build compiles it) and comment-stripped. Plain file I/O, not the codescout index, which is blind in
ggml-sycl.cpp.
"""

import re
import sys
import warnings
from pathlib import Path

import pytest

# sycl_gate sits beside this file. `python3 -I` leaves the script's directory off sys.path, so it is appended
# here; appended rather than prepended, a file in tests/ never shadows a standard module.
sys.path.append(str(Path(__file__).resolve().parent))
from sycl_gate import match_close, resolve_mutants, views  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SYCL_DIR = "ggml/src/ggml-sycl"
GS = SYCL_DIR + "/ggml-sycl.cpp"
UC = SYCL_DIR + "/unified-cache.cpp"
UCH = SYCL_DIR + "/unified-cache.hpp"
POOL = SYCL_DIR + "/pinned-pool.cpp"
POOL_H = SYCL_DIR + "/pinned-pool.hpp"
PLAN = SYCL_DIR + "/host-zone-plan.cpp"
HEADER = "ggml/include/ggml-sycl.h"
LC = "src/llama-context.cpp"
LCH = "src/llama-context.h"
SRC_DIR = "src"
NOTIFY = "ggml_backend_sycl_notify_host_buffer_sizes"
PLAN_ENTRY = "ggml_backend_sycl_plan_host_holders"
DOC = "docs/design/sycl-mmap-host-tier.md"


class ExpectedRed(UserWarning):
    """An expected-red clause failing as recorded, until the task named in it lands."""


# ---------------------------------------------------------------------------------------------------------------
# Text views. The source is read with its planted-fault blocks resolved by sycl_gate.resolve_mutants (every `#if` over a
# MUTANT macro evaluated with no mutant macro defined, as the production build compiles it), then with comments blanked
# (`code`) and with string and character literals blanked too (`bare`), by sycl_gate's views. Every view keeps the raw
# text's offsets.
# ---------------------------------------------------------------------------------------------------------------

_KEYWORDS = r"(?:if|for|while|switch|catch|return|else|do|sizeof|case|new|delete|throw|static_assert|alignof|decltype)"
# A definition head: a line whose text before its first `(` holds none of `;{}()#=`, ending in the (qualified) name.
HEAD_RE = re.compile(r"^[ \t]*(?!" + _KEYWORDS + r"\b)([^\n;{}()#=]*)\(", re.M)
_NAME_TAIL_RE = re.compile(r"((?:[A-Za-z_]\w*::)*~?[A-Za-z_]\w*)\s*$")
_STOP_RE = re.compile(r"[();={]")


class Source:
    """One file: the mutant-resolved text, its views, and its function definitions (column 0 or indented members)."""

    def __init__(self, raw: str):
        self.raw = resolve_mutants(raw)
        self.code, self.bare = views(self.raw)
        self._functions = None

    @property
    def functions(self):
        if self._functions is None:
            out = []
            bare = self.bare
            for m in HEAD_RE.finditer(bare):
                tail = _NAME_TAIL_RE.search(m.group(1))
                if not tail or re.fullmatch(_KEYWORDS, tail.group(1)):
                    continue
                name = tail.group(1)
                lp = m.end() - 1
                rp = match_close(bare, lp)
                if rp < 0:
                    continue
                depth, brace = 0, -1
                for stop in _STOP_RE.finditer(bare, rp + 1):
                    c = stop.group(0)
                    if c == "(":
                        depth += 1
                    elif c == ")":
                        depth -= 1
                        if depth < 0:
                            break
                    elif depth == 0 and c in ";=":
                        break
                    elif depth == 0 and c == "{":
                        brace = stop.start()
                        break
                if brace < 0:
                    continue
                close = match_close(bare, brace)
                if close < 0:
                    continue
                out.append({"name": name, "short": name.split("::")[-1], "head": m.start(),
                            "name_at": m.start(1) + tail.start(1), "params": bare[lp + 1:rp], "open": brace,
                            "close": close})
            self._functions = out
        return self._functions

    def enclosing(self, pos: int):
        best = None
        for f in self.functions:
            if f["open"] < pos < f["close"] and (best is None or f["open"] > best["open"]):
                best = f
        return best

    def find(self, name: str, params=None):
        """The definitions named `name` (qualified or its last component), optionally filtered by a params regex."""
        return [f for f in self.functions
                if (f["name"] == name or f["short"] == name) and (params is None or re.search(params, f["params"]))]

    def body(self, name: str, view: str = "bare", params=None) -> str:
        hits = self.find(name, params)
        if len(hits) != 1:
            return ""
        f = hits[0]
        return getattr(self, view)[f["open"]:f["close"] + 1]

    def line(self, pos: int) -> int:
        return self.raw.count("\n", 0, pos) + 1


def call_args(bare: str, lp: int):
    """The top-level argument spans of the call or initializer whose bracket opens at `lp`."""
    rp = match_close(bare, lp)
    args, depth, start = [], 0, lp + 1
    for i in range(lp + 1, rp):
        c = bare[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            args.append((start, i))
            start = i + 1
    if bare[start:rp].strip():
        args.append((start, rp))
    return args


# ---------------------------------------------------------------------------------------------------------------
# Tree access. `texts` maps a repo-relative path to its raw text; a control edits one entry. Parsed sources are
# reused while a file's text is unchanged.
# ---------------------------------------------------------------------------------------------------------------

_SOURCES = {}  # rel -> [(raw, Source)]: the first text seen (the tree's) is kept, plus the latest control's


def source_for(rel: str, raw: str) -> Source:
    entries = _SOURCES.setdefault(rel, [])
    for text, parsed in entries:
        if text == raw:
            return parsed
    parsed = Source(raw)
    entries[1:] = [(raw, parsed)]
    return parsed


def tree_texts() -> dict:
    """Each file's text with its planted-fault blocks resolved (sycl_gate.resolve_mutants), so a planted control edits
    production code alone: a fault block may repeat a production statement it varies."""
    texts = {}
    for p in sorted((ROOT / SYCL_DIR).glob("*")):
        if p.is_file() and p.suffix in (".cpp", ".hpp"):
            texts[f"{SYCL_DIR}/{p.name}"] = resolve_mutants(p.read_text(errors="replace"))
    for p in sorted((ROOT / SRC_DIR).rglob("*")):
        if p.is_file() and p.suffix in (".cpp", ".h"):
            texts[p.relative_to(ROOT).as_posix()] = resolve_mutants(p.read_text(errors="replace"))
    for rel in (HEADER, LC, LCH, DOC):
        p = ROOT / rel
        texts[rel] = p.read_text(errors="replace") if p.exists() else ""
    return texts


# ---------------------------------------------------------------------------------------------------------------
# The site extractor. A site is a mention of runtime_category::HOST_COMPUTE or ::EXPERT_CACHE in production SYCL
# source (ggml/src/ggml-sycl/*.cpp and *.hpp; tests/ excluded). A mention after `case`, beside `==`/`!=`, after
# `return`, or assigned into an alloc_intent the function receives by reference, is a mapping, not a request. A
# request is read in one of two forms:
#   assign  `VAR.category = ...;`: every assignment to VAR's role, lifetime, cohort_id and owner, and to its request's
#           tier constraints, between the request's declaration (or the midpoint to the previous `VAR.category =`)
#           and the next declaration (or midpoint), within 2500 characters;
#   group   the category is an argument of a call or of an alloc_intent brace initializer: the role and lifetime
#           tokens in that group, and the cohort and tier at the argument positions GROUP_CALLEES records for the
#           callee. An unknown callee reads "?", so a new form fails the table until it is classified here.
# A key is (file, function, ordinal of the mention in that function), so the table is stable across line drift.
# ---------------------------------------------------------------------------------------------------------------

SITE_CAT_RE = re.compile(r"\bruntime_category::(HOST_COMPUTE|EXPERT_CACHE)\b")
SITE_WINDOW = 2500

# callee -> (cohort argument index or None, tier: "host" / "device" / ("arg", i) for a must_host_pinned bool argument
# / ("constraints", i) for an alloc_constraints variable). Each tier claim is checked against the callee's body by
# clause_site_table.
GROUP_CALLEES = {
    "ggml_sycl_tp_alloc_tmp": (None, ("arg", 5)),
    "ensure": (5, "host"),
    "allocate_managed_host_pinned": (5, "host"),
    "unified_cache_adopt_raw_device_allocation": (5, "device"),
    "<intent>": (2, ("constraints", 3)),
}


def _clean(expr: str) -> str:
    return re.sub(r"\s+", " ", expr.replace("ggml_sycl::", "")).strip()


def _enum_values(expr: str, enum: str) -> str:
    """`alloc_role::KV` -> "KV"; a ternary over two values -> "KV/WEIGHT"; anything else -> the expression."""
    vals = re.findall(enum + r"::(\w+)", expr)
    return "/".join(vals) if vals else _clean(expr)


def _mapping_kind(bare: str, pos: int):
    ls = bare.rfind("\n", 0, pos) + 1
    before = bare[ls:pos]
    if re.search(r"\bcase\s+$", before):
        return "case"
    if re.search(r"(==|!=)\s*$", before) or re.match(r"\w+::\w+\s*(==|!=)", bare[pos:pos + 80]):
        return "compare"
    if re.search(r"\breturn\s+$", before):
        return "return"
    return None


def _enclosing_group(bare: str, pos: int):
    depth = {")": 0, "}": 0}
    i = pos
    while i > 0:
        i -= 1
        c = bare[i]
        if c in ")}":
            depth[c] += 1
        elif c == "(":
            if depth[")"] == 0:
                return i
            depth[")"] -= 1
        elif c == "{":
            if depth["}"] == 0:
                return i
            depth["}"] -= 1
        elif c == ";" and depth[")"] == 0 and depth["}"] == 0:
            return None
    return None


def _var_re(var: str) -> str:
    return r"(?<![\w.>])" + r"\s*\.\s*".join(re.escape(p) for p in var.split("."))


def _group_callee(bare: str, lp: int) -> str:
    before = bare[max(0, lp - 80):lp]
    if bare[lp] == "{":
        return "<intent>" if re.search(r"\.\s*intent\s*=\s*$", before) else "?"
    m = re.search(r"([A-Za-z_]\w*)\s*$", before)
    return m.group(1) if m else "?"


def extract_sites(texts: dict) -> list:
    sites = []
    for rel in sorted(texts):
        if not (rel.startswith(SYCL_DIR + "/") and rel.count("/") == SYCL_DIR.count("/") + 1):
            continue
        if not rel.endswith((".cpp", ".hpp")) or not SITE_CAT_RE.search(texts[rel]):
            continue
        s = source_for(rel, texts[rel])
        counter = {}
        pending = []
        for m in SITE_CAT_RE.finditer(s.code):
            pos = m.start()
            if s.bare[pos] == " ":  # inside a string literal
                continue
            f = s.enclosing(pos)
            fname = f["name"] if f else "<file scope>"
            counter[fname] = counter.get(fname, 0) + 1
            site = {"file": rel, "func": fname, "ordinal": counter[fname], "line": s.line(pos),
                    "category": m.group(1), "pos": pos, "fn": f}
            kind = _mapping_kind(s.bare, pos)
            stmt = max(s.bare.rfind(";", 0, pos), s.bare.rfind("{", 0, pos), s.bare.rfind("}", 0, pos)) + 1
            am = re.search(r"([A-Za-z_]\w*(?:\s*\.\s*[A-Za-z_]\w*)*)\s*\.\s*category\s*=(?!=)", s.bare[stmt:pos])
            if not kind and am:
                var = re.sub(r"\s+", "", am.group(1))
                if "." not in var and f and re.search(r"\balloc_intent\s*&\s*" + re.escape(var) + r"\b", f["params"]):
                    kind = "assign"
            if kind:
                site.update(form="mapping:" + kind, role="-", lifetime="-", cohort="-", owner="-", tier="-")
                sites.append(site)
                continue
            if am:
                site.update(form="assign", var=re.sub(r"\s+", "", am.group(1)), own=stmt + am.start(1))
            else:
                lp = _enclosing_group(s.bare, pos)
                site.update(form="group" if lp is not None else "unknown", lp=lp)
            pending.append(site)
        for site in pending:
            if site["form"] == "assign":
                _read_assign(s, site)
            elif site["form"] == "group":
                _read_group(s, site)
            else:
                site.update(role="-", lifetime="-", cohort="-", owner="-", tier="-")
            sites.append(site)
    for site in sites:
        site.pop("fn", None)
        site.pop("lp", None)
        site.pop("var", None)
        site.pop("own", None)
    return sorted(sites, key=lambda x: (x["file"], x["pos"]))


def _read_assign(s: Source, site: dict):
    pos, fn, var = site["pos"], site["fn"], site["var"]
    f_lo, f_hi = (fn["open"], fn["close"]) if fn else (0, len(s.bare))
    req = var[:-len(".intent")] if var.endswith(".intent") else var
    body = s.bare[f_lo:f_hi]
    # The request's own span: from its declaration to the next declaration of the same name, and between the
    # neighbouring `VAR.category =` assignments of any category.
    decl_re = r"\b(?:alloc_request|offload_buffer_request|alloc_intent)\s+" + re.escape(req.split(".")[0]) + r"\b"
    decls = [f_lo + m.start() for m in re.finditer(decl_re, body)]
    peers = [f_lo + m.start() for m in re.finditer(_var_re(var) + r"\s*\.\s*category\s*=(?!=)", body)]
    lo = max([d for d in decls if d < pos] + [f_lo, pos - SITE_WINDOW])
    hi = min([d for d in decls if d > pos] + [f_hi, pos + SITE_WINDOW])
    before = [p for p in peers if p < site["own"]]
    after = [p for p in peers if p > site["own"]]
    if before:
        lo = max(lo, (before[-1] + pos) // 2)
    if after:
        hi = min(hi, (pos + after[0]) // 2)

    def values(text, field_re, enum=None):
        out = []
        for m in re.finditer(_var_re(var) + r"\s*\.\s*" + field_re + r"\s*=(?!=)\s*([^;]+);", text[lo:hi]):
            out.append(_enum_values(m.group(1), enum) if enum else _clean(m.group(1)))
        return out

    roles = values(s.bare, "role", "alloc_role")
    lifetimes = values(s.bare, "lifetime", "alloc_lifetime")
    cohorts = values(s.code, "cohort_id")
    owners = [_enum_values(v, "alloc_owner_kind") for v in values(s.bare, "owner")]
    tier = []
    for m in re.finditer(_var_re(req) + r"\s*\.\s*(?:intent\s*\.\s*)?constraints\s*\.\s*(must_device|must_host_pinned)"
                         r"\s*=(?!=)\s*(\w+)\s*;", s.bare[lo:hi]):
        if m.group(2) == "true":
            tier.append("device" if m.group(1) == "must_device" else "host")
    site.update(role="|".join(roles) or "-", lifetime="|".join(lifetimes) or "-", cohort="|".join(cohorts) or "-",
                owner="|".join(owners) or "-", tier="|".join(sorted(set(tier))) or "-")


def _read_group(s: Source, site: dict):
    lp = site["lp"]
    rp = match_close(s.bare, lp)
    grp = s.bare[lp:rp]
    callee = _group_callee(s.bare, lp)
    args = call_args(s.bare, lp)
    arg = [_clean(s.code[a:b]) for a, b in args]
    cohort_at, tier_rule = GROUP_CALLEES.get(callee, (None, "?"))
    cohort = arg[cohort_at] if cohort_at is not None and cohort_at < len(arg) else "-"
    if isinstance(tier_rule, tuple) and tier_rule[0] == "arg":
        tier = "host" if tier_rule[1] < len(arg) and arg[tier_rule[1]] == "true" else "-"
    elif isinstance(tier_rule, tuple):
        name = arg[tier_rule[1]] if tier_rule[1] < len(arg) else ""
        fn = site["fn"]
        lo, hi = (fn["open"], site["pos"]) if fn else (0, site["pos"])
        tier = "|".join(sorted({"device" if m.group(1) == "must_device" else "host" for m in re.finditer(
            _var_re(name) + r"\s*\.\s*(must_device|must_host_pinned)\s*=\s*true\s*;", s.bare[lo:hi])})) or "-"
    else:
        tier = tier_rule
    site.update(role="|".join(re.findall(r"alloc_role::(\w+)", grp)) or "-",
                lifetime="|".join(re.findall(r"alloc_lifetime::(\w+)", grp)) or "-", cohort=cohort, owner="-",
                tier=tier, form="group:" + callee)


# ---------------------------------------------------------------------------------------------------------------
# select_zone, read from the tree. Its if-chain is parsed into rules, so a clause restored there moves the rows it
# matches and the table differs (the "category clause restored" mutant). An atom the reader does not know fails.
# ---------------------------------------------------------------------------------------------------------------

_ZONE_ATOMS = {
    "kv_spill_to_host": lambda r: r["kv_spill"],
    "persistent_owner": lambda r: r["persistent"],
}


def select_zone_rules(texts: dict):
    """[(condition atoms as a list of conjunction lists, zone)] and the final zone.

    Raises on a construct it cannot read."""
    s = source_for(PLAN, texts[PLAN])
    body = s.body("select_zone")
    assert body, "select_zone not found in host-zone-plan.cpp"
    rules = []
    for m in re.finditer(r"\bif\s*\(", body):
        lp = m.end() - 1
        rp = match_close(body, lp)
        cond = body[lp + 1:rp]
        ret = re.match(r"\s*\{\s*return\s+host_zone_id::(\w+)\s*;\s*\}", body[rp + 1:])
        assert ret, f"select_zone: an if without a lone return: {cond.strip()!r}"
        disj = []
        for alt in cond.split("||"):
            conj = []
            for atom in alt.split("&&"):
                atom = re.sub(r"\s+", " ", atom.strip().strip("()").strip())
                assert (atom in _ZONE_ATOMS or re.fullmatch(r"(role|lifetime|category) == \w+::\w+", atom)), \
                    f"select_zone condition not understood: {atom!r}"
                conj.append(atom)
            disj.append(conj)
        rules.append((disj, ret.group(1)))
    final = re.findall(r"return\s+host_zone_id::(\w+)\s*;\s*\}\s*$", body)
    assert final, "select_zone has no final return"
    return rules, final[0]


def zone_for(rules, final, role: str, lifetime: str, category: str, persistent=False, kv_spill=False) -> str:
    row = {"role": role, "lifetime": lifetime, "category": category, "persistent": persistent, "kv_spill": kv_spill}

    def holds(atom):
        if atom in _ZONE_ATOMS:
            return _ZONE_ATOMS[atom](row)
        field, value = re.fullmatch(r"(role|lifetime|category) == \w+::(\w+)", atom).groups()
        return row[field] == value

    for disj, zone in rules:
        if any(all(holds(a) for a in conj) for conj in disj):
            return zone
    return final


def site_zones(rules, final, site: dict) -> str:
    """The zone of each (role, lifetime) pair the site stamps; a ternary's alternatives are joined with '/'."""
    if site["form"].startswith("mapping") or site["tier"] == "device":
        return "-"
    roles = site["role"].split("|") if site["role"] != "-" else ["OTHER"]
    lifes = site["lifetime"].split("|") if site["lifetime"] != "-" else ["GRAPH"]
    if len(lifes) == 1:
        lifes = lifes * len(roles)
    if len(lifes) != len(roles):
        return "?"
    out = []
    for role, life in zip(roles, lifes):
        out.append("/".join(zone_for(rules, final, r, life, site["category"]) for r in role.split("/")))
    return "|".join(out)


# ---------------------------------------------------------------------------------------------------------------
# The site table (§4.4 "The site table"), regenerated from this tree. One row per HOST_COMPUTE / EXPERT_CACHE mention:
#   (file, function, ordinal, form, category, role, lifetime, cohort, owner, tier, zone, holder, class)
# role/lifetime/cohort/owner/zone hold one value per variant, `|`-joined, and a ternary's alternatives `/`-joined. "-"
# is unset: lifetime "-" is the default GRAPH. zone is select_zone's answer, read from host-zone-plan.cpp, and "-" for
# a mapping or a device-tier row, which never reaches the host branch. holder is the token that holds the handle,
# present in the site's function, or "token@function" when the site returns the handle to the caller that holds it.
# The class is the table's own grouping:
#   GRAPH       released before the graph boundary          CONTEXT / MODEL  checked at that owner's boundary
#   BUILDER     the host bufts' one intent builder          DEVICE           shared by every backend on the device
#   UNOWNED     the interim value llama.cpp-30r1 deletes     CALLERS          takes its caller's role and lifetime
#   EXPERT      role EXPERT_STAGING, host                   WEIGHT           role WEIGHT on the host tier
#   DEVICE_TIER never reaches the host branch               NON_TENANT       the pool's own chunk backing
#   MAPPING     a category mapping, not a request
# ---------------------------------------------------------------------------------------------------------------

_C, _CD, _GS = "common.cpp", "cpu-dispatch.cpp", "ggml-sycl.cpp"
_UC, _HZ = "unified-cache.cpp", "host-zone-plan.cpp"
_HC, _EC = "HOST_COMPUTE", "EXPERT_CACHE"
_ENS, _AMHP = "group:ensure", "group:allocate_managed_host_pinned"
_ADOPT = "group:unified_cache_adopt_raw_device_allocation"
_ES = "EXPERT_STAGING"

SITE_TABLE = [
    (_C, "ggml_sycl_get_staged_handle_device", 1, "assign", _HC, "STAGING", "UNOWNED", '"runtime_staging_cache"', "-",
     "host", "WEIGHT", "g_runtime_staging_cache", "UNOWNED"),
    (_C, "ggml_sycl_get_staged_handle_device", 2, "assign", _HC, "STAGING", "-", "-", "-", "host", "SCRATCH",
     "host_handle", "GRAPH"),
    (_C, "ggml_sycl_tp_cache_ffn_norm", 1, "assign", _HC, "TP_TMP", "-", "-", "-", "host", "SCRATCH", "host_handle",
     "GRAPH"),
    (_C, "ggml_sycl_tp_ensure_host_staging", 1, "assign", _HC, "TP_TMP", "UNOWNED", '"tp_host_staging"', "-", "host",
     "WEIGHT", "g_tp_host_staging", "UNOWNED"),
    (_C, "ggml_sycl_tp_init_quant_comm_buffers_locked", 1, "assign", _HC, "TP_TMP", "UNOWNED", '"tp_quant_comm_bufs"',
     "-", "host", "WEIGHT", "g_tp_quant_comm_bufs", "UNOWNED"),
    (_C, "ggml_sycl_tp_ensure_shared_reduce_buffer", 1, "assign", _HC, "TP_TMP", "UNOWNED", '"tp_shared_reduce"', "-",
     "host", "WEIGHT", "g_tp_shared_reduce_handle", "UNOWNED"),
    (_C, "ggml_sycl_tp_get_host_reduce_buffers", 1, "assign", _HC, "TP_TMP", "UNOWNED", '"tp_host_bufs"', "-", "host",
     "WEIGHT", "g_tp_host_buf0_handle", "UNOWNED"),
    (_CD, "ggml_sycl_cpu_dispatch_register_host_ptr", 1, "assign", _HC, "WEIGHT", "-",
     '"cpu-dispatch:host-weight-copy"', "-", "host", "WEIGHT", "-", "WEIGHT"),
    (_CD, "ggml_sycl_cpu_retained_init", 1, "assign", _HC, "COMPUTE", "UNOWNED", '"cpu_offload"', "-", "host", "WEIGHT",
     "g_retained_scratch_lease", "UNOWNED"),
    (_GS, "moe_hybrid_init_once", 1, "assign", _HC, "STAGING", "-", "-", "-", "host", "SCRATCH", "d2h_staging",
     "GRAPH"),
    (_GS, "moe_hybrid_init_once", 2, "assign", _HC, "STAGING", "-", "-", "-", "host", "SCRATCH", "d2h_staging",
     "GRAPH"),
    (_GS, "store_routing_indices", 1, "assign", _HC, "STAGING", "UNOWNED", '"routing_indices_cache"', "-", "host",
     "WEIGHT", "g_routing_indices_cache", "UNOWNED"),
    (_GS, "ggml_sycl_reserve_host_tenants", 1, "assign", _HC, "STAGING", "CONTEXT", "info->name", "EXECUTION_CONTEXT",
     "host", "SCRATCH", "kv_tenant_slots", "CONTEXT"),
    (_GS, "arena_pinned_alloc", 1, "assign", _HC, "COMPUTE", "UNOWNED", '"moe_fusion_fused_buf"', "-", "host", "WEIGHT",
     "fused_handle@ensure_fused_buf", "UNOWNED"),
    (_GS, "ggml_sycl_fill_onednn_woq", 1, "assign", _HC, _ES, "-", "-", "-", "host", "SCRATCH", "-", "EXPERT"),
    (_GS, "ggml_sycl_fill_xmx_tiled", 1, "assign", _HC, _ES, "-", "-", "-", "host", "SCRATCH", "-", "EXPERT"),
    (_GS, "ggml_backend_sycl_buffer_get_tensor", 1, "assign", _HC, "STAGING", "-", "-", "-", "host", "SCRATCH",
     "fallback_handle", "GRAPH"),
    (_GS, "ggml_sycl_readback_via_shared_kernel", 1, "assign", _HC, "STAGING", "-", "-", "-", "host", "SCRATCH",
     "host_stage_handle", "GRAPH"),
    (_GS, "ggml_backend_sycl_split_buffer_init_tensor", 1, "assign", _HC, "WEIGHT", "-", "-", "-", "host", "WEIGHT",
     "-", "WEIGHT"),
    (_GS, "ggml_sycl_host_buft_intent", 1, "assign", _HC, "WEIGHT|COMPUTE|KV|COMPUTE", "MODEL|CONTEXT|CONTEXT|CONTEXT",
     '"backend_host_buffer"|"kv_host_buffer"', "-", "host", "WEIGHT|SCRATCH|KV|SCRATCH", "ggml_sycl_host_buft_kind",
     "BUILDER"),
    (_GS, "ggml_backend_sycl_context::ensure_readback_staging", 1, "assign", _HC, "STAGING", "CONTEXT",
     '"context_readback_staging"', "BACKEND", "host", "SCRATCH", "readback_staging_handle", "CONTEXT"),
    (_GS, "ggml_sycl_cpu_fallback_graph", 1, "assign", _HC, "GRAPH_TMP", "-", "-", "-", "host", "SCRATCH",
     "host_copies", "GRAPH"),
    (_GS, "tp_device1_worker_thread_func", 1, "group:ggml_sycl_tp_alloc_tmp", _HC, "TP_TMP", "-", "-", "-", "host",
     "SCRATCH", "w.results", "GRAPH"),
    (_GS, "ggml_sycl_tp_launch_async_ffn", 1, "group:ggml_sycl_tp_alloc_tmp", _HC, "TP_TMP", "-", "-", "-", "host",
     "SCRATCH", "g_tp_async_ffn_jobs", "GRAPH"),
    (_GS, "convert_tensor_layout", 1, "assign", _HC, _ES, "-", "-", "-", "host", "SCRATCH", "-", "EXPERT"),
    (_GS, "split_staging_ensure", 1, _ENS, _HC, "STAGING", "UNOWNED", '"tensor_split_src1_host"', "-", "host", "WEIGHT",
     "g_split_staging", "UNOWNED"),
    (_GS, "split_staging_ensure", 2, _ENS, _HC, "STAGING", "UNOWNED", '"tensor_split_output_host"', "-", "host",
     "WEIGHT", "g_split_staging", "UNOWNED"),
    (_GS, "split_weight_staging_ensure", 1, _ENS, _HC, "STAGING", "UNOWNED", '"tensor_split_weight_stage"', "-", "host",
     "WEIGHT", "g_split_weight_staging", "UNOWNED"),
    (_GS, "split_get_cached_weights", 1, "assign", _HC, "WEIGHT", "-", '"tensor_split_weight_cache"', "-", "host",
     "WEIGHT", "-", "WEIGHT"),
    (_GS, "ggml_sycl_mul_mat_tensor_split", 1, _ENS, _HC, "STAGING", "UNOWNED", '"tensor_split_cpu_out_ring"', "-",
     "host", "WEIGHT", "s_cpu_out_ring_alloc", "UNOWNED"),
    (_GS, "ensure_host", 1, _ENS, _HC, _ES, "-", "name", "-", "host", "SCRATCH", "-", "EXPERT"),
] + [
    (_GS, "dispatch_experts_secondary_gpu_impl", i + 1, _ENS, _HC, _ES, "-", cohort, "-", "host", "SCRATCH", "-",
     "EXPERT")
    for i, cohort in enumerate(['"moe_gpu1_act_staging"', '"moe_gpu1_act_staging_target"', '"moe_gpu1_out_staging"',
                                '"moe_gpu1_out_staging_target"', "name", '"moe_gpu1_pp_act_agg_target"',
                                '"moe_gpu1_pp_out_agg_target"', '"moe_ring_out_reduce"', '"moe_ring_out_reduce_target"',
                                '"moe_ring_out_agg"', '"moe_ring_out_agg_target"'])
] + [
    (_GS, "ggml_sycl_moe_hybrid_decode_table_allocate", 1, "assign", _EC, "COMPUTE", "MODEL",
     '"moe_hybrid_decode_table"', "MODEL", "host", "SCRATCH", "record->table@ggml_sycl_moe_hybrid_decode_record_build",
     "MODEL"),
    (_GS, "ggml_sycl_moe_hybrid_decode_event_chained", 1, _ENS, _HC, _ES, "-", '"moe_shared_activation_host"', "-",
     "host", "SCRATCH", "-", "EXPERT"),
    (_GS, "ggml_sycl_mul_mat_id", 1, _ENS, _HC, _ES, "-", '"moe_shared_activation_host"', "-", "host", "SCRATCH", "-",
     "EXPERT"),
] + [
    (_GS, "ggml_sycl_mul_mat_id", i + 2, _AMHP, _HC, _ES, "-", cohort, "-", "host", "SCRATCH", "-", "EXPERT")
    for i, cohort in enumerate(['"moe_cpu_dispatch_act"', '"moe_cpu_dispatch_out"', '"moe_q1_nvfp4_recipe_workspace"',
                                '"moe_planner_cpu_act"', '"moe_planner_cpu_out"', '"moe_planner_cpu_weight"'])
] + [
    (_GS, "ggml_sycl_mul_mat_id", 8, "assign", _HC, _ES, "-", '"moe_pp_cpu_island_act"', "-", "host", "SCRATCH", "-",
     "EXPERT"),
    (_GS, "ggml_sycl_attn_host_staging_block_alloc", 1, "assign", _HC, "COMPUTE", "DEVICE",
     "host_staging_kind_name(kind)", "DEVICE", "host", "SCRATCH", "ggml_sycl_attn_host_staging_token", "DEVICE"),
    (_GS, "ggml_sycl_attn_host_staging_per_call", 1, "group:<intent>", _HC, "COMPUTE", "-", "tag", "-", "host",
     "SCRATCH", "fallback_holder", "GRAPH"),
    (_GS, "ggml_sycl_dispatch_host_flash_attn_sync", 1, "group:<intent>", _HC, "COMPUTE", "-", '"attn_host_q_verify"',
     "-", "host", "SCRATCH", "vh", "GRAPH"),
    (_HZ, "select_zone", 1, "mapping:compare", _HC, "-", "-", "-", "-", "-", "-", "-", "MAPPING"),
    (_HZ, "select_zone", 2, "mapping:compare", _EC, "-", "-", "-", "-", "-", "-", "-", "MAPPING"),
    ("mem-ops.cpp", "host_endpoint_owner_in_scope", 1, "mapping:compare", _EC, "-", "-", "-", "-", "-", "-", "-",
     "MAPPING"),
    ("moe-stream-ring.cpp", "moe_stream_ring::reserve", 1, "assign", _EC, _ES, "-", '"moe_stream_ring"', "-", "device",
     "-", "-", "DEVICE_TIER"),
    ("pinned-buffer-pool.cpp", "PinnedBufferPool::init", 1, "assign", _EC, "COMPUTE", "DEVICE", '"moe_act_pool"',
     "DEVICE", "host", "SCRATCH", "act_handle_", "DEVICE"),
    ("pinned-pool.cpp", "allocate_pinned_chunk_owner", 1, "assign", _HC, "STAGING", "-",
     'runtime_pool ? "pinned_chunk:runtime" : "pinned_chunk:base"', "-", "host", "SCRATCH", "-", "NON_TENANT"),
    (_UC, "category_from_role", 1, "mapping:return", _EC, "-", "-", "-", "-", "-", "-", "-", "MAPPING"),
    (_UC, "unified_cache::ensure_cached", 1, _ADOPT, _EC, "WEIGHT", "-", '"unified_cache:realloc"', "-", "device", "-",
     "-", "DEVICE_TIER"),
    (_UC, "unified_cache::ensure_cached", 2, _ADOPT, _EC, "WEIGHT", "-", '"unified_cache:alloc"', "-", "device", "-",
     "-", "DEVICE_TIER"),
    (_UC, "unified_cache::direct_stage_weight", 1, "assign", _EC, "WEIGHT", "-", '"direct_stage_weight"', "-", "device",
     "-", "-", "DEVICE_TIER"),
    (_UC, "unified_cache::direct_stage_expert", 1, "assign", _EC, "WEIGHT", "-", '"direct_stage_expert"', "-", "device",
     "-", "-", "DEVICE_TIER"),
    (_UC, "unified_cache::direct_stage_expert", 2, "assign", _EC, "WEIGHT", "-", '"direct_stage_expert_host"', "-",
     "host", "WEIGHT", "-", "WEIGHT"),
    (_UC, "unified_cache::allocate_slot", 1, "assign", _EC, "WEIGHT", "-", '"unified_cache:slot"', "-", "device", "-",
     "-", "DEVICE_TIER"),
    (_UC, "acquire_offload_buffer", 1, "assign", _HC, "STAGING", "-", "-", "-", "host", "SCRATCH",
     "offload_buffer_lease", "CALLERS"),
    (_UC, "category_to_intent", 1, "mapping:assign", _EC, "-", "-", "-", "-", "-", "-", "-", "MAPPING"),
    (_UC, "unified_cache_get_runtime_bytes_by_category", 1, "mapping:case", _HC, "-", "-", "-", "-", "-", "-", "-",
     "MAPPING"),
    (_UC, "unified_cache_get_runtime_bytes_by_category", 2, "mapping:case", _EC, "-", "-", "-", "-", "-", "-", "-",
     "MAPPING"),
    (_UC, "unified_cache::load_partial_rows", 1, "assign", _EC, "WEIGHT", "-", "-", "-", "device", "-", "-",
     "DEVICE_TIER"),
    (_UC, "unified_cache::ensure_cached_alloc", 1, _ADOPT, _EC, "WEIGHT", "-", '"unified_cache:alloc"', "-", "device",
     "-", "-", "DEVICE_TIER"),
    (_UC, "unified_cache::ensure_cached_alloc", 2, _ADOPT, _EC, "WEIGHT", "-", '"unified_cache:alloc"', "-", "device",
     "-", "-", "DEVICE_TIER"),
    (_UC, "unified_cache_zone_allocate_owner", 1, "assign", _EC, "KV/WEIGHT", "-", '"exact_zone_allocate_owner"', "-",
     "device", "-", "-", "DEVICE_TIER"),
    (_UC, "unified_cache_zone_allocate", 1, "assign", _EC, "KV/WEIGHT", "-", '"exact_zone_allocate"', "-", "device",
     "-", "-", "DEVICE_TIER"),
    (_UC, "device_pool_alloc_chunk", 1, "assign", _EC, "WEIGHT", "-", 'tag ? tag : "layout_pool:chunk"', "-", "device",
     "-", "-", "DEVICE_TIER"),
    ("vram-pool.cpp", "vram_pool::allocate", 1, "assign", _EC, "WEIGHT", "-", "-", "-", "device", "-", "-",
     "DEVICE_TIER"),
]
SITE_FIELDS = ("form", "category", "role", "lifetime", "cohort", "owner", "tier", "zone")
HOLDER_CLASSES = {"GRAPH", "CONTEXT", "MODEL", "BUILDER", "DEVICE", "UNOWNED", "CALLERS"}


def table_rows() -> dict:
    out = {}
    for r in SITE_TABLE:
        key = (r[0], r[1], r[2])
        assert key not in out, f"SITE_TABLE lists {key} twice"
        out[key] = dict(zip(SITE_FIELDS, r[3:11]), holder=r[11], cls=r[12])
    return out


def site_key(site: dict):
    return (site["file"].split("/")[-1], site["func"], site["ordinal"])


# The twelve UNOWNED lines (addendum 10 (b)): each carries a cohort, and only these may carry UNOWNED.
UNOWNED_KEYS = {k for k, v in table_rows().items() if v["cls"] == "UNOWNED"}
assert len(UNOWNED_KEYS) == 12, "the site table carries the 12 UNOWNED lines of addendum 10 (b)"

# Each GRAPH row's plan term (§4.4 "GRAPH tenants"; r3 m4, r4 m-12). A GRAPH row with no term is UNSIZED, listed
# here with the reason, and ratcheted like UNOWNED.
# A single-tensor copy is bounded by the largest tensor a host zone can be asked to copy: a gather-only tensor the plan
# keeps on the host is read in place, so max_copied_tensor_bytes leaves it out.
SINGLE_TENSOR = ("max_copied_tensor_bytes", "k_scratch_headroom")
MOE_STAGING = ("moe_cpu_expert_staging_bytes", "moe_host_recipe_workspace_bytes")
SLOT_CAPACITY = ("slot_bytes",)  # the planned capacity of the staging kind, host_holder_device_state::slot_bytes
GRAPH_TERMS = {
    (_C, "ggml_sycl_get_staged_handle_device", 2): SINGLE_TENSOR,
    (_GS, "moe_hybrid_init_once", 1): SINGLE_TENSOR,  # one expert slice per iteration, released before the next
    (_GS, "moe_hybrid_init_once", 2): SINGLE_TENSOR,
    (_GS, "ggml_backend_sycl_buffer_get_tensor", 1): SINGLE_TENSOR,
    (_GS, "ggml_sycl_readback_via_shared_kernel", 1): SINGLE_TENSOR,
    (_GS, "ggml_sycl_attn_host_staging_per_call", 1): SLOT_CAPACITY,
    (_GS, "ggml_sycl_dispatch_host_flash_attn_sync", 1): SLOT_CAPACITY,  # a re-copy of Q, the Q kind's size
}
UNSIZED = {
    (_C, "ggml_sycl_tp_cache_ffn_norm", 1): "TP_TMP: tp_vram_runtime_bytes sizes device buffers, so it leaves host "
                                            "SCRATCH (§4.4 term trace, r4 m-9); TP is off on every lead arm",
    (_GS, "tp_device1_worker_thread_func", 1): "TP_TMP, as above",
    (_GS, "ggml_sycl_tp_launch_async_ffn", 1): "TP_TMP, as above",
    (_GS, "ggml_sycl_cpu_fallback_graph", 1): "one copy per device-resident tensor the CPU fallback graph reads, all "
                                              "live until the graph's exit: max_copied_tensor_bytes bounds one copy, "
                                              "not their sum",
}

# The DEVICE holders (§M62 (D)) and the call that releases each at module shutdown.
DEVICE_RELEASES = {
    ("pinned-buffer-pool.cpp", "PinnedBufferPool::init", 1):
        ("ggml_sycl_shutdown_global_runtime_pinned_owners",
         r"\bg_pinned_buffer_pools\s*\[\s*d\s*\]\s*\.\s*shutdown\s*\("),
    (_GS, "ggml_sycl_attn_host_staging_block_alloc", 1):
        ("ggml_sycl_attn_host_staging_free_all", r"\bg_host_staging_slots\s*->\s*release_all\s*\("),
}

# ---------------------------------------------------------------------------------------------------------------
# §4.4's published table, read from docs/design/sycl-mmap-host-tier.md and compared with the regenerated one. Each doc
# row maps to the tree rows it describes (a builder row names its variant) and the token its holder cell must carry.
# Line numbers are not compared: the doc's are rev 3aa's anchors, and the key here is the function and ordinal. Every
# other difference is listed row by row in DOC_DRIFT, and a listed difference that no longer differs fails as stale.
# ---------------------------------------------------------------------------------------------------------------

_B = (_GS, "ggml_sycl_host_buft_intent", 1)
DOC_ROW_MAP = {
    "`common.cpp:551`": ([((_C, "ggml_sycl_get_staged_handle_device", 2), 0)], "host_handle"),
    "`common.cpp:1414`": ([((_C, "ggml_sycl_tp_cache_ffn_norm", 1), 0)], "host_handle"),
    "`gs:8549`, `gs:8711`": ([((_GS, "moe_hybrid_init_once", 1), 0), ((_GS, "moe_hybrid_init_once", 2), 0)],
                             "d2h_staging"),
    "`gs:36787`": ([((_GS, "ggml_backend_sycl_buffer_get_tensor", 1), 0)], "fallback_handle"),
    "`gs:36906`": ([((_GS, "ggml_sycl_readback_via_shared_kernel", 1), 0)], "host_stage_handle"),
    "`gs:44294`": ([((_GS, "ggml_sycl_cpu_fallback_graph", 1), 0)], "host_copies"),
    "`gs:48244`": ([((_GS, "tp_device1_worker_thread_func", 1), 0)], "w.results"),
    "`gs:48982`": ([((_GS, "ggml_sycl_tp_launch_async_ffn", 1), 0)], "g_tp_async_ffn_jobs"),
    "`gs:82683`": ([((_GS, "ggml_sycl_dispatch_host_flash_attn_sync", 1), 0)], "vh"),
    "`gs:42652`, post-load": ([(_B, 1)], "SYCL_Host"),
    "`gs:42652`, `in_model_load`": ([(_B, 0)], "SYCL_Host"),
    "new, `SYCL_KV_Host`'s `alloc_buffer` (C4)": ([(_B, 2)], "idx_"),
    "`gs:42837`": ([(_B, 3)], "SYCL_HostCompute"),
    "`gs:44102`": ([((_GS, "ggml_backend_sycl_context::ensure_readback_staging", 1), 0)], "readback_staging_handle"),
    "`gs:57652`": ([], "moe_ids_cache"),
    "`uc:15652`": ([((_UC, "acquire_offload_buffer", 1), 0)], "offload_buffer_lease"),
    "`cpu-expert-pool.cpp:77`": ([], "g_cpu_expert_pools"),
    "`pinned-buffer-pool.cpp:69`": ([(("pinned-buffer-pool.cpp", "PinnedBufferPool::init", 1), 0)],
                                    "g_pinned_buffer_pools"),
    "`gs:82001`, `:82050`": ([((_GS, "ggml_sycl_attn_host_staging_block_alloc", 1), 0)], "g_host_staging_slots"),
    "`gs:82021`, `:82032`": ([((_GS, "ggml_sycl_attn_host_staging_per_call", 1), 0)], "fallback_holder"),
    "`common.cpp:443`": ([((_C, "ggml_sycl_get_staged_handle_device", 1), 0)], "g_runtime_staging_cache"),
    "`common.cpp:1970`": ([((_C, "ggml_sycl_tp_ensure_host_staging", 1), 0)], "g_tp_host_staging"),
    "`common.cpp:2083`": ([((_C, "ggml_sycl_tp_init_quant_comm_buffers_locked", 1), 0)], "g_tp_quant_comm_bufs"),
    "`common.cpp:2279`": ([((_C, "ggml_sycl_tp_ensure_shared_reduce_buffer", 1), 0)], "g_tp_shared_reduce_handle"),
    "`common.cpp:2328`": ([((_C, "ggml_sycl_tp_get_host_reduce_buffers", 1), 0)], "g_tp_host_buf0"),
    "`cpu-dispatch.cpp:2160`": ([((_CD, "ggml_sycl_cpu_retained_init", 1), 0)], "g_retained_scratch_lease"),
    "`gs:10882`": ([((_GS, "store_routing_indices", 1), 0)], "g_routing_indices_cache"),
    "`gs:21527`": ([((_GS, "arena_pinned_alloc", 1), 0)], "fused_handle"),
    "`gs:62932`, `gs:62947`": ([((_GS, "split_staging_ensure", 1), 0), ((_GS, "split_staging_ensure", 2), 0)],
                               "g_split_staging"),
    "`gs:63150`": ([((_GS, "split_weight_staging_ensure", 1), 0)], "g_split_weight_staging"),
    "`gs:63553`": ([((_GS, "ggml_sycl_mul_mat_tensor_split", 1), 0)], "s_cpu_out_ring_alloc"),
}

# (doc row, aspect) -> what the tree has instead. Aspects: row (no tree row), sites (a different number of request
# sites), holder (the doc's holder cell does not name the tree's holder), role, zone, lifetime, owner.
DOC_DRIFT = {
    ("`gs:48244`", "holder"): "the job slot is the TP worker's own w.results[layer], not a g_tp_* global; the TP sync "
                              "still consumes and clears it",
    ("`gs:42652`, `in_model_load`", "owner"): "the MODEL row carries its weight buffer's BUFFER owner, checked at that "
                                              "buffer's free_buffer: the deviation ggml_sycl_host_buft_intent records",
    ("`gs:57652`", "row"): "the MoE ids cache makes no host request in this tree",
    ("`cpu-expert-pool.cpp:77`", "row"): "the CpuExpertPool ring is gone: cpu-expert-pool.cpp makes no host request, "
                                         "and the pool holds only its workers",
    ("`gs:82001`, `:82050`", "sites"): "one request site: ggml_sycl_attn_host_staging_block_alloc allocates every "
                                       "kind's slot block",
    ("`gs:82001`, `:82050`", "holder"): "the slots are held by g_host_staging_slots (host_staging_slots), not "
                                        "g_attn_host_staging",
    ("`gs:82021`, `:82032`", "sites"): "the two per-call fallbacks are one function, "
                                       "ggml_sycl_attn_host_staging_per_call",
}

# Rows the tree has that rev 3aa's table does not.
NEW_ROWS = {
    (_GS, "ggml_sycl_reserve_host_tenants", 1): "the execution context's host tenant reservation: CONTEXT, owned by "
                                                "its EXECUTION_CONTEXT, landed after the table",
    (_GS, "ggml_sycl_moe_hybrid_decode_table_allocate", 1): "the MoE hybrid decode table: MODEL, owned by its MODEL, "
                                                            "landed after the table",
}

# The doc's per-class counts that the tree no longer matches: (phrase in the doc, class, role, the tree's count).
DOC_COUNT_DRIFT = [
    ("23 `EXPERT_STAGING`", "EXPERT", _ES, 24,
     "ggml_sycl_mul_mat_id's moe_q1_nvfp4_recipe_workspace is new; moe_stream_ring::reserve is a 25th, device tier"),
]


# ---------------------------------------------------------------------------------------------------------------
# Helpers over a tree. Each clause reads `texts` and returns its failures, one string each, empty when it holds.
# ---------------------------------------------------------------------------------------------------------------


def _src(texts: dict, rel: str) -> Source:
    return source_for(rel, texts[rel])


def _fn(texts: dict, rel: str, name: str, params=None):
    s = _src(texts, rel)
    hits = s.find(name, params)
    return s, (hits[0] if len(hits) == 1 else None)


def _body(texts: dict, rel: str, name: str, view: str = "bare", params=None) -> str:
    s, f = _fn(texts, rel, name, params)
    return getattr(s, view)[f["open"]:f["close"] + 1] if f else ""


def _production(texts: dict):
    return [rel for rel in sorted(texts) if rel.startswith(SYCL_DIR + "/") and rel.endswith((".cpp", ".hpp"))]


def _call_sites(texts: dict, rels, name_re: str):
    """(file, Source, enclosing function or None, call start, opening bracket) of each call matching `name_re`
    followed by `(`; definition heads and namespace-scope declarations are not calls."""
    out = []
    for rel in rels:
        if not re.search(name_re, texts[rel]):
            continue
        s = _src(texts, rel)
        heads = {f["name_at"] + len(f["name"]) for f in s.functions}
        for m in re.finditer(r"(?<![\w~])(?:" + name_re + r")\s*\(", s.bare):
            name_end = m.start() + len(re.match(r"[\w:]+", s.bare[m.start():]).group(0))
            if name_end in heads:
                continue
            f = s.enclosing(m.start())
            if f is None and re.search(r"[\w*&>]\s*$", s.bare[max(0, m.start() - 200):m.start()]):
                continue  # a declaration at namespace scope: a type precedes the name
            out.append((rel, s, f, m.start(), m.end() - 1))
    return out


def _calls(texts: dict, rels, name_re: str):
    """(file, enclosing function name or None, line) of each call matching `name_re` followed by `(`."""
    return [(rel, f["name"] if f else None, s.line(at)) for rel, s, f, at, _ in _call_sites(texts, rels, name_re)]


def _rows_by_key(texts: dict) -> dict:
    return {site_key(site): site for site in extract_sites(texts)}


def _variants(value: str, n: int):
    vals = value.split("|")
    return vals if len(vals) == n else vals * n if len(vals) == 1 else vals


def _row_lifetimes(row: dict):
    return ["GRAPH" if v == "-" else v for v in row["lifetime"].split("|")]


# ---------------------------------------------------------------------------------------------------------------
# REQUIRED clauses.
# ---------------------------------------------------------------------------------------------------------------


def clause_site_table(texts: dict) -> list:
    """§4.4: the site table regenerated from the tree, field by field, with each holder in its function and each group
    callee's tier claim in its body."""
    fails = []
    want = table_rows()
    got = _rows_by_key(texts)
    rules, final = select_zone_rules(texts)
    for key in sorted(set(got) - set(want)):
        fails.append(f"site table differs: {key} at line {got[key]['line']} is not in the site table (classify it: "
                     f"role, zone, lifetime, owner, holder)")
    for key in sorted(set(want) - set(got)):
        fails.append(f"site table differs: {key} is in the site table but not in the tree")
    for key in sorted(set(want) & set(got)):
        row, exp = dict(got[key]), want[key]
        row["zone"] = site_zones(rules, final, row)
        for field in SITE_FIELDS:
            if row[field] != exp[field]:
                fails.append(f"site table differs: {key} {field} is {row[field]!r} in the tree, {exp[field]!r} in the "
                             f"table")
        holder = exp["holder"]
        if exp["cls"] in HOLDER_CLASSES and holder == "-":
            fails.append(f"site table: {key} ({exp['cls']}) names no holder")
        if holder != "-":
            token, _, where = holder.partition("@")
            rel = SYCL_DIR + "/" + key[0]
            bodies = [_src(texts, rel).code[f["head"]:f["close"]] for f in _src(texts, rel).find(where or key[1])]
            if not any(re.search(r"(?<![\w])" + re.escape(token), b) for b in bodies):
                fails.append(f"site table: holder {token!r} of {key} is not in {where or key[1]}")
    for callee, (_, tier) in GROUP_CALLEES.items():
        if callee in ("<intent>", "unified_cache_adopt_raw_device_allocation") or not isinstance(tier, str):
            continue
        defs = [b for rel in _production(texts) for f in _src(texts, rel).find(callee)
                if "runtime_category" in f["params"]
                for b in [_src(texts, rel).bare[f["open"]:f["close"]]]]
        if not defs or not all(re.search(r"must_host_pinned\s*=\s*true", b) for b in defs):
            fails.append(f"site table: the group callee {callee} is read as tier {tier}, but its definition does not "
                         f"set must_host_pinned")
    return fails


def _doc_table(texts: dict):
    doc = texts.get(DOC, "")
    at = doc.find("- **The site table** (addendum 7 (c))")
    if at < 0:
        return None
    rows = {}
    for line in doc[at:].split("\n")[1:]:
        if line.startswith("  - **Addendum 10 (b)"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")] if line.strip().startswith("| ") else []
        if len(cells) == 6 and not cells[0].startswith(("site", "---")):
            rows[cells[0]] = cells
    return rows


def _first_word(cell: str) -> str:
    return re.sub(r"[*`]", "", cell).split(",")[0].split()[0] if cell.strip() else ""


def _doc_owner(cell: str) -> str:
    c = cell.strip()
    if c == "-":
        return "-"
    if "ggml_backend_buffer" in c:
        return "BUFFER"
    if "ModelId" in c:
        return "MODEL"
    if "backend context" in c:
        return "BACKEND"
    if "last live backend" in c or c.startswith("the same"):
        return "DEVICE"
    if c == "caller's":
        return "caller's"
    if "after llama.cpp-30r1" in c or c == "same":
        return "none until llama.cpp-30r1"
    return "?" + c


def _tree_doc_view(row: dict, cls: str, variant: int):
    n = len(row["role"].split("|"))
    if cls == "CALLERS":
        return {"role": "caller's", "zone": row["zone"], "lifetime": "caller's", "owner": "caller's"}
    owner = "none until llama.cpp-30r1" if cls == "UNOWNED" else "BUFFER" if cls == "BUILDER" else \
        _variants(row["owner"], n)[variant]
    lifetime = _variants(row["lifetime"], n)[variant]
    return {"role": row["role"].split("|")[variant], "zone": _variants(row["zone"], n)[variant],
            "lifetime": "GRAPH" if lifetime == "-" else lifetime, "owner": owner}


_DOC_FILES = {"gs": "ggml-sycl.cpp", "uc": "unified-cache.cpp"}


def clause_doc_table(texts: dict) -> list:
    """§4.4: the published table against the regenerated one, row by row; every difference is listed in DOC_DRIFT."""
    fails = []
    doc_rows = _doc_table(texts)
    if doc_rows is None:
        return ["doc table: §4.4's site table is not in " + DOC]
    want = table_rows()
    differs = set()
    for cell in sorted(set(doc_rows) - set(DOC_ROW_MAP)):
        fails.append(f"doc table: the §4.4 row {cell} is not mapped to the tree (DOC_ROW_MAP)")
    for cell in sorted(set(DOC_ROW_MAP) - set(doc_rows)):
        fails.append(f"doc table: DOC_ROW_MAP names {cell}, which §4.4's table no longer has")
    for cell, (targets, doc_holder) in DOC_ROW_MAP.items():
        if cell not in doc_rows:
            continue
        _, holder_cell, role, zone, lifetime, owner = doc_rows[cell]
        if not targets:
            differs.add((cell, "row"))
            continue
        refs = re.findall(r"`(?:([\w.-]+):)?:?(\d+)`", cell)
        if max(1, len(refs)) != len(targets):
            differs.add((cell, "sites"))
        files = {_DOC_FILES.get(f, f) for f, _ in refs if f}
        for key, _ in targets:
            if files and key[0] not in files:
                differs.add((cell, "file"))
        if doc_holder not in holder_cell:
            differs.add((cell, "holder"))
        doc_view = {"role": _first_word(role), "zone": _first_word(zone), "lifetime": _first_word(lifetime),
                    "owner": _doc_owner(owner)}
        for key, variant in targets:
            if key not in want:
                fails.append(f"doc table: {cell} maps to {key}, which is not in the site table")
                continue
            tree_view = _tree_doc_view(want[key], want[key]["cls"], variant)
            for aspect in ("role", "zone", "lifetime", "owner"):
                if doc_view[aspect] != tree_view[aspect]:
                    differs.add((cell, aspect))
                    if (cell, aspect) not in DOC_DRIFT:
                        fails.append(f"doc table differs: {cell} {aspect} is {doc_view[aspect]!r} in §4.4 and "
                                     f"{tree_view[aspect]!r} in the tree")
    for item in sorted(differs - set(DOC_DRIFT)):
        if item[1] in ("row", "sites", "file", "holder"):
            fails.append(f"doc table differs: {item[0]} {item[1]} differs from the tree and is not listed in DOC_DRIFT")
    for item in sorted(set(DOC_DRIFT) - differs):
        fails.append(f"doc table: stale drift entry {item}: §4.4 and the tree agree there now")
    mapped = {key for targets, _ in DOC_ROW_MAP.values() for key, _ in targets}
    for key, row in sorted(want.items()):
        if row["cls"] in HOLDER_CLASSES and key not in mapped and key not in NEW_ROWS:
            fails.append(f"doc table: {key} ({row['cls']}) is in neither §4.4's table nor NEW_ROWS")
    for key in sorted(set(NEW_ROWS) & mapped):
        fails.append(f"doc table: stale NEW_ROWS entry {key}: §4.4's table has it")
    doc = texts.get(DOC, "")
    for phrase, cls, role, count, _ in DOC_COUNT_DRIFT:
        n = sum(1 for r in want.values() if r["cls"] == cls and r["role"] == role)
        if phrase not in doc:
            fails.append(f"doc table: stale count drift {phrase!r}: §4.4 no longer says it")
        elif n != count:
            fails.append(f"doc table: the tree has {n} {cls} rows of role {role}, DOC_COUNT_DRIFT records {count}")
    return fails


def clause_select_zone(texts: dict) -> list:
    """§4.4 / addendum 10 (b): WEIGHT has no category clause; UNOWNED goes to WEIGHT; HOST_COMPUTE and EXPERT_CACHE
    go to SCRATCH by role and lifetime."""
    fails = []
    try:
        rules, final = select_zone_rules(texts)
    except AssertionError as e:
        return [f"select_zone: {e}"]
    for disj, zone in rules:
        if zone == "WEIGHT" and any(a.startswith("category") for conj in disj for a in conj):
            fails.append("select_zone has a category clause for WEIGHT: routing is by role and lifetime")
    cases = [
        (("STAGING", "UNOWNED", "HOST_COMPUTE"), "WEIGHT"), (("TP_TMP", "UNOWNED", "HOST_COMPUTE"), "WEIGHT"),
        (("COMPUTE", "UNOWNED", "HOST_COMPUTE"), "WEIGHT"), (("STAGING", "GRAPH", "HOST_COMPUTE"), "SCRATCH"),
        (("COMPUTE", "CONTEXT", "HOST_COMPUTE"), "SCRATCH"), (("COMPUTE", "DEVICE", "EXPERT_CACHE"), "SCRATCH"),
        (("EXPERT_STAGING", "GRAPH", "HOST_COMPUTE"), "SCRATCH"), (("KV", "CONTEXT", "HOST_COMPUTE"), "KV"),
        (("WEIGHT", "MODEL", "HOST_COMPUTE"), "WEIGHT"),
    ]
    for (role, life, cat), zone in cases:
        got = zone_for(rules, final, role, life, cat)
        if got != zone:
            fails.append(f"select_zone routes role {role} lifetime {life} category {cat} to {got}, not {zone}")
    return fails


def clause_owners(texts: dict) -> list:
    """§7 C4: every non-GRAPH row names an owner; the builder's rows get the buffer's, minted by its alloc_buffer."""
    fails = []
    want = table_rows()
    for key, row in sorted(_rows_by_key(texts).items()):
        cls = want.get(key, {}).get("cls")
        if cls in (None, "MAPPING", "NON_TENANT", "CALLERS", "DEVICE_TIER") or row["tier"] == "device":
            continue
        lifes = _row_lifetimes(row)
        owners = _variants(row["owner"], len(lifes))
        for life, owner in zip(lifes, owners):
            if life in ("GRAPH", "UNOWNED"):
                continue
            if cls == "BUILDER":
                continue
            if owner == "-":
                fails.append(f"owner: {key} at line {row['line']} is lifetime {life} and names no owner")
    alloc = _body(texts, GS, "ggml_sycl_host_buft_alloc")
    if not re.search(r"\bintent\s*\.\s*owner\s*=\s*\{\s*(?:ggml_sycl::)?alloc_owner_kind::BUFFER\b", alloc):
        fails.append("owner: ggml_sycl_host_buft_alloc does not mint the builder rows' BUFFER owner")
    for kind in ("HOST", "KV_HOST"):
        n = len(re.findall(r"ggml_sycl_host_buft_alloc\s*\([^;]*ggml_sycl_host_buft_intent\s*\(\s*"
                           r"ggml_sycl_host_buft_kind::" + kind + r"\s*\)", _src(texts, GS).bare))
        if n != 1:
            fails.append(f"owner: the {kind} buft's alloc_buffer does not allocate through the builder (found {n})")
    return fails


def clause_unowned_ratchet(texts: dict) -> list:
    """Addendum 10 (b): only the table's 12 lines carry UNOWNED, each with a cohort."""
    fails = []
    rows = _rows_by_key(texts)
    for key, row in sorted(rows.items()):
        if "UNOWNED" in row["lifetime"].split("|"):
            if key not in UNOWNED_KEYS:
                fails.append(f"UNOWNED ratchet: {key} at line {row['line']} carries UNOWNED and is not one of the "
                             f"site table's 12 lines; llama.cpp-30r1 moves the 12 to an owner, and no line joins them")
            elif row["cohort"] == "-":
                fails.append(f"UNOWNED ratchet: {key} carries UNOWNED with no cohort")
    for key in sorted(UNOWNED_KEYS):
        if key in rows and "UNOWNED" not in rows[key]["lifetime"]:
            fails.append(f"UNOWNED ratchet: {key} no longer carries UNOWNED; move it out of the 12 with its owner "
                         f"(llama.cpp-30r1)")
    allowed = {}
    for key in UNOWNED_KEYS:
        allowed[(SYCL_DIR + "/" + key[0], key[1])] = allowed.get((SYCL_DIR + "/" + key[0], key[1]), 0) + 1
    seen = {}
    for rel in _production(texts):
        if "alloc_lifetime::UNOWNED" not in texts[rel]:
            continue
        s = _src(texts, rel)
        for m in re.finditer(r"\balloc_lifetime::UNOWNED\b", s.bare):
            if _mapping_kind(s.bare, m.start()):
                continue
            f = s.enclosing(m.start())
            k = (rel, f["name"] if f else None)
            seen[k] = seen.get(k, 0) + 1
            if seen[k] > allowed.get(k, 0):
                fails.append(f"UNOWNED ratchet: {rel}:{s.line(m.start())} ({k[1]}) stamps UNOWNED outside the site "
                             f"table's 12 lines; llama.cpp-30r1 is the only way out of UNOWNED")
    return fails


def _scratch_fold_terms(texts: dict):
    """The identifiers the host SCRATCH zone sum folds, with a local defined as a sum expanded."""
    s = _src(texts, UC)
    m = re.search(r"plan\s*\.\s*host_zone_scratch_bytes\s*=\s*std::max<size_t>\(\s*k_min_zone_bytes\s*,\s*"
                  r"host_scratch_total\s*\)", s.bare)
    if not m:
        return None
    f = s.enclosing(m.start())
    body = s.bare[f["open"]:m.start()]
    folds = list(re.finditer(r"for\s*\(\s*size_t\s+bytes\s*:\s*\{([^}]*)\}\s*\)", body))
    if not folds:
        return None
    fold = folds[-1]
    terms = set()
    for expr in fold.group(1).split(","):
        name = re.findall(r"\w+", expr)[-1]
        local = re.search(r"\b(?:const\s+)?size_t\s+" + name + r"\s*=\s*([^;]+);", body)
        if local and "plan" in local.group(1):
            terms |= set(re.findall(r"plan\s*\.\s*(\w+)", local.group(1)))
        else:
            terms.add(name)
    return terms


def clause_unsized_ratchet(texts: dict) -> list:
    """§4.4 "GRAPH tenants" (r3 m4): each GRAPH row maps to a plan term the host SCRATCH sum reads, or is UNSIZED."""
    fails = []
    want = table_rows()
    folded = _scratch_fold_terms(texts) or set()
    hzp_h = texts.get(SYCL_DIR + "/host-zone-plan.hpp", "")
    for key, row in sorted(_rows_by_key(texts).items()):
        cls = want.get(key, {}).get("cls")
        lifes = _row_lifetimes(row)
        if row["tier"] != "host" or set(lifes) != {"GRAPH"} or cls in ("MAPPING", "NON_TENANT", "CALLERS", "WEIGHT"):
            continue
        terms = MOE_STAGING if row["role"] == "EXPERT_STAGING" else GRAPH_TERMS.get(key)
        if terms is None:
            if key not in UNSIZED:
                fails.append(f"UNSIZED ratchet: the GRAPH row {key} at line {row['line']} maps to no plan term and is "
                             f"not one of the UNSIZED rows (r3 m4)")
            continue
        for term in terms:
            if term == "slot_bytes":
                if not re.search(r"\bslot_bytes\b", hzp_h):
                    fails.append(f"UNSIZED ratchet: {key} maps to the planned slot capacity, which "
                                 f"host_holder_device_state no longer carries")
            elif term not in folded:
                fails.append(f"UNSIZED ratchet: {key} maps to {term}, which the host SCRATCH sum does not read")
    for key in sorted(set(UNSIZED) & set(GRAPH_TERMS)):
        fails.append(f"UNSIZED ratchet: {key} is both mapped and UNSIZED")
    return fails


def clause_device_holders(texts: dict) -> list:
    """§M62 (D): the DEVICE holders' requests, their release at shutdown, and the DEVICE check after delete sycl_ctx."""
    fails = []
    rows = _rows_by_key(texts)
    device_rows = {k for k, r in rows.items() if "DEVICE" in r["lifetime"].split("|")}
    for key in sorted(set(DEVICE_RELEASES) - device_rows):
        fails.append(f"DEVICE holder: {key} does not set lifetime = DEVICE")
    for key in sorted(device_rows - set(DEVICE_RELEASES)):
        fails.append(f"DEVICE holder: {key} sets lifetime = DEVICE and has no recorded release")
    for key in sorted(device_rows & set(DEVICE_RELEASES)):
        if rows[key]["owner"] != "DEVICE":
            fails.append(f"DEVICE holder: {key} names no DEVICE owner")
        fn, release = DEVICE_RELEASES[key]
        if not re.search(release, _body(texts, GS, fn)):
            fails.append(f"DEVICE holder: {fn} no longer releases {key[1]}'s holder")
    shutdown = _body(texts, GS, "ggml_sycl_shutdown_global_runtime_pinned_owners")
    if re.search(r"owner_check|boundary_check|host_zone_record", shutdown):
        fails.append("DEVICE holder: ggml_sycl_shutdown_global_runtime_pinned_owners checks a record (§M62 (D))")
    module = _body(texts, GS, "ggml_backend_sycl_shutdown")
    a = module.find("ggml_sycl_shutdown_global_runtime_pinned_owners(")
    b = module.find("ggml_sycl_attn_host_staging_free_all(")
    if a < 0 or b < 0 or b < a:
        fails.append("DEVICE holder: module shutdown does not release the staging slots after the pinned owners")
    free = _body(texts, GS, "ggml_backend_sycl_free")
    d = free.find("delete sycl_ctx")
    check = re.search(r"unified_cache_host_zone_owner_check\s*\(\s*ggml_sycl::alloc_lifetime::DEVICE", free)
    if d < 0 or not check:
        fails.append("DEVICE holder: ggml_backend_sycl_free has no DEVICE check after delete sycl_ctx")
    else:
        if check.start() < d or re.search(r"alloc_lifetime::DEVICE", free[:d]):
            fails.append("DEVICE holder: the DEVICE check runs before delete sycl_ctx")
        gate = free.rfind("host_zone_device_check_due(", d, check.start())
        if gate < 0 or free.find("{", gate) > check.start() or free[free.find("{", gate):check.start()].count("}"):
            fails.append("DEVICE holder: the DEVICE check is not gated on host_zone_device_check_due (the device's "
                         "live-backend count reaching 0)")
    return fails


def clause_slot_refusal(texts: dict) -> list:
    """§M62 (E): a planned slot refuses above its capacity by name; the IDLE and persistent paths go through admit."""
    fails = []
    admit = _body(texts, PLAN, "host_staging_slots::admit")
    planned_at = admit.find("if (bytes <= slot.cap)")
    if planned_at < 0:
        return ["slot refusal: host_staging_slots::admit has no planned capacity test"]
    planned = admit[planned_at:]
    if re.search(r"\bslot_replace\s*\(", planned):
        fails.append("slot refusal: a planned slot grows at dispatch above its capacity")
    if "exceeds the planned" not in _body(texts, PLAN, "host_staging_slots::admit", "code")[planned_at:] or \
            not re.search(r"counters\s*->\s*refusal\s*\(", planned):
        fails.append("slot refusal: the planned refusal no longer WARNs by name and counts")
    get = _body(texts, GS, "ggml_sycl_attn_host_staging_get")
    idle = get.find("if (idle_slots")
    per_call = get.find("ggml_sycl_attn_host_staging_per_call(", idle)
    if idle < 0 or not (idle < get.find("g_host_staging_slots->admit(", idle) < per_call):
        fails.append("slot refusal: the IDLE path hands out a per-call buffer without the slot's admission")
    if not re.search(r"return\s+g_host_staging_slots\s*->\s*admit\s*\(", get):
        fails.append("slot refusal: the persistent path does not return the slot's admission")
    if len(re.findall(r"\bggml_sycl_attn_host_staging_block_alloc\b", _src(texts, GS).bare)) != 2:
        fails.append("slot refusal: ggml_sycl_attn_host_staging_block_alloc is reached other than through the "
                     "staging allocator")
    return fails


def clause_staging_plan(texts: dict) -> list:
    """§M69 (a), (d), r5 m-6: the plan stages only with the host dispatch on, selects by the interception's predicate,
    and takes each node's device from the scheduler's backend."""
    fails = []
    visit = _body(texts, PLAN, "walk_visit_host_kv")
    if not re.search(r"if\s*\(\s*!\s*w\s*\.\s*env\s*->\s*host_dispatch_on\s*\)\s*\{\s*return\s*;", visit):
        fails.append("staging plan: walk_visit_host_kv stages with the host dispatch off (§M69 (a))")
    walk = _body(texts, PLAN, "host_holder_plans::walk", "code")
    if not re.search(r"staging_planned\s*=\s*env\s*\.\s*host_dispatch_on\s*;", walk) or \
            "staging not planned (host dispatch off)" not in walk:
        fails.append("staging plan: the walk does not print `staging not planned` when the host dispatch is off")
    pred = _body(texts, PLAN, "walk_in_kv_host")
    if not re.search(r"return\s+t\s*&&\s*env\s*\.\s*in_kv_host_buft\s*\(\s*t\s*\)", pred):
        fails.append("staging plan: walk_in_kv_host is not the interception's predicate (r5 m-6)")
    for fn in ("walk_in_kv_host", "walk_is_host_kv_node", "walk_visit_host_kv", "host_holder_plans::walk",
               "walk_staging_device"):
        if re.search(r"\bggml_backend_buffer_is_host\s*\(", _body(texts, PLAN, fn)):
            fails.append(f"staging plan: {fn} calls ggml_backend_buffer_is_host (r5 m-6)")
    env = _body(texts, GS, "ggml_sycl_host_holder_walk_env")
    if not re.search(r"env\s*\.\s*in_kv_host_buft\s*=\s*ggml_sycl_tensor_is_in_kv_host_buft\s*;", env):
        fails.append("staging plan: the walk's predicate is not ggml_sycl_tensor_is_in_kv_host_buft")
    if not re.search(r"env\s*\.\s*host_dispatch_on\s*=\s*ggml_sycl_attn_host_dispatch_enabled\s*\(\s*\)\s*;", env):
        fails.append("staging plan: the walk's switch is not ggml_sycl_attn_host_dispatch_enabled()")
    dev = _body(texts, PLAN, "walk_staging_device")
    if not re.search(r"return\s+w\s*\.\s*env\s*->\s*backend_device\s*\(\s*backend\s*\)\s*;", dev):
        fails.append("staging plan: a slot's device is not the scheduler's assigned backend's (§M69 (d))")
    if not re.search(r"env\s*\.\s*backend_device\s*=\s*ggml_sycl_host_holder_backend_device\s*;", env) or \
            not re.search(r"backend\s*->\s*context\s*\)\s*->\s*device",
                          _body(texts, GS, "ggml_sycl_host_holder_backend_device")):
        fails.append("staging plan: the backend's device does not come from the backend's own context (§M69 (d))")
    return fails


def clause_pool_dims(texts: dict) -> list:
    """r5 m-7, r7 I-2: one dimension fold, host_pool_dims_add; the pools' eligibility is placement's fact."""
    fails = []
    prod = _production(texts)
    for rel in prod:
        if "max_dispatch_count" in texts[rel] and re.search(r"\bmax_dispatch_count\b", _src(texts, rel).bare):
            fails.append(f"pool dims: {rel} computes max_dispatch_count outside host_pool_dims_add")
        s = _src(texts, rel)
        for m in re.finditer(r"\bids\s*->\s*ne\s*\[\s*0\s*\]\s*\*\s*std::max<int64_t>", s.bare):
            f = s.enclosing(m.start())
            if not f or f["name"] != "host_pool_dims_add":
                fails.append(f"pool dims: {rel}:{s.line(m.start())} computes the dispatch count outside "
                             f"host_pool_dims_add")
    callers = {c[1] for c in _calls(texts, prod, r"host_pool_dims_add")}
    if callers != {"host_pool_scan", "walk_visit_mul_mat_id"}:
        fails.append(f"pool dims: host_pool_dims_add is called by {sorted(map(str, callers))}, not by the scan and "
                     f"the plan walk alone")
    scans = sorted(map(str, {c[1] for c in _calls(texts, prod, r"host_pool_scan")}))
    if scans != ["host_pools_init_from_plan", "moe_hybrid_init_once"]:
        fails.append(f"pool dims: host_pool_scan is called by {scans}, not by moe_hybrid_init_once and the plan's "
                     f"init check alone")
    init = _body(texts, GS, "moe_hybrid_init_once")
    if re.search(r"\bholder_demand\s*\(|host_scratch_ledger|ensure_host_scratch", init):
        fails.append("pool dims: moe_hybrid_init_once registers SCRATCH demand; the plan's commit does (r7 I-1)")
    placed = _body(texts, PLAN, "pool_host_placed")
    if re.search(r"\bggml_backend_buffer_is_host\s*\(", placed) or \
            not re.search(r"return\s+host_experts\s*&&\s*host_experts\s*\(", placed):
        fails.append("pool dims: the pools' eligibility is not placement's executor fact (r7 I-2)")
    # The pointer pool_host_placed calls is that fact only if every binding of it names the function that reads
    # the MoE plan: the walk env's member and the pools' init argument.
    fact = "ggml_sycl_host_holder_host_experts"
    binds = []
    for rel in prod:
        if "host_experts" in texts[rel]:
            s = _src(texts, rel)
            binds += [(f"{rel}:{s.line(m.start())}", m.group(1))
                      for m in re.finditer(r"\.\s*host_experts\s*=(?!=)\s*(\w+)\s*;", s.bare)
                      if m.group(1) != "host_experts"]  # the plan's state copying its own parameter
    if not binds or any(v != fact for _, v in binds):
        fails.append(f"pool dims: the walk env's host_experts is bound to {binds}, not to {fact} alone (r7 I-2)")
    inits = _call_sites(texts, prod, r"(?:ggml_sycl::)?host_pools_init_from_plan")
    for rel, s, _, at, lp in inits:
        args = [s.bare[a:b].strip() for a, b in call_args(s.bare, lp)]
        if len(args) < 4 or args[3] != fact:
            fails.append(f"pool dims: {rel}:{s.line(at)} passes host_pools_init_from_plan the host-experts fact "
                         f"{args[3] if len(args) > 3 else None!r}, not {fact} (r7 I-2)")
    if not inits:
        fails.append("pool dims: nothing calls host_pools_init_from_plan")
    reads = _body(texts, GS, fact)
    if not re.search(r"\bggml_sycl_moe_plan_has_host_experts\s*\(", reads) or \
            re.search(r"\bggml_backend_buffer_is_host\s*\(", reads):
        fails.append(f"pool dims: {fact} does not read ggml_sycl_moe_plan_has_host_experts, placement's plan "
                     f"(r7 I-2)")
    return fails


def clause_marked_set_reach(texts: dict) -> list:
    """§M75: the pool frees a chunk only in its destructor and the carrier's; release_zone_chunk is reached only from
    the marked-set cores, whose callers are the listed entries."""
    fails = []
    pool = [POOL]
    frees = {c[1] for c in _calls(texts, pool, r"free_chunk_owner")}
    if frees != {"pinned_chunk_pool::~pinned_chunk_pool", "pinned_chunk_pool::released_chunk::~released_chunk"}:
        fails.append(f"marked set: free_chunk_owner is called by {sorted(map(str, frees))}, not only by the pool's "
                     f"destructor and the carrier's")
    s = _src(texts, POOL)
    for m in re.finditer(r"\bremoved_\s*=(?!=)", s.bare):
        f = s.enclosing(m.start())
        stmt = s.bare[m.start():s.bare.find(";", m.start())]
        if not (f and f["name"] == "pinned_chunk_pool::release_zone_chunk") and \
                not re.fullmatch(r"removed_\s*=\s*std::move\s*\(\s*other\s*\.\s*removed_\s*\)", stmt.strip()):
            fails.append(f"marked set: a carrier's chunk is filled at pinned-pool.cpp:{s.line(m.start())}, outside "
                         f"release_zone_chunk")
    rz = {c[1] for c in _calls(texts, _production(texts), r"(?:\w+\s*(?:->|\.)\s*)?release_zone_chunk")}
    if rz != {"unified_cache::release_marked_chunk_locked"}:
        fails.append(f"marked set: release_zone_chunk is called by {sorted(map(str, rz))}, not by "
                     f"release_marked_chunk alone")
    allowed = {"unified_cache::host_scratch_begin_release", "unified_cache::host_scratch_return_chunks",
               "unified_cache::host_scratch_owner_end", "unified_cache::configure_host_zones",
               "unified_cache::release_marked_chunk", "unified_cache::host_release_chunk_lease",
               "unified_cache::host_zone_free"}
    rm = {c[1] for c in _calls(texts, _production(texts), r"release_marked_chunk(?:_locked)?")}
    for name in sorted(map(str, rm - allowed)):
        fails.append(f"marked set: release_marked_chunk is called by {name}, which is not one of the begin entry, "
                     f"the returns, the owner's end and configure_host_zones")
    return fails


def clause_return_files(texts: dict) -> list:
    """§M75, r8 m-1: RETURN withdraws each chunk and then files it in the marked set, in one scope of its lock."""
    fails = []
    body = _body(texts, UC, "unified_cache::host_scratch_return_chunks")
    lock = re.search(r"lock_guard<std::mutex>\s+\w+\s*\(\s*marked_set_mutex_\s*\)", body)
    w = re.search(r"\bwithdraw_zone_chunk\s*\(\s*host_zone_id::SCRATCH", body)
    p = re.search(r"\bmarked_set_\s*\.\s*push_back\s*\(", body)
    if not (lock and w and p):
        return ["marked set: RETURN files a chunk it did not withdraw, or outside marked_set_mutex_"]
    if not (lock.start() < w.start() < p.start()):
        fails.append("marked set: RETURN files a chunk before withdrawing it")
    between = body[w.start():p.start()]
    if not re.search(r"continue\s*;", between):
        fails.append("marked set: RETURN files a chunk whose withdrawal was refused")
    return fails


def clause_withdrawal(texts: dict) -> list:
    """r9 m-3, r10 m-4: withdrawal is the one writer that lowers a zone's size, on the flag's false-to-true edge;
    release_zone_chunk writes no zone."""
    fails = []
    s = _src(texts, POOL)
    for m in re.finditer(r"zones_\s*\[[^\]]*\]\s*\.\s*size\s*-=", s.bare):
        f = s.enclosing(m.start())
        if not f or f["name"] != "pinned_chunk_pool::withdraw_zone_chunk":
            fails.append(f"withdrawal: pinned-pool.cpp:{s.line(m.start())} lowers a zone's size outside "
                         f"withdraw_zone_chunk")
    w = _body(texts, POOL, "pinned_chunk_pool::withdraw_zone_chunk")
    if not re.search(r"if\s*\(\s*!\s*zcs\s*\.\s*withdrawn\s*\)\s*\{\s*zcs\s*\.\s*withdrawn\s*=\s*true\s*;\s*"
                     r"zones_\s*\[\s*zi\s*\]\s*\.\s*size\s*-=\s*zcs\s*\.\s*zone_size\s*;\s*\}", w):
        fails.append("withdrawal: the zone's size is not lowered exactly once, on the withdrawn flag's false-to-true "
                     "edge")
    r = _body(texts, POOL, "pinned_chunk_pool::release_zone_chunk")
    if re.search(r"zones_\s*\[[^\]]*\]\s*\.\s*\w+\s*[-+]?=(?!=)", r):
        fails.append("withdrawal: release_zone_chunk writes a zone")
    return fails


def clause_notify_device(texts: dict) -> list:
    """§4.4 "One source for the host bufts' device" (I-D), r3 I-1: the notify grows the host buft's device's SCRATCH
    through the ledger's ensure call, and its growth path resolves the ledger of that device, not device 0's."""
    fails = []
    notify = _body(texts, GS, "ggml_backend_sycl_notify_host_buffer_sizes")
    # The notify and the commit's mark build their view of the backend in one helper.
    env = _body(texts, GS, "ggml_sycl_notify_env")
    if not re.search(r"env\s*\.\s*host_buft_device\s*=\s*ggml_sycl_notify_host_buft_device\s*;", env):
        fails.append("notify device: the notify's device is not ggml_sycl_notify_host_buft_device")
    if not re.search(r"env\s*\.\s*ledger_of\s*=\s*ggml_sycl_notify_ledger_of\s*;", env) or \
            not re.search(r"return\s+ggml_sycl::host_scratch_notify\s*\(\s*ggml_sycl_notify_env\s*\(\s*\)\s*,\s*"
                          r"backend\s*,", notify):
        fails.append("notify device: the notify does not grow through host_scratch_notify, the ledger's "
                     "ensure-capacity call, with ggml_sycl_notify_ledger_of as its growth path")
    ledger = _body(texts, GS, "ggml_sycl_notify_ledger_of")
    if not re.search(r"\bunified_cache_host_scratch_ledger\s*\(\s*device\s*,", ledger):
        fails.append("notify device: ggml_sycl_notify_ledger_of does not resolve the ledger of the device "
                     "host_scratch_notify hands it")
    dev = _body(texts, GS, "ggml_sycl_notify_host_buft_device")
    if not re.search(r"return\s+ggml_sycl_host_buft_device\s*\(\s*ggml_backend_dev_host_buffer_type\s*\(\s*"
                     r"ggml_backend_get_device\s*\(\s*backend\s*\)\s*\)\s*\)\s*;", dev):
        fails.append("notify device: the notify's device is not the host buft's (the buft llama_context allocates "
                     "from)")
    for fn in ("ggml_backend_sycl_notify_host_buffer_sizes", "ggml_sycl_notify_env",
               "ggml_sycl_notify_host_buft_device", "ggml_sycl_notify_ledger_of"):
        b = _body(texts, GS, fn)
        if re.search(r"backend\s*->\s*context\s*\)\s*->\s*device|ctx\s*->\s*device|get_current_device_id|"
                     r"grow_(?:host_)?scratch_zone|\(\s*0\s*\)", b):
            fails.append(f"notify device: {fn} resolves the notifying backend's device, device 0, or grows "
                         f"additively")
    return fails


def clause_grow_host_scratch_gone(texts: dict) -> list:
    """§M68 (a): unified_cache_grow_host_scratch_zone has zero definitions and zero callers."""
    fails = []
    for rel in _production(texts) + [HEADER]:
        if rel in texts and re.search(r"\bunified_cache_grow_host_scratch_zone\b", _src(texts, rel).bare):
            fails.append(f"grow_host_scratch_zone: {rel} defines or calls unified_cache_grow_host_scratch_zone")
    return fails


def clause_notify_compute_unused(texts: dict) -> list:
    """§7 C4: nothing in src/ calls ggml_backend_sycl_notify_compute_buffer_sizes, by name or through a proc address
    looked up by its name (its VRAM half is not 5efe's)."""
    fails = []
    for rel in sorted(texts):
        if rel.startswith(SRC_DIR + "/") and "ggml_backend_sycl_notify_compute_buffer_sizes" in texts[rel]:
            s = _src(texts, rel)
            for m in re.finditer(r"\bggml_backend_sycl_notify_compute_buffer_sizes\b", s.code):
                fails.append(f"notify compute: {rel}:{s.line(m.start())} names "
                             f"ggml_backend_sycl_notify_compute_buffer_sizes")
    return fails


_SCALAR_RE = r"(?:const\s+)?(?:size_t|int|int32_t|int64_t|uint32_t|uint64_t|bool|float|double)(?:\s+\w+)?"


def _fn_type(text: str, m):
    """(return type, [parameters]) of a declaration whose return type is `m`'s group 1 and whose parameter list
    opens at the end of `m`, whitespace collapsed."""
    return (re.sub(r"\s+", " ", m.group(1)).strip(),
            [re.sub(r"\s+", " ", text[a:b]).strip() for a, b in call_args(text, m.end() - 1)])


def _notify_shape(what: str, ret: str, params: list) -> list:
    """The notify's dlopen shape: an opaque backend handle plus scalars, returning bool (r3 m5)."""
    fails = []
    if ret != "bool":
        fails.append(f"dlopen shape: {what} returns {ret!r}, not bool")
    if not params or not re.fullmatch(r"ggml_backend_t(?:\s+\w+)?", params[0]) or \
            not all(re.fullmatch(_SCALAR_RE, q) for q in params[1:]):
        fails.append(f"dlopen shape: {what} takes {params}, not an opaque backend handle and scalars")
    return fails


def _plan_shape(what: str, ret: str, params: list) -> list:
    """The plan entry's dlopen shape: a ggml_backend_sched_t and a struct ggml_cgraph *, and no llama type (r3 m5)."""
    if len(params) == 2 and re.fullmatch(r"ggml_backend_sched_t(?:\s+\w+)?", params[0]) and \
            re.fullmatch(r"(?:struct\s+)?ggml_cgraph\s*\*\s*\w*", params[1]) and "llama" not in ret:
        return []
    return [f"dlopen shape: {what} is {ret} ({', '.join(params)}), not a ggml_backend_sched_t and a ggml_cgraph * "
            f"with no llama type"]


_DLOPEN_ENTRIES = ((NOTIFY, "notify", _notify_shape), (PLAN_ENTRY, "plan entry", _plan_shape))


def clause_dlopen_header(texts: dict) -> list:
    """§7 C4 dlopen shape (r3 m5), the header half: ggml-sycl.h declares the notify with an opaque backend handle
    plus scalars, returning bool, and the plan entry with ggml-base handles only."""
    fails = []
    b = _src(texts, HEADER).bare
    for name, label, shape in _DLOPEN_ENTRIES:
        m = re.search(r"\bGGML_BACKEND_API\s+([\w\s*]+?)\s*\b" + name + r"\s*\(", b)
        if not m:
            fails.append(f"dlopen shape: ggml-sycl.h does not declare {name}")
            continue
        fails += shape(f"ggml-sycl.h's {label}", *_fn_type(b, m))
    return fails


def clause_slot_growth_plan_only(texts: dict) -> list:
    """§M62 (E): a slot grows only under the committed plan: slot_replace is called from host_staging_slots::apply
    alone, so the unplanned admission refuses rather than growing at dispatch."""
    calls = {c[1] for c in _calls(texts, [PLAN], r"slot_replace")}
    if calls != {"host_staging_slots::apply"}:
        return [f"slot growth: slot_replace is called by {sorted(map(str, calls))}; a slot grows only under "
                f"ggml_backend_sycl_plan_host_holders' committed plan (§M62 (E))"]
    get = _body(texts, GS, "ggml_sycl_attn_host_staging_get")
    if re.search(r"\bggml_sycl_attn_host_staging_block_alloc\s*\(", get):
        return ["slot growth: ggml_sycl_attn_host_staging_get allocates a slot block itself"]
    return []


def clause_pools_from_plan(texts: dict) -> list:
    """r6 I-3: the pools' init takes the device's committed plan figures, never a local scan's."""
    body = _body(texts, PLAN, "host_pools_init_from_plan")
    if not body:
        return ["pools from plan: host_pools_init_from_plan is gone"]
    bad = [m.group(0) for m in re.finditer(r"\bd\s*\.\s*dims\s*=(?!=)\s*([^;]+);", body)
           if not re.fullmatch(r"state\s*\.\s*pools", m.group(1).strip())]
    if bad:
        return [f"pools from plan: the pools' init is sized from {bad[0]!r}, not the committed plan (r6 I-3)"]
    return []


def _proc_pointer(text: str, proc: str):
    m = re.search(r"([\w.>-]+)\s*=\s*(?:&\s*" + proc + r"\b|[^;]*\"" + proc + r"\")", text)
    return m.group(1).split(".")[-1].split(">")[-1] if m else None


def _pointer_calls(s: Source, ptr: str, lo: int = 0, hi: int = -1) -> list:
    """Positions of the calls through the function pointer member `ptr` between lo and hi."""
    hi = len(s.bare) if hi < 0 else hi
    return [lo + m.start() for m in re.finditer(r"\b" + re.escape(ptr) + r"\s*\(", s.bare[lo:hi])]


def _lambda_span(bare: str, name: str):
    """(open, close) of the body of the lambda assigned to `name`, or None."""
    m = re.search(r"\b" + name + r"\s*=\s*\[[^\]]*\]\s*\([^)]*\)[^{]*\{", bare)
    return (m.end() - 1, match_close(bare, m.end() - 1)) if m else None


def _block_open(bare: str, pos: int) -> int:
    """The offset of the `{` opening the innermost block that holds `pos`."""
    depth = 0
    for i in range(pos - 1, -1, -1):
        if bare[i] == "}":
            depth += 1
        elif bare[i] == "{":
            if depth == 0:
                return i
            depth -= 1
    return 0


def _in_loop(bare: str, lo: int, pos: int) -> bool:
    """Whether `pos` lies in the body of a for or while loop whose head starts between lo and pos."""
    for m in re.finditer(r"\b(?:for|while)\s*\(", bare[lo:pos]):
        rp = match_close(bare, lo + m.end() - 1)
        k = rp + 1
        while k < len(bare) and bare[k].isspace():
            k += 1
        end = match_close(bare, k) if k < len(bare) and bare[k] == "{" else bare.find(";", k)
        if rp < pos <= end:
            return True
    return False


_DEV0_RE = r"\bmodel\s*\.\s*devices\s*\[\s*0\s*\]"


def _from_dev0(bare: str, lo: int, pos: int, arg: str) -> bool:
    """Whether a call argument is model.devices[0]'s backend: it names model.devices[0], or it is a local whose last
    assignment between lo and the call does."""
    if re.search(_DEV0_RE, arg):
        return True
    if not re.fullmatch(r"\w+", arg):
        return False
    sets = list(re.finditer(r"\b" + arg + r"\s*=(?!=)([^;]*);", bare[lo:pos]))
    return bool(sets) and bool(re.search(_DEV0_RE, sets[-1].group(1)))


def _reserves(s: Source, lo: int, hi: int) -> list:
    """(position, split_only) of each graph_reserve call between lo and hi. A call is split-only when its sixth
    argument is the literal true; a graph_reserve_shift call measures the K-shift graph, so it is split-only too."""
    out = []
    for m in re.finditer(r"\bgraph_reserve(_shift)?\s*\(", s.bare[lo:hi]):
        args = [s.bare[a:b].strip() for a, b in call_args(s.bare, lo + m.end() - 1)]
        out.append((lo + m.start(), bool(m.group(1)) or (len(args) > 5 and args[5] == "true")))
    return out


def _sched_path(texts: dict):
    """The order sched_reserve's transaction runs llama-context.cpp in, as a key per offset: the transaction before
    its MEASURE call (0), the measure (1), the transaction between MEASURE and ALLOC (2), the ALLOC reserve after its
    MEASURE dispatch (3), and the transaction from ALLOC on (4). An offset off the path keys to None. Returns
    {key, measure, reserve, reserve_lo}, or None when a function or a call is missing."""
    s, tr = _fn(texts, LC, "llama_context::sched_reserve_transaction")
    _, me = _fn(texts, LC, "llama_context::sched_measure_impl")
    _, im = _fn(texts, LC, "llama_context::sched_reserve_impl")
    if not (tr and me and im):
        return None
    body = s.bare[tr["open"]:tr["close"]]
    mc = re.search(r"\bsched_reserve_impl\s*\(\s*sched_reserve_mode::MEASURE\b", body)
    ac = re.search(r"\bsched_reserve_impl\s*\(\s*sched_reserve_mode::ALLOC\b", body)
    if not (mc and ac):
        return None
    m_at, a_at = tr["open"] + mc.start(), tr["open"] + ac.start()
    d = re.search(r"\breturn\s+sched_measure_impl\s*\([^;]*;\s*\}", s.bare[im["open"]:im["close"]])
    im_lo = im["open"] + d.end() if d else im["open"]

    def key(p):
        if me["open"] < p < me["close"]:
            return (1, p)
        if im_lo < p < im["close"]:
            return (3, p)
        if tr["open"] < p < tr["close"]:
            return (0, p) if p < m_at else (2, p) if p < a_at else (4, p)
        return None
    return {"key": key, "measure": me, "reserve": im, "reserve_lo": im_lo}


def _trial_flags(s, sr):
    """The auto-n_ubatch trial's function, the one enclosing try_candidate, and its flags: the identifiers it sets
    true, its own or this->'s, minus every identifier sched_reserve (`sr`) assigns. A flag sched_reserve sets
    itself, as it resets sched_need_reserve at its top, holds the same value under every caller, so it cannot tell
    the trial apart."""
    cand = _lambda_span(s.bare, "try_candidate")
    fn = s.enclosing(cand[0]) if cand else None
    body = s.bare[fn["open"]:fn["close"]] if fn else ""
    flags = {m.group(1) for m in re.finditer(r"(?<![\w.>])(?:this\s*->\s*)?([A-Za-z_]\w*)\s*=\s*true\b", body)}
    own = s.bare[sr["open"]:sr["close"]] if sr else ""
    flags -= {m.group(1) for m in re.finditer(r"([A-Za-z_]\w*)\s*(?:=(?!=)|\+=|-=|\+\+|--)", own)}
    flags -= {m.group(1) for m in re.finditer(r"(?:\+\+|--)\s*([A-Za-z_]\w*)", own)}
    return fn, flags


def _trial_excepted(s, fn: dict, flags: set, p: int) -> bool:
    """Whether `p` sits in a block of `fn` guarded by an if whose condition reads, negated, one of the trial's
    `flags`, as a plain identifier or this->'s member, with no ||: the step it guards is then skipped whenever the
    auto-n_ubatch trial runs, which a disjunct could override."""
    for m in re.finditer(r"\bif\s*\(", s.bare[fn["open"]:p]):
        lp = fn["open"] + m.end() - 1
        rp = match_close(s.bare, lp)
        k = rp + 1
        while k < p and s.bare[k].isspace():
            k += 1
        if s.bare[k] != "{" or not k < p < match_close(s.bare, k):
            continue
        cond = s.bare[lp + 1:rp]
        if "||" in cond:
            continue
        negated = set(re.findall(r"!\s*(?:this\s*->\s*)?([A-Za-z_]\w*)\b(?!\s*(?:\(|\.|->|\[))", cond))
        negated |= set(re.findall(r"(?<![\w.>])(?:this\s*->\s*)?([A-Za-z_]\w*)\s*==\s*(?:0|false)\b", cond))
        if negated & flags:
            return True
    return False


def _ok_arm(code: str, fn):
    """(open, end) of `fn`'s success arm, the block of its `if (result.status == sched_reserve_status::OK)`, up to
    the arm's `return true;`, or None."""
    m = re.search(r"\bif\s*\(\s*result\s*\.\s*status\s*==\s*sched_reserve_status\s*::\s*OK\s*\)\s*\{",
                  code[fn["open"]:fn["close"]]) if fn else None
    if not m:
        return None
    a_open = fn["open"] + m.end() - 1
    ret = re.search(r"\breturn\s+true\s*;", code[a_open:match_close(code, a_open)])
    return (a_open, a_open + ret.start()) if ret else None


def _catch_arms(code: str, t_close: int) -> list:
    """(declaration, open, close) of each catch arm that follows the try block closing at `t_close`."""
    arms, k = [], t_close + 1
    while True:
        c = re.match(r"\s*catch\s*\(([^)]*)\)\s*\{", code[k:])
        if not c:
            return arms
        a_open = k + c.end() - 1
        arms.append((c.group(1).strip(), a_open, match_close(code, a_open)))
        k = arms[-1][2] + 1


def _rethrowing_tries(code: str, fn, returns: list) -> list:
    """(open, close) of each try block in `fn` whose catch (...) arm RETURNs in the arm's own block and then ends in
    a rethrow: every throw inside that try block leaves through the arm, so the RETURN runs on its path."""
    out = []
    for m in re.finditer(r"\btry\s*\{", code[fn["open"]:fn["close"]]):
        t_open = fn["open"] + m.end() - 1
        t_close = match_close(code, t_open)
        for decl, a_open, a_close in _catch_arms(code, t_close):
            rethrow = re.search(r"\bthrow\s*;\s*$", code[a_open + 1:a_close]) if decl == "..." else None
            if rethrow and any(a_open < r < a_open + 1 + rethrow.start() and _block_open(code, r) == a_open
                               for r in returns):
                out.append((t_open, t_close))
    return out


def _statement_block(bare: str, lo: int, pos: int) -> int:
    """The `{` of the block that holds the statement at `pos`: its innermost block, or, when `pos` is in loops whose
    heads start after lo, the block holding the outermost of them."""
    heads = []
    for m in re.finditer(r"\b(?:for|while)\s*\(", bare[lo:pos]):
        rp = match_close(bare, lo + m.end() - 1)
        k = rp + 1
        while k < len(bare) and bare[k].isspace():
            k += 1
        end = match_close(bare, k) if k < len(bare) and bare[k] == "{" else bare.find(";", k)
        if rp < pos <= end:
            heads.append(lo + m.start())
    return _block_open(bare, min(heads) if heads else pos)


def _trial_steps(s) -> dict:
    """{op: [positions]} of the calls through llama_context's SCRATCH transaction pointer, by op."""
    trial = _proc_pointer(s.code, "ggml_backend_sycl_host_scratch_trial")
    at = {}
    if trial:
        for m in re.finditer(r"\b" + re.escape(trial) + r"\s*\([^;]*?\b\w*?(RETURN_TO_MARK|OPEN|KEEP|RETURN|MARK)\b",
                             s.bare):
            at.setdefault(m.group(1), []).append(m.start())
    return at


def clause_context_wiring(texts: dict) -> list:
    """r6 I-1, r7 I-1, r7 m-1: llama_context resolves the notify, begin, plan and commit entries; each begin follows
    a scheduler creation before any graph_reserve, and each plan call follows a split_only graph_reserve. The commit
    calls run at the end of sched_reserve on its success for every caller except the auto-n_ubatch trial, so that
    commit sits under an if whose condition, with no ||, reads negated a flag the trial sets true and sched_reserve
    never assigns; the trial's try_candidate commits instead, after its host-fallback check (design :4571-4574,
    :1932-1943). sched_reserve_nothrow, decode's and encode's reserve, commits in its success arm before its return
    true, under the same exception (§M75)."""
    lc = texts.get(LC, "")
    if not lc:
        return ["context wiring: src/llama-context.cpp is missing"]
    s = source_for(LC, lc)
    fails = []
    ptr = {}
    for name in ("notify_host_buffer_sizes", "begin_host_holders", "plan_host_holders", "commit_host_holders"):
        ptr[name] = _proc_pointer(s.code, "ggml_backend_sycl_" + name)
        if not ptr[name]:
            fails.append(f"context wiring: llama_context does not resolve ggml_backend_sycl_{name}")
    if fails:
        return fails
    for f in s.functions:
        body = s.bare[f["open"]:f["close"]]
        for m in re.finditer(r"\bggml_backend_sched_new\s*\(", body):
            nxt_reserve = re.search(r"\bgraph_reserve\s*\(", body[m.end():])
            nxt_begin = re.search(r"\b" + re.escape(ptr["begin_host_holders"]) + r"\s*\(", body[m.end():])
            if not nxt_begin or (nxt_reserve and nxt_reserve.start() < nxt_begin.start()):
                fails.append(f"context wiring: {f['name']} creates a scheduler and reserves before the begin entry")
        for m in re.finditer(r"\b" + re.escape(ptr["plan_host_holders"]) + r"\s*\(", body):
            prev = list(re.finditer(r"\bgraph_reserve\s*\(", body[:m.start()]))
            if not prev or not re.search(r"\btrue\b", body[prev[-1].start():body.find(";", prev[-1].start())]):
                fails.append(f"context wiring: {f['name']} calls the plan entry not right after a split_only "
                             f"graph_reserve")
    _, sr = _fn(texts, LC, "llama_context::sched_reserve")
    throws = [sr["open"] + m.start() for m in re.finditer(r"\bthrow\b", s.bare[sr["open"]:sr["close"]])] if sr else []
    cand = _lambda_span(s.bare, "try_candidate")
    fallback = re.search(r"\breturn\s+\"compute buffer fell back to host\"\s*;", s.code[cand[0]:cand[1]]) \
        if cand else None
    trial_fn, trial_flags = _trial_flags(s, sr)
    _, nt = _fn(texts, LC, "llama_context::sched_reserve_nothrow")
    ok_arm = _ok_arm(s.bare, nt)
    at_end = at_trial = at_nothrow = 0
    for p in _pointer_calls(s, ptr["commit_host_holders"]):
        if ok_arm and ok_arm[0] < p < ok_arm[1]:
            at_nothrow += 1
            if not _trial_excepted(s, nt, _trial_flags(s, nt)[1], p):
                fails.append(f"context wiring: sched_reserve_nothrow's success commit at line {s.line(p)} is not "
                             f"under an if that excepts the auto-n_ubatch trial (a negated read, with no ||, of a "
                             f"flag {trial_fn['name'] if trial_fn else 'the trial'} sets true and the nothrow form "
                             f"never assigns)")
        elif sr and sr["open"] < p < sr["close"] and all(t < p for t in throws):
            at_end += 1
            if not _trial_excepted(s, sr, trial_flags, p):
                fails.append(f"context wiring: sched_reserve's end commit at line {s.line(p)} is not under an if "
                             f"that excepts the auto-n_ubatch trial (a negated read, with no ||, of a flag "
                             f"{trial_fn['name'] if trial_fn else 'the trial'} sets true and sched_reserve never "
                             f"assigns)")
        elif cand and fallback and cand[0] + fallback.end() < p < cand[1]:
            at_trial += 1
        else:
            f = s.enclosing(p)
            fails.append(f"context wiring: {f['name'] if f else 'namespace scope'} calls the commit entry at line "
                         f"{s.line(p)}, outside sched_reserve's success end, sched_reserve_nothrow's success arm and "
                         f"try_candidate's point after its host-fallback check")
    if not at_end:
        fails.append("context wiring: sched_reserve does not call the commit entry at its end, on its success path")
    if not at_trial:
        fails.append("context wiring: try_candidate does not call the commit entry after its host-fallback check")
    if not at_nothrow:
        fails.append("context wiring: sched_reserve_nothrow does not call the commit entry in its success arm, before "
                     "its return true")
    return fails


_DEV0_WARN = "llama_context_sycl_warn_dev0"


def clause_dev0_warn(texts: dict) -> list:
    """The host SCRATCH entries resolve from model.devices[0] alone (§4.4), so when it is not SYCL a SYCL device later
    in the list gets no notify, holder plan or SCRATCH transaction from the context. The context says so once, where
    it resolves them: the constructor calls the warning right after llama_context_sycl_host_procs and nowhere else,
    and the warning names the device list, fires only when devices[0] is not SYCL and some device is, and touches no
    hook, so nothing but the log changes."""
    lc = texts.get(LC, "")
    if not lc:
        return ["dev0 warning: src/llama-context.cpp is missing"]
    s = source_for(LC, lc)
    fails = []
    flat = re.sub(r"\s+", " ", s.code)
    if not re.search(r"\bsycl_host_hooks = llama_context_sycl_host_procs\([^;]*\); " + _DEV0_WARN +
                     r"\(model\.devices\);", flat):
        fails.append(f"dev0 warning: the constructor does not call {_DEV0_WARN}(model.devices) right after it resolves "
                     "the host SCRATCH entries")
    calls = len(re.findall(r"\b" + _DEV0_WARN + r"\s*\(", s.bare)) - 1
    if calls > 1:
        fails.append(f"dev0 warning: {_DEV0_WARN} is called {calls} times, not once per context")
    _, f = _fn(texts, LC, _DEV0_WARN)
    if not f:
        return fails + [f"dev0 warning: no single definition of {_DEV0_WARN}"]
    body = re.sub(r"\s+", " ", s.code[f["open"]:f["close"] + 1])
    head = re.sub(r"\s+", " ", s.code[max(0, f["open"] - 200):f["open"]])
    if not re.search(r"static void " + _DEV0_WARN + r"\(const std::vector<llama_device> & devices\) $", head):
        fails.append(f"dev0 warning: {_DEV0_WARN} is not a static void function of the device list alone")
    if not re.search(r"if \(devices\.empty\(\) \|\| llama_context_dev_is_sycl\(devices\[0\]\.dev\)\) \{ return; \}",
                     body):
        fails.append("dev0 warning: the warning does not return when devices[0] is SYCL")
    if not re.search(r"for \(const llama_device & d : devices\) \{[^}]*llama_context_dev_is_sycl\(d\.dev\)", body):
        fails.append("dev0 warning: the warning does not ask whether any device in the list is SYCL")
    if not re.search(r"for \(const llama_device & d : devices\) \{[^}]*ggml_backend_dev_name\(d\.dev\)", body) or \
            not re.search(r"LLAMA_LOG_WARN\([^;]*names\.c_str\(\)\);", body):
        fails.append("dev0 warning: the warning does not name the device list")
    if re.search(r"\bhooks\b|=\s*nullptr|\bdevices\s*\.\s*(?:push_back|erase|clear|insert|emplace)", body):
        fails.append("dev0 warning: the warning touches a hook or the device list, so more than the log changes")
    return fails


def clause_trial_ops(texts: dict) -> list:
    """§M75, r9 n-6, r10 m-5: try_candidate OPENs before its reserve, KEEPs beside the commit and RETURNs on every
    loss path (the reserve's throw, the host fallback and the hold spill); sched_reserve RETURNs on its transaction's
    throw path, MARKs before the measure, RETURN_TO_MARKs before the pipeline-parallel retry, and KEEPs on its success
    path except under the auto-n_ubatch trial: like the end commit, that KEEP sits under an if whose condition, with
    no ||, reads negated a flag the trial sets true and sched_reserve never assigns, because inside try_candidate it
    would run before the host-fallback check and keep a losing candidate's growth. A throw in a try block whose
    catch (...) arm RETURNs in its own block and rethrows is on that RETURN's path. sched_reserve_nothrow KEEPs beside
    its success commit under the same exception, and RETURNs on every way out but OK, after its catch arms."""
    lc = texts.get(LC, "")
    if not lc:
        return ["trial ops: src/llama-context.cpp is missing"]
    s = source_for(LC, lc)
    trial = _proc_pointer(s.code, "ggml_backend_sycl_host_scratch_trial")
    if not trial:
        return ["trial ops: llama_context does not resolve ggml_backend_sycl_host_scratch_trial"]
    code = s.bare
    at = _trial_steps(s)
    fails = [f"trial ops: llama_context never calls the trial entry with {op}"
             for op in ("OPEN", "KEEP", "RETURN", "MARK", "RETURN_TO_MARK") if op not in at]
    every = sorted(p for ps in at.values() for p in ps)
    commit = _proc_pointer(s.code, "ggml_backend_sycl_commit_host_holders")

    def ops(op, lo, hi):
        return [p for p in at.get(op, []) if lo < p < hi]

    def beside(k, c):
        a, b = min(k, c), max(k, c)
        return not re.search(r"\breturn\b", code[a:b]) and not any(a < q < b for q in every if q != k)

    cand = _lambda_span(code, "try_candidate")
    if not cand:
        fails.append("trial ops: no try_candidate in llama_context")
    else:
        lo, hi = cand
        r = re.search(r"\bsched_reserve\w*\s*\(", code[lo:hi])
        opens = ops("OPEN", lo, hi)
        if not opens or (r and lo + r.start() < opens[0]):
            fails.append("trial ops: try_candidate does not OPEN before its reserve")
        for label, text in (("reserve-failure", "compute buffers did not fit"),
                            ("host-fallback", "compute buffer fell back to host"),
                            ("hold-spill", "hold spill left no headroom")):
            m = re.search(r"\breturn\s+\"" + text + r"\"\s*;", s.code[lo:hi])
            if not m:
                fails.append(f"trial ops: try_candidate has no {label} return")
                continue
            ret = lo + m.start()
            returns = ops("RETURN", _block_open(code, ret), ret)
            if not returns or re.search(r"\breturn\b", code[returns[-1]:ret]):
                fails.append(f"trial ops: try_candidate does not RETURN on every loss path: none before its {label} "
                             f"return")
        commits = _pointer_calls(s, commit, lo, hi) if commit else []
        keeps = ops("KEEP", lo, hi)
        if not keeps or not all(any(beside(k, c) for c in commits) for k in keeps):
            fails.append("trial ops: try_candidate does not KEEP beside the commit, with no exit or trial step "
                         "between them (§M75)")
    _, sr = _fn(texts, LC, "llama_context::sched_reserve")
    if not sr:
        fails.append("trial ops: no llama_context::sched_reserve")
    else:
        tx = re.search(r"\bsched_reserve_transaction\s*\(", code[sr["open"]:sr["close"]])
        throws = [sr["open"] + m.start() for m in re.finditer(r"\bthrow\b", code[sr["open"]:sr["close"]])]
        if not (tx and throws):
            fails.append("trial ops: sched_reserve does not RETURN on its transaction's throw path: it has no "
                         "transaction call or no throw (r10 m-5)")
        rethrowing = _rethrowing_tries(code, sr, at.get("RETURN", []))
        for t_at in throws if tx else []:
            if any(a < t_at < b for a, b in rethrowing):
                continue
            chain = set()  # the blocks inside sched_reserve that enclose this throw
            b = _block_open(code, t_at)
            while b > sr["open"]:
                chain.add(b)
                b = _block_open(code, b)
            if not [r for r in ops("RETURN", sr["open"] + tx.start(), t_at) if _block_open(code, r) in chain]:
                name = re.match(r"throw\s+([\w:]+)", code[t_at:])
                fails.append(f"trial ops: sched_reserve does not RETURN on its transaction's throw path before its "
                             f"{repr('throw ' + name.group(1)) if name else 'rethrow'}, in a block enclosing that "
                             f"throw (r10 m-5)")
        success_keeps = ops("KEEP", throws[-1], sr["close"]) if throws else []
        if not success_keeps:
            fails.append("trial ops: sched_reserve does not KEEP on its success path, after its throws (r10 m-5)")
        trial_fn, trial_flags = _trial_flags(s, sr)
        for k in success_keeps:
            if not _trial_excepted(s, sr, trial_flags, k):
                fails.append(f"trial ops: sched_reserve's success KEEP at line {s.line(k)} is not under an if that "
                             f"excepts the auto-n_ubatch trial (a negated read, with no ||, of a flag "
                             f"{trial_fn['name'] if trial_fn else 'the trial'} sets true and sched_reserve never "
                             f"assigns): inside try_candidate it would keep a losing candidate's growth before the "
                             f"host-fallback check (r10 m-5, §M75)")
    _, nt = _fn(texts, LC, "llama_context::sched_reserve_nothrow")
    if not nt:
        fails.append("trial ops: no llama_context::sched_reserve_nothrow")
    else:
        arm = _ok_arm(code, nt)
        commits = _pointer_calls(s, commit, *arm) if arm and commit else []
        keeps = ops("KEEP", *arm) if arm else []
        if not keeps or not all(any(beside(k, c) for c in commits) for k in keeps):
            fails.append("trial ops: sched_reserve_nothrow does not KEEP beside its success commit, before its return "
                         "true, with no exit or trial step between them (§M75)")
        nt_fn, nt_flags = _trial_flags(s, nt)
        for k in keeps:
            if not _trial_excepted(s, nt, nt_flags, k):
                fails.append(f"trial ops: sched_reserve_nothrow's success KEEP at line {s.line(k)} is not under an if "
                             f"that excepts the auto-n_ubatch trial (a negated read, with no ||, of a flag "
                             f"{nt_fn['name'] if nt_fn else 'the trial'} sets true and the nothrow form never "
                             f"assigns) (§M75)")
        tr = re.search(r"\btry\s*\{", code[nt["open"]:nt["close"]])
        arms = _catch_arms(code, match_close(code, nt["open"] + tr.end() - 1)) if tr else []
        fin = [nt["open"] + m.start() for m in re.finditer(r"\breturn\s+false\s*;", code[nt["open"]:nt["close"]])]
        rets = [r for r in ops("RETURN", arms[-1][2], fin[-1]) if _block_open(code, r) == nt["open"]] \
            if arms and fin else []
        if not rets or re.search(r"\breturn\b", code[rets[-1]:fin[-1]]):
            fails.append("trial ops: sched_reserve_nothrow does not RETURN on every way out but OK: no RETURN in its "
                         "body after its catch arms and before its return false (§M75)")
    path = _sched_path(texts)
    im = path["reserve"] if path else None
    lo2 = path["reserve_lo"] if path else 0
    retry = re.search(r"\bpipeline_parallel\s*=\s*false\s*;[^;]*?\bggml_backend_sched_new\s*\(",
                      code[lo2:im["close"]]) if im else None
    failed = [p for p, _ in _reserves(s, lo2, lo2 + retry.start())] if retry else []
    if not failed:
        fails.append("trial ops: sched_reserve_impl has no reserve followed by a pipeline-parallel retry")
    elif "RETURN_TO_MARK" in at:
        key, f_at, retry_new = path["key"], failed[-1], lo2 + retry.end() - 1
        if not ops("RETURN_TO_MARK", f_at, retry_new):
            fails.append("trial ops: sched_reserve does not RETURN_TO_MARK between the failed reserve and the "
                         "pipeline-parallel retry's scheduler (r9 n-6)")
        allocating = [key(p) for p, so in _reserves(s, lo2, im["close"]) if not so]
        marked = [q for q in at.get("MARK", []) if key(q) and key(q) < key(f_at)
                  and not any(key(q) < a < key(f_at) for a in allocating)]
        if not marked:
            fails.append("trial ops: a RETURN_TO_MARK is not preceded by its MARK before the reserve the "
                         "pipeline-parallel retry repeats (r9 n-6)")
    return fails


def clause_notify_position(texts: dict) -> list:
    """§7 C4, r3 A.2 and A.5: llama_context calls the notify before the allocation it covers, once with
    model.devices[0]'s backend and not in a loop over backends. In output_reserve the call sits between the old
    buffer's free and ggml_backend_buft_alloc_buffer; on sched_reserve's path it follows the measure's last size-only
    graph_reserve and precedes the first allocating one. Each commit is followed by the re-notify with the allocated
    size (§4.4 "The same fact for the notify's measure"): a notify after the commit call in the block that holds the
    commit's statement, with no exit and no trial step between them, so the KEEP comes after it."""
    lc = texts.get(LC, "")
    if not lc:
        return ["notify position: src/llama-context.cpp is missing"]
    s = source_for(LC, lc)
    ptr = _proc_pointer(s.code, NOTIFY)
    if not ptr:
        return [f"notify position: llama_context does not resolve {NOTIFY}"]
    _, out = _fn(texts, LC, "llama_context::output_reserve")
    path = _sched_path(texts)
    if not out or not path:
        return ["notify position: output_reserve or sched_reserve's transaction is missing"]
    key, me, im = path["key"], path["measure"], path["reserve"]
    span = s.bare[out["open"]:out["close"]]
    frees = [out["open"] + m.end() for m in re.finditer(r"\bbuf_output\s*=\s*nullptr\s*;", span)]
    alloc = re.search(r"\bggml_backend_buft_alloc_buffer\s*\(", span)
    size_only = [key(p) for p, so in _reserves(s, me["open"], me["close"]) if so]
    allocating = [key(p) for p, so in _reserves(s, path["reserve_lo"], im["close"]) if not so]
    commit = _proc_pointer(s.code, "ggml_backend_sycl_commit_host_holders")
    commits = _pointer_calls(s, commit) if commit else []
    steps = sorted(q for ps in _trial_steps(s).values() for q in ps)

    def renotifies(c, p):
        f = s.enclosing(c)
        if not f or not c < p < f["close"]:
            return False
        blk = _statement_block(s.bare, f["open"], c)
        return blk < p < match_close(s.bare, blk) and not re.search(r"\breturn\b", s.bare[c:p]) and \
            not any(c < q < p for q in steps)
    fails = []
    in_out = in_path = 0
    notifies = _pointer_calls(s, ptr)
    for c in commits:
        if not any(renotifies(c, p) for p in notifies):
            f = s.enclosing(c)
            fails.append(f"notify position: the commit at {f['name'] if f else 'namespace scope'}:{s.line(c)} has no "
                         f"re-notify after it in its block (§4.4 \"The same fact for the notify's measure\")")
    for p in notifies:
        f = s.enclosing(p)
        lo = f["open"] if f else 0
        where = f"{f['name'] if f else 'namespace scope'}:{s.line(p)}"
        args = [s.bare[a:b].strip() for a, b in call_args(s.bare, s.bare.index("(", p))]
        if _in_loop(s.bare, lo, p):
            fails.append(f"notify position: {where} calls the notify in a loop, not once with one backend (r3 A.5)")
        if not args or not _from_dev0(s.bare, lo, p, args[0]):
            fails.append(f"notify position: {where} does not pass model.devices[0]'s backend (r3 A.5)")
        if out["open"] < p < out["close"]:
            in_out += 1
            if not (frees and alloc and frees[-1] < p < out["open"] + alloc.start()):
                fails.append(f"notify position: {where} calls the notify outside the span between the old buffer's "
                             f"free and ggml_backend_buft_alloc_buffer (r3 A.2)")
        elif key(p):
            in_path += 1
            if not (size_only and allocating and size_only[-1] < key(p) < allocating[0]):
                fails.append(f"notify position: {where} calls the notify on sched_reserve's path outside the span "
                             f"from the measure's last size-only graph_reserve to the first allocating one (r3 A.2)")
        elif not any(renotifies(c, p) for c in commits):
            fails.append(f"notify position: {where} calls the notify outside output_reserve, sched_reserve's path "
                         f"and a commit's re-notify")
    if not in_out:
        fails.append("notify position: output_reserve does not call the notify")
    if not in_path:
        fails.append("notify position: sched_reserve's path does not call the notify")
    return fails


def _pointer_type(texts: dict, member: str):
    """How llama_context declares the function pointer `member`: ("decltype", function) for decltype(&function), or
    (return type, [parameters]) for a function pointer type written out, directly or through a using or typedef
    alias. None when no declaration is found."""
    for rel in (LC, LCH):
        if not texts.get(rel):
            continue
        b = _src(texts, rel).bare
        m = re.search(r"\bdecltype\s*\(\s*&\s*(\w+)\s*\)\s+" + member + r"\b", b)
        if m:
            return "decltype", m.group(1)
        m = re.search(r"([\w:]+(?:\s*\*)?)\s*\(\s*\*\s*" + member + r"\s*\)\s*\(", b)
        if m:
            return _fn_type(b, m)
        for a in re.finditer(r"\b(\w+)\s+" + member + r"\s*(?:=|;|\{)", b):
            alias = (re.search(r"\busing\s+" + a.group(1) + r"\s*=\s*([\w:]+(?:\s*\*)?)\s*\(\s*\*\s*\)\s*\(", b)
                     or re.search(r"\btypedef\s+([\w:]+(?:\s*\*)?)\s*\(\s*\*\s*" + a.group(1) + r"\s*\)\s*\(", b))
            if alias:
                return _fn_type(b, alias)
    return None


def clause_dlopen_context(texts: dict) -> list:
    """§7 C4 dlopen shape (r3 m5), the llama_context half: the function pointer types llama_context resolves the
    notify and the plan entry into have the header's shape; a decltype of the header's function inherits it."""
    lc = texts.get(LC, "")
    if not lc:
        return ["dlopen shape: src/llama-context.cpp is missing"]
    code = source_for(LC, lc).code
    fails = []
    for name, label, shape in _DLOPEN_ENTRIES:
        ptr = _proc_pointer(code, name)
        if not ptr:
            fails.append(f"dlopen shape: llama_context does not resolve {name}")
            continue
        declared = _pointer_type(texts, ptr)
        if declared is None:
            fails.append(f"dlopen shape: llama_context declares no type for its {ptr} pointer")
        elif declared[0] == "decltype":
            if declared[1] != name:
                fails.append(f"dlopen shape: llama_context's {ptr} pointer is decltype(&{declared[1]}), not {name}'s")
        else:
            fails += shape(f"llama_context's {label} pointer type", *declared)
    return fails


def clause_owner_end_wired(texts: dict) -> list:
    """§M75, r9 m-5, r10 m-5: ggml_backend_sycl_free calls the owner's end before delete sycl_ctx."""
    free = _body(texts, GS, "ggml_backend_sycl_free")
    if not free:
        return ["owner end: ggml_backend_sycl_free is gone"]
    d = free.find("delete sycl_ctx")
    m = re.search(r"\b\w*host_scratch_owner_end\s*\(", free)
    if not m:
        return ["owner end: ggml_backend_sycl_free does not call the owner's end (host_scratch_owner_end)"]
    if d < 0 or m.start() > d:
        return ["owner end: ggml_backend_sycl_free calls the owner's end after delete sycl_ctx"]
    return []


def clause_open_touches_no_set(texts: dict) -> list:
    """r8 m-1: nothing in OPEN touches the marked set."""
    found = False
    fails = []
    for rel in _production(texts):
        if "host_scratch_trial" not in texts[rel]:
            continue
        s = _src(texts, rel)
        for f in s.functions:
            if "host_scratch_trial" not in f["name"]:
                continue
            body = s.bare[f["open"]:f["close"]]
            parts = re.split(r"\bcase\s+[\w:]*\b", body)
            labels = re.findall(r"\bcase\s+([\w:]*)\b", body)
            branches = [p for lab, p in zip(labels, parts[1:]) if lab.endswith("OPEN")]
            if f["name"].endswith("_open"):
                branches.append(body)
            for b in branches:
                found = True
                if re.search(r"\bmarked_set_\b|host_scratch_return_chunks|release_marked_chunk", b):
                    fails.append(f"OPEN: {f['name']} touches the marked set in its OPEN step")
    if not found:
        fails.append("OPEN: no host_scratch_trial OPEN step in the tree")
    return fails


def clause_scratch_fold(texts: dict) -> list:
    """§M62 (A): host SCRATCH folds exactly max_copied_tensor_bytes, k_scratch_headroom,
    moe_cpu_expert_staging_bytes and moe_host_recipe_workspace_bytes."""
    terms = _scratch_fold_terms(texts)
    if terms is None:
        return ["SCRATCH fold: the host SCRATCH zone sum is not found"]
    want = set(SINGLE_TENSOR) | set(MOE_STAGING)
    if terms != want:
        extra, missing = sorted(terms - want), sorted(want - terms)
        parts = ([f"reads {extra} beyond the four host terms"] if extra else []) + \
            ([f"lacks {missing}"] if missing else [])
        return ["SCRATCH fold: the host SCRATCH sum " + " and ".join(parts) + " (§M62 (A))"]
    return []


def clause_runtime_term(texts: dict) -> list:
    """r3 m3: the host-zone runtime term names neither PP term."""
    m = re.search(r"\bterms\s*\.\s*runtime\s*=\s*([^;]+);", _src(texts, UCH).bare)
    if not m:
        return ["runtime term: placement_plan_host_zone_commit_terms sets no runtime term"]
    named = [t for t in ("pp_pipeline_scratch_bytes", "pp_moe_onednn_scratch_bytes")
             if re.search(r"\b" + t, m.group(1))]
    if named:
        return [f"runtime term: the host-zone runtime term names {named} (r3 m3)"]
    return []


REQUIRED, EXPECTED_RED = "REQUIRED", "EXPECTED-RED"

# (id, status, function, landing task, the reason an expected-red clause fails with on this tree). The reason
# names the transitional code its landing task removes, so a clause red for any other cause does not score as
# expected-red.
CLAUSES = [
    ("site-table", REQUIRED, clause_site_table, None, None),
    ("doc-table", REQUIRED, clause_doc_table, None, None),
    ("select-zone", REQUIRED, clause_select_zone, None, None),
    ("owners", REQUIRED, clause_owners, None, None),
    ("unowned-ratchet", REQUIRED, clause_unowned_ratchet, None, None),
    ("unsized-ratchet", REQUIRED, clause_unsized_ratchet, None, None),
    ("device-holders", REQUIRED, clause_device_holders, None, None),
    ("slot-refusal", REQUIRED, clause_slot_refusal, None, None),
    ("staging-plan", REQUIRED, clause_staging_plan, None, None),
    ("pool-dims", REQUIRED, clause_pool_dims, None, None),
    ("marked-set-reach", REQUIRED, clause_marked_set_reach, None, None),
    ("return-files", REQUIRED, clause_return_files, None, None),
    ("withdrawal", REQUIRED, clause_withdrawal, None, None),
    ("notify-device", REQUIRED, clause_notify_device, None, None),
    ("grow-host-scratch-gone", REQUIRED, clause_grow_host_scratch_gone, None, None),
    ("notify-compute-unused", REQUIRED, clause_notify_compute_unused, None, None),
    ("dlopen-header", REQUIRED, clause_dlopen_header, None, None),
    ("open-touches-no-set", REQUIRED, clause_open_touches_no_set, None, None),
    ("owner-end-wired", REQUIRED, clause_owner_end_wired, None, None),
    ("slot-growth-plan-only", REQUIRED, clause_slot_growth_plan_only, None, None),
    ("pools-from-plan", REQUIRED, clause_pools_from_plan, None, None),
    ("context-wiring", REQUIRED, clause_context_wiring, None, None),
    ("dev0-warn", REQUIRED, clause_dev0_warn, None, None),
    ("trial-ops", REQUIRED, clause_trial_ops, None, None),
    ("notify-position", REQUIRED, clause_notify_position, None, None),
    ("dlopen-context", REQUIRED, clause_dlopen_context, None, None),
    ("scratch-fold", REQUIRED, clause_scratch_fold, None, None),
    ("runtime-term", REQUIRED, clause_runtime_term, None, None),
]


# ---------------------------------------------------------------------------------------------------------------
# Controls. An edit is (file, old text, new text): the old text must occur exactly once in that file, an old text
# of None appends, and (file, None, None) removes the file, which then reads as a missing file reads. A fixture is
# the landed shape an EXPECTED-RED clause passes on. A mutant must make its clause fail with its quoted reason; an
# EXPECTED-RED clause's mutants are planted on its fixture, so each fails for its own reason and not for the
# transitional code the fixture removes.
# ---------------------------------------------------------------------------------------------------------------

PBP = SYCL_DIR + "/pinned-buffer-pool.cpp"
CD = SYCL_DIR + "/cpu-dispatch.cpp"

_PLANTED_SITE = (CD, None, "\nstatic void routing_gate_planted_site(ggml_sycl::alloc_request & out) {\n"
                           "    ggml_sycl::alloc_request req{};\n"
                           "    req.intent.category                     = ggml_sycl::runtime_category::HOST_COMPUTE;\n"
                           "    req.intent.role                         = ggml_sycl::alloc_role::STAGING;\n"
                           "    req.intent.lifetime                     = ggml_sycl::alloc_lifetime::GRAPH;\n"
                           "    req.intent.constraints.must_host_pinned = true;\n"
                           "    out                                     = req;\n"
                           "}\n")
_PBP_DEVICE = "    req_act.intent.lifetime                     = alloc_lifetime::DEVICE;\n"
# The planned branch's fit test: the planned and unplanned branches of admit share one refusal after them.
_PLANNED_FIT = "        if (bytes <= slot.cap) {\n            return slot.ptr;\n        }\n"
_PLANNED_GROW = (PLAN, _PLANNED_FIT,
                 _PLANNED_FIT + "        if (slot_replace(slot, device, kind, host_staging_slot_bytes(bytes), "
                                "alloc)) {\n            return slot.ptr;\n        }\n")
_IDLE = "    if (idle_slots && ggml_sycl_attn_host_staging_enabled(kind)) {\n"
_INIT_SCAN = "        ggml_sycl::host_pool_scan(cgraph, ggml_sycl::host_pool_has_storage, nullptr);\n"
_FOLD_FOUR = "{ plan.max_copied_tensor_bytes, k_scratch_headroom, moe_host_scratch }"
_RUNTIME_FOUR = "    terms.runtime = plan.onednn_scratchpad_bytes + plan.dma_staging_pool_bytes;\n"
_HOOKS_AT = "        ggml_backend_reg_get_proc_address(reg, \"ggml_backend_sycl_has_active_placement_plan\"));\n"
_OWNER_END = ("    if (owner_end) {\n"
              "        (void) ggml_sycl::unified_cache_host_scratch_owner_end(owner_end_device, freed_owner_id);\n"
              "    }\n")
_ARENA_RESERVED = ("        GGML_ABORT(\"[SYCL-PLAN] compute arena reserve smaller than scheduler requirement\");\n"
                   "    }\n")
_HDR_ARG = " " * 65  # the column of the notify's parameters in ggml-sycl.h


def _hook(name: str) -> str:
    """llama_context's resolve of one host SCRATCH entry, by proc name."""
    return (f"    hooks.{name} = reinterpret_cast<decltype(hooks.{name})>(\n"
            f"        ggml_backend_reg_get_proc_address(reg, \"ggml_backend_sycl_{name}\"));\n")


# The tree's own text at each site a llama_context mutant edits: each old text must occur exactly once.
_STEP = "sycl_host_hooks.host_scratch_trial(sycl_dev0, GGML_SYCL_HOST_SCRATCH_TRIAL_"
_OPEN = "        " + _STEP + "OPEN, c);\n"
_FLAG = "        sycl_ubatch_trial_active = true;\n"
_DEV0_CALL = "        llama_context_sycl_warn_dev0(model.devices);\n"
_DEV0_GUARD = "    if (devices.empty() || llama_context_dev_is_sycl(devices[0].dev)) {\n"
_DEV0_NAME = "        names += std::string(names.empty() ? \"\" : \", \") + ggml_backend_dev_name(d.dev);\n"
_RESERVED_OK = "        sched_matches_last_good = true;\n"
# Each loss returns its growth first; the line before that RETURN anchors it, so a mutant drops the RETURN alone.
_LOSS_FIT_AT = ("            LLAMA_LOG_INFO(\"[SYCL-PLAN] auto n_ubatch candidate %u: compute buffer reserve failed: "
                "%s\\n\", c, e.what());\n")
_LOSS_HOST_AT = "            if (fallback_fn(sb.dev_index) > 0) {\n"
_LOSS_SPILL_AT = "                    c, rung_largest_ub);\n"
_LOSS_FIT = _LOSS_FIT_AT + "            " + _STEP + "RETURN, c);\n"
_LOSS_HOST = _LOSS_HOST_AT + "                " + _STEP + "RETURN, c);\n"
_LOSS_SPILL = _LOSS_SPILL_AT + "                " + _STEP + "RETURN, c);\n"
_CAND_COMMIT = ("        for (auto & sb : sycl_backends) {\n"
                "            if (sycl_host_hooks.commit_host_holders) {\n"
                "                sycl_host_hooks.commit_host_holders(sb.backend);\n"
                "            }\n"
                "        }\n")
_RENOTIFY = ("        if (!measure_only && sycl_dev0 && sycl_host_hooks.notify_host_buffer_sizes && "
             "!model.hparams.no_alloc) {\n"
             "            sycl_host_compute_bytes =\n"
             "                llama_context_sycl_host_compute_allocated(model, sched.get(), backend_ptrs, "
             "backend_buft);\n"
             "            llama_context_sycl_notify_checked(\n"
             "                sycl_host_hooks.notify_host_buffer_sizes(sycl_dev0, sycl_host_compute_bytes, "
             "sycl_host_output_bytes),\n"
             "                sycl_host_compute_bytes, sycl_host_output_bytes);\n"
             "        }\n")
_CAND_KEEP = "        " + _STEP + "KEEP, c);\n"
_SR_CATCH = "        " + _STEP + "RETURN, cparams.n_ubatch);\n        throw;\n"
_SR_TRY = ("    try {\n"
           "        const sched_reserve_result result = sched_reserve_transaction();\n"
           "        if (result.status != sched_reserve_status::OK) {\n")
_SR_FIT_REFUSAL = ("            if (result.fit_refusal) {\n"
                   "                throw llama_auto_ubatch_fit_refusal(result.reason);\n")
_SR_COMMIT = ("    if (!sycl_ubatch_trial_active) {\n"
              "        for (auto & backend : backends) {\n"
              "            if (sycl_host_hooks.commit_host_holders && llama_context_backend_is_sycl(backend.get())) {\n"
              "                sycl_host_hooks.commit_host_holders(backend.get());\n"
              "            }\n"
              "        }\n")
_SR_KEEP = "        " + _STEP + "KEEP, cparams.n_ubatch);\n"
_SR_CLOSE = "}\n\nbool llama_context::sched_reserve_nothrow() {\n"
_SR_END = "    }\n" + _SR_CLOSE
_NT_COMMIT = ("                for (auto & backend : backends) {\n"
              "                    if (sycl_host_hooks.commit_host_holders && "
              "llama_context_backend_is_sycl(backend.get())) {\n"
              "                        sycl_host_hooks.commit_host_holders(backend.get());\n"
              "                    }\n"
              "                }\n")
_NT_GUARD = "            if (!sycl_ubatch_trial_active) {\n"
_NT_KEEP_END = ("                " + _STEP + "KEEP, cparams.n_ubatch);\n"
                "            }\n"
                "            return true;\n")
_NT_RETURN = "    " + _STEP + "RETURN, cparams.n_ubatch);\n    sched_need_reserve = true;\n"
_MARK = "        " + _STEP + "MARK, state.cparams.n_ubatch);\n"
_RTM = ("                " + _STEP + "RETURN_TO_MARK,\n"
        "                                                   state.cparams.n_ubatch);\n")
_PP_SPLITS = "        n_splits_pp        = ggml_backend_sched_get_n_splits(state.sched.get());\n"
_RETRY_RESERVE = "                gf = graph_reserve(state, n_tokens, n_seqs, n_outputs_pp, mctx.get());\n"
_RETRY_BEGIN = ("                if (sycl_host_hooks.begin_host_holders) {\n"
                "                    sycl_host_hooks.begin_host_holders(state.sched.get());\n"
                "                }\n")
_BEGIN = ("    if (sycl_host_hooks.begin_host_holders) {\n"
          "        sycl_host_hooks.begin_host_holders(state.sched.get());\n"
          "    }\n")
_BEGIN_0_NOTE = "    // scheduler's compute buffers, and the walks of this reserve fill the pending plans afresh.\n"
_SCHED_NEW_0 = ("    state.sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), "
                "backend_ptrs.size(), max_nodes,\n"
                "                                             state.cparams.pipeline_parallel, "
                "state.cparams.op_offload));\n")
_MEASURE_CALL = "    sycl_host_measure();\n\n"
_MEASURE_NOTIFY = ("        if (!measure_only) {  // a measure-only context resolves no hooks and prints nothing a "
                   "vehicle arm scores\n"
                   "            llama_context_sycl_notify_checked(\n"
                   "                sycl_host_hooks.notify_host_buffer_sizes(sycl_dev0, sycl_host_compute_bytes, "
                   "sycl_host_output_bytes),\n"
                   "                sycl_host_compute_bytes, sycl_host_output_bytes);\n"
                   "        }\n"
                   "    };\n")
_IMPL_DEV0 = ("    ggml_backend_t sycl_dev0 =\n"
              "        model.devices.empty() ? nullptr : llama_context_sycl_backend_of(backends, "
              "model.devices[0].dev);\n"
              "    bool   sycl_host_measured = false;\n")
_OUT_BLOCK = ("        if (sycl_host_hooks.notify_host_buffer_sizes && llama_context_dev_is_sycl(output_dev) &&\n"
              "            buft == output_dev_host_buft) {\n"
              "            ggml_backend_t sycl_dev0 = llama_context_sycl_backend_of(backends, "
              "model.devices[0].dev);\n"
              "            if (sycl_dev0) {\n"
              "                sycl_host_output_bytes = new_size;\n"
              "                const bool grown       = sycl_host_hooks.notify_host_buffer_sizes(sycl_dev0, "
              "sycl_host_compute_bytes,\n"
              "                                                                                  "
              "sycl_host_output_bytes);\n"
              "                llama_context_sycl_notify_checked(grown, sycl_host_compute_bytes, "
              "sycl_host_output_bytes);\n"
              "            }\n"
              "        }\n")
_OUT_CALL = ("                const bool grown       = sycl_host_hooks.notify_host_buffer_sizes(sycl_dev0, "
             "sycl_host_compute_bytes,\n"
             "                                                                                  "
             "sycl_host_output_bytes);\n")
_OUT_ALLOC = "        buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));\n"
_LCH_NOTIFY = "    decltype(&ggml_backend_sycl_notify_host_buffer_sizes) notify_host_buffer_sizes = nullptr;\n"
_LCH_PLAN = "    decltype(&ggml_backend_sycl_plan_host_holders)        plan_host_holders        = nullptr;\n"
_UNPLANNED_SLOT = ("        // No host holder plan is committed for this device, so nothing sized its slots: each has "
                   "capacity 0 and\n")
_UNPLANNED_POOLS = ("        if (!warnings.unplanned.exchange(true)) {\n"
                    "            GGML_LOG_WARN(\"[HOST-ARENA] DEVICE pools dev=%d unplanned; not initialized\\n\", "
                    "device);\n"
                    "        }\n"
                    "        return d;\n")


def _outdent(text: str) -> str:
    """`text` one level, 4 columns, to the left."""
    return "".join(line[4:] if line.startswith("    ") else line for line in text.splitlines(True))


# Guards that do not except the trial: a flag sched_reserve resets itself, a disjunction, another object's member.
_GUARD = "if (!sycl_ubatch_trial_active) {"
_GUARD_RESET = "if (!sched_need_reserve) {"
_GUARD_OR = "if (true || !sycl_ubatch_trial_active) {"
_GUARD_MEMBER = "if (cparams.pipeline_parallel == false) {"
_WITHDRAW_CHECKED = ("            if (!host_arena_->withdraw_zone_chunk(host_zone_id::SCRATCH, chunk_idx)) {\n"
                     "                GGML_LOG_WARN(\n"
                     "                    \"[HOST-ARENA] SCRATCH return skipped dev=%d chunk=%zu: not a chunk of the "
                     "SCRATCH zone alone\\n\",\n"
                     "                    dev, chunk_idx);\n"
                     "                continue;\n"
                     "            }\n")
_FILED = "            marked_set_.push_back({ chunk_idx, owner });\n"
_ERASED = "    states.erase(state);\n"
_OWNER_END_CANDIDATES = "            (void) release_marked_chunk_locked(candidate.chunk_idx, released, true);\n"
_NOTIFY_DEVICE = ("    return ggml_sycl_host_buft_device(ggml_backend_dev_host_buffer_type(ggml_backend_get_device("
                  "backend)));\n")
_DEVICE_GATE = "        if (ggml_sycl::host_zone_device_check_due(static_cast<int>(live_on_device))) {\n"
_POOLS_SHUTDOWN = "            g_pinned_buffer_pools[d].shutdown();\n"
_DEVICE_CHECK = ("    ggml_sycl::unified_cache_host_zone_owner_check(ggml_sycl::alloc_lifetime::DEVICE,\n"
                 "                                                   { ggml_sycl::alloc_owner_kind::DEVICE, 0 });\n")
_TENANT_LIFETIME = "        req.intent.lifetime                     = ggml_sycl::alloc_lifetime::CONTEXT;\n"
_TENANT_OWNER = ("        req.intent.owner = { ggml_sycl::alloc_owner_kind::EXECUTION_CONTEXT, "
                 "ggml_sycl_context_execution_id(ctx) };\n")

FIXTURES = {}

# (id, clause, edits, the reason its clause must fail with)
MUTANTS = [
    ("site-lifetime-changed", "site-table", [(PBP, _PBP_DEVICE, _PBP_DEVICE.replace("DEVICE", "GRAPH"))],
     "('pinned-buffer-pool.cpp', 'PinnedBufferPool::init', 1) lifetime is 'GRAPH' in the tree, 'DEVICE' in the table"),
    ("site-unclassified", "site-table", [_PLANTED_SITE],
     "('cpu-dispatch.cpp', 'routing_gate_planted_site', 1) at line"),
    ("doc-role-differs", "doc-table",
     [(DOC, "| `gs:36787` | local `fallback_handle` (`buffer_get_tensor`) | STAGING |",
       "| `gs:36787` | local `fallback_handle` (`buffer_get_tensor`) | TP_TMP |")],
     "doc table differs: `gs:36787` role is 'TP_TMP' in §4.4 and 'STAGING' in the tree"),
    ("doc-drift-stale", "doc-table",
     [(DOC, "| `gs:48244` | `g_tp_*` job slot,", "| `gs:48244` | `w.results` job slot,")],
     "stale drift entry ('`gs:48244`', 'holder')"),
    ("zone-weight-category-clause", "select-zone",
     [(PLAN, "    if (role == alloc_role::WEIGHT) {\n",
       "    if (role == alloc_role::WEIGHT || category == runtime_category::HOST_COMPUTE) {\n")],
     "select_zone has a category clause for WEIGHT"),
    ("zone-unowned-dropped", "select-zone",
     [(PLAN, "    if (lifetime == alloc_lifetime::UNOWNED || persistent_owner) {\n", "    if (persistent_owner) {\n")],
     "routes role STAGING lifetime UNOWNED category HOST_COMPUTE to SCRATCH, not WEIGHT"),
    ("owner-dropped", "owners", [(GS, _TENANT_OWNER, "")],
     "('ggml-sycl.cpp', 'ggml_sycl_reserve_host_tenants', 1) at line"),
    ("owner-builder-not-buffer", "owners",
     [(GS, "    intent.owner                       = { ggml_sycl::alloc_owner_kind::BUFFER, owner_id };\n",
       "    intent.owner                       = { ggml_sycl::alloc_owner_kind::DEVICE, owner_id };\n")],
     "ggml_sycl_host_buft_alloc does not mint the builder rows' BUFFER owner"),
    ("unowned-row-added", "unowned-ratchet",
     [(GS, _TENANT_LIFETIME + _TENANT_OWNER, _TENANT_LIFETIME.replace("CONTEXT", "UNOWNED") + _TENANT_OWNER)],
     "carries UNOWNED and is not one of the site table's 12 lines"),
    ("unowned-stamped", "unowned-ratchet",
     [(CD, None, "\nstatic ggml_sycl::alloc_lifetime routing_gate_planted_lifetime() {\n"
                 "    ggml_sycl::alloc_lifetime life = ggml_sycl::alloc_lifetime::UNOWNED;\n"
                 "    return life;\n"
                 "}\n")],
     "(routing_gate_planted_lifetime) stamps UNOWNED outside the site table's 12 lines"),
    ("unsized-term-unfolded", "unsized-ratchet",
     [(UC, _FOLD_FOUR, "{ k_scratch_headroom, moe_host_scratch }")],
     "maps to max_copied_tensor_bytes, which the host SCRATCH sum does not read"),
    ("unsized-graph-row", "unsized-ratchet", [_PLANTED_SITE],
     "maps to no plan term and is not one of the UNSIZED rows"),
    ("device-lifetime-lost", "device-holders", [(PBP, _PBP_DEVICE, _PBP_DEVICE.replace("DEVICE", "GRAPH"))],
     "('pinned-buffer-pool.cpp', 'PinnedBufferPool::init', 1) does not set lifetime = DEVICE"),
    ("device-release-lost", "device-holders", [(GS, _POOLS_SHUTDOWN, "")],
     "ggml_sycl_shutdown_global_runtime_pinned_owners no longer releases PinnedBufferPool::init's holder"),
    ("device-shutdown-checks", "device-holders",
     [(GS, _POOLS_SHUTDOWN + "        }\n", _POOLS_SHUTDOWN + "        }\n" + "    " + _DEVICE_CHECK)],
     "ggml_sycl_shutdown_global_runtime_pinned_owners checks a record"),
    ("device-slots-unreleased", "device-holders", [(GS, "    ggml_sycl_attn_host_staging_free_all();\n", "")],
     "module shutdown does not release the staging slots after the pinned owners"),
    ("device-check-before-delete", "device-holders",
     [(GS, "    const int      freed_device   = sycl_ctx ? sycl_ctx->device : -1;\n",
       _DEVICE_CHECK + "    const int      freed_device   = sycl_ctx ? sycl_ctx->device : -1;\n")],
     "the DEVICE check runs before delete sycl_ctx"),
    ("device-check-ungated", "device-holders", [(GS, _DEVICE_GATE, "        if (true) {\n")],
     "the DEVICE check is not gated on host_zone_device_check_due"),
    ("slot-planned-grows", "slot-refusal", [_PLANNED_GROW], "a planned slot grows at dispatch above its capacity"),
    ("slot-idle-unadmitted", "slot-refusal",
     [(GS, _IDLE + "        if (!g_host_staging_slots->admit(device, (int) kind, bytes, "
                   "ggml_sycl_attn_host_staging_allocator(),\n"
                   "                                         g_attn_host_dispatch_counters)) {\n"
                   "            return nullptr;\n"
                   "        }\n", _IDLE)],
     "the IDLE path hands out a per-call buffer without the slot's admission"),
    ("slot-refusal-silent", "slot-refusal",
     [(PLAN, "\n        \"[ATTN-HOST] staging request %zu B for kind %s exceeds the planned %zu B on device %d; "
             "declining",
       "\n        \"[ATTN-HOST] staging request %zu B for kind %s over its slot on device %d; declining")],
     "the planned refusal no longer WARNs by name and counts"),
    ("plan-stages-with-dispatch-off", "staging-plan",
     [(PLAN, "    if (!w.env->host_dispatch_on) {\n        return;\n    }\n", "")],
     "walk_visit_host_kv stages with the host dispatch off"),
    ("plan-predicate-is-host", "staging-plan",
     [(PLAN, "    return t && env.in_kv_host_buft(t);\n",
       "    return t && t->buffer && ggml_backend_buffer_is_host(t->buffer);\n")],
     "walk_in_kv_host is not the interception's predicate"),
    ("plan-device-zero", "staging-plan",
     [(PLAN, "    return w.env->backend_device(backend);\n", "    (void) backend;\n    return 0;\n")],
     "a slot's device is not the scheduler's assigned backend's"),
    ("pool-own-dispatch-count", "pool-dims",
     [(GS, _INIT_SCAN, _INIT_SCAN + "    const int64_t max_dispatch_count = graph_needs.D;\n"
                                    "    (void) max_dispatch_count;\n")],
     "computes max_dispatch_count outside host_pool_dims_add"),
    ("pool-eligibility-is-host", "pool-dims",
     [(PLAN, "    return host_experts && host_experts(src0, device, known, cause);\n",
       "    return src0->buffer && ggml_backend_buffer_is_host(src0->buffer);\n")],
     "the pools' eligibility is not placement's executor fact"),
    ("pool-fact-reads-buffer", "pool-dims",
     [(GS, "    const bool host       = ggml_sycl_moe_plan_has_host_experts(src0, device, &fact_known);\n",
       "    const bool host       = src0->buffer && ggml_backend_buffer_is_host(src0->buffer);\n"
       "    fact_known            = true;\n")],
     "ggml_sycl_host_holder_host_experts does not read ggml_sycl_moe_plan_has_host_experts"),
    ("pool-walk-fact-rebound", "pool-dims",
     [(GS, "    env.host_experts     = ggml_sycl_host_holder_host_experts;\n",
       "    env.host_experts     = ggml_sycl_routing_gate_planted_is_host;\n")],
     "the walk env's host_experts is bound to"),
    ("pool-init-fact-rebound", "pool-dims",
     [(GS, "ggml_sycl_host_holder_host_experts, g_host_pool_init_warnings[device]",
       "ggml_sycl_routing_gate_planted_is_host, g_host_pool_init_warnings[device]")],
     "passes host_pools_init_from_plan the host-experts fact 'ggml_sycl_routing_gate_planted_is_host'"),
    ("pool-init-registers-demand", "pool-dims", [(GS, _INIT_SCAN, _INIT_SCAN + "    holder_demand(graph_needs);\n")],
     "moe_hybrid_init_once registers SCRATCH demand"),
    ("pool-scan-new-caller", "pool-dims",
     [(PLAN, None, "\nhost_pool_dims routing_gate_planted_scan(const ggml_cgraph * graph) {\n"
                   "    return host_pool_scan(graph, host_pool_has_storage, nullptr);\n"
                   "}\n")],
     "host_pool_scan is called by"),
    ("reach-free-in-release", "marked-set-reach", [(POOL, _ERASED, _ERASED + "    free_chunk_owner(c);\n")],
     "free_chunk_owner is called by"),
    ("reach-release-from-owner-end", "marked-set-reach",
     [(UC, _OWNER_END_CANDIDATES, _OWNER_END_CANDIDATES + "            (void) host_arena_->release_zone_chunk("
                                                          "host_zone_id::SCRATCH, candidate.chunk_idx, "
                                                          "released.back(), true);\n")],
     "release_zone_chunk is called by"),
    ("reach-new-release-caller", "marked-set-reach",
     [(UC, None, "\nvoid unified_cache::routing_gate_planted_release(size_t chunk_idx) {\n"
                 "    (void) release_marked_chunk(chunk_idx);\n"
                 "}\n")],
     "release_marked_chunk is called by unified_cache::routing_gate_planted_release"),
    ("return-ignores-refusal", "return-files",
     [(UC, _WITHDRAW_CHECKED,
       "            (void) host_arena_->withdraw_zone_chunk(host_zone_id::SCRATCH, chunk_idx);\n")],
     "RETURN files a chunk whose withdrawal was refused"),
    ("return-files-first", "return-files", [(UC, _FILED, ""), (UC, _WITHDRAW_CHECKED, _FILED + _WITHDRAW_CHECKED)],
     "RETURN files a chunk before withdrawing it"),
    ("withdrawal-in-release", "withdrawal", [(POOL, _ERASED, _ERASED + "    zones_[zi].size -= 0;\n")],
     "lowers a zone's size outside withdraw_zone_chunk"),
    ("withdrawal-unguarded", "withdrawal",
     [(POOL, "        if (!zcs.withdrawn) {\n            zcs.withdrawn = true;\n            zones_[zi].size -= "
             "zcs.zone_size;\n        }\n",
       "        zcs.withdrawn = true;\n        zones_[zi].size -= zcs.zone_size;\n")],
     "the zone's size is not lowered exactly once"),
    ("notify-device-zero", "notify-device", [(GS, _NOTIFY_DEVICE, "    (void) backend;\n    return 0;\n")],
     "the notify's device is not the host buft's"),
    ("notify-backend-device", "notify-device",
     [(GS, _NOTIFY_DEVICE, "    return static_cast<const ggml_backend_sycl_context *>(backend->context)->device;\n")],
     "ggml_sycl_notify_host_buft_device resolves the notifying backend's device"),
    ("notify-additive-grow", "notify-device",
     [(GS, "    return ggml_sycl::host_scratch_notify(ggml_sycl_notify_env(), backend, host_compute_bytes, "
           "host_output_bytes);\n",
       "    return ggml_sycl::unified_cache_grow_scratch_zone(ggml_sycl_notify_host_buft_device(backend),\n"
       "                                                      host_compute_bytes + host_output_bytes);\n")],
     "the notify does not grow through host_scratch_notify"),
    ("notify-growth-device0", "notify-device",
     [(GS, "    return ggml_sycl::unified_cache_host_scratch_ledger(device, "
           "ggml_backend_sycl_host_buffer_type_chunk_cap(), ledger,\n",
       "    (void) device;\n"
       "    return ggml_sycl::unified_cache_host_scratch_ledger(0, "
       "ggml_backend_sycl_host_buffer_type_chunk_cap(), ledger,\n")],
     "ggml_sycl_notify_ledger_of does not resolve the ledger of the device"),
    ("grow-host-scratch-called", "grow-host-scratch-gone",
     [(GS, _ARENA_RESERVED, _ARENA_RESERVED + "    ggml_sycl::unified_cache_grow_host_scratch_zone(total_bytes);\n")],
     "ggml/src/ggml-sycl/ggml-sycl.cpp defines or calls unified_cache_grow_host_scratch_zone"),
    ("notify-compute-called", "notify-compute-unused",
     [("src/llama-model.cpp", None, "\nstatic void routing_gate_planted(ggml_backend_t b, const size_t * s) {\n"
                                    "    ggml_backend_sycl_notify_compute_buffer_sizes(b, s, 1);\n"
                                    "}\n")],
     "notify compute: src/llama-model.cpp:"),
    ("notify-compute-by-proc", "notify-compute-unused",
     [(LC, _HOOKS_AT, _HOOKS_AT + _hook("notify_compute_buffer_sizes"))],
     "notify compute: src/llama-context.cpp:"),
    ("dlopen-notify-struct", "dlopen-header",
     [(HEADER, "(ggml_backend_t backend,\n" + _HDR_ARG + "size_t         host_compute_bytes,\n"
               + _HDR_ARG + "size_t         host_output_bytes);\n",
       "(ggml_backend_t backend,\n" + _HDR_ARG + "const struct ggml_sycl_host_sizes * sizes);\n")],
     "ggml-sycl.h's notify takes"),
    ("dlopen-notify-void", "dlopen-header",
     [(HEADER, "GGML_BACKEND_API bool ggml_backend_sycl_notify_host_buffer_sizes(",
       "GGML_BACKEND_API void ggml_backend_sycl_notify_host_buffer_sizes(")],
     "ggml-sycl.h's notify returns 'void', not bool"),
    ("dlopen-plan-llama", "dlopen-header",
     [(HEADER, "(ggml_backend_sched_t sched, struct ggml_cgraph * graph);",
       "(ggml_backend_sched_t sched, struct ggml_cgraph * graph,\n"
       "                                                         struct llama_context * lctx);")],
     "ggml-sycl.h's plan entry is"),
    ("grow-host-scratch-defined", "grow-host-scratch-gone",
     [(UC, None, "\nvoid unified_cache_grow_host_scratch_zone(int device, size_t bytes) {\n"
                 "    (void) device;\n"
                 "    (void) bytes;\n"
                 "}\n")],
     "defines or calls unified_cache_grow_host_scratch_zone"),
    ("growth-planned-branch", "slot-growth-plan-only", [_PLANNED_GROW], "slot_replace is called by"),
    ("growth-unplanned-grows", "slot-growth-plan-only",
     [(PLAN, _UNPLANNED_SLOT,
       "        if (bytes > slot.cap && !slot_replace(slot, device, kind, host_staging_slot_bytes(bytes), alloc)) {\n"
       "            return nullptr;\n"
       "        }\n" + _UNPLANNED_SLOT)],
     "slot_replace is called by ['host_staging_slots::admit'"),
    ("growth-idle-allocates", "slot-growth-plan-only",
     [(GS, _IDLE, _IDLE + "        ggml_sycl::host_staging_block grown{};\n"
                          "        (void) ggml_sycl_attn_host_staging_block_alloc(nullptr, device, (int) kind, bytes, "
                          "&grown);\n")],
     "ggml_sycl_attn_host_staging_get allocates a slot block itself"),
    ("pools-own-scan-planned", "pools-from-plan",
     [(PLAN, "    d.dims = state.pools;\n", "    d.dims = host_pool_scan(graph, host_pool_has_storage, nullptr);\n")],
     "the pools' init is sized from"),
    ("pools-unplanned-scan", "pools-from-plan",
     [(PLAN, _UNPLANNED_POOLS, "        d.dims = host_pool_scan(graph, host_pool_has_storage, nullptr);\n"
                               "        return d;\n")],
     "the pools' init is sized from 'd.dims = host_pool_scan("),
    ("wiring-retry-unbegun", "context-wiring", [(LC, _RETRY_BEGIN, "")],
     "creates a scheduler and reserves before the begin entry"),
    ("wiring-begin-before-creation", "context-wiring",
     [(LC, _SCHED_NEW_0, _BEGIN + _SCHED_NEW_0), (LC, _BEGIN_0_NOTE + _BEGIN, _BEGIN_0_NOTE)],
     "sched_reserve_impl creates a scheduler and reserves before the begin entry"),
    ("wiring-plan-after-full-reserve", "context-wiring",
     [(LC, "n_outputs_pp, mctx.get(), true, sizes.data()))", "n_outputs_pp, mctx.get(), false, sizes.data()))")],
     "calls the plan entry not right after a split_only graph_reserve"),
    ("wiring-commit-unresolved", "context-wiring", [(LC, _hook("commit_host_holders"), "")],
     "does not resolve ggml_backend_sycl_commit_host_holders"),
    ("wiring-commit-before-alloc", "context-wiring",
     [(LC, _MEASURE_CALL, _MEASURE_CALL + "    sycl_host_hooks.commit_host_holders(sycl_dev0);\n")],
     "llama_context::sched_reserve_impl calls the commit entry"),
    ("wiring-end-commit-unguarded", "context-wiring",
     [(LC, _SR_COMMIT, _outdent(_SR_COMMIT.split("\n", 1)[1]) + "    if (!sycl_ubatch_trial_active) {\n")],
     "is not under an if that excepts the auto-n_ubatch trial"),
    ("wiring-end-commit-flag-unset", "context-wiring", [(LC, _FLAG, "")],
     "is not under an if that excepts the auto-n_ubatch trial"),
    ("wiring-end-commit-reset-flag", "context-wiring", [(LC, _SR_COMMIT, _SR_COMMIT.replace(_GUARD, _GUARD_RESET))],
     "is not under an if that excepts the auto-n_ubatch trial"),
    ("wiring-end-commit-disjunction", "context-wiring", [(LC, _SR_COMMIT, _SR_COMMIT.replace(_GUARD, _GUARD_OR))],
     "is not under an if that excepts the auto-n_ubatch trial"),
    ("wiring-end-commit-member", "context-wiring", [(LC, _SR_COMMIT, _SR_COMMIT.replace(_GUARD, _GUARD_MEMBER))],
     "is not under an if that excepts the auto-n_ubatch trial"),
    ("wiring-trial-commit-before-fallback", "context-wiring",
     [(LC, _CAND_COMMIT + _RENOTIFY + _CAND_KEEP, _RENOTIFY + _CAND_KEEP),
      (LC, _RESERVED_OK, _RESERVED_OK + _CAND_COMMIT)],
     "try_candidate does not call the commit entry after its host-fallback check"),
    ("wiring-nothrow-commit-missing", "context-wiring", [(LC, _NT_COMMIT, "")],
     "sched_reserve_nothrow does not call the commit entry in its success arm"),
    ("wiring-nothrow-commit-unguarded", "context-wiring",
     [(LC, _NT_GUARD + _NT_COMMIT, _outdent(_NT_COMMIT) + _NT_GUARD)],
     "sched_reserve_nothrow's success commit at line"),
    ("trial-open-missing", "trial-ops", [(LC, _OPEN, "")], "try_candidate does not OPEN before its reserve"),
    ("trial-open-after-reserve", "trial-ops",
     [(LC, _OPEN, ""), (LC, _RESERVED_OK, _RESERVED_OK + _OPEN)],
     "try_candidate does not OPEN before its reserve"),
    ("trial-one-return", "trial-ops", [(LC, _LOSS_HOST, _LOSS_HOST_AT)],
     "try_candidate does not RETURN on every loss path: none before its host-fallback return"),
    ("trial-catch-unreturned", "trial-ops", [(LC, _LOSS_FIT, _LOSS_FIT_AT)],
     "try_candidate does not RETURN on every loss path: none before its reserve-failure return"),
    ("trial-hold-spill-unreturned", "trial-ops", [(LC, _LOSS_SPILL, _LOSS_SPILL_AT)],
     "try_candidate does not RETURN on every loss path: none before its hold-spill return"),
    ("trial-keep-before-fallback", "trial-ops",
     [(LC, _RENOTIFY + _CAND_KEEP, _RENOTIFY), (LC, _RESERVED_OK, _RESERVED_OK + _CAND_KEEP)],
     "try_candidate does not KEEP beside the commit"),
    ("trial-sched-return-on-success", "trial-ops",
     [(LC, _SR_CATCH, "        throw;\n"),
      (LC, _SR_KEEP + _SR_END, _SR_KEEP + "    }\n" + _outdent(_SR_CATCH.split("\n")[0] + "\n") + _SR_CLOSE)],
     "sched_reserve does not RETURN on its transaction's throw path"),
    ("trial-sched-return-one-branch", "trial-ops",
     [(LC, _SR_CATCH, "        throw;\n"),
      (LC, _SR_FIT_REFUSAL, _SR_FIT_REFUSAL.replace("{\n", "{\n        " + _SR_CATCH.split("\n")[0] + "\n"))],
     "sched_reserve does not RETURN on its transaction's throw path before its 'throw std::runtime_error'"),
    ("trial-sched-catch-unreturned", "trial-ops", [(LC, _SR_CATCH, "        throw;\n")],
     "sched_reserve does not RETURN on its transaction's throw path before its 'throw llama_auto_ubatch_fit_refusal'"),
    ("trial-sched-catch-swallows", "trial-ops",
     [(LC, _SR_CATCH, "        " + _STEP + "RETURN, cparams.n_ubatch);\n")],
     "sched_reserve does not RETURN on its transaction's throw path before its 'throw llama_auto_ubatch_fit_refusal'"),
    ("trial-sched-keep-before-transaction", "trial-ops",
     [(LC, _SR_KEEP + _SR_END, _SR_END),
      (LC, _SR_TRY, "    if (!sycl_ubatch_trial_active) {\n" + _SR_KEEP + "    }\n" + _SR_TRY)],
     "sched_reserve does not KEEP on its success path"),
    ("trial-sched-keep-unguarded", "trial-ops",
     [(LC, _SR_KEEP + _SR_END, "    }\n" + _outdent(_SR_KEEP) + _SR_CLOSE)],
     "is not under an if that excepts the auto-n_ubatch trial"),
    ("trial-sched-keep-flag-unset", "trial-ops", [(LC, _FLAG, "")],
     "is not under an if that excepts the auto-n_ubatch trial"),
    ("trial-sched-keep-reset-flag", "trial-ops", [(LC, _SR_COMMIT, _SR_COMMIT.replace(_GUARD, _GUARD_RESET))],
     "is not under an if that excepts the auto-n_ubatch trial"),
    ("trial-sched-keep-disjunction", "trial-ops", [(LC, _SR_COMMIT, _SR_COMMIT.replace(_GUARD, _GUARD_OR))],
     "is not under an if that excepts the auto-n_ubatch trial"),
    ("trial-sched-keep-member", "trial-ops", [(LC, _SR_COMMIT, _SR_COMMIT.replace(_GUARD, _GUARD_MEMBER))],
     "is not under an if that excepts the auto-n_ubatch trial"),
    ("trial-nothrow-keep-unguarded", "trial-ops",
     [(LC, _NT_KEEP_END,
       "            }\n" + _outdent(_NT_KEEP_END.split("\n")[0] + "\n") + "            return true;\n")],
     "sched_reserve_nothrow's success KEEP at line"),
    ("trial-nothrow-unreturned", "trial-ops", [(LC, _NT_RETURN, "    sched_need_reserve = true;\n")],
     "sched_reserve_nothrow does not RETURN on every way out but OK"),
    ("trial-mark-after-measure", "trial-ops",
     [(LC, _MARK, ""), (LC, _PP_SPLITS, _PP_SPLITS + _MARK)],
     "a RETURN_TO_MARK is not preceded by its MARK before the reserve"),
    ("trial-rtm-after-retry-scheduler", "trial-ops",
     [(LC, _RTM, ""), (LC, _RETRY_RESERVE, _RTM + _RETRY_RESERVE)],
     "does not RETURN_TO_MARK between the failed reserve and the pipeline-parallel retry's scheduler"),
    ("trial-mark-missing", "trial-ops", [(LC, _MARK, "")], "a RETURN_TO_MARK is not preceded by its MARK"),
    ("notify-output-after-alloc", "notify-position", [(LC, _OUT_BLOCK + _OUT_ALLOC, _OUT_ALLOC + _OUT_BLOCK)],
     "calls the notify outside the span between the old buffer's free and ggml_backend_buft_alloc_buffer"),
    ("notify-sched-after-alloc", "notify-position",
     [(LC, _MEASURE_NOTIFY, "    };\n"), (LC, _PP_SPLITS, _PP_SPLITS + _MEASURE_NOTIFY[:-len("    };\n")])],
     "calls the notify on sched_reserve's path outside the span"),
    ("notify-backend-loop", "notify-position",
     [(LC, _OUT_CALL, "                bool grown = true;\n"
                      "                for (auto & backend : backends) {\n"
                      "                    grown = sycl_host_hooks.notify_host_buffer_sizes(backend.get(), 0,\n"
                      "                                                                     new_size) && grown;\n"
                      "                }\n")],
     "calls the notify in a loop, not once with one backend"),
    ("dev0-warn-dropped", "dev0-warn", [(LC, _DEV0_CALL, "")],
     "does not call llama_context_sycl_warn_dev0(model.devices) right after it resolves"),
    ("dev0-warn-twice", "dev0-warn", [(LC, _FLAG, _FLAG + _DEV0_CALL)], "is called 2 times, not once per context"),
    ("dev0-warn-inverted", "dev0-warn", [(LC, _DEV0_GUARD, _DEV0_GUARD.replace("|| ll", "|| !ll"))],
     "does not return when devices[0] is SYCL"),
    ("dev0-warn-unnamed", "dev0-warn", [(LC, _DEV0_NAME, "")], "does not name the device list"),
    ("notify-other-backend", "notify-position",
     [(LC, _IMPL_DEV0,
       "    ggml_backend_t sycl_dev0 = backends.back().get();\n    bool   sycl_host_measured = false;\n")],
     "does not pass model.devices[0]'s backend"),
    ("notify-renotify-after-keep", "notify-position",
     [(LC, _RENOTIFY + _SR_KEEP + _SR_END, _SR_KEEP + "    }\n" + _outdent(_RENOTIFY) + _SR_CLOSE)],
     "calls the notify outside output_reserve, sched_reserve's path and a commit's re-notify"),
    ("notify-renotify-dropped", "notify-position", [(LC, _RENOTIFY + _CAND_KEEP, _CAND_KEEP)],
     "has no re-notify after it in its block"),
    ("dlopen-context-notify-struct", "dlopen-context",
     [(LCH, _LCH_NOTIFY,
       "    bool (*notify_host_buffer_sizes)(ggml_backend_t, const llama_host_sizes *) = nullptr;\n")],
     "llama_context's notify pointer type takes"),
    ("dlopen-context-plan-llama", "dlopen-context",
     [(LCH, _LCH_PLAN, "    bool (*plan_host_holders)(ggml_backend_sched_t, llama_context *) = nullptr;\n")],
     "llama_context's plan entry pointer type is"),
    ("owner-end-after-delete", "owner-end-wired",
     [(GS, _OWNER_END + "    delete sycl_ctx;\n", "    delete sycl_ctx;\n" + _OWNER_END)],
     "calls the owner's end after delete sycl_ctx"),
    ("open-returns-chunks", "open-touches-no-set",
     [(UC, "            return host_scratch_trial_open(owner, n_ubatch);\n",
       "            return host_scratch_return_chunks(host_scratch_trial_chunks(owner), owner);\n")],
     "touches the marked set in its OPEN step"),
    # The host SCRATCH fold and the runtime term.
    ("fold-device-term", "scratch-fold",
     [(UC, _FOLD_FOUR, _FOLD_FOUR.replace(" }", ", plan.pp_moe_onednn_scratch_bytes }"))],
     "reads ['pp_moe_onednn_scratch_bytes'] beyond the four host terms"),
    ("fold-term-dropped", "scratch-fold", [(UC, _FOLD_FOUR, "{ plan.max_copied_tensor_bytes, moe_host_scratch }")],
     "lacks ['k_scratch_headroom']"),
    # The fold back on the model-wide maximum: a host-kept gather-only table would size SCRATCH again.
    ("fold-global-max", "scratch-fold",
     [(UC, _FOLD_FOUR, _FOLD_FOUR.replace("max_copied_tensor_bytes", "max_tensor_bytes"))],
     "reads ['max_tensor_bytes'] beyond the four host terms"),
    ("runtime-pp-term", "runtime-term",
     [(UCH, _RUNTIME_FOUR, _RUNTIME_FOUR.replace(";\n", " + plan.pp_moe_onednn_scratch_bytes;\n"))],
     "names ['pp_moe_onednn_scratch_bytes']"),
]

# A target-gone control removes what an EXPECTED-RED clause inspects. The clause must then fail with a distinct
# reason and never with its recorded one, so a tree that deletes the target does not score as still waiting for its
# landing task. A clause whose red state is its target's absence itself goes in NO_GONE instead.
GONE = {}
NO_GONE = ()

# An edit that changes no clause's subject: every REQUIRED clause must still hold, so a mutant's failure is its edit's.
NEGATIVE_CONTROL = [(rel, None, "\n// routing gate negative control: a comment changes nothing\n")
                    for rel in (GS, PLAN, UC, POOL)]


# ---------------------------------------------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------------------------------------------

_TREE = {}
_BY_ID = {c[0]: c for c in CLAUSES}


def _tree() -> dict:
    if not _TREE:
        _TREE.update(tree_texts())
    return _TREE


def apply_edits(texts: dict, edits) -> dict:
    out = dict(texts)
    for rel, old, new in edits:
        if old is None:
            out[rel] = "" if new is None else out[rel] + new
            continue
        n = out[rel].count(old)
        assert n == 1, f"control edit for {rel} matches {n} times (must be once): {old[:100]!r}"
        out[rel] = out[rel].replace(old, new)
    return out


def _base(cid: str) -> dict:
    return apply_edits(_tree(), FIXTURES[cid]) if _BY_ID[cid][1] == EXPECTED_RED else _tree()


@pytest.mark.parametrize("cid", [c[0] for c in CLAUSES if c[1] == REQUIRED])
def test_required_clause_holds_on_the_tree(cid):
    fails = _BY_ID[cid][2](_tree())
    assert not fails, "\n".join(fails)


@pytest.mark.parametrize("cid", [c[0] for c in CLAUSES if c[1] == EXPECTED_RED])
def test_expected_red_clause_scores_red_until_its_task(cid):
    _, _, fn, landing, reason = _BY_ID[cid]
    fails = fn(_tree())
    assert fails, (f"{cid} is EXPECTED-RED until {landing} and now passes on this tree: the commit that satisfies it "
                   f"moves it to REQUIRED")
    assert any(reason in f for f in fails), (f"{cid} is red on this tree, but not for its recorded reason {reason!r}; "
                                             f"it reported:\n" + "\n".join(fails))
    warnings.warn(ExpectedRed(f"{cid}: EXPECTED-RED until {landing}: " + " | ".join(fails)))


@pytest.mark.parametrize("cid", [c[0] for c in CLAUSES if c[1] == EXPECTED_RED])
def test_expected_red_clause_passes_on_its_landed_fixture(cid):
    fails = _BY_ID[cid][2](_base(cid))
    assert not fails, f"{cid} fails on the fixture of its landed shape:\n" + "\n".join(fails)


@pytest.mark.parametrize("mid,cid,edits,reason", MUTANTS, ids=[m[0] for m in MUTANTS])
def test_planted_mutant_fails_with_its_reason(mid, cid, edits, reason):
    fails = _BY_ID[cid][2](apply_edits(_base(cid), edits))
    assert any(reason in f for f in fails), (f"mutant {mid} did not fail {cid} with {reason!r}; it reported:\n"
                                             + ("\n".join(fails) or "(nothing: the clause passed)"))


@pytest.mark.parametrize("cid", sorted(GONE))
def test_expected_red_clause_with_its_target_gone_fails_as_gone(cid):
    edits, gone = GONE[cid]
    reason = _BY_ID[cid][4]
    fails = _BY_ID[cid][2](apply_edits(_tree(), edits))
    assert any(gone in f for f in fails), (f"{cid} with its target gone does not fail with {gone!r}; it reported:\n"
                                           + ("\n".join(fails) or "(nothing: the clause passed)"))
    assert not any(reason in f for f in fails), (f"{cid} with its target gone still reports its recorded reason "
                                                 f"{reason!r}, so it would score as expected-red")


def test_negative_control_passes():
    texts = apply_edits(_tree(), NEGATIVE_CONTROL)
    for cid, status, fn, _, _ in CLAUSES:
        if status == REQUIRED:
            fails = fn(texts)
            assert not fails, f"{cid} fails on the negative control:\n" + "\n".join(fails)


def test_every_clause_has_a_planted_mutant_and_every_expected_red_clause_a_fixture():
    ids = [c[0] for c in CLAUSES]
    assert len(ids) == len(set(ids)), "a clause id is listed twice"
    planted = {m[1] for m in MUTANTS}
    assert planted <= set(ids), f"a mutant names no clause: {sorted(planted - set(ids))}"
    assert not set(ids) - planted, f"clauses with no planted mutant: {sorted(set(ids) - planted)}"
    assert len({m[0] for m in MUTANTS}) == len(MUTANTS), "a mutant id is listed twice"
    for cid, status, _, landing, reason in CLAUSES:
        if status == EXPECTED_RED:
            assert landing and FIXTURES.get(cid), f"EXPECTED-RED clause {cid} names no landing task or fixture"
            assert reason, f"EXPECTED-RED clause {cid} records no reason"
        else:
            assert landing is None and reason is None and cid not in FIXTURES, \
                f"REQUIRED clause {cid} carries a landing, a reason or a fixture"
    expected_red = {c[0] for c in CLAUSES if c[1] == EXPECTED_RED}
    assert set(GONE) | set(NO_GONE) == expected_red and not set(GONE) & set(NO_GONE), \
        "every EXPECTED-RED clause needs a target-gone control or a place in NO_GONE, and not both"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
