#!/usr/bin/env python3
"""A process-lifetime host owner never sits in a boundary-swept host zone, and a refused boundary check stops walking
the allocation registry once its diagnostics are spent (llama.cpp-nqbe).

THE COST.  The host STAGING and SCRATCH zones must be empty at every graph boundary: the graph code calls
unified_cache_host_zone_boundary_check() on both, once per graph compute.  The CPU expert job's input marker word
(llama.cpp-42zj) is a 64-byte host-pinned CONTROL allocation that the unified cache keeps for its own lifetime, and the
allocator's host-zone choice sent it to STAGING by fallthrough, since only KV, EXPERT_STAGING, WEIGHT and the
HOST_COMPUTE/EXPERT_CACHE categories went anywhere else.  So every boundary refused, and every refusal walked the whole
registry under its mutex, on the submitting thread, for the detail lines and the WARN's oldest epoch, even after both
budgets were spent.  Without -c the direct path reaches a boundary about 25 times per token, and with
GGML_SYCL_MOE_CPU_JOB_DEVICE_SIGNAL=1 Qwen3.8 IQ3_XXS decode on the B70 fell to 2.97 t/s from 9.7; the settle held 79%
of the submitting thread's samples.

Not every CONTROL request outlives a graph: the scoped host transients (ggml_sycl_transient_host_pinned_intent) and the
get_rows host index stage are CONTROL too, and they must stay in STAGING so the boundary check still catches one that
leaks.  So the persistence is a field of the request, alloc_constraints::persistent_owner, never a role or a cohort
string.

THE CONTRACT.  With comments and string literals blanked:
  Z1. alloc_constraints has `bool persistent_owner = false;`.  unified_cache::arm_cpu_job_input_marker builds the
      word's request with role CONTROL (its only role write), must_host_pinned = true and persistent_owner = true,
      each an unconditional statement between the request's declaration and its unified_allocate_owner call.
  Z2. unified_alloc_host_zone, the allocator's host-zone choice, returns KV exactly for `kv_spill_to_host ||
      req.intent.role == alloc_role::KV`, SCRATCH exactly for `req.intent.role == alloc_role::EXPERT_STAGING`, WEIGHT
      for a condition with `req.intent.constraints.persistent_owner` among its top-level `||` operands, and STAGING
      only as its last statement; it has no other return.  So a persistent CONTROL request lands in WEIGHT.
  Z3. unified_alloc chooses every host zone through unified_alloc_host_zone(req, cat, kv_spill_to_host), twice (the
      pinned-pool branch and the direct-zone branch, each the initialiser of its zone variable), and no lambda of its
      returns a host_zone_id of its own.
  Z4. host_zone_settle asserts that WEIGHT is never settled, which is what keeps a persistent owner out of the sweep.
  S1. host_zone_settle takes its refusal report once, as `const host_zone_refusal_report report =
      host_zone_refusal_take(zone_refusals, audit.active());`, inside the block of
      `if (runtime_registry_host_zone_live_locked(zone)) {`, and calls runtime_registry_scan_host_zone_locked exactly
      once, inside `if (report.scan) {` within that block, passing report.log_detail as its detail flag.  Both WARNs
      sit inside `if (report.warn) {` within that block, and the block ends with its refusal's `return;`.
  S2. host_zone_refusal_take is exactly the budgeted take: a detail budget of 4 and a WARN budget of 8, a count that
      stops at the WARN budget, and `scan = log_detail || warn || audit_active`.  So past the eighth refusal of a zone
      a refusal reads one counter and returns, unless a zone audit is active.

NOT COVERED: this reads text.  The decision table itself runs in test-sycl-runtime-registry-containment (host-only, the
hook allocation_registry_test_host_zone_refusals), and a device run is what shows the boundary no longer refuses.

Run with --self-test: each mutant is written to a copy of the sources in a temporary directory and must fail on its
own pin, and the unmodified tree must pass.  --root DIR checks another tree (the directory holding ggml/).
"""

import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = "(llama.cpp-nqbe)"
FILES = {
    "cache": "ggml/src/ggml-sycl/unified-cache.cpp",
    "header": "ggml/src/ggml-sycl/unified-cache.hpp",
}

_LEXEME_RE = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|//[^\n]*|/\*.*?\*/', flags=re.DOTALL)


def blank(src: str) -> str:
    """Comments dropped (newlines kept), string and char literals emptied, so neither counts as code."""

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] == '"':
            return '""'
        if tok[0] == "'":
            return "' '"
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def nospace(s: str) -> str:
    return re.sub(r"\s+", "", s)


def block_at(code: str, open_idx: int):
    """(body, start) of the brace block whose `{` is at open_idx; body excludes the braces."""
    depth = 0
    for i in range(open_idx, len(code)):
        if code[i] == "{":
            depth += 1
        elif code[i] == "}":
            depth -= 1
            if depth == 0:
                return code[open_idx + 1 : i], open_idx + 1
    return None, None


def definition(code: str, signature: str):
    m = re.search(signature, code)
    if not m:
        return None
    body, _ = block_at(code, m.end() - 1)
    return body


def top_level_or_operands(cond: str) -> list[str]:
    out, depth, cur, i = [], 0, "", 0
    while i < len(cond):
        c = cond[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if depth == 0 and cond.startswith("||", i):
            out.append(cur.strip())
            cur, i = "", i + 2
            continue
        cur += c
        i += 1
    out.append(cur.strip())
    return out


def if_blocks(body: str):
    """(condition, block body, block start) of every `if (...) {` in body, outermost and nested."""
    out = []
    for m in re.finditer(r"\bif\s*\(", body):
        depth, j = 0, m.end() - 1
        while j < len(body):
            if body[j] == "(":
                depth += 1
            elif body[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        cond = body[m.end() : j]
        brace = re.match(r"\s*\{", body[j + 1 :])
        if not brace:
            continue
        b, start = block_at(body, j + 1 + brace.end() - 1)
        out.append((cond, b, start))
    return out


ARM_SIG = r"bool\s+unified_cache::arm_cpu_job_input_marker\s*\([^)]*\)\s*\{"
ZONE_SIG = (r"static\s+host_zone_id\s+unified_alloc_host_zone\s*\(\s*const\s+alloc_request\s*&\s*req\s*,\s*"
            r"runtime_category\s+cat\s*,\s*bool\s+kv_spill_to_host\s*\)\s*noexcept\s*\{")
ALLOC_SIG = r"\nbool\s+unified_alloc\s*\(\s*const\s+alloc_request\s*&\s*req_in\s*,\s*alloc_handle\s*\*\s*out\s*\)\s*\{"
SETTLE_SIG = r"void\s+unified_cache::host_zone_settle\s*\(\s*host_zone_id\s+zone\s*\)\s*\{"
TAKE_SIG = (r"static\s+host_zone_refusal_report\s+host_zone_refusal_take\s*\(\s*std::atomic<int>\s*&\s*refusals\s*,\s*"
            r"bool\s+audit_active\s*\)\s*noexcept\s*\{")
ZONE_CALL = "unified_alloc_host_zone(req, cat, kv_spill_to_host)"
SCAN = "runtime_registry_scan_host_zone_locked"
LIVE_GATE = "runtime_registry_host_zone_live_locked(zone)"
TAKE_STMT = "const host_zone_refusal_report report = host_zone_refusal_take(zone_refusals, audit.active());"
TAKE_BODY = """
    constexpr int detail_budget = 4;
    constexpr int warn_budget   = 8;
    int           index         = refusals.load(std::memory_order_relaxed);
    if (index < warn_budget) {
        index = refusals.fetch_add(1, std::memory_order_relaxed);
    }
    host_zone_refusal_report report{};
    report.log_detail = index < detail_budget;
    report.warn       = index < warn_budget;
    report.scan       = report.log_detail || report.warn || audit_active;
    return report;
"""


def check_marker_request(code: str, fails: list) -> None:
    arm = definition(code, ARM_SIG)
    if arm is None:
        fails.append(("Z1", "unified_cache::arm_cpu_job_input_marker is not defined"))
        return
    decl = re.search(r"\balloc_request\s+req\s*\{\s*\}\s*;", arm)
    alloc = re.search(r"\bunified_allocate_owner\s*\(\s*req\s*\)", arm)
    if not decl or not alloc or alloc.start() < decl.start():
        fails.append(("Z1", "arm_cpu_job_input_marker does not declare its request and then allocate it with "
                            "unified_allocate_owner(req)"))
        return
    run = arm[decl.end() : alloc.start()]
    stmts = [nospace(s) for s in run.split(";")]
    want = ("req.intent.role=alloc_role::CONTROL", "req.intent.constraints.must_host_pinned=true",
            "req.intent.constraints.persistent_owner=true")
    roles = [s for s in stmts if s.startswith("req.intent.role=")]
    if "{" in run or "}" in run or any(w not in stmts for w in want) or roles != [want[0]]:
        fails.append(("Z1", "the input marker's request does not unconditionally set role CONTROL, must_host_pinned "
                            "and persistent_owner: the word lives as long as the cache, so without persistent_owner "
                            "it lands in the boundary-swept STAGING zone and every graph boundary refuses on it"))


def check_zone_choice(code: str, fails: list) -> None:
    body = definition(code, ZONE_SIG)
    if body is None:
        fails.append(("Z2", "unified_alloc_host_zone(const alloc_request & req, runtime_category cat, bool "
                            "kv_spill_to_host) noexcept is not defined"))
        return
    rets = re.findall(r"\breturn\s+host_zone_id::(\w+)\s*;", body)
    if rets != ["KV", "SCRATCH", "WEIGHT", "STAGING"] or len(re.findall(r"\breturn\b", body)) != 4:
        fails.append(("Z2", f"unified_alloc_host_zone returns {rets}; want KV, SCRATCH, WEIGHT, then STAGING and no "
                            "other return"))
        return
    if not nospace(body).endswith("returnhost_zone_id::STAGING;"):
        fails.append(("Z2", "STAGING is not unified_alloc_host_zone's last statement"))
    want = {
        "KV": "kv_spill_to_host||req.intent.role==alloc_role::KV",
        "SCRATCH": "req.intent.role==alloc_role::EXPERT_STAGING",
    }
    seen = {}
    for cond, b, _ in if_blocks(body):
        r = re.search(r"\breturn\s+host_zone_id::(\w+)\s*;", b)
        if r and nospace(b) == f"returnhost_zone_id::{r.group(1)};":
            seen[r.group(1)] = cond
    for zone, cond in want.items():
        if nospace(seen.get(zone, "")) != cond:
            fails.append(("Z2", f"unified_alloc_host_zone does not return {zone} exactly for `{cond}`: a CONTROL "
                                "request could land in a zone the graph boundary sweeps"))
    weight = seen.get("WEIGHT")
    if weight is None or "req.intent.constraints.persistent_owner" not in \
            [nospace(o) for o in top_level_or_operands(weight)]:
        fails.append(("Z2", "unified_alloc_host_zone's WEIGHT condition does not take a request with "
                            "persistent_owner: such an owner would fall through to STAGING"))
    if body.count("persistent_owner") != 1:
        fails.append(("Z2", "unified_alloc_host_zone reads persistent_owner outside its WEIGHT condition"))


def check_alloc_uses_choice(code: str, fails: list) -> None:
    body = definition(code, ALLOC_SIG)
    if body is None:
        fails.append(("Z3", "unified_alloc is not defined"))
        return
    picks = re.findall(r"\bhost_zone_id\s+\w+\s*=\s*([^;]*);", body)
    if body.count(ZONE_CALL) != 2 or [nospace(p) for p in picks] != [nospace(ZONE_CALL)] * 2 or \
            re.search(r"\breturn\s+host_zone_id::", body):
        fails.append(("Z3", f"unified_alloc does not choose both of its host zones through {ZONE_CALL}, or picks a "
                            "host zone of its own"))


def check_weight_never_settled(code: str, fails: list) -> None:
    body = definition(code, SETTLE_SIG)
    if body is None or "GGML_ASSERT(zone!=host_zone_id::WEIGHT&&" not in nospace(body):
        fails.append(("Z4", "host_zone_settle does not assert that WEIGHT is never settled; a persistent owner placed "
                            "there is safe only because no settle sweeps it"))


def check_settle(code: str, fails: list) -> None:
    body = definition(code, SETTLE_SIG)
    if body is None:
        fails.append(("S1", "unified_cache::host_zone_settle is not defined"))
        return
    gates = [g for g in if_blocks(body) if nospace(g[0]) == nospace(LIVE_GATE)]
    if len(gates) != 1:
        fails.append(("S1", f"host_zone_settle has {len(gates)} `if ({LIVE_GATE}) {{` blocks; want one"))
        return
    _, live, _ = gates[0]
    if body.count(SCAN + "(") != 1 or live.count(SCAN + "(") != 1:
        fails.append(("S1", f"host_zone_settle does not call {SCAN} exactly once, inside the live-row block"))
        return
    if nospace(body).count(nospace("host_zone_refusal_take(")) != 1 or nospace(TAKE_STMT) not in nospace(live):
        fails.append(("S1", f"host_zone_settle does not take its refusal report once, as `{TAKE_STMT}`, inside the "
                            "live-row block"))
        return
    take_at = nospace(live).index(nospace(TAKE_STMT))
    scans = [g for g in if_blocks(live) if nospace(g[0]) == "report.scan"]
    if len(scans) != 1 or SCAN + "(" not in scans[0][1] or nospace(live).index("if(report.scan)") < take_at:
        fails.append(("S1", "the registry scan is not inside `if (report.scan) {` after the report is taken: a refusal "
                            "past its diagnostic budgets would walk the whole registry on the submitting thread at "
                            "every graph boundary"))
        return
    call = re.search(re.escape(SCAN) + r"\s*\(([^;]*)\)\s*;", scans[0][1])
    args = [nospace(a) for a in call.group(1).split(",")] if call else []
    if len(args) != 5 or args[0] != "zone" or args[1] != "report.log_detail":
        fails.append(("S1", f"the scan is not {SCAN}(zone, report.log_detail, ...)"))
    warns = [m.start() for m in re.finditer(r"\bGGML_LOG_WARN\s*\(", live)]
    warn_blocks = [g for g in if_blocks(live) if nospace(g[0]) == "report.warn"]
    if len(warns) != 2 or len(warn_blocks) != 1 or \
            not all(warn_blocks[0][2] <= w < warn_blocks[0][2] + len(warn_blocks[0][1]) for w in warns):
        fails.append(("S1", "the refusal's WARNs are not both inside `if (report.warn) {`: an unbudgeted WARN at "
                            "every graph boundary"))
    if not nospace(live).endswith("return;"):
        fails.append(("S1", "the live-row block does not end with the refusal's `return;`: a zone with a live row "
                            "would be settled"))


def check_take(code: str, fails: list) -> None:
    body = definition(code, TAKE_SIG)
    if body is None or nospace(body) != nospace(TAKE_BODY):
        fails.append(("S2", "host_zone_refusal_take is not exactly the budgeted take (detail 4, WARN 8, a count that "
                            "stops at the WARN budget, scan = log_detail || warn || audit_active)"))


def check_header(header: str, fails: list) -> None:
    m = re.search(r"struct\s+alloc_constraints\s*\{", header)
    body = block_at(header, m.end() - 1)[0] if m else None
    if body is None or not re.search(r"\bbool\s+persistent_owner\s*=\s*false\s*;", body):
        fails.append(("Z1", "alloc_constraints has no `bool persistent_owner = false;`"))


def check(sources: dict) -> list:
    fails = []
    code = blank(sources["cache"])
    check_header(blank(sources["header"]), fails)
    check_marker_request(code, fails)
    check_zone_choice(code, fails)
    check_alloc_uses_choice(code, fails)
    check_weight_never_settled(code, fails)
    check_settle(code, fails)
    check_take(code, fails)
    return fails


def read_sources(root: Path) -> dict:
    return {k: (root / v).read_text() for k, v in FILES.items()}


def sub(pattern: str, repl: str, count: int = 1):
    def f(text: str) -> str:
        return re.sub(pattern, repl, text, count=count)

    return f


def mutants():
    return [
        ("Z1", "header", "the constraint does not exist",
         sub(r"\n\s*bool\s+persistent_owner\s*=\s*false;", "")),
        ("Z1", "cache", "the marker's request does not declare a persistent owner",
         sub(r"\n\s*req\.intent\.constraints\.persistent_owner\s*= true;", "")),
        ("Z1", "cache", "the marker's persistent owner is conditional",
         sub(r"(\n\s*)(req\.intent\.constraints\.persistent_owner\s*= true;)", r"\1if (device >= 0) { \2 }")),
        ("Z1", "cache", "the marker's request is not CONTROL",
         sub(r"(cohort_id\s*= \"cpu_job_input_marker\";)", r"\1\n        req.intent.role = alloc_role::STAGING;")),
        ("Z2", "cache", "the WEIGHT branch ignores persistent_owner",
         sub(r"\s*\|\|\s*req\.intent\.constraints\.persistent_owner", "")),
        ("Z2", "cache", "CONTROL is sent to SCRATCH",
         sub(r"if \(req\.intent\.role == alloc_role::EXPERT_STAGING\) \{",
             r"if (req.intent.role == alloc_role::EXPERT_STAGING || req.intent.role == alloc_role::CONTROL) {")),
        ("Z2", "cache", "CONTROL is sent to STAGING ahead of WEIGHT",
         sub(r"(\n(\s*)if \(req\.intent\.role == alloc_role::WEIGHT)",
             r"\n\2if (req.intent.role == alloc_role::CONTROL) {\n\2    return host_zone_id::STAGING;\n\2}\1")),
        ("Z3", "cache", "the direct-zone branch picks STAGING itself",
         sub(r"host_zone_id zone\s*= unified_alloc_host_zone\(req, cat, kv_spill_to_host\);",
             r"host_zone_id zone = host_zone_id::STAGING;")),
        ("Z4", "cache", "WEIGHT may be settled",
         sub(r"\n\s*GGML_ASSERT\(zone != host_zone_id::WEIGHT && \"WEIGHT host zone must not be reset\"\);", "")),
        ("S1", "cache", "the refusal always scans", sub(r"if \(report\.scan\) \{", r"if (true) {")),
        ("S1", "cache", "the scan is hoisted out of the report's branch",
         sub(r"(\n(\s*)const host_zone_refusal_report report\s*= host_zone_refusal_take\(zone_refusals)",
             r"\n\2(void) runtime_registry_scan_host_zone_locked(zone, false, false, audit, oldest_epoch);\1")),
        ("S1", "cache", "the WARN is unbudgeted", sub(r"if \(report\.warn\) \{", r"if (true) {")),
        ("S1", "cache", "the refusal does not return",
         sub(r"(\n\s*\}\n)(\s*)return;\n(\s*\}\n\s*\}\n\n\s*// GGML_SYCL_ZONE_RESET_AUDIT=2)", r"\1\3")),
        ("S2", "cache", "the take always scans",
         sub(r"report\.scan\s*= report\.log_detail \|\| report\.warn \|\| audit_active;", r"report.scan = true;")),
        ("S2", "cache", "the count never stops",
         sub(r"if \(index < warn_budget\) \{\n(\s*)index = refusals\.fetch_add",
             r"if (true) {\n\1index = refusals.fetch_add")),
        ("S2", "cache", "the WARN budget is unbounded",
         sub(r"constexpr int warn_budget\s*= 8;", r"constexpr int warn_budget = 1 << 30;")),
    ]


def self_test(root: Path) -> int:
    base = read_sources(root)
    failures, caught = [], 0
    with tempfile.TemporaryDirectory(prefix="nqbe-pin-") as tmp:
        for pin, key, name, mutate in mutants():
            mutated = mutate(base[key])
            if mutated == base[key]:
                failures.append(f"mutant '{name}' did not apply")
                print(f"  self-test: mutant '{name}' did not apply")
                continue
            copy = Path(tmp) / name.replace(" ", "_").replace("'", "")
            for k, rel in FILES.items():
                dst = copy / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(mutated if k == key else base[k])
            got = check(read_sources(copy))
            pins = {p for p, _ in got}
            if pin in pins:
                caught += 1
                print(f"  self-test: mutant '{name}' caught by pin {pin}")
            elif got:
                failures.append(f"mutant '{name}' failed on the wrong pin: {sorted(pins)}")
                print(f"  self-test: mutant '{name}' failed on the wrong pin: {got}")
            else:
                failures.append(f"mutant '{name}' passed")
                print(f"  self-test: mutant '{name}' PASSED (not caught)")
    clean = check(base)
    if clean:
        failures.append(f"unmodified tree fails: {clean}")
        print(f"  self-test: unmodified tree FAILS: {clean}")
    else:
        print("  self-test: unmodified tree passes")
    if failures:
        print(f"SELF-TEST FAIL: {', '.join(failures)}")
        return 1
    print(f"SELF-TEST PASS: {caught} mutants caught, unmodified tree passes")
    return 0


def main(argv: list) -> int:
    root = ROOT
    if "--root" in argv:
        root = Path(argv[argv.index("--root") + 1]).resolve()
    if "--self-test" in argv:
        rc = self_test(root)
        if rc != 0:
            return rc
    fails = check(read_sources(root))
    for pin, why in fails:
        print(f"FAIL [pin {pin}]: {why} {TAG}")
    if fails:
        return 1
    print("PASS: the CPU job input marker's word is a persistent owner in the WEIGHT host zone, which no graph "
          "boundary sweeps, and a refused host-zone boundary check walks the registry only for its first 8 refusals "
          f"per zone or an active zone audit {TAG}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
