#!/usr/bin/env python3
"""A hybrid MoE CPU-expert dispatch never silently drops host-expert rows (llama.cpp-93tw).

THE DEFECT.  Qwen1.5-MoE on the B50 with GGML_SYCL_VRAM_BUDGET_PCT=30 produced different
(coherent) text on 1 run in 4.  Only that run logged, twice,

    [MoE-CPU] Failed to allocate pinned host memory (1507328 act + 1036288 out bytes)

In ggml_sycl_mul_mat_id's dispatch_cpu_compute lambda a failed act/out staging allocation
logged an ERROR and `return result;` an invalid cpu_dispatch_result.  The callers
(dispatch_cpu_and_scatter, and the async TG join) hand that to apply_cpu_result_to_scatter,
which returns early on `!r.valid`.  Those host-expert rows were never computed and never
scattered; dst kept stale values and inference carried on.  An allocation failure became
dropped work.

Four more paths in the same family drop rows the same way, and are closed here too:

  * dispatch_cpu_compute's "Skipping expert N: no host-accessible weight" `continue` -- the
    expert gets no task, yet its scatter entry (zeroed output rows) is still published;
  * the planner CPU path (dispatch_cpu_entries_now) logs "Failed planner CPU dispatch staging
    alloc" and `return`s, leaving dst unwritten for every entry of that dispatch;
  * flush_pending_cpu_scatter / flush_pending_cpu_pipeline wrap `future.get()` and the scatter
    H2D submission in `catch (std::exception)` that only logs, so a throwing CPU worker or a
    failed scatter submission clears the pending state and loses the rows.

THE CONTRACT (comments blanked).  Each of those sites ABORTS, naming the site and the failed
request (GGML_ABORT), instead of returning, continuing or logging:

  1. dispatch_cpu_compute: the `!act_pinned || !out_pinned` block aborts and contains no return;
  2. dispatch_cpu_compute returns an INVALID result only for an empty entry list (its first
     return) and a valid one at its end -- exactly two `return result;`, the second straight
     after `result.valid = true;`;
  3. the `!host_weight` block aborts unconditionally (not only for planner-host entries) and
     does not `continue`;
  4. dispatch_cpu_and_scatter returns on an empty entry list and otherwise asserts the result
     is valid before apply_cpu_result_to_scatter; the async join does the same before either
     apply (the apply lambdas keep their own `if (!r.valid) return;` -- that is pinned by
     test-sycl-moe-cpu-activation-row-source.py and is now unreachable for a non-empty dispatch);
  5. dispatch_cpu_entries_now's staging-failure block aborts and contains no return;
  6. the catch blocks in flush_pending_cpu_scatter and flush_pending_cpu_pipeline abort.

THE MOST LIKELY CAUSE of the intermittent allocation failure (the other half of llama.cpp-93tw;
inferred from the source, supported by the GPU result alloc_err=4 in 6 of 9 runs before the fix
and 0 of 14 after, and not yet observed directly -- run with GGML_SYCL_UNIFIED_ALLOC_LIFETIME_TRACE=1
and look for [UNIFIED-ALLOC-STALE-CLAIM]).  A release marks its registry row RELEASING under g_runtime_alloc_mutex, drops the lock, frees the
physical block (host_zone_free returns it to the TLSF immediately), and only then re-locks to
erase the row.  An allocation on another thread that is handed the recycled address in that
window published its control, found the stale row, "never replaced a live pointer row" and
failed -- unified_allocate_owner then labels any post-publication failure
metadata_publication_failed (alloc_err=4), and allocate_managed_host_pinned returns false.  The
act/out staging sizes repeat every layer, so the same address is handed straight back, which is
why the window is reachable at all.  A RELEASING row whose address the allocator has just
handed out is by construction stale (its block is already free); a LIVE row at that address is
real corruption and still fails.

  7. both registration sites (arena_runtime_registry_commit and unified_alloc's registration)
     go through runtime_registry_claim_ptr_locked(ptr, report), which erases a RELEASING row,
     refuses a LIVE one, and is the only registry-presence check before the emplace at either
     site; the claim is logged under the lifetime trace (stale_claim_report), and a release
     records its thread (release_tid, "release-begin" line);
  8. flush_pending_cpu_scatter and flush_pending_cpu_pipeline assert a stream, an output buffer
     and a destination for every entry instead of skipping quietly;
  9. every expert_dispatch_entry the HYBRID branch builds (the region between the first and the
     second `std::vector<expert_dispatch_entry> cpu_entries;`) passes allow_cpu_fallback=false.
     That is what keeps dispatch_cpu_compute's allow_cpu_fallback=true arm of the `!host_weight`
     block dead: the planner CPU path (second cpu_entries, built with allow_cpu_fallback=true)
     is consumed by dispatch_cpu_entries_now, which already aborted on an unresolved weight.

WHAT THIS DOES NOT PROVE.  It reads source text.  It does not show the race is the cause of the
observed 1-in-4 failure (that needs the GPU repro: repeat the PCT=30 Qwen1.5 run serially), and
it cannot see a conditional hidden behind a macro or helper.

NOT VACUOUS.  --self-test runs the gate on in-memory mutants of the real sources and requires
each to FAIL, then requires the unmodified tree to pass.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
CACHE = ROOT / "ggml/src/ggml-sycl/unified-cache.cpp"

LAMBDA_MARKER = "auto dispatch_cpu_compute = [&]("
TAG = "llama.cpp-93tw"


class ContractError(AssertionError):
    pass


def blank_comments(src: str) -> str:
    """Replace // and /* */ comment bodies with spaces, preserving offsets and newlines."""
    out = []
    i = 0
    n = len(src)
    while i < n:
        two = src[i : i + 2]
        if two == "//":
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif two == "/*":
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join("\n" if c == "\n" else " " for c in src[i:j]))
            i = j
        elif src[i] == '"':
            j = i + 1
            while j < n and src[j] != '"':
                if src[j] == "\\":
                    j += 1
                j += 1
            out.append(src[i : j + 1])
            i = j + 1
        else:
            out.append(src[i])
            i += 1
    return "".join(out)


def brace_block_from(src: str, start: int) -> str:
    """The `{ ... }` block whose opening brace is the first `{` at or after `start`."""
    open_at = src.find("{", start)
    if open_at < 0:
        raise ContractError(f"FAIL: expected a brace block and found none ({TAG})")
    depth = 0
    for k in range(open_at, len(src)):
        c = src[k]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[open_at : k + 1]
    raise ContractError(f"FAIL: brace block is unbalanced ({TAG})")


def squash(text: str) -> str:
    return re.sub(r"\(\s+", "(", re.sub(r"\s+", " ", text))


def top_level(block: str) -> str:
    """The block's own statements: the text with every nested `{ ... }` body removed."""
    out = []
    depth = 0
    for c in block:
        if c == "{":
            depth += 1
            if depth == 1:
                out.append(c)
        elif c == "}":
            depth -= 1
            if depth == 0:
                out.append(c)
        elif depth == 1:
            out.append(c)
    return "".join(out)


def block_after(code: str, pattern: str, what: str, *, within: str | None = None) -> str:
    text = code if within is None else within
    m = re.search(pattern, text)
    if not m:
        raise ContractError(f"FAIL: {what} was not found ({TAG})")
    return brace_block_from(text, m.end())


def require_abort(block: str, what: str) -> None:
    if "GGML_ABORT(" not in block:
        raise ContractError(
            f"FAIL: {what} does not GGML_ABORT, so a failure there drops host-expert rows silently ({TAG})"
        )


def forbid(block: str, token: str, what: str) -> None:
    if re.search(token, block):
        raise ContractError(f"FAIL: {what} ({TAG})")


def check_backend(backend_src: str) -> None:
    code = blank_comments(backend_src)
    lam_at = code.find(LAMBDA_MARKER)
    if lam_at < 0 or code.count(LAMBDA_MARKER) != 1:
        raise ContractError(f"FAIL: dispatch_cpu_compute lambda not found exactly once ({TAG})")
    lam = brace_block_from(code, lam_at)
    lam_sq = squash(lam)

    # 1. the staging-failure block aborts and never returns.
    fail = block_after(lam, r"if\s*\(\s*!act_pinned\s*\|\|\s*!out_pinned\s*\)", "dispatch_cpu_compute's staging-failure block")
    require_abort(fail, "dispatch_cpu_compute's `!act_pinned || !out_pinned` block")
    forbid(fail, r"\breturn\b", "dispatch_cpu_compute's staging-failure block returns, handing an invalid result to a caller that drops it")

    # 2. an invalid result only for an empty list; a valid one at the end.
    returns = [m.start() for m in re.finditer(r"\breturn\s+result\s*;", lam_sq)]
    if len(returns) != 2:
        raise ContractError(
            f"FAIL: dispatch_cpu_compute has {len(returns)} `return result;` sites, expected 2 (the empty-list "
            f"early return and the valid one at the end) ({TAG})"
        )
    if not re.search(r"if \(entries\.empty\(\)\) \{ return result; \}", lam_sq):
        raise ContractError(f"FAIL: dispatch_cpu_compute's first return is not the empty-entries guard ({TAG})")
    if not re.search(r"result\.valid = true; return result; \}$", lam_sq):
        raise ContractError(f"FAIL: dispatch_cpu_compute does not end with `result.valid = true; return result;` ({TAG})")

    # 3. a host expert without a host weight aborts for every entry, and never `continue`s.
    nohost = block_after(lam, r"if\s*\(\s*!host_weight\s*\)", "the `!host_weight` block")
    require_abort(top_level(nohost), "the `!host_weight` block (at its own level, not only under an inner if)")
    forbid(nohost, r"\bcontinue\b", "the `!host_weight` block skips the expert with `continue` but its scatter entry is still published")

    # 4. the callers refuse an invalid result.
    d_and_s = block_after(code, r"auto dispatch_cpu_and_scatter\s*=\s*\[&\]\s*\(", "dispatch_cpu_and_scatter")
    d_sq = squash(d_and_s)
    m = re.search(
        r"if \(entries\.empty\(\)\) \{ return; \} auto r = dispatch_cpu_compute\(entries, pool_first_entry\); "
        r"GGML_ASSERT\(r\.valid &&",
        d_sq,
    )
    if not m or "apply_cpu_result_to_scatter(r)" not in d_sq[m.end():]:
        raise ContractError(
            f"FAIL: dispatch_cpu_and_scatter must return on an empty list and otherwise GGML_ASSERT(r.valid && ...) "
            f"before apply_cpu_result_to_scatter ({TAG})"
        )
    join = re.search(r"auto cpu_result = cpu_compute_future\.get\(\);\s*GGML_ASSERT\(\s*cpu_result\.valid\s*&&", code)
    if not join:
        raise ContractError(
            f"FAIL: the async TG join does not GGML_ASSERT(cpu_result.valid && ...) straight after "
            f"cpu_compute_future.get() ({TAG})"
        )

    # 5. the planner CPU path.
    planner = block_after(
        code, r"if\s*\(\s*!act_pinned\s*\|\|\s*!out_pinned\s*\|\|\s*\(need_weight_d2h\s*&&\s*!weight_pinned\)\s*\)",
        "dispatch_cpu_entries_now's staging-failure block",
    )
    require_abort(planner, "dispatch_cpu_entries_now's staging-failure block")
    forbid(planner, r"\breturn\b", "dispatch_cpu_entries_now's staging-failure block returns, leaving dst unwritten")

    # 6. the deferred flushes do not swallow a failed CPU future or scatter submission.
    for fn, tagname in (("flush_pending_cpu_scatter", "[CPU-TG]"), ("flush_pending_cpu_pipeline", "[PIPELINE-CPU]")):
        body = block_after(code, r"static void " + fn + r"\(\)\s*", fn)
        catch = block_after(body, r"catch\s*\(\s*const std::exception\s*&\s*ex\s*\)", f"the catch block of {fn}")
        require_abort(catch, f"the catch block of {fn} ({tagname})")

    # 8. the scatter flush does not skip quietly.
    scatter = block_after(code, r"static void flush_pending_cpu_scatter\(\)\s*", "flush_pending_cpu_scatter")
    sc = squash(scatter)
    if not re.search(r"GGML_ASSERT\(g_pending_scatter\.stream && g_pending_scatter\.out_pinned &&", sc):
        raise ContractError(
            f"FAIL: flush_pending_cpu_scatter does not GGML_ASSERT a stream and an output buffer; a pending scatter "
            f"without them would be skipped and its rows lost ({TAG})"
        )
    if not re.search(r"GGML_ASSERT\(entries\[i\]\.dst_device &&", sc):
        raise ContractError(
            f"FAIL: flush_pending_cpu_scatter does not GGML_ASSERT every entry's destination; a null one would be "
            f"skipped and its row lost ({TAG})"
        )
    if re.search(r"if \(!entries\[i\]\.dst_device\)", sc):
        raise ContractError(f"FAIL: flush_pending_cpu_scatter skips a destination-less entry again ({TAG})")
    if re.search(r"if \(g_pending_scatter\.stream && g_pending_scatter\.out_pinned\)", sc):
        raise ContractError(f"FAIL: flush_pending_cpu_scatter makes the scatter conditional on a stream again ({TAG})")

    # 8b. so does the (opt-in) pipeline flush.
    pipe = squash(block_after(code, r"static void flush_pending_cpu_pipeline\(\)\s*", "flush_pending_cpu_pipeline"))
    if not re.search(r"GGML_ASSERT\(g_pending_cpu_pipeline\.stream && g_pending_cpu_pipeline\.out_pinned &&", pipe):
        raise ContractError(
            f"FAIL: flush_pending_cpu_pipeline does not GGML_ASSERT a stream and an output buffer; a pending pipeline "
            f"merge without them would be skipped and its rows lost ({TAG})"
        )
    if not re.search(r"GGML_ASSERT\(e\.dst_device &&", pipe):
        raise ContractError(
            f"FAIL: flush_pending_cpu_pipeline does not GGML_ASSERT every entry's destination ({TAG})"
        )
    if re.search(r"if \(e\.dst_device\)", pipe) or re.search(
        r"if \(g_pending_cpu_pipeline\.stream && g_pending_cpu_pipeline\.out_pinned\)", pipe
    ):
        raise ContractError(f"FAIL: flush_pending_cpu_pipeline skips an entry or the merge quietly again ({TAG})")

    # 9. the hybrid branch builds only allow_cpu_fallback=false entries.
    decl = [m.start() for m in re.finditer(r"std::vector<expert_dispatch_entry>\s+cpu_entries\s*;", code)]
    if len(decl) != 2:
        raise ContractError(
            f"FAIL: expected exactly two `cpu_entries` declarations (hybrid, then planner), found {len(decl)} ({TAG})"
        )
    region = backend_src[decl[0] : decl[1]]  # comment-blanking preserves offsets; the marker is a comment
    calls = [m.start() for m in re.finditer(r"ggml_sycl_make_(secondary_)?expert_dispatch_entry\(", region)]
    if len(calls) < 2:
        raise ContractError(f"FAIL: found {len(calls)} entry constructions in the hybrid branch, expected at least 2 ({TAG})")
    for at in calls:
        depth = 0
        end = at
        for k in range(region.index("(", at), len(region)):
            if region[k] == "(":
                depth += 1
            elif region[k] == ")":
                depth -= 1
                if depth == 0:
                    end = k
                    break
        call = region[at:end]
        if "/*allow_cpu_fallback=*/false" not in call:
            raise ContractError(
                f"FAIL: a hybrid-branch expert_dispatch_entry does not pass `/*allow_cpu_fallback=*/false`: "
                f"{' '.join(call.split())[:100]!r}; dispatch_cpu_compute's allow_cpu_fallback=true arm would be live ({TAG})"
            )


def check_cache(cache_src: str) -> None:
    code = blank_comments(cache_src)
    helper = block_after(
        code,
        r"static bool runtime_registry_claim_ptr_locked\s*\(\s*void\s*\*\s*ptr\s*,\s*stale_claim_report\s*&\s*report\s*\)",
        "runtime_registry_claim_ptr_locked",
    )
    h = squash(helper)
    if not re.search(r"state\s*!=\s*runtime_alloc_state::RELEASING\s*\)\s*\{\s*return false;", h):
        raise ContractError(f"FAIL: runtime_registry_claim_ptr_locked must refuse a row that is not RELEASING ({TAG})")
    if not re.search(r"g_runtime_alloc_registry\.erase\(", h):
        raise ContractError(f"FAIL: runtime_registry_claim_ptr_locked does not erase the stale RELEASING row ({TAG})")

    commit = block_after(code, r"static bool arena_runtime_registry_commit\s*\(", "arena_runtime_registry_commit")
    c = squash(commit)
    if not re.search(r"if \(!runtime_registry_claim_ptr_locked\(ptr, stale_claim\)\) (\{ )?return false;", c):
        raise ContractError(f"FAIL: arena_runtime_registry_commit does not claim the pointer through the helper ({TAG})")
    if re.search(r"g_runtime_alloc_registry\.find\(ptr\)\s*!=\s*g_runtime_alloc_registry\.end\(\)", c):
        raise ContractError(
            f"FAIL: arena_runtime_registry_commit still rejects any row at the address, including a stale RELEASING one ({TAG})"
        )

    reg_at = code.find("rec.handle = owner_control->metadata();")
    if reg_at < 0:
        raise ContractError(f"FAIL: unified_alloc's registration site was not found ({TAG})")
    site = squash(code[reg_at : reg_at + 2500])
    if "runtime_registry_claim_ptr_locked(ptr, stale_claim)" not in site.split("g_runtime_alloc_registry.emplace(ptr, rec)")[0]:
        raise ContractError(f"FAIL: unified_alloc's registration does not claim the pointer through the helper before its emplace ({TAG})")
    if re.search(r"g_runtime_alloc_registry\.find\(ptr\)\s*==\s*g_runtime_alloc_registry\.end\(\)", site):
        raise ContractError(
            f"FAIL: unified_alloc's registration still requires NO row at the address, so a stale RELEASING row fails it ({TAG})"
        )


def check_cache_trace(cache_src: str) -> None:
    code = blank_comments(cache_src)
    if not re.search(r"struct stale_claim_report \{.*?~stale_claim_report\(\)", squash(code), re.S) or \
            "[UNIFIED-ALLOC-STALE-CLAIM]" not in code:
        raise ContractError(f"FAIL: the stale-claim report (struct + [UNIFIED-ALLOC-STALE-CLAIM] line) is gone ({TAG})")
    helper = squash(block_after(
        code, r"static bool runtime_registry_claim_ptr_locked\s*\(", "runtime_registry_claim_ptr_locked"))
    if "unified_alloc_lifetime_trace_enabled()" not in helper or "report.claimed = true" not in helper:
        raise ContractError(f"FAIL: the claim helper no longer records the stale row under the lifetime trace ({TAG})")
    rel = squash(block_after(
        code, r"static registered_release_status release_registered_allocation_owned\(\s*const alloc_metadata & requested",
        "release_registered_allocation_owned"))
    if "it->second.release_tid = alloc_trace_thread_id();" not in rel or "[UNIFIED-ALLOC-LIFE] release-begin" not in rel:
        raise ContractError(f"FAIL: the release path no longer records/logs its thread at the RELEASING point ({TAG})")


def check(backend_src: str, cache_src: str) -> None:
    check_backend(backend_src)
    check_cache(cache_src)
    check_cache_trace(cache_src)


def mutate(src: str, old: str, new: str, *, count: int = 1) -> str:
    if src.count(old) < 1:
        raise ContractError(f"self-test anchor missing: {old!r}")
    return src.replace(old, new, count)


def mutants(backend: str, cache: str):
    """(name, backend_src, cache_src); each must FAIL the gate."""
    stage_abort = 'GGML_ABORT(\n                        "[MoE-CPU] pinned host staging allocation failed'
    yield "alloc failure logs and returns again", mutate(
        backend, stage_abort, stage_abort.replace("GGML_ABORT(", "GGML_LOG_ERROR(")
    ), cache
    yield "alloc-failure block returns before the abort", mutate(
        backend, "so refuse to continue.\n", "so refuse to continue.\n                    return result;\n"
    ), cache
    yield "a third return result", mutate(
        backend,
        "if (entries.empty()) {\n                    return result;\n                }\n\n                const size_t n_cpu",
        "if (entries.empty()) {\n                    return result;\n                }\n"
        "                if (entries.size() > 4096) {\n                    return result;\n                }\n\n"
        "                const size_t n_cpu",
    ), cache
    no_host = 'GGML_ABORT(\n                            "[MoE-CPU] expert has no host-accessible weight'
    yield "skipped expert continues again", mutate(
        backend, no_host, "continue;\n                        " + no_host
    ), cache
    yield "no-host-weight only logs again", mutate(
        backend, no_host, no_host.replace("GGML_ABORT(", "GGML_LOG_WARN(")
    ), cache
    yield "dispatch_cpu_and_scatter drops the validity assert", mutate(
        backend, "GGML_ASSERT(r.valid &&", "GGML_ASSERT(true &&"
    ), cache
    yield "dispatch_cpu_and_scatter loses its empty-list return", mutate(
        backend, "if (entries.empty()) {\n                    return;\n                }\n                auto r = dispatch_cpu_compute",
        "auto r = dispatch_cpu_compute",
    ), cache
    yield "async join drops the validity assert", mutate(
        backend, "GGML_ASSERT(cpu_result.valid &&", "GGML_ASSERT(true &&"
    ), cache
    planner = 'GGML_ABORT(\n                        "[MoE] Failed planner CPU dispatch staging alloc'
    yield "planner staging failure logs again", mutate(
        backend, planner, planner.replace("GGML_ABORT(", "GGML_LOG_ERROR(")
    ), cache
    yield "planner staging failure returns again", mutate(
        backend, planner, "return;\n                    " + planner
    ), cache
    yield "deferred scatter swallows exceptions again", mutate(
        backend, 'GGML_ABORT("[CPU-TG] Deferred scatter failed', 'GGML_LOG_ERROR("[CPU-TG] Deferred scatter failed'
    ), cache
    yield "scatter flush skips a missing stream quietly again", mutate(
        backend, "GGML_ASSERT(g_pending_scatter.stream && g_pending_scatter.out_pinned &&", "GGML_ASSERT(true &&"
    ), cache
    yield "scatter flush skips a destination-less entry again", mutate(
        backend, "GGML_ASSERT(entries[i].dst_device &&", "if (!entries[i].dst_device) { i++; continue; } GGML_ASSERT(true &&"
    ), cache
    yield "pipeline flush skips a missing stream quietly again", mutate(
        backend, "GGML_ASSERT(g_pending_cpu_pipeline.stream && g_pending_cpu_pipeline.out_pinned &&", "GGML_ASSERT(true &&"
    ), cache
    yield "pipeline flush skips a destination-less entry again", mutate(
        backend, "GGML_ASSERT(e.dst_device &&", "if (!e.dst_device) { continue; } GGML_ASSERT(true &&"
    ), cache
    yield "pipeline flush guards on dst_device again", mutate(
        backend, "GGML_ASSERT(e.dst_device &&", "if (e.dst_device) GGML_ASSERT(true &&"
    ), cache
    yield "a hybrid-branch entry allows CPU fallback", mutate(
        backend, "operand.actual_layout(), operand.lease(), /*allow_cpu_fallback=*/false);",
        "operand.actual_layout(), operand.lease(), /*allow_cpu_fallback=*/true);",
    ), cache
    yield "a hybrid-branch entry omits allow_cpu_fallback (default true)", mutate(
        backend, "operand.actual_layout(), operand.lease(), /*allow_cpu_fallback=*/false);",
        "operand.actual_layout(), operand.lease());",
    ), cache
    yield "deferred pipeline merge swallows exceptions again", mutate(
        backend, 'GGML_ABORT("[PIPELINE-CPU] Deferred merge failed', 'GGML_LOG_ERROR("[PIPELINE-CPU] Deferred merge failed'
    ), cache
    yield "claim helper accepts a LIVE row", backend, mutate(
        cache, "if (it->second.state != runtime_alloc_state::RELEASING) {", "if (false) {"
    )
    yield "claim helper no longer erases the stale row", backend, mutate(
        cache, "    g_runtime_alloc_registry.erase(it);\n    return true;\n}\n\nstatic bool arena_runtime_registry_commit",
        "    return true;\n}\n\nstatic bool arena_runtime_registry_commit",
    )
    yield "unified_alloc registration is back to a bare find", backend, mutate(
        cache,
        "if (runtime_registry_claim_ptr_locked(ptr, stale_claim)) {\n                    auto inserted",
        "if (g_runtime_alloc_registry.find(ptr) == g_runtime_alloc_registry.end()) {\n                    auto inserted",
    )
    yield "claim helper stops recording the stale row", backend, mutate(
        cache, "report.claimed     = true;", "report.claimed     = false;"
    )
    yield "release stops recording its thread", backend, mutate(
        cache, "it->second.release_tid = alloc_trace_thread_id();", "it->second.release_tid = 0;"
    )
    yield "release-begin trace line is gone", backend, mutate(
        cache, "[UNIFIED-ALLOC-LIFE] release-begin", "[UNIFIED-ALLOC-LIFE] release"
    )
    yield "arena commit is back to the bare live-row check", backend, mutate(
        cache,
        "if (!runtime_registry_claim_ptr_locked(ptr, stale_claim)) {\n            return false;\n        }",
        "if (g_runtime_alloc_registry.find(ptr) != g_runtime_alloc_registry.end()) return false;",
    )


def self_test() -> int:
    backend = BACKEND.read_text()
    cache = CACHE.read_text()
    try:
        check(backend, cache)
    except ContractError as ex:
        print(f"self-test: the unmodified tree FAILS the gate: {ex}")
        return 1
    survivors = []
    n = 0
    for name, b, c in mutants(backend, cache):
        n += 1
        try:
            check(b, c)
        except ContractError:
            print(f"  mutant killed: {name}")
            continue
        survivors.append(name)
        print(f"  MUTANT SURVIVED: {name}")
    if survivors:
        print(f"self-test FAILED: {len(survivors)} of {n} mutants survived")
        return 1
    print(f"self-test OK: {n} mutants killed, unmodified tree passes")
    return 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return self_test()
    try:
        check(BACKEND.read_text(), CACHE.read_text())
    except ContractError as ex:
        print(ex)
        return 1
    print("OK: CPU-expert dispatch fails closed and registry registration tolerates a stale RELEASING row")
    return 0


if __name__ == "__main__":
    sys.exit(main())
