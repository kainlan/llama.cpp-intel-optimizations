#!/usr/bin/env python3
"""Source contract: where the routed-expert counts of the MoE routes are taken, keyed and printed.

Host-only: reads ggml-sycl.cpp, expert-prefetch.cpp and moe-route-counts.hpp, builds nothing, loads no model and
touches no device.

The backend counts how often each expert of each MoE block is routed (ggml-sycl/moe-route-counts.hpp, tested by
tests/test-sycl-moe-route-counts.cpp). The table is only as good as where its increments sit, and nothing but the text
shows that. This gate holds them:

  P0  the event-chained hybrid decode route and its pool-worker lambda `resolve` exist once each;
  P1  the decode count is taken in that lambda, on the CPU pool worker: `add_decode(` appears once in the backend,
      inside `resolve`, and not on the submitting path of the route;
  P2  only the gate tensor's op counts, so a (token, expert) pair counts once and not for gate, up and down: the count
      sits under `route_block >= 0`, and `route_block` is assigned once, inside the record build's gate test;
  P3  the key is the block number (ggml_sycl_tp_extract_layer_number, what the planner keys by) and not the hash
      layer id or the sequence layer: the assignment, the count's and the residency record's key;
  P4  the count precedes moe_host_slots_from_ids in the lambda, which overwrites the ids with its flags;
  P5  the count adds no wait, lock, allocation or copy: its loop calls add_decode and nothing else;
  P6  the final summary is WARN and both [MOE-ROUTE-STATS FINAL] print sites call the report;
  P7  the per-block table and the CSV are written only when GGML_SYCL_MOE_ROUTE_COUNTS_CSV is set;
  P8  the prompt count reads the prompt ids snapshot, after it is taken, on the gate tensor only, into the prompt
      counts only;
  P9  the table is sized at the hybrid init from the model's own blocks and experts, with no clamp;
  P10 the counts feed only the expert swap driver (GGML_SYCL_MOE_SWAP=1): the backend takes the table in three places
      only (the record build, the prompt helper and the swap driver), and reads a count back only in the swap driver,
      the decode and the prompt count once each;
  P11 the header is plain host code: standard includes only, no SYCL, no device allocation;
  P12 lifetime: the record holds the table by shared_ptr, the registry replaces it under its lock and is never
      destroyed;
  P14 the one counting tensor of a block is named by one predicate, true for its gate tensor and for its fused gate/up
      tensor and for no other, which both counting sites call; the count loop's bound, its range test and its
      device-or-host sense are held;
  P15 the report prints when the decode pairs have grown since the last report (a compare-exchange on one atomic), not
      once; the env-var row says so;
  P16 the summary line says `process tokens`, labels the top-N coverage as over the counted blocks with the mean slots
      each, and carries the prompt-ranked coverage right after it; the prompt-ranked figure ranks by prompt counts with
      ties to the lower expert index and sums decode counts, and is weighed by decode pairs; the comments and the
      env-var row say what each figure covers;
  P13 the summary line carries the derived model-wide hit rate, labelled as derived, right after the counted-block
      figure, computed by the one inline function with its guard for no counted block; the line's buffer holds its
      longest possible text; the env-var row of docs/backend/sycl-env-vars.md names it.

Checks run on comment-blanked text. NOT VACUOUS: --self-test runs the gate on in-memory mutants of the real sources.
Each mutant names the pin that must catch it, and is reported if it survives, if a different pin catches it, or if its
anchor no longer matches exactly once (NOT APPLIED). The self-test also proves that last report fires. ctest registers
it with --self-test.

WHAT THIS DOES NOT PROVE. It reads text. That the counts are right is the host test's job and the lead's run on a
device; that the increments cost little is the lead's before/after decode reading.
"""

from __future__ import annotations

import functools
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
PREFETCH = ROOT / "ggml/src/ggml-sycl/expert-prefetch.cpp"
HEADER = ROOT / "ggml/src/ggml-sycl/moe-route-counts.hpp"
DOCS = ROOT / "docs/backend/sycl-env-vars.md"

HYBRID_FN = "static bool ggml_sycl_moe_hybrid_decode_event_chained("
RESOLVE = "auto resolve = ["
BUILD_FN = "static ggml_sycl::moe_decode_direct_outcome ggml_sycl_moe_hybrid_decode_record_build("
INIT_FN = "static void moe_hybrid_init_once(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph) {"
PROMPT_HELPER = "static void ggml_sycl_moe_route_counts_add_prompt("
SWAP_FN = "static void ggml_sycl_run_expert_swap_claimed("
REPORT_FN = "void moe_route_counts_report_final(int device, int64_t tokens) {"
INSTALL_FN = "std::shared_ptr<moe_route_counts> moe_route_counts_install("
COUNT_GUARD = "if (record->route_counts && record->route_block >= 0) {"
BUILD_GATE_TEST = "if (ggml_sycl_moe_route_counts_tensor(src0->name)) {"
PREDICATE_FN = "static bool ggml_sycl_moe_route_counts_tensor(const char * name) {"
COUNT_FOR = "for (int64_t k = 0; k < n_ids; ++k) {"
COUNT_IDS = "const int32_t e = ids_rows[k];"
COUNT_RANGE = "if (e >= 0 && static_cast<size_t>(e) < record->n_as) {"
COUNT_SENSE = "record->host_weight[static_cast<size_t>(e)] == nullptr);"
CLAIM_FN = "bool claim_report(uint64_t decode_pairs) {"
ROUTE_BLOCK_ASSIGN = "record->route_block = ggml_sycl_tp_extract_layer_number(src0->name);"
RAW_ROUTE_BLOCK_ASSIGN = "record->route_block  = ggml_sycl_tp_extract_layer_number(src0->name);"
ADD_DECODE_CALL = "record->route_counts->add_decode(record->route_block, e,"
SLOTS_NEW = "static moe_route_counts_slot * const slots = new moe_route_counts_slot[GGML_SYCL_MAX_DEVICES];"
INSTALL_CALL = "(void) ggml_sycl::moe_route_counts_install(device, n_route_blocks, n_experts_per_layer);"
BLOCKS_MAX = "n_route_blocks = std::max(n_route_blocks, ggml_sycl_tp_extract_layer_number(src0->name) + 1);"
SNAPSHOT = re.compile(r"prompt_ids_snapshot\s*=\s*prompt_ids_entry\.host_ids;")
RAW_SNAPSHOT = "prompt_ids_snapshot            = prompt_ids_entry.host_ids;"
PROMPT_CALL = (
    "ggml_sycl_moe_route_counts_add_prompt(ctx.device, src0, prompt_ids_snapshot.data(), prompt_ids_snapshot.size());"
)
STREAM_PREPARE = "ggml_sycl_moe_stream_prepare(ctx, src0, ids, stream_call)"
FINAL_BACKEND = (
    'stats.print_summary("FINAL"); '
    "ggml_sycl::moe_route_counts_report_final(d, stats.total_tokens.load(std::memory_order_relaxed));"
)
FINAL_PREFETCH = (
    'stats.print_summary("FINAL"); '
    "moe_route_counts_report_final(d, stats.total_tokens.load(std::memory_order_relaxed));"
)
WARN_LINE = 'GGML_LOG_WARN("%s\\n", moe_route_summary_line(*table, tokens).c_str());'
ENV_NAME = "GGML_SYCL_MOE_ROUTE_COUNTS_CSV"
MODEL_LABEL = "model hit rate=%.1f%% (blocks not counted taken as all device)"
COUNTED_FIGURE = "hit rate=%.1f%% (device %llu, host %llu) "
MODEL_FORMULA = "return (hits + not_all) / (pairs + not_all);"
MODEL_GUARD = "if (counted <= 0) { return 0.0; }"
# Widest text each conversion can print: a 64-bit integer is 20 characters, an int 11, a percentage "100.0" plus its
# sign, a double the mean of resident experts (at most 2^31 here) with one decimal.
CONVERSION_WIDTH = {"%lld": 20, "%llu": 20, "%d": 11, "%.1f%%": 8, "%.1f": 14}
COVERAGE_LABEL = "top-%d coverage=%.1f%% over the counted blocks (each given the model's mean %.1f slots); "
PROMPT_LABEL = "prompt-ranked top-%d coverage=%.1f%%"
TIE_COMPARE = "return pa != pb ? pa > pb : a < b;"
ENV_GUARD = "if (path == nullptr || path[0] == '\\0') { return; }"
FORBIDDEN_IN_COUNT = re.compile(
    r"\b(?:wait|wait_and_throw|lock|lock_guard|unique_lock|mutex|malloc|calloc|new|delete|push_back|emplace_back|"
    r"resize|reserve|assign|insert|mem_copy|mem_copy_async|memcpy|submit|std::vector|std::string|std::function|"
    r"make_shared|make_unique)\b|\bsycl::"
)


class ContractError(AssertionError):
    pass


class MutantNotApplied(Exception):
    pass


_COMMENT_OR_LITERAL = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'", re.DOTALL
)


# The self-test runs the gate on a mutant of each source in turn. The unmutated sources, and what is cut from them,
# stay cached because every mutant reads them; a mutated text is used for its one check and evicted. The caches are
# small on purpose: a key holds a whole 7 MB source, so a large cache would keep hundreds of dead mutants in memory. A
# cache hit returns what the same text gave before, so each mutant is still judged on its own text.
_WHOLE_SOURCE = 1 << 20  # a text this long is a whole source or a whole stripped copy of one, not a body cut from it


def content_cache(fn):
    """Cache `fn` by its text argument in two tiers: the few whole-source texts stay hot, and the many short texts cut
    from them are held in a larger cache whose keys are each under a megabyte."""
    short = functools.lru_cache(maxsize=256)(fn)
    whole = functools.lru_cache(maxsize=8)(fn)

    @functools.wraps(fn)
    def cached(text):
        return (whole if len(text) >= _WHOLE_SOURCE else short)(text)

    return cached


@functools.lru_cache(maxsize=8)
def _blank(text: str, keep_literals: bool) -> str:
    def repl(m: re.Match) -> str:
        t = m.group(0)
        if t.startswith("//") or t.startswith("/*"):
            return " " * len(t) if "\n" not in t else re.sub(r"[^\n]", " ", t)
        if keep_literals:
            return t
        return t[0] + " " * (len(t) - 2) + t[-1] if len(t) >= 2 else t

    return _COMMENT_OR_LITERAL.sub(repl, text)


class Source:
    """One file: `code` has comments blanked (strings kept), `skel` also has literal bodies blanked, same offsets."""

    def __init__(self, raw: str):
        self.raw = raw
        self.code = _blank(raw, True)
        self.skel = _blank(raw, False)

    def block_after(self, marker: str, pin: str, start: int = 0) -> tuple[int, int]:
        """(begin, end) offsets of the `{ ... }` block opened by the first `{` after the one `marker`."""
        n = self.code.count(marker)
        if n != 1:
            raise ContractError(f"FAIL [pin {pin}]: `{marker}` appears {n} times, not once")
        at = self.code.find(marker, start)
        open_at = self.skel.find("{", at + len(marker) - 1 if marker.endswith("{") else at + len(marker))
        if open_at < 0:
            raise ContractError(f"FAIL [pin {pin}]: no block after `{marker}`")
        depth = 0
        for k in range(open_at, len(self.skel)):
            c = self.skel[k]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return open_at, k + 1
        raise ContractError(f"FAIL [pin {pin}]: the block after `{marker}` is unbalanced")

    def block_text(self, marker: str, pin: str) -> str:
        b, e = self.block_after(marker, pin)
        return squash(self.code[b:e])

    def block_in(self, outer: str, marker: str, pin: str) -> str:
        """The `{ ... }` block after `marker`, searched only inside the squashed text `outer`."""
        sq_marker = squash(marker)
        if outer.count(sq_marker) != 1:
            raise ContractError(f"FAIL [pin {pin}]: `{marker}` appears {outer.count(sq_marker)} times, not once")
        at = outer.find(sq_marker) + len(sq_marker)
        if sq_marker.endswith("{"):
            at -= 1
        return brace_text(outer, outer.find("{", at), pin)


def brace_text(text: str, open_at: int, pin: str) -> str:
    """The balanced block starting at `open_at` in already literal-light squashed code (strings may hold braces)."""
    depth = 0
    k = open_at
    n = len(text)
    while k < n:
        c = text[k]
        if c == '"':
            k += 1
            while k < n and text[k] != '"':
                k += 2 if text[k] == "\\" else 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[open_at : k + 1]
        k += 1
    raise ContractError(f"FAIL [pin {pin}]: unbalanced block")


@content_cache
def squash(text: str) -> str:
    t = re.sub(r"\s+", " ", text)
    t = re.sub(r"\( ", "(", t)
    return re.sub(r" \)", ")", t)


@functools.lru_cache(maxsize=8)
def without_private_testing(text: str) -> str:
    """`text` with the PRIVATE_TESTING arm of each `#if defined(GGML_SYCL_PRIVATE_TESTING)` blanked, newlines kept, so
    the pins read what production runs: the private device tests count into a table and take it from the registry on
    purpose. The arm after a depth-1 `#else` or `#elif` is production code and is kept."""
    out: list[str] = []
    depth = 0  # nesting of conditionals inside the conditional being read; 0 outside any private conditional
    keep = False  # inside the `#else`/`#elif` arm of a private conditional
    for line in text.split("\n"):
        stripped = line.strip()
        if depth == 0:
            if re.match(r"#\s*if\s+defined\(GGML_SYCL_PRIVATE_TESTING\)\s*$", stripped):
                depth, keep = 1, False
                out.append("")
            else:
                out.append(line)
            continue
        if re.match(r"#\s*if", stripped):
            depth += 1
        elif re.match(r"#\s*(else|elif)\b", stripped) and depth == 1:
            keep = True
            out.append("")
            continue
        elif re.match(r"#\s*endif", stripped):
            depth -= 1
            if depth == 0:
                keep = False
                out.append("")
                continue
        out.append(line if keep else "")
    return "\n".join(out)


class Sources:
    def __init__(self, backend: str, prefetch: str, header: str, docs: str):
        self.backend = Source(without_private_testing(backend))
        self.prefetch = Source(prefetch)
        self.header = Source(header)
        self.docs = docs


def need_once(hay: str, needle: str, pin: str, what: str) -> int:
    n = squash(needle)
    if hay.count(n) != 1:
        raise ContractError(f"FAIL [pin {pin}]: {what}: `{needle}` appears {hay.count(n)} times, not once")
    return hay.find(n)


def check(s: Sources) -> None:
    be = s.backend
    # P0 -------------------------------------------------------------------------------------------------------------
    hybrid = be.block_text(HYBRID_FN, "P0")
    resolve_at = need_once(hybrid, RESOLVE, "P0", "the hybrid decode route's pool-worker lambda")
    resolve = brace_text(hybrid, hybrid.find("{", resolve_at), "P0")

    # P1 -------------------------------------------------------------------------------------------------------------
    if be.code.count("add_decode(") != 1:
        raise ContractError(f"FAIL [pin P1]: `add_decode(` appears {be.code.count('add_decode(')} times in the backend")
    if "add_decode(" not in resolve:
        raise ContractError("FAIL [pin P1]: the decode count is not in the pool-worker lambda `resolve`")
    outside = hybrid.replace(resolve, "")
    if "route_counts" in outside or "add_decode" in outside:
        raise ContractError("FAIL [pin P1]: the hybrid route touches the route counts outside the pool-worker lambda")

    # P2 -------------------------------------------------------------------------------------------------------------
    guard_at = need_once(resolve, COUNT_GUARD, "P2", "the decode count's gate guard")
    count_block = brace_text(resolve, resolve.find("{", guard_at), "P2")
    if "add_decode(" not in count_block:
        raise ContractError("FAIL [pin P2]: the decode count is not under `route_block >= 0`")
    build = be.block_text(BUILD_FN, "P2")
    gate_at = need_once(build, BUILD_GATE_TEST, "P2", "the record build's gate test")
    gate_block = brace_text(build, build.find("{", gate_at), "P2")
    if gate_block.count("route_block") != 1 or "route_counts" not in gate_block:
        raise ContractError("FAIL [pin P2]: route_block and route_counts are not set together inside the gate test")
    if len(re.findall(r"\broute_block\s*=(?!=)", be.code)) != 2:  # the declaration's `= -1` and the one assignment
        raise ContractError("FAIL [pin P2]: route_block is written other than by its declaration and one assignment")

    # P3 -------------------------------------------------------------------------------------------------------------
    if squash(ROUTE_BLOCK_ASSIGN) not in gate_block:
        raise ContractError(f"FAIL [pin P3]: the block key is not `{ROUTE_BLOCK_ASSIGN}`")
    need_once(count_block, ADD_DECODE_CALL, "P3", "the count's key")
    if re.search(r"\b(?:cache_layer|seq_layer|layer|hash)\b", count_block):
        raise ContractError("FAIL [pin P3]: the count names a layer id other than the block number")
    if "set_block_resident(record->route_block," not in build:
        raise ContractError("FAIL [pin P3]: the block's resident experts are not recorded under the block number")

    # P4 -------------------------------------------------------------------------------------------------------------
    if resolve.find("add_decode(") > resolve.find("moe_host_slots_from_ids("):
        raise ContractError("FAIL [pin P4]: the count comes after moe_host_slots_from_ids, which overwrote the ids")

    # P5 -------------------------------------------------------------------------------------------------------------
    stripped = re.sub(r"\bstatic_cast<[^>]*>", "", count_block)
    bad = FORBIDDEN_IN_COUNT.search(stripped)
    if bad:
        raise ContractError(f"FAIL [pin P5]: the count loop does more than count (`{bad.group(0)}`)")

    # P6 -------------------------------------------------------------------------------------------------------------
    pf = s.prefetch
    report = pf.block_text(REPORT_FN, "P6")
    need_once(report, WARN_LINE, "P6", "the final summary line")
    if re.search(r"GGML_LOG_(?:INFO|DEBUG)", report):
        raise ContractError("FAIL [pin P6]: the report logs below WARN")
    for name, src, want in (("backend", be, FINAL_BACKEND), ("expert-prefetch.cpp", pf, FINAL_PREFETCH)):
        if squash(src.code).count(squash(want)) != 1 or src.code.count('print_summary("FINAL")') != 1:
            raise ContractError(f"FAIL [pin P6]: the [MOE-ROUTE-STATS FINAL] site in {name} does not call the report")

    # P7 -------------------------------------------------------------------------------------------------------------
    if pf.code.count(ENV_NAME) != 1 or report.count(ENV_NAME) != 1:
        raise ContractError(f"FAIL [pin P7]: {ENV_NAME} is not read once, in the report")
    guard = need_once(report, ENV_GUARD, "P7", "the CSV guard")
    for fn in ("moe_route_block_table(", "moe_route_csv(", "fopen("):
        if report.count(fn) != 1 or report.find(fn) < guard:
            raise ContractError(f"FAIL [pin P7]: `{fn}` is not used once, after the {ENV_NAME} guard")

    # P8 -------------------------------------------------------------------------------------------------------------
    helper = be.block_text(PROMPT_HELPER, "P8")
    if not helper.startswith("{ if (!src0 || !ggml_sycl_moe_route_counts_tensor(src0->name)) { return; }"):
        raise ContractError("FAIL [pin P8]: the prompt helper does not leave at once for a tensor that is not gate")
    if be.code.count("->add_prompt(") != 1 or "table->add_prompt(block, ids[i])" not in helper or "add_decode" in helper:
        raise ContractError("FAIL [pin P8]: the prompt helper does not count into the prompt counts alone")
    if "ggml_sycl_tp_extract_layer_number(src0->name)" not in helper:
        raise ContractError("FAIL [pin P8]: the prompt helper's key is not the block number")
    if "const int block = ggml_sycl_tp_extract_layer_number(src0->name); if (block < 0) { return; } for (" not in helper:
        raise ContractError("FAIL [pin P8]: the prompt helper does not leave at once for a tensor with no block number")
    if be.code.count("ggml_sycl_moe_route_counts_add_prompt(") != 2:
        raise ContractError("FAIL [pin P8]: the prompt helper is not defined once and called once")
    snaps = list(SNAPSHOT.finditer(be.code))
    if len(snaps) != 1:
        raise ContractError(f"FAIL [pin P8]: the prompt ids snapshot is taken {len(snaps)} times, not once")
    window = squash(be.code[snaps[0].start() : snaps[0].start() + 6000])
    call_at = need_once(window, PROMPT_CALL, "P8", "the prompt count's call")
    prep_at = window.find(squash(STREAM_PREPARE))
    if call_at > prep_at >= 0 or prep_at < 0:
        raise ContractError("FAIL [pin P8]: the prompt count is not between the ids snapshot and the stream prepare")

    # P9 -------------------------------------------------------------------------------------------------------------
    init = be.block_text(INIT_FN, "P9")
    need_once(init, INSTALL_CALL, "P9", "the table's install at the hybrid init")
    need_once(init, BLOCKS_MAX, "P9", "the block count from the model's expert tensors")
    if be.code.count("moe_route_counts_install(") != 1:
        raise ContractError("FAIL [pin P9]: the table is installed other than once, at the hybrid init")
    if re.search(r"\b(?:128|256|std::min)\b", INSTALL_CALL + " " + BLOCKS_MAX):
        raise ContractError("FAIL [pin P9]: the table's dimensions are clamped")

    # P10 ------------------------------------------------------------------------------------------------------------
    swap_b, swap_e = be.block_after(SWAP_FN, "P10")
    swap = be.code[swap_b:swap_e]
    if be.code.count("moe_route_counts_current(") != 3 or swap.count("moe_route_counts_current(") != 1:
        raise ContractError("FAIL [pin P10]: the backend takes the table other than in the record build, the helper "
                            "and the swap driver")
    readers = list(re.finditer(
        r"\b(?:decode_count|prompt_count|block_hits|block_misses|block_resident|moe_route_top_n_coverage\w*|"
        r"moe_route_overall_hit_rate|moe_route_mean_resident_experts|moe_route_csv|moe_route_block_table)\(",
        be.code,
    ))
    outside = [m.group(0) for m in readers if not swap_b <= m.start() < swap_e]
    if outside:
        raise ContractError(f"FAIL [pin P10]: the backend reads a route count back outside the swap driver ({outside[0]})")
    inside = sorted(m.group(0) for m in readers if swap_b <= m.start() < swap_e)
    if inside != ["decode_count(", "prompt_count("]:
        raise ContractError(f"FAIL [pin P10]: the swap driver reads other than the decode and prompt counts, once each "
                            f"({', '.join(inside) or 'none'})")

    # P11 ------------------------------------------------------------------------------------------------------------
    hdr = s.header.code
    if re.search(r'#\s*include\s*"', hdr) or re.search(r"\bsycl::|\bsycl/|malloc_device|malloc_host|unified_alloc", hdr):
        raise ContractError("FAIL [pin P11]: moe-route-counts.hpp includes a project header or reaches SYCL")
    if "#include <sycl" in hdr or "ggml.h" in hdr:
        raise ContractError("FAIL [pin P11]: moe-route-counts.hpp includes SYCL or ggml")

    # P12 ------------------------------------------------------------------------------------------------------------
    if "std::shared_ptr<ggml_sycl::moe_route_counts> route_counts;" not in squash(be.code):
        raise ContractError("FAIL [pin P12]: the hybrid record does not hold the table by shared_ptr")
    if "ggml_sycl::moe_route_counts *" in squash(be.code) or "moe_route_counts *" in s.prefetch.code:
        raise ContractError("FAIL [pin P12]: a raw pointer to the table is held")
    install = pf.block_text(INSTALL_FN, "P12")
    if "std::lock_guard<std::mutex> lock(slot->mutex);" not in install or "slot->table = fresh;" not in install:
        raise ContractError("FAIL [pin P12]: the install does not replace the table under the slot's lock")
    if squash(SLOTS_NEW) not in squash(pf.code):
        raise ContractError("FAIL [pin P12]: the table registry is destroyed at exit, where a late report reads it")

    check_model_rate(s)
    check_prompt_ranked(s)
    check_counting_tensor(s, count_block, build)
    check_report_when_grown(s, report)


def summary_format(header_code: str) -> tuple[str, int]:
    """The summary line's format text (its adjacent string literals) and the size of its buffer."""
    at = header_code.find("moe_route_summary_line(")
    body = header_code[at:]
    m = re.search(r'(?:"\[MOE-ROUTE-COUNTS\] (?:process )?tokens=(?:\\.|[^"\\])*"\s*)(?:"(?:\\.|[^"\\])*"\s*)*', body)
    buf = re.search(r"char\s+line\[(\d+)\]", body)
    if at < 0 or not m or not buf:
        raise ContractError("FAIL [pin P13]: the summary line's format or buffer is not found")
    fmt = "".join(re.findall(r'"((?:\\.|[^"\\])*)"', m.group(0)))
    return fmt, int(buf.group(1))


def check_model_rate(s: Sources) -> None:
    hdr = squash(s.header.code)
    fmt, buf = summary_format(s.header.code)
    if fmt.count(MODEL_LABEL) != 1:
        raise ContractError(f"FAIL [pin P13]: the summary line does not carry `{MODEL_LABEL}` once")
    if fmt.find(COUNTED_FIGURE) < 0 or fmt.find(MODEL_LABEL) != fmt.find(COUNTED_FIGURE) + len(COUNTED_FIGURE):
        raise ContractError("FAIL [pin P13]: the model hit rate does not follow the counted-block hit rate directly")
    if "moe_route_model_hit_rate(c)" not in hdr or hdr.count("inline double moe_route_model_hit_rate(") != 1:
        raise ContractError("FAIL [pin P13]: the model hit rate is not computed by moe_route_model_hit_rate")
    fn_at = hdr.find("inline double moe_route_model_hit_rate(")
    fn = brace_text(hdr, hdr.find("{", fn_at), "P13")
    if squash(MODEL_GUARD) not in fn or squash(MODEL_FORMULA) not in fn:
        raise ContractError("FAIL [pin P13]: the model hit rate is not (H + U*P/C) / (P + U*P/C) with a guard for C = 0")
    if "c.n_blocks() - counted" not in fn or "pairs / static_cast<double>(counted)" not in fn:
        raise ContractError("FAIL [pin P13]: the model hit rate does not take U from the blocks not counted, P/C each")
    worst = len(fmt)
    for conv in ("%.1f%%", "%lld", "%llu", "%d", "%.1f"):
        worst += fmt.count(conv) * (CONVERSION_WIDTH[conv] - len(conv))
        fmt = fmt.replace(conv, "\0" * len(conv))
    if buf < worst + 1:
        raise ContractError(f"FAIL [pin P13]: the summary buffer ({buf}) may not hold its longest line ({worst + 1})")
    row = next((ln for ln in s.docs.split("\n") if ln.startswith(f"| `{ENV_NAME}=")), "")
    if "model hit rate" not in row or "taken as all device" not in row:
        raise ContractError("FAIL [pin P13]: the env-var row does not name the model-wide hit rate and its assumption")


def check_prompt_ranked(s: Sources) -> None:
    hdr = squash(s.header.code)
    fmt, _ = summary_format(s.header.code)
    if not fmt.startswith("[MOE-ROUTE-COUNTS] process tokens=%lld "):
        raise ContractError("FAIL [pin P16]: the summary line does not label its token count as `process tokens`")
    if fmt.count(COVERAGE_LABEL) != 1:
        raise ContractError("FAIL [pin P16]: the top-N coverage is not labelled as over the counted blocks, once")
    if fmt.count(PROMPT_LABEL) != 1 or fmt.find(COVERAGE_LABEL + PROMPT_LABEL) < 0:
        raise ContractError("FAIL [pin P16]: the prompt-ranked coverage does not follow the top-N coverage directly")
    if "100.0 * moe_route_prompt_ranked_coverage_overall(c, top_n)," not in hdr:
        raise ContractError("FAIL [pin P16]: the summary does not print the overall prompt-ranked coverage at top_n")
    for fn in ("inline double moe_route_prompt_ranked_coverage(", "inline double moe_route_prompt_ranked_coverage_overall("):
        if hdr.count(fn) != 1:
            raise ContractError(f"FAIL [pin P16]: `{fn}` is not defined once")
    ranked = s.header.block_text("inline void prompt_ranked_sum(", "P16")
    by_prompt = "c.prompt_count(block, a)" in ranked and "c.prompt_count(block, b)" in ranked
    if ranked.count(squash(TIE_COMPARE)) != 1 or not by_prompt:
        raise ContractError("FAIL [pin P16]: the ranking is not by prompt count, ties to the lower expert index")
    if "top_sum += c.decode_count(block, rank[i]);" not in ranked or "sum += c.decode_count(block, e);" not in ranked:
        raise ContractError("FAIL [pin P16]: the ranked set's share is not a sum of decode counts over the decode total")
    overall = hdr[hdr.find("inline double moe_route_prompt_ranked_coverage_overall(") :]
    if "top_all += top;" not in overall or "sum_all += sum;" not in overall or "sum_all > 0" not in overall:
        raise ContractError("FAIL [pin P16]: the overall prompt-ranked coverage is not weighed by the decode pairs")
    raw = s.header.raw
    if raw.count("lower expert index") != 2:
        raise ContractError("FAIL [pin P16]: the tie-break to the lower expert index is not stated by both comments")
    flat = " ".join(raw.replace("//", " ").split())
    if "an upper bound for those blocks under equal slots, not a model-wide rate" not in flat:
        raise ContractError("FAIL [pin P16]: the coverage comment does not say it is an upper bound, not a model-wide rate")
    row = next((ln for ln in s.docs.split("\n") if ln.startswith(f"| `{ENV_NAME}=")), "")
    for need in ("`process tokens`", "prompt-ranked top-N coverage", "upper bound", "lower expert index"):
        if need not in row:
            raise ContractError(f"FAIL [pin P16]: the env-var row does not say `{need}`")


def check_counting_tensor(s: Sources, count_block: str, build: str) -> None:
    be = s.backend
    if be.code.count("ggml_sycl_moe_route_counts_tensor(") != 3:
        raise ContractError("FAIL [pin P14]: the counting-tensor predicate is not defined once and called twice")
    pred = be.block_text(PREDICATE_FN, "P14")
    if "role == MOE_TENSOR_GATE ||" not in pred or "role == MOE_TENSOR_GATE_UP;" not in pred:
        raise ContractError("FAIL [pin P14]: the predicate does not name both the gate role and the fused gate_up role")
    if re.search(r"MOE_TENSOR_(?:UP|DOWN|UNKNOWN)\b|ffn_\w+_exps", pred):
        raise ContractError("FAIL [pin P14]: the predicate admits a tensor that is neither gate nor fused gate/up")
    helper = be.block_text(PROMPT_HELPER, "P14")
    if "moe_classify_tensor(" in build or "moe_classify_tensor(" in helper:
        raise ContractError("FAIL [pin P14]: a counting site classifies the tensor itself, not through the predicate")
    for part in (COUNT_FOR, COUNT_IDS, COUNT_RANGE):
        if squash(part) not in count_block:
            raise ContractError(f"FAIL [pin P14]: the count loop lost `{part}`")
    if squash(COUNT_SENSE) not in count_block or "== nullptr" not in count_block or "!= nullptr" in count_block:
        raise ContractError("FAIL [pin P14]: a hit is not an expert whose host weight is null")


def check_report_when_grown(s: Sources, report: str) -> None:
    hdr = squash(s.header.code)
    at = hdr.find(squash(CLAIM_FN))
    if hdr.count(squash(CLAIM_FN)) != 1:
        raise ContractError("FAIL [pin P15]: claim_report(decode_pairs) is not defined once")
    claim = brace_text(hdr, hdr.find("{", at + len(squash(CLAIM_FN)) - 1), "P15")
    if "while (decode_pairs > last) {" not in claim or "compare_exchange_weak(last, decode_pairs," not in claim:
        raise ContractError("FAIL [pin P15]: claim_report is not a compare-exchange while the pairs are above the last")
    if re.search(r"\.exchange\(|reported_\b", hdr):
        raise ContractError("FAIL [pin P15]: the once-latch is back in the table")
    if "const uint64_t pairs = moe_route_decode_pairs(*table);" not in report or "table->claim_report(pairs)" not in report:
        raise ContractError("FAIL [pin P15]: the report does not claim by the decode pairs counted")
    row = next((ln for ln in s.docs.split("\n") if ln.startswith(f"| `{ENV_NAME}=")), "")
    if "grown since the table's last report" not in row:
        raise ContractError("FAIL [pin P15]: the env-var row does not say the report prints when the pairs have grown")


def self_test(raws: dict[str, str]) -> int:
    failures: list[str] = []
    n_mutants = 0

    def move_count_after_flags(text: str) -> str:
        """Cut the whole counting block out of the lambda and put it after moe_host_slots_from_ids."""
        start = text.find("        // The routed experts of this block")
        guard = text.find(COUNT_GUARD)
        if start < 0 or text.count(COUNT_GUARD) != 1 or guard < start:
            raise MutantNotApplied("the counting block is not found once")
        depth = 0
        end = -1
        for k in range(text.find("{", guard), len(text)):
            depth += text[k] == "{"
            depth -= text[k] == "}"
            if depth == 0:
                end = k + 1
                break
        block = text[start:end] + "\n"
        text = text[:start] + text[end + 1 :]
        anchor = "        tasks.reserve(host_slots.size());\n"
        if text.count(anchor) != 1:
            raise MutantNotApplied("the host slots' reserve is not found once")
        return text.replace(anchor, anchor + block, 1)

    def mutate(file: str, edits: list) -> dict[str, str]:
        out = dict(raws)
        text = out[file]
        for edit in edits:
            if callable(edit):
                text = edit(text)
                continue
            old, new = edit
            if text.count(old) != 1:
                raise MutantNotApplied(f"anchor matches {text.count(old)} times in {file}: {old[:70]!r}")
            text = text.replace(old, new, 1)
        out[file] = text
        return out

    def expect_fail(name: str, pin: str, file: str, edits: list) -> None:
        nonlocal n_mutants
        n_mutants += 1
        try:
            mutated = mutate(file, edits)
        except MutantNotApplied as e:
            failures.append(f"{name}: NOT APPLIED ({e})")
            print(f"  mutant {name}: NOT APPLIED ({e})")
            return
        try:
            check(Sources(mutated["backend"], mutated["prefetch"], mutated["header"], mutated["docs"]))
        except ContractError as e:
            if f"[pin {pin}]" not in str(e):
                failures.append(f"{name}: caught, but not by pin {pin}: {str(e)[:120]}")
                print(f"  mutant {name}: caught by the WRONG pin ({str(e)[:100]}...)")
                return
            print(f"  mutant {name}: caught (pin {pin})")
            return
        failures.append(f"{name}: survived")
        print(f"  mutant {name}: SURVIVED")

    count_loop = (
        "record->route_counts->add_decode(record->route_block, e,\n"
        "                                                     record->host_weight[static_cast<size_t>(e)] == nullptr);"
    )
    submit = "std::future<void> job = cpu_pool.submit_batch("
    flags_call = "std::vector<ggml_sycl::moe_host_slot> host_slots;"

    B = "backend"
    expect_fail("resolve-lambda-renamed", "P0", B, [(RESOLVE, "auto resolve_rows = [")])
    expect_fail("decode-count-twice", "P1", B,
                [(count_loop, count_loop + "\n                record->route_counts->add_decode(0, e, true);")])
    expect_fail("decode-count-on-submitting-thread", "P1", B,
                [(count_loop, "(void) e;"),
                 (submit, "record->route_counts->add_decode(record->route_block, 0, true);\n    " + submit)])
    expect_fail("decode-count-every-op", "P2", B, [(COUNT_GUARD, "if (record->route_counts) {")])
    expect_fail("gate-test-dropped", "P2", B,
                [(BUILD_GATE_TEST, "if (moe_classify_tensor(src0->name) != MOE_TENSOR_UNKNOWN) {")])
    expect_fail("up-tensor-counts", "P2", B, [(BUILD_GATE_TEST, "if (moe_classify_tensor(src0->name) == MOE_TENSOR_UP) {")])
    expect_fail("route-block-written-twice", "P2", B,
                [(RAW_ROUTE_BLOCK_ASSIGN,
                  RAW_ROUTE_BLOCK_ASSIGN + "\n        record->route_block = record->route_block + 0;")])
    expect_fail("key-is-hash-layer", "P3", B,
                [(RAW_ROUTE_BLOCK_ASSIGN, "record->route_block = moe_cache_layer_id(src0->name);")])
    expect_fail("count-keyed-by-cache-layer", "P3", B,
                [(ADD_DECODE_CALL, "record->route_counts->add_decode(record->cache_layer, e,")])
    expect_fail("residency-keyed-by-layer", "P3", B,
                [("set_block_resident(record->route_block,", "set_block_resident(record->layer,")])
    expect_fail("count-after-flags", "P4", B, [move_count_after_flags])
    expect_fail("count-allocates", "P5", B,
                [(count_loop, "std::vector<int> scratch(1);\n                " + count_loop)])
    expect_fail("count-waits", "P5", B, [(count_loop, "ctx.stream()->wait();\n                " + count_loop)])
    expect_fail("count-locks", "P5", B,
                [(count_loop, "std::lock_guard<std::mutex> g(unused_mutex);\n                " + count_loop)])
    expect_fail("summary-at-info", "P6", "prefetch", [(WARN_LINE, WARN_LINE.replace("GGML_LOG_WARN", "GGML_LOG_INFO"))])
    expect_fail("backend-final-site-no-report", "P6", B,
                [("ggml_sycl::moe_route_counts_report_final(d, stats.total_tokens.load(std::memory_order_relaxed));",
                  "")])
    expect_fail("prefetch-final-site-no-report", "P6", "prefetch",
                [("                moe_route_counts_report_final(d, stats.total_tokens.load(std::memory_order_relaxed));\n",
                  "")])
    expect_fail("csv-written-always", "P7", "prefetch",
                [("if (path == nullptr || path[0] == '\\0') {\n        return;\n    }", "")])
    expect_fail("csv-env-renamed", "P7", "prefetch", [(ENV_NAME, "GGML_SYCL_MOE_ROUTE_COUNTS_FILE")])
    expect_fail("prompt-counts-into-decode", "P1", B,
                [("table->add_prompt(block, ids[i]);", "table->add_decode(block, ids[i], false);")])
    expect_fail("prompt-counts-every-tensor", "P8", B,
                [("!ggml_sycl_moe_route_counts_tensor(src0->name)) {\n        return;",
                  "moe_classify_tensor(src0->name) == MOE_TENSOR_UNKNOWN) {\n        return;")])
    expect_fail("prompt-count-before-snapshot", "P8", B,
                [(PROMPT_CALL, ""), (RAW_SNAPSHOT, PROMPT_CALL + "\n        " + RAW_SNAPSHOT)])
    expect_fail("prompt-key-is-hash", "P8", B,
                [("    const int block = ggml_sycl_tp_extract_layer_number(src0->name);\n    if (block < 0) {",
                  "    const int block = moe_cache_layer_id(src0->name);\n    if (block < 0) {")])
    expect_fail("table-clamped", "P9", B,
                [(INSTALL_CALL, "(void) ggml_sycl::moe_route_counts_install(device, std::min(n_route_blocks, 128), "
                                "std::min(n_experts_per_layer, 256));")])
    expect_fail("install-gone", "P9", B, [(INSTALL_CALL, "")])
    expect_fail("blocks-from-constant", "P9", B, [(BLOCKS_MAX, "n_route_blocks = 128;")])
    expect_fail("count-read-back", "P10", B,
                [("        table->add_prompt(block, ids[i]);\n",
                  "        table->add_prompt(block, ids[i]);\n        (void) table->decode_count(block, ids[i]);\n")])
    expect_fail("swap-driver-reads-the-hits", "P10", B,
                [("table->prompt_count(b, e);", "table->block_hits(b);")])
    expect_fail("fourth-table-consumer", "P10", B,
                [("    record->route_counts = ggml_sycl::moe_route_counts_current(device);",
                  "    record->route_counts = ggml_sycl::moe_route_counts_current(device);\n"
                  "        (void) ggml_sycl::moe_route_counts_current(device);")])
    anchor = "    record->route_counts = ggml_sycl::moe_route_counts_current(device);"
    expect_fail("table-taken-in-the-else-arm-of-a-private-block", "P10", B,
                [(anchor, anchor + "\n#if defined(GGML_SYCL_PRIVATE_TESTING)\n        (void) 0;\n#else\n"
                  "        (void) ggml_sycl::moe_route_counts_current(device);\n#endif")])
    expect_fail("table-taken-in-a-nested-block-of-the-else-arm", "P10", B,
                [(anchor, anchor + "\n#if defined(GGML_SYCL_PRIVATE_TESTING)\n        (void) 0;\n#else\n#ifdef X\n"
                  "#endif\n        (void) ggml_sycl::moe_route_counts_current(device);\n#endif")])
    expect_fail("header-reaches-sycl", "P11", "header",
                [("namespace ggml_sycl {\n\nclass moe_route_counts {",
                  "namespace ggml_sycl {\n\ninline void* dev_alloc(sycl::queue& q) { return sycl::malloc_host(8, q); }\n"
                  "\nclass moe_route_counts {")])
    expect_fail("header-includes-project", "P11", "header",
                [("#include <algorithm>", '#include "common.hpp"\n#include <algorithm>')])
    expect_fail("record-holds-raw-pointer", "P12", B,
                [("std::shared_ptr<ggml_sycl::moe_route_counts> route_counts;",
                  "ggml_sycl::moe_route_counts * route_counts = nullptr;")])
    expect_fail("registry-destroyed-at-exit", "P12", "prefetch",
                [(SLOTS_NEW, "static moe_route_counts_slot slots[GGML_SYCL_MAX_DEVICES];")])
    expect_fail("install-without-lock", "P12", "prefetch",
                [("    std::lock_guard<std::mutex> lock(slot->mutex);\n    // The replaced table",
                  "    // The replaced table")])

    expect_fail("model-label-dropped", "P13", "header",
                [('"model hit rate=%.1f%% (blocks not counted taken as all device); top-%d', '"top-%d')])
    expect_fail("model-label-not-marked-derived", "P13", "header",
                [("(blocks not counted taken as all device)", "(model)")])
    expect_fail("model-rate-before-counted-figure", "P13", "header",
                [('"decode routed pairs=%llu hit rate=%.1f%% (device %llu, host %llu) "\n'
                  '                 "model hit rate=%.1f%% (blocks not counted taken as all device); ',
                  '"decode routed pairs=%llu model hit rate=%.1f%% (blocks not counted taken as all device); '
                  'hit rate=%.1f%% (device %llu, host %llu) ')])
    expect_fail("model-rate-ignores-uncounted", "P13", "header",
                [("return (hits + not_all) / (pairs + not_all);", "return hits / pairs;")])
    expect_fail("model-rate-guard-dropped", "P13", "header",
                [("    if (counted <= 0) {\n        return 0.0;\n    }\n    const double pairs  ",
                  "    const double pairs  ")])
    expect_fail("summary-buffer-too-small", "P13", "header", [("char         line[768];", "char         line[400];")])
    # The line's longest text, with the prompt-ranked figure, is 532 bytes: a buffer just below it must be caught.
    expect_fail("summary-buffer-below-the-longest-line", "P13", "header",
                [("char         line[768];", "char         line[500];")])
    expect_fail("docs-row-lacks-model-rate", "P13", "docs",
                [("(blocks not counted taken as all device)", "(blocks not counted)")])

    # The relabelled figures and the prompt-ranked coverage.
    H = "header"
    expect_fail("tokens-label-unqualified", "P16", H, [("process tokens=%lld", "tokens=%lld")])
    expect_fail("coverage-label-old", "P16", H,
                [("over the counted blocks (each given the model's mean %.1f slots); ",
                  "(a use-ranked set of the mean %.1f resident experts per block); ")])
    expect_fail("prompt-ranked-label-dropped", "P16", H,
                [("prompt-ranked top-%d coverage=%.1f%%; prompt routed", "prompt routed")])
    expect_fail("prompt-ranked-not-after-coverage", "P16", H,
                [("slots); \"", "slots); prompt routed pairs=%llu; \"")])
    expect_fail("prompt-ranked-prints-per-block-figure", "P16", H,
                [("100.0 * moe_route_prompt_ranked_coverage_overall(c, top_n),",
                  "100.0 * moe_route_prompt_ranked_coverage(c, 0, top_n),")])
    expect_fail("tie-break-reversed", "P16", H, [(TIE_COMPARE, "return pa != pb ? pa > pb : a > b;")])
    expect_fail("ranks-ascending", "P16", H, [(TIE_COMPARE, "return pa != pb ? pa < pb : a < b;")])
    expect_fail("ranks-by-decode-counts", "P16", H,
                [("const uint64_t pa = c.prompt_count(block, a);", "const uint64_t pa = c.decode_count(block, a);")])
    expect_fail("share-sums-prompt-counts", "P16", H,
                [("top_sum += c.decode_count(block, rank[i]);", "top_sum += c.prompt_count(block, rank[i]);")])
    expect_fail("overall-not-weighed-by-pairs", "P16", H,
                [("prompt_ranked_sum(c, b, n, &top, &sum);\n        top_all += top;\n        sum_all += sum;",
                  "prompt_ranked_sum(c, b, n, &top, &sum);\n        top_all += top;\n        sum_all += 1;")])
    expect_fail("tie-break-comment-dropped", "P16", H,
                [("is deterministic; a block", "is stable; a block"),
                 ("Ties in the prompt count go to the lower expert index, so", "Ties in the prompt count are kept, so")])
    expect_fail("upper-bound-comment-dropped", "P16", H,
                [("an upper bound for those blocks under equal slots, not a model-wide rate.", "the coverage.")])
    D = "docs"
    expect_fail("docs-row-lacks-process-tokens", "P16", D, [("`process tokens` (the process", "tokens (the process")])
    expect_fail("docs-row-lacks-prompt-ranked", "P16", D,
                [("(`prompt-ranked top-N coverage=...`:", "(`prompt-ranked coverage=...`:")])
    expect_fail("docs-row-lacks-upper-bound", "P16", D,
                [("an upper bound for the counted blocks", "a rate for the counted blocks")])

    # The fused gate/up tensor, the loop's bound and sense, the negative block, and the report that grows.
    fused = " || role == MOE_TENSOR_GATE_UP;"
    expect_fail("predicate-lacks-fused-role", "P14", B, [(fused, ";")])
    expect_fail("predicate-fused-by-name", "P14", B,
                [(fused, ' || std::strstr(name, "ffn_gate_up_exps") != nullptr;')])
    expect_fail("predicate-admits-down", "P14", B,
                [(fused, " || role == MOE_TENSOR_GATE_UP || role == MOE_TENSOR_DOWN;")])
    expect_fail("predicate-called-a-third-time", "P14", B,
                [(INSTALL_CALL, '(void) ggml_sycl_moe_route_counts_tensor("x");\n    ' + INSTALL_CALL)])
    expect_fail("build-site-classifies-itself", "P2", B,
                [(BUILD_GATE_TEST, "if (moe_classify_tensor(src0->name) == MOE_TENSOR_GATE) {")])
    expect_fail("count-sense-flipped", "P14", B,
                [("record->host_weight[static_cast<size_t>(e)] == nullptr);",
                  "record->host_weight[static_cast<size_t>(e)] != nullptr);")])
    expect_fail("count-loop-one-too-far", "P14", B, [(COUNT_FOR, "for (int64_t k = 0; k < n_ids + 1; ++k) {")])
    expect_fail("count-loop-starts-at-one", "P14", B, [(COUNT_FOR, "for (int64_t k = 1; k < n_ids; ++k) {")])
    expect_fail("count-range-test-dropped", "P14", B, [(COUNT_RANGE, "if (e >= 0) {")])
    expect_fail("prompt-negative-block-guard-dropped", "P8", B,
                [("    if (block < 0) {\n        return;\n    }\n    for (size_t i = 0; i < n; ++i) {",
                  "    for (size_t i = 0; i < n; ++i) {")])
    expect_fail("report-claims-by-latch", "P15", "header",
                [("while (decode_pairs > last) {", "while (decode_pairs != last) {")])
    expect_fail("report-claim-not-compare-exchange", "P15", "header",
                [("compare_exchange_weak(last, decode_pairs,", "compare_exchange_weak(last, last,")])
    expect_fail("report-claims-constant", "P15", "prefetch", [("table->claim_report(pairs)", "table->claim_report(1)")])
    expect_fail("docs-row-says-once", "P15", "docs",
                [("grown since the table's last report", "grown since the start")])

    # The NOT APPLIED report must fire on an anchor the sources do not hold, or a stale mutant would pass unseen.
    try:
        mutate(B, [("this anchor is in no source", "x")])
        failures.append("stale-anchor-not-reported")
        print("  self-test: a stale mutant anchor is NOT reported")
    except MutantNotApplied:
        print("  self-test: a stale mutant anchor is reported")

    try:
        check(Sources(raws["backend"], raws["prefetch"], raws["header"], raws["docs"]))
        print("  self-test: unmodified tree passes")
    except ContractError as e:
        print(f"  self-test: unmodified tree FAILS: {e}")
        failures.append("unmodified")

    if failures:
        print(f"SELF-TEST FAIL: {', '.join(failures)}")
        return 1
    print(f"SELF-TEST PASS: {n_mutants} mutants caught by their pins, unmodified tree passes")
    return 0


def main(argv: list[str]) -> int:
    raws = {
        "backend": BACKEND.read_text(),
        "prefetch": PREFETCH.read_text(),
        "header": HEADER.read_text(),
        "docs": DOCS.read_text(),
    }
    if "--self-test" in argv:
        return self_test(raws)
    try:
        check(Sources(raws["backend"], raws["prefetch"], raws["header"], raws["docs"]))
    except ContractError as e:
        print(e)
        return 1
    print("PASS: the routed-expert counts are taken on the pool worker's gate op, keyed by block, and reported at WARN")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
