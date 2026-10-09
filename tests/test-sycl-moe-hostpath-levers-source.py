#!/usr/bin/env python3
"""Source pins for the host-expert MoE decode levers of llama.cpp-z4kd.

Each pin below guards a change that no host test can execute, because the code that carries it
runs only inside the SYCL backend.  The decisions themselves (the GLU walk, the census classes)
are unit-tested in test-sycl-moe-decode-hostpath.cpp; these pins check that the backend still
calls them, and calls them the way the tests assume.

  G. GLU placement.  The GLU branch of ggml_sycl_op_is_planned_on_host asks
     ggml_sycl::moe_glu_input_host_produced with the ggml_sycl_glu_weight_executes_on_host
     callback, which answers ggml_sycl_weight_executes_on_host for the device it is handed.  The
     branch tests GGML_OP_GLU only (ADD_ID never reaches the function), and the function no longer
     calls ggml_sycl_tensor_depends_on_planned_host_weight, whose stale MUL_MAT_ID clause put the
     GLU of every host-expert layer on the CPU backend (19 CPU splits per graph on Qwen3.8).

  C. Wait census.  The expert-id readback (ggml_sycl_copy_ids_to_host) is timed as B1 in its own
     block; the per-token line is a GGML_LOG_WARN, so it survives the default log threshold; a CPU
     job join is timed as MOE_WAIT_JOIN and classified only inside moe_hostpath_wait_end, after the
     enabled check, so a run without GGML_SYCL_MOE_IDS_COPY_TRACE reads no tensor name.  The down
     activation gather's wait is B4; the graph-boundary flush (ggml_sycl_cpu_tg_flush_pending) opens
     with the B7 context and the hot-group flush runs under the B4b context, which are the B7 and B4b
     figures the lane's acceptance reads.

  P. pending_any.  ggml_sycl_cpu_tg_pending_any reads every slot that can hold scatter state between
     graphs: the direct-scatter list, the pending slot and its sibling (each with its previous
     buffers), the deferred secondary scatter, and the pipeline scatter ring.  Missing one lets a
     recording call reach its exit with state the exit hook does not see.

  R. Retired GGML_SYCL_PIPELINE_CPU.  ggml_check_sycl calls ggml_sycl_pipeline_cpu_warn_retired at
     init, in the settings-report block itself and not under its `if (!overrides.empty())`, so a run
     that sets only the retired variable still hears about it.  The helper latches through a
     `static const bool`, so a second backend init in the same process does not repeat the WARN.  The
     WARN and the env-var row cite llama.cpp-ytc9 and defer the direction under llama.cpp-3oju9.

WHAT THIS DOES NOT PROVE.  It reads source text with comments blanked.  It does not show which
executor a GLU ran on (that needs a GPU run) and cannot see a conditional hidden behind a macro.

NOT VACUOUS.  --self-test applies in-memory mutants to the real sources, requires each to fail,
then requires the unmodified tree to pass.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = "ggml/src/ggml-sycl/ggml-sycl.cpp"
ENV_DOC = "docs/backend/sycl-env-vars.md"


class ContractError(AssertionError):
    pass


def blank(src: str, strings: bool) -> str:
    """Blank // and /* */ comments (and string/char literal bodies when `strings`), keeping offsets."""
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
        elif src[i] in "\"'":
            q = src[i]
            j = i + 1
            while j < n and src[j] != q:
                if src[j] == "\\":
                    j += 1
                j += 1
            body = src[i + 1 : j]
            out.append(q + ("".join("\n" if c == "\n" else " " for c in body) if strings else body) + q)
            i = j + 1
        else:
            out.append(src[i])
            i += 1
    return "".join(out)


class Source:
    def __init__(self, text: str):
        self.code = blank(text, strings=False)  # comments blanked, strings kept
        self.bare = blank(text, strings=True)  # comments and strings blanked, for brace matching

    def match_close(self, open_at: int, open_ch: str, close_ch: str) -> int:
        depth = 0
        for k in range(open_at, len(self.bare)):
            c = self.bare[k]
            if c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    return k
        raise ContractError(f"unbalanced {open_ch} at offset {open_at}")

    def body(self, signature: str) -> tuple[int, str]:
        """The braced body of the one definition whose header matches `signature` (a regex)."""
        hits = [m for m in re.finditer(signature + r"\s*(?:try\s*)?\{", self.code)]
        if len(hits) != 1:
            raise ContractError(f"expected one definition matching {signature!r}, found {len(hits)}")
        open_at = hits[0].end() - 1
        close_at = self.match_close(open_at, "{", "}")
        return open_at, self.code[open_at : close_at + 1]

    def innermost_block(self, at: int) -> str:
        """The smallest braced block of `code` that contains offset `at`."""
        depth = 0
        for k in range(at, -1, -1):
            c = self.bare[k]
            if c == "}":
                depth += 1
            elif c == "{":
                if depth == 0:
                    return self.code[k : self.match_close(k, "{", "}") + 1]
                depth -= 1
        raise ContractError(f"offset {at} is not inside a block")


def squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def check_glu(src: Source) -> None:
    body_at, body = src.body(r"static bool ggml_sycl_op_is_planned_on_host\(const ggml_tensor \* op, int device\)")
    if "ggml_sycl_tensor_depends_on_planned_host_weight" in body:
        raise ContractError("G: ggml_sycl_op_is_planned_on_host calls ggml_sycl_tensor_depends_on_planned_host_weight")
    m = re.search(r"if\s*\(\s*op->op\s*==\s*GGML_OP_GLU\b", body)
    if not m:
        raise ContractError("G: no `if (op->op == GGML_OP_GLU` branch in ggml_sycl_op_is_planned_on_host")
    open_at = body.index("(", m.start())
    cond_end = _paren_end(body, open_at)
    cond = squash(body[open_at : cond_end + 1])
    want = "ggml_sycl::moe_glu_input_host_produced(op, ggml_sycl_glu_weight_executes_on_host, &device)"
    if want not in cond:
        raise ContractError(f"G: the GLU branch condition does not call {want}: {cond}")
    if cond != f"(op->op == GGML_OP_GLU && {want})":
        raise ContractError(f"G: the GLU branch condition carries more than GLU and the walk: {cond}")
    # The branch's true outcome keeps the GLU on the host (unless multi-GPU MoE), read from the branch's
    # own braced block, however long its comments are.
    brace = re.match(r"\s*\{", body[cond_end + 1 :])
    if not brace:
        raise ContractError("G: the GLU branch has no braced body")
    block_open = body_at + cond_end + 1 + brace.end() - 1
    branch = src.code[block_open : src.match_close(block_open, "{", "}") + 1]
    if not re.search(r"return\s+n04bq_tr_final\(\s*true\s*,", branch):
        raise ContractError("G: the GLU branch no longer returns n04bq_tr_final(true, ...)")

    _, cb = src.body(r"static bool ggml_sycl_glu_weight_executes_on_host\(const ggml_tensor \* weight, void \* ctx\)")
    if squash(cb) != "{ return ggml_sycl_weight_executes_on_host(weight, *static_cast<const int *>(ctx)); }":
        raise ContractError(f"G: the GLU residency callback changed: {squash(cb)}")


def _paren_end(text: str, open_at: int) -> int:
    depth = 0
    for k in range(open_at, len(text)):
        if text[k] == "(":
            depth += 1
        elif text[k] == ")":
            depth -= 1
            if depth == 0:
                return k
    raise ContractError("unbalanced (")


def check_census(src: Source) -> None:
    code = src.code
    b1 = [m.start() for m in re.finditer(r"moe_hostpath_wait_timer\s+\w+\(\s*MOE_WAIT_B1\s*\)\s*;", code)]
    if len(b1) != 1:
        raise ContractError(f"C: expected one B1 wait timer, found {len(b1)}")
    # The timer's own block is exactly {timer; readback;}: it times the expert-id readback and nothing else.
    block = squash(src.innermost_block(b1[0]))
    if not re.fullmatch(
        r"\{ moe_hostpath_wait_timer \w+\(MOE_WAIT_B1\); \w+ = ggml_sycl_copy_ids_to_host\(ctx, ids, ids_host\); \}",
        block,
    ):
        raise ContractError(f"C: the B1 timer does not scope exactly the ggml_sycl_copy_ids_to_host readback: {block[:200]}")

    warn = re.findall(r"GGML_LOG_(\w+)\(\s*\"\[MOE-HOSTPATH-WAITS\]", code)
    if warn != ["WARN"]:
        raise ContractError(f"C: the [MOE-HOSTPATH-WAITS] line must be emitted once at GGML_LOG_WARN, found {warn}")

    if "moe_hostpath_join_class(" in code:
        raise ContractError("C: ggml-sycl.cpp classifies a join itself; only moe_hostpath_wait_end may, past its check")
    joins = re.findall(r"moe_hostpath_wait_timer\s+\w+\(\s*MOE_WAIT_JOIN\s*,\s*slot\.dst_tensor\s*\)\s*;", code)
    if len(joins) != 1:
        raise ContractError(f"C: expected one MOE_WAIT_JOIN timer on slot.dst_tensor, found {len(joins)}")
    flat = squash(code)
    gather = re.findall(r"moe_hostpath_wait_timer \w+\(MOE_WAIT_B4\); sycl::event::wait\(copy_events\);", flat)
    if len(gather) != 1:
        raise ContractError(f"C: expected the down activation gather's wait timed as B4 once, found {len(gather)}")
    _, boundary = src.body(r"void ggml_sycl_cpu_tg_flush_pending\(\)")
    if not re.match(r"\{\s*moe_hostpath_wait_context\s+\w+\(\s*MOE_WAIT_B7\s*\)\s*;", boundary):
        raise ContractError("C: ggml_sycl_cpu_tg_flush_pending does not open with the B7 wait context")
    hot = re.findall(
        r"dispatch_cpu_and_scatter\(hot_entries, hot_first, pregather\); "
        r"moe_hostpath_wait_context \w+\(MOE_WAIT_B4B\); flush_pending_cpu_scatter\(\);",
        flat,
    )
    if len(hot) != 1:
        raise ContractError(f"C: expected the hot-group flush under the B4b wait context once, found {len(hot)}")
    _, end = src.body(
        r"static void moe_hostpath_wait_end\(int natural, const ggml_tensor \* joined, moe_hostpath_clock::time_point t0\)"
    )
    m = re.match(r"\{\s*if\s*\(\s*!moe_hostpath_waits_enabled\(\)\s*\)\s*\{\s*return;\s*\}", end)
    if not m:
        raise ContractError("C: moe_hostpath_wait_end does not start with the enabled check")
    if "moe_hostpath_wait_classify(" not in end[m.end() :]:
        raise ContractError("C: moe_hostpath_wait_end does not classify through moe_hostpath_wait_classify")


PENDING_TERMS = {
    "!g_cpu_tg_direct_pending_scatter.empty()",
    "g_pending_scatter.active",
    "g_pending_scatter.prev_bufs.pending",
    "g_pending_scatter_sibling.active",
    "g_pending_scatter_sibling.prev_bufs.pending",
    "ggml_sycl_pending_secondary_scatter_active()",
}


def check_pending_any(src: Source) -> None:
    _, body = src.body(r"bool ggml_sycl_cpu_tg_pending_any\(\)")
    m = re.match(r"\{\s*if\s*\(", body)
    if not m:
        raise ContractError("P: ggml_sycl_cpu_tg_pending_any does not open with the slot test")
    open_at = m.end() - 1
    close_at = _paren_end(body, open_at)
    terms = {squash(t) for t in body[open_at + 1 : close_at].split("||")}
    if terms != PENDING_TERMS:
        raise ContractError(
            f"P: pending_any's slot test reads {sorted(terms)}; missing {sorted(PENDING_TERMS - terms)}, "
            f"extra {sorted(terms - PENDING_TERMS)}"
        )
    if not re.match(r"\s*\{\s*return true;\s*\}", body[close_at + 1 :]):
        raise ContractError("P: the slot test does not return true")
    ring = squash(body[close_at + 1 :])
    for need in (
        "for (int i = 0; i < PIPELINE_SLOTS; i++)",
        "g_pipeline_scatter[i]",
        "slot.submitted.load(std::memory_order_acquire) && !slot.done.load(std::memory_order_acquire)",
    ):
        if need not in ring:
            raise ContractError(f"P: pending_any no longer reads the pipeline ring ({need})")


def check_retired(src: Source, doc: str) -> None:
    init_at, init = src.body(r"static void ggml_check_sycl\(\)")
    calls = list(re.finditer(r"\bggml_sycl_pipeline_cpu_warn_retired\(\)\s*;", init))
    if len(calls) != 1:
        raise ContractError(f"R: ggml_check_sycl must call ggml_sycl_pipeline_cpu_warn_retired once, found {len(calls)}")
    block = squash(src.innermost_block(init_at + calls[0].start()))
    if not block.startswith("{ std::string overrides;"):
        raise ContractError(f"R: the init call is not in the settings-report block itself: {block[:120]}")
    _, helper = src.body(r"static void ggml_sycl_pipeline_cpu_warn_retired\(\)")
    if not re.search(r"\bstatic\s+const\s+bool\s+set\s*=\s*\[", helper):
        raise ContractError("R: the retired WARN no longer latches through `static const bool set`")
    text = "".join(re.findall(r"\"((?:[^\"\\]|\\.)*)\"", helper))
    for need in ("GGML_SYCL_PIPELINE_CPU is retired", "llama.cpp-ytc9", "llama.cpp-3oju9"):
        if need not in text:
            raise ContractError(f"R: the retired WARN does not say {need!r}")
    if not re.search(r"GGML_LOG_WARN\(\s*\"GGML_SYCL_PIPELINE_CPU is retired", helper):
        raise ContractError("R: the retired message is not a GGML_LOG_WARN")
    rows = [line for line in doc.splitlines() if line.startswith("| ~~`GGML_SYCL_PIPELINE_CPU=1`~~")]
    if len(rows) != 1:
        raise ContractError(f"R: expected one retired GGML_SYCL_PIPELINE_CPU row in {ENV_DOC}, found {len(rows)}")
    if "llama.cpp-3oju9" not in rows[0] or "deferred under llama.cpp-z4kd" in rows[0]:
        raise ContractError("R: the env-var row must defer the direction under llama.cpp-3oju9, not z4kd")


def run(files: dict[str, str]) -> None:
    src = Source(files[BACKEND])
    check_glu(src)
    check_census(src)
    check_pending_any(src)
    check_retired(src, files[ENV_DOC])


def load() -> dict[str, str]:
    return {p: (ROOT / p).read_text() for p in (BACKEND, ENV_DOC)}


# (name, file, old, new): each must make run() fail.
MUTANTS = [
    (
        "G1 old walk restored in front of the GLU call",
        BACKEND,
        "ggml_sycl::moe_glu_input_host_produced(op, ggml_sycl_glu_weight_executes_on_host, &device)",
        "(ggml_sycl_tensor_depends_on_planned_host_weight(op, device) || "
        "ggml_sycl::moe_glu_input_host_produced(op, ggml_sycl_glu_weight_executes_on_host, &device))",
    ),
    (
        "G2 old walk replaces the GLU call",
        BACKEND,
        "ggml_sycl::moe_glu_input_host_produced(op, ggml_sycl_glu_weight_executes_on_host, &device)",
        "ggml_sycl_tensor_depends_on_planned_host_weight(op, device)",
    ),
    (
        "G3 ADD_ID back in the GLU branch",
        BACKEND,
        "if (op->op == GGML_OP_GLU &&\n",
        "if (op->op == GGML_OP_GLU && op->op != GGML_OP_ADD_ID &&\n",
    ),
    (
        "G4 callback ignores the device",
        BACKEND,
        "return ggml_sycl_weight_executes_on_host(weight, *static_cast<const int *>(ctx));",
        "return ggml_sycl_weight_executes_on_host(weight, 0);",
    ),
    (
        "G5 the GLU branch's true outcome turned false",
        BACKEND,
        'return n04bq_tr_final(true, "glu_input_host_produced");',
        'return n04bq_tr_final(false, "glu_input_host_produced");',
    ),
    (
        "C1 B1 readback recorded as B6",
        BACKEND,
        "moe_hostpath_wait_timer wait_timer(MOE_WAIT_B1);",
        "moe_hostpath_wait_timer wait_timer(MOE_WAIT_B6);",
    ),
    (
        "C2 B1 timer outside the readback block",
        BACKEND,
        "        {\n            moe_hostpath_wait_timer wait_timer(MOE_WAIT_B1);\n"
        "            ids_copied = ggml_sycl_copy_ids_to_host(ctx, ids, ids_host);\n        }\n",
        "        moe_hostpath_wait_timer wait_timer(MOE_WAIT_B1);\n        {\n"
        "            ids_copied = ggml_sycl_copy_ids_to_host(ctx, ids, ids_host);\n        }\n",
    ),
    (
        "C3 census line demoted to INFO",
        BACKEND,
        'GGML_LOG_WARN("[MOE-HOSTPATH-WAITS]',
        'GGML_LOG_INFO("[MOE-HOSTPATH-WAITS]',
    ),
    (
        "C4 join classified at the timer (strstr per flush when off)",
        BACKEND,
        "moe_hostpath_wait_timer wait_timer(MOE_WAIT_JOIN, slot.dst_tensor);",
        "moe_hostpath_wait_timer wait_timer(ggml_sycl::moe_hostpath_join_class(slot.dst_tensor->name));",
    ),
    (
        "C5 classification before the enabled check",
        BACKEND,
        "    if (!moe_hostpath_waits_enabled()) {\n        return;\n    }\n"
        "    moe_hostpath_wait_census & w = g_moe_hostpath_waits;\n"
        "    const int cls = ggml_sycl::moe_hostpath_wait_classify(",
        "    const int pre = ggml_sycl::moe_hostpath_wait_classify(natural, joined ? joined->name : nullptr, -1);\n"
        "    (void) pre;\n"
        "    if (!moe_hostpath_waits_enabled()) {\n        return;\n    }\n"
        "    moe_hostpath_wait_census & w = g_moe_hostpath_waits;\n"
        "    const int cls = ggml_sycl::moe_hostpath_wait_classify(",
    ),
    (
        "C6 down activation gather retagged B2",
        BACKEND,
        "moe_hostpath_wait_timer wait_timer(MOE_WAIT_B4);\n                sycl::event::wait(copy_events);",
        "moe_hostpath_wait_timer wait_timer(MOE_WAIT_B2);\n                sycl::event::wait(copy_events);",
    ),
    (
        "C7 boundary flush's B7 context dropped",
        BACKEND,
        "void ggml_sycl_cpu_tg_flush_pending() {\n    moe_hostpath_wait_context boundary_waits(MOE_WAIT_B7);\n",
        "void ggml_sycl_cpu_tg_flush_pending() {\n",
    ),
    (
        "C8 hot-group flush's B4b context dropped",
        BACKEND,
        "                        moe_hostpath_wait_context hot_join(MOE_WAIT_B4B);\n",
        "",
    ),
    (
        "P1 sibling slot dropped from pending_any",
        BACKEND,
        "g_pending_scatter_sibling.active || g_pending_scatter_sibling.prev_bufs.pending ||",
        "false || false ||",
    ),
    (
        "P2 secondary scatter dropped from pending_any",
        BACKEND,
        "||\n        ggml_sycl_pending_secondary_scatter_active()) {",
        ") {",
    ),
    (
        "P3 pipeline ring dropped from pending_any",
        BACKEND,
        "        if (slot.submitted.load(std::memory_order_acquire) && !slot.done.load(std::memory_order_acquire)) {\n"
        "            return true;",
        "        if (slot.submitted.load(std::memory_order_acquire) && false) {\n            return true;",
    ),
    (
        "R1 init-time retired WARN call removed",
        BACKEND,
        "            ggml_sycl_pipeline_cpu_warn_retired();\n        }\n",
        "        }\n",
    ),
    (
        "R4 retired WARN no longer latched: a second backend init would repeat it",
        BACKEND,
        "    static const bool set = [] {\n        const bool present = getenv(\"GGML_SYCL_PIPELINE_CPU\")",
        "    const bool set = [] {\n        const bool present = getenv(\"GGML_SYCL_PIPELINE_CPU\")",
    ),
    (
        "R5 init call only when another setting is non-default",
        BACKEND,
        "                GGML_LOG_WARN(\"[SYCL] non-default settings in effect: %s\\n\", overrides.c_str());\n"
        "            }\n"
        "            // A retired variable is not in sycl_env_settings, so it gets its own line here,\n"
        "            // whether or not any other setting is non-default.\n"
        "            ggml_sycl_pipeline_cpu_warn_retired();\n",
        "                GGML_LOG_WARN(\"[SYCL] non-default settings in effect: %s\\n\", overrides.c_str());\n"
        "                ggml_sycl_pipeline_cpu_warn_retired();\n"
        "            }\n",
    ),
    (
        "R2 WARN no longer cites 3oju9",
        BACKEND,
        "deferred under \"\n                \"llama.cpp-3oju9;",
        "deferred under \"\n                \"llama.cpp-z4kd;",
    ),
    (
        "R3 env-var row defers under z4kd again",
        ENV_DOC,
        "The design direction stays open under llama.cpp-3oju9:",
        "The design direction is deferred under llama.cpp-z4kd:",
    ),
]

# (name, file, old, new): harmless edits each must leave run() passing.
PROBES = [
    (
        "a 400-character comment inside the GLU branch, before the multi-GPU check",
        BACKEND,
        "        if (ggml_sycl_moe_multi_gpu_for_executor()) {\n"
        '            return n04bq_tr_final(false, "multi_gpu_moe_glu");',
        "".join("        // " + "x" * 77 + "\n" for _ in range(5))
        + "        if (ggml_sycl_moe_multi_gpu_for_executor()) {\n"
        '            return n04bq_tr_final(false, "multi_gpu_moe_glu");',
    ),
]


def self_test() -> int:
    base = load()
    failures = 0
    for name, path, old, new in PROBES:
        n = base[path].count(old)
        if n != 1:
            print(f"SELF-TEST BROKEN: probe {name!r} anchor found {n} times in {path}")
            failures += 1
            continue
        files = dict(base)
        files[path] = base[path].replace(old, new)
        try:
            run(files)
        except ContractError as e:
            print(f"SELF-TEST FAIL: harmless probe {name!r} failed the gate: {e}")
            failures += 1
            continue
        print(f"  passed probe: {name}")
    for name, path, old, new in MUTANTS:
        n = base[path].count(old)
        if n != 1:
            print(f"SELF-TEST BROKEN: mutant {name!r} anchor found {n} times in {path}")
            failures += 1
            continue
        files = dict(base)
        files[path] = base[path].replace(old, new)
        try:
            run(files)
        except ContractError as e:
            print(f"  caught {name}: {e}")
            continue
        print(f"SELF-TEST FAIL: mutant {name!r} survived")
        failures += 1
    if failures:
        return 1
    print(f"self-test: all {len(MUTANTS)} mutants caught, {len(PROBES)} harmless probe(s) passed")
    return 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        if self_test() != 0:
            return 1
    try:
        run(load())
    except ContractError as e:
        print(f"FAIL: {e}")
        return 1
    print("PASS: z4kd hostpath lever pins hold (GLU walk, wait census, pending_any, retired PIPELINE_CPU)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
