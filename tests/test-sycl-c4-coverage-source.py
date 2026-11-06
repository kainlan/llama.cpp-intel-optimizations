#!/usr/bin/env python3
"""Every test the design's C4 commit names is registered exactly once (docs/design/sycl-mmap-host-tier.md §7 C4).

The acceptance of C4's coverage is that `ctest -N` lists each test C4 names once. This pin states that on the CMake
files, so it runs without a configured build:

  N1. The names are read from the design, never copied: every distinct backticked `test-...` name in the C4 section,
      from its `**C4. ` heading to the `**C5. ` heading. The design's own C4 list is the one source, so a test the
      design adds or renames reaches this census without an edit here.
  N2. Each name has exactly one registration across tests/CMakeLists.txt and ggml/src/ggml-sycl/CMakeLists.txt. A
      registration counts when the ctest name it derives is the design's name or that name without its leading
      `test-`: the backend's CMakeLists names several host tests `sycl-<stem>` beside their `test-sycl-<stem>`
      executable, so `ctest -N | grep -cE ': (test-)?<stem>$'` = 1 is the one-line form of the same check, and it
      replaces the plan's literal `ctest -N | grep -c <name>` (a deviation the lead ruled). Seven names register
      without the prefix: sycl-host-zone-select, sycl-host-scratch-demand, sycl-plan-walk-sched,
      sycl-attn-host-staging-plan, unified-cache-bugs, sycl-kv-demand-idx and sycl-getrows-refusal-terminal. A
      mutant registration (`<name>-mutant-...`) is another name and does not count.
  N3. The registering helpers still derive the names this pin models. A helper whose naming changed would make N2
      count names CMake no longer produces, so each modelled helper's body is checked for the line that names its test.

A registration inside a function() body is that helper's template, not a registration, and is skipped; the helper's
call sites are the registrations. A registration through a helper this pin does not model is not counted, so the
census fails closed on it and names the test.

The dma_streamed criterion (§4.4, "The DMA staging pool in mmap mode"; §7 C4 lead step 2) is read from a mapped load's
log when one is given: `--load-log <stderr file>` requires at least one `[SYCL-PLAN] DMA staging pool: <MB> MB` line,
printed at INFO by the planner, and its MB field to read 0.0 on every such line. A missing line is a failed criterion,
not a zero. Without `--load-log` the criterion is the lead's and is reported as not read.

Positive controls run on every invocation: a copy of the CMake texts with one registration removed, one with a name
registered twice, a design with its C4 heading gone, a helper whose naming changed, and logs that miss the line or read
a non-zero pool, each of which must fail with its stated reason.
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sycl_gate import ROOT, cmake_strip, finish, gate  # noqa: E402

DESIGN = "docs/design/sycl-mmap-host-tier.md"
CMAKES = ("tests/CMakeLists.txt", "ggml/src/ggml-sycl/CMakeLists.txt")
# The six device REDs this census was written to close; N1 must find them, or the section it read is not C4's.
DEVICE_REDS = (
    "test-sycl-seg-refusal-cleanup",
    "test-sycl-file-mapped-stream-refusal",
    "test-sycl-file-mapped-bounce",
    "test-sycl-file-mapped-expert-owner",
    "test-sycl-file-mapped-resolve-abort",
    "test-sycl-pinned-meminfo-control",
)
# The count the design's C4 section named when this pin was written: 23 at rev 3aa, and one more,
# test-sycl-reserve-state-source, from the notify path's amendment. Fewer means the section was cut short.
MIN_NAMES = 24

# How each modelled helper names its ctest, and the body line that shows it still does.
#   kind "name":   the NAME argument, else the first argument (llama_test, llama_test_cmd)
#   kind "source": the NAME argument, else the first argument's file name without extension
#   kind "script": the NAME argument, else the SCRIPT argument's file name without extension
#   kind "first":  the first argument (ggml_sycl_scratch_return_test)
#   kind "prefix": "sycl-" + the first argument (ggml_sycl_host_zone_plan_test)
HELPERS = {
    "llama_build_and_test": ("source", "tests/CMakeLists.txt", "NAME ${TEST_TARGET}"),
    "llama_test": ("name", "tests/CMakeLists.txt", "NAME ${TEST_NAME}"),
    "llama_test_cmd": ("name", "tests/CMakeLists.txt", "NAME ${TEST_NAME}"),
    "llama_test_pytest": ("script", "tests/CMakeLists.txt", "get_filename_component(TEST_NAME ${LLAMA_TEST_SCRIPT}"),
    "ggml_sycl_scratch_return_test": ("first", "ggml/src/ggml-sycl/CMakeLists.txt", "add_test(NAME ${stem} COMMAND"),
    "ggml_sycl_host_zone_plan_test": ("prefix", "ggml/src/ggml-sycl/CMakeLists.txt", "add_test(NAME sycl-${_name}"),
}

LOAD_LINE_RE = re.compile(r"\[SYCL-PLAN\] DMA staging pool: ([0-9]+(?:\.[0-9]+)?) MB")


def c4_names(design: str) -> list:
    start = design.find("\n**C4. ")
    end = design.find("\n**C5. ", start + 1) if start >= 0 else -1
    if start < 0 or end < 0:
        return []
    return list(dict.fromkeys(re.findall(r"`(test-[a-z0-9-]+)`", design[start:end])))


def command_args(text: str, open_at: int):
    """The arguments of the command whose `(` is at `open_at`, quoted ones without their quotes, or None when it is not
    closed. Its own reader, not sycl_gate.cmake_args: a nested `(` (a generator expression, a shell snippet) is part of
    an argument here, so a registration that carries one is read instead of stalling the reader."""
    args, cur, depth, i, n = [], [], 0, open_at + 1, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            cur.append(text[i + 1:j])
            i = j + 1
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            if depth == 0:
                if cur:
                    args.append("".join(cur))
                return args
            depth -= 1
        if c.isspace() and depth == 0:
            if cur:
                args.append("".join(cur))
                cur = []
        else:
            cur.append(c)
        i += 1
    return None


def function_spans(cmake: str) -> list:
    return [(m.start(), m.end()) for m in re.finditer(
        r"^[ \t]*function\s*\(.*?^[ \t]*endfunction\s*\(\s*\)", cmake, re.M | re.S)]


def helper_body(cmake: str, func: str) -> str:
    m = re.search(r"^[ \t]*function\s*\(\s*" + func + r"\s.*?^[ \t]*endfunction\s*\(\s*\)", cmake, re.M | re.S)
    return m.group(0) if m else ""


def keyword(args: list, key: str):
    return args[args.index(key) + 1] if key in args and args.index(key) + 1 < len(args) else None


def stem_of(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1].split(".")[0]


def derived_name(command: str, args: list):
    if not args:
        return None
    if command == "add_test":
        name = keyword(args, "NAME")
    else:
        kind = HELPERS[command][0]
        name = keyword(args, "NAME") if kind in ("name", "source", "script") else None
        if name is None:
            if kind == "name" or kind == "first":
                name = args[0]
            elif kind == "source":
                name = stem_of(args[0])
            elif kind == "script":
                script = keyword(args, "SCRIPT")
                name = stem_of(script) if script else None
            elif kind == "prefix":
                name = "sycl-" + args[0]
    if name is None or "${" in name:
        return None
    return name


def registrations(texts: dict) -> list:
    """(ctest name, file, line) of every registration outside a function() body."""
    out = []
    commands = ("add_test",) + tuple(HELPERS)
    pattern = re.compile(r"^[ \t]*(" + "|".join(commands) + r")\s*\(", re.M)
    for rel in CMAKES:
        cmake = cmake_strip(texts.get(rel, ""))
        spans = function_spans(cmake)
        for m in pattern.finditer(cmake):
            if any(a <= m.start() < b for a, b in spans):
                continue
            args = command_args(cmake, m.end() - 1)
            name = derived_name(m.group(1), args or [])
            if name:
                out.append((name, rel, cmake.count("\n", 0, m.start()) + 1))
    return out


def census_problems(texts: dict) -> list:
    names = c4_names(texts.get(DESIGN, ""))
    if not names:
        return [f"{DESIGN}: no `**C4. ` section ending at `**C5. `, so no C4 test names were read"]
    fails = []
    if len(names) < MIN_NAMES:
        fails.append(f"{DESIGN}: the C4 section names {len(names)} test(s), at least {MIN_NAMES} expected")
    for red in DEVICE_REDS:
        if red not in names:
            fails.append(f"{DESIGN}: the C4 section does not name {red}")
    regs = registrations(texts)
    for name in names:
        accepted = (name, name[len("test-"):])
        found = [(n, rel, line) for n, rel, line in regs if n in accepted]
        if len(found) != 1:
            where = ", ".join(f"{rel}:{line} as {n}" for n, rel, line in found) or "nowhere"
            fails.append(f"{name}: {len(found)} registration(s), expected 1 ({where})")
    return fails


def helper_problems(texts: dict) -> list:
    fails = []
    for func, (_, rel, line) in HELPERS.items():
        body = helper_body(cmake_strip(texts.get(rel, "")), func)
        if not body:
            fails.append(f"{rel}: no function({func} ...), which this census models")
        elif line not in body:
            fails.append(f"{rel}: {func} no longer names its test as modelled: missing {line!r}")
    return fails


def load_log_problems(log: str) -> list:
    figures = LOAD_LINE_RE.findall(log)
    if not figures:
        return ["no `[SYCL-PLAN] DMA staging pool:` line in the load log: a missing line is a failed criterion"]
    bad = [f for f in figures if float(f) != 0.0]
    if bad:
        return [f"DMA staging pool reads {bad[0]} MB on a mapped load, expected 0.0 (dma_streamed is 0 when no host "
                f"weight is PINNED-backed)"]
    return []


def read_tree() -> dict:
    out = {}
    for rel in (DESIGN,) + CMAKES:
        p = ROOT / rel
        if p.exists():
            out[rel] = p.read_text(errors="replace")
    return out


def must_fail(problems: list, reason: str) -> None:
    assert any(reason in p for p in problems), f"the control was not rejected with {reason!r}: {problems}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--load-log", help="stderr of a mapped load run with INFO lines on (-v, or -lv 4 for llama-cli)")
    opts = ap.parse_args()

    texts = read_tree()
    for rel in (DESIGN,) + CMAKES:
        with gate(f"{rel} exists"):
            assert rel in texts, f"{rel} not found"

    with gate("N1-N2: every C4 test name is registered exactly once"):
        fails = census_problems(texts)
        assert not fails, "\n".join(fails)
    with gate("N3: the registering helpers still derive the names this census models"):
        fails = helper_problems(texts)
        assert not fails, "\n".join(fails)

    with gate("control: a copy with one registration removed fails naming it"):
        cmake = texts["ggml/src/ggml-sycl/CMakeLists.txt"]
        planted = dict(texts)
        planted["ggml/src/ggml-sycl/CMakeLists.txt"] = re.sub(
            r"^([ \t]*)ggml_sycl_host_zone_plan_test\(host-zone-select\b", r"\1# removed(host-zone-select", cmake,
            count=1, flags=re.M)
        assert planted["ggml/src/ggml-sycl/CMakeLists.txt"] != cmake, "the control's anchor did not match"
        must_fail(census_problems(planted), "test-sycl-host-zone-select: 0 registration(s), expected 1")
    with gate("control: a name registered twice fails with its count"):
        planted = dict(texts)
        planted["tests/CMakeLists.txt"] = texts["tests/CMakeLists.txt"] + (
            "\nadd_test(NAME test-sycl-mapped-loader-source COMMAND true)\n")
        must_fail(census_problems(planted), "test-sycl-mapped-loader-source: 2 registration(s), expected 1")
    with gate("control: a design with no C4 heading reads no names"):
        planted = dict(texts)
        planted[DESIGN] = texts[DESIGN].replace("\n**C4. ", "\n**C4 ")
        must_fail(census_problems(planted), "no `**C4. ` section")
    with gate("control: a helper whose naming changed fails N3"):
        rel = "ggml/src/ggml-sycl/CMakeLists.txt"
        planted = dict(texts)
        planted[rel] = texts[rel].replace("add_test(NAME sycl-${_name}", "add_test(NAME test-sycl-${_name}")
        assert planted[rel] != texts[rel], "the control's anchor did not match"
        must_fail(helper_problems(planted), "ggml_sycl_host_zone_plan_test no longer names its test")
    with gate("control: the load-log reader fails a missing line and a non-zero pool, and passes 0.0"):
        must_fail(load_log_problems("[SYCL-PLAN] KV cache: 64.0 MB\n"), "a missing line is a failed criterion")
        must_fail(load_log_problems(
            "[SYCL-PLAN] DMA staging pool: 1700.0 MB (2 x dma_streamed 850.0 MB; global max 850.0 MB)\n"),
            "reads 1700.0 MB")
        ok = "W 12.3 [SYCL-PLAN] DMA staging pool: 0.0 MB (no pinned host weights)\n"
        assert not load_log_problems(ok), load_log_problems(ok)

    if opts.load_log:
        with gate(f"dma_streamed: the mapped load's DMA staging pool reads 0.0 MB ({opts.load_log})"):
            log = Path(opts.load_log)
            assert log.exists(), f"{opts.load_log} not found"
            fails = load_log_problems(log.read_text(errors="replace"))
            assert not fails, "\n".join(fails)
    else:
        print("dma_streamed criterion not read: pass --load-log <mapped load stderr> (the lead's run)")

    finish("test-sycl-c4-coverage-source: PASS", min_checks=10)


if __name__ == "__main__":
    main()
