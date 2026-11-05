#!/usr/bin/env python3
"""Source contract for the device RED test-sycl-scratch-owner-end (docs/design/sycl-mmap-host-tier.md §7 C4; §4.4
"Every withdrawn chunk has an owner and a bounded release"; r9 m-5, r10 m-5, r11 m-5, r15 M-1, r17 M-4).

The device test is lead-run. This pin holds what a host can read:
1. the registration: the private-fixtures executable through the shared registering function (labels
   sycl;cache;mem-handle;gpu-serial, serial, skip code 77, level_zero:1, 64 MiB chunks, inside the
   `BUILD_TESTING AND NOT GGML_BACKEND_DL` block), and the design's six planted faults as registered arms with their
   cases and expected text, and no other arm;
2. the test source: the owner's end is reached only by freeing the context with ggml_backend_free, never called by
   name; each stand-in is a buffer of the host buffer type the scheduler allocates from, freed with
   ggml_backend_buffer_free; the transaction is driven through the proc-table entries llama_context resolves by name;
3. production: ggml_backend_sycl_free ends the context's SCRATCH ownership (unified_cache_host_scratch_owner_end, with
   the context's device and owner) before `delete sycl_ctx`, with the OWNER_END_REMOVED planted fault directly before
   that call and the production build keeping the call.

Each clause has a positive control: a planted edit of the tree text that the clause must reject with its message. The
production controls edit the tree as wired: when the tree does not yet make the call, the control first plants the
wiring the design names, so the clause is shown to accept it before it is shown to reject each fault.
"""

import re
import sys
from pathlib import Path

import pytest

# sycl_gate sits beside this file. `python3 -I` leaves the script's directory off sys.path, so it is appended here;
# appended rather than prepended, a file in tests/ never shadows a standard module.
sys.path.append(str(Path(__file__).resolve().parent))
from sycl_gate import ROOT, body, norm, ordered, resolve_mutants, run_without_source, strip_comments  # noqa: E402

SYCL = "ggml/src/ggml-sycl"
CMAKE = SYCL + "/CMakeLists.txt"
TEST = SYCL + "/tests/test-sycl-scratch-owner-end.cpp"
COMMON = SYCL + "/tests/scratch-return-test-common.hpp"
GS = SYCL + "/ggml-sycl.cpp"
FILES = (CMAKE, TEST, COMMON, GS)

STEM = "test-sycl-scratch-owner-end"
FUNC = "ggml_sycl_scratch_return_test"
BLOCK = "if (BUILD_TESTING AND NOT GGML_BACKEND_DL)"
CASES = ("orphaned", "empty-at-free", "two-empty", "lease-a", "lease-b")

# The design's planted faults for this test: the cases each runs, and the phrases of the design's failure text that
# the arm's expected text carries and the test prints (each phrase one run of literal text in the test's source).
ARMS = {
    "owner-end-removed": ("orphaned", (" not orphaned by the owner's end",)),
    "orphan-free-ignored": ("orphaned", ("after the stand-in's free",)),
    "owner-end-no-empty": ("empty-at-free", ("after the context is freed", " idle until pool destruction")),
    "release-no-erase": ("orphaned", ("entry(ies) after the release of chunk",)),
    "owner-end-first-only": ("two-empty", ("owner-end freed ", " of two chunks")),
    "lease-drop-no-index": ("lease-a", ("after the lease dropped", " waits for a begin")),
}

FREE_SIG = "static void ggml_backend_sycl_free(ggml_backend_t backend) {"
OWNER_END = "ggml_sycl::unified_cache_host_scratch_owner_end("
OWNER_END_ARGS = re.compile(r"ggml_sycl::unified_cache_host_scratch_owner_end\(([^,()]+(?:\(backend\))?), "
                            r"(?:sycl_ctx->host_zone_owner_id|freed_owner_id)\)")
# The device the owner's end runs on: the host buft's, where the context's notify grew SCRATCH and its transaction
# filed the chunks. SYCL_Host is one buft on reg device 0, so the context's own device walks another marked set.
HOST_BUFT_DEVICE = "ggml_sycl_notify_host_buft_device(backend)"
REMOVED = "scratch_test_mutant_on(scratch_test_mutant::OWNER_END_REMOVED)"
DELETE = "    delete sycl_ctx;\n"
# The wiring the design names, planted by the production controls when the tree does not yet make the call. The
# planted-fault block sits before the byte-identical production statement it disables.
WIRING = """\
    // The owner's end, before the context is deleted (§4.4 "Every withdrawn chunk has an owner and a bounded
    // release"): an open transaction is returned, empty marked chunks are released and the rest are orphaned.
    const int owner_end_device = ggml_sycl_notify_host_buft_device(backend);
    bool      owner_end        = freed_owner_id != 0;
#if defined(GGML_SYCL_SCRATCH_MUTANTS)
    using ggml_sycl::scratch_test_mutant;
    using ggml_sycl::scratch_test_mutant_on;
    if (scratch_test_mutant_on(scratch_test_mutant::OWNER_END_REMOVED)) {
        owner_end = false;
    }
#endif
    if (owner_end) {
        (void) ggml_sycl::unified_cache_host_scratch_owner_end(owner_end_device, freed_owner_id);
    }
"""


# ---- the tree -------------------------------------------------------------------------------------------------------

def tree_texts() -> dict:
    """Each file's raw text; a missing file is absent from the dict, so the clauses name it."""
    out = {}
    for rel in FILES:
        p = ROOT / rel
        if p.exists():
            out[rel] = p.read_text(errors="replace")
    return out


def missing(texts: dict, *rels: str) -> list:
    return [f"{rel} not found" for rel in rels if rel not in texts]


# ---- CMake ----------------------------------------------------------------------------------------------------------

def cmake_strip(text: str) -> str:
    """CMake text with `#` comments blanked (their line breaks kept); a `#` inside a quoted argument stays."""
    out, i, n, quoted = [], 0, len(text), False
    while i < n:
        c = text[i]
        if quoted:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                quoted = False
        elif c == '"':
            quoted = True
            out.append(c)
        elif c == "#":
            while i < n and text[i] != "\n":
                i += 1
            continue
        else:
            out.append(c)
        i += 1
    return "".join(out)


def cmake_args(text: str, open_at: int) -> list:
    """The arguments of the command whose `(` is at `open_at`: quoted ones as written between the quotes (escapes
    kept), unquoted ones as tokens. None when the command is not closed."""
    args, i, n = [], open_at + 1, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c == ")":
            return args
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            args.append(text[i + 1:j])
            i = j + 1
        else:
            j = i
            while j < n and not text[j].isspace() and text[j] not in "()":
                j += 1
            args.append(text[i:j])
            i = j
    return None


def registration(cmake: str, stem: str):
    """(offset, {"TIMEOUT": [...], "ENV": [...], "MUTANTS": [...]}) of the `FUNC(<stem> ...)` call, or None."""
    m = re.search(r"^[ \t]*" + FUNC + r"\s*\(\s*" + re.escape(stem) + r"\s", cmake, re.M)
    if not m:
        return None
    args = cmake_args(cmake, cmake.index("(", m.start()))
    if args is None:
        return None
    parsed, key = {"TIMEOUT": [], "ENV": [], "MUTANTS": []}, None
    for a in args[1:]:
        if a in parsed:
            key = a
        elif key is not None:
            parsed[key].append(a)
    return m.start(), parsed


def arms_of(parsed: dict):
    """The MUTANTS triples as {names: (cases, expected text)}, or None when the list is not whole triples."""
    flat = parsed["MUTANTS"]
    if len(flat) % 3:
        return None
    return {flat[i]: (flat[i + 1], flat[i + 2]) for i in range(0, len(flat), 3)}


def if_stack_at(cmake: str, offset: int) -> list:
    """The `if (...)` lines open at `offset`, outermost first."""
    stack = []
    for m in re.finditer(r"^[ \t]*(if|endif)\s*\((.*)$", cmake[:offset], re.M):
        if m.group(1) == "if":
            stack.append(("if (" + m.group(2)).strip())
        elif stack:
            stack.pop()
    return stack


def function_problems(cmake: str) -> list:
    """The registering function's contract (§M82 case (2), §M88 (2))."""
    m = re.search(r"^[ \t]*function\s*\(\s*" + FUNC + r"\s+stem\s*\)(.*?)^[ \t]*endfunction\s*\(\s*\)", cmake,
                  re.M | re.S)
    if not m:
        return [f"{CMAKE}: no function({FUNC} stem)"]
    f = m.group(1)
    fails = []
    if BLOCK not in if_stack_at(cmake, m.start()):
        fails.append(f"{FUNC} is defined outside `{BLOCK}`")
    need = {
        "target_link_libraries(${stem} PRIVATE ggml-sycl-private-fixtures": "links the private fixtures",
        '"ONEAPI_DEVICE_SELECTOR=level_zero:1"': "pins the selector to level_zero:1",
        '"GGML_SYCL_PINNED_CHUNK_MB=64"': "sets 64 MiB chunks",
        'LABELS "sycl;cache;mem-handle;gpu-serial"': "labels the base ctest",
        'LABELS "sycl;cache;mem-handle;gpu-serial;mutant"': "labels each arm",
        "COMMAND ${stem} --mutant ${_mutants} --cases ${_cases}": "runs each arm on its mutants and cases",
        'PASS_REGULAR_EXPRESSION "${_regex}"': "scores each arm by its expected text",
    }
    for text, what in need.items():
        if text not in f:
            fails.append(f"{FUNC} no longer {what}: missing {text!r}")
    for text, n in (("RUN_SERIAL TRUE", 2), ("SKIP_RETURN_CODE 77", 2), ('ENVIRONMENT "${_env}"', 2)):
        if f.count(text) != n:
            fails.append(f"{FUNC} carries {text!r} {f.count(text)} time(s), expected {n} (base and arm)")
    if re.search(r"target_link_libraries\([^)]*\bllama\b", f):
        fails.append(f"{FUNC} links libllama")
    return fails




def registration_problems(texts: dict) -> list:
    fails = missing(texts, CMAKE)
    if fails:
        return fails
    cmake = cmake_strip(texts[CMAKE])
    fails += function_problems(cmake)
    reg = registration(cmake, STEM)
    if reg is None:
        return fails + [f"{STEM} is not registered through {FUNC}"]
    at, parsed = reg
    if BLOCK not in if_stack_at(cmake, at):
        fails.append(f"{STEM} is registered outside `{BLOCK}`")
    arms = arms_of(parsed)
    if arms is None:
        return fails + [f"{STEM}'s MUTANTS list is not whole (names, cases, expected text) triples"]
    for name, (cases, phrases) in ARMS.items():
        if name not in arms:
            fails.append(f"arm {name} is not registered")
            continue
        if arms[name][0] != cases:
            fails.append(f"arm {name} runs cases {arms[name][0]!r}, expected {cases!r}")
        for phrase in phrases:
            if phrase not in arms[name][1].replace("\\\\", "\\").replace("\\", ""):
                fails.append(f"arm {name}'s expected text lacks the design's {phrase!r}")
        try:
            re.compile(arms[name][1].replace("\\\\", "\\"))
        except re.error as error:
            fails.append(f"arm {name}'s expected text is not a regular expression: {error}")
    for name in arms:
        if name not in ARMS:
            fails.append(f"arm {name} is registered but names no planted fault the design lists for {STEM}")
    return fails


CASE_FUNCS = {
    "orphaned": "int case_orphaned(checks & c, const device_fixture & f, const entry_points & e)",
    "empty-at-free": "int case_empty_at_free(checks & c, const device_fixture & f, const entry_points & e)",
    "two-empty": "int case_two_empty(checks & c, const device_fixture & f, const entry_points & e)",
    "lease": "int case_lease(checks & c, const device_fixture & f, const entry_points & e, bool stand_in_freed_first)",
}
NAMED_OWNER_END = ("host_scratch_owner_end", "unified_cache_host_scratch_trial", "->host_scratch_trial(")


def source_problems(texts: dict) -> list:
    fails = missing(texts, TEST, COMMON)
    if fails:
        return fails
    t = norm(texts[TEST])
    if '#include "scratch-return-test-common.hpp"' not in texts[TEST]:
        fails.append(f"{TEST} does not include scratch-return-test-common.hpp")
    for tok in NAMED_OWNER_END:
        if tok in t:
            fails.append(f"the test names {tok}: the owner's end and the transaction are reached only as llama_context "
                         "reaches them")
    for proc in ("ggml_backend_sycl_host_scratch_trial", "ggml_backend_sycl_notify_host_buffer_sizes"):
        if f'ggml_backend_reg_get_proc_address(reg, "{proc}")' not in t:
            fails.append(f"the test does not resolve {proc} by proc name")
        if re.search(r"(?<![\"\w])" + proc + r"\(", t):
            fails.append(f"the test calls {proc} directly rather than through the proc table")
    for case, sig in CASE_FUNCS.items():
        fn = body(t, sig)
        if not fn:
            fails.append(f"{TEST}: no {sig.split('(')[0].split()[-1]}")
            continue
        if "ggml_backend_free(ctx);" not in fn:
            fails.append(f"case {case} does not free its context with ggml_backend_free")
    stand_in = re.findall(r"ggml_backend_buft_alloc_buffer\((\w+), k_stand_in\)", t)
    if not stand_in or "ggml_backend_dev_host_buffer_type(ggml_backend_get_device(ctx))" not in t:
        fails.append("the stand-ins are not buffers of the context's host buffer type")
    if "host_zone_alloc(" in t:
        fails.append("the test allocates a SCRATCH block directly: each stand-in is a host buffer type buffer")
    if t.count("ggml_backend_buffer_free(") < 6:
        fails.append("the test does not free each stand-in with ggml_backend_buffer_free")
    run = body(t, "int run_case(const child_args & a)")
    main = body(t, "int main(int argc, char ** argv)")
    m = re.search(r"cases_main\(argc, argv, \{([^}]*)\}", main)
    listed = tuple(re.findall(r'"([a-z-]+)"', m.group(1))) if m else ()
    if listed != CASES:
        fails.append(f"main runs the cases {list(listed)}, expected {list(CASES)}")
    for case in CASES:
        if f'a.case_name == "{case}"' not in run:
            fails.append(f"run_case does not run case {case}")
    for name, (cases, phrases) in ARMS.items():
        if cases not in CASES:
            fails.append(f"arm {name} runs case {cases}, which the test does not run")
        for phrase in phrases:
            if phrase not in texts[TEST] + "\n" + texts[COMMON]:
                fails.append(f"the test prints no {phrase!r}, which arm {name} expects")
    return fails


def free_body(text: str) -> str:
    """ggml_backend_sycl_free's body, comment-stripped and whitespace-canonical."""
    return norm(body(strip_comments(text), FREE_SIG))


def production_problems(texts: dict) -> list:
    fails = missing(texts, GS)
    if fails:
        return fails
    raw = free_body(texts[GS])
    if not raw:
        return [f"{GS}: no {FREE_SIG!r}"]
    built = free_body(resolve_mutants(texts[GS]))
    if not ordered(built, OWNER_END, "delete sycl_ctx;"):
        return [f"ggml_backend_sycl_free does not end the owner's SCRATCH ({OWNER_END}) before the context is deleted"]
    if raw.count(OWNER_END) != 1:
        fails.append(f"ggml_backend_sycl_free makes {raw.count(OWNER_END)} owner-end calls, expected one")
    m = OWNER_END_ARGS.search(built)
    device = m.group(1) if m else ""
    bound = re.fullmatch(r"\w+", device) is not None and f"{device} = {HOST_BUFT_DEVICE};" in built[:m.start()]
    if not m:
        fails.append("the owner-end call does not end this context's owner")
    elif device != HOST_BUFT_DEVICE and not bound:
        fails.append(f"the owner-end call runs on {device!r}, not the host buft's device ({HOST_BUFT_DEVICE}) where "
                     "the context's transaction filed its chunks")
    at = raw.find(REMOVED)
    if at < 0:
        return fails + ["ggml_backend_sycl_free has no OWNER_END_REMOVED planted-fault block"]
    opened = raw.rfind("#if defined(GGML_SYCL_SCRATCH_MUTANTS)", 0, at)
    closed = raw.find("#endif", at)
    call = raw.find(OWNER_END, at)
    if opened < 0 or closed < 0 or raw.find("#", opened + 1, at) >= 0:
        fails.append("the OWNER_END_REMOVED test does not sit inside a GGML_SYCL_SCRATCH_MUTANTS block")
    elif call < 0 or ";" in raw[closed:call] or "#" in raw[closed + len("#endif"):call]:
        fails.append("the OWNER_END_REMOVED block does not sit directly before the owner-end call")
    if REMOVED in built:
        fails.append("the production build compiles the OWNER_END_REMOVED test")
    return fails


CLAUSES = {
    "registration": registration_problems,
    "source": source_problems,
    "production": production_problems,
}


# ---- the tree tests -------------------------------------------------------------------------------------------------

def test_sources_exist():
    texts = tree_texts()
    assert not missing(texts, *FILES), missing(texts, *FILES)


@pytest.mark.parametrize("clause", sorted(CLAUSES))
def test_clause_holds_on_the_tree(clause):
    fails = CLAUSES[clause](tree_texts())
    assert not fails, "\n".join(fails)


# ---- positive controls ----------------------------------------------------------------------------------------------

def planted(rel: str, old: str, new: str, texts: dict = None) -> dict:
    texts = dict(texts) if texts is not None else tree_texts()
    assert rel in texts, f"control precondition: {rel} not found"
    assert texts[rel].count(old) >= 1, f"control precondition: {old!r} is not in {rel}"
    texts[rel] = texts[rel].replace(old, new, 1)
    return texts


def rejects(clause: str, texts: dict, reason: str) -> None:
    fails = CLAUSES[clause](texts)
    assert any(reason in f for f in fails), f"{clause} did not reject the control with {reason!r}: {fails}"


def wired() -> dict:
    """The tree, with the design's owner-end wiring planted before `delete sycl_ctx` when the tree lacks the call."""
    texts = tree_texts()
    assert GS in texts, f"control precondition: {GS} not found"
    if OWNER_END in strip_comments(texts[GS]):
        return texts
    start = texts[GS].find(FREE_SIG)
    at = texts[GS].find(DELETE, start)
    assert 0 <= start < at, "control precondition: ggml_backend_sycl_free does not delete sycl_ctx"
    texts[GS] = texts[GS][:at] + WIRING + texts[GS][at:]
    return texts


def test_production_accepts_the_wiring():
    fails = production_problems(wired())
    assert not fails, "\n".join(fails)


def test_production_rejects_a_missing_or_late_call():
    call = "        (void) ggml_sycl::unified_cache_host_scratch_owner_end(owner_end_device, freed_owner_id);\n"
    rejects("production", planted(GS, call, "", wired()), "before the context is deleted")
    late = planted(GS, call, "", wired())
    late = planted(GS, DELETE, DELETE + "    (void) ggml_sycl::unified_cache_host_scratch_owner_end(owner_end_device, "
                                        "freed_owner_id);\n", late)
    rejects("production", late, "before the context is deleted")
    rejects("production", planted(GS, "owner_end(owner_end_device, freed_owner_id)", "owner_end(owner_end_device, 0)",
                                  wired()),
            "does not end this context's owner")
    rejects("production", planted(GS, "owner_end(owner_end_device, freed_owner_id)",
                                  "owner_end(freed_device, freed_owner_id)", wired()),
            "runs on 'freed_device', not the host buft's device")
    rejects("production", planted(GS, "owner_end_device = ggml_sycl_notify_host_buft_device(backend);",
                                  "owner_end_device = freed_device;", wired()),
            "runs on 'owner_end_device', not the host buft's device")


def test_production_rejects_a_lost_or_misplaced_block():
    block = ("#if defined(GGML_SYCL_SCRATCH_MUTANTS)\n"
             "    using ggml_sycl::scratch_test_mutant;\n"
             "    using ggml_sycl::scratch_test_mutant_on;\n"
             "    if (scratch_test_mutant_on(scratch_test_mutant::OWNER_END_REMOVED)) {\n"
             "        owner_end = false;\n    }\n#endif\n")
    rejects("production", planted(GS, block, "", wired()), "has no OWNER_END_REMOVED planted-fault block")
    rejects("production", planted(GS, block, block + "    ggml_sycl_quarantine_drain_shutdown();\n", wired()),
            "does not sit directly before the owner-end call")
    rejects("production", planted(GS, block, block.replace("#if defined(GGML_SYCL_SCRATCH_MUTANTS)\n", "")
                                  .replace("#endif\n", ""), wired()),
            "the production build compiles the OWNER_END_REMOVED test")


def test_source_rejects_a_named_owner_end_and_a_direct_trial():
    rejects("source", planted(TEST, "    ggml_backend_free(ctx);\n    c.expect(f.pool->chunk_count() == b.chunks + 1,",
                              "    (void) ggml_sycl::unified_cache_host_scratch_owner_end(0, 1);\n"
                              "    c.expect(f.pool->chunk_count() == b.chunks + 1,"),
            "the test names host_scratch_owner_end")
    rejects("source", planted(TEST, "    ggml_backend_free(ctx);\n    c.expect(f.pool->chunk_count() == b.chunks + 1,",
                              "    delete ctx;\n    c.expect(f.pool->chunk_count() == b.chunks + 1,"),
            "case orphaned does not free its context with ggml_backend_free")
    proc = '(ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_host_scratch_trial"))'
    rejects("source", planted(TEST, proc, "(&ggml_backend_sycl_host_scratch_trial)"),
            "does not resolve ggml_backend_sycl_host_scratch_trial by proc name")
    rejects("source", planted(TEST, "(void) e.trial(ctx, GGML_SYCL_HOST_SCRATCH_TRIAL_RETURN, 512);",
                              "(void) ggml_backend_sycl_host_scratch_trial(ctx, GGML_SYCL_HOST_SCRATCH_TRIAL_RETURN, "
                              "512);"),
            "calls ggml_backend_sycl_host_scratch_trial directly")


def test_source_rejects_a_direct_stand_in_and_a_lost_case():
    rejects("source", planted(TEST, "ggml_backend_buft_alloc_buffer(buft, k_stand_in) : nullptr;",
                              "nullptr;\n    (void) f.cache->host_zone_alloc(host_zone_id::SCRATCH, k_stand_in);"),
            "allocates a SCRATCH block directly")
    rejects("source", planted(TEST, '} else if (a.case_name == "two-empty") {', '} else if (a.case_name == "two") {'),
            "run_case does not run case two-empty")


def test_registration_rejects_a_moved_and_an_unlisted_arm():
    rejects("registration", planted(CMAKE, "        owner-end-first-only two-empty\n",
                                    "        owner-end-first-only orphaned\n"),
            "arm owner-end-first-only runs cases 'orphaned'")
    rejects("registration", planted(CMAKE, "        owner-end-removed orphaned\n",
                                    "        owner-end-gone orphaned\n"),
            "arm owner-end-removed is not registered")
    rejects("registration", planted(CMAKE, "        owner-end-removed orphaned\n",
                                    "        owner-end-removed,return-noop orphaned\n"),
            "arm owner-end-removed is not registered")


def test_negative_control_comments_change_nothing():
    texts = tree_texts()
    for rel in FILES:
        if rel in texts:
            texts[rel] += "\n# a comment\n" if rel.endswith(".txt") else \
                "\n// a comment: host_scratch_owner_end(0, 1); delete sycl_ctx;\n"
    for clause, check in CLAUSES.items():
        assert check(texts) == check(tree_texts()), f"{clause} changed under appended comments"


def test_missing_sources_fail_and_say_so():
    if missing(tree_texts(), *FILES):
        # The control's own copy runs in a tree without the sources, and must not run the control again.
        pytest.skip("the pinned sources are absent here: test_sources_exist reports them")
    rc, out = run_without_source(__file__)
    assert rc != 0, out
    assert "not found" in out, out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
