#!/usr/bin/env python3
"""Source contract for the device RED test-sycl-configure-refusal (docs/design/sycl-mmap-host-tier.md §7 C4; §4.4
"configure_zones refuses while a withdrawn chunk survives"; r11 m-6; r12 M-6; r13 M-4, M-5; r14 M-8; r15 M-4; r17 M-5;
§M81 (e)).

The device test is lead-run. This pin holds what a host can read:
1. the registration: the private-fixtures executable through the shared registering function (labels
   sycl;cache;mem-handle;gpu-serial, serial, skip code 77, level_zero:1, 64 MiB chunks, inside the
   `BUILD_TESTING AND NOT GGML_BACKEND_DL` block), a pinned budget bounded by GGML_SYCL_HOST_RESERVE_MB so the
   shortfall control cannot commit the host's memory, and the design's five planted faults as registered arms with
   their children and expected text, and no other arm;
2. the driver (r15 M-4): it spawns its own binary with fork and execve, each child's environment passed as envp, a
   copy of the driver's environ with GGML_SYCL_STRICT_LEASES=1 for the strict child only; nothing calls setenv,
   putenv or unsetenv; the driver touches no device; a child sets RLIMIT_CORE before it reaches a device; the abort
   children are scored as SIGABRT plus the line production prints; a VOID child makes the driver exit 77;
3. production (§4.4, r12 M-6): the pool's configure_zones scans for a withdrawn chunk and refuses by name before it
   computes or touches any zone, and the cache's configure_host_zones aborts under ggml_sycl_strict_enabled() and
   otherwise returns false at once, without touching the reserved figures.

Each clause has a positive control: a planted edit of the tree text that the clause must reject with its message. The
tree is read with plain file I/O, comment-stripped, and with every planted-fault block resolved as the production
build compiles it wherever a clause scores production code.
"""

import re
import sys
from pathlib import Path

import pytest

# sycl_gate sits beside this file. `python3 -I` leaves the script's directory off sys.path, so it is appended here;
# appended rather than prepended, a file in tests/ never shadows a standard module.
sys.path.append(str(Path(__file__).resolve().parent))
from sycl_gate import ROOT, body, norm, ordered, resolve_mutants, run_without_source  # noqa: E402

SYCL = "ggml/src/ggml-sycl"
CMAKE = SYCL + "/CMakeLists.txt"
TEST = SYCL + "/tests/test-sycl-configure-refusal.cpp"
COMMON = SYCL + "/tests/scratch-return-test-common.hpp"
HARNESS = SYCL + "/tests/host-ledger-test-harness.hpp"
POOL = SYCL + "/pinned-pool.cpp"
UC = SYCL + "/unified-cache.cpp"
FILES = (CMAKE, TEST, COMMON, HARNESS, POOL, UC)

STEM = "test-sycl-configure-refusal"
FUNC = "ggml_sycl_scratch_return_test"
BLOCK = "if (BUILD_TESTING AND NOT GGML_BACKEND_DL)"
CHILDREN = ("strict", "plain", "carrier")
# The shortfall control grows the pool to its budget; above this bound it would commit the host's memory.
RESERVE_MB_MAX = 2048

# The design's planted faults for this test: the child each runs, and the phrases of the design's failure text that
# the arm's expected text carries and the test prints (each phrase one run of literal text in the test's source).
ARMS = {
    "configure-rebuilds": ("plain", ("rebuilt over withdrawn chunk ", " before, marked set holds ")),
    "configure-direct-release": ("plain", ("entry(ies) after the release of chunk",)),
    "strict-abort-dropped": ("strict", (", expected SIGABRT",)),
    "abort-non-strict": ("plain", (", expected exit 0",)),
    "carrier-assert-removed": ("carrier", (", expected SIGABRT",)),
}

# The lines production prints that the abort children are scored on (pinned-pool.cpp).
POOL_REFUSED = "configure_zones refused: chunk"
CARRIER_LINE = "released_chunk outstanding at pool destruction"


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
    kept), unquoted ones as tokens. None when the command is not closed. The function re-expands its arguments as
    ${ARGN}, which splits a quoted argument at each unescaped `;`, so a quoted one is split there too: a semicolon in
    an expected text adds a list element and shifts every later triple, as CMake reads it."""
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
            args += re.split(r"(?<!\\);", text[i + 1:j])
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
    reserve = [e for e in parsed["ENV"] if e.startswith("GGML_SYCL_HOST_RESERVE_MB=")]
    if not reserve or not reserve[0].split("=", 1)[1].isdigit() or \
            not 0 < int(reserve[0].split("=", 1)[1]) <= RESERVE_MB_MAX:
        fails.append(f"{STEM}'s registration does not bound the pinned budget with GGML_SYCL_HOST_RESERVE_MB "
                     f"(1..{RESERVE_MB_MAX}): the shortfall control grows the pool to its budget")
    arms = arms_of(parsed)
    if arms is None:
        return fails + [f"{STEM}'s MUTANTS list is not whole (names, children, expected text) triples"]
    for name, (children, phrases) in ARMS.items():
        if name not in arms:
            fails.append(f"arm {name} is not registered")
            continue
        if arms[name][0] != children:
            fails.append(f"arm {name} runs children {arms[name][0]!r}, expected {children!r}")
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


DEVICE_TOKENS = ("ggml_backend_", "ggml_sycl_", "sycl::", "get_unified_cache", "device_fixture", "unified_cache")


def driver_problems(texts: dict) -> list:
    fails = missing(texts, TEST, COMMON, HARNESS)
    if fails:
        return fails
    t = norm(texts[TEST])
    common = norm(texts[COMMON])
    harness = norm(texts[HARNESS])
    if '#include "scratch-return-test-common.hpp"' not in texts[TEST]:
        fails.append(f"{TEST} does not include scratch-return-test-common.hpp")
    for rel, text in ((TEST, t), (COMMON, common), (HARNESS, harness)):
        for call in ("setenv(", "putenv(", "unsetenv("):
            if re.search(r"\b" + re.escape(call), text):
                fails.append(f"{rel} calls {call[:-1]}: ggml_sycl_strict_enabled() latches the environment on first "
                             "call, so a child's environment is its envp")
    run = body(harness, "inline child_run run_child(char * const * argv, char * const * envp, "
                        "std::chrono::milliseconds bound)")
    if not ordered(run, "fork()", 'execve("/proc/self/exe", argv, envp)'):
        fails.append("run_child does not fork and execve its child with the envp it is given")
    build = body(t, "void child_env_build(child_env & e, const std::vector<std::string> & extra)")
    if "environ" not in build or '"GGML_SYCL_STRICT_LEASES=", 24' not in build:
        fails.append("child_env_build does not copy the driver's environ without GGML_SYCL_STRICT_LEASES")
    if t.count('"GGML_SYCL_STRICT_LEASES=1"') != 1:
        fails.append(f"GGML_SYCL_STRICT_LEASES=1 is set {t.count(chr(34) + 'GGML_SYCL_STRICT_LEASES=1' + chr(34))} "
                     "time(s) in the test, expected once, for the strict child")
    drive = body(t, "int drive(const char * exe, const std::vector<const char *> & children, "
                    "const std::string & mutants)")
    if "run_child(argv.data(), envp, bound)" not in drive or \
            "child_env_build(strict_env, { \"GGML_SYCL_STRICT_LEASES=1\"" not in drive:
        fails.append("drive does not pass each child the envp it built, with the strict variable for the strict child")
    main = body(t, "int main(int argc, char ** argv)")
    for where, text in (("drive", drive), ("main", main)):
        if not text:
            fails.append(f"{TEST}: no {where}")
        for tok in DEVICE_TOKENS:
            if tok in text:
                fails.append(f"the driver touches a device: {where} names {tok}")
    m = re.search(r"cases_select\(\{([^}]*)\}, a\.cases", main)
    listed = tuple(re.findall(r'"([a-z-]+)"', m.group(1))) if m else ()
    if listed != CHILDREN:
        fails.append(f"main runs the children {list(listed)}, expected {list(CHILDREN)}")
    if "setrlimit(RLIMIT_CORE" not in body(common, "inline bool child_begin(const child_args & a)"):
        fails.append("child_begin does not set RLIMIT_CORE")
    child = body(t, "int run_child_case(const child_args & a)")
    if not ordered(child, "child_begin(a)", "entry_environment_ok(c)", "device_fixture_open(c, f)"):
        fails.append("a child reaches a device before child_begin sets RLIMIT_CORE")
    if "WTERMSIG(r.status) == SIGABRT" not in body(t, "verdict score_abort(const char * who, const child_run & r, "
                                                      "const char * line)"):
        fails.append("score_abort does not score SIGABRT")
    for const, value in (("k_pool_refused_line", POOL_REFUSED), ("k_carrier_line", CARRIER_LINE)):
        if f'constexpr const char * {const} = "{value}";' not in t:
            fails.append(f"{const} is not {value!r}, the line production prints")
    if not ordered(drive, "score_abort(who, r, k_pool_refused_line)", "score_clean(who, r)",
                   "score_abort(who, r, k_carrier_line)"):
        fails.append("drive does not score strict and carrier by their abort lines and plain by a clean exit")
    if t.count("SYCL_TEST_SKIP") < 3:
        fails.append("a VOID child does not make the driver exit 77")
    for name, (children, phrases) in ARMS.items():
        if children not in CHILDREN:
            fails.append(f"arm {name} runs child {children}, which the driver does not run")
        for phrase in phrases:
            if phrase not in t + "\n" + common:
                fails.append(f"the test prints no {phrase!r}, which arm {name} expects")
    return fails


POOL_CONFIGURE = ("bool pinned_chunk_pool::configure_zones(size_t weight_bytes, size_t kv_bytes, "
                  "size_t staging_bytes, size_t scratch_bytes)")
CACHE_CONFIGURE = ("bool unified_cache::configure_host_zones(size_t weight_bytes, size_t kv_bytes, "
                   "size_t staging_bytes, size_t scratch_bytes)")
REFUSAL_BRANCH = "if (!host_arena_->configure_zones(weight_bytes, kv_bytes, staging_bytes, scratch_bytes)) {"
REFUSED_FORMAT = '"[HOST-ARENA] configure_zones refused: chunk %zu still withdrawn, %zu B allocated\\n"'
ZONE_TOUCHES = ("total_zone_bytes", "weight_bytes = ", "grow(", ".clear()", "zone_allocators_[")


def production_problems(texts: dict) -> list:
    fails = missing(texts, POOL, UC)
    if fails:
        return fails
    pool = body(norm(resolve_mutants(texts[POOL])), POOL_CONFIGURE)
    if not pool:
        return [f"{POOL}: missing {POOL_CONFIGURE!r}"]
    refused = pool.find(REFUSED_FORMAT)
    if not ordered(pool, "if (zcs.withdrawn) {", REFUSED_FORMAT, "return false;"):
        fails.append("configure_zones does not refuse a withdrawn chunk by name")
    # Every token that computes or changes a zone sits after the refusal, so a refused call leaves the pool as it was.
    early = [tok for tok in ZONE_TOUCHES if 0 <= pool.find(tok) < refused]
    if refused < 0 or early:
        fails.append(f"configure_zones does not refuse a withdrawn chunk before it computes the zones: {early}")
    if POOL_REFUSED not in pool:
        fails.append(f"configure_zones does not print {POOL_REFUSED!r}")
    if CARRIER_LINE not in norm(texts[POOL]):
        fails.append(f"the pool's destructor does not assert {CARRIER_LINE!r}")
    cache = body(norm(resolve_mutants(texts[UC])), CACHE_CONFIGURE)
    if not cache:
        return fails + [f"{UC}: missing {CACHE_CONFIGURE!r}"]
    branch = body(cache, REFUSAL_BRANCH)
    if not branch:
        return fails + ["configure_host_zones does not branch on the pool's refusal"]
    if not ordered(branch, "if (ggml_sycl_strict_enabled()) {", "GGML_ABORT(", "return false;"):
        fails.append("configure_host_zones does not abort under ggml_sycl_strict_enabled() before it returns false")
    if "host_zone_reserved_bytes_" in branch or "zones_configured()" in branch:
        fails.append("configure_host_zones' refusal touches the reserved figures or reads zones_configured()")
    if "host_zone_reserved_bytes_[" in cache[:cache.find(REFUSAL_BRANCH)]:
        fails.append("configure_host_zones writes a reserved figure before the pool's answer")
    return fails


CLAUSES = {
    "registration": registration_problems,
    "driver": driver_problems,
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

def planted(rel: str, old: str, new: str) -> dict:
    texts = tree_texts()
    assert rel in texts, f"control precondition: {rel} not found"
    assert texts[rel].count(old) >= 1, f"control precondition: {old!r} is not in {rel}"
    texts[rel] = texts[rel].replace(old, new, 1)
    return texts


def rejects(clause: str, texts: dict, reason: str) -> None:
    fails = CLAUSES[clause](texts)
    assert any(reason in f for f in fails), f"{clause} did not reject the control with {reason!r}: {fails}"


def test_registration_rejects_a_semicolon_in_an_expected_text():
    rejects("registration", planted(CMAKE, '"strict child exited [0-9]+, expected SIGABRT"',
                                    '"strict child exited [0-9]+; expected SIGABRT"'),
            "MUTANTS list is not whole")


def test_registration_rejects_a_dropped_arm_and_an_unbounded_budget():
    rejects("registration", planted(CMAKE, "        configure-direct-release plain\n",
                                    "        configure-direct-release strict\n"),
            "arm configure-direct-release runs children 'strict'")
    rejects("registration", planted(CMAKE, 'ENV "GGML_SYCL_HOST_RESERVE_MB=1024"', ""),
            "does not bound the pinned budget")
    rejects("registration", planted(CMAKE, 'ENV "GGML_SYCL_HOST_RESERVE_MB=1024"',
                                    'ENV "GGML_SYCL_HOST_RESERVE_MB=262144"'),
            "does not bound the pinned budget")
    rejects("registration", planted(CMAKE, "        strict-abort-dropped strict\n",
                                    "        strict-abort-gone strict\n"),
            "arm strict-abort-dropped is not registered")


def test_driver_rejects_a_setenv_and_a_device_in_the_driver():
    rejects("driver", planted(TEST, "int child_plain(checks & c, const device_fixture & f) {",
                              'int child_plain(checks & c, const device_fixture & f) {\n    setenv("X", "1", 1);'),
            "calls setenv")
    rejects("driver", planted(TEST, "    // Every envp is built here, before any fork.",
                              "    (void) ggml_backend_sycl_init(0);"),
            "the driver touches a device: drive names ggml_backend_")


def test_driver_rejects_a_late_rlimit_and_a_wrong_score():
    rejects("driver", planted(TEST, "    if (!child_begin(a)) {\n        return 2;\n    }\n"
                                    "    if (!entry_environment_ok(c)) {\n        return c.status();\n    }\n",
                              "    if (!entry_environment_ok(c)) {\n        return c.status();\n    }\n"
                              "    device_fixture early;\n    (void) device_fixture_open(c, early);\n"
                              "    if (!child_begin(a)) {\n        return 2;\n    }\n"),
            "a child reaches a device before child_begin sets RLIMIT_CORE")
    rejects("driver", planted(TEST, "v = score_abort(who, r, k_carrier_line);", "v = score_clean(who, r);"),
            "does not score strict and carrier by their abort lines")
    rejects("driver", planted(TEST, '{ "GGML_SYCL_STRICT_LEASES=1", "GGML_NO_BACKTRACE=1" }',
                              '{ "GGML_NO_BACKTRACE=1" }'),
            "GGML_SYCL_STRICT_LEASES=1 is set 0 time(s)")


def test_production_rejects_a_lost_abort_and_a_late_scan():
    rejects("production", planted(UC, 'GGML_ABORT(\n                "[HOST-ARENA] configure_zones refused: a withdrawn '
                                      'SCRATCH chunk survives "\n                "(GGML_SYCL_STRICT_LEASES=1)");', ";"),
            "does not abort under ggml_sycl_strict_enabled()")
    texts = tree_texts()
    pool = texts.get(POOL, "")
    at = pool.find("    const size_t total_zone_bytes = weight_bytes + kv_bytes + staging_bytes + scratch_bytes;")
    scan = pool.find("    for (const auto & states : zone_allocators_) {")
    assert 0 <= scan < at, "control precondition: the scan does not precede the zone sum"
    texts[POOL] = pool[:scan] + "    size_t total_zone_bytes = 0;\n" + pool[scan:]
    rejects("production", texts, "before it computes the zones")


def test_production_rejects_a_reserved_write_in_the_refusal():
    strict = ('        if (ggml_sycl_strict_enabled()) {\n            GGML_ABORT(\n'
              '                "[HOST-ARENA] configure_zones refused')
    rejects("production", planted(UC, strict, "        host_zone_reserved_bytes_.fill(0);\n" + strict),
            "refusal touches the reserved figures")


def test_negative_control_comments_change_nothing():
    texts = tree_texts()
    for rel in FILES:
        if rel in texts:
            texts[rel] += "\n# a comment\n" if rel.endswith(".txt") else "\n// a comment: setenv(\"X\", \"1\", 1);\n"
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
