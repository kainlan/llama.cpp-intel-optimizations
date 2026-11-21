#!/usr/bin/env python3
"""Source contract: every oneDNN primitive the SYCL backend builds asks for a deterministic implementation.

Host-only: reads the backend's C++ sources and docs/backend/sycl-env-vars.md, builds nothing, loads no model and
touches no device.

Why: without the deterministic attribute the oneDNN GPU matmul may pick an implementation whose accumulation order
depends on how the work is scheduled, so identical inputs give different bits from one run to the next, and greedy
decoding parts where two candidates nearly tie. The attribute forbids those implementations. It only works if every
attribute object the backend builds carries it, so this gate holds that, and holds the one switch that turns it off:

  D1  dnnl-deterministic.hpp defines the helper ggml_sycl_dnnl_attr_set_reproducible(dnnl::primitive_attr & attr)
      once, and its body is exactly: when the switch is on, attr.set_deterministic(true); no other code calls
      set_deterministic, in the C++ API or the C API;
  D2  the switch GGML_SYCL_ONEDNN_DETERMINISTIC is ON by default and only the exact value "0" turns it off: its one
      accessor returns true when the variable is unset or is anything but "0" (an empty value, "false", "off" or a
      typo keeps the reproducible mode, the safe side of a mistyped variable, where std::atoi would read 0), its
      report row in ggml_check_sycl() has default 1, and the row's integer is overwritten from the accessor after the
      table parse, so the report says what is in effect;
  D3  every dnnl::primitive_attr the backend constructs is handed to the helper in the very next statement, or is
      initialised from a function that returns one (whose own construction is held by this pin); no other way of
      making an attribute object (a temporary, the C API, the type named without dnnl::) appears;
  D4  the variable is read in exactly two places, and the accessor's read is the one that decides: its name is
      spelled once (the constant beside the accessor), and the constant is used only by its definition, the
      accessor's getenv and the report row. The report row is the second read: the startup table reads every row's
      name with the generic integer parser, and this row's result is then overwritten from the accessor (D2), so it
      never decides anything. Any third read fails;
  D5  every primitive descriptor is built with an attribute: its last argument names an attribute object of the
      file (one D3 holds, or a const reference parameter that carries one), whether it is built with parentheses or
      braces; no descriptor is built through the C API, where a null attribute would pass unseen;
  D6  docs/backend/sycl-env-vars.md has the variable's row, default ON, stating that only the exact value 0 turns
      it off.

The scan covers every .cpp/.hpp/.h under ggml/src/ggml-sycl except its tests/ directory, plus the standalone oneDNN
benchmark tests/bench-dnnl-ops.cpp there, which includes dnnl-deterministic.hpp so that it times the implementation
the backend runs.

Checks run on comment-blanked text. NOT VACUOUS: --self-test runs the gate on in-memory mutants of the real sources.
Each mutant names the pin that must catch it, and is reported if it survives, if a different pin catches it, or if its
anchor no longer matches exactly once (NOT APPLIED). ctest registers it with --self-test.

WHAT THIS DOES NOT PROVE. It reads text. That the attribute makes a run reproducible is the lead's two-run device
measurement; what it costs is the lead's interleaved A/B with the switch flipped on one binary.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "ggml/src/ggml-sycl"
DOCS = ROOT / "docs/backend/sycl-env-vars.md"

DNNL_OPS = "ggml/src/ggml-sycl/dnnl-ops.hpp"
HDR = "ggml/src/ggml-sycl/dnnl-deterministic.hpp"
BENCH = "ggml/src/ggml-sycl/tests/bench-dnnl-ops.cpp"
GGML_SYCL = "ggml/src/ggml-sycl/ggml-sycl.cpp"
DOCS_KEY = "docs/backend/sycl-env-vars.md"

ENV_NAME = "GGML_SYCL_ONEDNN_DETERMINISTIC"
ENV_CONST = "GGML_SYCL_ONEDNN_DETERMINISTIC_ENV"
HELPER = "ggml_sycl_dnnl_attr_set_reproducible"
ACCESSOR = "ggml_sycl_onednn_deterministic_enabled"
REPORT_GLOBAL = "g_ggml_sycl_onednn_deterministic"

HELPER_DEF = f"inline void {HELPER}(dnnl::primitive_attr & attr) {{"
HELPER_BODY = f"{{ if ({ACCESSOR}()) {{ attr.set_deterministic(true); }} }}"
ACCESSOR_DEF = f"inline bool {ACCESSOR}() {{"
ACCESSOR_BODY = (
    f"{{ static const bool enabled = [] {{ const char * env = std::getenv({ENV_CONST}); "
    'return env == nullptr || std::strcmp(env, "0") != 0; }(); return enabled; }'
)
CONST_DEF = f'static constexpr const char * {ENV_CONST} = "{ENV_NAME}";'
REPORT_ROW = f"{{ {ENV_CONST}, &{REPORT_GLOBAL}, 1 }},"
REPORT_SET = f"{REPORT_GLOBAL} = {ACCESSOR}() ? 1 : 0;"
DOCS_ROW = f"| `{ENV_NAME}=0` | ON |"
DOCS_STRICT = "Only the exact value `0` turns the attribute off"

ATTR_TYPE = re.compile(r"\bdnnl::primitive_attr\b")
DECL_PLAIN = re.compile(r"dnnl::primitive_attr\s+([A-Za-z_]\w*)\s*;")
DECL_INIT = re.compile(r"(?:const\s+)?dnnl::primitive_attr\s+([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s*\(")
FACTORY_DEF = re.compile(r"dnnl::primitive_attr\s+([A-Za-z_]\w*)\s*\([^;{)]*\)\s*\{")
PARAM_REF = re.compile(r"(?:const\s+)?dnnl::primitive_attr\s*&\s*([A-Za-z_]\w*)\s*[,)]")
BARE_TYPE = re.compile(r"(?<![\w:])primitive_attr\s+[A-Za-z_]\w*\s*[;={(]")
C_API_CREATE = re.compile(r"\bdnnl_primitive_attr_(?:create|clone)\s*\(")
PD_CALL = re.compile(r"\bprimitive_desc\s*[({]|\bprimitive_desc\s+[A-Za-z_]\w*\s*[({]")
C_API_PD = re.compile(r"\bdnnl_\w*primitive_desc_create\s*\(")
SET_DETERMINISTIC = re.compile(r"set_deterministic\s*\(")
ENV_READ = re.compile(r"\b(?:std::)?(?:getenv|secure_getenv|get_sycl_env)\s*\(\s*([^,)]*)")


class ContractError(AssertionError):
    pass


class MutantNotApplied(Exception):
    pass


_COMMENT_OR_LITERAL = re.compile(r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'", re.DOTALL)


def blank_comments(text: str) -> str:
    """Comments become spaces (newlines kept); string and character literals stay."""

    def repl(m: re.Match) -> str:
        t = m.group(0)
        if t.startswith("//") or t.startswith("/*"):
            return "".join("\n" if c == "\n" else " " for c in t)
        return t

    return _COMMENT_OR_LITERAL.sub(repl, text)


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def line_of(text: str, at: int) -> int:
    return text.count("\n", 0, at) + 1


def brace_block(code: str, open_at: int, pin: str, what: str) -> str:
    depth = 0
    k = open_at
    n = len(code)
    while k < n:
        c = code[k]
        if c == '"':
            k += 1
            while k < n and code[k] != '"':
                k += 2 if code[k] == "\\" else 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return code[open_at : k + 1]
        k += 1
    raise ContractError(f"FAIL [pin {pin}]: the block of {what} is unbalanced")


def paren_args(code: str, open_at: int) -> list[str]:
    """Top-level comma-separated arguments of the call whose `(` is at `open_at`."""
    depth = 0
    args: list[str] = []
    cur = []
    k = open_at
    while k < len(code):
        c = code[k]
        if c == '"':
            j = k + 1
            while j < len(code) and code[j] != '"':
                j += 2 if code[j] == "\\" else 1
            cur.append(code[k : j + 1])
            k = j + 1
            continue
        if c in "([{":
            depth += 1
            if depth > 1:
                cur.append(c)
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                args.append("".join(cur).strip())
                return args
            cur.append(c)
        elif c == "," and depth == 1:
            args.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
        k += 1
    return args


def definition_body(files: dict[str, str], rel: str, marker: str, pin: str) -> str:
    code = files[rel]
    sq = squash(code)
    n = sq.count(marker)
    if n != 1:
        raise ContractError(f"FAIL [pin {pin}]: `{marker}` appears {n} times in {rel}, not once")
    at = code.find(marker.split("(")[0])
    open_at = code.find("{", at)
    return squash(brace_block(code, open_at, pin, marker))


def check(files: dict[str, str]) -> None:
    sources = {k: v for k, v in files.items() if k != DOCS_KEY}

    # D1 ---------------------------------------------------------------------------------------------------------------
    for rel, code in sources.items():
        if rel != HDR and re.search(rf"\b{HELPER}\s*\([^;)]*&", code):
            raise ContractError(f"FAIL [pin D1]: {rel} defines a second {HELPER}")
    body = definition_body(sources, HDR, HELPER_DEF, "D1")
    if body != HELPER_BODY:
        raise ContractError(f"FAIL [pin D1]: the helper's body is `{body}`, not `{HELPER_BODY}`")
    calls = [(rel, line_of(code, m.start())) for rel, code in sources.items() for m in SET_DETERMINISTIC.finditer(code)]
    if len(calls) != 1 or calls[0][0] != HDR:
        raise ContractError(f"FAIL [pin D1]: set_deterministic is called at {calls}; only the helper may call it")

    # D2 ---------------------------------------------------------------------------------------------------------------
    body = definition_body(sources, HDR, ACCESSOR_DEF, "D2")
    if body != squash(ACCESSOR_BODY):
        raise ContractError(f"FAIL [pin D2]: the accessor's body is `{body}`, not the default-ON reader that only "
                            "\"0\" turns off")
    main_sq = squash(sources[GGML_SYCL])
    for want, what in ((REPORT_ROW, "the report row with default 1"), (REPORT_SET, "the report's overwrite")):
        n = main_sq.count(want)
        if n != 1:
            raise ContractError(f"FAIL [pin D2]: {what} `{want}` appears {n} times in ggml-sycl.cpp, not once")
    if main_sq.find(REPORT_ROW) > main_sq.find(REPORT_SET):
        raise ContractError("FAIL [pin D2]: the report's overwrite comes before its table row")

    # D3 ---------------------------------------------------------------------------------------------------------------
    factories: set[str] = set()
    for code in sources.values():
        for m in FACTORY_DEF.finditer(code):
            factories.add(m.group(1))
    attr_names: dict[str, set[str]] = {}
    for rel, code in sources.items():
        names: set[str] = set()
        for pattern, what in ((C_API_CREATE, "the C API makes an attribute object"),
                              (BARE_TYPE, "an attribute object is declared without dnnl::")):
            m = pattern.search(code)
            if m:
                raise ContractError(f"FAIL [pin D3]: {rel}:{line_of(code, m.start())}: {what}")
        for m in ATTR_TYPE.finditer(code):
            tail = code[m.start() : m.start() + 400]
            # A reference parameter (the helper's own, the primitive cache's) names an object made elsewhere.
            pm = PARAM_REF.match(tail)
            if pm:
                names.add(pm.group(1))
                continue
            fm = FACTORY_DEF.match(tail)
            if fm:
                continue
            dm = DECL_PLAIN.match(tail)
            if dm:
                name = dm.group(1)
                after = code[m.start() + dm.end() :]
                if not re.match(rf"\s*{HELPER}\s*\(\s*{name}\s*\)\s*;", after):
                    raise ContractError(
                        f"FAIL [pin D3]: {rel}:{line_of(code, m.start())}: `{name}` is not handed to {HELPER} in the "
                        "statement right after its construction"
                    )
                names.add(name)
                continue
            im = DECL_INIT.match(tail)
            if im:
                if im.group(2) not in factories:
                    raise ContractError(
                        f"FAIL [pin D3]: {rel}:{line_of(code, m.start())}: `{im.group(1)}` comes from "
                        f"`{im.group(2)}`, which is not a function of the backend that returns an attribute"
                    )
                names.add(im.group(1))
                continue
            raise ContractError(
                f"FAIL [pin D3]: {rel}:{line_of(code, m.start())}: an attribute object made in a form this pin does "
                f"not know: `{squash(tail[:80])}`"
            )
        attr_names[rel] = names
    if not any(attr_names.values()):
        raise ContractError("FAIL [pin D3]: no attribute object found, so the pin read the wrong text")

    # D4 ---------------------------------------------------------------------------------------------------------------
    raw_count = 0
    for rel, code in sources.items():
        raw_count += code.count(f'"{ENV_NAME}"')
    if raw_count != 1:
        raise ContractError(f"FAIL [pin D4]: the name \"{ENV_NAME}\" is spelled {raw_count} times, not once")
    if squash(sources[HDR]).count(CONST_DEF) != 1:
        raise ContractError(f"FAIL [pin D4]: dnnl-deterministic.hpp does not define `{CONST_DEF}` once")
    uses = []
    for rel, code in sources.items():
        for m in re.finditer(rf"\b{ENV_CONST}\b", code):
            uses.append((rel, line_of(code, m.start())))
    by_file = {}
    for rel, _ in uses:
        by_file[rel] = by_file.get(rel, 0) + 1
    if by_file != {HDR: 2, GGML_SYCL: 1}:
        raise ContractError(
            f"FAIL [pin D4]: {ENV_CONST} is used at {uses}; it belongs to its definition, the accessor and the "
            "one report row only"
        )
    for rel, code in sources.items():
        for m in ENV_READ.finditer(code):
            arg = m.group(1).strip()
            if f'"{ENV_NAME}"' in arg or (re.search(rf"\b{ENV_CONST}\b", arg) and rel != HDR):
                raise ContractError(f"FAIL [pin D4]: {rel}:{line_of(code, m.start())}: a second reader of {ENV_NAME}")
    accessor_reads = sum(1 for m in ENV_READ.finditer(sources[HDR]) if m.group(1).strip() == ENV_CONST)
    if accessor_reads != 1:
        raise ContractError(f"FAIL [pin D4]: dnnl-deterministic.hpp reads {ENV_CONST} {accessor_reads} times, not once")

    # D5 ---------------------------------------------------------------------------------------------------------------
    n_pd = 0
    for rel, code in sources.items():
        m = C_API_PD.search(code)
        if m:
            raise ContractError(
                f"FAIL [pin D5]: {rel}:{line_of(code, m.start())}: a primitive descriptor is built through the C API, "
                "whose attribute argument (null included) this pin does not hold"
            )
        for m in PD_CALL.finditer(code):
            open_at = m.end() - 1  # the `(` or `{` the pattern ended on
            args = paren_args(code, open_at)
            n_pd += 1
            last = args[-1] if args else ""
            if last not in attr_names.get(rel, set()):
                raise ContractError(
                    f"FAIL [pin D5]: {rel}:{line_of(code, m.start())}: a primitive descriptor is built without an "
                    f"attribute object this pin holds (last argument `{last}`)"
                )
    if n_pd == 0:
        raise ContractError("FAIL [pin D5]: no primitive descriptor found, so the pin read the wrong text")

    # D6 ---------------------------------------------------------------------------------------------------------------
    rows = [line for line in files[DOCS_KEY].splitlines() if line.startswith(DOCS_ROW)]
    if len(rows) != 1:
        raise ContractError(f"FAIL [pin D6]: {DOCS_KEY} has no single row starting `{DOCS_ROW}`")
    if DOCS_STRICT not in rows[0]:
        raise ContractError(f"FAIL [pin D6]: the {ENV_NAME} row does not say `{DOCS_STRICT}`")


def load() -> dict[str, str]:
    raws: dict[str, str] = {}
    for path in sorted(BACKEND_DIR.rglob("*")):
        if path.suffix not in (".cpp", ".hpp", ".h") or not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith("ggml/src/ggml-sycl/tests/") and rel != BENCH:
            continue
        raws[rel] = path.read_text(errors="replace")
    raws[DOCS_KEY] = DOCS.read_text()
    return raws


def blanked(raws: dict[str, str]) -> dict[str, str]:
    return {k: (v if k == DOCS_KEY else blank_comments(v)) for k, v in raws.items()}


def mutate(raws: dict[str, str], rel: str, edits: list[tuple[str, str]]) -> dict[str, str]:
    out = dict(raws)
    text = out[rel]
    for old, new in edits:
        n = text.count(old)
        if n != 1:
            raise MutantNotApplied(f"NOT APPLIED: anchor appears {n} times in {rel}, not once: {old[:70]!r}")
        text = text.replace(old, new)
    out[rel] = text
    return out


def self_test(raws: dict[str, str]) -> int:
    failures: list[str] = []
    n_mutants = 0

    def expect_fail(name: str, pin: str, rel: str, edits: list[tuple[str, str]]) -> None:
        nonlocal n_mutants
        n_mutants += 1
        try:
            mutant = mutate(raws, rel, edits)
        except MutantNotApplied as e:
            failures.append(name)
            print(f"  self-test: {name}: {e}")
            return
        try:
            check(blanked(mutant))
        except ContractError as e:
            if f"[pin {pin}]" in str(e):
                return
            failures.append(name)
            print(f"  self-test: {name}: caught by a different pin than {pin}: {e}")
            return
        failures.append(name)
        print(f"  self-test: {name}: SURVIVED (pin {pin} did not catch it)")

    gemm = "ggml/src/ggml-sycl/gemm.hpp"
    softmax_decl = "        dnnl::primitive_attr attr;\n        ggml_sycl_dnnl_attr_set_reproducible(attr);\n" \
                   "        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n\n        auto softmax_pd"
    softmax_bare = "        dnnl::primitive_attr attr;\n        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n" \
                   "\n        auto softmax_pd"
    det_call = "        attr.set_deterministic(true);\n"
    accessor_ret = '        return env == nullptr || std::strcmp(env, "0") != 0;\n'

    # (a) an attribute built without the helper.
    expect_fail("helper-call-removed", "D3", DNNL_OPS, [(softmax_decl, softmax_bare)])
    expect_fail("helper-call-after-another-statement", "D3", DNNL_OPS,
                [(softmax_decl, "        dnnl::primitive_attr attr;\n"
                                "        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n"
                                "        ggml_sycl_dnnl_attr_set_reproducible(attr);\n\n        auto softmax_pd")])
    expect_fail("helper-on-another-object", "D3", DNNL_OPS,
                [(softmax_decl, softmax_decl.replace("set_reproducible(attr)", "set_reproducible(other)"))])
    expect_fail("new-attr-without-helper", "D3", gemm,
                [("        attr.set_scales(DNNL_ARG_WEIGHTS, mask, { group_size, 1 }, dt::f16);",
                  "        dnnl::primitive_attr extra;\n"
                  "        attr.set_scales(DNNL_ARG_WEIGHTS, mask, { group_size, 1 }, dt::f16);")])
    expect_fail("attr-from-unknown-function", "D3", gemm,
                [("const dnnl::primitive_attr attr2d = woq_mxfp4_2d_attr(group_size);",
                  "const dnnl::primitive_attr attr2d = some_other_attr(group_size);")])
    expect_fail("attr-temporary", "D3", DNNL_OPS,
                [(softmax_decl, softmax_decl + "\n        auto tmp = dnnl::primitive_attr();\n")])
    expect_fail("attr-without-namespace", "D3", DNNL_OPS,
                [(softmax_decl, "        using dnnl::primitive_attr;\n        primitive_attr spare;\n" + softmax_decl)])
    expect_fail("attr-through-c-api", "D3", DNNL_OPS,
                [(softmax_decl, softmax_decl + "\n        dnnl_primitive_attr_t raw = nullptr;\n"
                                "        dnnl_primitive_attr_create(&raw);\n")])
    # (a) a primitive descriptor built with no attribute.
    expect_fail("pd-without-attr", "D5", DNNL_OPS,
                [("            src_md, dst_md, 1, attr);", "            src_md, dst_md, 1);")])
    expect_fail("pd-with-a-non-attr-last-argument", "D5", DNNL_OPS,
                [("            src_md, dst_md, 1, attr);", "            src_md, dst_md, 1);  // attr")])
    # A default-constructed temporary in the call is a construction the helper never saw.
    expect_fail("pd-with-default-attr", "D3", DNNL_OPS,
                [("            src_md, dst_md, 1, attr);", "            src_md, dst_md, 1, dnnl::primitive_attr());")])
    expect_fail("pd-brace-initialised-without-attr", "D5", DNNL_OPS,
                [(softmax_decl, softmax_decl.replace(
                    "        auto softmax_pd",
                    "        dnnl::softmax_forward::primitive_desc extra_pd{ eng, dnnl::prop_kind::forward_inference,\n"
                    "            dnnl::algorithm::softmax_accurate, src_md, dst_md, 1 };\n        auto softmax_pd"))])
    expect_fail("pd-through-c-api-with-null-attr", "D5", DNNL_OPS,
                [(softmax_decl, softmax_decl.replace(
                    "        auto softmax_pd",
                    "        dnnl_primitive_desc_t raw_pd = nullptr;\n"
                    "        dnnl_softmax_forward_primitive_desc_create(&raw_pd, eng.get(), dnnl_forward_inference,\n"
                    "            dnnl_softmax_accurate, src_md.get(), dst_md.get(), 1, nullptr);\n"
                    "        auto softmax_pd"))])
    # The benchmark builds its descriptors through the helper too.
    expect_fail("bench-attr-without-helper", "D3", BENCH,
                [("    dnnl::primitive_attr attr;\n    ggml_sycl_dnnl_attr_set_reproducible(attr);\n"
                  "    auto binary_pd",
                  "    dnnl::primitive_attr attr;\n    auto binary_pd")])
    expect_fail("bench-pd-without-attr", "D5", BENCH,
                [("dst_md, eps, flags::use_scale, attr);", "dst_md, eps, flags::use_scale);")])
    # (b) the helper stops asking for determinism.
    expect_fail("helper-no-longer-deterministic", "D1", HDR, [(det_call, "        (void) attr;\n")])
    expect_fail("helper-asks-false", "D1", HDR, [(det_call, "        attr.set_deterministic(false);\n")])
    expect_fail("helper-ignores-the-switch", "D1", HDR,
                [(f"    if ({ACCESSOR}()) {{\n" + det_call + "    }\n", "    " + det_call.strip() + "\n")])
    expect_fail("second-helper", "D1", "ggml/src/ggml-sycl/common.hpp",
                [("#endif  // GGML_SYCL_COMMON_HPP",
                  f"inline void {HELPER}(int & attr) {{ (void) attr; }}\n#endif  // GGML_SYCL_COMMON_HPP")])
    # (c) set_deterministic called anywhere but the helper, through either API.
    expect_fail("set-deterministic-outside-helper", "D1", gemm,
                [("        attr.set_scales(DNNL_ARG_WEIGHTS, mask, { group_size, 1 }, dt::f16);",
                  "        attr.set_scales(DNNL_ARG_WEIGHTS, mask, { group_size, 1 }, dt::f16);\n"
                  "        attr.set_deterministic(false);")])
    expect_fail("set-deterministic-through-c-api", "D1", DNNL_OPS,
                [(softmax_decl, softmax_decl + "\n        dnnl_primitive_attr_set_deterministic(attr.get(), 0);\n")])
    # (d) the default flips to off, or something other than exactly "0" turns it off.
    expect_fail("default-off", "D2", HDR,
                [(accessor_ret, '        return env != nullptr && std::strcmp(env, "0") != 0;\n')])
    expect_fail("parse-by-atoi", "D2", HDR,
                [(accessor_ret, "        return env == nullptr || std::atoi(env) != 0;\n")])
    expect_fail("empty-value-turns-off", "D2", HDR,
                [(accessor_ret, '        return env == nullptr || (env[0] != 0 && std::strcmp(env, "0") != 0);\n')])
    expect_fail("prefix-zero-turns-off", "D2", HDR,
                [(accessor_ret, '        return env == nullptr || std::strncmp(env, "0", 1) != 0;\n')])
    expect_fail("report-default-off", "D2", GGML_SYCL, [(REPORT_ROW, REPORT_ROW.replace(", 1 }", ", 0 }"))])
    expect_fail("report-not-overwritten", "D2", GGML_SYCL, [(REPORT_SET, "")])
    # (e) a third read of the variable (the accessor and the report row are the two allowed).
    expect_fail("second-reader-literal", "D4", GGML_SYCL,
                [(REPORT_SET, REPORT_SET + f'\n        (void) std::getenv("{ENV_NAME}");')])
    expect_fail("second-reader-constant", "D4", GGML_SYCL,
                [(REPORT_SET, REPORT_SET + f"\n        (void) get_sycl_env({ENV_CONST}, 1);")])
    expect_fail("second-read-in-accessor-file", "D4", HDR,
                [(HELPER_DEF, f"inline bool other_reader() {{ return std::getenv({ENV_CONST}) != nullptr; }}\n\n"
                              + HELPER_DEF)])
    expect_fail("second-report-row", "D4", GGML_SYCL, [(REPORT_ROW, REPORT_ROW + "\n            " + REPORT_ROW.replace(
        f"&{REPORT_GLOBAL}", "&g_ggml_sycl_disable_dnn"))])
    # the documentation row.
    expect_fail("docs-row-missing", "D6", DOCS_KEY, [(DOCS_ROW, "| `GGML_SYCL_ONEDNN_DETERMINISTIC_X=0` | ON |")])
    expect_fail("docs-row-without-the-parse-rule", "D6", DOCS_KEY, [(DOCS_STRICT, "Only the value `0` turns it off")])

    try:
        mutate(raws, DNNL_OPS, [("this anchor is in no source", "x")])
        failures.append("stale-anchor-not-reported")
        print("  self-test: a stale mutant anchor is NOT reported")
    except MutantNotApplied:
        print("  self-test: a stale mutant anchor is reported")

    try:
        check(blanked(raws))
        print("  self-test: unmodified sources pass")
    except ContractError as e:
        print(f"  self-test: unmodified sources FAIL: {e}")
        failures.append("unmodified")

    if failures:
        print(f"SELF-TEST FAIL: {', '.join(failures)}")
        return 1
    print(f"SELF-TEST PASS: {n_mutants} mutants caught by their pins, unmodified sources pass")
    return 0


def main(argv: list[str]) -> int:
    raws = load()
    if "--self-test" in argv:
        return self_test(raws)
    try:
        check(blanked(raws))
    except ContractError as e:
        print(e)
        return 1
    print("PASS: every oneDNN attribute object of the backend asks for a deterministic implementation, default ON")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
