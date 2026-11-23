#!/usr/bin/env python3
"""An expert family's per-expert scale is a dense tensor to the planner AND to the S1 preload (Inkling 2 Task 8i).

THE DEFECT.  The model loads optional per-expert scale vectors beside an expert matrix
(`blk.N.ffn_gate_up_exps.scale`, `.scales`, `.input_scale`: {n_expert} F32 metadata of an NVFP4 or FP8 expert
weight).  Two sources classified them, and both called them expert weights:
  - the name rule, infer_tensor_usage() (common.hpp), returned MOE_EXPERT_WEIGHT for every `ffn_*_exps` name but
    `.bias`, so the planner split the 1-D scale into ONE expert entry, e0, whose (layer, expert, role) key
    duplicated expert 0's (Inkling: `[PLACEMENT-MOE] ... duplicates=40`), and gave it no dense entry;
  - the weight-usage registry, which the S1 preload reads first, stored the loader's family usage
    (MOE_EXPERT_WEIGHT, the suffix ignored), so the preload skipped the tensor (`ne[2] <= 1`).
So a card layer's scale never reached the card: the REPEAT in build_lora_mm_id read it from pinned host memory,
staged it every graph (bin-broadcast raw host staging), and that staging cannot be recorded into a SYCL graph.

THE CONTRACT.  One predicate says what an expert scale is, and both classifiers ask it.
  N. (common.hpp) `inline bool ggml_sycl_moe_expert_scale_name(const char * name)` is defined once and matches the
     "exps.scale" and "exps.input_scale" substrings (so ".scales", the chexps family and a view such as
     "... .scale (reshaped)", the name the REPEAT reads, all match).
  U. (common.hpp infer_tensor_usage) the one `return tensor_usage::MOE_EXPERT_WEIGHT;` is guarded by
     `!ggml_sycl_moe_expert_scale_name(name)`, and an expert-family name that fails the guard still returns
     MOE_INTERMEDIATE, the class the family's `.bias` already has: the planner then gives the scale a dense entry
     that moves with its layer.
  R. (ggml-sycl.cpp ggml_backend_sycl_try_register_weight_usage) before the registry stores a row, a name the
     predicate matches that arrives as MOE_EXPERT_WEIGHT is stored as MOE_INTERMEDIATE, so the S1 preload reads
     the class the planner placed it by.
  S. (one source) neither classifier restates the suffixes: "exps.scale" and "exps.input_scale" appear once in
     common.hpp (the predicate) and never in ggml-sycl.cpp.

The behaviour is checked by tests/test-sycl-tensor-usage.cpp (name rule, host only) and
ggml/src/ggml-sycl/tests/test-cross-model-weight-usage.cpp checks 7 and 8 (registry, opens a device).

NOT VACUOUS.  --self-test runs the gate on in-memory mutants of the real sources and requires each to FAIL on its
own pin, then requires the unmodified tree to pass.

RED on the tree before the change:
  FAIL [pin N]: common.hpp defines no ggml_sycl_moe_expert_scale_name(const char * name) (Inkling 2 Task 8i)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
COMMON  = ROOT / "ggml" / "src" / "ggml-sycl" / "common.hpp"
BACKEND = ROOT / "ggml" / "src" / "ggml-sycl" / "ggml-sycl.cpp"
TAG     = "(Inkling 2 Task 8i)"

PRED      = "ggml_sycl_moe_expert_scale_name"
PRED_SIG  = r"inline\s+bool\s+" + PRED + r"\s*\(\s*const\s+char\s*\*\s*name\s*\)\s*\{"
USAGE_SIG = r"inline\s+tensor_usage\s+infer_tensor_usage\s*\(\s*const\s+char\s*\*\s*name\s*\)\s*\{"
REG_SIG   = (r"\nbool\s+ggml_backend_sycl_try_register_weight_usage\s*\(\s*const\s+char\s*\*\s*tensor_name\s*,"
             r"\s*ggml_backend_sycl_tensor_usage\s+usage\s*\)\s*\{")
LITERALS  = ('"exps.scale"', '"exps.input_scale"')


class ContractError(Exception):
    pass


def fail(pin: str, msg: str) -> None:
    raise ContractError(f"FAIL [pin {pin}]: {msg} {TAG}")


_TOKENS = re.compile(r"(\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*')|//[^\n]*|/\*.*?\*/", re.S)


def blank_comments(src: str) -> str:
    """Drop // and /* */ comments so a pin cannot be satisfied by prose. String and char literals are kept, so a
    "//" inside a literal is not taken for a comment."""
    return _TOKENS.sub(lambda m: m.group(1) if m.group(1) is not None else " ", src)


def body_after(code: str, open_brace: int, pin: str, what: str) -> str:
    """The text between the brace at open_brace and its match. `code` has no comments."""
    depth = 0
    i, n = open_brace, len(code)
    while i < n:
        c = code[i]
        if c == '"' or c == "'":
            j = i + 1
            while j < n and code[j] != c:
                j += 2 if code[j] == "\\" else 1
            i = j + 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return code[open_brace + 1 : i]
        i += 1
    fail(pin, f"unbalanced body for {what}")
    return ""


def function_body(code: str, signature: str, pin: str, what: str, expect_one: bool = True) -> str:
    matches = list(re.finditer(signature, code))
    if not matches:
        fail(pin, f"{what} has no definition")
    if expect_one and len(matches) != 1:
        fail(pin, f"{what} is defined {len(matches)} times, expected once")
    return body_after(code, matches[0].end() - 1, pin, what)


def check_predicate(common: str) -> None:
    if not re.search(PRED_SIG, common):
        fail("N", f"common.hpp defines no {PRED}(const char * name)")
    body = function_body(common, PRED_SIG, "N", PRED)
    for lit in LITERALS:
        if not re.search(r"strstr\s*\(\s*name\s*,\s*" + re.escape(lit) + r"\s*\)", body):
            fail("N", f"{PRED} does not match the {lit} substring with strstr(name, {lit})")


def check_name_rule(common: str) -> None:
    body = function_body(common, USAGE_SIG, "U", "infer_tensor_usage")
    guarded = list(re.finditer(r"if\s*\(([^{};]*)\)\s*\{?\s*return\s+tensor_usage::MOE_EXPERT_WEIGHT\s*;", body))
    bare = re.findall(r"return\s+tensor_usage::MOE_EXPERT_WEIGHT\s*;", body)
    if len(guarded) != 1 or len(bare) != 1:
        fail("U", f"infer_tensor_usage must have exactly one guarded MOE_EXPERT_WEIGHT return, found "
                  f"{len(guarded)} guarded of {len(bare)}")
    cond = guarded[0].group(1)
    if "moe_exps_name" not in cond:
        fail("U", "the MOE_EXPERT_WEIGHT return in infer_tensor_usage is no longer limited to expert-family names")
    if not re.search(r"!\s*" + PRED + r"\s*\(\s*name\s*\)", cond):
        fail("U", f"the MOE_EXPERT_WEIGHT return in infer_tensor_usage is not guarded by !{PRED}(name): an expert "
                  "scale is still classified as an expert matrix")
    tail = body[guarded[0].end() :]
    if not re.search(r"if\s*\(\s*moe_exps_name\s*\)\s*\{?\s*return\s+tensor_usage::MOE_INTERMEDIATE\s*;", tail):
        fail("U", "an expert-family name that is not an expert matrix no longer falls to MOE_INTERMEDIATE")


def check_registry(backend: str) -> None:
    m = re.search(REG_SIG, backend)
    if not m:
        fail("R", "ggml-sycl.cpp has no definition of ggml_backend_sycl_try_register_weight_usage")
    body = body_after(backend, m.end() - 1, "R", "ggml_backend_sycl_try_register_weight_usage")
    fixes = list(re.finditer(r"if\s*\(([^{};]*)\)\s*\{?\s*mapped\s*=\s*tensor_usage::(\w+)\s*;", body))
    fix = None
    for f in fixes:
        cond = f.group(1)
        if (re.search(r"mapped\s*==\s*tensor_usage::MOE_EXPERT_WEIGHT", cond) and
                re.search(PRED + r"\s*\(\s*tensor_name\s*\)", cond)):
            fix = f
            break
    if fix is None:
        fail("R", f"ggml_backend_sycl_try_register_weight_usage does not ask {PRED}(tensor_name) before storing: an "
                  "expert scale registered as MOE_EXPERT_WEIGHT is stored as one, and the S1 preload skips it")
    if fix.group(2) != "MOE_INTERMEDIATE":
        fail("R", f"ggml_backend_sycl_try_register_weight_usage stores an expert scale as {fix.group(2)}, not "
                  "MOE_INTERMEDIATE (the class the name rule and the planner give it)")
    store = body.find("g_sycl_weight_usages")
    if store < 0 or fix.start() > store:
        fail("R", "the expert-scale correction in ggml_backend_sycl_try_register_weight_usage comes after the "
                  "registry row is looked up or stored")


def check_one_source(common: str, backend: str) -> None:
    for lit in LITERALS:
        n_common = common.count(lit)
        if n_common != 1:
            fail("S", f"common.hpp spells {lit} {n_common} times outside comments; the predicate is its one home")
        if lit in backend:
            fail("S", f"ggml-sycl.cpp restates {lit}; ask {PRED} instead")


def read_sources() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for key, path in (("common", COMMON), ("backend", BACKEND)):
        out[key] = path.read_text(encoding="utf-8") if path.exists() else None
    return out


def check(s: dict[str, str | None]) -> None:
    if s["common"] is None:
        fail("N0", f"{COMMON.relative_to(ROOT)} is missing")
    if s["backend"] is None:
        fail("R0", f"{BACKEND.relative_to(ROOT)} is missing")
    common  = blank_comments(s["common"])
    backend = blank_comments(s["backend"])
    check_predicate(common)
    check_name_rule(common)
    check_registry(backend)
    check_one_source(common, backend)


# --self-test ---------------------------------------------------------------------------------------------------


def self_test(s: dict[str, str | None]) -> int:
    reg_fix = re.compile(r"if\s*\(\s*mapped\s*==\s*tensor_usage::MOE_EXPERT_WEIGHT\s*&&\s*"
                         + PRED + r"\s*\(\s*tensor_name\s*\)\s*\)\s*\{\s*mapped\s*=\s*tensor_usage::MOE_INTERMEDIATE"
                         r"\s*;\s*\}")

    def move_fix_after_store(b: str) -> str:
        m = reg_fix.search(b)
        if not m:
            return b
        without = b[: m.start()] + b[m.end() :]
        anchor = without.find("return g_sycl_weight_usages.emplace", m.start())
        if anchor < 0:
            return b
        return without[:anchor] + m.group(0) + "\n            " + without[anchor:]

    mutants = [
        ("N", "common", "predicate drops input_scale",
         lambda c: re.sub(r'\s*\|\|\s*strstr\s*\(\s*name\s*,\s*"exps\.input_scale"\s*\)\s*!=\s*nullptr', "", c,
                          count=1)),
        ("N", "common", "predicate renamed away",
         lambda c: c.replace("inline bool " + PRED + "(const char * name)",
                             "inline bool ggml_sycl_moe_expert_aux_name(const char * name)", 1)),
        ("U", "common", "name rule no longer asks the predicate",
         lambda c: re.sub(r"\s*&&\s*!\s*" + PRED + r"\s*\(\s*name\s*\)", "", c, count=1)),
        ("U", "common", "expert scale classified MOE_EXPERT_WEIGHT by a second return",
         lambda c: c.replace("    if (moe_exps_name) {\n        return tensor_usage::MOE_INTERMEDIATE;",
                             "    if (moe_exps_name && " + PRED + "(name)) {\n"
                             "        return tensor_usage::MOE_EXPERT_WEIGHT;\n    }\n"
                             "    if (moe_exps_name) {\n        return tensor_usage::MOE_INTERMEDIATE;", 1)),
        ("R", "backend", "registry correction removed",
         lambda b: reg_fix.sub("", b, count=1)),
        ("R", "backend", "registry correction stores UNKNOWN",
         lambda b: reg_fix.sub(lambda m: m.group(0).replace("tensor_usage::MOE_INTERMEDIATE", "tensor_usage::UNKNOWN"),
                               b, count=1)),
        ("R", "backend", "registry correction moved after the store",
         move_fix_after_store),
        ("S", "backend", "registry restates the suffix",
         lambda b: reg_fix.sub(lambda m: m.group(0) + "\n        if (mapped == tensor_usage::MOE_EXPERT_WEIGHT && "
                               'strstr(tensor_name, "exps.input_scale")) {\n            mapped = '
                               "tensor_usage::MOE_INTERMEDIATE;\n        }", b, count=1)),
        ("S", "common", "name rule restates the suffix",
         lambda c: c.replace("    if (moe_exps_name) {\n        return tensor_usage::MOE_INTERMEDIATE;",
                             '    if (moe_exps_name && strstr(name, "exps.scale")) {\n'
                             "        return tensor_usage::MOE_INTERMEDIATE;\n    }\n"
                             "    if (moe_exps_name) {\n        return tensor_usage::MOE_INTERMEDIATE;", 1)),
    ]
    failures = []
    caught = 0
    for pin, key, name, mutate in mutants:
        m = dict(s)
        m[key] = mutate(s[key] or "")
        if m[key] == s[key]:
            failures.append(f"mutant '{name}' did not apply")
            print(f"  self-test: mutant '{name}' did not apply")
            continue
        try:
            check(m)
        except ContractError as e:
            if f"[pin {pin}]" in str(e):
                caught += 1
                print(f"  self-test: mutant '{name}' caught: {e}")
                continue
            failures.append(f"mutant '{name}' failed on the wrong pin: {e}")
            print(f"  self-test: mutant '{name}' failed on the wrong pin: {e}")
            continue
        failures.append(f"mutant '{name}' passed")
        print(f"  self-test: mutant '{name}' PASSED (not caught)")
    try:
        check(s)
        print("  self-test: unmodified tree passes")
    except ContractError as e:
        failures.append(f"unmodified tree fails: {e}")
        print(f"  self-test: unmodified tree FAILS: {e}")
    if failures:
        print(f"SELF-TEST FAIL: {', '.join(failures)}")
        return 1
    print(f"SELF-TEST PASS: {caught} mutants caught, unmodified tree passes")
    return 0


def main(argv: list[str]) -> int:
    s = read_sources()
    if "--self-test" in argv:
        rc = self_test(s)
        if rc != 0:
            return rc
    try:
        check(s)
    except ContractError as e:
        print(e)
        return 1
    print("PASS: an expert family's per-expert scale is MOE_INTERMEDIATE to the name rule and to the weight-usage "
          f"registry, through one predicate {TAG}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
