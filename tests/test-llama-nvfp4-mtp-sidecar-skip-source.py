#!/usr/bin/env python3
"""Source pins: a skipped MTP block's NVFP4 scale tensors are skipped with the block's weights.

An NVFP4 GGUF carries a ".scale" and an ".input_scale" tensor next to each quantized weight, and the MTP block's are in
the file too. When the loader skips that block's weights, their pointers stay null, and the generic sidecar pass in
llama_model::load_tensors only asks for the sidecar of a weight that was loaded. Nothing then counts the skipped
block's sidecars, and ml.done_getting_tensors() throws "wrong number of tensors; expected N, got N-22". The skip is
checked in glm4, bailingmoe2, exaone4, exaone-moe and dots3note, which skip the MTP block whatever ml.load_mtp says, and
in qwen35moe, which skips it when ml.load_mtp is false.

The rule the fix pins:
- the pass asks, for every layer in the MTP range [hparams.n_layer(), n_layer_all), for the sidecars of each weight kind
  whose pointer is null, with TENSOR_NOT_REQUIRED | TENSOR_SKIP, so a sidecar that is in the file is counted once and
  ignored, exactly as its weight was, and one that is not in the file counts nothing;
- both asks also carry TENSOR_SKIP_IF_VIRTUAL: a virtual load (no file) ignores TENSOR_SKIP and would build an orphan
  tensor for each ask, while that flag makes it return nullptr with nothing counted;
- it never reads ml.load_mtp: the architectures that skip the block whatever the caller asked would otherwise keep
  the hole, so the null weight, not the flag, says the loader skipped it;
- a sidecar of a loaded weight (non-null pointer) is asked for by the existing sites, without TENSOR_SKIP, and the
  two sets of asks are disjoint, so no sidecar is created twice (a second create_tensor of a present tensor would count
  it twice and fail with "too many tensors created");
- every weight kind that has a ".scale" or an ".input_scale" site in the existing pass has its skipped counterpart,
  with the same dims (create_tensor checks the dims before the skip path returns), so a sidecar site added later
  cannot leave a hole;
- all of it sits after load_arch_tensors(ml) and before ml.done_getting_tensors().
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LLAMA_MODEL_CPP = (ROOT / "src/llama-model.cpp").read_text()

# One left-to-right pass, so a `/*` inside a `//` comment cannot open a block comment.
_LEXEME_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"'  # string literal (kept)
    r"|'(?:\\.|[^'\\\n])*'"  # char literal (kept)
    r"|//[^\n]*"  # line comment (dropped)
    r"|/\*.*?\*/",  # block comment (dropped)
    flags=re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def squash(text: str) -> str:
    """One space for every whitespace run, and none after `(` or before `)` and `,`, so a call clang-format wraps
    still matches its one-line spelling."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return re.sub(r"\s+,", ",", text)


CODE = squash(strip_comments(LLAMA_MODEL_CPP))

_PASS_START = "load_arch_tensors(ml);"
_PASS_END = "ml.done_getting_tensors();"
_SKIP_HELPER = "const auto skip_scales = [&](int il, llm_tensor tensor, const ggml_tensor * weight, int64_t n) {"
_SKIP_LOOP = "for (int i = hparams.n_layer(); i < n_layer_all; ++i) {"
_SKIP_GUARD = "if (weight == nullptr) {"
_SKIP_FLAGS = "TENSOR_NOT_REQUIRED | TENSOR_SKIP | TENSOR_SKIP_IF_VIRTUAL"
_SKIP_FLAGS_DEF = f"const int skip = {_SKIP_FLAGS};"
_SKIP_SCALE = 'create_tensor(tn(tensor, "scale", il), {n}, skip);'
_SKIP_INPUT_SCALE = 'create_tensor(tn(tensor, "input_scale", il), {n}, skip);'


def _site_re(suffix: str) -> re.Pattern:
    # `if (!layer.wq_s && layer.wq) { layer.wq_s = create_tensor(tn(LLM_TENSOR_ATTN_Q, "scale", i), {1}, ...); }`
    return re.compile(
        r"if \(!layer\.[\w.]+ && layer\.([\w.]+)\) \{ layer\.[\w.]+ = create_tensor\(tn\((LLM_TENSOR_\w+), "
        rf'"{suffix}", i\), \{{([^}}]*)\}}, TENSOR_NOT_REQUIRED\); \}}'
    )


_SITE_RES = {"scale": _site_re("scale"), "input_scale": _site_re("input_scale")}
_CALL_RE = re.compile(r"skip_scales\(i, (LLM_TENSOR_\w+), layer\.([\w.]+), ([^)]*)\);")


def sidecar_pass(code: str) -> str:
    assert code.count(_PASS_START) == 1, f"not exactly one {_PASS_START!r}"
    start = code.find(_PASS_START)
    end = code.find(_PASS_END, start)
    assert end != -1, "the pass does not end at ml.done_getting_tensors()"
    return code[start:end]


def matching_brace(code: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(code)):
        if code[i] == "{":
            depth += 1
        elif code[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    raise AssertionError("unbalanced braces")


def skip_loop_body(body: str):
    """The text of the loop over the MTP range, or None."""
    at = body.find(_SKIP_LOOP)
    if at == -1:
        return None
    open_idx = at + len(_SKIP_LOOP) - 1
    return body[open_idx : matching_brace(body, open_idx) + 1]


def skip_helper_body(body: str):
    """The text of the helper that asks for one weight kind's sidecars, or None."""
    at = body.find(_SKIP_HELPER)
    if at == -1:
        return None
    open_idx = at + len(_SKIP_HELPER) - 1
    return body[open_idx : matching_brace(body, open_idx) + 1]


def skip_flags_violations(code: str) -> list:
    body = sidecar_pass(code)
    helper = skip_helper_body(body)
    if helper is None:
        return ["the skipped-sidecar helper is not found in the generic pass"]
    violations = []
    if not helper.startswith("{ " + _SKIP_GUARD):
        violations.append("the helper must ask only when the weight pointer is null")
    if helper.count(_SKIP_FLAGS_DEF) != 1:
        violations.append(f"the helper must define its flags once as `{_SKIP_FLAGS}` (the last keeps a virtual load clean)")
    if helper.count(_SKIP_SCALE) != 1:
        violations.append('the helper must ask for the ".scale" sidecar once, with the skip flags')
    if helper.count(_SKIP_INPUT_SCALE) != 1:
        violations.append('the helper must ask for the ".input_scale" sidecar once, with the skip flags')
    skips = len(re.findall(r"\bTENSOR_SKIP\b", body))
    if skips != 1:
        violations.append(f"TENSOR_SKIP must appear once in the pass (the helper's flags), not {skips}: a loaded sidecar stays")
    virtual = len(re.findall(r"\bTENSOR_SKIP_IF_VIRTUAL\b", body))
    if virtual != 1:
        violations.append(f"TENSOR_SKIP_IF_VIRTUAL must appear once in the pass (the helper's flags), not {virtual}")
    return violations


def skip_range_violations(code: str) -> list:
    body = sidecar_pass(code)
    loop = skip_loop_body(body)
    if loop is None:
        return ["the loop over the MTP range [hparams.n_layer(), n_layer_all) is not found in the generic pass"]
    violations = []
    helper_at = body.find(_SKIP_HELPER)
    if helper_at == -1 or helper_at > body.find(_SKIP_LOOP):
        violations.append("the helper must be defined before the loop that calls it")
    outside = body.replace(loop, "", 1)
    if "skip_scales(" in outside.replace(_SKIP_HELPER, "", 1):
        violations.append("the helper must be called only from the loop over the MTP range")
    if "ml.load_mtp" in loop:
        violations.append("the rule must not read ml.load_mtp: some architectures skip the MTP block unconditionally")
    return violations


def site_pairs(body: str, suffix: str) -> set:
    """(tensor, weight, dims) of every `suffix` sidecar site of the loop over all layers."""
    return {(m.group(2), m.group(1), m.group(3)) for m in _SITE_RES[suffix].finditer(body)}


def skip_pairs_violations(code: str) -> list:
    body = sidecar_pass(code)
    loop = skip_loop_body(body)
    if loop is None:
        return ["the loop over the MTP range is not found in the generic pass"]
    # the helper takes the bare dims figure; the existing sites write `{1}` and `{n_expert}`
    have = {(m.group(1), m.group(2), m.group(3)) for m in _CALL_RE.finditer(loop)}
    violations = []
    for suffix in _SITE_RES:
        wanted = site_pairs(body.replace(loop, "", 1), suffix)
        if not wanted:
            violations.append(f"no .{suffix} site is found in the generic pass")
            continue
        for tensor, weight, dims in sorted(wanted - have):
            violations.append(f".{suffix} of {tensor} (layer.{weight}, dims {{{dims}}}) has a site, no skipped counterpart")
        for tensor, weight, dims in sorted(have - wanted):
            violations.append(f"{tensor} (layer.{weight}, dims {{{dims}}}) is skipped but has no .{suffix} site")
    return violations


def skip_order_violations(code: str) -> list:
    start = code.find(_PASS_START)
    end = code.find(_PASS_END, start)
    helper = code.find(_SKIP_HELPER, start)
    loop = code.find(_SKIP_LOOP, start)
    if start == -1 or end == -1:
        return ["the generic pass anchors are not found"]
    if helper == -1 or loop == -1:
        return ["the skipped-sidecar helper or its loop is not found"]
    violations = []
    if not start < helper < end or not start < loop < end:
        violations.append("the skipped-sidecar asks must sit after load_arch_tensors(ml) and before done_getting_tensors()")
    all_layers = code.find("for (int i = 0; i < n_layer_all; ++i) {", start)
    if all_layers == -1 or not all_layers < helper:
        violations.append("the skipped-sidecar asks must follow the loop that asks for a loaded weight's sidecars")
    return violations


def test_skipped_sidecars_are_asked_with_not_required_skip():
    assert skip_flags_violations(CODE) == []


def test_skip_flags_check_has_mutation_witnesses():
    assert CODE.count(_SKIP_SCALE) == 1 and CODE.count(_SKIP_INPUT_SCALE) == 1, "each ask must appear exactly once"
    assert CODE.count(_SKIP_FLAGS_DEF) == 1, "the helper's flags must be defined exactly once"
    for needle, repl in (
        (_SKIP_FLAGS_DEF, _SKIP_FLAGS_DEF.replace(" | TENSOR_SKIP_IF_VIRTUAL", "")),
        (_SKIP_FLAGS_DEF, _SKIP_FLAGS_DEF.replace(" | TENSOR_SKIP |", " |")),
        (_SKIP_FLAGS_DEF, _SKIP_FLAGS_DEF.replace("TENSOR_NOT_REQUIRED | ", "")),
        (_SKIP_SCALE, _SKIP_SCALE.replace(", skip)", ", TENSOR_NOT_REQUIRED | TENSOR_SKIP)")),
        (_SKIP_INPUT_SCALE, _SKIP_INPUT_SCALE.replace(", skip)", ", TENSOR_NOT_REQUIRED)")),
        (_SKIP_GUARD, "if (weight != nullptr) {"),
        (_SKIP_GUARD, "if (true) {"),
    ):
        mutated = CODE.replace(needle, repl, 1)
        assert mutated != CODE
        assert skip_flags_violations(mutated) != [], f"the check must catch {needle!r} becoming {repl!r}"
    # a loaded weight's sidecar skipped too: one existing site gains the skip flags
    site = 'create_tensor(tn(LLM_TENSOR_ATTN_Q, "scale", i), {1}, TENSOR_NOT_REQUIRED);'
    assert CODE.count(site) == 1
    for extra in ("TENSOR_NOT_REQUIRED | TENSOR_SKIP", _SKIP_FLAGS):
        mutated = CODE.replace(site, site.replace("TENSOR_NOT_REQUIRED", extra), 1)
        assert mutated != CODE
        assert skip_flags_violations(mutated) != [], "the check must catch a loaded weight's sidecar being skipped"


def test_skipped_sidecars_cover_exactly_the_mtp_range():
    assert skip_range_violations(CODE) == []


def test_skip_range_check_has_mutation_witnesses():
    assert CODE.count(_SKIP_LOOP) == 1, "the loop must appear exactly once"
    for repl in (
        "for (int i = 0; i < n_layer_all; ++i) {",
        "for (int i = hparams.n_layer(); i < n_layer_all && !ml.load_mtp; ++i) {",
    ):
        mutated = CODE.replace(_SKIP_LOOP, repl, 1)
        assert mutated != CODE
        assert skip_range_violations(mutated) != [], f"the check must catch the loop becoming {repl!r}"
    mutated = CODE.replace(_SKIP_LOOP, _SKIP_LOOP + " if (!ml.load_mtp) { continue; }", 1)
    assert mutated != CODE
    assert skip_range_violations(mutated) != [], "the check must catch the loop reading ml.load_mtp"


def test_every_sidecar_site_has_a_skipped_counterpart():
    assert skip_pairs_violations(CODE) == []


def test_skip_pairs_check_has_mutation_witnesses():
    calls = _CALL_RE.findall(CODE)
    assert len(calls) >= 21, f"expected a call per weight kind, found {len(calls)}"
    first = _CALL_RE.search(CODE).group(0)
    assert CODE.count(first) == 1
    removed = CODE.replace(first, "", 1)
    assert removed != CODE
    assert skip_pairs_violations(removed) != [], "the check must catch a weight kind losing its skipped counterpart"
    exps = next(m.group(0) for m in _CALL_RE.finditer(CODE) if m.group(3).strip() == "n_expert")
    assert CODE.count(exps) == 1
    wrong_dims = CODE.replace(exps, exps.replace("n_expert", "1"), 1)
    assert wrong_dims != CODE
    assert skip_pairs_violations(wrong_dims) != [], "the check must catch a skipped counterpart with other dims"
    # a new sidecar site with no counterpart
    site = 'create_tensor(tn(LLM_TENSOR_ATTN_Q, "scale", i), {1}, TENSOR_NOT_REQUIRED); }'
    assert CODE.count(site) == 1
    extra = (
        "if (!layer.wx_s && layer.wx) { layer.wx_s = create_tensor(tn(LLM_TENSOR_ATTN_Q, "
        '"scale", i), {1}, TENSOR_NOT_REQUIRED); } '
    )
    added = CODE.replace(site, site + " " + extra, 1)
    assert added != CODE
    assert skip_pairs_violations(added) != [], "the check must catch a sidecar site added without a counterpart"
    # the same three mutations on the ".input_scale" sites, which the ".scale" sites alone would not catch
    in_site = 'create_tensor(tn(LLM_TENSOR_ATTN_Q, "input_scale", i), {1}, TENSOR_NOT_REQUIRED); }'
    assert CODE.count(in_site) == 1
    in_start = CODE.rfind("if (", 0, CODE.find(in_site))
    in_text = CODE[in_start : CODE.find(in_site) + len(in_site)]
    assert in_text.startswith("if (!layer.wq_in_s && layer.wq) {"), in_text
    no_in = CODE.replace(in_text, "", 1)
    assert no_in != CODE
    assert skip_pairs_violations(no_in) != [], "the check must catch a missing .input_scale site"
    in_added = CODE.replace(in_site, in_site + " " + extra.replace('"scale"', '"input_scale"'), 1)
    assert in_added != CODE
    assert skip_pairs_violations(in_added) != [], "the check must catch an .input_scale site without a counterpart"
    in_exps = 'create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "input_scale", i), {n_expert}, TENSOR_NOT_REQUIRED);'
    assert CODE.count(in_exps) == 1
    in_dims = CODE.replace(in_exps, in_exps.replace("{n_expert}", "{1}"), 1)
    assert in_dims != CODE
    assert skip_pairs_violations(in_dims) != [], "the check must catch an .input_scale site with other dims"


def test_skipped_sidecars_sit_between_the_arch_loader_and_the_tensor_count():
    assert skip_order_violations(CODE) == []


def test_skip_order_check_has_mutation_witnesses():
    assert CODE.count(_PASS_END) == 1, "the pass must end at one ml.done_getting_tensors()"
    # the count check moves in front of the skipped asks
    early = CODE.replace(_PASS_END, "", 1)
    at = early.find(_SKIP_HELPER)
    early = early[:at] + _PASS_END + " " + early[at:]
    assert early != CODE
    assert skip_order_violations(early) != [], "the check must catch the tensor count before the skipped asks"
    # the helper and its loop move in front of the loop for loaded weights
    helper_at = CODE.find(_SKIP_HELPER)
    all_layers = CODE.find("for (int i = 0; i < n_layer_all; ++i) {", CODE.find(_PASS_START))
    assert all_layers != -1 and all_layers < helper_at
    moved = CODE[:all_layers] + CODE[helper_at : CODE.find(_PASS_END, helper_at)] + CODE[all_layers:helper_at]
    moved += CODE[CODE.find(_PASS_END, helper_at) :]
    assert moved != CODE
    assert skip_order_violations(moved) != [], "the check must catch the skipped asks before the loaded weights' loop"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
