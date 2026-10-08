#!/usr/bin/env python3
"""Every fused-kernel site that reads an input while it writes an output must call the aliasing gate
(ggml/src/ggml-sycl/fusion-alias.hpp) before it launches -- source gated because the call is a
DECISION a later "simplify" pass could delete without any functional test noticing: the unfused
and the fused path print the same tokens on every layout except the one that races.

llama.cpp-rb2h: ggml_gallocr allocates in unfused order, so a fused chain's output can land
partially over a dead input (the fused ADD+RMS_NORM of a Qwen3.6-27B layer sat 8192 B below the ADD
operand it overlapped, and B50 perplexity varied run to run). Each site declines the fusion when the
gate does not admit its layout; the unfused kernels are correct for any layout.

Gated sites (the host test tests/test-sycl-fusion-alias.cpp pins what each admits and declines):
  bit1 RMS_NORM+MUL+ADD, bit4 RMS_NORM+MUL, bit5 ADD+RMS_NORM, bit6 MUL+ADD   (chains in graph_compute_impl)
  bit0 MUL_MAT+ADD (ggml_sycl_try_fuse_tg_mul_mat_add), bit0 router (ggml_sycl_try_fuse_tg_router_f32_add_argsort)
Not gated, by design and stated in the commit: bit2/bit3 (the activation is quantised to scratch by a
separate kernel before the MMQ kernel runs), and the opt-in fusions (MoE down-weighted-sum,
GGML_SYCL_FFN_FUSION, GGML_SYCL_PERSISTENT_TG_RMS_FUSION), which are tracked on a follow-up ticket.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"

# (site enumerator, launch marker, enclosing function signature or None for graph_compute_impl,
#  extra markers that must also come after the gate: a drain or state change that precedes the launch)
CHAIN_SITES = [
    ("GGML_SYCL_FUSION_SITE_RMS_NORM_MUL_ADD", "ggml_sycl_op_rms_norm_fused_add(*sycl_ctx"),
    ("GGML_SYCL_FUSION_SITE_RMS_NORM_MUL", "ggml_sycl_op_rms_norm_fused(*sycl_ctx"),
    ("GGML_SYCL_FUSION_SITE_ADD_RMS_NORM", "ggml_sycl_op_add_rms_norm_fused(*sycl_ctx"),
    ("GGML_SYCL_FUSION_SITE_MUL_ADD", "ggml_sycl_op_mul_add_fused(*sycl_ctx"),
]
FUNCTION_SITES = [
    (
        "GGML_SYCL_FUSION_SITE_MUL_MAT_ADD",
        "static bool ggml_sycl_try_fuse_tg_mul_mat_add(",
        ["split_merge_drain();", "ggml_sycl_mmvq_set_fused_add("],
    ),
    (
        "GGML_SYCL_FUSION_SITE_ROUTER",
        "static bool ggml_sycl_try_fuse_tg_router_f32_add_argsort(",
        ["split_merge_drain();", "ggml_sycl_router_f32_bias_argsort_sycl(*ctx.stream()"],
    ),
]


# What each bit0 site must hand to the gate, whitespace-insensitive: the output it writes (and, for the router,
# the argsort), the operands it reads, whether each may be in place, and the counts; and the decline itself, as the
# whole `if (!gate(...)) { return ...; }` statement, so a decline cannot be dead behind `&& false`. A different
# tensor, a dropped operand or `in_place_ok` flipped on the router activation (read by every subgroup) is a wiring
# bug the host test cannot see.
FUNCTION_PINS = {
    "GGML_SYCL_FUSION_SITE_MUL_MAT_ADD": [
        "constggml_sycl_fusion_operandwrites={add,out_ptr,true};",
        "constggml_sycl_fusion_operandreads={addend,addend_ptr,true};",
        "if(!ggml_sycl_fusion_alias_admit(GGML_SYCL_FUSION_SITE_MUL_MAT_ADD,add->name,&writes,1,&reads,1)){returnfalse;}",
    ],
    "GGML_SYCL_FUSION_SITE_ROUTER": [
        "constggml_sycl_fusion_operandwrites[2]={{add,probs_ptr,true},{sort,sort_ptr,true}};",
        "constggml_sycl_fusion_operandreads[2]={{act,act_ptr,false},{addend,bias_ptr,true}};",
        "if(!ggml_sycl_fusion_alias_admit(GGML_SYCL_FUSION_SITE_ROUTER,add->name,writes,2,reads,2)){returnreject(\"alias\");}",
    ],
}


# The three helpers every gate call goes through, pinned whole: the function text with comments stripped,
# whitespace squashed and the log message's wording elided must EQUAL the text below. The host test exercises
# the header's predicate; these live in ggml-sycl.cpp next to the process-global counters, so a helper that
# counted and then answered `true` (an early return, a changed comparison, a resolver lambda that hands the
# predicate no pointer) would pass the host test and every call-site pin above. Equality, not a substring: any
# statement added anywhere in a helper changes the text.
WRAPPER_BODIES = [
    (
        "static bool ggml_sycl_fusion_alias_account(",
        """
static bool ggml_sycl_fusion_alias_account(ggml_sycl_fusion_alias_site site, const char * start_name,
                                           const ggml_sycl_fusion_alias_result & r) {
    ggml_sycl_fusion_alias_stats & st = ggml_sycl_fusion_alias_stats_get();
    st.checked[site].fetch_add(1, std::memory_order_relaxed);
    if (r.verdict == GGML_SYCL_FUSION_ALIAS_SAFE) {
        return true;
    }
    st.declined[site][r.verdict].fetch_add(1, std::memory_order_relaxed);
    const uint64_t n = st.declined_total.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n <= 4 || (n & (n - 1)) == 0) {
        GGML_LOG_WARN(
            "[SYCL-FUSION] declined the %s fusion at %s: %s (%s against %s); running the unfused kernels "
            "(decline %llu)\n",
            ggml_sycl_fusion_alias_site_name(site), start_name ? start_name : "?",
            ggml_sycl_fusion_alias_verdict_name(r.verdict), r.write && r.write->name ? r.write->name : "?",
            r.other && r.other->name ? r.other->name : "?", (unsigned long long) n);
    }
    return false;
}
""",
    ),
    (
        "static bool ggml_sycl_fusion_alias_admit(",
        """
static bool ggml_sycl_fusion_alias_admit(ggml_sycl_fusion_alias_site site, const char * start_name,
                                         const ggml_sycl_fusion_operand * writes, int n_writes,
                                         const ggml_sycl_fusion_operand * reads, int n_reads) {
    return ggml_sycl_fusion_alias_account(site, start_name,
                                          ggml_sycl_fusion_alias_check(writes, n_writes, reads, n_reads));
}
""",
    ),
    (
        "static bool ggml_sycl_fusion_alias_admit_chain(",
        """
static bool ggml_sycl_fusion_alias_admit_chain(ggml_sycl_fusion_alias_site site, const ggml_cgraph * cgraph,
                                               int node_idx, int device) {
    const bool norm_chain =
        site == GGML_SYCL_FUSION_SITE_RMS_NORM_MUL_ADD || site == GGML_SYCL_FUSION_SITE_RMS_NORM_MUL;
    const ggml_tensor * norm_src =
        norm_chain && cgraph && node_idx >= 0 && node_idx < cgraph->n_nodes ? cgraph->nodes[node_idx]->src[0] : nullptr;
    const ggml_sycl_fusion_alias_result r = ggml_sycl_fusion_chain_alias_check(
        cgraph, node_idx, site, [&](const ggml_tensor * t, bool is_write) -> const void * {
            if (is_write || !norm_chain || t == norm_src) {
                return ggml_sycl_get_data_ptr(t, device);
            }
            return ggml_sycl_resolve_tensor_ptr(t, device);
        });
    return ggml_sycl_fusion_alias_account(
        site, cgraph && node_idx >= 0 && node_idx < cgraph->n_nodes ? cgraph->nodes[node_idx]->name : nullptr, r);
}
""",
    ),
]


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def elide_string_literals(text: str) -> str:
    """Replace every string literal with `""` (adjacent literals collapse into one), so rewording or re-wrapping
    the log message is free while every argument expression, and any statement hidden among the arguments, stays
    in the compared text. Char literals are skipped over, not elided."""
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '"':
            i += 1
            while i < len(text) and text[i] != '"':
                i += 2 if text[i] == "\\" else 1
            i += 1
            out.append('""')
        elif ch == "'":
            j = i + 1
            while j < len(text) and text[j] != "'":
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:j + 1])
            i = j + 1
        else:
            out.append(ch)
            i += 1
    return re.sub(r'(?:""\s*)+""|""', '""', "".join(out))


def first_difference(expected: str, actual: str, context: int = 40) -> str:
    at = next((k for k in range(min(len(expected), len(actual))) if expected[k] != actual[k]), min(len(expected), len(actual)))
    lo = max(0, at - context)
    return f"first difference at offset {at} of the whitespace-squashed, string-elided text:\n  expected: ...{expected[lo:at + context]}...\n  actual:   ...{actual[lo:at + context]}..."


# Macros that would change what a helper body means without touching the body. The gate cannot own other
# headers, but it owns this file. The set is derived, not listed: every identifier token of the pinned helper
# text (names, keywords, locals, members) plus the names the helpers rely on without spelling them.
def forbidden_macro_names() -> set[str]:
    names = {"GGML_LOG_WARN", "true", "false", "return", "if", "nullptr"}
    for _, expected in WRAPPER_BODIES:
        names.update(re.findall(r"[A-Za-z_]\w*", elide_string_literals(expected)))
    return names


def redefinition_violations(code: str) -> list[str]:
    """`#define` / `#undef` of a forbidden name anywhere in the file. Backslash-continued lines are joined first, so
    `#define \\<nl>true false` and `#\\<nl>define true false` are read as what the preprocessor reads."""
    joined = code.replace("\\\n", "")
    names = forbidden_macro_names()
    found: list[str] = []
    for m in re.finditer(r"^[ \t]*#[ \t]*(?:define|undef)[ \t]+([A-Za-z_]\w*)", joined, re.M):
        name = m.group(1)
        if name in names or name.startswith(("GGML_SYCL_FUSION_ALIAS_", "GGML_SYCL_FUSION_SITE_")):
            found.append(name)
    return found


def matching_brace(text: str, open_idx: int) -> int:
    """Comment/string/char-literal-aware brace match. A self-contained copy of the walker in this fork's
    other source-gate tests -- the convention is one self-contained file per check, not a shared import."""
    assert text[open_idx] == "{"
    depth = 0
    state = "code"
    i = open_idx
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                state = "line"
                i += 1
            elif ch == "/" and nxt == "*":
                state = "block"
                i += 1
            elif ch == '"':
                state = "string"
            elif ch == "'":
                state = "char"
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
        elif state == "line":
            if ch == "\n":
                state = "code"
        elif state == "block":
            if ch == "*" and nxt == "/":
                state = "code"
                i += 1
        else:
            quote = '"' if state == "string" else "'"
            if ch == "\\":
                i += 1
            elif ch == quote:
                state = "code"
        i += 1
    raise AssertionError("unclosed brace")


def function(text: str, signature: str) -> str | None:
    if signature not in text:
        return None
    start = text.index(signature)
    brace = text.index("{", start)
    return text[start:matching_brace(text, brace) + 1]


def strip_comments(text: str) -> str:
    """Blank out // and /* */ comments so a marker quoted in prose is not mistaken for a call."""
    text = re.sub(r"/\*.*?\*/", lambda m: " " * len(m.group(0)), text, flags=re.S)
    return re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), text)


def gate_call(site: str) -> re.Pattern:
    return re.compile(r"ggml_sycl_fusion_alias_admit(?:_chain)?\(\s*" + re.escape(site) + r"\b")


def matching_paren(text: str, open_idx: int) -> int:
    """Index of the ')' matching the '(' at open_idx. The text has had comments blanked; string and char
    literals are skipped."""
    assert text[open_idx] == "("
    depth = 0
    i = open_idx
    quote = ""
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 1
            elif ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise AssertionError("unclosed paren")


def top_level_split(cond: str) -> tuple[list[str], bool, bool]:
    """Split a condition at its depth-0 `&&`; the bools say whether a depth-0 `||` and a depth-0 `?` are present.
    An unparenthesised `?:` binds looser than `&&`, so `a ? b : c && gate` makes the gate the false arm only."""
    parts: list[str] = []
    depth = 0
    start = 0
    has_or = False
    has_ternary = False
    i = 0
    while i < len(cond):
        ch = cond[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0 and cond.startswith("&&", i):
            parts.append(cond[start:i].strip())
            start = i + 2
            i += 1
        elif depth == 0 and cond.startswith("||", i):
            has_or = True
            i += 1
        elif depth == 0 and ch == "?":
            has_ternary = True
        i += 1
    parts.append(cond[start:].strip())
    return parts, has_or, has_ternary


CONSTANT_CONJUNCTS = {"true", "false", "0", "1", "(true)", "(false)", "(0)", "(1)"}


def chain_wiring_violations(graph: str, site: str, gate_at: int, launch_at: int) -> list[str]:
    """The gate call must be a top-level `&&` conjunct, verbatim, of the `if (...)` condition whose body holds
    the launch: not negated, not OR'd, not behind a constant, not given another node index or device."""
    found: list[str] = []
    exact = re.compile(
        r"ggml_sycl_fusion_alias_admit_chain\(\s*" + re.escape(site) + r"\s*,\s*cgraph\s*,\s*i\s*,\s*sycl_ctx->device\s*\)"
    )
    call = exact.match(graph, gate_at)
    if call is None:
        return [f"{site}: the gate call must read exactly (SITE, cgraph, i, sycl_ctx->device): a different node index or device guards another chain"]

    # The condition that contains the call: the nearest `if (` whose paren is still open at the call.
    cond_open = -1
    for m in reversed(list(re.finditer(r"\bif\s*\(", graph[:gate_at]))):
        open_idx = m.end() - 1
        if matching_paren(graph, open_idx) > gate_at:
            cond_open = open_idx
            break
    if cond_open < 0:
        return [f"{site}: the gate call is not inside an `if (...)` condition guarding the launch"]
    cond_close = matching_paren(graph, cond_open)
    cond = graph[cond_open + 1:cond_close]
    conjuncts, has_or, has_ternary = top_level_split(cond)
    if has_or:
        found.append(f"{site}: the guarding condition contains a top-level `||`, so the gate need not hold")
    if has_ternary:
        found.append(f"{site}: the guarding condition contains a top-level `?:`, which can make the gate one arm only")
    if not any(exact.fullmatch(c) for c in conjuncts):
        found.append(
            f"{site}: the gate call is not a bare top-level `&&` conjunct of the guarding condition "
            "(negated, parenthesised, behind `0 &&`, or combined with another expression)"
        )
    for c in conjuncts:
        if c in CONSTANT_CONJUNCTS:
            found.append(f"{site}: the guarding condition has a constant conjunct `{c}`")
    # The launch lives in the body this condition guards.
    rest = graph[cond_close + 1:]
    body_open = cond_close + 1 + (len(rest) - len(rest.lstrip()))
    if graph[body_open:body_open + 1] != "{":
        found.append(f"{site}: the guarding `if` has no braced body")
    else:
        body_close = matching_brace(graph, body_open)
        if not body_open < launch_at < body_close:
            found.append(f"{site}: the launch is not inside the body the gate guards")
    return found


def gate_violations(source: str) -> list[str]:
    found: list[str] = []
    code = strip_comments(source)

    graph = function(code, "static void ggml_backend_sycl_graph_compute_impl(")
    if graph is None:
        # the graph loop has been renamed; search the whole file rather than silently passing
        graph = code

    for site, launch in CHAIN_SITES:
        launches = [m.start() for m in re.finditer(re.escape(launch), graph)]
        gates = [m.start() for m in gate_call(site).finditer(graph)]
        if len(launches) != 1:
            found.append(
                f"{site}: expected exactly one launch `{launch}`, found {len(launches)} -- a second launch of "
                "the fused kernel needs its own gate call"
            )
            continue
        if len(gates) != 1:
            found.append(f"{site}: expected exactly one gate call for this site, found {len(gates)}")
            continue
        if not gates[0] < launches[0]:
            found.append(f"{site}: the gate call must precede the launch `{launch}`")
            continue
        found.extend(chain_wiring_violations(graph, site, gates[0], launches[0]))

    for site, signature, markers in FUNCTION_SITES:
        body = function(code, signature)
        if body is None:
            found.append(f"{site}: {signature} is missing")
            continue
        gates = [m.start() for m in gate_call(site).finditer(body)]
        if len(gates) != 1:
            found.append(f"{site}: expected exactly one gate call inside {signature}, found {len(gates)}")
            continue
        for marker in markers:
            at = body.find(marker)
            if at < 0:
                found.append(f"{site}: `{marker}` is missing from {signature}; update this gate with the launch")
            elif not gates[0] < at:
                found.append(f"{site}: the gate call must precede `{marker}` in {signature}")
        squashed = re.sub(r"\s+", "", body)
        for pin in FUNCTION_PINS[site]:
            if pin not in squashed:
                found.append(f"{site}: the operands handed to the gate changed; expected `{pin}` in {signature}")
        # A declined fusion must return false (the router's `reject` helper returns false) so the unfused
        # kernels run.
        tail = body[gates[0]:gates[0] + 700]
        if "return false;" not in tail and "return reject(" not in tail:
            found.append(f"{site}: no `return false;` follows the gate call, so a decline would still launch")

    for name in redefinition_violations(code):
        found.append(f"gate helper: ggml-sycl.cpp redefines `{name}` with a preprocessor directive, which changes what the pinned helpers mean")

    for signature, expected in WRAPPER_BODIES:
        defined = code.count(signature)
        if defined != 1:
            found.append(
                f"gate helper: {signature} must be defined exactly once, found {defined}"
                + (" (an overload could take the calls)" if defined else " (it is missing)")
            )
            continue
        try:
            body = function(code, signature)
        except AssertionError as e:
            found.append(f"gate helper: {signature} is unparsable ({e})")
            continue
        assert body is not None
        actual = squash(elide_string_literals(body))
        want = squash(elide_string_literals(expected))
        if actual != want:
            found.append(
                f"gate helper: {signature} changed; it must read exactly the pinned text, so every verdict still "
                f"reaches the account and only SAFE admits.\n  {first_difference(want, actual)}"
            )

    return found


def test_every_fused_site_calls_the_gate() -> None:
    assert gate_violations(SOURCE.read_text()) == []


def mutate_remove_gate(source: str, site: str) -> str:
    """Replace the gate call at `site` with `true` / `(void) 0`, i.e. the fused kernel launches unconditionally."""
    pattern = re.compile(
        r"ggml_sycl_fusion_alias_admit(?:_chain)?\(\s*" + re.escape(site) + r"\b[^;{]*?\)(?=\s*(?:\)|\{|&&|;|\|\|))",
        re.S,
    )
    mutated, n = pattern.subn("true", source, count=1)
    assert n == 1, f"could not isolate the gate call for {site} to remove it"
    return mutated


def test_removing_each_gate_is_witnessed() -> None:
    """Delete the gate call at each site in turn and confirm the checker names that site: a positive control
    against the check itself being vacuous."""
    source = SOURCE.read_text()
    for site, _ in CHAIN_SITES:
        violations = gate_violations(mutate_remove_gate(source, site))
        assert any(v.startswith(site + ":") for v in violations), f"removing the {site} gate was not witnessed: {violations}"
    for site, _, _ in FUNCTION_SITES:
        violations = gate_violations(mutate_remove_gate(source, site))
        assert any(v.startswith(site + ":") for v in violations), f"removing the {site} gate was not witnessed: {violations}"


def test_a_second_ungated_launch_is_witnessed() -> None:
    source = SOURCE.read_text()
    for site, launch in CHAIN_SITES:
        mutated = source.replace(launch, launch + ");\n    " + launch, 1)
        assert mutated != source
        violations = gate_violations(mutated)
        assert any(v.startswith(site + ": expected exactly one launch") for v in violations), (
            f"a duplicated {site} launch was not witnessed: {violations}"
        )


def test_gate_after_launch_is_witnessed() -> None:
    """Move the bit0 gate calls below the launch: the check must notice the order, not just the presence."""
    source = SOURCE.read_text()
    for site, signature, markers in FUNCTION_SITES:
        body = function(source, signature)
        assert body is not None
        gate = gate_call(site).search(body)
        assert gate is not None
        # Take the gate call out of its place before the launch and append an equivalent call at the end of the
        # function, after the launch, so the call is present once but too late.
        moved = body.replace(gate.group(0), "(void) 0; (void) 0", 1)
        close = moved.rindex("}")
        moved = moved[:close] + gate.group(0) + "{}, 0, nullptr, 0);\n" + moved[close:]
        mutated_body = moved
        mutated = source.replace(body, mutated_body, 1)
        assert mutated != source
        violations = gate_violations(mutated)
        assert any(v.startswith(site + ":") and "precede" in v for v in violations), (
            f"a {site} gate after the launch was not witnessed: {violations}"
        )


def test_an_ignored_decline_is_witnessed() -> None:
    """The gate is called but its answer is dropped, so a decline would still launch the fused kernel."""
    source = SOURCE.read_text()
    for site, signature, _ in FUNCTION_SITES:
        body = function(source, signature)
        assert body is not None
        gate = gate_call(site).search(body)
        assert gate is not None
        tail = body[gate.start():gate.start() + 700]
        mutated_tail = re.sub(r"return (?:false|reject\(\"alias\"\));", "(void) 0;", tail, count=1)
        assert mutated_tail != tail, f"could not find the decline return after the {site} gate"
        mutated = source.replace(body, body.replace(tail, mutated_tail, 1), 1)
        violations = gate_violations(mutated)
        assert any(v.startswith(site + ":") and "return false" in v for v in violations), (
            f"an ignored {site} decline was not witnessed: {violations}"
        )


def chain_call_regex(site: str) -> re.Pattern:
    return re.compile(
        r"ggml_sycl_fusion_alias_admit_chain\(\s*" + re.escape(site) + r"\s*,\s*cgraph\s*,\s*i\s*,\s*sycl_ctx->device\s*\)"
    )


def test_neutralised_chain_gates_are_witnessed() -> None:
    """Valid C++ that keeps the call but throws its answer away, or points it at the wrong chain, must die at
    every chain site: `|| true`, `0 &&`, a negation, a constant conjunct, a discarded result, a wrong node
    index, a wrong site."""
    source = SOURCE.read_text()
    other = {
        "GGML_SYCL_FUSION_SITE_RMS_NORM_MUL_ADD": "GGML_SYCL_FUSION_SITE_MUL_ADD",
        "GGML_SYCL_FUSION_SITE_RMS_NORM_MUL": "GGML_SYCL_FUSION_SITE_MUL_ADD",
        "GGML_SYCL_FUSION_SITE_ADD_RMS_NORM": "GGML_SYCL_FUSION_SITE_MUL_ADD",
        "GGML_SYCL_FUSION_SITE_MUL_ADD": "GGML_SYCL_FUSION_SITE_ADD_RMS_NORM",
    }
    for site, _ in CHAIN_SITES:
        m = chain_call_regex(site).search(source)
        assert m is not None, f"no gate call to mutate for {site}"
        call = m.group(0)
        mutants = {
            "|| true (parenthesised)": f"({call} || true)",
            "|| true (top level)": f"{call} || true",
            "0 &&": f"(0 && {call})",
            "negated": f"!{call}",
            "&& true": f"{call} && true",
            "result discarded": f"(void) {call}, true",
            "wrong node index": call.replace("cgraph, i,", "cgraph, i + 1,", 1),
            "wrong device": call.replace("sycl_ctx->device", "0", 1),
            "wrong site": call.replace(site, other[site], 1),
        }
        for name, replacement in mutants.items():
            assert replacement != call, f"{name} did not change the call"
            mutated = source[:m.start()] + replacement + source[m.end():]
            violations = gate_violations(mutated)
            assert any(v.startswith(site + ":") or v.startswith(other[site] + ":") for v in violations), (
                f"{site}: the `{name}` mutant survived or died for another reason: {violations}"
            )


def test_ternary_swallowing_the_gate_is_witnessed() -> None:
    """`site_on ? true : !skip && ... && GATE` parses as `site_on ? true : (... && GATE)`: when the site is on the
    fused launch is ungated, yet the gate still looks like a bare conjunct of the tail."""
    source = SOURCE.read_text()
    for site, _ in CHAIN_SITES:
        m = chain_call_regex(site).search(source)
        assert m is not None
        cond_open = None
        for open_m in reversed(list(re.finditer(r"\bif\s*\(", source[:m.start()]))):
            if matching_paren(strip_comments(source), open_m.end() - 1) > m.start():
                cond_open = open_m.end()
                break
        assert cond_open is not None
        mutated = source[:cond_open] + "true ? true : " + source[cond_open:]
        violations = gate_violations(mutated)
        assert any(v.startswith(site + ":") and "?:" in v for v in violations), (
            f"{site}: a leading ternary survived or died for another reason: {violations}"
        )


def test_dead_bit0_decline_is_witnessed() -> None:
    """The decline return exists but can never run: `if (!gate(...) && false) { return ...; }`."""
    source = SOURCE.read_text()
    for site, signature, _ in FUNCTION_SITES:
        body = function(source, signature)
        assert body is not None
        pattern = re.compile(r"(if\s*\(\s*!\s*ggml_sycl_fusion_alias_admit\(\s*" + re.escape(site) + r"\b[^;{]*?\))\s*\)\s*\{")
        mutated_body, n = pattern.subn(lambda mm: mm.group(1) + " && false) {", body, count=1)
        assert n == 1, f"could not find the {site} decline to kill"
        violations = gate_violations(source.replace(body, mutated_body, 1))
        assert any(v.startswith(site + ":") for v in violations), f"a dead {site} decline survived: {violations}"


def test_gate_helpers_that_always_admit_are_witnessed() -> None:
    """A helper that counts the check and then answers `true` hides every decline behind a green host test.
    The ways to do it that a reviewer reaches for: replace the final return, return early, return before the
    accounting, change the verdict comparison, hand the predicate no pointer, or leave the pinned text behind
    as dead code or inside a string literal."""
    source = SOURCE.read_text()
    account, admit, chain = (sig for sig, _ in WRAPPER_BODIES)

    def after_open_brace(body: str) -> int:
        return body.index("{", body.index(")")) + 1

    def last_return(body: str, word: str) -> tuple[int, int]:
        at = body.rindex(word)
        return at, at + len(word)

    def final_return_true(body: str) -> str:
        if body.rstrip("}\n ").endswith("return false;"):
            at, end = last_return(body, "return false;")
            return body[:at] + "return true;" + body[end:]
        at, end = last_return(body, "return")
        return body[:at] + "(void) 0; return true; (void)" + body[end:]

    cases = {
        account: {
            "final return true": final_return_true,
            "early return true": lambda b: b[:after_open_brace(b)] + "\n    if (true) { return true; }" + b[after_open_brace(b):],
            "return true before the decline accounting": lambda b: b.replace(
                "    st.declined[site]", "    if (r.verdict != GGML_SYCL_FUSION_ALIAS_SAFE) { return true; }\n    st.declined[site]", 1),
            "verdict compared against INPLACE": lambda b: b.replace(
                "r.verdict == GGML_SYCL_FUSION_ALIAS_SAFE", "r.verdict != GGML_SYCL_FUSION_ALIAS_INPLACE", 1),
            "pinned text in a string literal": lambda b: b[:after_open_brace(b)]
            + '\n    if (true) { return true; }\n    (void) "if(r.verdict==GGML_SYCL_FUSION_ALIAS_SAFE){returntrue;}";' + b[after_open_brace(b):],
        },
        admit: {
            "final return true": final_return_true,
            "early return true": lambda b: b[:after_open_brace(b)] + "\n    if (n_writes >= 0) { return true; }" + b[after_open_brace(b):],
            "dead pinned return": lambda b: b[:after_open_brace(b)] + "\n    return true;" + b[after_open_brace(b):],
        },
        chain: {
            "final return true": final_return_true,
            "early return true": lambda b: b[:after_open_brace(b)] + "\n    if (node_idx >= 0) { return true; }" + b[after_open_brace(b):],
            "resolver hands reads no pointer": lambda b: b.replace(
                "            if (is_write || !norm_chain", "            if (!is_write) { return nullptr; }\n            if (is_write || !norm_chain", 1),
        },
    }
    for signature, mutants in cases.items():
        body = function(source, signature)
        assert body is not None
        for name, mutate in mutants.items():
            mutated_body = mutate(body)
            assert mutated_body != body, f"{signature}: the `{name}` mutant did not change the helper"
            violations = gate_violations(source.replace(body, mutated_body, 1))
            assert any(v.startswith("gate helper:") and signature in v for v in violations), (
                f"{signature}: the `{name}` mutant survived: {violations}"
            )


def test_log_arguments_are_pinned_but_wording_is_free() -> None:
    """The decline log sits inside the account helper, so a statement hidden among its arguments, a macro that
    swallows it, or a `//` inside a string must be caught as a gate-helper change (never an exception), while
    rewording or re-wrapping the message must not be."""
    source = SOURCE.read_text()
    account = "static bool ggml_sycl_fusion_alias_account("
    body = function(source, account)
    assert body is not None
    start = body.index("GGML_LOG_WARN(")
    end = start + len("GGML_LOG_WARN(")
    last_arg = body.index("(unsigned long long) n);")

    def with_body(new_body: str) -> str:
        return source.replace(body, new_body, 1)

    reworded = body.replace("[SYCL-FUSION] declined the", "[SYCL-FUSION] REFUSED the", 1).replace(
        '"(decline %llu)\\n"', '"(decline #%llu)\\n"', 1)
    assert reworded != body
    assert gate_violations(with_body(reworded)) == [], "rewording the log message must stay free"

    mutants = {
        "statement expression returning from the log arguments": body[:end] + '"x", ({ return true; 0; }), ' + body[end:],
        "statement expression appended as the last argument": body[:last_arg] + "(unsigned long long) n, ({ return true; 0; }));" + body[last_arg + len("(unsigned long long) n);"):],
        "side effect among the log arguments": body[:end] + '"x", st.declined_total.store(0), ' + body[end:],
        "// inside a string, then return true": body.replace("    return false;\n}", '    const char * u = "//"; return true;\n    return false;\n}'),
    }
    for name, mutated_body in mutants.items():
        assert mutated_body != body, name
        violations = gate_violations(with_body(mutated_body))
        assert any(v.startswith("gate helper:") and account in v for v in violations), f"`{name}` survived: {violations}"

    for macro in ("#undef GGML_LOG_WARN\n#define GGML_LOG_WARN(...) do { return true; } while (0)\n", "#define true false\n"):
        violations = gate_violations(source.replace(body, macro + body, 1))
        assert any(v.startswith("gate helper:") and "preprocessor" in v for v in violations), f"`{macro}` survived: {violations}"


def test_macro_redefinitions_of_anything_a_helper_uses_are_witnessed() -> None:
    """The forbidden set is every identifier of the pinned helpers, so redefining a called function, a keyword, a
    member or a local is caught, including through backslash-continued directive lines; and the current source
    has none."""
    source = SOURCE.read_text()
    assert redefinition_violations(strip_comments(source)) == []
    names = forbidden_macro_names()
    for needed in ("ggml_sycl_fusion_alias_check", "ggml_sycl_fusion_alias_site_name", "fetch_add", "if", "nullptr", "r", "GGML_LOG_WARN"):
        assert needed in names, needed
    account = "static bool ggml_sycl_fusion_alias_account("
    directives = {
        "the check function": "#define ggml_sycl_fusion_alias_check(...) ggml_sycl_fusion_alias_result{}\n",
        "if": "#define if(x) if (0)\n",
        "a member call": "#define fetch_add(...) load()\n",
        "the site name": '#define ggml_sycl_fusion_alias_site_name(x) "a"\n',
        "a local": "#define r r\n",
        "nullptr": "#define nullptr nullptr\n",
        "#undef": "#undef true\n",
        "spaced directive": "   #\tdefine true false\n",
        "continued define": "#define \\\ntrue false\n",
        "continued hash": "#\\\ndefine true false\n",
        "an enumerator by prefix": "#define GGML_SYCL_FUSION_SITE_COUNT 0\n",
    }
    for name, directive in directives.items():
        violations = gate_violations(source.replace(account, directive + account, 1))
        assert any(v.startswith("gate helper:") and "preprocessor" in v for v in violations), f"`{name}` survived: {violations}"


def test_miswired_bit0_operands_are_witnessed() -> None:
    """The bit0 sites keep their gate call but hand it the wrong tensor, a flipped in-place flag or a short list."""
    source = SOURCE.read_text()

    def mutate(signature: str, old: str, new: str) -> str:
        body = function(source, signature)
        assert body is not None
        # Whitespace-insensitive: the source is clang-formatted and the alignment moves.
        pattern = r"\s*".join(re.escape(t) for t in re.findall(r"\w+|[^\w\s]", old))
        mutated_body, n = re.subn(pattern, lambda _: new, body, count=1)
        assert n == 1, f"could not find `{old}` in {signature}"
        return source.replace(body, mutated_body, 1)

    cases = [
        ("static bool ggml_sycl_try_fuse_tg_mul_mat_add(", "GGML_SYCL_FUSION_SITE_MUL_MAT_ADD",
         "writes  = { add, out_ptr, true }", "writes  = { addend, addend_ptr, true }"),
        ("static bool ggml_sycl_try_fuse_tg_mul_mat_add(", "GGML_SYCL_FUSION_SITE_MUL_MAT_ADD",
         "&writes, 1, &reads, 1)", "&writes, 1, &reads, 0)"),
        ("static bool ggml_sycl_try_fuse_tg_router_f32_add_argsort(", "GGML_SYCL_FUSION_SITE_ROUTER",
         "{ act, act_ptr, false }", "{ act, act_ptr, true }"),
        ("static bool ggml_sycl_try_fuse_tg_router_f32_add_argsort(", "GGML_SYCL_FUSION_SITE_ROUTER",
         "writes, 2, reads, 2)", "writes, 1, reads, 2)"),
        ("static bool ggml_sycl_try_fuse_tg_router_f32_add_argsort(", "GGML_SYCL_FUSION_SITE_ROUTER",
         "{ sort, sort_ptr, true }", "{ sort, sort_ptr, true }, { add, probs_ptr, true }"),
    ]
    for signature, site, old, new in cases:
        violations = gate_violations(mutate(signature, old, new))
        assert any(v.startswith(site + ":") for v in violations), f"`{old}` -> `{new}` survived: {violations}"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
