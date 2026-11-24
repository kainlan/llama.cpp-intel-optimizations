#!/usr/bin/env python3
"""A MUL_MAT_ID refusal must say which clause refused, and its fields must be set.

Census 5 produced 264 decode refusals reading `reason=recipe_missing
source_reason=0`.  Two defects made that pair uninformative:

  1. `source_reason` was never ASSIGNED on the RECIPE_MISSING path, so 0 was the
     struct's default -- and 0 also happens to be a real value,
     expert_resolve_reason::FOUND.  An unset field that ties with a live enum
     value is indistinguishable from a measurement, and it invites exactly the
     wrong reading ("residency succeeded").
  2. `recipe_reason` -- which names the capability clause that declined -- was
     computed, stored on the route, and then dropped before it reached the log.

Both are fixed at the source of the number rather than in the reader, so every
future refusal arrives with reasons instead of archaeology.

The same holds for the Q1_0/NVFP4 direct route in ggml_sycl_mul_mat_id. A call
the direct block declines before its mark falls through to the retained route
core, which runs it, or names its own refusal; neither line says which step of
the direct route declined. Each exit that falls through the block therefore
prints a line naming its step. That line is a decline, not a refusal: it carries
the [MOE-DIRECT-DECLINE] tag and no refusal label, so a count of refusals does
not count an op that ran. The two refusals after the mark (the host job's input
marker and the kernel adapter) end the op with a throw; each prints its step
under the phase's refusal label before the throw (direct_decline_violations
below).
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
BATCH = ROOT / "ggml/src/ggml-sycl/moe-resolved-batch.hpp"


def violations(batch: str, sycl: str) -> list[str]:
    found: list[str] = []

    if not re.search(r"struct moe_resolved_batch_result \{", batch):
        return ["moe_resolved_batch_result is missing"]
    if not re.search(r"const char \*\s+recipe_reason\s*=\s*nullptr\s*;", batch):
        found.append("moe_resolved_batch_result does not carry recipe_reason")

    # Every early return that sets `reject` must also set both diagnostic
    # fields. A path that sets one and not the other is how the sentinel got
    # published in the first place.
    # Column spacing inside these assignments is clang-format alignment and
    # shifts whenever a neighbouring line changes width. Matching it literally
    # makes the checker report "the path is missing" for a path that is present
    # and correct -- a false label on a real finding, which is worse than either.
    for branch, marker in (("validate", r"out\.reject\s+= reject;"),
                           ("recipe", r"out\.reject\s+= moe_batch_reject_reason::RECIPE_MISSING;")):
        m = re.search(marker, batch)
        if m is None:
            found.append(f"the {branch} reject path is missing")
            continue
        window = batch[m.start():m.start() + 420]
        if not re.search(r"out\.source_reason\s*=\s*route\.source_reason;", window):
            found.append(f"the {branch} reject path does not assign source_reason")
        if not re.search(r"out\.recipe_reason\s*=\s*route\.recipe_reason;", window):
            found.append(f"the {branch} reject path does not assign recipe_reason")

    for tag in ("MOE-PROMPT-REFUSAL", "MOE-DECODE-REFUSAL"):
        at = sycl.find(f"[{tag}] tensor=")
        if at < 0:
            found.append(f"the {tag} log line is missing")
            continue
        window = sycl[at:at + 700]
        if "recipe_reason=%s" not in window:
            found.append(f"{tag} does not report recipe_reason")
        if "source_reason=%d" not in window:
            found.append(f"{tag} does not report source_reason")

    return found


DIRECT_DECLINE = "ggml_sycl_moe_direct_log_decline("
DIRECT_REFUSAL = "ggml_sycl_moe_direct_log_refusal("
# The decline's tag and format: a phase field, not a refusal label.
DIRECT_DECLINE_TAG = "[MOE-DIRECT-DECLINE]"
DIRECT_DECLINE_FORMAT = "[MOE-DIRECT-DECLINE] phase=%s tensor=%s route=direct step=%s %s\\n"
DIRECT_REFUSAL_FORMAT = "[%s] tensor=%s route=direct step=%s %s\\n"
DIRECT_GATE = "if ((q1_nvfp4_direct_b70_validated || q1_nvfp4_private_test_authorized) &&"
DIRECT_END = "// MoE hybrid GPU+CPU dispatch gate"
# The success path ends at the return right after the terminal host task is recorded.
DIRECT_SUCCESS = "ggml_sycl::unified_cache_note_host_task(*exact_queue, terminal_task);"
DIRECT_PRECONDITION = "if (card_or_host &&"
# Each step that can decline a call once it has a batch, in the order the route reaches them; "layout" is the gate's
# own remaining conjunct and the host_* steps are a mixed batch's host leg declined before admission. Each falls
# through to the retained route core.
DIRECT_DECLINE_STEPS = ("layout", "precondition", "host_input_marker", "host_task", "host_workspace",
                        "retained_table", "binding", "table_owner_retain", "private_table_owner", "queue_capability",
                        "invocation_snapshot", "submission_authority", "authority_issuer", "workspace_admission",
                        "workspace_slices")
# The steps after the mark, which refuse with a throw that fails the op: "host_marker_arm" (the cache refusing the host
# job's input marker) and "adapter" (the kernel adapter's refusal).
DIRECT_REFUSAL_STEPS = ("host_marker_arm", "adapter")
# The precondition's conjuncts are named bools, computed once: the condition and the decline's failed= list read the
# same bools, and each note's label is the bool's name.
CONJUNCT_DECL = re.compile(r"\bconst\s+bool\s+(\w+)\s*=")
NOTE_CALL = re.compile(r'\bnote\(\s*(\w+)\s*,\s*"(\w+)"\s*\)\s*;')
# A failed submit after the mark throws to the catch that drains the queue. The route's own refusals print their step
# first; the failures the test build injects are not refusals.
SUBMIT_FAILURE = "throw moe_direct_submit_failure{};"
TESTING_MACRO = "GGML_SYCL_Q1_NVFP4_ROUTE_TESTING"


def blank_comments_and_literals(text: str) -> str:
    """The same text, same length, with comments and string/char literals turned to spaces: a brace or parenthesis
    inside them is not code."""
    def blank(match: re.Match) -> str:
        return re.sub(r"[^\n]", " ", match.group(0))

    return re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])\'', blank, text, flags=re.DOTALL)


def matching(code: str, open_at: int) -> int:
    """Index of the bracket that closes the one at open_at."""
    opener = code[open_at]
    closer = {"{": "}", "(": ")"}[opener]
    depth = 0
    for i in range(open_at, len(code)):
        if code[i] == opener:
            depth += 1
        elif code[i] == closer:
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(f"unbalanced {opener} at {open_at}")


def enclosing_if_blocks(code: str, start: int, at: int) -> list[tuple[int, int]]:
    """(open brace, close brace) of each `if` block, outermost first, between start and at whose body holds at."""
    stack: list[int] = []
    for i in range(start, at):
        if code[i] == "{":
            stack.append(i)
        elif code[i] == "}" and stack:
            stack.pop()
    blocks = []
    for brace in stack:
        head = code[:brace].rstrip()
        if not head.endswith(")"):
            continue  # try, else, a struct or a lambda body
        open_paren = len(head) - 1
        depth = 0
        while open_paren >= 0:
            depth += {")": 1, "(": -1}.get(head[open_paren], 0)
            if depth == 0:
                break
            open_paren -= 1
        if re.search(r"\bif\s*$", head[:open_paren]):
            blocks.append((brace, matching(code, brace)))
    return blocks


def else_body(code: str, close_brace: int) -> tuple[int, int] | None:
    """(open, close) of the else (or else-if) body that follows an if block's closing brace, or None."""
    head = re.match(r"\s*else\s*", code[close_brace + 1:])
    if head is None:
        return None
    at = close_brace + 1 + head.end()
    if re.match(r"if\s*\(", code[at:]):
        at = matching(code, code.index("(", at)) + 1
    brace = code.find("{", at)
    if brace < 0 or code[at:brace].strip():
        return None
    return brace, matching(code, brace)


def spaced(text: str) -> str:
    """Whitespace runs as one space and none inside parentheses' edges, so a term survives re-wrapping."""
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"\s+\)", ")", re.sub(r"\(\s+", "(", text)).strip()


def top_level_terms(condition: str) -> list[str]:
    """The && operands of a parenthesized condition, each without its own outer parentheses."""
    inner = condition[1:-1]
    terms, depth, last = [], 0, 0
    for i, ch in enumerate(inner):
        depth += {"(": 1, ")": -1}.get(ch, 0)
        if depth == 0 and inner.startswith("&&", i):
            terms.append(inner[last:i])
            last = i + 2
    terms.append(inner[last:])
    out = []
    for term in (spaced(t) for t in terms):
        while term.startswith("(") and term.endswith(")") and matching(term, 0) == len(term) - 1:
            term = spaced(term[1:-1])
        out.append(term)
    return out


def under_testing_macro(sycl: str, start: int, at: int) -> bool:
    """Whether `at` sits in the test-build branch of an #ifdef of TESTING_MACRO opened after `start`."""
    stack: list[bool] = []
    for m in re.finditer(r"^[ \t]*#[ \t]*(ifdef|ifndef|if|elif|else|endif)\b(.*)$", sycl[start:at], re.M):
        kind = m.group(1)
        if kind in ("ifdef", "ifndef", "if"):
            stack.append(kind == "ifdef" and m.group(2).strip() == TESTING_MACRO)
        elif kind in ("elif", "else") and stack:
            stack[-1] = False
        elif kind == "endif" and stack:
            stack.pop()
    return any(stack)


def enclosing_open_brace(code: str, at: int) -> int:
    """Index of the brace that opens the block holding `at`."""
    depth = 0
    for i in range(at - 1, -1, -1):
        if code[i] == "}":
            depth += 1
        elif code[i] == "{":
            if depth == 0:
                return i
            depth -= 1
    return -1


def precondition_violations(sycl: str, code: str, condition_open: int, condition_close: int,
                            preamble: tuple[int, int], body: tuple[int, int] | None) -> list[str]:
    """The precondition tests named bools only, each declared once in `preamble`, and its decline notes each of them,
    in order, under its own name, and nothing else."""
    found: list[str] = []
    terms = top_level_terms(sycl[condition_open:condition_close + 1])
    declared = CONJUNCT_DECL.findall(code[preamble[0]:preamble[1]])
    for term in terms:
        if not re.fullmatch(r"\w+", term):
            found.append(f"the precondition's conjunct {term} is not a named bool")
        elif declared.count(term) != 1:
            found.append(f"the precondition's conjunct {term} is not declared once as a const bool before it")
    if len(set(terms)) != len(terms):
        found.append("the precondition tests a conjunct twice")
    if body is None:
        return found + ["the precondition has no decline"]
    notes = NOTE_CALL.findall(sycl[body[0]:body[1]])
    if len(re.findall(r"\bnote\s*\(", code[body[0]:body[1]])) != len(notes):
        found.append("the precondition's decline notes something other than a named bool")
    for expression, label in notes:
        if expression != label:
            found.append(f"the precondition's decline names the bool {expression} as {label}")
    noted = [expression for expression, _ in notes]
    if noted != terms:
        found.append(f"the precondition's decline notes {noted}, but the precondition tests {terms}")
    return found


def helper_body(sycl: str, call: str) -> str | None:
    """The body of the static helper whose call is `call`: the definition, not a declaration."""
    helper_at = sycl.find("static void " + call)
    if helper_at < 0:
        return None
    definition_at = sycl.find(") {", helper_at)
    return sycl[definition_at:sycl.find("\n}\n", definition_at)]


def direct_decline_violations(sycl: str) -> list[str]:
    """Every exit that falls through the Q1_0/NVFP4 direct block prints one decline line naming the step that declined,
    and every refusal after the mark prints one refusal line naming its step before it throws."""
    found: list[str] = []
    decline = helper_body(sycl, DIRECT_DECLINE)
    refusal = helper_body(sycl, DIRECT_REFUSAL)
    if decline is None:
        return ["the direct-route decline helper is missing"]
    if refusal is None:
        return ["the direct-route refusal helper is missing"]
    # A decline falls through to an executor that runs the op: its line is not a refusal and carries no refusal label.
    if DIRECT_DECLINE_FORMAT not in decline:
        found.append("the decline line does not print the decline tag, the phase, the tensor and its step")
    if not ('"decode"' in decline and '"prompt"' in decline):
        found.append("the decline line does not name its phase")
    if "REFUSAL" in decline:
        found.append("the decline line carries a refusal label, but the op it names runs on the retained route core")
    if DIRECT_REFUSAL_FORMAT not in refusal:
        found.append("the refusal line does not print its label, the tensor and its step")
    if not ("MOE-DECODE-REFUSAL" in refusal and "MOE-PROMPT-REFUSAL" in refusal):
        found.append("the refusal line does not carry the phase's refusal label")
    if DIRECT_DECLINE_TAG in refusal:
        found.append("the refusal line carries the decline tag")

    gate_at = sycl.find(DIRECT_GATE)
    end_at = sycl.find(DIRECT_END, gate_at)
    if gate_at < 0 or end_at < 0:
        return found + ["the direct block is missing"]
    code = blank_comments_and_literals(sycl)
    success = sycl.find(DIRECT_SUCCESS, gate_at, end_at)
    if success < 0:
        return found + ["the direct block's success path is missing"]
    success_return = code.index("return;", success)

    # Structural: each `if` the success path passes through has an else, and that else prints a decline line. An if
    # without one is an exit that falls through unnamed, whatever its condition is.
    blocks = enclosing_if_blocks(code, gate_at, success_return)
    if len(blocks) < 5:
        found.append(f"the success path passes {len(blocks)} if blocks; the gate, the precondition, the issuer, "
                     "the admission and the slice check are expected")
    for open_brace, close_brace in blocks:
        line = sycl.count("\n", 0, open_brace) + 1
        body = else_body(code, close_brace)
        if body is None:
            found.append(f"an exit falls through without a named line: the if block at line {line} has no else")
        elif DIRECT_DECLINE not in code[body[0]:body[1]]:
            found.append(f"an exit falls through without a named line: the else of the if block at line {line}")
        elif DIRECT_REFUSAL in code[body[0]:body[1]]:
            found.append(f"an exit that falls through prints a refusal line: the else of the if block at line {line}")

    # The step is a literal, so it is read from the source; a match counts only where the comment-blanked code still
    # has the helper there, so a comment that quotes the call names no step. Each helper names only its own steps.
    direct = sycl[gate_at:end_at]
    for helper, steps, other, kind in ((DIRECT_DECLINE, DIRECT_DECLINE_STEPS, DIRECT_REFUSAL_STEPS, "decline"),
                                       (DIRECT_REFUSAL, DIRECT_REFUSAL_STEPS, DIRECT_DECLINE_STEPS, "refusal")):
        calls = [m for m in re.finditer(re.escape(helper) + r'\s*src0,\s*ne12,\s*"(\w+)"', direct)
                 if code.startswith(helper, gate_at + m.start())]
        for step in steps:
            if not any(m.group(1) == step for m in calls):
                found.append(f"no {kind} line names the step {step}")
        for m in calls:
            if m.group(1) in other:
                found.append(f"a {kind} line names the step {m.group(1)}, which is not a {kind}")

    # A refusal that throws after the mark leaves the block by the catch, not by an else: its own block prints the
    # refusal line before the throw, and no decline line, since nothing runs the op after it.
    failures = [m.start() for m in re.finditer(re.escape(SUBMIT_FAILURE), code[gate_at:end_at])]
    if not any(not under_testing_macro(sycl, gate_at, gate_at + at) for at in failures):
        found.append("the direct block has no submit failure outside the test build")
    for at in failures:
        at += gate_at
        if under_testing_macro(sycl, gate_at, at):
            continue
        block = enclosing_open_brace(code, at)
        line = sycl.count(chr(10), 0, at) + 1
        if DIRECT_REFUSAL not in code[block:at]:
            found.append(f"the submit failure at line {line} throws without a named refusal line")
        if DIRECT_DECLINE in code[block:at]:
            found.append(f"the submit failure at line {line} prints a decline line, but the op fails")

    # The precondition's line names every conjunct that failed, from the bools the condition reads.
    precondition_at = code.find(DIRECT_PRECONDITION, gate_at, end_at)
    if precondition_at < 0:
        return found + ["the direct block's precondition is missing"]
    condition_open = code.index("(", precondition_at)
    condition_close = matching(code, condition_open)
    body = else_body(code, matching(code, code.index("{", condition_close)))
    return found + precondition_violations(sycl, code, condition_open, condition_close, (gate_at, precondition_at),
                                           body)


def test_anchors_exist() -> None:
    # Methodology control, first: with the struct or the log lines renamed,
    # every check below is vacuous rather than failing.
    batch = BATCH.read_text()
    sycl = SYCL.read_text()
    assert "struct moe_resolved_batch_result {" in batch
    assert "[MOE-PROMPT-REFUSAL] tensor=" in sycl and "[MOE-DECODE-REFUSAL] tensor=" in sycl


def test_refusals_carry_their_reasons() -> None:
    assert violations(BATCH.read_text(), SYCL.read_text()) == []


def test_mutations_are_witnessed() -> None:
    batch = BATCH.read_text()
    sycl = SYCL.read_text()
    batch_mutations = [
        # Restoring the sentinel: RECIPE_MISSING with source_reason left at 0.
        re.sub(r"(out\.reject        = moe_batch_reject_reason::RECIPE_MISSING;\n"
               r"            out\.occurrence    = i;\n"
               r"            out\.expert_id     = expert_id;\n)"
               r"            out\.source_reason = route\.source_reason;\n",
               r"\1", batch, count=1),
        batch.replace("    const char *            recipe_reason = nullptr;", "", 1),
        batch.replace("            out.recipe_reason = route.recipe_reason;\n", "", 1),
    ]
    for index, mutated in enumerate(batch_mutations):
        assert mutated != batch, f"batch mutation {index} did not change the source"
        assert violations(mutated, sycl), f"batch mutation {index} was not witnessed"

    sycl_mutations = [
        sycl.replace(" recipe_reason=%s\\n\",\n                src0->name ? src0->name : \"?\", "
                     "retained_prompt_batch_result.occurrence", "\\n\",\n                src0->name ? src0->name : \"?\", "
                     "retained_prompt_batch_result.occurrence", 1),
        sycl.replace(" recipe_reason=%s\\n\",\n                src0->name ? src0->name : \"?\", "
                     "retained_decode_batch_result.occurrence", "\\n\",\n                src0->name ? src0->name : \"?\", "
                     "retained_decode_batch_result.occurrence", 1),
    ]
    for index, mutated in enumerate(sycl_mutations):
        assert mutated != sycl, f"log mutation {index} did not change the source"
        assert violations(batch, mutated), f"log mutation {index} was not witnessed"


def test_direct_route_declines_are_named() -> None:
    assert direct_decline_violations(SYCL.read_text()) == []


def test_direct_route_decline_mutations_are_witnessed() -> None:
    sycl = SYCL.read_text()
    mutations = {
        # The admission's exit falls through with no line: its status is never printed.
        "admission-unnamed": re.sub(
            r'\} else \{\n\s*ggml_sycl_moe_direct_log_decline\(src0, ne12, "workspace_admission".*?\);\n\s*\}', "}",
            sycl, count=1, flags=re.DOTALL),
        # A new exit on the success path, an if with no else around the submit.
        "new-exit-unnamed": sycl.replace(
            "completion->bundle.mark_possible_submit()) {",
            "completion->bundle.mark_possible_submit()) {\nif (ne10 > 0) {", 1).replace(
            "                    } else {\n                        // mark_possible_submit() is asked last",
            "                    }\n                    } else {\n"
            "                        // mark_possible_submit() is asked last",
            1),
        # The else is there and prints nothing.
        "else-silent": sycl.replace(
            'ggml_sycl_moe_direct_log_decline(\n                            src0, ne12, "workspace_slices",',
            '(void) (\n                            src0, ne12, "workspace_slices",', 1),
        # One step of the issuer chain reported under another's name.
        "step-renamed": sycl.replace('"binding", "occurrence=%zu', '"retained_table", "occurrence=%zu', 1),
        # A named conjunct the decline line does not note.
        "conjunct-unnamed": sycl.replace(
            "        if (card_or_host && context_id",
            "        const bool src0_rows = ne01 > 0;\n        if (card_or_host && src0_rows && context_id", 1),
        # A conjunct spelled inline, whose text the decline's own line already carries.
        "conjunct-text-already-in-else": sycl.replace(
            "if (card_or_host && context_id", "if (card_or_host && direct_batch.operands.size() && context_id", 1),
        # A conjunct dropped from the condition that the decline still notes.
        "conjunct-removed-still-noted": sycl.replace("ids_device && ids_stride && src1_type",
                                                     "ids_device && src1_type", 1),
        # A note that names its bool as another conjunct.
        "note-mislabelled": sycl.replace('note(ids_device, "ids_device");', 'note(ids_device, "ids_stride");', 1),
        # A note that tests its own expression instead of the bool the condition reads.
        "note-of-an-expression": sycl.replace('note(ids_device, "ids_device");',
                                              'note(ids_device_ptr != nullptr, "ids_device");', 1),
        # The adapter's refusal after the mark throws with no line.
        "adapter-refusal-unnamed": re.sub(
            r'\n\s*ggml_sycl_moe_direct_log_refusal\(\s*src0, ne12, "adapter",.*?\);(?=\n\s*throw '
            r'moe_direct_submit_failure\{\};)', "", sycl, count=1, flags=re.DOTALL),
        # The adapter's line is gone and a comment names the helper in its place: the throw still prints nothing. A
        # real "adapter" line elsewhere keeps the step named, so only the throw's own block can witness this.
        "adapter-refusal-commented": re.sub(
            r'ggml_sycl_moe_direct_log_refusal\(\s*src0, ne12, "adapter",.*?\);(?=\n\s*throw '
            r'moe_direct_submit_failure\{\};)',
            '// ggml_sycl_moe_direct_log_refusal(src0, ne12, "adapter", ...) is printed by the caller', sycl,
            count=1, flags=re.DOTALL).replace(
            DIRECT_SUCCESS, DIRECT_SUCCESS + '\nggml_sycl_moe_direct_log_refusal(src0, ne12, "adapter", "");', 1),
        # The adapter's refusal prints a decline line: the op fails, but a refusal count would not see it.
        "adapter-refusal-as-decline": re.sub(
            r'ggml_sycl_moe_direct_log_refusal\((\s*src0, ne12, "adapter",)', r"ggml_sycl_moe_direct_log_decline(\1",
            sycl, count=1),
        # The host marker's refusal prints a decline line.
        "host-marker-refusal-as-decline": re.sub(
            r'ggml_sycl_moe_direct_log_refusal\((\s*src0, ne12, "host_marker_arm",)',
            r"ggml_sycl_moe_direct_log_decline(\1", sycl, count=1),
        # An exit that falls through prints a refusal line, though the retained route core runs the op.
        "admission-decline-as-refusal": sycl.replace(
            'ggml_sycl_moe_direct_log_decline(src0, ne12, "workspace_admission",',
            'ggml_sycl_moe_direct_log_refusal(src0, ne12, "workspace_admission",', 1),
        # The decline helper prints the phase's refusal label again.
        "decline-tagged-refusal": sycl.replace(
            '"[MOE-DIRECT-DECLINE] phase=%s tensor=%s route=direct step=%s %s\\n"',
            '"[%s] tensor=%s route=direct step=%s %s\\n"', 1).replace(
            'ne12 == 1 ? "decode" : "prompt"', 'ne12 == 1 ? "MOE-DECODE-REFUSAL" : "MOE-PROMPT-REFUSAL"', 1),
        # The refusal helper prints the decline tag.
        "refusal-tagged-decline": sycl.replace(
            '"[%s] tensor=%s route=direct step=%s %s\\n", ne12 == 1 ? "MOE-DECODE-REFUSAL" : "MOE-PROMPT-REFUSAL"',
            '"[MOE-DIRECT-DECLINE] phase=%s tensor=%s route=direct step=%s %s\\n", ne12 == 1 ? "decode" : "prompt"',
            1),
        # An else whose only mention of the helper is a comment prints nothing. A real "workspace_slices" line
        # elsewhere keeps the step named, so only the else check can witness this.
        "else-commented": re.sub(
            r'ggml_sycl_moe_direct_log_decline\(\s*src0, ne12, "workspace_slices",.*?\);',
            '// ggml_sycl_moe_direct_log_decline(src0, ne12, "workspace_slices", ...) was printed by the slicer',
            sycl, count=1, flags=re.DOTALL).replace(
            DIRECT_SUCCESS,
            DIRECT_SUCCESS + '\nggml_sycl_moe_direct_log_decline(src0, ne12, "workspace_slices", "");', 1),
        # The adapter's line names another step; only a comment names "adapter".
        "step-commented": sycl.replace(
            'ggml_sycl_moe_direct_log_refusal(\n                                    src0, ne12, "adapter",',
            '// ggml_sycl_moe_direct_log_refusal(src0, ne12, "adapter", ...)\n'
            '                                ggml_sycl_moe_direct_log_refusal(\n'
            '                                    src0, ne12, "binding",', 1),
        # The gate's layout conjunct falls through unnamed.
        "layout-unnamed": re.sub(r'\} else if \(\(src0->type == GGML_TYPE_Q1_0.*?"layout".*?\);\n    \}', "}",
                                 sycl, count=1, flags=re.DOTALL),
        # The decline line no longer says which step declined.
        "decline-format-without-step": sycl.replace(
            "[MOE-DIRECT-DECLINE] phase=%s tensor=%s route=direct step=%s %s\\n",
            "[MOE-DIRECT-DECLINE] phase=%s tensor=%s route=direct %s %s\\n", 1),
        # The refusal line no longer says which step refused.
        "refusal-format-without-step": sycl.replace(
            '"[%s] tensor=%s route=direct step=%s %s\\n", ne12 == 1',
            '"[%s] tensor=%s route=direct %s %s\\n", ne12 == 1', 1),
    }
    for name, mutated in mutations.items():
        assert mutated != sycl, f"mutation {name} did not change the source"
        assert direct_decline_violations(mutated), f"mutation {name} was not witnessed"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
