#!/usr/bin/env python3
"""Source gate for the MoE executor's knobs and its once-per-change log (llama.cpp-36j0 Task 4).

Contract (comments are not blanked: every probe is a quoted literal or a code-shaped token, so a comment cannot
satisfy one by accident, and each count is read against a positive control in the host test):

  1. each getenv literal, "GGML_SYCL_MOE_EXEC" and "GGML_SYCL_MOE_EXEC_TRACE", appears exactly once in
     ggml/src/ggml-sycl/ outside tests/ (the catalog's literal-search command is the oracle); the table and the
     parser both name them through constants, never a second literal, so one fact has one source.  tests/ is
     excluded because the host test spells the names as literals to drive its fake getenv, which deliberately pins
     the real names;
  2. the removed ring_layers knob has no literal and no config field;
  3. the per-change WARN format, the level-1 trace format and the teardown summary format each appear exactly once;
  4. sycl_env_settings[] carries a row for both names, so the non-default report names them;
  5. ggml_sycl_moe_exec_config() is defined once in ggml-sycl.cpp and the backend free path prints the summary;
  6. docs/backend/sycl-env-vars.md has a row for each name;
  7. the summary's "MiB streamed" is the windows' bytes (llama.cpp-4huq): note_window() takes the bytes a window staged,
     the layer's note() takes none, and the stream phase feeds note_window() the bytes stage_window() copied;
  8. GGML_SYCL_MOE_EXEC_DECLINE_LAYER (llama.cpp-ny6i) forces one layer off the executor: its accessor reads the one
     spelling, -1 unset, a layer from 0 to 511 otherwise; the backend hands it to the stream phase through the op;
     begin_layer turns that layer's weighed decision to CPU with reason DECLINED before the ring is asked, so
     executor_claims answers false and the layer stays hybrid; the decision log names the reason "declined". Each
     probe has mutants that must fail it.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SYCL = ROOT / "ggml" / "src" / "ggml-sycl"
NAMES = ["GGML_SYCL_MOE_EXEC", "GGML_SYCL_MOE_EXEC_TRACE", "GGML_SYCL_MOE_EXEC_DECLINE_LAYER"]
CATALOGED = NAMES


def sources():
    for p in sorted(SYCL.rglob("*")):
        if p.suffix in (".cpp", ".hpp", ".h") and "tests" not in p.relative_to(SYCL).parts:
            yield p


def strip_comments(src):
    """Drop // and /* */ comments, leaving string and char literals intact (a quoted "//" is not a comment)."""
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c in "\"'":
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            out.append(src[i : j + 1])
            i = j + 1
        elif src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            i = n if j < 0 else j + 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def count_in(src, needle):
    return strip_comments(src).count(needle)


def count_literal(text):
    return sum(count_in(p.read_text(errors="replace"), text) for p in sources())


def test_count_ignores_comments():
    lit = '"GGML_SYCL_X"'
    assert count_in('f(%s); // %s\n/* %s */' % (lit, lit, lit), lit) == 1
    assert count_in('// %s\n/* %s\n%s */' % (lit, lit, lit), lit) == 0
    assert count_in('s = "a // b"; g(%s);' % lit, lit) == 1


@pytest.mark.parametrize("name", NAMES)
def test_getenv_literal_appears_once(name):
    assert count_literal('"%s"' % name) == 1


def test_ring_layers_knob_is_removed():
    assert count_literal("GGML_SYCL_MOE_RING_LAYERS") == 0
    assert (SYCL / "moe-exec-policy.hpp").exists()
    assert "ring_layers" not in (SYCL / "moe-exec-policy.hpp").read_text()


def test_log_formats_appear_once():
    assert count_literal("[MOE-EXEC] layer %d: %s (p=%.2f t_cpu=%.0f us t_ring=%.0f us %s)") == 1
    assert count_literal("[MOE-EXEC-TRACE] dev %d layer %d: %s (p=%.2f t_cpu=%.0f us t_ring=%.0f us %s)") == 1
    assert count_literal("[MOE-EXEC] summary: %u layer-ubatches ring, %u cpu, %u forced, %zu MiB streamed, %u windows, %u exposed, %u misses") == 1


def code_of(name):
    return strip_comments((SYCL / name).read_text(errors="replace"))


def squash(text):
    return re.sub(r"\s+", " ", text)


def test_streamed_bytes_come_from_the_windows():
    policy = squash(code_of("moe-exec-policy.hpp"))
    assert policy.count("void note_window(bool exposed, uint32_t misses, size_t bytes) {") == 1
    assert "bytes_streamed += bytes;" in policy.split("void note_window(")[1].split("void note(")[0]
    note = policy.split("void note(")[1].split("bool take_summary(")[0]
    assert re.match(r"int device, int layer, const moe_exec_decision & d\) \{", note), "note() takes no bytes"
    assert "bytes_streamed" not in note
    plan = squash(code_of("moe-stream-plan.cpp"))
    assert plan.count("log_->note_window(exposed, missing, bytes_staged);") == 1
    wb = plan.split("moe_stream_phase::window_begin(")[1].split("void moe_stream_phase::window_done(")[0]
    assert wb.count("size_t bytes_staged = 0;") == 1
    assert wb.count("bytes_staged +=") == 1 and wb.count("bytes_staged") == 3, "one total, one accumulation, one use"
    lam = wb.split("const auto stage_or_abort = [&]")[1].split("return n; };")[0]
    assert lam.count("bytes_staged += n.bytes;") == 1, "accumulated in the lambda every stage goes through"
    # The call follows the prefetch, so the prefetch's bytes are in the total, and precedes the staged count.
    call = wb.index("log_->note_window(exposed, missing, bytes_staged);")
    assert wb.count("stage_or_abort(nitems, nrole, nphase)") == 1
    assert wb.index("stage_or_abort(nitems, nrole, nphase)") < call < wb.index("uint32_t staged = 0;")
    assert plan.count("log_->note(device_, layer, st.decision);") == 1


def test_env_settings_rows():
    text = (SYCL / "ggml-sycl.cpp").read_text(errors="replace")
    m = re.search(r"sycl_env_settings\[\]\s*=\s*\{(.*?)\n        \};", text, re.S)
    assert m, "sycl_env_settings[] not found"
    body = m.group(1)
    assert re.search(r"\{\s*ggml_sycl::MOE_EXEC_ENV_POLICY\s*,\s*&g_ggml_sycl_moe_exec_policy", body)
    assert re.search(r"\{\s*ggml_sycl::MOE_EXEC_ENV_TRACE\s*,\s*&g_ggml_sycl_moe_exec_trace", body)
    assert re.search(r"\{\s*GGML_SYCL_MOE_EXEC_DECLINE_LAYER_ENV\s*,\s*"
                     r"&g_ggml_sycl_moe_exec_decline_layer\s*,\s*-1\s*\}", body)
    # The table's integer parse would accept a layer the accessor refuses, so the destination is the accessor's answer.
    assert "g_ggml_sycl_moe_exec_decline_layer = ggml_sycl_moe_exec_decline_layer();" in squash(text)
    # The table's integer parse cannot read a policy word, so both destinations are overwritten from the config.
    assert "g_ggml_sycl_moe_exec_policy = static_cast<int>(ggml_sycl_moe_exec_config().policy);" in text
    assert "g_ggml_sycl_moe_exec_trace  = ggml_sycl_moe_exec_config().trace;" in text


def test_config_accessor_and_teardown_summary():
    text = (SYCL / "ggml-sycl.cpp").read_text(errors="replace")
    assert len(re.findall(r"ggml_sycl_moe_exec_config\(\)\s*\{", text)) == 1
    free = re.search(r"static void ggml_backend_sycl_free\(ggml_backend_t backend\) \{(.*?)\n\}\n", text, re.S)
    assert free, "ggml_backend_sycl_free not found"
    assert re.search(r"(?m)^\s*ggml_sycl_moe_exec_log_report\(\);", free.group(1))


@pytest.mark.parametrize("name", CATALOGED)
def test_catalog_has_row(name):
    # A table row of its own, not a mention in another row's prose: `| `NAME` |` or `| `NAME=<value>` |`.
    row = r"(?m)^\| `%s(=[^`]*)?` \|" % re.escape(name)
    assert re.search(row, (ROOT / "docs" / "backend" / "sycl-env-vars.md").read_text())


# ---- the forced-decline knob (llama.cpp-ny6i) ----------------------------------------------------------------

DECLINE_CONST = ('static constexpr const char * GGML_SYCL_MOE_EXEC_DECLINE_LAYER_ENV = '
                 '"GGML_SYCL_MOE_EXEC_DECLINE_LAYER";')
DECLINE_ACCESSOR_SIG = "static inline int ggml_sycl_moe_exec_decline_layer() {"
DECLINE_GETENV = "std::getenv(GGML_SYCL_MOE_EXEC_DECLINE_LAYER_ENV)"
DECLINE_UNSET = "if (env == nullptr || env[0] == '\\0') { return -1; }"
DECLINE_RANGE = "if (end == env || *end != '\\0' || value < 0 || value > GGML_SYCL_MOE_EXEC_DECLINE_LAYER_MAX) {"
DECLINE_MAX = "static constexpr int GGML_SYCL_MOE_EXEC_DECLINE_LAYER_MAX = "
DECLINE_MAX_ASSERT = "static_assert(GGML_SYCL_MOE_EXEC_DECLINE_LAYER_MAX == moe_exec_decision_log::MAX_LAYERS - 1,"
DECLINE_HELPER = ("static inline bool moe_stream_layer_declined(int layer, int knob) { "
                  "return knob >= 0 && knob == layer; }")
DECLINE_FIELD = "int decline_layer = -1;"
DECLINE_FILL = "call.op.decline_layer = ggml_sycl_moe_exec_decline_layer();"
DECLINE_GUARD = "if (moe_stream_layer_declined(layer, op.decline_layer)) {"
DECLINE_CHOICE = "st.decision.choice = moe_exec_choice::CPU;"
DECLINE_REASON = "st.decision.reason = moe_exec_reason::DECLINED;"
WEIGH_CALL = "st.decision = weigh(cur->second, op, 0.0);"
RING_GATE = "if (moe_stream_stage_now(st.decision, op.policy) &&"
REPORT_CALL = "report(layer, st);"
REASON_CASE = 'case moe_exec_reason::DECLINED: return "declined";'
REASON_ENUM_RE = re.compile(r"enum class moe_exec_reason : uint8_t \{([^}]*)\};")


def norm(text):
    return squash(strip_comments(text))


def between(text, start, end):
    """The text after the one `start` and before the first `end` that follows it, or "" when `start` is not there."""
    at = text.find(start)
    if at < 0:
        return ""
    rest = text[at + len(start):]
    stop = rest.find(end)
    return rest if stop < 0 else rest[:stop]


def claim_decline_accessor(common):
    """The accessor reads the knob through the one spelling, answers -1 unset and refuses anything outside
    0..GGML_SYCL_MOE_EXEC_DECLINE_LAYER_MAX, the named bound (claim_decline_bound ties it to the decision log)."""
    n = norm(common)
    acc = between(n, DECLINE_ACCESSOR_SIG, "return decline_layer; }")
    return (n.count(DECLINE_CONST) == 1 and n.count(DECLINE_ACCESSOR_SIG) == 1 and acc.count(DECLINE_GETENV) == 1
            and DECLINE_UNSET in acc and DECLINE_RANGE in acc and acc.count("return -1;") == 2
            and acc.index(DECLINE_UNSET) < acc.index(DECLINE_RANGE))


def claim_decline_bound(common, plan_cpp):
    """The bound is defined once, in common.hpp, and moe-stream-plan.cpp (which sees both) static_asserts it equals the
    decision log's last layer, MAX_LAYERS - 1: a larger knob would name a layer the log cannot record."""
    return norm(common).count(DECLINE_MAX) == 1 and norm(plan_cpp).count(DECLINE_MAX_ASSERT) == 1


def claim_decline_helper(plan_hpp):
    """The helper is true for the knob's layer only and never with the knob unset; the op carries the knob, unset by
    default, so a phase no backend op filled declines nothing."""
    n = norm(plan_hpp)
    op = between(n, "struct moe_stream_op {", "};")
    return n.count(DECLINE_HELPER) == 1 and op.count(DECLINE_FIELD) == 1


def claim_decline_fill(sycl):
    """The backend fills the op's knob in ggml_sycl_moe_stream_prepare, and nowhere else."""
    n = norm(sycl)
    prep = between(n, "static bool ggml_sycl_moe_stream_prepare(", "return true; }")
    return n.count(DECLINE_FILL) == 1 and prep.count(DECLINE_FILL) == 1


def override_block(text):
    """The declined-layer block of begin_layer as the claims read it: the guard through its closing brace, "" when the
    guard is not there."""
    n = norm(text)
    body = between(n, DECLINE_GUARD, "}")
    return DECLINE_GUARD + body + "}" if DECLINE_GUARD in n else ""


def claim_decline_override(plan_cpp):
    """begin_layer turns the declined layer's weighed decision to CPU, DECLINED: after weigh() and before the ring is
    asked (moe_stream_stage_now then answers false, so the ring is neither reserved nor claimed for it), and before the
    decision is reported. The block under the knob's guard sets both fields, in either order."""
    n = norm(plan_cpp)
    bl = between(n, "moe_exec_decision moe_stream_phase::begin_layer(", "moe_stream_phase::find(")
    body = between(bl, DECLINE_GUARD, "}")
    marks = [bl.find(m) for m in (WEIGH_CALL, DECLINE_GUARD, RING_GATE, REPORT_CALL)]
    return (n.count(DECLINE_GUARD) == 1 and body.count(DECLINE_CHOICE) == 1 and body.count(DECLINE_REASON) == 1
            and min(marks) >= 0 and marks == sorted(marks))


def claim_decline_reason(policy_hpp, cost_hpp):
    """DECLINED is a reason of its own, appended so the earlier values keep their numbers, and the log names it."""
    m = REASON_ENUM_RE.search(norm(cost_hpp))
    names = [x.strip() for x in m.group(1).split(",")] if m else []
    return (names[-1:] == ["DECLINED"] and names.count("DECLINED") == 1
            and norm(policy_hpp).count(REASON_CASE) == 1)


def mutate(text, old, new):
    """`text` normalised as the claims read it (comments dropped, whitespace squashed), with the one `old` replaced: the
    mutants are written against that form, so alignment padding cannot hide a probe."""
    n = norm(text)
    assert n.count(old) == 1, "mutant precondition: %r occurs once" % old
    return n.replace(old, new)


def src(name):
    return (SYCL / name).read_text(errors="replace")


def test_decline_accessor():
    assert claim_decline_accessor(src("common.hpp"))


def test_decline_accessor_mutants():
    common = src("common.hpp")
    # a second spelling of the name
    assert not claim_decline_accessor(mutate(common, DECLINE_GETENV, 'std::getenv("GGML_SYCL_MOE_EXEC_DECLINE_LAYER")'))
    # the range check widened past the decision log's layers, back to a literal, or the negative check dropped
    assert not claim_decline_accessor(mutate(common, "value > GGML_SYCL_MOE_EXEC_DECLINE_LAYER_MAX", "value > 1023"))
    assert not claim_decline_accessor(mutate(common, "value > GGML_SYCL_MOE_EXEC_DECLINE_LAYER_MAX", "value > 511"))
    assert not claim_decline_accessor(mutate(common, "value < 0 || ", ""))
    # unset answers layer 0 instead of -1
    assert not claim_decline_accessor(mutate(common, DECLINE_UNSET, DECLINE_UNSET.replace("return -1;", "return 0;")))


def test_decline_bound():
    assert claim_decline_bound(src("common.hpp"), src("moe-stream-plan.cpp"))


def test_decline_bound_mutants():
    common, cpp = src("common.hpp"), src("moe-stream-plan.cpp")
    # no named bound, or no assert tying it to the decision log, or an assert against the log's size
    assert not claim_decline_bound(mutate(common, DECLINE_MAX, "static constexpr int DECLINE_LAYER_LAST = "), cpp)
    assert not claim_decline_bound(common, mutate(cpp, DECLINE_MAX_ASSERT, "static_assert(true,"))
    assert not claim_decline_bound(common, mutate(cpp, DECLINE_MAX_ASSERT, DECLINE_MAX_ASSERT.replace(" - 1,", ",")))


def test_decline_helper():
    assert claim_decline_helper(src("moe-stream-plan.hpp"))


def test_decline_helper_mutants():
    hpp = src("moe-stream-plan.hpp")
    assert not claim_decline_helper(mutate(hpp, "return knob >= 0 && knob == layer;", "return knob == layer;"))
    assert not claim_decline_helper(mutate(hpp, "return knob >= 0 && knob == layer;", "return knob >= 0;"))
    # the op's default declines layer 0
    assert not claim_decline_helper(mutate(hpp, DECLINE_FIELD, "int decline_layer = 0;"))


def test_decline_fill():
    assert claim_decline_fill(src("ggml-sycl.cpp"))


def test_decline_fill_mutants():
    sycl = src("ggml-sycl.cpp")
    assert not claim_decline_fill(mutate(sycl, DECLINE_FILL, ""))
    # filled outside prepare
    assert not claim_decline_fill(mutate(sycl, DECLINE_FILL, "") + " void f() { " + DECLINE_FILL + " }")


def test_decline_override():
    assert claim_decline_override(src("moe-stream-plan.cpp"))


def test_decline_override_either_order():
    cpp = src("moe-stream-plan.cpp")
    block = override_block(cpp)
    swapped = DECLINE_GUARD + " " + DECLINE_REASON + " " + DECLINE_CHOICE + " }"
    assert block != swapped and claim_decline_override(mutate(cpp, block, swapped))


def test_decline_override_mutants():
    cpp = src("moe-stream-plan.cpp")
    block = override_block(cpp)
    # no override
    assert not claim_decline_override(mutate(cpp, block, ""))
    # the override after the ring gate, where a RING decision has already reserved and claimed the ring
    moved = mutate(cpp, block, "").replace(REPORT_CALL, block + " " + REPORT_CALL, 1)
    assert moved.count(DECLINE_GUARD) == 1 and not claim_decline_override(moved)
    # the wrong reason, or the choice left alone
    assert not claim_decline_override(mutate(cpp, block, block.replace("::DECLINED", "::FORCED")))
    ring = block.replace("moe_exec_choice::CPU", "moe_exec_choice::RING")
    assert not claim_decline_override(mutate(cpp, block, ring))
    assert not claim_decline_override(mutate(cpp, block, block.replace(DECLINE_CHOICE, "")))
    # an assignment outside the guarded block, which would apply to every layer
    assert not claim_decline_override(mutate(cpp, block, block.replace(DECLINE_REASON, "") + " " + DECLINE_REASON))
    # a fixed layer instead of the knob
    assert not claim_decline_override(mutate(cpp, block, block.replace("op.decline_layer", "3")))


def test_decline_reason():
    assert claim_decline_reason(src("moe-exec-policy.hpp"), src("moe-exec-cost.hpp"))


def test_decline_reason_mutants():
    policy, cost = src("moe-exec-policy.hpp"), src("moe-exec-cost.hpp")
    forced = 'case moe_exec_reason::DECLINED: return "forced";'
    assert not claim_decline_reason(mutate(policy, REASON_CASE, forced), cost)
    assert not claim_decline_reason(mutate(policy, REASON_CASE, ""), cost)
    assert not claim_decline_reason(policy, mutate(cost, ", DECLINED }", " }"))
    # inserted before an existing value, which renumbers it
    assert not claim_decline_reason(policy, mutate(cost, "NONE, FORCED,", "NONE, DECLINED, FORCED,")
                                    .replace(", NO_HOST_EXPERTS, DECLINED }", ", NO_HOST_EXPERTS }"))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
