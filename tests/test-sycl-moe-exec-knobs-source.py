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
  6. docs/backend/sycl-env-vars.md has a row for each name.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SYCL = ROOT / "ggml" / "src" / "ggml-sycl"
NAMES = ["GGML_SYCL_MOE_EXEC", "GGML_SYCL_MOE_EXEC_TRACE"]


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
    assert count_literal("[MOE-EXEC] summary: %u layer-ubatches ring, %u cpu, %u forced, %zu MiB streamed") == 1


def test_env_settings_rows():
    text = (SYCL / "ggml-sycl.cpp").read_text(errors="replace")
    m = re.search(r"sycl_env_settings\[\]\s*=\s*\{(.*?)\n        \};", text, re.S)
    assert m, "sycl_env_settings[] not found"
    body = m.group(1)
    assert re.search(r"\{\s*ggml_sycl::MOE_EXEC_ENV_POLICY\s*,\s*&g_ggml_sycl_moe_exec_policy", body)
    assert re.search(r"\{\s*ggml_sycl::MOE_EXEC_ENV_TRACE\s*,\s*&g_ggml_sycl_moe_exec_trace", body)
    # The table's integer parse cannot read a policy word, so both destinations are overwritten from the config.
    assert "g_ggml_sycl_moe_exec_policy = static_cast<int>(ggml_sycl_moe_exec_config().policy);" in text
    assert "g_ggml_sycl_moe_exec_trace  = ggml_sycl_moe_exec_config().trace;" in text


def test_config_accessor_and_teardown_summary():
    text = (SYCL / "ggml-sycl.cpp").read_text(errors="replace")
    assert len(re.findall(r"ggml_sycl_moe_exec_config\(\)\s*\{", text)) == 1
    free = re.search(r"static void ggml_backend_sycl_free\(ggml_backend_t backend\) \{(.*?)\n\}\n", text, re.S)
    assert free, "ggml_backend_sycl_free not found"
    assert re.search(r"(?m)^\s*ggml_sycl_moe_exec_log_report\(\);", free.group(1))


@pytest.mark.parametrize("name", NAMES)
def test_catalog_has_row(name):
    assert re.search(r"`%s`" % name, (ROOT / "docs" / "backend" / "sycl-env-vars.md").read_text())


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
