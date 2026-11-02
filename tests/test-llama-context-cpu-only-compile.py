"""CPU-only compile gate for src/llama-context.cpp (llama.cpp-txho).

A build with neither GGML_USE_SYCL nor GGML_BACKEND_DL must still compile llama-context.cpp: every helper the shared
(unguarded) code calls needs a definition in that configuration too. 78c358ee7 called llama_context_has_sycl_backend()
from the unguarded measure-scope requirement while the helper existed only under the SYCL / backend-DL guard, so the
CPU-only build broke at that line.

- the compile: `g++ -fsyntax-only` of the file with neither macro defined must report no error (the real check; it
  needs the source tree's own headers only, no build directory);
- the pin: the helper has a definition in the `#else` arm of the guard that defines the SYCL one, and that definition
  returns false (a CPU-only build has no SYCL backend), so the callers stay unguarded;
- the mutant: dropping that definition makes the compile fail (the compile gate would otherwise pass vacuously on a
  future change that guards the caller instead and leaves the helper SYCL-only; the pin keeps the shared form).

Host-only; collected by pytest. Skips when no g++ is installed.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CTX_CPP = ROOT / "src/llama-context.cpp"
INCLUDES = ["-Isrc", "-Iinclude", "-Iggml/include", "-Icommon", "-Iggml/src"]


def _compile(text, defines=()):
    # The source goes in on stdin: a mutant never touches the tree, so a concurrent build cannot see it.
    return subprocess.run(
        ["g++", "-std=c++17", "-fsyntax-only", *defines, *INCLUDES, "-x", "c++", "-"],
        cwd=ROOT, input=text, capture_output=True, text=True, timeout=300,
    )


def _errors(res):
    return [l for l in res.stderr.splitlines() if " error" in l or l.startswith("error")]


needs_gxx = pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not installed")


@needs_gxx
def test_cpu_only_build_compiles():
    res = _compile(CTX_CPP.read_text())
    assert not _errors(res), "llama-context.cpp must compile with neither GGML_USE_SYCL nor GGML_BACKEND_DL:\n" + \
        "\n".join(_errors(res)[:5])


@needs_gxx
@pytest.mark.parametrize("define", ["-DGGML_USE_SYCL", "-DGGML_BACKEND_DL"])
def test_sycl_builds_still_compile(define):
    res = _compile(CTX_CPP.read_text(), [define])
    assert not _errors(res), "\n".join(_errors(res)[:5])


def test_helper_has_a_non_sycl_definition_returning_false():
    src = CTX_CPP.read_text()
    m = re.search(
        r"#else\s*(?://[^\n]*\n\s*)*static bool llama_context_has_sycl_backend\(const std::vector<ggml_backend_ptr> &\s*\)"
        r"\s*\{\s*return false;\s*\}\s*#endif",
        src,
    )
    assert m, "llama_context_has_sycl_backend needs a `return false` definition in the #else arm of the SYCL/DL guard"
    head = src[: m.start()]
    guard = head.rindex("#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)")
    assert "static bool llama_context_has_sycl_backend(const std::vector<ggml_backend_ptr> & backends)" in head[guard:], \
        "the #else must pair with the guard that defines the SYCL helper"


@needs_gxx
def test_mutant_without_the_definition_fails_to_compile():
    src = CTX_CPP.read_text()
    mutated = re.sub(
        r"(#else\s*(?://[^\n]*\n\s*)*)static bool llama_context_has_sycl_backend\(const std::vector<ggml_backend_ptr> &\s*\)"
        r"\s*\{\s*return false;\s*\}",
        r"\1", src, count=1,
    )
    assert mutated != src
    res = _compile(mutated)
    assert any("llama_context_has_sycl_backend" in l for l in _errors(res)), \
        "mutant 'no non-SYCL definition' slipped through the compile gate"
