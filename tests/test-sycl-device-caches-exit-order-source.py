"""Source contract: the per-device unified caches are destroyed before dpct's device manager at exit.

Static objects are destroyed in the reverse order of their construction. dpct::dev_mgr is a function-local static of
dpct::dev_mgr::instance() (dpct/helper.hpp), constructed at its first use. A namespace-scope map of the caches is
constructed when the library loads, before that, so at exit it is destroyed after the manager, and ~unified_cache's
teardown (shutdown_resources -> arena_abandon -> ggml_sycl_get_device_id_from_queue) then asks the destroyed manager
which device its queue is on. It segfaulted once on a null device_ext and passes otherwise by what the freed memory
holds.

The order is held by construct-on-first-use: unified_cache_device_caches() in unified-cache.cpp touches
dpct::dev_mgr::instance() first and then holds the map as its own function-local static, so the map is constructed after
the manager and destroyed before it. This gate pins that:

  - the map is held by no namespace-scope object of any ggml-sycl source: its one static declaration is inside the
    accessor, and the old namespace-scope name is gone, so no use can bypass the accessor;
  - the accessor touches the device manager before the map's static is reached;
  - create_cache_for_device() takes the map before it constructs a cache. A cache's constructor registers the exit
    handler that marks SYCL as shutting down, and that handler must run before the map is destroyed, so the map must
    be constructed before the first cache. In `map[id] = std::make_unique<unified_cache>(...)` the right operand is
    evaluated first, so the accessor call must be a statement of its own, before that one.

It reads comment-stripped text, and each check is proven to catch its mutant below.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SYCL_DIR = ROOT / "ggml/src/ggml-sycl"
CACHE_CPP = SYCL_DIR / "unified-cache.cpp"

MAP_TYPE = r"std::unordered_map\s*<\s*int\s*,\s*std::shared_ptr\s*<\s*unified_cache\s*>\s*>"
ACCESSOR = "unified_cache_device_caches"
ACCESSOR_SIG = r"\bstatic\s+" + MAP_TYPE + r"\s*&\s*" + ACCESSOR + r"\s*\(\s*\)\s*\{"
STATIC_MAP_DECL = r"\bstatic\s+" + MAP_TYPE + r"\s+(\w+)\s*;"
TOUCH = r"\bdpct::dev_mgr::instance\s*\(\s*\)"
CREATE_SIG = r"\bstatic\s+unified_cache\s*\*\s*create_cache_for_device\s*\("
OLD_NAME = re.compile(r"\bg_device_caches\b")

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


def body_span(code: str, signature: str):
    """(start, end) of the body after the first match of `signature`, which ends at its opening brace."""
    m = re.search(signature, code)
    if not m:
        return None
    depth = 1
    i = m.end()
    while i < len(code) and depth:
        if code[i] == "{":
            depth += 1
        elif code[i] == "}":
            depth -= 1
        i += 1
    return m.end(), i - 1


def order_failures(cache_code: str, other_sources: dict) -> list:
    """Every way the code breaks the exit order; empty when it holds."""
    failures = []
    for rel, code in sorted(other_sources.items()):
        if OLD_NAME.search(code):
            failures.append(f"{rel} still names g_device_caches")
    if OLD_NAME.search(cache_code):
        failures.append("unified-cache.cpp still names g_device_caches")

    span = body_span(cache_code, ACCESSOR_SIG)
    if span is None:
        return failures + [f"no accessor `static <map> & {ACCESSOR}()` in unified-cache.cpp"]
    start, end = span

    decls = list(re.finditer(STATIC_MAP_DECL, cache_code))
    outside = [d for d in decls if not (start <= d.start() < end)]
    for d in outside:
        line = cache_code.count("\n", 0, d.start()) + 1
        failures.append(f"unified-cache.cpp:{line} holds the caches in a static `{d.group(1)}` outside the accessor")
    for rel, code in sorted(other_sources.items()):
        for d in re.finditer(STATIC_MAP_DECL, code):
            line = code.count("\n", 0, d.start()) + 1
            failures.append(f"{rel}:{line} holds a static map of caches `{d.group(1)}`")

    body = cache_code[start:end]
    inside = [d for d in re.finditer(STATIC_MAP_DECL, body)]
    if len(inside) != 1:
        failures.append(f"the accessor holds {len(inside)} static maps of caches, not one")
        return failures
    name = inside[0].group(1)
    touch = re.search(TOUCH, body)
    if not touch:
        failures.append("the accessor does not touch dpct::dev_mgr::instance()")
    elif touch.start() > inside[0].start():
        failures.append("the accessor touches the device manager after the map's static, not before it")
    if not re.search(r"\breturn\s+" + re.escape(name) + r"\s*;", body):
        failures.append(f"the accessor does not return its static map `{name}`")

    create = body_span(cache_code, CREATE_SIG)
    if create is None:
        failures.append("create_cache_for_device() not found")
    else:
        cbody = cache_code[create[0]:create[1]]
        construct = re.search(r"std::make_unique\s*<\s*unified_cache\s*>", cbody)
        if not construct:
            failures.append("create_cache_for_device() no longer constructs a unified_cache with make_unique")
        else:
            statement_start = cbody.rfind(";", 0, construct.start())
            statement_start = max(statement_start, cbody.rfind("{", 0, construct.start()))
            if not re.search(r"\b" + ACCESSOR + r"\s*\(\s*\)", cbody[:statement_start]):
                failures.append("create_cache_for_device() constructs a cache before it has taken the map")
    return failures


def sources():
    cache_code = strip_comments(CACHE_CPP.read_text(errors="replace"))
    others = {}
    for path in sorted(SYCL_DIR.rglob("*")):
        if path.suffix in {".cpp", ".hpp", ".h"} and path.is_file() and path != CACHE_CPP:
            others[path.relative_to(ROOT).as_posix()] = strip_comments(path.read_text(errors="replace"))
    return cache_code, others


def test_the_caches_are_constructed_after_the_device_manager():
    cache_code, others = sources()
    failures = order_failures(cache_code, others)
    assert not failures, "\n".join(failures)


def _accessor_body(code: str) -> str:
    start, end = body_span(code, ACCESSOR_SIG)
    return code[start:end]


def _mutate_accessor(code: str, edit) -> str:
    start, end = body_span(code, ACCESSOR_SIG)
    return code[:start] + edit(code[start:end]) + code[end:]


def _drop_touch(body: str) -> str:
    return re.sub(r"[^;{}]*" + TOUCH + r"[^;]*;", "", body)


def _touch_after_static(body: str) -> str:
    touch = re.search(r"[^;{}]*" + TOUCH + r"[^;]*;", body).group(0)
    body = body.replace(touch, "", 1)
    decl = re.search(STATIC_MAP_DECL, body)
    return body[: decl.end()] + touch + body[decl.end() :]


def _namespace_scope_map(code: str) -> str:
    name = re.search(STATIC_MAP_DECL, _accessor_body(code)).group(1)
    decl = "static std::unordered_map<int, std::shared_ptr<unified_cache>> g_device_caches;\n"
    code = _mutate_accessor(code, lambda body: re.sub(STATIC_MAP_DECL, "", body, count=1))
    code = re.sub(r"\breturn\s+" + name + r"\s*;", "return g_device_caches;", code, count=1)
    m = re.search(ACCESSOR_SIG, code)
    return code[: m.start()] + decl + code[m.start() :]


def _bypass(code: str) -> str:
    # a second, namespace-scope holder of caches that a later use reads directly
    return code + "\nstatic std::unordered_map<int, std::shared_ptr<unified_cache>> g_spare_caches;\n"


def _construct_in_the_same_statement(code: str) -> str:
    start, end = body_span(code, CREATE_SIG)
    body = code[start:end]
    body = re.sub(r"[^;{}]*\b" + ACCESSOR + r"\s*\(\s*\)[^;]*;", "", body)
    body = re.sub(r"(\w+)(\s*\[\s*\w+\s*\]\s*=\s*std::make_unique\s*<\s*unified_cache\s*>)",
                  ACCESSOR + r"()\2", body, count=1)
    return code[:start] + body + code[end:]


MUTANTS = {
    "the accessor does not touch the device manager": lambda c: _mutate_accessor(c, _drop_touch),
    "the accessor touches the device manager after the map's static": lambda c: _mutate_accessor(c, _touch_after_static),
    "the map is a namespace-scope static again": _namespace_scope_map,
    "a second static map of caches bypasses the accessor": _bypass,
    "the first cache is constructed before the map": _construct_in_the_same_statement,
}


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_each_check_catches_its_mutant(name):
    cache_code, others = sources()
    mutant = MUTANTS[name](cache_code)
    assert mutant != cache_code, f"the mutant '{name}' did not change the source"
    assert order_failures(mutant, others), f"the mutant '{name}' passed the gate"


def test_a_namespace_scope_name_in_another_source_is_caught():
    cache_code, others = sources()
    rel = sorted(others)[0]
    others = dict(others)
    others[rel] = others[rel] + "\nvoid f() { g_device_caches.clear(); }\n"
    assert order_failures(cache_code, others)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
