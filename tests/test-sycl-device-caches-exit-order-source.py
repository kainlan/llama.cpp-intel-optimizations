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

  - the map is held by no namespace-scope object of any ggml-sycl source, static or not, named or in an anonymous
    namespace: its one static declaration is inside the accessor, and the old namespace-scope name is gone, so no use
    can bypass the accessor;
  - the accessor touches the device manager before the map's static is reached;
  - the exit handler that marks SYCL as shutting down is registered after the map is constructed, so it runs before
    the map is destroyed and the caches the map destroys take the abandon branch of their teardown. A handler
    registered before an object's construction completes runs after that object is destroyed. The handler is
    registered by the first cache's constructor, and a cache may be constructed directly, by anyone, before any map
    lookup, so the constructor calls the accessor before any std::atexit in its body. create_cache_for_device() also
    takes the map before it constructs a cache: in `map[id] = std::make_unique<unified_cache>(...)` the right operand
    is evaluated first, so the accessor call there must be a statement of its own, before that one.

It reads comment-stripped text, and each check is proven to catch its mutant below. It cannot see code that is compiled
out (`#if 0`) or never runs (`if (false)`); the debugger run of a directly constructed cache is the behavioural gate.
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
# body_span() needs a signature that ends at the body's opening brace; the parameter list holds no brace or `;`
CREATE_SIG = r"\bstatic\s+unified_cache\s*\*\s*create_cache_for_device\s*\([^{;]*\{"
# the constructor's definition: its parameter and member-initializer lists hold no brace, so the first opens the body
CTOR_SIG = r"\bunified_cache::unified_cache\s*\([^{;]*\{"
ACCESSOR_CALL = r"\b" + ACCESSOR + r"\s*\(\s*\)"
ATEXIT = r"\bstd::atexit\s*\("
# any variable of the map's type, whatever its storage class: the scope is decided separately
MAP_VAR_DECL = MAP_TYPE + r"\s+(\w+)\s*(?=[;={])"
OLD_NAME = re.compile(r"\bg_device_caches\b")

# literals whose text could hold a brace, blanked for the scope walk; comments are already gone
_LITERAL_RE = re.compile(
    r'R"([^(\s]{0,16})\(.*?\)\1"'  # raw string literal
    r'|"(?:\\.|[^"\\\n])*"'  # string literal
    r"|'(?:\\.|[^'\\\n])*'",  # char literal
    flags=re.DOTALL,
)
_NAMESPACE_OPENER = re.compile(r"(?:\bnamespace(?:\s+[\w:]+)?|\bextern)\s*$")

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


def namespace_scope_map_decls(code: str) -> list:
    """Names and lines of the map-typed variables declared at namespace scope: every enclosing brace opens a namespace
    or an extern block. A file whose braces do not balance (a preprocessor branch opens one twice) cannot be walked, so
    every map-typed variable in it is reported, which fails closed."""
    blank = _LITERAL_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), code)
    decls = list(re.finditer(MAP_VAR_DECL, blank))
    if not decls:
        return []
    kinds = []  # True: the brace at this depth opens a namespace or an extern block
    scope_at = {}
    pending = sorted(d.start() for d in decls)
    balanced = True
    p = 0
    for i, ch in enumerate(blank):
        while p < len(pending) and pending[p] == i:
            scope_at[i] = all(kinds)
            p += 1
        if ch == "{":
            kinds.append(bool(_NAMESPACE_OPENER.search(blank[max(0, i - 200):i])))
        elif ch == "}":
            if not kinds:
                balanced = False
                break
            kinds.pop()
    balanced = balanced and not kinds
    found = []
    for d in decls:
        if not balanced or scope_at.get(d.start(), True):
            found.append((d.group(1), code.count("\n", 0, d.start()) + 1))
    return found


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
    for rel, code in [("unified-cache.cpp", cache_code)] + sorted(other_sources.items()):
        for name, line in namespace_scope_map_decls(code):
            failures.append(f"{rel}:{line} holds the caches in a namespace-scope map `{name}`")

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

    ctor = body_span(cache_code, CTOR_SIG)
    if ctor is None:
        failures.append("the unified_cache constructor's definition not found")
    else:
        kbody = cache_code[ctor[0]:ctor[1]]
        call = re.search(ACCESSOR_CALL, kbody)
        atexit = re.search(ATEXIT, kbody)
        if atexit is None:
            failures.append("the unified_cache constructor no longer registers the exit handler with std::atexit")
        if call is None:
            failures.append("the unified_cache constructor does not take the map before it registers the exit handler")
        elif atexit is not None and call.start() > atexit.start():
            failures.append("the unified_cache constructor takes the map after it registers the exit handler")
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


def _external_linkage_map(code: str) -> str:
    # a namespace-scope holder without `static`: external linkage, the same construction at load
    return code + "\nstd::unordered_map<int, std::shared_ptr<unified_cache>> g_spare_caches;\n"


def _anonymous_namespace_map(code: str) -> str:
    return code + "\nnamespace {\nstd::unordered_map<int, std::shared_ptr<unified_cache>> g_spare_caches{};\n}\n"


_CTOR_TAKE = r"[^;{}]*" + ACCESSOR_CALL + r"[^;]*;"


def _mutate_ctor(code: str, edit) -> str:
    start, end = body_span(code, CTOR_SIG)
    return code[:start] + edit(code[start:end]) + code[end:]


def _ctor_without_take(body: str) -> str:
    return re.sub(_CTOR_TAKE, "", body, count=1)


def _ctor_take_after_atexit(body: str) -> str:
    take = re.search(_CTOR_TAKE, body).group(0)
    body = body.replace(take, "", 1)
    first = re.search(ATEXIT + r"[^;]*;", body)
    return body[: first.end()] + take + body[first.end() :]


MUTANTS = {
    "the accessor does not touch the device manager": lambda c: _mutate_accessor(c, _drop_touch),
    "the accessor touches the device manager after the map's static": lambda c: _mutate_accessor(c, _touch_after_static),
    "the map is a namespace-scope static again": _namespace_scope_map,
    "a second static map of caches bypasses the accessor": _bypass,
    "a namespace-scope map without static bypasses the accessor": _external_linkage_map,
    "a map in an anonymous namespace bypasses the accessor": _anonymous_namespace_map,
    "the first cache is constructed before the map": _construct_in_the_same_statement,
    "the cache constructor does not take the map": lambda c: _mutate_ctor(c, _ctor_without_take),
    "the cache constructor takes the map after std::atexit": lambda c: _mutate_ctor(c, _ctor_take_after_atexit),
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


def test_a_map_local_to_a_function_is_not_a_holder():
    # the scope walk's control: shutdown_unified_cache() copies the map into a local of the same type, and a function
    # or class body is not a namespace scope, braces in literals included
    cache_code, others = sources()
    local = '\nvoid f() { const char * s = "}"; std::unordered_map<int, std::shared_ptr<unified_cache>> local; }\n'
    assert namespace_scope_map_decls(local) == []
    assert order_failures(cache_code + local, others) == order_failures(cache_code, others)


# The cache lock is a std::shared_mutex. A thread that holds it, in either mode, must not reach a lookup that takes it
# again: twice shared on one thread is undefined and deadlocks once a writer queues between the two, and shared under
# unique deadlocks outright. These are the lookups that take it shared for the map read.
CACHE_LOCK = re.compile(
    r"std::(?:shared_lock|unique_lock|lock_guard|scoped_lock)\s*<\s*std::shared_mutex\s*>\s*\w+\s*[({]\s*"
    r"g_cache_rw_mutex\s*[)}]"
)
RELOCKING_LOOKUPS = (
    "get_cache_shared",
    "get_existing_cache_for_device",
    "get_existing_unified_cache_for_device",
    "multi_device_plan_arena_cache",
    "multi_device_plan_subtracts_arena",
)
PUBLIC_LOOKUP_SIGS = {
    "get_existing_unified_cache_for_device": r"(?<!static )\bunified_cache\s*\*\s*get_existing_unified_cache_for_device"
    r"\s*\(\s*int\s+\w+\s*\)\s*\{",
    "multi_device_plan_subtracts_arena": r"\bbool\s+multi_device_plan_subtracts_arena\s*\(\s*int\s+\w+\s*\)\s*\{",
}


def nested_lock_failures(cache_code: str) -> list:
    """Every lookup of RELOCKING_LOOKUPS called while a scope holds g_cache_rw_mutex, and every public lookup that
    names the lock at all. A lock's scope runs from its declaration to the close of the block that holds it."""
    failures = []
    blank = _LITERAL_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), cache_code)
    calls = re.compile(r"\b(" + "|".join(RELOCKING_LOOKUPS) + r")\s*\(")
    for lock in CACHE_LOCK.finditer(blank):
        depth, j = 0, lock.end()
        while j < len(blank):
            if blank[j] == "{":
                depth += 1
            elif blank[j] == "}":
                if depth == 0:
                    break
                depth -= 1
            j += 1
        line = blank.count("\n", 0, lock.start()) + 1
        for call in calls.finditer(blank, lock.end(), j):
            failures.append(f"unified-cache.cpp:{line} holds g_cache_rw_mutex and calls {call.group(1)}(), which "
                            "takes it again")
    for name, sig in PUBLIC_LOOKUP_SIGS.items():
        span = body_span(blank, sig)
        if span is None:
            failures.append(f"{name}() definition not found")
        elif "g_cache_rw_mutex" in blank[span[0]:span[1]]:
            failures.append(f"{name}() names g_cache_rw_mutex; the lookup inside it takes the lock")
    return failures


def test_no_cache_lookup_runs_under_the_cache_lock():
    cache_code, _ = sources()
    failures = nested_lock_failures(cache_code)
    assert not failures, "\n".join(failures)


def _relock(name: str):
    def edit(code: str) -> str:
        start, end = body_span(code, PUBLIC_LOOKUP_SIGS[name])
        return code[:start] + "\n    std::shared_lock<std::shared_mutex> lock(g_cache_rw_mutex);" + code[start:]

    return edit


def _lookup_under_a_held_lock(code: str) -> str:
    # a function that already holds the lock, as unified_cache_has_evictions() does, asks a lookup
    m = re.search(r"std::shared_lock<std::shared_mutex>\s+read_lock\(g_cache_rw_mutex\);", code)
    return code[: m.end()] + "\n    (void) get_existing_cache_for_device(0);" + code[m.end() :]


LOCK_MUTANTS = {
    "get_existing_unified_cache_for_device locks again": _relock("get_existing_unified_cache_for_device"),
    "multi_device_plan_subtracts_arena locks again": _relock("multi_device_plan_subtracts_arena"),
    "a lookup under a held lock": _lookup_under_a_held_lock,
}


@pytest.mark.parametrize("name", sorted(LOCK_MUTANTS))
def test_the_nested_lock_check_catches_its_mutant(name):
    cache_code, _ = sources()
    mutant = LOCK_MUTANTS[name](cache_code)
    assert mutant != cache_code, f"the mutant '{name}' did not change the source"
    assert nested_lock_failures(mutant), f"the mutant '{name}' passed the gate"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
