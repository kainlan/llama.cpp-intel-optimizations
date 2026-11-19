"""Source contract: the per-device unified caches are destroyed before dpct's device manager at exit, and a lookup of
an existing cache takes the cache lock once.

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
    namespace, and by no static of another function: its one static declaration is inside the accessor, and the old
    namespace-scope name is gone, so no use can bypass the accessor;
  - the accessor touches the device manager before the map's static is reached;
  - the exit handler that marks SYCL as shutting down is registered after the map is constructed, so it runs before
    the map is destroyed and the caches the map destroys take the abandon branch of their teardown. A handler
    registered before an object's construction completes runs after that object is destroyed. The handler is
    registered by the first cache's constructor, and a cache may be constructed directly, by anyone, before any map
    lookup, so the constructor calls the accessor before any std::atexit in its body. create_cache_for_device() also
    takes the map before it constructs a cache: in `map[id] = std::make_unique<unified_cache>(...)` the right operand
    is evaluated first, so the accessor call there must be a statement of its own, before that one.

The cache lock g_cache_rw_mutex is a std::shared_mutex. A thread that holds it, in either mode, must not reach a lookup
that takes it again: twice shared on one thread is undefined and deadlocks once a writer queues between the two, and
shared under unique deadlocks outright. The gate requires every mention of the lock in unified-cache.cpp to be its
declaration or a lock the gate recognises, and no recognised lock's scope to call a lookup that takes the lock.

It reads comment-stripped text, and each check is proven to catch its mutant below, by the failure text of that check.
It cannot see code that is compiled out (`#if 0`) or never runs (`if (false)`), nor a call that reaches a lookup through
another function; the debugger run of a directly constructed cache is the behavioural gate for the order.
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

_LITERAL = (
    r'R"([^(\s]{0,16})\(.*?\)\1"'  # raw string literal
    r'|"(?:\\.|[^"\\\n])*"'  # string literal
    r"|'(?:\\.|[^'\\\n])*'"  # char literal
)
_LITERAL_RE = re.compile(_LITERAL, flags=re.DOTALL)
_LEXEME_RE = re.compile(
    _LITERAL  # kept
    + r"|//[^\n]*"  # line comment (dropped)
    + r"|/\*.*?\*/",  # block comment (dropped)
    flags=re.DOTALL,
)
_NAMESPACE_OPENER = re.compile(r"(?:\bnamespace(?:\s+[\w:]+)?|\bextern)\s*$")


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if not tok.startswith(("//", "/*")):
            return tok  # a literal
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def blank_literals(code: str) -> str:
    """The code with every literal's text replaced by spaces, offsets and lines kept: what every brace walk reads, so
    a brace inside a literal never counts."""
    return _LITERAL_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), code)


def body_span(code: str, signature: str):
    """(start, end) of the body after the first match of `signature`, which ends at its opening brace."""
    blank = blank_literals(code)
    m = re.search(signature, blank)
    if not m:
        return None
    depth = 1
    i = m.end()
    while i < len(blank) and depth:
        if blank[i] == "{":
            depth += 1
        elif blank[i] == "}":
            depth -= 1
        i += 1
    return m.end(), i - 1


def namespace_scope_map_decls(code: str) -> list:
    """Names and lines of the map-typed variables declared at namespace scope: every enclosing brace opens a namespace
    or an extern block. A file whose braces do not balance (a preprocessor branch opens one twice) cannot be walked, so
    every map-typed variable in it is reported, which fails closed."""
    blank = blank_literals(code)
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
            if not re.search(ACCESSOR_CALL, cbody[:statement_start]):
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
    # the map moved out of the accessor to a namespace-scope static under a name no other check knows
    def edit(body: str) -> str:
        name = re.search(STATIC_MAP_DECL, body).group(1)
        body = re.sub(STATIC_MAP_DECL, "", body, count=1)
        return re.sub(r"\breturn\s+" + name + r"\s*;", "return g_moved_caches;", body, count=1)

    code = _mutate_accessor(code, edit)
    m = re.search(ACCESSOR_SIG, code)
    decl = "static std::unordered_map<int, std::shared_ptr<unified_cache>> g_moved_caches;\n"
    return code[: m.start()] + decl + code[m.start() :]


def _bypass(code: str) -> str:
    # a second, namespace-scope holder of caches that a later use reads directly
    return code + "\nstatic std::unordered_map<int, std::shared_ptr<unified_cache>> g_spare_caches;\n"


def _external_linkage_map(code: str) -> str:
    # a namespace-scope holder without `static`: external linkage, the same construction at load
    return code + "\nstd::unordered_map<int, std::shared_ptr<unified_cache>> g_extern_caches;\n"


def _anonymous_namespace_map(code: str) -> str:
    return code + "\nnamespace {\nstd::unordered_map<int, std::shared_ptr<unified_cache>> g_unnamed_caches{};\n}\n"


def _static_in_another_function(code: str) -> str:
    # a second function-local holder: its own first use decides its order, not the accessor's touch
    decl = "static std::unordered_map<int, std::shared_ptr<unified_cache>> stray_caches;"
    return code + "\nstatic void spare() {\n    " + decl + "\n}\n"


def _construct_in_the_same_statement(code: str) -> str:
    start, end = body_span(code, CREATE_SIG)
    body = code[start:end]
    body = re.sub(r"[^;{}]*" + ACCESSOR_CALL + r"[^;]*;", "", body)
    body = re.sub(r"(\w+)(\s*\[\s*\w+\s*\]\s*=\s*std::make_unique\s*<\s*unified_cache\s*>)",
                  ACCESSOR + r"()\2", body, count=1)
    return code[:start] + body + code[end:]


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


# name -> (edit, a fragment of the failure only the check meant for it reports)
MUTANTS = {
    "the accessor does not touch the device manager": (
        lambda c: _mutate_accessor(c, _drop_touch), "the accessor does not touch dpct::dev_mgr::instance()"),
    "the accessor touches the device manager after the map's static": (
        lambda c: _mutate_accessor(c, _touch_after_static), "touches the device manager after the map's static"),
    "the map moves to a namespace-scope static": (
        _namespace_scope_map, "static `g_moved_caches` outside the accessor"),
    "a second static map of caches bypasses the accessor": (
        _bypass, "static `g_spare_caches` outside the accessor"),
    "a namespace-scope map without static bypasses the accessor": (
        _external_linkage_map, "namespace-scope map `g_extern_caches`"),
    "a map in an anonymous namespace bypasses the accessor": (
        _anonymous_namespace_map, "namespace-scope map `g_unnamed_caches`"),
    "a function-local static map in another function bypasses the accessor": (
        _static_in_another_function, "static `stray_caches` outside the accessor"),
    "the first cache is constructed before the map": (
        _construct_in_the_same_statement, "constructs a cache before it has taken the map"),
    "the cache constructor does not take the map": (
        lambda c: _mutate_ctor(c, _ctor_without_take), "does not take the map before it registers the exit handler"),
    "the cache constructor takes the map after std::atexit": (
        lambda c: _mutate_ctor(c, _ctor_take_after_atexit), "takes the map after it registers the exit handler"),
}


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_each_check_catches_its_mutant(name):
    cache_code, others = sources()
    edit, fragment = MUTANTS[name]
    mutant = edit(cache_code)
    assert mutant != cache_code, f"the mutant '{name}' did not change the source"
    failures = "\n".join(order_failures(mutant, others))
    assert fragment in failures, f"the mutant '{name}' was not caught by its check; failures:\n{failures}"


def test_the_old_name_in_another_source_is_caught():
    cache_code, others = sources()
    rel = sorted(others)[0]
    others = dict(others)
    others[rel] = others[rel] + "\nvoid f() { g_device_caches.clear(); }\n"
    assert f"{rel} still names g_device_caches" in order_failures(cache_code, others)


def test_a_map_local_to_a_function_is_not_a_holder():
    # the scope walk's control: shutdown_unified_cache() copies the map into a local of the same type, and a function
    # or class body is not a namespace scope, braces in literals included
    cache_code, others = sources()
    local = '\nvoid f() { const char * s = "}"; std::unordered_map<int, std::shared_ptr<unified_cache>> local; }\n'
    assert namespace_scope_map_decls(local) == []
    assert order_failures(cache_code + local, others) == order_failures(cache_code, others)


# A lock of g_cache_rw_mutex: explicit or deduced template argument, and an optional second constructor argument such
# as std::defer_lock. A deferred lock counts as held for its whole scope, which can only report more.
CACHE_LOCK = re.compile(
    r"\bstd::(?:shared_lock|unique_lock|lock_guard|scoped_lock)\s*(?:<\s*std::shared_mutex\s*>)?\s*\w+\s*[({]\s*"
    r"g_cache_rw_mutex\s*(?:,\s*[\w:]+\s*)?[)}]"
)
CACHE_LOCK_DECL = re.compile(r"\bstatic\s+std::shared_mutex\s+g_cache_rw_mutex\s*;")
CACHE_LOCK_NAME = re.compile(r"\bg_cache_rw_mutex\b")
# the lookups that take g_cache_rw_mutex shared for the map read, directly or through one of the others
RELOCKING_LOOKUPS = (
    "get_cache_shared",
    "get_existing_cache_for_device",
    "get_existing_unified_cache_for_device",
    "multi_device_plan_arena_cache",
    "multi_device_plan_subtracts_arena",
)
LOOKUP_SIGS = {
    "get_existing_unified_cache_for_device": r"(?<!static )\bunified_cache\s*\*\s*get_existing_unified_cache_for_device"
    r"\s*\(\s*int\s+\w+\s*\)\s*\{",
    "multi_device_plan_subtracts_arena": r"\bbool\s+multi_device_plan_subtracts_arena\s*\(\s*int\s+\w+\s*\)\s*\{",
}


def nested_lock_failures(cache_code: str) -> list:
    """Every mention of g_cache_rw_mutex that is neither its declaration nor a lock CACHE_LOCK recognises, so a new
    lock form fails here instead of passing unseen; and every lookup of RELOCKING_LOOKUPS called while a recognised
    lock is held. A lock's scope runs from its declaration to the close of the block that holds it."""
    failures = []
    blank = blank_literals(cache_code)
    locks = list(CACHE_LOCK.finditer(blank))
    covered = [(m.start(), m.end()) for m in locks]
    covered += [(m.start(), m.end()) for m in CACHE_LOCK_DECL.finditer(blank)]
    for name in CACHE_LOCK_NAME.finditer(blank):
        if not any(a <= name.start() < b for a, b in covered):
            line = blank.count("\n", 0, name.start()) + 1
            failures.append(f"unified-cache.cpp:{line} names g_cache_rw_mutex outside a lock the pin recognises")
    calls = re.compile(r"\b(" + "|".join(RELOCKING_LOOKUPS) + r")\s*\(")
    for lock in locks:
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
    return failures


def test_no_cache_lookup_runs_under_the_cache_lock():
    cache_code, _ = sources()
    failures = nested_lock_failures(cache_code)
    assert not failures, "\n".join(failures)


def test_the_lock_pattern_sees_every_lock_of_the_cache_mutex():
    # coverage's positive control: the file's own locks are all recognised, and there are some
    cache_code, _ = sources()
    assert len(CACHE_LOCK.findall(blank_literals(cache_code))) > 0


def _insert_into(name: str, text: str):
    def edit(code: str) -> str:
        start, _ = body_span(code, LOOKUP_SIGS[name])
        return code[:start] + text + code[start:]

    return edit


_HELD_READ_LOCK = r"std::shared_lock<std::shared_mutex>\s+read_lock\(g_cache_rw_mutex\);"


def _lookup_under_a_held_lock(code: str) -> str:
    # a function that already holds the lock, as get_unified_cache() does, asks a lookup
    m = re.search(_HELD_READ_LOCK, code)
    return code[: m.end()] + "\n    (void) get_cache_shared(0);" + code[m.end() :]


def _lookup_under_a_deduced_lock(code: str) -> str:
    # the same, with that function's lock written with its template argument deduced
    m = re.search(_HELD_READ_LOCK, code)
    return code[: m.start()] + "std::shared_lock read_lock(g_cache_rw_mutex);\n    (void) get_cache_shared(0);" + code[
        m.end() :]


LOCK_MUTANTS = {
    "get_existing_unified_cache_for_device locks again": (
        _insert_into("get_existing_unified_cache_for_device",
                     "\n    std::shared_lock<std::shared_mutex> lock(g_cache_rw_mutex);"),
        "holds g_cache_rw_mutex and calls get_existing_cache_for_device()"),
    "multi_device_plan_subtracts_arena locks again, template argument deduced": (
        _insert_into("multi_device_plan_subtracts_arena", "\n    std::shared_lock lock(g_cache_rw_mutex);"),
        "holds g_cache_rw_mutex and calls multi_device_plan_arena_cache()"),
    "get_existing_unified_cache_for_device locks again, deduced and deferred": (
        _insert_into("get_existing_unified_cache_for_device",
                     "\n    std::unique_lock lock(g_cache_rw_mutex, std::defer_lock);\n    lock.lock();"),
        "holds g_cache_rw_mutex and calls get_existing_cache_for_device()"),
    "a lookup under a held lock": (
        _lookup_under_a_held_lock, "holds g_cache_rw_mutex and calls get_cache_shared()"),
    "a lookup under a held lock, template argument deduced": (
        _lookup_under_a_deduced_lock, "holds g_cache_rw_mutex and calls get_cache_shared()"),
    "a lock form the pattern does not know": (
        _insert_into("get_existing_unified_cache_for_device", "\n    g_cache_rw_mutex.lock_shared();"),
        "names g_cache_rw_mutex outside a lock the pin recognises"),
}


@pytest.mark.parametrize("name", sorted(LOCK_MUTANTS))
def test_the_nested_lock_check_catches_its_mutant(name):
    cache_code, _ = sources()
    edit, fragment = LOCK_MUTANTS[name]
    mutant = edit(cache_code)
    assert mutant != cache_code, f"the mutant '{name}' did not change the source"
    failures = "\n".join(nested_lock_failures(mutant))
    assert fragment in failures, f"the mutant '{name}' was not caught by its check; failures:\n{failures}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
