#!/usr/bin/env python3
"""The SYCL backend does not execute a node the graph did not ask for.

A node without GGML_TENSOR_FLAG_COMPUTE is an unselected branch of ggml_build_forward_select. The scheduler does not
upload that node's inputs, so a kernel run for it reads memory nobody wrote (on the B50 at default context: a
GET_ROWS on the token embedding with garbage ids, a GPU page fault and a lost device).

With comments and string literals blanked, in ggml-sycl.cpp:
  H. ggml_sycl_node_is_computed, the one statement of the rule, tests `(node->flags & GGML_TENSOR_FLAG_COMPUTE) != 0`.
  P. ggml_sycl_is_noop, the predicate every dispatching walker consults, returns true for a node that is not
     computed: `!ggml_sycl_node_is_computed(node) ||`.
  E1-E5. The walks that must keep views (ggml_sycl_is_noop is true for views) test the helper themselves:
     the block executor's two alias walks (ggml_sycl_probe_block_exec_runtime_context,
     ggml_sycl_block_exec_publish_aliases) and extract_persistent_plan's recipe replay, fast path and full build.
  B1. extract_persistent_plan's cached path compares the selection hash stored with the plan against the current
     graph's before it replays anything, and on a mismatch drops the plan and rebuilds. Without it, a node computed
     now but not when the plan was built (a token decode after embedding decodes) would never run.
  B2. The full build stores that hash. B3. The hash folds every node's computed bit.
  S. ggml_sycl_graph_signature mixes node->flags, so a graph recorded for one selection is not replayed for another.
  W. Walker census: every function that indexes cgraph->nodes[ and calls ggml_sycl_compute_forward( also consults
     ggml_sycl_is_noop or ggml_sycl_node_is_computed, unless WALKER_ALLOW names it with its reason.
  W0. The census fails closed: every ggml_sycl_compute_forward( call must lie inside a function it parsed.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"

HELPER_SIG = "static inline bool ggml_sycl_node_is_computed(const ggml_tensor * node) {"
NOOP_SIG = "static inline bool ggml_sycl_is_noop(const ggml_tensor * node) {"
GRAPH_SIG = "static uint64_t ggml_sycl_graph_signature(const ggml_cgraph * cgraph) {"
SELECTION_SIG = "static uint64_t ggml_sycl_graph_selection_hash(const ggml_cgraph * cgraph) {"

HELPER_TEST = re.compile(r"\(\s*node->flags\s*&\s*GGML_TENSOR_FLAG_COMPUTE\s*\)\s*!=\s*0")
NOOP_TEST = re.compile(r"!\s*ggml_sycl_node_is_computed\(\s*node\s*\)\s*\|\|")
FLAG_MIX = re.compile(r"\bmix\(\s*static_cast<uint64_t>\(\s*node->flags\s*\)\s*\)\s*;")
ALIAS_TEST = re.compile(
    r"!\s*ggml_sycl_node_is_computed\(\s*node\s*\)\s*\|\|\s*ggml_sycl_block_exec_node_needs_local_output\(\s*node\s*\)")
RECIPE_TEST = re.compile(r"recipe_node\s*&&\s*!\s*ggml_sycl_node_is_computed\(\s*recipe_node\s*\)")
FAST_TEST = re.compile(r"if\s*\(\s*!node\s*\|\|\s*!\s*ggml_sycl_node_is_computed\(\s*node\s*\)\s*\)")
BUILD_TEST = re.compile(r"if\s*\(\s*node\s*&&\s*!\s*ggml_sycl_node_is_computed\(\s*node\s*\)\s*\)")
SELECTION_COMPARE = re.compile(r"kernel\.plan_selection_hash\(\)\s*!=\s*ggml_sycl_graph_selection_hash\(\s*cgraph\s*\)")
SELECTION_STORE = re.compile(r"kernel\.set_plan_selection_hash\(\s*ggml_sycl_graph_selection_hash\(\s*cgraph\s*\)\s*\)")
SELECTION_FOLD = re.compile(r"ggml_sycl_node_is_computed\(\s*node\s*\)")
WALKER_GUARD = re.compile(r"\bggml_sycl_(?:is_noop|node_is_computed)\(")

# Functions that walk cgraph->nodes and dispatch, yet need no guard of their own. Each entry says why.
WALKER_ALLOW = {
    "moe_graph_record_moe_dispatch_graph":
        "dispatches the MoE layer at the node index its callers pass; every caller (the graph_compute_impl node loop, "
        "moe_graph_record_segments, the segment dispatch) reaches that index only through a node it already "
        "filtered with ggml_sycl_is_noop, and MoE layer pairing never admits a node that is not computed",
}


def blank_comments(text):
    text = re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), text)
    return re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)


def blank_literals(text):
    """Blank preprocessor lines (with their continuations) and string and character literals, so braces inside them
    do not count and a directive does not join the next function's header (the file has no raw strings)."""
    text = re.sub(r"^[ \t]*#(?:[^\n]*\\\n)*[^\n]*", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.M)
    text = re.sub(r'"(?:\\.|[^"\\\n])*"', lambda m: '"' + " " * (len(m.group(0)) - 2) + '"', text)
    return re.sub(r"'(?:\\.|[^'\\\n])'", lambda m: "'" + " " * (len(m.group(0)) - 2) + "'", text)


def span(text, signature):
    """(open, close) offsets of the brace-balanced body that starts at the end of `signature`, or None."""
    start = text.find(signature)
    if start < 0:
        return None
    open_at = start + len(signature) - 1
    depth = 0
    for i in range(open_at, len(text)):
        depth += text[i] == "{"
        depth -= text[i] == "}"
        if depth == 0:
            return open_at, i + 1
    return None


def body(text, signature):
    s = span(text, signature)
    return text[s[0]:s[1]] if s else ""


def function_span(text, name):
    """Body span of the first function definition named `name` (the signature may wrap)."""
    call = re.compile(r"\b" + re.escape(name) + r"\s*\(")
    m = call.search(text)
    while m:
        depth = 0
        i = m.end() - 1
        while i < len(text):
            depth += text[i] == "("
            depth -= text[i] == ")"
            i += 1
            if depth == 0:
                break
        tail = re.match(r"\s*(?:const\s*)?(?:noexcept\s*)?\{", text[i:])
        if tail:
            return span_from(text, i + tail.end() - 1)
        m = call.search(text, m.end())
    return None


def span_from(text, open_at):
    depth = 0
    for i in range(open_at, len(text)):
        depth += text[i] == "{"
        depth -= text[i] == "}"
        if depth == 0:
            return open_at, i + 1
    return None


def persistent_regions(text):
    """extract_persistent_plan split at its full_build: label: (cached path, full build), or ('', '')."""
    s = function_span(text, "extract_persistent_plan")
    if not s:
        return "", ""
    b = text[s[0]:s[1]]
    label = re.search(r"^\s*full_build\s*:", b, flags=re.M)
    if not label:
        return b, ""
    return b[:label.start()], b[label.start():]


SCOPE_HEADER = re.compile(r"(?:\b(?:namespace|struct|class|union|enum)\b(?:\s*alignas\(\w+\))?[^()=]*|\bextern\s+\"\s*\"\s*)$", re.S)
FUNC_HEADER = re.compile(r"\)\s*(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?(?:->\s*[\w:<>*&\s]+)?(?:try\s*)?$", re.S)


def function_spans(text):
    """(name, open, close) for every function defined outside another function. Namespace, struct and class bodies are
    transparent, so their member functions count; any other brace at that level (an initialiser) is opaque."""
    out = []
    stack = []
    last = 0
    for m in re.finditer(r"[{};]", text):
        ch = m.group(0)
        in_fn = any(k in ("fn", "other") for k, _ in stack)
        if ch == ";":
            if not in_fn:
                last = m.end()
            continue
        if ch == "{":
            if in_fn:
                stack.append(("inner", m.start()))
                continue
            header = text[last:m.start()]
            if SCOPE_HEADER.search(header.strip()):
                stack.append(("scope", m.start()))
            elif all(("(" in header, FUNC_HEADER.search(header.rstrip()) is not None,
                      "=" not in header.split("(")[0] or "operator" in header)):
                pre = header[:header.find("(")]
                name = re.findall(r"[A-Za-z_][\w:~]*", pre)
                stack.append(("fn", (name[-1] if name else "?", m.start())))
            else:
                stack.append(("other", m.start()))
            last = m.end()
            continue
        if stack:
            kind, info = stack.pop()
            if kind == "fn":
                out.append((info[0].split("::")[-1], info[1], m.end()))
        if not any(k in ("fn", "other") for k, _ in stack):
            last = m.end()
    return out


def functions(text):
    return [(name, text[lo:hi]) for name, lo, hi in function_spans(text)]


def walkers(text):
    return [(name, b) for name, b in functions(text) if "cgraph->nodes[" in b and "ggml_sycl_compute_forward(" in b]


def stray_dispatch_calls(text):
    """Line numbers of ggml_sycl_compute_forward( calls outside every function the census parsed. The census fails
    closed on any: a parser miss must not hide a walker. Declarations and the definition do not count."""
    bodies = [(lo, hi) for _, lo, hi in function_spans(text)]
    stray = []
    for m in re.finditer(r"\bggml_sycl_compute_forward\(", text):
        if any(lo <= m.start() < hi for lo, hi in bodies):
            continue
        line_start = text.rfind("\n", 0, m.start()) + 1
        if re.fullmatch(r"\s*(?:static\s+)?(?:inline\s+)?bool\s*", text[line_start:m.start()]):
            continue
        stray.append(text.count("\n", 0, m.start()) + 1)
    return stray


def claims(sycl):
    helper = body(sycl, HELPER_SIG)
    noop = body(sycl, NOOP_SIG)
    sig = body(sycl, GRAPH_SIG)
    sel = body(sycl, SELECTION_SIG)
    probe = function_span(sycl, "ggml_sycl_probe_block_exec_runtime_context")
    aliases = function_span(sycl, "ggml_sycl_block_exec_publish_aliases")
    cached, build = persistent_regions(sycl)

    compare = SELECTION_COMPARE.search(cached)
    recipe_use = cached.find("kernel.get_update_recipe()")
    after = cached[compare.end():compare.end() + 600] if compare else ""
    compare_ok = all((compare is not None, 0 <= (compare.start() if compare else -1) < recipe_use,
                      "invalidate_plan_cache()" in after, re.search(r"goto\s+full_build\s*;", after) is not None))

    unguarded = [name for name, b in walkers(sycl) if not WALKER_GUARD.search(b) and name not in WALKER_ALLOW]
    stray = stray_dispatch_calls(sycl)
    return {
        "H ggml_sycl_node_is_computed tests GGML_TENSOR_FLAG_COMPUTE != 0": HELPER_TEST.search(helper) is not None,
        "P ggml_sycl_is_noop is true for a node that is not computed": NOOP_TEST.search(noop) is not None,
        "E1 ggml_sycl_probe_block_exec_runtime_context alias walk tests the helper":
            probe is not None and ALIAS_TEST.search(sycl[probe[0]:probe[1]]) is not None,
        "E2 ggml_sycl_block_exec_publish_aliases tests the helper":
            aliases is not None and ALIAS_TEST.search(sycl[aliases[0]:aliases[1]]) is not None,
        "E3 extract_persistent_plan recipe replay tests the helper": RECIPE_TEST.search(cached) is not None,
        "E4 extract_persistent_plan fast path tests the helper": FAST_TEST.search(cached) is not None,
        "E5 extract_persistent_plan full build tests the helper": BUILD_TEST.search(build) is not None,
        "B1 extract_persistent_plan compares the stored selection hash before it replays": compare_ok,
        "B2 extract_persistent_plan full build stores the selection hash": SELECTION_STORE.search(build) is not None,
        "B3 ggml_sycl_graph_selection_hash folds each node's computed bit": SELECTION_FOLD.search(sel) is not None,
        "S ggml_sycl_graph_signature mixes node->flags": FLAG_MIX.search(sig) is not None,
        "W every dispatching walker consults the rule (unguarded: " + ", ".join(unguarded) + ")": not unguarded,
        "W0 the census parsed every ggml_sycl_compute_forward call (stray at lines: %s)" % stray: not stray,
    }


def check(sycl):
    return [name for name, ok in claims(sycl).items() if not ok]


def mutate_span(text, s, pattern, repl):
    """Apply `pattern` -> `repl` once inside text[s[0]:s[1]] only."""
    if not s:
        return text
    return text[:s[0]] + pattern.sub(repl, text[s[0]:s[1]], count=1) + text[s[1]:]


def mutate_in_body(text, signature, pattern, repl):
    return mutate_span(text, span(text, signature), pattern, repl)


def persistent_span(text, part):
    s = function_span(text, "extract_persistent_plan")
    if not s:
        return None
    label = re.search(r"^\s*full_build\s*:", text[s[0]:s[1]], flags=re.M)
    if not label:
        return None
    cut = s[0] + label.start()
    return (s[0], cut) if part == "cached" else (cut, s[1])


def mutants(sycl):
    guarded = [name for name, b in walkers(sycl) if WALKER_GUARD.search(b)]
    victim = "moe_graph_replay_segments" if "moe_graph_replay_segments" in guarded else (guarded[0] if guarded else "")
    victim_span = function_span(sycl, victim) if victim else None
    strip_guards = sycl
    if victim_span:
        stripped = WALKER_GUARD.sub("ggml_sycl_guard_removed(", sycl[victim_span[0]:victim_span[1]])
        strip_guards = sycl[:victim_span[0]] + stripped + sycl[victim_span[1]:]
    return {
        "flag test removed from ggml_sycl_is_noop": mutate_in_body(sycl, NOOP_SIG, NOOP_TEST, ""),
        "flag test inverted in ggml_sycl_is_noop": mutate_in_body(
            sycl, NOOP_SIG, re.compile(r"!\s*(ggml_sycl_node_is_computed\(\s*node\s*\)\s*\|\|)"), r"\1"),
        "flag test inverted in ggml_sycl_node_is_computed": mutate_in_body(
            sycl, HELPER_SIG, re.compile(r"(\(\s*node->flags\s*&\s*GGML_TENSOR_FLAG_COMPUTE\s*\)\s*)!=(\s*0)"),
            r"\1==\2"),
        "helper test removed from ggml_sycl_probe_block_exec_runtime_context": mutate_span(
            sycl, function_span(sycl, "ggml_sycl_probe_block_exec_runtime_context"),
            re.compile(r"!\s*ggml_sycl_node_is_computed\(\s*node\s*\)\s*\|\|"), ""),
        "helper test removed from ggml_sycl_block_exec_publish_aliases": mutate_span(
            sycl, function_span(sycl, "ggml_sycl_block_exec_publish_aliases"),
            re.compile(r"!\s*ggml_sycl_node_is_computed\(\s*node\s*\)\s*\|\|"), ""),
        "helper test removed from the persistent recipe replay": mutate_span(
            sycl, persistent_span(sycl, "cached"), RECIPE_TEST, "false"),
        "helper test removed from the persistent fast path": mutate_span(
            sycl, persistent_span(sycl, "cached"), FAST_TEST, "if (!node)"),
        "helper test removed from the persistent full build": mutate_span(
            sycl, persistent_span(sycl, "build"), BUILD_TEST, "if (false)"),
        "selection hash compare removed from the persistent cached path": mutate_span(
            sycl, persistent_span(sycl, "cached"), SELECTION_COMPARE, "false"),
        "selection hash store removed from the persistent full build": mutate_span(
            sycl, persistent_span(sycl, "build"), re.compile(SELECTION_STORE.pattern + r"\s*;"), ""),
        "mix(node->flags) removed from ggml_sycl_graph_signature": mutate_in_body(sycl, GRAPH_SIG, FLAG_MIX, ""),
        f"guard stripped from walker {victim}": strip_guards,
    }


def main():
    sycl = blank_literals(blank_comments((SYCL / "ggml-sycl.cpp").read_text()))
    failed = check(sycl)
    found = walkers(sycl)
    print("compute flag source contract: walker census: %d functions index cgraph->nodes and call "
          "ggml_sycl_compute_forward (%d guarded, %d allowlisted): %s" %
          (len(found), sum(1 for _, b in found if WALKER_GUARD.search(b)),
           sum(1 for n, b in found if not WALKER_GUARD.search(b) and n in WALKER_ALLOW),
           ", ".join(n for n, _ in found)))
    if failed:
        print("compute flag source contract failed: " + "; ".join(failed), file=sys.stderr)
        return 1
    for name in WALKER_ALLOW:
        if name not in [n for n, _ in found]:
            print(f"compute flag source contract: allowlisted walker '{name}' is gone; drop its entry", file=sys.stderr)
            return 1
    if "--self-test" in sys.argv:
        for name, text in mutants(sycl).items():
            if text == sycl:
                print(f"compute flag source contract: mutant '{name}' did not apply", file=sys.stderr)
                return 1
            if not check(text):
                print(f"compute flag source contract: mutant '{name}' was not caught", file=sys.stderr)
                return 1
            print(f"compute flag source contract: mutant '{name}' caught")
    print("compute flag source contract: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
