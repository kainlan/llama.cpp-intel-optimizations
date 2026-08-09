#!/usr/bin/env python3
"""AST-backed contract for the canonical-checksum owner scope (llama.cpp-x3ou).

Covers exactly what the behavioural gate cannot see. test-canonical-checksum-
owner-scope.cpp proves that two live models read their own rows; it cannot
observe the teardown erase (a reused slot carries a new generation, so a missing
erase leaks memory rather than answers), it cannot observe the fail-closed
branch (its owner always resolves), and it cannot observe a removal. Those four
facts are structural, so they are checked structurally.

Comments and string literals are invisible here, by construction: the checks
walk identifier nodes. That is not incidental. A tombstone comment naming a
removed symbol, or a doc line quoting the bad key expression, must not trip the
check that forbids them -- the mutation matrix carries a positive case that
would fail if it did.

Usage:
  test-canonical-checksum-source-contract.py --source <ggml-sycl.cpp> \
      --header <ggml-sycl.h> [--mutation-matrix]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Iterable

try:
    from tree_sitter_language_pack import get_parser
except ImportError:
    print(
        "SKIP: canonical-checksum-source-contract requires the test-only tree-sitter-language-pack module",
        file=sys.stderr,
    )
    raise SystemExit(77)

MAP = "g_sycl_canonical_checksums"
KEY_FN = "ggml_sycl_canonical_checksum_key"
MATCHES_FN = "ggml_sycl_owner_name_key_matches"
ERASE_FN = "ggml_sycl_erase_weight_identities_for_owner"
CAPTURE_FN = "ggml_sycl_maybe_capture_canonical_checksum"
LOOKUP_FN = "ggml_sycl_get_canonical_checksum"
RESET_FN = "ggml_sycl_reset_canonical_checksums"
SHUTDOWN_SIBLING = "ggml_sycl_reset_pending_kv_layer_masks"

# Removed by the owner ruling on llama.cpp-x3ou (option 1, full removal).
REMOVED = (
    "g_moe_expert_split_active",
    "ggml_backend_sycl_set_moe_expert_split",
    "ggml_backend_sycl_query_moe_expert_split",
)

IDENTIFIER_KINDS = {"identifier", "field_identifier", "type_identifier", "namespace_identifier"}

# Statement kinds that execute whenever their enclosing block does. A loop is
# included because the erase IS a loop; a branch is not.
UNCONDITIONAL_CARRIER_KINDS = {
    "for_statement",
    "for_range_loop",
    "expression_statement",
    "declaration",
}


class ContractError(AssertionError):
    pass


def kind(node) -> str:
    return node.kind()


def children(node) -> list:
    return [node.named_child(i) for i in range(node.named_child_count())]


def walk(node) -> Iterable:
    yield node
    for child in children(node):
        yield from walk(child)


class Ast:
    def __init__(self, source: str) -> None:
        self.encoded = source.encode("utf-8")
        self.root = get_parser("cpp").parse(source).root_node()

    def text(self, node) -> str:
        return self.encoded[node.start_byte() : node.end_byte()].decode("utf-8")

    def functions(self, name: str) -> list:
        result = []
        for node in walk(self.root):
            if kind(node) != "function_definition":
                continue
            declarator = node.child_by_field_name("declarator")
            if declarator and any(
                kind(n) == "identifier" and self.text(n) == name for n in walk(declarator)
            ):
                result.append(node)
        return result

    def function(self, name: str):
        matches = self.functions(name)
        if len(matches) != 1:
            raise ContractError(f"expected one AST function {name}, found {len(matches)}")
        return matches[0]

    def identifiers(self, node, name: str) -> list:
        return [n for n in walk(node) if kind(n) in IDENTIFIER_KINDS and self.text(n) == name]

    def calls(self, node, name: str) -> list:
        result = []
        for candidate in walk(node):
            if kind(candidate) != "call_expression":
                continue
            function = candidate.child_by_field_name("function")
            if function and self.text(function).split("(")[0].strip().endswith(name):
                result.append(candidate)
        return result


def unconditional_statements(compound) -> list:
    """Flatten standalone compound statements only; never enter a control node."""
    result = []
    for child in children(compound):
        if kind(child) == "compound_statement":
            result.extend(unconditional_statements(child))
        else:
            result.append(child)
    return result


# ---------------------------------------------------------------------------
# C1/C2 -- the teardown erase exists, is owner-guarded, and is unconditional
# ---------------------------------------------------------------------------


def verify_owner_erase(ast: Ast) -> None:
    body = ast.function(ERASE_FN).child_by_field_name("body")
    if body is None:
        raise ContractError(f"{ERASE_FN}: no body")

    if not ast.identifiers(body, MAP):
        raise ContractError(f"owner erase: {ERASE_FN} does not touch {MAP}")

    guarded = [
        node
        for node in walk(body)
        if kind(node) == "conditional_expression" and ast.identifiers(node, MATCHES_FN) and ast.identifiers(node, MAP)
    ]
    if not guarded:
        raise ContractError(f"owner erase: {MAP} rows are not selected by {MATCHES_FN}")

    # The erase must be reachable without passing a branch. Flattening enters
    # bare scopes only, so a statement carrying the erase is on the
    # unconditional path exactly when it is not itself a control node -- which
    # is why the kind is checked rather than merely the presence of the
    # identifier somewhere in the subtree. `if (false) { <erase> }` keeps the
    # identifier reachable from the body and would otherwise read as compliant.
    carriers = [
        statement
        for statement in unconditional_statements(body)
        if ast.identifiers(statement, MAP)
    ]
    if not carriers:
        raise ContractError(f"owner erase: the {MAP} erase is not on the unconditional path")
    if not any(kind(statement) in UNCONDITIONAL_CARRIER_KINDS for statement in carriers):
        raise ContractError(
            f"owner erase: the {MAP} erase is not on the unconditional path "
            f"(reached only through {sorted({kind(s) for s in carriers})})"
        )


# ---------------------------------------------------------------------------
# C3/C4 -- both accessors key on the owner scope, and fail closed without one
# ---------------------------------------------------------------------------


def verify_accessor(ast: Ast, name: str) -> None:
    body = ast.function(name).child_by_field_name("body")
    if body is None:
        raise ContractError(f"{name}: no body")

    key_decls = [
        node
        for node in walk(body)
        if kind(node) == "declaration" and ast.calls(node, KEY_FN)
    ]
    if len(key_decls) != 1:
        raise ContractError(f"{name}: expected exactly one {KEY_FN} result binding, found {len(key_decls)}")

    key_names = {
        ast.text(n)
        for n in walk(key_decls[0])
        if kind(n) == "identifier" and ast.text(n) not in {KEY_FN, "tensor"}
    }
    if len(key_names) != 1:
        raise ContractError(f"{name}: could not resolve a single key binding name, saw {sorted(key_names)}")
    key_name = next(iter(key_names))

    # Every access to the map must go through that binding.
    for node in walk(body):
        if kind(node) == "subscript_expression":
            argument = node.child_by_field_name("argument")
            indices = node.child_by_field_name("indices")
            if argument is not None and ast.text(argument) == MAP:
                actual = children(indices) if indices is not None else []
                if len(actual) != 1 or ast.text(actual[0]) != key_name:
                    shown = ast.text(actual[0]) if actual else "?"
                    raise ContractError(f"{name}: {MAP} is indexed by {shown}, not the owner key")
        if kind(node) == "call_expression":
            function = node.child_by_field_name("function")
            arguments = node.child_by_field_name("arguments")
            if function is None or arguments is None:
                continue
            if ast.text(function) != f"{MAP}.find":
                continue
            actual = [c for c in children(arguments)]
            if len(actual) != 1 or ast.text(actual[0]) != key_name:
                raise ContractError(
                    f"{name}: {MAP}.find is called with "
                    f"{ast.text(actual[0]) if actual else '?'}, not the owner key"
                )

    # Fail closed: an empty key must return before the map is touched.
    statements = unconditional_statements(body)
    guard_index = None
    map_index = None
    for index, statement in enumerate(statements):
        if guard_index is None and kind(statement) == "if_statement":
            condition = statement.child_by_field_name("condition")
            consequence = statement.child_by_field_name("consequence")
            if (
                condition is not None
                and consequence is not None
                and ast.identifiers(condition, key_name)
                and any(kind(n) == "return_statement" for n in walk(consequence))
            ):
                guard_index = index
        if map_index is None and ast.identifiers(statement, MAP):
            map_index = index
    if guard_index is None:
        raise ContractError(f"{name}: no empty-key guard returns before the map is used")
    if map_index is not None and map_index < guard_index:
        raise ContractError(f"{name}: the map is used before the empty-key guard")


# ---------------------------------------------------------------------------
# C5 -- module shutdown drops the table
# ---------------------------------------------------------------------------


def verify_shutdown_reset(ast: Ast) -> None:
    reset = ast.function(RESET_FN)
    if not ast.identifiers(reset, MAP):
        raise ContractError(f"shutdown reset: {RESET_FN} does not clear {MAP}")

    callers = [
        node
        for node in walk(ast.root)
        if kind(node) == "function_definition"
        and ast.calls(node, RESET_FN)
        and ast.calls(node, SHUTDOWN_SIBLING)
    ]
    if not callers:
        raise ContractError(f"shutdown reset: {RESET_FN} is not called beside {SHUTDOWN_SIBLING} at module shutdown")


# ---------------------------------------------------------------------------
# C6/C7 -- the expert-split symbols are gone, as identifiers
# ---------------------------------------------------------------------------


def verify_removed(ast: Ast, path: str) -> None:
    for name in REMOVED:
        hits = ast.identifiers(ast.root, name)
        if hits:
            raise ContractError(f"{path}: removed symbol {name} is still referenced ({len(hits)} identifier(s))")


def verify_contract(source: str, header: str) -> None:
    ast = Ast(source)
    verify_owner_erase(ast)
    verify_accessor(ast, CAPTURE_FN)
    verify_accessor(ast, LOOKUP_FN)
    verify_shutdown_reset(ast)
    verify_removed(ast, "ggml-sycl.cpp")
    verify_removed(Ast(header), "ggml-sycl.h")


# ---------------------------------------------------------------------------
# Mutation matrix
# ---------------------------------------------------------------------------


def replace_once(text: str, old: str, new: str, label: str) -> str:
    """Mutate exactly one site, or fail loudly.

    The anchors below are verbatim copies of source text. When that source is
    reformatted they stop matching and this raises -- which is the intended
    failure. A mutation that silently rewrites nothing would leave the matrix
    reporting a check as unfired when it was never exercised.
    """
    if text.count(old) != 1:
        raise AssertionError(f"mutation {label}: anchor matched {text.count(old)} times, expected 1")
    return text.replace(old, new, 1)


ERASE_BLOCK = """    {
        // The canonical-checksum diagnostic carries the same owner prefix, so a
        // later load reusing this slot would otherwise score its own weights
        // against a dead model's rows.
        std::lock_guard<std::mutex> lock(g_sycl_canonical_checksum_mutex);
        for (auto it = g_sycl_canonical_checksums.begin(); it != g_sycl_canonical_checksums.end();) {
            it = ggml_sycl_owner_name_key_matches(it->first, owner) ? g_sycl_canonical_checksums.erase(it) :
                                                                      std::next(it);
        }
    }
"""

EMPTY_KEY_GUARD = """    if (key.empty()) {
        // No owner: record nothing.  Storing under the bare name would rebuild
        // the cross-model bucket the owner scope exists to remove.
        return;
    }

"""


def build_cases(source: str, header: str) -> list:
    cases = []

    cases.append(
        (
            "erase-block-deleted",
            replace_once(source, ERASE_BLOCK, "", "erase-block-deleted"),
            header,
            "owner erase: ggml_sycl_erase_weight_identities_for_owner does not touch g_sycl_canonical_checksums",
        )
    )
    cases.append(
        (
            "erase-block-conditional",
            replace_once(source, ERASE_BLOCK, "    if (false) {\n" + ERASE_BLOCK + "    }\n", "erase-block-conditional"),
            header,
            "owner erase: the g_sycl_canonical_checksums erase is not on the unconditional path",
        )
    )
    cases.append(
        (
            "erase-unguarded-clear",
            replace_once(
                source,
                ERASE_BLOCK,
                "    {\n"
                "        std::lock_guard<std::mutex> lock(g_sycl_canonical_checksum_mutex);\n"
                "        g_sycl_canonical_checksums.clear();\n"
                "    }\n",
                "erase-unguarded-clear",
            ),
            header,
            "owner erase: g_sycl_canonical_checksums rows are not selected by ggml_sycl_owner_name_key_matches",
        )
    )
    cases.append(
        (
            "capture-guard-removed",
            replace_once(source, EMPTY_KEY_GUARD, "", "capture-guard-removed"),
            header,
            "no empty-key guard returns before the map is used",
        )
    )
    cases.append(
        (
            "capture-bare-name-key",
            replace_once(
                source,
                "        g_sycl_canonical_checksums[key] = ggml_sycl_canonical_checksum_info{ bytes, checksum };",
                "        g_sycl_canonical_checksums[std::string(tensor->name)] = "
                "ggml_sycl_canonical_checksum_info{ bytes, checksum };",
                "capture-bare-name-key",
            ),
            header,
            "is indexed by",
        )
    )
    cases.append(
        (
            "lookup-bare-name-key",
            replace_once(
                source,
                "    auto                        it = g_sycl_canonical_checksums.find(key);",
                "    auto                        it = g_sycl_canonical_checksums.find(std::string(tensor->name));",
                "lookup-bare-name-key",
            ),
            header,
            "is called with",
        )
    )
    cases.append(
        (
            "shutdown-reset-uncalled",
            replace_once(source, "    ggml_sycl_reset_canonical_checksums();\n", "", "shutdown-reset-uncalled"),
            header,
            "ggml_sycl_reset_canonical_checksums is not called beside ggml_sycl_reset_pending_kv_layer_masks",
        )
    )
    cases.append(
        (
            "expert-split-flag-restored",
            replace_once(
                source,
                "static void ggml_sycl_reset_canonical_checksums() {",
                "static std::atomic<bool> g_moe_expert_split_active{ false };\n\n"
                "static void ggml_sycl_reset_canonical_checksums() {",
                "expert-split-flag-restored",
            ),
            header,
            "removed symbol g_moe_expert_split_active is still referenced",
        )
    )
    cases.append(
        (
            "expert-split-header-restored",
            source,
            replace_once(
                header,
                "GGML_BACKEND_API void ggml_backend_sycl_set_unified_cache_budget_pct(int pct);",
                "GGML_BACKEND_API void ggml_backend_sycl_set_moe_expert_split(int n_expert, int n_expert_used);\n"
                "GGML_BACKEND_API void ggml_backend_sycl_set_unified_cache_budget_pct(int pct);",
                "expert-split-header-restored",
            ),
            "removed symbol ggml_backend_sycl_set_moe_expert_split is still referenced",
        )
    )

    # Positive: a tombstone comment and a string literal naming every removed
    # symbol, plus the forbidden key expression, must NOT trip anything. This is
    # the case that would fail if the checks were textual instead of AST-based --
    # documenting a check must not break it.
    tombstone = (
        "    // llama.cpp-x3ou removed g_moe_expert_split_active,\n"
        "    // ggml_backend_sycl_set_moe_expert_split and\n"
        "    // ggml_backend_sycl_query_moe_expert_split; the old key was\n"
        "    // g_sycl_canonical_checksums[std::string(tensor->name)].\n"
        '    (void) "g_moe_expert_split_active ggml_backend_sycl_query_moe_expert_split '
        'g_sycl_canonical_checksums[std::string(tensor->name)]";\n'
    )
    cases.append(
        (
            "positive-tombstone-comments-literals",
            replace_once(
                source,
                "static void ggml_sycl_reset_canonical_checksums() {\n",
                "static void ggml_sycl_reset_canonical_checksums() {\n" + tombstone,
                "positive-tombstone-comments-literals",
            ),
            header,
            None,
        )
    )

    # Positive: an extra bare scope around the erase is still unconditional.
    cases.append(
        (
            "positive-erase-nested-scope",
            replace_once(source, ERASE_BLOCK, "    {\n" + ERASE_BLOCK + "    }\n", "positive-erase-nested-scope"),
            header,
            None,
        )
    )

    return cases


def run_mutation_matrix(source: str, header: str) -> None:
    verify_contract(source, header)
    print("PASS baseline AST contract")

    clean_env = subprocess.run(
        [sys.executable, "-S", str(Path(__file__)), "--source", "unused", "--header", "unused"],
        capture_output=True,
        text=True,
        check=False,
    )
    dependency_message = (
        "SKIP: canonical-checksum-source-contract requires the test-only tree-sitter-language-pack module"
    )
    if clean_env.returncode != 77 or dependency_message not in clean_env.stderr:
        raise AssertionError(
            f"clean dependency environment returned {clean_env.returncode}: {clean_env.stdout}{clean_env.stderr}"
        )
    print("PASS clean dependency environment: exit 77 with explicit skip message")

    for name, mutated_source, mutated_header, expected in build_cases(source, header):
        try:
            verify_contract(mutated_source, mutated_header)
        except ContractError as error:
            if expected is None:
                raise AssertionError(f"positive mutation {name} unexpectedly failed: {error}") from error
            if expected not in str(error):
                raise AssertionError(f"mutation {name} failed with {error!s}; expected {expected!r}") from error
            print(f"PASS mutation {name}: {error}")
        else:
            if expected is not None:
                raise AssertionError(f"mutation {name} unexpectedly passed")
            print(f"PASS positive mutation {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--header", required=True, type=Path)
    parser.add_argument("--mutation-matrix", action="store_true")
    args = parser.parse_args()
    source = args.source.read_text(encoding="utf-8")
    header = args.header.read_text(encoding="utf-8")
    if args.mutation_matrix:
        run_mutation_matrix(source, header)
    else:
        verify_contract(source, header)
        print("PASS AST contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
