#!/usr/bin/env python3
"""AST-backed MoE TLS reset contract and executable mutation matrix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Iterable

try:
    from tree_sitter_language_pack import get_parser
except ImportError:
    print(
        "SKIP: sycl-moe-reset-source-contract requires the test-only tree-sitter-language-pack module",
        file=sys.stderr,
    )
    raise SystemExit(77)

RESET = "ggml_sycl_moe_layer_ids_cache_new_graph"
CACHE = "g_moe_layer_ids_cache"


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


@dataclass
class Ast:
    source: str

    def __post_init__(self) -> None:
        self.encoded = self.source.encode("utf-8")
        self.root = get_parser("cpp").parse(self.source).root_node()

    def text(self, node) -> str:
        return self.encoded[node.start_byte() : node.end_byte()].decode("utf-8")

    def functions(self, name: str) -> list:
        result = []
        for node in walk(self.root):
            if kind(node) != "function_definition":
                continue
            declarator = node.child_by_field_name("declarator")
            if declarator and any(kind(n) == "identifier" and self.text(n) == name for n in walk(declarator)):
                result.append(node)
        return result

    def function(self, name: str):
        matches = self.functions(name)
        if len(matches) != 1:
            raise ContractError(f"expected one AST function {name}, found {len(matches)}")
        return matches[0]

    def calls(self, node, name: str) -> list:
        result = []
        for candidate in walk(node):
            if kind(candidate) != "call_expression":
                continue
            function = candidate.child_by_field_name("function")
            if function and self.text(function) == name:
                result.append(candidate)
        return result

    def identifiers(self, node, name: str) -> list:
        return [candidate for candidate in walk(node) if kind(candidate) == "identifier" and self.text(candidate) == name]


def unconditional_statements(compound) -> list:
    """Flatten only standalone compound statements; never enter control nodes."""
    result = []
    for child in children(compound):
        if kind(child) == "compound_statement":
            result.extend(unconditional_statements(child))
        else:
            result.append(child)
    return result


def statement_containing_call(ast: Ast, statements: list, call_name: str) -> list:
    return [
        statement
        for statement in statements
        if kind(statement) == "expression_statement" and ast.calls(statement, call_name)
    ]


def verify_reset_dominates(ast: Ast, body, path: str, targets: tuple[str, ...]) -> None:
    all_resets = ast.calls(body, RESET)
    if len(all_resets) != 1:
        raise ContractError(f"{path}: expected exactly one active reset call, found {len(all_resets)}")

    statements = unconditional_statements(body)
    reset_statements = statement_containing_call(ast, statements, RESET)
    if len(reset_statements) != 1:
        raise ContractError(f"{path}: reset is not on the unconditional branch path")
    reset_statement = reset_statements[0]
    reset_index = statements.index(reset_statement)

    terminators = {"return_statement", "co_return_statement", "throw_statement"}
    if any(kind(statement) in terminators for statement in statements[:reset_index]):
        raise ContractError(f"{path}: reset is unreachable after an unconditional terminator")

    labels = {}
    for label in (node for node in walk(body) if kind(node) == "labeled_statement"):
        name = label.child_by_field_name("label")
        if name:
            labels[ast.text(name)] = label
    for jump in (node for node in walk(body) if kind(node) == "goto_statement"):
        if jump.start_byte() >= reset_statement.start_byte():
            continue  # Existing post-reset dispatch-local gotos are legitimate.
        target_ids = [node for node in walk(jump) if kind(node) == "statement_identifier"]
        target = labels.get(ast.text(target_ids[0])) if len(target_ids) == 1 else None
        if target is None or any(kind(node) == "ERROR" for node in walk(jump)):
            raise ContractError(f"{path}: unresolved or indirect goto before reset")
        if target.start_byte() > reset_statement.end_byte():
            raise ContractError(f"{path}: forward goto can enter the post-reset region without reset")

    cache_accesses = ast.identifiers(body, CACHE)
    reset_cache_ids = ast.identifiers(all_resets[0], CACHE)
    if len(reset_cache_ids) != 1 or not cache_accesses or cache_accesses[0].start_byte() != reset_cache_ids[0].start_byte():
        raise ContractError(f"{path}: cache is accessed before reset")

    for target in targets:
        calls = ast.calls(body, target)
        if not calls:
            raise ContractError(f"{path}: missing required dispatch {target}")
        if any(call.start_byte() < reset_statement.end_byte() for call in calls):
            raise ContractError(f"{path}: {target} can execute before reset")


def find_segmented_body(ast: Ast):
    # The surrounding 90k-line TU contains conditional SYCL extensions that
    # tree-sitter intentionally recovers as ERROR nodes. Anchor the branch by
    # its parsed if-condition rather than relying on the outer function node;
    # the condition is unique and the consequence remains a proper AST subtree.
    matches = []
    for node in walk(ast.root):
        if kind(node) != "if_statement":
            continue
        condition = node.child_by_field_name("condition")
        consequence = node.child_by_field_name("consequence")
        if condition and consequence and "block_graphlet_executed" in ast.text(condition):
            if (kind(consequence) == "compound_statement" and ast.calls(consequence, "moe_graph_record_segments") and
                    ast.calls(consequence, "moe_graph_replay_segments")):
                matches.append(consequence)
    if len(matches) != 1:
        raise ContractError(f"segmented replay: expected one AST branch, found {len(matches)}")
    return matches[0]


def verify_tls_declaration(ast: Ast) -> None:
    declarations = []
    for node in walk(ast.root):
        if kind(node) != "declaration":
            continue
        declarator = node.child_by_field_name("declarator")
        while declarator and declarator.child_by_field_name("declarator"):
            declarator = declarator.child_by_field_name("declarator")
        if declarator and ast.text(declarator) == CACHE:
            declarations.append(node)
    if len(declarations) != 1:
        raise ContractError(f"production cache: expected one declaration, found {len(declarations)}")
    storage = {ast.text(node) for node in walk(declarations[0]) if kind(node) == "storage_class_specifier"}
    if not {"static", "thread_local"}.issubset(storage):
        raise ContractError("production cache: declaration is not static thread_local")


def verify_helper(header_source: str) -> None:
    ast = Ast(header_source)
    helper = ast.function(RESET)
    body = helper.child_by_field_name("body")
    loops = [node for node in children(body) if kind(node) == "for_range_loop"]
    entry_resets = ast.calls(body, "ggml_sycl_moe_layer_ids_cache_reset_entry")
    valid = len(loops) == 1 and len(entry_resets) == 1
    if valid:
        loop = loops[0]
        range_expression = loop.child_by_field_name("right")
        declarator = loop.child_by_field_name("declarator")
        loop_body = loop.child_by_field_name("body")
        binding_names = [node for node in walk(declarator) if kind(node) == "identifier"] if declarator else []
        body_statements = children(loop_body) if loop_body and kind(loop_body) == "compound_statement" else []
        direct_statement = body_statements[0] if len(body_statements) == 1 else None
        statement_expressions = children(direct_statement) if direct_statement else []
        direct_calls = ast.calls(direct_statement, "ggml_sycl_moe_layer_ids_cache_reset_entry") if direct_statement else []
        arguments = entry_resets[0].child_by_field_name("arguments")
        argument_nodes = children(arguments) if arguments else []
        valid = (
            range_expression is not None
            and ast.text(range_expression) == "cache"
            and declarator is not None
            and kind(declarator) == "reference_declarator"
            and sum(kind(node) == "structured_binding_declarator" for node in walk(declarator)) == 1
            and [ast.text(node) for node in binding_names] == ["_", "entry"]
            and direct_statement is not None
            and kind(direct_statement) == "expression_statement"
            and len(statement_expressions) == 1
            and kind(statement_expressions[0]) == "call_expression"
            and len(direct_calls) == 1
            and statement_expressions[0].start_byte() == entry_resets[0].start_byte()
            and statement_expressions[0].end_byte() == entry_resets[0].end_byte()
            and len(argument_nodes) == 1
            and kind(argument_nodes[0]) == "identifier"
            and ast.text(argument_nodes[0]) == "entry"
        )
    if not valid:
        raise ContractError(
            "reset helper: direct range(cache) body must reset exactly structured-binding entry"
        )


def verify_contract(source: str, header: str) -> None:
    ast = Ast(source)
    verify_tls_declaration(ast)

    normal = ast.function("ggml_backend_sycl_graph_compute_impl")
    verify_reset_dominates(ast, normal.child_by_field_name("body"), "normal", ("ggml_sycl_compute_forward",))

    block = ast.function("moe_graph_try_block_graphlets")
    verify_reset_dominates(
        ast,
        block.child_by_field_name("body"),
        "block graphlet",
        ("moe_graph_record_block_graphs", "moe_graph_replay_block_graphs"),
    )

    segmented = find_segmented_body(ast)
    verify_reset_dominates(
        ast,
        segmented,
        "segmented replay",
        ("moe_graph_record_segments", "moe_graph_replay_segments"),
    )
    verify_helper(header)


def replace_nth(text: str, needle: str, replacement: str, occurrence: int) -> str:
    start = -1
    for _ in range(occurrence + 1):
        start = text.find(needle, start + 1)
        if start < 0:
            raise RuntimeError(f"mutation anchor not found: {needle} occurrence {occurrence}")
    return text[:start] + replacement + text[start + len(needle) :]


def insert_dispatch_before_reset(source: str, reset_occurrence: int, dispatch: str) -> str:
    reset_line = "    ggml_sycl_moe_layer_ids_cache_new_graph(g_moe_layer_ids_cache);\n"
    return replace_nth(source, reset_line, f"    {dispatch}();\n" + reset_line, reset_occurrence)


def run_mutation_matrix(source: str, header: str) -> None:
    reset_line = "    ggml_sycl_moe_layer_ids_cache_new_graph(g_moe_layer_ids_cache);\n"
    cases: list[tuple[str, str, str, str | None]] = []
    paths = (("normal", 0, "return;"), ("block", 1, "return false;"), ("segmented", 2, "return GGML_STATUS_SUCCESS;"))
    for path, occurrence, terminator in paths:
        contract_path = {"normal": "normal", "block": "block graphlet", "segmented": "segmented replay"}[path]
        cases.append((f"{path}-commented-reset", replace_nth(source, reset_line, "    // " + reset_line.lstrip(), occurrence), header,
                      f"{contract_path}: expected exactly one active reset call"))
        wrapped = "    if constexpr (false) {\n" + reset_line + "    }\n"
        cases.append((f"{path}-if-constexpr-false", replace_nth(source, reset_line, wrapped, occurrence), header,
                      f"{contract_path}: reset is not on the unconditional branch path"))
        nested = "    {\n        " + terminator + "\n" + reset_line + "    }\n"
        cases.append((f"{path}-nested-unreachable", replace_nth(source, reset_line, nested, occurrence), header,
                      f"{contract_path}: reset is unreachable after an unconditional terminator"))
        label = f"h5m4_skip_{path}"
        goto_skip = f"    goto {label};\n" + reset_line + f"{label}:\n    ;\n"
        cases.append((f"{path}-goto-skip", replace_nth(source, reset_line, goto_skip, occurrence), header,
                      f"{contract_path}: forward goto can enter the post-reset region without reset"))
        computed_label = f"h5m4_computed_{path}"
        computed_goto = (
            f"    void * h5m4_target_{path} = &&{computed_label};\n"
            f"    goto *h5m4_target_{path};\n"
            + reset_line
            + f"{computed_label}:\n    ;\n"
        )
        cases.append((f"{path}-computed-goto", replace_nth(source, reset_line, computed_goto, occurrence), header,
                      f"{contract_path}: unresolved or indirect goto before reset"))

    unresolved = "    goto h5m4_missing_target;\n" + reset_line
    cases.append(("normal-unresolved-goto", replace_nth(source, reset_line, unresolved, 0), header,
                  "normal: unresolved or indirect goto before reset"))

    cases.extend(
        [
            ("block-record-before-reset", insert_dispatch_before_reset(source, 1, "moe_graph_record_block_graphs"), header,
             "block graphlet: moe_graph_record_block_graphs can execute before reset"),
            ("block-replay-before-reset", insert_dispatch_before_reset(source, 1, "moe_graph_replay_block_graphs"), header,
             "block graphlet: moe_graph_replay_block_graphs can execute before reset"),
            ("segmented-record-before-reset", insert_dispatch_before_reset(source, 2, "moe_graph_record_segments"), header,
             "segmented replay: moe_graph_record_segments can execute before reset"),
            ("segmented-replay-before-reset", insert_dispatch_before_reset(source, 2, "moe_graph_replay_segments"), header,
             "segmented replay: moe_graph_replay_segments can execute before reset"),
        ]
    )

    shared = source.replace(
        "static thread_local moe_layer_ids_cache g_moe_layer_ids_cache;",
        "static moe_layer_ids_cache g_moe_layer_ids_cache; // static thread_local moe_layer_ids_cache g_moe_layer_ids_cache;",
        1,
    )
    cases.append(("shared-cache-tls-comment-spoof", shared, header, "production cache: declaration is not static thread_local"))

    loop_start = "    for (auto & [_, entry] : cache) {\n"
    loop_body = "        ggml_sycl_moe_layer_ids_cache_reset_entry(entry);\n    }\n"
    cleared_header = header.replace(loop_start + loop_body, "    cache.clear();\n", 1)
    helper_contract_error = "reset helper: direct range(cache) body must reset exactly structured-binding entry"
    cases.append(("capacity-map-clear", source, cleared_header, helper_contract_error))

    begin_only_header = header.replace(
        loop_start + loop_body,
        "    for (auto & [_, entry] : cache) {\n"
        "        (void) _;\n"
        "        (void) entry;\n"
        "    }\n"
        "    if (!cache.empty()) {\n"
        "        ggml_sycl_moe_layer_ids_cache_reset_entry(cache.begin()->second);\n"
        "    }\n",
        1,
    )
    cases.append(("empty-loop-begin-only-reset", source, begin_only_header, helper_contract_error))

    conditional_entry_header = header.replace(
        loop_body,
        "        if (false) {\n"
        "            ggml_sycl_moe_layer_ids_cache_reset_entry(entry);\n"
        "        }\n"
        "    }\n",
        1,
    )
    cases.append(("helper-conditional-entry-reset", source, conditional_entry_header, helper_contract_error))

    nested_entry_header = header.replace(
        loop_body,
        "        {\n"
        "            ggml_sycl_moe_layer_ids_cache_reset_entry(entry);\n"
        "        }\n"
        "    }\n",
        1,
    )
    cases.append(("helper-nested-entry-reset", source, nested_entry_header, helper_contract_error))

    zero_trip_header = header.replace(
        "for (auto & [_, entry] : cache)",
        "for (auto & [_, entry] : std::array<moe_layer_ids_cache_entry, 0>{})",
        1,
    )
    cases.append(("helper-zero-trip-range", source, zero_trip_header, helper_contract_error))

    begin_argument_header = header.replace(
        "ggml_sycl_moe_layer_ids_cache_reset_entry(entry);",
        "ggml_sycl_moe_layer_ids_cache_reset_entry(cache.begin()->second);",
        1,
    )
    cases.append(("helper-cache-begin-argument", source, begin_argument_header, helper_contract_error))

    positive_comments = source
    for occurrence in reversed(range(3)):
        positive_comments = replace_nth(
            positive_comments,
            reset_line,
            "    // g_moe_layer_ids_cache and fake reset/TLS text must be ignored\n"
            "    (void) \"g_moe_layer_ids_cache ggml_sycl_moe_layer_ids_cache_new_graph static thread_local\";\n"
            + reset_line,
            occurrence,
        )
    cases.append(("positive-cache-name-comments-literals", positive_comments, header, None))

    positive_scopes = source
    for occurrence in reversed(range(3)):
        positive_scopes = replace_nth(positive_scopes, reset_line, "    {\n" + reset_line + "    }\n", occurrence)
    cases.append(("positive-unconditional-scopes", positive_scopes, header, None))

    verify_contract(source, header)
    print("PASS baseline AST/control-flow contract")

    clean_env = subprocess.run(
        [sys.executable, "-S", str(Path(__file__)), "--source", "unused", "--header", "unused"],
        capture_output=True,
        text=True,
        check=False,
    )
    dependency_message = "SKIP: sycl-moe-reset-source-contract requires the test-only tree-sitter-language-pack module"
    if clean_env.returncode != 77 or dependency_message not in clean_env.stderr:
        raise AssertionError(
            f"clean dependency environment returned {clean_env.returncode}: {clean_env.stdout}{clean_env.stderr}"
        )
    print("PASS clean dependency environment: exit 77 with explicit skip message")

    for name, mutated_source, mutated_header, expected in cases:
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
        print("PASS AST/control-flow contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
