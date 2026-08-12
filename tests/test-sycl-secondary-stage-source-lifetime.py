#!/usr/bin/env python3
"""Structural regression gate for async MoE secondary/materialization owners."""

from pathlib import Path
import argparse


def function_body(source: str, name: str) -> str:
    start = 0
    while True:
        start = source.index(name, start)
        brace = source.index("{", start)
        semicolon = source.find(";", start, brace)
        if semicolon < 0:
            break
        start += len(name)
    depth = 0
    for pos in range(brace, len(source)):
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
            if depth == 0:
                return source[brace : pos + 1]
    raise AssertionError(f"unterminated {name}")


def require(haystack: str, needle: str, message: str) -> None:
    if needle not in haystack:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    source = Path(args.source).read_text()

    helper = function_body(source, "ggml_sycl_retain_direct_stage_owners(")
    require(helper, "retain_handles_until_event(std::move(owners), result.event)",
            "successful stage does not retain its owner bundle to the terminal event")
    require(helper, "queue.wait_and_throw()",
            "failed/unknown stage does not drain before dropping local owners")

    prestage = function_body(source, "moe_prestage_popular_experts(")
    require(prestage, "{ item.source.handle, storage_handle }, result",
            "secondary GPU/preallocated-temp stage omits source or destination owner")
    hybrid_init = function_body(source, "static void moe_hybrid_init_once(")
    require(hybrid_init, "{ d2h_staging }, stage_result, *phase2_bcs",
            "secondary planner/greedy staging omits the local D2H owner")

    ensure = function_body(source, "moe_expert_ensure_soa_cached(")
    require(ensure, "{ source.handle, storage_handle }, result",
            "async SOA materialization omits source or destination owner")

    materialize = function_body(source, "ggml_sycl_materialize_planned_expert_layout(")
    require(materialize, "stage_from_soa ? source_layout_handle : source.handle, storage_handle",
            "XMX materialization does not retain its exact selected source and destination")
    require(materialize, "q->wait_and_throw()",
            "XMX throw path does not drain before local source unwind")

    xmx_fill = function_body(source, "ggml_sycl_fill_xmx_tiled(")
    assert xmx_fill.count("copy_event.wait_and_throw();") >= 3, (
        "ownerless std::vector XMX H2D fallback can return before DMA completion"
    )

    print("PASS: async secondary/XMX source owners reach terminal events or a proven wait")


if __name__ == "__main__":
    main()
