#!/usr/bin/env python3
"""Structural regression gate for exception-safe async MoE stage handoffs."""

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


def verify(source: str) -> None:
    # Ticket field precedes vector construction, and capacity/slots exist before
    # submit. This closes both the graph-drain race and post-submit vector-growth
    # exception window.
    handoff_pos = source.index("struct ggml_sycl_direct_stage_owner_handoff")
    helper_pos = source.index("static void ggml_sycl_retain_direct_stage_owners", handoff_pos)
    handoff = source[handoff_pos:helper_pos]
    require(handoff, "publish_ticket = ggml_sycl::begin_retained_handle_publish()",
            "direct-stage handoff does not acquire a retained publication ticket")
    assert handoff.index("publish_ticket") < handoff.index("owners;"), (
        "publication ticket must be constructed before owner-vector allocation"
    )
    require(handoff, "owners.reserve(2)", "owner handoff does not reserve before submission")
    require(handoff, "owners.emplace_back()", "destination owner slot is not preconstructed")

    helper = function_body(source, "ggml_sycl_retain_direct_stage_owners(")
    require(helper,
            "retain_handles_until_event(handoff.owners, result.event, std::move(handoff.publish_ticket))",
            "successful stage does not use the ticketed retained publication overload")
    require(helper, "catch (...) {\n            ggml_sycl_drain_direct_stage_queue(queue);",
            "publication allocation/worker-start failure does not drain the exact queue")
    # Passing a copy keeps handoff.owners alive if the by-value publication API
    # destroys its argument while throwing.
    assert "retain_handles_until_event(std::move(handoff.owners)" not in helper

    drain = function_body(source, "ggml_sycl_drain_direct_stage_queue(")
    require(drain, "queue.wait_and_throw()", "failure path does not synchronously drain its exact queue")

    prestage = function_body(source, "moe_prestage_popular_experts(")
    require(prestage, "ggml_sycl_direct_stage_owner_handoff handoff(item.source.handle)",
            "secondary GPU reorder does not open a retained handoff before submit")
    require(prestage, "ggml_sycl_direct_stage_owner_handoff handoff(ci.reorder_handle)",
            "CPU reorder fallback does not retain its staging source")
    require(prestage, "&storage_handle", "secondary staging does not request a destination handle")
    require(prestage, "handoff.set_destination(storage_handle)",
            "secondary staging does not install the destination owner")
    cpu = prestage[prestage.index("// DMA pre-reordered data to secondary devices"):]
    require(cpu, "ggml_sycl_drain_direct_stage_queue(*sec_bcs)",
            "CPU failure-after-enqueue still relies on the staged counter to drain")

    ensure = function_body(source, "moe_expert_ensure_soa_cached(")
    require(ensure, "ggml_sycl_direct_stage_owner_handoff handoff(source.handle)",
            "async SOA materialization omits the retained handoff")
    require(ensure, "handoff.set_destination(storage_handle)",
            "async SOA materialization omits its destination owner")

    materialize = function_body(source, "ggml_sycl_materialize_planned_expert_layout(")
    require(materialize,
            "ggml_sycl_direct_stage_owner_handoff handoff(stage_from_soa ? source_layout_handle : source.handle)",
            "XMX materialization does not retain its exact selected source")
    require(materialize, "ggml_sycl_drain_direct_stage_queue(*q)",
            "XMX submission throw does not drain before owner unwind")

    hybrid_init = function_body(source, "static void moe_hybrid_init_once(")
    assert hybrid_init.count("ggml_sycl_direct_stage_owner_handoff handoff(d2h_staging)") == 2, (
        "planner and greedy materialization paths must both open retained handoffs"
    )
    assert hybrid_init.count("&reorder_ctx, phase2_bcs, &storage_handle") == 2, (
        "planner and greedy materialization paths must both request destination owners"
    )

    xmx_fill = function_body(source, "ggml_sycl_fill_xmx_tiled(")
    assert xmx_fill.count("copy_event.wait_and_throw();") >= 3, (
        "ownerless std::vector XMX H2D fallback can return before DMA completion"
    )


def test_async_stage_handoff_source_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    verify((root / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    verify(Path(args.source).read_text())
    print("PASS: async stage owners publish with tickets or drain their exact queue")


if __name__ == "__main__":
    main()
