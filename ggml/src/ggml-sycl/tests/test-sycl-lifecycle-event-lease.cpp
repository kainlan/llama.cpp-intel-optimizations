#include "execution-lifecycle.hpp"

#include <cstdlib>
#include <iostream>

using namespace ggml_sycl::execution;
using ggml_sycl::lifecycle::LoadTxnId;
using ggml_sycl::lifecycle::ModelId;
using ggml_sycl::lifecycle::ModelToken;
using ggml_sycl::lifecycle::SlotToken;

static void require(bool value, const char * message) {
    if (!value) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

static ModelToken root_token(uint64_t base) {
    return { ModelId{ base }, LoadTxnId{ base + 1000 }, SlotToken{ static_cast<uint32_t>(base % 13), base + 2000 } };
}

static void g5a_multidevice_busy_aggregate() {
    Registry reg;
    error err = error::OK;
    const auto ctx = reg.create_context(err);
    require(err == error::OK, "G5a create failed");
    require(reg.bind_backend(ctx, 0) == error::OK && reg.bind_backend(ctx, 1) == error::OK,
            "G5a bind failed");
    SessionId session{};
    SessionResetEpoch reset{};
    const auto root = root_token(50);
    require(reg.attach_root(ctx, root, &session, &reset) == error::OK, "G5a attach failed");
    GraphEpoch graph{};
    require(reg.begin_graph(ctx, session, reset, root, &graph) == error::OK, "G5a graph failed");
    InvocationId invocation{};
    const int devices[] = { 0, 1 };
    require(reg.begin_invocation(ctx, session, reset, graph, root, devices, 2, &invocation) == error::OK,
            "G5a invocation failed");
    snapshot snap{};
    require(reg.extract(ctx, &snap) == error::OK && snap.busy_device_count == 2,
            "G5a aggregate busy count failed");
    require(reg.seal_invocation(ctx, session, reset, graph, invocation, root) == error::OK, "G5a seal failed");
    require(reg.extract(ctx, &snap) == error::OK && snap.graph_state == graph_phase::SEALED && snap.busy_device_count == 2,
            "G5a sealed busy snapshot failed");
}

static void g6_terminal_event_semantics() {
    Registry reg;
    error err = error::OK;
    const auto ctx = reg.create_context(err);
    require(err == error::OK, "G6 create failed");
    require(reg.bind_backend(ctx, 0) == error::OK, "G6 bind failed");
    SessionId session{};
    SessionResetEpoch reset{};
    const auto root = root_token(60);
    require(reg.attach_root(ctx, root, &session, &reset) == error::OK, "G6 attach failed");
    GraphEpoch graph{};
    require(reg.begin_graph(ctx, session, reset, root, &graph) == error::OK, "G6 graph failed");
    InvocationId invocation{};
    const int devices[] = { 0 };
    require(reg.begin_invocation(ctx, session, reset, graph, root, devices, 1, &invocation) == error::OK,
            "G6 invocation failed");
    require(reg.seal_invocation(ctx, session, reset, graph, invocation, root) == error::OK, "G6 seal failed");
    snapshot snap{};
    require(reg.extract(ctx, &snap) == error::OK && snap.graph_state == graph_phase::SEALED && snap.busy_device_count == 1,
            "G6 sealed event snapshot failed");
    require(reg.complete_invocation(ctx, session, reset, graph, invocation, root) == error::OK,
            "G6 complete failed");
    require(reg.extract(ctx, &snap) == error::OK && snap.graph_state == graph_phase::COMPLETE && snap.busy_device_count == 0,
            "G6 complete snapshot failed");
    require(reg.begin_graph(ctx, session, reset, root, &graph) == error::BUSY,
            "G6 terminal graph overwritten before retire");
    require(reg.retire_graph(ctx, session, reset, graph, root) == error::OK, "G6 retire failed");
    require(reg.begin_graph(ctx, session, reset, root, &graph) == error::OK, "G6 graph after retire failed");
}

static void g7_same_device_multiple_contexts() {
    Registry reg;
    error err = error::OK;
    const auto a = reg.create_context(err);
    require(err == error::OK, "G7 create A failed");
    const auto b = reg.create_context(err);
    require(err == error::OK, "G7 create B failed");
    require(reg.bind_backend(a, 2) == error::OK && reg.bind_backend(b, 2) == error::OK,
            "G7 bind failed");
    SessionId sa{};
    SessionResetEpoch ea{};
    const auto root_a = root_token(70);
    require(reg.attach_root(a, root_a, &sa, &ea) == error::OK, "G7 attach A failed");
    GraphEpoch ga{};
    require(reg.begin_graph(a, sa, ea, root_a, &ga) == error::OK, "G7 graph A failed");
    InvocationId ia{};
    const int devices[] = { 2 };
    require(reg.begin_invocation(a, sa, ea, ga, root_a, devices, 1, &ia) == error::OK, "G7 invoke A failed");

    SessionId sb{};
    SessionResetEpoch eb{};
    const auto root_b = root_token(71);
    require(reg.attach_root(b, root_b, &sb, &eb) == error::OK, "G7 attach B failed");
    GraphEpoch gb{};
    require(reg.begin_graph(b, sb, eb, root_b, &gb) == error::OK, "G7 graph B failed");
    InvocationId ib{};
    require(reg.begin_invocation(b, sb, eb, gb, root_b, devices, 1, &ib) == error::DEVICE_BUSY,
            "G7 missing same-device busy protection");
    require(reg.complete_invocation(a, sa, ea, ga, ia, root_a) == error::OK, "G7 complete A failed");
    require(reg.begin_invocation(b, sb, eb, gb, root_b, devices, 1, &ib) == error::OK,
            "G7 independent contexts did not coexist after release");
}

int main() {
    g5a_multidevice_busy_aggregate();
    g6_terminal_event_semantics();
    g7_same_device_multiple_contexts();
    return 0;
}
