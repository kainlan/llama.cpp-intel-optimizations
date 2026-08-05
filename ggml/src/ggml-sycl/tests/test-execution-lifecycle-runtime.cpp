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
    return { ModelId{ base }, LoadTxnId{ base + 1000 }, SlotToken{ static_cast<uint32_t>(base % 7), base + 2000 } };
}

static void h6_device_busy_exact_ownership() {
    Registry reg;
    error err = error::OK;
    const auto a = reg.create_context(err);
    require(err == error::OK, "H6 create A failed");
    const auto b = reg.create_context(err);
    require(err == error::OK, "H6 create B failed");
    require(reg.bind_backend(a, 0) == error::OK && reg.bind_backend(a, 1) == error::OK, "H6 bind A failed");
    require(reg.bind_backend(b, 1) == error::OK, "H6 bind B failed");

    SessionId sa{};
    SessionResetEpoch ea{};
    const auto root_a = root_token(1);
    require(reg.attach_root(a, root_a, &sa, &ea) == error::OK, "H6 attach A failed");
    GraphEpoch ga{};
    require(reg.begin_graph(a, sa, ea, root_a, &ga) == error::OK, "H6 graph A failed");
    InvocationId ia{};
    const int devices_a[] = { 0, 1 };
    require(reg.begin_invocation(a, sa, ea, ga, root_a, devices_a, 2, &ia) == error::OK, "H6 invoke A failed");

    SessionId sb{};
    SessionResetEpoch eb{};
    const auto root_b = root_token(2);
    require(reg.attach_root(b, root_b, &sb, &eb) == error::OK, "H6 attach B failed");
    GraphEpoch gb{};
    require(reg.begin_graph(b, sb, eb, root_b, &gb) == error::OK, "H6 graph B failed");
    InvocationId ib{};
    const int device_b[] = { 1 };
    require(reg.begin_invocation(b, sb, eb, gb, root_b, device_b, 1, &ib) == error::DEVICE_BUSY,
            "H6 missing DEVICE_BUSY");

    require(reg.seal_invocation(a, sa, ea, ga, ia, root_a) == error::OK, "H6 seal A failed");
    require(reg.complete_invocation(a, sa, ea, ga, ia, root_a) == error::OK, "H6 complete A failed");
    require(reg.begin_invocation(b, sb, eb, gb, root_b, device_b, 1, &ib) == error::OK,
            "H6 device owner not released exactly");
}

static void h7_exact_graph_retirement() {
    Registry reg;
    error err = error::OK;
    const auto ctx = reg.create_context(err);
    require(err == error::OK, "H7 create failed");
    require(reg.bind_backend(ctx, 0) == error::OK, "H7 bind failed");
    SessionId session{};
    SessionResetEpoch reset{};
    const auto root = root_token(10);
    require(reg.attach_root(ctx, root, &session, &reset) == error::OK, "H7 attach failed");
    GraphEpoch graph{};
    require(reg.begin_graph(ctx, session, reset, root, &graph) == error::OK, "H7 graph failed");
    InvocationId invocation{};
    const int devices[] = { 0 };
    require(reg.begin_invocation(ctx, session, reset, graph, root, devices, 1, &invocation) == error::OK,
            "H7 invocation failed");
    require(reg.retire_graph(ctx, session, reset, graph) == error::BUSY, "H7 retired active graph");
    require(reg.quarantine_invocation(ctx, session, reset, graph, invocation, root) == error::OK,
            "H7 quarantine failed");
    require(reg.retire_graph(ctx, session, reset, graph) == error::OK, "H7 retire failed");
    require(reg.retire_graph(ctx, session, reset, graph) == error::STALE, "H7 retired graph twice");
    require(reg.begin_graph(ctx, session, reset, root, &graph) == error::OK, "H7 second graph failed");
    require(reg.retire_graph(ctx, session, { reset.value + 1 }, graph) == error::STALE, "H7 stale reset mismatch lost");
}

static void h11_token_root_retention() {
    Registry reg;
    error err = error::OK;
    const auto ctx = reg.create_context(err);
    require(err == error::OK, "H11 create failed");
    require(reg.bind_backend(ctx, 0) == error::OK, "H11 bind failed");
    SessionId session{};
    SessionResetEpoch reset{};
    const auto root = root_token(20);
    require(reg.attach_root(ctx, root, &session, &reset) == error::OK, "H11 attach failed");
    GraphEpoch graph{};
    require(reg.begin_graph(ctx, session, reset, root, &graph) == error::OK, "H11 graph failed");
    InvocationId invocation{};
    const int devices[] = { 0 };
    require(reg.begin_invocation(ctx, session, reset, graph, root, devices, 1, &invocation) == error::OK,
            "H11 invocation failed");
    snapshot snap{};
    require(reg.extract(ctx, &snap) == error::OK && snap.token_root == root &&
                snap.token_root_state == token_root_phase::OPEN,
            "H11 OPEN retention failed");
    require(reg.seal_invocation(ctx, session, reset, graph, invocation, root) == error::OK, "H11 seal failed");
    require(reg.extract(ctx, &snap) == error::OK && snap.token_root_state == token_root_phase::SEALED,
            "H11 SEALED retention failed");
    require(reg.complete_invocation(ctx, session, reset, graph, invocation, root) == error::OK, "H11 complete failed");
    require(reg.extract(ctx, &snap) == error::OK && snap.token_root_state == token_root_phase::COMPLETE,
            "H11 COMPLETE retention failed");
    GraphEpoch graph2{};
    require(reg.begin_graph(ctx, session, reset, root, &graph2) == error::OK, "H11 graph2 failed");
    InvocationId invocation2{};
    require(reg.begin_invocation(ctx, session, reset, graph2, root, devices, 1, &invocation2) == error::OK,
            "H11 invocation2 failed");
    require(reg.quarantine_invocation(ctx, session, reset, graph2, invocation2, root) == error::OK,
            "H11 quarantine failed");
    require(reg.extract(ctx, &snap) == error::OK && snap.token_root_state == token_root_phase::QUARANTINED,
            "H11 QUARANTINED retention failed");
}

static void h13_reset_extract_and_mutations() {
    Registry reg;
    error err = error::OK;
    const auto ctx = reg.create_context(err);
    require(err == error::OK, "H13 create failed");
    require(reg.bind_backend(ctx, 3) == error::OK, "H13 bind failed");
    SessionId session{};
    SessionResetEpoch reset{};
    const auto root = root_token(30);
    require(reg.attach_root(ctx, root, &session, &reset) == error::OK, "H13 attach failed");
    snapshot snap{};
    require(reg.extract(ctx, &snap) == error::OK && snap.context == ctx && snap.session == session &&
                snap.reset_epoch == reset && snap.bound_device_count == 1,
            "H13 extract failed");
    require(reg.drain_context(ctx) == error::OK, "H13 drain scaffold failed");
    SessionResetEpoch next_reset{};
    require(reg.reset_session(ctx, session, reset, &next_reset) == error::OK && next_reset.value == reset.value + 1,
            "H13 reset scaffold failed");

    Registry m4(test_mutation::M4_CONTEXT_ID_OVERFLOW);
    m4.create_context(err);
    require(err == error::OVERFLOW, "H13 M4 hook failed");
    Registry m5(test_mutation::M5_SESSION_ID_OVERFLOW);
    const auto ctx5 = m5.create_context(err);
    require(err == error::OK, "H13 M5 create failed");
    require(m5.bind_backend(ctx5, 0) == error::OK, "H13 M5 bind failed");
    SessionId s5{};
    SessionResetEpoch e5{};
    require(m5.attach_root(ctx5, root, &s5, &e5) == error::OVERFLOW, "H13 M5 hook failed");
    Registry m6a(test_mutation::M6a_GRAPH_EPOCH_OVERFLOW);
    const auto ctx6a = m6a.create_context(err);
    require(err == error::OK && m6a.bind_backend(ctx6a, 0) == error::OK, "H13 M6a setup failed");
    SessionId s6a{};
    SessionResetEpoch e6a{};
    require(m6a.attach_root(ctx6a, root, &s6a, &e6a) == error::OK, "H13 M6a attach failed");
    GraphEpoch g6a{};
    require(m6a.begin_graph(ctx6a, s6a, e6a, root, &g6a) == error::OVERFLOW, "H13 M6a hook failed");
    Registry m6e(test_mutation::M6e_INVOCATION_ID_OVERFLOW);
    const auto ctx6e = m6e.create_context(err);
    require(err == error::OK && m6e.bind_backend(ctx6e, 0) == error::OK, "H13 M6e setup failed");
    SessionId s6e{};
    SessionResetEpoch e6e{};
    require(m6e.attach_root(ctx6e, root, &s6e, &e6e) == error::OK, "H13 M6e attach failed");
    GraphEpoch g6e{};
    require(m6e.begin_graph(ctx6e, s6e, e6e, root, &g6e) == error::OK, "H13 M6e graph failed");
    InvocationId i6e{};
    const int device[] = { 0 };
    require(m6e.begin_invocation(ctx6e, s6e, e6e, g6e, root, device, 1, &i6e) == error::OVERFLOW,
            "H13 M6e hook failed");
}

int main() {
    h6_device_busy_exact_ownership();
    h7_exact_graph_retirement();
    h11_token_root_retention();
    h13_reset_extract_and_mutations();
    return 0;
}
