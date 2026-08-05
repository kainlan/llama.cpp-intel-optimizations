#include "execution-lifecycle.hpp"

#include <cstdlib>
#include <cstring>
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
    return { ModelId{ base }, LoadTxnId{ base + 1000 }, SlotToken{ static_cast<uint32_t>(base % 11), base + 2000 } };
}

static void h6() {
    Registry reg;
    error err = error::OK;
    const auto a = reg.create_context(err), b = reg.create_context(err);
    require(err == error::OK, "H6 create failed");
    require(reg.bind_backend(a, 0) == error::OK && reg.bind_backend(a, 1) == error::OK && reg.bind_backend(b, 1) == error::OK, "H6 bind failed");
    SessionId sa{}, sb{}; SessionResetEpoch ea{}, eb{};
    auto ra = root_token(1), rb = root_token(2);
    require(reg.attach_root(a, ra, &sa, &ea) == error::OK && reg.attach_root(b, rb, &sb, &eb) == error::OK, "H6 attach failed");
    GraphEpoch ga{}, gb{}; InvocationId ia{}, ib{}; const int da[] = {0,1}; const int db[] = {1}; const int pa[] = {4}; const int pb[] = {9};
    require(reg.begin_graph(a, sa, ea, ra, &ga) == error::OK && reg.begin_graph(b, sb, eb, rb, &gb) == error::OK, "H6 graph failed");
    require(reg.begin_invocation(a, sa, ea, ga, ra, da, 2, pa, 1, 4, &ia) == error::OK, "H6 invoke A failed");
    require(reg.begin_invocation(b, sb, eb, gb, rb, db, 1, pb, 1, 9, &ib) == error::DEVICE_BUSY, "H6 missing DEVICE_BUSY");
    require(reg.complete_invocation(a, sa, ea, ga, ia, ra, 4) == error::OK, "H6 participant complete failed");
    require(reg.begin_invocation(b, sb, eb, gb, rb, db, 1, pb, 1, 9, &ib) == error::OK, "H6 exact owner release failed");
}

static void h7() {
    Registry reg; error err = error::OK; const auto ctx = reg.create_context(err); require(err == error::OK, "H7 create failed");
    require(reg.bind_backend(ctx, 0) == error::OK, "H7 bind failed");
    SessionId s{}; SessionResetEpoch e{}; auto root = root_token(10); require(reg.attach_root(ctx, root, &s, &e) == error::OK, "H7 attach failed");
    GraphEpoch g{}; InvocationId i{}; const int d[] = {0}; const int p[] = {13};
    require(reg.begin_graph(ctx, s, e, root, &g) == error::OK && reg.begin_invocation(ctx, s, e, g, root, d, 1, p, 1, 13, &i) == error::OK, "H7 invoke failed");
    require(reg.quarantine_invocation(ctx, s, e, g, i, root, 13) == error::OK, "H7 quarantine failed");
    DrainTicket dt{}; ResetTicket rt{};
    require(reg.begin_drain(ctx, &dt) == error::BUSY && reg.begin_reset(ctx, s, e, &rt) == error::BUSY, "H7 terminal graph not blocking");
    require(reg.begin_graph(ctx, s, e, root, &g) == error::BUSY, "H7 terminal overwritten");
    require(reg.retire_graph(ctx, s, e, g, root) == error::OK, "H7 retire failed");
    require(reg.retire_graph(ctx, s, e, g, root) == error::STALE, "H7 stale retire proof failed");
}

static void h11() {
    Registry reg; error err = error::OK; const auto ctx = reg.create_context(err); require(err == error::OK, "H11 create failed");
    require(reg.bind_backend(ctx, 0) == error::OK, "H11 bind failed");
    SessionId s{}; SessionResetEpoch e{}; auto root = root_token(20); require(reg.attach_root(ctx, root, &s, &e) == error::OK, "H11 attach failed");
    GraphEpoch g{}; InvocationId i{}; const int d[] = {0}; const int p[] = {21}; snapshot snap{};
    require(reg.begin_graph(ctx, s, e, root, &g) == error::OK && reg.begin_invocation(ctx, s, e, g, root, d, 1, p, 1, 21, &i) == error::OK, "H11 invoke failed");
    require(reg.extract(ctx, &snap) == error::OK && snap.token_root_state == token_root_phase::OPEN, "H11 open failed");
    require(reg.seal_invocation(ctx, s, e, g, i, root) == error::OK, "H11 seal failed");
    require(reg.extract(ctx, &snap) == error::OK && snap.token_root_state == token_root_phase::SEALED, "H11 sealed failed");
    require(reg.complete_invocation(ctx, s, e, g, i, root, 21) == error::OK, "H11 complete failed");
    require(reg.extract(ctx, &snap) == error::OK && snap.token_root_state == token_root_phase::COMPLETE, "H11 complete state failed");
    require(reg.retire_graph(ctx, s, e, g, root) == error::OK, "H11 retire complete failed");
    require(reg.begin_graph(ctx, s, e, root, &g) == error::OK && reg.begin_invocation(ctx, s, e, g, root, d, 1, p, 1, 21, &i) == error::OK, "H11 invoke2 failed");
    require(reg.quarantine_invocation(ctx, s, e, g, i, root, 21) == error::OK, "H11 quarantine failed");
    require(reg.extract(ctx, &snap) == error::OK && snap.token_root_state == token_root_phase::QUARANTINED, "H11 quarantine state failed");
}

static void h13() {
    Registry reg; error err = error::OK; const auto ctx = reg.create_context(err); require(err == error::OK, "H13 create failed");
    require(reg.bind_backend(ctx, 3) == error::OK, "H13 bind failed");
    SessionId s{}; SessionResetEpoch e{}; auto root = root_token(30); require(reg.attach_root(ctx, root, &s, &e) == error::OK, "H13 attach failed");
    snapshot snap{}; require(reg.extract(ctx, &snap) == error::OK && snap.bound_device_count == 1, "H13 extract failed");
    ResetTicket rt{}; require(reg.begin_reset(ctx, s, e, &rt) == error::OK, "H13 begin reset failed");
    require(reg.close_context_if_idle(ctx) == error::BUSY, "H13 close reset should block");
    require(reg.bind_backend(ctx, 4) == error::BUSY, "H13 bind during reset should block");
    require(reg.extract(ctx, &snap) == error::OK && snap.context_state == context_phase::RESETTING, "H13 resetting invisible");
    SessionResetEpoch next{}; require(reg.finish_reset(rt, &next) == error::OK && next.value == e.value + 1, "H13 finish reset failed");
    DrainTicket dt{}; require(reg.begin_drain(ctx, &dt) == error::OK, "H13 begin drain failed");
    require(reg.close_context_if_idle(ctx) == error::BUSY, "H13 close drain should block");
    require(reg.bind_backend(ctx, 4) == error::BUSY, "H13 bind during drain should block");
    require(reg.extract(ctx, &snap) == error::OK && snap.context_state == context_phase::DRAINING, "H13 draining invisible");
    require(reg.note_drain_extracted_control_host_allocs(&dt, 7) == error::OK && dt.extracted_control_host_allocs == 7, "H13 drain extract failed");
    require(reg.finish_drain(dt) == error::OK, "H13 finish drain failed");

    Registry m4(test_mutation::M4_CONTEXT_ID_OVERFLOW); m4.create_context(err); require(err == error::OVERFLOW, "H13 M4 failed");
    Registry m5(test_mutation::M5_SESSION_ID_OVERFLOW); const auto c5 = m5.create_context(err); require(err == error::OK && m5.bind_backend(c5,0)==error::OK, "H13 M5 setup failed"); SessionId s5{}; SessionResetEpoch e5{}; require(m5.attach_root(c5, root, &s5, &e5) == error::OVERFLOW, "H13 M5 failed");
    Registry m6a(test_mutation::M6a_GRAPH_EPOCH_OVERFLOW); const auto c6a = m6a.create_context(err); require(err == error::OK && m6a.bind_backend(c6a,0)==error::OK, "H13 M6a setup failed"); SessionId s6a{}; SessionResetEpoch e6a{}; GraphEpoch g6a{}; require(m6a.attach_root(c6a, root, &s6a, &e6a) == error::OK && m6a.begin_graph(c6a, s6a, e6a, root, &g6a) == error::OVERFLOW, "H13 M6a failed");
    Registry m6b(test_mutation::M6b_DRAIN_SERIAL_OVERFLOW); const auto c6b = m6b.create_context(err); require(err == error::OK && m6b.bind_backend(c6b,0)==error::OK, "H13 M6b setup failed"); SessionId s6b{}; SessionResetEpoch e6b{}; require(m6b.attach_root(c6b, root, &s6b, &e6b) == error::OK, "H13 M6b attach failed"); DrainTicket d6b{}; require(m6b.begin_drain(c6b, &d6b) == error::OVERFLOW, "H13 M6b failed");
    Registry m6c(test_mutation::M6c_RESET_SERIAL_OVERFLOW); const auto c6c = m6c.create_context(err); require(err == error::OK && m6c.bind_backend(c6c,0)==error::OK, "H13 M6c setup failed"); SessionId s6c{}; SessionResetEpoch e6c{}; require(m6c.attach_root(c6c, root, &s6c, &e6c) == error::OK, "H13 M6c attach failed"); ResetTicket r6c{}; require(m6c.begin_reset(c6c, s6c, e6c, &r6c) == error::OVERFLOW, "H13 M6c failed");
    Registry m6e(test_mutation::M6e_INVOCATION_ID_OVERFLOW); const auto c6e = m6e.create_context(err); require(err == error::OK && m6e.bind_backend(c6e,0)==error::OK, "H13 M6e setup failed"); SessionId s6e{}; SessionResetEpoch e6e{}; GraphEpoch g6e{}; InvocationId i6e{}; const int d[] = {0}; const int p6e[] = {29}; require(m6e.attach_root(c6e, root, &s6e, &e6e) == error::OK && m6e.begin_graph(c6e, s6e, e6e, root, &g6e) == error::OK && m6e.begin_invocation(c6e, s6e, e6e, g6e, root, d, 1, p6e, 1, 29, &i6e) == error::OVERFLOW, "H13 M6e failed");
    Registry dupreg; error dup_err = error::OK; const auto dupctx = dupreg.create_context(dup_err); require(dup_err == error::OK && dupreg.bind_backend(dupctx,0)==error::OK, "H13 dup setup create failed"); SessionId ds{}; SessionResetEpoch de{}; GraphEpoch gd{}; InvocationId id{}; const int dup_d[] = {0,0}; const int dup_p[] = {3,3}; const int dummy[] = {0}; require(dupreg.attach_root(dupctx, root, &ds, &de) == error::OK && dupreg.begin_graph(dupctx, ds, de, root, &gd) == error::OK, "H13 dup setup failed"); require(dupreg.begin_invocation(dupctx, ds, de, gd, root, dup_d, 2, p6e, 1, 29, &id) == error::MISMATCH, "H13 duplicate devices accepted"); require(dupreg.begin_invocation(dupctx, ds, de, gd, root, d, 1, dup_p, 2, 3, &id) == error::MISMATCH, "H13 duplicate participants accepted"); require(dupreg.begin_invocation(dupctx, ds, de, gd, root, dummy, 0, p6e, 1, 29, &id) == error::MISMATCH, "H13 empty devices accepted"); require(dupreg.begin_invocation(dupctx, ds, de, gd, root, d, 1, dummy, 0, 29, &id) == error::MISMATCH, "H13 empty participants accepted"); require(dupreg.begin_invocation(dupctx, ds, de, gd, root, d, 1, p6e, 1, 7, &id) == error::MISMATCH, "H13 missing caller participant accepted"); require(dupreg.begin_invocation(dupctx, ds, de, gd, root, d, 1, p6e, 1, 29, &id) == error::OK, "H13 valid begin after failed validation did not recover");
}

static void g2_multi_live() {
    Registry reg; error err = error::OK; const auto a = reg.create_context(err), b = reg.create_context(err); require(err == error::OK, "G2 create failed");
    require(reg.bind_backend(a, 2) == error::OK && reg.bind_backend(b, 2) == error::OK, "G2 bind failed");
    snapshot snap{}; require(reg.extract(a, &snap) == error::OK && snap.bound_device_count == 1, "G2 extract A failed");
    require(reg.extract(b, &snap) == error::OK && snap.bound_device_count == 1, "G2 extract B failed");
    DrainTicket dt{}; require(reg.begin_drain(a, &dt) == error::OK && reg.finish_drain(dt) == error::OK, "G2 drain A failed");
    require(reg.extract(b, &snap) == error::OK && snap.bound_device_count == 1, "G2 B damaged by A close");
}

int main(int argc, char ** argv) {
    const char * which = argc > 2 && std::strcmp(argv[1], "--case") == 0 ? argv[2] : "all";
    if (std::strcmp(which, "H6") == 0) h6();
    else if (std::strcmp(which, "H7") == 0) h7();
    else if (std::strcmp(which, "H11") == 0) h11();
    else if (std::strcmp(which, "H13") == 0) h13();
    else if (std::strcmp(which, "G2") == 0) g2_multi_live();
    else { h6(); h7(); h11(); h13(); g2_multi_live(); }
    return 0;
}
