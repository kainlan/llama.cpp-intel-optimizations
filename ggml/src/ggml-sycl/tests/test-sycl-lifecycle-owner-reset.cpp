#include "execution-lifecycle.hpp"
#include "model-lifecycle.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace ggml_sycl::execution;
using ggml_sycl::lifecycle::LoadTxnId;
using ggml_sycl::lifecycle::ModelId;
using ggml_sycl::lifecycle::ModelToken;
using ggml_sycl::lifecycle::SlotToken;

namespace model = ggml_sycl::lifecycle;

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

static void m6() {
    error err = error::OK;
    auto  root = root_token(30);
    Registry m6a(test_mutation::M6a_GRAPH_EPOCH_OVERFLOW); const auto c6a = m6a.create_context(err); require(err == error::OK && m6a.bind_backend(c6a,0)==error::OK, "M6a setup failed"); SessionId s6a{}; SessionResetEpoch e6a{}; GraphEpoch g6a{}; require(m6a.attach_root(c6a, root, &s6a, &e6a) == error::OK && m6a.begin_graph(c6a, s6a, e6a, root, &g6a) == error::OVERFLOW, "M6a failed");
    Registry m6b(test_mutation::M6b_DRAIN_SERIAL_OVERFLOW); const auto c6b = m6b.create_context(err); require(err == error::OK && m6b.bind_backend(c6b,0)==error::OK, "M6b setup failed"); SessionId s6b{}; SessionResetEpoch e6b{}; require(m6b.attach_root(c6b, root, &s6b, &e6b) == error::OK, "M6b attach failed"); DrainTicket d6b{}; require(m6b.begin_drain(c6b, &d6b) == error::OVERFLOW, "M6b failed");
    Registry m6c(test_mutation::M6c_RESET_SERIAL_OVERFLOW); const auto c6c = m6c.create_context(err); require(err == error::OK && m6c.bind_backend(c6c,0)==error::OK, "M6c setup failed"); SessionId s6c{}; SessionResetEpoch e6c{}; require(m6c.attach_root(c6c, root, &s6c, &e6c) == error::OK, "M6c attach failed"); ResetTicket r6c{}; require(m6c.begin_reset(c6c, s6c, e6c, &r6c) == error::OVERFLOW, "M6c failed");
    Registry m6e(test_mutation::M6e_INVOCATION_ID_OVERFLOW); const auto c6e = m6e.create_context(err); require(err == error::OK && m6e.bind_backend(c6e,0)==error::OK, "M6e setup failed"); SessionId s6e{}; SessionResetEpoch e6e{}; GraphEpoch g6e{}; InvocationId i6e{}; const int d[] = {0}; const int p6e[] = {29}; require(m6e.attach_root(c6e, root, &s6e, &e6e) == error::OK && m6e.begin_graph(c6e, s6e, e6e, root, &g6e) == error::OK && m6e.begin_invocation(c6e, s6e, e6e, g6e, root, d, 1, p6e, 1, 29, &i6e) == error::OVERFLOW, "M6e failed");
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
    require(reg.begin_drain_extract(dt) == error::OK, "H13 begin extract failed");
    require(reg.begin_drain_extract(dt) == error::BUSY, "H13 duplicate extract allowed");
    reg.cancel_drain_extract(dt);
    require(reg.begin_drain_extract(dt) == error::OK, "H13 extract retry failed");
    require(reg.note_drain_extracted_control_host_allocs(&dt, 7) == error::OK && dt.extracted_control_host_allocs == 7, "H13 drain extract failed");
    require(reg.finish_drain(dt) == error::OK, "H13 finish drain failed");

    Registry m4(test_mutation::M4_CONTEXT_ID_OVERFLOW); m4.create_context(err); require(err == error::OVERFLOW, "H13 M4 failed");
    Registry m5(test_mutation::M5_SESSION_ID_OVERFLOW); const auto c5 = m5.create_context(err); require(err == error::OK && m5.bind_backend(c5,0)==error::OK, "H13 M5 setup failed"); SessionId s5{}; SessionResetEpoch e5{}; require(m5.attach_root(c5, root, &s5, &e5) == error::OVERFLOW, "H13 M5 failed");
    m6();
    Registry dupreg; error dup_err = error::OK; const auto dupctx = dupreg.create_context(dup_err); require(dup_err == error::OK && dupreg.bind_backend(dupctx,0)==error::OK, "H13 dup setup create failed"); SessionId ds{}; SessionResetEpoch de{}; GraphEpoch gd{}; InvocationId id{}; const int d[] = {0,0}; const int dup_p[] = {3,3}; const int dummy[] = {0}; const int p6e[] = {29}; require(dupreg.attach_root(dupctx, root, &ds, &de) == error::OK && dupreg.begin_graph(dupctx, ds, de, root, &gd) == error::OK, "H13 dup setup failed"); require(dupreg.begin_invocation(dupctx, ds, de, gd, root, d, 2, p6e, 1, 29, &id) == error::MISMATCH, "H13 duplicate devices accepted"); require(dupreg.begin_invocation(dupctx, ds, de, gd, root, d, 1, dup_p, 2, 3, &id) == error::MISMATCH, "H13 duplicate participants accepted"); require(dupreg.begin_invocation(dupctx, ds, de, gd, root, dummy, 0, p6e, 1, 29, &id) == error::MISMATCH, "H13 empty devices accepted"); require(dupreg.begin_invocation(dupctx, ds, de, gd, root, d, 1, dummy, 0, 29, &id) == error::MISMATCH, "H13 empty participants accepted"); require(dupreg.begin_invocation(dupctx, ds, de, gd, root, d, 1, p6e, 1, 7, &id) == error::MISMATCH, "H13 missing caller participant accepted"); require(dupreg.begin_invocation(dupctx, ds, de, gd, root, d, 1, p6e, 1, 29, &id) == error::OK, "H13 valid begin after failed validation did not recover");
}

static void g2_multi_live() {
    Registry reg; error err = error::OK; const auto a = reg.create_context(err), b = reg.create_context(err); require(err == error::OK, "G2 create failed");
    require(reg.bind_backend(a, 2) == error::OK && reg.bind_backend(b, 2) == error::OK, "G2 bind failed");
    snapshot snap{}; require(reg.extract(a, &snap) == error::OK && snap.bound_device_count == 1, "G2 extract A failed");
    require(reg.extract(b, &snap) == error::OK && snap.bound_device_count == 1, "G2 extract B failed");
    DrainTicket dt{}; require(reg.begin_drain(a, &dt) == error::OK && reg.finish_drain(dt) == error::OK, "G2 drain A failed");
    require(reg.extract(b, &snap) == error::OK && snap.bound_device_count == 1, "G2 B damaged by A close");
}

// A published model owner, for the model half of the H14 identity matrix.
static ModelToken commit_model(model::Registry & reg) {
    auto begin = reg.begin_outer();
    require(begin.code == model::error::OK, "H14 model load begin failed");
    auto end = reg.end(begin.txn, true);
    require(end.committed, "H14 model load did not commit");
    return end.token;
}

// A context's whole observable owner state. Teardown of a different owner must
// leave every field of this untouched.
struct owner_state {
    uint64_t context = 0, session = 0, reset_epoch = 0, graph_epoch = 0, invocation = 0;
    context_phase context_state = context_phase::CLOSED;
    session_phase session_state = session_phase::IDLE;
    graph_phase   graph_state   = graph_phase::IDLE;
    ModelToken    token_root{};
    uint32_t      bound_device_count = 0;
};

static owner_state read_owner(Registry & reg, ContextId context, const char * message) {
    snapshot snap{};
    require(reg.extract(context, &snap) == error::OK, message);
    return { snap.context.value,  snap.session.value,     snap.reset_epoch.value,
             snap.graph_epoch.value, snap.invocation.value, snap.context_state,
             snap.session_state,  snap.graph_state,       snap.token_root,
             snap.bound_device_count };
}

static bool same_owner(const owner_state & a, const owner_state & b) {
    return a.context == b.context && a.session == b.session && a.reset_epoch == b.reset_epoch &&
           a.graph_epoch == b.graph_epoch && a.invocation == b.invocation &&
           a.context_state == b.context_state && a.session_state == b.session_state &&
           a.graph_state == b.graph_state && a.token_root == b.token_root &&
           a.bound_device_count == b.bound_device_count;
}

// H5: reset and teardown touch only the target model/context/session/epoch.
// The wrong-owner assertions are the host form of M4 (wildcard context owner)
// and M5 (all-epoch graph clear): with either mutation applied the mismatched
// call below would succeed and B's snapshot would move.
static void h5_owner_targeted_reset() {
    Registry reg;
    error    err = error::OK;
    const auto a = reg.create_context(err), b = reg.create_context(err);
    require(err == error::OK, "H5 create failed");
    // Same device for both contexts: a device-scoped teardown cannot pass this.
    require(reg.bind_backend(a, 5) == error::OK && reg.bind_backend(b, 5) == error::OK, "H5 bind failed");

    SessionId sa{}, sb{};
    SessionResetEpoch ea{}, eb{};
    auto ra = root_token(40), rb = root_token(41);
    require(reg.attach_root(a, ra, &sa, &ea) == error::OK && reg.attach_root(b, rb, &sb, &eb) == error::OK,
            "H5 attach failed");

    GraphEpoch gb{};
    InvocationId ib{};
    const int devices[] = { 5 };
    const int participants[] = { 2 };
    require(reg.begin_graph(b, sb, eb, rb, &gb) == error::OK, "H5 B graph failed");
    require(reg.begin_invocation(b, sb, eb, gb, rb, devices, 1, participants, 1, 2, &ib) == error::OK,
            "H5 B invoke failed");
    require(reg.complete_invocation(b, sb, eb, gb, ib, rb, 2) == error::OK, "H5 B complete failed");
    require(reg.retire_graph(b, sb, eb, gb, rb) == error::OK, "H5 B retire failed");

    const owner_state before = read_owner(reg, b, "H5 B extract failed");

    // M4 shape: A's reset ticket presented with B's session must be refused.
    ResetTicket wrong{};
    require(reg.begin_reset(a, sb, eb, &wrong) == error::STALE, "H5 cross-context session reset accepted");
    // M5 shape: retiring B's epoch through A must be refused.
    require(reg.retire_graph(a, sa, ea, gb, rb) == error::STALE, "H5 cross-context graph retire accepted");
    require(same_owner(read_owner(reg, b, "H5 B extract after refusals failed"), before),
            "H5 refused cross-owner call still changed B");

    // A's own reset, then A's own full drain, must both leave B alone.
    ResetTicket mine{};
    SessionResetEpoch next{};
    require(reg.begin_reset(a, sa, ea, &mine) == error::OK, "H5 A begin reset failed");
    require(reg.finish_reset(mine, &next) == error::OK && next.value == ea.value + 1, "H5 A finish reset failed");
    require(same_owner(read_owner(reg, b, "H5 B extract after A reset failed"), before), "H5 A reset changed B");

    DrainTicket drain{};
    require(reg.begin_drain(a, &drain) == error::OK, "H5 A begin drain failed");
    require(reg.finish_drain(drain) == error::OK, "H5 A finish drain failed");
    require(same_owner(read_owner(reg, b, "H5 B extract after A drain failed"), before), "H5 A drain changed B");

    // B is not merely intact in the snapshot -- it is still usable on the very
    // device A was torn down on.
    GraphEpoch gb2{};
    InvocationId ib2{};
    require(reg.begin_graph(b, sb, eb, rb, &gb2) == error::OK, "H5 B unusable after A teardown");
    require(reg.begin_invocation(b, sb, eb, gb2, rb, devices, 1, participants, 1, 2, &ib2) == error::OK,
            "H5 B device claim lost to A teardown");
}

// Shared H14 fixtures. Each --case below asserts only its own named scenario;
// the fixtures exist so that nine focused cases do not become nine copies of
// the same setup.

// Two published models. `survivor` must be untouched by whatever a case does
// to `doomed`.
struct model_fixture {
    model::Registry reg;
    ModelToken      doomed{};
    ModelToken      survivor{};
    std::shared_ptr<const model::ModelState> survivor_state;

    model_fixture() {
        doomed         = commit_model(reg);
        survivor       = commit_model(reg);
        survivor_state = reg.find(survivor.model);
        require(survivor_state != nullptr, "H14 survivor model not live");
    }

    void require_survivor_unchanged(const char * message) {
        require(reg.find(survivor.model) == survivor_state, message);
    }
};

// Two contexts on one device. A device-scoped teardown cannot pass these:
// `live` must be untouched by whatever a case does to `doomed`.
struct context_fixture {
    Registry          reg;
    ContextId         live{}, doomed{};
    SessionId         s_live{}, s_doomed{};
    SessionResetEpoch e_live{}, e_doomed{};
    ModelToken        r_live    = root_token(50);
    ModelToken        r_doomed  = root_token(51);
    GraphEpoch        g_live{};
    owner_state       before{};

    context_fixture() {
        error err = error::OK;
        live   = reg.create_context(err);
        doomed = reg.create_context(err);
        require(err == error::OK, "H14 context create failed");
        require(reg.bind_backend(live, 6) == error::OK && reg.bind_backend(doomed, 6) == error::OK,
                "H14 bind failed");
        require(reg.attach_root(live, r_live, &s_live, &e_live) == error::OK, "H14 live attach failed");
        require(reg.attach_root(doomed, r_doomed, &s_doomed, &e_doomed) == error::OK, "H14 doomed attach failed");
        require(reg.begin_graph(live, s_live, e_live, r_live, &g_live) == error::OK, "H14 live graph failed");
        before = read_owner(reg, live, "H14 live extract failed");
    }

    void require_live_unchanged(const char * message) {
        require(same_owner(read_owner(reg, live, "H14 live re-extract failed"), before), message);
    }

    // Drives `doomed` all the way to CLOSED so the stale-identity cases have a
    // previously-issued identity to present, as distinct from a never-issued one.
    DrainTicket close_doomed() {
        DrainTicket ticket{};
        require(reg.begin_drain(doomed, &ticket) == error::OK, "H14 doomed begin drain failed");
        require(reg.finish_drain(ticket) == error::OK, "H14 doomed finish drain failed");
        return ticket;
    }
};

// Repeat teardown of the same DEAD identity is an idempotent typed no-op.
static void h14_teardown_repeat() {
    model_fixture f;
    require(f.reg.teardown(f.doomed) == model::error::OK, "H14 first teardown failed");
    require(f.reg.teardown(f.doomed) == model::error::OK_ALREADY_DEAD, "H14 repeat teardown is not OK_ALREADY_DEAD");
    require(f.reg.teardown(f.doomed) == model::error::OK_ALREADY_DEAD, "H14 repeat teardown is not idempotent");
    f.require_survivor_unchanged("H14 repeat teardown changed the surviving model");
}

// A model identity this registry never issued is NOT_FOUND, not a sweep.
static void h14_teardown_unknown_model() {
    model_fixture    f;
    const ModelToken unknown{ ModelId{ f.doomed.model.value + 9999 }, LoadTxnId{ f.doomed.load.value + 9999 },
                              SlotToken{ 0, 1 } };
    require(f.reg.teardown(unknown) == model::error::NOT_FOUND, "H14 unknown model is not NOT_FOUND");
    f.require_survivor_unchanged("H14 unknown-model teardown changed the surviving model");
    require(f.reg.teardown(f.doomed) == model::error::OK, "H14 real teardown blocked by the unknown attempt");
}

// A live model's slot with the wrong generation is STALE_IDENTITY, and the
// model stays live and tearable through its real token.
//
// prepare_teardown() guards this input twice -- the slot generation check and
// the token-equality check below it -- so deleting either one alone is a null
// mutation that no test can catch. Deleting both does fail this case, and only
// this case. Assert the observable result, not one of the two source lines.
static void h14_teardown_stale_slot_generation() {
    model_fixture f;
    ModelToken    stale = f.survivor;
    stale.owner.generation = f.survivor.owner.generation + 1;
    require(f.reg.teardown(stale) == model::error::STALE_IDENTITY, "H14 stale slot generation is not STALE_IDENTITY");
    f.require_survivor_unchanged("H14 stale-generation teardown changed the surviving model");
    require(f.reg.teardown(f.survivor) == model::error::OK, "H14 exact token rejected after a stale attempt");
}

// A ContextId beyond anything issued cannot start or close a drain.
static void h14_teardown_unknown_context() {
    context_fixture f;
    const ContextId never_issued{ f.doomed.value + 4096 };
    DrainTicket     ticket{};
    require(f.reg.begin_drain(never_issued, &ticket) == error::STALE, "H14 never-issued context drain accepted");
    require(f.reg.close_context_if_idle(never_issued) == error::STALE, "H14 never-issued context close accepted");
    f.require_live_unchanged("H14 unknown context still mutated the live context");
}

// A SessionId this registry never issued cannot open a reset on a real context.
static void h14_teardown_never_issued_session() {
    context_fixture f;
    ResetTicket     ticket{};
    require(f.reg.begin_reset(f.live, SessionId{ f.s_live.value + 4096 }, f.e_live, &ticket) == error::STALE,
            "H14 never-issued session accepted");
    f.require_live_unchanged("H14 never-issued session still mutated the live context");
}

// A GraphEpoch this registry never issued cannot retire the current graph.
static void h14_teardown_never_issued_graph_epoch() {
    context_fixture f;
    require(f.reg.retire_graph(f.live, f.s_live, f.e_live, GraphEpoch{ f.g_live.value + 4096 }, f.r_live) ==
                error::STALE,
            "H14 never-issued graph epoch accepted");
    f.require_live_unchanged("H14 never-issued graph epoch still mutated the live context");
}

// A previously-issued ContextId that has already been drained to CLOSED, plus a
// replay of its own spent drain ticket.
static void h14_teardown_stale_context() {
    context_fixture   f;
    const DrainTicket spent = f.close_doomed();
    DrainTicket       ticket{};
    require(f.reg.begin_drain(f.doomed, &ticket) == error::STALE, "H14 stale context drain accepted");
    require(f.reg.finish_drain(spent) == error::STALE, "H14 spent drain ticket replayed");
    require(f.reg.close_context_if_idle(f.doomed) == error::STALE, "H14 stale context close accepted");
    f.require_live_unchanged("H14 draining the doomed context swept the live one");
}

// A previously-issued SessionId belonging to a different context, presented to
// this one. Distinct from the never-issued case above: this value really was
// handed out, just not for this owner.
static void h14_teardown_stale_session() {
    context_fixture f;
    ResetTicket     ticket{};
    require(f.reg.begin_reset(f.live, f.s_doomed, f.e_doomed, &ticket) == error::STALE,
            "H14 another context's session accepted");
    f.close_doomed();
    require(f.reg.begin_reset(f.live, f.s_doomed, f.e_doomed, &ticket) == error::STALE,
            "H14 closed context's session accepted");
    f.require_live_unchanged("H14 stale session still mutated the live context");
}

// A superseded reset epoch, and a graph epoch that has already been retired --
// both previously issued, both stale.
static void h14_teardown_stale_graph_epoch() {
    context_fixture f;
    ResetTicket     ticket{};
    require(f.reg.begin_reset(f.live, f.s_live, SessionResetEpoch{ f.e_live.value - 1 }, &ticket) == error::STALE,
            "H14 superseded reset epoch accepted");

    require(f.reg.rollback_graph(f.live, f.s_live, f.e_live, f.g_live, f.r_live) == error::OK,
            "H14 live graph rollback failed");
    GraphEpoch next{};
    require(f.reg.begin_graph(f.live, f.s_live, f.e_live, f.r_live, &next) == error::OK, "H14 live re-record failed");
    require(next.value != f.g_live.value, "H14 re-record reused the retired epoch");
    require(f.reg.retire_graph(f.live, f.s_live, f.e_live, f.g_live, f.r_live) == error::STALE,
            "H14 superseded graph epoch accepted");

    snapshot after{};
    require(f.reg.extract(f.live, &after) == error::OK && after.graph_epoch.value == next.value,
            "H14 stale epoch retire clobbered the current epoch");
}

int main(int argc, char ** argv) {
    std::vector<std::string> cases;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--case") == 0 && i + 1 < argc) {
            cases.emplace_back(argv[++i]);
        }
    }
    if (cases.empty()) {
        cases = { "all" };
    }
    for (const auto & which : cases) {
        if (which == "H6") h6();
        else if (which == "H7") h7();
        else if (which == "H11") h11();
        else if (which == "H13") h13();
        else if (which == "M6") m6();
        else if (which == "G2") g2_multi_live();
        else if (which == "H5" || which == "owner-targeted-reset") h5_owner_targeted_reset();
        else if (which == "teardown-repeat") h14_teardown_repeat();
        else if (which == "teardown-unknown-model") h14_teardown_unknown_model();
        else if (which == "teardown-stale-slot-generation") h14_teardown_stale_slot_generation();
        else if (which == "teardown-unknown-context") h14_teardown_unknown_context();
        else if (which == "teardown-never-issued-session") h14_teardown_never_issued_session();
        else if (which == "teardown-never-issued-graph-epoch") h14_teardown_never_issued_graph_epoch();
        else if (which == "teardown-stale-context") h14_teardown_stale_context();
        else if (which == "teardown-stale-session") h14_teardown_stale_session();
        else if (which == "teardown-stale-graph-epoch") h14_teardown_stale_graph_epoch();
        else if (which == "all") {
            h6(); h7(); h11(); h13(); m6(); g2_multi_live(); h5_owner_targeted_reset();
            h14_teardown_repeat(); h14_teardown_unknown_model(); h14_teardown_stale_slot_generation();
            h14_teardown_unknown_context(); h14_teardown_never_issued_session();
            h14_teardown_never_issued_graph_epoch(); h14_teardown_stale_context();
            h14_teardown_stale_session(); h14_teardown_stale_graph_epoch();
        } else {
            std::cerr << "unknown case: " << which << '\n';
            return 2;
        }
    }
    return 0;
}
