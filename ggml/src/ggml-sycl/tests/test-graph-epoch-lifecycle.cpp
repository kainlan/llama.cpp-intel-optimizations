#include "execution-lifecycle.hpp"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <memory>

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

static ModelToken root_token(uint64_t value) {
    return {
        ModelId{ value },
        LoadTxnId{ value + 100 },
        SlotToken{ 1, value + 200 }
    };
}

struct probe_terminal final : RetireTerminal {
    probe_terminal(Registry &              registry,
                   ContextId               owner,
                   std::atomic<unsigned> & wait_count,
                   std::atomic<unsigned> & destroy_count) :
        reg(registry),
        context(owner),
        waits(wait_count),
        destroys(destroy_count) {}

    ~probe_terminal() override {
        snapshot snap{};
        require(reg.extract(context, &snap) == error::OK, "terminal destroyed under registry lock");
        destroys.fetch_add(1, std::memory_order_relaxed);
    }

    Registry &              reg;
    ContextId               context;
    std::atomic<unsigned> & waits;
    std::atomic<unsigned> & destroys;

    void wait() noexcept override {
        // Re-entering the registry proves finish_retire does not hold its lock
        // while waiting on backend work.
        snapshot snap{};
        require(reg.extract(context, &snap) == error::OK, "terminal wait ran under registry lock");
        waits.fetch_add(1, std::memory_order_relaxed);
    }
};

int main() {
    Registry        reg;
    error           err     = error::OK;
    const ContextId context = reg.create_context(err);
    require(err == error::OK && reg.bind_backend(context, 0) == error::OK && reg.bind_backend(context, 1) == error::OK,
            "owner setup failed");
    SessionId         session{};
    SessionResetEpoch reset{};
    const ModelToken  root = root_token(7);
    require(reg.attach_root(context, root, &session, &reset) == error::OK, "attach failed");

    GraphEpoch first{};
    require(reg.begin_record(context, session, reset, root, &first) == error::OK, "begin record failed");
    require(reg.activate(context, session, reset, first, root) == error::OK, "activate failed");

    InvocationId invocation1{}, invocation2{};
    require(reg.begin_invocation(context, session, reset, first, root, &invocation1) == error::OK &&
                reg.finish_invocation(context, session, reset, first, invocation1, root) == error::OK &&
                reg.begin_invocation(context, session, reset, first, root, &invocation2) == error::OK &&
                invocation2.value != invocation1.value &&
                reg.finish_invocation(context, session, reset, first, invocation2, root) == error::OK,
            "one epoch did not survive two invocations");

    GraphEpoch rolled_back{};
    require(reg.begin_record(context, session, reset, root, &rolled_back) == error::OK &&
                reg.rollback_record(context, session, reset, rolled_back, root) == error::OK,
            "rollback before activation failed");
    epoch_snapshot epoch_snap{};
    require(reg.extract_epoch(context, session, reset, rolled_back, root, &epoch_snap) == error::STALE,
            "rolled-back record remained visible");

    GraphEpoch second{};
    require(reg.begin_record(context, session, reset, root, &second) == error::OK &&
                reg.activate(context, session, reset, second, root) == error::OK,
            "replacement activation failed");
    require(reg.extract_epoch(context, session, reset, first, root, &epoch_snap) == error::OK &&
                epoch_snap.state == epoch_phase::REPLACED && !epoch_snap.is_active,
            "replacement destroyed or retained old active epoch");

    const int    devices[] = { 0, 1 };
    RetireTicket old_ticket{};
    require(reg.begin_retire(context, session, reset, first, root, devices, 2, &old_ticket) == error::OK,
            "old epoch begin retire failed");
    std::atomic<unsigned> waits{ 0 };
    std::atomic<unsigned> destroys{ 0 };
    require(reg.attach_retire_terminal(old_ticket, 0,
                                       std::make_shared<probe_terminal>(reg, context, waits, destroys)) == error::OK,
            "first terminal attach failed");
    require(reg.finish_retire(old_ticket) == error::BUSY, "partial multi-device terminal set was released");
    require(reg.attach_retire_terminal(old_ticket, 1,
                                       std::make_shared<probe_terminal>(reg, context, waits, destroys)) == error::OK,
            "second terminal attach failed");

    RetireTicket wrong_context = old_ticket;
    wrong_context.context      = { context.value + 999 };
    require(reg.finish_retire(wrong_context) == error::STALE, "wrong-context retire accepted");
    RetireTicket wrong_root = old_ticket;
    wrong_root.token_root   = root_token(99);
    require(reg.finish_retire(wrong_root) == error::MISMATCH, "wrong-root retire accepted");
    require(reg.finish_retire(old_ticket) == error::OK && waits.load() == 2 && destroys.load() == 2,
            "exact retire did not wait for and release every terminal");
    require(reg.finish_retire(old_ticket) == error::STALE, "stale retire ticket mutated replacement state");
    require(reg.extract_epoch(context, session, reset, second, root, &epoch_snap) == error::OK &&
                epoch_snap.state == epoch_phase::ACTIVE && epoch_snap.is_active,
            "old completion mutated replacement epoch");

    RetireTicket second_ticket{};
    require(reg.begin_retire(context, session, reset, second, root, devices, 2, &second_ticket) == error::OK &&
                reg.attach_retire_terminal(
                    second_ticket, 0, std::make_shared<probe_terminal>(reg, context, waits, destroys)) == error::OK &&
                reg.attach_retire_terminal(
                    second_ticket, 1, std::make_shared<probe_terminal>(reg, context, waits, destroys)) == error::OK &&
                reg.finish_retire(second_ticket) == error::OK,
            "active epoch retirement failed");

    ResetTicket       reset_ticket{};
    SessionResetEpoch next_reset{};
    require(reg.begin_reset(context, session, reset, &reset_ticket) == error::OK &&
                reg.finish_reset(reset_ticket, &next_reset) == error::OK && next_reset.value == reset.value + 1,
            "owner reset after epoch release failed");

    Registry          graph_overflow(test_mutation::M6a_GRAPH_EPOCH_OVERFLOW);
    const ContextId   overflow_context = graph_overflow.create_context(err);
    SessionId         overflow_session{};
    SessionResetEpoch overflow_reset{};
    GraphEpoch        overflow_epoch{};
    require(err == error::OK &&
                graph_overflow.attach_root(overflow_context, root, &overflow_session, &overflow_reset) == error::OK &&
                graph_overflow.begin_record(overflow_context, overflow_session, overflow_reset, root,
                                            &overflow_epoch) == error::OVERFLOW,
            "graph epoch overflow mutation survived");

    Registry          invocation_overflow(test_mutation::M6e_INVOCATION_ID_OVERFLOW);
    const ContextId   invocation_context = invocation_overflow.create_context(err);
    SessionId         invocation_session{};
    SessionResetEpoch invocation_reset{};
    GraphEpoch        invocation_epoch{};
    InvocationId      overflow_invocation{};
    require(err == error::OK &&
                invocation_overflow.attach_root(invocation_context, root, &invocation_session, &invocation_reset) ==
                    error::OK &&
                invocation_overflow.begin_record(invocation_context, invocation_session, invocation_reset, root,
                                                 &invocation_epoch) == error::OK &&
                invocation_overflow.activate(invocation_context, invocation_session, invocation_reset, invocation_epoch,
                                             root) == error::OK &&
                invocation_overflow.begin_invocation(invocation_context, invocation_session, invocation_reset,
                                                     invocation_epoch, root, &overflow_invocation) == error::OVERFLOW,
            "invocation id overflow mutation survived");

    std::cout << "graph epoch lifecycle: ok\n";
    return 0;
}
