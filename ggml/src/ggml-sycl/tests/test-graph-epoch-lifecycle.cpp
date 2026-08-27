#include "execution-lifecycle.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

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

struct delayed_terminal final : RetireTerminal {
    delayed_terminal(Registry &          registry,
                     ContextId           owner,
                     std::atomic<bool> & wait_started,
                     std::atomic<bool> & release_wait) :
        reg(registry),
        context(owner),
        started(wait_started),
        release(release_wait) {}

    void wait() noexcept override {
        snapshot snap{};
        require(reg.extract(context, &snap) == error::OK, "delayed terminal wait ran under registry lock");
        started.store(true, std::memory_order_release);
        while (!release.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    }

    Registry &          reg;
    ContextId           context;
    std::atomic<bool> & started;
    std::atomic<bool> & release;
};

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
    // Direct-decode authority is minted only for the exact live outer
    // invocation and pins its release boundary until explicitly finished.
    {
        Registry          pin_registry;
        error             pin_error   = error::OK;
        const auto        pin_context = pin_registry.create_context(pin_error);
        const auto        pin_root    = root_token(700);
        SessionId         pin_session{};
        SessionResetEpoch pin_reset{};
        GraphEpoch        pin_graph{};
        InvocationId      pin_invocation{};
        const int         pin_device[] = { 0 };
        require(pin_error == error::OK && pin_registry.bind_backend(pin_context, 0) == error::OK &&
                    pin_registry.attach_root(pin_context, pin_root, &pin_session, &pin_reset) == error::OK &&
                    pin_registry.begin_graph(pin_context, pin_session, pin_reset, pin_root, &pin_graph) == error::OK &&
                    pin_registry.begin_invocation(pin_context, pin_session, pin_reset, pin_graph, pin_root, pin_device,
                                                  1, pin_device, 1, 0, &pin_invocation) == error::OK,
                "authoritative invocation fixture failed");
        AuthoritativeInvocationSnapshot pin;
        require(pin_registry.mint_authoritative_invocation_snapshot(pin_context, pin_session, pin_reset, pin_graph,
                                                                    pin_invocation, pin_root, &pin) == error::OK &&
                    pin.active() && pin.context() == pin_context && pin.session() == pin_session &&
                    pin.reset_epoch() == pin_reset && pin.graph_epoch() == pin_graph &&
                    pin.invocation() == pin_invocation && pin.root() == pin_root &&
                    pin_registry.validate_authoritative_invocation_snapshot(pin) == error::OK,
                "exact authoritative invocation snapshot was not minted");
        {
            AuthoritativeInvocationSnapshot forgotten;
            require(
                pin_registry.mint_authoritative_invocation_snapshot(pin_context, pin_session, pin_reset, pin_graph,
                                                                    pin_invocation, pin_root, &forgotten) == error::OK,
                "forgotten-scope snapshot mint failed");
        }
        AuthoritativeInvocationSnapshot overwritten;
        require(
            pin_registry.mint_authoritative_invocation_snapshot(pin_context, pin_session, pin_reset, pin_graph,
                                                                pin_invocation, pin_root, &overwritten) == error::OK,
            "move-overwrite snapshot mint failed");
        overwritten = std::move(pin);
        require(!pin.active() && overwritten.active() &&
                    pin_registry.validate_authoritative_invocation_snapshot(overwritten) == error::OK,
                "move assignment did not finish old pin and steal source");
        require(pin_registry.submit_invocation(pin_context, pin_session, pin_reset, pin_graph, pin_invocation, pin_root,
                                               0) == error::OK &&
                    pin_registry.release_invocation(pin_context, pin_session, pin_reset, pin_graph, pin_invocation,
                                                    pin_root) == error::BUSY,
                "snapshot did not pin parent release");
        Registry foreign;
        require(foreign.validate_authoritative_invocation_snapshot(overwritten) == error::MISMATCH,
                "foreign registry accepted invocation authority");
        auto concurrent_finish = std::async(
            std::launch::async, [&] { return pin_registry.finish_authoritative_invocation_snapshot(&overwritten); });
        auto release_rc =
            pin_registry.release_invocation(pin_context, pin_session, pin_reset, pin_graph, pin_invocation, pin_root);
        require(concurrent_finish.get() == error::OK && !overwritten.active() &&
                    (release_rc == error::OK || release_rc == error::BUSY),
                "concurrent finish/release produced an invalid transition");
        if (release_rc == error::BUSY) {
            release_rc = pin_registry.release_invocation(pin_context, pin_session, pin_reset, pin_graph, pin_invocation,
                                                         pin_root);
        }
        require(release_rc == error::OK &&
                    pin_registry.finish_authoritative_invocation_snapshot(&overwritten) == error::MISMATCH,
                "authoritative pin finish/release semantics failed");
    }

    // A capability outliving its Registry cannot call through the destroyed
    // object or authenticate a new Registry constructed at the same address.
    {
        alignas(Registry) unsigned char storage[sizeof(Registry)];
        AuthoritativeInvocationSnapshot stale;
        auto *                          first_registry    = new (storage) Registry();
        error                           placement_error   = error::OK;
        const auto                      placement_context = first_registry->create_context(placement_error);
        const auto                      placement_root    = root_token(800);
        SessionId                       placement_session{};
        SessionResetEpoch               placement_reset{};
        GraphEpoch                      placement_graph{};
        InvocationId                    placement_invocation{};
        const int                       placement_device[] = { 0 };
        require(placement_error == error::OK && first_registry->bind_backend(placement_context, 0) == error::OK &&
                    first_registry->attach_root(placement_context, placement_root, &placement_session,
                                                &placement_reset) == error::OK &&
                    first_registry->begin_graph(placement_context, placement_session, placement_reset, placement_root,
                                                &placement_graph) == error::OK &&
                    first_registry->begin_invocation(placement_context, placement_session, placement_reset,
                                                     placement_graph, placement_root, placement_device, 1,
                                                     placement_device, 1, 0, &placement_invocation) == error::OK &&
                    first_registry->mint_authoritative_invocation_snapshot(
                        placement_context, placement_session, placement_reset, placement_graph, placement_invocation,
                        placement_root, &stale) == error::OK,
                "placement-new authority fixture failed");
        first_registry->~Registry();
        require(!stale.active(), "Registry destruction left snapshot apparently active");
        auto * reincarnated = new (storage) Registry();
        require(reincarnated->validate_authoritative_invocation_snapshot(stale) == error::MISMATCH,
                "same-address Registry reincarnation authenticated stale authority");
        reincarnated->~Registry();
    }

    // Incarnation exhaustion is a fail-closed mint condition, not an ABA
    // fallback to address identity.
    {
        Registry                        overflow_registry(test_mutation::M6f_REGISTRY_INCARNATION_OVERFLOW);
        error                           overflow_error   = error::OK;
        const auto                      overflow_context = overflow_registry.create_context(overflow_error);
        const auto                      overflow_root    = root_token(900);
        SessionId                       overflow_session{};
        SessionResetEpoch               overflow_reset{};
        GraphEpoch                      overflow_graph{};
        InvocationId                    overflow_invocation{};
        const int                       overflow_device[] = { 0 };
        AuthoritativeInvocationSnapshot rejected;
        require(overflow_error == error::OK && overflow_registry.bind_backend(overflow_context, 0) == error::OK &&
                    overflow_registry.attach_root(overflow_context, overflow_root, &overflow_session,
                                                  &overflow_reset) == error::OK &&
                    overflow_registry.begin_graph(overflow_context, overflow_session, overflow_reset, overflow_root,
                                                  &overflow_graph) == error::OK &&
                    overflow_registry.begin_invocation(overflow_context, overflow_session, overflow_reset,
                                                       overflow_graph, overflow_root, overflow_device, 1,
                                                       overflow_device, 1, 0, &overflow_invocation) == error::OK &&
                    overflow_registry.mint_authoritative_invocation_snapshot(
                        overflow_context, overflow_session, overflow_reset, overflow_graph, overflow_invocation,
                        overflow_root, &rejected) == error::OVERFLOW &&
                    !rejected.active(),
                "exhausted Registry incarnation minted authority");
    }

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

    const int      devices[] = { 0, 1 };
    epoch_snapshot epoch_snap{};

    // A failed recording remains RETIRING while its retained asynchronous work
    // drains. It is never erased directly by rollback_record().
    GraphEpoch failed_record{};
    require(reg.begin_record(context, session, reset, root, &failed_record) == error::OK &&
                reg.rollback_record(context, session, reset, failed_record, root) == error::OK &&
                reg.extract_epoch(context, session, reset, failed_record, root, &epoch_snap) == error::OK &&
                epoch_snap.state == epoch_phase::RETIRING,
            "failed record did not enter authoritative RETIRING phase");
    RetireTicket failed_ticket{};
    require(reg.begin_retire(context, session, reset, failed_record, root, devices, 1, &failed_ticket) == error::OK,
            "failed record retire ticket failed");
    std::atomic<bool> delayed_started{ false };
    std::atomic<bool> delayed_release{ false };
    require(reg.attach_retire_terminal(
                failed_ticket, 0, std::make_shared<delayed_terminal>(reg, context, delayed_started, delayed_release)) ==
                error::OK,
            "failed record retained terminal attach failed");
    auto delayed_finish = std::async(std::launch::async, [&] { return reg.finish_retire(failed_ticket); });
    while (!delayed_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    require(delayed_finish.wait_for(std::chrono::milliseconds(10)) == std::future_status::timeout,
            "failed record terminal was not retained");
    require(reg.extract_epoch(context, session, reset, first, root, &epoch_snap) == error::OK &&
                epoch_snap.state == epoch_phase::ACTIVE && epoch_snap.is_active,
            "failed-record retirement disturbed active epoch");
    delayed_release.store(true, std::memory_order_release);
    require(delayed_finish.get() == error::OK &&
                reg.extract_epoch(context, session, reset, failed_record, root, &epoch_snap) == error::OK &&
                epoch_snap.state == epoch_phase::RETIRED,
            "failed record did not reach authoritative RETIRED phase");

    // Replacement is legal with an old invocation outstanding. Activation
    // moves old ACTIVE directly to RETIRING; exact old completion remains valid.
    constexpr size_t                concurrent_count = 8;
    std::vector<InvocationId>       concurrent_invocations(concurrent_count);
    std::vector<std::future<error>> concurrent_completions;
    concurrent_completions.reserve(concurrent_count);
    std::atomic<size_t> concurrent_ready{ 0 };
    std::atomic<bool>   release_concurrent{ false };
    for (size_t i = 0; i < concurrent_count; ++i) {
        concurrent_completions.push_back(std::async(std::launch::async, [&, i] {
            const auto begin_rc =
                reg.begin_invocation(context, session, reset, first, root, &concurrent_invocations[i]);
            if (begin_rc != error::OK) {
                return begin_rc;
            }
            concurrent_ready.fetch_add(1, std::memory_order_release);
            while (!release_concurrent.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            return reg.finish_invocation(context, session, reset, first, concurrent_invocations[i], root);
        }));
    }
    while (concurrent_ready.load(std::memory_order_acquire) != concurrent_count) {
        std::this_thread::yield();
    }

    GraphEpoch second{};
    require(reg.begin_record(context, session, reset, root, &second) == error::OK &&
                reg.activate(context, session, reset, second, root) == error::OK,
            "concurrent replacement activation failed");
    require(reg.extract_epoch(context, session, reset, first, root, &epoch_snap) == error::OK &&
                epoch_snap.state == epoch_phase::RETIRING && epoch_snap.live_invocations == concurrent_count &&
                !epoch_snap.is_active,
            "replacement did not preserve concurrent old invocations");

    RetireTicket old_ticket{};
    require(reg.begin_retire(context, session, reset, first, root, devices, 2, &old_ticket) == error::OK,
            "old epoch begin retire with outstanding invocation failed");
    std::atomic<unsigned> waits{ 0 };
    std::atomic<unsigned> destroys{ 0 };
    require(reg.attach_retire_terminal(old_ticket, 0,
                                       std::make_shared<probe_terminal>(reg, context, waits, destroys)) == error::OK,
            "first terminal attach failed");
    require(reg.finish_retire(old_ticket) == error::BUSY, "partial multi-device terminal set was released");
    require(reg.attach_retire_terminal(old_ticket, 1,
                                       std::make_shared<probe_terminal>(reg, context, waits, destroys)) == error::OK,
            "second terminal attach failed");
    require(reg.finish_retire(old_ticket) == error::BUSY, "retirement ignored outstanding old invocation");
    release_concurrent.store(true, std::memory_order_release);
    for (auto & completion : concurrent_completions) {
        require(completion.get() == error::OK, "exact concurrent old invocation completion rejected");
    }
    require(reg.finish_invocation(context, session, reset, first, concurrent_invocations.front(), root) == error::STALE,
            "stale concurrent old invocation completion accepted");

    RetireTicket wrong_context = old_ticket;
    wrong_context.context      = { context.value + 999 };
    require(reg.finish_retire(wrong_context) == error::STALE, "wrong-context retire accepted");
    RetireTicket wrong_root = old_ticket;
    wrong_root.token_root   = root_token(99);
    require(reg.finish_retire(wrong_root) == error::MISMATCH, "wrong-root retire accepted");
    require(reg.finish_retire(old_ticket) == error::OK && waits.load() == 2 && destroys.load() == 2,
            "exact retire did not wait for and release every terminal");
    require(reg.finish_retire(old_ticket) == error::STALE, "stale retire ticket mutated replacement state");
    require(reg.extract_epoch(context, session, reset, first, root, &epoch_snap) == error::OK &&
                epoch_snap.state == epoch_phase::RETIRED,
            "old epoch did not reach authoritative RETIRED phase");
    require(reg.extract_epoch(context, session, reset, second, root, &epoch_snap) == error::OK &&
                epoch_snap.state == epoch_phase::ACTIVE && epoch_snap.is_active,
            "old completion mutated replacement epoch");

    NoResourcesProof active_bypass{};
    RetireTicket     bypass_ticket{};
    require(reg.fail_record_no_resources(context, session, reset, second, root, &active_bypass) == error::STALE &&
                !active_bypass.active() && reg.begin_retire_no_resources(active_bypass, &bypass_ticket) == error::STALE,
            "ACTIVE epoch obtained a no-resources retirement bypass");

    GraphEpoch       partially_published{};
    NoResourcesProof partial_proof{};
    RetireTicket     partial_ticket{};
    require(reg.begin_record(context, session, reset, root, &partially_published) == error::OK &&
                reg.note_record_resources_published(context, session, reset, partially_published, root) == error::OK &&
                reg.fail_record_no_resources(context, session, reset, partially_published, root, &partial_proof) ==
                    error::BUSY &&
                !partial_proof.active() &&
                reg.rollback_record(context, session, reset, partially_published, root) == error::OK &&
                reg.begin_retire(context, session, reset, partially_published, root, devices, 1, &partial_ticket) ==
                    error::OK &&
                reg.attach_retire_terminal(
                    partial_ticket, 0, std::make_shared<probe_terminal>(reg, context, waits, destroys)) == error::OK &&
                reg.finish_retire(partial_ticket) == error::OK,
            "partially published failed record bypassed terminal retirement");

    // Zero-terminal retirement requires a proof minted while the failed epoch
    // is still RECORDING and before resource publication.
    GraphEpoch       empty_failed_record{};
    RetireTicket     empty_ticket{};
    NoResourcesProof empty_proof{};
    require(reg.begin_record(context, session, reset, root, &empty_failed_record) == error::OK &&
                reg.fail_record_no_resources(context, session, reset, empty_failed_record, root, &empty_proof) ==
                    error::OK &&
                empty_proof.active() && reg.begin_retire_no_resources(empty_proof, &empty_ticket) == error::OK &&
                reg.finish_retire(empty_ticket) == error::OK &&
                reg.extract_epoch(context, session, reset, empty_failed_record, root, &epoch_snap) == error::OK &&
                epoch_snap.state == epoch_phase::RETIRED,
            "explicit no-retained-resources retirement failed");

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

    Registry          record_allocation_failure(test_mutation::M8a_RECORD_ALLOCATION_FAILURE);
    const ContextId   allocation_context = record_allocation_failure.create_context(err);
    SessionId         allocation_session{};
    SessionResetEpoch allocation_reset{};
    GraphEpoch        allocation_epoch{};
    require(err == error::OK &&
                record_allocation_failure.attach_root(allocation_context, root, &allocation_session,
                                                      &allocation_reset) == error::OK &&
                record_allocation_failure.begin_record(allocation_context, allocation_session, allocation_reset, root,
                                                       &allocation_epoch) == error::ALLOCATION_FAILED,
            "record allocation failure did not return typed status");

    Registry          invocation_allocation_failure(test_mutation::M8b_INVOCATION_ALLOCATION_FAILURE);
    const ContextId   invocation_allocation_context = invocation_allocation_failure.create_context(err);
    SessionId         invocation_allocation_session{};
    SessionResetEpoch invocation_allocation_reset{};
    GraphEpoch        invocation_allocation_epoch{};
    InvocationId      allocation_invocation{};
    require(err == error::OK &&
                invocation_allocation_failure.attach_root(invocation_allocation_context, root,
                                                          &invocation_allocation_session,
                                                          &invocation_allocation_reset) == error::OK &&
                invocation_allocation_failure.begin_record(invocation_allocation_context, invocation_allocation_session,
                                                           invocation_allocation_reset, root,
                                                           &invocation_allocation_epoch) == error::OK &&
                invocation_allocation_failure.activate(invocation_allocation_context, invocation_allocation_session,
                                                       invocation_allocation_reset, invocation_allocation_epoch,
                                                       root) == error::OK &&
                invocation_allocation_failure.begin_invocation(
                    invocation_allocation_context, invocation_allocation_session, invocation_allocation_reset,
                    invocation_allocation_epoch, root, &allocation_invocation) == error::ALLOCATION_FAILED,
            "invocation allocation failure did not return typed status");

    Registry          retire_allocation_failure(test_mutation::M8c_RETIRE_ALLOCATION_FAILURE);
    const ContextId   retire_allocation_context = retire_allocation_failure.create_context(err);
    SessionId         retire_allocation_session{};
    SessionResetEpoch retire_allocation_reset{};
    GraphEpoch        retire_allocation_epoch{};
    RetireTicket      allocation_ticket{};
    require(
        err == error::OK && retire_allocation_failure.bind_backend(retire_allocation_context, 0) == error::OK &&
            retire_allocation_failure.attach_root(retire_allocation_context, root, &retire_allocation_session,
                                                  &retire_allocation_reset) == error::OK &&
            retire_allocation_failure.begin_record(retire_allocation_context, retire_allocation_session,
                                                   retire_allocation_reset, root,
                                                   &retire_allocation_epoch) == error::OK &&
            retire_allocation_failure.activate(retire_allocation_context, retire_allocation_session,
                                               retire_allocation_reset, retire_allocation_epoch, root) == error::OK &&
            retire_allocation_failure.begin_retire(retire_allocation_context, retire_allocation_session,
                                                   retire_allocation_reset, retire_allocation_epoch, root, devices, 1,
                                                   &allocation_ticket) == error::ALLOCATION_FAILED,
        "retire allocation failure did not return typed status");

    Registry          terminal_allocation_failure(test_mutation::M8d_TERMINAL_ALLOCATION_FAILURE);
    const ContextId   terminal_allocation_context = terminal_allocation_failure.create_context(err);
    SessionId         terminal_allocation_session{};
    SessionResetEpoch terminal_allocation_reset{};
    GraphEpoch        terminal_allocation_epoch{};
    require(err == error::OK && terminal_allocation_failure.bind_backend(terminal_allocation_context, 0) == error::OK &&
                terminal_allocation_failure.attach_root(terminal_allocation_context, root, &terminal_allocation_session,
                                                        &terminal_allocation_reset) == error::OK &&
                terminal_allocation_failure.begin_record(terminal_allocation_context, terminal_allocation_session,
                                                         terminal_allocation_reset, root,
                                                         &terminal_allocation_epoch) == error::OK &&
                terminal_allocation_failure.activate(terminal_allocation_context, terminal_allocation_session,
                                                     terminal_allocation_reset, terminal_allocation_epoch,
                                                     root) == error::OK &&
                terminal_allocation_failure.begin_retire(terminal_allocation_context, terminal_allocation_session,
                                                         terminal_allocation_reset, terminal_allocation_epoch, root,
                                                         devices, 1, &allocation_ticket) == error::OK &&
                terminal_allocation_failure.attach_retire_terminal(
                    allocation_ticket, 0,
                    std::make_shared<probe_terminal>(terminal_allocation_failure, terminal_allocation_context, waits,
                                                     destroys)) == error::ALLOCATION_FAILED,
            "terminal allocation failure did not return typed status");

    Registry          lock_allocation(test_mutation::M9_PERSISTENT_ALLOCATION_UNDER_LOCK);
    const ContextId   lock_context = lock_allocation.create_context(err);
    SessionId         lock_session{};
    SessionResetEpoch lock_reset{};
    GraphEpoch        lock_epoch{};
    require(err == error::OK &&
                lock_allocation.attach_root(lock_context, root, &lock_session, &lock_reset) == error::OK &&
                lock_allocation.begin_record(lock_context, lock_session, lock_reset, root, &lock_epoch) ==
                    error::LOCK_HELD_ALLOCATION,
            "lock-held allocation instrumentation mutation survived");

    // llama.cpp-8q35 (TKV-15): host-only reproduction of the cross-context
    // DEVICE_BUSY collision -- context A submits an invocation (drives its
    // graph to COMPLETE) but is never released; context B then tries to open
    // a fresh invocation on the SAME device and is refused DEVICE_BUSY, even
    // though A's graph is a drainable terminal, not genuinely in-flight work.
    // This is the exact registry-level shape of the production defect (probe
    // evidence: requester_context=3 blocked_by_context=2, blocked_by_state=
    // COMPLETE). The fix under review (device_owner_context() plus a foreign-
    // terminal drain in ggml_backend_sycl_graph_compute_impl() -- the CALLER
    // of ggml_sycl_execution_begin_graph(), deliberately not inside
    // begin_graph's own locked body; see the rationale comment at that call
    // site) is a ggml-sycl.cpp change verified separately on real hardware;
    // this test proves the
    // Registry-level state machine the fix depends on: (a) the new accessor
    // correctly identifies the blocking owner, (b) that owner's graph really
    // is COMPLETE (never OPEN -- the two must not be conflated), and (c)
    // releasing it unblocks the second context's identical retry.
    {
        Registry          cross_ctx_registry;
        error             cross_ctx_err = error::OK;
        const auto        ctx_a         = cross_ctx_registry.create_context(cross_ctx_err);
        const auto        root_a        = root_token(800);
        SessionId         session_a{};
        SessionResetEpoch reset_a{};
        GraphEpoch        graph_a{};
        InvocationId      invocation_a{};
        const int         device0[] = { 0 };
        require(cross_ctx_err == error::OK && cross_ctx_registry.bind_backend(ctx_a, 0) == error::OK &&
                    cross_ctx_registry.attach_root(ctx_a, root_a, &session_a, &reset_a) == error::OK &&
                    cross_ctx_registry.begin_graph(ctx_a, session_a, reset_a, root_a, &graph_a) == error::OK &&
                    cross_ctx_registry.begin_invocation(ctx_a, session_a, reset_a, graph_a, root_a, device0, 1, device0,
                                                        1, 0, &invocation_a) == error::OK,
                "cross-context fixture: context A failed to claim device 0");
        require(cross_ctx_registry.submit_invocation(ctx_a, session_a, reset_a, graph_a, invocation_a, root_a, 0) ==
                    error::OK,
                "cross-context fixture: context A failed to submit (reach COMPLETE)");

        // A's graph is now COMPLETE but deliberately unreleased -- exactly the
        // "terminal-but-unreleased sibling" state the probe caught in
        // production. Confirm it via extract() before using it as a fixture.
        snapshot a_snap{};
        require(cross_ctx_registry.extract(ctx_a, &a_snap) == error::OK && a_snap.graph_state == graph_phase::COMPLETE,
                "cross-context fixture: context A did not reach COMPLETE");

        const auto        ctx_b  = cross_ctx_registry.create_context(cross_ctx_err);
        const auto        root_b = root_token(801);
        SessionId         session_b{};
        SessionResetEpoch reset_b{};
        GraphEpoch        graph_b{};
        InvocationId      invocation_b{};
        require(cross_ctx_err == error::OK && cross_ctx_registry.bind_backend(ctx_b, 0) == error::OK &&
                    cross_ctx_registry.attach_root(ctx_b, root_b, &session_b, &reset_b) == error::OK &&
                    cross_ctx_registry.begin_graph(ctx_b, session_b, reset_b, root_b, &graph_b) == error::OK,
                "cross-context fixture: context B failed to attach/begin its own graph");

        // RED (today): B is refused, even though A's graph is a drainable
        // terminal -- this is execution-lifecycle.cpp:1128 reproduced host-only.
        require(cross_ctx_registry.begin_invocation(ctx_b, session_b, reset_b, graph_b, root_b, device0, 1, device0, 1,
                                                    0, &invocation_b) == error::DEVICE_BUSY,
                "context B unexpectedly succeeded while A still holds device 0 -- fixture invalid");

        // device_owner_context() must identify A as the blocker, and extract()
        // on that ContextId must confirm COMPLETE -- the exact two facts a
        // foreign-terminal drain needs before it may touch another context's
        // invocation (never on OPEN/SEALED -- canonical contract §12.3).
        ContextId blocker{};
        require(cross_ctx_registry.device_owner_context(0, &blocker) == error::OK && blocker.value == ctx_a.value,
                "device_owner_context did not identify context A as the blocking owner");
        snapshot blocker_snap{};
        require(cross_ctx_registry.extract(blocker, &blocker_snap) == error::OK &&
                    blocker_snap.graph_state == graph_phase::COMPLETE,
                "blocking owner's graph was not reported COMPLETE");

        // GREEN after drain: releasing A's terminal invocation (what the real
        // fix's foreign-terminal drain does, after actually waiting on A's
        // device queue -- device-side, not reproducible host-only) frees
        // device 0, and B's IDENTICAL retry now succeeds.
        require(cross_ctx_registry.release_invocation(ctx_a, session_a, reset_a, graph_a, invocation_a, root_a) ==
                    error::OK,
                "releasing context A's terminal invocation failed");
        require(cross_ctx_registry.device_owner_context(0, &blocker) == error::NOT_FOUND,
                "device 0 still reports an owner after A's release");
        require(cross_ctx_registry.begin_invocation(ctx_b, session_b, reset_b, graph_b, root_b, device0, 1, device0, 1,
                                                    0, &invocation_b) == error::OK,
                "context B's retry after A's release still failed");
    }

    // llama.cpp-8q35: the Q2 reconciliation case -- a caller's own LOCALLY
    // CACHED invocation id can go stale relative to the registry's current
    // one. This is exactly what ggml_sycl_execution_release_graph()'s
    // ctx->execution_invocation_id gate is vulnerable to in production: it
    // trusts the caller's last-seen id, not the registry's current one.
    // Releasing with a STALE graph_epoch/invocation must be refused, never
    // silently succeed against the WRONG (now-current) one -- and the drain
    // path must recover by re-reading current state via extract()/
    // device_owner_context() rather than trusting any caller's cached id.
    {
        Registry          stale_registry;
        error             stale_err = error::OK;
        const auto        ctx_c     = stale_registry.create_context(stale_err);
        const auto        root_c    = root_token(802);
        SessionId         session_c{};
        SessionResetEpoch reset_c{};
        GraphEpoch        graph_c1{};
        InvocationId      invocation_c1{};
        const int         device0[] = { 0 };
        require(stale_err == error::OK && stale_registry.bind_backend(ctx_c, 0) == error::OK &&
                    stale_registry.attach_root(ctx_c, root_c, &session_c, &reset_c) == error::OK &&
                    stale_registry.begin_graph(ctx_c, session_c, reset_c, root_c, &graph_c1) == error::OK &&
                    stale_registry.begin_invocation(ctx_c, session_c, reset_c, graph_c1, root_c, device0, 1, device0, 1,
                                                    0, &invocation_c1) == error::OK &&
                    stale_registry.submit_invocation(ctx_c, session_c, reset_c, graph_c1, invocation_c1, root_c, 0) ==
                        error::OK &&
                    stale_registry.release_invocation(ctx_c, session_c, reset_c, graph_c1, invocation_c1, root_c) ==
                        error::OK &&
                    stale_registry.retire_graph(ctx_c, session_c, reset_c, graph_c1, root_c) == error::OK,
                "stale-id fixture: first begin/submit/release/retire cycle failed");

        // Second cycle on the SAME context: graph_epoch/invocation both
        // advance, mirroring ctx_src's graph_epoch=7 in the production probe.
        GraphEpoch   graph_c2{};
        InvocationId invocation_c2{};
        require(stale_registry.begin_graph(ctx_c, session_c, reset_c, root_c, &graph_c2) == error::OK &&
                    stale_registry.begin_invocation(ctx_c, session_c, reset_c, graph_c2, root_c, device0, 1, device0, 1,
                                                    0, &invocation_c2) == error::OK &&
                    stale_registry.submit_invocation(ctx_c, session_c, reset_c, graph_c2, invocation_c2, root_c, 0) ==
                        error::OK,
                "stale-id fixture: second begin/submit cycle failed");

        // A caller still holding the FIRST (stale) graph_epoch/invocation --
        // exactly what a desynced ctx->execution_invocation_id would replay --
        // must be refused, never silently released against the CURRENT one.
        require(
            stale_registry.release_invocation(ctx_c, session_c, reset_c, graph_c1, invocation_c1, root_c) != error::OK,
            "release with a stale graph_epoch/invocation incorrectly succeeded");
        snapshot still_blocked{};
        require(stale_registry.extract(ctx_c, &still_blocked) == error::OK &&
                    still_blocked.graph_state == graph_phase::COMPLETE &&
                    still_blocked.invocation.value == invocation_c2.value,
                "device 0 was incorrectly freed by the stale release attempt");

        // The CURRENT (correct) id still releases cleanly -- this is what a
        // drain that re-reads registry state via extract()/device_owner_
        // context() (rather than trusting a caller's cached id) achieves.
        require(
            stale_registry.release_invocation(ctx_c, session_c, reset_c, graph_c2, invocation_c2, root_c) == error::OK,
            "release with the current graph_epoch/invocation failed");
    }

    std::cout << "graph epoch lifecycle: ok\n";
    return 0;
}
