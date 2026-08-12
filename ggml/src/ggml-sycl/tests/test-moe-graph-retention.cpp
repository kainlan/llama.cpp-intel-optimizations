#define GGML_SYCL_RETENTION_TESTING 1
#include "../moe-graph-retention.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

using namespace ggml_sycl;
using namespace ggml_sycl::execution;
using namespace ggml_sycl::moe;

static void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

static lifecycle::ModelToken token(uint64_t id = 1) {
    return {
        lifecycle::ModelId{ id },
        lifecycle::LoadTxnId{ id + 10 },
        lifecycle::SlotToken{ 1, id + 20 }
    };
}

struct fixture {
    explicit fixture(graph_retention_registry & retention) : retention(retention) {
        context = execution.create_context(err);
        require(err == error::OK && execution.bind_backend(context, 0) == error::OK &&
                    execution.bind_backend(context, 1) == error::OK &&
                    execution.attach_root(context, token(), &session, &reset) == error::OK,
                "fixture setup failed");
        const int devices[] = { 0 };
        const int participants[] = { 0 };
        require(execution.begin_graph(context, session, reset, token(), &outer_epoch) == error::OK &&
                    execution.begin_invocation(context, session, reset, outer_epoch, token(), devices, 1,
                                               participants, 1, 0, &outer_invocation) == error::OK,
                "fixture outer invocation setup failed");
    }

    graph_retention_registry & retention;
    Registry                   execution;
    error                      err = error::OK;
    ContextId                  context{};
    SessionId                  session{};
    SessionResetEpoch          reset{};
    GraphEpoch                outer_epoch{};
    InvocationId              outer_invocation{};
};

class test_terminal final : public device_terminal {
  public:
    test_terminal(std::atomic<bool> &        ready,
                  std::atomic<unsigned> &    waits,
                  graph_retention_registry * registry = nullptr) :
        ready_(ready),
        waits_(waits),
        registry_(registry) {}

    bool ready() const noexcept override { return ready_.load(std::memory_order_acquire); }

    void wait() noexcept override {
        if (registry_) {
            (void) registry_->size();
        }
        waits_.fetch_add(1, std::memory_order_relaxed);
    }
  private:
    std::atomic<bool> &        ready_;
    std::atomic<unsigned> &    waits_;
    graph_retention_registry * registry_;
};

class blocking_terminal final : public device_terminal {
  public:
    blocking_terminal(std::atomic<bool> & entered, std::atomic<bool> & release) :
        entered_(entered),
        release_(release) {}

    bool ready() const noexcept override { return true; }

    void wait() noexcept override {
        entered_.store(true, std::memory_order_release);
        while (!release_.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    }
  private:
    std::atomic<bool> & entered_;
    std::atomic<bool> & release_;
};

struct drain_state {
    std::atomic<bool> *     ready;
    std::atomic<bool> *     succeeds;
    std::atomic<unsigned> * waits;
};

static bool drain_ready_callback(const void * opaque) noexcept {
    return static_cast<const drain_state *>(opaque)->ready->load(std::memory_order_acquire);
}

static bool drain_wait_callback(void * opaque) noexcept {
    auto * state = static_cast<drain_state *>(opaque);
    state->waits->fetch_add(1, std::memory_order_relaxed);
    return state->succeeds->load(std::memory_order_acquire);
}

struct blocking_drain_state {
    std::atomic<bool> *     entered;
    std::atomic<bool> *     release;
    std::atomic<unsigned> * waits;
};

static bool blocking_drain_ready_callback(const void *) noexcept {
    return true;
}

static bool blocking_drain_wait_callback(void * opaque) noexcept {
    auto * state = static_cast<blocking_drain_state *>(opaque);
    state->waits->fetch_add(1, std::memory_order_relaxed);
    state->entered->store(true, std::memory_order_release);
    while (!state->release->load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    return true;
}

struct lock_probe {
    lock_probe(graph_retention_registry & registry, std::atomic<unsigned> & drops) : registry(registry), drops(drops) {}

    ~lock_probe() {
        (void) registry.size();
        drops.fetch_add(1, std::memory_order_relaxed);
    }

    graph_retention_registry & registry;
    std::atomic<unsigned> &    drops;
};

static retained_allocation_owner owner_capability(uint64_t                            allocation,
                                                  const std::shared_ptr<const void> & handle,
                                                  int                                 device     = 0,
                                                  uint64_t                            generation = 1,
                                                  size_t                              extent     = 4096) {
    return retained_allocation_test_factory::mint(allocation, generation, device, extent, handle);
}

static mmid_batch_binding binding(uint64_t allocation, const std::shared_ptr<const void> & handle, int device = 0) {
    return {
        { allocation, 1, 9, device, 16, 64, 1 },
        owner_capability(allocation, handle, device)
    };
}

static graph_recording_transaction begin_tx(fixture & f) {
    graph_recording_transaction tx;
    require(graph_recording_transaction::begin(f.retention, f.execution, f.context, f.session, f.reset, token(), &tx) ==
                retention_error::OK,
            "transaction begin failed");
    return tx;
}

static void require_bad_identity(mmid_operand_identity identity, uint64_t owner_id, const char * message) {
    graph_retention_registry registry;
    fixture                  f(registry);
    auto                     tx    = begin_tx(f);
    auto                     owner = std::make_shared<int>(99);
    std::atomic<bool>        ready{ true };
    std::atomic<unsigned>    waits{ 0 };
    require(tx.add_batch({ identity, owner_capability(owner_id, owner) }) == retention_error::OK &&
                tx.note_submission(0, submit_outcome::SUBMITTED) == retention_error::OK &&
                tx.set_terminal(0, std::make_shared<test_terminal>(ready, waits)) == retention_error::OK,
            "bad identity setup failed");
    tx.mark_finalized();
    require(tx.commit() == retention_error::MISMATCH, message);
}

int main() {
    // A retained child epoch records, publishes, replays, and retires while an
    // outer compatibility invocation remains open on the same session.
    graph_retention_registry child_retention;
    fixture                  child_fixture(child_retention);
    const GraphEpoch   outer_epoch      = child_fixture.outer_epoch;
    const InvocationId outer_invocation = child_fixture.outer_invocation;
    std::atomic<bool>     child_ready{ true };
    std::atomic<unsigned> child_waits{ 0 };
    auto                  child_tx    = begin_tx(child_fixture);
    auto                  child_owner = std::make_shared<int>(6);
    require(child_tx.add_batch(binding(96, child_owner)) == retention_error::OK &&
                child_tx.set_terminal(0, std::make_shared<test_terminal>(child_ready, child_waits)) ==
                    retention_error::OK &&
                child_tx.note_submission(0, submit_outcome::SUBMITTED) == retention_error::OK,
            "child record inside outer invocation failed");
    child_tx.mark_finalized();
    require(child_tx.commit() == retention_error::OK, "child publication inside outer invocation failed");
    published_graph_token child_token;
    InvocationId          child_invocation{};
    InvocationId child_invocation2{};
    require(child_retention.acquire_published_token(child_tx.key(), &child_token) == retention_error::OK &&
                child_retention.begin_invocation(child_token, &child_invocation) == retention_error::OK &&
                child_retention.finish_invocation(child_token, child_invocation) == retention_error::OK &&
                child_retention.begin_invocation(child_token, &child_invocation2) == retention_error::OK &&
                child_invocation2.value != child_invocation.value &&
                child_retention.finish_invocation(child_token, child_invocation2) == retention_error::OK &&
                child_retention.retire_exact(child_tx.key()) == retention_error::OK,
            "child multiple replay/retirement inside outer invocation failed");
    snapshot outer_snapshot{};
    require(child_fixture.execution.extract(child_fixture.context, &outer_snapshot) == error::OK &&
                outer_snapshot.graph_epoch == outer_epoch && outer_snapshot.invocation == outer_invocation,
            "child retirement disturbed outer invocation");

    // A live child invocation pins the exact parent lease. Final submit/release
    // is BUSY without mutating the parent; after child completion it succeeds.
    graph_retention_registry parent_registry;
    fixture                  parent_fixture(parent_registry);
    auto                     parent_tx    = begin_tx(parent_fixture);
    auto                     parent_owner = std::make_shared<int>(61);
    require(parent_tx.add_owner(owner_capability(961, parent_owner)) == retention_error::OK &&
                parent_tx.set_terminal(0, std::make_shared<test_terminal>(child_ready, child_waits)) ==
                    retention_error::OK &&
                parent_tx.note_submission(0, submit_outcome::SUBMITTED) == retention_error::OK,
            "parent-pin child setup failed");
    parent_tx.mark_finalized();
    require(parent_tx.commit() == retention_error::OK, "parent-pin child publication failed");
    published_graph_token parent_token;
    InvocationId          live_child{};
    unsigned              parent_replay_count = 0;
    require(parent_registry.acquire_published_token(parent_tx.key(), &parent_token) == retention_error::OK &&
                parent_registry.begin_invocation(parent_token, &live_child) == retention_error::OK &&
                (++parent_replay_count == 1) &&
                parent_fixture.execution.complete_invocation(
                    parent_fixture.context, parent_fixture.session, parent_fixture.reset,
                    parent_fixture.outer_epoch, parent_fixture.outer_invocation, token(), 0) == error::BUSY,
            "live child did not pin parent final completion");
    snapshot pinned_parent{};
    require(parent_fixture.execution.extract(parent_fixture.context, &pinned_parent) == error::OK &&
                pinned_parent.invocation == parent_fixture.outer_invocation &&
                pinned_parent.graph_state == graph_phase::OPEN,
            "BUSY parent completion mutated state or released lease");
    require(parent_registry.finish_invocation(parent_token, live_child) == retention_error::OK &&
                parent_fixture.execution.complete_invocation(
                    parent_fixture.context, parent_fixture.session, parent_fixture.reset,
                    parent_fixture.outer_epoch, parent_fixture.outer_invocation, token(), 0) == error::OK &&
                parent_fixture.execution.retire_graph(parent_fixture.context, parent_fixture.session,
                                                      parent_fixture.reset, parent_fixture.outer_epoch,
                                                      token()) == error::OK,
            "parent completion did not succeed after child completion");

    // The published executable survives outer-token replacement. Recording and
    // activation were tied to token N; replay binds to the compatible current
    // token N+1 and pins that exact parent invocation.
    GraphEpoch replacement_outer{};
    InvocationId replacement_outer_invocation{};
    const int replacement_devices[] = { 0 };
    const int replacement_participants[] = { 0 };
    require(parent_fixture.execution.begin_graph(parent_fixture.context, parent_fixture.session,
                                                  parent_fixture.reset, token(), &replacement_outer) == error::OK &&
                parent_fixture.execution.begin_invocation(
                    parent_fixture.context, parent_fixture.session, parent_fixture.reset, replacement_outer,
                    token(), replacement_devices, 1, replacement_participants, 1, 0,
                    &replacement_outer_invocation) == error::OK &&
                parent_registry.begin_invocation(parent_token, &live_child) == retention_error::OK &&
                (++parent_replay_count == 2),
            "published child token did not replay under compatible outer token N+1");
    InvocationId wrong_root_child{};
    require(parent_fixture.execution.child_begin_invocation(
                parent_fixture.context, parent_fixture.session, parent_fixture.reset, parent_tx.key().epoch,
                token(99), &wrong_root_child) == error::MISMATCH &&
                parent_fixture.execution.complete_invocation(
                    parent_fixture.context, parent_fixture.session, parent_fixture.reset, replacement_outer,
                    replacement_outer_invocation, token(), 0) == error::BUSY &&
                parent_registry.finish_invocation(parent_token, live_child) == retention_error::OK &&
                parent_fixture.execution.complete_invocation(
                    parent_fixture.context, parent_fixture.session, parent_fixture.reset, replacement_outer,
                    replacement_outer_invocation, token(), 0) == error::OK &&
                parent_fixture.execution.retire_graph(parent_fixture.context, parent_fixture.session,
                                                       parent_fixture.reset, replacement_outer, token()) == error::OK &&
                parent_registry.retire_exact(parent_tx.key()) == retention_error::OK && parent_replay_count == 2,
            "cross-token replay did not pin/count/finish/retire exact token N+1 or rejected wrong root");

    // Race the exact final parent transition against child completion. Either
    // the child wins, or the parent observes BUSY and succeeds on retry; it may
    // never release the device lease while the child remains live.
    graph_retention_registry race_registry;
    fixture                  race_fixture(race_registry);
    auto                     race_tx    = begin_tx(race_fixture);
    auto                     race_owner = std::make_shared<int>(62);
    require(race_tx.add_owner(owner_capability(962, race_owner)) == retention_error::OK &&
                race_tx.set_terminal(0, std::make_shared<test_terminal>(child_ready, child_waits)) ==
                    retention_error::OK &&
                race_tx.note_submission(0, submit_outcome::SUBMITTED) == retention_error::OK,
            "parent/child race setup failed");
    race_tx.mark_finalized();
    require(race_tx.commit() == retention_error::OK, "parent/child race publication failed");
    published_graph_token race_token;
    InvocationId          race_child{};
    require(race_registry.acquire_published_token(race_tx.key(), &race_token) == retention_error::OK &&
                race_registry.begin_invocation(race_token, &race_child) == retention_error::OK,
            "parent/child race invocation setup failed");
    std::atomic<bool> race_go{ false };
    retention_error child_race_rc = retention_error::STALE;
    error parent_race_rc = error::STALE;
    std::thread child_racer([&] {
        while (!race_go.load(std::memory_order_acquire)) std::this_thread::yield();
        child_race_rc = race_registry.finish_invocation(race_token, race_child);
    });
    std::thread parent_racer([&] {
        while (!race_go.load(std::memory_order_acquire)) std::this_thread::yield();
        parent_race_rc = race_fixture.execution.complete_invocation(
            race_fixture.context, race_fixture.session, race_fixture.reset, race_fixture.outer_epoch,
            race_fixture.outer_invocation, token(), 0);
    });
    race_go.store(true, std::memory_order_release);
    child_racer.join();
    parent_racer.join();
    require(child_race_rc == retention_error::OK &&
                (parent_race_rc == error::OK || parent_race_rc == error::BUSY),
            "parent/child race returned invalid status");
    if (parent_race_rc == error::BUSY) {
        parent_race_rc = race_fixture.execution.complete_invocation(
            race_fixture.context, race_fixture.session, race_fixture.reset, race_fixture.outer_epoch,
            race_fixture.outer_invocation, token(), 0);
    }
    snapshot raced_parent{};
    require(parent_race_rc == error::OK &&
                race_fixture.execution.extract(race_fixture.context, &raced_parent) == error::OK &&
                raced_parent.invocation.value == 0,
            "parent/child race released early or failed retry");
    require(race_registry.retire_exact(race_tx.key()) == retention_error::OK,
            "parent/child race cleanup failed");

    // Every pre-commit failpoint uses abort_partial and returns both registries
    // to baseline without requiring a fabricated terminal for untouched queues.
    for (int fail_after = 0; fail_after < 5; ++fail_after) {
        graph_retention_registry partial_registry;
        fixture                  partial_fixture(partial_registry);
        auto                     partial        = begin_tx(partial_fixture);
        const auto               partial_key    = partial.key();
        auto                     partial_owner0 = std::make_shared<int>(70 + fail_after);
        auto                     partial_owner1 = std::make_shared<int>(80 + fail_after);
        require(partial.add_owner(owner_capability(970 + fail_after, partial_owner0, 0)) == retention_error::OK,
                "partial first-owner setup failed");
        if (fail_after >= 1) {
            require(partial.add_owner(owner_capability(980 + fail_after, partial_owner1, 1)) == retention_error::OK,
                    "partial second-owner setup failed");
        }
        if (fail_after >= 2) {
            require(partial.set_terminal(0, std::make_shared<test_terminal>(child_ready, child_waits)) ==
                        retention_error::OK,
                    "partial primary-terminal setup failed");
        }
        if (fail_after >= 3) {
            require(partial.set_terminal(1, std::make_shared<test_terminal>(child_ready, child_waits)) ==
                        retention_error::OK,
                    "partial secondary-terminal setup failed");
        }
        if (fail_after >= 4) {
            require(partial.note_submission(0, submit_outcome::SUBMITTED) == retention_error::OK,
                    "partial submission setup failed");
        }
        require(partial.abort_partial() == retention_error::OK && partial_registry.size() == 0,
                "partial abort did not restore retention baseline");
        epoch_snapshot removed{};
        require(partial_fixture.execution.child_extract_epoch(
                    partial_fixture.context, partial_fixture.session, partial_fixture.reset, partial_key.epoch,
                    token(), &removed) == error::STALE,
                "partial abort left child lifecycle epoch behind");
    }

    // abort_partial freezes an exact device-1 snapshot before unlocking. Its
    // deterministic wait gives every mutator class a chance to race the frozen
    // record; all must return BUSY and device 0 must not join the drain set.
    graph_retention_registry freeze_registry;
    fixture                  freeze_fixture(freeze_registry);
    auto                     freeze_tx  = begin_tx(freeze_fixture);
    const auto               freeze_key = freeze_tx.key();
    std::atomic<bool>        freeze_entered{ false };
    std::atomic<bool>        freeze_release{ false };
    std::atomic<unsigned>    freeze_waits{ 0 };
    auto freeze_state = std::make_shared<blocking_drain_state>(
        blocking_drain_state{ &freeze_entered, &freeze_release, &freeze_waits });
    require(freeze_tx.note_submission(1, submit_outcome::UNKNOWN) == retention_error::OK &&
                freeze_tx.set_quiescence_proof(
                    1, queue_quiescence_test_factory::mint(freeze_state, blocking_drain_ready_callback,
                                                           blocking_drain_wait_callback)) == retention_error::OK,
            "freeze-race device-1 setup failed");
    retention_error freeze_abort_rc = retention_error::STALE;
    std::thread freeze_aborter([&] { freeze_abort_rc = freeze_tx.abort_partial(); });
    while (!freeze_entered.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    auto frozen_owner = std::make_shared<int>(125);
    auto frozen_table = graph_private_table_owner::create(
        freeze_key, 2125, 1, 0, { owner_capability(2126, frozen_owner) });
    auto extra_proof_state = std::make_shared<drain_state>(
        drain_state{ &freeze_entered, &freeze_release, &freeze_waits });
    require(freeze_tx.add_batch(binding(2127, frozen_owner)) == retention_error::BUSY &&
                freeze_tx.add_owner(owner_capability(2128, frozen_owner)) == retention_error::BUSY &&
                freeze_tx.add_table({ 2125, 1, 0, frozen_table }) == retention_error::BUSY &&
                freeze_tx.set_terminal(0, std::make_shared<test_terminal>(child_ready, child_waits)) ==
                    retention_error::BUSY &&
                freeze_tx.note_submission(0, submit_outcome::UNKNOWN) == retention_error::BUSY &&
                freeze_tx.set_quiescence_proof(
                    0, queue_quiescence_test_factory::mint(extra_proof_state, drain_ready_callback,
                                                           drain_wait_callback)) == retention_error::BUSY,
            "assembly mutator crossed partial-abort freeze");
    auto frozen_snapshot = freeze_registry.snapshot(freeze_key);
    require(frozen_snapshot && frozen_snapshot->phase == retention_phase::QUARANTINED &&
                frozen_snapshot->submissions.size() == 1 && frozen_snapshot->submissions.count(1) == 1 &&
                frozen_snapshot->quiescence_proofs.size() == 1 && frozen_snapshot->quiescence_proofs.count(1) == 1 &&
                frozen_snapshot->batches.empty() && frozen_snapshot->generic_owners.empty() &&
                frozen_snapshot->tables.empty() && frozen_snapshot->terminals.empty(),
            "frozen abort set changed after concurrent mutators");
    frozen_snapshot.reset();
    freeze_release.store(true, std::memory_order_release);
    freeze_aborter.join();
    require(freeze_abort_rc == retention_error::OK && freeze_waits.load() == 1 && freeze_registry.size() == 0,
            "abort did not wait the exact frozen device-1 set");

    // Immediate successful abort races every wrapper entry point. Regardless
    // of which side reaches the shared control mutex first, terminal state is
    // stable and every later mutator returns BUSY without touching reset state.
    for (int operation = 0; operation < 6; ++operation) {
        // Callback storage precedes registry/fixture/transaction declarations,
        // so retained proofs can never outlive their ASAN-visible stack state.
        std::atomic<bool>     wrapper_ready{ true };
        std::atomic<bool>     wrapper_succeeds{ true };
        std::atomic<unsigned> wrapper_waits{ 0 };
        auto wrapper_proof_state = std::make_shared<drain_state>(
            drain_state{ &wrapper_ready, &wrapper_succeeds, &wrapper_waits });
        graph_retention_registry wrapper_registry;
        fixture                  wrapper_fixture(wrapper_registry);
        auto                     wrapper_tx = begin_tx(wrapper_fixture);
        require(wrapper_tx.set_quiescence_proof(
                    0, queue_quiescence_test_factory::mint(wrapper_proof_state, drain_ready_callback,
                                                           drain_wait_callback)) == retention_error::OK,
                "immediate-abort baseline proof setup failed");
        auto                     wrapper_owner = std::make_shared<int>(2200 + operation);
        auto wrapper_table = graph_private_table_owner::create(
            wrapper_tx.key(), 2300 + operation, 1, 0, { owner_capability(2400 + operation, wrapper_owner) });
        std::atomic<bool> wrapper_go{ false };
        retention_error abort_rc = retention_error::STALE;
        retention_error mutate_rc = retention_error::STALE;
        auto mutate = [&]() {
            switch (operation) {
                case 0: return wrapper_tx.add_batch(binding(2500 + operation, wrapper_owner));
                case 1: return wrapper_tx.add_owner(owner_capability(2500 + operation, wrapper_owner));
                case 2: return wrapper_tx.add_table({ static_cast<uint64_t>(2300 + operation), 1, 0, wrapper_table });
                case 3: return wrapper_tx.set_terminal(0, std::make_shared<test_terminal>(child_ready, child_waits));
                case 4: return wrapper_tx.note_submission(0, submit_outcome::UNKNOWN);
                default: {
                    return wrapper_tx.set_quiescence_proof(
                        1, queue_quiescence_test_factory::mint(wrapper_proof_state, drain_ready_callback,
                                                               drain_wait_callback));
                }
            }
        };
        std::thread aborter([&] {
            while (!wrapper_go.load(std::memory_order_acquire)) std::this_thread::yield();
            abort_rc = wrapper_tx.abort_partial();
        });
        std::thread mutator([&] {
            while (!wrapper_go.load(std::memory_order_acquire)) std::this_thread::yield();
            mutate_rc = mutate();
        });
        wrapper_go.store(true, std::memory_order_release);
        aborter.join();
        mutator.join();
        require(abort_rc == retention_error::OK &&
                    (mutate_rc == retention_error::OK || mutate_rc == retention_error::BUSY) &&
                    mutate() == retention_error::BUSY && wrapper_registry.size() == 0,
                "successful abort race left wrapper mutable or dereferenced detached record");
    }

    // Commit and simultaneous aborts share the same protocol. A moved alias
    // may be destroyed concurrently; its destructor joins the control-state
    // transition rather than destroying transaction ownership independently.
    graph_retention_registry control_registry;
    fixture                  control_fixture(control_registry);
    auto                     control_tx = begin_tx(control_fixture);
    control_tx.mark_finalized();
    std::atomic<bool> control_go{ false };
    retention_error abort_a = retention_error::STALE;
    retention_error abort_b = retention_error::STALE;
    retention_error commit_race = retention_error::STALE;
    std::thread abort_one([&] {
        while (!control_go.load(std::memory_order_acquire)) std::this_thread::yield();
        abort_a = control_tx.abort_partial();
    });
    std::thread abort_two([&] {
        while (!control_go.load(std::memory_order_acquire)) std::this_thread::yield();
        abort_b = control_tx.abort_partial();
    });
    std::thread committer([&] {
        while (!control_go.load(std::memory_order_acquire)) std::this_thread::yield();
        commit_race = control_tx.commit();
    });
    control_go.store(true, std::memory_order_release);
    abort_one.join();
    abort_two.join();
    committer.join();
    require((abort_a == retention_error::OK || abort_b == retention_error::OK) &&
                (abort_a == retention_error::OK || abort_a == retention_error::BUSY) &&
                (abort_b == retention_error::OK || abort_b == retention_error::BUSY) &&
                (commit_race == retention_error::INCOMPLETE_TERMINALS || commit_race == retention_error::BUSY) &&
                control_registry.size() == 0 &&
                control_tx.add_owner(owner_capability(2600, std::make_shared<int>(2600))) == retention_error::BUSY,
            "commit/simultaneous-abort control race failed");

    graph_retention_registry move_registry;
    fixture                  move_fixture(move_registry);
    auto move_source = std::make_unique<graph_recording_transaction>(begin_tx(move_fixture));
    auto move_target = std::make_unique<graph_recording_transaction>(std::move(*move_source));
    require(move_source->abort_partial() == retention_error::BUSY,
            "moved-from wrapper retained transaction ownership");
    move_source.reset();
    std::thread moved_destroyer([&] { move_target.reset(); });
    moved_destroyer.join();
    require(move_registry.size() == 0, "moved transaction destructor did not settle shared control");

    // Re-begin atomically replaces wrapper control even across independent
    // fixtures. The old transaction is settled outside the handle lock and
    // both registries/lifecycles return to baseline.
    graph_retention_registry rebegin_old_registry;
    fixture                  rebegin_old_fixture(rebegin_old_registry);
    graph_retention_registry rebegin_new_registry;
    fixture                  rebegin_new_fixture(rebegin_new_registry);
    graph_recording_transaction rebegin_tx = begin_tx(rebegin_old_fixture);
    const auto rebegin_old_key = rebegin_tx.key();
    require(graph_recording_transaction::begin(
                rebegin_new_registry, rebegin_new_fixture.execution, rebegin_new_fixture.context,
                rebegin_new_fixture.session, rebegin_new_fixture.reset, token(), &rebegin_tx) == retention_error::OK &&
                rebegin_old_registry.size() == 0 && rebegin_new_registry.size() == 1,
            "cross-fixture re-begin did not settle old registry");
    epoch_snapshot rebegin_old_epoch{};
    require(rebegin_old_fixture.execution.child_extract_epoch(
                rebegin_old_fixture.context, rebegin_old_fixture.session, rebegin_old_fixture.reset,
                rebegin_old_key.epoch, token(), &rebegin_old_epoch) == error::STALE &&
                rebegin_tx.abort_partial() == retention_error::OK && rebegin_new_registry.size() == 0,
            "cross-fixture re-begin did not restore lifecycle/registry baselines");

    // If old cleanup cannot settle, begin rolls back and detaches the newly
    // installed control. Callback state is declared before fixtures so it
    // outlives every retained proof under normal ASAN destruction ordering.
    std::atomic<bool>     rebegin_drain_ready{ true };
    std::atomic<bool>     rebegin_drain_succeeds{ false };
    std::atomic<unsigned> rebegin_drain_waits{ 0 };
    auto rebegin_drain_state = std::make_shared<drain_state>(
        drain_state{ &rebegin_drain_ready, &rebegin_drain_succeeds, &rebegin_drain_waits });
    graph_retention_registry rebegin_fail_old_registry;
    fixture                  rebegin_fail_old_fixture(rebegin_fail_old_registry);
    graph_retention_registry rebegin_fail_new_registry;
    fixture                  rebegin_fail_new_fixture(rebegin_fail_new_registry);
    graph_recording_transaction rebegin_fail_tx = begin_tx(rebegin_fail_old_fixture);
    const auto rebegin_fail_old_key = rebegin_fail_tx.key();
    require(rebegin_fail_tx.note_submission(1, submit_outcome::UNKNOWN) == retention_error::OK &&
                rebegin_fail_tx.set_quiescence_proof(
                    1, queue_quiescence_test_factory::mint(rebegin_drain_state, drain_ready_callback,
                                                           drain_wait_callback)) == retention_error::OK,
            "failing re-begin old transaction setup failed");
    require(graph_recording_transaction::begin(
                rebegin_fail_new_registry, rebegin_fail_new_fixture.execution, rebegin_fail_new_fixture.context,
                rebegin_fail_new_fixture.session, rebegin_fail_new_fixture.reset, token(), &rebegin_fail_tx) ==
                    retention_error::MISSING_QUIESCENCE_PROOF &&
                rebegin_fail_new_registry.size() == 0 && rebegin_fail_old_registry.size() == 1 &&
                rebegin_fail_tx.add_owner(owner_capability(2700, std::make_shared<int>(2700))) ==
                    retention_error::BUSY,
            "failed re-begin leaked wrapper/new transaction or dropped old quarantine");
    rebegin_drain_succeeds.store(true, std::memory_order_release);
    require(rebegin_fail_old_registry.abort_partial(rebegin_fail_old_key) == retention_error::OK &&
                rebegin_fail_old_registry.size() == 0,
            "failed re-begin durable old quarantine was not retryable");

    // A failed partial drain remains quarantined with all owners and can be
    // retried after the exact queue proof succeeds.
    graph_retention_registry abort_registry;
    fixture                  abort_fixture(abort_registry);
    auto                     abort_tx    = begin_tx(abort_fixture);
    const auto               abort_key   = abort_tx.key();
    auto                     abort_owner = std::make_shared<int>(123);
    std::weak_ptr<int>       abort_weak  = abort_owner;
    std::atomic<bool>        abort_drain_ready{ true };
    std::atomic<bool>        abort_drain_succeeds{ false };
    std::atomic<unsigned>    abort_drain_waits{ 0 };
    auto abort_drain_state = std::make_shared<drain_state>(
        drain_state{ &abort_drain_ready, &abort_drain_succeeds, &abort_drain_waits });
    require(abort_tx.add_owner(owner_capability(1123, abort_owner, 1)) == retention_error::OK &&
                abort_tx.note_submission(1, submit_outcome::UNKNOWN) == retention_error::OK &&
                abort_tx.set_quiescence_proof(
                    1, queue_quiescence_test_factory::mint(abort_drain_state, drain_ready_callback,
                                                           drain_wait_callback)) == retention_error::OK,
            "partial drain retry setup failed");
    abort_owner.reset();
    require(abort_tx.abort_partial() == retention_error::MISSING_QUIESCENCE_PROOF &&
                abort_registry.snapshot(abort_key) && !abort_weak.expired(),
            "failed partial drain did not retain quarantine and owners");
    abort_drain_succeeds.store(true, std::memory_order_release);
    require(abort_tx.abort_partial() == retention_error::OK && abort_registry.size() == 0 && abort_weak.expired() &&
                abort_drain_waits.load() == 2,
            "successful partial drain retry did not erase quarantine and owners");
    epoch_snapshot abort_removed{};
    require(abort_fixture.execution.child_extract_epoch(abort_fixture.context, abort_fixture.session,
                                                        abort_fixture.reset, abort_key.epoch, token(),
                                                        &abort_removed) == error::STALE,
            "successful partial drain retry left lifecycle epoch");

    // Even if the transaction object is destroyed while drain is unsettled,
    // registry adoption preserves the last owners and exposes an exact retry.
    graph_retention_registry destructor_registry;
    fixture                  destructor_fixture(destructor_registry);
    graph_owner_key          destructor_key{};
    std::atomic<bool>        destructor_ready{ true };
    std::atomic<bool>        destructor_succeeds{ false };
    std::atomic<unsigned>    destructor_waits{ 0 };
    auto destructor_state = std::make_shared<drain_state>(
        drain_state{ &destructor_ready, &destructor_succeeds, &destructor_waits });
    auto destructor_owner = std::make_shared<int>(124);
    std::weak_ptr<int> destructor_weak = destructor_owner;
    {
        auto unsettled = begin_tx(destructor_fixture);
        destructor_key = unsettled.key();
        require(unsettled.add_owner(owner_capability(1124, destructor_owner, 1)) == retention_error::OK &&
                    unsettled.note_submission(1, submit_outcome::UNKNOWN) == retention_error::OK &&
                    unsettled.set_quiescence_proof(
                        1, queue_quiescence_test_factory::mint(destructor_state, drain_ready_callback,
                                                               drain_wait_callback)) == retention_error::OK,
                "destructor quarantine setup failed");
        destructor_owner.reset();
    }
    require(destructor_registry.snapshot(destructor_key) && !destructor_weak.expired(),
            "unsettled destructor destroyed last durable owner");
    destructor_succeeds.store(true, std::memory_order_release);
    require(destructor_registry.abort_partial(destructor_key) == retention_error::OK &&
                destructor_registry.size() == 0 && destructor_weak.expired(),
            "registry partial-abort retry did not settle destructor quarantine");

    // Deterministic assembly allocation failures prove the record is adopted
    // before mutation: prior owners/tables/terminals remain registry-owned and
    // the same transaction can drain and retry its abort.
    for (retention_fault fault : { retention_fault::ADD_OWNER_ONCE, retention_fault::ADD_TABLE_ONCE,
                                   retention_fault::ADD_TERMINAL_ONCE }) {
        graph_retention_registry oom_registry(fault);
        fixture                  oom_fixture(oom_registry);
        auto                     oom_tx = begin_tx(oom_fixture);
        const auto               oom_key = oom_tx.key();
        auto                     durable = std::make_shared<int>(300 + static_cast<int>(fault));
        std::weak_ptr<int>       durable_weak = durable;
        require(oom_tx.add_batch(binding(1300 + static_cast<int>(fault), durable)) == retention_error::OK,
                "durable pre-failpoint owner setup failed");
        retention_error injected = retention_error::OK;
        if (fault == retention_fault::ADD_OWNER_ONCE) {
            injected = oom_tx.add_owner(owner_capability(1400, durable));
        } else if (fault == retention_fault::ADD_TABLE_ONCE) {
            auto owner = graph_private_table_owner::create(
                oom_key, 1500 + static_cast<int>(fault), 1, 0, { owner_capability(1501, durable) });
            injected = oom_tx.add_table({ 1500 + static_cast<uint64_t>(fault), 1, 0, owner });
        } else {
            injected = oom_tx.set_terminal(0, std::make_shared<test_terminal>(child_ready, child_waits));
        }
        durable.reset();
        auto oom_snapshot = oom_registry.snapshot(oom_key);
        require(injected == retention_error::BUSY && oom_snapshot && !durable_weak.expired(),
                "assembly OOM lost pre-adopted durable owner");
        require((fault != retention_fault::ADD_OWNER_ONCE || oom_snapshot->generic_owners.size() == 1) &&
                    (fault != retention_fault::ADD_TABLE_ONCE || oom_snapshot->tables.size() == 1) &&
                    (fault != retention_fault::ADD_TERMINAL_ONCE || oom_snapshot->terminals.size() == 1),
                "post-insertion OOM did not preserve the inserted durable record");
        oom_snapshot.reset();
        require(oom_tx.abort_partial() == retention_error::OK && oom_registry.size() == 0 && durable_weak.expired(),
                "assembly OOM abort did not restore registry baseline");
    }
    graph_retention_registry adopt_fault(retention_fault::ADOPT_ONCE);
    fixture                  adopt_fixture(adopt_fault);
    graph_recording_transaction rejected_adoption;
    require(graph_recording_transaction::begin(adopt_fault, adopt_fixture.execution, adopt_fixture.context,
                                                adopt_fixture.session, adopt_fixture.reset, token(),
                                                &rejected_adoption) == retention_error::BUSY &&
                adopt_fault.size() == 0,
            "adoption failure left retention or lifecycle recording poisoned");
    auto adopted_retry = begin_tx(adopt_fixture);
    require(adopted_retry.abort_partial() == retention_error::OK,
            "adoption failure lifecycle was not retryable");

    graph_retention_registry retention;
    fixture                  f(retention);
    std::atomic<bool>        ready{ true };
    std::atomic<unsigned>    waits{ 0 };

    auto                        first         = begin_tx(f);
    const auto                  first_key     = first.key();
    auto                        raw           = std::make_shared<int>(7);
    std::shared_ptr<const void> raw_handle    = raw;
    auto                        first_binding = binding(101, raw_handle);
    require(first.add_batch(first_binding) == retention_error::OK &&
                first.note_submission(0, submit_outcome::SUBMITTED) == retention_error::OK &&
                first.set_terminal(0, std::make_shared<test_terminal>(ready, waits, &retention)) == retention_error::OK,
            "first resources failed");
    auto table = graph_private_table_owner::create(first_key, 501, 77, 0, { owner_capability(101, raw_handle) });
    require(first.add_table({ 501, 77, 0, table }) == retention_error::OK, "first table failed");
    first.mark_finalized();
    require(first.commit() == retention_error::OK, "first commit failed");
    auto snap = retention.snapshot(first_key);
    require(snap && snap->find_batch(first_binding.identity), "exact binding absent");
    auto different = first_binding.identity;
    different.allocation_id++;
    require(!snap->find_batch(different), "raw pointer incorrectly used as identity");

    std::weak_ptr<const void>                      weak_operand = raw_handle;
    std::weak_ptr<const graph_private_table_owner> weak_table   = table;
    raw.reset();
    raw_handle.reset();
    table.reset();
    snap.reset();
    first_binding = binding(999, std::make_shared<int>(999));
    require(!weak_operand.expired() && !weak_table.expired(), "installed owners dropped early");

    auto       replacement       = begin_tx(f);
    const auto replacement_key   = replacement.key();
    auto       replacement_owner = std::make_shared<int>(8);
    require(replacement.add_batch(binding(102, replacement_owner)) == retention_error::OK &&
                replacement.note_submission(0, submit_outcome::SUBMITTED) == retention_error::OK &&
                replacement.set_terminal(0, std::make_shared<test_terminal>(ready, waits)) == retention_error::OK,
            "replacement resources failed");
    replacement.mark_finalized();
    require(replacement.commit() == retention_error::OK && retention.active(f.context) == replacement_key,
            "replacement visibility failed");
    published_graph_token replacement_token;
    InvocationId          managed_invocation{};
    require(retention.acquire_published_token(replacement_key, &replacement_token) == retention_error::OK &&
                retention.begin_invocation(replacement_token, &managed_invocation) == retention_error::OK &&
                retention.finish_invocation(replacement_token, managed_invocation) == retention_error::OK,
            "published-token invocation handshake failed");
    published_graph_token forged_token;
    require(retention.begin_invocation(forged_token, &managed_invocation) == retention_error::STALE,
            "forged publication token admitted invocation");
    require(retention.retire_exact(first_key) == retention_error::OK && retention.active(f.context) == replacement_key,
            "old exact retirement cleared replacement");
    epoch_snapshot epoch{};
    require(f.execution.extract_epoch(f.context, f.session, f.reset, first_key.epoch, token(), &epoch) == error::OK &&
                epoch.state == epoch_phase::RETIRED,
            "retention erased before lifecycle RETIRED");
    require(weak_operand.expired() && weak_table.expired(), "retired owners survived");

    // Finalization refuses incomplete device coverage.
    auto incomplete       = begin_tx(f);
    auto incomplete_owner = std::make_shared<int>(9);
    require(incomplete.add_batch(binding(103, incomplete_owner)) == retention_error::OK &&
                incomplete.note_submission(0, submit_outcome::SUBMITTED) == retention_error::OK,
            "incomplete setup failed");
    incomplete.mark_finalized();
    require(incomplete.commit() == retention_error::INCOMPLETE_TERMINALS, "commit accepted incomplete terminal set");
    require(incomplete.set_terminal(0, std::make_shared<test_terminal>(ready, waits)) == retention_error::OK &&
                incomplete.rollback() == retention_error::OK,
            "incomplete transaction could not be repaired/retired");

    graph_retention_registry extra_registry;
    fixture                  extra_fixture(extra_registry);
    auto                     extra       = begin_tx(extra_fixture);
    auto                     extra_owner = std::make_shared<int>(90);
    require(extra.add_batch(binding(190, extra_owner)) == retention_error::OK &&
                extra.note_submission(0, submit_outcome::SUBMITTED) == retention_error::OK &&
                extra.set_terminal(0, std::make_shared<test_terminal>(ready, waits)) == retention_error::OK &&
                extra.set_terminal(1, std::make_shared<test_terminal>(ready, waits)) == retention_error::OK,
            "extra terminal setup failed");
    extra.mark_finalized();
    require(extra.commit() == retention_error::INCOMPLETE_TERMINALS,
            "terminal superset accepted instead of exact device set");

    // Generic workspace/intermediate owners contribute their canonical device
    // to the exact terminal set, including secondary devices.
    graph_retention_registry mixed_registry;
    fixture                  mixed_fixture(mixed_registry);
    auto                     mixed       = begin_tx(mixed_fixture);
    auto                     mixed_owner = std::make_shared<int>(91);
    require(mixed.add_batch(binding(191, mixed_owner, 0)) == retention_error::OK &&
                mixed.add_owner(owner_capability(192, mixed_owner, 1)) == retention_error::OK &&
                mixed.note_submission(0, submit_outcome::SUBMITTED) == retention_error::OK &&
                mixed.set_terminal(0, std::make_shared<test_terminal>(ready, waits)) == retention_error::OK,
            "mixed-device owner setup failed");
    mixed.mark_finalized();
    require(mixed.commit() == retention_error::INCOMPLETE_TERMINALS,
            "secondary generic owner did not require its exact terminal");
    require(mixed.set_terminal(1, std::make_shared<test_terminal>(ready, waits)) == retention_error::OK &&
                mixed.commit() == retention_error::OK && mixed_registry.retire_exact(mixed.key()) == retention_error::OK,
            "mixed primary/secondary owner retirement failed");

    // UNKNOWN cannot use a ready event as proof. Failed drain remains retained
    // and retryable; successful queue quiescence permits exact retirement.
    auto       unknown     = begin_tx(f);
    const auto unknown_key = unknown.key();
    require(unknown.note_submission(1, submit_outcome::UNKNOWN) == retention_error::OK &&
                unknown.set_terminal(1, std::make_shared<test_terminal>(ready, waits)) == retention_error::OK,
            "unknown setup failed");
    unknown.mark_finalized();
    require(unknown.commit() == retention_error::MISSING_QUIESCENCE_PROOF, "arbitrary ready event discharged UNKNOWN");
    std::atomic<bool>     drain_ready{ true };
    std::atomic<bool>     drain_succeeds{ false };
    std::atomic<unsigned> drain_waits{ 0 };
    auto drain_state_owner = std::make_shared<drain_state>(drain_state{ &drain_ready, &drain_succeeds, &drain_waits });
    require(unknown.set_quiescence_proof(
                1, queue_quiescence_test_factory::mint(drain_state_owner, drain_ready_callback, drain_wait_callback)) ==
                    retention_error::OK &&
                unknown.commit() == retention_error::PENDING,
            "failed drain did not retain quarantine");
    require(retention.snapshot(unknown_key) && retention.active(f.context) == replacement_key,
            "UNKNOWN quarantine became externally active or disappeared");
    drain_succeeds.store(true, std::memory_order_release);
    require(unknown.commit() == retention_error::OK && !retention.snapshot(unknown_key) && drain_waits.load() == 2,
            "UNKNOWN drain retry did not retire exactly");

    // Prepare failure preserves transaction owners and can be retried.
    graph_retention_registry prepare_fault(retention_fault::PREPARE_ONCE);
    fixture                  pf(prepare_fault);
    auto                     prepare_tx    = begin_tx(pf);
    auto                     prepare_owner = std::make_shared<int>(10);
    std::weak_ptr<int>       prepare_weak  = prepare_owner;
    require(prepare_tx.add_batch(binding(201, prepare_owner)) == retention_error::OK &&
                prepare_tx.note_submission(0, submit_outcome::SUBMITTED) == retention_error::OK &&
                prepare_tx.set_terminal(0, std::make_shared<test_terminal>(ready, waits)) == retention_error::OK,
            "prepare fault setup failed");
    prepare_owner.reset();
    prepare_tx.mark_finalized();
    require(prepare_tx.commit() == retention_error::BUSY && !prepare_weak.expired() &&
                !prepare_fault.active(pf.context).epoch.value,
            "prepare failure dropped owner or exposed active");
    require(prepare_tx.commit() == retention_error::OK, "prepare failure was not retryable");

    // Publish failure immediately retires the activated epoch; it can never
    // issue a managed invocation without a publication token.
    graph_retention_registry publish_fault(retention_fault::PUBLISH_ONCE);
    fixture                  pubf(publish_fault);
    auto                     publish_tx    = begin_tx(pubf);
    auto                     publish_owner = std::make_shared<int>(11);
    std::atomic<bool>        publish_ready{ false };
    std::weak_ptr<int>       publish_weak = publish_owner;
    require(
        publish_tx.add_batch(binding(202, publish_owner)) == retention_error::OK &&
            publish_tx.note_submission(0, submit_outcome::SUBMITTED) == retention_error::OK &&
            publish_tx.set_terminal(0, std::make_shared<test_terminal>(publish_ready, waits)) == retention_error::OK,
        "publish fault setup failed");
    publish_owner.reset();
    publish_tx.mark_finalized();
    require(publish_tx.commit() == retention_error::PENDING && !publish_weak.expired() &&
                publish_fault.active(pubf.context).epoch.value == 0,
            "delayed publish failure did not retain pending owner without visibility");
    published_graph_token unpublished_token;
    InvocationId          unpublished_invocation{};
    require(publish_fault.acquire_published_token(publish_tx.key(), &unpublished_token) == retention_error::STALE &&
                publish_fault.begin_invocation(unpublished_token, &unpublished_invocation) == retention_error::STALE,
            "publish failure admitted token invocation");
    epoch_snapshot publish_epoch{};
    require(pubf.execution.extract_epoch(pubf.context, pubf.session, pubf.reset, publish_tx.key().epoch, token(),
                                         &publish_epoch) == error::OK &&
                publish_epoch.state == epoch_phase::RETIRING &&
                pubf.execution.begin_invocation(pubf.context, pubf.session, pubf.reset, publish_tx.key().epoch, token(),
                                                &unpublished_invocation) == error::BUSY,
            "pending publish failure left raw lifecycle invocation ACTIVE");
    publish_ready.store(true, std::memory_order_release);
    require(publish_tx.commit() == retention_error::OK && publish_weak.expired() &&
                pubf.execution.extract_epoch(pubf.context, pubf.session, pubf.reset, publish_tx.key().epoch, token(),
                                             &publish_epoch) == error::OK &&
                publish_epoch.state == epoch_phase::RETIRED,
            "released publish failure did not complete exact retirement");

    // Lifecycle activation failure leaves the prepared owner quarantined and
    // intact; exact rollback remains possible without external visibility.
    graph_retention_registry activation_failure;
    fixture                  af(activation_failure);
    auto                     activation_tx    = begin_tx(af);
    auto                     activation_owner = std::make_shared<int>(111);
    std::weak_ptr<int>       activation_weak  = activation_owner;
    require(activation_tx.add_batch(binding(211, activation_owner)) == retention_error::OK &&
                activation_tx.note_submission(0, submit_outcome::SUBMITTED) == retention_error::OK &&
                activation_tx.set_terminal(0, std::make_shared<test_terminal>(ready, waits)) == retention_error::OK,
            "activation failure setup failed");
    activation_owner.reset();
    activation_tx.mark_finalized();
    require(
        af.execution.rollback_record(af.context, af.session, af.reset, activation_tx.key().epoch, token()) == error::OK,
        "activation failure injection failed");
    require(activation_tx.commit() == retention_error::LIFECYCLE_ERROR && activation_weak.expired() &&
                activation_failure.active(af.context).epoch.value == 0,
            "activation failure did not retire exactly or leaked visibility");

    // Shared-table conflict quarantines an intact retryable transaction.
    auto table_a = begin_tx(f);
    auto a_owner = std::make_shared<int>(12);
    auto a_table = graph_private_table_owner::create(table_a.key(), 777, 1, 0, { owner_capability(301, a_owner) });
    require(table_a.add_table({ 777, 1, 0, a_table }) == retention_error::OK &&
                table_a.note_submission(0, submit_outcome::SUBMITTED) == retention_error::OK &&
                table_a.set_terminal(0, std::make_shared<test_terminal>(ready, waits)) == retention_error::OK,
            "table A setup failed");
    table_a.mark_finalized();
    require(table_a.commit() == retention_error::OK, "table A commit failed");
    require(retention.begin_invocation(replacement_token, &managed_invocation) == retention_error::STALE,
            "stale replacement publication token admitted invocation");

    auto               table_b = begin_tx(f);
    auto               b_owner = std::make_shared<int>(13);
    std::weak_ptr<int> b_weak  = b_owner;
    auto b_table = graph_private_table_owner::create(table_b.key(), 777, 1, 0, { owner_capability(302, b_owner) });
    require(table_b.add_table({ 777, 1, 0, b_table }) == retention_error::OK &&
                table_b.note_submission(0, submit_outcome::SUBMITTED) == retention_error::OK &&
                table_b.set_terminal(0, std::make_shared<test_terminal>(ready, waits)) == retention_error::OK,
            "table B setup failed");
    b_owner.reset();
    b_table.reset();
    table_b.mark_finalized();
    require(table_b.commit() == retention_error::BUSY && !b_weak.expired(),
            "table conflict dropped intact transaction");
    require(retention.retire_exact(table_a.key()) == retention_error::OK && table_b.commit() == retention_error::OK,
            "table conflict was not retryable after exact owner retirement");

    // One-shot retirement setup failure retains quarantine and exact owners.
    graph_retention_registry retire_fault(retention_fault::RETIRE_SETUP_ONCE);
    fixture                  rf(retire_fault);
    auto                     retire_tx    = begin_tx(rf);
    auto                     retire_owner = std::make_shared<int>(14);
    std::weak_ptr<int>       retire_weak  = retire_owner;
    require(retire_tx.add_batch(binding(401, retire_owner)) == retention_error::OK &&
                retire_tx.note_submission(0, submit_outcome::SUBMITTED) == retention_error::OK &&
                retire_tx.set_terminal(0, std::make_shared<test_terminal>(ready, waits)) == retention_error::OK,
            "retire fault setup failed");
    retire_owner.reset();
    retire_tx.mark_finalized();
    require(retire_tx.commit() == retention_error::OK &&
                retire_fault.retire_exact(retire_tx.key()) == retention_error::BUSY && !retire_weak.expired(),
            "retirement setup failure dropped quarantine");
    require(retire_fault.retire_exact(retire_tx.key()) == retention_error::OK && retire_weak.expired(),
            "retirement setup failure was not retryable");

    // Terminal bookkeeping uses a fixed device bitset. A deterministic refusal
    // before lifecycle attachment cannot terminate or lose owners; retry
    // attaches once and the post-attachment bookkeeping itself cannot allocate.
    graph_retention_registry terminal_bookkeep_fault(retention_fault::TERMINAL_BOOKKEEP_ONCE);
    fixture                  tbf(terminal_bookkeep_fault);
    auto                     terminal_bookkeep_tx = begin_tx(tbf);
    auto                     terminal_bookkeep_owner = std::make_shared<int>(15);
    std::weak_ptr<int>       terminal_bookkeep_weak = terminal_bookkeep_owner;
    require(terminal_bookkeep_tx.add_batch(binding(402, terminal_bookkeep_owner)) == retention_error::OK &&
                terminal_bookkeep_tx.note_submission(0, submit_outcome::SUBMITTED) == retention_error::OK &&
                terminal_bookkeep_tx.set_terminal(0, std::make_shared<test_terminal>(ready, waits)) ==
                    retention_error::OK,
            "terminal bookkeeping fault setup failed");
    terminal_bookkeep_owner.reset();
    terminal_bookkeep_tx.mark_finalized();
    require(terminal_bookkeep_tx.commit() == retention_error::OK &&
                terminal_bookkeep_fault.retire_exact(terminal_bookkeep_tx.key()) == retention_error::BUSY &&
                !terminal_bookkeep_weak.expired(),
            "terminal bookkeeping fault terminated or dropped owner");
    require(terminal_bookkeep_fault.retire_exact(terminal_bookkeep_tx.key()) == retention_error::OK &&
                terminal_bookkeep_weak.expired(),
            "terminal bookkeeping fault was not exactly retryable");

    // Every durable identity component is validated, including range overflow
    // and correlation with the typed allocation owner.
    const mmid_operand_identity valid_identity{ 501, 1, 9, 0, 16, 64, 1 };
    auto                        invalid = valid_identity;
    invalid.layout_id                   = 0;
    require_bad_identity(invalid, 501, "zero layout accepted");
    invalid            = valid_identity;
    invalid.generation = 2;
    require_bad_identity(invalid, 501, "cross-generation owner accepted");
    invalid        = valid_identity;
    invalid.device = static_cast<int>(execution::max_devices);
    require_bad_identity(invalid, 501, "invalid device accepted");
    invalid           = valid_identity;
    invalid.byte_size = 0;
    require_bad_identity(invalid, 501, "empty range accepted");
    invalid             = valid_identity;
    invalid.byte_offset = std::numeric_limits<size_t>::max();
    invalid.byte_size   = 2;
    require_bad_identity(invalid, 501, "overflowing range accepted");
    invalid             = valid_identity;
    invalid.byte_offset = 4080;
    invalid.byte_size   = 32;
    require_bad_identity(invalid, 501, "range outside canonical owner extent accepted");
    invalid            = valid_identity;
    invalid.occurrence = 0;
    require_bad_identity(invalid, 501, "zero occurrence accepted");
    require_bad_identity(valid_identity, 999, "cross-owner allocation identity accepted");

    // Concurrent replay/invalidation: successful replays finish exactly; once
    // invalidation makes the epoch RETIRING, stale token starts are rejected.
    graph_retention_registry invalidate_registry;
    fixture                  invalidate_fixture(invalidate_registry);
    auto                     invalidate_tx = begin_tx(invalidate_fixture);
    auto                     invalidate_owner = std::make_shared<int>(16);
    require(invalidate_tx.add_batch(binding(403, invalidate_owner)) == retention_error::OK &&
                invalidate_tx.note_submission(0, submit_outcome::SUBMITTED) == retention_error::OK &&
                invalidate_tx.set_terminal(0, std::make_shared<test_terminal>(ready, waits)) == retention_error::OK,
            "replay/invalidate setup failed");
    invalidate_tx.mark_finalized();
    require(invalidate_tx.commit() == retention_error::OK, "replay/invalidate publication failed");
    published_graph_token invalidate_token;
    require(invalidate_registry.acquire_published_token(invalidate_tx.key(), &invalidate_token) == retention_error::OK,
            "replay/invalidate token acquisition failed");
    std::atomic<bool> invalidate_go{ false };
    std::atomic<unsigned> replay_successes{ 0 };
    std::vector<std::thread> replay_threads;
    for (int i = 0; i < 8; ++i) {
        replay_threads.emplace_back([&] {
            while (!invalidate_go.load(std::memory_order_acquire)) std::this_thread::yield();
            InvocationId replay{};
            if (invalidate_registry.begin_invocation(invalidate_token, &replay) == retention_error::OK) {
                replay_successes.fetch_add(1, std::memory_order_relaxed);
                require(invalidate_registry.finish_invocation(invalidate_token, replay) == retention_error::OK,
                        "successful concurrent replay did not finish exactly");
            }
        });
    }
    invalidate_go.store(true, std::memory_order_release);
    auto invalidate_rc = invalidate_registry.retire_exact(invalidate_tx.key());
    for (auto & thread : replay_threads) thread.join();
    if (invalidate_rc == retention_error::PENDING) {
        invalidate_rc = invalidate_registry.retire_exact(invalidate_tx.key());
    }
    InvocationId stale_after_invalidate{};
    require(invalidate_rc == retention_error::OK && replay_successes.load() <= replay_threads.size() &&
                invalidate_registry.begin_invocation(invalidate_token, &stale_after_invalidate) ==
                    retention_error::STALE,
            "concurrent replay/invalidation admitted stale token or lost invocation");

    // Real concurrent readers race an exact delayed retirement under TSAN.
    ready.store(false, std::memory_order_release);
    const auto               concurrent_key = table_b.key();
    std::atomic<bool>        stop{ false };
    std::vector<std::thread> readers;
    for (int i = 0; i < 6; ++i) {
        readers.emplace_back([&] {
            while (!stop.load(std::memory_order_acquire)) {
                (void) retention.snapshot(concurrent_key);
                (void) retention.active(f.context);
                (void) retention.size();
            }
        });
    }
    require(retention.retire_exact(concurrent_key) == retention_error::PENDING,
            "delayed concurrent terminal did not remain pending");
    ready.store(true, std::memory_order_release);
    require(retention.retire_exact(concurrent_key) == retention_error::OK, "concurrent exact retirement failed");
    stop.store(true, std::memory_order_release);
    for (auto & reader : readers) {
        reader.join();
    }

    auto              two_retire = begin_tx(f);
    auto              two_owner  = std::make_shared<int>(902);
    std::atomic<bool> retire_entered{ false };
    std::atomic<bool> retire_release{ false };
    require(two_retire.add_batch(binding(902, two_owner)) == retention_error::OK &&
                two_retire.note_submission(0, submit_outcome::SUBMITTED) == retention_error::OK &&
                two_retire.set_terminal(0, std::make_shared<blocking_terminal>(retire_entered, retire_release)) ==
                    retention_error::OK,
            "two-retirer setup failed");
    two_retire.mark_finalized();
    require(two_retire.commit() == retention_error::OK, "two-retirer commit failed");
    retention_error first_retire = retention_error::STALE;
    std::thread     retire_thread([&] { first_retire = retention.retire_exact(two_retire.key()); });
    while (!retire_entered.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    retention_error   second_retire = retention_error::OK;
    std::atomic<bool> second_started{ false };
    std::thread       second_retire_thread([&] {
        second_started.store(true, std::memory_order_release);
        second_retire = retention.retire_exact(two_retire.key());
    });
    while (!second_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    retire_release.store(true, std::memory_order_release);
    retire_thread.join();
    second_retire_thread.join();
    require(first_retire == retention_error::OK && second_retire == retention_error::STALE &&
                retention.retire_exact(two_retire.key()) == retention_error::STALE,
            "single retirer did not serialize waiting stale followers");

    // Last-owner destruction and terminal waits both re-enter registry unlocked.
    std::atomic<unsigned> drops{ 0 };
    auto                  probe       = begin_tx(f);
    auto                  probe_owner = std::make_shared<lock_probe>(retention, drops);
    require(probe.add_batch(binding(901, probe_owner)) == retention_error::OK &&
                probe.note_submission(0, submit_outcome::SUBMITTED) == retention_error::OK &&
                probe.set_terminal(0, std::make_shared<test_terminal>(ready, waits, &retention)) == retention_error::OK,
            "lock probe setup failed");
    probe.mark_finalized();
    require(probe.commit() == retention_error::OK, "lock probe commit failed");
    probe_owner.reset();
    require(retention.retire_exact(probe.key()) == retention_error::OK && drops.load() == 1,
            "owner destruction ran under retention lock");

    std::cout << "moe graph retention: ok\n";
}
