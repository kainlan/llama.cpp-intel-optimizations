#include "model-lifecycle.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <optional>
#include <thread>
#include <vector>

using namespace ggml_sycl::lifecycle;

static void require(bool value, const char * message) {
    if (!value) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

static ModelToken commit_one(Registry & registry) {
    const auto begin = registry.begin_outer();
    require(begin.code == error::OK, "begin failed");
    const auto end = registry.end(begin.txn, true);
    require(end.code == error::OK && end.committed, "commit failed");
    return end.token;
}

static void candidate_binding_case(bool inner_success) {
    Registry   registry;
    const auto begin = registry.begin_outer();
    registry.bind_candidate(begin.txn);
    registry.bind_candidate(begin.txn);  // wrapper pushes before depth entry
    require(registry.enter_nested(begin.txn) == error::OK, "nested begin failed");

    const auto inner = registry.end(begin.txn, inner_success);
    require(inner.code == (inner_success ? error::NESTED : error::POISONED), "inner result mismatch");
    registry.unbind_candidate(begin.txn);
    require(registry.bound_candidate() == begin.txn, "inner end discarded outer candidate binding");

    auto       terminal = std::async(std::launch::async, [&] { return registry.end(begin.txn, true); });
    const auto result   = terminal.get();
    require(result.committed == inner_success, "outer poison/commit result mismatch");
    require(registry.bound_candidate().value == 0, "cross-thread terminal end did not invalidate keyed binding");
}

static void candidate_depth_failure_case() {
    Registry   registry(UINT64_MAX, 1);
    const auto begin = registry.begin_outer();
    registry.bind_candidate(begin.txn);
    registry.bind_candidate(begin.txn);
    require(registry.enter_nested(begin.txn) == error::DEPTH_OVERFLOW, "depth limit accepted nested entry");
    registry.unbind_candidate(begin.txn);
    require(registry.bound_candidate() == begin.txn, "failed nested entry discarded outer binding");
    require(!registry.end(begin.txn, true).committed, "depth failure did not poison outer transaction");
    registry.unbind_candidate(begin.txn);
}

static void durable_replay_case() {
    Registry  registry;
    LoadTxnId first_load{};
    for (size_t i = 0; i < 300; ++i) {
        const auto begin = registry.begin_outer();
        if (i == 0) {
            first_load = begin.txn;
        }
        require(registry.end(begin.txn, false).code == error::MISSING_SUCCESS, "abort failed");
    }
    require(registry.end(first_load, false).code == error::MISSING_SUCCESS, "process-lifetime abort replay was lost");

    ModelToken first_model{};
    for (size_t i = 0; i < 300; ++i) {
        const auto model = commit_one(registry);
        if (i == 0) {
            first_model = model;
        }
        require(registry.teardown(model) == error::OK, "teardown failed");
    }
    require(registry.teardown(first_model) == error::OK_ALREADY_DEAD, "process-lifetime dead-model replay was lost");
}

static void finite_id_limit_case() {
    Registry   registry(2);
    const auto first = registry.begin_outer();
    require(first.token.model.value == 1 && first.token.load.value == 1, "first finite identity mismatch");
    require(registry.end(first.txn, false).code == error::MISSING_SUCCESS, "first finite abort failed");
    const auto last = registry.begin_outer();
    require(last.token.model.value == 2 && last.token.load.value == 2, "last finite identity mismatch");
    require(registry.end(last.txn, false).code == error::MISSING_SUCCESS, "last finite abort failed");
    require(registry.begin_outer().code == error::ID_EXHAUSTED, "finite identities wrapped or reused");
    require(registry.end(first.txn, false).code == error::MISSING_SUCCESS,
            "ID exhaustion discarded durable terminal replay");
}

static void quarantine_wrapper_busy_overlap_case() {
    Registry         registry;
    quarantine_queue queue;
    const ModelToken model   = commit_one(registry);
    const auto       blocker = registry.begin_outer();
    require(blocker.code == error::OK, "blocking load begin failed");

    // Destructor wrapper order: retain the exact token first, then make LIVE
    // state non-runnable. The first reaper pass sees BUSY from the active load.
    require(queue.enqueue(model), "destructor quarantine enqueue failed");
    require(registry.defer_quarantine(model) == error::OK, "destructor quarantine defer failed");
    std::array<ModelToken, model_slot_count> first_pass{};
    require(queue.take_all(first_pass) == 1, "reaper did not take destructor token");
    require(registry.prepare_teardown(first_pass[0]).code == error::BUSY,
            "reaper teardown did not observe active-load BUSY");

    // Deterministic overlap: a second destructor enqueue occurs while the first
    // reaper pass owns its snapshot. Requeueing BUSY must deduplicate, not drop
    // or multiply, the exact owner token.
    require(queue.enqueue(model), "overlapping destructor enqueue failed");
    require(queue.enqueue(first_pass[0]), "BUSY reaper retry enqueue failed");
    require(queue.size() == 1, "BUSY destructor/reaper overlap lost or duplicated token");

    require(registry.end(blocker.txn, false).code == error::MISSING_SUCCESS, "blocking load abort failed");
    std::array<ModelToken, model_slot_count> retry_pass{};
    require(queue.take_all(retry_pass) == 1, "reaper retry token missing");
    const auto teardown = registry.prepare_teardown(retry_pass[0]);
    require(teardown.finisher && registry.finalize_teardown(teardown, true) == error::OK,
            "reaper retry did not finalize quarantined model");
}

static void live_update_ticket_saturation_case() {
    Registry registry;
    const ModelToken model = commit_one(registry);
    std::vector<live_update_ticket> held;
    held.reserve(model_slot_count);
    for (size_t i = 0; i < model_slot_count; ++i) {
        auto ticket = registry.prepare_live_update(model);
        require(ticket.active, "failed to saturate live-update ticket capacity");
        held.emplace_back(std::move(ticket));
    }
    require(registry.prepare_live_update(model).code == error::BUSY,
            "saturated live-update pool did not report BUSY");

    std::thread release_one([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        require(registry.finalize_live_update(held[0]) == error::OK,
                "failed to free transient ticket capacity");
    });
    std::optional<live_update_ticket> acquired;
    for (int wait = 0; wait < 7 && !acquired; ++wait) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1u << wait));
        auto attempt = registry.prepare_live_update(model);
        if (attempt.active) {
            acquired.emplace(std::move(attempt));
        }
    }
    release_one.join();
    require(acquired && acquired->active, "bounded backoff did not survive transient ticket exhaustion");
    require(registry.finalize_live_update(*acquired) == error::OK, "transient ticket finalize failed");
    auto replacement = registry.prepare_live_update(model);
    require(replacement.active, "failed to restore saturated capacity");

    const auto started = std::chrono::steady_clock::now();
    error persistent = error::BUSY;
    for (int wait = 0; persistent == error::BUSY && wait < 7; ++wait) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1u << wait));
        auto attempt = registry.prepare_live_update(model);
        persistent = attempt.code;
    }
    const auto elapsed = std::chrono::steady_clock::now() - started;
    require(persistent == error::BUSY, "persistent exhaustion changed result contract");
    require(elapsed < std::chrono::seconds(1), "persistent exhaustion was not bounded");

    require(registry.finalize_live_update(replacement) == error::OK, "replacement ticket finalize failed");
    for (size_t i = 1; i < held.size(); ++i) {
        require(registry.finalize_live_update(held[i]) == error::OK, "held ticket finalize failed");
    }
    require(registry.teardown(model) == error::OK, "saturation model teardown failed");
}

static void shutdown_reservation_state_machine_case() {
    Registry registry;

    const auto active = registry.begin_outer();
    require(active.code == error::OK, "active-load reservation fixture failed");
    require(registry.reserve_shutdown() == error::BUSY, "shutdown admitted ACTIVE load");
    const auto committing = registry.prepare_end(active.txn, true);
    require(committing.finisher && committing.commit, "commit reservation fixture failed");
    require(registry.reserve_shutdown() == error::BUSY, "shutdown admitted COMMITTING load");
    require(registry.finalize_end(committing, true).committed, "commit fixture finalize failed");

    const ModelToken model = committing.token;
    auto update = registry.prepare_live_update(model);
    require(update.active, "teardown-overlap update fixture failed");
    auto draining = std::async(std::launch::async, [&] { return registry.prepare_teardown(model); });
    require(draining.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout,
            "teardown did not wait in DRAINING_UPDATES");
    require(registry.reserve_shutdown() == error::BUSY, "shutdown admitted DRAINING_UPDATES model");
    require(registry.finalize_live_update(update) == error::OK, "draining update release failed");
    const auto tearing = draining.get();
    require(tearing.finisher, "teardown did not enter TEARING_DOWN");
    require(registry.reserve_shutdown() == error::BUSY, "shutdown admitted TEARING_DOWN model");
    require(registry.finalize_teardown(tearing, false) == error::EFFECT_FAILED,
            "failed teardown did not quarantine model");
    require(registry.reserve_shutdown() == error::BUSY, "shutdown admitted QUARANTINED model");
    const auto retry = registry.prepare_teardown(model);
    require(retry.finisher && registry.finalize_teardown(retry, true) == error::OK,
            "quarantined model recovery failed");

    require(registry.acquire_backend_context(), "backend-context fixture admission failed");
    require(registry.reserve_shutdown() == error::BUSY, "shutdown admitted live backend context");
    registry.release_backend_context();

    require(registry.reserve_shutdown() == error::OK, "clean shutdown reservation failed");
    require(registry.shutdown_reserved(), "successful shutdown was not reserved");
    require(registry.begin_outer().code == error::LOAD_BUSY, "post-check begin crossed shutdown reservation");
    require(!registry.acquire_backend_context(), "post-check backend context crossed shutdown reservation");
    require(registry.prepare_live_update(model).code == error::BUSY,
            "post-check update crossed shutdown reservation");
    require(registry.prepare_teardown(model).code == error::BUSY,
            "post-check teardown crossed shutdown reservation");
    registry.release_shutdown();
    require(!registry.shutdown_reserved(), "shutdown cancellation retained reservation");
    const auto after = registry.begin_outer();
    require(after.code == error::OK, "shutdown cancellation did not restore admission");
    const auto rolling = registry.prepare_end(after.txn, false);
    require(rolling.finisher && !rolling.commit, "rollback reservation fixture failed");
    require(registry.reserve_shutdown() == error::BUSY, "shutdown admitted ROLLING_BACK load");
    require(registry.finalize_end(rolling, true).code == error::MISSING_SUCCESS,
            "rollback fixture finalize failed");

    require(registry.reserve_shutdown() == error::OK, "completion reservation failed");
    registry.complete_shutdown();
    require(registry.shutdown_reserved(), "completed shutdown did not remain inactive");
    require(registry.begin_outer().code == error::LOAD_BUSY,
            "completed shutdown reopened mutation admission without reactivation");
    require(!registry.acquire_backend_context(),
            "completed shutdown reopened backend admission without reactivation");
    registry.reactivate();
    const auto reactivated = registry.begin_outer();
    require(reactivated.code == error::OK, "explicit registry reactivation did not reopen admission");
    require(registry.end(reactivated.txn, false).code == error::MISSING_SUCCESS,
            "reactivation cleanup failed");
}

static void backend_context_destructor_tail_case() {
    Registry registry;
    require(registry.acquire_backend_context(), "backend context admission failed");
    registry.test_block_backend_context_release();
    auto final_release = std::async(std::launch::async, [&] { registry.release_backend_context(); });
    registry.test_wait_for_backend_context_release();
    require(registry.reserve_shutdown() == error::BUSY,
            "checked unload crossed final backend destructor tail");
    registry.test_allow_backend_context_release();
    final_release.get();
    require(registry.reserve_shutdown() == error::OK,
            "completed backend destructor did not balance context admission");
    registry.release_shutdown();

    // Failed/duplicate releases never underflow the count, and later balanced
    // admissions still reserve cleanly.
    registry.release_backend_context();
    require(registry.acquire_backend_context(), "post-failure context admission failed");
    registry.release_backend_context();
    registry.release_backend_context();
    require(registry.reserve_shutdown() == error::OK, "context admission count underflowed");
    registry.release_shutdown();
}

static void allocation_free_live_update_guard_case() {
    Registry registry;
    const ModelToken model = commit_one(registry);
    try {
        auto guard = registry.acquire_live_update(model);
        require(bool(guard), "RAII update guard acquisition failed");
        throw std::bad_alloc(); // deterministic post-prepare allocation failure
    } catch (const std::bad_alloc &) {
    }

    // If guard construction/allocation leaked the prepared ticket, teardown
    // would block or the finite capacity below would be one short.
    std::vector<live_update_guard> capacity;
    capacity.reserve(model_slot_count);
    for (size_t i = 0; i < model_slot_count; ++i) {
        auto guard = registry.acquire_live_update(model);
        require(bool(guard), "post-fault ticket capacity did not recover");
        capacity.emplace_back(std::move(guard));
    }
    require(!registry.acquire_live_update(model), "ticket pool exceeded fixed capacity");
    capacity.clear();
    require(registry.teardown(model) == error::OK, "post-fault teardown remained wedged");
}

static void duplicate_publication_case() {
    Registry           registry;
    const auto         begin = registry.begin_outer();
    std::promise<void> prepared;
    std::promise<void> publish;
    auto               publish_gate = publish.get_future().share();
    std::atomic<bool>  publication_finalized{ false };

    std::thread finisher([&] {
        const auto ticket = registry.prepare_end(begin.txn, true);
        require(ticket.finisher, "first load_end was not finisher");
        prepared.set_value();
        publish_gate.wait();
        publication_finalized.store(true, std::memory_order_release);
        require(registry.finalize_end(ticket, true).committed, "publication finalize failed");
    });
    prepared.get_future().wait();

    auto duplicate = std::async(std::launch::async, [&] {
        const auto ticket = registry.prepare_end(begin.txn, true);
        require(publication_finalized.load(std::memory_order_acquire),
                "duplicate load_end escaped before publication finalization");
        return ticket;
    });
    require(duplicate.wait_for(std::chrono::milliseconds(30)) == std::future_status::timeout,
            "duplicate load_end did not wait in COMMITTING");
    publish.set_value();
    finisher.join();
    const auto replay = duplicate.get();
    require(!replay.finisher && replay.replay.committed, "duplicate load_end terminal replay mismatch");
}

int main() {
    candidate_binding_case(true);
    candidate_binding_case(false);
    candidate_depth_failure_case();
    durable_replay_case();
    finite_id_limit_case();
    quarantine_wrapper_busy_overlap_case();
    live_update_ticket_saturation_case();
    shutdown_reservation_state_machine_case();
    backend_context_destructor_tail_case();
    allocation_free_live_update_guard_case();
    duplicate_publication_case();
    return 0;
}
