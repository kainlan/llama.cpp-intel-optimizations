#include "model-lifecycle.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <thread>

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
    duplicate_publication_case();
    return 0;
}
