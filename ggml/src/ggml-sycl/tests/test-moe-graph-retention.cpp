#include "../moe-graph-retention.hpp"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <memory>

using namespace ggml_sycl;
using namespace ggml_sycl::execution;
using namespace ggml_sycl::moe;

static void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

static lifecycle::ModelToken token() {
    return {
        lifecycle::ModelId{ 1 },
        lifecycle::LoadTxnId{ 2 },
        lifecycle::SlotToken{ 3, 4 }
    };
}

class test_terminal final : public device_terminal {
  public:
    test_terminal(std::atomic<bool> & ready, std::atomic<unsigned> & waits) : ready_(ready), waits_(waits) {}

    bool ready() const noexcept override { return ready_.load(std::memory_order_acquire); }

    void wait() noexcept override { waits_.fetch_add(1, std::memory_order_relaxed); }
  private:
    std::atomic<bool> &     ready_;
    std::atomic<unsigned> & waits_;
};

struct lock_probe {
    lock_probe(graph_retention_registry & registry, std::atomic<unsigned> & drops) : registry(registry), drops(drops) {}

    ~lock_probe() {
        (void) registry.size();
        drops.fetch_add(1, std::memory_order_relaxed);
    }

    graph_retention_registry & registry;
    std::atomic<unsigned> &    drops;
};

int main() {
    Registry                 lifecycle_registry;
    graph_retention_registry retention;
    error                    err     = error::OK;
    const ContextId          context = lifecycle_registry.create_context(err);
    require(err == error::OK && lifecycle_registry.bind_backend(context, 0) == error::OK &&
                lifecycle_registry.bind_backend(context, 1) == error::OK,
            "setup failed");
    SessionId         session{};
    SessionResetEpoch reset{};
    require(lifecycle_registry.attach_root(context, token(), &session, &reset) == error::OK, "root attach failed");

    graph_recording_transaction first;
    require(graph_recording_transaction::begin(retention, lifecycle_registry, context, session, reset, token(),
                                               &first) == retention_error::OK,
            "first begin failed");
    const graph_owner_key       first_key  = first.key();
    auto                        raw_owner  = std::make_shared<int>(17);
    std::shared_ptr<const void> raw_handle = raw_owner;
    const mmid_operand_identity operand{ 10, 20, 0, 16, 64, 2 };
    require(first.add_batch({ operand, raw_handle }) == retention_error::OK, "batch add failed");
    auto table = graph_private_table_owner::create(first_key, 100, 200, 0, { raw_handle });
    require(first.add_table({ 100, 200, 0, table }) == retention_error::OK, "table add failed");
    first.mark_finalized();
    require(first.commit() == retention_error::OK, "first commit failed");
    auto first_snapshot = retention.snapshot(first_key);
    require(first_snapshot && first_snapshot->phase == retention_phase::INSTALLED &&
                first_snapshot->find_batch(operand) != nullptr,
            "installed record missing exact batch");
    auto different_identity = operand;
    different_identity.allocation_id++;
    require(first_snapshot->find_batch(different_identity) == nullptr,
            "same raw address with different durable identity matched");

    // The record, not caller locals, owns operand and immutable table payload.
    std::weak_ptr<const void>                      weak_operand = raw_handle;
    std::weak_ptr<const graph_private_table_owner> weak_table   = table;
    raw_handle.reset();
    raw_owner.reset();
    table.reset();
    first_snapshot.reset();
    require(!weak_operand.expired() && !weak_table.expired(), "record dropped operand/table owner early");

    graph_recording_transaction second;
    require(graph_recording_transaction::begin(retention, lifecycle_registry, context, session, reset, token(),
                                               &second) == retention_error::OK,
            "replacement begin failed");
    const graph_owner_key second_key = second.key();
    second.mark_finalized();
    require(second.commit() == retention_error::OK && retention.active(context) == second_key,
            "replacement install failed");
    require(retention.retire_exact(first_key) == retention_error::OK && retention.active(context) == second_key,
            "stale epoch retirement cleared replacement");
    require(weak_operand.expired() && weak_table.expired(), "retirement retained operand/table owner");

    // A failed record with known per-device submission remains pending until its
    // exact terminal reports ready; wait is issued only after readiness.
    graph_recording_transaction failed;
    require(graph_recording_transaction::begin(retention, lifecycle_registry, context, session, reset, token(),
                                               &failed) == retention_error::OK,
            "rollback begin failed");
    const graph_owner_key failed_key = failed.key();
    std::atomic<bool>     ready{ false };
    std::atomic<unsigned> waits{ 0 };
    require(failed.add_handle(std::make_shared<int>(9)) == retention_error::OK &&
                failed.note_submission(0, submit_outcome::SUBMITTED) == retention_error::OK &&
                failed.set_terminal(0, std::make_shared<test_terminal>(ready, waits)) == retention_error::OK &&
                failed.rollback() == retention_error::OK,
            "exact rollback retention failed");
    require(retention.retire_exact(failed_key) == retention_error::PENDING && waits.load() == 0,
            "pending terminal was waited or released");
    ready.store(true, std::memory_order_release);
    require(retention.retire_exact(failed_key) == retention_error::OK && waits.load() == 1,
            "delayed per-device terminal did not retire");

    // Unknown submission is never installed; it follows the quarantine path.
    graph_recording_transaction unknown;
    require(graph_recording_transaction::begin(retention, lifecycle_registry, context, session, reset, token(),
                                               &unknown) == retention_error::OK,
            "unknown begin failed");
    const graph_owner_key unknown_key = unknown.key();
    std::atomic<bool>     unknown_ready{ true };
    require(unknown.note_submission(1, submit_outcome::UNKNOWN) == retention_error::OK &&
                unknown.set_terminal(1, std::make_shared<test_terminal>(unknown_ready, waits)) == retention_error::OK,
            "unknown submit setup failed");
    unknown.mark_finalized();
    require(unknown.commit() == retention_error::OK, "unknown quarantine failed");
    auto unknown_snapshot = retention.snapshot(unknown_key);
    require(unknown_snapshot && unknown_snapshot->phase == retention_phase::QUARANTINED &&
                retention.active(context) == second_key,
            "unknown submit became executable");
    unknown_snapshot.reset();
    require(retention.retire_exact(unknown_key) == retention_error::OK, "unknown quarantine did not retire");

    // Table IDs are exclusive while a graph owns them; overwrite is rejected.
    graph_recording_transaction owner_a;
    require(graph_recording_transaction::begin(retention, lifecycle_registry, context, session, reset, token(),
                                               &owner_a) == retention_error::OK,
            "table owner begin failed");
    auto shared_id_a = graph_private_table_owner::create(owner_a.key(), 777, 1, 0, {});
    require(owner_a.add_table({ 777, 1, 0, shared_id_a }) == retention_error::OK, "table owner add failed");
    owner_a.mark_finalized();
    require(owner_a.commit() == retention_error::OK, "table owner commit failed");

    graph_recording_transaction owner_b;
    require(graph_recording_transaction::begin(retention, lifecycle_registry, context, session, reset, token(),
                                               &owner_b) == retention_error::OK,
            "overwrite begin failed");
    auto shared_id_b = graph_private_table_owner::create(owner_b.key(), 777, 1, 0, {});
    require(owner_b.add_table({ 777, 1, 0, shared_id_b }) == retention_error::OK, "overwrite staging failed");
    owner_b.mark_finalized();
    require(owner_b.commit() == retention_error::BUSY, "shared table overwrite accepted");
    require(owner_b.rollback() == retention_error::INCOMPLETE_TERMINALS,
            "rejected overwrite did not leave recording state");

    // Empty abandoned transactions use the explicit no-resources proof.
    {
        graph_recording_transaction empty;
        require(graph_recording_transaction::begin(retention, lifecycle_registry, context, session, reset, token(),
                                                   &empty) == retention_error::OK,
                "empty RAII begin failed");
    }

    // Last-owner destruction can re-enter the registry, proving it is unlocked.
    std::atomic<unsigned>  drops{ 0 };
    graph_retention_record probe_record;
    probe_record.key = { ContextId{ 999 }, GraphEpoch{ 999 } };
    probe_record.generic_handles.emplace_back(std::make_shared<lock_probe>(retention, drops));
    require(retention.quarantine(std::move(probe_record)) == retention_error::OK &&
                retention.retire_exact({ ContextId{ 999 }, GraphEpoch{ 999 } }) == retention_error::OK &&
                drops.load() == 1,
            "owner destroyed under retention lock");

    std::cout << "moe graph retention: ok\n";
}
