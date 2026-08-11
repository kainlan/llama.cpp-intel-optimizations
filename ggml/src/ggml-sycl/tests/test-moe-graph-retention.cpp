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
    }

    graph_retention_registry & retention;
    Registry                   execution;
    error                      err = error::OK;
    ContextId                  context{};
    SessionId                  session{};
    SessionResetEpoch          reset{};
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
