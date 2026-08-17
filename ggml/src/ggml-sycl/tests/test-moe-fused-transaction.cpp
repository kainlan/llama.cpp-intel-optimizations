#include "../moe-fused-transaction.hpp"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace ggml_sycl::moe_fused;

namespace {
int failures = 0;
#define CHECK(x)                                                                    \
    do {                                                                            \
        if (!(x)) {                                                                 \
            std::cerr << __FILE__ << ':' << __LINE__ << ": CHECK(" #x ") failed\n"; \
            ++failures;                                                             \
        }                                                                           \
    } while (false)

struct Trace {
    void add(std::string value) {
        std::lock_guard<std::mutex> lock(mutex);
        values.push_back(std::move(value));
    }

    std::vector<std::string> copy() const {
        std::lock_guard<std::mutex> lock(mutex);
        return values;
    }

    mutable std::mutex       mutex;
    std::vector<std::string> values;
};

FusionPlan ordinary_plan() {
    return { true, true, true, true, DownMode::ordinary, false };
}

struct TestPreflight final : Preflight {
    Status result = Status::ok();
    bool   throws = false;

    Status run(const FusionPlan &) override {
        if (throws) {
            throw std::runtime_error("preflight");
        }
        return result;
    }
};

struct TraceOwner final : Owner {
    TraceOwner(Trace & trace, std::string name) : trace(trace), name(std::move(name)) {}

    ~TraceOwner() override { trace.add("owner-" + name); }

    Trace &     trace;
    std::string name;
};

struct TraceDrain final : EmergencyDrain {
    explicit TraceDrain(Trace & trace) : trace(trace) {}

    void drain() noexcept override { trace.add("drain"); }

    Trace & trace;
};

struct TraceEvent final : TerminalEvent {
    explicit TraceEvent(Trace & trace) : trace(trace) {}

    void wait() noexcept override { trace.add("wait"); }

    Trace & trace;
};

struct ReentrantEvent final : TerminalEvent {
    ReentrantEvent(PublicationStore & store, std::atomic<bool> & entered) : store(store), entered(entered) {}

    void wait() noexcept override {
        (void) store.snapshot();
        entered.store(true, std::memory_order_release);
    }

    PublicationStore &  store;
    std::atomic<bool> & entered;
};

OwnerBundle owners_for(const FusionPlan & plan, Trace & trace) {
    OwnerBundle owners;
    owners.add(OwnerRole::gate, std::make_unique<TraceOwner>(trace, "gate"));
    owners.add(OwnerRole::up, std::make_unique<TraceOwner>(trace, "up"));
    owners.add(OwnerRole::glu, std::make_unique<TraceOwner>(trace, "glu"));
    if (plan.down == DownMode::fused) {
        owners.add(OwnerRole::down, std::make_unique<TraceOwner>(trace, "down"));
    }
    return owners;
}

Status install_escrow(const FusionPlan & plan, SubmitRecorder & recorder, Trace & trace) {
    return recorder.escrow(owners_for(plan, trace), EmergencyToken(std::make_unique<TraceDrain>(trace)));
}

enum class Behavior {
    success,
    no_write,
    missing_terminal,
    throw_before,
    throw_after_before_terminal,
    duplicate_terminal
};

struct TestSubmitter final : Submitter {
    TestSubmitter(Behavior behavior, Trace & trace) : behavior(behavior), trace(trace) {}

    Status submit(const FusionPlan & plan, SubmitRecorder & recorder) override {
        if (behavior == Behavior::throw_before) {
            throw std::runtime_error("before");
        }
        if (behavior == Behavior::no_write) {
            return { ErrorCode::submit_failed_no_write, "no write" };
        }
        Status status = install_escrow(plan, recorder, trace);
        if (!status) {
            return status;
        }
        status = recorder.mark_write_started();
        if (!status) {
            return status;
        }
        if (behavior == Behavior::throw_after_before_terminal) {
            throw std::runtime_error("after");
        }
        if (behavior == Behavior::missing_terminal) {
            return Status::ok();
        }
        status = recorder.install_terminal(TerminalToken(std::make_unique<TraceEvent>(trace)));
        if (!status) {
            return status;
        }
        if (behavior == Behavior::duplicate_terminal) {
            (void) recorder.install_terminal(TerminalToken(std::make_unique<TraceEvent>(trace)));
        }
        return Status::ok();
    }

    Behavior behavior;
    Trace &  trace;
};

void preflight_ok(Transaction & transaction) {
    TestPreflight preflight;
    CHECK(transaction.preflight(preflight));
}

void test_prewrite_rejection_and_mutations() {
    TestPreflight preflight;
    FusionPlan    plan = ordinary_plan();
    plan.gate_up_pair  = false;
    Transaction absent_pair(plan);
    CHECK(absent_pair.preflight(preflight).code == ErrorCode::absent_pair);
    CHECK(absent_pair.fallback_allowed());

    for (int mutation = 0; mutation < 3; ++mutation) {
        plan = ordinary_plan();
        if (mutation == 0) {
            plan.gate_role = false;
        }
        if (mutation == 1) {
            plan.up_role = false;
        }
        if (mutation == 2) {
            plan.glu_role = false;
        }
        Transaction transaction(plan);
        CHECK(transaction.preflight(preflight).code == ErrorCode::absent_role);
    }

    Transaction failed(ordinary_plan());
    preflight.result = { ErrorCode::preflight_failed, "fail" };
    CHECK(failed.preflight(preflight).code == ErrorCode::preflight_failed);
    CHECK(failed.fallback_allowed());

    Trace       trace;
    Transaction no_write(ordinary_plan());
    preflight_ok(no_write);
    TestSubmitter submitter(Behavior::no_write, trace);
    CHECK(no_write.submit(submitter).code == ErrorCode::submit_failed_no_write);
    CHECK(no_write.fallback_allowed());
}

void test_escrow_validation() {
    Trace      trace;
    FusionPlan plan = ordinary_plan();

    Transaction missing(plan);
    preflight_ok(missing);

    struct Missing final : Submitter {
        Status submit(const FusionPlan &, SubmitRecorder & recorder) override { return recorder.mark_write_started(); }
    } missing_submitter;

    CHECK(missing.submit(missing_submitter).code == ErrorCode::escrow_missing);
    CHECK(missing.fallback_allowed());

    Transaction no_emergency(plan);
    preflight_ok(no_emergency);

    struct NoEmergency final : Submitter {
        explicit NoEmergency(Trace & trace) : trace(trace) {}

        Status submit(const FusionPlan & plan, SubmitRecorder & recorder) override {
            (void) recorder.escrow(owners_for(plan, trace), EmergencyToken{});
            return recorder.mark_write_started();
        }

        Trace & trace;
    } no_emergency_submitter(trace);

    CHECK(no_emergency.submit(no_emergency_submitter).code == ErrorCode::emergency_missing);
    CHECK(no_emergency.fallback_allowed());

    Transaction invalid(plan);
    preflight_ok(invalid);

    struct Invalid final : Submitter {
        explicit Invalid(Trace & trace) : trace(trace) {}

        Status submit(const FusionPlan & plan, SubmitRecorder & recorder) override {
            OwnerBundle owners = owners_for(plan, trace);
            owners.add(static_cast<OwnerRole>(99), std::make_unique<TraceOwner>(trace, "invalid"));
            return recorder.escrow(std::move(owners), EmergencyToken(std::make_unique<TraceDrain>(trace)));
        }

        Trace & trace;
    } invalid_submitter(trace);

    CHECK(invalid.submit(invalid_submitter).code == ErrorCode::invalid_owner_role);
    CHECK(invalid.fallback_allowed());

    Transaction duplicate(plan);
    preflight_ok(duplicate);

    struct Duplicate final : Submitter {
        explicit Duplicate(Trace & trace) : trace(trace) {}

        Status submit(const FusionPlan & plan, SubmitRecorder & recorder) override {
            CHECK(install_escrow(plan, recorder, trace));
            OwnerBundle  replacement = owners_for(plan, trace);
            const Status duplicate_status =
                recorder.escrow(std::move(replacement), EmergencyToken(std::make_unique<TraceDrain>(trace)));
            CHECK(duplicate_status.code == ErrorCode::duplicate_install);
            return recorder.mark_write_started();
        }

        Trace & trace;
    } duplicate_submitter(trace);

    CHECK(duplicate.submit(duplicate_submitter).code == ErrorCode::duplicate_install);
    CHECK(duplicate.fallback_allowed());  // known-bad recorder state cannot cross the write boundary
}

void test_postwrite_escrow_lifetime() {
    for (Behavior behavior : { Behavior::missing_terminal, Behavior::throw_after_before_terminal }) {
        Trace trace;
        {
            Transaction transaction(ordinary_plan());
            preflight_ok(transaction);
            TestSubmitter submitter(behavior, trace);
            const Status  status = transaction.submit(submitter);
            CHECK(status.code == (behavior == Behavior::missing_terminal ? ErrorCode::submitted_without_terminal :
                                                                           ErrorCode::exception_after_write));
            CHECK(!transaction.fallback_allowed());
            CHECK(trace.copy().empty());
        }
        const auto values = trace.copy();
        CHECK(!values.empty() && values.front() == "drain");
        CHECK(values.size() >= 2 && values[1].rfind("owner-", 0) == 0);
    }

    Trace duplicate_trace;
    {
        Transaction transaction(ordinary_plan());
        preflight_ok(transaction);
        TestSubmitter submitter(Behavior::duplicate_terminal, duplicate_trace);
        CHECK(transaction.submit(submitter).code == ErrorCode::duplicate_install);
        CHECK(!transaction.fallback_allowed());
    }
    const auto values = duplicate_trace.copy();
    CHECK(values.size() >= 3);
    CHECK(values[0] == "wait");  // rejected duplicate waited immediately
    CHECK(values[1] == "wait");  // original terminal settled before escrow release
    CHECK(values[2].rfind("owner-", 0) == 0);
}

void test_publication_lifetime_and_failure() {
    Trace                      trace;
    PublicationStore::Snapshot lease;
    {
        PublicationStore store;
        {
            Transaction transaction(ordinary_plan());
            preflight_ok(transaction);
            TestSubmitter submitter(Behavior::success, trace);
            CHECK(transaction.submit(submitter));
            CHECK(transaction.commit(store));
            CHECK(transaction.commit(store).code == ErrorCode::invalid_state);
            lease = store.snapshot();
            CHECK(lease.publication.ready && lease.publication.skip_gate && lease.publication.skip_up &&
                  lease.publication.skip_glu && !lease.publication.skip_down);
            CHECK(trace.copy().empty());
        }
        CHECK(trace.copy().empty());
    }
    CHECK(trace.copy().empty());
    lease.ownership_lease.reset();
    const auto values = trace.copy();
    CHECK(!values.empty() && values.front() == "wait");

    Trace            failure_trace;
    PublicationStore store;
    {
        Transaction transaction(ordinary_plan());
        preflight_ok(transaction);
        TestSubmitter submitter(Behavior::success, failure_trace);
        CHECK(transaction.submit(submitter));
        CHECK(transaction.commit(store, true).code == ErrorCode::publication_failed);
        CHECK(!store.snapshot().publication.ready);
        CHECK(failure_trace.copy().empty());
    }
    CHECK(failure_trace.copy().front() == "wait");
}

void test_wait_reentry_and_contended_publication() {
    // Trace must outlive the store because store retirement waits before releasing owners.
    Trace             trace;
    PublicationStore  store;
    std::atomic<bool> reentered{ false };

    struct ReentrantSubmitter final : Submitter {
        ReentrantSubmitter(PublicationStore & store, std::atomic<bool> & entered, Trace & trace) :
            store(store),
            entered(entered),
            trace(trace) {}

        Status submit(const FusionPlan & plan, SubmitRecorder & recorder) override {
            CHECK(install_escrow(plan, recorder, trace));
            CHECK(recorder.mark_write_started());
            return recorder.install_terminal(TerminalToken(std::make_unique<ReentrantEvent>(store, entered)));
        }

        PublicationStore &  store;
        std::atomic<bool> & entered;
        Trace &             trace;
    } first_submitter(store, reentered, trace);

    Transaction first(ordinary_plan(), 0);
    preflight_ok(first);
    CHECK(first.submit(first_submitter));
    CHECK(first.commit(store));

    std::atomic<bool>        stop{ false };
    std::atomic<bool>        partial{ false };
    std::vector<std::thread> readers;
    for (int i = 0; i < 4; ++i) {
        readers.emplace_back([&] {
            while (!stop.load(std::memory_order_acquire)) {
                const auto snapshot = store.snapshot();
                if (snapshot.publication.ready && (!snapshot.publication.skip_gate || !snapshot.publication.skip_up ||
                                                   !snapshot.publication.skip_glu || !snapshot.ownership_lease)) {
                    partial.store(true, std::memory_order_release);
                }
            }
        });
    }

    Transaction second(ordinary_plan(), 1);
    preflight_ok(second);
    TestSubmitter second_submitter(Behavior::success, trace);
    CHECK(second.submit(second_submitter));
    CHECK(second.commit(store));  // retirement waits outside lock and re-enters snapshot()
    stop.store(true, std::memory_order_release);
    for (auto & reader : readers) {
        reader.join();
    }
    CHECK(reentered.load(std::memory_order_acquire));
    CHECK(!partial.load(std::memory_order_acquire));
    CHECK(store.snapshot().publication.generation == 2);
}

void test_publication_store_flush() {
    // llama.cpp-kzjv: a commit only retires the PREVIOUS publication (the one
    // it replaces). The publication that nothing ever supersedes -- the last
    // commit into a store shared across a whole phase -- must still release
    // its owners once nobody is coming to replace it. flush() is that
    // explicit release.
    Trace            trace;
    PublicationStore store;
    Transaction      transaction(ordinary_plan());
    preflight_ok(transaction);
    TestSubmitter submitter(Behavior::success, trace);
    CHECK(transaction.submit(submitter));
    CHECK(transaction.commit(store));
    CHECK(trace.copy().empty());  // nothing retired it yet -- store still owns it

    store.flush();
    const auto values = trace.copy();
    CHECK(!values.empty() && values.front() == "wait");  // flush waited on the terminal, then released owners

    // A flushed store is left in a normal, reusable state: generation is
    // untouched, and a later commit builds on it exactly as if the flushed
    // publication had been retired by a real supersession.
    CHECK(store.snapshot().publication.generation == 1);
    CHECK(!store.snapshot().ownership_lease);

    Trace       second_trace;
    Transaction second(ordinary_plan(), store.snapshot().publication.generation);
    preflight_ok(second);
    TestSubmitter second_submitter(Behavior::success, second_trace);
    CHECK(second.submit(second_submitter));
    CHECK(second.commit(store));
    CHECK(store.snapshot().publication.generation == 2);

    // flush() on an empty store (nothing published, or already flushed) is a
    // harmless no-op -- it must not crash or disturb publication_ state.
    PublicationStore empty_store;
    empty_store.flush();
    CHECK(!empty_store.snapshot().ownership_lease);
    store.flush();
    CHECK(!second_trace.copy().empty() && second_trace.copy().front() == "wait");
}

void test_optional_down() {
    Trace      trace;
    FusionPlan plan = ordinary_plan();
    plan.down       = DownMode::fused;
    plan.down_role  = true;
    Transaction transaction(plan);
    preflight_ok(transaction);
    TestSubmitter submitter(Behavior::success, trace);
    CHECK(transaction.submit(submitter));
    PublicationStore store;
    CHECK(transaction.commit(store));
    CHECK(store.snapshot().publication.skip_down);
}

PromptFusionBundle prompt_bundle() {
    PromptFusionBundle bundle;
    auto fill = [](RetainedRoleView & role, std::uintptr_t handle, std::uintptr_t table, std::uintptr_t ready,
                   std::int64_t layout) {
        role.present         = true;
        role.handle_identity = handle;
        role.table_identity  = table;
        role.ready_identity  = ready;
        role.occurrences     = {
            { 7, 0, 0, 0, handle + 10, layout, OccurrenceResidency::local },
            { 7, 1, 0, 1, handle + 10, layout, OccurrenceResidency::local },
            { 3, 2, 1, 0, handle + 11, layout, OccurrenceResidency::local }
        };
    };
    fill(bundle.gate, 11, 12, 13, 1);
    fill(bundle.up, 21, 22, 23, 2);
    fill(bundle.down, 31, 32, 33, 3);
    bundle.activation_identity   = 41;
    bundle.ids_identity          = 42;
    bundle.glu_identity          = 43;
    bundle.intermediate_identity = 44;
    return bundle;
}

OwnerBundle prompt_owners(Trace & trace, bool omit_down_table = false) {
    OwnerBundle owners;
    auto        add = [&](OwnerRole role, const char * name) {
        owners.add(role, std::make_unique<TraceOwner>(trace, name));
    };
    add(OwnerRole::gate, "gate");
    add(OwnerRole::gate_table, "gate-table");
    add(OwnerRole::up, "up");
    add(OwnerRole::up_table, "up-table");
    add(OwnerRole::activation, "activation");
    add(OwnerRole::ids, "ids");
    add(OwnerRole::glu, "glu");
    add(OwnerRole::intermediate, "intermediate");
    add(OwnerRole::down, "down");
    if (!omit_down_table) {
        add(OwnerRole::down_table, "down-table");
    }
    return owners;
}

enum class PromptBehavior { success, no_write, missing_terminal, exception_after_write, partial_down };

struct PromptExecutor final : PromptFusionExecutor {
    PromptExecutor(Trace & trace, PromptBehavior behavior = PromptBehavior::success) :
        trace(trace),
        behavior(behavior) {}

    Status preflight(const PromptFusionBundle &) override {
        ++preflight_calls;
        return preflight_result;
    }

    OwnerBundle owners() override { return prompt_owners(trace, omit_down_table); }

    EmergencyToken emergency() override { return EmergencyToken(std::make_unique<TraceDrain>(trace)); }

    Status submit_writes(const PromptFusionBundle &, SubmitRecorder & recorder) override {
        ++submit_calls;
        if (behavior == PromptBehavior::no_write) {
            return { ErrorCode::submit_failed_no_write, "refused" };
        }
        CHECK(!recorder.write_started());
        CHECK(recorder.mark_write_started());
        CHECK(recorder.write_started());
        if (behavior == PromptBehavior::exception_after_write) {
            throw std::runtime_error("post-write");
        }
        if (behavior == PromptBehavior::missing_terminal || behavior == PromptBehavior::partial_down) {
            return behavior == PromptBehavior::partial_down ?
                       Status{ ErrorCode::submit_failed_no_write, "down failed" } :
                       Status::ok();
        }
        return recorder.install_terminal(TerminalToken(std::make_unique<TraceEvent>(trace)));
    }

    Trace &        trace;
    PromptBehavior behavior;
    Status         preflight_result = Status::ok();
    bool           omit_down_table  = false;
    int            preflight_calls  = 0;
    int            submit_calls     = 0;
};

void test_prompt_bundle_preflight_and_no_write() {
    PublicationStore store;
    for (int mutation = 0; mutation < 5; ++mutation) {
        Trace              trace;
        PromptFusionBundle bundle = prompt_bundle();
        if (mutation == 0) {
            bundle.up.present = false;
        }
        if (mutation == 1) {
            bundle.down.occurrences[1].expert_id = 8;
        }
        if (mutation == 2) {
            bundle.gate.occurrences[0].residency = OccurrenceResidency::host;
        }
        if (mutation == 3) {
            bundle.up.occurrences[2].residency = OccurrenceResidency::secondary;
        }
        if (mutation == 4) {
            bundle.ids_identity = 0;
        }
        PromptExecutor executor(trace);
        const auto     result = submit_prompt_fusion(bundle, executor, store, 0);
        CHECK(result.status.code == ErrorCode::preflight_failed);
        CHECK(result.phase == Phase::rolled_back && result.fallback_allowed);
        CHECK(executor.submit_calls == 0 && trace.copy().empty());
    }

    Trace          trace;
    PromptExecutor refused(trace, PromptBehavior::no_write);
    const auto     result = submit_prompt_fusion(prompt_bundle(), refused, store, 0);
    CHECK(result.status.code == ErrorCode::submit_failed_no_write);
    CHECK(result.phase == Phase::rolled_back && result.fallback_allowed);
    const auto values = trace.copy();
    CHECK(values.size() == 11 && values.front() == "drain");  // quiesce before releasing all ten owners
}

void test_prompt_postwrite_quarantine_failpoints() {
    for (PromptBehavior behavior :
         { PromptBehavior::missing_terminal, PromptBehavior::exception_after_write, PromptBehavior::partial_down }) {
        Trace            trace;
        PublicationStore store;
        PromptExecutor   executor(trace, behavior);
        const auto       result = submit_prompt_fusion(prompt_bundle(), executor, store, 0);
        CHECK(!result.status);
        CHECK(result.phase == Phase::quarantined && !result.fallback_allowed);
        const auto values = trace.copy();
        CHECK(!values.empty() && values.front() == "drain");
        CHECK(!store.snapshot().publication.ready);
    }

    Trace            missing_owner_trace;
    PublicationStore missing_owner_store;
    PromptExecutor   missing_owner(missing_owner_trace);
    missing_owner.omit_down_table = true;
    const auto missing            = submit_prompt_fusion(prompt_bundle(), missing_owner, missing_owner_store, 0);
    CHECK(missing.status.code == ErrorCode::owner_mismatch);
    CHECK(missing.phase == Phase::rolled_back && missing.fallback_allowed);
}

void test_prompt_atomic_publish_stale_and_failure() {
    Trace            trace;
    PublicationStore store;
    PromptExecutor   first(trace);
    const auto       published = submit_prompt_fusion(prompt_bundle(), first, store, 0);
    CHECK(published.status && published.phase == Phase::published && !published.fallback_allowed);
    const auto snapshot = store.snapshot();
    CHECK(snapshot.publication.ready && snapshot.publication.skip_gate && snapshot.publication.skip_up &&
          snapshot.publication.skip_glu && snapshot.publication.skip_down && snapshot.ownership_lease);

    Trace          stale_trace;
    PromptExecutor stale(stale_trace);
    const auto     stale_result = submit_prompt_fusion(prompt_bundle(), stale, store, 0);
    CHECK(stale_result.status.code == ErrorCode::stale_commit);
    CHECK(stale_result.phase == Phase::quarantined && !stale_result.fallback_allowed);
    CHECK(!stale_trace.copy().empty() && stale_trace.copy().front() == "wait");
    CHECK(store.snapshot().publication.generation == 1);

    Trace            failure_trace;
    PublicationStore failure_store;
    PromptExecutor   failure(failure_trace);
    const auto       failure_result = submit_prompt_fusion(prompt_bundle(), failure, failure_store, 0, true);
    CHECK(failure_result.status.code == ErrorCode::publication_failed);
    CHECK(failure_result.phase == Phase::quarantined && !failure_result.fallback_allowed);
    CHECK(!failure_store.snapshot().publication.ready);
    CHECK(!failure_trace.copy().empty() && failure_trace.copy().front() == "wait");
}

}  // namespace

int main() {
    test_prewrite_rejection_and_mutations();
    test_escrow_validation();
    test_postwrite_escrow_lifetime();
    test_publication_lifetime_and_failure();
    test_wait_reentry_and_contended_publication();
    test_publication_store_flush();
    test_optional_down();
    test_prompt_bundle_preflight_and_no_write();
    test_prompt_postwrite_quarantine_failpoints();
    test_prompt_atomic_publish_stale_and_failure();
    if (failures) {
        std::cerr << failures << " test(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "moe-fused-transaction: all tests passed\n";
    return EXIT_SUCCESS;
}
