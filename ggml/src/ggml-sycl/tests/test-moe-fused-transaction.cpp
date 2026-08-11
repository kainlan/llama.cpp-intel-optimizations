#include "../moe-fused-transaction.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ggml_sycl::moe_fused;

namespace {

int failures = 0;

#define CHECK(expr)                                                                    \
    do {                                                                               \
        if (!(expr)) {                                                                 \
            std::cerr << __FILE__ << ':' << __LINE__ << ": CHECK(" #expr ") failed\n"; \
            ++failures;                                                                \
        }                                                                              \
    } while (false)

FusionPlan ordinary_plan() {
    return { true, true, true, true, DownMode::ordinary, false };
}

struct TestPreflight final : Preflight {
    Status result          = Status::ok();
    bool   throw_exception = false;

    Status run(const FusionPlan &) override {
        if (throw_exception) {
            throw std::runtime_error("preflight boom");
        }
        return result;
    }
};

struct TraceEvent final : TerminalEvent {
    explicit TraceEvent(std::vector<std::string> & trace) : trace(trace) {}

    ~TraceEvent() override { trace.push_back("event-destroy"); }

    void wait() noexcept override { trace.push_back("wait"); }

    std::vector<std::string> & trace;
};

struct TraceOwner final : Owner {
    TraceOwner(std::vector<std::string> & trace, std::string name) : trace(trace), name(std::move(name)) {}

    ~TraceOwner() override { trace.push_back("owner-" + name); }

    std::vector<std::string> & trace;
    std::string                name;
};

OwnerBundle owners_for(const FusionPlan & plan, std::vector<std::string> & trace) {
    OwnerBundle owners;
    owners.add(OwnerRole::gate, std::make_unique<TraceOwner>(trace, "gate"));
    owners.add(OwnerRole::up, std::make_unique<TraceOwner>(trace, "up"));
    owners.add(OwnerRole::glu, std::make_unique<TraceOwner>(trace, "glu"));
    if (plan.down == DownMode::fused) {
        owners.add(OwnerRole::down, std::make_unique<TraceOwner>(trace, "down"));
    }
    return owners;
}

enum class SubmitBehavior {
    no_write_failure,
    missing_terminal,
    missing_owners,
    mismatched_owners,
    success,
    throw_before,
    throw_after
};

struct TestSubmitter final : Submitter {
    TestSubmitter(SubmitBehavior behavior, std::vector<std::string> & trace) : behavior(behavior), trace(trace) {}

    SubmitBehavior             behavior;
    std::vector<std::string> & trace;

    Status submit(const FusionPlan & plan, SubmitRecorder & recorder) override {
        if (behavior == SubmitBehavior::throw_before) {
            throw std::runtime_error("before");
        }
        if (behavior == SubmitBehavior::no_write_failure) {
            return { ErrorCode::submit_failed_no_write, "backend rejected submit" };
        }
        recorder.mark_write_started();
        if (behavior == SubmitBehavior::missing_terminal) {
            recorder.install_terminal_owners(TerminalToken{}, owners_for(plan, trace));
        } else if (behavior == SubmitBehavior::missing_owners) {
            recorder.install_terminal_owners(TerminalToken(std::make_unique<TraceEvent>(trace)), OwnerBundle{});
        } else {
            OwnerBundle owners = owners_for(plan, trace);
            if (behavior == SubmitBehavior::mismatched_owners) {
                owners.add(OwnerRole::down, std::make_unique<TraceOwner>(trace, "unexpected-down"));
            }
            recorder.install_terminal_owners(TerminalToken(std::make_unique<TraceEvent>(trace)), std::move(owners));
        }
        if (behavior == SubmitBehavior::throw_after) {
            throw std::runtime_error("after");
        }
        return Status::ok();
    }
};

void preflight_ok(Transaction & transaction) {
    TestPreflight preflight;
    CHECK(transaction.preflight(preflight));
}

void test_plan_mutations_rejected() {
    TestPreflight preflight;
    FusionPlan    plan = ordinary_plan();
    plan.gate_up_pair  = false;
    Transaction absent_pair(plan);
    CHECK(absent_pair.preflight(preflight).code == ErrorCode::absent_pair);
    CHECK(absent_pair.fallback_allowed());

    for (int mutation = 0; mutation != 3; ++mutation) {
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
        Transaction absent_role(plan);
        CHECK(absent_role.preflight(preflight).code == ErrorCode::absent_role);
    }

    plan      = ordinary_plan();
    plan.down = DownMode::fused;
    Transaction absent_down(plan);
    CHECK(absent_down.preflight(preflight).code == ErrorCode::absent_role);
}

void test_preflight_and_no_write_failures_allow_fallback() {
    TestPreflight failing;
    failing.result = { ErrorCode::preflight_failed, "unsupported" };
    Transaction preflight_failure(ordinary_plan());
    CHECK(preflight_failure.preflight(failing).code == ErrorCode::preflight_failed);
    CHECK(preflight_failure.phase() == Phase::rolled_back);
    CHECK(preflight_failure.fallback_allowed());

    TestPreflight throwing;
    throwing.throw_exception = true;
    Transaction preflight_exception(ordinary_plan());
    CHECK(preflight_exception.preflight(throwing).code == ErrorCode::preflight_failed);
    CHECK(preflight_exception.fallback_allowed());

    std::vector<std::string> trace;
    Transaction              submit_failure(ordinary_plan());
    preflight_ok(submit_failure);
    TestSubmitter no_write{ SubmitBehavior::no_write_failure, trace };
    CHECK(submit_failure.submit(no_write).code == ErrorCode::submit_failed_no_write);
    CHECK(submit_failure.fallback_allowed());

    Transaction before_exception(ordinary_plan());
    preflight_ok(before_exception);
    TestSubmitter throw_before{ SubmitBehavior::throw_before, trace };
    CHECK(before_exception.submit(throw_before).code == ErrorCode::exception_before_write);
    CHECK(before_exception.fallback_allowed());
}

void test_post_write_contract_failures_quarantine() {
    std::vector<std::string> trace;
    for (SubmitBehavior behavior : { SubmitBehavior::missing_terminal, SubmitBehavior::missing_owners }) {
        Transaction transaction(ordinary_plan());
        preflight_ok(transaction);
        TestSubmitter submitter{ behavior, trace };
        const Status  status = transaction.submit(submitter);
        CHECK(status.code == (behavior == SubmitBehavior::missing_terminal ? ErrorCode::submitted_without_terminal :
                                                                             ErrorCode::submitted_without_owners));
        CHECK(transaction.phase() == Phase::quarantined);
        CHECK(!transaction.fallback_allowed());
    }

    {
        Transaction transaction(ordinary_plan());
        preflight_ok(transaction);
        TestSubmitter submitter{ SubmitBehavior::mismatched_owners, trace };
        CHECK(transaction.submit(submitter).code == ErrorCode::owner_mismatch);
        CHECK(transaction.phase() == Phase::quarantined);
        CHECK(!transaction.fallback_allowed());
    }

    trace.clear();
    {
        Transaction transaction(ordinary_plan());
        preflight_ok(transaction);
        TestSubmitter submitter{ SubmitBehavior::throw_after, trace };
        CHECK(transaction.submit(submitter).code == ErrorCode::exception_after_write);
        CHECK(transaction.phase() == Phase::quarantined);
        CHECK(!transaction.fallback_allowed());
        CHECK(trace.empty());
    }
    CHECK(!trace.empty() && trace.front() == "wait");
}

void test_success_atomic_publication_and_lifetime() {
    std::vector<std::string>   trace;
    PublicationStore::Snapshot lease;
    {
        PublicationStore store;
        {
            Transaction transaction(ordinary_plan());
            preflight_ok(transaction);
            TestSubmitter submitter{ SubmitBehavior::success, trace };
            CHECK(transaction.submit(submitter));
            CHECK(!transaction.fallback_allowed());
            CHECK(transaction.commit(store));
            CHECK(transaction.phase() == Phase::published);
            CHECK(transaction.commit(store).code == ErrorCode::invalid_state);  // idempotence rejection
            lease = store.snapshot();
            CHECK(lease.publication.ready);
            CHECK(lease.publication.skip_gate && lease.publication.skip_up && lease.publication.skip_glu);
            CHECK(!lease.publication.skip_down);  // optional down remains ordinary
            CHECK(lease.ownership_lease != nullptr);
            CHECK(trace.empty());
        }
        CHECK(trace.empty());  // PublicationStore, not Transaction, owns the terminal and owners.
    }
    CHECK(trace.empty());      // A consumer snapshot lease extends ownership beyond the store.
    lease.ownership_lease.reset();
    CHECK(!trace.empty() && trace.front() == "wait");
}

void test_atomic_failure_and_stale_commit() {
    std::vector<std::string> failure_trace;
    PublicationStore         store;
    {
        Transaction transaction(ordinary_plan());
        preflight_ok(transaction);
        TestSubmitter submitter{ SubmitBehavior::success, failure_trace };
        CHECK(transaction.submit(submitter));
        CHECK(transaction.commit(store, true).code == ErrorCode::publication_failed);
        const auto snapshot = store.snapshot();
        CHECK(!snapshot.publication.ready);
        CHECK(!snapshot.publication.skip_gate && !snapshot.publication.skip_up && !snapshot.publication.skip_glu);
        CHECK(snapshot.ownership_lease == nullptr);
        CHECK(!transaction.fallback_allowed());
        CHECK(failure_trace.empty());
    }
    CHECK(!failure_trace.empty() && failure_trace.front() == "wait");

    std::vector<std::string> first_trace;
    std::vector<std::string> stale_trace;
    Transaction              first(ordinary_plan(), 0);
    Transaction              stale(ordinary_plan(), 0);
    preflight_ok(first);
    preflight_ok(stale);
    TestSubmitter first_submit{ SubmitBehavior::success, first_trace };
    TestSubmitter stale_submit{ SubmitBehavior::success, stale_trace };
    CHECK(first.submit(first_submit));
    CHECK(stale.submit(stale_submit));
    CHECK(first.commit(store));
    CHECK(stale.commit(store).code == ErrorCode::stale_commit);
    CHECK(stale.phase() == Phase::quarantined);
    CHECK(!stale.fallback_allowed());
    CHECK(store.snapshot().publication.generation == 1);
}

void test_fused_down_publishes_and_requires_exact_owner_set() {
    std::vector<std::string> trace;
    FusionPlan               plan = ordinary_plan();
    plan.down                     = DownMode::fused;
    plan.down_role                = true;
    Transaction transaction(plan);
    preflight_ok(transaction);
    TestSubmitter submitter{ SubmitBehavior::success, trace };
    CHECK(transaction.submit(submitter));
    PublicationStore store;
    CHECK(transaction.commit(store));
    CHECK(store.snapshot().publication.skip_down);
}

}  // namespace

int main() {
    test_plan_mutations_rejected();
    test_preflight_and_no_write_failures_allow_fallback();
    test_post_write_contract_failures_quarantine();
    test_success_atomic_publication_and_lifetime();
    test_atomic_failure_and_stale_commit();
    test_fused_down_publishes_and_requires_exact_owner_set();
    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "moe-fused-transaction: all tests passed\n";
    return EXIT_SUCCESS;
}
