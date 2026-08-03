#include "model-lifecycle.hpp"

#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace ggml_sycl::lifecycle;

static void require(bool v, const char * marker) {
    if (!v) {
        std::cerr << marker << '\n';
        std::exit(1);
    }
}

static ModelToken commit_one(Registry & r, publication_data p = {}) {
    auto b = r.begin_outer();
    require(b.code == error::OK, "begin failed");
    auto e = r.end(b.txn, true, p.planned_host_bytes, p.actual_host_bytes, p.verdict);
    require(e.code == error::OK && e.committed, "outer commit failed");
    return e.token;
}

static void run_case(const std::string & name, test_mutation mutation) {
    if (name == "nested-success") {
        Registry r(UINT64_MAX, UINT64_MAX, mutation);
        auto     b = r.begin_outer();
        require(r.enter_nested(b.txn) == error::OK, "nested begin failed");
        auto inner = r.end(b.txn, true);
        if (mutation == test_mutation::M2_NESTED_COMMIT) {
            require(!inner.committed, "nested load committed");
        }
        require(inner.code == error::NESTED && !inner.committed, "nested load committed");
        require(r.end(b.txn, true).committed && r.publication_count() == 1, "outer did not commit exactly once");
    } else if (name == "inner-failure" || name == "cancel") {
        Registry r(UINT64_MAX, UINT64_MAX, mutation);
        auto     b = r.begin_outer();
        r.enter_nested(b.txn);
        r.end(b.txn, false);
        auto e = r.end(b.txn, true);
        if (mutation == test_mutation::M3_CLEAR_POISON) {
            require(!e.committed, "poisoned transaction published LIVE");
        }
        require(!e.committed && !r.find(b.token.model), "poisoned transaction published LIVE");
    } else if (name == "missing-success") {
        Registry r;
        auto     b = r.begin_outer();
        auto     e = r.end(b.txn, false);
        require(e.code == error::MISSING_SUCCESS && !r.last_success(), "missing success published LIVE");
    } else if (name == "wrong-txn") {
        Registry r;
        auto     b = r.begin_outer();
        require(r.end({ b.txn.value + 1 }, true).code == error::WRONG_TRANSACTION, "wrong txn accepted");
        require(!r.end(b.txn, true).committed, "wrong transaction did not poison outer");
    } else if (name == "depth-underflow") {
        Registry r;
        auto     b      = r.begin_outer();
        auto     ticket = r.prepare_end(b.txn, true);
        require(ticket.finisher, "no finisher");
        require(r.enter_nested(b.txn) == error::DEPTH_UNDERFLOW, "depth underflow accepted");
        r.finalize_end(ticket, true);
    } else if (name == "depth-overflow") {
        Registry r(UINT64_MAX, 1);
        auto     b = r.begin_outer();
        require(r.enter_nested(b.txn) == error::DEPTH_OVERFLOW, "depth overflow accepted");
        require(!r.end(b.txn, true).committed, "poisoned transaction published LIVE");
    } else if (name == "rollback-idempotence") {
        Registry r;
        auto     b     = r.begin_outer();
        auto     a     = r.end(b.txn, false);
        auto     again = r.end(b.txn, false);
        require(a.code == again.code && r.rollback_count() == 1, "rollback was not idempotent");
    } else if (name == "concurrent-end") {
        Registry           r;
        auto               b = r.begin_outer();
        std::promise<void> prepared, release;
        auto               gate = release.get_future().share();
        end_result         first;
        auto               t1 = std::thread([&] {
            auto t = r.prepare_end(b.txn, true);
            require(t.finisher, "multiple finishers");
            prepared.set_value();
            gate.wait();
            first = r.finalize_end(t, true);
        });
        prepared.get_future().wait();
        require(!r.last_success() && !r.find(b.token.model), "last-success/LIVE published before effects");
        auto second_future = std::async(std::launch::async, [&] {
            auto t = r.prepare_end(b.txn, true);
            require(!t.finisher, "multiple finishers");
            return t.replay;
        });
        require(second_future.wait_for(std::chrono::milliseconds(30)) == std::future_status::timeout,
                "duplicate end did not wait for effects");
        release.set_value();
        t1.join();
        auto second = second_future.get();
        require(first.committed && second.committed && r.publication_count() == 1,
                "concurrent terminal replay differed");
    } else if (name == "abort-reuse-aba") {
        Registry r;
        auto     b = r.begin_outer();
        auto     t = r.prepare_end(b.txn, false);
        require(r.begin_outer().code == error::LOAD_BUSY, "slot/coordinator reused during rollback effects");
        r.finalize_end(t, true);
        auto next = r.begin_outer();
        require(next.token.owner.generation != b.token.owner.generation, "abort slot ABA");
        r.end(next.txn, false);
    } else if (name == "teardown-reuse-aba") {
        Registry r;
        auto     old = commit_one(r);
        auto     t   = r.prepare_teardown(old);
        require(t.finisher, "no teardown finisher");
        auto b = r.begin_outer();
        require(b.token.owner.slot != old.owner.slot, "teardown slot reused before effects");
        r.end(b.txn, false);
        require(r.finalize_teardown(t, true) == error::OK, "teardown finalize failed");
    } else if (name == "concurrent-teardown") {
        Registry r;
        auto     model = commit_one(r);
        auto     first = r.prepare_teardown(model);
        require(first.finisher, "no teardown finisher");
        auto duplicate = std::async(std::launch::async, [&] { return r.prepare_teardown(model); });
        require(duplicate.wait_for(std::chrono::milliseconds(30)) == std::future_status::timeout,
                "duplicate teardown did not wait for effects");
        require(r.finalize_teardown(first, true) == error::OK, "teardown finalize failed");
        auto replay = duplicate.get();
        require(!replay.finisher && replay.code == error::OK_ALREADY_DEAD,
                "duplicate teardown terminal replay differed");
    } else if (name == "durable-replay") {
        Registry   r;
        LoadTxnId  first_txn{};
        ModelToken first_model{};
        for (int i = 0; i < 300; ++i) {
            auto b = r.begin_outer();
            if (i == 0) {
                first_txn = b.txn;
            }
            r.end(b.txn, false);
        }
        require(r.end(first_txn, false).code == error::MISSING_SUCCESS, ">256 terminal replay was lossy");
        for (int i = 0; i < 300; ++i) {
            auto m = commit_one(r);
            if (i == 0) {
                first_model = m;
            }
            r.teardown(m);
        }
        require(r.teardown(first_model) == error::OK_ALREADY_DEAD, ">256 dead identity replay was lossy");
    } else if (name == "waiter-rollover") {
        Registry r;
        for (int round = 0; round < 300; ++round) {
            auto b      = r.begin_outer();
            auto first  = r.prepare_end(b.txn, false);
            auto waiter = std::async(std::launch::async, [&] { return r.prepare_end(b.txn, false); });
            require(waiter.wait_for(std::chrono::milliseconds(1)) == std::future_status::timeout,
                    "waiter did not block");
            r.finalize_end(first, true);
            require(waiter.get().code == error::MISSING_SUCCESS, "waiter replay mismatch");
        }
    } else if (name == "committing-teardown") {
        Registry r;
        auto     b      = r.begin_outer();
        auto     finish = r.prepare_end(b.txn, true);
        require(r.prepare_teardown(b.token).code == error::BUSY, "COMMITTING teardown was not BUSY");
        r.finalize_end(finish, true);
        require(r.teardown(b.token) == error::OK, "post-commit teardown failed");
    } else if (name == "load-effect-failure-quarantine") {
        Registry r;
        auto     b      = r.begin_outer();
        auto     finish = r.prepare_end(b.txn, true);
        auto     failed = r.finalize_end(finish, false);
        require(failed.code == error::EFFECT_FAILED && !failed.committed, "load effect failure not reported");
        auto next = r.begin_outer();
        require(next.token.owner.slot != b.token.owner.slot, "failed-load slot reused");
        r.end(next.txn, false);
        auto retry = r.prepare_teardown(b.token);
        require(retry.finisher, "failed-load quarantine not retryable");
        require(r.finalize_teardown(retry, true) == error::OK, "failed-load quarantine retry failed");
    } else if (name == "effect-failure-quarantine") {
        Registry r;
        auto     model    = commit_one(r);
        auto     teardown = r.prepare_teardown(model);
        require(r.finalize_teardown(teardown, false) == error::EFFECT_FAILED, "effect failure not reported");
        require(!r.find(model.model), "quarantined model restored LIVE");
        auto next = r.begin_outer();
        require(next.token.owner.slot != model.owner.slot, "quarantined slot reused");
        r.end(next.txn, false);
        auto retry = r.prepare_teardown(model);
        require(retry.finisher, "quarantine was not retryable");
        require(r.finalize_teardown(retry, true) == error::OK, "quarantine retry failed");
    } else if (name == "begin-allocation-failure") {
        Registry r;
        r.test_fail_next_begin_allocation();
        require(r.begin_outer().code == error::ALLOCATION_FAILED, "begin allocation failure mutated registry");
        auto b = r.begin_outer();
        require(b.token.model.value == 1 && b.token.load.value == 1 && b.token.owner.generation == 1,
                "begin failure consumed identity/generation");
        r.end(b.txn, false);
    } else if (name == "null-output") {
        Registry r;
        auto     b = r.begin_outer();
        auto     t = r.prepare_end(b.txn, true, false);
        require(t.finisher && !t.commit && t.code == error::NULL_OUTPUT, "null outer token output accepted");
        auto e = r.finalize_end(t, true);
        require(!e.committed && !r.last_success(), "null output published LIVE");
    } else if (name == "stale-token-after-reuse") {
        Registry r;
        auto     old = commit_one(r);
        require(r.teardown(old) == error::OK, "teardown failed");
        auto fresh = commit_one(r);
        require(fresh.owner.slot == old.owner.slot && fresh.owner.generation != old.owner.generation,
                "slot not reused with generation");
        require(r.teardown(old) == error::STALE_IDENTITY, "stale token after reuse accepted");
    } else if (name == "slot-exhaustion") {
        Registry r;
        for (int i = 0; i < 32; ++i) {
            (void) commit_one(r);
        }
        const auto pubs = r.publication_count();
        auto       fail = r.begin_outer();
        require(fail.code == error::SLOT_EXHAUSTED && r.publication_count() == pubs && r.live_mask() == UINT32_MAX,
                "33rd slot had side effects");
    } else if (name == "model-id-overflow") {
        Registry r;
        r.test_set_next_ids(UINT64_MAX, 7);
        auto max = r.begin_outer();
        require(max.token.model.value == UINT64_MAX, "model max not issued");
        r.end(max.txn, false);
        require(r.begin_outer().code == error::ID_EXHAUSTED, "model id wrapped after max");
    } else if (name == "load-txn-id-overflow") {
        Registry r;
        r.test_set_next_ids(7, UINT64_MAX);
        auto max = r.begin_outer();
        require(max.token.load.value == UINT64_MAX, "load max not issued");
        r.end(max.txn, false);
        require(r.begin_outer().code == error::ID_EXHAUSTED, "load txn wrapped after max");
    } else if (name == "slot-generation-overflow") {
        Registry r;
        r.test_set_slot_generation(0, UINT64_MAX - 1);
        auto max = r.begin_outer();
        require(max.token.owner.generation == UINT64_MAX, "slot max generation not issued");
        r.end(max.txn, false);
        auto next = r.begin_outer();
        require(next.token.owner.slot != 0, "exhausted slot was not retired");
        r.end(next.txn, false);
    } else if (name == "invocation-id-counter-overflow") {
        CheckedCounter c(UINT64_MAX);
        uint64_t       v = 0;
        require(c.take(v) == error::OK && v == UINT64_MAX && c.take(v) == error::ID_EXHAUSTED,
                "checked counter wrapped");
    } else if (name == "stale-generation") {
        Registry r(UINT64_MAX, UINT64_MAX, mutation);
        auto     first = commit_one(r);
        require(r.teardown(first) == error::OK, "teardown failed");
        auto second = commit_one(r);
        if (mutation == test_mutation::M1_SKIP_GENERATION) {
            require(second.owner.generation != first.owner.generation, "stale slot generation accepted");
        }
        require(second.owner.generation != first.owner.generation, "stale slot generation accepted");
    } else if (name == "multi-model" || name == "sequential-aba") {
        Registry r;
        auto     a       = commit_one(r, { 11, 7, tier_verdict::MIXED });
        auto     a_state = r.find(a.model);
        require(a_state && a_state->planned_host_bytes == 11 && a_state->actual_host_bytes == 7 &&
                    a_state->verdict == tier_verdict::MIXED,
                "published planner reporting snapshot mismatch");
        auto b = commit_one(r, { 22, 0, tier_verdict::HOST });
        require(r.find(a.model) == a_state && r.find(b.model), "B load changed A");
        auto bad = r.begin_outer();
        r.end(bad.txn, false);
        require(r.find(a.model) == a_state && r.find(b.model), "failed load changed live model");
        if (name == "sequential-aba") {
            r.teardown(a);
            auto a2 = commit_one(r, { 11, 7, tier_verdict::MIXED });
            require(a2.model.value != a.model.value && a2.owner.generation != a.owner.generation &&
                        r.find(a2.model)->planned_host_bytes == 11,
                    "A-B-A identity/plan contamination");
        }
    } else {
        std::cerr << "unknown case: " << name << '\n';
        std::exit(2);
    }
}

int main(int argc, char ** argv) {
    std::vector<std::string> cases;
    test_mutation            mutation = test_mutation::NONE;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--case" && i + 1 < argc) {
            cases.emplace_back(argv[++i]);
        } else if (arg == "--mutation" && i + 1 < argc) {
            std::string m = argv[++i];
            mutation      = m == "M1" ? test_mutation::M1_SKIP_GENERATION :
                            m == "M2" ? test_mutation::M2_NESTED_COMMIT :
                                        test_mutation::M3_CLEAR_POISON;
        }
    }
    if (cases.empty()) {
        cases = { "nested-success" };
    }
    for (const auto & c : cases) {
        run_case(c, mutation);
    }
    return 0;
}
