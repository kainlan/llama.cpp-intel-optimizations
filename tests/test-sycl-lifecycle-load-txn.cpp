#include "model-lifecycle.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace ggml_sycl::lifecycle;

static void require(bool value, const char * marker) {
    if (!value) { std::cerr << marker << '\n'; std::exit(1); }
}

static ModelToken commit_one(Registry & r) {
    auto b = r.begin_outer(); require(b.code == error::OK, "begin failed");
    auto e = r.end(b.txn, true); require(e.code == error::OK && e.committed, "outer commit failed");
    return e.token;
}

static void run_case(const std::string & name, test_mutation mutation) {
    if (name == "nested-success") {
        Registry r(UINT64_MAX, UINT64_MAX, mutation); auto b = r.begin_outer();
        require(r.enter_nested(b.txn) == error::OK, "nested begin failed");
        auto inner = r.end(b.txn, true);
        if (mutation == test_mutation::M2_NESTED_COMMIT) require(!inner.committed, "nested load committed");
        require(inner.code == error::NESTED && !inner.committed, "nested load committed");
        require(r.end(b.txn, true).committed && r.publication_count() == 1, "outer did not commit exactly once");
    } else if (name == "inner-failure" || name == "cancel") {
        Registry r(UINT64_MAX, UINT64_MAX, mutation); auto b = r.begin_outer(); r.enter_nested(b.txn);
        r.end(b.txn, false); auto e = r.end(b.txn, true);
        if (mutation == test_mutation::M3_CLEAR_POISON) require(!e.committed, "poisoned transaction published LIVE");
        require(!e.committed && !r.find(b.token.model), "poisoned transaction published LIVE");
    } else if (name == "missing-success") {
        Registry r(UINT64_MAX, UINT64_MAX, mutation); auto b = r.begin_outer(); auto e = r.end(b.txn, false);
        require(e.code == error::MISSING_SUCCESS && !r.last_success(), "missing success published LIVE");
    } else if (name == "wrong-txn") {
        Registry r; auto b = r.begin_outer(); require(r.end({b.txn.value + 1}, true).code == error::WRONG_TRANSACTION, "wrong txn accepted"); require(!r.end(b.txn, true).committed, "wrong transaction did not poison outer");
    } else if (name == "depth-underflow") {
        Registry r; auto b = r.begin_outer(); r.end(b.txn, false); require(r.enter_nested(b.txn) == error::DEPTH_UNDERFLOW, "depth underflow accepted");
    } else if (name == "depth-overflow") {
        Registry r(UINT64_MAX, 1); auto b = r.begin_outer(); require(r.enter_nested(b.txn) == error::DEPTH_OVERFLOW, "depth overflow accepted"); require(!r.end(b.txn, true).committed, "poisoned transaction published LIVE");
    } else if (name == "rollback-idempotence") {
        Registry r; auto b = r.begin_outer(); auto a = r.end(b.txn, false); auto again = r.end(b.txn, false);
        require(a.code == again.code && r.rollback_count() == 1, "rollback was not idempotent");
    } else if (name == "slot-exhaustion") {
        Registry r; std::vector<ModelToken> tokens; for (int i = 0; i < 32; ++i) tokens.push_back(commit_one(r));
        const auto pubs = r.publication_count(); auto fail = r.begin_outer();
        require(fail.code == error::SLOT_EXHAUSTED && r.publication_count() == pubs && r.live_mask() == UINT32_MAX, "33rd slot had side effects");
    } else if (name == "model-id-overflow") {
        Registry r; r.test_set_next_ids(0, 1); require(r.begin_outer().code == error::ID_EXHAUSTED, "model id wrapped");
    } else if (name == "load-txn-id-overflow") {
        Registry r; r.test_set_next_ids(1, 0); require(r.begin_outer().code == error::ID_EXHAUSTED, "load txn id wrapped");
    } else if (name == "slot-generation-overflow") {
        Registry r; r.test_set_slot_generation(0, UINT64_MAX); auto b = r.begin_outer(); require(b.code == error::OK && b.token.owner.slot != 0, "exhausted slot was not retired"); r.end(b.txn, false);
    } else if (name == "invocation-id-counter-overflow") {
        CheckedCounter c(UINT64_MAX); uint64_t v = 0; require(c.take(v) == error::OK && v == UINT64_MAX && c.take(v) == error::ID_EXHAUSTED, "checked counter wrapped");
    } else if (name == "stale-generation") {
        Registry r(UINT64_MAX, UINT64_MAX, mutation); auto first = commit_one(r); require(r.teardown(first) == error::OK, "teardown failed"); auto second = commit_one(r);
        if (mutation == test_mutation::M1_SKIP_GENERATION) require(second.owner.generation != first.owner.generation, "stale slot generation accepted");
        require(second.owner.generation != first.owner.generation, "stale slot generation accepted");
    } else { std::cerr << "unknown case: " << name << '\n'; std::exit(2); }
}

int main(int argc, char ** argv) {
    std::vector<std::string> cases; test_mutation mutation = test_mutation::NONE;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--case" && i + 1 < argc) cases.emplace_back(argv[++i]);
        else if (arg == "--mutation" && i + 1 < argc) {
            std::string m = argv[++i]; mutation = m == "M1" ? test_mutation::M1_SKIP_GENERATION : m == "M2" ? test_mutation::M2_NESTED_COMMIT : test_mutation::M3_CLEAR_POISON;
        }
    }
    if (cases.empty()) cases = {"nested-success"};
    for (const auto & c : cases) run_case(c, mutation);
    return 0;
}
