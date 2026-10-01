// Host test for L0, the re-plan transaction mutex, and the always-compiled
// witness (llama.cpp-moua's token, defined by llama.cpp-zhcn C6; H9's real-object
// half).  The mechanics under test are the REAL ggml_sycl_replan_token and
// GGML_SYCL_WITNESS of unified-cache.cpp, built into the private fixture carrier:
//
//   * one outermost lock; a nested acquire does not lock and never changes the
//     outermost kind; the outermost destructor unlocks;
//   * the held state is per thread; another thread's try form fails while L0 is
//     held and succeeds once it is released; a blocking acquire waits;
//   * the accessor answers ANY / TRANSACTION / LOAD / LIFECYCLE against the
//     OUTERMOST token only;
//   * the three illegal nestings and the outermost-only form are witness
//     failures, each scored by its message in a re-exec'd child (the test builds
//     Release with -DNDEBUG, so a death arm that passes here is not an assert);
//   * with GGML_SYCL_WITNESS_CHECKS=0 the same illegal nesting runs unchecked and
//     the witness does not evaluate its condition.
//
// Nothing here touches a device: the registration pins the selector to the
// OpenCL CPU device and no queue is created.
//
// Usage:
//   ./build/bin/test-sycl-replan-token             # every case
//   ./build/bin/test-sycl-replan-token <child>     # one death child (see main)

#include "ggml.h"
#include "unified-cache.hpp"

#include <sys/wait.h>

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

using namespace ggml_sycl;

namespace {

int g_failures = 0;

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            ++g_failures;                                                       \
        }                                                                       \
    } while (0)

constexpr ggml_sycl_replan_kind TXN  = GGML_SYCL_REPLAN_KIND_TRANSACTION;
constexpr ggml_sycl_replan_kind LOAD = GGML_SYCL_REPLAN_KIND_LOAD;
constexpr ggml_sycl_replan_kind LIFE = GGML_SYCL_REPLAN_KIND_LIFECYCLE;
constexpr ggml_sycl_replan_kind ANY  = GGML_SYCL_REPLAN_KIND_ANY;

void test_single() {
    CHECK(!ggml_sycl_replan_token_held(), "free: not held (ANY)");
    CHECK(!ggml_sycl_replan_token_held(TXN), "free: not held (TRANSACTION)");
    {
        ggml_sycl_replan_token t(TXN);
        CHECK(t.owns(), "a blocking acquire owns");
        CHECK(ggml_sycl_replan_token_held(), "held: ANY");
        CHECK(ggml_sycl_replan_token_held(ANY), "held: ANY spelled out");
        CHECK(ggml_sycl_replan_token_held(TXN), "held: TRANSACTION");
        CHECK(!ggml_sycl_replan_token_held(LOAD), "held: not LOAD");
        CHECK(!ggml_sycl_replan_token_held(LIFE), "held: not LIFECYCLE");
    }
    CHECK(!ggml_sycl_replan_token_held(), "released: not held");
    CHECK(!ggml_sycl_replan_token_held(TXN), "released: kind forgotten");
}

// Every legal (outer, inner) pair: the inner is a no-op, the outermost kind
// stays, and only the outermost destructor releases.
void test_legal_nesting() {
    const ggml_sycl_replan_kind kinds[3] = { TXN, LOAD, LIFE };
    for (const ggml_sycl_replan_kind outer : kinds) {
        for (const ggml_sycl_replan_kind inner : kinds) {
            const bool illegal = (inner == TXN && (outer == LOAD || outer == LIFE)) || (inner == LOAD && outer == TXN);
            if (illegal) {
                continue;
            }
            const std::string name =
                std::string(ggml_sycl_replan_kind_name(inner)) + " under " + ggml_sycl_replan_kind_name(outer);
            {
                ggml_sycl_replan_token o(outer);
                {
                    ggml_sycl_replan_token i(inner);
                    CHECK(i.owns(), (name + ": the nested acquire owns").c_str());
                    CHECK(ggml_sycl_replan_token_held(outer), (name + ": the outermost kind stays").c_str());
                    CHECK(inner == outer || !ggml_sycl_replan_token_held(inner),
                          (name + ": the inner kind is not reported").c_str());
                }
                CHECK(ggml_sycl_replan_token_held(outer), (name + ": still held after the inner drops").c_str());
            }
            CHECK(!ggml_sycl_replan_token_held(), (name + ": released with the outermost").c_str());
        }
    }
    // A three-deep stack releases once, at the outermost.
    {
        ggml_sycl_replan_token a(TXN);
        {
            ggml_sycl_replan_token b(LIFE);
            {
                ggml_sycl_replan_token c(TXN);
            }
            CHECK(ggml_sycl_replan_token_held(TXN), "depth 3 -> 2: held");
        }
        CHECK(ggml_sycl_replan_token_held(TXN), "depth 2 -> 1: held");
    }
    CHECK(!ggml_sycl_replan_token_held(), "depth 0: released");
}

void test_threads() {
    std::atomic<bool> other_try_owned{ true };
    std::atomic<bool> other_saw_held{ true };
    {
        ggml_sycl_replan_token t(LOAD);
        std::thread            th([&] {
            other_saw_held.store(ggml_sycl_replan_token_held());
            ggml_sycl_replan_token probe(LIFE, std::try_to_lock);
            other_try_owned.store(probe.owns());
            // a failed try changed nothing on that thread
            if (ggml_sycl_replan_token_held()) {
                other_saw_held.store(true);
            }
        });
        th.join();
    }
    CHECK(!other_saw_held.load(), "the held state is per thread: another thread holds nothing");
    CHECK(!other_try_owned.load(), "the try form fails while another thread holds L0");

    // After the release the try form owns, and takes the outermost kind.
    {
        ggml_sycl_replan_token t(LIFE, std::try_to_lock);
        CHECK(t.owns(), "the try form owns a free L0");
        CHECK(ggml_sycl_replan_token_held(LIFE), "the try form records its kind");
    }
    CHECK(!ggml_sycl_replan_token_held(), "the try form releases");

    // A try under a hold of this thread is a nested hold, not a failure.
    {
        ggml_sycl_replan_token o(TXN);
        ggml_sycl_replan_token t(LIFE, std::try_to_lock);
        CHECK(t.owns(), "the try form nests under this thread's own hold");
    }

    // A blocking acquire on another thread waits for the release and then runs.
    std::atomic<bool> entered{ false };
    std::thread       waiter;
    {
        ggml_sycl_replan_token t(TXN);
        waiter                  = std::thread([&] {
            ggml_sycl_replan_token w(TXN);
            entered.store(true);
        });
        // The waiter cannot enter while this thread holds L0; a try on a third
        // thread proves the mutex is locked, which is what the waiter queues on.
        bool        third_owned = true;
        std::thread third([&] {
            ggml_sycl_replan_token p(TXN, std::try_to_lock);
            third_owned = p.owns();
        });
        third.join();
        CHECK(!third_owned, "L0 is locked while the waiter is queued");
    }
    waiter.join();
    CHECK(entered.load(), "the waiter ran after the release");
}

// A re-exec'd child that takes `args`, with `env` prefixed, and returns its
// merged output and exit status.
bool run_child(const char * self, const char * env, const char * child, std::string & out, int & status) {
    const std::string cmd = std::string("GGML_NO_BACKTRACE=1 ") + env + " " + self + " " + child + " 2>&1";
    FILE *            p   = popen(cmd.c_str(), "r");
    if (!p) {
        return false;
    }
    char buf[512];
    while (std::fgets(buf, sizeof(buf), p)) {
        out += buf;
    }
    status = pclose(p);
    return true;
}

void test_death(const char * self, const char * child, const char * message) {
    std::string out;
    int         status = 0;
    CHECK(run_child(self, "GGML_SYCL_WITNESS_CHECKS=1", child, out, status), "death child started");
    // popen runs the child under a shell, which reports a signal as 128 + signo.
    const bool aborted = (WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT) ||
                         (WIFEXITED(status) && WEXITSTATUS(status) == 128 + SIGABRT);
    const bool printed = out.find(message) != std::string::npos;
    CHECK(aborted, (std::string(child) + ": the child aborted").c_str());
    CHECK(printed, (std::string(child) + ": the child printed its own message").c_str());
    CHECK(out.find("returned without aborting") == std::string::npos,
          (std::string(child) + ": the child did not run past the witness").c_str());
    if (!aborted || !printed) {
        std::fprintf(stderr, "%s printed:\n%s", child, out.c_str());
    }
}

void test_unchecked(const char * self, const char * child) {
    std::string out;
    int         status = 0;
    CHECK(run_child(self, "GGML_SYCL_WITNESS_CHECKS=0", child, out, status), "unchecked child started");
    CHECK(status == 0, (std::string(child) + ": runs to the end with the witness off").c_str());
    CHECK(out.find("returned without aborting") != std::string::npos,
          (std::string(child) + ": reached its marker").c_str());
    if (status != 0) {
        std::fprintf(stderr, "%s printed:\n%s", child, out.c_str());
    }
}

int child_marker() {
    std::printf("returned without aborting\n");
    return 0;
}

int nest_txn_under_load() {
    ggml_sycl_replan_token o(LOAD);
    ggml_sycl_replan_token i(TXN);
    return child_marker();
}

int nest_txn_under_lifecycle() {
    ggml_sycl_replan_token o(LIFE);
    ggml_sycl_replan_token i(TXN);
    return child_marker();
}

int nest_load_under_txn() {
    ggml_sycl_replan_token o(TXN);
    ggml_sycl_replan_token i(LOAD);
    return child_marker();
}

int outermost_only_held() {
    ggml_sycl_replan_token o(TXN);
    ggml_sycl_replan_token i(LIFE, ggml_sycl_replan_outermost_only);
    return child_marker();
}

int outermost_only_free() {
    ggml_sycl_replan_token i(LIFE, ggml_sycl_replan_outermost_only);
    CHECK(ggml_sycl_replan_token_held(LIFE), "the outermost-only form locks when free");
    return g_failures == 0 ? child_marker() : 1;
}

int witness_false() {
    GGML_SYCL_WITNESS(1 + 1 == 3, "[REPLAN-TOKEN] test witness fired");
    return child_marker();
}

int witness_lazy() {
    int evaluated = 0;
    GGML_SYCL_WITNESS((++evaluated, false), "[REPLAN-TOKEN] test witness fired");
    if (evaluated != 0) {
        std::fprintf(stderr, "the condition was evaluated with the witness off\n");
        return 1;
    }
    return child_marker();
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc > 1) {
        const char * c = argv[1];
        if (std::strcmp(c, "nest-txn-under-load") == 0) {
            return nest_txn_under_load();
        }
        if (std::strcmp(c, "nest-txn-under-lifecycle") == 0) {
            return nest_txn_under_lifecycle();
        }
        if (std::strcmp(c, "nest-load-under-txn") == 0) {
            return nest_load_under_txn();
        }
        if (std::strcmp(c, "outermost-only-held") == 0) {
            return outermost_only_held();
        }
        if (std::strcmp(c, "outermost-only-free") == 0) {
            return outermost_only_free();
        }
        if (std::strcmp(c, "witness-false") == 0) {
            return witness_false();
        }
        if (std::strcmp(c, "witness-lazy") == 0) {
            return witness_lazy();
        }
        std::fprintf(stderr, "unknown child %s\n", c);
        return 2;
    }
    test_single();
    test_legal_nesting();
    test_threads();
    test_death(argv[0], "nest-txn-under-load", "[REPLAN-TOKEN] illegal nesting: TRANSACTION under LOAD");
    test_death(argv[0], "nest-txn-under-lifecycle", "[REPLAN-TOKEN] illegal nesting: TRANSACTION under LIFECYCLE");
    test_death(argv[0], "nest-load-under-txn", "[REPLAN-TOKEN] illegal nesting: LOAD under TRANSACTION");
    test_death(argv[0], "outermost-only-held", "[REPLAN-TOKEN] release proc entered with L0 held");
    test_death(argv[0], "witness-false", "[REPLAN-TOKEN] test witness fired");
    // The same arms with the switch off reach the unchecked path.
    test_unchecked(argv[0], "nest-txn-under-load");
    test_unchecked(argv[0], "nest-load-under-txn");
    test_unchecked(argv[0], "outermost-only-held");
    test_unchecked(argv[0], "witness-false");
    test_unchecked(argv[0], "witness-lazy");
    // Controls: the outermost-only form is legal on a free thread.
    test_unchecked(argv[0], "outermost-only-free");
    if (g_failures != 0) {
        std::fprintf(stderr, "test-sycl-replan-token: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test-sycl-replan-token: all ok\n");
    return 0;
}
