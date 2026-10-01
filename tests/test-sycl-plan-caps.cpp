// Test: the chunk-cap core, the per-context plan_caps copy and the plan scopes
// (zhcn-design §2.4, §3.4; H6a).
//
// A device buft made by the PRIVATE_TESTING factory touches no device, so this runs
// where there is none.  The copy is frozen through the production core and store
// (ggml_backend_sycl_plan_caps_freeze_core), and the buft's in-scope get_max_size is
// read through the production branch.  The wrapper is never called: with no device its
// backing-kind read would be a plan defect, so the VM freeze path is lead-run on a card.
//
//  * the core: USM and VM expressions, the all-zero capacities guard, A's fallback and
//    the named refusal;
//  * the state machine accepts only UNARMED -> FREEZING -> FROZEN;
//  * an in-scope read returns the copy's value, records it, and calls no zone code;
//  * a read the copy cannot answer, outside FREEZING, is E5: the per-process constant,
//    nothing stored, the scope marked failed;
//  * the SYCL_Host buft freezes under a TRANSACTION token in FREEZING, once;
//  * a scope nest, and a scope with the wrong copy for its mode, are refused;
//  * the live counter returns to its start after free.
//
// Usage:
//   ./build/bin/test-sycl-plan-caps

#include "chunk-cap.hpp"
#include "common.hpp"
#include "ggml-sycl.h"
#include "unified-cache.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

int g_failures = 0;

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            ++g_failures;                                                       \
        }                                                                       \
    } while (0)

constexpr size_t MiB = 1024ULL * 1024ULL;
constexpr size_t GiB = 1024ULL * MiB;

void test_core() {
    // USM: min(2 GiB, A); the capacities are not read.
    auto r = ggml_sycl_chunk_cap_core(false, 9 * GiB, 9 * GiB, 9 * GiB, 16 * GiB, 16 * GiB);
    CHECK(r.cap == 2 * GiB && !r.refusal, "USM: 2 GiB when A is larger");
    r = ggml_sycl_chunk_cap_core(false, 0, 0, 0, 1 * GiB, 4 * GiB);
    CHECK(r.cap == 1 * GiB && !r.refusal, "USM: A bounds it");
    // A falls back to max_alloc when safe_alloc is 0.
    r = ggml_sycl_chunk_cap_core(false, 0, 0, 0, 0, 512 * MiB);
    CHECK(r.cap == 512 * MiB && !r.refusal, "A falls back to max_alloc");
    // Both 0: the named refusal.
    r = ggml_sycl_chunk_cap_core(false, 0, 0, 0, 0, 0);
    CHECK(r.refusal && std::strstr(r.refusal, "no allocation limit known"), "both A inputs 0 is the named refusal");
    r = ggml_sycl_chunk_cap_core(true, 512 * MiB, 0, 512 * MiB, 0, 0);
    CHECK(r.refusal != nullptr, "the refusal holds on VM too");

    // VM: min(max(R, KV, S), 2 GiB, A).
    r = ggml_sycl_chunk_cap_core(true, 512 * MiB, 0, 512 * MiB, 16 * GiB, 16 * GiB);
    CHECK(r.cap == 512 * MiB, "VM: the largest planned capacity bounds it");
    r = ggml_sycl_chunk_cap_core(true, 1536 * MiB, 0, 512 * MiB, 16 * GiB, 16 * GiB);
    CHECK(r.cap == 1536 * MiB, "VM: max over RUNTIME, KV, SCRATCH");
    r = ggml_sycl_chunk_cap_core(true, 0, 3 * GiB, 0, 16 * GiB, 16 * GiB);
    CHECK(r.cap == 2 * GiB, "VM: never above 2 GiB");
    r = ggml_sycl_chunk_cap_core(true, 1 * GiB, 0, 0, 256 * MiB, 0);
    CHECK(r.cap == 256 * MiB, "VM: A bounds it");
    // All three 0: no capacity configured, so the 2 GiB bound (then A).
    r = ggml_sycl_chunk_cap_core(true, 0, 0, 0, 16 * GiB, 16 * GiB);
    CHECK(r.cap == 2 * GiB, "VM: all capacities 0 answers 2 GiB");
    // The capacities do not matter on USM.
    CHECK(ggml_sycl_chunk_cap_core(false, 512 * MiB, 0, 512 * MiB, 16 * GiB, 16 * GiB).cap == 2 * GiB,
          "USM ignores the capacities it is handed");
}

void test_state_machine() {
    ggml_backend_sycl_plan_caps_t caps = ggml_backend_sycl_plan_caps_new();
    CHECK(caps != nullptr, "caps_new");
    CHECK(!ggml_backend_sycl_plan_caps_set_state(caps, GGML_SYCL_PLAN_CAPS_FROZEN), "UNARMED -> FROZEN is refused");
    CHECK(!ggml_backend_sycl_plan_caps_set_state(caps, GGML_SYCL_PLAN_CAPS_UNARMED), "UNARMED -> UNARMED is refused");
    CHECK(ggml_backend_sycl_plan_caps_set_state(caps, GGML_SYCL_PLAN_CAPS_FREEZING), "UNARMED -> FREEZING");
    CHECK(!ggml_backend_sycl_plan_caps_set_state(caps, GGML_SYCL_PLAN_CAPS_FREEZING),
          "FREEZING -> FREEZING is refused");
    CHECK(!ggml_backend_sycl_plan_caps_set_state(caps, GGML_SYCL_PLAN_CAPS_UNARMED), "FREEZING -> UNARMED is refused");
    CHECK(ggml_backend_sycl_plan_caps_set_state(caps, GGML_SYCL_PLAN_CAPS_FROZEN), "FREEZING -> FROZEN");
    CHECK(!ggml_backend_sycl_plan_caps_set_state(caps, GGML_SYCL_PLAN_CAPS_FREEZING), "FROZEN has no way back");
    CHECK(!ggml_backend_sycl_plan_caps_set_state(nullptr, GGML_SYCL_PLAN_CAPS_FREEZING), "a null copy is refused");
    ggml_backend_sycl_plan_caps_free(caps);
}

void test_in_scope_reads() {
    ggml_backend_buffer_type_t    buft = ggml_backend_sycl_buffer_type_make_for_testing(0);
    ggml_backend_sycl_plan_caps_t caps = ggml_backend_sycl_plan_caps_new();

    // Freeze through the production core and store: R = S = 512 MiB, KV 0, VM.
    const size_t frozen =
        ggml_backend_sycl_plan_caps_freeze_core(caps, buft, true, 512 * MiB, 0, 512 * MiB, 16 * GiB, 16 * GiB);
    CHECK(frozen == 512 * MiB, "freeze_core stores the core's value");
    CHECK(ggml_backend_sycl_plan_caps_freeze_count(caps, buft) == 1, "one freeze counted for the buft");

    void * scope = ggml_backend_sycl_plan_scope_open(1, GGML_SYCL_PLAN_SCOPE_MEASURE, caps);
    CHECK(scope != nullptr, "MEASURE scope opens with a copy");
    CHECK(ggml_backend_buft_get_max_size(buft) == 512 * MiB, "an in-scope read returns the copy's value");
    CHECK(ggml_backend_buft_get_max_size(buft) == 512 * MiB, "a second read returns the same value");
    CHECK(ggml_backend_sycl_plan_scope_failure(scope) == nullptr, "a held read is not a failure");
    CHECK(ggml_backend_sycl_plan_caps_freeze_count(caps, buft) == 1, "reads do not freeze again");

    // A second scope on this thread is a nest: refused, the outer scope is untouched.
    CHECK(ggml_backend_sycl_plan_scope_open(2, GGML_SYCL_PLAN_SCOPE_ALLOC, caps) == nullptr,
          "a nested scope is refused");
    CHECK(ggml_backend_buft_get_max_size(buft) == 512 * MiB, "the outer scope still answers");
    ggml_backend_sycl_plan_scope_close(scope);

    // A scope with the wrong copy for its mode is refused.
    CHECK(ggml_backend_sycl_plan_scope_open(3, GGML_SYCL_PLAN_SCOPE_MEASURE, nullptr) == nullptr,
          "MEASURE with no copy is refused");
    CHECK(ggml_backend_sycl_plan_scope_open(3, GGML_SYCL_PLAN_SCOPE_ALLOC, nullptr) == nullptr,
          "ALLOC with no copy is refused");
    CHECK(ggml_backend_sycl_plan_scope_open(3, GGML_SYCL_PLAN_SCOPE_LOAD_MEASURE, caps) == nullptr,
          "LOAD_MEASURE holding a copy is refused");

    // Closed: the scope no longer answers, and the next scope sees the same copy.
    scope = ggml_backend_sycl_plan_scope_open(4, GGML_SYCL_PLAN_SCOPE_ALLOC, caps);
    CHECK(scope && ggml_backend_buft_get_max_size(buft) == 512 * MiB, "a later scope reads the same copy");
    ggml_backend_sycl_plan_scope_close(scope);
    ggml_backend_sycl_plan_caps_free(caps);
}

void test_unfrozen_read_is_e5() {
    ggml_backend_buffer_type_t    buft = ggml_backend_sycl_buffer_type_make_for_testing(0);
    ggml_backend_sycl_plan_caps_t caps = ggml_backend_sycl_plan_caps_new();

    // UNARMED: a scope opened before the fixpoint entered.
    void *       scope = ggml_backend_sycl_plan_scope_open(5, GGML_SYCL_PLAN_SCOPE_MEASURE, caps);
    const size_t read  = ggml_backend_buft_get_max_size(buft);
    CHECK(read == 2 * GiB, "an unheld read returns the per-process constant (no device: 2 GiB)");
    const char * failure = ggml_backend_sycl_plan_scope_failure(scope);
    CHECK(failure && std::strstr(failure, "chunk cap unfrozen during this reserve"),
          "the scope is marked failed by name");
    CHECK(ggml_backend_sycl_plan_caps_freeze_count(caps, buft) == 0, "nothing was stored by the E5 read");
    ggml_backend_sycl_plan_scope_close(scope);

    // FROZEN: a buft the fixpoint never measured.
    CHECK(ggml_backend_sycl_plan_caps_set_state(caps, GGML_SYCL_PLAN_CAPS_FREEZING), "arm");
    CHECK(ggml_backend_sycl_plan_caps_set_state(caps, GGML_SYCL_PLAN_CAPS_FROZEN), "freeze");
    scope = ggml_backend_sycl_plan_scope_open(5, GGML_SYCL_PLAN_SCOPE_ALLOC, caps);
    (void) ggml_backend_buft_get_max_size(buft);
    CHECK(ggml_backend_sycl_plan_scope_failure(scope) != nullptr, "FROZEN, unheld: E5 as well");
    CHECK(ggml_backend_sycl_plan_caps_freeze_count(caps, buft) == 0, "FROZEN never freezes a new buft");
    ggml_backend_sycl_plan_scope_close(scope);
    ggml_backend_sycl_plan_caps_free(caps);
}

void test_host_buft_freezes_once() {
    ggml_backend_buffer_type_t    host = ggml_backend_sycl_host_buffer_type_make_for_testing();
    ggml_backend_sycl_plan_caps_t caps = ggml_backend_sycl_plan_caps_new();
    CHECK(ggml_backend_sycl_plan_caps_set_state(caps, GGML_SYCL_PLAN_CAPS_FREEZING), "arm for the host freeze");
    size_t first = 0;
    {
        // The fixpoint holds the TRANSACTION token from before its first MEASURE.
        ggml_sycl::ggml_sycl_replan_token token(ggml_sycl::GGML_SYCL_REPLAN_KIND_TRANSACTION);
        void * scope = ggml_backend_sycl_plan_scope_open(6, GGML_SYCL_PLAN_SCOPE_MEASURE, caps);
        first        = ggml_backend_buft_get_max_size(host);
        CHECK(first > 0 && first <= 2 * GiB, "the host cap is bounded by one pool chunk");
        CHECK(ggml_backend_sycl_plan_scope_failure(scope) == nullptr, "the freeze under the token is clean");
        CHECK(ggml_backend_buft_get_max_size(host) == first, "a second read is the held value");
        ggml_backend_sycl_plan_scope_close(scope);
    }
    CHECK(ggml_backend_sycl_plan_caps_freeze_count(caps, host) == 1, "the host buft froze exactly once");
    CHECK(ggml_backend_sycl_plan_caps_set_state(caps, GGML_SYCL_PLAN_CAPS_FROZEN), "freeze the copy");
    void * scope = ggml_backend_sycl_plan_scope_open(6, GGML_SYCL_PLAN_SCOPE_ALLOC, caps);
    CHECK(ggml_backend_buft_get_max_size(host) == first, "ALLOC reads the value MEASURE froze");
    ggml_backend_sycl_plan_scope_close(scope);
    ggml_backend_sycl_plan_caps_free(caps);
}

void test_live_counter() {
    const int                     start = ggml_backend_sycl_plan_caps_live();
    ggml_backend_sycl_plan_caps_t a     = ggml_backend_sycl_plan_caps_new();
    ggml_backend_sycl_plan_caps_t b     = ggml_backend_sycl_plan_caps_new();
    CHECK(ggml_backend_sycl_plan_caps_live() == start + 2, "two copies are live");
    ggml_backend_sycl_plan_caps_free(a);
    ggml_backend_sycl_plan_caps_free(nullptr);
    ggml_backend_sycl_plan_caps_free(b);
    CHECK(ggml_backend_sycl_plan_caps_live() == start, "the counter returns to its start");
}

}  // namespace

int main() {
    setenv("GGML_SYCL_STRICT_LEASES", "0", 1);
    test_core();
    test_state_machine();
    test_in_scope_reads();
    test_unfrozen_read_is_e5();
    test_host_buft_freezes_once();
    test_live_counter();
    if (g_failures != 0) {
        std::fprintf(stderr, "test-sycl-plan-caps: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test-sycl-plan-caps: all ok\n");
    return 0;
}
