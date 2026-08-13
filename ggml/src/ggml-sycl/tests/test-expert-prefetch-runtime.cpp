// Runtime ownership and lifecycle tests for ExpertPrefetcher.
//
// These tests inject already-submitted requests so they exercise the public
// handoff/cancellation behavior without manufacturing model-registry state.

#include "../expert-prefetch.hpp"
#include "sycl-test-skip.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <mutex>
#include <optional>
#include <unordered_map>

using ggml_sycl::expert_prefetch_cache_key;
using ggml_sycl::expert_prefetch_cache_key_hash;
using ggml_sycl::ExpertPrefetcher;
using ggml_sycl::prefetch_request;

namespace {

template <typename Tag, typename Tag::type Member> struct private_member_access {
    friend typename Tag::type get(Tag) { return Member; }
};

struct inflight_tag {
    using type = std::unordered_map<expert_prefetch_cache_key, prefetch_request, expert_prefetch_cache_key_hash>
                ExpertPrefetcher::*;
    friend type get(inflight_tag);
};
template struct private_member_access<inflight_tag, &ExpertPrefetcher::inflight_>;

struct mutex_tag {
    using type = std::mutex ExpertPrefetcher::*;
    friend type             get(mutex_tag);
};
template struct private_member_access<mutex_tag, &ExpertPrefetcher::mutex_>;

struct gc_tag {
    using type = void (ExpertPrefetcher::*)();
    friend type get(gc_tag);
};
template struct private_member_access<gc_tag, &ExpertPrefetcher::gc_completed>;

[[noreturn]] void fail(const char * message) {
    std::fprintf(stderr, "FAILED: %s\n", message);
    std::exit(1);
}

void require(bool condition, const char * message) {
    if (!condition) {
        fail(message);
    }
}

ggml_sycl_cache_id make_id(uint64_t model_id, uint64_t expert_identity) {
    ggml_sycl_cache_id id{};
    id.valid         = true;
    id.model_id      = model_id;
    id.aux_id        = expert_identity;
    id.name_hash     = expert_identity;
    id.nbytes        = sizeof(int);
    id.type          = GGML_TYPE_F32;
    id.tp_world_size = 1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        id.ne[i]          = 1;
        id.tp_local_ne[i] = 1;
    }
    return id;
}

prefetch_request make_request(uint64_t model_id, int layer, int expert, int * payload, const sycl::event & ready) {
    prefetch_request request{};
    request.key.id     = make_id(model_id, static_cast<uint64_t>(expert + 1));
    request.key.layout = GGML_LAYOUT_SOA;
    request.event      = ready;
    request.device_ptr = payload;
    request.destination_owner =
        ggml_sycl::mem_handle::from_direct(payload, GGML_LAYOUT_SOA, false, ggml_sycl::mem_handle::HOST_DEVICE);
    request.cache_key = request.key.id;
    request.layout    = request.key.layout;
    request.size      = sizeof(*payload);
    request.layer_id  = layer;
    request.expert_id = expert;
    return request;
}

void inject(ExpertPrefetcher & prefetcher, prefetch_request request) {
    auto &                      mutex    = prefetcher.*get(mutex_tag{});
    auto &                      inflight = prefetcher.*get(inflight_tag{});
    std::lock_guard<std::mutex> lock(mutex);
    inflight.emplace(request.key, std::move(request));
}

void test_owned_handoff(sycl::queue & queue) {
    std::fprintf(stderr, "[TEST] owned handoff ... ");
    ExpertPrefetcher prefetcher;
    prefetcher.init(queue);
    int payload = 0x1234;
    inject(prefetcher, make_request(101, 4, 7, &payload, sycl::event{}));

    auto result = prefetcher.await_owned(4, 7);
    require(bool(result), "await_owned did not transfer a valid destination lease");
    require(result.ptr == &payload, "await_owned returned the wrong ABI view");
    require(result.owner.resolve().ptr == &payload, "transferred owner did not retain the payload view");
    require(prefetcher.pending_count() == 0, "owned handoff left its request pending");
    require(prefetcher.completed_count() == 1, "owned handoff did not update completion statistics");
    prefetcher.shutdown();
    std::fprintf(stderr, "PASSED\n");
}

void test_model_switch_ambiguity(sycl::queue & queue) {
    std::fprintf(stderr, "[TEST] model-switch identity ... ");
    ExpertPrefetcher prefetcher;
    prefetcher.init(queue);
    int old_payload = 1;
    int new_payload = 2;
    inject(prefetcher, make_request(201, 5, 3, &old_payload, sycl::event{}));
    inject(prefetcher, make_request(202, 5, 3, &new_payload, sycl::event{}));

    auto result = prefetcher.await_owned(5, 3);
    require(!result, "coordinate reuse across model identities did not fail closed");
    require(prefetcher.pending_count() == 2, "ambiguous await consumed one model's request");
    prefetcher.cancel_all();
    require(prefetcher.pending_count() == 0, "model-switch cleanup left requests pending");
    prefetcher.shutdown();
    std::fprintf(stderr, "PASSED\n");
}

sycl::event blocked_event(sycl::queue & queue, const std::shared_future<void> & gate) {
    return queue.submit([&](sycl::handler & cgh) { cgh.host_task([gate] { gate.wait(); }); });
}

void test_dependency_wait(sycl::queue & queue) {
    std::fprintf(stderr, "[TEST] dependency wait ... ");
    ExpertPrefetcher prefetcher;
    prefetcher.init(queue);
    std::promise<void> release;
    auto               gate    = release.get_future().share();
    int                payload = 9;
    inject(prefetcher, make_request(301, 8, 2, &payload, blocked_event(queue, gate)));

    auto waiter = std::async(std::launch::async, [&] { return prefetcher.await_ready(8, 2); });
    require(waiter.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout,
            "await_ready ignored the request dependency");
    release.set_value();
    require(waiter.get(), "await_ready failed after its dependency completed");
    prefetcher.shutdown();
    std::fprintf(stderr, "PASSED\n");
}

void test_cancel_waits(sycl::queue & queue) {
    std::fprintf(stderr, "[TEST] cancellation drain ... ");
    ExpertPrefetcher prefetcher;
    prefetcher.init(queue);
    std::promise<void> release;
    auto               gate    = release.get_future().share();
    int                payload = 11;
    inject(prefetcher, make_request(401, 9, 1, &payload, blocked_event(queue, gate)));

    auto cancel = std::async(std::launch::async, [&] { prefetcher.cancel_all(); });
    require(cancel.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout,
            "cancel_all released an in-flight request before its terminal event");
    release.set_value();
    cancel.get();
    require(prefetcher.pending_count() == 0, "cancel_all left a request pending");
    prefetcher.shutdown();
    std::fprintf(stderr, "PASSED\n");
}

void test_completed_gc(sycl::queue & queue) {
    std::fprintf(stderr, "[TEST] completed-request GC ... ");
    ExpertPrefetcher prefetcher;
    prefetcher.init(queue);
    int         payload = 13;
    sycl::event ready   = queue.submit([](sycl::handler & cgh) { cgh.single_task([] {}); });
    ready.wait_and_throw();
    inject(prefetcher, make_request(501, 10, 6, &payload, ready));

    {
        auto &                      mutex = prefetcher.*get(mutex_tag{});
        std::lock_guard<std::mutex> lock(mutex);
        (prefetcher.*get(gc_tag{}))();
    }
    require(prefetcher.pending_count() == 0, "GC retained a completed speculative request");
    require(prefetcher.completed_count() == 1, "GC did not account for a completed request");
    prefetcher.shutdown();
    std::fprintf(stderr, "PASSED\n");
}

}  // namespace

int main() {
    auto device = sycl_test_prefer_gpu("ExpertPrefetcher runtime ownership tests");
    if (!device) {
        return SYCL_TEST_SKIP;
    }
    sycl::queue queue(*device);

    test_owned_handoff(queue);
    test_model_switch_ambiguity(queue);
    test_dependency_wait(queue);
    test_cancel_waits(queue);
    test_completed_gc(queue);
    std::fprintf(stderr, "ALL EXPERT PREFETCH RUNTIME TESTS PASSED\n");
    return 0;
}
