// Runtime ownership and lifecycle tests for ExpertPrefetcher.
//
// These tests inject already-submitted requests so they exercise the production
// handoff/cancellation/GC code without manufacturing model-registry state. They
// deliberately do not claim coverage of hint(), registry lookup, or real staging.

#include "../expert-prefetch.hpp"
#include "../common.hpp"
#include "sycl-test-skip.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>

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

struct request_fixture {
    prefetch_request request;
    void *           source_ptr      = nullptr;
    void *           destination_ptr = nullptr;
};

request_fixture make_request(sycl::queue & queue,
                             uint64_t      model_id,
                             int           layer,
                             int           expert,
                             const sycl::event & ready) {
    const int device = ggml_sycl_get_device_id_from_queue(queue);
    auto allocate = [&] {
        ggml_sycl::alloc_request allocation_request{};
        allocation_request.queue                          = &queue;
        allocation_request.device                         = device;
        allocation_request.size                           = sizeof(int);
        allocation_request.intent.role                    = ggml_sycl::alloc_role::COMPUTE;
        allocation_request.intent.category                = ggml_sycl::runtime_category::COMPUTE;
        allocation_request.intent.constraints.must_device = true;
        ggml_sycl::alloc_handle allocation{};
        require(ggml_sycl::unified_alloc(allocation_request, &allocation), "fixture allocation failed");
        void * ptr = allocation.ptr;
        return std::pair{ ptr,
                          ggml_sycl::mem_handle::from_owned_alloc(std::move(allocation), GGML_LAYOUT_SOA) };
    };

    auto [source_ptr, source_owner]           = allocate();
    auto [destination_ptr, destination_owner] = allocate();
    request_fixture fixture{};
    fixture.source_ptr                         = source_ptr;
    fixture.destination_ptr                    = destination_ptr;
    fixture.request.key.id                     = make_id(model_id, static_cast<uint64_t>(expert + 1));
    fixture.request.key.layout                 = GGML_LAYOUT_SOA;
    fixture.request.event                      = ready;
    fixture.request.device_ptr                 = destination_ptr;
    fixture.request.source_owner               = std::move(source_owner);
    fixture.request.destination_owner          = std::move(destination_owner);
    fixture.request.cache_key                  = fixture.request.key.id;
    fixture.request.layout                     = fixture.request.key.layout;
    fixture.request.size                       = sizeof(int);
    fixture.request.layer_id                   = layer;
    fixture.request.expert_id                  = expert;
    return fixture;
}

bool allocation_live(void * ptr) {
    ggml_sycl::alloc_handle allocation{};
    return ggml_sycl::unified_lookup(ptr, &allocation);
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
    auto fixture = make_request(queue, 101, 4, 7, sycl::event{});
    inject(prefetcher, std::move(fixture.request));

    auto result = prefetcher.await_owned(4, 7);
    require(bool(result), "await_owned did not transfer a valid destination lease");
    require(result.ptr == fixture.destination_ptr, "await_owned returned the wrong ABI view");
    require(result.owner.resolve().ptr == fixture.destination_ptr, "transferred owner did not retain the payload view");
    require(!allocation_live(fixture.source_ptr), "owned handoff retained the source after its terminal event");
    require(allocation_live(fixture.destination_ptr), "owned result did not retain its destination allocation");
    require(prefetcher.pending_count() == 0, "owned handoff left its request pending");
    require(prefetcher.completed_count() == 1, "owned handoff did not update completion statistics");
    result.owner = {};
    require(!allocation_live(fixture.destination_ptr), "destination survived release of the handed-off owner");
    prefetcher.shutdown();
    std::fprintf(stderr, "PASSED\n");
}

void test_model_switch_ambiguity(sycl::queue & queue) {
    std::fprintf(stderr, "[TEST] model-switch identity ... ");
    ExpertPrefetcher prefetcher;
    prefetcher.init(queue);
    auto old_fixture = make_request(queue, 201, 5, 3, sycl::event{});
    auto new_fixture = make_request(queue, 202, 5, 3, sycl::event{});
    inject(prefetcher, std::move(old_fixture.request));
    inject(prefetcher, std::move(new_fixture.request));

    auto result = prefetcher.await_owned(5, 3);
    require(!result, "coordinate reuse across model identities did not fail closed");
    require(prefetcher.pending_count() == 2, "ambiguous await consumed one model's request");
    prefetcher.cancel_all();
    require(prefetcher.pending_count() == 0, "model-switch cleanup left requests pending");
    require(!allocation_live(old_fixture.source_ptr) && !allocation_live(old_fixture.destination_ptr),
            "model-switch cleanup retained the old-model owners");
    require(!allocation_live(new_fixture.source_ptr) && !allocation_live(new_fixture.destination_ptr),
            "model-switch cleanup retained the new-model owners");
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
    auto               fixture = make_request(queue, 301, 8, 2, sycl::event{});
    fixture.request.event       = blocked_event(queue, gate);
    inject(prefetcher, std::move(fixture.request));

    auto waiter = std::async(std::launch::async, [&] { return prefetcher.await_ready(8, 2); });
    require(waiter.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout,
            "await_ready ignored the request dependency");
    require(allocation_live(fixture.source_ptr) && allocation_live(fixture.destination_ptr),
            "await_ready dropped request owners before its dependency completed");
    release.set_value();
    require(waiter.get(), "await_ready failed after its dependency completed");
    require(!allocation_live(fixture.source_ptr) && !allocation_live(fixture.destination_ptr),
            "boolean await retained request owners after handoff completion");
    prefetcher.shutdown();
    std::fprintf(stderr, "PASSED\n");
}

void test_cancel_waits(sycl::queue & queue) {
    std::fprintf(stderr, "[TEST] cancellation drain ... ");
    ExpertPrefetcher prefetcher;
    prefetcher.init(queue);
    std::promise<void> release;
    auto               gate    = release.get_future().share();
    auto               fixture = make_request(queue, 401, 9, 1, sycl::event{});
    fixture.request.event       = blocked_event(queue, gate);
    inject(prefetcher, std::move(fixture.request));

    auto cancel = std::async(std::launch::async, [&] { prefetcher.cancel_all(); });
    require(cancel.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout,
            "cancel_all released an in-flight request before its terminal event");
    require(allocation_live(fixture.source_ptr) && allocation_live(fixture.destination_ptr),
            "cancel_all dropped request owners before its terminal event");
    release.set_value();
    cancel.get();
    require(prefetcher.pending_count() == 0, "cancel_all left a request pending");
    require(!allocation_live(fixture.source_ptr) && !allocation_live(fixture.destination_ptr),
            "cancel_all retained request owners after its terminal event");
    prefetcher.shutdown();
    std::fprintf(stderr, "PASSED\n");
}

void test_completed_gc(sycl::queue & queue) {
    std::fprintf(stderr, "[TEST] completed-request GC ... ");
    ExpertPrefetcher prefetcher;
    prefetcher.init(queue);
    sycl::event ready = queue.submit([](sycl::handler & cgh) { cgh.single_task([] {}); });
    ready.wait_and_throw();
    auto fixture = make_request(queue, 501, 10, 6, ready);
    inject(prefetcher, std::move(fixture.request));

    {
        auto &                      mutex = prefetcher.*get(mutex_tag{});
        std::lock_guard<std::mutex> lock(mutex);
        (prefetcher.*get(gc_tag{}))();
    }
    require(prefetcher.pending_count() == 0, "GC retained a completed speculative request");
    require(prefetcher.completed_count() == 1, "GC did not account for a completed request");
    require(!allocation_live(fixture.source_ptr) && !allocation_live(fixture.destination_ptr),
            "GC retained completed request owners");
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
