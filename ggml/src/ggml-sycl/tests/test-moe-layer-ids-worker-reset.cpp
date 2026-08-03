#include "moe-layer-ids-cache.hpp"

#include <condition_variable>
#include <exception>
#include <functional>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

static thread_local moe_layer_ids_cache g_moe_layer_ids_cache;

static void check(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

class persistent_worker {
  public:
    persistent_worker() {
        // Constructor body runs only after every member is initialized.
        thread_ = std::thread([this] { run(); });
    }

    ~persistent_worker() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            ready_    = true;
        }
        cv_.notify_one();
        thread_.join();
    }

    void dispatch(std::function<void()> work) {
        std::unique_lock<std::mutex> lock(mutex_);
        done_  = false;
        error_ = nullptr;
        work_  = std::move(work);
        ready_ = true;
        cv_.notify_one();
        cv_.wait(lock, [this] { return done_; });
        if (error_) {
            std::rethrow_exception(error_);
        }
    }

  private:
    void run() {
        for (;;) {
            std::function<void()> work;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return ready_; });
                if (stopping_) {
                    return;
                }
                work   = std::move(work_);
                ready_ = false;
            }
            try {
                work();
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex_);
                error_ = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                done_ = true;
            }
            cv_.notify_one();
        }
    }

    // thread_ is deliberately last: C++ initializes members in declaration
    // order, so run() cannot observe any synchronization/state member before
    // that member's construction has completed.
    std::mutex              mutex_;
    std::condition_variable cv_;
    std::function<void()>   work_;
    std::exception_ptr      error_;
    bool                    ready_    = false;
    bool                    done_     = false;
    bool                    stopping_ = false;
    std::thread             thread_;
};

struct worker_snapshot {
    std::thread::id thread_id;
    size_t          bucket_count  = 0;
    const void *    entry_address = nullptr;
    size_t          ids_capacity  = 0;
    size_t          cpu_capacity  = 0;
    size_t          sec_capacity  = 0;
    size_t          gpu_capacity  = 0;
};

static void seed_worker(worker_snapshot & snapshot, int worker_index) {
    check(g_moe_layer_ids_cache.empty(), "new persistent worker did not start with empty TLS state");
    g_moe_layer_ids_cache.reserve(8);
    auto & entry = g_moe_layer_ids_cache[17];
    entry.ids_host.reserve(32);
    entry.partition.cpu_indices.reserve(16);
    entry.partition.sec_indices.reserve(16);
    entry.partition.gpu0_cached_indices.reserve(16);
    entry.ids_host              = { 100 + worker_index, 110 + worker_index };
    entry.partition.cpu_indices = { static_cast<size_t>(worker_index + 1) };
    entry.partition.sec_indices = {
        { static_cast<size_t>(worker_index + 2), worker_index }
    };
    entry.partition.gpu0_cached_indices = { static_cast<size_t>(worker_index + 3) };
    entry.partition.valid               = true;

    snapshot.thread_id     = std::this_thread::get_id();
    snapshot.bucket_count  = g_moe_layer_ids_cache.bucket_count();
    snapshot.entry_address = &entry;
    snapshot.ids_capacity  = entry.ids_host.capacity();
    snapshot.cpu_capacity  = entry.partition.cpu_indices.capacity();
    snapshot.sec_capacity  = entry.partition.sec_indices.capacity();
    snapshot.gpu_capacity  = entry.partition.gpu0_cached_indices.capacity();
}

static void cross_graph_reset_then_refill(const worker_snapshot & snapshot, int worker_index) {
    check(std::this_thread::get_id() == snapshot.thread_id, "task migrated away from its persistent worker");

    // This is the exact synchronous helper called at all production graph boundaries.
    ggml_sycl_moe_layer_ids_cache_new_graph(g_moe_layer_ids_cache);

    check(g_moe_layer_ids_cache.size() == 1, "graph reset erased the retained map node");
    check(g_moe_layer_ids_cache.bucket_count() == snapshot.bucket_count, "graph reset replaced map allocation");
    auto it = g_moe_layer_ids_cache.find(17);
    check(it != g_moe_layer_ids_cache.end(), "graph reset erased layer key");
    auto & entry = it->second;
    check(&entry == snapshot.entry_address, "graph reset replaced retained map entry");
    check(entry.ids_host.empty(), "stale IDs were readable after graph reset");
    check(entry.partition.cpu_indices.empty(), "stale CPU partition was readable after graph reset");
    check(entry.partition.sec_indices.empty(), "stale secondary partition was readable after graph reset");
    check(entry.partition.gpu0_cached_indices.empty(), "stale GPU partition was readable after graph reset");
    check(!entry.partition.valid, "stale partition remained valid after graph reset");
    check(entry.ids_host.capacity() == snapshot.ids_capacity, "IDs vector capacity was released");
    check(entry.partition.cpu_indices.capacity() == snapshot.cpu_capacity, "CPU vector capacity was released");
    check(entry.partition.sec_indices.capacity() == snapshot.sec_capacity, "secondary vector capacity was released");
    check(entry.partition.gpu0_cached_indices.capacity() == snapshot.gpu_capacity, "GPU vector capacity was released");

    // Refill and consume within the same graph, proving reset did not disable bounded reuse.
    entry.ids_host              = { 200 + worker_index, 210 + worker_index };
    entry.partition.cpu_indices = { static_cast<size_t>(20 + worker_index) };
    entry.partition.sec_indices = {
        { static_cast<size_t>(30 + worker_index), worker_index + 4 }
    };
    entry.partition.gpu0_cached_indices = { static_cast<size_t>(40 + worker_index) };
    entry.partition.valid               = true;
    check(entry.ids_host[0] == 200 + worker_index && entry.partition.valid, "same-graph refill was not readable");
}

static void probe_same_worker(int worker_index) {
    auto it = g_moe_layer_ids_cache.find(17);
    check(it != g_moe_layer_ids_cache.end(), "worker lost its retained TLS layer");
    check(it->second.ids_host.size() == 2 && it->second.ids_host[1] == 210 + worker_index,
          "worker observed another worker's graph-local IDs");
    check(it->second.partition.valid && it->second.partition.cpu_indices[0] == static_cast<size_t>(20 + worker_index),
          "worker observed another worker's graph-local partition");
}

int main() {
    try {
        persistent_worker workers[2];
        worker_snapshot   snapshots[2];

        // Main-thread handoff is intentionally alternating and synchronous.
        workers[0].dispatch([&] { seed_worker(snapshots[0], 0); });
        workers[1].dispatch([&] { seed_worker(snapshots[1], 1); });
        check(snapshots[0].thread_id != snapshots[1].thread_id, "fixture did not create two persistent workers");
        constexpr int stress_iterations = 1000;
        for (int iteration = 0; iteration < stress_iterations; ++iteration) {
            workers[0].dispatch([&] { cross_graph_reset_then_refill(snapshots[0], 0); });
            workers[1].dispatch([&] { cross_graph_reset_then_refill(snapshots[1], 1); });
            workers[0].dispatch([&] { probe_same_worker(0); });
            workers[1].dispatch([&] { probe_same_worker(1); });
        }

        std::cout << "two-worker TLS graph-boundary reset passed " << stress_iterations << " iterations\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
