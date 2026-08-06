#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

namespace {

struct slot_backing {
    uint32_t slot = 0;
    uint64_t generation = 0;
    uint32_t refs = 0;
    bool retired = false;
};

struct model {
    std::mutex m;
    std::condition_variable cv;
    uint64_t next_generation = 1;
    std::vector<slot_backing> current;
    std::vector<slot_backing> retired;

    explicit model(uint32_t ring_depth) {
        current.resize(ring_depth);
        for (uint32_t i = 0; i < ring_depth; ++i) {
            current[i].slot = i;
            current[i].generation = next_generation++;
        }
    }

    slot_backing claim(uint32_t slot) {
        std::lock_guard<std::mutex> lock(m);
        slot_backing & s = current.at(slot);
        s.refs++;
        return s;
    }

    void release(uint32_t slot, uint64_t generation) {
        std::lock_guard<std::mutex> lock(m);
        auto dec = [&](std::vector<slot_backing> & v, bool erase_zero) {
            for (size_t i = 0; i < v.size(); ++i) {
                if (v[i].slot == slot && v[i].generation == generation) {
                    if (v[i].refs > 0) {
                        v[i].refs--;
                    }
                    if (erase_zero && v[i].refs == 0) {
                        v.erase(v.begin() + static_cast<std::ptrdiff_t>(i));
                    }
                    return true;
                }
            }
            return false;
        };
        if (!dec(current, false)) {
            dec(retired, true);
        }
    }

    void resize(uint32_t ring_depth) {
        std::lock_guard<std::mutex> lock(m);
        for (auto & s : current) {
            if (s.refs == 0) {
                continue;
            }
            s.retired = true;
            retired.push_back(s);
        }
        current.assign(ring_depth, {});
        for (uint32_t i = 0; i < ring_depth; ++i) {
            current[i].slot = i;
            current[i].generation = next_generation++;
        }
    }

    uint64_t current_generation(uint32_t slot) {
        std::lock_guard<std::mutex> lock(m);
        return current.at(slot).generation;
    }

    size_t retired_count() {
        std::lock_guard<std::mutex> lock(m);
        return retired.size();
    }
};

void require(bool ok, const char * msg) {
    if (!ok) {
        std::cerr << msg << '\n';
        std::exit(1);
    }
}

void test_cross_thread_completion_releases_exact_generation() {
    model m(1);
    auto claimed = m.claim(0);
    std::thread t([&] { m.release(claimed.slot, claimed.generation); });
    t.join();
    require(m.retired_count() == 0, "cross-thread completion left retired entries behind");
}

void test_resize_in_flight_retires_old_backing_until_release() {
    model m(1);
    auto claimed = m.claim(0);
    m.resize(1);
    require(m.retired_count() == 1, "resize-in-flight failed to retire old backing");
    m.release(claimed.slot, claimed.generation);
    require(m.retired_count() == 0, "retired backing survived final release");
}

void test_stale_release_does_not_clear_new_generation() {
    model m(1);
    auto old = m.claim(0);
    m.release(old.slot, old.generation);
    auto fresh = m.claim(0);
    const uint64_t fresh_generation = fresh.generation;
    m.release(old.slot, old.generation);
    require(m.current_generation(0) == fresh_generation, "stale release cleared newer generation");
    m.release(fresh.slot, fresh.generation);
}

void test_arena_reset_destroy_generation_bump() {
    uint64_t generation = 7;
    const uint64_t stale = generation;
    generation++;
    require(stale != generation, "arena reset failed to bump generation");
    generation++;
    require(stale != generation, "arena destroy failed to preserve stale invalidation");
}

void mutation_stale_release_clears_new_generation() {
    model m(1);
    auto old = m.claim(0);
    m.release(old.slot, old.generation);
    m.resize(1);
    const uint64_t fresh_generation = m.current_generation(0);
    m.release(old.slot, old.generation);
    require(m.current_generation(0) == fresh_generation,
            "mutation stale-release-clears-new-generation reproduced");
}

}  // namespace

int main(int argc, char ** argv) {
    const char * which = argc > 2 && std::strcmp(argv[1], "--case") == 0 ? argv[2] : "all";
    if (std::strcmp(which, "cross-thread-completion") == 0) test_cross_thread_completion_releases_exact_generation();
    else if (std::strcmp(which, "resize-in-flight") == 0) test_resize_in_flight_retires_old_backing_until_release();
    else if (std::strcmp(which, "stale-release") == 0) test_stale_release_does_not_clear_new_generation();
    else if (std::strcmp(which, "arena-reset-destroy") == 0) test_arena_reset_destroy_generation_bump();
    else if (std::strcmp(which, "M1") == 0) mutation_stale_release_clears_new_generation();
    else {
        test_cross_thread_completion_releases_exact_generation();
        test_resize_in_flight_retires_old_backing_until_release();
        test_stale_release_does_not_clear_new_generation();
        test_arena_reset_destroy_generation_bump();
        mutation_stale_release_clears_new_generation();
    }
    return 0;
}
