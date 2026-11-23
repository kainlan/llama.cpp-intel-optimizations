// The per-device asynchronous-failure latch (async-failure.hpp) that turns a delivered SYCL async error into a failed
// graph boundary. Host-only: it calls the latch's record and read functions directly; no device is enumerated.
#include "../async-failure.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// The build is Release (-DNDEBUG): an explicit check, never assert().
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

int main() {
    // 1. A cursor taken before any failure sees none.
    const ggml_sycl_async_failure_cursor dev0 = ggml_sycl_async_failure_snapshot(0);
    CHECK(!ggml_sycl_async_failure_since(dev0), "case 1: a fresh cursor reports a failure");

    // 2. A failure on device 0 fails device 0's cursor and not device 1's.
    const ggml_sycl_async_failure_cursor dev1 = ggml_sycl_async_failure_snapshot(1);
    ggml_sycl_async_failure_record(0, "first failure");
    CHECK(ggml_sycl_async_failure_since(dev0), "case 2: device 0's cursor missed its failure");
    CHECK(!ggml_sycl_async_failure_since(dev1), "case 2: a device 0 failure failed device 1");
    const char * first = ggml_sycl_async_failure_first_message(0);
    CHECK(first && std::strcmp(first, "first failure") == 0, "case 2: the first message was not kept");

    // 3. Sticky, and the first message stays the first.
    ggml_sycl_async_failure_record(0, "second failure");
    CHECK(ggml_sycl_async_failure_since(dev0), "case 3: the failure is not sticky");
    CHECK(std::strcmp(ggml_sycl_async_failure_first_message(0), "first failure") == 0,
          "case 3: first message replaced");
    CHECK(ggml_sycl_async_failure_count(0) == 2, "case 3: count is not 2");

    // 4. A cursor taken after the failure (a context created later) starts clean.
    const ggml_sycl_async_failure_cursor later = ggml_sycl_async_failure_snapshot(0);
    CHECK(!ggml_sycl_async_failure_since(later), "case 4: a later cursor inherited an earlier failure");

    // 5. A failure on the shared slot (a queue bound to no device) fails every device's cursor.
    const ggml_sycl_async_failure_cursor dev1b = ggml_sycl_async_failure_snapshot(1);
    ggml_sycl_async_failure_record(GGML_SYCL_ASYNC_FAILURE_SLOT_SHARED, "shared failure");
    CHECK(ggml_sycl_async_failure_since(dev1b) && ggml_sycl_async_failure_since(later),
          "case 5: a shared failure did not fail every device");

    // 6. Out-of-range slots go to the shared slot, never out of bounds.
    const uint64_t shared_before = ggml_sycl_async_failure_count(GGML_SYCL_ASYNC_FAILURE_SLOT_SHARED);
    ggml_sycl_async_failure_record(-1, "negative slot");
    ggml_sycl_async_failure_record(GGML_SYCL_MAX_DEVICES + 5, "slot past the end");
    CHECK(ggml_sycl_async_failure_count(GGML_SYCL_ASYNC_FAILURE_SLOT_SHARED) == shared_before + 2,
          "case 6: out-of-range slots were not counted on the shared slot");

    // 7. The handler runs on any thread that waits: concurrent records are all counted; null and long messages are safe.
    const uint64_t           c7 = ggml_sycl_async_failure_count(7);
    const std::string        long_message(4096, 'x');
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&long_message, t] {
            for (int i = 0; i < 1000; ++i) {
                ggml_sycl_async_failure_record(7, (i + t) % 2 ? nullptr : long_message.c_str());
            }
        });
    }
    for (auto & thread : threads) {
        thread.join();
    }
    CHECK(ggml_sycl_async_failure_count(7) == c7 + 8000, "case 7: concurrent records were lost");
    const char * m7 = ggml_sycl_async_failure_first_message(7);
    CHECK(m7 && std::strlen(m7) < 256, "case 7: the first message is missing or unbounded");

    std::puts("async failure latch: PASS");
    return 0;
}
