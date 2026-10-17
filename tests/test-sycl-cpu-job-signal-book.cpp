// The book of the CPU expert job signal ring (llama.cpp-qwn5): which host-signalled device events may be reset and
// reused, and when.
//
// A reset entry that a device command still waits on parks that command for good (the xe job timeout does not end a
// wait on a host-signalled event), and an entry reused while still signalled lets the next job's scatter read its
// outputs before the job wrote them. So the book frees an entry only when the job has signalled it, the pending
// scatter's half is gone, and the host has waited an event of a flush at least as late as the one that made the device
// wait on it. Each case below breaks one of those conditions. The ring resets the native events with its lock released
// (llama.cpp-42zj), so a recycle runs in two phases, and between them its entries belong to no one else.
//
// The gate list (llama.cpp-qwn5) is the jobs' futures, which a host wait on the device sleeps on first. Its cases
// check that a waiter sleeps rather than spins, that it holds no lock a producer needs while it sleeps, that a job
// that failed or was destroyed unrun still lets it go, and which gates a wait and a clear cover. The copy a
// retained-handle record takes (append_outstanding, llama.cpp-42zj) waits on none and keeps every gate.
//
// The input marker (llama.cpp-42zj) is how a CPU expert job's worker waits for its inputs without the driver. Its
// cases check that an unarmed marker is passed at once, that an armed one holds the waiter until the word reaches its
// serial and lets it go promptly then, that the comparison survives the 32-bit word wrapping, that a word the cache
// abandoned lets the waiter go, that the abandon does not return while a waiter may still load the word (the cache
// releases the word's allocation right after it), and that a wait outlasting the word's slow_after is reported once.
// Here a host thread stands in for the device's store.
//
// ggml-sycl/cpu-job-signal.hpp includes only the standard library, so this test links nothing and opens no device.

#include "ggml-sycl/cpu-job-signal.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

using ggml_sycl::cpu_job_gate_list;
using ggml_sycl::cpu_job_input_marker;
using ggml_sycl::cpu_job_input_marker_word;
using ggml_sycl::cpu_job_signal_book;
using entry_state = cpu_job_signal_book::entry_state;

static int g_failures = 0;

static void expect_true(const char * what, bool ok) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        g_failures++;
    }
}

static void expect_eq(const char * what, unsigned long long got, unsigned long long want) {
    if (got != want) {
        std::printf("FAIL: %s = %llu, want %llu\n", what, got, want);
        g_failures++;
    }
}

static size_t recycle_all_reset(cpu_job_signal_book & b, uint64_t upto, std::vector<uint32_t> * reset = nullptr) {
    return b.recycle(upto, [reset](uint32_t id) {
        if (reset) {
            reset->push_back(id);
        }
        return true;
    });
}

// Entries come only from add(), each acquire takes a distinct one, and an empty book refuses rather than blocks.
static void test_acquire() {
    cpu_job_signal_book b;
    uint32_t            id = 0, gen = 0;
    expect_true("an empty book refuses an acquire", !b.acquire(&id, &gen));
    expect_eq("add returns the first new index", b.add(4), 0);
    expect_eq("a second block starts after the first", b.add(4), 4);
    std::set<uint32_t> seen;
    for (int i = 0; i < 8; ++i) {
        expect_true("an acquire from a book with free entries succeeds", b.acquire(&id, &gen));
        seen.insert(id);
        const cpu_job_signal_book::entry & e = b.at(id);
        expect_true("an acquired entry is ACQUIRED with a live slot half and no device wait",
                    e.state == entry_state::ACQUIRED && e.ref_live && e.serial == 0);
    }
    expect_eq("eight acquires take eight distinct entries", seen.size(), 8);
    expect_true("a book with every entry taken refuses", !b.acquire(&id, &gen));
    const cpu_job_signal_book::counts c = b.count();
    expect_eq("count: total", c.total, 8);
    expect_eq("count: acquired", c.acquired, 8);
    expect_eq("count: free", c.free, 0);
}

// The native signal is sent exactly once per acquire, and never for a stale generation.
static void test_signal_once() {
    cpu_job_signal_book b;
    b.add(1);
    uint32_t id = 0, gen = 0;
    b.acquire(&id, &gen);
    expect_true("a stale generation does not signal", !b.signal(id, gen + 1));
    expect_true("the first signal of an acquire signals", b.signal(id, gen));
    expect_true("a second signal of the same acquire does not", !b.signal(id, gen));
    expect_true("the entry is SIGNALLED", b.at(id).state == entry_state::SIGNALLED);
}

// An unsignalled entry is never freed, whatever else holds: its job has not finished writing, and the device command
// waiting on it has not run.
static void test_unsignalled_is_kept() {
    cpu_job_signal_book b;
    b.add(2);
    uint32_t a = 0, ga = 0, d = 0, gd = 0;
    b.acquire(&a, &ga);
    b.acquire(&d, &gd);
    b.drop(a, ga);
    const uint64_t s = b.begin_flush();
    expect_true("consume of a live slot half succeeds", b.consume(d, gd, s));
    expect_eq("unsignalled entries are kept at any serial", recycle_all_reset(b, UINT64_MAX - 1), 0);
    expect_eq("both still acquired", b.count().acquired, 2);
}

// A signalled entry whose slot half is still live is kept: the pending scatter may still make the device wait on it.
static void test_live_ref_is_kept() {
    cpu_job_signal_book b;
    b.add(1);
    uint32_t id = 0, gen = 0;
    b.acquire(&id, &gen);
    b.signal(id, gen);
    expect_eq("a signalled entry with a live slot half is kept", recycle_all_reset(b, UINT64_MAX - 1), 0);
    b.drop(id, gen);
    std::vector<uint32_t> reset;
    expect_eq("once the slot half is dropped unused it is freed at any recycle", recycle_all_reset(b, 0, &reset), 1);
    expect_true("its native event was reset before it was freed", reset.size() == 1 && reset[0] == id);
    expect_true("it is FREE again", b.at(id).state == entry_state::FREE);
    expect_eq("its generation moved on", b.at(id).generation, gen + 1);
}

// A device wait is released only by a host wait on a flush at least as late as the one that made it.
static void test_serial_bounds_recycle() {
    cpu_job_signal_book b;
    b.add(2);
    uint32_t e1 = 0, g1 = 0, e2 = 0, g2 = 0;
    b.acquire(&e1, &g1);
    b.acquire(&e2, &g2);
    const uint64_t s1 = b.begin_flush();
    const uint64_t s2 = b.begin_flush();
    expect_true("serials grow with each flush", s2 > s1 && s1 > 0);
    b.consume(e1, g1, s1);
    b.consume(e2, g2, s2);
    b.signal(e1, g1);
    b.signal(e2, g2);
    expect_eq("a host wait before the first flush frees nothing", recycle_all_reset(b, s1 - 1), 0);
    std::vector<uint32_t> reset;
    expect_eq("a host wait on the first flush frees only its entry", recycle_all_reset(b, s1, &reset), 1);
    expect_true("the freed one is the first flush's", reset.size() == 1 && reset[0] == e1);
    expect_true("the later flush's entry, which no host wait covered yet, is still in use",
                b.at(e2).state == entry_state::SIGNALLED);
    expect_eq("a host wait on the later flush frees it", recycle_all_reset(b, s2), 1);
}

// The two-card shape: each device's ring has its own book, and a graph-boundary drain waits the events of one
// device's stream only. A recycle on that ring must leave the other ring's entries in use, however late its serial.
static void test_other_ring_is_kept() {
    cpu_job_signal_book waited, not_waited;
    waited.add(1);
    not_waited.add(1);
    uint32_t a = 0, ga = 0, d = 0, gd = 0;
    waited.acquire(&a, &ga);
    not_waited.acquire(&d, &gd);
    const uint64_t sa = waited.begin_flush();
    const uint64_t sd = not_waited.begin_flush();
    waited.consume(a, ga, sa);
    not_waited.consume(d, gd, sd);
    waited.signal(a, ga);
    not_waited.signal(d, gd);
    expect_eq("the waited ring frees its entry", recycle_all_reset(waited, sa), 1);
    expect_true("the ring whose stream was not waited keeps its entry in use",
                not_waited.at(d).state == entry_state::SIGNALLED);
    expect_eq("and frees it only on its own recycle", recycle_all_reset(not_waited, sd), 1);
}

// The slot half makes at most one device wait, only for the current generation, and not after it was dropped.
static void test_consume_rules() {
    cpu_job_signal_book b;
    b.add(1);
    uint32_t id = 0, gen = 0;
    b.acquire(&id, &gen);
    const uint64_t s = b.begin_flush();
    expect_true("consume with serial 0 is refused", !b.consume(id, gen, 0));
    expect_true("consume with a stale generation is refused", !b.consume(id, gen + 1, s));
    expect_true("the first consume succeeds", b.consume(id, gen, s));
    expect_true("a second consume of the same half is refused", !b.consume(id, gen, b.begin_flush()));
    expect_eq("the device wait keeps the first flush's serial", b.at(id).serial, s);

    cpu_job_signal_book d;
    d.add(1);
    d.acquire(&id, &gen);
    d.drop(id, gen);
    expect_true("consume after a drop is refused", !d.consume(id, gen, d.begin_flush()));
}

// A recycled entry is a new acquire: halves of the old generation can neither signal it nor make a device wait.
static void test_stale_generation_after_recycle() {
    cpu_job_signal_book b;
    b.add(1);
    uint32_t id = 0, gen = 0;
    b.acquire(&id, &gen);
    b.signal(id, gen);
    b.drop(id, gen);
    recycle_all_reset(b, 0);
    uint32_t id2 = 0, gen2 = 0;
    expect_true("the freed entry is acquired again", b.acquire(&id2, &gen2) && id2 == id);
    expect_true("with a new generation", gen2 != gen);
    expect_true("the old job half cannot signal the new acquire", !b.signal(id, gen));
    expect_true("the old slot half cannot make a device wait on it", !b.consume(id, gen, b.begin_flush()));
    b.drop(id, gen);
    expect_true("the old slot half's drop does not drop the new one", b.at(id).ref_live);
}

// An entry whose native reset failed is retired, never freed: reused still signalled, the next job's device commands
// would run before that job wrote anything.
static void test_failed_reset_retires() {
    cpu_job_signal_book b;
    b.add(1);
    uint32_t id = 0, gen = 0;
    b.acquire(&id, &gen);
    b.signal(id, gen);
    b.drop(id, gen);
    expect_eq("a failed reset frees nothing", b.recycle(0, [](uint32_t) { return false; }), 0);
    expect_eq("the entry is retired", b.at(id).serial, cpu_job_signal_book::retired);
    expect_eq("a later recycle does not free it", recycle_all_reset(b, UINT64_MAX - 1), 0);
    uint32_t id2 = 0, gen2 = 0;
    expect_true("and it is never acquired again", !b.acquire(&id2, &gen2));
}

// Between a recycle's two phases an entry is RESETTING: no other recycle, acquire or signal_all takes it, and only
// finishing it frees it (its reset done) or retires it (its reset failed). Finishing anything else does nothing.
static void test_recycle_two_phase() {
    cpu_job_signal_book b;
    b.add(2);
    uint32_t a = 0, ga = 0, c = 0, gc = 0;
    b.acquire(&a, &ga);
    b.signal(a, ga);
    b.drop(a, ga);
    b.acquire(&c, &gc);
    b.signal(c, gc);
    b.drop(c, gc);
    std::vector<uint32_t> ids;
    expect_eq("the first phase takes both entries", b.recycle_begin(0, &ids), 2);
    expect_true("and lists them", ids.size() == 2);
    expect_true("both are RESETTING",
                b.at(a).state == entry_state::RESETTING && b.at(c).state == entry_state::RESETTING);
    expect_eq("the book counts them as resetting", b.count().resetting, 2);
    std::vector<uint32_t> again;
    expect_eq("a second recycle takes neither", b.recycle_begin(UINT64_MAX - 1, &again), 0);
    uint32_t x = 0, gx = 0;
    expect_true("no acquire takes a resetting entry", !b.acquire(&x, &gx));
    expect_eq("signal_all leaves them alone", b.signal_all([](uint32_t) {}), 0);
    expect_true("an entry whose reset is done is freed", b.recycle_finish(a, true));
    expect_true("under a new generation", b.at(a).state == entry_state::FREE && b.at(a).generation != ga);
    expect_true("an entry whose reset failed is not freed", !b.recycle_finish(c, false));
    expect_true("it is retired",
                b.at(c).state == entry_state::SIGNALLED && b.at(c).serial == cpu_job_signal_book::retired);
    expect_true("finishing an entry that is not resetting does nothing",
                !b.recycle_finish(a, true) && !b.recycle_finish(c, true) && b.count().free == 1);
    expect_true("the freed entry is acquired again", b.acquire(&x, &gx) && x == a);
    expect_true("the retired one never", !b.acquire(&x, &gx));
}

// signal_all signals each acquired entry once, and leaves signalled and free entries alone.
static void test_signal_all() {
    cpu_job_signal_book b;
    b.add(3);
    uint32_t a = 0, ga = 0, s = 0, gs = 0;
    b.acquire(&a, &ga);
    b.acquire(&s, &gs);
    b.signal(s, gs);
    std::vector<uint32_t> sent;
    expect_eq("signal_all signals the one acquired entry", b.signal_all([&sent](uint32_t id) { sent.push_back(id); }),
              1);
    expect_true("and only it", sent.size() == 1 && sent[0] == a);
    expect_true("it is SIGNALLED now", b.at(a).state == entry_state::SIGNALLED);
    expect_true("its job half's own signal then sends nothing", !b.signal(a, ga));
    expect_eq("a second signal_all sends nothing", b.signal_all([](uint32_t) {}), 0);
}

// --- The gate list ---

using test_clock = std::chrono::steady_clock;

static double ms_since(test_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(test_clock::now() - t0).count();
}

// The calling thread's CPU time: what a waiter that polls instead of sleeping would burn.
static double thread_cpu_ms() {
    timespec ts{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static std::shared_future<void> ready_gate() {
    std::promise<void> p;
    p.set_value();
    return p.get_future().share();
}

// Runs fn on its own thread and fails the test at once if it has not returned within `ms`: a wait that never returns
// would otherwise hang the test until its ctest TIMEOUT.
static void expect_returns_within(const char * what, const std::function<void()> & fn, int ms) {
    auto              done = std::make_shared<std::promise<void>>();
    std::future<void> f    = done->get_future();
    std::thread       t([fn, done] {
        fn();
        done->set_value();
    });
    if (f.wait_for(std::chrono::milliseconds(ms)) != std::future_status::ready) {
        std::printf("FAIL: %s (still waiting after %d ms)\n", what, ms);
        std::fflush(stdout);
        std::_Exit(1);
    }
    t.join();
}

// A waiter sleeps through a 200 ms job: it uses under 20 ms of CPU, and a producer pushing a gate meanwhile is not
// held behind it, so the list's lock is not held across the wait.
static void test_gate_sleeps() {
    cpu_job_gate_list  gates;
    std::promise<void> job;
    std::promise<void> later;
    const uint64_t     s1 = gates.push(job.get_future().share());
    expect_true("a pushed gate gets a serial", s1 != 0);
    std::atomic<bool>            started{ false };
    std::atomic<bool>            returned{ false };
    double                       wall_ms = 0.0;
    double                       cpu_ms  = 0.0;
    size_t                       waited  = 0;
    uint64_t                     newest  = 0;
    const test_clock::time_point t0      = test_clock::now();
    std::thread                  waiter([&] {
        const double c0 = thread_cpu_ms();
        started         = true;
        waited          = gates.wait_all(&newest);
        cpu_ms          = thread_cpu_ms() - c0;
        wall_ms         = ms_since(t0);
        returned        = true;
    });
    while (!started) {
        std::this_thread::yield();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    // From a third thread: if the waiter held the lock, this push would wait for the job, not for this thread.
    double      push_ms = 0.0;
    uint64_t    s2      = 0;
    std::thread pusher([&] {
        const test_clock::time_point p0 = test_clock::now();
        s2                              = gates.push(later.get_future().share());
        push_ms                         = ms_since(p0);
    });
    std::this_thread::sleep_until(t0 + std::chrono::milliseconds(200));
    expect_true("the waiter is still asleep while its job runs", !returned);
    job.set_value();
    later.set_value();
    waiter.join();
    pusher.join();
    expect_true("the waiter returned only after its job ended", wall_ms >= 150.0);
    if (!(cpu_ms < 20.0)) {
        std::printf("FAIL: the waiter used %.1f ms of CPU over a %.0f ms job; it must sleep\n", cpu_ms, wall_ms);
        g_failures++;
    }
    if (!(push_ms < 100.0)) {
        std::printf("FAIL: a push took %.1f ms while a waiter slept; the wait holds the list's lock\n", push_ms);
        g_failures++;
    }
    expect_true("it waited the gate pushed before it", waited >= 1 && newest >= s1);
    expect_true("a later gate has a larger serial", s2 > s1);
}

// A job destroyed unrun breaks its promise, so its gate opens and the waiter returns without throwing. A future that
// holds an exception opens it the same way. CpuExpertPool never stores one: a job that throws ends the process with
// its promise unset (cpu_job_gate_list's hazard), so that half pins the list's contract for any producer that does.
static void test_gate_failed_job() {
    cpu_job_gate_list gates;
    {
        std::promise<void> unrun;
        gates.push(unrun.get_future().share());
    }
    std::promise<void> stored;
    gates.push(stored.get_future().share());
    stored.set_exception(std::make_exception_ptr(std::runtime_error("job failed")));
    bool     thrown = false;
    size_t   waited = 0;
    uint64_t newest = 0;
    expect_returns_within(
        "a waiter returns from a broken promise and from a stored exception",
        [&] {
            try {
                waited = gates.wait_all(&newest);
            } catch (...) {
                thrown = true;
            }
        },
        2000);
    expect_true("the wait does not throw the stored exception", !thrown);
    expect_eq("both gates were waited", waited, 2);
}

// wait_all covers the gates pushed before it and names the newest; wait_upto ignores later ones; clear_upto drops
// only what it covers; an empty future is not kept.
static void test_gate_cover() {
    cpu_job_gate_list gates;
    uint64_t          newest = 7;
    expect_eq("an empty list waits nothing", gates.wait_all(&newest), 0);
    expect_eq("and covers no serial", newest, 0);
    expect_eq("an empty future is not kept", gates.push(std::shared_future<void>{}), 0);
    expect_eq("so the list stays empty", gates.size(), 0);
    const uint64_t     s1 = gates.push(ready_gate());
    std::promise<void> open_later;
    const uint64_t     s2 = gates.push(open_later.get_future().share());
    size_t             w  = 0;
    expect_returns_within("wait_upto does not wait a later gate", [&] { w = gates.wait_upto(s1); }, 2000);
    expect_eq("wait_upto waited the gate it covers", w, 1);
    open_later.set_value();
    const uint64_t s3 = gates.push(ready_gate());
    expect_eq("wait_all waits every gate", gates.wait_all(&newest), 3);
    expect_eq("and names the newest", newest, s3);
    gates.clear_upto(s2);
    expect_eq("clear_upto drops the gates it covers and keeps the later one", gates.size(), 1);
    expect_eq("a wait up to a cleared serial waits nothing", gates.wait_upto(s2), 0);
    gates.clear_upto(s3);
    expect_eq("clear_upto of the newest empties the list", gates.size(), 0);
    gates.push(ready_gate());
    gates.clear();
    expect_eq("clear empties the list", gates.size(), 0);
}

// append_outstanding copies every gate onto what *out already holds, in push order, waits on none and keeps the list as
// it was; an empty list appends nothing and so allocates nothing, and a cleared gate is not copied (llama.cpp-42zj).
static void test_gate_append() {
    cpu_job_gate_list                     gates;
    std::vector<std::shared_future<void>> out;
    gates.append_outstanding(&out);
    expect_eq("an empty list appends nothing", out.size(), 0);
    expect_eq("and allocates nothing", out.capacity(), 0);
    std::promise<void> first;
    std::promise<void> second;
    const uint64_t     s1 = gates.push(first.get_future().share());
    gates.push(second.get_future().share());
    out.push_back(ready_gate());
    expect_returns_within(
        "append_outstanding does not wait a shut gate", [&] { gates.append_outstanding(&out); }, 2000);
    expect_eq("every gate is appended after what *out held", out.size(), 3);
    expect_eq("and the list keeps them", gates.size(), 2);
    first.set_value();
    expect_true("the gates are appended in push order",
                out.size() == 3 && out[1].wait_for(std::chrono::seconds(0)) == std::future_status::ready &&
                    out[2].wait_for(std::chrono::seconds(0)) != std::future_status::ready);
    second.set_value();
    gates.clear_upto(s1);
    out.clear();
    gates.append_outstanding(&out);
    expect_eq("a cleared gate is not appended", out.size(), 1);
}

// Past max_gates a push drops the oldest gate, never the new one. The ring never lets it come to that (a gate per
// entry in use, and max_gates is at least a ring's entries); the drop only bounds a list that stops clearing.
static void test_gate_cap() {
    cpu_job_gate_list gates;
    const size_t      n = cpu_job_gate_list::max_gates + 3;
    for (size_t i = 0; i < n; ++i) {
        gates.push(ready_gate());
    }
    expect_eq("the list holds at most max_gates", gates.size(), cpu_job_gate_list::max_gates);
    expect_eq("the three oldest were dropped", gates.wait_upto(3), 0);
    expect_eq("the fourth is kept", gates.wait_upto(4), 1);
    uint64_t newest = 0;
    expect_eq("every kept gate is waited", gates.wait_all(&newest), cpu_job_gate_list::max_gates);
    expect_eq("and the newest is the last pushed", newest, n);
}

// --- The input marker ---

// A marker over a fresh word holding `start`, armed with `serial`. The word lives in the returned state, as the
// cache's allocation outlives every job in production.
struct marker_fixture {
    uint32_t                                   storage = 0;
    std::shared_ptr<cpu_job_input_marker_word> word    = std::make_shared<cpu_job_input_marker_word>();
    cpu_job_input_marker                       marker;

    marker_fixture(uint32_t start, uint32_t serial) {
        storage       = start;
        word->word    = &storage;
        marker.word   = word;
        marker.serial = serial;
    }

    // What the device's task does: a release store of the serial it was armed with.
    void store(uint32_t v) { __atomic_store_n(&storage, v, __ATOMIC_RELEASE); }
};

// No input events: the prologue arms nothing, and the worker passes at once.
static void test_marker_unarmed() {
    const cpu_job_input_marker m;
    expect_true("an unarmed marker is not armed", !m.armed());
    expect_true("an unarmed marker is seen", ggml_sycl::cpu_job_input_marker_seen(m));
    expect_returns_within(
        "an unarmed marker's wait returns at once", [&] { ggml_sycl::cpu_job_input_marker_wait(m); }, 2000);
}

// Runs a waiter on a marker armed with serial 1, waits `hold_ms` (long past the pause loop), stores the serial and
// returns how many ms after the store the waiter returned. Ends the test at once if it never does.
static double marker_release_latency_ms(int hold_ms) {
    marker_fixture         f(0, 1);
    std::atomic<bool>      returned{ false };
    test_clock::time_point released_at;
    double                 latency_ms = -1.0;
    std::thread            waiter([&] {
        ggml_sycl::cpu_job_input_marker_wait(f.marker);
        latency_ms = ms_since(released_at);
        returned   = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(hold_ms));
    if (returned) {
        std::printf("FAIL: the waiter returned before the word reached its serial\n");
        g_failures++;
    }
    released_at = test_clock::now();
    f.store(1);
    for (int i = 0; i < 2000 && !returned; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!returned) {
        std::printf("FAIL: the waiter did not return within 2 s of the word reaching its serial\n");
        std::fflush(stdout);
        std::_Exit(1);
    }
    waiter.join();
    return latency_ms;
}

// The waiter holds until the word reaches its serial, not before: an earlier job's serial does not release it.
static void test_marker_waits_for_serial() {
    marker_fixture f(0, 3);
    expect_true("a word below the serial is not seen", !ggml_sycl::cpu_job_input_marker_seen(f.marker));
    std::atomic<bool> returned{ false };
    std::thread       waiter([&] {
        ggml_sycl::cpu_job_input_marker_wait(f.marker);
        returned = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    expect_true("the waiter waits while the word is below its serial", !returned);
    f.store(2);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    expect_true("an earlier job's serial does not release the waiter", !returned);
    f.store(3);
    for (int i = 0; i < 2000 && !returned; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!returned) {
        std::printf("FAIL: the waiter did not return within 2 s of the word reaching its serial\n");
        std::fflush(stdout);
        std::_Exit(1);
    }
    waiter.join();
    expect_true("a word at the serial is seen", ggml_sycl::cpu_job_input_marker_seen(f.marker));
    f.store(9);
    expect_true("a word past the serial is seen (a later job's marker was written)",
                ggml_sycl::cpu_job_input_marker_seen(f.marker));
    expect_eq("every waiter left the reader count it joined", f.word->readers.load(), 0);
}

// Once the word reaches its serial the waiter returns promptly, every time: the loop pauses and yields, it does not
// sleep. Eight rounds, each past the pause loop, so a sleep of any length shows in at least one.
static void test_marker_returns_promptly() {
    double worst = 0.0;
    for (int round = 0; round < 8; ++round) {
        const double ms = marker_release_latency_ms(20);
        worst           = ms > worst ? ms : worst;
    }
    if (!(worst < 50.0)) {
        std::printf("FAIL: a waiter returned %.1f ms after the word reached its serial; it must not sleep\n", worst);
        g_failures++;
    }
}

// The word is 32 bits and wraps: the comparison is modulo 2^32, so a serial just past the wrap is not seen from a word
// just before it, and a word just past the wrap has passed a serial just before it.
static void test_marker_wraps() {
    expect_true("a serial is reached by the word that equals it", ggml_sycl::cpu_job_input_marker_reached(7, 7));
    expect_true("a serial is not reached by the word one before it", !ggml_sycl::cpu_job_input_marker_reached(6, 7));
    expect_true("a serial past the wrap is not reached from before the wrap",
                !ggml_sycl::cpu_job_input_marker_reached(0xfffffffeu, 2));
    expect_true("a serial before the wrap is reached from past the wrap",
                ggml_sycl::cpu_job_input_marker_reached(1, 0xffffffffu));
    expect_true("serial 0, taken at the wrap, is not reached from the serial before it",
                !ggml_sycl::cpu_job_input_marker_reached(0xffffffffu, 0));
    marker_fixture f(0xfffffffeu, 1);
    expect_true("an armed marker across the wrap is not seen early", !ggml_sycl::cpu_job_input_marker_seen(f.marker));
    f.store(0xffffffffu);
    expect_true("nor one serial short of it", !ggml_sycl::cpu_job_input_marker_seen(f.marker));
    f.store(0);
    expect_true("nor at the wrap itself", !ggml_sycl::cpu_job_input_marker_seen(f.marker));
    f.store(1);
    expect_true("it is seen once the wrapped word reaches it", ggml_sycl::cpu_job_input_marker_seen(f.marker));
}

// The cache abandoned the word (teardown, or the SYCL runtime gone): no device write will come, and the waiter stops.
static void test_marker_abandoned() {
    marker_fixture    f(0, 1);
    std::atomic<bool> returned{ false };
    std::thread       waiter([&] {
        ggml_sycl::cpu_job_input_marker_wait(f.marker);
        returned = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    expect_true("the waiter waits on a live word below its serial", !returned);
    expect_returns_within(
        "the abandon returns once no waiter is between its test of abandoned and its load of the word",
        [&] { ggml_sycl::cpu_job_input_marker_abandon(*f.word); }, 2000);
    for (int i = 0; i < 2000 && !returned; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!returned) {
        std::printf("FAIL: the waiter did not return within 2 s of its word being abandoned\n");
        std::fflush(stdout);
        std::_Exit(1);
    }
    waiter.join();
    expect_true("an abandoned marker is seen whatever its word holds", ggml_sycl::cpu_job_input_marker_seen(f.marker));
}

// The cache releases the word's allocation as soon as cpu_job_input_marker_abandon returns, and no job holds it, so the
// abandon must set abandoned first and then hold while any waiter is between its test of that flag and its load of the
// word. The reader counted here stands for such a waiter, caught between the two.
static void test_marker_abandon_waits_for_readers() {
    marker_fixture f(0, 1);
    f.word->readers.fetch_add(1);
    std::atomic<bool> returned{ false };
    std::thread       abandoner([&] {
        ggml_sycl::cpu_job_input_marker_abandon(*f.word);
        returned = true;
    });
    for (int i = 0; i < 2000 && !f.word->abandoned.load(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!f.word->abandoned.load()) {
        // Without the flag, the null word below would be loaded.
        std::printf("FAIL: the abandon does not set abandoned before it waits for the readers\n");
        std::fflush(stdout);
        std::_Exit(1);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    expect_true("the abandon does not return while a waiter may still load the word", !returned);
    // A waiter that arrives now sees the flag and does not load the word: a null word would fault.
    f.word->word = nullptr;
    expect_true("a waiter arriving during the abandon is released without loading the word",
                ggml_sycl::cpu_job_input_marker_seen(f.marker));
    expect_eq("that waiter leaves the reader count as it found it", f.word->readers.load(), 1);
    f.word->readers.fetch_sub(1);
    for (int i = 0; i < 2000 && !returned; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!returned) {
        std::printf("FAIL: the abandon did not return within 2 s of the last reader leaving\n");
        std::fflush(stdout);
        std::_Exit(1);
    }
    abandoner.join();
}

// What the slow-wait report was told.
static std::atomic<int>      g_slow_calls{ 0 };
static std::atomic<uint32_t> g_slow_serial{ 0 };
static std::atomic<int>      g_slow_device{ 0 };
static std::atomic<double>   g_slow_waited_s{ 0.0 };

static void record_slow(const cpu_job_input_marker_word & w, uint32_t serial, double waited_s) {
    g_slow_serial   = serial;
    g_slow_device   = w.device;
    g_slow_waited_s = waited_s;
    g_slow_calls.fetch_add(1);
}

// A wait that outlasts the word's slow_after is reported once, with its serial, the word's device and how long it has
// waited, and goes on waiting; a wait that ends sooner is not reported. Without the report, a marker that never
// arrives with the SYCL watchdog off is a silent spin.
static void test_marker_slow_wait_reported_once() {
    g_slow_calls = 0;
    marker_fixture f(0, 5);
    f.word->device     = 3;
    f.word->slow       = record_slow;
    f.word->slow_after = std::chrono::milliseconds(30);
    std::atomic<bool> returned{ false };
    std::thread       waiter([&] {
        ggml_sycl::cpu_job_input_marker_wait(f.marker);
        returned = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    expect_true("a slow wait goes on waiting after its report", !returned);
    expect_eq("a wait ten times past its slow_after is reported once", g_slow_calls.load(), 1);
    expect_eq("the report names the marker's serial", g_slow_serial.load(), 5);
    expect_eq("the report names the word's device", static_cast<unsigned long long>(g_slow_device.load()), 3);
    expect_true("the report says how long the wait has lasted, at least slow_after",
                g_slow_waited_s.load() >= 0.03 && g_slow_waited_s.load() < 10.0);
    f.store(5);
    for (int i = 0; i < 2000 && !returned; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!returned) {
        std::printf("FAIL: a reported wait did not return within 2 s of the word reaching its serial\n");
        std::fflush(stdout);
        std::_Exit(1);
    }
    waiter.join();
    expect_eq("a reported wait is reported no more", g_slow_calls.load(), 1);

    g_slow_calls = 0;
    marker_fixture g(0, 1);
    g.word->slow       = record_slow;
    g.word->slow_after = std::chrono::milliseconds(2000);
    std::thread prompt([&] { ggml_sycl::cpu_job_input_marker_wait(g.marker); });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    g.store(1);
    prompt.join();
    expect_eq("a wait shorter than its slow_after is not reported", g_slow_calls.load(), 0);
}

int main() {
    test_acquire();
    test_signal_once();
    test_unsignalled_is_kept();
    test_live_ref_is_kept();
    test_serial_bounds_recycle();
    test_other_ring_is_kept();
    test_consume_rules();
    test_stale_generation_after_recycle();
    test_failed_reset_retires();
    test_recycle_two_phase();
    test_signal_all();
    test_gate_sleeps();
    test_gate_failed_job();
    test_gate_cover();
    test_gate_append();
    test_gate_cap();
    test_marker_unarmed();
    test_marker_waits_for_serial();
    test_marker_returns_promptly();
    test_marker_wraps();
    test_marker_abandoned();
    test_marker_abandon_waits_for_readers();
    test_marker_slow_wait_reported_once();
    if (g_failures != 0) {
        std::printf("test-sycl-cpu-job-signal-book: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test-sycl-cpu-job-signal-book: PASS\n");
    return 0;
}
