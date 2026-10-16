// The book of the CPU expert job signal ring (llama.cpp-qwn5): which host-signalled device events may be reset and
// reused, and when.
//
// A reset entry that a device command still waits on parks that command for good (the xe job timeout does not end a
// wait on a host-signalled event), and an entry reused while still signalled lets the next job's scatter read its
// outputs before the job wrote them. So the book frees an entry only when the job has signalled it, the pending
// scatter's half is gone, and the host has waited an event of a flush at least as late as the one that made the device
// wait on it. Each case below breaks one of those conditions.
//
// ggml-sycl/cpu-job-signal.hpp includes only the standard library, so this test links nothing and opens no device.

#include "ggml-sycl/cpu-job-signal.hpp"

#include <cstdint>
#include <cstdio>
#include <set>
#include <vector>

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
    test_signal_all();
    if (g_failures != 0) {
        std::printf("test-sycl-cpu-job-signal-book: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test-sycl-cpu-job-signal-book: PASS\n");
    return 0;
}
