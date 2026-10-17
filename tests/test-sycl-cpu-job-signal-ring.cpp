// Device test of the CPU expert job device-signal ring (llama.cpp-qwn5).
//
// A device command that depends on a ring entry's wrapper must run only after the entry is signalled: by the job
// half's signal(), by its destructor, or by cpu_job_signal_signal_all(). A recycled entry must be reset, so the next
// job's command parks again until that job signals. An entry stays in use until its own ring's stream has been
// waited. A CpuExpertPool job whose prologue throws must release the device before the process ends; a control arm
// throws past a job half with no handler and must find the device still parked, so the probe can see the failure.
//
// Lead-run (labelled cache): it opens a Level Zero GPU, and exits 77 when there is none. A deadline thread bounds the
// whole run: past it, every ring is signalled and the test fails, so a regression parks nothing past the deadline.
// Nothing here queries the status of a parked command (on a profiling queue the driver busy-waits such a query);
// whether a command ran is read from the host memory it writes.
//
// A retained-handle record published while a ring holds a job's gate carries that gate, with no scope and whether or
// not its own event stands behind the job, and the drain worker holds the record's handles until the gate opens
// (llama.cpp-42zj).

#include "cpu-expert-pool.hpp"
#include "mem-handle.hpp"
#include "unified-cache.hpp"

#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <future>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <thread>
#include <vector>

using ggml_sycl::cpu_job_signal;
using ggml_sycl::cpu_job_signal_mark;
using ggml_sycl::cpu_job_signal_ref;
using ggml_sycl::cpu_job_signal_ring;

static int g_failures = 0;

static void expect(const char * what, bool ok) {
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) {
        g_failures++;
    }
}

static std::atomic<bool> g_done{ false };

// Signals every ring and fails the run if it has not finished by then.
static void start_deadline(int seconds) {
    std::thread([seconds] {
        for (int i = 0; i < seconds * 10 && !g_done.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (!g_done.load()) {
            std::printf("FAIL: deadline of %d s passed; signalling every ring\n", seconds);
            std::fflush(stdout);
            ggml_sycl::cpu_job_signal_signal_all();
            std::this_thread::sleep_for(std::chrono::seconds(2));
            std::_Exit(1);
        }
    }).detach();
}

static sycl::property_list production_queue_properties() {
    return sycl::property_list{ sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{} };
}

static int read_flag(const volatile int * flag) {
    const int v = *flag;
    std::atomic_thread_fence(std::memory_order_acquire);
    return v;
}

// True once the device has written `want`, polling host memory for at most `ms`.
static bool wait_flag(const volatile int * flag, int want, int ms) {
    for (int i = 0; i < ms; ++i) {
        if (read_flag(flag) == want) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return read_flag(flag) == want;
}

// One device command gated on `gate`: writes `value` to host memory.
static sycl::event submit_gated_write(sycl::queue & q, const sycl::event & gate, int * flag, int value) {
    return q.submit([&](sycl::handler & h) {
        h.depends_on(gate);
        h.single_task([=]() { *flag = value; });
    });
}

static bool parked(const volatile int * flag, int not_yet) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    return read_flag(flag) != not_yet;
}

static void case_signal_and_recycle(sycl::queue & q, int * flag) {
    std::shared_ptr<cpu_job_signal_ring> ring = ggml_sycl::cpu_job_signal_ring_create_for_testing(q);
    expect("a ring is made on a Level Zero queue", ring != nullptr);
    if (!ring) {
        return;
    }
    uint32_t first_id = 0;
    for (int round = 1; round <= 2; ++round) {
        cpu_job_signal     job;
        cpu_job_signal_ref ref;
        expect("acquire gives both halves", ggml_sycl::cpu_job_signal_acquire_for_testing(ring, &job, &ref) &&
                                                static_cast<bool>(job) && static_cast<bool>(ref));
        if (round == 1) {
            first_id = ref.id();
        } else {
            expect("the second acquire reuses the recycled entry", ref.id() == first_id);
        }
        cpu_job_signal_mark mark;
        sycl::event         gate;
        expect("wrap succeeds and consumes the slot half",
               ggml_sycl::cpu_job_signal_wrap(ref, mark, &gate) && !ref && static_cast<bool>(mark));
        *flag                = 0;
        const sycl::event kw = submit_gated_write(q, gate, flag, round);
        // Round 2 reuses the recycled entry: it parks only if the recycle reset the native event.
        expect(round == 1 ? "the gated command waits for the signal" :
                            "after a recycle the reused entry parks the next command again",
               parked(flag, round));
        job.signal();
        expect("the signal releases the gated command", wait_flag(flag, round, 5000));
        sycl::event(kw).wait();
        ggml_sycl::cpu_job_signal_recycle(mark);
        expect("recycle clears the mark", !mark);
        const auto c = ggml_sycl::cpu_job_signal_ring_counts_for_testing(ring);
        expect("every entry is free again after the waited flush is recycled", c.free == c.total && c.total > 0);
    }
    ggml_sycl::cpu_job_signal_ring_shutdown_for_testing(ring);
}

static void case_abandoned_job_half(sycl::queue & q, int * flag) {
    std::shared_ptr<cpu_job_signal_ring> ring = ggml_sycl::cpu_job_signal_ring_create_for_testing(q);
    if (!ring) {
        expect("a ring is made for the abandoned-job case", false);
        return;
    }
    cpu_job_signal_mark mark;
    sycl::event         kw;
    *flag = 0;
    {
        cpu_job_signal     job;
        cpu_job_signal_ref ref;
        expect("acquire for the abandoned job", ggml_sycl::cpu_job_signal_acquire_for_testing(ring, &job, &ref));
        sycl::event gate;
        expect("wrap for the abandoned job", ggml_sycl::cpu_job_signal_wrap(ref, mark, &gate));
        kw = submit_gated_write(q, gate, flag, 7);
        expect("the abandoned job's command waits while its job half lives", parked(flag, 7));
    }
    expect("destroying an unsignalled job half releases the command", wait_flag(flag, 7, 5000));
    kw.wait();
    ggml_sycl::cpu_job_signal_recycle(mark);
    ggml_sycl::cpu_job_signal_ring_shutdown_for_testing(ring);
}

static void case_signal_all(sycl::queue & q, int * flag) {
    std::shared_ptr<cpu_job_signal_ring> ring = ggml_sycl::cpu_job_signal_ring_create_for_testing(q);
    if (!ring) {
        expect("a ring is made for the signal-all case", false);
        return;
    }
    cpu_job_signal     job;
    cpu_job_signal_ref ref;
    expect("acquire for signal-all", ggml_sycl::cpu_job_signal_acquire_for_testing(ring, &job, &ref));
    cpu_job_signal_mark mark;
    sycl::event         gate;
    expect("wrap for signal-all", ggml_sycl::cpu_job_signal_wrap(ref, mark, &gate));
    *flag                = 0;
    const sycl::event kw = submit_gated_write(q, gate, flag, 9);
    expect("the command waits before signal-all", parked(flag, 9));
    ggml_sycl::cpu_job_signal_signal_all();
    expect("signal-all releases a command parked on a live job", wait_flag(flag, 9, 5000));
    sycl::event(kw).wait();
    job.signal();  // already signalled: a no-op
    ggml_sycl::cpu_job_signal_recycle(mark);
    const auto c = ggml_sycl::cpu_job_signal_ring_counts_for_testing(ring);
    expect("the entry signal-all released is recycled", c.free == c.total);
    ggml_sycl::cpu_job_signal_ring_shutdown_for_testing(ring);
}

// Two streams, as on two cards: recycling the first ring's waited flush must not free the second ring's entry, whose
// consuming command has not been waited yet.
static void case_unwaited_stream_keeps_its_entries(sycl::queue & q1, sycl::queue & q2, int * flag1, int * flag2) {
    std::shared_ptr<cpu_job_signal_ring> r1 = ggml_sycl::cpu_job_signal_ring_create_for_testing(q1);
    std::shared_ptr<cpu_job_signal_ring> r2 = ggml_sycl::cpu_job_signal_ring_create_for_testing(q2);
    if (!r1 || !r2) {
        expect("two rings are made for the two-stream case", false);
        return;
    }
    cpu_job_signal      j1, j2;
    cpu_job_signal_ref  f1, f2;
    cpu_job_signal_mark m1, m2;
    sycl::event         g1, g2;
    expect("acquire on both rings", ggml_sycl::cpu_job_signal_acquire_for_testing(r1, &j1, &f1) &&
                                        ggml_sycl::cpu_job_signal_acquire_for_testing(r2, &j2, &f2));
    expect("wrap on both rings",
           ggml_sycl::cpu_job_signal_wrap(f1, m1, &g1) && ggml_sycl::cpu_job_signal_wrap(f2, m2, &g2));
    *flag1               = 0;
    *flag2               = 0;
    const sycl::event k1 = submit_gated_write(q1, g1, flag1, 1);
    const sycl::event k2 = submit_gated_write(q2, g2, flag2, 2);
    j1.signal();
    expect("the first stream's command runs", wait_flag(flag1, 1, 5000));
    sycl::event(k1).wait();
    ggml_sycl::cpu_job_signal_recycle(m1);
    const auto c2 = ggml_sycl::cpu_job_signal_ring_counts_for_testing(r2);
    expect("the second ring keeps its entry in use while its stream is unwaited", c2.free + 1 == c2.total);
    expect("the second stream's command is still parked", read_flag(flag2) == 0);
    j2.signal();
    expect("the second stream's command runs once its job signals", wait_flag(flag2, 2, 5000));
    sycl::event(k2).wait();
    ggml_sycl::cpu_job_signal_recycle(m2);
    const auto c2b = ggml_sycl::cpu_job_signal_ring_counts_for_testing(r2);
    expect("the second ring's entry is free after its own flush is waited", c2b.free == c2b.total);
    ggml_sycl::cpu_job_signal_ring_shutdown_for_testing(r1);
    ggml_sycl::cpu_job_signal_ring_shutdown_for_testing(r2);
}

// --- retained-handle records take the rings' gates (llama.cpp-42zj) ----------------------------------------------

static std::atomic<int> g_records_released{ 0 };

static ggml_sycl::release_attempt count_record_release(const ggml_sycl::alloc_metadata &, void *) noexcept {
    g_records_released.fetch_add(1);
    return { ggml_sycl::release_attempt_status::RELEASED };
}

// An owner handle over a fresh owner control with an injected release in place of a physical free, so "released"
// below is the owner's last reference going.
static ggml_sycl::mem_handle make_record_owner(size_t size) {
    static std::vector<std::vector<unsigned char>> backing;
    static uint64_t                                next_id = 4200;
    backing.emplace_back(size);
    const uint64_t            id = next_id++;
    ggml_sycl::alloc_metadata metadata{};
    metadata.ptr      = backing.back().data();
    metadata.size     = size;
    metadata.device   = 0;
    metadata.id       = id;
    metadata.alloc_id = id;
    metadata.epoch_id = 1;
    metadata.tier     = ggml_sycl::alloc_tier::HOST_PINNED;
    auto fixture      = ggml_sycl::allocation_owner_test_create(metadata, count_record_release, nullptr);
    if (!fixture.result) {
        return ggml_sycl::mem_handle{};
    }
    return ggml_sycl::mem_handle::from_owned_alloc(std::move(fixture.result.owner));
}

// The record's event is complete, as a device command's is once it ran, and stands behind no job: a worker that waited
// the event alone would release the handle at once. This one must hold it until the ring's gate opens.
static void case_record_carries_ring_gates(sycl::queue & q) {
    std::shared_ptr<cpu_job_signal_ring> ring = ggml_sycl::cpu_job_signal_ring_create_for_testing(q);
    if (!ring) {
        expect("a ring is made for the record-gates case", false);
        return;
    }
    cpu_job_signal     job;
    cpu_job_signal_ref ref;
    expect("acquire for the record-gates case", ggml_sycl::cpu_job_signal_acquire_for_testing(ring, &job, &ref));
    cpu_job_signal_mark mark;
    sycl::event         gate_event;
    expect("wrap for the record-gates case", ggml_sycl::cpu_job_signal_wrap(ref, mark, &gate_event));

    // Control: with no gate outstanding, a published record is released as soon as the worker reaches it.
    const int released = g_records_released.load();
    {
        ggml_sycl::mem_handle owner = make_record_owner(256);
        expect("an owner is made for the control record", owner.valid());
        ggml_sycl::retain_handles_until_event({ owner }, sycl::event{});
    }
    expect("with no gate outstanding the worker finishes the record",
           ggml_sycl::drain_retained_handles(true, 10000) && g_records_released.load() - released == 1);

    std::promise<void> job_done;
    ggml_sycl::cpu_job_signal_gate_push(mark, job_done.get_future().share());
    expect("the snapshot holds the ring's one gate", ggml_sycl::cpu_job_signal_gates_outstanding().size() == 1);
    {
        ggml_sycl::mem_handle owner = make_record_owner(256);
        expect("an owner is made for the gated record", owner.valid());
        ggml_sycl::retain_handles_until_event({ owner }, sycl::event{});
    }
    // A bound on a wait for an outcome, not an ordering: the record's event is complete.
    expect("the worker does not finish a record while a ring's gate is shut",
           !ggml_sycl::drain_retained_handles(true, 500));
    expect("the record's handle is held while the gate is shut", g_records_released.load() - released == 1);
    job_done.set_value();
    expect("the worker finishes the record once the gate opens",
           ggml_sycl::drain_retained_handles(true, 10000) && g_records_released.load() - released == 2);

    job.signal();
    ggml_sycl::cpu_job_signal_recycle(mark);
    expect("the recycle clears the ring's gate", ggml_sycl::cpu_job_signal_gates_outstanding().empty());
    ggml_sycl::cpu_job_signal_ring_shutdown_for_testing(ring);
}

// --- child arms: a throw on the thread that holds the job half ---------------------------------------------------

static int * g_child_flag = nullptr;

// How a child ends when std::terminate ran. _Exit, not abort: a core-dumping process can hold device memory past its
// exit, and the parent only needs to know that the throw ended the process.
static constexpr int child_terminated_status = 86;

// Runs as std::terminate: reports whether the gated command ran before the process ends, then releases anything
// still parked so the child never dies with a parked command.
[[noreturn]] static void child_terminate() {
    const bool ran = wait_flag(g_child_flag, 5, 5000);
    std::printf("CHILD %s\n", ran ? "DEVICE_RELEASED" : "DEVICE_PARKED");
    std::fflush(stdout);
    if (!ran) {
        ggml_sycl::cpu_job_signal_signal_all();
        (void) wait_flag(g_child_flag, 5, 5000);
    }
    std::_Exit(child_terminated_status);
}

static int child_main(const char * arm) {
    sycl::queue q(sycl::gpu_selector_v, production_queue_properties());
    g_child_flag                              = sycl::malloc_host<int>(1, q);
    *g_child_flag                             = 0;
    std::shared_ptr<cpu_job_signal_ring> ring = ggml_sycl::cpu_job_signal_ring_create_for_testing(q);
    cpu_job_signal                       job;
    cpu_job_signal_ref                   ref;
    cpu_job_signal_mark                  mark;
    sycl::event                          gate;
    if (!ring || !ggml_sycl::cpu_job_signal_acquire_for_testing(ring, &job, &ref) ||
        !ggml_sycl::cpu_job_signal_wrap(ref, mark, &gate)) {
        std::printf("CHILD SETUP_FAILED\n");
        return 2;
    }
    (void) submit_gated_write(q, gate, g_child_flag, 5);
    std::set_terminate(child_terminate);
    if (std::strcmp(arm, "pool") == 0) {
        // The pool's worker holds the job half; its prologue throws.
        static ggml_sycl::CpuExpertPool pool;
        pool.init(1);
        (void) pool.submit_batch(
            {}, []() { throw std::runtime_error("planted prologue throw"); }, {}, nullptr, std::move(job));
    } else {
        // Control: a thread holding the job half throws with no handler on its stack, so nothing unwinds.
        std::thread([j = std::move(job)]() mutable {
            (void) j;
            throw std::runtime_error("planted unhandled throw");
        }).detach();
    }
    std::this_thread::sleep_for(std::chrono::seconds(20));
    std::printf("CHILD NO_TERMINATE\n");
    std::fflush(stdout);
    // A regression that swallows the throw must not end the child with a parked command either.
    ggml_sycl::cpu_job_signal_signal_all();
    (void) wait_flag(g_child_flag, 5, 5000);
    return 3;
}

// Runs this binary as `--child <arm>` and returns its stdout and its exit status (-1 when a signal ended it).
static std::string run_child(const char * self, const char * arm, int * exit_status) {
    int fds[2];
    if (pipe(fds) != 0) {
        return "";
    }
    const pid_t pid = fork();
    if (pid == 0) {
        const struct rlimit no_core = { 0, 0 };
        (void) setrlimit(RLIMIT_CORE, &no_core);
        dup2(fds[1], STDOUT_FILENO);
        close(fds[0]);
        close(fds[1]);
        execl(self, self, "--child", arm, static_cast<char *>(nullptr));
        std::_Exit(127);
    }
    close(fds[1]);
    std::string out;
    char        buf[256];
    ssize_t     n;
    while ((n = read(fds[0], buf, sizeof(buf))) > 0) {
        out.append(buf, static_cast<size_t>(n));
    }
    close(fds[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    *exit_status = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    return out;
}

static void case_throwing_job(const char * self) {
    int               rc       = 0;
    const std::string pool_out = run_child(self, "pool", &rc);
    expect("a pool job whose prologue throws still ends the process through std::terminate",
           rc == child_terminated_status);
    expect("the pool job's throw released the device before the process ended",
           pool_out.find("CHILD DEVICE_RELEASED") != std::string::npos);
    const std::string control_out = run_child(self, "control", &rc);
    expect("control: an unhandled throw past a job half ends the process through std::terminate",
           rc == child_terminated_status);
    expect("control: with nothing unwound the device is still parked when the process ends",
           control_out.find("CHILD DEVICE_PARKED") != std::string::npos);
    if (pool_out.find("CHILD DEVICE_RELEASED") == std::string::npos ||
        control_out.find("CHILD DEVICE_PARKED") == std::string::npos) {
        std::printf("pool child said: %s\ncontrol child said: %s\n", pool_out.c_str(), control_out.c_str());
    }
}

int main(int argc, char ** argv) {
    if (argc >= 3 && std::strcmp(argv[1], "--child") == 0) {
        return child_main(argv[2]);
    }
    sycl::device dev;
    try {
        dev = sycl::device(sycl::gpu_selector_v);
    } catch (const sycl::exception &) {
        std::printf("SKIP: no SYCL GPU device\n");
        return 77;
    }
    if (dev.get_backend() != sycl::backend::ext_oneapi_level_zero) {
        std::printf("SKIP: the GPU is not on the Level Zero backend\n");
        return 77;
    }
    start_deadline(120);
    sycl::queue q1(dev, production_queue_properties());
    sycl::queue q2(dev, production_queue_properties());
    int *       flag1 = sycl::malloc_host<int>(1, q1);
    int *       flag2 = sycl::malloc_host<int>(1, q2);

    case_signal_and_recycle(q1, flag1);
    case_abandoned_job_half(q1, flag1);
    case_signal_all(q1, flag1);
    case_unwaited_stream_keeps_its_entries(q1, q2, flag1, flag2);
    case_record_carries_ring_gates(q1);
    case_throwing_job("/proc/self/exe");

    sycl::free(flag1, q1);
    sycl::free(flag2, q2);
    g_done.store(true);
    if (g_failures != 0) {
        std::printf("test-sycl-cpu-job-signal-ring: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test-sycl-cpu-job-signal-ring: PASS\n");
    return 0;
}
