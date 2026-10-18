//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

// Host-signalled device events for the CPU expert jobs of the hybrid MoE decode (llama.cpp-qwn5).
//
// Why. A device command that depends on an unfinished host_task is held by the SYCL runtime as a blocked user, and so
// is every command submitted after it on the same in-order queue. When the host_task ends, the runtime's single
// host-task thread enqueues that whole backlog serially, about 4 us per command (measured on the B70, oneAPI 2026.1,
// driver 26.35: 3.3-3.8 ms for 1000 queued kernels against 0.05 ms behind a native event). The decode's scatter used to
// depend on a join host_task per CPU expert job, so the next layer's GPU work started only after that thread had
// enqueued it. A Level Zero event created host-visible with host signal and wait scopes is waited by the device itself:
// the runtime treats a command that depends on it as enqueued, and the job's own thread releases the device with
// zeEventHostSignal.
//
// The hazard. The xe job timeout does not end a command parked on such an event, so every path that abandons a job must
// still signal it. A job's event is signalled by its cpu_job_signal (the job half): explicitly after the job's last
// write, by the destructor otherwise, and by cpu_job_signal_signal_all() on the abort and watchdog paths.
//
// Ownership. The events belong to a per-device ring that the unified cache owns (unified-cache.cpp). They are driver
// objects, not memory: nothing here allocates outside the cache. No raw pointer or address is a key: an entry is named
// by its ring, index and generation, and a stale generation is refused.
//
// Lifetime of one entry, in cpu_job_signal_book:
//   FREE -> ACQUIRED (acquire: the producer gets the job half and the slot half)
//        -> SIGNALLED (the job half signals, once)
//   the slot half either makes the device wait on it in a flush (consume, with that flush's serial) or is dropped;
//   FREE again (recycle) only when it is SIGNALLED, the slot half is gone, and the host has waited an event of a flush
//   on the ring's stream whose serial is at least the one that made the device wait on it. On an in-order stream that
//   event completes after every earlier command, so no command still waits on the entry when its native event is reset.
//
// This header includes only the standard library, so tests/test-sycl-cpu-job-signal-book.cpp checks the book, the
// gate list and the input marker's wait without a device.

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace ggml_sycl {

class cpu_job_signal_book {
  public:
    enum class entry_state : uint8_t {
        FREE,
        ACQUIRED,
        SIGNALLED,
        RESETTING,  // between a recycle's two phases: its native event is being reset with the ring's lock released
    };

    struct entry {
        entry_state state      = entry_state::FREE;
        bool        ref_live   = false;  // the slot half exists and may still make the device wait on it
        uint64_t    serial     = 0;      // the flush that made the device wait on it; 0 = no device wait
        uint32_t    generation = 0;      // bumped at every recycle, so a stale token or ref is refused
    };

    // The serial of an entry retired by a failed reset: larger than any flush's.
    static constexpr uint64_t retired = UINT64_MAX;

    struct counts {
        size_t total     = 0;
        size_t free      = 0;
        size_t acquired  = 0;
        size_t signalled = 0;
        size_t resetting = 0;
    };

    // Adds n FREE entries and returns the index of the first.
    uint32_t add(uint32_t n) {
        const uint32_t first = static_cast<uint32_t>(entries_.size());
        entries_.resize(entries_.size() + n);
        for (uint32_t i = 0; i < n; ++i) {
            free_.push_back(first + n - 1 - i);
        }
        return first;
    }

    // Takes a FREE entry: ACQUIRED, with a live slot half and no device wait. False when none is free.
    bool acquire(uint32_t * id, uint32_t * generation) {
        if (free_.empty()) {
            return false;
        }
        const uint32_t i = free_.back();
        free_.pop_back();
        entry & e   = entries_[i];
        e.state     = entry_state::ACQUIRED;
        e.ref_live  = true;
        e.serial    = 0;
        *id         = i;
        *generation = e.generation;
        return true;
    }

    // ACQUIRED -> SIGNALLED. True exactly once per acquire, and only for the current generation: the caller then
    // signals the native event.
    bool signal(uint32_t id, uint32_t generation) {
        if (!current(id, generation) || entries_[id].state != entry_state::ACQUIRED) {
            return false;
        }
        entries_[id].state = entry_state::SIGNALLED;
        return true;
    }

    // The serial of a new flush. Serials grow with each flush, so on one in-order stream a larger serial's commands
    // were submitted after a smaller one's.
    uint64_t begin_flush() { return ++serial_; }

    // The slot half makes the device wait on the entry in flush `serial`. True when the slot half was live for the
    // current generation; it is gone afterwards.
    bool consume(uint32_t id, uint32_t generation, uint64_t serial) {
        if (serial == 0 || !current(id, generation) || !entries_[id].ref_live) {
            return false;
        }
        entries_[id].ref_live = false;
        entries_[id].serial   = serial;
        return true;
    }

    // The slot half is gone without making the device wait on the entry.
    void drop(uint32_t id, uint32_t generation) {
        if (current(id, generation)) {
            entries_[id].ref_live = false;
        }
    }

    // A recycle frees every SIGNALLED entry whose slot half is gone and whose device wait, if it had one, was in a flush
    // with serial <= upto, once its native event is reset. It runs in two phases because the ring resets the events
    // with its lock released (llama.cpp-42zj): a driver call may block, and the job's own signal takes that lock.
    //
    // Phase one marks each such entry RESETTING and appends its id to *ids; no other recycle, acquire or signal_all
    // takes a RESETTING entry. Returns how many.
    size_t recycle_begin(uint64_t upto, std::vector<uint32_t> * ids) {
        size_t n = 0;
        for (uint32_t i = 0; i < entries_.size(); ++i) {
            entry & e = entries_[i];
            if (e.state != entry_state::SIGNALLED || e.ref_live || e.serial > upto) {
                continue;
            }
            e.state = entry_state::RESETTING;
            ids->push_back(i);
            n += 1;
        }
        return n;
    }

    // Phase two, for one RESETTING entry: freed, under a new generation, when its native event was reset. One that
    // could not be reset is retired, never freed: reused still signalled, it would let the next job's device commands
    // run before that job wrote anything. True when it was freed; an entry that is not RESETTING is left alone.
    bool recycle_finish(uint32_t id, bool reset) {
        if (id >= entries_.size() || entries_[id].state != entry_state::RESETTING) {
            return false;
        }
        entry & e = entries_[id];
        if (!reset) {
            e.state  = entry_state::SIGNALLED;
            e.serial = retired;
            return false;
        }
        e.state  = entry_state::FREE;
        e.serial = 0;
        e.generation += 1;
        free_.push_back(id);
        return true;
    }

    // Both phases, with `reset(id)` (the native reset, returning whether it reset the event) between them. Returns how
    // many were freed.
    template <typename F> size_t recycle(uint64_t upto, F && reset) {
        std::vector<uint32_t> ids;
        recycle_begin(upto, &ids);
        size_t n = 0;
        for (uint32_t id : ids) {
            n += recycle_finish(id, reset(id)) ? 1 : 0;
        }
        return n;
    }

    // Marks every ACQUIRED entry SIGNALLED, calling `fn(id)` for each first (the native signal). Returns how many.
    template <typename F> size_t signal_all(F && fn) {
        size_t n = 0;
        for (uint32_t i = 0; i < entries_.size(); ++i) {
            if (entries_[i].state == entry_state::ACQUIRED) {
                fn(i);
                entries_[i].state = entry_state::SIGNALLED;
                n += 1;
            }
        }
        return n;
    }

    counts count() const {
        counts c;
        c.total = entries_.size();
        for (const entry & e : entries_) {
            c.free += e.state == entry_state::FREE ? 1 : 0;
            c.acquired += e.state == entry_state::ACQUIRED ? 1 : 0;
            c.signalled += e.state == entry_state::SIGNALLED ? 1 : 0;
            c.resetting += e.state == entry_state::RESETTING ? 1 : 0;
        }
        return c;
    }

    size_t size() const { return entries_.size(); }

    const entry & at(uint32_t id) const { return entries_.at(id); }

  private:
    bool current(uint32_t id, uint32_t generation) const {
        return id < entries_.size() && entries_[id].generation == generation;
    }

    std::vector<entry>    entries_;
    std::vector<uint32_t> free_;
    uint64_t              serial_ = 0;
};

// The gates of one ring (llama.cpp-qwn5): the future of each CPU expert job the device waits on its own event, so that
// a host thread about to wait on the device behind such a job first sleeps until the job is done.
//
// Why. With the job's event waited by the device itself, a host wait on the stream behind the job is a wait inside the
// driver, which polls for the job's whole length. On the B70 at -ngl 40 two threads polled there: the submitting
// thread, in the whole-queue waits of the backend's graph code (ggml_sycl_trace_queue_wait; the periodic drain runs
// one every GGML_SYCL_DRAIN_INTERVAL nodes of a directly executed split), and the retained-handle reaper. Both took
// CPU from the job's own row threads. Sleeping on the job's future first leaves the driver only the device work after
// the job. The backend's synchronize, which the scheduler calls for every cross-backend input of a single-device run,
// sleeps the same way, though its stream wait was short there: its own CPU-expert flush runs the graph-boundary drain
// first, and that drain sleeps on the gates and drops them.
//
// A host wait inside the driver may also hold a device-level lock while it polls the device (on driver 26.35, the one
// hostSynchronize takes). A job that needed that lock before it could signal would never start behind such a wait,
// and the wait would never return: that was the first prompt graph of llama.cpp-42zj, with the job's prologue waiting
// its input events natively. The prologue now waits its inputs on a device-written marker with host loads
// (cpu_job_input_marker, below), and the job's one driver call is its signal, which takes no such lock. The gates still
// keep a host wait from polling in front of a running job: every host wait on an event that may stand behind one
// sleeps on that job's gate first, the retained-handle drain worker's included, since each record it drains carries
// the gates of every live ring (cpu_job_signal_gates_outstanding). A gate leaves its list only once its job is known
// to have ended: cleared by a serial that covers it after a sleep that waited it, or by a recycle, which the
// boundary's sleep on the same gates precedes, or with its ring's shutdown, after which no device command waits on the
// ring's events. The list's drop of its oldest gate never runs: one gate is pushed per wrapped entry and a recycle
// clears its entries' gates, so a list holds no more gates than its ring has entries in use, and max_gates is at least
// that (a static_assert after the ring's definition). Past that, a sleep on a gate lasts no longer than the device
// wait behind it would have. The one exception is a job that throws (below), where a gate holds the host until an
// abort that is already under way.
//
// A gate holds only the job's shared_future: no job, no buffer, no event. wait() returns once the job's promise holds a
// value or an exception, a broken promise included, and never throws; nothing calls get(), wait_for() or wait_until()
// on a gate. The waits run on a snapshot taken under the list's lock and released before the first wait, so a producer
// pushing a gate never blocks behind a sleeper; a caller holds no other lock across them either.
//
// The hazard. A gate opens when its job's promise is set, by the worker after the kernel, or broken, by a job destroyed
// unrun. Two paths leave it shut. A job that throws signals its event while it unwinds to the worker's handlers, which
// rethrow; worker_thread has no handler, so std::terminate aborts the process without unwinding further, and the
// promise is neither set nor broken (CpuExpertPool::submit_batch). A waiter on that gate sleeps until the abort, where
// the device wait alone would have returned once the device had copied the job's partial output. And
// CpuExpertPool::stop_workers detaches a worker that did not join within 5 s, whose job may never end: its gate never
// opens, and a waiter that reaches it sleeps for good. The device command parked on that job's event does not end
// either, so the device wait the gate stands in front of would not have returned. A process killed in that state is
// reaped: on the B50, 8 SIGKILLs with a copy and a kernel parked on an unsignalled event were reaped in 30-47 ms, none
// in D-state. The first drew one per-context Engine reset in the kernel log; the card and a co-tenant kept working
// (llama.cpp-qwn5 c-a8ea).
class cpu_job_gate_list {
  public:
    // At least a ring's entries (asserted after the ring's definition), so a push never drops the oldest gate. A dropped
    // gate could leave a host wait in front of a job that has not started.
    static constexpr size_t max_gates = 1024;

    // Adds a job's future and returns its serial, which grows with each push. An empty future is not kept: 0.
    uint64_t push(std::shared_future<void> future) {
        if (!future.valid()) {
            return 0;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (gates_.size() >= max_gates) {
            gates_.pop_front();
        }
        gates_.push_back({ ++serial_, std::move(future) });
        return serial_;
    }

    // Sleeps until every gate pushed before the call has opened. *newest is the newest serial it covered, 0 for none,
    // for clear_upto() once the caller's own device wait has returned. Returns how many gates it waited.
    size_t wait_all(uint64_t * newest) const { return sleep_on(snapshot(UINT64_MAX, newest)); }

    // Sleeps until every gate up to serial `upto` has opened; later gates are not waited. Returns how many it waited.
    size_t wait_upto(uint64_t upto) const {
        uint64_t newest = 0;
        return sleep_on(snapshot(upto, &newest));
    }

    // Drops the gates up to serial `upto`, all of which the caller waited.
    void clear_upto(uint64_t upto) {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!gates_.empty() && gates_.front().serial <= upto) {
            gates_.pop_front();
        }
    }

    // Drops every gate. For a ring that is shut down: its jobs no longer hold the device.
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        gates_.clear();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return gates_.size();
    }

    // Appends every gate's future to *out, in push order, copied under the lock; waits on none. One reserve per call,
    // and none past *out's capacity for an empty list, so an empty list allocates nothing. For a retained-handle record
    // (cpu_job_signal_gates_outstanding).
    void append_outstanding(std::vector<std::shared_future<void>> * out) const {
        std::lock_guard<std::mutex> lock(mutex_);
        out->reserve(out->size() + gates_.size());
        for (const gate & g : gates_) {
            out->push_back(g.future);
        }
    }

  private:
    struct gate {
        uint64_t                 serial = 0;
        std::shared_future<void> future;
    };

    // The futures up to serial `upto`, in push order, copied under the lock.
    std::vector<std::shared_future<void>> snapshot(uint64_t upto, uint64_t * newest) const {
        std::vector<std::shared_future<void>> futures;
        std::lock_guard<std::mutex>           lock(mutex_);
        *newest = 0;
        for (const gate & g : gates_) {
            if (g.serial > upto) {
                break;
            }
            futures.push_back(g.future);
            *newest = g.serial;
        }
        return futures;
    }

    // Called with no lock held.
    static size_t sleep_on(const std::vector<std::shared_future<void>> & futures) {
        for (const std::shared_future<void> & future : futures) {
            future.wait();
        }
        return futures.size();
    }

    mutable std::mutex mutex_;
    std::deque<gate>   gates_;  // in serial order
    uint64_t           serial_ = 0;
};

// The ring itself is defined in unified-cache.cpp; these halves only name an entry in it.
class cpu_job_signal_ring;

// The job half. Signals the entry's native event once: signal() after the job's last host write, or the destructor if
// the job never got that far (a throw unwound past it, or the job was destroyed unrun). Move-only.
class cpu_job_signal {
  public:
    cpu_job_signal() = default;

    cpu_job_signal(std::shared_ptr<cpu_job_signal_ring> ring, uint32_t id, uint32_t generation) :
        ring_(std::move(ring)),
        id_(id),
        generation_(generation) {}

    cpu_job_signal(cpu_job_signal && o) noexcept : ring_(std::move(o.ring_)), id_(o.id_), generation_(o.generation_) {}

    cpu_job_signal & operator=(cpu_job_signal && o) noexcept {
        if (this != &o) {
            signal();
            ring_       = std::move(o.ring_);
            id_         = o.id_;
            generation_ = o.generation_;
        }
        return *this;
    }

    cpu_job_signal(const cpu_job_signal &)             = delete;
    cpu_job_signal & operator=(const cpu_job_signal &) = delete;

    ~cpu_job_signal() { signal(); }

    explicit operator bool() const { return ring_ != nullptr; }

    // Idempotent; a release fence orders the job's host writes before the device can observe the signal.
    void signal() noexcept;

  private:
    std::shared_ptr<cpu_job_signal_ring> ring_;
    uint32_t                             id_         = 0;
    uint32_t                             generation_ = 0;
};

// The flush that makes the device wait on entries: the ring and the flush's serial. Set by the first
// cpu_job_signal_wrap() of a flush, carried with the flush's events, and handed to cpu_job_signal_recycle() once the
// host has waited one of them. gate_serial is set by cpu_job_signal_gate_push() for the same flush; the host sleeps on
// the ring's gates up to it before it waits those events, and the recycle clears them.
struct cpu_job_signal_mark {
    std::shared_ptr<cpu_job_signal_ring> ring;
    uint64_t                             serial      = 0;
    uint64_t                             gate_serial = 0;  // the newest gate the flush pushed (cpu_job_gate_list)

    explicit operator bool() const { return ring != nullptr && serial != 0; }
};

// The slot half: what the pending scatter keeps to make the device wait on the job (cpu_job_signal_wrap, in
// unified-cache.hpp). Dropping it unused tells the ring no device command waits on the entry. Move-only.
class cpu_job_signal_ref {
  public:
    cpu_job_signal_ref() = default;

    cpu_job_signal_ref(std::shared_ptr<cpu_job_signal_ring> ring, uint32_t id, uint32_t generation) :
        ring_(std::move(ring)),
        id_(id),
        generation_(generation) {}

    cpu_job_signal_ref(cpu_job_signal_ref && o) noexcept :
        ring_(std::move(o.ring_)),
        id_(o.id_),
        generation_(o.generation_) {}

    cpu_job_signal_ref & operator=(cpu_job_signal_ref && o) noexcept {
        if (this != &o) {
            reset();
            ring_       = std::move(o.ring_);
            id_         = o.id_;
            generation_ = o.generation_;
        }
        return *this;
    }

    cpu_job_signal_ref(const cpu_job_signal_ref &)             = delete;
    cpu_job_signal_ref & operator=(const cpu_job_signal_ref &) = delete;

    ~cpu_job_signal_ref() { reset(); }

    explicit operator bool() const { return ring_ != nullptr; }

    // Drops the slot half without a device wait.
    void reset() noexcept;

    const std::shared_ptr<cpu_job_signal_ring> & ring() const { return ring_; }

    uint32_t id() const { return id_; }

    uint32_t generation() const { return generation_; }

    // Forgets the entry without telling the ring: the caller has consumed it.
    void release() noexcept { ring_.reset(); }

  private:
    std::shared_ptr<cpu_job_signal_ring> ring_;
    uint32_t                             id_         = 0;
    uint32_t                             generation_ = 0;
};

// The input marker of a CPU expert job (llama.cpp-42zj): how the pool worker learns that the job's inputs have
// landed without entering the driver.
//
// Why. The retained-handle drain worker waited a record's event natively, and libze held a device-level lock across
// that wait while it polled the device. The device work behind the event stood behind a CPU expert job, and the job's
// prologue, waiting its input events natively on the pool's worker, blocked on the same lock: the job could not
// signal, so the drain worker's wait could not end. Any thread that waits inside the driver may be the holder, so the
// job must not need the lock at all.
//
// How. The submitting thread arms the marker on the op's in-order stream right after the job's last input: one
// device task that depends on the job's input events, then makes the device's earlier writes visible to the host
// with a system-scope release fence and stores the marker's serial into a host-pinned word that the unified cache
// owns (unified_cache_cpu_job_input_marker_arm). Discrete BMG has no host-USM atomics, so the device side is a fence
// and a plain store. Each arm on that stream takes the next serial, and the in-order stream runs the tasks in that
// order, so the word only advances: a word at or past a job's serial means its inputs are complete. The word is 32
// bits, which the device stores in one write, and it is compared modulo 2^32, so the serial may wrap (see
// cpu_job_input_marker_reached for why the compare holds). The worker reads the word with acquire loads, pausing and
// then yielding between them: no sleep, no lock and no driver call while it waits, and a wait still unanswered
// after its slow_after is reported once through the log callback, which takes the log's lock. Its job's one driver
// call is its signal (zeEventHostSignal), which on driver 26.35 reaches no lock a host wait holds.
//
// Ownership. The word's allocation belongs to the unified cache, which lets it go at teardown, after its queues
// drained, on the tearing-down thread. A job holds only the cpu_job_input_marker_word below, so a worker that drops
// the last reference to one frees no driver memory. A job does not hold the allocation either, so before the cache
// lets it go it abandons the word (cpu_job_input_marker_abandon): it sets abandoned, then waits until no waiter is
// between its test of that flag and its load of the word. A waiter counts itself in readers for exactly that span, so
// no load of the word can follow its release, and a job still waiting then stops waiting, as the ring's halves make no
// driver call once their ring is shut down. On the two shutdown branches that run once the SYCL runtime is gone, no
// device write will come, so that job is released to compute on inputs that may never have landed: acceptable only
// because those branches run as the process exits.
//
// An unarmed marker is passed at once: a job with no input events, or one whose arm was refused, after which the
// submitting thread waited the inputs itself.
struct cpu_job_input_marker_word;

// Told, on the waiting thread and once per wait, that a wait on `w`'s marker `serial` has lasted `waited_s` seconds.
// It must make no driver call: it runs where the wait runs.
using cpu_job_input_marker_slow_fn = void (*)(const cpu_job_input_marker_word & w, uint32_t serial, double waited_s);

struct cpu_job_input_marker_word {
    uint32_t *                    ptr = nullptr;       // host-pinned, device-written: a view of the cache's allocation
    std::atomic<bool>             abandoned{ false };  // set before that allocation goes: the device writes it no more
    mutable std::atomic<uint32_t> readers{ 0 };        // waiters between their test of abandoned and their word load
    int                           device     = -1;     // the cache's device, for the slow-wait report
    // A wait still yielding after slow_after is reported once through slow (unset: never). With the SYCL watchdog off
    // (GGML_SYCL_OP_TIMEOUT_MS=0) a marker that never arrives is otherwise a silent spin on one core.
    cpu_job_input_marker_slow_fn  slow       = nullptr;
    std::chrono::milliseconds     slow_after = std::chrono::milliseconds(10000);
};

struct cpu_job_input_marker {
    std::shared_ptr<const cpu_job_input_marker_word> word;
    uint32_t                                         serial = 0;  // the inputs are complete once the word reaches it

    bool armed() const { return word != nullptr; }
};

// Why a CPU expert job's input marker was not armed. The submitting thread then waits that job's inputs itself, and
// the first job refused for each reason is logged with that reason's text, so one reason does not hide another.
enum cpu_job_input_marker_refusal {
    CPU_JOB_INPUT_MARKER_REFUSAL_GRAPH_RECORDING,   // a command graph records on the stream
    CPU_JOB_INPUT_MARKER_REFUSAL_NOT_CACHE_STREAM,  // the stream is not a unified cache's queue_
    CPU_JOB_INPUT_MARKER_REFUSAL_NO_WORD,           // the word could not be made, or was let go at teardown
    CPU_JOB_INPUT_MARKER_REFUSAL_NOT_IN_ORDER,      // the stream is not in order
    CPU_JOB_INPUT_MARKER_REFUSAL_ALLOC_FAILED,      // the unified cache could not allocate the word
    CPU_JOB_INPUT_MARKER_REFUSAL_NOT_HOST,          // the word did not resolve to host memory
    CPU_JOB_INPUT_MARKER_REFUSAL_SUBMIT_THREW,      // submitting the marker's device task threw
    CPU_JOB_INPUT_MARKER_REFUSAL_COUNT,
};

inline const char * cpu_job_input_marker_refusal_text(cpu_job_input_marker_refusal why) {
    switch (why) {
        case CPU_JOB_INPUT_MARKER_REFUSAL_GRAPH_RECORDING:
            return "a command graph is recording";
        case CPU_JOB_INPUT_MARKER_REFUSAL_NOT_CACHE_STREAM:
            return "the stream is not a cache's compute stream";
        case CPU_JOB_INPUT_MARKER_REFUSAL_NO_WORD:
            return "its host-pinned word could not be made, or was let go at teardown";
        case CPU_JOB_INPUT_MARKER_REFUSAL_NOT_IN_ORDER:
            return "the stream is not in order, so its markers could be written out of serial order";
        case CPU_JOB_INPUT_MARKER_REFUSAL_ALLOC_FAILED:
            return "allocating its host-pinned word failed";
        case CPU_JOB_INPUT_MARKER_REFUSAL_NOT_HOST:
            return "its word did not resolve to host memory";
        case CPU_JOB_INPUT_MARKER_REFUSAL_SUBMIT_THREW:
            return "submitting its device task threw";
        case CPU_JOB_INPUT_MARKER_REFUSAL_COUNT:
            break;
    }
    return "an unknown reason";
}

// How many pauses the waiter makes before it starts yielding the core instead.
constexpr uint32_t cpu_job_input_marker_pause_spins = 2048;

inline void cpu_job_input_marker_pause() {
#if (defined(__x86_64__) || defined(__i386__)) && !defined(__SYCL_DEVICE_ONLY__)
    __builtin_ia32_pause();
#else
    std::this_thread::yield();
#endif
}

// Whether the word written `seen` has reached `serial`, modulo 2^32: the signed distance from `serial` to `seen`.
//
// The invariant the compare needs: fewer than 2^31 serials separate the word from a waiter's serial. Below the serial,
// the word trails it only by the markers queued ahead of it on the stream and not yet run, far fewer than 2^31, so the
// compare never says reached early, which would let a job read inputs that have not landed. Past the serial, the word
// leads it only by the markers armed and run since; a waiter that slept through 2^31 of those would read not reached
// and wait on until the word came round again: a late return, never an early one.
inline bool cpu_job_input_marker_reached(uint32_t seen, uint32_t serial) {
    return static_cast<int32_t>(seen - serial) >= 0;
}

// True once the job's inputs are complete, or the marker is unarmed or abandoned. The waiter counts itself in readers
// before it tests abandoned and out after its load of the word, which it makes only when the word is not abandoned:
// cpu_job_input_marker_abandon waits for that count, so the word's allocation is never let go under the load. The
// count and the flag are sequentially consistent on both sides (the waiter counts, then tests; the abandoner sets,
// then reads the count), so either the abandoner sees the waiter counted or the waiter sees the flag set. Atomic loads
// and read-modify-writes only.
inline bool cpu_job_input_marker_seen(const cpu_job_input_marker & m) {
    if (!m.armed()) {
        return true;
    }
    const cpu_job_input_marker_word & w = *m.word;
    w.readers.fetch_add(1, std::memory_order_seq_cst);
    const bool seen = w.abandoned.load(std::memory_order_seq_cst) ||
                      cpu_job_input_marker_reached(__atomic_load_n(w.ptr, __ATOMIC_ACQUIRE), m.serial);
    w.readers.fetch_sub(1, std::memory_order_release);
    return seen;
}

// Abandons `w`: sets abandoned, then returns once no waiter is between its test of abandoned and its load of the word,
// after which the word's allocation may go. A waiter stays there for a few instructions, so this yields until it
// leaves: no lock and no driver call.
inline void cpu_job_input_marker_abandon(cpu_job_input_marker_word & w) {
    w.abandoned.store(true, std::memory_order_seq_cst);
    while (w.readers.load(std::memory_order_seq_cst) != 0) {
        std::this_thread::yield();
    }
}

// Returns once cpu_job_input_marker_seen(m). On the pool worker: it never sleeps, and its wait takes no lock and makes
// no driver call. The one exception is the slow report, at most once per wait: the word's slow logs through the log
// callback, which takes the log's lock. The wait does not tick the SYCL watchdog either, so a marker that never
// arrives shows as the watchdog firing, and as that one call of the word's slow once the wait has yielded for its
// slow_after.
inline void cpu_job_input_marker_wait(const cpu_job_input_marker & m) {
    uint32_t                              spins = 0;
    bool                                  told  = false;
    std::chrono::steady_clock::time_point since;
    while (!cpu_job_input_marker_seen(m)) {
        if (spins < cpu_job_input_marker_pause_spins) {
            ++spins;
            cpu_job_input_marker_pause();
            continue;
        }
        if (!told && m.word->slow) {
            const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            if (spins == cpu_job_input_marker_pause_spins) {
                ++spins;
                since = now;
            } else if (now - since >= m.word->slow_after) {
                told = true;
                m.word->slow(*m.word, m.serial, std::chrono::duration<double>(now - since).count());
            }
        }
        std::this_thread::yield();
    }
}

// Signals every acquired entry of every live ring. For the paths that end the process (the ggml abort callback the
// rings install, and the SYCL watchdog's exit), so no device command stays parked on a job that will never signal.
// Takes each lock with a bounded wait and skips a ring it cannot lock: it must not deadlock an abort raised while that
// lock is held.
void cpu_job_signal_signal_all() noexcept;

// The futures of every gate of every live ring, in each ring's push order: every CPU expert job a device command may
// still wait on. A retained-handle record takes them when it is published, and the drain worker sleeps on them before
// its driver wait, so a record behind none of those jobs waits for them too (llama.cpp-42zj). Holds the registry lock
// only to copy the live rings, and each gate list's lock only to copy its gates; waits on nothing and makes no driver
// call. A ring with no gate adds nothing.
std::vector<std::shared_future<void>> cpu_job_signal_gates_outstanding();

}  // namespace ggml_sycl
