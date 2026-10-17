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

    // Frees every SIGNALLED entry whose slot half is gone and whose device wait, if it had one, was in a flush with
    // serial <= upto. `reset(id)` runs for each before it is freed (the native event reset) and returns whether it
    // reset the event. An entry it could not reset is retired, never freed: reused still signalled, it would let the
    // next job's device commands run before that job wrote anything. Returns how many were freed.
    template <typename F> size_t recycle(uint64_t upto, F && reset) {
        size_t n = 0;
        for (uint32_t i = 0; i < entries_.size(); ++i) {
            entry & e = entries_[i];
            if (e.state != entry_state::SIGNALLED || e.ref_live || e.serial > upto) {
                continue;
            }
            if (!reset(i)) {
                e.serial = retired;
                continue;
            }
            e.state  = entry_state::FREE;
            e.serial = 0;
            e.generation += 1;
            free_.push_back(i);
            n += 1;
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
// bits, which the device stores in one write, and it is compared modulo 2^32, so the serial may wrap. The worker reads
// the word with acquire loads, pausing and then yielding between them: no sleep, no lock and no driver call. Its job's
// one driver call is its signal (zeEventHostSignal), which on driver 26.35 reaches no lock a host wait holds.
//
// Ownership. The word's allocation belongs to the unified cache, which lets it go at teardown, after its queues
// drained, on the tearing-down thread. A job holds only the cpu_job_input_marker_word below, so a worker that drops
// the last reference to one frees no driver memory. The cache sets abandoned before it lets the allocation go, and a
// waiter reads that flag before each load of the word, as the ring's halves make no driver call once their ring is
// shut down: a job still waiting then stops waiting.
//
// An unarmed marker is passed at once: a job with no input events, or one whose arm was refused, after which the
// submitting thread waited the inputs itself.
struct cpu_job_input_marker_word {
    uint32_t *        word = nullptr;      // host-pinned, written by the device: a view of the cache's allocation
    std::atomic<bool> abandoned{ false };  // set before that allocation goes: the device writes it no more
};

struct cpu_job_input_marker {
    std::shared_ptr<const cpu_job_input_marker_word> word;
    uint32_t                                         serial = 0;  // the inputs are complete once the word reaches it

    bool armed() const { return word != nullptr; }
};

// How many pauses the waiter makes before it starts yielding the core instead.
constexpr uint32_t cpu_job_input_marker_pause_spins = 2048;

inline void cpu_job_input_marker_pause() {
#if (defined(__x86_64__) || defined(__i386__)) && !defined(__SYCL_DEVICE_ONLY__)
    __builtin_ia32_pause();
#else
    std::this_thread::yield();
#endif
}

// Whether the word written `seen` has reached `serial`, modulo 2^32: true while fewer than 2^31 arms separate them.
inline bool cpu_job_input_marker_reached(uint32_t seen, uint32_t serial) {
    return static_cast<int32_t>(seen - serial) >= 0;
}

// True once the job's inputs are complete, or the marker is unarmed or abandoned. Acquire loads only.
inline bool cpu_job_input_marker_seen(const cpu_job_input_marker & m) {
    if (!m.armed() || m.word->abandoned.load(std::memory_order_acquire)) {
        return true;
    }
    return cpu_job_input_marker_reached(__atomic_load_n(m.word->word, __ATOMIC_ACQUIRE), m.serial);
}

// Returns once cpu_job_input_marker_seen(m). On the pool worker: it never sleeps, takes no lock and makes no driver
// call. It does not tick the SYCL watchdog either, so a marker that never arrives shows as the watchdog firing.
inline void cpu_job_input_marker_wait(const cpu_job_input_marker & m) {
    for (uint32_t spins = 0; !cpu_job_input_marker_seen(m);) {
        if (spins < cpu_job_input_marker_pause_spins) {
            ++spins;
            cpu_job_input_marker_pause();
        } else {
            std::this_thread::yield();
        }
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
