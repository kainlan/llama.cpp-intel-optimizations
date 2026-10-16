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
// host-task thread enqueues that whole backlog serially, about 4 us per command (probe P1, B70, oneAPI 2026.1, driver
// 26.35: 3.3-3.8 ms for 1000 queued kernels against 0.05 ms behind a native event). The decode's scatter used to
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
//   FREE again (recycle) only when it is SIGNALLED, the slot half is gone, and the host has waited an event of a flush on
//   the ring's stream whose serial is at least the one that made the device wait on it. On an in-order stream that
//   event completes after every earlier command, so no command still waits on the entry when its native event is reset.
//
// This header includes only the standard library, so tests/test-sycl-cpu-job-signal-book.cpp checks the book without
// a device.

#include <cstddef>
#include <cstdint>
#include <memory>
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
// host has waited one of them.
struct cpu_job_signal_mark {
    std::shared_ptr<cpu_job_signal_ring> ring;
    uint64_t                             serial = 0;

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

// Signals every acquired entry of every live ring. For the paths that end the process (the ggml abort callback the
// rings install, and the SYCL watchdog's exit), so no device command stays parked on a job that will never signal. Takes each lock with a bounded wait and skips a ring it cannot lock: it must not deadlock an abort
// raised while that lock is held.
void cpu_job_signal_signal_all() noexcept;

}  // namespace ggml_sycl
