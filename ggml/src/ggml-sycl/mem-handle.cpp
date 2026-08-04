// Smart handle implementation for SYCL unified memory manager.
// See mem-handle.hpp for design and docs/smart-handle-design.md for architecture.

#include "mem-handle.hpp"

#include "common.hpp"
#include "pinned-pool.hpp"    // pinned_chunk_pool chunk-lease API (dyhdl)
#include "unified-cache.hpp"  // get_unified_cache_for_device, unified_cache

#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <iterator>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

namespace ggml_sycl {

// === Global generation counter ===
// File-scoped static — same pattern as g_graph_compute_active in unified-cache.cpp.
// Relaxed ordering: bumps happen under rw_mutex_; resolve_slow acquires the lock
// and sees the consistent cache state regardless of generation ordering.
static std::atomic<uint64_t> g_cache_generation{ 0 };

uint64_t cache_generation() {
    return g_cache_generation.load(std::memory_order_relaxed);
}

void cache_generation_bump() {
    g_cache_generation.fetch_add(1, std::memory_order_relaxed);
}

namespace {

bool valid_cache_device_id(int device) {
    return device >= 0 && device < GGML_SYCL_MAX_DEVICES;
}

struct retained_handle_record {
    std::vector<mem_handle> handles;
    sycl::event             event;
};

struct retained_handle_state {
    std::mutex                         mutex;
    std::condition_variable            cv;
    std::deque<retained_handle_record> queue;
    std::vector<mem_handle>            graph_unwaitable;
    size_t                             active = 0;
};

// The detached drain worker can still be waiting while process shutdown tears down
// static objects.  Keep the synchronization state alive until process exit.
retained_handle_state *                g_retained_handles_state = new retained_handle_state();
std::once_flag                         g_retained_drain_worker_once;
thread_local std::vector<mem_handle> * g_graph_retained_handle_sink = nullptr;

// Whether THIS thread's retained handles belong to a command graph.
//
// Deliberately narrower than ggml_sycl_graph_recording_active() (common.hpp),
// which is true whenever *any* thread is recording -- the depth counter is a
// process-global counter while g_ggml_sycl_graph_recording is thread_local.
// The wide predicate is right for "may I emit a memcpy node?", a conservative
// process-wide constraint. It is the wrong question for "who owns this handle's
// release?", because the sink it selects (g_graph_retained_handle_sink) is
// thread_local: a thread that is not itself recording has no sink, so every
// handle it retains falls through to the process-global graph_unwaitable list.
// That list is one drain_retained_handles() deliberately does not wait on, and
// only a graph invalidation on some other context ever clears it.
//
// Two consequences, both observed in test-thread-safety (llama.cpp-oze0):
//   * transient scratch (get_rows:seq_device in VRAM SCRATCH,
//     get_rows_indices_small_host in host STAGING) stays live for as long as any
//     other context keeps recording, so the next graph boundary's zone reset is
//     correctly refused and graph launch aborts;
//   * release_graph_retained_handles() then drops those handles from whichever
//     context next invalidates its graph, not from the one whose work still
//     depends on them.
//
// A non-recording thread's event is a real, waitable event, so the normal
// event-bound path releases it naturally. If that turns out to be a recorded
// event after all, the drain worker's "command graph" catch below parks it in
// graph_unwaitable anyway -- so this narrowing fails safe.
bool graph_lifetime_retention_active() {
    return g_graph_retained_handle_sink != nullptr || g_ggml_sycl_graph_recording;
}

void retain_handles_for_current_graph(std::vector<mem_handle> handles) {
    if (handles.empty()) {
        return;
    }

    if (g_graph_retained_handle_sink) {
        g_graph_retained_handle_sink->insert(g_graph_retained_handle_sink->end(),
                                             std::make_move_iterator(handles.begin()),
                                             std::make_move_iterator(handles.end()));
        return;
    }

    auto &                      state = *g_retained_handles_state;
    std::lock_guard<std::mutex> lock(state.mutex);
    state.graph_unwaitable.insert(state.graph_unwaitable.end(), std::make_move_iterator(handles.begin()),
                                  std::make_move_iterator(handles.end()));
}

void retained_handle_drain_loop() {
    for (;;) {
        retained_handle_record record;
        {
            auto &                       state = *g_retained_handles_state;
            std::unique_lock<std::mutex> lock(state.mutex);
            state.cv.wait(lock, [&state] { return !state.queue.empty(); });
            record = std::move(state.queue.front());
            state.queue.pop_front();
            ++state.active;
        }

        try {
            struct event_wait_watchdog_guard {
                event_wait_watchdog_guard() {
                    ggml_sycl_watchdog_start();
                    ggml_sycl_watchdog_heartbeat();
                    ggml_sycl_watchdog_non_graph_begin();
                }

                ~event_wait_watchdog_guard() { ggml_sycl_watchdog_non_graph_end(); }
            } watchdog_guard;

            record.event.wait_and_throw();
        } catch (const std::exception & e) {
            const std::string msg = e.what();
            if (msg.find("command graph") != std::string::npos || msg.find("Command Graph") != std::string::npos) {
                auto &                      state = *g_retained_handles_state;
                std::lock_guard<std::mutex> lock(state.mutex);
                state.graph_unwaitable.insert(state.graph_unwaitable.end(),
                                              std::make_move_iterator(record.handles.begin()),
                                              std::make_move_iterator(record.handles.end()));
                record.handles.clear();
                GGML_SYCL_DEBUG(
                    "[MEM-HANDLE] retained %zu leases for command-graph lifetime; graph events are not waitable\n",
                    state.graph_unwaitable.size());
            } else {
                GGML_LOG_ERROR("[MEM-HANDLE] event-bound lease wait failed: %s\n", e.what());
            }
        } catch (...) {
            GGML_LOG_ERROR("[MEM-HANDLE] event-bound lease wait failed with unknown exception\n");
        }
        record.handles.clear();

        {
            auto &                      state = *g_retained_handles_state;
            std::lock_guard<std::mutex> lock(state.mutex);
            --state.active;
        }
        g_retained_handles_state->cv.notify_all();
    }
}

void start_retained_handle_drain_worker() {
    std::thread(retained_handle_drain_loop).detach();
}

}  // namespace

static size_t mem_handle_hash_combine(size_t seed, size_t value) {
    return detail::cache_hash_combine(seed, value);
}

static bool mem_handle_cache_identity_equal(const mem_handle & a, const mem_handle & b) {
    if (a.kind() != b.kind() || a.device() != b.device()) {
        return false;
    }

    if (a.is_weight()) {
        return a.key() == b.key();
    }

    if (a.is_arena() || a.kind() == mem_handle_kind::CHUNK_LEASE) {
        return a.zone_id() == b.zone_id() && a.offset() == b.offset() && a.size() == b.size() &&
               a.generation() == b.generation();
    }

    return false;
}

// === mem_handle factory methods ===

mem_handle mem_handle::from_weight(const unified_cache_key & key, int device) {
    mem_handle h;
    h.kind_   = mem_handle_kind::WEIGHT;
    h.device_ = device;
    h.key_    = key;
    h.gen_    = 0;  // Stale — first resolve() will query the cache
    h.cached_ = {};
    return h;
}

// llama.cpp-vtf7f: package a pre-acquired lease into a mem_handle.  The
// caller has already incremented entry->in_use_count via
// unified_cache::acquire_weight_lease — ownership of that increment is
// transferred to the new handle, whose dtor will release exactly once.
mem_handle mem_handle::from_weight_lease_locked(const ggml_sycl_cache_id & key_id,
                                                int                        device,
                                                void *                     ptr,
                                                ggml_layout_mode           layout,
                                                bool                       on_device,
                                                unified_cache_entry *      entry) {
    unified_cache_key key;
    key.type      = cache_entry_type::DENSE_WEIGHT;
    key.id        = key_id;
    key.layer_id  = -1;
    key.expert_id = -1;
    return from_weight_lease_locked(key, device, ptr, layout, on_device, entry);
}

mem_handle mem_handle::from_weight_lease_snapshot(const ggml_sycl_cache_id & key_id,
                                                  int                        device,
                                                  void *                     ptr,
                                                  ggml_layout_mode           layout,
                                                  bool                       on_device,
                                                  unified_cache_entry *      entry,
                                                  std::shared_ptr<void>      storage_owner,
                                                  bool                       has_ready_event,
                                                  const sycl::event &        ready_event) {
    unified_cache_key key{ cache_entry_type::DENSE_WEIGHT, key_id, -1, -1 };
    return from_weight_lease_snapshot(key, device, ptr, layout, on_device, entry, std::move(storage_owner),
                                      has_ready_event, ready_event);
}

mem_handle mem_handle::from_weight_lease_locked(const unified_cache_key & key,
                                                int                       device,
                                                void *                    ptr,
                                                ggml_layout_mode          layout,
                                                bool                      on_device,
                                                unified_cache_entry *     entry) {
    const auto        owner     = entry ? entry->storage_owner : std::shared_ptr<void>{};
    const bool        has_event = entry && entry->has_ready_event;
    const sycl::event event     = has_event ? entry->ready_event : sycl::event{};
    return from_weight_lease_snapshot(key, device, ptr, layout, on_device, entry, owner, has_event, event);
}

mem_handle mem_handle::from_weight_lease_snapshot(const unified_cache_key & key,
                                                  int                       device,
                                                  void *                    ptr,
                                                  ggml_layout_mode          layout,
                                                  bool                      on_device,
                                                  unified_cache_entry *     entry,
                                                  std::shared_ptr<void>     storage_owner,
                                                  bool                      has_ready_event,
                                                  const sycl::event &       ready_event) {
    mem_handle h;
    h.kind_   = mem_handle_kind::WEIGHT;
    h.device_ = device;
    h.key_    = key;
    h.gen_    = cache_generation();  // Fresh — no slow-path re-query
    h.cached_ = { ptr, layout, on_device };
    if (has_ready_event) {
        h.cached_.has_ready_event = true;
        h.cached_.ready_event     = ready_event;
    }
    h.leased_entry_         = entry;  // ownership of the refcount bump transferred
    h.leased_storage_owner_ = std::move(storage_owner);

    if (ptr != nullptr && valid_cache_device_id(device)) {
        unified_cache * cache = get_existing_unified_cache_for_device(device);
        if (cache) {
            const int vram_idx = cache->arena_acquire_chunk_lease(ptr);
            if (vram_idx >= 0) {
                h.chunk_source_      = 2;
                h.chunk_device_      = device;
                h.vram_chunk_idx_    = vram_idx;
                h.host_chunk_handle_ = UINT64_MAX;
            } else {
                const uint64_t host_handle = cache->host_acquire_chunk_lease(ptr);
                if (host_handle != pinned_chunk_pool::INVALID_CHUNK_HANDLE) {
                    h.chunk_source_      = 1;
                    h.chunk_device_      = device;
                    h.host_chunk_handle_ = host_handle;
                    h.vram_chunk_idx_    = -1;
                }
            }
        }
    }
    return h;
}

mem_handle mem_handle::from_cache_id(const ggml_sycl_cache_id & id, int device) {
    unified_cache_key key;
    key.type      = cache_entry_type::DENSE_WEIGHT;
    key.id        = id;
    key.layer_id  = -1;
    key.expert_id = -1;
    return from_weight(key, device);
}

mem_handle mem_handle::from_direct(void * ptr, ggml_layout_mode layout, bool on_device, int device) {
    mem_handle h;
    h.kind_   = mem_handle_kind::DIRECT;
    h.device_ = device;
    h.key_    = {};
    h.gen_    = 0;
    h.cached_ = { ptr, layout, on_device };
    return h;
}

mem_handle mem_handle::from_arena_zone(int zone_id, size_t offset, size_t size, int device, uint64_t generation) {
    mem_handle h;
    // Map zone_id to the appropriate arena handle kind.
    // vram_zone_id: KV=0, WEIGHT=1, ONEDNN=2, RUNTIME=3, SCRATCH=4
    switch (zone_id) {
        case static_cast<int>(vram_zone_id::RUNTIME):
            h.kind_ = mem_handle_kind::ARENA_RUNTIME;
            break;
        case static_cast<int>(vram_zone_id::SCRATCH):
            h.kind_ = mem_handle_kind::ARENA_SCRATCH;
            break;
        case static_cast<int>(vram_zone_id::ONEDNN):
            h.kind_ = mem_handle_kind::ARENA_ONEDNN;
            break;
        default:
            GGML_LOG_WARN("[MEM-HANDLE] from_arena_zone: unexpected zone_id %d, defaulting to ARENA_RUNTIME\n",
                          zone_id);
            h.kind_ = mem_handle_kind::ARENA_RUNTIME;
            break;
    }
    h.device_    = device;
    h.zone_id_   = zone_id;
    h.offset_    = offset;
    h.size_      = size;
    h.arena_gen_ = generation;
    h.gen_       = 0;  // Force first resolve
    h.cached_    = {};
    return h;
}

// Compatibility/test bridge: wrap a raw pointer in a handle that refcounts the
// owning arena chunk for the lifetime of the returned mem_handle. Production
// allocation paths should keep the mem_handle they received when allocating.
mem_handle mem_handle::from_chunk_ptr(void * ptr, int device, ggml_layout_mode layout, bool on_device) {
    mem_handle h;
    h.device_ = device;
    h.cached_ = { ptr, layout, on_device };
    h.gen_    = 0;

    if (ptr == nullptr) {
        h.kind_ = mem_handle_kind::DIRECT;
        return h;
    }

    unified_cache * cache = valid_cache_device_id(device) ? get_existing_unified_cache_for_device(device) : nullptr;
    if (cache) {
        // Priority 1: VRAM arena (pointer is device-resident).
        const int vram_idx = cache->arena_acquire_chunk_lease(ptr);
        if (vram_idx >= 0) {
            h.kind_              = mem_handle_kind::CHUNK_LEASE;
            h.chunk_source_      = 2;
            h.chunk_device_      = device;
            h.vram_chunk_idx_    = vram_idx;
            h.host_chunk_handle_ = UINT64_MAX;
            return h;
        }

        // Priority 2: host pinned_chunk_pool.
        const uint64_t host_handle = cache->host_acquire_chunk_lease(ptr);
        if (host_handle != pinned_chunk_pool::INVALID_CHUNK_HANDLE) {
            h.kind_              = mem_handle_kind::CHUNK_LEASE;
            h.chunk_source_      = 1;
            h.chunk_device_      = device;
            h.host_chunk_handle_ = host_handle;
            h.vram_chunk_idx_    = -1;
            return h;
        }
    }

    // Not in any known arena — downgrade to raw DIRECT.  This is correct:
    // the pointer belongs to an allocation whose lifetime we don't manage
    // (mmap, external malloc, etc.), so there's nothing to refcount.
    h.kind_ = mem_handle_kind::DIRECT;
    return h;
}

namespace {

void release_owned_alloc_handle(alloc_handle * handle) {
    if (!handle) {
        return;
    }
    if (handle->ptr && !ggml_sycl_is_shutting_down()) {
        bool released = unified_free(*handle);
        if (!released) {
            GGML_LOG_WARN("[MEM-HANDLE] owning alloc release failed ptr=%p size=%zu device=%d\n", handle->ptr,
                          handle->size, handle->device);
        }
    }
    delete handle;
}

}  // namespace

mem_handle mem_handle::from_owned_alloc(alloc_handle handle, ggml_layout_mode layout) {
    if (!handle.ptr) {
        return {};
    }

    const bool on_device = handle.tier == alloc_tier::DEVICE_VRAM;
    mem_handle h         = from_direct(handle.ptr, layout, on_device, on_device ? handle.device : HOST_DEVICE);
    h.offset_            = 0;
    h.size_              = handle.size;
    h.owned_alloc_ = std::shared_ptr<alloc_handle>(new alloc_handle(std::move(handle)), release_owned_alloc_handle);
    return h;
}

mem_handle mem_handle::slice(size_t byte_offset, size_t byte_size) const {
    if (!owned_alloc_) {
        return {};
    }
    if (byte_offset > size_ || byte_size > size_ - byte_offset) {
        return {};
    }

    resolved_ptr base = resolve();
    if (!base.ptr) {
        return {};
    }

    mem_handle h = *this;
    h.cached_    = base;
    h.cached_.ptr = static_cast<void *>(static_cast<uint8_t *>(base.ptr) + byte_offset);
    h.offset_ += byte_offset;
    h.size_ = byte_size;
    return h;
}

void mem_handle::set_ready_event(const sycl::event & event) {
    mem_handle_lock_guard g(lock_);
    cached_.has_ready_event = true;
    cached_.ready_event     = event;
}

void mem_handle::clear_ready_event() {
    mem_handle_lock_guard g(lock_);
    cached_.has_ready_event = false;
    cached_.ready_event     = {};
}

// === resolve ===

resolved_ptr mem_handle::resolve() const {
    // Fast path under the handle lock.  The copy of `cached_` made here is what
    // used to race: resolved_ptr carries a sycl::event, i.e. a shared_ptr, so a
    // concurrent resolve_slow() rewriting cached_ turned this copy into a
    // refcount operation on a freed control block — a SIGSEGV inside resolve()
    // itself, which is exactly where llama.cpp-c48l lands.  The lock is dropped
    // before any slow path so it is never held across a unified_cache call.
    {
        mem_handle_lock_guard g(lock_);

        // DIRECT and CHUNK_LEASE handles are never stale — they wrap a raw
        // pointer that is kept alive by either the caller's lifetime (DIRECT)
        // or by the chunk lease refcount (CHUNK_LEASE, dyhdl).
        if (kind_ == mem_handle_kind::DIRECT || kind_ == mem_handle_kind::CHUNK_LEASE) {
            return cached_;
        }

        // Arena handles: check arena generation, then resolve base + offset.
        if (kind_ >= mem_handle_kind::ARENA_RUNTIME && kind_ <= mem_handle_kind::ARENA_ONEDNN) {
            // If we have a cached pointer and the generation hasn't changed,
            // return immediately.
            if (cached_.ptr != nullptr && gen_ == arena_gen_) {
                return cached_;
            }
        } else {
            // WEIGHT handle: compare cached generation against global.
            if (gen_ == cache_generation() && cached_.ptr != nullptr) {
                return cached_;
            }
        }
    }

    if (kind_ >= mem_handle_kind::ARENA_RUNTIME && kind_ <= mem_handle_kind::ARENA_ONEDNN) {
        return resolve_arena();
    }

    return resolve_slow();
}

// === resolve(device_id) — dispatch-device overload ===
// Resolves the pointer first, then rejects only device-resident pointers whose
// allocator owner is not the caller's device. Host-pinned/host-mmap pointers are
// device-agnostic from the dispatcher's perspective; device_ remains the cache
// owner used for re-resolution and cleanup.

resolved_ptr mem_handle::resolve(int device_id) const {
    resolved_ptr r = resolve();
    if (!r.ptr || !r.on_device) {
        return r;
    }
    if (device_ != HOST_DEVICE && device_ != device_id) {
        GGML_LOG_WARN(
            "mem_handle::resolve(device_id=%d): wrong-device access — handle owns "
            "device %d (kind=%d). Returning null.\n",
            device_id, device_, static_cast<int>(kind_));
        return {};
    }
    return r;
}

// === resolve_slow ===
// Re-query the unified cache and acquire a lease on the resolved entry.
// Called ~0-3 times per inference run (only on generation mismatch, which
// means an eviction/promotion just happened) — or on the first resolve of
// a newly-constructed handle.
//
// Lifetime contract (llama.cpp-vtf7f): while this handle is alive and holds
// a lease (leased_entry_ != nullptr), the backing cache entry cannot be
// evicted or freed.  On generation mismatch we MUST release the old lease
// before acquiring a new one — otherwise two handles exist on the same
// entry instance, and leak tracking breaks.

resolved_ptr mem_handle::resolve_slow() const {
    // Two threads resolving the SAME shared handle both land here (a generation
    // bump invalidates every handle at once).  The old code released the lease
    // in place before re-acquiring, so both threads decremented the one lease
    // that was held: in_use_count fell below the number of live references and
    // the evictor was free to free an entry still in use.
    //
    // Instead: detach the current lease under the lock, acquire a fresh one
    // OUTSIDE the lock, then publish.  Each thread releases exactly what it
    // detached, so every acquire is matched by exactly one release and a
    // resolve that loses the publish race releases its own lease rather than
    // leaking it.
    if (!valid_cache_device_id(device_)) {
        lease_state stale;
        {
            mem_handle_lock_guard g(lock_);
            stale   = take_lease_state_locked();
            cached_ = {};
            gen_    = cache_generation();
        }
        release_lease_state(stale);
        return {};
    }

    unified_cache * cache = get_unified_cache_for_device(device_);
    if (!cache) {
        return {};
    }

    // Acquire under shared_lock; visible to any future evictor via acq_rel
    // ordering on the in_use_count atomic.  Done with the handle lock dropped:
    // the handle lock must never be held across a unified_cache call.
    auto result = cache->acquire_entry_lease(key_);
    if (!result) {
        // No cache hit; leave handle unpinned.
        lease_state stale;
        {
            mem_handle_lock_guard g(lock_);
            stale   = take_lease_state_locked();
            cached_ = {};
            gen_    = cache_generation();
        }
        release_lease_state(stale);
        return {};
    }

    resolved_ptr resolved = { result.ptr, result.layout, result.on_device };
    if (result.has_ready_event) {
        resolved.has_ready_event = true;
        resolved.ready_event     = result.ready_event;
    }

    lease_state fresh;
    fresh.entry         = result.entry;  // may be nullptr for S1-PRELOAD direct entries
    fresh.storage_owner = std::move(result.storage_owner);

    // llama.cpp-dyhdl: also pin the underlying arena chunk.  Belt + suspenders
    // alongside the cache_entry lease: entry refcount prevents cache-layer
    // eviction, chunk refcount prevents arena-layer munmap.  If the resolved
    // ptr is not in any known arena (e.g. mmap-backed S1-PRELOAD direct
    // entries), chunk_source stays 0 and release is a no-op.
    const int vram_idx = cache->arena_acquire_chunk_lease(resolved.ptr);
    if (vram_idx >= 0) {
        fresh.chunk_source      = 2;
        fresh.chunk_device      = device_;
        fresh.vram_chunk_idx    = vram_idx;
        fresh.host_chunk_handle = UINT64_MAX;
    } else {
        const uint64_t host_handle = cache->host_acquire_chunk_lease(resolved.ptr);
        if (host_handle != pinned_chunk_pool::INVALID_CHUNK_HANDLE) {
            fresh.chunk_source      = 1;
            fresh.chunk_device      = device_;
            fresh.host_chunk_handle = host_handle;
            fresh.vram_chunk_idx    = -1;
        }
    }

    lease_state stale;
    {
        mem_handle_lock_guard g(lock_);
        stale = take_lease_state_locked();
        store_lease_state_locked(fresh);
        cached_ = resolved;
        gen_    = cache_generation();
    }
    release_lease_state(stale);

    return resolved;
}

// === lease_state helpers ===

mem_handle::lease_state mem_handle::take_lease_state_locked() const {
    lease_state state;
    state.entry             = leased_entry_;
    state.chunk_source      = chunk_source_;
    state.host_chunk_handle = host_chunk_handle_;
    state.vram_chunk_idx    = vram_chunk_idx_;
    state.chunk_device      = chunk_device_;
    state.storage_owner     = std::move(leased_storage_owner_);

    leased_entry_      = nullptr;
    chunk_source_      = 0;
    host_chunk_handle_ = UINT64_MAX;  // pinned_chunk_pool::INVALID_CHUNK_HANDLE
    vram_chunk_idx_    = -1;
    chunk_device_      = -1;
    leased_storage_owner_.reset();

    return state;
}

void mem_handle::store_lease_state_locked(const lease_state & state) const {
    leased_entry_      = state.entry;
    chunk_source_      = state.chunk_source;
    host_chunk_handle_ = state.host_chunk_handle;
    vram_chunk_idx_    = state.vram_chunk_idx;
    chunk_device_      = state.chunk_device;
    leased_storage_owner_ = state.storage_owner;
}

void mem_handle::release_lease_state(const lease_state & state) noexcept {
    if (state.entry) {
        // fetch_sub on copyable_atomic_u32::v.  The entry is guaranteed to
        // still exist (this lease held it); after this decrement the entry
        // may be evicted, but we never dereference the pointer again.
        state.entry->in_use_count.fetch_sub(1);
    }

    // llama.cpp-dyhdl: release chunk-level lease if held.  Chunk leases are
    // orthogonal to cache_entry leases — a WEIGHT handle may hold both
    // (cache_entry + its backing arena chunk), a CHUNK_LEASE handle holds
    // only the chunk.
    if (state.chunk_source != 0 && state.chunk_device >= 0) {
        unified_cache * cache = valid_cache_device_id(state.chunk_device) ?
                                    get_existing_unified_cache_for_device(state.chunk_device) :
                                    nullptr;
        if (cache) {
            if (state.chunk_source == 1) {
                cache->host_release_chunk_lease(state.host_chunk_handle);
            } else if (state.chunk_source == 2) {
                cache->arena_release_chunk_lease(state.vram_chunk_idx);
            }
        }
    }
}

// === release_lease ===
// Decrement the backing entry's in_use_count if we're currently leasing one.
// Safe to call from any context; no lock acquired.  After this call,
// leased_entry_ is nulled so the dtor / next release is idempotent.

void mem_handle::release_lease() noexcept {
    // Detach under the lock, release outside it: release_lease_state() calls
    // into unified_cache, and the handle lock must never be held across such a
    // call.  Detaching first also makes concurrent releases exactly-once —
    // whichever caller wins the swap owns the decrement, the loser sees a null
    // state and does nothing.
    lease_state state;
    {
        mem_handle_lock_guard g(lock_);
        state = take_lease_state_locked();
    }
    release_lease_state(state);
}

bool mem_handle::operator==(const mem_handle & other) const {
    resolved_ptr a = resolve();
    resolved_ptr b = other.resolve();

    if (a.ptr != nullptr || b.ptr != nullptr) {
        return a.ptr != nullptr && a.ptr == b.ptr;
    }

    return mem_handle_cache_identity_equal(*this, other);
}

size_t mem_handle::hash() const {
    resolved_ptr r = resolve();
    if (r.ptr != nullptr) {
        return std::hash<void *>()(r.ptr);
    }

    size_t h = 0;
    h        = mem_handle_hash_combine(h, std::hash<int>()(static_cast<int>(kind_)));
    h        = mem_handle_hash_combine(h, std::hash<int>()(device_));

    if (is_weight()) {
        h = mem_handle_hash_combine(h, unified_cache_key_hash{}(key_));
    } else if (is_arena() || kind_ == mem_handle_kind::CHUNK_LEASE) {
        h = mem_handle_hash_combine(h, std::hash<int>()(zone_id_));
        h = mem_handle_hash_combine(h, std::hash<size_t>()(offset_));
        h = mem_handle_hash_combine(h, std::hash<size_t>()(size_));
        h = mem_handle_hash_combine(h, std::hash<uint64_t>()(arena_gen_));
    }

    return h;
}

size_t mem_handle::stable_identity_hash() const {
    mem_handle_lock_guard g(lock_);
    return stable_identity_hash_locked();
}

size_t mem_handle::stable_identity_hash_locked() const {
    size_t h = 0;
    h        = mem_handle_hash_combine(h, std::hash<int>()(static_cast<int>(kind_)));
    h        = mem_handle_hash_combine(h, std::hash<int>()(device_));

    if (is_weight()) {
        h = mem_handle_hash_combine(h, unified_cache_key_hash{}(key_));
    } else if (is_arena()) {
        h = mem_handle_hash_combine(h, std::hash<int>()(zone_id_));
        h = mem_handle_hash_combine(h, std::hash<size_t>()(offset_));
        h = mem_handle_hash_combine(h, std::hash<size_t>()(size_));
        h = mem_handle_hash_combine(h, std::hash<uint64_t>()(arena_gen_));
    } else if (kind_ == mem_handle_kind::CHUNK_LEASE) {
        h = mem_handle_hash_combine(h, std::hash<int>()(chunk_device_));
        h = mem_handle_hash_combine(h, std::hash<uint8_t>()(chunk_source_));
        h = mem_handle_hash_combine(h, std::hash<uint64_t>()(host_chunk_handle_));
        h = mem_handle_hash_combine(h, std::hash<int32_t>()(vram_chunk_idx_));
        h = mem_handle_hash_combine(h, std::hash<void *>()(cached_.ptr));
        h = mem_handle_hash_combine(h, std::hash<size_t>()(size_));
    } else if (owned_alloc_) {
        h = mem_handle_hash_combine(h, std::hash<uint64_t>()(owned_alloc_->alloc_id));
        h = mem_handle_hash_combine(h, std::hash<int>()(owned_alloc_->device));
        h = mem_handle_hash_combine(h, std::hash<int>()(static_cast<int>(owned_alloc_->tier)));
        h = mem_handle_hash_combine(h, std::hash<int>()(static_cast<int>(owned_alloc_->role)));
        h = mem_handle_hash_combine(h, std::hash<int>()(static_cast<int>(owned_alloc_->category)));
        h = mem_handle_hash_combine(h, std::hash<size_t>()(owned_alloc_->size));
        h = mem_handle_hash_combine(h, std::hash<size_t>()(offset_));
        h = mem_handle_hash_combine(h, std::hash<size_t>()(size_));
    } else {
        h = mem_handle_hash_combine(h, std::hash<void *>()(cached_.ptr));
        h = mem_handle_hash_combine(h, std::hash<size_t>()(size_));
    }

    return h;
}

bool mem_handle::stable_identity_equal(const mem_handle & other) const {
    if (kind_ != other.kind_ || device_ != other.device_) {
        return false;
    }

    if (is_weight()) {
        return key_ == other.key_;
    }

    if (is_arena()) {
        return zone_id_ == other.zone_id_ && offset_ == other.offset_ && size_ == other.size_ &&
               arena_gen_ == other.arena_gen_;
    }

    // The remaining branches read mutable state on BOTH handles.  Snapshot each
    // side under its own lock rather than holding two locks at once — there is
    // no lock order between two arbitrary handles.
    struct identity_snapshot {
        const void * ptr               = nullptr;
        uint8_t      chunk_source      = 0;
        uint64_t     host_chunk_handle = UINT64_MAX;
        int32_t      vram_chunk_idx    = -1;
        int          chunk_device      = -1;
    };

    auto snapshot = [](const mem_handle & h) {
        mem_handle_lock_guard g(h.lock_);
        identity_snapshot     s;
        s.ptr               = h.cached_.ptr;
        s.chunk_source      = h.chunk_source_;
        s.host_chunk_handle = h.host_chunk_handle_;
        s.vram_chunk_idx    = h.vram_chunk_idx_;
        s.chunk_device      = h.chunk_device_;
        return s;
    };
    const identity_snapshot self   = snapshot(*this);
    const identity_snapshot theirs = snapshot(other);

    if (kind_ == mem_handle_kind::CHUNK_LEASE) {
        return self.chunk_device == theirs.chunk_device && self.chunk_source == theirs.chunk_source &&
               self.host_chunk_handle == theirs.host_chunk_handle && self.vram_chunk_idx == theirs.vram_chunk_idx &&
               self.ptr == theirs.ptr && size_ == other.size_;
    }

    if (owned_alloc_ || other.owned_alloc_) {
        return owned_alloc_ && other.owned_alloc_ && owned_alloc_->alloc_id == other.owned_alloc_->alloc_id &&
               owned_alloc_->device == other.owned_alloc_->device && owned_alloc_->tier == other.owned_alloc_->tier &&
               owned_alloc_->role == other.owned_alloc_->role &&
               owned_alloc_->category == other.owned_alloc_->category && owned_alloc_->size == other.owned_alloc_->size &&
               offset_ == other.offset_ && size_ == other.size_;
    }

    return self.ptr == theirs.ptr && size_ == other.size_;
}

bool mem_handle::has_stable_owner_identity() const {
    return is_weight() || is_arena() || kind_ == mem_handle_kind::CHUNK_LEASE || owned_alloc_ != nullptr;
}

void mem_handle::set_debug_owner(const char * owner_tag) {
    debug_owner_tag_ = owner_tag ? owner_tag : "";
}

mem_handle_debug_info mem_handle::debug_info() const {
    mem_handle_debug_info info;
    info.kind                = kind_;
    info.device              = device_;
    info.zone_id             = zone_id_;
    info.offset              = offset_;
    info.size                = size_;
    info.generation          = arena_gen_;
    info.has_stable_identity = has_stable_owner_identity();
    info.owner_tag           = debug_owner_tag_ ? debug_owner_tag_ : "";

    // One critical section for every mutable field: valid() and
    // stable_identity_hash() each take the lock, and the spinlock is not
    // recursive, so use the *_locked form here.
    mem_handle_lock_guard g(lock_);
    info.valid                = cached_.ptr != nullptr;
    info.has_ready_event      = cached_.has_ready_event;
    info.stable_identity_hash = stable_identity_hash_locked();
    return info;
}

// === destructor / copy / move ===

mem_handle::~mem_handle() {
    release_lease();
}

// llama.cpp-dyhdl helper: re-acquire a chunk lease when a handle is copied.
// Acquires via the pool API rather than bumping the atomic directly — keeps
// all count mutations behind the pool's API and correctly handles the case
// where the pool has been destroyed between the original acquisition and
// the copy (returns a null handle, chunk_source_ gets zeroed below).
static void bump_chunk_lease_for_copy(uint8_t      chunk_source,
                                      int          chunk_device,
                                      const void * ptr,
                                      uint64_t &   out_host_handle,
                                      int32_t &    out_vram_idx) {
    if (chunk_source == 0 || chunk_device < 0 || ptr == nullptr) {
        out_host_handle = UINT64_MAX;
        out_vram_idx    = -1;
        return;
    }
    unified_cache * cache =
        valid_cache_device_id(chunk_device) ? get_existing_unified_cache_for_device(chunk_device) : nullptr;
    if (!cache) {
        out_host_handle = UINT64_MAX;
        out_vram_idx    = -1;
        return;
    }
    if (chunk_source == 1) {
        out_host_handle = cache->host_acquire_chunk_lease(ptr);
        out_vram_idx    = -1;
    } else if (chunk_source == 2) {
        out_host_handle = UINT64_MAX;
        out_vram_idx    = cache->arena_acquire_chunk_lease(ptr);
    } else {
        out_host_handle = UINT64_MAX;
        out_vram_idx    = -1;
    }
}

// Copying a handle that another thread is concurrently resolving is legitimate:
// ggml_tensor_extra_gpu::data_handle[] entries are handed out as copies while
// other contexts resolve them.  So the copy reads the source's mutable state
// under the SOURCE's lock, then finishes (chunk lease, publish) with no handle
// lock held — bump_chunk_lease_for_copy() calls into unified_cache.
mem_handle::mem_handle(const mem_handle & other) :
    kind_(other.kind_),
    device_(other.device_),
    key_(other.key_),
    zone_id_(other.zone_id_),
    offset_(other.offset_),
    size_(other.size_),
    arena_gen_(other.arena_gen_),
    owned_alloc_(other.owned_alloc_),
    host_chunk_handle_(UINT64_MAX),
    vram_chunk_idx_(-1),
    debug_owner_tag_(other.debug_owner_tag_) {
    {
        // Read `other`'s mutable resolve state under its lock, and bump the
        // cache_entry lease refcount there too so each handle independently
        // keeps the entry alive.  fetch_add on copyable_atomic_u32 is lock-free
        // and touches no cache lock, so it is safe inside the critical section.
        mem_handle_lock_guard g(other.lock_);
        gen_          = other.gen_;
        cached_       = other.cached_;
        leased_entry_         = other.leased_entry_;
        leased_storage_owner_ = other.leased_storage_owner_;
        chunk_source_         = other.chunk_source_;
        chunk_device_ = other.chunk_device_;
        if (leased_entry_) {
            leased_entry_->in_use_count.fetch_add(1);
            // llama.cpp-2wv5: a copy takes its own lease, so it -- not whoever
            // acquired the original -- is the site that owes the release.  Like
            // the fetch_add above, this writes to the CACHE ENTRY rather than to
            // handle state and takes no cache lock, so it is equally safe inside
            // this critical section.
            leased_entry_->debug_last_lease_site = "mem_handle/copy-ctor";
        }
    }
    // llama.cpp-dyhdl: independently acquire a chunk lease for the copy.  This
    // calls into unified_cache, so it runs with both handle locks dropped.
    bump_chunk_lease_for_copy(chunk_source_, chunk_device_, cached_.ptr, host_chunk_handle_, vram_chunk_idx_);
    if (chunk_source_ == 1 && host_chunk_handle_ == UINT64_MAX) {
        chunk_source_ = 0;
        chunk_device_ = -1;
    }
    if (chunk_source_ == 2 && vram_chunk_idx_ < 0) {
        chunk_source_ = 0;
        chunk_device_ = -1;
    }
}

mem_handle::mem_handle(mem_handle && other) noexcept :
    kind_(other.kind_),
    device_(other.device_),
    key_(other.key_),
    zone_id_(other.zone_id_),
    offset_(other.offset_),
    size_(other.size_),
    arena_gen_(other.arena_gen_),
    owned_alloc_(std::move(other.owned_alloc_)),
    debug_owner_tag_(other.debug_owner_tag_) {
    // Transfer ownership — no refcount change.  `other` is left with no leases
    // so its dtor does not release ours.  Held under `other`'s lock so a
    // concurrent resolve() on it cannot observe or write a half-moved state.
    mem_handle_lock_guard g(other.lock_);
    gen_                 = other.gen_;
    cached_              = other.cached_;
    const lease_state st = other.take_lease_state_locked();
    leased_entry_         = st.entry;
    leased_storage_owner_ = std::move(st.storage_owner);
    chunk_source_         = st.chunk_source;
    host_chunk_handle_   = st.host_chunk_handle;
    vram_chunk_idx_      = st.vram_chunk_idx;
    chunk_device_        = st.chunk_device;
}

mem_handle & mem_handle::operator=(const mem_handle & other) {
    if (this == &other) {
        return *this;
    }

    // 1. Snapshot the source under ITS lock and take our own entry lease.
    resolved_ptr          new_cached;
    uint64_t              new_gen          = 0;
    unified_cache_entry * new_entry        = nullptr;
    uint8_t               new_chunk_source = 0;
    int                   new_chunk_device = -1;
    std::shared_ptr<void> new_storage_owner;
    {
        mem_handle_lock_guard g(other.lock_);
        new_gen          = other.gen_;
        new_cached       = other.cached_;
        new_entry        = other.leased_entry_;
        new_chunk_source = other.chunk_source_;
        new_chunk_device  = other.chunk_device_;
        new_storage_owner = other.leased_storage_owner_;
        if (new_entry) {
            new_entry->in_use_count.fetch_add(1);
            // llama.cpp-2wv5: see the copy ctor -- the copy holds the lease it
            // just took, so it is the site that owes the release.
            new_entry->debug_last_lease_site = "mem_handle/copy-assign";
        }
    }

    // 2. Acquire our own chunk lease with no handle lock held.
    uint64_t new_host_chunk_handle = UINT64_MAX;
    int32_t  new_vram_chunk_idx    = -1;
    bump_chunk_lease_for_copy(new_chunk_source, new_chunk_device, new_cached.ptr, new_host_chunk_handle,
                              new_vram_chunk_idx);
    if (new_chunk_source == 1 && new_host_chunk_handle == UINT64_MAX) {
        new_chunk_source = 0;
        new_chunk_device = -1;
    }
    if (new_chunk_source == 2 && new_vram_chunk_idx < 0) {
        new_chunk_source = 0;
        new_chunk_device = -1;
    }

    // 3. Publish, detaching the leases we are dropping.
    lease_state stale;
    {
        mem_handle_lock_guard g(lock_);
        stale            = take_lease_state_locked();
        kind_            = other.kind_;
        device_          = other.device_;
        key_             = other.key_;
        zone_id_         = other.zone_id_;
        offset_          = other.offset_;
        size_            = other.size_;
        arena_gen_       = other.arena_gen_;
        owned_alloc_     = other.owned_alloc_;
        debug_owner_tag_ = other.debug_owner_tag_;
        gen_             = new_gen;
        cached_          = new_cached;
        lease_state fresh;
        fresh.entry             = new_entry;
        fresh.chunk_source      = new_chunk_source;
        fresh.host_chunk_handle = new_host_chunk_handle;
        fresh.vram_chunk_idx    = new_vram_chunk_idx;
        fresh.chunk_device      = new_chunk_device;
        fresh.storage_owner     = std::move(new_storage_owner);
        store_lease_state_locked(fresh);
    }

    // 4. Release the old leases outside the lock.
    release_lease_state(stale);
    return *this;
}

mem_handle & mem_handle::operator=(mem_handle && other) noexcept {
    if (this == &other) {
        return *this;
    }

    resolved_ptr new_cached;
    uint64_t     new_gen = 0;
    lease_state  fresh;
    {
        mem_handle_lock_guard g(other.lock_);
        new_gen    = other.gen_;
        new_cached = other.cached_;
        fresh      = other.take_lease_state_locked();
    }

    lease_state stale;
    {
        mem_handle_lock_guard g(lock_);
        stale            = take_lease_state_locked();
        kind_            = other.kind_;
        device_          = other.device_;
        key_             = other.key_;
        zone_id_         = other.zone_id_;
        offset_          = other.offset_;
        size_            = other.size_;
        arena_gen_       = other.arena_gen_;
        owned_alloc_     = std::move(other.owned_alloc_);
        debug_owner_tag_ = other.debug_owner_tag_;
        gen_             = new_gen;
        cached_          = new_cached;
        store_lease_state_locked(fresh);
    }

    release_lease_state(stale);
    return *this;
}

// === resolve_arena ===
// Resolve an arena handle by querying the arena base pointer from unified_cache,
// then adding the zone start + offset.  Returns nullptr if the arena has been
// recreated (generation mismatch).

resolved_ptr mem_handle::resolve_arena() const {
    // Device arena: query unified_cache for arena methods.
    if (!valid_cache_device_id(device_)) {
        return {};
    }

    unified_cache * cache = get_existing_unified_cache_for_device(device_);
    if (!cache) {
        return {};
    }

    if (!cache->arena_active()) {
        return {};
    }

    // Check generation: if the arena was destroyed and recreated, our handle
    // is stale.
    uint64_t current_gen = cache->arena_generation();
    if (arena_gen_ != current_gen) {
        return {};
    }

    // Resolve: zone_alloc returned an offset within the arena, but our offset_
    // is the raw arena offset (base-relative).  Use offset_to_ptr directly.
    void * ptr = cache->offset_to_ptr(offset_);
    if (!ptr) {
        return {};
    }

    // Cache the resolved pointer.  Arena handles are always on-device with
    // AOS layout (arena zones hold raw allocations, not cache-managed weights).
    // Published under the lock; the cache queries above ran without it.
    const resolved_ptr resolved = { ptr, GGML_LAYOUT_AOS, true };
    {
        mem_handle_lock_guard g(lock_);
        cached_ = resolved;
        gen_    = arena_gen_;
    }
    return resolved;
}

// === layer_weight_handles ===

bool layer_weight_handles::resolve_all(layer_weight_pointers & out) const {
    auto r_attn_norm = attn_norm.resolve();
    auto r_q_proj    = q_proj.resolve();
    auto r_k_proj    = k_proj.resolve();
    auto r_v_proj    = v_proj.resolve();
    auto r_o_proj    = o_proj.resolve();
    auto r_ffn_norm  = ffn_norm.resolve();
    auto r_gate_proj = gate_proj.resolve();
    auto r_up_proj   = up_proj.resolve();
    auto r_down_proj = down_proj.resolve();

    if (!r_attn_norm || !r_q_proj || !r_k_proj || !r_v_proj || !r_o_proj || !r_ffn_norm || !r_gate_proj || !r_up_proj ||
        !r_down_proj) {
        return false;
    }

    out.attn_norm = r_attn_norm.ptr;
    out.q_proj    = r_q_proj.ptr;
    out.k_proj    = r_k_proj.ptr;
    out.v_proj    = r_v_proj.ptr;
    out.o_proj    = r_o_proj.ptr;
    out.ffn_norm  = r_ffn_norm.ptr;
    out.gate_proj = r_gate_proj.ptr;
    out.up_proj   = r_up_proj.ptr;
    out.down_proj = r_down_proj.ptr;

    // Optional fused weights — resolve if handle is valid
    auto r_qkv           = attn_qkv_proj.resolve();
    auto r_gate_up       = ffn_gate_up_proj.resolve();
    out.attn_qkv_proj    = r_qkv ? r_qkv.ptr : nullptr;
    out.ffn_gate_up_proj = r_gate_up ? r_gate_up.ptr : nullptr;

    return true;
}

layer_weight_handles layer_weight_handles::from_weight_set(const layer_weight_set & ws, int device) {
    layer_weight_handles h;
    h.attn_norm = mem_handle::from_cache_id(ws.attn_norm, device);
    h.q_proj    = mem_handle::from_cache_id(ws.q_proj, device);
    h.k_proj    = mem_handle::from_cache_id(ws.k_proj, device);
    h.v_proj    = mem_handle::from_cache_id(ws.v_proj, device);
    h.o_proj    = mem_handle::from_cache_id(ws.o_proj, device);
    h.ffn_norm  = mem_handle::from_cache_id(ws.ffn_norm, device);
    h.gate_proj = mem_handle::from_cache_id(ws.gate_proj, device);
    h.up_proj   = mem_handle::from_cache_id(ws.up_proj, device);
    h.down_proj = mem_handle::from_cache_id(ws.down_proj, device);

    // Optional fused weights
    if (ws.attn_qkv_proj.valid) {
        h.attn_qkv_proj = mem_handle::from_cache_id(ws.attn_qkv_proj, device);
    }
    if (ws.ffn_gate_up_proj.valid) {
        h.ffn_gate_up_proj = mem_handle::from_cache_id(ws.ffn_gate_up_proj, device);
    }

    return h;
}

// === build_layer_handles ===

bool build_layer_handles(int device, int layer_id, layer_weight_handles & out) {
    unified_cache * cache = get_unified_cache_for_device(device);
    if (!cache) {
        return false;
    }

    layer_weight_set ws;
    if (!cache->get_layer_weight_set(layer_id, ws)) {
        return false;
    }

    out = layer_weight_handles::from_weight_set(ws, device);
    return true;
}

void drain_retained_handles(bool wait_all) {
    if (!wait_all) {
        // Retained handles are released by the background drain worker.  Avoid
        // get_info(command_execution_status) polling on inference threads:
        // that Level Zero query can block on in-flight events.
        return;
    }

    auto &                       state = *g_retained_handles_state;
    std::unique_lock<std::mutex> lock(state.mutex);
    state.cv.wait(lock, [&state] { return state.queue.empty() && state.active == 0; });
}

size_t graph_retained_handle_count() {
    auto &                      state = *g_retained_handles_state;
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.graph_unwaitable.size();
}

void release_graph_retained_handles() {
    auto &                      state = *g_retained_handles_state;
    std::lock_guard<std::mutex> lock(state.mutex);
    const size_t                n = state.graph_unwaitable.size();
    state.graph_unwaitable.clear();
    GGML_SYCL_DEBUG("[MEM-HANDLE] released %zu command-graph retained leases\n", n);
}

void retain_handles_until_event(std::vector<mem_handle> handles, sycl::event event) {
    if (handles.empty()) {
        return;
    }

    if (graph_lifetime_retention_active()) {
        retain_handles_for_current_graph(std::move(handles));
        return;
    }

    std::call_once(g_retained_drain_worker_once, start_retained_handle_drain_worker);

    {
        auto &                      state = *g_retained_handles_state;
        std::lock_guard<std::mutex> lock(state.mutex);
        state.queue.push_back({ std::move(handles), std::move(event) });
    }
    g_retained_handles_state->cv.notify_one();
}

void set_graph_retained_handle_sink(std::vector<mem_handle> * sink) {
    g_graph_retained_handle_sink = sink;
}

}  // namespace ggml_sycl
