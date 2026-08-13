// Smart handle implementation for SYCL unified memory manager.
// See mem-handle.hpp for design and docs/smart-handle-design.md for architecture.

#include "mem-handle.hpp"

#include <chrono>

#include "common.hpp"
#include "moe-graph-retention.hpp"
#include "pinned-pool.hpp"    // pinned_chunk_pool chunk-lease API (dyhdl)
#include "unified-cache.hpp"  // get_unified_cache_for_device, unified_cache

#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <iterator>
#include <mutex>
#include <new>
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
    size_t                             active     = 0;
    size_t                             publishers = 0;
};

// The detached drain worker can still be waiting while process shutdown tears down
// static objects.  Keep the synchronization state alive until process exit.
retained_handle_state *                g_retained_handles_state = new retained_handle_state();
std::once_flag                         g_retained_drain_worker_once;
#ifdef GGML_SYCL_RETAINED_PUBLICATION_TESTING
std::atomic<bool> g_fail_next_retained_handle_publication{ false };
#endif
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
        return a.stable_identity_equal(b);
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
    h.gen_            = 0;  // Stale — first resolve() will query the cache
    h.size_           = key.id.nbytes;
    h.backing_extent_ = key.id.nbytes;
    h.cached_         = {};
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
    // A leased entry, not the caller, is authoritative for owning device.
    h.device_ = entry ? entry->owner_device : device;
    h.key_    = key;
    h.gen_    = cache_generation();  // Fresh — no slow-path re-query
    const size_t snapshot_extent = entry ? entry->size : key.id.nbytes;
    h.cached_ = { ptr, snapshot_extent, layout, on_device };
    if (has_ready_event) {
        h.cached_.has_ready_event = true;
        h.cached_.ready_event     = ready_event;
    }
    // A cache entry is the authority for both backing and capability. Refuse
    // attempts to attach its identity to an arbitrary pointer/layout/device
    // tuple. A live lease prevents cache mutation while this snapshot is read.
    if (entry && (ptr != entry->device_ptr || layout != entry->layout ||
                  on_device != (entry->location == cache_location::DEVICE) ||
                  !entry->has_retention_identity() ||
                  (entry->location == cache_location::DEVICE && entry->owner_device < 0))) {
        entry->in_use_count.fetch_sub(1);
        return {};
    }
    h.leased_entry_         = entry;  // ownership of the refcount bump transferred
    h.leased_storage_owner_ = std::move(storage_owner);
    if (entry) {
        h.canonical_allocation_id_ = entry->allocation_identity();
        h.canonical_generation_    = entry->replacement_identity();
        h.canonical_extent_        = entry->size;
        h.offset_                  = 0;
        h.size_                    = entry->size;
        h.backing_extent_          = entry->size;
    } else {
        h.offset_         = 0;
        h.size_           = snapshot_extent;
        h.backing_extent_ = snapshot_extent;
    }

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

mem_handle mem_handle::from_direct(void * ptr,
                                   ggml_layout_mode layout,
                                   bool on_device,
                                   int device,
                                   size_t extent) {
    mem_handle h;
    h.kind_           = mem_handle_kind::DIRECT;
    h.device_         = device;
    h.key_            = {};
    h.gen_            = 0;
    h.offset_         = 0;
    h.size_           = extent;
    h.backing_extent_ = extent;
    h.cached_         = { ptr, extent, layout, on_device };
    return h;
}

mem_handle mem_handle::from_arena_zone(int      zone_id,
                                       size_t   offset,
                                       size_t   size,
                                       int      device,
                                       uint64_t generation,
                                       uint64_t allocation_id,
                                       size_t   allocation_extent) {
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
    // `offset` is a physical arena coordinate. Canonical view offsets are
    // allocation-relative and therefore start at zero for the minted root.
    h.backing_offset_ = offset;
    h.offset_         = 0;
    h.size_           = size;
    h.backing_extent_ = size;
    h.arena_gen_              = generation;
    h.canonical_allocation_id_ = allocation_id;
    h.canonical_generation_    = generation;
    h.canonical_extent_ = allocation_extent;
    h.gen_                = 0;  // Force first resolve
    h.cached_    = {};
    return h;
}

// Compatibility/test bridge: wrap a raw pointer in a handle that refcounts the
// owning arena chunk for the lifetime of the returned mem_handle. Production
// allocation paths should keep the mem_handle they received when allocating.
mem_handle mem_handle::from_chunk_ptr(void * ptr, int device, ggml_layout_mode layout, bool on_device) {
    mem_handle h;
    h.device_ = device;
    h.cached_ = { ptr, 0, layout, on_device };
    h.gen_    = 0;

    if (ptr == nullptr) {
        h.kind_ = mem_handle_kind::DIRECT;
        return h;
    }

    // Only the runtime allocation registry may mint a range for this legacy
    // bridge. A containing arena chunk is physical lifetime ownership, not
    // authority over unrelated suballocations in that chunk.
    alloc_handle allocation{};
    if (unified_lookup_runtime_allocation(ptr, &allocation, nullptr) && allocation.ptr && allocation.size != 0) {
        const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        const uintptr_t base = reinterpret_cast<uintptr_t>(allocation.ptr);
        if (addr >= base && addr - base < allocation.size) {
            const size_t allocation_offset = static_cast<size_t>(addr - base);
            h.offset_                   = allocation_offset;
            h.size_                     = allocation.size - allocation_offset;
            h.backing_extent_           = h.size_;
            h.cached_.extent            = h.size_;
            h.canonical_allocation_id_  = allocation.alloc_id;
            h.canonical_generation_     = allocation.epoch_id ? allocation.epoch_id : 1;
            h.canonical_extent_         = allocation.size;
        }
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
    mem_handle h         = from_direct(handle.ptr, layout, on_device,
                                       on_device ? handle.device : HOST_DEVICE, handle.size);
    h.offset_                   = 0;
    h.size_                     = handle.size;
    h.backing_extent_            = handle.size;
    h.canonical_allocation_id_  = handle.alloc_id;
    h.canonical_generation_     = handle.epoch_id ? handle.epoch_id : 1;
    h.canonical_extent_         = handle.size;
    h.owned_alloc_ = std::shared_ptr<alloc_handle>(new alloc_handle(std::move(handle)), release_owned_alloc_handle);
    return h;
}

mem_handle mem_handle::slice(size_t byte_offset, size_t byte_size) const {
    // Raw external DIRECT pointers have no retained owner and no trustworthy
    // extent. Slicing them would manufacture a capability rather than derive one.
    if (!owned_alloc_ && kind_ != mem_handle_kind::WEIGHT && !is_arena() &&
        kind_ != mem_handle_kind::CHUNK_LEASE) {
        return {};
    }

    // Unresolved key-only weights learn their logical storage range through
    // the normal generation-checked resolver before deriving a view.
    if (kind_ == mem_handle_kind::WEIGHT && size() == 0 && !resolve()) {
        return {};
    }

    size_t parent_size = 0;
    size_t parent_offset = 0;
    size_t parent_slice_offset = 0;
    {
        mem_handle_lock_guard g(lock_);
        parent_size         = size_;
        parent_offset       = offset_;
        parent_slice_offset = slice_offset_;
    }

    // Unknown extent is not authority. In particular a compatibility chunk
    // root may not let its first slice manufacture an allocation bound.
    if (parent_size == 0 || byte_offset > parent_size || byte_size > parent_size - byte_offset ||
        byte_offset > SIZE_MAX - parent_slice_offset || byte_offset > SIZE_MAX - parent_offset) {
        return {};
    }

    mem_handle h = *this;
    {
        mem_handle_lock_guard g(h.lock_);
        h.slice_offset_ = parent_slice_offset + byte_offset;
        h.offset_       = parent_offset + byte_offset;
        h.size_         = byte_size;
        h.is_slice_     = true;
    }
    return h;
}

resolved_ptr mem_handle::resolved_view_locked() const {
    resolved_ptr view = cached_;
    if (!view.ptr) {
        return view;
    }
    if (!is_slice_) {
        view.extent = size_;
        return view;
    }
    if ((backing_extent_ != 0 &&
         (slice_offset_ > backing_extent_ || size_ > backing_extent_ - slice_offset_)) ||
        slice_offset_ > UINTPTR_MAX - reinterpret_cast<uintptr_t>(view.ptr)) {
        return {};
    }
    view.ptr    = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(view.ptr) + slice_offset_);
    view.extent = size_;
    return view;
}

#ifdef GGML_SYCL_RETENTION_IDENTITY_TESTING
void mem_handle::set_canonical_identity_for_test(uint64_t allocation_id, uint64_t generation, size_t extent,
                                                 size_t offset, size_t size) noexcept {
    mem_handle_lock_guard guard(lock_);
    canonical_allocation_id_ = allocation_id;
    canonical_generation_ = generation ? generation : 1;
    canonical_extent_ = extent;
    offset_ = offset;
    size_ = size;
}
#endif

std::optional<moe::retained_allocation_owner> moe::canonical_allocation_integration::retain(
    const mem_handle & handle) noexcept {
    try {
        uint64_t allocation_id = 0;
        uint64_t generation    = 0;
        int      device        = mem_handle::HOST_DEVICE;
        size_t   extent        = 0;
        {
            mem_handle_lock_guard guard(handle.lock_);
            if (!handle.cached_.ptr) {
                return std::nullopt;
            }
            device        = handle.owned_alloc_ ? handle.owned_alloc_->device : handle.device_;
            allocation_id = handle.canonical_allocation_id_;
            generation    = handle.canonical_generation_;
            extent        = handle.canonical_extent_;
            if (handle.kind_ == mem_handle_kind::WEIGHT && allocation_id == 0) {
                // Cache-entry minted IDs/replacement generations are not yet
                // propagated by unified-cache. Never substitute a key hash.
                return std::nullopt;
            }
            if (!handle.owned_alloc_ && !handle.is_arena() && handle.kind_ != mem_handle_kind::WEIGHT) {
                // Non-owning DIRECT and compatibility CHUNK_LEASE handles do
                // not carry an exact allocation identity/range.
                return std::nullopt;
            }
        }
        if (allocation_id == 0 || generation == 0 || device < 0 ||
            device >= static_cast<int>(execution::max_devices) || extent == 0) {
            return std::nullopt;
        }
        auto retained_handle = std::make_shared<mem_handle>(handle);
        std::shared_ptr<const void> owner = retained_handle;
        return retained_allocation_owner(allocation_id, generation, device, extent, std::move(owner));
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<moe::mmid_batch_binding> moe::canonical_allocation_integration::bind(
    const mem_handle & handle, uint64_t layout_id, uint32_t occurrence) noexcept {
    if (layout_id == 0) {
        return std::nullopt;
    }
    auto owner = retain(handle);
    if (!owner) {
        return std::nullopt;
    }
    size_t byte_offset = 0;
    size_t byte_size   = 0;
    {
        mem_handle_lock_guard guard(handle.lock_);
        byte_offset = handle.offset_;
        byte_size   = handle.size_;
    }
    if (byte_size == 0 || byte_offset > owner->extent() || byte_size > owner->extent() - byte_offset) {
        return std::nullopt;
    }
    return mmid_batch_binding{ { owner->allocation_id(), owner->generation(), layout_id, owner->device(), byte_offset,
                                 byte_size, occurrence },
                               std::move(*owner) };
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
            return resolved_view_locked();
        }

        // Arena handles deliberately have no handle-local fast path. The arena
        // owner's current generation is authoritative and must be checked on
        // every resolve, including after this handle cached a pointer.
        if (!(kind_ >= mem_handle_kind::ARENA_RUNTIME && kind_ <= mem_handle_kind::ARENA_ONEDNN)) {
            // WEIGHT handle: compare cached generation against global.
            if (gen_ == cache_generation() && cached_.ptr != nullptr) {
                return resolved_view_locked();
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

    resolved_ptr resolved = { result.ptr, result.byte_size, result.layout, result.on_device };
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
    resolved_ptr view;
    {
        mem_handle_lock_guard g(lock_);
        stale = take_lease_state_locked();
        store_lease_state_locked(fresh);
        cached_                   = resolved;
        canonical_allocation_id_ = result.allocation_id;
        canonical_generation_    = result.replacement_generation;
        canonical_extent_        = result.allocation_extent;
        backing_extent_          = result.byte_size;
        if (is_slice_) {
            // Preserve the derived range across cache generation changes. The
            // cache resolver supplies the current logical-storage base/range;
            // the view is valid only when it still fits that replacement.
            if (slice_offset_ <= result.byte_size && size_ <= result.byte_size - slice_offset_ &&
                slice_offset_ <= SIZE_MAX - result.byte_offset) {
                offset_ = result.byte_offset + slice_offset_;
            } else {
                cached_ = {};
            }
        } else {
            offset_ = result.byte_offset;
            size_   = result.byte_size;
        }
        gen_ = cache_generation();
        view = resolved_view_locked();
    }
    release_lease_state(stale);

    return view;
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
        // llama.cpp-fzem: record the release BEFORE the decrement below, not
        // after -- once in_use_count drops the entry may be evicted, and the
        // comment on the next line is explicit that this call must never
        // dereference `entry` again past that point. No caller-site context
        // is available here (this is the single release path shared by the
        // destructor, move-assign detach, and copy-assign's stale release),
        // so every event is tagged with this function's own name; the
        // acquire-side tags (recorded at the copy ctor/assign call sites)
        // carry the specific site.
        state.entry->record_lease_event(false, "mem_handle/release_lease_state");
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
        mem_handle_lock_guard g(lock_);
        if (is_slice_) {
            h = mem_handle_hash_combine(h, std::hash<size_t>()(offset_));
            h = mem_handle_hash_combine(h, std::hash<size_t>()(size_));
        }
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
        if (is_slice_) {
            h = mem_handle_hash_combine(h, std::hash<size_t>()(offset_));
            h = mem_handle_hash_combine(h, std::hash<size_t>()(size_));
        }
    } else if (is_arena()) {
        h = mem_handle_hash_combine(h, std::hash<uint64_t>()(canonical_allocation_id_));
        h = mem_handle_hash_combine(h, std::hash<uint64_t>()(canonical_generation_));
        h = mem_handle_hash_combine(h, std::hash<size_t>()(canonical_extent_));
        h = mem_handle_hash_combine(h, std::hash<int>()(zone_id_));
        h = mem_handle_hash_combine(h, std::hash<size_t>()(offset_));
        h = mem_handle_hash_combine(h, std::hash<size_t>()(size_));
        h = mem_handle_hash_combine(h, std::hash<uint64_t>()(arena_gen_));
    } else if (kind_ == mem_handle_kind::CHUNK_LEASE) {
        h = mem_handle_hash_combine(h, std::hash<int>()(chunk_device_));
        h = mem_handle_hash_combine(h, std::hash<uint8_t>()(chunk_source_));
        h = mem_handle_hash_combine(h, std::hash<uint64_t>()(host_chunk_handle_));
        h = mem_handle_hash_combine(h, std::hash<int32_t>()(vram_chunk_idx_));
        uintptr_t absolute = reinterpret_cast<uintptr_t>(cached_.ptr);
        if (slice_offset_ <= UINTPTR_MAX - absolute) {
            absolute += slice_offset_;
        } else {
            absolute = 0;
        }
        h = mem_handle_hash_combine(h, std::hash<uintptr_t>()(absolute));
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
        if (!(key_ == other.key_)) {
            return false;
        }
        struct weight_view_identity {
            bool   sliced;
            size_t offset;
            size_t size;
        };
        auto snapshot_weight = [](const mem_handle & h) {
            mem_handle_lock_guard g(h.lock_);
            return weight_view_identity{ h.is_slice_, h.offset_, h.size_ };
        };
        const auto self   = snapshot_weight(*this);
        const auto theirs = snapshot_weight(other);
        return (!self.sliced && !theirs.sliced) ||
               (self.sliced && theirs.sliced && self.offset == theirs.offset && self.size == theirs.size);
    }

    if (is_arena()) {
        return canonical_allocation_id_ == other.canonical_allocation_id_ &&
               canonical_generation_ == other.canonical_generation_ && canonical_extent_ == other.canonical_extent_ &&
               zone_id_ == other.zone_id_ && offset_ == other.offset_ && size_ == other.size_ &&
               arena_gen_ == other.arena_gen_;
    }

    // The remaining branches read mutable state on BOTH handles.  Snapshot each
    // side under its own lock rather than holding two locks at once — there is
    // no lock order between two arbitrary handles.
    struct identity_snapshot {
        uintptr_t absolute_ptr         = 0;
        size_t    extent               = 0;
        uint8_t   chunk_source         = 0;
        uint64_t  host_chunk_handle    = UINT64_MAX;
        int32_t   vram_chunk_idx       = -1;
        int       chunk_device         = -1;
    };

    auto snapshot = [](const mem_handle & h) {
        mem_handle_lock_guard g(h.lock_);
        identity_snapshot     s;
        s.absolute_ptr = reinterpret_cast<uintptr_t>(h.cached_.ptr);
        if (h.slice_offset_ <= UINTPTR_MAX - s.absolute_ptr) {
            s.absolute_ptr += h.slice_offset_;
        } else {
            s.absolute_ptr = 0;
        }
        s.extent            = h.size_;
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
               self.absolute_ptr == theirs.absolute_ptr && self.extent == theirs.extent;
    }

    if (owned_alloc_ || other.owned_alloc_) {
        return owned_alloc_ && other.owned_alloc_ && owned_alloc_->alloc_id == other.owned_alloc_->alloc_id &&
               owned_alloc_->device == other.owned_alloc_->device && owned_alloc_->tier == other.owned_alloc_->tier &&
               owned_alloc_->role == other.owned_alloc_->role &&
               owned_alloc_->category == other.owned_alloc_->category && owned_alloc_->size == other.owned_alloc_->size &&
               offset_ == other.offset_ && size_ == other.size_;
    }

    return self.absolute_ptr == theirs.absolute_ptr && self.extent == theirs.extent;
}

bool mem_handle::has_stable_owner_identity() const {
    return is_weight() || is_arena() || kind_ == mem_handle_kind::CHUNK_LEASE || owned_alloc_ != nullptr;
}

void mem_handle::set_debug_owner(const char * owner_tag) {
    debug_owner_tag_ = owner_tag ? owner_tag : "";
}

void mem_handle::tag_persistent_lease_site(const char * site) const {
    if (!site) {
        return;
    }
    // Mirrors the copy-ctor/copy-assign write of debug_last_lease_site: no
    // cache lock required, this is a plain pointer write to a debug-only
    // field on the cache entry (llama.cpp-2wv5's precedent).
    mem_handle_lock_guard g(lock_);
    if (leased_entry_) {
        leased_entry_->debug_last_lease_site = site;
    }
}

mem_handle_debug_info mem_handle::debug_info() const {
    mem_handle_debug_info info;
    info.kind                = kind_;
    info.device              = device_;
    info.zone_id             = zone_id_;
    info.generation          = arena_gen_;
    info.has_stable_identity = has_stable_owner_identity();
    info.owner_tag           = debug_owner_tag_ ? debug_owner_tag_ : "";

    // One critical section for every mutable field: valid() and
    // stable_identity_hash() each take the lock, and the spinlock is not
    // recursive, so use the *_locked form here.
    mem_handle_lock_guard g(lock_);
    info.valid                   = cached_.ptr != nullptr;
    info.offset                  = offset_;
    info.size                    = size_;
    info.canonical_allocation_id = canonical_allocation_id_;
    info.canonical_generation    = canonical_generation_;
    info.canonical_extent        = canonical_extent_;
    info.has_ready_event         = cached_.has_ready_event;
    info.stable_identity_hash    = stable_identity_hash_locked();
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
mem_handle::mem_handle(const mem_handle & other) {
    {
        // Snapshot identity, range, owner, and resolved state in one critical
        // section.  A canonical capability is valid only as that complete tuple;
        // publishing an owner with zero/stale identity would create a forged view.
        mem_handle_lock_guard g(other.lock_);
        kind_                    = other.kind_;
        device_                  = other.device_;
        key_                     = other.key_;
        zone_id_                 = other.zone_id_;
        offset_                  = other.offset_;
        size_                    = other.size_;
        backing_extent_          = other.backing_extent_;
        backing_offset_          = other.backing_offset_;
        slice_offset_            = other.slice_offset_;
        is_slice_                = other.is_slice_;
        arena_gen_               = other.arena_gen_;
        canonical_allocation_id_ = other.canonical_allocation_id_;
        canonical_generation_    = other.canonical_generation_;
        canonical_extent_        = other.canonical_extent_;
        owned_alloc_             = other.owned_alloc_;
        debug_owner_tag_         = other.debug_owner_tag_;
        gen_                     = other.gen_;
        cached_                  = other.cached_;
        leased_entry_            = other.leased_entry_;
        leased_storage_owner_    = other.leased_storage_owner_;
        chunk_source_            = other.chunk_source_;
        chunk_device_            = other.chunk_device_;
        if (leased_entry_) {
            leased_entry_->in_use_count.fetch_add(1);
            // llama.cpp-2wv5: a copy takes its own lease, so it -- not whoever
            // acquired the original -- is the site that owes the release.  Like
            // the fetch_add above, this writes to the CACHE ENTRY rather than to
            // handle state and takes no cache lock, so it is equally safe inside
            // this critical section.
            leased_entry_->debug_last_lease_site = "mem_handle/copy-ctor";
            leased_entry_->record_lease_event(true, "mem_handle/copy-ctor");
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

mem_handle::mem_handle(mem_handle && other) noexcept {
    // Transfer the complete capability under one lock, then leave `other` as a
    // normal invalid handle.  In particular it must not retain a canonical ID
    // or cached pointer after its allocation owner has moved away.
    mem_handle_lock_guard g(other.lock_);
    kind_                    = other.kind_;
    device_                  = other.device_;
    key_                     = other.key_;
    zone_id_                 = other.zone_id_;
    offset_                  = other.offset_;
    size_                    = other.size_;
    backing_extent_          = other.backing_extent_;
    backing_offset_          = other.backing_offset_;
    slice_offset_            = other.slice_offset_;
    is_slice_                = other.is_slice_;
    arena_gen_               = other.arena_gen_;
    canonical_allocation_id_ = other.canonical_allocation_id_;
    canonical_generation_    = other.canonical_generation_;
    canonical_extent_        = other.canonical_extent_;
    owned_alloc_             = std::move(other.owned_alloc_);
    debug_owner_tag_         = other.debug_owner_tag_;
    gen_                     = other.gen_;
    cached_                  = std::move(other.cached_);
    const lease_state st     = other.take_lease_state_locked();
    leased_entry_            = st.entry;
    leased_storage_owner_    = std::move(st.storage_owner);
    chunk_source_            = st.chunk_source;
    host_chunk_handle_       = st.host_chunk_handle;
    vram_chunk_idx_          = st.vram_chunk_idx;
    chunk_device_            = st.chunk_device;

    other.kind_                    = mem_handle_kind::DIRECT;
    other.device_                  = HOST_DEVICE;
    other.key_                     = {};
    other.zone_id_                 = 0;
    other.offset_                  = 0;
    other.size_                    = 0;
    other.backing_extent_          = 0;
    other.backing_offset_          = 0;
    other.slice_offset_            = 0;
    other.is_slice_                = false;
    other.arena_gen_               = 0;
    other.canonical_allocation_id_ = 0;
    other.canonical_generation_    = 0;
    other.canonical_extent_        = 0;
    other.debug_owner_tag_         = "";
    other.gen_                     = 0;
    other.cached_                  = {};
}

mem_handle & mem_handle::operator=(const mem_handle & other) {
    if (this == &other) {
        return *this;
    }

    // 1. Snapshot the source under ITS lock and take our own entry lease.
    resolved_ptr          new_cached;
    uint64_t              new_gen                    = 0;
    mem_handle_kind       new_kind                   = mem_handle_kind::DIRECT;
    int                   new_device                 = HOST_DEVICE;
    unified_cache_key     new_key                    = {};
    int                   new_zone_id                = 0;
    size_t                new_offset                 = 0;
    size_t                new_size                   = 0;
    size_t                new_backing_extent         = 0;
    size_t                new_backing_offset         = 0;
    size_t                new_slice_offset           = 0;
    bool                  new_is_slice               = false;
    uint64_t              new_arena_gen              = 0;
    uint64_t              new_canonical_allocation_id = 0;
    uint64_t              new_canonical_generation   = 0;
    size_t                new_canonical_extent       = 0;
    std::shared_ptr<alloc_handle> new_owned_alloc;
    const char *          new_debug_owner_tag        = "";
    unified_cache_entry * new_entry                  = nullptr;
    uint8_t               new_chunk_source           = 0;
    int                   new_chunk_device           = -1;
    std::shared_ptr<void> new_storage_owner;
    {
        mem_handle_lock_guard g(other.lock_);
        new_kind                    = other.kind_;
        new_device                  = other.device_;
        new_key                     = other.key_;
        new_zone_id                 = other.zone_id_;
        new_offset                  = other.offset_;
        new_size                    = other.size_;
        new_backing_extent          = other.backing_extent_;
        new_backing_offset          = other.backing_offset_;
        new_slice_offset            = other.slice_offset_;
        new_is_slice                = other.is_slice_;
        new_arena_gen               = other.arena_gen_;
        new_canonical_allocation_id = other.canonical_allocation_id_;
        new_canonical_generation    = other.canonical_generation_;
        new_canonical_extent        = other.canonical_extent_;
        new_owned_alloc             = other.owned_alloc_;
        new_debug_owner_tag         = other.debug_owner_tag_;
        new_gen                     = other.gen_;
        new_cached                  = other.cached_;
        new_entry                   = other.leased_entry_;
        new_chunk_source            = other.chunk_source_;
        new_chunk_device            = other.chunk_device_;
        new_storage_owner           = other.leased_storage_owner_;
        if (new_entry) {
            new_entry->in_use_count.fetch_add(1);
            // llama.cpp-2wv5: see the copy ctor -- the copy holds the lease it
            // just took, so it is the site that owes the release.
            new_entry->debug_last_lease_site = "mem_handle/copy-assign";
            new_entry->record_lease_event(true, "mem_handle/copy-assign");
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

    // 3. Publish, detaching every releasable owner we are dropping. Moving
    // owned_alloc_ out first is essential: its deleter calls unified_free(),
    // which may re-enter handle/cache code and must never run under this leaf lock.
    lease_state                   stale;
    std::shared_ptr<alloc_handle> stale_owned_alloc;
    {
        mem_handle_lock_guard g(lock_);
        stale                    = take_lease_state_locked();
        stale_owned_alloc        = std::move(owned_alloc_);
        kind_                    = new_kind;
        device_                  = new_device;
        key_                     = new_key;
        zone_id_                 = new_zone_id;
        offset_                  = new_offset;
        size_                    = new_size;
        backing_extent_          = new_backing_extent;
        backing_offset_          = new_backing_offset;
        slice_offset_            = new_slice_offset;
        is_slice_                = new_is_slice;
        arena_gen_               = new_arena_gen;
        canonical_allocation_id_ = new_canonical_allocation_id;
        canonical_generation_    = new_canonical_generation;
        canonical_extent_        = new_canonical_extent;
        owned_alloc_             = std::move(new_owned_alloc);
        debug_owner_tag_         = new_debug_owner_tag;
        gen_                     = new_gen;
        cached_                  = new_cached;
        lease_state fresh;
        fresh.entry             = new_entry;
        fresh.chunk_source      = new_chunk_source;
        fresh.host_chunk_handle = new_host_chunk_handle;
        fresh.vram_chunk_idx    = new_vram_chunk_idx;
        fresh.chunk_device      = new_chunk_device;
        fresh.storage_owner     = std::move(new_storage_owner);
        store_lease_state_locked(fresh);
    }

    // 4. Release the old leases and allocation owner outside the lock.
    release_lease_state(stale);
    stale_owned_alloc.reset();
    return *this;
}

mem_handle & mem_handle::operator=(mem_handle && other) noexcept {
    if (this == &other) {
        return *this;
    }

    resolved_ptr          new_cached;
    uint64_t              new_gen                     = 0;
    mem_handle_kind       new_kind                    = mem_handle_kind::DIRECT;
    int                   new_device                  = HOST_DEVICE;
    unified_cache_key     new_key                     = {};
    int                   new_zone_id                 = 0;
    size_t                new_offset                  = 0;
    size_t                new_size                    = 0;
    size_t                new_backing_extent          = 0;
    size_t                new_backing_offset          = 0;
    size_t                new_slice_offset            = 0;
    bool                  new_is_slice                = false;
    uint64_t              new_arena_gen               = 0;
    uint64_t              new_canonical_allocation_id = 0;
    uint64_t              new_canonical_generation    = 0;
    size_t                new_canonical_extent        = 0;
    std::shared_ptr<alloc_handle> new_owned_alloc;
    const char *          new_debug_owner_tag = "";
    lease_state           fresh;
    {
        mem_handle_lock_guard g(other.lock_);
        new_kind                    = other.kind_;
        new_device                  = other.device_;
        new_key                     = other.key_;
        new_zone_id                 = other.zone_id_;
        new_offset                  = other.offset_;
        new_size                    = other.size_;
        new_backing_extent          = other.backing_extent_;
        new_backing_offset          = other.backing_offset_;
        new_slice_offset            = other.slice_offset_;
        new_is_slice                = other.is_slice_;
        new_arena_gen               = other.arena_gen_;
        new_canonical_allocation_id = other.canonical_allocation_id_;
        new_canonical_generation    = other.canonical_generation_;
        new_canonical_extent        = other.canonical_extent_;
        new_owned_alloc             = std::move(other.owned_alloc_);
        new_debug_owner_tag         = other.debug_owner_tag_;
        new_gen                     = other.gen_;
        new_cached                  = std::move(other.cached_);
        fresh                      = other.take_lease_state_locked();

        other.kind_                    = mem_handle_kind::DIRECT;
        other.device_                  = HOST_DEVICE;
        other.key_                     = {};
        other.zone_id_                 = 0;
        other.offset_                  = 0;
        other.size_                    = 0;
        other.backing_extent_          = 0;
        other.backing_offset_          = 0;
        other.slice_offset_            = 0;
        other.is_slice_                = false;
        other.arena_gen_               = 0;
        other.canonical_allocation_id_ = 0;
        other.canonical_generation_    = 0;
        other.canonical_extent_        = 0;
        other.debug_owner_tag_         = "";
        other.gen_                     = 0;
        other.cached_                  = {};
    }

    lease_state                   stale;
    std::shared_ptr<alloc_handle> stale_owned_alloc;
    {
        mem_handle_lock_guard g(lock_);
        stale                    = take_lease_state_locked();
        stale_owned_alloc        = std::move(owned_alloc_);
        kind_                    = new_kind;
        device_                  = new_device;
        key_                     = new_key;
        zone_id_                 = new_zone_id;
        offset_                  = new_offset;
        size_                    = new_size;
        backing_extent_          = new_backing_extent;
        backing_offset_          = new_backing_offset;
        slice_offset_            = new_slice_offset;
        is_slice_                = new_is_slice;
        arena_gen_               = new_arena_gen;
        canonical_allocation_id_ = new_canonical_allocation_id;
        canonical_generation_    = new_canonical_generation;
        canonical_extent_        = new_canonical_extent;
        owned_alloc_             = std::move(new_owned_alloc);
        debug_owner_tag_         = new_debug_owner_tag;
        gen_                     = new_gen;
        cached_                  = std::move(new_cached);
        store_lease_state_locked(fresh);
    }

    release_lease_state(stale);
    stale_owned_alloc.reset();
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

    // Resolve the original arena allocation, not a manufactured slice pointer;
    // resolved_view_locked() applies the derived offset after the generation
    // check on every call.
    size_t backing_offset = 0;
    {
        mem_handle_lock_guard g(lock_);
        if (slice_offset_ > offset_ ||
            (backing_extent_ != 0 &&
             (slice_offset_ > backing_extent_ || size_ > backing_extent_ - slice_offset_))) {
            return {};
        }
        backing_offset = backing_offset_;
    }
    void * ptr = cache->offset_to_ptr(backing_offset);
    if (!ptr) {
        return {};
    }

    // Cache the unsliced backing pointer. Arena handles are always on-device
    // with AOS layout (arena zones hold raw allocations, not managed weights).
    const resolved_ptr resolved = { ptr, backing_extent_, GGML_LAYOUT_AOS, true };
    resolved_ptr       view;
    {
        mem_handle_lock_guard g(lock_);
        cached_ = resolved;
        gen_    = arena_gen_;
        view    = resolved_view_locked();
    }
    return view;
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

retained_handle_publish_ticket begin_retained_handle_publish() {
    auto &                      state = *g_retained_handles_state;
    std::lock_guard<std::mutex> lock(state.mutex);
    ++state.publishers;
    return retained_handle_publish_ticket(true);
}

retained_handle_publish_ticket::retained_handle_publish_ticket(retained_handle_publish_ticket && other) noexcept :
    active_(std::exchange(other.active_, false)) {}

retained_handle_publish_ticket & retained_handle_publish_ticket::operator=(retained_handle_publish_ticket && other) noexcept {
    if (this != &other) {
        reset();
        active_ = std::exchange(other.active_, false);
    }
    return *this;
}

retained_handle_publish_ticket::~retained_handle_publish_ticket() {
    reset();
}

void retained_handle_publish_ticket::reset() noexcept {
    if (!active_) {
        return;
    }

    active_ = false;
    auto & state = *g_retained_handles_state;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        GGML_ASSERT(state.publishers > 0);
        --state.publishers;
    }
    state.cv.notify_all();
}

bool drain_retained_handles(bool wait_all, uint32_t timeout_ms) {
    if (!wait_all) {
        // Retained handles are released by the background drain worker.  Avoid
        // get_info(command_execution_status) polling on inference threads:
        // that Level Zero query can block on in-flight events.
        return true;
    }

    auto &                       state = *g_retained_handles_state;
    std::unique_lock<std::mutex> lock(state.mutex);
    return state.cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&state] {
        return state.queue.empty() && state.active == 0 && state.publishers == 0;
    });
}

size_t graph_retained_handle_count() {
    auto &                      state = *g_retained_handles_state;
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.graph_unwaitable.size();
}

void release_graph_retained_handles() {
    std::vector<mem_handle> released;
    size_t                  n = 0;
    {
        auto &                      state = *g_retained_handles_state;
        std::lock_guard<std::mutex> lock(state.mutex);
        n = state.graph_unwaitable.size();
        released.swap(state.graph_unwaitable);
    }
    GGML_SYCL_DEBUG("[MEM-HANDLE] released %zu command-graph retained leases\n", n);
}

static void publish_handles_until_event(std::vector<mem_handle> handles, sycl::event event) {
    if (handles.empty()) {
        return;
    }

#ifdef GGML_SYCL_RETAINED_PUBLICATION_TESTING
    if (g_fail_next_retained_handle_publication.exchange(false, std::memory_order_acq_rel)) {
        throw std::bad_alloc();
    }
#endif

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

void retain_handles_until_event(std::vector<mem_handle> handles, sycl::event event) {
    publish_handles_until_event(std::move(handles), std::move(event));
}

void retain_handles_until_event(std::vector<mem_handle> handles, sycl::event event,
                                retained_handle_publish_ticket ticket) {
    publish_handles_until_event(std::move(handles), std::move(event));
    GGML_UNUSED(ticket);
}

void retain_handles_until_event_transactional(std::vector<mem_handle> handles, sycl::event event,
                                              retained_handle_publish_ticket & ticket) {
    publish_handles_until_event(std::move(handles), std::move(event));
    ticket.reset();
}

#ifdef GGML_SYCL_RETAINED_PUBLICATION_TESTING
void fail_next_retained_handle_publication_for_test() {
    g_fail_next_retained_handle_publication.store(true, std::memory_order_release);
}
#endif

void set_graph_retained_handle_sink(std::vector<mem_handle> * sink) {
    g_graph_retained_handle_sink = sink;
}

}  // namespace ggml_sycl
