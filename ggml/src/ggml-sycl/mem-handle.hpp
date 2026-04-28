// Smart handle infrastructure for SYCL unified memory manager.
// Handles cache pointer resolution with generation-based staleness detection.
// P10 of the unified memory manager epic.
//
// See docs/smart-handle-design.md for architecture details.

#pragma once

#include <cstdint>

#include "unified-cache.hpp"  // unified_cache_key, ggml_layout_mode

namespace ggml_sycl {

// === Generation counter ===
// Single global atomic, bumped on eviction/promotion/flush (~0-3 times per
// inference run).  Hot path: compare + return cached ptr (~3 ns).

// Current global generation.  Callers compare their cached generation against
// this to detect staleness.  Relaxed ordering is sufficient — generation bumps
// happen under rw_mutex_ in unified_cache, and any resolve_slow() that follows
// a stale check will acquire the cache lock and see the new state.
uint64_t cache_generation();

// Bump the global generation.  Called from unified_cache when any pointer could
// have moved: evict_one, promote_to_device, finalize_evictions_locked.
void cache_generation_bump();

// === Resolved pointer ===
// The result of resolving a mem_handle.  Contains the current pointer and
// metadata needed by the caller.

struct resolved_ptr {
    void *           ptr       = nullptr;
    ggml_layout_mode layout    = GGML_LAYOUT_AOS;
    bool             on_device = false;

    explicit operator bool() const { return ptr != nullptr; }
};

// === mem_handle ===
// Lightweight handle that caches pointer resolution.  Two kinds:
//
// WEIGHT: cache-managed, gen-checked.  resolve() compares cached generation
//         against global; if stale, calls resolve_slow() which queries the
//         unified cache.  ~3 ns hot path when pointer hasn't moved.
//
// DIRECT: raw pointer wrapper.  resolve() always returns the cached pointer.
//         Used for scratch/KV/staging buffers that are never moved by the cache.

enum class mem_handle_kind : uint8_t {
    WEIGHT         = 0,  // Cache-managed, generation-checked
    DIRECT         = 1,  // Raw pointer, never stale
    ARENA_RUNTIME  = 2,  // Handle into RUNTIME zone (ggml compute buffers)
    ARENA_SCRATCH  = 3,  // Handle into SCRATCH zone (pool_leg per-op scratch)
    ARENA_ONEDNN   = 4,  // Handle into ONEDNN zone (oneDNN scratchpad)
    CHUNK_LEASE    = 5,  // Raw pointer + arena chunk lease (llama.cpp-dyhdl).
                         // Protects pointers derived from host or VRAM arena
                         // chunks (e.g. tensor->data from host_arena) against
                         // sycl::free of the underlying chunk while this
                         // handle is alive.
};

// Forward declaration: the backing cache_entry type whose in_use_count we
// hold a lease on.  Full definition in unified-cache.hpp.
struct unified_cache_entry;

class mem_handle {
public:
    // Create an invalid handle.
    mem_handle() = default;

    // Destructor: if this handle holds a WEIGHT lease on a cache entry,
    // decrement that entry's in_use_count.  Dtor is the only release point;
    // after `~mem_handle` returns, the underlying cache entry may be evicted
    // on the next eviction scan.
    ~mem_handle();

    // Copy / move: maintain the lease refcount correctly.  A copy bumps the
    // target entry's count (so two handles each hold an independent lease);
    // a move transfers ownership (net refcount unchanged).
    mem_handle(const mem_handle & other);
    mem_handle(mem_handle && other) noexcept;
    mem_handle & operator=(const mem_handle & other);
    mem_handle & operator=(mem_handle && other) noexcept;

    // Create a WEIGHT handle from a cache key + device.
    // The handle starts with gen_ = 0 (stale), so the first resolve() will
    // query the cache.
    static mem_handle from_weight(const unified_cache_key & key, int device);

    // Create a WEIGHT handle whose lease has already been acquired by the
    // caller (via unified_cache::acquire_weight_lease).  The caller transfers
    // ownership of the in_use_count bump to the new mem_handle — do NOT
    // decrement it yourself.  Used by cpu-dispatch.cpp get_host_ptr() to
    // package the lease returned by acquire_weight_lease into a handle
    // whose dtor releases automatically (llama.cpp-vtf7f).
    //
    // `entry` may be nullptr for S1-PRELOAD direct entries (no refcount).
    static mem_handle from_weight_lease(const ggml_sycl_cache_id & key_id,
                                        int                        device,
                                        void *                     ptr,
                                        ggml_layout_mode           layout,
                                        bool                       on_device,
                                        unified_cache_entry *      entry);

    // Create a WEIGHT handle from a bare cache ID + device.
    // Convenience factory for callers that have ggml_sycl_cache_id (e.g.
    // layer_weight_set fields) rather than a full unified_cache_key.
    static mem_handle from_cache_id(const ggml_sycl_cache_id & id, int device);

    // Sentinel for host pointers where device ownership is not applicable.
    // Pass as device to from_direct() for host-pinned or CPU-resident pointers.
    static constexpr int HOST_DEVICE = -1;

    // Create a DIRECT handle from a raw pointer.
    // device: owning SYCL device index, or HOST_DEVICE (-1) for host-pinned /
    //   CPU-resident pointers that are not owned by any specific GPU device.
    // resolve(device_id) checks that the caller's device matches device (when device >= 0)
    //   and returns null with a diagnostic on mismatch.
    // The zero-arg resolve() always returns this pointer without any device or cache check.
    static mem_handle from_direct(void * ptr, ggml_layout_mode layout, bool on_device,
                                  int device = HOST_DEVICE);

    // Create an arena zone handle.
    // zone_id maps to vram_zone_id (KV=0, WEIGHT=1, ONEDNN=2, RUNTIME=3, SCRATCH=4).
    // The handle kind is derived from zone_id:
    //   RUNTIME -> ARENA_RUNTIME, SCRATCH -> ARENA_SCRATCH, ONEDNN -> ARENA_ONEDNN.
    static mem_handle from_arena_zone(int zone_id, size_t offset, size_t size,
                                      int device, uint64_t generation);

    // llama.cpp-dyhdl: create a handle from a raw pointer whose arena-chunk
    // ownership should be reference-counted for the handle's lifetime.
    //
    // Resolution order:
    //   1. Query unified_cache on `device`: if ptr is in the VRAM arena,
    //      bump that chunk's lease and return a CHUNK_LEASE handle.
    //   2. Else query the cache's host_arena: if ptr is in a pinned chunk,
    //      bump that chunk's lease and return a CHUNK_LEASE handle.
    //   3. Else return a DIRECT handle (no protection — the pointer is not
    //      in any known arena chunk, so there's nothing to refcount).
    //
    // Callers MUST keep the returned handle alive across any use of `ptr`
    // that survives into another thread / queue submit / future.  The
    // destructor releases the chunk lease exactly once.
    static mem_handle from_chunk_ptr(void * ptr, int device,
                                     ggml_layout_mode layout    = GGML_LAYOUT_AOS,
                                     bool             on_device = false);

    // Resolve with explicit device check.  Returns null resolved_ptr and logs a
    // diagnostic if device_id does not match this handle's device_ (when device_
    // is >= 0).  HOST_DEVICE (-1) handles always pass.  Use this overload from
    // any dispatch path that knows which device it is submitting work to.
    resolved_ptr resolve(int device_id) const;

    // Resolve the current pointer.  Hot path (~3 ns):
    //   if (kind == DIRECT || gen_ == cache_generation())
    //       return cached resolved_ptr
    //   else
    //       return resolve_slow()
    resolved_ptr resolve() const;

    // True if this handle has ever been successfully resolved.
    // A handle is valid iff it has a resolved pointer; the kind check was previously
    // included to allow DIRECT handles with nullptr but DIRECT handles must always
    // have an explicit non-null pointer, so the kind check was incorrect for
    // default-constructed handles (kind_==DIRECT, ptr==nullptr → falsely valid).
    bool valid() const { return cached_.ptr != nullptr; }

    // Access the cache key (only meaningful for WEIGHT handles).
    const unified_cache_key & key() const { return key_; }

    // Access the device ID.
    int device() const { return device_; }

    // Access the handle kind.
    mem_handle_kind kind() const { return kind_; }

    // True if this is a cache-managed WEIGHT handle (not a raw DIRECT pointer).
    bool is_weight() const { return kind_ == mem_handle_kind::WEIGHT; }

    // True if this is any arena-backed handle.
    bool is_arena() const {
        return kind_ >= mem_handle_kind::ARENA_RUNTIME &&
               kind_ <= mem_handle_kind::ARENA_ONEDNN;
    }

    // Arena-specific accessors (meaningful only for arena handles).
    int      zone_id()    const { return zone_id_; }
    size_t   offset()     const { return offset_; }
    size_t   size()       const { return size_; }
    uint64_t generation() const { return arena_gen_; }

private:
    // Slow path: re-query the unified cache for the current pointer.
    resolved_ptr resolve_slow() const;

    // Slow path for arena handles: resolve base + offset from arena.
    resolved_ptr resolve_arena() const;

    // Release any held lease on the backing cache entry.  Called from dtor,
    // copy-assign, and move-assign before transitioning to a new state.
    // No-op if no lease is held.
    void release_lease() noexcept;

    mem_handle_kind    kind_   = mem_handle_kind::DIRECT;
    int                device_ = HOST_DEVICE;
    unified_cache_key  key_    = {};

    // Arena-specific fields (only used for ARENA_* kinds).
    int      zone_id_   = 0;       // Which zone (maps to vram_zone_id)
    size_t   offset_    = 0;       // Offset within the zone
    size_t   size_      = 0;       // Allocation size
    uint64_t arena_gen_ = 0;       // Arena generation (for invalidation)

    // Mutable because resolve() is logically const (returns the current
    // pointer) but updates the cache as a side effect.
    mutable uint64_t     gen_    = 0;
    mutable resolved_ptr cached_ = {};

    // Lease-protected backing entry pointer (llama.cpp-vtf7f).  When non-null,
    // this handle has incremented the entry's in_use_count, guaranteeing the
    // backing allocation cannot be sycl::free'd or evicted until the handle
    // is destroyed.  Only WEIGHT-kind handles acquire leases — DIRECT and
    // ARENA_* handles ignore this field.
    //
    // Pointer stability: std::unordered_map guarantees pointers to elements
    // remain valid across insert/rehash operations (C++17 §26.2.7); erase
    // only invalidates pointers to the erased element.  Eviction paths
    // MUST NOT erase entries with in_use_count > 0, which is the contract
    // enforced in unified_cache::evict_one / remove / evict_and_flush.
    mutable unified_cache_entry * leased_entry_ = nullptr;

    // llama.cpp-dyhdl: chunk-level lease backref.  Defense-in-depth beneath
    // the cache_entry refcount: this stops the underlying arena chunk from
    // being sycl::free'd while any mem_handle holds a pointer derived from it.
    //
    // Source encoding:
    //   0 = none (no chunk lease held — DIRECT handle, or ptr not in any arena)
    //   1 = host pinned_chunk_pool (handle stored in host_chunk_handle_)
    //   2 = VRAM arena (handle stored in vram_chunk_idx_)
    //
    // Populated:
    //   - by from_chunk_ptr() for raw-pointer escape migration
    //   - by resolve_slow() for WEIGHT handles, so cached weights auto-pin
    //     their chunk (belt + suspenders alongside leased_entry_ refcount)
    //
    // A CHUNK_LEASE-kind handle stores its protected raw ptr in cached_.ptr;
    // resolve() returns cached_ directly (never re-queries the cache).
    mutable uint8_t  chunk_source_      = 0;
    mutable uint64_t host_chunk_handle_ = UINT64_MAX;  // pinned_chunk_pool::INVALID_CHUNK_HANDLE
    mutable int32_t  vram_chunk_idx_    = -1;
    mutable int32_t  chunk_device_      = -1;
};

// === layer_weight_handles ===
// Smart handle version of layer_weight_pointers (unified-cache.hpp).
// Holds mem_handle per weight field.  Call resolve_all() to produce a
// layer_weight_pointers struct of raw void* for immediate use.

struct layer_weight_handles {
    mem_handle attn_norm;
    mem_handle q_proj;
    mem_handle k_proj;
    mem_handle v_proj;
    mem_handle o_proj;
    mem_handle ffn_norm;
    mem_handle gate_proj;
    mem_handle up_proj;
    mem_handle down_proj;
    mem_handle attn_qkv_proj;     // Fused QKV (optional)
    mem_handle ffn_gate_up_proj;  // Fused gate+up (optional)

    // Resolve all handles and write raw pointers into a layer_weight_pointers.
    // Returns true if all required (non-optional) handles resolve successfully.
    bool resolve_all(layer_weight_pointers & out) const;

    // Construct handles for a layer from a weight set + device ID.
    static layer_weight_handles from_weight_set(const layer_weight_set & ws, int device);
};

// Build layer weight handles for a prefetched layer.
// Queries the unified cache for the layer's weight set (cache IDs) and
// constructs mem_handle per weight.  Returns false if the layer has not
// been prefetched.
bool build_layer_handles(int device, int layer_id, layer_weight_handles & out);

}  // namespace ggml_sycl
