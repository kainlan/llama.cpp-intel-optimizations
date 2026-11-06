#pragma once

// The lazy tables' IO pool: its registry, the counted lease that pins it, and the reap that frees a pool whose last
// lease is gone (docs/design/sycl-storage-tier-lazy-tables.md §3.3).
//
// One registry per process, defined once in lazy-io-pool.cpp beside the leaf mutex that guards it, never in this
// header: a header-only registry would give every target that includes it a registry of its own, two ledgers for one
// fact. The pool here holds no ring and starts no thread; the block ring and the pread workers arrive with the IO
// engine. What it already decides is coverage: a planner reads the live ring's bytes through a held lease, so a
// successful take is the pin, and a plan made without one carries no STORAGE table (§4.3 step 4).
//
// Device-free on purpose, like host-cpu-mapping.hpp: this header and lazy-io-pool.cpp include no SYCL header, so a
// host test links the translation unit alone.
//
// The registry's leaf mutex (canonical contract §12.5, L5) is a true leaf: nothing is taken, allocated, freed, waited
// on or logged under it. A take reserves a generation under it, builds its pool outside it and installs the pool under
// it on a generation match; a final drop only parks the pool, and the reap frees parked pools after its unlock.

#include <cstddef>
#include <cstdint>

namespace ggml_sycl {

// The block ring's queue depth (§3.3): the planner sizes block_ring_need as this many read spans of the widest STORAGE
// table, and the ring the IO engine allocates holds exactly that, so both read this constant and nothing else.
constexpr size_t LAZY_BLOCK_RING_QD = 64;

// The O_DIRECT block (§5.2): a read span starts and ends on one.
constexpr size_t LAZY_IO_BLOCK_BYTES = 4096;

// One row's read span (§5.2): the block-aligned O_DIRECT window that holds a row of `row_size` bytes starting anywhere
// in a block, ceil((LAZY_IO_BLOCK_BYTES - 1 + row_size) / LAZY_IO_BLOCK_BYTES) blocks. A row of at most 4097 bytes
// spans two blocks.
constexpr size_t lazy_block_ring_span(size_t row_size) {
    return (row_size + 2 * (LAZY_IO_BLOCK_BYTES - 1)) / LAZY_IO_BLOCK_BYTES * LAZY_IO_BLOCK_BYTES;
}

// The bytes a HOST table's body requests for a table at `file_offset` of `table_bytes` (§3.3, D1): the O_DIRECT window
// over the table, from the start of the block that holds its first byte to the end of the block that holds its last,
// so the table sits at body + file_offset % LAZY_IO_BLOCK_BYTES. The HOST term the planner charges and the request the
// materialization makes both read this one rule, here so that neither needs the lazy table's SYCL header. SIZE_MAX when
// the window does not fit a size_t.
constexpr size_t lazy_table_host_body_request_bytes(uint64_t file_offset, size_t table_bytes) {
    const size_t head = static_cast<size_t>(file_offset % LAZY_IO_BLOCK_BYTES);
    if (table_bytes > SIZE_MAX - head - (LAZY_IO_BLOCK_BYTES - 1)) {
        return SIZE_MAX;
    }
    return (head + table_bytes + LAZY_IO_BLOCK_BYTES - 1) / LAZY_IO_BLOCK_BYTES * LAZY_IO_BLOCK_BYTES;
}

// A STORAGE row cache's slab grows in chunks of this many bytes (§3.3, §6.3). The planner charges a cache's slab as its
// whole chunks and one remainder, and the row cache the T4 row (design §11) adds allocates the same chunks, so its
// activation check compares like with like.
constexpr size_t LAZY_ROW_CACHE_CHUNK_BYTES = 256ull << 20;

// A STORAGE row cache's index load factor, LAZY_ROW_CACHE_LOAD_NUM / LAZY_ROW_CACHE_LOAD_DEN = 0.75 (§6.3): the index
// has a bucket per 0.75 slot, and a shard holds at most 0.75 of its buckets, so a probe always meets an empty bucket.
// Enumerators, so the factor adds no namespace-scope object (§11).
enum lazy_row_cache_load_factor : size_t { LAZY_ROW_CACHE_LOAD_NUM = 3, LAZY_ROW_CACHE_LOAD_DEN = 4 };

// A STORAGE row cache's index buckets at `slots` slots: slots / 0.75, rounded up.
constexpr size_t lazy_row_cache_index_buckets(size_t slots) {
    return (LAZY_ROW_CACHE_LOAD_DEN * slots + LAZY_ROW_CACHE_LOAD_NUM - 1) / LAZY_ROW_CACHE_LOAD_NUM;
}

// The live rows `buckets` index buckets may hold: 0.75 of them, rounded down.
constexpr size_t lazy_row_cache_bucket_cap(size_t buckets) {
    return LAZY_ROW_CACHE_LOAD_NUM * buckets / LAZY_ROW_CACHE_LOAD_DEN;
}

// A STORAGE row cache's index row at `slots` slots, before its mapping (§3.3, §6.3): one 8 B key word per slot, then
// one 4 B slot id per bucket.
constexpr size_t lazy_row_cache_index_raw_bytes(size_t slots) {
    return sizeof(uint64_t) * slots + sizeof(uint32_t) * lazy_row_cache_index_buckets(slots);
}

struct lazy_io_pool;

// A counted reference on the IO pool. Move-only: the count changes only under the registry's leaf mutex, at a take, a
// clone and a drop. An increment on a DYING pool fails, and the drop that reaches zero marks the pool DYING and parks
// it in the same critical section, so a take and the last drop cannot interleave: the take lands first and the pool
// lives, or the pool is DYING and a take builds a new one.
class ring_lease {
  public:
    ring_lease() noexcept = default;
    ~ring_lease();
    ring_lease(ring_lease && other) noexcept;
    ring_lease & operator=(ring_lease && other) noexcept;
    ring_lease(const ring_lease &)             = delete;
    ring_lease & operator=(const ring_lease &) = delete;

    // Another counted reference on the same pool, taken under the leaf mutex; it allocates nothing. It is called only
    // on a held lease, so the pool's count is at least 1 and the pool is not DYING. A clone that finds it DYING reads
    // the flag under the lock, releases the lock, and only then logs [HOST-PLAN-BUG] at WARN and aborts: a noexcept
    // clone cannot return a typed refusal, and a silent empty lease is forbidden (§10). A clone of an empty lease is
    // empty.
    ring_lease clone() const noexcept;

    bool held() const noexcept { return pool_ != nullptr; }

    // The live ring's bytes, read through this lease: the coverage a planner reads, 0 when the pool has no ring yet.
    // A pool that holds leases but no ring covers 0, and this pool never has one, so a held lease always reads 0.
    uint64_t ring_bytes() const noexcept;

    // Drops this reference now and leaves the lease empty. The drop that reaches zero marks the pool DYING and parks it
    // on the reap list; once the registry is shut down a drop only decrements.
    void reset() noexcept;

  private:
    friend ring_lease lazy_io_pool_take();
#if defined(GGML_SYCL_PRIVATE_TESTING)
    friend void ggml_sycl_test_lazy_mark_dying(const ring_lease & lease);
#endif

    explicit ring_lease(lazy_io_pool * pool) noexcept : pool_(pool) {}

    lazy_io_pool * pool_ = nullptr;
};

// Takes a lease on the live pool, creating the pool when there is none: find the slot empty and reserve the generation
// after the last one installed under the lock, build the pool and its reap node outside it, and install it under the
// lock if the generation still matches, that is, if no take installed a pool meanwhile. A take that lost joins the pool
// installed meanwhile, or, when that pool has already parked, takes again; either way it destroys its own pool outside
// the lock. Returns an empty lease once the registry is shut down. Throws std::bad_alloc when the pool cannot be built.
ring_lease lazy_io_pool_take();

enum class lazy_reap_scope : uint8_t {
    FULL,  // frees every parked pool and then its node
};

// Reaps the parked pools: swaps the reap list out under the leaf mutex, so concurrent reapers each take their own
// nodes, and frees them after the unlock. A FULL reap counts even when the list is empty. Never throws, and never runs
// under a lock this registry knows of: every caller is an unlocked context.
void ggml_sycl_lazy_ring_reap(lazy_reap_scope scope) noexcept;

// Marks the registry shut down: from then on a take returns an empty lease and a lease's drop only decrements, so a
// lease destroyed during static destruction never touches the registry's mutex. No production caller sets it yet: the
// T5 engine (design §11 row T5) adds the lease drain at backend shutdown, which sets it after its own FULL reap.
void lazy_io_pool_mark_shut_down() noexcept;

// The one wrapper a planner caller holds around its planning (§3.3): its constructor runs the entry FULL reap, take_if
// takes the provisional lease, pin() hands it to the planners, and its destructor drops the lease and then runs the
// closing FULL reap, by member order (the reap guard is declared before the lease, so it is built first and destroyed
// last). Declared in an unlocked context. It holds only the lease and pin(). No production caller builds this scope
// yet; the T5 engine (design §11 row T5) adds it.
class lazy_coverage_scope {
  public:
    lazy_coverage_scope() noexcept                               = default;
    ~lazy_coverage_scope()                                       = default;
    lazy_coverage_scope(const lazy_coverage_scope &)             = delete;
    lazy_coverage_scope & operator=(const lazy_coverage_scope &) = delete;

    // Takes the provisional lease when the inventory has at least one lazy-table candidate. Returns whether a lease is
    // held: false for no candidate, and false once the registry is shut down. Throws std::bad_alloc like the take.
    bool take_if(bool has_lazy_candidate);

    // The pin a planner reads coverage through; null when no lease is held.
    const ring_lease * pin() const noexcept { return lease_.held() ? &lease_ : nullptr; }

  private:
    struct reap_guard {
        reap_guard() noexcept;
        ~reap_guard();
        reap_guard(const reap_guard &)             = delete;
        reap_guard & operator=(const reap_guard &) = delete;
    };

    reap_guard guard_;
    ring_lease lease_;
};

#if defined(GGML_SYCL_PRIVATE_TESTING)
// The registry's state, read under its leaf mutex (§11's wrapper arm). The two take-time snapshots are the reaps run
// and the parked nodes when the last successful take installed or joined a pool, so a fixture can tell an entry reap
// that ran before a take from one that ran after it, which the final totals cannot.
struct lazy_registry_state {
    uint64_t live_pools             = 0;  // pools in the registry's slot: 0 or 1
    uint64_t live_leases            = 0;  // the slot pool's lease count
    uint64_t generation             = 0;  // the last installed pool's generation
    uint64_t live_generation        = 0;  // the slot pool's generation; 0 when the slot is empty
    uint64_t parked_nodes           = 0;  // DYING pools parked on the reap list
    uint64_t full_reaps_run         = 0;  // FULL reaps run, an empty list included
    uint64_t reaps_run_at_last_take = 0;
    uint64_t parked_at_last_take    = 0;
};

// Copies the registry's state into *out. Returns false, writing nothing, for a null pointer.
bool ggml_sycl_lazy_registry_for_testing(lazy_registry_state * out);

// The registry's one-shot hold hook, armed at one of two points; arming either point replaces whatever was armed, and
// arming with a null callback disarms. The callback runs once, with no lock held, and blocks for as long as the fixture
// wants its caller held.
using lazy_hold_fn = void (*)(void * ctx);

// Arms the hook in the FULL reap: the first FULL reap that starts after the arming calls it after its count has landed
// and before it swaps the list out.
void ggml_sycl_test_lazy_reap_hold(lazy_hold_fn hold, void * ctx);

// Arms the hook in the take: the first take that reserves a generation after the arming calls it after building its
// pool and before the install, so a fixture can install and park another pool inside that window.
void ggml_sycl_test_lazy_take_hold(lazy_hold_fn hold, void * ctx);

// The coverage pin a direct-planner fixture passes (§3.3): an RAII guard that owns the lease, so the const ring_lease *
// the planner takes cannot dangle. It takes the pool through the registry's take path, never off-registry, and its
// destructor drops the lease and then runs an unlocked FULL reap, so the next fixture's pin finds no pool.
class lazy_test_coverage_pin {
  public:
    explicit lazy_test_coverage_pin(ring_lease lease) noexcept;
    ~lazy_test_coverage_pin();
    lazy_test_coverage_pin(const lazy_test_coverage_pin &)             = delete;
    lazy_test_coverage_pin & operator=(const lazy_test_coverage_pin &) = delete;

    const ring_lease * pin() const noexcept { return lease_.held() ? &lease_ : nullptr; }

    const ring_lease & lease() const noexcept { return lease_; }

  private:
    ring_lease lease_;
};

// Marks the pool `lease` holds DYING under the leaf mutex, without dropping a reference or parking it, so the next
// clone of that lease finds the state a held lease never shows and takes its abort path. Only a child process that is
// about to abort calls it: the pool can no longer be parked or reaped. An empty lease is ignored.
void ggml_sycl_test_lazy_mark_dying(const ring_lease & lease);

// Only ring_bytes == 0, a ring-less pool, is accepted here: every table is then STORAGE-eligible and block_ring_charge
// is the mapped need. Any other value aborts naming it, since a ring arrives only with the IO engine. A call while a
// guard is live joins the live pool with a counted reference.
lazy_test_coverage_pin ggml_sycl_test_make_coverage_pin(uint64_t ring_bytes);
#endif

}  // namespace ggml_sycl
