#pragma once

// The lazy tables' IO pool: its registry, the counted lease that pins it, and the reap that frees a pool whose last
// lease is gone (docs/design/sycl-storage-tier-lazy-tables.md §3.3).
//
// One registry per process, defined once in lazy-io-pool.cpp beside the leaf mutex that guards it, never in this
// header: a header-only registry would give every target that includes it a registry of its own, two ledgers for one
// fact. The pool decides coverage: a planner reads the live ring's bytes through a held lease, so a successful take is
// the pin, and a plan made without one carries no STORAGE table (§4.3 step 4). The pool's block ring is created by
// block_ring_ensure, once, with the IO engine whose pread workers read into its slots (§3.3, §5.2), and both are
// released by the reap after the pool's last lease is gone.
//
// Device-free on purpose, like host-cpu-mapping.hpp: this header and lazy-io-pool.cpp include no SYCL header, so a
// host test links the translation unit alone. It has two link seams, each defined by the backend and by a fixture in
// each host test: the ring's memory (lazy-table.cpp) and the strict-leases switch (unified-cache.cpp).
//
// The registry's leaf mutex (canonical contract §12.5, L5) is a true leaf: nothing is taken, allocated, freed, waited
// on or logged under it. A take reserves a generation under it, builds its pool outside it and installs the pool under
// it on a generation match; a final drop only parks the pool, and the reap frees parked pools after its unlock.
//
// witnessed-mutex.hpp comes with this header: the reap refuses under any of the five witnessed locks (§3.3), and
// the units that declare the inventory and plan locks, ggml-sycl.cpp and unified-cache.cpp, reach the type here.

#include "witnessed-mutex.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

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

// ---- The block ring's memory: the one link seam (§3.3) ----

// The block ring's memory, owner-holding and type-erased: the backend's object holds the owner-first mem_handle of the
// ring's HOST_CPU / LAZY_STAGING allocation with the CPU view resolved from it once, and destroying it drops that
// handle, the sole release point, whose free credits the budget ledger. `bytes` are the usable bytes, `charged` the
// ledger's charge for the allocation (host_cpu_mapped_size of the handle).
class lazy_io_host_memory {
  public:
    virtual ~lazy_io_host_memory()                               = default;
    lazy_io_host_memory(const lazy_io_host_memory &)             = delete;
    lazy_io_host_memory & operator=(const lazy_io_host_memory &) = delete;

    uint8_t * data() const noexcept { return data_; }

    size_t size() const noexcept { return bytes_; }

    size_t charged_bytes() const noexcept { return charged_; }

  protected:
    lazy_io_host_memory(uint8_t * data, size_t bytes, size_t charged) noexcept :
        data_(data),
        bytes_(bytes),
        charged_(charged) {}

  private:
    uint8_t * data_    = nullptr;
    size_t    bytes_   = 0;
    size_t    charged_ = 0;
};

// The block ring's allocation of `bytes` usable bytes, O_DIRECT-aligned, on `device`'s cache (§3.3), owned by the pool
// whose registry generation is `pool_generation` and charged to the ledger under `load_txn_id`, whose own chunks a
// yield never gives back (§4.2): null when it is refused. May throw std::bad_alloc. Defined once per link:
// lazy-table.cpp for the backend, a fixture in each host test that links lazy-io-pool.cpp.
std::unique_ptr<lazy_io_host_memory> lazy_io_allocate_host(size_t   bytes,
                                                           int      device,
                                                           uint64_t pool_generation,
                                                           uint64_t load_txn_id);

// GGML_SYCL_STRICT_LEASES=1, the switch under which every [HOST-PLAN-BUG] line of the reap and the drop aborts after
// its WARN (§10). Defined once per link: unified-cache.cpp for the backend, which declares it in unified-cache.hpp too,
// and the fixture in each host test that links lazy-io-pool.cpp, whose switch only a death child sets.
bool ggml_sycl_strict_enabled();

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

    // The live ring's usable bytes, read through this lease: the coverage a planner reads, 0 when the pool has no ring
    // yet. A planner reads it through lazy_ring_coverage_read, which counts the read.
    uint64_t ring_bytes() const noexcept;

    // Drops this reference now and leaves the lease empty. The drop that reaches zero marks the pool DYING and parks it
    // on the reap list; once the registry is shut down a drop only decrements.
    void reset() noexcept;

  private:
    friend ring_lease lazy_io_pool_take();
    friend struct lazy_io_pool_access;  // the engine's paths in lazy-io-pool.cpp: join, ensure, submit, shutdown
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

// A lease on the live pool, never a new one (§3.3): what a read takes so the pool and its ring outlive it. Empty when
// no pool is live, when the live pool has no ring, or once the registry is shut down.
ring_lease lazy_io_pool_join();

enum class lazy_reap_scope : uint8_t {
    FULL,    // frees every parked pool, which stops and joins its workers, and then its node
    WORKER,  // frees only the parked pools' rings and leaves each node, with the pool and its workers, on the list
};

// Reaps the parked pools. FULL swaps the reap list out under the leaf mutex, so concurrent reapers each take their own
// nodes, and frees them after the unlock; it counts even when the list is empty. WORKER, the read-completion path's
// scope, takes each parked pool's ring out under the mutex and frees the rings after the unlock, so a pool worker
// never joins itself (§3.3). Never throws, and never runs under a lock this registry knows of: every caller is an
// unlocked context.
void ggml_sycl_lazy_ring_reap(lazy_reap_scope scope) noexcept;

// The scope guard of §3.3's points (ii) and (iii): declared before the try of a model's unload or load end, in an
// unlocked context, its destructor runs a FULL reap after the try/catch has returned. So the reap follows every erase
// and abort the call makes, and runs outside the try whose catch re-finalizes the ticket.
class lazy_reap_at_exit {
  public:
    lazy_reap_at_exit() noexcept = default;

    ~lazy_reap_at_exit() { ggml_sycl_lazy_ring_reap(lazy_reap_scope::FULL); }

    lazy_reap_at_exit(const lazy_reap_at_exit &)             = delete;
    lazy_reap_at_exit & operator=(const lazy_reap_at_exit &) = delete;
};

// Marks the registry shut down: from then on a take returns an empty lease and a lease's drop only decrements, so a
// lease destroyed during static destruction never touches the registry's mutex. Its production caller is the lease
// drain at backend shutdown, ggml_sycl_lazy_lease_drain in ggml-sycl.cpp, which sets it after its own FULL reap.
void lazy_io_pool_mark_shut_down() noexcept;

// Whether `generation` names a pool the registry still holds, in its slot or parked on the reap list, read under the
// leaf mutex (§3.3): the owner check's predicate for the block ring's POOL rows, whose IO_POOL owner id is that
// generation. False for 0 and once the FULL reap has freed the pool. Never called under another lock.
bool lazy_io_pool_generation_live(uint64_t generation) noexcept;

// The one wrapper a planner caller holds around its planning (§3.3): its constructor runs the entry FULL reap, take_if
// takes the provisional lease, pin() hands it to the planners, and its destructor drops the lease and then runs the
// closing FULL reap, by member order (the reap guard is declared before the lease, so it is built first and destroyed
// last). Declared in an unlocked context. It holds only the lease and pin(). Its production callers are the stage,
// ggml_backend_sycl_stage_load_inventory_plan, which declares it before its try so the closing reap runs after the
// rollback guard, and the plan dry run.
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

// ---- The IO engine's queue (§5.2.1) and the §5.3 types a read delivers through ----

class lazy_row_cache;
struct lazy_io_ticket;

// A read's priority class (§5.2.1): the prefetched ubatches are served before lookahead.
enum class lazy_io_class : uint8_t {
    PREFETCH  = 0,  // class 0
    LOOKAHEAD = 1,  // class 1 (§5.4): served only while no context has a class-0 read queued
};

// What a read feeds, the key's last term (§5.2.1): at an equal layer a lazy table's rows go before an expert slice,
// because the engram consumer runs at the head of its layer and the expert FFN runs at its end.
enum class lazy_io_kind : uint8_t {
    TABLE_ROWS   = 0,
    EXPERT_SLICE = 1,
};

// The longest read (§5.2.1): a span is queued as reads of at most this many bytes, so a read queued with an earlier key
// waits at most for the chunk a worker has already started. A row block is 4 or 8 KiB and is one read. An enumerator,
// so the bound adds no namespace-scope object (§11).
enum lazy_io_chunk : uint64_t { LAZY_IO_CHUNK_BYTES = 512ull << 10 };

// A queued read's key within its context (§5.2.1): (class, epoch, deadline, kind), compared lexicographically. The
// epoch is the ubatch's (§5.3), compared by serial-number arithmetic so the order survives the counter's wrap; the
// deadline is the layer whose execution needs the read. Keys of two contexts never compare: their epochs come from
// independent counters.
struct lazy_io_key {
    lazy_io_class cls      = lazy_io_class::PREFETCH;
    uint32_t      epoch    = 0;
    int32_t       deadline = 0;
    lazy_io_kind  kind     = lazy_io_kind::TABLE_ROWS;
};

// A read's place in its file (§5.2): a block-aligned offset and length. The in-flight table dedups on all three terms,
// so two requests join only when they ask for the same bytes. `file` is the file's identity, never a descriptor
// number: the submitter derives it from the file's device and inode, so two descriptors on one file read one id.
struct lazy_io_span {
    uint64_t file   = 0;
    uint64_t offset = 0;
    uint64_t length = 0;
};

// The slot of the landing slab a STORAGE ticket's rows land in (§5.3), indexed by the epoch mod 2. Its leaf lock guards
// the epoch and the copy: the reaper copies only while the slot's epoch is the waiter's. The memory is the context's;
// the slot only names it.
struct lazy_landing_slot {
    std::mutex lock;  // a leaf (§5.3): nothing is taken under it
    uint32_t   epoch = 0;
    uint8_t *  data  = nullptr;
    size_t     bytes = 0;
};

// One epoch's gather request (§5.3), never reused. The cancel step sets `cancelled`, and `drain` on an epoch already
// enqueued; either stops the reaper's copy into the landing slot. The gather context is T7's and opaque here.
struct lazy_gather_request {
    void *            gather_ctx = nullptr;
    uint32_t          epoch      = 0;
    lazy_io_ticket *  ticket     = nullptr;  // STORAGE only
    std::atomic<bool> cancelled{ false };
    std::atomic<bool> drain{ false };
};

// A STORAGE request's ticket (§5.3): `outstanding` counts the waiters not yet delivered, so the consumer waits for 0.
// `read_error` keeps the first failed read's errno, stored before that waiter's decrement, so a consumer that sees 0
// also sees the failure.
struct lazy_io_ticket {
    lazy_gather_request * request = nullptr;
    std::atomic<int>      outstanding{ 0 };
    std::atomic<int>      read_error{ 0 };
};

// One waiter on a read (§5.2.1, §5.3): its ticket, its epoch, and where its bytes go. `src_offset` and `bytes` select
// the waiter's bytes in the read; the landing copy goes to landing->data + landing_offset, and `row` is inserted into
// `cache` whether or not the landing copy happens. The caller keeps the ticket, the request and the landing slot alive
// until the ticket's count reaches 0.
struct lazy_io_waiter {
    lazy_io_ticket *    ticket         = nullptr;
    uint32_t            epoch          = 0;
    lazy_landing_slot * landing        = nullptr;  // null for a read with no landing, a lookahead
    size_t              landing_offset = 0;
    uint64_t            src_offset     = 0;
    size_t              bytes          = 0;
    lazy_row_cache *    cache          = nullptr;  // null for no insert
    int64_t             row            = -1;
};

// One read: queued in one context's queue, or in several after a join from another context, then dispatched to one
// worker, then completed. It holds a lease on the pool from its submission until its waiters are delivered, so the
// pool and its ring outlive every read in flight (§3.3).
struct lazy_io_read {
    lazy_io_span                span;
    int                         fd = -1;
    std::vector<lazy_io_waiter> waiters;
    ring_lease                  lease;
    bool                        dispatched   = false;
    uint64_t                    dispatch_seq = 0;  // the dispatch log's number, taken when a worker picks it up
};

#if defined(GGML_SYCL_PRIVATE_TESTING)
// One dispatch log entry (§5.2.1): which read a pick took, from which context's queue, and at which key and enqueue
// sequence number. The ordering witnesses score this log, never the completion order.
struct lazy_io_dispatch_record {
    uint64_t     dispatch_seq = 0;
    uint64_t     context      = 0;
    lazy_io_span span;
    lazy_io_key  key;
    uint64_t     enqueue_seq = 0;
};
#endif

// The one queue of §5.2.1 and its in-flight table. Class 0 keeps one queue per context, ordered by key and then by the
// enqueue sequence number, and the picks take the contexts in turn; class 1 is picked only while no class-0 read is
// queued anywhere. Not synchronized: the pool calls it under its queue lock, a leaf (§5.3), and a host test calls it on
// one thread, so the order can be exercised with no worker and no device.
class lazy_io_queue {
  public:
    lazy_io_queue();
    ~lazy_io_queue();
    lazy_io_queue(const lazy_io_queue &)             = delete;
    lazy_io_queue & operator=(const lazy_io_queue &) = delete;

    // Queues the reads that cover `span` for `context` at `key`, one per chunk of at most LAZY_IO_CHUNK_BYTES, each
    // with its share of `waiter` (the part of [src_offset, src_offset + bytes) that falls in that chunk). A chunk
    // already in flight is joined instead (§5.2.1): a queued read joined from the same context is re-keyed to the more
    // urgent of the two keys and keeps its sequence number, one joined from another context gains a second entry in the
    // joiner's queue at the joiner's key, and a dispatched read only gains the waiter. A new read takes its lease from
    // the back of `spare_leases` when one is left; the caller drops what remains after it unlocks. Each waiter added
    // counts once in its ticket's `outstanding`. Returns the waiters added; 0 for a span that is empty or not
    // block-aligned. May throw std::bad_alloc; a throw leaves every queue entry owned by a queued read and every queued
    // read with an entry, and the chunks queued before it stay queued.
    size_t submit(uint64_t                  context,
                  int                       fd,
                  const lazy_io_span &      span,
                  const lazy_io_key &       key,
                  const lazy_io_waiter &    waiter,
                  std::vector<ring_lease> & spare_leases);

    // A worker's pick (§5.2.1): the next read in the order, marked dispatched and numbered in the dispatch log, and
    // every queue entry it had removed. Null when nothing is queued. The read stays in the in-flight table, so a
    // request for it joins it, until complete() takes it out.
    lazy_io_read * pick();

    // Takes a dispatched read out of the in-flight table and hands it over: the caller delivers its waiters and then
    // destroys it, dropping its lease, after unlocking.
    std::unique_ptr<lazy_io_read> complete(lazy_io_read * read);

    // Takes every read no worker has picked out of the queues and the in-flight table and hands them over, for
    // shutdown's cancel: the caller delivers their waiters and destroys them, dropping their leases, after unlocking.
    std::vector<std::unique_ptr<lazy_io_read>> cancel_queued();

    size_t queued_entries() const;  // queue entries, a read with two entries counted twice
    size_t in_flight() const;       // reads in the in-flight table, queued or dispatched

#if defined(GGML_SYCL_PRIVATE_TESTING)
    // The test-only FIFO ordering override (§5.2.1): picks serve entries in enqueue order, whatever their class,
    // context or key. Witness 1's positive control; nothing outside a test sets it.
    void set_fifo_for_testing(bool fifo);

    const std::vector<lazy_io_dispatch_record> & dispatch_log_for_testing() const;

    // The queue's indexes checked against the reads that own their entries: empty when every class queue and FIFO
    // entry is owned by a queued read, every owned entry is in both, no queued read lacks an entry, and the class-0
    // count matches; otherwise the first discrepancy. It follows no pointer it has not found owned.
    std::string check_indexes_for_testing() const;
#endif

  private:
    struct state;
    std::unique_ptr<state> s_;
};

// The reaper's delivery of one waiter of a completed read (§5.3): under the landing slot's lock, the copy only while
// the slot's epoch is the waiter's and its request is neither cancelled nor draining; then, with no lock held, the
// row's insert into the cache, whether or not the copy happened; then the ticket's decrement. A read that failed
// (`read_errno` nonzero) copies and inserts nothing, stores the errno in the ticket if none is stored, and still
// decrements. A landing range that does not fit its live slot copies nothing and stores EINVAL the same way, with one
// WARN after the unlock. Returns whether this delivery took the ticket's count to 0.
bool lazy_io_deliver(const lazy_io_waiter & waiter, const uint8_t * read_bytes, int read_errno) noexcept;

// ---- The block ring and the IO engine (§3.3, §5.2) ----

// block_ring_ensure's answer. Every refusal but OK is typed; each name is the lazy table's refusal for it.
enum class lazy_ring_ensure_status : uint8_t {
    OK,
    UNPINNED,    // lazy-table-ring-unpinned: no live pool, so no lease pins one
    TOO_NARROW,  // lazy-table-ring-too-narrow: the need or the span does not fit the ring (§3.3, §10)
    CONCURRENT,  // lazy-table-ring-concurrent-ensure: another caller holds the ensuring state
    SHUT_DOWN,   // lazy-table-ring-shut-down: the registry is shut down
    MISMATCH,    // lazy-table-ring-mismatch: the need is not whole blocks per slot, or the ring's bytes or charge
                 // differ from the plan's (§5.2, §10's ring check)
    REFUSED,     // lazy-table-ring-refused: the planned allocation was refused
};

const char * lazy_ring_ensure_status_name(lazy_ring_ensure_status status);

// The pool's block ring for a STORAGE table (§3.3), called by the table's materialization with the plan's
// block_ring_need as `need`, the table's own read span, the ring's device, the plan's block_ring_charge and the load's
// transaction, which the ring's charge carries (§4.2). The first call allocates exactly `need` through
// lazy_io_allocate_host and starts the engine's LAZY_BLOCK_RING_QD workers, one per slot of need / LAZY_BLOCK_RING_QD
// bytes. A later call allocates nothing and accepts the live ring when it holds the need and the span and the plan's
// charge is the ring's own for the transaction whose ensure created it, and 0 for any other (§10's ring check). It
// keeps one piece of state, `ensuring`, in three steps: reserve under the registry's leaf mutex, allocate with no lock
// held, install under it; a scope guard clears it on every exit. A caller that finds it set refuses at once and never
// waits. A need that is not a multiple of LAZY_BLOCK_RING_QD x LAZY_IO_BLOCK_BYTES is refused as MISMATCH before any
// of that, never rounded up, since O_DIRECT alignment is a plan fact (§5.2). Each mismatch with the plan is a
// [HOST-PLAN-BUG] WARN naming both numbers, a concurrent call one naming the need and the span, and the shut-down and
// unpinned refusals plain WARNs; every refusal returns its typed status. Ensure never aborts: the strict abort is its
// caller's (§10). May throw what the allocation or a worker's start throws, after the guard has cleared `ensuring`.
lazy_ring_ensure_status block_ring_ensure(size_t   need,
                                          size_t   span,
                                          int      device,
                                          size_t   plan_charge,
                                          uint64_t load_txn_id = 0);

// Queues the read of `span` for `context` at `key` with `waiter` (§5.2.1, §5.3), on the live pool's engine, and wakes a
// worker. The read takes a lease on the pool through lazy_io_pool_join, so the pool and its ring outlive it. Returns
// the waiters added, as lazy_io_queue::submit counts them; 0, with a WARN, when no live pool has a ring or the span is
// longer than one slot.
size_t lazy_io_submit(uint64_t               context,
                      int                    fd,
                      const lazy_io_span &   span,
                      const lazy_io_key &    key,
                      const lazy_io_waiter & waiter);

// Backend shutdown's step (§3.3): cancels the live engine's queued reads, delivering each waiter ECANCELED, and waits
// for the reads the workers have picked to finish, their leases dropped. Shutdown is already a waiting context. It
// sets no flag and reaps nothing; the lease drain (§3.3) is to do both.
void lazy_io_pool_shutdown() noexcept;

// One coverage read (§10's lazy_ring_coverage_reads): the live ring's bytes through `pin`, counted. No pin is no
// coverage read: 0, and nothing counted.
uint64_t lazy_ring_coverage_read(const ring_lease * pin) noexcept;

// The coverage reads counted since the process started: one relaxed counter, unconditional in every build (§10).
uint64_t lazy_ring_coverage_reads() noexcept;

// The live ring's usable bytes, read under the leaf mutex: the pool's published figure (§10's accessor). False, writing
// nothing, when no live pool has a ring.
bool lazy_block_ring_bytes(uint64_t * out);

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
    uint64_t ring_bytes             = 0;      // the slot pool's ring, usable bytes; 0 with no ring
    uint64_t ring_allocations       = 0;      // rings ensure allocated for the slot pool, installed or not
    bool     ensuring               = false;  // the slot pool's ensuring state
    // The witness's counts (§3.3, §11 T5), process totals.
    uint64_t reaps_refused          = 0;  // reaps that found a witnessed lock held and reaped nothing
    uint64_t reaps_failed           = 0;  // reaps whose step threw, logged [HOST-PLAN-BUG] and returned
    uint64_t lease_drops_locked     = 0;  // held leases dropped with a witnessed lock held
    uint64_t worker_reaps           = 0;  // WORKER reaps that ran to their end
    // The join census (§3.3: a pool worker never joins itself): every join of an IO engine worker, by the joiner.
    uint64_t joins                  = 0;  // joins run
    uint64_t joins_from_worker      = 0;  // joins run on a pool worker thread
    uint64_t self_joins             = 0;  // joins a worker asked of itself, recorded and never run
    // Held leases destroyed after the registry shut down (§3.3's drain): a record's lease the drain missed, dropped by
    // static destruction or a test's clear. The drain moves every record's lease out before it marks shut-down.
    uint64_t drops_after_shut_down  = 0;
};

// Copies the registry's state into *out. Returns false, writing nothing, for a null pointer.
bool ggml_sycl_lazy_registry_for_testing(lazy_registry_state * out);

// The registry's one-shot hold hook, armed at one of seven points; arming any point replaces whatever was armed, and
// arming with a null callback disarms. The callback runs once, with no registry or IO-engine lock held (the plan
// point's caller holds its inventory guard), and blocks for as long as the fixture wants its caller held.
using lazy_hold_fn = void (*)(void * ctx);

// Arms the hook in the FULL reap: the first FULL reap that starts after the arming calls it after its count has landed
// and before it swaps the list out.
void ggml_sycl_test_lazy_reap_hold(lazy_hold_fn hold, void * ctx);

// Arms the hook in the take: the first take that reserves a generation after the arming calls it after building its
// pool and before the install, so a fixture can install and park another pool inside that window.
void ggml_sycl_test_lazy_take_hold(lazy_hold_fn hold, void * ctx);

// Arms the hook in block_ring_ensure: the first ensure that reserves `ensuring` after the arming calls it between the
// reserve and the allocation (§3.3), so a fixture can hold the reserver while another ensure runs.
void ggml_sycl_test_lazy_ensure_hold(lazy_hold_fn hold, void * ctx);

// Arms the hook at the plan point: the first planner caller that reaches it after the arming calls it once its plan is
// made and before the plan is staged, under that caller's inventory guard (g_tensor_inventory_mutex) and with no
// registry or IO-engine lock held, so a fixture can drop a lease inside that window (§3.3, the window arm). The planner
// caller reaches the point through ggml_sycl_test_lazy_plan_point().
void ggml_sycl_test_lazy_plan_hold(lazy_hold_fn hold, void * ctx);

// Arms the hook in lazy_io_pool_shutdown: the first shutdown that reaches its wait for the picked reads after the
// arming calls it once the queued reads are cancelled and before that wait, with no registry or IO-engine lock held, so
// a fixture can hold the picked reads until shutdown is about to wait for them (§3.3).
void ggml_sycl_test_lazy_shutdown_hold(lazy_hold_fn hold, void * ctx);

// Arms the hook in the coverage scope's take (§3.3's race): the first planner caller's take after the arming calls it
// after the scope's entry reap and before lazy_io_pool_take, with no registry or IO-engine lock held, so a fixture can
// park the live pool between the entry reap and the take.
void ggml_sycl_test_lazy_stage_take_hold(lazy_hold_fn hold, void * ctx);

// Arms the hook in ring_lease::reset: the first drop of a held lease after the arming calls it before the drop's
// critical section, with no registry lock held, so a fixture can hold a drop while a take joins the pool (§3.3's race).
void ggml_sycl_test_lazy_drop_hold(lazy_hold_fn hold, void * ctx);

// The plan point: claims the hook if it is armed there and calls it with no registry or IO-engine lock held; its one
// caller, compute_and_store_plan_for_inventory, holds the inventory guard.
void ggml_sycl_test_lazy_plan_point();

// Whether the calling thread holds the registry's leaf mutex (§11 T5), in the style of the retained store's
// retained_store_mutex_held(): the witness every guard of that mutex carries (take, clone, drop, park, reap, install).
bool registry_mutex_held();

// The five witnessed locks (§11 T5), numbered as the reap's positive control takes them.
enum lazy_witnessed_lock : int {
    LAZY_LOCK_REGISTRY  = 0,  // the IO pool registry's leaf mutex
    LAZY_LOCK_INVENTORY = 1,  // g_tensor_inventory_mutex
    LAZY_LOCK_PLAN      = 2,  // g_lifecycle_plan_mutex
    LAZY_LOCK_QUEUE     = 3,  // the live IO engine's queue lock
    LAZY_LOCK_STRAND    = 4,  // a table's strand lock
};

// What a holding seam did with the lock it was asked for.
enum class lazy_holding_status : uint8_t {
    HELD,       // it took the lock and ran the callback under it
    NOT_WIRED,  // the lock is not in this tree yet: a table's strand lock, which the reader's strand brings
    NO_ENGINE,  // the queue lock was asked for with no live IO engine
    UNKNOWN,    // no such lock, or one this half of the control does not take
};

const char * lazy_holding_status_name(lazy_holding_status status);

// One run of the reap's positive control: a FULL reap called on a thread that holds one witnessed lock.
struct lazy_reap_holding_result {
    lazy_holding_status status        = lazy_holding_status::UNKNOWN;
    bool                witnessed     = false;    // the witness read a lock held when the reap was called
    bool                registry_held = false;    // registry_mutex_held() when the reap was called
    bool                refused       = false;    // the reap refused: its refusal count moved
    const char *        held          = nullptr;  // the lock the witness named, or nullptr
};

// Runs `fn(ctx)` on the calling thread with LAZY_LOCK_REGISTRY or LAZY_LOCK_QUEUE held (the queue lock of the live
// IO engine, whose pool a join lease keeps alive across the call) and returns HELD; any other lock runs nothing.
lazy_holding_status ggml_sycl_test_lazy_run_holding(int lock, lazy_hold_fn fn, void * ctx);

// Calls a FULL reap on the calling thread, which holds the lock under test, and reports what the witness read and
// whether the reap refused, by its refusal count read lock-free, so the call is safe under the registry's own mutex.
// `status` is left to the caller, which took the lock.
lazy_reap_holding_result ggml_sycl_test_lazy_reap_under_held_lock();

// The pool's half of the reap's positive control: ggml_sycl_test_lazy_run_holding with the reap under it, for the
// registry's leaf mutex and the queue lock; a table's strand lock is NOT_WIRED. The full control, over all five, is
// ggml_sycl_test_lazy_ring_reap_holding in ggml-sycl.cpp.
lazy_reap_holding_result ggml_sycl_test_lazy_reap_holding_pool(int lock);

// Makes the next FULL reap's first step throw std::system_error, as std::mutex::lock can (§3.3: a failure to reset is
// logged and never thrown). One-shot.
void ggml_sycl_test_lazy_reap_throw_next();

// REAP_INSIDE_TRY's reap: the reap's passes without its catch, so the seam's std::system_error reaches the caller.
void ggml_sycl_test_lazy_ring_reap_uncaught(lazy_reap_scope scope);

// Runs `fn(ctx)` on the calling thread holding one witnessed lock (§11 T5): g_tensor_inventory_mutex itself,
// g_lifecycle_plan_mutex through the unified cache's seam, and the registry's leaf mutex and the queue lock through
// ggml_sycl_test_lazy_run_holding. A table's strand lock is NOT_WIRED. Defined in ggml-sycl.cpp.
lazy_holding_status ggml_sycl_test_lazy_run_holding_any(int lock, lazy_hold_fn fn, void * ctx);

// The reap's positive control over all five locks: a FULL reap called on a thread that holds `lock`, its status the
// holding seam's. Defined in ggml-sycl.cpp.
lazy_reap_holding_result ggml_sycl_test_lazy_ring_reap_holding(int lock);

// The engine's read source (§5.2): `length` bytes of `fd` at `offset` into `dst`; returns the bytes read, or -errno.
// A fixture's source replaces pread for every worker until it is cleared with null; it runs with no lock held.
using lazy_io_read_fn = long (*)(void * ctx, int fd, uint8_t * dst, size_t length, uint64_t offset);

void ggml_sycl_test_lazy_io_read_source(lazy_io_read_fn read, void * ctx);

// The largest end of a worker's write into the ring, as an offset from the ring's start, over every read since the
// last reset: slot offset plus read length (§11's ring-write recorder). Reset to 0 by the reset call.
uint64_t ggml_sycl_test_lazy_ring_write_end();
void     ggml_sycl_test_lazy_ring_write_end_reset();

// The registered PRIVATE_TESTING runtime mutants (one selector, §11): each plants one defect a ring arm must catch,
// selected by the arm's registration and NONE everywhere else.
enum class lazy_ring_mutant : uint8_t {
    NONE,
    NO_ENSURING,                 // ensure keeps no ensuring state, so a concurrent caller allocates too
    ENSURING_KEPT_ON_THROW,      // the guard leaves ensuring set when the allocation throws
    INSTALL_AFTER_SHUT_DOWN,     // step 3 installs the ring although the registry was shut down
    NO_SPAN_CHECK,               // the first ensure skips span <= need / QD
    LEASE_PRESENCE_AS_COVERAGE,  // the planner reads a held lease as a covering ring
    TAKE_WITHOUT_CANDIDATE,      // the coverage scope takes a lease with no lazy-table candidate
    COUNT_WITHOUT_PIN,           // a coverage read with no pin is counted
    NO_ENTRY_REAP,               // the coverage scope runs no entry reap
    NO_CLOSING_REAP,             // the coverage scope runs no closing reap
    IGNORE_LIVE_RING,            // a coverage read returns 0 for a live ring, still counted
    LEASE_AT_STAGE,              // the stage drops its lease once planned and takes the plan's past the plan point
    NO_CANDIDATE_LEASE,          // a staged plan with a STORAGE table carries no lease
    REPLACE_KEEPS_DISPLACED,     // a candidate's replacement keeps the displaced candidate's lease alive
    ROLLBACK_KEEPS_CLONE,        // the stage's rollback leaves the candidate's lease alive
    LEASE_BEFORE_GUARD,          // the stage keeps a lease built before its scope's reap guard, so it outlives the reap
    GENERATION_LIVE_IGNORES_PARKED,  // the generation predicate reads only the slot, so a parked pool reads gone
    STRICT_ABORT_DROPPED,            // the STORAGE materialization does not abort on a ring refusal under strict leases
    NO_ALIGNMENT_CHECK,              // ensure accepts a need that is not whole 4 KiB blocks per slot
    SHUTDOWN_NO_WAIT,                // shutdown returns without waiting for the reads the workers picked
    TAKE_IGNORES_DYING,              // a take joins a parked pool whose ring is still held
    WORKER_REAP_DESTROYS_POOL,       // a WORKER reap runs the parked pools' destruction, joining their workers there
    SNAPSHOT_CARRIES_LEASE,          // the published snapshot owns a lease on the pool
    REAP_INSIDE_TRY,                 // the teardown reap runs inside the try that re-finalizes the ticket
    SKIP_LEASE_DRAIN,                // shutdown moves no record's lease out before it marks the registry shut down
    POOL_STRICT_ABORT_DROPPED,       // the reap's and the drop's [HOST-PLAN-BUG] lines continue under strict leases
    DRAIN_STRICT_ABORT_DROPPED,      // the shutdown drain logs a leaked lease and continues under strict leases
    ENSURE_WITNESS_DROPPED,          // ensure runs when it is entered with a witnessed lock held
};

void ggml_sycl_test_lazy_ring_mutant_select(lazy_ring_mutant mutant);
bool lazy_ring_mutant_is(lazy_ring_mutant mutant) noexcept;

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

// ring_bytes == 0 gives a ring-less pool: every table is then STORAGE-eligible and block_ring_charge is the mapped
// need. A nonzero ring_bytes ensures that ring through block_ring_ensure on device 0, with a span of ring_bytes /
// LAZY_BLOCK_RING_QD and the charge of a creating plan, so a planner sees a live ring of that size; an ensure that
// refuses aborts naming the status. A call while a guard is live joins the live pool with a counted reference; when
// that pool holds a ring, ring_bytes must equal it and nothing is ensured, and any other ring_bytes aborts naming both.
lazy_test_coverage_pin ggml_sycl_test_make_coverage_pin(uint64_t ring_bytes);
#endif

}  // namespace ggml_sycl
