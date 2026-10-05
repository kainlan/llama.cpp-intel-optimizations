#pragma once

// Ordered address-range index: "which registered range contains this address", in O(log n).
//
// The runtime allocation registry (unified-cache.cpp) is keyed by an allocation's base pointer, so it answers an exact
// pointer in O(1) but answers a pointer INSIDE an allocation only by scanning every row. A 512-expert MoE registers tens
// of thousands of rows and the per-op tensor lookups made that scan 61.65% of decode CPU (llama.cpp-ii25). This
// structure sits beside the registry and answers the containment query.
//
// PRECEDENCE. Ranges nest: an arena or cache chunk is registered as well as the suballocations carved from it, so more
// than one range can contain an address. find_innermost() returns the one with the GREATEST BASE. In a properly nested
// family that is the smallest enclosing range (the suballocation, never the chunk around it), which is what every
// caller needs: each wants the allocation that is the authority for its pointer, and a containing chunk is physical
// lifetime ownership, not authority over a suballocation inside it (see mem_handle's legacy bridge). It is a total,
// deterministic order, so it also decides partially overlapping ranges (a stale RELEASING row whose recycled address now
// sits under a new allocation): the later-opening range wins. Bases are unique: the registry cannot hold two rows at one
// pointer, and insert() refuses a second range at a base.
//
// COST. A treap keyed by base whose nodes carry the maximum end of their subtree. Every operation is O(log n) expected,
// with no dependence on nesting depth or on how many sibling ranges sit between a suballocation and its chunk; that is why
// the lookup is not a plain upper_bound plus a walk back over enclosing ranges, which is O(n) when a chunk holds many
// suballocations. Priorities come from a fixed-seed generator that never sees the keys, so the shape is balanced for any
// insertion order.
//
// THREAD SAFETY. None. The owner serialises every call; the registry holds g_runtime_alloc_mutex.
//
// Plain C++ with no SYCL dependency, so tests/test-address-range-index.cpp builds it without a device.

#include <cstddef>
#include <cstdint>

namespace ggml_sycl {

class address_range_index {
  public:
    struct entry {
        uintptr_t base = 0;
        uintptr_t end  = 0;  // one past the last byte; clamped to UINTPTR_MAX rather than wrapped
        void *    key  = nullptr;
    };

    address_range_index() = default;

    ~address_range_index() { destroy(root_); }

    address_range_index(const address_range_index &)             = delete;
    address_range_index & operator=(const address_range_index &) = delete;

    // Adds [base, base + size) owned by `key`. Returns false, changing nothing, for an empty range or a base that is
    // already present. Throws std::bad_alloc, changing nothing.
    bool insert(uintptr_t base, size_t size, void * key) {
        if (size == 0 || contains_base(base)) {
            return false;
        }
        node * n   = new node();
        n->e.base  = base;
        n->e.end   = base + size < base ? UINTPTR_MAX : base + size;
        n->e.key   = key;
        n->max_end = n->e.end;
        n->prio    = next_priority();
        node * lo  = nullptr;
        node * hi  = nullptr;
        split(root_, base, &lo, &hi);  // lo: bases < base, hi: bases >= base
        root_ = merge(merge(lo, n), hi);
        count_++;
        return true;
    }

    // Removes the range at `base` if it is owned by `key`; a row owned by another key is left alone, so a stale erase can
    // never remove a newer row's entry.
    bool erase(uintptr_t base, void * key) noexcept {
        node * n = root_;
        while (n != nullptr && n->e.base != base) {
            n = base < n->e.base ? n->left : n->right;
        }
        if (n == nullptr || n->e.key != key) {
            return false;
        }
        root_ = erase_node(root_, base);
        count_--;
        return true;
    }

    // The range with the greatest base among those containing `addr`; see PRECEDENCE above.
    bool find_innermost(uintptr_t addr, entry * out) const noexcept {
        const node * n = stab_rightmost(root_, addr);
        if (n == nullptr) {
            return false;
        }
        if (out != nullptr) {
            *out = n->e;
        }
        return true;
    }

    // The range whose base is exactly `base`.
    bool find_exact(uintptr_t base, entry * out) const noexcept {
        const node * n = root_;
        while (n != nullptr && n->e.base != base) {
            n = base < n->e.base ? n->left : n->right;
        }
        if (n == nullptr) {
            return false;
        }
        if (out != nullptr) {
            *out = n->e;
        }
        return true;
    }

    size_t size() const noexcept { return count_; }

    // Walks the whole tree: key order, heap order, subtree max_end, and the node count. O(n); for tests and
    // consistency audits, never a hot path.
    bool check_invariants() const noexcept {
        size_t seen = 0;
        return check(root_, nullptr, nullptr, &seen) && seen == count_;
    }

  private:
    struct node {
        entry     e;
        uintptr_t max_end = 0;
        uint64_t  prio    = 0;
        node *    left    = nullptr;
        node *    right   = nullptr;
    };

    node *   root_  = nullptr;
    size_t   count_ = 0;
    uint64_t rng_   = 0x9e3779b97f4a7c15ull;

    uint64_t next_priority() noexcept {
        rng_ ^= rng_ << 13;
        rng_ ^= rng_ >> 7;
        rng_ ^= rng_ << 17;
        return rng_;
    }

    bool contains_base(uintptr_t base) const noexcept { return find_exact(base, nullptr); }

    static void destroy(node * n) noexcept {
        while (n != nullptr) {
            destroy(n->left);
            node * next = n->right;
            delete n;
            n = next;
        }
    }

    static void update(node * n) noexcept {
        uintptr_t m = n->e.end;
        if (n->left != nullptr && n->left->max_end > m) {
            m = n->left->max_end;
        }
        if (n->right != nullptr && n->right->max_end > m) {
            m = n->right->max_end;
        }
        n->max_end = m;
    }

    // Splits t into bases < base (*lo) and bases >= base (*hi).
    static void split(node * t, uintptr_t base, node ** lo, node ** hi) noexcept {
        if (t == nullptr) {
            *lo = nullptr;
            *hi = nullptr;
            return;
        }
        if (t->e.base < base) {
            split(t->right, base, &t->right, hi);
            *lo = t;
            update(t);
        } else {
            split(t->left, base, lo, &t->left);
            *hi = t;
            update(t);
        }
    }

    // Every base in a is below every base in b.
    static node * merge(node * a, node * b) noexcept {
        if (a == nullptr) {
            return b;
        }
        if (b == nullptr) {
            return a;
        }
        if (a->prio > b->prio) {
            a->right = merge(a->right, b);
            update(a);
            return a;
        }
        b->left = merge(a, b->left);
        update(b);
        return b;
    }

    static node * erase_node(node * t, uintptr_t base) noexcept {
        if (t == nullptr) {
            return nullptr;
        }
        if (t->e.base == base) {
            node * merged = merge(t->left, t->right);
            delete t;
            return merged;
        }
        if (base < t->e.base) {
            t->left = erase_node(t->left, base);
        } else {
            t->right = erase_node(t->right, base);
        }
        update(t);
        return t;
    }

    // The node with the greatest base <= addr whose end > addr, within t.
    //
    // Nodes with base > addr are never candidates, so the descent follows the search path for addr; at each node on it
    // the right subtree is tried first (greater bases), then the node, then the left subtree. A failed right-subtree
    // probe is a max_end test that costs O(1) unless that subtree straddles addr, which only the path nodes do; the left
    // subtree of a path node lies wholly at or below addr, so max_end > addr there guarantees a hit found in one
    // straight descent. The total is O(depth), whatever the nesting.
    static const node * stab_rightmost(const node * t, uintptr_t addr) noexcept {
        if (t == nullptr || t->max_end <= addr) {
            return nullptr;
        }
        if (t->e.base > addr) {
            return stab_rightmost(t->left, addr);
        }
        if (const node * hit = stab_rightmost(t->right, addr)) {
            return hit;
        }
        if (t->e.end > addr) {
            return t;
        }
        return stab_rightmost(t->left, addr);
    }

    static bool check(const node * t, const uintptr_t * lo, const uintptr_t * hi, size_t * seen) noexcept {
        if (t == nullptr) {
            return true;
        }
        (*seen)++;
        if ((lo != nullptr && t->e.base <= *lo) || (hi != nullptr && t->e.base >= *hi) || t->e.end <= t->e.base) {
            return false;
        }
        uintptr_t m = t->e.end;
        if (t->left != nullptr) {
            if (t->left->prio > t->prio) {
                return false;
            }
            m = t->left->max_end > m ? t->left->max_end : m;
        }
        if (t->right != nullptr) {
            if (t->right->prio > t->prio) {
                return false;
            }
            m = t->right->max_end > m ? t->right->max_end : m;
        }
        if (m != t->max_end) {
            return false;
        }
        return check(t->left, lo, &t->e.base, seen) && check(t->right, &t->e.base, hi, seen);
    }
};

}  // namespace ggml_sycl
