#pragma once
// An unordered bag of handles with an index by key, so every handle on one key comes out without a walk of the bag.
//
// Why: the MoE pointer table holds one lease per expert it names (moe_expert_ptrs_leases), and an expert loan or evict
// must take every lease on the device entry it retired off each table it touches. A plain vector made that a walk of
// every lease of the tensor, per member moved. Here the walk is an index lookup by the entry's key.
//
// The bag keeps its handles in one vector, so it iterates, indexes and moves out as one; its order is not meaningful
// (a removal moves the last handle into the gap). A handle the key function gives no key (not a cache entry's weight)
// is held and iterated like the rest and is never found by key. Several handles may share a key: a table recorded
// under a graph appends its leases without replacing the ones it held. Single writer, as for a std::vector.
//
// The index is built on first use (take_key) and kept until the handles are replaced. A table update replaces them
// on every prompt micro-batch and every graph compute without a keyed decode slot, so assign, take_all, clear and
// remove_if are vector moves or erasures that only drop the index; an append to an indexed bag indexes the new handle.
// Only an expert move asks by key, so a bag no move touches never builds an index.
#include <algorithm>
#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ggml_sycl {

// KeyOf: bool operator()(const Handle &, Key * out) const, true with *out set when the handle has a key.
template <typename Handle, typename Key, typename KeyHash, typename KeyOf> class keyed_handle_bag {
  public:
    using value_type     = Handle;
    using const_iterator = typename std::vector<Handle>::const_iterator;

    size_t size() const { return items_.size(); }

    bool empty() const { return items_.empty(); }

    const Handle & operator[](size_t i) const { return items_[i]; }

    const_iterator begin() const { return items_.begin(); }

    const_iterator end() const { return items_.end(); }

    // Appends `h`. An indexed bag indexes it too, after the append: an index entry that fails to allocate pops the
    // handle again, so the bag never indexes a position it does not hold and never holds an unindexed keyed handle.
    void push_back(Handle h) {
        items_.push_back(std::move(h));
        if (!indexed_) {
            return;
        }
        Key key;
        if (KeyOf{}(items_.back(), &key)) {
            try {
                index_.emplace(std::move(key), items_.size() - 1);
            } catch (...) {
                items_.pop_back();
                throw;
            }
        }
    }

    // Replaces the bag's handles with `handles`: a move, and the index is dropped.
    void assign(std::vector<Handle> && handles) {
        items_ = std::move(handles);
        drop_index();
    }

    // Moves every handle out, leaving the bag empty.
    std::vector<Handle> take_all() {
        std::vector<Handle> out = std::move(items_);
        items_.clear();
        drop_index();
        return out;
    }

    void clear() {
        items_.clear();
        drop_index();
    }

    // Moves every handle on `key` into `out` and returns how many: one index lookup, and per handle a swap-remove.
    // Everything that allocates (the index, when this is its first use; the positions; room in `out`) is done before
    // the bag changes, so a bad_alloc leaves the bag and `out` as they were. What follows only moves handles into room
    // already reserved and rewrites index entries in place.
    size_t take_key(const Key & key, std::vector<Handle> & out) {
        build_index();
        auto range = index_.equal_range(key);
        if (range.first == range.second) {
            return 0;
        }
        std::vector<size_t> at;
        for (auto it = range.first; it != range.second; ++it) {
            at.push_back(it->second);
        }
        out.reserve(out.size() + at.size());
        index_.erase(range.first, range.second);
        // Highest position first: every handle past the one removed is then on another key (or has none), so the
        // handle the swap moves down is never one still to take.
        std::sort(at.begin(), at.end(), [](size_t a, size_t b) { return a > b; });
        for (const size_t pos : at) {
            out.push_back(std::move(items_[pos]));
            remove_at(pos);
        }
        return at.size();
    }

    // Removes every handle `pred` matches and returns how many; the index is dropped.
    template <typename Pred> size_t remove_if(const Pred & pred) {
        const size_t before = items_.size();
        items_.erase(std::remove_if(items_.begin(), items_.end(), pred), items_.end());
        if (items_.size() != before) {
            drop_index();
        }
        return before - items_.size();
    }

    // For the tests: whether the index names position `pos` under `key` (building it), how many positions it names
    // now (0 when it is not built), how many handles have a key, and how many times the index was built.
    bool holds_at(const Key & key, size_t pos) {
        build_index();
        auto range = index_.equal_range(key);
        return std::any_of(range.first, range.second, [pos](const auto & kv) { return kv.second == pos; });
    }

    size_t indexed() const { return index_.size(); }

    size_t keyed_count() const {
        size_t n = 0;
        Key    key;
        for (const Handle & h : items_) {
            n += KeyOf{}(h, &key) ? 1 : 0;
        }
        return n;
    }

    size_t index_builds() const { return index_builds_; }

  private:
    // Fills the gap at `pos` with the last handle and points that handle's index entry at its new position.
    void remove_at(size_t pos) {
        const size_t last = items_.size() - 1;
        if (pos != last) {
            Key key;
            if (KeyOf{}(items_[last], &key)) {
                auto range = index_.equal_range(key);
                for (auto it = range.first; it != range.second; ++it) {
                    if (it->second == last) {
                        it->second = pos;
                        break;
                    }
                }
            }
            items_[pos] = std::move(items_[last]);
        }
        items_.pop_back();
    }

    void build_index() {
        if (indexed_) {
            return;
        }
        index_.clear();
        index_.reserve(items_.size());
        Key key;
        for (size_t i = 0; i < items_.size(); ++i) {
            if (KeyOf{}(items_[i], &key)) {
                index_.emplace(key, i);
            }
        }
        indexed_ = true;
        ++index_builds_;
    }

    void drop_index() {
        if (indexed_) {
            index_.clear();
            indexed_ = false;
        }
    }

    std::vector<Handle>                           items_;
    std::unordered_multimap<Key, size_t, KeyHash> index_;
    bool                                          indexed_      = false;
    size_t                                        index_builds_ = 0;
};

}  // namespace ggml_sycl
