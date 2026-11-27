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

    void push_back(Handle h) {
        Key key;
        if (KeyOf{}(h, &key)) {
            index_.emplace(std::move(key), items_.size());
        }
        items_.push_back(std::move(h));
    }

    // Replaces the bag's handles with `handles`, indexed afresh.
    void assign(std::vector<Handle> && handles) {
        items_ = std::move(handles);
        reindex();
    }

    // Moves every handle out, leaving the bag empty.
    std::vector<Handle> take_all() {
        std::vector<Handle> out = std::move(items_);
        items_.clear();
        index_.clear();
        return out;
    }

    void clear() {
        items_.clear();
        index_.clear();
    }

    // Moves every handle on `key` into `out` and returns how many: one index lookup, and per handle a swap-remove.
    size_t take_key(const Key & key, std::vector<Handle> & out) {
        auto range = index_.equal_range(key);
        if (range.first == range.second) {
            return 0;
        }
        std::vector<size_t> at;
        for (auto it = range.first; it != range.second; ++it) {
            at.push_back(it->second);
        }
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

    // Removes every handle `pred` matches and returns how many; the index is rebuilt once.
    template <typename Pred> size_t remove_if(const Pred & pred) {
        const size_t before = items_.size();
        items_.erase(std::remove_if(items_.begin(), items_.end(), pred), items_.end());
        if (items_.size() != before) {
            reindex();
        }
        return before - items_.size();
    }

    // For the tests: whether the index names position `pos` under `key`, and how many positions it names in all.
    bool holds_at(const Key & key, size_t pos) const {
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

    void reindex() {
        index_.clear();
        index_.reserve(items_.size());
        Key key;
        for (size_t i = 0; i < items_.size(); ++i) {
            if (KeyOf{}(items_[i], &key)) {
                index_.emplace(key, i);
            }
        }
    }

    std::vector<Handle>                           items_;
    std::unordered_multimap<Key, size_t, KeyHash> index_;
};

}  // namespace ggml_sycl
