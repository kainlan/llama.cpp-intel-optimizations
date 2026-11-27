#pragma once
// A vector whose elements live in fixed-size chunks that copies share, so copying it costs one shared pointer per chunk
// and a write clones only the chunk it lands in.
//
// Why: a published snapshot (the placement plan, the hybrid decode record) is never written, because readers on other
// threads may still hold it. Every change is made to a copy that is published in its place. A plain std::vector made
// that copy O(elements) per change: with 512 experts the plan is ~75k entries, and an expert move changes a few
// hundred of them. With this store the copy is O(chunks) and the change clones O(chunks written).
//
// Rules a user relies on:
// - Const access never clones: operator[] const, front/back const, the const iterators, index_of, size.
// - Non-const operator[], front() and back() clone the one chunk they reach when another store shares it, so a write
//   can allocate (and throw) where a std::vector write could not. unshare_at() takes that clone ahead of the write.
// - Non-const begin()/end() clone every shared chunk first, because a mutable iterator can write anywhere. A range-for
//   over a non-const store, even with a const element reference, takes that path: iterate a const reference to the
//   store (or index it) where a shared copy must stay shared.
// - push_back, emplace_back, pop_back, resize, erase and clear change only this store.
//
// Single writer per store: the "is this chunk shared" test is use_count() == 1. A count of 1 means no other store
// holds the chunk, and no other store can start sharing it except by copying this store, which is a read of this
// store that must not race its writer (the same rule as std::vector). A count read as 2 that is falling to 1 only
// causes an unneeded clone. So a store has one writer at a time, and a copy taken for a publication is written only by
// the thread that took it, before it is published.
//
// Aliasing hazard: a reference obtained through a non-const access is into a chunk this store owns alone at that
// moment. If the store is then copied, the chunk becomes shared and writing through the old reference writes into the
// copy too. Take references after the copy, not before.
#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

namespace ggml_sycl {

template <typename T, size_t CHUNK = 32> class cow_chunked_vector {
    static_assert(CHUNK > 0, "a chunk holds at least one element");

    using chunk_type = std::vector<T>;

  public:
    using value_type      = T;
    using size_type       = size_t;
    using difference_type = std::ptrdiff_t;
    using reference       = T &;
    using const_reference = const T &;

    static constexpr size_t npos       = static_cast<size_t>(-1);
    static constexpr size_t chunk_size = CHUNK;

    // A random-access iterator by position. A mutable one is handed out only after every chunk is unshared, so it
    // reads the chunk directly.
    template <bool IS_CONST> class iterator_base {
        using owner_type = std::conditional_t<IS_CONST, const cow_chunked_vector, cow_chunked_vector>;

      public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type        = T;
        using difference_type   = std::ptrdiff_t;
        using pointer           = std::conditional_t<IS_CONST, const T *, T *>;
        using reference         = std::conditional_t<IS_CONST, const T &, T &>;

        iterator_base() = default;

        iterator_base(owner_type * owner, size_t index) : owner_(owner), index_(index) {}

        // A mutable iterator converts to a const one.
        template <bool OTHER_CONST, typename = std::enable_if_t<IS_CONST && !OTHER_CONST>>
        iterator_base(const iterator_base<OTHER_CONST> & other) : owner_(other.owner_), index_(other.index_) {}

        reference operator*() const { return owner_->element_unchecked(index_); }

        pointer operator->() const { return &owner_->element_unchecked(index_); }

        reference operator[](difference_type n) const { return owner_->element_unchecked(index_ + n); }

        // The element's position in its store.
        size_t index() const { return index_; }

        iterator_base & operator++() {
            ++index_;
            return *this;
        }

        iterator_base operator++(int) {
            iterator_base before = *this;
            ++index_;
            return before;
        }

        iterator_base & operator--() {
            --index_;
            return *this;
        }

        iterator_base operator--(int) {
            iterator_base before = *this;
            --index_;
            return before;
        }

        iterator_base & operator+=(difference_type n) {
            index_ += n;
            return *this;
        }

        iterator_base & operator-=(difference_type n) {
            index_ -= n;
            return *this;
        }

        friend iterator_base operator+(iterator_base it, difference_type n) { return it += n; }

        friend iterator_base operator+(difference_type n, iterator_base it) { return it += n; }

        friend iterator_base operator-(iterator_base it, difference_type n) { return it -= n; }

        friend difference_type operator-(const iterator_base & a, const iterator_base & b) {
            return static_cast<difference_type>(a.index_) - static_cast<difference_type>(b.index_);
        }

        friend bool operator==(const iterator_base & a, const iterator_base & b) { return a.index_ == b.index_; }

        friend bool operator!=(const iterator_base & a, const iterator_base & b) { return a.index_ != b.index_; }

        friend bool operator<(const iterator_base & a, const iterator_base & b) { return a.index_ < b.index_; }

        friend bool operator>(const iterator_base & a, const iterator_base & b) { return a.index_ > b.index_; }

        friend bool operator<=(const iterator_base & a, const iterator_base & b) { return a.index_ <= b.index_; }

        friend bool operator>=(const iterator_base & a, const iterator_base & b) { return a.index_ >= b.index_; }

      private:
        template <bool> friend class iterator_base;

        owner_type * owner_ = nullptr;
        size_t       index_ = 0;
    };

    using iterator       = iterator_base<false>;
    using const_iterator = iterator_base<true>;

    cow_chunked_vector() = default;

    // A copy shares every chunk and has cloned nothing yet.
    cow_chunked_vector(const cow_chunked_vector & other) : chunks_(other.chunks_), size_(other.size_) {}

    cow_chunked_vector & operator=(const cow_chunked_vector & other) {
        if (this != &other) {
            chunks_ = other.chunks_;
            size_   = other.size_;
            cloned_ = 0;
        }
        return *this;
    }

    cow_chunked_vector(cow_chunked_vector && other) noexcept :
        chunks_(std::move(other.chunks_)),
        size_(std::exchange(other.size_, 0)),
        cloned_(std::exchange(other.cloned_, 0)) {}

    cow_chunked_vector & operator=(cow_chunked_vector && other) noexcept {
        if (this != &other) {
            chunks_ = std::move(other.chunks_);
            size_   = std::exchange(other.size_, 0);
            cloned_ = std::exchange(other.cloned_, 0);
        }
        return *this;
    }

    size_t size() const { return size_; }

    bool empty() const { return size_ == 0; }

    const T & operator[](size_t i) const { return element_unchecked(i); }

    T & operator[](size_t i) { return (*unique_chunk(i / CHUNK))[i % CHUNK]; }

    const T & front() const { return (*this)[0]; }

    const T & back() const { return (*this)[size_ - 1]; }

    T & front() { return (*this)[0]; }

    T & back() { return (*this)[size_ - 1]; }

    const_iterator begin() const { return const_iterator(this, 0); }

    const_iterator end() const { return const_iterator(this, size_); }

    const_iterator cbegin() const { return begin(); }

    const_iterator cend() const { return end(); }

    iterator begin() {
        unshare_all();
        return iterator(this, 0);
    }

    iterator end() {
        unshare_all();
        return iterator(this, size_);
    }

    // Clones the chunk holding element i now, when another store shares it, so a later write there allocates nothing.
    // A caller that must not fail part-way through a series of writes takes the clones first.
    void unshare_at(size_t i) { (void) unique_chunk(i / CHUNK); }

    // Room for n elements' chunk pointers; the chunks themselves are allocated whole as they are first written.
    void reserve(size_t n) { chunks_.reserve((n + CHUNK - 1) / CHUNK); }

    void clear() {
        chunks_.clear();
        size_ = 0;
    }

    void push_back(const T & value) { emplace_back(value); }

    void push_back(T && value) { emplace_back(std::move(value)); }

    template <typename... Args> T & emplace_back(Args &&... args) {
        if (size_ % CHUNK == 0) {
            auto fresh = std::make_shared<chunk_type>();
            fresh->reserve(CHUNK);
            chunks_.push_back(std::move(fresh));
        }
        chunk_type & last = *unique_chunk(chunks_.size() - 1);
        last.emplace_back(std::forward<Args>(args)...);
        ++size_;
        return last.back();
    }

    void pop_back() {
        unique_chunk(chunks_.size() - 1)->pop_back();
        --size_;
        if (size_ % CHUNK == 0) {
            chunks_.pop_back();
        }
    }

    void resize(size_t n) {
        while (size_ > n) {
            pop_back();
        }
        while (size_ < n) {
            emplace_back();
        }
    }

    // Removes [first, last) and closes the gap, as std::vector::erase does. Returns the position after the last
    // element removed.
    iterator erase(const_iterator first, const_iterator last) {
        const size_t from = first.index();
        const size_t to   = last.index();
        if (from == to) {
            return iterator(this, from);
        }
        unshare_all();
        size_t dst = from;
        for (size_t src = to; src < size_; ++src, ++dst) {
            element_unchecked(dst) = std::move(element_unchecked(src));
        }
        while (size_ > dst) {
            pop_back();
        }
        return iterator(this, from);
    }

    iterator erase(const_iterator pos) { return erase(pos, pos + 1); }

    // The position of the element `value` refers to, or npos when it is not one of this store's elements. Walks the
    // chunks: for a caller with a position at hand (an index or an iterator) that is the cheaper answer.
    size_t index_of(const T & value) const {
        const T *                  p = &value;
        const std::less<const T *> less{};
        for (size_t c = 0; c < chunks_.size(); ++c) {
            const chunk_type & chunk = *chunks_[c];
            if (chunk.empty()) {
                continue;
            }
            const T * first = chunk.data();
            const T * last  = first + chunk.size();
            if (!less(p, first) && less(p, last)) {
                return c * CHUNK + static_cast<size_t>(p - first);
            }
        }
        return npos;
    }

    size_t chunk_count() const { return chunks_.size(); }

    // The chunks this store has cloned since it was constructed or last assigned: the cost of its copy-on-write. A copy
    // starts at 0.
    size_t chunks_cloned() const { return cloned_; }

    // How many of this store's chunks `other` holds too, chunk by chunk at the same position.
    size_t shared_chunks_with(const cow_chunked_vector & other) const {
        size_t       n     = 0;
        const size_t count = std::min(chunks_.size(), other.chunks_.size());
        for (size_t c = 0; c < count; ++c) {
            n += chunks_[c] == other.chunks_[c] ? 1 : 0;
        }
        return n;
    }

  private:
    const T & element_unchecked(size_t i) const { return (*chunks_[i / CHUNK])[i % CHUNK]; }

    T & element_unchecked(size_t i) { return (*chunks_[i / CHUNK])[i % CHUNK]; }

    // The chunk at `c`, cloned first when another store shares it (single writer: see the header comment).
    chunk_type * unique_chunk(size_t c) {
        std::shared_ptr<chunk_type> & held = chunks_[c];
        if (held.use_count() != 1) {
            auto clone = std::make_shared<chunk_type>();
            clone->reserve(CHUNK);
            clone->insert(clone->end(), held->begin(), held->end());
            held = std::move(clone);
            ++cloned_;
        }
        return held.get();
    }

    void unshare_all() {
        for (size_t c = 0; c < chunks_.size(); ++c) {
            (void) unique_chunk(c);
        }
    }

    std::vector<std::shared_ptr<chunk_type>> chunks_;
    size_t                                   size_   = 0;
    size_t                                   cloned_ = 0;
};

}  // namespace ggml_sycl
