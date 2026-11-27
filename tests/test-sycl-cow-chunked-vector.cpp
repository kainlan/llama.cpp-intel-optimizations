// The copy-on-write chunked vector the placement plan's entries and the hybrid decode record's leases live in: a copy
// shares every chunk, a write through the copy clones only the chunk it lands in, and a const read of a shared copy
// clones nothing. ggml-sycl/cow-chunked-vector.hpp is SYCL-free, so this links nothing and opens no device.
#include "ggml-sycl/cow-chunked-vector.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(cond, msg)                      \
    do {                                      \
        if (!(cond)) {                        \
            std::printf("FAILED: %s\n", msg); \
            return 1;                         \
        }                                     \
    } while (0)

// A small chunk, so a few dozen elements span several chunks.
using store = ggml_sycl::cow_chunked_vector<std::string, 4>;

static store make_store(size_t n) {
    store s;
    for (size_t i = 0; i < n; ++i) {
        s.push_back("e" + std::to_string(i));
    }
    return s;
}

static bool same_contents(const store & s, const std::vector<std::string> & want) {
    if (s.size() != want.size()) {
        return false;
    }
    for (size_t i = 0; i < want.size(); ++i) {
        if (s[i] != want[i]) {
            return false;
        }
    }
    return true;
}

static int test_copy_shares_chunks() {
    const store a = make_store(10);  // chunks of 4: 3 chunks
    CHECK(a.size() == 10 && a.chunk_count() == 3, "10 elements in chunks of 4 take 3 chunks");
    const store b = a;
    CHECK(b.size() == 10, "the copy has every element");
    CHECK(b.shared_chunks_with(a) == 3, "a copy shares every chunk with its source");
    CHECK(b.chunks_cloned() == 0, "copying clones nothing");
    return 0;
}

static int test_write_to_copy_leaves_original() {
    const store a = make_store(10);
    store       b = a;
    b[5]          = "changed";
    CHECK(b[5] == "changed", "the copy sees its own write");
    CHECK(a[5] == "e5", "the original keeps its element");
    CHECK(b.chunks_cloned() == 1, "one write clones the one chunk it lands in");
    CHECK(b.shared_chunks_with(a) == 2, "the copy still shares the chunks it did not write");
    b[6] = "again";
    CHECK(b.chunks_cloned() == 1, "a second write into a chunk the copy owns clones nothing");
    CHECK(a[6] == "e6", "the original is untouched by the second write");
    b.push_back("tail");
    CHECK(a.size() == 10 && b.size() == 11, "a push_back on the copy does not grow the original");
    CHECK(b.chunks_cloned() == 2, "the push_back cloned the shared last chunk it appended into");
    CHECK(a[9] == "e9" && b[10] == "tail", "the appended element is the copy's alone");

    // The source written after the copy: the copy keeps the snapshot it was taken from.
    store c = make_store(6);
    store d = c;
    c[0]    = "source";
    CHECK(d[0] == "e0", "a write to the source leaves an earlier copy as it was");
    CHECK(c.chunks_cloned() == 1, "the source clones the chunk it shares with the copy");
    return 0;
}

static int test_const_iteration_clones_nothing() {
    const store   a  = make_store(9);
    store         b  = a;
    const store & cb = b;
    size_t        n  = 0;
    for (const std::string & e : cb) {
        n += e.empty() ? 0 : 1;
    }
    CHECK(n == 9, "the const iteration visits every element");
    CHECK(b.chunks_cloned() == 0, "a const iteration of a shared copy clones nothing");
    CHECK(b.shared_chunks_with(a) == 3, "every chunk is still shared after the const iteration");
    CHECK(std::count(cb.begin(), cb.end(), std::string("e4")) == 1, "a const algorithm reads through the chunks");
    CHECK(b.chunks_cloned() == 0, "the const algorithm cloned nothing");
    return 0;
}

static int test_mutable_iteration_unshares() {
    const store a = make_store(9);
    store       b = a;
    for (std::string & e : b) {
        e += "!";
    }
    CHECK(b.chunks_cloned() == 3, "a non-const iteration unshares every chunk first");
    CHECK(b.shared_chunks_with(a) == 0, "nothing is shared after a non-const iteration");
    CHECK(a[0] == "e0" && b[0] == "e0!", "the original keeps its elements");
    return 0;
}

static int test_sort() {
    store     s;
    const int values[] = { 7, 3, 9, 1, 8, 2, 6, 0, 5, 4 };
    for (const int v : values) {
        s.push_back(std::to_string(v));
    }
    const store before = s;
    std::sort(s.begin(), s.end());
    CHECK(same_contents(s, { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" }), "std::sort orders across chunks");
    CHECK(before[0] == "7" && before[9] == "4", "sorting the copy leaves the original's order");
    std::sort(s.begin(), s.end(), [](const std::string & x, const std::string & y) { return x > y; });
    CHECK(s.front() == "9" && s.back() == "0", "std::sort with a comparator orders across chunks");
    return 0;
}

static int test_erase_resize_clear() {
    store       s        = make_store(10);
    const store snapshot = s;
    s.erase(s.begin() + 2, s.begin() + 5);
    CHECK(same_contents(s, { "e0", "e1", "e5", "e6", "e7", "e8", "e9" }), "a range erase closes the gap");
    s.erase(s.begin());
    CHECK(same_contents(s, { "e1", "e5", "e6", "e7", "e8", "e9" }), "a single erase removes the first element");
    s.erase(std::remove_if(s.begin(), s.end(), [](const std::string & e) { return e == "e7"; }), s.end());
    CHECK(same_contents(s, { "e1", "e5", "e6", "e8", "e9" }), "the erase-remove idiom works");
    CHECK(snapshot.size() == 10 && snapshot[2] == "e2", "erasing from a copy leaves the original whole");

    s.resize(8);
    CHECK(s.size() == 8 && s[7].empty() && s[4] == "e9", "resize grows with default elements");
    s.resize(2);
    CHECK(same_contents(s, { "e1", "e5" }) && s.chunk_count() == 1, "resize shrinks and drops whole chunks");
    s.clear();
    CHECK(s.empty() && s.size() == 0 && s.chunk_count() == 0, "clear empties the store");
    CHECK(snapshot.size() == 10 && snapshot[9] == "e9", "clearing a copy leaves the original");
    s.emplace_back(3, 'x');
    CHECK(s.size() == 1 && s.back() == "xxx", "emplace_back after clear starts a new chunk");
    return 0;
}

static int test_unshare_at_moves_the_clone_ahead_of_the_write() {
    const store a = make_store(10);
    store       b = a;
    b.unshare_at(5);
    CHECK(b.chunks_cloned() == 1, "unshare_at clones the chunk holding the element");
    CHECK(b.shared_chunks_with(a) == 2, "unshare_at leaves the other chunks shared");
    b.unshare_at(6);
    CHECK(b.chunks_cloned() == 1, "unshare_at on a chunk the store owns clones nothing");
    b[4] = "x";
    b[7] = "y";
    CHECK(b.chunks_cloned() == 1, "a write into an unshared chunk clones nothing");
    CHECK(a[4] == "e4" && a[7] == "e7", "the original keeps its elements");
    return 0;
}

static int test_iterator_index() {
    const store a = make_store(10);
    size_t      i = 0;
    for (auto it = a.begin(); it != a.end(); ++it, ++i) {
        CHECK(it.index() == i, "an iterator names its position");
    }
    return 0;
}

// erase hands back a mutable iterator even when it removes nothing, so it unshares first, as begin() does: a write
// through that iterator must not reach another store that shares the chunk.
static int test_empty_erase_unshares() {
    const store   a  = make_store(10);
    store         b  = a;
    const store & cb = b;
    auto          it = b.erase(cb.begin() + 3, cb.begin() + 3);
    CHECK(b.size() == 10 && it.index() == 3, "an empty erase removes nothing and returns its position");
    *it = "written";
    CHECK(cb[3] == "written", "the write lands in the store erase was called on");
    CHECK(a[3] == "e3", "the write does not reach the store it was copied from");
    CHECK(b.chunks_cloned() == 3 && b.shared_chunks_with(a) == 0, "the mutable iterator unshared every chunk first");
    return 0;
}

// chunks_cloned() counts the clones made for the store's current contents: copy assignment and clear() replace the
// contents wholesale, so both start the count again.
static int test_clear_resets_the_clone_count() {
    const store a = make_store(10);
    store       b = a;
    b[1]          = "x";
    CHECK(b.chunks_cloned() == 1, "the write cloned one chunk");
    b.clear();
    CHECK(b.chunks_cloned() == 0, "clear starts the count again with the contents");
    b.push_back("fresh");
    b[0] = "y";
    CHECK(b.chunks_cloned() == 0, "chunks the cleared store allocates itself are never clones");
    CHECK(a[1] == "e1" && a.size() == 10, "clearing a copy leaves the original");
    return 0;
}

// Copy assignment replaces the contents, so it starts the clone count again; a move carries the count to the target
// and leaves the source empty.
static int test_assignment_and_move_counts() {
    const store a = make_store(10);
    store       b = a;
    b[0]          = "x";
    CHECK(b.chunks_cloned() == 1, "the write cloned one chunk");
    b = a;
    CHECK(b.chunks_cloned() == 0 && b.shared_chunks_with(a) == 3, "copy assignment shares again and counts from 0");

    b[5] = "y";
    store c(std::move(b));
    CHECK(c.chunks_cloned() == 1 && c[5] == "y", "a move carries the count and the contents");
    CHECK(b.size() == 0 && b.chunk_count() == 0 && b.chunks_cloned() == 0, "the moved-from store is empty");
    store d = make_store(2);
    d       = std::move(c);
    CHECK(d.chunks_cloned() == 1 && d.size() == 10, "move assignment carries the count and the contents");
    CHECK(c.size() == 0 && c.chunk_count() == 0 && c.chunks_cloned() == 0, "the moved-from store is empty");
    return 0;
}

// Non-const front() and back() hand out a writable reference, so each clones the one chunk it reaches.
static int test_front_back_clone() {
    const store a = make_store(10);
    store       b = a;
    b.front()     = "first";
    CHECK(b.chunks_cloned() == 1 && a.front() == "e0", "front() clones the first chunk alone");
    b.back() = "last";
    CHECK(b.chunks_cloned() == 2 && a.back() == "e9", "back() clones the last chunk alone");
    CHECK(b.shared_chunks_with(a) == 1, "the middle chunk stays shared");
    return 0;
}

// An element whose constructor throws, so a test can fail an append at a chunk boundary.
static int g_throw_on = -1;

struct throwing_element {
    int value = 0;

    throwing_element() = default;

    explicit throwing_element(int v) : value(v) {
        if (v == g_throw_on) {
            throw std::runtime_error("element constructor");
        }
    }
};

// An append whose element constructor throws leaves the store as it was, at a chunk boundary as well as inside a chunk.
static int test_throwing_append_leaves_the_store() {
    ggml_sycl::cow_chunked_vector<throwing_element, 4> s;
    for (int i = 0; i < 4; ++i) {
        s.emplace_back(i);  // one full chunk: the next append opens a new one
    }
    g_throw_on = 4;
    bool threw = false;
    try {
        s.emplace_back(4);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    CHECK(threw && s.size() == 4 && s.chunk_count() == 1, "a throw at a chunk boundary leaves no empty chunk");
    g_throw_on = -1;
    s.emplace_back(5);
    s.emplace_back(6);
    g_throw_on = 7;
    threw      = false;
    try {
        s.emplace_back(7);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    g_throw_on      = -1;
    const auto & cs = s;
    CHECK(threw && s.size() == 6 && s.chunk_count() == 2, "a throw inside a chunk leaves the store's size");
    CHECK(cs[4].value == 5 && cs[5].value == 6, "the elements after the failed append are where they belong");

    // Inside a shared last chunk the append clones that chunk before it constructs, so a throw there leaves the copy's
    // elements and size and the source as they were, with the clone already taken and counted.
    ggml_sycl::cow_chunked_vector<throwing_element, 4> copy          = s;
    const size_t                                       source_cloned = s.chunks_cloned();
    const auto &                                       ccopy         = copy;
    CHECK(copy.chunks_cloned() == 0 && copy.shared_chunks_with(s) == 2, "the copy shares both chunks");
    g_throw_on = 8;
    threw      = false;
    try {
        copy.emplace_back(8);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    g_throw_on = -1;
    CHECK(threw && copy.size() == 6 && copy.chunk_count() == 2, "a throw in a shared chunk leaves the copy's size");
    for (size_t i = 0; i < 6; ++i) {
        CHECK(ccopy[i].value == cs[i].value, "a throw in a shared chunk leaves the copy's elements");
    }
    CHECK(copy.chunks_cloned() == 1 && copy.shared_chunks_with(s) == 1,
          "the failed append cloned the shared last chunk first, and counted it");
    CHECK(s.size() == 6 && s.chunks_cloned() == source_cloned && cs[4].value == 5 && cs[5].value == 6,
          "the source is unchanged");
    return 0;
}

int main() {
    int (*const tests[])() = { test_copy_shares_chunks,
                               test_write_to_copy_leaves_original,
                               test_const_iteration_clones_nothing,
                               test_mutable_iteration_unshares,
                               test_sort,
                               test_erase_resize_clear,
                               test_unshare_at_moves_the_clone_ahead_of_the_write,
                               test_iterator_index,
                               test_empty_erase_unshares,
                               test_clear_resets_the_clone_count,
                               test_assignment_and_move_counts,
                               test_front_back_clone,
                               test_throwing_append_leaves_the_store };
    for (int (*t)() : tests) {
        if (t() != 0) {
            return 1;
        }
    }
    std::printf("test-sycl-cow-chunked-vector: all passed\n");
    return 0;
}
