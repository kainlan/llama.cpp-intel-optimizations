// The copy-on-write chunked vector the placement plan's entries and the hybrid decode record's leases live in: a copy
// shares every chunk, a write through the copy clones only the chunk it lands in, and a const read of a shared copy
// clones nothing. ggml-sycl/cow-chunked-vector.hpp is SYCL-free, so this links nothing and opens no device.
#include "ggml-sycl/cow-chunked-vector.hpp"

#include <algorithm>
#include <cstdio>
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

static int test_index_of() {
    const store a = make_store(10);
    for (size_t i = 0; i < a.size(); ++i) {
        CHECK(a.index_of(a[i]) == i, "index_of names each element's position");
    }
    const std::string outside = "e3";
    CHECK(a.index_of(outside) == store::npos, "an element not in the store has no index");
    size_t i = 0;
    for (auto it = a.begin(); it != a.end(); ++it, ++i) {
        CHECK(it.index() == i, "an iterator names its position");
    }
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
                               test_index_of };
    for (int (*t)() : tests) {
        if (t() != 0) {
            return 1;
        }
    }
    std::printf("test-sycl-cow-chunked-vector: all passed\n");
    return 0;
}
