// The keyed handle bag the MoE pointer table's per-expert leases live in: an unordered bag of handles with an index
// by key, so an expert move takes every lease on one cache entry off the table without walking the bag.
// ggml-sycl/keyed-handle-bag.hpp is SYCL-free, so this links nothing and opens no device.
#include "ggml-sycl/keyed-handle-bag.hpp"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <map>
#include <random>
#include <utility>
#include <vector>

#define CHECK(cond, msg)                      \
    do {                                      \
        if (!(cond)) {                        \
            std::printf("FAILED: %s\n", msg); \
            return 1;                         \
        }                                     \
    } while (0)

// A stand-in handle: a key when it is keyed (a weight's), and a serial so the tests can tell copies apart.
struct fake_handle {
    int  key    = -1;  // -1: no key (not a weight)
    int  serial = 0;
    bool valid  = true;
};

struct fake_key_of {
    bool operator()(const fake_handle & h, int * key) const {
        if (h.key < 0) {
            return false;
        }
        *key = h.key;
        return true;
    }
};

using bag = ggml_sycl::keyed_handle_bag<fake_handle, int, std::hash<int>, fake_key_of>;

static std::vector<int> serials(bag & b) {
    std::vector<int> out;
    for (const fake_handle & h : b) {
        out.push_back(h.serial);
    }
    std::sort(out.begin(), out.end());
    return out;
}

static bool index_matches(bag & b) {
    // Every keyed handle is found through its key, at its position.
    for (size_t i = 0; i < b.size(); ++i) {
        int key = -1;
        if (!fake_key_of{}(b[i], &key)) {
            continue;
        }
        if (!b.holds_at(key, i)) {
            return false;
        }
    }
    return b.indexed() == b.keyed_count();
}

static int test_take_key_moves_every_lease_on_the_key() {
    bag b;
    b.push_back({ 7, 1 });
    b.push_back({ 3, 2 });
    b.push_back({ 7, 3 });  // a second lease on the same entry (a table recorded twice under a graph)
    b.push_back({ -1, 4 });
    b.push_back({ 5, 5 });
    std::vector<fake_handle> out;
    CHECK(b.take_key(7, out) == 2, "both leases on the key are taken");
    CHECK(out.size() == 2, "the taken leases are moved out");
    CHECK(serials(b) == std::vector<int>({ 2, 4, 5 }), "every other lease stays");
    CHECK(index_matches(b), "the index follows the swap-removes");
    CHECK(b.take_key(7, out) == 0, "a key no longer held takes nothing");
    CHECK(b.take_key(9, out) == 0 && out.size() == 2, "an unknown key takes nothing");
    return 0;
}

static int test_take_key_with_the_last_element_on_the_key() {
    bag b;
    for (int i = 0; i < 6; ++i) {
        b.push_back({ i % 2, i });  // keys 0,1,0,1,0,1: the last element is on key 1
    }
    std::vector<fake_handle> out;
    CHECK(b.take_key(1, out) == 3, "every lease on the key is taken, the last element's included");
    CHECK(serials(b) == std::vector<int>({ 0, 2, 4 }), "the other key's leases stay");
    CHECK(index_matches(b), "the index is whole after taking the tail's key");
    CHECK(b.take_key(0, out) == 3 && b.empty(), "the other key empties the bag");
    CHECK(b.indexed() == 0, "an empty bag indexes nothing");
    return 0;
}

static int test_assign_take_all_and_clear() {
    bag                      b;
    std::vector<fake_handle> v = {
        { 1,  1 },
        { 2,  2 },
        { -1, 3 },
    };
    b.assign(std::move(v));
    CHECK(b.size() == 3 && index_matches(b), "assign indexes what it is given");
    std::vector<fake_handle> all = b.take_all();
    CHECK(all.size() == 3 && b.empty() && b.indexed() == 0, "take_all empties the bag and its index");
    b.push_back({ 4, 4 });
    b.clear();
    CHECK(b.empty() && b.indexed() == 0, "clear empties the bag and its index");
    return 0;
}

static int test_remove_if_reindexes() {
    bag b;
    for (int i = 0; i < 8; ++i) {
        b.push_back({ i, i });
    }
    const size_t removed = b.remove_if([](const fake_handle & h) { return h.serial % 3 == 0; });
    CHECK(removed == 3, "remove_if removes each match");
    CHECK(serials(b) == std::vector<int>({ 1, 2, 4, 5, 7 }), "remove_if keeps the rest");
    CHECK(index_matches(b), "remove_if leaves the index whole");
    std::vector<fake_handle> out;
    CHECK(b.take_key(4, out) == 1 && out.back().serial == 4, "a key after remove_if is still found");
    return 0;
}

// The index is built on first use and kept until the bag's handles are replaced: a table update (assign, take_all,
// clear, remove_if) is a move with no index work, and a run of take_key calls between updates builds it once.
static int test_index_is_built_lazily() {
    bag b;
    for (int i = 0; i < 6; ++i) {
        b.push_back({ i % 3, i });
    }
    CHECK(b.index_builds() == 0, "appending builds no index");
    std::vector<fake_handle> out;
    CHECK(b.take_key(1, out) == 2 && b.index_builds() == 1, "the first take_key builds the index");
    CHECK(b.take_key(2, out) == 2 && b.index_builds() == 1, "a second take_key reuses it");
    b.push_back({ 2, 9 });
    CHECK(b.take_key(2, out) == 1 && b.index_builds() == 1, "an append to an indexed bag keeps the index current");
    CHECK(index_matches(b) && b.index_builds() == 1, "the kept index names every keyed handle where it is");

    std::vector<fake_handle> v = {
        { 1, 20 },
        { 1, 21 },
        { 4, 22 },
    };
    b.assign(std::move(v));
    CHECK(b.indexed() == 0, "assign drops the index");
    CHECK(b.take_key(1, out) == 2 && b.index_builds() == 2, "the next take_key builds it again, over the new handles");
    (void) b.remove_if([](const fake_handle & h) { return h.serial == 22; });
    CHECK(b.indexed() == 0 && b.take_key(4, out) == 0 && b.index_builds() == 3, "remove_if drops it too");
    (void) b.take_all();
    CHECK(b.indexed() == 0, "take_all drops it");
    return 0;
}

// A long random run of every operation, checked after each step against a std::multimap of (key, serial) and a list
// of the keyless handles: what each take_key moves out and what the bag holds must match the reference.
static int test_random_operations_match_a_reference() {
    std::mt19937             rng(20261127);
    bag                      b;
    std::multimap<int, int>  keyed;    // key -> serial
    std::vector<int>         keyless;  // serials of handles with no key
    int                      next_serial = 0;
    std::vector<fake_handle> out;
    const auto               reference_serials = [&]() {
        std::vector<int> all = keyless;
        for (const auto & kv : keyed) {
            all.push_back(kv.second);
        }
        std::sort(all.begin(), all.end());
        return all;
    };
    for (int step = 0; step < 20000; ++step) {
        const int op  = static_cast<int>(rng() % 100);
        const int key = static_cast<int>(rng() % 9) - 1;  // -1: no key
        if (op < 50) {
            const fake_handle h{ key, next_serial++ };
            b.push_back(h);
            if (key < 0) {
                keyless.push_back(h.serial);
            } else {
                keyed.emplace(key, h.serial);
            }
        } else if (op < 85) {
            if (key < 0) {
                continue;
            }
            out.clear();
            const size_t     n = b.take_key(key, out);
            std::vector<int> got;
            for (const fake_handle & h : out) {
                got.push_back(h.serial);
                CHECK(h.key == key, "take_key moves out only handles on the key");
            }
            std::vector<int> want;
            const auto       range = keyed.equal_range(key);
            for (auto it = range.first; it != range.second; ++it) {
                want.push_back(it->second);
            }
            keyed.erase(range.first, range.second);
            std::sort(got.begin(), got.end());
            std::sort(want.begin(), want.end());
            CHECK(n == want.size() && got == want, "take_key moves out exactly the reference's handles on the key");
        } else if (op < 92) {
            const int  mod = 2 + static_cast<int>(rng() % 5);
            const int  rem = static_cast<int>(rng() % static_cast<unsigned>(mod));
            const auto hit = [&](int serial) {
                return serial % mod == rem;
            };
            (void) b.remove_if([&](const fake_handle & h) { return hit(h.serial); });
            for (auto it = keyed.begin(); it != keyed.end();) {
                it = hit(it->second) ? keyed.erase(it) : std::next(it);
            }
            keyless.erase(std::remove_if(keyless.begin(), keyless.end(), hit), keyless.end());
        } else if (op < 97) {
            std::vector<fake_handle> v = b.take_all();
            CHECK(b.empty(), "take_all empties the bag");
            std::shuffle(v.begin(), v.end(), rng);
            b.assign(std::move(v));  // the same handles back, in another order
        } else {
            b.clear();
            keyed.clear();
            keyless.clear();
        }
        CHECK(serials(b) == reference_serials(), "the bag holds exactly the reference's handles");
        if (step % 997 == 0) {
            CHECK(index_matches(b), "the index names every keyed handle where it is");
        }
    }
    return 0;
}

int main() {
    int (*const tests[])() = { test_take_key_moves_every_lease_on_the_key,
                               test_take_key_with_the_last_element_on_the_key,
                               test_assign_take_all_and_clear,
                               test_remove_if_reindexes,
                               test_index_is_built_lazily,
                               test_random_operations_match_a_reference };
    for (int (*t)() : tests) {
        if (t() != 0) {
            return 1;
        }
    }
    std::printf("test-sycl-keyed-handle-bag: all passed\n");
    return 0;
}
