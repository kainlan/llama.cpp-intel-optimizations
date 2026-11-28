// The in-place compaction the unified cache's deferred-free walks drop their ready entries with: one pass, each kept
// element moves at most once, survivors keep their order, and a dropped element is destroyed as soon as its visit
// returns true, before the next element is visited.
// ggml-sycl/vector-compact.hpp is SYCL-free, so this links nothing and opens no device.
#include "ggml-sycl/vector-compact.hpp"

#include <cstdio>
#include <stdexcept>
#include <vector>

#define CHECK(cond, msg)                      \
    do {                                      \
        if (!(cond)) {                        \
            std::printf("FAILED: %s\n", msg); \
            return 1;                         \
        }                                     \
    } while (0)

namespace {

// Counts what the helper does to its elements. A value is live until it is moved from; only the destruction of a live
// value is logged, so the moved-from shells the compaction leaves behind and then erases stay silent.
struct probe_stats {
    size_t           move_constructs   = 0;
    size_t           move_assigns      = 0;
    size_t           assigns_onto_live = 0;
    std::vector<int> destroyed_live;  // the ids of live values destroyed, in order
};

probe_stats g_stats;

struct probe {
    int  id      = -1;
    int  payload = 0;
    bool live    = false;

    probe() = default;

    explicit probe(int id_) : id(id_), payload(id_), live(true) {}

    probe(const probe &)             = delete;
    probe & operator=(const probe &) = delete;

    probe(probe && other) noexcept : id(other.id), payload(other.payload), live(other.live) {
        other.live = false;
        ++g_stats.move_constructs;
    }

    probe & operator=(probe && other) noexcept {
        if (live) {
            ++g_stats.assigns_onto_live;
        }
        id         = other.id;
        payload    = other.payload;
        live       = other.live;
        other.live = false;
        ++g_stats.move_assigns;
        return *this;
    }

    ~probe() {
        if (live) {
            g_stats.destroyed_live.push_back(id);
        }
    }
};

std::vector<probe> make_probes(int n) {
    std::vector<probe> v;
    v.reserve(n);
    for (int i = 0; i < n; ++i) {
        v.emplace_back(i);
    }
    return v;
}

size_t moves() {
    return g_stats.move_constructs + g_stats.move_assigns;
}

bool destroyed_once(int id) {
    size_t n = 0;
    for (int d : g_stats.destroyed_live) {
        n += d == id ? 1 : 0;
    }
    return n == 1;
}

}  // namespace

static int test_empty_vector() {
    std::vector<probe> v;
    size_t             visits  = 0;
    const size_t       dropped = ggml_sycl::vector_compact_visit(v, [&](probe &) {
        ++visits;
        return true;
    });
    CHECK(visits == 0, "an empty vector is not visited");
    CHECK(dropped == 0, "an empty vector drops nothing");
    CHECK(v.empty(), "an empty vector stays empty");
    return 0;
}

static int test_mixed_keep_drop() {
    g_stats              = {};
    std::vector<probe> v = make_probes(10);
    std::vector<int>   visited;
    const size_t       dropped = ggml_sycl::vector_compact_visit(v, [&](probe & p) {
        visited.push_back(p.id);
        return p.id % 3 == 0;  // drop 0, 3, 6, 9
    });
    CHECK(dropped == 4, "the return value is the number dropped");
    CHECK(visited == std::vector<int>({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }), "every element is visited once, in order");
    const std::vector<int> want = { 1, 2, 4, 5, 7, 8 };
    CHECK(v.size() == want.size(), "the vector holds the survivors alone");
    for (size_t i = 0; i < want.size(); ++i) {
        CHECK(v[i].id == want[i] && v[i].live, "the survivors keep their relative order");
    }
    CHECK(g_stats.destroyed_live == std::vector<int>({ 0, 3, 6, 9 }), "the drops alone are destroyed, in order");
    return 0;
}

static int test_visit_modifies_kept_element() {
    std::vector<probe> v = make_probes(6);
    ggml_sycl::vector_compact_visit(v, [](probe & p) {
        if (p.id % 2 == 0) {
            return true;
        }
        p.payload = 100 + p.id;  // as the device walk sets has_event and the event on an entry it keeps
        return false;
    });
    CHECK(v.size() == 3, "three odd elements survive");
    for (const probe & p : v) {
        CHECK(p.payload == 100 + p.id, "a change made to a kept element during its visit persists");
    }
    return 0;
}

static int test_drop_destroyed_before_next_visit() {
    g_stats                         = {};
    std::vector<probe> v            = make_probes(8);
    bool               ordered      = true;
    int                last_dropped = -1;
    ggml_sycl::vector_compact_visit(v, [&](probe & p) {
        if (last_dropped >= 0 && !destroyed_once(last_dropped)) {
            ordered = false;
        }
        last_dropped = -1;
        if (p.id == 1 || p.id == 2 || p.id == 5) {
            last_dropped = p.id;
            return true;
        }
        return false;
    });
    CHECK(ordered, "a dropped element is destroyed before the next element is visited");
    CHECK(destroyed_once(5), "the last element dropped is destroyed by the time the call returns");
    CHECK(g_stats.destroyed_live == std::vector<int>({ 1, 2, 5 }), "the drops alone are destroyed, in order");
    return 0;
}

static int test_drop_may_move_element_out() {
    g_stats              = {};
    std::vector<probe> v = make_probes(6);
    std::vector<probe> released;
    released.reserve(6);
    ggml_sycl::vector_compact_visit(v, [&](probe & p) {
        if (p.id % 2 == 1) {
            released.push_back(std::move(p));  // as the optional-layout yield takes its matches out
            return true;
        }
        return false;
    });
    CHECK(g_stats.destroyed_live.empty(), "a drop moved out by visit leaves a shell whose destruction frees nothing");
    CHECK(released.size() == 3 && released[0].id == 1 && released[1].id == 3 && released[2].id == 5,
          "visit moved each drop out, in order");
    CHECK(v.size() == 3 && v[0].id == 0 && v[1].id == 2 && v[2].id == 4, "the kept elements stay, in order");
    return 0;
}

static int test_linear_moves() {
    const int n = 10000;
    {
        std::vector<probe> v = make_probes(n);
        g_stats              = {};
        size_t       visits  = 0;
        const size_t dropped = ggml_sycl::vector_compact_visit(v, [&](probe & p) {
            ++visits;
            return p.id % 2 == 0;
        });
        CHECK(visits == (size_t) n && dropped == (size_t) n / 2, "alternating drop visits all and drops half");
        // Each drop is move-constructed out once; slot 0 is dropped, so every survivor is move-assigned down once.
        CHECK(g_stats.move_constructs == (size_t) n / 2 && g_stats.move_assigns == (size_t) n / 2,
              "alternating drop moves each element once: n/2 constructs and n/2 assigns, not O(n^2)");
        CHECK(g_stats.assigns_onto_live == 0, "a kept element never lands on a live element");
        CHECK(g_stats.destroyed_live.size() == (size_t) n / 2, "each drop is destroyed once");
        CHECK(v.size() == (size_t) n / 2, "half survive");
        for (int i = 0; i < n / 2; ++i) {
            CHECK(v[i].id == 2 * i + 1 && v[i].live, "the survivors keep their order");
        }
    }
    {
        std::vector<probe> v = make_probes(n);
        g_stats              = {};
        const size_t dropped = ggml_sycl::vector_compact_visit(v, [](probe &) { return true; });
        CHECK(dropped == (size_t) n && v.empty(), "all-drop empties the vector");
        CHECK(g_stats.move_constructs == (size_t) n && g_stats.move_assigns == 0,
              "all-drop moves each element out once and assigns nothing");
        CHECK(g_stats.destroyed_live.size() == (size_t) n, "all-drop destroys every element once");
    }
    {
        std::vector<probe> v = make_probes(n);
        g_stats              = {};
        const size_t dropped = ggml_sycl::vector_compact_visit(v, [](probe &) { return false; });
        CHECK(dropped == 0 && v.size() == (size_t) n, "all-keep keeps every element");
        CHECK(moves() == 0, "all-keep moves nothing: every element is already in its slot");
        CHECK(g_stats.destroyed_live.empty(), "all-keep destroys nothing");
    }
    return 0;
}

static int test_visit_throws() {
    const int          n = 10;
    const int          k = 6;
    std::vector<probe> v = make_probes(n);
    g_stats              = {};
    bool threw           = false;
    try {
        ggml_sycl::vector_compact_visit(v, [&](probe & p) {
            if (p.id == k) {
                throw std::runtime_error("visit failed");
            }
            return p.id % 2 == 0;  // drop 0, 2, 4 before k
        });
    } catch (const std::runtime_error &) {
        threw = true;
    }
    CHECK(threw, "the exception from visit propagates");
    const std::vector<int> want = { 1, 3, 5, 6, 7, 8, 9 };
    CHECK(v.size() == want.size(), "the vector holds the survivors before k, then k..n-1");
    for (size_t i = 0; i < want.size(); ++i) {
        CHECK(v[i].id == want[i], "survivors before k in order, then the unvisited tail unchanged");
        CHECK(v[i].live, "no moved-from element remains in the vector");
        CHECK(v[i].payload == want[i], "the tail's elements are unchanged");
    }
    CHECK(g_stats.destroyed_live == std::vector<int>({ 0, 2, 4 }), "each drop before k was destroyed once");
    return 0;
}

// The documented limit of the throw guarantee: a visit that moves the element out and then throws leaves the
// moved-from shell in its slot, so a visitor that moves out must do so as its last step.
static int test_visit_moves_out_then_throws() {
    const int          n = 10;
    const int          k = 6;
    std::vector<probe> v = make_probes(n);
    std::vector<probe> sink;
    sink.reserve(1);
    g_stats    = {};
    bool threw = false;
    try {
        ggml_sycl::vector_compact_visit(v, [&](probe & p) {
            if (p.id == k) {
                sink.push_back(std::move(p));
                throw std::runtime_error("visit failed after moving its element out");
            }
            return p.id % 2 == 0;  // drop 0, 2, 4 before k
        });
    } catch (const std::runtime_error &) {
        threw = true;
    }
    CHECK(threw, "the exception from visit propagates");
    CHECK(sink.size() == 1 && sink[0].id == k && sink[0].live, "visit took element k out before it threw");
    const std::vector<int> want = { 1, 3, 5, k, 7, 8, 9 };
    CHECK(v.size() == want.size(), "the survivors before k, the shell at k, then the tail");
    for (size_t i = 0; i < want.size(); ++i) {
        CHECK(v[i].id == want[i], "survivors before k compacted, then the shell, then the tail in order");
        CHECK(v[i].live == (want[i] != k), "only the slot visit moved out of holds a moved-from shell");
    }
    for (size_t i = 4; i < want.size(); ++i) {
        CHECK(v[i].payload == want[i], "the tail after k is untouched");
    }
    CHECK(g_stats.destroyed_live == std::vector<int>({ 0, 2, 4 }), "each drop before k was destroyed once");
    return 0;
}

int main() {
    int (*const tests[])() = { test_empty_vector,
                               test_mixed_keep_drop,
                               test_visit_modifies_kept_element,
                               test_drop_destroyed_before_next_visit,
                               test_drop_may_move_element_out,
                               test_linear_moves,
                               test_visit_throws,
                               test_visit_moves_out_then_throws };
    for (int (*t)() : tests) {
        if (t() != 0) {
            return 1;
        }
    }
    std::printf("test-sycl-vector-compact: all passed\n");
    return 0;
}
