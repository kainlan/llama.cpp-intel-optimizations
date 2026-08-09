// Mutation blast-radius simulator for layer_stream_manager's ownership logic
// (llama.cpp-vbeb). Host-only, no SYCL, no device: g++ and run.
//
//     g++ -std=c++17 -o /tmp/sim scripts/sim-layer-stream-ownership.cpp && /tmp/sim
//
// WHY THIS EXISTS. ggml/src/ggml-sycl/tests/test-layer-stream-owner-lifecycle.cpp
// documents, per mutation, which checks that mutation fires -- and that table is
// the thing a reviewer scores a RED run against. Deriving it by reading the code
// does not work: three independent hand-traces of ONE row gave three different
// answers ("2d only", "2d and 2g", and the ten-check truth), because the cases
// share a manager whose state carries between them. Rebuilding the real binary
// to find out costs ~50 minutes, since layer-streaming.hpp is included by the
// 100k-line ggml-sycl.cpp.
//
// So the ownership logic is transcribed here and every mutation is scored
// against every case mechanically, in under a second.
//
// WHAT IT IS NOT. This is a TRANSCRIPTION, not the shipped code, and nothing
// keeps it in step automatically. It models only what the assertions read:
// ownership transitions, the layer map's size and names, loaded_layers_, and
// whether the buffers are installed. It deliberately does NOT model mem_handle,
// DMA, locking, or the unified cache -- the real binary is the authority on
// those. Treat a disagreement between this and a real RED run as evidence that
// THIS file is stale, and fix it here.
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// ---- mutation switches (set per run) --------------------------------------
static bool M_no_adopt_in_register = false;
static bool M_no_adopt_in_build    = false;
static bool M_no_drain_in_build    = false;
static bool M_owner_valid_always   = false;
static bool M_release_uncond       = false;

struct Owner {
    unsigned long long model = 0, load = 0, gen = 0;
    unsigned           slot = 0xFFFFFFFFu;
};

static bool valid(const Owner & o) {
    if (M_owner_valid_always) {
        return true;
    }
    return o.model != 0 && o.load != 0 && o.slot != 0xFFFFFFFFu && o.gen != 0;
}

static bool same(const Owner & a, const Owner & b) {
    return a.model == b.model && a.load == b.load && a.slot == b.slot && a.gen == b.gen;
}

enum Kind { NONE, ADOPT, KEEP, DISPLACE };

struct Gate {
    Owner owner{};
    bool  owned = false;

    Kind observe(const Owner & in, Owner & displaced) {
        if (!valid(in)) {
            return NONE;
        }
        if (!owned) {
            owner = in;
            owned = true;
            return ADOPT;
        }
        if (same(owner, in)) {
            return KEEP;
        }
        displaced = owner;
        owner     = in;
        return DISPLACE;
    }

    Owner current() const { return owned ? owner : Owner{}; }

    bool is_owner(const Owner & o) const { return owned && same(owner, o); }

    void forget() {
        owner = Owner{};
        owned = false;
    }
};

// ---- the registry stand-in: which token is the ACTIVE load transaction -----
static Owner g_active{};

struct Mgr {
    std::vector<std::vector<std::string>> layers;  // layer -> weight names
    std::map<std::string, int>            name_to_layer;
    size_t                                max_layer_size = 0;
    void *                                buffers[2]     = { nullptr, nullptr };
    int                                   loaded[2]      = { -1, -1 };
    size_t                                buffer_size    = 0;
    Gate                                  gate;

    bool is_active() const { return buffers[0] != nullptr; }

    int n_layers() const { return (int) layers.size(); }

    int buffer_for_layer(int l) const { return loaded[0] == l ? 0 : (loaded[1] == l ? 1 : -1); }

    void drain_and_invalidate_buffers() {
        loaded[0] = -1;
        loaded[1] = -1;
    }

    void release_model_state() {
        drain_and_invalidate_buffers();
        buffers[0] = buffers[1] = nullptr;
        buffer_size             = 0;
        layers.clear();
        name_to_layer.clear();
        max_layer_size = 0;
    }

    void adopt_current_owner() {
        Owner disp{};
        if (gate.observe(g_active, disp) == DISPLACE) {
            release_model_state();
        }
    }

    void build_layer_map(int n, const char * suffix, size_t bytes) {
        if (!M_no_adopt_in_build) {
            adopt_current_owner();
        }
        if (!M_no_drain_in_build) {
            drain_and_invalidate_buffers();
        }
        layers.clear();
        name_to_layer.clear();
        max_layer_size = 0;
        layers.resize(n);
        for (int i = 0; i < n; i++) {
            std::string nm = "blk." + std::to_string(i) + "." + suffix;
            layers[i].push_back(nm);
            name_to_layer[nm] = i;
            if (bytes > max_layer_size) {
                max_layer_size = bytes;
            }
        }
    }

    void register_host_ptr(const char * name) {
        if (!M_no_adopt_in_register) {
            adopt_current_owner();
        }
        (void) name;  // the map write itself is irrelevant to every assertion
    }

    void * get_weight_device_ptr(const char * name) {
        if (!is_active()) {
            return nullptr;
        }
        auto it = name_to_layer.find(name);
        if (it == name_to_layer.end()) {
            return nullptr;
        }
        int b = buffer_for_layer(it->second);
        if (b < 0 || !buffers[b]) {
            return nullptr;
        }
        return buffers[b];
    }

    void shutdown() {
        release_model_state();
        gate.forget();
    }

    bool release_if_owner(const Owner & o) {
        if (M_release_uncond) {
            gate.forget();
            release_model_state();
            return true;
        }
        if (!gate.is_owner(o)) {
            return false;
        }
        gate.forget();
        release_model_state();
        return true;
    }

    void test_install(void * b0, int l0, void * b1, int l1) {
        buffers[0]  = b0;
        loaded[0]   = l0;
        buffers[1]  = b1;
        loaded[1]   = l1;
        buffer_size = 4096;
    }
};

// ---- harness ---------------------------------------------------------------
static std::vector<std::string> g_fails;

static void ck(bool cond, const char * label) {
    if (!cond) {
        g_fails.push_back(label);
    }
}

static unsigned long long g_next = 1;

struct Txn {
    Owner tok;

    Txn() {
        tok = { g_next, g_next * 10, g_next, (unsigned) (g_next % 32) };
        g_next++;
        g_active = tok;
    }

    void close() { g_active = Owner{}; }

    bool owns(const Owner & o) const { return same(o, tok); }
};

static char B0[2], B1[2];

static void case1() {
    Mgr   m;
    Owner ao;
    {
        Txn a;
        m.build_layer_map(2, "attn_q.weight", 1024);
        m.register_host_ptr("blk.0.attn_q.weight");
        m.test_install(B0, 0, B1, 1);
        ck(m.is_active(), "1-pre: buffers");
        ck(m.n_layers() == 2, "1-pre: map");
        ck(m.get_weight_device_ptr("blk.0.attn_q.weight") != nullptr, "1-pre: resolves");
        ck(a.owns(m.gate.current()), "1-pre: A owns");
        ao = m.gate.current();
        a.close();
    }
    Txn b;
    m.register_host_ptr("blk.0.attn_q.weight");
    ck(!m.is_active(), "1a");
    ck(m.get_weight_device_ptr("blk.0.attn_q.weight") == nullptr, "1b");
    ck(m.n_layers() == 0, "1c");
    ck(b.owns(m.gate.current()), "1d");
    b.close();
    (void) ao;
}

static void case2() {
    Mgr m;
    {
        Txn a;
        m.build_layer_map(4, "attn_q.weight", 1024);
        m.test_install(B0, 2, B1, 3);
        ck(m.n_layers() == 4, "2-pre: map");
        ck(m.buffer_for_layer(3) == 1, "2-pre: loaded");
        ck(m.is_active(), "2-pre: buffers");
        a.close();
    }
    Txn b;
    m.build_layer_map(2, "ffn_up.weight", 4096);
    ck(m.n_layers() == 2, "2a");
    ck(m.buffer_for_layer(2) == -1, "2b");
    ck(m.buffer_for_layer(3) == -1, "2c");
    ck(!m.is_active(), "2d");
    ck(m.get_weight_device_ptr("blk.3.attn_q.weight") == nullptr, "2e");
    ck(m.max_layer_size == 4096, "2f");
    ck(b.owns(m.gate.current()), "2g");
    b.close();
}

static void case3() {
    Mgr m;
    Txn a;
    m.build_layer_map(4, "attn_q.weight", 1024);
    m.test_install(B0, 2, B1, 3);
    ck(m.buffer_for_layer(3) == 1, "3-pre: loaded");
    m.build_layer_map(4, "attn_q.weight", 1024);
    ck(m.is_active(), "3a");
    ck(a.owns(m.gate.current()), "3b");
    ck(m.buffer_for_layer(2) == -1, "3c");
    ck(m.buffer_for_layer(3) == -1, "3d");
    a.close();
}

static void case4() {
    Mgr   m;
    Owner oa;
    {
        Txn a;
        m.build_layer_map(2, "attn_q.weight", 1024);
        m.test_install(B0, 0, B1, 1);
        oa = m.gate.current();
        ck(a.owns(oa), "4-pre: A owns");
        ck(m.is_active(), "4-pre: buffers");
        a.close();
    }
    m.register_host_ptr("blk.0.attn_q.weight");
    ck(m.is_active(), "4a");
    ck(m.n_layers() == 2, "4b");
    ck(same(m.gate.current(), oa), "4c");
}

static void case5() {
    Mgr m;
    {
        Txn a1;
        m.build_layer_map(3, "attn_q.weight", 1024);
        m.test_install(B0, 0, B1, 1);
        ck(m.n_layers() == 3, "5-pre: A map");
        a1.close();
    }
    {
        Txn b;
        m.build_layer_map(2, "ffn_up.weight", 2048);
        ck(m.n_layers() == 2, "5-pre: B map");
        b.close();
    }
    Txn a2;
    m.register_host_ptr("blk.0.attn_q.weight");
    ck(m.n_layers() == 0, "5a");
    ck(!m.is_active(), "5b");
    ck(a2.owns(m.gate.current()), "5c");
    a2.close();
}

static void case6() {
    Mgr m;
    Txn a;
    m.build_layer_map(2, "attn_q.weight", 1024);
    m.test_install(B0, 0, B1, 1);
    ck(m.is_active(), "6-pre: buffers");
    m.shutdown();
    ck(!m.is_active(), "6a");
    ck(m.n_layers() == 0, "6b");
    ck(m.max_layer_size == 0, "6c");
    ck(!valid(m.gate.current()), "6d");
    a.close();
}

// Teardown path (new): A streams, B loads, A tears down -> B intact; B tears
// down -> released.
static void case7() {
    Mgr   m;
    Owner oa, ob;
    {
        Txn a;
        m.build_layer_map(3, "attn_q.weight", 1024);
        m.test_install(B0, 0, B1, 1);
        oa = m.gate.current();
        a.close();
    }
    {
        Txn b;
        m.build_layer_map(2, "ffn_up.weight", 2048);
        m.test_install(B0, 0, B1, 1);
        ob = m.gate.current();
        b.close();
    }
    ck(m.is_active() && m.n_layers() == 2, "7-pre: B holds the working set");
    ck(!same(oa, ob), "7-pre: A and B are distinct owners");
    bool released_a = m.release_if_owner(oa);
    ck(!released_a, "7a: tearing down A released nothing");
    ck(m.is_active(), "7b: B's buffers survived A's teardown");
    ck(m.n_layers() == 2, "7c: B's map survived A's teardown");
    ck(same(m.gate.current(), ob), "7d: B is still the owner");
    bool released_b = m.release_if_owner(ob);
    ck(released_b, "7e: tearing down B released its state");
    ck(!m.is_active(), "7f: B's buffers are gone");
    ck(m.n_layers() == 0, "7g: B's map is gone");
    ck(!valid(m.gate.current()), "7h: the owner record is gone");
}

static void run_all() {
    g_fails.clear();
    case1();
    case2();
    case3();
    case4();
    case5();
    case6();
    case7();
}

int main() {
    struct M {
        const char * name;
        bool *       flag;
    };

    M muts[] = {
        { "BASELINE (no mutation)",                                 nullptr                 },
        { "drop adopt_current_owner from register_host_ptr",        &M_no_adopt_in_register },
        { "drop adopt_current_owner from build_layer_map",          &M_no_adopt_in_build    },
        { "drop drain_and_invalidate_buffers from build_layer_map", &M_no_drain_in_build    },
        { "owner_valid() always true",                              &M_owner_valid_always   },
        { "release_if_owner unconditional",                         &M_release_uncond       },
    };
    for (M & mu : muts) {
        M_no_adopt_in_register = M_no_adopt_in_build = M_no_drain_in_build = false;
        M_owner_valid_always = M_release_uncond = false;
        if (mu.flag) {
            *mu.flag = true;
        }
        g_next = 1;
        run_all();
        printf("%-52s -> %2zu FAIL:", mu.name, g_fails.size());
        for (const std::string & f : g_fails) {
            printf(" %s", f.c_str());
        }
        printf("\n");
    }
    return 0;
}
