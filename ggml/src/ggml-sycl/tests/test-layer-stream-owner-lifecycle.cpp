//
// MIT license
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

// Cross-model survival regression for layer_stream_manager (llama.cpp-vbeb,
// child of the k7b0 file-scope-static audit).
//
// The defect: layer_stream_manager is a per-device singleton whose layer map,
// name -> host-pointer registry and two device staging buffers had no owner.
// build_layer_map() rebuilt the map only when a model ACTIVATED streaming, and
// a model that did not activate streaming never reached the class at all. So
// after a streaming model A, an incoming model B found is_active() still true
// and resolved its own blk.N.* weights -- names essentially every dense
// architecture shares -- into A's buffers. Wrong bytes, no error, no log line.
// A second instance: build_layer_map() cleared the map but never reset
// loaded_layers_, so a rebuild for B left A's layer ids marked loaded and
// get_weight_device_ptr() returned B's offset into A's contents.
//
// This exercises the REAL ggml_sycl::layer_stream_manager and the REAL
// ggml_sycl::lifecycle::Registry -- no mock reimplementation of either. Model
// identity comes from actual begin_outer()/end() load transactions, which is
// how the production code learns it. The one seam is
// test_install_loaded_buffers(), which installs the buffer bookkeeping a
// successful allocate_buffers() plus two layer loads would leave behind: that
// step needs a queue and a device, and the ownership arithmetic under test
// needs neither. Nothing here creates a SYCL queue or touches a GPU.
//
// The carryover oracle is deliberately NOT raw-pointer identity. Buffer
// addresses can repeat across allocations, so "B got a different pointer" both
// false-positives and false-negatives. Every check below is on names, counts
// and layer ids.
//
// Mutation controls. Each is a one-line revert of one part of the fix, and the
// cases it fires are stated exactly -- not rounded down to "only mine", because
// two of them legitimately reach a second case:
//   * drop `adopt_current_owner();` from register_host_ptr()
//         -> cases 1 (1a-1d) and 5 (5a-5c). Both reach the class only through
//            register_host_ptr, so both lose displacement together.
//   * drop `adopt_current_owner();` from build_layer_map()
//         -> case 2, check 2d only. 2b/2c survive because build_layer_map's own
//            drain still resets loaded_layers_; what is lost is the buffer
//            release, which is exactly what 2d asks about.
//   * drop `drain_and_invalidate_buffers();` from build_layer_map()
//         -> case 3 (3c, 3d) only. Case 2 is unaffected because displacement
//            has already reset those ids via release_model_state().
//   * make layer_stream_owner_valid() return true for a zeroed owner
//         -> case 4 (4a-4c) only. Every other case calls in with a real
//            transaction open, so their verdicts do not change.

#include "../layer-streaming.hpp"
#include "../model-lifecycle.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "SKIP: GGML_USE_SYCL not enabled; this proves NOTHING about layer-stream ownership.\n");
    return 77;
}
#else

using ggml_sycl::layer_stream_manager;
using ggml_sycl::layer_stream_owner;
namespace lifecycle = ggml_sycl::lifecycle;

static int g_checks   = 0;
static int g_failures = 0;

static void check(bool cond, const char * label) {
    g_checks++;
    if (!cond) {
        g_failures++;
        printf("  FAIL: %s\n", label);
    }
}

// A load transaction on the process registry. Scoped so a failing check cannot
// leave the registry with a transaction open and strand every later case in
// LOAD_BUSY.
struct scoped_load {
    lifecycle::LoadTxnId  txn{};
    lifecycle::ModelToken token{};
    bool                  open = false;

    scoped_load() {
        auto begun = lifecycle::global_registry().begin_outer();
        if (begun.code != lifecycle::error::OK) {
            printf("  FAIL: begin_outer failed (code %d); the registry is not usable\n", (int) begun.code);
            g_failures++;
            g_checks++;
            return;
        }
        txn   = begun.txn;
        token = begun.token;
        open  = true;
    }

    ~scoped_load() { close(); }

    scoped_load(const scoped_load &)             = delete;
    scoped_load & operator=(const scoped_load &) = delete;

    void close() {
        if (open) {
            lifecycle::global_registry().end(txn, true);
            open = false;
        }
    }

    bool owns(const layer_stream_owner & owner) const {
        return owner.model_id == token.model.value && owner.load_txn_id == token.load.value &&
               owner.slot == token.owner.slot && owner.slot_generation == token.owner.generation;
    }
};

static std::vector<std::pair<std::string, size_t>> inventory_of(int n_layers, const char * suffix, size_t bytes) {
    std::vector<std::pair<std::string, size_t>> inventory;
    for (int i = 0; i < n_layers; i++) {
        inventory.emplace_back("blk." + std::to_string(i) + "." + suffix, bytes);
    }
    return inventory;
}

// Opaque non-null tokens standing in for device buffers. Never dereferenced;
// the manager only does offset arithmetic on them, which the checks below never
// read back.
static char g_fake_buffer_a[2];
static char g_fake_buffer_b[2];

static void install_two_loaded_layers(layer_stream_manager & mgr, int layer0, int layer1) {
    mgr.test_install_loaded_buffers(0, 4096, &g_fake_buffer_a[0], layer0, &g_fake_buffer_b[0], layer1);
}

// ---------------------------------------------------------------------------
// Case 1: streaming A, then a NON-streaming B. B never calls build_layer_map --
// it only uploads weights, which is what buffer_set_tensor() does for every
// model. A's buffers must be gone before B's first name can resolve.
// ---------------------------------------------------------------------------
static void case_1_streaming_then_non_streaming() {
    printf("Case 1: streaming A -> non-streaming B\n");

    layer_stream_manager mgr;
    const auto           inv_a  = inventory_of(2, "attn_q.weight", 1024);
    long                 host_a = 0;
    long                 host_b = 0;

    {
        scoped_load a;
        mgr.build_layer_map(inv_a.data(), inv_a.size());
        mgr.register_host_ptr("blk.0.attn_q.weight", &host_a, 1024);
        install_two_loaded_layers(mgr, 0, 1);

        // Positive control. Every assertion below is an absence, and an absence
        // proves nothing until the thing is shown to be present first.
        check(mgr.is_active(), "1-pre: A's buffers are installed");
        check(mgr.n_layers() == 2, "1-pre: A's layer map is built");
        check(mgr.get_weight_device_ptr("blk.0.attn_q.weight") != nullptr,
              "1-pre: A's weight resolves into A's buffer");
        check(a.owns(mgr.owner()), "1-pre: A owns the working set");
    }

    scoped_load b;
    mgr.register_host_ptr("blk.0.attn_q.weight", &host_b, 2048);

    check(!mgr.is_active(), "1a: A's buffers did not survive B's load");
    check(mgr.get_weight_device_ptr("blk.0.attn_q.weight") == nullptr,
          "1b: a name B shares with A does not resolve into A's buffer");
    check(mgr.n_layers() == 0, "1c: A's layer map did not survive B's load");
    check(b.owns(mgr.owner()), "1d: B owns the working set");
}

// ---------------------------------------------------------------------------
// Case 2: streaming A, then a streaming B with a different shape. B rebuilds
// the map, so the map itself was already replaced before the fix -- what was
// not is everything around it.
// ---------------------------------------------------------------------------
static void case_2_streaming_then_different_shape() {
    printf("Case 2: streaming A -> different-shape streaming B\n");

    layer_stream_manager mgr;
    const auto           inv_a = inventory_of(4, "attn_q.weight", 1024);
    const auto           inv_b = inventory_of(2, "ffn_up.weight", 4096);

    {
        scoped_load a;
        mgr.build_layer_map(inv_a.data(), inv_a.size());
        install_two_loaded_layers(mgr, 2, 3);

        check(mgr.n_layers() == 4, "2-pre: A's four layers are mapped");
        check(mgr.buffer_for_layer(3) == 1, "2-pre: A's layer 3 is marked loaded in buffer 1");
        check(mgr.is_active(), "2-pre: A's buffers are installed");
    }

    scoped_load b;
    mgr.build_layer_map(inv_b.data(), inv_b.size());

    check(mgr.n_layers() == 2, "2a: the map is B's, not A's");
    check(mgr.buffer_for_layer(2) == -1, "2b: A's layer 2 is no longer marked loaded");
    check(mgr.buffer_for_layer(3) == -1, "2c: A's layer 3 is no longer marked loaded");
    check(!mgr.is_active(), "2d: B did not inherit A's buffers");
    check(mgr.get_weight_device_ptr("blk.3.attn_q.weight") == nullptr, "2e: an A-only name does not resolve");
    check(mgr.max_layer_size() == 4096, "2f: max_layer_size is B's, not the larger of A and B");
    check(b.owns(mgr.owner()), "2g: B owns the working set");
}

// ---------------------------------------------------------------------------
// Case 3: the SAME model rebuilding its own map keeps its buffers -- the fix is
// owner displacement, not "clear on every call". But the rebuild must still
// forget which layers those buffers hold, because the offsets those ids index
// into have just been recomputed.
// ---------------------------------------------------------------------------
static void case_3_same_owner_rebuild() {
    printf("Case 3: same owner rebuilds its own map\n");

    layer_stream_manager mgr;
    const auto           inv = inventory_of(4, "attn_q.weight", 1024);

    scoped_load a;
    mgr.build_layer_map(inv.data(), inv.size());
    install_two_loaded_layers(mgr, 2, 3);
    check(mgr.buffer_for_layer(3) == 1, "3-pre: layer 3 is marked loaded");

    mgr.build_layer_map(inv.data(), inv.size());

    check(mgr.is_active(), "3a: a rebuild by the owner does not release its own buffers");
    check(a.owns(mgr.owner()), "3b: the owner is unchanged");
    check(mgr.buffer_for_layer(2) == -1, "3c: the rebuild forgot which layer buffer 0 held");
    check(mgr.buffer_for_layer(3) == -1, "3d: the rebuild forgot which layer buffer 1 held");
}

// ---------------------------------------------------------------------------
// Case 4: fail closed. A caller with no load transaction open is unattributed:
// it must be able neither to claim the working set nor to release the owner's.
// Without this the fix degrades into a load-boundary sweep that any inference
// -time call could trigger against a live model.
// ---------------------------------------------------------------------------
static void case_4_unattributed_caller_cannot_displace() {
    printf("Case 4: an unattributed caller cannot displace the owner\n");

    layer_stream_manager mgr;
    const auto           inv  = inventory_of(2, "attn_q.weight", 1024);
    long                 host = 0;
    layer_stream_owner   owner_a{};

    {
        scoped_load a;
        mgr.build_layer_map(inv.data(), inv.size());
        install_two_loaded_layers(mgr, 0, 1);
        owner_a = mgr.owner();
        check(a.owns(owner_a), "4-pre: A owns the working set");
        check(mgr.is_active(), "4-pre: A's buffers are installed");
    }

    // No transaction is open here.
    mgr.register_host_ptr("blk.0.attn_q.weight", &host, 1024);

    check(mgr.is_active(), "4a: an unattributed registration did not release A's buffers");
    check(mgr.n_layers() == 2, "4b: an unattributed registration did not clear A's map");
    check(ggml_sycl::layer_stream_owner_same(mgr.owner(), owner_a), "4c: the owner is still A");
}

// ---------------------------------------------------------------------------
// Case 5: A -> B -> A. There is no park/restore half here, so a returning model
// is a new load transaction and therefore a new owner: it must find an empty
// working set, not the one it built the first time.
// ---------------------------------------------------------------------------
static void case_5_a_then_b_then_a() {
    printf("Case 5: A -> B -> A does not restore A's first working set\n");

    layer_stream_manager mgr;
    const auto           inv_a = inventory_of(3, "attn_q.weight", 1024);
    const auto           inv_b = inventory_of(2, "ffn_up.weight", 2048);
    long                 host  = 0;

    {
        scoped_load a1;
        mgr.build_layer_map(inv_a.data(), inv_a.size());
        install_two_loaded_layers(mgr, 0, 1);
        check(mgr.n_layers() == 3, "5-pre: A's first map is built");
    }
    {
        scoped_load b;
        mgr.build_layer_map(inv_b.data(), inv_b.size());
        check(mgr.n_layers() == 2, "5-pre: B's map replaced A's");
    }

    scoped_load a2;
    mgr.register_host_ptr("blk.0.attn_q.weight", &host, 1024);

    check(mgr.n_layers() == 0, "5a: the returning model did not inherit B's map");
    check(!mgr.is_active(), "5b: the returning model did not inherit B's buffers");
    check(a2.owns(mgr.owner()), "5c: the second load of A is a distinct owner");
}

// ---------------------------------------------------------------------------
// Case 6: shutdown() drops the ownership record as well as the state. If it
// released the state but kept the record, the next model would be told KEEP
// instead of ADOPT and would run on top of a torn-down working set.
// ---------------------------------------------------------------------------
static void case_6_shutdown_forgets_owner() {
    printf("Case 6: shutdown() forgets the owner\n");

    layer_stream_manager mgr;
    const auto           inv = inventory_of(2, "attn_q.weight", 1024);

    scoped_load a;
    mgr.build_layer_map(inv.data(), inv.size());
    install_two_loaded_layers(mgr, 0, 1);
    check(mgr.is_active(), "6-pre: A's buffers are installed");

    mgr.shutdown();

    check(!mgr.is_active(), "6a: shutdown released the buffers");
    check(mgr.n_layers() == 0, "6b: shutdown cleared the layer map");
    check(mgr.max_layer_size() == 0, "6c: shutdown cleared max_layer_size");
    check(!ggml_sycl::layer_stream_owner_valid(mgr.owner()), "6d: shutdown forgot the owner");
}

int main(int argc, char ** argv) {
    const char * only = nullptr;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--case") == 0 && i + 1 < argc) {
            only = argv[++i];
        }
    }

    struct entry {
        const char * name;
        void (*fn)();
    };

    const entry cases[] = {
        { "1", case_1_streaming_then_non_streaming        },
        { "2", case_2_streaming_then_different_shape      },
        { "3", case_3_same_owner_rebuild                  },
        { "4", case_4_unattributed_caller_cannot_displace },
        { "5", case_5_a_then_b_then_a                     },
        { "6", case_6_shutdown_forgets_owner              },
    };

    int selected = 0;
    for (const entry & c : cases) {
        if (only && std::strcmp(only, c.name) != 0) {
            continue;
        }
        selected++;
        c.fn();
    }

    if (only && selected == 0) {
        printf("FAIL: --case %s matched nothing; this run proves NOTHING\n", only);
        return 1;
    }
    if (g_checks == 0) {
        printf("FAIL: no checks ran; this run proves NOTHING\n");
        return 1;
    }

    printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

#endif  // GGML_USE_SYCL
