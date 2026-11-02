// The layer id of a KV cache tensor must come from "cache_<tag>(k|v)_l<N>" for any tag, and anything else
// must read as "not a KV cache tensor" (-1). The tiered KV buffer remaps a tensor onto its layer's allocation
// by this id; the prefix-only parser it used to have gave name-tagged caches (qwen4exp's "cache_idx_k_l3")
// layer -1, left them on the buffer's synthetic host span, and aborted the first SET_ROWS (llama.cpp-4ot7).
//
// Host-only: the header includes <climits>/<cstring> alone.
#include "ggml-sycl/kv-cache-tensor-name.hpp"

#include <cstdio>
#include <string>

static int g_failures = 0;

static void expect_layer(const char * name, int want) {
    const int got = ggml_sycl::kv_cache_tensor_layer_id(name);
    if (got != want) {
        std::printf("FAILED: layer_id(\"%.64s\") = %d, want %d\n", name ? name : "(null)", got, want);
        g_failures++;
    }
}

int main() {
    // untagged main attention cache
    expect_layer("cache_k_l3", 3);
    expect_layer("cache_v_l12", 12);
    expect_layer("cache_k_l0", 0);
    // tagged caches
    expect_layer("cache_idx_k_l3", 3);
    expect_layer("cache_idx_v_l47", 47);
    expect_layer("cache_swa_k_l0", 0);
    expect_layer("cache_a_b_v_l9", 9);
    // largest int is still a layer id; one past it is not
    expect_layer("cache_k_l2147483647", 2147483647);
    expect_layer("cache_k_l2147483648", -1);
    expect_layer("cache_k_l99999999999999999999", -1);

    // not KV cache tensors
    expect_layer(nullptr, -1);
    expect_layer("", -1);
    expect_layer("cache_", -1);
    expect_layer("cache_k_l", -1);    // no digits
    expect_layer("cache_idx_k_l", -1);
    expect_layer("cache_k_lx", -1);   // junk instead of digits
    expect_layer("cache_k_l3x", -1);  // trailing junk
    expect_layer("cache_k_l3 ", -1);
    expect_layer("cache_k_l3_copy", -1);
    expect_layer("cache_k_l-3", -1);  // negative
    expect_layer("cache_k_l+3", -1);
    expect_layer("cache_r_l3", -1);   // recurrent state, not attention KV
    expect_layer("cache_s_l3", -1);
    expect_layer("cache_ple_r_l3", -1);
    expect_layer("cache_k3", -1);      // no "_l"
    expect_layer("cache_kxl3", -1);    // 'l' preceded by something other than '_'
    expect_layer("cache_k_l_l4", -1);  // "_l" is there but the letter before it is 'l', not k/v
    expect_layer("cachek_l3", -1);     // no "cache_" prefix
    expect_layer("xcache_k_l3", -1);
    expect_layer("k_l3", -1);
    expect_layer("blk.3.attn_k.weight", -1);
    expect_layer("dsv4_csa_state_kv_l3", -1);
    expect_layer("_l3", -1);

    // an arbitrary-length tag must not overrun anything
    expect_layer((std::string("cache_") + std::string(4096, 'x') + "k_l5").c_str(), 5);

    if (g_failures != 0) {
        std::printf("%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
