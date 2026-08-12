// SYCL weight cache identity policy test (was: MoE Identity Hash Collision Test).
//
// Policy under test -- owner ruling on llama.cpp-n3pw, comment c-0bvm, "Option A,
// physical identity":
//
//   A GGUF-backed weight is identified by WHERE ITS BYTES LIVE:
//   (file_id, file_idx, file_offs, nbytes, type, ne).  Neither the model that
//   loaded it nor the name a graph node gave it takes part in that comparison.
//   Model separation comes from FILE IDENTITY, not from a model_id field.
//   A weight with no GGUF identity keys on everything it has -- model_id,
//   name_hash and aux_id included -- and so never shares across models.
//
// This file previously asserted the opposite for the GGUF-backed case: it
// required model_id in the key ("FAIL: model_id mismatch", llama.cpp-qvid) and
// required two names over one file offset to stay distinct.  Both expectations
// are wrong under the ruling, and the second is the bug the ruling closes: it
// is what kept a tied embedding/output-head pair staged twice.  The tests below
// assert the policy instead.
//
// MoE expert slices use a stricter policy than dense weights: their canonical
// keys name both the exact ModelToken owner and their slice of the registered
// GGUF parent. The legacy logical-key collision regression remains covered for
// non-GGUF ids, while test 8 pins the load-scoped replacement contract.
//
// Related: llama.cpp-qvid (this file's failing key dump), llama.cpp-i14 (stable
// logical + representation identity keys), llama.cpp-twc (the original expert
// identity collision).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl.h"
#include "ggml-sycl/unified-cache.hpp"

#if !defined(GGML_USE_SYCL)
int main() {
    // 77 is ctest's SKIP_RETURN_CODE. Exiting 0 here would report a run that
    // tested nothing as a pass -- see CLAUDE.md, "a SKIP line with status 0 is
    // not a pass".
    fprintf(stderr, "SKIP: GGML_USE_SYCL not enabled; this run proves NOTHING about cache identity.\n");
    return 77;
}
#else

// Test counters
static int g_tests_run    = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_ASSERT(cond, msg)                                                                                         \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stderr, "  FAIL: %s\n", msg);                                                                      \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

#define RUN_TEST(fn)                                                                                                   \
    do {                                                                                                               \
        g_tests_run++;                                                                                                 \
        if (fn()) {                                                                                                    \
            g_tests_passed++;                                                                                          \
            printf("  PASS: %s\n", #fn);                                                                               \
        } else {                                                                                                       \
            g_tests_failed++;                                                                                          \
            fprintf(stderr, "  FAIL: %s\n", #fn);                                                                      \
        }                                                                                                              \
    } while (0)

// Helper: allocate a weight buffer
static ggml_backend_buffer_t alloc_weight_buffer(ggml_backend_buffer_type_t buft, ggml_tensor * weight) {
    const size_t          buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t buffer   = ggml_backend_buft_alloc_buffer(buft, buf_size);
    if (!buffer) {
        return nullptr;
    }
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(buffer, weight, ggml_backend_buffer_get_base(buffer));
    return buffer;
}

// Helper: print cache ID for debugging
static void print_cache_id(const char * label, const ggml_sycl_cache_id & id) {
    fprintf(stderr,
            "  %s: valid=%d model=%llu has_gguf=%d file_id=0x%llx file_idx=%u file_offs=%zu nbytes=%zu "
            "name_hash=0x%llx type=%d aux_id=%llu\n",
            label, id.valid, (unsigned long long) id.model_id, id.has_gguf, (unsigned long long) id.file_id,
            id.file_idx, id.file_offs, id.nbytes, (unsigned long long) id.name_hash, id.type,
            (unsigned long long) id.aux_id);
}

// Scratch state shared by the tensor-level tests.
struct test_ctx {
    ggml_backend_t                     backend = nullptr;
    ggml_context *                     ctx     = nullptr;
    ggml_backend_buffer_type_t         buft    = nullptr;
    std::vector<ggml_backend_buffer_t> buffers;

    bool init(size_t mem_size) {
        backend = ggml_backend_cpu_init();
        if (!backend) {
            return false;
        }
        ggml_init_params params = {
            /*.mem_size   =*/mem_size,
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
        };
        ctx  = ggml_init(params);
        buft = ctx ? ggml_backend_get_default_buffer_type(backend) : nullptr;
        return ctx != nullptr && buft != nullptr;
    }

    // A named, buffer-backed weight tensor. Returns nullptr on any failure.
    ggml_tensor * weight(const char * name, ggml_type type, int64_t ne0, int64_t ne1) {
        ggml_tensor * tensor = ggml_new_tensor_2d(ctx, type, ne0, ne1);
        if (!tensor) {
            return nullptr;
        }
        ggml_set_name(tensor, name);
        ggml_backend_buffer_t buffer = alloc_weight_buffer(buft, tensor);
        if (!buffer) {
            return nullptr;
        }
        buffers.push_back(buffer);
        return tensor;
    }

    ~test_ctx() {
        for (ggml_backend_buffer_t buffer : buffers) {
            ggml_backend_buffer_free(buffer);
        }
        if (ctx) {
            ggml_free(ctx);
        }
        if (backend) {
            ggml_backend_free(backend);
        }
    }
};

using key_set =
    std::unordered_set<ggml_sycl_cache_id, ggml_sycl::detail::cache_id_hash, ggml_sycl::detail::cache_id_equal_fn>;

// =============================================================================
// Test 1: a GGUF-backed key carries the file identity it was registered with
//
// model_id is still reported, because eviction and ownership need to know which
// model a cached entry belongs to.  It is NOT what separates models -- test 2
// covers that -- and asserting it here would be asserting the wrong thing, which
// is what this test used to do (llama.cpp-qvid: "FAIL: model_id mismatch").
// =============================================================================
static bool test_gguf_key_carries_file_identity() {
    printf("TEST: test_gguf_key_carries_file_identity\n");

    test_ctx t;
    TEST_ASSERT(t.init(4 * 1024 * 1024), "test context init failed");

    ggml_tensor * weight = t.weight("blk.0.ffn_gate.weight", GGML_TYPE_Q4_0, 256, 64);
    TEST_ASSERT(weight != nullptr, "weight allocation failed");

    const size_t   nbytes   = ggml_nbytes(weight);
    const uint64_t model_id = 1;
    ggml_backend_sycl_register_weight_identity(weight, 0, 4096, nbytes, model_id);

    ggml_sycl_cache_id key = ggml_backend_sycl_get_weight_cache_key(weight, 0);
    print_cache_id("weight key", key);

    TEST_ASSERT(key.valid, "cache key should be valid");
    TEST_ASSERT(key.has_gguf, "registered weight should carry GGUF identity");
    TEST_ASSERT(key.file_offs == 4096, "file_offs should be what was registered");
    TEST_ASSERT(key.nbytes == nbytes, "nbytes should be what was registered");
    // A zero file_id would mean "no file identity", which is the state that
    // makes the model_id-free comparison unsafe. It must never be zero once
    // has_gguf is set.
    TEST_ASSERT(key.file_id != 0, "GGUF-backed key must carry a non-zero file identity");
    TEST_ASSERT(key.model_id == model_id, "key should report the model it was registered under");
    TEST_ASSERT(key.name_hash != 0, "name_hash should still be carried for diagnostics");

    // The key looked up twice must be the same key: cache lookups happen on
    // every dispatch, and an identity that drifts stages the weight again.
    ggml_sycl_cache_id again = ggml_backend_sycl_get_weight_cache_key(weight, 0);
    TEST_ASSERT(ggml_sycl::detail::cache_id_equal(key, again), "cache key changed between lookups");

    return true;
}

// =============================================================================
// Test 2: model separation comes from file identity, not from model_id
//
// The policy statement, and the one that must not regress: two models holding
// a tensor with the SAME name, shape, type and file offset are still distinct
// weights, because their files are distinct.  If this ever passes only because
// model_id is back in the comparison, test 4 fails.
// =============================================================================
static bool test_model_separation_comes_from_file_identity() {
    printf("TEST: test_model_separation_comes_from_file_identity\n");

    test_ctx t;
    TEST_ASSERT(t.init(4 * 1024 * 1024), "test context init failed");

    // Same name, same shape, same split index, same offset -- everything a
    // name+shape key would look at is identical between the two models.
    const char * shared_name = "blk.0.ffn_gate.weight";

    ggml_tensor * from_model_a = t.weight(shared_name, GGML_TYPE_Q4_0, 256, 64);
    TEST_ASSERT(from_model_a != nullptr, "model A weight allocation failed");
    ggml_backend_sycl_register_weight_identity(from_model_a, 0, 4096, ggml_nbytes(from_model_a), 1);
    const ggml_sycl_cache_id key_a = ggml_backend_sycl_get_weight_cache_key(from_model_a, 0);
    print_cache_id("model 1", key_a);

    ggml_tensor * from_model_b = t.weight(shared_name, GGML_TYPE_Q4_0, 256, 64);
    TEST_ASSERT(from_model_b != nullptr, "model B weight allocation failed");
    ggml_backend_sycl_register_weight_identity(from_model_b, 0, 4096, ggml_nbytes(from_model_b), 2);
    const ggml_sycl_cache_id key_b = ggml_backend_sycl_get_weight_cache_key(from_model_b, 0);
    print_cache_id("model 2", key_b);

    TEST_ASSERT(key_a.valid && key_b.valid, "both keys should be valid");
    TEST_ASSERT(key_a.has_gguf && key_b.has_gguf, "both weights should carry GGUF identity");
    TEST_ASSERT(!ggml_sycl::detail::cache_id_equal(key_a, key_b), "two models' weights must not share one key");
    // And it must be the FILE that separates them. Everything else about these
    // two weights is identical, so if file_id matched they would be one entry.
    TEST_ASSERT(key_a.file_id != key_b.file_id, "separation must come from file identity");
    TEST_ASSERT(key_a.file_idx == key_b.file_idx, "the split index is the same in both models, by construction");
    TEST_ASSERT(key_a.file_offs == key_b.file_offs, "the offset is the same in both models, by construction");
    TEST_ASSERT(key_a.name_hash == key_b.name_hash, "the name is the same in both models, by construction");

    printf("  distinct file identities separate two models sharing a tensor name\n");
    return true;
}

// =============================================================================
// Test 3: distinct weights within one model stay distinct
//
// Real weights sit at distinct offsets. The identity is physical, so that is
// what keeps them apart -- not their names.
// =============================================================================
static bool test_distinct_weights_have_distinct_keys() {
    printf("TEST: test_distinct_weights_have_distinct_keys\n");

    test_ctx t;
    TEST_ASSERT(t.init(8 * 1024 * 1024), "test context init failed");

    const char * tensor_names[] = { "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
                                    "blk.1.ffn_gate.weight", "blk.1.ffn_up.weight", "blk.1.ffn_down.weight" };
    const int    num_tensors    = sizeof(tensor_names) / sizeof(tensor_names[0]);

    key_set unique_keys;
    for (int i = 0; i < num_tensors; ++i) {
        ggml_tensor * weight = t.weight(tensor_names[i], GGML_TYPE_Q4_0, 256, 64);
        TEST_ASSERT(weight != nullptr, "tensor allocation failed");

        const size_t nbytes = ggml_nbytes(weight);
        ggml_backend_sycl_register_weight_identity(weight, 0, static_cast<size_t>(i) * nbytes, nbytes, 1);

        ggml_sycl_cache_id key = ggml_backend_sycl_get_weight_cache_key(weight, 0);
        print_cache_id(tensor_names[i], key);

        TEST_ASSERT(key.valid, "cache key should be valid");
        TEST_ASSERT(key.has_gguf, "registered weight should carry GGUF identity");
        TEST_ASSERT(unique_keys.emplace(key).second, "distinct weights collided on one key");
    }

    printf("  all %d weights have distinct keys\n", num_tensors);
    return true;
}

// =============================================================================
// Test 4: two names over one set of bytes are ONE weight
//
// The tied embedding / output head case, stated as policy.  It is the inverse
// of test 3 and the reason name_hash cannot be in the GGUF-backed comparison:
// the two tensors differ in name and in nothing else that matters.
// =============================================================================
static bool test_same_bytes_are_one_weight() {
    printf("TEST: test_same_bytes_are_one_weight\n");

    test_ctx t;
    TEST_ASSERT(t.init(4 * 1024 * 1024), "test context init failed");

    ggml_tensor * as_embedding = t.weight("token_embd.weight", GGML_TYPE_Q4_0, 256, 64);
    ggml_tensor * as_output    = t.weight("output.weight", GGML_TYPE_Q4_0, 256, 64);
    TEST_ASSERT(as_embedding != nullptr && as_output != nullptr, "tied tensor allocation failed");

    const size_t   nbytes   = ggml_nbytes(as_embedding);
    const uint64_t model_id = 5;
    ggml_backend_sycl_register_weight_identity(as_embedding, 0, 8192, nbytes, model_id);
    ggml_backend_sycl_register_weight_identity(as_output, 0, 8192, nbytes, model_id);

    const ggml_sycl_cache_id embedding_key = ggml_backend_sycl_get_weight_cache_key(as_embedding, 0);
    const ggml_sycl_cache_id output_key    = ggml_backend_sycl_get_weight_cache_key(as_output, 0);
    print_cache_id("token_embd.weight", embedding_key);
    print_cache_id("output.weight", output_key);

    TEST_ASSERT(embedding_key.valid && output_key.valid, "both keys should be valid");
    TEST_ASSERT(embedding_key.has_gguf && output_key.has_gguf, "both should carry GGUF identity");
    TEST_ASSERT(embedding_key.name_hash != output_key.name_hash, "the two names differ, by construction");
    TEST_ASSERT(ggml_sycl::detail::cache_id_equal(embedding_key, output_key),
                "tied weights over one file offset must share one cache identity");

    // Equality is only half of it: an unordered_map also needs the hash to
    // agree, or the entry the cache already holds is never found.
    key_set unique_keys;
    unique_keys.emplace(embedding_key);
    TEST_ASSERT(!unique_keys.emplace(output_key).second, "tied weights must land in the same hash bucket");

    printf("  a tied embedding/output head resolves to one cache entry\n");
    return true;
}

// =============================================================================
// Test 5: without GGUF identity, every logical field still counts
//
// MoE intermediates and other runtime tensors have no bytes on disk to key on.
// They must fail toward isolation: name and model stay in their identity.
// =============================================================================
static bool test_non_gguf_tensors_keep_logical_identity() {
    printf("TEST: test_non_gguf_tensors_keep_logical_identity\n");

    test_ctx t;
    TEST_ASSERT(t.init(4 * 1024 * 1024), "test context init failed");

    const char * tensor_names[] = { "ffn_moe_probs",   "ffn_moe_probs.0", "ffn_moe_probs.1",
                                    "ffn_moe_indices", "expert_weights",  "expert_ids" };
    const int    num_tensors    = sizeof(tensor_names) / sizeof(tensor_names[0]);

    key_set unique_keys;
    for (int i = 0; i < num_tensors; ++i) {
        // No register_weight_identity call: these are compute tensors.
        ggml_tensor * tensor = t.weight(tensor_names[i], GGML_TYPE_F32, 128, 32);
        TEST_ASSERT(tensor != nullptr, "tensor allocation failed");

        ggml_sycl_cache_id key = ggml_backend_sycl_get_weight_cache_key(tensor, 0);
        print_cache_id(tensor_names[i], key);

        TEST_ASSERT(key.valid, "cache key should be valid");
        TEST_ASSERT(!key.has_gguf, "a tensor with no registered identity must say so");
        TEST_ASSERT(key.file_id == 0, "a tensor with no file must carry no file identity");
        TEST_ASSERT(unique_keys.emplace(key).second, "non-GGUF tensors collided on one key");
    }

    printf("  all %d runtime tensors have distinct keys and no file identity\n", num_tensors);
    return true;
}

// =============================================================================
// Test 6: cache_id_equal implements the policy on both sides of has_gguf
//
// Built from literal ids rather than tensors, so it pins the comparator itself
// rather than whatever the key builder happens to produce.
// =============================================================================
static bool test_cache_id_equality_policy() {
    printf("TEST: test_cache_id_equality_policy\n");

    ggml_sycl_cache_id gguf{};
    gguf.valid     = true;
    gguf.model_id  = 1;
    gguf.has_gguf  = true;
    gguf.file_id   = 0xF11E1D;
    gguf.file_idx  = 0;
    gguf.file_offs = 4096;
    gguf.nbytes    = 1024;
    gguf.name_hash = 0x12345678;
    gguf.type      = GGML_TYPE_Q4_0;
    gguf.aux_id    = 0;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        gguf.ne[i]           = 0;
        gguf.tp_local_ne[i]  = 0;
        gguf.tp_offset_ne[i] = 0;
    }
    gguf.ne[0]         = 256;
    gguf.ne[1]         = 64;
    gguf.tp_sharded    = false;
    gguf.tp_rank       = 0;
    gguf.tp_world_size = 1;

    ggml_sycl_cache_id other = gguf;
    TEST_ASSERT(ggml_sycl::detail::cache_id_equal(gguf, other), "identical ids should be equal");

    // The two exclusions, asserted directly. Reintroducing either field into the
    // GGUF-backed comparison fails here and nowhere else in this file.
    other           = gguf;
    other.name_hash = 0x87654321;
    TEST_ASSERT(ggml_sycl::detail::cache_id_equal(gguf, other), "name must not split a GGUF-backed identity");
    TEST_ASSERT(ggml_sycl::detail::cache_id_hash{}(gguf) == ggml_sycl::detail::cache_id_hash{}(other),
                "hash must not split a GGUF-backed identity by name either");

    other          = gguf;
    other.model_id = 99;
    TEST_ASSERT(ggml_sycl::detail::cache_id_equal(gguf, other), "model must not split a GGUF-backed identity");
    TEST_ASSERT(ggml_sycl::detail::cache_id_hash{}(gguf) == ggml_sycl::detail::cache_id_hash{}(other),
                "hash must not split a GGUF-backed identity by model either");

    // What DOES separate two GGUF-backed weights: the bytes they name.
    other         = gguf;
    other.file_id = 0xD1FF;
    TEST_ASSERT(!ggml_sycl::detail::cache_id_equal(gguf, other), "different files are different weights");

    other           = gguf;
    other.file_offs = 8192;
    TEST_ASSERT(!ggml_sycl::detail::cache_id_equal(gguf, other), "different offsets are different weights");

    other        = gguf;
    other.nbytes = 2048;
    TEST_ASSERT(!ggml_sycl::detail::cache_id_equal(gguf, other), "different sizes are different weights");

    other       = gguf;
    other.ne[1] = 128;
    TEST_ASSERT(!ggml_sycl::detail::cache_id_equal(gguf, other), "different shapes are different weights");

    // Without file identity, the logical fields are all there is, so both come
    // back into force. This is the isolation half of the policy.
    ggml_sycl_cache_id logical = gguf;
    logical.has_gguf           = false;
    logical.file_id            = 0;
    logical.file_offs          = 0;

    other           = logical;
    other.name_hash = 0x87654321;
    TEST_ASSERT(!ggml_sycl::detail::cache_id_equal(logical, other), "name must split an identity with no file");

    other          = logical;
    other.model_id = 99;
    TEST_ASSERT(!ggml_sycl::detail::cache_id_equal(logical, other), "model must split an identity with no file");

    other        = logical;
    other.aux_id = 123;
    TEST_ASSERT(!ggml_sycl::detail::cache_id_equal(logical, other), "aux_id must split an identity with no file");

    // A GGUF-backed id and a logical one are never the same weight, whatever
    // else matches.
    TEST_ASSERT(!ggml_sycl::detail::cache_id_equal(gguf, logical), "has_gguf must separate the two identity kinds");

    printf("  cache_id_equal keys GGUF-backed weights physically and the rest logically\n");
    return true;
}

// =============================================================================
// Test 7: MoE expert keys stay unique (llama.cpp-twc)
//
// This preserves the original logical-id collision regression independently
// of the canonical GGUF/load-scoped builder. The original regression was a
// null identity hash collapsing every expert onto one key.
// =============================================================================
static bool test_moe_expert_keys_unique() {
    printf("TEST: test_moe_expert_keys_unique\n");

    const uint64_t model_id         = 1;
    const int      num_weight_types = 3;  // gate, up, down
    const int      num_experts      = 8;
    const int      num_layers       = 2;

    key_set  unique_keys;
    size_t   n_keys          = 0;
    uint64_t base_cache_uuid = 1000;

    for (int l = 0; l < num_layers; ++l) {
        for (int w = 0; w < num_weight_types; ++w) {
            const uint64_t tensor_cache_uuid = base_cache_uuid++;

            for (int e = 0; e < num_experts; ++e) {
                ggml_sycl_cache_id id{};
                id.valid         = true;
                id.model_id      = model_id;
                id.has_gguf      = false;
                id.file_id       = 0;
                id.file_idx      = 0;
                id.file_offs     = 0;
                id.nbytes        = 0;
                id.name_hash     = 0;  // MoE expert keys separate on aux_id
                id.type          = GGML_TYPE_COUNT;
                id.tp_sharded    = false;
                id.tp_rank       = 0;
                id.tp_world_size = 1;
                for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                    id.ne[i]           = 0;
                    id.tp_local_ne[i]  = 0;
                    id.tp_offset_ne[i] = 0;
                }

                // Mirrors ggml_sycl_hash_combine(cache_uuid, expert_id).
                size_t aux = tensor_cache_uuid;
                aux        = aux ^ (static_cast<size_t>(e) + 0x9e3779b97f4a7c15ULL + (aux << 6) + (aux >> 2));
                id.aux_id  = aux;

                if (!unique_keys.emplace(id).second) {
                    fprintf(stderr, "  FAIL: collision for layer=%d weight=%d expert=%d\n", l, w, e);
                    return false;
                }
                ++n_keys;
            }
        }
    }

    printf("  all %zu MoE expert keys are unique on the name_hash=0 path\n", n_keys);

    // Same tensor, same expert: one key. Same tensor, different expert: two.
    const uint64_t shared_uuid = 9999;

    ggml_sycl_cache_id expert_0{};
    expert_0.valid     = true;
    expert_0.model_id  = model_id;
    expert_0.name_hash = 0;
    expert_0.type      = GGML_TYPE_COUNT;
    size_t aux_0       = shared_uuid;
    aux_0              = aux_0 ^ (0ULL + 0x9e3779b97f4a7c15ULL + (aux_0 << 6) + (aux_0 >> 2));
    expert_0.aux_id    = aux_0;

    ggml_sycl_cache_id expert_0_again = expert_0;
    TEST_ASSERT(ggml_sycl::detail::cache_id_equal(expert_0, expert_0_again),
                "the same tensor and expert should be one key");

    ggml_sycl_cache_id expert_1 = expert_0;
    size_t             aux_1    = shared_uuid;
    aux_1                       = aux_1 ^ (1ULL + 0x9e3779b97f4a7c15ULL + (aux_1 << 6) + (aux_1 >> 2));
    expert_1.aux_id             = aux_1;
    TEST_ASSERT(!ggml_sycl::detail::cache_id_equal(expert_0, expert_1), "different experts should be different keys");

    return true;
}

// =============================================================================
// Test 8: MoE identities are exact ModelToken + GGUF parent-slice identities.
// =============================================================================
static bool test_load_scoped_moe_identity_contract() {
    printf("TEST: test_load_scoped_moe_identity_contract\n");

    ggml_sycl_cache_id expert{};
    expert.valid           = true;
    expert.load_scoped     = true;
    expert.model_id        = 11;
    expert.load_txn_id     = 21;
    expert.model_slot      = 3;
    expert.slot_generation = 7;
    expert.has_gguf        = true;
    expert.file_id         = 0xabc;
    expert.file_idx        = 1;
    expert.file_offs       = 0x10000 + 4 * 4096;
    expert.nbytes          = 4096;
    expert.name_hash       = 0x111;
    expert.type            = GGML_TYPE_Q4_0;
    expert.ne[0]           = 256;
    expert.ne[1]           = 64;
    expert.ne[2]           = 1;
    expert.ne[3]           = 1;
    expert.tp_world_size   = 1;

    // Two graph wrappers over the same token and parent bytes are one expert;
    // wrapper-local metadata churn cannot split the physical slice.
    ggml_sycl_cache_id wrapper = expert;
    wrapper.name_hash = 0x222;
    TEST_ASSERT(ggml_sycl::detail::cache_id_equal(expert, wrapper),
                "same-token wrappers over one GGUF expert slice must compare equal");
    TEST_ASSERT(ggml_sycl::detail::cache_id_hash{}(expert) == ggml_sycl::detail::cache_id_hash{}(wrapper),
                "same-token wrappers must hash equally");

    ggml_sycl_cache_id other = expert;
    other.model_id = 12;
    TEST_ASSERT(!ggml_sycl::detail::cache_id_equal(expert, other), "different models must be isolated");
    other = expert;
    other.load_txn_id = 22;
    TEST_ASSERT(!ggml_sycl::detail::cache_id_equal(expert, other), "different loads must be isolated");
    other = expert;
    other.model_slot = 4;
    TEST_ASSERT(!ggml_sycl::detail::cache_id_equal(expert, other), "different slots must be isolated");
    other = expert;
    other.slot_generation = 8;
    TEST_ASSERT(!ggml_sycl::detail::cache_id_equal(expert, other), "stale slot generations must be isolated");
    other = expert;
    other.file_offs += expert.nbytes;
    TEST_ASSERT(!ggml_sycl::detail::cache_id_equal(expert, other), "different GGUF expert offsets must be isolated");

    // Dense GGUF policy remains physical and intentionally ignores lifecycle fields.
    ggml_sycl_cache_id dense_a = expert;
    dense_a.load_scoped = false;
    ggml_sycl_cache_id dense_b = dense_a;
    dense_b.model_id = 99;
    dense_b.load_txn_id = 100;
    dense_b.model_slot = 9;
    dense_b.slot_generation = 10;
    TEST_ASSERT(ggml_sycl::detail::cache_id_equal(dense_a, dense_b),
                "dense GGUF identity policy must remain load-independent");

    return true;
}

// =============================================================================
// Main
// =============================================================================
int main() {
    printf("=== SYCL weight cache identity policy tests (llama.cpp-n3pw Option A) ===\n\n");

    RUN_TEST(test_gguf_key_carries_file_identity);
    RUN_TEST(test_model_separation_comes_from_file_identity);
    RUN_TEST(test_distinct_weights_have_distinct_keys);
    RUN_TEST(test_same_bytes_are_one_weight);
    RUN_TEST(test_non_gguf_tensors_keep_logical_identity);
    RUN_TEST(test_cache_id_equality_policy);
    RUN_TEST(test_moe_expert_keys_unique);
    RUN_TEST(test_load_scoped_moe_identity_contract);

    printf("\n=== Summary ===\n");
    printf("Tests run: %d, Passed: %d, Failed: %d\n", g_tests_run, g_tests_passed, g_tests_failed);

    if (g_tests_failed > 0) {
        printf("\nFAILED: the weight cache key no longer implements the n3pw physical-identity ruling.\n");
        return 1;
    }

    printf("\nPASS: weight cache identity follows the physical-identity policy.\n");
    return 0;
}

#endif  // GGML_USE_SYCL
