//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

// Unit tests for tuning-cache-io.hpp
// Tests JSON serialization/deserialization, file I/O, and atomic writes
//
// llama.cpp-7n6n: several of the tests below (the
// matmul-dispatch-tuning ones: cache_file_roundtrip, version_check,
// atomic_write) exercise save_cache()/load_cache()/get_cache_file(), which
// resolve their directory through get_cache_dir() -- XDG_CACHE_HOME if set,
// else the REAL $HOME/.cache/llama.cpp/sycl-tuning. This binary must never
// touch that real directory, so `main()` below refuses to run at all
// (SKIP, exit 77) unless XDG_CACHE_HOME is already set to a scratch
// directory. tests/CMakeLists.txt's registration sets it (and
// GGML_SYCL_TUNING_CACHE_DIR alongside it) via ENVIRONMENT for exactly this
// reason -- a direct invocation of this binary must set both the same way.
// The NEWER "ubatch" entry-kind tests further below (ubatch_cache_*) do NOT
// need either variable: they pass an explicit cache_dir argument straight
// to save_ubatch_cache()/load_ubatch_cache(), never through get_cache_dir().

#include "../ggml/src/ggml-sycl/tuning-cache-io.hpp"

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

using namespace ggml_sycl_tuning;

// Test counter
static int g_passed = 0;
static int g_failed = 0;

// llama.cpp-7n6n: the macro used to also declare
// `static bool test_##name##_registered = [] { return true; }();` -- dead
// code (RUN_TEST() below calls test_##name() by name; nothing ever reads
// the `_registered` variable), and the sole source of the 26
// -Wunused-variable warnings this task made visible in every build log.
#define TEST(name)             \
    static bool test_##name(); \
    static bool test_##name()

#define ASSERT(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "  ASSERT FAILED: " << #cond << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            return false; \
        } \
    } while(0)

#define RUN_TEST(name) \
    do { \
        std::cout << "Running " << #name << "... "; \
        if (test_##name()) { \
            std::cout << "[PASS]\n"; \
            g_passed++; \
        } else { \
            std::cout << "[FAIL]\n"; \
            g_failed++; \
        } \
    } while(0)

// =============================================================================
// Test: get_cache_dir returns valid directory
// =============================================================================
TEST(get_cache_dir) {
    std::string dir = get_cache_dir();
    ASSERT(!dir.empty());
    ASSERT(dir.find("llama.cpp") != std::string::npos);
    ASSERT(dir.find("sycl-tuning") != std::string::npos);
    return true;
}

// =============================================================================
// Test: get_cache_dir respects XDG_CACHE_HOME
// =============================================================================
TEST(get_cache_dir_xdg) {
    // Save original value
    const char* original = std::getenv("XDG_CACHE_HOME");
    std::string original_str = original ? original : "";

    // Set custom XDG_CACHE_HOME
    setenv("XDG_CACHE_HOME", "/custom/cache/path", 1);
    std::string dir = get_cache_dir();
    ASSERT(dir == "/custom/cache/path/llama.cpp/sycl-tuning");

    // Restore original
    if (original_str.empty()) {
        unsetenv("XDG_CACHE_HOME");
    } else {
        setenv("XDG_CACHE_HOME", original_str.c_str(), 1);
    }
    return true;
}

// =============================================================================
// Test: sanitize_device_name handles special characters
// =============================================================================
TEST(sanitize_device_name) {
    // Normal device name with spaces
    ASSERT(sanitize_device_name("Intel Arc A770") == "Intel_Arc_A770");

    // Device name with special characters
    ASSERT(sanitize_device_name("Intel(R) Arc(TM) A770") == "IntelR_ArcTM_A770");

    // Device name with colons and slashes
    ASSERT(sanitize_device_name("Device: GPU/0") == "Device_GPU0");

    // Empty string
    ASSERT(sanitize_device_name("") == "");

    // Already clean name
    ASSERT(sanitize_device_name("Arc-A770_v2") == "Arc-A770_v2");

    return true;
}

// =============================================================================
// Test: get_cache_file generates valid path
// =============================================================================
TEST(get_cache_file) {
    std::string file = get_cache_file("Intel Arc A770");
    ASSERT(file.find("Intel_Arc_A770") != std::string::npos);
    ASSERT(file.find(".json") != std::string::npos);
    ASSERT(file.find("llama.cpp") != std::string::npos);
    return true;
}

// =============================================================================
// Test: params_to_json serialization
// =============================================================================
TEST(params_to_json) {
    TunedParams p;
    p.tile_m = 8;
    p.tile_n = 32;
    p.tile_k = 64;
    p.workgroup_size = 256;
    p.slm_kb = 32;
    p.prefetch_depth = 2;
    p.use_dpas = true;
    p.layout_mode = 2;

    std::string json = params_to_json(p);

    ASSERT(json.find("\"tile_m\":8") != std::string::npos);
    ASSERT(json.find("\"tile_n\":32") != std::string::npos);
    ASSERT(json.find("\"tile_k\":64") != std::string::npos);
    ASSERT(json.find("\"workgroup_size\":256") != std::string::npos);
    ASSERT(json.find("\"slm_kb\":32") != std::string::npos);
    ASSERT(json.find("\"prefetch_depth\":2") != std::string::npos);
    ASSERT(json.find("\"use_dpas\":true") != std::string::npos);
    ASSERT(json.find("\"layout_mode\":2") != std::string::npos);

    return true;
}

// =============================================================================
// Test: params_from_json deserialization
// =============================================================================
TEST(params_from_json) {
    std::string json = R"({"tile_m":16,"tile_n":64,"tile_k":32,"workgroup_size":128,"slm_kb":16,"prefetch_depth":3,"use_dpas":false,"layout_mode":1})";

    TunedParams p = params_from_json(json);

    ASSERT(p.tile_m == 16);
    ASSERT(p.tile_n == 64);
    ASSERT(p.tile_k == 32);
    ASSERT(p.workgroup_size == 128);
    ASSERT(p.slm_kb == 16);
    ASSERT(p.prefetch_depth == 3);
    ASSERT(p.use_dpas == false);
    ASSERT(p.layout_mode == 1);

    return true;
}

// =============================================================================
// Test: TunedParams roundtrip serialization
// =============================================================================
TEST(params_roundtrip) {
    TunedParams original;
    original.tile_m = 32;
    original.tile_n = 16;
    original.tile_k = 128;
    original.workgroup_size = 512;
    original.slm_kb = 64;
    original.prefetch_depth = 4;
    original.use_dpas = true;
    original.layout_mode = 3;

    std::string json = params_to_json(original);
    TunedParams restored = params_from_json(json);

    ASSERT(original.tile_m == restored.tile_m);
    ASSERT(original.tile_n == restored.tile_n);
    ASSERT(original.tile_k == restored.tile_k);
    ASSERT(original.workgroup_size == restored.workgroup_size);
    ASSERT(original.slm_kb == restored.slm_kb);
    ASSERT(original.prefetch_depth == restored.prefetch_depth);
    ASSERT(original.use_dpas == restored.use_dpas);
    ASSERT(original.layout_mode == restored.layout_mode);

    return true;
}

// =============================================================================
// Test: key_to_string and key_from_string
// =============================================================================
TEST(key_serialization) {
    TuningKey key;
    key.quant_type = 2;  // GGML_TYPE_Q4_0
    key.batch_bucket = BatchBucket::MEDIUM;
    key.K = 4096;
    key.N = 32000;

    std::string str = key_to_string(key);
    ASSERT(str == "2_2_4096_32000");

    TuningKey restored = key_from_string(str);
    ASSERT(restored.quant_type == 2);
    ASSERT(restored.batch_bucket == BatchBucket::MEDIUM);
    ASSERT(restored.K == 4096);
    ASSERT(restored.N == 32000);

    return true;
}

// =============================================================================
// Test: entry_to_json and entry_from_json
// =============================================================================
TEST(entry_serialization) {
    TuningEntry entry;
    entry.key.quant_type = 8;  // GGML_TYPE_Q8_0
    entry.key.batch_bucket = BatchBucket::SINGLE;
    entry.key.K = 4096;
    entry.key.N = 4096;
    entry.params.tile_m = 8;
    entry.params.tile_n = 32;
    entry.params.tile_k = 64;
    entry.params.workgroup_size = 256;
    entry.params.slm_kb = 32;
    entry.params.prefetch_depth = 2;
    entry.params.use_dpas = true;
    entry.params.layout_mode = 1;
    entry.measured_tflops = 12.5f;
    entry.timestamp = 1700000000;

    std::string json = entry_to_json(entry);

    // Verify key is present
    ASSERT(json.find("\"key\":\"8_0_4096_4096\"") != std::string::npos);
    ASSERT(json.find("\"params\":") != std::string::npos);
    ASSERT(json.find("\"measured_tflops\":12.5") != std::string::npos);
    ASSERT(json.find("\"timestamp\":1700000000") != std::string::npos);

    TuningEntry restored = entry_from_json(json);
    ASSERT(restored.key.quant_type == 8);
    ASSERT(restored.key.batch_bucket == BatchBucket::SINGLE);
    ASSERT(restored.key.K == 4096);
    ASSERT(restored.key.N == 4096);
    ASSERT(restored.params.tile_m == 8);
    ASSERT(restored.params.use_dpas == true);
    ASSERT(std::abs(restored.measured_tflops - 12.5f) < 0.01f);

    return true;
}

// =============================================================================
// Test: create_dir_recursive
// =============================================================================
TEST(create_dir_recursive) {
    std::string test_dir = "/tmp/llama_test_cache_dir_" + std::to_string(getpid());
    std::string nested_dir = test_dir + "/level1/level2/level3";

    // Create nested directory
    create_dir_recursive(nested_dir);

    // Check directory exists
    struct stat st;
    ASSERT(stat(nested_dir.c_str(), &st) == 0);
    ASSERT(S_ISDIR(st.st_mode));

    // Cleanup
    rmdir((test_dir + "/level1/level2/level3").c_str());
    rmdir((test_dir + "/level1/level2").c_str());
    rmdir((test_dir + "/level1").c_str());
    rmdir(test_dir.c_str());

    return true;
}

// =============================================================================
// Test: TuningCache for_each iteration
// =============================================================================
TEST(cache_for_each) {
    TuningCache cache;

    // Insert some entries
    TuningEntry e1;
    e1.key.quant_type = 2;
    e1.key.batch_bucket = BatchBucket::SINGLE;
    e1.key.K = 4096;
    e1.key.N = 4096;
    e1.params.tile_m = 8;
    cache.insert(e1);

    TuningEntry e2;
    e2.key.quant_type = 8;
    e2.key.batch_bucket = BatchBucket::MEDIUM;
    e2.key.K = 4096;
    e2.key.N = 32000;
    e2.params.tile_m = 16;
    cache.insert(e2);

    // Count entries via for_each
    int count = 0;
    cache.for_each([&count](const TuningKey& key, const TuningEntry& entry) {
        (void)key;
        (void)entry;
        count++;
    });

    ASSERT(count == 2);
    ASSERT(cache.size() == 2);

    return true;
}

// =============================================================================
// Test: save_cache and load_cache roundtrip
// =============================================================================
TEST(cache_file_roundtrip) {
    // Use unique device name to avoid conflicts
    std::string device_name = "TestDevice_" + std::to_string(getpid());

    // Create cache with entries
    TuningCache cache;

    TuningEntry e1;
    e1.key.quant_type = 2;
    e1.key.batch_bucket = BatchBucket::SINGLE;
    e1.key.K = 4096;
    e1.key.N = 4096;
    e1.params.tile_m = 8;
    e1.params.tile_n = 32;
    e1.params.tile_k = 64;
    e1.params.workgroup_size = 256;
    e1.params.slm_kb = 32;
    e1.params.prefetch_depth = 2;
    e1.params.use_dpas = true;
    e1.params.layout_mode = 1;
    e1.measured_tflops = 15.5f;
    e1.timestamp = 1700000000;
    cache.insert(e1);

    TuningEntry e2;
    e2.key.quant_type = 8;
    e2.key.batch_bucket = BatchBucket::MEDIUM;
    e2.key.K = 4096;
    e2.key.N = 32000;
    e2.params.tile_m = 16;
    e2.params.tile_n = 64;
    e2.params.tile_k = 32;
    e2.params.workgroup_size = 512;
    e2.params.slm_kb = 64;
    e2.params.prefetch_depth = 4;
    e2.params.use_dpas = false;
    e2.params.layout_mode = 2;
    e2.measured_tflops = 8.2f;
    e2.timestamp = 1700001000;
    cache.insert(e2);

    // Save cache
    bool save_result = save_cache(cache, device_name);
    ASSERT(save_result);

    // Verify file exists
    ASSERT(cache_exists(device_name));

    // Load into new cache
    TuningCache loaded_cache;
    bool load_result = load_cache(loaded_cache, device_name);
    ASSERT(load_result);

    // Verify loaded cache has same entries
    ASSERT(loaded_cache.size() == 2);

    // Verify first entry
    auto lookup1 = loaded_cache.lookup(e1.key);
    ASSERT(lookup1.has_value());
    ASSERT(lookup1->params.tile_m == 8);
    ASSERT(lookup1->params.use_dpas == true);

    // Verify second entry
    auto lookup2 = loaded_cache.lookup(e2.key);
    ASSERT(lookup2.has_value());
    ASSERT(lookup2->params.tile_m == 16);
    ASSERT(lookup2->params.use_dpas == false);

    // Cleanup
    delete_cache(device_name);
    ASSERT(!cache_exists(device_name));

    return true;
}

// =============================================================================
// Test: load_cache handles missing file gracefully
// =============================================================================
TEST(load_missing_file) {
    TuningCache cache;
    bool result = load_cache(cache, "NonExistentDevice_12345678");
    ASSERT(result == false);
    ASSERT(cache.size() == 0);
    return true;
}

// =============================================================================
// Test: version check rejects incompatible versions
// =============================================================================
TEST(version_check) {
    std::string device_name = "TestVersionDevice_" + std::to_string(getpid());
    std::string path = get_cache_file(device_name);

    // Create directory and write file with wrong version
    create_dir_recursive(get_cache_dir());
    std::ofstream f(path);
    f << "{\n";
    f << "  \"version\": 999,\n";  // Invalid version
    f << "  \"device\": \"" << device_name << "\",\n";
    f << "  \"entries\": []\n";
    f << "}\n";
    f.close();

    // Try to load - should fail due to version mismatch
    TuningCache cache;
    bool result = load_cache(cache, device_name);
    ASSERT(result == false);

    // Cleanup
    std::remove(path.c_str());

    return true;
}

// =============================================================================
// Test: CACHE_VERSION constant
// =============================================================================
TEST(cache_version) {
    ASSERT(CACHE_VERSION >= 1);
    return true;
}

// =============================================================================
// Test: parse_int handles edge cases
// =============================================================================
TEST(parse_int_edge_cases) {
    // Normal case
    ASSERT(parse_int("{\"value\":42}", "value") == 42);

    // Negative number
    ASSERT(parse_int("{\"value\":-123}", "value") == -123);

    // Zero
    ASSERT(parse_int("{\"value\":0}", "value") == 0);

    // Missing key
    ASSERT(parse_int("{\"other\":42}", "value") == 0);

    // With whitespace
    ASSERT(parse_int("{\"value\":  42}", "value") == 42);

    return true;
}

// =============================================================================
// Test: parse_bool handles edge cases
// =============================================================================
TEST(parse_bool_edge_cases) {
    // True
    ASSERT(parse_bool("{\"flag\":true}", "flag") == true);

    // False
    ASSERT(parse_bool("{\"flag\":false}", "flag") == false);

    // Missing key
    ASSERT(parse_bool("{\"other\":true}", "flag") == false);

    // With whitespace
    ASSERT(parse_bool("{\"flag\":  true}", "flag") == true);

    return true;
}

// =============================================================================
// Test: parse_string handles edge cases
// =============================================================================
TEST(parse_string_edge_cases) {
    // Normal string
    ASSERT(parse_string("{\"name\":\"hello\"}", "name") == "hello");

    // Empty string
    ASSERT(parse_string("{\"name\":\"\"}", "name") == "");

    // String with spaces
    ASSERT(parse_string("{\"name\":\"hello world\"}", "name") == "hello world");

    // Missing key
    ASSERT(parse_string("{\"other\":\"hello\"}", "name") == "");

    return true;
}

// =============================================================================
// Test: Atomic write behavior (temp file cleanup on success)
// =============================================================================
TEST(atomic_write) {
    std::string device_name = "TestAtomicDevice_" + std::to_string(getpid());
    std::string path = get_cache_file(device_name);
    std::string temp_path = path + ".tmp";

    TuningCache cache;
    TuningEntry e;
    e.key.quant_type = 2;
    e.key.batch_bucket = BatchBucket::SINGLE;
    e.key.K = 4096;
    e.key.N = 4096;
    cache.insert(e);

    // Save cache
    bool result = save_cache(cache, device_name);
    ASSERT(result);

    // Verify final file exists
    struct stat st;
    ASSERT(stat(path.c_str(), &st) == 0);

    // Verify temp file does NOT exist (was renamed)
    ASSERT(stat(temp_path.c_str(), &st) != 0);

    // Cleanup
    delete_cache(device_name);

    return true;
}

// =============================================================================
// llama.cpp-7n6n (wires nphx Task 5): the "ubatch" entry kind (v2) --
// UbatchCacheKey/UbatchCacheEntry, their JSON (de)serialization, and
// save_ubatch_cache()/load_ubatch_cache(). Every test below passes its own
// explicit, unique per-pid /tmp directory straight to those two functions
// -- never get_cache_dir(), so these specific tests never depend on
// XDG_CACHE_HOME or the real $HOME. This is NOT true of every test in this
// file: see the file's own top-of-file comment for the
// pre-existing matmul-dispatch-tuning tests that DO resolve through
// get_cache_dir(), and the main()-level guard that keeps this binary from
// ever running against the real cache directory unsandboxed.
// =============================================================================

// Test: CACHE_VERSION was bumped to 5 (llama.cpp-1oa3, a device_key that
// names every participating device and its budget) -- v1 stays reserved for
// the pre-existing matmul dispatch-tuning format alone, v2 for the
// pre-kv_unified ubatch key shape, v3 (llama.cpp-3aos) for the pre-swa_full
// one, v4 (llama.cpp-uajm) for the single-device device_key.
TEST(cache_version_is_5) {
    ASSERT(CACHE_VERSION == 5);
    return true;
}

// Test: UbatchCacheKey::operator== compares every field; changing any ONE
// field must make two otherwise-identical keys compare unequal.
TEST(ubatch_key_equality) {
    UbatchCacheKey a;
    a.device_key = "Arc_Pro_B70@1.3.12345";
    a.model_name = "mistral-7b-v0.1";
    a.model_size = 4108931808ULL;
    a.model_hash = 0x0123456789abcdefULL;
    a.n_ctx      = 4096;
    a.n_batch    = 2048;
    a.flash_attn = true;

    UbatchCacheKey b = a;
    ASSERT(a == b);
    ASSERT(!(a != b));

    b            = a;
    b.device_key = "Arc_Pro_B50@1.3.12345";
    ASSERT(a != b);
    b            = a;
    b.model_name = "mistral-7b-v0.2";
    ASSERT(a != b);
    b            = a;
    b.model_size = a.model_size + 1;
    ASSERT(a != b);
    b            = a;
    b.model_hash = a.model_hash ^ 1ULL;
    ASSERT(a != b);
    b       = a;
    b.n_ctx = a.n_ctx + 1;
    ASSERT(a != b);
    b         = a;
    b.n_batch = a.n_batch + 1;
    ASSERT(a != b);
    b            = a;
    b.flash_attn = !a.flash_attn;
    ASSERT(a != b);

    // llama.cpp-7n6n: the fields added to UbatchCacheKey must each
    // independently affect equality too.
    a.n_seq_max = 1;
    a.type_k    = 1;  // GGML_TYPE_F16
    a.type_v    = 1;
    b           = a;
    ASSERT(a == b);

    b.n_seq_max = a.n_seq_max + 1;
    ASSERT(a != b);
    b        = a;
    b.type_k = a.type_k + 1;
    ASSERT(a != b);
    b        = a;
    b.type_v = a.type_v + 1;
    ASSERT(a != b);

    // llama.cpp-3aos: kv_unified must independently affect equality too --
    // added in CACHE_VERSION 3 once KV sizing started depending on it.
    a.kv_unified = false;
    b            = a;
    ASSERT(a == b);
    b.kv_unified = !a.kv_unified;
    ASSERT(a != b);

    // llama.cpp-uajm: swa_full must independently affect equality too --
    // added in CACHE_VERSION 4 once SWA-layer KV sizing started depending
    // on it (an SWA layer is sized as FULL when llama allocates it so).
    a.swa_full = false;
    b          = a;
    ASSERT(a == b);
    b.swa_full = !a.swa_full;
    ASSERT(a != b);

    return true;
}

// Exercise the persisted key boundary without requiring a kv_unified struct
// member, so this same regression can run against the pre-v3 header.
TEST(ubatch_kv_mode_json_roundtrip) {
    const std::string prefix =
        "{\"device_key\":\"Dev@1.0\",\"model_name\":\"mode-regression\","
        "\"model_size\":4096,\"model_hash\":17,\"n_ctx\":4096,\"n_batch\":2048,"
        "\"flash_attn\":true,\"n_seq_max\":4,\"type_k\":1,\"type_v\":1,"
        "\"swa_full\":false,\"kv_unified\":";
    const UbatchCacheKey separate = ubatch_key_from_json(prefix + "false}");
    const UbatchCacheKey unified  = ubatch_key_from_json(prefix + "true}");
    // Positive parse controls must pass before the mode distinction is tested.
    ASSERT(separate.device_key == "Dev@1.0");
    ASSERT(separate.model_name == "mode-regression");
    ASSERT(separate.n_ctx == 4096 && separate.n_seq_max == 4);
    ASSERT(unified.device_key == separate.device_key);
    ASSERT(unified.model_name == separate.model_name);
    ASSERT(unified.n_ctx == separate.n_ctx && unified.n_seq_max == separate.n_seq_max);
    ASSERT(separate != unified);

    const UbatchCacheKey separate_restored = ubatch_key_from_json("{" + ubatch_key_to_json(separate) + "}");
    const UbatchCacheKey unified_restored  = ubatch_key_from_json("{" + ubatch_key_to_json(unified) + "}");
    ASSERT(separate_restored == separate);
    ASSERT(unified_restored == unified);
    ASSERT(separate_restored != unified_restored);
    return true;
}

// llama.cpp-uajm: the swa_full half of the persisted key boundary. A raw-API
// context (llama_context_default_params(): swa_full=true) and a CLI run
// (common's default: swa_full=false) differ ONLY in this field, and their
// plans' KV bytes differ (an SWA layer is sized as FULL under swa_full), so
// the two must parse, round-trip, and compare as distinct keys.
TEST(ubatch_swa_full_json_roundtrip) {
    const std::string prefix =
        "{\"device_key\":\"Dev@1.0\",\"model_name\":\"swa-full-regression\","
        "\"model_size\":4096,\"model_hash\":17,\"n_ctx\":4096,\"n_batch\":2048,"
        "\"flash_attn\":true,\"n_seq_max\":1,\"type_k\":1,\"type_v\":1,"
        "\"kv_unified\":false,\"swa_full\":";
    const UbatchCacheKey windowed = ubatch_key_from_json(prefix + "false}");
    const UbatchCacheKey full     = ubatch_key_from_json(prefix + "true}");
    ASSERT(windowed.model_name == "swa-full-regression");
    ASSERT(windowed.swa_full == false);
    ASSERT(full.swa_full == true);
    ASSERT(full.n_ctx == windowed.n_ctx && full.kv_unified == windowed.kv_unified);
    ASSERT(windowed != full);

    const UbatchCacheKey windowed_restored = ubatch_key_from_json("{" + ubatch_key_to_json(windowed) + "}");
    const UbatchCacheKey full_restored     = ubatch_key_from_json("{" + ubatch_key_to_json(full) + "}");
    ASSERT(windowed_restored == windowed);
    ASSERT(full_restored == full);
    ASSERT(windowed_restored != full_restored);
    // The serialized form must carry the field by name -- a writer that
    // silently dropped it would still round-trip through the default.
    ASSERT(ubatch_key_to_json(full).find("\"swa_full\":true") != std::string::npos);
    ASSERT(ubatch_key_to_json(windowed).find("\"swa_full\":false") != std::string::npos);
    return true;
}

// Test: ubatch_entry_to_json / ubatch_entry_from_json round-trip every
// field, including the nested key.
TEST(ubatch_entry_serialization) {
    UbatchCacheEntry e;
    e.key.device_key = "Arc_Pro_B70@1.3.12345";
    e.key.model_name = "mistral-7b-v0.1";
    e.key.model_size = 4108931808ULL;
    e.key.model_hash = 0x0123456789abcdefULL;
    e.key.n_ctx      = 4096;
    e.key.n_batch    = 2048;
    e.key.flash_attn = true;
    e.n_ubatch       = 1024;
    e.reason         = "ladder";
    e.created        = "2026-09-11T12:34:56Z";

    std::string json = ubatch_entry_to_json(e);
    ASSERT(json.find("\"device_key\":\"Arc_Pro_B70@1.3.12345\"") != std::string::npos);
    ASSERT(json.find("\"model_size\":4108931808") != std::string::npos);
    ASSERT(json.find("\"n_ubatch\":1024") != std::string::npos);
    ASSERT(json.find("\"reason\":\"ladder\"") != std::string::npos);

    UbatchCacheEntry restored = ubatch_entry_from_json(json);
    ASSERT(restored.key == e.key);
    ASSERT(restored.n_ubatch == e.n_ubatch);
    ASSERT(restored.reason == e.reason);
    ASSERT(restored.created == e.created);

    return true;
}

// Test: save_ubatch_cache()/load_ubatch_cache() round-trip several entries
// for one device, under an explicit temp cache_dir (never get_cache_dir()).
TEST(ubatch_cache_file_roundtrip) {
    std::string cache_dir   = "/tmp/llama_test_ubatch_cache_" + std::to_string(getpid());
    std::string device_name = "TestUbatchDevice_" + std::to_string(getpid());

    UbatchCacheEntry e1;
    e1.key.device_key = "Arc_Pro_B70@1.3.12345";
    e1.key.model_name = "mistral-7b-v0.1";
    e1.key.model_size = 4108931808ULL;
    e1.key.model_hash = 111ULL;
    e1.key.n_ctx      = 4096;
    e1.key.n_batch    = 2048;
    e1.key.flash_attn = true;
    e1.n_ubatch       = 1024;
    e1.reason         = "ladder";
    e1.created        = "2026-09-11T12:34:56Z";

    UbatchCacheEntry e2 = e1;
    e2.key.n_ctx        = 8192;
    e2.n_ubatch         = 512;
    e2.reason           = "cached";

    std::vector<UbatchCacheEntry> entries{ e1, e2 };
    ASSERT(save_ubatch_cache(cache_dir, device_name, entries));

    struct stat st;
    ASSERT(stat(get_ubatch_cache_file(cache_dir, device_name).c_str(), &st) == 0);
    ASSERT(S_ISREG(st.st_mode));

    std::vector<UbatchCacheEntry> loaded;
    ASSERT(load_ubatch_cache(cache_dir, device_name, loaded));
    ASSERT(loaded.size() == 2);

    bool found_e1 = false, found_e2 = false;
    for (const auto & e : loaded) {
        if (e.key == e1.key) {
            ASSERT(e.n_ubatch == 1024);
            ASSERT(e.reason == "ladder");
            found_e1 = true;
        }
        if (e.key == e2.key) {
            ASSERT(e.n_ubatch == 512);
            ASSERT(e.reason == "cached");
            found_e2 = true;
        }
    }
    ASSERT(found_e1 && found_e2);

    std::remove(get_ubatch_cache_file(cache_dir, device_name).c_str());
    return true;
}

// Test: a v1 (or any non-current-version) ubatch cache file is rejected --
// load_ubatch_cache() must return false, leaving `entries` empty, exactly
// like load_cache()'s own version check.
TEST(ubatch_cache_v1_file_rejected) {
    std::string cache_dir   = "/tmp/llama_test_ubatch_cache_v1_" + std::to_string(getpid());
    std::string device_name = "TestUbatchV1Device_" + std::to_string(getpid());

    create_dir_recursive(cache_dir);
    std::string   path = get_ubatch_cache_file(cache_dir, device_name);
    std::ofstream f(path);
    f << "{\n";
    f << "  \"version\": 1,\n";
    f << "  \"device\": \"" << device_name << "\",\n";
    f << "  \"entries\": []\n";
    f << "}\n";
    f.close();

    std::vector<UbatchCacheEntry> loaded;
    bool                          result = load_ubatch_cache(cache_dir, device_name, loaded);
    ASSERT(result == false);
    ASSERT(loaded.empty());

    // A store afterwards must rewrite it wholesale at the current version --
    // never merge with (or preserve) the rejected v1 content.
    UbatchCacheEntry e;
    e.key.device_key = "Dev@1.0";
    e.key.model_name = "m";
    e.n_ubatch       = 777;
    e.reason         = "ladder";
    ASSERT(save_ubatch_cache(cache_dir, device_name, { e }));
    ASSERT(load_ubatch_cache(cache_dir, device_name, loaded));
    ASSERT(loaded.size() == 1);
    ASSERT(loaded[0].n_ubatch == 777);

    std::remove(path.c_str());
    return true;
}

// Test: a v2 ubatch cache file (the pre-llama.cpp-3aos shape, entries with
// no "kv_unified" key at all) is ALSO rejected now that CACHE_VERSION is 3
// -- this is the RED case the version bump exists for: without it, a v2
// entry would parse via ubatch_key_from_json() with kv_unified defaulting
// to false (parse_bool() on a missing key), and could then falsely MATCH a
// real kv_unified=false lookup that was never actually validated against a
// kv_unified-aware KV-sizing formula.
TEST(ubatch_cache_v2_file_rejected) {
    std::string cache_dir   = "/tmp/llama_test_ubatch_cache_v2_" + std::to_string(getpid());
    std::string device_name = "TestUbatchV2Device_" + std::to_string(getpid());

    create_dir_recursive(cache_dir);
    std::string   path = get_ubatch_cache_file(cache_dir, device_name);
    std::ofstream f(path);
    f << "{\n";
    f << "  \"version\": 2,\n";
    f << "  \"device\": \"" << device_name << "\",\n";
    f << "  \"entries\": [{\"key\":{\"device_key\":\"Dev@1.0\",\"model_name\":\"m\",\"model_size\":0,"
      << "\"model_hash\":0,\"n_ctx\":4096,\"n_batch\":2048,\"flash_attn\":false,\"n_seq_max\":1,"
      << "\"type_k\":0,\"type_v\":0,\"device_set_hash\":0},\"n_ubatch\":512,\"reason\":\"ladder\","
      << "\"created\":\"2026-09-11T12:34:56Z\"}]\n";
    f << "}\n";
    f.close();

    std::vector<UbatchCacheEntry> loaded;
    bool                          result = load_ubatch_cache(cache_dir, device_name, loaded);
    ASSERT(result == false);
    ASSERT(loaded.empty());

    std::remove(path.c_str());
    return true;
}

// Test: a v3 ubatch cache file (the pre-llama.cpp-uajm shape: entries with
// a "kv_unified" key but no "swa_full" key) is rejected now that
// CACHE_VERSION is 4 -- the RED case this bump exists for: without it a v3
// entry written by a CLI run (swa_full=false, the field absent) would parse
// with swa_full defaulting to false and could then falsely MATCH a real
// swa_full=false lookup, or be consulted for a swa_full=true raw-API context
// whose KV bytes it was never validated against.
TEST(ubatch_cache_v3_file_rejected) {
    std::string cache_dir   = "/tmp/llama_test_ubatch_cache_v3_" + std::to_string(getpid());
    std::string device_name = "TestUbatchV3Device_" + std::to_string(getpid());

    create_dir_recursive(cache_dir);
    std::string   path = get_ubatch_cache_file(cache_dir, device_name);
    std::ofstream f(path);
    f << "{\n";
    f << "  \"version\": 3,\n";
    f << "  \"device\": \"" << device_name << "\",\n";
    f << "  \"entries\": [{\"key\":{\"device_key\":\"Dev@1.0\",\"model_name\":\"m\",\"model_size\":0,"
      << "\"model_hash\":0,\"n_ctx\":4096,\"n_batch\":2048,\"flash_attn\":false,\"n_seq_max\":1,"
      << "\"type_k\":0,\"type_v\":0,\"device_set_hash\":0,\"kv_unified\":false},\"n_ubatch\":512,"
      << "\"reason\":\"ladder\",\"created\":\"2026-09-17T06:00:00Z\"}]\n";
    f << "}\n";
    f.close();

    std::vector<UbatchCacheEntry> loaded;
    bool                          result = load_ubatch_cache(cache_dir, device_name, loaded);
    ASSERT(result == false);
    ASSERT(loaded.empty());

    std::remove(path.c_str());
    rmdir(cache_dir.c_str());
    return true;
}

// Test: a v4 ubatch cache file (the pre-llama.cpp-1oa3 shape: a device_key
// naming one card, plus an index-only device_set_hash) is rejected now that
// CACHE_VERSION is 5. Without the bump, the entry a level_zero:0 run stored
// would still be read back for a collapsed level_zero:0,1 split: both runs
// file under the first card, and v4 keyed them identically.
TEST(ubatch_cache_v4_file_rejected) {
    std::string cache_dir   = "/tmp/llama_test_ubatch_cache_v4_" + std::to_string(getpid());
    std::string device_name = "TestUbatchV4Device_" + std::to_string(getpid());

    create_dir_recursive(cache_dir);
    std::string   path = get_ubatch_cache_file(cache_dir, device_name);
    std::ofstream f(path);
    f << "{\n";
    f << "  \"version\": 4,\n";
    f << "  \"device\": \"" << device_name << "\",\n";
    f << "  \"entries\": [{\"key\":{\"device_key\":\"Dev@1.0\",\"model_name\":\"m\",\"model_size\":0,"
      << "\"model_hash\":0,\"n_ctx\":4096,\"n_batch\":2048,\"flash_attn\":false,\"n_seq_max\":1,"
      << "\"type_k\":0,\"type_v\":0,\"device_set_hash\":84696351,\"kv_unified\":false,\"swa_full\":false},"
      << "\"n_ubatch\":2048,\"reason\":\"ladder exhausted\",\"created\":\"2026-09-25T00:00:00Z\"}]\n";
    f << "}\n";
    f.close();

    std::vector<UbatchCacheEntry> loaded;
    bool                          result = load_ubatch_cache(cache_dir, device_name, loaded);
    ASSERT(result == false);
    ASSERT(loaded.empty());

    std::remove(path.c_str());
    rmdir(cache_dir.c_str());
    return true;
}

// Test: an unwritable directory makes save_ubatch_cache() return false (no
// throw), and a missing file makes load_ubatch_cache() return false (no
// throw, entries left empty) -- the two halves of the "unwritable dir"
// acceptance criterion.
TEST(ubatch_cache_unwritable_dir) {
    std::string device_name = "TestUbatchUnwritable_" + std::to_string(getpid());

    // A regular FILE occupying the path a directory is expected at: mkdir()
    // (inside create_dir_recursive()) fails, and so does opening the
    // "<file>/<sanitized>-ubatch.json.tmp" temp path underneath it --
    // reliable and portable, unlike relying on permission bits (which root
    // or a container can bypass).
    std::string blocker = "/tmp/llama_test_ubatch_cache_blocker_" + std::to_string(getpid());
    std::remove(blocker.c_str());
    std::ofstream(blocker) << "not a directory\n";

    std::string cache_dir = blocker + "/nested";  // blocker is a FILE, so this can never be created

    UbatchCacheEntry e;
    e.key.device_key = "Dev@1.0";
    e.key.model_name = "m";
    e.n_ubatch       = 1;
    e.reason         = "ladder";
    ASSERT(save_ubatch_cache(cache_dir, device_name, { e }) == false);

    std::vector<UbatchCacheEntry> loaded;
    ASSERT(load_ubatch_cache(cache_dir, device_name, loaded) == false);
    ASSERT(loaded.empty());

    std::remove(blocker.c_str());
    return true;
}

// Test: load_ubatch_cache() on a device with no file at all returns false
// and leaves `entries` empty -- mirrors load_missing_file above, for the
// ubatch entry kind.
TEST(ubatch_cache_load_missing_file) {
    std::string                   cache_dir = "/tmp/llama_test_ubatch_cache_missing_" + std::to_string(getpid());
    std::vector<UbatchCacheEntry> entries;
    bool                          result = load_ubatch_cache(cache_dir, "NonExistentUbatchDevice_12345678", entries);
    ASSERT(result == false);
    ASSERT(entries.empty());
    return true;
}

// Test: a model_name/device_key/reason containing a `"` or `\` (a Windows
// path, or a model name with a quote in it) round-trips byte-for-byte
// -- without json_escape()/the parse_string() fix,
// these would round-trip to a DIFFERENT string, so a stored key would never
// match the same lookup key again.
TEST(ubatch_cache_string_escaping_roundtrip) {
    std::string cache_dir   = "/tmp/llama_test_ubatch_cache_escape_" + std::to_string(getpid());
    std::string device_name = "TestUbatchEscape_" + std::to_string(getpid());

    UbatchCacheEntry e;
    e.key.device_key = "Arc_Pro_B70@driver\\with\\backslash";
    e.key.model_name = "C:\\models\\foo \"v2\".gguf";
    e.key.n_ctx      = 4096;
    e.n_ubatch       = 1024;
    e.reason         = "cached \"result\": ok\\done";
    e.created        = "2026-09-12T00:00:00Z";

    ASSERT(save_ubatch_cache(cache_dir, device_name, { e }));

    std::vector<UbatchCacheEntry> loaded;
    ASSERT(load_ubatch_cache(cache_dir, device_name, loaded));
    ASSERT(loaded.size() == 1);
    ASSERT(loaded[0].key == e.key);
    ASSERT(loaded[0].key.device_key == e.key.device_key);
    ASSERT(loaded[0].key.model_name == e.key.model_name);
    ASSERT(loaded[0].reason == e.reason);

    std::remove(get_ubatch_cache_file(cache_dir, device_name).c_str());
    rmdir(cache_dir.c_str());
    return true;
}

// Test: a truncated/malformed JSON file loads without throwing or invoking
// UB. NOTE: the specific outcome checked here was
// verified empirically, not assumed -- a truncated entry whose opening
// brace never closes still gets pushed by load_ubatch_cache()'s scan (the
// brace-counting loop simply runs out of characters with brace_count > 0,
// and the code pushes whatever substring it scanned regardless of whether
// the count reached zero), producing ONE entry with garbage/empty fields
// rather than zero entries. The load-bearing guarantee -- and the one this
// test actually checks -- is that this never throws or invokes UB, and
// that the resulting garbage entry's fields are empty/zero, so it can
// never coincidentally equal a real lookup key.
TEST(ubatch_cache_corrupt_json) {
    std::string cache_dir    = "/tmp/llama_test_ubatch_cache_corrupt_" + std::to_string(getpid());
    std::string device_name  = "TestUbatchCorrupt_" + std::to_string(getpid());
    std::string device_name2 = device_name + "_bin";
    create_dir_recursive(cache_dir);

    {
        std::ofstream f(get_ubatch_cache_file(cache_dir, device_name));
        f << "{\"version\": " << CACHE_VERSION << ", \"entries\": [{\"device_key\":\"a";
    }
    std::vector<UbatchCacheEntry> loaded;
    bool                          result = load_ubatch_cache(cache_dir, device_name, loaded);
    ASSERT(result == true);
    ASSERT(loaded.size() == 1);
    ASSERT(loaded[0].key.device_key == "a");
    ASSERT(loaded[0].n_ubatch == 0);

    {
        std::ofstream f(get_ubatch_cache_file(cache_dir, device_name2), std::ios::binary);
        char          bytes[] = { 0x00, 0x01, static_cast<char>(0xFF), static_cast<char>(0xFE), 0x02 };
        f.write(bytes, sizeof(bytes));
    }
    std::vector<UbatchCacheEntry> loaded2;
    bool                          result2 = load_ubatch_cache(cache_dir, device_name2, loaded2);
    ASSERT(result2 == false);
    ASSERT(loaded2.empty());

    std::remove(get_ubatch_cache_file(cache_dir, device_name).c_str());
    std::remove(get_ubatch_cache_file(cache_dir, device_name2).c_str());
    rmdir(cache_dir.c_str());
    return true;
}

// Test: no "*.tmp" file remains in the cache directory after a save
// -- globs the directory (rather than checking
// one fixed name) since the temp name now carries the writer's pid. Also
// confirms the resulting file actually parses.
TEST(ubatch_cache_atomic_write) {
    std::string cache_dir   = "/tmp/llama_test_ubatch_cache_atomic_" + std::to_string(getpid());
    std::string device_name = "TestUbatchAtomic_" + std::to_string(getpid());

    UbatchCacheEntry e;
    e.key.device_key = "Dev@1.0";
    e.key.model_name = "m";
    e.n_ubatch       = 512;
    e.reason         = "ladder exhausted";
    e.created        = "2026-09-12T00:00:00Z";
    ASSERT(save_ubatch_cache(cache_dir, device_name, { e }));

    DIR * dir = opendir(cache_dir.c_str());
    ASSERT(dir != nullptr);
    bool found_tmp = false;
    for (struct dirent * ent = readdir(dir); ent != nullptr; ent = readdir(dir)) {
        std::string name = ent->d_name;
        if (name.find(".tmp") != std::string::npos) {
            found_tmp = true;
        }
    }
    closedir(dir);
    ASSERT(!found_tmp);

    std::vector<UbatchCacheEntry> loaded;
    ASSERT(load_ubatch_cache(cache_dir, device_name, loaded));
    ASSERT(loaded.size() == 1);

    std::remove(get_ubatch_cache_file(cache_dir, device_name).c_str());
    rmdir(cache_dir.c_str());
    return true;
}

// Test: cap_ubatch_cache_entries() keeps only the 64 most recently created
// entries -- store 70 distinct keys (via the
// helper directly, then a real save/load round-trip), and confirm exactly
// 64 remain and the newest 64 (by `created`) survive.
TEST(ubatch_cache_entry_cap) {
    std::string cache_dir   = "/tmp/llama_test_ubatch_cache_cap_" + std::to_string(getpid());
    std::string device_name = "TestUbatchCap_" + std::to_string(getpid());

    std::vector<UbatchCacheEntry> entries;
    for (int i = 0; i < 70; ++i) {
        UbatchCacheEntry e;
        e.key.device_key = "Dev@1.0";
        e.key.model_name = "m";
        e.key.n_ctx      = static_cast<uint32_t>(i);  // distinct key per entry
        e.n_ubatch       = 512;
        e.reason         = "ladder exhausted";
        // Zero-padded so lexicographic (string) order matches numeric order
        // -- entry i is "created" i seconds after entry 0, so higher i is
        // newer.
        char buf[32];
        std::snprintf(buf, sizeof(buf), "2026-09-12T00:%02d:%02dZ", i / 60, i % 60);
        e.created = buf;
        entries.push_back(e);
    }
    ASSERT(entries.size() == 70);

    ASSERT(save_ubatch_cache(cache_dir, device_name, entries));  // no cap here -- direct save, not through the store
    std::vector<UbatchCacheEntry> loaded;
    ASSERT(load_ubatch_cache(cache_dir, device_name, loaded));
    ASSERT(loaded.size() == 70);  // confirms save_ubatch_cache() itself is uncapped -- capping is the caller's job

    cap_ubatch_cache_entries(loaded);
    ASSERT(loaded.size() == 64);

    // The newest 64 are n_ctx == 6..69 (the oldest 6, n_ctx == 0..5, were
    // evicted).
    bool has_oldest_survivor = false, has_evicted = false;
    for (const auto & e : loaded) {
        if (e.key.n_ctx < 6) {
            has_evicted = true;
        }
        if (e.key.n_ctx == 69) {
            has_oldest_survivor = true;  // the single newest entry must always survive
        }
    }
    ASSERT(!has_evicted);
    ASSERT(has_oldest_survivor);

    return true;
}

// Test: a wildly oversized digit run (n_ctx, parsed by parse_int()) loads
// without UB and simply fails to match any real key -- parse_int()'s
// overflow check exists specifically to avoid overflowing a signed
// accumulator, which would be UB; this test's real assertion is "no crash,
// no match", not any particular numeric value.
TEST(ubatch_cache_oversized_digit_no_ub) {
    std::string cache_dir   = "/tmp/llama_test_ubatch_cache_oversized_" + std::to_string(getpid());
    std::string device_name = "TestUbatchOversized_" + std::to_string(getpid());
    create_dir_recursive(cache_dir);

    {
        std::ofstream f(get_ubatch_cache_file(cache_dir, device_name));
        f << "{\n"
             "  \"version\": "
          << CACHE_VERSION
          << ",\n"
             "  \"entries\": [\n"
             "    {\"device_key\":\"Dev@1.0\",\"model_name\":\"m\",\"model_size\":0,\"model_hash\":0,"
             "\"n_ctx\":123456789012345678901234567890,\"n_batch\":2048,\"flash_attn\":false,"
             "\"n_seq_max\":1,\"type_k\":1,\"type_v\":1,"
             "\"n_ubatch\":512,\"reason\":\"ladder exhausted\",\"created\":\"2026-09-12T00:00:00Z\"}\n"
             "  ]\n"
             "}\n";
    }

    std::vector<UbatchCacheEntry> loaded;
    bool                          result = load_ubatch_cache(cache_dir, device_name, loaded);
    ASSERT(result == true);
    ASSERT(loaded.size() == 1);  // no crash, no UB -- parsed one entry

    UbatchCacheKey lookup_key;
    lookup_key.device_key = "Dev@1.0";
    lookup_key.model_name = "m";
    lookup_key.n_ctx      = 512;  // any real n_ctx a caller would actually look up with
    lookup_key.n_batch    = 2048;
    lookup_key.n_seq_max  = 1;
    lookup_key.type_k     = 1;
    lookup_key.type_v     = 1;
    ASSERT(!(loaded[0].key == lookup_key));  // the corrupted n_ctx never matches a real lookup

    std::remove(get_ubatch_cache_file(cache_dir, device_name).c_str());
    rmdir(cache_dir.c_str());
    return true;
}

// Test: parse_u64() must accept every value up to and including UINT64_MAX,
// not just values up to a fixed digit count. A fixed 19-digit cap (the
// first overflow fix) rejected legitimate 20-digit FNV-1a model_hash values --
// found on live GPU hardware (llama.cpp-7n6n): a Mistral run's own
// stored model_hash was 12629460749384247297, a 20-digit value the 19-digit
// cap silently truncated on load, so the parsed key never matched the
// entry that had just been written and every subsequent start missed.
TEST(parse_u64_rejects_overflow) {
    // UINT64_MAX itself (20 digits) must parse back exactly.
    ASSERT(parse_u64("{\"h\":18446744073709551615}", "h") == UINT64_MAX);

    // 2^64 (one past UINT64_MAX, also 20 digits) must NOT silently wrap to 0
    // or to any other value that could pass as a real hash -- the overflow
    // check must engage before the last digit and stop accumulating there,
    // pinning the result at the safe 19-digit prefix.
    ASSERT(parse_u64("{\"h\":18446744073709551616}", "h") == 1844674407370955161ULL);

    // A 21-digit value overflows even sooner; same "pin, don't wrap" outcome.
    ASSERT(parse_u64("{\"h\":999999999999999999999}", "h") == 9999999999999999999ULL);

    return true;
}

// Test: a 20-digit model_hash (>= 10^19, i.e. the upper half of the 64-bit
// hash space) round-trips through a real save/load cycle and matches its
// own lookup key -- the file-level counterpart to
// parse_u64_rejects_overflow above, exercising save_ubatch_cache()/
// load_ubatch_cache() end to end rather than the parser alone. Covers both
// boundary values from the GPU-run defect: the exact model_hash observed
// live, and UINT64_MAX itself.
TEST(ubatch_cache_u64_hash_boundary_roundtrip) {
    std::string cache_dir   = "/tmp/llama_test_ubatch_cache_u64hash_" + std::to_string(getpid());
    std::string device_name = "TestUbatchU64Hash_" + std::to_string(getpid());

    UbatchCacheEntry e1;
    e1.key.device_key = "Arc_Pro_B70@1.0";
    e1.key.model_name = "mistral-7b-v0.1.Q4_0.gguf";
    e1.key.model_hash = 12629460749384247297ULL;  // the exact value from the live GPU run
    e1.key.n_ctx      = 4096;
    e1.n_ubatch       = 1024;
    e1.reason         = "ladder exhausted";
    e1.created        = "2026-09-12T00:00:00Z";

    UbatchCacheEntry e2 = e1;
    e2.key.model_hash   = UINT64_MAX;
    e2.n_ubatch         = 2048;

    ASSERT(save_ubatch_cache(cache_dir, device_name, { e1, e2 }));

    std::vector<UbatchCacheEntry> loaded;
    ASSERT(load_ubatch_cache(cache_dir, device_name, loaded));
    ASSERT(loaded.size() == 2);

    bool found1 = false, found2 = false;
    for (const auto & e : loaded) {
        if (e.key == e1.key) {
            ASSERT(e.key.model_hash == 12629460749384247297ULL);
            ASSERT(e.n_ubatch == 1024);
            found1 = true;
        }
        if (e.key == e2.key) {
            ASSERT(e.key.model_hash == UINT64_MAX);
            ASSERT(e.n_ubatch == 2048);
            found2 = true;
        }
    }
    ASSERT(found1);
    ASSERT(found2);

    std::remove(get_ubatch_cache_file(cache_dir, device_name).c_str());
    rmdir(cache_dir.c_str());
    return true;
}

// The participating-device-set half of the key. The identities below mirror
// this host: logical 0 = B70, logical 1 = B50 (logical 1 is the iGPU in the
// "different second card" case). A level_zero:0,1 run without
// GGML_SYCL_SPLIT_RATIO/TENSOR_SPLIT exposes only device 0 to the scheduler,
// while the placement planner still puts layers and KV on device 1.
static UbatchDeviceIdentity ubatch_test_identity(const char * name, int budget_pct) {
    UbatchDeviceIdentity id;
    id.device_name       = name;
    id.driver_version    = "1.17.39395+13";
    id.budget_pct        = budget_pct;
    id.external_headroom = 2048ull << 20;
    return id;
}

static UbatchDeviceTopology ubatch_test_topology(std::vector<int> scheduler_devices,
                                                 int              scheduler_visible_count,
                                                 int              total_gpu_count,
                                                 bool             hidden_gpus_participate) {
    UbatchDeviceTopology topo;
    topo.scheduler_devices       = std::move(scheduler_devices);
    topo.scheduler_visible_count = scheduler_visible_count;
    topo.total_gpu_count         = total_gpu_count;
    topo.hidden_gpus_participate = hidden_gpus_participate;
    return topo;
}

static const char * const k_b70  = "Intel(R) Arc(TM) Pro B70 Graphics";
static const char * const k_b50  = "Intel(R) Arc(TM) Pro B50 Graphics";
static const char * const k_igpu = "Intel(R) Graphics";

// Test: a collapsed two-card split and the first card alone are different
// device sets, at the same budget percentage. This is the dense case too:
// the hidden-GPU gate (ggml_backend_sycl_moe_multi_gpu_requested()) is
// total_gpu_count >= 2 whatever the model, so a dense Mistral
// level_zero:0,1 run has hidden_gpus_participate=true and places a layer
// block on the B50. test-sycl-auto-ubatch-source pins that gate's model
// independence.
TEST(ubatch_device_set_key_split_differs_from_first_card_alone) {
    const std::string alone =
        ubatch_device_set_key(ubatch_test_topology({ 0 }, 1, 1, false), { ubatch_test_identity(k_b70, 100) });
    const std::string split =
        ubatch_device_set_key(ubatch_test_topology({ 0 }, 1, 2, true),
                              { ubatch_test_identity(k_b70, 100), ubatch_test_identity(k_b50, 100) });
    ASSERT(!alone.empty());
    ASSERT(split != alone);
    return true;
}

// Test: the same device set under a different VRAM budget percentage (or a
// different external headroom) fits differently, so it is a different key.
TEST(ubatch_device_set_key_includes_budget) {
    const UbatchDeviceTopology alone_topo = ubatch_test_topology({ 0 }, 1, 1, false);
    const std::string          at_100     = ubatch_device_set_key(alone_topo, { ubatch_test_identity(k_b70, 100) });
    const std::string          at_22      = ubatch_device_set_key(alone_topo, { ubatch_test_identity(k_b70, 22) });
    ASSERT(at_100 != at_22);

    UbatchDeviceIdentity other_headroom = ubatch_test_identity(k_b70, 100);
    other_headroom.external_headroom += 1ull << 20;
    ASSERT(ubatch_device_set_key(alone_topo, { other_headroom }) != at_100);

    // A budget change on the SECOND card alone must move the key too.
    const UbatchDeviceTopology split_topo = ubatch_test_topology({ 0 }, 1, 2, true);
    const std::string          both_22 =
        ubatch_device_set_key(split_topo, { ubatch_test_identity(k_b70, 22), ubatch_test_identity(k_b50, 22) });
    const std::string b50_at_30 =
        ubatch_device_set_key(split_topo, { ubatch_test_identity(k_b70, 22), ubatch_test_identity(k_b50, 30) });
    ASSERT(both_22 != b50_at_30);
    return true;
}

// Test: every participating device's identity is keyed, not just its index --
// [B70, B50] and [B70, iGPU] share indices [0, 1].
TEST(ubatch_device_set_key_identifies_every_device) {
    const UbatchDeviceTopology visible_pair = ubatch_test_topology({ 0, 1 }, 2, 2, true);
    const std::string          with_b50 =
        ubatch_device_set_key(visible_pair, { ubatch_test_identity(k_b70, 100), ubatch_test_identity(k_b50, 100) });
    const std::string with_igpu =
        ubatch_device_set_key(visible_pair, { ubatch_test_identity(k_b70, 100), ubatch_test_identity(k_igpu, 100) });
    ASSERT(with_b50 != with_igpu);
    return true;
}

// Test: order is part of the set's identity, and identical inputs give an
// identical key (the property a cache hit depends on).
TEST(ubatch_device_set_key_order_and_stability) {
    const std::vector<UbatchDeviceIdentity> ids = { ubatch_test_identity(k_b70, 100),
                                                    ubatch_test_identity(k_b50, 100) };
    const std::string forward  = ubatch_device_set_key(ubatch_test_topology({ 0, 1 }, 2, 2, true), ids);
    const std::string reversed = ubatch_device_set_key(ubatch_test_topology({ 1, 0 }, 2, 2, true), ids);
    ASSERT(forward != reversed);
    ASSERT(forward == ubatch_device_set_key(ubatch_test_topology({ 0, 1 }, 2, 2, true), ids));
    return true;
}

// Test: a collapsed split (scheduler [0], device 1 hidden) and a
// SPLIT_RATIO/TENSOR_SPLIT split (scheduler [0, 1]) involve the same devices
// but not the same demand -- only the latter puts compute buffers on device
// 1 -- so they must not share a key.
TEST(ubatch_device_set_key_collapsed_split_differs_from_visible_split) {
    const std::vector<UbatchDeviceIdentity> ids       = { ubatch_test_identity(k_b70, 100),
                                                          ubatch_test_identity(k_b50, 100) };
    const UbatchDeviceTopology              collapsed = ubatch_test_topology({ 0 }, 1, 2, true);
    const UbatchDeviceTopology              visible   = ubatch_test_topology({ 0, 1 }, 2, 2, true);
    ASSERT(ubatch_participating_devices(collapsed) == ubatch_participating_devices(visible));
    const std::string collapsed_key = ubatch_device_set_key(collapsed, ids);
    const std::string visible_key   = ubatch_device_set_key(visible, ids);
    ASSERT(!collapsed_key.empty());
    ASSERT(collapsed_key != visible_key);
    ASSERT(collapsed_key.find("hidden:") != std::string::npos);
    ASSERT(visible_key.find("hidden:") == std::string::npos);
    return true;
}

// Test: the multi-GPU placement knobs divide work across the same devices
// differently, so each set value is keyed; an unset one adds nothing, which
// keeps a single-device key free of them.
TEST(ubatch_device_set_key_includes_placement_config) {
    const std::vector<UbatchDeviceIdentity> ids       = { ubatch_test_identity(k_b70, 100),
                                                          ubatch_test_identity(k_b50, 100) };
    UbatchDeviceTopology                    layer     = ubatch_test_topology({ 0 }, 1, 2, true);
    UbatchDeviceTopology                    hybrid    = layer;
    const std::string                       unset_key = ubatch_device_set_key(layer, ids);
    layer.placement_config                            = "mode=layer";
    hybrid.placement_config                           = "mode=hybrid";
    ASSERT(ubatch_device_set_key(layer, ids) != unset_key);
    ASSERT(ubatch_device_set_key(layer, ids) != ubatch_device_set_key(hybrid, ids));
    ASSERT(unset_key.find('|') == std::string::npos);
    return true;
}

// Test: the backend treats an empty knob as unset (ggml_sycl_env_is_set()),
// so an empty value must compose the same config as an absent one; keying
// "NAME=" would make the same placement miss.
TEST(ubatch_placement_config_empty_value_is_unset) {
    const char * const names[]      = { "GGML_SYCL_MULTI_GPU_MODE", "GGML_SYCL_SPLIT_RATIO", "GGML_SYCL_TENSOR_SPLIT" };
    const char * const unset[]      = { nullptr, nullptr, nullptr };
    const char * const empty[]      = { nullptr, "", nullptr };
    const char * const set_mode[]   = { "layer", nullptr, nullptr };
    const char * const mode_empty[] = { "layer", "", "" };
    ASSERT(ubatch_placement_config(names, empty, 3) == ubatch_placement_config(names, unset, 3));
    ASSERT(ubatch_placement_config(names, unset, 3).empty());
    ASSERT(ubatch_placement_config(names, mode_empty, 3) == ubatch_placement_config(names, set_mode, 3));
    ASSERT(ubatch_placement_config(names, set_mode, 3) == "GGML_SYCL_MULTI_GPU_MODE=layer");
    return true;
}

// Test: a hidden GPU that the placement planner does not use (multi-GPU
// placement disabled) is not part of the set, so that run shares the
// first-card-alone entry.
TEST(ubatch_device_set_key_idle_hidden_gpu_is_not_keyed) {
    const std::string alone =
        ubatch_device_set_key(ubatch_test_topology({ 0 }, 1, 1, false), { ubatch_test_identity(k_b70, 100) });
    const std::string idle_split =
        ubatch_device_set_key(ubatch_test_topology({ 0 }, 1, 2, false),
                              { ubatch_test_identity(k_b70, 100), ubatch_test_identity(k_b50, 100) });
    ASSERT(idle_split == alone);
    return true;
}

// Test: the participating list is the scheduler devices in order, then each
// hidden physical GPU in index order, without repeating one.
TEST(ubatch_participating_devices_order) {
    ASSERT((ubatch_participating_devices(ubatch_test_topology({ 0 }, 1, 2, true)) == std::vector<int>{ 0, 1 }));
    ASSERT((ubatch_participating_devices(ubatch_test_topology({ 0 }, 1, 3, true)) == std::vector<int>{ 0, 1, 2 }));
    ASSERT((ubatch_participating_devices(ubatch_test_topology({ 0 }, 1, 2, false)) == std::vector<int>{ 0 }));
    ASSERT((ubatch_participating_devices(ubatch_test_topology({ 1, 0 }, 2, 2, true)) == std::vector<int>{ 1, 0 }));
    return true;
}

// =============================================================================
// Main test runner
// =============================================================================
int main() {
    // llama.cpp-7n6n: several tests in this binary
    // (cache_file_roundtrip, version_check, atomic_write) resolve their
    // directory through get_cache_dir(), which falls back to the REAL
    // $HOME/.cache/llama.cpp/sycl-tuning whenever XDG_CACHE_HOME is unset.
    // Refuse outright rather than let a direct invocation reach that real
    // directory -- tests/CMakeLists.txt's registration always sets this
    // (via ENVIRONMENT) before ctest runs the binary, so this only fires
    // for someone invoking the built binary by hand without also setting
    // it, matching CLAUDE.md's "what the registration provides, direct
    // invocation does not" lesson.
    const char * xdg_cache_home = std::getenv("XDG_CACHE_HOME");
    if (xdg_cache_home == nullptr || xdg_cache_home[0] == '\0') {
        std::cerr << "SKIP: XDG_CACHE_HOME must be set to a scratch directory before running this binary "
                     "directly -- several of its tests fall back to the real $HOME/.cache/llama.cpp/sycl-tuning "
                     "otherwise. See tests/CMakeLists.txt's test-tuning-cache-io registration for the values "
                     "ctest itself uses.\n";
        return 77;
    }

    std::cout << "=== Tuning Cache I/O Tests ===\n\n";

    RUN_TEST(get_cache_dir);
    RUN_TEST(get_cache_dir_xdg);
    RUN_TEST(sanitize_device_name);
    RUN_TEST(get_cache_file);
    RUN_TEST(params_to_json);
    RUN_TEST(params_from_json);
    RUN_TEST(params_roundtrip);
    RUN_TEST(key_serialization);
    RUN_TEST(entry_serialization);
    RUN_TEST(create_dir_recursive);
    RUN_TEST(cache_for_each);
    RUN_TEST(cache_file_roundtrip);
    RUN_TEST(load_missing_file);
    RUN_TEST(version_check);
    RUN_TEST(cache_version);
    RUN_TEST(parse_int_edge_cases);
    RUN_TEST(parse_bool_edge_cases);
    RUN_TEST(parse_string_edge_cases);
    RUN_TEST(atomic_write);

    RUN_TEST(cache_version_is_5);
    RUN_TEST(ubatch_key_equality);
    RUN_TEST(ubatch_kv_mode_json_roundtrip);
    RUN_TEST(ubatch_swa_full_json_roundtrip);
    RUN_TEST(ubatch_entry_serialization);
    RUN_TEST(ubatch_cache_file_roundtrip);
    RUN_TEST(ubatch_cache_v1_file_rejected);
    RUN_TEST(ubatch_cache_v2_file_rejected);
    RUN_TEST(ubatch_cache_v3_file_rejected);
    RUN_TEST(ubatch_cache_v4_file_rejected);
    RUN_TEST(ubatch_cache_unwritable_dir);
    RUN_TEST(ubatch_cache_load_missing_file);
    RUN_TEST(ubatch_cache_string_escaping_roundtrip);
    RUN_TEST(ubatch_cache_corrupt_json);
    RUN_TEST(ubatch_cache_atomic_write);
    RUN_TEST(ubatch_cache_entry_cap);
    RUN_TEST(ubatch_cache_oversized_digit_no_ub);
    RUN_TEST(parse_u64_rejects_overflow);
    RUN_TEST(ubatch_cache_u64_hash_boundary_roundtrip);
    RUN_TEST(ubatch_device_set_key_split_differs_from_first_card_alone);
    RUN_TEST(ubatch_device_set_key_includes_budget);
    RUN_TEST(ubatch_device_set_key_identifies_every_device);
    RUN_TEST(ubatch_device_set_key_order_and_stability);
    RUN_TEST(ubatch_device_set_key_collapsed_split_differs_from_visible_split);
    RUN_TEST(ubatch_device_set_key_includes_placement_config);
    RUN_TEST(ubatch_placement_config_empty_value_is_unset);
    RUN_TEST(ubatch_device_set_key_idle_hidden_gpu_is_not_keyed);
    RUN_TEST(ubatch_participating_devices_order);

    std::cout << "\n=== Summary ===\n";
    std::cout << "Passed: " << g_passed << ", Failed: " << g_failed << "\n";

    return g_failed > 0 ? 1 : 0;
}
