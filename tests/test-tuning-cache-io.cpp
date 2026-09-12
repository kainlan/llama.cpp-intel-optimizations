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

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

#include "../ggml/src/ggml-sycl/tuning-cache-io.hpp"

using namespace ggml_sycl_tuning;

// Test counter
static int g_passed = 0;
static int g_failed = 0;

#define TEST(name) \
    static bool test_##name(); \
    static bool test_##name##_registered = [] { \
        return true; \
    }(); \
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

// Test: CACHE_VERSION was bumped to 2 for this task (v1 stays reserved for
// the pre-existing matmul dispatch-tuning format alone).
TEST(cache_version_is_2) {
    ASSERT(CACHE_VERSION == 2);
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
// like load_cache()'s own version check (tuning-cache-io.hpp:~380-384).
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

    RUN_TEST(cache_version_is_2);
    RUN_TEST(ubatch_key_equality);
    RUN_TEST(ubatch_entry_serialization);
    RUN_TEST(ubatch_cache_file_roundtrip);
    RUN_TEST(ubatch_cache_v1_file_rejected);
    RUN_TEST(ubatch_cache_unwritable_dir);
    RUN_TEST(ubatch_cache_load_missing_file);

    std::cout << "\n=== Summary ===\n";
    std::cout << "Passed: " << g_passed << ", Failed: " << g_failed << "\n";

    return g_failed > 0 ? 1 : 0;
}
