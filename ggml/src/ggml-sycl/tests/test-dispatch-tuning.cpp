//
// Test: Dispatch Tuning Mapping
//
// Verifies that dispatch tuning maps benchmark-summary "winner" strings to
// their kernel enums (unified_matmul, mmvq_/mmq_ coalesced+soa sub-branches,
// onednn_), and that a single loaded cache correctly discriminates between
// several distinct (type,m,n,k) keys rather than merely answering one.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "dispatch-tuning.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define TEST_BEGIN(name)                         \
    do {                                         \
        g_tests_run++;                           \
        fprintf(stderr, "[TEST] %s ... ", name); \
    } while (0)

#define TEST_PASS()                  \
    do {                             \
        g_tests_passed++;            \
        fprintf(stderr, "PASSED\n"); \
    } while (0)

#define TEST_FAIL(msg)                        \
    do {                                      \
        fprintf(stderr, "FAILED: %s\n", msg); \
        return false;                         \
    } while (0)

#define TEST_ASSERT(cond, msg) \
    do {                       \
        if (!(cond)) {         \
            TEST_FAIL(msg);    \
        }                      \
    } while (0)

static bool write_temp_file(std::string & path, const std::string & content) {
    char tmpl[] = "/tmp/ggml_dispatch_tuningXXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) {
        return false;
    }
    FILE * file = fdopen(fd, "w");
    if (!file) {
        close(fd);
        return false;
    }
    const size_t written = fwrite(content.data(), 1, content.size(), file);
    fclose(file);
    path = tmpl;
    return written == content.size();
}

static bool test_unified_matmul_mapping() {
    TEST_BEGIN("unified_matmul winner maps to UNIFIED_MATMUL");
    const std::string json =
        "{\n"
        "  \"results\": [\n"
        "    {\n"
        "      \"quant\": \"Q4_0\",\n"
        "      \"dim_m\": 1,\n"
        "      \"dim_n\": 256,\n"
        "      \"dim_k\": 256,\n"
        "      \"tensor_instances\": 1,\n"
        "      \"winner\": \"unified_matmul\"\n"
        "    }\n"
        "  ]\n"
        "}\n";

    std::string path;
    TEST_ASSERT(write_temp_file(path, json), "failed to create temp tuning file");

    ggml_sycl::dispatch_tuning::DispatchTuningCache cache;
    std::string error;
    const bool loaded = ggml_sycl::dispatch_tuning::load_dispatch_tuning_from_file(path, cache, &error);
    std::remove(path.c_str());

    TEST_ASSERT(loaded, "failed to load tuning entries");
    auto key = ggml_sycl::dispatch_tuning::make_dispatch_tuning_key(GGML_TYPE_Q4_0, 1, 256, 256);
    auto entry = cache.lookup(key);
    TEST_ASSERT(entry.has_value(), "missing tuning entry");
    TEST_ASSERT(entry->kernel == ggml_sycl_mul_mat_kernel::UNIFIED_MATMUL,
                "winner did not map to UNIFIED_MATMUL");

    TEST_PASS();
    return true;
}

// Shared body for the single-entry winner-mapping cases below: the winner
// string is the only variable, so a difference in outcome can only come from
// the winner parser. (The quant/dims in the fixture are irrelevant to
// map_winner_to_kernel().) Named for what it now covers -- onednn_ and mmq_
// winners, not just mmvq_ -- since the four cases below share it.
static bool winner_maps_to(const std::string & winner, ggml_sycl_mul_mat_kernel expected, const char * mismatch_msg) {
    const std::string json =
        "{\n"
        "  \"results\": [\n"
        "    {\n"
        "      \"quant\": \"Q4_0\",\n"
        "      \"dim_m\": 16,\n"
        "      \"dim_n\": 4096,\n"
        "      \"dim_k\": 4096,\n"
        "      \"tensor_instances\": 1,\n"
        "      \"winner\": \"" +
        winner +
        "\"\n"
        "    }\n"
        "  ]\n"
        "}\n";

    std::string path;
    TEST_ASSERT(write_temp_file(path, json), "failed to create temp tuning file");

    ggml_sycl::dispatch_tuning::DispatchTuningCache cache;
    std::string error;
    const bool loaded = ggml_sycl::dispatch_tuning::load_dispatch_tuning_from_file(path, cache, &error);
    std::remove(path.c_str());

    TEST_ASSERT(loaded, "failed to load tuning entries");
    auto key = ggml_sycl::dispatch_tuning::make_dispatch_tuning_key(GGML_TYPE_Q4_0, 16, 4096, 4096);
    auto entry = cache.lookup(key);
    TEST_ASSERT(entry.has_value(), "missing tuning entry");
    TEST_ASSERT(entry->kernel == expected, mismatch_msg);

    return true;
}

static bool test_mmvq_coalesced_mapping() {
    TEST_BEGIN("mmvq_coalesced winner maps to MMVQ_COALESCED");
    if (!winner_maps_to("mmvq_coalesced", ggml_sycl_mul_mat_kernel::MMVQ_COALESCED,
                        "winner did not map to MMVQ_COALESCED")) {
        return false;
    }
    TEST_PASS();
    return true;
}

// The SOA arm shares the mmvq_ prefix branch but not the coalesced sub-branch,
// so it stays green when only the coalesced return is disturbed.
static bool test_mmvq_soa_mapping() {
    TEST_BEGIN("mmvq_soa winner maps to MMVQ_SOA");
    if (!winner_maps_to("mmvq_soa", ggml_sycl_mul_mat_kernel::MMVQ_SOA, "winner did not map to MMVQ_SOA")) {
        return false;
    }
    TEST_PASS();
    return true;
}

// The two assertions the unregistered repo-root twin (tests/test-dispatch-tuning.cpp,
// deleted under llama.cpp-la7d) carried that this file did not: the onednn_ prefix
// branch and the mmq_ soa sub-branch. Ported so deleting the twin loses no coverage.
static bool test_onednn_mapping() {
    TEST_BEGIN("onednn_woq_gemm winner maps to ONEDNN_AOS");
    if (!winner_maps_to("onednn_woq_gemm", ggml_sycl_mul_mat_kernel::ONEDNN_AOS, "winner did not map to ONEDNN_AOS")) {
        return false;
    }
    TEST_PASS();
    return true;
}

static bool test_mmq_soa_mapping() {
    TEST_BEGIN("mmq_soa winner maps to MMQ_SOA");
    if (!winner_maps_to("mmq_soa", ggml_sycl_mul_mat_kernel::MMQ_SOA, "winner did not map to MMQ_SOA")) {
        return false;
    }
    TEST_PASS();
    return true;
}

// The cases above each build a fresh single-entry cache, so none of them can
// show that ONE loaded cache correctly discriminates between several distinct
// (type,m,n,k) keys rather than just answering whichever one it was asked.
// This is the property the deleted repo-root orphan (tests/test-dispatch-tuning.cpp,
// llama.cpp-la7d) covered that the single-entry winner_maps_to() helper above
// cannot reproduce: a >=3-entry tuning file whose keys each resolve to their
// own kernel out of the same cache. Ported here rather than left uncovered.
static bool test_multi_entry_cache_discrimination() {
    TEST_BEGIN("multi-entry cache discriminates by (type,m,n,k) key");
    const std::string json =
        "{\n"
        "  \"results\": [\n"
        "    {\n"
        "      \"quant\": \"Q4_0\",\n"
        "      \"dim_m\": 1,\n"
        "      \"dim_n\": 4096,\n"
        "      \"dim_k\": 4096,\n"
        "      \"tensor_instances\": 3,\n"
        "      \"winner\": \"onednn_woq_gemm\"\n"
        "    },\n"
        "    {\n"
        "      \"quant\": \"Q4_0\",\n"
        "      \"dim_m\": 16,\n"
        "      \"dim_n\": 4096,\n"
        "      \"dim_k\": 4096,\n"
        "      \"tensor_instances\": 2,\n"
        "      \"winner\": \"mmvq_coalesced\"\n"
        "    },\n"
        "    {\n"
        "      \"quant\": \"Q6_K\",\n"
        "      \"dim_m\": 128,\n"
        "      \"dim_n\": 4096,\n"
        "      \"dim_k\": 4096,\n"
        "      \"tensor_instances\": 1,\n"
        "      \"winner\": \"mmq_soa\"\n"
        "    }\n"
        "  ]\n"
        "}\n";

    std::string path;
    TEST_ASSERT(write_temp_file(path, json), "failed to create temp tuning file");

    ggml_sycl::dispatch_tuning::DispatchTuningCache cache;
    std::string error;
    const bool loaded = ggml_sycl::dispatch_tuning::load_dispatch_tuning_from_file(path, cache, &error);
    std::remove(path.c_str());
    TEST_ASSERT(loaded, "failed to load multi-entry tuning file");

    auto key_onednn = ggml_sycl::dispatch_tuning::make_dispatch_tuning_key(GGML_TYPE_Q4_0, 1, 4096, 4096);
    auto entry_onednn = cache.lookup(key_onednn);
    TEST_ASSERT(entry_onednn.has_value(), "missing onednn entry in multi-entry cache");
    TEST_ASSERT(entry_onednn->kernel == ggml_sycl_mul_mat_kernel::ONEDNN_AOS,
                "onednn key resolved to the wrong kernel from a shared multi-entry cache");

    auto key_mmvq = ggml_sycl::dispatch_tuning::make_dispatch_tuning_key(GGML_TYPE_Q4_0, 16, 4096, 4096);
    auto entry_mmvq = cache.lookup(key_mmvq);
    TEST_ASSERT(entry_mmvq.has_value(), "missing mmvq entry in multi-entry cache");
    TEST_ASSERT(entry_mmvq->kernel == ggml_sycl_mul_mat_kernel::MMVQ_COALESCED,
                "mmvq key resolved to the wrong kernel from a shared multi-entry cache");

    auto key_mmq = ggml_sycl::dispatch_tuning::make_dispatch_tuning_key(GGML_TYPE_Q6_K, 128, 4096, 4096);
    auto entry_mmq = cache.lookup(key_mmq);
    TEST_ASSERT(entry_mmq.has_value(), "missing mmq entry in multi-entry cache");
    TEST_ASSERT(entry_mmq->kernel == ggml_sycl_mul_mat_kernel::MMQ_SOA,
                "mmq key resolved to the wrong kernel from a shared multi-entry cache");

    // A key never present in the file must not spuriously resolve to any of
    // the three entries above -- the cache must discriminate, not just answer.
    auto key_absent = ggml_sycl::dispatch_tuning::make_dispatch_tuning_key(GGML_TYPE_Q8_0, 8, 4096, 4096);
    auto entry_absent = cache.lookup(key_absent);
    TEST_ASSERT(!entry_absent.has_value(), "absent key spuriously resolved to a cached entry");

    TEST_PASS();
    return true;
}

int main() {
    bool ok = true;
    ok = test_unified_matmul_mapping() && ok;
    ok = test_mmvq_coalesced_mapping() && ok;
    ok = test_mmvq_soa_mapping() && ok;
    ok = test_onednn_mapping() && ok;
    ok = test_mmq_soa_mapping() && ok;
    ok = test_multi_entry_cache_discrimination() && ok;

    if (!ok || g_tests_passed != g_tests_run) {
        fprintf(stderr, "\n[TEST SUMMARY] %d/%d tests passed\n", g_tests_passed, g_tests_run);
        return 1;
    }
    fprintf(stderr, "\n[TEST SUMMARY] %d/%d tests passed\n", g_tests_passed, g_tests_run);
    return 0;
}
