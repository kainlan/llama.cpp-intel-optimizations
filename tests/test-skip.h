#pragma once

// Shared skip policy for tests that require a model file.
//
// Exit code 77 is ctest's conventional "skipped" status: llama_build_and_test()
// in tests/CMakeLists.txt sets SKIP_RETURN_CODE 77 on every test it registers,
// so a skip is reported as "Skipped" rather than "Passed", and a bare shell run
// gets a non-zero status instead of a silent green. Skipping is still allowed --
// a model-less or CPU-only runner legitimately skips -- it just may not claim to
// have passed.
//
// This header is fork-local on purpose. tests/get-model.cpp is upstream code, so
// a rebase can revert the two lines there that call into this policy; it cannot
// take the policy itself, and this comment, with it. See llama.cpp-nwip: exiting
// EXIT_SUCCESS made test-autorelease, test-model-load-cancel and
// test-backend-sampler pass vacuously in 0.18 s on every local ctest run.

#include <cstdio>
#include <cstdlib>

#define LLAMA_TEST_EXIT_SKIP 77

[[noreturn]] static inline void test_skip_no_model() {
    fprintf(stderr,
            "\033[33mWARNING: No model file provided. Skipping this test. "
            "Set LLAMACPP_TEST_MODELFILE=<gguf_model_path> to silence this warning and run this test.\n\033[0m");
    exit(LLAMA_TEST_EXIT_SKIP);
}
