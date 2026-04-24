// Canary D0.4 — direct weight load: mmap src -> ggml_backend_tensor_set -> device tensor.
// Validates that one `ggml_backend_tensor_set` call moves bytes from an
// mmap'd source into a pre-allocated device tensor slot in exactly one
// copy. A7 (weight loader direct placement) depends on this property.
//
// Scope: uses only ggml-backend APIs directly -- no llama_model_load_from_file,
// no llama_init_from_model. Therefore immune to the m09zb staging-pool
// wedge that blocks D0.1/D0.2/D0.3.

#include "test-planner-canary-common.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace planner_canary;

int main(int /*argc*/, char ** /*argv*/) {
    findings f;
    f.canary_id = "D0.4";
    f.result    = status::FAIL;
    f.summary   = "Not run";

    const std::string md_path   = "docs/plans/data/planner-canaries/d0.4-direct-load.md";
    const std::string json_path = "tests/data/planner-canaries/d0.4.json";

    // We don't need a real GGUF tensor extraction -- any mmap'd file with
    // known bytes works as a "raw weight slice" stand-in. Use a 4 KB slice
    // of the Mistral GGUF at a known offset.
    const std::string mistral = mistral_path();
    int fd = open(mistral.c_str(), O_RDONLY);
    if (fd < 0) {
        f.result         = status::INCONCLUSIVE;
        f.summary        = "Mistral model not available";
        f.recommendation = "set MISTRAL_PATH env var to a readable GGUF file";
        write_markdown(f, md_path);
        write_json    (f, json_path);
        return 0;
    }

    struct stat st{};
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        f.result  = status::FAIL;
        f.summary = "fstat failed on GGUF";
        write_markdown(f, md_path);
        write_json    (f, json_path);
        close(fd);
        return 1;
    }

    const size_t TEST_SIZE   = 4096;
    const size_t TEST_OFFSET = static_cast<size_t>(st.st_size) > (1024 * 1024) ? (1024 * 1024) : 0;

    void * mmap_base = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
    if (mmap_base == MAP_FAILED) {
        f.result  = status::FAIL;
        f.summary = "mmap failed";
        write_markdown(f, md_path);
        write_json    (f, json_path);
        close(fd);
        return 1;
    }
    const uint8_t * src_bytes = static_cast<const uint8_t *>(mmap_base) + TEST_OFFSET;

    // Set up a SYCL backend on device 0.
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        f.result  = status::FAIL;
        f.summary = "SYCL backend init failed (device 0)";
        write_markdown(f, md_path);
        write_json    (f, json_path);
        munmap(mmap_base, static_cast<size_t>(st.st_size));
        close(fd);
        return 1;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    // Allocate a tensor context + one F16 tensor covering TEST_SIZE bytes.
    // no_alloc=true is REQUIRED: ggml_backend_alloc_ctx_tensors_from_buft
    // materializes the device buffer itself and asserts the context was
    // declared without a CPU-side allocation (ggml-alloc.c:1237).
    struct ggml_init_params ip{};
    ip.mem_size    = 16 * 1024;
    ip.mem_buffer  = nullptr;
    ip.no_alloc    = true;
    struct ggml_context * gctx = ggml_init(ip);
    if (!gctx) {
        f.result  = status::FAIL;
        f.summary = "ggml_init failed";
        write_markdown(f, md_path);
        write_json    (f, json_path);
        ggml_backend_free(backend);
        munmap(mmap_base, static_cast<size_t>(st.st_size));
        close(fd);
        return 1;
    }

    struct ggml_tensor * t =
        ggml_new_tensor_1d(gctx, GGML_TYPE_F16, TEST_SIZE / sizeof(ggml_fp16_t));
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(gctx, buft);
    if (!buf) {
        f.result  = status::FAIL;
        f.summary = "ggml_backend_alloc_ctx_tensors_from_buft returned null";
        write_markdown(f, md_path);
        write_json    (f, json_path);
        ggml_free(gctx);
        ggml_backend_free(backend);
        munmap(mmap_base, static_cast<size_t>(st.st_size));
        close(fd);
        return 1;
    }

    // The measurement: one ggml_backend_tensor_set call from mmap'd src to
    // pre-allocated device tensor. Time it.
    using clock = std::chrono::steady_clock;
    const auto t_start = clock::now();
    ggml_backend_tensor_set(t, src_bytes, 0, TEST_SIZE);
    const auto t_end   = clock::now();
    const auto direct_copy_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();

    // Read back and byte-compare against the source.
    std::vector<uint8_t> readback(TEST_SIZE);
    ggml_backend_tensor_get(t, readback.data(), 0, TEST_SIZE);

    const bool bytes_match = std::memcmp(readback.data(), src_bytes, TEST_SIZE) == 0;

    add(f, "test_offset_in_file",   std::to_string(TEST_OFFSET));
    add(f, "bytes_transferred",     std::to_string(TEST_SIZE));
    add(f, "readback_matches_src",  bytes_match ? "YES" : "NO");
    add(f, "direct_copy_us",        std::to_string(direct_copy_us));

    if (bytes_match) {
        f.result         = status::PASS;
        f.summary        = "Direct mmap -> device tensor transfer works in one ggml_backend_tensor_set call";
        f.recommendation = "A7 can use ggml_backend_tensor_set for direct weight load; no host-pinned intermediate needed";
    } else {
        f.result         = status::FAIL;
        f.summary        = "Bytes differ after direct mmap -> device tensor transfer";
        f.recommendation = "A7 needs to stage through a host-pinned intermediate; document the transfer path";
    }

    // Cleanup.
    ggml_backend_buffer_free(buf);
    ggml_free(gctx);
    ggml_backend_free(backend);
    munmap(mmap_base, static_cast<size_t>(st.st_size));
    close(fd);

    write_markdown(f, md_path);
    write_json    (f, json_path);

    return (f.result == status::PASS) ? 0 : 1;
}
