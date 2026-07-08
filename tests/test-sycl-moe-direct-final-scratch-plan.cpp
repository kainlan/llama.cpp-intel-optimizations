#include "ggml-sycl/ggml-sycl-test.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

static std::string join_path(const std::string & root, const char * rel) {
    if (root.empty() || root == ".") {
        return rel;
    }
    return root.back() == '/' ? root + rel : root + "/" + rel;
}

static std::vector<std::string> candidate_roots() {
    std::vector<std::string> roots;
    if (const char * env = std::getenv("LLAMA_CPP_REPO_ROOT")) {
        roots.emplace_back(env);
    }
    const std::string source_file = __FILE__;
    const std::string suffix      = "/tests/test-sycl-moe-direct-final-scratch-plan.cpp";
    const size_t      pos         = source_file.rfind(suffix);
    if (pos != std::string::npos) {
        roots.emplace_back(source_file.substr(0, pos));
    }
    roots.emplace_back(".");
    roots.emplace_back("..");
    roots.emplace_back("../..");
    roots.emplace_back("../../..");
    return roots;
}

static std::string read_required_file(const char * rel) {
    for (const std::string & root : candidate_roots()) {
        std::ifstream in(join_path(root, rel), std::ios::binary);
        if (!in.good()) {
            continue;
        }
        std::ostringstream ss;
        ss << in.rdbuf();
        return ss.str();
    }
    std::fprintf(stderr, "FAIL: could not read required source file: %s\n", rel);
    std::exit(1);
}

static int test_scratch_plan_accepts_small_tg() {
    ggml_sycl::test_moe_direct_final_scratch_plan_input in{};
    in.n_tokens         = 1;
    in.n_ids            = 4;
    in.nrows_per_expert = 2880;
    in.element_size     = sizeof(float);
    auto out = ggml_sycl::test_moe_direct_final_scratch_plan(in);
    CHECK(out.accepted, "valid scratch plan must accept");
    CHECK(out.bytes == 1ull * 4ull * 2880ull * sizeof(float), "scratch bytes mismatch");
    CHECK(out.requires_reduce, "scratch plan must require a reduce kernel");
    return 0;
}

static int test_direct_final_capacity_math_is_checked_before_multiply() {
    const std::string mmvq = read_required_file("ggml/src/ggml-sycl/mmvq.cpp");
    const size_t start = mmvq.find("bool mmvq_moe_batched_dispatch_down_sum_from_cached_q8_mxfp4(");
    CHECK(start != std::string::npos, "direct-final cached-Q8 dispatcher must exist");
    const size_t end = mmvq.find("\nbool ggml_sycl_mul_mat_id_vec_q(", start);
    CHECK(end != std::string::npos && end > start, "direct-final dispatcher source range must be bounded");
    const std::string body = mmvq.substr(start, end - start);

    const size_t batch_check = body.find("n_ids > max_i64 / n_tokens");
    const size_t batch_mul   = body.find("const int64_t total_batches = n_ids * n_tokens");
    CHECK(batch_check != std::string::npos, "direct-final total_batches overflow check must be present");
    CHECK(batch_mul != std::string::npos, "direct-final total_batches multiply must be present");
    CHECK(batch_check < batch_mul, "direct-final total_batches multiply must happen after overflow check");

    const size_t bytes_check = body.find("std::numeric_limits<size_t>::max() / static_cast<uint64_t>(q8_row_size)");
    const size_t bytes_mul   = body.find("const size_t required_q8_bytes = static_cast<size_t>(total_batches) * static_cast<size_t>(q8_row_size)");
    CHECK(bytes_check != std::string::npos, "direct-final required_q8_bytes overflow check must be present");
    CHECK(bytes_mul != std::string::npos, "direct-final required_q8_bytes multiply must be present");
    CHECK(bytes_check < bytes_mul, "direct-final required_q8_bytes multiply must happen after overflow check");
    return 0;
}

static int test_scratch_plan_rejects_overflow_and_bad_shapes() {
    ggml_sycl::test_moe_direct_final_scratch_plan_input in{};
    in.n_tokens         = 1;
    in.n_ids            = std::numeric_limits<int64_t>::max();
    in.nrows_per_expert = 2880;
    in.element_size     = sizeof(float);
    auto overflow = ggml_sycl::test_moe_direct_final_scratch_plan(in);
    CHECK(!overflow.accepted, "overflow must reject");
    CHECK(overflow.reason && std::string(overflow.reason) == "overflow", "overflow reason must be overflow");

    in.n_ids = 0;
    auto shape = ggml_sycl::test_moe_direct_final_scratch_plan(in);
    CHECK(!shape.accepted, "zero ids must reject");
    CHECK(shape.reason && std::string(shape.reason) == "shape", "shape reason must be shape");
    return 0;
}

int main() {
    if (test_scratch_plan_accepts_small_tg() != 0) {
        return 1;
    }
    if (test_scratch_plan_rejects_overflow_and_bad_shapes() != 0) {
        return 1;
    }
    if (test_direct_final_capacity_math_is_checked_before_multiply() != 0) {
        return 1;
    }
    std::puts("PASS: direct-final scratch plan");
    return 0;
}
