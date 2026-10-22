//
// Test: the expert loan transaction (llama.cpp-mwmr, plan Task 11b of llama.cpp-36j0).
//
// A prompt phase lends the device's least popular resident expert triplets to the prefill ring: each is copied to
// pinned host storage, the published plan flips it to the host, and its device entry is retired so the storage goes
// back. No MoE model fixture exists host-side, so the cases build synthetic 1 MiB experts through
// begin_planned_materialization + direct_stage_expert (the bulk case through the tensor-bulk stager) inside the
// backend, where the file-static transaction lives, and report here:
//
//   lends_two_triplets      2 layers x 4 experts, expert 3 on the host: three requests leave the largest pending,
//                           the graph-boundary runner lends two triplets and clears it, the plan shows them on the
//                           host, resolve_expert finds host-pinned copies equal to the originals, the device entries
//                           are gone within 2 s and the device gets their storage back.
//   probe_changes_nothing   c-tdra first: inside a graph the transaction refuses "graph active" and a request waits.
//                           Then the same plan probed: the selection is reported and nothing is copied or published.
//   refuses_shared_storage  c-sepd: a bulk-staged pair with expert 1 flipped to the host is refused with "shared
//                           storage"; nothing is copied or published.
//
// The ctest registration sets GGML_SYCL_STRICT_LEASES=1, so an ownerless lease anywhere in the loan aborts the run.
// Each case puts back the plan publication it found, so the next one stages under the same coherence. A setup
// failure names the cache's reason. The shared-storage case reserves a WEIGHT arena; it runs last so the others
// stage raw device allocations.
//
// MIT license
// SPDX-License-Identifier: MIT
//

#include "../common.hpp"
#include "../ggml-sycl-test.hpp"
#include "sycl-test-skip.hpp"

#include <cstdio>
#include <optional>
#include <string>
#include <sycl/sycl.hpp>

static bool run_case(const char * name, std::string (*body)()) {
    fprintf(stderr, "[TEST] %s ... ", name);
    const std::string failure = body();
    if (!failure.empty()) {
        fprintf(stderr, "FAILED: %s\n", failure.c_str());
        return false;
    }
    fprintf(stderr, "PASSED\n");
    return true;
}

int main() {
    std::optional<sycl::device> device = sycl_test_require_gpu("expert loan transaction");
    if (!device) {
        return SYCL_TEST_SKIP;
    }
    if (ggml_backend_sycl_get_device_count() == 0 || ggml_sycl_info().total_gpu_count <= 0) {
        sycl_test_print_skip("expert loan transaction backend device");
        return SYCL_TEST_SKIP;
    }
    fprintf(stderr, "Device: %s\n", device->get_info<sycl::info::device::name>().c_str());

    bool ok = true;
    ok &= run_case("lends_two_triplets", ggml_sycl::test_expert_loan_lends_two_triplets);
    ok &= run_case("probe_changes_nothing", ggml_sycl::test_expert_loan_probe_changes_nothing);
    ok &= run_case("refuses_shared_storage", ggml_sycl::test_expert_loan_refuses_shared_storage);

    fprintf(stderr, "%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? 0 : 1;
}
