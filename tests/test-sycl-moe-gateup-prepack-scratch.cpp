#include "ggml-sycl/moe-layer-plan.hpp"

#include <cstdio>

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

namespace {

constexpr int TEST_DEVICE       = 0;
constexpr int TEST_RUNTIME_ZONE = 3;
constexpr int TEST_SCRATCH_ZONE = 4;

static ggml_sycl::mem_handle make_arena_handle(int zone, size_t offset, size_t size, uint64_t generation) {
    return ggml_sycl::mem_handle::from_arena_zone(zone, offset, size, TEST_DEVICE, generation);
}

static ggml_sycl::moe_gateup_prepack_scratch_descriptor configured_descriptor(
    uint64_t metadata_signature = 0xabcddcbaULL) {
    ggml_sycl::moe_gateup_prepack_scratch_descriptor desc;
    desc.configure(/* layer */ 11, TEST_DEVICE,
                   /* selected_entries */ 4,
                   /* selected_batches */ 1,
                   /* scratch_bytes */ 4096, metadata_signature);
    return desc;
}

static int add_required_artifacts(ggml_sycl::moe_gateup_prepack_scratch_descriptor & desc) {
    using role = ggml_sycl::moe_gateup_prepack_artifact_role;

    CHECK(desc.add_artifact(role::SOURCE_ACTIVATION, make_arena_handle(TEST_RUNTIME_ZONE, 0x1000, 1024, 1), 1024),
          "source activation handle must be accepted");
    CHECK(desc.add_artifact(role::GATE_WEIGHT, make_arena_handle(TEST_RUNTIME_ZONE, 0x2000, 2048, 1), 2048),
          "gate weight handle must be accepted");
    CHECK(desc.add_artifact(role::UP_WEIGHT, make_arena_handle(TEST_RUNTIME_ZONE, 0x3000, 2048, 1), 2048),
          "up weight handle must be accepted");
    CHECK(desc.add_artifact(role::SCRATCH, make_arena_handle(TEST_SCRATCH_ZONE, 0x4000, 4096, 1), 4096),
          "scratch handle must be accepted");
    CHECK(desc.add_artifact(role::ROUTE_METADATA, make_arena_handle(TEST_RUNTIME_ZONE, 0x5000, 512, 1), 512),
          "route metadata handle must be accepted");
    return 0;
}

static int test_empty_invalid() {
    ggml_sycl::moe_gateup_prepack_scratch_descriptor desc;
    CHECK(desc.empty(), "new descriptor must start empty");
    CHECK(!desc.valid(), "new descriptor must start invalid");
    CHECK(desc.retained_handle_count() == 0, "new descriptor must retain no handles");
    CHECK(desc.dependency_count() == 0, "new descriptor must retain no dependencies");
    CHECK(!desc.uses_raw_pointer_identity_for_test(), "empty descriptor cannot use raw pointer identity");
    return 0;
}

static int test_handle_retention_counts() {
    using role = ggml_sycl::moe_gateup_prepack_artifact_role;

    auto desc = configured_descriptor();
    CHECK(!desc.valid(), "configured descriptor without artifacts remains invalid");

    CHECK(desc.add_artifact(role::SOURCE_ACTIVATION, make_arena_handle(TEST_RUNTIME_ZONE, 0x1000, 1024, 1), 1024),
          "source handle must be retained");
    CHECK(desc.retained_handle_count() == 1, "first handle must increase retained count");
    CHECK(desc.artifact_count(role::SOURCE_ACTIVATION) == 1, "source artifact count must be tracked");

    CHECK(desc.add_artifact(role::GATE_WEIGHT, make_arena_handle(TEST_RUNTIME_ZONE, 0x2000, 2048, 1), 2048),
          "gate handle must be retained");
    CHECK(desc.retained_handle_count() == 2, "second handle must increase retained count");

    CHECK(desc.add_artifact(role::UP_WEIGHT, make_arena_handle(TEST_RUNTIME_ZONE, 0x3000, 2048, 1), 2048),
          "up handle must be retained");
    CHECK(desc.retained_handle_count() == 3, "third handle must increase retained count");

    CHECK(desc.add_artifact(role::SCRATCH, make_arena_handle(TEST_SCRATCH_ZONE, 0x4000, 4096, 1), 4096),
          "scratch handle must be retained");
    CHECK(desc.retained_handle_count() == 4, "fourth handle must increase retained count");
    CHECK(desc.artifact_count(role::SCRATCH) == 1, "scratch artifact count must be tracked");

    CHECK(desc.add_artifact(role::ROUTE_METADATA, make_arena_handle(TEST_RUNTIME_ZONE, 0x5000, 512, 1), 512),
          "metadata handle must be retained");
    CHECK(desc.retained_handle_count() == 5, "metadata handle must increase retained count");
    CHECK(desc.valid(), "descriptor with route metadata and required artifacts must be valid");
    CHECK(!desc.uses_raw_pointer_identity_for_test(), "retained artifacts must use stable handle identities");
    return 0;
}

static int test_dependency_retention_counts() {
    auto desc = configured_descriptor();
    CHECK(desc.add_dependency(sycl::event{}), "first dependency must be accepted");
    CHECK(desc.dependency_count() == 1, "first dependency must increase dependency count");
    CHECK(desc.add_dependency(sycl::event{}), "second dependency must be accepted");
    CHECK(desc.dependency_count() == 2, "second dependency must increase dependency count");
    return 0;
}

static int test_reset_releases_retained_state() {
    auto desc = configured_descriptor();
    CHECK(add_required_artifacts(desc) == 0, "required handles must be retained before reset");
    CHECK(desc.add_dependency(sycl::event{}), "dependency must be retained before reset");
    CHECK(desc.valid(), "descriptor must be valid before reset");
    CHECK(desc.retained_handle_count() == 5, "descriptor must have retained handles before reset");
    CHECK(desc.dependency_count() == 1, "descriptor must have retained dependency before reset");

    desc.reset();
    CHECK(desc.empty(), "reset descriptor must be empty");
    CHECK(!desc.valid(), "reset descriptor must be invalid");
    CHECK(desc.retained_handle_count() == 0, "reset must clear retained handles");
    CHECK(desc.dependency_count() == 0, "reset must clear retained dependencies");
    return 0;
}

static int test_identity_is_handle_and_metadata_based() {
    auto a = configured_descriptor(0x11112222ULL);
    auto b = configured_descriptor(0x11112222ULL);
    auto c = configured_descriptor(0x33334444ULL);
    CHECK(add_required_artifacts(a) == 0, "descriptor A must retain required artifacts");
    CHECK(add_required_artifacts(b) == 0, "descriptor B must retain required artifacts");
    CHECK(add_required_artifacts(c) == 0, "descriptor C must retain required artifacts");

    CHECK(a.identity_hash() != 0, "identity hash must be populated from metadata and handles");
    CHECK(a.same_identity_as(b), "matching metadata and stable handle identities must compare equal");
    CHECK(a.identity_hash() == b.identity_hash(), "matching metadata and handles must hash equally");
    CHECK(!a.same_identity_as(c), "different route metadata must change descriptor identity");
    CHECK(a.identity_hash() != c.identity_hash(), "different route metadata must change identity hash");

    using role             = ggml_sycl::moe_gateup_prepack_artifact_role;
    auto different_scratch = configured_descriptor(0x11112222ULL);
    CHECK(different_scratch.add_artifact(role::SOURCE_ACTIVATION, make_arena_handle(TEST_RUNTIME_ZONE, 0x1000, 1024, 1),
                                         1024),
          "source artifact must be retained");
    CHECK(
        different_scratch.add_artifact(role::GATE_WEIGHT, make_arena_handle(TEST_RUNTIME_ZONE, 0x2000, 2048, 1), 2048),
        "gate artifact must be retained");
    CHECK(different_scratch.add_artifact(role::UP_WEIGHT, make_arena_handle(TEST_RUNTIME_ZONE, 0x3000, 2048, 1), 2048),
          "up artifact must be retained");
    CHECK(different_scratch.add_artifact(role::SCRATCH, make_arena_handle(TEST_SCRATCH_ZONE, 0x7000, 4096, 1), 4096),
          "different scratch artifact must be retained");
    CHECK(
        different_scratch.add_artifact(role::ROUTE_METADATA, make_arena_handle(TEST_RUNTIME_ZONE, 0x5000, 512, 1), 512),
        "metadata artifact must be retained");
    CHECK(!a.same_identity_as(different_scratch), "different stable handle identity must change descriptor identity");

    int  external_a = 0;
    int  external_b = 0;
    auto raw_a      = ggml_sycl::mem_handle::from_direct(&external_a, GGML_LAYOUT_AOS, false);
    auto raw_b      = ggml_sycl::mem_handle::from_direct(&external_b, GGML_LAYOUT_AOS, false);
    auto raw_desc   = configured_descriptor();
    CHECK(!raw_desc.add_artifact(role::SOURCE_ACTIVATION, raw_a, sizeof(external_a)),
          "raw direct pointer handle without stable owner identity must be rejected");
    CHECK(!raw_desc.add_artifact(role::SOURCE_ACTIVATION, raw_b, sizeof(external_b)),
          "a different raw direct pointer must also be rejected, not keyed by address");
    CHECK(raw_desc.retained_handle_count() == 0, "rejected raw pointer handles must not be retained");
    CHECK(!raw_desc.uses_raw_pointer_identity_for_test(), "descriptor must not store raw pointer identities");
    return 0;
}

}  // namespace

int main() {
    if (test_empty_invalid() != 0) {
        return 1;
    }
    if (test_handle_retention_counts() != 0) {
        return 1;
    }
    if (test_dependency_retention_counts() != 0) {
        return 1;
    }
    if (test_reset_releases_retained_state() != 0) {
        return 1;
    }
    if (test_identity_is_handle_and_metadata_based() != 0) {
        return 1;
    }
    std::puts("PASS: MoE gate/up prepack scratch descriptor");
    return 0;
}
