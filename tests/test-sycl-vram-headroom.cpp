// VRAM external headroom: the arena's driver-reserved headroom must widen
// when the oneDNN batched MoE PP pipeline is planned, and an explicit env
// override always wins verbatim (llama.cpp perf-recovery epic, track D,
// llama.cpp-seno; spec: llama.cpp-dp5i c-ni04/c-gkai).
#include "ggml-sycl/vram-headroom.hpp"

#include <cstdio>
#define CHECK(cond, msg)                      \
    do {                                      \
        if (!(cond)) {                        \
            std::printf("FAILED: %s\n", msg); \
            return 1;                         \
        }                                     \
    } while (0)

int main() {
    constexpr size_t mib = 1024ull * 1024ull;
    constexpr size_t gib = 1024ull * mib;

    // Env override wins verbatim, regardless of pipeline flag or total VRAM,
    // including the degenerate total_vram_bytes == 0 case.
    CHECK(ggml_sycl::ggml_sycl_vram_external_headroom_bytes(16304ull * mib, false, "2048") == 2048ull * mib,
          "positive env override must win verbatim over the pipeline-off default");
    CHECK(ggml_sycl::ggml_sycl_vram_external_headroom_bytes(16304ull * mib, true, "2048") == 2048ull * mib,
          "positive env override must win verbatim over the pipeline-on default");
    CHECK(ggml_sycl::ggml_sycl_vram_external_headroom_bytes(0, true, "2048") == 2048ull * mib,
          "env override must win even when total_vram_bytes is 0");
    // A non-positive or unparsable override must fall through to the default,
    // not degrade to zero headroom.
    CHECK(ggml_sycl::ggml_sycl_vram_external_headroom_bytes(16304ull * mib, false, "0") > 0,
          "a zero override must fall through to the default, not become 0");
    CHECK(ggml_sycl::ggml_sycl_vram_external_headroom_bytes(16304ull * mib, false, "-5") > 0,
          "a negative override must fall through to the default");
    CHECK(ggml_sycl::ggml_sycl_vram_external_headroom_bytes(16304ull * mib, false, "not-a-number") > 0,
          "an unparsable override must fall through to the default");
    CHECK(ggml_sycl::ggml_sycl_vram_external_headroom_bytes(16304ull * mib, false, nullptr) > 0,
          "nullptr override must fall through to the default");

    // Zero total VRAM opts out of the cap entirely (mirrors
    // arena_default_external_headroom's existing opt-out), independent of the
    // pipeline flag, when there is no override.
    CHECK(ggml_sycl::ggml_sycl_vram_external_headroom_bytes(0, false, nullptr) == 0,
          "zero VRAM total must opt out of headroom (pipeline off)");
    CHECK(ggml_sycl::ggml_sycl_vram_external_headroom_bytes(0, true, nullptr) == 0,
          "zero VRAM total must opt out of headroom (pipeline on)");

    // B50: 16304 MiB. proven-insufficient facts (llama.cpp-dp5i c-gkai): the
    // old flat 630 MB default was insufficient with the batched pipeline
    // planned; the new default must be strictly higher. 6% of 16304 MiB
    // (~978 MiB) is below the 1 GiB floor, so the floor applies exactly.
    const size_t b50_total         = 16304ull * mib;
    const size_t b50_with_pipeline = ggml_sycl::ggml_sycl_vram_external_headroom_bytes(b50_total, true, nullptr);
    CHECK(b50_with_pipeline == gib, "16 GB card with the pipeline planned must hit the 1 GiB floor (6% < floor)");
    CHECK(b50_with_pipeline > 630ull * mib,
          "16 GB card with the pipeline planned must exceed the old 630 MB default (proven insufficient)");

    // Without the pipeline: 4% of 16304 MiB (~652 MiB) exceeds the 630 MB
    // floor, so the proportional term wins -- computed via the same integer
    // arithmetic as the implementation, not a hand-rounded literal.
    const size_t b50_no_pipeline = ggml_sycl::ggml_sycl_vram_external_headroom_bytes(b50_total, false, nullptr);
    CHECK(b50_no_pipeline == b50_total * 4 / 100,
          "16 GB card without the pipeline should use the 4% proportional term (it exceeds the 630 MB floor)");
    CHECK(b50_no_pipeline < b50_with_pipeline,
          "the pipeline-planned default must exceed the pipeline-off default on the same card");

    // A small card where the proportional term (4%) is BELOW the 630 MB
    // floor must clamp to the floor.
    const size_t small_total = 8000ull * mib;
    CHECK(ggml_sycl::ggml_sycl_vram_external_headroom_bytes(small_total, false, nullptr) == 630ull * mib,
          "a small card's 4% term below the 630 MB floor must clamp to the floor");

    // B70: 32586 MiB. Proportionally larger reservation than the B50, purely
    // from the percentage term -- no card-name/device-id check (owner ruling,
    // llama.cpp-dp5i). 6% of 32586 MiB exceeds the 1 GiB floor, so the
    // proportional term wins here.
    const size_t b70_total         = 32586ull * mib;
    const size_t b70_with_pipeline = ggml_sycl::ggml_sycl_vram_external_headroom_bytes(b70_total, true, nullptr);
    CHECK(b70_with_pipeline == b70_total * 6 / 100,
          "32 GB card with the pipeline planned should use the 6% proportional term (it exceeds the 1 GiB floor)");
    CHECK(b70_with_pipeline > b50_with_pipeline,
          "the B70's pipeline-on default must exceed the B50's (proportional, not a flat MB bump)");

    // Monotonicity in total_vram, holding the pipeline flag fixed.
    CHECK(ggml_sycl::ggml_sycl_vram_external_headroom_bytes(b70_total, false, nullptr) >=
              ggml_sycl::ggml_sycl_vram_external_headroom_bytes(b50_total, false, nullptr),
          "headroom must not decrease as total VRAM grows (pipeline off)");
    CHECK(ggml_sycl::ggml_sycl_vram_external_headroom_bytes(b70_total, true, nullptr) >=
              ggml_sycl::ggml_sycl_vram_external_headroom_bytes(b50_total, true, nullptr),
          "headroom must not decrease as total VRAM grows (pipeline on)");

    std::printf("OK: VRAM external headroom semantics\n");
    return 0;
}
