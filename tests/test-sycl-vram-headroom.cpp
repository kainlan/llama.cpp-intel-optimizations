// VRAM external headroom: the arena's driver-reserved headroom must widen
// when the oneDNN batched MoE PP pipeline is planned, an explicit env
// override always wins verbatim WHEN it parses as a valid positive MB value,
// and a rejected/garbage/absurd override must never collapse to something
// smaller than the composed default (llama.cpp perf-recovery epic, track D,
// llama.cpp-seno; spec: llama.cpp-dp5i c-ni04/c-gkai; spec-review findings
// 1, 3, 5, 6).
#include "ggml-sycl/vram-headroom.hpp"

#include <algorithm>
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

    // --- vram_headroom_parse_positive_mb_to_bytes(): the hardened env parser ---
    CHECK(ggml_sycl::vram_headroom_parse_positive_mb_to_bytes(nullptr) == 0, "null override must parse as rejected");
    CHECK(ggml_sycl::vram_headroom_parse_positive_mb_to_bytes("") == 0,
          "empty-string override must be rejected (finding 6)");
    CHECK(ggml_sycl::vram_headroom_parse_positive_mb_to_bytes("0") == 0, "zero must be rejected");
    CHECK(ggml_sycl::vram_headroom_parse_positive_mb_to_bytes("-5") == 0, "negative must be rejected");
    CHECK(ggml_sycl::vram_headroom_parse_positive_mb_to_bytes("not-a-number") == 0, "unparsable must be rejected");
    CHECK(ggml_sycl::vram_headroom_parse_positive_mb_to_bytes("2048") == 2048ull * mib,
          "a clean positive value must parse to the exact MB->bytes conversion");
    // Finding 3: hardening cases, each must reject (return 0) rather than
    // silently produce a wrong/overflowed/wrapped headroom.
    CHECK(ggml_sycl::vram_headroom_parse_positive_mb_to_bytes("2147483648") == 0,
          "a bytes-instead-of-MB typo (2^31) must be rejected, not skip the arena cap entirely");
    CHECK(ggml_sycl::vram_headroom_parse_positive_mb_to_bytes("9999999999999999999") == 0,
          "an ERANGE-overflowing value must be rejected, not silently clamp to LONG_MAX");
    CHECK(ggml_sycl::vram_headroom_parse_positive_mb_to_bytes("17592186044416") == 0,
          "a value whose MB->bytes conversion would wrap size_t (2^44) must be rejected, not silently become 0");
    CHECK(ggml_sycl::vram_headroom_parse_positive_mb_to_bytes("1e9") == 0,
          "trailing non-digit characters (strtol stops at 'e') must reject the whole value, not parse the '1' prefix");

    // --- vram_external_headroom_bytes(): device+pipeline default/floor ---
    CHECK(ggml_sycl::vram_external_headroom_bytes(0, false) == 0,
          "zero VRAM total must opt out of headroom (pipeline off)");
    CHECK(ggml_sycl::vram_external_headroom_bytes(0, true) == 0,
          "zero VRAM total must opt out of headroom (pipeline on)");

    // B50: 16304 MiB. Proven-insufficient facts (llama.cpp-dp5i c-gkai): the
    // old flat 630 MB default was insufficient with the batched pipeline
    // planned; the new default must be strictly higher. 6% of 16304 MiB
    // (~978 MiB) is below the 1 GiB floor, so the floor applies exactly.
    const size_t b50_total         = 16304ull * mib;
    const size_t b50_with_pipeline = ggml_sycl::vram_external_headroom_bytes(b50_total, true);
    CHECK(b50_with_pipeline == gib, "16 GB card with the pipeline planned must hit the 1 GiB floor (6% < floor)");
    CHECK(b50_with_pipeline > 630ull * mib,
          "16 GB card with the pipeline planned must exceed the old 630 MB default (proven insufficient)");

    // Without the pipeline: 4% of 16304 MiB (~652 MiB) exceeds the 630 MB
    // floor, so the proportional term wins -- computed via the same integer
    // arithmetic as the implementation, not a hand-rounded literal.
    const size_t b50_no_pipeline = ggml_sycl::vram_external_headroom_bytes(b50_total, false);
    CHECK(b50_no_pipeline == b50_total * 4 / 100,
          "16 GB card without the pipeline should use the 4% proportional term (it exceeds the 630 MB floor)");
    CHECK(b50_no_pipeline < b50_with_pipeline,
          "the pipeline-planned default must exceed the pipeline-off default on the same card");

    // A small card where the proportional term (4%) is BELOW the 630 MB
    // floor must clamp to the floor.
    const size_t small_total = 8000ull * mib;
    CHECK(ggml_sycl::vram_external_headroom_bytes(small_total, false) == 630ull * mib,
          "a small card's 4% term below the 630 MB floor must clamp to the floor");

    // B70: 32586 MiB. Proportionally larger reservation than the B50, purely
    // from the percentage term -- no card-name/device-id check (owner ruling,
    // llama.cpp-dp5i). 6% of 32586 MiB exceeds the 1 GiB floor, so the
    // proportional term wins here.
    const size_t b70_total         = 32586ull * mib;
    const size_t b70_with_pipeline = ggml_sycl::vram_external_headroom_bytes(b70_total, true);
    CHECK(b70_with_pipeline == b70_total * 6 / 100,
          "32 GB card with the pipeline planned should use the 6% proportional term (it exceeds the 1 GiB floor)");
    CHECK(b70_with_pipeline > b50_with_pipeline,
          "the B70's pipeline-on default must exceed the B50's (proportional, not a flat MB bump)");

    // Monotonicity in total_vram, holding the pipeline flag fixed.
    CHECK(ggml_sycl::vram_external_headroom_bytes(b70_total, false) >=
              ggml_sycl::vram_external_headroom_bytes(b50_total, false),
          "headroom must not decrease as total VRAM grows (pipeline off)");
    CHECK(ggml_sycl::vram_external_headroom_bytes(b70_total, true) >=
              ggml_sycl::vram_external_headroom_bytes(b50_total, true),
          "headroom must not decrease as total VRAM grows (pipeline on)");

    // --- vram_external_headroom_effective(): the arena_reserve() call-site
    // wiring itself, as a pure function (spec-review finding 1). A legacy
    // default_headroom that has already been capped down (e.g. the exact
    // 630 MB c-gkai proved insufficient) must not be silently discarded by a
    // rejected env value -- the composition must still apply the
    // pipeline-aware floor on top of it.
    const size_t legacy_default = 630ull * mib;
    const size_t composed_pipeline_on =
        std::max(legacy_default, ggml_sycl::vram_external_headroom_bytes(b50_total, true));

    // Every rejected/unset override must compose to the SAME result -- NOT
    // the bare pipeline-off ggml_sycl::vram_external_headroom_bytes() result
    // (652 MiB) that the pre-fix wiring silently fell back to.
    CHECK(ggml_sycl::vram_external_headroom_effective(b50_total, legacy_default, true, nullptr) == composed_pipeline_on,
          "unset env must equal the composed default, not collapse to a smaller bare floor");
    CHECK(ggml_sycl::vram_external_headroom_effective(b50_total, legacy_default, true, "") == composed_pipeline_on,
          "empty-string env must compose exactly like unset (finding 6)");
    CHECK(ggml_sycl::vram_external_headroom_effective(b50_total, legacy_default, true, "0") == composed_pipeline_on,
          "a zero env override must compose exactly like unset");
    CHECK(ggml_sycl::vram_external_headroom_effective(b50_total, legacy_default, true, "-5") == composed_pipeline_on,
          "a negative env override must compose exactly like unset");
    CHECK(ggml_sycl::vram_external_headroom_effective(b50_total, legacy_default, true, "not-a-number") ==
              composed_pipeline_on,
          "an unparsable env override must compose exactly like unset");
    // Finding 3, re-verified at the wiring level: each hardening case must
    // compose as unset here too, not merely reject at the parser.
    CHECK(ggml_sycl::vram_external_headroom_effective(b50_total, legacy_default, true, "2147483648") ==
              composed_pipeline_on,
          "a bytes-instead-of-MB typo must compose as unset, not silently skip the arena cap");
    CHECK(ggml_sycl::vram_external_headroom_effective(b50_total, legacy_default, true, "9999999999999999999") ==
              composed_pipeline_on,
          "an ERANGE-overflowing override must compose as unset");
    CHECK(ggml_sycl::vram_external_headroom_effective(b50_total, legacy_default, true, "17592186044416") ==
              composed_pipeline_on,
          "a size_t-wrapping override must compose as unset, not silently become a near-zero headroom");
    CHECK(ggml_sycl::vram_external_headroom_effective(b50_total, legacy_default, true, "1e9") == composed_pipeline_on,
          "a trailing-garbage override must compose as unset");

    // A positive, fully-valid override wins verbatim -- even LOWER than both
    // default_headroom and the pipeline floor. An operator is allowed to
    // deliberately lower headroom, not just raise it.
    CHECK(ggml_sycl::vram_external_headroom_effective(b50_total, legacy_default, true, "300") == 300ull * mib,
          "a positive env override must win verbatim, even when lower than both the default and the floor");

    // Unset, pipeline off, with a default_headroom that already exceeds the
    // pipeline-off floor: the composition must honor default_headroom (the
    // max, not the floor alone).
    const size_t large_default = 2000ull * mib;
    CHECK(ggml_sycl::vram_external_headroom_effective(b50_total, large_default, false, nullptr) == large_default,
          "unset override must preserve a default_headroom that already exceeds the pipeline-off floor");

    // Finding 5: total_vram_bytes == 0 is asymmetric versus
    // vram_external_headroom_bytes(0, ...), which returns 0 outright.
    // vram_external_headroom_effective(0, ...) instead preserves
    // default_headroom unchanged, because the pipeline-aware floor collapses
    // to 0 at zero total: max(default_headroom, 0) == default_headroom.
    CHECK(
        ggml_sycl::vram_external_headroom_effective(0, legacy_default, true, nullptr) == legacy_default,
        "zero total_vram_bytes must preserve default_headroom unchanged (asymmetric vs vram_external_headroom_bytes)");

    std::printf("OK: VRAM external headroom semantics\n");
    return 0;
}
