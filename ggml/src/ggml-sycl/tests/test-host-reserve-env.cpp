//
// Test: GGML_SYCL_HOST_RESERVE_MB parse robustness (llama.cpp-16el).
//
// The variable is the byte-count override of the pinned-pool BUDGET (live
// since llama.cpp-glkg). Before this ticket it went through the lenient
// strtol helper shared with the DMA/staging tuning knobs, which accepted
// `6144junk` as 6144, ignored ERANGE, and let the MiB->byte multiply wrap.
//
// Host-only: parse_host_reserve_mb is a pure function of the raw string, so
// no GPU, no device, no environment mutation and no libggml-sycl link. Each
// row below is one raw value and the exact result the parser must produce;
// every failing row is reported (not just the first) so a RED run lists the
// whole defect at once.
//
// Written RED against a verbatim port of the lenient helper; the RED text is
// recorded in this ticket's commit body.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "host-reserve-env.hpp"

#include <cstdio>
#include <cstring>

using ggml_sycl::detail::host_reserve_parse_result;
using ggml_sycl::detail::host_reserve_parse_status;
using ggml_sycl::detail::k_host_reserve_mb_max;
using ggml_sycl::detail::parse_host_reserve_mb;

static const char * status_name(host_reserve_parse_status s) {
    switch (s) {
        case host_reserve_parse_status::UNSET:
            return "UNSET";
        case host_reserve_parse_status::AUTO:
            return "AUTO";
        case host_reserve_parse_status::OVERRIDE:
            return "OVERRIDE";
        case host_reserve_parse_status::REJECTED:
            return "REJECTED";
    }
    return "?";
}

struct case_row {
    const char *              raw;     // nullptr = variable unset
    host_reserve_parse_status status;  // expected
    size_t                    mb;      // expected (OVERRIDE / AUTO), ignored otherwise
    const char *              note;    // why this row exists
};

// Boundary MiB count and its byte value, spelled out so a wrong constant in
// the header cannot agree with itself here.
static const size_t MB_MAX = static_cast<size_t>(17592186044415ULL);

int main() {
    int failures = 0;

    if (k_host_reserve_mb_max != MB_MAX) {
        std::fprintf(stderr, "FAIL: k_host_reserve_mb_max=%zu, expected %zu (SIZE_MAX / 2^20 on a 64-bit host)\n",
                     k_host_reserve_mb_max, MB_MAX);
        failures++;
    }

    const case_row rows[] = {
        // --- the rows this ticket was opened for -------------------------------
        { "6144junk",                host_reserve_parse_status::REJECTED, 0,      "trailing junk after digits"            },
        { "-1",                      host_reserve_parse_status::REJECTED, 0,      "negative"                              },
        { "99999999999999999999999", host_reserve_parse_status::REJECTED, 0,      "ERANGE: exceeds unsigned long long"    },
        { "17592186044416",          host_reserve_parse_status::REJECTED, 0,      "MB_MAX+1: byte conversion wraps"       },
        { "18446744073709551615",    host_reserve_parse_status::REJECTED, 0,      "SIZE_MAX: fits ull, wraps on multiply" },
        { "  4096  ",                host_reserve_parse_status::OVERRIDE, 4096,   "surrounding whitespace is allowed"     },
        { "0",                       host_reserve_parse_status::AUTO,     0,      "zero keeps auto (documented)"          },
        { "4096",                    host_reserve_parse_status::OVERRIDE, 4096,   "plain value"                           },
        // --- syntax edges --------------------------------------------------------
        { nullptr,                   host_reserve_parse_status::UNSET,    0,      "variable unset"                        },
        { "",                        host_reserve_parse_status::UNSET,    0,      "empty string is unset, as before"      },
        { "   ",                     host_reserve_parse_status::REJECTED, 0,      "whitespace only: no digits"            },
        { "+4096",                   host_reserve_parse_status::REJECTED, 0,      "sign characters are not accepted"      },
        { "4096 junk",               host_reserve_parse_status::REJECTED, 0,      "junk after whitespace"                 },
        { "0x10",                    host_reserve_parse_status::REJECTED, 0,      "hex is not decimal"                    },
        { "1e3",                     host_reserve_parse_status::REJECTED, 0,      "exponent notation"                     },
        { "40.96",                   host_reserve_parse_status::REJECTED, 0,      "fraction"                              },
        { "-0",                      host_reserve_parse_status::REJECTED, 0,      "signed zero is still a sign"           },
        { "\t0\n",                   host_reserve_parse_status::AUTO,     0,      "zero with whitespace is still auto"    },
        { "0000",                    host_reserve_parse_status::AUTO,     0,      "leading zeros of zero"                 },
        { "0004096",                 host_reserve_parse_status::OVERRIDE, 4096,   "leading zeros of a value"              },
        { "1",                       host_reserve_parse_status::OVERRIDE, 1,      "smallest override"                     },
        { "17592186044415",          host_reserve_parse_status::OVERRIDE, MB_MAX, "MB_MAX: the largest accepted value"    },
    };

    for (const case_row & row : rows) {
        const host_reserve_parse_result got   = parse_host_reserve_mb(row.raw);
        const char *                    shown = row.raw ? row.raw : "<unset>";
        bool                            ok    = got.status == row.status;
        if (ok &&
            (row.status == host_reserve_parse_status::OVERRIDE || row.status == host_reserve_parse_status::AUTO)) {
            ok = got.mb == row.mb;
        }
        if (ok && row.status == host_reserve_parse_status::OVERRIDE) {
            // The byte conversion must be exact, never wrapped.
            ok = got.bytes == row.mb * 1024ULL * 1024ULL && got.bytes / (1024ULL * 1024ULL) == row.mb;
        }
        if (ok && row.status != host_reserve_parse_status::OVERRIDE) {
            ok = got.bytes == 0;
        }
        if (ok) {
            // A rejection must carry a reason; every other status must not.
            ok =
                (row.status == host_reserve_parse_status::REJECTED) == (got.reason != nullptr && got.reason[0] != '\0');
        }
        if (!ok) {
            std::fprintf(stderr, "FAIL: '%s' (%s): expected %s mb=%zu, got %s mb=%zu bytes=%zu reason=%s\n", shown,
                         row.note, status_name(row.status), row.mb, status_name(got.status), got.mb, got.bytes,
                         got.reason ? got.reason : "(none)");
            failures++;
        }
    }

    // The reason strings are part of the WARN the call site prints; a reason
    // that does not distinguish the three failure classes would send someone
    // reading the log after the wrong defect.
    {
        const char * junk_reason  = parse_host_reserve_mb("6144junk").reason;
        const char * neg_reason   = parse_host_reserve_mb("-1").reason;
        const char * range_reason = parse_host_reserve_mb("99999999999999999999999").reason;
        const char * mul_reason   = parse_host_reserve_mb("17592186044416").reason;
        if (!junk_reason || !neg_reason || !range_reason || !mul_reason) {
            std::fprintf(stderr, "FAIL: a rejected value carries no reason string\n");
            failures++;
        } else if (std::strcmp(junk_reason, neg_reason) == 0 || std::strcmp(junk_reason, range_reason) == 0 ||
                   std::strcmp(junk_reason, mul_reason) == 0 || std::strcmp(range_reason, mul_reason) == 0) {
            std::fprintf(stderr,
                         "FAIL: rejection reasons do not distinguish junk/negative/ERANGE/overflow: "
                         "'%s' / '%s' / '%s' / '%s'\n",
                         junk_reason, neg_reason, range_reason, mul_reason);
            failures++;
        }
    }

    if (failures) {
        std::fprintf(stderr, "FAIL: host-reserve-env: %d check(s) failed\n", failures);
        return 1;
    }
    std::printf("PASS: host-reserve-env: %zu parse rows + reason distinctness\n", sizeof(rows) / sizeof(rows[0]));
    return 0;
}
