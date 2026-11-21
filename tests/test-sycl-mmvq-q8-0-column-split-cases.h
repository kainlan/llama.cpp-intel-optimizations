// The case table of test-sycl-mmvq-q8-0-column-split (the device test), in a header that includes nothing so the
// host test test-sycl-mmvq-column-split-rule can hold properties of the table no device run states about itself: that
// every case takes the split on both cards, and that each layout has a case whose last work-group holds padded rows.
//
// Chunks the dispatch rule gives each shape (the same on 256 and on 128 compute units, every row count here being
// below both), rows per work-group (16 / chunks) and the padded rows of the last work-group:
//   SOA, step 8 blocks:        4 x 10240 -> 16 chunks, 1 row/wg,  0 padded
//                              48 x 2560 ->  8 chunks, 2 rows/wg, 0 padded
//                              24 x 2560 ->  8 chunks, 2 rows/wg, 0 padded
//                              23 x 2560 ->  8 chunks, 2 rows/wg, 1 padded
//                              47 x 2560 ->  8 chunks, 2 rows/wg, 1 padded
//   coalesced, step one tile:  4 x 10240 ->  8 chunks, 2 rows/wg, 0 padded
//                              5 x 10240 ->  8 chunks, 2 rows/wg, 1 padded
//                              48 x 2560 ->  2 chunks, 8 rows/wg, 0 padded
//                              45 x 2560 ->  2 chunks, 8 rows/wg, 3 padded

#pragma once

struct mmvq_q8_0_column_split_case {
    const char * name;
    bool         coalesced;
    int          ncols;
    int          total_rows;  // rows of the whole tensor
    int          row_low;     // first row of the slice the launcher is given
    int          nrows;       // rows of that slice
    bool         fused;
};

// clang-format off
static constexpr mmvq_q8_0_column_split_case MMVQ_Q8_0_COLUMN_SPLIT_CASES[] = {
    { "soa-4x10240",                   false, 10240, 4,  0,  4,  false },
    { "soa-4x10240-fused",             false, 10240, 4,  0,  4,  true  },
    { "soa-48x2560",                   false, 2560,  48, 0,  48, false },
    { "soa-48x2560-fused",             false, 2560,  48, 0,  48, true  },
    // A row slice of a SOA tensor, as a split-tensor dispatch passes it: the kernel addresses blocks by
    // row_low + row against the whole tensor's planes and the addend by its row base.
    { "soa-48x2560-slice-16-40-fused", false, 2560,  48, 16, 24, true  },
    // Row counts that leave padded rows in the last work-group: those rows must reach the barrier and write nothing.
    { "soa-48x2560-slice-16-39-fused", false, 2560,  48, 16, 23, true  },
    { "soa-47x2560",                   false, 2560,  47, 0,  47, false },
    { "coalesced-4x10240",             true,  10240, 4,  0,  4,  false },
    { "coalesced-4x10240-fused",       true,  10240, 4,  0,  4,  true  },
    { "coalesced-48x2560",             true,  2560,  48, 0,  48, false },
    { "coalesced-48x2560-fused",       true,  2560,  48, 0,  48, true  },
    { "coalesced-5x10240",             true,  10240, 5,  0,  5,  false },
    { "coalesced-45x2560-fused",       true,  2560,  45, 0,  45, true  },
};
// clang-format on
