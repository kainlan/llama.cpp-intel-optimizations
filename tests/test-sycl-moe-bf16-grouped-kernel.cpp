// The gather and the scatter of the BF16 grouped prompt arm of MUL_MAT_ID (ggml-sycl/moe-bf16-grouped.{hpp,cpp}, Task
// 24 of the Inkling 2 Mini plan), on rows laid out by the arm's own plan (moe-bf16-grouped-plan.hpp).
//
// Compiles moe-bf16-grouped.cpp into this binary and runs it on whatever device ONEAPI_DEVICE_SELECTOR picks: the
// registration pins opencl:cpu, so this needs no GPU. Run it with the selector set to a GPU to check the same kernels
// there.
//
// The gather is the arm's only rounding: the oneDNN GEMMs multiply the bf16 plane it writes. So:
//   - the rounding itself: a table of f32 inputs, each with the bf16 bits round-to-nearest-even gives it (ties to even
//     both ways, one ulp either side of a tie, the largest finite rounding to infinity, infinities, quiet and
//     signalling NaN made quiet, subnormals, zeros), must hold for ggml_sycl_bf16::f32_to_bf16_rne and for the CPU
//     backend's ggml_fp32_to_bf16, which agree on every input of every case;
//   - every routed row of the gathered plane is bit for bit f32_to_bf16_rne of its src1 row, on activations almost all
//     of which are not bf16 values: the table's inputs first (all but its few exact values, 1.0, the zeros, the
//     infinities and a quiet NaN, lie between two bf16 values), then random 32-bit patterns (NaN payloads and
//     subnormals included);
//   - the scatter writes every routed row's f32 output row to its (slot, token) row of dst, bit for bit;
//   - the plane's padding rows, every dst element no routed row names (row padding and gaps between tokens) and every
//     input come back untouched;
//   - a broadcast activation row and a row per slot, packed and padded strides, Inkling's two expert shapes;
//   - each refusal (a null pointer, a shape the kernels do not take) returns false.
//
// Exits 77 (skip) only when the selector leaves no device to make a queue on; any failure after that is a FAIL.

#include "ggml-sycl/moe-bf16-grouped.hpp"
#include "ggml-sycl/mul-mat-bf16.hpp"  // ggml_sycl_bf16::f32_to_bf16_rne: the rounding the gather must match
#include "ggml.h"                      // ggml_fp32_to_bf16: the CPU backend's rounding
#include "test-skip.h"                 // LLAMA_TEST_EXIT_SKIP

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <optional>
#include <random>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond, ...)                       \
    do {                                       \
        if (!(cond)) {                         \
            std::printf("FAIL: " __VA_ARGS__); \
            std::printf("\n");                 \
            ++g_failures;                      \
        }                                      \
    } while (0)

float bits_to_f32(uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

uint16_t cpu_bf16(float f) {
    return ggml_fp32_to_bf16(f).bits;
}

// One f32 input and the bf16 bits round-to-nearest-even gives it.
struct rounding_point {
    uint32_t     f32;
    uint16_t     bf16;
    const char * what;
};

const rounding_point k_rounding[] = {
    { 0x3f800000u, 0x3f80u, "1.0, a bf16 value"                                            },
    { 0x3f808000u, 0x3f80u, "a tie, upper half even: down"                                 },
    { 0x3f818000u, 0x3f82u, "a tie, upper half odd: up"                                    },
    { 0xbf808000u, 0xbf80u, "a negative tie, even: toward zero"                            },
    { 0xbf818000u, 0xbf82u, "a negative tie, odd: away from zero"                          },
    { 0x3f807fffu, 0x3f80u, "one ulp below a tie: down"                                    },
    { 0x3f808001u, 0x3f81u, "one ulp above a tie: up"                                      },
    { 0x3f80c000u, 0x3f81u, "three quarters of an ulp, even: up"                           },
    { 0x3f81ffffu, 0x3f82u, "just below the next bf16 value: up"                           },
    { 0x40490fdbu, 0x4049u, "pi in f32: down"                                              },
    { 0x7f7f7fffu, 0x7f7fu, "below the tie under the largest finite bf16: down"            },
    { 0x7f7f8000u, 0x7f80u, "a tie at the largest finite bf16, odd: up to infinity"        },
    { 0x7f7fffffu, 0x7f80u, "the largest finite f32: infinity"                             },
    { 0xff7fffffu, 0xff80u, "the most negative finite f32: minus infinity"                 },
    { 0x7f800000u, 0x7f80u, "infinity"                                                     },
    { 0xff800000u, 0xff80u, "minus infinity"                                               },
    { 0x7fc00000u, 0x7fc0u, "a quiet NaN"                                                  },
    { 0x7f800001u, 0x7fc0u, "a signalling NaN, payload in the low half: made quiet"        },
    { 0x7fa00001u, 0x7fe0u, "a signalling NaN, payload in the high half: kept, made quiet" },
    { 0xffc00001u, 0xffc0u, "a negative quiet NaN"                                         },
    { 0x00000001u, 0x0000u, "the smallest subnormal: zero"                                 },
    { 0x00008000u, 0x0000u, "a subnormal tie, even: zero"                                  },
    { 0x00018000u, 0x0002u, "a subnormal tie, odd: up"                                     },
    { 0x007fffffu, 0x0080u, "the largest subnormal: the smallest normal"                   },
    { 0x80000001u, 0x8000u, "the smallest negative subnormal: minus zero"                  },
    { 0x00000000u, 0x0000u, "zero"                                                         },
    { 0x80000000u, 0x8000u, "minus zero"                                                   },
};

void check_rounding_table() {
    for (const rounding_point & p : k_rounding) {
        const float    f   = bits_to_f32(p.f32);
        const uint16_t rne = ggml_sycl_bf16::f32_to_bf16_rne(f);
        const uint16_t cpu = cpu_bf16(f);
        CHECK(rne == p.bf16, "f32_to_bf16_rne(0x%08x) = 0x%04x, want 0x%04x (%s)", p.f32, rne, p.bf16, p.what);
        CHECK(cpu == p.bf16, "ggml_fp32_to_bf16(0x%08x) = 0x%04x, want 0x%04x (%s)", p.f32, cpu, p.bf16, p.what);
    }
}

struct grouped_case {
    const char * name;
    int          k;          // an activation row's width
    int          m;          // an output row's width
    int          n_slots;    // ids per token
    int          n_tokens;
    int          src1_rows;  // 1: one activation row broadcast to every slot; n_slots: one row per slot
    int          n_experts;
    int          x_pad;      // floats after each src1 row
    int          x_gap;      // floats after each token's src1 rows
    int          y_pad;      // floats after each dst row
    int          y_gap;      // floats after each token's dst rows
};

constexpr uint16_t k_act_guard = 0xa5a5u;      // a padding row of the plane
constexpr uint32_t k_dst_guard = 0x7fbadbadu;  // a dst element no routed row names

void run_one(sycl::queue & q, const grouped_case & c, std::mt19937 & rng) {
    // ids: slot 0 routes to expert 0 and slot s > 0 to 1 + (3 t + s + t / 2) mod (n_experts - 1), distinct within a
    // token (a one-slot case never reaches the modulus). Expert 0 takes a row of every token, so a case of more than
    // 64 tokens gives it a wider GEMM slot than the others: two widths in one plane.
    std::vector<int32_t> ids(size_t(c.n_tokens) * c.n_slots);
    for (int t = 0; t < c.n_tokens; ++t) {
        for (int s = 0; s < c.n_slots; ++s) {
            ids[size_t(t) * c.n_slots + s] = s == 0 ? 0 : 1 + (3 * t + s + t / 2) % (c.n_experts - 1);
        }
    }
    ggml_sycl::moe_bf16_grouped_plan              plan;
    const ggml_sycl::moe_bf16_grouped_plan_status status = ggml_sycl::moe_bf16_grouped_plan_build(
        ids.data(), size_t(c.n_tokens), size_t(c.n_slots), size_t(c.n_experts), &plan);
    CHECK(status == ggml_sycl::moe_bf16_grouped_plan_status::OK, "%s: the plan did not build (%s)", c.name,
          ggml_sycl::moe_bf16_grouped_plan_status_name(status));
    if (status != ggml_sycl::moe_bf16_grouped_plan_status::OK) {
        return;
    }

    const size_t k        = size_t(c.k);
    const size_t m        = size_t(c.m);
    const size_t x_nb1    = k + size_t(c.x_pad);  // in floats
    const size_t x_nb2    = size_t(c.src1_rows) * x_nb1 + size_t(c.x_gap);
    const size_t y_nb1    = m + size_t(c.y_pad);
    const size_t y_nb2    = size_t(c.n_slots) * y_nb1 + size_t(c.y_gap);
    const size_t x_elems  = size_t(c.n_tokens) * x_nb2;
    const size_t y_elems  = size_t(c.n_tokens) * y_nb2;
    const size_t act_size = plan.packed_rows * k;
    const size_t out_size = plan.packed_rows * m;

    ggml_sycl::moe_bf16_grouped_io_shape io;
    io.k         = c.k;
    io.m         = c.m;
    io.src1_rows = c.src1_rows;
    io.src1_nb1  = x_nb1 * sizeof(float);
    io.src1_nb2  = x_nb2 * sizeof(float);
    io.dst_nb1   = y_nb1 * sizeof(float);
    io.dst_nb2   = y_nb2 * sizeof(float);
    io.n_rows    = plan.rows.size();
    CHECK(ggml_sycl::moe_bf16_grouped_io_shape_ok(io), "%s: the shape was refused", c.name);

    // src1: the rounding table's inputs first, then random 32-bit patterns, in every element a row reads.
    std::vector<uint32_t> x_bits(x_elems);
    for (uint32_t & b : x_bits) {
        b = rng();
    }
    size_t special = 0;
    for (int t = 0; t < c.n_tokens; ++t) {
        for (int r = 0; r < c.src1_rows; ++r) {
            for (size_t j = 0; j < k && special < std::size(k_rounding); ++j) {
                x_bits[size_t(t) * x_nb2 + size_t(r) * x_nb1 + j] = k_rounding[special++].f32;
            }
        }
    }
    std::vector<float>                    out_host(out_size);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    for (float & v : out_host) {
        v = dist(rng);
    }

    auto *     rows_dev = sycl::malloc_device<ggml_sycl::moe_bf16_grouped_row>(plan.rows.size(), q);
    uint32_t * x_dev    = sycl::malloc_device<uint32_t>(x_elems, q);
    uint16_t * act_dev  = sycl::malloc_device<uint16_t>(act_size, q);
    float *    out_dev  = sycl::malloc_device<float>(out_size, q);
    uint32_t * y_dev    = sycl::malloc_device<uint32_t>(y_elems, q);
    CHECK(rows_dev && x_dev && act_dev && out_dev && y_dev, "%s: device allocation failed", c.name);
    if (!(rows_dev && x_dev && act_dev && out_dev && y_dev)) {
        sycl::free(rows_dev, q);
        sycl::free(x_dev, q);
        sycl::free(act_dev, q);
        sycl::free(out_dev, q);
        sycl::free(y_dev, q);
        return;
    }
    q.memcpy(rows_dev, plan.rows.data(), plan.rows.size() * sizeof(ggml_sycl::moe_bf16_grouped_row)).wait();
    q.memcpy(x_dev, x_bits.data(), x_elems * sizeof(uint32_t)).wait();
    q.memcpy(out_dev, out_host.data(), out_size * sizeof(float)).wait();
    q.fill(act_dev, k_act_guard, act_size).wait();
    q.fill(y_dev, k_dst_guard, y_elems).wait();

    sycl::event gathered;
    const bool  gather_ok = ggml_sycl::moe_bf16_grouped_gather_submit(
        q, rows_dev, reinterpret_cast<const float *>(x_dev), act_dev, io, {}, &gathered);
    CHECK(gather_ok, "%s: the gather was refused", c.name);
    sycl::event scattered;
    const bool  scatter_ok =
        gather_ok && ggml_sycl::moe_bf16_grouped_scatter_submit(q, rows_dev, out_dev, reinterpret_cast<float *>(y_dev),
                                                                io, { gathered }, &scattered);
    CHECK(scatter_ok, "%s: the scatter was refused", c.name);
    if (scatter_ok) {
        scattered.wait();
    }
    q.wait();

    std::vector<uint16_t> act(act_size);
    std::vector<uint32_t> y(y_elems);
    std::vector<uint32_t> x_after(x_elems);
    std::vector<float>    out_after(out_size);
    q.memcpy(act.data(), act_dev, act_size * sizeof(uint16_t)).wait();
    q.memcpy(y.data(), y_dev, y_elems * sizeof(uint32_t)).wait();
    q.memcpy(x_after.data(), x_dev, x_elems * sizeof(uint32_t)).wait();
    q.memcpy(out_after.data(), out_dev, out_size * sizeof(float)).wait();

    // The gathered plane: every routed row bit for bit, and the two host roundings agreeing on every input.
    std::vector<bool> packed_routed(plan.packed_rows, false);
    std::vector<bool> dst_routed(size_t(c.n_tokens) * c.n_slots, false);
    size_t            bad_act = 0, oracle_disagree = 0, bad_dst = 0;
    for (const ggml_sycl::moe_bf16_grouped_row & row : plan.rows) {
        packed_routed[size_t(row.packed_row)]                        = true;
        dst_routed[size_t(row.token) * c.n_slots + size_t(row.slot)] = true;
        const size_t x_row = size_t(row.token) * x_nb2 + size_t(row.slot % c.src1_rows) * x_nb1;
        for (size_t j = 0; j < k; ++j) {
            const float    f    = bits_to_f32(x_bits[x_row + j]);
            const uint16_t want = ggml_sycl_bf16::f32_to_bf16_rne(f);
            const uint16_t got  = act[size_t(row.packed_row) * k + j];
            if (want != cpu_bf16(f)) {
                if (oracle_disagree++ < 4) {
                    std::printf("  %s: f32 0x%08x: f32_to_bf16_rne 0x%04x, ggml_fp32_to_bf16 0x%04x\n", c.name,
                                x_bits[x_row + j], want, cpu_bf16(f));
                }
            }
            if (got != want) {
                if (bad_act++ < 4) {
                    std::printf("  %s: slot %d token %d col %zu: f32 0x%08x gathered 0x%04x, want 0x%04x\n", c.name,
                                row.slot, row.token, j, x_bits[x_row + j], got, want);
                }
            }
        }
        const size_t y_row = size_t(row.token) * y_nb2 + size_t(row.slot) * y_nb1;
        for (size_t i = 0; i < m; ++i) {
            uint32_t want;
            std::memcpy(&want, &out_host[size_t(row.packed_row) * m + i], sizeof(want));
            if (y[y_row + i] != want) {
                if (bad_dst++ < 4) {
                    std::printf("  %s: slot %d token %d col %zu: dst 0x%08x, want 0x%08x\n", c.name, row.slot,
                                row.token, i, y[y_row + i], want);
                }
            }
        }
    }
    CHECK(bad_act == 0, "%s: %zu gathered values differ from f32_to_bf16_rne", c.name, bad_act);
    CHECK(oracle_disagree == 0, "%s: %zu inputs where the two host roundings disagree", c.name, oracle_disagree);
    CHECK(bad_dst == 0, "%s: %zu scattered values differ from the output plane", c.name, bad_dst);

    // Untouched: the plane's padding rows, every dst element no routed row names, and both inputs.
    size_t padding_written = 0;
    for (size_t p = 0; p < plan.packed_rows; ++p) {
        if (packed_routed[p]) {
            continue;
        }
        for (size_t j = 0; j < k; ++j) {
            padding_written += act[p * k + j] != k_act_guard ? 1 : 0;
        }
    }
    CHECK(padding_written == 0, "%s: the gather wrote %zu values into padding rows", c.name, padding_written);
    size_t unrouted_written = 0;
    for (int t = 0; t < c.n_tokens; ++t) {
        for (size_t e = 0; e < y_nb2; ++e) {
            const size_t s      = e / y_nb1;
            const size_t i      = e % y_nb1;
            const bool   in_row = s < size_t(c.n_slots) && i < m;
            const bool   routed = in_row && dst_routed[size_t(t) * c.n_slots + s];
            if (!routed) {
                unrouted_written += y[size_t(t) * y_nb2 + e] != k_dst_guard ? 1 : 0;
            }
        }
    }
    CHECK(unrouted_written == 0, "%s: the scatter wrote %zu dst values no routed row names", c.name, unrouted_written);
    CHECK(x_after == x_bits, "%s: the gather changed src1", c.name);
    CHECK(std::memcmp(out_after.data(), out_host.data(), out_size * sizeof(float)) == 0,
          "%s: the scatter changed the output plane", c.name);

    std::printf(
        "CASE %s k=%d m=%d slots=%d tokens=%d src1_rows=%d experts=%zu routed_rows=%zu packed_rows=%zu "
        "widths=%zu\n",
        c.name, c.k, c.m, c.n_slots, c.n_tokens, c.src1_rows, plan.experts.size(), plan.rows.size(), plan.packed_rows,
        plan.gemm_widths.size());

    sycl::free(rows_dev, q);
    sycl::free(x_dev, q);
    sycl::free(act_dev, q);
    sycl::free(out_dev, q);
    sycl::free(y_dev, q);
}

void check_refusals(sycl::queue & q) {
    ggml_sycl::moe_bf16_grouped_io_shape shape;
    shape.k         = 64;
    shape.m         = 32;
    shape.src1_rows = 1;
    shape.src1_nb1  = 256;
    shape.src1_nb2  = 256;
    shape.dst_nb1   = 128;
    shape.dst_nb2   = 256;
    shape.n_rows    = 2;
    CHECK(ggml_sycl::moe_bf16_grouped_io_shape_ok(shape), "the control shape was refused");

    // Pointers are never dereferenced on a refusal, so host addresses stand in. A refusal leaves the caller's event as
    // it was: `ev` starts as a copy of `marker` and must still be one after every call (a SYCL event compares by the
    // operation it names).
    const ggml_sycl::moe_bf16_grouped_row rows[2] = {};
    float                                 x[64]   = {};
    uint16_t                              act[64] = {};
    float                                 out[32] = {};
    float                                 y[64]   = {};
    const sycl::event                     marker  = q.ext_oneapi_submit_barrier();
    sycl::event                           ev      = marker;
    const auto                            refusal = [&](bool accepted, const char * what) {
        CHECK(!accepted, "%s was accepted", what);
        CHECK(ev == marker, "%s: the refusal changed the caller's event", what);
        ev = marker;
    };
    refusal(ggml_sycl::moe_bf16_grouped_gather_submit(q, nullptr, x, act, shape, {}, &ev), "a gather of no row map");
    refusal(ggml_sycl::moe_bf16_grouped_gather_submit(q, rows, nullptr, act, shape, {}, &ev), "a gather of no src1");
    refusal(ggml_sycl::moe_bf16_grouped_gather_submit(q, rows, x, nullptr, shape, {}, &ev), "a gather into no plane");
    refusal(ggml_sycl::moe_bf16_grouped_gather_submit(q, rows, x, act, shape, {}, nullptr), "a gather with no event");
    refusal(ggml_sycl::moe_bf16_grouped_scatter_submit(q, nullptr, out, y, shape, {}, &ev), "a scatter of no row map");
    refusal(ggml_sycl::moe_bf16_grouped_scatter_submit(q, rows, nullptr, y, shape, {}, &ev),
            "a scatter of no output plane");
    refusal(ggml_sycl::moe_bf16_grouped_scatter_submit(q, rows, out, nullptr, shape, {}, &ev), "a scatter into no dst");
    refusal(ggml_sycl::moe_bf16_grouped_scatter_submit(q, rows, out, y, shape, {}, nullptr), "a scatter with no event");

    const auto refused = [&](ggml_sycl::moe_bf16_grouped_io_shape bad, const char * what) {
        CHECK(!ggml_sycl::moe_bf16_grouped_io_shape_ok(bad), "%s: the shape was accepted", what);
        refusal(ggml_sycl::moe_bf16_grouped_gather_submit(q, rows, x, act, bad, {}, &ev), what);
        refusal(ggml_sycl::moe_bf16_grouped_scatter_submit(q, rows, out, y, bad, {}, &ev), what);
    };
    ggml_sycl::moe_bf16_grouped_io_shape bad = shape;
    bad.k                                    = 0;
    refused(bad, "k=0");
    bad   = shape;
    bad.m = 0;
    refused(bad, "m=0");
    bad           = shape;
    bad.src1_rows = 0;
    refused(bad, "src1_rows=0");
    bad        = shape;
    bad.n_rows = 0;
    refused(bad, "no rows");
    bad          = shape;
    bad.src1_nb1 = 254;
    refused(bad, "a src1 row stride that is not whole floats");
    bad         = shape;
    bad.dst_nb2 = 258;
    refused(bad, "a dst token stride that is not whole floats");
    // A launch's range must fit an int: rows times the wider plane's width (k here), and the last row count that fits
    // is a shape the kernels take.
    bad        = shape;
    bad.n_rows = size_t(INT32_MAX) / size_t(shape.k) + 1;
    refused(bad, "rows times k past an int");
    bad        = shape;
    bad.n_rows = size_t(INT32_MAX) / size_t(shape.k);
    CHECK(ggml_sycl::moe_bf16_grouped_io_shape_ok(bad), "the most rows whose range fits an int were refused");
    bad        = shape;
    bad.m      = 2 * shape.k;
    bad.n_rows = size_t(INT32_MAX) / size_t(bad.m) + 1;
    refused(bad, "rows times m past an int, m the wider");
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    check_rounding_table();

    // The registration pins opencl:cpu; a host without that runtime has no device to test on.
    std::optional<sycl::queue> queue;
    try {
        queue.emplace(sycl::default_selector_v, sycl::property::queue::in_order());
    } catch (const sycl::exception & e) {
        std::printf("SKIP: no SYCL device for this selector: %s\n", e.what());
        return LLAMA_TEST_EXIT_SKIP;
    }
    sycl::queue & q = *queue;
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    check_refusals(q);

    std::mt19937       rng(24);
    const grouped_case cases[] = {
        // name                                            k     m     slt tok s1 exp xp xg yp yg
        { "rounding table and random bits, row per slot", 40,   24,   2, 3,  2, 3,  0, 0, 0, 0 },
        { "Inkling gate_up shape, broadcast row",         2304, 2304, 8, 9,  1, 16, 0, 0, 0, 0 },
        { "Inkling down shape, row per slot",             1152, 2304, 8, 9,  8, 16, 0, 0, 0, 0 },
        { "padded strides everywhere",                    33,   17,   3, 5,  3, 4,  3, 5, 2, 7 },
        { "padded strides, broadcast row",                48,   20,   4, 70, 1, 6,  1, 2, 3, 4 },
        { "one token, one slot, one expert",              1,    1,    1, 1,  1, 1,  0, 0, 0, 0 },
    };
    for (const grouped_case & c : cases) {
        run_one(q, c, rng);
    }

    if (g_failures != 0) {
        std::printf("FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
