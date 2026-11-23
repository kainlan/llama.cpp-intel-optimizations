// The BF16 MUL_MAT_ID expert kernel (ggml-sycl/moe-bf16-direct.{hpp,cpp}, Task 8g of the Inkling 2 Mini plan)
// against a double-precision reference.
//
// Compiles moe-bf16-direct.cpp into this binary and runs it on whatever device ONEAPI_DEVICE_SELECTOR picks: the
// registration pins opencl:cpu, so this needs no GPU. Run it with the selector set to a GPU to check the same kernel
// there.
//
// Each case builds an expert table whose entries point into one device buffer, ids, activations and an output with
// their own byte strides, and checks every output row:
//   - a valid slot matches sum_k bf16(W[id][row][k]) * x[k] within 2e-5 of sum_k |w * x|. The kernel sums in F32:
//     each lane chains k / 16 FMAs and the 16 lanes meet in four reduction steps, so its rounding error stays below
//     (k / 16 + 4) * 2^-24 of that magnitude, under 1e-5 for every k here;
//   - an id of -1 writes a row of exact zeros (the CPU backend's unused slot);
//   - any other id outside [0, n_experts), and an expert whose table entry is null, writes a row of quiet NaN;
//   - the row padding of the output, the guards around it and every input come back untouched, and a second launch
//     writes the same bits.

#include "ggml-sycl/moe-bf16-direct.hpp"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, ...)                                     \
    do {                                                     \
        if (!(cond)) {                                       \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                        \
            std::printf("\n");                               \
            ++g_failures;                                    \
        }                                                    \
    } while (0)

struct kernel_case {
    const char * name;
    int64_t      k;
    int64_t      rows;
    int64_t      n_experts;
    int64_t      n_slots;
    int64_t      n_tokens;
    int64_t      src1_rows;          // 1 broadcasts one activation row to every slot, else one row per slot
    int64_t      w_row_pad;          // extra BF16 elements per weight row
    int64_t      src1_row_pad;       // extra floats per activation row
    int64_t      src1_offset;        // floats skipped at the activation base (defeats the float4 load)
    int64_t      dst_row_pad;        // extra floats per output row
    int64_t      ids_stride;         // int32 elements between two slots of one token
    int64_t      misaligned_expert;  // expert whose base sits 2 bytes off an 8-byte boundary, -1 for none
    bool         bad_slots;          // token 0 carries -1, out-of-range and null-entry slots; the last entry is null
};

// Guard floats on both sides of the output; poisoned so a stray write is detected.
constexpr int64_t GUARD   = 1024;
constexpr float   POISON  = -12345.0f;
constexpr double  REL_TOL = 2e-5;
constexpr size_t  W_ALIGN = 64;  // every expert base but the misaligned one is 64-byte aligned in the buffer

static void run_one(sycl::queue & q, const kernel_case & c) {
    std::mt19937                          rng(4242 + static_cast<unsigned>(c.k * 131 + c.rows * 17 + c.n_slots));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const int64_t w_row_elems  = c.k + c.w_row_pad;
    const size_t  w_nb1        = static_cast<size_t>(w_row_elems) * sizeof(uint16_t);
    const size_t  expert_bytes = static_cast<size_t>(c.rows) * w_nb1;
    const size_t  expert_slot  = (expert_bytes + 2 + W_ALIGN - 1) / W_ALIGN * W_ALIGN;

    // Weights: one byte image holding every expert at its own base, plus the decoded values for the reference.
    std::vector<uint8_t> w_bytes(expert_slot * static_cast<size_t>(c.n_experts), 0xA5);
    std::vector<size_t>  w_base(static_cast<size_t>(c.n_experts));
    std::vector<float>   w_dec(static_cast<size_t>(c.n_experts * c.rows * c.k));
    for (int64_t e = 0; e < c.n_experts; ++e) {
        w_base[e] = static_cast<size_t>(e) * expert_slot + (e == c.misaligned_expert ? 2 : 0);
        for (int64_t r = 0; r < c.rows; ++r) {
            for (int64_t i = 0; i < c.k; ++i) {
                const ggml_bf16_t b = ggml_fp32_to_bf16(dist(rng));
                std::memcpy(&w_bytes[w_base[e] + static_cast<size_t>(r) * w_nb1 + static_cast<size_t>(i) * 2], &b.bits,
                            sizeof(uint16_t));
                w_dec[(e * c.rows + r) * c.k + i] = ggml_bf16_to_fp32(b);
            }
        }
    }

    // Activations: [offset][n_tokens x src1_rows rows of (k + pad) floats].
    const int64_t      src1_row_elems = c.k + c.src1_row_pad;
    const size_t       src1_nb1       = static_cast<size_t>(src1_row_elems) * sizeof(float);
    const size_t       src1_nb2       = static_cast<size_t>(c.src1_rows) * src1_nb1;
    std::vector<float> x_host(static_cast<size_t>(c.src1_offset + c.n_tokens * c.src1_rows * src1_row_elems));
    for (float & v : x_host) {
        v = dist(rng);
    }

    // Ids: slot s of token t at t * ids_nb1 + s * ids_nb0. The null entry (bad_slots) is never a valid choice.
    const int64_t        usable  = c.bad_slots ? c.n_experts - 1 : c.n_experts;
    const size_t         ids_nb0 = static_cast<size_t>(c.ids_stride) * sizeof(int32_t);
    const size_t         ids_nb1 = static_cast<size_t>(c.n_slots * c.ids_stride + 1) * sizeof(int32_t);
    std::vector<int32_t> ids_host(static_cast<size_t>(c.n_tokens) * ids_nb1 / sizeof(int32_t), 0x7FFF0000);
    auto                 id_at = [&](int64_t t, int64_t s) -> int32_t & {
        return ids_host[(static_cast<size_t>(t) * ids_nb1 + static_cast<size_t>(s) * ids_nb0) / sizeof(int32_t)];
    };
    for (int64_t t = 0; t < c.n_tokens; ++t) {
        for (int64_t s = 0; s < c.n_slots; ++s) {
            id_at(t, s) = static_cast<int32_t>((s * 3 + t * 5) % usable);
        }
    }
    if (c.bad_slots) {
        const int32_t bad[] = { -1, static_cast<int32_t>(c.n_experts), -7, static_cast<int32_t>(c.n_experts - 1) };
        for (int64_t s = 0; s < c.n_slots && s < 4; ++s) {
            id_at(0, s) = bad[s];
        }
    }

    // Output: [guard][n_tokens x n_slots rows of (rows + pad) floats][guard], all poisoned.
    const int64_t      dst_row_elems = c.rows + c.dst_row_pad;
    const size_t       dst_nb1       = static_cast<size_t>(dst_row_elems) * sizeof(float);
    const size_t       dst_nb2       = static_cast<size_t>(c.n_slots) * dst_nb1;
    const int64_t      dst_elems     = c.n_tokens * c.n_slots * dst_row_elems;
    std::vector<float> y_init(static_cast<size_t>(dst_elems + 2 * GUARD), POISON);

    uint8_t *     w_dev   = sycl::malloc_device<uint8_t>(w_bytes.size(), q);
    float *       x_dev   = sycl::malloc_device<float>(x_host.size(), q);
    int32_t *     ids_dev = sycl::malloc_device<int32_t>(ids_host.size(), q);
    float *       y_dev   = sycl::malloc_device<float>(y_init.size(), q);
    const void ** tbl_dev = sycl::malloc_device<const void *>(static_cast<size_t>(c.n_experts), q);

    std::vector<const void *> table(static_cast<size_t>(c.n_experts));
    for (int64_t e = 0; e < c.n_experts; ++e) {
        table[e] = (c.bad_slots && e == c.n_experts - 1) ? nullptr : static_cast<const void *>(w_dev + w_base[e]);
    }
    q.memcpy(w_dev, w_bytes.data(), w_bytes.size());
    q.memcpy(x_dev, x_host.data(), x_host.size() * sizeof(float));
    q.memcpy(ids_dev, ids_host.data(), ids_host.size() * sizeof(int32_t));
    q.memcpy(y_dev, y_init.data(), y_init.size() * sizeof(float));
    q.memcpy(tbl_dev, table.data(), table.size() * sizeof(const void *));
    q.wait();

    ggml_sycl::moe_bf16_direct_shape shape;
    shape.k         = c.k;
    shape.rows      = c.rows;
    shape.n_experts = c.n_experts;
    shape.n_slots   = c.n_slots;
    shape.n_tokens  = c.n_tokens;
    shape.src1_rows = c.src1_rows;
    shape.w_nb1     = w_nb1;
    shape.src1_nb1  = src1_nb1;
    shape.src1_nb2  = src1_nb2;
    shape.ids_nb0   = ids_nb0;
    shape.ids_nb1   = ids_nb1;
    shape.dst_nb1   = dst_nb1;
    shape.dst_nb2   = dst_nb2;

    const float * x_base = x_dev + c.src1_offset;
    float *       y_base = y_dev + GUARD;
    sycl::event   ev;
    const bool    ok = ggml_sycl::moe_bf16_direct_submit(q, tbl_dev, x_base, ids_dev, y_base, shape, &ev);
    CHECK(ok, "%s: the submitter refused a valid shape", c.name);
    if (!ok) {
        sycl::free(w_dev, q);
        sycl::free(x_dev, q);
        sycl::free(ids_dev, q);
        sycl::free(y_dev, q);
        sycl::free(tbl_dev, q);
        return;
    }
    ev.wait();
    std::vector<float> y_all(y_init.size());
    q.memcpy(y_all.data(), y_dev, y_all.size() * sizeof(float)).wait();

    sycl::event ev2;
    ggml_sycl::moe_bf16_direct_submit(q, tbl_dev, x_base, ids_dev, y_base, shape, &ev2);
    ev2.wait();
    std::vector<float> y_again(y_init.size());
    q.memcpy(y_again.data(), y_dev, y_again.size() * sizeof(float)).wait();

    std::vector<uint8_t> w_back(w_bytes.size());
    std::vector<float>   x_back(x_host.size());
    std::vector<int32_t> ids_back(ids_host.size());
    q.memcpy(w_back.data(), w_dev, w_back.size()).wait();
    q.memcpy(x_back.data(), x_dev, x_back.size() * sizeof(float)).wait();
    q.memcpy(ids_back.data(), ids_dev, ids_back.size() * sizeof(int32_t)).wait();

    const float * x = x_host.data() + c.src1_offset;
    const float * y = y_all.data() + GUARD;

    int    bad       = 0;
    int    zero_rows = 0;
    int    nan_rows  = 0;
    double max_err   = 0.0;
    auto   report    = [&](const char * what, int64_t t, int64_t s, int64_t r, double got, double ref) {
        if (bad < 3) {
            std::printf("  %s: %s t=%lld s=%lld row=%lld got=%.9g ref=%.9g\n", c.name, what, (long long) t,
                             (long long) s, (long long) r, got, ref);
        }
        ++bad;
    };
    for (int64_t t = 0; t < c.n_tokens; ++t) {
        for (int64_t s = 0; s < c.n_slots; ++s) {
            const int32_t id    = id_at(t, s);
            const float * y_row = y + (static_cast<size_t>(t) * dst_nb2 + static_cast<size_t>(s) * dst_nb1) / 4;
            const float * x_row =
                x + (static_cast<size_t>(t) * src1_nb2 + static_cast<size_t>(s % c.src1_rows) * src1_nb1) / 4;
            const bool null_entry = id >= 0 && id < c.n_experts && table[id] == nullptr;
            const bool nan_slot   = id != -1 && (id < 0 || id >= c.n_experts || null_entry);
            for (int64_t r = 0; r < c.rows; ++r) {
                const double got = y_row[r];
                if (id == -1) {
                    if (!(y_row[r] == 0.0f)) {
                        report("unused slot (-1) is not a zero row", t, s, r, got, 0.0);
                    }
                    continue;
                }
                if (nan_slot) {
                    if (!std::isnan(y_row[r])) {
                        report("bad slot is not a NaN row", t, s, r, got, NAN);
                    }
                    continue;
                }
                double ref = 0.0;
                double mag = 0.0;
                for (int64_t i = 0; i < c.k; ++i) {
                    const double p = static_cast<double>(w_dec[(id * c.rows + r) * c.k + i]) * x_row[i];
                    ref += p;
                    mag += std::fabs(p);
                }
                const double err = std::fabs(got - ref);
                max_err          = std::fmax(max_err, mag > 0.0 ? err / mag : err);
                if (!(err <= REL_TOL * mag + 1e-30)) {
                    report("mismatch", t, s, r, got, ref);
                }
            }
            zero_rows += id == -1 ? 1 : 0;
            nan_rows += nan_slot ? 1 : 0;
            for (int64_t r = c.rows; r < dst_row_elems; ++r) {
                if (y_row[r] != POISON) {
                    report("row padding was written", t, s, r, y_row[r], POISON);
                }
            }
        }
    }
    for (int64_t i = 0; i < GUARD; ++i) {
        if (y_all[i] != POISON || y_all[GUARD + dst_elems + i] != POISON) {
            std::printf("  %s: output guard element %lld was written\n", c.name, (long long) i);
            ++bad;
            break;
        }
    }
    if (std::memcmp(w_back.data(), w_bytes.data(), w_bytes.size()) != 0 ||
        std::memcmp(x_back.data(), x_host.data(), x_host.size() * sizeof(float)) != 0 ||
        std::memcmp(ids_back.data(), ids_host.data(), ids_host.size() * sizeof(int32_t)) != 0) {
        std::printf("  %s: an input was written\n", c.name);
        ++bad;
    }
    if (std::memcmp(y_again.data(), y_all.data(), y_all.size() * sizeof(float)) != 0) {
        std::printf("  %s: second launch differs from the first (non-deterministic)\n", c.name);
        ++bad;
    }
    if (c.bad_slots) {
        CHECK(zero_rows >= 1 && nan_rows >= 3, "%s: the bad-slot case did not exercise -1 and three NaN slots", c.name);
    }
    CHECK(bad == 0, "%s: %d bad outputs", c.name, bad);
    std::printf("  %-38s k=%-5lld rows=%-4lld experts=%-3lld slots=%lld tokens=%lld vec4=%d max_rel_err=%.3g %s\n",
                c.name, (long long) c.k, (long long) c.rows, (long long) c.n_experts, (long long) c.n_slots,
                (long long) c.n_tokens, ggml_sycl::moe_bf16_direct_vec4(shape) ? 1 : 0, max_err,
                bad == 0 ? "ok" : "BAD");

    sycl::free(w_dev, q);
    sycl::free(x_dev, q);
    sycl::free(ids_dev, q);
    sycl::free(y_dev, q);
    sycl::free(tbl_dev, q);
}

// The submitter's own refusals: nothing is enqueued and false comes back.
static void check_refusals(sycl::queue & q) {
    ggml_sycl::moe_bf16_direct_shape shape;
    shape.k         = 64;
    shape.rows      = 4;
    shape.n_experts = 2;
    shape.n_slots   = 2;
    shape.n_tokens  = 1;
    shape.src1_rows = 1;
    shape.w_nb1     = 128;
    shape.src1_nb1  = 256;
    shape.src1_nb2  = 256;
    shape.ids_nb0   = 4;
    shape.ids_nb1   = 8;
    shape.dst_nb1   = 16;
    shape.dst_nb2   = 32;

    // Pointers are never dereferenced on a refusal, so host addresses stand in.
    const void * table[2] = { nullptr, nullptr };
    float        x[64]    = {};
    int32_t      ids[2]   = {};
    float        y[8]     = {};
    sycl::event  ev;
    CHECK(!ggml_sycl::moe_bf16_direct_submit(q, nullptr, x, ids, y, shape, &ev), "a null table was accepted");
    CHECK(!ggml_sycl::moe_bf16_direct_submit(q, table, nullptr, ids, y, shape, &ev), "a null src1 was accepted");
    CHECK(!ggml_sycl::moe_bf16_direct_submit(q, table, x, nullptr, y, shape, &ev), "null ids were accepted");
    CHECK(!ggml_sycl::moe_bf16_direct_submit(q, table, x, ids, nullptr, shape, &ev), "a null dst was accepted");
    CHECK(!ggml_sycl::moe_bf16_direct_submit(q, table, x, ids, y, shape, nullptr), "a null event was accepted");

    ggml_sycl::moe_bf16_direct_shape bad = shape;
    bad.src1_rows                        = 0;
    CHECK(!ggml_sycl::moe_bf16_direct_submit(q, table, x, ids, y, bad, &ev), "src1_rows=0 was accepted");
    bad       = shape;
    bad.w_nb1 = 64;
    CHECK(!ggml_sycl::moe_bf16_direct_submit(q, table, x, ids, y, bad, &ev),
          "a weight row shorter than k was accepted");
    bad         = shape;
    bad.n_slots = 0;
    CHECK(!ggml_sycl::moe_bf16_direct_submit(q, table, x, ids, y, bad, &ev), "n_slots=0 was accepted");

    // The vec4 form: k a multiple of 4 and 8-byte weight rows, nothing else.
    bad = shape;
    CHECK(ggml_sycl::moe_bf16_direct_vec4(bad), "k=64 with 128-byte rows did not take vec4");
    bad.k     = 99;
    bad.w_nb1 = 198;
    CHECK(!ggml_sycl::moe_bf16_direct_vec4(bad), "k=99 took vec4");
    bad.k     = 100;
    bad.w_nb1 = 202;
    CHECK(!ggml_sycl::moe_bf16_direct_vec4(bad), "k=100 with a 202-byte row stride took vec4");
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    sycl::queue q{ sycl::default_selector_v };
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    check_refusals(q);

    const kernel_case cases[] = {
        // name                                   k     rows exp slt tok s1 wp xp xo yp is mis bad
        { "inkling gate_up k, broadcast row",     2304, 96, 8,  8, 2, 1, 0, 0, 0, 0, 1, -1, false },
        { "inkling down k, row per slot",         1152, 72, 8,  8, 3, 8, 0, 0, 0, 0, 1, -1, false },
        { "inkling k, one row",                   2304, 1,  8,  8, 1, 1, 0, 0, 0, 0, 1, -1, false },
        { "k=99 scalar form",                     99,   50, 16, 4, 5, 4, 0, 0, 0, 0, 1, -1, false },
        { "k=100 vec4, rows not a multiple of 4", 100,  37, 16, 4, 3, 1, 0, 0, 0, 0, 1, -1, false },
        { "k=3 below the lane count",             3,    5,  2,  2, 1, 1, 0, 0, 0, 0, 1, -1, false },
        { "x base misaligned (no float4)",        256,  20, 6,  3, 2, 3, 0, 0, 1, 0, 1, -1, false },
        { "weight row stride breaks vec4",        128,  12, 4,  4, 2, 1, 1, 0, 0, 0, 1, -1, false },
        { "padded strides everywhere",            64,   33, 5,  4, 3, 4, 0, 4, 0, 3, 2, -1, false },
        { "expert base off 8 bytes",              256,  16, 4,  4, 2, 1, 0, 0, 0, 0, 1, 2,  false },
        { "id -1 zero row, bad ids NaN rows",     64,   10, 6,  6, 2, 1, 0, 0, 0, 2, 1, -1, true  },
        { "bad slots, scalar form",               99,   9,  5,  4, 3, 4, 0, 0, 0, 0, 2, -1, true  },
    };
    for (const kernel_case & c : cases) {
        run_one(q, c);
    }

    if (g_failures != 0) {
        std::printf("FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
