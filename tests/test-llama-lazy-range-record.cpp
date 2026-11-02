// The per-model lazy-range record (llama.cpp-5efe C1, design sycl-mmap-host-tier.md §3.1).
//
// Drives the pre-fill and the record's queries over a model of five tensors in two
// files, three of them declared lazy, one of those a sparse table of 6 GiB (see
// lazy-range-fixture.h). Host-only: no backend, device or model.
//
// What it pins:
//   - rows equal the declared tensors' (file, offset, size) exactly, whatever the
//     mode, and a same-named tensor under an arch that does not declare it gets no row;
//   - the eligibility matrix over taker x mode_pre x mode_post x SUPPORTED:
//     lazy_eligible = llama_lazy_eligible(taker ? mode_pre : mode_post, size,
//     taker ? true : SUPPORTED), STORAGE iff eligible and taker, LAZY_READ iff
//     eligible and not taker, and `deferral != EAGER` iff `lazy_eligible`;
//   - the witness cell: AUTO requested, a device that cannot mmap and owns no
//     placement, taker -> the loader's AUTO resolver (llama_lazy_auto_enabled) turns
//     mode_post OFF, yet the sparse row is eligible and STORAGE and the early plan's
//     filter excludes it. A fill that reads mode_post for the taker (the post-fallback
//     fill) clears the bit and fails here;
//   - off the taker without mmap the bit is clear under every mode and the WARN
//     prints once per row that would have been lazy;
//   - the attributes are per model, from hparams: two qwen4exp hparams sets give each
//     record its own rows_per_token and consumer_layer, whichever was filled last;
//   - deferred_for_file holds exactly the LAZY_READ rows, storage_for_file exactly the
//     STORAGE rows, excluded_for_file their union;
//   - one INFO line per model: `lazy-range record: mode_pre=<m> mode_post=<m> taker=<0|1>`,
//     printed even when the model has no declared name;
//   - a second write to a filled record aborts.

#include "../src/llama-lazy-mode.h"
#include "../src/llama-model.h"
#include "lazy-range-fixture.h"

#include <sys/prctl.h>
#include <sys/wait.h>

#include <algorithm>
#include <csignal>
#include <cstdio>
#include <string>
#include <vector>

static int n_failures = 0;

#define CHECK(cond, ...)                                     \
    do {                                                     \
        if (!(cond)) {                                       \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                        \
            std::printf("\n");                               \
            n_failures++;                                    \
        }                                                    \
    } while (0)

static std::vector<std::string> g_log;

static void capture_log(enum ggml_log_level, const char * text, void *) {
    g_log.emplace_back(text);
}

static size_t count_log(const char * needle) {
    size_t n = 0;
    for (const std::string & line : g_log) {
        if (line.find(needle) != std::string::npos) {
            n++;
        }
    }
    return n;
}

static const char * mode_name(llama_lazy_mode m) {
    return m == LLAMA_LAZY_MODE_OFF ? "off" : m == LLAMA_LAZY_MODE_AUTO ? "auto" : "on";
}

static const llama_lazy_mode k_modes[] = { LLAMA_LAZY_MODE_OFF, LLAMA_LAZY_MODE_AUTO, LLAMA_LAZY_MODE_ON };

// qwen4exp-shaped hparams: `n_heads` PLE heads, the PLE layer at `ple_layer`
static llama_hparams qwen_hparams(uint32_t n_heads, uint32_t ple_layer) {
    llama_hparams hp;
    hp.n_layer_all = 8;
    hp.ple_n_heads = n_heads;
    hp.is_ple_impl.reset();
    hp.is_ple_impl.set(ple_layer);
    return hp;
}

// runs `fn` in a child and reports whether it died of SIGABRT
static bool aborts(void (*fn)(void *), void * arg) {
    fflush(nullptr);
    const pid_t pid = fork();
    if (pid == 0) {
        setenv("GGML_NO_BACKTRACE", "1", 1);  // the abort path would otherwise attach a debugger
        prctl(PR_SET_DUMPABLE, 0);            // and a core dump, which apport would write for a deliberate abort
        freopen("/dev/null", "w", stderr);
        fn(arg);
        _exit(0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}

static void fill_twice(void * arg) {
    llama_lazy_range_record * rec = (llama_lazy_range_record *) arg;
    rec->fill({});
}

int main() {
    llama_log_set(capture_log, nullptr);

    lazy_fixture                   fx;
    const std::vector<std::string> declared = { "lz_plus1", "lz_exact", "lz_sparse" };
    const llama_hparams            hp       = qwen_hparams(16, 1);

    // ---- rows equal the weights' (file, offset, size), whatever the mode ----
    for (llama_lazy_mode mode : k_modes) {
        llama_lazy_range_record rec;
        llama_lazy_range_record_prefill(rec, LLM_ARCH_LLAMA, hp, declared, fx.finder(), mode, mode, false, true);
        CHECK(rec.rows().size() == 3, "mode %s: %zu rows, expected 3", mode_name(mode), rec.rows().size());
        for (const std::string & name : declared) {
            const auto & w = fx.weights.at(name);
            CHECK(rec.has(name), "mode %s: no row for %s", mode_name(mode), name.c_str());
            if (!rec.has(name)) {
                continue;
            }
            const llama_lazy_range_value v = rec.value(name);
            CHECK(v.file_idx == w.idx && v.offs == w.offs && v.size == ggml_nbytes(w.tensor),
                  "mode %s: %s value (%u, %zu, %zu) != weight (%u, %zu, %zu)", mode_name(mode), name.c_str(),
                  v.file_idx, v.offs, v.size, (unsigned) w.idx, w.offs, ggml_nbytes(w.tensor));
        }
        CHECK(!rec.has("plain"), "an undeclared tensor got a row");
    }

    // ---- the (arch, name) key: gemma3n creates the same name unflagged ----
    {
        llama_lazy_range_record qwen;
        llama_lazy_range_record_prefill(qwen, LLM_ARCH_QWEN4EXP, hp, llama_model_lazy_tensor_names(LLM_ARCH_QWEN4EXP),
                                        fx.finder(), LLAMA_LAZY_MODE_ON, LLAMA_LAZY_MODE_ON, false, true);
        CHECK(qwen.has("per_layer_token_embd.weight"), "qwen4exp declares per_layer_token_embd.weight and got no row");

        llama_lazy_range_record g3n;
        llama_lazy_range_record_prefill(g3n, LLM_ARCH_GEMMA3N, hp, llama_model_lazy_tensor_names(LLM_ARCH_GEMMA3N),
                                        fx.finder(), LLAMA_LAZY_MODE_ON, LLAMA_LAZY_MODE_ON, false, true);
        CHECK(!g3n.has("per_layer_token_embd.weight"), "gemma3n got a row for the same-named, unflagged tensor");
        CHECK(g3n.rows().empty(), "gemma3n declares nothing, yet %zu rows", g3n.rows().size());
    }

    // ---- the eligibility matrix ----
    for (int taker = 0; taker <= 1; ++taker) {
        for (llama_lazy_mode pre : k_modes) {
            for (llama_lazy_mode post : k_modes) {
                for (int supported = 0; supported <= 1; ++supported) {
                    llama_lazy_range_record rec;
                    llama_lazy_range_record_prefill(rec, LLM_ARCH_LLAMA, hp, declared, fx.finder(), pre, post,
                                                    taker != 0, supported != 0);
                    for (const llama_lazy_range_row & row : rec.rows()) {
                        // T6a's function, called and not restated
                        const bool expected =
                            llama_lazy_eligible(taker ? pre : post, row.value.size, taker ? true : supported != 0);
                        const llama_lazy_deferral expected_deferral = !expected ? LLAMA_LAZY_DEFERRAL_EAGER :
                                                                      taker     ? LLAMA_LAZY_DEFERRAL_STORAGE :
                                                                                  LLAMA_LAZY_DEFERRAL_LAZY_READ;
                        CHECK(row.lazy_eligible == expected,
                              "taker=%d pre=%s post=%s supported=%d %s: eligible=%d expected %d", taker, mode_name(pre),
                              mode_name(post), supported, row.name.c_str(), (int) row.lazy_eligible, (int) expected);
                        CHECK(row.deferral == expected_deferral,
                              "taker=%d pre=%s post=%s supported=%d %s: deferral=%s expected %s", taker, mode_name(pre),
                              mode_name(post), supported, row.name.c_str(), llama_lazy_deferral_name(row.deferral),
                              llama_lazy_deferral_name(expected_deferral));
                        CHECK((row.deferral != LLAMA_LAZY_DEFERRAL_EAGER) == row.lazy_eligible,
                              "taker=%d pre=%s post=%s supported=%d %s: deferral != EAGER is not the bit", taker,
                              mode_name(pre), mode_name(post), supported, row.name.c_str());
                        CHECK(!(!taker && row.deferral == LLAMA_LAZY_DEFERRAL_STORAGE),
                              "-ngl 0 shape: a non-taker row is STORAGE");
                    }
                }
            }
        }
    }

    // ---- the witness cell: AUTO, a device without mmap that owns no placement, the taker ----
    {
        // mode_post as the loader's AUTO block computes it, through the one resolver it calls
        // (tests/test-lazy-mode-resolve.cpp drives that resolver over its full truth table)
        const llama_lazy_device_caps no_mmap   = { /*mmap_support=*/false, /*planner_owns_placement=*/false };
        const llama_lazy_mode        mode_pre  = LLAMA_LAZY_MODE_AUTO;
        const llama_lazy_mode        mode_post = llama_lazy_auto_enabled(&no_mmap, 1) ? mode_pre : LLAMA_LAZY_MODE_OFF;
        CHECK(mode_post == LLAMA_LAZY_MODE_OFF, "the resolver did not turn AUTO into OFF: %s", mode_name(mode_post));

        llama_lazy_range_record rec;
        llama_lazy_range_record_prefill(rec, LLM_ARCH_LLAMA, hp, declared, fx.finder(), mode_pre, mode_post, true,
                                        true);
        CHECK(rec.eligible("lz_sparse"), "the witness cell: the sparse row's bit is clear (post-fallback fill)");
        CHECK(rec.deferral("lz_sparse") == LLAMA_LAZY_DEFERRAL_STORAGE, "the witness cell: the sparse row is %s",
              llama_lazy_deferral_name(rec.deferral("lz_sparse")));
        CHECK(llama_lazy_early_plan_excludes(rec, "lz_sparse"),
              "the witness cell: the early plan's filter includes the table");

        // ineligible rows are the ordinary weights they are: AUTO at exactly 4 GiB
        CHECK(!rec.eligible("lz_exact") && !llama_lazy_early_plan_excludes(rec, "lz_exact"),
              "AUTO at 4 GiB must stay an ordinary weight in the early plan");
        CHECK(!llama_lazy_early_plan_excludes(rec, "plain"), "an undeclared tensor was excluded");
    }

    // ---- off the taker, no mmap: the bit is clear under every mode; WARN per row that would have been lazy ----
    for (llama_lazy_mode mode : k_modes) {
        g_log.clear();
        llama_lazy_range_record rec;
        llama_lazy_range_record_prefill(rec, LLM_ARCH_LLAMA, hp, declared, fx.finder(), mode, mode, false, false);
        for (const llama_lazy_range_row & row : rec.rows()) {
            CHECK(!row.lazy_eligible && row.deferral == LLAMA_LAZY_DEFERRAL_EAGER, "no mmap, mode %s: %s is lazy",
                  mode_name(mode), row.name.c_str());
        }
        const size_t expected = mode == LLAMA_LAZY_MODE_OFF ? 0 : mode == LLAMA_LAZY_MODE_ON ? 3 : 2;
        CHECK(count_log("mmap is not available") == expected, "no mmap, mode %s: %zu warnings, expected %zu",
              mode_name(mode), count_log("mmap is not available"), expected);
    }

    // ---- the attributes are per model, from hparams ----
    {
        const std::vector<std::string> names = llama_model_lazy_tensor_names(LLM_ARCH_QWEN4EXP);
        llama_lazy_range_record        a, b;
        llama_lazy_range_record_prefill(a, LLM_ARCH_QWEN4EXP, qwen_hparams(16, 1), names, fx.finder(),
                                        LLAMA_LAZY_MODE_ON, LLAMA_LAZY_MODE_ON, true, true);
        llama_lazy_range_record_prefill(b, LLM_ARCH_QWEN4EXP, qwen_hparams(8, 5), names, fx.finder(),
                                        LLAMA_LAZY_MODE_ON, LLAMA_LAZY_MODE_ON, true, true);
        const std::string t = "per_layer_token_embd.weight";
        CHECK(a.attrs(t).rows_per_token == 16 && a.attrs(t).consumer_layer == 1,
              "record a read (%u, %d), expected (16, 1) after record b was filled", a.attrs(t).rows_per_token,
              a.attrs(t).consumer_layer);
        CHECK(b.attrs(t).rows_per_token == 8 && b.attrs(t).consumer_layer == 5,
              "record b read (%u, %d), expected (8, 5)", b.attrs(t).rows_per_token, b.attrs(t).consumer_layer);

        // not a literal 1; more than one PLE layer is "not known"
        llama_hparams two = qwen_hparams(16, 1);
        two.is_ple_impl.set(5);
        CHECK(llama_model_lazy_tensor_attrs(LLM_ARCH_QWEN4EXP, two, t).consumer_layer == -1,
              "two PLE layers must read as not known");
        const llama_lazy_tensor_attrs g4 = llama_model_lazy_tensor_attrs(LLM_ARCH_GEMMA4, hp, t);
        CHECK(g4.rows_per_token == 1 && g4.consumer_layer == 0, "gemma4 attrs (%u, %d), expected (1, 0)",
              g4.rows_per_token, g4.consumer_layer);
        const llama_lazy_tensor_attrs none = llama_model_lazy_tensor_attrs(LLM_ARCH_LLAMA, hp, t);
        CHECK(none.rows_per_token == 0 && none.consumer_layer == -1, "an undeclared arch reads (%u, %d)",
              none.rows_per_token, none.consumer_layer);
    }

    // ---- deferred / storage / excluded ----
    {
        const auto & plus1  = fx.weights.at("lz_plus1");
        const auto & exact  = fx.weights.at("lz_exact");
        const auto & sparse = fx.weights.at("lz_sparse");

        llama_lazy_range_record lazy;
        llama_lazy_range_record_prefill(lazy, LLM_ARCH_LLAMA, hp, declared, fx.finder(), LLAMA_LAZY_MODE_ON,
                                        LLAMA_LAZY_MODE_ON, false, true);
        llama_lazy_range_record storage;
        llama_lazy_range_record_prefill(storage, LLM_ARCH_LLAMA, hp, declared, fx.finder(), LLAMA_LAZY_MODE_ON,
                                        LLAMA_LAZY_MODE_ON, true, true);

        auto range_of = [](const lazy_fixture::weights_t::mapped_type & w) {
            return std::make_pair(w.offs, w.offs + ggml_nbytes(w.tensor));
        };
        llama_mmap::ranges file0 = { range_of(plus1), range_of(exact) };
        std::sort(file0.begin(), file0.end());
        const llama_mmap::ranges file1 = { range_of(sparse) };

        CHECK(lazy.deferred_for_file(0) == file0, "LAZY_READ rows of file 0 are not the declared ranges");
        CHECK(lazy.deferred_for_file(1) == file1, "LAZY_READ rows of file 1 are not the declared range");
        CHECK(lazy.storage_for_file(0).empty() && lazy.storage_for_file(1).empty(),
              "a LAZY_READ model has STORAGE ranges");
        CHECK(lazy.excluded_for_file(0) == file0 && lazy.excluded_for_file(1) == file1,
              "excluded != deferred when no row is STORAGE");

        CHECK(storage.deferred_for_file(0).empty() && storage.deferred_for_file(1).empty(),
              "STORAGE rows leaked into deferred_for_file (lazy_read would map them)");
        CHECK(storage.storage_for_file(0) == file0 && storage.storage_for_file(1) == file1,
              "storage_for_file is not the STORAGE ranges");
        CHECK(storage.excluded_for_file(0) == file0 && storage.excluded_for_file(1) == file1,
              "excluded_for_file is not the union");
    }

    // ---- lazy_read reads the record: add() accepts exactly the LAZY_READ rows, under every mode it holds ----
    {
        struct cell {
            const char *    label;
            llama_lazy_mode mode;  // what the record was filled with
            bool            taker;
        };

        const cell cells[] = {
            { "ON, off the taker (LAZY_READ rows)",          LLAMA_LAZY_MODE_ON,   false },
            { "ON, the taker (STORAGE rows)",                LLAMA_LAZY_MODE_ON,   true  },
            { "AUTO, off the taker (4 GiB row stays EAGER)", LLAMA_LAZY_MODE_AUTO, false },
            { "OFF, off the taker (EAGER rows)",             LLAMA_LAZY_MODE_OFF,  false },
        };
        for (const cell & c : cells) {
            llama_lazy_range_record rec;
            llama_lazy_range_record_prefill(rec, LLM_ARCH_LLAMA, hp, declared, fx.finder(), c.mode, c.mode, c.taker,
                                            true);

            // lazy_read's own mode must not decide anything: upstream's add() read it
            for (llama_lazy_mode held : k_modes) {
                // the fit pass (no_alloc): no weight, so nothing is mapped, but the decision is the same
                llama_model_loader::lazy_read fit;
                fit.mode   = held;
                fit.record = &rec;
                // a real load: the weight is given, so the name becomes a member
                llama_model_loader::lazy_read real;
                real.mode   = held;
                real.record = &rec;

                size_t n_lazy_read = 0;
                for (const std::string & name : declared) {
                    const bool    want = rec.deferral(name) == LLAMA_LAZY_DEFERRAL_LAZY_READ;
                    ggml_tensor * t    = fx.weights.at(name).tensor;
                    n_lazy_read += want ? 1 : 0;

                    const bool fit_got  = fit.add(name, t, nullptr);
                    const bool real_got = real.add(name, t, &fx.weights.at(name));
                    CHECK(fit_got == want, "%s, held mode %s, fit pass: add(%s) = %d, expected %d", c.label,
                          mode_name(held), name.c_str(), (int) fit_got, (int) want);
                    CHECK(real_got == want, "%s, held mode %s: add(%s) = %d, expected %d", c.label, mode_name(held),
                          name.c_str(), (int) real_got, (int) want);
                    CHECK(real.has(t) == want, "%s: membership of %s is %d, expected %d", c.label, name.c_str(),
                          (int) real.has(t), (int) want);
                    CHECK(!fit.has(t), "%s: the fit pass (no weight) recorded %s as a member", c.label, name.c_str());
                }
                CHECK(!fit.any(), "%s: the fit pass maps something (any() is true)", c.label);
                CHECK(real.any() == (n_lazy_read > 0), "%s: any() = %d with %zu LAZY_READ rows", c.label,
                      (int) real.any(), n_lazy_read);
            }
        }
    }

    // ---- one INFO line per model, even without a declared name ----
    {
        g_log.clear();
        llama_lazy_range_record rec;
        llama_lazy_range_record_prefill(rec, LLM_ARCH_LLAMA, hp, {}, fx.finder(), LLAMA_LAZY_MODE_AUTO,
                                        LLAMA_LAZY_MODE_OFF, true, true);
        CHECK(count_log("lazy-range record: mode_pre=auto mode_post=off taker=1") == 1,
              "the per-model line printed %zu times", count_log("lazy-range record: mode_pre="));
        CHECK(rec.rows().empty() && rec.filled(),
              "a model with no declared name must still have a filled, empty record");
    }

    // ---- immutability ----
    {
        llama_lazy_range_record rec;
        llama_lazy_range_record_prefill(rec, LLM_ARCH_LLAMA, hp, declared, fx.finder(), LLAMA_LAZY_MODE_ON,
                                        LLAMA_LAZY_MODE_ON, false, true);
        CHECK(aborts(fill_twice, &rec), "a second write to a filled record did not abort");
    }

    if (n_failures != 0) {
        std::printf("FAIL: %d check(s)\n", n_failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
