// The pre-fill and the create_tensor() check of the lazy-range record
// (llama.cpp-5efe C1, design sycl-mmap-host-tier.md §3.1).
//
// Three names go through the pre-fill and the check: a declared lazy name, an
// undeclared flagged name, and a declared-but-unflagged name.
//   - the first gives one row, filled before any create_tensor() call, carrying the
//     attributes llama_model_lazy_tensor_attrs() returns for it, and its check passes;
//   - the other two abort, naming the tensor and the architecture, each in its own
//     process, scored on SIGABRT plus the text on stderr;
//   - the abort is mode-independent: the unflagged case aborts under OFF and under
//     AUTO at exactly 4 GiB, both of which leave the row ineligible and EAGER. A check
//     that compares the flag only against non-EAGER rows lets both through;
//   - value(name) equals the weight's (file, offset, size);
//   - with the taker, the early plan's filter includes an ineligible row (AUTO at
//     4 GiB, OFF) as the ordinary weight it is and excludes the eligible one.
//
// Host-only: no backend, device or model.

#include "lazy-range-fixture.h"

#include "../src/llama-model.h"

#include <sys/prctl.h>
#include <sys/wait.h>

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

struct check_args {
    const llama_lazy_range_record * rec;
    llm_arch                        arch;
    std::string                     name;
    bool                            flagged;
};

static void run_check(void * p) {
    const check_args * a = (const check_args *) p;
    llama_lazy_range_check_create(*a->rec, a->arch, a->name, a->flagged);
}

// runs the check in a child; true when it died of SIGABRT with both `name` and `arch` on stderr
static bool aborts_naming(const check_args & a, const char * arch_name) {
    int fds[2];
    if (pipe(fds) != 0) {
        return false;
    }
    fflush(nullptr);
    const pid_t pid = fork();
    if (pid == 0) {
        setenv("GGML_NO_BACKTRACE", "1", 1); // the abort path would otherwise attach a debugger
        prctl(PR_SET_DUMPABLE, 0);           // and a core dump, which apport would write for a deliberate abort
        close(fds[0]);
        dup2(fds[1], STDERR_FILENO);
        run_check((void *) &a);
        _exit(0);
    }
    close(fds[1]);
    std::string err;
    char        buf[512];
    ssize_t     n;
    while ((n = read(fds[0], buf, sizeof(buf))) > 0) {
        err.append(buf, (size_t) n);
    }
    close(fds[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT && err.find(a.name) != std::string::npos &&
           err.find(arch_name) != std::string::npos;
}

static bool passes(const check_args & a) {
    fflush(nullptr);
    const pid_t pid = fork();
    if (pid == 0) {
        run_check((void *) &a);
        _exit(0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

int main() {
    lazy_fixture fx;

    const llm_arch    arch     = LLM_ARCH_QWEN4EXP;
    const char *      arch_str = llm_arch_name(arch);
    const std::string lazy     = "per_layer_token_embd.weight"; // declared by qwen4exp
    const std::string unflagged = "lz_exact";                    // a tensor of the file the arch does not declare
    llama_hparams     hp;
    hp.n_layer_all = 8;
    hp.ple_n_heads = 16;
    hp.is_ple_impl.set(1);

    // the unflagged-but-declared case needs a row for a name the arch creates without the flag:
    // declare it by hand, as a drifted declaration would
    const std::vector<std::string> declared = { lazy, unflagged };

    for (llama_lazy_mode mode : { LLAMA_LAZY_MODE_OFF, LLAMA_LAZY_MODE_AUTO, LLAMA_LAZY_MODE_ON }) {
        llama_lazy_range_record rec;
        CHECK(!rec.filled(), "a record is filled before the pre-fill");
        llama_lazy_range_record_prefill(rec, arch, hp, declared, fx.finder(), mode, mode, false, true);

        // filled before any create_tensor() call, with the attributes the function returns
        CHECK(rec.filled() && rec.has(lazy), "mode %d: the declared name has no row after the pre-fill", (int) mode);
        const llama_lazy_tensor_attrs want = llama_model_lazy_tensor_attrs(arch, hp, lazy);
        CHECK(rec.attrs(lazy).rows_per_token == want.rows_per_token && rec.attrs(lazy).consumer_layer == want.consumer_layer,
              "the row's attributes are not llama_model_lazy_tensor_attrs()'s");
        const auto & w = fx.weights.at(lazy);
        const llama_lazy_range_value v = rec.value(lazy);
        CHECK(v.file_idx == w.idx && v.offs == w.offs && v.size == ggml_nbytes(w.tensor), "value() is not the weight's place");

        // 1. declared and flagged: passes
        CHECK(passes({ &rec, arch, lazy, true }), "mode %d: a declared, flagged tensor was refused", (int) mode);
        // 2. flagged, no row: aborts naming the tensor and the arch
        CHECK(aborts_naming({ &rec, arch, "not_declared.weight", true }, arch_str),
              "mode %d: an undeclared flagged tensor did not abort naming it and the arch", (int) mode);
        // 3. a row, no flag: aborts, whatever the mode (OFF and AUTO at 4 GiB leave the row EAGER)
        CHECK(aborts_naming({ &rec, arch, unflagged, false }, arch_str),
              "mode %d: a declared but unflagged tensor did not abort naming it and the arch", (int) mode);
        // and an undeclared, unflagged tensor is an ordinary weight
        CHECK(passes({ &rec, arch, "plain", false }), "mode %d: an ordinary weight was refused", (int) mode);
    }

    // the unflagged case is EAGER under OFF and under AUTO at 4 GiB, and still aborts
    {
        llama_lazy_range_record off;
        llama_lazy_range_record_prefill(off, arch, hp, declared, fx.finder(), LLAMA_LAZY_MODE_OFF, LLAMA_LAZY_MODE_OFF, false, true);
        CHECK(off.deferral(unflagged) == LLAMA_LAZY_DEFERRAL_EAGER, "OFF: lz_exact is not EAGER");
        llama_lazy_range_record at4;
        llama_lazy_range_record_prefill(at4, arch, hp, declared, fx.finder(), LLAMA_LAZY_MODE_AUTO, LLAMA_LAZY_MODE_AUTO, false, true);
        CHECK(at4.deferral(unflagged) == LLAMA_LAZY_DEFERRAL_EAGER, "AUTO at 4 GiB: lz_exact is not EAGER");
        CHECK(aborts_naming({ &off, arch, unflagged, false }, arch_str), "OFF: the EAGER row's drift went unnoticed");
        CHECK(aborts_naming({ &at4, arch, unflagged, false }, arch_str), "AUTO at 4 GiB: the EAGER row's drift went unnoticed");
    }

    // the early plan, with the taker: ineligible rows stay in, the eligible one is out
    for (llama_lazy_mode mode : { LLAMA_LAZY_MODE_OFF, LLAMA_LAZY_MODE_AUTO }) {
        llama_lazy_range_record rec;
        llama_lazy_range_record_prefill(rec, arch, hp, declared, fx.finder(), mode, mode, true, true);
        CHECK(!llama_lazy_early_plan_excludes(rec, unflagged),
              "taker, mode %d: the filter excluded an ineligible row (AUTO at 4 GiB / OFF)", (int) mode);
        CHECK(llama_lazy_early_plan_excludes(rec, lazy) == (mode == LLAMA_LAZY_MODE_AUTO),
              "taker, mode %d: the filter decision on the 5 GiB table is wrong", (int) mode);
    }

    if (n_failures != 0) {
        std::printf("FAIL: %d check(s)\n", n_failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
