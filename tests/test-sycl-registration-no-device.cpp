// Regression gate for llama.cpp-1lrh.
//
// ggml_backend_reg_by_name("SYCL") is a discovery/registration query -- exactly
// what a caller does when it does not yet know whether the SYCL backend is
// available. Until this fix, dpct::dev_mgr::dev_mgr() (helper.hpp) called
// GGML_ABORT("No SYCL devices available") the moment platform enumeration
// found zero devices, so a registration query on a device-less environment
// SIGABRTed the whole process (rc=134) instead of answering "zero devices".
//
// This test forces that zero-device condition without touching any real GPU:
// ONEAPI_DEVICE_SELECTOR=level_zero:99 selects a device INDEX that cannot
// exist (this host never has 100 Level-Zero devices), which the loader
// accepts as syntactically valid and resolves to "zero devices" rather than a
// selector parse error -- verified empirically for llama.cpp-1lrh by
// comparing against the pre-fix build, where this exact selector reproduced
// the abort at helper.hpp:1186 through the real
// ggml_backend_reg_by_name -> ggml_backend_sycl_reg -> ggml_sycl_init ->
// dpct::dev_mgr::dev_mgr() call chain.
//
// The fix draws a registration-vs-use split, and this test asserts BOTH
// halves, each in its own forked child so the two are independent and a
// crash in one cannot corrupt the other's result:
//
//   1. Registration must not abort: ggml_backend_reg_by_name("SYCL") returns
//      a non-null registry reporting exactly 0 devices.
//   2. A genuine attempt to USE a device must still fail LOUDLY, never
//      silently succeed. Measured for llama.cpp-1lrh: with 0 registered
//      devices, ggml_backend_sycl_init(0) reaches ggml_backend_reg_dev_get ->
//      ggml_backend_sycl_reg_get_device (ggml-sycl.cpp ~103790), whose
//      GGML_ASSERT(index < ctx->devices.size()) is unconditional (unlike a
//      bare assert(), it is NOT compiled out under -DNDEBUG) and calls
//      GGML_ABORT. That reg_dev_get call happens while building the
//      ggml_backend struct literal inside ggml_backend_sycl_init's try{}
//      block (ggml-sycl.cpp ~104138-104199), but a hard abort is a signal,
//      not a C++ exception, so the surrounding catch(std::exception&) never
//      gets a chance to run -- the process dies right there. The catch
//      block's own "backend construction failed" ERROR-log-and-nullptr-return
//      path is real (it is what turns a genuine std::exception, e.g. a
//      dev_mgr check_id() throw reached some other way, into a graceful
//      failure) but is NOT the path this exact call takes today. This test
//      accepts either outcome, since both satisfy "fails loudly, not
//      silently": (a) SIGABRT with the GGML_ASSERT text on stderr, or (b) a
//      clean exit whose stderr carries "backend construction failed". A
//      clean exit with NEITHER string present -- e.g. ggml_backend_sycl_init
//      returning a non-null backend -- is treated as a silent success and
//      fails the test.
//
// The use-path check runs in a forked child BEFORE that child does any SYCL
// work, following two in-repo precedents for fork-before-SYCL-init harnesses:
// ggml/src/ggml-sycl/tests/test-mem-handle-byte-contract.cpp's expect_fatal()
// (fork, capture child stderr via a pipe, waitpid, assert
// WIFSIGNALED+WTERMSIG==SIGABRT plus a specific stderr string) and
// tests/test-sycl-fattn-onednn-descriptors.cpp's main() (fork before any SYCL
// call so the parent never enumerates a device itself). An earlier version of
// this file's comment claimed a fork()+waitpid() harness "was considered and
// rejected as unsafe to add after a SYCL runtime has begun platform
// enumeration" -- that was false (both precedents above already do exactly
// this in this repo) and also moot, because every fork here happens strictly
// before the child's first SYCL/dev_mgr call, matching both precedents.
//
// This test does not use test-skip.h's LLAMA_TEST_EXIT_SKIP path as its
// normal outcome: with ONEAPI_DEVICE_SELECTOR pinned by the ENVIRONMENT
// property below, the zero-device condition is forced on every host, so the
// property under test is exercised unconditionally. SKIP_RETURN_CODE is
// still registered (matching the family convention -- see
// test-sycl-env-report.cpp) as a defensive exit for the one case this test
// cannot itself repair: a host whose SYCL loader does not treat an
// out-of-range device index as "zero devices" the way this one does.

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "test-skip.h"

#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// Child exit codes for the registration half, distinguishing "the property
// does not apply on this host" (skip) from pass/fail.
static constexpr int REG_CHILD_PASS = 0;
static constexpr int REG_CHILD_FAIL = 1;
static constexpr int REG_CHILD_SKIP = 2;

// Runs `child_body` in a forked child with its stderr captured through a
// pipe, then waitpid's. Returns false (and leaves *status/*captured unset on
// failure) only if fork()/pipe()/waitpid() themselves fail -- never as a
// verdict on the child's behaviour, which the caller inspects via *status and
// *captured.
static bool run_forked(void (*child_body)(), int * out_status, std::string * out_captured) {
    int pipefd[2];
    if (pipe(pipefd) != 0) {
        std::fprintf(stderr, "run_forked: pipe() failed: %s\n", strerror(errno));
        return false;
    }
    const pid_t child = fork();
    if (child == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);
        child_body();
        _exit(0);
    }
    if (child < 0) {
        std::fprintf(stderr, "run_forked: fork() failed: %s\n", strerror(errno));
        close(pipefd[0]);
        close(pipefd[1]);
        return false;
    }
    close(pipefd[1]);
    std::string captured;
    char        buf[4096];
    ssize_t     n;
    while ((n = read(pipefd[0], buf, sizeof(buf))) > 0) {
        captured.append(buf, static_cast<size_t>(n));
    }
    close(pipefd[0]);

    int status = 0;
    if (waitpid(child, &status, 0) != child) {
        std::fprintf(stderr, "run_forked: waitpid failed\n");
        return false;
    }
    *out_status   = status;
    *out_captured = captured;
    return true;
}

// Child body for the registration half: reaching past
// ggml_backend_reg_by_name("SYCL") at all (rather than SIGABRT) is the proof
// that the pre-fix dev_mgr() constructor no longer aborts from inside this
// call.
static void registration_child() {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("SYCL");
    if (reg == nullptr) {
        std::fprintf(stderr, "registration_child: ggml_backend_reg_by_name(\"SYCL\") returned null\n");
        _exit(REG_CHILD_FAIL);
    }
    size_t dev_count = ggml_backend_reg_dev_count(reg);
    if (dev_count != 0) {
        // The chosen mechanism did not yield zero devices on this host --
        // this test cannot exercise the property it exists to check.
        std::fprintf(stderr,
                     "registration_child: SKIP: ONEAPI_DEVICE_SELECTOR=level_zero:99 reported %zu "
                     "device(s) on this host; expected 0 -- cannot exercise the zero-device "
                     "registration path\n",
                     dev_count);
        _exit(REG_CHILD_SKIP);
    }
    std::fprintf(stderr, "registration_child: PASS: registration reported 0 devices without aborting\n");
    _exit(REG_CHILD_PASS);
}

// Child body for the use half: a genuine attempt to use device 0 when 0
// devices are registered. See the file header for why both an abort and a
// graceful null return are accepted, and why a non-null backend is not.
static void use_attempt_child() {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("SYCL");
    if (reg == nullptr || ggml_backend_reg_dev_count(reg) != 0) {
        std::fprintf(stderr, "use_attempt_child: preconditions not met (reg=%p)\n", (void *) reg);
        _exit(1);
    }
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (backend != nullptr) {
        std::fprintf(stderr, "use_attempt_child: ggml_backend_sycl_init(0) SILENTLY SUCCEEDED with 0 devices\n");
        ggml_backend_free(backend);
        _exit(1);
    }
    std::fprintf(stderr, "use_attempt_child: ggml_backend_sycl_init(0) returned null, no abort\n");
    _exit(0);
}

static bool expect_use_fails() {
    int         status = 0;
    std::string captured;
    if (!run_forked(use_attempt_child, &status, &captured)) {
        return false;
    }

    bool signaled_abort  = WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
    bool saw_assert_text = captured.find("GGML_ASSERT(index < ctx->devices.size())") != std::string::npos;
    bool clean_null_init = WIFEXITED(status) && WEXITSTATUS(status) == 0 &&
                           captured.find("backend construction failed") != std::string::npos;

    bool ok = (signaled_abort && saw_assert_text) || clean_null_init;
    if (!ok) {
        std::fprintf(stderr,
                     "expect_use_fails: signaled_abort=%d saw_assert_text=%d clean_null_init=%d status=%d "
                     "captured=[%s]\n",
                     signaled_abort, saw_assert_text, clean_null_init, status, captured.c_str());
    } else {
        std::fprintf(stderr, "PASS: a genuine use attempt failed loudly (signaled_abort=%d clean_null_init=%d)\n",
                     signaled_abort, clean_null_init);
    }
    return ok;
}

int main() {
    int         reg_status = 0;
    std::string reg_captured;
    if (!run_forked(registration_child, &reg_status, &reg_captured)) {
        std::fprintf(stderr, "FAIL: could not fork the registration child\n");
        return 1;
    }
    std::fputs(reg_captured.c_str(), stderr);

    if (WIFSIGNALED(reg_status)) {
        std::fprintf(stderr, "FAIL: registration child was killed by signal %d (expected a clean exit)\n",
                     WTERMSIG(reg_status));
        return 1;
    }
    int reg_exit = WIFEXITED(reg_status) ? WEXITSTATUS(reg_status) : -1;
    if (reg_exit == REG_CHILD_SKIP) {
        return LLAMA_TEST_EXIT_SKIP;
    }
    if (reg_exit != REG_CHILD_PASS) {
        std::fprintf(stderr, "FAIL: registration child exited %d\n", reg_exit);
        return 1;
    }

    if (!expect_use_fails()) {
        std::fprintf(stderr, "FAIL: use-path did not fail loudly (see expect_use_fails output above)\n");
        return 1;
    }

    return 0;
}
