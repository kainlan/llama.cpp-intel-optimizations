// A process the backend ends after a fatal error must not run its exit handlers or static destructors: after a
// device loss the Level Zero library's own destructor does not return. Each mode runs this binary again as a child
// that plants an atexit handler and a static destructor, and checks from outside which of them ran.

#include "common.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

constexpr const char * k_atexit_marker = "TEST-ATEXIT-RAN";
constexpr const char * k_dtor_marker   = "TEST-STATIC-DTOR-RAN";

void atexit_marker() {
    std::fprintf(stderr, "%s\n", k_atexit_marker);
    std::fflush(stderr);
}

struct dtor_marker {
    ~dtor_marker() {
        std::fprintf(stderr, "%s\n", k_dtor_marker);
        std::fflush(stderr);
    }
};

void plant_markers() {
    static dtor_marker marker;
    std::atexit(atexit_marker);
}

struct child_result {
    bool        exited  = false;
    int         status  = -1;
    int         signal  = 0;
    double      seconds = 0.0;
    std::string err;
};

child_result run_child(const char * self, const char * mode) {
    int fds[2];
    if (pipe(fds) != 0) {
        std::perror("pipe");
        std::exit(1);
    }
    const auto  t0  = std::chrono::steady_clock::now();
    const pid_t pid = fork();
    if (pid < 0) {
        std::perror("fork");
        std::exit(1);
    }
    if (pid == 0) {
        dup2(fds[1], STDERR_FILENO);
        close(fds[0]);
        close(fds[1]);
        execl(self, self, "--child", mode, static_cast<char *>(nullptr));
        _exit(127);
    }
    close(fds[1]);
    child_result r;
    char         buf[4096];
    ssize_t      n = 0;
    while ((n = read(fds[0], buf, sizeof(buf))) > 0) {
        r.err.append(buf, static_cast<size_t>(n));
    }
    close(fds[0]);
    int st = 0;
    waitpid(pid, &st, 0);
    r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    r.exited  = WIFEXITED(st);
    r.status  = r.exited ? WEXITSTATUS(st) : -1;
    r.signal  = WIFSIGNALED(st) ? WTERMSIG(st) : 0;
    return r;
}

bool has(const child_result & r, const char * needle) {
    return r.err.find(needle) != std::string::npos;
}

int fail(const char * mode, const child_result & r, const char * why) {
    std::fprintf(stderr, "FAIL mode=%s: %s (exited=%d status=%d signal=%d seconds=%.1f)\n--- child stderr ---\n%s\n",
                 mode, why, r.exited ? 1 : 0, r.status, r.signal, r.seconds, r.err.c_str());
    return 1;
}

// The fast exit ran: the given status, its banner, and neither marker.
int check_fast_exit(const char * mode, const child_result & r, int want_status) {
    if (!r.exited || r.status != want_status) {
        return fail(mode, r, "wrong exit status");
    }
    if (!has(r, "[SYCL-FATAL]")) {
        return fail(mode, r, "no [SYCL-FATAL] line");
    }
    if (has(r, k_atexit_marker) || has(r, k_dtor_marker)) {
        return fail(mode, r, "an exit handler or a static destructor ran");
    }
    if (r.seconds > 30.0) {
        return fail(mode, r, "the child took over 30 s to end");
    }
    return 0;
}

int child_main(const char * mode) {
    if (std::strcmp(mode, "helper") == 0) {
        plant_markers();
        ggml_sycl_fatal_exit(GGML_SYCL_EXIT_DEVICE_LOST, "test: helper mode");
    }
    if (std::strcmp(mode, "control") == 0) {
        plant_markers();
        std::exit(GGML_SYCL_EXIT_DEVICE_LOST);
    }
    std::fprintf(stderr, "unknown child mode %s\n", mode);
    return 2;
}

int parent_main(const char * self, const char * mode) {
    if (std::strcmp(mode, "helper") == 0) {
        const child_result r = run_child(self, mode);
        if (!has(r, "[SYCL-FATAL] test: helper mode")) {
            return fail(mode, r, "the message is not on the [SYCL-FATAL] line");
        }
        return check_fast_exit(mode, r, GGML_SYCL_EXIT_DEVICE_LOST);
    }
    if (std::strcmp(mode, "control") == 0) {
        const child_result r = run_child(self, mode);
        if (!r.exited || r.status != GGML_SYCL_EXIT_DEVICE_LOST || !has(r, k_atexit_marker) || !has(r, k_dtor_marker)) {
            return fail(mode, r, "std::exit did not run both markers, so their absence proves nothing");
        }
        return 0;
    }
    if (std::strcmp(mode, "classify") == 0) {
        const char * lost = "level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)";
        const char * oom  = "level_zero backend failed with error: 39 (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)";
        const bool   ok   = ggml_sycl_what_is_device_lost(lost) && !ggml_sycl_what_is_device_lost(oom) &&
                        !ggml_sycl_what_is_device_lost(nullptr) &&
                        ggml_sycl_fatal_exit_code(lost) == GGML_SYCL_EXIT_DEVICE_LOST &&
                        ggml_sycl_fatal_exit_code(oom) == 1 && ggml_sycl_fatal_exit_code(nullptr) == 1;
        if (!ok) {
            std::fprintf(stderr, "FAIL mode=classify\n");
            return 1;
        }
        return 0;
    }
    std::fprintf(stderr, "usage: %s helper|control|classify\n", self);
    return 2;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--child") == 0) {
        return child_main(argv[2]);
    }
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s helper|control|classify\n", argv[0]);
        return 2;
    }
    const int rc = parent_main(argv[0], argv[1]);
    if (rc == 0) {
        std::fprintf(stderr, "PASS mode=%s\n", argv[1]);
    }
    return rc;
}
