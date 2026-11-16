// A process the backend ends after a fatal error must not run its exit handlers or static destructors: after a
// device loss the Level Zero library's own destructor does not return. Each mode runs this binary again as a child
// that plants an atexit handler and a static destructor, and checks from outside which of them ran.

#include "common.hpp"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "ggml-sycl.h"
#include "ggml.h"

#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr const char * k_atexit_marker = "TEST-ATEXIT-RAN";
constexpr const char * k_dtor_marker   = "TEST-STATIC-DTOR-RAN";

// A child still running after this long is hung: the parent kills it and reports what it printed, because a hang is
// the failure the fast exit exists to prevent and must not hang the test with it.
constexpr int k_child_limit_s = 30;

// How long the parent keeps reading the pipe and waiting for the child after the kill. Bounded because a child the
// kernel holds in an uninterruptible wait cannot be reaped, and a process the child started can still hold the pipe.
constexpr int k_child_grace_s = 5;

// How often the parent looks at the pipe and at the child while it waits.
constexpr int k_child_slice_ms = 10;

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
    bool        exited    = false;
    int         status    = -1;
    int         signal    = 0;
    bool        timed_out = false;
    bool        reaped    = false;
    bool        pipe_open = false;  // something still held the child's output pipe when the parent stopped reading
    double      seconds   = 0.0;    // until the child was reaped, or until the parent stopped if it never was
    std::string out;                // the child's stdout and stderr, which share one pipe
};

int ms_until(std::chrono::steady_clock::time_point t) {
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(t - std::chrono::steady_clock::now());
    return left.count() > 0 ? static_cast<int>(left.count()) : 0;
}

// Appends what arrives on the pipe within wait_ms. Returns false once the pipe is at end of file or cannot be read,
// true otherwise, including when nothing arrived in time.
bool read_pipe(int fd, int wait_ms, std::string & out) {
    pollfd    pfd   = { fd, POLLIN, 0 };
    const int ready = poll(&pfd, 1, wait_ms);
    if (ready < 0) {
        return errno == EINTR;
    }
    if (ready == 0) {
        return true;
    }
    char          buf[4096];
    const ssize_t n = read(fd, buf, sizeof(buf));
    if (n > 0) {
        out.append(buf, static_cast<size_t>(n));
        return true;
    }
    return n < 0 && errno == EINTR;
}

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
        // Stdout goes to the pipe as well as stderr, so the child holds no descriptor of the parent's: a child the
        // kernel keeps alive after the kill would otherwise hold the harness's output pipe open, and the harness would
        // wait on it past the parent's own report.
        if (dup2(fds[1], STDOUT_FILENO) < 0 || dup2(fds[1], STDERR_FILENO) < 0) {
            std::perror("dup2");
            _exit(127);
        }
        close(fds[0]);
        close(fds[1]);
        // /proc/self/exe rather than argv[0]: execl does no PATH search, so a run started by bare name would fail.
        execl("/proc/self/exe", self, "--child", mode, static_cast<char *>(nullptr));
        std::perror("execl");
        _exit(127);
    }
    close(fds[1]);
    // The parent waits for both the pipe's end and the child's exit, because neither proves the other: a child stuck
    // in the kernel while it closes its device files has already closed the pipe. Both waits are bounded, by the
    // limit and then, after a kill, by the grace period, so a hang is reported with what the child printed before it.
    child_result r;
    auto         until = t0 + std::chrono::seconds(k_child_limit_s);
    bool         open  = true;
    int          st    = 0;
    while (open || !r.reaped) {
        if (ms_until(until) == 0) {
            if (r.timed_out || r.reaped) {
                break;
            }
            r.timed_out = true;
            kill(pid, SIGKILL);
            until = std::chrono::steady_clock::now() + std::chrono::seconds(k_child_grace_s);
        }
        if (open) {
            open = read_pipe(fds[0], std::min(ms_until(until), k_child_slice_ms), r.out);
        } else {
            usleep(k_child_slice_ms * 1000);
        }
        if (!r.reaped) {
            const pid_t done = waitpid(pid, &st, WNOHANG);
            if (done < 0 && errno != EINTR) {
                std::perror("waitpid");
                break;
            }
            if (done == pid) {
                r.reaped  = true;
                r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            }
        }
    }
    close(fds[0]);
    r.pipe_open = open;
    if (!r.reaped) {
        r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    r.exited = r.reaped && WIFEXITED(st);
    r.status = r.exited ? WEXITSTATUS(st) : -1;
    r.signal = r.reaped && WIFSIGNALED(st) ? WTERMSIG(st) : 0;
    return r;
}

bool has(const child_result & r, const char * needle) {
    return r.out.find(needle) != std::string::npos;
}

int fail(const char * mode, const child_result & r, const char * why) {
    std::fprintf(stderr,
                 "FAIL mode=%s: %s (exited=%d status=%d signal=%d timed_out=%d reaped=%d pipe_open=%d seconds=%.1f)\n"
                 "--- child stdout and stderr ---\n%s\n",
                 mode, why, r.exited ? 1 : 0, r.status, r.signal, r.timed_out ? 1 : 0, r.reaped ? 1 : 0,
                 r.pipe_open ? 1 : 0, r.seconds, r.out.c_str());
    return 1;
}

// A run the parent could not finish cleanly is reported for what happened, before any check that would blame the
// child's status or its output.
int check_ended(const char * mode, const child_result & r) {
    char why[128];
    if (r.timed_out) {
        std::snprintf(why, sizeof(why), "the child did not end within %d s", k_child_limit_s);
        return fail(mode, r, why);
    }
    if (!r.reaped) {
        return fail(mode, r, "the parent could not wait for the child");
    }
    if (r.pipe_open) {
        std::snprintf(why, sizeof(why), "the child ended, but a process it started still held its output at %d s",
                      k_child_limit_s);
        return fail(mode, r, why);
    }
    return 0;
}

// The fast exit ran: the given status, its banner, and neither marker.
int check_fast_exit(const char * mode, const child_result & r, int want_status) {
    if (const int rc = check_ended(mode, r)) {
        return rc;
    }
    if (!r.exited || r.status != want_status) {
        return fail(mode, r, "wrong exit status");
    }
    if (!has(r, "[SYCL-FATAL]")) {
        return fail(mode, r, "no [SYCL-FATAL] line");
    }
    if (has(r, k_atexit_marker) || has(r, k_dtor_marker)) {
        return fail(mode, r, "an exit handler or a static destructor ran");
    }
    if (r.seconds > k_child_limit_s) {
        return fail(mode, r, "the child ended, but only after the limit");
    }
    return 0;
}

ggml_backend_buffer_t alloc_tensor_buffer(ggml_backend_buffer_type_t buft,
                                          ggml_tensor *              tensor,
                                          ggml_backend_buffer_usage  usage) {
    const size_t          size   = ggml_backend_buft_get_alloc_size(buft, tensor);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, size);
    if (!buffer) {
        return nullptr;
    }
    ggml_backend_buffer_set_usage(buffer, usage);
    ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer));
    return buffer;
}

void fill_random(std::vector<float> & data, float scale, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (float & v : data) {
        v = dist(rng);
    }
}

void quantize_weights_mxfp4(const std::vector<float> & src,
                            int                        in_dim,
                            int                        out_dim,
                            int                        n_experts,
                            std::vector<block_mxfp4> & dst) {
    const int blocks_per_row = in_dim / QK_MXFP4;
    dst.resize(static_cast<size_t>(n_experts) * out_dim * blocks_per_row);
    for (int e = 0; e < n_experts; ++e) {
        for (int r = 0; r < out_dim; ++r) {
            const size_t  row_idx = static_cast<size_t>(e) * out_dim + r;
            const float * row     = src.data() + row_idx * in_dim;
            block_mxfp4 * out     = dst.data() + row_idx * blocks_per_row;
            quantize_row_mxfp4_ref(row, out, in_dim);
        }
    }
}

// Computes one small MUL_MAT_ID graph on the SYCL backend, with its weights in host memory. Returns false when the
// setup or the compute reported a failure. The device modes only need the compute to reach the op: a compute that
// returns at all means the injected loss did not end the process, so nothing is read back and nothing is freed.
bool run_one_mmid(ggml_backend_t backend) {
    const int          in_dim = 64, out_dim = 32, n_experts = 4, n_used = 2, n_tokens = 8;
    std::mt19937       rng(7);
    std::vector<float> weight_data(size_t(in_dim) * out_dim * n_experts);
    std::vector<float> input_data(size_t(in_dim) * n_tokens);
    fill_random(weight_data, 1.0f, rng);
    fill_random(input_data, 1.0f, rng);
    std::vector<block_mxfp4> packed;
    quantize_weights_mxfp4(weight_data, in_dim, out_dim, n_experts, packed);
    std::vector<int32_t> ids_data(size_t(n_used) * n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        for (int i = 0; i < n_used; ++i) {
            ids_data[size_t(t) * n_used + i] = (t + i) % n_experts;
        }
    }

    const ggml_init_params params = {
        64 * 1024 * 1024,
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }
    ggml_tensor * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, in_dim, out_dim, n_experts);
    ggml_set_name(weights, "moe_weights");
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, in_dim, n_used, n_tokens);
    ggml_set_name(input, "moe_input");
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_set_name(ids, "moe_ids");
    ggml_tensor * out = ggml_mul_mat_id(ctx, weights, input, ids);
    ggml_set_name(out, "moe_out");

    if (!alloc_tensor_buffer(ggml_backend_sycl_host_buffer_type(), weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS)) {
        return false;
    }
    if (ggml_backend_dev_t dev = ggml_backend_get_device(backend)) {
        ggml_backend_sycl_register_host_weight_tensor(dev, weights);
    }
    ggml_backend_buffer_type_t dev_buft = ggml_backend_get_default_buffer_type(backend);
    if (!alloc_tensor_buffer(dev_buft, input, GGML_BACKEND_BUFFER_USAGE_COMPUTE) ||
        !alloc_tensor_buffer(dev_buft, ids, GGML_BACKEND_BUFFER_USAGE_COMPUTE) ||
        !alloc_tensor_buffer(dev_buft, out, GGML_BACKEND_BUFFER_USAGE_COMPUTE)) {
        return false;
    }
    ggml_backend_tensor_set(weights, packed.data(), 0, packed.size() * sizeof(block_mxfp4));
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(ids, ids_data.data(), 0, ids_data.size() * sizeof(int32_t));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    // The setup's copies finish before the compute starts, so copy-engine work still on the card when the process
    // ends is the backend's own and not this test's. The injected loss is thrown at the op's entry, before any of
    // the op's work, so waiting here does not change what the test exercises.
    ggml_backend_synchronize(backend);
    return ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
}

// The device modes: the injected loss is in the environment the registration sets.
int child_device(const char * mode) {
    plant_markers();
    // Counted before the backend is made, because on a host with no device ggml_backend_sycl_init(0) does not return
    // null: the skip below it would not fire there.
    if (ggml_backend_sycl_get_device_count() == 0) {
        std::fprintf(stderr, "SKIP: no SYCL device\n");
        return 77;
    }
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (backend == nullptr) {
        std::fprintf(stderr, "SKIP: no SYCL backend\n");
        return 77;
    }
    const bool computed = run_one_mmid(backend);
    std::fprintf(stderr, "TEST-CHILD-SURVIVED mode=%s computed=%d\n", mode, computed ? 1 : 0);
    return 3;
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
    if (std::strcmp(mode, "mmid") == 0 || std::strcmp(mode, "op") == 0) {
        return child_device(mode);
    }
    std::fprintf(stderr, "unknown child mode %s\n", mode);
    return 2;
}

int parent_main(const char * self, const char * mode) {
    if (std::strcmp(mode, "helper") == 0) {
        const child_result r = run_child(self, mode);
        if (const int rc = check_fast_exit(mode, r, GGML_SYCL_EXIT_DEVICE_LOST)) {
            return rc;
        }
        if (!has(r, "[SYCL-FATAL] test: helper mode")) {
            return fail(mode, r, "the message is not on the [SYCL-FATAL] line");
        }
        return 0;
    }
    if (std::strcmp(mode, "control") == 0) {
        const child_result r = run_child(self, mode);
        if (const int rc = check_ended(mode, r)) {
            return rc;
        }
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
    if (std::strcmp(mode, "mmid") == 0 || std::strcmp(mode, "op") == 0) {
        const child_result r = run_child(self, mode);
        if (r.exited && r.status == 77) {
            std::fprintf(stderr, "%s", r.out.c_str());
            return 77;
        }
        if (has(r, "TEST-CHILD-SURVIVED")) {
            const std::string why = std::string(
                                        "the injected loss was never thrown: the op did not reach the site "
                                        "(GGML_SYCL_TEST_INJECT_DEVICE_LOST=") +
                                    mode + ":1 is set by the ctest registration)";
            return fail(mode, r, why.c_str());
        }
        const bool   mmid      = std::strcmp(mode, "mmid") == 0;
        const char * site_line = mmid ? "[MUL_MAT_ID-FAIL]" : "Error OP";
        if (!has(r, site_line) || !has(r, "Exception caught at file:") || !has(r, "UR_RESULT_ERROR_DEVICE_LOST")) {
            return fail(mode, r, "the catch's own diagnostic lines are missing");
        }
        if (const int rc = check_fast_exit(mode, r, GGML_SYCL_EXIT_DEVICE_LOST)) {
            return rc;
        }
        // The site's own call names its function. Without this, a site that rethrew instead would pass on the exit of
        // the catch around it.
        const char * banner =
            mmid ? "[SYCL-FATAL] ggml_sycl_mul_mat_id:" : "[SYCL-FATAL] ggml_sycl_compute_forward_impl:";
        if (!has(r, banner)) {
            return fail(mode, r, "the process did not end through the site's own fast exit");
        }
        return 0;
    }
    std::fprintf(stderr, "usage: %s helper|control|classify|mmid|op\n", self);
    return 2;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--child") == 0) {
        return child_main(argv[2]);
    }
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s helper|control|classify|mmid|op\n", argv[0]);
        return 2;
    }
    const int rc = parent_main(argv[0], argv[1]);
    if (rc == 0) {
        std::fprintf(stderr, "PASS mode=%s\n", argv[1]);
    }
    return rc;
}
