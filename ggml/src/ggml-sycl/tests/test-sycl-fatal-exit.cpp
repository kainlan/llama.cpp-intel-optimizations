// A process the backend ends after a fatal error must not run its exit handlers or static destructors: after a
// device loss the Level Zero library's own destructor does not return. Each mode runs this binary again as a child
// that plants an atexit handler and a static destructor, and checks from outside which of them ran.

#include "../../../../tests/test-skip.h"  // LLAMA_TEST_EXIT_SKIP: the one definition of "77 means skip"
#include "common.hpp"
#include "exit-settle.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "ggml-sycl.h"
#include "ggml.h"

#include <dirent.h>
#include <poll.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/stat.h>
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
#include <thread>
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

// How long a device mode's parent holds the next process off the card after its child ran. A child that ends through
// the fast exit can leave copy-engine work on the card: a process that started on the same card about a second later
// was seen to hang on its first copy, and the kernel reset the whole GT about 35 s later (B50, driver 26.44). Runs
// spaced about 11 s apart left only lines scoped to the dead context. So the parent waits this long before it
// returns, and the next test cannot start on the card sooner.
constexpr std::chrono::seconds k_device_settle{ 12 };

// The capture switch, the line the backend prints when it wrote all the files, and the line it prints for each file
// it could not write. The second does not contain the first, so counting capture lines does not count failures.
constexpr const char * k_capture_env         = "GGML_SYCL_FATAL_CAPTURE_DIR";
constexpr const char * k_capture_line        = "[SYCL-FATAL] capture: ";
constexpr const char * k_capture_failed_line = "[SYCL-FATAL] capture failed: ";
constexpr const char * k_settle_env          = "GGML_SYCL_FATAL_EXIT_SETTLE_MS";
constexpr const char * k_settle_line         = "[SYCL-FATAL] settle: ";
constexpr const char * k_settle_t0_line      = "TEST-SETTLE-T0 ";

// The values the settle-cap child gives GGML_SYCL_FATAL_EXIT_SETTLE_MS (null: unset) and the wait each must read as.
struct settle_case {
    const char * label;
    const char * value;
    long         want_ms;
};

constexpr settle_case k_settle_cases[] = {
    { "unset",        nullptr,                200  },
    { "empty",        "",                     200  },
    { "garbage",      "abc",                  200  },
    { "negative",     "-5",                   200  },
    { "zero",         "0",                    0    },
    { "in-range",     "150",                  150  },
    { "above-cap",    "999999",               5000 },
    { "out-of-range", "99999999999999999999", 5000 },
};

// The files a capture writes, as <dir>/sycl-fatal-<pid><ext>.
constexpr const char * k_capture_exts[] = { ".maps", ".smaps", ".status" };
constexpr size_t       k_capture_n_exts = sizeof(k_capture_exts) / sizeof(k_capture_exts[0]);

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
    long long   reaped_ns = 0;      // steady_clock at the reap, in the clock's own epoch; 0 if never reaped
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
                const auto now = std::chrono::steady_clock::now();
                r.reaped       = true;
                r.seconds      = std::chrono::duration<double>(now - t0).count();
                r.reaped_ns    = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
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

// The steady_clock time, in nanoseconds, that the child printed after `tag`. steady_clock is CLOCK_MONOTONIC on
// Linux, so it is one clock in the parent and the child. False when the line is missing or does not hold a number.
bool child_time_ns(const child_result & r, const char * tag, long long & ns) {
    const size_t at = r.out.find(tag);
    if (at == std::string::npos) {
        return false;
    }
    const char * begin = r.out.c_str() + at + std::strlen(tag);
    char *       end   = nullptr;
    errno              = 0;
    ns                 = std::strtoll(begin, &end, 10);
    return errno == 0 && end != begin && (*end == '\n' || *end == '\0') && ns > 0;
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

// Runs one small MUL_MAT_ID graph on the SYCL backend, with its weights in host memory. Returns false when the setup
// or the compute reported a failure. On a bare backend the compute always fails: with no model there is no planned
// route for the MUL_MAT_ID, and the backend refuses it at the op. The modes that use this (mmid and op) only need the
// compute to reach the op: a compute that returns at all means the injected loss did not end the process, so nothing
// is read back and nothing is freed.
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

// Runs one ggml_add of two F32 device tensors, a graph that needs no model, no planned route and no weight cache, so
// it computes on a bare backend. Returns true when the compute succeeded. The sync mode uses it because its loss must
// land on a wait after a compute that worked, and the graph mode because a failed compute can then only be the
// injected one. Built the way test-rms-norm-mul-add-broadcast.cpp builds its graph. Its context and buffer are leaked
// by intent: the child ends without teardown.
bool run_one_add(ggml_backend_t backend) {
    const int              n      = 256;
    const ggml_init_params params = {
        ggml_tensor_overhead() * 4 + ggml_graph_overhead(),
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_input(a);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_input(b);
    ggml_tensor * sum = ggml_add(ctx, a, b);
    ggml_set_output(sum);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, sum);
    if (!ggml_backend_alloc_ctx_tensors(ctx, backend)) {
        return false;
    }
    std::vector<float> data(n, 1.0f);
    ggml_backend_tensor_set(a, data.data(), 0, data.size() * sizeof(float));
    ggml_backend_tensor_set(b, data.data(), 0, data.size() * sizeof(float));
    return ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
}

// The injection a device mode's child sets for itself, <site>:<n>, which throws from the site's nth call on. The sync
// mode's count skips the one synchronize call before the one it targets: the one ggml_backend_graph_compute ends with.
// The graph mode's 1 is the child's first graph compute.
const char * device_injection(const char * mode) {
    if (std::strcmp(mode, "sync") == 0) {
        return "sync:2";
    }
    if (std::strcmp(mode, "graph") == 0) {
        return "graph:1";
    }
    return std::strcmp(mode, "mmid") == 0 ? "mmid:1" : "op:1";
}

// The device modes. The child sets its own injection before any backend call, so the binary needs no environment
// beyond the device selector: `test-sycl-fatal-exit sync` runs as is.
int child_device(const char * mode) {
    if (setenv("GGML_SYCL_TEST_INJECT_DEVICE_LOST", device_injection(mode), 1) != 0) {
        std::perror("setenv");
        return 2;
    }
    plant_markers();
    // Counted before the backend is made, because on a host with no device ggml_backend_sycl_init(0) does not return
    // null: the skip below it would not fire there.
    if (ggml_backend_sycl_get_device_count() == 0) {
        std::fprintf(stderr, "SKIP: no SYCL device\n");
        return LLAMA_TEST_EXIT_SKIP;
    }
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (backend == nullptr) {
        std::fprintf(stderr, "SKIP: no SYCL backend\n");
        return LLAMA_TEST_EXIT_SKIP;
    }
    const bool sync     = std::strcmp(mode, "sync") == 0;
    const bool graph    = std::strcmp(mode, "graph") == 0;
    const bool computed = (sync || graph) ? run_one_add(backend) : run_one_mmid(backend);
    if (graph && !computed) {
        // The boundary reported the failure and returned. An application would now wind down and return from main.
        std::fprintf(stderr, "TEST-GRAPH-FAILED\n");
        return 0;
    }
    if (sync) {
        // The loss must land on this wait, after the compute returned, and not on the earlier one that
        // ggml_backend_graph_compute ends with. The parent requires this line.
        std::fprintf(stderr, "TEST-SYNC-COMPUTED computed=%d\n", computed ? 1 : 0);
        ggml_backend_synchronize(backend);
    }
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
    if (std::strcmp(mode, "helper-small-files") == 0) {
        // The helper's fast exit under a file size limit of 64 bytes. Each of the three /proc files the capture copies
        // is larger, so every copy fails after its file was opened. The limit applies to regular files only: the pipe
        // to the parent carries the output as usual. SIGXFSZ is ignored, so a write past the limit returns EFBIG
        // instead of ending the process.
        plant_markers();
        if (signal(SIGXFSZ, SIG_IGN) == SIG_ERR) {
            std::fprintf(stderr, "TEST-SETUP-FAILED signal(SIGXFSZ): %s\n", std::strerror(errno));
            return 4;
        }
        rlimit limit{};
        limit.rlim_cur = 64;
        limit.rlim_max = 64;
        if (setrlimit(RLIMIT_FSIZE, &limit) != 0) {
            std::fprintf(stderr, "TEST-SETUP-FAILED setrlimit(RLIMIT_FSIZE): %s\n", std::strerror(errno));
            return 4;
        }
        ggml_sycl_fatal_exit(GGML_SYCL_EXIT_DEVICE_LOST, "test: helper mode");
    }
    if (std::strcmp(mode, "lost-exit") == 0) {
        // A loss recorded with no backend and no device, after which the child returns from main as an application
        // would: the exit handler the loss registered ends the process.
        plant_markers();
        ggml_sycl_note_exception("level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)");
        std::fprintf(stderr, "TEST-LOSS-RECORDED\n");
        return 0;
    }
    if (std::strcmp(mode, "settle") == 0 || std::strcmp(mode, "settle-off") == 0) {
        plant_markers();
        // The time the settle starts from. The parent measures from here to the reap, so the child's start-up, which
        // a measure from the fork would include, cannot stand in for a settle that does not sleep.
        const long long t0_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count();
        std::fprintf(stderr, "%s%lld\n", k_settle_t0_line, t0_ns);
        std::fflush(stderr);
        ggml_sycl_fatal_exit(GGML_SYCL_EXIT_DEVICE_LOST, "test: settle mode");
    }
    if (std::strcmp(mode, "settle-cap") == 0) {
        // The parser is read back, not slept through: a child that slept for the cap would hold the test for five
        // seconds.
        for (const settle_case & c : k_settle_cases) {
            if (c.value == nullptr) {
                unsetenv(k_settle_env);
            } else {
                setenv(k_settle_env, c.value, 1);
            }
            std::fprintf(stderr, "TEST-SETTLE-MS %s %ld\n", c.label, ggml_sycl::exit_settle_ms());
        }
        unsetenv(k_settle_env);
        return 0;
    }
    if (std::strcmp(mode, "mmid") == 0 || std::strcmp(mode, "op") == 0 || std::strcmp(mode, "sync") == 0 ||
        std::strcmp(mode, "graph") == 0) {
        return child_device(mode);
    }
    std::fprintf(stderr, "unknown child mode %s\n", mode);
    return 2;
}

// The checks on a site mode's child (mmid, op) once it ran.
int check_site_child(const char * mode, const child_result & r) {
    if (has(r, "TEST-CHILD-SURVIVED")) {
        const std::string why = std::string(
                                    "the injected loss was never thrown: the op did not reach the site "
                                    "(GGML_SYCL_TEST_INJECT_DEVICE_LOST=") +
                                device_injection(mode) + " is what the child set)";
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
    const char * banner = mmid ? "[SYCL-FATAL] ggml_sycl_mul_mat_id:" : "[SYCL-FATAL] ggml_sycl_compute_forward_impl:";
    if (!has(r, banner)) {
        return fail(mode, r, "the process did not end through the site's own fast exit");
    }
    return 0;
}

// The checks on a boundary mode's child (sync, graph) once it ran.
int check_boundary_child(const char * mode, const child_result & r) {
    const bool sync = std::strcmp(mode, "sync") == 0;
    if (has(r, "TEST-CHILD-SURVIVED")) {
        const std::string why = std::string(
                                    "the injected loss was never thrown: the wait or the boundary did not reach "
                                    "the site (GGML_SYCL_TEST_INJECT_DEVICE_LOST=") +
                                device_injection(mode) + " is what the child set)";
        return fail(mode, r, why.c_str());
    }
    if (sync && !has(r, "TEST-SYNC-COMPUTED computed=1")) {
        return fail(mode, r,
                    "the loss did not land on the synchronize after a successful compute: it landed on an "
                    "earlier call, or the compute failed");
    }
    if (!sync && !has(r, "TEST-GRAPH-FAILED")) {
        return fail(mode, r, "the graph boundary did not return a failure to its caller");
    }
    if (!has(r, "UR_RESULT_ERROR_DEVICE_LOST")) {
        return fail(mode, r, "the loss was not reported");
    }
    if (!has(r, "[SYCL-FATAL] device lost:")) {
        return fail(mode, r, "the loss was not recorded as a device loss");
    }
    if (const int rc = check_fast_exit(mode, r, GGML_SYCL_EXIT_DEVICE_LOST)) {
        return rc;
    }
    // The path that ended the process names itself: the fatal check after the wait in sync mode, the exit handler
    // the loss registered in graph mode.
    const char * banner =
        sync ? "[SYCL-FATAL] SYCL_CHECK failed after a device loss:" : "[SYCL-FATAL] process exit after a device loss:";
    if (!has(r, banner)) {
        return fail(mode, r, "the process did not end through the path the loss's site takes");
    }
    return 0;
}

// Called once a device mode's child has run, whether the mode passed or failed. Not after a skip: no device process
// ran then.
void settle_after_device_child() {
    std::fprintf(stderr, "TEST-SETTLE waiting %lld s before the next process may use the card\n",
                 static_cast<long long>(k_device_settle.count()));
    std::fflush(stderr);
    std::this_thread::sleep_for(k_device_settle);
}

bool read_file(const std::string & path, std::string & text) {
    std::FILE * f = std::fopen(path.c_str(), "r");
    if (f == nullptr) {
        return false;
    }
    char   buf[4096];
    size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) {
        text.append(buf, n);
    }
    std::fclose(f);
    return true;
}

// The entries of `dir` other than . and .., in readdir order. False when the directory cannot be read.
bool list_dir(const std::string & dir, std::vector<std::string> & names) {
    DIR * d = opendir(dir.c_str());
    if (d == nullptr) {
        return false;
    }
    for (const dirent * e = readdir(d); e != nullptr; e = readdir(d)) {
        if (std::strcmp(e->d_name, ".") != 0 && std::strcmp(e->d_name, "..") != 0) {
            names.push_back(e->d_name);
        }
    }
    closedir(d);
    return true;
}

// The capture a child left: one capture line naming <dir>/sycl-fatal-<pid>, and in <dir> exactly the three files under
// that name, none of them empty, the map with the stack's mapping in it, and none open to group or others, since the
// kernel lets only a ptrace-capable reader open a process's maps. One line and three entries are what "written once"
// looks like from outside: a second write would print a second line or leave another file. A pass removes the files
// and the directory; a failure leaves them for a look.
int check_capture_files(const char * mode, const child_result & r, const std::string & dir) {
    if (has(r, k_capture_failed_line)) {
        return fail(mode, r, "a capture file could not be written, although the directory exists");
    }
    size_t lines = 0;
    for (size_t at = r.out.find(k_capture_line); at != std::string::npos; at = r.out.find(k_capture_line, at + 1)) {
        ++lines;
    }
    if (lines == 0) {
        return fail(mode, r, "no capture line");
    }
    if (lines > 1) {
        return fail(mode, r, "more than one capture line: the capture ran more than once");
    }
    const size_t      begin  = r.out.find(k_capture_line) + std::strlen(k_capture_line);
    const std::string base   = r.out.substr(begin, r.out.find('\n', begin) - begin);
    const std::string prefix = dir + "/sycl-fatal-";
    if (base.compare(0, prefix.size(), prefix) != 0 || base.size() == prefix.size() ||
        base.find_first_not_of("0123456789", prefix.size()) != std::string::npos) {
        return fail(mode, r, "the capture line does not name <dir>/sycl-fatal-<pid> in the named directory");
    }
    std::string why;
    for (const char * ext : k_capture_exts) {
        const std::string path = base + ext;
        std::string       text;
        if (!read_file(path, text)) {
            why = "a capture file is missing: " + path;
            return fail(mode, r, why.c_str());
        }
        if (text.empty() || (std::strcmp(ext, ".maps") == 0 && text.find("[stack]") == std::string::npos)) {
            why = "a capture file is empty or is not a memory map: " + path;
            return fail(mode, r, why.c_str());
        }
        struct stat st{};
        if (stat(path.c_str(), &st) != 0) {
            why = "a capture file cannot be inspected: " + path;
            return fail(mode, r, why.c_str());
        }
        if ((st.st_mode & 077) != 0) {
            char octal[16];
            std::snprintf(octal, sizeof(octal), "%04o", static_cast<unsigned>(st.st_mode & 07777));
            why = "a capture file is open to group or others: " + path + " (mode " + octal + ")";
            return fail(mode, r, why.c_str());
        }
    }
    std::vector<std::string> names;
    if (!list_dir(dir, names)) {
        return fail(mode, r, "the capture directory cannot be read");
    }
    // The three files exist, so any entry that is not one of them is extra.
    const std::string file = base.substr(dir.size() + 1);
    for (const std::string & name : names) {
        bool expected = false;
        for (const char * ext : k_capture_exts) {
            expected = expected || name == file + ext;
        }
        if (!expected) {
            why = "the capture directory holds more than the three files: " + name;
            return fail(mode, r, why.c_str());
        }
    }
    for (const char * ext : k_capture_exts) {
        std::remove((base + ext).c_str());
    }
    rmdir(dir.c_str());
    return 0;
}

// A capture that failed: no capture line, which would claim files that are not complete, and exactly three capture
// failed lines, one naming each file as <dir>/sycl-fatal-<pid><ext> followed by ": " and the reason. When `reason` is
// not null, each line must give exactly that reason.
int check_capture_failed(const char * mode, const child_result & r, const std::string & dir, const char * reason) {
    if (has(r, k_capture_line)) {
        return fail(mode, r, "a capture line, although the capture failed");
    }
    std::vector<std::string> failed;  // each line's text after the marker
    size_t                   at = r.out.find(k_capture_failed_line);
    while (at != std::string::npos) {
        at += std::strlen(k_capture_failed_line);
        failed.push_back(r.out.substr(at, r.out.find('\n', at) - at));
        at = r.out.find(k_capture_failed_line, at);
    }
    std::string why;
    if (failed.size() != k_capture_n_exts) {
        why = "expected one capture failed line per file, three in all, and found " + std::to_string(failed.size());
        return fail(mode, r, why.c_str());
    }
    const std::string prefix = dir + "/sycl-fatal-";
    for (const char * ext : k_capture_exts) {
        const std::string tail  = std::string(ext) + ": ";
        bool              named = false;
        for (const std::string & line : failed) {
            if (line.compare(0, prefix.size(), prefix) != 0) {
                continue;
            }
            const size_t digits_end = line.find_first_not_of("0123456789", prefix.size());
            if (digits_end == std::string::npos || digits_end == prefix.size() ||
                line.compare(digits_end, tail.size(), tail) != 0) {
                continue;
            }
            named = named || reason == nullptr || line.substr(digits_end + tail.size()) == reason;
        }
        if (!named) {
            why = "no capture failed line for " + prefix + "<pid>" + ext +
                  (reason != nullptr ? std::string(" with the reason \"") + reason + "\"" : std::string());
            return fail(mode, r, why.c_str());
        }
    }
    return 0;
}

// One child of the capture mode, run with GGML_SYCL_FATAL_CAPTURE_DIR naming a new directory of its own. Every failure
// text names that directory through the mode name.
int run_capture_child(const char * self, const char * child_mode) {
    setenv(k_settle_env, "0", 1);
    char dir[] = "/tmp/sycl-fatal-capture-XXXXXX";
    if (mkdtemp(dir) == nullptr) {
        std::perror("mkdtemp");
        return 1;
    }
    if (setenv(k_capture_env, dir, 1) != 0) {
        std::perror("setenv");
        return 1;
    }
    const child_result r = run_child(self, child_mode);
    unsetenv(k_capture_env);
    const std::string label     = std::string("capture child=") + child_mode + " dir=" + dir;
    const char *      mode      = label.c_str();
    const bool        lost_exit = std::strcmp(child_mode, "lost-exit") == 0;
    if (lost_exit && !has(r, "TEST-LOSS-RECORDED")) {
        return fail(mode, r, "the child did not reach the recorded loss");
    }
    if (lost_exit && !has(r, "[SYCL-FATAL] device lost:")) {
        return fail(mode, r, "the loss was not recorded as a device loss");
    }
    if (const int rc = check_fast_exit(mode, r, GGML_SYCL_EXIT_DEVICE_LOST)) {
        return rc;
    }
    if (lost_exit) {
        // The loss and the exit both reach the capture. The map must be the one taken at the loss, before the exit
        // handler's banner: on the latch path the process may run on, and a later map no longer shows what was mapped
        // when the device faulted.
        const size_t banner  = r.out.find("[SYCL-FATAL] process exit after a device loss");
        const size_t capture = r.out.find(k_capture_line);
        if (banner == std::string::npos) {
            return fail(mode, r, "the process did not end through the exit handler the loss registered");
        }
        if (capture != std::string::npos && capture > banner) {
            return fail(mode, r, "the capture was taken at the exit, not at the loss");
        }
    }
    return check_capture_files(mode, r, dir);
}

// The helper child with GGML_SYCL_FATAL_CAPTURE_DIR naming a directory that does not exist. The capture must not
// create it and must not print the capture line, which would claim files that are not there; it must name each file
// it could not write instead.
int run_capture_child_missing_dir(const char * self) {
    setenv(k_settle_env, "0", 1);
    char dir[] = "/tmp/sycl-fatal-capture-XXXXXX";
    if (mkdtemp(dir) == nullptr) {
        std::perror("mkdtemp");
        return 1;
    }
    if (rmdir(dir) != 0) {
        std::perror("rmdir");
        return 1;
    }
    if (setenv(k_capture_env, dir, 1) != 0) {
        std::perror("setenv");
        return 1;
    }
    const child_result r = run_child(self, "helper");
    unsetenv(k_capture_env);
    const std::string label = std::string("capture child=helper dir=") + dir + " (removed before the run)";
    const char *      mode  = label.c_str();
    if (const int rc = check_fast_exit(mode, r, GGML_SYCL_EXIT_DEVICE_LOST)) {
        return rc;
    }
    if (const int rc = check_capture_failed(mode, r, dir, nullptr)) {
        return rc;
    }
    if (access(dir, F_OK) == 0) {
        return fail(mode, r, "the directory exists after the run: the capture created it");
    }
    return 0;
}

// The helper child under a file size limit below the size of every file the capture copies, so each copy fails after
// its file was opened. Each file must be named with the error the write gave, and none may be left behind: a partial
// copy under the final name would read as a complete one once the log is gone.
int run_capture_child_small_files(const char * self) {
    setenv(k_settle_env, "0", 1);
    char dir[] = "/tmp/sycl-fatal-capture-XXXXXX";
    if (mkdtemp(dir) == nullptr) {
        std::perror("mkdtemp");
        return 1;
    }
    if (setenv(k_capture_env, dir, 1) != 0) {
        std::perror("setenv");
        return 1;
    }
    const child_result r = run_child(self, "helper-small-files");
    unsetenv(k_capture_env);
    const std::string label = std::string("capture child=helper-small-files dir=") + dir;
    const char *      mode  = label.c_str();
    if (const int rc = check_fast_exit(mode, r, GGML_SYCL_EXIT_DEVICE_LOST)) {
        return rc;
    }
    const std::string efbig = std::strerror(EFBIG);
    if (const int rc = check_capture_failed(mode, r, dir, efbig.c_str())) {
        return rc;
    }
    std::vector<std::string> names;
    if (!list_dir(dir, names)) {
        return fail(mode, r, "the capture directory cannot be read");
    }
    if (!names.empty()) {
        const std::string why = "a failed capture left a file behind: " + names.front();
        return fail(mode, r, why.c_str());
    }
    rmdir(dir);
    return 0;
}

int parent_main(const char * self, const char * mode) {
    if (std::strcmp(mode, "helper") == 0) {
        // Cleared so that a value in the caller's environment cannot make the child capture: this mode checks that a
        // run without the variable leaves no capture.
        unsetenv(k_capture_env);
        setenv(k_settle_env, "0", 1);
        const child_result r = run_child(self, mode);
        if (const int rc = check_fast_exit(mode, r, GGML_SYCL_EXIT_DEVICE_LOST)) {
            return rc;
        }
        if (!has(r, "[SYCL-FATAL] test: helper mode")) {
            return fail(mode, r, "the message is not on the [SYCL-FATAL] line");
        }
        if (has(r, k_capture_line) || has(r, k_capture_failed_line)) {
            return fail(mode, r,
                        "a capture line or a capture failed line, although GGML_SYCL_FATAL_CAPTURE_DIR is unset");
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
        // Neither another error's text nor a null records a loss, so the latch must still be clear below.
        ggml_sycl_note_exception(oom);
        ggml_sycl_note_exception(nullptr);
        const bool ok = ggml_sycl_what_is_device_lost(lost) && !ggml_sycl_what_is_device_lost(oom) &&
                        !ggml_sycl_what_is_device_lost(nullptr) &&
                        ggml_sycl_fatal_exit_code(lost) == GGML_SYCL_EXIT_DEVICE_LOST &&
                        ggml_sycl_fatal_exit_code(oom) == 1 && ggml_sycl_fatal_exit_code(nullptr) == 1 &&
                        !ggml_sycl_device_lost_latched();
        if (!ok) {
            std::fprintf(stderr, "FAIL mode=classify\n");
            return 1;
        }
        return 0;
    }
    if (std::strcmp(mode, "capture") == 0) {
        // The two ways to the capture: the fast exit with no loss recorded (the helper child), and a recorded loss the
        // process then leaves through its exit handler (the lost-exit child), whose map must be the one taken at the
        // loss. Then two captures that cannot be written, which must say so: one whose directory does not exist and
        // one whose files cannot grow past 64 bytes. A host mode: no child uses a device, so the card needs no settling
        // after it.
        if (const int rc = run_capture_child(self, "helper")) {
            return rc;
        }
        if (const int rc = run_capture_child(self, "lost-exit")) {
            return rc;
        }
        if (const int rc = run_capture_child_missing_dir(self)) {
            return rc;
        }
        return run_capture_child_small_files(self);
    }
    if (std::strcmp(mode, "settle") == 0 || std::strcmp(mode, "settle-off") == 0) {
        const bool off = std::strcmp(mode, "settle-off") == 0;
        unsetenv(k_capture_env);
        setenv(k_settle_env, off ? "0" : "150", 1);
        const child_result r = run_child(self, mode);
        unsetenv(k_settle_env);
        if (const int rc = check_fast_exit(mode, r, GGML_SYCL_EXIT_DEVICE_LOST)) {
            return rc;
        }
        if (off) {
            return has(r, k_settle_line) ? fail(mode, r, "a settle line, although the settle is disabled") : 0;
        }
        if (!has(r, "[SYCL-FATAL] settle: waiting 150 ms")) {
            return fail(mode, r, "no settle line with the requested time");
        }
        long long t0_ns = 0;
        if (!child_time_ns(r, k_settle_t0_line, t0_ns)) {
            return fail(mode, r, "no readable TEST-SETTLE-T0 line, so the settle cannot be timed");
        }
        // A lower bound only: the reap is seen up to one poll slice late, and load can lengthen the wait, but neither
        // can make a real 150 ms settle read as shorter.
        if (r.reaped_ns - t0_ns < 150000000LL) {
            return fail(mode, r, "the child ended less than 150 ms after it started the settle");
        }
        return 0;
    }
    if (std::strcmp(mode, "settle-cap") == 0) {
        const child_result r = run_child(self, mode);
        if (const int rc = check_ended(mode, r)) {
            return rc;
        }
        if (!r.exited || r.status != 0) {
            return fail(mode, r, "the child did not read every value and return 0");
        }
        for (const settle_case & c : k_settle_cases) {
            char line[96];
            std::snprintf(line, sizeof(line), "TEST-SETTLE-MS %s %ld\n", c.label, c.want_ms);
            if (!has(r, line)) {
                const std::string why = std::string("the settle does not read as expected: want ") + line;
                return fail(mode, r, why.c_str());
            }
        }
        return 0;
    }
    if (std::strcmp(mode, "mmid") == 0 || std::strcmp(mode, "op") == 0) {
        const child_result r = run_child(self, mode);
        if (r.exited && r.status == LLAMA_TEST_EXIT_SKIP) {
            std::fprintf(stderr, "%s", r.out.c_str());
            return LLAMA_TEST_EXIT_SKIP;
        }
        const int rc = check_site_child(mode, r);
        settle_after_device_child();
        return rc;
    }
    if (std::strcmp(mode, "sync") == 0 || std::strcmp(mode, "graph") == 0) {
        const child_result r = run_child(self, mode);
        if (r.exited && r.status == LLAMA_TEST_EXIT_SKIP) {
            std::fprintf(stderr, "%s", r.out.c_str());
            return LLAMA_TEST_EXIT_SKIP;
        }
        const int rc = check_boundary_child(mode, r);
        settle_after_device_child();
        return rc;
    }
    std::fprintf(stderr, "usage: %s helper|control|classify|capture|settle|settle-off|settle-cap|mmid|op|sync|graph\n",
                 self);
    return 2;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--child") == 0) {
        return child_main(argv[2]);
    }
    if (argc != 2) {
        std::fprintf(stderr,
                     "usage: %s helper|control|classify|capture|settle|settle-off|settle-cap|mmid|op|sync|graph\n",
                     argv[0]);
        return 2;
    }
    const int rc = parent_main(argv[0], argv[1]);
    if (rc == 0) {
        std::fprintf(stderr, "PASS mode=%s\n", argv[1]);
    }
    return rc;
}
