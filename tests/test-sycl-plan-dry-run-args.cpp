// The plan dry run's command line (tools/sycl-plan-dry-run): every option reaches the field the export reads, and
// every malformed value is refused with a message rather than read as a default. Host-only: the parser is a header
// that includes llama.h for its constants and nothing else, so no backend, device or model is touched.

#include "sycl-plan-dry-run-args.h"

#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <vector>

namespace {

constexpr uint64_t GiB = 1024ull * 1024ull * 1024ull;

int g_failures = 0;

void check(bool ok, const std::string & what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

struct parsed {
    bool                   ok;
    sycl_plan_dry_run_args args;
    std::string            err;
};

parsed parse(std::initializer_list<const char *> words) {
    std::vector<const char *> argv = { "llama-sycl-plan-dry-run" };
    argv.insert(argv.end(), words.begin(), words.end());
    parsed p{};
    p.ok = sycl_plan_dry_run_parse_args(static_cast<int>(argv.size()), argv.data(), p.args, p.err);
    return p;
}

void refused(std::initializer_list<const char *> words, const std::string & what) {
    const parsed p = parse(words);
    check(!p.ok, what + " was accepted");
    check(p.ok || !p.err.empty(), what + " was refused without a message");
}

}  // namespace

int main() {
    // defaults: only the model named; nothing overridden
    {
        const parsed p = parse({ "-m", "/models/x.gguf" });
        check(p.ok, "-m alone was refused: " + p.err);
        check(p.args.model == "/models/x.gguf", "the model path is '" + p.args.model + "'");
        check(!p.args.mem_available_set, "MemAvailable is overridden without --mem-available");
        check(p.args.n_device_budgets == 0, "device budgets are set without --device-budget");
        check(p.args.n_ctx == LLAMA_N_CTX_HINT_DEFAULT, "the context is not the default without -c");
        check(!p.args.lazy_mode_set, "the lazy mode is set without -lzm");
        check(!p.args.help, "help is set without -h");
    }
    // --mem-available takes GiB, fractions included
    {
        const parsed p = parse({ "-m", "x", "--mem-available", "64" });
        check(p.ok && p.args.mem_available_set && p.args.mem_available_bytes == 64 * GiB,
              "--mem-available 64 is not 64 GiB");
        const parsed h = parse({ "-m", "x", "--mem-available", "0.5" });
        check(h.ok && h.args.mem_available_bytes == GiB / 2, "--mem-available 0.5 is not half a GiB");
    }
    refused({ "-m", "x", "--mem-available", "0" }, "--mem-available 0");
    refused({ "-m", "x", "--mem-available", "-1" }, "--mem-available -1");
    refused({ "-m", "x", "--mem-available", "abc" }, "--mem-available abc");
    refused({ "-m", "x", "--mem-available", "64x" }, "--mem-available 64x");
    refused({ "-m", "x", "--mem-available" }, "--mem-available with no value");
    // --device-budget D=GiB, repeatable; an unnamed device below D keeps the plan's own budget (0)
    {
        const parsed p = parse({ "-m", "x", "--device-budget", "1=10" });
        check(p.ok && p.args.n_device_budgets == 2, "--device-budget 1=10 does not cover devices 0 and 1");
        check(p.ok && p.args.device_budget_bytes[0] == 0, "--device-budget 1=10 overrides device 0");
        check(p.ok && p.args.device_budget_bytes[1] == 10 * GiB, "--device-budget 1=10 is not 10 GiB on device 1");
        const parsed two = parse({ "-m", "x", "--device-budget", "1=10", "--device-budget", "0=5" });
        check(two.ok && two.args.n_device_budgets == 2 && two.args.device_budget_bytes[0] == 5 * GiB &&
                  two.args.device_budget_bytes[1] == 10 * GiB,
              "two --device-budget options do not both apply");
        const parsed last = parse({ "-m", "x", "--device-budget", "15=1" });
        check(last.ok && last.args.n_device_budgets == SYCL_PLAN_DRY_RUN_ARGS_MAX_DEVICES,
              "--device-budget for the last device is refused");
    }
    refused({ "-m", "x", "--device-budget", "16=1" }, "--device-budget past the last device");
    refused({ "-m", "x", "--device-budget", "-1=1" }, "--device-budget for device -1");
    refused({ "-m", "x", "--device-budget", "1" }, "--device-budget with no '='");
    refused({ "-m", "x", "--device-budget", "1=0" }, "--device-budget of 0 GiB");
    refused({ "-m", "x", "--device-budget", "a=1" }, "--device-budget for device 'a'");
    // -c: 0 is the default context, as for llama-cli
    {
        const parsed p = parse({ "-m", "x", "-c", "4096" });
        check(p.ok && p.args.n_ctx == 4096, "-c 4096 is not 4096");
        const parsed z = parse({ "-m", "x", "-c", "0" });
        check(z.ok && z.args.n_ctx == LLAMA_N_CTX_HINT_DEFAULT, "-c 0 is not the default context");
    }
    refused({ "-m", "x", "-c", "-1" }, "-c -1");
    refused({ "-m", "x", "-c", "abc" }, "-c abc");
    // -lzm on|auto|off, as for llama-cli
    {
        const parsed on  = parse({ "-m", "x", "-lzm", "on" });
        const parsed aut = parse({ "-m", "x", "--lazy-mode", "auto" });
        const parsed off = parse({ "-m", "x", "-lzm", "off" });
        check(on.ok && on.args.lazy_mode_set && on.args.lazy_mode == LLAMA_LAZY_MODE_ON, "-lzm on is not ON");
        check(aut.ok && aut.args.lazy_mode_set && aut.args.lazy_mode == LLAMA_LAZY_MODE_AUTO,
              "--lazy-mode auto is not AUTO");
        check(off.ok && off.args.lazy_mode_set && off.args.lazy_mode == LLAMA_LAZY_MODE_OFF, "-lzm off is not OFF");
    }
    refused({ "-m", "x", "-lzm", "maybe" }, "-lzm maybe");
    // -h needs no model; anything else does
    {
        const parsed h = parse({ "-h" });
        check(h.ok && h.args.help, "-h is refused or does not set help");
    }
    refused({}, "no options");
    refused({ "--mem-available", "8" }, "no -m");
    refused({ "-m", "x", "--frobnicate" }, "an unknown option");
    refused({ "-m", "x", "-ngl", "99" }, "-ngl");

    std::printf("%s: %d failures\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}
