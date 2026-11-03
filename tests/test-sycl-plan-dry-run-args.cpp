// The plan dry run's command line (tools/sycl-plan-dry-run): every option reaches the field the export reads, and
// every malformed value is refused with a message rather than read as a default. Then the tool's header read
// (llama_model_sycl_header_inventory) on two GGUFs the test writes. On an architecture no loader knows it returns false
// and names the architecture, before any model exists. On a known one (llama, with the hparams its loader requires and
// one tensor) it hands the callback the inventory the load's own builder makes of the header: one call, one tensor.
// Host-only: no backend is initialised, and no device or model is touched.

#include "sycl-plan-dry-run-args.h"

// The tool's header read, an undecorated libllama internal reached the way tests/test-llama-lazy-range-prefill.cpp
// reaches the loader's.
#include "../src/llama-model.h"
#include "ggml-sycl.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <mutex>
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

std::mutex  g_log_mutex;
std::string g_log;

void capture_log(ggml_log_level level, const char * text, void * user) {
    (void) level;
    (void) user;
    if (text != nullptr) {
        std::fputs(text, stderr);
        std::lock_guard<std::mutex> lock(g_log_mutex);
        g_log += text;
    }
}

// What the header read handed its callback: how often it was called, and the last inventory's size and first tensor.
struct header_seen {
    int         calls = 0;
    size_t      count = 0;
    std::string first_name;
    size_t      first_size = 0;
};

void record_inventory(const ggml_sycl_tensor_inventory *   inventory,
                      const ggml_sycl_placement_envelope * envelope,
                      void *                               user_data) {
    (void) envelope;
    header_seen & seen = *static_cast<header_seen *>(user_data);
    ++seen.calls;
    seen.count = inventory != nullptr ? inventory->count : 0;
    if (seen.count > 0 && inventory->tensors != nullptr) {
        seen.first_name = inventory->tensors[0].name != nullptr ? inventory->tensors[0].name : "";
        seen.first_size = inventory->tensors[0].size;
    }
}

constexpr int64_t TENSOR_ELEMENTS = 16;

// Writes `path`: a GGUF whose general.architecture is `arch`, with one zeroed F32 tensor of TENSOR_ELEMENTS. An unknown
// `arch` makes the loader read a complete header and stop at the architecture alone. With `llama_hparams`, it also
// carries the keys the llama loader requires (context length, embedding length, block count, RMS epsilon) and a head
// count, so the header read reaches the inventory builder.
bool write_gguf(const std::string & path, const char * arch, bool llama_hparams) {
    ggml_init_params params = { /*.mem_size =*/1024 * 1024, /*.mem_buffer =*/nullptr, /*.no_alloc =*/false };
    ggml_context *   ctx    = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }
    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, TENSOR_ELEMENTS);
    ggml_set_name(tensor, "token_embd.weight");
    std::memset(tensor->data, 0, ggml_nbytes(tensor));
    gguf_context * gguf = gguf_init_empty();
    gguf_set_val_str(gguf, "general.architecture", arch);
    if (llama_hparams) {
        gguf_set_val_u32(gguf, "llama.context_length", 64);
        gguf_set_val_u32(gguf, "llama.embedding_length", static_cast<uint32_t>(TENSOR_ELEMENTS));
        gguf_set_val_u32(gguf, "llama.block_count", 1);
        gguf_set_val_u32(gguf, "llama.feed_forward_length", 32);
        gguf_set_val_u32(gguf, "llama.attention.head_count", 1);
        gguf_set_val_f32(gguf, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    }
    gguf_add_tensor(gguf, tensor);
    const bool ok = gguf_write_to_file(gguf, path.c_str(), /*only_meta =*/false);
    gguf_free(gguf);
    ggml_free(ctx);
    return ok;
}

// Runs the header read on a GGUF `write_gguf` makes for `arch`, in a scratch directory of the test's own that is
// removed after, capturing libllama's log. Returns false when the file could not be made; `read` is the call's result.
bool header_read(const char * arch, bool llama_hparams, bool & read, header_seen & seen, std::string & log) {
    std::string tmpl = (std::filesystem::temp_directory_path() / "test-sycl-plan-dry-run-args-XXXXXX").string();
    if (mkdtemp(&tmpl[0]) == nullptr) {
        check(false, "the scratch directory could not be made under " + tmpl);
        return false;
    }
    const std::filesystem::path dir     = tmpl;
    const std::string           path    = (dir / "header.gguf").string();
    const bool                  written = write_gguf(path, arch, llama_hparams);
    if (!written) {
        check(false, "the '" + std::string(arch) + "' GGUF could not be written to " + path);
    } else {
        {
            std::lock_guard<std::mutex> lock(g_log_mutex);
            g_log.clear();
        }
        llama_log_set(capture_log, nullptr);
        read = llama_model_sycl_header_inventory(path.c_str(), llama_model_default_params(), record_inventory, &seen);
        llama_log_set(nullptr, nullptr);
        std::lock_guard<std::mutex> lock(g_log_mutex);
        log = g_log;
    }
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    check(!ec && !std::filesystem::exists(dir), "the scratch directory " + dir.string() + " was not removed");
    return written;
}

// The header read on an unknown architecture returns false, calls back with nothing, and logs the loader's refusal
// naming the architecture: the false is the architecture's, not a missing or unreadable file's.
void header_read_refuses_an_unknown_architecture() {
    const char * const arch = "plan-dry-run-no-such-arch";
    bool               read = false;
    header_seen        seen;
    std::string        log;
    if (!header_read(arch, /*llama_hparams =*/false, read, seen, log)) {
        return;
    }
    check(!read, "the header read accepted a GGUF whose architecture is '" + std::string(arch) + "'");
    check(seen.calls == 0,
          "the header read called back " + std::to_string(seen.calls) + " times for an unknown architecture");
    check(log.find("unknown model architecture") != std::string::npos && log.find(arch) != std::string::npos,
          "the header read's refusal does not name the architecture '" + std::string(arch) + "'");
}

// The header read on a known architecture returns true and calls back exactly once, with the inventory the load's
// builder makes of the header: the file's one tensor, by name and byte size.
void header_read_builds_the_inventory_of_a_known_architecture() {
    bool        read = false;
    header_seen seen;
    std::string log;
    if (!header_read("llama", /*llama_hparams =*/true, read, seen, log)) {
        return;
    }
    check(read, "the header read refused a llama GGUF carrying the hparams its loader requires");
    check(seen.calls == 1, "the header read called back " + std::to_string(seen.calls) + " times, want 1");
    check(seen.count == 1, "the inventory holds " + std::to_string(seen.count) + " tensors, want the file's 1");
    check(seen.first_name == "token_embd.weight",
          "the inventory's tensor is '" + seen.first_name + "', want the file's token_embd.weight");
    const size_t want_size = static_cast<size_t>(TENSOR_ELEMENTS) * sizeof(float);
    check(seen.first_size == want_size,
          "the inventory's tensor is " + std::to_string(seen.first_size) + " bytes, want " + std::to_string(want_size));
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

    header_read_refuses_an_unknown_architecture();
    header_read_builds_the_inventory_of_a_known_architecture();

    std::printf("%s: %d failures\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}
