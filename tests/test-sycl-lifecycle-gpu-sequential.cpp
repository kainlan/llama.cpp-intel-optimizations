#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "llama-ext.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
constexpr const char * k_prompt           = "1, 2, 3, 4, 5,";
constexpr const char * k_oneapi_selector  = "level_zero:0,1";
constexpr const char * k_logical_selector = "1";
constexpr int          k_seed             = 42;
constexpr int          k_n_predict        = 8;

struct options {
    std::string a, b, prompt, run;
    int         n_predict   = 0;
    int         seed        = 0;
    float       temperature = 0.0f;
};

bool parse(int argc, char ** argv, options & o) {
    if (argc < 3 || argc % 2 == 0) {
        return false;
    }
    for (int i = 1; i < argc; i += 2) {
        const std::string key = argv[i], value = argv[i + 1];
        if (key == "--model-a") {
            o.a = value;
        } else if (key == "--model-b") {
            o.b = value;
        } else if (key == "--prompt") {
            o.prompt = value;
        } else if (key == "--n-predict") {
            o.n_predict = std::atoi(value.c_str());
        } else if (key == "--seed") {
            o.seed = std::atoi(value.c_str());
        } else if (key == "--temp") {
            o.temperature = std::strtof(value.c_str(), nullptr);
        } else if (key == "--run") {
            o.run = value;
        } else {
            return false;
        }
    }
    return !o.a.empty() && !o.b.empty() && o.prompt == k_prompt && o.seed == k_seed && o.temperature == 0.0f &&
           o.n_predict == k_n_predict && (o.run == "A" || o.run == "B" || o.run == "A,B,A");
}

std::vector<llama_token> infer(const std::string & path, const options & o, ggml_backend_dev_t selected) {
    ggml_backend_dev_t devices[] = { selected, nullptr };
    llama_model_params mp        = llama_model_default_params();
    mp.devices                   = devices;
    mp.n_gpu_layers              = -1;
    mp.split_mode                = LLAMA_SPLIT_MODE_NONE;
    mp.main_gpu                  = 0;
    llama_model * model          = llama_model_load_from_file(path.c_str(), mp);
    if (!model) {
        return {};
    }
    if (llama_model_n_devices(model) != 1 || llama_model_get_device(model, 0) != selected) {
        llama_model_free(model);
        return {};
    }

    const llama_vocab *      vocab = llama_model_get_vocab(model);
    const int                count = -llama_tokenize(vocab, o.prompt.data(), o.prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> prompt(count > 0 ? count : 0);
    if (count <= 0 || llama_tokenize(vocab, o.prompt.data(), o.prompt.size(), prompt.data(), count, true, true) < 0) {
        llama_model_free(model);
        return {};
    }

    llama_context_params cp          = llama_context_default_params();
    cp.n_ctx                         = count + o.n_predict + 8;
    cp.n_batch                       = count;
    cp.n_ubatch                      = count;
    llama_context *          ctx     = llama_init_from_model(model, cp);
    llama_sampler *          sampler = llama_sampler_init_greedy();
    std::vector<llama_token> result;
    llama_batch              batch = llama_batch_get_one(prompt.data(), prompt.size());
    for (int i = 0; ctx && sampler && i < o.n_predict; ++i) {
        if (llama_decode(ctx, batch) != 0) {
            result.clear();
            break;
        }
        result.push_back(llama_sampler_sample(sampler, ctx, -1));
        batch = llama_batch_get_one(&result.back(), 1);
    }
    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    return result;
}

bool selected_device(ggml_backend_dev_t & backend_device, std::string & uuid) {
    std::vector<ggml_backend_dev_t> backend_devices;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev  = ggml_backend_dev_get(i);
        const char *       name = ggml_backend_dev_name(dev);
        if (name && std::string(name).rfind("SYCL", 0) == 0) {
            backend_devices.push_back(dev);
        }
    }
    if (backend_devices.size() != 1 || std::string(ggml_backend_dev_name(backend_devices[0])) != "SYCL0" ||
        ggml_backend_dev_type(backend_devices[0]) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        return false;
    }
    backend_device          = backend_devices[0];
    uint8_t native_uuid[16] = {};
    if (!ggml_backend_sycl_get_device_uuid(backend_device, native_uuid)) {
        return false;
    }
    char text[37];
    std::snprintf(text, sizeof(text), "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                  native_uuid[0], native_uuid[1], native_uuid[2], native_uuid[3], native_uuid[4], native_uuid[5],
                  native_uuid[6], native_uuid[7], native_uuid[8], native_uuid[9], native_uuid[10], native_uuid[11],
                  native_uuid[12], native_uuid[13], native_uuid[14], native_uuid[15]);
    uuid = text;
    return true;
}

void print_run(const char * label, const std::vector<llama_token> & tokens, bool comma) {
    if (comma) {
        std::printf(",");
    }
    std::printf("{\"model\":\"%s\",\"tokens\":[", label);
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::printf("%s%d", i ? "," : "", tokens[i]);
    }
    std::printf("]}");
}
}  // namespace

int main(int argc, char ** argv) {
    options o;
    if (!parse(argc, argv, o)) {
        return 2;
    }
    const char * oneapi  = std::getenv("ONEAPI_DEVICE_SELECTOR");
    const char * logical = std::getenv("GGML_SYCL_DEVICE");
    if (!oneapi || std::string(oneapi) != k_oneapi_selector || !logical || std::string(logical) != k_logical_selector) {
        return 77;
    }

    llama_backend_init();
    ggml_backend_load_all();
    ggml_backend_dev_t selected = nullptr;
    std::string        uuid;
    if (!llama_supports_gpu_offload() || !selected_device(selected, uuid)) {
        llama_backend_free();
        return 77;
    }

    std::vector<std::pair<const char *, std::vector<llama_token>>> runs;
    if (o.run == "A") {
        runs.push_back({ "A", infer(o.a, o, selected) });
    } else if (o.run == "B") {
        runs.push_back({ "B", infer(o.b, o, selected) });
    } else {
        runs.push_back({ "A", infer(o.a, o, selected) });
        runs.push_back({ "B", infer(o.b, o, selected) });
        runs.push_back({ "A", infer(o.a, o, selected) });
    }
    llama_backend_free();
    for (const auto & run : runs) {
        if (run.second.size() != (size_t) o.n_predict) {
            return 1;
        }
    }

    std::printf("{\"device_uuid\":\"%s\",\"runs\":[", uuid.c_str());
    for (size_t i = 0; i < runs.size(); ++i) {
        print_run(runs[i].first, runs[i].second, i != 0);
    }
    std::printf("]}\n");
    return 0;
}
