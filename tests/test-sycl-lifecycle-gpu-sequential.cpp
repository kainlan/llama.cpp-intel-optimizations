#include "ggml-backend.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef G1_HAVE_LEVEL_ZERO
#include <level_zero/ze_api.h>
#endif

namespace {
struct options {
    std::string a, b, a_shared, prompt, run;
    int n_predict = 0;
    int seed = 0;
    float temperature = 0.0f;
};

bool parse(int argc, char ** argv, options & o) {
    if (argc < 3 || argc % 2 == 0) {
        return false;
    }
    for (int i = 1; i < argc; i += 2) {
        const std::string key = argv[i], value = argv[i + 1];
        if (key == "--model-a") o.a = value;
        else if (key == "--model-b") o.b = value;
        else if (key == "--model-a-shared") o.a_shared = value;
        else if (key == "--prompt") o.prompt = value;
        else if (key == "--n-predict") o.n_predict = std::atoi(value.c_str());
        else if (key == "--seed") o.seed = std::atoi(value.c_str());
        else if (key == "--temp") o.temperature = std::strtof(value.c_str(), nullptr);
        else if (key == "--run") o.run = value;
        else return false;
    }
    return !o.a.empty() && !o.b.empty() && !o.a_shared.empty() && !o.prompt.empty() && o.n_predict > 0 &&
           (o.run == "A" || o.run == "B" || o.run == "A,B,A");
}

std::vector<llama_token> infer(const std::string & path, const options & o) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(path.c_str(), mp);
    if (!model) return {};

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int count = -llama_tokenize(vocab, o.prompt.data(), o.prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> prompt(count > 0 ? count : 0);
    if (count <= 0 || llama_tokenize(vocab, o.prompt.data(), o.prompt.size(), prompt.data(), count, true, true) < 0) {
        llama_model_free(model);
        return {};
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = count + o.n_predict + 8;
    cp.n_batch = count;
    cp.n_ubatch = count;
    llama_context * ctx = llama_init_from_model(model, cp);
    // The canonical temperature is zero, so greedy sampling is deterministic;
    // seed remains an explicit part of the launcher contract.
    llama_sampler * sampler = o.temperature == 0.0f ? llama_sampler_init_greedy() : llama_sampler_init_dist((uint32_t) o.seed);
    std::vector<llama_token> result;
    llama_batch batch = llama_batch_get_one(prompt.data(), prompt.size());
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

#ifdef G1_HAVE_LEVEL_ZERO
std::string level_zero_uuid_for_description(const std::string & description, int requested_ordinal) {
    if (zeInit(0) != ZE_RESULT_SUCCESS) return {};
    uint32_t driver_count = 0;
    if (zeDriverGet(&driver_count, nullptr) != ZE_RESULT_SUCCESS || driver_count == 0) return {};
    std::vector<ze_driver_handle_t> drivers(driver_count);
    if (zeDriverGet(&driver_count, drivers.data()) != ZE_RESULT_SUCCESS) return {};
    int ordinal = 0;
    for (ze_driver_handle_t driver : drivers) {
        uint32_t count = 0;
        if (zeDeviceGet(driver, &count, nullptr) != ZE_RESULT_SUCCESS) continue;
        std::vector<ze_device_handle_t> devices(count);
        if (zeDeviceGet(driver, &count, devices.data()) != ZE_RESULT_SUCCESS) continue;
        for (ze_device_handle_t device : devices) {
            ze_device_properties_t props = {};
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            if (zeDeviceGetProperties(device, &props) != ZE_RESULT_SUCCESS) continue;
            const std::string ze_name = props.name;
            const bool name_matches = description.find(ze_name) != std::string::npos || ze_name.find(description) != std::string::npos;
            if (!name_matches || (requested_ordinal >= 0 && ordinal != requested_ordinal)) {
                ++ordinal;
                continue;
            }
            char uuid[37];
            const uint8_t * u = props.uuid.id;
            std::snprintf(uuid, sizeof(uuid),
                          "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                          u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12],
                          u[13], u[14], u[15]);
            return uuid;
        }
    }
    return {};
}
#endif

std::string selected_physical_uuid(const std::string & logical, const std::string & oneapi) {
    size_t sycl_index = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const std::string name = ggml_backend_dev_name(dev) ? ggml_backend_dev_name(dev) : "";
        if (name.rfind("SYCL", 0) != 0) continue;
        const bool selected = logical == name || logical == std::to_string(sycl_index);
        ++sycl_index;
        if (!selected) continue;
#ifdef G1_HAVE_LEVEL_ZERO
        int requested_ordinal = -1;
        const size_t colon = oneapi.rfind(':');
        if (colon != std::string::npos && colon + 1 < oneapi.size()) {
            char * end = nullptr;
            const long parsed = std::strtol(oneapi.c_str() + colon + 1, &end, 10);
            if (end && *end == '\0' && parsed >= 0) requested_ordinal = (int) parsed;
        }
        const char * description = ggml_backend_dev_description(dev);
        return description ? level_zero_uuid_for_description(description, requested_ordinal) : std::string();
#else
        return {};
#endif
    }
    return {};
}

void print_run(const char * label, const std::vector<llama_token> & tokens, bool comma) {
    if (comma) std::printf(",");
    std::printf("{\"model\":\"%s\",\"tokens\":[", label);
    for (size_t i = 0; i < tokens.size(); ++i) std::printf("%s%d", i ? "," : "", tokens[i]);
    std::printf("]}");
}
} // namespace

int main(int argc, char ** argv) {
    options o;
    if (!parse(argc, argv, o)) return 2;
    const char * oneapi = std::getenv("ONEAPI_DEVICE_SELECTOR");
    const char * logical = std::getenv("GGML_SYCL_DEVICE");
    if (!oneapi || !*oneapi || !logical || !*logical) return 77;

    llama_backend_init();
    ggml_backend_load_all();
    const std::string uuid = selected_physical_uuid(logical, oneapi);
    if (uuid.empty()) {
        llama_backend_free();
        return 77;
    }

    std::vector<std::pair<const char *, std::vector<llama_token>>> runs;
    if (o.run == "A") {
        runs.push_back({"A", infer(o.a, o)});
    } else if (o.run == "B") {
        runs.push_back({"B", infer(o.b, o)});
    } else {
        runs.push_back({"A", infer(o.a, o)});
        runs.push_back({"B", infer(o.b, o)});
        runs.push_back({"A", infer(o.a_shared, o)});
    }
    llama_backend_free();
    for (const auto & run : runs) {
        if (run.second.size() != (size_t) o.n_predict) return 1;
    }

    std::printf("{\"device_uuid\":\"%s\",\"runs\":[", uuid.c_str());
    for (size_t i = 0; i < runs.size(); ++i) print_run(runs[i].first, runs[i].second, i != 0);
    std::printf("]}\n");
    return 0;
}
