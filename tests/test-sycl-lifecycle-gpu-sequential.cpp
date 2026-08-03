#include "ggml-backend.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
struct options {
    std::string a, b, a_shared, prompt = "1, 2, 3, 4, 5,";
    int         n_predict = 8;
};

bool parse(int argc, char ** argv, options & o) {
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i], v = argv[i + 1];
        if (k == "--model-a") {
            o.a = v;
        } else if (k == "--model-b") {
            o.b = v;
        } else if (k == "--model-a-shared") {
            o.a_shared = v;
        } else if (k == "--prompt") {
            o.prompt = v;
        } else if (k == "--n-predict") {
            o.n_predict = std::atoi(v.c_str());
        } else if (k == "--seed" || k == "--temp") { /* fixed greedy contract */
        } else {
            return false;
        }
    }
    return !o.a.empty() && !o.b.empty() && !o.a_shared.empty() && o.n_predict > 0;
}

std::vector<llama_token> infer(const std::string & path, const options & o, bool expect_success) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers       = 99;
    llama_model * model   = llama_model_load_from_file(path.c_str(), mp);
    if (!model) {
        if (expect_success) {
            std::fprintf(stderr, "failed to load fixture: %s\n", path.c_str());
        }
        return {};
    }
    const llama_vocab *      vocab = llama_model_get_vocab(model);
    const int                count = -llama_tokenize(vocab, o.prompt.data(), o.prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> prompt(count);
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
        const llama_token token = llama_sampler_sample(sampler, ctx, -1);
        result.push_back(token);
        batch = llama_batch_get_one(&result.back(), 1);
    }
    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    return result;
}
}  // namespace

int main(int argc, char ** argv) {
    options o;
    if (!parse(argc, argv, o)) {
        return 2;
    }
    const char * selector = std::getenv("ONEAPI_DEVICE_SELECTOR");
    const char * logical  = std::getenv("GGML_SYCL_LIFECYCLE_TEST_DEVICE");
    if (!selector || !*selector || !logical || !*logical) {
        return 77;
    }
    std::fprintf(stderr, "[G1] selector=%s logical-device=%s sequence=A,B,failed-C,A\n", selector, logical);
    llama_backend_init();
    ggml_backend_load_all();
    const auto a1 = infer(o.a, o, true);
    const auto b  = infer(o.b, o, true);
    const auto c  = infer(o.b + ".deliberately-missing", o, false);
    const auto a2 = infer(o.a_shared, o, true);
    llama_backend_free();
    if (a1.size() != (size_t) o.n_predict || b.size() != (size_t) o.n_predict || !c.empty() || a1 != a2 || a1 == b) {
        std::fprintf(stderr, "G1 assertion failed: A=%zu B=%zu C=%zu A2=%zu sameA=%d distinctB=%d\n", a1.size(),
                     b.size(), c.size(), a2.size(), a1 == a2, a1 != b);
        return 1;
    }
    std::fprintf(stderr, "[G1] PASS same-device A->B->failed-C->A; fixed greedy tokens restored\n");
    return 0;
}
