#pragma once

// Shared fixture of test-llama-lazy-range-record and test-llama-lazy-range-prefill
// (llama.cpp-5efe C1): a model of five tensors in two GGUF files whose data
// regions are sparse holes, so that tables of several GiB cost no disk.
//
//   file 0: lz_plus1 (4 GiB + 1 byte), lz_exact (4 GiB), plain (F32 x16)
//   file 1: lz_sparse (6 GiB), per_layer_token_embd.weight (5 GiB)
//
// The offsets are the GGUF's own, read back through the same llama_tensor_weight
// the loader builds its weights_map from. Host-only: no backend, device or model.

#include "../src/llama-lazy-range-record.h"
#include "../src/llama-model-loader.h"
#include "ggml.h"
#include "gguf.h"

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct lazy_fixture {
    using weights_t =
        std::map<std::string, llama_model_loader::llama_tensor_weight, llama_model_loader::weight_name_comparer>;

    static constexpr size_t GIB = 1024ull * 1024 * 1024;

    std::vector<std::string>                 paths;
    std::vector<std::unique_ptr<llama_file>> files;
    std::vector<gguf_context *>              ggufs;
    std::vector<ggml_context *>              ctxs;
    weights_t                                weights;
    std::filesystem::path                    dir;

    struct spec {
        const char * name;
        ggml_type    type;
        int64_t      n;
    };

    lazy_fixture() {
        char tmpl[] = "/tmp/lazy-range-fixture-XXXXXX";
        dir         = mkdtemp(tmpl);

        add_file({
            { "lz_plus1", GGML_TYPE_I8,  (int64_t) (4 * GIB + 1) },
            { "lz_exact", GGML_TYPE_I8,  (int64_t) (4 * GIB)     },
            { "plain",    GGML_TYPE_F32, 16                      }
        });
        add_file({
            { "lz_sparse",                   GGML_TYPE_I8, (int64_t) (6 * GIB) },
            { "per_layer_token_embd.weight", GGML_TYPE_I8, (int64_t) (5 * GIB) }
        });
    }

    ~lazy_fixture() {
        files.clear();
        for (gguf_context * g : ggufs) {
            gguf_free(g);
        }
        for (ggml_context * c : ctxs) {
            ggml_free(c);
        }
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }

    lazy_fixture(const lazy_fixture &)             = delete;
    lazy_fixture & operator=(const lazy_fixture &) = delete;

    // a find function over this fixture's weights, as the loader's pre-fill gives it
    llama_lazy_range_find_fn finder() const {
        return [this](const std::string & name, llama_lazy_range_source & out) {
            const auto it = weights.find(name);
            if (it == weights.end()) {
                return false;
            }
            out.file_idx = it->second.idx;
            out.offs     = it->second.offs;
            out.size     = ggml_nbytes(it->second.tensor);
            return true;
        };
    }

  private:
    void add_file(const std::vector<spec> & specs) {
        const uint16_t    idx  = (uint16_t) files.size();
        const std::string path = (dir / ("f" + std::to_string(idx) + ".gguf")).string();

        // the tensors the writer sees; their data is never touched
        ggml_init_params ip   = { ggml_tensor_overhead() * (specs.size() + 1), nullptr, true };
        ggml_context *   wctx = ggml_init(ip);
        gguf_context *   w    = gguf_init_empty();
        size_t           data = 0;
        for (const spec & s : specs) {
            ggml_tensor * t = ggml_new_tensor_1d(wctx, s.type, s.n);
            ggml_set_name(t, s.name);
            gguf_add_tensor(w, t);
            data += GGML_PAD(ggml_nbytes(t), gguf_get_alignment(w));
        }
        gguf_write_to_file(w, path.c_str(), /*only_meta =*/true);
        const size_t meta = gguf_get_meta_size(w);
        gguf_free(w);
        ggml_free(wctx);

        // the data region is a hole
        if (truncate(path.c_str(), (off_t) (meta + data)) != 0) {
            std::fprintf(stderr, "truncate failed for %s\n", path.c_str());
            std::exit(2);
        }

        ggml_context *   meta_ctx = nullptr;
        gguf_init_params gp       = { /*no_alloc =*/true, &meta_ctx };
        gguf_context *   g        = gguf_init_from_file(path.c_str(), gp);
        if (g == nullptr) {
            std::fprintf(stderr, "cannot read back %s\n", path.c_str());
            std::exit(2);
        }

        paths.push_back(path);
        files.push_back(std::make_unique<llama_file>(path.c_str(), "rb"));
        ggufs.push_back(g);
        ctxs.push_back(meta_ctx);

        for (ggml_tensor * t = ggml_get_first_tensor(meta_ctx); t != nullptr; t = ggml_get_next_tensor(meta_ctx, t)) {
            weights.emplace(ggml_get_name(t), llama_model_loader::llama_tensor_weight(files.back().get(), idx, g, t));
        }
    }
};
