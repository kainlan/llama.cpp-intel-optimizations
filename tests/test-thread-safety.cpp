// thread safety test
// - Loads a copy of the same model on each GPU, plus a copy on the CPU
// - Creates n_parallel (--parallel) contexts per model
// - Runs inference in parallel on each context
//
// SYCL: this test's ctest registration (tests/CMakeLists.txt) sets
// GGML_SYCL_DISABLE_GRAPH=1 in its ENVIRONMENT -- see the comment there for
// why (llama.cpp-b16a: SYCL command-graph replay can hold the execution
// registry's per-device invocation across a context's whole run, which this
// test's deliberate same-device multi-context load trips). A direct
// `./build/bin/test-thread-safety ...` invocation does NOT get that
// ENVIRONMENT (only `ctest -R test-thread-safety` does -- see CLAUDE.md,
// "what the registration provides, direct invocation does not"), so it may
// fail with DEVICE_BUSY where the same run under ctest passes. Set the
// variable by hand for a direct run.

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "test-skip.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

static bool argv_has_model_arg(int argc, char ** argv) {
    static const char * const model_args[] = {
        "-m", "--model", "-mu", "--model-url", "-hf", "-hfr", "--hf-repo", "-hff", "--hf-file", "-dr", "--docker-repo",
    };
    for (int i = 1; i < argc; ++i) {
        for (const char * a : model_args) {
            if (std::strcmp(argv[i], a) == 0) {
                return true;
            }
        }
    }
    return false;
}

int main(int argc, char ** argv) {
    std::vector<char *> argv_owned;
    std::string         env_model_arg;
    if (!argv_has_model_arg(argc, argv)) {
        const char * env_model = std::getenv("LLAMACPP_TEST_MODELFILE");
        if (!env_model || std::strlen(env_model) == 0) {
            test_skip_no_model();
        }
        env_model_arg = env_model;
        argv_owned.reserve(argc + 2);
        argv_owned.push_back(argv[0]);
        argv_owned.push_back(const_cast<char *>("--model"));
        argv_owned.push_back(env_model_arg.data());
        for (int i = 1; i < argc; ++i) {
            argv_owned.push_back(argv[i]);
        }
        argc = (int) argv_owned.size();
        argv = argv_owned.data();
    }

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    LOG_INF("%s\n", common_params_get_system_info(params).c_str());

    //llama_log_set([](ggml_log_level level, const char * text, void * /*user_data*/) {
    //    if (level == GGML_LOG_LEVEL_ERROR) {
    //        common_log_add(common_log_main(), level, "%s", text);
    //    }
    //}, NULL);

    auto cparams = common_context_params_to_llama(params);

    // each context has a single sequence
    cparams.n_seq_max = 1;

    int dev_count = ggml_backend_dev_count();
    std::vector<std::array<ggml_backend_dev_t, 2>> gpus;
    for (int i = 0; i < dev_count; ++i) {
        auto * dev = ggml_backend_dev_get(i);
        if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpus.push_back({dev, nullptr});
        }
    }
    const int gpu_dev_count = (int)gpus.size();
    const int num_models = gpu_dev_count + 1 + 1; // GPUs + 1 CPU model + 1 layer split
    //const int num_models = std::max(1, gpu_dev_count);
    const int num_contexts = std::max(1, params.n_parallel);

    // Same-device concurrent inference is unsupported by the SYCL backend:
    // context-keyed KV/RUNTIME arena ownership does not exist yet, so the
    // execution registry rejects a second concurrent user of a device that
    // already owns one (docs/design/sycl-canonical-memory-architecture.md
    // §5.3, §7.3, §12.3 -- "Same-device concurrent inference: Unsupported
    // today; no optimistic overlap"). Serialize every device-touching phase
    // that this test actually runs concurrently on the same physical SYCL
    // device -- context creation and decode -- so this test exercises the
    // concurrency the backend actually supports -- different devices, and
    // CPU alongside GPU -- rather than asserting an out-of-contract
    // guarantee. This is backend-gated: on any backend other than SYCL the
    // lock set stays empty and every phase runs exactly as before.
    //
    // Context creation is guarded, not just decode, because
    // llama_init_from_model() touches SYCL execution-registry state per
    // device before any graph is built: it creates a ggml_backend per device
    // (ggml_backend_dev_init) and, for a SYCL device, creates/binds a fresh
    // execution-registry context as part of that (see
    // ggml_backend_sycl_execution_context_create/bind_backend, wired up in
    // src/llama-context.cpp) -- this happens on every thread's very first
    // device touch, unguarded, if only decode is locked.
    //
    // Model load (llama_model_load_from_file(), below) is deliberately NOT
    // guarded: this test's load loop is sequential in the main thread and
    // every load completes before any thread is spawned, so by the time a
    // load's lock would release there is no other load/init/decode call for
    // it to contend with -- a guard there would be dead machinery that
    // invites false confidence rather than protection. If this test is ever
    // restructured to load models concurrently (or from a thread pool),
    // that load would need the same guard as context creation below,
    // including its deferred-release synchronize -- which model load has no
    // API to perform: llama_synchronize() is context-scoped, and no
    // lower-level ggml-sycl.h entry point drains a device's SYCL work given
    // only a model/device index.
    const bool sycl_same_device_lock_needed =
        gpu_dev_count > 0 && std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(gpus[0][0])), "SYCL") == 0;
    std::vector<std::mutex> device_exec_mutex(sycl_same_device_lock_needed ? gpu_dev_count : 0);

    // Devices touched by model m's init/decode calls: a single-GPU model
    // owns just its one device, the CPU model owns none, and the trailing
    // layer-split model spans every GPU device. Locking every GPU for the
    // layer-split model's whole decode over-serializes it against the
    // single-GPU models (it forfeits concurrency it could in principle have
    // on devices its current decode isn't using yet) -- that's intentionally
    // conservative, not a bug to "optimize" back toward per-device-only
    // locking, since a multi-device invocation's actual device set for a
    // given decode isn't observable from here.
    auto model_devices = [&](int m) {
        std::vector<int> devices;
        if (m < gpu_dev_count) {
            devices.push_back(m);
        } else if (m == num_models - 1) {
            for (int d = 0; d < gpu_dev_count; ++d) {
                devices.push_back(d);
            }
        }
        return devices;
    };

    // RAII guard that locks every device an init/decode call touches, in
    // ascending device-index order. That order is the same for every thread,
    // so lock acquisition can't cycle and no separate deadlock-avoidance is
    // needed.
    struct device_exec_guard {
        std::vector<std::unique_lock<std::mutex>> locks;

        device_exec_guard(std::vector<std::mutex> & mutexes, const std::vector<int> & devices) {
            for (int d : devices) {
                locks.emplace_back(mutexes[d]);
            }
        }
    };

    std::vector<llama_model_ptr> models;
    std::vector<std::thread> threads;
    std::atomic<bool> failed = false;

    for (int m = 0; m < num_models; ++m) {
        auto mparams = common_model_params_to_llama(params);

        if (m < gpu_dev_count) {
            mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
            mparams.devices = gpus[m].data();
        } else if (m == gpu_dev_count) {
            mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
            mparams.main_gpu = -1; // CPU model
        } else {
            mparams.split_mode = LLAMA_SPLIT_MODE_LAYER;
        }

        // Not guarded -- see the header comment above device_exec_mutex for
        // why this loop's sequential, single-threaded shape makes a lock
        // here dead machinery today.
        llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
        if (model == NULL) {
            LOG_ERR("%s: failed to load model '%s'\n", __func__, params.model.path.c_str());
            return 1;
        }

        models.emplace_back(model);
    }

    for  (int m = 0; m < num_models; ++m) {
        auto * model = models[m].get();
        for (int c = 0; c < num_contexts; ++c) {
            threads.emplace_back([&, m, c, model]() {
                LOG_INF("Creating context %d/%d for model %d/%d\n", c + 1, num_contexts, m + 1, num_models);

                const std::vector<int> devices = sycl_same_device_lock_needed ? model_devices(m) : std::vector<int>{};

                llama_context_ptr ctx;
                {
                    // See the header comment above device_exec_mutex:
                    // llama_init_from_model() touches SYCL execution-registry
                    // state per device before any graph is built, so guard it
                    // the same as decode below. Force the deferred-release
                    // settle point (see the decode guard's comment) before
                    // unlocking -- this is the only chance to do so before
                    // the lock opens this device up to another thread.
                    device_exec_guard guard(device_exec_mutex, devices);
                    ctx.reset(llama_init_from_model(model, cparams));
                    if (sycl_same_device_lock_needed) {
                        if (ctx != NULL) {
                            llama_synchronize(ctx.get());
                        }
                        // else: llama_init_from_model() failed, so there is no
                        // llama_context to synchronize. llama_synchronize() is
                        // the only drain primitive SYCL exposes and it is
                        // context-scoped; no lower-level ggml-sycl.h entry
                        // point can drain a device's SYCL work given only a
                        // model/device index (same limitation documented on
                        // model load above). Whatever a failed init call left
                        // in flight is not drained by this guard.
                    }
                }
                if (ctx == NULL) {
                    LOG_ERR("failed to create context\n");
                    failed.store(true);
                    return;
                }

                std::unique_ptr<common_sampler, decltype(&common_sampler_free)> sampler { common_sampler_init(model, params.sampling), common_sampler_free };
                if (sampler == NULL) {
                    LOG_ERR("failed to create sampler\n");
                    failed.store(true);
                    return;
                }

                llama_batch batch = {};
                {
                    auto prompt = common_tokenize(ctx.get(), params.prompt, true);
                    if (prompt.empty()) {
                        LOG_ERR("failed to tokenize prompt\n");
                        failed.store(true);
                        return;
                    }
                    batch = llama_batch_get_one(prompt.data(), prompt.size());
                    device_exec_guard guard(device_exec_mutex, devices);
                    const int         prompt_decode_rc = llama_decode(ctx.get(), batch);
                    if (sycl_same_device_lock_needed) {
                        // llama_decode() can return before the SYCL backend
                        // actually releases the device's execution lease
                        // (deferred-exit-wait path, ggml-sycl.cpp
                        // can_defer_exit_wait): the lease is released lazily on
                        // the next synchronize, which is normally triggered
                        // later by a logit read. Force that release here, still
                        // under the lock, so we never unlock a device this
                        // thread's invocation still owns in the execution
                        // registry. Gated so non-SYCL backends keep the async
                        // decode overlap this stress test is meant to exercise.
                        // Unconditional on prompt_decode_rc: a failed decode
                        // can still have partially submitted work, and this is
                        // the last chance to drain it before the guard below
                        // releases the lock.
                        llama_synchronize(ctx.get());
                    }
                    if (prompt_decode_rc) {
                        LOG_ERR("failed to decode prompt\n");
                        failed.store(true);
                        return;
                    }
                }

                const auto * vocab = llama_model_get_vocab(model);
                std::string result = params.prompt;

                for (int i = 0; i < params.n_predict; i++) {
                    llama_token token;
                    if (batch.n_tokens > 0) {
                        token = common_sampler_sample(sampler.get(), ctx.get(), batch.n_tokens - 1);
                    } else {
                        token = llama_vocab_bos(vocab);
                    }

                    result += common_token_to_piece(ctx.get(), token);

                    if (llama_vocab_is_eog(vocab, token)) {
                        break;
                    }

                    batch = llama_batch_get_one(&token, 1);

                    int ret;
                    {
                        device_exec_guard guard(device_exec_mutex, devices);
                        ret = llama_decode(ctx.get(), batch);
                        if (sycl_same_device_lock_needed) {
                            // See the matching comment on the prompt decode
                            // above: force the deferred device-lease release to
                            // happen before we unlock, not on the next
                            // (unguarded) logit read at the top of the
                            // following iteration. Unconditional on ret: a
                            // failed decode can still have partially submitted
                            // work, and this is the last chance to drain it
                            // before the guard releases the lock.
                            llama_synchronize(ctx.get());
                        }
                    }
                    if (ret == 1 && i > 0) {
                        LOG_INF("Context full, stopping generation.\n");
                        break;
                    }

                    if (ret != 0) {
                        LOG_ERR("Model %d/%d, Context %d/%d: failed to decode\n", m + 1, num_models, c + 1, num_contexts);
                        failed.store(true);
                        return;
                    }
                }

                LOG_INF("Model %d/%d, Context %d/%d: %s\n\n", m + 1, num_models, c + 1, num_contexts, result.c_str());

                llama_synchronize(ctx.get());
            });
        }
    }

    for (auto & thread : threads) {
        thread.join();
    }

    if (failed) {
        LOG_ERR("One or more threads failed.\n");
        return 1;
    }

    LOG_INF("All threads finished without errors.\n");
    return 0;
}
