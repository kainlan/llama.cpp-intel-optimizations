# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# Intel's SYCL link step records transient icpx/ocloc objects (written under
# $TMPDIR) in the linker depfile and then deletes them. Ninja sees a dependency
# that no longer exists and relinks the target on every build, forever, so the
# tree never reaches a no-op build. Same remedy as
# ggml/src/ggml-sycl/CMakeLists.txt:21-26, set here at directory scope so it is
# inherited by every tool subdirectory added below -- sycl-source-line-probe,
# sycl-kernel-bench (which also builds sycl-mxfp4-source-line-probe),
# sycl-mxfp4-moe-bench and onednn-woq-repro all link with icpx and were all
# permanently dirty. See llama.cpp-9ubg.
if (CMAKE_GENERATOR STREQUAL "Ninja" AND CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    set(CMAKE_CXX_LINK_DEPENDS_USE_LINKER FALSE)
endif()

# tools

if (EMSCRIPTEN)
else()
    add_subdirectory(batched-bench)
    add_subdirectory(gguf-split)
    add_subdirectory(imatrix)
    add_subdirectory(llama-bench)
    add_subdirectory(completion)
    add_subdirectory(perplexity)
    add_subdirectory(quantize)
    if (LLAMA_BUILD_SERVER)
        add_subdirectory(ui)
        add_subdirectory(cli)
        add_subdirectory(server)
    endif()
    add_subdirectory(tokenize)
    add_subdirectory(parser)
    add_subdirectory(tts)
    add_subdirectory(mtmd)
    if (GGML_RPC)
        add_subdirectory(rpc)
    endif()
    if (NOT GGML_BACKEND_DL AND GGML_CPU)
        # these tools use backends directly (no dynamic loading) and depend on CPU backend symbols
        add_subdirectory(cvector-generator)
        add_subdirectory(export-lora)
    endif()
    if (GGML_SYCL AND NOT GGML_BACKEND_DL)
        # These are SYCL development benchmarks with no role in a
        # GGML_BACKEND_DL runtime image, and sycl-kernel-bench consumes symbols
        # internal to the ggml-sycl MODULE (ggml_sycl_info,
        # ggml_sycl_get_device_id_from_queue, the ggml_sycl::*_bench_launch
        # family). Under DL those live only inside the loadable plugin, so such
        # a target configures cleanly and then dies at LINK on dozens of
        # undefined references -- the nightly Docker intel failure. Guarded as a
        # whole block rather than per-target so a newly added SYCL tool cannot
        # silently re-break the nightly, matching the fork test tail in
        # ggml/src/ggml-sycl/CMakeLists.txt. See llama.cpp-o6tf.
        add_subdirectory(sycl-kernel-bench)
        add_subdirectory(sycl-source-line-probe)
        add_subdirectory(sycl-mxfp4-moe-bench)
        add_subdirectory(onednn-woq-repro)
    endif()
    add_subdirectory(fit-params)
    add_subdirectory(results)
endif()
