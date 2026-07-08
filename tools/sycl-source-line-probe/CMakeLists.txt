set(TARGET sycl-source-line-probe)
add_executable(sycl-source-line-probe main.cpp)

target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_compile_options(${TARGET} PRIVATE "-fsycl" "-fsycl-device-code-split=per_kernel")
target_link_options(${TARGET} PRIVATE "-fsycl" "-fsycl-device-code-split=per_kernel")

if (GGML_SYCL_PROFILING_DEBUG)
    target_compile_options(${TARGET} PRIVATE
        "-g"
        "-gline-tables-only"
        "-fdebug-info-for-profiling"
        "-fsycl-instrument-device-code")
    target_link_options(${TARGET} PRIVATE "-fsycl-instrument-device-code")
endif()

if (LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
