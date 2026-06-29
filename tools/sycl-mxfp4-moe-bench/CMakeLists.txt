set(TARGET sycl-mxfp4-moe-bench)
add_executable(${TARGET}
    main.cpp
    tg_bench_common.cpp
    route_baseline.cpp
    route_prepack.cpp
    route_row_parallel.cpp
    route_fused_layer.cpp
    route_launch.cpp
    route_host_bounce.cpp
)

target_link_libraries(${TARGET} PRIVATE llama-common ggml)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}/ggml/include
    ${PROJECT_SOURCE_DIR}/ggml/src
)

if (GGML_SYCL)
    target_compile_options(${TARGET} PRIVATE "-fsycl" "-Wno-narrowing" "-fsycl-device-code-split=per_kernel")
    target_link_options(${TARGET} PRIVATE "-fsycl" "-fsycl-device-code-split=per_kernel")
    target_compile_definitions(${TARGET} PRIVATE GGML_SYCL_WARP_SIZE=32)
endif()

if (LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
