if (NOT DEFINED GGML_SYCL_TARGET)
    set(GGML_SYCL_TARGET "INTEL")
endif()

message(STATUS  "GGML_SYCL_TARGET=${GGML_SYCL_TARGET}")

check_cxx_compiler_flag("-fsycl" SUPPORTS_SYCL)

if (DEFINED ENV{ONEAPI_ROOT})
    message(STATUS "Using oneAPI Release SYCL compiler (icpx).")
elseif(SUPPORTS_SYCL)
    message(WARNING "Using open-source SYCL compiler (clang++). Didn't detect ENV {ONEAPI_ROOT}.
        If you expected the oneAPI Release compiler, please install oneAPI & source it, like:
        source /opt/intel/oneapi/setvars.sh")
else()
    message(FATAL_ERROR "C++ compiler lacks SYCL support.")
endif()
message(STATUS "SYCL found")
#todo: AOT

if (CMAKE_GENERATOR STREQUAL "Ninja" AND CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    # Intel's SYCL link step records transient icpx/ocloc objects in the linker
    # depfile. Those files are deleted after linking, so Ninja considers
    # libggml-sycl dirty on every build and repeats the expensive device link.
    #
    # This `set()` is directory-scoped, so it reaches only this directory's
    # targets. Every OTHER directory holding a target that links with icpx
    # needs its own copy: tools/CMakeLists.txt carries it for the tools, and
    # tests/CMakeLists.txt for the four SYCL-linking tests
    # (test-sycl-fattn-onednn-gates, test-sycl-kernel-profiler,
    # test-unified-cache-unpin-event, test-sycl-zone-reset-live-refusal).
    # For a long time only this site had it and those nine targets were
    # relinked on every build, silently. tests/test-sycl-link-depends-coverage.py
    # is the gate that keeps a fifth directory from repeating it.
    # See llama.cpp-9ubg.
    set(CMAKE_CXX_LINK_DEPENDS_USE_LINKER FALSE)
endif()

ggml_add_backend_library(ggml-sycl
                         ggml-sycl.cpp
                         kv-tier-manager.cpp
                         ../../include/ggml-sycl.h
                        )

file(GLOB   GGML_HEADERS_SYCL "*.hpp")

# Non-DL builds preserve the optimized CPU traits and full graph fallback.
# DL modules remain completely independent of ggml-cpu.
if (NOT GGML_BACKEND_DL)
    target_link_libraries(ggml-sycl PRIVATE ggml-cpu)
endif()

option(GGML_SYCL_MMQ_XMX "Enable experimental XMX MMQ kernels for SYCL backend" OFF)
option(GGML_SYCL_XMX_GEMM "Enable experimental XMX GEMM for quantized matmul (known to be slower)" OFF)
option(GGML_SYCL_DEBUG_SOA_LAYOUT "Enable SoA layout verification debug checks" OFF)
option(GGML_SYCL_QUANTIZER_DEBUG "Enable quantizer kernel debug output" OFF)
option(GGML_SYCL_ASAN_PARTIAL "Enable ASan for ggml-sycl.cpp and unified-cache.cpp only" OFF)
option(GGML_SYCL_PROFILING_DEBUG "Enable SYCL device debug info for VTune profiling" OFF)

# GGML_SYCL_XMX_GEMM and GGML_SYCL_MMQ_XMX read as two independent options but
# are not separable: the XMX GEMM dispatch sites in ggml-sycl.cpp call
# ggml_sycl_xmx_available() / ggml_sycl_xmx_supports_type(), which are declared
# in mmq_xmx.hpp -- and ggml-sycl.cpp includes that header only under
# GGML_SYCL_MMQ_XMX. Enabling just the first is a hard compile failure whose
# four undeclared-identifier errors name nothing that hints at the missing
# option, so it reads as "the XMX path is broken" rather than "the configure
# line is incomplete". Fail here instead, naming the flag. See llama.cpp-d6d6.
#
# This is a hard error rather than an implicit force-ON because a forced
# `set(... CACHE BOOL ... FORCE)` would rewrite the caller's cache behind their
# back and leave GGML_SYCL_MMQ_XMX stuck ON after GGML_SYCL_XMX_GEMM is turned
# back off. ggml-sycl.cpp carries the same condition as an #error, so a direct
# -DGGML_SYCL_XMX_GEMM that bypasses CMake still gets a named diagnostic.
if (GGML_SYCL_XMX_GEMM AND NOT GGML_SYCL_MMQ_XMX)
    message(FATAL_ERROR
        "GGML_SYCL_XMX_GEMM=ON requires GGML_SYCL_MMQ_XMX=ON.\n"
        "The XMX GEMM dispatch calls ggml_sycl_xmx_available() and "
        "ggml_sycl_xmx_supports_type(), declared in mmq_xmx.hpp, which "
        "ggml-sycl.cpp includes only under GGML_SYCL_MMQ_XMX.\n"
        "Re-run with both, e.g. -DGGML_SYCL_XMX_GEMM=ON -DGGML_SYCL_MMQ_XMX=ON")
endif()

file(GLOB   GGML_SOURCES_SYCL "*.cpp")
list(FILTER GGML_SOURCES_SYCL EXCLUDE REGEX "test_joint_matrix.*\\.cpp$")
list(FILTER GGML_SOURCES_SYCL EXCLUDE REGEX "test_tensor_parallel\\.cpp$")

if (GGML_SYCL_MMQ_XMX)
    message(STATUS "Compiling with experimental GGML_SYCL_MMQ_XMX kernels.")
    target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_MMQ_XMX)
    list(APPEND GGML_SOURCES_SYCL mmq_xmx.cpp)
    list(APPEND GGML_HEADERS_SYCL mmq_xmx.hpp)
endif()

if (GGML_SYCL_DEBUG_SOA_LAYOUT)
    message(STATUS "Compiling with GGML_SYCL_DEBUG_SOA_LAYOUT enabled - will verify SoA layout at runtime.")
    add_compile_definitions(GGML_SYCL_DEBUG_SOA_LAYOUT=1)
endif()

if (GGML_SYCL_QUANTIZER_DEBUG)
    message(STATUS "Compiling with GGML_SYCL_QUANTIZER_DEBUG enabled - will print quantizer d values.")
    add_compile_definitions(GGML_SYCL_QUANTIZER_DEBUG=1)
endif()

target_sources(ggml-sycl PRIVATE ${GGML_HEADERS_SYCL} ${GGML_SOURCES_SYCL})

# P1 lifecycle registry tests are deliberately host-only: model-lifecycle.cpp
# contains no SYCL headers, device calls, or GPU fixture dependency.
if (BUILD_TESTING)
    if (NOT GGML_BACKEND_DL)
        add_executable(test-sycl-cpu-traits-parity
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-cpu-traits-parity.cpp
            cpu-traits-support.cpp)
        target_include_directories(test-sycl-cpu-traits-parity PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
        target_link_libraries(test-sycl-cpu-traits-parity PRIVATE ggml-base ggml-cpu)
        add_test(NAME sycl-cpu-traits-parity COMMAND test-sycl-cpu-traits-parity)
        set_tests_properties(sycl-cpu-traits-parity PROPERTIES LABELS "sycl;cpu;host")
    endif()

    add_executable(test-sycl-lifecycle-load-txn
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-lifecycle-load-txn.cpp
        model-lifecycle.cpp)
    target_include_directories(test-sycl-lifecycle-load-txn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    find_package(Threads REQUIRED)
    target_link_libraries(test-sycl-lifecycle-load-txn PRIVATE Threads::Threads)
    set_target_properties(test-sycl-lifecycle-load-txn PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    add_test(NAME sycl-lifecycle-load-txn COMMAND test-sycl-lifecycle-load-txn)
    add_executable(test-sycl-lifecycle-runtime-host
        tests/test-model-lifecycle-runtime.cpp
        model-lifecycle.cpp)
    target_include_directories(test-sycl-lifecycle-runtime-host PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test-sycl-lifecycle-runtime-host PRIVATE Threads::Threads)
    set_target_properties(test-sycl-lifecycle-runtime-host PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    add_test(NAME sycl-lifecycle-runtime-host COMMAND test-sycl-lifecycle-runtime-host)
    set_tests_properties(sycl-lifecycle-runtime-host PROPERTIES LABELS "sycl;lifecycle;host;tsan-capable")
    add_executable(test-sycl-lifecycle-owner-reset
        tests/test-sycl-lifecycle-owner-reset.cpp
        execution-lifecycle.cpp)
    target_include_directories(test-sycl-lifecycle-owner-reset PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test-sycl-lifecycle-owner-reset PRIVATE Threads::Threads)
    set_target_properties(test-sycl-lifecycle-owner-reset PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    add_test(NAME sycl-lifecycle-owner-reset COMMAND test-sycl-lifecycle-owner-reset)
    add_test(NAME sycl-lifecycle-h6 COMMAND test-sycl-lifecycle-owner-reset --case H6)
    add_test(NAME sycl-lifecycle-h7 COMMAND test-sycl-lifecycle-owner-reset --case H7)
    add_test(NAME sycl-lifecycle-h11 COMMAND test-sycl-lifecycle-owner-reset --case H11)
    add_test(NAME sycl-lifecycle-h13 COMMAND test-sycl-lifecycle-owner-reset --case H13)
    add_test(NAME sycl-lifecycle-m6 COMMAND test-sycl-lifecycle-owner-reset --case M6)
    add_test(NAME sycl-lifecycle-g2 COMMAND test-sycl-lifecycle-owner-reset --case G2)
    set_tests_properties(sycl-lifecycle-owner-reset sycl-lifecycle-h6 sycl-lifecycle-h7 sycl-lifecycle-h11 sycl-lifecycle-h13 sycl-lifecycle-m6 sycl-lifecycle-g2 PROPERTIES LABELS "sycl;execution;lifecycle;host;tsan-capable")
    add_executable(test-sycl-lifecycle-event-lease
        tests/test-sycl-lifecycle-event-lease.cpp
        execution-lifecycle.cpp)
    target_include_directories(test-sycl-lifecycle-event-lease PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test-sycl-lifecycle-event-lease PRIVATE Threads::Threads)
    set_target_properties(test-sycl-lifecycle-event-lease PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    add_test(NAME sycl-lifecycle-event-lease COMMAND test-sycl-lifecycle-event-lease)
    add_test(NAME sycl-lifecycle-g5a COMMAND test-sycl-lifecycle-event-lease --case G5a)
    add_test(NAME sycl-lifecycle-g6 COMMAND test-sycl-lifecycle-event-lease --case G6)
    add_test(NAME sycl-lifecycle-g6b COMMAND test-sycl-lifecycle-event-lease --case G6b)
    add_test(NAME sycl-lifecycle-g7 COMMAND test-sycl-lifecycle-event-lease --case G7)
    add_test(NAME sycl-lifecycle-h8 COMMAND test-sycl-lifecycle-event-lease --case H8)
    add_test(NAME sycl-lifecycle-h12 COMMAND test-sycl-lifecycle-event-lease --case H12)
    add_test(NAME sycl-lifecycle-h12b COMMAND test-sycl-lifecycle-event-lease --case H12b)
    add_test(NAME sycl-lifecycle-h13b COMMAND test-sycl-lifecycle-event-lease --case H13b)
    add_test(NAME sycl-lifecycle-m7 COMMAND test-sycl-lifecycle-event-lease --case M7)
    add_test(NAME sycl-lifecycle-m7b COMMAND test-sycl-lifecycle-event-lease --case M7b)
    set_tests_properties(sycl-lifecycle-event-lease sycl-lifecycle-g5a sycl-lifecycle-g6 sycl-lifecycle-g6b sycl-lifecycle-g7 sycl-lifecycle-h8 sycl-lifecycle-h12 sycl-lifecycle-h12b sycl-lifecycle-h13b sycl-lifecycle-m7 sycl-lifecycle-m7b PROPERTIES LABELS "sycl;execution;lifecycle;host;tsan-capable")
    add_test(NAME sycl-lifecycle-placement-publication
        COMMAND test-sycl-lifecycle-load-txn --case publication-concurrency)
    set_tests_properties(sycl-lifecycle-placement-publication PROPERTIES LABELS "sycl;lifecycle;host;tsan-capable")
    add_executable(test-sycl-lifecycle-public-api
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-lifecycle-public-api.cpp)
    target_include_directories(test-sycl-lifecycle-public-api PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
    add_test(NAME sycl-lifecycle-public-wrappers COMMAND test-sycl-lifecycle-public-api)
    if (GGML_BACKEND_DL)
        add_executable(test-sycl-module-dlopen
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-module-dlopen.cpp)
        target_compile_definitions(test-sycl-module-dlopen PRIVATE
            GGML_SYCL_RUNTIME_MODULE="$<TARGET_FILE:ggml-sycl>")
        target_link_libraries(test-sycl-module-dlopen PRIVATE ${CMAKE_DL_LIBS})
        add_dependencies(test-sycl-module-dlopen ggml-sycl)
        add_test(NAME sycl-module-dlopen-now COMMAND test-sycl-module-dlopen)
        set_tests_properties(sycl-module-dlopen-now PROPERTIES LABELS "sycl;module;host")
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
        add_test(NAME sycl-module-no-cpu-dependency
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-module-dependencies.py
                    $<TARGET_FILE:ggml-sycl>)
        set_tests_properties(sycl-module-no-cpu-dependency PROPERTIES LABELS "sycl;module;host")
        add_test(NAME sycl-module-nodelete-source
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-module-nodelete-source.py)
        set_tests_properties(sycl-module-nodelete-source PROPERTIES LABELS "sycl;module;host")
    endif()

    add_executable(test-sycl-lifecycle-runtime-wrapper
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-lifecycle-runtime-wrapper.cpp)
    target_include_directories(test-sycl-lifecycle-runtime-wrapper PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
    if (GGML_BACKEND_DL)
        set(GGML_SYCL_RENAMED_RUNTIME_MODULE
            "$<TARGET_FILE_DIR:ggml-sycl>/ggml-sycl-lifecycle-renamed$<TARGET_FILE_SUFFIX:ggml-sycl>")
        add_custom_command(TARGET ggml-sycl POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "$<TARGET_FILE:ggml-sycl>" "${GGML_SYCL_RENAMED_RUNTIME_MODULE}"
            COMMENT "Creating renamed SYCL lifecycle DSO fixture")
        target_compile_definitions(test-sycl-lifecycle-runtime-wrapper PRIVATE
            GGML_SYCL_RUNTIME_MODULE="$<TARGET_FILE:ggml-sycl>"
            GGML_SYCL_RENAMED_RUNTIME_MODULE="${GGML_SYCL_RENAMED_RUNTIME_MODULE}")
        target_link_libraries(test-sycl-lifecycle-runtime-wrapper PRIVATE ggml ${CMAKE_DL_LIBS})
        add_dependencies(test-sycl-lifecycle-runtime-wrapper ggml-sycl)
    else()
        # The wrapper exercises both generic registry APIs and direct static
        # SYCL APIs. Link the aggregate target once: it owns the registry TUs
        # and propagates the selected SYCL/CPU backend libraries without
        # introducing duplicate backend objects.
        target_link_libraries(test-sycl-lifecycle-runtime-wrapper PRIVATE ggml)
    endif()
    add_test(NAME sycl-lifecycle-runtime-wrapper COMMAND test-sycl-lifecycle-runtime-wrapper)
    set_tests_properties(sycl-lifecycle-runtime-wrapper PROPERTIES RUN_SERIAL TRUE)
    add_executable(test-sycl-lifecycle-wrapper-overlap
        tests/test-lifecycle-wrapper-overlap.cpp)
    target_include_directories(test-sycl-lifecycle-wrapper-overlap PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
    if (GGML_BACKEND_DL)
        target_compile_definitions(test-sycl-lifecycle-wrapper-overlap PRIVATE
            GGML_SYCL_RUNTIME_MODULE="$<TARGET_FILE:ggml-sycl>")
        target_link_libraries(test-sycl-lifecycle-wrapper-overlap PRIVATE ggml ${CMAKE_DL_LIBS})
        add_dependencies(test-sycl-lifecycle-wrapper-overlap ggml-sycl)
    else()
        target_link_libraries(test-sycl-lifecycle-wrapper-overlap PRIVATE ggml-sycl ggml-cpu)
    endif()
    add_test(NAME sycl-lifecycle-wrapper-overlap COMMAND test-sycl-lifecycle-wrapper-overlap)
    set_tests_properties(sycl-lifecycle-wrapper-overlap PROPERTIES RUN_SERIAL TRUE LABELS "sycl;lifecycle;host")

    # Public UUID ABI smoke test is host-only: null-device behavior needs no GPU.
    # Exercise both direct static linkage and registry lookup from a DL module.
    add_executable(test-sycl-device-uuid-api
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-device-uuid-api.cpp)
    target_include_directories(test-sycl-device-uuid-api PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
    if (GGML_BACKEND_DL)
        target_compile_definitions(test-sycl-device-uuid-api PRIVATE
            GGML_SYCL_RUNTIME_MODULE="$<TARGET_FILE:ggml-sycl>")
        target_link_libraries(test-sycl-device-uuid-api PRIVATE ggml ${CMAKE_DL_LIBS})
        add_dependencies(test-sycl-device-uuid-api ggml-sycl)
    else()
        # The static fixture calls registry/device APIs implemented by ggml-base
        # directly; do not rely on transitive DSO linkage from ggml-sycl.
        target_link_libraries(test-sycl-device-uuid-api PRIVATE ggml-base ggml-sycl ggml-cpu)
    endif()
    set_target_properties(test-sycl-device-uuid-api PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    add_test(NAME sycl-device-uuid-api COMMAND test-sycl-device-uuid-api)
    set_tests_properties(sycl-device-uuid-api PROPERTIES RUN_SERIAL TRUE LABELS "sycl;api;host")

    foreach(_case inner-failure missing-success wrong-txn depth-underflow depth-overflow cancel rollback-idempotence
                  concurrent-end abort-reuse-aba teardown-reuse-aba concurrent-teardown concurrent-model-teardown
                  runtime-update-teardown load-effect-drain load-effect-abort-drain
                  cache-effect-drain cache-effect-abort-drain
                  planning-effect-drain planning-effect-abort-drain
                  identity-effect-drain identity-effect-abort-drain load-effect-allocation-failure
                  finisher-effect-drain
                  durable-replay waiter-rollover
                  committing-teardown load-effect-failure-quarantine rollback-effect-replay dead-allocation-failure
                  latest-live-restoration busy-deferred-quarantine restore-failure-quarantine effect-failure-quarantine
                  begin-allocation-failure
                  null-output stale-token-after-reuse)
        add_test(NAME sycl-lifecycle-h2-${_case} COMMAND test-sycl-lifecycle-load-txn --case ${_case})
    endforeach()
    foreach(_case slot-exhaustion model-id-overflow load-txn-id-overflow slot-generation-overflow invocation-id-counter-overflow)
        add_test(NAME sycl-lifecycle-h10-${_case} COMMAND test-sycl-lifecycle-load-txn --case ${_case})
    endforeach()
    add_test(NAME sycl-lifecycle-multi-model COMMAND test-sycl-lifecycle-load-txn --case multi-model --case sequential-aba)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    foreach(_mutation M1 M2 M3)
        add_test(NAME sycl-lifecycle-mutation-${_mutation}
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-lifecycle-mutations.py
                    $<TARGET_FILE:test-sycl-lifecycle-load-txn> ${_mutation})
    endforeach()

    add_test(NAME sycl-lifecycle-source-contract
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-lifecycle-source-contract.py)
    add_test(NAME sycl-private-cpu-traits-source
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-private-cpu-traits-source.py)
    add_test(NAME sycl-fallback-propagation
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-fallback-propagation.py)
    add_test(NAME sycl-fallback-exception-safety
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-fallback-exception-safety.py)
endif()

# Enable AVX2 + VNNI for CPU vec_dot kernels in tensor split / CPU offload.
# Applied ONLY to cpu-dispatch.cpp to avoid affecting SYCL device code compilation.
# Note: Arrow Lake does NOT support AVX-512 (removed from consumer chips).
# AVX-VNNI (_mm256_dpbusd_epi32, unsigned x signed) and AVX-VNNIINT8 (_mm256_dpbssd_epi32,
# signed x signed) are available and used for Q4_0/MXFP4 fast-paths.
if (GGML_SYCL_TARGET STREQUAL "INTEL" AND NOT WIN32)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" HAS_AVX2)
    check_cxx_compiler_flag("-mavxvnni" HAS_AVXVNNI)
    check_cxx_compiler_flag("-mavxvnniint8" HAS_AVXVNNIINT8)
    if (HAS_AVX2)
        set(_CPU_DISPATCH_AVX_FLAGS "-mavx2;-mfma")
        if (HAS_AVXVNNI)
            list(APPEND _CPU_DISPATCH_AVX_FLAGS "-mavxvnni")
        endif()
        if (HAS_AVXVNNIINT8)
            list(APPEND _CPU_DISPATCH_AVX_FLAGS "-mavxvnniint8")
        endif()
        set_source_files_properties(cpu-dispatch.cpp PROPERTIES
            COMPILE_OPTIONS "${_CPU_DISPATCH_AVX_FLAGS}")
        message(STATUS "cpu-dispatch.cpp AVX flags: ${_CPU_DISPATCH_AVX_FLAGS}")
    endif()
endif()

if (GGML_SYCL_ASAN_PARTIAL)
    message(STATUS "GGML_SYCL_ASAN_PARTIAL: enabling ASan for ggml-sycl.cpp + unified-cache.cpp")
    set_source_files_properties(ggml-sycl.cpp unified-cache.cpp
        PROPERTIES COMPILE_OPTIONS "-fsanitize=address;-fno-omit-frame-pointer")
    target_link_options(ggml-sycl PRIVATE "-fsanitize=address")
endif()

if (WIN32)
    # To generate a Visual Studio solution, using Intel C++ Compiler for ggml-sycl is mandatory
    if( ${CMAKE_GENERATOR} MATCHES "Visual Studio" AND NOT (${CMAKE_GENERATOR_TOOLSET} MATCHES "Intel C"))
        set_target_properties(ggml-sycl PROPERTIES VS_PLATFORM_TOOLSET "Intel C++ Compiler 2025")
        set(CMAKE_CXX_COMPILER "icx")
        set(CMAKE_CXX_COMPILER_ID "IntelLLVM")
    endif()
endif()

find_package(IntelSYCL)
if (IntelSYCL_FOUND)
    # Use oneAPI CMake when possible
    target_link_libraries(ggml-sycl PRIVATE IntelSYCL::SYCL_CXX)
else()
    # Fallback to the simplest way of enabling SYCL when using intel/llvm nightly for instance
    target_compile_options(ggml-sycl PRIVATE "-fsycl")
    target_link_options(ggml-sycl PRIVATE "-fsycl")
endif()

# Add profiling-friendly compiler flags for VTune kernel name visibility
if (GGML_SYCL_PROFILING_DEBUG)
    message(STATUS "GGML_SYCL_PROFILING_DEBUG: enabling SYCL profiling debug flags")
    # -fsycl-instrument-device-code: enables profiling tools to map GPU activities to kernel function names
    # -fdebug-info-for-profiling: generates debug info optimized for profilers (line tables, etc.)
    # -gline-tables-only: required for accurate source attribution in VTune
    # Per-kernel device code split improves VTune kernel name mapping when debug info is present.
    target_compile_options(ggml-sycl PRIVATE "-Wno-narrowing" "-g" "-fsycl-device-code-split=per_kernel"
                           "-fsycl-instrument-device-code" "-fdebug-info-for-profiling" "-gline-tables-only")
    target_link_options(ggml-sycl PRIVATE "-fsycl-instrument-device-code")
else()
    target_compile_options(ggml-sycl PRIVATE "-Wno-narrowing")
endif()

# Battlemage (B580/G21) AOT avoids first-run JIT overhead, but requires Intel's
# offline compiler (ocloc). GGML_SYCL_BMG_AOT=AUTO (default) enables AOT when
# ocloc is present (dev hosts with the GPU compute runtime) and falls back to
# JIT-only (spir64) otherwise, so compiler-only environments such as the SYCL CI
# runners (which ship icpx but not ocloc) still link. Set ON/OFF to force.
set(GGML_SYCL_BMG_AOT "AUTO" CACHE STRING "Battlemage AOT: ON, OFF, or AUTO (detect ocloc)")
if (GGML_SYCL_BMG_AOT STREQUAL "AUTO")
    find_program(GGML_SYCL_OCLOC ocloc)
    if (GGML_SYCL_OCLOC)
        set(GGML_SYCL_BMG_AOT_ENABLED ON)
    else()
        set(GGML_SYCL_BMG_AOT_ENABLED OFF)
    endif()
elseif (GGML_SYCL_BMG_AOT)
    set(GGML_SYCL_BMG_AOT_ENABLED ON)
else()
    set(GGML_SYCL_BMG_AOT_ENABLED OFF)
endif()
if (GGML_SYCL_BMG_AOT_ENABLED)
    message(STATUS "ggml-sycl: Battlemage AOT enabled (-fsycl-targets=intel_gpu_bmg_g21,spir64)")
    set(GGML_SYCL_INTEL_TARGETS "intel_gpu_bmg_g21,spir64")
else()
    message(STATUS "ggml-sycl: Battlemage AOT disabled; JIT only (-fsycl-targets=spir64) [ocloc not found or GGML_SYCL_BMG_AOT=OFF]")
    set(GGML_SYCL_INTEL_TARGETS "spir64")
endif()

# Enable SPIR-V split barrier extension for cooperative ESIMD kernels on Arc GPUs
# Split barriers allow non-blocking arrival + deferred wait, improving performance
# over monolithic esimd::barrier() by overlapping compute with synchronization.
# Required for: __spirv_ControlBarrierArriveINTEL / __spirv_ControlBarrierWaitINTEL
# The BMG per-target translator opts are added only when AOT is enabled (the
# intel_gpu_bmg_g21 target only exists in -fsycl-targets in that case).
target_link_options(ggml-sycl PRIVATE
    "SHELL:-Xspirv-translator=spir64 -spirv-ext=+SPV_INTEL_split_barrier")
if (GGML_SYCL_BMG_AOT_ENABLED)
    target_link_options(ggml-sycl PRIVATE
        "SHELL:-Xspirv-translator=intel_gpu_bmg_g21 -spirv-ext=+SPV_INTEL_split_barrier")
endif()

# Link against oneDNN (primary math backend for Intel targets)
set(GGML_SYCL_DNNL 0)
set(GGML_SYCL_MKL_FALLBACK_ENABLED 0)

# Deprecation warning for old GGML_SYCL_DNN flag
if(DEFINED GGML_SYCL_DNN)
    message(DEPRECATION "GGML_SYCL_DNN is deprecated. oneDNN is now the primary backend for Intel targets.")
endif()

# New option for MKL fallback
option(GGML_SYCL_USE_MKL_FALLBACK "Enable oneMKL as optional fallback math backend" OFF)

if (GGML_SYCL_TARGET STREQUAL "INTEL")
    # oneDNN is REQUIRED for Intel targets (primary math backend)
    find_package(DNNL REQUIRED)
    if(DNNL_FOUND)
        if (NOT DEFINED DNNL_GPU_VENDOR)
            set(DNNL_GPU_VENDOR "INTEL")
        endif()

        if("${GGML_SYCL_TARGET}" STREQUAL "${DNNL_GPU_VENDOR}")
            target_link_libraries(ggml-sycl PRIVATE DNNL::dnnl)
            set(GGML_SYCL_DNNL 1)
            get_target_property(CONFIGS DNNL::dnnl IMPORTED_CONFIGURATIONS)
            foreach(CONFIG ${CONFIGS})
                get_target_property(DNNL_LIB DNNL::dnnl IMPORTED_LOCATION_${CONFIG})
                message(STATUS "Found oneDNN (primary): ${DNNL_LIB}")
            endforeach()
        else()
            message(FATAL_ERROR
                "oneDNN must be compiled for INTEL target.
                 llama.cpp: ${GGML_SYCL_TARGET}, oneDNN: ${DNNL_GPU_VENDOR}")
        endif()
    endif()

    # Link TBB when available. cpu-dispatch.cpp uses oneapi::tbb::parallel_for
    # under a __has_include gate, so if the headers are visible at compile time
    # the library must be visible at link time or BFD fails the shared-object
    # link with undefined-reference errors on the tbb::detail::r1 runtime.
    find_package(TBB QUIET)
    if(TBB_FOUND)
        target_link_libraries(ggml-sycl PRIVATE TBB::tbb)
        message(STATUS "Found TBB for ggml-sycl CPU dispatch: ${TBB_DIR}")
    endif()

    # solve_tri.cpp calls oneMKL BLAS directly for batched TRSM, so Intel SYCL
    # builds must link the SYCL BLAS component even when other math paths use oneDNN.
    find_package(MKL REQUIRED)
    target_link_libraries(ggml-sycl PRIVATE MKL::MKL_SYCL::BLAS)
    set(GGML_SYCL_MKL_FALLBACK_ENABLED 1)
    message(STATUS "Found oneMKL BLAS for SYCL solve_tri")
else()
    # For NVIDIA/AMD: oneDNN is optional (they use oneMath with cuBLAS/rocBLAS)
    find_package(DNNL QUIET)
    if(DNNL_FOUND)
        if (NOT DEFINED DNNL_GPU_VENDOR)
            set(DNNL_GPU_VENDOR "INTEL")
        endif()
        if("${GGML_SYCL_TARGET}" STREQUAL "${DNNL_GPU_VENDOR}")
            target_link_libraries(ggml-sycl PRIVATE DNNL::dnnl)
            set(GGML_SYCL_DNNL 1)
            message(STATUS "Found oneDNN for ${GGML_SYCL_TARGET}")
        endif()
    endif()
endif()

# Enable SYCL queue profiling to activate L0 counter-based events.
# The enable_profiling queue property is the SYCL API-correct way to request
# profiling-capable event pools.  On the L0 backend this triggers counter-based
# events (ZE_experimental_event_pool_counter_based) which use GPU hardware
# counters instead of signal-based synchronization, eliminating CPU busy-wait
# in oneDNN's after_exec_hook.  Result: ~15% TG improvement on Arc B580.
# Users can also set UR_L0_USE_DRIVER_COUNTER_BASED_EVENTS=1 before launch.
target_compile_definitions(ggml-sycl PRIVATE DPCT_PROFILING_ENABLED)

target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_DNNL=${GGML_SYCL_DNNL})
target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_MKL_FALLBACK=${GGML_SYCL_MKL_FALLBACK_ENABLED})
if (GGML_SYCL_XMX_GEMM)
    target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_XMX_GEMM)
    message(STATUS "XMX GEMM enabled (experimental, known to be 5-11x slower for quantized models)")
endif()

# Link against oneCCL for tensor parallelism collective operations
set(GGML_SYCL_CCL 0)
option(GGML_SYCL_ONECCL "Enable oneCCL for tensor parallelism collectives" ON)
if(GGML_SYCL_ONECCL)
    find_package(oneCCL QUIET)
    if(oneCCL_FOUND)
        if (TARGET oneCCL)
            target_link_libraries(ggml-sycl PRIVATE oneCCL)
        elseif (TARGET ccl)
            target_link_libraries(ggml-sycl PRIVATE ccl)
        else()
            message(FATAL_ERROR "oneCCL was found but neither imported target 'oneCCL' nor 'ccl' is available")
        endif()
        set(GGML_SYCL_CCL 1)
        message(STATUS "Found oneCCL: enabled for tensor parallelism")
    else()
        message(STATUS "oneCCL not found, tensor parallelism will use manual ALL_REDUCE")
    endif()
else()
    message(STATUS "oneCCL support disabled by the user")
endif()
target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_CCL=${GGML_SYCL_CCL})

if (GGML_SYCL_F16)
    if (GGML_SYCL_TARGET STREQUAL "AMD")
        message(WARNING "AMD target does not entirely support FP16 in the SYCL backend.")
    endif()
    # The directory-scoped add_compile_definitions reaches ggml-sycl's own TUs
    # (this CMakeLists builds the ggml-sycl backend library). The target-scoped
    # PUBLIC definition on `ggml` propagates the flag to every consumer that
    # links ggml — notably src/llama-graph.cpp, which needs it to cast Q at
    # graph-build time via GGML_SYCL_FATTN_Q_TYPE (declared in ggml-sycl.h).
    # Both are required: the first scope serves ggml-sycl code, the second
    # scope serves llama code. They are not redundant.
    add_compile_definitions(GGML_SYCL_F16)
    target_compile_definitions(ggml PUBLIC GGML_SYCL_F16)
endif()

if (GGML_SYCL_TARGET STREQUAL "INTEL")
    # AOT (Battlemage) + JIT fallback, or JIT-only, per the GGML_SYCL_BMG_AOT
    # decision computed above (GGML_SYCL_INTEL_TARGETS). AOT avoids runtime JIT
    # overhead but needs ocloc; JIT-only keeps compiler-only environments building.
    target_compile_options(ggml-sycl PRIVATE "-fsycl-targets=${GGML_SYCL_INTEL_TARGETS}")
    target_link_options(ggml-sycl PRIVATE "-fsycl-targets=${GGML_SYCL_INTEL_TARGETS}")

    # Force WARP_SIZE=32 for Intel to match MMQ tile sizes (32) and fix division-by-zero
    # issues in load_tiles_q4_0. Intel hardware supports subgroup size 32.
    add_compile_definitions(GGML_SYCL_WARP_SIZE=32)
    # Enable stateless addressing for >4GB device allocations (required for VRAM arena).
    # Without this, kernels use 32-bit offsets and hang when reading beyond 4GB.
    # See: https://github.com/intel/compute-runtime/blob/master/programmers-guide/ALLOCATIONS_GREATER_THAN_4GB.md
    target_link_options(ggml-sycl PRIVATE  -Xs   "-cl-intel-greater-than-4GB-buffer-required")
elseif (GGML_SYCL_TARGET STREQUAL "NVIDIA")
    add_compile_definitions(GGML_SYCL_WARP_SIZE=32)
elseif (GGML_SYCL_TARGET STREQUAL "AMD")
    # INFO: Allowed Sub_group_sizes are not consistent through all
    # hip targets. For example, 64 is used for certain models, but the backend
    # does not support it.
    # Target archs tested working: gfx1030, gfx1031, (Only tested sub_group_size = 32)
    add_compile_definitions(GGML_SYCL_WARP_SIZE=32)
else()
    # default for other target
    add_compile_definitions(GGML_SYCL_WARP_SIZE=32)
endif()

if (GGML_SYCL_GRAPH)
    target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_GRAPH)
endif()

# Optional VTune ITT API profiling support
option(GGML_SYCL_PROFILE "Enable VTune ITT API profiling instrumentation" OFF)
if (GGML_SYCL_PROFILE)
    # Find ITT library from oneAPI installation
    if (DEFINED ENV{ONEAPI_ROOT})
        set(ITT_ROOT "$ENV{ONEAPI_ROOT}/vtune/latest" CACHE PATH "Path to VTune installation")
    else()
        set(ITT_ROOT "/opt/intel/oneapi/vtune/latest" CACHE PATH "Path to VTune installation")
    endif()

    find_path(ITT_INCLUDE_DIR ittnotify.h
        PATHS ${ITT_ROOT}/sdk/include ${ITT_ROOT}/include
        NO_DEFAULT_PATH)
    find_library(ITT_LIBRARY NAMES ittnotify
        PATHS ${ITT_ROOT}/sdk/lib64 ${ITT_ROOT}/lib64
        NO_DEFAULT_PATH)

    if (ITT_INCLUDE_DIR AND ITT_LIBRARY)
        target_include_directories(ggml-sycl PRIVATE ${ITT_INCLUDE_DIR})
        target_link_libraries(ggml-sycl PRIVATE ${ITT_LIBRARY})
        target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_PROFILE)
        message(STATUS "VTune ITT profiling enabled: ${ITT_LIBRARY}")
    else()
        message(WARNING "VTune ITT library not found. Profiling disabled.
            ITT_INCLUDE_DIR=${ITT_INCLUDE_DIR}
            ITT_LIBRARY=${ITT_LIBRARY}
            Set ITT_ROOT to your VTune installation path.")
    endif()
endif()

# Link against oneMath for NVIDIA/AMD (Intel uses oneDNN as primary)
if (GGML_SYCL_TARGET STREQUAL "INTEL")
    # Intel uses oneDNN as primary math backend (configured above)
    # MKL is only used if GGML_SYCL_USE_MKL_FALLBACK=ON
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(SYCL_COMPILER ON)
    endif()
    if(GGML_SYCL_MKL_FALLBACK_ENABLED)
        target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_USE_INTEL_ONEMKL)
    endif()
    message(STATUS "Intel SYCL backend: oneDNN primary, MKL fallback=${GGML_SYCL_MKL_FALLBACK_ENABLED}")

else()
    find_package(oneMath QUIET)
    if (NOT oneMath_FOUND)
        message(STATUS "oneMath not found: oneMath will be automatically downloaded")
        # Use FetchContent to automatically pull and build oneMath
        include(FetchContent)
        set(BUILD_FUNCTIONAL_TESTS False)
        set(BUILD_EXAMPLES False)
        set(TARGET_DOMAINS blas)
        if (GGML_SYCL_TARGET STREQUAL "NVIDIA")
            set(ENABLE_MKLCPU_BACKEND False)
            set(ENABLE_MKLGPU_BACKEND False)
            set(ENABLE_CUBLAS_BACKEND True)
        elseif (GGML_SYCL_TARGET STREQUAL "AMD")
            set(ENABLE_MKLCPU_BACKEND False)
            set(ENABLE_MKLGPU_BACKEND False)
            set(ENABLE_ROCBLAS_BACKEND True)
            # Ensure setting a string variable here is not overriden by oneMath CACHE variables
            cmake_policy(SET CMP0126 NEW)
            # Setting the device architecture is only needed and useful for AMD devices in oneMath
            set(HIP_TARGETS ${GGML_SYCL_DEVICE_ARCH} CACHE STRING "oneMath HIP target" FORCE)
        endif()
        FetchContent_Declare(
            ONEMATH
            GIT_REPOSITORY https://github.com/uxlfoundation/oneMath.git
            GIT_TAG 8efe85f5aaebb37f1d8c503b7af66315feabf142
        )
        FetchContent_MakeAvailable(ONEMATH)
        # Create alias to match with find_package targets name
        function(onemath_alias target)
            if (TARGET ${target}_obj)
                # Silence verbose warnings from external libraries
                target_compile_options(${target}_obj PRIVATE -w)
            endif()
            if (TARGET ${target})
                add_library(ONEMATH::${target} ALIAS ${target})
            endif()
        endfunction()
        onemath_alias(onemath)
        onemath_alias(onemath_blas_mklcpu)
        onemath_alias(onemath_blas_mklgpu)
        onemath_alias(onemath_blas_cublas)
        onemath_alias(onemath_blas_rocblas)
    endif()

    # Below oneMath compile-time dispatching is used for better performance
    if (GGML_SYCL_TARGET STREQUAL "NVIDIA")
        target_link_libraries(ggml-sycl PRIVATE ONEMATH::onemath_blas_cublas)
        target_compile_options(ggml-sycl PRIVATE "-fsycl-targets=nvptx64-nvidia-cuda")
        target_link_options(ggml-sycl PRIVATE "-fsycl-targets=nvptx64-nvidia-cuda")
        target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_NVIDIA)
    elseif (GGML_SYCL_TARGET STREQUAL "AMD")
        if (NOT GGML_SYCL_DEVICE_ARCH)
            message(FATAL_ERROR "Can't enable SYCL hip backend, GGML_SYCL_DEVICE_ARCH has not been set.")
        endif()
        target_link_libraries(ggml-sycl PRIVATE ONEMATH::onemath_blas_rocblas)
        target_compile_options(ggml-sycl PRIVATE "-fsycl-targets=amdgcn-amd-amdhsa")
        target_link_options(ggml-sycl PRIVATE "-fsycl-targets=amdgcn-amd-amdhsa")
        target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_AMD)
    else()
        # Fallback to oneMath runtime dispatcher
        target_link_libraries(ggml-sycl PRIVATE ONEMATH::onemath)
        target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_GENERIC)
    endif()
endif()

if (GGML_SYCL_DEVICE_ARCH)
    target_compile_options(ggml-sycl PRIVATE -Xsycl-target-backend --offload-arch=${GGML_SYCL_DEVICE_ARCH})
    target_link_options(ggml-sycl PRIVATE -Xsycl-target-backend --offload-arch=${GGML_SYCL_DEVICE_ARCH})
endif()

# =============================================================================
# XMX ESIMD Tests
# =============================================================================
option(GGML_SYCL_BUILD_XMX_TESTS "Build XMX ESIMD unit tests" OFF)

if (GGML_SYCL_BUILD_XMX_TESTS AND GGML_SYCL_TARGET STREQUAL "INTEL")
    message(STATUS "Building XMX ESIMD unit tests")

    # Common SYCL compile/link options for tests
    set(XMX_TEST_SYCL_OPTIONS
        "-fsycl"
        "-fsycl-targets=intel_gpu_bmg_g21,spir64"
        "-Wno-narrowing"
    )

    # ⚠️ UNIFIED_KERNEL_TEST_STANDALONE IS RETIRED. Do not reintroduce it.
    #
    # Every target below that compiles unified-kernel.cpp used to define it, on
    # the premise that the define made that TU self-contained -- it supplied
    # stub definitions of g_ggml_sycl_debug, g_ggml_sycl_debug_forced_off and
    # ggml_sycl_info() "for symbols normally provided by common.cpp", so a test
    # need not link the backend library.
    #
    # That premise was false. unified-kernel.cpp:25 includes mmvq.hpp, and
    # mmvq.hpp:16 includes common.hpp unconditionally -- sixteen lines BEFORE
    # the #ifdef. common.hpp's real declarations are therefore always in scope,
    # so the stubs were not stale copies that drifted; they were unconditional
    # redefinitions, and no retyping could have made them legal. That is why
    # GGML_SYCL_BUILD_XMX_TESTS=ON failed to configure-and-build at all
    # (llama.cpp-9c0s): three hard errors in the production file, hit by every
    # one of these targets regardless of its own source.
    #
    # The fix is to stop pretending and link the library: no define, and
    # ggml-base + ggml + ggml-sycl on the link line. test-xmx-config below
    # already had exactly this shape.
    #
    # The define was ALSO not just stubs, which is the part worth remembering.
    # It selected different IMPLEMENTATIONS at five further sites --
    # unified-kernel.cpp:7850/7859 used raw queue_.memcpy/memset where
    # production uses mem_copy_async/mem_fill_async over mem_handle, and
    # :9174/:9262 took a raw-pointer deferred-copy path instead of the
    # handle-retaining one. So these tests, had they ever built, would have
    # been exercising a memory path that production does not take, against the
    # one subsystem this fork's canonical-memory contract governs. Linking the
    # real library is not merely cheaper than maintaining stubs; it is the only
    # arrangement in which these tests test the shipped code.

    # Options for the eight targets that compile unified-kernel.cpp as their own
    # TU. This is XMX_TEST_SYCL_OPTIONS plus the two split-barrier translator
    # requests, collected so the eight stay identical.
    #
    # ⚠️ AOT IS REQUIRED HERE. Do not "contain the cost" by dropping the target
    # list, however tempting the arithmetic is -- unified-kernel.cpp is ~12,700
    # lines of SYCL and ocloc-compiling it eight times really does dominate the
    # build. It was tried, and it produces a test binary that builds, links and
    # then dies at run time:
    #
    #   test-esimd-prefetch, test_slm_capacity_max_tile:
    #   sycl::exception: No device image found for external symbol
    #   ggml_sycl_unified::unified_matmul_xmx_kernel_impl<8,16,32>
    #   (thrown from launch_unified_matmul; SIGABRT, core dumped)
    #
    # Measured 2026-08-01 on the B50, JIT vs AOT, same sources and the same
    # libggml-sycl.so: JIT aborts at test 6 of 8; AOT passes 8 of 8. The cause
    # is the mismatch, not JIT as such. These targets link libggml-sycl, whose
    # device images are AOT, while their own would be JIT -- and a device symbol
    # defined in one image is not resolvable from the other. Every target here
    # links that library, so the whole set has to match it.
    #
    # This is also a live hazard in 7881f453d, which relocated two of these
    # tests JIT-only and recorded "NOT YET VERIFIED: neither target has been
    # compiled in this configuration and neither has been run". The
    # configuration does not work.
    set(SYCL_UNIFIED_KERNEL_TEST_OPTIONS ${XMX_TEST_SYCL_OPTIONS})
    set(SYCL_UNIFIED_KERNEL_TEST_LINK_OPTIONS
        ${XMX_TEST_SYCL_OPTIONS}
        "SHELL:-Xspirv-translator=intel_gpu_bmg_g21 -spirv-ext=+SPV_INTEL_split_barrier"
        "SHELL:-Xspirv-translator=spir64 -spirv-ext=+SPV_INTEL_split_barrier"
    )

    # Test 1: XMX Hardware Detection Tests
    # This test is standalone and only needs xmx-esimd-common.hpp
    add_executable(test-xmx-hardware-detect
        tests/test-xmx-hardware-detect.cpp
    )
    target_include_directories(test-xmx-hardware-detect PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )
    target_compile_options(test-xmx-hardware-detect PRIVATE ${XMX_TEST_SYCL_OPTIONS})
    target_link_options(test-xmx-hardware-detect PRIVATE ${XMX_TEST_SYCL_OPTIONS})
    target_compile_definitions(test-xmx-hardware-detect PRIVATE
        GGML_SYCL_WARP_SIZE=32
        GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
        XMX_TEST_STANDALONE=1
    )
    if (IntelSYCL_FOUND)
        target_link_libraries(test-xmx-hardware-detect PRIVATE IntelSYCL::SYCL_CXX)
    endif()

    # Test 2: XMX ESIMD Basic Operations Tests
    # This test is standalone and only needs xmx-esimd-common.hpp
    add_executable(test-xmx-esimd-basic
        tests/test-xmx-esimd-basic.cpp
    )
    target_include_directories(test-xmx-esimd-basic PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )
    target_compile_options(test-xmx-esimd-basic PRIVATE ${XMX_TEST_SYCL_OPTIONS})
    target_link_options(test-xmx-esimd-basic PRIVATE ${XMX_TEST_SYCL_OPTIONS})
    target_compile_definitions(test-xmx-esimd-basic PRIVATE
        GGML_SYCL_WARP_SIZE=32
        GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
        XMX_TEST_STANDALONE=1
    )
    if (IntelSYCL_FOUND)
        target_link_libraries(test-xmx-esimd-basic PRIVATE IntelSYCL::SYCL_CXX)
    endif()

    # Test 6: XMX MoE MXFP4 Fused Kernel Tests
    # Tests the fused MXFP4 MoE GEMM kernel with E8M0 exponents and kvalues LUT
    add_executable(test-xmx-moe-mxfp4
        tests/test-xmx-moe-mxfp4.cpp
    )
    target_include_directories(test-xmx-moe-mxfp4 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )
    target_compile_options(test-xmx-moe-mxfp4 PRIVATE ${XMX_TEST_SYCL_OPTIONS})
    target_link_options(test-xmx-moe-mxfp4 PRIVATE ${XMX_TEST_SYCL_OPTIONS})
    target_compile_definitions(test-xmx-moe-mxfp4 PRIVATE
        GGML_SYCL_WARP_SIZE=32
        GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
        XMX_TEST_STANDALONE=1
    )
    target_link_libraries(test-xmx-moe-mxfp4 PRIVATE ggml-base ggml ggml-sycl)
    if (IntelSYCL_FOUND)
        target_link_libraries(test-xmx-moe-mxfp4 PRIVATE IntelSYCL::SYCL_CXX)
    endif()

    # Register tests with CTest
    include(CTest)
    add_test(NAME xmx-hardware-detect COMMAND test-xmx-hardware-detect)
    add_test(NAME xmx-esimd-basic COMMAND test-xmx-esimd-basic)
    add_test(NAME xmx-moe-mxfp4 COMMAND test-xmx-moe-mxfp4)

    # Test: MoE MXFP4 DP4A vs Scalar Correctness
    # Verifies that DP4A-accelerated MXFP4 dot product matches scalar FP32 reference
    # within Q8_1 quantization tolerance bounds
    add_executable(test-moe-mxfp4-dp4a
        tests/test-moe-mxfp4-dp4a.cpp
    )
    target_include_directories(test-moe-mxfp4-dp4a PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )
    target_compile_options(test-moe-mxfp4-dp4a PRIVATE ${XMX_TEST_SYCL_OPTIONS})
    target_link_options(test-moe-mxfp4-dp4a PRIVATE ${XMX_TEST_SYCL_OPTIONS})
    target_compile_definitions(test-moe-mxfp4-dp4a PRIVATE
        GGML_SYCL_WARP_SIZE=32
        GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
    )
    if (IntelSYCL_FOUND)
        target_link_libraries(test-moe-mxfp4-dp4a PRIVATE IntelSYCL::SYCL_CXX)
    endif()
    add_test(NAME moe-mxfp4-dp4a COMMAND test-moe-mxfp4-dp4a)

    # Test: MXFP4 vectorized per-block dequant equivalence (Track A Task 1)
    # Host-only: compares dequant_mxfp4_block_half8 to the scalar
    # dequant_mxfp4_half byte-exact across 256 qs patterns x 8 exponents.
    add_executable(test-mxfp4-vector-dequant
        tests/test-mxfp4-vector-dequant.cpp
    )
    target_include_directories(test-mxfp4-vector-dequant PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )
    target_compile_options(test-mxfp4-vector-dequant PRIVATE ${XMX_TEST_SYCL_OPTIONS})
    target_link_options(test-mxfp4-vector-dequant PRIVATE ${XMX_TEST_SYCL_OPTIONS})
    target_compile_definitions(test-mxfp4-vector-dequant PRIVATE
        GGML_SYCL_WARP_SIZE=32
        GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
        XMX_TEST_STANDALONE=1
    )
    if (IntelSYCL_FOUND)
        target_link_libraries(test-mxfp4-vector-dequant PRIVATE IntelSYCL::SYCL_CXX)
    endif()
    add_test(NAME mxfp4-vector-dequant COMMAND test-mxfp4-vector-dequant)

    # Set test properties
    set_tests_properties(xmx-hardware-detect xmx-esimd-basic xmx-moe-mxfp4 moe-mxfp4-dp4a mxfp4-vector-dequant PROPERTIES
        LABELS "sycl;xmx"
        TIMEOUT 120
    )

    # These four exit 77 when no SYCL device is available; without SKIP_RETURN_CODE
    # ctest renders that as FAILED rather than Skipped (llama.cpp-ivin).  Listed
    # separately from the block above rather than folded into it because
    # mxfp4-vector-dequant is host-only -- it never returns 77, so claiming it can
    # would be documenting a path that does not exist.
    set_tests_properties(xmx-hardware-detect xmx-esimd-basic xmx-moe-mxfp4 moe-mxfp4-dp4a PROPERTIES
        SKIP_RETURN_CODE 77
    )

    # MoE MXFP4 tests need longer timeout for large matrix tests
    set_tests_properties(xmx-moe-mxfp4 PROPERTIES
        TIMEOUT 300
    )

    # Test 7 (MMQ XMX Dispatch Integration) used to be registered here. Moved
    # outside this guard (see the "un-guarded" section after the endif) because
    # GGML_SYCL_BUILD_XMX_TESTS defaults OFF, so it never built -- same pattern
    # as Test 14/15 above and llama.cpp-5exz generally (llama.cpp-9c0s).

    # Test 8: Unified Kernel Tests
    # Tests the unified matmul kernel architecture with Q4_0 support
    add_executable(test-unified-kernel
        tests/test-unified-kernel.cpp
        unified-kernel.cpp
    )
    target_include_directories(test-unified-kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )
    target_compile_options(test-unified-kernel PRIVATE ${SYCL_UNIFIED_KERNEL_TEST_OPTIONS})
    target_link_options(test-unified-kernel PRIVATE ${SYCL_UNIFIED_KERNEL_TEST_LINK_OPTIONS})
    target_compile_definitions(test-unified-kernel PRIVATE
        GGML_SYCL_WARP_SIZE=32
        GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
    )
    target_link_libraries(test-unified-kernel PRIVATE ggml-base ggml ggml-sycl)
    if (IntelSYCL_FOUND)
        target_link_libraries(test-unified-kernel PRIVATE IntelSYCL::SYCL_CXX)
    endif()
    add_test(NAME unified-kernel COMMAND test-unified-kernel)
    set_tests_properties(unified-kernel PROPERTIES
        LABELS "sycl;unified;matmul"
        TIMEOUT 180
        # Exits 77 when no SYCL device is available; without SKIP_RETURN_CODE
        # ctest renders that as FAILED rather than Skipped (llama.cpp-ivin).
        SKIP_RETURN_CODE 77
    )

    # Test 9: XMX Compute Path Tests
    # Tests the XMX dpas compute path using joint_matrix extensions
    add_executable(test-xmx-compute
        tests/test-xmx-compute.cpp
        unified-kernel.cpp
    )
    target_include_directories(test-xmx-compute PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )
    target_compile_options(test-xmx-compute PRIVATE ${SYCL_UNIFIED_KERNEL_TEST_OPTIONS})
    target_link_options(test-xmx-compute PRIVATE ${SYCL_UNIFIED_KERNEL_TEST_LINK_OPTIONS})
    target_compile_definitions(test-xmx-compute PRIVATE
        GGML_SYCL_WARP_SIZE=32
        GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
    )
    target_link_libraries(test-xmx-compute PRIVATE ggml-base ggml ggml-sycl)
    if (IntelSYCL_FOUND)
        target_link_libraries(test-xmx-compute PRIVATE IntelSYCL::SYCL_CXX)
    endif()
    add_test(NAME xmx-compute COMMAND test-xmx-compute)
    set_tests_properties(xmx-compute PROPERTIES
        LABELS "sycl;xmx;unified;matmul"
        TIMEOUT 300
        # Exits 77 when no SYCL device is available; without SKIP_RETURN_CODE
        # ctest renders that as FAILED rather than Skipped (llama.cpp-ivin).
        SKIP_RETURN_CODE 77
    )

    # Test 10: XMX Default Enable Tests
    # Tests that XMX is enabled by default and uses XMXCapabilities
    add_executable(test-xmx-default-enable
        tests/test-xmx-default-enable.cpp
        unified-kernel.cpp
    )
    target_include_directories(test-xmx-default-enable PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )
    target_compile_options(test-xmx-default-enable PRIVATE ${SYCL_UNIFIED_KERNEL_TEST_OPTIONS})
    target_link_options(test-xmx-default-enable PRIVATE ${SYCL_UNIFIED_KERNEL_TEST_LINK_OPTIONS})
    target_compile_definitions(test-xmx-default-enable PRIVATE
        GGML_SYCL_WARP_SIZE=32
        GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
    )
    target_link_libraries(test-xmx-default-enable PRIVATE ggml-base ggml ggml-sycl)
    if (IntelSYCL_FOUND)
        target_link_libraries(test-xmx-default-enable PRIVATE IntelSYCL::SYCL_CXX)
    endif()
    add_test(NAME xmx-default-enable COMMAND test-xmx-default-enable)
    set_tests_properties(xmx-default-enable PROPERTIES
        LABELS "sycl;xmx;unified;default"
        TIMEOUT 120
        # Exits 77 when no SYCL device is available; without SKIP_RETURN_CODE
        # ctest renders that as FAILED rather than Skipped (llama.cpp-ivin).
        SKIP_RETURN_CODE 77
    )

    # Test 11: XMX Optimization Tests (TDD)
    # Tests double-buffering optimization for XMX kernel
    add_executable(test-xmx-optimization
        tests/test-xmx-optimization.cpp
        unified-kernel.cpp
    )
    target_include_directories(test-xmx-optimization PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )
    target_compile_options(test-xmx-optimization PRIVATE ${SYCL_UNIFIED_KERNEL_TEST_OPTIONS})
    target_link_options(test-xmx-optimization PRIVATE ${SYCL_UNIFIED_KERNEL_TEST_LINK_OPTIONS})
    target_compile_definitions(test-xmx-optimization PRIVATE
        GGML_SYCL_WARP_SIZE=32
        GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
    )
    target_link_libraries(test-xmx-optimization PRIVATE ggml-base ggml ggml-sycl)
    if (IntelSYCL_FOUND)
        target_link_libraries(test-xmx-optimization PRIVATE IntelSYCL::SYCL_CXX)
    endif()
    add_test(NAME xmx-optimization COMMAND test-xmx-optimization)
    set_tests_properties(xmx-optimization PROPERTIES
        LABELS "sycl;xmx;unified;optimization;tdd"
        TIMEOUT 300
        # Exits 77 when no SYCL device is available; without SKIP_RETURN_CODE
        # ctest renders that as FAILED rather than Skipped (llama.cpp-ivin).
        SKIP_RETURN_CODE 77
    )

    # Test 12: XMXConfig struct unit tests
    # Tests hardware-queried dimensions for ESIMD dpas kernel configuration
    add_executable(test-xmx-config
        tests/test-xmx-config.cpp
        unified-kernel.cpp
    )
    target_include_directories(test-xmx-config PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )
    target_compile_options(test-xmx-config PRIVATE ${SYCL_UNIFIED_KERNEL_TEST_OPTIONS})
    target_link_options(test-xmx-config PRIVATE ${SYCL_UNIFIED_KERNEL_TEST_LINK_OPTIONS})
    target_compile_definitions(test-xmx-config PRIVATE
        GGML_SYCL_WARP_SIZE=32
        GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
    )
    target_link_libraries(test-xmx-config PRIVATE ggml-sycl ggml-base ggml)
    if (IntelSYCL_FOUND)
        target_link_libraries(test-xmx-config PRIVATE IntelSYCL::SYCL_CXX)
    endif()
    add_test(NAME xmx-config COMMAND test-xmx-config)
    set_tests_properties(xmx-config PROPERTIES
        LABELS "sycl;xmx;unified;config"
        TIMEOUT 60
    )

    # Test 13: ESIMD Vectorized Dequantization Tests
    # Tests vectorized Q4_0 and Q8_0 dequantization for ESIMD dpas kernels
    add_executable(test-esimd-vectorized-dequant
        tests/test-esimd-vectorized-dequant.cpp
    )
    target_include_directories(test-esimd-vectorized-dequant PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )
    target_compile_options(test-esimd-vectorized-dequant PRIVATE ${XMX_TEST_SYCL_OPTIONS})
    target_link_options(test-esimd-vectorized-dequant PRIVATE ${XMX_TEST_SYCL_OPTIONS})
    target_compile_definitions(test-esimd-vectorized-dequant PRIVATE
        GGML_SYCL_WARP_SIZE=32
        GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
        XMX_TEST_STANDALONE=1
    )
    if (IntelSYCL_FOUND)
        target_link_libraries(test-esimd-vectorized-dequant PRIVATE IntelSYCL::SYCL_CXX)
    endif()
    add_test(NAME esimd-vectorized-dequant COMMAND test-esimd-vectorized-dequant)
    set_tests_properties(esimd-vectorized-dequant PROPERTIES
        LABELS "sycl;xmx;esimd;dequant"
        TIMEOUT 120
        # Exits 77 when no SYCL device is available; without SKIP_RETURN_CODE
        # ctest renders that as FAILED rather than Skipped (llama.cpp-ivin).
        SKIP_RETURN_CODE 77
    )

    # NOTE: Test 14 (kernel-dispatch) and Test 15 (sycl-dispatch-tuning) used to
    # be registered here.  Both were moved outside this guard (see the
    # "un-guarded" section after the endif) because GGML_SYCL_BUILD_XMX_TESTS
    # defaults OFF, so neither ever built -- llama.cpp-5exz.

    # Test 16: ESIMD SLM Prefetch Optimization Tests (Task llama.cpp-attk)
    # Tests LSC prefetch with cache hints for memory/compute overlap
    add_executable(test-esimd-prefetch
        tests/test-esimd-prefetch.cpp
        unified-kernel.cpp
    )
    target_include_directories(test-esimd-prefetch PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )
    target_compile_options(test-esimd-prefetch PRIVATE ${SYCL_UNIFIED_KERNEL_TEST_OPTIONS})
    target_link_options(test-esimd-prefetch PRIVATE ${SYCL_UNIFIED_KERNEL_TEST_LINK_OPTIONS})
    # XMX_TEST_STANDALONE stays: that one is xmx-esimd-common.hpp's own switch
    # and is unrelated to the retired UNIFIED_KERNEL_TEST_STANDALONE.
    target_compile_definitions(test-esimd-prefetch PRIVATE
        GGML_SYCL_WARP_SIZE=32
        GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
        XMX_TEST_STANDALONE=1
    )
    target_link_libraries(test-esimd-prefetch PRIVATE ggml-base ggml ggml-sycl)
    if (IntelSYCL_FOUND)
        target_link_libraries(test-esimd-prefetch PRIVATE IntelSYCL::SYCL_CXX)
    endif()
    add_test(NAME esimd-prefetch COMMAND test-esimd-prefetch)
    set_tests_properties(esimd-prefetch PROPERTIES
        LABELS "sycl;xmx;esimd;prefetch;lsc"
        TIMEOUT 180
        # Exits 77 when no SYCL device is available; without SKIP_RETURN_CODE
        # ctest renders that as FAILED rather than Skipped (llama.cpp-ivin).
        SKIP_RETURN_CODE 77
    )

    # NOTE: the unified-cache / mem_handle / allocator suite used to be
    # registered here -- unified-cache-fast-path, unified-cache-bugs,
    # sycl-layout-choice, sycl-kv-view-resolution, mem-handle-eviction,
    # mem-handle-wrong-device, mem-ops and sycl-runtime-alloc.  None of them is
    # an XMX or ESIMD test and none compiles device-kernel code of its own, yet
    # all sat behind GGML_SYCL_BUILD_XMX_TESTS, which defaults OFF -- so the one
    # subsystem this fork's core design constraint governs had no running tests
    # at all.  They are registered after the endif now; see llama.cpp-5exz.

    # Test 18: UnifiedKernel Operation Tests (TDD)
    # Tests RMS norm and SiLU mul operations for persistent kernel
    #
    # ⚠️ These two are the ONLY tests of the twelve named in llama.cpp-5exz that
    # are still here, and it is a measured result rather than a judgement call.
    # They were moved outside the guard in 7881f453d and the build FAILED, so
    # they were moved back in de1c62fa0.
    #
    # TWO independent breakages, neither caused by the move (llama.cpp-9c0s).
    # ONE of them is now fixed:
    #
    #   1. FIXED.  unified-kernel.cpp could not be compiled standalone at all,
    #      which broke every target here including those whose own source was
    #      fine.  The cause was not stub drift but a false premise -- see the
    #      UNIFIED_KERNEL_TEST_STANDALONE note at the top of this guard.  All
    #      eight unified-kernel.cpp targets now link ggml-sycl instead.
    #
    #   2. OPEN, and it blocks test-unified-kernel-persistent ONLY.
    #      tests/test-unified-kernel-persistent.cpp has rotted against the
    #      production API: "no member named 'type_size' in 'StridedCopyMeta'",
    #      "no member named 'pad'", plus five pointer-vs-const-reference
    #      mismatches where add_set_rows/add_strided_copy now take
    #      `const SetRowsMeta &` / `const StridedCopyMeta &` and the test still
    #      passes a device pointer.  Dereferencing that pointer host-side is NOT
    #      the fix; the meta contract itself changed, so repairing this needs
    #      someone who knows the current one.
    #
    #      It also declares an EXTERNAL ggml_sycl_test_extract_layer_index in
    #      its non-standalone branch, and no such symbol exists anywhere in the
    #      tree -- so that branch has never been compiled either.
    #
    # So GGML_SYCL_BUILD_XMX_TESTS=ON still fails, on this one target and for a
    # reason unrelated to the guard.  Twelve of the thirteen targets build.
    # Do NOT "fix" this by commenting the target out: a test deleted to make a
    # flag green is the exact failure mode llama.cpp-5exz documents.
    add_executable(test-unified-kernel-ops
        tests/test-unified-kernel-ops.cpp
        unified-kernel.cpp
    )
    target_include_directories(test-unified-kernel-ops PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )
    target_compile_options(test-unified-kernel-ops PRIVATE ${SYCL_UNIFIED_KERNEL_TEST_OPTIONS})
    target_link_options(test-unified-kernel-ops PRIVATE ${SYCL_UNIFIED_KERNEL_TEST_LINK_OPTIONS})
    target_compile_definitions(test-unified-kernel-ops PRIVATE
        GGML_SYCL_WARP_SIZE=32
        GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
    )
    target_link_libraries(test-unified-kernel-ops PRIVATE ggml-base ggml ggml-sycl)
    if (IntelSYCL_FOUND)
        target_link_libraries(test-unified-kernel-ops PRIVATE IntelSYCL::SYCL_CXX)
    endif()
    add_test(NAME unified-kernel-ops COMMAND test-unified-kernel-ops)
    set_tests_properties(unified-kernel-ops PROPERTIES
        LABELS "sycl;unified;ops;tdd"
        ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1;LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
        TIMEOUT 180
    )

    # Test 19: UnifiedKernel Persistent Execution Integration Tests
    # Tests the persistent plan-build-execute workflow. See the note above.
    add_executable(test-unified-kernel-persistent
        tests/test-unified-kernel-persistent.cpp
        unified-kernel.cpp
    )
    target_include_directories(test-unified-kernel-persistent PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )
    target_compile_options(test-unified-kernel-persistent PRIVATE ${SYCL_UNIFIED_KERNEL_TEST_OPTIONS})
    target_link_options(test-unified-kernel-persistent PRIVATE ${SYCL_UNIFIED_KERNEL_TEST_LINK_OPTIONS})
    target_compile_definitions(test-unified-kernel-persistent PRIVATE
        GGML_SYCL_WARP_SIZE=32
        GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
    )
    target_link_libraries(test-unified-kernel-persistent PRIVATE ggml-base ggml ggml-sycl)
    if (IntelSYCL_FOUND)
        target_link_libraries(test-unified-kernel-persistent PRIVATE IntelSYCL::SYCL_CXX)
    endif()
    add_test(NAME unified-kernel-persistent COMMAND test-unified-kernel-persistent)
    set_tests_properties(unified-kernel-persistent PROPERTIES
        LABELS "sycl;unified;persistent;tdd"
        ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1;LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
        TIMEOUT 300
    )
endif()

# =============================================================================
# Un-guarded SYCL tests
#
# Everything below this point is registered OUTSIDE the
# GGML_SYCL_BUILD_XMX_TESTS guard above, which defaults OFF.  A test registered
# inside that guard is not merely skipped -- its target is never created, so it
# does not appear in the ctest registry at all, and `ctest -R <name>` reports
# success because it matched zero tests.  That is llama.cpp-5exz.
#
# Two rules for anything added here.
#
# 1. Compile options.  XMX_TEST_SYCL_OPTIONS is defined INSIDE the guard and is
#    unavailable here.  A test with no device kernel code of its own needs only
#    `-fsycl -Wno-narrowing` to link against the prebuilt ggml-sycl library --
#    not the AOT target list -- so use SYCL_LINK_ONLY_TEST_OPTIONS below.  A
#    test that really does compile device kernels (i.e. that pulls in
#    unified-kernel.cpp) belongs inside the guard.
#
# 2. LABELS.  CLAUDE.md's throttled sweep runs
#    `-LE 'residency|mem-handle|cache'`, and -LE is a SUBSTRING regex over the
#    labels.  A label containing any of those three words excludes the test
#    from the only sweep that would run it -- guarded out of the build, then
#    filtered out of the run, two independent layers of invisibility.  So a
#    label here must avoid those substrings unless the test is genuinely a
#    member of the GPU-allocating OOM-hazard family (note "unified-cache"
#    contains "cache" and would be excluded; use "unified").
#
# ENVIRONMENT pins ONEAPI_DEVICE_SELECTOR=level_zero:1 (the B50 validation
# card) on every test here that touches a device.  This is load-bearing, not
# decoration: test-unified-cache-bugs setenv()s itself to level_zero:0 -- the
# B70 benchmark card -- when nothing is set, so an unpinned `ctest` would
# perturb a benchmark in progress.
# =============================================================================

set(SYCL_LINK_ONLY_TEST_OPTIONS
    "-fsycl"
    "-Wno-narrowing"
)

# Batch-size gating logic for ESIMD dispatch. Pure C++: no SYCL headers, no
# device, no GPU allocation -- hence no compile options at all.
add_executable(test-kernel-dispatch
    tests/test-kernel-dispatch.cpp
)
target_include_directories(test-kernel-dispatch PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
target_compile_definitions(test-kernel-dispatch PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
add_test(NAME kernel-dispatch COMMAND test-kernel-dispatch)
set_tests_properties(kernel-dispatch PROPERTIES
    LABELS "sycl;xmx;dispatch;gating"
    TIMEOUT 60
)

# unified_matmul winners map to the orchestrator entrypoint. Host-only.
add_executable(test-sycl-dispatch-tuning
    tests/test-dispatch-tuning.cpp
    dispatch-tuning.cpp
)
target_include_directories(test-sycl-dispatch-tuning PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
target_link_libraries(test-sycl-dispatch-tuning PRIVATE ggml)
add_test(NAME sycl-dispatch-tuning COMMAND test-sycl-dispatch-tuning)
set_tests_properties(sycl-dispatch-tuning PROPERTIES
    LABELS "sycl;dispatch;tuning"
    TIMEOUT 60
)

# KV per-layer slice sizing (llama.cpp-2120): the buffer must be divided by the
# layers that actually HOLD KV, not by the model's total layer count.  Pure
# arithmetic over the real kv_tier_manager -- no device, no allocation, so it is
# safe at any parallelism and safe in a subagent.  It compiles kv-tier-manager.cpp
# directly and stubs only unified_cache_get_layer_vram_bytes(), so it exercises
# the shipped class rather than a mock of it.
#
# It needs -fsycl solely because unified-cache.hpp (for placement_plan) pulls in
# SYCL headers; there is no device kernel here, hence
# SYCL_LINK_ONLY_TEST_OPTIONS per rule 1 above.  ggml-base alone satisfies the
# link (ggml_log_internal); linking ggml would also pull libggml-sycl.so, whose
# own copy of unified_cache_get_layer_vram_bytes would collide with the stub.
#
# NAME: deliberately not "kv-tier-manager".  ctest -R/-E take UNANCHORED
# regexes, so a gate named after the thing it guards lands inside any
# pre-existing exclusion of that name.  Checked against the exclusions in force:
# .github/workflows/* exclude "test-llama-archs", "tokenizer" and
# "test-backend-ops"; CLAUDE.md's throttled sweep excludes labels matching
# residency|mem-handle|cache.  "sycl-kv-slice-sizing" and its labels below hit
# none of them.
add_executable(test-kv-slice-sizing
    tests/test-kv-slice-sizing.cpp
    kv-tier-manager.cpp
)
target_include_directories(test-kv-slice-sizing PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
target_compile_options(test-kv-slice-sizing PRIVATE ${SYCL_LINK_ONLY_TEST_OPTIONS})
target_link_options(test-kv-slice-sizing PRIVATE ${SYCL_LINK_ONLY_TEST_OPTIONS})
target_compile_definitions(test-kv-slice-sizing PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-kv-slice-sizing PRIVATE ggml-base)
add_test(NAME sycl-kv-slice-sizing COMMAND test-kv-slice-sizing)
set_tests_properties(sycl-kv-slice-sizing PROPERTIES
    # No ONEAPI_DEVICE_SELECTOR: this test never creates a queue.  LD_LIBRARY_PATH
    # is still needed -- the binary links the SYCL runtime because of the headers.
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    LABELS "sycl;kv;sizing;hostonly"
    TIMEOUT 60
)

# Tiny model-free packed-K live lifecycle.  This compiles the Intel production
# packed consumer leaf from fattn-xmx-f16-v2.hpp, so its device image must use
# the same AOT/JIT target set and split-barrier options as libggml-sycl.
# Its deterministic incomplete dependency is a native Level Zero event.
find_library(LEVEL_ZERO_LOADER
    NAMES ze_loader
    HINTS
        ENV LEVEL_ZERO_V1_SDK_PATH
        ENV LEVEL_ZERO_SDK_PATH
    PATH_SUFFIXES lib lib/x64
    REQUIRED)
if (GGML_SYCL_TARGET STREQUAL "INTEL" AND NOT GGML_BACKEND_DL)
add_executable(test-fattn-packed-k-lifecycle
    tests/test-fattn-packed-k-lifecycle.cpp
)
target_include_directories(test-fattn-packed-k-lifecycle PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-fattn-packed-k-lifecycle PRIVATE
    "-fsycl"
    "-fsycl-targets=${GGML_SYCL_INTEL_TARGETS}"
    "-Wno-narrowing"
)
target_link_options(test-fattn-packed-k-lifecycle PRIVATE
    "-fsycl"
    "-fsycl-targets=${GGML_SYCL_INTEL_TARGETS}"
    "SHELL:-Xspirv-translator=spir64 -spirv-ext=+SPV_INTEL_split_barrier"
)
if (GGML_SYCL_BMG_AOT_ENABLED)
    target_link_options(test-fattn-packed-k-lifecycle PRIVATE
        "SHELL:-Xspirv-translator=intel_gpu_bmg_g21 -spirv-ext=+SPV_INTEL_split_barrier")
endif()
target_compile_definitions(test-fattn-packed-k-lifecycle PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-fattn-packed-k-lifecycle PRIVATE ggml-base ggml ggml-sycl ${LEVEL_ZERO_LOADER})
if (IntelSYCL_FOUND)
    target_link_libraries(test-fattn-packed-k-lifecycle PRIVATE IntelSYCL::SYCL_CXX)
endif()
foreach(_packed_k_checkpoint IN ITEMS
        sidecar-before-initial-fill
        sidecar-zero-to-update
        sidecar-unregister-overlap
        materializer-zero-to-pack
        packed-first-to-merge)
    add_test(NAME fattn-packed-k-lifecycle-${_packed_k_checkpoint}
        COMMAND test-fattn-packed-k-lifecycle --checkpoint ${_packed_k_checkpoint})
    set_tests_properties(fattn-packed-k-lifecycle-${_packed_k_checkpoint} PROPERTIES
        LABELS "sycl;fattn;lifecycle;live"
        ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1;LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
        SKIP_RETURN_CODE 77
        TIMEOUT 120
    )
endforeach()
endif()

# MMQ XMX dispatch integration: XMX-path selection, MMQ fallback, and edge
# cases (empty batch, non-aligned dims). Moved outside the guard
# (llama.cpp-9c0s) -- it spent its life behind GGML_SYCL_BUILD_XMX_TESTS
# (defaults OFF) and so never built, same as test-kernel-dispatch and
# test-sycl-dispatch-tuning above (llama.cpp-5exz).
#
# It compiles xmx-esimd-common.hpp in XMX_TEST_STANDALONE mode, which under
# that define is pure host-side device-info queries (dev.get_info<...>()) --
# no parallel_for/submit/joint_matrix of its own -- so per rule 1 above it
# needs only SYCL_LINK_ONLY_TEST_OPTIONS, not the AOT target list. It needs a
# GPU: main() calls sycl::device::get_devices(gpu) and exits 77 if none is
# found, so SKIP_RETURN_CODE 77 below is what makes ctest render that as
# *skipped*. It returned 1 until llama.cpp-k208 -- a hard FAILURE on a
# CPU-only runner, which is the mirror image of the exit-0 vacuous pass that
# ticket is named for.
add_executable(test-mmq-xmx-dispatch
    tests/test-mmq-xmx-dispatch.cpp
)
target_include_directories(test-mmq-xmx-dispatch PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
target_compile_options(test-mmq-xmx-dispatch PRIVATE ${SYCL_LINK_ONLY_TEST_OPTIONS})
target_link_options(test-mmq-xmx-dispatch PRIVATE ${SYCL_LINK_ONLY_TEST_OPTIONS})
target_compile_definitions(test-mmq-xmx-dispatch PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
    XMX_TEST_STANDALONE=1
)
if (IntelSYCL_FOUND)
    target_link_libraries(test-mmq-xmx-dispatch PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME mmq-xmx-dispatch COMMAND test-mmq-xmx-dispatch)
set_tests_properties(mmq-xmx-dispatch PROPERTIES
    LABELS "sycl;xmx;dispatch"
    ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1;LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    # The test exits 77 when no GPU is present. Without this, ctest renders that
    # as FAILED rather than the intended Skipped.
    SKIP_RETURN_CODE 77
    TIMEOUT 120
)

# Reader-writer lock optimization on the cache-hit fast path (TDD).
# LABELS drop the old "cache" -- see rule 2 above.
# This fixture compiles unified-cache.cpp directly and relies on the static
# ggml-sycl target to close its full internal symbol graph. A MODULE library is
# not linkable into an executable, so the same partial-TU fixture is invalid in
# DL builds; dynamic lifecycle/module coverage remains registered above.
if (NOT GGML_BACKEND_DL)
add_executable(test-unified-cache-fast-path
    tests/test-unified-cache-fast-path.cpp
    unified-cache.cpp
    model-lifecycle.cpp
    pinned-pool.cpp
)
target_include_directories(test-unified-cache-fast-path PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
target_compile_options(test-unified-cache-fast-path PRIVATE ${SYCL_LINK_ONLY_TEST_OPTIONS})
target_link_options(test-unified-cache-fast-path PRIVATE ${SYCL_LINK_ONLY_TEST_OPTIONS})
target_compile_definitions(test-unified-cache-fast-path PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-unified-cache-fast-path PRIVATE ggml-base ggml)
if (IntelSYCL_FOUND)
    target_link_libraries(test-unified-cache-fast-path PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME unified-cache-fast-path COMMAND test-unified-cache-fast-path)
set_tests_properties(unified-cache-fast-path PROPERTIES
    LABELS "sycl;unified;fast-path;tdd"
    ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1;LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
    # Exits 77 when no SYCL device is available; without SKIP_RETURN_CODE
    # ctest renders that as FAILED rather than Skipped (llama.cpp-ivin).
    SKIP_RETURN_CODE 77
)
endif() # direct-source static SYCL fixture

# Evict accounting, realloc failure handling, unaligned hash.
# LABELS drop the old "cache" -- see rule 2 above.
if (NOT GGML_BACKEND_DL)
add_executable(test-unified-cache-bugs
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-unified-cache-bugs.cpp
)
target_include_directories(test-unified-cache-bugs PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-unified-cache-bugs PRIVATE ${SYCL_LINK_ONLY_TEST_OPTIONS})
target_link_options(test-unified-cache-bugs PRIVATE ${SYCL_LINK_ONLY_TEST_OPTIONS})
target_compile_definitions(test-unified-cache-bugs PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-unified-cache-bugs PRIVATE ggml-base ggml ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-unified-cache-bugs PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME unified-cache-bugs COMMAND test-unified-cache-bugs)
set_tests_properties(unified-cache-bugs PROPERTIES
    LABELS "sycl;unified;bugfix;tdd"
    ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1;LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 180
)

# Synthetic KV/view pointer resolution and handle publication.
# LABELS drop the old "mem-handle" -- see rule 2 above.
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-sycl-kv-view-resolution
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-kv-view-resolution.cpp
)
target_include_directories(test-sycl-kv-view-resolution PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-sycl-kv-view-resolution PRIVATE ${SYCL_LINK_ONLY_TEST_OPTIONS})
target_link_options(test-sycl-kv-view-resolution PRIVATE ${SYCL_LINK_ONLY_TEST_OPTIONS})
target_compile_definitions(test-sycl-kv-view-resolution PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-sycl-kv-view-resolution PRIVATE ggml-base ggml ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-sycl-kv-view-resolution PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME sycl-kv-view-resolution COMMAND test-sycl-kv-view-resolution)
set_tests_properties(sycl-kv-view-resolution PROPERTIES
    LABELS "sycl;kv;view;handle;tdd"
    ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1;LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH};GGML_SYCL_TEST_KV_VIEW_RUNTIME=0"
    TIMEOUT 120
)

# mem_handle eviction lifecycle (RESEARCH-A4, llama.cpp-2jmzn): cache_generation
# bumps + entry removal across the sync and async eviction paths.
# LABELS drop the old "cache" and "mem-handle" -- see rule 2 above.
endif() # static SYCL backend fixture

# Direct internal-TU fixture: static ggml-sycl supplies the remaining runtime,
# watchdog, queue, debug, and placement definitions. Do not manufacture mocks
# for the DL build that would test a different dependency graph.
if (NOT GGML_BACKEND_DL)
add_executable(test-mem-handle-eviction
    tests/test-mem-handle-eviction.cpp
    unified-cache.cpp
    model-lifecycle.cpp
    pinned-pool.cpp
    mem-handle.cpp
)
target_include_directories(test-mem-handle-eviction PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
target_compile_options(test-mem-handle-eviction PRIVATE ${SYCL_LINK_ONLY_TEST_OPTIONS})
target_link_options(test-mem-handle-eviction PRIVATE ${SYCL_LINK_ONLY_TEST_OPTIONS})
target_compile_definitions(test-mem-handle-eviction PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-mem-handle-eviction PRIVATE ggml-base ggml)
if (IntelSYCL_FOUND)
    target_link_libraries(test-mem-handle-eviction PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME mem-handle-eviction COMMAND test-mem-handle-eviction)
set_tests_properties(mem-handle-eviction PROPERTIES
    LABELS "sycl;handle;eviction;tdd"
    ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1;LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
    # Exits 77 when no SYCL device is available; without SKIP_RETURN_CODE
    # ctest renders that as FAILED rather than Skipped (llama.cpp-ivin).
    SKIP_RETURN_CODE 77
)
endif() # direct-source static SYCL fixture

# mem_handle device identity and wrong-device resolution (llama.cpp-32dg8.15.11).
# Single-GPU: tests 1-4 run; multi-device tests skip gracefully.
# LABELS drop the old "cache" and "mem-handle" -- see rule 2 above.
#
# ⚠️ DELIBERATE EXCEPTION to rule 1: this target and test-mem-ops below set no
# target_compile_options/target_link_options at all, unlike their five siblings
# in this section which use SYCL_LINK_ONLY_TEST_OPTIONS. That is inherited
# verbatim from the original guarded registration -- neither had those calls
# even inside the guard with XMX_TEST_SYCL_OPTIONS. It was left alone when the
# tests were relocated (llama.cpp-5exz) rather than normalised, because
# silently "tidying" code while moving it turns a mechanical change into an
# unreviewed behavioural one. Both compile and pass without the options; they
# link ggml-sycl rather than compiling SYCL of their own. Do not add the
# options as a cleanup -- if they are ever needed it will be because something
# real changed, and that deserves its own commit.
if (NOT GGML_BACKEND_DL)
add_executable(test-mem-handle-wrong-device
    tests/test-mem-handle-wrong-device.cpp
)
target_include_directories(test-mem-handle-wrong-device PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_definitions(test-mem-handle-wrong-device PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-mem-handle-wrong-device PRIVATE ggml-base ggml ggml-sycl)
add_test(NAME mem-handle-wrong-device COMMAND test-mem-handle-wrong-device)
set_tests_properties(mem-handle-wrong-device PROPERTIES
    LABELS "sycl;handle;device-identity;tdd"
    # setvars.sh puts compiler/lib before redist/lib on this workstation, but the
    # backend-visible Level Zero devices are exposed via redist/lib.
    ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1;LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)

# Canonical event-returning mem_handle memory operations (llama.cpp-32dg8.15.13).
# LABELS drop the old "cache" and "mem-handle" -- see rule 2 above.
# Sets no compile/link options, same deliberate exception to rule 1 as
# test-mem-handle-wrong-device above; the reasoning is written up there.
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-mem-ops
    tests/test-mem-ops.cpp
)
target_include_directories(test-mem-ops PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_definitions(test-mem-ops PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-mem-ops PRIVATE ggml-base ggml ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-mem-ops PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME mem-ops COMMAND test-mem-ops)
set_tests_properties(mem-ops PROPERTIES
    LABELS "sycl;handle;memory-ops;tdd"
    ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1;LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    # The test exits 77 when no device is present. Without this, ctest renders
    # that as FAILED rather than the intended Skipped -- honest but wrong, and
    # only on a device-less machine, so this host cannot show the difference.
    SKIP_RETURN_CODE 77
    TIMEOUT 120
)

# Runtime allocation policy: reserve/allocate/rollback, tier selection, scoped
# unified_alloc lifetime, offload-pool hit/miss accounting.
endif() # static SYCL backend fixture

# Like the two direct-source cache fixtures above, this target depends on the
# static backend to complete unified-cache.cpp's internal symbol closure.
if (NOT GGML_BACKEND_DL)
add_executable(test-sycl-runtime-alloc
    tests/test-unified-runtime-alloc.cpp
    unified-cache.cpp
    model-lifecycle.cpp
    pinned-pool.cpp
)
target_include_directories(test-sycl-runtime-alloc PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
target_compile_options(test-sycl-runtime-alloc PRIVATE ${SYCL_LINK_ONLY_TEST_OPTIONS})
target_link_options(test-sycl-runtime-alloc PRIVATE ${SYCL_LINK_ONLY_TEST_OPTIONS})
target_compile_definitions(test-sycl-runtime-alloc PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-sycl-runtime-alloc PRIVATE ggml-base ggml)
if (IntelSYCL_FOUND)
    target_link_libraries(test-sycl-runtime-alloc PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME sycl-runtime-alloc COMMAND test-sycl-runtime-alloc)
set_tests_properties(sycl-runtime-alloc PROPERTIES
    LABELS "sycl;allocator;runtime;tdd"
    ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1;LD_LIBRARY_PATH=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}:/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 180
    # Exits 77 when no SYCL device is available; without SKIP_RETURN_CODE
    # ctest renders that as FAILED rather than Skipped (llama.cpp-ivin).
    SKIP_RETURN_CODE 77
)
endif() # direct-source static SYCL fixture

# Test: unified-cache layout and multi-device placement choices, plus the
# fused-gate_up expert-tensor classifier regression guard (llama.cpp-zozu).
#
# Registered OUTSIDE the GGML_SYCL_BUILD_XMX_TESTS guard (which defaults OFF)
# so it always builds -- it spent its life inside that guard and therefore
# never compiled, which is llama.cpp-5exz.  Mirrors the _residency_test loop
# and test-sycl-reset-model-weight-lease-preserve below: this test has no
# device kernel code of its own, so it only needs -fsycl to link against the
# prebuilt ggml-sycl library, not the full XMX AOT target list
# (XMX_TEST_SYCL_OPTIONS is defined inside the guard and is unavailable here).
#
# It needs no GPU by default: main() runs the fused-role guard and then a block
# driven by make_mock_sycl_info() via test_sycl_info_override_guard, and the
# backend section SKIPs unless GGML_SYCL_TEST_LAYOUT_CHOICE_BACKEND=1.
#
# LABELS deliberately DROP the old "cache" label.  CLAUDE.md's throttled sweep
# runs `-LE 'residency|mem-handle|cache'`, so keeping "cache" would have left
# this test excluded from the very sweep it was just un-hidden for -- guarded
# out of the build, then filtered out of the run.  It allocates no GPU memory
# in the default path, so it does not belong in that hazard denylist.
# ENVIRONMENT pins the B50 rather than the B70 benchmark card, matching
# test-sycl-reset-model-weight-lease-preserve, so that if anything does touch a
# device it cannot perturb a benchmark in progress.
set(LAYOUT_CHOICE_TEST_SYCL_OPTIONS
    "-fsycl"
    "-Wno-narrowing"
)
if (NOT GGML_BACKEND_DL)
add_executable(test-sycl-layout-choice
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-layout-choice.cpp
)
target_include_directories(test-sycl-layout-choice PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-sycl-layout-choice PRIVATE ${LAYOUT_CHOICE_TEST_SYCL_OPTIONS})
target_link_options(test-sycl-layout-choice PRIVATE ${LAYOUT_CHOICE_TEST_SYCL_OPTIONS})
target_compile_definitions(test-sycl-layout-choice PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-sycl-layout-choice PRIVATE ggml-base ggml ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-sycl-layout-choice PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME sycl-layout-choice COMMAND test-sycl-layout-choice)
set_tests_properties(sycl-layout-choice PROPERTIES
    LABELS "sycl;layout;planner;tdd"
    ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1;LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 180
)

# Test: mem_handle::resolve() is a MUTATING operation on shared state
# (llama.cpp-c48l).  extra->data_handle[dev] is resolved concurrently by every
# context thread of a model, so a handle that publishes cached_/leased_entry_
# without synchronisation corrupts a sycl::event refcount and double-releases a
# cache-entry lease -- observed as a SIGSEGV inside resolve() itself.
#
# CPU-only by construction: it builds DIRECT/WEIGHT handles by hand and never
# touches a device, a queue, or the unified cache.  It is therefore registered
# OUTSIDE the _residency_test loop below, whose "cache;residency;mem-handle"
# labels are exactly what CLAUDE.md's throttled sweep excludes
# (-LE 'residency|mem-handle|cache').  Putting a GPU-free test behind that
# filter would hide it from the sweep it belongs in; the labels here
# deliberately avoid every word in that pattern.
set(MEM_HANDLE_CONCURRENCY_TEST_SYCL_OPTIONS
    "-fsycl"
    "-Wno-narrowing"
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-sycl-mem-handle-concurrent-resolve
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-mem-handle-concurrent-resolve.cpp
)
target_include_directories(test-sycl-mem-handle-concurrent-resolve PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-sycl-mem-handle-concurrent-resolve PRIVATE ${MEM_HANDLE_CONCURRENCY_TEST_SYCL_OPTIONS})
target_link_options(test-sycl-mem-handle-concurrent-resolve PRIVATE ${MEM_HANDLE_CONCURRENCY_TEST_SYCL_OPTIONS})
target_compile_definitions(test-sycl-mem-handle-concurrent-resolve PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-sycl-mem-handle-concurrent-resolve PRIVATE ggml-base ggml ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-sycl-mem-handle-concurrent-resolve PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-sycl-mem-handle-concurrent-resolve COMMAND test-sycl-mem-handle-concurrent-resolve)
set_tests_properties(test-sycl-mem-handle-concurrent-resolve PROPERTIES
    LABELS "sycl;handle;concurrency;tdd"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)

# Test: retain_handles_until_event() must route a handle by THIS thread's
# recording state (llama.cpp-oze0).  Its guard was
# ggml_sycl_graph_recording_active(), true whenever ANY thread records, while
# the sink that guard selects is thread_local -- so a non-recording thread fell
# through to the process-global graph_unwaitable list, which
# drain_retained_handles() does not wait on.  get_rows' two staging cohorts
# (get_rows:seq_device in VRAM SCRATCH, get_rows_indices_small_host in host
# STAGING) then stayed live across a graph boundary and the zone reset was
# correctly refused, aborting graph launch in test-thread-safety.
#
# CPU-only by construction: DIRECT handle views over stack buffers plus
# default-constructed (already complete) events -- no device, queue or cache.
# Registered outside the _residency_test loop below and labelled to avoid every
# word in CLAUDE.md's throttled-sweep denylist (-LE 'residency|mem-handle|cache'),
# which is about memory-exhaustion hazard rather than subject matter; a GPU-free
# gate carrying one of those words would be hidden from the sweep it belongs in.
set(GRAPH_RETENTION_TEST_SYCL_OPTIONS
    "-fsycl"
    "-Wno-narrowing"
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-sycl-graph-retention-scope
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-graph-retention-scope.cpp
)
target_include_directories(test-sycl-graph-retention-scope PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-sycl-graph-retention-scope PRIVATE ${GRAPH_RETENTION_TEST_SYCL_OPTIONS})
target_link_options(test-sycl-graph-retention-scope PRIVATE ${GRAPH_RETENTION_TEST_SYCL_OPTIONS})
target_compile_definitions(test-sycl-graph-retention-scope PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-sycl-graph-retention-scope PRIVATE ggml-base ggml ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-sycl-graph-retention-scope PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-sycl-graph-retention-scope COMMAND test-sycl-graph-retention-scope)
set_tests_properties(test-sycl-graph-retention-scope PROPERTIES
    LABELS "sycl;handle;concurrency;tdd"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)

add_executable(test-sycl-retained-handoff-barrier
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-retained-handoff-barrier.cpp
)
target_include_directories(test-sycl-retained-handoff-barrier PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-sycl-retained-handoff-barrier PRIVATE ${GRAPH_RETENTION_TEST_SYCL_OPTIONS})
target_link_options(test-sycl-retained-handoff-barrier PRIVATE ${GRAPH_RETENTION_TEST_SYCL_OPTIONS})
target_compile_definitions(test-sycl-retained-handoff-barrier PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-sycl-retained-handoff-barrier PRIVATE ggml-base ggml ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-sycl-retained-handoff-barrier PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-sycl-retained-handoff-barrier COMMAND test-sycl-retained-handoff-barrier)
set_tests_properties(test-sycl-retained-handoff-barrier PROPERTIES
    LABELS "sycl;handle;concurrency;tdd"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)

# Test: the scoped transient allocators must decide allocation SHAPE -- role, category, arena
# zone, allocator path -- from THIS thread's recording state (llama.cpp-f9tg).  The three
# scoped_unified_*_temp templates in ggml-sycl.cpp took it from
# ggml_sycl_graph_recording_active(), true whenever ANY thread records, so one context's
# COMPUTE/CONTROL transient was labelled GRAPH and steered out of the SCRATCH zone whenever an
# unrelated context happened to be recording.
#
# Unlike its sibling test-sycl-graph-retention-scope this guards ATTRIBUTION, not correctness:
# the wide predicate is a superset, so it only over-applies graph treatment and fails safe.  The
# corrupted fields are exactly the ones llama.cpp-iiff Phase 0's inventory is built from.
#
# CPU-only by construction: it calls the two pure intent builders in common.hpp and reads/writes
# the recording flags -- no device, queue or cache.  The allocation itself is NOT reachable
# device-free, which is why the decision was extracted from it.  Labels avoid every word in
# CLAUDE.md's throttled-sweep denylist (-LE 'residency|mem-handle|cache'), which is about
# memory-exhaustion hazard rather than subject matter.
set(TRANSIENT_INTENT_TEST_SYCL_OPTIONS
    "-fsycl"
    "-Wno-narrowing"
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-sycl-transient-alloc-intent-scope
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-transient-alloc-intent-scope.cpp
)
target_include_directories(test-sycl-transient-alloc-intent-scope PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-sycl-transient-alloc-intent-scope PRIVATE ${TRANSIENT_INTENT_TEST_SYCL_OPTIONS})
target_link_options(test-sycl-transient-alloc-intent-scope PRIVATE ${TRANSIENT_INTENT_TEST_SYCL_OPTIONS})
target_compile_definitions(test-sycl-transient-alloc-intent-scope PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-sycl-transient-alloc-intent-scope PRIVATE ggml-base ggml ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-sycl-transient-alloc-intent-scope PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-sycl-transient-alloc-intent-scope COMMAND test-sycl-transient-alloc-intent-scope)
set_tests_properties(test-sycl-transient-alloc-intent-scope PROPERTIES
    LABELS "sycl;alloc;concurrency;tdd"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)

# Test: infer_tensor_usage() (common.hpp) and expert_tensor_role_from_tensor_name()
# (unified-cache.hpp) must classify MoE expert tensors correctly by name,
# including grovemoe's chunked "_chexps" family (llama.cpp-u2mz). This test
# previously existed only as a self-contained mock that reimplemented (and
# diverged from) the real classifiers, so it never observed the real bug: the
# real infer_tensor_usage() did not recognize "_chexps" tensors as expert
# weights, which aborted MUL_MAT_ID with "unresolved planner expert" and was
# only found by a SIGABRT in an unrelated sweep. This version calls the real
# classifiers.
#
# CPU-only by construction: both functions under test are pure string
# classifiers over a tensor name -- no device, queue, or cache is touched.
set(TENSOR_PLACEMENT_TEST_SYCL_OPTIONS
    "-fsycl"
    "-Wno-narrowing"
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-sycl-tensor-placement
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-tensor-placement.cpp
)
target_include_directories(test-sycl-tensor-placement PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-sycl-tensor-placement PRIVATE ${TENSOR_PLACEMENT_TEST_SYCL_OPTIONS})
target_link_options(test-sycl-tensor-placement PRIVATE ${TENSOR_PLACEMENT_TEST_SYCL_OPTIONS})
target_compile_definitions(test-sycl-tensor-placement PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-sycl-tensor-placement PRIVATE ggml-base ggml ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-sycl-tensor-placement PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-sycl-tensor-placement COMMAND test-sycl-tensor-placement)
set_tests_properties(test-sycl-tensor-placement PROPERTIES
    LABELS "sycl;moe;tensor-placement;tdd"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)

# Test: ggml_sycl_get_tensor_usage() (ggml-sycl.cpp) -- the production entry
# point wrapping infer_tensor_usage() with a name->usage cache -- must
# classify real ggml_tensor objects (via ggml_new_tensor_2d + ggml_set_name)
# the same way the pure name classifier does. Registered at 3c8f296fd, lost
# with the rest of the tests/CMakeLists.txt registrations (llama.cpp-0igs);
# restored here as it directly complements test-sycl-tensor-placement above.
#
# CPU-only by construction: creates tensors through the CPU backend and never
# touches a SYCL device or queue.
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-sycl-tensor-usage
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-tensor-usage.cpp
)
target_include_directories(test-sycl-tensor-usage PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-sycl-tensor-usage PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-sycl-tensor-usage PRIVATE "-fsycl")
target_compile_definitions(test-sycl-tensor-usage PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-sycl-tensor-usage PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-sycl-tensor-usage PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-sycl-tensor-usage COMMAND test-sycl-tensor-usage)
set_tests_properties(test-sycl-tensor-usage PROPERTIES
    LABELS "sycl;moe;tensor-placement;tdd"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)

# Exact-owner weight-usage regression: concurrent LIVE A/B models classify a
# same-name tensor differently, then A is reactivated. The lookup must follow
# the exact ModelToken and load boundaries must not clear another live owner.
# This fixture loads no model and performs no GPU work.
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-cross-model-weight-usage
    tests/test-cross-model-weight-usage.cpp
)
target_include_directories(test-cross-model-weight-usage PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-cross-model-weight-usage PRIVATE ${SYCL_LINK_ONLY_TEST_OPTIONS})
target_link_options(test-cross-model-weight-usage PRIVATE ${SYCL_LINK_ONLY_TEST_OPTIONS})
target_compile_definitions(test-cross-model-weight-usage PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-cross-model-weight-usage PRIVATE ggml-base ggml ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-cross-model-weight-usage PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME cross-model-weight-usage COMMAND test-cross-model-weight-usage)
set_tests_properties(cross-model-weight-usage PROPERTIES
    LABELS "sycl;moe;tensor-placement;cross-model;tdd"
    ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1;LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    SKIP_RETURN_CODE 77
    TIMEOUT 120
)

# Test: ggml_sycl_classify_tensor() (ggml-sycl.cpp) -- the dense/expert/other
# classifier used for tiered-memory buffer-type routing, distinct from
# tensor_usage/expert_tensor_role above. Registered at 3c8f296fd with bare
# assert() only, which -O3 -DNDEBUG (this project's Release flags) compiles
# out entirely, leaving `return 0` as the sole exit path (llama.cpp-0igs).
# 50723136a already added the `#undef NDEBUG` fix to the source; only the
# registration itself was missing.
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-tensor-placement
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-tensor-placement.cpp
)
target_include_directories(test-tensor-placement PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-tensor-placement PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-tensor-placement PRIVATE "-fsycl")
target_compile_definitions(test-tensor-placement PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-tensor-placement PRIVATE ggml-base ggml ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-tensor-placement PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-tensor-placement COMMAND test-tensor-placement)
set_tests_properties(test-tensor-placement PROPERTIES
    LABELS "sycl;moe;tensor-placement;tdd"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)

set(RESIDENCY_TEST_SYCL_OPTIONS
    "-fsycl"
    "-Wno-narrowing"
)
foreach(_residency_test
    test-sycl-mem-handle-lifetime
    test-sycl-residency-reservation
    test-sycl-descriptor-retention
    test-sycl-moe-residency-preflight
    test-sycl-residency-diagnostics
    test-sycl-moe-fused-down-sum-policy
    test-sycl-moe-token-major-metadata
    test-sycl-moe-direct-final-token-major-bridge
    test-sycl-moe-direct-final-scratch-plan
    test-sycl-moe-same-expert-grouping
    test-sycl-moe-glu-q8-artifact-policy
    test-sycl-moe-glu-q8-fused-store-policy
    test-sycl-moe-gateup-prepack-policy
    test-sycl-moe-gateup-prepack-scratch
    test-sycl-moe-xmx-tiled-single-layout-policy
    test-sycl-moe-xmx-tiled-single-layout-planner
    test-sycl-moe-xmx-tiled-materialization
    test-sycl-moe-fusion-noactivation
    test-sycl-moe-sequence-graphlet-policy
)
    add_executable(${_residency_test}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/${_residency_test}.cpp
    )
    target_include_directories(${_residency_test} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${SYCL_INCLUDE_DIR}
    )
    target_compile_options(${_residency_test} PRIVATE ${RESIDENCY_TEST_SYCL_OPTIONS})
    target_link_options(${_residency_test} PRIVATE ${RESIDENCY_TEST_SYCL_OPTIONS})
    target_compile_definitions(${_residency_test} PRIVATE
        GGML_SYCL_WARP_SIZE=32
        GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
    )
    target_link_libraries(${_residency_test} PRIVATE ggml-base ggml ggml-sycl)
    if (IntelSYCL_FOUND)
        target_link_libraries(${_residency_test} PRIVATE IntelSYCL::SYCL_CXX)
    endif()
    add_test(NAME ${_residency_test} COMMAND ${_residency_test})
    set_tests_properties(${_residency_test} PROPERTIES
        LABELS "sycl;cache;residency;mem-handle;tdd"
        ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
        TIMEOUT 120
    )
endforeach()

# ---------------------------------------------------------------------------
# llama.cpp-0igs restoration batch 1: registrations lost near 4974bf53c,
# restored from known-good blocks at 3c8f296fd and adapted to the current
# ggml-base/ggml/ggml-sycl link topology (the old blocks link bare `ggml`,
# which no longer pulls in ggml-sycl as a separate shared library). Selected
# for this batch because each covers a subsystem this branch's 305 commits
# changed (unified cache, mem_handle, KV tier, MoE weight/expert
# classification, fattn dispatch) and none submit device work directly in
# the test itself (a cost/timing probe, see llama.cpp-u2mz).
# ---------------------------------------------------------------------------
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-sycl-weight-key-stability
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-weight-key-stability.cpp
)
target_include_directories(test-sycl-weight-key-stability PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-sycl-weight-key-stability PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-sycl-weight-key-stability PRIVATE "-fsycl")
target_compile_definitions(test-sycl-weight-key-stability PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-sycl-weight-key-stability PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-sycl-weight-key-stability PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-sycl-weight-key-stability COMMAND test-sycl-weight-key-stability)
set_tests_properties(test-sycl-weight-key-stability PROPERTIES
    LABELS "sycl;moe;hostonly;tdd;0igs-batch1"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-tensor-classification
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-tensor-classification.cpp
)
target_include_directories(test-tensor-classification PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-tensor-classification PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-tensor-classification PRIVATE "-fsycl")
target_compile_definitions(test-tensor-classification PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-tensor-classification PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-tensor-classification PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-tensor-classification COMMAND test-tensor-classification)
set_tests_properties(test-tensor-classification PROPERTIES
    LABELS "sycl;moe;hostonly;tdd;0igs-batch1"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-sycl-kv-planned-device-materialization
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-kv-planned-device-materialization.cpp
)
target_include_directories(test-sycl-kv-planned-device-materialization PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-sycl-kv-planned-device-materialization PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-sycl-kv-planned-device-materialization PRIVATE "-fsycl")
target_compile_definitions(test-sycl-kv-planned-device-materialization PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-sycl-kv-planned-device-materialization PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-sycl-kv-planned-device-materialization PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-sycl-kv-planned-device-materialization COMMAND test-sycl-kv-planned-device-materialization)
set_tests_properties(test-sycl-kv-planned-device-materialization PROPERTIES
    LABELS "sycl;kv;hostonly;tdd;0igs-batch1"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-sycl-moe-q8-scratch
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-moe-q8-scratch.cpp
)
target_include_directories(test-sycl-moe-q8-scratch PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-sycl-moe-q8-scratch PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-sycl-moe-q8-scratch PRIVATE "-fsycl")
target_compile_definitions(test-sycl-moe-q8-scratch PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-sycl-moe-q8-scratch PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-sycl-moe-q8-scratch PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-sycl-moe-q8-scratch COMMAND test-sycl-moe-q8-scratch)
set_tests_properties(test-sycl-moe-q8-scratch PROPERTIES
    LABELS "sycl;moe;hostonly;tdd;0igs-batch1"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-sycl-onednn-packed-cache
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-onednn-packed-cache.cpp
)
target_include_directories(test-sycl-onednn-packed-cache PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-sycl-onednn-packed-cache PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-sycl-onednn-packed-cache PRIVATE "-fsycl")
target_compile_definitions(test-sycl-onednn-packed-cache PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-sycl-onednn-packed-cache PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-sycl-onednn-packed-cache PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-sycl-onednn-packed-cache COMMAND test-sycl-onednn-packed-cache)
set_tests_properties(test-sycl-onednn-packed-cache PROPERTIES
    LABELS "sycl;onednn;hostonly;tdd;0igs-batch1"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-sycl-weight-key-uniqueness
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-weight-key-uniqueness.cpp
)
target_include_directories(test-sycl-weight-key-uniqueness PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-sycl-weight-key-uniqueness PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-sycl-weight-key-uniqueness PRIVATE "-fsycl")
target_compile_definitions(test-sycl-weight-key-uniqueness PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-sycl-weight-key-uniqueness PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-sycl-weight-key-uniqueness PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-sycl-weight-key-uniqueness COMMAND test-sycl-weight-key-uniqueness)
set_tests_properties(test-sycl-weight-key-uniqueness PROPERTIES
    LABELS "sycl;moe;hostonly;tdd;0igs-batch1"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-unified-cache-integrity
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-unified-cache-integrity.cpp
)
target_include_directories(test-unified-cache-integrity PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-unified-cache-integrity PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-unified-cache-integrity PRIVATE "-fsycl")
target_compile_definitions(test-unified-cache-integrity PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-unified-cache-integrity PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-unified-cache-integrity PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-unified-cache-integrity COMMAND test-unified-cache-integrity)
set_tests_properties(test-unified-cache-integrity PROPERTIES
    LABELS "sycl;hostonly;tdd;0igs-batch1"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-sycl-fattn-onednn-materialization
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-fattn-onednn-materialization.cpp
)
target_include_directories(test-sycl-fattn-onednn-materialization PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-sycl-fattn-onednn-materialization PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-sycl-fattn-onednn-materialization PRIVATE "-fsycl")
target_compile_definitions(test-sycl-fattn-onednn-materialization PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-sycl-fattn-onednn-materialization PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-sycl-fattn-onednn-materialization PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-sycl-fattn-onednn-materialization COMMAND test-sycl-fattn-onednn-materialization)
set_tests_properties(test-sycl-fattn-onednn-materialization PROPERTIES
    LABELS "sycl;fattn;onednn;tdd;0igs-batch1"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-sycl-fattn-xmx-policy
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-fattn-xmx-policy.cpp
)
target_include_directories(test-sycl-fattn-xmx-policy PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-sycl-fattn-xmx-policy PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-sycl-fattn-xmx-policy PRIVATE "-fsycl")
target_compile_definitions(test-sycl-fattn-xmx-policy PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-sycl-fattn-xmx-policy PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-sycl-fattn-xmx-policy PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-sycl-fattn-xmx-policy COMMAND test-sycl-fattn-xmx-policy)
set_tests_properties(test-sycl-fattn-xmx-policy PROPERTIES
    LABELS "sycl;fattn;xmx;tdd;0igs-batch1"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-sycl-kernel-selection
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-kernel-selection.cpp
)
target_include_directories(test-sycl-kernel-selection PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-sycl-kernel-selection PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-sycl-kernel-selection PRIVATE "-fsycl")
target_compile_definitions(test-sycl-kernel-selection PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-sycl-kernel-selection PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-sycl-kernel-selection PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-sycl-kernel-selection COMMAND test-sycl-kernel-selection)
set_tests_properties(test-sycl-kernel-selection PROPERTIES
    LABELS "sycl;dispatch;tdd;0igs-batch1"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-cold-start
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-cold-start.cpp
)
target_include_directories(test-cold-start PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-cold-start PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-cold-start PRIVATE "-fsycl")
target_compile_definitions(test-cold-start PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-cold-start PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-cold-start PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-cold-start COMMAND test-cold-start)
set_tests_properties(test-cold-start PROPERTIES
    LABELS "sycl;gpu-arch;tdd;0igs-batch1"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-unified-cache-concurrent
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-unified-cache-concurrent.cpp
)
target_include_directories(test-unified-cache-concurrent PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-unified-cache-concurrent PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-unified-cache-concurrent PRIVATE "-fsycl")
target_compile_definitions(test-unified-cache-concurrent PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-unified-cache-concurrent PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-unified-cache-concurrent PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-unified-cache-concurrent COMMAND test-unified-cache-concurrent)
set_tests_properties(test-unified-cache-concurrent PROPERTIES
    LABELS "sycl;hostonly;concurrency;tdd;0igs-batch1"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-sycl-moe-handle-resolution
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-moe-handle-resolution.cpp
)
target_include_directories(test-sycl-moe-handle-resolution PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-sycl-moe-handle-resolution PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-sycl-moe-handle-resolution PRIVATE "-fsycl")
target_compile_definitions(test-sycl-moe-handle-resolution PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-sycl-moe-handle-resolution PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-sycl-moe-handle-resolution PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-sycl-moe-handle-resolution COMMAND test-sycl-moe-handle-resolution)
set_tests_properties(test-sycl-moe-handle-resolution PROPERTIES
    LABELS "sycl;moe;hostonly;tdd;0igs-batch1"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)

# ---------------------------------------------------------------------------
# llama.cpp-0igs restoration batch 2: registrations lost near 4974bf53c,
# restored from known-good blocks at 3c8f296fd and adapted to the current
# ggml-base/ggml/ggml-cpu/ggml-sycl link topology. Confirmed host-only (no
# gpu_selector/malloc_device/queue.submit/parallel_for in the test itself).
# Labels deliberately avoid the bare words "cache"/"mem-handle", which
# collide with CLAUDE.md's throttled-sweep denylist -- batch 1 shipped that
# mistake on 9 of 13 tests (llama.cpp-u2mz/0igs); use *-hostonly or an
# unrelated subsystem word instead.
# ---------------------------------------------------------------------------
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-dmmv-q4-0-coalesced
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-dmmv-q4-0-coalesced.cpp
)
target_include_directories(test-dmmv-q4-0-coalesced PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-dmmv-q4-0-coalesced PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-dmmv-q4-0-coalesced PRIVATE "-fsycl")
target_compile_definitions(test-dmmv-q4-0-coalesced PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-dmmv-q4-0-coalesced PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-dmmv-q4-0-coalesced PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-dmmv-q4-0-coalesced COMMAND test-dmmv-q4-0-coalesced)
set_tests_properties(test-dmmv-q4-0-coalesced PROPERTIES
    LABELS "sycl;dequant;dmmv;tdd;0igs-batch2"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-dmmv-q6k-coalesced
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-dmmv-q6k-coalesced.cpp
)
target_include_directories(test-dmmv-q6k-coalesced PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-dmmv-q6k-coalesced PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-dmmv-q6k-coalesced PRIVATE "-fsycl")
target_compile_definitions(test-dmmv-q6k-coalesced PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-dmmv-q6k-coalesced PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-dmmv-q6k-coalesced PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-dmmv-q6k-coalesced COMMAND test-dmmv-q6k-coalesced)
set_tests_properties(test-dmmv-q6k-coalesced PROPERTIES
    LABELS "sycl;dequant;dmmv;tdd;0igs-batch2"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-fattn-thread-local
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-fattn-thread-local.cpp
)
target_include_directories(test-fattn-thread-local PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-fattn-thread-local PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-fattn-thread-local PRIVATE "-fsycl")
target_compile_definitions(test-fattn-thread-local PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-fattn-thread-local PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-fattn-thread-local PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-fattn-thread-local COMMAND test-fattn-thread-local)
set_tests_properties(test-fattn-thread-local PROPERTIES
    LABELS "sycl;fattn;tdd;0igs-batch2"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-ggml-sycl-soa
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-ggml-sycl-soa.cpp
)
target_include_directories(test-ggml-sycl-soa PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-ggml-sycl-soa PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-ggml-sycl-soa PRIVATE "-fsycl")
target_compile_definitions(test-ggml-sycl-soa PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-ggml-sycl-soa PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-ggml-sycl-soa PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-ggml-sycl-soa COMMAND test-ggml-sycl-soa)
set_tests_properties(test-ggml-sycl-soa PROPERTIES
    LABELS "sycl;layout;tdd;0igs-batch2"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-layout-bytes
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-layout-bytes.cpp
)
target_include_directories(test-layout-bytes PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-layout-bytes PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-layout-bytes PRIVATE "-fsycl")
target_compile_definitions(test-layout-bytes PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-layout-bytes PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-layout-bytes PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-layout-bytes COMMAND test-layout-bytes)
set_tests_properties(test-layout-bytes PROPERTIES
    LABELS "sycl;layout;tdd;0igs-batch2"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-mmq-q6k-gpu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-mmq-q6k-gpu.cpp
)
target_include_directories(test-mmq-q6k-gpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-mmq-q6k-gpu PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-mmq-q6k-gpu PRIVATE "-fsycl")
target_compile_definitions(test-mmq-q6k-gpu PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-mmq-q6k-gpu PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-mmq-q6k-gpu PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-mmq-q6k-gpu COMMAND test-mmq-q6k-gpu)
set_tests_properties(test-mmq-q6k-gpu PROPERTIES
    LABELS "sycl;mmq;tdd;0igs-batch2"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-moe-mini-graph
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-moe-mini-graph.cpp
)
target_include_directories(test-moe-mini-graph PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-moe-mini-graph PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-moe-mini-graph PRIVATE "-fsycl")
target_compile_definitions(test-moe-mini-graph PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-moe-mini-graph PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-moe-mini-graph PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-moe-mini-graph COMMAND test-moe-mini-graph)
set_tests_properties(test-moe-mini-graph PROPERTIES
    LABELS "sycl;moe;tdd;0igs-batch2"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-moe-mul-mat-id
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-moe-mul-mat-id.cpp
)
target_include_directories(test-moe-mul-mat-id PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-moe-mul-mat-id PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-moe-mul-mat-id PRIVATE "-fsycl")
target_compile_definitions(test-moe-mul-mat-id PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-moe-mul-mat-id PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-moe-mul-mat-id PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-moe-mul-mat-id COMMAND test-moe-mul-mat-id)
set_tests_properties(test-moe-mul-mat-id PROPERTIES
    LABELS "sycl;moe;tdd;0igs-batch2"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-moe-mul-mat-id-q4q8
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-moe-mul-mat-id-q4q8.cpp
)
target_include_directories(test-moe-mul-mat-id-q4q8 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-moe-mul-mat-id-q4q8 PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-moe-mul-mat-id-q4q8 PRIVATE "-fsycl")
target_compile_definitions(test-moe-mul-mat-id-q4q8 PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-moe-mul-mat-id-q4q8 PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-moe-mul-mat-id-q4q8 PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-moe-mul-mat-id-q4q8 COMMAND test-moe-mul-mat-id-q4q8)
set_tests_properties(test-moe-mul-mat-id-q4q8 PROPERTIES
    LABELS "sycl;moe;tdd;0igs-batch2"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-mul-mat-host-streaming
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-mul-mat-host-streaming.cpp
)
target_include_directories(test-mul-mat-host-streaming PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-mul-mat-host-streaming PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-mul-mat-host-streaming PRIVATE "-fsycl")
target_compile_definitions(test-mul-mat-host-streaming PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-mul-mat-host-streaming PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-mul-mat-host-streaming PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-mul-mat-host-streaming COMMAND test-mul-mat-host-streaming)
set_tests_properties(test-mul-mat-host-streaming PROPERTIES
    LABELS "sycl;dispatch;tdd;0igs-batch2"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-onednn-fallback
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-onednn-fallback.cpp
)
target_include_directories(test-onednn-fallback PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-onednn-fallback PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-onednn-fallback PRIVATE "-fsycl")
target_compile_definitions(test-onednn-fallback PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-onednn-fallback PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-onednn-fallback PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-onednn-fallback COMMAND test-onednn-fallback)
set_tests_properties(test-onednn-fallback PROPERTIES
    LABELS "sycl;onednn;tdd;0igs-batch2"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-onednn-woq
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-onednn-woq.cpp
)
target_include_directories(test-onednn-woq PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-onednn-woq PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-onednn-woq PRIVATE "-fsycl")
target_compile_definitions(test-onednn-woq PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-onednn-woq PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (GGML_SYCL_DNNL)
    target_link_libraries(test-onednn-woq PRIVATE DNNL::dnnl)
endif()
if (IntelSYCL_FOUND)
    target_link_libraries(test-onednn-woq PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-onednn-woq COMMAND test-onednn-woq)
set_tests_properties(test-onednn-woq PROPERTIES
    LABELS "sycl;onednn;tdd;0igs-batch2"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-q6k-dispatch
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-q6k-dispatch.cpp
)
target_include_directories(test-q6k-dispatch PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-q6k-dispatch PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-q6k-dispatch PRIVATE "-fsycl")
target_compile_definitions(test-q6k-dispatch PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-q6k-dispatch PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-q6k-dispatch PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-q6k-dispatch COMMAND test-q6k-dispatch)
set_tests_properties(test-q6k-dispatch PROPERTIES
    LABELS "sycl;dispatch;tdd;0igs-batch2"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-q8-0-layout-cache-path
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-q8-0-layout-cache-path.cpp
)
target_include_directories(test-q8-0-layout-cache-path PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-q8-0-layout-cache-path PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-q8-0-layout-cache-path PRIVATE "-fsycl")
target_compile_definitions(test-q8-0-layout-cache-path PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-q8-0-layout-cache-path PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-q8-0-layout-cache-path PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-q8-0-layout-cache-path COMMAND test-q8-0-layout-cache-path)
set_tests_properties(test-q8-0-layout-cache-path PROPERTIES
    LABELS "sycl;layout;tdd;0igs-batch2"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-q8-0-layout-cache-path-mmvq
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-q8-0-layout-cache-path-mmvq.cpp
)
target_include_directories(test-q8-0-layout-cache-path-mmvq PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-q8-0-layout-cache-path-mmvq PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-q8-0-layout-cache-path-mmvq PRIVATE "-fsycl")
target_compile_definitions(test-q8-0-layout-cache-path-mmvq PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-q8-0-layout-cache-path-mmvq PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-q8-0-layout-cache-path-mmvq PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-q8-0-layout-cache-path-mmvq COMMAND test-q8-0-layout-cache-path-mmvq)
set_tests_properties(test-q8-0-layout-cache-path-mmvq PROPERTIES
    LABELS "sycl;layout;mmvq;tdd;0igs-batch2"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-sycl-cpu-dispatch
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-cpu-dispatch.cpp
)
target_include_directories(test-sycl-cpu-dispatch PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-sycl-cpu-dispatch PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-sycl-cpu-dispatch PRIVATE "-fsycl")
target_compile_definitions(test-sycl-cpu-dispatch PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-sycl-cpu-dispatch PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-sycl-cpu-dispatch PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-sycl-cpu-dispatch COMMAND test-sycl-cpu-dispatch)
set_tests_properties(test-sycl-cpu-dispatch PROPERTIES
    LABELS "sycl;dispatch;tdd;0igs-batch2"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-sycl-moe-expert-parallelism
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-moe-expert-parallelism.cpp
)
target_include_directories(test-sycl-moe-expert-parallelism PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-sycl-moe-expert-parallelism PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-sycl-moe-expert-parallelism PRIVATE "-fsycl")
target_compile_definitions(test-sycl-moe-expert-parallelism PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-sycl-moe-expert-parallelism PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-sycl-moe-expert-parallelism PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-sycl-moe-expert-parallelism COMMAND test-sycl-moe-expert-parallelism)
set_tests_properties(test-sycl-moe-expert-parallelism PROPERTIES
    LABELS "sycl;moe;tdd;0igs-batch2"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-sycl-moe-identity-hash
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-moe-identity-hash.cpp
)
target_include_directories(test-sycl-moe-identity-hash PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-sycl-moe-identity-hash PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-sycl-moe-identity-hash PRIVATE "-fsycl")
target_compile_definitions(test-sycl-moe-identity-hash PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-sycl-moe-identity-hash PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-sycl-moe-identity-hash PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-sycl-moe-identity-hash COMMAND test-sycl-moe-identity-hash)
set_tests_properties(test-sycl-moe-identity-hash PROPERTIES
    LABELS "sycl;moe;tdd;0igs-batch2"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-sycl-orchestrator
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-orchestrator.cpp
)
target_include_directories(test-sycl-orchestrator PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-sycl-orchestrator PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-sycl-orchestrator PRIVATE "-fsycl")
target_compile_definitions(test-sycl-orchestrator PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-sycl-orchestrator PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-sycl-orchestrator PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-sycl-orchestrator COMMAND test-sycl-orchestrator)
set_tests_properties(test-sycl-orchestrator PROPERTIES
    LABELS "sycl;orchestrator;tdd;0igs-batch2"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

# ---------------------------------------------------------------------------
# Task 16 accepted restoration set: previously inactive registrations.
#
# These fixtures use the static backend topology. A GGML_BACKEND_DL build makes
# ggml-sycl a MODULE library, which cannot be linked into an executable; the DL
# lifecycle fixtures above exercise that topology through the public registry.
# Each target stays explicit so its source and link contract can be audited
# independently by the remaining Task 17 stages.
# ---------------------------------------------------------------------------

if (NOT GGML_BACKEND_DL)
    add_executable(test-cpu-gpu-soa-interaction
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-cpu-gpu-soa-interaction.cpp
    )
    target_include_directories(test-cpu-gpu-soa-interaction PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${SYCL_INCLUDE_DIR}
    )
    target_compile_options(test-cpu-gpu-soa-interaction PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-cpu-gpu-soa-interaction PRIVATE "-fsycl")
    target_compile_definitions(test-cpu-gpu-soa-interaction PRIVATE
        GGML_SYCL_WARP_SIZE=32
        GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
    )
    target_link_libraries(test-cpu-gpu-soa-interaction PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
    if (IntelSYCL_FOUND)
        target_link_libraries(test-cpu-gpu-soa-interaction PRIVATE IntelSYCL::SYCL_CXX)
    endif()
    add_test(NAME test-cpu-gpu-soa-interaction COMMAND test-cpu-gpu-soa-interaction)
    set_tests_properties(test-cpu-gpu-soa-interaction PROPERTIES
        LABELS "sycl;sycl-restored;gpu;soa;interaction;restored"
        ENVIRONMENT "GGML_SYCL_VRAM_ARENA=0;GGML_SYCL_ASYNC_MEM=1"
        RUN_SERIAL TRUE
    )
endif()

# Source-only contract: valid for both static and dynamic backend topologies.
add_test(NAME test-sycl-soa-reorder-source-contract
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/ggml-sycl.cpp
        -DCACHE_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/unified-cache.cpp
        -DFIXTURE=${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-cpu-gpu-soa-interaction.cpp
        -P ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-soa-reorder-source-contract.cmake
)
set_tests_properties(test-sycl-soa-reorder-source-contract PROPERTIES
    LABELS "sycl;source-contract;soa;restored"
)

if (NOT GGML_BACKEND_DL)
    add_executable(test-mmvq-q8-0-streaming-bench
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-mmvq-q8-0-streaming-bench.cpp
    )
    target_include_directories(test-mmvq-q8-0-streaming-bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${SYCL_INCLUDE_DIR}
    )
    target_compile_options(test-mmvq-q8-0-streaming-bench PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-mmvq-q8-0-streaming-bench PRIVATE "-fsycl")
    target_compile_definitions(test-mmvq-q8-0-streaming-bench PRIVATE
        GGML_SYCL_WARP_SIZE=32
        GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
    )
    target_link_libraries(test-mmvq-q8-0-streaming-bench PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
    if (IntelSYCL_FOUND)
        target_link_libraries(test-mmvq-q8-0-streaming-bench PRIVATE IntelSYCL::SYCL_CXX)
    endif()
    add_test(NAME test-mmvq-q8-0-streaming-bench COMMAND test-mmvq-q8-0-streaming-bench)
    add_test(NAME test-mmvq-q8-0-streaming-smoke COMMAND test-mmvq-q8-0-streaming-bench)
    set_tests_properties(test-mmvq-q8-0-streaming-smoke PROPERTIES
        ENVIRONMENT "GGML_SYCL_MMVQ_BENCH=1;GGML_SYCL_MMVQ_BENCH_MODE=cache;GGML_SYCL_MMVQ_BENCH_NROWS=256;GGML_SYCL_MMVQ_BENCH_NCOLS=512;GGML_SYCL_MMVQ_BENCH_ITERS=1;GGML_SYCL_MMVQ_BENCH_WARMUP=0;GGML_SYCL_MMVQ_BENCH_SLICE_MB=1;GGML_SYCL_MMVQ_BENCH_BUFFERS=1"
    )
    add_test(NAME test-mmq-q8-0-streaming-smoke COMMAND test-mmvq-q8-0-streaming-bench)
    set_tests_properties(test-mmq-q8-0-streaming-smoke PROPERTIES
        ENVIRONMENT "GGML_SYCL_MMVQ_BENCH=1;GGML_SYCL_MMVQ_BENCH_MODE=mmq;GGML_SYCL_MMVQ_BENCH_MMQ_NROWS=512;GGML_SYCL_MMVQ_BENCH_MMQ_NCOLS=2048;GGML_SYCL_MMVQ_BENCH_MMQ_NTOKENS=4;GGML_SYCL_MMVQ_BENCH_MMQ_ITERS=1;GGML_SYCL_MMVQ_BENCH_MMQ_WARMUP=0;GGML_SYCL_MMVQ_BENCH_MMQ_BUDGET_MB=64"
    )
    add_test(NAME test-mmq-q8-0-streaming-forced COMMAND test-mmvq-q8-0-streaming-bench)
    set_tests_properties(test-mmq-q8-0-streaming-forced PROPERTIES
        ENVIRONMENT "GGML_SYCL_MMVQ_BENCH=1;GGML_SYCL_MMVQ_BENCH_MODE=mmq;GGML_SYCL_MMVQ_BENCH_MMQ_NROWS=512;GGML_SYCL_MMVQ_BENCH_MMQ_NCOLS=2048;GGML_SYCL_MMVQ_BENCH_MMQ_NTOKENS=4;GGML_SYCL_MMVQ_BENCH_MMQ_ITERS=1;GGML_SYCL_MMVQ_BENCH_MMQ_WARMUP=0;GGML_SYCL_MMVQ_BENCH_MMQ_BUDGET_MB=1"
    )
    set_tests_properties(
        test-mmvq-q8-0-streaming-bench
        test-mmvq-q8-0-streaming-smoke
        test-mmq-q8-0-streaming-smoke
        test-mmq-q8-0-streaming-forced
        PROPERTIES
        LABELS "sycl;sycl-restored;gpu;benchmark;manual;streaming;restored"
    )
    set_tests_properties(
        test-mmvq-q8-0-streaming-bench
        test-mmq-q8-0-streaming-smoke
        test-mmq-q8-0-streaming-forced
        PROPERTIES
        SKIP_RETURN_CODE 77
    )
endif()

if (NOT GGML_BACKEND_DL)
    add_executable(test-mxfp4-xmx-tiled
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-mxfp4-xmx-tiled.cpp
    )
    target_include_directories(test-mxfp4-xmx-tiled PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${SYCL_INCLUDE_DIR}
    )
    target_compile_options(test-mxfp4-xmx-tiled PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-mxfp4-xmx-tiled PRIVATE "-fsycl")
    target_compile_definitions(test-mxfp4-xmx-tiled PRIVATE
        GGML_SYCL_WARP_SIZE=32
        GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
    )
    target_link_libraries(test-mxfp4-xmx-tiled PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
    if (IntelSYCL_FOUND)
        target_link_libraries(test-mxfp4-xmx-tiled PRIVATE IntelSYCL::SYCL_CXX)
    endif()
    add_test(NAME test-mxfp4-xmx-tiled COMMAND test-mxfp4-xmx-tiled)
    set_tests_properties(test-mxfp4-xmx-tiled PROPERTIES
        LABELS "sycl;sycl-restored;gpu;xmx;tiled;restored"
        RUN_SERIAL TRUE
        SKIP_RETURN_CODE 77
    )
endif()

if (NOT GGML_BACKEND_DL)
    add_executable(test-q6k-reorder-dispatch
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-q6k-reorder-dispatch.cpp
    )
    target_include_directories(test-q6k-reorder-dispatch PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${SYCL_INCLUDE_DIR}
    )
    target_compile_options(test-q6k-reorder-dispatch PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-q6k-reorder-dispatch PRIVATE "-fsycl")
    target_compile_definitions(test-q6k-reorder-dispatch PRIVATE
        GGML_SYCL_WARP_SIZE=32
        GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
    )
    target_link_libraries(test-q6k-reorder-dispatch PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
    if (IntelSYCL_FOUND)
        target_link_libraries(test-q6k-reorder-dispatch PRIVATE IntelSYCL::SYCL_CXX)
    endif()
    add_test(NAME test-q6k-reorder-dispatch COMMAND test-q6k-reorder-dispatch)
    set_tests_properties(test-q6k-reorder-dispatch PROPERTIES
        LABELS "sycl;sycl-restored;gpu;q6k;reorder;dispatch;restored"
        RUN_SERIAL TRUE
    )
endif()

if (NOT GGML_BACKEND_DL AND GGML_SYCL_DNNL)
    add_executable(test-sycl-fattn-onednn-descriptors
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-fattn-onednn-descriptors.cpp
    )
    target_include_directories(test-sycl-fattn-onednn-descriptors PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${SYCL_INCLUDE_DIR}
    )
    target_compile_options(test-sycl-fattn-onednn-descriptors PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-sycl-fattn-onednn-descriptors PRIVATE "-fsycl")
    target_compile_definitions(test-sycl-fattn-onednn-descriptors PRIVATE
        GGML_SYCL_WARP_SIZE=32
        GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
    )
    target_link_libraries(test-sycl-fattn-onednn-descriptors PRIVATE ggml-base ggml ggml-cpu ggml-sycl DNNL::dnnl)
    if (IntelSYCL_FOUND)
        target_link_libraries(test-sycl-fattn-onednn-descriptors PRIVATE IntelSYCL::SYCL_CXX)
    endif()
    add_test(NAME test-sycl-fattn-onednn-descriptors COMMAND test-sycl-fattn-onednn-descriptors)
    set_tests_properties(test-sycl-fattn-onednn-descriptors PROPERTIES
        LABELS "sycl;sycl-restored;gpu;onednn;fattn;descriptors;restored"
        RUN_SERIAL TRUE
        SKIP_RETURN_CODE 77
    )
endif()

if (NOT GGML_BACKEND_DL)
    add_executable(test-sycl-set-rows-owner-routing
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-set-rows-owner-routing.cpp
    )
    target_include_directories(test-sycl-set-rows-owner-routing PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${SYCL_INCLUDE_DIR}
    )
    target_compile_options(test-sycl-set-rows-owner-routing PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-sycl-set-rows-owner-routing PRIVATE "-fsycl")
    target_compile_definitions(test-sycl-set-rows-owner-routing PRIVATE
        GGML_SYCL_WARP_SIZE=32
        GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
    )
    target_link_libraries(test-sycl-set-rows-owner-routing PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
    if (IntelSYCL_FOUND)
        target_link_libraries(test-sycl-set-rows-owner-routing PRIVATE IntelSYCL::SYCL_CXX)
    endif()
    add_test(NAME test-sycl-set-rows-owner-routing COMMAND test-sycl-set-rows-owner-routing)
    set_tests_properties(test-sycl-set-rows-owner-routing PROPERTIES
        LABELS "sycl;sycl-restored;gpu;set-rows;owner-routing;restored"
        RUN_SERIAL TRUE
    )
endif()

if (NOT GGML_BACKEND_DL)
    add_executable(test-unified-dispatch-integration
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-unified-dispatch-integration.cpp
    )
    target_include_directories(test-unified-dispatch-integration PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${SYCL_INCLUDE_DIR}
    )
    target_compile_options(test-unified-dispatch-integration PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-unified-dispatch-integration PRIVATE "-fsycl")
    target_compile_definitions(test-unified-dispatch-integration PRIVATE
        GGML_SYCL_WARP_SIZE=32
        GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
    )
    target_link_libraries(test-unified-dispatch-integration PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
    if (IntelSYCL_FOUND)
        target_link_libraries(test-unified-dispatch-integration PRIVATE IntelSYCL::SYCL_CXX)
    endif()
    add_test(NAME test-unified-dispatch-integration COMMAND test-unified-dispatch-integration)
    set_tests_properties(test-unified-dispatch-integration PROPERTIES
        LABELS "sycl;sycl-restored;gpu;unified;dispatch;integration;restored"
        RUN_SERIAL TRUE
        SKIP_RETURN_CODE 77
    )
endif()

if (NOT GGML_BACKEND_DL)
add_executable(test-sycl-unified-cache
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-unified-cache.cpp
)
target_include_directories(test-sycl-unified-cache PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-sycl-unified-cache PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-sycl-unified-cache PRIVATE "-fsycl")
target_compile_definitions(test-sycl-unified-cache PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-sycl-unified-cache PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-sycl-unified-cache PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-sycl-unified-cache COMMAND test-sycl-unified-cache)
set_tests_properties(test-sycl-unified-cache PROPERTIES
    LABELS "sycl;hostonly;tdd;0igs-batch2"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-sycl-unified-memory-e2e
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-unified-memory-e2e.cpp
)
target_include_directories(test-sycl-unified-memory-e2e PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-sycl-unified-memory-e2e PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-sycl-unified-memory-e2e PRIVATE "-fsycl")
target_compile_definitions(test-sycl-unified-memory-e2e PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-sycl-unified-memory-e2e PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-sycl-unified-memory-e2e PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-sycl-unified-memory-e2e COMMAND test-sycl-unified-memory-e2e)
set_tests_properties(test-sycl-unified-memory-e2e PROPERTIES
    LABELS "sycl;hostonly;tdd;0igs-batch2"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-sycl-xmx-unified-correctness
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-xmx-unified-correctness.cpp
)
target_include_directories(test-sycl-xmx-unified-correctness PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-sycl-xmx-unified-correctness PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-sycl-xmx-unified-correctness PRIVATE "-fsycl")
target_compile_definitions(test-sycl-xmx-unified-correctness PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-sycl-xmx-unified-correctness PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-sycl-xmx-unified-correctness PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-sycl-xmx-unified-correctness COMMAND test-sycl-xmx-unified-correctness)
set_tests_properties(test-sycl-xmx-unified-correctness PROPERTIES
    LABELS "sycl;xmx;tdd;0igs-batch2"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-tiered-dispatch
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-tiered-dispatch.cpp
)
target_include_directories(test-tiered-dispatch PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-tiered-dispatch PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-tiered-dispatch PRIVATE "-fsycl")
target_compile_definitions(test-tiered-dispatch PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-tiered-dispatch PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-tiered-dispatch PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-tiered-dispatch COMMAND test-tiered-dispatch)
set_tests_properties(test-tiered-dispatch PROPERTIES
    LABELS "sycl;dispatch;tdd;0igs-batch2"
    ENVIRONMENT "GGML_SYCL_VRAM_ARENA=1;LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    SKIP_RETURN_CODE 77
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-xmx-host-streaming
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-xmx-host-streaming.cpp
)
target_include_directories(test-xmx-host-streaming PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-xmx-host-streaming PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-xmx-host-streaming PRIVATE "-fsycl")
target_compile_definitions(test-xmx-host-streaming PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-xmx-host-streaming PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-xmx-host-streaming PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-xmx-host-streaming COMMAND test-xmx-host-streaming)
set_tests_properties(test-xmx-host-streaming PROPERTIES
    LABELS "sycl;xmx;tdd;0igs-batch2"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-xmx-kernel-config
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-xmx-kernel-config.cpp
)
target_include_directories(test-xmx-kernel-config PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-xmx-kernel-config PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-xmx-kernel-config PRIVATE "-fsycl")
target_compile_definitions(test-xmx-kernel-config PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-xmx-kernel-config PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-xmx-kernel-config PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-xmx-kernel-config COMMAND test-xmx-kernel-config)
set_tests_properties(test-xmx-kernel-config PROPERTIES
    LABELS "sycl;xmx;tdd;0igs-batch2"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-xmx-quant-loaders
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-xmx-quant-loaders.cpp
)
target_include_directories(test-xmx-quant-loaders PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-xmx-quant-loaders PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-xmx-quant-loaders PRIVATE "-fsycl")
target_compile_definitions(test-xmx-quant-loaders PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-xmx-quant-loaders PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-xmx-quant-loaders PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-xmx-quant-loaders COMMAND test-xmx-quant-loaders)
set_tests_properties(test-xmx-quant-loaders PROPERTIES
    LABELS "sycl;xmx;tdd;0igs-batch2"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-xmx-unified-kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-xmx-unified-kernel.cpp
)
target_include_directories(test-xmx-unified-kernel PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-xmx-unified-kernel PRIVATE "-fsycl" "-Wno-narrowing")
target_link_options(test-xmx-unified-kernel PRIVATE "-fsycl")
target_compile_definitions(test-xmx-unified-kernel PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-xmx-unified-kernel PRIVATE ggml-base ggml ggml-cpu ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-xmx-unified-kernel PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-xmx-unified-kernel COMMAND test-xmx-unified-kernel)
set_tests_properties(test-xmx-unified-kernel PROPERTIES
    LABELS "sycl;xmx;tdd;0igs-batch2"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)

# Test: reset_model_weight_entries() must preserve entries still leased by a
# live model at a model-load boundary instead of aborting (llama.cpp-ljb9,
# commit acdb192d4), and must keep id_to_key_ pointed at a surviving layout
# sibling rather than dangling.
#
# It also carries the model-OWNERSHIP tests for llama.cpp-0qlw / 9f3a2e0f0: a
# live model's IDLE weights (in_use_count == 0 between graphs) must survive
# another model's load boundary, and the two reclaim modes must keep their
# opposite treatment of unattributed entries. Each preserve assertion is paired
# with a mutation control that forces live_model_mask_ to 0 and requires the
# same entry to be reclaimed -- without which "preserved N of N" is equally
# what a no-op prints. This file is the affordable instrument for that claim:
# test-thread-safety needs 156-180 GB of Shmem and currently SEGFAULTs for two
# unrelated reasons, while this test measured peak RSS 317 MB (see below).
#
# Registered outside the GGML_SYCL_BUILD_XMX_TESTS
# guard (which defaults OFF) so it always runs -- mirrors the _residency_test
# loop above: no device kernel code of its own, so it only needs -fsycl to
# link against the prebuilt ggml-sycl library, not the full XMX AOT target list.
set(RESET_MODEL_WEIGHT_TEST_SYCL_OPTIONS
    "-fsycl"
    "-Wno-narrowing"
)
endif() # static SYCL backend fixture

if (NOT GGML_BACKEND_DL)
add_executable(test-sycl-reset-model-weight-lease-preserve
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-reset-model-weight-lease-preserve.cpp
)
target_include_directories(test-sycl-reset-model-weight-lease-preserve PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-sycl-reset-model-weight-lease-preserve PRIVATE ${RESET_MODEL_WEIGHT_TEST_SYCL_OPTIONS})
target_link_options(test-sycl-reset-model-weight-lease-preserve PRIVATE ${RESET_MODEL_WEIGHT_TEST_SYCL_OPTIONS})
target_compile_definitions(test-sycl-reset-model-weight-lease-preserve PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
target_link_libraries(test-sycl-reset-model-weight-lease-preserve PRIVATE ggml-base ggml ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-sycl-reset-model-weight-lease-preserve PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME sycl-reset-model-weight-lease-preserve COMMAND test-sycl-reset-model-weight-lease-preserve)
# ENVIRONMENT pins the B50 (the validation card) rather than the B70, and it is
# load-bearing: the test's own source falls back to level_zero:0 -- the B70, i.e.
# the benchmark card -- when nothing is set, so a bare `ctest -R` or the throttled
# full sweep would otherwise land it there and perturb a run in progress.  Matches
# test-sycl-zone-reset-live-refusal / unpin-event / env-report in tests/CMakeLists.txt.
# The surrounding tests in THIS file predate that convention and pin nothing; do not
# take their silence as the local rule.
#
# LABELS deliberately omit a task id.  Neither "sycl" nor "lease" nor "bugfix" is in
# CLAUDE.md's `-LE 'residency|mem-handle|cache'` denylist, so the throttled sweep runs
# this test -- which is the point.  Measured before dropping the hazard labels:
# free -g flat at 210 GB, peak RSS 317 MB, against a family that reaches ~227 GB.
set_tests_properties(sycl-reset-model-weight-lease-preserve PROPERTIES
    LABELS "sycl;lease;bugfix"
    ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1;LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    # The test exits 77 when no GPU is present, and when built without the SYCL
    # backend at all. Without this, ctest renders 77 as FAILED rather than the
    # intended Skipped. Matches mmq-xmx-dispatch (0f48c5352, llama.cpp-k208).
    SKIP_RETURN_CODE 77
    TIMEOUT 120
)
endif() # static SYCL backend fixture

add_executable(test-sycl-e2e-profile
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-e2e-profile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e-profile.cpp
)
target_include_directories(test-sycl-e2e-profile PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
add_test(NAME test-sycl-e2e-profile COMMAND test-sycl-e2e-profile)
set_tests_properties(test-sycl-e2e-profile PROPERTIES
    LABELS "sycl;profile;tdd"
    TIMEOUT 120
)

add_executable(test-sycl-timeline
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-timeline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl-timeline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl-itt.cpp
)
target_include_directories(test-sycl-timeline PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
add_test(NAME test-sycl-timeline COMMAND test-sycl-timeline)
set_tests_properties(test-sycl-timeline PROPERTIES
    LABELS "sycl;profile;timeline;tdd"
    TIMEOUT 120
)

add_executable(test-sycl-esimd-float-atomic-compile
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-esimd-float-atomic-compile.cpp
)
target_compile_options(test-sycl-esimd-float-atomic-compile PRIVATE -fsycl -Wno-narrowing)
target_link_options(test-sycl-esimd-float-atomic-compile PRIVATE -fsycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-sycl-esimd-float-atomic-compile PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-sycl-esimd-float-atomic-compile COMMAND test-sycl-esimd-float-atomic-compile)
set_tests_properties(test-sycl-esimd-float-atomic-compile PROPERTIES LABELS "sycl;esimd;compile;tdd" TIMEOUT 120)

# LEVEL_ZERO_LOADER was located above for the packed-K native-event fixture.
add_executable(test-sycl-level-zero-vmem-feasibility
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-level-zero-vmem-feasibility.cpp
)
target_link_libraries(test-sycl-level-zero-vmem-feasibility PRIVATE ${LEVEL_ZERO_LOADER})
add_test(NAME test-sycl-level-zero-vmem-feasibility COMMAND test-sycl-level-zero-vmem-feasibility)
set_tests_properties(test-sycl-level-zero-vmem-feasibility PROPERTIES
    LABELS "sycl;level-zero;vmem;tdd"
    # 77 = genuine absence (no Level Zero, no driver, no device). The test used
    # to route EVERY early exit through one skip helper returning 0, including
    # "zeVirtualMemReserve failed" and "zeVirtualMemFree failed" -- which fire
    # only when Level Zero IS present and a call that should have worked did
    # not. That is precisely the result this test exists to find, and it was
    # reported green. Those now exit 1; only real absences exit 77.
    SKIP_RETURN_CODE 77
    TIMEOUT 120
)

set(ONEDNN_MXFP4_FEASIBILITY_SYCL_OPTIONS
    "-fsycl"
    "-Wno-narrowing"
)
add_executable(test-sycl-onednn-mxfp4-feasibility
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-sycl-onednn-mxfp4-feasibility.cpp
)
target_include_directories(test-sycl-onednn-mxfp4-feasibility PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${SYCL_INCLUDE_DIR}
)
target_compile_options(test-sycl-onednn-mxfp4-feasibility PRIVATE ${ONEDNN_MXFP4_FEASIBILITY_SYCL_OPTIONS})
target_link_options(test-sycl-onednn-mxfp4-feasibility PRIVATE ${ONEDNN_MXFP4_FEASIBILITY_SYCL_OPTIONS})
target_compile_definitions(test-sycl-onednn-mxfp4-feasibility PRIVATE
    GGML_SYCL_WARP_SIZE=32
    GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
)
if (GGML_SYCL_DNNL)
    target_link_libraries(test-sycl-onednn-mxfp4-feasibility PRIVATE DNNL::dnnl)
endif()
if (IntelSYCL_FOUND)
    target_link_libraries(test-sycl-onednn-mxfp4-feasibility PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME test-sycl-onednn-mxfp4-feasibility COMMAND test-sycl-onednn-mxfp4-feasibility)
set_tests_properties(test-sycl-onednn-mxfp4-feasibility PROPERTIES
    LABELS "sycl;onednn;mxfp4;tdd"
    ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120
)

# Standalone oneDNN vs SYCL kernel microbenchmark
if (GGML_SYCL_DNNL)
    add_executable(bench-dnnl-ops
        tests/bench-dnnl-ops.cpp
    )
    target_include_directories(bench-dnnl-ops PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )
    target_compile_options(bench-dnnl-ops PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(bench-dnnl-ops PRIVATE "-fsycl")
    target_compile_definitions(bench-dnnl-ops PRIVATE
        GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
    )
    target_link_libraries(bench-dnnl-ops PRIVATE DNNL::dnnl)
    if (IntelSYCL_FOUND)
        target_link_libraries(bench-dnnl-ops PRIVATE IntelSYCL::SYCL_CXX)
    endif()
endif()

# Test: architecture-derived GPU family detection (B70/G31 regression).
# Host-only — family_from_architecture / family_from_name are pure functions,
# so no GPU and no AOT target are needed. Registered outside the
# GGML_SYCL_BUILD_XMX_TESTS guard (which defaults OFF) so it always runs, and
# compiled against gpu-arch.cpp directly so it does not pull in libggml-sycl.
# cold-start.cpp joins it for the same reason: derive_initial_config is a pure
# function whose tile choice depends on the family helper, and compiling the
# one TU is far cheaper than linking the backend.
set(GPU_ARCH_TEST_SYCL_OPTIONS
    "-fsycl"
    "-Wno-narrowing"
)
add_executable(test-gpu-arch
    tests/test-gpu-arch.cpp
    gpu-arch.cpp
    cold-start.cpp
)
target_include_directories(test-gpu-arch PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
target_compile_options(test-gpu-arch PRIVATE ${GPU_ARCH_TEST_SYCL_OPTIONS})
target_link_options(test-gpu-arch PRIVATE ${GPU_ARCH_TEST_SYCL_OPTIONS})
# gpu-arch.cpp logs its name-heuristic fallback through GGML_SYCL_DEBUG, whose
# g_ggml_sycl_debug globals are defined in a libggml-sycl TU. This test does not
# link that library (deliberately — it stays host-only and builds in seconds),
# so the macro is stubbed out here, mirroring unified-kernel.cpp's
# XMX_TEST_STANDALONE pattern. Production builds get the real logging.
target_compile_definitions(test-gpu-arch PRIVATE GGML_SYCL_GPU_ARCH_STANDALONE=1)
if (IntelSYCL_FOUND)
    target_link_libraries(test-gpu-arch PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME gpu-arch COMMAND test-gpu-arch)
set_tests_properties(gpu-arch PROPERTIES
    LABELS "sycl;arch;capability;tdd"
    TIMEOUT 60
)

# Test: structural path-scoped arena zone maxima.
# Host-only — zone_scoped_maxima and its predicates are pure functions over a
# vector of descriptors, so no GPU, no AOT target, and no libggml-sycl link.
# Mirrors the test-gpu-arch pattern above for the same reason: the production
# consumer (populate_host_zone_sizing) is static inside a ~94k-line SYCL TU.
# Unlike test-gpu-arch this target deliberately takes NO -fsycl: zone-sizing.cpp
# includes no SYCL header, and staying SYCL-free is what makes it build in
# seconds. (The same file is also globbed into libggml-sycl at :42, where it is
# compiled with -fsycl like every other backend TU; that is the production
# build, this target is the host-only one.)
add_executable(test-zone-sizing
    tests/test-zone-sizing.cpp
    zone-sizing.cpp
)
target_include_directories(test-zone-sizing PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
# zone-sizing.cpp reports predicate under-estimates and classifier collapse
# through GGML_LOG_WARN, which lives in ggml-impl.h and resolves to a symbol in
# ggml-base. This test deliberately links neither (that is what keeps it
# host-only and building in seconds), so the log macro is stubbed out here --
# the same arrangement test-gpu-arch uses for GGML_SYCL_DEBUG above. Production
# builds get the real logging.
target_compile_definitions(test-zone-sizing PRIVATE ZONE_SIZING_STANDALONE=1)
add_test(NAME zone-sizing COMMAND test-zone-sizing)
set_tests_properties(zone-sizing PROPERTIES
    LABELS "sycl;memory;planner;tdd"
    TIMEOUT 60
)

# Test: a Battlemage device is never gated onto the legacy mul_mat path.
# Guards the gate dispatch.hpp reads (family -> supports_esimd_dpas), which is
# what silently demoted the B70, and calls is_unified_kernel_enabled_for_device
# itself so the disable diagnostic is covered by a run rather than by reading.
# Registered outside the GGML_SYCL_BUILD_XMX_TESTS guard (which defaults OFF)
# and compiled against gpu-arch.cpp directly instead of linking libggml-sycl:
# the backend library would bring in dpct::dev_mgr, whose first use enumerates
# every SYCL device on the machine. The test supplies its own definitions of
# the two backend symbols dispatch.hpp's gate needs (see the link-seam comment
# in the test), so it stays host-only and needs no GPU.
#
# ggml-base is linked for one reason: dpct::dev_mgr's constructor references
# ggml_abort. That code is never executed here — the negative-device-id path
# returns before any dev_mgr use — but it is emitted, and at -O0 nothing folds
# it away. Linking ggml-base makes the test build at any optimization level
# instead of depending on the optimizer to delete an unreachable branch.
set(ESIMD_DPAS_GATE_TEST_SYCL_OPTIONS
    "-fsycl"
    "-Wno-narrowing"
)
add_executable(test-esimd-dpas-gate
    tests/test-esimd-dpas-gate.cpp
    gpu-arch.cpp
)
target_include_directories(test-esimd-dpas-gate PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
target_compile_options(test-esimd-dpas-gate PRIVATE ${ESIMD_DPAS_GATE_TEST_SYCL_OPTIONS})
target_link_options(test-esimd-dpas-gate PRIVATE ${ESIMD_DPAS_GATE_TEST_SYCL_OPTIONS})
# gpu-arch.cpp's GGML_SYCL_DEBUG fallback logging resolves to globals defined in
# a libggml-sycl TU, which this test deliberately does not link. Same stub as
# test-gpu-arch uses.
target_compile_definitions(test-esimd-dpas-gate PRIVATE GGML_SYCL_GPU_ARCH_STANDALONE=1)
target_link_libraries(test-esimd-dpas-gate PRIVATE ggml-base)
if (IntelSYCL_FOUND)
    target_link_libraries(test-esimd-dpas-gate PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME esimd-dpas-gate COMMAND test-esimd-dpas-gate)
set_tests_properties(esimd-dpas-gate PROPERTIES
    LABELS "sycl;arch;dispatch;tdd"
    TIMEOUT 60
)

# test-rms-norm-mul-add-broadcast: discriminator for bit1's broadcast-vs-full-residual
# axis (llama.cpp-81gx, source committed c725c4a75, registration supplied by impl-w2).
# Submits real device work, so ONEAPI_DEVICE_SELECTOR is pinned to the validation card
# (level_zero:1) rather than left to whatever a bare ctest run picks up.
if (NOT GGML_BACKEND_DL)
add_executable(test-rms-norm-mul-add-broadcast
    tests/test-rms-norm-mul-add-broadcast.cpp
)
target_include_directories(test-rms-norm-mul-add-broadcast PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include ${SYCL_INCLUDE_DIR})
target_compile_definitions(test-rms-norm-mul-add-broadcast PRIVATE
    GGML_SYCL_WARP_SIZE=32 GGML_SYCL_DNNL=${GGML_SYCL_DNNL})
target_link_libraries(test-rms-norm-mul-add-broadcast PRIVATE ggml-base ggml ggml-sycl)
if (IntelSYCL_FOUND)
    target_link_libraries(test-rms-norm-mul-add-broadcast PRIVATE IntelSYCL::SYCL_CXX)
endif()
add_test(NAME rms-norm-mul-add-broadcast COMMAND test-rms-norm-mul-add-broadcast)
set_tests_properties(rms-norm-mul-add-broadcast PROPERTIES
    LABELS "sycl;fusion;rms-norm;tdd"
    ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1;LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 120)

# Persistent-worker graph-boundary reset proof. Pure host C++: it creates two
# std::threads but no SYCL queue, links no backend library, and takes no -fsycl
# options. A repeated handoff/reset cycle stress-tests the fixture and TLS reuse.
endif() # static SYCL backend fixture

add_executable(test-moe-layer-ids-worker-reset
    tests/test-moe-layer-ids-worker-reset.cpp
)
target_include_directories(test-moe-layer-ids-worker-reset PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(test-moe-layer-ids-worker-reset PRIVATE Threads::Threads)
add_test(NAME sycl-moe-worker-reset COMMAND test-moe-layer-ids-worker-reset)
set_tests_properties(sycl-moe-worker-reset PROPERTIES
    # Deliberately no ONEAPI_DEVICE_SELECTOR: this fixture never enumerates or
    # submits to a device. Labels also avoid the throttled sweep's cache denylist.
    LABELS "sycl;moe;host-only;tls"
    TIMEOUT 30
)

# AST/control-flow contract plus executable mutation matrix. tree-sitter parses
# active C++ nodes, so comments and literals cannot spoof reset/TLS/order checks.
# Python is test-only: its absence must not break a clean SYCL configuration.
find_package(Python3 QUIET COMPONENTS Interpreter)
if (Python3_Interpreter_FOUND)
    add_test(NAME sycl-moe-reset-source-contract
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test-moe-layer-ids-source-contract.py
            --source ${CMAKE_CURRENT_SOURCE_DIR}/ggml-sycl.cpp
            --header ${CMAKE_CURRENT_SOURCE_DIR}/moe-layer-ids-cache.hpp
            --mutation-matrix
    )
    set_tests_properties(sycl-moe-reset-source-contract PROPERTIES
        LABELS "sycl;moe;host-only;ast;mutation"
        SKIP_RETURN_CODE 77
        TIMEOUT 120
    )
else()
    message(STATUS "ggml-sycl: Python3 not found; AST reset contract test is not registered")
endif()
