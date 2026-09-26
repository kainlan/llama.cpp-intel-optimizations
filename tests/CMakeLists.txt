llama_add_compile_flags()

# Intel's SYCL link step records transient icpx/ocloc objects (written under
# $TMPDIR) in the linker depfile and then deletes them. Ninja sees a dependency
# that no longer exists and relinks the target on every build, forever. Same
# remedy as ggml/src/ggml-sycl/CMakeLists.txt and tools/CMakeLists.txt; it is a
# directory-scoped variable, so each directory holding an icpx-linked target
# needs its own copy. Four targets here link with -fsycl and were permanently
# dirty: test-sycl-fattn-onednn-gates, test-sycl-kernel-profiler,
# test-unified-cache-unpin-event, test-sycl-zone-reset-live-refusal.
# test-sycl-link-depends-coverage.py is the gate that keeps this from being
# forgotten in a fifth directory. See llama.cpp-9ubg.
if (CMAKE_GENERATOR STREQUAL "Ninja" AND CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    set(CMAKE_CXX_LINK_DEPENDS_USE_LINKER FALSE)
endif()

function(llama_build source)
    set(TEST_SOURCES ${source} ${ARGN})

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_link_libraries(${TEST_TARGET} PRIVATE llama llama-common)
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
endfunction()

function(llama_test target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    set(TEST_TARGET ${target})

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

function(llama_test_cmd target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND ${target}
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# Runs a pytest-style Python test file (module-level test_* functions, no
# __main__ guard) under the interpreter given as `target`.
#
# Such a file MUST be handed to pytest: `python3 <file>` would define the test
# functions, call none of them, and exit 0 for any state of the code under
# test. A plain script with a __main__ guard does NOT belong here -- register
# those with llama_test_cmd so they still run on a machine lacking pytest.
#
# Required args:
# - SCRIPT: path to the .py file
# Optional args:
# - NAME: name of the test (defaults to the script file name without extension)
# - LABEL: label(s) for the test (defaults to main)
# - TIMEOUT: per-test timeout in seconds (defaults to ctest's own default)
# - WORKING_DIRECTORY
function(llama_test_pytest target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME SCRIPT TIMEOUT WORKING_DIRECTORY)
    # LABEL is multi-valued so a ";"-joined list survives: ARGN flattens it into
    # separate arguments, and a one-value LABEL would silently keep only the
    # first and drop the rest into UNPARSED_ARGUMENTS.
    set(multiValueArgs LABEL)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_SCRIPT)
        message(FATAL_ERROR "llama_test_pytest: SCRIPT is required")
    endif()
    if (LLAMA_TEST_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "llama_test_pytest: unexpected arguments: ${LLAMA_TEST_UNPARSED_ARGUMENTS}")
    endif()
    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_NAME ${LLAMA_TEST_SCRIPT} NAME_WE)
    endif()

    # Exit 77 (skip) when pytest is not installed, then hand the file to
    # pytest. Probing at run time rather than at configure time keeps the test
    # registered on every machine, so a missing pytest surfaces as ctest
    # "Skipped" -- distinguishable from both "not registered" and "Passed".
    # A configure-time bail-out would instead leave `ctest -N` reporting
    # "Total Tests: 0", byte-identical to the test never having been added.
    #
    # pytest.main's exit code 5 ("no tests collected") is deliberately left as
    # a failure, so a file that stops collecting tests cannot pass vacuously.
    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND ${target}
            -c [=[
import importlib.util
import sys
if importlib.util.find_spec('pytest') is None:
    sys.exit(77)
import pytest
sys.exit(pytest.main(['-q', sys.argv[1]]))
]=]
            ${LLAMA_TEST_SCRIPT})

    set_tests_properties(${TEST_NAME} PROPERTIES
        LABELS "${LLAMA_TEST_LABEL}"
        SKIP_RETURN_CODE 77)
    if (DEFINED LLAMA_TEST_TIMEOUT)
        set_tests_properties(${TEST_NAME} PROPERTIES TIMEOUT ${LLAMA_TEST_TIMEOUT})
    endif()
endfunction()

# Builds and runs a test source file.
# Optional args:
# - NAME: name of the executable & test target (defaults to the source file name without extension)
# - LABEL: label for the test (defaults to main)
# - ARGS: arguments to pass to the test executable
# - WORKING_DIRECTORY
function(llama_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
    target_link_libraries(${TEST_TARGET} PRIVATE llama-common)

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    # Exit 77 is the skip status, matching llama_test_pytest above and the
    # test-sycl-*-policy.sh family. Without it a test that skips at run time --
    # the model-requiring ones via test_skip_no_model() in tests/test-skip.h --
    # would be reported "Passed", indistinguishable from a run that did the
    # work. Set here rather than per-registration so no future model-requiring
    # test can be added without it (llama.cpp-nwip).
    set_tests_properties(${TEST_TARGET} PROPERTIES
        LABELS "${LLAMA_TEST_LABEL}"
        SKIP_RETURN_CODE 77)
endfunction()

# build test-tokenizer-0 target once and add many tests
llama_build(test-tokenizer-0.cpp)

llama_test(test-tokenizer-0 NAME test-tokenizer-0-bert-bge          ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-bert-bge.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-command-r         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-command-r.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-coder.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm      ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-llm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-falcon            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gemma-4           ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gemma-4.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-bpe         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-spm         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-mpt               ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-phi-3             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-phi-3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen35            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen35.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-refact            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

if (NOT WIN32)
    llama_test_cmd(
        ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizers-repo.sh
        NAME test-tokenizers-ggml-vocabs
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        ARGS https://huggingface.co/ggml-org/vocabs ${PROJECT_SOURCE_DIR}/models/ggml-vocabs
    )
endif()

if (LLAMA_LLGUIDANCE)
    llama_build_and_test(test-grammar-llguidance.cpp ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
endif ()

llama_build(test-recurrent-state-rollback.cpp)

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API (when building with shared libraries)
    llama_build_and_test(test-unicode.cpp)
    llama_build_and_test(test-sampling.cpp)
    llama_build_and_test(test-reasoning-budget.cpp)
    llama_build_and_test(test-grammar-parser.cpp)
    llama_build_and_test(test-grammar-integration.cpp)
    llama_build_and_test(test-llama-grammar.cpp)
    llama_build_and_test(test-batch-alloc.cpp)
    llama_build_and_test(test-chat.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    target_include_directories(test-chat PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-chat PRIVATE server-context)
    # TODO: disabled on loongarch64 because the ggml-ci node lacks Python 3.8
    if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "loongarch64")
        llama_build_and_test(test-json-schema-to-grammar.cpp   WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
        target_include_directories(test-json-schema-to-grammar PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    endif()

    if (NOT GGML_BACKEND_DL)
        llama_build(test-quantize-stats.cpp)
    endif()

    llama_build(test-gbnf-validator.cpp)

    # build test-tokenizer-1-bpe target once and add many tests
    llama_build(test-tokenizer-1-bpe.cpp)

    # TODO: disabled due to slowness
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-aquila    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-aquila.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-falcon    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-2     ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-neox  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-neox.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-llama-bpe ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf --ignore-merges)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-mpt       ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-refact    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-starcoder ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

    # build test-tokenizer-1-spm target once and add many tests
    llama_build(test-tokenizer-1-spm.cpp)

    llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-llama-spm ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
    #llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-baichuan  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-baichuan.gguf)

    # llama_build_and_test(test-double-float.cpp) # SLOW

    llama_build_and_test(test-llama-archs.cpp)

    # Gates -x/--exclude of test-llama-archs (llama.cpp-fepd). Inside this block, not beside
    # test-archs-table below, because it calls llm_arch_from_string()/llm_arch_all() -- the same
    # internal symbols test-llama-archs uses, and the reason for the NOT WIN32 OR NOT
    # BUILD_SHARED_LIBS guard in the first place. It still loads no model and touches no device.
    #
    # On the name: NOT test-llama-archs-exclude. Four CI workflows carry `ctest -E
    # "test-llama-archs"` and -E is an UNANCHORED regex, so that name would be swallowed by the
    # pattern meant for the test it gates, and the gate would silently never run. See the note
    # on test-archs-table below -- this is the same trap, already sprung once.
    llama_build_and_test(test-archs-exclude.cpp)

    if (NOT WIN32)
        # The other half: that main() actually rejects a bad -x, which the unit test above
        # cannot see. Every invocation in the script exits inside argument parsing, so it never
        # reaches the model-loading sweep -- see the warning at the top of the script before
        # adding a probe to it.
        llama_test_cmd(
            bash
            NAME test-archs-exclude-cli
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            ARGS ${CMAKE_CURRENT_SOURCE_DIR}/test-archs-exclude-cli.sh $<TARGET_FILE:test-llama-archs>
        )
        # The script exits 77 (skip) when the binary it drives was not built.
        set_tests_properties(test-archs-exclude-cli PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    set(MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-models/")
    file(MAKE_DIRECTORY "${MODEL_DIR}")

    llama_test(
        test-llama-archs
        NAME test-generate-models
        LABEL main
        ARGS -o "${MODEL_DIR}"
    )
    set_tests_properties(test-generate-models PROPERTIES
        FIXTURES_SETUP generate-models
    )

    llama_test(
        test-recurrent-state-rollback
        LABEL main
        ARGS -m "${MODEL_DIR}/qwen35-dense.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback PROPERTIES
        FIXTURES_REQUIRED generate-models
    )

    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-nemotron-h
        LABEL main
        ARGS -m "${MODEL_DIR}/nemotron_h-dense.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback-nemotron-h PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-dsv4
        LABEL main
        ARGS -m "${MODEL_DIR}/deepseek4-moe.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback-dsv4 PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
endif()

# Gates the results table of test-llama-archs against the log output it shares a stream with
# (llama.cpp-to9m). Registered unconditionally and outside the block above: it drives the
# emitter in test-archs-table.h directly, so it loads no model, touches no device, and
# needs neither a vocab nor a backend.
#
# NOT named test-llama-archs-table, however obvious that name looks. Four CI workflows carry
# `ctest -E "test-llama-archs"` (build-apple.yml, build-openvino.yml x3) to disable the test
# this one gates, and `-E` is an UNANCHORED regex -- so that name is swallowed by the same
# pattern and the gate would silently never run. Measured: `ctest -N -E "test-llama-archs"`
# listed neither, `-E "^test-llama-archs$"` listed the gate. Anchoring the workflows would
# also fix it, but they are upstream files that a rebase reverts without a word, and the cost
# of losing that fix is a gate nobody knows has stopped running. A name the pattern cannot
# catch needs no upstream cooperation. Do not "tidy" this back.
llama_build_and_test(test-archs-table.cpp)

# Host-only gate for the batched-MoE MUL_MAT_ID coverage tables (llama.cpp-gx30).
# Deliberately built unconditionally and NOT under a SYCL guard: it includes only
# ggml.h plus a SYCL-free header, so it runs on any machine and needs no oneAPI
# and no GPU. Guarding it behind GGML_SYCL would make the one check that catches
# an over-advertised capability unavailable to exactly the agents who cannot
# build the backend.
llama_build_and_test(test-sycl-moe-mmvq-tables.cpp)
set_tests_properties(test-sycl-moe-mmvq-tables PROPERTIES LABELS "sycl;moe;source;hostonly")

# MoE route-table stamp: pure invalidation-decision logic (perf-recovery epic,
# track B, llama.cpp-s6ac). No device or oneDNN dependency in the tested
# portion -- see ggml-sycl/moe-route-table.hpp. The header's only backend
# include is mem-handle.hpp, which pulls in <sycl/sycl.hpp>, so (unlike
# test-sycl-moe-mmvq-tables.cpp above) this target needs the SYCL include dir
# and -fsycl even though the tested logic itself touches no device. The WHOLE
# registration is gated -- not just the extra include/compile wiring -- so the
# target and its ctest entry simply don't exist on a GGML_SYCL=OFF or
# GGML_BACKEND_DL=ON configure; mirrors test-unified-cache-unpin-event's
# pattern below, but links no backend library since the header's inline
# function is self-contained.
if (GGML_SYCL AND NOT GGML_BACKEND_DL)
    llama_build_and_test(test-sycl-route-table-stamp.cpp)
    set_tests_properties(test-sycl-route-table-stamp PROPERTIES LABELS "sycl;moe;route-table")
    target_include_directories(test-sycl-route-table-stamp PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src)
    if (TARGET IntelSYCL::SYCL_CXX)
        target_link_libraries(test-sycl-route-table-stamp PRIVATE IntelSYCL::SYCL_CXX)
    else()
        target_include_directories(test-sycl-route-table-stamp PRIVATE ${SYCL_INCLUDE_DIR})
        target_compile_options(test-sycl-route-table-stamp PRIVATE "-fsycl")
        target_link_options(test-sycl-route-table-stamp PRIVATE "-fsycl")
    endif()
endif()

# VRAM external headroom: pure default/floor arithmetic for the arena's
# driver-reserved headroom (perf-recovery epic, track D, llama.cpp-seno). No
# device or oneDNN dependency -- ggml-sycl/vram-headroom.hpp includes only
# <algorithm>/<cstddef>/<cstdlib>, not even mem-handle.hpp, so unlike
# test-sycl-route-table-stamp above this target needs neither the SYCL
# include dir nor -fsycl. Still registered under the same guard, adjacent to
# test-sycl-route-table-stamp, for consistency with the rest of this epic's
# SYCL-labelled test family.
if (GGML_SYCL AND NOT GGML_BACKEND_DL)
    llama_build_and_test(test-sycl-vram-headroom.cpp)
    set_tests_properties(test-sycl-vram-headroom PROPERTIES LABELS "sycl;vram-arena;headroom")
    target_include_directories(test-sycl-vram-headroom PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src)
endif()

# WOQ-MXFP4 capability spike: does this device's oneDNN build a matmul with
# f16 src, f4_e2m1 weights, and e8m0 grouped-{32,1} scales directly (perf-
# recovery epic, track C gate, llama.cpp-4m9p)? Unlike test-sycl-route-table-
# stamp above, this one calls real oneDNN entry points (primitive_desc,
# sycl_interop::make_engine/make_stream, prim.execute), so it needs actual
# DNNL::dnnl symbols, not just headers. That imported target is created by
# ggml/src/ggml-sycl/CMakeLists.txt's find_package(DNNL), which is directory-
# scoped to that subtree and not visible from here; re-issuing find_package
# (DNNL) in this directory is the standard fix -- the search itself is
# cached, so the second call is cheap, and it creates the target locally so
# it can be linked here. Exit codes: 0 = supported & numerics match (rel err
# <= 2e-2), 42 = primitive creation refused (documented RED outcome), 1 =
# wrong numerics, 43 = dnnl runtime error after primitive creation, 77 = no
# GPU (inherited SKIP_RETURN_CODE from llama_build_and_test).
if (GGML_SYCL AND NOT GGML_BACKEND_DL)
    llama_build_and_test(test-onednn-woq-mxfp4.cpp)
    set_tests_properties(test-onednn-woq-mxfp4 PROPERTIES LABELS "sycl;onednn;woq;mxfp4;tdd")
    if (TARGET IntelSYCL::SYCL_CXX)
        target_link_libraries(test-onednn-woq-mxfp4 PRIVATE IntelSYCL::SYCL_CXX)
    else()
        target_include_directories(test-onednn-woq-mxfp4 PRIVATE ${SYCL_INCLUDE_DIR})
        target_compile_options(test-onednn-woq-mxfp4 PRIVATE "-fsycl")
        target_link_options(test-onednn-woq-mxfp4 PRIVATE "-fsycl")
    endif()
    find_package(DNNL QUIET)
    if (TARGET DNNL::dnnl)
        target_link_libraries(test-onednn-woq-mxfp4 PRIVATE DNNL::dnnl)
    else()
        message(WARNING "test-onednn-woq-mxfp4: DNNL::dnnl not found in this directory scope; the target will fail to link")
    endif()
endif()

# SOA MXFP4 -> WOQ repack kernel device test (perf-recovery epic, track C,
# llama.cpp-8rug): the production repack_mxfp4_soa_to_woq (ggml-sycl/convert.
# cpp/.hpp) must reproduce a CPU reference of the nibble de-interleave +
# transpose byte-for-byte. Links the real ggml-sycl library (not just headers)
# to call the production kernel, mirroring test-unified-cache-unpin-event's
# pattern below rather than test-onednn-woq-mxfp4 above (that one only needs
# DNNL::dnnl + SYCL headers; this one needs an actual backend symbol).
if (GGML_SYCL AND NOT GGML_BACKEND_DL)
    llama_build_and_test(test-sycl-mxfp4-woq-repack.cpp)
    set_tests_properties(test-sycl-mxfp4-woq-repack PROPERTIES LABELS "sycl;moe;mxfp4;woq;tdd")
    target_include_directories(test-sycl-mxfp4-woq-repack PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src)
    target_link_libraries(test-sycl-mxfp4-woq-repack PRIVATE ggml-sycl)
    if (TARGET IntelSYCL::SYCL_CXX)
        target_link_libraries(test-sycl-mxfp4-woq-repack PRIVATE IntelSYCL::SYCL_CXX)
    else()
        target_include_directories(test-sycl-mxfp4-woq-repack PRIVATE ${SYCL_INCLUDE_DIR})
        target_compile_options(test-sycl-mxfp4-woq-repack PRIVATE "-fsycl")
        target_link_options(test-sycl-mxfp4-woq-repack PRIVATE "-fsycl")
    endif()
    # Pin off the measurement card (see test-unified-cache-unpin-event above for
    # why this is load-bearing, not decorative: an unqualified run otherwise
    # lands on whatever device SYCL enumerates first).
    set_tests_properties(test-sycl-mxfp4-woq-repack PROPERTIES
        ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1"
    )
endif()

# XMX_TILED MXFP4 -> WOQ repack kernel device test (perf-recovery epic, track
# C2b Option T, llama.cpp-ntfx): the production repack_mxfp4_xmx_tiled_to_woq
# (ggml-sycl/convert.cpp/.hpp) must invert the tiled materializer's byte
# layout into the same WOQ destination format as repack_mxfp4_soa_to_woq
# above, checked byte-for-byte against both a hand fixture and an
# independent-writer randomized fixture. Same gated pattern as
# test-sycl-mxfp4-woq-repack above (links the real ggml-sycl library).
if (GGML_SYCL AND NOT GGML_BACKEND_DL)
    llama_build_and_test(test-sycl-mxfp4-woq-tiled-repack.cpp)
    set_tests_properties(test-sycl-mxfp4-woq-tiled-repack PROPERTIES LABELS "sycl;moe;mxfp4;woq;tdd")
    target_include_directories(test-sycl-mxfp4-woq-tiled-repack PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src)
    target_link_libraries(test-sycl-mxfp4-woq-tiled-repack PRIVATE ggml-sycl)
    if (TARGET IntelSYCL::SYCL_CXX)
        target_link_libraries(test-sycl-mxfp4-woq-tiled-repack PRIVATE IntelSYCL::SYCL_CXX)
    else()
        target_include_directories(test-sycl-mxfp4-woq-tiled-repack PRIVATE ${SYCL_INCLUDE_DIR})
        target_compile_options(test-sycl-mxfp4-woq-tiled-repack PRIVATE "-fsycl")
        target_link_options(test-sycl-mxfp4-woq-tiled-repack PRIVATE "-fsycl")
    endif()
    # Pin off the measurement card -- see test-sycl-mxfp4-woq-repack above.
    set_tests_properties(test-sycl-mxfp4-woq-tiled-repack PROPERTIES
        ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1"
    )
endif()

# Regression gate for llama.cpp-yke2 -- see dpct/helper.hpp's
# create_queue_impl() comment for the interposition mechanism this fixes,
# and the test file's own header comment for why it has two phases. Links
# the real ggml-sycl library (like test-sycl-mxfp4-woq-repack above) to
# exercise the actual ggml_backend_sycl_context::stream() path, not a
# private workaround queue.
if (GGML_SYCL AND NOT GGML_BACKEND_DL)
    llama_build_and_test(test-sycl-profiling-queue-property.cpp)
    set_tests_properties(test-sycl-profiling-queue-property PROPERTIES LABELS "sycl;profiler;gpu-serial")
    target_include_directories(test-sycl-profiling-queue-property PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src)
    target_link_libraries(test-sycl-profiling-queue-property PRIVATE ggml-sycl)
    if (TARGET IntelSYCL::SYCL_CXX)
        target_link_libraries(test-sycl-profiling-queue-property PRIVATE IntelSYCL::SYCL_CXX)
    else()
        target_include_directories(test-sycl-profiling-queue-property PRIVATE ${SYCL_INCLUDE_DIR})
        target_compile_options(test-sycl-profiling-queue-property PRIVATE "-fsycl")
        target_link_options(test-sycl-profiling-queue-property PRIVATE "-fsycl")
    endif()
    # RUN_SERIAL and the LD_LIBRARY_PATH addition mirror
    # test-sycl-two-context-ownership above: a direct (non-ctest) invocation
    # of a SYCL test binary without oneAPI's redist lib on LD_LIBRARY_PATH
    # prints SKIP with exit 0 (CLAUDE.md: "a SYCL test binary run WITHOUT
    # sourcing setvars.sh prints SKIP and exits 0") -- ctest supplies it here
    # so a `ctest -R` run does not silently pass for the wrong reason.
    set_tests_properties(test-sycl-profiling-queue-property PROPERTIES
        RUN_SERIAL TRUE
        TIMEOUT 300
        ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1;LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}")
endif()

# WOQ repack kernel microbench (perf-recovery epic, track D, llama.cpp-0vqt):
# NOT a ctest -- it is a device-event throughput measurement, not a
# correctness gate (that's test-sycl-mxfp4-woq-{,tiled-}repack above), so it
# uses add_executable directly rather than llama_build_and_test. Build with
# `./scripts/sycl-build.sh test-sycl-mxfp4-woq-repack-bench` and run manually
# (see the file header for the full command, device selector required).
if (GGML_SYCL AND NOT GGML_BACKEND_DL)
    add_executable(test-sycl-mxfp4-woq-repack-bench test-sycl-mxfp4-woq-repack-bench.cpp)
    target_include_directories(test-sycl-mxfp4-woq-repack-bench PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src ${CMAKE_SOURCE_DIR}/ggml/include)
    target_link_libraries(test-sycl-mxfp4-woq-repack-bench PRIVATE ggml-sycl)
    if (TARGET IntelSYCL::SYCL_CXX)
        target_link_libraries(test-sycl-mxfp4-woq-repack-bench PRIVATE IntelSYCL::SYCL_CXX)
    else()
        target_include_directories(test-sycl-mxfp4-woq-repack-bench PRIVATE ${SYCL_INCLUDE_DIR})
        target_compile_options(test-sycl-mxfp4-woq-repack-bench PRIVATE "-fsycl")
        target_link_options(test-sycl-mxfp4-woq-repack-bench PRIVATE "-fsycl")
    endif()
endif()

# oneDNN int8-with-grouped-scales matmul probe + microbench (track prefill of
# epic llama.cpp-xihy, llama.cpp-ovkn): a capability/perf probe, NOT a
# correctness gate and NOT a ctest -- it opens a GPU device and must never run
# inside a ctest sweep, so add_executable only (no llama_build_and_test, no
# add_test). Standalone like test-onednn-woq-mxfp4 above (dnnl + sycl headers
# only, no ggml/backend deps), so it needs the same directory-local
# find_package(DNNL) re-issue and IntelSYCL::SYCL_CXX/-fsycl fallback --
# copied from that block -- plus GGML_SYCL_DNNL=1 so the file's `#if
# GGML_SYCL_DNNL` guard compiles the real probe body instead of the SKIP
# stub. Build with `./scripts/sycl-build.sh test-sycl-onednn-int8-grouped-
# probe` and run manually (see the file header for the full command, device
# selector required).
if (GGML_SYCL AND NOT GGML_BACKEND_DL)
    add_executable(test-sycl-onednn-int8-grouped-probe test-sycl-onednn-int8-grouped-probe.cpp)
    if (TARGET IntelSYCL::SYCL_CXX)
        target_link_libraries(test-sycl-onednn-int8-grouped-probe PRIVATE IntelSYCL::SYCL_CXX)
    else()
        target_include_directories(test-sycl-onednn-int8-grouped-probe PRIVATE ${SYCL_INCLUDE_DIR})
        target_compile_options(test-sycl-onednn-int8-grouped-probe PRIVATE "-fsycl")
        target_link_options(test-sycl-onednn-int8-grouped-probe PRIVATE "-fsycl")
    endif()
    find_package(DNNL QUIET)
    if (TARGET DNNL::dnnl)
        target_link_libraries(test-sycl-onednn-int8-grouped-probe PRIVATE DNNL::dnnl)
        target_compile_definitions(test-sycl-onednn-int8-grouped-probe PRIVATE GGML_SYCL_DNNL=1)
    else()
        message(WARNING "test-sycl-onednn-int8-grouped-probe: DNNL::dnnl not found in this directory scope; the target will fail to link")
    endif()
endif()

# Custom PP GEMM activation-format microbench (llama.cpp-iikr, stage 1):
# same rationale as the repack bench above -- a device-event throughput
# measurement, not a correctness gate, so add_executable directly. Build
# with `./scripts/sycl-build.sh test-sycl-mxfp4-woq-gemm-bench` and run
# manually (see the file header for the full command, device selector
# required).
if (GGML_SYCL AND NOT GGML_BACKEND_DL)
    add_executable(test-sycl-mxfp4-woq-gemm-bench test-sycl-mxfp4-woq-gemm-bench.cpp)
    target_include_directories(test-sycl-mxfp4-woq-gemm-bench PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src ${CMAKE_SOURCE_DIR}/ggml/include)
    target_link_libraries(test-sycl-mxfp4-woq-gemm-bench PRIVATE ggml-sycl)
    if (TARGET IntelSYCL::SYCL_CXX)
        target_link_libraries(test-sycl-mxfp4-woq-gemm-bench PRIVATE IntelSYCL::SYCL_CXX)
    else()
        target_include_directories(test-sycl-mxfp4-woq-gemm-bench PRIVATE ${SYCL_INCLUDE_DIR})
        target_compile_options(test-sycl-mxfp4-woq-gemm-bench PRIVATE "-fsycl")
        target_link_options(test-sycl-mxfp4-woq-gemm-bench PRIVATE "-fsycl")
    endif()
endif()

# MoE routing device-vs-CPU correctness oracle (llama.cpp-iikr, device-
# routing pivot stage i, owner ruling task comment c-cgta): a correctness
# gate, not a throughput measurement, but registered add_executable-only
# like the two SYCL benches above rather than via llama_build_and_test --
# it needs GGML_SYCL_MOE_ROUTING_DEVICE semantics (set internally by the
# test itself before any SYCL call, see the file header) and RED/GREEN
# verification on real hardware by hand, not a default ctest sweep run.
# Uses only the portable ggml-backend API (ggml_backend_cpu_init/
# ggml_backend_sycl_init/ggml_backend_graph_compute) -- no raw SYCL/oneDNN
# code in the test file itself, so it needs llama-common's link surface
# (same as test-onednn-woq-mxfp4 above) but none of that test's
# IntelSYCL::SYCL_CXX / -fsycl compile-flag handling.
if (GGML_SYCL AND NOT GGML_BACKEND_DL)
    add_executable(test-sycl-moe-routing-device-oracle test-sycl-moe-routing-device-oracle.cpp)
    target_link_libraries(test-sycl-moe-routing-device-oracle PRIVATE llama-common)
endif()

llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp)
llama_build_and_test(test-jinja.cpp)
llama_test(test-jinja NAME test-jinja-py ARGS -py LABEL python)
llama_build_and_test(test-chat-auto-parser.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
llama_build_and_test(test-chat-template.cpp)
# debug tool for chat template differential analysis (not registered as a test, run it manually)
llama_build(test-chat-analysis.cpp)
llama_build_and_test(test-log.cpp)
llama_build_and_test(
    test-peg-parser.cpp
    peg-parser/simple-tokenize.cpp
    peg-parser/test-basic.cpp
    peg-parser/test-gbnf-generation.cpp
    peg-parser/test-json-parser.cpp
    peg-parser/test-json-serialization.cpp
    peg-parser/test-python-dict-parser.cpp
    peg-parser/test-unicode.cpp
    peg-parser/tests.h
)

if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "s390x")
    set(MODEL_NAME "tinyllamas/stories15M-q4_0.gguf")
    set(MODEL_HASH "SHA256=66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739")
else()
    set(MODEL_NAME "tinyllamas/stories15M-be.Q4_0.gguf")
    set(MODEL_HASH "SHA256=9aec857937849d976f30397e97eb1cabb53eb9dcb1ce4611ba8247fb5f44c65d")
endif()
set(MODEL_DEST "${CMAKE_BINARY_DIR}/${MODEL_NAME}")

add_test(NAME test-download-model COMMAND ${CMAKE_COMMAND}
    -DDEST=${MODEL_DEST}
    -DNAME=${MODEL_NAME}
    -DHASH=${MODEL_HASH}
    -P ${CMAKE_SOURCE_DIR}/cmake/download-models.cmake
)
set_tests_properties(test-download-model PROPERTIES FIXTURES_SETUP test-download-model)

option(LLAMA_SYCL_G1_STRICT "Make missing G1 acceptance prerequisites fail instead of skip" OFF)
if (LLAMA_SYCL_G1_STRICT AND NOT GGML_SYCL)
    message(FATAL_ERROR "LLAMA_SYCL_G1_STRICT requires GGML_SYCL=ON")
endif()
find_package(Python3 QUIET COMPONENTS Interpreter)
if (LLAMA_SYCL_G1_STRICT AND NOT Python3_Interpreter_FOUND)
    message(FATAL_ERROR "LLAMA_SYCL_G1_STRICT requires a Python 3 interpreter")
endif()
if (Python3_Interpreter_FOUND)
    add_test(NAME test-backend-count-get-null-census
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-backend-count-get-null-census.py)
    set_tests_properties(test-backend-count-get-null-census PROPERTIES
        LABELS "backend;lifecycle;source;host" TIMEOUT 30)
    # llama.cpp-ttws: the host-weight registry must classify a name collision
    # (tied token_embd/output alias vs a genuine conflict) before it warns.
    # Structural gate on the print site in ggml-sycl.cpp; the classifier's own
    # behaviour is covered by sycl-host-weight-alias (ggml/src/ggml-sycl).
    add_test(NAME test-sycl-host-weight-alias-source
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-host-weight-alias-source.py)
    set_tests_properties(test-sycl-host-weight-alias-source PROPERTIES
        LABELS "sycl;registry;source;source-contract;hostonly" TIMEOUT 30)
    # An over-capacity CPU expert dispatch must fall back, not abort (llama.cpp-sfal).
    # Run WITH --self-test: a text-only check whose anchors stop matching real code
    # passes vacuously, and --self-test requires its 5 mutants to still be caught.
    add_test(NAME test-sycl-cpu-expert-pool-capacity-source
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-cpu-expert-pool-capacity-source.py --self-test)
    set_tests_properties(test-sycl-cpu-expert-pool-capacity-source PROPERTIES
        LABELS "sycl;moe;source;source-contract;hostonly" TIMEOUT 30)
    # A MoE host-expert dispatch must bridge a storage-handle miss, not kill the
    # graph (llama.cpp-srti). Also runs WITH --self-test: the per-site checks are
    # anchored on refusal-message text, so a renamed site would silently stop
    # being checked -- the mutants prove each anchor is still load-bearing.
    add_test(NAME test-sycl-moe-cpu-dispatch-storage-bridge-source
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-moe-cpu-dispatch-storage-bridge-source.py --self-test)
    set_tests_properties(test-sycl-moe-cpu-dispatch-storage-bridge-source PROPERTIES
        LABELS "sycl;moe;source;source-contract;hostonly" TIMEOUT 30)
    # Consumer-side half of the MMVQ coverage invariant (llama.cpp-gx30). The C++
    # gate checks capability against the launchers that EXIST; this checks it
    # against what each consumer's own switch ENUMERATES. Census 9b died in the
    # gap between those two readings.
    add_test(NAME test-sycl-moe-mmvq-consumer-coverage
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-moe-mmvq-consumer-coverage.py)
    set_tests_properties(test-sycl-moe-mmvq-consumer-coverage PROPERTIES
        LABELS "sycl;moe;source;hostonly" TIMEOUT 30)
    add_test(NAME test-sycl-lifecycle-g1-harness
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-lifecycle-g1-harness.py)
    set_tests_properties(test-sycl-lifecycle-g1-harness PROPERTIES
        LABELS "sycl;lifecycle;g1;python"
        TIMEOUT 30)
    llama_test_pytest(${Python3_EXECUTABLE}
        NAME test-sycl-device-uuid-source
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-device-uuid-source.py
        LABEL "sycl;api;source;host"
        TIMEOUT 30)
    # WOQ scratch-slot byte formula must match between its two independent
    # sites (plan-time src/llama-model.cpp, per-dispatch ggml-sycl.cpp) --
    # they cannot share a helper, so this is the only thing keeping them
    # from drifting apart. llama.cpp-sr83, quality review coverage gap (a).
    #
    # llama_test_cmd rather than llama_test_pytest: this is a plain script
    # with top-level checks, not pytest test_ functions (see
    # llama_test_pytest's own doc comment above; matches
    # test-sycl-zone-reset-audit-source's own llama_test_cmd registration
    # further below in this file -- test-sycl-pp-moe-scratch-source.py is
    # NOT a llama_test_cmd precedent, it is registered via a bare add_test
    # in ggml/src/ggml-sycl/CMakeLists.txt instead). Host-only -- reads two
    # source files as text, builds nothing, touches no device.
    llama_test_cmd(
        ${Python3_EXECUTABLE}
        NAME test-sycl-woq-slot-formula-source
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        ARGS ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-woq-slot-formula-source.py
    )
    set_tests_properties(test-sycl-woq-slot-formula-source PROPERTIES
        LABELS "sycl;moe;woq;source;hostonly"
        TIMEOUT 30)
endif()

if (GGML_SYCL)
    set(LLAMA_SYCL_G1_FIXTURE "${CMAKE_CURRENT_SOURCE_DIR}/sycl-lifecycle-fixtures.json" CACHE FILEPATH
        "Canonical G1 fixture JSON (lead-populated external file is recommended)")
    llama_build(test-sycl-lifecycle-gpu-sequential.cpp)
    target_include_directories(test-sycl-lifecycle-gpu-sequential PRIVATE ${PROJECT_SOURCE_DIR}/src)
    add_test(NAME sycl-lifecycle-g1-parser
        COMMAND test-sycl-lifecycle-gpu-sequential --self-test-parser)
    set_tests_properties(sycl-lifecycle-g1-parser PROPERTIES LABELS "sycl;lifecycle;g1;host")

    set(G1_LAUNCH_ARGS --fixture ${LLAMA_SYCL_G1_FIXTURE})
    if (LLAMA_SYCL_G1_STRICT)
        list(PREPEND G1_LAUNCH_ARGS --strict)
    endif()
    # Exact lab gate:
    # cmake -S . -B build -DGGML_SYCL=ON -DLLAMA_SYCL_G1_STRICT=ON \
    #   -DLLAMA_SYCL_G1_FIXTURE=/absolute/path/g1-canonical.json
    # ctest --test-dir build -R '^sycl-lifecycle-gpu-sequential$' --output-on-failure
    # The test name is intentionally identical in normal and strict modes.
    if (Python3_Interpreter_FOUND)
        add_test(NAME sycl-lifecycle-gpu-sequential
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-lifecycle-g1-aba.py
                    ${G1_LAUNCH_ARGS} $<TARGET_FILE:test-sycl-lifecycle-gpu-sequential>)
        set_tests_properties(sycl-lifecycle-gpu-sequential PROPERTIES
            SKIP_RETURN_CODE 77
            RUN_SERIAL TRUE
            RESOURCE_LOCK sycl-lifecycle-device
            TIMEOUT 950
            LABELS "sycl;lifecycle;g1;hash-pinned")
    endif()
endif()

llama_build_and_test(test-thread-safety.cpp ARGS -m "${MODEL_DEST}" -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2)
set_tests_properties(test-thread-safety PROPERTIES FIXTURES_REQUIRED test-download-model)
# NOTE (llama.cpp-u7vj): this test used to pin GGML_SYCL_DISABLE_GRAPH=1 here.
# That pin was a workaround for a production defect, and the comment justifying
# it blamed the wrong mechanism -- it read as though graph replay itself held
# the execution registry's per-device invocation. It did not. The invocation was
# acquired on every path but submitted on only one: thirteen bypass paths in
# ggml_backend_sycl_graph_compute_unchecked() ran the compute and returned early,
# leaving the graph SEALED rather than COMPLETE, so release_invocation() refused
# it forever and every other context's first decode was refused DEVICE_BUSY. Two
# of those bypasses were not even graph-gated, so disabling replay was never
# general immunity -- it only steered past the ones this workload happened to hit.
#
# The pairing is fixed in ggml-sycl.cpp (execution_tail_guard), so this test now
# runs the DEFAULT configuration, and running it unpinned is the regression test
# for that fix: a regression reappears as "error=DEVICE_BUSY" in the log and a
# failure of every context after the first. Do not re-add the pin to make a
# failure go away -- it would hide exactly the defect this test now covers.

# Two live contexts on one model, different n_ctx, interleaved decodes, each
# step compared against a single-context reference (llama.cpp-32dg8.15.1). This
# is the executable half of docs/design/sycl-memory-ownership-table.md. It
# LOADS A MODEL, so it carries the "cache" label on purpose: CLAUDE.md's
# full-sweep form excludes `-LE 'residency|mem-handle|cache'` and runs that
# family serially -- an unlabelled model-loading test slips past that filter.
# The selector pin is what keeps the iGPU out of the run (llama.cpp-403s).
# Guarded by GGML_SYCL so a build without the SYCL backend compiled in never
# tries to build/register this test at all (precedent:
# test-sycl-model-lifecycle-hooks below). That is all this CMake guard claims:
# a SYCL build that enumerates no GPU device, or whose backend init silently
# falls back to CPU, is instead caught at RUNTIME by the
# llama_supports_gpu_offload() check inside the test itself, which exits 77
# (skip) rather than passing vacuously.
if (GGML_SYCL)
    llama_build_and_test(test-sycl-two-context-ownership.cpp ARGS "${MODEL_DEST}" -ngl 99 --ctx-a 256 --ctx-b 512)
    set_tests_properties(test-sycl-two-context-ownership PROPERTIES
        FIXTURES_REQUIRED test-download-model
        LABELS "main;sycl;cache;ownership;two-context"
        RUN_SERIAL TRUE
        TIMEOUT 300
        ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1;LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}")
endif()

# llama.cpp-tsfl (F-lead-1, orphan-deliverable fix): calls
# ggml_backend_sycl_probe_runtime_context_for_model() on a live GPT-OSS 20B
# context -- the Task 2 plan's own acceptance criterion ("probe(512) on B50
# GPT-OSS returns accepted ...; probe(8192) returns refused ... (lead-run)")
# had no caller anywhere in the tree before this harness. Guarded the same
# way test-quantize-stats.cpp is above (NOT GGML_BACKEND_DL): this harness
# includes the private src/llama-model.h AND calls ggml-sycl.h functions
# directly (not merely referencing their types the way
# test-sycl-lifecycle-public-api.cpp's ABI gate does), both of which require
# ggml-sycl to be linked in statically rather than loaded as a
# GGML_BACKEND_DL module. The model is LOCAL (not the downloaded fixture),
# so unlike test-sycl-two-context-ownership above, no FIXTURES_REQUIRED.
if (GGML_SYCL AND NOT GGML_BACKEND_DL)
    llama_build_and_test(test-sycl-runtime-context-probe.cpp ARGS /models/gpt-oss-20b-mxfp4.gguf)
    set_tests_properties(test-sycl-runtime-context-probe PROPERTIES
        LABELS "main;sycl;cache;probe"
        RUN_SERIAL TRUE
        TIMEOUT 600
        SKIP_RETURN_CODE 77
        ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1;LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}")
endif()

llama_build_and_test(test-arg-parser.cpp)
# llama.cpp-y8xv quality round 1, Q5: host-only test for the -ub range
# parser hoisted into tools/llama-bench/llama-bench-parse.hpp -- pulls in
# only that header, not llama-bench itself.
llama_build_and_test(test-llama-bench-parse.cpp)
target_include_directories(test-llama-bench-parse PRIVATE ${PROJECT_SOURCE_DIR}/tools/llama-bench)
llama_build_and_test(test-model-resolution.cpp)
# the test serves its repos from an httplib server, and the library links it privately
target_link_libraries(test-model-resolution PRIVATE cpp-httplib)

if (NOT WIN32)
    llama_test_cmd(
        bash
        NAME test-sycl-alloc-policy
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        ARGS ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-alloc-policy.sh
    )
    # The script exits 77 (skip) when neither ripgrep nor GNU grep is available
    # (e.g. macOS BSD grep runners); treat that as a skip, not a failure.
    set_tests_properties(test-sycl-alloc-policy PROPERTIES SKIP_RETURN_CODE 77)
endif()

if (NOT WIN32)
    # bench-guard preflight/postflight unit tests (perf-recovery epic, track A,
    # llama.cpp-r780): refuses measurement on a throttled/tenant-occupied/
    # Shmem-elevated host and stamps every archived log VALID/SUSPECT. Pure
    # bash against a fake sysfs/meminfo tree via overridable --sysfs-card/
    # --meminfo/--pgrep-cmd/--journalctl-cmd roots -- no GPU, safe at any
    # parallelism. See docs/backend/sycl-perf-baselines.md "Measurement
    # protocol" for the mandate this enforces.
    llama_test_cmd(
        bash
        NAME test-bench-guard
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        ARGS ${CMAKE_CURRENT_SOURCE_DIR}/test-bench-guard.sh
    )
    # NOT `LABEL "sycl;source"` on the llama_test_cmd call above -- same
    # llama_test_cmd single-valued-LABEL caveat as test-sycl-decode-mode-
    # capture's own registration below.
    set_tests_properties(test-bench-guard PROPERTIES
        LABELS "sycl;source"
        SKIP_RETURN_CODE 77
    )

    # sycl-gpu-preflight unit tests (llama.cpp-o4fs): the live B50 PCI-
    # address derivation sycl_preflight_b50_pci_address/
    # sycl_preflight_b50_sysfs_bad now share with bench-guard.sh via
    # scripts/sycl-gpu-sysfs.sh (above), instead of the fixed
    # /sys/bus/pci/devices/0000:07:00.0 address this file hardcoded before.
    # Pure bash against a fake DRM/PCI sysfs tree via
    # SYCL_PREFLIGHT_DRM_ROOT/SYCL_PREFLIGHT_PCI_ROOT env overrides -- no
    # GPU, safe at any parallelism.
    llama_test_cmd(
        bash
        NAME test-sycl-gpu-preflight
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        ARGS ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-gpu-preflight.sh
    )
    # NOT `LABEL "sycl;source"` on the llama_test_cmd call above -- same
    # llama_test_cmd single-valued-LABEL caveat as test-sycl-decode-mode-
    # capture's own registration below.
    set_tests_properties(test-sycl-gpu-preflight PROPERTIES
        LABELS "sycl;source"
        SKIP_RETURN_CODE 77
    )

    # sycl-decode-mode-capture unit tests (plan task S4, llama.cpp-gvu7): the
    # one-run freq/throttle/RssAnon timeline + kprof capture for the B70
    # GPT-OSS decode slow mode. Invokes bench-guard.sh (above) as a child for
    # all host-corruption preflight; this suite exercises only the capture
    # script itself, against a fake sysfs tree and a fake bench command --
    # no GPU, safe at any parallelism.
    llama_test_cmd(
        bash
        NAME test-sycl-decode-mode-capture
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        ARGS ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-decode-mode-capture.sh
    )
    # NOT `LABEL "sycl;source"` on the llama_test_cmd call above:
    # llama_test_cmd's LABEL is a single-valued (oneValueArgs) argument, so
    # the embedded ";" in "sycl;source" splits into two separate arguments
    # when cmake_parse_arguments consumes it -- LABEL takes only "sycl", and
    # "source" silently falls into UNPARSED_ARGUMENTS, which the function
    # never reads. llama_test_pytest's own LABEL is multi-valued precisely
    # to avoid this (see its own doc comment above), which is why an earlier
    # round of this comment cited a llama_test_pytest call as precedent for
    # llama_test_cmd's LABEL argument -- llama_test_pytest is a different
    # function with a different LABEL, and the precedent didn't transfer
    # (round-3 review, llama.cpp-gvu7). No fixed TIMEOUT is set here despite
    # the neighbouring test-sycl-handle-policy registration carrying one:
    # this suite's own runtime varies with host load by design (see its own
    # file header), and a fixed timeout tuned for a quiet host would flake
    # under load the same way an under-provisioned test fixture already did
    # once (quality review, finding 8).
    set_tests_properties(test-sycl-decode-mode-capture PROPERTIES
        LABELS "sycl;source"
        SKIP_RETURN_CODE 77
    )

    # sycl-prefill-scaling unit tests (plan task L3, llama.cpp-dfo0): the
    # pp128/512/1024/2048 markdown-table parser and the ratio1024=pp1024/
    # pp512 >= 0.9 scaling-collapse verdict, for all six model/card pairs.
    # Invokes bench-guard.sh (above) as a child for all host-corruption
    # preflight, exactly like test-sycl-decode-mode-capture above; this
    # suite exercises only the parser and verdict logic, against a fake
    # sysfs tree and a fake `llama-bench` -- no GPU, safe at any
    # parallelism.
    llama_test_cmd(
        bash
        NAME test-sycl-prefill-scaling
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        ARGS ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-prefill-scaling.sh
    )
    # Same LABEL caveat as test-sycl-decode-mode-capture above applies here
    # (llama_test_cmd's LABEL is single-valued) -- set via
    # set_tests_properties instead.
    set_tests_properties(test-sycl-prefill-scaling PROPERTIES
        LABELS "sycl;source"
        SKIP_RETURN_CODE 77
    )
endif()

if (NOT WIN32)
    llama_test_cmd(
        bash
        NAME test-sycl-build-oneccl-discovery
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        ARGS ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-build-oneccl-discovery.sh
    )
    set_tests_properties(test-sycl-build-oneccl-discovery PROPERTIES
        LABELS "sycl;build;hostonly"
        TIMEOUT 30
    )
endif()

# llama.cpp-vuy0: the SYCL device-link launcher (private TMPDIR removed on
# exit or signal, toolchain-keyed ocloc cache, inventory recorder), exercised
# against a mock compiler and ocloc; the list-only report of temporaries
# earlier, unwrapped links stranded; sycl-build.sh's multi-target build and
# private TMPDIR, against a mock cmake; the device-image inventory comparison;
# and a check on the generated Ninja files
# that every device-linking edge actually runs under that launcher and pool.
if (NOT WIN32)
    llama_test_cmd(
        bash
        NAME test-sycl-device-link-launcher
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        ARGS ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-device-link-launcher.sh
    )
    set_tests_properties(test-sycl-device-link-launcher PROPERTIES
        LABELS "sycl;build;hostonly"
        TIMEOUT 60
    )
    llama_test_cmd(
        bash
        NAME test-sycl-tmp-leak-report
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        ARGS ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-tmp-leak-report.sh
    )
    # Scans all of /proc twice; 33 s measured at load 51.
    set_tests_properties(test-sycl-tmp-leak-report PROPERTIES
        LABELS "sycl;build;hostonly"
        TIMEOUT 180
    )
    llama_test_cmd(
        bash
        NAME test-sycl-build-targets
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        ARGS ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-build-targets.sh
    )
    set_tests_properties(test-sycl-build-targets PROPERTIES
        LABELS "sycl;build;hostonly"
        TIMEOUT 30
    )
    llama_test_cmd(
        bash
        NAME test-sycl-device-image-inventory
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        ARGS ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-device-image-inventory.sh
    )
    set_tests_properties(test-sycl-device-image-inventory PROPERTIES
        LABELS "sycl;build;hostonly"
        TIMEOUT 30
    )
endif()
# Embedded in a parent project, the device-link routing must leave the
# parent's own targets alone. Configures small fixtures with the host C
# compiler; the routing module needs CMake 3.21, as in ggml-sycl.
if (NOT WIN32 AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.21)
    llama_test_cmd(
        bash
        NAME test-sycl-device-link-embedding
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        ARGS ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-device-link-embedding.sh
             "${CMAKE_COMMAND}" "${CMAKE_GENERATOR}" "${CMAKE_MAKE_PROGRAM}"
    )
    set_tests_properties(test-sycl-device-link-embedding PROPERTIES
        LABELS "sycl;build;hostonly"
        SKIP_RETURN_CODE 77
        TIMEOUT 60
    )
endif()
if (GGML_SYCL AND NOT WIN32 AND CMAKE_GENERATOR MATCHES "Ninja" AND Python3_Interpreter_FOUND)
    add_test(NAME test-sycl-device-link-routing
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-device-link-routing.py ${CMAKE_BINARY_DIR})
    set_tests_properties(test-sycl-device-link-routing PROPERTIES
        LABELS "sycl;build;hostonly"
        SKIP_RETURN_CODE 77
        TIMEOUT 60
    )
    # Tests that read repo sources must find the tree without an absolute
    # __FILE__, which ccache base_dir (opt-in, GGML_SYCL_CCACHE_BASE_DIR=1)
    # takes away (llama.cpp-vuy0).
    add_test(NAME test-sycl-source-root-anchor
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-source-root-anchor.py ${CMAKE_BINARY_DIR})
    set_tests_properties(test-sycl-source-root-anchor PROPERTIES
        LABELS "sycl;build;hostonly;source-contract"
        SKIP_RETURN_CODE 77
        TIMEOUT 60
    )
endif()
# The commit id and build number must stay off every edge command that does not
# need them, or each commit reruns those edges (llama.cpp-vuy0). The script
# reads build.ninja, so single-config Ninja only, and it checks ggml-base's
# compiles, which a LLAMA_USE_SYSTEM_GGML build does not have.
if (CMAKE_GENERATOR STREQUAL "Ninja" AND NOT LLAMA_USE_SYSTEM_GGML AND Python3_Interpreter_FOUND)
    add_test(NAME test-build-info-scope
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-build-info-scope.py ${CMAKE_BINARY_DIR})
    set_tests_properties(test-build-info-scope PROPERTIES
        LABELS "build;hostonly"
        SKIP_RETURN_CODE 77
        TIMEOUT 60
    )
endif()

if (NOT WIN32)
    llama_test_cmd(
        bash
        NAME test-sycl-handle-policy
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        ARGS ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-handle-policy.sh
    )
    set_tests_properties(test-sycl-handle-policy PROPERTIES
        LABELS "sycl;policy"
        SKIP_RETURN_CODE 77
        TIMEOUT 120
    )
endif()

# ensure_cached_alloc() is a TEST-ONLY seam whose enforcement was, until now,
# the zero-production-caller state itself plus a grep written into
# unified-cache.hpp's declaration comment -- a convention nothing executed
# (llama.cpp-kwwv, following llama.cpp-og9dt). This runs that grep.
#
# LABELS avoid the throttled sweep's -LE 'residency|mem-handle|cache' denylist
# (rule 2 in ggml/src/ggml-sycl/CMakeLists.txt): the script greps text, opens no
# device and allocates nothing, so it must not be filtered out of the sweep --
# which is the only thing that would ever run it.
if (NOT WIN32)
    llama_test_cmd(
        bash
        NAME test-sycl-ensure-cached-alloc-policy
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        ARGS ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-ensure-cached-alloc-policy.sh
    )
    set_tests_properties(test-sycl-ensure-cached-alloc-policy PROPERTIES
        LABELS "sycl;policy;source-contract;hostonly"
        SKIP_RETURN_CODE 77
        TIMEOUT 120
    )
endif()

if (NOT WIN32)
    # Source assertion, so it is registered here rather than under the SYCL
    # backend's own test guard: the code it checks lives inside
    # #ifdef GGML_SYCL_XMX_GEMM (OFF by default, and not independently buildable
    # -- llama.cpp-d6d6), so a gate that needed that code compiled could never
    # run. Grepping the source sidesteps both.
    llama_test_cmd(
        bash
        NAME test-sycl-xmx-threshold-policy
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        ARGS ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-xmx-threshold-policy.sh
    )
    set_tests_properties(test-sycl-xmx-threshold-policy PROPERTIES
        LABELS "sycl;policy"
        SKIP_RETURN_CODE 77
        TIMEOUT 120
    )
endif()

if (NOT WIN32)
    # Source assertion for the ggml_tensor_extra_gpu device-index invariant. A
    # member added without a bounds check compiles, links and passes every
    # runtime gate -- nothing executes the out-of-range path -- so only the
    # source can be interrogated.
    llama_test_cmd(
        bash
        NAME test-sycl-device-index-policy
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        ARGS ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-device-index-policy.sh
    )
    # Labels deliberately match the siblings: neither "sycl" nor "policy" is in
    # the residency|mem-handle|cache denylist, so this stays inside the throttled
    # sweep that is the only suite routinely running it. A test can be correctly
    # registered and still never execute.
    #
    # The script's own skip condition is `command -v awk`, NOT the siblings'
    # "no ripgrep and no GNU grep". That is deliberate: those scripts depend on
    # GNU-specific regex behavior and this one does not, being POSIX awk plus a
    # plain grep -c. Copying their wider condition would skip on machines where
    # the check runs fine -- a vacuous pass in the opposite direction.
    set_tests_properties(test-sycl-device-index-policy PROPERTIES
        LABELS "sycl;policy"
        SKIP_RETURN_CODE 77
        TIMEOUT 120
    )
endif()

if (NOT WIN32)
    # Source assertion for the oneDNN SDPA scale invariant (llama.cpp-4jlv).
    # Guards the half a unit test structurally cannot: a suite is green both
    # when the execute-time GGML_ASSERT is present-and-unreached and when
    # somebody deletes it. Only the source distinguishes those two.
    #
    # Registered HERE rather than in ggml/src/ggml-sycl/CMakeLists.txt, which is
    # what the requesting task asked for so that it would run in a non-SYCL
    # configure. That placement would not have achieved it: the SYCL
    # CMakeLists is only add_subdirectory'd when the backend is enabled, so a
    # test registered anywhere in that file -- inside the XMX guard or outside
    # it -- does not exist in a non-SYCL build. tests/CMakeLists.txt is always
    # processed, so this is the location that delivers the stated intent.
    #
    # The registered script is the tests/ WRAPPER, not
    # scripts/check-sycl-fattn-onednn-scale-gate.sh directly -- same arrangement
    # as test-sycl-device-guard-symmetry-policy below, and for a sharper reason
    # here. The wrapper's controls include a case that runs the real invariant
    # 30x and requires identical statuses, which exists because the checker had
    # an INTERMITTENT defect a single-shot assertion structurally cannot see:
    # `printf '%s\n' "$VAR" | grep -q PAT` under `set -o pipefail` lets grep -q
    # exit early, printf takes SIGPIPE (141), and pipefail promotes that -- so a
    # MATCH intermittently read as a FAILURE. Registering the checker directly
    # would enforce the invariant but never exercise those controls.
    #
    # Labels match the siblings: neither "sycl" nor "policy" is in the throttled
    # sweep's -LE 'residency|mem-handle|cache' denylist.
    llama_test_cmd(
        bash
        NAME test-sycl-fattn-onednn-scale-gate
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        ARGS ${CMAKE_SOURCE_DIR}/tests/test-sycl-fattn-onednn-scale-gate.sh
    )
    set_tests_properties(test-sycl-fattn-onednn-scale-gate PROPERTIES
        LABELS "sycl;policy"
        SKIP_RETURN_CODE 77
        TIMEOUT 120
    )
endif()

if (NOT WIN32)
    # Source assertion for the device-index guard SYMMETRY invariant
    # (llama.cpp-vvcq). Sibling of test-sycl-device-index-policy, which asks a
    # different question: that one asks "is this subscript bound-checked at
    # all", this one asks "does this guard check BOTH ends". A guard that
    # checks only the lower bound compiles, links and passes every runtime
    # gate -- and is not merely cosmetic: the instance this was written for
    # answered TRUE for a device that does not exist, in GLOBAL unified-cache
    # mode, which is what AUTO resolves to on every single-GPU gate run.
    #
    # The registered script is the tests/ wrapper, not scripts/check-sycl-
    # device-guard-symmetry.sh directly. They are not duplicates: the wrapper
    # invokes the checker and asserts an EXACT exit status, because this is an
    # assertion of ABSENCE and such a check passes cleanly against an empty or
    # restructured file. "Non-zero" and "zero" are each reachable for the wrong
    # reason here, so both are pinned.
    #
    # SKIP_RETURN_CODE matches the script's own narrow `command -v awk` guard,
    # not the wider "no ripgrep and no GNU grep" condition two of its siblings
    # use -- this one is POSIX awk plus a plain grep, so the wider condition
    # would skip on machines where it runs fine, a vacuous pass in the opposite
    # direction. Labels match the siblings and are denylist-safe.
    llama_test_cmd(
        bash
        NAME test-sycl-device-guard-symmetry-policy
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        ARGS ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-device-guard-symmetry-policy.sh
    )
    set_tests_properties(test-sycl-device-guard-symmetry-policy PROPERTIES
        LABELS "sycl;policy"
        SKIP_RETURN_CODE 77
        TIMEOUT 120
    )
endif()

# sycl-canonical-gates (llama.cpp-rou3): the two prose gates in CLAUDE.md
# ("Verification Commands & Correctness Gates" -- the Mistral completion gate
# and the GPT-OSS chat gate) as ONE executable, ctest-registered runner. Not
# guarded by `if (GGML_SYCL)`: the whole point of the script's own
# CMakeCache/ldd precondition check is to catch a build where GGML_SYCL was
# silently reset OFF (CLAUDE.md: this happened 2026-07-25 and the Mistral
# gate still passed, on the CPU fallback, at ~8.38 tok/s instead of ~108). If
# registration itself were gated on the CMake-configure-time GGML_SYCL
# variable, that exact scenario would make this test vanish from `ctest -N`
# instead of failing loudly -- so it is registered unconditionally (mirroring
# the test-sycl-*-policy.sh family above) and fails closed (rc=1) at runtime
# instead.
#
# LABELS "gpu;model-loading": this test loads two real models on a real GPU
# (llama-completion, llama-cli), which the never-loop / GPU-serialization
# rules in CLAUDE.md forbid running unpinned, in a subagent, or concurrently
# with any other GPU-allocating test. Neither "gpu" nor "model-loading" is in
# CLAUDE.md's documented safe-sweep denylist (`-LE
# 'residency|mem-handle|cache'`) as written today -- that denylist must grow
# a `|gpu` term before any documented "safe" sweep can be trusted not to
# schedule this test (flagged on the tracking epic, llama.cpp-97yn).
# RESOURCE_LOCK gpu-ttm-memory: llama.cpp-ezfm's fail-closed GPU/TTM
# serialization policy had not landed a shared resource-lock name as of this
# commit; adopt ezfm's wrapper here once it exists instead of this literal.
#
# OPT-IN (llama.cpp-9hz0, 2026-09-02). The paragraph above correctly identified
# that neither of this test's labels is in CLAUDE.md's documented safe-sweep
# denylist, and deferred the fix. Deferring it shipped the hazard: a plain
# `ctest --test-dir build` or the documented "safe" form
# (-LE 'residency|mem-handle|cache' -E '^test-backend-ops$') selected this test
# and started TWO real model loads, and scripts/check-ctest-safety-net.sh --
# which exists to police exactly this -- checks only for backend-ops and then
# prints "safety net intact". CLAUDE.md measures a single model-loading test at
# 195-206 GB of 255 GB and states that two do not fit; this host has been
# OOM-killed by that class twice.
#
# So registration is now behind an explicit opt-in that defaults OFF. This
# preserves the fail-loudly-on-a-CPU-fallback-build intent above: the script's
# own CMakeCache/ldd precondition check still runs and still fails closed --
# it simply only runs for someone who asked for it, e.g.
#     cmake -B build -DLLAMA_SYCL_CANONICAL_GATES=ON && ctest -R sycl-canonical-gates
# The proper fix is to make the denylist label-driven (add gpu|model-loading to
# CLAUDE.md's documented sweep AND to check-ctest-safety-net.sh, and make that
# script fail when any GPU-labelled test survives the documented sweep). That
# edits an owner-owned file, so it is tracked on llama.cpp-9hz0 rather than done
# here. Note the stub test sycl-canonical-gates-script below is UNAFFECTED and
# still runs in every sweep -- it is CPU-only and is what keeps the gate script
# itself from being decorative.
option(LLAMA_SYCL_CANONICAL_GATES
       "Register the canonical Mistral/GPT-OSS gates as a ctest test (loads two real models on a real GPU; off by default so no sweep can schedule it)"
       OFF)
if (NOT WIN32 AND LLAMA_SYCL_CANONICAL_GATES)
    llama_test_cmd(
        bash
        NAME sycl-canonical-gates
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        ARGS ${PROJECT_SOURCE_DIR}/scripts/sycl-canonical-gates.sh
             --build-dir ${CMAKE_BINARY_DIR} --gate all --selector level_zero:1
    )
    set_tests_properties(sycl-canonical-gates PROPERTIES
        SKIP_RETURN_CODE 77
        RUN_SERIAL TRUE
        RESOURCE_LOCK gpu-ttm-memory
        ENVIRONMENT ONEAPI_DEVICE_SELECTOR=level_zero:1
        TIMEOUT 1800
        LABELS "gpu;model-loading"
    )
endif()

# sycl-canonical-gates-script: the stub-driven unit test of the gate script's
# OWN logic (tests/test-sycl-canonical-gates-script.sh) -- fail-closed
# preconditions, the no-device probe, and both pass/fail digit checks, all
# against fake llama-completion/llama-cli executables. CPU-only, no GPU
# label: this is what stops the gate script itself from being decorative,
# and it must run in every sweep that the real gpu-labelled test above is
# excluded from.
if (NOT WIN32)
    llama_test_cmd(
        bash
        NAME sycl-canonical-gates-script
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        ARGS ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-canonical-gates-script.sh
    )
    set_tests_properties(sycl-canonical-gates-script PROPERTIES
        LABELS "sycl;policy"
        TIMEOUT 120
    )
endif()

if (NOT WIN32)
    # Python source-assertion test for the decode-timeline profiling pipeline.
    # Registered because tasks that modify sycl-gptoss-decode-timeline-profile.sh
    # or parse-sycl-timeline.py need a gate the lead can run; tests/ holds dozens
    # of unregistered .py tests and this is the first one wired to ctest (see
    # llama.cpp-t5a2).
    find_package(Python3 COMPONENTS Interpreter)
    if (Python3_Interpreter_FOUND)
        set(LLAMA_PYTHON3 ${Python3_EXECUTABLE})
    else()
        # Register anyway. A configure-time bail-out would leave `ctest -N`
        # reporting "Total Tests: 0", which is byte-identical to the test never
        # having been registered at all -- the failure this whole block exists
        # to prevent. Fall back to PATH and let a genuinely missing python3
        # fail loudly at run time, as test-sycl-alloc-policy above would if
        # bash were missing.
        set(LLAMA_PYTHON3 python3)
    endif()

    # Parser census regression: establishes a freshly generated matching
    # inventory, proves --check accepts it, then line-shifts an input and proves
    # --check rejects drift without rewriting the snapshot. The plain unittest
    # script exits 77 when g++ or pinned parser packages are unavailable, so
    # dependency absence is visible as Skipped rather than a vacuous pass.
    llama_test_cmd(
        ${LLAMA_PYTHON3}
        NAME test-sycl-static-storage-audit
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        ARGS ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-static-storage-audit.py
    )
    set_tests_properties(test-sycl-static-storage-audit PROPERTIES
        LABELS "sycl;policy;audit"
        SKIP_RETURN_CODE 77
        TIMEOUT 300
    )

    # Wiring gate for the Phase 0 zone-reset escape audit (llama.cpp-iiff). The
    # audit is env-gated instrumentation whose output IS the deliverable, and its
    # characteristic failure is silent: a reset site carrying no hook contributes
    # nothing to the capture, and a capture missing a site looks exactly like a
    # site with no escapes. It also encodes the canonical §12.5 lock order --
    # the audit mutex is isolated D and stays clear of the L4/L5 allocator locks
    # only through the declaration order of a stack guard, which is invisible at
    # the call site and easy to "tidy" away.
    #
    # llama_test_cmd rather than llama_test_pytest: this is a plain script with
    # top-level checks, not pytest test_ functions (see llama_test_pytest's own
    # doc comment above). --self-test is passed so the positive controls for the
    # order- and absence-based checks run in CI, not only by hand: those checks
    # pass vacuously on any tree that never had the defect.
    #
    # Host-only -- reads sources, builds nothing, loads no model, touches no
    # device. Labels deliberately avoid residency/mem-handle/cache so the
    # documented `-LE` throttled sweep still runs it.
    llama_test_cmd(
        ${LLAMA_PYTHON3}
        NAME test-sycl-zone-reset-audit-source
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        ARGS ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-zone-reset-audit-source.py --self-test
    )
    set_tests_properties(test-sycl-zone-reset-audit-source PROPERTIES
        LABELS "sycl;policy;audit"
        SKIP_RETURN_CODE 77
        TIMEOUT 120
    )

    # Source contract for the tracked-execution abort/drain path. It existed
    # unregistered since the abort work and so had never run in CI; llama.cpp-2dgc
    # is exactly the defect it is shaped to catch (a begin/submit/release ordering
    # invariant that no GPU-free test can reach at run time), so it is wired up
    # here rather than left to be run by hand.
    #
    # llama_test_cmd rather than llama_test_pytest: a plain script with top-level
    # checks, not pytest test_ functions (see llama_test_pytest's doc comment).
    #
    # Host-only -- reads sources, builds nothing, loads no model, touches no
    # device. Labels deliberately avoid residency/mem-handle/cache so the
    # documented `-LE` throttled sweep still runs it.
    llama_test_cmd(
        ${LLAMA_PYTHON3}
        NAME test-sycl-execution-abort-source
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        ARGS ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-execution-abort-source.py
    )
    set_tests_properties(test-sycl-execution-abort-source PROPERTIES
        LABELS "sycl;execution;policy"
        TIMEOUT 120
    )

    llama_test_pytest(
        ${LLAMA_PYTHON3}
        NAME test-sycl-decode-timeline-profile-script
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-decode-timeline-profile-script.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;profiling;tdd"
        TIMEOUT 120
    )

    # Placement gate for the one-shot SYCL timeline flush. Unregistered until
    # llama.cpp-sacs, which is why the first-step-wins decode capture shipped:
    # both call sites were individually asserted here, their composition was
    # not, and nothing ran the file at all.
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-timeline-flush-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;profiling;tdd"
        TIMEOUT 120
    )

    # Coverage gate for CMAKE_CXX_LINK_DEPENDS_USE_LINKER. The variable is
    # directory-scoped, so every directory holding an icpx-linked target needs
    # its own copy; for a long time only ggml/src/ggml-sycl/ had one and nine
    # targets across four other directories were relinked on every build,
    # silently -- nothing failed and nothing warned, builds were merely never
    # incremental (llama.cpp-9ubg). This asserts directory five fails loudly.
    #
    # Registered rather than left to pytest-by-hand for the reason this file's
    # own llama_test_pytest doc comment gives: an unregistered pytest-style
    # file exits 0 for any state of the code under test, which would make a
    # gate against vacuous passes itself a vacuous pass.
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-link-depends-coverage.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;build;policy"
        TIMEOUT 120
    )

    # Coverage gate for src/CMakeLists.txt's add_library(llama ...) source
    # list: every tracked top-level src/*.cpp must appear in it (src/models/
    # is exempted -- it is covered by a `file(GLOB ...)`, checked separately
    # by the same script). llama.cpp-habh: an upstream-merge conflict
    # resolution (ff333a570) silently dropped four fork-local modules from
    # this list -- they kept compiling into stale build trees for months
    # (build-13qq, build_no_f16) while HEAD's own build silently linked
    # without them. A source present on disk but absent from the build fails
    # at LINK time in whatever *test* happens to need its symbols, which
    # names the wrong culprit -- see this script's module docstring.
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-src-cmake-coverage.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "build;policy"
        TIMEOUT 60
    )

    # Gate for the skip policy itself: a test that skips must exit 77, and
    # llama_build_and_test() must keep SKIP_RETURN_CODE 77 so ctest scores that
    # as "Skipped" rather than "Passed" (or, for a test that already exits 77
    # deliberately like test-llama-archs, rather than "Failed"). llama.cpp-nwip:
    # get_model_or_exit() exited EXIT_SUCCESS, so three model-requiring tests
    # reported green in 0.18 s having loaded nothing. Both halves of the fix are
    # one edit from gone -- tests/get-model.cpp is upstream code a rebase can
    # revert, and the property is a single line here -- so this asserts the
    # source text AND runs a binary to check the exit code. Needs no model and
    # no GPU: the skip path returns before llama is touched.
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-skip-policy.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "build;policy"
        TIMEOUT 180
    )
    # Resolved at generate time, so the target may be declared further down this
    # file. Passing the path beats having the script guess a build directory --
    # it cannot know which one ctest is running from.
    set_property(TEST test-skip-policy PROPERTY
        ENVIRONMENT "LLAMA_TEST_SKIP_BINARY=$<TARGET_FILE:test-autorelease>")

    # Gate for the single _gzip exclusion predicate in tools/ui/CMakeLists.txt,
    # which is consumed by both list(FILTER ... REGEX) and if(... MATCHES). It
    # was two hand-written patterns and they drifted, so a real directory merely
    # beginning "_gzip" was silently dropped from the watch while its files
    # stayed declared -- see llama.cpp-9ubg.
    #
    # Driven with cmake -P rather than pytest because the property under test is
    # CMake's own regex behaviour under those two operators; Python's `re` is a
    # different engine and would agree only by luck. Needs no build and no
    # dist/ fixture -- it reads the pattern out of the file and runs it against
    # a synthetic path list, including a negative control that re-runs the same
    # cases against the known-bad pattern and requires them to be rejected.
    llama_test_cmd(
        ${CMAKE_COMMAND}
        NAME test-ui-gzip-exclude
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        ARGS -DUI_CMAKE=${PROJECT_SOURCE_DIR}/tools/ui/CMakeLists.txt
             -P ${CMAKE_CURRENT_SOURCE_DIR}/test-ui-gzip-exclude.cmake
    )
    set_tests_properties(test-ui-gzip-exclude PROPERTIES
        LABELS "build;policy"
        TIMEOUT 60
    )

    # Conservation gate for parse-sycl-timeline.py's three-way gap-class split,
    # driven by the checked-in synthetic trace next to it (never live hardware).
    #
    # Unlike its sibling above this file is a plain script (def main() + a
    # __main__ guard), so it needs python3 only -- deliberately NOT registered
    # under a pytest probe, which would make a test that has no pytest
    # dependency silently vanish on a machine that merely lacks pytest.
    llama_test_cmd(
        ${LLAMA_PYTHON3}
        NAME test-sycl-timeline-gap-class-conservation
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        ARGS ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-timeline-gap-class-conservation.py
    )
    set_tests_properties(test-sycl-timeline-gap-class-conservation PROPERTIES
        LABELS "sycl;profiling;tdd"
        TIMEOUT 120
    )

    # Conservation + non-vacuity gate for parse-sycl-gap-causes.py's split of
    # runtime_idle into its four causes. The sibling gate above proves the three
    # gap *classes* balance; it cannot see whether runtime_idle -- the
    # classifier's else branch -- is signal or residual, which is the number
    # phase-2 scoping is drawn from. Same plain-script rationale as above.
    llama_test_cmd(
        ${LLAMA_PYTHON3}
        NAME test-sycl-gap-causes
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        ARGS ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-gap-causes.py
    )
    set_tests_properties(test-sycl-gap-causes PROPERTIES
        LABELS "sycl;profiling;tdd"
        TIMEOUT 120
    )

    # Regression + arithmetic gate for parse-sycl-gap-device-split.py, the
    # device-clock A/B decomposition the Cluster B verdict rests on. Registered
    # because that script shipped with a reproducible ZeroDivisionError: its
    # aggregate percentage divided by a gap total that is 0.0 whenever no gap
    # between consecutive device events is ever recorded -- a single device
    # event, or a pair where the second is fully contained in the first. Neither
    # input is malformed, and the adjacent "no device events at all" case was
    # already guarded, so this was the same failure one step later. The zero-gap
    # cases here are that regression test.
    #
    # It also pins the numbers, not just the exit code: the A/B split and the
    # equal-node_idx floor-calibration case are asserted against hand-computed
    # totals, because the floor is what makes the measured 30.42 us meaningful.
    # Same plain-script rationale as its two siblings above.
    llama_test_cmd(
        ${LLAMA_PYTHON3}
        NAME test-sycl-gap-device-split
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        ARGS ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-gap-device-split.py
    )
    set_tests_properties(test-sycl-gap-device-split PROPERTIES
        LABELS "sycl;profiling;tdd"
        TIMEOUT 120
    )

    # ---------------------------------------------------------------------
    # RESTORED COVERAGE, batch 1/N (llama.cpp-0igs): 37 pytest-style gates for
    # the scripts/ profiling toolchain that were in-tree and UNREGISTERED.
    #
    # Unregistered is not "not run" here, it is WORSE: every one of these files
    # is pytest-style (module-level test_* functions, no __main__ guard), so
    # `python3 <file>` defines the tests, calls none, and exits 0 for ANY state
    # of the code under test.  See this file's llama_test_pytest doc comment.
    #
    # Each was RUN before being registered, and each was MUTATION-CHECKED: a
    # shadow repo root (every path symlinked back except one writable copy of
    # the target) with the target gutted, confirming the gate goes red.  Six
    # were checked this way -- timeline-parser, kernel-profiler-source-binbcast,
    # vtune-source-line-checker, zebin-line-table-parser, staged-ledger-merger,
    # moe-gateup-bundle4-fullgroup -- all six detected the mutation, and a
    # no-op mutation was run as a negative control to prove the harness was not
    # simply reporting failure unconditionally.  Registering a test that cannot
    # fail is worse than leaving it unregistered: it manufactures a green ctest
    # line and the authority that carries.
    #
    # These import the target out of scripts/ and drive it with synthetic input,
    # so they are behavioural, not textual, and do not go red on reformatting.
    #
    # LABELS: `sycl;python;script-gate`.  None of the three matches the
    # throttled sweep's -LE 'residency|mem-handle|cache' denylist, so the sweep
    # actually runs them -- which is the entire point.  They allocate nothing,
    # touch no device and need no oneAPI, so they are safe at any parallelism.
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-ablation-delta-parser.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-down-variant-profile-script.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-dpas-capabilities.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-full-attribution-profile-script.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-gptoss-moe-gateup-work-reduction-harness.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-gptoss-moe-multirhs-gateup-harness.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-gtpin-bbl-pc-count-extractor.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-iga-pc-disasm-parser.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-iga-zebin-extractor.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-intel-pc-sampling-capability-script.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-kernel-profile-bandwidth.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-kernel-profile-coverage.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-kernel-profile-parser.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-layer-ledger-parser.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-logit-oracle.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-pc-sample-source-line-resolver.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-profile-ledger-parser.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-pti-instcount-extractor.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-pti-l0-parser.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-source-attribution-parser.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-source-line-debug-matrix-script.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-staged-attribution-profile-script.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-staged-ledger-merger.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-stage-manifest-parser.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-timeline-parser.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-ur-stderr-converter.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-ur-trace-parser.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-vtune-asm-parser.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-vtune-exports-parser.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-vtune-l0-host-task-converter.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-vtune-source-line-checker.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-vtune-source-line-enablement-docs.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-vtune-source-line-feasibility-script.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-vtune-task-parser.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-zebin-asm-source-line-resolver.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-zebin-line-table-parser.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-zebin-line-table-source-csv.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;script-gate"
        TIMEOUT 120
    )

    # ---------------------------------------------------------------------
    # RESTORED COVERAGE, batch 2/N (llama.cpp-0igs): 26 pytest-style SOURCE
    # assertions over ggml-sycl/ and tools/, same unregistered-so-vacuous
    # situation as batch 1 above.
    #
    # These differ from batch 1 in kind and the difference matters when one
    # goes red: batch 1 drives a scripts/ module with synthetic input and
    # asserts on BEHAVIOUR, whereas these assert that specific LITERAL TEXT is
    # present in a source file.  So a red gate here means either the invariant
    # was lost OR the text was merely reworded/realigned, and the two are not
    # distinguishable from the failure alone.  Read the diff before concluding
    # a defect.
    #
    # That brittleness is not hypothetical and it is why four sibling files are
    # deliberately NOT registered here -- see the DECLINED block below.
    #
    # Registered anyway, because the alternative is worse: a source assertion
    # is the only thing that can see an invariant deleted from a file that
    # still compiles and still passes every runtime gate.
    #
    # Mutation-checked like batch 1 (shadow root, target gutted, gate must go
    # red), including a no-op negative control.
    #
    # LABELS avoid the throttled sweep's -LE 'residency|mem-handle|cache'
    # denylist.  Host-only: no device, no oneAPI, no allocation.
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-fp16-converters.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert;converter"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-e2e-profile-compute-forward-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-e2e-profile-fattn-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-fattn-packed-k-lifecycle-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert;lifecycle"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-onednn-graph-allocator-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert;onednn;allocator"
        TIMEOUT 120
    )
    # llama.cpp-o3a0: keeps src/llama-model.cpp's replicated oneDNN SDPA
    # head-dim eligibility rule (it cannot include the SYCL-only
    # fattn-onednn.cpp to share the real predicate) in sync with the rule it
    # mirrors -- same env var, same parse, same D thresholds.
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-onednn-graph-floor-eligibility-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert;onednn;eligibility"
        TIMEOUT 120
    )
    # llama.cpp-3aos: pins the shape of the SYCL KV planner's per-layer
    # sizing fix -- n_seq_max actually consumed by the SWA cell formula, the
    # three-kind (FULL/SWA/SHARED) switch with SHARED charged 0, and the
    # runtime transaction body threading its caller's n_seq_max into
    # next_kv_info before the KV budget is recomputed.
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-kv-layer-sizing-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert;kv;sizing"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-direct-mem-op-extents.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert;mem-handle;mutation"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-nonfa-attn-scratch-guard-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert"
        TIMEOUT 120
    )
    # llama.cpp-o65k: dispatch-tuning JSON becomes opt-in via
    # GGML_SYCL_DISPATCH_TUNING_JSON (no more unconditional /tmp default and
    # per-start WARN).
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-dispatch-tuning-optin-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-supports-op-foreign-buffer-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert;placement"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-mmid-admission-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert;placement"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-supports-op-indexed-moe-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert;placement;moe"
        TIMEOUT 120
    )
    # llama.cpp-zoly / llama.cpp-nkfc: the MoE MMVQ-ID layout advertisement
    # (ggml_sycl_select_moe_mmvq_layout, ggml_sycl_select_moe_expert_cache_layout)
    # and refusal (ggml_sycl_mul_mat_id_vec_q) sites must be roster-derived
    # (moe-mmvq-tables.hpp), not a hand-maintained {Q4_0, Q8_0} type list.
    # Content-greps the whole SYCL backend at run time rather than trusting a
    # frozen list of sites -- see the module docstring for the deliberately
    # unfixed common.hpp exception.
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-moe-mmvq-layout-roster-agreement-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert;moe;hostonly"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-fattn-packed-k-live-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert;lifecycle"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-secondary-stage-source-lifetime.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert;lifecycle"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-e2e-profile-memory-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-kernel-bench-profiling-debug-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-kernel-profiler-context-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-kernel-profiler-source-attention-ops.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-kernel-profiler-source-binbcast.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-kernel-profiler-source-copy.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-kernel-profiler-source-fattn.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-kernel-profiler-source-gemm.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-kernel-profiler-source-mmvq.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-kernel-profiler-source-norm.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-kernel-profiler-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-kernel-profiler-source-row-mem-ops.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-kernel-profiler-source-unified.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-moe-gateup-bundle4-fullgroup-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-moe-gateup-bundle4-layout-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-moe-gateup-bundle4-route-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-moe-gateup-optimization-path-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-moe-gateup-work-reduction-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-moe-resolved-batch-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert;moe;retained-route;mutation"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-canonical-aos-expert-handles-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert;moe;canonical-handle;mutation"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-buffer-base-alignment-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert;alignment;mutation"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-graph-weight-layout-fallback-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert;graph;mul-mat;mutation"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-fa-graph-allow-default-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert;graph;fa;mutation"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-fa-decode-kernel-allowlist-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert;graph;fa;mutation"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-pool-legacy-release-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert;graph;mutation"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-rmsnorm-fusion-route-decline-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert;mulmat;fusion;mutation"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-universal-provenance-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert;provenance;lifecycle;mutation"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-moe-refusal-reason-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert;moe;diagnostics;mutation"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-moe-prestage-block-guard-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert;moe;prestage;mutation"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-mmid-publish-parity-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert;moe;publication;mutation"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-private-route-timeout-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert;moe;production-route;timeout"
        TIMEOUT 30
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-moe-profile-parser.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-moe-rolecol-dpas-feasibility-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-mxfp4-source-line-probe-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-op-support-predicates.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-source-line-probe-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-timeline-cache-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-timeline-compute-forward-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-timeline-wait-graph-source.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert"
        TIMEOUT 120
    )
    llama_test_pytest(
        ${LLAMA_PYTHON3}
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-mmvq-event-duration-guard.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        LABEL "sycl;python;source-assert"
        TIMEOUT 120
    )

    # DECLINED, with reasons (llama.cpp-0igs).  Sibling .py gates in this
    # directory that are deliberately left unregistered.  Registering them would
    # put permanently-red lines in ctest for reasons that are NOT code defects,
    # and a suite with known-red lines in it stops being read.
    #
    #   test-sycl-end-to-end-profiling-docs.py
    #   test-sycl-mxfp4-tg-speedup-docs.py
    #       DELETED (llama.cpp-zxlp).  Both asserted on files that HAVE NEVER
    #       EXISTED in this repository: `research.md` at the root, and
    #       `activation/sycl-mxfp4-tg-speedup-final-review-20260707.md`.
    #       `git log --all --` on either path returned nothing, so they were not
    #       deleted artifacts, they were never committed (and no renamed
    #       in-tree equivalent was found).  Unpassable as written, so removed
    #       rather than registered; the deletion commit carries the reasoning.
    #
    #   test-sycl-moe-multirhs-gateup-source.py
    #       Asserts the literal "multi_rhs_gateup = false".  The field IS
    #       present at HEAD -- ggml-sycl-bench.hpp reads
    #       `bool  multi_rhs_gateup         = false;`.  The gate is red purely
    #       because of VERTICAL ALIGNMENT, which CLAUDE.md mandates as a house
    #       convention.  This is the brittleness the batch-2 header warns about,
    #       caught in the wild: a whitespace-insensitive assertion fixes it.
    #
    #   test-sycl-staged-profiling-docs.py
    #       Asserts "Exact source-line attribution"; docs/backend/SYCL.md now
    #       reads "Exact sampled source-line attribution" (ca3bbca3e, "align
    #       source-line closure statuses" -- an intentional rewording), and the
    #       strings moved out of the section the test slices.
    #
    # All four were RUN, not guessed: 62 of the 66 unregistered pytest-style
    # files in tests/ passed, and these were the 4 that did not.
endif()

if (GGML_SYCL AND NOT GGML_BACKEND_DL)
    llama_build(test-sycl-kernel-profiler.cpp)
    target_sources(test-sycl-kernel-profiler PRIVATE
        ${CMAKE_SOURCE_DIR}/ggml/src/ggml-sycl/sycl-kernel-profiler.cpp)
    target_include_directories(test-sycl-kernel-profiler PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src)
    target_compile_definitions(test-sycl-kernel-profiler PRIVATE GGML_SYCL_PRIVATE_TESTING=1)
    # The seam-bearing profiler TU is compiled directly; ggml-sycl supplies
    # only its production timeline/runtime dependencies.
    target_link_libraries(test-sycl-kernel-profiler PRIVATE ggml-sycl)
    if (TARGET IntelSYCL::SYCL_CXX)
        target_link_libraries(test-sycl-kernel-profiler PRIVATE IntelSYCL::SYCL_CXX)
    else()
        target_include_directories(test-sycl-kernel-profiler PRIVATE ${SYCL_INCLUDE_DIR})
        target_compile_options(test-sycl-kernel-profiler PRIVATE "-fsycl")
        target_link_options(test-sycl-kernel-profiler PRIVATE "-fsycl")
    endif()
    llama_test(test-sycl-kernel-profiler LABEL "main")

    # Stress gate for unified_cache::unpin_on_event -- the weight-cache lease
    # release that binbcast's post-op event gates. Registered 2026-07-29; the
    # file had been in-tree and UNREGISTERED, so it was never compiled by this
    # build at all and `ctest -R unpin-event` reported "No tests were found",
    # which reads identically to a passing selection. (An `add_test` for it does
    # exist in the stale `build-tests-sycl/` tree from an older configure --
    # do not mistake that for coverage here.)
    #
    # Labelled `cache;mem-handle` deliberately: it allocates GPU buffer objects,
    # so it must be EXCLUDED from the throttled full-suite run
    # (`ctest -LE 'residency|mem-handle|cache'`, see CLAUDE.md) whose parallelism
    # has OOM-killed this host. An explicit `-R` still selects it.
    llama_build(test-unified-cache-unpin-event.cpp)
    target_include_directories(test-unified-cache-unpin-event PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src)
    target_link_libraries(test-unified-cache-unpin-event PRIVATE ggml-sycl)
    if (TARGET IntelSYCL::SYCL_CXX)
        target_link_libraries(test-unified-cache-unpin-event PRIVATE IntelSYCL::SYCL_CXX)
    else()
        target_include_directories(test-unified-cache-unpin-event PRIVATE ${SYCL_INCLUDE_DIR})
        target_compile_options(test-unified-cache-unpin-event PRIVATE "-fsycl")
        target_link_options(test-unified-cache-unpin-event PRIVATE "-fsycl")
    endif()
    #
    # ENVIRONMENT pins the card, and it is load-bearing because the test's OWN
    # default is silently ineffective. main() opens with
    #     if (!getenv("ONEAPI_DEVICE_SELECTOR")) setenv(..., "level_zero:0", 1);
    # but by then the SYCL runtime has already enumerated devices, so the setenv
    # changes nothing. Measured 2026-07-30: run with the selector unset, the test
    # prints BOTH GPUs (B70 at level_zero:gpu:0, B50 at gpu:1) and selects logical
    # device 0 -- the Arc Pro B70. So an unqualified `ctest -R unpin-event` lands on
    # the BENCHMARK/measurement card and can perturb a concurrent capture. It does
    # NOT abort, and it does not honour its own level_zero:0 intent either; it just
    # takes whatever device 0 happens to be.
    #
    # Pinned to level_zero:1 (Arc Pro B50, the validation card) so a bare ctest run
    # is both self-consistent and off the measurement card. Device numbering is
    # machine-specific -- see CLAUDE.md's selector table before changing this; the
    # `device=N` in SYCL logs is the post-filter in-process index, not the physical
    # card.
    #
    # SEPARATE PREREQUISITE, do not conflate the two: oneAPI must be sourced
    # (`source /opt/intel/oneapi/setvars.sh --force`) or the runtime finds no
    # platform at all and the test aborts with "no devices found on any platform"
    # at dpct/helper.hpp. That is a missing runtime environment, not a selector
    # problem, and ENVIRONMENT here cannot fix it.
    llama_test(test-unified-cache-unpin-event LABEL "main" ARGS --mode=compare --iters=50)
    set_tests_properties(test-unified-cache-unpin-event PROPERTIES
        LABELS "sycl;cache;mem-handle"
        ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1"
        TIMEOUT 600
    )

    # Gate for llama.cpp-0oxf: the oneDNN Graph-scratch DIRECT-allocation
    # path must bound its outstanding bytes (checking, then waiting on, a
    # size-bucketed reuse pool and a byte cap rather than piling DIRECT
    # buffers up without limit) and must never hand oneDNN a null scratch
    # pointer on exhaustion -- it must log and abort instead. See the file
    # header comment for the full mechanism (the reuse pool, its per-size
    # depth bound, and the byte cap) and why the abort itself is exercised
    # through an always-compiled test hook rather than a fork()-based
    # death-check (SYCL + fork is a documented hang hazard on this host,
    # CLAUDE.md's SYCL Device Selection section).
    #
    # $<TARGET_PROPERTY:ggml-sycl,COMPILE_DEFINITIONS> mirrors
    # test-sycl-fattn-onednn-gates's own pattern below: this test calls
    # onednn_graph_scratch_* symbols that only exist in the linked ggml-sycl
    # library when it was itself built with GGML_SYCL_DNNL=1, so the test's
    # own `#if GGML_SYCL_DNNL` compile-time skip must see the SAME value the
    # library was actually built with, not an independent guess.
    #
    # Labelled `cache;mem-handle` like test-unified-cache-unpin-event above --
    # it allocates real GPU buffer objects, so the throttled full-suite sweep
    # (`ctest -LE 'residency|mem-handle|cache'`) must exclude it.
    #
    # ENVIRONMENT pins the B50 (validation card), same reasoning as every
    # sibling SYCL gate registered in this file: a bare `ctest` cannot then
    # perturb a concurrent measurement on the B70. Device numbering is
    # machine-specific -- see CLAUDE.md's selector table before changing this.
    llama_build(test-sycl-onednn-graph-scratch-direct.cpp)
    target_include_directories(test-sycl-onednn-graph-scratch-direct PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src)
    target_link_libraries(test-sycl-onednn-graph-scratch-direct PRIVATE ggml-sycl)
    target_compile_definitions(test-sycl-onednn-graph-scratch-direct PRIVATE
        $<TARGET_PROPERTY:ggml-sycl,COMPILE_DEFINITIONS>)
    if (TARGET IntelSYCL::SYCL_CXX)
        target_link_libraries(test-sycl-onednn-graph-scratch-direct PRIVATE IntelSYCL::SYCL_CXX)
    else()
        target_include_directories(test-sycl-onednn-graph-scratch-direct PRIVATE ${SYCL_INCLUDE_DIR})
        target_compile_options(test-sycl-onednn-graph-scratch-direct PRIVATE "-fsycl")
        target_link_options(test-sycl-onednn-graph-scratch-direct PRIVATE "-fsycl")
    endif()
    # LABEL not passed: llama_test() defaults it to "main" already, and the
    # set_tests_properties() call right below immediately replaces LABELS
    # with the real set anyway, so passing it here would be dead.
    llama_test(test-sycl-onednn-graph-scratch-direct)
    # LD_LIBRARY_PATH addition: mirrors every sibling SYCL gate's ENVIRONMENT
    # in this file -- a direct (non-ctest) invocation without oneAPI's redist
    # lib on LD_LIBRARY_PATH prints SKIP with exit 0 (CLAUDE.md), so ctest
    # supplies it here rather than relying on the launching shell to have
    # sourced setvars.sh. GGML_SYCL_ONEDNN_GRAPH_TEST_HOOKS=1 is what lets
    # this specific binary call ggml_sycl_test_onednn_graph_scratch_force_direct_alloc_fail()/
    # _suppress_abort() at all -- see their gated setters in unified-cache.cpp;
    # no other registration in this file needs it, since no other test binary
    # calls those two symbols.
    set_tests_properties(test-sycl-onednn-graph-scratch-direct PROPERTIES
        LABELS "sycl;cache;mem-handle;onednn"
        ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1;LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH};GGML_SYCL_ONEDNN_GRAPH_TEST_HOOKS=1"
        SKIP_RETURN_CODE 77
        TIMEOUT 300
    )

    # Host-only companion to the gate above: pure-function coverage of
    # onednn_graph_scratch_zone_floor_bytes()'s shape-derived floor formula
    # (llama.cpp-0oxf), exercised through its exported test wrapper
    # (ggml_sycl_test_onednn_graph_scratch_zone_floor_bytes() -- the real
    # function has internal linkage). No device needed; it still links
    # ggml-sycl and needs -fsycl to compile the TU that declares the wrapper,
    # but performs no queue or device work, so it is NOT given the
    # `cache;mem-handle` labels above and DOES run in the throttled
    # full-suite sweep.
    #
    # Registered TWICE against the same binary (mirrors
    # test-unified-cache-unpin-event's --mode= argument pattern) because the
    # floor function's env-var read is memoized once per process -- the
    # "default" and "override" cases cannot both be exercised in one
    # invocation. `default` must run WITHOUT GGML_SYCL_ONEDNN_GRAPH_ZONE_MB
    # set; ctest inherits the launching shell's environment, so this also
    # relies on the two invocations being separate processes (true for any
    # ctest-registered test).
    llama_build(test-sycl-onednn-graph-floor.cpp)
    target_include_directories(test-sycl-onednn-graph-floor PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src)
    target_link_libraries(test-sycl-onednn-graph-floor PRIVATE ggml-sycl)
    target_compile_definitions(test-sycl-onednn-graph-floor PRIVATE
        $<TARGET_PROPERTY:ggml-sycl,COMPILE_DEFINITIONS>)
    if (TARGET IntelSYCL::SYCL_CXX)
        target_link_libraries(test-sycl-onednn-graph-floor PRIVATE IntelSYCL::SYCL_CXX)
    else()
        target_include_directories(test-sycl-onednn-graph-floor PRIVATE ${SYCL_INCLUDE_DIR})
        target_compile_options(test-sycl-onednn-graph-floor PRIVATE "-fsycl")
        target_link_options(test-sycl-onednn-graph-floor PRIVATE "-fsycl")
    endif()
    # LABEL not passed here either -- same reasoning as above.
    llama_test(test-sycl-onednn-graph-floor ARGS --mode=default)
    # ENVIRONMENT additions:
    #   - LD_LIBRARY_PATH: mirrors every sibling SYCL gate in this file, same
    #     reasoning as test-sycl-onednn-graph-scratch-direct above -- this
    #     binary links ggml-sycl and IntelSYCL::SYCL_CXX even though it
    #     touches no device, so a direct invocation without oneAPI's redist
    #     lib on the path is still the "prints SKIP, exits 0" trap CLAUDE.md
    #     documents.
    #   - GGML_SYCL_ONEDNN_GRAPH_ZONE_MB= (empty, i.e. explicitly cleared):
    #     onednn_graph_scratch_zone_floor_bytes() treats an empty value the
    #     same as unset (unified-cache.cpp: `env && env[0] != '\0'`), so this
    #     is what actually forces --mode=default to run the FORMULA rather
    #     than silently inheriting whatever value the invoking shell happens
    #     to have set for this var -- without it, all seven rows in
    #     test_default_formula() fail if the launching environment carries an
    #     override.
    set_tests_properties(test-sycl-onednn-graph-floor PROPERTIES
        LABELS "sycl;onednn"
        ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1;LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH};GGML_SYCL_ONEDNN_GRAPH_ZONE_MB="
        SKIP_RETURN_CODE 77
        TIMEOUT 60
    )
    # Registered via llama_test()'s NAME/ARGS override (not a hand-rolled
    # add_test()) so it goes through the same COMMAND-construction path as
    # every other test in this file, keeping its ENVIRONMENT properties.
    llama_test(test-sycl-onednn-graph-floor NAME test-sycl-onednn-graph-floor-override ARGS --mode=override)
    set_tests_properties(test-sycl-onednn-graph-floor-override PROPERTIES
        LABELS "sycl;onednn"
        ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1;LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH};GGML_SYCL_ONEDNN_GRAPH_ZONE_MB=99"
        SKIP_RETURN_CODE 77
        TIMEOUT 60
    )

    # Decisive-experiment probe for llama.cpp-c6ah's own premise (see the
    # file's own header comment): does the bare
    # evt.get_info<command_execution_status>() query actually block on a
    # profiling-enabled queue, for both a host_task-produced event and a
    # device-kernel-produced one. Pure raw SYCL -- does not touch
    # onednn_graph_scratch_*, unified_cache, or any other ggml-sycl symbol,
    # so unlike the two registrations above it needs neither
    # target_link_libraries(... ggml-sycl) nor the matching
    # COMPILE_DEFINITIONS propagation.
    #
    # Labelled `sycl` only (no `cache`/`mem-handle`) since it allocates one
    # small device buffer, not the pool/arena machinery those labels exist
    # to flag for the throttled full-suite sweep.
    llama_build(test-sycl-event-status-blocking-probe.cpp)
    if (TARGET IntelSYCL::SYCL_CXX)
        target_link_libraries(test-sycl-event-status-blocking-probe PRIVATE IntelSYCL::SYCL_CXX)
    else()
        target_include_directories(test-sycl-event-status-blocking-probe PRIVATE ${SYCL_INCLUDE_DIR})
        target_compile_options(test-sycl-event-status-blocking-probe PRIVATE "-fsycl")
        target_link_options(test-sycl-event-status-blocking-probe PRIVATE "-fsycl")
    endif()
    llama_test(test-sycl-event-status-blocking-probe)
    # ENVIRONMENT: same LD_LIBRARY_PATH/selector reasoning as every sibling
    # SYCL gate above. GGML_TEST_SPIN_ITERATIONS is deliberately NOT set
    # here -- the binary's own built-in default is what ctest exercises;
    # override it by hand for a bare (non-ctest) re-calibration run.
    set_tests_properties(test-sycl-event-status-blocking-probe PROPERTIES
        LABELS "sycl"
        ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1;LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
        SKIP_RETURN_CODE 77
        TIMEOUT 60
    )

    # Host-only gate for llama.cpp-oyfl: the non-flash-attention batched
    # mul_mat scratch demand formula (unified_cache_nonfa_attn_scratch_demand_bytes()
    # and its inverse) -- the sibling of the oneDNN Graph-scratch floor gate
    # immediately above, but NOT gated behind GGML_SYCL_DNNL (the path it
    # sizes for is the native SYCL/oneMath batched-mul_mat route, not a
    # oneDNN one). Same multi-registration shape for the same reason: the
    # formula's env-var read is memoized once per process, so "default",
    # "override", and "override-zero" cannot share one invocation.
    llama_build(test-sycl-nonfa-attn-scratch-demand.cpp)
    target_include_directories(test-sycl-nonfa-attn-scratch-demand PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src)
    target_link_libraries(test-sycl-nonfa-attn-scratch-demand PRIVATE ggml-sycl)
    target_compile_definitions(test-sycl-nonfa-attn-scratch-demand PRIVATE
        $<TARGET_PROPERTY:ggml-sycl,COMPILE_DEFINITIONS>)
    if (TARGET IntelSYCL::SYCL_CXX)
        target_link_libraries(test-sycl-nonfa-attn-scratch-demand PRIVATE IntelSYCL::SYCL_CXX)
    else()
        target_include_directories(test-sycl-nonfa-attn-scratch-demand PRIVATE ${SYCL_INCLUDE_DIR})
        target_compile_options(test-sycl-nonfa-attn-scratch-demand PRIVATE "-fsycl")
        target_link_options(test-sycl-nonfa-attn-scratch-demand PRIVATE "-fsycl")
    endif()
    llama_test(test-sycl-nonfa-attn-scratch-demand ARGS --mode=default)
    # ENVIRONMENT: LD_LIBRARY_PATH per every sibling SYCL gate (see the
    # onednn-graph-floor registration's comment above for why);
    # GGML_SYCL_NONFA_ATTN_SCRATCH_MB= (empty, explicitly cleared) forces
    # --mode=default to exercise the FORMULA rather than silently inheriting
    # whatever the launching shell happens to have set.
    set_tests_properties(test-sycl-nonfa-attn-scratch-demand PROPERTIES
        LABELS "sycl"
        ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1;LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH};GGML_SYCL_NONFA_ATTN_SCRATCH_MB="
        SKIP_RETURN_CODE 77
        TIMEOUT 60
    )
    llama_test(test-sycl-nonfa-attn-scratch-demand NAME test-sycl-nonfa-attn-scratch-demand-override ARGS --mode=override)
    set_tests_properties(test-sycl-nonfa-attn-scratch-demand-override PROPERTIES
        LABELS "sycl"
        ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1;LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH};GGML_SYCL_NONFA_ATTN_SCRATCH_MB=77"
        SKIP_RETURN_CODE 77
        TIMEOUT 60
    )
    # Third registration: GGML_SYCL_NONFA_ATTN_SCRATCH_MB=0 is a DISTINCT
    # override value from both "unset" (--mode=default above) and "77"
    # (--mode=override above) -- env_mb_override()'s memoized read means all
    # three must run as separate processes, same reasoning as the two
    # registrations above.
    llama_test(test-sycl-nonfa-attn-scratch-demand NAME test-sycl-nonfa-attn-scratch-demand-override-zero ARGS --mode=override-zero)
    set_tests_properties(test-sycl-nonfa-attn-scratch-demand-override-zero PROPERTIES
        LABELS "sycl"
        ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1;LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH};GGML_SYCL_NONFA_ATTN_SCRATCH_MB=0"
        SKIP_RETURN_CODE 77
        TIMEOUT 60
    )

    # Runtime gate for the SYCL settings self-report (llama.cpp-5bd8). Drives a
    # real registry -> ggml_backend_dev_init -> ggml_check_sycl() init and
    # asserts the report reaches the log callback. A source assertion would not
    # do: the reporting code was present and correct for months while its output
    # went nowhere, because upstream 67b2b7f2f pushed backend GGML_LOG_INFO above
    # the default log threshold.
    #
    # Labels deliberately exclude `cache`, `mem-handle` and `residency` -- the
    # throttled full-suite sweep in CLAUDE.md runs
    # `ctest -LE 'residency|mem-handle|cache'`, and a gate carrying one of those
    # would never run there.
    #
    # ENVIRONMENT pins the B50 (the validation card) rather than the B70, so a
    # bare `ctest` cannot perturb a concurrent measurement -- same reasoning as
    # test-unified-cache-unpin-event above, and the same caveat applies: device
    # numbering is machine-specific, see CLAUDE.md's selector table. oneAPI must
    # be sourced or the runtime finds no platform and the test aborts rather
    # than skipping.
    llama_build(test-sycl-env-report.cpp)
    target_link_libraries(test-sycl-env-report PRIVATE ggml-sycl)
    llama_test(test-sycl-env-report LABEL "main")
    set_tests_properties(test-sycl-env-report PROPERTIES
        LABELS "sycl;env"
        ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1"
        SKIP_RETURN_CODE 77
        TIMEOUT 300
    )

    # Gate for llama.cpp-fz2u: host_zone_settle() / zone_settle() (named
    # host_zone_reset() / zone_reset() before llama.cpp-37ba's rename) must
    # REFUSE to reset a zone that still holds a live registered allocation, instead of
    # aborting (9a0670712) or purging the registry and resetting anyway (the
    # behaviour before it, which recycled addresses still owned by live handles).
    #
    # The test asserts BOTH halves, and the second is the one that is easy to
    # lose: that a later reset SUCCEEDS once the owner releases. The refusal is
    # only safe because unified_free() drops the registry record for every zone,
    # including the reset-only SCRATCH/STAGING host zones that take no per-block
    # free -- if that erase ever becomes conditional, the zone would never reset
    # again for the process lifetime, silently. The refusal half alone still
    # passes in that broken world.
    #
    # Labels deliberately exclude `cache`, `mem-handle` and `residency`. Those
    # three are the CLAUDE.md throttled-sweep denylist
    # (`ctest -LE 'residency|mem-handle|cache'`), and the denylist is about
    # MEMORY-EXHAUSTION HAZARD, not subject matter -- carrying one would keep
    # this gate out of the only sweep that routinely runs it. Measured before
    # choosing: `free -g` available moved 0 GB across a run (see the fz2u task),
    # so it does not belong in the hazard family. Re-measure before adding a
    # label here; do not copy test-unified-cache-unpin-event's, which are
    # correct for it because it allocates GPU buffer objects in a loop.
    #
    # ENVIRONMENT pins the B50 (validation card) so a bare ctest cannot perturb a
    # measurement on the B70 -- same reasoning as the two gates above, and the
    # same caveat: device numbering is machine-specific, see CLAUDE.md's selector
    # table. oneAPI must be sourced or the SYCL runtime finds no platform.
    llama_build(test-sycl-zone-reset-live-refusal.cpp)
    target_include_directories(test-sycl-zone-reset-live-refusal PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src)
    target_link_libraries(test-sycl-zone-reset-live-refusal PRIVATE ggml-sycl)
    if (TARGET IntelSYCL::SYCL_CXX)
        target_link_libraries(test-sycl-zone-reset-live-refusal PRIVATE IntelSYCL::SYCL_CXX)
    else()
        target_include_directories(test-sycl-zone-reset-live-refusal PRIVATE ${SYCL_INCLUDE_DIR})
        target_compile_options(test-sycl-zone-reset-live-refusal PRIVATE "-fsycl")
        target_link_options(test-sycl-zone-reset-live-refusal PRIVATE "-fsycl")
    endif()
    llama_test(test-sycl-zone-reset-live-refusal LABEL "main")
    set_tests_properties(test-sycl-zone-reset-live-refusal PROPERTIES
        LABELS "sycl;lease;bugfix"
        ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1"
        TIMEOUT 300
    )

    # test-sycl-model-lifecycle-hooks: the weight-ownership hooks must actually
    # be INVOKED on a real llama_model load and teardown (llama.cpp-viga).
    #
    # test-sycl-reset-model-weight-lease-preserve proves what those hooks DECIDE,
    # by driving unified_cache directly -- which is exactly what makes it cheap
    # and what makes it blind to whether the backend ever calls them. This is the
    # other half. It is the only affordable test that walks the real path:
    # test-thread-safety also does, but peaks at 156-180 GB of Shmem and its exit
    # status is already consumed by two unrelated crashes (llama.cpp-oze0,
    # llama.cpp-09ep), so a verdict read from it is absorbed noise.
    #
    # Links ggml-sycl for the probe symbol but is NOT compiled with -fsycl: the
    # probe header deliberately pulls in no SYCL headers, so this stays a plain
    # C++ TU and adds no icpx device link to the build.
    #
    # SKIP_RETURN_CODE 77 is load-bearing, not boilerplate. This test's whole
    # value is proving a call happened; the failure mode to design against is a
    # run that reports green having loaded nothing. No device, no model file, or
    # no -m argument all exit 77, so ctest says "skipped" and a bare shell run
    # gets a non-zero status. The device check in particular is not cosmetic:
    # with no SYCL device there is no unified cache, yet the backend entry points
    # still run and would still advance the counters.
    #
    # Labels deliberately exclude `residency|mem-handle|cache` -- the CLAUDE.md
    # throttled-sweep denylist, which is about MEMORY-EXHAUSTION HAZARD, not
    # subject matter. This test DOES load a model onto a GPU, which is the
    # never-loop family's defining property, so the exclusion needs a number
    # rather than an assumption.
    #
    # EVIDENCE, with its date and conditions, because a threshold outlives the
    # hardware it was measured on and this file has been wrong that way before.
    # Measured on THIS binary, under the selector this registration pins:
    #
    #   2026-08-01, ONEAPI_DEVICE_SELECTOR=level_zero:1 (the pin below):
    #     Shmem   2.3 GB before load -> 2.3 GB after load -> 2.3 GB after free
    #     peak RSS 564.5 MB
    #   Shmem delta: ZERO. Not "small" -- nothing measurable.
    #
    # An earlier draft of this comment justified the label by inference from
    # llama.cpp-403s (~2.2 GB, measured under level_zero:0,1 on a different
    # binary). That figure was for a DUAL-GPU load and did not describe this
    # test; the number above replaces it. Retained only for the direction it
    # establishes: before llama.cpp-403s the same load cost ~127.8 GB, which
    # WOULD have belonged in the denylist -- so this label is a consequence of
    # that commit, not an intrinsic property of loading a model here.
    #
    # The test prints its own Shmem/MemAvailable before load, after load and
    # after free, so every run re-checks this rather than trusting the comment.
    # If a run shows a real delta, move it into the denylist and record the new
    # number, date and selector here. Note this holds for ONE load: the test
    # loads once and never loops, which is the never-loop rule, not an
    # exception to it.
    llama_build(test-sycl-model-lifecycle-hooks.cpp)
    target_include_directories(test-sycl-model-lifecycle-hooks PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src)
    target_link_libraries(test-sycl-model-lifecycle-hooks PRIVATE ggml-sycl)
    llama_test(test-sycl-model-lifecycle-hooks LABEL "main" ARGS -m "${MODEL_DEST}")
    set_tests_properties(test-sycl-model-lifecycle-hooks PROPERTIES
        LABELS "sycl;lease;lifecycle;bugfix"
        FIXTURES_REQUIRED test-download-model
        ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1"
        SKIP_RETURN_CODE 77
        TIMEOUT 300
    )

    # test-layout-cache: RESTORED per the llama.cpp-b8uv adjudication (see the
    # file's own header comment). Registered at 3c8f296fd:928-940, wiped by the
    # registration loss 0igs tracks; not a mock (all 5 entry points reach
    # production in ggml-sycl.cpp) and not rotted (its 10-arg
    # ensure_cached_alloc call matches the current signature).
    # Two controls the 3c8f296fd block lacked, both mandatory:
    #  - main() setenv()s ONEAPI_DEVICE_SELECTOR to level_zero:0 (the B70
    #    benchmark card) when unset; the setenv only fires when nothing is set,
    #    so pinning level_zero:1 here defeats it. Same trap as
    #    test-unified-cache-bugs.
    #  - It loads models (OOM-hazard family): RUN_SERIAL, never looped, and
    #    LABELS avoid the residency/mem-handle/cache denylist substrings per
    #    rule 2 so the -j 1 throttled sweep can see it. Exits 77 on no-device.
    add_executable(test-layout-cache test-layout-cache.cpp)
    target_link_libraries(test-layout-cache PRIVATE ggml-sycl-private-fixtures)
    target_include_directories(test-layout-cache PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_definitions(test-layout-cache PRIVATE
        GGML_SYCL_WARP_SIZE=32
        GGML_SYCL_PRIVATE_TESTING=1)
    target_compile_options(test-layout-cache PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-layout-cache PRIVATE "-fsycl")
    add_test(NAME test-layout-cache COMMAND $<TARGET_FILE:test-layout-cache>)
    set_tests_properties(test-layout-cache PROPERTIES
        LABELS "sycl;layout;model-load;gpu-serial"
        ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1;LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
        RUN_SERIAL TRUE
        SKIP_RETURN_CODE 77
        TIMEOUT 300
        # DISABLED REMOVED -- all five of the first restored run's failures are
        # adjudicated, every one a stale oracle, every fix test-side. No product
        # change was needed or made by any of them.
        #
        #   test_mul_mat_layout_choice          llama.cpp-0tkh  the override is a
        #     selection hint; CPU routing is decided upstream and never consults
        #     it. Confirmed empirically -- host/auto == host/coalesced on both
        #     axes in artifacts/b8uv/0tkh-run.txt.
        #   test_layout_selection               llama.cpp-no1t  layout_policy no
        #     longer returns XMX_TILED on ANY branch; 3c8f296fd moved that to the
        #     planner (common.hpp:1070) and the MoE branch now yields SOA, or
        #     COALESCED without XMX (common.hpp:1359-1374).
        #   test_aos_drop                       llama.cpp-778b  \  one root cause:
        #   test_device_weight_layout_cache     llama.cpp-doyp  /  38c02b099 and
        #     b8afde8e9 turned ggml_sycl_get_weight_layout_ptr from a materializer
        #     into a lookup over S1-PRELOAD-staged weights. The AoS-drop invariant
        #     survives in the cache's own keying (unified-cache.cpp:14252-14298),
        #     so the fixtures now drive ensure_cached_alloc, the sanctioned
        #     test-only entry seam, and assert the identity fast path
        #     (ggml-sycl.cpp:29403-29422) on the device buft.
        #   test_model_load_host_buffer_is_pinned  llama.cpp-6qgl  renamed: both
        #     historical reasons to avoid pinning were retired by 73376bdfd and
        #     03dd78bab, and must_host_pinned is now unconditional at
        #     ggml-sycl.cpp:35391.
        #
        # This registration has never yet had a green run -- the previous two were
        # 1/7 and 3/7. If it reddens, read the PASS/FAIL roll-up main() prints
        # rather than the first stderr line: every sub-test runs to completion and
        # each failure names its own mechanism. Do NOT re-DISABLE and do NOT weaken
        # an oracle; take the named mechanism back to its ticket above.
        # It loads models, so run it ONCE. Never in a loop.
    )

    # llama.cpp-6cgq: Q8_0 SOA-vs-COALESCED MMVQ numerics. The ticket tried
    # two kernel-side fixes for a measured small-N bandwidth deficit (an
    # occupancy-geometry change, then a D-plane SLM-staging change); both
    # were refuted by GPU measurement and neither shipped (see the test
    # file's own header comment for the numbers) -- what remains is this
    # numerics coverage, kept because it is useful independent of which
    # layout wins any future default-flip decision: it compares the SOA and
    # COALESCED Q8_0 MMVQ kernels against each other AND against a CPU
    # reference that correctly quantizes activations to Q8_1 before the dot
    # product (a real bug in an earlier reference was caught and fixed
    # during this ticket). Needs the ggml-sycl-private-fixtures
    # layout-override hook (same mechanism test-layout-cache above and
    # test-q8-0-layout-cache-path-mmvq use) to force each weight tensor's
    # materialized layout deterministically -- a plain device-buffer upload
    # does not exercise SOA/COALESCED materialization at all, only the
    # model-load path does.
    add_executable(test-sycl-mmvq-q8-0-soa-numerics test-sycl-mmvq-q8-0-soa-numerics.cpp)
    target_link_libraries(test-sycl-mmvq-q8-0-soa-numerics PRIVATE ggml-sycl-private-fixtures)
    target_include_directories(test-sycl-mmvq-q8-0-soa-numerics PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_definitions(test-sycl-mmvq-q8-0-soa-numerics PRIVATE
        GGML_SYCL_WARP_SIZE=32
        GGML_SYCL_PRIVATE_TESTING=1)
    target_compile_options(test-sycl-mmvq-q8-0-soa-numerics PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-sycl-mmvq-q8-0-soa-numerics PRIVATE "-fsycl")
    add_test(NAME test-sycl-mmvq-q8-0-soa-numerics COMMAND $<TARGET_FILE:test-sycl-mmvq-q8-0-soa-numerics>)
    set_tests_properties(test-sycl-mmvq-q8-0-soa-numerics PROPERTIES
        LABELS "sycl;mmvq;quant;soa;gpu-serial"
        ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1;LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
        RUN_SERIAL TRUE
        SKIP_RETURN_CODE 77
        TIMEOUT 300
    )
endif()

# test-sycl-fattn-onednn-gates: oneDNN SDPA must reject unsafe FA layouts.
#
# RESTORED, not newly written.  This registration existed at
# 3c8f296fd:tests/CMakeLists.txt:990, built, ran and PASSED -- the proof is
# committed in this repo at artifacts/kkxtv7-5/ctest-materialization.log
# ("Test #100: test-sycl-fattn-onednn-gates ... Passed 0.11 sec").  At the next
# commit touching this file, d3dce4e0a, the file is upstream's 262-line version
# and its add_test count has gone 147 -> 3.  141 registrations disappeared at
# once; three have been rebuilt since, one at a time, by tasks that each
# happened to notice one missing test.  The test sources survived in the tree
# with nothing pointing at them, so passing tests became fossils: committed
# logs of runs that can no longer happen.
#
# HOW they were removed is NOT established, and this comment deliberately does
# not guess.  d3dce4e0a is a single-parent commit, so it was not a merge
# resolution there; a rebase onto upstream, a merge resolved in upstream's
# favour earlier in the chain, and a manual overwrite are all still open.  The
# counts are facts, the mechanism is not, and this backlog is largely made of
# plausible stories that hardened into repeated fact.
#
# That is a fourth failure mode distinct from the guard (llama.cpp-5exz),
# from never-registered sources (llama.cpp-0igs), and from a denylisted label.
# Those three are static states of the tree that a grep can find.  This one is
# a loss event, invisible in the current tree, detectable only by noticing that
# evidence of a passing test has no registration behind it.
#
# The block below is the ORIGINAL, verbatim apart from the added LABELS: it is
# known to have built and run, which a fresh guess would not be.  Note it links
# ggml + ggml-cpu and inherits ggml-sycl's COMPILE_DEFINITIONS through a
# TARGET_PROPERTY generator expression rather than linking the SYCL library --
# the test is host-only, allocates nothing and touches no device, so it needs
# no device pinning and no OOM-hazard label.
#
# LABELS are new.  The original had none, which meant it inherited only the
# default and no label denylist could reach it -- survivable here, but "only
# main" means "nobody classified this", not "this is safe" (CLAUDE.md).  None
# of these matches the throttled sweep's -LE 'residency|mem-handle|cache', so
# the sweep runs it, which is the entire point of restoring it.
if(GGML_SYCL)
    add_executable(test-sycl-fattn-onednn-gates test-sycl-fattn-onednn-gates.cpp)
    target_link_libraries(test-sycl-fattn-onednn-gates PRIVATE ggml)
    if (NOT GGML_BACKEND_DL)
        target_link_libraries(test-sycl-fattn-onednn-gates PRIVATE ggml-cpu)
    endif()
    target_include_directories(test-sycl-fattn-onednn-gates PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_definitions(test-sycl-fattn-onednn-gates PRIVATE
        $<TARGET_PROPERTY:ggml-sycl,COMPILE_DEFINITIONS>)
    # Absolute repo root for the source-reading gates: when scripts/sycl-build.sh
    # runs ccache with base_dir (opt-in, GGML_SYCL_CCACHE_BASE_DIR=1) __FILE__ is
    # relative to the build directory, and this binary is install()ed and run
    # from anywhere (llama.cpp-vuy0).
    target_compile_definitions(test-sycl-fattn-onednn-gates PRIVATE
        LLAMA_CPP_SOURCE_ROOT="${PROJECT_SOURCE_DIR}")
    target_compile_options(test-sycl-fattn-onednn-gates PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-sycl-fattn-onednn-gates PRIVATE "-fsycl")
    install(TARGETS test-sycl-fattn-onednn-gates RUNTIME)
    add_test(NAME test-sycl-fattn-onednn-gates COMMAND $<TARGET_FILE:test-sycl-fattn-onednn-gates>)
    # SKIP_RETURN_CODE is new (llama.cpp-ay8c), and is the one departure from
    # "verbatim apart from LABELS" above.  Built without GGML_SYCL_DNNL the
    # binary's main() is the compile-guard stub, which used to print and return
    # 0 -- a green pass for a run that evaluated no gate.  It now returns 77 and
    # ctest reports the run as skipped.
    set_tests_properties(test-sycl-fattn-onednn-gates PROPERTIES
        LABELS "sycl;fattn;onednn;gates"
        SKIP_RETURN_CODE 77
        TIMEOUT 120
    )
endif()

# test-sycl-zero-alloc-check: llama.cpp-dcx6 -- the zero-alloc-check baseline
# tracker must re-baseline on phase transitions, not on every call. Host-only:
# it drives the pure state machine directly with plain (phase, bytes) pairs,
# no device, no allocation, no oneDNN.
if(GGML_SYCL)
    add_executable(test-sycl-zero-alloc-check test-sycl-zero-alloc-check.cpp)
    target_link_libraries(test-sycl-zero-alloc-check PRIVATE ggml)
    if (NOT GGML_BACKEND_DL)
        target_link_libraries(test-sycl-zero-alloc-check PRIVATE ggml-cpu)
    endif()
    target_include_directories(test-sycl-zero-alloc-check PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_definitions(test-sycl-zero-alloc-check PRIVATE
        $<TARGET_PROPERTY:ggml-sycl,COMPILE_DEFINITIONS>)
    target_compile_options(test-sycl-zero-alloc-check PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-sycl-zero-alloc-check PRIVATE "-fsycl")
    install(TARGETS test-sycl-zero-alloc-check RUNTIME)
    add_test(NAME test-sycl-zero-alloc-check COMMAND $<TARGET_FILE:test-sycl-zero-alloc-check>)
    set_tests_properties(test-sycl-zero-alloc-check PROPERTIES
        LABELS "sycl;zero-alloc;gates"
        SKIP_RETURN_CODE 77
        TIMEOUT 60
    )
endif()

# test-sycl-getrows-q8-0-index-math: llama.cpp-go37 -- the 2-elements/thread
# Q8_0 AoS GET_ROWS kernel's index math (block/quant-index split, row-bounds
# guard). Host-only: calls the pure functions the device kernel calls
# directly, no device, no allocation.
if(GGML_SYCL)
    add_executable(test-sycl-getrows-q8-0-index-math test-sycl-getrows-q8-0-index-math.cpp)
    target_link_libraries(test-sycl-getrows-q8-0-index-math PRIVATE ggml)
    if (NOT GGML_BACKEND_DL)
        target_link_libraries(test-sycl-getrows-q8-0-index-math PRIVATE ggml-cpu)
    endif()
    target_include_directories(test-sycl-getrows-q8-0-index-math PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_definitions(test-sycl-getrows-q8-0-index-math PRIVATE
        $<TARGET_PROPERTY:ggml-sycl,COMPILE_DEFINITIONS>)
    target_compile_options(test-sycl-getrows-q8-0-index-math PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-sycl-getrows-q8-0-index-math PRIVATE "-fsycl")
    install(TARGETS test-sycl-getrows-q8-0-index-math RUNTIME)
    add_test(NAME test-sycl-getrows-q8-0-index-math COMMAND $<TARGET_FILE:test-sycl-getrows-q8-0-index-math>)
    set_tests_properties(test-sycl-getrows-q8-0-index-math PROPERTIES
        LABELS "sycl;getrows;gates"
        SKIP_RETURN_CODE 77
        TIMEOUT 60
    )
endif()

if (NOT LLAMA_SANITIZE_ADDRESS AND NOT GGML_SCHED_NO_REALLOC)
  # TODO: repair known memory leaks
  llama_build_and_test(test-opt.cpp)
endif()
llama_build_and_test(test-backend-ops.cpp)

llama_build_and_test(test-model-load-cancel.cpp LABEL "model")
llama_build_and_test(test-autorelease.cpp       LABEL "model")
llama_build_and_test(test-backend-sampler.cpp   LABEL "model")

# Test for state restore with fragmented KV cache
# Requires a model, uses same args pattern as test-thread-safety
llama_build_and_test(test-state-restore-fragmented.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-state-restore-fragmented PROPERTIES FIXTURES_REQUIRED test-download-model)

# Test state save/load functionality
llama_build_and_test(test-save-load-state.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-save-load-state PROPERTIES FIXTURES_REQUIRED test-download-model)

if (APPLE)
    llama_build(test-rset-release.cpp)
endif()

if (NOT GGML_BACKEND_DL)
    # these tests use the backends directly and cannot be built with dynamic loading
    llama_build_and_test(test-barrier.cpp)
    llama_build_and_test(test-quantize-fns.cpp)
    llama_build_and_test(test-quantize-perf.cpp)
    llama_build_and_test(test-rope.cpp)
    llama_build_and_test(test-col2im-1d.cpp)

    # RESTORED (llama.cpp-0igs): 3c8f296fd:258 had this immediately above
    # test-rope, inside this same NOT GGML_BACKEND_DL guard -- it uses the
    # backend vec_dot directly, so the guard is load-bearing, not decoration.
    # Put back in its original position rather than appended somewhere
    # convenient, so the reason it is guarded stays legible.
    llama_build_and_test(test-q4-0-q8-0-vec-dot-regression.cpp)
    # NB: llama_build_and_test's LABEL is a oneValueArgs, so a ";"-joined list
    # does NOT survive it -- ARGN flattens it and every element after the first
    # falls into UNPARSED_ARGUMENTS, which this helper appends to TEST_SOURCES.
    # `LABEL "main;quant"` therefore fails configure with `Cannot find source
    # file: quant`. Only llama_test_pytest takes a multi-valued LABEL. Set
    # multi-labels through the test property instead.
    set_tests_properties(test-q4-0-q8-0-vec-dot-regression PROPERTIES LABELS "main;quant")
    # The 3c8f296fd block was a bare llama_build_and_test with no include dirs
    # and it no longer compiles: the source includes ggml-cpu/quants.h, which
    # opens with #include "ggml-common.h", and ggml/src is not on this target's
    # include path today. The test source is BYTE-IDENTICAL to 3c8f296fd, so
    # this is the include path regressing underneath an unchanged test, not the
    # test drifting -- which is exactly why a restored block has to be built
    # and not just pasted back. Same remedy as test-alloc below.
    target_include_directories(test-q4-0-q8-0-vec-dot-regression PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
endif()

# libmtmd
set(LLAMA_TEST_NAME test-mtmd-c-api)
llama_build_and_test(test-mtmd-c-api.c)
target_link_libraries(${LLAMA_TEST_NAME} PRIVATE mtmd)
unset(LLAMA_TEST_NAME)

llama_build_and_test(test-mtmd-impl.cpp)
target_link_libraries(test-mtmd-impl PRIVATE mtmd)

# GGUF model data fetcher library for tests that need real model metadata
# Only compile when cpp-httplib has SSL support (CPPHTTPLIB_OPENSSL_SUPPORT)
if (TARGET cpp-httplib)
    get_target_property(_cpp_httplib_defs cpp-httplib INTERFACE_COMPILE_DEFINITIONS)
    if (_cpp_httplib_defs MATCHES "CPPHTTPLIB_OPENSSL_SUPPORT")
        add_library(gguf-model-data STATIC gguf-model-data.cpp)
        target_link_libraries(gguf-model-data PRIVATE llama-common cpp-httplib)
        target_include_directories(gguf-model-data PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

        add_executable(test-gguf-model-data test-gguf-model-data.cpp)
        target_link_libraries(test-gguf-model-data PRIVATE gguf-model-data llama-common)
        llama_test(test-gguf-model-data LABEL "model")

        # test-quant-type-selection requires gguf-model-data for remote model metadata
        llama_build_and_test(test-quant-type-selection.cpp LABEL "model")
        target_link_libraries(test-quant-type-selection PRIVATE gguf-model-data)
        target_compile_definitions(test-quant-type-selection PRIVATE
            SNAPSHOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/snapshots")
    endif()
endif()

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)

if (NOT LLAMA_USE_SYSTEM_GGML)
    # Needs non-public ggml-impl.h
    llama_build_and_test(test-gguf.cpp)

    # Needs non-public ggml{,-backend}-impl.h
    llama_build_and_test(test-alloc.cpp)
endif()

# -------------------------------------------------------------------------
# RESTORED COVERAGE, batch 3/N (llama.cpp-0igs): the host-only C++ tests.
#
# These five had known-good llama_build_and_test blocks in
# 3c8f296fd:tests/CMakeLists.txt (lines 196, 197, 274, 513, 516) and are the
# subset that needs NO SYCL device link -- they build and run in a plain host
# configure, so restoring them costs a normal compile rather than an ~15-minute
# ocloc link. Verified built and run before landing.
#
# Each was checked for a real failure path first, not just for compiling.
# That check is the point of this ticket and it disqualified others: this
# build is Release, CMAKE_CXX_FLAGS_RELEASE is `-O3 -DNDEBUG`, so a test whose
# only failure signal is bare assert() compiles to a program whose sole exit
# path is `return 0`. These five signal through `return failed > 0 ? 1 : 0`,
# `return 1`, or GGML_ASSERT, all of which survive NDEBUG.
#
# LABELS are new -- the originals had none, and "only main" means "nobody
# classified this", not "this is safe" (CLAUDE.md). None matches the throttled
# sweep's -LE 'residency|mem-handle|cache' denylist, so the sweep runs them.
llama_build_and_test(test-tensor-data.cpp)
set_tests_properties(test-tensor-data       PROPERTIES LABELS "main;tensor")
llama_build_and_test(test-q8-0-soa.cpp)
set_tests_properties(test-q8-0-soa          PROPERTIES LABELS "main;quant;soa")
llama_build_and_test(test-mmq-soa-q4-0.cpp)
set_tests_properties(test-mmq-soa-q4-0      PROPERTIES LABELS "main;quant;soa")

# Self-contained: builds its own temporary GGUF with mkstemps and parses it
# back, so it needs no model fixture and no network.
llama_build_and_test(test-sycl-model-shapes.cpp)
set_tests_properties(test-sycl-model-shapes PROPERTIES LABELS "main;sycl;shapes")

# ARGS and the ENVIRONMENT property are restored verbatim from 3c8f296fd:197.
# The golden files are the assertion -- without them the test has nothing to
# compare against -- and both are present in tests/golden/.
#
# llama.cpp-habh fixed the underlying gap this comment used to describe:
# llama-tensor-class.cpp, llama-kv-block.cpp, llama-pp-scheduler.cpp and
# llama-moe-profile.cpp are back in src/CMakeLists.txt's add_library(llama ...)
# list (dropped by ff333a570's upstream-merge conflict resolution on the first
# three; never added at all for the fourth). test-tensor-class now links the
# classifier from libllama like every other test, instead of separately
# compiling src/llama-tensor-class.cpp into the test binary -- keeping that
# workaround after the fix landed would risk two definitions of
# llama_tensor_classify et al. reaching the same executable.
# tests/test-src-cmake-coverage.py guards against this regressing again.
llama_build_and_test(test-tensor-class.cpp
    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/golden/place2-mistral7b.txt
         ${CMAKE_CURRENT_SOURCE_DIR}/golden/place2-gpt-oss-20b.txt)
target_include_directories(test-tensor-class PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_tests_properties(test-tensor-class      PROPERTIES LABELS "main;tensor")
set_property(TEST test-tensor-class PROPERTY
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}:$ENV{LD_LIBRARY_PATH}")

# 178 assertions over the paged-KV block manager, block table, defragmentation
# planner and prefix cache. Two things had to be true before this could be
# registered honestly, and neither was assumed:
#
#   1. The header had to be reachable. It was not -- src/llama-kv-block.cpp was
#      absent from src/CMakeLists.txt's add_library(llama ...) list, so the
#      module compiled into nothing and "'llama-kv-block.h' file not found" was
#      the symptom, not test rot. Fixed in 1fb188c15 (llama.cpp-habh); libllama
#      now exports the llama_kv_block_manager symbols this test links against.
#
#   2. The assertions had to be able to FAIL, and then had to actually PASS.
#      Under -O3 -DNDEBUG all 178 were compiled out; the test was green with
#      every one of them forced false. tests/test-kv-block.cpp carries the
#      `#undef NDEBUG` that restores the signal, and the mutation matrix in its
#      header comment (A/B/C/D, run out-of-tree) is the evidence for both
#      halves. Registering it before that would have added a permanently-green
#      line claiming 178 assertions and carrying none.
#
# No SYCL device link, no model fixture, no network -- a plain host compile.
#
# LABELS deliberately avoid "cache" despite the prefix-caching coverage: the
# throttled full-suite sweep is -LE 'residency|mem-handle|cache', so that word
# would silently exclude this host-only test from every sweep that runs it.
llama_build_and_test(test-kv-block.cpp)
target_include_directories(test-kv-block PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_tests_properties(test-kv-block          PROPERTIES LABELS "main;kv;paged")

# DECLINED in this batch, with reasons (llama.cpp-0igs):
#
#   test-fattn-determinism.cpp (20 asserts), test-fattn-mask.cpp (19)
#       VACUOUS under this build's own flags. Measured, not inferred: compiled
#       with -O3 -DNDEBUG and then REBUILT with every assert condition forced
#       false, both still exit 0. The same all-false mutants built WITHOUT
#       -DNDEBUG abort with 134, which is the control proving the mutation is
#       real and that NDEBUG is what erases the test. Neither file carries
#       `#undef NDEBUG`. Registering them would add two permanently-green lines
#       that cannot fail. One line each (`#undef NDEBUG` after the includes)
#       makes them real; that is a tests/*.cpp change and is tracked separately.
#       Six more restorable sources were in the same state. The largest,
#       test-kv-block.cpp at 178 asserts, has since been fixed and registered
#       above (llama.cpp-4hvq) -- its assertions pass once live, which is what
#       separates it from these two. The rest are still pending.
#
#   test-sycl-model-repro.cpp
#       On a missing model it prints "SKIP:" and returns 0, not 77, so
#       registered bare it is a guaranteed green on any machine without
#       LLAMA_SYCL_TEST_MODEL set. It needs either an ENVIRONMENT pinning a
#       real model -- which makes it a GPU test, not a host one, and moves it
#       out of this batch -- or a source change to return 77 so ctest can tell
#       "skipped" from "passed".

llama_build(test-export-graph-ops.cpp)
if (TARGET gguf-model-data)
    target_link_libraries(test-export-graph-ops PRIVATE gguf-model-data)
    target_compile_definitions(test-export-graph-ops PRIVATE LLAMA_HF_FETCH)
endif()

# Regression gate for llama.cpp-1lrh: a SYCL backend *registration* query
# (ggml_backend_reg_by_name("SYCL")) must report zero devices rather than
# abort the process when platform enumeration finds none. Forces the
# zero-device condition on every host via ONEAPI_DEVICE_SELECTOR=level_zero:99
# (an out-of-range device index -- verified empirically to resolve to "zero
# devices enumerated", not a selector parse error) so the property is
# exercised unconditionally rather than only on device-less runners. See
# test-sycl-registration-no-device.cpp for the full rationale.
if (GGML_SYCL)
    llama_build(test-sycl-registration-no-device.cpp)
    target_link_libraries(test-sycl-registration-no-device PRIVATE ggml-sycl)
    llama_test(test-sycl-registration-no-device LABEL "main")
    set_tests_properties(test-sycl-registration-no-device PROPERTIES
        LABELS "sycl;env"
        ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:99"
        SKIP_RETURN_CODE 77
        TIMEOUT 120
    )
endif()

# llama.cpp-qq19: divergence guard for the single owner-resolution authority
# ggml_sycl_resolve_tensor_owner() -- asserts the "resolve owner from tensor
# extra->model_id via registry, else fall back to the identity plan" block
# exists exactly once and every former inline site (and any future one) calls
# it rather than copying it. Host-only -- reads ggml-sycl.cpp as text, builds
# nothing, touches no device. Mirrors test-sycl-device-uuid-source's
# llama_test_pytest registration shape.
if (Python3_Interpreter_FOUND)
    llama_test_pytest(${Python3_EXECUTABLE}
        NAME test-sycl-resolve-tensor-owner-source
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-resolve-tensor-owner-source.py
        LABEL "sycl;source;host"
        TIMEOUT 30)
endif()

# llama.cpp-nz1k (prefill L2b phase 1): the Q8_0 SOA scale-plane index mapping
# (ggml/src/ggml-sycl/q8-scale-plane.hpp) that the oneDNN WoQ-int8 arm's scale
# staging kernel calls, checked against a scalar reference with a positive
# control. Pure C++ -- no SYCL, no device, no ggml link -- so it runs in every
# sweep at any parallelism. Labelled like the other host-only quant tests.
llama_build_and_test(test-sycl-q8-scale-plane-index.cpp)
target_include_directories(test-sycl-q8-scale-plane-index PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src)
set_tests_properties(test-sycl-q8-scale-plane-index PROPERTIES LABELS "main;sycl;quant;soa;host")

# llama.cpp-pktr: the Q8_0 dense-weight coalesced/SOA tile-alignment predicate
# (ggml/src/ggml-sycl/q8-dense-layout-rule.hpp) that
# ggml_sycl_adjust_layout_for_tensor and planner_default_device_layout both
# call to pick SOA for a tile-misaligned dense Q8_0 row (e.g. gemma4-E4B's
# K=2560 head) and COALESCED otherwise (Mistral K=4096/14336, gemma4
# K=10240/2048). Checked against an independent reference over an exhaustive
# sweep. Pure C++ -- no SYCL, no device, no ggml link -- so it runs in every
# sweep at any parallelism. Labelled like the other host-only quant tests.
llama_build_and_test(test-sycl-q8-dense-layout-rule.cpp)
target_include_directories(test-sycl-q8-dense-layout-rule PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src)
set_tests_properties(test-sycl-q8-dense-layout-rule PROPERTIES LABELS "main;sycl;quant;soa;host")

# llama.cpp-vtfs (option C task 1, owner ruling 2026-08-23): host CPU
# reference oracle for MXFP4 GEMMs on the two device-materialized expert
# weight layouts -- SOA (down role, ggml/src/ggml-sycl/quants.hpp:190-211)
# and XMX_TILED (gate/up role, ggml/src/ggml-sycl/moe-xmx-fused.hpp:118-153,
# inverse map at ggml/src/ggml-sycl/convert.cpp:2111-2146). Every later
# stored-layout MXFP4 kernel (tasks G4-G8) is scored against this file's
# reference_gemm() + max_rel_violations(..., 0.0258 -- the proven 2-D WOQ
# GEMM arm's hardware verdict, llama.cpp-sr83 fix cycle 2, documented at
# ggml/src/ggml-sycl/gemm.hpp, not tests/test-sycl-mxfp4-woq-gemm-bench.cpp).
# Pure C++ -- touches no device and enumerates no SYCL device (in a
# GGML_SYCL=ON build it links libggml-sycl transitively through
# llama-common, same as any other llama_build_and_test target, but never
# calls into it), so it runs in every sweep at any parallelism.
llama_build_and_test(test-sycl-mxfp4-stored-layout-gemm-oracle.cpp)
target_include_directories(test-sycl-mxfp4-stored-layout-gemm-oracle PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src)
set_tests_properties(test-sycl-mxfp4-stored-layout-gemm-oracle PROPERTIES LABELS "main;sycl;mxfp4;host")

# llama.cpp-nz1k: source gate -- the WoQ-Q8 arm (ONEDNN_SOA) is reachable only
# with GGML_SYCL_ONEDNN_WOQ_Q8=1 and a Q8_0 weight the planner materialized SOA
# (dense projections: GGML_SYCL_Q8_DENSE_LAYOUT=soa), applies the codebase's
# standard eligibility predicate before its single lookup, and both env vars
# default OFF (ruling 9 / nz1k c-c4jp). Host-only, reads the sources as text. Same registration shape as
# test-sycl-resolve-tensor-owner-source.
if (Python3_Interpreter_FOUND)
    llama_test_pytest(${Python3_EXECUTABLE}
        NAME test-sycl-onednn-woq-q8-source
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-onednn-woq-q8-source.py
        LABEL "sycl;source;host"
        TIMEOUT 30)
endif()

# llama.cpp-je3b: source gate -- "may oneDNN PP run here" has one source,
# ggml_sycl_onednn_pp_candidate. The MXFP4 direct oneDNN arms ask it (with the
# MXFP4_DIRECT route, which moves only the batch floor) instead of admitting
# themselves with M >= 2 && executable_on_device. Host-only, reads the sources
# as text. Same registration shape as test-sycl-onednn-woq-q8-source.
if (Python3_Interpreter_FOUND)
    llama_test_pytest(${Python3_EXECUTABLE}
        NAME test-sycl-onednn-pp-admission-source
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-onednn-pp-admission-source.py
        LABEL "sycl;source;host"
        TIMEOUT 30)
endif()

# llama.cpp-pktr: source gate -- the Q8_0 dense-weight coalesced/SOA
# tile-alignment predicate must be defined exactly once
# (q8-dense-layout-rule.hpp) and every chokepoint that resolves a dense Q8_0
# weight's layout (ggml_sycl_adjust_layout_for_tensor's runtime resolution,
# both overloads of planner_default_device_layout's planning-time resolution)
# must call it rather than re-deriving the alignment arithmetic inline.
# Host-only, reads the sources as text. Same registration shape as
# test-sycl-onednn-woq-q8-source.
if (Python3_Interpreter_FOUND)
    llama_test_pytest(${Python3_EXECUTABLE}
        NAME test-sycl-q8-dense-layout-rule-source
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-q8-dense-layout-rule-source.py
        LABEL "sycl;source;host"
        TIMEOUT 30)
endif()

# llama.cpp-dfo0 (plan task L1): source gate for the env-gated leak probe in
# ggml_backend_sycl_buffer_reset's COMPUTE early return -- probe present,
# WARN level, env-gated, zero-cost when unset. Host-only, reads the sources
# as text. Same registration shape as test-sycl-onednn-woq-q8-source.
if (Python3_Interpreter_FOUND)
    llama_test_pytest(${Python3_EXECUTABLE}
        NAME test-sycl-extra-leak-probe-source
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-extra-leak-probe-source.py
        LABEL "sycl;source;host"
        TIMEOUT 30)
endif()

# llama.cpp-5q1r (plan task S6): source gate asserting every tests/*.cpp that
# setenv()s ONEAPI_DEVICE_SELECTOR in its own main() does so through the
# shared tests/sycl-selector-fallback.hpp re-exec helper, not a no-op bare
# setenv() or an independent execv() copy -- see that file and the gate
# script's own docstring for why a plain setenv() cannot work here. Host-only,
# reads the sources as text. Same registration shape as
# test-sycl-extra-leak-probe-source above.
if (Python3_Interpreter_FOUND)
    llama_test_pytest(${Python3_EXECUTABLE}
        NAME test-sycl-selector-fallback-source
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-selector-fallback-source.py
        LABEL "sycl;source;host"
        TIMEOUT 30)
endif()

# Pure-Python source-level companion to the GPU test
# test-sycl-profiling-queue-property.cpp (llama.cpp-yke2): checks that
# create_cache_for_device()'s call to ensure_single_device_context_queue() is
# unconditional (not gated on total_gpus > 1). Host-only, reads the sources
# as text -- same registration shape as test-sycl-extra-leak-probe-source
# above. Registered here, well after find_package(Python3) above, rather than
# next to the GPU test itself (much earlier in this file, before that
# find_package call) -- Python3_Interpreter_FOUND is unset that early, so a
# registration placed there is silently never added at all (ctest -N reports
# it as if it never existed, not merely skipped).
if (Python3_Interpreter_FOUND)
    llama_test_pytest(${Python3_EXECUTABLE}
        NAME test-sycl-profiling-queue-property-source
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-profiling-queue-property-source.py
        LABEL "sycl;source;host"
        TIMEOUT 30)
endif()

# llama.cpp-pktr: test-sycl-layout-choice.cpp is already registered, in
# ggml/src/ggml-sycl/CMakeLists.txt, as ctest `sycl-layout-choice` (note the
# ctest name differs from the executable target name). Do not register it
# here too.

# llama.cpp-ibj0: host-only gate for the pure functions behind re-planning the
# PP MoE oneDNN scratch ring for the RUNTIME n_ubatch (the loader always plans
# for its own hardcoded n_ubatch=512, src/llama-model.cpp). No SYCL device is
# touched -- every function under test is a pure function of integers, same
# reasoning as test-sycl-nonfa-attn-scratch-demand.cpp. ONE arm (no env-var
# axis: unlike that sibling, nothing here reads an environment override).
if (GGML_SYCL)
    llama_build(test-sycl-ubatch-ring-plan.cpp)
    target_include_directories(test-sycl-ubatch-ring-plan PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src)
    target_link_libraries(test-sycl-ubatch-ring-plan PRIVATE ggml-sycl)
    target_compile_definitions(test-sycl-ubatch-ring-plan PRIVATE
        $<TARGET_PROPERTY:ggml-sycl,COMPILE_DEFINITIONS>)
    if (TARGET IntelSYCL::SYCL_CXX)
        target_link_libraries(test-sycl-ubatch-ring-plan PRIVATE IntelSYCL::SYCL_CXX)
    else()
        target_include_directories(test-sycl-ubatch-ring-plan PRIVATE ${SYCL_INCLUDE_DIR})
        target_compile_options(test-sycl-ubatch-ring-plan PRIVATE "-fsycl")
        target_link_options(test-sycl-ubatch-ring-plan PRIVATE "-fsycl")
    endif()
    llama_test(test-sycl-ubatch-ring-plan)
    # ENVIRONMENT: LD_LIBRARY_PATH per every sibling SYCL gate (see the
    # nonfa-attn-scratch-demand registration's comment for why -- a SYCL test
    # binary run without it prints "SKIP: no SYCL GPU devices available" and
    # exits 0, which is NOT what happens here since this test never enumerates
    # a device, but the library link still needs it to load at all).
    set_tests_properties(test-sycl-ubatch-ring-plan PROPERTIES
        LABELS "sycl"
        ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
        SKIP_RETURN_CODE 77
        TIMEOUT 60
    )
endif()

# llama.cpp-ibj0: source gate -- ggml_backend_sycl_set_runtime_context() must
# re-plan the PP MoE oneDNN scratch ring for the runtime n_ubatch (after the
# non-FA attention scratch check, before the plan is published), refuse with
# the largest-fitting -ub when it cannot, and the loader must store both
# per-row fields. Host-only, reads the sources as text. Same registration
# shape as test-sycl-resolve-tensor-owner-source.
if (Python3_Interpreter_FOUND)
    llama_test_pytest(${Python3_EXECUTABLE}
        NAME test-sycl-ubatch-ring-replan-source
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-ubatch-ring-replan-source.py
        LABEL "sycl;python;source-assert"
        TIMEOUT 120)
endif()

# llama.cpp-u1bb: source gate for how the runtime transaction grows the PP MoE
# oneDNN ring past the RUNTIME zone into the shared KV zone's post-KV headroom
# (after KV, against the one live KV capacity, charged, never spilling). The
# admission arithmetic itself is host-tested by test-pp-moe-ring-admission.
# Host-only, reads the sources as text.
if (Python3_Interpreter_FOUND)
    llama_test_pytest(${Python3_EXECUTABLE}
        NAME test-sycl-pp-moe-ring-kv-zone-source
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-pp-moe-ring-kv-zone-source.py
        LABEL "sycl;python;source-assert"
        TIMEOUT 120)
endif()

# llama.cpp-tsfl: source gate for nphx Task 2 -- (A) the two compute-buffer
# host-pinned fallback paths inside
# ggml_backend_sycl_buffer_type_alloc_buffer() must be GGML_LOG_WARN (not
# INFO) and counted in a per-device g_compute_buffer_host_fallbacks[]
# counter, reset on the runtime-context transaction's own successful
# (publishing) path; (B) the transaction and a new non-publishing probe,
# ggml_backend_sycl_probe_runtime_context_for_model(), share one static
# body up to (not including) the plan-publication CAS, with candidate
# refusals dropping to GGML_LOG_INFO in probe mode. Host-only, reads the
# sources as text. Same registration shape as
# test-sycl-ubatch-ring-replan-source above.
if (Python3_Interpreter_FOUND)
    llama_test_pytest(${Python3_EXECUTABLE}
        NAME test-sycl-compute-buffer-fallback-source
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-compute-buffer-fallback-source.py
        LABEL "sycl;python;source-assert"
        TIMEOUT 120)
endif()

# llama.cpp-xojq: source gate for nphx Task 4b -- the SYCL auto micro-batch
# selection trial (llama_context::sycl_select_auto_ubatch()) and its one
# call site in the llama_context constructor. Pins the four-condition gate
# (n_ubatch_auto, a SYCL backend, GGML_SYCL_AUTO_UBATCH, causal_attn), the
# {512, 1024, 2048, 4096} ladder and its n_batch/n_ctx/MoE-ceiling caps, the
# per-candidate probe (BUSY backoff, STALE_IDENTITY, would_demote_kv),
# publish-before-reserve ordering, the host-fallback query consulted after
# reserve, the settle step, the nine stop-reason strings (including "cached"
# for a tuning-cache hit, see below, and "compute buffers did not fit" for a
# candidate whose compute-buffer reserve throws), and exactly one [SYCL-PLAN]
# auto n_ubatch= WARN. Host-only, reads the sources as text. Same
# registration shape as test-sycl-compute-buffer-fallback-source above.
#
# llama.cpp-7n6n (nphx Task 5, wired into the same trial): the persisted
# auto n_ubatch cache -- tried BEFORE the ladder (a clean hit sets the new
# "cached" stop reason and skips every rung); a store that follows any
# ladder run, never a hit; GGML_SYCL_TUNING_CACHE=0 disabling both. This
# file's own tests cover the ordering and the new stop reason; the on-disk
# format (tuning-cache-io.hpp's UbatchCacheKey/UbatchCacheEntry) is covered
# by test-tuning-cache-io below instead, since that part is pure host I/O
# with no SYCL device involved either way.
if (Python3_Interpreter_FOUND)
    llama_test_pytest(${Python3_EXECUTABLE}
        NAME test-sycl-auto-ubatch-source
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-auto-ubatch-source.py
        LABEL "sycl;python;source-assert"
        TIMEOUT 120)
endif()

# llama.cpp-pyu4: executes the pure helpers in src/llama-auto-ubatch.h --
# llama_auto_ubatch_ladder_has_candidate(), the predicate behind the trial's
# early exit before its ladder loop, on the shapes where no rung lies in
# [fallback_ubatch, cap]; and llama_auto_ubatch_settle_needs_publish(), the
# settle step's republish decision, including a publish that threw after
# landing on some devices. Header-only and pure integers: no device, no
# model, no allocation, so it builds and runs in every configuration.
llama_build_and_test(test-auto-ubatch-ladder.cpp)

# llama.cpp-9goq: source gate over ggml/src/ggml-sycl/CMakeLists.txt itself --
# (A) every target_link_libraries(...) naming the bare `ggml-sycl` MODULE
# target must sit under a guard that guarantees GGML_BACKEND_DL is off (the
# regression: three test targets landed without this and broke
# .devops/intel.Dockerfile's GGML_BACKEND_DL=ON configure -- CMake hard-errors
# linking a MODULE_LIBRARY into another target); (B) every add_executable()
# test target must sit under a BUILD_TESTING-mentioning guard, so
# LLAMA_BUILD_TESTS=OFF actually stops them from being added, rather than
# silently ignoring it. Host-only, reads ggml/src/ggml-sycl/CMakeLists.txt as
# text -- same registration shape as test-sycl-compute-buffer-fallback-source
# above. Registered here (not in ggml/src/ggml-sycl/CMakeLists.txt) for the
# same reason test-sycl-fattn-onednn-scale-gate is: that file is only
# add_subdirectory'd when the SYCL backend is enabled, so a test registered
# inside it would not exist in a non-SYCL configure and could not catch a
# regression in the very file it is meant to guard.
if (Python3_Interpreter_FOUND)
    llama_test_pytest(${Python3_EXECUTABLE}
        NAME test-sycl-cmake-dl-link-source
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-cmake-dl-link-source.py
        LABEL "sycl;python;source-assert;cmake"
        TIMEOUT 120)
endif()

# llama.cpp-n4ee: DL=OFF-reachable equivalent of tests/test-sycl-module-
# dependencies.py's POST_BUILD audit (that audit only runs under
# -DGGML_BACKEND_DL=ON, independent of BUILD_TESTING -- the ordinary
# workstation build never exercises it, which is how the TKV-13 (B2)
# attention host-dispatch regression this ticket fixes went unnoticed for two
# weeks). Tracks #ifdef/#ifndef/#else/#endif nesting for GGML_BACKEND_DL over
# ggml-sycl.cpp as pure text and fails, naming file:line, on any of that
# audit's forbidden CPU-backend symbols found outside a region the nesting
# proves is excluded from a GGML_BACKEND_DL build. The forbidden-symbol list
# is extracted from test-sycl-module-dependencies.py's own source, never a
# second copy. Host-only, no SYCL device, no DL build -- same registration
# shape as test-sycl-cmake-dl-link-source above.
if (Python3_Interpreter_FOUND)
    llama_test_pytest(${Python3_EXECUTABLE}
        NAME test-sycl-attn-host-dl-source
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-attn-host-dl-source.py
        LABEL "sycl;python;source-assert;dl"
        TIMEOUT 120)
endif()

# llama.cpp-9goq: configure-only smoke test that reproduces
# .devops/intel.Dockerfile's server-image build stage's exact `cmake -B build`
# flags (parsed live from the Dockerfile, never a hardcoded copy) against a
# throwaway build directory -- no build step, so this is safe at any
# parallelism and needs no GPU. The fork's GitHub Actions are disabled (memory
# fork-github-actions-disabled-manually), so nothing else exercises the
# Dockerfile's GGML_BACKEND_DL=ON + GGML_CPU_ALL_VARIANTS=ON +
# LLAMA_BUILD_TESTS=OFF combination; this is the substitute.
#
# The script itself exits 77 (SKIP) when oneAPI is not sourced and icx/icpx
# are not on PATH, matching SKIP_RETURN_CODE below -- a machine with no SYCL
# toolchain reports Skipped, not Passed or Failed.
if (NOT WIN32)
    llama_test_cmd(
        bash
        NAME test-sycl-dockerfile-configure-check
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        ARGS ${PROJECT_SOURCE_DIR}/scripts/sycl-dockerfile-configure-check.sh
    )
    # Same LABEL caveat as test-sycl-prefill-scaling above (llama_test_cmd's
    # LABEL is single-valued) -- set via set_tests_properties instead.
    set_tests_properties(test-sycl-dockerfile-configure-check PROPERTIES
        LABELS "sycl;cmake;configure"
        SKIP_RETURN_CODE 77
        TIMEOUT 300
    )
endif()

# llama.cpp-7n6n: unit tests for tuning-cache-io.hpp (JSON I/O, atomic
# writes, XDG cache-dir resolution, and -- as of this task -- the "ubatch"
# entry kind wired for the persisted auto n_ubatch cache: UbatchCacheKey/
# UbatchCacheEntry round-trip, key equality per field, CACHE_VERSION bump
# rejecting a v1 file, and an unwritable directory failing save/load
# cleanly). Pure header + STL: no ggml/llama/SYCL dependency at all, so this
# was previously an unregistered file (zero production callers, per
# plan-research/tuning-cache.md) -- this task adds the registration.
#
# llama.cpp-7n6n: this file's NEW ubatch-cache tests
# each use their own unique per-pid /tmp directory, passed as an explicit
# argument -- but THREE PRE-EXISTING tests (cache_file_roundtrip,
# version_check, atomic_write) call the matmul-dispatch-tuning save_cache()/
# load_cache()/get_cache_file(), which go through get_cache_dir() and so
# fall back to the REAL $HOME/.cache/llama.cpp/sycl-tuning whenever
# XDG_CACHE_HOME is unset -- registering this binary at all made them reach
# that real directory for the first time (caught in review: an empty
# ~/.cache/llama.cpp/sycl-tuning/ left behind by a local pre-commit run).
# ENVIRONMENT pins BOTH variables so ctest can never repeat that: XDG_
# CACHE_HOME confines the pre-existing tests' get_cache_dir() fallback to a
# build-tree scratch directory, and GGML_SYCL_TUNING_CACHE_DIR is set
# alongside it for consistency with the runtime accessor of the same name
# (ubatch-tuning-cache.cpp) even though this binary's own tests do not
# currently read it (they pass an explicit cache_dir argument instead).
llama_build_and_test(test-tuning-cache-io.cpp)
set_tests_properties(test-tuning-cache-io PROPERTIES
    ENVIRONMENT "XDG_CACHE_HOME=${CMAKE_CURRENT_BINARY_DIR}/test-tuning-cache-io-home;GGML_SYCL_TUNING_CACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/test-tuning-cache-io-dir")
