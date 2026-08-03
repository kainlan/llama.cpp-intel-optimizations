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

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS} get-model.cpp)

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

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API (when building with shared libraries)
    llama_build_and_test(test-sampling.cpp)
    llama_build_and_test(test-reasoning-budget.cpp)
    llama_build_and_test(test-grammar-parser.cpp)
    llama_build_and_test(test-grammar-integration.cpp)
    llama_build_and_test(test-llama-grammar.cpp)
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

llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp)
llama_build_and_test(test-jinja.cpp)
llama_test(test-jinja NAME test-jinja-py ARGS -py LABEL python)
llama_build_and_test(test-chat-auto-parser.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
llama_build_and_test(test-chat-template.cpp)
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

llama_build_and_test(test-thread-safety.cpp ARGS -m "${MODEL_DEST}" -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2)
set_tests_properties(test-thread-safety PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-arg-parser.cpp)

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
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-kernel-profiler-source-mmvq.py
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

    # DECLINED, with reasons (llama.cpp-0igs).  Four sibling .py gates in this
    # directory are deliberately left unregistered.  Registering them would put
    # four permanently-red lines in ctest for reasons that are NOT code
    # defects, and a suite with known-red lines in it stops being read.
    #
    #   test-sycl-end-to-end-profiling-docs.py
    #   test-sycl-mxfp4-tg-speedup-docs.py
    #       Both assert on files that HAVE NEVER EXISTED in this repository:
    #       `research.md` at the root, and
    #       `activation/sycl-mxfp4-tg-speedup-final-review-20260707.md`.
    #       `git log --all --` on either path returns nothing, so these are not
    #       deleted artifacts, they were never committed.  Unpassable as
    #       written; they need rewriting or removing, not registering.
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
    # files in tests/ pass, and these are the 4 that do not.
endif()

if (GGML_SYCL)
    llama_build(test-sycl-kernel-profiler.cpp)
    target_include_directories(test-sycl-kernel-profiler PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src)
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

    # Gate for llama.cpp-fz2u: host_zone_reset() / zone_reset() must REFUSE to
    # reset a zone that still holds a live registered allocation, instead of
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
    target_link_libraries(test-sycl-fattn-onednn-gates PRIVATE ggml ggml-cpu)
    target_include_directories(test-sycl-fattn-onednn-gates PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_definitions(test-sycl-fattn-onednn-gates PRIVATE
        $<TARGET_PROPERTY:ggml-sycl,COMPILE_DEFINITIONS>)
    target_compile_options(test-sycl-fattn-onednn-gates PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-sycl-fattn-onednn-gates PRIVATE "-fsycl")
    install(TARGETS test-sycl-fattn-onednn-gates RUNTIME)
    add_test(NAME test-sycl-fattn-onednn-gates COMMAND $<TARGET_FILE:test-sycl-fattn-onednn-gates>)
    set_tests_properties(test-sycl-fattn-onednn-gates PROPERTIES
        LABELS "sycl;fattn;onednn;gates"
        TIMEOUT 120
    )
endif()

if (NOT LLAMA_SANITIZE_ADDRESS AND NOT GGML_SCHED_NO_REALLOC)
  # TODO: repair known memory leaks
  llama_build_and_test(test-opt.cpp)
endif()
llama_build_and_test(test-gguf.cpp)
llama_build_and_test(test-backend-ops.cpp)

llama_build_and_test(test-model-load-cancel.cpp LABEL "model")
llama_build_and_test(test-autorelease.cpp       LABEL "model")
llama_build_and_test(test-backend-sampler.cpp   LABEL "model")

# Test for state restore with fragmented KV cache
# Requires a model, uses same args pattern as test-thread-safety
llama_build_and_test(test-state-restore-fragmented.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-state-restore-fragmented PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-recurrent-state-rollback.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-recurrent-state-rollback PROPERTIES FIXTURES_REQUIRED test-download-model)

# Test state save/load functionality
llama_build_and_test(test-save-load-state.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-save-load-state PROPERTIES FIXTURES_REQUIRED test-download-model)

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

llama_build_and_test(test-alloc.cpp)
target_include_directories(test-alloc PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)

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
target_include_directories(test-export-graph-ops PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
if (TARGET gguf-model-data)
    target_link_libraries(test-export-graph-ops PRIVATE gguf-model-data)
    target_compile_definitions(test-export-graph-ops PRIVATE LLAMA_HF_FETCH)
endif()
