# Gate for tools/ui/CMakeLists.txt's UI_DIST_GZIP_EXCLUDE predicate.
#
# That one pattern is consumed at two sites with two different CMake operators:
#
#     list(FILTER UI_DIST_FILES EXCLUDE REGEX "${UI_DIST_GZIP_EXCLUDE}")
#     if (IS_DIRECTORY "${entry}" AND NOT entry MATCHES "${UI_DIST_GZIP_EXCLUDE}")
#
# It used to be two hand-written patterns, and they drifted: `MATCHES` is an
# unanchored substring test, so the directory guard also excluded any real
# directory merely BEGINNING "_gzip". dist/_gzipfoo/ went unwatched while its
# files stayed declared as inputs -- silently unembedded assets in one
# direction, and a hard "missing and no known rule to make it" in the other.
#
# This is a cmake -P script rather than a pytest file on purpose. The property
# under test is the behaviour of CMake's own regex engine under `list(FILTER)`
# and `if(MATCHES)`. Re-implementing that in Python's `re` would test a
# different engine and agree with the real one only by luck -- the same
# "verdict sourced from something adjacent to the work" this file guards.
#
# Run:  cmake -DUI_CMAKE=<path to tools/ui/CMakeLists.txt> -P test-ui-gzip-exclude.cmake

cmake_minimum_required(VERSION 3.18)

if (NOT DEFINED UI_CMAKE)
    get_filename_component(_here "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
    set(UI_CMAKE "${_here}/../tools/ui/CMakeLists.txt")
endif()
if (NOT EXISTS "${UI_CMAKE}")
    message(FATAL_ERROR "test-ui-gzip-exclude: no such file: ${UI_CMAKE}")
endif()

file(READ "${UI_CMAKE}" _ui_text)

# Read the pattern back OUT of the file under test. A retyped copy would only
# prove that a string agrees with itself.
if (NOT _ui_text MATCHES "set\\(UI_DIST_GZIP_EXCLUDE \"([^\"]+)\"\\)")
    message(FATAL_ERROR
        "test-ui-gzip-exclude: could not find set(UI_DIST_GZIP_EXCLUDE \"...\") "
        "in ${UI_CMAKE}. If the variable was renamed or inlined, update this "
        "test -- do not delete it.")
endif()
set(PATTERN "${CMAKE_MATCH_1}")

# Anti-vacuity, part 1: the pattern must be non-empty. An empty pattern matches
# everything under `MATCHES` and nothing useful under FILTER, and would sail
# through a case list that only ever checked "did it not crash".
if (PATTERN STREQUAL "")
    message(FATAL_ERROR "test-ui-gzip-exclude: extracted pattern is empty")
endif()

# Anti-vacuity, part 2: both consuming sites must still exist and still use the
# variable. If someone re-splits them into literals, the pattern above could
# stay perfect while the file itself regresses -- exactly the round-six defect.
string(REGEX MATCHALL "\\$\\{UI_DIST_GZIP_EXCLUDE\\}" _uses "${_ui_text}")
list(LENGTH _uses _use_count)
if (_use_count LESS 2)
    message(FATAL_ERROR
        "test-ui-gzip-exclude: expected the single pattern to be USED at both "
        "exclusion sites, found ${_use_count}. Two hand-maintained copies is "
        "how they drifted the first time.")
endif()
if (NOT _ui_text MATCHES "list\\(FILTER UI_DIST_FILES EXCLUDE REGEX \"\\$\\{UI_DIST_GZIP_EXCLUDE\\}\"\\)")
    message(FATAL_ERROR "test-ui-gzip-exclude: file-filter site no longer uses the shared pattern")
endif()
if (NOT _ui_text MATCHES "NOT entry MATCHES \"\\$\\{UI_DIST_GZIP_EXCLUDE\\}\"")
    message(FATAL_ERROR "test-ui-gzip-exclude: directory-guard site no longer uses the shared pattern")
endif()

# path | kind | expected verdict
set(CASES
    "/a/dist/_gzip/x.js|FILE|EXCLUDE"       # generated gzip output -> never an input
    "/a/dist/_gzip/deep/y.js|FILE|EXCLUDE"
    "/a/dist/_gzipfoo/x.js|FILE|KEEP"       # real asset dir that merely starts "_gzip"
    "/a/dist/_app/x.js|FILE|KEEP"
    "/a/dist/index.html|FILE|KEEP"
    "/a/dist/_gzip|DIR|EXCLUDE"             # the gzip dir itself -> not watched
    "/a/dist/_gzipfoo|DIR|KEEP"             # real dir -> MUST be watched
    "/a/dist/_app|DIR|KEEP"
    "/a/dist/_app/immutable|DIR|KEEP"
)

# Drive BOTH sites with the same inputs, using the same operators the real file
# uses. `pat` is a parameter so the negative control below can re-run the whole
# case set against the pattern that was actually broken.
function(run_cases pat label out_failures)
    set(failures 0)
    foreach(case ${CASES})
        string(REPLACE "|" ";" parts "${case}")
        list(GET parts 0 path)
        list(GET parts 1 kind)
        list(GET parts 2 want)

        if (kind STREQUAL "FILE")
            # mirrors: list(FILTER UI_DIST_FILES EXCLUDE REGEX "${pat}")
            set(one "${path}")
            list(FILTER one EXCLUDE REGEX "${pat}")
            if (one STREQUAL "")
                set(got "EXCLUDE")
            else()
                set(got "KEEP")
            endif()
        else()
            # mirrors: NOT entry MATCHES "${pat}"
            if (path MATCHES "${pat}")
                set(got "EXCLUDE")
            else()
                set(got "KEEP")
            endif()
        endif()

        if (NOT got STREQUAL want)
            message(STATUS "  [${label}] ${kind} ${path} -> ${got}, want ${want}")
            math(EXPR failures "${failures}+1")
        endif()
    endforeach()
    set(${out_failures} ${failures} PARENT_SCOPE)
endfunction()

list(LENGTH CASES _case_count)
if (_case_count LESS 4)
    message(FATAL_ERROR "test-ui-gzip-exclude: case list is too small to be meaningful")
endif()

# The real assertion.
run_cases("${PATTERN}" "unexpected-mismatch" real_failures)
if (real_failures GREATER 0)
    message(FATAL_ERROR
        "test-ui-gzip-exclude: pattern '${PATTERN}' failed ${real_failures} of "
        "${_case_count} cases (see [unexpected-mismatch] lines above)")
endif()

# Anti-vacuity, part 3 -- the strongest one. A harness that cannot fail is not a
# test. Re-run the identical case set against the pattern that WAS shipped and
# was wrong; it must be rejected. If this passes, the cases have stopped
# discriminating and every green above is meaningless.
run_cases("/dist/_gzip" "control: expected mismatch" control_failures)
if (control_failures EQUAL 0)
    message(FATAL_ERROR
        "test-ui-gzip-exclude: the known-bad unanchored pattern '/dist/_gzip' "
        "PASSED all ${_case_count} cases. The case set no longer detects the "
        "bug this test exists for -- fix the cases, not this check.")
endif()

message(STATUS "test-ui-gzip-exclude: OK -- pattern '${PATTERN}' passes ${_case_count} cases; "
               "known-bad control correctly rejected (${control_failures} failures)")
