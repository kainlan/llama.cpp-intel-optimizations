# Gate for the UI bundle version resolution in ui/apps/chat/ui-version.cmake.
#
# The fork's own build number is never a published bundle version, so a build
# that derives the version from it embeds the floating `latest` bundle, which
# can stop matching this tree's server code at any time. The resolver therefore
# prefers a committed pin over the build-number derivation. This test fixes the
# order: HF_WEBUI_VERSION, HF_UI_VERSION, the pin, then b<build number>, then
# empty.
#
# Run:  cmake -DUI_VERSION_CMAKE=<path to ui/apps/chat/ui-version.cmake> -P test-ui-version-pin.cmake

cmake_minimum_required(VERSION 3.18)

if (NOT DEFINED UI_VERSION_CMAKE)
    get_filename_component(_here "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
    set(UI_VERSION_CMAKE "${_here}/../ui/apps/chat/ui-version.cmake")
endif()
if (NOT EXISTS "${UI_VERSION_CMAKE}")
    message(FATAL_ERROR "test-ui-version-pin: no such file: ${UI_VERSION_CMAKE}")
endif()
if (NOT DEFINED UI_CMAKE)
    get_filename_component(_here "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
    set(UI_CMAKE "${_here}/../ui/apps/chat/CMakeLists.txt")
endif()
if (NOT EXISTS "${UI_CMAKE}")
    message(FATAL_ERROR "test-ui-version-pin: no such file: ${UI_CMAKE}")
endif()

# The caller's environment must not change any verdict.
unset(ENV{HF_WEBUI_VERSION})
unset(ENV{HF_UI_VERSION})

# Case a needs the default, so the pin must be undefined at include time.
unset(LLAMA_UI_PINNED_VERSION)
unset(LLAMA_UI_PINNED_VERSION CACHE)
unset(LLAMA_BUILD_NUMBER)
include("${UI_VERSION_CMAKE}")

if (NOT COMMAND llama_ui_resolve_hf_version)
    message(FATAL_ERROR "test-ui-version-pin: ${UI_VERSION_CMAKE} does not define llama_ui_resolve_hf_version()")
endif()

set(_pass 0)
macro(expect case want)
    llama_ui_resolve_hf_version(_got)
    if (NOT "${_got}" STREQUAL "${want}")
        message(FATAL_ERROR "test-ui-version-pin: case ${case}: got '${_got}', want '${want}'")
    endif()
    message(STATUS "PASS ${case}: '${_got}'")
    math(EXPR _pass "${_pass}+1")
endmacro()

# a. default pin, build number set: the pin wins over the build number.
set(_default_pin "${LLAMA_UI_PINNED_VERSION}")
if (NOT _default_pin MATCHES "^b[0-9]+$")
    message(FATAL_ERROR "test-ui-version-pin: case a: default pin '${_default_pin}' does not match ^b[0-9]+$")
endif()
set(LLAMA_BUILD_NUMBER 19064)
expect("a (default pin beats build number)" "${_default_pin}")

# b. empty pin: the build-number derivation returns.
set(LLAMA_UI_PINNED_VERSION "")
expect("b (empty pin, build number)" "b19064")

# c. empty pin, no build number: empty.
unset(LLAMA_BUILD_NUMBER)
expect("c (empty pin, no build number)" "")

# d. env HF_UI_VERSION beats the pin and the build number.
set(ENV{HF_UI_VERSION} "b777")
set(LLAMA_UI_PINNED_VERSION "${_default_pin}")
set(LLAMA_BUILD_NUMBER 19064)
expect("d (HF_UI_VERSION beats pin)" "b777")

# e. env HF_WEBUI_VERSION beats HF_UI_VERSION.
set(ENV{HF_WEBUI_VERSION} "b555")
expect("e (HF_WEBUI_VERSION beats HF_UI_VERSION)" "b555")

# f. the default must not be a cache entry: a cached default would outlive a pin
# moved in the tree, and every existing build tree would keep the old bundle.
file(READ "${UI_VERSION_CMAKE}" _text)
string(REGEX REPLACE "(^|\n)[ \t]*#[^\n]*" "\\1" _code "${_text}")
if (NOT _code MATCHES "set\\([ \t\n]*LLAMA_UI_PINNED_VERSION")
    message(FATAL_ERROR "test-ui-version-pin: case f: no set(LLAMA_UI_PINNED_VERSION ...) found; the check is vacuous")
endif()
if (_code MATCHES "set\\([ \t\n]*LLAMA_UI_PINNED_VERSION[^)]*CACHE")
    message(FATAL_ERROR "test-ui-version-pin: case f: the default pin is a set(... CACHE ...) entry")
endif()
message(STATUS "PASS f (default pin is not a cache entry)")
math(EXPR _pass "${_pass}+1")

# g. editing the pin must re-run provisioning: ui/apps/chat/CMakeLists.txt has to
# include the file (or the resolver is not defined) and list it among the
# provisioning step's dependencies.
file(READ "${UI_CMAKE}" _ui_text)
string(REGEX REPLACE "(^|\n)[ \t]*#[^\n]*" "\\1" _ui_code "${_ui_text}")
if (NOT _ui_code MATCHES "include\\([^)]*ui-version\\.cmake\"?[ \t\n]*\\)")
    message(FATAL_ERROR "test-ui-version-pin: case g: ${UI_CMAKE} has no include() of ui-version.cmake")
endif()
if (NOT _ui_code MATCHES "(set|list)\\((APPEND )?UI_ASSET_DEPENDS[^)]*ui-version\\.cmake")
    message(FATAL_ERROR "test-ui-version-pin: case g: ui-version.cmake is not in UI_ASSET_DEPENDS in ${UI_CMAKE}, so editing the pin would not re-run provisioning")
endif()
message(STATUS "PASS g (ui-version.cmake is included and listed in UI_ASSET_DEPENDS)")
math(EXPR _pass "${_pass}+1")

unset(ENV{HF_WEBUI_VERSION})
unset(ENV{HF_UI_VERSION})
message(STATUS "test-ui-version-pin: OK -- ${_pass} cases passed")
