# Which published UI bundle version this tree embeds.
#
# The fork's own build number is never published to the bundle bucket, so
# without a pin the build embeds the floating `latest` bundle, which can stop
# matching this tree's server code at any time. The pin is the nearest published
# version at or before the upstream base (base 694ec2354 counts 11204 commits;
# b11204 is not published, b11203 is, and 694ec2354 itself does not touch the
# UI). Move it when merging a newer upstream.
#
# The default is deliberately a plain variable, not a cache entry: a cached
# default would outlive a pin moved in the tree, and every existing build tree
# would keep embedding the old bundle. An explicit -DLLAMA_UI_PINNED_VERSION=...
# is the user's override and persists as usual; an empty value restores the
# derivation from the build number.
if(NOT DEFINED LLAMA_UI_PINNED_VERSION)
    set(LLAMA_UI_PINNED_VERSION "b11203")
endif()

# Resolve HF asset version: env HF_WEBUI_VERSION > env HF_UI_VERSION > pin >
# derived from build number > unset.
function(llama_ui_resolve_hf_version out_var)
    if(DEFINED ENV{HF_WEBUI_VERSION})
        set(_version "$ENV{HF_WEBUI_VERSION}")
        message(DEPRECATION "HF_WEBUI_VERSION env var is deprecated, use HF_UI_VERSION instead")
    elseif(DEFINED ENV{HF_UI_VERSION})
        set(_version "$ENV{HF_UI_VERSION}")
    elseif(NOT "${LLAMA_UI_PINNED_VERSION}" STREQUAL "")
        set(_version "${LLAMA_UI_PINNED_VERSION}")
    elseif(DEFINED LLAMA_BUILD_NUMBER)
        set(_version "b${LLAMA_BUILD_NUMBER}")
    else()
        set(_version "")
    endif()
    set(${out_var} "${_version}" PARENT_SCOPE)
endfunction()
