"""Shared loader for the gates that score the dense MUL_MAT type list in place (llama.cpp-gldu).

ggml_sycl_mul_mat_type_supported delegates to the list in unified-types.hpp, which the zone planner reads as well. The
gates that parse and mutate the list itself splice the header's switch back into ggml-sycl.cpp's function text on
load; a wrapper that stops delegating, or a header that loses the list, fails loudly.
"""

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_HEADER = _ROOT / "ggml/src/ggml-sycl/unified-types.hpp"


def with_dense_type_list_inlined(source: str) -> str:
    header = _HEADER.read_text(encoding="utf-8")
    listed = re.search(r"inline bool dense_mul_mat_type_supported\(ggml_type type\) \{(.*?)\n\}\n", header, re.S)
    wrapper = re.search(
        r"static bool ggml_sycl_mul_mat_type_supported\(ggml_type type\) \{[^{}]*"
        r"return ggml_sycl::dense_mul_mat_type_supported\(type\);\n\}\n",
        source,
    )
    assert listed is not None and wrapper is not None, "the dense MUL_MAT type list is no longer shared"
    inlined = "static bool ggml_sycl_mul_mat_type_supported(ggml_type type) {" + listed.group(1) + "\n}\n"
    return source[: wrapper.start()] + inlined + source[wrapper.end():]
