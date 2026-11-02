#!/usr/bin/env python3
"""Source gate for llama.cpp-s36q phase 3: the legacy standard-PP MUL_MAT_ID route must
not divert device-resident experts of a type the GPU can serve to the CPU.

ggml-sycl.cpp keeps `device_weight_cpu_fallback = (src0->type == A || src0->type == B)`.
For a type named there, the route pushes every device-resident expert into cpu_entries,
and dispatch_cpu_entries_now D2H-copies that expert's weights into pinned host memory on
EVERY dispatch (need_weight_d2h / mem_copy_async from the entry's lease) and runs the
expert on the CPU. That breaks both halves of the placement ruling (docs/backend/
sycl-memory-design.md, "Placement decides the executor"): a non-GPU executor for
device-resident data, and weight streaming. A type that the MoE MMVQ tables advertise
has an _id kernel (moe_mmvq_capability_supports_layout), so naming it in the flag also
makes that kernel unreachable on this route.

The gate: every type in the flag that the capability table advertises must be on the
EXEMPT list, which carries the ticket that owns it. An exemption whose type is no longer
in the flag is stale and fails (a premise assert, so the list cannot quietly decay).

Runs under pytest and as a plain script. Alternate copies (for a deliberately broken
tree) via GGML_SYCL_S36Q_{BACKEND,TABLES}_SOURCE. No device or build is touched.
"""

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"
BACKEND = Path(os.environ.get("GGML_SYCL_S36Q_BACKEND_SOURCE", str(SYCL / "ggml-sycl.cpp")))
TABLES = Path(os.environ.get("GGML_SYCL_S36Q_TABLES_SOURCE", str(SYCL / "moe-mmvq-tables.hpp")))

# Types still named in the flag although the capability table advertises them, with the
# ticket that owns removing each.
EXEMPT = {
    "GGML_TYPE_Q4_K": "llama.cpp-zzb5",
}


def flag_types():
    text = BACKEND.read_text(encoding="utf-8")
    m = re.search(r"\bdevice_weight_cpu_fallback\s*=\s*([^;]*);", text)
    assert m, "could not find the device_weight_cpu_fallback definition; every check would pass vacuously"
    types = set(re.findall(r"src0->type\s*==\s*(GGML_TYPE_[A-Z0-9_]+)", m.group(1)))
    assert types, f"device_weight_cpu_fallback names no type ({m.group(1)!r}); the parse is broken"
    return types


def capability_types():
    text = TABLES.read_text(encoding="utf-8")
    m = re.search(r"moe_mmvq_capability_supports_layout\s*\([^)]*\)\s*\{(.*?)\n\}", text, re.S)
    assert m, "could not parse moe_mmvq_capability_supports_layout"
    types = set(re.findall(r"case\s+(GGML_TYPE_[A-Z0-9_]+)\s*:", m.group(1)))
    assert len(types) >= 10, f"capability parse found only {len(types)} types; it is broken"
    return types


def test_no_advertised_type_is_diverted_to_the_cpu():
    bad = sorted((flag_types() & capability_types()) - set(EXEMPT))
    assert not bad, (
        f"device_weight_cpu_fallback names {', '.join(bad)}, which the MoE MMVQ capability table "
        f"advertises. The legacy PP route would D2H-copy those device-resident experts per dispatch "
        f"and run them on the CPU instead of the GPU _id kernel."
    )


def test_every_exemption_is_still_in_the_flag():
    stale = sorted(set(EXEMPT) - flag_types())
    assert not stale, (
        f"EXEMPT names {', '.join(stale)}, which device_weight_cpu_fallback no longer names. "
        f"Drop the entry: an exemption that covers nothing hides the next real one."
    )


if __name__ == "__main__":
    import sys

    failures = 0
    for fn_name, fn in sorted(list(globals().items())):
        if fn_name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {fn_name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {fn_name}: {exc}")
    if failures:
        print(f"{failures} test(s) failed")
        sys.exit(1)
    print("all tests passed")
