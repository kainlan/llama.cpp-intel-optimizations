#!/usr/bin/env python3
# llama.cpp-38af: pins the dedicated CPU-activation buffer type that closes the
# "CPU-produced activation consumed raw by a SYCL op" seam.
#
# The defect: llama-context gives the CPU backend the generic SYCL_Host buft
# for its compute buffer, and ggml_backend_sycl_device_supports_buft accepts
# SYCL_Host (weights live there too). ggml-backend-sched therefore sees "SYCL
# supports the source buffer", inserts NO split-input copy, and a SYCL op reads
# a CPU-produced activation straight out of pinned host memory -- a GPU
# zero-copy read the placement ruling forbids, which binbcast refuses
# mid-recording ("bin-broadcast raw host staging is not graph-recordable") and
# every other consumer performs silently.
#
# The fix is an identity, not a code path: the CPU backend's compute buffer
# gets its own buft (a clone of the host buft with its own .get_name, the
# SYCL_KV_Host pattern) that SYCL never accepts, so sched's standard
# split-input copy lands the activation in planned device memory before the
# SYCL split runs. Weights stay in SYCL_Host and are untouched.
#
# Each check documents its RED state against c9f464b48 (the commit this task
# branched from). Run with --root <dir> to point at an extracted tree. With
# --self-test every check is re-evaluated against a mutated copy of the text it
# reads and must flip to RED, so a check whose anchor silently stopped matching
# real code cannot pass vacuously.
import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--root", default=None, help="repo root to check (default: this checkout)")
parser.add_argument("--self-test", action="store_true", help="run mutation witnesses proving each check is not vacuous")
args = parser.parse_args()

root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
sycl_cpp = (root / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()
sycl_h = (root / "ggml/include/ggml-sycl.h").read_text()
ctx_cpp = (root / "src/llama-context.cpp").read_text()

BUFT_FN = "ggml_backend_sycl_cpu_activation_buffer_type"
NAME_FN = BUFT_FN + "_name"


def function_window(source: str, signature_anchor: str, window: int = 12000) -> str:
    """Exact function body starting at `signature_anchor`, by brace matching
    from the first '{' after the anchor. Returns "" when the anchor is absent,
    so every absence check must also require a non-empty window."""
    idx = source.find(signature_anchor)
    if idx < 0:
        return ""
    start = source.find("{", idx)
    if start < 0:
        return ""
    depth = 0
    end = min(len(source), start + window)
    for i in range(start, end):
        c = source[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return source[start:i + 1]
    return source[start:end]


def statement_window(source: str, anchor: str, window: int = 1400) -> str:
    idx = source.find(anchor)
    if idx < 0:
        return ""
    return source[idx:idx + window]


texts = {
    "sycl_cpp": sycl_cpp,
    "sycl_h": sycl_h,
    "buft_fn": function_window(sycl_cpp, "ggml_backend_buffer_type_t " + BUFT_FN + "() {"),
    "supports_buft_fn": function_window(
        sycl_cpp,
        "static bool ggml_backend_sycl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {",
        window=20000),
    "get_proc_address_fn": function_window(
        sycl_cpp, "static void * ggml_backend_sycl_reg_get_proc_address(", window=40000),
    # The CPU-backend branch of the backend/buft enumeration loop, up to the
    # SYCL (GPU) branch that follows it.
    "ctx_cpu_branch": statement_window(
        ctx_cpp, "if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {", window=1800),
    "ctx_cpp": ctx_cpp,
}


# --- (i) a distinct buft identity exists. RED at c9f464b48: no such function.
def check_i_distinct_identity(t):
    body = t["buft_fn"]
    name_fn = "static const char * " + NAME_FN + "(ggml_backend_buffer_type_t buft)"
    return (bool(body)
            and "*ggml_backend_sycl_host_buffer_type()" in body
            and "t.iface.get_name" in body
            and NAME_FN in body.split("t.iface.get_name", 1)[1].split(";", 1)[0]
            and name_fn in t["sycl_cpp"]
            and 'GGML_SYCL_NAME "_CpuActivation"' in t["sycl_cpp"])


def witness_i(t):
    t["buft_fn"] = t["buft_fn"].replace("t.iface.get_name", "t.iface.get_alignment")
    return t


# --- (ii) SYCL never accepts it, and still accepts the generic host buft so
# weights keep their executor. RED at c9f464b48: the second half already holds
# (that is the positive control); the identity check (i) is what fails.
def check_ii_never_accepted(t):
    body = t["supports_buft_fn"]
    return (bool(body)
            and "cpu_activation" not in body
            and "ggml_backend_sycl_host_buffer_type_name" in body
            and "ggml_backend_sycl_kv_host_buffer_type_name" in body)


def witness_ii(t):
    t["supports_buft_fn"] = t["supports_buft_fn"].replace(
        "return false;\n}", "if (buft->iface.get_name == ggml_backend_sycl_cpu_activation_buffer_type_name) { return true; }\n    return false;\n}")
    if t["supports_buft_fn"].endswith("return false;\n}") and "cpu_activation" not in t["supports_buft_fn"]:
        t["supports_buft_fn"] = t["supports_buft_fn"][:-1] + "    (void) ggml_backend_sycl_cpu_activation_buffer_type_name;\n}"
    return t


# --- (iii) llama-context selects it for the CPU backend in BOTH arms.
# RED at c9f464b48: the CPU branch uses only ggml_backend_dev_host_buffer_type.
def check_iii_context_selects_it(t):
    branch = t["ctx_cpu_branch"]
    ctx = t["ctx_cpp"]
    usm_arm = BUFT_FN + "()" in ctx and "#ifdef GGML_USE_SYCL" in ctx
    dl_arm = ('"' + BUFT_FN + '"' in ctx) and "ggml_backend_reg_get_proc_address" in ctx
    return (bool(branch)
            and "llama_context_sycl_cpu_activation_buft(" in branch
            and usm_arm
            and dl_arm)


def witness_iii(t):
    t["ctx_cpu_branch"] = t["ctx_cpu_branch"].replace("llama_context_sycl_cpu_activation_buft(", "unrelated(")
    return t


# --- (iv) the export surface: header declaration and proc-address entry.
# RED at c9f464b48: neither exists.
def check_iv_exported(t):
    proc = t["get_proc_address_fn"]
    return (bool(proc)
            and ("GGML_BACKEND_API ggml_backend_buffer_type_t " + BUFT_FN + "(void);") in t["sycl_h"]
            and ('strcmp(name, "' + BUFT_FN + '") == 0') in proc
            and ("(void *) " + BUFT_FN + ";") in proc)


def witness_iv(t):
    t["sycl_h"] = t["sycl_h"].replace(BUFT_FN + "(void);", "unrelated_fn(void);")
    return t


# --- (v) the debug-build whitelist in graph_compute must NOT admit the new
# buft: a CPU-activation tensor reaching a SYCL node means sched failed to
# insert the copy, and that assert is the tripwire for exactly this seam.
def check_v_not_whitelisted(t):
    idx = t["sycl_cpp"].find("auto is_supported_buft = [&](ggml_backend_buffer_type_t buft) {")
    if idx < 0:
        return False
    window = t["sycl_cpp"][idx:idx + 900]
    return "ggml_backend_sycl_buffer_type(sycl_ctx->device)" in window and BUFT_FN not in window


def witness_v(t):
    t["sycl_cpp"] = t["sycl_cpp"].replace(
        "buft == ggml_backend_sycl_kv_host_buffer_type();",
        "buft == ggml_backend_sycl_kv_host_buffer_type() || buft == " + BUFT_FN + "();")
    return t


checks = [
    ("i: the CPU-activation buft is a host-buft clone with its own .get_name", check_i_distinct_identity, witness_i),
    ("ii: supports_buft never accepts it and still accepts SYCL_Host (weights keep their executor)",
     check_ii_never_accepted, witness_ii),
    ("iii: llama-context selects it for the CPU backend in both the GGML_USE_SYCL and BACKEND_DL arms",
     check_iii_context_selects_it, witness_iii),
    ("iv: exported in ggml-sycl.h and through the backend proc-address table", check_iv_exported, witness_iv),
    ("v: the graph_compute debug whitelist does not admit it (tripwire for an uncopied activation)",
     check_v_not_whitelisted, witness_v),
]

failed = [name for name, check, _w in checks if not check(texts)]

mutant_failures = []
witnessed = 0
if args.self_test:
    # Self-test is only meaningful on a tree where the checks pass; on a RED
    # tree it would report every witness as "already failing".
    if not failed:
        for name, check, witness in checks:
            mutated = witness(dict(texts))
            if mutated == texts:
                mutant_failures.append(name + ": witness did not change the text (anchor missing)")
            elif check(mutated):
                mutant_failures.append(name + ": mutation was not detected (check is vacuous)")
            else:
                witnessed += 1
                print("cpu-activation-buft source: witness flipped -- " + name)

if failed or mutant_failures:
    if failed:
        print("cpu-activation-buft source contract failed:", file=sys.stderr)
        for name in failed:
            print("  FAIL " + name, file=sys.stderr)
    if mutant_failures:
        print("cpu-activation-buft source self-test failed: " + ", ".join(mutant_failures), file=sys.stderr)
    raise SystemExit(1)

if args.self_test:
    print("cpu-activation-buft source: self-test PASS (%d witness mutants, one per check)" % witnessed)
print("cpu-activation-buft source: PASS")
