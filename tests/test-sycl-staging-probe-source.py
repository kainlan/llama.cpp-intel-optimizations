"""Host-checkable pins on tests/test-sycl-staging-probe.cpp (llama.cpp-23mk, G0 D1).

The probe decided its skip with ggml_backend_sycl_get_device_count(), which is the SCHEDULER-visible count:
the backend exposes only device 0 by default ("Multi-GPU: exposing only device 0 to scheduler"), so the count
is 1 on a two-card host and the probe exited 77 without running. These pins keep the skip on the physical
device count and keep the source queue off ggml_backend_sycl_init(1), which does not exist in that mode.
Each pin proves itself on a mutant.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "tests/test-sycl-staging-probe.cpp"


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def replace_last(text: str, old: str, new: str) -> str:
    """The code occurrence, not an earlier mention in the header comment."""
    i = text.rindex(old)
    return text[:i] + new + text[i + len(old):]


def violations(text: str) -> list:
    code = strip_comments(text)
    out = []
    if "ggml_sycl::test_physical_device_count()" not in code:
        out.append("the skip does not use ggml_sycl::test_physical_device_count()")
    if re.search(r"ggml_backend_sycl_get_device_count\s*\(\s*\)\s*<\s*[2-9]", code):
        out.append("the scheduler-visible device count is compared against two or more")
    if re.search(r"ggml_backend_sycl_init\s*\(\s*1\s*\)", code):
        out.append("device 1 is initialised as a scheduler backend, which is absent in the default mode")
    if "ggml_sycl::get_shared_context_queue(1)" not in code:
        out.append("the source queue is not the per-device shared-context queue of device 1")
    # The shared-context queues exist only once init_shared_context_queues() has run (the MoE and split
    # paths' lazy creator), so the probe calls it with the physical count BEFORE fetching device 1's queue.
    init = re.search(r"ggml_sycl::init_shared_context_queues\s*\(\s*physical_devices\s*\)", code)
    get = re.search(r"ggml_sycl::get_shared_context_queue\s*\(\s*1\s*\)", code)
    if init is None:
        out.append("the shared-context queues are fetched without init_shared_context_queues(physical_devices)")
    elif get is not None and init.start() > get.start():
        out.append("init_shared_context_queues() runs after the queue fetch")
    # No wait may be unbounded: a watchdog thread leaves the process, naming the step, and the controls that
    # discriminate where a hang lives run before the cross-context dependency.
    if "_Exit(1)" not in code or "FAIL: HANG at %s" not in code:
        out.append("no step watchdog that names the hung step and leaves with _Exit(1)")
    # G0's first probe held leg 1 behind a kernel spinning on a host-USM word the host wrote, and the gate never
    # opened: discrete Battlemage has no usm_atomic_host_allocations, so a device poll of a host-written word
    # (and a host read of a device-written one) has no visibility guarantee. The hold is a calibrated busy
    # kernel, and nothing in the code polls a flag.
    if re.search(r"\bvolatile\b", code) or re.search(r"while\s*\(\s*\*", code):
        out.append("a device or host loop polls a memory word (volatile or while(*flag)); the hold must be the "
                   "calibrated busy kernel")
    if "submit_busy(" not in code or "k_target_busy_ms" not in code or "staging_probe_classify(busy_ms" not in code:
        out.append("the hold is not the calibrated busy kernel read against its measured time")
    controls = ["control_g_busy_alone", "control_i_leg1_alone", "control_ii_leg2_alone",
                "control_iv_host_wait_between", "iii_leg2_submit"]
    at = [code.find('"%s"' % name) for name in controls]
    if min(at) < 0:
        out.append("a control or step is missing: " + ", ".join(n for n, a in zip(controls, at) if a < 0))
    elif at != sorted(at):
        out.append("the controls do not run before the cross-context dependency step")
    for m in re.finditer(r"\b(\w+)\.wait\(\)", code):
        before = code[: m.start()]
        # Every .wait() sits after a step_scope or inside a run_step lambda opened earlier in main.
        if before.rfind("step_scope") < 0 and before.rfind("run_step") < 0:
            out.append("a wait() before any bounded step")
    if re.search(r"physical_devices\s*<\s*2", code) is None:
        out.append("the skip does not compare the physical count against two")
    return out


def mutants_of(text: str) -> dict:
    return {
        "skip on the scheduler count": replace_last(text, "physical_devices < 2 ||", "ggml_backend_sycl_get_device_count() < 2 ||"),
        "device 1 as a backend": replace_last(text, "ggml_sycl::get_shared_context_queue(1)", "ggml_backend_sycl_init(1)"),
        "queue fetched without being created": replace_last(
            text, "ggml_sycl::init_shared_context_queues(physical_devices);", ""),
        "queue created after the fetch": replace_last(
            replace_last(text, "ggml_sycl::init_shared_context_queues(physical_devices);", ""),
            "sycl::queue * q_source_ptr = ggml_sycl::get_shared_context_queue(1);",
            "sycl::queue * q_source_ptr = ggml_sycl::get_shared_context_queue(1);\n"
            "    ggml_sycl::init_shared_context_queues(physical_devices);"),
        "watchdog does not exit": replace_last(text, "_Exit(1);", "std::fflush(stdout);"),
        "dependency step before the controls": text.replace(
            "namespace {", 'const char * k_early = "iii_leg2_submit";\nnamespace {', 1),
        "polling gate re-introduced": replace_last(
            text, "x ^= x >> 13;", "x ^= x >> 13;\n                volatile int * flag = nullptr;\n"
            "                while (*flag == 0) {\n                }"),
        "hold no longer the calibrated kernel": text.replace("k_target_busy_ms", "k_busy"),
        "no physical count": replace_last(text, "ggml_sycl::test_physical_device_count()", "2"),
    }


def test_probe_counts_physical_devices() -> None:
    assert violations(PROBE.read_text(encoding="utf-8")) == []


def test_each_pin_catches_its_mutant() -> None:
    text = PROBE.read_text(encoding="utf-8")
    for name, mutated in mutants_of(text).items():
        assert mutated != text, name
        assert violations(mutated), "mutant not caught: " + name


def main() -> int:
    text = PROBE.read_text(encoding="utf-8")
    found = violations(text)
    for v in found:
        print("FAIL:", v)
    mutants = mutants_of(text)
    for name, mutated in mutants.items():
        if mutated == text or not violations(mutated):
            print("FAIL: mutant not caught:", name)
            found.append(name)
    if not found:
        print("PASS: the probe counts physical devices (%d mutants caught)" % len(mutants))
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
