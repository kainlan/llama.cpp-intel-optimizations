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
