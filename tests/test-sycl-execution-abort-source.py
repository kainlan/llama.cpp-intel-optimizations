#!/usr/bin/env python3
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
backend = (root / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()
registry_h = (root / "ggml/src/ggml-sycl/execution-lifecycle.hpp").read_text()
registry_cpp = (root / "ggml/src/ggml-sycl/execution-lifecycle.cpp").read_text()

checks = {
    "registry abort api declared": "error abort_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch," in registry_h,
    "registry abort api defined": "error Registry::abort_invocation(ContextId context, SessionId session, SessionResetEpoch reset_epoch," in registry_cpp,
    "registry abort marks quarantined terminal without releasing": "graph.state = graph_phase::QUARANTINED;" in registry_cpp and "graph.pending_participant_count = 0;" in registry_cpp and "device_owners_[claimed_device] = {};" not in registry_cpp.split("error Registry::abort_invocation",1)[1].split("error Registry::retire_graph",1)[0],
    "binding drain pins exist": "pin_count = 0;" in backend and "draining = false;" in backend and "ggml_sycl_execution_release_backend_pin" in backend,
    "callback snapshot pins before deref": "ggml_sycl_execution_pin_bound_backends_locked(context_id)" in backend and "fn(pin.backend, *pin.binding);" in backend,
    "destructor drains callbacks before unbind": "binding->cv.wait(lock, [&] { return binding->pin_count == 0; });" in backend,
    "pp-moe detached waiter removed": "std::thread([device, ring_depth, slot, generation" not in backend,
    "pp-moe shutdown drain exists": "pp_moe_onednn_drain_scratch_slots(device);" in backend,
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    print("execution abort source contract failed: " + ", ".join(failed), file=sys.stderr)
    raise SystemExit(1)
print("execution abort source contract: PASS")
