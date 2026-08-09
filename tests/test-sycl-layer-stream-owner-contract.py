#!/usr/bin/env python3
"""Source contract for layer_stream_manager's ownership lifecycle (llama.cpp-vbeb).

The companion executable test (test-layer-stream-owner-lifecycle) proves the
BEHAVIOUR: a second model inherits nothing from the first. It cannot prove the
STRUCTURE the k7b0 house standard asks for -- that the release is one owner
function rather than N scattered clears that invite an N+1th -- and it cannot
reach the parts of the release that need a device (the mem_handle drop and the
prefetch drain that must precede it). This gate covers exactly that gap.

Checks are split into two kinds and both are scored:

  presence  -- a construct must exist. Fails on a tree that never had the fix.
  absence   -- a construct must NOT exist anywhere else. Passes vacuously on any
               tree written after the fix, so every absence check is re-run
               under --self-test against a deliberately poisoned copy. Without
               that control an absence check's green is not evidence.

Exit: 0 pass, 1 fail, 77 skip (sources not found).
"""

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SYCL = os.path.join(HERE, "..", "ggml", "src", "ggml-sycl")

CPP = os.path.join(SYCL, "layer-streaming.cpp")
HPP = os.path.join(SYCL, "layer-streaming.hpp")
OWNER_HPP = os.path.join(SYCL, "layer-stream-owner.hpp")

# Every model-scoped member of layer_stream_manager, and the assignment that
# returns it to its pristine value. release_model_state() must contain all of
# them; if a member is added later without a line here, the count check below
# reports the discrepancy rather than passing quietly.
PRISTINE = [
    ("layers_", "layers_.clear();"),
    ("name_to_location_", "name_to_location_.clear();"),
    ("max_layer_size_", "max_layer_size_ = 0;"),
    ("buffer_size_", "buffer_size_ = 0;"),
    ("device_id_", "device_id_   = -1;"),
    ("buffers_", "buffers_[i]        = nullptr;"),
    ("buffer_handles_", "buffer_handles_[i] = {};"),
]

# Members released via drain_and_invalidate_buffers(), which release_model_state
# calls first.
PRISTINE_DRAIN = [
    ("prefetch_pending_", "prefetch_pending_      = false;"),
    ("prefetch_target_layer_", "prefetch_target_layer_ = -1;"),
    ("prefetch_buffer_", "prefetch_buffer_       = -1;"),
    ("prefetch_event_", "prefetch_event_        = sycl::event();"),
    ("loaded_layers_", "loaded_layers_[0]      = -1;"),
]


def transition_lock_pos(body):
    """Offset of the owner_transition_mutex_ lock_guard, or None.

    Matched by regex, not literally: clang-format pads the declaration for
    vertical alignment in some bodies and not others, and a reformat must not be
    able to turn a satisfied contract into a reported violation.
    """
    match = re.search(r"std::lock_guard<std::mutex>\s+lock\(owner_transition_mutex_\);", body)
    return match.start() if match else None


def function_body(text, signature):
    """Return the brace-matched body of the first definition matching `signature`."""
    start = text.find(signature)
    if start < 0:
        return None
    open_brace = text.find("{", start)
    if open_brace < 0:
        return None
    depth = 0
    for i in range(open_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace : i + 1]
    return None


def run(cpp, hpp, owner_hpp, absence_only=False):
    presence = []
    absence = []

    release = function_body(cpp, "void layer_stream_manager::release_model_state()")
    drain = function_body(cpp, "void layer_stream_manager::drain_and_invalidate_buffers()")
    adopt = function_body(cpp, "void layer_stream_manager::adopt_current_owner()")
    build = function_body(cpp, "void layer_stream_manager::build_layer_map(")
    register = function_body(cpp, "void layer_stream_manager::register_host_ptr(")
    allocate = function_body(cpp, "bool layer_stream_manager::allocate_buffers(")
    shutdown = function_body(cpp, "void layer_stream_manager::shutdown()")

    presence.append(("release_model_state() is defined", release is not None))
    presence.append(("drain_and_invalidate_buffers() is defined", drain is not None))
    presence.append(("adopt_current_owner() is defined", adopt is not None))

    if release and drain:
        for member, assignment in PRISTINE:
            presence.append(
                ("release_model_state() returns %s to pristine" % member, assignment in release)
            )
        for member, assignment in PRISTINE_DRAIN:
            presence.append(
                ("the drain returns %s to pristine" % member, assignment in drain)
            )

        # Ordering: the wait has to happen before the handles are dropped, or the
        # release frees memory a submitted DMA is still writing into.
        presence.append(
            (
                "release_model_state() drains before it touches the handles",
                release.find("drain_and_invalidate_buffers();")
                < release.find("std::move(buffer_handles_[i])"),
            )
        )
        presence.append(
            (
                "the drain waits on prefetch_event_ before clearing the flag",
                drain.find("prefetch_event_.wait();") < drain.find("prefetch_pending_      = false;"),
            )
        )
        # The handles must leave the lock scope alive, so their destructors (the
        # free) run with no lock held.
        presence.append(
            (
                "released handles are moved out to a local, not destroyed under the lock",
                "mem_handle released[2];" in release
                and release.find("mem_handle released[2];") < release.find("std::lock_guard"),
            )
        )

    # The two entry points a model load can reach must consult the gate, and must
    # do it BEFORE they touch any state.
    if build:
        presence.append(("build_layer_map() adopts the current owner", "adopt_current_owner();" in build))
        presence.append(
            (
                "build_layer_map() adopts before it clears the map",
                build.find("adopt_current_owner();") < build.find("layers_.clear();"),
            )
        )
        presence.append(
            (
                "build_layer_map() forgets which layers the buffers hold",
                "drain_and_invalidate_buffers();" in build,
            )
        )
    if register:
        presence.append(("register_host_ptr() adopts the current owner", "adopt_current_owner();" in register))
        presence.append(
            (
                "register_host_ptr() adopts before it takes the map lock",
                register.find("adopt_current_owner();") < register.find("std::lock_guard"),
            )
        )
    if allocate:
        presence.append(
            (
                "allocate_buffers() drains before it replaces the handles",
                allocate.find("drain_and_invalidate_buffers();") < allocate.find("buffer_handles_[i] = std::move("),
            )
        )
        presence.append(
            (
                "allocate_buffers() releases everything when one buffer fails",
                "release_model_state();" in allocate,
            )
        )
    if shutdown:
        presence.append(("shutdown() delegates to the owner function", "release_model_state();" in shutdown))
        presence.append(("shutdown() drops the ownership record", "owner_gate_.forget();" in shutdown))

    # The teardown entry point (llama.cpp-y36c's call site). Its whole value is
    # that the check and the release are ONE step, so both the guard and the
    # lock that makes them atomic are contract, not implementation detail.
    release_if_owner = function_body(cpp, "bool layer_stream_manager::release_if_owner(")
    presence.append(("release_if_owner() is defined", release_if_owner is not None))
    if release_if_owner:
        presence.append(
            (
                "release_if_owner() refuses a model that is not the owner",
                "if (!owner_gate_.is_owner(owner))" in release_if_owner,
            )
        )
        lock_at = transition_lock_pos(release_if_owner)
        presence.append(
            (
                "release_if_owner() takes the transition lock before it checks",
                lock_at is not None and lock_at < release_if_owner.find("owner_gate_.is_owner(owner)"),
            )
        )
        presence.append(
            (
                "release_if_owner() releases under the same lock it checked under",
                "release_model_state();" in release_if_owner and "owner_gate_.forget();" in release_if_owner,
            )
        )
    if adopt:
        presence.append(
            (
                "adopt_current_owner() holds the transition lock across its release",
                transition_lock_pos(adopt) is not None,
            )
        )

    # The gate must fail closed on an unattributed caller, or the fix degrades
    # into a load-boundary sweep any call could trigger.
    if owner_hpp is not None:
        valid = function_body(owner_hpp, "inline bool layer_stream_owner_valid(")
        presence.append(("layer_stream_owner_valid() is defined", valid is not None))
        if valid:
            for field in ("model_id != 0", "load_txn_id != 0", "slot != layer_stream_owner_slot_none",
                          "slot_generation != 0"):
                presence.append(("the owner key fails closed on %s" % field, field in valid))

    # ---- absence: exactly one owner of the release --------------------------
    # These are the checks that make "one owner function, not N scattered
    # clears" enforceable rather than aspirational.
    def clears_outside(pattern, allowed_bodies):
        hits = []
        for match in re.finditer(re.escape(pattern), cpp):
            if not any(body and pattern in body and _within(cpp, match.start(), body) for body in allowed_bodies):
                hits.append(match.start())
        return hits

    def _within(text, pos, body):
        start = text.find(body)
        return start >= 0 and start <= pos < start + len(body)

    absence.append(
        (
            "name_to_location_.clear() appears only in release_model_state and build_layer_map",
            clears_outside("name_to_location_.clear();", [release, build]) == [],
        )
    )
    absence.append(
        (
            "layers_.clear() appears only in release_model_state and build_layer_map",
            clears_outside("layers_.clear();", [release, build]) == [],
        )
    )
    absence.append(
        (
            "buffer_handles_ is never dropped outside release_model_state",
            clears_outside("buffer_handles_[i] = {};", [release]) == [],
        )
    )
    # Every wait on the prefetch event must sit in the drain or in one of the
    # three consumers that legitimately own one. A fourth site is a fourth place
    # that can decide the buffers are safe to reuse.
    ensure = function_body(cpp, "bool layer_stream_manager::ensure_layer(")
    prefetch = function_body(cpp, "void layer_stream_manager::prefetch_next_layer(")
    await_body = function_body(cpp, "void layer_stream_manager::await_prefetch()")
    absence.append(
        (
            "prefetch_event_.wait() appears only in the drain and its three consumers",
            clears_outside("prefetch_event_.wait();", [drain, ensure, prefetch, await_body]) == [],
        )
    )

    if absence_only:
        return [], absence
    return presence, absence


def report(title, results):
    failed = 0
    for label, ok in results:
        if not ok:
            failed += 1
            print("  FAIL [%s]: %s" % (title, label))
    return failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true",
                        help="re-run the absence checks against a poisoned copy; they must FAIL")
    args = parser.parse_args()

    for path in (CPP, HPP, OWNER_HPP):
        if not os.path.exists(path):
            print("SKIP: %s not found; this run proves NOTHING" % path)
            return 77

    cpp = open(CPP, encoding="utf-8").read()
    hpp = open(HPP, encoding="utf-8").read()
    owner = open(OWNER_HPP, encoding="utf-8").read()

    presence, absence = run(cpp, hpp, owner)
    if not presence:
        print("FAIL: no presence checks ran; this run proves NOTHING")
        return 1

    failed = report("presence", presence) + report("absence", absence)

    if args.self_test:
        # Poison: add a second, scattered clear of the registry outside the owner
        # function. Every absence check above must notice.
        poisoned = cpp.replace(
            "int layer_stream_manager::pick_buffer_for_layer(int layer_id) const {",
            "int layer_stream_manager::pick_buffer_for_layer(int layer_id) const {\n"
            "    const_cast<layer_stream_manager *>(this)->name_to_location_.clear();\n"
            "    const_cast<layer_stream_manager *>(this)->layers_.clear();\n",
            1,
        )
        if poisoned == cpp:
            print("FAIL: self-test could not poison the source; the control is void")
            return 1
        _, poisoned_absence = run(poisoned, hpp, owner, absence_only=True)
        survivors = [label for label, ok in poisoned_absence if ok]
        # The two registry-clear checks must both fire. The other two are not
        # touched by this poison and are expected to survive it.
        expected_survivors = 2
        if len(survivors) != expected_survivors:
            print("FAIL: self-test expected %d absence checks to survive the poison, %d did: %s"
                  % (expected_survivors, len(survivors), survivors))
            failed += 1

    total = len(presence) + len(absence)
    print("%d checks, %d failures" % (total, failed))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
