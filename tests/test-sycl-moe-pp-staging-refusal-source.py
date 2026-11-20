"""Source contract: a staging buffer the MoE prompt path of ggml_sycl_mul_mat_id cannot get ends the op with a named
refusal, never a process abort.

A buffer the card does not grant is a fit problem, not a broken part. Each such site of the prompt path throws
ggml_sycl_fallback_error through one helper, ggml_sycl_moe_pp_refuse_staging, whose WARN names the tensor, the buffer,
the rows and bytes asked and the device; graph_compute's catch drains the queue and returns GGML_STATUS_FAILED.

Host-only: reads sources, builds nothing, loads no model and touches no device.

What this gate pins:
  - the helper: [[noreturn]], one WARN naming tensor, buffer, rows, bytes and device, then the throw, and no abort;
  - the row-map staging (moe_pp_row_mappings) refuses when its allocation is null;
  - the per-expert path's output staging (moe_pp_dst_contiguous) refuses when its allocation is null, where it used to
    run on with a null base that every later kernel wrote through;
  - the F32 activation staging (moe_pp_src1_f32) refuses at each of its three sites (the setup, the per-expert check
    and the F32 batch dispatch) through one lambda;
  - the CPU island's host-pinned staging (moe_pp_cpu_island_act / _out) refuses when either handle does not resolve;
  - the unwind: the per-expert path's exit work (the done marker, the scratch slot recorded with its owners, the expert
    leases and the pointer-table bundle retained until that marker, the transient down SOA released) is one lambda
    that runs at most once (a flag set before its first submit). One try covers the path from the scratch setup to the
    normal exit: the output and activation refusals, the loop's own fallback throws (a refused secondary executor, an
    expert unresolved before submit) and the island all sit inside it, and its handler for ggml_sycl_fallback_error
    runs the exit work and rethrows. The normal exit and that handler are the lambda's only callers: a refusal only
    refuses. The row-map refusal stays before the try (nothing is claimed or submitted there yet);
  - none of the old abort texts is left in the backend;
  - the private-testing failpoints that let the device test force each refusal exist only under
    GGML_SYCL_Q1_NVFP4_ROUTE_TESTING (the private fixture objects), fold to the constant false otherwise, and sit in
    front of the allocation they stand for; the seam header names them;
  - the device test test-moe-pp-staging-refusal is registered against the private fixtures with the `model` label and
    arms all three failpoints.

Every claim is checked on COMMENT-STRIPPED, whitespace-normalized text with adjacent string literals joined, and has a
mutant that must make it fail.

Pytest-style (module-level test_* functions): registered with llama_test_pytest.
"""

from sycl_gate import body, first_missing, mutate, norm, ordered, read

SYCL_REL = "ggml/src/ggml-sycl/ggml-sycl.cpp"
SEAM_REL = "ggml/src/ggml-sycl/tests/q1-nvfp4-production-route-test-seam.hpp"
CMAKE_REL = "ggml/src/ggml-sycl/CMakeLists.txt"
TEST_REL = "ggml/src/ggml-sycl/tests/test-moe-pp-staging-refusal.cpp"

SYCL = read(SYCL_REL)
SEAM = read(SEAM_REL)
CMAKE = read(CMAKE_REL)
TEST = read(TEST_REL)

MMID_SIG = "static void ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx, ggml_tensor * dst) try {"
HELPER_SIG = "[[noreturn]] static void ggml_sycl_moe_pp_refuse_staging("
HELPER_WARN = ('GGML_LOG_WARN("[MOE-PP-STAGING] refused tensor=%s buffer=%s rows=%zu bytes=%zu device=%d')
HELPER_THROW = "throw ggml_sycl_fallback_error(reason);"
REFUSE = "ggml_sycl_moe_pp_refuse_staging("

ROW_MAP_SITE = ("if (!GGML_SYCL_MOE_PP_STAGING_FAILPOINT(GGML_SYCL_Q1_NVFP4_TEST_FAILURE_PP_ROW_MAP_STAGING)) { "
                "dev_row_mappings.alloc(*stream, ctx.device, row_maps_total, \"moe_pp_row_mappings\"); } "
                "if (!dev_row_mappings.get()) { "
                "ggml_sycl_moe_pp_refuse_staging(src0, ctx.device, \"moe_pp_row_mappings\", row_maps_total,")

DST_SITE = ("if (!GGML_SYCL_MOE_PP_STAGING_FAILPOINT(GGML_SYCL_Q1_NVFP4_TEST_FAILURE_PP_DST_STAGING)) { "
            "dst_contiguous.alloc(*stream, ctx.device, sizeof(float) * ggml_nelements(dst), "
            "\"moe_pp_dst_contiguous\"); } dst_contiguous_ptr = dst_contiguous.get(); if (!dst_contiguous_ptr) { "
            "ggml_sycl_moe_pp_refuse_staging(src0, ctx.device, \"moe_pp_dst_contiguous\",")

SRC1_FAILPOINT = ("if (src1_contiguous_f32.get()) { return true; } "
                  "if (GGML_SYCL_MOE_PP_STAGING_FAILPOINT(GGML_SYCL_Q1_NVFP4_TEST_FAILURE_PP_SRC1_F32_STAGING)) { "
                  "return false; } try { src1_contiguous_f32.alloc(")
SRC1_REFUSE_LAMBDA = ("auto refuse_src1_f32_staging = [&]() { "
                      "ggml_sycl_moe_pp_refuse_staging(src0, ctx.device, \"moe_pp_src1_f32\",")
SRC1_SETUP = "if (!pp_mxfp4_soa_f16_ready && !ensure_src1_contiguous_f32()) { refuse_src1_f32_staging(); }"
SRC1_EXPERT = "if (!expert_use_mxfp4_soa_f16 && !copy_src1_current_f32()) { refuse_src1_f32_staging(); }"
SRC1_BATCHES = "auto dispatch_f32_batches = [&]() { if (!copy_src1_current_f32()) { refuse_src1_f32_staging(); }"

ISLAND_SIG = "auto dispatch_host_expert_groups = [&](std::vector<pp_host_expert_group> & groups) {"
ISLAND_CHECK = "if (!act_r || !out_r) {"
ISLAND_AFTER = "float * act_pinned = static_cast<float *>(act_r.ptr);"
ISLAND_BUFFERS = ('"moe_pp_cpu_island_act"', '"moe_pp_cpu_island_out"')

RETAIN_SIG = "auto retain_pp_dispatch_until_done = [&]() {"
LEASES_DECL = "std::vector<ggml_sycl::mem_handle> pp_gpu_leases;"
CLAIM_DECL = "pp_moe_onednn_scratch_claim pp_moe_onednn_scratch_claim_state;"
RETAIN_STEPS = (
    "prompt_down_dispatch_done = ggml_sycl_submit_marker<pp_moe_prompt_down_dispatch_done_marker>(*stream);",
    "pp_moe_onednn_scratch_claim_state.bind_owners(std::move(pp_moe_slot_owners));",
    "pp_moe_onednn_scratch_claim_state.record(prompt_down_dispatch_done);",
    "ggml_sycl::retain_handles_until_event(std::move(pp_gpu_leases),",
    "ggml_sycl_snapshot_moe_ptr_table_dispatch_bundle(src0_extra, ctx.device);",
    "ggml_sycl_release_moe_tensor_layout(src0, ctx.device, GGML_LAYOUT_SOA);",
)
RETAIN_ONCE = ("bool pp_exit_work_done = false; auto retain_pp_dispatch_until_done = [&]() { "
               "if (pp_exit_work_done) { return; } pp_exit_work_done = true; sycl::event prompt_down_dispatch_done;")
RETAIN_CALL = "retain_pp_dispatch_until_done();"
# The try opens right after the activation refusal lambda, before the scratch setup; its handler follows its body.
TRY_SIG = "\"MUL_MAT_ID prompt activation staging did not fit\"); }; try {"
HANDLER = " catch (const ggml_sycl_fallback_error &) { retain_pp_dispatch_until_done(); throw; }"
SCRATCH_CLAIM = "pp_moe_onednn_claim_scratch_slot(ctx.device, planned_ring_depth, scratch_slot)"
LOOP_HEAD = "for (int64_t i02 = 0; i02 < n_as; i02++) {"
SECONDARY_THROW = "throw ggml_sycl_fallback_error(\"MUL_MAT_ID retained prompt secondary executor refused exact handle\");"
UNRESOLVED_THROW = "throw ggml_sycl_fallback_error(\"MUL_MAT_ID selected prompt expert unresolved before submit\");"
ISLAND_CALL = "dispatch_host_expert_groups(pp_host_groups); pp_phase_log(\"pp-host-dispatch-done\","
NORMAL_EXIT = "retain_pp_dispatch_until_done(); pp_phase_log(\"pp-retain-submitted\","

OLD_ABORTS = (
    "failed to allocate PP row-map staging",
    "Failed to allocate PP F32 activation staging",
    "Failed host-pinned alloc for PP CPU island",
)

FAILPOINT_MACRO = ("#ifdef GGML_SYCL_Q1_NVFP4_ROUTE_TESTING # define GGML_SYCL_MOE_PP_STAGING_FAILPOINT(failure) "
                   "ggml_sycl_q1_nvfp4_test_consume_failure(failure) #else # define "
                   "GGML_SYCL_MOE_PP_STAGING_FAILPOINT(failure) false #endif")
FAILURES = (
    "GGML_SYCL_Q1_NVFP4_TEST_FAILURE_PP_ROW_MAP_STAGING",
    "GGML_SYCL_Q1_NVFP4_TEST_FAILURE_PP_DST_STAGING",
    "GGML_SYCL_Q1_NVFP4_TEST_FAILURE_PP_SRC1_F32_STAGING",
)

TEST_NAME = "test-moe-pp-staging-refusal"
REG_STEPS = (
    "if (NOT GGML_BACKEND_DL) add_executable(test-moe-pp-staging-refusal "
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test-moe-pp-staging-refusal.cpp)",
    "target_link_libraries(test-moe-pp-staging-refusal PRIVATE ggml-sycl-private-fixtures)",
    "add_test(NAME test-moe-pp-staging-refusal COMMAND test-moe-pp-staging-refusal)",
    "set_tests_properties(test-moe-pp-staging-refusal PROPERTIES LABELS \"sycl;moe;device;B50;model\" "
    "SKIP_RETURN_CODE 77 RUN_SERIAL TRUE",
)


def mmid_of(sycl: str) -> str:
    return body(norm(sycl), MMID_SIG)


def mutate_in(sycl: str, signature: str, old: str, new: str) -> str:
    """The normalized text with `old` replaced once inside the first body opened by `signature` (inside
    ggml_sycl_mul_mat_id), for a mutant whose text also appears elsewhere in the file."""
    n = norm(sycl)
    region = body(body(n, MMID_SIG), signature)
    assert old in region, f"mutant precondition: {old!r} is not in the body of {signature!r}"
    return n.replace(region, region.replace(old, new, 1), 1)


# ---- the helper -----------------------------------------------------------------------------------------------------


def claim_helper(sycl: str) -> bool:
    """One [[noreturn]] helper: a WARN naming tensor, buffer, rows, bytes and device, then the fallback throw."""
    n = norm(sycl)
    h = body(n, HELPER_SIG)
    return (n.count(HELPER_SIG) == 1 and h != "" and ordered(h, HELPER_WARN, HELPER_THROW)
            and "GGML_ABORT" not in h and "GGML_LOG_INFO" not in h)


def test_helper():
    assert claim_helper(SYCL), first_missing(SYCL_REL, body(norm(SYCL), HELPER_SIG), HELPER_WARN, HELPER_THROW,
                                             signature=HELPER_SIG)


def test_helper_mutants():
    assert not claim_helper(mutate(SYCL, "bytes, device); " + HELPER_THROW, 'bytes, device); GGML_ABORT("%s", reason);'))
    assert not claim_helper(mutate(SYCL, "GGML_LOG_WARN(\"[MOE-PP-STAGING]", "GGML_LOG_INFO(\"[MOE-PP-STAGING]"))
    assert not claim_helper(mutate(SYCL, "rows=%zu bytes=%zu", "rows=%zu"))
    assert not claim_helper(mutate(SYCL, HELPER_SIG, "static void ggml_sycl_moe_pp_refuse_staging("))


# ---- each site refuses ------------------------------------------------------------------------------------------------


def claim_sites(sycl: str) -> bool:
    """Every staging site of the prompt path refuses through the helper, the F32 sites through one lambda."""
    m = mmid_of(sycl)
    island = body(m, ISLAND_SIG)
    at = island.find(ISLAND_CHECK)
    tail = island[at:island.find(ISLAND_AFTER, at)] if at >= 0 else ""
    return (m.count(ROW_MAP_SITE) == 1 and m.count(DST_SITE) == 1 and m.count(SRC1_FAILPOINT) == 1
            and m.count(SRC1_REFUSE_LAMBDA) == 1 and m.count(SRC1_SETUP) == 1 and m.count(SRC1_EXPERT) == 1
            and m.count(SRC1_BATCHES) == 1 and m.count("refuse_src1_f32_staging();") == 3
            and ordered(m, SRC1_REFUSE_LAMBDA, SRC1_SETUP, SRC1_EXPERT, SRC1_BATCHES)
            and tail != "" and REFUSE in tail and all(b in tail for b in ISLAND_BUFFERS) and "GGML_ABORT" not in tail)


def test_sites():
    assert claim_sites(SYCL), first_missing(SYCL_REL + ": body of ggml_sycl_mul_mat_id", mmid_of(SYCL), ROW_MAP_SITE,
                                            DST_SITE, SRC1_FAILPOINT, SRC1_REFUSE_LAMBDA, SRC1_SETUP, SRC1_EXPERT,
                                            SRC1_BATCHES, ISLAND_CHECK, *ISLAND_BUFFERS, signature=MMID_SIG)


def test_sites_mutants():
    c = claim_sites
    # The row-map site aborts again, or reads past a null buffer.
    assert not c(mutate(SYCL, "if (!dev_row_mappings.get()) { ggml_sycl_moe_pp_refuse_staging(",
                        "if (!dev_row_mappings.get()) { GGML_ABORT(\"x\"); ggml_sycl_moe_pp_refuse_staging_x("))
    # The output staging is used with no check, as it was.
    assert not c(mutate(SYCL, "if (!dst_contiguous_ptr) { ggml_sycl_moe_pp_refuse_staging(",
                        "if (false) { ggml_sycl_moe_pp_refuse_staging("))
    # One F32 site aborts instead of refusing.
    assert not c(mutate(SYCL, SRC1_EXPERT, "if (!expert_use_mxfp4_soa_f16 && !copy_src1_current_f32()) { "
                                           "GGML_ABORT(\"x\"); }"))
    assert not c(mutate(SYCL, SRC1_BATCHES, "auto dispatch_f32_batches = [&]() { if (!copy_src1_current_f32()) { "
                                            "GGML_ABORT(\"x\"); }"))
    assert not c(mutate(SYCL, SRC1_SETUP, "if (!pp_mxfp4_soa_f16_ready && !ensure_src1_contiguous_f32()) { "
                                          "GGML_ABORT(\"x\"); }"))
    # The island aborts again.
    assert not c(mutate(SYCL, ISLAND_CHECK + " const bool act_missing", ISLAND_CHECK + " GGML_ABORT(\"x\"); "
                                                                                       "const bool act_missing"))


# ---- the unwind ------------------------------------------------------------------------------------------------------


def try_of(m: str) -> tuple[str, str]:
    """The body of the try that covers the per-expert path, and the text that follows it."""
    t = body(m, TRY_SIG)
    at = m.find(t) if t else -1
    return (t, m[at + len(t):]) if at >= 0 else ("", "")


def claim_unwind(sycl: str) -> bool:
    """The per-expert path's exit work is one lambda, run at most once, declared after the scratch claim and the expert
    leases it hands on. One try covers that path from the scratch setup to the normal exit, and its handler runs the
    exit work and rethrows; the normal exit and the handler are the lambda's only callers."""
    m = mmid_of(sycl)
    r = body(m, RETAIN_SIG)
    t, after = try_of(m)
    return (m.count(RETAIN_SIG) == 1 and r != "" and ordered(r, *RETAIN_STEPS)
            and m.count(RETAIN_ONCE) == 1 and m.count("pp_exit_work_done = true;") == 1
            and m.count(LEASES_DECL) == 1 and m.count("std::move(pp_gpu_leases)") == 1
            and m.count(RETAIN_STEPS[2]) == 1 and r.count(RETAIN_STEPS[5]) == 1
            and m.count(TRY_SIG) == 1 and ordered(m, ROW_MAP_SITE, CLAIM_DECL, LEASES_DECL, RETAIN_ONCE, TRY_SIG)
            and t != "" and ROW_MAP_SITE not in t and after.startswith(HANDLER)
            and ordered(t, SCRATCH_CLAIM, DST_SITE, SRC1_SETUP, LOOP_HEAD, SECONDARY_THROW, UNRESOLVED_THROW,
                        SRC1_EXPERT, SRC1_BATCHES, ISLAND_CALL, NORMAL_EXIT)
            and m.count(NORMAL_EXIT) == 1 and m.count(RETAIN_CALL) == 2)


def test_unwind():
    m = mmid_of(SYCL)
    t, _ = try_of(m)
    assert claim_unwind(SYCL), first_missing(SYCL_REL + ": body of ggml_sycl_mul_mat_id", m, CLAIM_DECL, LEASES_DECL,
                                             RETAIN_ONCE, *RETAIN_STEPS, TRY_SIG, HANDLER.strip(), SCRATCH_CLAIM,
                                             DST_SITE, SRC1_SETUP, LOOP_HEAD, SECONDARY_THROW, UNRESOLVED_THROW,
                                             SRC1_EXPERT, SRC1_BATCHES, ISLAND_CALL, NORMAL_EXIT, signature=MMID_SIG)


def test_unwind_mutants():
    c = claim_unwind
    # The normal exit stops running the shared exit work.
    assert not c(mutate(SYCL, NORMAL_EXIT, "pp_phase_log(\"pp-retain-submitted\","))
    # The handler stops running it, or swallows the refusal instead of rethrowing it.
    assert not c(mutate(SYCL, HANDLER, " catch (const ggml_sycl_fallback_error &) { throw; }"))
    assert not c(mutate(SYCL, HANDLER, " catch (const ggml_sycl_fallback_error &) { retain_pp_dispatch_until_done(); }"))
    # The exit work can run twice: its once-flag is not checked.
    assert not c(mutate(SYCL, "if (pp_exit_work_done) { return; } ", ""))
    # A refusal runs the exit work itself again instead of only refusing.
    assert not c(mutate(SYCL, SRC1_REFUSE_LAMBDA, SRC1_REFUSE_LAMBDA.replace("{ ", "{ " + RETAIN_CALL + " ", 1)))
    # A fallback throw inside the loop no longer reaches the handler: the try closes before the loop, so both of the
    # loop's existing throws (the refused secondary executor, the unresolved expert) unwind past it.
    assert not c(mutate(SYCL, LOOP_HEAD, "} catch (const ggml_sycl_fallback_error &) { throw; } { " + LOOP_HEAD))
    # One of those throws moved out of the try's reach.
    assert not c(mutate(SYCL, UNRESOLVED_THROW, "").replace(HANDLER, HANDLER + " " + UNRESOLVED_THROW, 1))
    assert not c(mutate(SYCL, SECONDARY_THROW, "").replace(HANDLER, HANDLER + " " + SECONDARY_THROW, 1))
    # The scratch slot is claimed before the try opens, where a refusal would leave it unrecorded.
    assert not c(mutate(SYCL, SCRATCH_CLAIM, "false").replace(
        "auto refuse_src1_f32_staging = [&]() {", SCRATCH_CLAIM + "; auto refuse_src1_f32_staging = [&]() {", 1))
    # The island's call sits outside the try.
    assert not c(mutate(SYCL, ISLAND_CALL, "pp_phase_log(\"pp-host-dispatch-done\",")
                 .replace(HANDLER, HANDLER + " dispatch_host_expert_groups(pp_host_groups);", 1))
    # The lambda forgets a step the normal exit performs.
    assert not c(mutate_in(SYCL, RETAIN_SIG, RETAIN_STEPS[5], "(void) 0;"))
    assert not c(mutate_in(SYCL, RETAIN_SIG, RETAIN_STEPS[2], "(void) 0;"))
    assert not c(mutate_in(SYCL, RETAIN_SIG, RETAIN_STEPS[3], "(void) (std::move(pp_gpu_leases),"))
    # The expert leases declared after the try opens, where the handler cannot reach them.
    m = mutate(SYCL, LEASES_DECL, "")
    m = m.replace("size_t pp_route_dev0 = 0;", LEASES_DECL + " size_t pp_route_dev0 = 0;", 1)
    assert not c(m)


def claim_no_old_aborts(sycl: str) -> bool:
    """None of the abort texts the staging sites carried is left anywhere in the backend file."""
    n = norm(sycl)
    return n != "" and not any(t in n for t in OLD_ABORTS)


def test_no_old_aborts():
    n = norm(SYCL)
    assert claim_no_old_aborts(SYCL), [t for t in OLD_ABORTS if t in n]


def test_no_old_aborts_mutants():
    for t in OLD_ABORTS:
        assert not claim_no_old_aborts(SYCL + '\nvoid f() { GGML_ABORT("' + t + '"); }\n')


# ---- the failpoints and the device test -------------------------------------------------------------------------------


def claim_failpoints(sycl: str, seam: str) -> bool:
    """The failpoint macro reaches the seam only under GGML_SYCL_Q1_NVFP4_ROUTE_TESTING and is false otherwise; each
    failure is named in the seam header and armed at exactly one site."""
    n = norm(sycl)
    s = norm(seam)
    return (n.count(FAILPOINT_MACRO) == 1 and all(s.count(f + ",") == 1 for f in FAILURES)
            and all(n.count("GGML_SYCL_MOE_PP_STAGING_FAILPOINT(" + f + ")") == 1 for f in FAILURES))


def test_failpoints():
    assert claim_failpoints(SYCL, SEAM), first_missing(SYCL_REL + " and " + SEAM_REL, norm(SYCL) + norm(SEAM),
                                                       FAILPOINT_MACRO, *FAILURES)


def test_failpoints_mutants():
    c = claim_failpoints
    # The production branch consumes the seam too.
    assert not c(mutate(SYCL, "GGML_SYCL_MOE_PP_STAGING_FAILPOINT(failure) false",
                        "GGML_SYCL_MOE_PP_STAGING_FAILPOINT(failure) ggml_sycl_q1_nvfp4_test_consume_failure(failure)"),
                 SEAM)
    # A failure the seam header does not name.
    assert not c(SYCL, mutate(SEAM, FAILURES[1] + ",", ""))
    # A failpoint that is never placed.
    assert not c(mutate(SYCL, "GGML_SYCL_MOE_PP_STAGING_FAILPOINT(" + FAILURES[2] + ")", "false"), SEAM)


def claim_device_test(cmake: str, test: str) -> bool:
    """The device test is registered as the Q1/NVFP4 refusal test is, against the private fixtures with the `model`
    label, and arms every failpoint."""
    c = " ".join(cmake.split())
    t = norm(test)
    return (ordered(c, *REG_STEPS) and c.count("add_test(NAME " + TEST_NAME + " ") == 1
            and all(f in t for f in FAILURES) and "ggml_sycl_q1_nvfp4_test_failure_once(" in t)


def test_device_test():
    c = " ".join(CMAKE.split())
    assert claim_device_test(CMAKE, TEST), first_missing(CMAKE_REL + " and " + TEST_REL, c + norm(TEST), *REG_STEPS,
                                                         *FAILURES)


def test_device_test_mutants():
    c = claim_device_test
    flat = " ".join(CMAKE.split())
    assert not c(flat.replace("LABELS \"sycl;moe;device;B50;model\" SKIP_RETURN_CODE 77 RUN_SERIAL TRUE",
                              "LABELS \"sycl;moe;device;B50\" SKIP_RETURN_CODE 77 RUN_SERIAL TRUE"), TEST)
    assert not c(flat.replace("target_link_libraries(test-moe-pp-staging-refusal PRIVATE ggml-sycl-private-fixtures)",
                              "target_link_libraries(test-moe-pp-staging-refusal PRIVATE ggml-sycl)"), TEST)
    assert not c(CMAKE, TEST.replace(FAILURES[0], "GGML_SYCL_Q1_NVFP4_TEST_FAILURE_NONE"))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
