#!/usr/bin/env python3
"""MMVQ sub-group-per-row launch geometry source contract (llama.cpp-99ke).

Host-only: reads sources, runs no build, loads no model and touches no device.

The companion executable test (test-mmvq-launch-geometry) proves the padding
ARITHMETIC against row counts it constructs. It cannot see whether PRODUCTION
uses that arithmetic, which is what this gate covers: that every one of the
`num_subgroups = 16` launch sites in mmvq.cpp pads its global range, that none
of them still passes the raw row count, and that the kernel each one launches
opens with a `row >= nrows` early return derived the same way the padding
assumes -- `group_id * subgroup_range + subgroup_id`.

The pairing is the point. Padding without a guard computes garbage rows past the
end of `dst`; a guard without padding is the bug this fixes, because the
dispatcher throws "Non-uniform work-groups are not supported by the target
device" before any kernel body runs. Either half alone looks correct in review.

Every check below fails against the pre-99ke tree. Run it that way to confirm:

    ./tests/test-sycl-mmvq-launch-geometry-contract.py \\
        --mmvq <(git show f46129c5b:ggml/src/ggml-sycl/mmvq.cpp)

The `anchors` section proves every region the checks read was actually found, so
a renamed dispatch reports a missing anchor instead of passing vacuously. Run
with --self-test to prove the checks fire against mutants.
"""
import argparse
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--mmvq", default=str(root / "ggml/src/ggml-sycl/mmvq.cpp"))
parser.add_argument("--header", default=str(root / "ggml/src/ggml-sycl/mmvq-launch-geometry.hpp"))
parser.add_argument("--self-test", action="store_true",
                    help="prove the checks fire when the construct they require is removed")
args = parser.parse_args()

# The launch sites this gate governs, identified exactly as the ticket did.
SITE_ANCHOR = "constexpr size_t num_subgroups = 16;"

# The ticket's own inventory. Pinned as a number so a NEW site added later --
# the way all eleven of these accumulated -- fails the gate instead of quietly
# inheriting the defect.
EXPECTED_SITES = 11


def strip_comments(source):
    """Remove C and C++ comments, preserving string literals and line count.

    Every check below reads active code only. Without this, "no site still uses
    the raw row count" is spoofed by the comment that explains why none does --
    which the header's own rationale would do if it were pasted here.
    """
    out = []
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        if ch in ('"', "'"):
            quote = ch
            out.append(ch)
            i += 1
            while i < n:
                out.append(source[i])
                if source[i] == "\\":
                    if i + 1 < n:
                        out.append(source[i + 1])
                        i += 2
                        continue
                elif source[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        if source.startswith("//", i):
            while i < n and source[i] != "\n":
                i += 1
            continue
        if source.startswith("/*", i):
            end = source.find("*/", i + 2)
            end = n if end < 0 else end + 2
            out.append("\n" * source.count("\n", i, end))
            i = end
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def squeeze(source):
    """Collapse runs of spaces/tabs to one, preserving line structure.

    Every needle below is written with single spaces so clang-format's column
    alignment cannot break a check by padding an assignment.
    """
    return re.sub(r"[ \t]+", " ", source)


def ordered(text, *needles):
    """True when every needle appears, in the given order."""
    at = -1
    for needle in needles:
        found = text.find(needle, at + 1)
        if found < 0:
            return False
        at = found
    return True


def launch_sites(mmvq):
    """One text region per launch site: the anchor through its submitted lambda.

    Bounded by the closing of the `parallel_for` rather than a fixed line count,
    so a site that grows a `depends_on` clause -- the persistent-TG wrappers do
    -- still yields its whole geometry and its kernel call. The terminator is
    matched with leading whitespace allowed: these are indented, and a
    column-anchored `});` would run every region on to the next one that
    happened to sit at column zero, silently merging all eleven.
    """
    sites = []
    for match in re.finditer(re.escape(SITE_ANCHOR), mmvq):
        end = re.search(r"\n[ \t]*\}\);", mmvq[match.start():])
        stop = match.start() + (end.end() if end else len(mmvq) - match.start())
        sites.append(mmvq[match.start():stop])
    return sites


def kernel_launched_by(site):
    """The kernel function a site's submitted lambda calls, or None.

    Read from inside the lambda rather than from the `parallel_for` type tag:
    the tag is a naming class (`mmvq_reorder_kernel_name<...>`) that says
    nothing about which body actually runs.
    """
    lambda_at = site.find("[=](sycl::nd_item<3>")
    if lambda_at < 0:
        return None
    match = re.search(r"\b(mul_mat_vec_[A-Za-z0-9_]+)\s*[<(]", site[lambda_at:])
    return match.group(1) if match else None


def kernel_body(mmvq, name):
    """A kernel definition's body, from its signature to the next top-level `}`."""
    match = re.search(r"^static void " + re.escape(name) + r"\(", mmvq, re.M)
    if not match:
        return ""
    stop = mmvq.find("\n}\n", match.start())
    return mmvq[match.start():stop] if stop >= 0 else mmvq[match.start():]


def guards_padded_rows(body):
    """True when the body rejects padded rows before it touches anything.

    Three properties, and the derivation is the one that is easy to lose:

    * `row` comes from `group_id * subgroup_range + subgroup_id`, which is what
      makes the padding count right AND makes the early return sub-group-uniform
      -- a per-work-item guard on a differently-derived row would let part of a
      sub-group exit and desynchronise the `reduce_over_group` below it.
    * the guard follows that derivation and returns.
    * nothing indexes memory in between, so a padded row cannot read past the
      end of a slice before being rejected.
    """
    if not ordered(body,
                   "= workgroup_id * sg_range + sg_id;",
                   "if (row >= nrows) {",
                   "return;"):
        return False
    derivation_at = body.find("= workgroup_id * sg_range + sg_id;")
    guard_at = body.find("if (row >= nrows) {", derivation_at)
    return "[" not in body[derivation_at:guard_at]


def evaluate(mmvq, header):
    mmvq, header = squeeze(mmvq), squeeze(header)

    sites = launch_sites(mmvq)
    kernels = [kernel_launched_by(site) for site in sites]
    bodies = {name: kernel_body(mmvq, name) for name in kernels if name}

    anchors = {
        "mmvq.cpp launch sites": sites,
        "launch geometry header": header,
        "kernel bodies reached from the launch sites": bodies and all(bodies.values()),
    }

    checks = {
        # --- the site inventory, so a twelfth site cannot arrive unguarded ---
        "the launch site count still matches the ticket's inventory":
            len(sites) == EXPECTED_SITES,
        "every launch site names the kernel it submits":
            all(kernels) and len(kernels) == EXPECTED_SITES,

        # --- the fix: pad, at every site ---
        "every launch site pads its row count to whole work-groups":
            all("padded_num_y = ggml_sycl::mmvq_pad_rows_to_workgroups(block_num_y, (int) num_subgroups);" in site
                for site in sites),
        "every launch site builds its global range from the padded count":
            all("global_size(1, GGML_SYCL_MMV_Y, padded_num_y * WARP_SIZE)" in site for site in sites),
        "every launch site pads before it builds the range":
            all(ordered(site, "mmvq_pad_rows_to_workgroups(", "sycl::range<3> global_size(") for site in sites),
        # ABSENCE: the defect itself. `ceil_div(nrows, GGML_SYCL_MMV_Y)` is a
        # no-op because GGML_SYCL_MMV_Y is 1, so a site that feeds block_num_y
        # straight to the range requires nrows % 16 == 0 and throws otherwise.
        "no launch site still submits the raw row count":
            not any(re.search(r"global_size\(1, GGML_SYCL_MMV_Y, \(?block_num_y \* WARP_SIZE\)?\)", site)
                    for site in sites),
        "every launch site keeps the work-group size it had":
            all("workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE)" in site for site in sites),

        # --- the other half: guard, in every kernel a site reaches ---
        "every kernel reached from a padded site rejects rows past nrows":
            bool(bodies) and all(guards_padded_rows(body) for body in bodies.values()),

        # --- the header stays host-testable ---
        # ABSENCE: one SYCL include here and test-mmvq-launch-geometry stops
        # building, taking the only executable proof of the arithmetic with it.
        "the geometry header includes no SYCL or backend header":
            not re.search(r'#\s*include\s*[<"](?:sycl/|CL/|.*common\.hpp|ggml)', header),
        "the header rounds up rather than down":
            "(rows + subgroups_per_workgroup - 1) / subgroups_per_workgroup;" in header,
        "the header launches nothing for an empty slice":
            ordered(header, "if (rows <= 0) {", "return 0;"),
    }

    return sorted(name for name, text in anchors.items() if not text), \
           sorted(name for name, ok in checks.items() if not ok), len(checks)


# Positive controls. Each mutation restores one half of the defect, and the
# named check must fail. Without these the absence-based checks pass on any tree
# written after this fix -- which is every tree from here on.
MUTANTS = {
    # Half one: the pre-99ke geometry, restored at a single site. One unpadded
    # site out of eleven is exactly how this would regress.
    "no launch site still submits the raw row count": (
        "mmvq",
        "const int padded_num_y = ggml_sycl::mmvq_pad_rows_to_workgroups(block_num_y, (int) num_subgroups);\n\n"
        " const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, padded_num_y * WARP_SIZE);\n"
        " const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);\n\n"
        " stream->submit([&](sycl::handler & cgh) {\n"
        " cgh.parallel_for<mmvq_reorder_kernel_name<GGML_TYPE_Q4_0>>(",
        "\n const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);\n"
        " const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);\n\n"
        " stream->submit([&](sycl::handler & cgh) {\n"
        " cgh.parallel_for<mmvq_reorder_kernel_name<GGML_TYPE_Q4_0>>("),
    # Half two: the guard removed from the kernel eight of the eleven sites
    # share. Padding without it writes past the end of a row slice.
    "every kernel reached from a padded site rejects rows past nrows": (
        "mmvq",
        " const int row = workgroup_id * sg_range + sg_id;\n\n"
        " if (row >= nrows) {\n"
        " return;\n"
        " }\n\n"
        " const int blocks_per_row = ncols / block_traits::qk;",
        " const int row = workgroup_id * sg_range + sg_id;\n\n"
        " const int blocks_per_row = ncols / block_traits::qk;"),
    # The derivation, which the guard check reads and a reviewer would not.
    # A row taken from the global id counts work-items, not sub-groups, so the
    # padding granularity stops matching and the early return stops being
    # sub-group-uniform -- while `if (row >= nrows)` still sits right there
    # looking correct.
    "every launch site names the kernel it submits": (
        "mmvq",
        " mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>>(\n"
        " vx, vy, dst, ncols, nrows, total_nrows, row_low, nd_item, fused_add, fused_add_ne0, fused_add_nb0,",
        " some_other_kernel<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>>(\n"
        " vx, vy, dst, ncols, nrows, total_nrows, row_low, nd_item, fused_add, fused_add_ne0, fused_add_nb0,"),
    "the geometry header includes no SYCL or backend header": (
        "header",
        "#pragma once",
        "#pragma once\n#include <sycl/sycl.hpp>"),
    "the header rounds up rather than down": (
        "header",
        "const int workgroups = (rows + subgroups_per_workgroup - 1) / subgroups_per_workgroup;",
        "const int workgroups = rows / subgroups_per_workgroup;"),
    "the header launches nothing for an empty slice": (
        "header",
        "if (rows <= 0) {\n return 0;\n }",
        "if (rows < 0) {\n return rows;\n }"),
}


def self_test(mmvq, header):
    """Every check must fail once the construct it requires is removed."""
    problems = []
    for name, (target, anchor, replacement) in MUTANTS.items():
        sources = {"mmvq": squeeze(mmvq), "header": squeeze(header)}
        anchor, replacement = squeeze(anchor), squeeze(replacement)
        if anchor not in sources[target]:
            problems.append("mutation anchor missing for: " + name)
            continue
        sources[target] = sources[target].replace(anchor, replacement, 1)
        missing, failed, _ = evaluate(sources["mmvq"], sources["header"])
        if name not in failed and name not in missing:
            problems.append("check did not fire on its mutant: " + name)
    return problems


def read(path):
    return strip_comments(Path(path).read_text()) if Path(path).exists() else ""


mmvq, header = read(args.mmvq), read(args.header)

missing_anchors, failed, n_checks = evaluate(mmvq, header)

for name in missing_anchors:
    print("MISSING ANCHOR: " + name)
for name in failed:
    print("FAIL: " + name)

if missing_anchors or failed:
    sys.exit(1)

if args.self_test:
    problems = self_test(mmvq, header)
    for problem in problems:
        print("SELF-TEST FAIL: " + problem)
    if problems:
        sys.exit(1)
    print("sycl mmvq launch geometry contract: self-test PASS ({} mutants)".format(len(MUTANTS)))

print("sycl mmvq launch geometry contract: PASS ({} checks)".format(n_checks))
