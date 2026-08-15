#!/usr/bin/env python3
"""Compile-failure gate: cache_backing_token must be unforgeable BY THE COMPILER.

CACHE_BACKING is an authority -- shutdown exempts that ownership class from
destructive-teardown refusal -- and allocation-provenance.hpp claims the token
carrying it is "enforced by the compiler". A running test cannot witness a
compile-time guarantee, so this driver compiles probe translation units and
requires the forgery arms to be REJECTED.

That claim was false when this gate was written (llama.cpp-fhqe): at C++17
`cache_backing_token() = default;` on the first declaration is user-DECLARED but
not user-PROVIDED, so the class was an aggregate and `cache_backing_token{}` was
aggregate initialization -- which never consults the private constructor. Only
C++20 excludes user-DECLARED constructors from aggregates, and this repo is
pinned at C++17 (ggml/CMakeLists.txt: CMAKE_CXX_STANDARD 17). The fix is to make
the constructor user-provided (`{}` rather than `= default`); this gate is what
keeps it that way, because no source-text check can see the difference.

The probe TUs are generated into a system temp directory rather than committed.
They must `#include "allocation-provenance.hpp"`, and
tests/test-sycl-owner-allocation-migration.py restricts that include to
unified-cache.cpp and pinned-pool.cpp -- a committed probe would break the very
contract it is here to defend. That gate scans only .c/.cpp/.h/.hpp under the
repo root, so holding the probe text in this .py, outside the tree, satisfies
both.

Arms, in order (the control runs FIRST -- a control that ran after the arms it
guards could only explain a vacuous result, never prevent one):

  control  includes the header and nothing else       -> must COMPILE
  forge    `cache_backing_token{}`                    -> must be REJECTED
  derive   aggregate-init through a derived class     -> must be REJECTED

Both rejections must also NAME cache_backing_token in the diagnostic. Without
that, a typo or a broken include path would reject the arm for an unrelated
reason and the gate would pass while proving nothing.

Usage: test-cache-backing-forgery-compile.py <compiler> [compile args...]
Exit:  0 pass, 1 failure, 77 skip (toolchain unusable -- NOT a pass).
"""

import os
import shutil
import subprocess
import sys
import tempfile

SKIP = 77

TOKEN = "cache_backing_token"

PROBES = {
    # Must compile. Proves the compiler, flags and include paths are usable, so
    # that a rejection below is attributable to access control rather than to a
    # misconfigured command line.
    "control": """
#include "allocation-provenance.hpp"
ggml_sycl::allocation_result legitimate(const ggml_sycl::alloc_request & req,
                                        ggml_sycl::cache_backing_token   token) {
    return ggml_sycl::unified_allocate_owner_backing(req, token);
}
""",
    # Direct forgery: mint a token from a translation unit that is not the friend.
    "forge": """
#include "allocation-provenance.hpp"
ggml_sycl::allocation_result forge(const ggml_sycl::alloc_request & req) {
    return ggml_sycl::unified_allocate_owner_backing(req, ggml_sycl::cache_backing_token{});
}
""",
    # Derived-class escape: aggregate-initialize the base SUBOBJECT instead of
    # the token directly. A second mechanism, NOT a strictly stronger arm -- an
    # earlier draft of this comment claimed it would survive a partial fix that
    # the direct arm missed, and measuring it disproved that. Both candidate
    # repairs reject both arms: `token() {}` (user-provided), and keeping
    # `= default` while adding a private data member. It earns its place by
    # exercising a different initialization path, which matters where the two
    # already diverge -- the direct escape exists only at C++17 and earlier.
    "derive": """
#include "allocation-provenance.hpp"
namespace {
struct evil_minter : ggml_sycl::cache_backing_token {
    evil_minter() = default;
};
}
ggml_sycl::allocation_result forge(const ggml_sycl::alloc_request & req) {
    return ggml_sycl::unified_allocate_owner_backing(req, evil_minter{});
}
""",
}


def compile_probe(compiler, args, workdir, name):
    source = os.path.join(workdir, "probe-%s.cpp" % name)
    with open(source, "w") as handle:
        handle.write(PROBES[name])
    completed = subprocess.run(
        [compiler] + args + ["-fsyntax-only", source],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.returncode, completed.stdout


def main():
    if len(sys.argv) < 2:
        print("usage: %s <compiler> [compile args...]" % sys.argv[0], file=sys.stderr)
        return 1
    compiler, args = sys.argv[1], sys.argv[2:]

    workdir = tempfile.mkdtemp(prefix="cache-backing-forgery-")
    try:
        try:
            control_rc, control_output = compile_probe(compiler, args, workdir, "control")
        except OSError as error:
            print(
                "SKIP: cannot run the compiler %r (%s); this run proves NOTHING about token forgery.\n"
                "      Source oneAPI and re-run to actually test it:\n"
                "      (source /opt/intel/oneapi/setvars.sh --force)" % (compiler, error),
                file=sys.stderr,
            )
            return SKIP

        if control_rc != 0:
            # Not a pass and not a failure of the invariant: nothing here can be
            # compiled at all, so the rejection arms below would be rejected for
            # the wrong reason. Report it as a skip, loudly.
            print(
                "SKIP: the control probe does not compile, so this run proves NOTHING about token forgery -- "
                "the rejections below would not be attributable to access control.\n"
                "      Compiler output follows.\n%s" % control_output,
                file=sys.stderr,
            )
            return SKIP
        print("PASS forgery-compile-control-probe-compiles")

        problems = []
        for name in ("forge", "derive"):
            rc, output = compile_probe(compiler, args, workdir, name)
            if rc == 0:
                problems.append(
                    "the %r probe COMPILED: %s can be minted outside its friend, so any translation unit "
                    "can forge CACHE_BACKING -- the ownership class shutdown exempts from teardown refusal"
                    % (name, TOKEN)
                )
                continue
            if TOKEN not in output:
                problems.append(
                    "the %r probe was rejected, but no diagnostic mentions %s, so the rejection is not "
                    "attributable to access control (a typo or a stale include path rejects too). "
                    "Compiler output:\n%s" % (name, TOKEN, output)
                )
                continue
            print("PASS forgery-compile-%s-probe-rejected" % name)

        if problems:
            for problem in problems:
                print("FAIL: %s" % problem, file=sys.stderr)
            return 1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print("PASS cache-backing-token-forgery-compile-gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
