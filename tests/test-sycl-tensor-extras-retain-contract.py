#!/usr/bin/env python3
"""Every ctx->tensor_extras entry must own a reference (llama.cpp-yy17).

THE CONTRACT.  ~ggml_backend_sycl_buffer_context (and the non-COMPUTE
buffer_reset path) call release_extra_gpu() once per entry in tensor_extras.  An
entry is therefore an owning reference, and a push_back that does not carry one
is a second claimant on somebody else's refcount.

That is not hypothetical accounting.  The site this gate was written for pushed
an extra produced by ggml_backend_sycl_register_host_weight_tensor, whose
registry row holds that extra's ONLY reference; ~llama_model then ran both
releases in a fixed order -- pimpl.reset() frees the buffers (refcount 1 -> 0,
delete), then hooks.unload() -> ggml_sycl_release_host_weight_extras_for_owner()
decrements the freed object.

WHY A SOURCE GATE AND NOT A RUNTIME ONE.  The failure is a decrement of freed
memory.  glibc writes a tcache `next` pointer over offset 0, which is where
`std::atomic<int> refcount` lives, so the second release usually reads back a
recycled non-zero value, takes the `prev > 1` early return, and the process
carries on -- the defect's ordinary appearance is a clean run.  Only when that
word happens to be NULL does GGML_ASSERT(prev > 0) fire.  A runtime gate would
therefore be a coin flip and a green one would prove nothing.  The structural
property -- "this push_back is not covered by a reference" -- is exact.

WHAT IT CHECKS, PER PUSH SITE.  Comments and string literals are blanked first
(preserving offsets) so brace counting cannot be thrown by a brace inside either.
For each `tensor_extras.push_back({ t, e })`, find the INNERMOST brace block
containing it -- branch scoping is the whole difficulty here, because the
compliant sites push from inside an `else { extra = new ...; push }` while a
sibling `if` branch a few lines above reuses an existing extra.  Then, within
that block only:

  * assignments to `e` from `new ggml_tensor_extra_gpu` are creations: the push
    inherits the fresh object's initial refcount of 1 -- OK;
  * an assignment to `e` from an existing extra (anything reading `->extra`)
    means a reusing path reaches this push, and a `retain_extra_gpu(e)` must
    appear between that assignment and the push.

NOT VACUOUS.  A pattern that cannot match returns clean forever, so the gate
fails rather than passes unless it found (a) several push sites and (b) at least
one reusing site.  If the reuse site is ever deleted outright the gate says so,
because at that point it is no longer testing anything.

THE ORIGINAL RED, already observed.  Run against the parent commit of the fix --
that is, the unmodified pre-yy17 tree -- this gate fails with

    ggml-sycl.cpp:21143: tensor_extras.push_back of `extra` is reachable with an
    extra that already had an owner, and no retain_extra_gpu(extra) covers it

which is the ticket's line, named without being told about it.  Reproduce with:

    git stash && python3 tests/test-sycl-tensor-extras-retain-contract.py; git stash pop
    #   -- or, once the fix is committed:
    git -C . show <fix>^:ggml/src/ggml-sycl/ggml-sycl.cpp > /tmp/pre.cpp   # inspect

POSITIVE CONTROL (re-derives that RED from the current tree):

    cd <worktree>
    python3 tests/test-sycl-tensor-extras-retain-contract.py        # PASS
    python3 - <<'EOF'
    import pathlib
    p = pathlib.Path("ggml/src/ggml-sycl/ggml-sycl.cpp")
    s = p.read_text()
    old = "        if (!created_extra) {\n            retain_extra_gpu(extra);\n"
    assert s.count(old) == 1, "mutation target missing -- do not trust the control"
    p.write_text(s.replace(old, "        if (!created_extra) {\n", 1))
    EOF
    python3 tests/test-sycl-tensor-extras-retain-contract.py        # must FAIL, exit 1
    git checkout -- ggml/src/ggml-sycl/ggml-sycl.cpp   # ONLY once the fix is committed

The `assert s.count(old) == 1` is deliberate: a mutation that silently misses its
target reads exactly like a gate that did not fire.  The `git checkout` caveat is
equally deliberate -- run against an uncommitted fix it discards the fix rather
than the mutation, and the follow-up run then looks like a gate that stayed red.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"

raw = SRC.read_text()


def blank_comments_and_strings(s):
    """Replace comment and string-literal bodies with spaces, keeping offsets.

    Newlines are preserved so line numbers computed on the result still match the
    original file.
    """
    out = list(s)
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == "/" and i + 1 < n and s[i + 1] == "/":
            while i < n and s[i] != "\n":
                out[i] = " "
                i += 1
        elif c == "/" and i + 1 < n and s[i + 1] == "*":
            out[i] = out[i + 1] = " "
            i += 2
            while i < n and not (s[i] == "*" and i + 1 < n and s[i + 1] == "/"):
                if s[i] != "\n":
                    out[i] = " "
                i += 1
            if i < n:
                out[i] = " "
                if i + 1 < n:
                    out[i + 1] = " "
                i += 2
        elif c in "\"'":
            quote = c
            i += 1
            while i < n and s[i] != quote:
                if s[i] == "\\":
                    out[i] = " "
                    i += 1
                    if i < n and s[i] != "\n":
                        out[i] = " "
                elif s[i] != "\n":
                    out[i] = " "
                i += 1
            i += 1
        else:
            i += 1
    return "".join(out)


text = blank_comments_and_strings(raw)

PUSH = re.compile(r"tensor_extras\.push_back\(\s*\{\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*\}")
NEW_EXTRA = "new ggml_tensor_extra_gpu"
READS_AN_EXTRA = re.compile(r"->\s*extra\b")


def innermost_block_start(pos):
    """Offset just past the `{` opening the innermost block containing pos."""
    depth = 0
    i = pos - 1
    while i >= 0:
        c = text[i]
        if c == "}":
            depth += 1
        elif c == "{":
            if depth == 0:
                return i + 1
            depth -= 1
        i -= 1
    return 0


def line_of(pos):
    return raw.count("\n", 0, pos) + 1


failures = []
push_sites = 0
reuse_sites = 0

for m in PUSH.finditer(text):
    push_sites += 1
    ident = m.group(2)
    block = text[innermost_block_start(m.start()):m.start()]

    # Assignments to `ident` itself.  The lookbehind keeps `tensor->extra = extra;`
    # from being read as an assignment to `extra`.
    assign = re.compile(r"(?<![\w.>])" + re.escape(ident) + r"\s*=\s*([^;]*);", re.S)
    assignments = list(assign.finditer(block))

    if not assignments:
        failures.append(
            f"{SRC.name}:{line_of(m.start())}: `{ident}` is pushed but never assigned inside its "
            f"enclosing block -- this gate cannot tell whether the entry owns a reference"
        )
        continue

    reusing = [a for a in assignments if READS_AN_EXTRA.search(a.group(1))]
    if not reusing:
        continue  # every path into this push created the extra; it owns that count

    reuse_sites += 1
    tail = block[reusing[-1].end():]
    if not re.search(r"retain_extra_gpu\(\s*" + re.escape(ident) + r"\s*\)", tail):
        failures.append(
            f"{SRC.name}:{line_of(m.start())}: tensor_extras.push_back of `{ident}` is reachable with "
            f"an extra that already had an owner, and no retain_extra_gpu({ident}) covers it -- the "
            f"buffer context's release would be a second claimant on one refcount"
        )

if push_sites < 5:
    print(
        f"tensor_extras retain contract: INCONCLUSIVE -- found only {push_sites} push_back sites in "
        f"{SRC}; the pattern no longer matches the code it is meant to check",
        file=sys.stderr,
    )
    raise SystemExit(1)

if reuse_sites < 1:
    print(
        "tensor_extras retain contract: INCONCLUSIVE -- no push_back site reuses an existing extra, so "
        "nothing exercised the rule this gate exists for (llama.cpp-yy17's site was the only one; if it "
        "was removed on purpose, retire this gate rather than leaving it green)",
        file=sys.stderr,
    )
    raise SystemExit(1)

if failures:
    print("tensor_extras retain contract FAILED:", file=sys.stderr)
    for f in failures:
        print("  " + f, file=sys.stderr)
    raise SystemExit(1)

print(
    f"tensor_extras retain contract: PASS ({push_sites} push sites, {reuse_sites} reachable with an "
    f"existing extra, all covered by retain_extra_gpu)"
)
