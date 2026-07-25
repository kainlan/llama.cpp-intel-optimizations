# AGENTS.md

**`CLAUDE.md` is the source of truth for this repository.** Read it first. This
file holds only the material that is *not* there.

This file used to mirror ~70% of `CLAUDE.md` — build commands, architecture,
verification gates, device selection, performance tables, environment variables.
That duplication was removed on 2026-07-25 because it had actively caused harm:
one copy was corrected for the B580→B70 hardware change on 2026-07-24 and this
one was not, so for five weeks it described a GPU that was not installed and
carried a GPT-OSS throughput row of `~66 PP512 / ~17 TG128` against an actual
~1400 / ~44 — an error of 21× that a reader had no way to spot from context.

Duplicated numbers rot silently, because two documents cannot disagree loudly.

## Where things live now

| You want | Read |
|---|---|
| Build, test, format commands | `CLAUDE.md` → *Build Commands* |
| Directory layout, key binaries, SYCL backend files | `CLAUDE.md` → *Project Architecture* |
| Memory-ownership contract (unified cache, `mem_handle`) | `CLAUDE.md` → *SYCL Memory Ownership*; enforceable version in `docs/design/sycl-canonical-memory-architecture.md` |
| ggml naming/matmul conventions | `CLAUDE.md` → *ggml Conventions* |
| Model paths, correctness gates, `llama-bench` traps | `CLAUDE.md` → *Verification Commands & Correctness Gates* |
| GPT-OSS prompt/template rules, sources | `CLAUDE.md` → *GPT-OSS Prompt Template Rule*; full rationale and the seven checked sources in `docs/backend/gpt-oss-testing.md` |
| Device selection, PCI/DRM mapping, hang hazards | `CLAUDE.md` → *SYCL Device Selection* |
| Patched compute-runtime, P2P topology | `CLAUDE.md`; install/rollback detail in `docs/backend/compute-runtime.md` |
| Throughput figures and regression guardrails | `docs/backend/sycl-perf-baselines.md` (the numeric gate), summarised in `CLAUDE.md` |
| Environment variables | `docs/backend/sycl-env-vars.md` (240+ vars), load-bearing subset in `CLAUDE.md` |
| Hard-won rules (safety, workflow, architecture) | `CLAUDE.md` → *Hard-Won Rules* |
| Live debugging state, bisects, regression hunts | the codescout task tracker (`task_list`, `task_show`) — **not a document** |

## Intel Arc GPU Memory Architecture (Critical for Multi-GPU)

On Intel Arc discrete GPUs (Xe architecture), there are two GPU address
translation tables:

- **GGTT** (Global Graphics Translation Table): 32-bit, 4 GB address space.
  Reserved for **kernel/privileged** resources only (GuC firmware, display
  engine).  User-space USM allocations do **NOT** consume GGTT aperture.

- **PPGTT** (Per-Process GTT): 48-bit, 256 TB address space per process.
  This is where **all user-space allocations** (device, host, shared USM)
  are mapped.  The PPGTT is effectively unlimited for practical purposes.

**Implication**: USM host memory (`sycl::malloc_host` / `zeMemAllocHost`) is
mapped through the PPGTT, not the GGTT.  There is no GGTT aperture constraint
on pinned host memory, even in multi-GPU setups.  Do NOT cap pinned host
memory budgets based on GGTT size estimates — the previous code that did this
reduced pinned budgets from 128 GB to ~1.9 GB in multi-GPU, preventing large
model support for no valid reason.

The previous multi-GPU hang was caused by a `ggml_sycl_info()` re-entry
deadlock (calling a function with a C++ static-init guard from within the
function that fills that static), NOT by GGTT overflow.

Source: Intel GPU PRM Vol06 "Memory Views"; Level Zero Core Spec §Memory;
Xe kernel driver documentation (`Documentation/gpu/xe/xe_mm.rst`).

## Profiling

VTune GPU offload example:

```bash
source /opt/intel/oneapi/setvars.sh --force
GGML_SYCL_DEVICE=0 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
vtune -collect gpu-offload -knob enable-stack-collection=true \
  -result-dir /tmp/vtune_llama \
  -- ./build/bin/llama-bench \
    -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 64 -n 8 --tg-batch 4 -ngl 99 -fa 1
```

If VTune only shows memcpy tasks, use PTI and UR tracers for kernel time and
launch shapes. **The full tracer setup is not currently documented anywhere** —
this section previously pointed at `CLAUDE.md` for it, which has never contained
it. If you work it out, write it down here.

Profiling runs are throughput measurements: check for competing load first
(`uptime`, `pgrep -af 'codescout|ninja|icpx|ffmpeg'`), and remember `llama-bench`
discards all `GGML_LOG_INFO` without `-v`.

## Professional Engineering Standards

Spinach Rule: when you detect a visible flaw the user may not see, correction is
mandatory. Do not optimize for agreement.

- Challenge wrong assumptions directly
- Question unclear requirements before implementing risky changes
- Identify performance and security trade-offs
- Never fake progress or certainty

This is not decorative. Every significant defect found in the path-scoped zone
sizing work (2026-07-25) surfaced because an implementer declined to execute an
instruction that did not match the code — a plan that named the wrong sizing
site, a guard that did not exist, a formula that was arithmetically wrong. Each
pushed back with evidence rather than preference, which is what made the
objection actionable instead of a negotiation.

## Landing the Plane (Session Completion)

When ending a work session, complete all steps below.

1. File issues for remaining work in the codescout tracker (`task_create`)
2. Run quality gates if code changed
3. Update task status — reset any `in_progress` task you are no longer working
   on back to `open`; `in_progress` is a lease, not a label
4. Push to remote:

```bash
git pull --rebase
git push
git status
```

5. Clean up stale stashes or branches where appropriate
6. Verify all intended changes are committed
7. Hand off remaining context

Critical rules:

- Do not stop with work uncommitted
- Do not say "ready to push when you are" — either push, or state plainly that
  you are leaving it unpushed and why
- If push fails, resolve and retry

> **Note on pushing:** the older form of this checklist said "work is not
> complete until `git push` succeeds" and included a `bd sync` step. `bd` (beads)
> was retired 2026-07-01 in favour of the codescout tracker and the step is gone.
> Whether to push is the repository owner's call — confirm rather than assuming,
> particularly on a long-lived feature branch.
