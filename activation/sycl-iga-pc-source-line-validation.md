# SYCL IGA PC Source-Line Attribution Validation

Date: 2026-07-05
Tracker issue: `llama.cpp-xame`
Scope: Task 5 source/doc scaffold only

## Status

Lead validation is not yet executed in this worktree. This note is a validation report template and status record for the IGA PC static source-line attribution path. It intentionally records only source-only gates until the lead runs the real GPU/profiler/model validation.

Required semantics:

```text
source_line.status asm-line-static-cost means exact static source-line cost from IGA PC rows, not sampled runtime timing.
source_line.status pass remains the only sampled VTune exact status.
```

## Source-only gates

Allowed source-only command:

```bash
bash -n scripts/sycl-source-line-debug-matrix.sh scripts/sycl-vtune-source-line-feasibility.sh scripts/sycl-gptoss-full-attribution-profile.sh scripts/sycl-gptoss-staged-attribution-profile.sh && \
python3 -m pytest tests/test-sycl-iga-pc-disasm-parser.py tests/test-sycl-iga-zebin-extractor.py tests/test-sycl-zebin-asm-source-line-resolver.py tests/test-sycl-vtune-source-line-enablement-docs.py -q
```

Result for this scaffold change:

```text
/home/kainlan/miniconda3/lib/python3.13/site-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3 (2.3.0) or chardet (6.0.0.post1)/charset_normalizer (3.4.7) doesn't match a supported version!
  warnings.warn(
.........................                                                [100%]
25 passed in 0.80s
```

## Lead-only validation template

Do not fill this section from worker source-only runs. The lead-owned run should record:

- Artifact roots for the probe matrix and MXFP4 feasibility run.
- IGA tool path and version/help evidence, if available.
- IGA platform string passed through `--iga-platform` / `SYCL_IGA_PLATFORM`.
- Selected `.text.*` section name and section base address from `iga-disasm/iga-disasm-manifest.json`.
- Whether `iga-pc-instructions.csv` exists and is non-empty.
- Selected matrix parse path and exact parse contents.
- MXFP4 `source-line-feasibility.parse` contents, including `source_line.status asm-line-static-cost` if IGA PC rows validate.
- Exact blocker text if validation does not reach `asm-line-static-cost`.

## Interpretation guide

- `source_line.status asm-line-static-cost` is acceptable static line-ranked optimization evidence when it comes from kernel-matched IGA PC rows joined to DWARF ranges.
- `source_line.status pass` is still required for sampled VTune exact source-line timing.
- Label-only `ocloc` rows such as `L0:` are not byte PC evidence and must not be promoted to static source-line cost.
- If IGA preparation, disassembly, parsing, or resolver mapping fails, record the blocker and keep the issue open for follow-up validation.
