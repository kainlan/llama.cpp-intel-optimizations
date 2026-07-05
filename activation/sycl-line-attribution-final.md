# SYCL line-attribution capability final report

Date: 2026-07-05
Tracker issue: `llama.cpp-y9qm`
Validation owner: lead

## Static line attribution

Status: **implemented but not validated as `asm-line-static-cost` on real artifacts**.

Lead validation was executed after the source-only implementation landed:

- Probe matrix root: `/tmp/sycl_source_line_iga_matrix_20260705_173448`
- Selected matrix parse: `/tmp/sycl_source_line_iga_matrix_20260705_173448/build-matrix/debug_full/source-line-feasibility.parse`
- MXFP4 root: `/tmp/sycl_mxfp4_iga_source_line_20260705_173611`
- MXFP4 build root: `/tmp/sycl_mxfp4_iga_source_line_build_20260705_173611`

Results:

- Probe matrix selected row: `source_line.status dwarf-line-table-only`
- MXFP4 feasibility: `source_line.status dwarf-line-table-only`
- MXFP4 DWARF coverage: `source_line.dwarf_source_rows 923`
- MXFP4 static IGA blocker: `extract.status missing_kernel_text_section`
- MXFP4 sampled VTune/GTPin blocker: `source_line.gtpin_register_pressure 1`

Detailed report: `activation/sycl-iga-pc-source-line-validation.md`.

Existing implementation artifacts are present:

- `scripts/prepare-sycl-iga-disasm-inputs.py` selects a `.text.*` ZEBin section and writes an IGA command manifest.
- `scripts/parse-sycl-iga-pc-disasm.py` parses IGA JSON/text rows with explicit PCs.
- `scripts/resolve-sycl-zebin-asm-source-lines.py --iga-instructions-csv --pc-base` maps kernel-matched IGA section-relative PCs through DWARF line ranges.
- Matrix and MXFP4 runners are wired to prefer `iga-pc-instructions.csv` and fall back to `ocloc`/DWARF without fabricating static-cost evidence.

Result: `asm-line-static-cost` remains the intended static evidence level only after a lead run produces and maps real kernel-matched IGA PC rows. Follow-up `llama.cpp-040b` remains open.

## Runtime sampled line attribution

Status: **schema implemented; runtime PC sampling not available**.

Lead PC-sampling capability probe root:

```text
/tmp/sycl_intel_pc_sampling_capability_20260705_175719
```

Result:

```text
pc_sampling.status metrics_only
pc_sampling.blocker no_public_pc_sample_api_confirmed
pc_sampling.blocker vtune_source_rows_empty
pc_sampling.blocker gtpin_not_found
pc_sampling.blocker pti_files_found_but_no_pc_sample_producer
```

Detailed report: `activation/sycl-intel-pc-sampling-capability.md`.

Existing implementation artifacts are present:

- `scripts/resolve-sycl-pc-samples-to-source-lines.py` maps real `kernel,pc,sample_count,sample_kind` rows to DWARF source lines.
- `scripts/check-sycl-vtune-source-lines.py`, `scripts/parse-sycl-source-attribution.py`, and `scripts/merge-sycl-staged-ledger.py` keep `sampled-line-cost` / `sampled_line_cost` distinct from VTune exact source rows.
- `scripts/sycl-intel-pc-sampling-capability.sh` reports `available`, `metrics_only`, or `unavailable`, and never synthesizes `pc-samples.csv`.

No true sampled runtime source-line attribution is available from public/local tooling on this host yet.

## Exact source-line attribution rule

`source_attribution.status exact_source_line` remains reserved for `source_line.status pass` from VTune GPU source rows. Runtime sampled PC rows may produce `sampled-line-cost`; IGA rows may produce `asm-line-static-cost`; DWARF coverage may produce `dwarf-line-table-only`. None of those non-`pass` paths should be promoted to `exact_source_line`.

## Evidence levels

| Evidence | Runtime sampled? | Source-line ranked? | Accepted status |
|---|---:|---:|---|
| VTune GPU source rows | yes | yes | `source_line.status pass` |
| PC sample CSV mapped through DWARF | yes | yes | `source_line.status sampled-line-cost` |
| IGA PC static instruction rows mapped through DWARF | no | yes | `source_line.status asm-line-static-cost` |
| DWARF line-table coverage only | no | no cost ranking | `source_line.status dwarf-line-table-only` |

## Next optimization use

For TG optimization today, use:

1. `pass` / `exact_source_line` if lead VTune source rows are available.
2. `sampled-line-cost` only after a real positive-count `pc-samples.csv` is produced and mapped.
3. `asm-line-static-cost` only after lead IGA PC validation maps kernel-matched rows through DWARF.
4. `dwarf-line-table-only` as coverage/fallback when cost-ranked source rows are not available.

Current validated state: source-line coverage exists (`dwarf-line-table-only`), but neither static-cost line ranking nor runtime sampled line ranking is available yet.
