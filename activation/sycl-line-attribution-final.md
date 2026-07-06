# SYCL line-attribution capability final report

Date: 2026-07-05
Tracker issue: `llama.cpp-040b`
Validation owner: lead

## Static line attribution

Status: **validated for the standalone probe; still blocked for the MXFP4 target kernel.**

Latest lead validation artifacts:

- Fixed probe matrix root: `/tmp/sycl_source_line_iga_matrix_fix_20260705_192345`
- Selected matrix parse: `/tmp/sycl_source_line_iga_matrix_fix_20260705_192345/build-matrix/debug_full/source-line-feasibility.parse`
- Latest MXFP4 root: `/tmp/sycl_mxfp4_iga_source_line_fix4_20260705_210052`
- Latest MXFP4 build root: `/tmp/sycl_mxfp4_iga_source_line_build_fix4_20260705_210052`

Results:

- Probe matrix selected row: `source_line.status asm-line-static-cost`
- Probe static mapping: `asm_source.mapped_instruction_count 18`, `asm_source.source_line_rows 2`, top line `main.cpp:148`
- MXFP4 task-to-ZEBin selection: fixed; selected task maps to `.text._ZTS39mxfp4_pair_glu_xmx_tiled_dpas_m2_kernelILi8ELi3ELb0ELb0EE`
- MXFP4 IGA extraction: fixed; `extract.status ok`
- MXFP4 IGA PC rows: present; `1583` CSV lines in `iga-pc-instructions.csv`
- MXFP4 remaining blocker: selected compute ZEBin has no usable DWARF source rows (`failed to check source lines: no source rows found`)

Detailed report: `activation/sycl-iga-pc-source-line-validation.md`.

Existing implementation artifacts are present:

- `scripts/prepare-sycl-iga-disasm-inputs.py` selects a `.text.*` ZEBin section and writes an IGA command manifest.
- `scripts/parse-sycl-iga-pc-disasm.py` parses IGA JSON/text rows with explicit PCs, including real JSON v2 `elems` / `kind: "I"` output.
- `scripts/resolve-sycl-zebin-asm-source-lines.py --iga-instructions-csv --pc-base` maps kernel-matched IGA section-relative PCs through DWARF line ranges.
- Matrix and MXFP4 runners prefer `iga-pc-instructions.csv` and fall back to `ocloc`/DWARF without fabricating static-cost evidence.
- The MXFP4 runner now selects the ZEBin matching the VTune-selected compute task instead of trusting the first archived binary.

Result: `asm-line-static-cost` is a real validated static evidence level for source-line probe artifacts. For MXFP4, follow-up `llama.cpp-040b` remains open because the selected compute ZEBin does not carry usable source-line DWARF for PC-to-line mapping.

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
3. `asm-line-static-cost` for artifacts whose IGA PC rows and DWARF line rows both validate; currently this is true for the standalone probe, not for the MXFP4 target ZEBin.
4. `dwarf-line-table-only` as coverage/fallback when cost-ranked source rows are not available.

Current MXFP4 state: task-to-ZEBin selection and IGA PC extraction work, but cost-ranked source rows remain blocked by missing source-line DWARF in the selected compute ZEBin.
