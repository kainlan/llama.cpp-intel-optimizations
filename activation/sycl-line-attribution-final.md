# SYCL line-attribution capability final report

Date: 2026-07-05
Tracker issue: `llama.cpp-y9qm`
Scope: Task 8 final documentation/report only

## Static line attribution

Status: source/doc scaffold in place; real IGA lead validation was not executed in this session.

Existing implementation artifacts are present:

- `scripts/prepare-sycl-iga-disasm-inputs.py` selects a `.text.*` ZEBin section and writes an IGA command manifest.
- `scripts/parse-sycl-iga-pc-disasm.py` parses IGA JSON/text rows with explicit PCs.
- `scripts/resolve-sycl-zebin-asm-source-lines.py --iga-instructions-csv --pc-base` maps kernel-matched IGA section-relative PCs through DWARF line ranges.
- Matrix and MXFP4 runners are wired to prefer `iga-pc-instructions.csv` and fall back to `ocloc`/DWARF without fabricating static-cost evidence.

Existing reports do not prove real `asm-line-static-cost` validation from IGA in this session. `activation/sycl-iga-pc-source-line-validation.md` is explicitly a source-only validation template/status record. Older non-VTUNE validation in `activation/sycl-non-vtune-source-line-validation.md` reached `source_line.status dwarf-line-table-only`; it did not validate `asm-line-static-cost` because label-only `ocloc` assembly lacked numeric instruction addresses.

Result: `asm-line-static-cost` remains available as the intended static evidence level only after a lead run produces and maps real kernel-matched IGA PC rows. Follow-up `llama.cpp-040b` should remain open unless/until MXFP4 validates `source_line.status asm-line-static-cost`.

## Runtime sampled line attribution

Status: schema and capability probe scaffold are in place; no real sampled PC CSV was produced by this worker/session.

Existing implementation artifacts are present:

- `scripts/resolve-sycl-pc-samples-to-source-lines.py` maps real `kernel,pc,sample_count,sample_kind` rows to DWARF source lines.
- `scripts/check-sycl-vtune-source-lines.py`, `scripts/parse-sycl-source-attribution.py`, and `scripts/merge-sycl-staged-ledger.py` keep `sampled-line-cost` / `sampled_line_cost` distinct from VTune exact source rows.
- `scripts/sycl-intel-pc-sampling-capability.sh` can report `available`, `metrics_only`, or `unavailable`, but it never synthesizes `pc-samples.csv`.

`activation/sycl-intel-pc-sampling-capability.md` records source-only status and says lead execute validation is still required. No worker-produced `pc-samples.csv` is claimed here.

## Exact source-line attribution rule

`source_attribution.status exact_source_line` remains reserved for `source_line.status pass` from VTune GPU source rows. Runtime sampled PC rows may produce `sampled-line-cost`; IGA rows may produce `asm-line-static-cost`; DWARF coverage may produce `dwarf-line-table-only`. None of those non-`pass` paths should be promoted to `exact_source_line`.

## Next optimization use

For TG optimization today, use:

1. `pass` / `exact_source_line` if lead VTune source rows are available.
2. `sampled-line-cost` only after a real positive-count `pc-samples.csv` is produced and mapped.
3. `asm-line-static-cost` only after lead IGA PC validation maps kernel-matched rows through DWARF.
4. `dwarf-line-table-only` as coverage/fallback when cost-ranked source rows are not available.

Current documented state from this session: source-only scaffolds and docs are complete; real static IGA and runtime sampled PC lead validations are not claimed.
