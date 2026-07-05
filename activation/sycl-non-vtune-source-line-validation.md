# SYCL Non-VTUNE Source-Line Rows Validation

Date: 2026-07-05
Branch: `feature/sycl-mxfp4-tg-runtime`
Execution task: `llama.cpp-qku5`
Follow-up tracker: `llama.cpp-040b`

## Safe gates

Commands:

```bash
bash -n scripts/sycl-source-line-debug-matrix.sh scripts/sycl-vtune-source-line-feasibility.sh scripts/sycl-gptoss-full-attribution-profile.sh scripts/sycl-gptoss-staged-attribution-profile.sh
python3 -m pytest \
  tests/test-sycl-vtune-asm-parser.py \
  tests/test-sycl-zebin-asm-source-line-resolver.py \
  tests/test-sycl-zebin-line-table-parser.py \
  tests/test-sycl-zebin-line-table-source-csv.py \
  tests/test-sycl-vtune-source-line-checker.py \
  tests/test-sycl-source-attribution-parser.py \
  tests/test-sycl-staged-ledger-merger.py \
  tests/test-sycl-source-line-debug-matrix-script.py \
  tests/test-sycl-vtune-source-line-feasibility-script.py \
  tests/test-sycl-full-attribution-profile-script.py \
  tests/test-sycl-staged-attribution-profile-script.py \
  tests/test-sycl-vtune-source-line-enablement-docs.py -q
git diff --check
```

Output:

```text
102 passed in 2.63s
```

## Probe matrix

Command:

```bash
OUT=/tmp/sycl_source_line_asm_matrix_$(date +%Y%m%d_%H%M%S)
set +u
source /opt/intel/oneapi/setvars.sh --force
set -u
scripts/sycl-source-line-debug-matrix.sh \
  --execute \
  --i-understand-this-runs-gpu-source-probe \
  --out-root "$OUT" \
  --device-selector level_zero:1 \
  --vtune-target-gpu 0:7:0.0
```

Root: `/tmp/sycl_source_line_asm_matrix_20260705_130626`
Selected row: `debug_full`
Selected parse: `/tmp/sycl_source_line_asm_matrix_20260705_130626/build-matrix/debug_full/source-line-feasibility.parse`

Selected parse contents:

```text
source_line.debug_line_present 1
source_line.non_unknown_rows 0
source_line.vtune_sampled_non_unknown_rows 0
source_line.vtune_no_gpu_side_trace 1
source_line.gtpin_no_kernels 0
source_line.gtpin_register_pressure 0
source_line.required_kernel sycl_source_line_probe
source_line.dwarf_status ok
source_line.dwarf_source_rows 17
source_line.dwarf_required_path_present 1
source_line.dwarf_source_line_rows 17
source_line.allow_dwarf_line_table_only 1
source_line.asm_source_line_rows 0
source_line.allow_asm_line_static_cost 1
source_line.source_attribution_mode dwarf-line-table
source_line.blocker none
source_line.status dwarf-line-table-only
```

ASM resolver summary: `/tmp/sycl_source_line_asm_matrix_20260705_130626/build-matrix/debug_full/asm-source-lines.parse`

```text
asm_source.status no_asm_source_matches
asm_source.mapped_instruction_count 0
asm_source.unmapped_instruction_count 0
asm_source.source_line_rows 0
asm_source.blocker no_asm_source_matches
```

Representative disassembly path: `/tmp/sycl_source_line_asm_matrix_20260705_130626/build-matrix/debug_full/zebin-disasm/dump/.text._ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E29sycl_source_line_probe_kernel.asm`

Observation: `ocloc disasm` produced assembly text, but instruction rows use labels like `L0:` and do not include numeric instruction addresses, so the address-based resolver cannot join ASM rows to DWARF line-table addresses.

## MXFP4 feasibility

Command:

```bash
MATRIX_PARSE=/tmp/sycl_source_line_asm_matrix_20260705_130626/build-matrix/debug_full/source-line-feasibility.parse
OUT=/tmp/sycl_mxfp4_asm_source_line_$(date +%Y%m%d_%H%M%S)
BUILD=/tmp/sycl_mxfp4_asm_source_line_build_$(date +%Y%m%d_%H%M%S)
set +u
source /opt/intel/oneapi/setvars.sh --force
set -u
scripts/sycl-vtune-source-line-feasibility.sh \
  --execute \
  --i-understand-this-runs-gpu-microbenchmarks \
  --out-root "$OUT" \
  --build-dir "$BUILD" \
  --device-selector level_zero:1 \
  --vtune-target-gpu 0:7:0.0 \
  --require-matrix-pass "$MATRIX_PARSE"
```

Root: `/tmp/sycl_mxfp4_asm_source_line_20260705_130742`
Build root: `/tmp/sycl_mxfp4_asm_source_line_build_20260705_130742`
Parse: `/tmp/sycl_mxfp4_asm_source_line_20260705_130742/source-line-feasibility.parse`

Parse contents:

```text
source_line.debug_line_present 1
source_line.non_unknown_rows 0
source_line.vtune_sampled_non_unknown_rows 0
source_line.vtune_no_gpu_side_trace 0
source_line.gtpin_no_kernels 0
source_line.gtpin_register_pressure 1
source_line.required_kernel mxfp4_pair_glu_xmx_tiled_packed_r8_m2_sparse32_bias
source_line.dwarf_status ok
source_line.dwarf_source_rows 923
source_line.dwarf_required_path_present 1
source_line.dwarf_source_line_rows 923
source_line.allow_dwarf_line_table_only 1
source_line.asm_source_line_rows 0
source_line.allow_asm_line_static_cost 1
source_line.source_attribution_mode dwarf-line-table
source_line.blocker none
source_line.status dwarf-line-table-only
```

ASM resolver summary: `/tmp/sycl_mxfp4_asm_source_line_20260705_130742/asm-source-lines.parse`

```text
asm_source.status no_asm_source_matches
asm_source.mapped_instruction_count 0
asm_source.unmapped_instruction_count 0
asm_source.source_line_rows 0
asm_source.blocker no_asm_source_matches
```

Disassembly artifacts:

- `ocloc.stderr`: `/tmp/sycl_mxfp4_asm_source_line_20260705_130742/zebin-disasm/ocloc.stderr`
- Representative ASM: `/tmp/sycl_mxfp4_asm_source_line_20260705_130742/zebin-disasm/dump/.text._ZTSZZZL38mxfp4_dpas_pack_q8_grouped_chunks_syclRN4sycl3_V15queueEPKvPaPfPKiS8_S8_S8_iiiiillRKSt6vectorINS0_5eventESaISA_EES8_ENKUlS2_E0_clES2_ENKUlRNS0_7handlerEE_clESH_EUlNS0_2idILi1EEEE_.asm`

Observation: `ocloc disasm` produced non-empty ASM files, but the lines use labels (`L0:`, `L304:`) rather than numeric instruction addresses. The current address-based resolver therefore finds no mapped ASM source rows and correctly falls back to DWARF-only evidence.

## Interpretation

- `pass` means sampled VTune exact source-line rows.
- `asm-line-static-cost` means exact static source-line rows from DWARF plus assembly addresses.
- `dwarf-line-table-only` means line-table coverage without instruction-level or sampled cost rows.

Both lead-only validations reached accepted `dwarf-line-table-only` coverage. Neither validated `asm-line-static-cost` on this workstation because real `ocloc disasm` output did not provide numeric instruction addresses usable by the resolver.

## Follow-up

TG optimization cannot yet use `asm-line-static-cost` rows from the current `ocloc disasm` output. The validated safe fallback remains `dwarf-line-table-only`. A separate disassembler adapter is needed to map label-based EU assembly (`L0:`, `L304:`) back to instruction offsets or to obtain an alternate disassembly format with numeric instruction addresses before `asm-line-static-cost` can be promoted for real MXFP4 optimization attribution.
