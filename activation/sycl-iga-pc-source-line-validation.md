# SYCL IGA PC Source-Line Attribution Validation

Date: 2026-07-05
Tracker issue: `llama.cpp-xame`
Validation owner: lead

## Summary

Lead validation was executed on B50 (`ONEAPI_DEVICE_SELECTOR=level_zero:1`) with oneAPI sourced via:

```bash
set +u
source /opt/intel/oneapi/setvars.sh --force
set -u
```

Result: **`asm-line-static-cost` did not validate**. Both the probe matrix and MXFP4 run ended at `source_line.status dwarf-line-table-only`.

Required semantics remain:

```text
source_line.status asm-line-static-cost means exact static source-line cost from IGA PC rows, not sampled runtime timing.
source_line.status pass remains the only sampled VTune exact status.
```

## Probe matrix validation

Artifact root:

```text
/tmp/sycl_source_line_iga_matrix_20260705_173448
```

Selected parse:

```text
/tmp/sycl_source_line_iga_matrix_20260705_173448/build-matrix/debug_full/source-line-feasibility.parse
```

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
source_line.source_attribution_mode dwarf-line-table
source_line.blocker none
source_line.status dwarf-line-table-only
```

IGA manifest for selected row:

```json
{
  "extract.kernel_match": "sycl_source_line_probe",
  "extract.platform": "xe2",
  "extract.section": ".text._ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E29sycl_source_line_probe_kernel",
  "extract.section_addr": "0x0",
  "extract.status": "ok"
}
```

IGA PC parser output for selected row:

```text
kernel,pc,pc_hex,opcode,text,raw,send_comment,source
iga_pc.status no_pc_rows
```

Interpretation: IGA text-section selection succeeded for the probe, but the IGA disassembly/parser path produced no PC instruction rows. The checker correctly rejected static attribution and accepted only DWARF coverage.

## MXFP4 feasibility validation

Artifact roots:

```text
/tmp/sycl_mxfp4_iga_source_line_20260705_173611
/tmp/sycl_mxfp4_iga_source_line_build_20260705_173611
```

MXFP4 parse contents:

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
source_line.source_attribution_mode dwarf-line-table
source_line.blocker none
source_line.status dwarf-line-table-only
```

MXFP4 IGA manifest:

```json
{
  "extract.kernel_match": "mxfp4_pair_glu_xmx_tiled_packed_r8_m2_sparse32_bias",
  "extract.platform": "xe2",
  "extract.section": "",
  "extract.section_addr": "",
  "extract.status": "missing_kernel_text_section"
}
```

Interpretation: MXFP4 DWARF line coverage is present (`923` DWARF source rows), but IGA section selection failed because no `.text.*` section matched the required kernel name. VTune/GTPin also reported register pressure (`source_line.gtpin_register_pressure 1`). The run therefore did not produce static source-line cost rows.

## Final status

- Static IGA line attribution: **blocked**.
- Current accepted status: `source_line.status dwarf-line-table-only`.
- `llama.cpp-040b` should remain open for a follow-up that can map the MXFP4 required kernel to the correct ZEBin `.text.*` section name and/or produce usable IGA PC rows.

## Follow-up blocker

The next fix should address at least one of:

1. IGA disassembly/parser emits no PC rows for the probe even when section extraction succeeds.
2. MXFP4 required kernel name does not match a `.text.*` section (`extract.status missing_kernel_text_section`).
3. VTune sampled exact source rows remain unavailable and GTPin reports register pressure on MXFP4 kernels.

Label-only `ocloc` rows such as `L0:` are still not byte PC evidence and must not be promoted to `asm-line-static-cost`.
