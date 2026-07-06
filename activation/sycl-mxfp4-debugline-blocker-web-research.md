# SYCL MXFP4 `.debug_line` Blocker Web/Source Research

Date: 2026-07-06
Tracker: `llama.cpp-040b`

## Local symptom to explain

Full benchmark and narrow `sycl-mxfp4-source-line-probe` both compile/run the MXFP4 pair-GLU path. IGA can emit numeric PC rows, but the selected ZEBin has no `.debug_line`:

```text
/tmp/sycl_mxfp4_source_line_probe_gpu_20260705_222039
source_line.debug_line_present 0
source_line.dwarf_error no source rows found
source_line.blocker missing_debug_line
```

Earlier full-target builds also emitted:

```text
VCDebugInfo: only modules with one CU are supported at the moment, the debug information for Module will be dropped out.
```

## Finding 1: the exact warning is from Intel Graphics Compiler VectorCompiler

Source: public Intel Graphics Compiler clone, commit `f1922cb05735b4c4bb417ec24170768430840471`:

- `https://github.com/intel/intel-graphics-compiler/blob/f1922cb05735b4c4bb417ec24170768430840471/IGC/VectorCompiler/lib/Utils/General/DebugInfo.cpp`

Relevant code:

```cpp
bool vc::DIBuilder::checkIfModuleHasDebugInfo(const llvm::Module &M) {
  unsigned NumDebugCUs =
      std::distance(M.debug_compile_units_begin(), M.debug_compile_units_end());

  if (NumDebugCUs == 0)
    return false;

  if (NumDebugCUs > 1)
    vc::warn(M.getContext(), "VCDebugInfo",
             "only modules with one CU are supported at the moment, "
             "the debug information for Module will be dropped out.");
  return NumDebugCUs == 1;
}
```

Interpretation: this is not a VTune reporting artifact. IGC explicitly refuses this debug-info path when a device LLVM module has more than one DWARF compile unit. If the backend module has multiple CUs, downstream ZEBin can legitimately lack `.debug_line` even with `-g` / profiling flags.

## Finding 2: SYCL `per_kernel` split does not guarantee one CU per backend module

Source: public Intel LLVM clone, commit `72c37b0d71123bc97f89b1d9c1cacafc7fde8507`:

- `https://github.com/intel/llvm/blob/72c37b0d71123bc97f89b1d9c1cacafc7fde8507/sycl/doc/UsersManual.md`
- `https://github.com/intel/llvm/blob/72c37b0d71123bc97f89b1d9c1cacafc7fde8507/sycl/doc/design/OptionalDeviceFeatures.md`

The user manual defines:

```text
per_kernel - creates a separate device code module for each SYCL kernel.
Each device code module will contain a kernel and all its dependencies,
such as called functions and used variables.

per_source - creates a separate device code module for each source
(translation unit). Each device code module will contain a bunch of kernels
grouped on per-source basis and all their dependencies...
```

Interpretation: `-fsycl-device-code-split=per_kernel` isolates device images by kernel entry point, but each image still carries dependencies. If those dependencies come from a module with multiple CUs, IGC can still see `NumDebugCUs > 1`. So `per_kernel` alone is not a sufficient fix.

## Finding 3: source mapping requires DWARF line info; `-gline-tables-only` is not useful for this GPU target

Source: public Intel LLVM/Clang docs, commit `72c37b0d71123bc97f89b1d9c1cacafc7fde8507`:

- `https://github.com/intel/llvm/blob/72c37b0d71123bc97f89b1d9c1cacafc7fde8507/clang/docs/UsersManual.rst`

Clang profiler docs say DWARF source line info is required for profilers to map instructions to source lines, and `-fdebug-info-for-profiling` / `-funique-internal-linkage-names` can improve usefulness. The same manual defines `-gline-tables-only` as generating line number tables only.

But local oneAPI validation logs repeatedly showed:

```text
icpx: warning: ignoring '-gline-tables-only' option as it is not currently supported for target 'spir64_gen-unknown-unknown'; only supported for host compilation
```

Interpretation: `-gline-tables-only` is not the root fix for the Intel GPU device image. The useful debug flags are likely `-g`, `-fdebug-info-for-profiling`, possibly `-funique-internal-linkage-names`, but the main requirement is structuring the device backend module so IGC sees exactly one CU.

## Finding 4: the current narrow executable is narrow only at the host level

Local validation: `sycl-mxfp4-source-line-probe` links only:

- `mxfp4_source_line_probe.cpp`
- `kernels/reference/mxfp4_inline_dot.cpp`

However, `mxfp4_inline_dot.cpp` itself contains many SYCL kernel families and helper paths. Even with `-ffunction-sections` / `--gc-sections`, device compilation still emits a large multi-section ZEBin with many MXFP4 `.text.*` entries and no `.debug_line`.

Interpretation: host linker GC is not enough. The device compiler still processes the large source/device image. The next experiment should reduce the device translation unit contents, not only the executable's host object list.

## Recommended next experiments

1. **Source-split the pair-GLU source-line path**
   - Move or duplicate only the minimal pair-GLU launcher and required helper/device templates for `mxfp4_pair_glu_xmx_tiled_packed_r8_m2_sparse32_bias` into a source-line-only TU.
   - Keep normal benchmark/runtime untouched.
   - Build `sycl-mxfp4-source-line-probe` against that small TU.
   - Expected win: IGC module may contain one CU and preserve `.debug_line`.

2. **Macro-guard `mxfp4_inline_dot.cpp` for the source-line probe target**
   - Add a source-line-only define, e.g. `SYCL_MXFP4_SOURCE_LINE_PROBE_ONLY`, around unrelated benchmark families so only the required pair-GLU definitions instantiate.
   - Faster to test than a clean source split, but must not alter default `sycl-kernel-bench` behavior.

3. **Try `-funique-internal-linkage-names` as a low-risk adjunct**
   - Clang docs recommend it with `-fdebug-info-for-profiling` for profiler correlation.
   - It is unlikely to fix `NumDebugCUs > 1`, but may improve source/symbol correlation once `.debug_line` exists.

4. **Stop spending time on `-gline-tables-only` for the Intel GPU target**
   - Local `icpx` explicitly ignores it for `spir64_gen`.

5. **Do not expect `per_kernel` alone to solve it**
   - Intel LLVM docs say per-kernel images include dependencies; IGC's multi-CU warning can still fire.

## Practical implementation sketch

- Create `tools/sycl-kernel-bench/kernels/reference/mxfp4_pair_glu_source_line.cpp` or similar.
- Copy/move only the helper functions/templates needed by the default source-line kernel shape.
- Expose a narrow `run_mxfp4_pair_glu_source_line_probe(...)` or preserve the existing signature if practical.
- Link `sycl-mxfp4-source-line-probe` against this small file instead of full `mxfp4_inline_dot.cpp`.
- Source/build gates first:

```bash
python3 -m pytest tests/test-sycl-mxfp4-source-line-probe-source.py \
  tests/test-sycl-vtune-source-line-feasibility-script.py -q
./scripts/sycl-build.sh sycl-mxfp4-source-line-probe
```

- Lead-only real validation after build succeeds:

```bash
scripts/sycl-vtune-source-line-feasibility.sh --execute ...
```

Success criterion: selected ZEBin has `source_line.debug_line_present 1`, then ideally `source_line.status asm-line-static-cost`.

## Residual risks

- Copying only the required pair-GLU path can drift from benchmark/runtime code if not kept tightly scoped.
- Moving helpers out of `mxfp4_inline_dot.cpp` is cleaner long term but may be invasive.
- If the small TU still gets multiple CUs because of headers or device library linkage, this may require a compiler-side workaround or upstream IGC report.
