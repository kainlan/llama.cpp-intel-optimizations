# oneAPI DPAS Capability Report

Command:

```bash
bash scripts/check-sycl-dpas-capabilities.sh
```

Observed installed oneAPI capability:

```text
dpas.header=/opt/intel/oneapi/compiler/2025.3/include/sycl/ext/intel/esimd/xmx/dpas.hpp
dpas.common_header=/opt/intel/oneapi/compiler/2025.3/include/sycl/ext/intel/esimd/xmx/common.hpp
dpas.repeat_count.max=8
dpas.systolic_depth=8
dpas.exec_size.allowed=8,16
dpas.bdpas.present=0
dpas.fp4_e2m1.present=0
```

Decision:

- The transposed-B gate/up route in `docs/plans/2026-06-28-sycl-mxfp4-transposed-gateup-tg.md` used ordinary `xmx::dpas` only, so it did not require a source-built oneAPI toolchain. Lead validation rejected and removed that route because it regressed TG.
- Do not replace `/opt/intel/oneapi` or system Level Zero for this rejected route.
- The default checker prefers `/opt/intel/oneapi/compiler/latest/include/sycl/ext/intel/esimd/xmx/dpas.hpp`, falls back to the 2025.3 path, and prints the resolved real header path. On this system, `latest` resolves to `/opt/intel/oneapi/compiler/2025.3/include/sycl/ext/intel/esimd/xmx/dpas.hpp`.
- Do not build Intel LLVM/DPC++ from source for the rejected transposed-B route. Reconsider an isolated source build only if a new, separate route has evidence that newer `bdpas`/`e2m1` support can directly target MXFP4/FP4 and improve TG safely.
