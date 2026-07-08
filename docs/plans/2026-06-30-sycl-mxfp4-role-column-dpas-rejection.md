# SYCL MXFP4 Role-Column DPAS Gate/Up Rejection

Role-column gate/up fusion is rejected before implementation. The current M2 direct-Q8 path uses DPAS as `C[m,n] = sum_k A[m,k] * B[k,n]`: `A` is either the gate weight rows or the up weight rows, and `B` is the Q8 activation tile. DPAS N columns vary RHS vectors for the same A matrix. They cannot make column 0 use gate weights and column 1 use up weights for the same logical row.

Current code evidence:

- `ggml/src/ggml-sycl/mmvq.cpp:9659` starts `mxfp4_pair_glu_xmx_tiled_dpas_m2_direct_q8_sycl()`.
- `ggml/src/ggml-sycl/mmvq.cpp:9769-9770` issues separate DPAS calls for `gate_a_vec0` and `up_a_vec0`.
- `ggml/src/ggml-sycl/mmvq.cpp:9773-9774` extracts output column zero from each separate DPAS result.
- `/opt/intel/oneapi/compiler/2025.3/include/sycl/ext/intel/esimd/xmx/dpas.hpp` caps `RepeatCount` at 8, so row-doubling the existing `Repeat=8` path is not available.

A valid use of DPAS N columns is batching multiple RHS vectors that share the same A matrix. Therefore the next implementation candidate is same-expert multi-RHS gate/up batching, not role-column gate/up fusion.
