## 5edo8.4 TG Decode Optimization - Deep Analysis

### Target: tg128 ≥ 80 tok/s with FA-on
### Current: tg128 = 62.82 tok/s

### Profile Breakdown (FA-on TG128 per layer, graph replay)
- MUL_MAT: 57.9ms (61.1%) - 1125 calls @ 0.051ms/call
- FLASH_ATTN_EXT: 26.2ms (27.7%) - 160 calls @ 0.164ms/call
- Other (MUL, SET_ROWS, ROPE, GLU, GET_ROWS, CPY): 10.7ms (11.3%)
- Total per step: ~94.7ms

### Non-FA TG Profile (for comparison)
- MUL_MAT: 70.3ms (84.8%) - 1445 calls @ 0.049ms/call
- SOFT_MAX: 1.7ms (2.1%) - 160 calls @ 0.011ms/call
- Other: 10.8ms (13.1%)
- Total per step: ~82.8ms

### The Math
FA-on vs FA-off:
- MUL_MAT saves: 70.3 - 57.9 = 12.4ms
- FLASH_ATTN_EXT adds: 26.2 - 1.7 = 24.5ms
- Net overhead: 24.5 - 12.4 = 12.1ms (matches observation: 94.7 - 82.8 = 11.9ms)

To reach 80 tok/s: 1/0.0125 = 80 tok/s → 12.5ms/step needed
Current: 94.7ms/step → need to save 82.2ms

Even if FLASH_ATTN_EXT was eliminated (impossible): 
57.9ms MUL_MAT + 10.7ms other = 68.6ms → 14.6 tok/s (only 63% of current!)

The 80 tok/s target on FA-on requires:
- FLASH_ATTN_EXT: 26.2ms → ~4ms (92% reduction)
- MUL_MAT: 57.9ms → ~58ms (already minimal)
Total: 62ms → still only 16.1 tok/s

CONCLUSION: tg128 ≥ 80 tok/s on FA-on is NOT attainable with current kernel design.
The target is incompatible with flash attention overhead (~26ms on top of ~58ms GEMM).

### Tests Performed
1. GGML_SYCL_FA_ESIMD=0 → 62.86 tok/s (same as baseline)
2. GGML_SYCL_FORCE_MMVQ=1 → 62.80 tok/s
3. GGML_SYCL_FORCE_ESIMD=1 → 62.84 tok/s
4. GGML_SYCL_FORCE_MMQ=1 → 62.80 tok/s
5. ESIMD D<=64 (D=128→VEC) → 62.81 tok/s
6. VEC D<=32, XMX v2 nc=1 D=128 → 60.65 tok/s (WORSE)

All dispatch paths converge to ~62.8 tok/s for nc=1 D=128.

### Dispatch Order (confirmed)
1. ESIMD nc=1: returns first for all nc<=1 (no D check in current code)
2. VEC nc=1: only reached if ESIMD disabled (same performance)
3. oneDNN: skipped for nc<8 (ne01 < onednn_min_ncols=8)
4. XMX v2: reaches nc=1 only when VEC is disabled AND oneDNN skipped

### Kernel Comparisons for nc=1 D=128
- ESIMD: 32 partitions, 16.25KB SLM, cooperative reduction. 0.164ms/call
- VEC: 16-sub-group, ZERO SLM, register-only. 0.164ms/call (identical)
- XMX v2 NCOLS=1: 1 tile-wide, wastes 15/16 capacity. 0.164ms/call (worse with NCOLS=8)

### Root Cause
Flash attention at nc=1 is **compute-bound** with ~1536 exp() calls per KV step per head pair.
Both ESIMD and VEC kernels hit the same theoretical minimum.
The nc=1 flash attention case simply cannot compete with non-FA GEMM-heavy path.