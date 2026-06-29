# MXFP4 TG Microbench Lead Validation Matrix

## Worker-Forbidden Commands

Workers must not run B50/B580 model gates, sycl-ls, multi-GPU selectors, or commands that load /Storage/GenAI/models/.

Do not run non-dry-run GPU probes, B50/B580 `llama-cli` gates, B50/B580 `llama-bench` gates, commands using `ONEAPI_DEVICE_SELECTOR=level_zero:0`, `ONEAPI_DEVICE_SELECTOR=level_zero:1`, or `ONEAPI_DEVICE_SELECTOR=level_zero:0,1`, or any command that opens models from `/Storage/GenAI/models/` unless you are the lead.

## Worker-Safe Dry Run

```bash
./scripts/sycl-build.sh sycl-mxfp4-moe-bench
python3 scripts/run-sycl-mxfp4-tg-microbenches.py --dry-run --out-dir /tmp/mxfp4_tg_dryrun
python3 -m pytest tools/sycl-mxfp4-moe-bench/tests -q
```

## Current Safe Route Environment

Record this known-safe route before comparing any synthetic winner:

```bash
export GGML_SYCL_MOE_PHASE_MATERIALIZE=1
export GGML_SYCL_MOE_PHASE_BULK_XMX=1
export GGML_SYCL_MOE_DOWN_SUM_DIRECT=1
```

## Lead-Only B50 Gate Template

The following command is lead-only. Workers must document it but must not run it.

```bash
source /opt/intel/oneapi/setvars.sh --force
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-cli \
  -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf -ngl 99 \
  -cnv -st --simple-io --no-display-prompt \
  --chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --reasoning-format none --reasoning-budget 0 \
  -p 'Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5' \
  -n 48 --seed 42 --temp 0
```

Expected output starts with : 1, 2, 3, 4, 5.

Lead-only performance validation must also collect B50 PP512 and TG128 evidence for the candidate route and the current safe route. Workers must not run those model commands.

## Promotion Checklist

Promote a synthetic winner only when lead-owned evidence shows all of the following:

- Exact count output from the canonical GPT-OSS B50 count gate starts with `: 1, 2, 3, 4, 5`.
- `fatal.total 0` in the microbench and full-model evidence.
- Required route path evidence is present for the candidate route.
- Forbidden fallback absence is documented; the run must not silently fall back to rejected, legacy, CPU, transposed-B, or non-MXFP4 paths.
- MXFP4 profile evidence is present, including the expected route/profile buckets rather than only aggregate throughput.
- B50 `PP512 >= 1100`.
- B50 `TG128 >= 45`, unless the lead explicitly accepts a near-target result.
- A separate runtime implementation plan is approved before any backend dispatch changes.

B580 validation is blocked until the B50 count, `fatal.total 0`, required path evidence, forbidden fallback absence, MXFP4 profile evidence, `PP512 >= 1100`, and `TG128 >= 45` gate passes, unless the lead explicitly accepts a near-target B50 TG result.
