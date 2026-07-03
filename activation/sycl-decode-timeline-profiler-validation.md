# SYCL decode timeline profiler validation

Date: 2026-07-03
Branch: `feature/sycl-mxfp4-tg-runtime`
Scope: Task `llama.cpp-rwjy` automated integration gates only. No real model/GPU validation was run.

## Python source/parser/script gates

Command:

```bash
python3 -m pytest \
  tests/test-sycl-timeline-parser.py \
  tests/test-sycl-timeline-compute-forward-source.py \
  tests/test-sycl-timeline-wait-graph-source.py \
  tests/test-sycl-timeline-cache-source.py \
  tests/test-sycl-decode-timeline-profile-script.py \
  tests/test-sycl-kernel-profiler-source.py \
  tests/test-sycl-kernel-profiler-source-mmvq.py \
  tests/test-sycl-kernel-profiler-source-copy.py \
  -q
```

Result: PASS — `34 passed in 0.31s`.

## SYCL build gates

Command:

```bash
set +u; source /opt/intel/oneapi/setvars.sh --force; set -u
./scripts/sycl-build.sh test-sycl-timeline test-sycl-kernel-profiler llama-bench llama-cli
```

Result: PASS — all requested targets built successfully.

## Selected CTest gates

Command:

```bash
ctest --test-dir build -R "test-sycl-timeline|test-sycl-kernel-profiler|test-sycl-e2e-profile" -V
```

Result: PASS — `100% tests passed, 0 tests failed out of 3`.

Passing tests:
- `test-sycl-e2e-profile`
- `test-sycl-timeline`
- `test-sycl-kernel-profiler`

## Dry-run profile script gate

Command:

```bash
bash scripts/sycl-gptoss-decode-timeline-profile.sh
```

Result: PASS — printed dry-run command and artifact paths only; no model execution.

Dry-run included:
- `GGML_SYCL_TIMELINE=timeline+events`
- `GGML_SYCL_TIMELINE_TOKEN_START=1`
- `GGML_SYCL_KERNEL_PROFILE=1`
- FA-on MoE phase envs
- `./build/bin/llama-bench ... -fa 1 -p 512 -n 128 -r 1`
- `sycl-timeline.json`, `sycl-kernels`, `bench.stdout`, `bench.stderr`, `timeline.parse`, `kernels.parse`

## Notes

- This validation intentionally did not access `/Storage/GenAI/models` or run `llama-bench`/`llama-cli` with a model.
- Lead-only full GPT-OSS B50 timeline validation remains Task `llama.cpp-0aqr`.
