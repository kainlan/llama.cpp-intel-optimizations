# SYCL Decode Profiling Completeness Validation

Date: 2026-07-04
Branch: `feature/sycl-mxfp4-tg-runtime`
Validated commit: `5832bfc33` (`docs(sycl): record profiling completeness safe gates`)
Artifact root: `/tmp/sycl_profile_gates_20260704_021951`

## Scope

Task 11 safe integration gates only. No real model/GPU profiling run was executed.
The profile script was exercised in dry-run mode only; the printed command still
references `/Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf`, but dry-run does not
execute `llama-bench` or access the model.

This rerun includes the remote code commit `b77374210` (`fix(sycl): label
binbcast ops by operation`) that landed before this validation record was
finalized.

## Gates

| Gate | Result | Artifact |
| --- | --- | --- |
| Selected Python/source tests | PASS: `47 passed in 0.41s` | `/tmp/sycl_profile_gates_20260704_021951/pytest.log` |
| Build `test-sycl-timeline test-sycl-kernel-profiler llama-bench` | PASS | `/tmp/sycl_profile_gates_20260704_021951/build.log` |
| CTest `test-sycl-timeline` and `test-sycl-kernel-profiler` | PASS: `2/2` | `/tmp/sycl_profile_gates_20260704_021951/ctest.log` |
| GPT-OSS decode profile script dry-run | PASS | `/tmp/sycl_profile_gates_20260704_021951/dry-run.log` |

## Python tests

Command:

```bash
python3 -m pytest \
  tests/test-sycl-timeline-parser.py \
  tests/test-sycl-kernel-profiler-context-source.py \
  tests/test-sycl-kernel-profiler-source-mmvq.py \
  tests/test-sycl-kernel-profiler-source-binbcast.py \
  tests/test-sycl-kernel-profiler-source-attention-ops.py \
  tests/test-sycl-kernel-profiler-source-row-mem-ops.py \
  tests/test-sycl-kernel-profiler-source-unified.py \
  tests/test-sycl-decode-timeline-profile-script.py -q
```

Result:

```text
47 passed in 0.41s
```

## Build and CTest

Commands:

```bash
set +u
source /opt/intel/oneapi/setvars.sh --force
set -u
./scripts/sycl-build.sh test-sycl-timeline test-sycl-kernel-profiler llama-bench
ctest --test-dir build -R 'test-sycl-(timeline|kernel-profiler)' -V
```

Results:

```text
Build succeeded.
100% tests passed, 0 tests failed out of 2
```

The build emitted existing SYCL compiler warnings, but no new gate failure.

## Dry-run script contract

Command:

```bash
./scripts/sycl-gptoss-decode-timeline-profile.sh \
  --dry-run \
  --out-root /tmp/sycl_profile_gates_20260704_021951/dry-run-artifacts \
  --device-selector level_zero:1
```

Dry-run output included the expected artifact parse commands:

```text
python3 scripts/parse-sycl-timeline.py /tmp/sycl_profile_gates_20260704_021951/dry-run-artifacts/sycl-timeline.json >/tmp/sycl_profile_gates_20260704_021951/dry-run-artifacts/timeline.parse
python3 scripts/parse-sycl-timeline.py --top-gaps 20 --top-host-gap-overlaps 40 /tmp/sycl_profile_gates_20260704_021951/dry-run-artifacts/sycl-timeline.json >/tmp/sycl_profile_gates_20260704_021951/dry-run-artifacts/timeline.gaps.parse
python3 scripts/parse-sycl-kernel-profile.py /tmp/sycl_profile_gates_20260704_021951/dry-run-artifacts/sycl-kernels.csv >/tmp/sycl_profile_gates_20260704_021951/dry-run-artifacts/kernels.parse
```

## Conclusion

Safe integration gates pass for the decode profiling-completeness implementation
wave. This validates parser/source/build wiring only. The next task is lead-only
B50 GPT-OSS profiling validation; that task is the first one allowed to execute
real GPU/model workloads.
