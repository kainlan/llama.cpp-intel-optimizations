# GPT-OSS Testing & Prompt Template Rationale

This document holds the detailed rationale and source provenance for the
GPT-OSS correctness gate. The operational rule and the canonical gate command
live in `CLAUDE.md` ("Verification Commands & Correctness Gates"); this file is
the "why" behind it.

## Automated runner (llama.cpp-rou3)

The Mistral completion gate and this GPT-OSS chat gate are both executable
from `scripts/sycl-canonical-gates.sh` (`--gate mistral|gptoss|all`), ctest-
registered as `sycl-canonical-gates` (`LABELS "gpu;model-loading"`). That
label is NOT currently excluded by CLAUDE.md's documented safe-sweep
denylist (`-LE 'residency|mem-handle|cache'`, which contains neither `gpu`
nor `model-loading`) -- excluding it requires that denylist to grow a
`|gpu` term (tracked on the test-hygiene epic, llama.cpp-97yn; a
task-local, non-authoritative `|gpu` variant is used as this task's own
acceptance bar). Do not assume today's documented sweep skips this gate.
It fails closed on a missing binary or a build that is not actually
SYCL-linked, SKIPs (rc 77) when no SYCL device is enumerated or a model
file is missing, and never runs a model-loading binary more than once per
invocation. Its own logic (preconditions, the no-device probe, the
digit-regex pass/fail checks) is covered without a GPU by
`tests/test-sycl-canonical-gates-script.sh`, registered as
`sycl-canonical-gates-script`. Command line construction for the GPT-OSS
gate inside the script follows the post-b10630 form (no `-cnv`, `-c 4096`
pinned, digit-line-only expected output) shown in the "Canonical B50
GPT-OSS correctness gate" section immediately below, which this task
brought in line with CLAUDE.md's own post-b10630 update -- it previously
still showed the pre-b10630 `-cnv` form and a colon-prefixed expected
output the script's anchored regex does not accept.

## The rule

Use `llama-cli -cnv` so the CLI applies the model's embedded GGUF/Jinja chat
template to the prompt as a user message. Do not hand-render a raw Harmony
prompt, and do not pass `--chat-template gpt-oss` or a custom template unless the
test is explicitly about that formatter. `llama-cli --help` reports Jinja enabled
by default and `--chat-template` as a custom override whose default is the
template from model metadata.

For cross-branch regression tests, pin the Harmony `reasoning_effort` template
argument to `medium` with `--chat-template-kwargs`. The known-good B50 GPT-OSS
prompt rendered `Reasoning: medium`; pinning prevents accidental changes in
template metadata, CLI defaults, or test harness behavior from moving the prompt
while comparing backend performance. `--reasoning-format none` controls how
reasoning output is shown or hidden and is not a substitute for pinning the
template argument. The deterministic count gate deliberately uses
`--reasoning-budget 0` with hidden reasoning so the expected answer is a short
final-channel string; for normal GPT-OSS chat/server parser validation, use
llama.cpp's automatic reasoning handling instead of treating `none` as a
model-format requirement.

## Canonical B50 GPT-OSS correctness gate

Post-b10630 form (verified in CLAUDE.md and matching what
`scripts/sycl-canonical-gates.sh` runs): no `-cnv`, `-c 4096` pinned
(llama.cpp-uize -- the model's `n_ctx_train` default does not fit the B50).
An earlier revision of this section showed the pre-b10630 `-cnv` form with
no `-c` pin and a colon-prefixed expected output; that form is retired --
do not use it for new comparisons.

```bash
source /opt/intel/oneapi/setvars.sh --force
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-cli \
  -m /models/gpt-oss-20b-mxfp4.gguf -ngl 99 -c 4096 \
  -st --simple-io --no-display-prompt \
  --chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --reasoning-format none --reasoning-budget 0 \
  -p 'Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5' \
  -n 48 --seed 42 --temp 0
```

Expected output is the digit sequence alone, on its own line: `1, 2, 3, 4, 5`
(with `--no-display-prompt`, the echoed prompt -- which itself ends
`...only: 1, 2, 3, 4, 5` -- lands on the interactive `> ` line, not this
one; grepping for a leading colon or for the digits anywhere in the output
is the documented false-fail this gate's anchored `^1, 2, 3, 4, 5\s*$` regex
exists to avoid). `llama-bench` is valid for PP/TG throughput, but it does
not prove chat-template correctness; use the gate above before trusting
GPT-OSS performance numbers.

## Web and local verification (rechecked 2026-06-19)

- GPT-OSS models were trained for OpenAI's Harmony response format and should
  not be run with raw text or a generic chat format.
- OpenAI's implementation-verification guide warns that inference providers must
  map inputs to Harmony correctly; wrong prompt formatting can cause cascading
  generation issues.
- OpenAI's GPT-OSS Transformers guide says prompts should be built with the
  tokenizer chat template or `openai-harmony`.
- The OpenAI Hugging Face model card says the Transformers chat template
  automatically applies Harmony and direct `model.generate` callers must apply
  Harmony manually.
- The `openai/gpt-oss-20b` Jinja template accepts `reasoning_effort`, defaults it
  to `medium`, renders `Reasoning: medium` in the Harmony system message, renders
  the user prompt as a Harmony user message, and appends `<|start|>assistant` as
  the generation prompt.
- The llama.cpp GPT-OSS guide says `--jinja` uses the Jinja chat template
  embedded in the GGUF and that the `ggml-org/gpt-oss` GGUFs have a built-in chat
  template used by default; manual template overrides are only for known template
  bugs or specialized experiments.
- Local `llama-cli --help` confirms `--jinja` defaults to enabled and the chat
  template defaults to the one taken from model metadata.

## Sources checked 2026-06-19

- `https://developers.openai.com/cookbook/articles/openai-harmony`
- `https://developers.openai.com/cookbook/articles/gpt-oss/verifying-implementations`
- `https://developers.openai.com/cookbook/articles/gpt-oss/run-transformers`
- `https://developers.openai.com/cookbook/articles/gpt-oss/handle-raw-cot`
- `https://huggingface.co/openai/gpt-oss-20b`
- `https://huggingface.co/openai/gpt-oss-20b/blob/main/chat_template.jinja`
- `https://github.com/ggml-org/llama.cpp/discussions/15396`
