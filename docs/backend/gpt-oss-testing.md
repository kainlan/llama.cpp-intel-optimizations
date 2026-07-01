# GPT-OSS Testing & Prompt Template Rationale

This document holds the detailed rationale and source provenance for the
GPT-OSS correctness gate. The operational rule and the canonical gate command
live in `CLAUDE.md` ("Verification Commands & Correctness Gates"); this file is
the "why" behind it.

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

Expected output starts with `: 1, 2, 3, 4, 5`. The leading colon is normal for
this CLI/Harmony rendering. `llama-bench` is valid for PP/TG throughput, but it
does not prove chat-template correctness; use the gate above before trusting
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
