#!/usr/bin/env python3
"""Writes the prompt files for the expert-routing capture (llama.cpp-05mh).

Three prompt sets, each file one chat turn in the model's ChatML template
(tokenizer.chat_template of Qwen3.8-Flash-Next: <|im_start|>role\\n...<|im_end|>,
generation prompt `<|im_start|>assistant\\n<think>\\n`, no BOS), so llama-moe-trace
reads them as raw text with special tokens parsed:

  code-0..3   short programming requests in four languages   (~100-200 tokens)
  chat-0..3   short everyday conversation                    (~50-150 tokens)
  long-0..1   an ~8K-token document from this repository and a question about it

The long prompts are cut from files of this checkout so they need no download;
they are deterministic for a given tree. The cut is by characters: check the
prompt token count llama-moe-trace prints (the tool refuses a prompt that does not
fit n_ctx) and adjust --long-scale if it is far from 8192.

    python3 examples/moe-trace/make-prompts.py OUTDIR [--long-scale 1.0]
"""
import argparse
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]

CODE = [
    "Write a Python class LRUCache with get(key) and put(key, value) in O(1) using only the "
    "standard library. Include type hints, a capacity check that raises ValueError for "
    "capacity < 1, and three pytest tests: eviction order, update-moves-to-front, and a miss.",
    "This C function is meant to copy a NUL-terminated string into a fixed buffer but it "
    "sometimes corrupts the stack. Find every bug and give a corrected version.\n\n"
    "void set_name(char *dst, const char *src) {\n    int n = strlen(src);\n"
    "    for (int i = 0; i <= n; i++) dst[i] = src[i];\n}\n\n"
    "char buf[8];\nset_name(buf, user_input);",
    "Given tables orders(id, customer_id, total, created_at) and customers(id, name, country), "
    "write a PostgreSQL query returning, for each country, the top 3 customers by total spend "
    "in 2025 with their share of that country's spend. Explain the window functions you use "
    "and what index would help.",
    "In Rust, implement a generic trait Shape with area() and perimeter(), implement it for "
    "Circle and Rectangle, then write a function that takes a Vec<Box<dyn Shape>> and returns "
    "the shape with the largest area. Explain why the Box<dyn Shape> is needed here.",
]

CHAT = [
    "I have four days in Lisbon in November and I like food markets and quiet museums more "
    "than nightlife. Can you sketch an itinerary and tell me what the weather is usually like?",
    "Why did the Western Roman Empire fall while the Eastern one lasted another thousand years? "
    "Give me the strongest two or three explanations and say where historians disagree.",
    "I want to make a vegetarian version of beef bourguignon for six people, one of whom is "
    "allergic to mushrooms. What should I use instead and how do the cooking times change?",
    "I've been a backend developer for six years and I'm thinking about moving into engineering "
    "management. What should I try first, before committing, to find out if I'd like it?",
]

# (name, file, text to start at or None, characters, question). The cut is by
# characters, ~4.2 per token for the prose and ~3.2 for the code, for ~8K tokens.
LONG = [
    ("long-0", "docs/backend/SYCL.md", None, 34000,
     "Summarise the build and run options described above that matter for a Linux machine "
     "with an Intel GPU, and list the pitfalls the text warns about."),
    ("long-1", "src/llama-graph.cpp", "ggml_tensor * llm_graph_context::build_moe_ffn(", 26000,
     "Explain how the mixture-of-experts feed-forward block in the code above selects and "
     "weights experts, step by step, and mention each place the selection can be customised."),
]


def chatml(user):
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def read_text(rel):
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--long-scale", type=float, default=1.0,
                    help="multiply the long prompts' character counts")
    a = ap.parse_args()
    out = pathlib.Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(CODE):
        (out / f"code-{i}.txt").write_text(chatml(p), encoding="utf-8")
    for i, p in enumerate(CHAT):
        (out / f"chat-{i}.txt").write_text(chatml(p), encoding="utf-8")
    for name, rel, start, chars, question in LONG:
        text = read_text(rel)
        if start is not None:
            text = text[text.index(start):]
        body = f"### {rel}\n\n{text}"[:int(chars * a.long_scale)]
        (out / f"{name}.txt").write_text(chatml(f"{body}\n\n---\n\n{question}"), encoding="utf-8")
    for f in sorted(out.glob("*.txt")):
        print(f"{f.name}\t{f.stat().st_size} bytes")


if __name__ == "__main__":
    main()
