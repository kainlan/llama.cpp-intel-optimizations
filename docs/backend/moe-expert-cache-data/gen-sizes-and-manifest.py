#!/usr/bin/env python3
"""Writes expert-sizes.csv (bytes of one expert per MoE layer, from the GGUF tensor
tables) and traces-manifest.csv (what each trace file is). Not reproducible without
the GGUFs and the trace files, which are not committed; the manifest's sha256 says
which files the curves came from.

  PYTHONPATH=gguf-py python3 gen-sizes-and-manifest.py TRACE.moetrace...
"""
import csv, hashlib, importlib.util, os, sys

SIM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "scripts", "moe-cache-sim.py")
spec = importlib.util.spec_from_file_location("moe_cache_sim", SIM)
sim = importlib.util.module_from_spec(spec)
sys.modules["moe_cache_sim"] = sim
spec.loader.exec_module(sim)

HERE = os.path.dirname(os.path.abspath(__file__))
GGUF = {
    "iq3": "/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
    "q8": "/models/Qwen3.8-Flash-Next-GGUF/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf",
}

with open(os.path.join(HERE, "expert-sizes.csv"), "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["format", "layer", "expert_bytes", "tensor_types"])
    for fmt, path in GGUF.items():
        s = sim.load_expert_sizes_gguf([path])
        for layer, nbytes in s["layer_bytes"].items():
            w.writerow([fmt, layer, nbytes, "+".join(s["types"][layer])])

with open(os.path.join(HERE, "traces-manifest.csv"), "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["id", "set", "n_prompt", "n_predict", "n_batch", "n_ubatch", "prefill_steps",
                "decode_steps", "file_bytes", "sha256"])
    for p in sorted(sys.argv[1:]):
        t = sim.read_trace(p)
        h = t.header
        raw = open(p, "rb").read()
        w.writerow([h["id"], h["set"], h["n_prompt"], h["n_predict"], h["n_batch"], h["n_ubatch"],
                    sum(1 for s in t.steps if s.phase == 0), sum(1 for s in t.steps if s.phase == 1),
                    len(raw), hashlib.sha256(raw).hexdigest()])
