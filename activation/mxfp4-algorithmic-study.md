# MXFP4 Algorithmic Route Quality/Speed Study

This report documents the offline gate for future lead-owned captures of MXFP4
algorithmic routes such as top-k pruning, approximate SwiGLU, and speculative
batching. Workers do not run model prompts, perplexity, or GPU probes for this
study.

Count smoke alone is insufficient for algorithmic routes. A route can still
print a short deterministic count answer while perturbing hidden activations,
top-token logits, or perplexity enough to invalidate the optimization.

The offline gate is:

```bash
python3 scripts/sycl-mxfp4-algorithmic-study.py <capture.json>
```

The capture must be exactly one JSON object with:

- `baseline_output`
- `candidate_output`
- `baseline_logits_top10`
- `candidate_logits_top10`
- `baseline_ms_per_token`
- `candidate_ms_per_token`

The script reports `relative_l2`, `top10_logit_mae`, `speed_ceiling_tok_s`, and
`recommendation`. It kills candidate routes when relative output-vector L2 is
greater than `1e-3` or top-10 logit MAE is greater than `1e-2`.

This does not prove model quality by itself; it creates the fail-closed offline
quality/speed ceiling check for future captures owned by the lead validation
flow.
