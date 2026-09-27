#!/usr/bin/env python3
"""Gate for parse-sycl-sched-census.py, the parser behind the qwen4exp op census.

The census (docs/backend/sycl-qwen4exp-op-census.md) is read from the
GGML_SCHED_DEBUG=2 dump that ggml_backend_sched_print_assignments() writes
(ggml/src/ggml-backend.cpp).  Its whole value is the CPU splits it finds, so the
failure that matters is a parser that reports none: an empty census reads as
"no gaps".  This file gates that failure from four sides:

  * **A planted split is counted.**  The synthetic log below holds two graph
    dumps with CPU splits planted in known places.  The per-op CPU-split counts
    must come out exactly as planted; a parser that ignores split headers, or
    attributes every node to the first split, fails here.  This is the RED.

  * **An empty or header-only log is VOID, not clean.**  A log with no
    "## SPLIT" line (GGML_SCHED_DEBUG unset, or the DEBUG lines dropped by the
    common_log verbosity threshold) and a log with split headers but no node
    lines (GGML_SCHED_DEBUG=1) must both exit 2.

  * **The prediction diff sees all three outcomes.**  AGREE, DISAGREE and an
    unpredicted CPU node must each be reported, so a diff that stopped comparing
    anything cannot pass.

  * **The checked-in qwen4exp prediction names real ops.**  Every rule's op
    must be a name ggml actually prints (read from ggml/src/ggml.c), otherwise a
    typo'd rule matches nothing and reads as "op absent from the graph".

The log lines are generated with the exact printf formats of
ggml_backend_sched_print_assignments(), including its fixed-width truncation
("%10.10s" op, "%20.20s" name), the per-source fragments it logs separately,
and one line torn by an unrelated log entry landing mid-line.

Plain script, not pytest: run with `python3 tests/<this file>`.
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PARSER = ROOT / "scripts" / "parse-sycl-sched-census.py"
PREDICTION = ROOT / "scripts" / "sycl-qwen4exp-op-prediction.json"
GGML_C = ROOT / "ggml" / "src" / "ggml.c"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"FAIL: cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules[cls.__module__]
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CENSUS = load_module(PARSER, "parse_sycl_sched_census")

N_EMBD = 2560


def fmt_size(nbytes: int) -> str:
    # ggml-backend.cpp fmt_size(): integer M above 1 MiB, integer K below
    if nbytes >= 1024 * 1024:
        return "%dM" % (nbytes // 1024 // 1024)
    return "%dK" % (nbytes // 1024)


def split_header(i: int, backend: str, inputs: list[tuple[str, int]]) -> list[str]:
    # print_assignments logs the header, ": ", each input and the newline as
    # separate GGML_LOG_DEBUG calls; the concatenation is what reaches stderr
    out = "\n## SPLIT #%d: %s # %d inputs" % (i, backend, len(inputs))
    for j, (name, size) in enumerate(inputs):
        if j == 0:
            out += ": "
        out += "[%s (%5.5s)] " % (name, fmt_size(size))
    return (out + "\n").splitlines()


def node_line(i: int, op: str, name: str, size: int, backend: str, srcs: list[tuple[str, int, str]]) -> str:
    out = "node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:" % (
        i, op, name, fmt_size(size), backend, "", 1, 1)
    for sname, ssize, sbackend in srcs:
        out += " %20.20s (%5.5s) [%5.5s %8.8s]" % (sname, fmt_size(ssize), sbackend, "")
    return out


def build_dump(n_tokens: int, torn: bool) -> list[str]:
    """One dump of a two-layer toy graph with CPU islands planted.

    Splits:  #0 CPU   GET_ROWS model.input_embed                 (by design)
             #1 SYCL0 RMS_NORM, MUL_MAT (layer 0)
             #2 CPU   UNARY SOFTPLUS (layer 0)                   planted gap
             #3 SYCL0 MUL, GATED_DELTA_NET, MUL_MAT (layer 0)
             #4 CPU   UNARY SOFTPLUS (layer 1), TOP_K (layer 1)  planted gap
             #5 SYCL0 MUL_MAT (layer 1)
             #6 CPU   MUL_MAT (layer 1)                          planted DISAGREE
             #7 CPU   CONCAT (layer 1)                           planted UNPREDICTED
             #8 SYCL0 MUL_MAT result_output
    """
    act = N_EMBD * 4 * n_tokens
    lines: list[str] = []
    lines += split_header(0, "CPU", [])
    lines.append(node_line(0, "GET_ROWS", "model.input_embed", act, "CPU",
                           [("token_embd.weight", 680 * 1024 * 1024, "CPU"), ("inp_tokens", 4 * n_tokens, "CPU")]))
    lines += split_header(1, "SYCL0", [("model.input_embed", act)])
    lines.append(node_line(1, "RMS_NORM", "norm-0", act, "SYCL0", [("model.input_embed", act, "SYCL0")]))
    lines.append(node_line(2, "MUL_MAT", "alpha-0", 48 * 4 * n_tokens, "SYCL0",
                           [("blk.0.ssm_alpha.weight", 130 * 1024, "SYCL0"), ("norm-0", act, "SYCL0")]))
    lines += split_header(2, "CPU", [("alpha-0", 48 * 4 * n_tokens)])
    lines.append(node_line(3, "SOFTPLUS", "a_softplus-0", 48 * 4 * n_tokens, "CPU", [("alpha-0", 48 * 4 * n_tokens, "CPU")]))
    lines += split_header(3, "SYCL0", [("a_softplus-0", 48 * 4 * n_tokens)])
    lines.append(node_line(4, "MUL", "gate-0", 48 * 4 * n_tokens, "SYCL0",
                           [("a_softplus-0", 48 * 4 * n_tokens, "SYCL0"), ("blk.0.ssm_a", 192, "SYCL0")]))
    gdn = node_line(5, "GATED_DELTA_NET", "__fgdn__-0", 4 * 1024 * 1024, "SYCL0",
                    [("q_conv_predelta-0", 8192, "SYCL0"), ("k_conv_predelta-0", 8192, "SYCL0")])
    if torn:
        # an unrelated log entry lands between the node head and its first
        # source fragment, as a concurrent logger does in real captures
        head = gdn[: gdn.index(":", gdn.index("use=")) + 1]
        tail = gdn[len(head):]
        lines.append(head + "[UNIFIED-CACHE] unrelated line from another thread")
        lines.append(tail)
    else:
        lines.append(gdn)
    lines.append(node_line(6, "MUL_MAT", "blk.0.attn_output (cont)", act, "SYCL0",
                           [("blk.0.ssm_out.weight", 16 * 1024 * 1024, "SYCL0"), ("final_output-0", 6144 * 4 * n_tokens, "SYCL0")]))
    lines += split_header(4, "CPU", [("alpha-1", 48 * 4 * n_tokens), ("indexer_score_tokens-1", 4096 * 4 * n_tokens)])
    lines.append(node_line(7, "SOFTPLUS", "a_softplus-1", 48 * 4 * n_tokens, "CPU", [("alpha-1", 48 * 4 * n_tokens, "CPU")]))
    lines.append(node_line(8, "TOP_K", "indexer_top_k-1", 2051 * 4 * n_tokens, "CPU",
                           [("indexer_score_tokens-1", 4096 * 4 * n_tokens, "CPU")]))
    lines += split_header(5, "SYCL0", [("a_softplus-1", 48 * 4 * n_tokens)])
    lines.append(node_line(9, "MUL_MAT", "attn_output-1", act, "SYCL0",
                           [("blk.1.attn_output.weight", 16 * 1024 * 1024, "SYCL0"), ("x-1", act, "SYCL0")]))
    lines += split_header(6, "CPU", [("x-1", act)])
    lines.append(node_line(10, "MUL_MAT", "ffn_moe_logits-1", 512 * 4 * n_tokens, "CPU",
                           [("blk.1.ffn_gate_inp.weight", 5 * 1024 * 1024, "CPU"), ("x-1", act, "CPU")]))
    lines += split_header(7, "CPU", [])
    lines.append(node_line(11, "CONCAT", "conv_input-1", 13 * 10240 * 4, "CPU", [("conv_state_at-1", 3 * 10240 * 4, "CPU")]))
    lines += split_header(8, "SYCL0", [("ffn_moe_logits-1", 512 * 4 * n_tokens)])
    lines.append(node_line(12, "MUL_MAT", "result_output", 248320 * 4 * n_tokens, "SYCL0",
                           [("output.weight", 675 * 1024 * 1024, "SYCL0"), ("result_norm", act, "SYCL0")]))
    return lines


def write(tmp: Path, name: str, lines: list[str]) -> Path:
    p = tmp / name
    p.write_text("\n".join(lines) + "\n")
    return p


PREDICTION_FIXTURE = {
    "model": "synthetic",
    "rules": [
        {"id": "R-INPUT", "op": "GET_ROWS", "name": r"^model\.input_embed$", "expect": "CPU", "verdict": "BY-DESIGN",
         "cite": "fixture"},
        {"id": "R-SOFTPLUS", "op": "SOFTPLUS", "expect": "CPU", "verdict": "UNSUPPORTED", "cite": "fixture"},
        {"id": "R-TOPK", "op": "TOP_K", "expect": "CPU", "verdict": "UNSUPPORTED", "cite": "fixture"},
        {"id": "R-GDN", "op": "GATED_DELTA_NET", "expect": "SYCL", "verdict": "SUSPECT", "cite": "fixture"},
        {"id": "R-OUTPUT", "op": "MUL_MAT", "src0": r"^output\.weight$", "expect": "SYCL", "verdict": "SUPPORTED",
         "cite": "fixture"},
        {"id": "R-MUL_MAT", "op": "MUL_MAT", "expect": "SYCL", "verdict": "SUPPORTED", "cite": "fixture"},
        {"id": "R-ELTWISE", "op": "MUL", "expect": "SYCL", "verdict": "SUPPORTED", "cite": "fixture"},
        {"id": "R-NORM", "op": "RMS_NORM", "expect": "SYCL", "verdict": "SUPPORTED", "cite": "fixture"},
        {"id": "R-ABSENT", "op": "SSM_CONV", "expect": "SYCL", "verdict": "SUPPORTED", "cite": "fixture"},
    ],
}

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)


def run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(PARSER), *args], capture_output=True, text=True)


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # a reserve-shaped dump (n_tokens=512) followed by a decode dump
        # (n_tokens=1) with a torn node line, then unrelated log noise
        log_lines = ["llama_context: constructing"] + build_dump(512, torn=False) + \
                    ["print_info: n_ctx = 4096"] + build_dump(1, torn=True) + ["llama_perf_context_print: done"]
        log = write(tmp, "census.err", log_lines)

        # positive control on the fixture itself: the planted CPU splits must be
        # present as raw text, or every count below is vacuous
        raw_cpu_headers = sum(1 for line in log_lines if re.search(r"## SPLIT #\d+: CPU ", line))
        check(raw_cpu_headers == 10, f"fixture sanity: expected 10 raw CPU split headers, found {raw_cpu_headers}")

        dumps = CENSUS.parse_dumps(log.read_text().splitlines())
        check(len(dumps) == 2, f"expected 2 graph dumps, parsed {len(dumps)}")
        if len(dumps) == 2:
            check([d.n_tokens(N_EMBD) for d in dumps] == [512, 1],
                  f"n_tokens classes: expected [512, 1], got {[d.n_tokens(N_EMBD) for d in dumps]}")
            for d in dumps:
                c = CENSUS.census(d)
                tag = f"n_tokens={d.n_tokens(N_EMBD)}"
                check(c.n_splits == 9, f"{tag}: expected 9 splits, got {c.n_splits}")
                check(c.n_cpu_splits == 5, f"{tag}: expected 5 CPU splits, got {c.n_cpu_splits}")
                check(c.n_nodes == 13, f"{tag}: expected 13 non-view nodes, got {c.n_nodes}")
                # the planted gap: SOFTPLUS in two different CPU splits
                check(c.cpu_splits_by_op.get("SOFTPLUS", 0) == 2,
                      f"{tag}: SOFTPLUS must sit in 2 CPU splits, got {c.cpu_splits_by_op.get('SOFTPLUS', 0)}")
                check(c.cpu_splits_by_op.get("TOP_K", 0) == 1,
                      f"{tag}: TOP_K must sit in 1 CPU split, got {c.cpu_splits_by_op.get('TOP_K', 0)}")
                check(c.nodes_by_op_backend.get(("MUL_MAT", "SYCL0"), 0) == 4 and
                      c.nodes_by_op_backend.get(("MUL_MAT", "CPU"), 0) == 1,
                      f"{tag}: MUL_MAT per backend wrong: {dict(c.nodes_by_op_backend)}")
                # the fixed-width op field truncates GATED_DELTA_NET; the torn
                # line in the decode dump must still yield this node
                check(c.nodes_by_op_backend.get(("GATED_DELT", "SYCL0"), 0) == 1,
                      f"{tag}: GATED_DELTA_NET node lost: {dict(c.nodes_by_op_backend)}")
                check(c.split_inputs == 8, f"{tag}: expected 8 split inputs, got {c.split_inputs}")

            diff = CENSUS.diff_prediction(CENSUS.census(dumps[1]), PREDICTION_FIXTURE)
            status = {row.rule_id: row.status for row in diff.rules}
            check(status.get("R-SOFTPLUS") == "AGREE", f"R-SOFTPLUS: expected AGREE, got {status.get('R-SOFTPLUS')}")
            check(status.get("R-GDN") == "AGREE", f"R-GDN: expected AGREE, got {status.get('R-GDN')}")
            check(status.get("R-MUL_MAT") == "DISAGREE", f"R-MUL_MAT: expected DISAGREE, got {status.get('R-MUL_MAT')}")
            # a src0 rule claims its node before the generic rule sees it
            on_expected = {row.rule_id: row.on_expected for row in diff.rules}
            check(status.get("R-OUTPUT") == "AGREE" and on_expected.get("R-OUTPUT") == 1,
                  f"R-OUTPUT (src0 rule): expected AGREE on 1 node, got {status.get('R-OUTPUT')}/{on_expected.get('R-OUTPUT')}")
            check(on_expected.get("R-MUL_MAT") == 3,
                  f"R-MUL_MAT: expected 3 nodes on SYCL after R-OUTPUT, got {on_expected.get('R-MUL_MAT')}")
            check(status.get("R-ABSENT") == "ABSENT", f"R-ABSENT: expected ABSENT, got {status.get('R-ABSENT')}")
            unpredicted = {(u.op, u.backend) for u in diff.unpredicted}
            check(("CONCAT", "CPU") in unpredicted, f"unpredicted CPU CONCAT not reported: {unpredicted}")

        pred = write(tmp, "pred.json", [json.dumps(PREDICTION_FIXTURE)])

        # CLI: a satisfied control exits 0, an unsatisfied one exits 2 (VOID)
        ok = run_cli([str(log), "--prediction", str(pred), "--require", "SOFTPLUS=CPU:2"])
        check(ok.returncode == 0, f"satisfied --require must exit 0, got {ok.returncode}: {ok.stderr.strip()}")
        check("| SOFTPLUS" in ok.stdout, "census table missing the SOFTPLUS row")
        bad = run_cli([str(log), "--require", "SOFTPLUS=CPU:3"])
        check(bad.returncode == 2, f"unsatisfied --require must exit 2, got {bad.returncode}")
        any_ok = run_cli([str(log), "--require", "MUL_MAT=ANY:5"])
        check(any_ok.returncode == 0, f"--require ANY must count every backend (4 SYCL0 + 1 CPU), got {any_ok.returncode}")
        bad_backend = run_cli([str(log), "--require", "SOFTPLUS=SYCL:1"])
        check(bad_backend.returncode == 2, f"--require on the wrong backend must exit 2, got {bad_backend.returncode}")

        # --strict turns a DISAGREE / unpredicted CPU node into a failure
        strict = run_cli([str(log), "--prediction", str(pred), "--strict"])
        check(strict.returncode == 4, f"--strict with a DISAGREE must exit 4, got {strict.returncode}")

        # --tensor-types resolves the printer's 20-char-truncated src[0] name
        types = CENSUS.load_tensor_types(write(tmp, "types.txt", [
            "T blk.0.ssm_alpha.weight (2560, 48) 8",
            "T blk.3.indexer.q_proj.weight (2560, 512) 30",
        ]))
        check(CENSUS.resolve_type("blk.0.ssm_alpha.weig", types) == "Q8_0",
              f"truncated src0 type: expected Q8_0, got {CENSUS.resolve_type('blk.0.ssm_alpha.weig', types)}")
        check(CENSUS.resolve_type("blk.3.indexer.q_proj", types) == "BF16",
              f"BF16 id 30: expected BF16, got {CENSUS.resolve_type('blk.3.indexer.q_proj', types)}")

        # VOID: no dump at all (sched debug off, or DEBUG dropped by -lv < 5)
        empty = write(tmp, "empty.err", ["llama_context: constructing", "main: done"])
        r = run_cli([str(empty)])
        check(r.returncode == 2, f"a log with no SPLIT dump must exit 2 (VOID), got {r.returncode}")

        # VOID: GGML_SCHED_DEBUG=1 prints split headers but no node lines
        headers_only = write(tmp, "headers.err", [l for l in log_lines if not l.lstrip().startswith("node #")])
        r = run_cli([str(headers_only)])
        check(r.returncode == 2, f"split headers without node lines must exit 2 (VOID), got {r.returncode}")

    # the checked-in prediction must be well-formed and name ops ggml prints
    printed = CENSUS.ggml_printed_op_names(GGML_C.read_text())
    check(len(printed) > 100, f"read only {len(printed)} op names from ggml.c; the reader is broken")
    pred_real = json.loads(PREDICTION.read_text())
    rules = pred_real.get("rules", [])
    check(len(rules) >= 20, f"qwen4exp prediction has only {len(rules)} rules")
    for rule in rules:
        rid = rule.get("id", "?")
        check(rule.get("op") in printed, f"{rid}: op {rule.get('op')!r} is not a name ggml prints")
        check(rule.get("expect") in ("SYCL", "CPU"), f"{rid}: expect must be SYCL or CPU")
        check(rule.get("verdict") in CENSUS.VERDICTS, f"{rid}: verdict {rule.get('verdict')!r} unknown")
        check(bool(rule.get("cite")), f"{rid}: no cite")
        if "name" in rule:
            try:
                re.compile(rule["name"])
            except re.error as e:
                check(False, f"{rid}: bad name regex: {e}")

    if failures:
        for f in failures:
            print("FAIL:", f)
        return 1
    print("PASS: sched census parser")
    return 0


if __name__ == "__main__":
    sys.exit(main())
