#!/usr/bin/env python3
"""Canonical G1 acceptance launcher.

The checked-in JSON is the only source of model, generation, device, and oracle
values.  In normal mode unavailable/incomplete acceptance fixtures are a CTest
skip (77); --strict turns every missing prerequisite into a failure.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

SKIP = 77
PLACEHOLDER = "REPLACE_ON_G1_"


def unavailable(message: str, strict: bool) -> int:
    print(("FAIL" if strict else "SKIP") + f": {message}", file=sys.stderr)
    return 1 if strict else SKIP


def token_hash(tokens):
    # Deliberately language-independent canonical representation.
    return hashlib.sha256((",".join(str(x) for x in tokens) + "\n").encode()).hexdigest()


def load_contract(path: Path, strict: bool):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"fixture JSON unavailable or invalid: {exc}")
    required = [
        ("schema", data.get("schema")),
        ("fixtures.A.path", data.get("fixtures", {}).get("A", {}).get("path")),
        ("fixtures.A.sha256", data.get("fixtures", {}).get("A", {}).get("sha256")),
        ("fixtures.B.path", data.get("fixtures", {}).get("B", {}).get("path")),
        ("fixtures.B.sha256", data.get("fixtures", {}).get("B", {}).get("sha256")),
        ("fixtures.A-shared.path", data.get("fixtures", {}).get("A-shared", {}).get("path")),
        ("fixtures.A-shared.sha256", data.get("fixtures", {}).get("A-shared", {}).get("sha256")),
        ("generation.prompt", data.get("generation", {}).get("prompt")),
        ("generation.seed", data.get("generation", {}).get("seed")),
        ("generation.temperature", data.get("generation", {}).get("temperature")),
        ("generation.n_predict", data.get("generation", {}).get("n_predict")),
        ("expected.A.tokens", data.get("expected", {}).get("A", {}).get("tokens")),
        ("expected.A.token_sha256", data.get("expected", {}).get("A", {}).get("token_sha256")),
        ("expected.B.tokens", data.get("expected", {}).get("B", {}).get("tokens")),
        ("expected.B.token_sha256", data.get("expected", {}).get("B", {}).get("token_sha256")),
        ("device.oneapi_selector", data.get("device", {}).get("oneapi_selector")),
        ("device.ggml_logical_selector", data.get("device", {}).get("ggml_logical_selector")),
        ("device.physical_uuid", data.get("device", {}).get("physical_uuid")),
    ]
    missing = [name for name, value in required if value is None or value == "" or value == [] or
               (isinstance(value, str) and value.startswith(PLACEHOLDER))]
    if data.get("schema") != 2:
        missing.append("schema=2")
    if missing:
        raise RuntimeError("incomplete canonical fixture: " + ", ".join(missing))
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="fail rather than skip when acceptance prerequisites are absent")
    parser.add_argument("runner")
    parser.add_argument("build", type=Path)
    parser.add_argument("--fixture", type=Path, default=Path(__file__).with_name("sycl-lifecycle-fixtures.json"))
    args = parser.parse_args()

    try:
        contract = load_contract(args.fixture, args.strict)
    except RuntimeError as exc:
        return unavailable(str(exc), args.strict)

    runner = Path(args.runner)
    if not runner.is_file() or not os.access(runner, os.X_OK):
        return unavailable(f"runner is not executable: {runner}", args.strict)

    models = {name: args.build / spec["path"] for name, spec in contract["fixtures"].items()}
    for name, path in models.items():
        if not path.is_file():
            return unavailable(f"fixture {name} is missing: {path}", args.strict)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != contract["fixtures"][name]["sha256"]:
            print(f"FAIL: fixture {name} sha256 {digest} != canonical {contract['fixtures'][name]['sha256']}", file=sys.stderr)
            return 1

    generation = contract["generation"]
    common = [str(runner), "--model-a", str(models["A"]), "--model-b", str(models["B"]),
              "--model-a-shared", str(models["A-shared"]), "--prompt", generation["prompt"],
              "--seed", str(generation["seed"]), "--temp", str(generation["temperature"]),
              "--n-predict", str(generation["n_predict"])]
    env = os.environ.copy()
    env["ONEAPI_DEVICE_SELECTOR"] = contract["device"]["oneapi_selector"]
    env["GGML_SYCL_DEVICE"] = str(contract["device"]["ggml_logical_selector"])
    env["GGML_SYCL_LIFECYCLE_TEST_DEVICE"] = str(contract["device"]["ggml_logical_selector"])

    results = {}
    for label, sequence in (("A-reference", "A"), ("B-reference", "B"), ("ABA", "A,B,A")):
        proc = subprocess.run(common + ["--run", sequence], env=env, text=True, capture_output=True)
        if proc.returncode == SKIP:
            return unavailable(f"{label} reports unsupported device/UUID", args.strict)
        if proc.returncode != 0:
            print(f"FAIL: {label} exited {proc.returncode}\n{proc.stderr}", file=sys.stderr)
            return 1
        try:
            results[label] = json.loads(proc.stdout)
        except ValueError as exc:
            print(f"FAIL: {label} emitted invalid JSON: {exc}\n{proc.stdout}", file=sys.stderr)
            return 1

    expected_uuid = contract["device"]["physical_uuid"].lower()
    for label, result in results.items():
        if not isinstance(result, dict) or str(result.get("device_uuid", "")).lower() != expected_uuid:
            print(f"FAIL: {label} physical UUID differs from canonical device", file=sys.stderr)
            return 1
    if len({str(result.get("device_uuid", "")).lower() for result in results.values()}) != 1:
        print("FAIL: isolated and sequential runs reported different physical UUIDs", file=sys.stderr)
        return 1

    expected_runs = {"A-reference": ["A"], "B-reference": ["B"], "ABA": ["A", "B", "A"]}
    for process, labels in expected_runs.items():
        runs = results[process].get("runs", [])
        if [run.get("model") for run in runs] != labels:
            print(f"FAIL: {process} did not execute canonical {labels} sequence", file=sys.stderr)
            return 1
        for run, model in zip(runs, labels):
            tokens = run.get("tokens")
            oracle = contract["expected"][model]
            if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens) or \
                    tokens != oracle["tokens"] or token_hash(tokens) != oracle["token_sha256"]:
                print(f"FAIL: {process}/{model} tokens differ from canonical oracle", file=sys.stderr)
                return 1

    print("PASS: canonical G1 isolated A/B references and A/B/A sequence match hashes, tokens, and physical UUID")
    return 0


if __name__ == "__main__":
    sys.exit(main())
