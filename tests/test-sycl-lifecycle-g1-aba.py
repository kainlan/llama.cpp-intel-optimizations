#!/usr/bin/env python3
"""Canonical G1 launcher; only paths, physical UUID, and token oracle are external."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

SKIP = 77
MODEL_SHA256 = {
    "A": "66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739",
    "B": "270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d",
    "A-shared": "66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739",
}
PROMPT = "1, 2, 3, 4, 5,"
SEED = 42
TEMPERATURE = 0
N_PREDICT = 8
PROCESS_TIMEOUT_SECONDS = 300
ONEAPI_SELECTOR = "level_zero:0,1"
GGML_LOGICAL_SELECTOR = "1"
ROUTING_ENV_PREFIXES = ("GGML_", "SYCL_", "ONEAPI_", "UR_", "ZE_", "ZES_", "LLAMA_ARG_")
ROUTING_ENV_NAMES = {
    "CUDA_DEVICE_ORDER",
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "VK_ICD_FILENAMES",
    "VK_DRIVER_FILES",
}
CANONICAL_RUNTIME_ENV = {
    "ONEAPI_DEVICE_SELECTOR": ONEAPI_SELECTOR,
    "GGML_SYCL_DEVICE": GGML_LOGICAL_SELECTOR,
}


def is_backend_routing_env(name):
    return name in ROUTING_ENV_NAMES or name.startswith(ROUTING_ENV_PREFIXES)


def canonical_environment(source):
    env = {key: value for key, value in source.items() if not is_backend_routing_env(key)}
    env.update(CANONICAL_RUNTIME_ENV)
    if any(
        is_backend_routing_env(key) and CANONICAL_RUNTIME_ENV.get(key) != value
        for key, value in env.items()
    ):
        raise RuntimeError("failed to sanitize backend-routing environment")
    return env


class PrerequisiteError(Exception):
    pass


def unavailable(message, strict):
    print(("FAIL" if strict else "SKIP") + f": {message}", file=sys.stderr)
    return 1 if strict else SKIP


def token_hash(tokens):
    return hashlib.sha256((",".join(str(x) for x in tokens) + "\n").encode()).hexdigest()


def _is_sha(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(c in "0123456789abcdef" for c in value)
    )


def _is_uuid(value):
    if not isinstance(value, str) or value != value.lower() or len(value) != 36:
        return False
    return value[8] == value[13] == value[18] == value[23] == "-" and all(
        c in "0123456789abcdef" for i, c in enumerate(value) if i not in (8, 13, 18, 23)
    )


def load_contract(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PrerequisiteError(f"fixture JSON unavailable or invalid: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema") != 4:
        raise PrerequisiteError("fixture schema must be 4")
    if set(data) != {"schema", "fixtures", "device", "expected"}:
        raise PrerequisiteError("fixture may contain only schema, fixtures, device, and expected")
    fixtures, device, expected = (data.get(key) for key in ("fixtures", "device", "expected"))
    if not all(isinstance(value, dict) for value in (fixtures, device, expected)):
        raise PrerequisiteError("fixtures, device, and expected must be objects")

    errors = []
    if set(fixtures) != {"A", "B", "A-shared"}:
        errors.append("fixtures must contain exactly A, B, and A-shared")
    else:
        for name in ("A", "B", "A-shared"):
            spec = fixtures[name]
            if (
                not isinstance(spec, dict)
                or set(spec) != {"path"}
                or not isinstance(spec.get("path"), str)
                or not spec["path"]
            ):
                errors.append(f"fixtures.{name}.path")
        if all(isinstance(fixtures[name], dict) and isinstance(fixtures[name].get("path"), str) for name in fixtures):
            if len({fixtures[name]["path"] for name in fixtures}) != 3:
                errors.append("A, B, and A-shared paths must be distinct")

    if set(device) != {"physical_uuid"} or not _is_uuid(device.get("physical_uuid")):
        errors.append("device.physical_uuid")
    if set(expected) != {"A", "B"}:
        errors.append("expected must contain exactly A and B")
    else:
        for name in ("A", "B"):
            oracle = expected[name]
            if (
                not isinstance(oracle, dict)
                or set(oracle) != {"tokens", "token_sha256"}
                or not isinstance(oracle.get("tokens"), list)
                or len(oracle.get("tokens", [])) != N_PREDICT
                or not all(isinstance(token, int) and not isinstance(token, bool) for token in oracle.get("tokens", []))
                or not _is_sha(oracle.get("token_sha256"))
            ):
                errors.append(f"expected.{name}.tokens/token_sha256")
    if errors:
        raise PrerequisiteError("incomplete/invalid canonical fixture: " + ", ".join(errors))
    return data


def model_paths(contract, fixture_path):
    paths = {}
    for name in ("A", "B", "A-shared"):
        path = Path(contract["fixtures"][name]["path"]).expanduser()
        if not path.is_absolute():
            path = fixture_path.parent / path
        if not path.is_file():
            raise PrerequisiteError(f"fixture {name} is missing: {path}")
        digest_state = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest_state.update(chunk)
        digest = digest_state.hexdigest()
        if digest != MODEL_SHA256[name]:
            raise RuntimeError(f"fixture {name} sha256 {digest} != canonical {MODEL_SHA256[name]}")
        paths[name] = path
    if len({path.resolve() for path in paths.values()}) != 3:
        raise PrerequisiteError("A, B, and A-shared must resolve to three distinct files")
    return paths


def run(argv=None):
    parser = argparse.ArgumentParser(description="Lab: --strict --fixture /path/g1.json RUNNER")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--fixture", type=Path, default=Path(__file__).with_name("sycl-lifecycle-fixtures.json"))
    parser.add_argument("runner")
    args = parser.parse_args(argv)
    try:
        contract = load_contract(args.fixture)
        runner = Path(args.runner)
        if not runner.is_file() or not os.access(runner, os.X_OK):
            raise PrerequisiteError(f"runner is not executable: {runner}")
        models = model_paths(contract, args.fixture)
    except PrerequisiteError as exc:
        return unavailable(str(exc), args.strict)
    except RuntimeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    common = [
        str(runner),
        "--model-a",
        str(models["A"]),
        "--model-b",
        str(models["B"]),
        "--model-a-shared",
        str(models["A-shared"]),
        "--prompt",
        PROMPT,
        "--seed",
        str(SEED),
        "--temp",
        str(TEMPERATURE),
        "--n-predict",
        str(N_PREDICT),
    ]
    env = canonical_environment(os.environ)
    results = {}
    for label, sequence in (("A-reference", "A"), ("B-reference", "B"), ("ABA", "A,B,A")):
        try:
            proc = subprocess.run(
                common + ["--run", sequence],
                env=env,
                text=True,
                capture_output=True,
                timeout=PROCESS_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            print(f"FAIL: {label} timed out", file=sys.stderr)
            return 1
        if proc.returncode == SKIP:
            return unavailable(f"{label} reports unsupported device/UUID/offload", args.strict)
        if proc.returncode != 0:
            print(f"FAIL: {label} exited {proc.returncode}\n{proc.stderr}", file=sys.stderr)
            return 1
        try:
            result = json.loads(proc.stdout)
        except ValueError as exc:
            print(f"FAIL: {label} emitted invalid JSON: {exc}", file=sys.stderr)
            return 1
        if (
            not isinstance(result, dict)
            or not _is_uuid(result.get("device_uuid"))
            or not isinstance(result.get("runs"), list)
        ):
            print(f"FAIL: {label} output has invalid nested types", file=sys.stderr)
            return 1
        results[label] = result

    expected_uuid = contract["device"]["physical_uuid"]
    if (
        any(result["device_uuid"] != expected_uuid for result in results.values())
        or len({result["device_uuid"] for result in results.values()}) != 1
    ):
        print("FAIL: process UUIDs differ from the canonical physical UUID", file=sys.stderr)
        return 1
    expected_runs = {"A-reference": ["A"], "B-reference": ["B"], "ABA": ["A", "B", "A"]}
    for process, labels in expected_runs.items():
        runs = results[process]["runs"]
        if not all(isinstance(item, dict) for item in runs) or [item.get("model") for item in runs] != labels:
            print(f"FAIL: {process} did not execute canonical sequence", file=sys.stderr)
            return 1
        for item, model in zip(runs, labels):
            tokens, oracle = item.get("tokens"), contract["expected"][model]
            if (
                not isinstance(tokens, list)
                or not all(isinstance(token, int) and not isinstance(token, bool) for token in tokens)
                or tokens != oracle["tokens"]
                or token_hash(tokens) != oracle["token_sha256"]
            ):
                print(f"FAIL: {process}/{model} differs from canonical token oracle", file=sys.stderr)
                return 1
    print("PASS: canonical G1 A/B references and A/B/A match hashes, tokens, offload device, and physical UUID")
    return 0


if __name__ == "__main__":
    sys.exit(run())
