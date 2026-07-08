#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any

REQUIRED_TOP = (
    "schema_version",
    "stage",
    "artifact_root",
    "build_sha",
    "model",
    "device_selector",
    "fa",
    "moe_knobs",
    "prompt_tokens",
    "gen_tokens",
    "repeat",
    "artifacts",
)
REQUIRED_MOE = (
    "GGML_SYCL_MOE_PHASE_MATERIALIZE",
    "GGML_SYCL_MOE_PHASE_BULK_XMX",
    "GGML_SYCL_MOE_DOWN_SUM_DIRECT",
)


class ManifestError(ValueError):
    pass


def require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"invalid integer field {name}")
    return value


def require_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"invalid string field {name}")
    return value


def load_manifest(path: pathlib.Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ManifestError(f"{path}: manifest is not an object")
    for key in REQUIRED_TOP:
        if key not in obj:
            raise ManifestError(f"{path}: missing {key}")
    require_int(obj["schema_version"], "schema_version")
    require_str(obj["stage"], "stage")
    require_str(obj["artifact_root"], "artifact_root")
    require_str(obj["build_sha"], "build_sha")
    require_str(obj["device_selector"], "device_selector")
    require_int(obj["fa"], "fa")
    if not isinstance(obj["model"], dict) or not isinstance(obj["moe_knobs"], dict) or not isinstance(obj["artifacts"], dict):
        raise ManifestError(f"{path}: model, moe_knobs, and artifacts must be objects")
    require_str(obj["model"].get("path"), "model.path")
    for key in REQUIRED_MOE:
        require_str(obj["moe_knobs"].get(key), f"moe_knobs.{key}")
    require_int(obj["prompt_tokens"], "prompt_tokens")
    require_int(obj["gen_tokens"], "gen_tokens")
    require_int(obj["repeat"], "repeat")
    return obj


def merge_identity(obj: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": obj["schema_version"],
        "build_sha": obj["build_sha"],
        "model": obj["model"],
        "device_selector": obj["device_selector"],
        "fa": obj["fa"],
        "moe_knobs": obj["moe_knobs"],
        "prompt_tokens": obj["prompt_tokens"],
        "gen_tokens": obj["gen_tokens"],
        "repeat": obj["repeat"],
    }


def key_for(identity: dict[str, Any]) -> str:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Validate staged SYCL profiling manifests")
    parser.add_argument("manifest", nargs="+", type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        manifests = [load_manifest(path) for path in args.manifest]
        identity = merge_identity(manifests[0])
        merge_key = key_for(identity)
        for obj in manifests[1:]:
            if merge_identity(obj) != identity:
                raise ManifestError("metadata mismatch across stage manifests")
    except (OSError, json.JSONDecodeError, ManifestError) as exc:
        print(f"failed to parse stage manifests: {exc}")
        return 2
    print("manifest.status ok")
    print(f"manifest.count {len(manifests)}")
    print(f"manifest.schema_version {manifests[0]['schema_version']}")
    print(f"manifest.merge_key {merge_key}")
    for obj in sorted(manifests, key=lambda item: item["stage"]):
        print(f"manifest.stage.{obj['stage']}.root {obj['artifact_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
