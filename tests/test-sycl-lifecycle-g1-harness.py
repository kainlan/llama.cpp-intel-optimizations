#!/usr/bin/env python3
"""Non-GPU unit tests for the canonical G1 launcher."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

MODULE_PATH = Path(__file__).with_name("test-sycl-lifecycle-g1-aba.py")
spec = importlib.util.spec_from_file_location("g1_launcher", MODULE_PATH)
g1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g1)


class G1HarnessTests(unittest.TestCase):
    uuid = "00112233-4455-6677-8899-aabbccddeeff"
    a_tokens = list(range(10, 18))
    b_tokens = list(range(30, 38))

    def test_a_shared_is_mandatory_but_not_inferred(self):
        source = Path(__file__).with_name("test-sycl-lifecycle-gpu-sequential.cpp").read_text()
        self.assertIn("!o.a_shared.empty()", source)
        self.assertEqual(source.count("infer(o.a_shared, o, selected)"), 1)
        sequence = '\n'.join([
            '        runs.push_back({ "A", infer(o.a, o, selected) });',
            '        runs.push_back({ "B", infer(o.b, o, selected) });',
            '        // The final A oracle must come from the distinct renamed A-shared',
        ])
        self.assertIn(sequence, source)

    def make_fixture(self, root):
        data = {
            "schema": 4,
            "fixtures": {
                "A": {"path": "a.gguf"},
                "B": {"path": "b.gguf"},
                "A-shared": {"path": "as.gguf"},
            },
            "device": {"physical_uuid": self.uuid},
            "expected": {
                "A": {"tokens": self.a_tokens, "token_sha256": g1.token_hash(self.a_tokens)},
                "B": {"tokens": self.b_tokens, "token_sha256": g1.token_hash(self.b_tokens)},
            },
        }
        path = root / "fixture.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    @staticmethod
    def paths(root):
        return {"A": root / "a.gguf", "B": root / "b.gguf", "A-shared": root / "as.gguf"}

    def test_rejects_malformed_json_nested_and_identical_paths(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.json"
            path.write_text("{", encoding="utf-8")
            with self.assertRaises(g1.PrerequisiteError):
                g1.load_contract(path)
            path.write_text('{"schema":4,"fixtures":[]}', encoding="utf-8")
            with self.assertRaises(g1.PrerequisiteError):
                g1.load_contract(path)
            fixture = self.make_fixture(Path(td))
            data = json.loads(fixture.read_text(encoding="utf-8"))
            data["fixtures"]["A-shared"]["path"] = data["fixtures"]["A"]["path"]
            fixture.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(g1.PrerequisiteError):
                g1.load_contract(fixture)

    def test_constants_are_immutable_and_external_overrides_are_rejected(self):
        self.assertEqual(
            g1.MODEL_SHA256,
            {
                "A": "66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739",
                "B": "270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d",
                "A-shared": "66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739",
            },
        )
        self.assertEqual((g1.PROMPT, g1.SEED, g1.TEMPERATURE, g1.N_PREDICT), ("1, 2, 3, 4, 5,", 42, 0, 8))
        self.assertEqual((g1.ONEAPI_SELECTOR, g1.GGML_LOGICAL_SELECTOR), ("level_zero:0,1", "1"))
        mutations = (
            lambda data: data.update(generation={"prompt": "changed"}),
            lambda data: data.update(generation={"seed": 1}),
            lambda data: data.update(generation={"temperature": 1}),
            lambda data: data.update(generation={"n_predict": 7}),
            lambda data: data["device"].update(ggml_logical_selector="0"),
            lambda data: data["device"].update(oneapi_selector="level_zero:0"),
            lambda data: data["fixtures"]["A"].update(sha256="0" * 64),
            lambda data: data["fixtures"]["B"].update(sha256="0" * 64),
            lambda data: data["fixtures"]["A-shared"].update(sha256="0" * 64),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate), tempfile.TemporaryDirectory() as td:
                fixture = self.make_fixture(Path(td))
                data = json.loads(fixture.read_text(encoding="utf-8"))
                mutate(data)
                fixture.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(g1.PrerequisiteError):
                    g1.load_contract(fixture)

    def test_scrubs_environment_and_asserts_exact_runner_arguments(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fixture = self.make_fixture(root)
            calls = []
            observed_sequences = []

            def fake_run(command, **kwargs):
                calls.append((command, kwargs))
                sequence = command[command.index("--run") + 1].split(",")
                observed_sequences.append(sequence)
                token_map = {"A": self.a_tokens, "B": self.b_tokens}
                output = {
                    "device_uuid": self.uuid,
                    "runs": [{"model": name, "tokens": token_map[name]} for name in sequence],
                }
                return subprocess.CompletedProcess(command, 0, json.dumps(output), "")

            polluted = {
                "GGML_SYCL_UNIFIED_FORCE_LEGACY": "1",
                "GGML_FUTURE_BACKEND_ROUTE": "conflict",
                "SYCL_PI_TRACE": "1",
                "ONEAPI_DEVICE_SELECTOR": "opencl:cpu",
                "UR_DEVICE_AFFINITY_MASK": "0",
                "ZE_AFFINITY_MASK": "0",
                "ZES_ENABLE_SYSMAN": "0",
                "LLAMA_ARG_N_GPU_LAYERS": "0",
                "CUDA_VISIBLE_DEVICES": "",
                "PATH": "/canonical/bin",
                "LD_LIBRARY_PATH": "/canonical/lib",
            }
            with (
                mock.patch.dict(g1.os.environ, polluted, clear=False),
                mock.patch.object(g1, "model_paths", return_value=self.paths(root)),
                mock.patch.object(g1.subprocess, "run", fake_run),
            ):
                self.assertEqual(g1.run(["--strict", "--fixture", str(fixture), sys.executable]), 0)
            self.assertEqual(len(calls), 3)
            self.assertEqual(
                [command[command.index("--run") + 1] for command, _ in calls],
                ["A", "B", "A,B,A"],
            )
            self.assertEqual(observed_sequences, [["A"], ["B"], ["A", "B", "A"]])
            for command, kwargs in calls:
                self.assertEqual(command[command.index("--model-a") + 1], str(root / "a.gguf"))
                self.assertEqual(command[command.index("--model-b") + 1], str(root / "b.gguf"))
                self.assertEqual(command[command.index("--model-a-shared") + 1], str(root / "as.gguf"))
                self.assertEqual(command[command.index("--prompt") + 1], g1.PROMPT)
                self.assertEqual(command[command.index("--seed") + 1], "42")
                self.assertEqual(command[command.index("--temp") + 1], "0")
                self.assertEqual(command[command.index("--n-predict") + 1], "8")
                env = kwargs["env"]
                self.assertEqual(env["ONEAPI_DEVICE_SELECTOR"], "level_zero:0,1")
                self.assertEqual(env["GGML_SYCL_DEVICE"], "1")
                self.assertEqual(env["PATH"], "/canonical/bin")
                self.assertEqual(env["LD_LIBRARY_PATH"], "/canonical/lib")
                for key in polluted:
                    if key not in {"PATH", "LD_LIBRARY_PATH", *g1.CANONICAL_RUNTIME_ENV}:
                        self.assertNotIn(key, env)
                self.assertFalse(
                    any(
                        g1.is_backend_routing_env(key) and key not in g1.CANONICAL_RUNTIME_ENV
                        for key in env
                    )
                )
                self.assertEqual(kwargs["timeout"], g1.PROCESS_TIMEOUT_SECONDS)

    def test_token_mismatch_is_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root, fixture = Path(td), self.make_fixture(Path(td))

            def bad_tokens(command, **kwargs):
                sequence = command[command.index("--run") + 1].split(",")
                output = {"device_uuid": self.uuid, "runs": [{"model": name, "tokens": [99] * 8} for name in sequence]}
                return subprocess.CompletedProcess(command, 0, json.dumps(output), "")

            with (
                mock.patch.object(g1, "model_paths", return_value=self.paths(root)),
                mock.patch.object(g1.subprocess, "run", bad_tokens),
            ):
                self.assertEqual(g1.run(["--fixture", str(fixture), sys.executable]), 1)

    def test_timeout_is_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root, fixture = Path(td), self.make_fixture(Path(td))
            with (
                mock.patch.object(g1, "model_paths", return_value=self.paths(root)),
                mock.patch.object(g1.subprocess, "run", side_effect=subprocess.TimeoutExpired("runner", 1)),
            ):
                self.assertEqual(g1.run(["--fixture", str(fixture), sys.executable]), 1)

    def test_committed_placeholder_skips_normally_and_fails_strict(self):
        fixture = Path(__file__).with_name("sycl-lifecycle-fixtures.json")
        self.assertEqual(g1.run(["--fixture", str(fixture), sys.executable]), 77)
        self.assertEqual(g1.run(["--strict", "--fixture", str(fixture), sys.executable]), 1)


if __name__ == "__main__":
    unittest.main()
