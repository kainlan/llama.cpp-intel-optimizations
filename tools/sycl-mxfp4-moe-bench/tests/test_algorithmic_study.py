import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "sycl-mxfp4-algorithmic-study.py"


def valid_capture() -> dict:
    return {
        "baseline_output": [1.0, 2.0, 3.0, 4.0],
        "candidate_output": [1.0, 2.001, 2.999, 4.0],
        "baseline_logits_top10": [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4],
        "candidate_logits_top10": [0.5001, 0.3999, 0.3001, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4],
        "baseline_ms_per_token": 27.14,
        "candidate_ms_per_token": 22.0,
    }


def write_capture(path: Path, capture: dict) -> None:
    path.write_text(json.dumps(capture) + "\n", encoding="utf-8")


def make_capture(path: Path) -> None:
    write_capture(path, valid_capture())


def test_algorithmic_study_accepts_fixture(tmp_path: Path) -> None:
    capture = tmp_path / "capture.json"
    make_capture(capture)
    result = subprocess.run(["python3", str(SCRIPT), str(capture)], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert result.returncode == 0, result.stderr
    assert "relative_l2" in result.stdout
    assert "top10_logit_mae" in result.stdout
    assert "speed_ceiling_tok_s" in result.stdout


def test_algorithmic_study_rejects_empty(tmp_path: Path) -> None:
    capture = tmp_path / "empty.json"
    capture.write_text("", encoding="utf-8")
    result = subprocess.run(["python3", str(SCRIPT), str(capture)], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert result.returncode != 0
    assert "error: empty algorithmic capture" in result.stderr


def test_algorithmic_study_rejects_oversized_integer_without_traceback(tmp_path: Path) -> None:
    capture = tmp_path / "oversized.json"
    data = valid_capture()
    data["candidate_output"] = [10**400, 2.001, 2.999, 4.0]
    write_capture(capture, data)
    result = subprocess.run(["python3", str(SCRIPT), str(capture)], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert result.returncode != 0
    assert "error:" in result.stderr
    assert "numeric value is too large" in result.stderr
    assert "Traceback" not in result.stderr


def test_algorithmic_study_rejects_zero_candidate_ms_without_traceback(tmp_path: Path) -> None:
    capture = tmp_path / "zero-ms.json"
    data = valid_capture()
    data["candidate_ms_per_token"] = 0.0
    write_capture(capture, data)
    result = subprocess.run(["python3", str(SCRIPT), str(capture)], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert result.returncode != 0
    assert "error: candidate_ms_per_token must be positive" in result.stderr
    assert "Traceback" not in result.stderr
