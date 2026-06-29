import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "check-sycl-native-fp4-bdpas.sh"
REPORT = ROOT / "activation" / "native-fp4-bdpas-capability.md"


def test_native_fp4_probe_dry_run_report(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    result = subprocess.run(["bash", str(SCRIPT), "--output", str(out)], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert result.returncode == 0, result.stderr
    text = out.read_text(encoding="utf-8")
    assert "native_fp4.usable=" in text
    assert "bdpas.present=" in text
    assert "/opt/intel/oneapi" in text
