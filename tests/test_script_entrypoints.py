import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_builder_is_directly_executable() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_xinjiang_2021_manifest.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--source-manifest" in result.stdout


def test_formal_smoke_is_directly_executable() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "smoke_e0.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    assert "--output-dir" in result.stdout


def test_formal_smoke_parser_accepts_e1_config_and_output() -> None:
    from scripts.smoke_e0 import parse_args

    config = ROOT / "configs" / "models" / "tsvit_wpe_basic.yaml"
    output = ROOT / ".smoke" / "e1-parser-test"
    args = parse_args(["--config", str(config), "--output-dir", str(output)])

    assert args.config == config
    assert args.output_dir == output
