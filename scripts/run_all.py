"""One-command full reproduction, using an already installed environment."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


if __name__ == "__main__":
    run("scripts/download_assets.py")
    run("-m", "src.common")
    run("-m", "pytest", "-q")
    run("-m", "src.classical", "--glove", "artifacts/glove.6B.100d.txt")
    run("-m", "src.bert", "--model-dir", "artifacts/bert-base-uncased")
    run("scripts/audit_results.py")
    run("scripts/build_report.py")
