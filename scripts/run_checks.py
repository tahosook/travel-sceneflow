from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> int:
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    return int(result.returncode)


def main() -> int:
    commands = [
        ["ruff", "check", "."],
        [sys.executable, "-m", "pytest"],
    ]

    for cmd in commands:
        code = run(cmd)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
