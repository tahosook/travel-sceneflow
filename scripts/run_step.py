from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sceneflow.pipeline import candidates, llm_plan, meanings, render, representatives, scan, sceneify, structure, tagging


STEP_MAIN = {
    "scan": scan.main,
    "sceneify": sceneify.main,
    "representatives": representatives.main,
    "tagging": tagging.main,
    "candidates": candidates.main,
    "meanings": meanings.main,
    "structure": structure.main,
    "llm": llm_plan.main,
    "render": render.main,
}


def _run_step(step_name: str, argv: list[str]) -> int:
    previous_argv = sys.argv[:]
    try:
        sys.argv = [step_name, *argv]
        return int(STEP_MAIN[step_name]())
    finally:
        sys.argv = previous_argv


def _usage() -> str:
    steps = ", ".join(STEP_MAIN.keys())
    return "\n".join(
        [
            "Usage:",
            "  python -B scripts/run_step.py <step> [step options...]",
            "",
            f"Available steps: {steps}",
            "",
            "Examples:",
            "  python -B scripts/run_step.py scan --help",
            "  python -B scripts/run_step.py sceneify --input outputs/sample-run/scan/media_info.csv --run-dir outputs/sample-run",
        ]
    )


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(_usage())
        return 0

    step_name = sys.argv[1].strip().lower()
    if step_name not in STEP_MAIN:
        print(f"Unknown step: {step_name}", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2

    return _run_step(step_name, sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())
