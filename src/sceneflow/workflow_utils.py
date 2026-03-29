from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO


OUTPUT_ROOT = "outputs"
STEP_SCAN = "scan"
STEP_SCENEIFY = "sceneify"
STEP_REPRESENT = "representatives"
STEP_TAG = "tagging"
STEP_BUILD = "candidates"


def timestamp_slug(now: datetime | None = None) -> str:
    if now is None:
        now = datetime.now().astimezone()
    return now.strftime("%Y%m%d-%H%M%S")


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def infer_run_dir_from_path(path_value: object) -> Path | None:
    if path_value is None:
        return None

    path = Path(str(path_value))
    parents = list(path.parents)
    for index, parent in enumerate(parents):
        if parent.name == OUTPUT_ROOT and index > 0:
            return parents[index - 1]
    return None


def resolve_output_path(
    *,
    default_filename: str,
    step_name: str,
    output: str | Path | None = None,
    output_dir: str | Path | None = None,
    run_dir: str | Path | None = None,
    input_path: object = None,
) -> Path:
    if output is not None:
        return Path(output)

    if output_dir is not None:
        return Path(output_dir) / default_filename

    resolved_run_dir = Path(run_dir) if run_dir is not None else infer_run_dir_from_path(input_path)
    if resolved_run_dir is None:
        resolved_run_dir = Path(OUTPUT_ROOT) / timestamp_slug()

    return resolved_run_dir / step_name / default_filename


def scene_band_counts(total: int) -> tuple[int, int]:
    """Return opening and closing counts that leave room for a body section.

    Short scenes should still have a middle beat when possible, so we avoid
    consuming the whole sequence with opening/closing buckets.
    """

    if total <= 0:
        return 0, 0
    if total == 1:
        return 1, 0
    if total == 2:
        return 1, 1

    opening = min(2, max(1, math.ceil(total * 0.15)))
    closing = min(2, max(1, math.ceil(total * 0.15)))
    if opening + closing >= total:
        opening = 1
        closing = 1 if total > 1 else 0
    return opening, closing


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_manifest(output_path: Path, payload: dict[str, object]) -> Path:
    manifest_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    write_json(manifest_path, payload)
    return manifest_path


def log(message: str, *, stream: TextIO = sys.stderr) -> None:
    print(message, file=stream)


@dataclass
class ProgressReporter:
    label: str
    total: int
    interval: int = 25
    stream: TextIO = sys.stderr

    def __post_init__(self) -> None:
        self._started_at = time.monotonic()
        self._last_rendered = 0
        self._finished = False

    def _render(self, current: int, detail: str | None = None, *, final: bool = False) -> None:
        if self.total <= 0:
            return

        if not final and current < self.total and current - self._last_rendered < self.interval:
            return

        self._last_rendered = current
        elapsed = time.monotonic() - self._started_at
        rate = current / elapsed if elapsed > 0 and current > 0 else None
        remaining = None
        if rate and current < self.total:
            remaining = (self.total - current) / rate

        prefix = f"[{self.label}] {current}/{self.total}"
        parts = [prefix, f"elapsed={format_duration(elapsed)}"]
        if remaining is not None:
            parts.append(f"eta={format_duration(remaining)}")
        if detail:
            parts.append(detail)

        end = "\n" if final or current >= self.total else "\r"
        print(" ".join(parts), file=self.stream, end=end, flush=True)

    def update(self, current: int, detail: str | None = None) -> None:
        self._render(current, detail)

    def finish(self, detail: str | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        self._render(self.total, detail, final=True)


def summarize_elapsed(started_at: float) -> str:
    return format_duration(time.monotonic() - started_at)
