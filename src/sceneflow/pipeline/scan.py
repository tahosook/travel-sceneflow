from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from sceneflow.workflow_utils import ProgressReporter, log, resolve_output_path, summarize_elapsed, write_manifest


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS
TIMESTAMP_RE = re.compile(r"(?P<date>\d{8})[._-](?P<time>\d{6})(?P<msec>\d{0,3})")
IMAGE_SAMPLE_SIZE = 64


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


def media_paths(root: Path) -> list[Path]:
    return [path for path in sorted(root.rglob("*")) if path.is_file() and path.suffix.lower() in MEDIA_EXTS]


def run_json_command(cmd: list[str]) -> dict:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return {}
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}
    if isinstance(data, list):
        return data[0] if data else {}
    return data


def exiftool_metadata(path: Path) -> dict:
    cmd = [
        "exiftool",
        "-json",
        "-n",
        "-DateTimeOriginal",
        "-CreateDate",
        "-GPSLatitude",
        "-GPSLongitude",
        "-Model",
        "-ImageWidth",
        "-ImageHeight",
    ]
    if is_video(path):
        cmd.extend(["-api", "QuickTimeUTC=1"])
    cmd.append(str(path))
    return run_json_command(cmd)


def ffprobe_metadata(path: Path) -> dict:
    return run_json_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )


def parse_datetime_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    formats = [
        "%Y:%m:%d %H:%M:%S",
        "%Y:%m:%d %H:%M:%S.%f",
        "%Y:%m:%d %H:%M:%S%z",
        "%Y:%m:%d %H:%M:%S.%f%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S.%f%z",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(text, fmt)
        except ValueError:
            continue
        return dt.isoformat(sep=" ")
    return text


def infer_captured_at_from_name(path: Path) -> str | None:
    match = TIMESTAMP_RE.search(path.name)
    if not match:
        return None

    date_text = match.group("date")
    time_text = match.group("time")
    msec_text = match.group("msec") or "000"
    dt_text = f"{date_text} {time_text}{msec_text}"

    for fmt in ("%Y%m%d %H%M%S%f", "%Y%m%d %H%M%S"):
        try:
            dt = datetime.strptime(dt_text, fmt)
        except ValueError:
            continue
        return dt.isoformat(sep=" ")

    return None


def extract_captured_at(meta: dict, path: Path) -> str | None:
    date_time_original = parse_datetime_text(meta.get("DateTimeOriginal"))
    if date_time_original:
        return date_time_original

    create_date = parse_datetime_text(meta.get("CreateDate"))
    if create_date:
        return create_date

    return infer_captured_at_from_name(path)


def parse_gps_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return text


def image_metrics(path: Path) -> tuple[object, object]:
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(path),
            "-filter_complex",
            f"scale={IMAGE_SAMPLE_SIZE}:{IMAGE_SAMPLE_SIZE}:force_original_aspect_ratio=decrease,pad={IMAGE_SAMPLE_SIZE}:{IMAGE_SAMPLE_SIZE}:(ow-iw)/2:(oh-ih)/2,format=gray",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout:
        return None, None

    pixels = result.stdout
    expected = IMAGE_SAMPLE_SIZE * IMAGE_SAMPLE_SIZE
    if len(pixels) < expected:
        return None, None

    return laplacian_variance(IMAGE_SAMPLE_SIZE, IMAGE_SAMPLE_SIZE, pixels[:expected]), brightness_mean(pixels[:expected])


def brightness_mean(pixels: bytes) -> float:
    if not pixels:
        return 0.0
    return sum(pixels) / len(pixels)


def laplacian_variance(width: int, height: int, pixels: bytes) -> float:
    if width < 3 or height < 3:
        return 0.0

    total = 0.0
    total_sq = 0.0
    count = 0

    for y in range(height):
        row_start = y * width
        for x in range(width):
            c = pixels[row_start + x]
            up = pixels[row_start - width + x] if y > 0 else 0
            down = pixels[row_start + width + x] if y + 1 < height else 0
            left = pixels[row_start + x - 1] if x > 0 else 0
            right = pixels[row_start + x + 1] if x + 1 < width else 0
            response = up + down + left + right - 4 * c
            total += response
            total_sq += response * response
            count += 1

    mean = total / count
    return total_sq / count - mean * mean


def video_metadata(path: Path) -> tuple[object, object]:
    meta = ffprobe_metadata(path)
    duration = None
    fmt = meta.get("format", {})
    if isinstance(fmt, dict):
        raw_duration = fmt.get("duration")
        if raw_duration not in (None, ""):
            try:
                duration = float(raw_duration)
            except (TypeError, ValueError):
                duration = None

    has_audio = False
    streams = meta.get("streams", [])
    if isinstance(streams, list):
        has_audio = any(isinstance(stream, dict) and stream.get("codec_type") == "audio" for stream in streams)

    return duration, has_audio


def media_rows(root: Path, reporter: ProgressReporter | None = None, candidates: list[Path] | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if candidates is None:
        candidates = media_paths(root)
    for index, path in enumerate(candidates, start=1):
        if reporter is not None:
            reporter.update(index, path.name)

        meta = exiftool_metadata(path)
        row: dict[str, object] = {
            "path": str(path.resolve()),
            "kind": "image" if is_image(path) else "video" if is_video(path) else "other",
            "captured_at": extract_captured_at(meta, path),
            "date_time_original": parse_datetime_text(meta.get("DateTimeOriginal")),
            "create_date": parse_datetime_text(meta.get("CreateDate")),
            "gps_latitude": parse_gps_value(meta.get("GPSLatitude")),
            "gps_longitude": parse_gps_value(meta.get("GPSLongitude")),
            "model": meta.get("Model") or None,
            "laplacian": None,
            "brightness": None,
            "duration_seconds": None,
            "has_audio": None,
        }

        if row["kind"] == "image":
            laplacian, brightness = image_metrics(path)
            row["laplacian"] = laplacian
            row["brightness"] = brightness
        elif row["kind"] == "video":
            duration, has_audio = video_metadata(path)
            row["duration_seconds"] = duration
            row["has_audio"] = has_audio

        rows.append(row)

    if reporter is not None:
        reporter.finish("media scan complete")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan media files and output basic metrics and metadata.")
    parser.add_argument("--root", default=".", help="Root directory to scan")
    parser.add_argument("--output", default=None, help="CSV output path")
    parser.add_argument("--output-dir", default=None, help="Directory for outputs. Defaults to outputs/<run>/scan/")
    parser.add_argument("--run-dir", default=None, help="Shared run directory for the whole workflow")
    parser.add_argument("--progress-interval", type=int, default=25, help="How often to print progress updates")
    args = parser.parse_args()

    root = Path(args.root)
    output = resolve_output_path(
        default_filename="media_info.csv",
        step_name="scan",
        output=args.output,
        output_dir=args.output_dir,
        run_dir=args.run_dir,
        input_path=root,
    )

    log(f"[scan] root={root.resolve()}")
    log(f"[scan] output={output}")
    start = time.monotonic()
    candidates = media_paths(root)
    reporter = ProgressReporter("scan", total=len(candidates), interval=args.progress_interval)
    rows = media_rows(root, reporter=reporter, candidates=candidates)
    df = pd.DataFrame(rows)

    columns = [
        "path",
        "kind",
        "captured_at",
        "date_time_original",
        "create_date",
        "gps_latitude",
        "gps_longitude",
        "model",
        "laplacian",
        "brightness",
        "duration_seconds",
        "has_audio",
    ]
    df = df.reindex(columns=columns)

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, na_rep="None")
    write_manifest(
        output,
        {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "step": "scan",
            "root": str(root.resolve()),
            "output": str(output),
            "row_count": int(len(df)),
            "elapsed_seconds": round(time.monotonic() - start, 3),
        },
    )
    log(f"[scan] wrote {output} ({len(df)} rows, elapsed={summarize_elapsed(start)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
