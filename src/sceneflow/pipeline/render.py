from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

from sceneflow.workflow_utils import log, resolve_output_path, summarize_elapsed, write_manifest


DEFAULT_INPUT = "edit_plan.json"
DEFAULT_OUTPUT = "preview.mp4"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
FRAME_W = 1280
FRAME_H = 720
FPS = 30


def is_missing(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "none"


def load_plan(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "plan" not in payload:
        raise ValueError("edit_plan.json must contain a top-level 'plan' object")
    return payload


def source_duration(item: dict[str, object], source_count: int = 1) -> float:
    value = item.get("planned_duration_seconds")
    try:
        duration = float(value)
    except Exception:
        duration = 2.0
    if source_count > 1:
        return max(0.8, min(duration / max(source_count, 1), 1.6))
    return max(1.0, min(duration, 4.0))


def resolve_source_path(path_value: object, root: Path) -> Path | None:
    if is_missing(path_value):
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return root / path


def preview_sources(item: dict[str, object]) -> list[dict[str, object]]:
    raw_sources = item.get("preview_sources")
    if isinstance(raw_sources, list):
        sources = [source for source in raw_sources if isinstance(source, dict) and not is_missing(source.get("path"))]
        if sources:
            return sources
    if is_missing(item.get("representative_path")):
        return []
    return [
        {
            "path": item.get("representative_path"),
            "kind": item.get("representative_kind"),
            "duration_seconds": None,
            "final_timestamp": None,
        }
    ]


def representative_source(item: dict[str, object]) -> dict[str, object] | None:
    if is_missing(item.get("representative_path")):
        return None
    return {
        "path": item.get("representative_path"),
        "kind": item.get("representative_kind"),
        "duration_seconds": None,
        "final_timestamp": None,
    }


def resolve_preview_sources(item: dict[str, object], root: Path) -> list[dict[str, object]]:
    resolved_sources: list[dict[str, object]] = []
    for source in preview_sources(item):
        src = resolve_source_path(source.get("path"), root)
        if src is None or not src.exists():
            continue
        resolved_sources.append(
            {
                "source_path": src,
                "kind": source.get("kind"),
                "final_timestamp": source.get("final_timestamp"),
            }
        )

    if resolved_sources:
        return resolved_sources

    representative = representative_source(item)
    if representative is None:
        return []

    src = resolve_source_path(representative.get("path"), root)
    if src is None or not src.exists():
        return []
    return [
        {
            "source_path": src,
            "kind": representative.get("kind"),
            "final_timestamp": representative.get("final_timestamp"),
        }
    ]


def normalize_clip(input_path: Path, output_path: Path, duration: float) -> None:
    suffix = input_path.suffix.lower()
    vf = f"scale={FRAME_W}:{FRAME_H}:force_original_aspect_ratio=decrease,pad={FRAME_W}:{FRAME_H}:(ow-iw)/2:(oh-ih)/2,format=yuv420p,fps={FPS}"
    if suffix in IMAGE_EXTS:
        raster_path = output_path.with_suffix(".png")
        raster_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-frames:v",
            "1",
            str(raster_path),
        ]
        raster_result = subprocess.run(raster_cmd, capture_output=True, text=True, check=False)
        if raster_result.returncode != 0 or not raster_path.exists():
            raise RuntimeError(
                f"Failed to rasterize image: {input_path}\n"
                f"stdout={raster_result.stdout[-1000:]}\n"
                f"stderr={raster_result.stderr[-1000:]}"
            )
        cmd = [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-t",
            f"{duration:.2f}",
            "-i",
            str(raster_path),
            "-vf",
            vf,
            "-an",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            "0",
            "-t",
            f"{duration:.2f}",
            "-i",
            str(input_path),
            "-vf",
            vf,
            "-an",
            "-movflags",
            "+faststart",
            str(output_path),
        ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not output_path.exists():
        raise RuntimeError(
            f"Failed to normalize clip: {input_path}\n"
            f"stdout={result.stdout[-1000:]}\n"
            f"stderr={result.stderr[-1000:]}"
        )


def build_clip_list(plan: dict[str, object], root: Path, work_dir: Path) -> list[dict[str, object]]:
    plan_data = plan.get("plan", {})
    if not isinstance(plan_data, dict):
        raise ValueError("edit_plan.json must contain a top-level 'plan' object")

    render_guidance = plan_data.get("render_guidance", {})
    if not isinstance(render_guidance, dict):
        render_guidance = {}

    sequence = list(render_guidance.get("preferred_order", []))
    edit_items = list(plan_data.get("edit_sequence", []))
    if not edit_items:
        edit_items = list(plan_data.get("edit_sequence_notes", []))
    order_lookup = {item.get("scene_id"): item for item in edit_items if isinstance(item, dict)}

    ordered_items: list[dict[str, object]] = []
    for scene_id in sequence:
        item = order_lookup.get(scene_id)
        if item is None:
            continue
        ordered_items.append(item)

    if not ordered_items:
        ordered_items = edit_items

    clips: list[dict[str, object]] = []
    clips_dir = work_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    clip_index = 1
    for item in ordered_items:
        resolved_sources = resolve_preview_sources(item, root)
        if not resolved_sources:
            continue

        duration = source_duration(item, len(resolved_sources))
        for preview_index, source in enumerate(resolved_sources, start=1):
            clip_path = clips_dir / f"{clip_index:03d}_scene_{item.get('scene_id')}_{preview_index:02d}.mp4"
            normalize_clip(source["source_path"], clip_path, duration)
            clips.append(
                {
                    "scene_id": item.get("scene_id"),
                    "clip_path": str(clip_path),
                    "source_path": str(source["source_path"]),
                    "duration_seconds": duration,
                    "chapter_id": item.get("chapter_id"),
                    "edit_action": item.get("edit_action"),
                    "transition_hint": item.get("transition_hint") if preview_index == 1 else "cut",
                    "preview_index": preview_index,
                    "preview_kind": source.get("kind"),
                    "preview_final_timestamp": source.get("final_timestamp"),
                }
            )
            clip_index += 1

    return clips


def write_concat_list(clips: list[dict[str, object]], path: Path) -> None:
    lines = [f"file '{clip['clip_path']}'" for clip in clips]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def concat_clips(list_path: Path, output_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not output_path.exists():
        raise RuntimeError(
            f"Failed to concat preview video\n"
            f"stdout={result.stdout[-1000:]}\n"
            f"stderr={result.stderr[-1000:]}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a simple preview video from the edit plan.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input edit_plan.json path")
    parser.add_argument("--output", default=None, help="Output preview.mp4 path")
    parser.add_argument("--output-dir", default=None, help="Directory for outputs. Defaults to outputs/<run>/render/")
    parser.add_argument("--run-dir", default=None, help="Shared run directory for the whole workflow")
    parser.add_argument("--root", default=".", help="Workspace root for representative paths")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = resolve_output_path(
        default_filename=DEFAULT_OUTPUT,
        step_name="render",
        output=args.output,
        output_dir=args.output_dir,
        run_dir=args.run_dir,
        input_path=input_path,
    )
    root = Path(args.root).resolve()

    log(f"[render] input={input_path}")
    log(f"[render] output={output_path}")
    log(f"[render] root={root}")
    start = time.monotonic()

    plan_payload = load_plan(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="photos-ooarai-render-") as tmp_dir:
        work_dir = Path(tmp_dir)
        clips = build_clip_list(plan_payload, root, work_dir)
        if not clips:
            raise RuntimeError("No renderable clips were found in the plan")

        list_path = work_dir / "concat.txt"
        write_concat_list(clips, list_path)
        concat_clips(list_path, output_path)

        report = {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "input": str(input_path),
            "output": str(output_path),
            "clip_count": len(clips),
            "duration_seconds": round(sum(clip["duration_seconds"] for clip in clips), 2),
            "scene_ids": [clip["scene_id"] for clip in clips],
            "clips": clips,
        }
        report_path = output_path.with_suffix(".json")
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        write_manifest(
            output_path,
            {
                "generated_at": report["generated_at"],
                "step": "render",
                "input": str(input_path),
                "output": str(output_path),
                "report": str(report_path),
                "clip_count": len(clips),
                "elapsed_seconds": round(time.monotonic() - start, 3),
            },
        )

    log(f"[render] wrote {output_path} ({len(clips)} clips, elapsed={summarize_elapsed(start)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
