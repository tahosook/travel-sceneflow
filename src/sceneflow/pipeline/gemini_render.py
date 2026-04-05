from __future__ import annotations

import argparse
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path

from sceneflow.pipeline import render
from sceneflow.workflow_utils import log, resolve_output_path, summarize_elapsed, write_manifest


DEFAULT_INPUT = "slideshow_plan.json"
DEFAULT_OUTPUT = "preview.mp4"


def is_missing(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "none"


def normalize_text(value: object) -> str:
    if is_missing(value):
        return ""
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split()).strip()


def parse_float(value: object, default: float = 0.0) -> float:
    if is_missing(value):
        return default
    try:
        return float(value)
    except Exception:
        return default


def load_slideshow(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing slideshow_plan.json: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("plan"), dict):
        raise ValueError("slideshow_plan.json must contain a top-level 'plan' object")
    return payload


def load_structure(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing edit_structure.json: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "edit_sequence" not in payload:
        raise ValueError("edit_structure.json must contain a top-level 'edit_sequence' array")
    return payload


def scene_ids_from_values(values: object) -> list[int]:
    scene_ids: list[int] = []
    seen: set[int] = set()
    for value in list(values or []):
        if is_missing(value):
            continue
        try:
            scene_id = int(value)
        except Exception:
            continue
        if scene_id <= 0 or scene_id in seen:
            continue
        seen.add(scene_id)
        scene_ids.append(scene_id)
    return scene_ids


def slideshow_scene_order(slideshow_plan: dict[str, object], fallback_sequence: list[dict[str, object]]) -> list[int]:
    ordered_scene_ids: list[int] = []
    seen: set[int] = set()

    for chapter in list(slideshow_plan.get("chapter_list") or []):
        if not isinstance(chapter, dict):
            continue
        for scene_id in scene_ids_from_values(chapter.get("scene_ids")):
            if scene_id in seen:
                continue
            seen.add(scene_id)
            ordered_scene_ids.append(scene_id)

    for direction in list(slideshow_plan.get("scene_directions") or []):
        if not isinstance(direction, dict):
            continue
        for scene_id in scene_ids_from_values(direction.get("scene_ids")):
            if scene_id in seen:
                continue
            seen.add(scene_id)
            ordered_scene_ids.append(scene_id)

    if ordered_scene_ids:
        return ordered_scene_ids

    for item in fallback_sequence:
        if not isinstance(item, dict) or is_missing(item.get("scene_id")):
            continue
        try:
            scene_id = int(item.get("scene_id"))
        except Exception:
            continue
        if scene_id <= 0 or scene_id in seen:
            continue
        seen.add(scene_id)
        ordered_scene_ids.append(scene_id)
    return ordered_scene_ids


def subtitle_items_from_slideshow(slideshow_plan: dict[str, object], ordered_scene_ids: list[int]) -> list[dict[str, object]]:
    subtitle_items: list[dict[str, object]] = []
    seen_scene_ids: set[int] = set()

    raw_subtitle_plan = slideshow_plan.get("subtitle_plan")
    raw_items = raw_subtitle_plan.get("items") if isinstance(raw_subtitle_plan, dict) else []
    for item in list(raw_items or []):
        if not isinstance(item, dict):
            continue
        scene_ids = scene_ids_from_values(item.get("scene_ids"))
        if not scene_ids:
            continue
        scene_id = scene_ids[0]
        if scene_id not in ordered_scene_ids or scene_id in seen_scene_ids:
            continue
        text = normalize_text(item.get("text"))
        if not text:
            continue
        seen_scene_ids.add(scene_id)
        subtitle_items.append(
            {
                "scene_id": scene_id,
                "text": text,
                "position": "bottom_center",
                "start_seconds": 0.35 if not subtitle_items else 0.45,
                "duration_seconds": 2.8 if not subtitle_items else 2.5,
                "origin": "gemini",
            }
        )

    if subtitle_items:
        return subtitle_items

    for direction in list(slideshow_plan.get("scene_directions") or []):
        if not isinstance(direction, dict):
            continue
        scene_ids = scene_ids_from_values(direction.get("scene_ids"))
        if not scene_ids:
            continue
        scene_id = scene_ids[0]
        if scene_id not in ordered_scene_ids or scene_id in seen_scene_ids:
            continue
        text = normalize_text(direction.get("subtitle"))
        if not text:
            continue
        seen_scene_ids.add(scene_id)
        subtitle_items.append(
            {
                "scene_id": scene_id,
                "text": text,
                "position": "bottom_center",
                "start_seconds": 0.35 if not subtitle_items else 0.45,
                "duration_seconds": 2.8 if not subtitle_items else 2.5,
                "origin": "gemini",
            }
        )
    return subtitle_items


def build_renderable_plan(slideshow_payload: dict[str, object], structure_payload: dict[str, object]) -> dict[str, object]:
    slideshow_plan = slideshow_payload.get("plan")
    if not isinstance(slideshow_plan, dict):
        raise ValueError("slideshow_plan.json must contain a top-level 'plan' object")

    raw_sequence = [item for item in list(structure_payload.get("edit_sequence") or []) if isinstance(item, dict)]
    sequence_by_scene_id = {
        int(item.get("scene_id")): json.loads(json.dumps(item, ensure_ascii=False))
        for item in raw_sequence
        if not is_missing(item.get("scene_id"))
    }
    ordered_scene_ids = slideshow_scene_order(slideshow_plan, raw_sequence)

    directions_by_scene_id: dict[int, dict[str, object]] = {}
    for direction in list(slideshow_plan.get("scene_directions") or []):
        if not isinstance(direction, dict):
            continue
        scene_ids = scene_ids_from_values(direction.get("scene_ids"))
        if not scene_ids:
            continue
        directions_by_scene_id[scene_ids[0]] = direction

    chapter_by_scene_id: dict[int, str] = {}
    for chapter in list(slideshow_plan.get("chapter_list") or []):
        if not isinstance(chapter, dict):
            continue
        chapter_id = normalize_text(chapter.get("chapter_id")) or "body"
        for scene_id in scene_ids_from_values(chapter.get("scene_ids")):
            chapter_by_scene_id[scene_id] = chapter_id

    merged_sequence: list[dict[str, object]] = []
    for scene_id in ordered_scene_ids:
        item = sequence_by_scene_id.get(scene_id)
        if item is None:
            continue
        direction = directions_by_scene_id.get(scene_id)
        if direction is not None:
            duration = parse_float(direction.get("recommended_duration_seconds"), default=parse_float(item.get("planned_duration_seconds"), default=2.0))
            item["planned_duration_seconds"] = round(max(0.8, min(duration, 6.0)), 2)
            item["gemini_direction"] = normalize_text(direction.get("direction"))
            item["gemini_emphasis"] = normalize_text(direction.get("emphasis"))
        chapter_id = chapter_by_scene_id.get(scene_id)
        if chapter_id:
            item["chapter_id"] = chapter_id
        merged_sequence.append(item)

    if not merged_sequence:
        merged_sequence = raw_sequence

    subtitle_items = subtitle_items_from_slideshow(slideshow_plan, [int(item.get("scene_id")) for item in merged_sequence if not is_missing(item.get("scene_id"))])

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": slideshow_payload.get("source"),
        "generated_by": "gemini_render_bridge",
        "plan": {
            "title": normalize_text(slideshow_plan.get("title")) or "旅の記録",
            "logline": normalize_text(slideshow_plan.get("logline")),
            "chapter_list": [chapter for chapter in list(slideshow_plan.get("chapter_list") or []) if isinstance(chapter, dict)],
            "edit_sequence": merged_sequence,
            "subtitle_plan": {
                "style": "overlay",
                "enabled": bool(subtitle_items),
                "items": subtitle_items,
                "notes": "Gemini slideshow_plan の subtitle/direction を render 用 subtitle_plan に反映した。",
            },
            "render_guidance": {
                "preferred_order": [item.get("scene_id") for item in merged_sequence if isinstance(item, dict)],
                "use_optional_scenes": False,
                "source": "gemini_slideshow_plan",
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a preview video directly from Gemini slideshow_plan.json and edit_structure.json.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input slideshow_plan.json path")
    parser.add_argument("--structure-input", required=True, help="Input edit_structure.json path")
    parser.add_argument("--output", default=None, help="Output preview.mp4 path")
    parser.add_argument("--output-dir", default=None, help="Directory for outputs. Defaults to outputs/<run>/render/")
    parser.add_argument("--run-dir", default=None, help="Shared run directory for the whole workflow")
    parser.add_argument("--root", default=".", help="Workspace root for representative paths")
    parser.add_argument("--plan-output", default=None, help="Optional path to write the bridged renderable plan JSON")
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
    structure_path = Path(args.structure_input)
    plan_output_path = Path(args.plan_output) if args.plan_output is not None else output_path.with_suffix(".edit_plan.json")

    log(f"[gemini-render] input={input_path}")
    log(f"[gemini-render] structure={structure_path}")
    log(f"[gemini-render] output={output_path}")
    log(f"[gemini-render] root={root}")
    start = time.monotonic()

    slideshow_payload = load_slideshow(input_path)
    structure_payload = load_structure(structure_path)
    renderable_plan = build_renderable_plan(slideshow_payload, structure_payload)

    plan_output_path.parent.mkdir(parents=True, exist_ok=True)
    plan_output_path.write_text(json.dumps(renderable_plan, ensure_ascii=False, indent=2), encoding="utf-8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="photos-ooarai-gemini-render-") as tmp_dir:
        work_dir = Path(tmp_dir)
        clips = render.build_clip_list(renderable_plan, root, work_dir)
        if not clips:
            raise RuntimeError("No renderable clips were found in the Gemini slideshow plan")

        list_path = work_dir / "concat.txt"
        render.write_concat_list(clips, list_path)
        render.concat_clips(list_path, output_path)

        report = {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "input": str(input_path),
            "structure_input": str(structure_path),
            "bridged_plan": str(plan_output_path),
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
                "step": "gemini_render",
                "input": str(input_path),
                "structure_input": str(structure_path),
                "bridged_plan": str(plan_output_path),
                "output": str(output_path),
                "report": str(report_path),
                "clip_count": len(clips),
                "elapsed_seconds": round(time.monotonic() - start, 3),
            },
        )

    log(f"[gemini-render] wrote {output_path} ({len(clips)} clips, elapsed={summarize_elapsed(start)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
