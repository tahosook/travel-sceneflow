from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from sceneflow.workflow_utils import log, resolve_output_path, scene_band_counts, summarize_elapsed, write_manifest


DEFAULT_INPUT = "scene_meanings.json"
DEFAULT_OUTPUT = "edit_structure.json"


def is_missing(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "none"


def parse_float(value: object) -> float | None:
    if is_missing(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def normalize_modifiers(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if not is_missing(item)]
    if is_missing(value):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def load_meanings(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "scenes" not in payload:
        raise ValueError("scene_meanings.json must contain a top-level 'scenes' array")
    return payload


def chapter_title(chapter_id: str) -> str:
    return {
        "opening": "旅のはじまり",
        "body": "旅の流れ",
        "closing": "旅の余韻",
    }.get(chapter_id, chapter_id)


def chapter_purpose(chapter_id: str) -> str:
    return {
        "opening": "導入として場所と気分をつかませる",
        "body": "場面の変化と見どころをつなぐ",
        "closing": "締めとして印象を残す",
    }.get(chapter_id, "編集構成の一部")


def chapter_pace(chapter_id: str) -> str:
    return {
        "opening": "ゆるやか",
        "body": "一定",
        "closing": "余韻重視",
    }.get(chapter_id, "一定")


def render_duration_hint(duration_seconds: object, edit_action: str, modifiers: object = None, trip_type: object = None) -> float:
    value = parse_float(duration_seconds) or 0.0
    if edit_action == "optional":
        base_duration = min(max(value * 0.6, 1.0), 1.8)
    elif edit_action == "support":
        base_duration = min(max(value * 0.8, 1.2), 2.4)
    else:
        base_duration = min(max(value, 1.8), 3.5)

    modifiers_list = normalize_modifiers(modifiers)
    if "food" in modifiers_list:
        multiplier = 0.85 if str(trip_type or "general") == "gourmet" else 0.7
        base_duration *= multiplier
    return max(0.8, round(base_duration, 2))


def assign_chapter(index: int, total: int, opening_end: int, closing_start: int) -> str:
    if index < opening_end:
        return "opening"
    if index >= closing_start:
        return "closing"
    return "body"


def build_sequence_item(scene: dict[str, object], index: int, total: int, opening_end: int, closing_start: int) -> dict[str, object]:
    chapter_id = assign_chapter(index, total, opening_end, closing_start)
    edit_action = str(scene.get("edit_action") or "keep")
    role = str(scene.get("role") or "flow")
    priority = str(scene.get("priority_band") or "low")
    modifiers = normalize_modifiers(scene.get("modifiers"))
    trip_type = str(scene.get("trip_type") or "general")
    duration_hint = render_duration_hint(scene.get("duration_seconds"), edit_action, modifiers, trip_type)
    transition_hint = "fade_in" if index == 0 else "fade_out" if index == total - 1 else "cut"
    if chapter_id == "body" and role == "transition":
        transition_hint = "soft_cut"
    representative = scene.get("representative") if isinstance(scene.get("representative"), dict) else {}
    include = bool(scene.get("selected_for_edit")) or chapter_id in {"opening", "closing"}

    return {
        "order": index + 1,
        "scene_id": scene.get("scene_id"),
        "chapter_id": chapter_id,
        "start_at": scene.get("start_at"),
        "end_at": scene.get("end_at"),
        "role": role,
        "edit_action": edit_action,
        "priority_band": priority,
        "include": include,
        "selected_for_edit": bool(scene.get("selected_for_edit")),
        "selection_rank": scene.get("selection_rank"),
        "selection_score": scene.get("selection_score"),
        "primary_type": scene.get("primary_type"),
        "modifiers": modifiers,
        "food_confidence": scene.get("food_confidence"),
        "food_confidence_representative": scene.get("food_confidence_representative"),
        "food_sample_count": scene.get("food_sample_count"),
        "food_evidence_count": scene.get("food_evidence_count"),
        "food_sample_paths": list(scene.get("food_sample_paths") or []),
        "trip_type": trip_type,
        "selection_reasons": list(scene.get("notes") or []),
        "transition_hint": transition_hint,
        "planned_duration_seconds": round(duration_hint, 2),
        "summary": scene.get("summary"),
        "flow_summary": scene.get("flow_summary"),
        "semantic_summary": scene.get("semantic_summary"),
        "semantic_confidence": scene.get("semantic_confidence"),
        "clip_hints": list(scene.get("clip_hints") or []),
        "classification_source": scene.get("classification_source"),
        "representative_tag": scene.get("representative_tag"),
        "preview_sources": list(scene.get("preview_sources") or []),
        "representative_path": representative.get("path"),
        "representative_kind": representative.get("kind"),
        "representative_caption": representative.get("caption"),
    }


def build_chapter(chapter_id: str, scenes: list[dict[str, object]]) -> dict[str, object]:
    tag_counts: dict[str, int] = {}
    role_counts: dict[str, int] = {}
    priority_counts: dict[str, int] = {}
    total_duration = 0.0
    selected_scene_count = 0
    for scene in scenes:
        tag = str(scene.get("representative_tag") or "未判定")
        role = str(scene.get("role") or "flow")
        priority = str(scene.get("priority_band") or "low")
        duration = parse_float(scene.get("duration_seconds")) or 0.0
        total_duration += duration
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
        role_counts[role] = role_counts.get(role, 0) + 1
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
        if bool(scene.get("selected_for_edit")):
            selected_scene_count += 1

    scene_ids = [scene.get("scene_id") for scene in scenes]
    anchors = [
        scene.get("scene_id")
        for scene in scenes
        if bool(scene.get("selected_for_edit")) or str(scene.get("priority_band") or "low") != "low"
    ]
    return {
        "chapter_id": chapter_id,
        "title": chapter_title(chapter_id),
        "purpose": chapter_purpose(chapter_id),
        "pace": chapter_pace(chapter_id),
        "scene_ids": scene_ids,
        "anchor_scene_ids": anchors,
        "scene_count": len(scenes),
        "selected_scene_count": selected_scene_count,
        "estimated_duration_seconds": round(total_duration, 2),
        "tag_counts": dict(sorted(tag_counts.items(), key=lambda item: (-item[1], item[0]))),
        "role_counts": dict(sorted(role_counts.items())),
        "priority_counts": dict(sorted(priority_counts.items())),
    }


def build_structure(meanings: dict[str, object]) -> dict[str, object]:
    scenes = list(meanings.get("scenes") or [])
    total = len(scenes)
    opening_span, closing_span = scene_band_counts(total)
    opening_end = min(total, opening_span)
    closing_start = max(opening_end, total - closing_span)

    sequence = [build_sequence_item(scene, index, total, opening_end, closing_start) for index, scene in enumerate(scenes)]

    opening_scenes = scenes[:opening_end]
    body_scenes = scenes[opening_end:closing_start] if closing_start > opening_end else []
    closing_scenes = scenes[closing_start:] if closing_start < total else []

    chapters = [
        build_chapter("opening", opening_scenes),
        build_chapter("body", body_scenes),
        build_chapter("closing", closing_scenes),
    ]

    summary = {
        "scene_count": total,
        "chapter_count": len(chapters),
        "opening_count": len(opening_scenes),
        "body_count": len(body_scenes),
        "closing_count": len(closing_scenes),
        "selected_scene_count": sum(1 for scene in scenes if bool(scene.get("selected_for_edit"))),
        "trip_type": meanings.get("summary", {}).get("trip_type", "general"),
        "sequence_length_seconds": round(sum(item["planned_duration_seconds"] for item in sequence), 2),
        "priority_counts": meanings.get("summary", {}).get("priority_counts", {}),
    }

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": meanings.get("generated_at"),
        "scene_count": total,
        "summary": summary,
        "chapters": chapters,
        "edit_sequence": sequence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build edit structure from meaning annotations.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input scene_meanings.json path")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--output-dir", default=None, help="Directory for outputs. Defaults to outputs/<run>/structure/")
    parser.add_argument("--run-dir", default=None, help="Shared run directory for the whole workflow")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = resolve_output_path(
        default_filename=DEFAULT_OUTPUT,
        step_name="structure",
        output=args.output,
        output_dir=args.output_dir,
        run_dir=args.run_dir,
        input_path=input_path,
    )

    log(f"[structure] input={input_path}")
    log(f"[structure] output={output_path}")
    start = time.monotonic()

    meanings = load_meanings(input_path)
    output_payload = build_structure(meanings)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_manifest(
        output_path,
        {
            "generated_at": output_payload["generated_at"],
            "step": "structure",
            "input": str(input_path),
            "output": str(output_path),
            "scene_count": int(output_payload["scene_count"]),
            "elapsed_seconds": round(time.monotonic() - start, 3),
        },
    )
    log(f"[structure] wrote {output_path} ({output_payload['scene_count']} scenes, elapsed={summarize_elapsed(start)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
