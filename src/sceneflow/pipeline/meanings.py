from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from sceneflow.workflow_utils import log, resolve_output_path, summarize_elapsed, write_manifest


DEFAULT_INPUT = "scene_edit_candidates.json"
DEFAULT_OUTPUT = "scene_meanings.json"

OPENING_COUNT = 2
CLOSING_COUNT = 2

OPENING_TAGS = {"風景", "夜景"}
HIGHLIGHT_TAGS = {"食事", "寺社", "人物", "集合写真", "駅"}
TRANSITION_TAGS = {"移動", "駅"}
SUPPORT_TAGS = {"風景", "建物"}


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


def format_duration_label(seconds: object) -> str:
    value = parse_float(seconds)
    if value is None:
        return "不明"
    total = max(0, int(round(value)))
    if total < 60:
        return f"{total}秒"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}分{secs:02d}秒"
    hours, rem_minutes = divmod(minutes, 60)
    return f"{hours}時間{rem_minutes:02d}分"


def build_scene_summary(scene: dict[str, object]) -> str:
    duration = format_duration_label(scene.get("duration_seconds"))
    parts: list[str] = [f"{duration}の場面"]

    tag = scene.get("representative_tag")
    if not is_missing(tag):
        parts.append(f"主題:{tag}")

    asset_count = scene.get("asset_count")
    if isinstance(asset_count, int):
        parts.append(f"{asset_count}素材")

    gps = scene.get("gps") if isinstance(scene.get("gps"), dict) else {}
    if gps.get("has_gps"):
        parts.append("GPSあり")

    rep = scene.get("representative") if isinstance(scene.get("representative"), dict) else {}
    caption = rep.get("caption") if isinstance(rep, dict) else None
    if not is_missing(caption):
        parts.append(str(caption))

    return " / ".join(parts)


def classify_role(scene: dict[str, object], index: int, total: int) -> str:
    tag = str(scene.get("representative_tag") or "")
    priority = str(scene.get("priority_band") or "low")
    importance = parse_float(scene.get("importance_score")) or 0.0

    if index < OPENING_COUNT:
        return "opening"
    if index >= max(0, total - CLOSING_COUNT):
        return "closing"
    if tag in TRANSITION_TAGS:
        return "transition"
    if tag in HIGHLIGHT_TAGS:
        return "highlight"
    if tag in OPENING_TAGS and priority != "low":
        return "establishing"
    if importance >= 0.75:
        return "highlight"
    if tag in SUPPORT_TAGS or priority == "low":
        return "support"
    return "flow"


def classify_action(role: str, priority_band: str) -> str:
    if role in {"opening", "closing", "highlight"}:
        return "keep"
    if role in {"transition", "establishing", "flow"}:
        return "keep"
    if priority_band == "low":
        return "optional"
    return "support"


def build_scene_record(scene: dict[str, object], index: int, total: int) -> dict[str, object]:
    role = classify_role(scene, index, total)
    priority_band = str(scene.get("priority_band") or "low")
    action = classify_action(role, priority_band)
    summary = build_scene_summary(scene)
    rep = scene.get("representative") if isinstance(scene.get("representative"), dict) else {}
    reasons = list(scene.get("importance_reasons") or [])

    return {
        "scene_id": scene.get("scene_id"),
        "order": index + 1,
        "start_at": scene.get("start_at"),
        "end_at": scene.get("end_at"),
        "duration_seconds": scene.get("duration_seconds"),
        "importance_score": scene.get("importance_score"),
        "priority_band": priority_band,
        "representative_tag": scene.get("representative_tag"),
        "tag_strength": scene.get("tag_strength"),
        "role": role,
        "edit_action": action,
        "summary": summary,
        "flow_summary": scene.get("flow_summary"),
        "chapter_hint": "opening" if role == "opening" else "ending" if role == "closing" else "body",
        "notes": reasons[:6],
        "representative": {
            "path": rep.get("path"),
            "kind": rep.get("kind"),
            "caption": rep.get("caption"),
            "face_count": rep.get("face_count"),
            "blur_score": rep.get("blur_score"),
        },
    }


def build_summary(records: list[dict[str, object]]) -> dict[str, object]:
    role_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    priority_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}

    for record in records:
        role = str(record.get("role") or "flow")
        action = str(record.get("edit_action") or "keep")
        priority = str(record.get("priority_band") or "low")
        tag = str(record.get("representative_tag") or "未判定")

        role_counts[role] = role_counts.get(role, 0) + 1
        action_counts[action] = action_counts.get(action, 0) + 1
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
        tag_counts[tag] = tag_counts.get(tag, 0) + 1

    return {
        "scene_count": len(records),
        "role_counts": dict(sorted(role_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "priority_counts": dict(sorted(priority_counts.items())),
        "representative_tag_counts": dict(sorted(tag_counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def load_candidates(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "scenes" not in payload:
        raise ValueError("scene_edit_candidates.json must contain a top-level 'scenes' array")
    return payload


def build_meanings(input_payload: dict[str, object]) -> dict[str, object]:
    scenes = list(input_payload.get("scenes") or [])
    records = [build_scene_record(scene, index, len(scenes)) for index, scene in enumerate(scenes)]
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": input_payload.get("generated_at"),
        "scene_count": len(records),
        "summary": build_summary(records),
        "scenes": records,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build meaning annotations from scene edit candidates.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input scene_edit_candidates.json path")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--output-dir", default=None, help="Directory for outputs. Defaults to outputs/<run>/meaning/")
    parser.add_argument("--run-dir", default=None, help="Shared run directory for the whole workflow")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = resolve_output_path(
        default_filename=DEFAULT_OUTPUT,
        step_name="meaning",
        output=args.output,
        output_dir=args.output_dir,
        run_dir=args.run_dir,
        input_path=input_path,
    )

    log(f"[meaning] input={input_path}")
    log(f"[meaning] output={output_path}")
    start = time.monotonic()

    input_payload = load_candidates(input_path)
    output_payload = build_meanings(input_payload)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_manifest(
        output_path,
        {
            "generated_at": output_payload["generated_at"],
            "step": "meaning",
            "input": str(input_path),
            "output": str(output_path),
            "scene_count": int(output_payload["scene_count"]),
            "elapsed_seconds": round(time.monotonic() - start, 3),
        },
    )
    log(f"[meaning] wrote {output_path} ({output_payload['scene_count']} scenes, elapsed={summarize_elapsed(start)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
