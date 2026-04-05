from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from sceneflow.workflow_utils import log, resolve_output_path, summarize_elapsed, write_manifest


DEFAULT_INPUT = "edit_structure.json"
DEFAULT_OUTPUT = "slideshow_plan.json"
DEFAULT_MODEL = "gemini-2.5-flash-lite"
DEFAULT_TOKEN_BUDGET = 6000

PRESERVE_ROLES = {"opening", "closing", "highlight", "transition"}
JSON_MIME_TYPE = "application/json"
SHORT_SEQUENCE_SUBTITLE_LIMIT = 3
MEDIUM_SEQUENCE_SUBTITLE_LIMIT = 4
LONG_SEQUENCE_SUBTITLE_LIMIT = 6
UNSUPPORTED_TEXT_HINTS = ("笑顔", "感動", "BGM", "bgm", "広角", "クローズアップ", "夕暮れ", "夕方", "表情", "全員", "楽しげ", "楽しさ", "楽しかった", "挿入")

SLIDESHOW_PLAN_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "logline", "chapter_list", "scene_directions", "subtitle_plan", "ending_note"],
    "properties": {
        "title": {"type": "string"},
        "logline": {"type": "string"},
        "chapter_list": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["chapter_id", "title", "purpose", "scene_ids", "editing_note"],
                "properties": {
                    "chapter_id": {"type": "string"},
                    "title": {"type": "string"},
                    "purpose": {"type": "string"},
                    "scene_ids": {"type": "array", "items": {"type": "integer"}},
                    "editing_note": {"type": "string"},
                },
            },
        },
        "scene_directions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["scene_ids", "chapter_id", "emphasis", "recommended_duration_seconds", "direction"],
                "properties": {
                    "scene_ids": {"type": "array", "items": {"type": "integer"}},
                    "chapter_id": {"type": "string"},
                    "emphasis": {"type": "string"},
                    "recommended_duration_seconds": {"type": "number"},
                    "direction": {"type": "string"},
                    "subtitle": {"type": "string"},
                },
            },
        },
        "subtitle_plan": {
            "type": "object",
            "additionalProperties": False,
            "required": ["enabled", "style", "items"],
            "properties": {
                "enabled": {"type": "boolean"},
                "style": {"type": "string"},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["scene_ids", "text"],
                        "properties": {
                            "scene_ids": {"type": "array", "items": {"type": "integer"}},
                            "text": {"type": "string"},
                        },
                    },
                },
            },
        },
        "ending_note": {"type": "string"},
    },
}


def is_missing(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "none"


def normalize_text(value: object) -> str:
    if is_missing(value):
        return ""
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split()).strip()


def shorten_text(value: object, limit: int = 160) -> str:
    text = normalize_text(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def parse_float(value: object, default: float = 0.0) -> float:
    if is_missing(value):
        return default
    try:
        return float(value)
    except Exception:
        return default


def load_structure(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "edit_sequence" not in payload:
        raise ValueError("edit_structure.json must contain a top-level 'edit_sequence' array")
    return payload


def resolve_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        return api_key
    raise RuntimeError("Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY before running the gemini step.")


def build_gemini_client(api_key: str | None = None):
    from google import genai

    return genai.Client(api_key=api_key or resolve_api_key())


def usage_metadata_to_dict(usage_metadata: object) -> dict[str, object]:
    if usage_metadata is None:
        return {}
    if hasattr(usage_metadata, "model_dump"):
        return dict(usage_metadata.model_dump(exclude_none=True))
    if isinstance(usage_metadata, dict):
        return dict(usage_metadata)
    return {}


def coerce_scene_id(value: object) -> int | None:
    if is_missing(value):
        return None
    try:
        scene_id = int(value)
    except Exception:
        return None
    return scene_id if scene_id > 0 else None


def included_sequence(structure: dict[str, object]) -> list[dict[str, object]]:
    sequence: list[dict[str, object]] = []
    for item in list(structure.get("edit_sequence") or []):
        if not isinstance(item, dict):
            continue
        if item.get("include") is False:
            continue
        sequence.append(item)
    return sequence


def scene_order_map(sequence: list[dict[str, object]]) -> dict[int, int]:
    order_map: dict[int, int] = {}
    for index, item in enumerate(sequence):
        scene_id = coerce_scene_id(item.get("scene_id"))
        if scene_id is None or scene_id in order_map:
            continue
        order_map[scene_id] = index
    return order_map


def sequence_map(sequence: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    mapping: dict[int, dict[str, object]] = {}
    for item in sequence:
        scene_id = coerce_scene_id(item.get("scene_id"))
        if scene_id is None or scene_id in mapping:
            continue
        mapping[scene_id] = item
    return mapping


def filtered_scene_ids(values: object, *, allowed_scene_ids: set[int], order_map: dict[int, int]) -> list[int]:
    filtered: list[int] = []
    seen: set[int] = set()
    for value in list(values or []):
        scene_id = coerce_scene_id(value)
        if scene_id is None or scene_id not in allowed_scene_ids or scene_id in seen:
            continue
        seen.add(scene_id)
        filtered.append(scene_id)
    return sorted(filtered, key=lambda scene_id: order_map.get(scene_id, 10**9))


def scene_digest_entry(item: dict[str, object]) -> dict[str, object]:
    scene_id = int(item.get("scene_id") or 0)
    return {
        "scene_ids": [scene_id],
        "chapter_id": str(item.get("chapter_id") or "body"),
        "role": str(item.get("role") or "flow"),
        "priority_band": str(item.get("priority_band") or "low"),
        "planned_duration_seconds": round(parse_float(item.get("planned_duration_seconds"), default=2.0), 2),
        "representative_tag": str(item.get("representative_tag") or "未判定"),
        "semantic_summary": shorten_text(item.get("semantic_summary") or item.get("summary") or item.get("flow_summary"), limit=160),
        "selection_reasons": [normalize_text(reason) for reason in list(item.get("selection_reasons") or []) if normalize_text(reason)][:4],
    }


def is_compressible_scene(entry: dict[str, object]) -> bool:
    return (
        str(entry.get("chapter_id") or "") == "body"
        and str(entry.get("role") or "") not in PRESERVE_ROLES
        and str(entry.get("priority_band") or "low") == "low"
    )


def collapse_scene_group(entries: list[dict[str, object]]) -> dict[str, object]:
    scene_ids: list[int] = []
    tags: list[str] = []
    reasons: list[str] = []
    summaries: list[str] = []
    total_duration = 0.0

    for entry in entries:
        scene_ids.extend(int(scene_id) for scene_id in list(entry.get("scene_ids") or []) if int(scene_id) > 0)
        tag = normalize_text(entry.get("representative_tag"))
        if tag and tag not in tags:
            tags.append(tag)
        for reason in list(entry.get("selection_reasons") or []):
            normalized_reason = normalize_text(reason)
            if normalized_reason and normalized_reason not in reasons:
                reasons.append(normalized_reason)
        summary = normalize_text(entry.get("semantic_summary"))
        if summary:
            summaries.append(summary)
        total_duration += parse_float(entry.get("planned_duration_seconds"), default=0.0)

    summary_text = " / ".join(summaries[:2]) if summaries else f"{len(scene_ids)} scene のつなぎ"
    return {
        "scene_ids": scene_ids,
        "chapter_id": str(entries[0].get("chapter_id") or "body"),
        "role": "grouped_support",
        "priority_band": "low",
        "planned_duration_seconds": round(total_duration, 2),
        "representative_tag": ",".join(tags[:3]) if tags else "flow",
        "semantic_summary": shorten_text(summary_text, limit=180),
        "selection_reasons": reasons[:3],
        "grouped_scene_count": len(scene_ids),
    }


def compress_scene_entries(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    compressed: list[dict[str, object]] = []
    buffer: list[dict[str, object]] = []

    def flush_buffer() -> None:
        nonlocal buffer
        if not buffer:
            return
        if len(buffer) == 1:
            compressed.extend(buffer)
        else:
            compressed.append(collapse_scene_group(buffer))
        buffer = []

    for entry in entries:
        if is_compressible_scene(entry):
            buffer.append(entry)
            continue
        flush_buffer()
        compressed.append(entry)

    flush_buffer()
    return compressed


def trim_scene_entries(entries: list[dict[str, object]], *, summary_limit: int = 120, max_reasons: int = 2) -> list[dict[str, object]]:
    trimmed: list[dict[str, object]] = []
    for entry in entries:
        copied = dict(entry)
        copied["semantic_summary"] = shorten_text(copied.get("semantic_summary"), limit=summary_limit)
        copied["selection_reasons"] = [normalize_text(reason) for reason in list(copied.get("selection_reasons") or []) if normalize_text(reason)][:max_reasons]
        trimmed.append(copied)
    return trimmed


def build_prompt_payload(structure: dict[str, object], scene_entries: list[dict[str, object]]) -> dict[str, object]:
    sequence = included_sequence(structure)
    order_map = scene_order_map(sequence)
    sequence_scene_ids = set(order_map)
    allowed_scene_ids = {
        scene_id
        for entry in scene_entries
        for scene_id in filtered_scene_ids(entry.get("scene_ids"), allowed_scene_ids=sequence_scene_ids, order_map=order_map)
    }
    chapters_payload = []
    for chapter in list(structure.get("chapters") or []):
        if not isinstance(chapter, dict):
            continue
        scene_ids = filtered_scene_ids(chapter.get("scene_ids"), allowed_scene_ids=allowed_scene_ids, order_map=order_map)
        if not scene_ids:
            continue
        anchor_scene_ids = filtered_scene_ids(chapter.get("anchor_scene_ids"), allowed_scene_ids=allowed_scene_ids, order_map=order_map)
        chapters_payload.append(
            {
                "chapter_id": chapter.get("chapter_id"),
                "title": chapter.get("title"),
                "purpose": chapter.get("purpose"),
                "pace": chapter.get("pace"),
                "scene_ids": scene_ids,
                "anchor_scene_ids": anchor_scene_ids,
            }
        )

    return {
        "summary": structure.get("summary", {}),
        "chapters": chapters_payload,
        "scene_digest": scene_entries,
    }


def build_prompt_text(payload: dict[str, object]) -> str:
    instructions = [
        "あなたは旅行スライドショーの編集構成を考えるアシスタントです。",
        "動画として見やすい流れ、scene 間のつながり、強弱、余韻を優先してください。",
        "scene_digest は圧縮済みの補助情報です。生素材の推測に広げず、与えられた内容だけで構成してください。",
        "scene_digest に出ていない scene は出力に含めないでください。scene の復活は禁止です。",
        "opening / closing / highlight / transition は大きく崩さないでください。",
        "時間帯、人数、表情、感情、カメラアングル、挿入ショット、BGM の具体像は、入力に明示された場合だけ書いてください。迷ったら一般表現にとどめてください。",
        "subtitle は短く、過剰に説明しすぎないでください。全体尺が短い場合は opening / highlight / closing を優先し、2-4件程度に絞ってください。",
        "出力は必ず JSON のみで返してください。",
    ]
    return "\n".join(
        [
            "# Gemini Slideshow Prompt",
            "",
            "## Instructions",
            *[f"- {line}" for line in instructions],
            "",
            "## Input JSON",
            "```json",
            json.dumps(payload, ensure_ascii=False, indent=2),
            "```",
        ]
    )


def count_prompt_tokens(client: object, *, model: str, prompt_text: str) -> int:
    response = client.models.count_tokens(model=model, contents=prompt_text)
    return int(getattr(response, "total_tokens", 0) or 0)


def fit_prompt_to_budget(client: object, *, structure: dict[str, object], model: str, token_budget: int) -> tuple[str, dict[str, object], int]:
    base_entries = [scene_digest_entry(item) for item in included_sequence(structure)]
    candidate_sets = [
        base_entries,
        compress_scene_entries(base_entries),
        trim_scene_entries(compress_scene_entries(base_entries)),
    ]

    best_prompt = ""
    best_payload: dict[str, object] = {}
    best_token_count = 0

    for entries in candidate_sets:
        payload = build_prompt_payload(structure, entries)
        prompt_text = build_prompt_text(payload)
        token_count = count_prompt_tokens(client, model=model, prompt_text=prompt_text)
        best_prompt = prompt_text
        best_payload = payload
        best_token_count = token_count
        if token_count <= token_budget:
            return prompt_text, payload, token_count

    return best_prompt, best_payload, best_token_count


def default_editing_note(chapter_id: str, scene_ids: list[int]) -> str:
    if chapter_id == "opening":
        return "導入は静かに入り、旅の空気をつかませる。"
    if chapter_id == "closing":
        return "締めは余韻を優先し、最後の印象を整える。"
    if len(scene_ids) <= 1:
        return "見どころを素直に見せる。"
    return "場面のつながりを優先し、テンポよくつなぐ。"


def default_scene_direction(item: dict[str, object]) -> str:
    tag = normalize_text(item.get("representative_tag")) or "旅の場面"
    role = normalize_text(item.get("role")) or "flow"
    summary = shorten_text(item.get("semantic_summary") or item.get("summary") or item.get("flow_summary"), limit=60)
    if role == "opening":
        return shorten_text(f"{tag}から静かに導入し、旅の空気をつかませる。", limit=72)
    if role == "closing":
        return shorten_text(f"{tag}で締め、旅の余韻を自然に残す。", limit=72)
    if role == "highlight":
        return shorten_text(f"{tag}を見どころとして素直に見せる。{summary}", limit=72)
    return shorten_text(f"{tag}の流れを自然につなぐ。{summary}", limit=72)


def subtitle_limit_for_structure(structure: dict[str, object], sequence: list[dict[str, object]]) -> int:
    if not sequence:
        return 0
    summary = structure.get("summary") if isinstance(structure.get("summary"), dict) else {}
    sequence_length_seconds = parse_float(summary.get("sequence_length_seconds"), default=0.0)
    scene_count = len(sequence)
    if sequence_length_seconds <= 0:
        return min(MEDIUM_SEQUENCE_SUBTITLE_LIMIT, scene_count)
    if sequence_length_seconds <= 30:
        return min(SHORT_SEQUENCE_SUBTITLE_LIMIT, scene_count)
    if sequence_length_seconds <= 60:
        return min(MEDIUM_SEQUENCE_SUBTITLE_LIMIT, scene_count)
    return min(LONG_SEQUENCE_SUBTITLE_LIMIT, scene_count)


def subtitle_rank(item: dict[str, object]) -> tuple[int, int, int]:
    role = normalize_text(item.get("role")) or "flow"
    role_score = {
        "highlight": 4,
        "opening": 3,
        "closing": 3,
        "transition": 2,
    }.get(role, 1)
    selected_score = 2 if bool(item.get("selected_for_edit")) else 0
    priority_score = {
        "high": 2,
        "medium": 1,
    }.get(str(item.get("priority_band") or "low"), 0)
    return (role_score, selected_score, priority_score)


def scene_supports_group(item: dict[str, object]) -> bool:
    tag = normalize_text(item.get("representative_tag"))
    primary_type = normalize_text(item.get("primary_type"))
    semantic_summary = normalize_text(item.get("semantic_summary"))
    return tag == "集合写真" or primary_type == "group" or "人物:複数" in semantic_summary or "主題:集合写真" in semantic_summary


def contains_unsupported_scene_detail(text: str, item: dict[str, object]) -> bool:
    normalized_text = normalize_text(text)
    if not normalized_text:
        return False
    if any(hint in normalized_text for hint in UNSUPPORTED_TEXT_HINTS):
        return True
    if "グループ写真" in normalized_text and not scene_supports_group(item):
        return True
    return False


def default_logline(structure: dict[str, object]) -> str:
    sequence = included_sequence(structure)
    scene_count = len(sequence)
    chapters = [chapter for chapter in list(structure.get("chapters") or []) if isinstance(chapter, dict)]
    chapter_count = len(chapters) or 1
    tags: list[str] = []
    for item in sequence:
        tag = normalize_text(item.get("representative_tag"))
        if tag and tag not in tags:
            tags.append(tag)
    if tags:
        return f"{scene_count} scene を {chapter_count} 章でつなぎ、{ 'と'.join(tags[:2]) }を軸に旅の流れを整える。"
    return f"{scene_count} scene を {chapter_count} 章でつなぎ、自然な旅の流れを整える。"


def default_ending_note(structure: dict[str, object]) -> str:
    sequence = included_sequence(structure)
    if not sequence:
        return "scene の順番と尺を見ながら、全体の流れが自然に見えるよう微調整してください。"
    return "scene のつながり、尺、字幕の密度を見ながら、全体の流れが自然に見えるよう微調整してください。"


def sanitize_chapter_list(
    chapter_list: object,
    *,
    allowed_scene_ids: set[int],
    order_map: dict[int, int],
    fallback_sequence: list[dict[str, object]],
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for chapter in list(chapter_list or []):
        if not isinstance(chapter, dict):
            continue
        chapter_id = normalize_text(chapter.get("chapter_id")) or "body"
        scene_ids = filtered_scene_ids(chapter.get("scene_ids"), allowed_scene_ids=allowed_scene_ids, order_map=order_map)
        if not scene_ids:
            continue
        items.append(
            {
                "chapter_id": chapter_id,
                "title": normalize_text(chapter.get("title")) or "旅の流れ",
                "purpose": normalize_text(chapter.get("purpose")) or "scene を自然につなぐ",
                "scene_ids": scene_ids,
                "editing_note": default_editing_note(chapter_id, scene_ids)
                if contains_unsupported_scene_detail(normalize_text(chapter.get("editing_note")), {"chapter_id": chapter_id})
                else normalize_text(chapter.get("editing_note")) or default_editing_note(chapter_id, scene_ids),
            }
        )

    if items:
        return items

    chapters_by_id: dict[str, list[int]] = {}
    for item in fallback_sequence:
        scene_id = coerce_scene_id(item.get("scene_id"))
        if scene_id is None:
            continue
        chapter_id = normalize_text(item.get("chapter_id")) or "body"
        chapters_by_id.setdefault(chapter_id, []).append(scene_id)

    fallback_titles = {
        "opening": ("旅のはじまり", "導入として場所と気分をつかませる"),
        "body": ("旅の流れ", "場面の変化と見どころをつなぐ"),
        "closing": ("旅の余韻", "締めとして印象を残す"),
    }
    return [
        {
            "chapter_id": chapter_id,
            "title": fallback_titles.get(chapter_id, ("旅の流れ", "scene を自然につなぐ"))[0],
            "purpose": fallback_titles.get(chapter_id, ("旅の流れ", "scene を自然につなぐ"))[1],
            "scene_ids": scene_ids,
            "editing_note": default_editing_note(chapter_id, scene_ids),
        }
        for chapter_id, scene_ids in chapters_by_id.items()
        if scene_ids
    ]


def sanitize_scene_directions(
    scene_directions: object,
    *,
    allowed_scene_ids: set[int],
    order_map: dict[int, int],
    sequence_by_id: dict[int, dict[str, object]],
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for direction in list(scene_directions or []):
        if not isinstance(direction, dict):
            continue
        scene_ids = filtered_scene_ids(direction.get("scene_ids"), allowed_scene_ids=allowed_scene_ids, order_map=order_map)
        if not scene_ids:
            continue
        primary_item = sequence_by_id.get(scene_ids[0], {})
        entry = {
            "scene_ids": scene_ids,
            "chapter_id": normalize_text(direction.get("chapter_id")) or normalize_text(primary_item.get("chapter_id")) or "body",
            "emphasis": normalize_text(direction.get("emphasis")) or "中",
            "recommended_duration_seconds": round(
                parse_float(
                    direction.get("recommended_duration_seconds"),
                    default=parse_float(primary_item.get("planned_duration_seconds"), default=2.0),
                ),
                2,
            ),
            "direction": default_scene_direction(primary_item)
            if contains_unsupported_scene_detail(normalize_text(direction.get("direction")), primary_item)
            else normalize_text(direction.get("direction")) or default_scene_direction(primary_item),
        }
        subtitle = normalize_text(direction.get("subtitle"))
        if subtitle:
            entry["subtitle"] = subtitle
        items.append(entry)

    if items:
        return items

    fallback: list[dict[str, object]] = []
    for scene_id in sorted(allowed_scene_ids, key=lambda current: order_map.get(current, 10**9)):
        item = sequence_by_id.get(scene_id, {})
        fallback.append(
            {
                "scene_ids": [scene_id],
                "chapter_id": normalize_text(item.get("chapter_id")) or "body",
                "emphasis": "高" if normalize_text(item.get("role")) == "highlight" else "中",
                "recommended_duration_seconds": round(parse_float(item.get("planned_duration_seconds"), default=2.0), 2),
                "direction": default_scene_direction(item),
            }
        )
    return fallback


def sanitize_subtitle_plan(
    subtitle_plan: object,
    *,
    structure: dict[str, object],
    allowed_scene_ids: set[int],
    order_map: dict[int, int],
    sequence_by_id: dict[int, dict[str, object]],
) -> dict[str, object]:
    raw = subtitle_plan if isinstance(subtitle_plan, dict) else {}
    filtered_items: list[dict[str, object]] = []
    seen_keys: set[tuple[int, ...]] = set()
    for item in list(raw.get("items") or []):
        if not isinstance(item, dict):
            continue
        scene_ids = filtered_scene_ids(item.get("scene_ids"), allowed_scene_ids=allowed_scene_ids, order_map=order_map)
        if not scene_ids:
            continue
        text = normalize_text(item.get("text"))
        if not text:
            continue
        scene_key = tuple(scene_ids)
        if scene_key in seen_keys:
            continue
        seen_keys.add(scene_key)
        filtered_items.append({"scene_ids": scene_ids, "text": text})

    sequence = [sequence_by_id[scene_id] for scene_id in sorted(allowed_scene_ids, key=lambda current: order_map.get(current, 10**9)) if scene_id in sequence_by_id]
    limit = subtitle_limit_for_structure(structure, sequence)
    if limit and len(filtered_items) > limit:
        ranked_items = sorted(
            filtered_items,
            key=lambda item: (
                subtitle_rank(sequence_by_id.get(item["scene_ids"][0], {})),
                -order_map.get(item["scene_ids"][0], 10**9),
            ),
            reverse=True,
        )
        selected_keys = {tuple(item["scene_ids"]) for item in ranked_items[:limit]}
        filtered_items = [item for item in filtered_items if tuple(item["scene_ids"]) in selected_keys]
        filtered_items.sort(key=lambda item: order_map.get(item["scene_ids"][0], 10**9))

    return {
        "enabled": bool(filtered_items),
        "style": normalize_text(raw.get("style")) or "シンプル",
        "items": filtered_items,
    }


def sanitize_slideshow_plan(plan: dict[str, object], structure: dict[str, object]) -> dict[str, object]:
    sequence = included_sequence(structure)
    order_map = scene_order_map(sequence)
    sequence_by_id = sequence_map(sequence)
    allowed_scene_ids = set(sequence_by_id)

    sanitized = dict(plan)
    sanitized["title"] = normalize_text(sanitized.get("title")) or "旅の記録"
    raw_logline = normalize_text(sanitized.get("logline"))
    sanitized["logline"] = default_logline(structure) if contains_unsupported_scene_detail(raw_logline, {}) else raw_logline or default_logline(structure)
    sanitized["chapter_list"] = sanitize_chapter_list(
        sanitized.get("chapter_list"),
        allowed_scene_ids=allowed_scene_ids,
        order_map=order_map,
        fallback_sequence=sequence,
    )
    sanitized["scene_directions"] = sanitize_scene_directions(
        sanitized.get("scene_directions"),
        allowed_scene_ids=allowed_scene_ids,
        order_map=order_map,
        sequence_by_id=sequence_by_id,
    )
    sanitized["subtitle_plan"] = sanitize_subtitle_plan(
        sanitized.get("subtitle_plan"),
        structure=structure,
        allowed_scene_ids=allowed_scene_ids,
        order_map=order_map,
        sequence_by_id=sequence_by_id,
    )
    raw_ending_note = normalize_text(sanitized.get("ending_note"))
    sanitized["ending_note"] = default_ending_note(structure) if contains_unsupported_scene_detail(raw_ending_note, {}) else raw_ending_note or default_ending_note(structure)
    return sanitized


def generate_slideshow_plan(
    structure: dict[str, object],
    *,
    client: object,
    model: str,
    token_budget: int,
) -> tuple[dict[str, object], str, dict[str, object]]:
    prompt_text, prompt_payload, prompt_token_count = fit_prompt_to_budget(
        client,
        structure=structure,
        model=model,
        token_budget=token_budget,
    )
    del prompt_payload

    from google.genai import types

    response = client.models.generate_content(
        model=model,
        contents=prompt_text,
        config=types.GenerateContentConfig(
            response_mime_type=JSON_MIME_TYPE,
            response_json_schema=SLIDESHOW_PLAN_SCHEMA,
            temperature=0.2,
            max_output_tokens=2048,
        ),
    )

    plan = getattr(response, "parsed", None)
    if not isinstance(plan, dict):
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Gemini returned an empty response for slideshow_plan")
        plan = json.loads(text)
    if not isinstance(plan, dict):
        raise RuntimeError("Gemini response could not be parsed into a JSON object")
    plan = sanitize_slideshow_plan(plan, structure)

    token_usage = {
        "request_token_count": prompt_token_count,
        **usage_metadata_to_dict(getattr(response, "usage_metadata", None)),
    }
    return plan, prompt_text, token_usage


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a Gemini-based slideshow plan from edit_structure.json.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input edit_structure.json path")
    parser.add_argument("--output", default=None, help="Output slideshow_plan.json path")
    parser.add_argument("--prompt-output", default=None, help="Output prompt markdown path")
    parser.add_argument("--output-dir", default=None, help="Directory for outputs. Defaults to outputs/<run>/gemini/")
    parser.add_argument("--run-dir", default=None, help="Shared run directory for the whole workflow")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model to call")
    parser.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET, help="Maximum prompt token budget before low-priority body scenes are compressed")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = resolve_output_path(
        default_filename=DEFAULT_OUTPUT,
        step_name="gemini",
        output=args.output,
        output_dir=args.output_dir,
        run_dir=args.run_dir,
        input_path=input_path,
    )
    prompt_path = Path(args.prompt_output) if args.prompt_output is not None else output_path.with_suffix(".prompt.md")

    log(f"[gemini] input={input_path}")
    log(f"[gemini] output={output_path}")
    log(f"[gemini] prompt={prompt_path}")
    log(f"[gemini] model={args.model}")
    start = time.monotonic()

    structure = load_structure(input_path)
    client = build_gemini_client()
    plan, prompt_text, token_usage = generate_slideshow_plan(
        structure,
        client=client,
        model=args.model,
        token_budget=max(256, int(args.token_budget)),
    )

    output_payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": str(input_path),
        "prompt_path": str(prompt_path),
        "generated_by": "gemini_api",
        "model": args.model,
        "token_usage": token_usage,
        "plan": plan,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt_text, encoding="utf-8")
    write_manifest(
        output_path,
        {
            "generated_at": output_payload["generated_at"],
            "step": "gemini",
            "input": str(input_path),
            "output": str(output_path),
            "prompt": str(prompt_path),
            "model": args.model,
            "scene_count": int(structure.get("scene_count") or 0),
            "elapsed_seconds": round(time.monotonic() - start, 3),
        },
    )
    log(f"[gemini] wrote {output_path} and {prompt_path} (elapsed={summarize_elapsed(start)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
