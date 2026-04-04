from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import TextIO

from sceneflow.workflow_utils import log, resolve_output_path, summarize_elapsed, write_manifest


DEFAULT_INPUT = "edit_structure.json"
DEFAULT_OUTPUT = "edit_plan.json"

CHAPTER_TITLES = {
    "opening": "旅のはじまり",
    "body": "旅の流れ",
    "closing": "旅の余韻",
}
CHAPTER_RENDER_TITLES = {
    "opening": "Opening",
    "body": "Journey",
    "closing": "Closing",
}
ROLE_SUBTITLE_HINTS = {
    "opening": "旅の始まりを映す場面",
    "transition": "移動の流れで次の場面へつなぐ",
    "highlight": "この旅らしさが伝わる見どころ",
    "establishing": "場所の空気をゆっくり見せる",
    "support": "旅の空気をやわらかくつなぐ",
    "flow": "旅の流れを自然につなぐ",
    "closing": "静かな余韻で旅を締めくくる",
}
DISABLE_RESPONSE = "-"


def load_structure(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "edit_sequence" not in payload:
        raise ValueError("edit_structure.json must contain a top-level 'edit_sequence' array")
    return payload


def is_missing(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "none"


def top_tags(structure: dict[str, object], limit: int = 3) -> list[str]:
    tags = Counter()
    for item in structure.get("edit_sequence", []):
        tag = item.get("representative_tag")
        if tag:
            tags[str(tag)] += 1
    return [tag for tag, _ in tags.most_common(limit)]


def parse_timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def format_date_title(value: object) -> str | None:
    timestamp = parse_timestamp(value)
    if timestamp is None:
        return None
    return f"{timestamp.year}年{timestamp.month}月{timestamp.day}日"


def format_date_render_title(value: object) -> str | None:
    timestamp = parse_timestamp(value)
    if timestamp is None:
        return None
    return timestamp.strftime("%Y.%m.%d")


def build_title_cards(sequence: list[dict[str, object]]) -> list[dict[str, object]]:
    cards: list[dict[str, object]] = []
    previous_date: str | None = None
    previous_chapter: str | None = None

    for item in sequence:
        scene_id = item.get("scene_id")
        chapter_id = str(item.get("chapter_id") or "body")
        date_title = format_date_title(item.get("start_at"))
        date_key = date_title or ""
        date_changed = date_key != previous_date
        chapter_changed = chapter_id != previous_chapter

        title: str | None = None
        subtitle: str | None = None
        kind = "chapter"
        duration_seconds = 1.2

        if date_title is not None and date_changed:
            title = date_title
            subtitle = CHAPTER_TITLES.get(chapter_id)
            kind = "date"
            duration_seconds = 1.8
        elif chapter_changed:
            title = CHAPTER_TITLES.get(chapter_id, chapter_id)
            kind = "chapter"
            duration_seconds = 1.2

        if title:
            cards.append(
                {
                    "scene_id": scene_id,
                    "kind": kind,
                    "presentation": "overlay",
                    "title": title,
                    "subtitle": subtitle,
                    "render_title": format_date_render_title(item.get("start_at")) if kind == "date" else CHAPTER_RENDER_TITLES.get(chapter_id, chapter_id.title()),
                    "render_subtitle": CHAPTER_RENDER_TITLES.get(chapter_id, chapter_id.title()) if kind == "date" else None,
                    "duration_seconds": duration_seconds,
                }
            )

        previous_date = date_key
        previous_chapter = chapter_id

    return cards


def normalize_text(value: object) -> str:
    if is_missing(value):
        return ""
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split()).strip()


def shorten_text(text: str, limit: int = 30) -> str:
    normalized = normalize_text(text)
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 1)].rstrip() + "…"


def clean_representative_caption(value: object) -> str:
    text = normalize_text(value)
    if not text:
        return ""

    replacements = [
        ("一枚です。", ""),
        ("一枚です", ""),
        ("写真です。", ""),
        ("写真です", ""),
        ("カットです。", ""),
        ("カットです", ""),
    ]
    for before, after in replacements:
        text = text.replace(before, after)

    text = text.strip("。 ")
    if not text:
        return ""
    if text.endswith(("伝わる", "捉えた", "映した", "見せる", "残した")):
        text = f"{text}場面"
    return shorten_text(text, limit=26)


def prepare_sequence(sequence: list[dict[str, object]]) -> list[dict[str, object]]:
    prepared: list[dict[str, object]] = []
    for item in sequence:
        if not isinstance(item, dict):
            continue
        copied = dict(item)
        if is_missing(copied.get("base_planned_duration_seconds")):
            copied["base_planned_duration_seconds"] = copied.get("planned_duration_seconds")
        prepared.append(copied)
    return prepared


def default_editor_brief() -> dict[str, object]:
    return {
        "audience": "",
        "tone": "gentle",
        "pacing": "steady",
        "caption_density": "standard",
        "focus_scene_ids": [],
        "focus_keywords": [],
        "downplay_scene_ids": [],
        "downplay_keywords": [],
        "must_mention": [],
        "ending_feel": "",
        "caption_style_notes": "",
        "resolved_focus_scene_ids": [],
        "resolved_downplay_scene_ids": [],
    }


def split_prompt_tokens(value: object) -> list[str]:
    text = normalize_text(value)
    if not text:
        return []
    parts = re.split(r"[,\u3001/\u30fb]+", text)
    return [part.strip() for part in parts if part.strip()]


def normalize_tone(value: object) -> str:
    text = normalize_text(value).lower()
    if any(token in text for token in ["fact", "事実", "説明", "simple", "plain"]):
        return "factual"
    if any(token in text for token in ["bright", "light", "playful", "明", "軽", "楽"]):
        return "bright"
    if any(token in text for token in ["cinematic", "emotional", "余韻", "情緒", "映画", "しみ"]):
        return "cinematic"
    return "gentle"


def normalize_pacing(value: object) -> str:
    text = normalize_text(value).lower()
    if any(token in text for token in ["snappy", "fast", "quick", "早", "短", "テンポ速"]):
        return "snappy"
    if any(token in text for token in ["calm", "slow", "steady slow", "ゆっくり", "静", "余韻"]):
        return "calm"
    return "steady"


def normalize_caption_density(value: object) -> str:
    text = normalize_text(value).lower()
    if any(token in text for token in ["minimal", "light", "少", "控えめ"]):
        return "minimal"
    if any(token in text for token in ["guided", "dense", "多", "多め", "しっかり"]):
        return "guided"
    return "standard"


def parse_focus_value(value: object) -> tuple[list[int], list[str]]:
    scene_ids: list[int] = []
    keywords: list[str] = []
    for token in split_prompt_tokens(value):
        if token.isdigit():
            scene_id = int(token)
            if scene_id not in scene_ids:
                scene_ids.append(scene_id)
            continue
        if token not in keywords:
            keywords.append(token)
    return scene_ids, keywords


def text_haystack(item: dict[str, object]) -> str:
    parts = [
        item.get("scene_id"),
        item.get("representative_tag"),
        item.get("summary"),
        item.get("flow_summary"),
        item.get("representative_caption"),
    ]
    return normalize_text(" ".join(str(part) for part in parts if not is_missing(part))).lower()


def matches_targets(item: dict[str, object], scene_ids: list[int], keywords: list[str]) -> bool:
    try:
        current_scene_id = int(item.get("scene_id"))
    except Exception:
        current_scene_id = None
    if current_scene_id is not None and current_scene_id in scene_ids:
        return True

    haystack = text_haystack(item)
    return any(normalize_text(keyword).lower() in haystack for keyword in keywords if normalize_text(keyword))


def is_focus_scene(item: dict[str, object], brief: dict[str, object]) -> bool:
    return matches_targets(
        item,
        list(brief.get("focus_scene_ids") or []),
        list(brief.get("focus_keywords") or []),
    )


def is_downplay_scene(item: dict[str, object], brief: dict[str, object]) -> bool:
    if is_focus_scene(item, brief):
        return False
    return matches_targets(
        item,
        list(brief.get("downplay_scene_ids") or []),
        list(brief.get("downplay_keywords") or []),
    )


def pacing_multiplier(item: dict[str, object], brief: dict[str, object]) -> float:
    pacing = str(brief.get("pacing") or "steady")
    role = normalize_text(item.get("role")) or "flow"
    if pacing == "calm":
        if role in {"opening", "closing", "establishing"}:
            return 1.18
        if role == "transition":
            return 0.96
        return 1.08
    if pacing == "snappy":
        if role in {"opening", "closing"}:
            return 0.96
        if role == "highlight":
            return 1.0
        return 0.88
    return 1.0


def apply_editor_brief_to_sequence(sequence: list[dict[str, object]], brief: dict[str, object]) -> list[dict[str, object]]:
    updated: list[dict[str, object]] = []
    focus_scene_ids: list[int] = []
    downplay_scene_ids: list[int] = []

    for item in sequence:
        copied = dict(item)
        base_duration = float(copied.get("base_planned_duration_seconds") or copied.get("planned_duration_seconds") or 2.0)
        focus = is_focus_scene(copied, brief)
        downplay = is_downplay_scene(copied, brief)

        multiplier = pacing_multiplier(copied, brief)
        if focus:
            multiplier *= 1.22
        elif downplay:
            multiplier *= 0.82

        role = normalize_text(copied.get("role")) or "flow"
        if focus and role in {"opening", "closing", "highlight"}:
            multiplier = max(multiplier, 1.18)
        if downplay and normalize_text(copied.get("edit_action")) in {"support", "optional"}:
            multiplier = min(multiplier, 0.75)

        updated_duration = max(0.8, min(base_duration * multiplier, 4.8))
        copied["planned_duration_seconds"] = round(updated_duration, 2)
        copied["editorial_emphasis"] = "focus" if focus else "downplay" if downplay else "default"

        try:
            scene_id = int(copied.get("scene_id"))
        except Exception:
            scene_id = None
        if focus and scene_id is not None:
            focus_scene_ids.append(scene_id)
        if downplay and scene_id is not None:
            downplay_scene_ids.append(scene_id)
        updated.append(copied)

    brief["resolved_focus_scene_ids"] = sorted(set(focus_scene_ids))
    brief["resolved_downplay_scene_ids"] = sorted(set(downplay_scene_ids))
    return updated


def should_include_subtitle(item: dict[str, object], brief: dict[str, object]) -> bool:
    density = str(brief.get("caption_density") or "standard")
    focus = is_focus_scene(item, brief)
    downplay = is_downplay_scene(item, brief)
    role = normalize_text(item.get("role")) or "flow"

    if density == "minimal":
        return focus or role in {"opening", "closing", "highlight"}
    if density == "standard":
        if downplay and role not in {"opening", "closing", "highlight"}:
            return False
        return True
    return True


def build_scene_subtitle(item: dict[str, object], brief: dict[str, object] | None = None) -> str:
    brief = brief or default_editor_brief()
    caption = clean_representative_caption(item.get("representative_caption"))
    tag = normalize_text(item.get("representative_tag"))
    role = normalize_text(item.get("role")) or "flow"
    tone = str(brief.get("tone") or "gentle")
    ending_feel = normalize_text(brief.get("ending_feel"))
    focus = is_focus_scene(item, brief)

    if role == "closing" and ending_feel:
        if tone == "factual":
            return shorten_text(f"{ending_feel}で締める場面", limit=24)
        return shorten_text(f"{ending_feel}を残して締めくくる", limit=24)

    if focus:
        if tone == "factual":
            return caption or shorten_text(f"{tag or '見どころ'}を押さえる場面", limit=24)
        if tone == "cinematic":
            return shorten_text(f"{tag or 'この場面'}が旅の記憶として残る", limit=24)
        return shorten_text(f"{tag or 'この場面'}を旅の見せ場として残す", limit=24)

    if caption and tone != "factual":
        return caption

    if tone == "factual":
        if tag:
            return shorten_text(f"{tag}の場面", limit=18)
        return shorten_text(f"{role}の場面", limit=18)
    if tone == "bright":
        if tag and role == "opening":
            return shorten_text(f"{tag}から気分が上がっていく", limit=24)
        if tag and role in {"support", "flow"}:
            return shorten_text(f"{tag}で旅気分をつないでいく", limit=24)
    if tone == "cinematic":
        if role == "opening":
            return "旅の気配が静かに立ち上がる"
        if role == "transition":
            return "移動のリズムで次の場面へつなぐ"
        if tag:
            return shorten_text(f"{tag}の余韻を丁寧に拾う", limit=24)

    if tag and role == "opening":
        return shorten_text(f"{tag}から旅が始まる", limit=24)
    if tag and role == "closing":
        return shorten_text(f"{tag}の余韻で旅を締めくくる", limit=24)
    if tag and role in {"highlight", "establishing"}:
        return shorten_text(f"{tag}が印象に残る見どころ", limit=24)
    if tag and role in {"support", "flow"}:
        return shorten_text(f"{tag}の空気を自然につなぐ", limit=24)
    if caption:
        return caption
    return ROLE_SUBTITLE_HINTS.get(role, ROLE_SUBTITLE_HINTS["flow"])


def build_subtitle_plan(sequence: list[dict[str, object]], brief: dict[str, object] | None = None) -> dict[str, object]:
    brief = brief or default_editor_brief()
    items: list[dict[str, object]] = []
    for index, item in enumerate(sequence):
        scene_id = item.get("scene_id")
        if is_missing(scene_id):
            continue
        if not should_include_subtitle(item, brief):
            continue
        text = build_scene_subtitle(item, brief)
        if not text:
            continue
        items.append(
            {
                "scene_id": scene_id,
                "text": text,
                "position": "bottom_center",
                "start_seconds": 0.35 if index == 0 else 0.45,
                "duration_seconds": 2.8 if index == 0 else 2.5,
                "origin": "auto",
            }
        )

    return {
        "style": "overlay",
        "enabled": bool(items),
        "items": items,
        "notes": "scene の冒頭に短いテロップをオーバーレイし、人間の brief をもとに密度と文面を調整する。",
    }


def build_title(structure: dict[str, object], brief: dict[str, object] | None = None) -> str:
    brief = brief or default_editor_brief()
    keywords = list(brief.get("must_mention") or [])
    if keywords:
        lead = normalize_text(keywords[0])
        if lead:
            return shorten_text(f"{lead}の旅", limit=24)

    tags = top_tags(structure)
    if not tags:
        return "旅の記録"
    if len(tags) == 1:
        return f"{tags[0]}を巡る旅"
    return f"{tags[0]}と{tags[1]}の旅"


def build_logline(structure: dict[str, object], brief: dict[str, object] | None = None) -> str:
    brief = brief or default_editor_brief()
    summary = structure.get("summary", {})
    scene_count = summary.get("scene_count") or structure.get("scene_count") or 0
    chapter_count = summary.get("chapter_count") or len(structure.get("chapters") or [])
    tags = top_tags(structure)
    tone_phrase = {
        "gentle": "やわらかく",
        "bright": "軽やかに",
        "cinematic": "余韻を残して",
        "factual": "すっきりと",
    }.get(str(brief.get("tone") or "gentle"), "自然に")
    audience = normalize_text(brief.get("audience"))
    ending_feel = normalize_text(brief.get("ending_feel"))

    if tags:
        line = f"{scene_count}のsceneを{chapter_count}章でつなぎ、{ '、'.join(tags[:2]) }を軸に{tone_phrase}旅の流れを作る。"
    else:
        line = f"{scene_count}のsceneを{chapter_count}章でつなぎ、{tone_phrase}自然な旅の流れを作る。"
    if audience:
        line = f"{audience}に向けて、{line}"
    if ending_feel:
        line = line.rstrip("。") + f" 終盤は{ending_feel}を残す。"
    return line


def chapter_outline(structure: dict[str, object]) -> list[dict[str, object]]:
    chapters = []
    for chapter in structure.get("chapters", []):
        chapters.append(
            {
                "chapter_id": chapter.get("chapter_id"),
                "title": chapter.get("title"),
                "purpose": chapter.get("purpose"),
                "pace": chapter.get("pace"),
                "scene_ids": chapter.get("scene_ids", []),
                "scene_count": chapter.get("scene_count", 0),
                "estimated_duration_seconds": chapter.get("estimated_duration_seconds", 0),
            }
        )
    return chapters


def build_prompt(structure: dict[str, object]) -> str:
    instructions = [
        "あなたは旅行映像の編集者です。",
        "次の構造化データをもとに、自然で見やすい動画の構成案を JSON で作ってください。",
        "素材理解そのものではなく、動画としての流れ、リズム、見やすさを最優先してください。",
        "OCR と顔検出は補助情報として扱い、局所最適に入り込みすぎないでください。",
        "出力 JSON には title, logline, chapter_list, edit_sequence_notes, narration_plan, subtitle_plan, title_cards, editor_brief を含めてください。",
        "各 chapter は scene_id の配列を保持し、どの scene を強調するか明示してください。",
        "skip は最終手段とし、基本は並び替えと尺調整で解決してください。",
        "title_cards は日付の切り替わりや章の切り替わりなど、大きな節目だけに絞ってください。",
        "タイトルやテロップは独立したカード差し込みより、映像へのオーバーレイ表示を優先してください。",
        "subtitle_plan は scene_id ごとの短いテロップ文面まで含めてください。",
        "editor_brief があれば、誰に向けた動画か、強調したい scene、テンポ感、終わりの余韻を優先してください。",
    ]
    payload = {
        "summary": structure.get("summary", {}),
        "chapters": chapter_outline(structure),
        "edit_sequence": structure.get("edit_sequence", []),
    }
    return "\n".join(
        [
            "# LLM Prompt",
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


def build_edit_sequence_notes(sequence: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "scene_id": item.get("scene_id"),
            "chapter_id": item.get("chapter_id"),
            "edit_action": item.get("edit_action"),
            "transition_hint": item.get("transition_hint"),
            "planned_duration_seconds": item.get("planned_duration_seconds"),
            "editorial_emphasis": item.get("editorial_emphasis", "default"),
        }
        for item in sequence
    ]


def apply_editor_brief(draft_plan: dict[str, object], brief: dict[str, object]) -> dict[str, object]:
    sequence = prepare_sequence(list(draft_plan.get("edit_sequence") or []))
    adjusted_sequence = apply_editor_brief_to_sequence(sequence, brief)
    draft_plan["editor_brief"] = brief
    draft_plan["edit_sequence"] = adjusted_sequence
    draft_plan["edit_sequence_notes"] = build_edit_sequence_notes(adjusted_sequence)
    draft_plan["title"] = build_title({"edit_sequence": adjusted_sequence, "summary": {"scene_count": len(adjusted_sequence), "chapter_count": len(draft_plan.get("chapter_list") or [])}}, brief)
    draft_plan["logline"] = build_logline({"edit_sequence": adjusted_sequence, "summary": {"scene_count": len(adjusted_sequence), "chapter_count": len(draft_plan.get("chapter_list") or [])}}, brief)
    draft_plan["subtitle_plan"] = build_subtitle_plan(adjusted_sequence, brief)

    render_guidance = draft_plan.get("render_guidance")
    if not isinstance(render_guidance, dict):
        render_guidance = {}
        draft_plan["render_guidance"] = render_guidance
    render_guidance["preferred_order"] = [item.get("scene_id") for item in adjusted_sequence]
    render_guidance["focus_scene_ids"] = list(brief.get("resolved_focus_scene_ids") or [])
    render_guidance["downplay_scene_ids"] = list(brief.get("resolved_downplay_scene_ids") or [])
    render_guidance["caption_density"] = brief.get("caption_density")
    render_guidance["tone"] = brief.get("tone")
    render_guidance["pacing"] = brief.get("pacing")
    return draft_plan


def build_draft_plan(structure: dict[str, object]) -> dict[str, object]:
    chapters = chapter_outline(structure)
    brief = default_editor_brief()
    sequence = prepare_sequence(list(structure.get("edit_sequence", [])))
    sequence = apply_editor_brief_to_sequence(sequence, brief)
    title = build_title({"edit_sequence": sequence, "summary": structure.get("summary", {})}, brief)
    logline = build_logline({"edit_sequence": sequence, "summary": structure.get("summary", {})}, brief)
    title_cards = build_title_cards(sequence)
    subtitle_plan = build_subtitle_plan(sequence, brief)

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "generated_by": "heuristic_fallback",
        "editor_brief": brief,
        "title": title,
        "logline": logline,
        "chapter_list": chapters,
        "title_cards": title_cards,
        "edit_sequence": sequence,
        "edit_sequence_notes": build_edit_sequence_notes(sequence),
        "narration_plan": [
            {
                "chapter_id": chapter.get("chapter_id"),
                "narration": "なし",
                "reason": "映像の流れを優先し、現段階ではナレーションを足さない",
            }
            for chapter in chapters
        ],
        "subtitle_plan": subtitle_plan,
        "render_guidance": {
            "preferred_order": [item.get("scene_id") for item in sequence],
            "use_optional_scenes": True,
            "chapter_titles": [chapter.get("title") for chapter in chapters],
            "title_overlay_mode": "overlay",
            "subtitle_overlay_mode": "overlay",
            "focus_scene_ids": list(brief.get("resolved_focus_scene_ids") or []),
            "downplay_scene_ids": list(brief.get("resolved_downplay_scene_ids") or []),
            "caption_density": brief.get("caption_density"),
            "tone": brief.get("tone"),
            "pacing": brief.get("pacing"),
        },
    }


def describe_scene(item: dict[str, object]) -> str:
    scene_id = item.get("scene_id")
    chapter_id = normalize_text(item.get("chapter_id")) or "body"
    tag = normalize_text(item.get("representative_tag")) or "未判定"
    role = normalize_text(item.get("role")) or "flow"
    return f"scene {scene_id} / {chapter_id} / {role} / {tag}"


def print_scene_guide(sequence: list[dict[str, object]], output_stream: TextIO) -> None:
    print("[llm] scene guide:", file=output_stream)
    for item in sequence:
        if not isinstance(item, dict):
            continue
        summary = normalize_text(item.get("summary")) or normalize_text(item.get("flow_summary"))
        if summary:
            print(f"  - {describe_scene(item)} / {shorten_text(summary, limit=40)}", file=output_stream)
        else:
            print(f"  - {describe_scene(item)}", file=output_stream)


def prompt_text(
    *,
    message: str,
    default: str | None = None,
    allow_disable: bool = False,
    input_stream: TextIO,
    output_stream: TextIO,
) -> str | None:
    print(message, file=output_stream)
    if default is not None:
        print(f"  default: {default}", file=output_stream)
    prompt_label = "  answer"
    if allow_disable:
        prompt_label += f" (Enter=keep, {DISABLE_RESPONSE}=disable)"
    else:
        prompt_label += " (Enter=keep)"
    print(f"{prompt_label}:", end=" ", file=output_stream, flush=True)

    raw = input_stream.readline()
    if raw == "":
        print("", file=output_stream)
        return default

    answer = normalize_text(raw)
    if answer == "":
        return default
    if allow_disable and answer == DISABLE_RESPONSE:
        return None
    return answer


def apply_interactive_overrides(
    draft_plan: dict[str, object],
    *,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> dict[str, object]:
    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stderr

    print("[llm] interactive caption editing enabled", file=output_stream)
    print("[llm] Press Enter to keep the default text, or '-' to disable a telop.", file=output_stream)
    sequence = [item for item in draft_plan.get("edit_sequence", []) if isinstance(item, dict)]
    print_scene_guide(sequence, output_stream)

    brief = draft_plan.get("editor_brief")
    if not isinstance(brief, dict):
        brief = default_editor_brief()

    audience = prompt_text(
        message="誰に見せたい動画かを入れてください。",
        default=normalize_text(brief.get("audience")) or None,
        input_stream=input_stream,
        output_stream=output_stream,
    )
    brief["audience"] = audience or ""

    tone_value = prompt_text(
        message="動画の雰囲気を入れてください。例: やわらかめ / 明るめ / 余韻重視 / 事実寄り",
        default=str(brief.get("tone") or "gentle"),
        input_stream=input_stream,
        output_stream=output_stream,
    )
    brief["tone"] = normalize_tone(tone_value)

    pacing_value = prompt_text(
        message="テンポ感を入れてください。例: ゆっくり / 標準 / 速め",
        default=str(brief.get("pacing") or "steady"),
        input_stream=input_stream,
        output_stream=output_stream,
    )
    brief["pacing"] = normalize_pacing(pacing_value)

    focus_value = prompt_text(
        message="強く見せたい scene id やタグをカンマ区切りで入れてください。",
        default=",".join([*map(str, brief.get("focus_scene_ids") or []), *(brief.get("focus_keywords") or [])]) or None,
        input_stream=input_stream,
        output_stream=output_stream,
    )
    focus_scene_ids, focus_keywords = parse_focus_value(focus_value)
    brief["focus_scene_ids"] = focus_scene_ids
    brief["focus_keywords"] = focus_keywords

    downplay_value = prompt_text(
        message="短めにしたい scene id やタグをカンマ区切りで入れてください。",
        default=",".join([*map(str, brief.get("downplay_scene_ids") or []), *(brief.get("downplay_keywords") or [])]) or None,
        input_stream=input_stream,
        output_stream=output_stream,
    )
    downplay_scene_ids, downplay_keywords = parse_focus_value(downplay_value)
    brief["downplay_scene_ids"] = downplay_scene_ids
    brief["downplay_keywords"] = downplay_keywords

    mention_value = prompt_text(
        message="タイトルやテロップに入れたいキーワードをカンマ区切りで入れてください。",
        default=",".join(brief.get("must_mention") or []) or None,
        input_stream=input_stream,
        output_stream=output_stream,
    )
    brief["must_mention"] = split_prompt_tokens(mention_value)

    ending_feel = prompt_text(
        message="最後に残したい余韻や気持ちを入れてください。",
        default=normalize_text(brief.get("ending_feel")) or None,
        input_stream=input_stream,
        output_stream=output_stream,
    )
    brief["ending_feel"] = ending_feel or ""

    density_value = prompt_text(
        message="テロップ密度を入れてください。例: 控えめ / 標準 / 多め",
        default=str(brief.get("caption_density") or "standard"),
        input_stream=input_stream,
        output_stream=output_stream,
    )
    brief["caption_density"] = normalize_caption_density(density_value)

    draft_plan = apply_editor_brief(draft_plan, brief)

    title = prompt_text(
        message="動画タイトルを確認してください。",
        default=normalize_text(draft_plan.get("title")) or None,
        input_stream=input_stream,
        output_stream=output_stream,
    )
    if title:
        draft_plan["title"] = title

    logline = prompt_text(
        message="動画の一言説明を確認してください。",
        default=normalize_text(draft_plan.get("logline")) or None,
        input_stream=input_stream,
        output_stream=output_stream,
    )
    if logline:
        draft_plan["logline"] = logline

    title_cards = [card for card in draft_plan.get("title_cards", []) if isinstance(card, dict)]
    for card in title_cards:
        label = f"{card.get('kind') or 'title'} overlay for scene {card.get('scene_id')}"
        title_value = prompt_text(
            message=f"{label} のタイトルを確認してください。",
            default=normalize_text(card.get("title")) or None,
            allow_disable=True,
            input_stream=input_stream,
            output_stream=output_stream,
        )
        if title_value is None:
            card["presentation"] = "hidden"
            card["title"] = ""
            card["render_title"] = ""
            card["subtitle"] = None
            card["render_subtitle"] = None
            continue

        card["presentation"] = "overlay"
        card["title"] = title_value
        if card.get("kind") == "date":
            card["render_title"] = title_value

        if not is_missing(card.get("subtitle")):
            subtitle_value = prompt_text(
                message=f"{label} のサブタイトルを確認してください。",
                default=normalize_text(card.get("subtitle")) or None,
                allow_disable=True,
                input_stream=input_stream,
                output_stream=output_stream,
            )
            card["subtitle"] = subtitle_value
            if card.get("kind") == "date":
                card["render_subtitle"] = subtitle_value

    subtitle_plan = draft_plan.get("subtitle_plan")
    if not isinstance(subtitle_plan, dict):
        subtitle_plan = build_subtitle_plan([item for item in draft_plan.get("edit_sequence", []) if isinstance(item, dict)])
        draft_plan["subtitle_plan"] = subtitle_plan

    subtitle_items = [item for item in subtitle_plan.get("items", []) if isinstance(item, dict)]
    subtitles_by_scene = {item.get("scene_id"): item for item in subtitle_items if not is_missing(item.get("scene_id"))}
    updated_items: list[dict[str, object]] = []

    for sequence_item in [item for item in draft_plan.get("edit_sequence", []) if isinstance(item, dict)]:
        scene_id = sequence_item.get("scene_id")
        existing = subtitles_by_scene.get(scene_id)
        default_text = normalize_text(existing.get("text")) if isinstance(existing, dict) else build_scene_subtitle(sequence_item)
        response = prompt_text(
            message=f"{describe_scene(sequence_item)} のテロップを確認してください。",
            default=default_text or None,
            allow_disable=True,
            input_stream=input_stream,
            output_stream=output_stream,
        )
        if response is None:
            continue

        updated_items.append(
            {
                "scene_id": scene_id,
                "text": response,
                "position": "bottom_center",
                "start_seconds": float(existing.get("start_seconds") or 0.45) if isinstance(existing, dict) else 0.45,
                "duration_seconds": float(existing.get("duration_seconds") or 2.5) if isinstance(existing, dict) else 2.5,
                "origin": "interactive" if response != default_text else (existing.get("origin") if isinstance(existing, dict) else "auto"),
            }
        )

    subtitle_plan["enabled"] = bool(updated_items)
    subtitle_plan["style"] = "overlay"
    subtitle_plan["items"] = updated_items
    subtitle_plan["notes"] = "scene 冒頭のテロップをオーバーレイし、対話入力で必要に応じて調整した。"
    return draft_plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an LLM-ready edit plan and prompt.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input edit_structure.json path")
    parser.add_argument("--output", default=None, help="Output edit_plan.json path")
    parser.add_argument("--prompt-output", default=None, help="Output prompt markdown path")
    parser.add_argument("--output-dir", default=None, help="Directory for outputs. Defaults to outputs/<run>/llm/")
    parser.add_argument("--run-dir", default=None, help="Shared run directory for the whole workflow")
    parser.add_argument("--interactive", action="store_true", help="Ask for title/telop overrides before writing edit_plan.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = resolve_output_path(
        default_filename=DEFAULT_OUTPUT,
        step_name="llm",
        output=args.output,
        output_dir=args.output_dir,
        run_dir=args.run_dir,
        input_path=input_path,
    )
    prompt_path = Path(args.prompt_output) if args.prompt_output is not None else output_path.with_suffix(".prompt.md")

    log(f"[llm] input={input_path}")
    log(f"[llm] output={output_path}")
    log(f"[llm] prompt={prompt_path}")
    start = time.monotonic()

    structure = load_structure(input_path)
    prompt_text = build_prompt(structure)
    draft_plan = build_draft_plan(structure)
    if args.interactive:
        draft_plan = apply_interactive_overrides(draft_plan)

    output_payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": str(input_path),
        "prompt_path": str(prompt_path),
        "generated_by": "heuristic_fallback",
        "plan": draft_plan,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt_text, encoding="utf-8")
    write_manifest(
        output_path,
        {
            "generated_at": output_payload["generated_at"],
            "step": "llm",
            "input": str(input_path),
            "output": str(output_path),
            "prompt": str(prompt_path),
            "scene_count": int(structure.get("scene_count") or 0),
            "elapsed_seconds": round(time.monotonic() - start, 3),
        },
    )
    log(f"[llm] wrote {output_path} and {prompt_path} (elapsed={summarize_elapsed(start)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
