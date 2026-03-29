from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from sceneflow.workflow_utils import log, resolve_output_path, summarize_elapsed, write_manifest


DEFAULT_INPUT = "edit_structure.json"
DEFAULT_OUTPUT = "edit_plan.json"


def load_structure(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "edit_sequence" not in payload:
        raise ValueError("edit_structure.json must contain a top-level 'edit_sequence' array")
    return payload


def top_tags(structure: dict[str, object], limit: int = 3) -> list[str]:
    tags = Counter()
    for item in structure.get("edit_sequence", []):
        tag = item.get("representative_tag")
        if tag:
            tags[str(tag)] += 1
    return [tag for tag, _ in tags.most_common(limit)]


def build_title(structure: dict[str, object]) -> str:
    tags = top_tags(structure)
    if not tags:
        return "旅の記録"
    if len(tags) == 1:
        return f"{tags[0]}を巡る旅"
    return f"{tags[0]}と{tags[1]}の旅"


def build_logline(structure: dict[str, object]) -> str:
    summary = structure.get("summary", {})
    scene_count = summary.get("scene_count") or structure.get("scene_count") or 0
    chapter_count = summary.get("chapter_count") or len(structure.get("chapters") or [])
    tags = top_tags(structure)
    if tags:
        return f"{scene_count}のsceneを{chapter_count}章でつなぎ、{ '、'.join(tags[:2]) }を軸に自然な旅の流れを作る。"
    return f"{scene_count}のsceneを{chapter_count}章でつなぎ、自然な旅の流れを作る。"


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
        "出力 JSON には title, logline, chapter_list, edit_sequence_notes, narration_plan, subtitle_plan を含めてください。",
        "各 chapter は scene_id の配列を保持し、どの scene を強調するか明示してください。",
        "skip は最終手段とし、基本は並び替えと尺調整で解決してください。",
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


def build_draft_plan(structure: dict[str, object]) -> dict[str, object]:
    chapters = chapter_outline(structure)
    sequence = list(structure.get("edit_sequence", []))
    title = build_title(structure)
    logline = build_logline(structure)

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "generated_by": "heuristic_fallback",
        "title": title,
        "logline": logline,
        "chapter_list": chapters,
        "edit_sequence": sequence,
        "edit_sequence_notes": [
            {
                "scene_id": item.get("scene_id"),
                "chapter_id": item.get("chapter_id"),
                "edit_action": item.get("edit_action"),
                "transition_hint": item.get("transition_hint"),
                "planned_duration_seconds": item.get("planned_duration_seconds"),
            }
            for item in sequence
        ],
        "narration_plan": [
            {
                "chapter_id": chapter.get("chapter_id"),
                "narration": "なし",
                "reason": "映像の流れを優先し、現段階ではナレーションを足さない",
            }
            for chapter in chapters
        ],
        "subtitle_plan": {
            "style": "minimal",
            "enabled": False,
            "notes": "まずは映像の流れを固める。必要なら後でタイトル字幕を追加する。",
        },
        "render_guidance": {
            "preferred_order": [item.get("scene_id") for item in sequence],
            "use_optional_scenes": True,
            "chapter_titles": [chapter.get("title") for chapter in chapters],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an LLM-ready edit plan and prompt.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input edit_structure.json path")
    parser.add_argument("--output", default=None, help="Output edit_plan.json path")
    parser.add_argument("--prompt-output", default=None, help="Output prompt markdown path")
    parser.add_argument("--output-dir", default=None, help="Directory for outputs. Defaults to outputs/<run>/llm/")
    parser.add_argument("--run-dir", default=None, help="Shared run directory for the whole workflow")
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
