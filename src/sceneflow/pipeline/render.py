from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from sceneflow.workflow_utils import log, resolve_output_path, summarize_elapsed, write_manifest


DEFAULT_INPUT = "edit_plan.json"
DEFAULT_OUTPUT = "preview.mp4"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
FRAME_W = 1280
FRAME_H = 720
FPS = 30
TITLE_CARD_BG_BGR = (32, 24, 16)
TITLE_CARD_ACCENT_BGR = (222, 241, 244)
TITLE_CARD_SUBTEXT_BGR = (217, 217, 217)
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
OVERLAY_FONT_GLOBS = [
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/ヒラ*W6.ttc",
    "/System/Library/Fonts/ヒラ*W5.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
]
SWIFT_OVERLAY_SCRIPT = r"""
import AppKit
import Foundation

guard CommandLine.arguments.count >= 4 else {
    fputs("usage: overlay_text.swift <output> <style> <text>\n", stderr)
    exit(2)
}

let outputPath = CommandLine.arguments[1]
let style = CommandLine.arguments[2]
let text = CommandLine.arguments[3]

let frameWidth: CGFloat = 1280
let frameHeight: CGFloat = 720

struct Config {
    let fontSize: CGFloat
    let padding: CGFloat
    let maxWidth: CGFloat
    let leftInset: CGFloat
    let topInset: CGFloat
    let bottomInset: CGFloat
    let centered: Bool
}

func config(for style: String) -> Config {
    switch style {
    case "title":
        return Config(fontSize: 44, padding: 18, maxWidth: 540, leftInset: 56, topInset: 50, bottomInset: 0, centered: false)
    case "label":
        return Config(fontSize: 28, padding: 14, maxWidth: 520, leftInset: 56, topInset: 122, bottomInset: 0, centered: false)
    default:
        return Config(fontSize: 36, padding: 16, maxWidth: 880, leftInset: 0, topInset: 0, bottomInset: 56, centered: true)
    }
}

let cfg = config(for: style)
let font = NSFont(name: "Hiragino Sans W6", size: cfg.fontSize)
    ?? NSFont(name: "Arial Unicode MS", size: cfg.fontSize)
    ?? NSFont.systemFont(ofSize: cfg.fontSize, weight: style == "title" ? .semibold : .regular)
let paragraph = NSMutableParagraphStyle()
paragraph.alignment = cfg.centered ? .center : .left
paragraph.lineBreakMode = .byWordWrapping

let attrs: [NSAttributedString.Key: Any] = [
    .font: font,
    .foregroundColor: NSColor.white,
    .paragraphStyle: paragraph,
]

let attributed = NSAttributedString(string: text, attributes: attrs)
let textBounds = attributed.boundingRect(
    with: NSSize(width: cfg.maxWidth, height: frameHeight),
    options: [.usesLineFragmentOrigin, .usesFontLeading]
)
let textSize = NSSize(width: ceil(min(textBounds.width, cfg.maxWidth)), height: ceil(textBounds.height))

let boxRect: NSRect
if cfg.centered {
    let x = floor((frameWidth - textSize.width) / 2.0 - cfg.padding)
    boxRect = NSRect(
        x: x,
        y: cfg.bottomInset - cfg.padding,
        width: textSize.width + cfg.padding * 2.0,
        height: textSize.height + cfg.padding * 2.0
    )
} else {
    boxRect = NSRect(
        x: cfg.leftInset,
        y: frameHeight - cfg.topInset - textSize.height - cfg.padding * 2.0,
        width: textSize.width + cfg.padding * 2.0,
        height: textSize.height + cfg.padding * 2.0
    )
}

let textRect = NSRect(
    x: boxRect.minX + cfg.padding,
    y: boxRect.minY + cfg.padding,
    width: textSize.width,
    height: textSize.height
)

let image = NSImage(size: NSSize(width: frameWidth, height: frameHeight))
image.lockFocus()
NSColor.clear.setFill()
NSRect(x: 0, y: 0, width: frameWidth, height: frameHeight).fill()

NSColor(calibratedWhite: 0.0, alpha: 0.45).setFill()
let boxPath = NSBezierPath(roundedRect: boxRect, xRadius: 16, yRadius: 16)
boxPath.fill()
attributed.draw(with: textRect, options: [.usesLineFragmentOrigin, .usesFontLeading])

image.unlockFocus()

guard
    let tiffData = image.tiffRepresentation,
    let bitmap = NSBitmapImageRep(data: tiffData),
    let pngData = bitmap.representation(using: .png, properties: [:])
else {
    fputs("failed to encode PNG\n", stderr)
    exit(1)
}

do {
    try pngData.write(to: URL(fileURLWithPath: outputPath))
} catch {
    fputs("failed to write PNG: \(error)\n", stderr)
    exit(1)
}
"""


def is_missing(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "none"


def parse_float(value: object, default: float = 0.0) -> float:
    if is_missing(value):
        return default
    try:
        return float(value)
    except Exception:
        return default


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def load_plan(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "plan" not in payload:
        raise ValueError("edit_plan.json must contain a top-level 'plan' object")
    return payload


def parse_timestamp(value: object) -> datetime | None:
    if is_missing(value):
        return None
    try:
        return datetime.fromisoformat(str(value).strip())
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


def title_cards_from_sequence(edit_items: list[dict[str, object]]) -> list[dict[str, object]]:
    cards: list[dict[str, object]] = []
    previous_date: str | None = None
    previous_chapter: str | None = None

    for item in edit_items:
        if not isinstance(item, dict):
            continue

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


def load_title_cards(plan_data: dict[str, object], edit_items: list[dict[str, object]]) -> dict[object, list[dict[str, object]]]:
    raw_cards = plan_data.get("title_cards")
    card_list = [card for card in raw_cards if isinstance(card, dict)] if isinstance(raw_cards, list) else title_cards_from_sequence(edit_items)
    cards_by_scene: dict[object, list[dict[str, object]]] = {}
    for card in card_list:
        scene_id = card.get("scene_id")
        cards_by_scene.setdefault(scene_id, []).append(card)
    return cards_by_scene


def load_subtitle_items(plan_data: dict[str, object]) -> dict[object, list[dict[str, object]]]:
    subtitle_plan = plan_data.get("subtitle_plan")
    if not isinstance(subtitle_plan, dict) or not subtitle_plan.get("enabled"):
        return {}

    raw_items = subtitle_plan.get("items")
    if not isinstance(raw_items, list):
        return {}

    subtitles_by_scene: dict[object, list[dict[str, object]]] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        scene_id = item.get("scene_id")
        subtitles_by_scene.setdefault(scene_id, []).append(item)
    return subtitles_by_scene


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


def discover_overlay_font() -> str | None:
    for pattern in OVERLAY_FONT_GLOBS:
        direct_path = Path(pattern)
        if direct_path.exists():
            return str(direct_path)

        parent = direct_path.parent
        if parent.exists():
            matches = sorted(parent.glob(direct_path.name))
            if matches:
                return str(matches[0])
    return None


@lru_cache(maxsize=None)
def ffmpeg_supports_filter(filter_name: str) -> bool:
    result = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return False
    return filter_name in result.stdout


@lru_cache(maxsize=1)
def overlay_backend() -> str:
    if ffmpeg_supports_filter("drawtext"):
        return "drawtext"
    if ffmpeg_supports_filter("overlay") and command_exists("swiftc"):
        return "overlay_image"
    return "none"


def ensure_swift_overlay_script(work_dir: Path) -> Path:
    script_path = work_dir / "render_overlay_text.swift"
    if not script_path.exists():
        script_path.write_text(SWIFT_OVERLAY_SCRIPT, encoding="utf-8")
    return script_path


def ensure_swift_overlay_binary(work_dir: Path) -> Path:
    script_path = ensure_swift_overlay_script(work_dir)
    binary_path = work_dir / "render_overlay_text"
    if binary_path.exists():
        return binary_path

    module_cache = work_dir / "swift-module-cache"
    module_cache.mkdir(parents=True, exist_ok=True)
    swift_env = dict(os.environ)
    swift_env["CLANG_MODULE_CACHE_PATH"] = str(module_cache)
    swift_env["SWIFT_MODULECACHE_PATH"] = str(module_cache)

    cmd = ["swiftc", str(script_path), "-o", str(binary_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=swift_env)
    if result.returncode != 0 or not binary_path.exists():
        raise RuntimeError(
            f"Failed to compile overlay renderer: {binary_path}\n"
            f"stdout={result.stdout[-1000:]}\n"
            f"stderr={result.stderr[-1000:]}"
        )
    return binary_path


def normalize_overlay_text(value: object) -> str:
    if is_missing(value):
        return ""
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split()).strip()


def wrap_overlay_text(text: str, *, line_limit: int, max_lines: int) -> str:
    normalized = normalize_overlay_text(text)
    if not normalized:
        return ""
    if len(normalized) <= line_limit:
        return normalized

    lines: list[str] = []
    remaining = normalized
    while remaining and len(lines) < max_lines:
        lines.append(remaining[:line_limit])
        remaining = remaining[line_limit:]

    if remaining and lines:
        if len(lines[-1]) >= 1:
            lines[-1] = lines[-1][:-1] + "…"
        else:
            lines[-1] = "…"
    return "\n".join(line for line in lines if line)


def escape_filter_value(value: object) -> str:
    text = str(value)
    text = text.replace("\\", "\\\\")
    text = text.replace(":", r"\:")
    text = text.replace("'", r"\'")
    return f"'{text}'"


def should_render_title_card_as_clip(card: dict[str, object]) -> bool:
    presentation = str(card.get("presentation") or card.get("render_mode") or "overlay").strip().lower()
    return presentation in {"card", "full_card", "standalone"}


def build_scene_overlays(scene_cards: list[dict[str, object]], subtitle_items: list[dict[str, object]]) -> list[dict[str, object]]:
    overlays: list[dict[str, object]] = []
    title_overlay_present = False

    for card in scene_cards:
        presentation = str(card.get("presentation") or card.get("render_mode") or "overlay").strip().lower()
        if presentation in {"hidden", "none", "off"}:
            continue
        if should_render_title_card_as_clip(card):
            continue

        duration_seconds = max(0.8, parse_float(card.get("duration_seconds"), 1.6))
        title = normalize_overlay_text(card.get("title") or card.get("render_title"))
        subtitle = normalize_overlay_text(card.get("subtitle") or card.get("render_subtitle"))

        if title:
            overlays.append(
                {
                    "text": title,
                    "style": "title",
                    "position": "top_left",
                    "start_seconds": 0.0,
                    "duration_seconds": duration_seconds,
                }
            )
            title_overlay_present = True
        if subtitle:
            overlays.append(
                {
                    "text": subtitle,
                    "style": "label",
                    "position": "top_left_secondary",
                    "start_seconds": 0.1,
                    "duration_seconds": duration_seconds,
                }
            )
            title_overlay_present = True

    subtitle_delay = 0.8 if title_overlay_present else 0.2
    for item in subtitle_items:
        text = normalize_overlay_text(item.get("text"))
        if not text:
            continue
        overlays.append(
            {
                "text": text,
                "style": "subtitle",
                "position": str(item.get("position") or "bottom_center"),
                "start_seconds": max(parse_float(item.get("start_seconds"), 0.0), subtitle_delay),
                "duration_seconds": max(1.2, parse_float(item.get("duration_seconds"), 2.5)),
            }
        )

    return overlays


def overlay_style_config(style: str, position: str) -> dict[str, object]:
    if style == "title":
        return {
            "font_size": 44,
            "line_limit": 18,
            "max_lines": 2,
            "x": "56",
            "y": "50",
            "box_border": 18,
        }
    if style == "label":
        return {
            "font_size": 28,
            "line_limit": 24,
            "max_lines": 2,
            "x": "56",
            "y": "122",
            "box_border": 14,
        }
    if position == "bottom_center":
        return {
            "font_size": 36,
            "line_limit": 22,
            "max_lines": 2,
            "x": "(w-text_w)/2",
            "y": "h-text_h-56",
            "box_border": 16,
        }
    return {
        "font_size": 34,
        "line_limit": 22,
        "max_lines": 2,
        "x": "(w-text_w)/2",
        "y": "h-text_h-56",
        "box_border": 16,
    }


def build_overlay_filters(overlays: list[dict[str, object]], output_path: Path, clip_duration: float) -> list[str]:
    if not overlays:
        return []

    fontfile = discover_overlay_font()
    filters: list[str] = []

    for index, overlay in enumerate(overlays, start=1):
        style = str(overlay.get("style") or "subtitle")
        position = str(overlay.get("position") or "bottom_center")
        style_config = overlay_style_config(style, position)
        text = wrap_overlay_text(
            str(overlay.get("text") or ""),
            line_limit=int(style_config["line_limit"]),
            max_lines=int(style_config["max_lines"]),
        )
        if not text:
            continue

        start_seconds = max(0.0, parse_float(overlay.get("start_seconds"), 0.0))
        overlay_duration = max(0.3, parse_float(overlay.get("duration_seconds"), 2.0))
        end_seconds = min(clip_duration, start_seconds + overlay_duration)
        if end_seconds <= start_seconds:
            continue

        textfile_path = output_path.with_name(f"{output_path.stem}_overlay_{index:02d}.txt")
        textfile_path.write_text(text, encoding="utf-8")

        parts = [
            f"textfile={escape_filter_value(textfile_path)}",
            "fontcolor=white",
            f"fontsize={style_config['font_size']}",
            f"x={style_config['x']}",
            f"y={style_config['y']}",
            "line_spacing=10",
            "box=1",
            "boxcolor=black@0.45",
            f"boxborderw={style_config['box_border']}",
            "borderw=1",
            "bordercolor=black@0.25",
            "fix_bounds=true",
            f"enable='between(t,{start_seconds:.2f},{end_seconds:.2f})'",
        ]
        if fontfile:
            parts.insert(1, f"fontfile={escape_filter_value(fontfile)}")
        filters.append("drawtext=" + ":".join(parts))

    return filters


def build_overlay_image_assets(overlays: list[dict[str, object]], output_path: Path, clip_duration: float) -> list[dict[str, object]]:
    if not overlays:
        return []

    binary_path = ensure_swift_overlay_binary(output_path.parent)
    module_cache = output_path.parent / "swift-module-cache"
    module_cache.mkdir(parents=True, exist_ok=True)
    swift_env = dict(os.environ)
    swift_env["CLANG_MODULE_CACHE_PATH"] = str(module_cache)
    swift_env["SWIFT_MODULECACHE_PATH"] = str(module_cache)
    assets: list[dict[str, object]] = []

    for index, overlay in enumerate(overlays, start=1):
        style = str(overlay.get("style") or "subtitle")
        position = str(overlay.get("position") or "bottom_center")
        style_config = overlay_style_config(style, position)
        text = wrap_overlay_text(
            str(overlay.get("text") or ""),
            line_limit=int(style_config["line_limit"]),
            max_lines=int(style_config["max_lines"]),
        )
        if not text:
            continue

        start_seconds = max(0.0, parse_float(overlay.get("start_seconds"), 0.0))
        overlay_duration = max(0.3, parse_float(overlay.get("duration_seconds"), 2.0))
        end_seconds = min(clip_duration, start_seconds + overlay_duration)
        if end_seconds <= start_seconds:
            continue

        asset_path = output_path.with_name(f"{output_path.stem}_overlay_{index:02d}.png")
        cmd = [str(binary_path), str(asset_path), style, text]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=swift_env)
        if result.returncode != 0 or not asset_path.exists():
            raise RuntimeError(
                f"Failed to render overlay asset: {asset_path}\n"
                f"stdout={result.stdout[-1000:]}\n"
                f"stderr={result.stderr[-1000:]}"
            )

        assets.append(
            {
                "path": asset_path,
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
            }
        )

    return assets


def build_overlay_complex_filter(overlay_assets: list[dict[str, object]]) -> tuple[str, str]:
    base_filters = [
        f"[0:v]scale={FRAME_W}:{FRAME_H}:force_original_aspect_ratio=decrease",
        f"pad={FRAME_W}:{FRAME_H}:(ow-iw)/2:(oh-ih)/2",
        f"fps={FPS}",
        "format=rgba[base0]",
    ]
    filter_parts = [",".join(base_filters)]

    current_label = "base0"
    for index, asset in enumerate(overlay_assets, start=1):
        next_label = f"base{index}"
        filter_parts.append(
            f"[{current_label}][{index}:v]overlay=0:0:enable='between(t,{asset['start_seconds']:.2f},{asset['end_seconds']:.2f})'[{next_label}]"
        )
        current_label = next_label

    filter_parts.append(f"[{current_label}]format=yuv420p[outv]")
    return ";".join(filter_parts), "[outv]"


def normalize_clip(input_path: Path, output_path: Path, duration: float, overlays: list[dict[str, object]] | None = None) -> None:
    suffix = input_path.suffix.lower()
    backend = overlay_backend() if overlays else "none"
    vf_filters = [
        f"scale={FRAME_W}:{FRAME_H}:force_original_aspect_ratio=decrease",
        f"pad={FRAME_W}:{FRAME_H}:(ow-iw)/2:(oh-ih)/2",
        "format=yuv420p",
        f"fps={FPS}",
    ]
    if backend == "drawtext":
        vf_filters.extend(build_overlay_filters(overlays or [], output_path, duration))
    vf = ",".join(vf_filters)
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
        source_path = raster_path
        source_cmd = ["-loop", "1", "-t", f"{duration:.2f}", "-i", str(source_path)]
    else:
        source_path = input_path
        source_cmd = ["-ss", "0", "-t", f"{duration:.2f}", "-i", str(source_path)]

    if backend == "overlay_image":
        overlay_assets = build_overlay_image_assets(overlays or [], output_path, duration)
        cmd = ["ffmpeg", "-y", *source_cmd]
        for asset in overlay_assets:
            cmd.extend(["-loop", "1", "-i", str(asset["path"])])
        if overlay_assets:
            filter_complex, output_label = build_overlay_complex_filter(overlay_assets)
            cmd.extend(
                [
                    "-filter_complex",
                    filter_complex,
                    "-map",
                    output_label,
                    "-an",
                    "-shortest",
                    "-t",
                    f"{duration:.2f}",
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ]
            )
        else:
            cmd.extend(
                [
                    "-vf",
                    vf,
                    "-an",
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ]
            )
    else:
        cmd = [
            "ffmpeg",
            "-y",
            *source_cmd,
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


def render_title_card(card: dict[str, object], output_path: Path) -> None:
    title = str(card.get("title") or card.get("render_title") or "").strip()
    subtitle = str(card.get("subtitle") or card.get("render_subtitle") or "").strip()
    if not title:
        raise ValueError("Title card requires a title")

    duration = max(0.8, float(card.get("duration_seconds") or 1.2))
    canvas = np.full((FRAME_H, FRAME_W, 3), TITLE_CARD_BG_BGR, dtype=np.uint8)
    center_x = FRAME_W // 2
    title_font = cv2.FONT_HERSHEY_DUPLEX
    subtitle_font = cv2.FONT_HERSHEY_SIMPLEX

    title_scale = 1.6 if len(title) <= 12 else 1.2
    title_thickness = 3
    title_size, _ = cv2.getTextSize(title, title_font, title_scale, title_thickness)
    title_x = max(40, center_x - title_size[0] // 2)
    title_y = int(FRAME_H * 0.43)
    cv2.putText(canvas, title, (title_x, title_y), title_font, title_scale, (0, 0, 0), title_thickness + 4, cv2.LINE_AA)
    cv2.putText(canvas, title, (title_x, title_y), title_font, title_scale, TITLE_CARD_ACCENT_BGR, title_thickness, cv2.LINE_AA)

    if subtitle:
        subtitle_scale = 0.95
        subtitle_thickness = 2
        subtitle_size, _ = cv2.getTextSize(subtitle, subtitle_font, subtitle_scale, subtitle_thickness)
        subtitle_x = max(40, center_x - subtitle_size[0] // 2)
        subtitle_y = int(FRAME_H * 0.58)
        cv2.putText(canvas, subtitle, (subtitle_x, subtitle_y), subtitle_font, subtitle_scale, (0, 0, 0), subtitle_thickness + 4, cv2.LINE_AA)
        cv2.putText(canvas, subtitle, (subtitle_x, subtitle_y), subtitle_font, subtitle_scale, TITLE_CARD_SUBTEXT_BGR, subtitle_thickness, cv2.LINE_AA)

    image_path = output_path.with_name(output_path.stem + "_source.png")
    if not cv2.imwrite(str(image_path), canvas):
        raise RuntimeError(f"Failed to write title card image: {image_path}")
    normalize_clip(image_path, output_path, duration)


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

    cards_by_scene = load_title_cards(plan_data, ordered_items)
    subtitles_by_scene = load_subtitle_items(plan_data)
    clips: list[dict[str, object]] = []
    clips_dir = work_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    clip_index = 1
    for item in ordered_items:
        scene_cards = cards_by_scene.get(item.get("scene_id"), [])
        standalone_cards = [card for card in scene_cards if should_render_title_card_as_clip(card)]
        for card_index, card in enumerate(standalone_cards, start=1):
            clip_path = clips_dir / f"{clip_index:03d}_title_{item.get('scene_id')}_{card_index:02d}.mp4"
            render_title_card(card, clip_path)
            card_transition = item.get("transition_hint") if card_index == 1 else "cut"
            clips.append(
                {
                    "scene_id": item.get("scene_id"),
                    "clip_kind": "title_card",
                    "clip_path": str(clip_path),
                    "source_path": None,
                    "duration_seconds": float(card.get("duration_seconds") or 1.2),
                    "chapter_id": item.get("chapter_id"),
                    "edit_action": item.get("edit_action"),
                    "transition_hint": card_transition,
                    "preview_index": 0,
                    "preview_kind": "title_card",
                    "preview_final_timestamp": item.get("start_at"),
                    "title_card_kind": card.get("kind"),
                    "title": card.get("title"),
                    "subtitle": card.get("subtitle"),
                }
            )
            clip_index += 1

        resolved_sources = resolve_preview_sources(item, root)
        if not resolved_sources:
            continue

        duration = source_duration(item, len(resolved_sources))
        scene_overlays = build_scene_overlays(scene_cards, subtitles_by_scene.get(item.get("scene_id"), []))
        for preview_index, source in enumerate(resolved_sources, start=1):
            clip_path = clips_dir / f"{clip_index:03d}_scene_{item.get('scene_id')}_{preview_index:02d}.mp4"
            clip_overlays = scene_overlays if preview_index == 1 else []
            normalize_clip(source["source_path"], clip_path, duration, overlays=clip_overlays)
            transition_hint = item.get("transition_hint") if preview_index == 1 else "cut"
            if standalone_cards and preview_index == 1:
                transition_hint = "cut"
            clips.append(
                {
                    "scene_id": item.get("scene_id"),
                    "clip_kind": "media",
                    "clip_path": str(clip_path),
                    "source_path": str(source["source_path"]),
                    "duration_seconds": duration,
                    "chapter_id": item.get("chapter_id"),
                    "edit_action": item.get("edit_action"),
                    "transition_hint": transition_hint,
                    "preview_index": preview_index,
                    "preview_kind": source.get("kind"),
                    "preview_final_timestamp": source.get("final_timestamp"),
                    "overlays": clip_overlays,
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
