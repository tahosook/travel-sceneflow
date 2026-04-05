from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import subprocess
import tempfile
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd

from sceneflow.workflow_utils import ProgressReporter, log, resolve_output_path, summarize_elapsed, write_manifest


DAY_START_HOUR = 6
DAY_END_HOUR = 18

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tif", ".tiff"}
SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules"}

TAG_ORDER = ["人物", "集合写真", "駅", "寺社", "食事", "建物", "風景", "夜景", "移動"]

# Weighted OCR keywords. Higher weight means stronger evidence.
OCR_RULES: list[tuple[str, list[tuple[str, int]]]] = [
    ("集合写真", [("集合写真", 4), ("group photo", 4), ("group shot", 4), ("groupshot", 4), ("集合", 2), ("全員", 2), ("みんな", 2), ("family", 1)]),
    ("駅", [("駅", 4), ("駅名", 4), ("station", 3), ("platform", 3), ("改札", 3), ("乗換", 3), ("乗り換え", 3), ("電車", 3), ("train", 2), ("jr", 2)]),
    ("寺社", [("神社", 4), ("寺社", 4), ("寺", 3), ("神宮", 3), ("temple", 3), ("shrine", 3), ("torii", 2), ("御朱印", 3)]),
    ("食事", [("食事", 4), ("ランチ", 4), ("ディナー", 4), ("restaurant", 3), ("cafe", 3), ("coffee", 2), ("ramen", 3), ("sushi", 3), ("udon", 3), ("そば", 3), ("ラーメン", 4), ("寿司", 4), ("ごはん", 2)]),
    ("建物", [("建物", 4), ("building", 3), ("hotel", 3), ("museum", 3), ("tower", 3), ("ビル", 3), ("館", 2), ("入口", 2), ("castle", 3)]),
    ("夜景", [("夜景", 4), ("nightscape", 4), ("nightview", 4), ("illumination", 3), ("night", 2), ("light", 1)]),
    ("移動", [("移動", 4), ("出発", 4), ("到着", 4), ("徒歩", 3), ("boarding", 3), ("bus", 3), ("car", 3), ("drive", 3), ("walk", 2), ("walking", 2), ("train", 2)]),
    ("人物", [("人物", 4), ("person", 3), ("people", 3), ("face", 2), ("selfie", 3), ("portrait", 3)]),
    ("風景", [("風景", 4), ("景色", 4), ("scenery", 3), ("landscape", 3), ("view", 2), ("sea", 2), ("ocean", 2), ("sky", 2), ("mountain", 2), ("bridge", 2), ("海", 2), ("山", 2), ("空", 1)]),
]

TAG_CAPTIONS = {
    "人物": "人物が写る一枚です。",
    "集合写真": "集合写真の一枚です。",
    "駅": "駅の様子が伝わる一枚です。",
    "寺社": "寺社の雰囲気が伝わる一枚です。",
    "食事": "食事の記録が残る一枚です。",
    "建物": "建物の様子がわかる一枚です。",
    "風景": "風景の広がりが伝わる一枚です。",
    "夜景": "夜の景色が印象的な一枚です。",
    "移動": "移動中の様子を捉えた一枚です。",
}

OCR_TIMEOUT_SECONDS = 10
FRAME_EXTRACTION_TIMEOUT_SECONDS = 10
OCR_TARGET_WIDTH = 1280
OCR_CENTER_CROP_RATIO = 0.5
OCR_POOL_WORKERS = 4
OCR_CACHE_VERSION = "v3-resize1280-crop50-pool4"
FOOD_MODIFIER_THRESHOLD = 0.7

PRIMARY_TYPE_TO_TAG = {
    "people": "人物",
    "group": "集合写真",
    "station": "駅",
    "temple": "寺社",
    "building": "建物",
    "landscape": "風景",
    "night": "夜景",
    "transit": "移動",
}
TAG_TO_PRIMARY_TYPE = {tag: primary_type for primary_type, tag in PRIMARY_TYPE_TO_TAG.items()}
PRIMARY_OCR_RULES = [(tag, keywords) for tag, keywords in OCR_RULES if tag != "食事"]
FOOD_OCR_KEYWORDS = next((keywords for tag, keywords in OCR_RULES if tag == "食事"), [])


def is_missing(value: object) -> bool:
    if value is None:
        return True
    if pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "none"


def day_or_night(timestamp_value: object) -> str:
    if is_missing(timestamp_value):
        return "day"
    try:
        ts = pd.Timestamp(timestamp_value)
    except Exception:
        return "day"
    return "day" if DAY_START_HOUR <= ts.hour < DAY_END_HOUR else "night"


def normalize_for_match(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"[^\w\u3040-\u30ff\u3400-\u9fff]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def normalize_ocr_output(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"[^\w\u3040-\u30ff\u3400-\u9fff]+", " ", text)
    tokens = re.sub(r"\s+", " ", text).strip().split()

    cleaned: list[str] = []
    prev = ""
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if len(token) == 1 and re.fullmatch(r"[A-Za-z0-9]", token):
            continue
        if token == prev:
            continue
        cleaned.append(token)
        prev = token

    return " ".join(cleaned).strip()


def normalize_stem(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", unicodedata.normalize("NFKC", text).lower())


def make_temp_png_path(prefix: str) -> Path:
    fd, tmp_name = tempfile.mkstemp(suffix=".png", prefix=prefix)
    os.close(fd)
    path = Path(tmp_name)
    path.unlink(missing_ok=True)
    return path


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def cache_key_for_path(path: Path, variant: str) -> str:
    try:
        stat = path.stat()
        fingerprint = f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"
    except Exception:
        fingerprint = f"{path.resolve()}|missing"
    return sha256_text(f"{OCR_CACHE_VERSION}|{variant}|{fingerprint}")


def read_ocr_cache(cache_dir: Path, cache_key: str) -> dict[str, object] | None:
    cache_path = cache_dir / f"{cache_key}.json"
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_ocr_cache(cache_dir: Path, cache_key: str, payload: dict[str, object]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.json"
    tmp_path = cache_dir / f".{cache_key}.{os.getpid()}.tmp"
    try:
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, cache_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def run_tesseract(image_path: Path) -> str:
    try:
        result = subprocess.run(
            ["tesseract", str(image_path), "stdout", "--oem", "1", "--psm", "6", "-l", "jpn+eng"],
            capture_output=True,
            check=False,
            timeout=OCR_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.decode("utf-8", errors="ignore").strip()


def resize_image_for_ocr(image_path: Path, target_width: int = OCR_TARGET_WIDTH) -> tuple[Path | None, list[Path]]:
    image = cv2.imread(str(image_path))
    if image is None:
        return None, []

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return None, []

    if width == target_width:
        return image_path, []

    scale = target_width / float(width)
    target_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (target_width, target_height), interpolation=interpolation)

    resized_path = make_temp_png_path("ocr-resize-")
    ok = cv2.imwrite(str(resized_path), resized)
    if not ok:
        resized_path.unlink(missing_ok=True)
        return None, []
    return resized_path, [resized_path]


def center_crop_for_ocr(image_path: Path, crop_ratio: float = OCR_CENTER_CROP_RATIO) -> tuple[Path | None, list[Path]]:
    image = cv2.imread(str(image_path))
    if image is None:
        return None, []

    height, width = image.shape[:2]
    if height <= 2 or width <= 2:
        return None, []

    crop_w = max(1, int(round(width * crop_ratio)))
    crop_h = max(1, int(round(height * crop_ratio)))
    x0 = max(0, (width - crop_w) // 2)
    y0 = max(0, (height - crop_h) // 2)
    crop = image[y0 : y0 + crop_h, x0 : x0 + crop_w]
    if crop.size == 0:
        return None, []

    crop_path = make_temp_png_path("ocr-crop-")
    ok = cv2.imwrite(str(crop_path), crop)
    if not ok:
        crop_path.unlink(missing_ok=True)
        return None, []
    return crop_path, [crop_path]


def ocr_image_with_cache(
    source_path: Path,
    variant_name: str,
    cache_dir: Path,
    analysis_path: Path,
) -> tuple[str, int]:
    cache_key = cache_key_for_path(source_path, variant_name)
    cached = read_ocr_cache(cache_dir, cache_key)
    if cached is not None:
        return str(cached.get("text", "") or ""), int(cached.get("score", 0) or 0)

    text = run_tesseract(analysis_path)
    score, _ = score_ocr_text(normalize_ocr_output(text))
    write_ocr_cache(
        cache_dir,
        cache_key,
        {
            "text": text,
            "score": score,
            "source_path": str(source_path),
            "variant": variant_name,
        },
    )
    return text, score

def extract_video_frame(video_path: Path) -> tuple[Path | None, list[Path]]:
    frame_path = make_temp_png_path("frame-")
    seek_seconds = 0.5
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_entries",
                "format=duration",
                str(video_path),
            ],
            capture_output=True,
            check=False,
            timeout=FRAME_EXTRACTION_TIMEOUT_SECONDS,
        )
        if result.returncode == 0 and result.stdout:
            data = json.loads(result.stdout)
            fmt = data.get("format", {}) if isinstance(data, dict) else {}
            raw_duration = fmt.get("duration") if isinstance(fmt, dict) else None
            if raw_duration not in (None, ""):
                duration = float(raw_duration)
                if duration > 0:
                    seek_seconds = max(0.5, duration / 2.0)
    except (subprocess.TimeoutExpired, ValueError, TypeError, json.JSONDecodeError):
        seek_seconds = 0.5

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-ss",
                f"{seek_seconds:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                str(frame_path),
            ],
            capture_output=True,
            check=False,
            timeout=FRAME_EXTRACTION_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        frame_path.unlink(missing_ok=True)
        return None, []
    if result.returncode == 0 and frame_path.exists():
        return frame_path, [frame_path]
    frame_path.unlink(missing_ok=True)
    return None, []


def prepare_image_for_analysis(image_path: Path) -> tuple[Path | None, list[Path]]:
    if image_path.suffix.lower() in IMAGE_EXTENSIONS:
        image = cv2.imread(str(image_path))
        if image is not None:
            return image_path, []

    raster_path = make_temp_png_path("analysis-")
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(image_path),
                "-frames:v",
                "1",
                str(raster_path),
            ],
            capture_output=True,
            check=False,
            timeout=FRAME_EXTRACTION_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raster_path.unlink(missing_ok=True)
        return None, []
    if result.returncode == 0 and raster_path.exists():
        return raster_path, [raster_path]
    raster_path.unlink(missing_ok=True)
    return None, []


def build_face_cascade() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade: {cascade_path}")
    return cascade


FACE_CASCADE = build_face_cascade()


def detect_face_counts(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        return 0, 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    raw_faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(24, 24),
    )
    filtered_faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(48, 48),
    )

    filtered = [face for face in filtered_faces if face[2] >= 48 and face[3] >= 48 and face[2] * face[3] >= 48 * 48]
    return int(len(raw_faces)), int(len(filtered))


def detect_food_score(image_path: Path) -> int:
    image = cv2.imread(str(image_path))
    if image is None:
        return 0

    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 1.5)

    min_radius = max(12, min(height, width) // 14)
    max_radius = max(min_radius + 1, min(height, width) // 3)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, min(height, width) // 6),
        param1=120,
        param2=28,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    circle_count = 0 if circles is None else int(circles.shape[1])
    aspect_ratio = width / float(height)
    wide_bonus = 1 if aspect_ratio >= 1.15 else 0

    score = 0
    if wide_bonus:
        score += 1
    if circle_count >= 1:
        score += 1
    if circle_count >= 2:
        score += 1
    if wide_bonus and circle_count >= 1:
        score += 1
    return score


def build_image_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}

    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if not name.startswith(".") and name not in SKIP_DIRS]
        current_dir = Path(current_root)
        if current_dir.name in SKIP_DIRS:
            continue

        for filename in filenames:
            path = current_dir / filename
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            key = normalize_stem(path.stem)
            if not key:
                continue
            index.setdefault(key, []).append(path)

    for paths in index.values():
        paths.sort(key=lambda p: str(p))
    return index


def find_counterpart_image(path_value: object, kind_value: object, image_index: dict[str, list[Path]]) -> Path | None:
    if is_missing(path_value):
        return None

    path = Path(str(path_value))
    kind = str(kind_value).strip().lower()
    if kind != "video":
        return path

    key = normalize_stem(path.stem)
    candidates = image_index.get(key, [])
    if not candidates:
        return None

    same_parent = [candidate for candidate in candidates if candidate.parent == path.parent]
    if same_parent:
        return same_parent[0]
    return candidates[0]


POOL_IMAGE_INDEX: dict[str, list[Path]] | None = None
POOL_OCR_CACHE_DIR: Path | None = None


def init_ocr_worker(image_index: dict[str, list[Path]], cache_dir: Path) -> None:
    global POOL_IMAGE_INDEX, POOL_OCR_CACHE_DIR
    POOL_IMAGE_INDEX = image_index
    POOL_OCR_CACHE_DIR = cache_dir


def get_ocr_worker_context() -> tuple[dict[str, list[Path]], Path]:
    if POOL_IMAGE_INDEX is None or POOL_OCR_CACHE_DIR is None:
        raise RuntimeError("OCR worker context has not been initialized")
    return POOL_IMAGE_INDEX, POOL_OCR_CACHE_DIR


def analyze_image_for_ocr(
    source_path: Path,
    cache_dir: Path,
    analysis_input_path: Path | None = None,
) -> tuple[str, Path | None, list[Path], int]:
    input_path = analysis_input_path or source_path
    resized_path, cleanup_paths = resize_image_for_ocr(input_path)
    if resized_path is None:
        return "", None, cleanup_paths, 0

    full_text, full_score = ocr_image_with_cache(source_path, "full", cache_dir, resized_path)
    if full_score >= 4:
        return full_text, resized_path, cleanup_paths, full_score

    crop_path, crop_cleanup = center_crop_for_ocr(resized_path)
    cleanup_paths.extend(crop_cleanup)
    if crop_path is None:
        return full_text, resized_path, cleanup_paths, full_score

    crop_text, crop_score = ocr_image_with_cache(source_path, "crop", cache_dir, crop_path)
    if crop_score > full_score:
        return crop_text, crop_path, cleanup_paths, crop_score
    return full_text, resized_path, cleanup_paths, full_score


def ocr_text_for_asset(
    path_value: object,
    kind_value: object,
    image_index: dict[str, list[Path]],
    cache_dir: Path | None = None,
) -> tuple[str, Path | None, list[Path], int]:
    if is_missing(path_value):
        return "", None, [], 0

    path = Path(str(path_value))
    kind = str(kind_value).strip().lower()
    cache_dir = Path(cache_dir) if cache_dir is not None else Path(".ocr_cache")

    if kind == "video":
        counterpart = find_counterpart_image(path_value, kind_value, image_index)
        if counterpart is not None:
            text, analysis_path, cleanup_paths, score = analyze_image_for_ocr(counterpart, cache_dir)
            if analysis_path is not None:
                return text, analysis_path, cleanup_paths, score

        frame_path, cleanup_paths = extract_video_frame(path)
        if frame_path is None:
            return "", None, [], 0
        text, analysis_path, analysis_cleanup_paths, score = analyze_image_for_ocr(path, cache_dir, analysis_input_path=frame_path)
        analysis_cleanup_paths.extend(cleanup_paths)
        return text, analysis_path, analysis_cleanup_paths, score

    return analyze_image_for_ocr(path, cache_dir)


def score_tag_candidates(text: str, rules: list[tuple[str, list[tuple[str, int]]]]) -> list[tuple[str, int, str]]:
    normalized = normalize_for_match(text)
    if not normalized:
        return []

    scores: list[tuple[str, int, str]] = []
    for tag, keywords in rules:
        score = 0
        matched = ""
        for keyword, weight in keywords:
            norm_keyword = normalize_for_match(keyword)
            if norm_keyword and norm_keyword in normalized:
                score += weight
                if not matched:
                    matched = keyword
        if score > 0:
            scores.append((tag, score, matched))

    scores.sort(key=lambda item: (-item[1], TAG_ORDER.index(item[0]) if item[0] in TAG_ORDER else len(TAG_ORDER)))
    return scores


def score_ocr_text(text: str) -> tuple[int, str | None]:
    scores = score_tag_candidates(text, OCR_RULES)
    if not scores:
        return 0, None
    return scores[0][1], scores[0][2] or None


def infer_primary_tag_from_ocr(ocr_text: str) -> tuple[str | None, bool, str | None]:
    text = normalize_ocr_output(ocr_text)
    candidates = score_tag_candidates(text, PRIMARY_OCR_RULES)
    if not candidates:
        return None, False, None

    tag, tag_score, tag_match = candidates[0]
    return tag, tag_score >= 4, tag_match


def infer_tag_from_face(face_count_filtered: int) -> str | None:
    if face_count_filtered >= 2:
        return "集合写真"
    if face_count_filtered == 1:
        return "人物"
    return None


def infer_tag_from_time(timestamp_value: object) -> str:
    return "夜景" if day_or_night(timestamp_value) == "night" else "風景"


def normalize_primary_type(tag: str | None) -> str:
    if tag is None:
        return "landscape"
    return TAG_TO_PRIMARY_TYPE.get(tag, "landscape")


def food_ocr_score(ocr_text: str) -> int:
    normalized = normalize_for_match(normalize_ocr_output(ocr_text))
    if not normalized:
        return 0

    score = 0
    for keyword, weight in FOOD_OCR_KEYWORDS:
        norm_keyword = normalize_for_match(keyword)
        if norm_keyword and norm_keyword in normalized:
            score += weight
    return score


def compute_food_confidence(food_score: int, ocr_text: str) -> float:
    # Keep food as a conservative modifier: circle-heavy images alone should
    # not dominate classification without at least some corroborating signal.
    geometric_signal = min(max(food_score, 0), 4) / 4.0
    ocr_signal = min(food_ocr_score(ocr_text), 6) / 6.0
    confidence = 0.65 * geometric_signal + 0.35 * ocr_signal
    return round(max(0.0, min(confidence, 1.0)), 3)


def analyze_food_for_asset(
    path_value: object,
    kind_value: object,
    image_index: dict[str, list[Path]],
    cache_dir: Path | None = None,
) -> dict[str, object]:
    raw_ocr_text, analysis_path, cleanup_paths, _ = ocr_text_for_asset(
        path_value,
        kind_value,
        image_index,
        cache_dir=cache_dir,
    )
    food_score = detect_food_score(analysis_path) if analysis_path is not None else 0
    food_confidence = compute_food_confidence(food_score, raw_ocr_text)
    cleaned_ocr_text = normalize_ocr_output(raw_ocr_text)
    if len(cleaned_ocr_text) > 240:
        cleaned_ocr_text = cleaned_ocr_text[:240]

    for cleanup_path in cleanup_paths:
        try:
            cleanup_path.unlink(missing_ok=True)
        except Exception:
            pass

    return {
        "ocr_text": cleaned_ocr_text,
        "food_score": food_score,
        "food_confidence": food_confidence,
    }


def build_modifiers(food_confidence: float) -> list[str]:
    modifiers: list[str] = []
    if food_confidence >= FOOD_MODIFIER_THRESHOLD:
        modifiers.append("food")
    return modifiers


def classify_scene_attributes(
    row: pd.Series,
    ocr_text: str,
    face_count_filtered: int,
    food_confidence: float,
) -> tuple[str, str, list[str], str, str | None]:
    modifiers = build_modifiers(food_confidence)
    ocr_tag, ocr_strong, ocr_match = infer_primary_tag_from_ocr(ocr_text)
    if ocr_tag and ocr_strong:
        return ocr_tag, normalize_primary_type(ocr_tag), modifiers, "ocr", ocr_match

    face_tag = infer_tag_from_face(face_count_filtered)
    if face_tag:
        return face_tag, normalize_primary_type(face_tag), modifiers, "face", None

    time_tag = infer_tag_from_time(row["representative_final_timestamp"])
    return time_tag, normalize_primary_type(time_tag), modifiers, "time", None


def build_caption(
    tag: str,
    source: str,
    ocr_match: str | None,
    face_count_filtered: int,
    timestamp_value: object,
    modifiers: list[str] | None = None,
) -> str:
    time_label = "昼" if day_or_night(timestamp_value) == "day" else "夜"
    modifiers = modifiers or []
    has_food = "food" in modifiers

    if source == "ocr":
        if tag in {"駅", "寺社", "建物", "移動"}:
            if ocr_match:
                return f"{ocr_match}が見える一枚です。"
            return TAG_CAPTIONS.get(tag, f"{time_label}の様子が伝わる一枚です。")
        if tag == "夜景":
            return "夜の景色が印象的な一枚です。"
        if tag == "風景":
            return "風景の広がりが伝わる一枚です。"
        if tag == "集合写真":
            return "集合写真の一枚です。"
        if tag == "人物":
            return "人物が写る一枚です。"

    if source == "face":
        if tag == "集合写真":
            return f"{face_count_filtered}人ほど写る集合写真です。"
        return "人物が写る一枚です。"

    if has_food and tag == "夜景":
        return "夜の食事を含む旅の一枚です。"
    if has_food and tag == "風景":
        return "食事を含む旅の雰囲気が伝わる一枚です。"

    if tag == "夜景":
        return f"{time_label}の景色が印象的な一枚です。"
    if tag == "風景":
        return f"{time_label}の雰囲気が伝わる一枚です。"
    if tag == "移動":
        return f"{time_label}の移動の様子を捉えた一枚です。"
    if tag == "駅":
        return "駅の様子が伝わる一枚です。"
    if tag == "寺社":
        return "寺社の雰囲気が伝わる一枚です。"
    if tag == "建物":
        return "建物の様子がわかる一枚です。"
    if tag == "人物":
        return "人物が写る一枚です。"
    if tag == "集合写真":
        return f"{face_count_filtered}人ほど写る集合写真です。" if face_count_filtered else "集合写真の一枚です。"

    return TAG_CAPTIONS.get(tag, f"{time_label}の様子が伝わる一枚です。")


def process_representative_row(item: tuple[int, dict[str, object]]) -> dict[str, object]:
    row_index, row_data = item
    image_index, cache_dir = get_ocr_worker_context()
    raw_ocr_text, analysis_path, cleanup_paths, _ = ocr_text_for_asset(
        row_data["representative_path"],
        row_data["representative_kind"],
        image_index,
        cache_dir=cache_dir,
    )
    face_count_raw, face_count_filtered = detect_face_counts(analysis_path) if analysis_path is not None else (0, 0)
    food_score = detect_food_score(analysis_path) if analysis_path is not None else 0
    food_confidence = compute_food_confidence(food_score, raw_ocr_text)
    row = pd.Series(row_data)
    tag, primary_type, modifiers, source, ocr_match = classify_scene_attributes(row, raw_ocr_text, face_count_filtered, food_confidence)
    caption = build_caption(tag, source, ocr_match, face_count_filtered, row_data["representative_final_timestamp"], modifiers)

    cleaned_ocr_text = normalize_ocr_output(raw_ocr_text)
    if len(cleaned_ocr_text) > 240:
        cleaned_ocr_text = cleaned_ocr_text[:240]

    for cleanup_path in cleanup_paths:
        try:
            cleanup_path.unlink(missing_ok=True)
        except Exception:
            pass

    return {
        "scene_id": row_data["scene_id"],
        "asset_count": row_data["asset_count"],
        "representative_path": row_data["representative_path"],
        "representative_kind": row_data["representative_kind"],
        "representative_final_timestamp": row_data["representative_final_timestamp"],
        "representative_laplacian": row_data.get("representative_laplacian"),
        "representative_brightness": row_data.get("representative_brightness"),
        "representative_duration_seconds": row_data.get("representative_duration_seconds"),
        "representative_has_audio": row_data.get("representative_has_audio"),
        "ocr_text": cleaned_ocr_text,
        "face_count_raw": face_count_raw,
        "face_count_filtered": face_count_filtered,
        "face_count": face_count_filtered,
        "tag": tag,
        "primary_type": primary_type,
        "modifiers": modifiers,
        "food_confidence": food_confidence,
        "caption": caption,
        "_row_index": row_index,
    }


def annotate_representatives(
    input_data: Path | pd.DataFrame,
    root: Path,
    reporter: ProgressReporter | None = None,
    *,
    workers: int = OCR_POOL_WORKERS,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    if isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        df = pd.read_csv(input_data)
    if "representative_path" not in df.columns:
        raise ValueError("representative_path column is required")

    image_index = build_image_index(root)
    cache_dir = Path(cache_dir) if cache_dir is not None else Path("outputs") / "ocr_cache"

    rows = [row.to_dict() for _, row in df.iterrows()]
    df = df.copy()

    if workers <= 1 or len(rows) <= 1:
        results = []
        init_ocr_worker(image_index, cache_dir)
        for index, row_data in enumerate(rows, start=1):
            if reporter is not None:
                reporter.update(index, Path(str(row_data["representative_path"])).name if not is_missing(row_data["representative_path"]) else None)
            results.append(process_representative_row((index - 1, row_data)))
    else:
        ctx = mp.get_context("spawn")
        results = [None] * len(rows)
        with ctx.Pool(
            processes=workers,
            initializer=init_ocr_worker,
            initargs=(image_index, cache_dir),
        ) as pool:
            for done, result in enumerate(pool.imap_unordered(process_representative_row, enumerate(rows), chunksize=1), start=1):
                results[int(result["_row_index"])] = result
                if reporter is not None:
                    reporter.update(done, Path(str(result["representative_path"])).name if not is_missing(result["representative_path"]) else None)

    results = sorted(results, key=lambda item: (item["scene_id"] if item is not None else 0))

    ocr_texts = [row["ocr_text"] for row in results]
    face_counts_raw = [int(row["face_count_raw"]) for row in results]
    face_counts_filtered = [int(row["face_count_filtered"]) for row in results]
    face_counts_alias = [int(row["face_count"]) for row in results]
    tags = [row["tag"] for row in results]
    primary_types = [row["primary_type"] for row in results]
    modifiers = [row["modifiers"] for row in results]
    food_confidences = [row["food_confidence"] for row in results]
    captions = [row["caption"] for row in results]

    if reporter is not None:
        reporter.finish("tagging complete")

    df["ocr_text"] = ocr_texts
    df["face_count_raw"] = face_counts_raw
    df["face_count_filtered"] = face_counts_filtered
    df["face_count"] = face_counts_alias
    df["tag"] = tags
    df["primary_type"] = primary_types
    df["modifiers"] = modifiers
    df["food_confidence"] = food_confidences
    df["caption"] = captions
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Add OCR and face-based tag and caption to scene representative assets.")
    parser.add_argument("--input", default="scene_representatives.csv", help="Input representative CSV path")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--output-dir", default=None, help="Directory for outputs. Defaults to outputs/<run>/tagging/")
    parser.add_argument("--run-dir", default=None, help="Shared run directory for the whole workflow")
    parser.add_argument("--root", default=".", help="Media root directory used to find matching images for videos")
    parser.add_argument("--progress-interval", type=int, default=1, help="How often to print progress updates")
    parser.add_argument("--workers", type=int, default=OCR_POOL_WORKERS, help="Number of OCR worker processes to use")
    parser.add_argument("--ocr-cache-dir", default=None, help="Directory for OCR result cache. Defaults to <output-dir>/ocr_cache")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = resolve_output_path(
        default_filename="scene_representatives_tagged.csv",
        step_name="tagging",
        output=args.output,
        output_dir=args.output_dir,
        run_dir=args.run_dir,
        input_path=input_path,
    )
    root = Path(args.root)

    log(f"[tagging] input={input_path}")
    log(f"[tagging] output={output_path}")
    log(f"[tagging] root={root.resolve()}")
    start = time.monotonic()
    input_df = pd.read_csv(input_path)
    total = int(input_df.shape[0])
    ocr_cache_dir = Path(args.ocr_cache_dir) if args.ocr_cache_dir is not None else output_path.parent / "ocr_cache"
    df = annotate_representatives(
        input_df,
        root,
        reporter=ProgressReporter("tagging", total=total, interval=args.progress_interval),
        workers=args.workers,
        cache_dir=ocr_cache_dir,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, na_rep="None")
    write_manifest(
        output_path,
        {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "step": "tagging",
            "input": str(input_path),
            "output": str(output_path),
            "root": str(root.resolve()),
            "row_count": int(len(df)),
            "elapsed_seconds": round(time.monotonic() - start, 3),
        },
    )
    log(f"[tagging] wrote {output_path} ({len(df)} rows, elapsed={summarize_elapsed(start)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
