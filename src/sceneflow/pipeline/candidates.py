from __future__ import annotations

import argparse
import ast
import json
import math
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from sceneflow.pipeline import tagging as tagging_pipeline
from sceneflow.workflow_utils import log, resolve_output_path, summarize_elapsed, write_manifest


DEFAULT_MEDIA_SCENE = "media_scene.csv"
DEFAULT_REPRESENTATIVES = "scene_representatives_tagged.csv"
DEFAULT_OUTPUT = "scene_edit_candidates.json"
TZ_NAME = "Asia/Tokyo"
PREVIEW_SOURCE_LIMIT = 3
SELECTION_TOP_K_RATIO = 0.3
GENERAL_MAX_FOOD_RATIO = 0.25
GOURMET_MAX_FOOD_RATIO = 0.4
GENERAL_CONSECUTIVE_FOOD_PENALTY = 0.12
GOURMET_CONSECUTIVE_FOOD_PENALTY = 0.06
FOOD_SCENE_SAMPLE_LIMIT = 5
FOOD_SCENE_MIN_SAMPLE_LIMIT = 3
FOOD_SCENE_THRESHOLD = 0.6
FOOD_SCENE_SINGLE_SAMPLE_THRESHOLD = 0.7

TAG_STRENGTH = {
    "人物": "strong",
    "集合写真": "strong",
    "駅": "strong",
    "寺社": "strong",
    "建物": "medium",
    "移動": "medium",
    "風景": "weak",
    "夜景": "weak",
}

OCR_DICTIONARY = [
    "集合写真",
    "駅",
    "駅名",
    "改札",
    "乗換",
    "乗り換え",
    "電車",
    "神社",
    "寺社",
    "寺",
    "神宮",
    "建物",
    "ビル",
    "museum",
    "tower",
    "移動",
    "出発",
    "到着",
    "徒歩",
    "風景",
    "景色",
    "夜景",
]

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


def is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, set, dict)):
        return False
    if pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "none"


def parse_timestamp(value: object) -> pd.Timestamp | None:
    if is_missing(value):
        return None
    try:
        ts = pd.Timestamp(str(value).strip())
    except Exception:
        return None
    if ts.tzinfo is None:
        return ts.tz_localize(TZ_NAME)
    return ts.tz_convert(TZ_NAME)


def format_timestamp(value: pd.Timestamp | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return value.isoformat()


def parse_float(value: object) -> float | None:
    if is_missing(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def relpath(value: object, root: Path) -> str | None:
    if is_missing(value):
        return None
    path = Path(str(value))
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(value, upper))


def unique_strings(values: list[object]) -> list[str]:
    items: list[str] = []
    for value in values:
        if is_missing(value):
            continue
        text = str(value).strip()
        if not text or text in items:
            continue
        items.append(text)
    return items


def parse_string_list(value: object) -> list[str]:
    if is_missing(value):
        return []
    if isinstance(value, list):
        return unique_strings(value)
    if isinstance(value, (tuple, set)):
        return unique_strings(list(value))

    text = str(value).strip()
    if not text:
        return []

    parsed: object = None
    if text.startswith("[") or text.startswith("("):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                break
            except Exception:
                continue
    if isinstance(parsed, (list, tuple, set)):
        return unique_strings(list(parsed))
    if "," in text:
        return unique_strings([part.strip() for part in text.split(",")])
    return unique_strings([text])


def normalized_asset_key(path_value: object) -> str:
    if is_missing(path_value):
        return ""
    path = Path(str(path_value))
    return tagging_pipeline.normalize_stem(path.stem)


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return pd.read_csv(path)


def load_representatives(path_candidates: list[Path]) -> pd.DataFrame:
    for path in path_candidates:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError("Missing representative CSV. Expected one of: " + ", ".join(str(p) for p in path_candidates))


def pick_representative_row(group: pd.DataFrame, reps: pd.DataFrame) -> pd.Series | None:
    match = reps[reps["scene_id"] == group.iloc[0]["scene_id"]]
    if match.empty:
        return None
    return match.iloc[0]


def unique_nonmissing(values: pd.Series) -> list[str]:
    items: list[str] = []
    for value in values.tolist():
        if is_missing(value):
            continue
        text = str(value)
        if text not in items:
            items.append(text)
    return items


def normalize_ocr_tokens(text: object) -> list[str]:
    if is_missing(text):
        return []
    normalized = normalize_for_tokens(str(text))
    if not normalized:
        return []

    tokens: list[str] = []
    seen: set[str] = set()
    for token in normalized.split():
        token = token.strip()
        if not token:
            continue
        if len(token) < 2:
            continue
        if token.isascii() and token.replace("-", "").replace("_", "").isalpha() and len(token) <= 3:
            continue
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens[:12]


def normalize_for_tokens(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = "".join(ch if (ch.isalnum() or "\u3040" <= ch <= "\u30ff" or "\u3400" <= ch <= "\u9fff" or ch.isspace()) else " " for ch in text)
    text = " ".join(text.split())
    return text.lower()


def meaningful_ocr_tokens(text: object) -> list[str]:
    tokens = normalize_ocr_tokens(text)
    if not tokens:
        return []

    meaningful: list[str] = []
    seen: set[str] = set()
    dictionary = [normalize_for_tokens(item) for item in OCR_DICTIONARY]

    for token in tokens:
        norm_token = normalize_for_tokens(token)
        if not norm_token:
            continue
        if len(norm_token) < 2:
            continue
        for keyword in dictionary:
            if keyword and (keyword in norm_token or norm_token in keyword):
                if norm_token not in seen:
                    seen.add(norm_token)
                    meaningful.append(token)
                break

    return meaningful[:8]


def compute_gps_summary(group: pd.DataFrame) -> dict[str, object]:
    gps = group[["gps_latitude", "gps_longitude"]].copy()
    gps["gps_latitude"] = pd.to_numeric(gps["gps_latitude"], errors="coerce")
    gps["gps_longitude"] = pd.to_numeric(gps["gps_longitude"], errors="coerce")
    gps = gps.dropna(subset=["gps_latitude", "gps_longitude"])

    if gps.empty:
        return {
            "has_gps": False,
            "count": 0,
            "center_latitude": None,
            "center_longitude": None,
            "radius_meters": None,
        }

    points = list(zip(gps["gps_latitude"].tolist(), gps["gps_longitude"].tolist()))
    center_lat = sum(lat for lat, _ in points) / len(points)
    center_lon = sum(lon for _, lon in points) / len(points)
    radius = max(haversine_meters(center_lat, center_lon, lat, lon) for lat, lon in points)

    return {
        "has_gps": True,
        "count": len(points),
        "center_latitude": round(center_lat, 7),
        "center_longitude": round(center_lon, 7),
        "radius_meters": round(radius, 1),
    }


def preview_source_indices(count: int, limit: int = PREVIEW_SOURCE_LIMIT) -> list[int]:
    if count <= 0:
        return []
    if count <= limit:
        return list(range(count))
    if limit <= 1:
        return [count // 2]

    last_index = count - 1
    indices: list[int] = []
    for offset in range(limit):
        index = int(round(offset * last_index / (limit - 1)))
        if index not in indices:
            indices.append(index)
    return indices


def scene_sample_indices(count: int) -> list[int]:
    if count <= 0:
        return []
    if count == 1:
        return [0]
    target = min(FOOD_SCENE_SAMPLE_LIMIT, count)
    if count >= FOOD_SCENE_MIN_SAMPLE_LIMIT:
        target = max(target, min(FOOD_SCENE_MIN_SAMPLE_LIMIT, count))

    indices = {0, count - 1}
    if count > 2:
        indices.add(count // 2)

    last_index = count - 1
    while len(indices) < target:
        for offset in range(target):
            index = int(round(offset * last_index / max(target - 1, 1)))
            indices.add(index)
            if len(indices) >= target:
                break
        else:
            break

    return sorted(indices)


def choose_scene_food_sample_rows(group: pd.DataFrame, rep: pd.Series | None) -> list[pd.Series]:
    ordered = group.sort_values("final_timestamp_dt", kind="stable").reset_index(drop=True)
    rows: list[pd.Series] = []
    seen_keys: set[str] = set()
    representative_path = None if rep is None else relpath(rep.get("representative_path"), Path("."))

    preferred_indices: list[int] = []
    if rep is not None:
        rep_path_value = rep.get("representative_path")
        for index, row in ordered.iterrows():
            if str(row.get("path")) == str(rep_path_value):
                preferred_indices.append(index)
                break
    preferred_indices.extend(scene_sample_indices(len(ordered)))

    for index in preferred_indices:
        if index < 0 or index >= len(ordered):
            continue
        row = ordered.iloc[index]
        key = normalized_asset_key(row.get("path")) or str(row.get("path"))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        rows.append(row)
        if len(rows) >= FOOD_SCENE_SAMPLE_LIMIT:
            break

    if len(rows) < min(FOOD_SCENE_MIN_SAMPLE_LIMIT, len(ordered)):
        image_rows = ordered[ordered["kind"].astype(str).str.lower().eq("image")]
        for _, row in image_rows.iterrows():
            key = normalized_asset_key(row.get("path")) or str(row.get("path"))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            rows.append(row)
            if len(rows) >= FOOD_SCENE_SAMPLE_LIMIT:
                break

    return rows[:FOOD_SCENE_SAMPLE_LIMIT]


def aggregate_food_samples(sample_confidences: list[float]) -> float:
    if not sample_confidences:
        return 0.0
    ordered = sorted((clamp(value) for value in sample_confidences), reverse=True)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    return round(clamp(0.65 * ordered[0] + 0.35 * ordered[1]), 3)


def analyze_scene_food(
    group: pd.DataFrame,
    rep: pd.Series | None,
    root: Path,
    image_index: dict[str, list[Path]],
    cache_dir: Path,
) -> dict[str, object]:
    sample_rows = choose_scene_food_sample_rows(group, rep)
    sample_confidences: list[float] = []
    sample_paths: list[str] = []
    evidence_count = 0

    for row in sample_rows:
        path_value = row.get("path")
        if is_missing(path_value):
            continue
        path = Path(str(path_value))
        if not path.exists():
            continue
        sample_paths.append(relpath(path, root) or str(path))
        try:
            analysis = tagging_pipeline.analyze_food_for_asset(
                path,
                row.get("kind"),
                image_index,
                cache_dir=cache_dir,
            )
        except Exception:
            continue
        confidence = float(analysis.get("food_confidence") or 0.0)
        sample_confidences.append(confidence)
        if confidence >= FOOD_SCENE_THRESHOLD:
            evidence_count += 1

    scene_food_confidence = aggregate_food_samples(sample_confidences)
    threshold = FOOD_SCENE_SINGLE_SAMPLE_THRESHOLD if len(sample_confidences) <= 1 else FOOD_SCENE_THRESHOLD

    return {
        "food_confidence": scene_food_confidence,
        "food_sample_count": len(sample_confidences),
        "food_evidence_count": evidence_count,
        "food_sample_paths": sample_paths,
        "has_food_modifier": scene_food_confidence >= threshold,
    }


def build_preview_sources(group: pd.DataFrame, root: Path) -> list[dict[str, object]]:
    ordered = group.sort_values("final_timestamp_dt", kind="stable").reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for index in preview_source_indices(len(ordered)):
        row = ordered.iloc[index]
        rows.append(
            {
                "path": relpath(row.get("path"), root),
                "kind": row.get("kind"),
                "final_timestamp": format_timestamp(row.get("final_timestamp_dt")),
                "duration_seconds": parse_float(row.get("duration_seconds")),
            }
        )
    return rows


def compute_max_gap_seconds(group: pd.DataFrame) -> float:
    timestamps = group["final_timestamp_dt"].dropna().sort_values(kind="stable")
    if len(timestamps) <= 1:
        return 0.0
    gaps = timestamps.diff().dt.total_seconds().dropna()
    if gaps.empty:
        return 0.0
    return float(max(gaps.max(), 0.0))


def normalize_tag(tag: object) -> str | None:
    if is_missing(tag):
        return None
    text = str(tag).strip()
    return text or None


def normalize_primary_type(primary_type: object, fallback_tag: object = None) -> str:
    normalized = normalize_tag(primary_type)
    if normalized in PRIMARY_TYPE_TO_TAG:
        return normalized

    fallback = normalize_tag(fallback_tag)
    if fallback == "食事":
        return "landscape"
    if fallback is None:
        return "landscape"
    return TAG_TO_PRIMARY_TYPE.get(fallback, "landscape")


def legacy_tag_for_primary_type(primary_type: object, fallback_tag: object = None) -> str:
    normalized = normalize_primary_type(primary_type, fallback_tag)
    return PRIMARY_TYPE_TO_TAG.get(normalized, normalize_tag(fallback_tag) or "風景")


def normalize_modifiers(value: object, *, fallback_tag: object = None, food_confidence: float | None = None) -> list[str]:
    modifiers = parse_string_list(value)
    if normalize_tag(fallback_tag) == "食事" and "food" not in modifiers:
        modifiers.append("food")
    if food_confidence is not None and food_confidence >= 0.7 and "food" not in modifiers:
        modifiers.append("food")
    return unique_strings(modifiers)


def has_modifier(record: dict[str, object], modifier: str) -> bool:
    return modifier in normalize_modifiers(record.get("modifiers"))


def tag_strength(tag: object) -> str:
    normalized = normalize_tag(tag)
    if normalized is None:
        return "weak"
    return TAG_STRENGTH.get(normalized, "weak")


def priority_band(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def format_duration_label(seconds: object) -> str:
    value = parse_float(seconds)
    if value is None:
        return "duration unknown"
    total = max(0, int(round(value)))
    if total < 60:
        return f"{total}秒"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}分{secs:02d}秒"
    hours, rem_minutes = divmod(minutes, 60)
    return f"{hours}時間{rem_minutes:02d}分"


def build_flow_summary(scene_record: dict[str, object]) -> str:
    duration_label = format_duration_label(scene_record.get("duration_seconds"))
    asset_count = scene_record.get("asset_count")
    image_count = scene_record.get("image_count")
    video_count = scene_record.get("video_count")
    gps = scene_record.get("gps") if isinstance(scene_record.get("gps"), dict) else {}
    tag = scene_record.get("representative_tag") or "未判定"
    priority = scene_record.get("priority_band") or "low"
    modifiers = normalize_modifiers(scene_record.get("modifiers"))

    kind_parts: list[str] = []
    if isinstance(asset_count, int):
        kind_parts.append(f"{asset_count}素材")
    if isinstance(image_count, int) and isinstance(video_count, int):
        if image_count and video_count:
            kind_parts.append(f"{image_count}枚/{video_count}本")
        elif image_count:
            kind_parts.append(f"{image_count}枚")
        elif video_count:
            kind_parts.append(f"{video_count}本")

    gps_label = "GPSあり" if gps.get("has_gps") else "GPSなし"
    modifier_text = f"mod:{','.join(modifiers)}" if modifiers else None
    return " / ".join([part for part in [duration_label, *kind_parts, gps_label, f"tag:{tag}", modifier_text, f"priority:{priority}"] if part])


def compute_scene_signals(
    group: pd.DataFrame,
    rep: pd.Series | None,
    gps_summary: dict[str, object],
    scene_food: dict[str, object] | None = None,
) -> tuple[dict[str, float], list[str], dict[str, object]]:
    reasons: list[str] = []
    signals = {
        "visual_quality": 0.18,
        "face_signal": 0.0,
        "motion_signal": 0.0,
        "novelty_signal": 0.1,
        "food_signal": 0.0,
    }

    asset_count = int(len(group))
    duration_seconds = group["final_timestamp_dt"].max() - group["final_timestamp_dt"].min()
    total_seconds = duration_seconds.total_seconds() if duration_seconds is not None else None
    max_gap_seconds = compute_max_gap_seconds(group)
    scene_meta: dict[str, object] = {
        "primary_type": "landscape",
        "modifiers": [],
        "food_confidence": 0.0,
        "food_confidence_representative": 0.0,
        "food_sample_count": 0,
        "food_evidence_count": 0,
        "food_sample_paths": [],
        "representative_tag": "風景",
        "tag_strength": "weak",
        "meaningful_tokens": [],
    }

    if asset_count >= 2:
        signals["visual_quality"] += min(0.12 + 0.03 * min(asset_count, 6), 0.30)
        reasons.append("continuous_assets")

    if total_seconds is not None and total_seconds >= 30:
        signals["visual_quality"] += min(total_seconds / 900.0, 0.15)
        reasons.append("scene_duration")

    if gps_summary.get("has_gps"):
        signals["novelty_signal"] += 0.18
        reasons.append("gps")

    if max_gap_seconds <= 120 and asset_count >= 2:
        signals["motion_signal"] += 0.24
        reasons.append("tight_flow")
    elif max_gap_seconds <= 300:
        signals["motion_signal"] += 0.12
        reasons.append("moderate_flow")

    if rep is not None:
        raw_tag = normalize_tag(rep.get("tag"))
        representative_food_confidence = parse_float(rep.get("food_confidence")) or 0.0
        primary_type = normalize_primary_type(rep.get("primary_type"), raw_tag)
        representative_tag = legacy_tag_for_primary_type(primary_type, raw_tag)
        strength = tag_strength(representative_tag)
        face_count = parse_float(rep.get("face_count_filtered")) or 0.0
        rep_blur = parse_float(rep.get("representative_laplacian"))
        rep_kind = str(rep.get("representative_kind")) if not is_missing(rep.get("representative_kind")) else None
        rep_duration = parse_float(rep.get("representative_duration_seconds"))
        meaningful_tokens = meaningful_ocr_tokens(rep.get("ocr_text"))

        if rep_blur is not None:
            blur_boost = min(max(rep_blur, 0.0) / 4000.0, 1.0) * 0.22
            signals["visual_quality"] += blur_boost
            if blur_boost > 0:
                reasons.append("sharp")

        if rep_kind == "video":
            signals["motion_signal"] += 0.16
            reasons.append("video")
            if rep_duration is not None and rep_duration <= 8:
                signals["visual_quality"] += 0.08
                reasons.append("short_video")
            elif rep_duration is not None and rep_duration > 10:
                signals["visual_quality"] -= 0.08
                reasons.append("long_video")

        if face_count > 0:
            signals["face_signal"] += min(face_count * 0.22, 0.65)
            reasons.append("face")

        if primary_type in {"people", "group"}:
            signals["face_signal"] += 0.22
            reasons.append(f"primary:{primary_type}")
        elif primary_type in {"station", "transit"}:
            signals["motion_signal"] += 0.18
            reasons.append(f"primary:{primary_type}")
        elif primary_type in {"temple", "building"}:
            signals["novelty_signal"] += 0.14
            reasons.append(f"primary:{primary_type}")
        elif primary_type == "night":
            signals["novelty_signal"] += 0.10
            reasons.append("primary:night")
        elif primary_type == "landscape":
            signals["visual_quality"] += 0.04
            reasons.append("primary:landscape")

        if strength == "strong":
            signals["novelty_signal"] += 0.08
            reasons.append("strength:strong")
        elif strength == "medium":
            signals["novelty_signal"] += 0.04
            reasons.append("strength:medium")
        else:
            reasons.append("strength:weak")

        if meaningful_tokens:
            signals["novelty_signal"] += min(0.10 + 0.03 * len(meaningful_tokens), 0.22)
            reasons.append("ocr")

        scene_food = scene_food or {}
        scene_food_confidence = float(scene_food.get("food_confidence") or 0.0)
        food_sample_count = int(scene_food.get("food_sample_count") or 0)
        food_evidence_count = int(scene_food.get("food_evidence_count") or 0)
        food_sample_paths = list(scene_food.get("food_sample_paths") or [])
        has_food_modifier = bool(scene_food.get("has_food_modifier"))
        modifiers = normalize_modifiers([], food_confidence=scene_food_confidence if has_food_modifier else None)

        if food_sample_count > 0:
            reasons.append("food_scene_sampled")
        if food_evidence_count > 0:
            reasons.append(f"food_scene_evidence:{food_evidence_count}")
        if has_food_modifier:
            signals["food_signal"] = max(scene_food_confidence, 0.35)
            reasons.append("food_modifier:scene")
        elif representative_food_confidence >= FOOD_SCENE_THRESHOLD:
            reasons.append("food_modifier:representative_only")

        if representative_tag == "風景" and not meaningful_tokens:
            signals["novelty_signal"] -= 0.08
            reasons.append("sparse_ocr")

        scene_meta = {
            "primary_type": primary_type,
            "modifiers": modifiers,
            "food_confidence": round(scene_food_confidence, 3),
            "food_confidence_representative": round(representative_food_confidence, 3),
            "food_sample_count": food_sample_count,
            "food_evidence_count": food_evidence_count,
            "food_sample_paths": food_sample_paths,
            "representative_tag": representative_tag,
            "tag_strength": strength,
            "meaningful_tokens": meaningful_tokens,
        }

    if total_seconds is not None:
        if total_seconds > 1200:
            signals["novelty_signal"] -= 0.24
            reasons.append("very_long")
        elif total_seconds > 600:
            signals["novelty_signal"] -= 0.12
            reasons.append("long")

    if asset_count > 15:
        signals["novelty_signal"] -= 0.10
        reasons.append("many_assets")

    if max_gap_seconds > 600:
        signals["motion_signal"] -= 0.22
        reasons.append("big_gap")

    normalized_signals = {name: round(clamp(value), 3) for name, value in signals.items()}
    scene_meta["max_gap_seconds"] = round(max_gap_seconds, 3)
    return normalized_signals, unique_strings(reasons), scene_meta


def infer_trip_type(records: list[dict[str, object]]) -> str:
    if not records:
        return "general"
    food_count = sum(1 for record in records if has_modifier(record, "food"))
    food_ratio = food_count / max(len(records), 1)
    return "gourmet" if food_ratio > 0.4 else "general"


def selection_weights(trip_type: str) -> dict[str, float]:
    if trip_type == "gourmet":
        return {
            "visual_quality": 0.38,
            "face_signal": 0.20,
            "motion_signal": 0.17,
            "novelty_signal": 0.17,
            "food_signal": 0.08,
        }
    return {
        "visual_quality": 0.40,
        "face_signal": 0.22,
        "motion_signal": 0.18,
        "novelty_signal": 0.16,
        "food_signal": 0.04,
    }


def rank_scene_indices(records: list[dict[str, object]]) -> list[int]:
    return sorted(
        range(len(records)),
        key=lambda idx: (
            -float(records[idx].get("selection_score") or 0.0),
            str(records[idx].get("start_at") or ""),
            int(records[idx].get("scene_id") or 0),
        ),
    )


def target_selected_count(total: int) -> int:
    if total <= 0:
        return 0
    return max(1, int(math.ceil(total * SELECTION_TOP_K_RATIO)))


def max_food_selection_count(selected_count: int, trip_type: str) -> int:
    if selected_count <= 0:
        return 0
    ratio = GOURMET_MAX_FOOD_RATIO if trip_type == "gourmet" else GENERAL_MAX_FOOD_RATIO
    return max(1, int(math.floor(selected_count * ratio)))


def append_selection_reason(record: dict[str, object], reason: str) -> None:
    reasons = unique_strings(list(record.get("selection_reasons") or []))
    if reason not in reasons:
        reasons.append(reason)
    record["selection_reasons"] = reasons


def enforce_food_quota(records: list[dict[str, object]], ranked_indices: list[int], trip_type: str) -> set[int]:
    selected_set = set(ranked_indices[: target_selected_count(len(records))])
    max_food_count = max_food_selection_count(len(selected_set), trip_type)
    selected_food_indices = [idx for idx in selected_set if has_modifier(records[idx], "food")]
    if len(selected_food_indices) <= max_food_count:
        return selected_set

    removable = sorted(
        selected_food_indices,
        key=lambda idx: (
            float(records[idx].get("selection_score") or 0.0),
            str(records[idx].get("start_at") or ""),
            int(records[idx].get("scene_id") or 0),
        ),
    )
    replacements = [idx for idx in ranked_indices if idx not in selected_set and not has_modifier(records[idx], "food")]

    remove_count = len(selected_food_indices) - max_food_count
    for index in removable[:remove_count]:
        selected_set.discard(index)
        append_selection_reason(records[index], "quota_replaced")
        if replacements:
            replacement = replacements.pop(0)
            selected_set.add(replacement)
            append_selection_reason(records[replacement], "quota_promoted")
    return selected_set


def apply_selection_strategy(records: list[dict[str, object]]) -> None:
    if not records:
        return

    trip_type = infer_trip_type(records)
    weights = selection_weights(trip_type)
    food_penalty = GOURMET_CONSECUTIVE_FOOD_PENALTY if trip_type == "gourmet" else GENERAL_CONSECUTIVE_FOOD_PENALTY
    previous_food = False

    for record in records:
        components = record.get("selection_components") if isinstance(record.get("selection_components"), dict) else {}
        score = 0.0
        for name, weight in weights.items():
            score += weight * float(components.get(name) or 0.0)
        record["trip_type"] = trip_type
        record["selection_reasons"] = unique_strings(list(record.get("selection_reasons") or []) + [f"trip_type:{trip_type}"])
        if previous_food and has_modifier(record, "food"):
            score -= food_penalty
            append_selection_reason(record, "food_penalty:consecutive")
        record["selection_score"] = round(clamp(score), 3)
        previous_food = has_modifier(record, "food")

    ranked_indices = rank_scene_indices(records)
    selected_set = enforce_food_quota(records, ranked_indices, trip_type)

    for rank, index in enumerate(ranked_indices, start=1):
        records[index]["selection_rank"] = rank

    for index, record in enumerate(records):
        selected_for_edit = index in selected_set
        record["selected_for_edit"] = selected_for_edit
        record["priority_band"] = priority_band(float(record.get("selection_score") or 0.0))
        record["importance_score"] = record.get("selection_score")
        record["importance_reasons"] = list(record.get("selection_reasons") or [])


def build_scene_records(media_scene: pd.DataFrame, reps: pd.DataFrame, root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    image_index = tagging_pipeline.build_image_index(root)
    cache_dir = Path("outputs") / "ocr_cache"

    media_scene = media_scene.copy()
    media_scene["final_timestamp_dt"] = media_scene["final_timestamp"].map(parse_timestamp)
    media_scene = media_scene[media_scene["scene_id"].notna()].copy()
    media_scene["scene_id"] = pd.to_numeric(media_scene["scene_id"], errors="coerce").astype("Int64")

    reps = reps.copy()
    reps["scene_id"] = pd.to_numeric(reps["scene_id"], errors="coerce").astype("Int64")

    for scene_id, group in media_scene.groupby("scene_id", sort=False):
        if pd.isna(scene_id):
            continue

        group = group.sort_values("final_timestamp_dt", kind="stable").reset_index(drop=True)
        start_ts = group["final_timestamp_dt"].min()
        end_ts = group["final_timestamp_dt"].max()
        duration_seconds = None
        if pd.notna(start_ts) and pd.notna(end_ts):
            duration_seconds = (end_ts - start_ts).total_seconds()

        rep_match = reps[reps["scene_id"] == scene_id]
        rep_row = rep_match.iloc[0] if not rep_match.empty else None

        gps_summary = compute_gps_summary(group)
        scene_food = analyze_scene_food(group, rep_row, root, image_index, cache_dir)
        signals, selection_reasons, scene_meta = compute_scene_signals(group, rep_row, gps_summary, scene_food=scene_food)

        video_count = int((group["kind"].astype(str).str.lower() == "video").sum())
        image_count = int((group["kind"].astype(str).str.lower() == "image").sum())
        models = unique_nonmissing(group["model"])
        kinds = unique_nonmissing(group["kind"])
        tokens = list(scene_meta.get("meaningful_tokens") or [])
        rep_tag = str(scene_meta.get("representative_tag") or "風景")
        rep_strength = str(scene_meta.get("tag_strength") or "weak")
        modifiers = list(scene_meta.get("modifiers") or [])
        primary_type = str(scene_meta.get("primary_type") or "landscape")
        food_confidence = float(scene_meta.get("food_confidence") or 0.0)
        representative_food_confidence = float(scene_meta.get("food_confidence_representative") or 0.0)
        food_sample_count = int(scene_meta.get("food_sample_count") or 0)
        food_evidence_count = int(scene_meta.get("food_evidence_count") or 0)
        food_sample_paths = list(scene_meta.get("food_sample_paths") or [])
        max_gap_seconds = float(scene_meta.get("max_gap_seconds") or 0.0)

        scene_record: dict[str, object] = {
            "scene_id": int(scene_id),
            "start_at": format_timestamp(start_ts),
            "end_at": format_timestamp(end_ts),
            "duration_seconds": round(duration_seconds, 3) if duration_seconds is not None else None,
            "max_gap_seconds": round(max_gap_seconds, 3),
            "asset_count": int(len(group)),
            "image_count": image_count,
            "video_count": video_count,
            "kinds": kinds,
            "models": models,
            "gps": gps_summary,
            "importance_score": None,
            "importance_reasons": [],
            "priority_band": "low",
            "primary_type": primary_type,
            "modifiers": modifiers,
            "food_confidence": round(food_confidence, 3),
            "food_confidence_representative": round(representative_food_confidence, 3),
            "food_sample_count": food_sample_count,
            "food_evidence_count": food_evidence_count,
            "food_sample_paths": food_sample_paths,
            "selection_components": signals,
            "selection_score": None,
            "selection_rank": None,
            "selected_for_edit": False,
            "selection_reasons": selection_reasons,
            "trip_type": None,
            "tag_strength": rep_strength,
            "representative_tag": rep_tag,
            "meaningful_ocr_token_count": len(tokens),
            "meaningful_ocr_tokens": tokens,
            "preview_sources": build_preview_sources(group, root),
            "representative": None,
        }

        if rep_row is not None:
            scene_record["representative"] = {
                "path": relpath(rep_row.get("representative_path"), root),
                "kind": rep_row.get("representative_kind"),
                "captured_at": format_timestamp(parse_timestamp(rep_row.get("representative_final_timestamp"))),
                "tag": rep_tag,
                "tag_strength": rep_strength,
                "caption": None if is_missing(rep_row.get("caption")) else str(rep_row.get("caption")),
                "primary_type": primary_type,
                "modifiers": modifiers,
                "food_confidence": round(food_confidence, 3),
                "food_confidence_representative": round(representative_food_confidence, 3),
                "meaningful_ocr_token_count": len(tokens),
                "meaningful_ocr_tokens": tokens,
                "face_count_raw": None if is_missing(rep_row.get("face_count_raw")) else int(float(rep_row.get("face_count_raw"))),
                "face_count_filtered": None if is_missing(rep_row.get("face_count_filtered")) else int(float(rep_row.get("face_count_filtered"))),
                "face_count": None if is_missing(rep_row.get("face_count")) else int(float(rep_row.get("face_count"))),
                "duration_seconds": parse_float(rep_row.get("representative_duration_seconds")),
                "blur_score": parse_float(rep_row.get("representative_laplacian")),
            }

        records.append(scene_record)

    records.sort(key=lambda item: (item.get("start_at") or "", item.get("scene_id") or 0))
    apply_selection_strategy(records)
    for record in records:
        record["flow_summary"] = build_flow_summary(record)
    return records


def build_payload_summary(records: list[dict[str, object]]) -> dict[str, object]:
    priority_counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    tag_counts: dict[str, int] = {}
    gps_scene_count = 0
    mixed_scene_count = 0
    video_only_scene_count = 0
    image_only_scene_count = 0
    food_scene_count = 0
    selected_scene_count = 0
    selected_food_scene_count = 0
    trip_type = str(records[0].get("trip_type") or "general") if records else "general"

    for record in records:
        priority = str(record.get("priority_band") or "low")
        if priority not in priority_counts:
            priority_counts[priority] = 0
        priority_counts[priority] += 1

        tag = record.get("representative_tag")
        if tag is not None:
            tag_text = str(tag)
            tag_counts[tag_text] = tag_counts.get(tag_text, 0) + 1

        gps = record.get("gps") if isinstance(record.get("gps"), dict) else {}
        if gps.get("has_gps"):
            gps_scene_count += 1

        image_count = int(record.get("image_count") or 0)
        video_count = int(record.get("video_count") or 0)
        if image_count > 0 and video_count > 0:
            mixed_scene_count += 1
        elif image_count > 0:
            image_only_scene_count += 1
        elif video_count > 0:
            video_only_scene_count += 1

        is_food = has_modifier(record, "food")
        if is_food:
            food_scene_count += 1
        if bool(record.get("selected_for_edit")):
            selected_scene_count += 1
            if is_food:
                selected_food_scene_count += 1

    return {
        "scene_count": len(records),
        "trip_type": trip_type,
        "priority_counts": priority_counts,
        "gps_scene_count": gps_scene_count,
        "mixed_scene_count": mixed_scene_count,
        "image_only_scene_count": image_only_scene_count,
        "video_only_scene_count": video_only_scene_count,
        "food_scene_count": food_scene_count,
        "food_scene_ratio": round(food_scene_count / max(len(records), 1), 3) if records else 0.0,
        "selected_scene_count": selected_scene_count,
        "selected_food_scene_count": selected_food_scene_count,
        "representative_tag_counts": dict(sorted(tag_counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build scene edit candidate JSON from scene CSVs.")
    parser.add_argument("--media-scene", default=DEFAULT_MEDIA_SCENE, help="Scene CSV with per-asset rows")
    parser.add_argument("--representatives", default=DEFAULT_REPRESENTATIVES, help="Scene representative CSV with tags")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--output-dir", default=None, help="Directory for outputs. Defaults to outputs/<run>/candidates/")
    parser.add_argument("--run-dir", default=None, help="Shared run directory for the whole workflow")
    parser.add_argument("--root", default=".", help="Workspace root used to shorten paths")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    media_scene_path = Path(args.media_scene)
    reps_path = Path(args.representatives)
    output_path = resolve_output_path(
        default_filename="scene_edit_candidates.json",
        step_name="candidates",
        output=args.output,
        output_dir=args.output_dir,
        run_dir=args.run_dir,
        input_path=media_scene_path,
    )

    log(f"[candidates] media_scene={media_scene_path}")
    log(f"[candidates] representatives={reps_path}")
    log(f"[candidates] output={output_path}")
    start = time.monotonic()

    media_scene = load_csv(media_scene_path)
    reps = load_representatives([reps_path])

    records = build_scene_records(media_scene, reps, root)
    summary = build_payload_summary(records)
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "root": str(root),
        "scene_count": len(records),
        "summary": summary,
        "scenes": records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_manifest(
        output_path,
        {
            "generated_at": payload["generated_at"],
            "step": "candidates",
            "media_scene": str(media_scene_path),
            "representatives": str(reps_path),
            "output": str(output_path),
            "scene_count": int(len(records)),
            "elapsed_seconds": round(time.monotonic() - start, 3),
        },
    )
    log(f"[candidates] wrote {output_path} ({len(records)} scenes, elapsed={summarize_elapsed(start)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
