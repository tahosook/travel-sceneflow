from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from sceneflow.workflow_utils import log, resolve_output_path, summarize_elapsed, write_manifest


DEFAULT_MEDIA_SCENE = "media_scene.csv"
DEFAULT_REPRESENTATIVES = "scene_representatives_tagged.csv"
DEFAULT_OUTPUT = "scene_edit_candidates.json"
TZ_NAME = "Asia/Tokyo"
PREVIEW_SOURCE_LIMIT = 3

TAG_STRENGTH = {
    "人物": "strong",
    "集合写真": "strong",
    "駅": "strong",
    "寺社": "strong",
    "食事": "strong",
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
    "食事",
    "ランチ",
    "ディナー",
    "ラーメン",
    "寿司",
    "そば",
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


def is_missing(value: object) -> bool:
    if value is None:
        return True
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
    return " / ".join([part for part in [duration_label, *kind_parts, gps_label, f"tag:{tag}", f"priority:{priority}"] if part])


def compute_importance(group: pd.DataFrame, rep: pd.Series | None, gps_summary: dict[str, object]) -> tuple[float, list[str]]:
    # This score is meant to help choose scenes for the final edit, so we bias it
    # toward scene continuity and viewing flow rather than OCR/face semantics.
    score = 0.4
    reasons: list[str] = []

    asset_count = int(len(group))
    duration_seconds = group["final_timestamp_dt"].max() - group["final_timestamp_dt"].min()
    max_gap_seconds = compute_max_gap_seconds(group)

    if asset_count >= 2:
        score += min(0.08 + 0.01 * min(asset_count, 8), 0.14)
        reasons.append("continuous_assets")

    if duration_seconds is not None:
        total_seconds = duration_seconds.total_seconds()
        if total_seconds >= 30:
            score += min(total_seconds / 1800.0, 0.12)
            reasons.append("scene_duration")

    if gps_summary.get("has_gps"):
        score += 0.05
        reasons.append("gps")

    if max_gap_seconds <= 120 and asset_count >= 2:
        score += 0.08
        reasons.append("tight_flow")
    elif max_gap_seconds <= 300:
        score += 0.04
        reasons.append("moderate_flow")

    if rep is not None:
        tag = normalize_tag(rep.get("tag"))
        strength = tag_strength(tag)
        face_count = parse_float(rep.get("face_count_filtered")) or 0.0
        rep_blur = parse_float(rep.get("representative_laplacian"))
        rep_kind = str(rep.get("representative_kind")) if not is_missing(rep.get("representative_kind")) else None
        rep_duration = parse_float(rep.get("representative_duration_seconds"))
        meaningful_tokens = meaningful_ocr_tokens(rep.get("ocr_text"))

        if tag in {"食事", "駅", "寺社", "集合写真", "人物"}:
            score += 0.05
            reasons.append(f"tag:{tag}")
        elif tag in {"建物", "移動"}:
            score += 0.03
            reasons.append(f"tag:{tag}")
        elif tag == "風景":
            score += 0.01
            reasons.append("tag:風景")

        if strength == "strong":
            score += 0.02
            reasons.append("strength:strong")
        elif strength == "medium":
            score += 0.01
            reasons.append("strength:medium")
        else:
            reasons.append("strength:weak")

        if face_count > 0:
            score += min(face_count * 0.01, 0.03)
            reasons.append("face")

        if meaningful_tokens:
            score += min(0.01 + 0.005 * len(meaningful_tokens), 0.03)
            reasons.append("ocr")

        if rep_kind == "video" and rep_duration is not None and rep_duration <= 8:
            score += 0.04
            reasons.append("short_video")

        if rep_kind == "video" and rep_duration is not None and rep_duration > 10:
            score -= 0.05
            reasons.append("long_video")

        if rep_blur is not None:
            blur_boost = min(max(rep_blur, 0.0) / 4000.0, 1.0) * 0.05
            score += blur_boost
            if blur_boost > 0:
                reasons.append("sharp")

        if tag == "風景" and not meaningful_tokens:
            score -= 0.04
            reasons.append("sparse_ocr")

    if duration_seconds is not None:
        total_seconds = duration_seconds.total_seconds()
        if total_seconds > 1200:
            score -= 0.18
            reasons.append("very_long")
        elif total_seconds > 600:
            score -= 0.10
            reasons.append("long")

    if asset_count > 15:
        score -= 0.08
        reasons.append("many_assets")

    if max_gap_seconds > 600:
        score -= 0.14
        reasons.append("big_gap")

    score = max(0.0, min(score, 1.0))
    return round(score, 3), reasons


def build_scene_records(media_scene: pd.DataFrame, reps: pd.DataFrame, root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

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
        importance_score, importance_reasons = compute_importance(group, rep_row, gps_summary)
        max_gap_seconds = compute_max_gap_seconds(group)

        video_count = int((group["kind"].astype(str).str.lower() == "video").sum())
        image_count = int((group["kind"].astype(str).str.lower() == "image").sum())
        models = unique_nonmissing(group["model"])
        kinds = unique_nonmissing(group["kind"])
        tokens = meaningful_ocr_tokens(rep_row.get("ocr_text")) if rep_row is not None else []
        rep_tag = normalize_tag(rep_row.get("tag")) if rep_row is not None else None
        rep_strength = tag_strength(rep_tag) if rep_row is not None else "weak"
        scene_priority = priority_band(importance_score)

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
            "importance_score": importance_score,
            "importance_reasons": importance_reasons,
            "priority_band": scene_priority,
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
                "meaningful_ocr_token_count": len(tokens),
                "meaningful_ocr_tokens": tokens,
                "face_count_raw": None if is_missing(rep_row.get("face_count_raw")) else int(float(rep_row.get("face_count_raw"))),
                "face_count_filtered": None if is_missing(rep_row.get("face_count_filtered")) else int(float(rep_row.get("face_count_filtered"))),
                "face_count": None if is_missing(rep_row.get("face_count")) else int(float(rep_row.get("face_count"))),
                "duration_seconds": parse_float(rep_row.get("representative_duration_seconds")),
                "blur_score": parse_float(rep_row.get("representative_laplacian")),
            }

        scene_record["flow_summary"] = build_flow_summary(scene_record)
        records.append(scene_record)

    records.sort(key=lambda item: (item.get("start_at") or "", item.get("scene_id") or 0))
    return records


def build_payload_summary(records: list[dict[str, object]]) -> dict[str, object]:
    priority_counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    tag_counts: dict[str, int] = {}
    gps_scene_count = 0
    mixed_scene_count = 0
    video_only_scene_count = 0
    image_only_scene_count = 0

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

    return {
        "scene_count": len(records),
        "priority_counts": priority_counts,
        "gps_scene_count": gps_scene_count,
        "mixed_scene_count": mixed_scene_count,
        "image_only_scene_count": image_only_scene_count,
        "video_only_scene_count": video_only_scene_count,
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
