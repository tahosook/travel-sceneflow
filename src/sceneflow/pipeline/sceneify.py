from __future__ import annotations

import argparse
import math
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from sceneflow.workflow_utils import log, resolve_output_path, summarize_elapsed, write_manifest


TIMESTAMP_RE = re.compile(r"(?P<date>\d{8})[._-](?P<time>\d{6})(?P<msec>\d{0,3})")
NORMALIZE_RE = re.compile(r"[-_.]+")


def is_missing(value: object) -> bool:
    if value is None:
        return True
    if pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "none"


def parse_timestamp(value: object, tz_name: str) -> pd.Timestamp | None:
    if is_missing(value):
        return None

    try:
        ts = pd.Timestamp(str(value).strip())
    except Exception:
        return None

    if ts.tzinfo is None:
        return ts.tz_localize(tz_name)
    return ts.tz_convert(tz_name)


def parse_numeric(value: object) -> float | None:
    if is_missing(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def infer_timestamp_from_path(path_value: object, tz_name: str) -> pd.Timestamp | None:
    if is_missing(path_value):
        return None

    match = TIMESTAMP_RE.search(Path(str(path_value)).name)
    if not match:
        return None

    text = f"{match.group('date')} {match.group('time')}{match.group('msec') or '000'}"
    for fmt in ("%Y%m%d %H%M%S%f", "%Y%m%d %H%M%S"):
        try:
            dt = datetime.strptime(text, fmt)
        except Exception:
            continue
        return pd.Timestamp(dt).tz_localize(tz_name)
    return None


def build_final_timestamp(df: pd.DataFrame, tz_name: str) -> pd.Series:
    captured = df["captured_at"].map(lambda v: parse_timestamp(v, tz_name))
    created = df["create_date"].map(lambda v: parse_timestamp(v, tz_name))
    inferred = df["path"].map(lambda v: infer_timestamp_from_path(v, tz_name))

    final = captured.copy()
    final = final.where(final.notna(), created)
    final = final.where(final.notna(), inferred)
    return final


def content_key(path_value: object) -> str | None:
    if is_missing(path_value):
        return None
    name = Path(str(path_value)).stem.lower()
    name = NORMALIZE_RE.sub("-", name).strip("-")
    return name or None


def format_timestamp(value: object) -> object:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat(sep=" ", timespec="seconds")


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def dedupe_videos(df: pd.DataFrame) -> pd.DataFrame:
    video_mask = df["kind"].eq("video")
    if not video_mask.any():
        return df

    videos = df.loc[video_mask].copy()
    videos["create_date_dt"] = videos["create_date"].map(lambda v: parse_timestamp(v, "Asia/Tokyo"))
    videos["duration_key"] = pd.to_numeric(videos["duration_seconds"], errors="coerce").round(3)

    # If the same video was exported twice, keep the copy with the newer timestamp.
    videos = videos.sort_values(
        ["final_timestamp_dt", "duration_key", "create_date_dt", "path"],
        kind="stable",
        na_position="last",
    )
    videos = videos.drop_duplicates(subset=["final_timestamp_dt", "duration_key"], keep="last")
    videos = videos.drop(columns=["create_date_dt", "duration_key"])

    others = df.loc[~video_mask]
    return pd.concat([others, videos], ignore_index=True)


def prefer_video_over_image(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["content_key"] = df["path"].map(content_key)

    video_keys = set(df.loc[df["kind"].eq("video") & df["content_key"].notna(), "content_key"])
    if not video_keys:
        return df.drop(columns=["content_key"])

    keep_mask = ~(
        df["kind"].eq("image")
        & df["content_key"].isin(video_keys)
    )
    return df.loc[keep_mask].drop(columns=["content_key"]).reset_index(drop=True)


def assign_scene_ids(df: pd.DataFrame, gap_minutes: int, gps_gap_meters: float, max_scene_minutes: int) -> pd.DataFrame:
    scene_ids: list[object] = []
    current_scene = 0
    previous_ts: pd.Timestamp | None = None
    scene_start_ts: pd.Timestamp | None = None
    previous_lat: float | None = None
    previous_lon: float | None = None

    time_gap = pd.Timedelta(minutes=gap_minutes)
    max_scene_gap = pd.Timedelta(minutes=max_scene_minutes)

    for _, row in df.iterrows():
        ts = row["final_timestamp_dt"]
        if pd.isna(ts):
            scene_ids.append(None)
            continue

        current_ts = pd.Timestamp(ts)
        current_lat = parse_numeric(row.get("gps_latitude"))
        current_lon = parse_numeric(row.get("gps_longitude"))

        if previous_ts is None:
            current_scene = 1
            scene_start_ts = current_ts
        else:
            elapsed = current_ts - previous_ts
            gps_gap = False
            if None not in (previous_lat, previous_lon, current_lat, current_lon):
                gps_gap = haversine_meters(previous_lat, previous_lon, current_lat, current_lon) >= gps_gap_meters

            too_long = scene_start_ts is not None and (current_ts - scene_start_ts) >= max_scene_gap

            if elapsed >= time_gap or gps_gap or too_long:
                current_scene += 1
                scene_start_ts = current_ts

        scene_ids.append(current_scene)
        previous_ts = current_ts
        previous_lat = current_lat
        previous_lon = current_lon

    df["scene_id"] = scene_ids
    return df


def absorb_singleton_scenes(df: pd.DataFrame, gps_gap_meters: float) -> pd.DataFrame:
    df = df.copy()
    scene_order = [scene_id for scene_id in df["scene_id"].dropna().drop_duplicates().tolist()]
    scene_groups = {scene_id: group.copy() for scene_id, group in df[df["scene_id"].notna()].groupby("scene_id", sort=False)}

    for index, scene_id in enumerate(scene_order):
        group = scene_groups.get(scene_id)
        if group is None or len(group) != 1:
            continue

        prev_scene = scene_order[index - 1] if index > 0 else None
        next_scene = scene_order[index + 1] if index + 1 < len(scene_order) else None
        current_ts = pd.Timestamp(group.iloc[0]["final_timestamp_dt"])

        target_scene = None
        prev_gap = None
        next_gap = None
        prev_gps = None
        next_gps = None

        if prev_scene is not None:
            prev_group = scene_groups.get(prev_scene)
            if prev_group is not None and not prev_group.empty:
                prev_ts = pd.Timestamp(prev_group["final_timestamp_dt"].max())
                prev_gap = current_ts - prev_ts
                prev_tail = prev_group.sort_values("final_timestamp_dt", kind="stable").iloc[-1]
                current_lat = parse_numeric(group.iloc[0].get("gps_latitude"))
                current_lon = parse_numeric(group.iloc[0].get("gps_longitude"))
                prev_lat = parse_numeric(prev_tail.get("gps_latitude"))
                prev_lon = parse_numeric(prev_tail.get("gps_longitude"))
                if None not in (prev_lat, prev_lon, current_lat, current_lon):
                    prev_gps = haversine_meters(prev_lat, prev_lon, current_lat, current_lon)

        if next_scene is not None:
            next_group = scene_groups.get(next_scene)
            if next_group is not None and not next_group.empty:
                next_ts = pd.Timestamp(next_group["final_timestamp_dt"].min())
                next_gap = next_ts - current_ts
                next_head = next_group.sort_values("final_timestamp_dt", kind="stable").iloc[0]
                current_lat = parse_numeric(group.iloc[0].get("gps_latitude"))
                current_lon = parse_numeric(group.iloc[0].get("gps_longitude"))
                next_lat = parse_numeric(next_head.get("gps_latitude"))
                next_lon = parse_numeric(next_head.get("gps_longitude"))
                if None not in (next_lat, next_lon, current_lat, current_lon):
                    next_gps = haversine_meters(current_lat, current_lon, next_lat, next_lon)

        if prev_gap is None and next_gap is None:
            continue
        if prev_gap is None and (next_gps is None or next_gps <= gps_gap_meters):
            target_scene = next_scene
        elif next_gap is None and (prev_gps is None or prev_gps <= gps_gap_meters):
            target_scene = prev_scene
        elif (prev_gps is None or prev_gps <= gps_gap_meters) and (next_gps is None or next_gps <= gps_gap_meters) and prev_gap <= next_gap:
            target_scene = prev_scene
        elif (prev_gps is None or prev_gps <= gps_gap_meters) and (next_gps is None or next_gps <= gps_gap_meters):
            target_scene = next_scene

        if target_scene is not None:
            df.loc[df["scene_id"] == scene_id, "scene_id"] = target_scene
            scene_groups[target_scene] = pd.concat([scene_groups[target_scene], group], ignore_index=True)
            scene_groups.pop(scene_id, None)

    return df


def renumber_scene_ids(df: pd.DataFrame) -> pd.DataFrame:
    mapping: dict[object, int] = {}
    next_id = 1
    new_ids: list[object] = []
    for value in df["scene_id"]:
        if pd.isna(value):
            new_ids.append(None)
            continue
        if value not in mapping:
            mapping[value] = next_id
            next_id += 1
        new_ids.append(mapping[value])
    df = df.copy()
    df["scene_id"] = new_ids
    return df


def process_csv(
    input_path: Path,
    output_path: Path,
    tz_name: str,
    gap_minutes: int,
    gps_gap_meters: float,
    max_scene_minutes: int,
) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    df["final_timestamp_dt"] = build_final_timestamp(df, tz_name)
    df = prefer_video_over_image(df)
    df = dedupe_videos(df)
    df = df.sort_values(["final_timestamp_dt", "path"], kind="stable", na_position="last").reset_index(drop=True)
    df = assign_scene_ids(df, gap_minutes, gps_gap_meters, max_scene_minutes)
    df["final_timestamp"] = df["final_timestamp_dt"].map(format_timestamp)
    df["initial_scene_id"] = df["scene_id"]

    final_scene_ids: list[object] = []
    current_scene = 0

    for _, group in df.groupby("initial_scene_id", sort=False):
        if group["initial_scene_id"].isna().all():
            final_scene_ids.extend([None] * len(group))
            continue

        group = group.sort_values(["final_timestamp_dt", "path"], kind="stable", na_position="last")
        span = group["final_timestamp_dt"].max() - group["final_timestamp_dt"].min()
        is_giant = len(group) >= 30 or span > pd.Timedelta(minutes=30)

        if not is_giant:
            current_scene += 1
            final_scene_ids.extend([current_scene] * len(group))
            continue

        previous_ts: pd.Timestamp | None = None
        giant_gap = pd.Timedelta(minutes=5)
        for ts in group["final_timestamp_dt"]:
            if pd.isna(ts):
                final_scene_ids.append(None)
                continue

            current_ts = pd.Timestamp(ts)
            if previous_ts is None or current_ts - previous_ts >= giant_gap:
                current_scene += 1

            final_scene_ids.append(current_scene)
            previous_ts = current_ts

    df["scene_id"] = final_scene_ids
    df = absorb_singleton_scenes(df, gps_gap_meters)
    df = df.sort_values(["final_timestamp_dt", "path"], kind="stable", na_position="last").reset_index(drop=True)
    df = renumber_scene_ids(df)
    df = df.drop(columns=["final_timestamp_dt"], errors="ignore")
    df = df.drop(columns=["initial_scene_id"], errors="ignore")

    columns = list(df.columns)
    if "final_timestamp" in columns:
        columns.remove("final_timestamp")
        insert_at = columns.index("captured_at") + 1 if "captured_at" in columns else len(columns)
        columns.insert(insert_at, "final_timestamp")
    if "scene_id" in columns:
        columns.remove("scene_id")
        columns.append("scene_id")
    return df.reindex(columns=columns)


def main() -> int:
    parser = argparse.ArgumentParser(description="Add final_timestamp and scene_id to a media CSV.")
    parser.add_argument("--input", default="media_info.csv", help="Input CSV path")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--output-dir", default=None, help="Directory for outputs. Defaults to outputs/<run>/sceneify/")
    parser.add_argument("--run-dir", default=None, help="Shared run directory for the whole workflow")
    parser.add_argument("--timezone", default="Asia/Tokyo", help="Timezone for naive timestamps")
    parser.add_argument("--gap-minutes", type=int, default=15, help="Split a scene when time gap exceeds this value")
    parser.add_argument("--gps-gap-meters", type=float, default=100.0, help="Split a scene when GPS distance exceeds this value")
    parser.add_argument("--max-scene-minutes", type=int, default=60, help="Split a scene when it gets longer than this value")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = resolve_output_path(
        default_filename="media_scene.csv",
        step_name="sceneify",
        output=args.output,
        output_dir=args.output_dir,
        run_dir=args.run_dir,
        input_path=input_path,
    )

    log(f"[sceneify] input={input_path}")
    log(f"[sceneify] output={output_path}")
    start = time.monotonic()
    df = process_csv(
        input_path=input_path,
        output_path=output_path,
        tz_name=args.timezone,
        gap_minutes=args.gap_minutes,
        gps_gap_meters=args.gps_gap_meters,
        max_scene_minutes=args.max_scene_minutes,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, na_rep="None")
    write_manifest(
        output_path,
        {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "step": "sceneify",
            "input": str(input_path),
            "output": str(output_path),
            "row_count": int(len(df)),
            "scene_count": int(df["scene_id"].dropna().nunique()),
            "elapsed_seconds": round(time.monotonic() - start, 3),
        },
    )
    log(
        f"[sceneify] wrote {output_path} ({len(df)} rows, {df['scene_id'].dropna().nunique()} scenes, elapsed={summarize_elapsed(start)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
