from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from sceneflow.workflow_utils import ProgressReporter, log, resolve_output_path, summarize_elapsed, write_manifest


def to_float(value: object) -> float:
    try:
        if pd.isna(value):
            return float("-inf")
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return float("-inf")


def to_numeric(value: object) -> float | None:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def video_priority(duration_seconds: object) -> tuple[int, float]:
    duration = to_numeric(duration_seconds)
    if duration is None:
        return (1, float("inf"))
    if duration <= 10:
        return (0, abs(duration - 6.0))
    return (1, abs(duration - 6.0))


def image_priority(blur_score: object) -> tuple[int, float]:
    blur = to_float(blur_score)
    return (0, -blur)


def pick_representative(group: pd.DataFrame) -> pd.Series:
    videos = group[group["kind"].astype(str).str.lower().eq("video")].copy()
    images = group[group["kind"].astype(str).str.lower().eq("image")].copy()

    # OCR and face detection are much more reliable on stills than on video grabs,
    # so prefer the sharpest image whenever a scene contains one.
    if not images.empty:
        images["_image_priority"] = images["laplacian"].map(image_priority)
        images["_blur_score"] = images["laplacian"].map(to_float)
        ordered = images.sort_values(
            by=["_blur_score", "final_timestamp", "path"],
            ascending=[False, True, True],
            kind="stable",
        )
        return ordered.iloc[0]

    short_videos = videos[pd.to_numeric(videos["duration_seconds"], errors="coerce") <= 10].copy()
    if not short_videos.empty:
        short_videos["_video_priority"] = short_videos["duration_seconds"].map(lambda v: video_priority(v)[1])
        short_videos["_blur_score"] = short_videos["laplacian"].map(to_float)
        ordered = short_videos.sort_values(
            by=["_video_priority", "_blur_score", "final_timestamp", "path"],
            ascending=[True, False, True, True],
            kind="stable",
        )
        return ordered.iloc[0]

    # Fallback: long videos only.
    videos["_video_priority"] = videos["duration_seconds"].map(lambda v: video_priority(v)[1])
    videos["_blur_score"] = videos["laplacian"].map(to_float)
    ordered = videos.sort_values(
        by=["_video_priority", "final_timestamp", "path"],
        ascending=[True, True, True],
        kind="stable",
    )
    return ordered.iloc[0]


def build_representatives(input_data: Path | pd.DataFrame, reporter: ProgressReporter | None = None) -> pd.DataFrame:
    if isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        df = pd.read_csv(input_data)
    if "scene_id" not in df.columns:
        raise ValueError("scene_id column is required")

    df = df[df["scene_id"].notna()].copy()

    rows = []
    scene_groups = list(df.groupby("scene_id", sort=True))
    for index, (scene_id, group) in enumerate(scene_groups, start=1):
        if reporter is not None:
            reporter.update(index, f"scene={scene_id}")
        rep = pick_representative(group)
        rows.append(
            {
                "scene_id": scene_id,
                "asset_count": len(group),
                "representative_path": rep["path"],
                "representative_kind": rep["kind"],
                "representative_final_timestamp": rep.get("final_timestamp"),
                "representative_laplacian": rep.get("laplacian"),
                "representative_brightness": rep.get("brightness"),
                "representative_duration_seconds": rep.get("duration_seconds"),
                "representative_has_audio": rep.get("has_audio"),
            }
        )

    if reporter is not None:
        reporter.finish("representative selection complete")

    out = pd.DataFrame(rows).sort_values("scene_id", kind="stable").reset_index(drop=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Pick one representative asset per scene.")
    parser.add_argument("--input", default="media_scene.csv", help="Input CSV path with scene_id")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--output-dir", default=None, help="Directory for outputs. Defaults to outputs/<run>/representatives/")
    parser.add_argument("--run-dir", default=None, help="Shared run directory for the whole workflow")
    parser.add_argument("--progress-interval", type=int, default=5, help="How often to print progress updates")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = resolve_output_path(
        default_filename="scene_representatives.csv",
        step_name="representatives",
        output=args.output,
        output_dir=args.output_dir,
        run_dir=args.run_dir,
        input_path=input_path,
    )

    log(f"[representatives] input={input_path}")
    log(f"[representatives] output={output_path}")
    start = time.monotonic()
    input_df = pd.read_csv(input_path)
    df = build_representatives(input_df, reporter=ProgressReporter("representatives", total=int(input_df["scene_id"].dropna().nunique()), interval=args.progress_interval))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, na_rep="None")
    write_manifest(
        output_path,
        {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "step": "representatives",
            "input": str(input_path),
            "output": str(output_path),
            "row_count": int(len(df)),
            "scene_count": int(len(df)),
            "elapsed_seconds": round(time.monotonic() - start, 3),
        },
    )
    log(f"[representatives] wrote {output_path} ({len(df)} scenes, elapsed={summarize_elapsed(start)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
