from __future__ import annotations

from pathlib import Path

from sceneflow.pipeline import candidates, llm_plan, meanings, render, representatives, sceneify, structure
from tests.helpers import build_tagged_representatives


def test_sceneify_and_representatives_contracts(media_info_csv: Path) -> None:
    scene_df = sceneify.process_csv(
        input_path=media_info_csv,
        output_path=Path("unused.csv"),
        tz_name="Asia/Tokyo",
        gap_minutes=15,
        gps_gap_meters=100.0,
        max_scene_minutes=60,
    )

    assert scene_df["scene_id"].tolist() == [1, 1, 1, 2, 2, 3]
    assert "final_timestamp" in scene_df.columns
    assert scene_df["final_timestamp"].notna().all()

    reps_df = representatives.build_representatives(scene_df)
    rep_paths = {
        int(row.scene_id): Path(str(row.representative_path)).name
        for row in reps_df.itertuples(index=False)
    }

    assert rep_paths == {
        1: "overview-02.jpg",
        2: "transit-01.mp4",
        3: "temple-01.jpg",
    }


def test_candidates_to_llm_contracts(media_info_csv: Path, sample_root: Path) -> None:
    scene_df = sceneify.process_csv(
        input_path=media_info_csv,
        output_path=Path("unused.csv"),
        tz_name="Asia/Tokyo",
        gap_minutes=15,
        gps_gap_meters=100.0,
        max_scene_minutes=60,
    )
    reps_df = representatives.build_representatives(scene_df)
    tagged_reps = build_tagged_representatives(reps_df)

    records = candidates.build_scene_records(scene_df, tagged_reps, sample_root)
    payload_summary = candidates.build_payload_summary(records)
    meanings_payload = meanings.build_meanings(
        {
            "generated_at": "2026-03-29T11:00:00+09:00",
            "scenes": records,
        }
    )
    structure_payload = structure.build_structure(meanings_payload)
    prompt_text = llm_plan.build_prompt(structure_payload)
    draft_plan = llm_plan.build_draft_plan(structure_payload)

    assert len(records) == 3
    assert payload_summary["scene_count"] == 3
    assert records[0]["representative"]["path"] == "trip/day1/overview-02.jpg"
    assert "浅草寺" in records[2]["representative"]["meaningful_ocr_tokens"]
    assert [source["path"] for source in records[0]["preview_sources"]] == [
        "trip/day1/overview-01.jpg",
        "trip/day1/overview-clip.mp4",
        "trip/day1/overview-02.jpg",
    ]
    assert [source["path"] for source in records[1]["preview_sources"]] == [
        "trip/day1/transit-01.mp4",
        "trip/day1/transit-02.mp4",
    ]

    assert set(meanings_payload) >= {"generated_at", "summary", "scenes", "scene_count"}
    assert set(structure_payload) >= {"generated_at", "summary", "chapters", "edit_sequence", "scene_count"}
    assert set(draft_plan) >= {"title", "logline", "chapter_list", "edit_sequence", "render_guidance"}

    assert [item["scene_id"] for item in structure_payload["edit_sequence"]] == [1, 2, 3]
    assert all(item["planned_duration_seconds"] > 0 for item in structure_payload["edit_sequence"])
    assert len(structure_payload["edit_sequence"][0]["preview_sources"]) == 3
    assert len(draft_plan["edit_sequence"][1]["preview_sources"]) == 2
    assert draft_plan["render_guidance"]["preferred_order"] == [1, 2, 3]
    assert "自然で見やすい動画の構成案" in prompt_text


def test_render_builds_multiple_clips_from_preview_sources(
    media_info_csv: Path,
    materialized_media_files: list[Path],
    sample_root: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    scene_df = sceneify.process_csv(
        input_path=media_info_csv,
        output_path=Path("unused.csv"),
        tz_name="Asia/Tokyo",
        gap_minutes=15,
        gps_gap_meters=100.0,
        max_scene_minutes=60,
    )
    reps_df = representatives.build_representatives(scene_df)
    tagged_reps = build_tagged_representatives(reps_df)
    records = candidates.build_scene_records(scene_df, tagged_reps, sample_root)
    meanings_payload = meanings.build_meanings({"generated_at": "2026-03-29T11:00:00+09:00", "scenes": records})
    structure_payload = structure.build_structure(meanings_payload)
    draft_plan = llm_plan.build_draft_plan(structure_payload)

    def fake_normalize_clip(input_path: Path, output_path: Path, duration: float) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(f"{input_path.name}:{duration:.2f}".encode("utf-8"))

    monkeypatch.setattr(render, "normalize_clip", fake_normalize_clip)

    clips = render.build_clip_list({"plan": draft_plan}, sample_root, tmp_path)

    assert len(materialized_media_files) == 6
    assert len(clips) == 6
    assert [clip["scene_id"] for clip in clips] == [1, 1, 1, 2, 2, 3]
    assert clips[1]["transition_hint"] == "cut"
    assert clips[0]["preview_index"] == 1
    assert clips[2]["preview_index"] == 3
    assert clips[0]["duration_seconds"] < structure_payload["edit_sequence"][0]["planned_duration_seconds"]


def test_render_falls_back_to_representative_when_preview_sources_are_missing(
    media_info_csv: Path,
    materialized_media_files: list[Path],
    sample_root: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    scene_df = sceneify.process_csv(
        input_path=media_info_csv,
        output_path=Path("unused.csv"),
        tz_name="Asia/Tokyo",
        gap_minutes=15,
        gps_gap_meters=100.0,
        max_scene_minutes=60,
    )
    reps_df = representatives.build_representatives(scene_df)
    tagged_reps = build_tagged_representatives(reps_df)
    records = candidates.build_scene_records(scene_df, tagged_reps, sample_root)
    meanings_payload = meanings.build_meanings({"generated_at": "2026-03-29T11:00:00+09:00", "scenes": records})
    structure_payload = structure.build_structure(meanings_payload)
    draft_plan = llm_plan.build_draft_plan(structure_payload)

    first_scene = draft_plan["edit_sequence"][0]
    first_scene["preview_sources"] = [{"path": "trip/day1/missing.jpg", "kind": "image"}]

    def fake_normalize_clip(input_path: Path, output_path: Path, duration: float) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(f"{input_path.name}:{duration:.2f}".encode("utf-8"))

    monkeypatch.setattr(render, "normalize_clip", fake_normalize_clip)

    clips = render.build_clip_list({"plan": draft_plan}, sample_root, tmp_path)

    assert len(materialized_media_files) == 6
    assert clips[0]["scene_id"] == 1
    assert clips[0]["source_path"].endswith("overview-02.jpg")
    assert clips[0]["preview_index"] == 1
