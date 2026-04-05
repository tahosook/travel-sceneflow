from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from sceneflow.pipeline import candidates, llm_plan, meanings, render, representatives, sceneify, structure
from tests.helpers import build_tagged_representatives


CAPTION_BY_TAG = {
    "風景": "風景の広がりが伝わる一枚です。",
    "移動": "移動中の様子を捉えた一枚です。",
    "寺社": "寺社の雰囲気が伝わる一枚です。",
    "人物": "人物が写る一枚です。",
    "集合写真": "集合写真の一枚です。",
    "駅": "駅の様子が伝わる一枚です。",
    "建物": "建物の様子がわかる一枚です。",
    "夜景": "夜の景色が印象的な一枚です。",
}


def build_synthetic_inputs(tmp_path: Path, scene_specs: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    media_rows: list[dict[str, object]] = []
    representative_rows: list[dict[str, object]] = []
    base_time = pd.Timestamp("2024-05-01T09:00:00+09:00")

    for scene_id, spec in enumerate(scene_specs, start=1):
        asset_count = int(spec.get("asset_count", 2))
        asset_kind = str(spec.get("kind", "image"))
        representative_kind = str(spec.get("representative_kind", asset_kind))
        scene_start = base_time + pd.Timedelta(minutes=(scene_id - 1) * 5)
        rep_extension = ".mp4" if representative_kind == "video" else ".jpg"
        representative_path = tmp_path / f"scene-{scene_id}{rep_extension}"

        for asset_index in range(asset_count):
            current_kind = representative_kind if asset_index == asset_count - 1 else asset_kind
            extension = ".mp4" if current_kind == "video" else ".jpg"
            media_rows.append(
                {
                    "scene_id": scene_id,
                    "path": str(tmp_path / f"scene-{scene_id}-{asset_index}{extension}"),
                    "kind": current_kind,
                    "final_timestamp": (scene_start + pd.Timedelta(seconds=asset_index * 18)).isoformat(),
                    "duration_seconds": float(spec.get("clip_duration_seconds", 6.0 if current_kind == "video" else 0.0)),
                    "gps_latitude": spec.get("gps_latitude"),
                    "gps_longitude": spec.get("gps_longitude"),
                    "model": spec.get("model", "camera"),
                }
            )

        tag = str(spec.get("tag", "風景"))
        representative_rows.append(
            {
                "scene_id": scene_id,
                "representative_path": str(representative_path),
                "representative_kind": representative_kind,
                "representative_final_timestamp": scene_start.isoformat(),
                "representative_laplacian": float(spec.get("blur_score", 2500.0)),
                "representative_brightness": float(spec.get("brightness", 128.0)),
                "representative_duration_seconds": float(spec.get("representative_duration_seconds", 6.0 if representative_kind == "video" else 0.0)),
                "representative_has_audio": bool(spec.get("representative_has_audio", representative_kind == "video")),
                "ocr_text": str(spec.get("ocr_text", "")),
                "face_count_raw": int(spec.get("face_count", 0)),
                "face_count_filtered": int(spec.get("face_count", 0)),
                "face_count": int(spec.get("face_count", 0)),
                "tag": tag,
                "primary_type": str(spec.get("primary_type", "landscape")),
                "modifiers": list(spec.get("modifiers", [])),
                "food_confidence": float(spec.get("food_confidence", 0.0)),
                "caption": str(spec.get("caption", CAPTION_BY_TAG.get(tag, "旅の一枚です。"))),
            }
        )

    return pd.DataFrame(media_rows), pd.DataFrame(representative_rows)


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
    assert set(draft_plan) >= {"title", "logline", "chapter_list", "title_cards", "edit_sequence", "render_guidance", "editor_brief"}

    assert meanings_payload["summary"]["role_counts"] == {"closing": 1, "opening": 1, "transition": 1}
    assert meanings_payload["summary"]["action_counts"] == {"keep": 3}
    assert [chapter["scene_count"] for chapter in structure_payload["chapters"]] == [1, 1, 1]
    assert [item["scene_id"] for item in structure_payload["edit_sequence"]] == [1, 2, 3]
    assert structure_payload["edit_sequence"][0]["start_at"] == "2024-05-01T09:00:00+09:00"
    assert all(item["planned_duration_seconds"] > 0 for item in structure_payload["edit_sequence"])
    assert len(structure_payload["edit_sequence"][0]["preview_sources"]) == 3
    assert [card["title"] for card in draft_plan["title_cards"]] == ["2024年5月1日", "旅の流れ", "旅の余韻"]
    assert all(card["presentation"] == "overlay" for card in draft_plan["title_cards"])
    assert draft_plan["title_cards"][0]["subtitle"] == "旅のはじまり"
    assert len(draft_plan["edit_sequence"][1]["preview_sources"]) == 2
    assert draft_plan["editor_brief"]["caption_density"] == "standard"
    assert draft_plan["editor_brief"]["tone"] == "gentle"
    assert all(item["base_planned_duration_seconds"] > 0 for item in draft_plan["edit_sequence"])
    assert all(item["editorial_emphasis"] == "default" for item in draft_plan["edit_sequence"])
    assert draft_plan["subtitle_plan"]["enabled"] is True
    assert [item["text"] for item in draft_plan["subtitle_plan"]["items"]] == [
        "風景の広がりが伝わる場面",
        "移動中の様子を捉えた場面",
        "寺社の雰囲気が伝わる場面",
    ]
    assert draft_plan["render_guidance"]["preferred_order"] == [1, 2, 3]
    assert draft_plan["render_guidance"]["title_overlay_mode"] == "overlay"
    assert draft_plan["render_guidance"]["caption_density"] == "standard"
    assert "自然で見やすい動画の構成案" in prompt_text


def test_food_modifier_does_not_become_primary_or_dominate(tmp_path: Path) -> None:
    scene_df, reps_df = build_synthetic_inputs(
        tmp_path,
        [
            {
                "tag": "風景",
                "primary_type": "landscape",
                "modifiers": ["food"],
                "food_confidence": 0.92,
                "asset_count": 2,
                "blur_score": 1500.0,
            },
            {
                "tag": "人物",
                "primary_type": "people",
                "face_count": 3,
                "asset_count": 4,
                "kind": "video",
                "representative_kind": "video",
                "representative_duration_seconds": 5.0,
                "blur_score": 3200.0,
            },
            {
                "tag": "寺社",
                "primary_type": "temple",
                "asset_count": 3,
                "blur_score": 2800.0,
                "ocr_text": "temple",
            },
        ],
    )

    records = candidates.build_scene_records(scene_df, reps_df, tmp_path)
    food_scene = records[0]
    people_scene = records[1]

    assert food_scene["primary_type"] == "landscape"
    assert food_scene["representative_tag"] == "風景"
    assert food_scene["modifiers"] == ["food"]
    assert food_scene["selection_score"] < people_scene["selection_score"]
    assert food_scene["selection_rank"] > people_scene["selection_rank"]


def test_food_quota_replaces_extra_food_scenes_in_general_trip(tmp_path: Path) -> None:
    scene_specs = [
        {
            "tag": "風景",
            "primary_type": "landscape",
            "modifiers": ["food"],
            "food_confidence": 0.95,
            "asset_count": 6,
            "kind": "video",
            "representative_kind": "video",
            "representative_duration_seconds": 4.5,
            "blur_score": 3900.0,
        },
        {
            "tag": "風景",
            "primary_type": "landscape",
            "modifiers": ["food"],
            "food_confidence": 0.91,
            "asset_count": 5,
            "kind": "video",
            "representative_kind": "video",
            "representative_duration_seconds": 5.0,
            "blur_score": 3600.0,
        },
        {
            "tag": "人物",
            "primary_type": "people",
            "face_count": 2,
            "asset_count": 4,
            "kind": "video",
            "representative_kind": "video",
            "representative_duration_seconds": 5.0,
            "blur_score": 3100.0,
        },
        {
            "tag": "寺社",
            "primary_type": "temple",
            "asset_count": 3,
            "blur_score": 2800.0,
            "ocr_text": "temple",
        },
    ]
    scene_specs.extend(
        {
            "tag": "風景",
            "primary_type": "landscape",
            "asset_count": 1,
            "blur_score": 1200.0,
        }
        for _ in range(6)
    )

    scene_df, reps_df = build_synthetic_inputs(tmp_path, scene_specs)
    records = candidates.build_scene_records(scene_df, reps_df, tmp_path)
    selected_records = [record for record in records if record["selected_for_edit"]]
    selected_food_records = [record for record in selected_records if "food" in record["modifiers"]]

    assert len(selected_records) == 3
    assert len(selected_food_records) == 1
    assert {record["scene_id"] for record in selected_records} >= {3, 4}
    assert any("quota_promoted" in record["selection_reasons"] for record in selected_records)
    assert any("quota_replaced" in record["selection_reasons"] for record in records if "food" in record["modifiers"])


def test_consecutive_food_scenes_receive_temporal_penalty(tmp_path: Path) -> None:
    scene_df, reps_df = build_synthetic_inputs(
        tmp_path,
        [
            {"tag": "風景", "primary_type": "landscape", "asset_count": 2, "blur_score": 2200.0},
            {
                "tag": "風景",
                "primary_type": "landscape",
                "modifiers": ["food"],
                "food_confidence": 0.84,
                "asset_count": 3,
                "kind": "video",
                "representative_kind": "video",
                "representative_duration_seconds": 5.0,
                "blur_score": 2900.0,
            },
            {
                "tag": "風景",
                "primary_type": "landscape",
                "modifiers": ["food"],
                "food_confidence": 0.84,
                "asset_count": 3,
                "kind": "video",
                "representative_kind": "video",
                "representative_duration_seconds": 5.0,
                "blur_score": 2900.0,
            },
            {"tag": "人物", "primary_type": "people", "face_count": 2, "asset_count": 3, "blur_score": 3000.0},
            {"tag": "寺社", "primary_type": "temple", "asset_count": 2, "blur_score": 2400.0, "ocr_text": "temple"},
        ],
    )

    records = candidates.build_scene_records(scene_df, reps_df, tmp_path)
    first_food = records[1]
    second_food = records[2]

    assert first_food["selection_score"] > second_food["selection_score"]
    assert "food_penalty:consecutive" in second_food["selection_reasons"]


def test_trip_type_adapts_food_quota_and_duration(tmp_path: Path) -> None:
    def build_trip_specs(food_scene_count: int) -> list[dict[str, object]]:
        specs: list[dict[str, object]] = []
        for index in range(20):
            if index < food_scene_count:
                specs.append(
                    {
                        "tag": "風景",
                        "primary_type": "landscape",
                        "modifiers": ["food"],
                        "food_confidence": 0.9,
                        "asset_count": 6,
                        "kind": "video",
                        "representative_kind": "video",
                        "representative_duration_seconds": 5.0,
                        "blur_score": 3800.0,
                    }
                )
            elif index == 19:
                specs.append(
                    {
                        "tag": "風景",
                        "primary_type": "landscape",
                        "asset_count": 6,
                        "kind": "video",
                        "representative_kind": "video",
                        "representative_duration_seconds": 5.0,
                        "blur_score": 3800.0,
                    }
                )
            elif index == 10:
                specs.append(
                    {
                        "tag": "人物",
                        "primary_type": "people",
                        "face_count": 2,
                        "asset_count": 4,
                        "kind": "video",
                        "representative_kind": "video",
                        "representative_duration_seconds": 5.0,
                        "blur_score": 3000.0,
                    }
                )
            else:
                specs.append(
                    {
                        "tag": "風景",
                        "primary_type": "landscape",
                        "asset_count": 1,
                        "blur_score": 1400.0,
                    }
                )
        return specs

    gourmet_scene_df, gourmet_reps_df = build_synthetic_inputs(tmp_path / "gourmet", build_trip_specs(9))
    general_scene_df, general_reps_df = build_synthetic_inputs(tmp_path / "general", build_trip_specs(4))

    gourmet_records = candidates.build_scene_records(gourmet_scene_df, gourmet_reps_df, tmp_path)
    general_records = candidates.build_scene_records(general_scene_df, general_reps_df, tmp_path)

    gourmet_summary = candidates.build_payload_summary(gourmet_records)
    general_summary = candidates.build_payload_summary(general_records)

    assert gourmet_summary["trip_type"] == "gourmet"
    assert general_summary["trip_type"] == "general"
    assert gourmet_summary["selected_food_scene_count"] > general_summary["selected_food_scene_count"]

    gourmet_structure = structure.build_structure(meanings.build_meanings({"generated_at": "2026-03-29T11:00:00+09:00", "scenes": gourmet_records}))
    general_structure = structure.build_structure(meanings.build_meanings({"generated_at": "2026-03-29T11:00:00+09:00", "scenes": general_records}))

    gourmet_first_food = next(item for item in gourmet_structure["edit_sequence"] if item["scene_id"] == 1)
    general_first_food = next(item for item in general_structure["edit_sequence"] if item["scene_id"] == 1)
    general_last_non_food = next(item for item in general_structure["edit_sequence"] if item["scene_id"] == 20)

    assert gourmet_first_food["planned_duration_seconds"] > general_first_food["planned_duration_seconds"]
    assert general_first_food["planned_duration_seconds"] < general_last_non_food["planned_duration_seconds"]


def test_editor_brief_updates_sequence_and_subtitle_density(media_info_csv: Path, sample_root: Path) -> None:
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

    brief = dict(draft_plan["editor_brief"])
    brief.update(
        {
            "audience": "家族",
            "tone": "cinematic",
            "pacing": "calm",
            "caption_density": "minimal",
            "focus_scene_ids": [3],
            "focus_keywords": [],
            "downplay_scene_ids": [2],
            "downplay_keywords": [],
            "must_mention": ["浅草寺"],
            "ending_feel": "静かな余韻",
        }
    )

    updated = llm_plan.apply_editor_brief(draft_plan, brief)

    assert updated["title"] == "浅草寺の旅"
    assert updated["logline"].startswith("家族に向けて、")
    assert updated["editor_brief"]["resolved_focus_scene_ids"] == [3]
    assert updated["editor_brief"]["resolved_downplay_scene_ids"] == [2]
    assert updated["edit_sequence"][2]["planned_duration_seconds"] > updated["edit_sequence"][2]["base_planned_duration_seconds"]
    assert updated["edit_sequence"][1]["planned_duration_seconds"] < updated["edit_sequence"][1]["base_planned_duration_seconds"]
    assert [item["scene_id"] for item in updated["subtitle_plan"]["items"]] == [1, 3]
    assert updated["subtitle_plan"]["items"][1]["text"] == "静かな余韻を残して締めくくる"


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

    def fake_normalize_clip(input_path: Path, output_path: Path, duration: float, overlays: list[dict[str, object]] | None = None) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_count = len(overlays or [])
        output_path.write_bytes(f"{input_path.name}:{duration:.2f}:{overlay_count}".encode("utf-8"))

    def fake_render_title_card(card: dict[str, object], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(f"{card['title']}:{card.get('subtitle')}".encode("utf-8"))

    monkeypatch.setattr(render, "normalize_clip", fake_normalize_clip)
    monkeypatch.setattr(render, "render_title_card", fake_render_title_card)

    clips = render.build_clip_list({"plan": draft_plan}, sample_root, tmp_path)

    assert len(materialized_media_files) == 6
    assert len(clips) == 6
    assert [clip["scene_id"] for clip in clips] == [1, 1, 1, 2, 2, 3]
    assert all(clip["clip_kind"] == "media" for clip in clips)
    assert clips[0]["transition_hint"] == "fade_in"
    assert clips[0]["preview_index"] == 1
    assert [overlay["style"] for overlay in clips[0]["overlays"]] == ["title", "label", "subtitle"]
    assert clips[1]["preview_index"] == 2
    assert clips[1]["overlays"] == []
    assert clips[2]["preview_index"] == 3
    assert clips[3]["transition_hint"] == "soft_cut"
    assert [overlay["style"] for overlay in clips[3]["overlays"]] == ["title", "subtitle"]
    assert [overlay["style"] for overlay in clips[5]["overlays"]] == ["title", "subtitle"]
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

    def fake_normalize_clip(input_path: Path, output_path: Path, duration: float, overlays: list[dict[str, object]] | None = None) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_count = len(overlays or [])
        output_path.write_bytes(f"{input_path.name}:{duration:.2f}:{overlay_count}".encode("utf-8"))

    def fake_render_title_card(card: dict[str, object], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(f"{card['title']}:{card.get('subtitle')}".encode("utf-8"))

    monkeypatch.setattr(render, "normalize_clip", fake_normalize_clip)
    monkeypatch.setattr(render, "render_title_card", fake_render_title_card)

    clips = render.build_clip_list({"plan": draft_plan}, sample_root, tmp_path)

    assert len(materialized_media_files) == 6
    assert clips[0]["clip_kind"] == "media"
    assert clips[0]["scene_id"] == 1
    assert clips[0]["source_path"].endswith("overview-02.jpg")
    assert clips[0]["preview_index"] == 1
    assert [overlay["style"] for overlay in clips[0]["overlays"]] == ["title", "label", "subtitle"]


def test_render_overlay_image_backend_limits_output_duration(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "clip.mp4"
    output_path = tmp_path / "clip.normalized.mp4"
    seen_commands: list[list[str]] = []

    class FakeResult:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = ""
            self.stderr = ""

    def fake_subprocess_run(cmd: list[str], capture_output: bool, text: bool, check: bool) -> FakeResult:
        seen_commands.append(cmd)
        output_path.write_bytes(b"ok")
        return FakeResult()

    monkeypatch.setattr(render, "overlay_backend", lambda: "overlay_image")
    monkeypatch.setattr(
        render,
        "build_overlay_image_assets",
        lambda overlays, output_path_arg, duration: [
            {
                "path": output_path_arg.with_name("overlay.png"),
                "start_seconds": 0.0,
                "end_seconds": duration,
            }
        ],
    )
    monkeypatch.setattr(render.subprocess, "run", fake_subprocess_run)

    render.normalize_clip(
        input_path,
        output_path,
        1.5,
        overlays=[{"text": "旅のはじまり", "style": "title", "duration_seconds": 1.5}],
    )

    ffmpeg_cmd = seen_commands[-1]
    assert ffmpeg_cmd[:2] == ["ffmpeg", "-y"]
    assert "-filter_complex" in ffmpeg_cmd
    assert "-t" in ffmpeg_cmd
    assert ffmpeg_cmd[ffmpeg_cmd.index("-t") + 1] == "1.50"
    assert ffmpeg_cmd[-1] == str(output_path)


def test_interactive_overrides_update_overlay_titles_and_subtitles(media_info_csv: Path, sample_root: Path) -> None:
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

    responses = io.StringIO(
        "\n".join(
            [
                "家族",
                "余韻重視",
                "ゆっくり",
                "3,寺社",
                "2",
                "浅草寺",
                "静かな余韻",
                "控えめ",
                "東京さんぽ",
                "",
                "2024年5月1日 東京",
                "-",
                "",
                "",
                "朝の景色からスタート",
                "-",
                "",
            ]
        )
        + "\n"
    )
    output = io.StringIO()

    updated = llm_plan.apply_interactive_overrides(draft_plan, input_stream=responses, output_stream=output)

    assert updated["title"] == "東京さんぽ"
    assert updated["editor_brief"]["audience"] == "家族"
    assert updated["editor_brief"]["tone"] == "cinematic"
    assert updated["editor_brief"]["pacing"] == "calm"
    assert updated["editor_brief"]["caption_density"] == "minimal"
    assert updated["editor_brief"]["resolved_focus_scene_ids"] == [3]
    assert updated["editor_brief"]["resolved_downplay_scene_ids"] == [2]
    assert updated["title_cards"][0]["title"] == "2024年5月1日 東京"
    assert updated["title_cards"][0]["subtitle"] is None
    assert updated["title_cards"][0]["presentation"] == "overlay"
    assert [item["scene_id"] for item in updated["subtitle_plan"]["items"]] == [1, 3]
    assert updated["subtitle_plan"]["items"][0]["text"] == "朝の景色からスタート"
    assert updated["subtitle_plan"]["items"][0]["origin"] == "interactive"
    assert updated["subtitle_plan"]["items"][1]["origin"] == "auto"
    assert updated["edit_sequence"][2]["planned_duration_seconds"] > updated["edit_sequence"][2]["base_planned_duration_seconds"]
    assert updated["edit_sequence"][1]["planned_duration_seconds"] < updated["edit_sequence"][1]["base_planned_duration_seconds"]
