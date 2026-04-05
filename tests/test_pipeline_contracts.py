from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd

from sceneflow.pipeline import candidates, gemini_plan, llm_plan, meanings, render, representatives, sceneify, structure, tagging
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
                "clip_primary_type": spec.get("clip_primary_type"),
                "clip_primary_score": float(spec.get("clip_primary_score", 0.0)),
                "clip_top_labels": list(spec.get("clip_top_labels", [])),
                "tag": tag,
                "primary_type": str(spec.get("primary_type", "landscape")),
                "modifiers": list(spec.get("modifiers", [])),
                "food_confidence": float(spec.get("food_confidence", 0.0)),
                "caption": str(spec.get("caption", CAPTION_BY_TAG.get(tag, "旅の一枚です。"))),
                "classification_source": str(spec.get("classification_source", "clip" if spec.get("clip_primary_type") else "time")),
                "classification_confidence": float(spec.get("classification_confidence", spec.get("clip_primary_score", 0.35))),
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
    assert records[2]["clip_hints"][0]["label"] == "temple or shrine"
    assert "CLIP:temple or shrine" in records[2]["semantic_summary"]
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
    assert meanings_payload["scenes"][2]["semantic_summary"] == records[2]["semantic_summary"]
    assert structure_payload["edit_sequence"][2]["clip_hints"][0]["label"] == "temple or shrine"
    assert structure_payload["edit_sequence"][1]["classification_source"] == "ocr"

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

    def fake_analyze_scene_food(group, rep, root, image_index, cache_dir):
        scene_id = int(group.iloc[0]["scene_id"])
        if scene_id == 1:
            return {
                "food_confidence": 0.92,
                "food_sample_count": 4,
                "food_evidence_count": 2,
                "food_sample_paths": ["scene-1-0.jpg", "scene-1-1.jpg"],
                "has_food_modifier": True,
            }
        return {
            "food_confidence": 0.0,
            "food_sample_count": 3,
            "food_evidence_count": 0,
            "food_sample_paths": [],
            "has_food_modifier": False,
        }

    original = candidates.analyze_scene_food
    candidates.analyze_scene_food = fake_analyze_scene_food
    try:
        records = candidates.build_scene_records(scene_df, reps_df, tmp_path)
    finally:
        candidates.analyze_scene_food = original
    food_scene = records[0]
    people_scene = records[1]

    assert food_scene["primary_type"] == "landscape"
    assert food_scene["representative_tag"] == "風景"
    assert food_scene["modifiers"] == ["food"]
    assert food_scene["food_sample_count"] == 4
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
    def fake_analyze_scene_food(group, rep, root, image_index, cache_dir):
        scene_id = int(group.iloc[0]["scene_id"])
        if scene_id in {1, 2}:
            return {
                "food_confidence": 0.9,
                "food_sample_count": 5,
                "food_evidence_count": 2,
                "food_sample_paths": [f"scene-{scene_id}-food.jpg"],
                "has_food_modifier": True,
            }
        return {
            "food_confidence": 0.0,
            "food_sample_count": 3,
            "food_evidence_count": 0,
            "food_sample_paths": [],
            "has_food_modifier": False,
        }

    original = candidates.analyze_scene_food
    candidates.analyze_scene_food = fake_analyze_scene_food
    try:
        records = candidates.build_scene_records(scene_df, reps_df, tmp_path)
    finally:
        candidates.analyze_scene_food = original
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

    def fake_analyze_scene_food(group, rep, root, image_index, cache_dir):
        scene_id = int(group.iloc[0]["scene_id"])
        if scene_id in {2, 3}:
            return {
                "food_confidence": 0.84,
                "food_sample_count": 4,
                "food_evidence_count": 2,
                "food_sample_paths": [f"scene-{scene_id}-food.jpg"],
                "has_food_modifier": True,
            }
        return {
            "food_confidence": 0.0,
            "food_sample_count": 3,
            "food_evidence_count": 0,
            "food_sample_paths": [],
            "has_food_modifier": False,
        }

    original = candidates.analyze_scene_food
    candidates.analyze_scene_food = fake_analyze_scene_food
    try:
        records = candidates.build_scene_records(scene_df, reps_df, tmp_path)
    finally:
        candidates.analyze_scene_food = original
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

    def fake_analyze_scene_food(group, rep, root, image_index, cache_dir):
        scene_id = int(group.iloc[0]["scene_id"])
        scene_count = int(group["scene_id"].nunique())
        del scene_count
        is_food = float(rep.get("food_confidence") or 0.0) >= 0.9
        return {
            "food_confidence": 0.9 if is_food else 0.0,
            "food_sample_count": 4 if is_food else 3,
            "food_evidence_count": 2 if is_food else 0,
            "food_sample_paths": [f"scene-{scene_id}-food.jpg"] if is_food else [],
            "has_food_modifier": is_food,
        }

    original = candidates.analyze_scene_food
    candidates.analyze_scene_food = fake_analyze_scene_food
    try:
        gourmet_records = candidates.build_scene_records(gourmet_scene_df, gourmet_reps_df, tmp_path)
        general_records = candidates.build_scene_records(general_scene_df, general_reps_df, tmp_path)
    finally:
        candidates.analyze_scene_food = original

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


def test_scene_food_sampling_caps_and_deduplicates_rows(tmp_path: Path) -> None:
    scene_df, _ = build_synthetic_inputs(
        tmp_path,
        [
            {
                "tag": "風景",
                "primary_type": "landscape",
                "asset_count": 8,
                "blur_score": 1500.0,
            }
        ],
    )
    scene_df.loc[1, "path"] = scene_df.loc[0, "path"]
    scene_df.loc[2, "path"] = scene_df.loc[0, "path"]
    group = scene_df.copy()
    group["final_timestamp_dt"] = group["final_timestamp"].map(pd.Timestamp)

    rows = candidates.choose_scene_food_sample_rows(group, None)
    keys = [candidates.normalized_asset_key(row.get("path")) for row in rows]

    assert len(rows) <= candidates.FOOD_SCENE_SAMPLE_LIMIT
    assert len(keys) == len(set(keys))


def test_scene_food_aggregation_uses_top_two_without_dilution() -> None:
    assert candidates.aggregate_food_samples([0.95, 0.7, 0.05, 0.0]) == 0.862
    assert candidates.aggregate_food_samples([0.4, 0.35, 0.3]) < candidates.FOOD_SCENE_THRESHOLD


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


def test_classify_scene_attributes_prefers_ocr_face_clip_and_time() -> None:
    row = pd.Series({"representative_final_timestamp": "2024-05-01T20:00:00+09:00"})
    confident_clip = tagging.ClipPrediction(
        primary_type="temple",
        tag="寺社",
        primary_score=0.76,
        top_labels=[{"label": "temple or shrine", "score": 0.76}],
        modifier_scores={"food": 0.08},
    )
    weak_clip = tagging.ClipPrediction(
        primary_type="temple",
        tag="寺社",
        primary_score=0.12,
        top_labels=[{"label": "temple or shrine", "score": 0.12}],
        modifier_scores={},
    )

    tag, primary_type, modifiers, source, _, confidence = tagging.classify_scene_attributes(
        row,
        "浅草寺 temple",
        0,
        0.0,
        clip_prediction=confident_clip,
    )
    assert (tag, primary_type, modifiers, source) == ("寺社", "temple", [], "ocr")
    assert confidence > 0.8

    tag, primary_type, modifiers, source, _, confidence = tagging.classify_scene_attributes(
        row,
        "",
        2,
        0.0,
        clip_prediction=confident_clip,
    )
    assert (tag, primary_type, modifiers, source) == ("集合写真", "group", [], "face")
    assert confidence > 0.7

    tag, primary_type, modifiers, source, _, confidence = tagging.classify_scene_attributes(
        row,
        "",
        0,
        0.0,
        clip_prediction=confident_clip,
    )
    assert (tag, primary_type, modifiers, source) == ("寺社", "temple", [], "clip")
    assert confidence == 0.76

    tag, primary_type, modifiers, source, _, confidence = tagging.classify_scene_attributes(
        row,
        "",
        0,
        0.0,
        clip_prediction=weak_clip,
    )
    assert (tag, primary_type, modifiers, source) == ("夜景", "night", [], "time")
    assert confidence == 0.35


def test_classify_clip_predictions_uses_cache(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "analysis.jpg"
    image_path.write_bytes(b"clip-fixture")
    cache_dir = tmp_path / "clip-cache"
    call_count = {"count": 0}

    class FakeClassifier:
        def classify_paths(self, image_paths: list[Path]) -> list[tagging.ClipPrediction]:
            call_count["count"] += 1
            assert image_paths == [image_path]
            return [
                tagging.ClipPrediction(
                    primary_type="temple",
                    tag="寺社",
                    primary_score=0.74,
                    top_labels=[{"label": "temple or shrine", "score": 0.74, "primary_type": "temple", "tag": "寺社"}],
                    modifier_scores={"food": 0.05},
                )
            ]

    monkeypatch.setattr(tagging, "build_clip_classifier", lambda **_: FakeClassifier())

    prepared_rows = [
        {
            "representative_path": str(image_path),
            "_analysis_path": str(image_path),
        }
    ]

    first = tagging.classify_clip_predictions(
        prepared_rows,
        model_name=tagging.DEFAULT_CLIP_MODEL,
        device="cpu",
        batch_size=2,
        cache_dir=cache_dir,
    )
    second = tagging.classify_clip_predictions(
        prepared_rows,
        model_name=tagging.DEFAULT_CLIP_MODEL,
        device="cpu",
        batch_size=2,
        cache_dir=cache_dir,
    )

    assert call_count["count"] == 1
    assert first[0].tag == "寺社"
    assert second[0].primary_score == 0.74
    assert any(cache_dir.iterdir())


def test_gemini_prompt_compresses_low_priority_scenes_and_omits_raw_fields() -> None:
    structure_payload = {
        "summary": {"scene_count": 6, "chapter_count": 3},
        "chapters": [
            {"chapter_id": "opening", "title": "旅のはじまり", "purpose": "導入", "pace": "ゆるやか", "scene_ids": [1], "anchor_scene_ids": [1]},
            {"chapter_id": "body", "title": "旅の流れ", "purpose": "つなぐ", "pace": "一定", "scene_ids": [2, 3, 4, 5], "anchor_scene_ids": [5]},
            {"chapter_id": "closing", "title": "旅の余韻", "purpose": "締め", "pace": "余韻重視", "scene_ids": [6], "anchor_scene_ids": [6]},
        ],
        "edit_sequence": [
            {"scene_id": 1, "chapter_id": "opening", "role": "opening", "priority_band": "high", "planned_duration_seconds": 2.2, "representative_tag": "風景", "semantic_summary": "導入の景色", "selection_reasons": ["opening"], "representative_path": "/secret/a.jpg"},
            {"scene_id": 2, "chapter_id": "body", "role": "support", "priority_band": "low", "planned_duration_seconds": 1.2, "representative_tag": "風景", "semantic_summary": "つなぎ1", "selection_reasons": ["support"], "gps": {"has_gps": True}},
            {"scene_id": 3, "chapter_id": "body", "role": "support", "priority_band": "low", "planned_duration_seconds": 1.1, "representative_tag": "建物", "semantic_summary": "つなぎ2", "selection_reasons": ["support"]},
            {"scene_id": 4, "chapter_id": "body", "role": "support", "priority_band": "low", "planned_duration_seconds": 1.0, "representative_tag": "風景", "semantic_summary": "つなぎ3", "selection_reasons": ["support"]},
            {"scene_id": 5, "chapter_id": "body", "role": "highlight", "priority_band": "high", "planned_duration_seconds": 2.8, "representative_tag": "寺社", "semantic_summary": "見どころ", "selection_reasons": ["highlight"]},
            {"scene_id": 6, "chapter_id": "closing", "role": "closing", "priority_band": "medium", "planned_duration_seconds": 2.0, "representative_tag": "夜景", "semantic_summary": "締め", "selection_reasons": ["closing"]},
        ],
    }

    class FakeModels:
        def count_tokens(self, *, model: str, contents: str):
            del model
            token_count = 3200 if "grouped_scene_count" in contents else 9000
            return type("CountTokensResponse", (), {"total_tokens": token_count})()

    fake_client = type("FakeClient", (), {"models": FakeModels()})()
    prompt_text, prompt_payload, token_count = gemini_plan.fit_prompt_to_budget(
        fake_client,
        structure=structure_payload,
        model=gemini_plan.DEFAULT_MODEL,
        token_budget=6000,
    )

    assert token_count == 3200
    assert len(prompt_payload["scene_digest"]) < len(structure_payload["edit_sequence"])
    assert "representative_path" not in prompt_text
    assert '"gps"' not in prompt_text


def test_generate_slideshow_plan_uses_structured_output() -> None:
    structure_payload = {
        "summary": {"scene_count": 2, "chapter_count": 2},
        "chapters": [
            {"chapter_id": "opening", "title": "旅のはじまり", "purpose": "導入", "pace": "ゆるやか", "scene_ids": [1], "anchor_scene_ids": [1]},
            {"chapter_id": "closing", "title": "旅の余韻", "purpose": "締め", "pace": "余韻重視", "scene_ids": [2], "anchor_scene_ids": [2]},
        ],
        "edit_sequence": [
            {"scene_id": 1, "chapter_id": "opening", "role": "opening", "priority_band": "high", "planned_duration_seconds": 2.0, "representative_tag": "風景", "semantic_summary": "導入", "selection_reasons": ["opening"]},
            {"scene_id": 2, "chapter_id": "closing", "role": "closing", "priority_band": "medium", "planned_duration_seconds": 2.1, "representative_tag": "夜景", "semantic_summary": "締め", "selection_reasons": ["closing"]},
        ],
    }
    plan_payload = {
        "title": "東京の旅",
        "logline": "導入から余韻まで自然につなぐ。",
        "chapter_list": [{"chapter_id": "opening", "title": "旅のはじまり", "purpose": "導入", "scene_ids": [1], "editing_note": "ゆっくり入る"}],
        "scene_directions": [{"scene_ids": [1], "chapter_id": "opening", "emphasis": "high", "recommended_duration_seconds": 2.0, "direction": "景色から始める"}],
        "subtitle_plan": {"enabled": True, "style": "overlay", "items": [{"scene_ids": [1], "text": "旅のはじまり"}]},
        "ending_note": "静かな余韻で終える",
    }

    class FakeUsage:
        def model_dump(self, exclude_none: bool = True) -> dict[str, object]:
            del exclude_none
            return {"prompt_token_count": 111, "total_token_count": 222}

    class FakeModels:
        def count_tokens(self, *, model: str, contents: str):
            del model, contents
            return type("CountTokensResponse", (), {"total_tokens": 111})()

        def generate_content(self, *, model: str, contents: str, config: object):
            del model, contents, config
            return type(
                "GenerateContentResponse",
                (),
                {
                    "parsed": plan_payload,
                    "text": json.dumps(plan_payload, ensure_ascii=False),
                    "usage_metadata": FakeUsage(),
                },
            )()

    fake_client = type("FakeClient", (), {"models": FakeModels()})()
    plan, prompt_text, token_usage = gemini_plan.generate_slideshow_plan(
        structure_payload,
        client=fake_client,
        model=gemini_plan.DEFAULT_MODEL,
        token_budget=6000,
    )

    assert plan["title"] == "東京の旅"
    assert token_usage["request_token_count"] == 111
    assert token_usage["total_token_count"] == 222
    assert "Gemini Slideshow Prompt" in prompt_text


def test_tagging_to_gemini_smoke(monkeypatch, media_info_csv: Path, sample_root: Path, materialized_media_files: list[Path], tmp_path: Path) -> None:
    del materialized_media_files
    scene_df = sceneify.process_csv(
        input_path=media_info_csv,
        output_path=Path("unused.csv"),
        tz_name="Asia/Tokyo",
        gap_minutes=15,
        gps_gap_meters=100.0,
        max_scene_minutes=60,
    )
    reps_df = representatives.build_representatives(scene_df)

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    def fake_analysis_path(path_value: object) -> Path:
        path = Path(str(path_value))
        analysis_path = analysis_dir / f"{path.stem}.jpg"
        analysis_path.write_bytes(b"analysis")
        return analysis_path

    def fake_ocr_text_for_asset(path_value, kind_value, image_index, cache_dir=None):
        del kind_value, image_index, cache_dir
        path = Path(str(path_value))
        if "temple" in path.name:
            return "浅草寺 temple", fake_analysis_path(path), [], 6
        if "transit" in path.name:
            return "boarding platform", fake_analysis_path(path), [], 4
        return "", fake_analysis_path(path), [], 0

    def fake_detect_face_counts(image_path: Path):
        del image_path
        return (0, 0)

    def fake_detect_food_score(image_path: Path):
        del image_path
        return 0

    class FakeClassifier:
        def classify_paths(self, image_paths: list[Path]) -> list[tagging.ClipPrediction]:
            predictions: list[tagging.ClipPrediction] = []
            for path in image_paths:
                if "temple" in path.name:
                    predictions.append(
                        tagging.ClipPrediction(
                            primary_type="temple",
                            tag="寺社",
                            primary_score=0.8,
                            top_labels=[{"label": "temple or shrine", "score": 0.8, "primary_type": "temple", "tag": "寺社"}],
                            modifier_scores={},
                        )
                    )
                elif "transit" in path.name:
                    predictions.append(
                        tagging.ClipPrediction(
                            primary_type="transit",
                            tag="移動",
                            primary_score=0.41,
                            top_labels=[{"label": "travel transit", "score": 0.41, "primary_type": "transit", "tag": "移動"}],
                            modifier_scores={},
                        )
                    )
                else:
                    predictions.append(
                        tagging.ClipPrediction(
                            primary_type="landscape",
                            tag="風景",
                            primary_score=0.45,
                            top_labels=[{"label": "landscape or city view", "score": 0.45, "primary_type": "landscape", "tag": "風景"}],
                            modifier_scores={},
                        )
                    )
            return predictions

    monkeypatch.setattr(tagging, "ocr_text_for_asset", fake_ocr_text_for_asset)
    monkeypatch.setattr(tagging, "detect_face_counts", fake_detect_face_counts)
    monkeypatch.setattr(tagging, "detect_food_score", fake_detect_food_score)
    monkeypatch.setattr(tagging, "build_clip_classifier", lambda **_: FakeClassifier())

    tagged_reps = tagging.annotate_representatives(
        reps_df,
        sample_root,
        workers=1,
        cache_dir=tmp_path / "ocr-cache",
        clip_device="cpu",
        clip_cache_dir=tmp_path / "clip-cache",
    )
    records = candidates.build_scene_records(scene_df, tagged_reps, sample_root)
    meanings_payload = meanings.build_meanings({"generated_at": "2026-03-29T11:00:00+09:00", "scenes": records})
    structure_payload = structure.build_structure(meanings_payload)

    structure_path = tmp_path / "run" / "structure" / "edit_structure.json"
    structure_path.parent.mkdir(parents=True, exist_ok=True)
    structure_path.write_text(json.dumps(structure_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    plan_payload = {
        "title": "東京スライドショー",
        "logline": "scene の流れを自然につなぐ。",
        "chapter_list": [{"chapter_id": "opening", "title": "旅のはじまり", "purpose": "導入", "scene_ids": [1], "editing_note": "景色から始める"}],
        "scene_directions": [{"scene_ids": [1], "chapter_id": "opening", "emphasis": "high", "recommended_duration_seconds": 2.0, "direction": "overview から導入する"}],
        "subtitle_plan": {"enabled": True, "style": "overlay", "items": [{"scene_ids": [1], "text": "旅のはじまり"}]},
        "ending_note": "余韻を残して締める",
    }

    class FakeUsage:
        def model_dump(self, exclude_none: bool = True) -> dict[str, object]:
            del exclude_none
            return {"prompt_token_count": 222, "total_token_count": 333}

    class FakeModels:
        def count_tokens(self, *, model: str, contents: str):
            del model, contents
            return type("CountTokensResponse", (), {"total_tokens": 222})()

        def generate_content(self, *, model: str, contents: str, config: object):
            del model, contents, config
            return type(
                "GenerateContentResponse",
                (),
                {
                    "parsed": plan_payload,
                    "text": json.dumps(plan_payload, ensure_ascii=False),
                    "usage_metadata": FakeUsage(),
                },
            )()

    monkeypatch.setattr(gemini_plan, "build_gemini_client", lambda api_key=None: type("FakeClient", (), {"models": FakeModels()})())
    monkeypatch.setattr(sys, "argv", ["gemini", "--input", str(structure_path), "--output", str(tmp_path / "run" / "gemini" / "slideshow_plan.json")])

    assert gemini_plan.main() == 0

    output_path = tmp_path / "run" / "gemini" / "slideshow_plan.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert tagged_reps["clip_primary_type"].notna().all()
    assert payload["plan"]["title"] == "東京スライドショー"
    assert payload["token_usage"]["request_token_count"] == 222
