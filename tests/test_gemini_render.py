from __future__ import annotations

from sceneflow.pipeline import gemini_render


def test_gemini_render_builds_renderable_plan_from_slideshow() -> None:
    structure_payload = {
        "edit_sequence": [
            {"scene_id": 1, "chapter_id": "opening", "planned_duration_seconds": 2.4, "preview_sources": [{"path": "a.jpg", "kind": "image"}], "representative_path": "a.jpg", "representative_kind": "image"},
            {"scene_id": 2, "chapter_id": "body", "planned_duration_seconds": 1.6, "preview_sources": [{"path": "b.jpg", "kind": "image"}], "representative_path": "b.jpg", "representative_kind": "image"},
            {"scene_id": 3, "chapter_id": "body", "planned_duration_seconds": 2.0, "preview_sources": [{"path": "c.mp4", "kind": "video"}], "representative_path": "c.mp4", "representative_kind": "video"},
            {"scene_id": 4, "chapter_id": "closing", "planned_duration_seconds": 2.2, "preview_sources": [{"path": "d.jpg", "kind": "image"}], "representative_path": "d.jpg", "representative_kind": "image"},
        ]
    }
    slideshow_payload = {
        "generated_at": "2026-04-05T13:00:00+09:00",
        "source": "outputs/run/structure/edit_structure.json",
        "plan": {
            "title": "Gemini 旅",
            "logline": "Gemini 構成案",
            "chapter_list": [
                {"chapter_id": "opening", "title": "旅のはじまり", "purpose": "導入", "scene_ids": [1]},
                {"chapter_id": "body", "title": "見どころ", "purpose": "強調", "scene_ids": [3]},
                {"chapter_id": "closing", "title": "余韻", "purpose": "締め", "scene_ids": [4]},
            ],
            "scene_directions": [
                {"scene_ids": [1], "chapter_id": "opening", "recommended_duration_seconds": 3.0, "direction": "導入", "subtitle": "はじまり"},
                {"scene_ids": [3], "chapter_id": "body", "recommended_duration_seconds": 4.2, "direction": "見どころ", "subtitle": "ハイライト"},
                {"scene_ids": [4], "chapter_id": "closing", "recommended_duration_seconds": 3.4, "direction": "締め", "subtitle": "おわり"},
            ],
            "subtitle_plan": {
                "enabled": True,
                "style": "simple",
                "items": [
                    {"scene_ids": [1], "text": "はじまり"},
                    {"scene_ids": [3], "text": "ハイライト"},
                    {"scene_ids": [4], "text": "おわり"},
                ],
            },
            "ending_note": "余韻",
        },
    }

    bridged = gemini_render.build_renderable_plan(slideshow_payload, structure_payload)
    plan = bridged["plan"]

    assert plan["title"] == "Gemini 旅"
    assert [item["scene_id"] for item in plan["edit_sequence"]] == [1, 3, 4]
    assert plan["edit_sequence"][1]["planned_duration_seconds"] == 4.2
    assert [item["scene_id"] for item in plan["subtitle_plan"]["items"]] == [1, 3, 4]
    assert plan["render_guidance"]["preferred_order"] == [1, 3, 4]
    assert plan["render_guidance"]["source"] == "gemini_slideshow_plan"
