from __future__ import annotations

import pandas as pd


PRIMARY_TYPE_BY_TAG = {
    "風景": "landscape",
    "移動": "transit",
    "寺社": "temple",
    "人物": "people",
    "集合写真": "group",
    "駅": "station",
    "建物": "building",
    "夜景": "night",
}


TAG_DATA = {
    1: {
        "ocr_text": "",
        "face_count_raw": 0,
        "face_count_filtered": 0,
        "face_count": 0,
        "clip_primary_type": "landscape",
        "clip_primary_score": 0.42,
        "clip_top_labels": [
            {"label": "landscape or city view", "score": 0.42, "primary_type": "landscape", "tag": "風景"},
            {"label": "night view", "score": 0.11, "primary_type": "night", "tag": "夜景"},
        ],
        "tag": "風景",
        "primary_type": "landscape",
        "modifiers": [],
        "food_confidence": 0.0,
        "caption": "風景の広がりが伝わる一枚です。",
        "classification_source": "clip",
        "classification_confidence": 0.42,
    },
    2: {
        "ocr_text": "boarding platform",
        "face_count_raw": 0,
        "face_count_filtered": 0,
        "face_count": 0,
        "clip_primary_type": "transit",
        "clip_primary_score": 0.38,
        "clip_top_labels": [
            {"label": "travel transit", "score": 0.38, "primary_type": "transit", "tag": "移動"},
            {"label": "train station", "score": 0.27, "primary_type": "station", "tag": "駅"},
        ],
        "tag": "移動",
        "primary_type": "transit",
        "modifiers": [],
        "food_confidence": 0.0,
        "caption": "移動中の様子を捉えた一枚です。",
        "classification_source": "ocr",
        "classification_confidence": 0.87,
    },
    3: {
        "ocr_text": "浅草寺 temple",
        "face_count_raw": 0,
        "face_count_filtered": 0,
        "face_count": 0,
        "clip_primary_type": "temple",
        "clip_primary_score": 0.76,
        "clip_top_labels": [
            {"label": "temple or shrine", "score": 0.76, "primary_type": "temple", "tag": "寺社"},
            {"label": "building or landmark", "score": 0.16, "primary_type": "building", "tag": "建物"},
        ],
        "tag": "寺社",
        "primary_type": "temple",
        "modifiers": [],
        "food_confidence": 0.0,
        "caption": "寺社の雰囲気が伝わる一枚です。",
        "classification_source": "ocr",
        "classification_confidence": 0.95,
    },
}


def build_tagged_representatives(representatives_df: pd.DataFrame) -> pd.DataFrame:
    tagged = representatives_df.copy()
    scene_ids = pd.to_numeric(tagged["scene_id"], errors="raise").astype(int)

    for column in [
        "ocr_text",
        "face_count_raw",
        "face_count_filtered",
        "face_count",
        "clip_primary_type",
        "clip_primary_score",
        "clip_top_labels",
        "tag",
        "primary_type",
        "modifiers",
        "food_confidence",
        "caption",
        "classification_source",
        "classification_confidence",
    ]:
        tagged[column] = scene_ids.map(lambda scene_id: TAG_DATA.get(scene_id, {}).get(column))

    if "primary_type" in tagged.columns:
        tagged["primary_type"] = tagged["primary_type"].fillna(scene_ids.map(lambda scene_id: PRIMARY_TYPE_BY_TAG.get(TAG_DATA[scene_id]["tag"], "landscape")))
    if "modifiers" in tagged.columns:
        tagged["modifiers"] = tagged["modifiers"].map(lambda value: value if isinstance(value, list) else [])
    if "food_confidence" in tagged.columns:
        tagged["food_confidence"] = tagged["food_confidence"].fillna(0.0)
    if "classification_confidence" in tagged.columns:
        tagged["classification_confidence"] = tagged["classification_confidence"].fillna(0.35)

    return tagged
