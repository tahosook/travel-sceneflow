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
        "tag": "風景",
        "primary_type": "landscape",
        "modifiers": [],
        "food_confidence": 0.0,
        "caption": "風景の広がりが伝わる一枚です。",
    },
    2: {
        "ocr_text": "boarding platform",
        "face_count_raw": 0,
        "face_count_filtered": 0,
        "face_count": 0,
        "tag": "移動",
        "primary_type": "transit",
        "modifiers": [],
        "food_confidence": 0.0,
        "caption": "移動中の様子を捉えた一枚です。",
    },
    3: {
        "ocr_text": "浅草寺 temple",
        "face_count_raw": 0,
        "face_count_filtered": 0,
        "face_count": 0,
        "tag": "寺社",
        "primary_type": "temple",
        "modifiers": [],
        "food_confidence": 0.0,
        "caption": "寺社の雰囲気が伝わる一枚です。",
    },
}


def build_tagged_representatives(representatives_df: pd.DataFrame) -> pd.DataFrame:
    tagged = representatives_df.copy()
    scene_ids = pd.to_numeric(tagged["scene_id"], errors="raise").astype(int)

    for column in ["ocr_text", "face_count_raw", "face_count_filtered", "face_count", "tag", "primary_type", "modifiers", "food_confidence", "caption"]:
        tagged[column] = scene_ids.map(lambda scene_id: TAG_DATA.get(scene_id, {}).get(column))

    if "primary_type" in tagged.columns:
        tagged["primary_type"] = tagged["primary_type"].fillna(scene_ids.map(lambda scene_id: PRIMARY_TYPE_BY_TAG.get(TAG_DATA[scene_id]["tag"], "landscape")))
    if "modifiers" in tagged.columns:
        tagged["modifiers"] = tagged["modifiers"].map(lambda value: value if isinstance(value, list) else [])
    if "food_confidence" in tagged.columns:
        tagged["food_confidence"] = tagged["food_confidence"].fillna(0.0)

    return tagged
