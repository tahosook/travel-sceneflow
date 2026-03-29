from __future__ import annotations

import pandas as pd


TAG_DATA = {
    1: {
        "ocr_text": "",
        "face_count_raw": 0,
        "face_count_filtered": 0,
        "face_count": 0,
        "tag": "風景",
        "caption": "風景の広がりが伝わる一枚です。",
    },
    2: {
        "ocr_text": "boarding platform",
        "face_count_raw": 0,
        "face_count_filtered": 0,
        "face_count": 0,
        "tag": "移動",
        "caption": "移動中の様子を捉えた一枚です。",
    },
    3: {
        "ocr_text": "浅草寺 temple",
        "face_count_raw": 0,
        "face_count_filtered": 0,
        "face_count": 0,
        "tag": "寺社",
        "caption": "寺社の雰囲気が伝わる一枚です。",
    },
}


def build_tagged_representatives(representatives_df: pd.DataFrame) -> pd.DataFrame:
    tagged = representatives_df.copy()
    scene_ids = pd.to_numeric(tagged["scene_id"], errors="raise").astype(int)

    for column in ["ocr_text", "face_count_raw", "face_count_filtered", "face_count", "tag", "caption"]:
        tagged[column] = scene_ids.map(lambda scene_id: TAG_DATA[scene_id][column])

    return tagged
