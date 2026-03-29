# Photos-ooarai workflow

このリポジトリは、写真・動画を走査して scene 単位にまとめ、後で編集依頼しやすい JSON を作るための作業手順を含みます。  
動画として見やすい流れを優先し、OCR / 顔検出は補助情報として扱います。

この文書は元ワークフローを整理した詳細メモです。  
現行リポジトリでは個別 step 実行に `scripts/run_step.py`、一括実行に `scripts/run_pipeline.py` を使います。

## 目的

- 画像・動画のメタデータを集める
- scene を作る
- scene ごとの代表素材を決める
- 代表素材に OCR / 顔検出 / 補助タグ付けを行う
- scene ごとの編集候補 JSON を出力する

## 前提

- `uv` が使えること
- `tesseract` が使えること
- `ffmpeg` と `exiftool` が使えること
- `pandas` と `opencv-python` は `uv` 環境に入っていること
- 出力は `outputs/<run-id>/<step>/` にまとまるようになっています
- 各出力には `*.meta.json` の sidecar が付き、生成時刻や件数を残します
- 実行中は進捗と ETA を標準エラーに表示します

## 実行手順

まず、1 回の実行で使う出力先を決めます。

```bash
RUN_DIR="outputs/$(date +%Y%m%d-%H%M%S)"
```

### 1. 素材を走査してメタデータを作る

まず、フォルダ内の画像・動画を走査して基本メタデータを CSV にします。

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -B scripts/run_step.py scan --root . --run-dir "$RUN_DIR"
```

生成物:

- `outputs/<run-id>/scan/media_info.csv`
- `outputs/<run-id>/scan/media_info.csv.meta.json`

この CSV には、たとえば次の情報が入ります。

- `captured_at`
- `date_time_original`
- `create_date`
- `gps_latitude`
- `gps_longitude`
- `model`
- `laplacian`
- `brightness`
- `duration_seconds`
- `has_audio`

### 2. final_timestamp を作って scene 分割する

次に、`media_info.csv` を読み込んで、時刻補正と scene 分割を行います。

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -B scripts/run_step.py sceneify --input "$RUN_DIR/scan/media_info.csv" --run-dir "$RUN_DIR"
```

生成物:

- `outputs/<run-id>/sceneify/media_scene.csv`
- `outputs/<run-id>/sceneify/media_scene.csv.meta.json`

この CSV には、各素材に `final_timestamp` と `scene_id` が付きます。

### 3. scene ごとの代表素材を選ぶ

scene ごとに代表素材を 1 件選びます。

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -B scripts/run_step.py representatives --input "$RUN_DIR/sceneify/media_scene.csv" --run-dir "$RUN_DIR"
```

生成物:

- `outputs/<run-id>/representatives/scene_representatives.csv`
- `outputs/<run-id>/representatives/scene_representatives.csv.meta.json`

### 4. 代表素材に OCR / 顔検出 / タグ付けを行う

代表素材に対して OCR と顔検出を行い、簡単なタグと caption を付けます。

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -B scripts/run_step.py tagging --input "$RUN_DIR/representatives/scene_representatives.csv" --root . --run-dir "$RUN_DIR"
```

生成物:

- `outputs/<run-id>/tagging/scene_representatives_tagged.csv`
- `outputs/<run-id>/tagging/scene_representatives_tagged.csv.meta.json`

この CSV には、たとえば次の列が入ります。

- `tag`
- `caption`
- `ocr_text`
- `face_count_raw`
- `face_count_filtered`

### 5. 編集候補 JSON を作る

最後に、scene ごとの情報をまとめて、編集依頼用の JSON を作ります。

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -B scripts/run_step.py candidates --media-scene "$RUN_DIR/sceneify/media_scene.csv" --representatives "$RUN_DIR/tagging/scene_representatives_tagged.csv" --root . --run-dir "$RUN_DIR"
```

生成物:

- `outputs/<run-id>/candidates/scene_edit_candidates.json`
- `outputs/<run-id>/candidates/scene_edit_candidates.json.meta.json`

この JSON は、後で LLM に渡しやすいように scene 単位で整理されています。
`summary` で全体傾向を先に見て、`flow_summary` と `priority_band` で各 scene の読みやすさを把握できます。

### 6. scene の意味づけを作る

scene ごとに、編集で扱いやすい意味を付けます。

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -B scripts/run_step.py meanings --input "$RUN_DIR/candidates/scene_edit_candidates.json" --run-dir "$RUN_DIR"
```

生成物:

- `outputs/<run-id>/meaning/scene_meanings.json`
- `outputs/<run-id>/meaning/scene_meanings.json.meta.json`

### 7. 編集構造を作る

意味づけをもとに、章立てと並び順を作ります。

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -B scripts/run_step.py structure --input "$RUN_DIR/meaning/scene_meanings.json" --run-dir "$RUN_DIR"
```

生成物:

- `outputs/<run-id>/structure/edit_structure.json`
- `outputs/<run-id>/structure/edit_structure.json.meta.json`

### 8. LLM 用の構成案を作る

編集構造をもとに、LLM に渡す入力と、今すぐ使えるドラフト構成案を作ります。

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -B scripts/run_step.py llm --input "$RUN_DIR/structure/edit_structure.json" --run-dir "$RUN_DIR"
```

生成物:

- `outputs/<run-id>/llm/edit_plan.json`
- `outputs/<run-id>/llm/edit_plan.prompt.md`
- `outputs/<run-id>/llm/edit_plan.json.meta.json`

### 9. プレビューをレンダリングする

最後に、構成案をもとに簡易プレビュー動画を作ります。

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -B scripts/run_step.py render --input "$RUN_DIR/llm/edit_plan.json" --root . --run-dir "$RUN_DIR"
```

生成物:

- `outputs/<run-id>/render/preview.mp4`
- `outputs/<run-id>/render/preview.json`
- `outputs/<run-id>/render/preview.mp4.meta.json`

## JSON の中身

各 scene に次のような情報が入ります。

- `scene_id`
- `start_at`
- `end_at`
- `duration_seconds`
- `max_gap_seconds`
- `asset_count`
- `image_count`
- `video_count`
- `gps`
- `importance_score`
- `importance_reasons`
- `tag_strength`
- `meaningful_ocr_tokens`
- `representative`

`assets` の全列挙は含めず、編集に必要な要約だけを残します。

## JSON の読み方

- `summary` で scene 全体の傾向を確認する
- `priority_band` で編集優先度の目安を見る
- `flow_summary` で scene の長さ、素材数、GPS の有無を一目で見る
- `representative` は補助情報つきの代表素材として見る
- `scene_meanings.json` では各 scene の役割を見る
- `edit_structure.json` では章立てと順番を見る
- `edit_plan.json` では LLM 用の構成案を見る
- `preview.mp4` では最終的な流れを確認する

## 推奨される作業順

1. `scripts/run_step.py scan` で `media_info.csv` を作る
2. `scripts/run_step.py sceneify` で `media_scene.csv` を作る
3. `scripts/run_step.py representatives` で代表素材を選ぶ
4. `scripts/run_step.py tagging` でタグを付ける
5. `scripts/run_step.py candidates` で JSON を作る

## 使い分け

- 途中の CSV を見たいときは、`outputs/<run-id>/...` を確認する
- 仕上げの編集依頼をしたいときは `scene_edit_candidates.json` を使う
- ルールを変えたら、前段の CSV から順に再生成する
- 実行が重いときは、標準エラーに出る進捗と ETA を見る
