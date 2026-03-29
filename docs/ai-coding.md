# AI Coding Guide

このファイルは、このリポジトリを AI コーディングで進めるときの参照メモです。  
`AGENTS.md` が常設の instruction layer、`PLANS.md` が長時間または複雑な作業向けの計画書です。  
このファイルはそれらを上書きせず、必要なときだけ pipeline map と変更影響を見るために使います。

## まず押さえること

- 北極星は「素材認識の正しさ」ではなく「見やすい travel video flow」です。
- OCR と顔検出は補助情報です。scene の意味や編集判断を支える材料として使います。
- 代表素材は scene の要約であって、scene 全体の真実ではありません。
- 局所精度の改善より、scene continuity、pacing、reviewability を優先します。

## 現在のパイプライン

現在の主な流れは次の通りです。

1. `scan`: 素材を走査してメタデータ CSV を作る
2. `sceneify`: timestamp と GPS を使って scene を作る
3. `representatives`: scene ごとに代表素材を 1 件選ぶ
4. `tagging`: 代表素材へ OCR / 顔検出 / 簡易タグを付ける
5. `candidates`: 編集判断向けの scene JSON に圧縮する
6. `meanings`: scene の役割と edit_action を付ける
7. `structure`: chapter と並びの骨格を作る
8. `llm`: LLM に渡す prompt とドラフト plan を作る
9. `render`: representative ベースで preview を作る

## モジュールごとの責務

`src/sceneflow/pipeline/scan.py`

- メディア列挙
- `exiftool` / `ffprobe` からのメタデータ抽出
- 画像の blur / brightness 指標作成
- scene 分割の前提になる CSV を作る

`src/sceneflow/pipeline/sceneify.py`

- `captured_at`, `create_date`, ファイル名から `final_timestamp` を作る
- video 重複排除
- 同名コンテンツでは image より video を優先
- gap, GPS, scene 長で `scene_id` を振る
- singleton scene 吸収と giant scene 分割を行う

`src/sceneflow/pipeline/representatives.py`

- scene ごとに代表素材を 1 件選ぶ
- image があれば sharp な image を優先
- image がなければ短めの video を優先

`src/sceneflow/pipeline/tagging.py`

- representative に対する OCR
- OCR cache の管理
- 顔数の推定
- 簡易 food score, tag, caption の生成
- ここでの出力は helper signal であり ground truth ではない

`src/sceneflow/pipeline/candidates.py`

- scene を編集向け JSON に要約する中心モジュール
- `importance_score`, `priority_band`, `flow_summary` を作る
- downstream が読む基本契約はここで決まる

`src/sceneflow/pipeline/meanings.py`

- scene の role を決める
- `edit_action` を決める
- LLM に渡す前の意味づけを付ける

`src/sceneflow/pipeline/structure.py`

- opening / body / closing の章立て
- sequence 上の duration hint と transition hint

`src/sceneflow/pipeline/llm_plan.py`

- LLM に渡す prompt text を作る
- fallback の edit plan を作る

`src/sceneflow/pipeline/render.py`

- representative path を使って preview を作る
- 現状は final render ではなく simple preview renderer

## 変更時に守る契約

- 各 step の出力は `outputs/<run-id>/<step>/` に出る
- 各成果物には `*.meta.json` sidecar を付ける
- `sceneify` の出力 CSV は downstream が読む列を極力壊さない
- `representatives` は scene ごとに 1 行であることを保つ
- `candidates` の top-level には `generated_at`, `root`, `scene_count`, `summary`, `scenes` があること
- `meanings` の top-level には `summary`, `scenes` があること
- `structure` の top-level には `chapters`, `edit_sequence` があること
- `llm` の top-level には `plan` があること
- `render` は `edit_plan.json` の representative path に依存する

## どこを変えるときに何を見るか

- scene 分割を変えるなら `sceneify.py` を触り、`representatives` 以降の scene 数と flow がどう変わるかを見る
- representative 選定を変えるなら `representatives.py` を触り、`tagging` と `candidates` の summary がどう変わるかを見る
- OCR / 顔 / tag を変えるなら `tagging.py` を触り、`candidates` の `importance_score` が過剰に振れないかを見る
- 編集判断を変えるなら `candidates.py`, `meanings.py`, `structure.py` を見る
- LLM 用の入力を変えるなら `llm_plan.py` を触り、前段の JSON 契約はできるだけ壊さない
- preview の見え方を変えるなら `render.py` を触り、scene 選択ではなく見せ方の調整に留める

## 良い変更の判断基準

- scene のつながりが自然になる
- representative や summary が人間にレビューしやすい
- OCR や顔検出が欠けても大崩れしない
- 特定データだけに効く枝分かれを増やしすぎない
- downstream JSON が読みやすく保たれる

## いま不足しやすい前提

- `docs/workflow.md` は元ワークフロー由来なので、読むときは「現行 repo の入口」に読み替える
- 現在の repo では個別 step 実行に `scripts/run_step.py` を使うのが安全
- `python -m sceneflow...` 直実行は、そのままだと import path で迷いやすい
- sample data は placeholder だけなので、smoke test は手元の少量素材を置いて行う
- `scripts/run_checks.py` と fixture ベースの契約テストは入っているが、`scan`, `tagging`, `render` の外部ツール依存部分はまだ薄い
- そのため、自動チェックに加えて生成物レビューも引き続き重要

## 実行入口

フル実行:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -B scripts/run_pipeline.py --root data/sample --run-dir outputs/sample-run
```

個別 step 実行:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -B scripts/run_step.py scan --help
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -B scripts/run_step.py sceneify --help
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -B scripts/run_step.py candidates --help
```

品質チェック:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --group dev
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -B scripts/run_checks.py
```

## AI 作業の最小チェックリスト

- 変更した step を最小単位で再実行したか
- 下流の 1 step か 2 step 先まで壊れていないか
- `flow_summary`, `priority_band`, `summary` が人間に読めるか
- OCR / 顔検出の局所改善に引っ張られすぎていないか
- 最終的に watchable flow に近づいたと説明できるか
