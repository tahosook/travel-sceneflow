# travel-sceneflow

旅行の写真・動画を scene 単位で整理し、見やすい動画の流れを作るためのリポジトリです。

## 目的

このリポジトリは、既存の scene workflow を開発しやすい形に整理したものです。  
主眼は認識精度そのものではなく、編集しやすく見やすい動画の流れを作ることにあります。  
OCR と顔検出は補助情報として扱います。

## 構成

- `src/sceneflow/pipeline/`: パイプライン本体
- `src/sceneflow/cli.py`: パッケージ側の実行入口
- `scripts/run_pipeline.py`: リポジトリ側の実行入口
- `scripts/run_step.py`: 個別 step を実行する入口
- `AGENTS.md`: 常設の判断基準
- `PLANS.md`: 長時間または複雑な作業向けのロードマップ
- `docs/workflow.md`: 元プロジェクトから引き継いだ詳細ワークフロー
- `docs/ai-coding.md`: 必要なときだけ見る補助的な参照メモ
- `data/sample/`: ローカル smoke test 用の素材置き場
- `outputs/`: 実行結果の出力先

## 最短の実行手順

まず依存を入れます。

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync
```

次に、`data/sample/` に少数のローカル素材を置いて実行します。

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -B scripts/run_pipeline.py --root data/sample --run-dir outputs/sample-run
```

`scene_meanings.json` と `edit_structure.json` まで続けて作りたい場合は、`--with-meanings --with-structure` を付けます。

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -B scripts/run_pipeline.py --root data/sample --run-dir outputs/sample-run --with-meanings --with-structure
```

`edit_plan.json` と `preview.mp4` まで進めたい場合は、`--with-llm --with-render` を使います。  
テロップやタイトルをその場で調整したいときは `--interactive` を付けると、`llm` step 中に質問が出ます。  
この対話では、タイトルだけでなく「誰に見せる動画か」「どの scene を強く見せたいか」「テンポ感」「最後に残したい余韻」も指定できます。

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -B scripts/run_pipeline.py --root data/sample --run-dir outputs/sample-run --with-render --interactive
```

個別 step だけ確認したいときは `scripts/run_step.py` を使えます。

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -B scripts/run_step.py scan --help
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -B scripts/run_step.py candidates --help
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -B scripts/run_step.py llm --help
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -B scripts/run_step.py render --help
```

## 品質チェック

開発用の check は dev 依存つきで入れます。

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --group dev
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -B scripts/run_checks.py
```

`tests/fixtures/media_info.csv` は、外部ツールなしで sceneify 以降の契約を確認するための固定 fixture です。

## 最初に読むとよいファイル

- `AGENTS.md`
- `src/sceneflow/cli.py`
- `src/sceneflow/pipeline/scan.py`
- `src/sceneflow/pipeline/sceneify.py`
- `src/sceneflow/pipeline/representatives.py`
- `src/sceneflow/pipeline/tagging.py`
- `src/sceneflow/pipeline/candidates.py`

## 補足

- `data/sample/` はローカル確認用で、実素材を commit する前提ではありません。
- `outputs/` は git 管理外のローカル生成物です。
- フルワークフローには `ffmpeg`, `exiftool`, `tesseract` が必要です。
- `AGENTS.md` を常設の instruction layer とし、`PLANS.md` は長時間または複雑な作業のときだけ使います。
- `docs/ai-coding.md` は参照用であり、`AGENTS.md` や `PLANS.md` を上書きしません。
- `docs/workflow.md` は元ワークフロー由来の詳細メモですが、現行 repo では `scripts/run_pipeline.py` と `scripts/run_step.py` を入口として使うのが安全です。
