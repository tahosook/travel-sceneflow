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
- `docs/workflow.md`: 元プロジェクトから引き継いだ詳細ワークフロー
- `docs/plans.md`: 開発メモと今後の方針
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

## 最初に読むとよいファイル

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
- より詳しい元ワークフローは `docs/workflow.md` を参照してください。
