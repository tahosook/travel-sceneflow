# Sample Data

This directory is a placeholder for a few local media files used in smoke tests.

Keep committed real travel media out of the repository. When you want to run a quick local check, drop a small handful of images or videos here and point the pipeline at this directory.

Example:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -B scripts/run_pipeline.py --root data/sample --run-dir outputs/sample-run
```
