from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from tests.helpers import build_tagged_representatives


ROOT = Path(__file__).resolve().parents[1]


def run_step(*args: str) -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_step.py", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"run_step failed: {' '.join(args)}\nstdout={result.stdout}\nstderr={result.stderr}")


def test_run_step_pipeline_smoke(media_info_csv: Path, sample_root: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"

    run_step("sceneify", "--input", str(media_info_csv), "--run-dir", str(run_dir))
    run_step(
        "representatives",
        "--input",
        str(run_dir / "sceneify" / "media_scene.csv"),
        "--run-dir",
        str(run_dir),
    )

    reps_path = run_dir / "representatives" / "scene_representatives.csv"
    tagged_path = run_dir / "tagging" / "scene_representatives_tagged.csv"
    tagged_path.parent.mkdir(parents=True, exist_ok=True)
    reps_df = pd.read_csv(reps_path)
    build_tagged_representatives(reps_df).to_csv(tagged_path, index=False)

    run_step(
        "candidates",
        "--media-scene",
        str(run_dir / "sceneify" / "media_scene.csv"),
        "--representatives",
        str(tagged_path),
        "--root",
        str(sample_root),
        "--run-dir",
        str(run_dir),
    )
    run_step("meanings", "--input", str(run_dir / "candidates" / "scene_edit_candidates.json"), "--run-dir", str(run_dir))
    run_step("structure", "--input", str(run_dir / "meaning" / "scene_meanings.json"), "--run-dir", str(run_dir))
    run_step("llm", "--input", str(run_dir / "structure" / "edit_structure.json"), "--run-dir", str(run_dir))

    candidates_payload = json.loads((run_dir / "candidates" / "scene_edit_candidates.json").read_text(encoding="utf-8"))
    structure_payload = json.loads((run_dir / "structure" / "edit_structure.json").read_text(encoding="utf-8"))
    llm_payload = json.loads((run_dir / "llm" / "edit_plan.json").read_text(encoding="utf-8"))

    assert candidates_payload["scene_count"] == 3
    assert len(candidates_payload["scenes"]) == 3
    assert len(candidates_payload["scenes"][0]["preview_sources"]) == 3
    assert len(structure_payload["edit_sequence"]) == 3
    assert structure_payload["edit_sequence"][0]["start_at"] == "2024-05-01T09:00:00+09:00"
    assert len(llm_payload["plan"]["edit_sequence"][0]["preview_sources"]) == 3
    assert [card["title"] for card in llm_payload["plan"]["title_cards"]] == ["2024年5月1日", "旅の流れ", "旅の余韻"]
    assert llm_payload["plan"]["render_guidance"]["preferred_order"] == [1, 2, 3]
