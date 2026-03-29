from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def sample_root(tmp_path: Path) -> Path:
    root = tmp_path / "sample-root"
    root.mkdir()
    return root


@pytest.fixture
def media_info_df(sample_root: Path) -> pd.DataFrame:
    df = pd.read_csv(FIXTURE_DIR / "media_info.csv")
    df["path"] = df["path"].map(lambda value: str((sample_root / str(value)).resolve()))
    return df


@pytest.fixture
def media_info_csv(tmp_path: Path, media_info_df: pd.DataFrame) -> Path:
    path = tmp_path / "inputs" / "media_info.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    media_info_df.to_csv(path, index=False)
    return path


@pytest.fixture
def materialized_media_files(media_info_df: pd.DataFrame) -> list[Path]:
    created: list[Path] = []
    for value in media_info_df["path"].tolist():
        path = Path(str(value))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        created.append(path)
    return created
