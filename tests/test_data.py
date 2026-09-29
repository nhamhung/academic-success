"""Tests for local and hosted competition-data loading."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from academic_success import config, data  # noqa: E402


def _point_config_at(monkeypatch, raw_dir: Path) -> Path:
    train_csv = raw_dir / "train.csv"
    monkeypatch.setattr(config, "DATA_RAW_DIR", raw_dir)
    monkeypatch.setattr(config, "TRAIN_CSV", train_csv)
    return train_csv


def test_download_train_reuses_existing_file(monkeypatch, tmp_path):
    train_csv = _point_config_at(monkeypatch, tmp_path / "raw")
    train_csv.parent.mkdir()
    train_csv.write_text("id,Target\n1,Graduate\n", encoding="utf-8")

    assert data.download_train() == train_csv


def test_download_train_requires_token_when_file_is_missing(monkeypatch, tmp_path):
    _point_config_at(monkeypatch, tmp_path / "raw")
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)

    with pytest.raises(FileNotFoundError, match="KAGGLE_API_TOKEN"):
        data.download_train()


def test_download_train_uses_kagglehub_without_persisting_token(monkeypatch, tmp_path):
    train_csv = _point_config_at(monkeypatch, tmp_path / "raw")
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)

    def fake_download(competition, *, path, output_dir):
        assert competition == config.KAGGLE_COMPETITION
        assert path == "train.csv"
        target = Path(output_dir) / path
        target.write_text("id,Target\n1,Graduate\n", encoding="utf-8")
        assert os.environ["KAGGLE_API_TOKEN"] == "test-token"
        return str(target)

    monkeypatch.setitem(
        sys.modules,
        "kagglehub",
        SimpleNamespace(competition_download=fake_download),
    )

    assert data.download_train(api_token="test-token") == train_csv
    assert "KAGGLE_API_TOKEN" not in os.environ
