"""Tests for local and Kaggle-API-backed competition data loading."""

from pathlib import Path

import pytest

from academic_success import config, data


def _point_config_at(monkeypatch, raw_dir: Path) -> Path:
    train_csv = raw_dir / "train.csv"
    monkeypatch.setattr(config, "DATA_RAW_DIR", raw_dir)
    monkeypatch.setattr(config, "TRAIN_CSV", train_csv)
    return train_csv


def test_load_train_reuses_existing_file(monkeypatch, tmp_path):
    train_csv = _point_config_at(monkeypatch, tmp_path / "raw")
    train_csv.parent.mkdir()
    train_csv.write_text("id,Target\n1,Graduate\n", encoding="utf-8")
    monkeypatch.setattr(data, "_download_from_kaggle", lambda: pytest.fail("unexpected download"))

    loaded = data.load_train()
    assert loaded.loc[1, "Target"] == "Graduate"


def test_load_train_reports_failed_api_download(monkeypatch, tmp_path):
    _point_config_at(monkeypatch, tmp_path / "raw")
    monkeypatch.setattr(data, "_download_from_kaggle", lambda: False)

    with pytest.raises(FileNotFoundError, match="automatic download via the Kaggle API"):
        data.load_train()


def test_load_train_uses_kaggle_api_when_missing(monkeypatch, tmp_path):
    train_csv = _point_config_at(monkeypatch, tmp_path / "raw")

    def fake_download():
        train_csv.parent.mkdir()
        train_csv.write_text("id,Target\n1,Graduate\n", encoding="utf-8")
        return True

    monkeypatch.setattr(data, "_download_from_kaggle", fake_download)

    loaded = data.load_train()
    assert loaded.loc[1, "Target"] == "Graduate"
