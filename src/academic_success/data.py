"""Data loading and optional Kaggle download helpers.

Raw CSVs are not committed to the repo (Kaggle competition data may not be
redistributed). Download them first — see the project README for the
`kaggle competitions download` command. Hosted applications can instead call
``download_train`` with a Kaggle API token supplied by their secrets manager.
"""

import os
from pathlib import Path

import pandas as pd

from . import config


def download_train(api_token: str | None = None) -> Path:
    """Download ``train.csv`` from Kaggle when it is not already available.

    The token is never persisted by this function. It is passed to KaggleHub
    through ``KAGGLE_API_TOKEN``, which is suitable for Streamlit Community
    Cloud and other secrets managers. The caller must have accepted the
    competition rules on Kaggle first.
    """
    if config.TRAIN_CSV.exists():
        return config.TRAIN_CSV

    token = api_token or os.getenv("KAGGLE_API_TOKEN")
    if not token:
        raise FileNotFoundError(
            f"{config.TRAIN_CSV} is missing and no Kaggle API token is configured. "
            "Download the competition data locally or set KAGGLE_API_TOKEN in "
            "your deployment secrets."
        )

    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    previous_token = os.environ.get("KAGGLE_API_TOKEN")
    os.environ["KAGGLE_API_TOKEN"] = token
    try:
        # Import only after exposing the token because authentication behavior
        # may be initialized while KaggleHub itself is imported.
        import kagglehub

        downloaded = Path(
            kagglehub.competition_download(
                config.KAGGLE_COMPETITION,
                path=config.TRAIN_CSV.name,
                output_dir=str(config.DATA_RAW_DIR),
            )
        )
    finally:
        if previous_token is None:
            os.environ.pop("KAGGLE_API_TOKEN", None)
        else:
            os.environ["KAGGLE_API_TOKEN"] = previous_token

    candidates = (config.TRAIN_CSV, downloaded, downloaded / config.TRAIN_CSV.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Kaggle reported a successful download, but train.csv was not found in "
        f"{config.DATA_RAW_DIR}."
    )


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the competition data first — see "
            "the README's 'Get the data' section, e.g.:\n"
            f"  kaggle competitions download -c {config.KAGGLE_COMPETITION} "
            f"-p {config.DATA_RAW_DIR}\n"
            f"  unzip -o {config.DATA_RAW_DIR / (config.KAGGLE_COMPETITION + '.zip')} "
            f"-d {config.DATA_RAW_DIR}"
        )
    return path


def load_train(path: Path | None = None) -> pd.DataFrame:
    """Load the labeled training data, indexed by id."""
    df = pd.read_csv(_require_file(path or config.TRAIN_CSV))
    return df.set_index(config.ID_COL)


def load_test() -> pd.DataFrame:
    """Load the unlabeled competition test set, indexed by id."""
    df = pd.read_csv(_require_file(config.TEST_CSV))
    return df.set_index(config.ID_COL)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a labeled frame into (X, y), restricted to the known feature schema."""
    missing = set(config.RAW_FEATURE_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Input frame is missing expected columns: {sorted(missing)}")
    X = df[config.RAW_FEATURE_COLS].copy()
    y = df[config.TARGET_COL].copy()
    return X, y
