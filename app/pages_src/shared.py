"""Cached data/model loaders shared across every page.

Centralized here (rather than re-loaded per page) so the training data and
trained pipeline are each read from disk exactly once per session, and every
page sees the same defaults.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from academic_success import config, data, features, interpretability, model  # noqa: E402


def _configure_kaggle_credentials() -> None:
    """Wire Kaggle API credentials from Streamlit secrets into the
    environment variables the `kaggle` package reads, so a deployment
    without a pre-baked Docker image (e.g. Streamlit Community Cloud)
    can fetch the competition data automatically on first load — see
    `data._download_from_kaggle`. A no-op if real environment variables
    are already set (e.g. running locally) or no `[kaggle]` secret is
    configured (falls back to `~/.kaggle/kaggle.json` if present, or to
    the manual-download error message if not).
    """
    if os.environ.get("KAGGLE_API_TOKEN") or (
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    ):
        return
    try:
        token = st.secrets.get("KAGGLE_API_TOKEN")
        if token:
            os.environ["KAGGLE_API_TOKEN"] = token
            return
    except Exception:
        pass
    try:
        os.environ["KAGGLE_USERNAME"] = st.secrets["kaggle"]["username"]
        os.environ["KAGGLE_KEY"] = st.secrets["kaggle"]["key"]
    except Exception:
        pass


_configure_kaggle_credentials()


@st.cache_resource
def get_pipeline():
    return model.load_pipeline()


@st.cache_data
def get_train_df() -> pd.DataFrame:
    return data.load_train()


@st.cache_data
def get_default_row() -> dict:
    """One representative "average" student: per-column median (numeric) or
    mode (categorical) across the full training set — used to pre-fill the
    form before any real student has been loaded.
    """
    train_df = get_train_df()
    defaults = {}
    for col in config.NUMERIC_COLS:
        defaults[col] = float(train_df[col].median())
    for col in config.CATEGORICAL_COLS:
        defaults[col] = int(train_df[col].mode().iloc[0])
    return defaults


@st.cache_data
def get_engineered_df() -> pd.DataFrame:
    """Training data with engineered features attached, for the EDA pages."""
    train_df = get_train_df()
    X = train_df[config.RAW_FEATURE_COLS]
    engineered = features.FeatureEngineer().fit_transform(X)
    engineered[config.TARGET_COL] = train_df[config.TARGET_COL].values
    return engineered


@st.cache_data(show_spinner="Computing SHAP values (first load only)...")
def get_shap_explanation(sample_size: int = 300):
    pipeline = get_pipeline()
    train_df = get_train_df()
    X = train_df[config.RAW_FEATURE_COLS]
    explanation, X_transformed = interpretability.compute_shap_values(
        pipeline, X, max_samples=sample_size
    )
    return explanation, X_transformed
