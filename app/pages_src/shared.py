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

from academic_success import config, data, features, model  # noqa: E402


@st.cache_resource
def get_pipeline():
    return model.load_pipeline()


@st.cache_data
def get_train_df() -> pd.DataFrame:
    if config.TRAIN_CSV.exists():
        return data.load_train()

    token = os.getenv("KAGGLE_API_TOKEN")
    if not token:
        try:
            token = st.secrets.get("KAGGLE_API_TOKEN")
        except FileNotFoundError:
            token = None

    try:
        downloaded_path = data.download_train(api_token=token)
    except (FileNotFoundError, RuntimeError) as exc:
        st.error(
            "The training data is unavailable. For a hosted deployment, add "
            "`KAGGLE_API_TOKEN` to the app's Streamlit secrets after accepting "
            "the competition rules on Kaggle."
        )
        st.exception(exc)
        st.stop()

    return data.load_train(downloaded_path)


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
    # SHAP is intentionally lazy-loaded. Importing it eagerly adds substantial
    # cold-start time even when a visitor never opens Model Insights.
    from academic_success import interpretability

    pipeline = get_pipeline()
    train_df = get_train_df()
    X = train_df[config.RAW_FEATURE_COLS]
    explanation, X_transformed = interpretability.compute_shap_values(
        pipeline, X, max_samples=sample_size
    )
    return explanation, X_transformed
