"""Feature engineering and the shared preprocessing pipeline.

This is the single source of truth for turning raw competition columns into
model-ready features. The notebook, `scripts/train.py`,
`scripts/make_submission.py`, and the Streamlit app all call
`build_preprocessor()` (wrapped inside the fitted pipeline saved to
`models/model.joblib`) so none of them can silently diverge from how the
model was actually trained.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

# Names of the derived columns FeatureEngineer adds, in the order they're
# created. Declared once here so build_preprocessor() knows which columns to
# treat as numeric without re-deriving them.
ENGINEERED_NUMERIC_COLS = [
    "approval_rate_1st_sem",
    "approval_rate_2nd_sem",
    "total_units_approved",
    "total_units_enrolled",
    "grade_trend",
    "no_evaluation_ratio",
    "max_parental_qualification",
]

# High-cardinality categorical columns where a long tail of near-zero-frequency
# codes (verified against the real training data: e.g. Father's occupation has
# 56 distinct codes, 44 of them below 1% frequency) would otherwise blow up the
# one-hot encoding with columns the model can barely learn from, and risks an
# unseen rare code at inference time landing in a completely different bucket
# than any it was trained on. RareCategoryGrouper collapses them into a shared
# "rare" sentinel before one-hot encoding.
RARE_GROUPED_COLS = [
    "Application mode",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

# Sentinel value for a grouped-rare category. All raw category codes in this
# dataset are positive integers, so -1 can never collide with a real code.
RARE_CATEGORY_SENTINEL = -1


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """numerator / denominator, mapping 0/0 and x/0 to 0 instead of NaN/inf."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = numerator / denominator
    return ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Derives ratio/trend features from the raw curricular-unit columns.

    Kept as a proper scikit-learn transformer (not a bare function) so it
    composes into a `Pipeline` and is persisted with the fitted model —
    inference always applies the exact same derivation as training.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        df["approval_rate_1st_sem"] = _safe_ratio(
            df["Curricular units 1st sem (approved)"],
            df["Curricular units 1st sem (enrolled)"],
        )
        df["approval_rate_2nd_sem"] = _safe_ratio(
            df["Curricular units 2nd sem (approved)"],
            df["Curricular units 2nd sem (enrolled)"],
        )
        df["total_units_approved"] = (
            df["Curricular units 1st sem (approved)"]
            + df["Curricular units 2nd sem (approved)"]
        )
        df["total_units_enrolled"] = (
            df["Curricular units 1st sem (enrolled)"]
            + df["Curricular units 2nd sem (enrolled)"]
        )
        df["grade_trend"] = (
            df["Curricular units 2nd sem (grade)"] - df["Curricular units 1st sem (grade)"]
        )
        total_evaluations = (
            df["Curricular units 1st sem (evaluations)"]
            + df["Curricular units 2nd sem (evaluations)"]
        )
        total_without_evaluations = (
            df["Curricular units 1st sem (without evaluations)"]
            + df["Curricular units 2nd sem (without evaluations)"]
        )
        df["no_evaluation_ratio"] = _safe_ratio(total_without_evaluations, total_evaluations)

        # NOTE ON ORDINALITY: qualification codes (see the UCI dataset's code
        # legend) are categorical labels, not a ranked scale — code 19 isn't
        # "better" than code 3 in any consistent sense. Taking their numeric
        # max is a common, cheap Kaggle proxy for "at least one highly-coded
        # parental qualification", not a claim that the codes are ordinal.
        # Documented here and in the notebook/report rather than silently
        # treated as a clean ordinal feature.
        df["max_parental_qualification"] = df[
            ["Mother's qualification", "Father's qualification"]
        ].max(axis=1)

        return df

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        return np.array(base + ENGINEERED_NUMERIC_COLS)


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Collapses low-frequency categories (below `min_frequency`) into a
    shared sentinel value, learned from the training data only.

    Applied to `RARE_GROUPED_COLS` — high-cardinality columns where a long
    tail of near-unique codes would otherwise (a) fragment the one-hot
    encoding into many columns the model rarely sees, and (b) risk an unseen
    code at inference time being silently treated as "just another unknown
    category" by `OneHotEncoder(handle_unknown="ignore")`'s all-zero
    encoding, rather than being recognised as the same kind of "rare code" the
    model *did* see examples of during training.
    """

    def __init__(self, columns: list[str] | None = None, min_frequency: float = 0.01):
        self.columns = columns
        self.min_frequency = min_frequency

    def fit(self, X: pd.DataFrame, y=None) -> "RareCategoryGrouper":
        columns = self.columns if self.columns is not None else RARE_GROUPED_COLS
        self.kept_categories_: dict[str, set] = {}
        for col in columns:
            freqs = X[col].value_counts(normalize=True)
            self.kept_categories_[col] = set(freqs[freqs >= self.min_frequency].index)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col, kept in self.kept_categories_.items():
            df[col] = df[col].where(df[col].isin(kept), RARE_CATEGORY_SENTINEL)
        return df

    def get_feature_names_out(self, input_features=None):
        return np.array(list(input_features) if input_features is not None else [])


def build_preprocessor() -> ColumnTransformer:
    """The ColumnTransformer applied after `FeatureEngineer`.

    Numeric columns (raw + engineered): median-impute, then standard-scale.
    Categorical columns (integer category codes): most-frequent-impute, then
    one-hot encode with unknown categories mapped to all-zero rather than
    raising at inference time.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            # sparse_output=False: several of these columns (Course, Mother's/
            # Father's occupation, Nacionality) have enough distinct category
            # codes that a sparse one-hot matrix would otherwise trip
            # HistGradientBoostingClassifier, which requires dense input.
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, config.NUMERIC_COLS + ENGINEERED_NUMERIC_COLS),
            ("categorical", categorical_pipeline, config.CATEGORICAL_COLS),
        ]
    )


def build_feature_pipeline() -> Pipeline:
    """FeatureEngineer + preprocessor, without a final estimator.

    Useful on its own for inspecting transformed features (e.g. in the
    notebook's EDA section) without needing a trained model.
    """
    return Pipeline(
        steps=[
            ("engineer", FeatureEngineer()),
            ("group_rare", RareCategoryGrouper()),
            ("preprocess", build_preprocessor()),
        ]
    )
