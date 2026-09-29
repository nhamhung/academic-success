"""Tests for SHAP-based interpretability helpers.

Uses the same synthetic-data fixture pattern as test_model.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from academic_success import config, interpretability, model  # noqa: E402


@pytest.fixture(scope="module")
def fitted_pipeline_and_data():
    rng = np.random.default_rng(1)
    n = 150
    data_dict = {}
    for col in config.CATEGORICAL_COLS:
        data_dict[col] = rng.integers(1, 5, size=n)
    for col in config.NUMERIC_COLS:
        data_dict[col] = rng.normal(10, 3, size=n)
    X = pd.DataFrame(data_dict)
    y = pd.Series(rng.choice(config.TARGET_CLASSES, size=n))

    pipeline = model.train_pipeline(X, y, estimator=model.default_estimator())
    return pipeline, X


class TestComputeShapValues:
    def test_returns_one_row_per_sample_and_matches_feature_names(self, fitted_pipeline_and_data):
        pipeline, X = fitted_pipeline_and_data
        explanation, X_transformed = interpretability.compute_shap_values(
            pipeline, X, max_samples=50
        )
        assert explanation.values.shape[0] == 50
        assert explanation.values.shape[1] == X_transformed.shape[1]
        assert list(explanation.feature_names) == list(X_transformed.columns)

    def test_respects_max_samples(self, fitted_pipeline_and_data):
        pipeline, X = fitted_pipeline_and_data
        explanation, _ = interpretability.compute_shap_values(pipeline, X, max_samples=20)
        assert explanation.values.shape[0] == 20


class TestTopShapFeatures:
    def test_returns_requested_number_ranked_by_importance(self, fitted_pipeline_and_data):
        pipeline, X = fitted_pipeline_and_data
        explanation, _ = interpretability.compute_shap_values(pipeline, X, max_samples=50)

        top = interpretability.top_shap_features(explanation, top_n=5)
        assert len(top) == 5
        assert list(top.columns) == ["feature", "mean_abs_shap"]
        assert (top["mean_abs_shap"].diff().dropna() <= 0).all()  # sorted descending
