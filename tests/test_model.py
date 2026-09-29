"""Tests for model construction, training, and persistence.

Uses a synthetic DataFrame covering every raw feature column (no Kaggle
download needed), sized large enough (200 rows, all 3 classes present) for
ADASYN's nearest-neighbor resampling and 5-fold `StackingClassifier`
cross-validation to run without error.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from academic_success import config, model  # noqa: E402


@pytest.fixture(scope="module")
def synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 200
    data_dict = {}
    for col in config.CATEGORICAL_COLS:
        data_dict[col] = rng.integers(1, 5, size=n)
    for col in config.NUMERIC_COLS:
        data_dict[col] = rng.normal(10, 3, size=n)
    X = pd.DataFrame(data_dict)
    y = pd.Series(rng.choice(config.TARGET_CLASSES, size=n))
    return X, y


class TestModelFactories:
    @pytest.mark.parametrize("name", list(model.MODEL_FACTORIES.keys()))
    def test_each_factory_trains_and_predicts(self, synthetic_data, name):
        X, y = synthetic_data
        factory = model.MODEL_FACTORIES[name]
        pipeline = model.train_pipeline(X, y, estimator=factory())

        predictions = pipeline.predict(X.iloc[:5])
        assert len(predictions) == 5
        assert set(predictions).issubset(set(config.TARGET_CLASSES))

        probabilities = pipeline.predict_proba(X.iloc[:5])
        assert probabilities.shape == (5, 3)


class TestResamplingAndStacking:
    def test_adasyn_resample_true_trains_and_predicts(self, synthetic_data):
        X, y = synthetic_data
        pipeline = model.train_pipeline(X, y, resample=True)
        predictions = pipeline.predict(X.iloc[:5])
        assert len(predictions) == 5

    def test_stacking_pipeline_trains_and_predicts(self, synthetic_data):
        X, y = synthetic_data
        pipeline = model.build_stacking_pipeline(resample=True)
        pipeline.fit(X, y)
        predictions = pipeline.predict(X.iloc[:5])
        assert len(predictions) == 5
        assert set(predictions).issubset(set(config.TARGET_CLASSES))


class TestPersistence:
    def test_save_and_load_roundtrip_predicts_identically(self, synthetic_data):
        X, y = synthetic_data
        pipeline = model.train_pipeline(X, y, estimator=model.xgboost_estimator())

        path = Path(tempfile.mkdtemp()) / "model.joblib"
        model.save_pipeline(pipeline, path)
        loaded = model.load_pipeline(path)

        assert (loaded.predict(X.iloc[:10]) == pipeline.predict(X.iloc[:10])).all()

    def test_load_missing_path_raises_helpful_error(self):
        with pytest.raises(FileNotFoundError, match="Train a model first"):
            model.load_pipeline(Path("/nonexistent/model.joblib"))
