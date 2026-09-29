"""Model pipeline construction, training, and persistence.

The saved artifact at `models/model.joblib` is a single fitted pipeline
covering feature engineering, preprocessing, (optional) class-imbalance
resampling, and the classifier — loading it and calling `.predict()` is the
*only* thing `scripts/make_submission.py` and the Streamlit app need to do.

Pipelines are built with `imblearn.pipeline.Pipeline` rather than
`sklearn.pipeline.Pipeline` throughout, even when no resampler is used: it is
a drop-in superset (ordinary transformers/estimators behave identically) that
also lets `build_pipeline(..., resample=True)` insert ADASYN as a step
without a separate code path. ADASYN only resamples during `.fit()` — at
`.predict()` time (and therefore for every validation fold during
cross-validation, and at real inference time) it is a no-op passthrough, so
synthetic samples can never leak into a score or a served prediction.
"""

from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

from . import config
from .features import build_feature_pipeline

# --- Individual model factories -----------------------------------------
# Each returns a fresh, unfitted estimator. Kept as small named functions
# (rather than one dict of instances) so every call gets an independent
# estimator instance — sharing one instance across cross_validate folds or
# multiple pipelines would let fitted state leak between them.


def default_estimator() -> BaseEstimator:
    """HistGradientBoostingClassifier: in scikit-learn core (no extra
    dependency), handles the mixed one-hot/scaled feature matrix well, and is
    a reasonable stand-in for the gradient-boosted trees that dominate this
    competition's leaderboard.
    """
    return HistGradientBoostingClassifier(random_state=config.RANDOM_SEED)


def random_forest_estimator() -> BaseEstimator:
    return RandomForestClassifier(n_estimators=300, random_state=config.RANDOM_SEED, n_jobs=-1)


def adaboost_estimator() -> BaseEstimator:
    return AdaBoostClassifier(random_state=config.RANDOM_SEED)


def gradient_boosting_estimator() -> BaseEstimator:
    """Classic (non-histogram) gradient boosting — slower than
    HistGradientBoostingClassifier but a useful comparison point since it's
    the "vanilla" boosting algorithm the histogram-based variant optimizes.
    """
    return GradientBoostingClassifier(random_state=config.RANDOM_SEED)


class _StringLabelXGBClassifier(BaseEstimator, ClassifierMixin):
    """Wraps XGBClassifier so it accepts and returns the original string
    class labels ("Dropout"/"Enrolled"/"Graduate"), like every other
    estimator in this module.

    Unlike sklearn's own classifiers (and LightGBM/CatBoost's sklearn APIs),
    XGBoost's multiclass objective requires `y` pre-encoded as contiguous
    integers 0..n_classes-1 — passing string labels directly raises
    `ValueError: Invalid classes inferred from unique values of y`. Handling
    that encoding here (rather than in every caller) keeps `train.py`, the
    notebook, and `build_stacking_pipeline()` free of a special case.
    """

    def __init__(self, n_estimators: int = 300, random_state: int | None = None):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        from xgboost import XGBClassifier

        self._label_encoder = LabelEncoder().fit(y)
        self._model = XGBClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            eval_metric="mlogloss",
        )
        self._model.fit(X, self._label_encoder.transform(y))
        self.classes_ = self._label_encoder.classes_
        return self

    def predict(self, X):
        return self._label_encoder.inverse_transform(self._model.predict(X))

    def predict_proba(self, X):
        return self._model.predict_proba(X)


def xgboost_estimator() -> BaseEstimator:
    return _StringLabelXGBClassifier(n_estimators=300, random_state=config.RANDOM_SEED)


def lightgbm_estimator() -> BaseEstimator:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(n_estimators=300, random_state=config.RANDOM_SEED, verbosity=-1)


class _FlattenedCatBoostClassifier(BaseEstimator, ClassifierMixin):
    """Wraps CatBoostClassifier because its `.predict()` returns a 2D
    `(n_samples, 1)` array instead of a flat 1D array of labels like every
    other classifier here (sklearn's, XGBoost's, and LightGBM's `.predict()`
    all return 1D arrays) — a real discrepancy that would otherwise silently
    break `scripts/make_submission.py`'s `submission[config.TARGET_COL] =
    predictions` column assignment if CatBoost were ever selected as the
    saved model.
    """

    def __init__(self, iterations: int = 300, random_state: int | None = None):
        self.iterations = iterations
        self.random_state = random_state

    def fit(self, X, y):
        from catboost import CatBoostClassifier

        self._model = CatBoostClassifier(
            iterations=self.iterations,
            random_state=self.random_state,
            verbose=False,
            allow_writing_files=False,
        )
        self._model.fit(X, y)
        self.classes_ = self._model.classes_
        return self

    def predict(self, X):
        return self._model.predict(X).ravel()

    def predict_proba(self, X):
        return self._model.predict_proba(X)


def catboost_estimator() -> BaseEstimator:
    """CatBoost is worth including specifically because it can consume the
    integer-coded categorical columns directly (via `cat_features`) without
    one-hot encoding at all — unlike every other model here. This project's
    shared pipeline still one-hot encodes upstream (so all models compare on
    the same feature representation), but this is a good place to note that
    tradeoff in the notebook: CatBoost's native categorical handling is an
    alternative to `RareCategoryGrouper` + one-hot, not a complement to it.
    """
    return _FlattenedCatBoostClassifier(iterations=300, random_state=config.RANDOM_SEED)


MODEL_FACTORIES: dict[str, "callable[[], BaseEstimator]"] = {
    "Logistic Regression": lambda: LogisticRegression(
        max_iter=1000, random_state=config.RANDOM_SEED
    ),
    "Random Forest": random_forest_estimator,
    "AdaBoost": adaboost_estimator,
    "Gradient Boosting": gradient_boosting_estimator,
    "HistGradientBoosting": default_estimator,
    "XGBoost": xgboost_estimator,
    "LightGBM": lightgbm_estimator,
    "CatBoost": catboost_estimator,
}


def build_pipeline(estimator: BaseEstimator | None = None, resample: bool = False) -> Pipeline:
    """Feature engineering + preprocessing + (optional ADASYN) + classifier.

    `resample=True` inserts ADASYN right after preprocessing (once the raw
    mixed categorical/numeric columns have already become a purely numeric
    matrix, which is what ADASYN's nearest-neighbor interpolation requires).
    """
    pipeline = build_feature_pipeline()
    steps = list(pipeline.steps)
    if resample:
        steps.append(("resample", ADASYN(random_state=config.RANDOM_SEED)))
    # NOTE: deliberately `is None`, not `estimator or default_estimator()` —
    # several sklearn ensemble classes (RandomForestClassifier, AdaBoost,
    # GradientBoosting) define `__len__` (their fitted estimator count) but
    # not `__bool__`, so `bool(estimator)` falls back to `__len__()` and
    # raises AttributeError on an unfitted instance before `or` can even
    # short-circuit. `is None` sidesteps truthiness entirely.
    steps.append(("model", estimator if estimator is not None else default_estimator()))
    return Pipeline(steps=steps)


def build_stacking_estimator() -> BaseEstimator:
    """The stacking ensemble as a standalone estimator (for passing to
    `cross_validate_pipeline`/`build_pipeline` like any other model factory):
    several strong base learners feed a logistic regression meta-learner,
    which learns how to weigh their predictions rather than simply averaging
    them. `StackingClassifier` generates each base learner's training-time
    input to the meta-learner via internal cross-validation (out-of-fold
    predictions), the built-in equivalent of the hand-rolled OOF loop in the
    classic "Introduction to Ensembling/Stacking" tutorial — without needing
    to reimplement that K-fold bookkeeping by hand.
    """
    base_estimators = [
        ("random_forest", random_forest_estimator()),
        ("xgboost", xgboost_estimator()),
        ("lightgbm", lightgbm_estimator()),
        ("catboost", catboost_estimator()),
    ]
    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED),
        cv=5,
        n_jobs=-1,
    )


def build_stacking_pipeline(resample: bool = True) -> Pipeline:
    """The full feature-engineering + preprocessing + stacking-ensemble pipeline."""
    return build_pipeline(build_stacking_estimator(), resample=resample)


def cross_validate_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    estimator: BaseEstimator | None = None,
    cv: int = 5,
    resample: bool = False,
) -> dict:
    """Stratified-by-default cross_validate; returns mean/std accuracy and macro-F1."""
    pipeline = build_pipeline(estimator, resample=resample)
    scores = cross_validate(
        pipeline, X, y, cv=cv, scoring=["accuracy", "f1_macro"], return_train_score=False
    )
    return {
        "accuracy_mean": scores["test_accuracy"].mean(),
        "accuracy_std": scores["test_accuracy"].std(),
        "f1_macro_mean": scores["test_f1_macro"].mean(),
        "f1_macro_std": scores["test_f1_macro"].std(),
    }


def train_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    estimator: BaseEstimator | None = None,
    resample: bool = False,
) -> Pipeline:
    """Fit a fresh pipeline on the full given data."""
    pipeline = build_pipeline(estimator, resample=resample)
    pipeline.fit(X, y)
    return pipeline


def save_pipeline(pipeline: Pipeline, path: Path = config.MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path: Path = config.MODEL_PATH) -> Pipeline:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Train a model first: `python scripts/train.py`."
        )
    return joblib.load(path)
