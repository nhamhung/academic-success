"""SHAP-based model interpretability.

SHAP (SHapley Additive exPlanations) attributes each prediction to individual
feature contributions, based on the game-theoretic idea of fairly splitting
credit among "players" (here, features) for a shared payout (here, the
prediction). `shap.TreeExplainer` computes exact Shapley values efficiently
for tree-based models by walking the tree structure directly, rather than the
slow model-agnostic sampling every other SHAP explainer needs.

That efficient path only works for a *single* tree-based estimator, not for a
`StackingClassifier` meta-ensemble (whose own decision logic is a linear
combination of other models' outputs, not a tree). So these helpers explain
one concrete tree model at a time — pass the best-performing single model
from the notebook's comparison, not the stacking ensemble itself.
"""

import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from .model import _FlattenedCatBoostClassifier, _StringLabelXGBClassifier

_WRAPPED_MODEL_TYPES = (_StringLabelXGBClassifier, _FlattenedCatBoostClassifier)


def _underlying_tree_model(fitted_model):
    """Unwrap this project's label/output-shape wrapper classes to the raw
    tree model `shap.TreeExplainer` expects; every other model factory here
    already exposes itself directly.
    """
    if isinstance(fitted_model, _WRAPPED_MODEL_TYPES):
        return fitted_model._model
    return fitted_model


def compute_shap_values(
    pipeline: Pipeline, X: pd.DataFrame, max_samples: int = 500, random_state: int = 42
) -> tuple[shap.Explanation, pd.DataFrame]:
    """Compute SHAP values for a fitted single-tree-model pipeline.

    Returns `(explanation, X_transformed)`: the SHAP explanation (one row per
    sampled input, one column per transformed feature, one "layer" per class
    for multiclass models) and the transformed feature matrix it was computed
    against (with real column names) — useful for scatter/dependence plots
    that need both the SHAP values and the underlying feature values.

    `max_samples` subsamples `X` before computing SHAP values: exact tree
    SHAP is fast per-row, but still linear in row count, and a summary plot
    doesn't need every training row to be informative.
    """
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=random_state)

    preprocessing = pipeline[:-1]
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    X_transformed = pd.DataFrame(
        preprocessing.transform(X), columns=feature_names, index=X.index
    )

    tree_model = _underlying_tree_model(pipeline.named_steps["model"])
    explainer = shap.TreeExplainer(tree_model)
    explanation = explainer(X_transformed)
    return explanation, X_transformed


def top_shap_features(explanation: shap.Explanation, top_n: int = 15) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value, averaged across classes for
    a multiclass explanation (i.e. overall importance, not per-class).
    """
    values = explanation.values
    if values.ndim == 3:  # (n_samples, n_features, n_classes)
        importance = abs(values).mean(axis=(0, 2))
    else:  # (n_samples, n_features)
        importance = abs(values).mean(axis=0)

    return (
        pd.DataFrame({"feature": explanation.feature_names, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
