#!/usr/bin/env python
"""Train the model on the full training set and save it to models/model.joblib.

Usage:
    python scripts/train.py                                  # fast default: HistGradientBoosting
    python scripts/train.py --model random_forest             # pick any single model (see --help)
    python scripts/train.py --resample                        # apply ADASYN class-imbalance resampling
    python scripts/train.py --stacking                        # stacking ensemble (slower, strongest)
    python scripts/train.py --stacking --resample              # combine both
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from academic_success import config, data, model  # noqa: E402

# CLI-friendly aliases for model.MODEL_FACTORIES's keys.
MODEL_CHOICES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "adaboost": "AdaBoost",
    "gradient_boosting": "Gradient Boosting",
    "hist_gradient_boosting": "HistGradientBoosting",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CHOICES),
        default="hist_gradient_boosting",
        help="Which single model to train (ignored if --stacking is set).",
    )
    parser.add_argument(
        "--stacking",
        action="store_true",
        help="Train the stacking ensemble (Random Forest + XGBoost + LightGBM + CatBoost -> Logistic Regression) instead of a single model. Slower.",
    )
    parser.add_argument(
        "--resample",
        action="store_true",
        help="Apply ADASYN oversampling for class imbalance (training folds only, never validation/test).",
    )
    args = parser.parse_args()

    print(f"Loading training data from {config.TRAIN_CSV} ...")
    train_df = data.load_train()
    X, y = data.split_features_target(train_df)
    print(f"Loaded {len(X)} rows, {X.shape[1]} raw feature columns.")

    if args.stacking:
        print("Cross-validating the stacking ensemble (5-fold) ...")
        cv_scores = model.cross_validate_pipeline(
            X, y, estimator=model.build_stacking_estimator(), cv=5, resample=args.resample
        )
    else:
        estimator_name = MODEL_CHOICES[args.model]
        print(f"Cross-validating {estimator_name} (5-fold) ...")
        cv_scores = model.cross_validate_pipeline(
            X, y, estimator=model.MODEL_FACTORIES[estimator_name](), cv=5, resample=args.resample
        )

    print(
        f"  accuracy = {cv_scores['accuracy_mean']:.4f} "
        f"(+/- {cv_scores['accuracy_std']:.4f})"
    )
    print(
        f"  macro F1 = {cv_scores['f1_macro_mean']:.4f} "
        f"(+/- {cv_scores['f1_macro_std']:.4f})"
    )

    print("Fitting final model on the full training set ...")
    if args.stacking:
        pipeline = model.build_stacking_pipeline(resample=args.resample)
        pipeline.fit(X, y)
    else:
        pipeline = model.train_pipeline(
            X, y, estimator=model.MODEL_FACTORIES[MODEL_CHOICES[args.model]](), resample=args.resample
        )

    model.save_pipeline(pipeline)
    print(f"Saved trained pipeline to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
