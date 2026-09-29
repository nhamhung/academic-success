"""Tests for the shared feature engineering pipeline.

These use small synthetic DataFrames — no Kaggle data download needed to run
`pytest`.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from academic_success import config  # noqa: E402
from academic_success.features import (  # noqa: E402
    ENGINEERED_NUMERIC_COLS,
    RARE_CATEGORY_SENTINEL,
    FeatureEngineer,
    RareCategoryGrouper,
    build_preprocessor,
)


def _sample_row(**overrides) -> dict:
    row = {
        "Curricular units 1st sem (credited)": 0,
        "Curricular units 1st sem (approved)": 5,
        "Curricular units 1st sem (enrolled)": 6,
        "Curricular units 1st sem (evaluations)": 6,
        "Curricular units 1st sem (grade)": 12.0,
        "Curricular units 1st sem (without evaluations)": 0,
        "Curricular units 2nd sem (credited)": 0,
        "Curricular units 2nd sem (approved)": 6,
        "Curricular units 2nd sem (enrolled)": 6,
        "Curricular units 2nd sem (evaluations)": 6,
        "Curricular units 2nd sem (grade)": 14.0,
        "Curricular units 2nd sem (without evaluations)": 0,
        "Mother's qualification": 1,
        "Father's qualification": 3,
    }
    row.update(overrides)
    return row


def _sample_raw_frame(n_rows: int = 4) -> pd.DataFrame:
    """A minimal frame covering every raw feature column, for pipeline smoke tests."""
    base_row = {col: 0 for col in config.CATEGORICAL_COLS}
    base_row.update(
        {
            "Previous qualification (grade)": 120.0,
            "Admission grade": 120.0,
            "Age at enrollment": 20,
            "Unemployment rate": 10.0,
            "Inflation rate": 1.0,
            "GDP": 1.0,
        }
    )
    base_row.update(_sample_row())
    return pd.DataFrame([base_row for _ in range(n_rows)])


class TestFeatureEngineer:
    def test_approval_rate_normal_case(self):
        df = pd.DataFrame([_sample_row()])
        result = FeatureEngineer().fit_transform(df)
        assert result["approval_rate_1st_sem"].iloc[0] == pytest.approx(5 / 6)
        assert result["approval_rate_2nd_sem"].iloc[0] == pytest.approx(1.0)

    def test_approval_rate_zero_enrolled_does_not_divide_by_zero(self):
        df = pd.DataFrame(
            [_sample_row(**{"Curricular units 1st sem (enrolled)": 0, "Curricular units 1st sem (approved)": 0})]
        )
        result = FeatureEngineer().fit_transform(df)
        assert result["approval_rate_1st_sem"].iloc[0] == 0.0
        assert not result["approval_rate_1st_sem"].isna().any()

    def test_total_units_and_grade_trend(self):
        df = pd.DataFrame([_sample_row()])
        result = FeatureEngineer().fit_transform(df)
        assert result["total_units_approved"].iloc[0] == 11
        assert result["total_units_enrolled"].iloc[0] == 12
        assert result["grade_trend"].iloc[0] == pytest.approx(2.0)

    def test_no_evaluation_ratio_zero_total_evaluations(self):
        df = pd.DataFrame(
            [
                _sample_row(
                    **{
                        "Curricular units 1st sem (evaluations)": 0,
                        "Curricular units 2nd sem (evaluations)": 0,
                        "Curricular units 1st sem (without evaluations)": 0,
                        "Curricular units 2nd sem (without evaluations)": 0,
                    }
                )
            ]
        )
        result = FeatureEngineer().fit_transform(df)
        assert result["no_evaluation_ratio"].iloc[0] == 0.0

    def test_output_columns_include_all_engineered_features(self):
        df = pd.DataFrame([_sample_row()])
        result = FeatureEngineer().fit_transform(df)
        for col in ENGINEERED_NUMERIC_COLS:
            assert col in result.columns

    def test_max_parental_qualification_takes_max(self):
        df = pd.DataFrame(
            [_sample_row(**{"Mother's qualification": 2, "Father's qualification": 19})]
        )
        result = FeatureEngineer().fit_transform(df)
        assert result["max_parental_qualification"].iloc[0] == 19


class TestRareCategoryGrouper:
    def _fit_frame(self) -> pd.DataFrame:
        # 200 common-category rows, 2 singleton rare rows (each ~0.5% frequency,
        # below the 1% default threshold), for one grouped column.
        col = "Nacionality"
        common = pd.DataFrame({col: [1] * 140 + [2] * 60})
        rare = pd.DataFrame({col: [999, 888]})
        return pd.concat([common, rare], ignore_index=True)

    def test_frequent_categories_are_preserved(self):
        df = self._fit_frame()
        grouper = RareCategoryGrouper(columns=["Nacionality"], min_frequency=0.01)
        result = grouper.fit_transform(df)
        assert (result.loc[df["Nacionality"] == 1, "Nacionality"] == 1).all()
        assert (result.loc[df["Nacionality"] == 2, "Nacionality"] == 2).all()

    def test_rare_categories_are_grouped_to_sentinel(self):
        df = self._fit_frame()
        grouper = RareCategoryGrouper(columns=["Nacionality"], min_frequency=0.01)
        result = grouper.fit_transform(df)
        assert (result.loc[df["Nacionality"].isin([999, 888]), "Nacionality"] == RARE_CATEGORY_SENTINEL).all()

    def test_unseen_category_at_transform_time_is_grouped_too(self):
        df = self._fit_frame()
        grouper = RareCategoryGrouper(columns=["Nacionality"], min_frequency=0.01)
        grouper.fit(df)

        new_df = pd.DataFrame({"Nacionality": [1, 2, 123456]})
        result = grouper.transform(new_df)
        assert result["Nacionality"].tolist() == [1, 2, RARE_CATEGORY_SENTINEL]


class TestPreprocessor:
    def test_fits_and_transforms_without_error(self):
        X = _sample_raw_frame()
        engineered = FeatureEngineer().fit_transform(X)
        preprocessor = build_preprocessor()
        transformed = preprocessor.fit_transform(engineered)
        assert transformed.shape[0] == len(X)

    def test_handles_missing_values(self):
        X = _sample_raw_frame()
        X.loc[0, "Admission grade"] = None
        X.loc[0, config.CATEGORICAL_COLS[0]] = None
        engineered = FeatureEngineer().fit_transform(X)
        preprocessor = build_preprocessor()
        transformed = preprocessor.fit_transform(engineered)
        assert transformed.shape[0] == len(X)

    def test_handles_unseen_category_at_transform_time(self):
        X = _sample_raw_frame()
        preprocessor = build_preprocessor()
        engineered = FeatureEngineer().fit_transform(X)
        preprocessor.fit(engineered)

        X_new = _sample_raw_frame(n_rows=1)
        X_new[config.CATEGORICAL_COLS[0]] = 999_999  # category never seen during fit
        engineered_new = FeatureEngineer().fit_transform(X_new)
        # Should not raise, thanks to handle_unknown="ignore".
        preprocessor.transform(engineered_new)
