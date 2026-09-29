"""Predict page: load a real student (or start from dataset averages), tweak
the handful of fields that matter most (per the project's SHAP analysis —
see the Model Insights page), and see the model's prediction update.
"""

import pandas as pd
import streamlit as st

from . import shared
from academic_success import config

# The "fields that matter most" section below (curricular unit counts/
# grades for both semesters, tuition status, scholarship, age) is the raw
# columns that most directly drive the top SHAP-ranked engineered features
# (approval_rate_*_sem, total_units_approved, grade_trend) plus the next few
# raw columns SHAP ranked highest on their own — see the Model Insights page.
# Promoting these to the top of the form, instead of burying them among 30
# other fields, is the whole point of this redesign.

BINARY_LABELS = {
    "Tuition fees up to date": [("Yes", 1), ("No", 0)],
    "Scholarship holder": [("No", 0), ("Yes", 1)],
    "Gender": [("Male", 1), ("Female", 0)],
    "International": [("No", 0), ("Yes", 1)],
    "Displaced": [("No", 0), ("Yes", 1)],
    "Educational special needs": [("No", 0), ("Yes", 1)],
    "Debtor": [("No", 0), ("Yes", 1)],
    "Daytime/evening attendance": [("Daytime", 1), ("Evening", 0)],
}

FIELD_LABELS = {
    "Curricular units 1st sem (credited)": "Units credited (1st sem)",
    "Curricular units 1st sem (enrolled)": "Units enrolled (1st sem)",
    "Curricular units 1st sem (evaluations)": "Units evaluated (1st sem)",
    "Curricular units 1st sem (approved)": "Units approved (1st sem)",
    "Curricular units 1st sem (grade)": "Average grade (1st sem)",
    "Curricular units 1st sem (without evaluations)": "Units without evaluation (1st sem)",
    "Curricular units 2nd sem (credited)": "Units credited (2nd sem)",
    "Curricular units 2nd sem (enrolled)": "Units enrolled (2nd sem)",
    "Curricular units 2nd sem (evaluations)": "Units evaluated (2nd sem)",
    "Curricular units 2nd sem (approved)": "Units approved (2nd sem)",
    "Curricular units 2nd sem (grade)": "Average grade (2nd sem)",
    "Curricular units 2nd sem (without evaluations)": "Units without evaluation (2nd sem)",
}

# Columns that are conceptually integer counts/codes, even though
# `get_default_row()`'s per-column median (for NUMERIC_COLS) comes back as a
# float — `Series.median()` always returns float64, even a whole number like
# 6.0, and worse, can land on a genuine half-integer like 5.5 for an
# even-length count column. Every value written into one of these fields'
# session-state slot is rounded to a Python `int` before storage, so it never
# collides with the `step=1`/`min_value`/`max_value` ints those widgets
# declare — `st.number_input` requires all of `value`, `min_value`,
# `max_value`, and `step` to share one numeric type (all `int` or all
# `float`), and silently mixing float session state with int bounds raises
# `StreamlitAPIException`.
INT_FIELDS = set(config.CATEGORICAL_COLS) | {
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (without evaluations)",
    "Age at enrollment",
    "Application order",
}


def _cast(col: str, value):
    return int(round(value)) if col in INT_FIELDS else float(value)


def _field_key(col: str) -> str:
    return f"field_{col}"


def _init_field_state(col: str, default_value):
    key = _field_key(col)
    if key not in st.session_state:
        st.session_state[key] = _cast(col, default_value)


def _load_random_student():
    train_df = shared.get_train_df()
    row = train_df.sample(1).iloc[0]
    for col in config.RAW_FEATURE_COLS:
        st.session_state[_field_key(col)] = _cast(col, row[col])
    st.session_state["actual_outcome"] = row[config.TARGET_COL]
    st.session_state["loaded_a_student"] = True


def _reset_to_average():
    defaults = shared.get_default_row()
    for col in config.RAW_FEATURE_COLS:
        st.session_state[_field_key(col)] = _cast(col, defaults[col])
    st.session_state.pop("actual_outcome", None)
    st.session_state["loaded_a_student"] = False


def _numeric_input(col: str, defaults: dict, **kwargs):
    _init_field_state(col, defaults[col])
    label = FIELD_LABELS.get(col, col)
    st.number_input(label, key=_field_key(col), **kwargs)


def _binary_select(col: str, defaults: dict):
    _init_field_state(col, defaults[col])
    codes = [code for _, code in BINARY_LABELS[col]]
    labels_by_code = {code: label for label, code in BINARY_LABELS[col]}
    # `key` holds the raw code (0/1) directly — format_func only changes the
    # displayed label, not what's stored in session_state — so this widget's
    # state is already the exact value the model row needs, and the "Load a
    # random student" callback can set it directly like every other field.
    st.selectbox(col, options=codes, format_func=lambda c: labels_by_code[c], key=_field_key(col))


def _code_input(col: str, defaults: dict, max_value: int):
    _init_field_state(col, defaults[col])
    st.number_input(f"{col} (code)", key=_field_key(col), min_value=0, max_value=max_value, step=1)


def render():
    st.title("🎯 Predict a Student's Outcome")
    st.caption(
        "Predicts whether a student is likely to drop out, stay enrolled, or "
        "graduate. Start from a real student, or an average one, then tweak "
        "the fields that matter most."
    )

    try:
        pipeline = shared.get_pipeline()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    defaults = shared.get_default_row()

    col1, col2 = st.columns(2)
    with col1:
        st.button("🎲 Load a random real student", on_click=_load_random_student, use_container_width=True)
    with col2:
        st.button("↺ Reset to dataset average", on_click=_reset_to_average, use_container_width=True)

    if st.session_state.get("loaded_a_student"):
        st.info("Loaded a real student from the training set. Their actual outcome is revealed after you predict.")

    st.subheader("The fields that matter most")
    st.caption("Promoted to the top based on this project's SHAP analysis — see Model Insights.")

    c1, c2, c3 = st.columns(3)
    with c1:
        _numeric_input("Curricular units 1st sem (enrolled)", defaults, min_value=0, max_value=30, step=1)
        _numeric_input("Curricular units 2nd sem (enrolled)", defaults, min_value=0, max_value=30, step=1)
    with c2:
        _numeric_input("Curricular units 1st sem (approved)", defaults, min_value=0, max_value=30, step=1)
        _numeric_input("Curricular units 2nd sem (approved)", defaults, min_value=0, max_value=30, step=1)
    with c3:
        _numeric_input("Curricular units 1st sem (grade)", defaults, min_value=0.0, max_value=20.0, step=0.1)
        _numeric_input("Curricular units 2nd sem (grade)", defaults, min_value=0.0, max_value=20.0, step=0.1)

    c4, c5, c6 = st.columns(3)
    with c4:
        _binary_select("Tuition fees up to date", defaults)
    with c5:
        _binary_select("Scholarship holder", defaults)
    with c6:
        _numeric_input("Age at enrollment", defaults, min_value=16, max_value=70, step=1)

    with st.expander("Other details (demographics, admission record, family background, macroeconomic context)"):
        st.markdown("**Demographics**")
        d1, d2, d3 = st.columns(3)
        with d1:
            _code_input("Marital status", defaults, 6)
            _binary_select("Gender", defaults)
        with d2:
            _code_input("Nacionality", defaults, 109)
            _binary_select("International", defaults)
        with d3:
            _binary_select("Displaced", defaults)
            _binary_select("Educational special needs", defaults)

        st.markdown("**Admission record**")
        a1, a2, a3 = st.columns(3)
        with a1:
            _code_input("Application mode", defaults, 60)
            _code_input("Application order", defaults, 9)
        with a2:
            _code_input("Course", defaults, 10000)
            _binary_select("Daytime/evening attendance", defaults)
        with a3:
            _code_input("Previous qualification", defaults, 50)
            _numeric_input("Previous qualification (grade)", defaults, min_value=0.0, max_value=200.0, step=1.0)
        _numeric_input("Admission grade", defaults, min_value=0.0, max_value=200.0, step=1.0)
        _binary_select("Debtor", defaults)

        st.markdown("**Family background**")
        f1, f2 = st.columns(2)
        with f1:
            _code_input("Mother's qualification", defaults, 50)
            _code_input("Mother's occupation", defaults, 200)
        with f2:
            _code_input("Father's qualification", defaults, 50)
            _code_input("Father's occupation", defaults, 200)

        st.markdown("**Remaining curricular detail**")
        r1, r2 = st.columns(2)
        with r1:
            _numeric_input("Curricular units 1st sem (credited)", defaults, min_value=0, max_value=30, step=1)
            _numeric_input("Curricular units 1st sem (evaluations)", defaults, min_value=0, max_value=30, step=1)
            _numeric_input("Curricular units 1st sem (without evaluations)", defaults, min_value=0, max_value=30, step=1)
        with r2:
            _numeric_input("Curricular units 2nd sem (credited)", defaults, min_value=0, max_value=30, step=1)
            _numeric_input("Curricular units 2nd sem (evaluations)", defaults, min_value=0, max_value=30, step=1)
            _numeric_input("Curricular units 2nd sem (without evaluations)", defaults, min_value=0, max_value=30, step=1)

        st.markdown("**Macroeconomic context (at enrollment)**")
        m1, m2, m3 = st.columns(3)
        with m1:
            _numeric_input("Unemployment rate", defaults, min_value=0.0, max_value=30.0, step=0.1)
        with m2:
            _numeric_input("Inflation rate", defaults, min_value=-5.0, max_value=15.0, step=0.1)
        with m3:
            _numeric_input("GDP", defaults, min_value=-10.0, max_value=10.0, step=0.1)

    if st.button("Predict outcome", type="primary"):
        row = {col: st.session_state[_field_key(col)] for col in config.RAW_FEATURE_COLS}
        X = pd.DataFrame([row])[config.RAW_FEATURE_COLS]

        prediction = pipeline.predict(X)[0]
        proba = pipeline.predict_proba(X)[0]
        classes = pipeline.named_steps["model"].classes_

        actual = st.session_state.get("actual_outcome")
        if actual is not None:
            match = "✅ matches the model" if actual == prediction else "❌ differs from the model"
            st.subheader(f"Prediction: **{prediction}**  |  Actual outcome: **{actual}** ({match})")
        else:
            st.subheader(f"Prediction: **{prediction}**")

        proba_df = pd.DataFrame({"Outcome": classes, "Probability": proba}).sort_values(
            "Probability", ascending=False
        )
        st.bar_chart(proba_df.set_index("Outcome"))
        st.dataframe(proba_df, hide_index=True)
