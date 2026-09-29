"""Feature Engineering page: a visual walkthrough of *why* the engineered
trajectory features exist, mirroring `report/report.qmd`'s rationale but
interactive.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from . import shared
from academic_success import config
from academic_success.features import ENGINEERED_NUMERIC_COLS

PALETTE = ["#2c5cc5", "#5b8def", "#8fb4f2"]

FEATURE_EXPLANATIONS = {
    "approval_rate_1st_sem": (
        "Fraction of enrolled 1st-semester units actually passed. Two students "
        "who both 'enroll in 6 units' look identical on the raw columns, but "
        "one who passes all 6 and one who passes 1 are very different risk "
        "profiles — the approval *rate* captures that, raw counts don't."
    ),
    "approval_rate_2nd_sem": (
        "Same idea as the 1st-semester rate, for the 2nd semester. This is the "
        "single most important feature to the final model — see Model Insights."
    ),
    "total_units_approved": "Cumulative units passed across both semesters — a coarser, more stable signal than either semester alone.",
    "total_units_enrolled": "Cumulative units enrolled across both semesters.",
    "grade_trend": (
        "Change in average grade from the 1st to the 2nd semester. A student "
        "improving over time is a different risk profile from one declining, "
        "even at the same average grade."
    ),
    "no_evaluation_ratio": (
        "Fraction of units never evaluated at all (as opposed to evaluated and "
        "failed) — a disengagement signal distinct from academic failure."
    ),
    "max_parental_qualification": (
        "The higher of the mother's and father's qualification codes. Caveat: "
        "these codes are categorical labels, not a ranked scale, so this is a "
        "common but imperfect Kaggle proxy, not a true ordinal measurement."
    ),
}


def render():
    st.title("🔧 Feature Engineering")
    st.caption(
        "Why derive new features instead of feeding the raw columns straight "
        "to a model? Pick one below to see how cleanly it separates the three "
        "outcomes."
    )

    engineered_df = shared.get_engineered_df()

    feature_choice = st.selectbox(
        "Engineered feature", options=ENGINEERED_NUMERIC_COLS, index=1  # default: approval_rate_2nd_sem
    )
    st.info(FEATURE_EXPLANATIONS[feature_choice])

    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.boxplot(
        x=engineered_df[config.TARGET_COL], y=engineered_df[feature_choice],
        hue=engineered_df[config.TARGET_COL], legend=False,
        order=config.TARGET_CLASSES, palette=PALETTE, ax=ax, width=0.5,
    )
    ax.set_xlabel("")
    ax.set_title(f"{feature_choice} by outcome")
    sns.despine(ax=ax)
    st.pyplot(fig)

    st.subheader("Rare-category grouping")
    st.caption(
        "Several categorical columns (Course, occupation codes, nationality) "
        "have a long tail of near-unique codes. `RareCategoryGrouper` collapses "
        "any category below 1% frequency into a shared sentinel *before* "
        "one-hot encoding, learned from training data only."
    )
    col_choice = st.selectbox(
        "Column",
        options=["Nacionality", "Mother's qualification", "Father's qualification",
                 "Mother's occupation", "Father's occupation", "Application mode"],
    )
    train_df = shared.get_train_df()
    freqs = train_df[col_choice].value_counts(normalize=True)
    rare_share = (freqs < 0.01).sum() / len(freqs)
    c1, c2 = st.columns(2)
    c1.metric("Distinct codes", len(freqs))
    c2.metric("Below 1% frequency", f"{(freqs < 0.01).sum()} ({rare_share:.0%})")
