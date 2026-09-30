"""Model Insights page: what the model comparison found, and what the final
model actually learned (SHAP). Numbers here mirror `report/report.qmd`.
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from . import shared
from academic_success.interpretability import top_shap_features

PALETTE = ["#2c5cc5", "#5b8def", "#8fb4f2", "#e07a5f", "#81b29a"]

# Transcribed from notebooks/01_eda_and_modeling.ipynb Sections 9-11 (run
# against the real downloaded competition data) — see report/report.qmd for
# the full writeup. Hard-coded rather than recomputed on every page load: an
# 8-model x 5-fold sweep plus a stacking ensemble takes several minutes, for
# no new information on every Streamlit rerun.
SWEEP_RESULTS = pd.DataFrame([
    {"model": "Gradient Boosting", "f1_macro": 0.7846},
    {"model": "HistGradientBoosting", "f1_macro": 0.7839},
    {"model": "Logistic Regression", "f1_macro": 0.7806},
    {"model": "LightGBM", "f1_macro": 0.7790},
    {"model": "CatBoost", "f1_macro": 0.7783},
    {"model": "Random Forest", "f1_macro": 0.7770},
    {"model": "XGBoost", "f1_macro": 0.7727},
    {"model": "AdaBoost", "f1_macro": 0.7692},
]).sort_values("f1_macro")

RESAMPLE_RESULTS = pd.DataFrame([
    {"resample": "None", "f1_macro": 0.7846},
    {"resample": "ADASYN", "f1_macro": 0.7829},
])

STACKING_RESULTS = pd.DataFrame([
    {"model": "Best single model\n(Gradient Boosting)", "f1_macro": 0.7772},
    {"model": "Stacking ensemble", "f1_macro": 0.7742},
])


def render():
    st.title("🧠 Model Insights")
    st.caption(
        "What the model comparison found, and — via SHAP — what the final "
        "saved model actually learned from the data."
    )

    st.subheader("Model sweep (15,000-row subsample, 5-fold CV)")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.barh(SWEEP_RESULTS["model"], SWEEP_RESULTS["f1_macro"], color=PALETTE[0])
    bars[-1].set_color(PALETTE[3])
    ax.set_xlim(0.76, 0.79)
    ax.set_xlabel("CV macro F1")
    for bar, value in zip(bars, SWEEP_RESULTS["f1_macro"]):
        ax.text(value + 0.0006, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=9)
    st.pyplot(fig)
    st.success(
        "The full 76,518-row training set reaches **0.7949 macro-F1** with the "
        "saved default (HistGradientBoosting) — beating every model above. "
        "More training data mattered more than which model family was used."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Class imbalance: ADASYN")
        fig2, ax2 = plt.subplots(figsize=(4.5, 3.3))
        bars2 = ax2.bar(RESAMPLE_RESULTS["resample"], RESAMPLE_RESULTS["f1_macro"], color=[PALETTE[0], PALETTE[3]], width=0.5)
        ax2.set_ylim(0.77, 0.79)
        ax2.set_ylabel("CV macro F1")
        for bar, value in zip(bars2, RESAMPLE_RESULTS["f1_macro"]):
            ax2.text(bar.get_x() + bar.get_width() / 2, value + 0.0008, f"{value:.4f}", ha="center", fontsize=9)
        st.pyplot(fig2)
    with c2:
        st.subheader("Ensembling: stacking")
        fig3, ax3 = plt.subplots(figsize=(4.5, 3.3))
        bars3 = ax3.bar(STACKING_RESULTS["model"], STACKING_RESULTS["f1_macro"], color=[PALETTE[4], PALETTE[3]], width=0.5)
        ax3.set_ylim(0.76, 0.79)
        ax3.set_ylabel("Holdout macro F1")
        for bar, value in zip(bars3, STACKING_RESULTS["f1_macro"]):
            ax3.text(bar.get_x() + bar.get_width() / 2, value + 0.0008, f"{value:.4f}", ha="center", fontsize=9)
        st.pyplot(fig3)

    st.warning(
        "Neither ADASYN nor stacking beat the simpler alternative here — two "
        "honest negative results, which is exactly why the saved model stays a "
        "single HistGradientBoostingClassifier."
    )

    st.subheader("What does the model actually rely on? (SHAP)")
    st.caption("Computed live from the saved model — first load takes a few seconds.")
    explanation, _ = shared.get_shap_explanation(sample_size=300)
    top = top_shap_features(explanation, top_n=10).sort_values("mean_abs_shap")

    fig4, ax4 = plt.subplots(figsize=(7, 4.5))
    is_engineered = top["feature"].str.contains("approval_rate|total_units|grade_trend|numeric__no_evaluation")
    colors = [PALETTE[3] if e else PALETTE[0] for e in is_engineered]
    ax4.barh(top["feature"], top["mean_abs_shap"], color=colors)
    ax4.set_xlabel("Mean |SHAP value|")
    ax4.set_title("Top 10 features (orange = engineered)")
    st.pyplot(fig4)
