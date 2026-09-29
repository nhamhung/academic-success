"""Dataset Overview page: the "what does this data look like" walkthrough."""

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from . import shared
from academic_success import config

PALETTE = ["#2c5cc5", "#5b8def", "#8fb4f2", "#e07a5f", "#81b29a"]


def render():
    st.title("📊 Dataset Overview")
    st.caption(
        "Kaggle's Playground Series S4E6 — a synthetic-augmented version of the "
        "UCI 'Predict Students' Dropout and Academic Success' dataset."
    )

    train_df = shared.get_train_df()

    c1, c2, c3 = st.columns(3)
    c1.metric("Students", f"{len(train_df):,}")
    c2.metric("Raw features", len(config.RAW_FEATURE_COLS))
    c3.metric("Missing values", int(train_df[config.RAW_FEATURE_COLS].isna().sum().sum()))

    st.subheader("Target class balance")
    counts = train_df[config.TARGET_COL].value_counts().reindex(config.TARGET_CLASSES)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(counts.index, counts.values, color=PALETTE[:3], width=0.6)
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value + 600,
            f"{value:,}\n({value / counts.sum():.0%})", ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylim(0, counts.max() * 1.25)
    ax.set_ylabel("Students")
    sns.despine(ax=ax)
    st.pyplot(fig)
    st.caption(
        "Moderately imbalanced — this is why every model comparison in this "
        "project uses macro-averaged F1 rather than raw accuracy."
    )

    st.subheader("Key numeric distributions")
    numeric_choice = st.selectbox(
        "Choose a numeric feature to plot",
        options=["Age at enrollment", "Admission grade", "Previous qualification (grade)",
                 "Unemployment rate", "Inflation rate", "GDP"],
    )
    fig2, ax2 = plt.subplots(figsize=(6.5, 3.5))
    sns.histplot(train_df[numeric_choice], bins=30, color=PALETTE[0], ax=ax2)
    ax2.set_title(f"Distribution of {numeric_choice}")
    sns.despine(ax=ax2)
    st.pyplot(fig2)

    st.subheader("Categorical cardinality")
    st.caption(
        "Several columns have a long tail of near-unique codes — this is why the "
        "project's feature pipeline groups rare categories before one-hot "
        "encoding (see Feature Engineering in `report/report.qmd`)."
    )
    cardinality = (
        train_df[config.CATEGORICAL_COLS].nunique().sort_values(ascending=False).head(10)
    )
    fig3, ax3 = plt.subplots(figsize=(6.5, 4))
    ax3.barh(cardinality.index[::-1], cardinality.values[::-1], color=PALETTE[1])
    ax3.set_xlabel("Distinct category codes")
    sns.despine(ax=ax3)
    st.pyplot(fig3)
