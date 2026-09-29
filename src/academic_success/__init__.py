"""Shared preprocessing/training pipeline for the Academic Success project.

Every entry point (notebook, training script, submission script, Streamlit
app) imports from this package instead of re-implementing feature
engineering, so train-time and serve-time transformations can never drift
apart.
"""
