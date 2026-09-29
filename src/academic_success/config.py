"""Paths, constants, and column schema for the Academic Success dataset.

Dataset: Kaggle Playground Series S4E6, "Classification with an Academic
Success Dataset" (https://www.kaggle.com/competitions/playground-series-s4e6),
itself a synthetic-augmented version of the UCI "Predict Students' Dropout
and Academic Success" dataset.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_CSV = DATA_RAW_DIR / "train.csv"
TEST_CSV = DATA_RAW_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = DATA_RAW_DIR / "sample_submission.csv"
MODEL_PATH = MODELS_DIR / "model.joblib"

# --- Kaggle competition ---------------------------------------------------

KAGGLE_COMPETITION = "playground-series-s4e6"

# --- Target ----------------------------------------------------------------

ID_COL = "id"
TARGET_COL = "Target"
TARGET_CLASSES = ["Dropout", "Enrolled", "Graduate"]

RANDOM_SEED = 42

# --- Column schema -----------------------------------------------------
# Every raw column that isn't the id or target falls into exactly one of
# these two groups. Columns are encoded as integer category codes in the
# source data (see the UCI dataset documentation for the code legend) but
# behave as categorical, not ordinal, features.

CATEGORICAL_COLS = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

NUMERIC_COLS = [
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

RAW_FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS
