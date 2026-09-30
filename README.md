# Academic Success Predictor

A worked, end-to-end data science project built around Kaggle's
[Playground Series S4E6 — "Classification with an Academic Success Dataset"](https://www.kaggle.com/competitions/playground-series-s4e6):
predicting whether a student will **Dropout**, stay **Enrolled**, or
**Graduate**, from their demographics, academic history, and first/second
semester performance.

It's designed for people who've completed Kaggle Learn's tabular-data
microcourses (**Pandas**, **Data Cleaning**, **Intro to Machine Learning**,
**Intermediate Machine Learning**, **Feature Engineering**) and want one
project that exercises all of those skills together — from a raw dataset to
a notebook, a research writeup, a usable app, and a real Kaggle submission.

## What's here

| Deliverable | Where |
|---|---|
| A well-documented notebook building the model, applying good practices | `notebooks/01_eda_and_modeling.ipynb` |
| A multi-page Streamlit app so a non-technical user can try the model and explore the data/findings | `app/streamlit_app.py` + `app/pages_src/` (+ `app/Dockerfile`) |
| A research-style writeup of key concepts and results | `report/report.qmd` |
| A script that produces a Kaggle-submittable `submission.csv` | `scripts/make_submission.py` |

All four share one preprocessing/training pipeline in `src/academic_success/`,
so the notebook, the app, and the submission script can never quietly drift
apart from each other — they all load the same trained
`models/model.joblib`.

Also included: a sweep across 8 model families (Logistic Regression, Random
Forest, AdaBoost, Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM,
CatBoost — see `src/academic_success/model.py`), ADASYN class-imbalance
resampling, a stacking ensemble, and SHAP-based model interpretability — all
demonstrated and evaluated (with honest results, including where they *don't*
help) in the notebook and `report/report.qmd`.

## Project layout

```
data/raw/               # downloaded Kaggle CSVs (gitignored — see below)
data/processed/          # any cached intermediate data (gitignored)
notebooks/               # the main EDA + modeling notebook
src/academic_success/    # shared config, data loading, feature engineering, model code, SHAP interpretability
models/                  # trained pipeline artifact (model.joblib)
app/                     # Streamlit app (multi-page, app/pages_src/) + Dockerfile
scripts/                 # train.py, make_submission.py
report/                  # Quarto research writeup
tests/                   # pytest tests for the feature engineering
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data

This project doesn't commit Kaggle's competition data (redistribution isn't
permitted). Download it yourself with the
[Kaggle API](https://www.kaggle.com/docs/api) (`pip install kaggle`, then put
your `kaggle.json` API token in `~/.kaggle/`):

```bash
kaggle competitions download -c playground-series-s4e6 -p data/raw
unzip -o data/raw/playground-series-s4e6.zip -d data/raw
```

This produces `data/raw/train.csv`, `data/raw/test.csv`, and
`data/raw/sample_submission.csv`.

## Run the notebook

```bash
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```

Walks through EDA, data cleaning, feature engineering, model comparison, and
cross-validation, and saves the final trained pipeline to
`models/model.joblib`.

## Train from the command line

Equivalent to the notebook's modeling steps, without the notebook:

```bash
python scripts/train.py                        # fast default: HistGradientBoosting
python scripts/train.py --model random_forest   # any model in MODEL_FACTORIES (see --help)
python scripts/train.py --resample              # apply ADASYN for class imbalance
python scripts/train.py --stacking              # stacking ensemble (slower, strongest)
python scripts/train.py --help                  # full list of options
```

The notebook's Sections 9–11 found that, on this dataset, plain
HistGradientBoosting already matches or beats both ADASYN resampling and the
stacking ensemble — which is why it stays the default. See
`report/report.qmd` for the actual numbers behind that call.

## Generate a Kaggle submission

```bash
python scripts/make_submission.py
kaggle competitions submit -c playground-series-s4e6 -f submission.csv -m "first submission"
```

## Run the app

A 4-page app: **Predict** (load a real student or start from the dataset
average, then tweak the fields that matter most and see the prediction
update), **Dataset Overview**, **Feature Engineering** (interactive
walkthrough of why the engineered features exist), and **Model Insights**
(model comparison results + live SHAP).

Locally:

```bash
streamlit run app/streamlit_app.py
```

Or in Docker (from the project root, after training a model):

```bash
docker build -t academic-success-app -f app/Dockerfile .
docker run -p 8501:8501 academic-success-app
```

Then open http://localhost:8501.

## Render the research writeup

Requires [Quarto](https://quarto.org/docs/get-started/) installed separately
(it's not a Python package):

```bash
quarto render report/report.qmd
```

## Run the tests

```bash
pytest tests/
```

These test the feature engineering logic directly with synthetic data — no
Kaggle download needed.

## Deploy

The Streamlit app starts immediately from a bundled 1,500-row stratified sample sourced from Kaggle. Set `USE_FULL_KAGGLE_DATA=true` to fetch and use the complete dataset through the Kaggle API; configure either `KAGGLE_API_TOKEN` or a `[kaggle]` secrets section containing username and key.

- Repository: <https://github.com/nhamhhung/academic-success>
- Report: <https://nhamhung.github.io/academic-success/>
- Streamlit: <https://academic-success.streamlit.app>
- Fork setup: [docs/SETUP_AND_DEPLOYMENT.md](docs/SETUP_AND_DEPLOYMENT.md)
