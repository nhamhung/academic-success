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

## Prerequisites

Install once, before Setup below:

| Dependency | Why | Install |
|---|---|---|
| **Python 3.12** | This project's `.venv` is built against 3.12 — a different version may resolve incompatible package versions from `requirements.txt`. | [python.org/downloads](https://www.python.org/downloads/) or a version manager (e.g. `pyenv install 3.12`) |
| **Quarto** | Renders `report/report.qmd` — a standalone binary, not a Python package, so `pip install` never gets it. | [quarto.org/docs/get-started](https://quarto.org/docs/get-started/) |
| **Kaggle API token** | Needed only for the complete dataset; the default Streamlit app uses a bundled sample, and `pytest` uses synthetic data. | Kaggle account → **Account → Create New API Token** → save the downloaded file as `~/.kaggle/kaggle.json` (`%USERPROFILE%\.kaggle\kaggle.json` on Windows). See the [Kaggle API docs](https://www.kaggle.com/docs/api). |
| **Docker** (optional) | Only if you want to run the app in its pre-baked container instead of `streamlit run`. | [docker.com/get-started](https://www.docker.com/get-started/) |

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

## Making changes

`src/academic_success/` is the single source of truth — `config.py`
(paths, schema, constants), `data.py` (loading/splitting), `features.py`
(feature engineering), `model.py` (pipelines, training, evaluation),
`interpretability.py` (SHAP). The notebook, the app, and
`scripts/make_submission.py` all import from here; nothing re-derives
logic locally, so a change here propagates everywhere automatically.

The edit loop:

```bash
# 1. Edit src/academic_success/*.py

# 2. Check it against the test suite (fast, synthetic data, no download needed)
pytest tests/

# 3. Retrain, so models/model.joblib reflects your change
python scripts/train.py   # or --model <name>, --stacking, etc. — see --help
```

`models/model.joblib` is what the notebook, the app, and the report all
load — retraining is the one step that makes a model-code change visible
everywhere else.

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
python3.12 -m venv .venv          # use the 3.12 interpreter specifically
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data

The complete Kaggle competition data is not committed (redistribution isn't
permitted). A bundled 1,500-row sample keeps the Streamlit app responsive.
Download the full training and test data with the
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

`report/report.qmd` expects a Jupyter kernel named `academic-success` (its
`jupyter:` front-matter key) — register it once from the activated venv
before rendering:

```bash
python -m ipykernel install --user --name academic-success
quarto render report/report.qmd
```

This regenerates both `report/report.html` and `report/report.pdf` (PDF
needs a LaTeX distribution — if you don't have one, run
`quarto install tinytex` once). Render just one format when you don't need
both:

```bash
quarto render report/report.qmd --to html
quarto render report/report.qmd --to pdf
```

Live-preview while editing (auto-rerenders on save):

```bash
quarto preview report/report.qmd
```

A `.qmd` file is Markdown prose plus fenced Python code chunks
(` ```{python} `/` ``` `), executed top to bottom by the kernel above, same
as a notebook cell. Common per-chunk options (a `#|` comment, first line of
the chunk): `#| echo: false` (hide this chunk's source code),
`#| output: false` (suppress its output, e.g. a setup/import cell),
`#| label: fig-foo` + `#| fig-cap: "..."` (name and caption a figure for
cross-referencing). The [Quarto VS Code
extension](https://marketplace.visualstudio.com/items?itemName=quarto.quarto)
adds syntax highlighting and a one-click Render button if you're doing more
than a one-line edit.

Troubleshooting:

| Symptom | Likely cause |
|---|---|
| `Jupyter engine failed ... kernel not found` | The `ipykernel install --name academic-success` step above hasn't been run yet. |
| `ModuleNotFoundError` inside a code chunk | `quarto render` runs with its working directory set to `report/`, not the project root — check the chunk's `sys.path.insert(0, "../src")` points at the right relative path. |
| Output looks stale after editing | Force a clean re-run: `quarto render report/report.qmd --execute-daemon-restart`. |
| PDF render fails, HTML succeeds | Missing LaTeX — run `quarto install tinytex` once, then retry. |

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
