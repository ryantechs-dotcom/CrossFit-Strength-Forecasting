# CrossFit ML Project

## Overview

This project implements an end-to-end, reproducible machine learning pipeline
for predicting CrossFit athlete overall competition strength (`total_lift =
deadlift + candj + snatch + backsq`), from demographic, lifestyle, and
experience features.

`total_lift`'s direct components (`deadlift`, `candj`, `snatch`, `backsq`)
are explicitly excluded from the model's feature set to prevent target
leakage.

The workflow combines three tools, each responsible for a different kind of
versioning/reproducibility:

| Tool       | Responsible for                                                   |
|------------|---------------------------------------------------------------------|
| **DVC**    | Data + pipeline stage versioning, reproducible re-execution         |
| **Feast**  | Feature store: feature definitions, feature *versioning*, retrieval |
| **MLflow** | Experiment tracking, model registry, model versioning, artifacts    |

> **Naming note:** `run_all.py` tags dataset snapshots in git as
> `dataset-raw` and `dataset-processed` (raw ingest vs. cleaned data).
> These are **not** the same thing as the Feast `v1`/`v2` **feature
> versions** described under "Feature Versioning" below — the two are
> independent versioning axes handled by different tools. Throughout
> this document, `v1`/`v2` refers exclusively to Feast feature
> versions; dataset snapshots are always referred to as
> `dataset-raw`/`dataset-processed`.

---

## Why This Stack (MLOps Platform Selection)

**MLflow** was chosen as the primary experiment-tracking / model-registry
platform because:

- It's framework-agnostic (works directly with `XGBRegressor` via
  `mlflow.xgboost`) with no vendor lock-in.
- It provides a **local, file/SQLite-backed Model Registry** — no external
  server needed, which fits a single-developer academic project running
  entirely on a local machine.
- Runs, params, metrics, model artifacts, signatures, and tags are all
  queryable from the same UI (`mlflow ui`), which made it easy to verify that
  each of the four required experiment combinations were tracked correctly
  with distinguishable feature versions.

**DVC** was chosen for pipeline orchestration and data versioning because its
`dvc.yaml`/`dvc.lock` stage graph gives a single, declarative source of truth
for how raw data becomes a trained model, and lets any stage be reproduced in
isolation (`dvc repro <stage>`) or as a full chain (`dvc repro compare`).

**Feast** was chosen as the feature store because it cleanly separates
*feature definition* (versioned, declarative feature views) from
*feature consumption* (a single `get_historical_features`/retrieval call in
`split_data.py`), which is exactly the separation the "Feature Store
Integration" and "Feature Versioning" rubric items ask for.

---

## Project Structure

```
athletes_dvc_project/
├── data/
│   ├── raw/crossfit.csv                  # ingested, unmodified
│   ├── processed/
│   │   ├── crossfit.csv                  # cleaned/transformed
│   │   └── crossfit_features.csv         # engineered features
│   ├── feature_store/
│   │   ├── crossfit_features.parquet     # Feast offline source
│   │   ├── online_store.db               # Feast online store
│   │   └── registry.db                   # Feast feature registry
│   ├── train/train.csv
│   └── test/test.csv
│
├── models/
│   └── model.pkl                         # local copy of latest trained model
│
├── mlruns/                                # MLflow tracking + model registry (local)
├── mlflow.db                              # MLflow SQLite backend store
│
├── outputs/eda/                           # v1 vs v2 EDA plots
├── reports/
│   ├── evaluation.txt
│   ├── model_comparison.txt
│   ├── split_documentation.txt
│   └── pylint_report.txt
│
├── src/
│   ├── ingest.py                # data ingestion
│   ├── preprocess.py            # cleaning/transformation
│   ├── eda.py                   # exploratory analysis
│   ├── feature_definitions.py   # Feast feature views (v1, v2)
│   ├── feature_store_setup.py   # Feast apply/materialize
│   ├── feature_engineering.py   # derived feature creation
│   ├── split_data.py            # Feast retrieval + train/test split
│   ├── train.py                 # model training + MLflow logging/registration
│   ├── evaluate.py              # evaluation + MLflow metric logging
│   ├── run_experiments.py       # runs the 2x2 experiment grid
│   ├── compare_versions.py      # v1 vs v2 comparison report
│   ├── automl_pycaret.py        # Assignment #3: PyCaret AutoML (all + top features)
│   ├── automl_h2o.py            # Assignment #3: H2O AutoML (all + top features)
│   └── automl_compare.py        # Assignment #3: cross-platform + baseline comparison
│
├── feature_store.yaml
├── dvc.yaml / dvc.lock
├── params.yaml
├── random_seed.py
├── run_all.py
└── requirements.txt
```

---

## Setup

```bash
git clone <your-repo-url>
cd athletes_dvc_project

python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

pip install -r requirements.txt
```

MLflow's local tracking store lives in `mlflow.db` (SQLite) with artifacts in
`mlruns/`. To view it:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## Running the Full Pipeline

```bash
python run_all.py
```

This runs ingestion → preprocessing → EDA → feature engineering → Feast
retrieval/split → training → evaluation → comparison, end to end, from a
clean environment.

Or via DVC directly:

```bash
dvc repro compare
```

which resolves the full upstream stage graph (git tags `dataset-raw` /
`dataset-processed` mark the dataset snapshots along the way):

```
ingest → preprocess → feature_engineering → split_data → train → evaluate → compare
```

Individual stages can be run in isolation, e.g. `dvc repro train`.

---

## Feature Store Integration (Feast)

Feature definitions live in `src/feature_definitions.py` and are applied via
`src/feature_store_setup.py` against `feature_store.yaml`. At training time,
`split_data.py` retrieves the active feature version's features directly
from Feast rather than reading engineered columns straight off a CSV:

```
[split_data] Retrieved 5 Feast features using Feature Version V1 (crossfit_fs_v1)
[split_data] Retrieved 12 Feast features using Feature Version V2 (crossfit_fs_v2)
```

This is a real retrieval step in the pipeline (not just unused config) — the
number and identity of columns fed into training changes depending on which
feature view is requested.

## Feature Versioning

Two Feast feature views are defined in `src/feature_definitions.py` and
stored in the Feast registry (`data/feature_store/registry.db`), served via
two `FeatureService`s:

| Version | Feast View / Service                          | # Features | Fields |
|---------|------------------------------------------------|------------|--------|
| v1      | `crossfit_athlete_features_v1` / `crossfit_fs_v1` | 5        | `gender`, `age`, `height`, `weight`, `howlong` — baseline demographic + training-history features |
| v2      | `crossfit_athlete_features_v2` / `crossfit_fs_v2` | 12       | All 5 of v1, **plus**: `total_lift`, `background__football`, `background__weightlifting`, `experience__beginner`, `experience__advanced`, `schedule__5_days`, `eat__paleo` |

**v1 → v2 difference:** v2 is a strict superset of v1. It adds `total_lift`
(see design note below) plus six engineered survey-derived binary/categorical
features covering training background, experience level, weekly training
schedule, and diet — none of which v1 exposes. The feature-engineering
pipeline guarantees all v2 columns exist in the parquet source even when a
given category isn't present in a particular row, so retrieval is
consistent across the dataset.

> **Design note — `total_lift` in the v2 schema:** `total_lift` is stored as
> a Feast field for lineage/convenience, but it is also the training
> **target**. It is never used as a model input: `train.py`'s
> `prepare_xy()` explicitly drops `total_lift` (and its raw components
> `deadlift`/`candj`/`snatch`/`backsq`) from `x` before fitting, regardless
> of which feature version is active. This is intentional, not a leakage
> bug — but it's called out here explicitly since including a target column
> in a feature view's schema is unusual and worth a reviewer's attention.

The active feature version is controlled via `params.yaml` → `features.version`,
and is logged as both an MLflow **run param** and, for the trained model, an
MLflow **model version tag** (`feature_version`), so it's visible both on the
Experiments page and in the Model Registry.

---

## Experimentation and Model Training

`src/run_experiments.py` runs the required 2×2 grid — two feature versions ×
two hyperparameter configurations, same algorithm (XGBoost) throughout, no
AutoML/automated model selection:

| Run                | Feature Version | n_estimators | Algorithm |
|---------------------|------------------|--------------|-----------|
| `v1_xgboost_100`    | v1 (crossfit_fs_v1) | 100 | XGBoost |
| `v1_xgboost_200`    | v1 (crossfit_fs_v1) | 200 | XGBoost |
| `v2_xgboost_100`    | v2 (crossfit_fs_v2) | 100 | XGBoost |
| `v2_xgboost_200`    | v2 (crossfit_fs_v2) | 200 | XGBoost |

Each run:

- Logs `feature_version`, `model_type`, `random_seed`, `n_estimators`,
  `max_depth`, `learning_rate`, `feature_count`, and `train_rows` as MLflow
  **params**.
- Trains an `XGBRegressor` and logs it via `mlflow.xgboost.log_model(...)`
  with an inferred **signature** and **input example**, and registers it
  under `crossfit_total_lift_model` in the Model Registry.
- Tags the resulting **model version** (not just the run) with
  `feature_version`, `n_estimators`, and `feature_count`, so feature-version
  differences are visible directly in the registry, not just per-run.
- Is evaluated in a separate `evaluate.py` run against the held-out test
  split, logging `RMSE`, `MAE`, and `R2` as MLflow metrics and writing
  `reports/evaluation.txt`.

To view results:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Runs, params, feature versions, metrics, and registered model versions
(with signatures/tags) are all visible from there.

> **Important — local files vs. MLflow history:** `models/model_{version}.pkl`
> and `reports/evaluation_{version}.txt` are keyed only by **feature
> version** (`v1`/`v2`), not by hyperparameter config. Since
> `run_experiments.py` runs both hyperparameter configs for a given
> feature version sequentially, the *small* config's local model/report
> file is overwritten by the *large* config's file for that version. So
> at any point in time, the repo's local files reflect only 2 of the 4
> experiments (the last hyperparameter config run per feature version)
> — this is expected, not a bug, and matches what `dvc.yaml` declares as
> stage outputs.
>
> All **4** experiments' full parameters, feature versions, and metrics
> are preserved permanently and independently in MLflow (as 4 distinct
> runs — `v1_xgboost_100`, `v1_xgboost_200`, `v2_xgboost_100`,
> `v2_xgboost_200` — each with its own params/metrics/registered model
> version). MLflow, not the local `models/`/`reports/` files, is the
> authoritative record of all 4 experiment combinations required by
> the assignment. Run `mlflow ui --backend-store-uri sqlite:///mlflow.db`
> and filter/sort by the `feature_version`, `n_estimators`, `max_depth`,
> and `learning_rate` params to see all 4 side by side.

---

## Dataset Preprocessing — Assumptions and Modifications

`src/preprocess.py` is the single cleaning path for the raw dataset
(`data/raw/crossfit.csv` → `data/processed/crossfit.csv`). There is no
parallel v1/v2 branching in code — dataset versioning (raw vs. cleaned) is
handled externally via DVC/git tags, not by producing multiple files here.

**Baseline cleaning (as specified in the assignment):**

- Drops any row missing a value in a core set of columns: `region`, `age`,
  `weight`, `height`, `howlong`, `gender`, `eat`, `background`, `experience`,
  `schedule`, `deadlift`, `candj`, `snatch`, `backsq`.
- Drops irrelevant columns entirely: `affiliate`, `team`, `name`, and the
  benchmark-workout columns `fran`, `helen`, `grace`, `filthy50`,
  `fgonebad`, `run400`, `run5k`, `pullups`, `train`.
- Removes physically implausible values via fixed thresholds:
  `weight < 1500`; `gender != '--'`; `age >= 18`; `48 < height < 96`;
  deadlift capped by gender (`≤1105` male, `≤636` female); `0 < candj ≤395`;
  `0 < snatch ≤496`; `0 < backsq ≤1069`.
- Treats `"Decline to answer"` survey responses (in `background`,
  `experience`, `schedule`, `howlong`, `eat`) as missing and drops those
  rows.

**Extension (beyond the assignment baseline, explicitly documented in
code):** an additional **z-score-based statistical outlier filter** is
applied on top of the baseline, since fixed thresholds catch
physically-impossible values but not statistical outliers that fall within
the plausible range. This computes a per-column z-score (mean/std) across
`age`, `weight`, `height`, `deadlift`, `candj`, `snatch`, `backsq` and drops
rows exceeding a configurable threshold (`params.yaml → preprocess.outlier_zscore`,
default `3.0`). Row counts before/after this step are printed for
transparency.

**Assumption worth flagging:** the z-score filter is computed on the
already-baseline-cleaned data, not the raw data — so its mean/std are not
skewed by the physically-impossible values the baseline step already
removed. This ordering (fixed-threshold filter → z-score filter) is
intentional.

---

## Assignment #3: AutoML Workflow (PyCaret + H2O)

This section documents the AutoML addendum to the pipeline above. Unlike
Assignment #2, AutoML is explicitly **required** here — no algorithms are
hand-picked; PyCaret and H2O AutoML each search their own model libraries
automatically.

### Setup (additional dependencies)

The base `requirements.txt` install (see Setup above) now also includes:

```
pycaret==3.3.2
h2o>=3.44.0
```

H2O additionally requires a local Java runtime (JDK 11+) on your machine —
check with `java -version` before running the H2O stage; install a JDK
first if missing. Both PyCaret and H2O AutoML run **fully locally**, with
no cloud account or credentials required.

Install (or reinstall from the updated `requirements.txt`):

```bash
pip install -r requirements.txt
```

> **Known issue — PyCaret/MLflow compatibility:** PyCaret's built-in
> `setup(log_experiment=True)` MLflow autologger crashes on newer MLflow
> versions due to an open PyCaret bug
> ([pycaret/pycaret#4100](https://github.com/pycaret/pycaret/issues/4100)):
> PyCaret's internal `_set_up_logging()` unconditionally calls a helper
> that assumes MLflow's private `_active_run_stack` is a plain list, which
> newer MLflow releases changed to a thread-safe object with no `.copy()`
> method. This crashes `setup()` regardless of what `log_experiment` is
> set to. `src/automl_pycaret.py` works around this with a targeted
> monkeypatch (`_patch_pycaret_mlflow_bug()`) applied before `setup()` is
> called; PyCaret's own MLflow logging is disabled
> (`log_experiment=False`) and replaced with manual logging via MLflow's
> public API instead. No action needed to run the pipeline — this is
> handled automatically — but it's documented here since it affects how
> MLflow experiment data for PyCaret runs is produced.

### Dataset used

Same processed, feature-engineered dataset as Assignment #2
(`data/processed/crossfit_features.csv`) — **not** raw data. See
"Dataset Preprocessing" above for how it was produced. Both AutoML scripts
drop the same leakage columns as `train.py` (`deadlift`, `candj`, `snatch`,
`backsq`, `athlete_id`, `event_timestamp`) before AutoML ever sees the
data. After these drops, 5 predictor features remain: `gender`, `age`,
`height`, `weight`, `howlong` (this matches the Feast `v1` feature set,
not the larger `v2` set — the engineered survey features were not present
in this particular `crossfit_features.csv` snapshot).

### Pipeline stages

Two new DVC stages run each AutoML platform, plus a comparison stage:

```
feature_engineering → automl_pycaret ┐
                    → automl_h2o     ├→ automl_compare
```

| Stage | Script | What it does |
|-------|--------|---------------|
| `automl_pycaret` | `src/automl_pycaret.py` | Two PyCaret passes: all features, then top-3 features (by importance). Saves leaderboards, feature-importance/residual plots, and a summary. |
| `automl_h2o` | `src/automl_h2o.py` | Same two-pass structure using H2O AutoML. Falls back to the best non-ensemble model for variable importance when the leaderboard-topping model is a Stacked Ensemble (which has no native `varimp()`). |
| `automl_compare` | `src/automl_compare.py` | Reads both platforms' leaderboards; produces top-3-by-score and top-3-by-speed tables for both feature sets, plus the Assignment #1 baseline comparison. |

Run the full AutoML workflow:

```bash
dvc repro automl_compare
```

This is also wired into `run_all.py`, so `python run_all.py` runs it as
part of the full end-to-end pipeline (after the Assignment #2 stages).

**Runtime note:** `automl_h2o` runs two full H2O AutoML searches (all
features, then top features), each budgeted `automl_h2o.max_runtime_secs`
(default 300s) via `params.yaml` — expect ~10 minutes total for this stage
alone. `automl_pycaret` is faster (PyCaret has no time cap; it simply
trains its full default algorithm library, both passes typically complete
in under a minute total for this dataset size).

### Configuration (`params.yaml`)

```yaml
automl_pycaret:
  session_id: 42        # same seed used throughout the project
  top_n_report: 5         # task 4: top-5 features to report
  top_n_rerun: 3            # task 5: fixed feature count for the rerun
  n_select: 3                 # how many top models compare_models() returns
  fold: 5                        # cross-validation folds

automl_h2o:
  max_runtime_secs: 300   # per pass (all-features AND top-features each get this budget)
  seed: 42
  nfolds: 5
  top_n_report: 5
  top_n_rerun: 3

automl_compare:
  assignment1_baseline:      # Assignment #1 Part A numbers, for Section 7 comparison
    model_name: ...
    validation_metric_name: ...
    validation_score: ...
    training_time_seconds: ...
```

### Platform classification

PyCaret is classified as **low-code**: no graphical UI (code is required
to call `setup()`/`compare_models()`), but a single function call replaces
what would otherwise be a hand-written loop over ~20 algorithms with
manual cross-validation and metric computation. Since neither PyCaret nor
H2O AutoML used here is a graphical no-code tool, there is no setup UI to
screenshot — the evidence trail is instead the console leaderboard output,
exported CSV leaderboards, and generated feature-importance/residual
plots, all referenced in the full report.

### Outputs

```
reports/
  automl_pycaret_leaderboard_all.csv
  automl_pycaret_leaderboard_top_features.csv
  automl_pycaret_summary.txt
  automl_h2o_leaderboard_all.csv
  automl_h2o_leaderboard_top_features.csv
  automl_h2o_summary.txt
  automl_comparison.txt          # cross-platform + Assignment #1 baseline comparison

outputs/automl/
  pycaret_feature_importance_all.png
  pycaret_residuals_all.png
  pycaret_feature_importance_top.png
  pycaret_residuals_top.png
  h2o_varimp_all.png
  h2o_varimp_top.png
```

MLflow experiments: `crossfit-automl-pycaret` and `crossfit-automl-h2o`
(separate from Assignment #2's `crossfit-total-lift` experiment).

**Full written report:** `AutoML_Workflow_Report.docx` — covers all 9
required tasks (dataset setup, platform configuration, all-features run,
data insights/feature importance, validation-score comparison, speed
comparison, Assignment #1 baseline comparison, platform mode assessment,
and the H2O repeat) with figures pulled directly from the CSVs/summaries
above, plus a dedicated assumptions/limitations section.

---

## Reproducibility

- Fixed random seed via `random_seed.py` / `params.yaml`
- Declarative pipeline via `dvc.yaml`, locked via `dvc.lock`
- Pinned dependencies in `requirements.txt`
- Deterministic train/test split (`split_data.py`)
- All experiment configuration (feature version, hyperparameters) passed
  explicitly via CLI args / `params.yaml`, never hardcoded per-run

---

## Static Analysis

Code quality checks are stored in `reports/pylint_report.txt`.