# CrossFit Strength Prediction: A Reproducible MLOps Pipeline

An end-to-end ML pipeline that predicts a CrossFit athlete's total strength (deadlift + clean & jerk + snatch + back squat) from demographics and training background. The model is simple on purpose. The point is the infrastructure: every dataset, feature set, and model is versioned, and the whole pipeline rebuilds from a clean clone with one command.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-13ADC7?style=flat-square&logo=dvc&logoColor=white)
![Feast](https://img.shields.io/badge/Feast-feature%20store-555?style=flat-square)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=flat-square)
![H2O](https://img.shields.io/badge/H2O%20%2B%20PyCaret-AutoML-FFD43B?style=flat-square)

## At a glance

| Versioning axis | Tool | What it gives you |
|---|---|---|
| Data & pipeline | **DVC** | `dvc.yaml` stage graph, `dvc.lock` hashes, `dvc repro <stage>` |
| Features | **Feast** | Declarative feature views (`v1`, `v2`) retrieved at training time |
| Experiments & models | **MLflow** | Params, metrics, signatures, and a model registry tagged with feature version |

```
ingest → preprocess → feature_engineering → split_data (Feast retrieval) → train → evaluate → compare
                                          └→ automl_pycaret ┐
                                          └→ automl_h2o     ┴→ automl_compare
```

## Results (held-out test set, n = 5,822)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| XGBoost (hand-configured) | 151.5 | 117.9 | 0.701 |
| PyCaret AutoML best: LightGBM (5-fold CV) | 151.3 | 118.0 | 0.694 |
| H2O AutoML best: Stacked Ensemble (5-fold CV) | 150.2 | 117.3 | n/a |

**Takeaways**

- Two independent AutoML searches landed within about 1 RMSE point of the hand-tuned XGBoost, which suggests the signal in these five demographic features is close to saturated.
- Cutting to the top 3 features cost about 0.15 R² (PyCaret) and 21 RMSE (H2O), so every feature earns its place.

## Engineering decisions worth noting

- **Leakage guard.** `total_lift` and its four components are dropped inside `prepare_xy()` no matter which feature version is active, even though `total_lift` is stored in the v2 feature view for lineage.
- **Two-stage outlier handling.** Physically impossible values are removed with fixed thresholds first, and a configurable z-score filter runs afterward, so the extreme values don't distort its mean and standard deviation.
- **2×2 experiment grid** (feature version × hyperparameters), with each run registered in MLflow and the model version tagged with `feature_version` and `feature_count`.
- **Upstream bug workaround.** PyCaret's MLflow autologger crashes on newer MLflow releases ([pycaret#4100](https://github.com/pycaret/pycaret/issues/4100)). A targeted monkeypatch fixes it, and runs are logged manually through MLflow's public API instead.

## Known issue

The v1 (5-feature) and v2 (11-feature) models produce identical metrics. The six engineered survey features in v2 appear to be constant in the current data snapshot, so they add no information. Fixing the upstream encoding is the next step before the v1-vs-v2 comparison means anything.

## Quick start

```bash
git clone https://github.com/ryantechs-dotcom/athletes_dvc_project.git
cd athletes_dvc_project
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt       # H2O also needs JDK 11+

python run_all.py                     # full pipeline incl. AutoML
# or
dvc repro compare                     # core pipeline via DVC

mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Repo layout

```
src/            ingest, preprocess, eda, feature_engineering, feature_definitions (Feast),
                split_data, train, evaluate, run_experiments, compare_versions,
                automl_pycaret, automl_h2o, automl_compare
data/           raw → processed → feature_store (Feast offline/online/registry) → train/test
reports/        evaluation, v1-vs-v2 comparison, AutoML leaderboards & summary
outputs/        EDA and AutoML feature-importance / residual plots
dvc.yaml · dvc.lock · params.yaml · feature_store.yaml · run_all.py
```

<details>
<summary>Preprocessing rules in full</summary>

- Drop rows missing any of: region, age, weight, height, howlong, gender, eat, background, experience, schedule, deadlift, candj, snatch, backsq
- Drop irrelevant columns: affiliate, team, name, benchmark workouts (fran, helen, grace, filthy50, fgonebad, run400, run5k, pullups, train)
- Keep rows with weight < 1500, age ≥ 18, 48 < height < 96, deadlift ≤ 1105 (M) / 636 (F), 0 < candj ≤ 395, 0 < snatch ≤ 496, 0 < backsq ≤ 1069
- Treat "Decline to answer" as missing
- Z-score filter (default 3.0, set in `params.yaml`) on age, weight, height, and the four lifts
</details>
