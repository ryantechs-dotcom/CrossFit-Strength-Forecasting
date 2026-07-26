"""
automl_pycaret.py
------------------
Assignment #3, Stage: PyCaret AutoML run.

Uses the SAME processed/feature-engineered dataset produced by
Assignment #2's pipeline (data/processed/crossfit_features.csv), NOT
raw or Assignment-#1-baseline data. This is the "processed dataset
version from Assignment #1/#2" referenced in the task 1 requirement --
documented explicitly here and in the report.

Target: total_lift (deadlift + candj + snatch + backsq), same
leakage-prevention exclusions as train.py: the four lift components
and athlete_id are dropped before AutoML sees the data. AutoML is,
by design, allowed here (unlike Assignment #2, which forbade it) --
this is the point of the assignment.

Runs TWO AutoML passes:
    1. All appropriate features (full engineered feature set)
    2. Only the top-N features by importance from pass 1
       (N = automl_pycaret.top_n_rerun, default 3 -- per the
       assignment's "use the top three features" instruction)

NOTE on feature counts: the assignment asks for two DIFFERENT numbers:
  - task 4: report the TOP FIVE features (top_n_report, default 5)
  - task 5: rerun AutoML using a FIXED number of top features, "top
    three" if you must pick a fixed count (top_n_rerun, default 3)
These are deliberately separate params -- do not conflate them.

Outputs:
    reports/automl_pycaret_leaderboard_all.csv
    reports/automl_pycaret_leaderboard_top_features.csv
    reports/automl_pycaret_summary.txt
    outputs/automl/pycaret_feature_importance_all.png
    outputs/automl/pycaret_residuals_all.png
    outputs/automl/pycaret_feature_importance_top.png
    outputs/automl/pycaret_residuals_top.png

MLflow: PyCaret's built-in log_experiment=True autolog relies on
private MLflow internals (_active_run_stack) that break across MLflow
versions -- it is NOT used here. Instead, after each compare_models()
call, the leaderboard's top N models are logged manually via the
public MLflow API (mlflow.start_run/log_params/log_metrics), under
the "crossfit-automl-pycaret" experiment. This mirrors the manual
logging approach used in automl_h2o.py and is robust to MLflow version
changes. This is separate from -- and does not overwrite -- the
"crossfit-total-lift" experiment used in Assignment #2.
"""

import shutil
import sys
from pathlib import Path

import mlflow
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "crossfit_features.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "automl"
PARAMS_PATH = PROJECT_ROOT / "params.yaml"

TARGET_COL = "total_lift"
LEAKAGE_COLS = ["deadlift", "candj", "snatch", "backsq"]
NON_FEATURE_COLS = ["athlete_id", "event_timestamp"]

MLFLOW_EXPERIMENT_NAME = "crossfit-automl-pycaret"


def load_params() -> dict:
    """Load AutoML parameters from params.yaml, with safe defaults."""
    defaults = {
        "session_id": 42,
        "top_n_report": 5,   # task 4: report top-5 features
        "top_n_rerun": 3,     # task 5: rerun AutoML on top-3 features
        "n_select": 3,
        "fold": 5,
    }
    if PARAMS_PATH.exists():
        with open(PARAMS_PATH, "r", encoding="utf-8") as f:
            all_params = yaml.safe_load(f) or {}
        return {**defaults, **all_params.get("automl_pycaret", {})}
    return defaults


def load_dataset() -> pd.DataFrame:
    """Load the processed/feature-engineered dataset, dropping leakage columns."""
    if not INPUT_PATH.exists():
        print(f"[automl_pycaret] ERROR: {INPUT_PATH} not found. "
              f"Run src/feature_engineering.py first.", file=sys.stderr)
        sys.exit(1)

    data = pd.read_csv(INPUT_PATH, low_memory=False)
    drop_cols = [c for c in (LEAKAGE_COLS + NON_FEATURE_COLS) if c in data.columns]
    data = data.drop(columns=drop_cols)
    print(f"[automl_pycaret] Loaded {INPUT_PATH.name}: {data.shape} "
          f"(dropped leakage/id columns: {drop_cols})")
    return data


def _patch_pycaret_mlflow_bug():
    """
    Work around a known, currently-open PyCaret bug (pycaret/pycaret#4100):
    PyCaret's internal _set_up_logging() unconditionally calls a helper,
    clean_active_mlflow_run(), which assumes MLflow's private
    _active_run_stack is a plain list. Newer MLflow releases replaced it
    with a thread-safe object with no .copy() method, so this crashes
    setup() with AttributeError -- regardless of what log_experiment is
    set to, since the buggy call happens before that flag is checked.

    This patches the one broken helper with a harmless no-op context
    manager so setup() can proceed. It does not affect this script's own
    MLflow logging (log_leaderboard_to_mlflow), which uses only the
    public MLflow API and never touches PyCaret's internal logger.
    """
    try:
        import contextlib
        import pycaret.loggers.mlflow_logger as pcaret_mlflow_module

        @contextlib.contextmanager
        def _noop_clean_active_mlflow_run():
            yield

        pcaret_mlflow_module.clean_active_mlflow_run = _noop_clean_active_mlflow_run
        print("[automl_pycaret] Applied PyCaret/MLflow compatibility patch (pycaret#4100).")
    except Exception as exc:
        print(f"[automl_pycaret] WARNING: could not apply PyCaret/MLflow patch: {exc}")


def run_compare(data: pd.DataFrame, params: dict, run_label: str):
    """
    Run PyCaret setup() + compare_models() on the given dataframe.
    Returns (leaderboard_df, best_models_list).
    """
    # Imported here (not top-of-file) so the rest of the pipeline doesn't
    # require pycaret installed unless this stage actually runs.
    from pycaret.regression import setup, compare_models, pull, plot_model

    _patch_pycaret_mlflow_bug()

    print(f"\n{'=' * 70}\n[automl_pycaret] Run: {run_label}\n{'=' * 70}")

    setup(
        data=data,
        target=TARGET_COL,
        session_id=params["session_id"],
        fold=params["fold"],
        log_experiment=False,
        log_plots=False,
        verbose=False,
    )

    best_models = compare_models(n_select=params["n_select"], verbose=False)
    if not isinstance(best_models, list):
        best_models = [best_models]

    leaderboard = pull()
    print(f"[automl_pycaret] Leaderboard ({run_label}):\n{leaderboard.to_string()}")

    return leaderboard, best_models


def log_leaderboard_to_mlflow(leaderboard: pd.DataFrame, run_label: str, params: dict, feature_count: int):
    """
    Manually log the top rows of a PyCaret leaderboard to MLflow, one
    MLflow run per model, using the public MLflow API. This replaces
    PyCaret's built-in log_experiment=True autolog, which depends on
    private MLflow internals that break across MLflow versions.
    """
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    numeric_cols = [c for c in leaderboard.columns if c not in ("Model",)]

    for model_name, row in leaderboard.iterrows():
        run_name = f"pycaret_{run_label}_{str(model_name).replace(' ', '_')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "model_name": model_name,
                "run_label": run_label,
                "session_id": params["session_id"],
                "fold": params["fold"],
                "feature_count": feature_count,
            })
            metrics = {}
            for col in numeric_cols:
                try:
                    metrics[col.replace(" ", "_").replace("(", "").replace(")", "")] = float(row[col])
                except (ValueError, TypeError):
                    continue
            if metrics:
                mlflow.log_metrics(metrics)
    print(f"[automl_pycaret] Logged {len(leaderboard)} models from '{run_label}' leaderboard to MLflow "
          f"experiment '{MLFLOW_EXPERIMENT_NAME}'")


def save_leaderboard(leaderboard: pd.DataFrame, filename: str):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / filename
    leaderboard.to_csv(path, index=True)
    print(f"[automl_pycaret] Wrote leaderboard to {path}")


def get_feature_importance_series(best_model) -> pd.Series:
    """
    Return the FULL feature importance series (all features, sorted
    descending) from the best model's feature_importances_ (tree/boosting
    models) or abs(coef_) (linear models). Callers slice .head(n) with
    whatever N they need (5 for reporting, 3 for the rerun) -- this
    function itself makes no assumption about how many to return.
    """
    from pycaret.regression import get_config

    x_train = get_config("X_train")

    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = abs(best_model.coef_)
    else:
        print("[automl_pycaret] WARNING: best model has no feature_importances_ "
              "or coef_ attribute; cannot extract feature importance.")
        return pd.Series(dtype=float)

    return pd.Series(importances, index=x_train.columns).sort_values(ascending=False)


def save_feature_plots(best_model, run_label: str):
    """Save feature-importance and residuals plots for data-insights reporting."""
    from pycaret.regression import plot_model

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        plot_path = plot_model(best_model, plot="feature", save=True)
        dest = OUTPUTS_DIR / f"pycaret_feature_importance_{run_label}.png"
        shutil.move(plot_path, dest)
        print(f"[automl_pycaret] Saved feature importance plot to {dest}")
    except Exception as exc:  # some models don't support this plot
        print(f"[automl_pycaret] WARNING: could not generate feature plot: {exc}")

    try:
        plot_path = plot_model(best_model, plot="residuals", save=True)
        dest = OUTPUTS_DIR / f"pycaret_residuals_{run_label}.png"
        shutil.move(plot_path, dest)
        print(f"[automl_pycaret] Saved residuals plot to {dest}")
    except Exception as exc:
        print(f"[automl_pycaret] WARNING: could not generate residuals plot: {exc}")


def write_summary(all_leaderboard, top_leaderboard, top5_report_features,
                   top3_rerun_features, importance_series, params):
    """Write a plain-text summary tying together both AutoML passes."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "automl_pycaret_summary.txt"

    lines = [
        "PyCaret AutoML Summary",
        "=" * 50,
        f"Target: {TARGET_COL}",
        f"Leakage columns excluded: {LEAKAGE_COLS}",
        f"Session ID (seed): {params['session_id']}",
        f"CV folds: {params['fold']}",
        "",
        "--- Pass 1: All Features ---",
        f"Best model (row 0 of leaderboard):\n{all_leaderboard.iloc[0].to_string()}",
        "",
        f"Top-{params['top_n_report']} features by importance (task 4 reporting):",
    ]
    lines.extend(f"  - {feat}: {importance_series[feat]:.4f}" for feat in top5_report_features)
    lines.append("")
    lines.append(
        f"Top-{params['top_n_rerun']} features used for Pass 2 rerun (task 5, "
        f"fixed-count subset of the list above):"
    )
    lines.extend(f"  - {feat}: {importance_series[feat]:.4f}" for feat in top3_rerun_features)
    lines.append("")
    lines.append(f"--- Pass 2: Top-{params['top_n_rerun']} Features Only ---")
    lines.append(f"Best model (row 0 of leaderboard):\n{top_leaderboard.iloc[0].to_string()}")
    lines.append("")
    lines.append("Full leaderboards saved to:")
    lines.append("  reports/automl_pycaret_leaderboard_all.csv")
    lines.append("  reports/automl_pycaret_leaderboard_top_features.csv")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[automl_pycaret] Wrote summary to {path}")


def main():
    params = load_params()
    print(f"[automl_pycaret] Params: {params}")

    data = load_dataset()
    all_feature_count = data.shape[1] - 1  # exclude target

    # --- Pass 1: all features ---
    all_leaderboard, all_best_models = run_compare(data, params, run_label="all_features")
    save_leaderboard(all_leaderboard, "automl_pycaret_leaderboard_all.csv")
    log_leaderboard_to_mlflow(all_leaderboard, "all_features", params, all_feature_count)
    save_feature_plots(all_best_models[0], run_label="all")

    importance_series = get_feature_importance_series(all_best_models[0])
    if importance_series.empty:
        print("[automl_pycaret] ERROR: could not extract feature importances; "
              "cannot run Pass 2. Exiting.", file=sys.stderr)
        sys.exit(1)

    top5_report_features = importance_series.head(params["top_n_report"]).index.tolist()
    top3_rerun_features = importance_series.head(params["top_n_rerun"]).index.tolist()
    print(f"[automl_pycaret] Top-{params['top_n_report']} features (report): {top5_report_features}")
    print(f"[automl_pycaret] Top-{params['top_n_rerun']} features (rerun):   {top3_rerun_features}")

    # --- Pass 2: top-N features only (N = top_n_rerun) ---
    top_data = data[top3_rerun_features + [TARGET_COL]].copy()
    top_leaderboard, top_best_models = run_compare(top_data, params, run_label="top_features")
    save_leaderboard(top_leaderboard, "automl_pycaret_leaderboard_top_features.csv")
    log_leaderboard_to_mlflow(top_leaderboard, "top_features", params, len(top3_rerun_features))
    save_feature_plots(top_best_models[0], run_label="top")

    write_summary(all_leaderboard, top_leaderboard, top5_report_features,
                  top3_rerun_features, importance_series, params)

    print("\n[automl_pycaret] COMPLETE.")


if __name__ == "__main__":
    main()