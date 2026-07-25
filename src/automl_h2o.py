"""
automl_h2o.py
--------------
Assignment #3, Task #9: H2O AutoML repeat.

Runs the SAME AutoML workflow as automl_pycaret.py (all-features pass),
on the SAME dataset (data/processed/crossfit_features.csv) and target
(total_lift), using H2O AutoML instead of PyCaret -- so the two are
directly comparable in the report (task 9's explicit requirement).

H2O runs fully local (starts its own local JVM cluster via h2o.init());
no cloud account or credentials required.

Outputs:
    reports/automl_h2o_leaderboard.csv
    reports/automl_h2o_summary.txt
    outputs/automl/h2o_varimp.png

MLflow: H2O does not have a first-class MLflow autolog integration
like PyCaret's log_experiment=True, so this script manually logs the
best model's params/metrics/training time to MLflow under the
experiment name below, for parity with the PyCaret and Assignment #2
tracking.
"""

import sys
import time
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

MLFLOW_EXPERIMENT_NAME = "crossfit-automl-h2o"


def load_params() -> dict:
    """Load H2O AutoML parameters from params.yaml, with safe defaults."""
    defaults = {
        "max_runtime_secs": 300,
        "seed": 42,
        "nfolds": 5,
        "top_n_features": 5,
    }
    if PARAMS_PATH.exists():
        with open(PARAMS_PATH, "r", encoding="utf-8") as f:
            all_params = yaml.safe_load(f) or {}
        return {**defaults, **all_params.get("automl_h2o", {})}
    return defaults


def load_dataset() -> pd.DataFrame:
    """Load the processed/feature-engineered dataset, dropping leakage columns."""
    if not INPUT_PATH.exists():
        print(f"[automl_h2o] ERROR: {INPUT_PATH} not found. "
              f"Run src/feature_engineering.py first.", file=sys.stderr)
        sys.exit(1)

    data = pd.read_csv(INPUT_PATH, low_memory=False)
    drop_cols = [c for c in (LEAKAGE_COLS + NON_FEATURE_COLS) if c in data.columns]
    data = data.drop(columns=drop_cols)
    print(f"[automl_h2o] Loaded {INPUT_PATH.name}: {data.shape} "
          f"(dropped leakage/id columns: {drop_cols})")
    return data


def find_varimp_model(aml, leaderboard):
    """
    H2O's overall best model (aml.leader) is very often a StackedEnsemble,
    which has NO native variable importance (an ensemble blends other
    models' predictions rather than weighting raw features directly --
    this is expected H2O AutoML behavior, not a bug). Walk the leaderboard
    in ranked order and return the first model that DOES expose varimp(),
    along with its rank position, so we can still report top features from
    the best model that supports it.
    """
    import h2o

    for i, model_id in enumerate(leaderboard["model_id"]):
        model = h2o.get_model(model_id)
        try:
            varimp = model.varimp(use_pandas=True)
            if varimp is not None and not varimp.empty:
                return model, i
        except Exception:
            continue
    return None, None


def main():
    import h2o
    from h2o.automl import H2OAutoML
    import matplotlib
    matplotlib.use("Agg")  # headless backend, required for server=True savefig
    import matplotlib.pyplot as plt

    params = load_params()
    print(f"[automl_h2o] Params: {params}")

    data = load_dataset()

    print("[automl_h2o] Starting local H2O cluster...")
    h2o.init()

    h2o_frame = h2o.H2OFrame(data)
    feature_cols = [c for c in h2o_frame.columns if c != TARGET_COL]

    aml = H2OAutoML(
        max_runtime_secs=params["max_runtime_secs"],
        seed=params["seed"],
        nfolds=params["nfolds"],
        sort_metric="RMSE",
    )

    print(f"[automl_h2o] Running AutoML (max_runtime_secs={params['max_runtime_secs']})...")
    start = time.time()
    aml.train(x=feature_cols, y=TARGET_COL, training_frame=h2o_frame)
    elapsed = time.time() - start
    print(f"[automl_h2o] AutoML run complete in {elapsed:.1f}s")

    leaderboard = aml.leaderboard.as_data_frame()
    print(f"[automl_h2o] Leaderboard:\n{leaderboard.to_string()}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard_path = REPORTS_DIR / "automl_h2o_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    print(f"[automl_h2o] Wrote leaderboard to {leaderboard_path}")

    best_model = aml.leader

    # --- Variable importance (top-5) ---
    # aml.leader is frequently a StackedEnsemble with no native varimp.
    # Fall back to the best-ranked individual model that DOES support it.
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    varimp_plot_path = OUTPUTS_DIR / "h2o_varimp.png"
    top_features = []
    varimp_source_model = None
    varimp_source_rank = None

    if hasattr(best_model, "varimp"):
        try:
            varimp_df = best_model.varimp(use_pandas=True)
        except Exception:
            varimp_df = None
    else:
        varimp_df = None

    if varimp_df is not None and not varimp_df.empty:
        varimp_source_model = best_model
        varimp_source_rank = 0
    else:
        print(f"[automl_h2o] Best model ({best_model.algo}) does not expose variable "
              f"importance (expected for Stacked Ensembles). Falling back to the "
              f"best-ranked individual model that does.")
        varimp_source_model, varimp_source_rank = find_varimp_model(aml, leaderboard)

    if varimp_source_model is not None:
        varimp_df = varimp_source_model.varimp(use_pandas=True)
        top_features = varimp_df.head(params["top_n_features"])["variable"].tolist()
        print(f"[automl_h2o] Top-{params['top_n_features']} features (from "
              f"{varimp_source_model.model_id}, leaderboard rank {varimp_source_rank}): "
              f"{top_features}")
        varimp_source_model.varimp_plot(num_of_features=params["top_n_features"], server=True)
        plt.savefig(varimp_plot_path)
        plt.close()
        print(f"[automl_h2o] Saved variable importance plot to {varimp_plot_path}")
    else:
        # No model in the entire leaderboard exposes varimp -- write a
        # placeholder image so the declared DVC output always exists.
        print("[automl_h2o] WARNING: no model in the leaderboard exposes variable "
              "importance; writing placeholder image.")
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No model in this H2O AutoML run exposes\nvariable importance.",
                ha="center", va="center", fontsize=11)
        ax.axis("off")
        plt.savefig(varimp_plot_path)
        plt.close()

    # --- Manual MLflow logging (H2O has no autolog like PyCaret) ---
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"h2o_best_{best_model.algo}"):
        mlflow.log_params({
            "algo": best_model.algo,
            "model_id": best_model.model_id,
            "max_runtime_secs": params["max_runtime_secs"],
            "seed": params["seed"],
            "nfolds": params["nfolds"],
            "feature_count": len(feature_cols),
            "train_rows": data.shape[0],
            "varimp_source_model_id": varimp_source_model.model_id if varimp_source_model else "none",
            "varimp_source_rank": varimp_source_rank if varimp_source_rank is not None else -1,
        })
        perf = best_model.model_performance(train=True)
        try:
            mlflow.log_metrics({
                "rmse": perf.rmse(),
                "mae": perf.mae(),
                "r2": perf.r2(),
            })
        except Exception as exc:
            print(f"[automl_h2o] WARNING: could not log all metrics: {exc}")
        mlflow.log_metric("automl_wall_clock_seconds", elapsed)
        mlflow.log_artifact(str(leaderboard_path))

    write_summary(leaderboard, best_model, top_features, elapsed, params,
                  varimp_source_model, varimp_source_rank)

    h2o.cluster().shutdown(prompt=False)
    print("\n[automl_h2o] COMPLETE.")


def write_summary(leaderboard, best_model, top_features, elapsed, params,
                   varimp_source_model=None, varimp_source_rank=None):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "automl_h2o_summary.txt"

    lines = [
        "H2O AutoML Summary",
        "=" * 50,
        f"Target: {TARGET_COL}",
        f"Leakage columns excluded: {LEAKAGE_COLS}",
        f"Seed: {params['seed']}   nfolds: {params['nfolds']}   "
        f"max_runtime_secs: {params['max_runtime_secs']}",
        f"Wall-clock AutoML run time: {elapsed:.1f}s",
        "",
        f"Best model: {best_model.model_id} ({best_model.algo})",
        "",
        "Top 3 rows of leaderboard (sorted by RMSE):",
        leaderboard.head(3).to_string(),
        "",
    ]

    if varimp_source_model is not None and varimp_source_rank == 0:
        lines.append(f"Top-{params['top_n_features']} features by variable importance "
                      f"(from the best model, {best_model.model_id}):")
    elif varimp_source_model is not None:
        lines.append(
            f"NOTE: the best model ({best_model.model_id}, {best_model.algo}) is a "
            f"Stacked Ensemble and does not expose native variable importance -- this "
            f"is expected H2O behavior, not an error. Feature importance below is "
            f"instead taken from the best-ranked individual model that does support it: "
            f"{varimp_source_model.model_id} (leaderboard rank {varimp_source_rank})."
        )
        lines.append("")
        lines.append(f"Top-{params['top_n_features']} features by variable importance:")
    else:
        lines.append(f"Top-{params['top_n_features']} features by variable importance:")

    if top_features:
        lines.extend(f"  - {feat}" for feat in top_features)
    else:
        lines.append("  (no model in this leaderboard exposes variable importance)")
    lines.append("")
    lines.append(f"Full leaderboard saved to: reports/automl_h2o_leaderboard.csv")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[automl_h2o] Wrote summary to {path}")


if __name__ == "__main__":
    main()