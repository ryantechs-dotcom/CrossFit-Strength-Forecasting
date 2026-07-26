"""
automl_h2o.py
--------------
Assignment #3, Task #9: H2O AutoML repeat.

Runs the SAME two-pass AutoML workflow as automl_pycaret.py, on the
SAME dataset (data/processed/crossfit_features.csv) and target
(total_lift), using H2O AutoML instead of PyCaret -- so the two are
directly comparable in the report.

Pass 1: all appropriate features.
Pass 2: only the top-N features by variable importance from Pass 1
        (N = automl_h2o.top_n_rerun, default 3, mirroring PyCaret's rerun
        for a fair, apples-to-apples comparison).

NOTE on feature counts (same convention as automl_pycaret.py):
  - top_n_report (default 5): task 9's "report the top five features"
  - top_n_rerun (default 3): fixed count used for the Pass 2 rerun

H2O runs fully local (starts its own local JVM cluster via h2o.init());
no cloud account or credentials required.

Speed: H2O's leaderboard doesn't include per-model timing by default.
This script requests it explicitly via
h2o.automl.get_leaderboard(aml, extra_columns=[...]), which adds
training_time_ms and predict_time_per_row_ms columns -- H2O's own
native, per-model speed metrics (analogous to PyCaret's "TT (Sec)").

Variable importance: the leaderboard-topping model is very often a
StackedEnsemble, which has NO native variable importance (an ensemble
blends other models' predictions rather than directly weighting raw
features -- expected H2O behavior, not an error). This script falls
back to the best-ranked individual model that DOES expose varimp(),
and documents which model that was.

Outputs (per pass, "all"/"top"):
    reports/automl_h2o_leaderboard_all.csv
    reports/automl_h2o_leaderboard_top_features.csv
    reports/automl_h2o_summary.txt
    outputs/automl/h2o_varimp_all.png
    outputs/automl/h2o_varimp_top.png

MLflow: H2O does not have a first-class MLflow autolog integration
like PyCaret's log_experiment=True, so this script manually logs each
pass's best model params/metrics/training time to MLflow under the
experiment name below, for parity with PyCaret and Assignment #2.
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
        "top_n_report": 5,
        "top_n_rerun": 3,
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


def get_extended_leaderboard(aml) -> pd.DataFrame:
    """
    Return the AutoML leaderboard with per-model speed columns
    (training_time_ms, predict_time_per_row_ms) included, where the
    installed H2O version supports it. Falls back to the standard
    leaderboard (no speed columns) if the extended API isn't available.
    """
    try:
        from h2o.automl import get_leaderboard
        extended = get_leaderboard(aml, extra_columns=["training_time_ms", "predict_time_per_row_ms"])
        return extended.as_data_frame()
    except Exception as exc:
        print(f"[automl_h2o] WARNING: could not fetch extended leaderboard with "
              f"speed columns ({exc}); falling back to standard leaderboard.")
        return aml.leaderboard.as_data_frame()


def find_varimp_model(leaderboard: pd.DataFrame):
    """
    Walk the leaderboard in ranked order and return the first model that
    exposes native variable importance, along with its rank position.
    Necessary because the overall best model (rank 0) is very often a
    StackedEnsemble, which has none.
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


def run_automl_pass(h2o_frame, feature_cols, params: dict, run_label: str):
    """
    Run one H2O AutoML pass (setup + train) on the given features.
    Returns (leaderboard_df, aml, best_model, elapsed_seconds).
    """
    from h2o.automl import H2OAutoML

    print(f"\n{'=' * 70}\n[automl_h2o] Run: {run_label}\n{'=' * 70}")

    aml = H2OAutoML(
        max_runtime_secs=params["max_runtime_secs"],
        seed=params["seed"],
        nfolds=params["nfolds"],
        sort_metric="RMSE",
    )

    start = time.time()
    aml.train(x=feature_cols, y=TARGET_COL, training_frame=h2o_frame)
    elapsed = time.time() - start
    print(f"[automl_h2o] AutoML run ({run_label}) complete in {elapsed:.1f}s")

    leaderboard = get_extended_leaderboard(aml)
    print(f"[automl_h2o] Leaderboard ({run_label}):\n{leaderboard.to_string()}")

    return leaderboard, aml, aml.leader, elapsed


def resolve_varimp(leaderboard: pd.DataFrame, best_model, top_n: int):
    """
    Get the top-N feature names by importance, falling back from the
    (possibly ensemble) best model to the best individual model that
    supports varimp(). Returns (top_features, source_model, source_rank).
    """
    varimp_df = None
    if hasattr(best_model, "varimp"):
        try:
            varimp_df = best_model.varimp(use_pandas=True)
        except Exception:
            varimp_df = None

    if varimp_df is not None and not varimp_df.empty:
        source_model, source_rank = best_model, 0
    else:
        print(f"[automl_h2o] Best model ({best_model.algo}) does not expose variable "
              f"importance (expected for Stacked Ensembles). Falling back to the "
              f"best-ranked individual model that does.")
        source_model, source_rank = find_varimp_model(leaderboard)
        varimp_df = source_model.varimp(use_pandas=True) if source_model else None

    if varimp_df is None or varimp_df.empty:
        return [], None, None

    top_features = varimp_df.head(top_n)["variable"].tolist()
    return top_features, source_model, source_rank


def save_varimp_plot(source_model, top_n: int, out_path: Path):
    """Save the variable importance plot, or a placeholder if unavailable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    if source_model is not None:
        source_model.varimp_plot(num_of_features=top_n, server=True)
        plt.savefig(out_path)
        plt.close()
        print(f"[automl_h2o] Saved variable importance plot to {out_path}")
    else:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No model in this H2O AutoML run exposes\nvariable importance.",
                ha="center", va="center", fontsize=11)
        ax.axis("off")
        plt.savefig(out_path)
        plt.close()
        print(f"[automl_h2o] WARNING: wrote placeholder image to {out_path} "
              f"(no model exposed variable importance)")


def log_pass_to_mlflow(best_model, run_label: str, feature_count: int, train_rows: int,
                        elapsed: float, params: dict, source_model, source_rank,
                        leaderboard_path: Path = None):
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"h2o_{run_label}_{best_model.algo}"):
        mlflow.log_params({
            "run_label": run_label,
            "algo": best_model.algo,
            "model_id": best_model.model_id,
            "max_runtime_secs": params["max_runtime_secs"],
            "seed": params["seed"],
            "nfolds": params["nfolds"],
            "feature_count": feature_count,
            "train_rows": train_rows,
            "varimp_source_model_id": source_model.model_id if source_model else "none",
            "varimp_source_rank": source_rank if source_rank is not None else -1,
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
        if leaderboard_path is not None:
            mlflow.log_artifact(str(leaderboard_path))


def write_summary(all_leaderboard, all_best_model, all_top_features,
                   all_source_model, all_source_rank,
                   top_leaderboard, top_best_model, top_features_used,
                   top_result_features, top_source_model, top_source_rank,
                   all_elapsed, top_elapsed, params):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "automl_h2o_summary.txt"

    def varimp_note(best_model, source_model, source_rank):
        if source_model is None:
            return "  (no model in this leaderboard exposes variable importance)"
        if source_rank == 0:
            return f"  (from best model {best_model.model_id})"
        return (f"  (best model {best_model.model_id} is a {best_model.algo} with no "
                f"native variable importance -- expected H2O behavior; features shown "
                f"are from {source_model.model_id}, leaderboard rank {source_rank})")

    lines = [
        "H2O AutoML Summary",
        "=" * 50,
        f"Target: {TARGET_COL}",
        f"Leakage columns excluded: {LEAKAGE_COLS}",
        f"Seed: {params['seed']}   nfolds: {params['nfolds']}   "
        f"max_runtime_secs (per pass): {params['max_runtime_secs']}",
        "",
        "--- Pass 1: All Features ---",
        f"Wall-clock AutoML run time: {all_elapsed:.1f}s",
        f"Best model: {all_best_model.model_id} ({all_best_model.algo})",
        "",
        "Top 3 rows of leaderboard (sorted by RMSE):",
        all_leaderboard.head(3).to_string(),
        "",
        f"Top-{params['top_n_report']} features by variable importance:",
        varimp_note(all_best_model, all_source_model, all_source_rank),
    ]
    if all_top_features:
        lines.extend(f"  - {feat}" for feat in all_top_features)
    else:
        lines.append("  (none available)")
    lines.append("")
    lines.append(f"Features used for Pass 2 rerun (top-{params['top_n_rerun']}): {top_features_used}")
    lines.append("")
    lines.append(f"--- Pass 2: Top-{params['top_n_rerun']} Features Only ---")
    lines.append(f"Wall-clock AutoML run time: {top_elapsed:.1f}s")
    lines.append(f"Best model: {top_best_model.model_id} ({top_best_model.algo})")
    lines.append("")
    lines.append("Top 3 rows of leaderboard (sorted by RMSE):")
    lines.append(top_leaderboard.head(3).to_string())
    lines.append("")
    lines.append(f"Top-{params['top_n_report']} features by variable importance (Pass 2 model):")
    lines.append(varimp_note(top_best_model, top_source_model, top_source_rank))
    if top_result_features:
        lines.extend(f"  - {feat}" for feat in top_result_features)
    else:
        lines.append("  (none available)")
    lines.append("")
    lines.append("Full leaderboards saved to:")
    lines.append("  reports/automl_h2o_leaderboard_all.csv")
    lines.append("  reports/automl_h2o_leaderboard_top_features.csv")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[automl_h2o] Wrote summary to {path}")


def main():
    import h2o

    params = load_params()
    print(f"[automl_h2o] Params: {params}")

    data = load_dataset()

    print("[automl_h2o] Starting local H2O cluster...")
    h2o.init()

    # ==================== Pass 1: all features ====================
    h2o_frame_all = h2o.H2OFrame(data)
    feature_cols_all = [c for c in h2o_frame_all.columns if c != TARGET_COL]

    all_leaderboard, all_aml, all_best_model, all_elapsed = run_automl_pass(
        h2o_frame_all, feature_cols_all, params, run_label="all_features"
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    all_leaderboard_path = REPORTS_DIR / "automl_h2o_leaderboard_all.csv"
    all_leaderboard.to_csv(all_leaderboard_path, index=False)
    print(f"[automl_h2o] Wrote leaderboard to {all_leaderboard_path}")

    all_top_features, all_source_model, all_source_rank = resolve_varimp(
        all_leaderboard, all_best_model, params["top_n_report"]
    )
    save_varimp_plot(all_source_model, params["top_n_report"], OUTPUTS_DIR / "h2o_varimp_all.png")
    log_pass_to_mlflow(all_best_model, "all_features", len(feature_cols_all), data.shape[0],
                        all_elapsed, params, all_source_model, all_source_rank,
                        leaderboard_path=all_leaderboard_path)

    if not all_top_features:
        print("[automl_h2o] ERROR: no model exposes variable importance; "
              "cannot select features for Pass 2. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Rerun feature set = top_n_rerun features from the FULL top_n_report list
    # (top_n_rerun <= top_n_report; if top_n_rerun > top_n_report, params.yaml
    # is misconfigured -- fail loudly rather than silently truncate wrong).
    if params["top_n_rerun"] > len(all_top_features):
        print(f"[automl_h2o] WARNING: top_n_rerun ({params['top_n_rerun']}) exceeds available "
              f"reported features ({len(all_top_features)}); using all available.")
    rerun_features = all_top_features[:params["top_n_rerun"]]

    # ==================== Pass 2: top-N features only ====================
    top_data = data[rerun_features + [TARGET_COL]].copy()
    h2o_frame_top = h2o.H2OFrame(top_data)
    feature_cols_top = [c for c in h2o_frame_top.columns if c != TARGET_COL]

    top_leaderboard, top_aml, top_best_model, top_elapsed = run_automl_pass(
        h2o_frame_top, feature_cols_top, params, run_label="top_features"
    )
    top_leaderboard_path = REPORTS_DIR / "automl_h2o_leaderboard_top_features.csv"
    top_leaderboard.to_csv(top_leaderboard_path, index=False)
    print(f"[automl_h2o] Wrote leaderboard to {top_leaderboard_path}")

    top_result_features, top_source_model, top_source_rank = resolve_varimp(
        top_leaderboard, top_best_model, params["top_n_report"]
    )
    save_varimp_plot(top_source_model, params["top_n_report"], OUTPUTS_DIR / "h2o_varimp_top.png")
    log_pass_to_mlflow(top_best_model, "top_features", len(feature_cols_top), top_data.shape[0],
                        top_elapsed, params, top_source_model, top_source_rank,
                        leaderboard_path=top_leaderboard_path)

    write_summary(
        all_leaderboard, all_best_model, all_top_features, all_source_model, all_source_rank,
        top_leaderboard, top_best_model, rerun_features, top_result_features,
        top_source_model, top_source_rank,
        all_elapsed, top_elapsed, params,
    )

    h2o.cluster().shutdown(prompt=False)
    print("\n[automl_h2o] COMPLETE.")


if __name__ == "__main__":
    main()