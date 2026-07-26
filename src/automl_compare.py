"""
automl_compare.py
-------------------
Assignment #3, Tasks #5-#7 and #9: cross-platform / cross-feature-set /
cross-assignment comparison.

Reads the leaderboards produced by automl_pycaret.py and automl_h2o.py
(each of which now runs two passes: all-features and top-features-only)
and produces a single consolidated comparison report covering:
    - Top 3 models by validation score (PyCaret: all vs. top features)
    - Top 3 models by speed (PyCaret: all vs. top features)
    - Top 3 H2O models by validation score (all vs. top features)
    - Top 3 H2O models by speed (all vs. top features)
    - PyCaret vs. H2O head-to-head comparison
    - Comparison against the Assignment #1 baseline model

The Assignment #1 baseline numbers are NOT re-derived here -- they are
read from params.yaml -> automl_compare.assignment1_baseline, which
must be filled in manually.

Output: reports/automl_comparison.txt
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
PARAMS_PATH = PROJECT_ROOT / "params.yaml"

PYCARET_ALL_PATH = REPORTS_DIR / "automl_pycaret_leaderboard_all.csv"
PYCARET_TOP_PATH = REPORTS_DIR / "automl_pycaret_leaderboard_top_features.csv"
H2O_ALL_PATH = REPORTS_DIR / "automl_h2o_leaderboard_all.csv"
H2O_TOP_PATH = REPORTS_DIR / "automl_h2o_leaderboard_top_features.csv"
OUTPUT_PATH = REPORTS_DIR / "automl_comparison.txt"


def load_params() -> dict:
    defaults = {
        "assignment1_baseline": {
            "model_name": "FILL_IN",
            "validation_metric_name": "FILL_IN e.g. RMSE",
            "validation_score": None,
            "training_time_seconds": None,
            "notes": "Populate from Assignment #1 Part A before finalizing report.",
        }
    }
    if PARAMS_PATH.exists():
        with open(PARAMS_PATH, "r", encoding="utf-8") as f:
            all_params = yaml.safe_load(f) or {}
        return {**defaults, **all_params.get("automl_compare", {})}
    return defaults


def require_file(path: Path, producing_script: str):
    if not path.exists():
        print(f"[automl_compare] ERROR: {path} not found. Run {producing_script} first.",
              file=sys.stderr)
        sys.exit(1)


def top_n_by_score(df: pd.DataFrame, score_col: str, n: int = 3, ascending=False):
    if score_col not in df.columns:
        return df.head(n)
    return df.sort_values(score_col, ascending=ascending).head(n)


def top_n_by_speed(df: pd.DataFrame, time_col: str, n: int = 3):
    if time_col not in df.columns:
        return None
    return df.sort_values(time_col, ascending=True).head(n)


def section(title: str) -> list:
    return ["", f"--- {title} ---", ""]


def main():
    params = load_params()
    baseline = params["assignment1_baseline"]

    require_file(PYCARET_ALL_PATH, "src/automl_pycaret.py")
    require_file(PYCARET_TOP_PATH, "src/automl_pycaret.py")
    require_file(H2O_ALL_PATH, "src/automl_h2o.py")
    require_file(H2O_TOP_PATH, "src/automl_h2o.py")

    pycaret_all = pd.read_csv(PYCARET_ALL_PATH, index_col=0)
    pycaret_top = pd.read_csv(PYCARET_TOP_PATH, index_col=0)
    h2o_all = pd.read_csv(H2O_ALL_PATH)
    h2o_top = pd.read_csv(H2O_TOP_PATH)

    # PyCaret leaderboard: model name is the index; sort/score/time columns
    # vary slightly by PyCaret version but typically include 'R2', 'RMSE',
    # 'MAE', and 'TT (Sec)'.
    pycaret_score_col = "R2" if "R2" in pycaret_all.columns else pycaret_all.columns[0]
    pycaret_time_col = "TT (Sec)" if "TT (Sec)" in pycaret_all.columns else None

    # H2O leaderboard: lower RMSE is better; speed columns only present if
    # get_leaderboard(extra_columns=...) succeeded in automl_h2o.py.
    h2o_score_col = "rmse" if "rmse" in h2o_all.columns else h2o_all.columns[1]
    h2o_time_col = "training_time_ms" if "training_time_ms" in h2o_all.columns else None

    lines = ["AutoML Comparison Report", "=" * 60]

    # ============== Task 5: Top 3 by validation score ==============
    lines += section("Task 5: Top 3 Models by Validation Score - PyCaret")
    lines.append(f"[All features, sorted by {pycaret_score_col}]")
    lines.append(top_n_by_score(pycaret_all, pycaret_score_col, ascending=False).to_string())
    lines.append("")
    lines.append(f"[Top features only, sorted by {pycaret_score_col}]")
    lines.append(top_n_by_score(pycaret_top, pycaret_score_col, ascending=False).to_string())

    all_best_score = pycaret_all.iloc[0][pycaret_score_col] if pycaret_score_col in pycaret_all.columns else None
    top_best_score = pycaret_top.iloc[0][pycaret_score_col] if pycaret_score_col in pycaret_top.columns else None
    if all_best_score is not None and top_best_score is not None:
        delta = top_best_score - all_best_score
        direction = "IMPROVED" if delta > 0 else ("DEGRADED" if delta < 0 else "MAINTAINED")
        lines.append("")
        lines.append(
            f"Feature reduction effect on validation score ({pycaret_score_col}): "
            f"{direction} (all={all_best_score:.4f} -> top={top_best_score:.4f}, "
            f"delta={delta:+.4f})"
        )

    lines += section("Task 9: Top 3 H2O Models by Validation Score")
    lines.append(f"[All features, sorted by {h2o_score_col} ascending (lower is better)]")
    lines.append(top_n_by_score(h2o_all, h2o_score_col, n=3, ascending=True).to_string())
    lines.append("")
    lines.append(f"[Top features only, sorted by {h2o_score_col} ascending]")
    lines.append(top_n_by_score(h2o_top, h2o_score_col, n=3, ascending=True).to_string())

    h2o_all_best = h2o_all.sort_values(h2o_score_col).iloc[0][h2o_score_col] if h2o_score_col in h2o_all.columns else None
    h2o_top_best = h2o_top.sort_values(h2o_score_col).iloc[0][h2o_score_col] if h2o_score_col in h2o_top.columns else None
    if h2o_all_best is not None and h2o_top_best is not None:
        delta = h2o_top_best - h2o_all_best  # RMSE: lower is better, so positive delta = worse
        direction = "IMPROVED" if delta < 0 else ("DEGRADED" if delta > 0 else "MAINTAINED")
        lines.append("")
        lines.append(
            f"Feature reduction effect on validation score ({h2o_score_col}, lower=better): "
            f"{direction} (all={h2o_all_best:.4f} -> top={h2o_top_best:.4f}, delta={delta:+.4f})"
        )

    # ============== Task 6: Top 3 by speed ==============
    lines += section("Task 6: Top 3 Models by Speed - PyCaret (TT (Sec) = training time)")
    if pycaret_time_col:
        lines.append("[All features]")
        lines.append(top_n_by_speed(pycaret_all, pycaret_time_col).to_string())
        lines.append("")
        lines.append("[Top features only]")
        lines.append(top_n_by_speed(pycaret_top, pycaret_time_col).to_string())
    else:
        lines.append("WARNING: 'TT (Sec)' column not found in PyCaret leaderboard "
                      "(PyCaret version dependent) -- speed unavailable from this export.")
    lines.append("")
    lines.append(
        "Speed definition (PyCaret): 'TT (Sec)' is per-model TRAINING time during "
        "compare_models() cross-validation, as reported natively by PyCaret."
    )

    lines += section("Task 9: Top 3 H2O Models by Speed (training_time_ms)")
    if h2o_time_col:
        lines.append("[All features]")
        lines.append(top_n_by_speed(h2o_all, h2o_time_col).to_string())
        lines.append("")
        lines.append("[Top features only]")
        lines.append(top_n_by_speed(h2o_top, h2o_time_col).to_string())
    else:
        lines.append("WARNING: 'training_time_ms' column not available from this H2O "
                      "version's leaderboard -- speed unavailable from this export.")
    lines.append("")
    lines.append(
        "Speed definition (H2O): 'training_time_ms' is per-model TRAINING time in "
        "milliseconds, as reported by h2o.automl.get_leaderboard(extra_columns=...)."
    )

    # ============== PyCaret vs. H2O head-to-head ==============
    lines += section("PyCaret vs. H2O AutoML Head-to-Head (All Features, Best Model Each)")
    lines.append(f"PyCaret best model:  {pycaret_all.index[0]}  "
                  f"({pycaret_score_col}={all_best_score})")
    lines.append(f"H2O best model:      {h2o_all.sort_values(h2o_score_col).iloc[0]['model_id']}  "
                  f"({h2o_score_col}={h2o_all_best})")
    lines.append("")
    lines.append(
        "NOTE: PyCaret sorts/reports R2 (higher=better) by default; H2O here is sorted "
        "by RMSE (lower=better). These are DIFFERENT metrics on DIFFERENT scales and are "
        "not directly comparable as numbers -- compare algorithm choice, relative ranking "
        "behavior, and (if both leaderboards report it) RMSE/MAE side by side instead of "
        "R2-vs-RMSE directly. See both leaderboard CSVs for the full metric sets."
    )

    # ============== Task 7: Assignment #1 baseline comparison ==============
    lines += section("Task 7: Comparison to Assignment #1 Baseline")
    lines.append(f"Baseline model:            {baseline.get('model_name')}")
    lines.append(f"Baseline validation metric: {baseline.get('validation_metric_name')}")
    lines.append(f"Baseline validation score:  {baseline.get('validation_score')}")
    lines.append(f"Baseline training time (s): {baseline.get('training_time_seconds')}")
    lines.append("")
    if baseline.get("validation_score") is None:
        lines.append(
            "NOTE: Assignment #1 baseline values are not yet populated in "
            "params.yaml -> automl_compare.assignment1_baseline. Fill these in "
            "before finalizing the report; this section cannot be auto-computed "
            "since Assignment #1 used a different pipeline/tool."
        )
    else:
        lines.append(f"Best PyCaret model (all features), {pycaret_score_col}: {all_best_score}")
        lines.append(f"Best H2O model (all features), {h2o_score_col}: {h2o_all_best}")
        lines.append(
            "See report narrative for discussion of whether AutoML improved "
            "result, reduced development effort, or introduced new complexity."
        )
    lines.append("")
    lines.append(str(baseline.get("notes", "")))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[automl_compare] Wrote comparison report to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()