"""
run_all.py
----------
Fully automated, end-to-end entry point. Satisfies assignment
requirement #5 ("code must run end-to-end without manual intervention")
for BOTH the ML pipeline stages AND the dataset versioning workflow.

Running this single command:

    python run_all.py

does everything:
    1. Ingest raw data via `dvc repro ingest`     -> data/raw/crossfit.csv
    2. git commit dvc.lock + tag it as dataset-raw
    3. Clean data via `dvc repro preprocess`      -> data/processed/crossfit.csv
    4. git commit dvc.lock + tag it as dataset-processed
    5. Feature engineering, split, train, evaluate (each via `dvc repro <stage>`)
    6. Final git commit of dvc.lock state

Idempotent: safe to re-run. If git/DVC are already initialized, a
remote is already configured, a git tag already exists, or a DVC file
is already tracked and unchanged, those steps are skipped automatically
rather than failing or repeating work.

Requires only that `git` and `dvc` are installed on the machine (as
CLI tools) — actual repo/remote initialization is handled by this
script itself on first run.

Note on naming: the git tags below (`dataset-raw` / `dataset-processed`)
mark DATASET snapshots (raw ingest vs. cleaned/preprocessed) and are a
completely separate versioning axis from the Feast "v1" / "v2" FEATURE
versions used in the experiments/compare stages (see
feature_definitions.py). Dataset snapshots are intentionally NOT named
"v1"/"v2" here to avoid being confused with feature versioning, which
is what assignment requirement #5 (Feature Versioning) actually
evaluates. Feast feature versions are the only place "v1"/"v2" should
appear in this project.
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run(cmd: list, allow_fail: bool = False) -> subprocess.CompletedProcess:
    """Run a command, streaming output, optionally tolerating non-zero exit."""
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False,
                             capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0 and not allow_fail:
        print(f"[run_all] Command failed (exit {result.returncode}): {' '.join(cmd)}",
              file=sys.stderr)
        sys.exit(result.returncode)
    return result


def tag_exists(tag: str) -> bool:
    """Check whether a git tag already exists."""
    result = run(["git", "tag", "-l", tag], allow_fail=True)
    return tag in result.stdout.split()


def dvc_repro_commit_tag(dvc_stage: str, tag: str, commit_msg: str):
    """
    Run a single DVC pipeline stage via `dvc repro` (which executes the
    stage's cmd AND updates dvc.lock in one step, since the stage's
    output is already declared in dvc.yaml), then git-commit dvc.lock
    and tag the commit. Skips steps already done (idempotent).
    """
    print(f"\n=== Running stage '{dvc_stage}' and versioning as '{tag}' ===")

    run(["dvc", "repro", dvc_stage])
    run(["git", "add", "dvc.lock", ".gitignore"], allow_fail=True)

    diff_check = run(["git", "diff", "--cached", "--quiet"], allow_fail=True)
    if diff_check.returncode != 0:
        run(["git", "commit", "-m", commit_msg])
    else:
        print(f"[run_all] No changes to commit for stage '{dvc_stage}' (already up to date).")

    if tag_exists(tag):
        print(f"[run_all] Tag '{tag}' already exists, skipping tag creation.")
    else:
        run(["git", "tag", tag])
        print(f"[run_all] Created tag '{tag}'.")

    run(["dvc", "push"], allow_fail=True)


def ensure_git_initialized():
    """Run `git init` only if this project isn't already a git repo."""
    git_dir = PROJECT_ROOT / ".git"
    if git_dir.exists():
        print("[run_all] Git already initialized, skipping.")
        return
    print("[run_all] Git not initialized — running git init.")
    run(["git", "init"])


def ensure_dvc_initialized():
    """Run `dvc init` only if this project isn't already a DVC repo."""
    dvc_dir = PROJECT_ROOT / ".dvc"
    if dvc_dir.exists():
        print("[run_all] DVC already initialized, skipping.")
        return
    print("[run_all] DVC not initialized — running dvc init.")
    run(["dvc", "init"])


def ensure_dvc_remote():
    """Configure a local DVC remote only if one isn't already set up."""
    result = run(["dvc", "remote", "list"], allow_fail=True)
    if result.stdout.strip():
        print("[run_all] DVC remote already configured, skipping.")
        return

    print("[run_all] No DVC remote configured — setting up local remote.")
    storage_path = PROJECT_ROOT.parent / "dvc_storage"
    storage_path.mkdir(parents=True, exist_ok=True)
    run(["dvc", "remote", "add", "-d", "localstorage", str(storage_path)])


def ensure_initial_commit():
    """
    Make an initial commit of DVC config files if nothing has been
    committed yet (dvc add later requires at least one commit to exist
    cleanly in some setups, and this also captures .dvc/config + .gitignore).
    """
    result = run(["git", "log", "-1"], allow_fail=True)
    if result.returncode == 0:
        print("[run_all] Git history already exists, skipping initial commit.")
        return

    print("[run_all] No commits yet — creating initial commit.")
    run(["git", "add", ".dvc/config", ".gitignore"], allow_fail=True)
    run(["git", "commit", "-m", "Configure DVC"], allow_fail=True)


def ensure_environment_ready():
    """Run all one-time setup checks, each skipped automatically if already done."""
    print("\n" + "=" * 60)
    print("Checking git/DVC environment setup")
    print("=" * 60)
    ensure_git_initialized()
    ensure_dvc_initialized()
    ensure_dvc_remote()
    ensure_initial_commit()


def main():
    """Run the full pipeline end-to-end, including dataset versioning, with no manual steps."""
    ensure_environment_ready()

    # --- Stage 1: ingest -> dataset-raw ---
    dvc_repro_commit_tag("ingest", "dataset-raw", "dataset-raw: raw ingested dataset")

    # --- Stage 2: preprocess -> dataset-processed ---
    dvc_repro_commit_tag("preprocess", "dataset-processed", "dataset-processed: cleaned and transformed dataset")

    # --- Task 4: EDA on raw and processed snapshots, before feature engineering ---
    run(["dvc", "repro", "eda"])

    # --- Stage 3: feature engineering ---
    run(["dvc", "repro", "feature_engineering"])

    # --- Feature Store setup ---
    run(["dvc", "repro", "feature_store_setup"])

    # --- Stages 4-7: Run experiments (split -> train -> evaluate) ---
    # Handles multiple Feast feature versions (v1/v2) and hyperparameter combinations.
    # NOTE: these Feast "v1"/"v2" feature versions are unrelated to the
    # "dataset-raw"/"dataset-processed" git tags above — see module docstring.
    run(["dvc", "repro", "experiments"])

    # --- Feature version comparison (v1 vs v2 Feast feature sets, via MLflow) ---
    run(["dvc", "repro", "compare"])

    # --- Assignment #3: AutoML (PyCaret + H2O), run on data/processed/crossfit_features.csv ---
    run(["dvc", "repro", "automl_pycaret"])
    run(["dvc", "repro", "automl_h2o"])
    run(["dvc", "repro", "automl_compare"])

    # --- Snapshot the final pipeline state ---
    run(["git", "add", "dvc.lock"], allow_fail=True)

    diff_check = run(["git", "diff", "--cached", "--quiet"], allow_fail=True)

    if diff_check.returncode != 0:
        run(
            [
                "git",
                "commit",
                "-m",
                "Full pipeline run: experiments, evaluation, comparison, and AutoML",
            ]
        )
    else:
        print("[run_all] No pipeline-state changes to commit.")

    run(["dvc", "push"], allow_fail=True)

    print("\n" + "=" * 60)
    print("[run_all] COMPLETE. All stages ran; dataset-raw and dataset-processed are tagged in git.")
    print("=" * 60)
    print("Verify with: git log --oneline --tags")


if __name__ == "__main__":
    main()