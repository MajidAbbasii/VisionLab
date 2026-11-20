"""Combine available experiment CSVs into a Markdown experiment summary."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import pandas as pd
from .common import load_yaml, project_path

FILES = {
    "Hyperparameter tuning": "results/metrics/hyperparameter_trials.csv",
    "Sensitivity analysis": "results/metrics/sensitivity.csv",
    "Ablation study": "results/metrics/ablation.csv",
    "Training history": "results/metrics/training_metrics.csv",
}

def build_report() -> str:
    """Build a Markdown report from current metrics and flag dataset leakage."""
    sections = ["# VisionLab Experiment Report", "",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}", "",
        "> This report only includes experiment artifacts found on disk. Missing sections mean the corresponding experiment has not been run.", ""]

    dataset = load_yaml("datasets/coco128.yaml")
    train_split = str(dataset.get("train", "not configured"))
    val_split = str(dataset.get("val", "not configured"))
    sections += ["## Dataset and validation integrity", "",
        f"- Dataset YAML: `datasets/coco128.yaml`",
        f"- Train split: `{train_split}`",
        f"- Validation split: `{val_split}`"]
    if train_split.strip() == val_split.strip():
        sections += ["",
            "> **Critical limitation:** train and validation point to the same configured split. "
            "All current validation metrics, tuning objectives, sensitivity results and ablation scores are exploratory only; "
            "they do not measure generalization to unseen images. Do not select or replace a production model from these scores.", ""]
    else:
        sections += ["",
            "> The configured split paths differ. This report does not verify file-level overlap or near-duplicate leakage.", ""]

    experiment_cfg = load_yaml("configs/experiments.yaml")
    configured_study = str(experiment_cfg.get("tuning", {}).get("study_name", "not configured"))
    trial_csv = project_path("results/metrics/hyperparameter_trials.csv")
    sections += ["## Optuna study provenance", "", f"- Study configured for the next run: `{configured_study}`"]
    if trial_csv.is_file():
        trials = pd.read_csv(trial_csv)
        study_names = set()
        if "user_attrs_json" in trials.columns:
            for raw in trials["user_attrs_json"].dropna():
                try:
                    run_dir = json.loads(raw).get("run_dir")
                except (TypeError, ValueError):
                    run_dir = None
                if run_dir:
                    folder = str(run_dir).replace("\\", "/").rsplit("/", 1)[-1]
                    if "_trial_" in folder:
                        study_names.add(folder.rsplit("_trial_", 1)[0])
        states = trials["state"].value_counts().to_dict() if "state" in trials.columns else {}
        sections += [f"- Study represented by the consolidated trial CSV: `{', '.join(sorted(study_names)) or 'unknown'}`",
            f"- Exported trial rows: {len(trials)}",
            f"- Trial states: `{json.dumps(states, sort_keys=True)}`"]
        completed = trials[trials.get("state", pd.Series(dtype=str)) == "COMPLETE"]
        if not completed.empty and "objective_map50_95" in completed.columns:
            best = completed.loc[pd.to_numeric(completed["objective_map50_95"], errors="coerce").idxmax()]
            sections.append(
                f"- Best completed trial in this CSV: #{int(best['trial_number'])} "
                f"(mAP50-95={float(best['objective_map50_95']):.5f}; exploratory because of the shared split)."
            )
        if study_names and configured_study not in study_names:
            sections += ["- **Provenance note:** the existing trial CSV belongs to a different study than the current configuration. The CSV is historical output; a run using the current configuration will create/resume its configured study separately."]
    sections.append("")

    for title, relative in FILES.items():
        path = project_path(relative)
        sections += [f"## {title}", ""]
        if not path.exists():
            sections += [f"Not run yet. Expected artifact: `{relative}`", ""]
            continue
        frame = pd.read_csv(path)
        sections += [frame.to_markdown(index=False), ""]
    text = "\n".join(sections)
    target = project_path("results/reports/experiment_report.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.parse_args()
    print(build_report())

