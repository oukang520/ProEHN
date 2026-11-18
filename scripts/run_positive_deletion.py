"""Score frozen four-training-fold models and pool patient-macro Top-k recovery."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from proehn.research.topk import score_fold
from proehn.research.protocols import (
    GENES,
    patient_digest,
    require_external_output,
    save_json,
)

METRICS = ["Top1", "Top3", "Top5", "MRR", "target_log_score"]


def load_model(directory, fold, kind, training):
    candidates = [
        directory / f"fold{fold}" / "model.json",
        directory / f"fold{fold}" / (kind + "_fit.json"),
    ]
    available = [path for path in candidates if path.exists()]
    if not available:
        raise FileNotFoundError(
            f"No {kind} model for fold{fold}; fit this partition explicitly with train_published.py"
        )
    artifact = json.loads(available[0].read_text(encoding="utf-8"))
    digest = artifact.get(
        "training_patient_sha256",
        artifact.get("run_binding", {}).get("training_patient_sha256"),
    )
    if digest != patient_digest(training):
        raise ValueError(
            "The model training-patient partition differs from this outer fold"
        )
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, choices=GENES)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--context-models", type=Path, required=True)
    parser.add_argument("--baseline-models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = require_external_output(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(
        args.data, float_precision="round_trip" if args.cohort != "GENIE" else None
    )
    if frame.patient_id.duplicated().any():
        raise ValueError("One native observed record per patient is required")
    for fold in range(5):
        training = frame[frame.fold.ne(fold)]
        fitted = {
            name: load_model(directory, fold, kind, training)
            for name, directory, kind in [
                ("Context model", args.context_models, "context"),
                ("Coupled baseline", args.baseline_models, "baseline"),
            ]
        }
        folder = args.output / f"fold{fold}"
        folder.mkdir(parents=True, exist_ok=True)
        score_fold(args.cohort, fold, frame, GENES[args.cohort], fitted, folder)
    patient = pd.concat(
        [
            pd.read_csv(
                args.output / f"fold{fold}" / "patient_metrics_private.csv"
            ).assign(fold=fold)
            for fold in range(5)
        ],
        ignore_index=True,
    )
    if patient.duplicated(["model", "patient_id"]).any():
        raise ValueError("A patient occurred in multiple outer test folds")
    pivot = {
        model: subset.set_index("patient_id")[METRICS].sort_index()
        for model, subset in patient.groupby("model")
    }
    ids = next(iter(pivot.values())).index
    if not all(table.index.equals(ids) for table in pivot.values()):
        raise ValueError("Model query populations differ")
    indices = np.random.default_rng(20261111).integers(0, len(ids), (1000, len(ids)))
    metrics = []
    pairs = []
    for model, table in pivot.items():
        for metric in METRICS:
            values = table[metric].to_numpy()
            low, high = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
            metrics.append(
                dict(
                    cohort=args.cohort,
                    model=model,
                    metric=metric,
                    n_patients=len(ids),
                    value=float(values.mean()),
                    low=float(low),
                    high=float(high),
                )
            )
    for comparator in ["Coupled baseline", "Training prevalence", "Uniform random"]:
        for metric in METRICS:
            delta = (
                pivot["Context model"][metric] - pivot[comparator][metric]
            ).to_numpy()
            low, high = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])
            pairs.append(
                dict(
                    cohort=args.cohort,
                    comparator=comparator,
                    metric=metric,
                    context_minus_comparator=float(delta.mean()),
                    low=float(low),
                    high=float(high),
                    n_patients=len(ids),
                )
            )
    patient.to_csv(args.output / "OOF_patient_metrics.csv", index=False)
    pd.DataFrame(metrics).to_csv(
        args.output / "pooled_patient_metrics.csv", index=False
    )
    pd.DataFrame(pairs).to_csv(args.output / "paired_comparisons.csv", index=False)
    save_json(
        args.output / "completion.json",
        dict(
            status="COMPLETE",
            cohort=args.cohort,
            outer_folds=5,
            eligible_patients=len(ids),
            bootstrap_replicates=1000,
            bootstrap_seed=20261111,
            model_refits=0,
        ),
    )


if __name__ == "__main__":
    main()
