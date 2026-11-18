"""Aggregate the complete 51-by-5 clinical refits with paired seed/patient bootstrap."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from proehn.label_noise import SEEDS
from proehn.research.perturbation import weighted
from proehn.research.protocols import require_external_output

MODELS = ("ProEHN gate", "Logistic regression", "Prevalence")
METRICS = ("AUROC", "AUPRC", "Brier", "log_loss")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Completed run_label_noise output directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = require_external_output(args.output)
    protocol = json.loads((args.input / "protocol.json").read_text(encoding="utf-8"))
    ids = y = predictions = None
    for si, seed in enumerate(SEEDS):
        for percent in range(51):
            folder = args.input / f"seed{seed}_noise{percent:02d}"
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            if summary["input_signature"] != protocol["input_signature"]:
                raise ValueError("Mixed source protocols")
            for name, digest in summary["artifact_sha256"].items():
                if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Changed completed artifact: {folder/name}")
            frame = pd.read_csv(
                folder / "predictions.csv", float_precision="round_trip"
            )
            frame = frame[frame.partition.eq("test") & frame.label.isin([0, 1])]
            if ids is None:
                reference = frame[frame.model.eq(MODELS[0])].sort_values("patient_id")
                ids = reference.patient_id.to_numpy()
                y = 1 - reference.label.to_numpy(int)
                predictions = np.empty((len(MODELS), 51, len(SEEDS), len(ids)))
            for mi, model in enumerate(MODELS):
                part = frame[frame.model.eq(model)].set_index("patient_id")
                if not part.index.is_unique or set(part.index) != set(ids):
                    raise ValueError("Unpaired patient predictions")
                part = part.loc[ids]
                np.testing.assert_array_equal(1 - part.label.to_numpy(int), y)
                predictions[mi, percent, si] = part.p_Go.to_numpy(float)
    if not np.isfinite(predictions).all() or np.any(
        (predictions < 0) | (predictions > 1)
    ):
        raise ValueError("Invalid prediction probabilities")
    rng = np.random.default_rng(20261112)
    counts = rng.multinomial(len(y), np.repeat(1 / len(y), len(y)), size=1000)
    seed_counts = rng.multinomial(
        len(SEEDS), np.repeat(1 / len(SEEDS), len(SEEDS)), size=1000
    )
    draws = np.empty((len(MODELS), 51, len(METRICS), 1000))
    points = np.empty(draws.shape[:-1])
    for mi in range(len(MODELS)):
        for percent in range(51):
            perseed = np.array(
                [weighted(y, p, counts) for p in predictions[mi, percent]]
            )
            draws[mi, percent] = np.einsum("sb,skb->kb", seed_counts.T, perseed) / len(
                SEEDS
            )
            points[mi, percent] = np.mean(
                [
                    weighted(y, p, np.ones((1, len(y))))[:, 0]
                    for p in predictions[mi, percent]
                ],
                axis=0,
            )
    rows, noise, contrasts = [], [], []
    for mi, model in enumerate(MODELS):
        for percent in range(51):
            for ki, metric in enumerate(METRICS):
                lo, hi = np.quantile(draws[mi, percent, ki], [0.025, 0.975])
                rows.append(
                    dict(
                        model=model,
                        noise_rate=percent / 100,
                        metric=metric,
                        mean=points[mi, percent, ki],
                        low=lo,
                        high=hi,
                        n_seeds=len(SEEDS),
                        n_test=len(y),
                    )
                )
                lo, hi = np.quantile(
                    draws[mi, percent, ki] - draws[mi, 0, ki], [0.025, 0.975]
                )
                noise.append(
                    dict(
                        model=model,
                        noise_rate=percent / 100,
                        metric=metric,
                        noisy_minus_clean=points[mi, percent, ki] - points[mi, 0, ki],
                        low=lo,
                        high=hi,
                    )
                )
    for percent in range(51):
        for ki, metric in enumerate(METRICS):
            lo, hi = np.quantile(
                draws[0, percent, ki] - draws[1, percent, ki], [0.025, 0.975]
            )
            contrasts.append(
                dict(
                    noise_rate=percent / 100,
                    metric=metric,
                    gate_minus_logistic=points[0, percent, ki] - points[1, percent, ki],
                    low=lo,
                    high=hi,
                )
            )
    args.output.mkdir(parents=True, exist_ok=False)
    for name, records in [
        ("bootstrap_summary", rows),
        ("paired_noise_contrasts", noise),
        ("paired_model_contrasts", contrasts),
    ]:
        pd.DataFrame(records).to_csv(args.output / (name + ".csv"), index=False)
    np.savez_compressed(
        args.output / "bootstrap_draws.npz", draws=draws, models=MODELS, metrics=METRICS
    )


if __name__ == "__main__":
    main()
