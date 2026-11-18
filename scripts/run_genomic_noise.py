"""Measure frozen clinical-gate robustness at every integer0..50% genotype noise."""

import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import jax.numpy as jnp
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)
from proehn.kinetic import ProEHNKineticGatekeeper
from proehn.research.perturbation import weighted
from proehn.research.protocols import require_external_output, save_json

SEEDS = [11, 23, 37, 53, 71]
RATES = np.arange(51) / 100
METRICS = ["AUROC", "AUPRC", "Brier", "log_loss"]
GROUPS = ["pt_genomic", "pt_dynamic", "mt_genomic", "mt_dynamic", "shared"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, choices=["GENIE", "MELA", "PACA"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    args = parser.parse_args()
    args.output = require_external_output(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.data)
    test = frame[frame.fold.eq(args.fold)].copy()
    known = test.Patient_Label.ne(-1).to_numpy()
    y = 1 - test.Patient_Label.to_numpy()[known].astype(int)
    if set(y) != {0, 1}:
        raise ValueError("Held-out known labels must contain Go and Stop")
    columns = [column for column in test if re.fullmatch(r"[PM]\..+ \(M\)", column)]
    if len(columns) != 24:
        raise ValueError(
            "Exactly 24 tissue-specific frozen 12-gene genotype columns are required"
        )
    original = test[columns].to_numpy(float)
    valid = np.isfinite(original)
    if not np.isin(original[valid], [0, 1]).all():
        raise ValueError("Observed genotype cells must be binary")
    binding = dict(
        cohort=args.cohort,
        fold=args.fold,
        seeds=SEEDS,
        rates=RATES.tolist(),
        source_sha256={
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in [
                ("data", args.data),
                ("params", args.params),
                ("metadata", args.metadata),
            ]
        },
    )
    path = args.output / "protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != binding:
            raise ValueError("Existing output belongs to another frozen experiment")
    else:
        save_json(path, binding)
    gate = ProEHNKineticGatekeeper(args.params, args.metadata)

    def score(part):
        x = gate.preprocessor.transform(part)
        logits = np.asarray(
            gate.model.apply(
                {"params": gate.params},
                *[jnp.asarray(x[group]) for group in GROUPS],
                train=False,
            )
        )
        return 1 - gate.calibration.transform(logits)

    predfile = args.output / "predictions.npz"
    if predfile.exists():
        saved = np.load(predfile)
        predictions = saved["predictions"]
        np.testing.assert_array_equal(saved["y"], y)
        np.testing.assert_array_equal(
            saved["patient_ids"], test.patient_id[known].to_numpy(dtype=str)
        )
    else:
        predictions = np.empty((5, 51, len(y)))
        valid_index = np.flatnonzero(valid.ravel())
        clean = score(test)[known]
        for si, seed in enumerate(SEEDS):
            order = np.random.default_rng(200000 + seed).permutation(valid_index)
            for ri, rate in enumerate(RATES):
                if ri == 0:
                    predictions[si, ri] = clean
                    continue
                count = int(np.floor(rate * len(valid_index)))
                bits = original.copy().ravel()
                bits[order[:count]] = 1 - bits[order[:count]]
                noisy = test.copy()
                noisy.loc[:, columns] = bits.reshape(original.shape)
                np.testing.assert_array_equal(
                    np.isnan(noisy[columns].to_numpy()), ~valid
                )
                predictions[si, ri] = score(noisy)[known]
            print("COMPLETE", args.cohort, "noise seed", seed, flush=True)
        np.savez_compressed(
            predfile,
            predictions=predictions,
            y=y,
            patient_ids=test.patient_id[known].to_numpy(dtype=str),
        )
    if predictions.shape != (5, 51, len(y)) or not np.isfinite(predictions).all():
        raise ValueError("Invalid cached noise prediction shape or values")
    rows = []
    for si, seed in enumerate(SEEDS):
        for ri, rate in enumerate(RATES):
            p = predictions[si, ri]
            values = [
                roc_auc_score(y, p),
                average_precision_score(y, p),
                brier_score_loss(y, p),
                log_loss(y, p, labels=[0, 1]),
            ]
            np.testing.assert_allclose(
                weighted(y, p, np.ones((1, len(y))))[:, 0], values, atol=1e-12, rtol=0
            )
            rows.append(
                dict(
                    cohort=args.cohort,
                    seed=seed,
                    noise_rate=rate,
                    n_test=len(y),
                    flipped_cells=int(np.floor(rate * valid.sum())),
                    observed_cells=int(valid.sum()),
                    **dict(zip(METRICS, map(float, values))),
                )
            )
    data = pd.DataFrame(rows)
    data.to_csv(args.output / "all_condition_metrics.csv", index=False)
    rng = np.random.default_rng(20261112)
    reps = 1000
    counts = np.zeros((reps, len(y)), int)
    for label in [0, 1]:
        ids = np.flatnonzero(y == label)
        counts[:, ids] = rng.multinomial(
            len(ids), np.repeat(1 / len(ids), len(ids)), size=reps
        )
    seed_counts = rng.multinomial(5, np.repeat(0.2, 5), size=reps)
    bootstrap = np.empty((51, 4, reps))
    summary = []
    contrasts = []
    for ri, rate in enumerate(RATES):
        values = np.array([weighted(y, p, counts) for p in predictions[:, ri]])
        bootstrap[ri] = np.einsum("sb,skb->kb", seed_counts.T, values) / 5
        point = data[data.noise_rate.eq(rate)][METRICS].mean().to_numpy()
        for mi, metric in enumerate(METRICS):
            low, high = np.quantile(bootstrap[ri, mi], [0.025, 0.975])
            summary.append(
                dict(
                    cohort=args.cohort,
                    noise_rate=rate,
                    metric=metric,
                    mean=float(point[mi]),
                    low=float(low),
                    high=float(high),
                    n_test=len(y),
                    n_seeds=5,
                )
            )
    for ri, rate in enumerate(RATES):
        for mi, metric in enumerate(METRICS):
            low, high = np.quantile(
                bootstrap[ri, mi] - bootstrap[0, mi], [0.025, 0.975]
            )
            delta = (
                data[data.noise_rate.eq(rate)][metric].mean()
                - data[data.noise_rate.eq(0)][metric].mean()
            )
            contrasts.append(
                dict(
                    cohort=args.cohort,
                    noise_rate=rate,
                    metric=metric,
                    noisy_minus_clean=float(delta),
                    low=float(low),
                    high=float(high),
                )
            )
    pd.DataFrame(summary).to_csv(args.output / "bootstrap_summary.csv", index=False)
    pd.DataFrame(contrasts).to_csv(
        args.output / "paired_noise_contrasts.csv", index=False
    )
    np.savez_compressed(
        args.output / "bootstrap_draws.npz", values=bootstrap, metrics=np.array(METRICS)
    )
    save_json(
        args.output / "completion.json",
        dict(
            status="COMPLETE",
            conditions=255,
            measured_rates=51,
            bootstrap_replicates=reps,
            bootstrap_seed=20261112,
            training_refits=0,
            bootstrap_units=["stratified patients", "paired noise seeds"],
            pointwise_intervals=True,
        ),
    )


if __name__ == "__main__":
    main()
