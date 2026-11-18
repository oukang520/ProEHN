"""Nested train-only label flips with clean validation and test outcomes."""

from __future__ import annotations
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    matthews_corrcoef,
)
from .probability import PlattCalibration
from .labels import build_progression_label, label_provenance
from .training import fit_kinetic_model

GROUPS = ("pt_genomic", "pt_dynamic", "mt_genomic", "mt_dynamic", "shared")
SEEDS = (11, 23, 37, 53, 71)


def noise_mask(labels, rate, seed):
    labels = np.asarray(labels)
    if not np.isin(labels, [-1, 0, 1]).all() or not 0 <= rate <= 0.5:
        raise ValueError("Known binary labels and a rate in [0,.5] are required")
    known = np.flatnonzero(labels != -1)
    order = np.random.default_rng(100000 + int(seed)).permutation(known)
    mask = np.zeros(len(labels), bool)
    mask[order[: int(np.floor(float(rate) * len(known)))]] = True
    return mask


def gate_probabilities(gate, frame):
    x = gate.preprocessor.transform(frame)
    logits = np.asarray(
        gate.model.apply(
            {"params": gate.params}, *[jnp.asarray(x[k]) for k in GROUPS], train=False
        )
    )
    return 1 - gate.calibration.transform(logits), expit(-logits)


def _signature(frames, config, schema):
    digest = hashlib.sha256(
        json.dumps(
            dict(config=config, labels=asdict(schema)), sort_keys=True, default=str
        ).encode()
    )
    for frame in frames:
        digest.update(
            pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
        )
        digest.update("\n".join(map(str, frame.columns)).encode())
    for filename in [
        "label_noise.py",
        "training.py",
        "kinetic.py",
        "preprocessing.py",
        "features.py",
        "probability.py",
        "labels.py",
    ]:
        digest.update((Path(__file__).parent / filename).read_bytes())
    return digest.hexdigest()


def run_noise(
    training, validation, test, *, model_config, label_schema, output, seeds=SEEDS
):
    """Refit at every integer percentage 0..50; completed matching fits are reused."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    parts = []
    for frame in (training, validation, test):
        frame = frame.copy()
        if frame.patient_id.isna().any() or frame.patient_id.duplicated().any():
            raise ValueError("One row per patient is required")
        frame["Patient_Label"] = build_progression_label(frame, label_schema)
        parts.append(frame)
    training, validation, test = parts
    if any(
        set(parts[a].patient_id) & set(parts[b].patient_id)
        for a, b in [(0, 1), (0, 2), (1, 2)]
    ):
        raise ValueError("Training, validation and test patients overlap")
    signature = _signature(parts, model_config, label_schema)
    manifest = output / "protocol.json"
    protocol = dict(
        input_signature=signature,
        noise_percent=list(range(51)),
        seeds=list(SEEDS),
        labels=asdict(label_schema),
        model_config=model_config,
    )
    try:
        with manifest.open("x", encoding="utf-8") as handle:
            json.dump(protocol, handle, indent=2, default=str)
            handle.write("\n")
    except FileExistsError:
        if (
            json.loads(manifest.read_text(encoding="utf-8")).get("input_signature")
            != signature
        ):
            raise ValueError(
                "Output directory belongs to different inputs or implementation"
            )
    known_val = validation.Patient_Label.ne(-1).to_numpy()
    known_test = test.Patient_Label.ne(-1).to_numpy()
    yv = 1 - validation.Patient_Label.to_numpy()[known_val]
    yt = 1 - test.Patient_Label.to_numpy()[known_test]
    if set(yv) != {0, 1} or set(yt) != {0, 1}:
        raise ValueError(
            "Clean validation and test both require the two clinical classes"
        )
    rows = []
    for seed in seeds:
        if seed not in SEEDS:
            raise ValueError(f"Published seeds are {SEEDS}")
        for percent in range(51):
            rate = percent / 100
            target = output / f"seed{seed}_noise{percent:02d}"
            completed = target / "summary.json"
            if completed.exists():
                old = json.loads(completed.read_text(encoding="utf-8"))
                if (
                    old["input_signature"] != signature
                    or old["seed"] != seed
                    or old["percent"] != percent
                ):
                    raise ValueError(f"Existing output uses different inputs: {target}")
                if not old.get("artifact_sha256"):
                    raise ValueError(f"Existing output lacks artifact hashes: {target}")
                for filename, digest in old["artifact_sha256"].items():
                    path = target / filename
                    if (
                        not path.is_file()
                        or hashlib.sha256(path.read_bytes()).hexdigest() != digest
                    ):
                        raise ValueError(
                            f"Completed output changed or is missing: {path}"
                        )
                rows.extend(old["metrics"])
                continue
            target.mkdir(parents=True, exist_ok=True)
            corrupted = training.copy()
            corrupted["Noisy_Patient_Label"] = corrupted.Patient_Label
            mask = noise_mask(corrupted.Patient_Label, rate, seed)
            corrupted.loc[mask, "Noisy_Patient_Label"] = (
                1 - corrupted.loc[mask, "Patient_Label"]
            )
            val = validation.copy()
            val["Noisy_Patient_Label"] = val.Patient_Label
            for frame in (corrupted, val):
                frame.attrs["label_provenance"] = dict(
                    frame.attrs.get("label_provenance", {}),
                    Noisy_Patient_Label=label_provenance(label_schema),
                )
            corrupted[["patient_id", "Patient_Label", "Noisy_Patient_Label"]].assign(
                flipped=mask
            ).to_csv(target / "training_labels.csv", index=False)
            gate = fit_kinetic_model(
                corrupted,
                val,
                patient_id_column="patient_id",
                label_column="Noisy_Patient_Label",
                model_config=dict(model_config, random_seed=seed),
                label_schema=label_schema,
            )
            gate.save(target / "gate.msgpack", target / "gate_metadata.pkl")
            gate_predictions = [gate_probabilities(gate, f) for f in (validation, test)]
            probabilities = {"ProEHN gate": [values[0] for values in gate_predictions]}
            raw_probabilities = {
                "ProEHN gate": [values[1] for values in gate_predictions]
            }
            matrices = {
                name: np.concatenate([x[k] for k in GROUPS], axis=1)
                for name, x in (
                    (name, gate.preprocessor.transform(f))
                    for name, f in [
                        ("train", training),
                        ("validation", validation),
                        ("test", test),
                    ]
                )
            }
            known = corrupted.Noisy_Patient_Label.ne(-1).to_numpy()
            ytrain = 1 - corrupted.Noisy_Patient_Label.to_numpy()[known]
            reference = LogisticRegression(C=1.0, max_iter=2000, random_state=11).fit(
                matrices["train"][known], ytrain
            )
            calibration = PlattCalibration.fit(
                reference.decision_function(matrices["validation"])[known_val],
                yv,
                validation.patient_id.to_numpy()[known_val],
                training_patient_ids=training.patient_id,
            )
            probabilities["Logistic regression"] = [
                calibration.transform(reference.decision_function(matrices[n]))
                for n in ("validation", "test")
            ]
            raw_probabilities["Logistic regression"] = [
                expit(reference.decision_function(matrices[n]))
                for n in ("validation", "test")
            ]
            probabilities["Prevalence"] = [
                np.repeat(ytrain.mean(), len(f)) for f in (validation, test)
            ]
            raw_probabilities["Prevalence"] = probabilities["Prevalence"]
            run_rows, predictions = [], []
            for model, (pv, pe) in probabilities.items():
                fpr, tpr, threshold = roc_curve(yv, pv[known_val])
                finite = np.isfinite(threshold)
                cutoff = float(threshold[finite][np.argmax((tpr - fpr)[finite])])
                p = pe[known_test]
                tn, fp, fn, tp = confusion_matrix(
                    yt, p >= cutoff, labels=[0, 1]
                ).ravel()
                bins = np.minimum((p * 10).astype(int), 9)
                ece = sum(
                    abs(float(np.sum((p - yt)[bins == b]))) for b in range(10)
                ) / len(yt)
                run_rows.append(
                    dict(
                        seed=seed,
                        noise_rate=rate,
                        model=model,
                        n_test=len(yt),
                        AUROC=float(roc_auc_score(yt, p)),
                        AUPRC=float(average_precision_score(yt, p)),
                        Brier=float(brier_score_loss(yt, p)),
                        log_loss=float(log_loss(yt, p, labels=[0, 1])),
                        ECE10=ece,
                        raw_AUROC=float(
                            roc_auc_score(yt, raw_probabilities[model][1][known_test])
                        ),
                        sensitivity=float(tp / (tp + fn)),
                        specificity=float(tn / (tn + fp)),
                        accuracy=float((tp + tn) / len(yt)),
                        MCC=float(matthews_corrcoef(yt, p >= cutoff)),
                        TP=int(tp),
                        FP=int(fp),
                        TN=int(tn),
                        FN=int(fn),
                        validation_threshold=cutoff,
                    )
                )
                for pi, (part, frame, values) in enumerate(
                    [("validation", validation, pv), ("test", test, pe)]
                ):
                    predictions.append(
                        pd.DataFrame(
                            dict(
                                patient_id=frame.patient_id,
                                partition=part,
                                label=frame.Patient_Label,
                                model=model,
                                p_Go=values,
                                raw_p_Go=raw_probabilities[model][pi],
                            )
                        )
                    )
            pd.concat(predictions, ignore_index=True).to_csv(
                target / "predictions.csv", index=False
            )
            summary = dict(
                input_signature=signature,
                seed=seed,
                percent=percent,
                flipped=int(mask.sum()),
                known_training=int(known.sum()),
                metrics=run_rows,
                flip_Go_to_Stop=int(
                    np.sum(mask & training.Patient_Label.eq(0).to_numpy())
                ),
                flip_Stop_to_Go=int(
                    np.sum(mask & training.Patient_Label.eq(1).to_numpy())
                ),
                fit_diagnostics=gate.fit_diagnostics,
                artifact_sha256={
                    filename: hashlib.sha256(
                        (target / filename).read_bytes()
                    ).hexdigest()
                    for filename in [
                        "gate.msgpack",
                        "gate_metadata.pkl",
                        "predictions.csv",
                        "training_labels.csv",
                    ]
                },
            )
            temporary = completed.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            temporary.replace(completed)
            rows.extend(run_rows)
    rows = []
    for path in sorted(output.glob("seed*_noise*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("input_signature") != signature:
            raise ValueError(f"Mixed run inputs in {path}")
        rows.extend(summary["metrics"])
    result = pd.DataFrame(rows)
    temporary = output / f"metrics_seed{seeds[0]}.tmp"
    result.to_csv(temporary, index=False)
    temporary.replace(output / "metrics.csv")
    return result
