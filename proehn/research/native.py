"""Cohort-specific native fits and inner-selected snapshot context models."""

import time
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score
from proehn.training import fit_topology_model
from .likelihood import records, training_buckets, loglikelihoods, compartment_masks
from .preprocessing import (
    genie_fit_transform,
    genie_transform,
    icgc_fit_transform,
    icgc_transform,
)
from .diagonal import fit_diagonal
from . import partial_pool
from .protocols import BASE_RIDGE, FULL_OPTIMIZER


def fit_native(frame, genes, cohort, *, constant=False, progress=None):
    """Fit raw prepared patient records, with a transform learned only here."""
    if len(genes) != 12 or cohort not in ("GENIE", "MELA", "PACA"):
        raise ValueError(
            "A registered cohort and frozen 12-gene vocabulary are required"
        )
    if not len(frame):
        raise ValueError("Empty training partition")
    started = time.time()
    if cohort == "GENIE":
        preprocessing = genie_fit_transform(frame)
        nf = 0 if constant else 5
        features = (
            np.ones((len(frame), 1))
            if constant
            else genie_transform(frame, preprocessing)
        )
        grouped, weights = training_buckets(records(frame, genes, features))
        model, parameters, _ = fit_topology_model(
            grouped,
            12,
            nf,
            len(frame),
            bucket_sample_weights=weights,
            random_seed=11,
            progress_callback=progress,
            **FULL_OPTIMIZER,
        )
        result = dict(
            parameters=parameters.tolist(),
            nf=nf,
            preprocessing=preprocessing,
            preprocessing_kind="GENIE_original",
            diagnostics=model.fit_diagnostics,
        )
    else:
        spec = dict(
            name="constant" if constant else "diagonal_burden_vaf",
            mode="constant" if constant else "diagonal",
            columns=[] if constant else [1, 2],
            base_ridge=BASE_RIDGE[cohort],
            context_ridge=0.0 if constant else 0.05,
        )
        result = fit_diagonal(
            frame, genes, icgc_fit_transform(frame), spec, progress=progress
        )
        result["preprocessing_kind"] = "ICGC_diagonal"
    result.update(
        cohort=cohort,
        n_train=len(frame),
        status="COMPLETE",
        seconds=time.time() - started,
    )
    return result


def feature_matrix(artifact, frame):
    if artifact["preprocessing_kind"] == "GENIE_original":
        return (
            np.ones((len(frame), 1))
            if artifact["nf"] == 0
            else genie_transform(frame, artifact["preprocessing"])
        )
    return icgc_transform(frame, artifact["preprocessing"], artifact["spec"]["columns"])


def record_scores(artifact, frame, genes):
    return loglikelihoods(
        np.asarray(artifact["parameters"]),
        records(frame, genes, feature_matrix(artifact, frame)),
        artifact["nf"],
    )


def predict_masked(artifact, frame, genes, compartment="PT"):
    """Native conditional baseline status, retaining other permitted tissue data."""
    source = records(frame, genes, feature_matrix(artifact, frame))
    masked, target, selected = compartment_masks(source, compartment)
    if not len(selected):
        return pd.DataFrame(columns=["patient_id", "gene", "label", "prediction"])
    likelihood = loglikelihoods(
        np.asarray(artifact["parameters"]), masked, artifact["nf"]
    )
    likelihood = likelihood.reshape(len(selected), 12, 2)
    probability = expit(likelihood[:, :, 1] - likelihood[:, :, 0])
    return pd.concat(
        [
            pd.DataFrame(
                dict(
                    patient_id=frame.iloc[selected].patient_id.to_numpy(),
                    gene=gene,
                    label=target[:, j],
                    prediction=probability[:, j],
                )
            )
            for j, gene in enumerate(genes)
        ],
        ignore_index=True,
    )


def _prediction_metrics(table, genes, eligible):
    rows = []
    for gene in genes:
        subset = table[table.gene.eq(gene)]
        y, p = subset.label.to_numpy(), subset.prediction.to_numpy()
        two = 0 < y.sum() < len(y)
        rows.append(
            dict(
                gene=gene,
                conditional_logloss=float(log_loss(y, p, labels=[0, 1])),
                AUROC=float(roc_auc_score(y, p)) if two else None,
                AUPRC=float(average_precision_score(y, p)) if two else None,
            )
        )
    result = pd.DataFrame(rows).set_index("gene").loc[eligible]
    return dict(
        conditional_macro_logloss=(
            float(result.conditional_logloss.mean()) if len(result) else None
        ),
        macro_AUROC=float(result.AUROC.mean()) if len(result) else None,
        macro_AUPRC=float(result.AUPRC.mean()) if len(result) else None,
    )


def select_snapshot_candidate(
    train, validation, genes, cohort, baseline, legacy, *, progress=None
):
    """Four independent reduced fits; no test rows enter the promotion rule."""
    if cohort == "GENIE" or len(train) >= 1000:
        return dict(legacy, selected_candidate="legacy"), []
    reference = predict_masked(legacy, validation, genes)
    eligible = [
        g
        for g in genes
        if 0 < reference[reference.gene.eq(g)].label.sum() < len(validation)
    ]
    anchor = _prediction_metrics(reference, genes, eligible)
    anchor_ll = record_scores(legacy, validation, genes)
    candidates = {"legacy": legacy}
    rows = []
    for spec in partial_pool.SPECS:
        artifact = partial_pool.fit(
            train, genes, spec, baseline, legacy, progress=progress
        )
        scores = record_scores(artifact, validation, genes)
        delta = scores - anchor_ll
        standard_error = float(np.std(delta, ddof=1) / np.sqrt(len(delta)))
        point = _prediction_metrics(
            predict_masked(artifact, validation, genes), genes, eligible
        )
        passes = bool(
            eligible
            and delta.mean() + 1e-12 >= standard_error
            and point["conditional_macro_logloss"]
            <= anchor["conditional_macro_logloss"] + 1e-12
            and point["macro_AUROC"] + 1e-12 >= anchor["macro_AUROC"]
            and point["macro_AUPRC"] + 1e-12 >= anchor["macro_AUPRC"]
        )
        rows.append(
            dict(
                candidate=spec["name"],
                inner_joint_NLL=float(-scores.mean()),
                paired_NLL_gain=float(delta.mean()),
                paired_NLL_gain_SE=standard_error,
                free_parameters=artifact["diagnostics"]["free_parameters"],
                gate_pass=passes,
                **point,
            )
        )
        candidates[spec["name"]] = artifact
    passing = [row for row in rows if row["gate_pass"]]
    selected = (
        min(
            passing,
            key=lambda row: (
                row["inner_joint_NLL"],
                row["free_parameters"],
                row["candidate"],
            ),
        )["candidate"]
        if passing
        else "legacy"
    )
    return dict(candidates[selected], selected_candidate=selected), rows
