"""Run the Supplement S4.1 shared eight-method patient-fold comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from proehn.comparators import (
    MODELS,
    conditional_probabilities,
    dependency_status,
    fit_hypertraps,
    fit_hypertraps_ct,
    fit_mhn,
    fit_official_metmhn,
    fit_oncotree,
    predict_official_metmhn,
    prediction_matrix,
    published_mcc_cutoffs,
    specificity_cutoffs,
    summarize_oof,
)
from proehn.research.decisions import select_policy
from proehn.research.native import fit_native, predict_masked, select_snapshot_candidate
from proehn.research.protocols import (
    GENES,
    partition,
    patient_digest,
    save_json,
    snapshot_frame,
)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, help="Prepared patient CSV with frozen fold 0..4"
    )
    parser.add_argument("--cohort", choices=GENES)
    parser.add_argument(
        "--output",
        type=Path,
        help="New results directory outside the source repository",
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument(
        "--fold",
        type=int,
        choices=range(5),
        help="Run one outer fold; default all five",
    )
    parser.add_argument("--rscript", help="Rscript executable path")
    parser.add_argument("--check-dependencies", action="store_true")
    return parser


def _outside_repository(path):
    path = path.resolve()
    repository = Path(__file__).resolve().parents[1]
    if path == repository or repository in path.parents:
        raise ValueError("--output must be outside the source repository")
    return path


def _frame_signature(frame):
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    digest.update(json.dumps(list(frame.columns)).encode())
    digest.update(json.dumps([str(value) for value in frame.dtypes]).encode())
    return digest.hexdigest()


def _cached_fit(path, build, train, *, context):
    digest = patient_digest(train)
    binding = dict(
        context, model_key=path.stem, training_frame_sha256=_frame_signature(train)
    )
    signature = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    if path.exists():
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if "fit_signature" not in artifact:
            raise ValueError(
                "Unsigned historical fit cannot be reused automatically; "
                "choose a new output directory: " + str(path)
            )
        if artifact["fit_signature"] != signature:
            raise ValueError(
                "Cached fit inputs, protocol or implementation changed: " + str(path)
            )
        payload = {
            key: value for key, value in artifact.items() if key != "fit_payload_sha256"
        }
        payload_digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        if artifact.get("fit_payload_sha256") != payload_digest:
            raise ValueError("Cached fit payload integrity check failed: " + str(path))
        if artifact.get("training_patient_sha256") != digest:
            raise ValueError("Cached model training membership mismatch: " + str(path))
        return artifact
    started = time.time()
    artifact = build()
    artifact.update(
        training_patient_sha256=digest,
        fit_signature=signature,
        fit_binding=binding,
        n_train=len(train),
        elapsed_seconds=time.time() - started,
    )
    artifact["fit_payload_sha256"] = hashlib.sha256(
        json.dumps(artifact, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    save_json(path, artifact)
    return artifact


def _predict(model, artifact, subset, genes):
    if model in ("proehn", "no_covariate"):
        return prediction_matrix(
            predict_masked(artifact, subset, genes), subset.patient_id, genes
        )
    if model == "metmhn":
        return predict_official_metmhn(artifact, subset, genes)
    if model == "prevalence":
        return np.broadcast_to(
            artifact["training_prior"], (len(subset), len(genes))
        ).copy()
    values = subset[["P." + gene for gene in genes]].to_numpy(np.int32)
    return conditional_probabilities(
        artifact["probability"],
        values,
        reverse=artifact.get("reverse_state_bits", False),
    )


def _long_predictions(
    subset, genes, p, model, fold, partition_name, *, thresholds=None
):
    if p.shape != (len(subset), len(genes)) or not np.isfinite(p).all():
        raise FloatingPointError("Incomplete or nonfinite model predictions")
    if np.any((p < 0) | (p > 1)):
        raise FloatingPointError("Model probability outside [0,1]")
    result = []
    for j, gene in enumerate(genes):
        block = pd.DataFrame(
            dict(
                patient_id=subset.patient_id.to_numpy(),
                gene=gene,
                label=subset["P." + gene].to_numpy(np.int32),
                prediction=p[:, j],
                model=MODELS[model],
                fold=fold,
                partition=partition_name,
            )
        )
        if thresholds is not None:
            tm, ts = thresholds
            block["hard_mcc"] = (p[:, j] >= tm[j]).astype(np.int8)
            block["hard_spec90"] = (p[:, j] >= ts[j]).astype(np.int8)
        result.append(block)
    return pd.concat(result, ignore_index=True)


def _run_fold(frame, genes, cohort, fold, models, output, rscript, run_signature):
    train, validation, test = partition(frame, fold)
    if not all(len(part) for part in (train, validation, test)):
        raise ValueError(
            "Every training, validation and test partition must be nonempty"
        )
    xtrain = train[["P." + gene for gene in genes]].to_numpy(np.int32)
    xvalidation = validation[["P." + gene for gene in genes]].to_numpy(np.int32)
    prior = (xtrain.sum(axis=0) + 0.5) / (len(train) + 1.0)
    root = output / f"fold{fold}"
    root.mkdir(parents=True, exist_ok=True)
    context = dict(
        run_signature=run_signature,
        cohort=cohort,
        genes=list(genes),
        outer_fold=fold,
        validation_frame_sha256=_frame_signature(validation),
    )
    baseline = None
    if set(models) & {"proehn", "no_covariate", "metmhn"}:
        baseline = _cached_fit(
            root / "no_covariate_fit.json",
            lambda: fit_native(train, genes, cohort, constant=True),
            train,
            context=context,
        )
    hyper = None
    if set(models) & {"hypertraps", "hypertraps_ct"}:
        hyper = _cached_fit(
            root / "hypertraps_fit.json",
            lambda: fit_hypertraps(xtrain),
            train,
            context=context,
        )
    fitted = {}
    if "no_covariate" in models:
        fitted["no_covariate"] = baseline
    if "hypertraps" in models:
        fitted["hypertraps"] = hyper

    def fit_context():
        legacy = _cached_fit(
            root / "legacy_context_fit.json",
            lambda: fit_native(train, genes, cohort),
            train,
            context=context,
        )
        selected, candidates = select_snapshot_candidate(
            train, validation, genes, cohort, baseline, legacy
        )
        save_json(
            root / "generative_candidate_selection.json",
            dict(
                selected_candidate=selected["selected_candidate"],
                candidates=candidates,
                validation_patient_sha256=patient_digest(validation),
                test_used=False,
            ),
        )
        return selected

    builders = {
        "proehn": fit_context,
        "metmhn": lambda: fit_official_metmhn(train, genes, baseline, cohort),
        "mhn": lambda: fit_mhn(xtrain),
        "oncotree": lambda: fit_oncotree(
            xtrain, genes, root / "oncotree_native", rscript=rscript
        ),
        "hypertraps_ct": lambda: fit_hypertraps_ct(
            xtrain,
            xvalidation,
            genes,
            hyper,
            root / "hypertraps_ct_native",
            seed=20261113 + 100 * list(GENES).index(cohort) + 10 * fold,
            rscript=rscript,
        ),
        "prevalence": lambda: dict(status="COMPLETE", training_prior=prior.tolist()),
    }
    predictions = []
    for model in models:
        print(f"Running {cohort} fold {fold}: {MODELS[model]}", flush=True)
        directory = root / model
        directory.mkdir(exist_ok=True)
        if model not in fitted:
            fitted[model] = _cached_fit(
                root / f"{model}_fit.json",
                builders[model],
                train,
                context=context,
            )
        artifact = fitted[model]
        pv = _predict(model, artifact, validation, genes)
        selection = select_policy(xvalidation, pv, prior, validation.patient_id)
        tm = (
            published_mcc_cutoffs(xvalidation, pv)
            if selection["policy_index"] == 0
            else np.asarray(selection["thresholds"])
        )
        ts = specificity_cutoffs(xvalidation, pv)
        selection["candidate_metrics"].to_csv(
            directory / "decision_candidates.csv", index=False, float_format="%.17g"
        )
        np.savez_compressed(
            directory / "validation_decisions.npz",
            ids=validation.patient_id.to_numpy(str),
            genes=np.asarray(genes),
            labels=xvalidation,
            probabilities=pv,
            inner_crossfit_decisions=selection["inner_crossfit_decisions"],
        )
        save_json(
            directory / "selection.json",
            dict(
                selected_policy=selection["policy"],
                mcc_thresholds=tm.tolist(),
                spec90_thresholds=ts.tolist(),
                training_prior=prior.tolist(),
                n_train=len(train),
                n_validation=len(validation),
                n_test=len(test),
                validation_patient_sha256=patient_digest(validation),
                training_patient_sha256=patient_digest(train),
                test_used_for_selection=False,
                specificity_single_class_fallback=[
                    gene
                    for j, gene in enumerate(genes)
                    if len(np.unique(xvalidation[:, j])) < 2
                ],
            ),
        )
        # Test predictions are computed only after the decision rules are saved.
        pt = _predict(model, artifact, test, genes)
        for subset, p, name in [(validation, pv, "validation"), (test, pt, "test")]:
            table = _long_predictions(
                subset, genes, p, model, fold, name, thresholds=(tm, ts)
            )
            table.to_csv(
                directory / f"{name}_predictions.csv", index=False, float_format="%.17g"
            )
            if name == "test":
                predictions.append(table)
    return pd.concat(predictions, ignore_index=True)


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if len(set(args.models)) != len(args.models):
        parser.error("--models cannot contain duplicate methods")
    dependencies = dependency_status(args.models, rscript=args.rscript)
    unavailable = {
        key: row for key, row in dependencies.items() if not row["available"]
    }
    if args.check_dependencies:
        print(json.dumps(dependencies, indent=2))
        return 1 if unavailable else 0
    if unavailable:
        parser.error("Comparator dependencies unavailable: " + json.dumps(unavailable))
    if args.input is None or args.output is None or args.cohort is None:
        parser.error("--input, --cohort and --output are required for training")
    output = _outside_repository(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(
        args.input, dtype={"patient_id": str}, float_precision="round_trip"
    )
    if "fold" not in frame or not frame.fold.isin(range(5)).all():
        raise ValueError(
            "Input must retain the frozen patient fold column with integer values 0..4"
        )
    if set(frame.fold.unique()) != set(range(5)):
        raise ValueError("All five frozen patient folds must be present in the input")
    genes = list(GENES[args.cohort])
    frame, _ = snapshot_frame(frame, genes)
    frame = frame.reset_index(drop=True)
    source_root = Path(__file__).resolve().parents[1]
    source_paths = [
        Path(__file__).resolve(),
        *sorted((source_root / "proehn" / "research").glob("*.py")),
        *sorted((source_root / "proehn").glob("comparator*")),
        source_root / "proehn" / "training.py",
        source_root / "proehn" / "topology.py",
        *sorted((source_root / "proehn" / "ctmc").glob("*.py")),
    ]
    implementations = {
        str(path.relative_to(source_root)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in source_paths
        if path.is_file()
    }
    manifest = dict(
        protocol="supplement-S4.1-eight-method-snapshot-v1",
        cohort=args.cohort,
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        genes=genes,
        implementation_sha256=implementations,
        models=args.models,
        dependencies=dependencies,
        patient_sha256=patient_digest(frame),
        folds="test=f, validation=(f+1)%5, other three folds train",
        projection="primary if observed, otherwise metastasis; primary-only adapter",
        mcc_decision="shared seven-rule patient-hash cross-fit; one-jackknife-SE utility",
        specificity="separate validation sensitivity >=90%; single-class fallback 0.5",
        uncertainty="1000 common-patient percentile draws; conditional on fitted models and selected rules",
    )
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old != manifest:
            raise ValueError(
                "Existing output manifest differs; choose a new --output directory"
            )
    else:
        save_json(manifest_path, manifest)
    run_signature = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    folds = [args.fold] if args.fold is not None else list(range(5))
    predictions = pd.concat(
        [
            _run_fold(
                frame,
                genes,
                args.cohort,
                fold,
                args.models,
                output,
                args.rscript,
                run_signature,
            )
            for fold in folds
        ],
        ignore_index=True,
    )
    predictions.to_csv(
        output / "pooled_predictions.csv", index=False, float_format="%.17g"
    )
    if args.fold is None:
        summarize_oof(frame, genes, predictions, output)
    save_json(
        output / "completion.json",
        dict(
            status="COMPLETE",
            outer_folds=folds,
            methods=args.models,
            out_of_fold_metrics=args.fold is None,
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
