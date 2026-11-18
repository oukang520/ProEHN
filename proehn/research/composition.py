"""Target-masked clinical gates and patient-weighted three-class scores (Eq. 12)."""

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import jax.numpy as jnp
from proehn.training import prepare_kinetic_frames, fit_kinetic_model
from proehn.kinetic import ProEHNKineticGatekeeper
from proehn.labels import build_progression_label
from .protocols import save_json, patient_digest

KEY = ["patient_id", "compartment", "gene"]
GROUPS = ("pt_genomic", "pt_dynamic", "mt_genomic", "mt_dynamic", "shared")
FLOOR = 1e-12


def three_class_scores(go, genotype, clinical_label, genotype_label):
    """Return full 0--2 Brier and the separately observed log-score components."""
    g, q = np.asarray(go, float), np.asarray(genotype, float)
    c, x = np.asarray(clinical_label), np.asarray(genotype_label)
    if (
        g.ndim != 1
        or any(v.shape != g.shape for v in (q, c, x))
        or not np.isfinite(g).all()
        or not np.isfinite(q).all()
        or np.any((g < 0) | (g > 1) | (q < 0) | (q > 1))
        or not np.isin(c, [0, 1]).all()
        or not np.isin(x, [0, 1]).all()
    ):
        raise ValueError(
            "Aligned finite probabilities and known binary clinical/genotype labels required"
        )
    probability = np.column_stack((g * (1 - q), g * q, 1 - g))
    y = np.where(c == 1, 2, x).astype(int)
    selected = probability[np.arange(len(y)), y]
    gate_ll = -np.log(np.maximum(np.where(c == 0, g, 1 - g), FLOOR))
    genotype_ll = -np.log(np.maximum(np.where(x == 1, q, 1 - q), FLOOR))
    weighted = (c == 0) * genotype_ll
    logloss = -np.log(np.maximum(selected, FLOOR))
    return pd.DataFrame(
        dict(
            p_Go_X0=probability[:, 0],
            p_Go_X1=probability[:, 1],
            p_Stop=probability[:, 2],
            label=y,
            prediction=probability.argmax(axis=1),
            Brier=((probability - np.eye(3)[y]) ** 2).sum(axis=1),
            log_loss=logloss,
            gate_log_loss=gate_ll,
            Go_weighted_genotype_log_loss=weighted,
            genotype_log_loss=genotype_ll,
            log_floor_adjustment=logloss - gate_ll - weighted,
            log_floor_active=selected < FLOOR,
        )
    )


def _gate_go(gate, frame, target_column):
    masked = frame.copy()
    masked[target_column] = np.nan
    x = gate.preprocessor.transform(masked)
    logits = gate.model.apply(
        {"params": gate.params}, *[jnp.asarray(x[k]) for k in GROUPS], train=False
    )
    if gate.calibration is None:
        raise ValueError("Target-specific validation Platt calibration is required")
    return 1 - gate.calibration.transform(np.asarray(logits))


def _check_masked_gate(gate, target_column):
    """Check the retained target input was all-missing when the gate was fitted."""
    group = next(
        (k for k, v in gate.feature_groups.items() if target_column in v), None
    )
    if group is None or gate.preprocessor is None:
        raise ValueError(
            "The target-masked gate must retain its original constant input column"
        )
    transform = gate.preprocessor.transforms[group]
    index = transform.columns.index(target_column)
    kept = np.flatnonzero(transform.keep_)
    local = np.flatnonzero(kept == index)
    if (
        transform.fill_[index] != 0
        or len(local) != 1
        or transform.mean_[local[0]] != 0
        or transform.scale_[local[0]] != 1
    ):
        raise ValueError(
            "A previously fitted unmasked gate cannot substitute for a target-masked gate"
        )


def fit_target_queries(
    clinical,
    genotype_predictions,
    config,
    output,
    *,
    folds=None,
    gate_directory=None,
    primary_model="ProEHN context",
    include_forced_go=True,
):
    """Fit/reuse each masked gate once and supply identical g to all native q models.

    Genotype input is a long table with patient_id, compartment (PT/MT), gene,
    model, label and prediction. It must contain frozen native conditional
    genotype probabilities, retaining the other permitted tissue information.
    """
    cohort = str(config["cohort"]).upper()
    genes = tuple(config["genes"])
    frame = clinical.copy()
    frame["patient_id"] = frame.patient_id.astype(str)
    if frame.patient_id.duplicated().any() or not frame.fold.isin(range(5)).all():
        raise ValueError("One patient per row with integer folds 0..4 is required")
    q = genotype_predictions.copy()
    required = set(KEY + ["model", "label", "prediction"])
    if not required <= set(q):
        raise ValueError(f"Genotype predictions require {sorted(required)}")
    q["patient_id"] = q.patient_id.astype(str)
    if (
        q.duplicated(KEY + ["model"]).any()
        or not q.compartment.isin(["PT", "MT"]).all()
    ):
        raise ValueError(
            "Unique native target/model predictions and PT/MT compartment labels required"
        )
    models = list(q.model.drop_duplicates())
    if primary_model not in models or len(models) < 2:
        raise ValueError(
            "Include the context model and at least one independently fitted genotype comparator"
        )
    if not set(q.gene) <= set(genes):
        raise ValueError("Genotype predictions exceed the frozen cohort vocabulary")
    if cohort == "GENIE":
        expected = {(comp, gene) for comp in ("PT", "MT") for gene in genes}
        for name, part in q.groupby("model"):
            if (
                set(part[["compartment", "gene"]].itertuples(index=False, name=None))
                != expected
            ):
                raise ValueError(
                    "GENIE composition requires all 24 native PT/MT target definitions for every model"
                )
    selected_folds = (
        list(folds)
        if folds is not None
        else ([0] if cohort == "GENIE" else list(range(5)))
    )
    if not selected_folds or not set(selected_folds) <= set(range(5)):
        raise ValueError("Choose outer folds 0..4")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    gate_root = Path(gate_directory) if gate_directory is not None else output / "gates"
    all_queries = []
    eligibility = []
    for fold in selected_folds:
        inner = (fold + 1) % 5
        train0 = frame[~frame.fold.isin([fold, inner])].copy()
        val0 = frame[frame.fold.eq(inner)].copy()
        test0 = frame[frame.fold.eq(fold)].copy()
        train, val, gate_cfg, schema = prepare_kinetic_frames(train0, val0, config)
        test, _, _, _ = prepare_kinetic_frames(test0, test0, config)
        test["Patient_Label"] = build_progression_label(test, schema)
        actual = train.copy()
        for comp, prefix in [("PT", "P"), ("MT", "M")]:
            for gene in genes:
                column = f"{prefix}.{gene} (M)"
                tag = f"{comp}_{gene}"
                key_q = q[
                    q.compartment.eq(comp)
                    & q.gene.eq(gene)
                    & q.patient_id.isin(test.patient_id)
                ].copy()
                aligned = key_q.pivot(
                    index="patient_id", columns="model", values="prediction"
                ).reindex(columns=models)
                if aligned.empty:
                    eligibility.append(
                        dict(fold=fold, compartment=comp, gene=gene, known_patients=0)
                    )
                    continue
                if aligned.isna().any().any():
                    raise ValueError(
                        "All genotype models must cover identical eligible native queries"
                    )
                patients = test.set_index("patient_id").loc[aligned.index].copy()
                known = patients.Patient_Label.isin([0, 1])
                patients = patients[known]
                aligned = aligned.loc[patients.index]
                if not len(patients):
                    continue
                x = pd.to_numeric(patients[column], errors="raise").to_numpy()
                if not np.isin(x, [0, 1]).all():
                    raise ValueError(
                        "Native target calls must be observed binary labels"
                    )
                truth = key_q.pivot(
                    index="patient_id", columns="model", values="label"
                ).loc[patients.index, models]
                if not np.array_equal(
                    truth.to_numpy(), np.repeat(x[:, None], len(models), axis=1)
                ):
                    raise ValueError("Clinical and native genotype labels disagree")
                folder = gate_root / f"fold{fold}"
                folder.mkdir(parents=True, exist_ok=True)
                params, meta = folder / f"{tag}.msgpack", folder / f"{tag}.pkl"
                # Original GENIE target-gate artifacts used a flat directory.
                if (
                    fold == 0
                    and not params.exists()
                    and (gate_root / f"{tag}.msgpack").exists()
                ):
                    params, meta = (
                        gate_root / f"{tag}.msgpack",
                        gate_root / f"{tag}.pkl",
                    )
                binding = dict(
                    cohort=cohort,
                    fold=fold,
                    compartment=comp,
                    gene=gene,
                    training_patients=patient_digest(train),
                    validation_patients=patient_digest(val),
                    kinetic_config=dict(gate_cfg, masked_columns=[column]),
                )
                binding_path = folder / f"{tag}_binding.json"
                if params.exists() or meta.exists():
                    if not (params.exists() and meta.exists()):
                        raise ValueError("Incomplete target-gate artifact pair")
                    if binding_path.exists() and json.loads(
                        binding_path.read_text(encoding="utf8")
                    ) != json.loads(json.dumps(binding)):
                        raise ValueError(
                            "Existing target gate belongs to a different training partition/configuration"
                        )
                    gate = ProEHNKineticGatekeeper(params, meta)
                    declared = gate.scientific_training_protocol
                    text = declared.get(
                        "training_feature_provenance_version", ""
                    ).lower()
                    if (
                        not binding_path.exists()
                        and column not in declared.get("masked_columns", ())
                        and not ("target" in text and "mask" in text)
                    ):
                        raise ValueError(
                            "Existing gate lacks target-masking provenance"
                        )
                else:
                    cfg = dict(gate_cfg, masked_columns=[column])
                    gate = fit_kinetic_model(
                        train,
                        val,
                        patient_id_column="patient_id",
                        model_config=cfg,
                        label_schema=schema,
                    )
                    gate.save(params, meta)
                    save_json(binding_path, binding)
                _check_masked_gate(gate, column)
                go = _gate_go(gate, patients.reset_index(), column)
                prevalence = float(pd.to_numeric(actual[column], errors="raise").mean())
                variants = [
                    (name, aligned[name].to_numpy(float), go, True) for name in models
                ]
                variants.append(
                    ("Training prevalence", np.full(len(go), prevalence), go, True)
                )
                if include_forced_go:
                    variants += [
                        (
                            name + "; Go=1",
                            aligned[name].to_numpy(float),
                            np.ones(len(go)),
                            False,
                        )
                        for name in models
                    ]
                for name, prob, g, shared in variants:
                    values = three_class_scores(
                        g, prob, patients.Patient_Label.to_numpy(), x
                    )
                    values.insert(0, "patient_id", patients.index.to_numpy())
                    values = values.assign(
                        fold=fold,
                        compartment=comp,
                        gene=gene,
                        model=name,
                        Patient_Label=patients.Patient_Label.to_numpy(),
                        X=x,
                        p_Go=g,
                        q=prob,
                        shared_gate=shared,
                    )
                    all_queries.append(values)
                eligibility.append(
                    dict(
                        fold=fold,
                        compartment=comp,
                        gene=gene,
                        known_patients=len(patients),
                        Go=int(patients.Patient_Label.eq(0).sum()),
                        Stop=int(patients.Patient_Label.eq(1).sum()),
                        target_gate_params_sha256=hashlib.sha256(
                            params.read_bytes()
                        ).hexdigest(),
                        target_gate_metadata_sha256=hashlib.sha256(
                            meta.read_bytes()
                        ).hexdigest(),
                    )
                )
    if not all_queries:
        raise ValueError("No eligible known-label native queries")
    return pd.concat(all_queries, ignore_index=True), pd.DataFrame(eligibility)


def summarize_composition(
    queries,
    *,
    primary_model="ProEHN context",
    brier_replicates=2000,
    decomposition_replicates=1000,
    brier_seed=20261113,
    decomposition_seed=20261114,
):
    """Patient-equal means with paired patient resampling shared across models."""
    if queries.duplicated(KEY + ["model"]).any():
        raise ValueError("Duplicate target/model query")
    metrics = [
        "Brier",
        "log_loss",
        "gate_log_loss",
        "Go_weighted_genotype_log_loss",
        "log_floor_adjustment",
    ]
    per = queries.groupby(["model", "patient_id"])[metrics].mean()
    models = list(queries.model.drop_duplicates())
    patients = sorted(queries.patient_id.unique())
    if primary_model not in models:
        raise ValueError("Missing primary model")
    tables = {m: per.loc[m].reindex(patients) for m in models}
    if any(t.isna().any().any() for t in tables.values()):
        raise ValueError("Paired summaries require the same patients")
    shared = queries[queries.shared_gate]
    for metric in ["p_Go", "gate_log_loss"]:
        if shared.groupby(KEY)[metric].agg(lambda x: np.ptp(x)).max() > 1e-12:
            raise ValueError(
                "Genotype comparisons must share the identical target gate"
            )
    stop = shared[shared.Patient_Label.eq(1)]
    if len(stop) and stop.groupby(KEY).log_loss.agg(lambda x: np.ptp(x)).max() > 1e-12:
        raise ValueError(
            "Stop queries must have zero paired shared-gate log-loss contribution"
        )

    def counts(seed, n):
        if int(n) != n or n < 2:
            raise ValueError("At least two patient-bootstrap replicates are required")
        rng = np.random.default_rng(seed)
        return np.asarray(
            [
                np.bincount(
                    rng.integers(0, len(patients), len(patients)),
                    minlength=len(patients),
                )
                for _ in range(n)
            ],
            float,
        )

    br = counts(brier_seed, brier_replicates)
    ll = counts(decomposition_seed, decomposition_replicates)

    def estimate(values, draws):
        values = np.asarray(values, float)
        den = draws.sum(axis=1)
        valid = den > 0
        boot = (draws[valid] @ values) / den[valid]
        lo, hi = np.quantile(boot, [0.025, 0.975])
        return dict(
            estimate=float(values.mean()),
            ci_low=float(lo),
            ci_high=float(hi),
            valid_resamples=len(boot),
        )

    scores = []
    contrasts = []
    components = []
    go_scores = []
    known = (
        queries.drop_duplicates("patient_id")
        .set_index("patient_id")
        .reindex(patients)
        .Patient_Label
    )
    go_index = np.flatnonzero(known.to_numpy() == 0)
    if not len(go_index):
        raise ValueError("Go-only genotype analysis requires Go patients")
    go_per = (
        queries[queries.Patient_Label.eq(0)]
        .groupby(["model", "patient_id"])
        .genotype_log_loss.mean()
    )
    for name in models:
        t = tables[name]
        for metric in ["Brier", "log_loss"]:
            draws = br if metric == "Brier" else ll
            scores.append(
                dict(
                    model=name,
                    metric=metric,
                    n_patients=len(patients),
                    **estimate(t[metric], draws),
                )
            )
            if name != primary_model:
                contrasts.append(
                    dict(
                        comparator=name,
                        primary=primary_model,
                        metric=metric,
                        direction="comparator minus primary; positive favors primary",
                        **estimate(t[metric] - tables[primary_model][metric], draws),
                    )
                )
        for metric in metrics[2:]:
            components.append(
                dict(model=name, component=metric, **estimate(t[metric], ll))
            )
        values = go_per.loc[name].reindex(np.asarray(patients)[go_index]).to_numpy()
        go_scores.append(
            dict(
                model=name,
                metric="Go_only_genotype_log_loss",
                n_patients=len(go_index),
                **estimate(values, ll[:, go_index]),
            )
        )
        if name != primary_model:
            reference = (
                go_per.loc[primary_model]
                .reindex(np.asarray(patients)[go_index])
                .to_numpy()
            )
            go_scores.append(
                dict(
                    model=name,
                    metric="Go_only_log_loss_reduction_vs_primary",
                    n_patients=len(go_index),
                    **estimate(values - reference, ll[:, go_index]),
                )
            )
    summary = dict(
        n_patients=len(patients),
        Go=len(go_index),
        Stop=len(patients) - len(go_index),
        target_patient_queries=int(len(queries[queries.model.eq(primary_model)])),
        target_definitions=int(
            queries[["compartment", "gene"]].drop_duplicates().shape[0]
        ),
        bootstrap_unit="patient",
        averaging="targets within patient, then equal patients",
        Brier_scale=[0, 2],
        Brier_replicates=brier_replicates,
        Brier_seed=brier_seed,
        decomposition_replicates=decomposition_replicates,
        decomposition_seed=decomposition_seed,
        shared_gate_Stop_logloss_difference=0.0,
        log_floor=FLOOR,
    )
    return (
        dict(
            metrics=pd.DataFrame(scores),
            paired_comparisons=pd.DataFrame(contrasts),
            logscore_decomposition=pd.DataFrame(components),
            Go_only_genotype_scores=pd.DataFrame(go_scores),
            patient_scores=per.reset_index(),
            per_target=queries.groupby(["model", "compartment", "gene"])[metrics]
            .mean()
            .reset_index(),
        ),
        summary,
    )
