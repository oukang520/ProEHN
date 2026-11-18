"""Training entry points for the ProEHN core models."""

from __future__ import annotations

from functools import partial, lru_cache
import json
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .preprocessing import (
    build_topology_training_data,
    calculate_marginal_rates,
    TopologyTrainingPreprocessor,
)
from .topology import ProEHNTopologyModel


@lru_cache(maxsize=32)
def _compiled_topology_functions(
    n_events, n_features, strength, l1_ratio, clip_min, clip_max, l2_floor
):
    """Reuse compiled likelihoods across independently initialized experiments."""
    model = ProEHNTopologyModel(
        n_events, n_features, strength, l1_ratio, clip_min, clip_max, l2_floor
    )

    @partial(jax.jit, static_argnums=(1, 2, 3))
    def loss_gradient(params, kind, n_primary, n_metastatic, genes, features, weights):
        return jax.value_and_grad(model.bucket_loss)(
            params, kind, n_primary, n_metastatic, genes, features, weights
        )

    @jax.jit
    def penalty_gradient(params):
        return jax.value_and_grad(model.regularization)(params)

    return loss_gradient, penalty_gradient


def fit_topology_model(
    buckets: list[tuple[int, int, int, np.ndarray, np.ndarray]],
    n_events: int,
    n_features: int,
    n_samples: int,
    regularization_strength: float = 0.002,
    l1_ratio: float = 0.0,
    log_rate_clip_min: float = -5.0,
    log_rate_clip_max: float = 5.0,
    optimizer_maxiter: int = 1200,
    random_seed: int = 11,
    l2_floor: float = 0.0,
    bucket_sample_weights=None,
    free_parameter_mask=None,
    optimizer_bound: float | None = 4.0,
    progress_callback=None,
    initial_parameters=None,
    optimizer_ftol: float = 1e-8,
    optimizer_gtol: float = 1e-5,
) -> tuple[ProEHNTopologyModel, np.ndarray, float]:
    """Fit the feature-modulated CTMC topology engine with L-BFGS-B."""

    if not jax.config.x64_enabled:
        raise RuntimeError(
            "Enable JAX_ENABLE_X64=1 before topology fitting; float64 likelihoods are required"
        )
    weights = (
        [np.ones(len(b[3])) for b in buckets]
        if bucket_sample_weights is None
        else [np.asarray(w, dtype=np.float64) for w in bucket_sample_weights]
    )
    if (
        not buckets
        or len(weights) != len(buckets)
        or any(
            w.shape != (len(b[3]),) or not np.isfinite(w).all() or np.any(w < 0)
            for b, w in zip(buckets, weights)
        )
        or n_samples <= 0
        or not np.isclose(n_samples, sum(w.sum() for w in weights), rtol=0, atol=1e-8)
    ):
        raise ValueError("Likelihood normalization must equal included observations")
    if not np.isfinite(optimizer_ftol) or optimizer_ftol <= 0:
        raise ValueError("Optimizer tolerance must be finite and positive")
    if optimizer_bound is not None and (
        not np.isfinite(optimizer_bound) or optimizer_bound <= 0
    ):
        raise ValueError("Optimizer bound must be finite and positive")
    model = ProEHNTopologyModel(
        n_events=n_events,
        n_features=n_features,
        regularization_strength=regularization_strength,
        l1_ratio=l1_ratio,
        l2_floor=l2_floor,
        log_rate_clip_min=log_rate_clip_min,
        log_rate_clip_max=log_rate_clip_max,
    )
    base_rates = calculate_marginal_rates(buckets, n_events, weights)
    init_params = jnp.array(
        model.init_params(jax.random.PRNGKey(random_seed), base_rates=base_rates),
        dtype=jnp.float64,
    )
    mask = (
        np.ones(model.shapes.total_size, dtype=bool)
        if free_parameter_mask is None
        else np.asarray(free_parameter_mask)
    )
    if mask.shape != (model.shapes.total_size,) or mask.dtype != bool or not mask.any():
        raise ValueError(
            "A nonempty boolean free-parameter mask of the full parameter shape is required"
        )
    if initial_parameters is not None:
        initial = np.asarray(initial_parameters, dtype=np.float64)
        if initial.shape != init_params.shape or not np.isfinite(initial).all():
            raise ValueError(
                "Initial parameters must be finite and match the complete topology shape"
            )
        init_params = initial
    # Fixed parameters are explicitly zero, useful for prespecified mechanism ablations.
    init_params = np.where(mask, np.asarray(init_params), 0.0)
    if optimizer_bound is not None:
        init_params = np.clip(init_params, -optimizer_bound, optimizer_bound)
    buckets_jax = [
        (
            bucket_type,
            n_primary,
            n_metastatic,
            jax.device_put(jnp.array(bucket_genotypes, dtype=jnp.int8)),
            jax.device_put(jnp.array(bucket_features, dtype=jnp.float64)),
            jax.device_put(jnp.asarray(weight, dtype=jnp.float64)),
        )
        for (
            bucket_type,
            n_primary,
            n_metastatic,
            bucket_genotypes,
            bucket_features,
        ), weight in zip(buckets, weights)
    ]
    bucket_loss_and_grad, regularization_and_grad = _compiled_topology_functions(
        n_events,
        n_features,
        regularization_strength,
        l1_ratio,
        log_rate_clip_min,
        log_rate_clip_max,
        l2_floor,
    )

    def objective(params_np: np.ndarray) -> tuple[float, np.ndarray]:
        expanded = np.zeros(model.shapes.total_size, dtype=np.float64)
        expanded[mask] = params_np
        params = jnp.array(expanded, dtype=jnp.float64)
        total_loss = 0.0
        total_grad = np.zeros(model.shapes.total_size, dtype=np.float64)
        for bucket in buckets_jax:
            loss, grad = bucket_loss_and_grad(params, *bucket)
            # Bound peak memory to one bucket on CPU; do not queue many large
            # paired-state reverse passes concurrently on a small workstation.
            loss_value, grad_value = jax.device_get((loss, grad))
            total_loss += float(loss_value)
            total_grad += np.asarray(grad_value)
        avg_loss = total_loss / max(1, n_samples)
        avg_grad = total_grad / max(1, n_samples)
        reg_loss, reg_grad = regularization_and_grad(params)
        value = float(avg_loss + reg_loss)
        gradient = np.asarray(avg_grad + reg_grad, dtype=np.float64)[mask]
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise FloatingPointError(
                "Non-finite likelihood/gradient; optimization aborted"
            )
        return value, gradient

    result = minimize(
        objective,
        np.asarray(init_params, dtype=np.float64)[mask],
        method="L-BFGS-B",
        jac=True,
        callback=progress_callback,
        bounds=(
            None
            if optimizer_bound is None
            else [(-optimizer_bound, optimizer_bound)] * int(mask.sum())
        ),
        options={
            "maxiter": int(optimizer_maxiter),
            "ftol": float(optimizer_ftol),
            "gtol": float(optimizer_gtol),
        },
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Topology optimizer did not converge: {result.message}")
    fitted = np.zeros(model.shapes.total_size, dtype=np.float64)
    fitted[mask] = result.x
    model.fit_diagnostics = dict(
        converged=bool(result.success),
        iterations=int(result.nit),
        objective=float(result.fun),
        message=str(result.message),
        free_parameters=int(mask.sum()),
        weighted_observations=float(n_samples),
        compressed_records=sum(len(b[3]) for b in buckets),
        optimizer_bound=optimizer_bound,
        optimizer_ftol=optimizer_ftol,
        optimizer_gtol=optimizer_gtol,
        optimizer_maxiter=int(optimizer_maxiter),
        function_evaluations=int(result.nfev),
        cumulative_iterations=None,
        parameters_at_bound=(
            0
            if optimizer_bound is None
            else int(np.sum(np.abs(result.x) >= optimizer_bound - 1e-6))
        ),
    )
    return model, fitted, float(result.fun)


def train_topology_from_csv(
    csv_path: str | Path,
    output_path: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit a training partition using the published cohort architecture.

    The cohort's twelve genes are supplied in fixed order, never reselected in a
    fold. ``training_folds`` optionally selects raw patients before transforms.
    The full snapshot inner-selection workflow is in ``train_published.py``.
    """
    from .features import EvolutionCovariatePreprocessor

    df = pd.read_csv(csv_path)
    data_cfg = dict(config.get("data", {}))
    df.attrs["feature_metadata"] = data_cfg.get("feature_metadata", {})
    topo_cfg = dict(config.get("topology", {}))
    cohort = str(config.get("cohort", "")).upper()
    if cohort not in ("GENIE", "MELA", "PACA"):
        raise ValueError("Specify the published cohort: GENIE, MELA or PACA")
    fold_column = data_cfg.get("fold_column", "fold")
    training_folds = config.get(
        "training_folds", config.get("folds", {}).get("native_fit")
    )
    if training_folds is not None and fold_column in df:
        df = df[df[fold_column].isin(training_folds)].copy()
    patient_column = data_cfg.get("patient_id_column", "patient_id")
    if (
        patient_column not in df
        or df[patient_column].isna().any()
        or df[patient_column].duplicated().any()
    ):
        raise ValueError("One complete selected record per known patient is required")
    genes = tuple(topo_cfg.get("gene_names") or config.get("genes", ()))
    if len(genes) != 12 or len(set(genes)) != 12:
        raise ValueError(
            "Supply the frozen twelve-gene cohort order in topology.gene_names"
        )
    # Analysis-ready research tables use P.GENE; legacy tables append " (M)".
    suffix = data_cfg.get("mutation_suffix", " (M)")
    if not all(f"P.{g}{suffix}" in df and f"M.{g}{suffix}" in df for g in genes):
        if all(f"P.{g}" in df and f"M.{g}" in df for g in genes):
            suffix = ""
        else:
            raise ValueError("The fixed primary/metastatic gene columns are missing")
    no_covariates = bool(topo_cfg.get("no_covariates", False))
    architecture = topo_cfg.get("architecture", "auto")
    if architecture == "auto":
        architecture = "diagonal" if cohort in ("MELA", "PACA") else "full"
    if cohort in ("MELA", "PACA"):
        if architecture != "diagonal":
            raise ValueError(
                "Native ICGC uses the accepted 273-parameter diagonal-context model"
            )
        from .research.native import fit_native

        raw = df.rename(
            columns={f"{p}.{g}{suffix}": f"{p}.{g}" for p in ("P", "M") for g in genes}
        )
        fit = fit_native(raw, list(genes), cohort, constant=no_covariates)
        params = np.asarray(fit["parameters"], float)
        covariates = EvolutionCovariatePreprocessor.from_historical(
            cohort, fit["preprocessing"], no_covariates=no_covariates
        )
        model = ProEHNTopologyModel(
            12, len(covariates.feature_names), regularization_strength=0.0
        )
        model.fit_diagnostics = fit["diagnostics"]
        final_loss = float(fit["diagnostics"]["objective"])
        n_events, n_features, n_samples = 12, len(covariates.feature_names), len(df)
        gene_names, feature_names = list(genes), covariates.feature_names
        training_protocol = dict(
            cohort=cohort,
            architecture=architecture,
            spec=fit["spec"],
            fit_diagnostics=model.fit_diagnostics,
        )
    else:
        if architecture != "full":
            raise ValueError("GENIE uses the full context-layer architecture")
        model, params, final_loss, covariates, summary = _fit_full_topology_csv(
            df, genes, suffix, topo_cfg, data_cfg, cohort, config
        )
        n_events, n_features, n_samples, gene_names, feature_names = summary
        training_protocol = dict(
            cohort=cohort,
            architecture=architecture,
            fit_diagnostics=model.fit_diagnostics,
        )
    training_protocol["training_folds"] = (
        list(training_folds)
        if training_folds is not None and fold_column in df
        else None
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from .artifacts import ScientificArtifactMetadata, save_topology_artifact

    metadata = ScientificArtifactMetadata(
        tuple(gene_names),
        tuple(feature_names),
        tuple(covariates.columns),
        covariates.metadata(),
        model.log_rate_clip_min,
        model.log_rate_clip_max,
        {
            k: getattr(model, k)
            for k in ("regularization_strength", "l1_ratio", "l2_floor")
        },
        data_cfg.get("target_event_schema_version", f"{cohort}-fixed12-v1"),
        data_cfg.get(
            "training_feature_provenance_version",
            "background-all-modeled-genes-excluded-v1",
        ),
        training_protocol=training_protocol,
    )
    save_topology_artifact(output_path, params, metadata)
    return dict(
        output_path=str(output_path),
        final_loss=final_loss,
        n_events=n_events,
        n_features=n_features,
        n_samples=n_samples,
        gene_names=gene_names,
        feature_names=feature_names,
        fit_diagnostics=model.fit_diagnostics,
    )


def _fit_full_topology_csv(df, genes, suffix, topo_cfg, data_cfg, cohort, config):
    preprocessor = TopologyTrainingPreprocessor(
        12,
        primary_prefix=data_cfg.get("primary_prefix", "P."),
        metastasis_prefix=data_cfg.get("metastasis_prefix", "M."),
        mutation_suffix=suffix,
        gene_names=genes,
        cohort=cohort,
        no_covariates=bool(topo_cfg.get("no_covariates", False)),
    ).fit(df)
    buckets, n_events, n_features, n_samples, gene_names, feature_names = (
        build_topology_training_data(
            df,
            preprocessor=preprocessor,
            top_n_genes=int(topo_cfg.get("top_n_genes", 12)),
            max_batch_size=int(topo_cfg.get("max_batch_size", 1)),
            max_active_events=int(topo_cfg.get("max_active_events", 25)),
            core_drivers=topo_cfg.get("core_drivers", ()),
            use_bio_pure_features=bool(topo_cfg.get("use_bio_pure_features", False)),
            seeding_column=data_cfg.get("seeding_column", "Seeding"),
            type_column=data_cfg.get("type_column", "type"),
            diagnosis_order_column=data_cfg.get("diagnosis_order_column", "diag_order"),
            primary_prefix=data_cfg.get("primary_prefix", "P."),
            metastasis_prefix=data_cfg.get("metastasis_prefix", "M."),
            mutation_suffix=suffix,
        )
    )
    model, params, final_loss = fit_topology_model(
        buckets,
        n_events,
        n_features,
        n_samples,
        regularization_strength=float(topo_cfg.get("regularization_strength", 0.002)),
        l1_ratio=float(topo_cfg.get("l1_ratio", 0.0)),
        l2_floor=float(topo_cfg.get("l2_floor", 0.0)),
        log_rate_clip_min=float(topo_cfg.get("log_rate_clip_min", -5.0)),
        log_rate_clip_max=float(topo_cfg.get("log_rate_clip_max", 5.0)),
        optimizer_maxiter=int(topo_cfg.get("optimizer_maxiter", 1200)),
        optimizer_bound=float(topo_cfg.get("optimizer_bound", 4.0)),
        optimizer_ftol=float(topo_cfg.get("optimizer_ftol", 1e-8)),
        optimizer_gtol=float(topo_cfg.get("optimizer_gtol", 1e-5)),
        random_seed=int(topo_cfg.get("random_seed", config.get("random_seed", 11))),
    )
    return (
        model,
        params,
        final_loss,
        preprocessor.covariates,
        (n_events, n_features, n_samples, gene_names, feature_names),
    )


def fit_kinetic_model(
    training_df,
    validation_df,
    *,
    patient_id_column,
    label_column="Patient_Label",
    model_config=None,
    label_schema=None,
):
    """Train formal gatekeeper after splitting; early stopping sees inner val only.

    This entry point has no test-data argument and writes no artifacts. It uses
    the operational Stop=1 convention. Focal/weighted outputs require independent
    calibration before an absolute-probability interpretation can be justified.
    """
    from .preprocessing import KineticPreprocessor, build_kinetic_matrices
    from .kinetic import ProEHNKineticGatekeeper, create_train_state, kinetic_loss_fn
    from .probability import KineticProbabilityProtocol, PlattCalibration
    from dataclasses import asdict
    from .config import KineticConfig

    config = dict(asdict(KineticConfig()), **(model_config or {}))
    if config.get("loss") == "bce" and float(config.get("focal_gamma", 0.0)) != 0:
        raise ValueError("BCE requires focal_gamma=0")
    if config.get("loss", "bce") not in ("bce", "focal"):
        raise ValueError("Unknown kinetic loss")
    probability_protocol = KineticProbabilityProtocol(
        float(config.get("focal_gamma", 0.0)), config.get("calibration", "platt")
    )
    from .labels import (
        require_frozen_label_protocol,
        precomputed_progression_labels,
        build_progression_label,
    )

    if label_schema is None:
        raise ValueError("Formal kinetic training requires a frozen label protocol")
    require_frozen_label_protocol(label_schema)
    partitions = []
    for frame in (training_df, validation_df):
        frame = frame.copy()
        if (
            label_column == "Patient_Label"
            and label_schema.pfs_column in frame
            and (
                label_schema.cohort in ("MELA", "PACA")
                or label_schema.event_column in frame
            )
        ):
            labels = build_progression_label(frame, label_schema)
            if label_column in frame and not np.array_equal(
                frame[label_column].fillna(-1).to_numpy(), labels
            ):
                raise ValueError(
                    "Cached progression labels conflict with raw clinical duration/event fields"
                )
            frame[label_column] = labels
        else:
            precomputed_progression_labels(frame, label_column, label_schema)
        partitions.append(frame)
    training_df, validation_df = partitions
    train_ids, val_ids = (
        training_df[patient_id_column],
        validation_df[patient_id_column],
    )
    if train_ids.isna().any() or val_ids.isna().any() or set(train_ids) & set(val_ids):
        raise ValueError(
            "Known, disjoint training and inner-validation patients required"
        )
    if "covariates" not in config:
        raise ValueError(
            "Formal kinetic training requires explicitly frozen covariates"
        )
    from .features import FrozenCovariateProtocol

    FrozenCovariateProtocol(
        label_schema.cohort,
        "explicit-training-covariates",
        (),
        tuple(config["covariates"]),
        True,
    ).validate(training_df).validate(validation_df)
    prep = KineticPreprocessor(
        config["covariates"],
        masked_columns=config.get("masked_columns", ()),
        gene_names=config.get("gene_names"),
    ).fit(training_df)
    xt, yt, _ = build_kinetic_matrices(training_df, label_column, preprocessor=prep)
    xv, yv, _ = build_kinetic_matrices(validation_df, label_column, preprocessor=prep)
    if not np.isin(yt, [0, 1]).any() or not np.isin(yv, [0, 1]).any():
        raise ValueError(
            "Training and validation each require observed progression labels"
        )
    key = jax.random.PRNGKey(int(config.get("random_seed", 11)))
    state = create_train_state(key, config, {k: v.shape[1] for k, v in xt.items()})
    train_batch = dict(x={k: jnp.asarray(v) for k, v in xt.items()}, y=jnp.asarray(yt))
    val_batch = dict(x={k: jnp.asarray(v) for k, v in xv.items()}, y=jnp.asarray(yv))
    # BCE by default: a proper scoring rule; no outcome-dependent class weights.
    gamma = float(config.get("focal_gamma", 0.0))
    best_loss, best_params, stale, best_epoch = np.inf, None, 0, None
    history = []
    for epoch in range(int(config.get("num_epochs", 150))):
        key, step_key = jax.random.split(key)
        loss, grad = jax.value_and_grad(kinetic_loss_fn)(
            state.params, state, train_batch, step_key, 1.0, gamma
        )
        if not np.isfinite(float(loss)):
            raise FloatingPointError("Non-finite kinetic training loss")
        state = state.apply_gradients(grads=grad)
        metrics, _ = kinetic_loss_fn(
            state.params, state, val_batch, step_key, 1.0, gamma, train=False
        )
        val_loss = float(metrics["loss"])
        if not np.isfinite(val_loss):
            raise FloatingPointError("Non-finite kinetic validation loss")
        if val_loss < best_loss:
            best_loss, best_params, stale = val_loss, state.params, 0
            best_epoch = epoch + 1
        else:
            stale += 1
        history.append(
            dict(epoch=epoch + 1, training_bce=float(loss), validation_bce=val_loss)
        )
        if config.get("early_stopping", False) and stale >= int(
            config.get("patience", 20)
        ):
            break
    if best_params is None:
        raise ValueError("num_epochs must be positive")
    from .kinetic import KineticGatekeeperNetwork

    gate = ProEHNKineticGatekeeper(None, None)
    gate.model = KineticGatekeeperNetwork(
        int(config.get("d_model", 32)),
        int(config.get("n_head_layers", 1)),
        float(config.get("dropout_rate", 0.2)),
    )
    gate.params, gate.preprocessor, gate.feature_groups = (
        best_params,
        prep,
        prep.feature_groups,
    )
    gate.model_config.update(config)
    gate.probability_protocol = probability_protocol
    gate.fit_diagnostics = dict(
        epochs_run=len(history),
        best_epoch=best_epoch,
        best_validation_bce=best_loss,
        history=history,
        configured_patience=int(config["patience"]),
        early_stopping=bool(config["early_stopping"]),
    )
    from .labels import label_provenance

    gate.scientific_training_protocol = dict(
        label_protocol=label_provenance(label_schema),
        covariates=asdict(
            FrozenCovariateProtocol(
                label_schema.cohort,
                "manuscript-clinical-v1",
                (),
                tuple(config["covariates"]),
                True,
            )
        ),
        event_schema_version="fixed-cohort-event-set-v1",
        training_feature_provenance_version="background-all-modeled-genes-excluded-v1",
        masked_columns=list(config.get("masked_columns", ())),
        fit_diagnostics=gate.fit_diagnostics,
    )
    if probability_protocol.calibration == "platt":
        logits = gate.model.apply(
            {"params": best_params},
            *[
                val_batch["x"][k]
                for k in (
                    "pt_genomic",
                    "pt_dynamic",
                    "mt_genomic",
                    "mt_dynamic",
                    "shared",
                )
            ],
            train=False,
        )
        known = np.isin(yv, [0, 1])
        gate.calibration = PlattCalibration.fit(
            np.asarray(logits)[known],
            yv[known],
            val_ids.to_numpy()[known],
            training_patient_ids=train_ids,
        )
    gate.ready = True
    return gate


def train_kinetic_from_csv(
    csv_path, params_output, metadata_output, config, *, validation_csv_path
):
    """Fit separate raw training/validation CSVs, calibrate and save the gate."""
    training_df, validation_df = pd.read_csv(csv_path), pd.read_csv(validation_csv_path)
    training_df, validation_df, kinetic_cfg, schema = prepare_kinetic_frames(
        training_df, validation_df, config
    )
    gate = fit_kinetic_model(
        training_df,
        validation_df,
        patient_id_column=config.get("data", {}).get("patient_id_column", "patient_id"),
        label_column=config.get("data", {}).get("stop_label_column", "Patient_Label"),
        model_config=kinetic_cfg,
        label_schema=schema,
    )
    gate.save(params_output, metadata_output)
    return dict(
        params_output=str(params_output),
        metadata_output=str(metadata_output),
        n_training=len(training_df),
        n_validation=len(validation_df),
        fit_diagnostics=gate.fit_diagnostics,
    )


def prepare_kinetic_frames(training_df, validation_df, config):
    """Normalize published clinical CSV schemas without fitting any transform."""
    from .labels import ProgressionLabelSchema, published_label_schema
    from .preprocessing import split_kinetic_feature_columns

    cohort = str(config.get("cohort", "")).upper()
    if cohort not in ("GENIE", "MELA", "PACA"):
        raise ValueError("Specify GENIE, MELA or PACA for clinical training")
    genes = tuple(
        config.get("genes") or config.get("topology", {}).get("gene_names", ())
    )
    if len(genes) != 12 or len(set(genes)) != 12:
        raise ValueError("Clinical input requires the frozen twelve-gene cohort order")
    rename = {f"{p}.{g}": f"{p}.{g} (M)" for p in ("P", "M") for g in genes}
    training_df = training_df.rename(
        columns={k: v for k, v in rename.items() if v not in training_df}
    )
    validation_df = validation_df.rename(
        columns={k: v for k, v in rename.items() if v not in validation_df}
    )
    raw_schema = config.get("label_protocol")
    schema = (
        ProgressionLabelSchema(**raw_schema)
        if raw_schema
        else published_label_schema(cohort)
    )
    kinetic_cfg = dict(config.get("kinetic", {}), gene_names=genes)
    kinetic_cfg.setdefault("random_seed", int(config.get("random_seed", 11)))
    if "covariates" not in kinetic_cfg:
        groups = split_kinetic_feature_columns(training_df.columns)
        covariates = [
            c for k in ("pt_dynamic", "mt_dynamic", "shared") for c in groups[k]
        ]
        # Global evolutionary summaries are not additional clinical measurements.
        if any(c.startswith("nMut_") for c in covariates):
            covariates = [
                c
                for c in covariates
                if c not in ("background_nMut", "background_VAF_mean")
            ]
        if "Age_at_Diagnosis" in covariates or any(
            c.endswith("AgeAtSeqRep") for c in covariates
        ):
            covariates = [c for c in covariates if c != "age"]
        kinetic_cfg["covariates"] = covariates
    from .features import FEATURE_METADATA
    from .labels import label_provenance

    for frame in (training_df, validation_df):
        frame.attrs["feature_metadata"] = {
            c: FEATURE_METADATA[name]
            for name in ("nMut", "VAF_mean", "CNA", "FGA", "CNA_adjusted")
            for c in frame
            if c in (f"{name}_Primary", f"{name}_Metastatic")
        }
        frame.attrs["label_provenance"] = dict(frame.attrs.get("label_provenance", {}))
        if "Patient_Label" in frame and schema.pfs_column in frame:
            frame.attrs["label_provenance"]["Patient_Label"] = label_provenance(schema)
    return training_df, validation_df, kinetic_cfg, schema
