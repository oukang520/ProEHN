"""Training entry points for the ProEHN core models."""

from __future__ import annotations

from functools import partial
import json
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .preprocessing import build_topology_training_data, calculate_marginal_rates, TopologyTrainingPreprocessor
from .topology import ProEHNTopologyModel


def fit_topology_model(
    buckets: list[tuple[int, int, int, np.ndarray, np.ndarray]],
    n_events: int,
    n_features: int,
    n_samples: int,
    regularization_strength: float = 0.01,
    l1_ratio: float = 1.0,
    log_rate_clip_min: float = -20.0,
    log_rate_clip_max: float = 20.0,
    optimizer_maxiter: int = 500,
    random_seed: int = 42,
    l2_floor: float = 0.0,
) -> tuple[ProEHNTopologyModel, np.ndarray, float]:
    """Fit the feature-modulated CTMC topology engine with L-BFGS-B."""

    if not jax.config.x64_enabled:
        raise RuntimeError('Enable JAX_ENABLE_X64=1 before topology fitting; float64 likelihoods are required')
    if not buckets or n_samples != sum(len(b[3]) for b in buckets):
        raise ValueError('Likelihood normalization must equal included observations')
    model = ProEHNTopologyModel(
        n_events=n_events,
        n_features=n_features,
        regularization_strength=regularization_strength,
        l1_ratio=l1_ratio,
        l2_floor=l2_floor,
        log_rate_clip_min=log_rate_clip_min,
        log_rate_clip_max=log_rate_clip_max,
    )
    base_rates = calculate_marginal_rates(buckets, n_events)
    init_params = jnp.array(model.init_params(jax.random.PRNGKey(random_seed), base_rates=base_rates), dtype=jnp.float64)
    buckets_jax = [
        (
            bucket_type,
            n_primary,
            n_metastatic,
            jax.device_put(jnp.array(bucket_genotypes, dtype=jnp.int8)),
            jax.device_put(jnp.array(bucket_features, dtype=jnp.float64)),
        )
        for bucket_type, n_primary, n_metastatic, bucket_genotypes, bucket_features in buckets
    ]

    @partial(jax.jit, static_argnums=(1, 2, 3))
    def bucket_loss_and_grad(params: jnp.ndarray, bucket_type: int, n_primary: int, n_metastatic: int, genes: jnp.ndarray, feats: jnp.ndarray):
        return jax.value_and_grad(model.bucket_loss)(params, bucket_type, n_primary, n_metastatic, genes, feats)

    @jax.jit
    def regularization_and_grad(params: jnp.ndarray):
        return jax.value_and_grad(model.regularization)(params)

    def objective(params_np: np.ndarray) -> tuple[float, np.ndarray]:
        params = jnp.array(params_np, dtype=jnp.float64)
        total_loss = 0.0
        total_grad = jnp.zeros_like(params)
        for bucket in buckets_jax:
            loss, grad = bucket_loss_and_grad(params, *bucket)
            if not jnp.isfinite(loss) or not jnp.all(jnp.isfinite(grad)):
                raise FloatingPointError("Non-finite likelihood/gradient; optimization aborted")
            else:
                total_loss += loss
                total_grad += grad
        avg_loss = total_loss / max(1, n_samples)
        avg_grad = total_grad / max(1, n_samples)
        reg_loss, reg_grad = regularization_and_grad(params)
        return float(avg_loss + reg_loss), np.asarray(avg_grad + reg_grad, dtype=np.float64)

    result = minimize(
        objective,
        np.asarray(init_params, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(optimizer_maxiter), "disp": False},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Topology optimizer did not converge: {result.message}")
    return model, np.asarray(result.x, dtype=np.float64), float(result.fun)


def train_topology_from_csv(
    csv_path: str | Path,
    output_path: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Train and save a ProEHN topology artifact from a feature table."""

    for required in ("target_event_schema_version", "training_feature_provenance_version"):
        if not config.get("data", {}).get(required):
            raise ValueError(f"Missing frozen artifact provenance: {required}")
    df = pd.read_csv(csv_path)
    data_cfg = config.get("data", {})
    df.attrs["feature_metadata"] = data_cfg.get("feature_metadata", {})
    topo_cfg = config.get("topology", {})
    from .features import TopologyCovariateSchema
    preprocessor = TopologyTrainingPreprocessor(
        int(topo_cfg.get('top_n_genes', 20)), topo_cfg.get('core_drivers', ()),
        TopologyCovariateSchema(tuple(topo_cfg['covariates'])) if 'covariates' in topo_cfg else None,
        data_cfg.get('primary_prefix', 'P.'), data_cfg.get('metastasis_prefix', 'M.'),
        data_cfg.get('mutation_suffix', ' (M)'),
    ).fit(df)
    buckets, n_events, n_features, n_samples, gene_names, feature_names = build_topology_training_data(
        df,
        preprocessor=preprocessor,
        top_n_genes=int(topo_cfg.get("top_n_genes", 20)),
        max_batch_size=int(topo_cfg.get("max_batch_size", 1)),
        max_active_events=int(topo_cfg.get("max_active_events", 18)),
        core_drivers=topo_cfg.get("core_drivers", ()),
        use_bio_pure_features=bool(topo_cfg.get("use_bio_pure_features", False)),
        seeding_column=data_cfg.get("seeding_column", "Seeding"),
        type_column=data_cfg.get("type_column", "type"),
        diagnosis_order_column=data_cfg.get("diagnosis_order_column", "diag_order"),
        primary_prefix=data_cfg.get("primary_prefix", "P."),
        metastasis_prefix=data_cfg.get("metastasis_prefix", "M."),
        mutation_suffix=data_cfg.get("mutation_suffix", " (M)"),
    )
    model, params, final_loss = fit_topology_model(
        buckets,
        n_events,
        n_features,
        n_samples,
        regularization_strength=float(topo_cfg.get("regularization_strength", 0.01)),
        l1_ratio=float(topo_cfg.get("l1_ratio", 1.0)),
        l2_floor=float(topo_cfg.get("l2_floor", 0.0)),
        log_rate_clip_min=float(topo_cfg.get("log_rate_clip_min", -20.0)),
        log_rate_clip_max=float(topo_cfg.get("log_rate_clip_max", 20.0)),
        optimizer_maxiter=int(topo_cfg.get("optimizer_maxiter", 500)),
        random_seed=int(config.get("random_seed", 42)),
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from .artifacts import ScientificArtifactMetadata, save_topology_artifact
    metadata = ScientificArtifactMetadata(tuple(gene_names), tuple(feature_names),
        tuple(preprocessor.covariates.columns), preprocessor.covariates.metadata(),
        model.log_rate_clip_min, model.log_rate_clip_max,
        {k: getattr(model, k) for k in ('regularization_strength', 'l1_ratio', 'l2_floor')},
        data_cfg['target_event_schema_version'], data_cfg['training_feature_provenance_version'])
    save_topology_artifact(output_path, params, metadata)
    return {
        "output_path": str(output_path),
        "final_loss": final_loss,
        "n_events": n_events,
        "n_features": n_features,
        "n_samples": n_samples,
        "gene_names": gene_names,
        "feature_names": feature_names,
    }


def fit_kinetic_model(training_df, validation_df, *, patient_id_column,
                      label_column='Patient_Label', model_config=None):
    """Train formal gatekeeper after splitting; early stopping sees inner val only.

    This entry point has no test-data argument and writes no artifacts. It uses
    the operational Stop=1 convention. Focal/weighted outputs require independent
    calibration before an absolute-probability interpretation can be justified.
    """
    from .preprocessing import KineticPreprocessor, build_kinetic_matrices
    from .kinetic import ProEHNKineticGatekeeper, create_train_state, kinetic_loss_fn
    from .probability import KineticProbabilityProtocol, PlattCalibration
    config = dict(model_config or {})
    probability_protocol = KineticProbabilityProtocol(float(config.get('focal_gamma', 0.)), config.get('calibration', 'none'))
    train_ids, val_ids = training_df[patient_id_column], validation_df[patient_id_column]
    if train_ids.isna().any() or val_ids.isna().any() or set(train_ids) & set(val_ids):
        raise ValueError('Known, disjoint training and inner-validation patients required')
    prep = KineticPreprocessor(config.get('covariates')).fit(training_df)
    xt, yt, _ = build_kinetic_matrices(training_df, label_column, preprocessor=prep)
    xv, yv, _ = build_kinetic_matrices(validation_df, label_column, preprocessor=prep)
    if not np.isin(yt, [0, 1]).any() or not np.isin(yv, [0, 1]).any():
        raise ValueError('Training and validation each require observed progression labels')
    key = jax.random.PRNGKey(int(config.get('random_seed', 42)))
    state = create_train_state(key, config, {k: v.shape[1] for k, v in xt.items()})
    train_batch = dict(x={k: jnp.asarray(v) for k, v in xt.items()}, y=jnp.asarray(yt))
    val_batch = dict(x={k: jnp.asarray(v) for k, v in xv.items()}, y=jnp.asarray(yv))
    # BCE by default: a proper scoring rule; no outcome-dependent class weights.
    gamma = float(config.get('focal_gamma', 0.))
    best_loss, best_params, stale = np.inf, None, 0
    for _ in range(int(config.get('num_epochs', 150))):
        key, step_key = jax.random.split(key)
        loss, grad = jax.value_and_grad(kinetic_loss_fn)(state.params, state, train_batch, step_key, 1., gamma)
        if not np.isfinite(float(loss)):
            raise FloatingPointError('Non-finite kinetic training loss')
        state = state.apply_gradients(grads=grad)
        metrics, _ = kinetic_loss_fn(state.params, state, val_batch, step_key, 1., gamma, train=False)
        val_loss = float(metrics['loss'])
        if not np.isfinite(val_loss):
            raise FloatingPointError('Non-finite kinetic validation loss')
        if val_loss < best_loss:
            best_loss, best_params, stale = val_loss, state.params, 0
        else:
            stale += 1
        if stale >= int(config.get('patience', 15)):
            break
    if best_params is None:
        raise ValueError('num_epochs must be positive')
    from .kinetic import KineticGatekeeperNetwork
    gate = ProEHNKineticGatekeeper(None, None)
    gate.model = KineticGatekeeperNetwork(int(config.get('d_model', 128)), int(config.get('n_head_layers', 2)), float(config.get('dropout_rate', .2)))
    gate.params, gate.preprocessor, gate.feature_groups = best_params, prep, prep.feature_groups
    gate.model_config.update(config)
    gate.probability_protocol = probability_protocol
    if probability_protocol.calibration == "platt":
        logits = gate.model.apply({"params": best_params}, *[val_batch["x"][k] for k in ("pt_genomic", "pt_dynamic", "mt_genomic", "mt_dynamic", "shared")], train=False)
        known = np.isin(yv, [0, 1])
        gate.calibration = PlattCalibration.fit(np.asarray(logits)[known], yv[known], val_ids.to_numpy()[known], training_patient_ids=train_ids)
    gate.ready = True
    return gate



def train_kinetic_from_csv(*args, **kwargs):
    """Retired single-table entry point that split after preprocessing.

    Use fit_kinetic_model(training_df, validation_df, patient_id_column=...) with
    raw patient-disjoint partitions and an explicit clinical label protocol.
    """
    raise ValueError(
        'Single-table kinetic training is disabled: split raw patients before '
        'preprocessing and call fit_kinetic_model with training/validation frames.'
    )
