"""Training entry points for the ProEHN core models."""

from __future__ import annotations

from functools import partial
import pickle
from pathlib import Path
from typing import Any, Mapping

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .kinetic import create_train_state, kinetic_loss_fn
from .preprocessing import (
    build_kinetic_matrices,
    build_topology_training_data,
    calculate_marginal_rates,
)
from .topology import ProEHNTopologyModel


KINETIC_GROUPS = ("pt_genomic", "pt_dynamic", "mt_genomic", "mt_dynamic", "shared")


def _kinetic_batch(
    matrices: Mapping[str, np.ndarray],
    labels: np.ndarray,
    indices: np.ndarray,
) -> dict[str, Any]:
    return {
        "x": {group: jnp.array(matrices[group][indices]) for group in KINETIC_GROUPS},
        "y": jnp.array(labels[indices]),
    }


def _evaluate_kinetic(
    state: Any,
    matrices: Mapping[str, np.ndarray],
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    positive_weight: float,
    focal_gamma: float,
) -> dict[str, float]:
    if len(indices) == 0:
        return {"loss": float("nan"), "accuracy": float("nan")}
    batch = _kinetic_batch(matrices, labels, indices)
    metrics, _ = kinetic_loss_fn(
        state.params,
        state,
        batch,
        jax.random.PRNGKey(0),
        positive_weight=positive_weight,
        focal_gamma=focal_gamma,
        train=False,
    )
    return {key: float(value) for key, value in metrics.items()}


def fit_kinetic_gatekeeper(
    matrices: Mapping[str, np.ndarray],
    labels: np.ndarray,
    metadata: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    random_seed: int = 42,
) -> tuple[Any, bytes, dict[str, Any]]:
    """Fit the kinetic gatekeeper and return state, serialized params and metadata."""

    kinetic_cfg = config.get("kinetic", {})
    valid_indices = np.flatnonzero(labels != -1)
    if len(valid_indices) == 0:
        raise ValueError("No valid kinetic labels were found; labels must be 0, 1 or -1.")

    rng_np = np.random.default_rng(random_seed)
    valid_indices = rng_np.permutation(valid_indices)
    validation_fraction = float(kinetic_cfg.get("validation_fraction", 0.2))
    n_val = int(round(len(valid_indices) * validation_fraction))
    if len(valid_indices) > 1:
        n_val = min(max(n_val, 1), len(valid_indices) - 1)
    else:
        n_val = 0
    val_indices = valid_indices[:n_val]
    train_indices = valid_indices[n_val:]

    y_train = labels[train_indices]
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    positive_weight = float(
        kinetic_cfg.get(
            "positive_weight",
            negatives / max(positives, 1),
        )
    )
    focal_gamma = float(kinetic_cfg.get("focal_gamma", 2.0))
    batch_size = int(kinetic_cfg.get("batch_size", 32))
    num_epochs = int(kinetic_cfg.get("num_epochs", 150))
    patience = int(kinetic_cfg.get("patience", 15))
    dynamic_dropout_rate = float(kinetic_cfg.get("dynamic_dropout_rate", 0.0))
    feature_sizes = {group: int(matrices[group].shape[1]) for group in KINETIC_GROUPS}

    key = jax.random.PRNGKey(random_seed)
    state = create_train_state(key, kinetic_cfg, feature_sizes)

    @jax.jit
    def train_step(train_state: Any, batch: Mapping[str, Any], step_key: jax.Array):
        def loss_fn(params: Mapping[str, Any]) -> jnp.ndarray:
            return kinetic_loss_fn(
                params,
                train_state,
                batch,
                step_key,
                positive_weight=positive_weight,
                focal_gamma=focal_gamma,
                train=True,
                dynamic_dropout_rate=dynamic_dropout_rate,
            )

        loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
        return train_state.apply_gradients(grads=grads), loss

    best_params = state.params
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, num_epochs + 1):
        train_indices = rng_np.permutation(train_indices)
        losses: list[float] = []
        for start in range(0, len(train_indices), batch_size):
            batch_indices = train_indices[start : start + batch_size]
            key, step_key = jax.random.split(key)
            state, loss = train_step(
                state,
                _kinetic_batch(matrices, labels, batch_indices),
                step_key,
            )
            losses.append(float(loss))

        train_loss = float(np.mean(losses)) if losses else float("nan")
        val_metrics = _evaluate_kinetic(
            state,
            matrices,
            labels,
            val_indices,
            positive_weight=positive_weight,
            focal_gamma=focal_gamma,
        )
        val_loss = val_metrics["loss"] if np.isfinite(val_metrics["loss"]) else train_loss
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": float(val_loss),
                "val_accuracy": float(val_metrics["accuracy"]),
            }
        )
        if val_loss < best_val_loss - 1e-8:
            best_val_loss = float(val_loss)
            best_params = state.params
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    state = state.replace(params=best_params)
    artifact_metadata = dict(metadata)
    artifact_metadata.update(
        {
            "model_config": dict(kinetic_cfg),
            "feature_sizes": feature_sizes,
            "train_samples": int(len(train_indices)),
            "validation_samples": int(len(val_indices)),
            "positive_weight": float(positive_weight),
            "history": history,
        }
    )
    return state, serialization.to_bytes(best_params), artifact_metadata


def train_kinetic_from_csv(
    csv_path: str | Path,
    params_output_path: str | Path,
    metadata_output_path: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Train and save a ProEHN kinetic gatekeeper artifact from a feature table."""

    df = pd.read_csv(csv_path)
    data_cfg = config.get("data", {})
    matrices, labels, metadata = build_kinetic_matrices(
        df,
        label_column=data_cfg.get("stop_label_column", "Patient_Label"),
    )
    _, params_bytes, artifact_metadata = fit_kinetic_gatekeeper(
        matrices,
        labels,
        metadata,
        config=config,
        random_seed=int(config.get("random_seed", 42)),
    )

    params_output_path = Path(params_output_path)
    metadata_output_path = Path(metadata_output_path)
    params_output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    params_output_path.write_bytes(params_bytes)
    with metadata_output_path.open("wb") as handle:
        pickle.dump(artifact_metadata, handle)

    valid_labels = labels[labels != -1]
    return {
        "params_output_path": str(params_output_path),
        "metadata_output_path": str(metadata_output_path),
        "n_samples": int(len(labels)),
        "valid_label_count": int(len(valid_labels)),
        "positive_label_count": int((valid_labels == 1).sum()),
        "negative_label_count": int((valid_labels == 0).sum()),
        "feature_sizes": artifact_metadata["feature_sizes"],
        "epochs_completed": len(artifact_metadata["history"]),
        "best_val_loss": min(item["val_loss"] for item in artifact_metadata["history"]),
    }


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
) -> tuple[ProEHNTopologyModel, np.ndarray, float]:
    """Fit the feature-modulated CTMC topology engine with L-BFGS-B."""

    model = ProEHNTopologyModel(
        n_events=n_events,
        n_features=n_features,
        regularization_strength=regularization_strength,
        l1_ratio=l1_ratio,
        log_rate_clip_min=log_rate_clip_min,
        log_rate_clip_max=log_rate_clip_max,
    )
    base_rates = calculate_marginal_rates(buckets, n_events)
    init_params = jnp.array(
        model.init_params(jax.random.PRNGKey(random_seed), base_rates=base_rates),
        dtype=jnp.float64,
    )
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
    def bucket_loss_and_grad(
        params: jnp.ndarray,
        bucket_type: int,
        n_primary: int,
        n_metastatic: int,
        genes: jnp.ndarray,
        feats: jnp.ndarray,
    ):
        return jax.value_and_grad(model.bucket_loss)(
            params,
            bucket_type,
            n_primary,
            n_metastatic,
            genes,
            feats,
        )

    @jax.jit
    def regularization_and_grad(params: jnp.ndarray):
        return jax.value_and_grad(model.regularization)(params)

    def objective(params_np: np.ndarray) -> tuple[float, np.ndarray]:
        params = jnp.array(params_np, dtype=jnp.float64)
        total_loss = 0.0
        total_grad = jnp.zeros_like(params)
        for bucket in buckets_jax:
            loss, grad = bucket_loss_and_grad(params, *bucket)
            if jnp.isnan(loss) or jnp.any(jnp.isnan(grad)):
                total_loss += 1e6
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
    return model, np.asarray(result.x, dtype=np.float64), float(result.fun)


def train_topology_from_csv(
    csv_path: str | Path,
    output_path: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Train and save a ProEHN topology artifact from a feature table."""

    df = pd.read_csv(csv_path)
    data_cfg = config.get("data", {})
    topo_cfg = config.get("topology", {})
    (
        buckets,
        n_events,
        n_features,
        n_samples,
        gene_names,
        feature_names,
    ) = build_topology_training_data(
        df,
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
    _, params, final_loss = fit_topology_model(
        buckets,
        n_events,
        n_features,
        n_samples,
        regularization_strength=float(topo_cfg.get("regularization_strength", 0.01)),
        l1_ratio=float(topo_cfg.get("l1_ratio", 1.0)),
        log_rate_clip_min=float(topo_cfg.get("log_rate_clip_min", -20.0)),
        log_rate_clip_max=float(topo_cfg.get("log_rate_clip_max", 20.0)),
        optimizer_maxiter=int(topo_cfg.get("optimizer_maxiter", 500)),
        random_seed=int(config.get("random_seed", 42)),
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        params=params,
        gene_names=np.array(gene_names, dtype=object),
        feature_names=np.array(feature_names, dtype=object),
        final_loss=final_loss,
        n_samples=n_samples,
    )
    return {
        "output_path": str(output_path),
        "final_loss": final_loss,
        "n_events": n_events,
        "n_features": n_features,
        "n_samples": n_samples,
        "gene_names": gene_names,
        "feature_names": feature_names,
    }
