"""Training entry points for the ProEHN core models."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .preprocessing import build_topology_training_data, calculate_marginal_rates
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
    buckets, n_events, n_features, n_samples, gene_names, feature_names = build_topology_training_data(
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
