"""Kinetic gatekeeper for operational Stop versus Go progression propensity."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax import serialization
from flax.training import train_state
from .probability import KineticProbabilityProtocol, PlattCalibration


class KineticGatekeeperNetwork(nn.Module):
    """Late-fusion MLP used to estimate P(Stop)."""

    d_model: int
    n_head_layers: int
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(
        self,
        x_pt_g: jnp.ndarray,
        x_pt_d: jnp.ndarray,
        x_mt_g: jnp.ndarray,
        x_mt_d: jnp.ndarray,
        x_shared: jnp.ndarray,
        train: bool = True,
        dynamic_dropout_rate: float = 0.0,
    ) -> jnp.ndarray:
        x_pt_d = nn.Dropout(rate=dynamic_dropout_rate, deterministic=not train)(x_pt_d)
        x_mt_d = nn.Dropout(rate=dynamic_dropout_rate, deterministic=not train)(x_mt_d)

        def dense_input(values, width):
            # A frozen schema may have an empty branch: a (0,width) zero kernel
            # gives an explicit bias-only map, without inventing a covariate.
            initializer = nn.initializers.zeros_init() if values.shape[-1] == 0 else nn.initializers.lecun_normal()
            return nn.Dense(features=width, kernel_init=initializer)(values)

        pt_features = dense_input(jnp.concatenate([x_pt_g, x_pt_d], axis=-1), self.d_model)
        pt_features = nn.LayerNorm(epsilon=1e-5)(pt_features)
        pt_features = nn.relu(pt_features)
        pt_features = nn.Dropout(rate=self.dropout_rate)(pt_features, deterministic=not train)

        mt_features = dense_input(jnp.concatenate([x_mt_g, x_mt_d], axis=-1), self.d_model)
        mt_features = nn.LayerNorm(epsilon=1e-5)(mt_features)
        mt_features = nn.relu(mt_features)
        mt_features = nn.Dropout(rate=self.dropout_rate)(mt_features, deterministic=not train)

        shared_features = dense_input(x_shared, max(1, self.d_model // 2))
        shared_features = nn.LayerNorm(epsilon=1e-5)(shared_features)
        shared_features = nn.relu(shared_features)

        fused = jnp.concatenate([pt_features, mt_features, shared_features], axis=-1)
        head = fused
        for _ in range(self.n_head_layers):
            head = nn.Dense(features=self.d_model)(head)
            head = nn.relu(head)
            head = nn.Dropout(rate=self.dropout_rate)(head, deterministic=not train)

        logits = nn.Dense(features=1)(head)
        return jnp.squeeze(logits, axis=-1)


class ProEHNTrainState(train_state.TrainState):
    """Flax train state for the kinetic network."""


def sigmoid_focal_loss(logits: jnp.ndarray, labels: jnp.ndarray, alpha: float = -1.0, gamma: float = 2.0) -> jnp.ndarray:
    """Binary focal loss used by the current source implementation."""

    p = jax.nn.sigmoid(logits)
    ce_loss = optax.sigmoid_binary_cross_entropy(logits, labels)
    p_t = p * labels + (1 - p) * (1 - labels)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
        loss = alpha_t * loss
    return loss


def masked_binary_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    positive_weight: float = 1.0,
    focal_gamma: float = 2.0,
    calculate_metrics: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Masked focal/BCE objective. Label -1 is treated as missing."""

    binary_labels = (labels == 1).astype(jnp.float32)
    mask = (labels != -1).astype(jnp.float32)
    loss_per_sample = sigmoid_focal_loss(logits, binary_labels, alpha=-1.0, gamma=focal_gamma)
    sample_weights = jnp.where(binary_labels == 1, positive_weight, 1.0)
    loss = (loss_per_sample * sample_weights * mask).sum() / (mask.sum() + 1e-6)

    if not calculate_metrics:
        return loss, jnp.array(0.0), mask.sum(), jnp.zeros_like(logits)

    preds = (logits > 0).astype(jnp.int32)
    correct = (preds == labels.astype(jnp.int32)).astype(jnp.float32)
    accuracy = (correct * mask).sum() / (mask.sum() + 1e-6)
    return loss, accuracy, mask.sum(), logits


def kinetic_loss_fn(
    params: Mapping[str, Any],
    state: ProEHNTrainState,
    batch: Mapping[str, Any],
    model_key: jax.Array,
    positive_weight: float,
    focal_gamma: float,
    train: bool = True,
    dynamic_dropout_rate: float = 0.0,
) -> Any:
    logits = state.apply_fn(
        {"params": params},
        x_pt_g=batch["x"]["pt_genomic"],
        x_pt_d=batch["x"]["pt_dynamic"],
        x_mt_g=batch["x"]["mt_genomic"],
        x_mt_d=batch["x"]["mt_dynamic"],
        x_shared=batch["x"]["shared"],
        train=train,
        dynamic_dropout_rate=dynamic_dropout_rate,
        rngs={"dropout": model_key},
    )
    loss, accuracy, _, raw_logits = masked_binary_loss(
        logits,
        batch["y"],
        positive_weight=positive_weight,
        focal_gamma=focal_gamma,
        calculate_metrics=not train,
    )
    if train:
        return loss
    return {"loss": loss, "accuracy": accuracy}, jax.nn.sigmoid(raw_logits)


def create_train_state(
    rng: jax.Array,
    model_config: Mapping[str, Any],
    feature_sizes: Mapping[str, int],
) -> ProEHNTrainState:
    """Initialize the kinetic gatekeeper and optimizer."""

    model = KineticGatekeeperNetwork(
        d_model=int(model_config.get("d_model", 128)),
        n_head_layers=int(model_config.get("n_head_layers", 2)),
        dropout_rate=float(model_config.get("dropout_rate", 0.2)),
    )
    dummy_inputs = (
        jnp.ones([1, feature_sizes["pt_genomic"]]),
        jnp.ones([1, feature_sizes["pt_dynamic"]]),
        jnp.ones([1, feature_sizes["mt_genomic"]]),
        jnp.ones([1, feature_sizes["mt_dynamic"]]),
        jnp.ones([1, feature_sizes["shared"]]),
    )
    params = model.init(rng, *dummy_inputs, train=False)["params"]
    tx = optax.adamw(
        learning_rate=float(model_config.get("learning_rate", 1e-4)),
        weight_decay=float(model_config.get("weight_decay", 1e-3)),
    )
    return ProEHNTrainState.create(apply_fn=model.apply, params=params, tx=tx)


class ProEHNKineticGatekeeper:
    """Inference wrapper for the trained kinetic gatekeeper."""

    def __init__(
        self,
        params_path: str | Path | None,
        metadata_path: str | Path | None,
        fallback_go_probability: float = 0.85,
    ) -> None:
        self.params_path = Path(params_path) if params_path else None
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.fallback_go_probability = float(fallback_go_probability)
        self.probability_protocol = KineticProbabilityProtocol()
        self.calibration = None
        self.ready = False
        self.preprocessor = None
        self.model: KineticGatekeeperNetwork | None = None
        self.params: Any = None
        self.scalers: Mapping[str, Any] = {}
        self.feature_groups: Mapping[str, list[str]] = {}
        self.model_config: dict[str, Any] = {"d_model": 128, "n_head_layers": 2, "dropout_rate": 0.2}

        if self.params_path and self.metadata_path and self.params_path.exists() and self.metadata_path.exists():
            self._load()

    def _load(self) -> None:
        with self.metadata_path.open("rb") as handle:
            metadata = pickle.load(handle)

        if "probability_protocol" not in metadata:
            raise ValueError("Formal kinetic artifact requires probability protocol provenance")
        if metadata.get('scientific_contract_version') != 3 or not metadata.get('scientific_training_protocol'):
            raise ValueError('Incomplete formal kinetic artifact provenance')
        self.scientific_training_protocol = metadata['scientific_training_protocol']
        self._validate_training_protocol(self.scientific_training_protocol)
        self.probability_protocol = KineticProbabilityProtocol(**metadata["probability_protocol"])
        self.calibration = PlattCalibration(**metadata["calibration"]) if metadata.get("calibration") else None
        if self.probability_protocol.calibration == "platt" and self.calibration is None:
            raise ValueError("Missing fitted validation calibration")
        self.scalers = metadata.get("scalers", metadata.get("scaler", {}))
        self.feature_groups = metadata["feature_groups"]
        self.preprocessor = metadata.get("preprocessor")
        self.model_config.update(metadata.get("model_config", {}))
        self.model = KineticGatekeeperNetwork(
            d_model=int(self.model_config.get("d_model", 128)),
            n_head_layers=int(self.model_config.get("n_head_layers", 2)),
            dropout_rate=float(self.model_config.get("dropout_rate", 0.2)),
        )
        dummy_inputs = tuple(jnp.ones((1, len(self.feature_groups[group]))) for group in ["pt_genomic", "pt_dynamic", "mt_genomic", "mt_dynamic", "shared"])
        variables = self.model.init(jax.random.PRNGKey(0), *dummy_inputs, train=False)
        self.params = serialization.from_bytes(variables["params"], self.params_path.read_bytes())
        self.ready = True

    @staticmethod
    def _validate_training_protocol(protocol):
        from .features import FrozenCovariateProtocol
        label = protocol.get('label_protocol', {})
        required = ('schema_id', 'version', 'cohort', 'source_variables', 'threshold', 'unit')
        if any(not label.get(k) for k in required) or label['unit'] != 'days' or not np.isfinite(label['threshold']) or label['threshold'] <= 0:
            raise ValueError('Kinetic artifact requires complete frozen clinical label provenance')
        FrozenCovariateProtocol(**protocol.get('covariates', {})).validate()
        if not protocol.get('event_schema_version') or not protocol.get('training_feature_provenance_version'):
            raise ValueError('Kinetic artifact requires event/feature provenance versions')

    def save(self, params_path, metadata_path):
        """Save the fitted network, train-only preprocessing and validation calibration."""
        from dataclasses import asdict
        if not self.ready or self.preprocessor is None or not getattr(self, 'scientific_training_protocol', None):
            raise ValueError('Complete fitted formal kinetic provenance required')
        if self.probability_protocol.calibration == 'platt' and self.calibration is None:
            raise ValueError('Missing fitted calibration')
        self._validate_training_protocol(self.scientific_training_protocol)
        metadata = dict(scientific_contract_version=3, model_version='ProEHN-round2',
            feature_groups=self.feature_groups, preprocessor=self.preprocessor, model_config=self.model_config,
            probability_protocol=asdict(self.probability_protocol),
            calibration=asdict(self.calibration) if self.calibration is not None else None,
            scientific_training_protocol=self.scientific_training_protocol)
        Path(params_path).write_bytes(serialization.to_bytes(self.params))
        Path(metadata_path).write_bytes(pickle.dumps(metadata))

    def _build_inputs(self, patient_data: Mapping[str, Any]) -> dict[str, jnp.ndarray]:
        if self.preprocessor is not None:
            import pandas as pd
            transformed = self.preprocessor.transform(pd.DataFrame([patient_data]))
            return {k: jnp.asarray(v) for k, v in transformed.items()}
        inputs: dict[str, jnp.ndarray] = {}
        for group in ["pt_genomic", "mt_genomic"]:
            values = [float(patient_data.get(col, 0.0)) for col in self.feature_groups[group]]
            inputs[group] = jnp.array([values])

        for group in ["pt_dynamic", "mt_dynamic", "shared"]:
            values_np = np.array([[float(patient_data.get(col, 0.0)) for col in self.feature_groups[group]]], dtype=np.float32)
            scaler = self.scalers.get(group) if isinstance(self.scalers, Mapping) else None
            if scaler is not None:
                values_np = scaler.transform(values_np)
            inputs[group] = jnp.array(values_np)
        return inputs

    def predict_stop_probability(self, patient_data: Mapping[str, Any]) -> float:
        """Return operational P(Stop); missing fitted components are an error."""

        if not self.ready or self.model is None:
            raise RuntimeError("Fitted kinetic components are required; fixed-probability surrogate is disabled")

        inputs = self._build_inputs(patient_data)
        logits = self.model.apply(
            {"params": self.params},
            inputs["pt_genomic"],
            inputs["pt_dynamic"],
            inputs["mt_genomic"],
            inputs["mt_dynamic"],
            inputs["shared"],
            train=False,
        )
        if self.probability_protocol.calibration == "platt":
            if self.calibration is None:
                raise ValueError("Missing fitted validation calibration")
            return float(self.calibration.transform(np.asarray(logits)).reshape(-1)[0])
        return float(np.asarray(jax.nn.sigmoid(logits)).reshape(-1)[0])

    def predict_go_probability(self, patient_data: Mapping[str, Any]) -> float:
        return 1.0 - self.predict_stop_probability(patient_data)
