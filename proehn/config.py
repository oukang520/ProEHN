"""Configuration helpers for ProEHN."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class KineticConfig:
    """Configuration for the kinetic gatekeeper K."""

    d_model: int = 128
    n_head_layers: int = 2
    dropout_rate: float = 0.2
    focal_gamma: float = 0.0
    calibration: str = "none"
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 150
    patience: int = 15
    stop_threshold: float = 0.90
    fallback_go_probability: float = 0.85


@dataclass(frozen=True)
class TopologyConfig:
    """Configuration for the feature-modulated topology engine T."""

    top_n_genes: int = 20
    max_batch_size: int = 1
    max_active_events: int = 18
    regularization_strength: float = 0.01
    l1_ratio: float = 1.0
    l2_floor: float = 0.0
    log_rate_clip_min: float = -20.0
    log_rate_clip_max: float = 20.0
    optimizer_maxiter: int = 500
    use_bio_pure_features: bool = False
    min_mutations: int | None = None
    max_mutations: int | None = None
    core_drivers: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArtifactConfig:
    """Model artifact paths."""

    kinetic_params: str = "artifacts/proehn_kinetic_params.msgpack"
    kinetic_metadata: str = "artifacts/proehn_kinetic_metadata.pkl"
    topology_model: str = "artifacts/proehn_topology_model.npz"


@dataclass(frozen=True)
class DataConfig:
    """Column and label conventions expected by the released code."""

    patient_id_column: str = "patient_id"
    seeding_column: str = "Seeding"
    type_column: str = "type"
    diagnosis_order_column: str = "diag_order"
    stop_label_column: str = "Patient_Label"
    primary_prefix: str = "P."
    metastasis_prefix: str = "M."
    mutation_suffix: str = " (M)"


@dataclass(frozen=True)
class ProEHNConfig:
    """Top-level ProEHN configuration."""

    project_name: str = "ProEHN"
    random_seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    kinetic: KineticConfig = field(default_factory=KineticConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)


def _deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _dataclass_to_dict(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {key: _dataclass_to_dict(getattr(obj, key)) for key in obj.__dataclass_fields__}
    if isinstance(obj, tuple):
        return list(obj)
    return obj


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config and merge it over the documented defaults."""

    defaults = _dataclass_to_dict(ProEHNConfig())
    if path is None:
        return defaults

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}
    merged = _deep_update(defaults, user_config)
    from .probability import KineticProbabilityProtocol
    KineticProbabilityProtocol(float(merged['kinetic']['focal_gamma']), merged['kinetic']['calibration'])
    return merged
