"""Deployment-owned model registry; no legacy inference code is executed."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
import math

from proehn.config import load_config
from proehn.engine import ProEHNEngine


@dataclass(frozen=True)
class ModelSpec:
    cohort: str
    suffix: str
    topology: str


MODELS = {
    "paca": ModelSpec("paca", "PACA", "M_model_final.npz"),
    "luad": ModelSpec("luad", "LUAD", "M_model_LC_final.npz"),
    "mela": ModelSpec("mela", "MELA_Final", "m_model_MELA_trained.npz"),
}


class ModelUnavailableError(RuntimeError):
    pass


class ModelAdapter:
    """One worker-owned model instance, configured only from trusted files.

    Do not change global JAX precision, rescale inputs, or normalize outputs here.
    Missing input rules remain in the core; schema requires explicit observation semantics.
    """

    def __init__(self, model_id: str, artifact_root: Path, config_root: Path):
        if model_id not in MODELS:
            raise ValueError("Unknown model")
        self.spec = MODELS[model_id]
        self.model_id = model_id
        self.config = load_config(config_root / f"{self.spec.cohort}.yaml")
        self.paths = (
            artifact_root / f"sg_metmhn_classifier_{self.spec.suffix}.pkl",
            artifact_root / f"sg_metmhn_scalers_{self.spec.suffix}.pkl",
            artifact_root / self.spec.topology,
        )
        if not all(path.is_file() for path in self.paths):
            raise ModelUnavailableError("Required model artifacts are missing")
        self.artifact_checksums = {path.name: sha256(path.read_bytes()).hexdigest() for path in self.paths}
        kinetic, topology = self.config["kinetic"], self.config["topology"]
        self.engine_options = {
            "fallback_go_probability": kinetic["fallback_go_probability"],
            "regularization_strength": topology["regularization_strength"],
            "log_rate_clip_min": topology["log_rate_clip_min"],
            "log_rate_clip_max": topology["log_rate_clip_max"],
        }
        self.engine = ProEHNEngine(*self.paths, **self.engine_options)
        if not self.engine.kinetic.ready or not self.engine.topology_ready:
            raise ModelUnavailableError("Model did not initialize completely")

    def schema(self) -> dict[str, Any]:
        groups = {name: list(columns) for name, columns in self.engine.kinetic.feature_groups.items()}
        topology_fields = [name for name in self.engine.feature_names if name.lower() not in {"bias", "intercept"}]
        mutations = [f"{prefix}.{gene} (M)" for prefix in ("P", "M") for gene in self.engine.gene_names]
        names = list(dict.fromkeys([col for columns in groups.values() for col in columns] + topology_fields + mutations + ["Seeding", "observation_type", "diag_order", "seeding_at_first_observation", "joint_snapshot"]))
        return {
            "model_id": self.model_id,
            "kinetic_groups": groups,
            "topology_features": topology_fields,
            "topology_genes": list(self.engine.gene_names),
            "display_genes": list(self.engine.gene_names[:15]),
            "preprocessing_status": (self.engine.preprocessing or {}).get("provenance_status","embedded_scientific_contract"),
            "fields": [{"name": name, "kind": "binary" if name.endswith((" (M)", " (Amp)", " (Del)")) or name == "Seeding" else "number",
                        "core_missing_default": None} for name in names],
            "artifact_checksums": self.artifact_checksums,
        }

    def predict(self, patient: Mapping[str, Any]) -> dict[str, Any]:
        schema = self.schema()
        known = {field["name"] for field in schema["fields"]}
        unknown = set(patient) - known
        if unknown:
            raise ValueError("Unknown input fields: " + ", ".join(sorted(unknown)))
        for field in schema["fields"]:
            name = field["name"]
            if name not in patient:
                continue
            value = patient[name]
            try:
                valid_number = type(value) in (int, float) and math.isfinite(value)
            except OverflowError:
                valid_number = False
            if not valid_number:
                raise ValueError(f"Expected finite numeric value: {name}")
            if field["kind"] == "binary" and value not in (0, 1):
                raise ValueError(f"Expected zero or one: {name}")
        missing = sorted(known - set(patient))
        # No imputation or scientific calculations in the product adapter.
        result = self.engine.predict(dict(patient))
        numbers = [result["prob_go"], result["prob_stop"]]
        numbers.extend(row[key] for row in result["integrated_event_scores"] for key in ("topology_probability", "integrated_event_score"))
        if not all(math.isfinite(value) for value in numbers):
            raise ModelUnavailableError("Scientific output contains non-finite values")
        return {"model_id": self.model_id, "artifact_checksums": self.artifact_checksums,
                "warnings": [{"code": "INPUT_FIELDS_OMITTED", "fields": missing}] if missing else [],
                "result": result}
