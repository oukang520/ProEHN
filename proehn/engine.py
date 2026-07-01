"""Unified ProEHN inference engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np

from .kinetic import ProEHNKineticGatekeeper
from .topology import ProEHNTopologyModel


class ProEHNEngine:
    """Three-stage ProEHN inference.

    Stage 1 estimates P(Go) = 1 - P(Stop), stage 2 computes accessible CTMC
    transition hazards, and stage 3 returns P(next event) = P(Go) * normalized
    topological hazard.
    """

    def __init__(
        self,
        kinetic_params_path: str | Path | None = None,
        kinetic_metadata_path: str | Path | None = None,
        topology_model_path: str | Path | None = None,
        stop_threshold: float = 0.90,
        fallback_go_probability: float = 0.85,
        regularization_strength: float = 0.01,
        log_rate_clip_min: float = -20.0,
        log_rate_clip_max: float = 20.0,
    ) -> None:
        self.stop_threshold = float(stop_threshold)
        self.kinetic = ProEHNKineticGatekeeper(
            kinetic_params_path,
            kinetic_metadata_path,
            fallback_go_probability=fallback_go_probability,
        )

        self.topology_ready = False
        self.topology_model: ProEHNTopologyModel | None = None
        self.topology_params: jnp.ndarray | None = None
        self.W_theta: jnp.ndarray | None = None
        self.W_dp: jnp.ndarray | None = None
        self.W_dm: jnp.ndarray | None = None
        self.gene_names: list[str] = []
        self.feature_names: list[str] = []

        if topology_model_path and Path(topology_model_path).exists():
            self._load_topology(
                topology_model_path,
                regularization_strength=regularization_strength,
                log_rate_clip_min=log_rate_clip_min,
                log_rate_clip_max=log_rate_clip_max,
            )

    def _load_topology(
        self,
        topology_model_path: str | Path,
        regularization_strength: float,
        log_rate_clip_min: float,
        log_rate_clip_max: float,
    ) -> None:
        data = np.load(topology_model_path, allow_pickle=True)
        self.topology_params = jnp.array(data["params"])
        self.gene_names = [str(x) for x in data["gene_names"].tolist()]
        self.feature_names = [str(x) for x in data["feature_names"].tolist()]
        self.topology_model = ProEHNTopologyModel(
            n_events=len(self.gene_names),
            n_features=len(self.feature_names),
            regularization_strength=regularization_strength,
            log_rate_clip_min=log_rate_clip_min,
            log_rate_clip_max=log_rate_clip_max,
        )
        self.W_theta, self.W_dp, self.W_dm = self.topology_model.parse_params(self.topology_params)
        self.topology_ready = True

    def _feature_vector(self, patient_data: Mapping[str, Any]) -> jnp.ndarray:
        values = [1.0]
        for name in self.feature_names:
            if name.lower() in {"bias", "intercept"}:
                continue
            values.append(float(patient_data.get(name, 0.0)))

        expected = self.topology_model.n_features if self.topology_model is not None else len(values)
        if len(values) < expected:
            values.extend([0.0] * (expected - len(values)))
        elif len(values) > expected:
            values = values[:expected]
        return jnp.array(values)

    def _genotype_vectors(self, patient_data: Mapping[str, Any]) -> tuple[list[int], list[int], bool]:
        pt_vec: list[int] = []
        mt_vec: list[int] = []
        has_metastasis = bool(patient_data.get("Seeding", patient_data.get("has_metastasis", 0)))

        for gene in self.gene_names:
            p_val = int(patient_data.get(f"P.{gene} (M)", patient_data.get(f"P.{gene}", 0)))
            m_val = int(patient_data.get(f"M.{gene} (M)", patient_data.get(f"M.{gene}", 0)))
            pt_vec.append(p_val)
            mt_vec.append(m_val)
            has_metastasis = has_metastasis or bool(m_val)
        return pt_vec, mt_vec, has_metastasis

    def transition_rates(self, patient_data: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Compute accessible topological hazards for one patient."""

        if not self.topology_ready or self.topology_model is None:
            return []

        features = self._feature_vector(patient_data)
        theta, _, _ = self.topology_model.compute_patient_params(
            self.W_theta,
            self.W_dp,
            self.W_dm,
            features,
            self.topology_model.log_rate_clip_min,
            self.topology_model.log_rate_clip_max,
        )
        theta_np = np.asarray(theta)
        pt_vec, mt_vec, has_metastasis = self._genotype_vectors(patient_data)
        rates: list[dict[str, Any]] = []

        active_pt = [idx for idx, value in enumerate(pt_vec) if value == 1]
        for idx, gene in enumerate(self.gene_names):
            if pt_vec[idx] == 0:
                log_rate = theta_np[idx, idx] + theta_np[idx, active_pt].sum()
                rates.append({"compartment": "primary", "event": gene, "rate": float(np.exp(log_rate))})

        if not has_metastasis:
            seed_idx = len(self.gene_names)
            log_seed = theta_np[seed_idx, seed_idx] + theta_np[seed_idx, active_pt].sum()
            rates.append({"compartment": "primary", "event": "Metastatic seeding", "rate": float(np.exp(log_seed))})

        if has_metastasis:
            active_mt = [idx for idx, value in enumerate(mt_vec) if value == 1]
            for idx, gene in enumerate(self.gene_names):
                if mt_vec[idx] == 0:
                    log_rate = theta_np[idx, idx] + theta_np[idx, active_mt].sum()
                    rates.append({"compartment": "metastasis", "event": gene, "rate": float(np.exp(log_rate))})
        return rates

    def predict(self, patient_data: Mapping[str, Any]) -> dict[str, Any]:
        """Return kinetic, topological and absolute next-event risks."""

        prob_go = self.kinetic.predict_go_probability(patient_data)
        prob_stop = 1.0 - prob_go
        rates = self.transition_rates(patient_data)
        total_rate = sum(item["rate"] for item in rates) + 1e-9
        absolute_risks = [
            {
                "compartment": item["compartment"],
                "event": item["event"],
                "rate": item["rate"],
                "topology_probability": item["rate"] / total_rate,
                "absolute_risk": (item["rate"] / total_rate) * prob_go,
            }
            for item in rates
        ]
        absolute_risks.sort(key=lambda item: item["absolute_risk"], reverse=True)
        return {
            "prob_go": prob_go,
            "prob_stop": prob_stop,
            "is_stable_predicted": prob_stop > self.stop_threshold,
            "has_metastasis": bool(patient_data.get("Seeding", patient_data.get("has_metastasis", 0))),
            "absolute_risks": absolute_risks,
        }
