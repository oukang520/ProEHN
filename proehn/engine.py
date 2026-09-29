"""Unified ProEHN inference engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np
import json
from scipy.special import softmax

from .kinetic import ProEHNKineticGatekeeper
from .topology import ProEHNTopologyModel
from .features import FoldPreprocessor
from .representation import accessible_transition_log_rates


class ProEHNEngine:
    """Three-stage ProEHN inference.

    Stage 1 estimates P(Go) = 1 - P(Stop), stage 2 computes accessible CTMC
    transition hazards, and stage 3 returns the integrated score P(Go) * normalized
    topological hazard.
    """

    def __init__(
        self,
        kinetic_params_path: str | Path | None = None,
        kinetic_metadata_path: str | Path | None = None,
        topology_model_path: str | Path | None = None,
        stop_threshold: float = 0.90,
        fallback_go_probability: float = 0.85,
        regularization_strength: float | None = None,
        log_rate_clip_min: float | None = None,
        log_rate_clip_max: float | None = None,
        topology_preprocessing_path: str | Path | None = None,
    ) -> None:
        self.preprocessing = None
        self.fold_preprocessor = None
        if topology_preprocessing_path:
            self.preprocessing = json.loads(Path(topology_preprocessing_path).read_text(encoding='utf-8'))
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
        from .artifacts import load_topology_artifact
        if self.preprocessing is not None:
            raise ValueError('Formal inference uses embedded training preprocessing only')
        self.topology_model, params, self.scientific_metadata, self.fold_preprocessor = load_topology_artifact(
            topology_model_path, regularization_strength=regularization_strength,
            log_rate_clip_min=log_rate_clip_min, log_rate_clip_max=log_rate_clip_max)
        self.topology_params = jnp.asarray(params)
        self.gene_names = list(self.scientific_metadata.gene_names)
        self.feature_names = list(self.scientific_metadata.feature_names)
        self.W_theta, self.W_dp, self.W_dm = self.topology_model.parse_params(self.topology_params)
        self.topology_ready = True

    def _feature_vector(self, patient_data: Mapping[str, Any]) -> jnp.ndarray:
        if self.fold_preprocessor is not None:
            import pandas as pd
            features = self.fold_preprocessor.transform(pd.DataFrame([patient_data]))[0]
            return jnp.asarray(np.r_[1., features])
        if self.feature_names and self.preprocessing is None:
            raise ValueError('Training preprocessing metadata is required; evaluation-batch fitting is forbidden')
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
        if self.preprocessing:
            features=np.asarray(values[1:],dtype=np.float64)
            if self.preprocessing['transform']=='log1p_nonnegative_zscore_clip3':
                features=np.log1p(np.maximum(features,0))
            features=(features-np.asarray(self.preprocessing['mean']))/np.asarray(self.preprocessing['scale'])
            if self.preprocessing['transform']=='log1p_nonnegative_zscore_clip3':
                features=np.clip(features,-3,3)
            values=[1.0,*features.tolist()]
        vector=jnp.array(values)
        if not np.isfinite(np.asarray(vector)).all():
            raise ValueError('Topology covariates exceed numeric range')
        return vector

    def _genotype_vectors(self, patient_data: Mapping[str, Any]) -> tuple[list[int], list[int], bool]:
        pt_vec: list[int] = []
        mt_vec: list[int] = []
        seed = patient_data.get("Seeding", patient_data.get("has_metastasis"))
        if seed not in (0, 1):
            raise ValueError("Specify observed Seeding as 0 or 1; unknown seeding requires latent-state inference")
        has_metastasis = bool(seed)
        if not has_metastasis and any(value == 1 for name, value in patient_data.items()
                                      if name.startswith("M.") and name.endswith(" (M)")):
            raise ValueError("Seeding=0 conflicts with observed metastatic mutations")

        for gene in self.gene_names:
            p_val = patient_data.get(f"P.{gene} (M)", patient_data.get(f"P.{gene}", 0))
            m_val = patient_data.get(f"M.{gene} (M)", patient_data.get(f"M.{gene}", 0))
            if p_val not in (0, 1) or m_val not in (0, 1):
                raise ValueError('Observed genomic indicators must be binary')
            pt_vec.append(p_val)
            mt_vec.append(m_val)
            has_metastasis = has_metastasis or bool(m_val)
        return pt_vec, mt_vec, has_metastasis

    def _transition_log_rates(self, patient_data: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Compute accessible topological hazards for one patient."""

        if not self.topology_ready or self.topology_model is None:
            raise RuntimeError("Fitted topology components are required")

        features = self._feature_vector(patient_data)
        theta, _, _ = self.topology_model.compute_patient_params(
            self.W_theta,
            self.W_dp,
            self.W_dm,
            features,
            self.topology_model.log_rate_clip_min,
            self.topology_model.log_rate_clip_max,
        )
        theta_np = np.asarray(theta, dtype=np.float64)
        pt_vec, mt_vec, has_metastasis = self._genotype_vectors(patient_data)
        return accessible_transition_log_rates(theta_np, self.gene_names, pt_vec, mt_vec, has_metastasis)

    @staticmethod
    def _probability_diagnostics(prob_go: float, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Expose probability-mass and boundary diagnostics without changing model scores.

        The current artifacts do not contain a fitted post-hoc calibration layer, so raw
        model probabilities are intentionally preserved.  These diagnostics make that
        fact explicit and let the UI flag near-boundary inputs instead of silently
        presenting a saturated value as a precisely calibrated 100% prediction.
        """

        conditional_mass = float(sum(row["topology_probability"] for row in rows))
        integrated_mass = float(sum(row["absolute_risk"] for row in rows))
        max_conditional = float(max((row["topology_probability"] for row in rows), default=0.0))
        boundary_tolerance = 1e-12
        saturated = bool(
            prob_go <= boundary_tolerance
            or prob_go >= 1.0 - boundary_tolerance
            or max_conditional >= 1.0 - boundary_tolerance
        )
        return {
            "calibration": "not_applied",
            "conditional_mass": conditional_mass,
            "integrated_mass": integrated_mass,
            "max_conditional_probability": max_conditional,
            "saturated": saturated,
            "boundary_tolerance": boundary_tolerance,
        }

    def predict(self, patient_data: Mapping[str, Any]) -> dict[str, Any]:
        """Return observation-time gate and conditional probabilities / integrated scores."""

        self._genotype_vectors(patient_data)
        prob_go = self.kinetic.predict_go_probability(patient_data)
        if not np.isfinite(prob_go) or not 0 <= prob_go <= 1:
            raise ValueError("Invalid progression propensity")
        prob_stop = 1.0 - prob_go
        rates = self._transition_log_rates(patient_data)
        probabilities = softmax([item["log_rate"] for item in rates]) if rates else []
        absolute_risks = [
            { "compartment":item["compartment"], "event":item["event"], "rate":float(np.exp(item["log_rate"])), "topology_probability": float(probability),
             "integrated_event_score": float(probability * prob_go), "absolute_risk": float(probability * prob_go)}
            for item, probability in zip(rates, probabilities)
        ]
        absolute_risks.sort(key=lambda item: item["absolute_risk"], reverse=True)
        return {
            "prob_go": prob_go,
            "prob_stop": prob_stop,
            "operational_stop_predicted": prob_stop > self.stop_threshold,
            "is_stable_predicted": prob_stop > self.stop_threshold,
            "score_semantics": "integrated ranking score; absolute_risk/is_stable_predicted are deprecated aliases, not absolute future risk/biological stasis",
            "has_metastasis": self._genotype_vectors(patient_data)[2],
            "absolute_risks": absolute_risks,
            "probability_diagnostics": self._probability_diagnostics(prob_go, absolute_risks),
        }

    def predict_topology(self, state, observed_patient, prob_go):
        """Evaluate hypothetical state with the original covariate vector."""
        original = self._feature_vector
        # Use a separate shallow engine view so no shared instance is mutated.
        import copy
        view = copy.copy(self)
        view._feature_vector = lambda _: original(observed_patient)
        rates = view._transition_log_rates(state)
        probabilities = softmax([r["log_rate"] for r in rates]) if rates else []
        rows = [{"compartment":r["compartment"], "event":r["event"], "rate":float(np.exp(r["log_rate"])), "topology_probability":float(p), "integrated_event_score":float(p*prob_go), "absolute_risk":float(p*prob_go)}
                for r,p in zip(rates,probabilities)]
        rows.sort(key=lambda r:r['topology_probability'],reverse=True)
        return {'prob_go':prob_go,'prob_stop':1-prob_go,'absolute_risks':rows,
                'probability_diagnostics':self._probability_diagnostics(prob_go, rows)}

    def transition_rates(self, patient_data):
        return [{"compartment":r["compartment"],"event":r["event"],"rate":float(np.exp(r["log_rate"]))}
                for r in self._transition_log_rates(patient_data)]
