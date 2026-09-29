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
from .representation import accessible_transition_log_rates, patient_transition_representation_observed, patient_transition_representation


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
            p_val = patient_data.get(f"P.{gene} (M)", patient_data.get(f"P.{gene}"))
            m_val = patient_data.get(f"M.{gene} (M)", patient_data.get(f"M.{gene}"))
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
        integrated_mass = float(sum(row["integrated_event_score"] for row in rows))
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

    def _representation(self, patient_data, *, covariates=None):
        if not self.topology_ready:
            raise RuntimeError('Fitted topology components required')
        z = self._feature_vector(patient_data if covariates is None else covariates)
        if 'observation_type' not in patient_data:
            if patient_data.get('joint_snapshot') != 1:
                raise ValueError('Explicit observation_type or joint_snapshot=1 is required')
            pt, mt, seed = self._genotype_vectors(patient_data)
            return patient_transition_representation(self.topology_model, self.topology_params, z,
                self.gene_names, pt, mt, seed)
        kind = int(patient_data['observation_type'])
        def genotype(prefix):
            return [patient_data.get(f'{prefix}.{g} (M)', patient_data.get(f'{prefix}.{g}')) for g in self.gene_names]
        seed = patient_data.get('observed_seeding', patient_data.get('Seeding'))
        if seed is not None and (seed not in (0, 1) or (kind != 4 and seed != (0 if kind == 0 else 1))):
            raise ValueError('Seeding annotation conflicts with observation type')
        return patient_transition_representation_observed(self.topology_model, self.topology_params, z,
            self.gene_names, genotype('P') if kind != 2 else None,
            genotype('M') if kind in (2, 3) else None, kind,
            diagnosis_order=patient_data.get('diag_order'),
            first_seeding=patient_data.get('seeding_at_first_observation', -1),
            joint_snapshot=patient_data.get('joint_snapshot', False))

    def _result(self, representation, prob_go):
        integrated = representation.integrated_scores(prob_go)
        rows = [dict(compartment=key[0], event=key[1], topology_probability=float(p),
                     integrated_event_score=float(score))
                for key, p, score in zip(representation.accessible_events,
                    representation.conditional_probabilities, integrated)]
        rows.sort(key=lambda row: row['integrated_event_score'], reverse=True)
        diagnostics = self._probability_diagnostics(prob_go, rows)
        diagnostics['calibration'] = self.kinetic.probability_protocol.calibration
        return dict(prob_go=float(prob_go), prob_stop=float(1-prob_go),
            operational_stop_predicted=bool(1-prob_go > self.stop_threshold),
            integrated_event_scores=rows, probability_diagnostics=diagnostics,
            representation_kind=getattr(representation, 'representation_kind', 'fully_observed'),
            terminal_probability=getattr(representation, 'terminal_probability', float(not rows)),
            score_semantics='P(Go) times posterior-averaged conditional transition direction; no time horizon')

    def predict(self, patient_data: Mapping[str, Any]) -> dict[str, Any]:
        representation = self._representation(patient_data)
        return self._result(representation, self.kinetic.predict_go_probability(patient_data))

    def predict_topology(self, state, observed_patient, prob_go):
        """Evaluate hypothetical state at the original observation-time covariates."""
        return self._result(self._representation(state, covariates=observed_patient), prob_go)

    def transition_rates(self, patient_data):
        return [{"compartment":r["compartment"],"event":r["event"],"rate":float(np.exp(r["log_rate"]))}
                for r in self._transition_log_rates(patient_data)]
