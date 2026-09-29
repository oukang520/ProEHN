"""Feature-modulated evolutionary topology model for ProEHN."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jrp
from jax import vmap

from .ctmc import likelihood as ctmc_likelihood


@dataclass(frozen=True)
class TopologyParameterShapes:
    """Shapes of the flattened topology parameter vector."""

    n_features_with_bias: int
    n_events_with_seeding: int

    @property
    def theta_size(self) -> int:
        return self.n_features_with_bias * self.n_events_with_seeding**2

    @property
    def diagnosis_size(self) -> int:
        return self.n_features_with_bias * self.n_events_with_seeding

    @property
    def total_size(self) -> int:
        return self.theta_size + 2 * self.diagnosis_size


class ProEHNTopologyModel:
    """Feature-conditioned CTMC topology engine.

    The model parameterizes a patient-specific log-transition matrix as

        theta_i[j, l] = sum_k z_i[k] * W_theta[k, j, l],

    where z_i includes a leading intercept. Diagonal entries encode basal hazards,
    off-diagonal entries encode epistatic modulation by active events, and the final
    event index represents metastatic seeding.
    """

    def __init__(
        self,
        n_events: int,
        n_features: int,
        regularization_strength: float = 0.01,
        l1_ratio: float = 1.0,
        log_rate_clip_min: float = -20.0,
        log_rate_clip_max: float = 20.0,
        l2_floor: float = 0.0,
    ) -> None:
        self.n_events = int(n_events)
        self.n_total = self.n_events + 1
        self.n_features = int(n_features) + 1
        self.regularization_strength = float(regularization_strength)
        self.l1_ratio = float(l1_ratio)
        self.l2_floor = float(l2_floor)
        if not 0 <= self.l1_ratio <= 1 or regularization_strength < 0 or l2_floor < 0:
            raise ValueError("Invalid regularization coefficients")
        self.log_rate_clip_min = float(log_rate_clip_min)
        self.log_rate_clip_max = float(log_rate_clip_max)
        self.shapes = TopologyParameterShapes(self.n_features, self.n_total)

    def init_params(self, key: jax.Array, base_rates: jnp.ndarray | None = None) -> jnp.ndarray:
        """Initialize flattened parameters, optionally warm-starting basal rates."""

        key, k1, k2, k3 = jrp.split(key, 4)
        W_theta = jrp.normal(k1, (self.n_features, self.n_total, self.n_total)) * 0.01
        idx = jnp.diag_indices(self.n_total)
        if base_rates is not None:
            W_theta = W_theta.at[0, idx[0], idx[1]].set(base_rates)
        else:
            W_theta = W_theta.at[0, idx[0], idx[1]].set(-2.0)
        W_dp = jrp.normal(k2, (self.n_features, self.n_total)) * 0.01
        W_dm = jrp.normal(k3, (self.n_features, self.n_total)) * 0.01
        return jnp.concatenate([W_theta.flatten(), W_dp.flatten(), W_dm.flatten()])

    def parse_params(self, params_flat: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Split a flattened parameter vector into transition and diagnosis tensors."""

        params = jnp.array(params_flat)
        split1 = self.shapes.theta_size
        split2 = split1 + self.shapes.diagnosis_size
        W_theta = params[:split1].reshape(self.n_features, self.n_total, self.n_total)
        W_dp = params[split1:split2].reshape(self.n_features, self.n_total)
        W_dm = params[split2:].reshape(self.n_features, self.n_total)
        return W_theta, W_dp, W_dm

    @staticmethod
    def compute_patient_params(
        W_theta: jnp.ndarray,
        W_dp: jnp.ndarray,
        W_dm: jnp.ndarray,
        features_with_bias: jnp.ndarray,
        clip_min: float = -20.0,
        clip_max: float = 20.0,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Project covariates into patient-specific transition and diagnosis rates."""

        theta = jnp.einsum("k,kij->ij", features_with_bias, W_theta)
        log_dp = jnp.einsum("k,ki->i", features_with_bias, W_dp)
        log_dm = jnp.einsum("k,ki->i", features_with_bias, W_dm)
        return (
            jnp.clip(theta, clip_min, clip_max),
            jnp.clip(log_dp, clip_min, clip_max),
            jnp.clip(log_dm, clip_min, clip_max),
        )

    @staticmethod
    def _loss_type1(
        W_theta: jnp.ndarray,
        W_dp: jnp.ndarray,
        W_dm: jnp.ndarray,
        features: jnp.ndarray,
        state_p: jnp.ndarray,
        n_prim_static: int,
        clip_min: float,
        clip_max: float,
    ) -> jnp.ndarray:
        theta, log_dp, _ = ProEHNTopologyModel.compute_patient_params(W_theta, W_dp, W_dm, features, clip_min, clip_max)
        return ctmc_likelihood._lp_prim_obs(theta, log_dp, state_p, int(n_prim_static))

    @staticmethod
    def _loss_type2(
        W_theta: jnp.ndarray,
        W_dp: jnp.ndarray,
        W_dm: jnp.ndarray,
        features: jnp.ndarray,
        state_m_full: jnp.ndarray,
        n_met_static: int,
        clip_min: float,
        clip_max: float,
    ) -> jnp.ndarray:
        theta, log_dp, log_dm = ProEHNTopologyModel.compute_patient_params(W_theta, W_dp, W_dm, features, clip_min, clip_max)
        return ctmc_likelihood._lp_met_obs(theta, log_dp, log_dm, state_m_full, int(n_met_static))

    @staticmethod
    def _loss_type3(
        W_theta: jnp.ndarray,
        W_dp: jnp.ndarray,
        W_dm: jnp.ndarray,
        features: jnp.ndarray,
        state_joint: jnp.ndarray,
        diag_order: jnp.ndarray,
        n_prim_static: int,
        n_met_static: int,
        clip_min: float,
        clip_max: float,
    ) -> jnp.ndarray:
        theta, log_dp, log_dm = ProEHNTopologyModel.compute_patient_params(W_theta, W_dp, W_dm, features, clip_min, clip_max)
        np_s = int(n_prim_static)
        nm_s = int(n_met_static)

        def sync(_: None) -> jnp.ndarray:
            return ctmc_likelihood._lp_coupled_0(theta, log_dp, log_dm, state_joint, np_s, nm_s)

        def pt_first(_: None) -> jnp.ndarray:
            return ctmc_likelihood._lp_coupled_1(theta, log_dp, log_dm, state_joint, np_s, nm_s)

        def mt_first(_: None) -> jnp.ndarray:
            return ctmc_likelihood._lp_coupled_2(theta, log_dp, log_dm, state_joint, np_s, nm_s)

        return jax.lax.switch(diag_order.astype(int), [sync, pt_first, mt_first], None)

    def bucket_loss(
        self,
        params_flat: jnp.ndarray,
        bucket_type: int,
        n_primary: int,
        n_metastatic: int,
        bucket_genotypes: jnp.ndarray,
        bucket_features: jnp.ndarray,
    ) -> jnp.ndarray:
        """Negative log-likelihood for one homogeneous observation bucket."""

        W_theta, W_dp, W_dm = self.parse_params(params_flat)
        clip_min = self.log_rate_clip_min
        clip_max = self.log_rate_clip_max

        if bucket_type == 4:
            def unknown_seed(feats, genes):
                a = self._loss_type1(W_theta, W_dp, W_dm, feats, genes.at[-1].set(0), n_primary, clip_min, clip_max)
                b = self._loss_type1(W_theta, W_dp, W_dm, feats, genes.at[-1].set(1), n_primary+1, clip_min, clip_max)
                return jnp.logaddexp(a, b)
            lls = vmap(unknown_seed)(bucket_features, bucket_genotypes)
        elif bucket_type in (0, 1):
            lls = vmap(self._loss_type1, in_axes=(None, None, None, 0, 0, None, None, None))(
                W_theta, W_dp, W_dm, bucket_features, bucket_genotypes, n_primary, clip_min, clip_max
            )
        elif bucket_type == 2:
            lls = vmap(self._loss_type2, in_axes=(None, None, None, 0, 0, None, None, None))(
                W_theta, W_dp, W_dm, bucket_features, bucket_genotypes, n_metastatic, clip_min, clip_max
            )
        elif bucket_type == 3:
            joint_len = 2 * self.n_events + 1
            b_joint = bucket_genotypes[:, :joint_len]
            b_order = bucket_genotypes[:, -1]
            lls = vmap(self._loss_type3, in_axes=(None, None, None, 0, 0, 0, None, None, None, None))(
                W_theta,
                W_dp,
                W_dm,
                bucket_features,
                b_joint,
                b_order,
                n_primary,
                n_metastatic,
                clip_min,
                clip_max,
            )
        else:
            raise ValueError("Unsupported observation type")
        return -jnp.sum(lls)

    def regularization(self, params_flat: jnp.ndarray) -> jnp.ndarray:
        """Explicit elastic net plus optional ridge floor on feature dimensions.

        Basal intercept rates are left unpenalized, matching the current source-code
        intent that cohort-level rates should remain anchored while covariate effects
        are sparsified.
        """

        W_theta, W_dp, W_dm = self.parse_params(params_flat)
        l1 = jnp.sum(jnp.abs(W_theta[1:])) + jnp.sum(jnp.abs(W_dp[1:])) + jnp.sum(jnp.abs(W_dm[1:]))
        l2 = jnp.sum(W_theta[1:] ** 2) + jnp.sum(W_dp[1:] ** 2) + jnp.sum(W_dm[1:] ** 2)
        alpha = self.l1_ratio
        return self.regularization_strength * (alpha * l1 + (1.0 - alpha + self.l2_floor) * l2)
