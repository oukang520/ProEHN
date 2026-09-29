"""Observation likelihoods for the ProEHN topology engine."""

from __future__ import annotations

from functools import partial

import jax.numpy as jnp
from jax import jit, lax

from . import one_event
from . import vanilla as mhn
from .vanilla import (
    R_inv_vec,
    diag_scal_m,
    diag_scal_p,
    diagnosis_theta,
    kron_diag,
    kronvec,
    obs_states,
)


@partial(jit, static_argnames=["transpose", "state_size"])
def R_i_inv_vec(
    log_theta: jnp.ndarray,
    log_d_p: jnp.ndarray,
    log_d_m: jnp.ndarray,
    x: jnp.ndarray,
    state: jnp.ndarray,
    state_size: int,
    transpose: bool = False,
) -> jnp.ndarray:
    """Solve the coupled observation resolvent used for paired samples."""

    lidg = -1.0 / (
        kron_diag(log_theta=log_theta, state=state, n_state=state_size)
        - (diag_scal_p(log_d_p, state, jnp.ones_like(x)) + diag_scal_m(log_d_m, state, jnp.ones_like(x)))
    )
    y = lidg * x

    def body_fun(_: int, carry: jnp.ndarray) -> jnp.ndarray:
        return lidg * (kronvec(log_theta=log_theta, p=carry, state=state, diag=False, transpose=transpose) + x)

    return lax.fori_loop(0, state_size + 1, body_fun, y)


def cond_p_obs(
    pTh1_joint: jnp.ndarray,
    state_joint: jnp.ndarray,
    n_joint: int,
    n_single: int,
    pt_first: bool,
) -> jnp.ndarray:
    # Project every compatible first-observation state, including seed=0 for PT.
    x, _ = mhn._states(state_joint, n_joint)
    n = (state_joint.shape[0]-1)//2
    single_target = jnp.r_[state_joint[1:2*n:2], 1] if pt_first else state_joint[0::2]
    single_x = jnp.column_stack((x[:, 1:2*n:2] if pt_first else x[:, 0:2*n:2], x[:, -1]))
    active = jnp.where(single_target == 1, size=n_single)[0]
    indices = single_x[:, active] @ (1 << jnp.arange(n_single))
    compatible = obs_states(n_joint, state_joint, pt_first)
    return jnp.zeros(2**n_single).at[indices].add(pTh1_joint*compatible)


def _lp_prim_obs(log_theta: jnp.ndarray, log_d_p: jnp.ndarray, state_pt: jnp.ndarray, n_prim: int) -> jnp.ndarray:
    """Primary-only marginal log-likelihood."""

    log_theta_pt = diagnosis_theta(log_theta.at[:-1, -1].set(0.0), log_d_p)
    p0 = jnp.zeros(2**n_prim).at[0].set(1.0)
    pTh = mhn.R_inv_vec(log_theta_pt, p0, state_pt, n_prim, jnp.ones_like(p0))
    return jnp.log(jnp.maximum(pTh[-1], 1e-50))


def _lp_met_obs(
    log_theta: jnp.ndarray,
    log_d_pt: jnp.ndarray,
    log_d_mt: jnp.ndarray,
    state_mt: jnp.ndarray,
    n_met: int,
) -> jnp.ndarray:
    """Metastasis-only marginal log-likelihood."""

    p0 = jnp.zeros(2**n_met).at[0].set(1.0)
    d_p, d_m = mhn.scal_d_pt(log_d_pt, log_d_mt, state_mt, jnp.ones(2**n_met))
    d_rates = d_p + d_m
    pTh = mhn.R_inv_vec(log_theta, p0, state_mt, n_met, d_rates, False)
    return jnp.log(jnp.maximum(pTh[-1] * d_rates[-1], 1e-50))


def _paired_order_probability(log_theta, log_d_p, log_d_m, state_joint,
                              n_prim, n_met, pt_first, first_seeding=-1):
    """Two observation phases. PT sampling does not imply prior seeding.

    After first PT sampling the unsampled lineage continues until MT diagnosis;
    MT diagnosis is zero before seeding. After first MT sampling the remaining
    PT continues without a metastatic seed effect. No elapsed time is asserted.
    """
    n_joint = n_prim+n_met-1
    p0 = jnp.zeros(2**n_joint).at[0].set(1.)
    occupation = R_i_inv_vec(log_theta, log_d_p, log_d_m, p0, state_joint, n_joint)
    diagnosed = (diag_scal_p(log_d_p, state_joint, occupation) if pt_first
                 else diag_scal_m(log_d_m, state_joint, occupation))
    x, _ = mhn._states(state_joint, n_joint)
    diagnosed *= jnp.where(first_seeding < 0, True, x[:, -1] == first_seeding)
    ns = n_met if pt_first else n_prim
    start = cond_p_obs(diagnosed, state_joint, n_joint, ns, pt_first)
    n = (state_joint.shape[0]-1)//2
    target = jnp.r_[state_joint[1:2*n:2], 1] if pt_first else state_joint[0::2]
    theta = log_theta if pt_first else log_theta.at[:-1, -1].set(0.)
    rates = (diag_scal_m(log_d_m, target, jnp.ones(2**ns)) if pt_first
             else diag_scal_p(log_d_p, target, jnp.ones(2**ns)))
    end = R_inv_vec(theta, start, target, ns, rates)
    return end[-1]*rates[-1]


def _log_probability(value):
    return jnp.log(jnp.maximum(value, jnp.finfo(value.dtype).tiny))


def _lp_coupled_0(log_theta, log_d_p, log_d_m, state_joint, n_prim, n_met, first_seeding=-1):
    """Unknown-order likelihood = sum of PT-first and MT-first joint masses."""
    args = log_theta, log_d_p, log_d_m, state_joint, n_prim, n_met
    return _log_probability(_paired_order_probability(*args, True, first_seeding)
                            + _paired_order_probability(*args, False, first_seeding))


def _lp_coupled_1(log_theta, log_d_p, log_d_m, state_joint, n_prim, n_met, first_seeding=-1):
    return _log_probability(_paired_order_probability(log_theta, log_d_p, log_d_m,
                           state_joint, n_prim, n_met, True, first_seeding))


def _lp_coupled_2(log_theta, log_d_p, log_d_m, state_joint, n_prim, n_met, first_seeding=-1):
    return _log_probability(_paired_order_probability(log_theta, log_d_p, log_d_m,
                           state_joint, n_prim, n_met, False, first_seeding))
