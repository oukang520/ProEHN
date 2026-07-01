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
        - (diag_scal_p(log_d_p, state, x) + diag_scal_m(log_d_m, state, x))
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
    compatible_states = obs_states(n_joint=n_joint, state=state_joint, pt_first=pt_first)
    poss_states_inds = jnp.where(compatible_states == 1.0, size=2 ** (n_single - 1))[0]
    pTh1_cond_obs = pTh1_joint[poss_states_inds]
    return jnp.append(jnp.zeros(2 ** (n_single - 1)), pTh1_cond_obs)


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


def _lp_coupled_0(
    log_theta: jnp.ndarray,
    log_d_p: jnp.ndarray,
    log_d_m: jnp.ndarray,
    state_joint: jnp.ndarray,
    n_prim: int,
    n_met: int,
) -> jnp.ndarray:
    """Paired-sample log-likelihood for simultaneous diagnosis."""

    if n_prim + n_met - 1 == 1:
        return one_event._lp_coupled_0(log_theta, log_d_p, log_d_m, state_joint)

    n_joint = n_prim + n_met - 1
    p0 = jnp.zeros(2**n_joint).at[0].set(1.0)
    pTh1_joint = R_i_inv_vec(log_theta, log_d_p, log_d_m, p0, state_joint, n_joint)
    pf_cond = cond_p_obs(diag_scal_p(log_d_p, state_joint, pTh1_joint), state_joint, n_joint, n_met, True)
    mf_cond = cond_p_obs(diag_scal_m(log_d_m, state_joint, pTh1_joint), state_joint, n_joint, n_prim, False)

    met = jnp.append(state_joint[1::2], 1)
    pf_pTh2 = R_inv_vec(diagnosis_theta(log_theta, log_d_m), pf_cond, met, n_met)

    prim = state_joint[0::2]
    theta_pt = diagnosis_theta(log_theta.at[:-1, -1].set(0.0), log_d_p)
    mf_pTh2 = R_inv_vec(theta_pt, mf_cond, prim, n_prim)
    return jnp.log(jnp.maximum(pf_pTh2[-1] + mf_pTh2[-1], 1e-50))


def _lp_coupled_1(
    log_theta: jnp.ndarray,
    log_d_p: jnp.ndarray,
    log_d_m: jnp.ndarray,
    state_joint: jnp.ndarray,
    n_prim: int,
    n_met: int,
) -> jnp.ndarray:
    """Paired-sample log-likelihood for primary-first diagnosis."""

    if n_prim + n_met - 1 == 1:
        return one_event._lp_coupled_1(log_theta, log_d_p, log_d_m, state_joint)

    joint_size = n_prim + n_met - 1
    p0 = jnp.zeros(2**joint_size).at[0].set(1.0)
    pTh1_joint = diag_scal_p(log_d_p, state_joint, R_i_inv_vec(log_theta, log_d_p, log_d_m, p0, state_joint, joint_size))
    compatible_states = obs_states(n_joint=joint_size, state=state_joint, pt_first=True)
    poss_states_inds = jnp.where(compatible_states == 1.0, size=2 ** (n_met - 1))[0]
    pTh1_cond = jnp.append(jnp.zeros(2 ** (n_met - 1)), pTh1_joint[poss_states_inds])
    met = jnp.append(state_joint[1::2], 1)
    pTh2 = R_inv_vec(diagnosis_theta(log_theta, log_d_m), pTh1_cond, met, n_met)
    return jnp.log(jnp.maximum(pTh2[-1], 1e-50))


def _lp_coupled_2(
    log_theta: jnp.ndarray,
    log_d_p: jnp.ndarray,
    log_d_m: jnp.ndarray,
    state_joint: jnp.ndarray,
    n_prim: int,
    n_met: int,
) -> jnp.ndarray:
    """Paired-sample log-likelihood for metastasis-first diagnosis."""

    if n_prim + n_met - 1 == 1:
        return one_event._lp_coupled_2(log_theta, log_d_p, log_d_m, state_joint)

    joint_size = n_prim + n_met - 1
    p0 = jnp.zeros(2**joint_size).at[0].set(1.0)
    pTh1_joint = diag_scal_m(log_d_m, state_joint, R_i_inv_vec(log_theta, log_d_p, log_d_m, p0, state_joint, joint_size))
    compatible_states = obs_states(joint_size, state_joint, False)
    poss_states_inds = jnp.where(compatible_states == 1.0, size=2 ** (n_prim - 1))[0]
    pTh1_cond = jnp.append(jnp.zeros(2 ** (n_prim - 1)), pTh1_joint[poss_states_inds])
    prim = state_joint[0::2]
    theta_pt = diagnosis_theta(log_theta.at[:-1, -1].set(0.0), log_d_p)
    pTh2 = R_inv_vec(theta_pt, pTh1_cond, prim, n_prim)
    return jnp.log(jnp.maximum(pTh2[-1], 1e-50))
