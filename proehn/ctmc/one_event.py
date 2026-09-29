"""Closed-form kernels for the one-event coupled CTMC edge case."""

from __future__ import annotations

import jax.numpy as jnp

from . import vanilla as mhn
from .vanilla import diagnosis_theta


def small_Q(log_theta: jnp.ndarray) -> jnp.ndarray:
    """Build the two-state generator used when the restricted state has size one."""

    base_r = jnp.diagonal(log_theta)
    b_r = jnp.exp(base_r[:-1])
    e_seed = jnp.exp(log_theta[:-1, -1]) + 1.0
    row1 = [-jnp.exp(base_r).sum(), 0.0]
    row2 = [jnp.exp(log_theta[-1, -1]), -jnp.sum(b_r * e_seed)]
    return jnp.array([row1, row2])


def R_i_inv_vec(
    log_theta: jnp.ndarray,
    x: jnp.ndarray,
    d_p_le: jnp.ndarray,
    d_m_le: jnp.ndarray,
    transpose: bool = False,
) -> jnp.ndarray:
    """Return (D - Q)^(-1) x for the one-event coupled system."""

    D = jnp.eye(2)
    D = D.at[-1, -1].set(d_p_le + d_m_le)
    R = D - small_Q(log_theta)
    b = x.copy()
    if not transpose:
        b = b.at[0].divide(R[0, 0])
        b = b.at[1].add(-(b[0] * R[1, 0]))
        b = b.at[1].divide(R[1, 1])
    else:
        b = b.at[1].divide(R[1, 1])
        b = b.at[0].add(-(b[1] * R[1, 0]))
        b = b.at[0].divide(R[0, 0])
    return b


def _lp_coupled_0(
    log_theta: jnp.ndarray,
    log_d_p: jnp.ndarray,
    log_d_m: jnp.ndarray,
    state_joint: jnp.ndarray,
) -> jnp.ndarray:
    """Paired likelihood with unknown diagnosis order; not exact synchrony."""

    p0 = jnp.zeros(2).at[0].set(1.0)
    d_m_le = jnp.exp(log_d_m[-1])
    d_p_le = jnp.exp(log_d_p[-1])
    pTh1_joint = R_i_inv_vec(log_theta, p0, d_p_le, d_m_le)
    pf_cond = pTh1_joint * jnp.array([0.0, d_p_le])
    mf_cond = pTh1_joint * jnp.array([0.0, d_m_le])

    met = jnp.append(state_joint[1::2], 1)
    pf_pTh2 = mhn.R_inv_vec(diagnosis_theta(log_theta, log_d_m), pf_cond, met, 1)

    prim = state_joint[0::2]
    theta_pt = diagnosis_theta(log_theta.at[:-1, -1].set(0.0), log_d_p)
    mf_pTh2 = mhn.R_inv_vec(theta_pt, mf_cond, prim, 1)
    return jnp.log(jnp.maximum(pf_pTh2[-1] + mf_pTh2[-1], 1e-50))


def _lp_coupled_1(
    log_theta: jnp.ndarray,
    log_d_p: jnp.ndarray,
    log_d_m: jnp.ndarray,
    state_joint: jnp.ndarray,
) -> jnp.ndarray:
    """Log-likelihood when primary tumor is observed before metastasis."""

    p0 = jnp.zeros(2).at[0].set(1.0)
    d_m_le = jnp.exp(log_d_m[-1])
    d_p_le = jnp.exp(log_d_p[-1])
    pTh1_joint = R_i_inv_vec(log_theta, p0, d_p_le, d_m_le)
    pTh1_cond = jnp.append(jnp.zeros(1), pTh1_joint[-1] * d_p_le)
    met = jnp.append(state_joint[1::2], 1)
    pTh2 = mhn.R_inv_vec(diagnosis_theta(log_theta, log_d_m), pTh1_cond, met, 1)
    return jnp.log(jnp.maximum(pTh2[-1], 1e-50))


def _lp_coupled_2(
    log_theta: jnp.ndarray,
    log_d_p: jnp.ndarray,
    log_d_m: jnp.ndarray,
    state_joint: jnp.ndarray,
) -> jnp.ndarray:
    """Log-likelihood when metastasis is observed before primary tumor."""

    p0 = jnp.zeros(2).at[0].set(1.0)
    d_m_le = jnp.exp(log_d_m[-1])
    d_p_le = jnp.exp(log_d_p[-1])
    pTh1_joint = R_i_inv_vec(log_theta, p0, d_p_le, d_m_le)
    pTh1_cond = jnp.append(jnp.zeros(1), pTh1_joint[-1] * d_m_le)
    prim = state_joint[0::2]
    theta_pt = diagnosis_theta(log_theta.at[:-1, -1].set(0.0), log_d_p)
    pTh2 = mhn.R_inv_vec(theta_pt, pTh1_cond, prim, 1)
    return jnp.log(jnp.maximum(pTh2[-1], 1e-50))
