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


def _lp_coupled_0(log_theta, log_d_p, log_d_m, state_joint):
    from .likelihood import _lp_coupled_0 as general
    return general(log_theta, log_d_p, log_d_m, state_joint, 1, 1)


def _lp_coupled_1(log_theta, log_d_p, log_d_m, state_joint):
    from .likelihood import _lp_coupled_1 as general
    return general(log_theta, log_d_p, log_d_m, state_joint, 1, 1)


def _lp_coupled_2(log_theta, log_d_p, log_d_m, state_joint):
    from .likelihood import _lp_coupled_2 as general
    return general(log_theta, log_d_p, log_d_m, state_joint, 1, 1)
