"""Sparse CTMC operators for bipartite primary-metastatic state spaces."""

from __future__ import annotations

from functools import partial

import jax.numpy as jnp
from jax import jit, lax


def diagnosis_theta(log_theta: jnp.ndarray, log_diag_rates: jnp.ndarray) -> jnp.ndarray:
    """Scale off-diagonal transition effects by diagnosis rates."""

    diagonal = jnp.diagonal(log_theta)
    scaled_theta = log_theta - log_diag_rates[:, None]
    n = scaled_theta.shape[0]
    return scaled_theta.at[jnp.diag_indices(n)].set(diagonal)


@partial(jit, static_argnames=["n_joint", "pt_first"])
def obs_states(n_joint: int, state: jnp.ndarray, pt_first: bool) -> jnp.ndarray:
    """Mask states compatible with the observed first compartment."""

    mask = jnp.ones(1)
    active_indices = jnp.where(state == 1, size=n_joint)[0]
    n_genes = (state.shape[0] - 1) // 2
    vec_01 = jnp.array([0.0, 1.0])
    vec_11 = jnp.array([1.0, 1.0])

    for i in range(n_joint):
        idx = active_indices[i]
        is_pt_event = (idx % 2 == 0) & (idx < 2 * n_genes)
        is_mt_event = (idx % 2 == 1) & (idx < 2 * n_genes)
        operand = lax.select(is_pt_event if pt_first else is_mt_event, vec_01, vec_11)
        mask = jnp.kron(mask, operand)
    return mask


@partial(jit, static_argnames=["n_state"])
def kron_diag(log_theta: jnp.ndarray, state: jnp.ndarray, n_state: int) -> jnp.ndarray:
    """Compute the diagonal of the restricted generator Q."""

    n_total = log_theta.shape[0]
    n_genes = n_total - 1
    diag_vals = jnp.zeros(2**n_state)
    active_indices = jnp.where(state == 1, size=n_state)[0]

    for j in range(n_total):
        current_rate_j = jnp.full(2**n_state, log_theta[j, j])

        for bit in range(n_state):
            active_k = active_indices[bit]
            theta_k = jnp.where(active_k < 2 * n_genes, active_k // 2, n_genes)
            interaction = log_theta[j, theta_k]

            dim_low = 2**bit
            dim_high = 2**n_state // (2 * dim_low)
            r_reshaped = current_rate_j.reshape((dim_high, 2, dim_low))
            r_reshaped = r_reshaped.at[:, 1, :].add(interaction)
            current_rate_j = r_reshaped.flatten()

        rates_exp = jnp.exp(current_rate_j)

        for bit in range(n_state):
            idx = active_indices[bit]
            theta_idx = jnp.where(idx < 2 * n_genes, idx // 2, n_genes)
            is_target_bit = theta_idx == j

            dim_low = 2**bit
            dim_high = 2**n_state // (2 * dim_low)
            mask_reshaped = jnp.ones((dim_high, 2, dim_low))
            mask_reshaped = mask_reshaped.at[:, 1, :].set(0.0)
            rates_exp = lax.select(is_target_bit, rates_exp * mask_reshaped.flatten(), rates_exp)

        diag_vals -= rates_exp
    return diag_vals


@partial(jit, static_argnames=["diag", "transpose"])
def kronvec(
    log_theta: jnp.ndarray,
    p: jnp.ndarray,
    state: jnp.ndarray,
    diag: bool = True,
    transpose: bool = False,
) -> jnp.ndarray:
    """Multiply a restricted generator Q by a vector without materializing Q."""

    n_state = p.shape[0].bit_length() - 1
    n_total = log_theta.shape[0]
    n_genes = n_total - 1

    y = kron_diag(log_theta, state, n_state) * p if diag else jnp.zeros_like(p)
    active_indices = jnp.where(state == 1, size=n_state)[0]

    for bit in range(n_state):
        target_j_idx = active_indices[bit]
        theta_idx = jnp.where(target_j_idx == 2 * n_genes, n_genes, target_j_idx // 2)

        current_rate = jnp.full(2**n_state, log_theta[theta_idx, theta_idx])
        for k_bit in range(n_state):
            if k_bit == bit:
                continue
            reg_idx = active_indices[k_bit]
            reg_theta_idx = jnp.where(reg_idx == 2 * n_genes, n_genes, reg_idx // 2)
            interaction = log_theta[theta_idx, reg_theta_idx]

            dim_low = 2**k_bit
            dim_high = 2**n_state // (2 * dim_low)
            r_reshaped = current_rate.reshape((dim_high, 2, dim_low))
            r_reshaped = r_reshaped.at[:, 1, :].add(interaction)
            current_rate = r_reshaped.flatten()

        real_rate = jnp.exp(current_rate)
        dim_low = 2**bit
        dim_high = 2**n_state // (2 * dim_low)
        p_reshaped = p.reshape((dim_high, 2, dim_low))
        y_reshaped = y.reshape((dim_high, 2, dim_low))
        rate_reshaped = real_rate.reshape((dim_high, 2, dim_low))
        rate_at_source = rate_reshaped[:, 0, :]

        if transpose:
            y_reshaped = y_reshaped.at[:, 0, :].add(rate_at_source * p_reshaped[:, 1, :])
        else:
            y_reshaped = y_reshaped.at[:, 1, :].add(rate_at_source * p_reshaped[:, 0, :])
        y = y_reshaped.flatten()
    return y


@jit
def diag_scal_p(log_d_p: jnp.ndarray, state: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
    """Primary-tumor diagnosis rates on the restricted state space."""

    n = log_d_p.shape[0] - 1
    d_p = jnp.exp(log_d_p)
    rates = jnp.zeros_like(p)
    n_state = p.shape[0].bit_length() - 1
    active_indices = jnp.where(state == 1, size=n_state)[0]

    for bit in range(n_state):
        idx = active_indices[bit]
        is_pt = (idx % 2 == 0) | (idx == 2 * n)
        dim_low = 2**bit
        dim_high = 2**n_state // (2 * dim_low)
        r_reshaped = rates.reshape((dim_high, 2, dim_low))
        rate_idx = jnp.where(idx < 2 * n, idx // 2, n)
        r_reshaped = r_reshaped.at[:, 1, :].add(lax.select(is_pt, d_p[rate_idx], 0.0))
        rates = r_reshaped.flatten()
    return rates


@jit
def diag_scal_m(log_d_m: jnp.ndarray, state: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
    """Metastasis diagnosis rates on the restricted state space."""

    n = log_d_m.shape[0] - 1
    d_m = jnp.exp(log_d_m)
    rates = jnp.zeros_like(p)
    n_state = p.shape[0].bit_length() - 1
    active_indices = jnp.where(state == 1, size=n_state)[0]

    for bit in range(n_state):
        idx = active_indices[bit]
        is_mt = ((idx % 2 == 1) & (idx < 2 * n)) | (idx == 2 * n)
        dim_low = 2**bit
        dim_high = 2**n_state // (2 * dim_low)
        r_reshaped = rates.reshape((dim_high, 2, dim_low))
        rate_idx = jnp.where(idx < 2 * n, idx // 2, n)
        r_reshaped = r_reshaped.at[:, 1, :].add(lax.select(is_mt, d_m[rate_idx], 0.0))
        rates = r_reshaped.flatten()
    return rates


@jit
def scal_d_pt(
    log_d_p: jnp.ndarray,
    log_d_m: jnp.ndarray,
    state: jnp.ndarray,
    vec: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return diag_scal_p(log_d_p, state, vec), diag_scal_m(log_d_m, state, vec)


@partial(jit, static_argnames=["transpose", "n_state"])
def R_inv_vec(
    log_theta: jnp.ndarray,
    x: jnp.ndarray,
    state: jnp.ndarray,
    n_state: int,
    d_rates: jnp.ndarray = 1,
    transpose: bool = False,
) -> jnp.ndarray:
    """Fixed-point solver for (D - Q)^(-1) x on the restricted space."""

    lidg = -1 / (kron_diag(log_theta, state, n_state) - d_rates)
    y = lidg * x

    def body_fun(_: int, val: jnp.ndarray) -> jnp.ndarray:
        return lidg * (kronvec(log_theta, val, state, False, transpose) + x)

    return lax.fori_loop(0, n_state + 5, body_fun, y)
