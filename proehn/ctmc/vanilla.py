"""Restricted, matrix-free CTMC operators with little-endian event bits.

Single-compartment states have n+1 entries; paired states interleave PT/MT
and end in seeding. Before seeding a mutation advances both copies; after
seeding the compartments evolve independently. Diagnosis hazards are
exp(sum(active log diagnosis effects)), with MT diagnosis gated by seeding.
This is the metMHN observation convention, not a sum of active hazards.
"""

from functools import partial
import jax.numpy as jnp
from jax import jit, lax, custom_vjp, vjp


def diagnosis_theta(log_theta, log_diag_rates):
    """Divide transition hazards by diagnosis: subtract regulator COLUMNS."""
    return (
        (log_theta - log_diag_rates[None, :])
        .at[jnp.diag_indices(log_theta.shape[0])]
        .set(jnp.diag(log_theta))
    )


def _states(state, size):
    active = jnp.where(state == 1, size=size)[0]
    bits = (jnp.arange(2**size)[:, None] >> jnp.arange(size)) & 1
    return (
        jnp.zeros((2**size, state.shape[0]), dtype=int).at[:, active].set(bits),
        active,
    )


def _transitions(theta, state, size):
    """Yield (source hazard, destination index, destination inside restriction)."""
    x, active = _states(state, size)
    weights = jnp.zeros(state.shape[0], dtype=int).at[active].set(1 << jnp.arange(size))
    source = jnp.arange(len(x))
    n = theta.shape[0] - 1
    if state.shape[0] == n + 1:
        for j in range(n + 1):
            regulators = x.at[:, j].set(0)
            rate = jnp.exp(theta[j, j] + regulators @ theta[j]) * (1 - x[:, j])
            yield rate, source + weights[j], state[j] == 1
    elif state.shape[0] == 2 * n + 1:
        pt, mt, seed = x[:, 0 : 2 * n : 2], x[:, 1 : 2 * n : 2], x[:, -1]
        for j in range(n):
            p_reg = pt.at[:, j].set(0)
            m_reg = mt.at[:, j].set(0)
            p_rate = jnp.exp(theta[j, j] + p_reg @ theta[j, :n])
            m_rate = jnp.exp(theta[j, j] + m_reg @ theta[j, :n] + theta[j, -1])
            yield p_rate * (1 - seed) * (1 - pt[:, j]) * (
                1 - mt[:, j]
            ), source + weights[2 * j] + weights[2 * j + 1], (
                state[2 * j] * state[2 * j + 1]
            ) == 1
            yield p_rate * seed * (1 - pt[:, j]), source + weights[2 * j], state[
                2 * j
            ] == 1
            yield m_rate * seed * (1 - mt[:, j]), source + weights[2 * j + 1], state[
                2 * j + 1
            ] == 1
        yield jnp.exp(theta[-1, -1] + pt @ theta[-1, :n]) * (
            1 - seed
        ), source + weights[-1], state[-1] == 1
    else:
        raise ValueError("State must have n+1 single or 2n+1 interleaved entries")


def _transition_arrays(theta, state, size):
    """Vectorized exact operators; retain all competing outside-state hazards."""
    x, active = _states(state, size)
    weights = jnp.zeros(state.shape[0], dtype=int).at[active].set(1 << jnp.arange(size))
    source = jnp.arange(len(x))[None, :]
    n = theta.shape[0] - 1
    if state.shape[0] == n + 1:
        diagonal = jnp.diag(theta)[:, None]
        rates = jnp.exp(diagonal + theta @ x.T - diagonal * x.T) * (1 - x.T)
        return rates, source + weights[:, None], state[:, None] == 1
    if state.shape[0] != 2 * n + 1:
        raise ValueError("State must have n+1 single or 2n+1 interleaved entries")
    pt, mt, seed = x[:, 0 : 2 * n : 2].T, x[:, 1 : 2 * n : 2].T, x[:, -1][None, :]
    diagonal = jnp.diag(theta)[:n, None]
    p = jnp.exp(diagonal + theta[:n, :n] @ pt - diagonal * pt)
    m = jnp.exp(diagonal + theta[:n, :n] @ mt - diagonal * mt + theta[:n, -1, None])
    rates = jnp.concatenate(
        (
            p * (1 - seed) * (1 - pt) * (1 - mt),
            p * seed * (1 - pt),
            m * seed * (1 - mt),
            jnp.exp(theta[-1, -1] + theta[-1, :n] @ pt)[None, :] * (1 - seed),
        ),
        axis=0,
    )
    destination = jnp.concatenate(
        (
            source + (weights[0 : 2 * n : 2] + weights[1 : 2 * n : 2])[:, None],
            source + weights[0 : 2 * n : 2, None],
            source + weights[1 : 2 * n : 2, None],
            source + weights[-1],
        ),
        axis=0,
    )
    inside = jnp.concatenate(
        (
            (state[0 : 2 * n : 2] * state[1 : 2 * n : 2]) == 1,
            state[0 : 2 * n : 2] == 1,
            state[1 : 2 * n : 2] == 1,
            state[-1:] == 1,
        )
    )[:, None]
    return rates, destination, inside


@partial(jit, static_argnames=["n_state"])
def kron_diag(log_theta, state, n_state):
    rates, _, _ = _transition_arrays(log_theta, state, n_state)
    return -jnp.sum(rates, axis=0)


@partial(jit, static_argnames=["diag", "transpose"])
def kronvec(log_theta, p, state, diag=True, transpose=False):
    size = p.shape[0].bit_length() - 1
    rates, destination, inside = _transition_arrays(log_theta, state, size)
    dest = jnp.minimum(destination, len(p) - 1)
    y = -jnp.sum(rates, axis=0) * p if diag else jnp.zeros_like(p)
    if transpose:
        return y + jnp.sum(rates * inside * p[dest], axis=0)
    return y.at[dest.ravel()].add((rates * inside * p[None, :]).ravel())


def _diagnosis(log_d, state, p, metastatic):
    x, _ = _states(state, p.shape[0].bit_length() - 1)
    n = log_d.shape[0] - 1
    if state.shape[0] == n + 1:
        cov = x
    else:
        cov = jnp.column_stack(
            (x[:, 1 : 2 * n : 2] if metastatic else x[:, 0 : 2 * n : 2], x[:, -1])
        )
    rate = jnp.exp(cov @ log_d)
    if metastatic:
        rate *= x[:, -1]
    return rate * p


@jit
def diag_scal_p(log_d_p, state, p):
    return _diagnosis(log_d_p, state, p, False)


@jit
def diag_scal_m(log_d_m, state, p):
    return _diagnosis(log_d_m, state, p, True)


@jit
def scal_d_pt(log_d_p, log_d_m, state, vec):
    # An MT-only lineage is diagnosed with PT hazard before seed, MT hazard after.
    x, _ = _states(state, vec.shape[0].bit_length() - 1)
    return diag_scal_p(log_d_p, state, vec) * (1 - x[:, -1]), diag_scal_m(
        log_d_m, state, vec
    )


@partial(jit, static_argnames=["n_joint", "pt_first"])
def obs_states(n_joint, state, pt_first):
    x, _ = _states(state, n_joint)
    n = (state.shape[0] - 1) // 2
    observed = x[:, 0 : 2 * n : 2] if pt_first else x[:, 1 : 2 * n : 2]
    target = state[0 : 2 * n : 2] if pt_first else state[1 : 2 * n : 2]
    return (
        jnp.all(observed == target, axis=1) & (True if pt_first else x[:, -1] == 1)
    ).astype(float)


def _resolvent_raw(log_theta, x, state, n_state, d_rates, transpose):
    """Exact finite triangular expansion, also used by the adjoint solve."""
    inverse_diag = 1 / (d_rates - kron_diag(log_theta, state, n_state))

    def step(_, y):
        return inverse_diag * (x + kronvec(log_theta, y, state, False, transpose))

    return lax.fori_loop(0, n_state + 1, step, inverse_diag * x)


@partial(custom_vjp, nondiff_argnums=(3, 5))
def _resolvent(log_theta, x, state, n_state, d_rates, transpose):
    return _resolvent_raw(log_theta, x, state, n_state, d_rates, transpose)


def _resolvent_fwd(log_theta, x, state, n_state, d_rates, transpose):
    y = _resolvent_raw(log_theta, x, state, n_state, d_rates, transpose)
    return y, (log_theta, y, state, d_rates)


def _resolvent_bwd(n_state, transpose, residual, cotangent):
    # A y=x, A=D-Q. The exact adjoint is A^-T cotangent. Differentiate
    # v^T(Q-D)y once instead of retaining every Neumann iterate's AD tape.
    theta, y, state, d_rates = residual
    adjoint = _resolvent_raw(theta, cotangent, state, n_state, d_rates, not transpose)
    _, pullback = vjp(
        lambda t, d: kronvec(t, y, state, True, transpose) - d * y, theta, d_rates
    )
    theta_grad, diagnosis_grad = pullback(adjoint)
    return theta_grad, adjoint, None, diagnosis_grad


_resolvent.defvjp(_resolvent_fwd, _resolvent_bwd)


@partial(jit, static_argnames=["transpose", "n_state"])
def R_inv_vec(log_theta, x, state, n_state, d_rates=1, transpose=False):
    """Exact resolvent with implicit adjoint; all competing hazards retained."""
    return _resolvent(
        log_theta, x, state, n_state, jnp.asarray(d_rates, dtype=x.dtype), transpose
    )
