"""Official metMHN kernels with an explicit unknown-first-seeding observation adapter.

No ProEHN rate/likelihood kernel is imported. The adapter includes primary sampling
before seeding, because GENIE does not annotate seeding at the first observation.
An implicit adjoint changes the differentiation implementation, not the likelihood.
"""

from functools import partial
import jax
import jax.numpy as jnp
from metmhn.jx import vanilla as single
from metmhn.jx import kronvec as coupled
from metmhn.jx import likelihood as official_likelihood
from metmhn.jx import one_event


def states(target, size):
    active = jnp.where(target == 1, size=size)[0]
    bits = (jnp.arange(2**size)[:, None] >> jnp.arange(size)) & 1
    return jnp.zeros((2**size, len(target)), dtype=int).at[:, active].set(bits)


def qvec(theta, p, target, mode, transpose=False, diag=True):
    if len(p) == 1:
        return -jnp.sum(jnp.exp(jnp.diag(theta))) * p if diag else jnp.zeros_like(p)
    if mode and len(p) == 2:
        return one_event.kronvec(theta, p, diag, transpose)
    return (coupled if mode else single).kronvec(theta, p, target, diag, transpose)


def diagonal(theta, target, size, mode):
    if size == 0:
        return -jnp.sum(jnp.exp(jnp.diag(theta))) * jnp.ones(1)
    if mode and size == 1:
        return jnp.diag(one_event.small_Q(theta))
    return (
        coupled.kron_diag(theta, target, size)
        if mode
        else single.kron_diag(theta, target, jnp.ones(2**size))
    )


def raw_resolve(theta, x, target, diagnosis, size, mode, transpose):
    inverse = 1 / (diagnosis - diagonal(theta, target, size, mode))
    return jax.lax.fori_loop(
        0,
        size + 1,
        lambda _, y: inverse * (x + qvec(theta, y, target, mode, transpose, False)),
        inverse * x,
    )


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def resolve(theta, x, target, diagnosis, size, mode, transpose=False):
    return raw_resolve(theta, x, target, diagnosis, size, mode, transpose)


def forward(theta, x, target, diagnosis, size, mode, transpose):
    y = raw_resolve(theta, x, target, diagnosis, size, mode, transpose)
    return y, (theta, y, target, diagnosis)


def backward(size, mode, transpose, residual, cotangent):
    theta, y, target, d = residual
    adjoint = raw_resolve(theta, cotangent, target, d, size, mode, not transpose)
    left, right = (y, adjoint) if transpose else (adjoint, y)
    if size == 0:
        dt = jnp.diag(-jnp.sum(left * right) * jnp.exp(jnp.diag(theta)))
    elif mode and size == 1:
        _, pullback = jax.vjp(lambda t: one_event.small_Q(t) @ right, theta)
        dt = pullback(left)[0]
    elif mode:
        dt = official_likelihood.x_partial_Q_y(theta, left, right, target)
    else:
        dt = single.x_partial_Q_y(theta, left, right, target)[0]
    return dt, adjoint, None, -adjoint * y


resolve.defvjp(forward, backward)


def paired_probability(theta, dp, dm, target, np_, nm_, pt_first):
    size = np_ + nm_ - 1
    x = states(target, size)
    p0 = jnp.zeros(2**size).at[0].set(1.0)
    n = len(dp) - 1
    dp_rates = jnp.exp(jnp.c_[x[:, 0 : 2 * n : 2], x[:, -1]] @ dp)
    dm_rates = jnp.exp(jnp.c_[x[:, 1 : 2 * n : 2], x[:, -1]] @ dm) * x[:, -1]
    d = dp_rates + dm_rates
    occupation = resolve(theta, p0, target, d, size, 1)
    diagnosed = (dp_rates if pt_first else dm_rates) * occupation
    observed = x[:, 0 : 2 * n : 2] if pt_first else x[:, 1 : 2 * n : 2]
    final = target[0 : 2 * n : 2] if pt_first else target[1 : 2 * n : 2]
    compatible = jnp.all(observed == final, axis=1)
    remaining = jnp.c_[
        x[:, 1 : 2 * n : 2] if pt_first else x[:, 0 : 2 * n : 2], x[:, -1]
    ]
    single_target = jnp.r_[target[1 : 2 * n : 2], 1] if pt_first else target[0::2]
    ns = nm_ if pt_first else np_
    active = jnp.where(single_target == 1, size=ns)[0]
    index = remaining[:, active] @ (1 << jnp.arange(ns))
    start = jnp.zeros(2**ns).at[index].add(diagnosed * compatible)
    # Official diagnosis operators act on paired states; obtain single-state
    # hazards directly from their documented multiplicative parameterization.
    sx = states(single_target, ns)
    hazard = jnp.exp(sx @ (dm if pt_first else dp))
    hazard = hazard * sx[:, -1] if pt_first else hazard
    theta2 = theta if pt_first else theta.at[:-1, -1].set(0.0)
    end = resolve(theta2, start, single_target, hazard, ns, 0)
    return end[-1] * hazard[-1]


def row_loglik(parameters, kind, np_, nm_, g):
    theta = jnp.clip(parameters[:169].reshape(13, 13), -5.0, 5.0)
    dp = jnp.clip(parameters[169:182], -5.0, 5.0)
    dm = jnp.clip(parameters[182:], -5.0, 5.0)
    if kind == 4:

        def probability(seed):
            target = g.at[-1].set(seed)
            size = np_ + seed
            x = states(target, size)
            d = jnp.exp(x @ dp)
            p0 = jnp.zeros(2**size).at[0].set(1.0)
            return (
                resolve(theta.at[:-1, -1].set(0.0), p0, target, d, size, 0)[-1] * d[-1]
            )

        value = probability(0) + probability(1)
    elif kind == 2:
        p0 = jnp.zeros(2**nm_).at[0].set(1.0)
        a, b = single.scal_d_pt(dp, dm, g, jnp.ones_like(p0))
        d = a + b
        value = resolve(theta, p0, g, d, nm_, 0)[-1] * d[-1]
    elif kind == 3:
        target = g[:-2]
        value = paired_probability(
            theta, dp, dm, target, np_, nm_, True
        ) + paired_probability(theta, dp, dm, target, np_, nm_, False)
    else:
        raise ValueError(
            "Official validation supports the frozen GENIE observation design"
        )
    return jnp.log(value)


@partial(jax.jit, static_argnums=(1, 2, 3))
def bucket_loss(parameters, kind, np_, nm_, genotypes, weights):
    return -jnp.sum(
        jax.vmap(lambda g: row_loglik(parameters, kind, np_, nm_, g))(genotypes)
        * weights
    )


bucket_loss_and_grad = jax.jit(
    jax.value_and_grad(bucket_loss), static_argnums=(1, 2, 3)
)
