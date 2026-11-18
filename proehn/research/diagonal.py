"""Prior-centered ICGC likelihood fits with diagonal context effects."""

import time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from proehn.training import _compiled_topology_functions
from proehn.preprocessing import calculate_marginal_rates
from .likelihood import records, training_buckets
from .preprocessing import icgc_transform, base_indices, diagonal_basis


def fit_diagonal(frame, genes, metadata, spec, *, start=None, progress=None):
    nf = len(spec["columns"])
    z = icgc_transform(frame, metadata, spec["columns"])
    rows = records(frame, genes, z)
    buckets, weights = training_buckets(rows)
    B = diagonal_basis(nf, spec["mode"])
    prior = np.zeros((nf + 1) * 195)
    rates = np.array(calculate_marginal_rates(buckets, 12, weights), copy=True)
    # The absolute seeding rate has no observed-negative frequency in type-4
    # records. A neutral log-rate anchor is explicit, rather than using the
    # ascertainment-conditioned near-one frequency of known metastatic samples.
    rates[-1] = 0.0
    prior[np.arange(13) * 14] = np.clip(rates, -4, 4)
    penalty = np.full(len(prior), spec["context_ridge"])
    penalty[base_indices(nf)] = spec["base_ridge"]
    penalty[np.arange(13) * 14] *= 0.1
    buckets_jax = [
        (
            kind,
            np_,
            nm_,
            jnp.asarray(g, dtype=jnp.int8),
            jnp.asarray(x, dtype=jnp.float64),
            jnp.asarray(w, dtype=jnp.float64),
        )
        for (kind, np_, nm_, g, x), w in zip(buckets, weights)
    ]
    loss_grad, _ = _compiled_topology_functions(12, nf, 0.0, 0.0, -5.0, 5.0, 0.0)
    first = np.zeros(B.shape[1])
    if start is not None:
        if len(start) == len(first):
            first[:] = start
        elif len(start) == 195:
            first[:195] = start
        else:
            raise ValueError(
                "Warm start must match reduced basis or the 195 intercepts"
            )

    # Penalization is a training objective, never a heldout scoring adjustment.
    def objective(beta):
        params = prior + B @ beta
        value = 0.0
        gradient = np.zeros(len(params))
        for bucket in buckets_jax:
            v, g = jax.device_get(loss_grad(jnp.asarray(params), *bucket))
            value += float(v)
            gradient += np.asarray(g)
        delta = params - prior
        value = value / len(frame) + np.sum(penalty * delta**2)
        gradient = gradient / len(frame) + 2 * penalty * delta
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite objective")
        return float(value), B.T @ gradient

    t = time.time()
    nit = [0]

    def callback(beta):
        nit[0] += 1
        if progress and nit[0] % 25 == 0:
            progress(beta, nit[0], time.time() - t)

    result = minimize(
        objective,
        first,
        jac=True,
        method="L-BFGS-B",
        callback=callback,
        bounds=[(-6.0, 6.0)] * len(first),
        options=dict(maxiter=1200, ftol=1e-8, gtol=1e-5),
    )
    if not result.success:
        raise RuntimeError(str(result.message))
    parameters = prior + B @ result.x
    return dict(
        parameters=parameters.tolist(),
        beta=result.x.tolist(),
        nf=nf,
        spec=spec,
        preprocessing=metadata,
        diagnostics=dict(
            converged=bool(result.success),
            iterations=int(result.nit),
            objective=float(result.fun),
            message=str(result.message),
            free_parameters=len(first),
            full_parameter_size=len(parameters),
            seconds=time.time() - t,
            max_abs_full_coefficient=float(np.abs(parameters).max()),
        ),
    )
