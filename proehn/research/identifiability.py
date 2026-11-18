"""Generating-point local sensitivity witness with all390 coefficients."""

from functools import partial
from itertools import combinations
import numpy as np
import jax
import jax.numpy as jnp
from proehn.topology import ProEHNTopologyModel
from .simulation import truth_parameters
from .likelihood import loglikelihoods

MODEL = ProEHNTopologyModel(12, 1, log_rate_clip_min=-5.0, log_rate_clip_max=5.0)


@partial(jax.jit, static_argnums=(1, 2, 3))
def row_scores(params, kind, np_, nm_, g, z):
    def one(a, b):
        return jax.value_and_grad(
            lambda p: -MODEL.bucket_loss(p, kind, np_, nm_, a[None, :], b[None, :])
        )(params)

    return jax.vmap(one)(g, z)


def observation_design():
    states = []
    for k in range(3):
        for active in combinations(range(12), k):
            x = np.zeros(12, int)
            x[list(active)] = 1
            states.append(tuple(x))
    grouped = {}
    allrows = []
    for z in [-1.0, 1.0]:
        for p in states:
            row = (4, sum(p), 0, p + (0,), (1.0, z))
            allrows.append(row)
            grouped.setdefault(row[:3], []).append((row, 0.25))
        for p in states:
            for m in states:
                for order in (1, 2):
                    g = tuple(v for pair in zip(p, m) for v in pair) + (1, -1, order)
                    row = (3, sum(p) + 1, sum(m) + 1, g, (1.0, z))
                    allrows.append(row)
                    grouped.setdefault(row[:3], []).append((row, 0.125))
    return grouped, allrows


def rank_witness(scenario, config):
    groups, allrows = observation_design()
    truth = truth_parameters(scenario)
    parts = []
    probability_mass = 0.0
    print("RANK_WITNESS", scenario, "observation atoms", len(allrows), flush=True)
    for (kind, np_, nm_), items in sorted(groups.items()):
        for first in range(0, len(items), 128):
            part = items[first : first + 128]
            size = len(part)
            padded = part + [part[0]] * (128 - size)
            ll, score = row_scores(
                jnp.asarray(truth),
                kind,
                np_,
                nm_,
                jnp.asarray([x[0][3] for x in padded], dtype=jnp.int8),
                jnp.asarray([x[0][4] for x in padded]),
            )
            ll = np.asarray(ll)[:size]
            score = np.asarray(score)[:size]
            weights = np.array([x[1] for x in part]) * np.exp(ll)
            if not np.isfinite(ll).all() or not np.isfinite(score).all():
                raise FloatingPointError("Nonfinite observation derivative")
            parts.append(np.sqrt(weights)[:, None] * score)
            probability_mass += weights.sum()
    jac = np.concatenate(parts)
    singular = np.linalg.svd(jac, compute_uv=False)
    norms = np.linalg.norm(jac, axis=0)
    scaled = np.linalg.svd(jac / np.where(norms > 0, norms, 1), compute_uv=False)
    rng = np.random.default_rng(config["directional_derivative_check"]["seed"])
    indices = rng.choice(
        len(allrows), config["directional_derivative_check"]["max_rows"], replace=False
    )
    subset = [allrows[i] for i in indices]
    epsilon = config["directional_derivative_check"]["epsilon"]
    max_error = 0.0
    # Score subset computed independently of weighted-rank matrix row ordering.
    for _ in range(config["directional_derivative_check"]["directions"]):
        direction = rng.normal(size=390)
        direction /= np.linalg.norm(direction)
        observed = (
            loglikelihoods(truth + epsilon * direction, subset, 1)
            - loglikelihoods(truth - epsilon * direction, subset, 1)
        ) / (2 * epsilon)
        exact = []
        for kind, np_, nm_, g, z in subset:
            _, gradient = row_scores(
                jnp.asarray(truth),
                kind,
                np_,
                nm_,
                jnp.asarray([g], dtype=jnp.int8),
                jnp.asarray([z]),
            )
            exact.append(float(np.asarray(gradient)[0] @ direction))
        max_error = max(max_error, float(np.max(np.abs(observed - exact))))
    if max_error > 1e-5:
        raise AssertionError(f"Directional derivative check failed: {max_error}")
    result = dict(
        status="COMPLETE",
        scenario=scenario,
        n_genes=12,
        parameters=390,
        observation_atoms=len(jac),
        partial_design_probability_mass=float(probability_mass),
        raw_rank={
            str(t): int(np.sum(singular > singular[0] * t))
            for t in config["relative_singular_thresholds"]
        },
        column_normalized_rank={
            str(t): int(np.sum(scaled > scaled[0] * t))
            for t in config["relative_singular_thresholds"]
        },
        largest_singular_value=float(singular[0]),
        smallest_singular_value=float(singular[-1]),
        condition_number=(
            float(singular[0] / singular[-1]) if singular[-1] > 0 else None
        ),
        directional_max_absolute_error=max_error,
        raw_singular_values=singular.tolist(),
        column_normalized_singular_values=scaled.tolist(),
    )
    return result
