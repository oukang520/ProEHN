"""Identifiable shared means and orthogonal zero-sum mutation contrasts."""

import time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.linalg import helmert
from scipy.optimize import minimize
from proehn.training import _compiled_topology_functions
from proehn.preprocessing import calculate_marginal_rates
from .likelihood import records, training_buckets, loglikelihoods
from .preprocessing import icgc_fit_transform, icgc_transform, base_indices

SPECS = [
    dict(
        name="E05",
        evolution=True,
        observation=False,
        deviation_ridge=0.05,
        shared_ridge=0.005,
    ),
    dict(
        name="EO05",
        evolution=True,
        observation=True,
        deviation_ridge=0.05,
        shared_ridge=0.005,
    ),
    dict(
        name="EO20",
        evolution=True,
        observation=True,
        deviation_ridge=0.20,
        shared_ridge=0.005,
    ),
    dict(
        name="O",
        evolution=False,
        observation=True,
        deviation_ridge=0.0,
        shared_ridge=0.005,
    ),
]


def basis(spec, nf=2):
    size = (nf + 1) * 195
    columns = []
    penalties = []
    labels = []
    for j, index in enumerate(base_indices(nf)):
        v = np.zeros(size)
        v[index] = 1.0
        columns.append(v)
        penalties.append(0.0)
        labels.append("intercept_" + str(j))
    H = helmert(12, full=False).T
    for feature in range(1, nf + 1):
        if spec["evolution"]:
            idx = feature * 169 + np.arange(12) * 14
            v = np.zeros(size)
            v[idx] = 1.0
            columns.append(v)
            penalties.append(spec["shared_ridge"])
            labels.append(f"feature{feature}_mutation_mean")
            for j in range(11):
                v = np.zeros(size)
                v[idx] = H[:, j]
                columns.append(v)
                penalties.append(spec["deviation_ridge"])
                labels.append(f"feature{feature}_mutation_contrast{j}")
        if spec["observation"]:
            # One shared PT diagnosis effect per acquired mutation, no seed term.
            idx = (nf + 1) * 169 + feature * 13 + np.arange(12)
            v = np.zeros(size)
            v[idx] = 1.0
            columns.append(v)
            penalties.append(spec["shared_ridge"])
            labels.append(f"feature{feature}_PT_observation_mean")
    return np.stack(columns, axis=1), np.array(penalties), labels


def prepare(train, genes, spec, base_ridge):
    metadata = icgc_fit_transform(train)
    z = icgc_transform(train, metadata, [1, 2])
    rows = records(train, genes, z)
    buckets, weights = training_buckets(rows)
    B, penalty, labels = basis(spec)
    rates = np.array(calculate_marginal_rates(buckets, 12, weights), copy=True)
    rates[-1] = 0.0
    prior = np.zeros(585)
    prior[np.arange(13) * 14] = np.clip(rates, -4, 4)
    penalty[:195] = base_ridge
    penalty[np.arange(13) * 14] *= 0.1
    packed = [
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
    native_grad, _ = _compiled_topology_functions(12, 2, 0.0, 0.0, -5.0, 5.0, 0.0)

    def objective(beta):
        parameters = prior + B @ beta
        value = 0.0
        gradient = np.zeros(585)
        for bucket in packed:
            v, g = jax.device_get(native_grad(jnp.asarray(parameters), *bucket))
            value += float(v)
            gradient += np.asarray(g)
        value = value / len(train) + np.sum(penalty * beta**2)
        reduced = B.T @ (gradient / len(train)) + 2 * penalty * beta
        if not np.isfinite(value) or not np.isfinite(reduced).all():
            raise FloatingPointError("Nonfinite native generative objective")
        return float(value), reduced

    return metadata, B, penalty, labels, prior, objective


def lift_constant(parameters):
    result = np.zeros(585)
    result[base_indices(2)] = np.asarray(parameters)
    return result


def fit(train, genes, spec, baseline, legacy, progress=None):
    base_ridge = baseline["spec"]["base_ridge"]
    metadata, B, penalty, labels, prior, objective = prepare(
        train, genes, spec, base_ridge
    )
    candidates = {}
    for name, physical in [
        ("same_fold_constant", lift_constant(baseline["parameters"])),
        ("same_fold_legacy", np.asarray(legacy["parameters"])),
    ]:
        initial = np.clip(
            np.linalg.lstsq(B, np.asarray(physical) - prior, rcond=None)[0], -6, 6
        )
        candidates[name] = (initial, objective(initial)[0])
    initialization = min(candidates, key=lambda k: (candidates[k][1], k))
    initial = candidates[initialization][0]
    started = time.time()
    count = [0]
    trace = []

    def callback(beta):
        count[0] += 1
        if count[0] % 25 == 0:
            trace.append(
                dict(iteration=count[0], elapsed_seconds=time.time() - started)
            )
            if progress:
                progress(beta, count[0], time.time() - started)

    result = minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        bounds=[(-6.0, 6.0)] * len(initial),
        callback=callback,
        options=dict(maxiter=1200, ftol=1e-8, gtol=1e-5),
    )
    if not result.success:
        raise RuntimeError("Native CTMC fit failed: " + str(result.message))
    physical = prior + B @ result.x
    return dict(
        status="COMPLETE",
        parameters=physical.tolist(),
        beta=result.x.tolist(),
        nf=2,
        preprocessing=metadata,
        preprocessing_kind="ICGC_diagonal",
        spec=dict(
            spec, mode="native_partial_pool", columns=[1, 2], base_ridge=base_ridge
        ),
        representation="Original ProEHN full standard(flat theta,PTdiagnosis,MTdiagnosis) parameter artifact; no extra prediction head",
        reduced_basis=dict(
            version="mean-helmert-zero-sum-v1",
            labels=labels,
            coefficient_ridge=penalty.tolist(),
            common_mean_penalized_once=True,
            MT_context_fixed_zero=True,
            seeding_context_fixed_zero=True,
            PT_observation_seed_context_fixed_zero=True,
        ),
        initialization=dict(
            chosen=initialization,
            training_objectives={k: v[1] for k, v in candidates.items()},
        ),
        diagnostics=dict(
            converged=bool(result.success),
            iterations=int(result.nit),
            function_evaluations=int(result.nfev),
            objective=float(result.fun),
            gradient_norm=float(np.linalg.norm(result.jac)),
            message=str(result.message),
            free_parameters=len(initial),
            full_parameter_size=585,
            seconds=time.time() - started,
            max_abs_full_coefficient=float(np.abs(physical).max()),
        ),
        trace=trace,
    )


def scores(artifact, frame, genes):
    z = icgc_transform(frame, artifact["preprocessing"], artifact["spec"]["columns"])
    return loglikelihoods(
        np.asarray(artifact["parameters"]), records(frame, genes, z), artifact["nf"]
    )
