"""Executable eight-method snapshot comparison from Supplement S4.1.

All methods receive the same primary-if-available twelve-event projection.
The exact HyperTraPS adapter evaluates state visitation probabilities; the
official HyperTraPS-CT adapter runs its native finite-budget untimed fit.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from functools import partial
from itertools import combinations
import importlib.metadata
import importlib.util
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, softmax
from sklearn.metrics import roc_curve

MODELS = {
    "proehn": "ProEHN",
    "no_covariate": "No-covariate coupled model",
    "metmhn": "metMHN",
    "mhn": "MHN",
    "oncotree": "Oncotree",
    "hypertraps": "HyperTraPS exact adapter",
    "hypertraps_ct": "HyperTraPS-CT",
    "prevalence": "Training prevalence",
}
METRICS = ("AUROC", "AUPRC", "Spec90_inner", "MCC_inner", "Brier", "log_loss")
METMHN_COMMIT = "616708143620aa0dfbab5d55f06d0413af21fb69"


def state_indices(values, *, reverse=False):
    values = np.asarray(values, dtype=np.int32)
    powers = np.arange(values.shape[1])
    return values @ (1 << (powers[::-1] if reverse else powers))


def conditional_probabilities(probability, values, *, reverse=False):
    """Condition on all other eleven observed events; never predict time."""
    probability = np.asarray(probability, dtype=np.float64)
    values = np.asarray(values, dtype=np.int32)
    indices = state_indices(values, reverse=reverse)
    result = np.empty(values.shape, dtype=np.float64)
    for j in range(values.shape[1]):
        bit = 1 << (values.shape[1] - 1 - j if reverse else j)
        off = indices & ~bit
        on = off | bit
        denominator = probability[off] + probability[on]
        if np.any(denominator <= 0):
            raise FloatingPointError(
                "Unobserved conditioning state has zero probability support"
            )
        result[:, j] = probability[on] / denominator
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite comparator conditional probability")
    return result


def exact_hypertraps_objective(theta, counts, *, penalty=0.002):
    """Exact model-2 forward visitation and analytic adjoint over 2**n states.

    This is the stage-vectorized form of the executed exact_visit_objective C
    adapter: theta[regulator,target], diagonal basal rates, normalized rates
    among accessible events, mean observed-state NLL plus L2 penalty.
    """
    theta = np.asarray(theta, dtype=np.float64)
    n = int(round(np.sqrt(theta.size)))
    theta = theta.reshape(n, n)
    counts = np.asarray(counts, dtype=np.float64)
    if counts.shape != (1 << n,) or counts.sum() <= 0:
        raise ValueError("Nonempty state counts of length 2**n are required")
    indices = np.arange(1 << n)
    bits = 1 << np.arange(n)
    states = ((indices[:, None] >> np.arange(n)) & 1).astype(np.float64)
    stage = states.sum(axis=1).astype(int)
    log_rates = np.diag(theta)[None, :] + states @ theta
    log_rates[states.astype(bool)] = -np.inf
    probabilities = np.zeros_like(log_rates)
    probabilities[:-1] = softmax(log_rates[:-1], axis=1)
    visitation = np.zeros(1 << n)
    visitation[0] = 1.0
    next_states = indices[:, None] | bits
    for k in range(n):
        selected = np.flatnonzero(stage == k)
        np.add.at(
            visitation,
            next_states[selected].ravel(),
            (visitation[selected, None] * probabilities[selected]).ravel(),
        )
    if np.any(visitation[counts > 0] <= 0):
        raise FloatingPointError("HyperTraPS observation has zero probability")
    adjoint = np.zeros_like(visitation)
    observed = counts > 0
    value = -np.sum(counts[observed] * np.log(visitation[observed])) / counts.sum()
    adjoint[observed] = -counts[observed] / (counts.sum() * visitation[observed])
    gradient = np.zeros_like(theta)
    for k in range(n - 1, -1, -1):
        selected = np.flatnonzero(stage == k)
        p = probabilities[selected]
        downstream = adjoint[next_states[selected]]
        average = np.sum(p * downstream, axis=1)
        adjoint[selected] += average
        g = visitation[selected, None] * p * (downstream - average[:, None])
        gradient += states[selected].T @ g
        gradient[np.diag_indices(n)] += g.sum(axis=0)
    value += penalty * np.sum(theta**2)
    gradient += 2 * penalty * theta
    return float(value), gradient.ravel(), visitation


def fit_hypertraps(values):
    values = np.asarray(values, dtype=np.int32)
    n = values.shape[1]
    counts = np.bincount(state_indices(values), minlength=1 << n).astype(float)
    frequency = (values.sum(axis=0) + 0.5) / (len(values) + 1)
    initial = np.zeros((n, n))
    np.fill_diagonal(initial, np.log(frequency))
    initial -= np.diag(np.full(n, np.diag(initial).mean()))

    def objective(parameters):
        value, gradient, _ = exact_hypertraps_objective(parameters, counts)
        return value, gradient

    result = minimize(
        objective,
        initial.ravel(),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-4.0, 4.0)] * (n * n),
        options=dict(maxiter=1200, ftol=1e-9, gtol=1e-6),
    )
    value, gradient, visitation = exact_hypertraps_objective(result.x, counts)
    population = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).sum(axis=1)
    for k in range(n + 1):
        np.testing.assert_allclose(visitation[population == k].sum(), 1.0, atol=1e-10)
    return dict(
        status="COMPLETE" if result.success else "FIT_LIMIT",
        backend="HyperTraPS model-2 exact-observation dynamic-programming adapter",
        parameters=result.x.tolist(),
        probability=(visitation / (n + 1)).tolist(),
        diagnostics=dict(
            converged=bool(result.success),
            iterations=int(result.nit),
            objective=value,
            gradient_norm=float(np.linalg.norm(gradient)),
            message=str(result.message),
            penalty_L2=0.002,
            bounds=[-4.0, 4.0],
        ),
    )


def fit_mhn(values):
    from mhn.optimizers import cMHNOptimizer, Penalty

    optimizer = cMHNOptimizer()
    values = np.asarray(values, dtype=np.int32)
    optimizer.load_data_matrix(values)
    optimizer.set_penalty(Penalty.L2)
    result = optimizer.train(lam=0.002, maxit=1200, reltol=1e-6, round_result=False)
    states = (
        (np.arange(1 << values.shape[1])[:, None] >> np.arange(values.shape[1])) & 1
    ).astype(np.int32)
    probability = np.array(
        [result.compute_marginal_likelihood(state) for state in states]
    )
    np.testing.assert_allclose(probability.sum(), 1.0, atol=1e-10)
    return dict(
        status="COMPLETE",
        backend="Official mhn 1.2.3 cMHN",
        parameters=result.log_theta.tolist(),
        probability=probability.tolist(),
        diagnostics={key: str(value) for key, value in result.meta.items()},
    )


def _rscript(executable=None):
    resolved = shutil.which(executable or "Rscript")
    if resolved is None:
        raise RuntimeError("Rscript is required; install R and pass --rscript PATH")
    return resolved


def _run_r(method, destination, rscript=None, *arguments):
    script = Path(__file__).with_name("comparator_runtime.R")
    command = [
        _rscript(rscript),
        "--vanilla",
        str(script),
        method,
        str(Path(destination).resolve()),
        *map(str, arguments),
    ]
    with (Path(destination) / "native_stdout.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            command, cwd=destination, stdout=log, stderr=subprocess.STDOUT, check=True
        )


def fit_oncotree(values, genes, destination, *, rscript=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(values, columns=genes).to_csv(
        destination / "training.csv", index=False
    )
    if not all(
        (destination / name).exists()
        for name in ("fit.rds", "states.csv", "errors.csv")
    ):
        _run_r("oncotree", destination, rscript)
    raw = pd.read_csv(destination / "states.csv", float_precision="round_trip")
    probability = np.zeros(1 << len(genes))
    probability[state_indices(raw[list(genes)].to_numpy(int))] = raw.Probability
    if np.any(probability <= 0):
        raise FloatingPointError("Oncotree support-floor adapter produced zero support")
    np.testing.assert_allclose(probability.sum(), 1.0, atol=1e-8)
    return dict(
        status="COMPLETE",
        backend="Official Oncotree 0.3.5 fit with positive-support observation channel",
        probability=probability.tolist(),
        error_rates=pd.read_csv(destination / "errors.csv").iloc[0].to_dict(),
    )


def fit_hypertraps_ct(
    values, validation, genes, warm, destination, *, seed, rscript=None
):
    """Native SA candidates; validation conditional log loss chooses penalty."""
    candidates = []
    artifacts = {}
    for index, penalty in enumerate((0.0, 0.002, 0.02)):
        directory = Path(destination) / f"candidate{index}"
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(values, columns=genes).to_csv(
            directory / "training.csv", index=False
        )
        np.savetxt(
            directory / "initial.txt", np.asarray(warm["parameters"])[None], fmt="%.17g"
        )
        if not all(
            (directory / name).exists()
            for name in ("native.json", "states.csv", "fit.rds")
        ):
            _run_r("hypertraps_ct", directory, rscript, seed + index, penalty)
        raw = pd.read_csv(directory / "states.csv", float_precision="round_trip")
        if raw.State.astype(int).tolist() != list(range(1 << len(genes))):
            raise ValueError(
                "Official CT readout must contain every ordered binary state"
            )
        probability = raw.Probability.to_numpy(float)
        population = (
            (np.arange(len(probability))[:, None] >> np.arange(len(genes))) & 1
        ).sum(axis=1)
        for k in range(len(genes) + 1):
            np.testing.assert_allclose(
                probability[population == k].sum(), 1.0, atol=1e-10
            )
        p = np.clip(
            conditional_probabilities(probability, validation, reverse=True),
            1e-12,
            1 - 1e-12,
        )
        loss = float(-np.mean(validation * np.log(p) + (1 - validation) * np.log1p(-p)))
        native = json.loads((directory / "native.json").read_text(encoding="utf-8"))
        if native["seed"] != seed + index or native["nominal_penalty"] != penalty:
            raise ValueError("Cached CT native candidate seed or penalty mismatch")
        candidates.append(
            dict(
                candidate=index,
                nominal_penalty=penalty,
                effective_penalty=np.floor(len(values) * penalty) / len(values),
                validation_conditional_log_loss=loss,
                seed=seed + index,
            )
        )
        artifacts[index] = dict(
            status="COMPLETE",
            backend="Official hypertrapsct 0.99.0 native SA, untimed",
            probability=probability.tolist(),
            reverse_state_bits=True,
            parameters=native["parameters"],
            diagnostics=native,
        )
    selected = min(
        candidates,
        key=lambda row: (row["validation_conditional_log_loss"], row["candidate"]),
    )
    return dict(
        artifacts[selected["candidate"]],
        selected_candidate=selected["candidate"],
        candidates=candidates,
    )


def fit_official_metmhn(train, genes, warm, cohort):
    """Optimize all 195 shared coefficients using official transition kernels."""
    import jax
    import jax.numpy as jnp
    from .comparator_metmhn import bucket_loss_and_grad
    from .preprocessing import calculate_marginal_rates
    from .research.likelihood import records, training_buckets

    if not jax.config.jax_enable_x64:
        raise RuntimeError("Enable JAX_ENABLE_X64=1 for official metMHN fitting")

    rows = records(train, genes, np.ones((len(train), 1)))
    grouped = defaultdict(list)
    for kind, np_, nm_, genotype, _ in rows:
        grouped[kind, np_, nm_].append(genotype)
    buckets = []
    for (kind, np_, nm_), genotypes in sorted(grouped.items()):
        counts = list(Counter(genotypes).items())
        active = np_ + nm_ if kind == 3 else np_ + 1 if kind == 4 else nm_
        size = min(16, max(1, 2048 // 2**active))
        for first in range(0, len(counts), size):
            part = counts[first : first + size]
            padded = part + [part[0]] * (size - len(part))
            buckets.append(
                (
                    kind,
                    np_,
                    nm_,
                    jnp.asarray([g for g, _ in padded], dtype=jnp.int8),
                    jnp.asarray(
                        [
                            count if i < len(part) else 0
                            for i, (_, count) in enumerate(padded)
                        ],
                        dtype=jnp.float64,
                    ),
                )
            )
    prior = np.zeros(195)
    ridge = np.zeros(195)
    if cohort != "GENIE":
        native_buckets, weights = training_buckets(rows)
        rates = np.array(
            calculate_marginal_rates(native_buckets, 12, weights), copy=True
        )
        rates[-1] = 0.0
        prior[np.arange(13) * 14] = np.clip(rates, -4.0, 4.0)
        ridge[:] = warm["spec"]["base_ridge"]
        ridge[np.arange(13) * 14] *= 0.1
        initial = np.asarray(warm["beta"])
        bound = 6.0
    else:
        initial = np.asarray(warm["parameters"])
        bound = 4.0

    def objective(beta):
        parameters = jnp.asarray(prior + beta)
        value, gradient = 0.0, np.zeros(195)
        for bucket in buckets:
            v, g = jax.device_get(bucket_loss_and_grad(parameters, *bucket))
            value += float(v)
            gradient += np.asarray(g)
        value = value / len(train) + np.sum(ridge * beta**2)
        gradient = gradient / len(train) + 2 * ridge * beta
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite official metMHN objective")
        return value, gradient

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=[(-bound, bound)] * 195,
        options=dict(maxiter=1200, ftol=1e-8, gtol=1e-5),
    )
    if not result.success:
        raise RuntimeError(
            "Official metMHN optimization failed: " + str(result.message)
        )
    return dict(
        status="COMPLETE",
        backend="Official metMHN transition and adjoint kernels",
        parameters=(prior + result.x).tolist(),
        beta=result.x.tolist(),
        prior=prior.tolist(),
        ridge=ridge.tolist(),
        diagnostics=dict(
            converged=bool(result.success),
            iterations=int(result.nit),
            function_evaluations=int(result.nfev),
            objective=float(result.fun),
            gradient_norm=float(np.linalg.norm(result.jac)),
            message=str(result.message),
        ),
    )


def predict_official_metmhn(artifact, frame, genes):
    import jax
    import jax.numpy as jnp
    from .comparator_metmhn import row_loglik
    from .research.likelihood import records, compartment_masks

    if not jax.config.jax_enable_x64:
        raise RuntimeError("Enable JAX_ENABLE_X64=1 for official metMHN inference")

    @partial(jax.jit, static_argnums=(1, 2, 3))
    def evaluate(parameters, kind, np_, nm_, genotypes):
        return jax.vmap(
            lambda genotype: row_loglik(parameters, kind, np_, nm_, genotype)
        )(genotypes)

    rows = records(frame, genes, np.ones((len(frame), 1)))
    masked, _, ids = compartment_masks(rows, "PT")
    if len(ids) != len(frame):
        raise ValueError(
            "Snapshot adapter requires one observed primary projection per patient"
        )
    grouped = defaultdict(list)
    for i, (kind, np_, nm_, genotype, _) in enumerate(masked):
        grouped[kind, np_, nm_].append((i, genotype))
    logp = np.empty(len(masked))
    for (kind, np_, nm_), items in sorted(grouped.items()):
        active = np_ + nm_ if kind == 3 else np_ + 1 if kind == 4 else nm_
        size = min(32, max(1, 4096 // 2**active))
        for first in range(0, len(items), size):
            part = items[first : first + size]
            padded = part + [part[0]] * (size - len(part))
            values = np.asarray(
                evaluate(
                    jnp.asarray(artifact["parameters"]),
                    kind,
                    np_,
                    nm_,
                    jnp.asarray([g for _, g in padded], dtype=jnp.int8),
                )
            )
            logp[[i for i, _ in part]] = values[: len(part)]
    if not np.isfinite(logp).all():
        raise FloatingPointError("Nonfinite official metMHN masked likelihood")
    logp = logp.reshape(len(frame), len(genes), 2)
    return expit(logp[:, :, 1] - logp[:, :, 0])


def dependency_status(methods, *, rscript=None):
    """Report actual installations, rather than silently replacing a method."""
    status = {}
    requirements = set()
    if "mhn" in methods:
        requirements.add(("mhn", "1.2.3"))
    if "metmhn" in methods:
        requirements.add(("metmhn", None))
    for module, expected in sorted(requirements):
        try:
            if importlib.util.find_spec(module) is None:
                raise ImportError("module not installed")
            version = importlib.metadata.version(module)
            if expected is not None and version != expected:
                raise ImportError(f"requires {expected}, found {version}")
            details = dict(available=True, version=version)
            if module == "metmhn":
                importlib.import_module("metmhn.jx.likelihood")
                distribution = importlib.metadata.distribution(module)
                raw_origin = distribution.read_text("direct_url.json")
                origin = json.loads(raw_origin) if raw_origin else {}
                commit = origin.get("vcs_info", {}).get("commit_id")
                if commit != METMHN_COMMIT:
                    raise ImportError(
                        "Install the pinned metMHN source commit " + METMHN_COMMIT
                    )
                details["source_commit"] = commit
            else:
                importlib.import_module("mhn.optimizers")
            status[module] = details
        except (ImportError, importlib.metadata.PackageNotFoundError) as error:
            status[module] = dict(available=False, error=str(error))
    for method, package, version in [
        ("oncotree", "Oncotree", "0.3.5"),
        ("hypertraps_ct", "hypertrapsct", "0.99.0"),
    ]:
        if method not in methods:
            continue
        expression = (
            f"stopifnot(requireNamespace('{package}',quietly=TRUE));"
            f"stopifnot(as.character(utils::packageVersion('{package}'))=='{version}');"
            + (
                "stopifnot(requireNamespace('jsonlite',quietly=TRUE));"
                if method == "hypertraps_ct"
                else ""
            )
        )
        try:
            result = subprocess.run(
                [_rscript(rscript), "--vanilla", "-e", expression],
                capture_output=True,
                text=True,
                check=True,
            )
            status[package] = dict(available=True, version=version)
        except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
            message = (
                error.stderr
                if isinstance(error, subprocess.CalledProcessError)
                else str(error)
            )
            status[package] = dict(available=False, error=message.strip())
    return status


def specificity_cutoffs(y, p):
    result = np.empty(y.shape[1])
    for j in range(y.shape[1]):
        if len(np.unique(y[:, j])) < 2:
            result[j] = 0.5
            continue
        _, sensitivity, cutoffs = roc_curve(y[:, j], p[:, j], drop_intermediate=False)
        finite = np.isfinite(cutoffs)
        sensitivity, cutoffs = sensitivity[finite], cutoffs[finite]
        result[j] = cutoffs[np.flatnonzero(sensitivity >= 0.9 - 1e-12)[0]]
    return result


def published_mcc_cutoffs(y, p):
    """Preserve the canonical published-rule strict argmax tie semantics."""
    result = np.empty(y.shape[1])
    for j in range(y.shape[1]):
        if len(np.unique(y[:, j])) < 2:
            result[j] = 0.5
            continue
        fpr, tpr, cutoffs = roc_curve(y[:, j], p[:, j], drop_intermediate=False)
        finite = np.isfinite(cutoffs)
        fpr, tpr, cutoffs = fpr[finite], tpr[finite], cutoffs[finite]
        positive, negative = float(y[:, j].sum()), float((1 - y[:, j]).sum())
        tp, fp = tpr * positive, fpr * negative
        fn, tn = positive - tp, negative - fp
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        score = np.divide(
            tp * tn - fp * fn, denominator, out=np.zeros_like(tp), where=denominator > 0
        )
        result[j] = cutoffs[np.argmax(score)]
    return result


def prediction_matrix(table, ids, genes):
    if table.duplicated(["patient_id", "gene"]).any() or len(table) != len(ids) * len(
        genes
    ):
        raise ValueError(
            "Predictions must cover each patient and frozen gene exactly once"
        )
    return (
        table.pivot(index="patient_id", columns="gene", values="prediction")
        .loc[list(ids), list(genes)]
        .to_numpy(float)
    )


def weighted_metrics(y, p, hard_spec, hard_mcc, counts):
    """Recompute target metrics for patient resamples, preserving score ties."""
    order = np.argsort(p, kind="stable")
    sp, sy = p[order], y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sp) != 0) + 1]
    positive = np.add.reduceat(counts[:, order] * sy, starts, axis=1)
    negative = np.add.reduceat(counts[:, order] * (1 - sy), starts, axis=1)
    npos, nneg = positive.sum(axis=1), negative.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        auc = (positive * (np.cumsum(negative, axis=1) - 0.5 * negative)).sum(
            axis=1
        ) / (npos * nneg)
        pp, nn = positive[:, ::-1], negative[:, ::-1]
        tp, seen = np.cumsum(pp, axis=1), np.cumsum(pp + nn, axis=1)
        precision = np.divide(
            tp, seen, out=np.zeros_like(tp, dtype=float), where=seen > 0
        )
        ap = (pp * precision).sum(axis=1) / npos
        spec = counts @ ((1 - y) * (1 - hard_spec)) / nneg
        tp, fp = counts @ (y * hard_mcc), counts @ ((1 - y) * hard_mcc)
        fn, tn = npos - tp, nneg - fp
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.divide(
            tp * tn - fp * fn,
            denominator,
            out=np.zeros_like(tp, dtype=float),
            where=denominator > 0,
        )
        brier = counts @ ((p - y) ** 2) / (npos + nneg)
        q = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
        loss = counts @ (-y * np.log(q) - (1 - y) * np.log1p(-q)) / (npos + nneg)
    return np.array([auc, ap, spec, mcc, brier, loss])


def summarize_oof(
    frame, genes, predictions, output, *, bootstrap_replicates=1000, seed=20261112
):
    """Same patient resamples across methods; macro means retain all 12 genes."""
    output = Path(output)
    ids = frame.patient_id.tolist()
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        len(ids), np.full(len(ids), 1 / len(ids)), size=bootstrap_replicates
    ).astype(np.float64)
    rows, macro_rows = [], []
    method_draws = {}
    method_points = {}
    for model, group in predictions.groupby("model", sort=False):
        draws = np.empty((len(genes), len(METRICS), bootstrap_replicates))
        points = np.empty((len(genes), len(METRICS)))
        for j, gene in enumerate(genes):
            subset = group[group.gene.eq(gene)].set_index("patient_id").loc[ids]
            y, p = subset.label.to_numpy(), subset.prediction.to_numpy()
            hs, hm = subset.hard_spec90.to_numpy(), subset.hard_mcc.to_numpy()
            points[j] = weighted_metrics(y, p, hs, hm, np.ones((1, len(ids))))[:, 0]
            draws[j] = weighted_metrics(y, p, hs, hm, counts)
            for k, metric in enumerate(METRICS):
                finite = draws[j, k, np.isfinite(draws[j, k])]
                low, high = (
                    np.quantile(finite, [0.025, 0.975])
                    if len(finite)
                    else (np.nan, np.nan)
                )
                rows.append(
                    dict(
                        model=model,
                        gene=gene,
                        metric=metric,
                        value=points[j, k],
                        low=low,
                        high=high,
                        valid_bootstrap_draws=len(finite),
                        n_patients=len(y),
                        positives=int(y.sum()),
                        status=(
                            "EVALUATED"
                            if np.isfinite(points[j, k])
                            else "NOT_EVALUABLE"
                        ),
                    )
                )
        for k, metric in enumerate(METRICS):
            macro = draws[:, k].mean(axis=0)
            finite = macro[np.isfinite(macro)]
            low, high = (
                np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
            )
            macro_rows.append(
                dict(
                    model=model,
                    metric=metric,
                    value=points[:, k].mean(),
                    low=low,
                    high=high,
                    genes=len(genes),
                    valid_bootstrap_draws=len(finite),
                )
            )
        method_draws[model] = draws
        method_points[model] = points
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False, float_format="%.17g")
    pd.DataFrame(macro_rows).to_csv(
        output / "macro_metrics.csv", index=False, float_format="%.17g"
    )
    np.savez_compressed(
        output / "patient_paired_bootstrap.npz",
        models=np.asarray(list(method_draws)),
        genes=np.asarray(genes),
        metrics=np.asarray(METRICS),
        draws=np.stack(list(method_draws.values())),
        seed=seed,
    )
    paired_rows = []
    for first, second in combinations(method_draws, 2):
        differences = method_draws[first] - method_draws[second]
        for k, metric in enumerate(METRICS):
            macro = differences[:, k].mean(axis=0)
            finite = macro[np.isfinite(macro)]
            low, high = (
                np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
            )
            paired_rows.append(
                dict(
                    model=first,
                    comparator=second,
                    metric=metric,
                    difference=(
                        method_points[first][:, k] - method_points[second][:, k]
                    ).mean(),
                    low=low,
                    high=high,
                    valid_bootstrap_draws=len(finite),
                )
            )
    pd.DataFrame(paired_rows).to_csv(
        output / "macro_paired_comparisons.csv", index=False, float_format="%.17g"
    )
