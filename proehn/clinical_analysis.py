"""Patient-level clinical analyses using fixed calibrated Go predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logit, logsumexp
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def _patients(frame, required):
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.patient_id.isna().any() or frame.patient_id.duplicated().any():
        raise ValueError("One row per identified patient is required")


def _survival(frame):
    _patients(frame, ["patient_id", "duration", "event", "p_Go"])
    t = frame.duration.to_numpy(float)
    e = frame.event.to_numpy(float)
    g = frame.p_Go.to_numpy(float)
    if not np.isfinite(t).all() or np.any(t <= 0) or not np.isin(e, [0, 1]).all():
        raise ValueError(
            "Use eligible patients with positive source OS days and binary event status"
        )
    if not np.isfinite(g).all() or np.any((g < 0) | (g > 1)):
        raise ValueError("Calibrated Go probabilities must be in [0,1]")
    return t, e.astype(int), g


def _folds(frame):
    if "fold" not in frame or not frame.fold.isin(range(5)).all():
        raise ValueError("Known patient outer folds 0..4 are required")


def _standardize(reference, frame, columns):
    result = frame.copy()
    preprocessing = {}
    for column in columns:
        raw = reference[column].to_numpy(float)
        if np.isinf(raw).any():
            raise ValueError(f"Infinite covariate: {column}")
        fill = float(np.nanmedian(raw)) if np.isfinite(raw).any() else 0.0
        values = np.where(np.isnan(raw), fill, raw)
        mean, scale = float(values.mean()), float(values.std(ddof=0))
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        target = frame[column].to_numpy(float)
        if np.isinf(target).any():
            raise ValueError(f"Infinite covariate: {column}")
        result[column] = (np.where(np.isnan(target), fill, target) - mean) / scale
        preprocessing[column] = dict(fill=fill, mean=mean, scale=scale)
    return result, preprocessing


def projected_seeding_share(theta, acquired):
    """Equation S24: project the observed gene set to an unseeded PT state."""
    theta = np.clip(np.asarray(theta, float), -5, 5)
    acquired = np.asarray(acquired)
    if theta.ndim != 2 or theta.shape[0] != theta.shape[1]:
        raise ValueError("A square projected transition coefficient matrix is required")
    n = len(theta) - 1
    if acquired.shape != (n,) or not np.isin(acquired, [0, 1]).all():
        raise ValueError("Aligned binary gene acquisition state is required")
    active = np.flatnonzero(acquired)
    candidates = np.r_[np.flatnonzero(acquired == 0), n]
    logs = np.diag(theta)[candidates] + theta[np.ix_(candidates, active)].sum(axis=1)
    return float(np.exp(logs[-1] - logsumexp(logs)))


def fixed_capacity_triage(
    validation,
    test,
    capacities=(0.1, 0.2, 0.3, 0.4, 0.5),
    *,
    bootstrap=1000,
    seed=20261114,
):
    """Transfer validation quantiles; retain unknown test labels in capacity counts."""
    for frame in (validation, test):
        _patients(frame, ["patient_id", "p_Go", "Patient_Label"])
        if not np.isin(frame.Patient_Label, [-1, 0, 1]).all():
            raise ValueError("Clinical labels use Go=0, Stop=1, unknown=-1")
        if not np.isfinite(frame.p_Go).all() or not frame.p_Go.between(0, 1).all():
            raise ValueError("Invalid calibrated Go probabilities")
    if set(validation.patient_id) & set(test.patient_id):
        raise ValueError("Validation and test patients overlap")
    label = test.Patient_Label.to_numpy()
    n, all_go, unknown = len(test), int(np.sum(label == 0)), label == -1
    known = ~unknown
    prevalence = all_go / known.sum() if known.any() else np.nan
    rng = np.random.default_rng(seed)
    # Performance and triage share the executed known-outcome/all-patient RNG stream.
    rng.integers(0, int(known.sum()), size=(bootstrap, int(known.sum())))
    indices = rng.integers(0, n, size=(bootstrap, n))
    rows = []
    for capacity in capacities:
        if not 0 < capacity < 1:
            raise ValueError("Capacity must be between zero and one")
        cutoff = float(np.quantile(validation.p_Go, 1 - capacity, method="higher"))
        selected = test.p_Go.to_numpy() >= cutoff
        ns, gs, ss, us = [
            int(x)
            for x in (
                selected.sum(),
                np.sum(selected & (label == 0)),
                np.sum(selected & (label == 1)),
                np.sum(selected & unknown),
            )
        ]
        nu, gu, uu = n - ns, all_go - gs, int(unknown.sum()) - us
        proportion = gs / (gs + ss) if gs + ss else np.nan
        # Numerator and denominator share the same unknown-label assignment.
        lower = (gs / ns) / ((all_go + uu) / n) if ns and all_go + uu else np.nan
        upper = ((gs + us) / ns) / ((all_go + us) / n) if ns and all_go + us else np.nan
        row = dict(
            validation_capacity=capacity,
            threshold=cutoff,
            selected_all=ns,
            selected_Go=gs,
            selected_Stop=ss,
            selected_unknown=us,
            unselected_all=nu,
            unselected_Go=gu,
            unselected_unknown=uu,
            observed_selected_Go_proportion=proportion,
            observed_selected_vs_all_enrichment=(
                proportion / prevalence if prevalence > 0 else np.nan
            ),
            full_population_enrichment_lower=lower,
            full_population_enrichment_upper=upper,
        )
        selected_known = selected & known
        selected_go = selected & (label == 0)
        boot_selected = selected_known[indices].sum(axis=1)
        boot_go = selected_go[indices].sum(axis=1)
        boot_all_go = (label[indices] == 0).sum(axis=1)
        boot_known = known[indices].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (boot_go / boot_selected) / (boot_all_go / boot_known)
            boot_proportion = boot_go / boot_selected
        for name, values in [
            ("observed_selected_Go_proportion", boot_proportion),
            ("observed_selected_vs_all_enrichment", ratio),
        ]:
            finite = values[np.isfinite(values)]
            ci = (
                np.quantile(finite, [0.025, 0.975]) if len(finite) else [np.nan, np.nan]
            )
            row[name + "_low"], row[name + "_high"] = map(float, ci)
        rows.append(row)
    return pd.DataFrame(rows)


def gate_performance(frame, *, bootstrap=1000, seed=20261114):
    _patients(frame, ["patient_id", "Patient_Label", "p_Go"])
    if not frame.Patient_Label.isin([-1, 0, 1]).all():
        raise ValueError("Clinical labels use Go=0, Stop=1, unknown=-1")
    known = frame.Patient_Label.isin([0, 1])
    y = 1 - frame.loc[known, "Patient_Label"].to_numpy(int)
    p = frame.loc[known, "p_Go"].to_numpy(float)
    if (
        set(np.unique(y)) != {0, 1}
        or not np.isfinite(p).all()
        or np.any((p < 0) | (p > 1))
    ):
        raise ValueError("Both clinical classes and valid probabilities are required")

    def metrics(indices):
        a, b = y[indices], p[indices]
        return [
            roc_auc_score(a, b) if len(np.unique(a)) == 2 else np.nan,
            average_precision_score(a, b) if a.any() else np.nan,
            brier_score_loss(a, b),
            log_loss(a, b, labels=[0, 1]),
        ]

    estimate = metrics(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [metrics(rng.integers(0, len(y), len(y))) for _ in range(bootstrap)]
    )
    interval = np.nanquantile(draws, [0.025, 0.975], axis=0)
    return pd.DataFrame(
        dict(
            metric=["AUROC", "AUPRC", "Brier", "log_loss"],
            estimate=estimate,
            low=interval[0],
            high=interval[1],
            n=len(y),
            Go=int(y.sum()),
        )
    )


def genie_cox(
    validation,
    test,
    *,
    bootstrap=2000,
    permutations=10000,
    seed=20261109,
    km_threshold=0.4677606944372642,
):
    """Fit ridge-.01 Cox models in validation; evaluate their fixed test risks."""
    from lifelines import CoxPHFitter
    from lifelines.statistics import logrank_test
    from lifelines.utils import concordance_index

    columns = ["age", "burden", "vaf", "metastatic"]
    for frame in (validation, test):
        _patients(frame, ["patient_id", *columns, "duration", "event", "p_Go"])
        _survival(frame)
    if set(validation.patient_id) & set(test.patient_id):
        raise ValueError("Validation and test patients overlap")
    frames = []
    for source in (validation, test):
        frame = source.copy()
        frame["Go_logit"] = logit(np.clip(frame.p_Go, 1e-6, 1 - 1e-6))
        frames.append(frame)
    val, prep = _standardize(frames[0], frames[0], columns + ["Go_logit"])
    heldout, _ = _standardize(frames[0], frames[1], columns + ["Go_logit"])
    coefficients, risks = [], []
    for name, covariates in [
        ("Clinical Cox", columns),
        ("Clinical Cox + Go", columns + ["Go_logit"]),
    ]:
        model = CoxPHFitter(penalizer=0.01).fit(
            val[covariates + ["duration", "event"]],
            duration_col="duration",
            event_col="event",
        )
        coefficients.append(model.summary.reset_index().assign(model=name))
        risks.append(model.predict_log_partial_hazard(heldout[covariates]).to_numpy())
    times, events, go = _survival(test)

    def c(score, index=None):
        i = np.arange(len(times)) if index is None else index
        try:
            return float(concordance_index(times[i], -score[i], events[i]))
        except ZeroDivisionError:
            return np.nan

    baseline, extended = risks
    observed = c(extended) - c(baseline)
    rng = np.random.default_rng(seed)
    draws = np.array(
        [
            c(extended, i) - c(baseline, i)
            for i in (rng.integers(0, len(test), len(test)) for _ in range(bootstrap))
        ]
    )
    exceeded, go_exceeded = 0, 0
    for _ in range(permutations):
        swap = rng.random(len(test)) < 0.5
        difference = c(np.where(swap, baseline, extended)) - c(
            np.where(swap, extended, baseline)
        )
        exceeded += abs(difference) >= abs(observed)
        order = rng.permutation(len(test))
        association = concordance_index(times[order], -go, events[order])
        go_exceeded += abs(association - 0.5) >= abs(c(go) - 0.5)
    high = go >= km_threshold
    if not high.any() or high.all():
        raise ValueError("Both fixed-threshold KM groups must be populated")
    lr = logrank_test(
        times[high],
        times[~high],
        event_observed_A=events[high],
        event_observed_B=events[~high],
    )
    grouping = pd.DataFrame(
        dict(duration=times, event=events, high_Go=high.astype(int))
    )
    group_model = CoxPHFitter().fit(
        grouping, duration_col="duration", event_col="event"
    )
    summary = dict(
        n_validation=len(val),
        n_test=len(test),
        test_deaths=int(events.sum()),
        clinical_C=c(baseline),
        clinical_plus_Go_C=c(extended),
        Go_C=c(go),
        delta_C=observed,
        paired_delta_C_95_CI=np.nanquantile(draws, [0.025, 0.975]).tolist(),
        paired_prediction_swap_p=(exceeded + 1) / (permutations + 1),
        Go_outcome_permutation_p=(go_exceeded + 1) / (permutations + 1),
        fixed_Go_threshold=km_threshold,
        logrank_p=float(lr.p_value),
        high_Go_n=int(high.sum()),
        low_Go_n=int((~high).sum()),
        bootstrap_replicates=bootstrap,
        permutations=permutations,
        preprocessing=prep,
    )
    prediction = test[["patient_id", "duration", "event", "p_Go"]].copy()
    prediction["clinical_risk"], prediction["clinical_plus_Go_risk"] = risks
    prediction["high_Go"] = high
    return (
        summary,
        pd.concat(coefficients, ignore_index=True),
        prediction,
        group_model.summary.reset_index(),
    )


def icgc_cox(frame, *, training_fold=0):
    """Pooled OOF diagnosis-origin association; covariate scaling uses folds !=0."""
    from lifelines import CoxPHFitter

    _patients(frame, ["patient_id", "fold", "age", "burden", "vaf", "p_Go"])
    _folds(frame)
    _survival(frame)
    frame = frame.copy()
    frame["Go_logit"] = logit(np.clip(frame.p_Go, 1e-6, 1 - 1e-6))
    columns = ["age", "burden", "vaf", "Go_logit"]
    frame, preprocessing = _standardize(
        frame[frame.fold != training_fold], frame, columns
    )
    model = CoxPHFitter(penalizer=0.1).fit(
        frame[columns + ["duration", "event", "fold"]],
        duration_col="duration",
        event_col="event",
        strata=["fold"],
    )
    return model.summary.reset_index(), preprocessing


def source_components(frame, *, bootstrap=1000, permutations=10000, seed=20261112):
    """Fixed Topology, Go and product scores; paired within-fold resampling."""
    from lifelines.utils import concordance_index

    _patients(frame, ["patient_id", "fold", "q_Seeding", "p_Go", "duration", "event"])
    _folds(frame)
    t, e, g = _survival(frame)
    q = frame.q_Seeding.to_numpy(float)
    if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError("Projected seeding probabilities must be in [0,1]")
    scores = np.column_stack([q, g, q * g])

    def c(indices, *, permuted_outcomes=False):
        ti, ei = t[indices], e[indices]
        si = scores if permuted_outcomes else scores[indices]
        try:
            return np.array([concordance_index(ti, -si[:, j], ei) for j in range(3)])
        except ZeroDivisionError:
            return np.full(3, np.nan)

    parts = [np.flatnonzero(frame.fold.to_numpy() == f) for f in np.unique(frame.fold)]
    observed = c(np.arange(len(frame)))
    rng = np.random.default_rng(seed)
    draws = np.array(
        [
            c(np.concatenate([rng.choice(p, len(p), replace=True) for p in parts]))
            for _ in range(bootstrap)
        ]
    )
    null = np.empty((permutations, 3))
    for b in range(permutations):
        indices = np.arange(len(frame))
        for part in parts:
            indices[part] = rng.permutation(part)
        null[b] = c(indices, permuted_outcomes=True)
    ci = np.nanquantile(draws, [0.025, 0.975], axis=0)
    table = pd.DataFrame(
        dict(
            model=["Topology", "Go", "Joint"],
            C_index=observed,
            low=ci[0],
            high=ci[1],
            outcome_permutation_p=(
                1 + (np.abs(null - 0.5) >= np.abs(observed - 0.5)).sum(axis=0)
            )
            / (permutations + 1),
            n=len(frame),
            deaths=int(e.sum()),
        )
    )
    paired = dict(
        Joint_minus_Topology=float(observed[2] - observed[0]),
        Joint_minus_Topology_95_CI=np.nanquantile(
            draws[:, 2] - draws[:, 0], [0.025, 0.975]
        ).tolist(),
        bootstrap_replicates=bootstrap,
        permutations=permutations,
    )
    return table, paired
