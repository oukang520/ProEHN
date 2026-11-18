"""The same seven inner-validation MCC rules for every comparison method."""

import hashlib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_curve

EPS = 1e-12
POLICIES = [
    "published_gene_mcc",
    "stable_gene_mcc",
    "pooled_rare",
    "pooled_all",
    "rare_shrink50",
    "rare_shrink80",
    "posterior_expected_mcc",
]


def mcc(tp, fp, fn, tn):
    tp, fp, fn, tn = [np.asarray(x, dtype=np.float64) for x in [tp, fp, fn, tn]]
    assert min(float(np.min(x)) for x in [tp, fp, fn, tn]) >= -1e-7
    tp, fp, fn, tn = [np.maximum(x, 0.0) for x in [tp, fp, fn, tn]]
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.divide(
        tp * tn - fp * fn, denominator, out=np.zeros_like(tp), where=denominator > 0
    )


def gene_cutoff(y, p, stable):
    if len(np.unique(y)) < 2:
        return 1.0 + EPS if stable else 0.5
    fpr, tpr, cutoffs = roc_curve(y, p, drop_intermediate=False)
    if stable:
        cutoffs = np.where(np.isfinite(cutoffs), cutoffs, 1.0 + EPS)
    else:
        finite = np.isfinite(cutoffs)
        fpr, tpr, cutoffs = fpr[finite], tpr[finite], cutoffs[finite]
    positive, negative = float(y.sum()), float((1 - y).sum())
    tp, fp = tpr * positive, fpr * negative
    score = mcc(tp, fp, positive - tp, negative - fp)
    return float(cutoffs[np.flatnonzero(score >= score.max() - 1e-12)[0]])


def expected_cutoff(p):
    order = np.argsort(-p, kind="stable")
    sorted_p = p[order]
    endpoints = np.r_[np.flatnonzero(np.diff(sorted_p) != 0), len(p) - 1]
    k = endpoints + 1.0
    tp = np.cumsum(sorted_p)[endpoints]
    fp = k - tp
    positive = float(p.sum())
    fn, tn = positive - tp, len(p) - positive - fp
    score = mcc(tp, fp, fn, tn)
    score[endpoints == len(p) - 1] = 0.0
    score = np.r_[0.0, score]
    cutoffs = np.r_[1.0 + EPS, sorted_p[endpoints]]
    return float(cutoffs[np.flatnonzero(score >= score.max() - 1e-12)[0]])


def all_cutoffs(y, p, prior):
    legacy = np.array([gene_cutoff(y[:, j], p[:, j], False) for j in range(12)])
    stable = np.array([gene_cutoff(y[:, j], p[:, j], True) for j in range(12)])
    grid = expit(logit(prior)[None, :] + np.linspace(4.0, -4.0, 41)[:, None])
    hard = p[None, :, :] >= grid[:, None, :]
    tp = np.sum(hard * y[None, :, :], axis=1, dtype=np.float64)
    fp = np.sum(hard * (1 - y)[None, :, :], axis=1, dtype=np.float64)
    values = mcc(tp, fp, y.sum(axis=0)[None, :] - tp, (1 - y).sum(axis=0)[None, :] - fp)
    rare = prior < 0.15
    pooled = []
    for mask in [rare, np.ones(12, dtype=bool)]:
        result = stable.copy()
        if mask.any():
            scores = values[:, mask].mean(axis=1)
            best = np.flatnonzero(scores >= scores.max() - 1e-12)[0]
            result[mask] = grid[best, mask]
        pooled.append(result)
    shrunk = []
    for weight in [0.5, 0.8]:
        result = stable.copy()
        result[rare] = expit(
            (1 - weight) * logit(np.clip(stable[rare], EPS, 1 - EPS))
            + weight * logit(np.clip(pooled[0][rare], EPS, 1 - EPS))
        )
        shrunk.append(result)
    expected = np.array([expected_cutoff(p[:, j]) for j in range(12)])
    return np.stack(
        [legacy, stable, pooled[0], pooled[1], shrunk[0], shrunk[1], expected]
    )


def select_policy(y, p, prior, ids):
    """Pure function: consumes only inner validation data and training priors."""
    assert y.shape == p.shape and y.shape[1] == 12 and len(ids) == len(y)
    groups = np.array(
        [int(hashlib.sha256(str(pid).encode()).hexdigest()[:12], 16) % 5 for pid in ids]
    )
    decisions = np.zeros((len(POLICIES), *y.shape), dtype=np.int8)
    for fold in range(5):
        held = groups == fold
        assert held.any() and (~held).any()
        cutoffs = all_cutoffs(y[~held], p[~held], prior)
        decisions[:, held] = p[None, held] >= cutoffs[:, None, :]
    tp = np.sum(decisions * y[None], axis=1, dtype=np.float64)
    fp = np.sum(decisions * (1 - y)[None], axis=1, dtype=np.float64)
    fn = np.sum((1 - decisions) * y[None], axis=1, dtype=np.float64)
    tn = np.sum((1 - decisions) * (1 - y)[None], axis=1, dtype=np.float64)
    points = mcc(tp, fp, fn, tn).mean(axis=1)
    deleted = mcc(
        tp[:, None] - decisions * y[None],
        fp[:, None] - decisions * (1 - y)[None],
        fn[:, None] - (1 - decisions) * y[None],
        tn[:, None] - (1 - decisions) * (1 - y)[None],
    ).mean(axis=2)
    delta_deleted = deleted - deleted[0]
    se = np.sqrt(
        (len(y) - 1)
        / len(y)
        * np.sum((delta_deleted - delta_deleted.mean(axis=1)[:, None]) ** 2, axis=1)
    )
    gain = points - points[0]
    utility = gain - se
    ranking = sorted(range(len(POLICIES)), key=lambda i: (-utility[i], -points[i], i))
    chosen = ranking[0] if utility[ranking[0]] > 1e-12 else 0
    table = pd.DataFrame(
        [
            dict(
                policy=policy,
                inner_MCC=float(points[i]),
                gain_over_anchor=float(gain[i]),
                paired_patient_jackknife_se=float(se[i]),
                selection_utility=float(utility[i]),
                fixed_complexity_order=i,
            )
            for i, policy in enumerate(POLICIES)
        ]
    )
    thresholds = all_cutoffs(y, p, prior)[chosen]
    return dict(
        policy=POLICIES[chosen],
        policy_index=chosen,
        thresholds=thresholds,
        candidate_metrics=table,
        inner_crossfit_decisions=decisions,
        top_minus_runnerup=float(utility[ranking[0]] - utility[ranking[1]]),
    )
