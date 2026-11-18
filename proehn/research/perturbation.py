"""Exact weighted classification metrics for paired patient/seed perturbations."""

import numpy as np


def weighted(y, p, counts):
    order = np.argsort(p, kind="stable")
    sp, sy = p[order], y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sp) != 0) + 1]
    pos = np.add.reduceat(counts[:, order] * sy, starts, axis=1)
    neg = np.add.reduceat(counts[:, order] * (1 - sy), starts, axis=1)
    np_, nn = pos.sum(1), neg.sum(1)
    n = np_ + nn
    auc = (pos * (np.cumsum(neg, axis=1) - neg * 0.5)).sum(1) / (np_ * nn)
    pp, nt = pos[:, ::-1], neg[:, ::-1]
    tp = np.cumsum(pp, axis=1)
    seen = np.cumsum(pp + nt, axis=1)
    precision = np.divide(tp, seen, out=np.zeros_like(tp, dtype=float), where=seen > 0)
    ap = (pp * precision).sum(1) / np_
    brier = counts @ ((p - y) ** 2) / n
    q = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
    ll = counts @ (-y * np.log(q) - (1 - y) * np.log1p(-q)) / n
    return np.array([auc, ap, brier, ll])
