"""Fixed-prediction calibration, hypothetical net benefit and unknown-label bounds."""

import numpy as np
import pandas as pd


def decision_curves(predictions, *, bootstrap=1000, seed=20261114):
    """Long patient/model rows; all models share patient bootstrap draws."""
    required = {"patient_id", "model", "label", "p_Go"}
    if required - set(predictions):
        raise ValueError(f"Missing columns: {sorted(required-set(predictions))}")
    if (
        predictions[["patient_id", "model"]].isna().any().any()
        or predictions.duplicated(["patient_id", "model"]).any()
    ):
        raise ValueError("Each patient/model pair must be present exactly once")
    models = sorted(predictions.model.unique(), key=str)
    if "ProEHN gate" in models:
        models.remove("ProEHN gate")
        models.insert(0, "ProEHN gate")
    ids = sorted(predictions.patient_id.unique(), key=str)
    tables = {
        m: predictions[predictions.model.eq(m)].set_index("patient_id").reindex(ids)
        for m in models
    }
    label = tables[models[0]].label.to_numpy()
    if not np.isin(label, [-1, 0, 1]).all():
        raise ValueError("Labels use Go=0, Stop=1, unknown=-1")
    known = label != -1
    y = 1 - label[known]
    n = int(known.sum())
    if not n:
        raise ValueError("At least one known clinical outcome is required")
    probs = {}
    for name, frame in tables.items():
        if not np.array_equal(frame.label.to_numpy(), label):
            raise ValueError("All models need the same patients and labels")
        p = frame.p_Go.to_numpy(float)
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Calibrated probabilities must be in [0,1]")
        probs[name] = p
    threshold = np.arange(5, 96) / 100
    odds = threshold / (1 - threshold)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(bootstrap, n))
    weights = np.array([np.bincount(i, minlength=n) for i in indices], float)
    contribution = {
        name: (p[known, None] >= threshold) * (y[:, None] - (1 - y[:, None]) * odds)
        for name, p in probs.items()
    }
    contribution["Act on all"] = y[:, None] - (1 - y[:, None]) * odds
    contribution["Act on none"] = np.zeros((n, len(threshold)))
    curves, draws = {}, {}
    rows = []
    for name, values in contribution.items():
        point, boot = values.mean(axis=0), weights @ values / n
        curves[name], draws[name] = point, boot
        lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
        for j, t in enumerate(threshold):
            rows.append(
                dict(
                    model=name,
                    threshold=t,
                    net_benefit=point[j],
                    low=lo[j],
                    high=hi[j],
                    n_known=n,
                )
            )
    curves["Best default"] = np.maximum(curves["Act on all"], 0)
    draws["Best default"] = np.maximum(draws["Act on all"], 0)
    paired = []
    for a_index, a in enumerate(models):
        for b in models[a_index + 1 :] + ["Best default"]:
            difference = curves[a] - curves[b]
            boot = draws[a] - draws[b]
            lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
            half = np.quantile(np.max(np.abs(boot - difference), axis=1), 0.95)
            for j, t in enumerate(threshold):
                paired.append(
                    dict(
                        model=a,
                        comparator=b,
                        threshold=t,
                        difference=difference[j],
                        low=lo[j],
                        high=hi[j],
                        uniform_low=difference[j] - half,
                        uniform_high=difference[j] + half,
                    )
                )
    bounds = []
    unknown = ~known
    for name, p in probs.items():
        select = p[:, None] >= threshold
        known_sum = (select[known] * (y[:, None] - (1 - y[:, None]) * odds)).sum(axis=0)
        count = select[unknown].sum(axis=0)
        for j, t in enumerate(threshold):
            bounds.append(
                dict(
                    model=name,
                    comparator="none",
                    kind="net_benefit",
                    threshold=t,
                    lower=(known_sum[j] - count[j] * odds[j]) / len(label),
                    upper=(known_sum[j] + count[j]) / len(label),
                    n_all=len(label),
                    unknown=int(unknown.sum()),
                )
            )
    for a_index, a in enumerate(models):
        for b in models[a_index + 1 :]:
            action = (probs[a][:, None] >= threshold).astype(int) - (
                probs[b][:, None] >= threshold
            ).astype(int)
            base = (action[known] * (y[:, None] - (1 - y[:, None]) * odds)).sum(axis=0)
            zero, one = -action[unknown] * odds, action[unknown]
            lo = (base + np.minimum(zero, one).sum(axis=0)) / len(label)
            hi = (base + np.maximum(zero, one).sum(axis=0)) / len(label)
            for j, t in enumerate(threshold):
                bounds.append(
                    dict(
                        model=a,
                        comparator=b,
                        kind="paired_net_benefit_difference",
                        threshold=t,
                        lower=lo[j],
                        upper=hi[j],
                        n_all=len(label),
                        unknown=int(unknown.sum()),
                    )
                )
    return {
        "net_benefit": pd.DataFrame(rows),
        "paired_net_benefit": pd.DataFrame(paired),
        "unknown_identification_bounds": pd.DataFrame(bounds),
    }


def reliability_bins(predictions):
    """Ten fixed bins, including empty bins; observed fractions use Wilson intervals."""
    from scipy.stats import norm

    z = norm.ppf(0.975)
    rows = []
    for model, frame in predictions.groupby("model", sort=False):
        frame = frame[frame.label.isin([0, 1])]
        p = frame.p_Go.to_numpy(float)
        y = 1 - frame.label.to_numpy(int)
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Invalid probabilities")
        bins = np.minimum((p * 10).astype(int), 9)
        for b in range(10):
            selected = bins == b
            n, k = int(selected.sum()), int(y[selected].sum())
            low = high = np.nan
            if n:
                rate = k / n
                denominator = 1 + z * z / n
                center = (rate + z * z / (2 * n)) / denominator
                half = (
                    z
                    * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n))
                    / denominator
                )
                low, high = center - half, center + half
            rows.append(
                dict(
                    model=model,
                    bin=b,
                    n=n,
                    predicted=float(p[selected].mean()) if n else np.nan,
                    observed=k / n if n else np.nan,
                    observed_low=low,
                    observed_high=high,
                )
            )
    return pd.DataFrame(rows)
