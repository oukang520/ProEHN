"""Independent observed-record development selection and paired uncertainty."""

from collections import OrderedDict
import numpy as np
from .likelihood import loglikelihoods


def strata(n):
    """Sampling allocation is fixed, including masked versus known order."""
    assert n % 4 == 0
    return np.tile(
        np.r_[np.zeros(n // 2, int), np.ones(n // 4, int), np.full(n // 4, 2, int)], 2
    ) + np.repeat([0, 3], n)


def paired_se(values, groups):
    v = np.asarray(values)
    return np.sqrt(
        sum(
            (np.sum(groups == g) / len(groups)) ** 2
            * np.var(v[groups == g], axis=0, ddof=1)
            / np.sum(groups == g)
            for g in np.unique(groups)
        )
    )


def candidate_set(base, full_by_init, weights):
    candidates = []
    seen = set()
    for init, full in sorted(full_by_init.items()):
        for a in weights:
            for b in weights:
                p = base.copy()
                p[:338] += a * (full[:338] - base[:338])
                p[338:] += b * (full[338:] - base[338:])
                key = p.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    dict(
                        id=f"i{init}_a{a:g}_b{b:g}",
                        initialization=init,
                        theta_weight=a,
                        clock_weight=b,
                        shrinkage_complexity=169 * a * a + 26 * b * b,
                        parameters=p,
                    )
                )
    assert 1 <= len(candidates) <= 49
    assert np.array_equal(candidates[0]["parameters"], base)
    return candidates


def compress_records(records):
    lookup = OrderedDict()
    inverse = []
    for kind, np_, nm_, genes, z in records:
        key = (kind, np_, nm_, tuple(genes), float(z))
        if key not in lookup:
            lookup[key] = len(lookup)
        inverse.append(lookup[key])
    rows = [(kind, np_, nm_, genes, (1.0, z)) for kind, np_, nm_, genes, z in lookup]
    return rows, np.array(inverse)


def losses(params, compressed):
    rows, inverse = compressed
    return -loglikelihoods(params, rows, 1)[inverse]


def choose(candidates, values, groups):
    """Inputs intentionally have no scenario, truth, latent paths or test data."""
    means = values.mean(axis=1)
    best = int(np.argmin(means))
    differences = values - values[best]
    standard_errors = np.array([paired_se(row, groups) for row in differences])
    eligible = means - means[best] <= standard_errors + 1e-12
    chosen = min(
        np.flatnonzero(eligible),
        key=lambda i: (
            candidates[i]["shrinkage_complexity"],
            means[i],
            candidates[i]["initialization"],
        ),
    )
    scores = []
    for i, c in enumerate(candidates):
        scores.append(
            {k: v for k, v in c.items() if k != "parameters"}
            | dict(
                dev_mean_nll=float(means[i]),
                excess_over_dev_min=float(means[i] - means[best]),
                paired_se_to_dev_min=float(standard_errors[i]),
                one_se_eligible=bool(eligible[i]),
                dev_minimum=(i == best),
                selected=(i == chosen),
            )
        )
    return int(chosen), scores


def bootstrap(values, groups, reps, seed):
    """Paired stratified multinomial counts are ordinary patient bootstrap."""
    rng = np.random.default_rng(seed)
    out = np.zeros((reps, values.shape[1]))
    for g in np.unique(groups):
        ids = np.flatnonzero(groups == g)
        v = values[ids]
        for start in range(0, reps, 32):
            n = min(32, reps - start)
            counts = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=n)
            out[start : start + n] += counts @ v / len(groups)
    return out


def order_losses(params, cases):
    paired = [c for c in cases if c["kind"] == 3]
    rows = []
    target = []
    for c in paired:
        kind, np_, nm_, genes, z = c["original_record"]
        for order in [1, 2]:
            rows.append((kind, np_, nm_, tuple(genes[:-1]) + (order,), (1.0, z)))
        target.append(0 if c["diagnosis_after_genetic_jumps"][0][0] == 25 else 1)
    # Genotypes and context define the same conditioning event for both orders.
    compressed_records = [(k, np_, nm_, g, z[1]) for k, np_, nm_, g, z in rows]
    ll = -losses(params, compress_records(compressed_records)).reshape(-1, 2)
    lp = ll - np.logaddexp(ll[:, 0], ll[:, 1])[:, None]
    y = np.array(target)
    p = np.exp(lp[:, 1])
    return (
        -lp[np.arange(len(lp)), y],
        (p - y) ** 2,
        np.array([i for i, c in enumerate(cases) if c["kind"] == 3]),
    )
