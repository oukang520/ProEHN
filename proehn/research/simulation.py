"""Frozen 12-gene generators, functional evaluation and coefficient recovery."""

from collections import defaultdict, Counter
import numpy as np
from .likelihood import loglikelihoods

SCENARIOS = ("no_context", "evolution_context", "observation_context", "joint_context")


def truth_parameters(scenario, n=12):
    if n != 12 or scenario not in SCENARIOS:
        raise ValueError("Frozen twelve-gene experiment")
    t = n + 1
    theta = np.zeros((2, t, t))
    dp = np.zeros((2, t))
    dm = np.zeros((2, t))
    theta[0, np.arange(t), np.arange(t)] = np.r_[np.linspace(-3.4, -2.4, n), -0.4]
    for j in range(n):
        theta[0, j, (j + 1) % n] = 0.30 * (-1 if j % 3 == 0 else 1)
        theta[0, j, -1] = 0.20 * (-1 if j % 2 else 1)
        theta[0, -1, j] = 0.12 * (-1 if j % 2 else 1)
    dp[0, :n] = 0.30
    dp[0, -1] = 0.40
    dm[0, :n] = 0.30
    dm[0, -1] = 0.90
    if scenario in ("evolution_context", "joint_context"):
        theta[1, np.arange(n), np.arange(n)] = np.array(
            [0.9, -0.8, 0.7, -0.65, 0.6, -0.55, 0.8, -0.7, 0.65, -0.6, 0.55, -0.5]
        )
        theta[1, -1, -1] = 0.7
        for j in range(n):
            theta[1, j, (j + 2) % n] = 0.18 * (-1 if j % 2 else 1)
    if scenario in ("observation_context", "joint_context"):
        dp[1, :n] = np.linspace(-0.1, 0.1, n)
        dp[1, -1] = 0.7
        dm[1, :n] = np.linspace(0.08, -0.08, n)
        dm[1, -1] = -0.6
    return np.r_[theta.ravel(), dp.ravel(), dm.ravel()]


def project(parameters, z, features=1, n=12):
    t = n + 1
    k = features + 1
    a = k * t * t
    b = a + k * t
    weights = np.array([1.0, z]) if features else np.ones(1)
    return tuple(
        np.clip(np.einsum("k,k...->...", weights, w), -5, 5)
        for w in (
            np.asarray(parameters[:a]).reshape(k, t, t),
            np.asarray(parameters[a:b]).reshape(k, t),
            np.asarray(parameters[b:]).reshape(k, t),
        )
    )


def sample_observations(parameters, per_context, seed, n=12):
    rng = np.random.default_rng(seed)
    records = []
    for z in (-1.0, 1.0):
        theta, dp, dm = project(parameters, z, n=n)
        for i in range(per_context):
            kind = 4 if i < per_context // 2 else 3
            mask_order = i >= 3 * per_context // 4
            pt = np.zeros(n, int)
            mt = np.zeros(n, int)
            seeded = 0
            p_done = False
            m_done = False
            observed_p = None
            observed_m = None
            first = 0
            for step in range(2 * n + 3):
                p_rate = np.exp(
                    np.diag(theta)[:n] + theta[:n, :n] @ pt - np.diag(theta)[:n] * pt
                ) * (1 - pt)
                m_rate = (
                    np.exp(
                        np.diag(theta)[:n]
                        + theta[:n, :n] @ mt
                        - np.diag(theta)[:n] * mt
                        + theta[:n, -1]
                    )
                    * (1 - mt)
                    * seeded
                )
                s_rate = np.exp(theta[-1, -1] + theta[-1, :n] @ pt) * (1 - seeded)
                d_p = np.exp(dp[:n] @ pt + dp[-1] * seeded) * (not p_done)
                d_m = np.exp(dm[:n] @ mt + dm[-1]) * seeded * (kind == 3 and not m_done)
                rates = np.r_[p_rate, m_rate, s_rate, d_p, d_m]
                event = rng.choice(len(rates), p=rates / rates.sum())
                if event < n:
                    pt[event] = 1
                    if not seeded:
                        mt[event] = 1
                elif event < 2 * n:
                    mt[event - n] = 1
                elif event == 2 * n:
                    seeded = 1
                    mt = pt.copy()
                elif event == 2 * n + 1:
                    observed_p = tuple(map(int, pt))
                    p_done = True
                    if not first:
                        first = 1
                else:
                    observed_m = tuple(map(int, mt))
                    m_done = True
                    if not first:
                        first = 2
                if p_done and (kind == 4 or m_done):
                    break
            if not p_done or kind == 3 and not m_done:
                raise RuntimeError("Observation sampler did not finish")
            if kind == 4:
                genes = (*observed_p, 0)
                np_, nm_ = sum(observed_p), 0
            else:
                genes = (
                    *[v for pair in zip(observed_p, observed_m) for v in pair],
                    1,
                    -1,
                    0 if mask_order else first,
                )
                np_, nm_ = sum(observed_p) + 1, sum(observed_m) + 1
            records.append((kind, np_, nm_, genes, z))
    return records


def buckets(records, features=1, batch_size=16):
    grouped = defaultdict(Counter)
    for kind, np_, nm_, genes, z in records:
        grouped[(kind, np_, nm_)][(genes, z)] += 1
    result = []
    weights = []
    for (kind, np_, nm_), counts in sorted(grouped.items()):
        items = list(counts.items())
        active = np_ + nm_ - 1 if kind == 3 else np_ + 1
        batch = min(batch_size, max(1, 32768 // (2**active)))
        for start in range(0, len(items), batch):
            part = items[start : start + batch]
            valid = len(part)
            data = [
                (k[0], np.array([1.0, k[1]]) if features else np.ones(1), v)
                for k, v in part
            ]
            data.extend([(data[0][0], data[0][1], 0)] * (batch - valid))
            result.append(
                (
                    kind,
                    np_,
                    nm_,
                    np.array([d[0] for d in data], np.int8),
                    np.array([d[1] for d in data]),
                )
            )
            weights.append(np.array([d[2] for d in data], float))
    return result, weights


def embed_constant(parameters):
    p = np.asarray(parameters)
    theta = p[:169].reshape(1, 13, 13)
    dp = p[169:182].reshape(1, 13)
    dm = p[182:].reshape(1, 13)
    return np.r_[
        np.r_[theta, np.zeros_like(theta)].ravel(),
        np.r_[dp, np.zeros_like(dp)].ravel(),
        np.r_[dm, np.zeros_like(dm)].ravel(),
    ]


def coefficient_groups():
    block = np.r_[
        np.repeat("evolution", 338),
        np.repeat("primary_observation", 26),
        np.repeat("metastasis_observation", 26),
    ]
    context = np.r_[
        np.repeat(["intercept", "context"], 169),
        np.repeat(["intercept", "context"], 13),
        np.repeat(["intercept", "context"], 13),
    ]
    return block, context


def precedence_probabilities(theta, uniforms):
    """Rate-based pure-birth sampler, separate from the observation CTMC sampler."""
    theta = np.asarray(theta).copy()
    theta[:12, 12] = 0.0
    n = len(theta)
    batch = len(uniforms)
    if uniforms.shape != (batch, n):
        raise ValueError("One uniform per patient and event required")
    rate = np.broadcast_to(np.exp(np.diag(theta)), (batch, n)).copy()
    position = np.empty((batch, n), int)
    index = np.arange(batch)
    for step in range(n):
        cumulative = np.cumsum(rate, axis=1)
        cut = uniforms[:, step] * cumulative[:, -1]
        event = np.minimum((cumulative <= cut[:, None]).sum(axis=1), n - 1)
        position[index, event] = step
        rate *= np.exp(theta[:, event].T)
        rate[index, event] = 0.0
    a, b = np.triu_indices(n, 1)
    return a, b, (position[:, a] < position[:, b]).mean(axis=0)


def evaluate(params, records, nf, probes):
    rows = [
        (kind, np_, nm_, genes, (1.0, z) if nf else (1.0,))
        for kind, np_, nm_, genes, z in records
    ]
    nll = float(-np.mean(loglikelihoods(params, rows, nf)))
    total = []
    bio = []
    for state, truth in probes:
        z, pt, mt, s, p_done, m_done, kind = state
        theta, dp, dm = project(params, z, features=nf)
        rates = event_rates(theta, dp, dm, pt, mt, s, p_done, m_done, kind)
        total.append(0.5 * np.abs(truth / truth.sum() - rates / rates.sum()).sum())
        if truth[:-2].sum() and rates[:-2].sum():
            bio.append(
                0.5
                * np.abs(
                    truth[:-2] / truth[:-2].sum() - rates[:-2] / rates[:-2].sum()
                ).sum()
            )
    return dict(
        heldout_nll=nll,
        competing_event_tv=float(np.mean(total)),
        biological_transition_tv=float(np.mean(bio)),
    )


def event_rates(theta, dp, dm, pt, mt, s, p_done, m_done, kind):
    p = np.exp(
        np.diag(theta)[:12] + theta[:12, :12] @ pt - np.diag(theta)[:12] * pt
    ) * (1 - pt)
    m = (
        np.exp(
            np.diag(theta)[:12]
            + theta[:12, :12] @ mt
            - np.diag(theta)[:12] * mt
            + theta[:12, -1]
        )
        * (1 - mt)
        * s
    )
    seed = np.exp(theta[-1, -1] + theta[-1, :12] @ pt) * (1 - s)
    return np.r_[
        p,
        m,
        seed,
        np.exp(dp[:12] @ pt + dp[-1] * s) * (not p_done),
        np.exp(dm[:12] @ mt + dm[-1]) * s * (kind == 3 and not m_done),
    ]


def state_probes(truth, seed, npatients=1000):
    rng = np.random.default_rng(seed)
    probes = []
    for i in range(npatients):
        z = (-1.0, 1.0)[i % 2]
        theta, dp, dm = project(truth, z)
        kind = 4 if i % 4 < 2 else 3
        pt = np.zeros(12, int)
        mt = pt.copy()
        s = 0
        pd = False
        md = False
        trajectory = []
        for step in range(27):
            rates = event_rates(theta, dp, dm, pt, mt, s, pd, md, kind)
            trajectory.append(
                ((z, pt.copy(), mt.copy(), s, pd, md, kind), rates.copy())
            )
            e = rng.choice(27, p=rates / rates.sum())
            if e < 12:
                pt[e] = 1
                if not s:
                    mt[e] = 1
            elif e < 24:
                mt[e - 12] = 1
            elif e == 24:
                s = 1
                mt = pt.copy()
            elif e == 25:
                pd = True
            else:
                md = True
            if pd and (kind == 4 or md):
                break
        probes.append(trajectory[rng.integers(len(trajectory))])
    return probes
