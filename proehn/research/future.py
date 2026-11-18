"""Exact lineage-preserving biological future-set probabilities and truth replay."""

from itertools import permutations
import numpy as np


def biological_rates(theta, states):
    """Batch coupled PT/MT hazards; event codes P0..Pn-1,M0..Mn-1,seed."""
    theta = np.asarray(theta, float)
    x = np.atleast_2d(np.asarray(states, np.int8))
    n = len(theta) - 1
    if theta.shape != (n + 1, n + 1) or x.shape[1] != 2 * n + 1:
        raise ValueError("Incompatible parameter/state shapes")
    p = x[:, :n]
    m = x[:, n : 2 * n]
    seed = x[:, -1:]
    diag = np.diag(theta)[:n]
    rp = np.exp(diag + p @ theta[:n, :n].T - diag * p) * (1 - p)
    rm = np.exp(diag + m @ theta[:n, :n].T - diag * m + theta[:n, -1]) * seed * (1 - m)
    rs = np.exp(theta[-1, -1] + p @ theta[-1, :n])[:, None] * (1 - seed)
    return np.c_[rp, rm, rs]


def probabilities(theta, states):
    rates = biological_rates(theta, states)
    total = rates.sum(axis=1)
    if np.any(total <= 0):
        raise ValueError("A terminal state has no next biological event")
    return rates / total[:, None]


def advance(states, events, valid=None):
    x = np.atleast_2d(np.asarray(states, np.int8)).copy()
    ev = np.asarray(events, int)
    n = (x.shape[1] - 1) // 2
    ok = np.ones(len(x), bool) if valid is None else np.asarray(valid, bool)
    for j in range(n):
        ids = ok & (ev == j)
        before = ids & (x[:, -1] == 0)
        x[ids, j] = 1
        x[before, n + j] = 1
        x[ok & (ev == n + j), n + j] = 1
    ids = ok & (ev == 2 * n)
    x[ids, n : 2 * n] = x[ids, :n]
    x[ids, -1] = 1
    return x


def ordered_log_probability(theta, states, event_sequences):
    x = np.atleast_2d(np.asarray(states, np.int8)).copy()
    seq = np.asarray(event_sequences, int)
    if seq.ndim != 2 or len(seq) != len(x):
        raise ValueError("Aligned future sequences required")
    out = np.zeros(len(x))
    active = np.ones(len(x), bool)
    rows = np.arange(len(x))
    for j in range(seq.shape[1]):
        # Prefix/horizon design guarantees nonterminal paths. Invalid candidate
        # permutations retain zero probability without changing their states.
        rates = biological_rates(theta, x)
        den = rates.sum(axis=1)
        p = np.divide(rates[rows, seq[:, j]], den, out=np.zeros(len(x)), where=den > 0)
        valid = active & (p > 0)
        out[valid] += np.log(p[valid])
        out[~valid] = -np.inf
        x = advance(x, seq[:, j], valid)
        active = valid
    return out


def unordered_log_probability(theta, states, event_sequences):
    seq = np.asarray(event_sequences, int)
    if any(len(set(row)) != len(row) for row in seq):
        raise ValueError("Distinct acquisition identities required")
    logs = np.array(
        [
            ordered_log_probability(theta, states, seq[:, perm])
            for perm in permutations(range(seq.shape[1]))
        ]
    )
    maximum = logs.max(axis=0)
    if not np.isfinite(maximum).all():
        raise ValueError("Observed future set has zero support")
    out = maximum + np.log(np.exp(logs - maximum).sum(axis=0))
    if np.any(out > 1e-10):
        raise AssertionError("Future-set mass exceeds one")
    return out


def forecast_metrics(theta, states, sequences):
    p = probabilities(theta, states)
    seq = np.asarray(sequences, int)
    actual = p[np.arange(len(p)), seq[:, 0]]
    if np.any(actual <= 0):
        raise ValueError("True next event has zero support")
    greater = (p > actual[:, None] + 1e-12).sum(axis=1)
    tied = ((p > 0) & (np.abs(p - actual[:, None]) <= 1e-12)).sum(axis=1)
    harmonic = np.r_[0.0, np.cumsum(1 / np.arange(1, p.shape[1] + 1))]
    seq_log = ordered_log_probability(theta, states, seq)
    set_log = unordered_log_probability(theta, states, seq)
    if np.any(set_log < seq_log - 1e-10):
        raise AssertionError("Set probability below its observed order")
    result = {
        "unordered_future_set_nll": -set_log,
        "ordered_future_sequence_nll": -seq_log,
        "next_event_nll": -np.log(actual),
        "next_event_brier": 1 + (p * p).sum(axis=1) - 2 * actual,
        "next_event_mrr": (harmonic[greater + tied] - harmonic[greater]) / tied,
    }
    for k in [1, 3, 5]:
        result[f"next_event_recall_at_{k}"] = np.clip((k - greater) / tied, 0, 1)
    if not all(np.isfinite(v).all() for v in result.values()):
        raise FloatingPointError("Nonfinite forecast metric")
    return result


def replay_observation_histories(source, truth, per_context, seed):
    """Exact original RNG replay up to final diagnosis; no continuation draws."""
    n = 12
    rng = np.random.default_rng(seed)
    cases = []
    records = []
    for ci, z in enumerate([-1.0, 1.0]):
        theta, dp, dm = source.project(truth, z, n=n)
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
            history = []
            obs_after = []
            for step in range(2 * n + 3):
                pr = np.exp(
                    np.diag(theta)[:n] + theta[:n, :n] @ pt - np.diag(theta)[:n] * pt
                ) * (1 - pt)
                mr = (
                    np.exp(
                        np.diag(theta)[:n]
                        + theta[:n, :n] @ mt
                        - np.diag(theta)[:n] * mt
                        + theta[:n, -1]
                    )
                    * (1 - mt)
                    * seeded
                )
                sr = np.exp(theta[-1, -1] + theta[-1, :n] @ pt) * (1 - seeded)
                dpr = np.exp(dp[:n] @ pt + dp[-1] * seeded) * (not p_done)
                dmr = np.exp(dm[:n] @ mt + dm[-1]) * seeded * (kind == 3 and not m_done)
                rates = np.r_[pr, mr, sr, dpr, dmr]
                event = int(rng.choice(len(rates), p=rates / rates.sum()))
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
                if event <= 2 * n:
                    history.append(event)
                else:
                    obs_after.append([event, len(history)])
                if p_done and (kind == 4 or m_done):
                    break
            if not p_done or kind == 3 and not m_done:
                raise RuntimeError("Original observation history incomplete")
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
            cases.append(
                {
                    "context": z,
                    "context_index": ci,
                    "within_context_patient": i,
                    "kind": kind,
                    "original_record": [kind, np_, nm_, list(genes), z],
                    "events_before_final_observation": history,
                    "diagnosis_after_genetic_jumps": obs_after,
                    "last_state": np.r_[pt, mt, seeded].tolist(),
                }
            )
    expected = source.sample_observations(truth, per_context, seed)
    if records != expected:
        raise AssertionError("Heldout observation replay differs from frozen source")
    return cases


def complete_truth_histories(cases, theta_by_context, training_seed, namespace):
    for case in cases:
        x = np.array(case["last_state"], np.int8)[None, :]
        events = list(case["events_before_final_observation"])
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    namespace,
                    training_seed,
                    case["context_index"],
                    case["within_context_patient"],
                ]
            )
        )
        theta = theta_by_context[case["context"]]
        for _ in range(25):
            rates = biological_rates(theta, x)[0]
            if rates.sum() == 0:
                break
            event = int(rng.choice(len(rates), p=rates / rates.sum()))
            events.append(event)
            x = advance(x, [event])
        if biological_rates(theta, x).sum() != 0 or len(events) != len(set(events)):
            raise AssertionError("True biological continuation failed to terminate")
        case["complete_genetic_events"] = events
    return cases


def prefix_arrays(cases, prefix, horizon):
    states = np.zeros((len(cases), 25), np.int8)
    for j in range(prefix):
        states = advance(
            states, np.array([c["complete_genetic_events"][j] for c in cases])
        )
    suffix = np.array(
        [c["complete_genetic_events"][prefix : prefix + horizon] for c in cases], int
    )
    if suffix.shape != (len(cases), horizon):
        raise ValueError("Insufficient future path under prespecified horizon")
    return states, suffix
