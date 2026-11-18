"""Frozen higher-order future shift and nested false-negative prefix masks."""

import numpy as np
from .future import biological_rates, advance


def shifted_rates(theta, states, amplitude):
    rates = biological_rates(theta, states)
    n = 12
    j = np.arange(n)
    r1 = (j + 1) % n
    r2 = (j + 4) % n
    sign = np.where(j % 2 == 0, 1.0, -1.0)
    rates[:, :n] *= np.exp(amplitude * sign * states[:, r1] * states[:, r2])
    rates[:, n : 2 * n] *= np.exp(
        amplitude * sign * states[:, n + r1] * states[:, n + r2]
    )
    return rates


def draw_future(states, contexts, truth_thetas, amplitude, uniforms):
    x = states.copy()
    events = np.empty((len(x), 3), int)
    for step in range(3):
        rates = np.empty((len(x), 25))
        for z in [-1.0, 1.0]:
            ids = contexts == z
            rates[ids] = shifted_rates(truth_thetas[z], x[ids], amplitude)
        probabilities = rates / rates.sum(axis=1)[:, None]
        cdf = np.cumsum(probabilities, axis=1)
        cdf[:, -1] = 1.0
        event = (uniforms[:, step, None] >= cdf).sum(axis=1)
        assert np.all(rates[np.arange(len(x)), event] > 0)
        events[:, step] = event
        x = advance(x, event)
    assert all(len(set(row)) == 3 for row in events)
    return events


def drop_calls(states, rate, uniforms):
    out = states.copy()
    pre = states[:, -1] == 0
    post = ~pre
    for j in range(12):
        remove = pre & (uniforms[:, j] < rate)
        out[remove, j] = 0
        out[remove, 12 + j] = 0
        out[post & (uniforms[:, j] < rate), j] = 0
        out[post & (uniforms[:, 12 + j] < rate), 12 + j] = 0
    assert np.array_equal(out[:, -1], states[:, -1]) and np.all(out <= states)
    assert np.array_equal(out[pre, :12], out[pre, 12:24])
    return out


def validate():
    rng = np.random.default_rng(661)
    theta = rng.normal(0, 0.3, (13, 13))
    x = np.zeros((3, 25), np.int8)
    x[0, 1] = x[0, 4] = 1
    x[0, 13] = x[0, 16] = 1
    x[1, 2] = x[1, 5] = 1
    x[1, 14] = x[1, 17] = 1
    x[1, -1] = 1
    x[2, 0:24] = 1
    x[2, 0] = x[2, 12] = 0
    x[2, -1] = 1
    raw = biological_rates(theta, x)
    assert np.array_equal(shifted_rates(theta, x, 0), raw)
    changed = shifted_rates(theta, x, 1)
    assert np.isclose(changed[0, 0] / raw[0, 0], np.e)
    assert np.isclose(changed[1, 1] / raw[1, 1], np.exp(-1))
    assert np.array_equal(changed[:, -1], raw[:, -1])
    uniforms = rng.random((3, 24))
    assert np.array_equal(drop_calls(x, 0, uniforms), x)
    a = drop_calls(x, 0.1, uniforms)
    b = drop_calls(x, 0.3, uniforms)
    assert np.all(b <= a)
    zero = drop_calls(x, 1.0, uniforms)
    assert np.all(zero[:, :24] == 0)
    return {
        "status": "PASS",
        "checks": [
            "amplitude0 exact original hazards",
            "fixed positive/negative third-order factors",
            "seeding hazard unchanged",
            "dropout0 identity",
            "nested dropout with no false positives",
            "preseed PT/MT inheritance invariant",
            "known Seeding retained",
        ],
    }


def conditions(p):
    rows = []
    for family, spec in p["families"].items():
        for level in spec["levels"]:
            rows.append(
                {
                    "condition": family + "__" + format(level, "g").replace(".", "p"),
                    "family": family,
                    "level": level,
                    "dropout_rate": level if family == "prefix_false_negative" else 0.0,
                    "weight_context_plus": (
                        level if family == "context_sampling_shift" else 0.5
                    ),
                    "higher_order_amplitude": (
                        level if family == "higher_order_future_generator" else 0.0
                    ),
                }
            )
    return rows
