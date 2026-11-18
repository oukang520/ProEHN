"""Exact patient likelihood, grouping and compartment masking for 12 genes."""

from collections import defaultdict, Counter
from functools import lru_cache, partial
import numpy as np
import jax
import jax.numpy as jnp
from pandas import isna
from proehn.topology import ProEHNTopologyModel


def records(frame, genes, features):
    """Encode permitted tissue records; missing annotations retain unknown order."""
    names = {
        "primary_only": 4,
        "primary_seeding_unknown": 4,
        "primary_no_dissemination": 0,
        "primary_disseminated": 1,
        "metastasis_only": 2,
        "paired": 3,
    }
    if len(frame) != len(features):
        raise ValueError("One feature vector is required per patient")
    output = []
    for (_, r), x in zip(frame.iterrows(), features):
        raw = r.get("observation_type", None)
        kind = names.get(raw, raw)
        if kind not in (0, 1, 2, 3, 4):
            raise ValueError(
                "Explicit observation_type 0..4 or registered name is required"
            )
        kind = int(kind)
        observed = r.get("observed_seeding", np.nan)
        if not isna(observed):
            if observed not in (0, 1):
                raise ValueError("Independently observed seeding must be binary")
            if kind == 4:
                kind = int(observed)
            elif observed != (0 if kind == 0 else 1):
                raise ValueError("Observed seeding conflicts with observation_type")
        p = None if kind == 2 else tuple(r[f"P.{g}"] for g in genes)
        m = tuple(r[f"M.{g}"] for g in genes) if kind in (2, 3) else None
        for calls in (p, m):
            if calls is not None and not np.isin(calls, [0, 1]).all():
                raise ValueError("Observed genotypes must be binary and callable")
        p = None if p is None else tuple(map(int, p))
        m = None if m is None else tuple(map(int, m))
        if kind == 3:
            order = r.get("diag_order", 0)
            order = 0 if isna(order) else order
            seed = r.get("seeding_at_first_observation", -1)
            seed = -1 if isna(seed) else seed
            if (
                order not in (0, 1, 2)
                or seed not in (-1, 0, 1)
                or (order == 2 and seed == 0)
            ):
                raise ValueError("Invalid paired order/first-observation seeding")
            if seed == 0 and np.any(np.asarray(p) > np.asarray(m)):
                raise ValueError("Pre-seeding PT mutations must be inherited by MT")
            output.append(
                (
                    3,
                    sum(p) + 1,
                    sum(m) + 1,
                    tuple(v for pair in zip(p, m) for v in pair)
                    + (1, int(seed), int(order)),
                    tuple(x),
                )
            )
        elif kind in (0, 1, 4):
            output.append(
                (kind, sum(p) + int(kind == 1), 0, p + (int(kind == 1),), tuple(x))
            )
        else:
            output.append((2, 0, sum(m) + 1, m + (1,), tuple(x)))
    return output


def batch_size(kind, np_, nm_):
    active = np_ + nm_ - 1 if kind == 3 else nm_ if kind == 2 else np_ + int(kind == 4)
    return min(128, max(1, 16384 // 2**active))


def training_buckets(rows):
    groups = defaultdict(Counter)
    for kind, np_, nm_, g, z in rows:
        groups[(kind, np_, nm_)][(g, z)] += 1
    bs = []
    ws = []
    for (kind, np_, nm_), count in sorted(groups.items()):
        items = list(count.items())
        size = batch_size(kind, np_, nm_)
        for start in range(0, len(items), size):
            part = items[start : start + size]
            n = len(part)
            part = part + [part[0]] * (size - n)
            bs.append(
                (
                    kind,
                    np_,
                    nm_,
                    np.asarray([x[0][0] for x in part], np.int8),
                    np.asarray([x[0][1] for x in part], float),
                )
            )
            ws.append(
                np.asarray([x[1] if i < n else 0 for i, x in enumerate(part)], float)
            )
    return bs, ws


@lru_cache(maxsize=4)
def row_function(nf):
    model = ProEHNTopologyModel(12, nf, log_rate_clip_min=-5.0, log_rate_clip_max=5.0)

    @partial(jax.jit, static_argnums=(1, 2, 3))
    def loglik(params, kind, np_, nm_, g, z):
        return jax.vmap(
            lambda a, b: -model.bucket_loss(
                params, kind, np_, nm_, a[None, :], b[None, :]
            )
        )(g, z)

    return loglik


def loglikelihoods(params, rows, nf):
    grouped = defaultdict(list)
    for i, (kind, np_, nm_, g, z) in enumerate(rows):
        grouped[(kind, np_, nm_)].append((i, g, z))
    result = np.empty(len(rows))
    f = row_function(nf)
    for (kind, np_, nm_), items in sorted(grouped.items()):
        size = batch_size(kind, np_, nm_)
        for start in range(0, len(items), size):
            part = items[start : start + size]
            n = len(part)
            padded = part + [part[0]] * (size - n)
            values = np.asarray(
                f(
                    jnp.asarray(params),
                    kind,
                    np_,
                    nm_,
                    jnp.asarray([x[1] for x in padded], dtype=jnp.int8),
                    jnp.asarray([x[2] for x in padded]),
                )
            )
            result[[x[0] for x in part]] = values[:n]
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite patient likelihood")
    return result


def compartment_masks(rows, compartment):
    if compartment not in ("PT", "MT"):
        raise ValueError("PT or MT required")
    masked = []
    labels = []
    selected = []
    for i, (kind, np_, nm_, g, z) in enumerate(rows):
        if (compartment == "PT" and kind == 2) or (
            compartment == "MT" and kind in (0, 1, 4)
        ):
            continue
        selected.append(i)
        truth = []
        for j in range(12):
            index = (2 * j + int(compartment == "MT")) if kind == 3 else j
            actual = g[index]
            truth.append(actual)
            for value in (0, 1):
                changed = list(g)
                changed[index] = value
                delta = value - actual
                masked.append(
                    (
                        kind,
                        np_ + delta if compartment == "PT" else np_,
                        nm_ + delta if compartment == "MT" else nm_,
                        tuple(changed),
                        z,
                    )
                )
        labels.append(truth)
    return masked, np.asarray(labels, int), np.asarray(selected, int)
