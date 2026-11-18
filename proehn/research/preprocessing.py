"""Distinct train-fitted GENIE and ICGC topology transforms."""

import numpy as np


def _raw(frame):
    values = frame[["age", "background_nMut", "background_VAF_mean"]].to_numpy(float)
    if (
        np.isinf(values).any()
        or np.any(values[:, 1] < 0)
        or np.any((values[:, 2] < 0) | (values[:, 2] > 1))
    ):
        raise ValueError(
            "Finite age, nonnegative variant counts and VAF in[0,1] are required; NaN marks missing"
        )
    values[:, 1] = np.log1p(values[:, 1])
    return values


def genie_fit_transform(train):
    a = _raw(train)
    safe = a.copy()
    safe[:, np.isnan(a).all(axis=0)] = 0.0
    fill = np.nanmedian(safe, axis=0)
    v = np.c_[np.where(np.isnan(a), fill, a), np.isnan(a[:, 0]), np.isnan(a[:, 2])]
    mean = v.mean(axis=0)
    scale = v.std(axis=0)
    scale[scale == 0] = 1
    return dict(fill=fill.tolist(), mean=mean.tolist(), scale=scale.tolist())


def genie_transform(frame, metadata):
    a = _raw(frame)
    v = np.c_[
        np.where(np.isnan(a), metadata["fill"], a), np.isnan(a[:, 0]), np.isnan(a[:, 2])
    ]
    return np.c_[np.ones(len(v)), (v - metadata["mean"]) / metadata["scale"]]


def icgc_fit_transform(frame):
    raw = _raw(frame)
    safe = raw.copy()
    safe[:, np.isnan(raw).all(axis=0)] = 0.0
    bounds = np.nanquantile(safe, [0.01, 0.99], axis=0)
    clipped = np.clip(raw, *bounds)
    safe = clipped.copy()
    safe[:, np.isnan(clipped).all(axis=0)] = 0.0
    fill = np.nanmedian(safe, axis=0)
    values = np.where(np.isnan(clipped), fill, clipped)
    mean = values.mean(0)
    scale = values.std(0)
    scale[scale < 1e-8] = 1
    return dict(
        bounds=bounds.tolist(),
        fill=fill.tolist(),
        mean=mean.tolist(),
        scale=scale.tolist(),
    )


def icgc_transform(frame, metadata, columns):
    raw = np.clip(_raw(frame), *metadata["bounds"])
    raw = (
        np.where(np.isnan(raw), metadata["fill"], raw) - metadata["mean"]
    ) / metadata["scale"]
    return np.c_[np.ones(len(frame)), raw[:, columns]]


def base_indices(nf):
    return np.r_[
        np.arange(169),
        np.arange((nf + 1) * 169, (nf + 1) * 169 + 13),
        np.arange((nf + 1) * 182, (nf + 1) * 182 + 13),
    ]


def diagonal_basis(nf, mode):
    """All 195 intercepts retained; pooling constrains covariate effects only."""
    size = (nf + 1) * 195
    columns = []
    for index in base_indices(nf):
        v = np.zeros(size)
        v[index] = 1
        columns.append(v)
    for k in range(1, nf + 1):
        if mode == "pooled":
            groups = [
                k * 169 + np.arange(12) * 14,
                np.array([k * 169 + 168]),
                (nf + 1) * 169 + k * 13 + np.arange(13),
                (nf + 1) * 182 + k * 13 + np.arange(13),
            ]
        elif mode == "diagonal":
            groups = [
                np.array([i])
                for i in np.r_[
                    k * 169 + np.arange(13) * 14,
                    (nf + 1) * 169 + k * 13 + np.arange(13),
                    (nf + 1) * 182 + k * 13 + np.arange(13),
                ]
            ]
        elif mode == "constant":
            raise ValueError("A constant model has no context features")
        else:
            raise ValueError(mode)
        for group in groups:
            v = np.zeros(size)
            v[group] = 1
            columns.append(v)
    return np.stack(columns, axis=1)
