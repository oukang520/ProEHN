"""Deprecated surrogate baseline; NOT the formal surrogate_baseline benchmark."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score, roc_curve
from sklearn.model_selection import KFold


DEFAULT_PACA_PROEHN_TARGETS = ("P.TP53 (M)", "P.KMT2C (M)", "P.ARID1A (M)")
DEFAULT_PACA_PROEHN_FEATURES = ("VAF_mean_Primary", "nMut_Primary")


def _state_to_int(row: Iterable[int]) -> int:
    out = 0
    for bit in row:
        out = (out << 1) | int(bit)
    return out


def _data_to_probability_distribution(data: np.ndarray) -> np.ndarray:
    n_events = data.shape[1]
    state_ints = np.array([_state_to_int(row) for row in data], dtype=int)
    counts = np.bincount(state_ints, minlength=2**n_events)
    return counts / np.sum(counts)


def _mhn_kronvec(
    theta_row: np.ndarray,
    event_idx: int,
    x: np.ndarray,
    diag: bool = False,
    transp: bool = False,
) -> np.ndarray:
    n_events = len(theta_row)
    n_states = len(x)
    half = n_states // 2
    out = np.empty(n_states, dtype=float)
    tmp = x.copy()

    for j in range(n_events):
        out[:half] = tmp[0:n_states:2]
        out[half:] = tmp[1:n_states:2]
        theta = theta_row[j]
        if j == event_idx:
            if not transp:
                out[half:] = out[:half] * theta
                out[:half] = -out[half:] if diag else 0.0
            else:
                out[:half] = out[half:] * theta
                out[half:] = -out[:half] if diag else 0.0
        else:
            out[half:] = out[half:] * theta
        tmp[:] = out[:]
    return out


def _mhn_q_subdiag(theta: np.ndarray, event_idx: int) -> np.ndarray:
    row = theta[event_idx, :]
    values = np.array([np.exp(row[event_idx])], dtype=float)
    for j in range(len(row)):
        multiplier = np.exp(row[j]) if event_idx != j else 0.0
        values = np.concatenate((values, values * multiplier))
    return values


def _mhn_q_diag(theta: np.ndarray) -> np.ndarray:
    n_events = theta.shape[1]
    diag = np.zeros(2**n_events, dtype=float)
    for event_idx in range(n_events):
        diag -= _mhn_q_subdiag(theta, event_idx)
    return diag


def _mhn_q_vec(theta: np.ndarray, x: np.ndarray, diag: bool = False, transp: bool = False) -> np.ndarray:
    n_events = theta.shape[1]
    y = np.zeros(2**n_events, dtype=float)
    for event_idx in range(n_events):
        y += _mhn_kronvec(np.exp(theta[event_idx, :]), event_idx, x, diag=diag, transp=transp)
    return y


def _mhn_jacobi(theta: np.ndarray, b: np.ndarray, transp: bool = False, x: np.ndarray | None = None) -> np.ndarray:
    n_events = theta.shape[1]
    if x is None:
        x = np.ones(2**n_events, dtype=float) / (2**n_events)
    diag = -_mhn_q_diag(theta) + 1.0
    for _ in range(n_events + 1):
        x = b + _mhn_q_vec(theta, x, transp=transp)
        x = x / diag
    return x


def _mhn_generate_pth(theta: np.ndarray) -> np.ndarray:
    n_events = theta.shape[1]
    p0 = np.zeros(2**n_events, dtype=float)
    p0[0] = 1.0
    return _mhn_jacobi(theta, p0)


def _mhn_score(theta: np.ndarray, p_data: np.ndarray) -> float:
    p_theta = _mhn_generate_pth(theta)
    return float(np.sum(p_data * np.log(np.maximum(p_theta, 1e-15))))


def _mhn_grad_loop_j(event_idx: int, n_events: int, rates: np.ndarray) -> np.ndarray:
    n_states = len(rates)
    half = n_states // 2
    grad = np.zeros(n_events, dtype=float)
    tmp = np.empty(n_states, dtype=float)
    work = rates.copy()
    for j in range(n_events):
        tmp[:half] = work[0:n_states:2]
        tmp[half:] = work[1:n_states:2]
        work[:] = tmp[:]
        grad[j] = np.sum(work[half:])
        if j == event_idx:
            grad[j] += np.sum(work[:half])
    return grad


def _mhn_grad(theta: np.ndarray, p_data: np.ndarray) -> np.ndarray:
    n_events = theta.shape[1]
    p0 = np.zeros(2**n_events, dtype=float)
    p0[0] = 1.0
    p_theta = _mhn_jacobi(theta, p0)
    q = _mhn_jacobi(theta, p_data / np.maximum(p_theta, 1e-15), transp=True)
    gradient = np.zeros((n_events, n_events), dtype=float)
    for event_idx in range(n_events):
        rates = q * _mhn_kronvec(np.exp(theta[event_idx, :]), event_idx, p_theta, diag=True)
        gradient[event_idx, :] = _mhn_grad_loop_j(event_idx, n_events, rates)
    return gradient


def _mhn_l1_penalty(theta: np.ndarray, eps: float = 1e-5) -> float:
    diag_mask = np.ones_like(theta) - np.eye(theta.shape[0])
    return float(np.sum(np.sqrt((theta * diag_mask) ** 2 + eps)))


def _mhn_l1_grad(theta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    diag_mask = np.ones_like(theta) - np.eye(theta.shape[0])
    return (theta * diag_mask) / np.sqrt((theta * diag_mask) ** 2 + eps)


def _mhn_learn_independent(p_data: np.ndarray) -> np.ndarray:
    n_events = int(np.log2(len(p_data)))
    theta = np.zeros((n_events, n_events), dtype=float)
    p_current = p_data.copy()
    for event_idx in range(n_events):
        p_matrix = p_current.reshape((2 ** (n_events - 1), 2))
        prevalence = np.sum(p_matrix[:, 1])
        prevalence = np.clip(prevalence, 1e-10, 1.0 - 1e-10)
        theta[event_idx, event_idx] = np.log(prevalence / (1.0 - prevalence))
        p_current = p_matrix.flatten(order="F")
    return np.round(theta, 2)


def _mhn_score_reg(theta_flat: np.ndarray, p_data: np.ndarray, n_events: int, lambda_reg: float) -> float:
    theta = theta_flat.reshape((n_events, n_events))
    return -(_mhn_score(theta, p_data) - lambda_reg * _mhn_l1_penalty(theta))


def _mhn_grad_reg(theta_flat: np.ndarray, p_data: np.ndarray, n_events: int, lambda_reg: float) -> np.ndarray:
    theta = theta_flat.reshape((n_events, n_events))
    gradient = _mhn_grad(theta, p_data) - lambda_reg * _mhn_l1_grad(theta)
    return -gradient.flatten()


def fit_mhn(data: np.ndarray, lambda_reg: float = 0.01, maxit: int = 5000, reltol: float = 1e-7) -> np.ndarray:
    """Fit the regularized MHN topology used as the surrogate_baseline baseline topology term."""

    data = np.asarray(data, dtype=int)
    p_data = _data_to_probability_distribution(data)
    n_events = int(np.log2(len(p_data)))
    init = _mhn_learn_independent(p_data)
    result = minimize(
        fun=_mhn_score_reg,
        x0=init.flatten(),
        args=(p_data, n_events, float(lambda_reg)),
        method="BFGS",
        jac=_mhn_grad_reg,
        options={"maxiter": int(maxit), "gtol": reltol, "disp": False},
    )
    return np.round(result.x.reshape((n_events, n_events)), 2)


def _sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -700, 700)))


def _binary_event_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    values = df.loc[:, columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return (values > 0).astype(int)


def _feature_frame(df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    work = pd.DataFrame(index=df.index)
    for col in feature_columns:
        if col in df.columns:
            work[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            work[col] = 0.0
    return work.fillna(work.mean(numeric_only=True)).fillna(0.0)


def _standardize_train_test(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train_arr = train.to_numpy(dtype=float)
    test_arr = test.to_numpy(dtype=float)
    mean = train_arr.mean(axis=0)
    std = train_arr.std(axis=0)
    std[std == 0.0] = 1.0
    return (train_arr - mean) / std, (test_arr - mean) / std


def _fixed_go_probability(raw_features: pd.DataFrame, vaf_threshold: float, nmut_threshold: float) -> np.ndarray:
    vaf = raw_features["VAF_mean_Primary"].to_numpy(dtype=float)
    nmut = raw_features["nMut_Primary"].to_numpy(dtype=float)
    stop_like = (vaf >= vaf_threshold) & (nmut <= nmut_threshold)
    return np.where(stop_like, 0.15, 0.85).astype(float)


def _fit_feature_shift(theta: np.ndarray, genotypes: np.ndarray, features: np.ndarray) -> np.ndarray:
    def objective(weights: np.ndarray) -> float:
        logits = theta[0, 0] + genotypes[:, 1:] @ theta[0, 1:] + features @ weights
        prob = _sigmoid(logits)
        y = genotypes[:, 0]
        log_likelihood = np.sum(y * np.log(prob + 1e-9) + (1 - y) * np.log(1 - prob + 1e-9))
        return float(-log_likelihood + 0.1 * np.sum(weights**2))

    return minimize(objective, np.zeros(features.shape[1], dtype=float), method="L-BFGS-B").x


def _cross_validated_proehn_predictions(
    gene_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    n_splits: int,
    random_seed: int,
    mhn_lambda: float,
    mhn_maxit: int,
    vaf_threshold: float,
    nmut_threshold: float,
) -> pd.DataFrame:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    rows: list[dict[str, float | int]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(gene_df), start=1):
        train_genes = gene_df.iloc[train_idx].to_numpy(dtype=int)
        test_genes = gene_df.iloc[test_idx].to_numpy(dtype=int)
        train_features, test_features = _standardize_train_test(
            feature_df.iloc[train_idx],
            feature_df.iloc[test_idx],
        )
        theta = fit_mhn(train_genes, lambda_reg=mhn_lambda, maxit=mhn_maxit)
        feature_shift = _fit_feature_shift(theta, train_genes, train_features)
        p_go = _fixed_go_probability(feature_df.iloc[test_idx], vaf_threshold, nmut_threshold)

        for local_idx, state in enumerate(test_genes):
            topology_logit = float(theta[0, 0] + state[1:] @ theta[0, 1:])
            proehn_logit = topology_logit + float(test_features[local_idx] @ feature_shift)
            rows.append(
                {
                    "row_index": int(test_idx[local_idx]),
                    "fold": int(fold_idx),
                    "y_true": int(state[0]),
                    "proehn_score": float(p_go[local_idx] * _sigmoid(proehn_logit)),
                }
            )
    return pd.DataFrame(rows)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def specificity_at_sensitivity(y_true: np.ndarray, y_score: np.ndarray, target_sensitivity: float = 0.90) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    indices = np.where(tpr >= target_sensitivity)[0]
    if len(indices) == 0:
        return 0.0
    return float(1.0 - fpr[indices[0]])


def max_mcc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    _, _, thresholds = roc_curve(y_true, y_score)
    best_score = -1.0
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        if len(np.unique(y_pred)) > 1:
            best_score = max(best_score, float(matthews_corrcoef(y_true, y_pred)))
    return best_score if best_score != -1.0 else 0.0


def _metric_rows(
    target: str,
    predictions: pd.DataFrame,
    n_splits: int,
    n_context_genes: int,
    vaf_threshold: float,
    nmut_threshold: float,
) -> list[dict[str, float | int | str]]:
    y_true = predictions["y_true"].to_numpy(dtype=int)
    score = predictions["proehn_score"].to_numpy(dtype=float)
    target_name = target.replace("P.", "").replace(" (M)", "")
    common = {
        "target": target_name,
        "target_column": target,
        "method": "surrogate_baseline",
        "n_samples": int(len(y_true)),
        "n_positive": int(np.sum(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "n_splits": int(n_splits),
        "n_context_genes": int(n_context_genes),
        "gate_vaf_threshold": float(vaf_threshold),
        "gate_nmut_threshold": float(nmut_threshold),
    }
    return [
        {**common, "metric": "auc_roc", "value": _safe_auc(y_true, score)},
        {**common, "metric": "auc_pr", "value": _safe_average_precision(y_true, score)},
        {
            **common,
            "metric": "specificity_at_90_sensitivity",
            "value": specificity_at_sensitivity(y_true, score),
        },
        {**common, "metric": "max_mcc", "value": max_mcc(y_true, score)},
    ]


def run_surrogate_baseline_metrics(
    df: pd.DataFrame,
    targets: Sequence[str] = DEFAULT_PACA_PROEHN_TARGETS,
    feature_columns: Sequence[str] = DEFAULT_PACA_PROEHN_FEATURES,
    n_context_genes: int = 9,
    n_splits: int = 5,
    random_seed: int = 2026,
    mhn_lambda: float = 0.01,
    mhn_maxit: int = 5000,
    vaf_threshold: float = 0.5,
    nmut_threshold: float = 20.0,
) -> dict[str, pd.DataFrame]:
    """Run surrogate_baseline-only PACA AUC/AUPRC/Spec90/MCC tables without result tuning."""

    primary_mutations = [col for col in df.columns if col.startswith("P.") and "(M)" in col]
    missing_targets = [target for target in targets if target not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing PACA target columns: {missing_targets}")
    if n_context_genes < 1:
        raise ValueError("n_context_genes must be at least 1.")

    feature_df = _feature_frame(df, feature_columns)
    metric_rows: list[dict[str, float | int | str]] = []
    context_rows: list[dict[str, float | int | str]] = []
    for target in targets:
        available_contexts = [gene for gene in primary_mutations if gene != target]
        frequencies = _binary_event_frame(df, available_contexts).mean().sort_values(ascending=False)
        context_genes = frequencies.head(n_context_genes).index.tolist()
        gene_df = _binary_event_frame(df, [target, *context_genes])
        predictions = _cross_validated_proehn_predictions(
            gene_df=gene_df,
            feature_df=feature_df,
            n_splits=n_splits,
            random_seed=random_seed,
            mhn_lambda=mhn_lambda,
            mhn_maxit=mhn_maxit,
            vaf_threshold=vaf_threshold,
            nmut_threshold=nmut_threshold,
        )
        metric_rows.extend(
            _metric_rows(
                target,
                predictions,
                n_splits=n_splits,
                n_context_genes=len(context_genes),
                vaf_threshold=vaf_threshold,
                nmut_threshold=nmut_threshold,
            )
        )
        for rank, gene in enumerate(context_genes, start=1):
            context_rows.append(
                {
                    "target": target.replace("P.", "").replace(" (M)", ""),
                    "target_column": target,
                    "context_rank": int(rank),
                    "context_gene": gene.replace("P.", "").replace(" (M)", ""),
                    "context_column": gene,
                    "frequency": float(frequencies[gene]),
                }
            )

    return {
        "paca_proehn_metrics": pd.DataFrame(metric_rows),
        "paca_proehn_context_genes": pd.DataFrame(context_rows),
    }
