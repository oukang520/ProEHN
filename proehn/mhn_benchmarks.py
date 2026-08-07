"""PACA baseline benchmarks adapted from the MHN experiment scripts."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score, roc_curve
from sklearn.model_selection import KFold


DEFAULT_PACA_MHN_TARGETS = ("P.TP53 (M)", "P.KMT2C (M)", "P.ARID1A (M)")


class OncotreeLearner:
    """Simple pairwise Oncotree learner used by the PACA MHN benchmarks."""

    def __init__(self, epsilon: float = 1e-9) -> None:
        self.epsilon = epsilon
        self.n_nodes = 0
        self.parents: dict[int, int] = {}
        self.cond_probs_1: dict[int, float] = {}
        self.cond_probs_0: dict[int, float] = {}
        self.marginals: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> OncotreeLearner:
        x = np.asarray(x, dtype=float)
        n_patients, self.n_nodes = x.shape
        self.marginals = np.sum(x, axis=0) / (n_patients + 1e-12)
        joints = (x.T @ x) / n_patients

        for j in range(self.n_nodes):
            best_parent = -1
            best_weight = 1.0
            for i in range(self.n_nodes):
                if i == j:
                    continue
                if self.marginals[i] > self.marginals[j]:
                    weight_ij = joints[i, j] / (self.marginals[i] * self.marginals[j] + 1e-12)
                    if weight_ij > best_weight:
                        best_weight = weight_ij
                        best_parent = i

            self.parents[j] = best_parent
            if best_parent != -1:
                self.cond_probs_1[j] = float(joints[best_parent, j] / (self.marginals[best_parent] + 1e-12))
                p_j1_i0 = max(0.0, float(self.marginals[j] - joints[best_parent, j]))
                p_i0 = max(1e-12, float(1.0 - self.marginals[best_parent]))
                self.cond_probs_0[j] = p_j1_i0 / p_i0
            else:
                self.cond_probs_1[j] = float(self.marginals[j])
                self.cond_probs_0[j] = float(self.marginals[j])
        return self

    def predict_next_prob(self, target_idx: int, context_vector: np.ndarray) -> float:
        if self.marginals is None:
            raise RuntimeError("OncotreeLearner must be fitted before prediction.")
        parent_idx = self.parents[target_idx]
        if parent_idx == -1:
            return max(self.epsilon, float(self.marginals[target_idx]))
        if context_vector[parent_idx] == 1:
            return max(self.epsilon, self.cond_probs_1[target_idx])
        return max(self.epsilon, self.cond_probs_0[target_idx])


class HyperTraPS:
    """Second-order HyperTraPS baseline used by the PACA MHN benchmarks."""

    def __init__(self, n_events: int, limit: float = 10.0, rng: np.random.RandomState | None = None) -> None:
        self.n_events = n_events
        self.limit = limit
        self.rng = rng or np.random.RandomState()
        self.pi = np.zeros((n_events, n_events), dtype=float)

    def _perturb(self, sigma: float) -> np.ndarray:
        previous = np.copy(self.pi)
        self.pi += self.rng.normal(0.0, sigma, size=(self.n_events, self.n_events))
        self.pi = np.clip(self.pi, -self.limit, self.limit)
        return previous

    @staticmethod
    def _is_compatible(current: np.ndarray, target: np.ndarray) -> bool:
        return not np.any((current == 1) & (target == 0))

    def _simulate_trajectory(self, start_state: np.ndarray, end_state: np.ndarray) -> float:
        current = np.copy(start_state)
        alpha = 1.0
        cumulative_alpha = 0.0
        while True:
            if self._is_compatible(current, end_state):
                cumulative_alpha += alpha
            if np.array_equal(current, end_state):
                break
            rate_log = np.diag(self.pi) + self.pi @ current
            rates = np.exp(np.clip(rate_log, -700, 230))
            rates = np.clip(rates, 2e-300, 1e100) * (current == 0)
            total_rate = np.sum(rates)
            if total_rate <= 0:
                break
            compatible_rates = rates * ((end_state == 1) | (end_state == 2))
            compatible_total = np.sum(compatible_rates)
            if compatible_total <= 0:
                break
            alpha *= compatible_total / total_rate
            probabilities = compatible_rates / compatible_total
            probabilities /= np.sum(probabilities)
            current[self.rng.choice(self.n_events, p=probabilities)] = 1
        return float(cumulative_alpha)

    def log_likelihood(self, dataset: np.ndarray, n_walks: int = 100) -> float:
        total_ll = 0.0
        start_state = np.zeros(self.n_events, dtype=int)
        unique_states, counts = np.unique(dataset, axis=0, return_counts=True)
        for state, count in zip(unique_states, counts):
            mean_alpha = np.mean([self._simulate_trajectory(start_state, state) for _ in range(n_walks)])
            total_ll += (np.log(mean_alpha) if mean_alpha > 0 else -1e10) * count
        return float(total_ll)

    def run_mcmc(self, dataset: np.ndarray, iterations: int = 150, sigma: float = 0.1, n_walks: int = 15) -> None:
        current_ll = self.log_likelihood(dataset, n_walks)
        best_pi = np.copy(self.pi)
        best_ll = current_ll
        for _ in range(iterations):
            previous_pi = self._perturb(sigma)
            proposed_ll = self.log_likelihood(dataset, n_walks)
            if (proposed_ll - current_ll) > np.log(self.rng.uniform(0.0, 1.0)):
                current_ll = proposed_ll
                best_ll = current_ll
                best_pi = np.copy(self.pi)
            else:
                self.pi = previous_pi
        if np.isfinite(best_ll):
            self.pi = best_pi


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


def fit_mhn(data: np.ndarray, lambda_reg: float | None = None, maxit: int = 5000, reltol: float = 1e-7) -> np.ndarray:
    """Fit a regularized MHN from a binary event matrix."""

    data = np.asarray(data, dtype=int)
    if lambda_reg is None:
        lambda_reg = 1.0 / data.shape[0]
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


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -700, 700))))


def _hypertraps_next_prob(model: HyperTraPS, target_idx: int, context_vector: np.ndarray) -> float:
    rate_log = np.diag(model.pi) + model.pi @ context_vector
    rates = np.exp(np.clip(rate_log, -700, 230))
    available_rates = rates * (context_vector == 0)
    total = float(np.sum(available_rates))
    if total <= 0.0:
        return 0.0
    return float(available_rates[target_idx] / total)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def specificity_at_sensitivity(y_true: np.ndarray, y_score: np.ndarray, target_sensitivity: float = 0.90) -> float:
    """Specificity at the first ROC threshold reaching the target sensitivity."""

    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    indices = np.where(tpr >= target_sensitivity)[0]
    if len(indices) == 0:
        return 0.0
    return float(1.0 - fpr[indices[0]])


def max_mcc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Best Matthews correlation coefficient across ROC thresholds."""

    if len(np.unique(y_true)) < 2:
        return float("nan")
    _, _, thresholds = roc_curve(y_true, y_score)
    best = -1.0
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        if len(np.unique(y_pred)) > 1:
            best = max(best, float(matthews_corrcoef(y_true, y_pred)))
    return best if best != -1.0 else 0.0


def _binary_event_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    values = df.loc[:, columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return (values > 0).astype(int)


def _cross_validated_predictions(
    gene_df: pd.DataFrame,
    n_splits: int,
    random_seed: int,
    mhn_lambda: float,
    mhn_maxit: int,
    hypertraps_iterations: int,
    hypertraps_walks: int,
    hypertraps_sigma: float,
) -> pd.DataFrame:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    rows: list[dict[str, float | int]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(gene_df), start=1):
        train = gene_df.iloc[train_idx].to_numpy(dtype=int)
        test = gene_df.iloc[test_idx].to_numpy(dtype=int)
        theta = fit_mhn(train, lambda_reg=mhn_lambda, maxit=mhn_maxit)
        oncotree = OncotreeLearner().fit(train)
        hypertraps = HyperTraPS(n_events=train.shape[1], rng=np.random.RandomState(random_seed + fold_idx))
        hypertraps.run_mcmc(train, iterations=hypertraps_iterations, sigma=hypertraps_sigma, n_walks=hypertraps_walks)

        for local_idx, state in enumerate(test):
            context = np.copy(state)
            context[0] = 0
            mhn_logit = float(theta[0, 0] + state[1:] @ theta[0, 1:])
            rows.append(
                {
                    "row_index": int(test_idx[local_idx]),
                    "fold": int(fold_idx),
                    "y_true": int(state[0]),
                    "Oncotrees": oncotree.predict_next_prob(0, context),
                    "HyperTraPS": _hypertraps_next_prob(hypertraps, 0, context),
                    "MHN": _sigmoid(mhn_logit),
                }
            )

    return pd.DataFrame(rows)


def _metric_rows(
    target: str,
    predictions: pd.DataFrame,
    n_splits: int,
    n_context_genes: int,
) -> list[dict[str, float | int | str]]:
    y_true = predictions["y_true"].to_numpy(dtype=int)
    methods = ("Oncotrees", "HyperTraPS", "MHN")
    target_name = target.replace("P.", "").replace(" (M)", "")
    n_positive = int(np.sum(y_true))
    positive_rate = float(np.mean(y_true))
    common = {
        "target": target_name,
        "target_column": target,
        "n_samples": int(len(y_true)),
        "n_positive": n_positive,
        "positive_rate": positive_rate,
        "n_splits": int(n_splits),
        "n_context_genes": int(n_context_genes),
    }
    rows: list[dict[str, float | int | str]] = [
        {**common, "metric": "auc_roc", "method": "Baseline", "value": 0.5},
        {**common, "metric": "auc_pr", "method": "Baseline", "value": positive_rate},
        {**common, "metric": "specificity_at_90_sensitivity", "method": "Baseline", "value": 0.10},
    ]
    for method in methods:
        score = predictions[method].to_numpy(dtype=float)
        rows.extend(
            [
                {**common, "metric": "auc_roc", "method": method, "value": _safe_auc(y_true, score)},
                {**common, "metric": "auc_pr", "method": method, "value": _safe_average_precision(y_true, score)},
                {
                    **common,
                    "metric": "specificity_at_90_sensitivity",
                    "method": method,
                    "value": specificity_at_sensitivity(y_true, score),
                },
                {**common, "metric": "max_mcc", "method": method, "value": max_mcc(y_true, score)},
            ]
        )
    return rows


def run_paca_mhn_benchmark(
    df: pd.DataFrame,
    targets: Sequence[str] = DEFAULT_PACA_MHN_TARGETS,
    n_context_genes: int = 9,
    n_splits: int = 5,
    random_seed: int = 2026,
    mhn_lambda: float = 0.01,
    mhn_maxit: int = 5000,
    hypertraps_iterations: int = 150,
    hypertraps_walks: int = 15,
    hypertraps_sigma: float = 0.1,
) -> dict[str, pd.DataFrame]:
    """Run cleaned PACA Oncotrees, HyperTraPS and MHN baseline benchmarks."""

    if n_context_genes < 1:
        raise ValueError("n_context_genes must be at least 1.")
    primary_mutations = [col for col in df.columns if col.startswith("P.") and "(M)" in col]
    missing_targets = [target for target in targets if target not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing PACA target columns: {missing_targets}")

    metric_rows: list[dict[str, float | int | str]] = []
    context_rows: list[dict[str, float | int | str]] = []
    for target in targets:
        available_contexts = [gene for gene in primary_mutations if gene != target]
        frequencies = _binary_event_frame(df, available_contexts).mean().sort_values(ascending=False)
        context_genes = frequencies.head(n_context_genes).index.tolist()
        gene_df = _binary_event_frame(df, [target, *context_genes])
        predictions = _cross_validated_predictions(
            gene_df=gene_df,
            n_splits=n_splits,
            random_seed=random_seed,
            mhn_lambda=mhn_lambda,
            mhn_maxit=mhn_maxit,
            hypertraps_iterations=hypertraps_iterations,
            hypertraps_walks=hypertraps_walks,
            hypertraps_sigma=hypertraps_sigma,
        )
        metric_rows.extend(_metric_rows(target, predictions, n_splits=n_splits, n_context_genes=len(context_genes)))
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
        "paca_mhn_benchmark_metrics": pd.DataFrame(metric_rows),
        "paca_mhn_context_genes": pd.DataFrame(context_rows),
    }
