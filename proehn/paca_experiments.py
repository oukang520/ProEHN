"""Pure-data PACA example experiments for the ProEHN manuscript panels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import average_precision_score, roc_auc_score

from .kinetic import ProEHNKineticGatekeeper
from .mhn_benchmarks import DEFAULT_PACA_MHN_TARGETS, run_paca_mhn_benchmark


FEATURE_CATEGORIES = {
    "Chromosomal (CNA)": ("cna", "fga"),
    "Mutation Load (TMB)": ("nmut", "tmb"),
    "Clonality (VAF)": ("vaf",),
    "Demographic (Age)": ("age",),
    "Demographic (Sex)": ("sex", "gender", "male", "female"),
    "Clinical": ("stage", "grade"),
}

PACA_ANCHOR_STATES = {
    "WT / early": {"include": (), "exclude": ("KRAS", "TP53", "SMAD4", "CDKN2A", "ARID1A")},
    "KRAS+": {"include": ("KRAS",), "exclude": ("TP53", "SMAD4")},
    "TP53+": {"include": ("TP53",), "exclude": ("KRAS",)},
    "KRAS+TP53+": {"include": ("KRAS", "TP53"), "exclude": ()},
    "ARID1A+": {"include": ("ARID1A",), "exclude": ()},
    "ATM+": {"include": ("ATM",), "exclude": ()},
}


@dataclass(frozen=True)
class TopologyArtifact:
    """Loaded topology artifact with parsed parameter tensors."""

    params: np.ndarray
    gene_names: list[str]
    feature_names: list[str]
    n_feature_rows: int
    W_theta: np.ndarray
    W_dp: np.ndarray
    W_dm: np.ndarray


def _as_list(values: Any) -> list[str]:
    if isinstance(values, np.ndarray):
        return [str(item) for item in values.tolist()]
    return [str(item) for item in values]


def load_topology_artifact(path: str | Path) -> TopologyArtifact:
    """Load a ProEHN topology artifact saved as `.npz`."""

    data = np.load(path, allow_pickle=True)
    params = np.asarray(data["params"], dtype=np.float64)
    gene_names = _as_list(data["gene_names"])
    feature_names = _as_list(data["feature_names"])
    n_total = len(gene_names) + 1
    row_width = n_total * (n_total + 2)
    if params.size % row_width != 0:
        raise ValueError(f"Topology parameter size {params.size} is incompatible with {len(gene_names)} genes.")
    n_feature_rows = params.size // row_width
    split1 = n_feature_rows * n_total**2
    split2 = split1 + n_feature_rows * n_total
    W_theta = params[:split1].reshape(n_feature_rows, n_total, n_total)
    W_dp = params[split1:split2].reshape(n_feature_rows, n_total)
    W_dm = params[split2:].reshape(n_feature_rows, n_total)
    return TopologyArtifact(params, gene_names, feature_names, n_feature_rows, W_theta, W_dp, W_dm)


def _feature_columns_for_model(artifact: TopologyArtifact) -> list[str]:
    if artifact.n_feature_rows == len(artifact.feature_names) + 1:
        return artifact.feature_names
    if artifact.n_feature_rows == len(artifact.feature_names):
        first = artifact.feature_names[0].lower() if artifact.feature_names else ""
        if first in {"bias", "intercept", "dummy"}:
            return artifact.feature_names[1:]
        return artifact.feature_names[: max(0, artifact.n_feature_rows - 1)]
    return artifact.feature_names[: max(0, artifact.n_feature_rows - 1)]


def build_standardized_feature_matrix(df: pd.DataFrame, artifact: TopologyArtifact) -> tuple[np.ndarray, list[str]]:
    """Build the source-compatible `[bias, z-scored features...]` matrix."""

    feature_cols = _feature_columns_for_model(artifact)
    work = df.copy()
    for col in feature_cols:
        if col not in work.columns:
            work[col] = 0.0
    raw = work[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    std[std == 0] = 1.0
    norm = (raw - mean) / std
    features = np.hstack([np.ones((len(df), 1), dtype=np.float64), norm])
    if features.shape[1] < artifact.n_feature_rows:
        pad = np.zeros((len(df), artifact.n_feature_rows - features.shape[1]), dtype=np.float64)
        features = np.hstack([features, pad])
    elif features.shape[1] > artifact.n_feature_rows:
        features = features[:, : artifact.n_feature_rows]
    return features, feature_cols


def build_genotype_matrix(df: pd.DataFrame, gene_names: list[str]) -> np.ndarray:
    """Build interleaved primary/metastatic genotype columns for selected genes."""

    columns: list[np.ndarray] = []
    for gene in gene_names:
        p_col = f"P.{gene} (M)"
        m_col = f"M.{gene} (M)"
        if p_col not in df.columns:
            p_col = next((col for col in df.columns if col.startswith(f"P.{gene}") and "(M)" in col), None)
        if m_col not in df.columns:
            m_col = next((col for col in df.columns if col.startswith(f"M.{gene}") and "(M)" in col), None)
        p_values = df[p_col].fillna(0).astype(int).to_numpy() if p_col else np.zeros(len(df), dtype=int)
        m_values = df[m_col].fillna(0).astype(int).to_numpy() if m_col else np.zeros(len(df), dtype=int)
        columns.extend([p_values, m_values])
    return np.column_stack(columns)


def compute_patient_theta(artifact: TopologyArtifact, feature_vector: np.ndarray) -> np.ndarray:
    """Project one standardized patient vector into theta."""

    theta = np.einsum("k,kij->ij", feature_vector, artifact.W_theta)
    return np.clip(theta, -20.0, 20.0)


def accessible_rates(theta: np.ndarray, gene_names: list[str], active_gene_indices: Iterable[int], include_seeding: bool = True) -> dict[str, float]:
    """Return accessible event hazards from one active gene set."""

    active = sorted(set(int(idx) for idx in active_gene_indices))
    rates: dict[str, float] = {}
    for idx, gene in enumerate(gene_names):
        if idx in active:
            continue
        log_rate = theta[idx, idx] + theta[idx, active].sum()
        rates[gene] = float(np.exp(log_rate))
    if include_seeding:
        seed_idx = len(gene_names)
        log_seed = theta[seed_idx, seed_idx] + theta[seed_idx, active].sum()
        rates["Metastatic seeding"] = float(np.exp(log_seed))
    return rates


def normalize_rates(rates: Mapping[str, float]) -> dict[str, float]:
    total = float(sum(rates.values())) + 1e-12
    return {event: float(rate / total) for event, rate in rates.items()}


def classify_feature(name: str) -> str:
    low = name.lower()
    for category, tokens in FEATURE_CATEGORIES.items():
        if any(token in low for token in tokens):
            return category
    return "Other"


def _survival_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    time_col = next((col for col in df.columns if col.lower() == "os_days"), None)
    if time_col is None:
        time_col = next((col for col in df.columns if "os" in col.lower() and "day" in col.lower()), None)
    status_col = next((col for col in df.columns if col.lower() == "donor_vital_status"), None)
    if status_col is None:
        status_col = next((col for col in df.columns if "vital" in col.lower() or "status" in col.lower()), None)
    return time_col, status_col


def _event_indicator(values: pd.Series) -> np.ndarray:
    return values.apply(lambda x: 1 if str(x).strip().lower() in {"deceased", "dead", "1", "true", "yes"} else 0).to_numpy(dtype=int)


def infer_stop_labels(df: pd.DataFrame) -> np.ndarray:
    """Infer `1=Stop, 0=Go` labels using source-compatible PACA columns."""

    for label_col in ["Patient_Label", "patient_label", "label"]:
        if label_col in df.columns:
            values = pd.to_numeric(df[label_col], errors="coerce").fillna(-1).astype(int).to_numpy()
            if np.isin(values, [0, 1]).any():
                return values
    if "PT_label" in df.columns or "MT_label" in df.columns:
        pt = pd.to_numeric(df.get("PT_label", pd.Series(1, index=df.index)), errors="coerce").fillna(1)
        mt = pd.to_numeric(df.get("MT_label", pd.Series(1, index=df.index)), errors="coerce").fillna(1)
        return np.where((pt == 0) | (mt == 0), 0, 1).astype(int)
    if "PFS_days" in df.columns:
        pfs = pd.to_numeric(df["PFS_days"], errors="coerce")
        missing = pd.to_numeric(df.get("PFS_days_is_Missing", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        labels = np.full(len(df), -1, dtype=int)
        valid = (missing == 0) & pfs.notna()
        labels[valid] = (pfs[valid] >= 9 * 30.4375).astype(int)
        return labels
    raise ValueError("Cannot infer PACA Stop/Go labels. Provide Patient_Label, PT/MT labels, or PFS columns.")


def km_curve(time: np.ndarray, event: np.ndarray, group_name: str) -> pd.DataFrame:
    """Kaplan-Meier curve as pure tabular data."""

    valid = np.isfinite(time) & np.isfinite(event)
    time = np.asarray(time[valid], dtype=float)
    event = np.asarray(event[valid], dtype=int)
    event_times = np.sort(np.unique(time[event == 1]))
    survival = 1.0
    rows: list[dict[str, Any]] = []
    for t in event_times:
        at_risk = int(np.sum(time >= t))
        events = int(np.sum((time == t) & (event == 1)))
        censored = int(np.sum((time == t) & (event == 0)))
        if at_risk > 0:
            survival *= 1.0 - events / at_risk
        rows.append(
            {
                "group": group_name,
                "time": float(t),
                "survival_probability": float(survival),
                "at_risk": at_risk,
                "events": events,
                "censored": censored,
            }
        )
    return pd.DataFrame(rows)


def median_survival(km_table: pd.DataFrame) -> float:
    if km_table.empty:
        return float("nan")
    reached = km_table[km_table["survival_probability"] <= 0.5]
    if reached.empty:
        return float("inf")
    return float(reached.iloc[0]["time"])


def logrank_p_value(time_a: np.ndarray, event_a: np.ndarray, time_b: np.ndarray, event_b: np.ndarray) -> float:
    """Two-group log-rank p-value using the standard chi-square approximation."""

    times = np.sort(np.unique(np.concatenate([time_a[event_a == 1], time_b[event_b == 1]])))
    observed_minus_expected = 0.0
    variance = 0.0
    for t in times:
        n_a = np.sum(time_a >= t)
        n_b = np.sum(time_b >= t)
        d_a = np.sum((time_a == t) & (event_a == 1))
        d_b = np.sum((time_b == t) & (event_b == 1))
        n = n_a + n_b
        d = d_a + d_b
        if n <= 1 or d == 0:
            continue
        expected_a = d * (n_a / n)
        var_a = (n_a * n_b * d * (n - d)) / (n**2 * (n - 1))
        observed_minus_expected += d_a - expected_a
        variance += var_a
    if variance <= 0:
        return float("nan")
    statistic = observed_minus_expected**2 / variance
    return float(chi2.sf(statistic, df=1))


def harrell_concordance_index(time: np.ndarray, predicted_score: np.ndarray, event: np.ndarray) -> float:
    """Harrell C-index for survival scores where larger scores imply longer survival."""

    time = np.asarray(time, dtype=float)
    predicted_score = np.asarray(predicted_score, dtype=float)
    event = np.asarray(event, dtype=int)
    valid = np.isfinite(time) & np.isfinite(predicted_score) & np.isfinite(event)
    time = time[valid]
    predicted_score = predicted_score[valid]
    event = event[valid]

    concordant = 0.0
    comparable = 0.0
    n = len(time)
    for i in range(n):
        for j in range(i + 1, n):
            if time[i] == time[j]:
                if event[i] == 1 and event[j] == 1:
                    comparable += 1.0
                    concordant += 1.0 if predicted_score[i] == predicted_score[j] else 0.5
                continue
            if time[i] < time[j] and event[i] == 1:
                comparable += 1.0
                if predicted_score[i] < predicted_score[j]:
                    concordant += 1.0
                elif predicted_score[i] == predicted_score[j]:
                    concordant += 0.5
            elif time[j] < time[i] and event[j] == 1:
                comparable += 1.0
                if predicted_score[j] < predicted_score[i]:
                    concordant += 1.0
                elif predicted_score[i] == predicted_score[j]:
                    concordant += 0.5
    return float(concordant / comparable) if comparable > 0 else float("nan")


class PACAExampleExperiments:
    """Run pure-data example experiments corresponding to PACA manuscript panels."""

    def __init__(
        self,
        topology_model_path: str | Path,
        topology_data_path: str | Path,
        kinetic_params_path: str | Path | None = None,
        kinetic_metadata_path: str | Path | None = None,
        kinetic_data_path: str | Path | None = None,
        fallback_go_probability: float = 0.85,
    ) -> None:
        self.topology_model_path = Path(topology_model_path)
        self.topology_data_path = Path(topology_data_path)
        self.kinetic_data_path = Path(kinetic_data_path) if kinetic_data_path else None
        self.artifact = load_topology_artifact(self.topology_model_path)
        self.topology_df = pd.read_csv(self.topology_data_path)
        self.X_features, self.feature_columns = build_standardized_feature_matrix(self.topology_df, self.artifact)
        self.X_genotypes = build_genotype_matrix(self.topology_df, self.artifact.gene_names)
        self.kinetic = ProEHNKineticGatekeeper(
            kinetic_params_path,
            kinetic_metadata_path,
            fallback_go_probability=fallback_go_probability,
        )

    def _feature_index(self, tokens: Iterable[str]) -> int | None:
        tokens = tuple(token.lower() for token in tokens)
        for idx, name in enumerate(self.feature_columns, start=1):
            low = name.lower()
            if any(token in low for token in tokens) and "missing" not in low:
                return idx
        return None

    def _predict_stop_table(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for idx, row in df.iterrows():
            patient = row.to_dict()
            p_stop = self.kinetic.predict_stop_probability(patient)
            rows.append({"row_index": int(idx), "p_stop": p_stop, "p_go": 1.0 - p_stop})
        return pd.DataFrame(rows)

    def _active_gene_indices(self, row_index: int) -> list[int]:
        return [
            gene_idx
            for gene_idx in range(len(self.artifact.gene_names))
            if self.X_genotypes[row_index, 2 * gene_idx] == 1
            or self.X_genotypes[row_index, 2 * gene_idx + 1] == 1
        ]

    def _seeding_hazard(self, row_index: int) -> float:
        seed_idx = len(self.artifact.gene_names)
        theta = compute_patient_theta(self.artifact, self.X_features[row_index])
        active = self._active_gene_indices(row_index)
        log_rate = theta[seed_idx, seed_idx] + theta[seed_idx, active].sum()
        return float(np.exp(log_rate))

    def _patient_scores(self) -> pd.DataFrame:
        labels = infer_stop_labels(self.topology_df)
        time_col, status_col = _survival_columns(self.topology_df)
        event = _event_indicator(self.topology_df[status_col]) if status_col else np.zeros(len(self.topology_df), dtype=int)

        rows = []
        for idx, row in self.topology_df.iterrows():
            patient = row.to_dict()
            p_stop = self.kinetic.predict_stop_probability(patient)
            p_go = 1.0 - p_stop
            evolution_score = self._seeding_hazard(idx)
            sample_type = pd.to_numeric(pd.Series([row.get("type", -1)]), errors="coerce").fillna(-1).iloc[0]
            result = {
                "row_index": int(idx),
                "label": int(labels[idx]) if labels[idx] in {0, 1} else -1,
                "type": int(sample_type),
                "p_stop": float(p_stop),
                "p_go": float(p_go),
                "evolution_only_score": evolution_score,
                "full_proehn_score": evolution_score * p_go,
            }
            if "patientID" in self.topology_df.columns:
                result["patientID"] = row.get("patientID")
            if time_col:
                result["survival_time"] = float(pd.to_numeric(pd.Series([row.get(time_col)]), errors="coerce").iloc[0])
            if status_col:
                result["survival_event"] = int(event[idx])
            rows.append(result)
        return pd.DataFrame(rows)

    def _heldout_target_cases(self, random_seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        labels = infer_stop_labels(self.topology_df)
        chooser = random.Random(random_seed)
        go_cases: list[dict[str, Any]] = []
        stop_cases: list[dict[str, Any]] = []

        for idx, row in self.topology_df.iterrows():
            label = labels[idx]
            if label == -1:
                continue
            patient = row.to_dict()
            p_stop = self.kinetic.predict_stop_probability(patient)
            if label == 1:
                stop_cases.append({"stable": True, "p_stop": p_stop, "target_rank": None})
                continue

            active = self._active_gene_indices(idx)
            if not active:
                continue
            target_idx = chooser.choice(active)
            context = [gene_idx for gene_idx in active if gene_idx != target_idx]
            candidates = [gene_idx for gene_idx in range(len(self.artifact.gene_names)) if gene_idx not in context]
            theta = compute_patient_theta(self.artifact, self.X_features[idx])
            scores = []
            for gene_idx in candidates:
                log_rate = theta[gene_idx, gene_idx] + theta[gene_idx, context].sum()
                scores.append((gene_idx, float(log_rate)))
            ranked = [gene_idx for gene_idx, _ in sorted(scores, key=lambda item: item[1], reverse=True)]
            target_rank = ranked.index(target_idx) + 1 if target_idx in ranked else 999
            go_cases.append(
                {
                    "stable": False,
                    "p_stop": p_stop,
                    "target_gene": self.artifact.gene_names[target_idx],
                    "target_rank": int(target_rank),
                }
            )
        return stop_cases, go_cases

    def fig2a_kinetic_gatekeeper(self, noise_levels: Iterable[float] = (0.0, 0.05, 0.10, 0.20, 0.30)) -> dict[str, pd.DataFrame]:
        """Fig.2A PACA pure-data outputs: metrics, robustness, KM and risk summaries."""

        if not self.kinetic.ready:
            raise FileNotFoundError("Fig.2A requires kinetic gatekeeper artifacts.")
        if self.kinetic_data_path is None:
            raise ValueError("Fig.2A requires a PACA feature table via --kinetic-data.")

        df = pd.read_csv(self.kinetic_data_path)
        labels = infer_stop_labels(df)
        valid = labels != -1
        df = df.loc[valid].reset_index(drop=True)
        labels = labels[valid]
        predictions = self._predict_stop_table(df)
        p_stop = predictions["p_stop"].to_numpy()

        metrics = {
            "cohort": "PACA",
            "n_samples": int(len(df)),
            "n_stop": int(np.sum(labels == 1)),
            "n_go": int(np.sum(labels == 0)),
            "auc_roc_stop": float(roc_auc_score(labels, p_stop)) if len(np.unique(labels)) > 1 else float("nan"),
            "auc_pr_stop": float(average_precision_score(labels, p_stop)) if len(np.unique(labels)) > 1 else float("nan"),
            "p_stop_median": float(np.median(p_stop)),
            "p_stop_p10": float(np.percentile(p_stop, 10)),
            "p_stop_p90": float(np.percentile(p_stop, 90)),
        }

        rng = np.random.default_rng(42)
        robustness_rows = []
        genomic_groups = ("pt_genomic", "mt_genomic")
        feature_groups = getattr(self.kinetic, "feature_groups", {})
        for level in noise_levels:
            noisy_df = df.copy()
            for group in genomic_groups:
                for col in feature_groups.get(group, []):
                    if col in noisy_df.columns:
                        mask = rng.random(len(noisy_df)) < float(level)
                        vals = pd.to_numeric(noisy_df[col], errors="coerce").fillna(0).astype(int).to_numpy()
                        vals[mask] = 1 - vals[mask]
                        noisy_df[col] = vals
            noisy_p_stop = self._predict_stop_table(noisy_df)["p_stop"].to_numpy()
            noisy_labels = labels.copy()
            flip_mask = rng.random(len(noisy_labels)) < float(level)
            noisy_labels[flip_mask] = 1 - noisy_labels[flip_mask]
            robustness_rows.append(
                {
                    "noise_level": float(level),
                    "auc_feature_noise": float(roc_auc_score(labels, noisy_p_stop)) if len(np.unique(labels)) > 1 else float("nan"),
                    "auc_label_noise": float(roc_auc_score(noisy_labels, p_stop)) if len(np.unique(noisy_labels)) > 1 else float("nan"),
                }
            )

        time_col, status_col = _survival_columns(df)
        km_tables: list[pd.DataFrame] = []
        survival_summary_rows: list[dict[str, Any]] = []
        decile_summary = pd.DataFrame()
        if time_col and status_col:
            time = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
            event = _event_indicator(df[status_col])
            median = np.median(p_stop)
            high_risk = p_stop <= median
            low_risk = p_stop > median
            km_high = km_curve(time[high_risk], event[high_risk], "High progression risk")
            km_low = km_curve(time[low_risk], event[low_risk], "Low progression risk")
            km_tables.extend([km_high, km_low])
            survival_summary_rows.append(
                {
                    "comparison": "median split by P(Stop)",
                    "n_high_progression_risk": int(high_risk.sum()),
                    "n_low_progression_risk": int(low_risk.sum()),
                    "median_os_high_progression_risk": median_survival(km_high),
                    "median_os_low_progression_risk": median_survival(km_low),
                    "logrank_p": logrank_p_value(time[high_risk], event[high_risk], time[low_risk], event[low_risk]),
                }
            )
            deciles = pd.qcut(p_stop, q=min(10, len(np.unique(p_stop))), duplicates="drop")
            decile_summary = (
                pd.DataFrame({"p_stop": p_stop, "time": time, "event": event, "decile": deciles})
                .groupby("decile", observed=True)
                .agg(n=("p_stop", "size"), p_stop_mean=("p_stop", "mean"), os_median=("time", "median"), event_rate=("event", "mean"))
                .reset_index()
            )
            decile_summary["decile"] = decile_summary["decile"].astype(str)

        return {
            "fig2a_metrics": pd.DataFrame([metrics]),
            "fig2a_robustness": pd.DataFrame(robustness_rows),
            "fig2a_survival_summary": pd.DataFrame(survival_summary_rows),
            "fig2a_km_curve": pd.concat(km_tables, ignore_index=True) if km_tables else pd.DataFrame(),
            "fig2a_partial_effect_proxy": decile_summary,
        }

    def fig3_host_modulation(self, grid_size: int = 31, top_n_features: int = 10, top_n_genes: int = 10) -> dict[str, pd.DataFrame]:
        """Fig.3A-D PACA pure-data outputs."""

        n_genes = len(self.artifact.gene_names)
        seed_idx = n_genes
        tmb_idx = self._feature_index(("nmut", "tmb"))
        age_idx = self._feature_index(("age",))
        if tmb_idx is None:
            raise ValueError("Cannot find a TMB/nMut feature for Fig.3A/D.")

        base = np.zeros(self.artifact.n_feature_rows, dtype=np.float64)
        base[0] = 1.0
        theta_base = compute_patient_theta(self.artifact, base)
        baseline_seed_hazard = float(np.exp(theta_base[seed_idx, seed_idx]))
        tmb_grid = np.linspace(-2.5, 3.5, grid_size)
        age_grid = np.linspace(-2.5, 2.5, grid_size)
        surface_rows = []
        for tmb_z in tmb_grid:
            for age_z in age_grid:
                vec = base.copy()
                vec[tmb_idx] = tmb_z
                if age_idx is not None:
                    vec[age_idx] = age_z
                theta = compute_patient_theta(self.artifact, vec)
                hazard = float(np.exp(theta[seed_idx, seed_idx]))
                surface_rows.append(
                    {
                        "tmb_z": float(tmb_z),
                        "age_z": float(age_z),
                        "seeding_hazard": hazard,
                        "relative_hazard_change_pct": (hazard - baseline_seed_hazard) / baseline_seed_hazard * 100.0,
                    }
                )

        feature_rows = []
        for row_idx, name in enumerate(self.feature_columns, start=1):
            if row_idx >= self.artifact.n_feature_rows:
                break
            raw_seeding = float(self.artifact.W_theta[row_idx, seed_idx, seed_idx])
            raw_progression = float(np.mean(np.diag(self.artifact.W_theta[row_idx, :n_genes, :n_genes])))
            feature_rows.append(
                {
                    "feature": name,
                    "category": classify_feature(name),
                    "raw_seeding_weight": raw_seeding,
                    "raw_local_progression_weight": raw_progression,
                }
            )
        feature_df = pd.DataFrame(feature_rows)
        feature_df["relative_seeding_pct"] = feature_df["raw_seeding_weight"] / (feature_df["raw_seeding_weight"].abs().sum() + 1e-12) * 100.0
        feature_df["relative_local_progression_pct"] = feature_df["raw_local_progression_weight"] / (feature_df["raw_local_progression_weight"].abs().sum() + 1e-12) * 100.0
        contribution = (
            feature_df.groupby("category", as_index=False)
            .agg(
                relative_seeding_pct=("relative_seeding_pct", "sum"),
                relative_local_progression_pct=("relative_local_progression_pct", "sum"),
                n_features=("feature", "count"),
            )
            .sort_values("relative_seeding_pct", key=lambda x: x.abs(), ascending=False)
        )
        top_features = feature_df.reindex(feature_df["relative_seeding_pct"].abs().sort_values(ascending=False).index).head(top_n_features)

        tmb_values = self.X_features[:, tmb_idx]
        delta_tmb = float(np.percentile(tmb_values, 95) - np.percentile(tmb_values, 5))
        matrix = (np.exp(self.artifact.W_theta[tmb_idx, :n_genes, :n_genes] * delta_tmb) - 1.0) * 100.0
        impact = np.sum(np.abs(matrix), axis=0) + np.sum(np.abs(matrix), axis=1)
        top_idx = np.argsort(impact)[::-1][: min(top_n_genes, n_genes)]
        impact_rows = []
        for i in top_idx:
            for j in top_idx:
                impact_rows.append(
                    {
                        "target_event": self.artifact.gene_names[i],
                        "regulator_event": self.artifact.gene_names[j],
                        "tmb_delta_effect_pct": float(matrix[i, j]),
                    }
                )

        return {
            "fig3a_seeding_surface": pd.DataFrame(surface_rows),
            "fig3b_feature_group_contribution": contribution.reset_index(drop=True),
            "fig3c_top_seeding_features": top_features.reset_index(drop=True),
            "fig3d_impact_matrix_long": pd.DataFrame(impact_rows),
        }

    def fig5_trajectories(self, top_k: int = 3, min_probability: float = 0.08) -> dict[str, pd.DataFrame]:
        """Fig.5B-C PACA one-step state exits and multi-step trajectories."""

        one_step_rows = []
        for row_idx in range(len(self.topology_df)):
            genes = {
                self.artifact.gene_names[g_idx]
                for g_idx in range(len(self.artifact.gene_names))
                if self.X_genotypes[row_idx, 2 * g_idx] == 1 or self.X_genotypes[row_idx, 2 * g_idx + 1] == 1
            }
            matched = []
            for state_name, rules in PACA_ANCHOR_STATES.items():
                if not all(gene in genes for gene in rules["include"]):
                    continue
                if any(gene in genes for gene in rules["exclude"]):
                    continue
                matched.append(state_name)
            if not matched:
                continue
            active_idx = [idx for idx, gene in enumerate(self.artifact.gene_names) if gene in genes]
            theta = compute_patient_theta(self.artifact, self.X_features[row_idx])
            rates = accessible_rates(theta, self.artifact.gene_names, active_idx, include_seeding=True)
            probs = normalize_rates(rates)
            top_event = max(probs, key=probs.get)
            for state_name in matched:
                for event, probability in probs.items():
                    one_step_rows.append(
                        {
                            "state": state_name,
                            "row_index": row_idx,
                            "target_event": event,
                            "exit_probability": probability,
                            "raw_rate": rates[event],
                            "is_top_choice": event == top_event,
                        }
                    )

        one_step_raw = pd.DataFrame(one_step_rows)
        if one_step_raw.empty:
            one_step_summary = pd.DataFrame()
        else:
            one_step_summary = (
                one_step_raw.groupby(["state", "target_event"], as_index=False)
                .agg(
                    matched_patients=("row_index", "nunique"),
                    mean_exit_probability=("exit_probability", "mean"),
                    median_exit_probability=("exit_probability", "median"),
                    mean_raw_rate=("raw_rate", "mean"),
                    top_choice_frequency=("is_top_choice", "mean"),
                )
                .sort_values(["state", "mean_exit_probability"], ascending=[True, False])
            )

        baseline = np.zeros(self.artifact.n_feature_rows, dtype=np.float64)
        baseline[0] = 1.0
        theta = compute_patient_theta(self.artifact, baseline)
        queue: list[tuple[tuple[int, ...], str, float, int]] = [(tuple(), "WT", 1.0, 0)]
        trajectory_rows: list[dict[str, Any]] = []
        max_depth = 3
        while queue:
            active, source, path_probability, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            rates = accessible_rates(theta, self.artifact.gene_names, active, include_seeding=False)
            probs = normalize_rates(rates)
            ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)[:top_k]
            for event, transition_probability in ranked:
                if transition_probability < min_probability:
                    continue
                gene_idx = self.artifact.gene_names.index(event)
                new_active = tuple(sorted(set(active) | {gene_idx}))
                target = "+".join(self.artifact.gene_names[idx] for idx in new_active)
                new_path_probability = path_probability * transition_probability
                trajectory_rows.append(
                    {
                        "step": depth + 1,
                        "source_state": source,
                        "target_state": target,
                        "added_event": event,
                        "transition_probability": float(transition_probability),
                        "path_probability": float(new_path_probability),
                    }
                )
                queue.append((new_active, target, new_path_probability, depth + 1))

        return {
            "fig5b_state_exit_probabilities": one_step_summary.reset_index(drop=True),
            "fig5b_state_exit_raw": one_step_raw,
            "fig5c_multistep_trajectories": pd.DataFrame(trajectory_rows),
        }

    def ablation_full_vs_evolution(
        self,
        ratios: Iterable[float] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
        top_ks: Iterable[int] = (1, 2, 3, 4, 5),
        thresholds: Iterable[float] | None = None,
        n_sim: int = 1000,
        random_seed: int = 42,
        min_subgroup_size: int = 10,
    ) -> dict[str, pd.DataFrame]:
        """PACA ablation tables for Full ProEHN versus evolution-only."""

        if thresholds is None:
            thresholds = np.linspace(0.01, 0.99, 50)
        thresholds = tuple(float(value) for value in thresholds)
        ratios = tuple(float(value) for value in ratios)
        top_ks = tuple(int(value) for value in top_ks)
        stop_cases, go_cases = self._heldout_target_cases(random_seed=random_seed)
        rng = np.random.RandomState(random_seed)
        topk_rows = []

        for go_ratio in ratios:
            n_go = int(n_sim * go_ratio)
            n_stop = n_sim - n_go
            batch: list[dict[str, Any]] = []
            if go_cases and n_go > 0:
                batch.extend(rng.choice(go_cases, n_go, replace=True).tolist())
            if stop_cases and n_stop > 0:
                batch.extend(rng.choice(stop_cases, n_stop, replace=True).tolist())

            for top_k in top_ks:
                evolution_hits = [
                    (not item["stable"]) and int(item["target_rank"]) <= top_k
                    for item in batch
                ]
                evolution_accuracy = float(np.mean(evolution_hits)) if batch else float("nan")

                best_accuracy = float("nan")
                best_threshold = float("nan")
                for threshold in thresholds:
                    full_hits = [
                        (
                            item["stable"]
                            and item["p_stop"] > threshold
                        )
                        or (
                            (not item["stable"])
                            and item["p_stop"] <= threshold
                            and int(item["target_rank"]) <= top_k
                        )
                        for item in batch
                    ]
                    accuracy = float(np.mean(full_hits)) if batch else float("nan")
                    if np.isnan(best_accuracy) or accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_threshold = float(threshold)
                topk_rows.append(
                    {
                        "go_ratio": go_ratio,
                        "top_k": top_k,
                        "n_simulated": int(len(batch)),
                        "n_go": n_go,
                        "n_stop": n_stop,
                        "evolution_only_accuracy": evolution_accuracy,
                        "full_proehn_accuracy": best_accuracy,
                        "accuracy_gain": best_accuracy - evolution_accuracy,
                        "full_proehn_threshold": best_threshold,
                    }
                )

        scores = self._patient_scores()
        cindex_rows = []
        if {"survival_time", "survival_event"}.issubset(scores.columns):
            valid = scores["survival_time"].notna()
            valid &= np.isfinite(scores["survival_time"].to_numpy(dtype=float))
            valid &= scores["survival_time"].to_numpy(dtype=float) > 0
            survival_scores = scores.loc[valid].reset_index(drop=True)
            subgroups: dict[str, pd.Series] = {
                "Full Cohort": pd.Series(True, index=survival_scores.index),
                "Type 1": survival_scores["type"] == 1,
                "Type 3": survival_scores["type"] == 3,
            }
            full_score = survival_scores["full_proehn_score"].to_numpy(dtype=float)
            time_all = survival_scores["survival_time"].to_numpy(dtype=float)
            event_all = survival_scores["survival_event"].to_numpy(dtype=int)
            base_ci = harrell_concordance_index(time_all, -full_score, event_all)
            score_sign = -1.0 if base_ci >= 0.5 else 1.0
            for group_name, mask in subgroups.items():
                if int(mask.sum()) < min_subgroup_size:
                    continue
                group = survival_scores.loc[mask]
                time = group["survival_time"].to_numpy(dtype=float)
                event = group["survival_event"].to_numpy(dtype=int)
                for model_name, score_col in [
                    ("evolution-only", "evolution_only_score"),
                    ("Full ProEHN", "full_proehn_score"),
                ]:
                    model_score = group[score_col].to_numpy(dtype=float)
                    cindex_rows.append(
                        {
                            "group": group_name,
                            "model": model_name,
                            "n": int(len(group)),
                            "events": int(event.sum()),
                            "c_index": harrell_concordance_index(time, score_sign * model_score, event),
                        }
                    )

        return {
            "paca_ablation_topk_accuracy": pd.DataFrame(topk_rows),
            "paca_ablation_cindex": pd.DataFrame(cindex_rows),
        }

    def mhn_baseline_benchmark(
        self,
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
        """PACA Oncotrees, HyperTraPS and MHN benchmark tables."""

        return run_paca_mhn_benchmark(
            self.topology_df,
            targets=targets,
            n_context_genes=n_context_genes,
            n_splits=n_splits,
            random_seed=random_seed,
            mhn_lambda=mhn_lambda,
            mhn_maxit=mhn_maxit,
            hypertraps_iterations=hypertraps_iterations,
            hypertraps_walks=hypertraps_walks,
            hypertraps_sigma=hypertraps_sigma,
        )
