"""Feature-table assembly utilities for ProEHN core training."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DEFAULT_FORBIDDEN_FEATURES = {
    "patientID",
    "patient_id",
    "Patient_ID",
    "cgc_donor_id",
    "icgc_donor_id",
    "icgc_specimen_id",
    "label",
    "Patient_Label",
    "Seeding",
    "type",
    "diag_order",
    "PT_label",
    "MT_label",
    "paired",
    "PT_label_last_followup",
    "MT_label_last_followup",
    "donor_vital_status",
    "disease_status_last_followup",
    "PFS_days",
    "OS_days",
    "response",
    "Unnamed: 0",
}


@dataclass(frozen=True)
class GenePair:
    """Matched primary/metastatic mutation columns for one canonical event."""

    name: str
    primary: str
    metastasis: str
    frequency: float = 0.0
    relevance: float = 0.0
    priority: float = 0.0

    @property
    def score(self) -> tuple[float, float]:
        return self.priority + self.relevance, self.frequency


def find_gene_pairs(
    columns: Iterable[str],
    primary_prefix: str = "P.",
    metastasis_prefix: str = "M.",
    mutation_suffix: str = " (M)",
) -> list[GenePair]:
    """Find P.gene and M.gene mutation-column pairs."""

    columns = list(columns)
    column_set = set(columns)
    pattern = re.compile(rf"^{re.escape(primary_prefix)}(.+?){re.escape(mutation_suffix)}$")
    pairs: list[GenePair] = []
    for col in columns:
        match = pattern.match(col)
        if not match:
            continue
        gene = match.group(1)
        m_col = f"{metastasis_prefix}{gene}{mutation_suffix}"
        if m_col in column_set:
            pairs.append(GenePair(gene, col, m_col))
    return pairs


def rank_gene_pairs(
    df: pd.DataFrame,
    gene_pairs: Iterable[GenePair],
    seeding_column: str = "Seeding",
    core_drivers: Iterable[str] = (),
) -> list[GenePair]:
    """Rank genes by core-driver priority, seeding relevance and mutation frequency."""

    core_drivers = tuple(core_drivers)
    seeding = (
        df[seeding_column].fillna(0).astype(int)
        if seeding_column in df.columns
        else pd.Series(0, index=df.index)
    )
    ranked: list[GenePair] = []
    for pair in gene_pairs:
        is_mutated = (
            (df[pair.primary].fillna(0) == 1)
            | (df[pair.metastasis].fillna(0) == 1)
        ).astype(int)
        frequency = float(is_mutated.sum())
        if frequency > 0 and seeding.nunique() > 1 and is_mutated.std() > 1e-9:
            corr = seeding.corr(is_mutated)
            relevance = float(abs(corr)) if not np.isnan(corr) else 0.0
        else:
            relevance = 0.0
        priority = 100.0 if any(driver in pair.name for driver in core_drivers) else 0.0
        ranked.append(
            GenePair(pair.name, pair.primary, pair.metastasis, frequency, relevance, priority)
        )
    return sorted(ranked, key=lambda item: item.score, reverse=True)


def calculate_marginal_rates(
    buckets: list[tuple[int, int, int, np.ndarray, np.ndarray]],
    n_events: int,
) -> jnp.ndarray:
    """Warm-start basal hazards from marginal event frequencies."""

    n_total = n_events + 1
    event_counts = np.zeros(n_total)
    total_samples = 0
    for bucket_type, _, _, bucket_genes, _ in buckets:
        genes = np.asarray(bucket_genes)
        batch_len = genes.shape[0]
        total_samples += batch_len
        if bucket_type == 1:
            event_counts[:n_events] += (genes == 1).sum(axis=0)
            event_counts[n_events] += batch_len
        elif bucket_type == 2:
            event_counts[:n_events] += (genes[:, :-1] == 1).sum(axis=0)
            event_counts[n_events] += (genes[:, -1] == 1).sum(axis=0)
        elif bucket_type == 3:
            genes_only = genes[:, : 2 * n_events]
            is_mutated = (genes_only[:, 0::2] + genes_only[:, 1::2]) > 0
            event_counts[:n_events] += is_mutated.sum(axis=0)
            event_counts[n_events] += batch_len

    freqs = np.clip(event_counts / (total_samples + 1e-9), 0.01, 0.99)
    return jnp.array(np.log(freqs / (1.0 - freqs)), dtype=jnp.float64)


def select_numeric_feature_columns(
    df: pd.DataFrame,
    excluded_columns: Iterable[str],
    forbidden_features: Iterable[str] = DEFAULT_FORBIDDEN_FEATURES,
    use_bio_pure_features: bool = False,
) -> list[str]:
    """Select non-leaking numeric covariates for topology modulation."""

    excluded = set(excluded_columns)
    forbidden = set(forbidden_features)
    selected: list[str] = []
    for col in df.columns:
        col_lower = col.lower()
        if col in excluded or col in forbidden:
            continue
        if col.startswith("P.") or col.startswith("M."):
            continue
        if any(token in col_lower for token in ["disease", "response", "label", "vital_status"]):
            continue
        if use_bio_pure_features and "_is_Missing" in col:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().any():
            selected.append(col)
    return selected


def standardize_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    mean_impute: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Return z-scored covariates without the intercept column."""

    if not feature_columns:
        return np.zeros((len(df), 1), dtype=np.float64), ["Dummy"]
    features = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    features = features.fillna(features.mean()).fillna(0.0) if mean_impute else features.fillna(0.0)
    values = features.to_numpy(dtype=np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std == 0] = 1.0
    return (values - mean) / std, feature_columns


def build_topology_training_data(
    df: pd.DataFrame,
    top_n_genes: int = 20,
    max_batch_size: int = 1,
    max_active_events: int = 18,
    core_drivers: Iterable[str] = (),
    use_bio_pure_features: bool = False,
    seeding_column: str = "Seeding",
    type_column: str = "type",
    diagnosis_order_column: str = "diag_order",
    primary_prefix: str = "P.",
    metastasis_prefix: str = "M.",
    mutation_suffix: str = " (M)",
) -> tuple[list[tuple[int, int, int, np.ndarray, np.ndarray]], int, int, int, list[str], list[str]]:
    """Create homogeneous CTMC likelihood buckets from cross-sectional data."""

    if type_column not in df.columns:
        df = df.copy()
        df[type_column] = 3
    if seeding_column not in df.columns:
        df = df.copy()
        df[seeding_column] = 0
    if diagnosis_order_column not in df.columns:
        df = df.copy()
        df[diagnosis_order_column] = 1

    gene_pairs = rank_gene_pairs(
        df,
        find_gene_pairs(df.columns, primary_prefix, metastasis_prefix, mutation_suffix),
        seeding_column=seeding_column,
        core_drivers=core_drivers,
    )[:top_n_genes]
    if not gene_pairs:
        raise ValueError("No paired primary/metastatic mutation columns were found.")

    selected_gene_cols = [col for pair in gene_pairs for col in (pair.primary, pair.metastasis)]
    feature_cols = select_numeric_feature_columns(
        df,
        excluded_columns=selected_gene_cols,
        use_bio_pure_features=use_bio_pure_features,
    )
    features_norm, feature_names = standardize_features(
        df,
        feature_cols,
        mean_impute=use_bio_pure_features,
    )
    features_with_bias = np.hstack([np.ones((len(df), 1), dtype=np.float64), features_norm])

    gene_data = (
        df[selected_gene_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.int8)
    )
    diag_orders = df[diagnosis_order_column].fillna(1).to_numpy(dtype=np.int8)
    types = df[type_column].fillna(3).to_numpy(dtype=np.int8)
    n_events = len(gene_pairs)
    buckets: list[tuple[int, int, int, np.ndarray, np.ndarray]] = []
    skipped = 0

    def add_to_bucket(
        bucket_type: int,
        n_primary: int,
        n_metastatic: int,
        data: np.ndarray,
        feats: np.ndarray,
    ) -> None:
        nonlocal skipped
        if (n_primary + n_metastatic) > max_active_events:
            skipped += len(data)
            return
        for start in range(0, data.shape[0], max_batch_size):
            end = min(start + max_batch_size, data.shape[0])
            buckets.append(
                (
                    bucket_type,
                    int(n_primary),
                    int(n_metastatic),
                    data[start:end],
                    feats[start:end],
                )
            )

    if np.any(types == 1):
        mask = types == 1
        sub_g = gene_data[mask]
        sub_f = features_with_bias[mask]
        n_p = (sub_g[:, 0::2] == 1).sum(axis=1)
        for n in np.unique(n_p):
            if n > 0:
                add_to_bucket(1, int(n), 0, sub_g[n_p == n][:, 0::2], sub_f[n_p == n])

    if np.any(types == 2):
        mask = types == 2
        sub_g = gene_data[mask]
        sub_f = features_with_bias[mask]
        mt_full = np.column_stack([sub_g[:, 1::2], np.ones(mask.sum(), dtype=np.int8)])
        n_m = (mt_full == 1).sum(axis=1)
        for n in np.unique(n_m):
            add_to_bucket(2, 0, int(n), mt_full[n_m == n], sub_f[n_m == n])

    if np.any(types == 3):
        mask = types == 3
        sub_g = gene_data[mask]
        sub_f = features_with_bias[mask]
        sub_o = diag_orders[mask]
        joint = np.column_stack([sub_g, np.ones(mask.sum(), dtype=np.int8)])
        n_p = (sub_g[:, 0::2] == 1).sum(axis=1)
        n_m = (sub_g[:, 1::2] == 1).sum(axis=1) + 1
        for n_primary, n_metastatic in np.unique(np.column_stack([n_p, n_m]), axis=0):
            if n_primary <= 0:
                skipped += int((n_p == n_primary).sum())
                continue
            idx = (n_p == n_primary) & (n_m == n_metastatic)
            add_to_bucket(
                3,
                int(n_primary),
                int(n_metastatic),
                np.column_stack([joint[idx], sub_o[idx]]),
                sub_f[idx],
            )

    gene_names = [pair.name for pair in gene_pairs]
    n_samples = len(df) - skipped
    return buckets, n_events, len(feature_names), n_samples, gene_names, feature_names


def split_kinetic_feature_columns(all_columns: Iterable[str]) -> dict[str, list[str]]:
    """Split columns into primary, metastatic and shared gatekeeper inputs."""

    groups = {"pt_genomic": [], "mt_genomic": [], "pt_dynamic": [], "mt_dynamic": [], "shared": []}
    dynamic_tokens = ("ageatseqrep", "vaf_mean", "nmut", "cna", "fga")
    shared_tokens = (
        "age_at_diagnosis",
        "sex_female",
        "sex_male",
        "sex_unknown",
        "sex_nan",
        "paired",
    )
    label_tokens = ("label", "status", "pfs", "os", "vital", "response")

    for col in all_columns:
        low = col.lower()
        if any(token in low for token in label_tokens):
            continue
        if col.startswith("P.") and " (M)" in col:
            groups["pt_genomic"].append(col)
        elif col.startswith("M.") and " (M)" in col:
            groups["mt_genomic"].append(col)
        elif col.startswith("P.") or "primary" in low:
            if any(token in low for token in dynamic_tokens) or "is_missing" in low:
                groups["pt_dynamic"].append(col)
        elif col.startswith("M.") or "metastatic" in low:
            if any(token in low for token in dynamic_tokens) or "is_missing" in low:
                groups["mt_dynamic"].append(col)
        elif any(token in low for token in shared_tokens) or "is_missing" in low:
            groups["shared"].append(col)
    return groups


def build_kinetic_matrices(
    df: pd.DataFrame,
    label_column: str = "Patient_Label",
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """Build gatekeeper matrices and fitted scalers from a feature table."""

    groups = split_kinetic_feature_columns(df.columns)
    if label_column not in df.columns:
        raise ValueError(f"Kinetic training table is missing label column: {label_column}")
    labels = df[label_column].fillna(-1).astype(int).to_numpy()

    matrices = {
        "pt_genomic": df[groups["pt_genomic"]].fillna(0).to_numpy(dtype=np.float32),
        "mt_genomic": df[groups["mt_genomic"]].fillna(0).to_numpy(dtype=np.float32),
        "pt_dynamic": (
            df[groups["pt_dynamic"]]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .to_numpy(dtype=np.float32)
        ),
        "mt_dynamic": (
            df[groups["mt_dynamic"]]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .to_numpy(dtype=np.float32)
        ),
        "shared": (
            df[groups["shared"]]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .to_numpy(dtype=np.float32)
        ),
    }
    scalers: dict[str, StandardScaler] = {}
    for group in ["pt_dynamic", "mt_dynamic", "shared"]:
        scaler = StandardScaler()
        matrices[group] = scaler.fit_transform(matrices[group])
        scalers[group] = scaler

    metadata = {"scalers": scalers, "feature_groups": groups}
    return matrices, labels, metadata
