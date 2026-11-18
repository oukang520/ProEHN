"""Data preprocessing utilities for ProEHN core training."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping

import jax.numpy as jnp
import numpy as np
import pandas as pd
from .features import (
    FoldPreprocessor,
    EvolutionCovariatePreprocessor,
    TopologyCovariateSchema,
    ALLOWED_COLUMNS,
    validate_genomic_summary_units,
)
from .observations import observation_types
from .labels import build_progression_label


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
    pattern = re.compile(
        rf"^{re.escape(primary_prefix)}(.+?){re.escape(mutation_suffix)}$"
    )
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
    training_df: pd.DataFrame,
    gene_pairs: Iterable[GenePair],
    seeding_column: str = "Seeding",
    core_drivers: Iterable[str] = (),
) -> list[GenePair]:
    """Training-only prevalence, with exact matches to frozen external drivers.

    Seeding is deliberately unused: progression-derived seeding labels cannot
    determine genomic context. Evaluation frames must never be passed here.
    """
    core = set(core_drivers)
    ranked = []
    for pair in gene_pairs:
        frequency = float(
            (
                (training_df[pair.primary] == 1) | (training_df[pair.metastasis] == 1)
            ).sum()
        )
        ranked.append(
            GenePair(
                pair.name,
                pair.primary,
                pair.metastasis,
                frequency,
                0.0,
                float(pair.name in core),
            )
        )
    return sorted(ranked, key=lambda p: (-p.priority, -p.frequency, p.name))


def calculate_marginal_rates(
    buckets: list[tuple[int, int, int, np.ndarray, np.ndarray]],
    n_events: int,
    sample_weights=None,
) -> jnp.ndarray:
    """Warm-start basal hazards from marginal event frequencies."""

    n_total = n_events + 1
    event_counts = np.zeros(n_total)
    total_samples = 0
    known_seeding_samples = 0
    for index, (bucket_type, _, _, bucket_genes, _) in enumerate(buckets):
        weight = (
            np.ones(len(bucket_genes))
            if sample_weights is None
            else np.asarray(sample_weights[index])
        )
        genes = np.asarray(bucket_genes)
        total_samples += weight.sum()
        if bucket_type != 4:
            known_seeding_samples += weight.sum()
        if bucket_type in (0, 1, 2, 4):
            event_counts[:n_events] += (genes[:, :n_events] * weight[:, None]).sum(
                axis=0
            )
            if bucket_type in (1, 2):
                event_counts[n_events] += weight.sum()
        elif bucket_type == 3:
            event_counts[:n_events] += (
                ((genes[:, : 2 * n_events : 2] + genes[:, 1 : 2 * n_events : 2]) > 0)
                * weight[:, None]
            ).sum(axis=0)
            event_counts[n_events] += weight.sum()

    freqs = np.clip(event_counts / (total_samples + 1e-9), 0.01, 0.99)
    # Unknown seeding contributes mutation observations, never seed negatives.
    freqs[-1] = (
        np.clip(event_counts[-1] / known_seeding_samples, 0.01, 0.99)
        if known_seeding_samples
        else 0.5
    )
    return jnp.array(np.log(freqs / (1.0 - freqs)), dtype=jnp.float64)


def select_numeric_feature_columns(
    df: pd.DataFrame,
    excluded_columns: Iterable[str],
    forbidden_features: Iterable[str] = DEFAULT_FORBIDDEN_FEATURES,
    use_bio_pure_features: bool = False,
    schema: TopologyCovariateSchema | None = None,
) -> list[str]:
    """Closed observation-time whitelist, including a closed missingness schema."""
    schema = schema or TopologyCovariateSchema()
    excluded = set(excluded_columns) | set(forbidden_features)
    return [c for c in schema.select(df) if c not in excluded]


def standardize_features(df, feature_columns, mean_impute=False, *, preprocessor=None):
    """Transform only; callers must explicitly fit training preprocessing first."""
    if preprocessor is None or list(preprocessor.columns) != list(feature_columns):
        raise ValueError("Provide a matching training-fitted FoldPreprocessor")
    return preprocessor.transform(df), preprocessor.feature_names


class TopologyTrainingPreprocessor:
    """Use a fixed cohort gene order and fit covariates on training patients."""

    def __init__(
        self,
        top_n_genes=12,
        core_drivers=(),
        schema=None,
        primary_prefix="P.",
        metastasis_prefix="M.",
        mutation_suffix=" (M)",
        *,
        gene_names=None,
        cohort=None,
        no_covariates=False,
    ):
        self.top_n_genes = top_n_genes
        self.core_drivers = core_drivers
        self.schema = schema or TopologyCovariateSchema()
        self.prefixes = (primary_prefix, metastasis_prefix, mutation_suffix)
        self.gene_names = None if gene_names is None else tuple(gene_names)
        self.cohort = cohort
        self.no_covariates = no_covariates

    def fit(self, training_df):
        validate_genomic_summary_units(training_df, require_provenance=False)
        if self.top_n_genes <= 0:
            raise ValueError("top_n_genes must be positive")
        available = find_gene_pairs(training_df.columns, *self.prefixes)
        if self.gene_names is not None:
            by_name = {p.name: p for p in available}
            if (
                len(self.gene_names) != self.top_n_genes
                or len(set(self.gene_names)) != len(self.gene_names)
                or not set(self.gene_names) <= set(by_name)
            ):
                raise ValueError(
                    "Frozen gene order must contain the complete unique cohort event set"
                )
            self.gene_pairs = [by_name[g] for g in self.gene_names]
        else:
            self.gene_pairs = rank_gene_pairs(
                training_df, available, core_drivers=self.core_drivers
            )[: self.top_n_genes]
        if not self.gene_pairs:
            raise ValueError("No paired primary/metastatic genomic columns")
        self.covariates = (
            EvolutionCovariatePreprocessor(
                self.cohort, no_covariates=self.no_covariates
            )
            if self.cohort
            else FoldPreprocessor(self.schema.select(training_df))
        ).fit(training_df)
        return self

    def transform(self, frame):
        validate_genomic_summary_units(frame, require_provenance=False)
        return self.covariates.transform(frame)


def build_topology_training_data(
    df: pd.DataFrame,
    top_n_genes=12,
    max_batch_size=1,
    max_active_events=25,
    core_drivers=(),
    use_bio_pure_features=False,
    seeding_column="Seeding",
    type_column="type",
    diagnosis_order_column="diag_order",
    primary_prefix="P.",
    metastasis_prefix="M.",
    mutation_suffix=" (M)",
    *,
    preprocessor=None,
    covariate_schema=None,
):
    """Build likelihood buckets from TRAINING rows; never silently drop cases.

    Explicit observation_type takes precedence. Legacy 0/1 both marginalize
    seeding; no Stop-derived Seeding field is treated as clinical evidence.
    Missing genotypes in the observed compartment are rejected, not made WT.
    max_active_events is a resource guard, not an undocumented sample filter.
    """
    if max_batch_size < 1 or max_active_events < 1 or not len(df):
        raise ValueError("Positive batch/state limits and nonempty data required")
    prep = preprocessor or TopologyTrainingPreprocessor(
        top_n_genes,
        core_drivers,
        covariate_schema,
        primary_prefix,
        metastasis_prefix,
        mutation_suffix,
    ).fit(df)
    pairs = prep.gene_pairs
    types = observation_types(df, type_column)
    features = np.column_stack((np.ones(len(df)), prep.transform(df)))
    p = (
        df[[g.primary for g in pairs]]
        .apply(pd.to_numeric, errors="raise")
        .to_numpy(float)
    )
    m = (
        df[[g.metastasis for g in pairs]]
        .apply(pd.to_numeric, errors="raise")
        .to_numpy(float)
    )
    if "observed_seeding" in df:
        observed = df["observed_seeding"].to_numpy()
        for i, kind in enumerate(types):
            if (
                not pd.isna(observed[i])
                and kind != 4
                and observed[i] != (0 if kind == 0 else 1)
            ):
                raise ValueError(
                    "Observation type conflicts with independently observed seeding"
                )
    orders = (
        df[diagnosis_order_column].fillna(0).to_numpy()
        if diagnosis_order_column in df
        else np.zeros(len(df))
    )
    groups = {}
    for i, kind in enumerate(types):
        for values in (
            [p[i]] if kind in (0, 1, 4) else [m[i]] if kind == 2 else [p[i], m[i]]
        ):
            if not np.isin(values, [0, 1]).all():
                raise ValueError("Observed genomic calls must be binary and nonmissing")
        if kind in (0, 1, 4):
            data = np.r_[p[i], int(kind == 1)]
            np_, nm_ = int(p[i].sum()) + int(kind == 1), 0
            active = np_ + int(kind == 4)
        elif kind == 2:
            data = np.r_[m[i], 1]
            np_, nm_ = 0, int(m[i].sum()) + 1
            active = nm_
        else:
            first_seed = (
                df["seeding_at_first_observation"].iloc[i]
                if "seeding_at_first_observation" in df
                else -1
            )
            first_seed = -1 if pd.isna(first_seed) else first_seed
            if first_seed not in (-1, 0, 1) or (orders[i] == 2 and first_seed == 0):
                raise ValueError(
                    "MT-first observation requires seeding; invalid first-observation seeding annotation"
                )
            if first_seed == 0 and np.any(p[i] > m[i]):
                raise ValueError(
                    "Pre-seeding primary mutations must be inherited by the metastatic lineage"
                )
            if orders[i] not in (0, 1, 2):
                raise ValueError(
                    "Diagnosis order must be 0 unknown, 1 PT-first or 2 MT-first"
                )
            data = np.r_[
                np.column_stack((p[i], m[i])).ravel(), 1, first_seed, orders[i]
            ]
            # Each single compartment count INCLUDES the shared seed bit.
            np_, nm_ = int(p[i].sum()) + 1, int(m[i].sum()) + 1
            active = np_ + nm_ - 1
        if active > max_active_events:
            raise ValueError(
                "State exceeds max_active_events; explicitly resolve inclusion before fitting"
            )
        groups.setdefault((int(kind), np_, nm_), []).append(
            (data.astype(np.int8), features[i])
        )
    buckets = []
    for key, rows in groups.items():
        for start in range(0, len(rows), max_batch_size):
            batch = rows[start : start + max_batch_size]
            buckets.append(
                (*key, np.stack([r[0] for r in batch]), np.stack([r[1] for r in batch]))
            )
    return (
        buckets,
        len(pairs),
        len(prep.covariates.feature_names),
        len(df),
        [g.name for g in pairs],
        prep.covariates.feature_names,
    )


def split_kinetic_feature_columns(all_columns):
    """Exact baseline whitelist plus genomic indicators; no broad missingness rule."""
    groups = {
        k: []
        for k in ("pt_genomic", "mt_genomic", "pt_dynamic", "mt_dynamic", "shared")
    }
    for col in all_columns:
        if re.fullmatch(r"[PM]\.[A-Za-z0-9_-]+ \(M\)", col):
            groups["pt_genomic" if col.startswith("P.") else "mt_genomic"].append(col)
        elif col in ALLOWED_COLUMNS:
            group = (
                "pt_dynamic"
                if "Primary" in col or col.startswith("P.")
                else (
                    "mt_dynamic"
                    if "Metastatic" in col or col.startswith("M.")
                    else "shared"
                )
            )
            groups[group].append(col)
    return groups


def generate_stop_label(patient_row, params, pt_driver_cols, mt_driver_cols):
    raise ValueError(
        "Genomic proxy labels are retired; use an explicit ProgressionLabelSchema with raw days and censoring"
    )


class KineticPreprocessor:
    def __init__(self, covariates=None, *, masked_columns=(), gene_names=None):
        self.covariates = None if covariates is None else tuple(covariates)
        self.masked_columns = tuple(masked_columns)
        self.gene_names = None if gene_names is None else tuple(gene_names)

    def fit(self, training_df):
        validate_genomic_summary_units(training_df, require_provenance=False)
        columns = training_df.columns
        if self.gene_names is not None:
            genomic = {f"{p}.{g} (M)" for p in ("P", "M") for g in self.gene_names}
            columns = [
                c
                for c in columns
                if not re.fullmatch(r"[PM]\.[A-Za-z0-9_-]+ \(M\)", c) or c in genomic
            ]
            columns = [
                f"{p}.{g} (M)"
                for p in ("P", "M")
                for g in self.gene_names
                if f"{p}.{g} (M)" in columns
            ] + [c for c in columns if c not in genomic]
        if self.covariates is not None:
            if not set(self.covariates) <= set(columns):
                raise ValueError("Frozen kinetic covariates unavailable")
            columns = [
                c
                for c in columns
                if c in self.covariates
                or re.fullmatch(r"[PM]\.[A-Za-z0-9_-]+ \(M\)", c)
            ]
        self.feature_groups = split_kinetic_feature_columns(columns)
        masked = training_df.copy()
        for column in self.masked_columns:
            if column not in masked:
                raise ValueError("The target-masked clinical column is unavailable")
            masked[column] = np.nan
        self.transforms = {
            k: FoldPreprocessor(v).fit(masked) for k, v in self.feature_groups.items()
        }
        return self

    def transform(self, frame):
        validate_genomic_summary_units(frame, require_provenance=False)
        masked = frame.copy()
        for column in getattr(self, "masked_columns", ()):
            masked[column] = np.nan
        return {
            k: t.transform(masked).astype(np.float32)
            for k, t in self.transforms.items()
        }


def build_kinetic_matrices(
    df,
    label_column="Patient_Label",
    label_params=None,
    *,
    preprocessor=None,
    label_schema=None,
):
    """Transform a partition using preprocessing already fitted on training rows."""
    if preprocessor is None:
        raise ValueError(
            "Split patients first; provide training-fitted KineticPreprocessor"
        )
    if label_schema is not None:
        labels = build_progression_label(df, label_schema)
    elif label_column in df:
        raw = pd.to_numeric(df[label_column], errors="raise").fillna(-1).to_numpy()
        if not np.isin(raw, [-1, 0, 1]).all():
            raise ValueError(
                "Labels must be -1 unknown, 0 operational Go, 1 operational Stop"
            )
        labels = raw.astype(int)
    else:
        raise ValueError(
            "Explicit label protocol is required; genomic proxy fallback is forbidden"
        )
    return (
        preprocessor.transform(df),
        labels,
        {"preprocessor": preprocessor, "feature_groups": preprocessor.feature_groups},
    )
