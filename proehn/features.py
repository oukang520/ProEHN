"""Observation-time covariates, feature provenance and fold-fitted transforms."""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from .burden import GenomicBurdenProtocol

# Closed vocabulary. Missingness is allowed only for these baseline measurements.
BASELINE_COLUMNS = (
    "age",
    "background_nMut",
    "background_VAF_mean",
    "Age_at_Diagnosis",
    "P.AgeAtSeqRep",
    "M.AgeAtSeqRep",
    "Sex_Female",
    "Sex_Male",
    "Sex_Unknown",
    "nMut_Primary",
    "nMut_Metastatic",
    "VAF_mean_Primary",
    "VAF_mean_Metastatic",
    "CNA_Primary",
    "CNA_Metastatic",
    "FGA_Primary",
    "FGA_Metastatic",
    "TMB_Primary",
    "TMB_Metastatic",
    "CNA_adjusted_Primary",
    "CNA_adjusted_Metastatic",
)
ALLOWED_COLUMNS = BASELINE_COLUMNS + tuple(c + "_is_Missing" for c in BASELINE_COLUMNS)


@dataclass(frozen=True)
class TopologyCovariateSchema:
    columns: tuple[str, ...] = BASELINE_COLUMNS

    def __post_init__(self):
        if len(set(self.columns)) != len(self.columns) or not set(self.columns) <= set(
            ALLOWED_COLUMNS
        ):
            raise ValueError("Only registered observation-time covariates are allowed")

    def select(self, frame):
        return [c for c in self.columns if c in frame]


class FoldPreprocessor:
    """Fit imputation, variance filter and scaling on training rows only.

    Transform never fits or adapts to an evaluation batch. Empty features produce
    shape (n, 0), leaving the topology intercept to the model. PCA can be composed
    downstream by fitting sklearn PCA on transform(training), never on OOF rows.
    """

    def __init__(self, columns, *, mean_impute=True, variance_threshold=None):
        self.columns = tuple(columns)
        self.mean_impute = mean_impute
        self.variance_threshold = variance_threshold
        self.fitted = False

    def _raw(self, frame):
        if not set(self.columns) <= set(frame):
            raise ValueError("Missing required feature columns")
        raw = (
            frame[list(self.columns)]
            .apply(pd.to_numeric, errors="raise")
            .to_numpy(float)
        )
        if np.isinf(raw).any():
            raise ValueError("Infinite covariates are invalid")
        return raw

    def fit(self, training_df):
        if not len(training_df):
            raise ValueError("Empty training partition")
        raw = self._raw(training_df)
        counts = np.isfinite(raw).sum(axis=0)
        self.fill_ = (
            np.divide(
                np.nansum(raw, axis=0),
                counts,
                out=np.zeros(len(self.columns)),
                where=counts > 0,
            )
            if self.mean_impute
            else np.zeros(len(self.columns))
        )
        raw = np.where(np.isnan(raw), self.fill_, raw)
        variance = raw.var(axis=0)
        self.keep_ = (
            np.ones(len(self.columns), dtype=bool)
            if self.variance_threshold is None
            else variance > self.variance_threshold
        )
        self.mean_ = raw[:, self.keep_].mean(axis=0)
        self.scale_ = raw[:, self.keep_].std(axis=0)
        self.scale_[self.scale_ == 0] = 1
        self.n_training_rows_ = len(training_df)
        self.fitted = True
        return self

    @property
    def feature_names(self):
        if not self.fitted:
            raise ValueError("Preprocessor is not fitted")
        return [c for c, keep in zip(self.columns, self.keep_) if keep]

    def transform(self, frame):
        if not self.fitted:
            raise ValueError("Fit on training partition before transform")
        raw = self._raw(frame)
        return (
            np.where(np.isnan(raw), self.fill_, raw)[:, self.keep_] - self.mean_
        ) / self.scale_

    def metadata(self):
        if not self.fitted:
            raise ValueError("Preprocessor is not fitted")
        return dict(
            columns=list(self.columns),
            fill=self.fill_.tolist(),
            keep=self.keep_.tolist(),
            mean=self.mean_.tolist(),
            scale=self.scale_.tolist(),
            version=1,
        )

    @classmethod
    def from_metadata(cls, metadata):
        if metadata.get("kind") == "evolution-covariates-v1":
            return EvolutionCovariatePreprocessor.from_metadata(metadata)
        if metadata["version"] != 1:
            raise ValueError("Unsupported preprocessing version")
        obj = cls(metadata["columns"])
        obj.fill_ = np.asarray(metadata["fill"], float)
        obj.keep_ = np.asarray(metadata["keep"], bool)
        obj.mean_ = np.asarray(metadata["mean"], float)
        obj.scale_ = np.asarray(metadata["scale"], float)
        if (
            obj.fill_.shape != (len(obj.columns),)
            or obj.keep_.shape != obj.fill_.shape
            or obj.mean_.shape != (int(obj.keep_.sum()),)
            or obj.scale_.shape != obj.mean_.shape
            or not all(np.isfinite(v).all() for v in [obj.fill_, obj.mean_, obj.scale_])
            or np.any(obj.scale_ <= 0)
        ):
            raise ValueError("Invalid preprocessing metadata")
        obj.fitted = True
        return obj


class EvolutionCovariatePreprocessor:
    """The manuscript's training-fitted evolutionary context transforms.

    Input summaries exclude all modeled genes before this transform is fitted.
    GENIE uses age, log1p background count, mean VAF and two missing indicators.
    ICGC fits winsorization/imputation/scaling to all three measurements, then
    retains background count and VAF. Clinical-gate preprocessing is separate.
    """

    columns = ("age", "background_nMut", "background_VAF_mean")

    def __init__(self, cohort, *, no_covariates=False):
        self.cohort = str(cohort).upper()
        if self.cohort not in ("GENIE", "MELA", "PACA"):
            raise ValueError("Evolutionary preprocessing requires GENIE, MELA or PACA")
        self.no_covariates = bool(no_covariates)
        self.fitted = False

    def _raw(self, frame):
        if not set(self.columns) <= set(frame):
            raise ValueError(
                "Evolutionary input requires age, background_nMut and background_VAF_mean"
            )
        raw = (
            frame[list(self.columns)]
            .apply(pd.to_numeric, errors="raise")
            .to_numpy(float)
        )
        count = raw[:, 1]
        observed_count = count[np.isfinite(count)]
        if (
            np.isinf(raw).any()
            or np.any(count < 0)
            or np.any(observed_count != np.floor(observed_count))
            or np.any((raw[:, 2] < 0) | (raw[:, 2] > 1))
        ):
            raise ValueError(
                "Invalid age, nonnegative background count or VAF fraction"
            )
        raw[:, 1] = np.log1p(raw[:, 1])
        return raw

    def _values(self, raw):
        clipped = (
            raw
            if self.cohort == "GENIE"
            else np.clip(raw, self.bounds_[0], self.bounds_[1])
        )
        values = np.where(np.isnan(clipped), self.fill_, clipped)
        if self.cohort == "GENIE":
            values = np.column_stack((values, np.isnan(raw[:, 0]), np.isnan(raw[:, 2])))
        return values

    def fit(self, training_df):
        if not len(training_df):
            raise ValueError("Empty training partition")
        raw = self._raw(training_df)
        # Entirely missing measurements are explicitly zero, without nan warnings.
        quantiles = [
            (
                np.quantile(c[np.isfinite(c)], [0.01, 0.99])
                if np.isfinite(c).any()
                else np.zeros(2)
            )
            for c in raw.T
        ]
        self.bounds_ = np.asarray(quantiles).T
        clipped = raw if self.cohort == "GENIE" else np.clip(raw, *self.bounds_)
        self.fill_ = np.asarray(
            [
                np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.0
                for c in clipped.T
            ]
        )
        values = self._values(raw)
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0, ddof=0)
        self.scale_[
            self.scale_ < (1e-8 if self.cohort != "GENIE" else np.finfo(float).tiny)
        ] = 1.0
        self.n_training_rows_ = len(training_df)
        self.fitted = True
        return self

    @property
    def selected_columns(self):
        return (
            []
            if self.no_covariates
            else list(range(5)) if self.cohort == "GENIE" else [1, 2]
        )

    @property
    def feature_names(self):
        names = [
            "age",
            "log1p_background_nMut",
            "background_VAF_mean",
            "age_is_Missing",
            "background_VAF_mean_is_Missing",
        ]
        return [names[j] for j in self.selected_columns]

    def transform(self, frame):
        if not self.fitted:
            raise ValueError(
                "Fit evolutionary preprocessing on training patients first"
            )
        values = (self._values(self._raw(frame)) - self.mean_) / self.scale_
        return values[:, self.selected_columns]

    def metadata(self):
        if not self.fitted:
            raise ValueError("Preprocessor is not fitted")
        return dict(
            kind="evolution-covariates-v1",
            version=1,
            cohort=self.cohort,
            columns=list(self.columns),
            no_covariates=self.no_covariates,
            bounds=self.bounds_.tolist(),
            fill=self.fill_.tolist(),
            mean=self.mean_.tolist(),
            scale=self.scale_.tolist(),
            n_training_rows=getattr(self, "n_training_rows_", None),
        )

    @classmethod
    def from_metadata(cls, metadata):
        if metadata.get("version") != 1 or tuple(metadata["columns"]) != cls.columns:
            raise ValueError("Unsupported evolutionary preprocessing metadata")
        obj = cls(
            metadata["cohort"], no_covariates=metadata.get("no_covariates", False)
        )
        for name in ("bounds", "fill", "mean", "scale"):
            setattr(obj, name + "_", np.asarray(metadata[name], float))
        width = 5 if obj.cohort == "GENIE" else 3
        if (
            obj.bounds_.shape != (2, 3)
            or obj.fill_.shape != (3,)
            or obj.mean_.shape != (width,)
            or obj.scale_.shape != (width,)
            or not all(
                np.isfinite(x).all()
                for x in (obj.bounds_, obj.fill_, obj.mean_, obj.scale_)
            )
            or np.any(obj.scale_ <= 0)
            or np.any(obj.bounds_[0] > obj.bounds_[1])
        ):
            raise ValueError("Invalid evolutionary preprocessing metadata")
        obj.n_training_rows_ = metadata.get("n_training_rows")
        obj.fitted = True
        return obj

    @classmethod
    def from_historical(cls, cohort, metadata, *, no_covariates=False):
        """Convert frozen published transforms without refitting them."""
        obj = cls(cohort, no_covariates=no_covariates)
        for name in ("fill", "mean", "scale"):
            setattr(obj, name + "_", np.asarray(metadata[name], float))
        obj.bounds_ = np.asarray(metadata.get("bounds", np.zeros((2, 3))), float)
        obj.fitted = True
        return cls.from_metadata(obj.metadata())


@dataclass(frozen=True)
class FeatureMetadata:
    name: str
    raw_unit: str
    transformation: str
    normalization: str
    valid_range: tuple[float, float]
    biological_meaning: str

    def validate(self, values, *, unit):
        a = np.asarray(values, float)
        if (
            unit != self.raw_unit
            or not np.isfinite(a).all()
            or np.any(a < self.valid_range[0])
            or np.any(a > self.valid_range[1])
        ):
            raise ValueError(f"{self.name}: invalid units or range")


FEATURE_METADATA = {
    "FGA": FeatureMetadata(
        "FGA",
        "fraction",
        "altered length divided by assayed length",
        "sample-specific assayed Mb",
        (0, 1),
        "Fraction of assayed genome altered",
    ),
    "CNA_adjusted": FeatureMetadata(
        "CNA_adjusted",
        "fraction",
        "altered loci divided by assayed loci",
        "sample-specific assayed loci",
        (0, 1),
        "Coverage-adjusted fraction of assayed loci altered",
    ),
    "nMut": FeatureMetadata(
        "nMut",
        "mutation count",
        "none",
        "none",
        (0, np.inf),
        "Count of observed variant loci; not mutations per Mb",
    ),
    "VAF_mean": FeatureMetadata(
        "VAF_mean",
        "fraction",
        "mean over observed mutant loci",
        "none",
        (0, 1),
        "Mean observed variant allele fraction",
    ),
    "CNA": FeatureMetadata(
        "CNA",
        "altered locus count",
        "sum binary locus calls",
        "none",
        (0, np.inf),
        "Count of altered loci, not FGA",
    ),
    "TMB": FeatureMetadata(
        "TMB",
        "mutations/Mb",
        "count divided by callable coverage",
        "sample-specific callable Mb",
        (0, np.inf),
        "Panel coverage normalized burden; assay eligibility must be harmonized",
    ),
}


def panel_corrected_mutation_burden(mutation_count, callable_mb):
    count = np.asarray(mutation_count, float)
    FEATURE_METADATA["nMut"].validate(count, unit="mutation count")
    if np.any(count != np.floor(count)):
        raise ValueError("Raw mutation counts must be integral")
    if callable_mb is None:
        raise ValueError(
            "REQUIRES_DATA_REPROCESSING_BEFORE_RERUN: callable panel coverage unavailable"
        )
    mb = np.asarray(callable_mb, float)
    if not np.isfinite(mb).all() or np.any(mb <= 0):
        raise ValueError("Callable coverage must be positive Mb")
    return count / mb


@dataclass(frozen=True)
class TargetMaskingProtocol:
    """Complete raw locus provenance, frozen independently of evaluation labels.

    Every output is reconstructed from registered baseline fields or raw loci.
    Unregistered columns and all precomputed summaries are dropped. Event-to-locus
    mappings must include ALL variants/CNAs belonging to the target definition.
    raw mutation indicators are locus calls (one per variant), not gene presence.
    """

    mutation_columns: tuple[str, ...]
    vaf_columns: tuple[str, ...]
    cna_columns: tuple[str, ...]
    target_mutation_columns: tuple[str, ...]
    target_cna_columns: tuple[str, ...]
    baseline_columns: tuple[str, ...] = ()
    compartment: str = "Primary"
    burden_protocol: GenomicBurdenProtocol | None = None
    cna_segment_length_columns: tuple[str, ...] = ()
    cna_segments_nonoverlapping: bool = False
    nonsynonymous_mutation_columns: tuple[str, ...] | None = None

    def __post_init__(self):
        if len(self.mutation_columns) != len(self.vaf_columns):
            raise ValueError("One VAF column per mutation locus is required")
        if self.nonsynonymous_mutation_columns is not None and (
            not set(self.nonsynonymous_mutation_columns) <= set(self.mutation_columns)
            or len(set(self.nonsynonymous_mutation_columns))
            != len(self.nonsynonymous_mutation_columns)
        ):
            raise ValueError(
                "Nonsynonymous loci must be unique registered mutation loci"
            )
        if (
            self.burden_protocol is not None
            and self.burden_protocol.mutation_definition
            == "nonsynonymous_mutation_count"
            and self.nonsynonymous_mutation_columns is None
        ):
            raise ValueError("Explicit nonsynonymous mutation annotation required")
        if not set(self.target_mutation_columns) <= set(
            self.mutation_columns
        ) or not set(self.target_cna_columns) <= set(self.cna_columns):
            raise ValueError("Target loci absent from provenance")
        if not self.target_mutation_columns and not self.target_cna_columns:
            raise ValueError("Target definition must contain at least one locus")
        # Only non-genomic baseline covariates may bypass reconstruction.
        safe = {
            "Age_at_Diagnosis",
            "P.AgeAtSeqRep",
            "M.AgeAtSeqRep",
            "Sex_Female",
            "Sex_Male",
            "Sex_Unknown",
        }
        if not set(self.baseline_columns) <= safe or self.compartment not in {
            "Primary",
            "Metastatic",
        }:
            raise ValueError("Unverified target-derived or future covariate")
        if (
            self.burden_protocol is not None
            and self.burden_protocol.cna_representation == "fga"
        ):
            if (
                len(self.cna_segment_length_columns) != len(self.cna_columns)
                or not self.cna_segments_nonoverlapping
            ):
                raise ValueError(
                    "FGA requires explicit nonoverlapping segment lengths aligned with CNA calls"
                )
        all_cols = (
            self.mutation_columns
            + self.vaf_columns
            + self.cna_columns
            + self.cna_segment_length_columns
        )
        if len(set(all_cols)) != len(all_cols):
            raise ValueError("Duplicate raw locus provenance")


def build_leave_target_out_features(raw_df, protocol):
    p = protocol
    mut = raw_df[list(p.mutation_columns)].to_numpy(float)
    cna = raw_df[list(p.cna_columns)].to_numpy(float)
    vaf = raw_df[list(p.vaf_columns)].to_numpy(float)
    if not np.isin(mut, [0, 1]).all() or not np.isin(cna, [0, 1]).all():
        raise ValueError("Raw locus calls must be observed binary values")
    observed_vaf = vaf[mut == 1]
    FEATURE_METADATA["VAF_mean"].validate(observed_vaf, unit="fraction")
    keep_m = np.array(
        [c not in p.target_mutation_columns for c in p.mutation_columns], bool
    )
    keep_c = np.array([c not in p.target_cna_columns for c in p.cna_columns], bool)
    result = raw_df[list(p.baseline_columns)].copy()
    for j, col in enumerate(p.mutation_columns):
        if keep_m[j]:
            result[col] = mut[:, j]
    for j, col in enumerate(p.cna_columns):
        if keep_c[j]:
            result[col] = cna[:, j]
    counts = mut[:, keep_m].sum(axis=1)
    total_vaf = np.where(mut[:, keep_m] == 1, vaf[:, keep_m], 0).sum(axis=1)
    result[f"nMut_{p.compartment}"] = counts
    result[f"VAF_mean_{p.compartment}"] = np.divide(
        total_vaf, counts, out=np.full(len(raw_df), np.nan), where=counts > 0
    )
    result[f"CNA_{p.compartment}"] = cna[:, keep_c].sum(axis=1)
    summary_names = ["nMut", "VAF_mean", "CNA"]
    if p.burden_protocol is not None:
        from .burden import genomic_burdens

        bp = p.burden_protocol
        mutation_name = "nMut" if bp.mutation_representation == "raw_count" else "TMB"
        cna_name = {"raw_count": "CNA", "fga": "FGA", "panel_adjusted": "CNA_adjusted"}[
            bp.cna_representation
        ]
        masked = raw_df.copy()
        burden_keep = keep_m
        if bp.mutation_definition == "nonsynonymous_mutation_count":
            burden_keep = keep_m & np.array(
                [c in p.nonsynonymous_mutation_columns for c in p.mutation_columns],
                bool,
            )
        masked["__masked_mutation_count"] = mut[:, burden_keep].sum(axis=1)
        masked["__masked_cna_numerator"] = cna[:, keep_c].sum(axis=1)
        if bp.cna_representation == "fga":
            lengths = raw_df[list(p.cna_segment_length_columns)].to_numpy(float)
            if not np.isfinite(lengths).all() or np.any(lengths < 0):
                raise ValueError(
                    "Measured nonnegative nonoverlapping segment lengths required"
                )
            masked["__masked_cna_numerator"] = (
                cna[:, keep_c] * lengths[:, keep_c]
            ).sum(axis=1)
        burden = (
            genomic_burdens(
                masked,
                bp,
                mutation_count_column="__masked_mutation_count",
                cna_numerator_column="__masked_cna_numerator",
            )
            if len(masked)
            else dict(mutation_burden=np.empty(0), cna_burden=np.empty(0))
        )
        result = result.drop(columns=[f"nMut_{p.compartment}", f"CNA_{p.compartment}"])
        result[f"{mutation_name}_{p.compartment}"] = burden["mutation_burden"]
        result[f"{cna_name}_{p.compartment}"] = burden["cna_burden"]
        summary_names = [mutation_name, "VAF_mean", cna_name]
    result.attrs["feature_metadata"] = {
        f"{name}_{p.compartment}": FEATURE_METADATA[name] for name in summary_names
    }
    return result


class FoldFittedTransformer:
    """Compose any fit/transform stage (e.g. PCA) after training preprocessing.

    The object represents a projection/summary, never the patient specific
    transition representation itself. No transformer is instantiated implicitly.
    """

    def __init__(self, preprocessor, transformer):
        self.preprocessor = preprocessor
        self.transformer = transformer
        self.fitted = False

    def fit(self, training_df):
        x = self.preprocessor.fit(training_df).transform(training_df)
        self.transformer.fit(x)
        self.fitted = True
        return self

    def transform(self, frame):
        if not self.fitted:
            raise ValueError("Fit the projection on training data first")
        return self.transformer.transform(self.preprocessor.transform(frame))


LEGACY_FEATURE_PROVENANCE_STATUS = "REQUIRES_DATA_REPROCESSING_BEFORE_RERUN"


def validate_genomic_summary_units(frame, *, require_provenance=True):
    """Validate raw counts/fractions, optionally checking declared provenance.

    CSV entry points use the manuscript's registered background-summary schema;
    explicit FeatureMetadata also detects previously scaled input tables.
    """
    declared = frame.attrs.get("feature_metadata", {})
    for column in frame:
        if column.endswith("_is_Missing"):
            continue
        base = next(
            (
                name
                for name in FEATURE_METADATA
                if column in (name + "_Primary", name + "_Metastatic")
            ),
            None,
        )
        if base is None:
            continue
        metadata = declared.get(column)
        if metadata is None and require_provenance:
            raise ValueError(
                f"{LEGACY_FEATURE_PROVENANCE_STATUS}: unit provenance missing for {column}"
            )
        if isinstance(metadata, dict):
            metadata = FeatureMetadata(**metadata)
        canonical = FEATURE_METADATA[base]
        if metadata is not None and (
            metadata.raw_unit != canonical.raw_unit
            or metadata.normalization != canonical.normalization
            or metadata.transformation != canonical.transformation
        ):
            raise ValueError(
                f"{column}: scaled/normalized legacy summaries must be reprocessed from raw measurements"
            )
        values = pd.to_numeric(frame[column], errors="raise").dropna().to_numpy(float)
        canonical.validate(values, unit=canonical.raw_unit)
        if base in ("nMut", "CNA") and np.any(values != np.floor(values)):
            raise ValueError(f"{column}: raw locus counts must be integral")


@dataclass(frozen=True)
class FrozenCovariateProtocol:
    cohort: str
    schema_id: str
    topology: tuple[str, ...]
    kinetic: tuple[str, ...]
    frozen: bool = False
    version: str = "1"

    def validate(self, frame=None):
        if not self.frozen or not self.schema_id or not self.cohort or not self.version:
            raise ValueError(
                "REQUIRES_PROTOCOL_FREEZE_BEFORE_RERUN: verified cohort covariates required"
            )
        for columns in (self.topology, self.kinetic):
            TopologyCovariateSchema(tuple(columns))
            for compartment in ("Primary", "Metastatic"):
                for pair in (("nMut", "TMB"), ("CNA", "FGA", "CNA_adjusted")):
                    if sum(f"{name}_{compartment}" in columns for name in pair) > 1:
                        raise ValueError(
                            "Duplicate burden representations require a separate scientific protocol"
                        )
            if frame is not None and not set(columns) <= set(frame):
                raise ValueError(
                    "Frozen cohort covariates unavailable; do not silently drop columns"
                )
        return self
