"""Explicit assay/coverage contracts; no inferred panel size or count units."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GenomicBurdenProtocol:
    cohort: str
    schema_id: str
    version: str
    mutation_representation: str
    cna_representation: str
    cna_input_unit: str
    assay_policy: str
    assay_id_column: str
    allowed_assays: tuple[str, ...]
    callable_mb_column: str | None = None
    cna_coverage_column: str | None = None
    frozen: bool = False
    mutation_definition: str = "registered_mutation_count"
    fga_available: bool | None = None
    cna_fallback_reason: str = ""

    def validate(self, frame):
        if self.mutation_definition not in (
            "registered_mutation_count",
            "nonsynonymous_mutation_count",
        ):
            raise ValueError(
                "Explicit supported mutation numerator definition required"
            )
        if not self.frozen or not self.cohort or not self.schema_id or not self.version:
            raise ValueError("Frozen genomic burden/assay protocol required")
        if self.mutation_representation not in (
            "raw_count",
            "tmb",
        ) or self.cna_representation not in ("raw_count", "fga", "panel_adjusted"):
            raise ValueError(
                "Explicit supported genomic burden representations required"
            )
        if (
            self.assay_policy not in ("shared_assay", "coverage_adjusted")
            or not self.allowed_assays
        ):
            raise ValueError(
                "Declare assay comparability; raw CNA counts are not comparable across arbitrary panels"
            )
        assays = frame[self.assay_id_column]
        if assays.isna().any() or not set(assays) <= set(self.allowed_assays):
            raise ValueError("Assay outside the frozen eligible assay set")
        if self.assay_policy == "shared_assay" and (
            len(self.allowed_assays) != 1 or assays.nunique() != 1
        ):
            raise ValueError(
                "Shared-assay raw counts require a single frozen assay subset"
            )
        if assays.nunique() > 1 and (
            self.cna_representation == "raw_count"
            or self.mutation_representation == "raw_count"
        ):
            raise ValueError(
                "Mixed-assay raw counts require a separately justified harmonization protocol"
            )
        expected = (
            "altered_length_mb"
            if self.cna_representation == "fga"
            else "altered_locus_count"
        )
        if self.cna_input_unit != expected:
            raise ValueError(
                "CNA numerator units conflict with the chosen representation"
            )
        if self.mutation_representation == "tmb" and not self.callable_mb_column:
            raise ValueError("TMB requires explicit per-sample callable Mb")
        if self.cna_representation != "raw_count" and not self.cna_coverage_column:
            raise ValueError("Adjusted CNA/FGA requires matched explicit coverage")
        return self


def validate_primary_burden(masking_protocols, config):
    """Bind a frozen scientific definition to explicit raw-locus/assay metadata.

    CNA fallback is selected for the protocol, never per row or from outcomes.
    """
    definition = config["primary_analysis"]["genomic_burden"]
    for masking in masking_protocols:
        protocol = masking.burden_protocol
        if (
            protocol is None
            or not protocol.frozen
            or protocol.cohort != config["cohort"]
        ):
            raise ValueError("Matching frozen raw burden protocol required")
        if (
            protocol.mutation_representation != definition["mutation_representation"]
            or protocol.mutation_definition != definition["mutation_definition"]
        ):
            raise ValueError(
                "Raw burden protocol conflicts with the frozen mutation definition"
            )
        if (
            protocol.mutation_definition == "nonsynonymous_mutation_count"
            and masking.nonsynonymous_mutation_columns is None
        ):
            raise ValueError(
                "Nonsynonymous mutation loci must be explicitly registered"
            )
        if (
            protocol.mutation_representation == "tmb"
            and not protocol.callable_mb_column
        ):
            raise ValueError("LC TMB requires per-sample callable panel coverage in Mb")
        if not protocol.cna_coverage_column:
            raise ValueError("CNA fraction requires explicit matched coverage")
        if protocol.cna_representation == "panel_adjusted":
            if protocol.fga_available is not False or not protocol.cna_fallback_reason:
                raise ValueError(
                    "CNA fraction fallback requires prespecified FGA unavailability and reason"
                )
        elif protocol.cna_representation != "fga" or protocol.fga_available is False:
            raise ValueError(
                "Primary CNA burden prefers FGA; raw CNA counts are not permitted"
            )


def genomic_burdens(frame, protocol, *, mutation_count_column, cna_numerator_column):
    """Coverage-adjusted values only from measured, aligned positive denominators."""
    from .features import panel_corrected_mutation_burden

    protocol.validate(frame)
    count = frame[mutation_count_column].to_numpy(float)
    cna = frame[cna_numerator_column].to_numpy(float)
    if (
        not np.isfinite(count).all()
        or np.any(count < 0)
        or np.any(count != np.floor(count))
    ):
        raise ValueError("Verified nonnegative integral mutation count required")
    if (
        not np.isfinite(cna).all()
        or np.any(cna < 0)
        or (
            protocol.cna_input_unit == "altered_locus_count"
            and np.any(cna != np.floor(cna))
        )
    ):
        raise ValueError("Invalid raw CNA numerator units")
    mutation = (
        count
        if protocol.mutation_representation == "raw_count"
        else panel_corrected_mutation_burden(
            count, frame[protocol.callable_mb_column].to_numpy(float)
        )
    )
    if protocol.cna_representation != "raw_count":
        coverage = frame[protocol.cna_coverage_column].to_numpy(float)
        if (
            not np.isfinite(coverage).all()
            or np.any(coverage <= 0)
            or np.any(cna > coverage)
        ):
            raise ValueError("CNA numerator requires measured matching coverage")
        if protocol.cna_representation == "panel_adjusted" and np.any(
            coverage != np.floor(coverage)
        ):
            raise ValueError(
                "Panel-adjusted CNA denominator is a count of assayed loci"
            )
        cna = cna / coverage
    return dict(mutation_burden=mutation, cna_burden=cna, protocol=protocol)
