"""Explicit clinical operational Stop=1 / Go=0 / unknown=-1 contracts.

No biological-stasis interpretation and no genomic proxy fallback. Historical
cohort thresholds conflict across source scripts; callers must provide a frozen
cohort protocol and declare units, rather than silently choosing a version.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProgressionLabelSchema:
    cohort: str
    pfs_column: str
    event_column: str
    time_unit: str
    durable_threshold_days: float
    response_column: str | None = None
    # Empty by default: RECIST mapping and treatment setting require a protocol.
    response_mapping: tuple[tuple[str, int], ...] = ()
    time_transformation: str = 'none'
    schema_id: str = ''
    version: str = ''
    frozen: bool = False

    def __post_init__(self):
        if self.time_unit != 'days' or self.time_transformation != 'none':
            raise ValueError('Labels require untransformed PFS in days')
        if not np.isfinite(self.durable_threshold_days) or self.durable_threshold_days <= 0:
            raise ValueError('Explicit positive durable threshold in days is required')
        if any(v not in (0, 1) for _, v in self.response_mapping):
            raise ValueError('Response mapping uses operational Stop=1 / Go=0')
        if len(dict(self.response_mapping)) != len(self.response_mapping):
            raise ValueError('Duplicate response mapping')


def build_progression_label(frame, schema):
    """Go: observed progression before cutoff. Stop: PFS reaches cutoff.

    Censoring before cutoff or missing event status before cutoff -> unknown.
    At cutoff an event counts as reaching the prespecified duration. Explicit
    response mappings override duration when present. Inputs are outcome-only;
    these columns must never be supplied as baseline model covariates.
    """
    t = pd.to_numeric(frame[schema.pfs_column], errors='raise').to_numpy(float)
    event = pd.to_numeric(frame[schema.event_column], errors='raise').to_numpy(float)
    if np.isinf(t).any() or np.any(t < 0) or not np.isin(event[~np.isnan(event)], [0, 1]).all():
        raise ValueError('Invalid PFS duration or event indicator')
    labels = np.full(len(frame), -1, int)
    labels[np.isfinite(t) & (t >= schema.durable_threshold_days)] = 1
    labels[np.isfinite(t) & (t < schema.durable_threshold_days) & (event == 1)] = 0
    missing = schema.pfs_column+'_is_Missing'
    if missing in frame:
        labels[frame[missing].fillna(1).to_numpy() != 0] = -1
    if schema.response_column:
        mapped = frame[schema.response_column].map(dict(schema.response_mapping))
        use = mapped.notna().to_numpy()
        labels[use] = mapped[use].to_numpy(int)
    return labels


def _cohort(frame, schema, name):
    if schema.cohort != name:
        raise ValueError('Label protocol belongs to a different cohort')
    return build_progression_label(frame, schema)


def build_paca_progression_label(frame, schema):
    return _cohort(frame, schema, 'PACA')


def build_mela_progression_label(frame, schema):
    return _cohort(frame, schema, 'MELA')


def build_lc_progression_label(frame, schema):
    return _cohort(frame, schema, 'LC')


def require_frozen_label_protocol(schema):
    if not schema.frozen or not schema.schema_id or not schema.version:
        raise ValueError('REQUIRES_PROTOCOL_FREEZE_BEFORE_RERUN: frozen cohort label schema id/version required')
    return schema


def precomputed_progression_labels(frame, column, schema):
    """Accept cached labels only with an exact, frozen clinical source contract."""
    require_frozen_label_protocol(schema)
    expected = label_provenance(schema)
    if frame.attrs.get('label_provenance', {}).get(column) != expected:
        raise ValueError('Precomputed label lacks matching frozen provenance')
    values = frame[column].to_numpy()
    if not np.isin(values, [-1, 0, 1]).all():
        raise ValueError('Invalid operational labels')
    return values


def label_provenance(schema):
    return dict(schema_id=schema.schema_id, version=schema.version, cohort=schema.cohort,
        source_variables=[schema.pfs_column, schema.event_column] + ([schema.response_column] if schema.response_column else []),
        threshold=schema.durable_threshold_days, unit=schema.time_unit,
        response_mapping=[list(x) for x in schema.response_mapping], time_transformation=schema.time_transformation)
