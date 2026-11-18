"""Observation contracts, independent of operational progression labels.

Legacy PACA types 0/1 were derived from Stop/Go (PACA-seeding.py), not
confirmed dissemination. Both therefore map to PRIMARY_SEEDING_UNKNOWN.
Paired order 0 marginalizes diagnosis order; it is NOT exact synchrony.
"""

from enum import IntEnum
from dataclasses import dataclass


class ObservationType(IntEnum):
    PRIMARY_NO_DISSEMINATION = 0
    PRIMARY_DISSEMINATED = 1
    METASTASIS_ONLY = 2
    PAIRED = 3
    PRIMARY_SEEDING_UNKNOWN = 4


class DiagnosisOrder(IntEnum):
    UNKNOWN = 0
    PRIMARY_FIRST = 1
    METASTASIS_FIRST = 2


def normalize_observation_type(value):
    aliases = {
        "primary_only": 4,
        "primary_seeding_unknown": 4,
        "primary_no_dissemination": 0,
        "primary_disseminated": 1,
        "metastasis_only": 2,
        "paired": 3,
    }
    return ObservationType(aliases.get(value, value))


def observed_record_type(value, observed_seeding=None):
    """Resolve an independently measured seed bit, leaving missing bits latent."""
    import pandas as pd

    kind = normalize_observation_type(value)
    if observed_seeding is None or pd.isna(observed_seeding):
        return kind
    if observed_seeding not in (0, 1):
        raise ValueError("Independently observed seeding must be binary or missing")
    if kind == ObservationType.PRIMARY_SEEDING_UNKNOWN:
        return ObservationType(int(observed_seeding))
    if observed_seeding != (0 if kind == 0 else 1):
        raise ValueError("Observed seeding conflicts with observation_type")
    return kind


@dataclass(frozen=True)
class ObservationSemantics:
    observed: str
    latent: str
    seeding: str
    likelihood_path: str


OBSERVATION_SEMANTICS = {
    ObservationType.PRIMARY_NO_DISSEMINATION: ObservationSemantics(
        "PT genotype; independently confirmed seeding absence at observation",
        "event times",
        "observed 0",
        "_lp_prim_obs(seed=0)",
    ),
    ObservationType.PRIMARY_DISSEMINATED: ObservationSemantics(
        "PT genotype; independently confirmed dissemination",
        "MT genotype and event times",
        "observed 1",
        "_lp_prim_obs(seed=1)",
    ),
    ObservationType.METASTASIS_ONLY: ObservationSemantics(
        "MT genotype", "PT genotype and event times", "observed 1", "_lp_met_obs"
    ),
    ObservationType.PAIRED: ObservationSemantics(
        "PT and MT genotypes; seeding occurred by final observation (not necessarily first); diagnosis order if known",
        "event times; unknown order",
        "observed 1",
        "_lp_coupled_0/1/2 by DiagnosisOrder",
    ),
    ObservationType.PRIMARY_SEEDING_UNKNOWN: ObservationSemantics(
        "PT genotype",
        "seeding, MT genotype, event times",
        "marginalized 0 and 1",
        "logaddexp(_lp_prim_obs(seed=0), _lp_prim_obs(seed=1))",
    ),
}


def observation_types(training_df, type_column="type"):
    import numpy as np

    if "observation_type" in training_df:
        values = training_df["observation_type"].to_numpy()
        values = np.asarray([int(normalize_observation_type(v)) for v in values])
    elif type_column in training_df:
        values = training_df[type_column].to_numpy().copy()
        values = np.where(np.isin(values, [0, 1]), 4, values)
    else:
        raise ValueError(
            "Explicit observation_type or documented legacy type is required"
        )
    if not np.isin(values, list(ObservationType)).all():
        raise ValueError("Unknown/missing observation type; do not infer from outcomes")
    if "observed_seeding" in training_df:
        values = np.asarray(
            [
                int(observed_record_type(kind, seed))
                for kind, seed in zip(values, training_df["observed_seeding"])
            ]
        )
    return values.astype(int)


OBSERVATION_SEMANTICS_VERSION = "paired-two-phase-v2"
