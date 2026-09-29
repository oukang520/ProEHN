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


@dataclass(frozen=True)
class ObservationSemantics:
    observed: str
    latent: str
    seeding: str
    likelihood_path: str


OBSERVATION_SEMANTICS = {
    ObservationType.PRIMARY_NO_DISSEMINATION: ObservationSemantics(
        'PT genotype; independently confirmed seeding absence at observation',
        'event times', 'observed 0', '_lp_prim_obs(seed=0)'),
    ObservationType.PRIMARY_DISSEMINATED: ObservationSemantics(
        'PT genotype; independently confirmed dissemination',
        'MT genotype and event times', 'observed 1', '_lp_prim_obs(seed=1)'),
    ObservationType.METASTASIS_ONLY: ObservationSemantics(
        'MT genotype', 'PT genotype and event times', 'observed 1', '_lp_met_obs'),
    ObservationType.PAIRED: ObservationSemantics(
        'PT and MT genotypes; seeding confirmed by first observation; diagnosis order if known', 'event times; unknown order',
        'observed 1', '_lp_coupled_0/1/2 by DiagnosisOrder'),
    ObservationType.PRIMARY_SEEDING_UNKNOWN: ObservationSemantics(
        'PT genotype', 'seeding, MT genotype, event times',
        'marginalized 0 and 1', 'logaddexp(_lp_prim_obs(seed=0), _lp_prim_obs(seed=1))'),
}


def observation_types(training_df, type_column='type'):
    import numpy as np
    if 'observation_type' in training_df:
        values = training_df['observation_type'].to_numpy()
    elif type_column in training_df:
        values = training_df[type_column].to_numpy().copy()
        values = np.where(np.isin(values, [0, 1]), 4, values)
    else:
        raise ValueError('Explicit observation_type or documented legacy type is required')
    if not np.isin(values, list(ObservationType)).all():
        raise ValueError('Unknown/missing observation type; do not infer from outcomes')
    return values.astype(int)
