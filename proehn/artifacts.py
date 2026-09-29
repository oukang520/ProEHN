"""Versioned scientific topology artifacts; runtime options cannot change semantics."""
from dataclasses import dataclass, asdict
import json
import numpy as np
from .features import FoldPreprocessor, TopologyCovariateSchema
from .observations import OBSERVATION_SEMANTICS_VERSION
from .topology import ProEHNTopologyModel


@dataclass(frozen=True)
class ScientificArtifactMetadata:
    gene_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    covariate_schema: tuple[str, ...]
    preprocessing: dict
    log_rate_clip_min: float
    log_rate_clip_max: float
    regularization: dict
    target_event_schema_version: str
    training_feature_provenance_version: str
    scientific_contract_version: int = 3
    model_version: str = 'ProEHN-round2'
    observation_semantics_version: str = OBSERVATION_SEMANTICS_VERSION

    def validate(self):
        if self.scientific_contract_version != 3 or self.observation_semantics_version != OBSERVATION_SEMANTICS_VERSION:
            raise ValueError('Unsupported scientific/observation contract')
        if not self.model_version or not self.target_event_schema_version or not self.training_feature_provenance_version:
            raise ValueError('Scientific provenance versions are required')
        if not self.gene_names or len(set(self.gene_names)) != len(self.gene_names):
            raise ValueError('Unique event names required')
        if not np.isfinite([self.log_rate_clip_min, self.log_rate_clip_max]).all() or self.log_rate_clip_min >= self.log_rate_clip_max:
            raise ValueError('Invalid saved clip bounds')
        TopologyCovariateSchema(tuple(self.covariate_schema))
        prep = FoldPreprocessor.from_metadata(self.preprocessing)
        if tuple(prep.columns) != tuple(self.covariate_schema) or tuple(prep.feature_names) != tuple(self.feature_names):
            raise ValueError('Saved covariate/preprocessing order mismatch')
        if set(self.regularization) != {'regularization_strength','l1_ratio','l2_floor'}:
            raise ValueError('Complete regularization metadata required')
        return self


def save_topology_artifact(path, params, metadata):
    metadata.validate()
    model = ProEHNTopologyModel(len(metadata.gene_names), len(metadata.feature_names),
        log_rate_clip_min=metadata.log_rate_clip_min, log_rate_clip_max=metadata.log_rate_clip_max, **metadata.regularization)
    p = np.asarray(params, float)
    if p.shape != (model.shapes.total_size,) or not np.isfinite(p).all():
        raise ValueError('Invalid artifact parameter vector')
    np.savez(path, params=p, scientific_metadata_json=json.dumps(asdict(metadata), allow_nan=False))


def load_topology_artifact(path, **runtime):
    with np.load(path, allow_pickle=False) as data:
        if 'scientific_metadata_json' not in data:
            raise ValueError('Formal artifact lacks scientific contract; legacy migration requires verified provenance')
        meta = ScientificArtifactMetadata(**json.loads(str(data['scientific_metadata_json'].item()))).validate()
        params = np.asarray(data['params'], float)
    saved = dict(meta.regularization, log_rate_clip_min=meta.log_rate_clip_min, log_rate_clip_max=meta.log_rate_clip_max)
    for key, value in runtime.items():
        if value is not None and (key not in saved or value != saved[key]):
            raise ValueError(f'Runtime configuration conflicts with artifact: {key}')
    model = ProEHNTopologyModel(len(meta.gene_names), len(meta.feature_names), **saved)
    if params.shape != (model.shapes.total_size,) or not np.isfinite(params).all():
        raise ValueError('Invalid artifact parameter vector')
    return model, params, meta, FoldPreprocessor.from_metadata(meta.preprocessing)
