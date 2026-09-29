"""Formal baseline contracts. No surrogate implementations or implicit software installs."""
from dataclasses import dataclass
import numpy as np
from .benchmark import FormalProEHNPredictor
from .events import select_event_schema, project_event_frame


@dataclass
class ProEHNTopologyOnlyPredictor:
    full: FormalProEHNPredictor
    fold: object

    @property
    def spec(self):
        return self.full.spec

    def predict_candidate_scores(self, features):
        return self.full.predict_candidate_scores(features, progression_override=1.)

    @property
    def inference_protocol(self):
        return self.full.inference_protocol

    @property
    def selected_event_schema(self):
        return self.full.selected_event_schema

    @property
    def topology_params(self):
        return self.full.topology_params

    def predict(self, features):
        return self.full._predict(features, progression_override=1.)


@dataclass
class ProEHNTopologyOnlyTrainer:
    requires_formal_folds = True
    """A paired ablation of an already fitted Full model, never a second fit.

    Construct with the SAME BenchmarkFold and global patient index used for Full.
    Full must be fitted first within that outer iteration; no new preprocessing,
    topology optimization or model selection is permitted for this ablation.
    """
    full: FormalProEHNPredictor
    fold: object
    patient_ids: tuple

    @property
    def spec(self):
        return self.full.spec

    def fit(self, training, validation):
        self.fold.validate(self.patient_ids)
        ids=self.full.spec.patient_id_column
        train=set(map(str,training[ids])); val=set(map(str,validation[ids]))
        expected_train={str(self.patient_ids[i]) for i in self.fold.train}
        expected_val={str(self.patient_ids[i]) for i in self.fold.validation}
        if train!=expected_train or val!=expected_val or train!=set(self.full.training_patient_ids) or val!=set(self.full.validation_patient_ids):
            raise ValueError('Topology-only ablation must share Full topology and exact training/validation fold')
        return ProEHNTopologyOnlyPredictor(self.full,self.fold)


@dataclass(frozen=True)
class ExternalBaselineContract:
    method: str
    implementation: str
    version: str
    event_schema_version: str
    supports_partial_observations: bool = False
    compartment_projection: str | None = None

    def validate(self):
        if self.compartment_projection not in (None,'primary','metastasis'):
            raise ValueError('Invalid classic-method compartment projection')
        if self.method not in ('MHN','Oncotree','HyperTraPS') or not all((self.implementation,self.version,self.event_schema_version)):
            raise ValueError('Named external method and versioned implementation/event contract required')


class ExternalBaselineTrainer:
    requires_formal_folds = True
    """Dependency shell: backend.fit(training, validation, spec=...) -> predictor.

    Backend must implement the actual named algorithm and declare conditional
    target probability output. Wiring/verification of external implementations is
    required before rerun; arbitrary proxy predictions cannot claim these names.
    """
    def __init__(self, spec, contract, backend=None):
        contract.validate()
        if spec is not None and contract.compartment_projection not in (None,spec.compartment):
            raise ValueError('Classic-method projection must match the common target compartment')
        self.spec,self.contract,self.backend=spec,contract,backend

    def fit(self, training, validation):
        if self.backend is None:
            raise RuntimeError(f'EXTERNAL_DEPENDENCY_REQUIRED_BEFORE_RERUN: {self.contract.method}')
        if getattr(self.backend,'scientific_contract',None)!=self.contract:
            raise ValueError('External backend identity/provenance mismatch')
        ids=self.spec.patient_id_column
        if training[ids].isna().any() or validation[ids].isna().any() or set(training[ids])&set(validation[ids]):
            raise ValueError('Baseline partitions must be patient-disjoint')
        train=self.spec.training_frame(training); val=self.spec.training_frame(validation)
        self._validate_observations(train); self._validate_observations(val)
        if self.contract.event_schema_version!=self.spec.event_selection_protocol.schema_version:
            raise ValueError('Classic baseline must use the shared event selection version')
        selected=select_event_schema(training,self.spec)
        predictor=self.backend.fit(project_event_frame(train,selected,compartment=self.contract.compartment_projection),project_event_frame(val,selected,compartment=self.contract.compartment_projection),spec=self.spec)
        return ExternalBaselinePredictor(self.spec,self.contract,predictor,selected)

    def _validate_observations(self, frame):
        if not self.contract.supports_partial_observations and np.any(frame.observation_type.to_numpy()!=3):
            raise ValueError('External implementation has no validated partial-observation likelihood')


@dataclass
class ExternalBaselinePredictor:
    spec: object
    contract: ExternalBaselineContract
    backend: object
    selected_event_schema: object

    def predict_candidate_scores(self,features):
        self._validate_features(features)
        if not hasattr(self.backend,'predict_conditional_events'):
            raise RuntimeError('EXTERNAL_DEPENDENCY_REQUIRED_BEFORE_RERUN: backend candidate event predictions')
        scores=np.asarray(self.backend.predict_conditional_events(project_event_frame(features,self.selected_event_schema,compartment=self.contract.compartment_projection),
            candidate_events=self.selected_event_schema.target_candidate_events(self.spec.compartment),spec=self.spec),float)
        if scores.shape!=(len(features),len(self.selected_event_schema.target_candidate_events(self.spec.compartment))) or not np.isfinite(scores).all() or np.any((scores<0)|(scores>1)):
            raise ValueError('Invalid external conditional candidate probabilities')
        return scores

    def _validate_features(self, features):
        for c in ('primary','metastasis'):
            if not (features[self.spec.target_column(c)]==0).all():
                raise ValueError('Common target masking is required')
        if not self.contract.supports_partial_observations and np.any(features.observation_type.to_numpy()!=3):
            raise ValueError('Partial-observation inference not supported by this external implementation')

    def predict(self, features):
        self._validate_features(features)
        p=np.asarray(self.backend.predict_conditional_target(project_event_frame(features,self.selected_event_schema,compartment=self.contract.compartment_projection),spec=self.spec),float)
        if p.shape!=(len(features),) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
            raise ValueError('External baseline must return aligned conditional target probabilities')
        return np.column_stack((np.ones(len(p)),p,p))


class MHNTrainer(ExternalBaselineTrainer):
    def __init__(self,spec,contract,backend=None):
        if contract.method!='MHN': raise ValueError('MHN contract required')
        super().__init__(spec,contract,backend)


class OncotreeTrainer(ExternalBaselineTrainer):
    def __init__(self,spec,contract,backend=None):
        if contract.method!='Oncotree': raise ValueError('Oncotree contract required')
        super().__init__(spec,contract,backend)


class HyperTraPSTrainer(ExternalBaselineTrainer):
    def __init__(self,spec,contract,backend=None):
        if contract.method!='HyperTraPS': raise ValueError('HyperTraPS contract required')
        super().__init__(spec,contract,backend)
