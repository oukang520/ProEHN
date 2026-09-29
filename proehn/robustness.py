"""Prespecified robustness interfaces; callers execute future experiments explicitly."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RobustnessProtocol:
    kind: str
    columns: tuple[str,...]
    magnitude: float
    random_seed: int
    version: str = 'robustness-v1'

    def __post_init__(self):
        if self.kind not in ('outer_test_feature_perturbation','training_label_noise','evaluation_label_sensitivity'):
            raise ValueError('Declare perturbation target')
        if not self.columns or not np.isfinite(self.magnitude) or not 0<=self.magnitude<=1:
            raise ValueError('Freeze columns and perturbation fraction')

    @property
    def requires_retraining(self):
        return self.kind=='training_label_noise'

    @property
    def interpretation(self):
        return 'evaluation-label sensitivity, not trained-model robustness' if self.kind=='evaluation_label_sensitivity' else self.kind


def outer_test_feature_perturbation(frame, patient_ids, fold, protocol, *, perturb, feature_builder, fitted_predictor):
    """No fit/trainer callback: perturb held-out raw inputs, then rebuild safe features.

    perturb receives only frozen genomic columns, fraction and RNG. The original
    frame is copied; labels and all other fields cannot be modified by callback.
    """
    fold.validate(patient_ids)
    if protocol.kind!='outer_test_feature_perturbation':
        raise ValueError('Training-label noise needs future retraining on corrupted training folds')
    test=frame.iloc[list(fold.test)].copy(deep=True)
    # Scientific caller explicitly registers raw genomic columns on the builder.
    raw_columns=getattr(feature_builder,'raw_genomic_columns',None)
    if raw_columns is None and hasattr(feature_builder,'__self__'):
        spec=feature_builder.__self__
        raw_columns={c for p in spec.masking_protocols for c in p.mutation_columns+p.vaf_columns+p.cna_columns}
    if raw_columns is None or not set(protocol.columns)<=set(raw_columns):
        raise ValueError('Only registered raw genomic inputs can be perturbed')
    changed=perturb(test[list(protocol.columns)].copy(),protocol.magnitude,np.random.default_rng(protocol.random_seed))
    if not changed.index.equals(test.index) or list(changed.columns)!=list(protocol.columns):
        raise ValueError('Perturbation must preserve held-out patient/feature alignment')
    test.loc[:,list(protocol.columns)]=changed
    return fitted_predictor.predict(feature_builder(test))


def corrupted_training_labels(training, labels, protocol, *, corrupt):
    """Return a training-only copy for a FUTURE retraining experiment; never score."""
    if not protocol.requires_retraining:
        raise ValueError('Evaluation-label flips cannot represent training robustness')
    values=np.asarray(labels)
    if values.shape!=(len(training),) or not np.isin(values,[-1,0,1]).all():
        raise ValueError('Aligned operational training labels required')
    changed=np.asarray(corrupt(values.copy(),protocol.magnitude,np.random.default_rng(protocol.random_seed)))
    if changed.shape!=values.shape or not np.isin(changed,[-1,0,1]).all() or np.any(changed[values==-1]!=-1):
        raise ValueError('Label noise must preserve missing-label status')
    return training.copy(deep=True),changed
