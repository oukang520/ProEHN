"""Patient-isolated evaluation infrastructure. No experiment runs at import.

All scores are OOF or explicitly held out. Operational label encoding for the
kinetic network is Stop=1; progression metrics use Go=1 (1-Stop).
"""
from dataclasses import dataclass
from typing import Protocol
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import matthews_corrcoef


@dataclass(frozen=True)
class BenchmarkFold:
    fold_id: int
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]

    def validate(self, patient_ids):
        groups = np.asarray(patient_ids)
        parts = [self.train, self.validation, self.test]
        if any(not p or len(set(p)) != len(p) or min(p) < 0 or max(p) >= len(groups) for p in parts):
            raise ValueError('Invalid fold indices')
        patient_sets = [set(groups[list(p)]) for p in parts]
        if any(patient_sets[i] & patient_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError('Patient leakage across partitions')
        if set().union(*(set(p) for p in parts)) != set(range(len(groups))):
            raise ValueError('Fold must partition all samples')


def patient_outer_folds(patient_ids, n_splits=5, validation_fraction=.25, random_seed=42):
    groups = np.asarray(patient_ids)
    if groups.ndim != 1 or pd.isna(groups).any() or len(np.unique(groups)) < n_splits:
        raise ValueError('Known patient groups and enough patients are required')
    folds = []
    for i, (outer_train, test) in enumerate(GroupKFold(n_splits).split(groups, groups=groups)):
        inner = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=random_seed+i)
        train, val = next(inner.split(outer_train, groups=groups[outer_train]))
        fold = BenchmarkFold(i, tuple(outer_train[train]), tuple(outer_train[val]), tuple(test))
        fold.validate(groups)
        folds.append(fold)
    return tuple(folds)


@dataclass(frozen=True)
class CalibratedThreshold:
    value: float
    criterion: str
    calibration_patient_ids: tuple[str, ...]

    def apply(self, scores, *, test_patient_ids):
        if set(map(str, test_patient_ids)) & set(self.calibration_patient_ids):
            raise ValueError('Threshold calibration and test patients overlap')
        s = _scores(scores)
        if len(test_patient_ids) != len(s) or pd.isna(np.asarray(test_patient_ids)).any():
            raise ValueError('Known patient IDs must align with scores')
        return s >= self.value


def _scores(scores):
    a = np.asarray(scores, float)
    if a.ndim != 1 or not np.isfinite(a).all() or np.any((a < 0) | (a > 1)):
        raise ValueError('Expected finite probability scores in [0,1]')
    return a


def _binary(labels, scores):
    s = _scores(scores)
    y = np.asarray(labels)
    if y.shape != s.shape or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError('Matching, nonempty binary labels required')
    return y, s


def calibrate_threshold(validation_labels, validation_scores, *, validation_patient_ids,
                        criterion='mcc', sensitivity=.9):
    """Select only on inner validation predictions; ties prefer larger cutoffs."""
    y, s = _binary(validation_labels, validation_scores)
    if len(validation_patient_ids) != len(s) or pd.isna(np.asarray(validation_patient_ids)).any() or len(np.unique(y)) < 2:
        raise ValueError('Calibration requires patient IDs and both classes')
    thresholds = np.r_[np.nextafter(1., np.inf), np.unique(s)[::-1]]
    if criterion == 'sensitivity':
        if not 0 < sensitivity <= 1:
            raise ValueError('Sensitivity must be in (0,1]')
        selected = next(t for t in thresholds if np.mean(s[y == 1] >= t) >= sensitivity)
    elif criterion == 'mcc':
        selected = max(thresholds, key=lambda t: matthews_corrcoef(y, s >= t))
    elif criterion == 'accuracy':
        selected = max(thresholds, key=lambda t: np.mean((s >= t) == y))
    else:
        raise ValueError('Unknown calibration criterion')
    return CalibratedThreshold(float(selected), criterion, tuple(map(str, validation_patient_ids)))


def fixed_threshold_mcc(labels, scores, threshold, *, test_patient_ids):
    y, s = _binary(labels, scores)
    return float(matthews_corrcoef(y, threshold.apply(s, test_patient_ids=test_patient_ids)))


def max_mcc_descriptive(labels, scores):
    """Descriptive curve maximum ONLY; not fixed-threshold test performance."""
    y, s = _binary(labels, scores)
    return float(max(matthews_corrcoef(y, s >= t) for t in np.r_[np.nextafter(1., np.inf), np.unique(s)]))


def specificity_at_calibrated_sensitivity(labels, scores, threshold, *, test_patient_ids):
    if threshold.criterion != 'sensitivity':
        raise ValueError('Requires a validation-sensitivity calibrated threshold')
    y, s = _binary(labels, scores)
    pred = threshold.apply(s, test_patient_ids=test_patient_ids)
    return {'specificity': float(np.mean(~pred[y == 0])) if np.any(y == 0) else float('nan'),
            'achieved_test_sensitivity': float(np.mean(pred[y == 1])) if np.any(y == 1) else float('nan')}


@dataclass(frozen=True)
class BenchmarkResult:
    model_name: str
    target: str
    patient_ids: tuple[str, ...]
    fold_ids: np.ndarray
    scores: np.ndarray

    def validate(self):
        if self.scores.shape[0] != len(self.patient_ids) or self.fold_ids.shape != (len(self.patient_ids),):
            raise ValueError('OOF alignment mismatch')
        if np.any(self.fold_ids < 0) or not np.isfinite(self.scores).all():
            raise ValueError('Incomplete OOF predictions')
        for patient in set(self.patient_ids):
            if len(set(self.fold_ids[np.asarray(self.patient_ids) == patient])) != 1:
                raise ValueError('Repeated patient assigned to different OOF folds')
        return self


class FoldPredictor(Protocol):
    def predict(self, features: pd.DataFrame) -> np.ndarray: ...


class FoldTrainer(Protocol):
    def fit(self, training: pd.DataFrame, validation: pd.DataFrame) -> FoldPredictor: ...


def run_oof_benchmark(frame, patient_ids, folds, trainers, *, target, test_feature_builder):
    """Shared folds/patients/target masking for every registered baseline.

    The trainer receives training and inner-validation only. The frozen feature
    builder removes target/outcome fields BEFORE predict; it must be the same
    object for all models. No metrics, files or results directories are generated.
    External Oncotree/HyperTraPS/MHN adapters implement FoldTrainer; unavailable
    methods must not be substituted with a surrogate under their formal name.
    """
    results = []
    if pd.isna(np.asarray(patient_ids)).any():
        raise ValueError('Missing patient identity')
    ids = tuple(map(str, patient_ids))
    if len(ids) != len(frame):
        raise ValueError('Patient IDs must align with frame')
    if len(set(f.fold_id for f in folds)) != len(folds) or any(f.fold_id < 0 for f in folds):
        raise ValueError('Fold identifiers must be unique and nonnegative')
    visits = np.zeros(len(frame), int)
    fold_ids = np.full(len(frame), -1, int)
    for fold in folds:
        fold.validate(ids)
        visits[list(fold.test)] += 1
        fold_ids[list(fold.test)] = fold.fold_id
    if not np.all(visits == 1):
        raise ValueError('Every sample must be held out exactly once')
    for name, factory in trainers.items():
        output = None
        for fold in folds:
            # Fresh adapter and defensive copies prevent cross-fold/model state reuse.
            predictor = factory().fit(frame.iloc[list(fold.train)].copy(), frame.iloc[list(fold.validation)].copy())
            safe = test_feature_builder(frame.iloc[list(fold.test)].copy())
            prediction = np.asarray(predictor.predict(safe), float)
            if prediction.shape[0] != len(fold.test) or not np.isfinite(prediction).all():
                raise ValueError('Invalid held-out prediction')
            if output is None:
                output = np.empty((len(frame), *prediction.shape[1:]))
            output[list(fold.test)] = prediction
        results.append(BenchmarkResult(name, target, ids, fold_ids.copy(), output).validate())
    return results


def survival_score_inputs(oof: BenchmarkResult, time_days, event, *, score_column=0, km_threshold):
    """OOF alignment and fixed direction: larger Go score = higher risk.

    Cox uses risk_score unchanged; a concordance API expecting longer survival
    uses predicted_survival_score=-risk_score. No observed-outcome sign flip.
    KM threshold is fixed externally on training/validation, never optimized here.
    """
    oof.validate()
    t, e = np.asarray(time_days, float), np.asarray(event)
    if t.shape != (len(oof.patient_ids),) or e.shape != t.shape or not np.isfinite(t).all() or np.any(t < 0) or not np.isin(e, [0, 1]).all():
        raise ValueError('Invalid survival units, missingness or alignment')
    risk = _scores(oof.scores if oof.scores.ndim == 1 else oof.scores[:, score_column])
    if isinstance(km_threshold, dict):
        # Each OOF partition uses its own training/inner-validation threshold.
        high = np.empty(len(risk), bool)
        for fold_id in np.unique(oof.fold_ids):
            selected = oof.fold_ids == fold_id
            if fold_id not in km_threshold:
                raise ValueError('Missing fold-specific KM threshold')
            ids = np.asarray(oof.patient_ids)[selected]
            high[selected] = km_threshold[fold_id].apply(risk[selected], test_patient_ids=ids)
    else:
        high = km_threshold.apply(risk, test_patient_ids=oof.patient_ids)
    return dict(time_days=t, event=e, risk_score=risk, predicted_survival_score=-risk, high_risk=high)


@dataclass(frozen=True)
class AblationInputs:
    """Separate tasks: topology alone has no progression classifier.

    conditional ranking uses identical known-event samples for all models,
    regardless of Stop/Go. Optional integrated decisions require each model's
    explicit independent gate and a validation-calibrated threshold.
    """
    progression_scores: np.ndarray | None
    conditional_event_probabilities: np.ndarray
    integrated_event_scores: np.ndarray | None = None


def ablation_full_vs_evolution(full: AblationInputs, evolution: AblationInputs, *, known_event_mask):
    mask = np.asarray(known_event_mask, bool)
    if full.conditional_event_probabilities.shape != evolution.conditional_event_probabilities.shape or mask.shape != (len(full.conditional_event_probabilities),):
        raise ValueError('Both models must share evaluation patients and event definitions')
    return {
        'progression_propensity': (full.progression_scores, evolution.progression_scores),
        'conditional_event_ranking': (full.conditional_event_probabilities[mask], evolution.conditional_event_probabilities[mask]),
        'optional_integrated_decision': (full.integrated_event_scores, evolution.integrated_event_scores),
    }


def select_on_inner_validation(training, validation, candidates, fit_and_score, *, patient_id_column):
    """Hyperparameter/regularization selection with no held-out test argument.

    fit_and_score receives defensive training/validation copies and must return
    a validation LOSS (smaller is better), never a training likelihood proxy.
    """
    train_ids, val_ids = training[patient_id_column], validation[patient_id_column]
    if train_ids.isna().any() or val_ids.isna().any() or set(train_ids) & set(val_ids):
        raise ValueError('Hyperparameter selection requires disjoint inner partitions')
    candidates = tuple(candidates)
    if not candidates:
        raise ValueError('Prespecify at least one candidate')
    best, loss = None, np.inf
    for candidate in candidates:
        value = float(fit_and_score(training.copy(), validation.copy(), candidate))
        if not np.isfinite(value):
            raise ValueError('Nonfinite validation loss')
        if value < loss:
            best, loss = candidate, value
    return best


class TargetNotEvaluable(ValueError):
    """Prespecified fold/class coverage cannot be met; no performance-driven retry."""


@dataclass(frozen=True)
class GroupedStratificationPolicy:
    outer_splits: int = 5
    inner_splits: int = 4
    minimum_positive_patients: int = 1
    minimum_negative_patients: int = 1
    minimum_labeled_kinetic_patients: int = 1
    shuffle: bool = False
    random_seed: int = 42

    def __post_init__(self):
        if self.outer_splits < 2 or self.inner_splits < 2 or min(self.minimum_positive_patients,self.minimum_negative_patients,self.minimum_labeled_kinetic_patients) < 1:
            raise ValueError('Freeze positive fold counts and coverage constraints')


def grouped_stratified_outer_folds(patient_ids, labels, *, target, policy, kinetic_labels=None):
    """Freeze target-specific splits before fitting; count independent patients.

    Multiple samples of a patient must agree on the stratification label. A
    different aggregation of conflicting sample labels requires a separate,
    prespecified biological target definition; it is not inferred here.
    Missing stratification labels are not eligible. No seed search, fold-number
    reduction or rebalancing based on model results is performed.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    groups=np.asarray(patient_ids); y=np.asarray(labels)
    if groups.ndim!=1 or y.shape!=groups.shape or pd.isna(groups).any() or not np.isin(y,[0,1]).all():
        raise TargetNotEvaluable(f'{target}: known patient IDs and binary target labels required')
    unique=list(dict.fromkeys(groups.tolist())); uy=[]
    ky=np.asarray(kinetic_labels) if kinetic_labels is not None else y
    if ky.shape!=y.shape or not np.isin(ky,[-1,0,1]).all():
        raise TargetNotEvaluable(f'{target}: invalid kinetic labels')
    for patient in unique:
        values=np.unique(y[groups==patient])
        if len(values)!=1:
            raise TargetNotEvaluable(f'{target}: inconsistent within-patient stratification labels')
        uy.append(values[0])
    unique=np.asarray(unique); uy=np.asarray(uy)
    if len(unique)<policy.outer_splits or any(np.sum(uy==v)<policy.outer_splits*minimum for v,minimum in [(1,policy.minimum_positive_patients),(0,policy.minimum_negative_patients)]):
        raise TargetNotEvaluable(f'{target}: insufficient class-bearing patients for frozen outer fold policy')
    def splitter(n):
        return StratifiedGroupKFold(n_splits=n,shuffle=policy.shuffle,
                                   random_state=policy.random_seed if policy.shuffle else None)
    def rows(patient_indices):
        return tuple(np.flatnonzero(np.isin(groups,unique[patient_indices])))
    result=[]
    for i,(outer,test) in enumerate(splitter(policy.outer_splits).split(unique,uy,unique)):
        if any(np.sum(uy[outer]==v)<policy.inner_splits for v in (0,1)):
            raise TargetNotEvaluable(f'{target}: insufficient inner-fold class coverage')
        train,val=next(splitter(policy.inner_splits).split(unique[outer],uy[outer],unique[outer]))
        fold=BenchmarkFold(i,rows(outer[train]),rows(outer[val]),rows(test))
        fold.validate(groups)
        for partition in (fold.train,fold.validation,fold.test):
            ix=np.asarray(partition)
            positive=len(set(groups[ix][y[ix]==1])); negative=len(set(groups[ix][y[ix]==0]))
            labeled=len(set(groups[ix][ky[ix]!=-1]))
            if positive<policy.minimum_positive_patients or negative<policy.minimum_negative_patients or labeled<policy.minimum_labeled_kinetic_patients:
                raise TargetNotEvaluable(f'{target}: fold {i} violates prespecified coverage constraints')
        result.append(fold)
    return tuple(result)
