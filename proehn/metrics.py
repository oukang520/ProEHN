"""Formal pooled-OOF target recovery metrics; no threshold selection here."""
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
from .evaluation import BenchmarkResult, FormalFoldProtocol, CalibratedThreshold, _binary


@dataclass(frozen=True)
class CandidateFoldScores:
    fold_id: int
    patient_ids: tuple[str,...]
    candidate_events: tuple[tuple[str,str],...]
    masked_target: tuple[str,str]
    event_schema_version: str
    scores: np.ndarray


@dataclass(frozen=True)
class TargetRecoveryMetricSchema:
    target: str
    target_event: tuple[str,str]
    eligible_patient_ids: tuple[str,...]
    event_schema_versions: tuple[tuple[int,str],...]
    candidates_by_fold: tuple[tuple[int,tuple[tuple[str,str],...]],...]
    top_k: int
    version: str = 'target-recovery-metrics-v1'
    tie_rule: str = 'candidate_order'


def _thresholds(protocol, supplied, criterion):
    if set(supplied)!={f.fold_id for f in protocol.folds}:
        raise ValueError('Exactly one frozen threshold per outer fold required')
    for fold in protocol.folds:
        threshold=supplied[fold.fold_id]
        validation={protocol.patient_ids[i] for i in fold.validation}
        if not isinstance(threshold,CalibratedThreshold) or threshold.criterion!=criterion or set(threshold.calibration_patient_ids)!=validation:
            raise ValueError('Threshold must originate from this fold\'s inner validation patients')
        if threshold.score_semantics!='target_presence' or not np.isfinite(threshold.value):
            raise ValueError('Invalid threshold score semantics/value')
        if criterion=='sensitivity' and threshold.sensitivity!=.9:
            raise ValueError('Spec@90 requires a frozen validation sensitivity target of 0.9')


def evaluate_target_recovery(oof, target_labels, *, sensitivity_thresholds, mcc_thresholds,
        event_schema, score_column=2):
    """Pooled AUC/AP, held-out threshold decisions, and positive-target Top-k.

    Ranking ties follow the frozen candidate order. Each model must use the same
    CandidateFoldScores schema and masked context; target scores are checked
    against the candidate matrix. Missing candidate scores are an error, not a
    silently different Top-k definition. No max-test-MCC or test ROC cutoffs.
    """
    if not isinstance(oof,BenchmarkResult) or not isinstance(oof.fold_protocol,FormalFoldProtocol):
        raise ValueError('Formal grouped-stratified OOF predictions required')
    oof.validate(); protocol=oof.fold_protocol.validate()
    schema=event_schema
    if not isinstance(schema,TargetRecoveryMetricSchema) or schema.version!='target-recovery-metrics-v1' or schema.tie_rule!='candidate_order' or schema.top_k<1 or int(schema.top_k)!=schema.top_k:
        raise ValueError('Frozen target/event metric contract required')
    if tuple(oof.patient_ids)!=schema.eligible_patient_ids or len(set(oof.patient_ids))!=len(oof.patient_ids) or tuple(oof.patient_ids)!=protocol.patient_ids:
        raise ValueError('One aligned OOF row per eligible patient required')
    if oof.target!=schema.target or oof.target!=protocol.target or oof.event_schema_versions!=schema.event_schema_versions:
        raise ValueError('OOF target/event schema mismatch')
    y,s=_binary(target_labels,oof.scores if oof.scores.ndim==1 else oof.scores[:,score_column])
    if tuple(y)!=protocol.target_labels or len(np.unique(y))!=2:
        raise ValueError('Target labels must match frozen folds and contain both classes')
    _thresholds(protocol,sensitivity_thresholds,'sensitivity'); _thresholds(protocol,mcc_thresholds,'mcc')
    sensitivity_decision=np.zeros(len(y),bool); mcc_decision=np.zeros(len(y),bool); recovered=np.zeros(len(y),bool)
    candidates={entry.fold_id:entry for entry in oof.candidate_predictions}
    spaces=dict(schema.candidates_by_fold); versions=dict(schema.event_schema_versions)
    if len(candidates)!=len(oof.candidate_predictions) or set(candidates)!=set(spaces) or set(candidates)!={f.fold_id for f in protocol.folds}:
        raise ValueError('Aligned candidate event matrices required for every fold')
    for fold in protocol.folds:
        ix=np.asarray(fold.test); ids=tuple(oof.patient_ids[i] for i in ix)
        if not np.all(oof.fold_ids[ix]==fold.fold_id): raise ValueError('OOF fold assignment mismatch')
        sensitivity_decision[ix]=sensitivity_thresholds[fold.fold_id].apply(s[ix],test_patient_ids=ids)
        mcc_decision[ix]=mcc_thresholds[fold.fold_id].apply(s[ix],test_patient_ids=ids)
        candidate=candidates[fold.fold_id]; matrix=np.asarray(candidate.scores,float)
        if candidate.patient_ids!=ids or candidate.masked_target!=schema.target_event or candidate.event_schema_version!=versions[fold.fold_id] or candidate.candidate_events!=spaces[fold.fold_id]:
            raise ValueError('Top-k must share patients, folds, masked target and event space')
        if len(set(candidate.candidate_events))!=len(candidate.candidate_events) or schema.target_event not in candidate.candidate_events or schema.top_k>len(candidate.candidate_events):
            raise ValueError('Invalid Top-k candidate space')
        if matrix.shape!=(len(ix),len(candidate.candidate_events)) or not np.isfinite(matrix).all() or np.any((matrix<0)|(matrix>1)):
            raise ValueError('Invalid OOF candidate scores')
        target_index=candidate.candidate_events.index(schema.target_event)
        if not np.allclose(matrix[:,target_index],s[ix],rtol=1e-10,atol=1e-12):
            raise ValueError('Target score differs from the shared masked-context candidate matrix')
        recovered[ix]=np.any(np.argsort(-matrix,axis=1,kind='stable')[:,:schema.top_k]==target_index,axis=1)
    return dict(auc=float(roc_auc_score(y,s)),auprc=float(average_precision_score(y,s)),
        specificity_at_validation_90_sensitivity=float(np.mean(~sensitivity_decision[y==0])),
        achieved_test_sensitivity=float(np.mean(sensitivity_decision[y==1])),
        mcc=float(matthews_corrcoef(y,mcc_decision)),top_k_recovery=float(np.mean(recovered[y==1])),
        top_k=schema.top_k,top_k_denominator='eligible target-positive patients',auprc_definition='average precision')
