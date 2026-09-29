from dataclasses import replace
import inspect
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import yaml
from proehn.evaluation import (FormalFoldProtocol,GroupedStratificationPolicy,TargetNotEvaluable,
    run_formal_oof_benchmark,patient_outer_folds,CalibratedThreshold,PrespecifiedThresholdProtocol,calibrate_threshold,BenchmarkResult)
from proehn.metrics import CandidateFoldScores,TargetRecoveryMetricSchema,evaluate_target_recovery
from proehn.engine import ProEHNEngine
from proehn.scientific_config import validate_model_parameters
from proehn.survival import SurvivalAdjustmentProtocol
from final_test_support import explicit_configs,event_protocol


def folds():
    ids=tuple(f'p{i:02}' for i in range(16));y=np.arange(16)%2
    return FormalFoldProtocol.create(ids,y,y,target='P.A (M)',policy=GroupedStratificationPolicy(4,2))


def test_formal_folds_reject_generic_and_require_both_stop_go():
    with pytest.raises(ValueError,match='provenance'):
        run_formal_oof_benchmark(None,patient_outer_folds(range(16),4),{},event_spec=None)
    protocol=folds().validate()
    for fold in protocol.folds:
        for partition in (fold.train,fold.validation):
            assert set(np.asarray(protocol.kinetic_labels)[list(partition)])=={0,1}
    with pytest.raises(TargetNotEvaluable,match='Stop/Go'):
        FormalFoldProtocol.create(protocol.patient_ids,protocol.target_labels,np.ones(16,int),target=protocol.target,policy=protocol.policy)
    with pytest.raises(ValueError,match='match'):
        replace(protocol,folds=patient_outer_folds(protocol.patient_ids,4)).validate()


def metric_fixture():
    protocol=folds();y=np.asarray(protocol.target_labels);scores=.2+.6*y
    fold_ids=np.empty(16,int);versions=[];candidates=[];spaces=[];threshold_s={};threshold_m={}
    events=(('primary','A'),('primary','B'))
    for f in protocol.folds:
        ix=np.array(f.test);fold_ids[ix]=f.fold_id;version=f'synthetic-{f.fold_id}'
        versions.append((f.fold_id,version));spaces.append((f.fold_id,events))
        candidates.append(CandidateFoldScores(f.fold_id,tuple(protocol.patient_ids[i] for i in ix),events,events[0],version,np.column_stack((scores[ix],1-scores[ix]))))
        validation=tuple(protocol.patient_ids[i] for i in f.validation)
        threshold_s[f.fold_id]=CalibratedThreshold(.5,'sensitivity',validation,.9)
        threshold_m[f.fold_id]=CalibratedThreshold(.5,'mcc',validation)
    result=BenchmarkResult('synthetic',protocol.target,protocol.patient_ids,fold_ids,scores,protocol,tuple(versions),tuple(candidates))
    schema=TargetRecoveryMetricSchema(protocol.target,events[0],protocol.patient_ids,tuple(versions),tuple(spaces),1)
    return result,y,schema,threshold_s,threshold_m


def test_metric_layer_only_applies_frozen_validation_thresholds(monkeypatch):
    import proehn.evaluation as evaluation
    result,y,schema,sensitivity,mcc=metric_fixture()
    def forbidden(*args,**kwargs): raise AssertionError('Test outcomes entered threshold selection')
    monkeypatch.setattr(evaluation,'calibrate_threshold',forbidden)
    values=evaluate_target_recovery(result,y,event_schema=schema,sensitivity_thresholds=sensitivity,mcc_thresholds=mcc)
    assert values['top_k_recovery']==1 and values['mcc']==1
    wrong=dict(mcc);wrong[0]=replace(wrong[0],calibration_patient_ids=tuple(result.patient_ids[i] for i in result.fold_protocol.folds[0].test))
    with pytest.raises(ValueError,match='validation'):
        evaluate_target_recovery(result,y,event_schema=schema,sensitivity_thresholds=sensitivity,mcc_thresholds=wrong)
    altered=replace(result,candidate_predictions=(replace(result.candidate_predictions[0],candidate_events=(('primary','B'),('primary','A'))),*result.candidate_predictions[1:]))
    with pytest.raises(ValueError,match='event space'):
        evaluate_target_recovery(altered,y,event_schema=schema,sensitivity_thresholds=sensitivity,mcc_thresholds=mcc)


def test_scientific_configs_are_explicit_and_do_not_freeze_clinical_labels():
    root=Path(__file__).resolve().parents[1]
    for cohort in ('paca','mela','luad'):
        config=yaml.safe_load((root/f'configs/cohorts/{cohort}_scientific.yaml').read_text())
        validate_model_parameters(config['kinetic'],config['topology'],config['configuration_source'])
        assert config['kinetic']['focal_gamma']==0 and config['label_protocol']['frozen'] is False
        assert config['posterior_inference']['method']=='sequential_importance'
    kinetic,topology=explicit_configs()
    with pytest.raises(ValueError,match='missing'):
        validate_model_parameters(kinetic,topology,'prespecified')


def test_engine_refuses_unprovenanced_runtime_stop_threshold():
    with pytest.raises(ValueError,match='provenance'):
        ProEHNEngine(stop_threshold=.9)
    threshold=PrespecifiedThresholdProtocol(.9,'synthetic-only','operational_stop')
    assert ProEHNEngine(stop_threshold=threshold).stop_threshold is threshold
    with pytest.raises(ValueError,match='provenance'):
        ProEHNEngine(stop_threshold=replace(threshold,score_semantics='target_presence'))


def test_survival_adjustment_requires_frozen_baseline_list():
    with pytest.raises(ValueError,match='Frozen'):
        SurvivalAdjustmentProtocol('toy',('Age_at_Diagnosis',),'toy','1','error').validate()
    with pytest.raises(ValueError,match='Outcome'):
        SurvivalAdjustmentProtocol('toy',('PFS_days',),'toy','1','error',True).validate()


def test_stop_decision_is_omitted_until_a_provenanced_threshold_is_supplied():
    from proehn.representation import PatientTransitionRepresentation
    representation=PatientTransitionRepresentation(np.zeros((2,2)),np.zeros(2),np.zeros(2),
        (('primary','A'),),np.ones(1))
    engine=ProEHNEngine()
    assert 'operational_stop_predicted' not in engine._result(representation,.2)
    engine.stop_threshold=PrespecifiedThresholdProtocol(.7,'toy-stop','operational_stop')
    assert engine._result(representation,.2)['operational_stop_predicted'] is True
    engine.stop_threshold=CalibratedThreshold(.7,'mcc',('validation',),score_semantics='operational_stop')
    with pytest.raises(ValueError,match='isolation'):
        engine._result(representation,.2)
    with pytest.raises(ValueError,match='overlap'):
        engine._result(representation,.2,patient_id='validation')
