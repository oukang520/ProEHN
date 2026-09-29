from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from proehn.events import select_event_schema
from proehn.baselines import ExternalBaselineContract,MHNTrainer,ProEHNTopologyOnlyPredictor
from proehn.burden import GenomicBurdenProtocol,genomic_burdens
from proehn.probability import evaluate_probability_calibration
from proehn.evaluation import BenchmarkResult
from final_test_support import event_protocol


def test_external_wrapper_shares_event_schema_and_never_passes_outcomes():
    from test_scientific_contracts import recovery_fixture
    frame,spec=recovery_fixture();protocol=event_protocol(2)
    spec=replace(spec,event_selection_protocol=protocol,event_schema_version=protocol.schema_version)
    expected=select_event_schema(frame.iloc[:8],spec)
    frame['secret_test_label']=np.arange(len(frame))
    contract=ExternalBaselineContract('MHN','official-backend-test-spy','test',protocol.schema_version)
    seen=[]
    class Backend:
        scientific_contract=contract
        def fit(self,training,validation,*,spec):
            assert 'Patient_Label' not in training and 'secret_test_label' not in validation
            assert len(training)==8 and len(validation)==2
            seen.append(tuple(training.columns))
            return self
        def predict_conditional_target(self,features,*,spec):
            assert 'secret_test_label' not in features and 'raw_pfs_days' not in features
            assert (features['P.A (M)']==0).all()
            return np.full(len(features),.5)
    predictor=MHNTrainer(spec,contract,Backend()).fit(frame.iloc[:8],frame.iloc[8:10])
    assert predictor.selected_event_schema.schema_version==expected.schema_version
    assert predictor.predict(spec.features(frame.iloc[10:])).shape==(2,3)
    full=SimpleNamespace(spec=spec,selected_event_schema=expected,
        predict_candidate_scores=lambda features,progression_override: np.ones((len(features),2))*progression_override)
    only=ProEHNTopologyOnlyPredictor(full,None)
    assert only.selected_event_schema is full.selected_event_schema
    assert only.selected_event_schema.target_candidate_events('primary')==expected.target_candidate_events('primary')


def test_raw_cna_counts_are_not_assumed_comparable_across_panels():
    frame=pd.DataFrame({'assay':['a','b'],'mutation':[2,4],'cna':[1,2],'mb':[1.,2.],'loci':[10,20]})
    raw=GenomicBurdenProtocol('GENIE','toy','1','raw_count','raw_count','altered_locus_count','coverage_adjusted','assay',('a','b'),frozen=True)
    with pytest.raises(ValueError,match='Mixed-assay'):
        genomic_burdens(frame,raw,mutation_count_column='mutation',cna_numerator_column='cna')
    adjusted=replace(raw,mutation_representation='tmb',cna_representation='panel_adjusted',callable_mb_column='mb',cna_coverage_column='loci')
    result=genomic_burdens(frame,adjusted,mutation_count_column='mutation',cna_numerator_column='cna')
    np.testing.assert_allclose(result['mutation_burden'],[2,2])
    np.testing.assert_allclose(result['cna_burden'],[.1,.1])


def test_calibration_diagnostics_are_separate_from_prediction_calibration():
    scores=np.array([.1,.3,.6,.8,.2,.7]);labels=np.array([0,1,0,1,0,1])
    oof=BenchmarkResult('synthetic','Go',tuple('abcdef'),np.arange(6)%3,scores.copy())
    result=evaluate_probability_calibration(oof,labels,bins=3)
    assert sum(bin['count'] for bin in result['reliability_curve'])==6
    assert 0<=result['brier_score']<=1
    np.testing.assert_array_equal(oof.scores,scores)


def test_adjusted_burden_is_rebuilt_after_target_masking():
    from proehn.features import TargetMaskingProtocol,build_leave_target_out_features
    bp=GenomicBurdenProtocol('toy','burden','1','tmb','panel_adjusted','altered_locus_count','coverage_adjusted','assay',('a','b'),callable_mb_column='mb',cna_coverage_column='covered',frozen=True)
    protocol=TargetMaskingProtocol(('target','other'),('target_vaf','other_vaf'),('target_cna','other_cna'),('target',),('target_cna',),
        burden_protocol=bp)
    raw=pd.DataFrame({'assay':['a','b'],'mb':[1.,2.],'covered':[10,20],
        'target':[1,0],'other':[1,1],'target_vaf':[.5,0.],'other_vaf':[.2,.2],
        'target_cna':[1,0],'other_cna':[1,1]})
    a=build_leave_target_out_features(raw,protocol)
    raw['target']=1-raw.target;raw['target_cna']=1-raw.target_cna;raw['target_vaf']=.9
    b=build_leave_target_out_features(raw,protocol)
    pd.testing.assert_frame_equal(a,b)
    assert 'nMut_Primary' not in a and 'CNA_Primary' not in a
    np.testing.assert_allclose(a.TMB_Primary,[1,.5])
    np.testing.assert_allclose(a.CNA_adjusted_Primary,[.1,.05])
