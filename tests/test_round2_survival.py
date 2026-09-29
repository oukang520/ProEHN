from dataclasses import replace
import numpy as np
import pytest
from proehn.evaluation import BenchmarkResult,CalibratedThreshold
from proehn.survival import OOFSurvivalScores,harrell_c_index,paired_delta_c_permutation,kaplan_meier,logrank


def toy():
    result=BenchmarkResult('synthetic','progression',tuple('abcdef'),np.array([0,0,1,1,2,2]),np.array([.9,.8,.7,.3,.2,.1]))
    return OOFSurvivalScores(result,np.arange(1,7),np.ones(6,int))


def test_synthetic_paired_null_and_alignment():
    full=toy()
    identical=replace(full,predictions=replace(full.predictions,model_name='synthetic topology'))
    result=paired_delta_c_permutation(full,identical,repetitions=9,random_seed=42)
    assert result['p_value']==1 and result['delta_c']==0
    wrong=replace(identical,predictions=replace(identical.predictions,patient_ids=tuple('bacdef')))
    with pytest.raises(ValueError,match='alignment'):
        paired_delta_c_permutation(full,wrong,repetitions=9,random_seed=42)


def test_fixed_risk_direction_and_validation_only_km():
    oof=toy()
    assert harrell_c_index(oof)==1
    assert harrell_c_index(replace(oof,predictions=replace(oof.predictions,scores=1-oof.risk)))==0
    threshold=CalibratedThreshold(.5,'prespecified',())
    assert set(kaplan_meier(oof,threshold))=={False,True}
    assert 0<=logrank(oof,threshold)['p_value']<=1
    with pytest.raises(ValueError,match='overlap'):
        oof.km_groups(CalibratedThreshold(.5,'validation',('a',)))
