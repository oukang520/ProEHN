import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
from proehn.robustness import RobustnessProtocol,outer_test_feature_perturbation
from proehn.evaluation import BenchmarkFold


def test_robustness_only_changes_outer_test_genomics_without_fitting():
    frame=pd.DataFrame({'id':['a','b','c'],'raw':[0,1,0],'outcome':[1,0,1]})
    fold=BenchmarkFold(0,(0,),(1,),(2,))
    protocol=RobustnessProtocol('outer_test_feature_perturbation',('raw',),1.,42)
    def builder(df):
        assert df.id.tolist()==['c'] and df.outcome.tolist()==[1]
        return df[['raw']]
    builder.raw_genomic_columns={'raw'}
    predictor=SimpleNamespace(predict=lambda x:x.to_numpy())
    result=outer_test_feature_perturbation(frame,frame.id,fold,protocol,perturb=lambda x,*args:1-x,feature_builder=builder,fitted_predictor=predictor)
    assert result.tolist()==[[1]] and frame.raw.tolist()==[0,1,0]
    assert RobustnessProtocol('training_label_noise',('label',),.1,42).requires_retraining
    assert 'not trained-model robustness' in RobustnessProtocol('evaluation_label_sensitivity',('label',),.1,42).interpretation
