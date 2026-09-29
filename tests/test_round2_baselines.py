from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from proehn.baselines import ProEHNTopologyOnlyTrainer,ExternalBaselineContract,MHNTrainer,OncotreeTrainer,HyperTraPSTrainer
from proehn.evaluation import BenchmarkFold


def test_topology_only_reuses_exact_parameters_and_fold_without_fit():
    params=np.arange(3.)
    full=SimpleNamespace(spec=SimpleNamespace(patient_id_column='id'),topology_params=params,
        training_patient_ids=('a',),validation_patient_ids=('b',),
        _predict=lambda f,progression_override:np.array([[progression_override,.2,.2]]))
    fold=BenchmarkFold(0,(0,),(1,),(2,))
    trainer=ProEHNTopologyOnlyTrainer(full,fold,('a','b','c'))
    result=trainer.fit(pd.DataFrame({'id':['a']}),pd.DataFrame({'id':['b']}))
    assert result.topology_params is params and result.fold is fold
    assert result.predict(None)[0,0]==1
    with pytest.raises(ValueError,match='exact'):
        trainer.fit(pd.DataFrame({'id':['c']}),pd.DataFrame({'id':['b']}))


@pytest.mark.parametrize('name,cls',[('MHN',MHNTrainer),('Oncotree',OncotreeTrainer),('HyperTraPS',HyperTraPSTrainer)])
def test_missing_baseline_dependency_is_not_replaced_with_surrogate(name,cls):
    contract=ExternalBaselineContract(name,'explicit external package','pending-verification','toy-event-v1')
    with pytest.raises(RuntimeError,match='EXTERNAL_DEPENDENCY'):
        cls(None,contract).fit(None,None)
