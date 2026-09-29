import inspect
import numpy as np
import pandas as pd
from types import SimpleNamespace
from proehn.benchmark import FormalProEHNTrainer
from proehn.labels import ProgressionLabelSchema


def test_formal_selection_only_receives_inner_partitions(monkeypatch):
    import proehn.benchmark as b
    schema=ProgressionLabelSchema('toy','pfs','event','days',10,schema_id='synthetic',version='1',frozen=True)
    spec=SimpleNamespace(patient_id_column='patient_id',training_frame=lambda x:x)
    trainer=FormalProEHNTrainer(spec,schema,configuration_source='inner_validation',topology_candidates=({'regularization_strength':.01},{'regularization_strength':.1}))
    seen=[]
    def fitted(self,t,v):
        seen.append((tuple(t.patient_id),tuple(v.patient_id)))
        return SimpleNamespace(topology=SimpleNamespace(bucket_loss=lambda *args:self.topology_config['regularization_strength']),topology_params=np.zeros(1),topology_preprocessor=None)
    monkeypatch.setattr(FormalProEHNTrainer,'_fit_prespecified',fitted)
    monkeypatch.setattr(b,'build_topology_training_data',lambda *args,**kwargs: ([(0,0,0,np.zeros((1,1)),np.ones((1,1)))],1,0,1,[],[]))
    result=trainer.fit(pd.DataFrame({'patient_id':['train']}),pd.DataFrame({'patient_id':['validation']}))
    assert seen==[(('train',),('validation',))]*2
    assert result.selection_metadata['selected']=={'regularization_strength':.01}
    assert 'test' not in inspect.signature(FormalProEHNTrainer.fit).parameters
