from dataclasses import replace
import pandas as pd
import pytest
from proehn.labels import ProgressionLabelSchema,require_frozen_label_protocol,precomputed_progression_labels,label_provenance
from proehn.benchmark import FormalProEHNTrainer


def test_formal_evaluation_refuses_unfrozen_protocol_before_data_access():
    schema=ProgressionLabelSchema('synthetic','pfs','event','days',10)
    with pytest.raises(ValueError,match='PROTOCOL_FREEZE'):
        FormalProEHNTrainer(None,schema).fit(None,None)
    frozen=replace(schema,schema_id='toy',version='v1',frozen=True)
    frame=pd.DataFrame({'Patient_Label':[0,1]})
    with pytest.raises(ValueError,match='provenance'):
        precomputed_progression_labels(frame,'Patient_Label',frozen)
    frame.attrs['label_provenance']={'Patient_Label':label_provenance(frozen)}
    assert precomputed_progression_labels(frame,'Patient_Label',frozen).tolist()==[0,1]
