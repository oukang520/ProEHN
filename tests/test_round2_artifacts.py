import numpy as np
import pandas as pd
import pytest
from proehn.features import FoldPreprocessor
from proehn.artifacts import ScientificArtifactMetadata, save_topology_artifact, load_topology_artifact
from proehn.topology import ProEHNTopologyModel


def test_artifact_roundtrip_preserves_projection_and_clip_bounds(tmp_path):
    prep=FoldPreprocessor(['Age_at_Diagnosis']).fit(pd.DataFrame({'Age_at_Diagnosis':[20,40]}))
    model=ProEHNTopologyModel(1,1,log_rate_clip_min=-2,log_rate_clip_max=3)
    params=np.arange(model.shapes.total_size,dtype=float)
    meta=ScientificArtifactMetadata(('A',),tuple(prep.feature_names),tuple(prep.columns),prep.metadata(),-2,3,
        dict(regularization_strength=.01,l1_ratio=1.,l2_floor=0.),'synthetic-event-v1','synthetic-raw-v1')
    path=tmp_path/'synthetic.npz'
    save_topology_artifact(path,params,meta)
    loaded,p,m,transform=load_topology_artifact(path)
    z=np.r_[1,transform.transform(pd.DataFrame({'Age_at_Diagnosis':[30]}))[0]]
    for a,b in zip(model.compute_patient_params(*model.parse_params(params),z,-2,3),loaded.compute_patient_params(*loaded.parse_params(p),z,m.log_rate_clip_min,m.log_rate_clip_max)):
        np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError,match='conflicts'):
        load_topology_artifact(path,log_rate_clip_max=20)
