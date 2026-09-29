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
    import json
    from dataclasses import asdict
    incomplete=asdict(meta);incomplete.pop('observation_semantics_version')
    np.savez(path,params=params,scientific_metadata_json=json.dumps(incomplete))
    with pytest.raises(ValueError,match='Incomplete'):
        load_topology_artifact(path)


def test_kinetic_artifact_preserves_validation_calibration_without_training(tmp_path):
    import jax
    import jax.numpy as jnp
    from proehn.kinetic import ProEHNKineticGatekeeper,KineticGatekeeperNetwork
    from proehn.preprocessing import KineticPreprocessor
    from proehn.probability import KineticProbabilityProtocol,PlattCalibration
    gate=ProEHNKineticGatekeeper(None,None)
    gate.preprocessor=KineticPreprocessor(('Age_at_Diagnosis',)).fit(pd.DataFrame({'Age_at_Diagnosis':[20,40]}))
    gate.feature_groups=gate.preprocessor.feature_groups
    gate.model_config=dict(d_model=2,n_head_layers=0,dropout_rate=0.)
    gate.model=KineticGatekeeperNetwork(**gate.model_config)
    args=[jnp.zeros((1,len(gate.feature_groups[k]))) for k in ('pt_genomic','pt_dynamic','mt_genomic','mt_dynamic','shared')]
    gate.params=gate.model.init(jax.random.PRNGKey(0),*args,train=False)['params']
    gate.probability_protocol=KineticProbabilityProtocol(1.,'platt')
    gate.calibration=PlattCalibration(.7,.2,('synthetic-validation',))
    from dataclasses import asdict
    from proehn.labels import ProgressionLabelSchema,label_provenance
    from proehn.features import FrozenCovariateProtocol
    gate.scientific_training_protocol=dict(
        label_protocol=label_provenance(ProgressionLabelSchema('toy','pfs','event','days',10,schema_id='toy',version='1',frozen=True)),
        covariates=asdict(FrozenCovariateProtocol('toy','toy',(),('Age_at_Diagnosis',),True)),
        event_schema_version='toy-v1',training_feature_provenance_version='toy-v1')
    gate.ready=True
    params=tmp_path/'synthetic.msgpack'; meta=tmp_path/'synthetic.pkl'
    gate.save(params,meta)
    loaded=ProEHNKineticGatekeeper(params,meta)
    assert loaded.probability_protocol.focal_gamma==1
    np.testing.assert_allclose(loaded.predict_go_probability({'Age_at_Diagnosis':30}),gate.predict_go_probability({'Age_at_Diagnosis':30}),rtol=0,atol=0)
