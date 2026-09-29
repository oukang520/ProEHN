import numpy as np
import pandas as pd
from proehn.benchmark import TargetRecoverySpec,FormalProEHNPredictor
from proehn.features import TargetMaskingProtocol,TopologyCovariateSchema
from proehn.preprocessing import TopologyTrainingPreprocessor
from proehn.kinetic import ProEHNKineticGatekeeper
from proehn.topology import ProEHNTopologyModel


def fixture():
    protocols=tuple(TargetMaskingProtocol((c+'a',),(c+'v',),(),(c+'a',),(),compartment=comp) for c,comp in [('p','Primary'),('m','Metastatic')])
    spec=TargetRecoverySpec('A','primary',protocols,(('P.A (M)',('pa',)),('M.A (M)',('ma',))))
    raw=pd.DataFrame({'pa':[1,0],'pv':[.3,np.nan],'ma':[np.nan,np.nan],'mv':[np.nan,np.nan],'observation_type':[4,4]})
    return raw,spec


def test_partial_recovery_needs_no_seeding_or_metastatic_plugin_state():
    raw,spec=fixture();safe=spec.features(raw)
    assert 'Seeding' not in safe
    assert safe.nMut_Metastatic.isna().all()
    assert (safe['P.A (M)']==0).all()
    prep=TopologyTrainingPreprocessor(1,schema=TopologyCovariateSchema(())).fit(spec.training_frame(raw))
    model=ProEHNTopologyModel(1,0)
    gate=ProEHNKineticGatekeeper(None,None);gate.ready=True;gate.predict_go_probability=lambda _: .3
    predictor=FormalProEHNPredictor(model,np.zeros(model.shapes.total_size),prep,gate,spec)
    result=predictor.predict(safe)
    np.testing.assert_allclose(result[:,2],.3*result[:,1])
    assert np.isfinite(result).all()


def test_absent_mt_raw_columns_and_shared_baseline_are_not_fake_measurements():
    from dataclasses import replace
    raw,spec=fixture()
    raw=raw.drop(columns=['ma','mv']);raw['Age_at_Diagnosis']=[40,50]
    spec=replace(spec,masking_protocols=tuple(replace(p,baseline_columns=('Age_at_Diagnosis',)) for p in spec.masking_protocols))
    safe=spec.features(raw)
    assert safe.Age_at_Diagnosis.tolist()==[40,50]
    assert safe.nMut_Metastatic.isna().all()
    assert spec.training_frame(raw)['M.A (M)'].isna().all()
