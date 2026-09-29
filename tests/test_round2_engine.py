import numpy as np
from proehn.engine import ProEHNEngine
from proehn.topology import ProEHNTopologyModel


def test_formal_engine_uses_partial_api_and_only_operational_score_names():
    engine=ProEHNEngine()
    engine.topology_model=ProEHNTopologyModel(1,0)
    engine.topology_params=np.zeros(engine.topology_model.shapes.total_size)
    engine.topology_ready=True
    engine.gene_names=['A']
    engine.kinetic.predict_go_probability=lambda _: .4
    result=engine.predict({'P.A (M)':0,'observation_type':4})
    assert result['representation_kind']=='latent_marginalized'
    assert 'operational_stop_predicted' not in result
    assert 'is_stable_predicted' not in result and 'absolute_risks' not in result
    assert all('absolute_risk' not in r for r in result['integrated_event_scores'])
    np.testing.assert_allclose(sum(r['integrated_event_score'] for r in result['integrated_event_scores']),.4)
