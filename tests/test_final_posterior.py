import numpy as np
import jax.numpy as jnp
import pytest
from proehn.topology import ProEHNTopologyModel
from proehn.representation import posterior_transition_representation as infer


@pytest.mark.parametrize('kind,pt,mt,order',[(0,[0],None,None),(1,[0],None,None),(4,[0],None,None),(2,None,[1],None),(3,[0],[1],0),(3,[0],[1],1),(3,[0],[1],2)])
def test_importance_backend_matches_exact_resolvent(kind,pt,mt,order):
    model=ProEHNTopologyModel(1,0);params=jnp.linspace(-.3,.4,model.shapes.total_size)
    args=(model,params,jnp.ones(1),['A'],pt,mt,kind)
    exact=infer(*args,method='exact',diagnosis_order=order)
    sample=infer(*args,method='sequential_importance',diagnosis_order=order,particles=6000,random_seed=17)
    truth=dict(zip(exact.accessible_events,exact.conditional_probabilities));estimate=dict(zip(sample.accessible_events,sample.conditional_probabilities))
    assert max(abs(truth.get(key,0)-estimate.get(key,0)) for key in set(truth)|set(estimate))<.04
    assert sample.inference_diagnostics['approximation'] is True
    np.testing.assert_allclose(sample.integrated_scores(.4).sum(),.4)


def test_large_posterior_does_not_call_full_joint_enumeration(monkeypatch):
    from proehn.ctmc import vanilla
    def forbidden(*args,**kwargs): raise AssertionError('Full joint states enumerated')
    monkeypatch.setattr(vanilla,'_states',forbidden)
    model=ProEHNTopologyModel(12,0)
    result=infer(model,jnp.zeros(model.shapes.total_size),jnp.ones(1),[str(i) for i in range(12)],[0]*12,None,4,
        method='sequential_importance',particles=256,minimum_effective_samples=2.)
    assert result.posterior_states.shape==(256,25)
    np.testing.assert_allclose(result.conditional_probabilities.sum(),1)
