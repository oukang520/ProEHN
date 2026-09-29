import numpy as np
import jax.numpy as jnp
import pytest
from proehn.representation import patient_transition_representation_observed as observed, ExactInferenceLimitError
from proehn.topology import ProEHNTopologyModel
from proehn.ctmc import vanilla as v, likelihood as ll


def setup(n=1):
    model=ProEHNTopologyModel(n,0)
    return model,jnp.zeros(model.shapes.total_size),jnp.ones(1),[str(i) for i in range(n)]


def test_partial_pt_posterior_matches_dense_resolvent():
    model,params,z,genes=setup()
    result=observed(model,params,z,genes,[0],None,4)
    theta,dp,dm=model.compute_patient_params(*model.parse_params(params),z)
    state=jnp.ones(3,int)
    q=np.column_stack([v.kronvec(theta,jnp.array(e),state) for e in np.eye(8)])
    d=np.asarray(v.diag_scal_p(dp,state,jnp.ones(8)))
    mass=d*np.linalg.solve(np.diag(d)-q,np.eye(8)[0])
    mass[1::2]=0
    expected=mass/mass.sum()
    indices=result.posterior_states@np.array([1,2,4])
    np.testing.assert_allclose(result.posterior_weights,expected[indices])
    np.testing.assert_allclose(result.conditional_probabilities.sum(),1.)
    np.testing.assert_allclose(result.integrated_scores(.3).sum(),.3)
    assert {int(s[-1]) for s in result.posterior_states} == {0,1}
    assert {int(s[1]) for s in result.posterior_states if s[-1]} == {0,1}


def test_partial_pt_posterior_seeding_matches_primary_likelihood():
    model,p,z,g=setup()
    result=observed(model,p,z,g,[0],None,4)
    theta,dp,_=model.compute_patient_params(*model.parse_params(p),z)
    a=np.exp(ll._lp_prim_obs(theta,dp,jnp.array([0,0]),0))
    b=np.exp(ll._lp_prim_obs(theta,dp,jnp.array([0,1]),1))
    np.testing.assert_allclose(result.posterior_weights[result.posterior_states[:,-1]==1].sum(),b/(a+b))


def test_known_snapshot_and_paired_history_are_distinct():
    args=setup()
    snapshot=observed(*args,[0],[0],3,joint_snapshot=True)
    history=observed(*args,[0],[0],3,diagnosis_order=1)
    assert snapshot.representation_kind=='fully_observed'
    assert history.representation_kind=='latent_marginalized'
    assert np.any(history.posterior_states[:,0]==1)  # PT can evolve after its sample
    np.testing.assert_allclose(history.posterior_weights.sum(),1)
    assert not np.any(history.posterior_states[:,-1]==0)


def test_no_silent_large_state_approximation():
    with pytest.raises(ExactInferenceLimitError):
        observed(*setup(2),[0,0],None,4,max_state_bits=3)


def test_terminal_mass_is_exposed_not_renormalized_away():
    result=observed(*setup(),[1],[1],3,joint_snapshot=True)
    assert result.terminal_probability==1
    assert result.accessible_events == (('terminal','No accessible event'),)
    assert result.conditional_probabilities.sum()==1
