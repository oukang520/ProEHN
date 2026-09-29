"""Manually constructed observation histories only; no cohort IO or fitting."""
import itertools
import numpy as np
import pandas as pd
import jax.numpy as jnp
import pytest
from proehn.ctmc import likelihood as ll, vanilla as v
from proehn.preprocessing import build_topology_training_data


def test_pt_first_can_seed_after_sampling():
    theta = jnp.zeros((2,2)); dp=dm=jnp.zeros(2); target=jnp.array([0,0,1])
    late = ll._paired_order_probability(theta,dp,dm,target,1,1,True,0)
    early = ll._paired_order_probability(theta,dp,dm,target,1,1,True,1)
    all_pt = np.exp(ll._lp_coupled_1(theta,dp,dm,target,1,1))
    assert late > 0 and early > 0
    np.testing.assert_allclose(all_pt,late+early)
    # Pre-first occupation 1/(PT diagnosis + mutation + seeding)=1/3;
    # then seed before mutation with probability 1/2; then MT diagnosis
    # before MT mutation with probability 1/2 => late-seeding mass 1/12.
    np.testing.assert_allclose(late, 1/12)


def test_mt_first_excludes_unseeded_first_observation():
    a=(jnp.zeros((2,2)),jnp.zeros(2),jnp.zeros(2),jnp.array([0,0,1]),1,1)
    assert ll._paired_order_probability(*a,False,0) == 0
    np.testing.assert_allclose(ll._paired_order_probability(*a,False),ll._paired_order_probability(*a,False,1))
    frame=pd.DataFrame({'P.A (M)':[0],'M.A (M)':[0],'observation_type':[3], 'diag_order':[2], 'seeding_at_first_observation':[0]})
    with pytest.raises(ValueError,match='MT-first'):
        build_topology_training_data(frame)


def test_unknown_order_and_first_seed_marginalize():
    for p,m in itertools.product([0,1],repeat=2):
        a=(jnp.array([[-.3,.1],[.2,-.4]]),jnp.array([.3,.2]),jnp.array([-.2,.1]),jnp.array([p,m,1]),p+1,m+1)
        np.testing.assert_allclose(np.exp(ll._lp_coupled_0(*a)),np.exp(ll._lp_coupled_1(*a))+np.exp(ll._lp_coupled_2(*a)))
        early=ll._paired_order_probability(*a,True,1)
        late=ll._paired_order_probability(*a,True,0)
        np.testing.assert_allclose(early+late,np.exp(ll._lp_coupled_1(*a)))


def test_paired_eventual_seeding_validation():
    frame=pd.DataFrame({'P.A (M)':[0],'M.A (M)':[0],'observation_type':[3], 'diag_order':[1], 'observed_seeding':[0]})
    with pytest.raises(ValueError,match='conflicts'):
        build_topology_training_data(frame)
