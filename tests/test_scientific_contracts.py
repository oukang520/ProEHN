"""Synthetic-only correctness checks. No cohorts, artifact loading, training or metrics."""
import ast
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import jax
import jax.numpy as jnp

from proehn.features import (FoldPreprocessor, TopologyCovariateSchema,
    TargetMaskingProtocol, build_leave_target_out_features, FEATURE_METADATA,
    panel_corrected_mutation_burden)
from proehn.preprocessing import (TopologyTrainingPreprocessor, KineticPreprocessor,
    build_topology_training_data, build_kinetic_matrices, rank_gene_pairs, find_gene_pairs,
    select_numeric_feature_columns, split_kinetic_feature_columns)
from proehn.observations import ObservationType, OBSERVATION_SEMANTICS
from proehn.topology import ProEHNTopologyModel
from proehn.ctmc import vanilla, likelihood
from proehn.evaluation import (patient_outer_folds, CalibratedThreshold, calibrate_threshold,
    BenchmarkResult, run_oof_benchmark, survival_score_inputs, AblationInputs, ablation_full_vs_evolution)
from proehn.labels import ProgressionLabelSchema, build_paca_progression_label, build_mela_progression_label, build_lc_progression_label
from proehn.benchmark import TargetRecoverySpec, FormalProEHNTrainer, FormalProEHNPredictor
from proehn.kinetic import ProEHNKineticGatekeeper
from proehn.representation import patient_transition_representation, model_implied_transition_path


@pytest.fixture(autouse=True)
def prohibit_cohort_io_and_training(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Scientific unit tests must not load cohort/artifact files or optimize models')
    monkeypatch.setattr(pd, 'read_csv', forbidden)
    monkeypatch.setattr(np, 'load', forbidden)
    monkeypatch.setattr('proehn.training.minimize', forbidden)
    monkeypatch.setattr('proehn.training.fit_kinetic_model', forbidden)
    monkeypatch.setattr('proehn.benchmark.fit_kinetic_model', forbidden)
    monkeypatch.setattr('proehn.benchmark.fit_topology_model', forbidden)


def tiny_frame(n=12):
    return pd.DataFrame({'patient_id': np.repeat(np.arange(n//2), 2),
        'P.A (M)': np.arange(n)%2, 'M.A (M)': np.zeros(n),
        'P.B (M)': (np.arange(n)//2)%2, 'M.B (M)': np.zeros(n),
        'Age_at_Diagnosis': np.arange(n)+40., 'Patient_Label': np.arange(n)%2,
        'type': np.arange(n)%2, 'Seeding': np.arange(n)%2})


def test_scaler_fits_only_training_and_is_batch_invariant():
    df = tiny_frame()
    fold = patient_outer_folds(df.patient_id, n_splits=3)[0]
    prep = FoldPreprocessor(['Age_at_Diagnosis']).fit(df.iloc[list(fold.train)])
    expected = df.iloc[list(fold.train)].Age_at_Diagnosis.mean()
    assert prep.mean_[0] == expected
    before = prep.metadata()
    test = df.iloc[list(fold.test)].copy()
    test['Age_at_Diagnosis'] = 10000.
    prep.transform(test)
    assert prep.metadata() == before
    np.testing.assert_allclose(prep.transform(test.iloc[:1]), prep.transform(test)[:1])
    np.testing.assert_allclose(FoldPreprocessor.from_metadata(before).transform(test), prep.transform(test))


def test_imputation_variance_selection_are_training_only():
    train = pd.DataFrame({'a': [1., np.nan, 3.], 'b': [2., 2., 2.]})
    prep = FoldPreprocessor(['a', 'b'], variance_threshold=0).fit(train)
    assert prep.feature_names == ['a']
    np.testing.assert_allclose(prep.transform(pd.DataFrame({'a': [np.nan], 'b': [1000]})), [[0.]])


@pytest.mark.parametrize('outcome', ['PFS_days', 'OS_days', 'PFS_availability', 'OS_availability',
    'survival_event', 'response', 'relapse', 'future_disease_status', 'followup_duration',
    'outcome_label', 'PFS_days_is_Missing', 'response_is_Missing', 'followup_completeness',
    'mystery_outcome_proxy', 'P.followup_is_Missing'])
def test_outcomes_and_missingness_never_enter_topology_or_kinetic(outcome):
    frame = tiny_frame()
    frame[outcome] = 1
    assert outcome not in select_numeric_feature_columns(frame, [])
    assert outcome not in sum(split_kinetic_feature_columns(frame.columns).values(), [])
    with pytest.raises(ValueError):
        TopologyCovariateSchema((outcome,))


def test_patient_groups_never_cross_folds():
    ids = np.repeat(np.arange(8), 2)
    folds = patient_outer_folds(ids, n_splits=4)
    assert sorted(i for f in folds for i in f.test) == list(range(16))
    for fold in folds:
        fold.validate(ids)
        assert not set(ids[list(fold.train)]) & set(ids[list(fold.test)])
        assert not set(ids[list(fold.validation)]) & set(ids[list(fold.test)])


def masking():
    return TargetMaskingProtocol(('target', 'other'), ('v_target', 'v_other'), ('c_target', 'c_other'), ('target',), ('c_target',))


def test_target_removed_from_every_derived_summary():
    raw = pd.DataFrame({'target': [1, 0], 'other': [1, 1], 'v_target': [.8, np.nan],
        'v_other': [.2, .2], 'c_target': [1, 0], 'c_other': [1, 1],
        'nMut_Primary': [999, 999], 'hidden_target_proxy': [1, 0], 'target_is_Missing': [0, 1]})
    out = build_leave_target_out_features(raw, masking())
    np.testing.assert_array_equal(out.nMut_Primary.to_numpy(), [1, 1])
    np.testing.assert_allclose(out.VAF_mean_Primary, [.2, .2])
    np.testing.assert_array_equal(out.CNA_Primary.to_numpy(), [1, 1])
    assert not any('target' in c for c in out)
    assert 'hidden_target_proxy' not in out
    pd.testing.assert_series_equal(out.iloc[0], out.iloc[1], check_names=False)


def test_genes_ignore_test_values_and_seeding_labels():
    df = tiny_frame()
    train, test = df.iloc[:8].copy(), df.iloc[8:].copy()
    pairs = find_gene_pairs(train)
    ranked = rank_gene_pairs(train, pairs)
    test['Seeding'] = 1-test.Seeding
    test['P.B (M)'] = 100
    train['Seeding'] = 1-train.Seeding
    assert ranked == rank_gene_pairs(train, pairs)


def test_test_labels_do_not_calibrate_thresholds():
    threshold = calibrate_threshold([0, 0, 1, 1], [.1, .3, .6, .8], validation_patient_ids=['v1','v2','v3','v4'], criterion='sensitivity')
    before = threshold.value
    for test_labels in ([0, 1], [1, 0]):
        # Test labels never occur in apply's signature.
        predictions = threshold.apply([.2, .9], test_patient_ids=['t1', 't2'])
        assert threshold.value == before
        np.testing.assert_equal(predictions, [False, True])
    with pytest.raises(ValueError):
        threshold.apply([.2], test_patient_ids=['v1'])


def test_labels_require_days_and_handle_censoring():
    for bad in ('months', 'normalized'):
        with pytest.raises(ValueError):
            ProgressionLabelSchema('PACA', 'pfs', 'event', bad, 100.)
    with pytest.raises(ValueError):
        ProgressionLabelSchema('PACA', 'pfs', 'event', 'days', 100., time_transformation='minmax')
    frame = pd.DataFrame({'pfs': [50, 50, 100, np.nan], 'event': [1, 0, 0, 1]})
    for name, builder in [('PACA', build_paca_progression_label), ('MELA', build_mela_progression_label), ('LC', build_lc_progression_label)]:
        schema = ProgressionLabelSchema(name, 'pfs', 'event', 'days', 100.)
        np.testing.assert_equal(builder(frame, schema), [0, -1, 1, -1])


def test_feature_units_and_panel_coverage():
    with pytest.raises(ValueError):
        FEATURE_METADATA['VAF_mean'].validate([1.5], unit='fraction')
    with pytest.raises(ValueError):
        FEATURE_METADATA['nMut'].validate([5], unit='mutations/Mb')
    with pytest.raises(ValueError, match='REQUIRES_DATA_REPROCESSING'):
        panel_corrected_mutation_burden([5], None)
    np.testing.assert_allclose(panel_corrected_mutation_burden([6, 6], [2, 3]), [3, 2])


def test_legacy_type_zero_reaches_latent_likelihood():
    df = tiny_frame(6)
    data = build_topology_training_data(df)
    buckets, ne, nf, ns, _, _ = data
    assert ns == sum(len(b[3]) for b in buckets) == 6
    assert {b[0] for b in buckets} == {4}
    model = ProEHNTopologyModel(ne, nf)
    params = jnp.zeros(model.shapes.total_size)
    assert all(np.isfinite(float(model.bucket_loss(params, *b))) for b in buckets)


@pytest.mark.parametrize('kind', list(ObservationType))
def test_each_observation_type_has_explicit_path_and_finite_likelihood(kind):
    df = tiny_frame(6).iloc[:1].copy()
    df['observation_type'] = int(kind)
    df['seeding_at_first_observation'] = 1
    df['diag_order'] = 0
    buckets, ne, nf, ns, _, _ = build_topology_training_data(df)
    assert kind in OBSERVATION_SEMANTICS
    assert OBSERVATION_SEMANTICS[kind].likelihood_path
    assert buckets[0][0] == int(kind)
    model = ProEHNTopologyModel(ne, nf)
    assert np.isfinite(float(model.bucket_loss(jnp.zeros(model.shapes.total_size), *buckets[0])))


def test_no_silent_sample_dropping_or_unknown_type():
    df = tiny_frame(6)
    df['observation_type'] = -1
    with pytest.raises(ValueError):
        build_topology_training_data(df)
    df['observation_type'] = 3
    df['diag_order'] = 9
    with pytest.raises(ValueError):
        build_topology_training_data(df)
    df['diag_order'] = 0
    df.loc[0, 'P.A (M)'] = np.nan
    with pytest.raises(ValueError):
        build_topology_training_data(df)


def test_diagnosis_is_linear_and_has_positive_baseline():
    state = jnp.array([1, 1, 1])
    dp = jnp.log(jnp.array([2., 3.]))
    p = jnp.arange(8.)+.5
    for fn in (vanilla.diag_scal_p, vanilla.diag_scal_m):
        np.testing.assert_allclose(fn(dp, state, 2*p), 2*fn(dp, state, p))
        np.testing.assert_allclose(fn(dp, state, jnp.zeros(8)), 0.)
    assert float(vanilla.diag_scal_p(dp, state, jnp.ones(8))[0]) == 1.
    assert float(vanilla.diag_scal_m(dp, state, jnp.ones(8))[0]) == 0.
    # Little endian PT=bit0, MT=bit1, seed=bit2.
    np.testing.assert_allclose(vanilla.diag_scal_p(dp, state, jnp.ones(8)), [1, 2, 1, 2, 3, 6, 3, 6])


def test_diagnosis_transform_subtracts_columns():
    theta = jnp.array([[1., 2.], [3., 4.]])
    np.testing.assert_allclose(vanilla.diagnosis_theta(theta, jnp.array([.1, .2])), [[1, 1.8], [2.9, 4]])


@pytest.mark.parametrize('joint', [False, True])
def test_generator_conserves_mass_and_resolvent_matches_dense(joint):
    theta = jnp.array([[-.3, .2], [-.1, -.5]])
    state = jnp.ones(3 if joint else 2, dtype=int)
    size = len(state)
    eye = np.eye(2**size)
    q = np.column_stack([vanilla.kronvec(theta, jnp.asarray(col), state) for col in eye.T])
    np.testing.assert_allclose(q.sum(axis=0), 0, atol=2e-6)
    np.testing.assert_allclose(np.diag(q), vanilla.kron_diag(theta, state, size), atol=2e-6)
    x = jnp.arange(2**size, dtype=float)+1
    for transposed in (False, True):
        expected = np.linalg.solve(np.eye(len(x))-(q.T if transposed else q), x)
        np.testing.assert_allclose(vanilla.R_inv_vec(theta, x, state, size, transpose=transposed), expected, rtol=2e-6)
    if joint:
        # No MT-only/PT-only mutation before seed: one shared preseed event.
        assert q[3, 0] > 0 and q[1, 0] == q[2, 0] == 0
        assert q[4, 0] > 0  # seed
        assert q[5, 4] > 0 and q[6, 4] > 0  # separate events after seed


def test_single_likelihood_normalizes_and_seed_is_marginalized():
    theta = jnp.array([[-.3, .2], [-.1, -.5]])
    dp = jnp.array([.4, -.2])
    probs = []
    for bits in itertools.product([0, 1], repeat=2):
        state = jnp.array(bits)
        probs.append(np.exp(float(likelihood._lp_prim_obs(theta, dp, state, sum(bits)))))
    np.testing.assert_allclose(sum(probs), 1., rtol=2e-6)
    model = ProEHNTopologyModel(1, 0)
    params = jnp.r_[theta.ravel(), dp, jnp.zeros(2)]
    unknown = float(model.bucket_loss(params, 4, 0, 0, jnp.array([[0, 0]]), jnp.ones((1, 1))))
    np.testing.assert_allclose(np.exp(-unknown), sum(probs[:2]), rtol=2e-6)


@pytest.mark.parametrize('pt,mt', [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_paired_unknown_order_sums_both_orders(pt, mt):
    theta = jnp.array([[-.3, .2], [-.1, -.5]])
    dp, dm = jnp.array([.2, .1]), jnp.array([-.1, .3])
    state = jnp.array([pt, mt, 1])
    args = theta, dp, dm, state, pt+1, mt+1
    p0, p1, p2 = [float(np.exp(f(*args))) for f in (likelihood._lp_coupled_0, likelihood._lp_coupled_1, likelihood._lp_coupled_2)]
    assert 0 <= p0 <= 1
    np.testing.assert_allclose(p0, p1+p2, rtol=2e-6)


def test_regularization_penalizes_features_not_intercepts():
    model = ProEHNTopologyModel(1, 1, regularization_strength=1, l1_ratio=1)
    w = jnp.zeros((2, 2, 2)).at[0].set(10)
    dp = dm = jnp.zeros((2, 2)).at[0].set(10)
    pack = lambda a,b,c: jnp.r_[a.ravel(), b.ravel(), c.ravel()]
    assert float(model.regularization(pack(w, dp, dm))) == 0
    for params in [pack(w.at[1, 0, 1].set(2), dp, dm), pack(w, dp.at[1, 0].set(2), dm), pack(w, dp, dm.at[1, 0].set(2))]:
        assert float(model.regularization(params)) == 2.


def test_conditional_distribution_and_integrated_score_mass():
    model = ProEHNTopologyModel(2, 0)
    rep = patient_transition_representation(model, jnp.zeros(model.shapes.total_size), jnp.ones(1), ['A', 'B'], [1, 0], [0, 0], 0)
    np.testing.assert_allclose(rep.conditional_probabilities.sum(), 1.)
    np.testing.assert_allclose(rep.integrated_scores(.37).sum(), .37)
    path = model_implied_transition_path(np.zeros((3, 3)), ['A', 'B'], [1, 0], [0, 0], 0)
    assert path and len(path) <= 3


def test_survival_direction_never_changes_with_outcomes():
    oof = BenchmarkResult('ProEHN', 'A', ('p1', 'p2'), np.array([0, 1]), np.array([.2, .8]))
    threshold = CalibratedThreshold(.5, 'prespecified', ())
    a = survival_score_inputs(oof, [1, 2], [0, 1], km_threshold=threshold)
    b = survival_score_inputs(oof, [2, 1], [1, 0], km_threshold=threshold)
    np.testing.assert_allclose(a['predicted_survival_score'], [-.2, -.8])
    np.testing.assert_equal(a['risk_score'], b['risk_score'])


def test_ablation_does_not_assign_stops_automatic_topology_failures():
    full = AblationInputs(np.array([.2, .8]), np.array([[.3, .7], [.4, .6]]))
    evolution = AblationInputs(None, full.conditional_event_probabilities.copy())
    result = ablation_full_vs_evolution(full, evolution, known_event_mask=[True, True])
    assert result['progression_propensity'][1] is None
    np.testing.assert_equal(*result['conditional_event_ranking'])


def test_default_benchmark_is_formal_and_missing_gate_fails():
    import proehn.paca_proehn_metrics as public
    assert public.FormalProEHNTrainer is FormalProEHNTrainer
    assert not hasattr(public, 'surrogate_baseline')
    from proehn.paca_experiments import PACAExampleExperiments
    with pytest.raises(RuntimeError):
        PACAExampleExperiments()
    with pytest.raises(RuntimeError):
        ProEHNKineticGatekeeper(None, None).predict_go_probability({})


def recovery_fixture():
    df = tiny_frame()
    protocols = []
    for prefix, compartment in [('p', 'Primary'), ('m', 'Metastatic')]:
        df[prefix+'a'], df[prefix+'b'] = np.arange(12)%2, np.ones(12)
        df[prefix+'va'], df[prefix+'vb'] = .4, .2
        protocols.append(TargetMaskingProtocol((prefix+'a', prefix+'b'), (prefix+'va', prefix+'vb'), (), (prefix+'a',), (), ('Age_at_Diagnosis',), compartment))
    df['observed_seeding'] = 1
    df['raw_pfs_days'] = 200.
    df['raw_progression_event'] = 0
    df['observation_type'] = 3
    df['seeding_at_first_observation'] = 1
    df['diag_order'] = 0
    spec = TargetRecoverySpec('A', 'primary', tuple(protocols),
        (('P.A (M)', ('pa',)), ('M.A (M)', ('ma',)), ('P.B (M)', ('pb',)), ('M.B (M)', ('mb',))))
    return df, spec


def test_formal_trainer_calls_formal_components_without_test_labels(monkeypatch):
    # Training functions are spies: no network or topology optimization occurs.
    import proehn.benchmark as benchmark
    from dataclasses import replace
    from proehn.evaluation import FormalFoldProtocol, GroupedStratificationPolicy
    from final_test_support import explicit_configs, event_protocol
    df, spec = recovery_fixture()
    protocol = event_protocol()
    spec = replace(spec,event_selection_protocol=protocol,event_schema_version=protocol.schema_version)
    kinetic_config,topology_config=explicit_configs()
    df['raw_pfs_days']=np.where(df.pa==1,200.,10.)
    df['raw_progression_event']=1
    seen = []
    def gate_fit(train, validation, **kwargs):
        assert not set(train.patient_id) & set(validation.patient_id)
        assert (train['P.A (M)'] == 0).all()
        seen.append(('kinetic', set(train.patient_id), set(validation.patient_id)))
        gate = ProEHNKineticGatekeeper(None, None)
        gate.ready = True
        # Formal network forward pass with synthetic initialization; NO optimization.
        from proehn.kinetic import KineticGatekeeperNetwork
        gate.preprocessor = KineticPreprocessor().fit(train)
        gate.model = KineticGatekeeperNetwork(d_model=2, n_head_layers=0, dropout_rate=0.)
        groups = gate.preprocessor.feature_groups
        inputs = [jnp.zeros((1, len(groups[k]))) for k in ('pt_genomic', 'pt_dynamic', 'mt_genomic', 'mt_dynamic', 'shared')]
        gate.params = gate.model.init(jax.random.PRNGKey(0), *inputs, train=False)['params']
        return gate
    def topology_fit(buckets, ne, nf, ns, **kwargs):
        seen.append(('topology', ns))
        model = ProEHNTopologyModel(ne, nf)
        return model, jnp.zeros(model.shapes.total_size), 0.
    monkeypatch.setattr(benchmark, 'fit_kinetic_model', gate_fit)
    monkeypatch.setattr(benchmark, 'fit_topology_model', topology_fit)
    fold_protocol=FormalFoldProtocol.create(df.patient_id,df.pa,df.pa,target=spec.target_column(),policy=GroupedStratificationPolicy(outer_splits=3,inner_splits=2))
    folds=fold_protocol.folds
    results = run_oof_benchmark(df, df.patient_id, folds, {'ProEHN': lambda: FormalProEHNTrainer(spec, ProgressionLabelSchema('PACA', 'raw_pfs_days', 'raw_progression_event', 'days', 100., schema_id='synthetic', version='1', frozen=True), kinetic_config=kinetic_config,topology_config=topology_config,covariate_protocol=__import__('proehn.features', fromlist=['FrozenCovariateProtocol']).FrozenCovariateProtocol('PACA', 'synthetic', (), (), True))},
                               target=spec.target_column(), test_feature_builder=spec.features,fold_protocol=fold_protocol)
    assert results[0].scores.shape == (12, 3)
    assert [s[0] for s in seen].count('kinetic') == 3
    assert [s[0] for s in seen].count('topology') == 3
    np.testing.assert_allclose(results[0].scores[:, 2], results[0].scores[:, 0]*results[0].scores[:, 1])


def test_baselines_share_folds_and_test_masking():
    df = tiny_frame()
    seen = []
    class FakeAdapter:
        def fit(self, train, validation):
            seen.append((tuple(train.index), tuple(validation.index)))
            return self
        def predict(self, frame):
            assert list(frame) == ['Age_at_Diagnosis']
            return np.zeros((len(frame), 1))
    folds = patient_outer_folds(df.patient_id, n_splits=3)
    results = run_oof_benchmark(df, df.patient_id, folds,
        {name: FakeAdapter for name in ('Oncotree', 'HyperTraPS', 'MHN', 'evolution_only', 'ProEHN')},
        target='synthetic', test_feature_builder=lambda f: f[['Age_at_Diagnosis']])
    assert all(seen[i:i+3] == seen[:3] for i in range(0, len(seen), 3))
    assert all(np.array_equal(r.fold_ids, results[0].fold_ids) for r in results)


def test_kinetic_requires_split_before_preprocessing():
    df = tiny_frame()
    with pytest.raises(ValueError):
        build_kinetic_matrices(df)
    prep = KineticPreprocessor().fit(df.iloc[:6])
    before = prep.transforms['shared'].metadata()
    modified = df.iloc[6:].copy()
    modified.Age_at_Diagnosis = 10000
    build_kinetic_matrices(modified, preprocessor=prep)
    assert prep.transforms['shared'].metadata() == before


def test_syntax_only_all_project_python():
    root = Path(__file__).resolve().parents[1]
    for folder in ('proehn', 'proehn_service', 'scripts', 'tests'):
        for path in (root/folder).rglob('*.py'):
            ast.parse(path.read_text(), filename=str(path))


def test_joint_two_gene_generator_against_independent_enumeration():
    theta = np.array([[-.2, .3, .1], [-.4, -.5, .2], [.1, -.2, -.3]])
    n = 2
    q = np.zeros((32, 32))
    for source in range(32):
        x = [(source >> bit) & 1 for bit in range(5)]
        pt, mt, seed = x[0:4:2], x[1:4:2], x[-1]
        transitions = []
        for j in range(n):
            prate = np.exp(theta[j,j]+sum(theta[j,k]*pt[k] for k in range(n) if k != j))
            mrate = np.exp(theta[j,j]+sum(theta[j,k]*mt[k] for k in range(n) if k != j)+theta[j,-1])
            if not seed and not pt[j] and not mt[j]:
                transitions.append((source+(1 << (2*j))+(1 << (2*j+1)), prate))
            if seed and not pt[j]:
                transitions.append((source+(1 << (2*j)), prate))
            if seed and not mt[j]:
                transitions.append((source+(1 << (2*j+1)), mrate))
        if not seed:
            transitions.append((source+16, np.exp(theta[-1,-1]+sum(theta[-1,k]*pt[k] for k in range(n)))))
        for dest, rate in transitions:
            q[dest, source] += rate
            q[source, source] -= rate
    state = jnp.ones(5, dtype=int)
    actual = np.column_stack([vanilla.kronvec(jnp.asarray(theta), jnp.asarray(col), state) for col in np.eye(32).T])
    np.testing.assert_allclose(actual, q, rtol=2e-6, atol=1e-7)
    # Restricted generator must retain outward loss, not renormalize transitions.
    restricted = jnp.array([1, 0, 1, 1, 1])
    inds = [i for i in range(32) if not (i & 2)]
    actual_restricted = np.column_stack([vanilla.kronvec(jnp.asarray(theta), jnp.asarray(col), restricted) for col in np.eye(16).T])
    np.testing.assert_allclose(actual_restricted, q[np.ix_(inds, inds)], rtol=2e-6, atol=1e-7)


def test_coupled_resolvent_diagnosis_and_gradient_consistency():
    theta = jnp.array([[-.2, .3], [.1, -.4]])
    state = jnp.ones(3, int)
    dp, dm = jnp.array([.3, -.2]), jnp.array([-.2, .4])
    p0 = jnp.zeros(8).at[0].set(1.)
    q = np.column_stack([vanilla.kronvec(theta, jnp.asarray(col), state) for col in np.eye(8).T])
    rates = vanilla.diag_scal_p(dp, state, jnp.ones(8))+vanilla.diag_scal_m(dm, state, jnp.ones(8))
    expected = np.linalg.solve(np.diag(rates)-q, p0)
    actual = likelihood.R_i_inv_vec(theta, dp, dm, p0, state, 3)
    np.testing.assert_allclose(actual, expected, rtol=2e-6)
    fn = lambda value: likelihood._lp_coupled_1(theta.at[0,1].set(value), dp, dm, state, 2, 2)
    h = 1e-4
    analytic = float(jax.grad(fn)(theta[0,1]))
    numeric = float((fn(theta[0,1]+h)-fn(theta[0,1]-h))/(2*h))
    assert np.isfinite(analytic)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-5)


def test_test_target_and_outcomes_cannot_change_recovery_inputs():
    raw, spec = recovery_fixture()
    before = spec.features(raw)
    raw['pa'], raw['ma'] = 1-raw.pa, 1-raw.ma
    raw['Patient_Label'] = 1-raw.Patient_Label
    raw['response_is_Missing'] = np.arange(len(raw))%2
    pd.testing.assert_frame_equal(before, spec.features(raw))


def test_projection_wrapper_never_refits_on_test():
    from proehn.features import FoldFittedTransformer
    class DummyProjection:
        def fit(self, x):
            self.training_x = x.copy()
        def transform(self, x):
            return x
    projection = DummyProjection()
    frame = tiny_frame()
    wrapper = FoldFittedTransformer(FoldPreprocessor(['Age_at_Diagnosis']), projection).fit(frame.iloc[:6])
    original = projection.training_x.copy()
    wrapper.transform(frame.iloc[6:])
    np.testing.assert_array_equal(original, projection.training_x)


def test_regularization_selection_only_has_inner_partitions():
    from proehn.evaluation import select_on_inner_validation
    frame = tiny_frame()
    train, val = frame.iloc[:6], frame.iloc[6:]
    calls = []
    def callback(t, v, candidate):
        assert not set(t.patient_id) & set(v.patient_id)
        calls.append(candidate)
        return float(candidate)
    assert select_on_inner_validation(train, val, [2., 1.], callback, patient_id_column='patient_id') == 1.
    assert calls == [2., 1.]


def test_undocumented_legacy_summary_units_block_new_fitting():
    from proehn.features import validate_genomic_summary_units
    frame = tiny_frame()
    frame['nMut_Primary'] = .5
    with pytest.raises(ValueError, match='REQUIRES_DATA_REPROCESSING'):
        TopologyTrainingPreprocessor().fit(frame)
    frame.attrs['feature_metadata'] = {'nMut_Primary': FEATURE_METADATA['nMut']}
    with pytest.raises(ValueError, match='integral'):
        validate_genomic_summary_units(frame)


def test_pt_first_paired_sample_does_not_require_seeding_at_first_observation():
    frame = tiny_frame(6).iloc[:1].copy()
    frame['observation_type'] = 3
    frame['diag_order'] = 1
    assert build_topology_training_data(frame)[3] == 1


def test_foldwise_survival_thresholds_preserve_test_isolation():
    result = BenchmarkResult('ProEHN', 'A', ('a','b','c','d'), np.array([0,0,1,1]), np.array([.1,.2,.7,.8]))
    thresholds = {0: CalibratedThreshold(.5, 'validation', ('c',)), 1: CalibratedThreshold(.4, 'validation', ('a',))}
    values = survival_score_inputs(result, [1,2,3,4], [0,1,0,1], km_threshold=thresholds)
    np.testing.assert_array_equal(values['high_risk'], [False, False, True, True])
