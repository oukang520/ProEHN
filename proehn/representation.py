"""Patient specific transition representation; projections are separate summaries."""
from dataclasses import dataclass
import numpy as np
from scipy.special import softmax


@dataclass(frozen=True)
class PatientTransitionRepresentation:
    theta: np.ndarray
    log_diagnosis_primary: np.ndarray
    log_diagnosis_metastasis: np.ndarray
    accessible_events: tuple[tuple[str, str], ...]
    conditional_probabilities: np.ndarray

    def integrated_scores(self, progression_propensity):
        if not np.isfinite(progression_propensity) or not 0 <= progression_propensity <= 1:
            raise ValueError('Progression propensity must lie in [0,1]')
        return self.conditional_probabilities * progression_propensity


def accessible_transition_log_rates(theta, genes, primary, metastasis, seeded):
    """Same compartment transition definition as the metMHN CTMC generator.

    Prior to seeding the MT copy is latent/inherited, not an independent event.
    Seeding copies the primary genotype. After seeding the two compartments
    have separate accessible mutation events; only MT hazards use seed effects.
    """
    pt, mt = np.asarray(primary), np.asarray(metastasis)
    n = len(genes)
    theta = np.asarray(theta, float)
    if theta.shape != (n+1, n+1) or not np.isfinite(theta).all():
        raise ValueError('Invalid transition parameters')
    if pt.shape != (n,) or mt.shape != (n,) or not np.isin(pt, [0, 1]).all() or not np.isin(mt, [0, 1]).all() or seeded not in (0, 1):
        raise ValueError('Known binary observation-time genotype and seeding required')
    if not seeded and mt.any():
        raise ValueError('Observed MT conflicts with absence of seeding')
    rows = []
    active_p = np.flatnonzero(pt)
    for j, gene in enumerate(genes):
        if not pt[j]:
            rows.append(dict(compartment='primary', event=gene, log_rate=float(theta[j, j]+theta[j, active_p].sum())))
    if not seeded:
        rows.append(dict(compartment='primary', event='Metastatic seeding', log_rate=float(theta[n, n]+theta[n, active_p].sum())))
    else:
        active_m = np.r_[np.flatnonzero(mt), n]
        for j, gene in enumerate(genes):
            if not mt[j]:
                rows.append(dict(compartment='metastasis', event=gene, log_rate=float(theta[j, j]+theta[j, active_m].sum())))
    return rows


def patient_transition_representation(model, params, features_with_bias, gene_names, primary, metastasis, seeded):
    theta, dp, dm = model.compute_patient_params(*model.parse_params(params), features_with_bias,
                                               model.log_rate_clip_min, model.log_rate_clip_max)
    rows = accessible_transition_log_rates(theta, gene_names, primary, metastasis, seeded)
    probabilities = softmax([r['log_rate'] for r in rows]) if rows else np.empty(0)
    return PatientTransitionRepresentation(np.asarray(theta), np.asarray(dp), np.asarray(dm),
        tuple((r['compartment'], r['event']) for r in rows), probabilities)


def model_implied_transition_path(theta, genes, primary, metastasis, seeded, *, max_steps=3):
    """Greedy model-implied path at fixed covariates; no time or future-risk claim."""
    pt, mt, seed = list(primary), list(metastasis), int(seeded)
    path = []
    for _ in range(max_steps):
        rows = accessible_transition_log_rates(theta, genes, pt, mt, seed)
        if not rows:
            break
        probabilities = softmax([r['log_rate'] for r in rows])
        index = int(np.argmax(probabilities))
        row = rows[index]
        path.append({**row, 'conditional_probability': float(probabilities[index])})
        if row['event'] == 'Metastatic seeding':
            seed, mt = 1, pt.copy()
        else:
            (pt if row['compartment'] == 'primary' else mt)[genes.index(row['event'])] = 1
    return path


@dataclass(frozen=True)
class ObservedTransitionRepresentation(PatientTransitionRepresentation):
    observation_type: int
    representation_kind: str
    posterior_states: np.ndarray
    posterior_weights: np.ndarray
    inference_method: str = 'exact_ctmc_resolvent'
    terminal_probability: float = 0.


class ExactInferenceLimitError(ValueError):
    pass


def patient_transition_representation_observed(model, params, features_with_bias, gene_names,
        observed_primary, observed_metastasis, observation_type, *, diagnosis_order=None,
        first_seeding=-1, joint_snapshot=False, max_state_bits=16):
    """Exact posterior marginalization under the fitted observation process.

    PT-only: PT diagnosis clock, irrespective of unobserved MT diagnosis, matching
    _lp_prim_obs. MT-only: PT diagnosis competes only before seeding, then MT
    diagnosis, matching _lp_met_obs. Paired HISTORY: both clocks until first
    diagnosis; only the remaining clock until second diagnosis. The earlier
    sampled compartment is latent at the later observation (it can evolve).
    joint_snapshot=True explicitly denotes a contemporaneous fully known state.

    All latent genotypes are enumerated; no seeding or MT-genotype plug-in is
    used. Beyond the explicit exponential resource bound this function refuses
    exact inference rather than silently truncating the posterior. A terminal
    atom ('terminal', 'No accessible event') preserves probability mass when all
    modeled events are acquired; it is not the operational Stop class.
    """
    import jax.numpy as jnp
    from .ctmc import vanilla as v
    from .observations import ObservationType
    kind = ObservationType(observation_type)
    n = len(gene_names)
    def checked(value, required):
        if value is None:
            if required:
                raise ValueError('Observed compartment genotype is required')
            return None
        a=np.asarray(value)
        if a.shape != (n,) or not np.isin(a,[0,1]).all():
            raise ValueError('Observed genotype must contain n binary calls')
        return a.astype(int)
    pt=checked(observed_primary,kind != ObservationType.METASTASIS_ONLY)
    mt=checked(observed_metastasis,kind in (ObservationType.METASTASIS_ONLY,ObservationType.PAIRED))
    theta,dp,dm=model.compute_patient_params(*model.parse_params(params),features_with_bias,
                                           model.log_rate_clip_min,model.log_rate_clip_max)
    if kind == ObservationType.PAIRED and joint_snapshot:
        states=np.r_[np.column_stack((pt,mt)).ravel(),1][None,:]
        weights=np.ones(1)
    else:
        if 2*n+1 > max_state_bits:
            raise ExactInferenceLimitError(f'Exact posterior needs 2^{2*n+1} joint states; bound={max_state_bits}. No approximation applied.')
        if kind == ObservationType.PAIRED and diagnosis_order not in (0,1,2):
            raise ValueError('Paired history needs diagnosis order; use explicit joint_snapshot for simultaneous states')
        if first_seeding not in (-1,0,1) or (diagnosis_order == 2 and first_seeding == 0):
            raise ValueError('MT-first history cannot have unseeded first observation')
        state=jnp.ones(2*n+1,dtype=int); size=2*n+1
        x,_=v._states(state,size); x=np.asarray(x)
        seed=x[:,-1]; xp=x[:,0:2*n:2]; xm=x[:,1:2*n:2]
        p0=jnp.zeros(len(x)).at[0].set(1.)
        d_p=v.diag_scal_p(dp,state,jnp.ones(len(x)))
        d_m=v.diag_scal_m(dm,state,jnp.ones(len(x)))
        def resolve(start, rates):
            return v.R_inv_vec(theta,start,state,size,rates)
        if kind == ObservationType.PAIRED:
            first=resolve(p0,d_p+d_m)
            annotation=np.ones(len(x),bool) if first_seeding == -1 else seed==first_seeding
            pt_first=resolve(first*d_p*jnp.asarray(np.all(xp==pt,axis=1)&annotation),d_m)*d_m
            mt_first=resolve(first*d_m*jnp.asarray(np.all(xm==mt,axis=1)&annotation),d_p)*d_p
            a=np.asarray(pt_first)*np.all(xm==mt,axis=1)
            b=np.asarray(mt_first)*np.all(xp==pt,axis=1)
            mass=(a+b if diagnosis_order == 0 else a if diagnosis_order == 1 else b)*seed
        elif kind == ObservationType.METASTASIS_ONLY:
            occupation=resolve(p0,d_p*jnp.asarray(1-seed)+d_m)
            mass=np.asarray(occupation*d_m)*np.all(xm==mt,axis=1)*seed
        else:
            mass=np.asarray(resolve(p0,d_p)*d_p)*np.all(xp==pt,axis=1)
            if kind in (ObservationType.PRIMARY_NO_DISSEMINATION,ObservationType.PRIMARY_DISSEMINATED):
                mass=mass*(seed==int(kind==ObservationType.PRIMARY_DISSEMINATED))
        if not np.isfinite(mass).all() or mass.sum() <= 0 or np.any(mass < -1e-10):
            raise ValueError('Observation has zero or invalid likelihood under the model')
        selected=mass>0
        states=x[selected]; weights=mass[selected]/mass[selected].sum()
    distribution={}; terminal=0.
    for state,weight in zip(states,weights):
        seeded=int(state[-1]); primary=state[:2*n:2]; metastatic=state[1:2*n:2]
        # Before seed the duplicated MT bits are an inherited lineage, not an MT observation.
        rates=accessible_transition_log_rates(theta,gene_names,primary,metastatic if seeded else np.zeros(n,int),seeded)
        if not rates:
            terminal+=weight
            distribution[('terminal','No accessible event')]=distribution.get(('terminal','No accessible event'),0.)+weight
        else:
            for row,p in zip(rates,softmax([r['log_rate'] for r in rates])):
                key=(row['compartment'],row['event'])
                distribution[key]=distribution.get(key,0.)+weight*p
    events=tuple(sorted(distribution))
    return ObservedTransitionRepresentation(np.asarray(theta),np.asarray(dp),np.asarray(dm),events,
        np.array([distribution[e] for e in events]),int(kind),
        'fully_observed' if joint_snapshot else 'latent_marginalized',states,weights,
        terminal_probability=float(terminal))
