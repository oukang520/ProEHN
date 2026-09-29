"""Sequential importance sampling of pure-birth observation histories.

The proposal removes jumps incompatible with still-pending observations, but
keeps their hazards in the target denominator. Every jump multiplies weight by
(sum allowed hazards)/(sum all competing hazards). Waiting times are integrated
out using the CTMC embedded jump chain. This is NOT plug-in seeding, MCMC, or an
exact posterior. Self-normalized estimates are consistent, finite-budget biased.
Memory is O(particles * genes); no joint-state table or generator is materialized.
This numerical core uses only the standard library to permit isolated oracle tests.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class ParticlePosterior:
    states: tuple
    weights: tuple
    effective_sample_size: float
    log_evidence: float
    particles: int
    random_seed: int


def _logsumexp(values):
    if not values:
        return -math.inf
    maximum=max(values)
    return maximum+math.log(math.fsum(math.exp(x-maximum) for x in values))


def sample_observation_posterior(theta, dp, dm, primary, metastasis, kind, *,
        diagnosis_order=None, first_seeding=-1, particles=8192, random_seed=42):
    """Importance sample the SAME observation process as exact resolvents.

    PT-only: PT clock. MT-only: competing (failure) PT clock before seeding,
    then MT clock. Paired: both clocks until first diagnosis, remaining clock
    until second. The earlier-sampled compartment is free to evolve afterwards.
    Incompatible diagnosis/order events are killing events, not removed hazards.
    """
    n=len(theta)-1
    if kind not in (0,1,2,3,4) or particles<2 or int(particles)!=particles:
        raise ValueError('Valid observation type and at least two particles required')
    if len(theta)!=n+1 or any(len(row)!=n+1 for row in theta) or len(dp)!=n+1 or len(dm)!=n+1:
        raise ValueError('Aligned finite CTMC parameter dimensions required')
    if not all(math.isfinite(float(x)) for row in theta for x in row) or not all(math.isfinite(float(x)) for x in (*dp,*dm)):
        raise ValueError('Nonfinite CTMC parameters')
    for genotype,needed in ((primary,kind!=2),(metastasis,kind in (2,3))):
        if needed and (genotype is None or len(genotype)!=n or any(x not in (0,1) for x in genotype)):
            raise ValueError('Binary observed compartment genotype required')
        if not needed and genotype is not None:
            raise ValueError('Genotype conflicts with observation type')
    if first_seeding not in (-1,0,1) or (kind==3 and diagnosis_order not in (0,1,2)):
        raise ValueError('Explicit paired order and valid first-seeding annotation required')
    if kind==3 and diagnosis_order==2 and first_seeding==0:
        raise ValueError('MT-first requires seeded first observation')
    rng=random.Random(random_seed); states=[]; logweights=[]
    for _ in range(particles):
        pt=[0]*n; mt=[0]*n; seed=0; p_done=False; m_done=False
        lp=[float(theta[j][j]) for j in range(n)]
        lm=[float(theta[j][j]+theta[j][n]) for j in range(n)]
        ls=float(theta[n][n]); ldp=0.; ldm=0.; weight=0.; complete=False
        for _step in range(2*n+3):
            pending_p=kind in (0,1,4) or (kind==3 and not p_done)
            pending_m=kind==2 or (kind==3 and not m_done)
            first=not p_done and not m_done
            all_rates=[]; allowed=[]
            def add(lograte, event, compatible):
                all_rates.append(lograte)
                if compatible:
                    allowed.append((event,lograte))
            for j in range(n):
                if not pt[j]:
                    compatible=(not pending_p or primary[j]==1)
                    if not seed:
                        compatible=compatible and (not pending_m or metastasis[j]==1)
                    add(lp[j],('p',j),compatible)
                if seed and not mt[j]:
                    add(lm[j],('m',j),not pending_m or metastasis[j]==1)
            if not seed:
                add(ls,('seed',0),kind!=0 and not (kind==3 and first and first_seeding==0))
            if kind in (0,1,4) or (kind==2 and not seed) or (kind==3 and not p_done):
                match=kind!=2 and pt==list(primary)
                if kind in (0,1): match=match and seed==kind
                if kind==3 and first:
                    match=match and diagnosis_order!=2 and (first_seeding==-1 or seed==first_seeding)
                add(ldp,('dp',0),match)
            if seed and (kind==2 or (kind==3 and not m_done)):
                match=mt==list(metastasis)
                if kind==3 and first:
                    match=match and diagnosis_order!=1 and (first_seeding==-1 or seed==first_seeding)
                add(ldm,('dm',0),match)
            if not allowed:
                weight=-math.inf
                break
            denominator=_logsumexp(all_rates); proposal=_logsumexp([x[1] for x in allowed])
            weight+=proposal-denominator
            draw=rng.random(); cumulative=0.; event=allowed[-1][0]
            for candidate,lograte in allowed:
                cumulative+=math.exp(lograte-proposal)
                if draw<cumulative:
                    event=candidate
                    break
            action,j=event
            if action=='p':
                pt[j]=1; ldp+=dp[j]; ls+=theta[n][j]
                for k in range(n): lp[k]+=theta[k][j]
                if not seed:
                    mt[j]=1; ldm+=dm[j]
                    for k in range(n): lm[k]+=theta[k][j]
            elif action=='m':
                mt[j]=1; ldm+=dm[j]
                for k in range(n): lm[k]+=theta[k][j]
            elif action=='seed':
                seed=1; ldp+=dp[n]; ldm+=dm[n]
            elif action=='dp': p_done=True
            else: m_done=True
            complete=(kind in (0,1,4) and p_done) or (kind==2 and m_done) or (kind==3 and p_done and m_done)
            if complete: break
        if not complete: weight=-math.inf
        states.append(tuple(x for pair in zip(pt,mt) for x in pair)+(seed,))
        logweights.append(weight)
    finite=[x for x in logweights if math.isfinite(x)]
    if not finite:
        raise ValueError('Observation has no supported sampled histories; no posterior returned')
    normalizer=_logsumexp(finite)
    weights=tuple(math.exp(x-normalizer) if math.isfinite(x) else 0. for x in logweights)
    ess=1/math.fsum(w*w for w in weights)
    return ParticlePosterior(tuple(states),weights,ess,normalizer-math.log(particles),particles,random_seed)


def transition_probabilities(theta, state):
    """Same normalized accessible CTMC hazards, including explicit terminal atom."""
    n=len(theta)-1; pt=state[:2*n:2]; mt=state[1:2*n:2]; seed=state[-1]
    rates=[]
    for j in range(n):
        if not pt[j]: rates.append((('primary',j),theta[j][j]+sum(theta[j][k] for k in range(n) if pt[k])))
        if seed and not mt[j]: rates.append((('metastasis',j),theta[j][j]+theta[j][n]+sum(theta[j][k] for k in range(n) if mt[k])))
    if not seed: rates.append((('primary','Metastatic seeding'),theta[n][n]+sum(theta[n][k] for k in range(n) if pt[k])))
    if not rates: return {('terminal','No accessible event'):1.}
    norm=_logsumexp([x[1] for x in rates])
    return {key:math.exp(rate-norm) for key,rate in rates}


def posterior_transition_moments(theta, posterior):
    """SNIS means and asymptotic Monte Carlo SEs; not exact error bounds."""
    cache={}; means={}
    for state,w in zip(posterior.states,posterior.weights):
        if not w: continue
        if state not in cache: cache[state]=transition_probabilities(theta,state)
        for key,value in cache[state].items(): means[key]=means.get(key,0.)+w*value
    errors={key:math.sqrt(math.fsum(w*w*(cache.get(state,{}).get(key,0.)-mean)**2
        for state,w in zip(posterior.states,posterior.weights))) for key,mean in means.items()}
    return means,errors
