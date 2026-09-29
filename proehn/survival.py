"""OOF survival framework. No analysis runs on import; larger score means risk.

Cox uses lifelines' Breslow baseline / Efron ties. External dependency is explicit.
Patient-level paired permutation tests exchangeability of the two model scores
conditional on survival outcomes, not the weaker null that delta C merely is zero.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import chi2
from .evaluation import BenchmarkResult, survival_score_inputs


@dataclass(frozen=True)
class OOFSurvivalScores:
    predictions: BenchmarkResult
    time_days: np.ndarray
    event: np.ndarray
    score_column: int = 0

    def validate(self):
        p=self.predictions.validate()
        ids=tuple(p.patient_ids)
        t=np.asarray(self.time_days,float); e=np.asarray(self.event)
        if len(set(ids)) != len(ids):
            raise ValueError('Survival requires one OOF prediction per independent patient')
        if t.shape!=(len(ids),) or e.shape!=t.shape or not np.isfinite(t).all() or np.any(t<0) or not np.isin(e,[0,1]).all():
            raise ValueError('Aligned raw days/event required')
        return self

    @property
    def risk(self):
        self.validate()
        s=self.predictions.scores
        return np.asarray(s if s.ndim==1 else s[:,self.score_column],float)

    def km_groups(self, thresholds):
        self.validate()
        return survival_score_inputs(self.predictions,self.time_days,self.event,
            score_column=self.score_column,km_threshold=thresholds)['high_risk']


def kaplan_meier(oof, thresholds):
    groups=oof.km_groups(thresholds)
    if len(np.unique(groups))!=2:
        raise ValueError('Both prespecified KM groups must be populated')
    t=np.asarray(oof.time_days); e=np.asarray(oof.event)
    curves={}
    for group in (False,True):
        times=np.unique(t[group==groups]); survival=1.; points=[]
        for time in times:
            at_risk=np.sum((groups==group)&(t>=time))
            deaths=np.sum((groups==group)&(t==time)&(e==1))
            survival*=1-deaths/at_risk
            points.append((float(time),float(survival),int(at_risk),int(deaths)))
        curves[group]=points
    return curves


def logrank(oof, thresholds):
    groups=oof.km_groups(thresholds); t=np.asarray(oof.time_days); e=np.asarray(oof.event)
    if len(np.unique(groups))!=2:
        raise ValueError('Both KM groups required')
    observed_minus_expected=0.; variance=0.
    for time in np.unique(t[e==1]):
        risk=t>=time; n=risk.sum(); n1=(risk&groups).sum()
        dead=(t==time)&(e==1); d=dead.sum(); d1=(dead&groups).sum()
        observed_minus_expected+=d1-d*n1/n
        if n>1:
            variance+=n1*(n-n1)*d*(n-d)/(n*n*(n-1))
    if variance<=0:
        raise ValueError('No informative log-rank variance')
    statistic=observed_minus_expected**2/variance
    return dict(statistic=float(statistic),p_value=float(chi2.sf(statistic,1)))


def harrell_c_index(oof):
    """Fixed risk direction; tied predictions contribute one half.

    Event/censor pairs at equal times are comparable (event before censor).
    Two events tied in time are not ordered. No data-driven sign maximization.
    """
    risk=oof.risk; t=np.asarray(oof.time_days); e=np.asarray(oof.event)
    concordant=0.; comparable=0
    for i in range(len(t)):
        for j in range(i+1,len(t)):
            earlier,later=(i,j) if t[i]<t[j] or (t[i]==t[j] and e[i] and not e[j]) else (j,i)
            if not e[earlier] or (t[i]==t[j] and e[i]==e[j]):
                continue
            comparable+=1
            concordant+=float(risk[earlier]>risk[later])+.5*float(risk[earlier]==risk[later])
    if not comparable:
        raise ValueError('No comparable survival pairs')
    return concordant/comparable


def cox_proportional_hazards(oof, *, baseline_confounders=None):
    oof.validate()
    try:
        from lifelines import CoxPHFitter
    except ImportError as exc:
        raise RuntimeError('EXTERNAL_DEPENDENCY_REQUIRED_BEFORE_RERUN: lifelines CoxPHFitter') from exc
    ids=list(oof.predictions.patient_ids)
    frame=pd.DataFrame({'risk_score':oof.risk,'time_days':oof.time_days,'event':oof.event},index=ids)
    if baseline_confounders is not None:
        if not isinstance(baseline_confounders,pd.DataFrame) or list(baseline_confounders.index)!=ids or not baseline_confounders.shape[1]:
            raise ValueError('Multivariable Cox requires a patient-aligned baseline confounder matrix')
        if set(frame)&set(baseline_confounders) or not baseline_confounders.columns.is_unique or not np.isfinite(baseline_confounders.to_numpy(float)).all():
            raise ValueError('Invalid confounder columns/values')
        frame=frame.join(baseline_confounders)
    fitted=CoxPHFitter().fit(frame,duration_col='time_days',event_col='event')
    return dict(model=fitted,analysis='multivariable' if baseline_confounders is not None else 'univariable',
        confounders=[] if baseline_confounders is None else list(baseline_confounders))


def paired_delta_c_permutation(full, topology, *, repetitions, random_seed):
    """Two-sided within-patient model-label swaps; OOF patients/folds must match."""
    from dataclasses import replace
    full.validate(); topology.validate()
    a,b=full.predictions,topology.predictions
    if tuple(a.patient_ids)!=tuple(b.patient_ids) or not np.array_equal(a.fold_ids,b.fold_ids) or not np.array_equal(full.time_days,topology.time_days) or not np.array_equal(full.event,topology.event):
        raise ValueError('Paired comparison requires identical patient/fold/outcome alignment')
    if repetitions<1:
        raise ValueError('Prespecify positive permutation count')
    observed=harrell_c_index(full)-harrell_c_index(topology)
    rng=np.random.default_rng(random_seed); exceeded=0
    for _ in range(repetitions):
        swap=rng.random(len(a.patient_ids))<.5
        x=replace(full,predictions=replace(a,scores=np.where(swap,topology.risk,full.risk)),score_column=0)
        y=replace(topology,predictions=replace(b,scores=np.where(swap,full.risk,topology.risk)),score_column=0)
        exceeded+=abs(harrell_c_index(x)-harrell_c_index(y))>=abs(observed)
    return dict(delta_c=observed,p_value=(exceeded+1)/(repetitions+1),
        null='within-patient exchangeability of model predictions conditional on outcomes',repetitions=repetitions)
