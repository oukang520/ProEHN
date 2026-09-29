"""Uniform operational progression probability protocol and inner-val calibration."""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize


@dataclass(frozen=True)
class KineticProbabilityProtocol:
    focal_gamma: float = 0.
    calibration: str = 'none'
    version: str = 'kinetic-probability-v1'

    def __post_init__(self):
        if not np.isfinite(self.focal_gamma) or self.focal_gamma<0 or self.calibration not in ('none','platt'):
            raise ValueError('Invalid kinetic probability protocol')
        if self.focal_gamma>0 and self.calibration=='none':
            raise ValueError('Uncalibrated focal-loss output is not a formal progression probability')


@dataclass(frozen=True)
class PlattCalibration:
    slope: float
    intercept: float
    validation_patient_ids: tuple[str,...]

    def __post_init__(self):
        if not np.isfinite([self.slope, self.intercept]).all() or self.slope < 0 or not self.validation_patient_ids:
            raise ValueError('Finite monotone calibration and validation provenance required')

    def transform(self, logits):
        x=np.asarray(logits,float)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite logits')
        return expit(self.slope*x+self.intercept)

    @classmethod
    def fit(cls, validation_logits, validation_labels, validation_patient_ids, *, training_patient_ids):
        x=np.asarray(validation_logits,float); y=np.asarray(validation_labels,float)
        ids=tuple(map(str,validation_patient_ids))
        if x.ndim!=1 or x.shape!=y.shape or len(ids)!=len(x) or not np.isfinite(x).all() or set(y)!={0.,1.}:
            raise ValueError('Calibration needs aligned finite validation logits and both classes')
        if set(ids)&set(map(str,training_patient_ids)):
            raise ValueError('Calibration must be outside network-training patients')
        def objective(ab):
            logits=ab[0]*x+ab[1]
            residual=expit(logits)-y
            return float(np.mean(np.logaddexp(0,logits)-y*logits)),np.array([np.mean(residual*x),np.mean(residual)])
        # Nonnegative slope preserves the predeclared operational score direction.
        fit=minimize(objective,[1.,0.],jac=True,bounds=[(0.,None),(None,None)],method='L-BFGS-B')
        if not fit.success or not np.isfinite(fit.x).all():
            raise ValueError('Probability calibration failed')
        return cls(float(fit.x[0]),float(fit.x[1]),ids)


def evaluate_probability_calibration(oof, labels, *, score_column=0, bins=10):
    """OOF-only diagnostics; does not fit a calibration layer used by predictions.

    Logistic calibration slope/intercept are unconstrained diagnostics (negative
    slope is reported, never used to flip a model). Equal-width reliability bins
    and ECE are prespecified. Separation/nonconvergence is reported unavailable.
    """
    from .evaluation import BenchmarkResult, _binary
    if not isinstance(oof,BenchmarkResult):
        raise ValueError('OOF predictions required')
    oof.validate()
    if len(set(oof.patient_ids))!=len(oof.patient_ids) or bins<2 or int(bins)!=bins:
        raise ValueError('Independent patients and fixed positive bin count required')
    y,p=_binary(labels,oof.scores if oof.scores.ndim==1 else oof.scores[:,score_column])
    if len(np.unique(y))!=2: raise ValueError('Both outcome classes required for calibration diagnostics')
    index=np.minimum((p*bins).astype(int),bins-1); reliability=[]; ece=0.
    for b in range(bins):
        selected=index==b; count=int(selected.sum())
        mean=float(p[selected].mean()) if count else None
        observed=float(y[selected].mean()) if count else None
        reliability.append(dict(lower=b/bins,upper=(b+1)/bins,count=count,predicted=mean,observed=observed))
        if count: ece+=count/len(p)*abs(mean-observed)
    clipped=np.clip(p,np.finfo(float).eps,1-np.finfo(float).eps)
    logits=np.log(clipped)-np.log1p(-clipped)
    def objective(ab):
        linear=ab[0]*logits+ab[1]; error=expit(linear)-y
        return float(np.mean(np.logaddexp(0,linear)-y*linear)),np.array([np.mean(error*logits),np.mean(error)])
    fit=minimize(objective,[1.,0.],jac=True,method='BFGS')
    separated=max(logits[y==0])<=min(logits[y==1]) or max(logits[y==1])<=min(logits[y==0])
    estimable=bool(fit.success and np.isfinite(fit.x).all() and np.ptp(logits)>0 and not separated)
    return dict(brier_score=float(np.mean((p-y)**2)),reliability_curve=reliability,ece=float(ece),
        calibration_slope=float(fit.x[0]) if estimable else None,
        calibration_intercept=float(fit.x[1]) if estimable else None,
        calibration_fit_status='estimated' if estimable else 'not_estimable',
        boundary_logit_epsilon=float(np.finfo(float).eps))
