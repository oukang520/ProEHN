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
