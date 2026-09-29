import numpy as np
import pytest
from proehn.probability import KineticProbabilityProtocol,PlattCalibration


def test_focal_requires_calibration():
    assert KineticProbabilityProtocol().focal_gamma==0
    with pytest.raises(ValueError,match='Uncalibrated'):
        KineticProbabilityProtocol(1.)
    assert KineticProbabilityProtocol(1.,'platt').calibration=='platt'


def test_synthetic_calibration_is_monotone_and_validation_only():
    cal=PlattCalibration.fit([-2,-1,1,2],[0,1,0,1],['v1','v2','v3','v4'],training_patient_ids=['t1'])
    p=cal.transform([-5,0,5])
    assert np.all(np.diff(p)>=0) and np.all((p>=0)&(p<=1))
    with pytest.raises(ValueError,match='outside'):
        PlattCalibration.fit([-1,1],[0,1],['v1','v2'],training_patient_ids=['v1'])
