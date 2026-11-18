"""Probability decision thresholds with explicit patient separation."""

from dataclasses import dataclass
import numpy as np
import pandas as pd


def _scores(values):
    values = np.asarray(values, float)
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or np.any((values < 0) | (values > 1))
    ):
        raise ValueError("Expected finite probability scores in [0,1]")
    return values


@dataclass(frozen=True)
class CalibratedThreshold:
    value: float
    criterion: str
    calibration_patient_ids: tuple[str, ...]
    sensitivity: float | None = None
    score_semantics: str = "target_presence"

    def __post_init__(self):
        if (
            not np.isfinite(self.value)
            or not 0 <= self.value <= np.nextafter(1.0, np.inf)
            or not self.score_semantics
        ):
            raise ValueError(
                "Finite probability threshold and score semantics required"
            )
        if self.criterion not in (
            "mcc",
            "sensitivity",
            "accuracy",
            "validation",
            "prespecified",
        ):
            raise ValueError("Unknown threshold provenance criterion")
        if self.criterion != "prespecified" and not self.calibration_patient_ids:
            raise ValueError("Validation patient provenance required")

    def apply(self, scores, *, test_patient_ids):
        if set(map(str, test_patient_ids)) & set(self.calibration_patient_ids):
            raise ValueError("Threshold calibration and test patients overlap")
        values = _scores(scores)
        if (
            len(values) != len(test_patient_ids)
            or pd.isna(np.asarray(test_patient_ids)).any()
        ):
            raise ValueError("Known patient IDs must align with scores")
        return values >= self.value


@dataclass(frozen=True)
class PrespecifiedThresholdProtocol:
    value: float
    schema_id: str
    score_semantics: str
    version: str = "1"

    def __post_init__(self):
        if (
            not self.schema_id
            or not self.version
            or not np.isfinite(self.value)
            or not 0 <= self.value <= 1
        ):
            raise ValueError(
                "Explicit finite prespecified threshold provenance required"
            )

    def apply(self, scores, *, test_patient_ids):
        values = _scores(scores)
        if len(values) != len(test_patient_ids):
            raise ValueError("Threshold patient alignment mismatch")
        return values >= self.value
