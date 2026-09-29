"""Formal target recovery API. No MHN/feature-shift/fixed-gate surrogate."""
from .benchmark import FormalProEHNTrainer, FormalProEHNPredictor, TargetRecoverySpec
from .evaluation import BenchmarkFold, BenchmarkResult, patient_outer_folds, run_oof_benchmark

__all__ = ['FormalProEHNTrainer', 'FormalProEHNPredictor', 'TargetRecoverySpec',
           'BenchmarkFold', 'BenchmarkResult', 'patient_outer_folds', 'run_oof_benchmark']


# Compatibility names for the incoming legacy command. Fail before evaluation.
DEFAULT_PACA_PROEHN_TARGETS = ("P.TP53 (M)", "P.KMT2C (M)", "P.ARID1A (M)")


def run_paca_proehn_metrics(*args, **kwargs):
    """Retired surrogate entry point; never silently report it as ProEHN."""
    raise RuntimeError(
        'The fixed-gate MHN surrogate is not ProEHN. Use FormalProEHNTrainer, '
        'TargetRecoverySpec and run_oof_benchmark with patient-isolated folds.'
    )
