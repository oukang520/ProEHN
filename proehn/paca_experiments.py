"""Retired manuscript-example entry point; use the formal OOF architecture."""
from .legacy.paca_examples import build_standardized_feature_matrix, normalize_rates
from .benchmark import FormalProEHNTrainer, FormalProEHNPredictor, TargetRecoverySpec
from .evaluation import run_oof_benchmark


class PACAExampleExperiments:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            'Legacy PACA in-sample manuscript pipeline is disabled. '
            'Use FormalProEHNTrainer with TargetRecoverySpec and run_oof_benchmark; '
            'raw target provenance and patient folds are required.'
        )
