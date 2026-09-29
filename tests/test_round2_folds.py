import numpy as np
import pytest
from proehn.evaluation import grouped_stratified_outer_folds,GroupedStratificationPolicy,TargetNotEvaluable


def test_grouped_stratification_coverage_and_isolation():
    ids=np.repeat(np.arange(16),2); y=np.repeat(np.arange(16)%2,2)
    policy=GroupedStratificationPolicy(outer_splits=4,inner_splits=2)
    folds=grouped_stratified_outer_folds(ids,y,target='synthetic target',policy=policy)
    for fold in folds:
        fold.validate(ids)
        for partition in (fold.train,fold.validation,fold.test):
            assert set(y[list(partition)])=={0,1}
    assert sorted(i for f in folds for i in f.test)==list(range(32))


def test_rare_target_not_evaluable_no_fold_reduction():
    ids=np.arange(12); y=np.array([1]+[0]*11)
    with pytest.raises(TargetNotEvaluable,match='frozen outer'):
        grouped_stratified_outer_folds(ids,y,target='rare',policy=GroupedStratificationPolicy(outer_splits=4))


def test_labeled_kinetic_minimum_is_checked():
    with pytest.raises(TargetNotEvaluable):
        grouped_stratified_outer_folds(np.arange(16),np.arange(16)%2,target='A',
            policy=GroupedStratificationPolicy(4,2),kinetic_labels=np.full(16,-1))
