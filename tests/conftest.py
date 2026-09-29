"""Optional CI guard: scientific tests may not read real cohorts/artifacts."""
import os
from pathlib import Path
import pytest


@pytest.fixture(autouse=True)
def synthetic_only_ci(monkeypatch,tmp_path_factory):
    if os.environ.get('PROEHN_SYNTHETIC_ONLY')!='1':
        return
    import numpy as np
    import pandas as pd
    def forbidden(*args,**kwargs):
        raise AssertionError('Real cohort file readers are forbidden in scientific CI')
    for name in ('read_csv','read_parquet','read_excel','read_feather'):
        monkeypatch.setattr(pd,name,forbidden)
    original=np.load; root=tmp_path_factory.getbasetemp().resolve()
    def synthetic_load(path,*args,**kwargs):
        if not hasattr(path,'read') and root not in Path(path).resolve().parents:
            raise AssertionError('CI may load only artifacts generated in synthetic test directories')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',synthetic_load)
