import pytest
from proehn.features import FrozenCovariateProtocol


def test_duplicate_burdens_and_unfrozen_schemas_are_rejected():
    with pytest.raises(ValueError,match='Duplicate burden'):
        FrozenCovariateProtocol('toy','v1',('nMut_Primary','TMB_Primary'),(),True).validate()
    with pytest.raises(ValueError,match='Duplicate burden'):
        FrozenCovariateProtocol('toy','v1',(),('CNA_Primary','FGA_Primary'),True).validate()
    with pytest.raises(ValueError,match='PROTOCOL_FREEZE'):
        FrozenCovariateProtocol('toy','v1',(),()).validate()
    FrozenCovariateProtocol('toy','v1',('nMut_Primary',),(),True).validate()
