import numpy as np
from proehn.preprocessing import calculate_marginal_rates


def test_unknown_seeding_is_not_negative_and_mutations_are_retained():
    unknown = (4, 1, 0, np.array([[1, 0], [1, 0]]), np.ones((2, 1)))
    rates = np.asarray(calculate_marginal_rates([unknown], 1))
    assert rates[-1] == 0
    assert rates[0] > 0
    known = (1, 1, 0, np.array([[0, 1]]), np.ones((1, 1)))
    assert calculate_marginal_rates([unknown, known], 1)[-1] == calculate_marginal_rates([known], 1)[-1]
