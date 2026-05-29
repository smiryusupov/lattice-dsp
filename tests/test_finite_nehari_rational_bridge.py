import numpy as np

from examples import finite_nehari_rational_bridge as bridge


def test_rational_tail_fit_recovers_exact_exponential_tail():
    n_terms = 80
    tail = bridge.synthetic_anticausal_tail(n_terms)

    denominator, numerator, poles = bridge.fit_rational_tail(tail, order=4)
    realized = bridge.rational_tail_response(denominator, numerator, n_terms)

    assert np.max(np.abs(poles)) < 1.0
    assert bridge.relative_error(tail, realized) < 1e-10


def test_lower_order_rational_bridge_improves_with_rank():
    rows = cols = 24
    tail = bridge.synthetic_anticausal_tail(rows + cols - 1)

    errors = []
    for rank in [2, 3, 4]:
        denominator, numerator, poles = bridge.fit_rational_tail(tail, order=rank)
        realized = bridge.rational_tail_response(denominator, numerator, tail.size)
        assert np.max(np.abs(poles)) < 1.0
        errors.append(bridge.relative_error(tail, realized))

    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 1e-10
