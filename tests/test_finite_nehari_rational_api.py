import numpy as np

import lattice_dsp as ld
from examples.finite_nehari_rational_bridge import synthetic_anticausal_tail


def test_fit_rational_tail_recovers_single_exponential():
    pole = 0.73
    tail = pole ** np.arange(30, dtype=float)

    fit = ld.fit_rational_tail(tail, order=1)
    response = ld.rational_tail_response(fit["denominator"], fit["numerator"], tail.size)

    np.testing.assert_allclose(response, tail, atol=1e-11, rtol=1e-11)
    assert np.max(np.abs(fit["poles"])) < 1.0


def test_finite_nehari_rational_candidates_selects_expected_rank():
    rows = cols = 24
    tail = synthetic_anticausal_tail(rows + cols - 1)
    criteria = ld.FiniteNehariCandidateCriteria(
        max_tail_error=1e-3,
        max_rational_error=1e-2,
        max_pole_radius=0.99,
    )

    candidates = ld.finite_nehari_rational_candidates(
        tail,
        ranks=[1, 2, 3, 4],
        rows=rows,
        cols=cols,
        criteria=criteria,
    )
    selected = ld.select_finite_nehari_candidate(candidates)

    assert bool(selected["accepted"])
    assert int(selected["rank"]) == 3
    assert float(selected["hankelized_tail_error"]) <= criteria.max_tail_error
    assert float(selected["rational_error"]) <= criteria.max_rational_error
    assert float(selected["max_pole_radius"]) <= criteria.max_pole_radius
    assert np.asarray(selected["denominator"]).ndim == 1
    assert np.asarray(selected["numerator"]).ndim == 1
    assert np.asarray(selected["rational_tail"]).shape == tail.shape


def test_select_finite_nehari_candidate_fallback():
    rows = [
        {"rank": 1, "accepted": False, "rational_error": 0.5},
        {"rank": 2, "accepted": False, "rational_error": 0.2},
        {"rank": 3, "accepted": False, "rational_error": 0.3},
    ]
    selected = ld.select_finite_nehari_candidate(rows)
    assert selected["rank"] == 2


def test_known_rank_three_tail_selects_exact_order_and_recovers_poles():
    from examples.finite_nehari_exact_rational_tail import exact_rational_tail, sorted_real_poles

    rows = cols = 32
    tail, true_poles, _weights = exact_rational_tail(rows + cols - 1)
    criteria = ld.FiniteNehariCandidateCriteria(
        max_tail_error=1e-7,
        max_rational_error=1e-7,
        max_pole_radius=0.99,
    )

    candidates = ld.finite_nehari_rational_candidates(
        tail,
        ranks=[1, 2, 3, 4],
        rows=rows,
        cols=cols,
        criteria=criteria,
    )
    selected = ld.select_finite_nehari_candidate(candidates)

    assert int(selected["rank"]) == 3
    assert bool(selected["accepted"])
    assert float(selected["hankelized_tail_error"]) < criteria.max_tail_error
    assert float(selected["rational_error"]) < criteria.max_rational_error
    assert float(selected["max_pole_radius"]) < 1.0
    np.testing.assert_allclose(
        sorted_real_poles(selected["poles"]),
        np.sort(true_poles),
        atol=1e-6,
        rtol=1e-6,
    )


def test_exact_rational_tail_example_candidate_rows_are_well_ordered():
    from examples.finite_nehari_exact_rational_tail import exact_rational_tail

    rows = cols = 28
    tail, _true_poles, _weights = exact_rational_tail(rows + cols - 1)
    criteria = ld.FiniteNehariCandidateCriteria(
        max_tail_error=1e-7,
        max_rational_error=1e-7,
        max_pole_radius=0.99,
    )
    candidates = ld.finite_nehari_rational_candidates(
        tail,
        ranks=[1, 2, 3, 4],
        rows=rows,
        cols=cols,
        criteria=criteria,
    )

    assert not bool(candidates[0]["accepted"])
    assert not bool(candidates[1]["accepted"])
    assert bool(candidates[2]["accepted"])
    assert float(candidates[2]["rational_error"]) < float(candidates[1]["rational_error"])
