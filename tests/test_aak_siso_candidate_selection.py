import numpy as np

from examples import aak_siso_candidate_selection as sel
from examples.finite_nehari_rational_bridge import synthetic_anticausal_tail


def test_candidate_selection_finds_first_acceptable_rank():
    rows = cols = 24
    tail = synthetic_anticausal_tail(rows + cols - 1)
    criteria = sel.CandidateCriteria(
        max_tail_error=1e-3, max_rational_error=1e-2, max_pole_radius=0.99
    )

    rows_out = sel.candidate_rows(tail, ranks=[1, 2, 3, 4], rows=rows, cols=cols, criteria=criteria)
    selected = sel.select_candidate(rows_out)

    assert bool(selected["accepted"])
    assert int(selected["rank"]) <= 4
    assert float(selected["hankelized_tail_error"]) <= criteria.max_tail_error
    assert float(selected["rational_error"]) <= criteria.max_rational_error
    assert float(selected["max_pole_radius"]) <= criteria.max_pole_radius


def test_candidate_sigma_next_is_nonincreasing_with_rank():
    rows = cols = 20
    tail = synthetic_anticausal_tail(rows + cols - 1)
    criteria = sel.CandidateCriteria()
    rows_out = sel.candidate_rows(tail, ranks=[1, 2, 3, 4], rows=rows, cols=cols, criteria=criteria)

    sigma = np.asarray([row["sigma_next"] for row in rows_out], dtype=float)
    assert np.all(np.diff(sigma) <= 1e-10)


def test_select_candidate_falls_back_to_lowest_rational_error():
    rows = [
        {"rank": 1, "accepted": False, "rational_error": 0.5},
        {"rank": 2, "accepted": False, "rational_error": 0.2},
        {"rank": 3, "accepted": False, "rational_error": 0.3},
    ]
    selected = sel.select_candidate(rows)
    assert selected["rank"] == 2
