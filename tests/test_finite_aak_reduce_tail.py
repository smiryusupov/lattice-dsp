import numpy as np

import lattice_dsp as ld
from examples.aak_siso_certificate_demo import exact_rational_tail
from examples.finite_aak_noisy_tail_demo import noisy_rational_tail


def test_finite_aak_reduce_tail_selects_exact_rank_three():
    tail, true_poles, _ = exact_rational_tail(95)
    criteria = ld.FiniteNehariCandidateCriteria(
        max_tail_error=1e-8,
        max_rational_error=1e-8,
        max_pole_radius=0.99,
    )

    result = ld.finite_aak_reduce_tail(
        tail,
        ranks=[1, 2, 3, 4],
        rows=48,
        cols=48,
        criteria=criteria,
    )

    assert result["method"] == "finite_section_siso_aak_nehari_reduction_candidate"
    assert result["accepted"] is True
    assert result["selected_rank"] == 3
    assert result["certificate"] is not None
    assert result["certificate"]["schmidt_left_residual"] < 1e-10
    assert result["certificate"]["schmidt_right_residual"] < 1e-10
    np.testing.assert_allclose(
        np.sort(np.real(result["selected"]["poles"])),
        np.sort(true_poles),
        atol=1e-8,
    )


def test_finite_aak_reduce_tail_handles_non_exact_tail_with_stable_candidate():
    tail, _, _, _ = noisy_rational_tail(95)
    criteria = ld.FiniteNehariCandidateCriteria(
        max_tail_error=2e-2,
        max_rational_error=3.5e-2,
        max_pole_radius=0.99,
    )

    result = ld.finite_aak_reduce_tail(
        tail,
        ranks=[1, 2, 3, 4, 5, 6, 8],
        rows=48,
        cols=48,
        criteria=criteria,
    )

    selected = result["selected"]
    assert result["accepted"] is True
    assert result["selected_rank"] >= 3
    assert selected["hankelized_tail_error"] <= criteria.max_tail_error
    assert selected["rational_error"] <= criteria.max_rational_error
    assert selected["max_pole_radius"] <= criteria.max_pole_radius
    assert result["certificate"] is not None
    assert result["certificate"]["rank"] == result["selected_rank"]


def test_finite_aak_reduce_tail_can_skip_certificate():
    tail, _, _, _ = noisy_rational_tail(95)
    result = ld.finite_aak_reduce_tail(
        tail,
        ranks=[3, 4, 5],
        rows=48,
        cols=48,
        attach_certificate=False,
    )
    assert result["certificate"] is None
    assert "selected" in result
