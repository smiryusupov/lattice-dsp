import numpy as np

import lattice_dsp as ld
from examples import finite_aak_iir_reduction_demo as demo


def test_synthetic_high_order_iir_is_stable_in_reflection_coordinates():
    model = demo.synthetic_high_order_iir()
    assert model["reflection"].shape == (8,)
    assert np.max(np.abs(model["reflection"])) < 1.0
    np.testing.assert_allclose(
        ld.reflection_to_denominator(model["reflection"].tolist()), model["denominator"], atol=1e-10
    )


def test_finite_aak_reduce_iir_selects_stable_lower_order_candidate():
    model = demo.synthetic_high_order_iir()
    criteria = ld.FiniteNehariCandidateCriteria(
        max_tail_error=1.0e-3,
        max_rational_error=5.0e-3,
        max_pole_radius=0.99,
    )
    result = ld.finite_aak_reduce_iir(
        model["reflection"],
        model["numerator"],
        ranks=[2, 3, 4, 5, 6, 8],
        n_impulse=192,
        rows=96,
        cols=96,
        criteria=criteria,
    )

    assert result["accepted"] is True
    assert result["stable"] is True
    assert result["selected_rank"] == 3
    assert len(result["reduced_reflection"]) == result["selected_rank"]
    assert result["relative_impulse_error"] < 5.0e-3
    assert result["selected"]["max_pole_radius"] < 1.0


def test_finite_aak_reduce_iir_rejects_too_short_impulse_window():
    model = demo.synthetic_high_order_iir()
    try:
        ld.finite_aak_reduce_iir(
            model["reflection"],
            model["numerator"],
            ranks=[1, 2],
            n_impulse=10,
            rows=8,
            cols=8,
        )
    except ValueError as exc:
        assert "n_impulse" in str(exc)
    else:
        raise AssertionError("expected ValueError")
