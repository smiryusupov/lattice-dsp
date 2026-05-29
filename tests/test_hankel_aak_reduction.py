import numpy as np
import pytest

import lattice_dsp as ld


def test_iir_impulse_response_first_order():
    a = 0.6
    h = np.asarray(ld.iir_impulse_response([1.0, -a], [1.0], 8))
    expected = a ** np.arange(8)
    np.testing.assert_allclose(h, expected, atol=1e-12, rtol=1e-12)


def test_hankel_singular_values_rank_one_exponential():
    a = 0.5
    h = a ** np.arange(80)
    sv = np.asarray(ld.hankel_singular_values(h, 12, 12, offset=1))
    assert sv[0] > 0.0
    assert np.sum(sv[1:] ** 2) / sv[0] ** 2 < 1e-15


def test_hankel_singular_values_match_numpy_svd():
    h = np.array([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625])
    rows = cols = 3
    hankel = np.array([[h[1], h[2], h[3]], [h[2], h[3], h[4]], [h[3], h[4], h[5]]])
    expected = np.linalg.svd(hankel, compute_uv=False)
    got = np.asarray(ld.hankel_singular_values(h, rows, cols, offset=1))
    np.testing.assert_allclose(got, expected, atol=1e-10, rtol=1e-10)


def test_finite_hankel_reduce_recovers_first_order_iir():
    reflection = [-0.55]
    numerator = [1.0, 0.2]
    result = ld.finite_hankel_reduce_iir(
        reflection,
        numerator,
        reduced_order=1,
        n_impulse=80,
        rows=20,
        cols=20,
    )

    assert result["stable"] is True
    assert result["method"] == "finite_hankel_ho_kalman"
    assert result["retained_hankel_energy"] > 1.0 - 1e-12
    assert result["relative_impulse_error"] < 1e-15
    assert len(result["denominator"]) == 2
    assert len(result["numerator"]) == 2
    np.testing.assert_allclose(
        result["denominator"], ld.reflection_to_denominator(reflection), atol=1e-10
    )
    np.testing.assert_allclose(result["numerator"], numerator, atol=1e-10)


def test_finite_hankel_aak_names_are_backward_compatible_aliases():
    reflection = [-0.55]
    numerator = [1.0, 0.2]
    new = ld.finite_hankel_reduce_iir(
        reflection, numerator, reduced_order=1, n_impulse=80, rows=20, cols=20
    )
    with pytest.warns(DeprecationWarning, match="finite_hankel_reduce_iir"):
        old = ld.finite_hankel_aak_reduce_iir(
            reflection, numerator, reduced_order=1, n_impulse=80, rows=20, cols=20
        )

    assert old["method"] == new["method"]
    np.testing.assert_allclose(old["denominator"], new["denominator"], atol=1e-12)
    np.testing.assert_allclose(old["numerator"], new["numerator"], atol=1e-12)


def test_finite_hankel_reduce_order_zero_is_fir_constant():
    h = [2.0, 0.5, 0.25, 0.125, 0.0625]
    result = ld.finite_hankel_reduce_impulse(h, reduced_order=0, rows=2, cols=2)
    assert result["stable"] is True
    assert result["denominator"] == [1.0]
    assert result["numerator"] == [h[0]]
    assert result["relative_impulse_error"] > 0.0


def test_finite_hankel_reduce_impulse_and_iir_are_consistent():
    reflection = [0.4, -0.2, 0.1]
    numerator = [1.0, 0.3, -0.1, 0.05]
    denominator = ld.reflection_to_denominator(reflection)
    impulse = ld.iir_impulse_response(denominator, numerator, 120)

    from_impulse = ld.finite_hankel_reduce_impulse(impulse, reduced_order=2, rows=24, cols=24)
    from_iir = ld.finite_hankel_reduce_iir(
        reflection, numerator, reduced_order=2, n_impulse=120, rows=24, cols=24
    )

    np.testing.assert_allclose(from_impulse["denominator"], from_iir["denominator"], atol=1e-10)
    np.testing.assert_allclose(from_impulse["numerator"], from_iir["numerator"], atol=1e-10)
    assert len(from_iir["denominator"]) == 3
    assert len(from_iir["numerator"]) == 3


def test_finite_hankel_reduce_rejects_bad_inputs():
    with pytest.raises(ValueError, match="non-finite"):
        ld.hankel_singular_values([1.0, np.nan, 0.5], 1, 1)

    with pytest.raises(ValueError, match="rows and cols"):
        ld.hankel_singular_values([1.0, 0.5, 0.25], 0, 2)

    with pytest.raises(ValueError, match="reduced_order"):
        ld.finite_hankel_reduce_impulse([1.0, 0.5, 0.25, 0.125], reduced_order=3, rows=2, cols=2)

    with pytest.raises(ValueError, match=r"rows \+ cols \+ 1"):
        ld.finite_hankel_reduce_impulse([1.0, 0.5, 0.25], reduced_order=1, rows=2, cols=2)

    with pytest.raises(ValueError, match="denominator"):
        ld.iir_impulse_response([], [1.0], 4)

    with pytest.raises(ValueError, match="non-zero"):
        ld.iir_impulse_response([0.0, 0.1], [1.0], 4)
