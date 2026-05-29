import numpy as np
import pytest

from lattice_dsp import (
    AdaptiveLatticeLadderNLMS,
    bounded_reflection_from_raw,
    denominator_raw_jacobian,
    denominator_raw_jacobian_finite_difference,
    denominator_reflection_jacobian,
    reflection_to_denominator,
)


def finite_difference_reflection_jacobian(reflection, step=1e-6):
    reflection = np.asarray(reflection, dtype=float)
    order = reflection.size
    jac = np.zeros((order, order), dtype=float)
    for j in range(order):
        h = step * max(1.0, abs(reflection[j]))
        plus = reflection.copy()
        minus = reflection.copy()
        plus[j] += h
        minus[j] -= h
        a_plus = np.asarray(reflection_to_denominator(plus.tolist()), dtype=float)[1:]
        a_minus = np.asarray(reflection_to_denominator(minus.tolist()), dtype=float)[1:]
        jac[j] = (a_plus - a_minus) / (2.0 * h)
    return jac


@pytest.mark.parametrize(
    "reflection",
    [
        [0.25],
        [0.35, -0.2],
        [0.35, -0.25, 0.15, -0.08],
        [0.7, -0.4, 0.2, -0.1, 0.05],
    ],
)
def test_denominator_reflection_jacobian_matches_finite_difference(reflection):
    analytic = np.asarray(denominator_reflection_jacobian(reflection), dtype=float)
    finite = finite_difference_reflection_jacobian(reflection)
    np.testing.assert_allclose(analytic, finite, rtol=2e-7, atol=2e-8)


def test_denominator_raw_jacobian_matches_debug_finite_difference():
    raw = [0.15, -0.5, 0.3, -0.1]
    analytic = np.asarray(denominator_raw_jacobian(raw, margin=1e-4), dtype=float)
    finite = np.asarray(
        denominator_raw_jacobian_finite_difference(raw, margin=1e-4, step_scale=1e-6),
        dtype=float,
    )
    np.testing.assert_allclose(analytic, finite, rtol=2e-6, atol=2e-8)


def test_raw_jacobian_includes_tanh_boundary_scaling():
    raw = [0.2, 8.0]
    reflection = bounded_reflection_from_raw(raw, margin=1e-3)
    jac_k = np.asarray(denominator_reflection_jacobian(reflection), dtype=float)
    jac_raw = np.asarray(denominator_raw_jacobian(raw, margin=1e-3), dtype=float)

    # The second raw parameter is close to the tanh boundary; its derivative
    # should be heavily damped compared with the reflection-domain Jacobian.
    assert np.linalg.norm(jac_raw[1]) < 1e-5 * np.linalg.norm(jac_k[1])


def test_adaptive_gradient_mode_can_use_finite_difference_reference():
    adaptive = AdaptiveLatticeLadderNLMS(
        [0.2, -0.1],
        [0.5, 0.0, 0.0],
        mu_taps=0.0,
        mu_reflection=0.0,
        gradient_mode="finite_difference",
    )
    assert adaptive.gradient_mode == "finite_difference"
    adaptive.gradient_mode = "analytic"
    assert adaptive.gradient_mode == "analytic"
    with pytest.raises(ValueError):
        adaptive.gradient_mode = "unknown"
