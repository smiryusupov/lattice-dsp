import numpy as np
import pytest

import lattice_dsp


def _toy_identification(samples=1200):
    rng = np.random.default_rng(123)
    x = rng.normal(size=samples)
    target = lattice_dsp.LatticeIIR([0.35, -0.2], [0.2, -0.05, 0.7])
    desired = np.asarray(target.process(x), dtype=float)
    return x, desired


def test_tune_reflection_update_period_returns_recommendation():
    x, desired = _toy_identification()
    result = lattice_dsp.tune_reflection_update_period(
        x,
        desired,
        periods=[1, 2, 4],
        order=2,
        mu_taps=0.05,
        mu_reflection=0.001,
        repeats=1,
        max_tail_mse_ratio=10.0,
        max_worst_tail_mse_ratio=10.0,
    )
    assert result["recommended_period"] in {1, 2, 4}
    assert len(result["results"]) == 3
    assert result["metadata"]["order"] == 2
    assert result["metadata"]["scale_reflection_mu_by_period"] is True


def test_tune_reflection_update_period_supports_trial_matrix():
    x, desired = _toy_identification(samples=1000)
    x2 = np.vstack([x, x * 0.75])
    desired2 = np.vstack([desired, desired * 0.75])
    result = lattice_dsp.tune_reflection_update_period(
        x2,
        desired2,
        periods=[1, 2],
        order=2,
        repeats=1,
        max_tail_mse_ratio=20.0,
        max_worst_tail_mse_ratio=20.0,
    )
    assert result["metadata"]["n_trials"] == 2
    for row in result["results"]:
        assert row["tail_mse_ratio_worst"] >= row["tail_mse_ratio_median"]


def test_tune_reflection_update_period_validates_inputs():
    x = np.ones(10)
    with pytest.raises(ValueError, match="same shape"):
        lattice_dsp.tune_reflection_update_period(x, np.ones(9), periods=[1], order=1)
    with pytest.raises(ValueError, match="positive"):
        lattice_dsp.tune_reflection_update_period(x, x, periods=[0], order=1)
    with pytest.raises(ValueError, match=r"order \+ 1"):
        lattice_dsp.tune_reflection_update_period(x, x, periods=[1], order=2, initial_taps=[0.0])


def test_tune_reflection_update_period_marks_single_trial_as_signal_specific():
    x, desired = _toy_identification(samples=1000)
    result = lattice_dsp.tune_reflection_update_period(
        x,
        desired,
        periods=[1, 2],
        order=2,
        repeats=1,
        max_tail_mse_ratio=20.0,
        max_worst_tail_mse_ratio=20.0,
    )
    assert result["metadata"]["recommendation_scope"] == "single_signal"
    assert result["metadata"]["is_robust_recommendation"] is False
    assert result["metadata"]["min_trials_for_robust"] == 2
    assert result["warnings"]
    assert "signal-specific" in result["warnings"][0]


def test_tune_reflection_update_period_marks_trial_matrix_as_robust():
    x, desired = _toy_identification(samples=1000)
    x2 = np.vstack([x, x * 0.5])
    desired2 = np.vstack([desired, desired * 0.5])
    result = lattice_dsp.tune_reflection_update_period(
        x2,
        desired2,
        periods=[1, 2],
        order=2,
        repeats=1,
        max_tail_mse_ratio=20.0,
        max_worst_tail_mse_ratio=20.0,
    )
    assert result["metadata"]["recommendation_scope"] == "robust"
    assert result["metadata"]["is_robust_recommendation"] is True
    assert result["warnings"] == []


def test_tune_reflection_update_period_validates_min_trials_for_robust():
    x = np.ones(10)
    with pytest.raises(ValueError, match="min_trials_for_robust"):
        lattice_dsp.tune_reflection_update_period(
            x,
            x,
            periods=[1],
            order=1,
            min_trials_for_robust=0,
        )
